import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data_config import INDEX_PATH, PROCESS_DIR, get_label2_classes, load_registry_dataframe
from src.dataset import ISCXStage2Dataset
from src.models import VPNClassifier
from src.utils import generate_markdown_report, plot_metrics, save_confusion_matrix


def run_stage2_nonvpn():
    report_dir = os.path.join(PROJECT_ROOT, "report", "stage2_nonvpn")
    os.makedirs(report_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_pin_memory = device.type == "cuda"
    batch_size = 512
    epochs = 50
    lr = 0.001
    num_workers = min(8, os.cpu_count() or 1)

    print(f"Loading sample index: {INDEX_PATH}")
    df_all, _ = load_registry_dataframe(INDEX_PATH)

    df = df_all[df_all["label1"] == "NonVPN"].copy()
    label2_classes = get_label2_classes(df)
    print(
        f"Selected {len(df)} NonVPN samples for app classification across "
        f"{len(label2_classes)} classes: {label2_classes}"
    )

    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["label2"], random_state=42
    )

    train_set = ISCXStage2Dataset(
        train_df, PROCESS_DIR, seq_len=64, label2_classes=label2_classes
    )
    val_set = ISCXStage2Dataset(
        val_df, PROCESS_DIR, seq_len=64, label2_classes=label2_classes
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    model = VPNClassifier(num_classes=len(label2_classes), seq_len=train_set.seq_len).to(
        device
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "acc": []}

    print(f"Training Stage 2 on {device}")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x_seq, x_stats, y in train_loader:
            x_seq = x_seq.to(device)
            x_stats = x_stats.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            out = model(x_seq, x_stats)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        y_true, y_pred, val_loss = [], [], 0.0
        with torch.no_grad():
            for x_seq, x_stats, y in val_loader:
                x_seq = x_seq.to(device)
                x_stats = x_stats.to(device)
                y = y.to(device)

                out = model(x_seq, x_stats)
                val_loss += criterion(out, y).item()
                _, preds = torch.max(out, 1)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        accuracy = (np.array(y_true) == np.array(y_pred)).mean()

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["acc"].append(accuracy)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:02d}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | Val Acc: {accuracy:.4f}"
            )

    plot_metrics(history, report_dir)
    save_confusion_matrix(
        y_true,
        y_pred,
        report_dir,
        labels=label2_classes,
        title="Stage 2: Non-VPN App Classification",
    )
    generate_markdown_report(
        history,
        y_true,
        y_pred,
        report_dir,
        target_names=label2_classes,
        stage_name="Stage 2 - Non-VPN App Classification",
    )

    model_path = os.path.join(report_dir, "stage2_nonvpn_expert.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model weights to: {model_path}")


if __name__ == "__main__":
    run_stage2_nonvpn()
