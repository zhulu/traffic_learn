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

from src.data_config import INDEX_PATH, PROCESS_DIR, load_registry_dataframe
from src.dataset import ISCXStage1Dataset
from src.models import VPNClassifier
from src.utils import generate_markdown_report, plot_metrics, save_confusion_matrix


def run_experiment():
    report_dir = os.path.join(PROJECT_ROOT, "report")
    os.makedirs(report_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_pin_memory = device.type == "cuda"
    batch_size = 512
    epochs = 50
    lr = 0.001
    num_workers = min(8, os.cpu_count() or 1)

    print(f"Loading sample index: {INDEX_PATH}")
    df, _ = load_registry_dataframe(INDEX_PATH)

    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["label1"], random_state=42
    )

    train_set = ISCXStage1Dataset(train_df, PROCESS_DIR)
    val_set = ISCXStage1Dataset(val_df, PROCESS_DIR)

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

    num_vpn = (train_df["label1"] == "VPN").sum()
    num_nonvpn = (train_df["label1"] == "NonVPN").sum()
    weight_ratio = num_nonvpn / num_vpn
    pos_weight = torch.tensor([weight_ratio], dtype=torch.float32, device=device)

    print(f"Class balancing enabled: VPN positive weight = {weight_ratio:.2f}")

    model = VPNClassifier(num_classes=1, seq_len=train_set.seq_len).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "acc": []}

    print(f"Training Stage 1 on {device} with {len(df)} samples")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for x_seq, x_stats, y in train_loader:
            x_seq = x_seq.to(device)
            x_stats = x_stats.to(device)
            y = y.to(device).unsqueeze(1)

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
                y = y.to(device).unsqueeze(1)

                out = model(x_seq, x_stats)
                val_loss += criterion(out, y).item()
                preds = (torch.sigmoid(out) > 0.5).float()
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
    save_confusion_matrix(y_true, y_pred, report_dir)
    generate_markdown_report(
        history,
        y_true,
        y_pred,
        report_dir,
        target_names=["NonVPN", "VPN"],
        stage_name="Stage 1 - VPN Detection",
    )

    model_path = os.path.join(report_dir, "stage1_vpn_detector.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model weights to: {model_path}")


if __name__ == "__main__":
    run_experiment()
