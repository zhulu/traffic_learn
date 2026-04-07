import argparse
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

from src.data_config import (
    INDEX_PATH,
    LABEL1_LABELS,
    PROCESS_DIR,
    get_label2_classes,
    load_registry_dataframe,
)
from src.dataset import ISCXStage2Dataset
from src.models import VPNClassifier
from src.utils import generate_markdown_report, plot_metrics, save_confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Stage 2 business classifier for a selected label1 branch."
    )
    parser.add_argument(
        "--label1",
        choices=LABEL1_LABELS,
        default="NonVPN",
        help="Train the Stage 2 expert for VPN or NonVPN.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size.")
    return parser.parse_args()


def run_stage2(label1="NonVPN", epochs=50, batch_size=512):
    label1_slug = str(label1).strip().lower()
    report_dir = os.path.join(PROJECT_ROOT, "report", f"stage2_{label1_slug}")
    os.makedirs(report_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_pin_memory = device.type == "cuda"
    lr = 0.001
    num_workers = min(8, os.cpu_count() or 1)

    print(f"Loading sample index: {INDEX_PATH}")
    df_all, _ = load_registry_dataframe(INDEX_PATH)

    df = df_all[df_all["label1"] == label1].copy()
    label2_classes = get_label2_classes(df)
    if len(label2_classes) < 2:
        raise ValueError(
            f"Stage 2 requires at least 2 label2 classes under {label1}, got {label2_classes}"
        )

    print(
        f"Selected {len(df)} {label1} samples for business classification across "
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

    print(f"Training Stage 2 ({label1}) on {device}")
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
        title=f"Stage 2: {label1} Business Classification",
    )
    generate_markdown_report(
        history,
        y_true,
        y_pred,
        report_dir,
        target_names=label2_classes,
        stage_name=f"Stage 2 - {label1} Business Classification",
    )

    model_path = os.path.join(report_dir, f"stage2_{label1_slug}_expert.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model weights to: {model_path}")


if __name__ == "__main__":
    args = parse_args()
    run_stage2(label1=args.label1, epochs=args.epochs, batch_size=args.batch_size)
