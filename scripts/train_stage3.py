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

from src.data_config import INDEX_PATH, PROCESS_DIR, get_label3_classes, load_registry_dataframe
from src.dataset import ISCXStage3Dataset
from src.models import VPNClassifier
from src.utils import generate_markdown_report, plot_metrics, save_confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Stage 3 experts for label3 application classification."
    )
    parser.add_argument("--label1", help="Restrict to one label1 branch, such as NonVPN or VPN.")
    parser.add_argument("--label2", help="Restrict to one label2 branch, such as Chat.")
    parser.add_argument(
        "--min-class-count",
        type=int,
        default=5,
        help="Minimum sample count required for a label3 class to be kept.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs per expert.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    parser.add_argument(
        "--report-root",
        default=os.path.join(PROJECT_ROOT, "report", "stage3"),
        help="Directory for stage3 reports.",
    )
    return parser.parse_args()


def slugify(value):
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def select_stage3_tasks(df, label1=None, label2=None, min_class_count=5):
    selected = df.copy()
    if label1:
        selected = selected[selected["label1"] == label1]
    if label2:
        selected = selected[selected["label2"] == label2]

    tasks = []
    for (curr_label1, curr_label2), group_df in selected.groupby(["label1", "label2"]):
        label3_classes = get_label3_classes(group_df, min_count=min_class_count)
        if len(label3_classes) < 2:
            continue

        group_df = group_df[group_df["label3"].isin(label3_classes)].copy()
        if len(group_df) < len(label3_classes) * 2:
            continue

        tasks.append(
            {
                "label1": curr_label1,
                "label2": curr_label2,
                "df": group_df,
                "label3_classes": label3_classes,
            }
        )

    return tasks


def train_single_expert(task, args, device):
    label1 = task["label1"]
    label2 = task["label2"]
    df = task["df"]
    label3_classes = task["label3_classes"]

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label3"],
        random_state=42,
    )

    train_set = ISCXStage3Dataset(
        train_df,
        PROCESS_DIR,
        seq_len=64,
        label3_classes=label3_classes,
    )
    val_set = ISCXStage3Dataset(
        val_df,
        PROCESS_DIR,
        seq_len=64,
        label3_classes=label3_classes,
    )

    num_workers = min(8, os.cpu_count() or 1)
    use_pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    model = VPNClassifier(num_classes=len(label3_classes), seq_len=train_set.seq_len).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    history = {"train_loss": [], "val_loss": [], "acc": []}
    y_true, y_pred = [], []

    print(
        f"\n[Stage3] {label1} / {label2} | "
        f"{len(df)} samples | classes={label3_classes}"
    )

    for epoch in range(args.epochs):
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
        val_loss = 0.0
        y_true, y_pred = [], []
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
                f"Epoch {epoch + 1:02d}/{args.epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | Val Acc: {accuracy:.4f}"
            )

    group_slug = f"{slugify(label1)}__{slugify(label2)}"
    report_dir = os.path.join(args.report_root, group_slug)
    os.makedirs(report_dir, exist_ok=True)

    plot_metrics(history, report_dir)
    save_confusion_matrix(
        y_true,
        y_pred,
        report_dir,
        labels=label3_classes,
        title=f"Stage 3: {label1} / {label2} App Classification",
    )
    generate_markdown_report(
        history,
        y_true,
        y_pred,
        report_dir,
        target_names=label3_classes,
        stage_name=f"Stage 3 - {label1} / {label2} App Classification",
    )

    model_path = os.path.join(report_dir, "stage3_expert.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model weights to: {model_path}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading sample index: {INDEX_PATH}")
    df_all, _ = load_registry_dataframe(INDEX_PATH)

    tasks = select_stage3_tasks(
        df_all,
        label1=args.label1,
        label2=args.label2,
        min_class_count=args.min_class_count,
    )
    if not tasks:
        raise ValueError("No valid stage3 expert tasks found for the current filters.")

    print(f"Training Stage 3 experts on {device}. Task count: {len(tasks)}")
    for task in tasks:
        train_single_expert(task, args, device)


if __name__ == "__main__":
    main()
