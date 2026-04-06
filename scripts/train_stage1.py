import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from src.dataset import ISCXStage1Dataset
from src.models import VPNClassifier
from src.utils import generate_markdown_report, plot_metrics, save_confusion_matrix


def run_experiment():
    report_dir = os.path.join(PROJECT_ROOT, "report")
    process_dir = os.path.join(PROJECT_ROOT, "data", "process")
    index_path = os.path.join(PROJECT_ROOT, "samples.npz")

    os.makedirs(report_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_pin_memory = device.type == "cuda"
    batch_size = 512
    epochs = 50
    lr = 0.001

    print(f"📂 正在读取索引文件: {index_path}")
    index_data = np.load(index_path, allow_pickle=True)
    df = pd.DataFrame(index_data["data"], columns=index_data["columns"])
    df["row"] = df["row"].astype(int)

    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["label1"], random_state=42
    )

    train_set = ISCXStage1Dataset(train_df, process_dir)
    val_set = ISCXStage1Dataset(val_df, process_dir)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=use_pin_memory,
    )

    num_vpn = (train_df["label1"] == "VPN").sum()
    num_nonvpn = (train_df["label1"] == "NonVPN").sum()
    weight_ratio = num_nonvpn / num_vpn
    pos_weight = torch.tensor([weight_ratio], dtype=torch.float32, device=device)

    print(f"⚖️ 类别失衡补偿: VPN 样本权重调整为 {weight_ratio:.2f} 倍")

    model = VPNClassifier(num_classes=1, seq_len=train_set.seq_len).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "acc": []}

    print(f"🔥 训练开始 | 设备: {device} | 样本总数: {len(df)}")
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
    generate_markdown_report(history, y_true, y_pred, report_dir)

    model_path = os.path.join(report_dir, "stage1_vpn_detector.pth")
    torch.save(model.state_dict(), model_path)
    print(f"💾 模型权重已保存: {model_path}")


if __name__ == "__main__":
    run_experiment()
