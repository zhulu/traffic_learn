import sys
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# --- 自动处理项目路径 ---
# 获取当前脚本所在目录的上一级，即项目根目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.append(PROJECT_ROOT)

from src.dataset import ISCXStage1Dataset
from src.models import VPNClassifier
from src.utils import plot_metrics, save_confusion_matrix, generate_markdown_report

def run_experiment():
    # --- 1. 动态路径配置 ---
    # 根据你的文件组织结构进行映射
    REPORT_DIR = os.path.join(PROJECT_ROOT, "report")
    PROCESS_DIR = os.path.join(PROJECT_ROOT, "data", "process")
    INDEX_PATH = os.path.join(PROJECT_ROOT, "samples.npz")
    
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    # --- 2. 硬件与超参数 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 512  # 充分利用 3090 的 24G 显存
    EPOCHS = 50
    LR = 0.001
    
    # --- 3. 数据加载逻辑 ---
    print(f"📂 正在读取索引文件: {INDEX_PATH}")
    index_data = np.load(INDEX_PATH, allow_pickle=True)
    df = pd.DataFrame(index_data['data'], columns=index_data['columns'])
    df['row'] = df['row'].astype(int)
    
    # 分层划分训练集与验证集 (80/20)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label1'], random_state=42)
    
    train_set = ISCXStage1Dataset(train_df, PROCESS_DIR)
    val_set = ISCXStage1Dataset(val_df, PROCESS_DIR)
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- 4. 动态计算类别权重与初始化模型 ---
    # 统计训练集中的样本数量
    num_vpn = (train_df['label1'] == 'VPN').sum()
    num_nonvpn = (train_df['label1'] == 'NonVPN').sum()
    
    # 计算正样本(VPN)的补偿权重
    # 如果 NonVPN 有 1375 个，VPN 有 402 个，weight 大约为 3.42
    weight_ratio = num_nonvpn / num_vpn
    pos_weight = torch.tensor([weight_ratio]).float().to(DEVICE)
    
    print(f"⚖️ 类别失衡补偿: VPN 样本权重调整为 {weight_ratio:.2f} 倍")

    model = VPNClassifier().to(DEVICE)
    
    # 引入代价敏感学习：错误识别 VPN 将面临更大的 Loss 惩罚
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {'train_loss': [], 'val_loss': [], 'acc': []}

    # --- 5. 训练主循环 ---
    print(f"🔥 训练开始 | 设备: {DEVICE} | 样本总数: {len(df)}")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        y_true, y_pred, val_loss = [], [], 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(1)
                out = model(x)
                val_loss += criterion(out, y).item()
                # 概率阈值判定
                preds = (torch.sigmoid(out) > 0.5).float()
                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        
        # 计算当前 epoch 指标
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        accuracy = (np.array(y_true) == np.array(y_pred)).mean()
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['acc'].append(accuracy)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Acc: {accuracy:.4f}")

    # --- 6. 生成实验报告与结果落盘 ---
    print("\n" + "—"*30)
    plot_metrics(history, REPORT_DIR)
    save_confusion_matrix(y_true, y_pred, REPORT_DIR)
    generate_markdown_report(history, y_true, y_pred, REPORT_DIR)
    
    # 保存权重以便第二阶段调用
    model_path = os.path.join(REPORT_DIR, "stage1_vpn_detector.pth")
    torch.save(model.state_dict(), model_path)
    print(f"💾 模型权重已保存: {model_path}")

if __name__ == "__main__":
    run_experiment()