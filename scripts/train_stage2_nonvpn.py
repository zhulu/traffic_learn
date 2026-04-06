import sys
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# 自动对齐项目根目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.append(PROJECT_ROOT)

from src.dataset import ISCXStage2Dataset
from src.models import VPNClassifier
from src.utils import plot_metrics, save_confusion_matrix, generate_markdown_report, STAGE2_LABELS

def run_stage2_nonvpn():
    # --- 1. 配置 ---
    REPORT_DIR = os.path.join(PROJECT_ROOT, "report", "stage2_nonvpn")
    PROCESS_DIR = os.path.join(PROJECT_ROOT, "data", "process")
    
    INDEX_PATH = os.path.join(PROJECT_ROOT, "samples_v2.npz")
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 512
    EPOCHS = 50 
    SEQ_LEN = 64 # [修改]
    LR = 0.001
    
    # --- 2. 数据准备：筛选出 Non-VPN 流量 ---
    print(f"📂 正在读取索引文件: {INDEX_PATH}")
    index_data = np.load(INDEX_PATH, allow_pickle=True)
    df_all = pd.DataFrame(index_data['data'], columns=index_data['columns'])
    
    # 仅保留 NonVPN 类别
    df = df_all[df_all['label1'] == 'NonVPN'].copy()
    df['row'] = df['row'].astype(int)
    
    print(f"📦 筛选完成：共 {len(df)} 个 Non-VPN 样本进行 5 分类训练")

    # 分层抽样确保 5 个类别的比例一致
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label2'], random_state=42)
    
    train_loader = DataLoader(ISCXStage2Dataset(train_df, PROCESS_DIR), 
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(ISCXStage2Dataset(val_df, PROCESS_DIR), 
                             batch_size=BATCH_SIZE, shuffle=False)

    # --- 3. 模型初始化 (7分类) ---
    model = VPNClassifier(num_classes=5).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {'train_loss': [], 'val_loss': [], 'acc': []}

    # --- 4. 训练循环 ---
    print(f"🔥 Stage 2 (Non-VPN) 训练启动 | 设备: {DEVICE}")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for x_seq, x_stats, y in train_loader:
            x_seq, x_stats, y = x_seq.to(DEVICE), x_stats.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            out = model(x_seq, x_stats)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 验证逻辑
        model.eval()
        y_true, y_pred, val_loss = [], [], 0
        with torch.no_grad():
            for x_seq, x_stats, y in val_loader:
                x_seq, x_stats, y = x_seq.to(DEVICE), x_stats.to(DEVICE), y.to(DEVICE)
                out = model(x_seq, x_stats)
                val_loss += criterion(out, y).item()
                
                _, preds = torch.max(out, 1)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        
        # 计算指标
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        accuracy = (np.array(y_true) == np.array(y_pred)).mean()
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['acc'].append(accuracy)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{EPOCHS} | Val Acc: {accuracy:.4f} | Loss: {avg_train_loss:.4f}")

    # --- 5. 生成报告 ---
    print("\n" + "—"*30)
    plot_metrics(history, REPORT_DIR)
    
    save_confusion_matrix(y_true, y_pred, REPORT_DIR, 
                          labels=STAGE2_LABELS, 
                          title='Stage 2: Non-VPN App Classification')
    
    generate_markdown_report(history, y_true, y_pred, REPORT_DIR, 
                             target_names=STAGE2_LABELS, 
                             stage_name="Stage 2 - Non-VPN Expert")
    
    model_path = os.path.join(REPORT_DIR, "stage2_nonvpn_expert.pth")
    torch.save(model.state_dict(), model_path)
    print(f"💾 Non-VPN 专家权重已保存: {model_path}")

if __name__ == "__main__":
    run_stage2_nonvpn()