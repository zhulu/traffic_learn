import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import math

# ==========================================
# 1. 数据集定义 (Dataset)
# ==========================================
class ISCXStage1Dataset(Dataset):
    def __init__(self, dataframe, process_dir):
        self.df = dataframe
        self.process_dir = process_dir
        self.l1_map = {'NonVPN': 0, 'VPN': 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = np.load(os.path.join(self.process_dir, row['file']))
        feature = data['features'][row['row']]  # (32, 3)
        
        # 归一化建议：长度除以1500，IAT取对数 [cite: 87, 89]
        feature[:, 0] = feature[:, 0] / 1500.0
        feature[:, 1] = np.log1p(feature[:, 1])
        
        x = torch.from_numpy(feature).float().permute(1, 0) # (3, 32)
        y = torch.tensor(self.l1_map[row['label1']], dtype=torch.float32)
        return x, y

# ==========================================
# 2. 模型架构 (1D-CNN + Transformer)
# ==========================================
class VPNClassifier(nn.Module):
    def __init__(self, seq_len=32, input_dim=3, d_model=64, nhead=4):
        super(VPNClassifier, self).__init__()
        
        # Branch 1: 1D-CNN (局部特征提取)
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, d_model, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Branch 2: Transformer (全局时序提取)
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Fusion & Head
        self.fc = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1) # 二分类输出 [cite: 135]
        )

    def forward(self, x):
        # x shape: (B, 3, 32)
        
        # CNN path
        cnn_feat = self.cnn(x).squeeze(-1) # (B, d_model)
        
        # Transformer path
        x_t = x.permute(0, 2, 1) # (B, 32, 3)
        x_t = self.embedding(x_t) + self.pos_encoder
        trans_feat = self.transformer(x_t)
        trans_feat = torch.mean(trans_feat, dim=1) # (B, d_model)
        
        # Concat
        combined = torch.cat([cnn_feat, trans_feat], dim=1)
        return self.fc(combined)

# ==========================================
# 3. 训练与评估逻辑
# ==========================================
def train_model():
    # 参数配置
    PROCESS_DIR = "data/process"
    INDEX_PATH = "sample.npz"
    BATCH_SIZE = 512
    EPOCHS = 30
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载索引并划分数据
    index_data = np.load(INDEX_PATH, allow_pickle=True)
    df = pd.DataFrame(index_data['data'], columns=index_data['columns'])
    df['row'] = df['row'].astype(int)
    
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label1'], random_state=42)
    
    train_loader = DataLoader(ISCXStage1Dataset(train_df, PROCESS_DIR), batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    test_loader = DataLoader(ISCXStage1Dataset(test_df, PROCESS_DIR), batch_size=BATCH_SIZE, shuffle=False)
    
    model = VPNClassifier().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    history = {'loss': [], 'acc': [], 'f1': [], 'val_loss': []}

    print(f"🚀 开始训练 Scenario A - 第一阶段 (VPN vs Non-VPN)")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE).view(-1, 1)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # 验证
        model.eval()
        y_true, y_pred = [], []
        val_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE).view(-1, 1)
                outputs = model(x)
                val_loss += criterion(outputs, y).item()
                preds = torch.sigmoid(outputs) > 0.5
                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        
        # 计算指标 [cite: 572, 573, 575]
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        avg_loss = total_loss/len(train_loader)
        
        history['loss'].append(avg_loss)
        history['acc'].append(acc)
        history['f1'].append(f1)
        history['val_loss'].append(val_loss/len(test_loader))
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

    # 生成实验报告
    generate_report(history, y_true, y_pred)

def generate_report(history, y_true, y_pred):
    # 1. 损失与精度曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Curve')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['acc'], label='Accuracy')
    plt.plot(history['f1'], label='F1-Score')
    plt.title('Metrics Curve')
    plt.legend()
    plt.savefig('learning_curves.png')
    
    # 2. 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['NonVPN', 'VPN'], yticklabels=['NonVPN', 'VPN'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Stage 1')
    plt.savefig('confusion_matrix.png')
    
    # 3. 详细报告 [cite: 572]
    print("\n" + "="*20 + " 最终实验报告 " + "="*20)
    print(classification_report(y_true, y_pred, target_names=['NonVPN', 'VPN']))

if __name__ == "__main__":
    train_model()