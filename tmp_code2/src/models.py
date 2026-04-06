import torch
import torch.nn as nn

class VPNClassifier(nn.Module):
    # 【改动重点】：input_dim 默认值从 3 改为 2
    def __init__(self, seq_len=32, input_dim=2, d_model=64, nhead=4):
        super(VPNClassifier, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, d_model, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        cnn_feat = self.cnn(x).squeeze(-1)
        
        x_t = x.permute(0, 2, 1) # 形状转换适应 Transformer
        x_t = self.embedding(x_t) + self.pos_encoder
        trans_feat = self.transformer(x_t)
        trans_feat = torch.mean(trans_feat, dim=1)
        
        return self.fc(torch.cat([cnn_feat, trans_feat], dim=1))