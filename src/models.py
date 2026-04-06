import torch
import torch.nn as nn


class VPNClassifier(nn.Module):
    """
    Shared traffic classifier for Stage 1 and Stage 2.
    """

    def __init__(
        self,
        num_classes=5,
        seq_len=64,
        input_dim=2,
        stats_dim=10,
        d_model=64,
        nhead=4,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, d_model, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(1),
        )

        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=256,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        fc_input_dim = d_model * 2 + stats_dim
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x_seq, x_stats):
        cnn_feat = self.cnn(x_seq).squeeze(-1)

        x_t = x_seq.permute(0, 2, 1)
        x_t = self.embedding(x_t)

        seq_len = x_t.size(1)
        max_len = self.pos_encoder.size(1)
        if seq_len > max_len:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds positional encoding length {max_len}"
            )
        x_t = x_t + self.pos_encoder[:, :seq_len, :]

        trans_feat = self.transformer(x_t)
        trans_feat = torch.mean(trans_feat, dim=1)

        combined = torch.cat([cnn_feat, trans_feat, x_stats], dim=1)
        return self.fc(combined)
