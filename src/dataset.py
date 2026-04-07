import os

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data_config import (
    LABEL1_LABELS,
    build_label_map,
    get_label2_classes,
    get_label3_classes,
)


class ISCXStage1Dataset(Dataset):
    def __init__(self, dataframe, process_dir, seq_len=32):
        self.df = dataframe.reset_index(drop=True).copy()
        self.process_dir = os.path.abspath(process_dir)
        self.seq_len = seq_len
        self.l1_map = build_label_map(LABEL1_LABELS)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = np.load(os.path.join(self.process_dir, row["file"]))

        feature = data["features"][int(row["row"])][: self.seq_len]
        lengths = feature[:, 0] / 1500.0
        iats = np.log1p(feature[:, 1])
        directions = feature[:, 2]

        signed_lengths = lengths * (2 * directions - 1)
        x_seq = np.stack([signed_lengths, iats], axis=0)
        x_seq = torch.from_numpy(x_seq).float()

        raw_stats = data["stats"][int(row["row"])]
        x_stats = np.log1p(np.abs(raw_stats))
        x_stats = torch.from_numpy(x_stats).float()

        y = torch.tensor(self.l1_map[str(row["label1"])], dtype=torch.float32)
        return x_seq, x_stats, y


class ISCXStage2Dataset(Dataset):
    def __init__(self, dataframe, process_dir, seq_len=64, label2_classes=None):
        self.df = dataframe.reset_index(drop=True).copy()
        self.process_dir = os.path.abspath(process_dir)
        self.seq_len = seq_len
        self.label2_classes = label2_classes or get_label2_classes(self.df)
        self.l2_map = build_label_map(self.label2_classes)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = np.load(os.path.join(self.process_dir, row["file"]))

        feature = data["features"][int(row["row"])][: self.seq_len]
        lengths = feature[:, 0] / 1500.0
        iats = np.log1p(feature[:, 1])
        directions = feature[:, 2]

        signed_lengths = lengths * (2 * directions - 1)
        x_seq = np.stack([signed_lengths, iats], axis=0)
        x_seq = torch.from_numpy(x_seq).float()

        raw_stats = data["stats"][int(row["row"])]
        x_stats = np.log1p(np.abs(raw_stats))
        x_stats = torch.from_numpy(x_stats).float()

        label2 = str(row["label2"])
        if label2 not in self.l2_map:
            raise KeyError(
                f"Unknown label2 '{label2}'. Available classes: {self.label2_classes}"
            )

        y = torch.tensor(self.l2_map[label2], dtype=torch.long)
        return x_seq, x_stats, y


class ISCXStage3Dataset(Dataset):
    def __init__(self, dataframe, process_dir, seq_len=64, label3_classes=None):
        self.df = dataframe.reset_index(drop=True).copy()
        self.process_dir = os.path.abspath(process_dir)
        self.seq_len = seq_len
        self.label3_classes = label3_classes or get_label3_classes(self.df)
        self.l3_map = build_label_map(self.label3_classes)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = np.load(os.path.join(self.process_dir, row["file"]))

        feature = data["features"][int(row["row"])][: self.seq_len]
        lengths = feature[:, 0] / 1500.0
        iats = np.log1p(feature[:, 1])
        directions = feature[:, 2]

        signed_lengths = lengths * (2 * directions - 1)
        x_seq = np.stack([signed_lengths, iats], axis=0)
        x_seq = torch.from_numpy(x_seq).float()

        raw_stats = data["stats"][int(row["row"])]
        x_stats = np.log1p(np.abs(raw_stats))
        x_stats = torch.from_numpy(x_stats).float()

        label3 = str(row["label3"])
        if label3 not in self.l3_map:
            raise KeyError(
                f"Unknown label3 '{label3}'. Available classes: {self.label3_classes}"
            )

        y = torch.tensor(self.l3_map[label3], dtype=torch.long)
        return x_seq, x_stats, y
