import os
import numpy as np
import torch
from torch.utils.data import Dataset

class ISCXStage1Dataset(Dataset):
    def __init__(self, dataframe, process_dir, seq_len=32):
        self.df = dataframe
        self.process_dir = process_dir
        self.seq_len = seq_len
        self.l1_map = {'NonVPN': 0, 'VPN': 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = np.load(os.path.join(self.process_dir, row['file']))
        feature = data['features'][row['row']]  # 原始形状: (32, 3)
        
        # 1. 提取独立特征
        lengths = feature[:, 0] / 1500.0        # 归一化长度
        iats = np.log1p(feature[:, 1])          # 时间对数化
        directions = feature[:, 2]              # 1为Forward, 0为Backward
        
        # 2. 构造“有符号长度” (Signed Length)
        # 算法: direction为1时乘1，为0时乘-1
        signed_lengths = lengths * (2 * directions - 1)
        
        # 3. 组合新特征矩阵: 维度从 3 降为 2
        # 新形状: (2, 32)
        new_feature = np.stack([signed_lengths, iats], axis=0) 
        
        x = torch.from_numpy(new_feature).float()
        y = torch.tensor(self.l1_map[row['label1']], dtype=torch.float32)
        
        return x, y