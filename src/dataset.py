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
        
        # 1. 微观序列特征 (根据当前阶段的 seq_len 进行裁剪)
        # 如果预处理生成了 64 个包，这里会截取前 32 个给 Stage 1 使用
        feature = data['features'][row['row']][:self.seq_len] 
        
        lengths = feature[:, 0] / 1500.0
        iats = np.log1p(feature[:, 1])
        directions = feature[:, 2]
        
        # 融合有符号长度：正值代表前向，负值代表后向
        signed_lengths = lengths * (2 * directions - 1)
        x_seq = np.stack([signed_lengths, iats], axis=0)
        x_seq = torch.from_numpy(x_seq).float()
        
        # 2. 宏观统计特征
        raw_stats = data['stats'][row['row']]
        x_stats = np.log1p(np.abs(raw_stats)) 
        x_stats = torch.from_numpy(x_stats).float()
        
        # Stage 1 使用 BCEWithLogitsLoss，y 需要是 float
        y = torch.tensor(self.l1_map[row['label1']], dtype=torch.float32)
        
        return x_seq, x_stats, y

class ISCXStage2Dataset(Dataset):
    def __init__(self, dataframe, process_dir, seq_len=64):
        self.df = dataframe
        self.process_dir = process_dir
        self.seq_len = seq_len # [修复]：之前漏掉了这一行
        
        # 必须与 STAGE2_LABELS 顺序完全一致
        self.l2_map = {
            'File_Transfer': 0, 
            'Streaming': 1, 
            'VoIP': 2, 
            'Email': 3, 
            'Chat': 4
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = np.load(os.path.join(self.process_dir, row['file']))
        
        # 1. 序列特征 (这里会完整截取 64 个包)
        feature = data['features'][row['row']][:self.seq_len] 
        
        lengths = feature[:, 0] / 1500.0
        iats = np.log1p(feature[:, 1])
        directions = feature[:, 2]
        
        signed_lengths = lengths * (2 * directions - 1)
        x_seq = np.stack([signed_lengths, iats], axis=0)
        x_seq = torch.from_numpy(x_seq).float()
        
        # 2. 宏观统计特征
        raw_stats = data['stats'][row['row']]
        x_stats = np.log1p(np.abs(raw_stats)) 
        x_stats = torch.from_numpy(x_stats).float()
        
        # Stage 2 使用 CrossEntropyLoss，y 必须是 long 类型 (即整数类标)
        y = torch.tensor(self.l2_map[row['label2']], dtype=torch.long)
        
        return x_seq, x_stats, y