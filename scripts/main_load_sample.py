import os
import numpy as np
import argparse
import json

def load_sample_feature(process_dir, npz_name, row_idx):
    """
    核心接口：加载指定文件的指定行特征
    """
    # 如果 npz_name 已经包含路径，直接使用；否则拼接 process_dir
    if os.path.sep in npz_name:
        path = npz_name
    else:
        path = os.path.join(process_dir, npz_name)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ 未找到特征文件: {path}")
    
    data = np.load(path)
    features = data['features']
    
    if row_idx >= len(features):
        raise IndexError(f"❌ 行索引 {row_idx} 超出文件范围 (总行数: {len(features)})")
    
    return features[row_idx]

def main():
    parser = argparse.ArgumentParser(description="🚀 科研导师工具：特征采样检查器")
    
    # 必须参数
    parser.add_argument("--file", type=str, required=True, help="NPZ文件名 (例如: aim_chat_3a.npz)")
    parser.add_argument("--row", type=int, required=True, help="流样本的行索引 (Row Index)")
    
    # 可选参数
    parser.add_argument("--dir", type=str, default="data/process", help="特征存储目录 (默认: data/process)")
    parser.add_argument("--detail", action="store_true", help="是否打印完整的 32x3 矩阵")

    args = parser.parse_args()

    print("="*50)
    print(f"🔍 正在检索样本: {args.file} | Row: {args.row}")
    print("="*50)

    try:
        # 调用接口
        feature_matrix = load_sample_feature(args.dir, args.file, args.row)
        
        # 基础信息审计
        shape = feature_matrix.shape
        # 统计非零包（Padding之前的数据）
        active_packets = np.count_nonzero(np.any(feature_matrix != 0, axis=1))
        
        print(f"📊 矩阵形状: {shape} (包数量 x 特征维度)")
        print(f"📦 有效报文数: {active_packets} / {shape[0]}")
        
        # 解析前3个包的特征含义 [cite: 81, 87, 89]
        print("\n[前5个报文特征采样] - (格式: [Length, IAT, Direction])")
        for i in range(min(5, active_packets)):
            p = feature_matrix[i]
            print(f"  Pkt {i+1}: Length={p[0]:.4f} | IAT={p[1]:.6f} | Dir={int(p[2])}")
        
        if args.detail:
            print("\n[完整矩阵数据]:")
            print(feature_matrix)

        print("\n" + "="*50)
        print("💡 导师提醒：请关注 IAT 的量级变化。")
        print("VPN 流量的 IAT 序列通常比常规流量具有更高的熵（扰动更大） [cite: 62, 592]。")

    except Exception as e:
        print(f"INTERNAL ERROR: {e}")

if __name__ == "__main__":
    main()

