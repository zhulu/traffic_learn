import os
import json
import numpy as np
from src.TrafficDataFactory import TrafficDataFactory  # 假设之前的代码保存在此文件中

def main():
    # --- 1. 配置路径 ---
    # 根据你的数据组织结构进行配置
    CONFIG_PATH = "data/label.json"
    PROCESS_DIR = "data/process"
    REGISTRY_FILE = "data/samples.npz"
    
    # --- 2. 初始化预处理引擎 ---
    # 严格遵循论文结论：15s超时，提取前32个包 [cite: 207, 209]
    factory = TrafficDataFactory(
        config_path=CONFIG_PATH,
        output_dir=PROCESS_DIR,
        max_pkts=32,
        timeout=15.0
    )
    
    print("="*50)
    print("🚀 科研导师指令：启动高性能流量特征预处理流水线")
    print(f"📍 原始数据根目录: {factory.raw_root}")
    print(f"📍 特征存储目录: {PROCESS_DIR}")
    print(f"⚙️  硬件配置: 14核并行处理 (留2核保命)")
    print("="*50)

    # --- 3. 执行并行抽取 ---
    # 这一步会遍历 PCAP，生成单个 NPZ，并返回全局索引 [cite: 144, 145]
    factory.run_parallel(workers=14)

    # --- 4. 统计结果分析与数据审计 ---
    if os.path.exists(REGISTRY_FILE):
        registry = np.load(REGISTRY_FILE, allow_pickle=True)
        
        # 解析统计信息
        label1_stats = json.loads(str(registry['stats_label1']))
        label2_stats = json.loads(str(registry['stats_label2']))
        
        print("\n" + "="*20 + " 📊 数据审计报告 " + "="*20)
        
        print(f"\n[维度 1: 隧道类型 (Tunnel)]")
        for k, v in label1_stats.items():
            print(f" - {k:15}: {v} 个流样本")

        print(f"\n[维度 2: 应用类型 (Application)]")
        for k, v in label2_stats.items():
            # 导师提醒：重点关注样本量过少的类别 [cite: 75]
            warning = " ⚠️ (样本极少，建议数据增强)" if v < 100 else ""
            print(f" - {k:15}: {v} 个流样本{warning}")

        # 检查总样本数
        total_samples = len(registry['data'])
        print(f"\n✅ 预处理任务圆满完成！")
        print(f"📊 总计抽取有效流样本: {total_samples}")
        print("="*55)
    else:
        print("❌ 错误：未能生成全局索引文件 samples.npz，请检查日志。")

if __name__ == "__main__":
    main()