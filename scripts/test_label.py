import numpy as np
import pandas as pd
import json

# 加载索引
index_data = np.load("samples.npz", allow_pickle=True)
df = pd.DataFrame(index_data['data'], columns=index_data['columns'])

# 1. 检查 NonVPN 下 label2 的真实取值
non_vpn_df = df[df['label1'] == 'NonVPN']
print("🔍 Non-VPN 下各应用的真实分布：")
print(non_vpn_df['label2'].value_counts())

# 2. 检查一下 STAGE2_LABELS 里的名称是否写错了
# 比如数据里叫 'Browsing'，但我们代码里映射的是 'Web Browsing'


# 重点看 Non-VPN 的分布
non_vpn_df = df[df['label1'] == 'NonVPN']

print("📊 --- 类别与原始文件映射审计 ---")
for label in non_vpn_df['label2'].unique():
    files_in_label = non_vpn_df[non_vpn_df['label2'] == label]['file'].unique()
    print(f"\n【类别: {label}】")
    print(f"包含原始文件数: {len(files_in_label)}")
    print(f"文件名示例: {files_in_label[:10]}") # 打印前10个文件名看看

