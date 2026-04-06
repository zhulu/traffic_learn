import os
import json
import numpy as np
import pandas as pd
from scapy.all import PcapReader
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time

class TrafficDataFactory:
    def __init__(self, config_path, output_dir="process", max_pkts=32, timeout=15.0):
        with open(config_path, 'r') as f:
            self.config = json.load(f)['datasets']['iscx_vpn_nonvpn']
        
        self.raw_root = self.config['raw_root']
        self.output_dir = output_dir
        self.max_pkts = max_pkts
        self.timeout = timeout
        os.makedirs(self.output_dir, exist_ok=True)

    def _extract_pcap_logic(self, file_info):
        """
        单文件提取逻辑：15s内、前N个包 [cite: 50, 591]
        """
        pcap_path = os.path.join(self.raw_root, file_info['relative_path'])
        stem = file_info['stem']
        
        flows = {}
        try:
            with PcapReader(pcap_path) as pcap:
                for pkt in pcap:
                    if not (pkt.haslayer('IP') and (pkt.haslayer('TCP') or pkt.haslayer('UDP'))):
                        continue
                    
                    # 五元组聚合双向流 [cite: 141]
                    src_ip, dst_ip = pkt['IP'].src, pkt['IP'].dst
                    src_port, dst_port = pkt.sport, pkt.dport
                    proto = pkt.proto
                    fid = tuple(sorted([(src_ip, src_port), (dst_ip, dst_port)]) + [proto])
                    
                    ts = float(pkt.time)
                    length = len(pkt)
                    
                    if fid not in flows:
                        flows[fid] = {'start': ts, 'last': ts, 'pkts': []}
                    
                    # 15s 约束 [cite: 616]
                    if ts - flows[fid]['start'] > self.timeout: continue
                    
                    if len(flows[fid]['pkts']) < self.max_pkts:
                        iat = ts - flows[fid]['last']
                        direct = 1 if src_ip == fid[0][0] else 0
                        # 特征：[长度, IAT, 方向]
                        flows[fid]['pkts'].append([length, iat, direct])
                        flows[fid]['last'] = ts

            # 封装并保存
            data_list = []
            for info in flows.values():
                if len(info['pkts']) < 3: continue
                mat = info['pkts']
                while len(mat) < self.max_pkts:
                    mat.append([0.0, 0.0, 0.0])
                data_list.append(np.array(mat, dtype=np.float32))

            if data_list:
                save_path = os.path.join(self.output_dir, f"{stem}.npz")
                np.savez_compressed(save_path, features=np.array(data_list))
                
                # 显式清理内存
                del flows, data_list
                gc.collect()
                
                return {
                    "stem": stem,
                    "count": len(np.load(save_path)['features']),
                    "label1": file_info['label1'],
                    "label2": file_info['label2']
                }
        except Exception as e:
            print(f"Error processing {stem}: {e}")
        return None

    def run_parallel(self, workers=14):
        tasks = self.config['files']
        sample_registry = []
        
        print(f"🚀 开始并行抽取特征（最大包数: {self.max_pkts}, 超时: {self.timeout}s）...")
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(self._extract_pcap_logic, tasks))
        
        # 构建全局 sample.npz 和统计信息
        registry_data = []
        stats = {"label1": {}, "label2": {}}
        
        for res in results:
            if res:
                stem = res['stem']
                for i in range(res['count']):
                    registry_data.append([f"{stem}.npz", i, res['label1'], res['label2']])
                
                # 统计
                stats["label1"][res['label1']] = stats["label1"].get(res['label1'], 0) + res['count']
                stats["label2"][res['label2']] = stats["label2"].get(res['label2'], 0) + res['count']

        # 保存全局索引
        df = pd.DataFrame(registry_data, columns=['file', 'row', 'label1', 'label2'])
        np.savez_compressed("samples.npz", 
                            data=df.values.astype(str), 
                            columns=df.columns.values.astype(str),
                            stats_label1=json.dumps(stats["label1"]),
                            stats_label2=json.dumps(stats["label2"]))
        
        print(f"✅ 处理完成。全局索引已存至 samples.npz")
        print(f"📊 统计信息: {stats}")
    
    def run_parallel(self, workers=14):
        tasks = self.config['files']
        total_files = len(tasks)
        sample_registry = []
        stats = {"label1": {}, "label2": {}}
        
        print(f"🚀 [导师指令]：启动 {workers} 核并行引擎，准备解析 {total_files} 个文件...")
        
        # 使用 tqdm 创建主进度条
        with tqdm(total=total_files, desc="📊 流量解析进度", unit="file") as pbar:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                # 提交所有任务
                future_to_file = {executor.submit(self._extract_pcap_logic, task): task for task in tasks}
                
                for future in as_completed(future_to_file):
                    task_info = future_to_file[future]
                    fname = task_info['file_name']
                    
                    try:
                        res = future.result()
                        if res:
                            stem = res['stem']
                            # 更新全局索引数据
                            for i in range(res['count']):
                                sample_registry.append([f"{stem}.npz", i, res['label1'], res['label2']])
                            
                            # 更新统计字典
                            stats["label1"][res['label1']] = stats["label1"].get(res['label1'], 0) + res['count']
                            stats["label2"][res['label2']] = stats["label2"].get(res['label2'], 0) + res['count']
                            
                            # 动态更新进度条右侧的描述
                            pbar.set_postfix({"last_file": f"{fname[:10]}...", "flows": res['count']})
                    except Exception as e:
                        print(f"\n❌ 文件 {fname} 处理失败: {e}")
                    
                    # 无论成功失败，进度条都推进一步
                    pbar.update(1)

        # 保存全局索引 samples.npz (逻辑同前)
        self._save_registry(sample_registry, stats)
        
    def _save_registry(self, registry_data, stats):
        # 封装保存逻辑
        df = pd.DataFrame(registry_data, columns=['file', 'row', 'label1', 'label2'])
        np.savez_compressed("data/samples.npz", 
                            data=df.values.astype(str), 
                            columns=df.columns.values.astype(str),
                            stats_label1=json.dumps(stats["label1"]),
                            stats_label2=json.dumps(stats["label2"]))

# --- 接口：解析 NPZ 数据 ---
def load_sample_feature(process_dir, npz_name, row_idx):
    """
    根据索引加载单个样本特征
    """
    data = np.load(os.path.join(process_dir, npz_name))
    return data['features'][row_idx]

