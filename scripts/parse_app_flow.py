import os
import subprocess
import json
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import defaultdict

def analyze_single_pcap(pcap_path, output_dir):
    """
    针对复杂协议优化的 PCAP 解析逻辑
    """
    # 构造唯一的 JSON 文件名
    rel_path = os.path.relpath(pcap_path, start=os.path.dirname(output_dir))
    safe_name = rel_path.replace(os.sep, "_").replace(".pcapng", "").replace(".pcap", "")
    json_path = os.path.join(output_dir, f"{safe_name}.json")

    # --- 关键修改：将分隔符改为制表符 \t ---
    cmd = [
        "tshark", "-r", pcap_path,
        "-T", "fields",
        "-E", "separator=/t",  # 使用制表符，避免协议名中的逗号干扰
        "-e", "ip.src",
        "-e", "tcp.srcport",
        "-e", "udp.srcport",
        "-e", "ip.dst",
        "-e", "tcp.dstport",
        "-e", "udp.dstport",
        "-e", "ip.proto",
        "-e", "_ws.col.Protocol"
    ]

    flows = defaultdict(lambda: {"app_protocols": set(), "packet_count": 0, "transport": ""})

    try:
        # 启动 tshark
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        
        for line in process.stdout:
            # 使用 \t 进行切割
            parts = line.strip().split('\t')
            
            # 防御性检查：确保至少有 8 个字段
            if len(parts) < 8:
                continue

            # 安全取值，防止某些字段缺失导致越界
            src_ip = parts[0]
            tcp_sp = parts[1]
            udp_sp = parts[2]
            dst_ip = parts[3]
            tcp_dp = parts[4]
            udp_dp = parts[5]
            proto_num = parts[6]
            app_proto = parts[7] # 哪怕协议名里有逗号，现在也能完整拿到
            
            is_tcp = (proto_num == '6')
            src_port = tcp_sp if is_tcp else udp_sp
            dst_port = tcp_dp if is_tcp else udp_dp
            transport = "TCP" if is_tcp else ("UDP" if proto_num == '17' else "Other")

            if not src_port or not dst_port or not src_ip or not dst_ip:
                continue

            # 五元组聚合 (排序确保双向流合一)
            flow_key = tuple(sorted([(src_ip, src_port), (dst_ip, dst_port)]) + [transport])

            flows[flow_key]["packet_count"] += 1
            flows[flow_key]["transport"] = transport
            
            # 协议过滤逻辑
            base_protos = {"TCP", "UDP", "IPv4", "IPv6", "TLSv1.2", "TLSv1.3", "SSL", "HTTP"}
            if app_proto not in base_protos:
                flows[flow_key]["app_protocols"].add(app_proto)
            elif not flows[flow_key]["app_protocols"]:
                flows[flow_key]["app_protocols"].add(app_proto)

        process.stdout.close()
        process.wait()

        # 整理结果
        output_result = []
        for key, info in flows.items():
            output_result.append({
                "five_tuple": {
                    "src": f"{key[0][0]}:{key[0][1]}",
                    "dst": f"{key[1][0]}:{key[1][1]}",
                    "proto": key[2]
                },
                "application": list(info["app_protocols"]),
                "packet_count": info["packet_count"]
            })

        # 写入 JSON
        if output_result:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(output_result, f, indent=4)

    except Exception as e:
        return f"Error {pcap_path}: {str(e)}"
    
    finally:
        # 内存释放
        del flows
        if 'output_result' in locals(): del output_result
        gc.collect()
        
    return f"Success: {safe_name}"

def run_parallel_audit(target_dir, output_dir, max_workers=14):
    os.makedirs(output_dir, exist_ok=True)
    
    extensions = ('.pcap', '.pcapng')
    all_files = []
    for root, _, files in os.walk(target_dir):
        for file in files:
            if file.lower().endswith(extensions):
                all_files.append(os.path.join(root, file))

    print(f"🚀 并行审计启动 | 线程数: {max_workers} | 正在处理 {len(all_files)} 个文件...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pcap = {executor.submit(analyze_single_pcap, f, output_dir): f for f in all_files}
        
        with tqdm(total=len(all_files), desc="📦 协议深度检测中") as pbar:
            for future in as_completed(future_to_pcap):
                res = future.result()
                if "Error" in res:
                    print(f"\n❌ {res}")
                pbar.update(1)

if __name__ == "__main__":
    # 路径配置
    RAW_DATA_PATH = "./data/ISCX_VPN-NonVPN" # 修改为你的路径
    APP_JSON_PATH = "./data/app"
    
    run_parallel_audit(RAW_DATA_PATH, APP_JSON_PATH, max_workers=14)
    print(f"\n✅ 审计完成！所有 JSON 已存入: {APP_JSON_PATH}")
