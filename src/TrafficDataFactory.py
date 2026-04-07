import gc
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scapy.all import PcapReader
from tqdm import tqdm


class TrafficDataFactory:
    def __init__(
        self,
        config_path,
        output_dir="data/process",
        max_pkts=32,
        timeout=15.0,
        app_label_path=None,
    ):
        self.config_path = os.path.abspath(config_path)
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)["datasets"]["iscx_vpn_nonvpn"]

        self.project_root = os.path.dirname(os.path.dirname(self.config_path))
        self.raw_root = self._resolve_path(self.config["raw_root"])
        self.output_dir = self._resolve_output_path(output_dir)
        self.max_pkts = max_pkts
        self.timeout = timeout
        self.app_label_path = self._resolve_path(
            app_label_path or os.path.join("data", "app_label.json")
        )
        self.flow_filter_map = self._load_flow_filter_map(self.app_label_path)
        self.registry_path = self._resolve_output_path("samples.npz")
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def _label_to_key(label):
        return str(label).strip().lower().replace(" ", "_").replace("-", "_")

    def _resolve_path(self, path):
        if os.path.isabs(path):
            return path

        config_dir = os.path.dirname(self.config_path)
        candidates = [
            os.path.abspath(path),
            os.path.abspath(os.path.join(config_dir, path)),
            os.path.abspath(os.path.join(self.project_root, path)),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return candidates[-1]

    def _resolve_output_path(self, path):
        if os.path.isabs(path):
            return path
        return os.path.abspath(os.path.join(self.project_root, path))

    @staticmethod
    def _normalize_proto(proto):
        if isinstance(proto, str):
            value = proto.strip().upper()
            if value in {"6", "TCP"}:
                return "TCP"
            if value in {"17", "UDP"}:
                return "UDP"
            return value

        if proto == 6:
            return "TCP"
        if proto == 17:
            return "UDP"
        return str(proto)

    @staticmethod
    def _parse_endpoint(endpoint):
        if endpoint is None:
            return None

        value = str(endpoint).strip()
        if ":" not in value:
            return None

        host, port = value.rsplit(":", 1)
        try:
            return (host, int(port))
        except ValueError:
            return None

    @classmethod
    def _build_flow_key(cls, src_ip, src_port, dst_ip, dst_port, proto):
        endpoints = sorted(
            [(str(src_ip), int(src_port)), (str(dst_ip), int(dst_port))],
            key=lambda item: (item[0], item[1]),
        )
        return (endpoints[0], endpoints[1], cls._normalize_proto(proto))

    @classmethod
    def _normalize_labeled_flow(cls, item):
        five_tuple = (item or {}).get("five_tuple") or {}
        src = cls._parse_endpoint(five_tuple.get("src"))
        dst = cls._parse_endpoint(five_tuple.get("dst"))
        proto = five_tuple.get("proto")
        if src is None or dst is None or proto is None:
            return None
        return cls._build_flow_key(src[0], src[1], dst[0], dst[1], proto)

    def _load_flow_filter_map(self, app_label_path):
        with open(app_label_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        files = payload.get("files", {})
        flow_filter_map = {}

        for stem, label_groups in files.items():
            stem_map = {}
            seen_flow_labels = {}
            for label2, items in label_groups.items():
                flow_keys = set()
                for item in items:
                    flow_key = self._normalize_labeled_flow(item)
                    if flow_key is None:
                        continue

                    existed = seen_flow_labels.get(flow_key)
                    if existed is not None and existed != label2:
                        raise ValueError(
                            f"Conflicting labels for {stem} {flow_key}: {existed} vs {label2}"
                        )
                    seen_flow_labels[flow_key] = label2
                    flow_keys.add(flow_key)

                if flow_keys:
                    stem_map[label2] = flow_keys
            flow_filter_map[stem] = stem_map

        return flow_filter_map

    def _extract_pcap_logic(self, file_info):
        pcap_path = os.path.join(self.raw_root, file_info["relative_path"])
        stem = file_info["stem"]
        target_label2 = file_info["label2"]
        labeled_groups = self.flow_filter_map.get(stem, {})
        target_flows = labeled_groups.get(target_label2, set())
        if not target_flows:
            return None

        flows = {}
        try:
            with PcapReader(pcap_path) as pcap:
                for pkt in pcap:
                    if not (pkt.haslayer("IP") and (pkt.haslayer("TCP") or pkt.haslayer("UDP"))):
                        continue

                    src_ip, dst_ip = pkt["IP"].src, pkt["IP"].dst
                    src_port, dst_port = pkt.sport, pkt.dport
                    flow_key = self._build_flow_key(
                        src_ip, src_port, dst_ip, dst_port, pkt.proto
                    )
                    if flow_key not in target_flows:
                        continue

                    ts = float(pkt.time)
                    length = len(pkt)

                    if flow_key not in flows:
                        flows[flow_key] = {
                            "start": ts,
                            "last": ts,
                            "pkts": [],
                        }

                    if ts - flows[flow_key]["start"] > self.timeout:
                        continue

                    iat = ts - flows[flow_key]["last"]
                    direct = 1 if (str(src_ip), int(src_port)) == flow_key[0] else 0
                    flows[flow_key]["pkts"].append([length, iat, direct])
                    flows[flow_key]["last"] = ts

            data_list = []
            stats_list = []

            for info in flows.values():
                pkts = info["pkts"]
                if len(pkts) < 3:
                    continue

                all_lengths = [p[0] for p in pkts]
                all_iats = [p[1] for p in pkts]
                duration = info["last"] - info["start"]
                duration = duration if duration > 0.0001 else 0.0001

                flow_stats = [
                    np.mean(all_lengths),
                    np.std(all_lengths),
                    np.max(all_lengths),
                    np.min(all_lengths),
                    np.mean(all_iats),
                    np.std(all_iats),
                    np.max(all_iats),
                    np.min(all_iats),
                    sum(all_lengths) / duration,
                    len(pkts) / duration,
                ]

                mat = pkts[: self.max_pkts]
                while len(mat) < self.max_pkts:
                    mat.append([0.0, 0.0, 0.0])

                data_list.append(np.array(mat, dtype=np.float32))
                stats_list.append(np.array(flow_stats, dtype=np.float32))

            if data_list:
                save_path = os.path.join(self.output_dir, f"{stem}.npz")
                np.savez_compressed(
                    save_path,
                    features=np.array(data_list),
                    stats=np.array(stats_list),
                )

                count = len(data_list)
                del flows, data_list, stats_list
                gc.collect()

                return {
                    "stem": stem,
                    "count": count,
                    "label1": file_info["label1"],
                    "label2": target_label2,
                    "label3": file_info.get("label3", ""),
                    "label1_label3": file_info.get("label1_label3", ""),
                }
        except Exception as e:
            print(f"Error processing {stem}: {e}")
        return None

    def run_parallel(self, workers=14):
        tasks = self.config["files"]
        total_files = len(tasks)
        sample_registry = []
        stats_info = {"label1": {}, "label2": {}}

        print(f"Starting preprocessing with {workers} workers...")

        with tqdm(total=total_files, desc="Processing PCAPs", unit="file") as pbar:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                future_to_file = {
                    executor.submit(self._extract_pcap_logic, task): task for task in tasks
                }

                for future in as_completed(future_to_file):
                    task_info = future_to_file[future]
                    curr_stem = task_info["stem"]

                    try:
                        res = future.result()
                        if res:
                            label1_label2 = (
                                f"{res['label1']}_{self._label_to_key(res['label2'])}"
                            )
                            for i in range(res["count"]):
                                sample_registry.append(
                                    [
                                        f"{res['stem']}.npz",
                                        i,
                                        res["label1"],
                                        res["label2"],
                                        res["label3"],
                                        label1_label2,
                                        res["label1_label3"],
                                    ]
                                )

                            stats_info["label1"][res["label1"]] = (
                                stats_info["label1"].get(res["label1"], 0) + res["count"]
                            )
                            stats_info["label2"][res["label2"]] = (
                                stats_info["label2"].get(res["label2"], 0) + res["count"]
                            )

                            pbar.set_postfix({"file": curr_stem[:10], "samples": res["count"]})
                    except Exception as e:
                        print(f"\nFailed to process {curr_stem}: {e}")

                    pbar.update(1)

        self._save_registry(sample_registry, stats_info)

    def _save_registry(self, registry_data, stats_info):
        df = pd.DataFrame(
            registry_data,
            columns=[
                "file",
                "row",
                "label1",
                "label2",
                "label3",
                "label1_label2",
                "label1_label3",
            ],
        )
        np.savez_compressed(
            self.registry_path,
            data=df.values.astype(str),
            columns=df.columns.values.astype(str),
            stats_label1=json.dumps(stats_info["label1"]),
            stats_label2=json.dumps(stats_info["label2"]),
        )
        print(f"Updated registry: {self.registry_path}")


def load_sample_full(process_dir, npz_name, row_idx):
    data_path = npz_name if os.path.isabs(npz_name) else os.path.join(process_dir, npz_name)
    data = np.load(data_path)
    return data["features"][row_idx], data["stats"][row_idx]
