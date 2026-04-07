import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data_config import INDEX_PATH, LABEL_CONFIG_PATH


def parse_args():
    parser = argparse.ArgumentParser(
        description="Repair samples.npz so registry labels follow data/label.json ground truth."
    )
    parser.add_argument(
        "--index-path",
        default=INDEX_PATH,
        help="Path to samples.npz.",
    )
    parser.add_argument(
        "--label-config",
        default=LABEL_CONFIG_PATH,
        help="Path to data/label.json.",
    )
    parser.add_argument(
        "--keep-mismatched",
        action="store_true",
        help="Keep rows whose current label2 disagrees with file-level label2 and overwrite labels in place.",
    )
    return parser.parse_args()


def load_truth_map(label_config_path):
    with open(label_config_path, "r", encoding="utf-8") as handle:
        files = json.load(handle)["datasets"]["iscx_vpn_nonvpn"]["files"]

    truth_map = {}
    for item in files:
        truth_map[f"{item['stem']}.npz"] = {
            "label1": item["label1"],
            "label2": item["label2"],
            "label3": item["label3"],
            "label1_label2": item["label1_label2"],
            "label1_label3": item["label1_label3"],
        }
    return truth_map


def main():
    args = parse_args()
    truth_map = load_truth_map(args.label_config)

    index_data = np.load(args.index_path, allow_pickle=True)
    df = pd.DataFrame(index_data["data"], columns=index_data["columns"])
    df["row"] = df["row"].astype(int)

    truth_df = pd.DataFrame.from_dict(truth_map, orient="index").reset_index()
    truth_df = truth_df.rename(columns={"index": "file"})
    merged = df.merge(truth_df, on="file", how="left", suffixes=("", "_truth"))

    missing_truth = merged["label2_truth"].isna().sum()
    if missing_truth:
        raise ValueError(f"{missing_truth} rows could not be matched to label.json")

    mismatched = merged["label2"] != merged["label2_truth"]
    mismatch_count = int(mismatched.sum())

    if args.keep_mismatched:
        fixed = merged.copy()
    else:
        fixed = merged.loc[~mismatched].copy()

    for col in ["label1", "label2", "label3", "label1_label2", "label1_label3"]:
        fixed[col] = fixed[f"{col}_truth"]

    fixed = fixed[df.columns].copy()

    stats_label1 = fixed["label1"].value_counts().sort_index().to_dict()
    stats_label2 = fixed["label2"].value_counts().sort_index().to_dict()

    np.savez_compressed(
        args.index_path,
        data=fixed.values.astype(str),
        columns=fixed.columns.values.astype(str),
        stats_label1=json.dumps(stats_label1, ensure_ascii=False),
        stats_label2=json.dumps(stats_label2, ensure_ascii=False),
    )

    print(f"Updated registry: {args.index_path}")
    print(f"Total rows before: {len(df)}")
    print(f"Mismatched label2 rows: {mismatch_count}")
    print(f"Total rows after: {len(fixed)}")
    print(f"Mode: {'overwrite mismatched rows' if args.keep_mismatched else 'drop mismatched rows'}")


if __name__ == "__main__":
    main()
