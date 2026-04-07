import os
import sys
import json


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data_config import INDEX_PATH, LABEL_CONFIG_PATH, load_registry_dataframe


df, _ = load_registry_dataframe(INDEX_PATH)
non_vpn_df = df[df["label1"] == "NonVPN"].copy()

with open(LABEL_CONFIG_PATH, "r", encoding="utf-8") as handle:
    file_truth = {
        f"{item['stem']}.npz": item["label2"]
        for item in json.load(handle)["datasets"]["iscx_vpn_nonvpn"]["files"]
    }

df["file_label2_truth"] = df["file"].map(file_truth)
label_mismatch_df = df[df["label2"] != df["file_label2_truth"]].copy()

print("NonVPN label2 distribution:")
print(non_vpn_df["label2"].value_counts().to_string())

print("\nRegistry/file label2 mismatches:")
print(f"Mismatch rows: {len(label_mismatch_df)} / {len(df)}")
if not label_mismatch_df.empty:
    print(
        label_mismatch_df.groupby(["file", "file_label2_truth", "label2"])
        .size()
        .sort_values(ascending=False)
        .head(20)
        .to_string()
    )

print("\nLabel-to-file audit:")
for label in sorted(non_vpn_df["label2"].unique()):
    files_in_label = non_vpn_df[non_vpn_df["label2"] == label]["file"].unique()
    print(f"\n[{label}]")
    print(f"File count: {len(files_in_label)}")
    print(f"Examples: {list(files_in_label[:10])}")
