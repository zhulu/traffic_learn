import os
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data_config import INDEX_PATH, load_registry_dataframe


df, _ = load_registry_dataframe(INDEX_PATH)
non_vpn_df = df[df["label1"] == "NonVPN"].copy()

print("NonVPN label2 distribution:")
print(non_vpn_df["label2"].value_counts().to_string())

print("\nLabel-to-file audit:")
for label in sorted(non_vpn_df["label2"].unique()):
    files_in_label = non_vpn_df[non_vpn_df["label2"] == label]["file"].unique()
    print(f"\n[{label}]")
    print(f"File count: {len(files_in_label)}")
    print(f"Examples: {list(files_in_label[:10])}")
