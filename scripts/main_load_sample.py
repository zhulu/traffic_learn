import argparse
import os

import numpy as np


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))


def load_sample_feature(process_dir, npz_name, row_idx):
    if os.path.sep in npz_name:
        path = npz_name
    else:
        path = os.path.join(process_dir, npz_name)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature file not found: {path}")

    data = np.load(path)
    features = data["features"]

    if row_idx >= len(features):
        raise IndexError(
            f"Row index {row_idx} is out of range for {path} (total rows: {len(features)})"
        )

    return features[row_idx]


def main():
    parser = argparse.ArgumentParser(description="Inspect one processed flow sample.")
    parser.add_argument("--file", type=str, required=True, help="NPZ file name")
    parser.add_argument("--row", type=int, required=True, help="Row index")
    parser.add_argument(
        "--dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "process"),
        help="Processed feature directory",
    )
    parser.add_argument("--detail", action="store_true", help="Print the full packet matrix")

    args = parser.parse_args()

    print("=" * 50)
    print(f"Inspecting sample: {args.file} | row={args.row}")
    print("=" * 50)

    try:
        feature_matrix = load_sample_feature(args.dir, args.file, args.row)

        shape = feature_matrix.shape
        active_packets = np.count_nonzero(np.any(feature_matrix != 0, axis=1))

        print(f"Feature matrix shape: {shape}")
        print(f"Active packets: {active_packets} / {shape[0]}")

        print("\nFirst packets [Length, IAT, Direction]:")
        for i in range(min(5, active_packets)):
            packet = feature_matrix[i]
            print(
                f"  Pkt {i + 1}: Length={packet[0]:.4f} | "
                f"IAT={packet[1]:.6f} | Dir={int(packet[2])}"
            )

        if args.detail:
            print("\nFull matrix:")
            print(feature_matrix)

    except Exception as exc:
        print(f"INTERNAL ERROR: {exc}")


if __name__ == "__main__":
    main()
