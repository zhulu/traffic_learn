import json
import os
import sys

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from src.TrafficDataFactory import TrafficDataFactory


def main():
    config_path = os.path.join(PROJECT_ROOT, "data", "label.json")
    app_label_path = os.path.join(PROJECT_ROOT, "data", "app_label.json")
    process_dir = os.path.join(PROJECT_ROOT, "data", "process")
    registry_file = os.path.join(PROJECT_ROOT, "samples.npz")

    factory = TrafficDataFactory(
        config_path=config_path,
        output_dir=process_dir,
        max_pkts=64,
        timeout=15.0,
        app_label_path=app_label_path,
    )
    factory.registry_path = registry_file

    print("=" * 50)
    print("Starting packet-level preprocessing with five-tuple labels")
    print(f"Label config: {config_path}")
    print(f"Five-tuple label file: {app_label_path}")
    print(f"Output feature dir: {process_dir}")
    print(f"Registry file: {registry_file}")
    print("=" * 50)

    factory.run_parallel(workers=14)

    if os.path.exists(registry_file):
        registry = np.load(registry_file, allow_pickle=True)
        label1_stats = json.loads(str(registry["stats_label1"]))
        label2_stats = json.loads(str(registry["stats_label2"]))

        print("\n" + "=" * 20 + " Summary " + "=" * 20)
        print("\n[Label1]")
        for key, value in label1_stats.items():
            print(f" - {key:15}: {value} samples")

        print("\n[Label2]")
        for key, value in label2_stats.items():
            warning = " [low-sample]" if value < 100 else ""
            print(f" - {key:15}: {value} samples{warning}")

        print(f"\nTotal valid samples: {len(registry['data'])}")
        print("=" * 55)
    else:
        print(f"Failed to generate registry file: {registry_file}")


if __name__ == "__main__":
    main()
