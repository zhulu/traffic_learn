import json
import os

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESS_DIR = os.path.join(DATA_DIR, "process")
INDEX_PATH = os.path.join(PROJECT_ROOT, "samples.npz")
APP_LABEL_PATH = os.path.join(DATA_DIR, "app_label.json")
LABEL_CONFIG_PATH = os.path.join(DATA_DIR, "label.json")

LABEL1_LABELS = ["NonVPN", "VPN"]
DEFAULT_APP_LABEL_ORDER = [
    "Chat",
    "Email",
    "File Transfer",
    "P2P",
    "Streaming",
    "VoIP",
]


def _ordered_labels(labels, preferred_order=None):
    ordered = []
    preferred_order = preferred_order or []

    for label in preferred_order:
        if label in labels and label not in ordered:
            ordered.append(label)

    for label in sorted(set(labels)):
        if label not in ordered:
            ordered.append(label)

    return ordered


def load_registry_dataframe(index_path=INDEX_PATH):
    index_data = np.load(index_path, allow_pickle=True)
    df = pd.DataFrame(index_data["data"], columns=index_data["columns"])

    if "row" in df.columns:
        df["row"] = df["row"].astype(int)

    return df, index_data


def load_app_label_payload(app_label_path=APP_LABEL_PATH):
    with open(app_label_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def get_label2_classes(dataframe=None, app_label_path=APP_LABEL_PATH):
    labels = []

    if dataframe is not None and "label2" in dataframe.columns:
        labels.extend(dataframe["label2"].dropna().astype(str).unique().tolist())

    if not labels and os.path.exists(app_label_path):
        payload = load_app_label_payload(app_label_path)
        summary = payload.get("summary", {})
        label_counts = summary.get("label_flow_counts", {})
        labels.extend(label_counts.keys())

    return _ordered_labels(labels, preferred_order=DEFAULT_APP_LABEL_ORDER)


def get_label3_classes(dataframe, min_count=1):
    if dataframe is None or "label3" not in dataframe.columns:
        return []

    counts = dataframe["label3"].dropna().astype(str).value_counts()
    labels = counts[counts >= min_count].index.tolist()
    return _ordered_labels(labels)


def build_label_map(labels):
    return {label: idx for idx, label in enumerate(labels)}
