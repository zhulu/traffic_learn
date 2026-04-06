import argparse
import json
import re
from collections import Counter
from pathlib import Path


APP_RULES = [
    ("P2P", {"BitTorrent", "LSD", "NAT-PMP"}),
    ("VoIP", {"STUN", "DTLS", "DTLSv1.0", "RTCP", "SIP"}),
    ("File Transfer", {"SSH", "SSHv2", "FTP"}),
    ("Chat", {"XMPP/XML"}),
    ("Streaming", {"RTMP"}),
    ("Web Browsing", {"BROWSER", "HTTP"}),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build five-tuple label2 annotations for ISCX_VPN-NonVPN."
    )
    parser.add_argument(
        "--app-dir",
        default="data/app",
        help="Directory that stores tshark-derived per-pcap application JSON files.",
    )
    parser.add_argument(
        "--label-config",
        default="data/label.json",
        help="Dataset label config used to copy label2 class metadata.",
    )
    parser.add_argument(
        "--output",
        default="data/iscx_vpn_nonvpn_five_tuple_label2.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def extract_stem(json_name: str) -> str:
    return re.sub(r"^.*?-\d+_", "", Path(json_name).stem)


def classify_by_app(apps):
    app_set = set(apps)
    for label, protocols in APP_RULES:
        if app_set & protocols:
            return label
    return None


def classify_by_stem(stem: str):
    name = stem.lower()

    if "bittorrent" in name or "torrent" in name:
        return "P2P"
    if "email" in name:
        return "Email"
    if any(token in name for token in ["scp", "sftp", "ftps", "file"]):
        return "File Transfer"
    if "voipbuster" in name or "audio" in name:
        return "VoIP"
    if any(token in name for token in ["chat", "aim", "icq", "gmailchat"]):
        return "Chat"
    if any(token in name for token in ["video", "netflix", "youtube", "vimeo", "spotify"]):
        return "Streaming"
    if any(token in name for token in ["firefox", "chrome", "browser", "web"]):
        return "Web Browsing"

    raise ValueError(f"Unable to classify pcap stem by filename fallback: {stem}")


def build_records(app_dir: Path):
    records = []
    label_counts = Counter()
    source_counts = Counter()
    source_label_counts = Counter()

    for path in sorted(app_dir.glob("*.json")):
        if path.name == "app_total.json":
            continue

        stem = extract_stem(path.name)
        items = json.loads(path.read_text(encoding="utf-8"))

        if not isinstance(items, list):
            raise TypeError(f"{path} must contain a list, got {type(items).__name__}")

        for item in items:
            apps = item.get("application", [])
            label = classify_by_app(apps)
            label_source = "app" if label else "filename"
            if label is None:
                label = classify_by_stem(stem)

            record = {
                "pcap": stem,
                "five_tuple": item.get("five_tuple"),
                "label2": label,
                "label_source": label_source,
                "application": apps,
                "packet_count": item.get("packet_count"),
            }
            records.append(record)
            label_counts[label] += 1
            source_counts[label_source] += 1
            source_label_counts[f"{label_source}:{label}"] += 1

    return records, label_counts, source_counts, source_label_counts


def main():
    args = parse_args()
    app_dir = Path(args.app_dir)
    label_config_path = Path(args.label_config)
    output_path = Path(args.output)

    dataset_config = json.loads(label_config_path.read_text(encoding="utf-8"))["datasets"][
        "iscx_vpn_nonvpn"
    ]

    records, label_counts, source_counts, source_label_counts = build_records(app_dir)

    payload = {
        "dataset": "iscx_vpn_nonvpn",
        "priority": ["application", "pcap_name"],
        "label2_classes": dataset_config["label2_classes"],
        "app_rules": {label: sorted(protocols) for label, protocols in APP_RULES},
        "summary": {
            "record_count": len(records),
            "label2_counts": dict(label_counts),
            "label_source_counts": dict(source_counts),
            "label_source_label2_counts": dict(source_label_counts),
        },
        "records": records,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote {output_path}")
    print(f"Records: {len(records)}")
    print(f"Label counts: {dict(label_counts)}")
    print(f"Source counts: {dict(source_counts)}")


if __name__ == "__main__":
    main()
