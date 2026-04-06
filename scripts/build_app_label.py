import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


EXPLICIT_APP_RULES = [
    ("Email", {"SMTP", "SMTPS", "POP3", "POP3S", "IMAP", "IMAPS"}),
    ("P2P", {"BitTorrent", "LSD", "NAT-PMP"}),
    ("VoIP", {"STUN", "DTLS", "DTLSv1.0", "RTCP", "SIP"}),
    ("File Transfer", {"SSH", "SSHv2", "FTP"}),
    ("Chat", {"XMPP/XML"}),
    ("Streaming", {"RTMP"}),
]


EXCLUDED_PROTOCOLS = {
    "? KNXnet/IP",
    "BJNP",
    "Chargen",
    "DB-LSP-DISC/JSON",
    "DCERPC",
    "DCP-AF",
    "DCP-PFT",
    "DHCP",
    "DNS",
    "ENIP",
    "Elasticsearch",
    "ICMP",
    "ICMPv6",
    "IEEE 802.15.4",
    "IPv6",
    "LANMAN",
    "LLMNR",
    "MDNS",
    "NBNS",
    "NBSS",
    "NTP",
    "NXP 802.15.4 SNIFFER",
    "OCSP",
    "Pathport",
    "R-GOOSE",
    "SMB",
    "SMB2",
    "SNMP",
    "SPOOLSS",
    "SRVLOC",
    "SSDP",
    "THRIFT",
    "VNC",
    "WireGuard",
    "X11",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build multi-label pcap grouping from tshark app flow JSON files."
    )
    parser.add_argument(
        "--app-dir",
        default="data/app",
        help="Directory containing per-pcap five-tuple application JSON files.",
    )
    parser.add_argument(
        "--output",
        default="data/app_label.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def extract_stem(json_name: str) -> str:
    return re.sub(r"^.*?-\d+_", "", Path(json_name).stem)


def classify_by_filename(stem: str):
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

    return None


def classify_flow(apps, stem):
    filtered_apps = [app for app in apps if app not in EXCLUDED_PROTOCOLS]
    if not filtered_apps:
        return None

    app_set = set(filtered_apps)
    matched_labels = [
        label for label, protocols in EXPLICIT_APP_RULES if app_set & protocols
    ]

    # App-first only when the flow can be mapped clearly to a single class.
    if len(matched_labels) == 1:
        return matched_labels[0]

    # If app evidence is weak or conflicting, fall back to the pcap filename.
    return classify_by_filename(stem)


def main():
    args = parse_args()
    app_dir = Path(args.app_dir)
    output_path = Path(args.output)

    files = {}
    label_flow_counts = Counter()
    label_file_counts = Counter()
    multi_label_files = 0
    total_selected_flows = 0

    for path in sorted(app_dir.glob("*.json")):
        if path.name == "app_total.json":
            continue

        stem = extract_stem(path.name)
        flow_items = json.loads(path.read_text(encoding="utf-8"))
        grouped = defaultdict(list)

        for item in flow_items:
            packet_count = item.get("packet_count", 0)
            if not isinstance(packet_count, int) or packet_count <= 5:
                continue

            label = classify_flow(item.get("application", []), stem)
            if label is None:
                continue

            grouped[label].append({"five_tuple": item.get("five_tuple")})
            label_flow_counts[label] += 1
            total_selected_flows += 1

        ordered_grouped = {
            label: grouped[label] for label in sorted(grouped)
        }
        files[stem] = ordered_grouped

        if len(ordered_grouped) > 1:
            multi_label_files += 1
        for label in ordered_grouped:
            label_file_counts[label] += 1

    payload = {
        "summary": {
            "file_count": len(files),
            "multi_label_file_count": multi_label_files,
            "selected_flow_count": total_selected_flows,
            "label_file_counts": dict(label_file_counts),
            "label_flow_counts": dict(label_flow_counts),
        },
        "files": files,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote {output_path}")
    print(f"Files: {len(files)}")
    print(f"Multi-label files: {multi_label_files}")
    print(f"Selected flows: {total_selected_flows}")


if __name__ == "__main__":
    main()
