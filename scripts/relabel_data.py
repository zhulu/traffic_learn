import json
import os

import numpy as np
import pandas as pd


def run_relabel():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    old_index = os.path.join(project_root, "samples.npz")
    new_index = os.path.join(project_root, "samples_v2.npz")

    if not os.path.exists(old_index):
        print(f"Original index file not found: {old_index}")
        return

    index_data = np.load(old_index, allow_pickle=True)
    df = pd.DataFrame(index_data["data"], columns=index_data["columns"])
    df["row"] = df["row"].astype(int)

    def get_clean_label(filename):
        fname = filename.lower()

        if any(k in fname for k in ["scp", "ftps", "sftp"]):
            return "File_Transfer"
        if any(k in fname for k in ["video", "netflix", "youtube", "vimeo"]):
            return "Streaming"
        if any(k in fname for k in ["audio", "voip", "skype_audio"]):
            return "VoIP"
        if "email" in fname:
            return "Email"
        if any(k in fname for k in ["chat", "aim", "messenger", "whatsapp", "gmailchat"]):
            return "Chat"

        return "Unknown"

    mask_nonvpn = df["label1"] == "NonVPN"
    df.loc[mask_nonvpn, "label2"] = df.loc[mask_nonvpn, "file"].apply(get_clean_label)

    df = df[df["label2"] != "Unknown"].copy()

    print("\n" + "=" * 30)
    print("Non-VPN distribution after relabel:")
    print(df[df["label1"] == "NonVPN"]["label2"].value_counts())
    print("=" * 30)

    np.savez_compressed(
        new_index,
        data=df.values.astype(str),
        columns=df.columns.values.astype(str),
        stats_label1=index_data["stats_label1"],
        stats_label2="{}",
    )
    print(f"Saved new index to: {new_index}")


if __name__ == "__main__":
    run_relabel()
