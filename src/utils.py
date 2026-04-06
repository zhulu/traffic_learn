import json
import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def plot_metrics(history, report_dir):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Model Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["acc"], label="Validation Accuracy")
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "learning_curves.png"))
    plt.close()


def save_confusion_matrix(
    y_true,
    y_pred,
    report_dir,
    labels=None,
    title="Confusion Matrix",
):
    labels = labels or ["NonVPN", "VPN"]
    label_indices = list(range(len(labels)))
    cm = confusion_matrix(y_true, y_pred, labels=label_indices)

    csv_path = os.path.join(report_dir, "confusion_matrix.csv")
    json_path = os.path.join(report_dir, "confusion_matrix.json")
    with open(csv_path, "w", encoding="utf-8") as handle:
        handle.write("actual/predicted," + ",".join(labels) + "\n")
        for label, row in zip(labels, cm):
            handle.write(label + "," + ",".join(str(int(value)) for value in row) + "\n")

    with open(json_path, "w", encoding="utf-8") as handle:
        rows = []
        for label, row in zip(labels, cm):
            rows.append(
                {
                    "actual_label": label,
                    "predicted_counts": {
                        pred_label: int(value) for pred_label, value in zip(labels, row)
                    },
                }
            )
        handle.write(
            json.dumps(
                {"labels": labels, "matrix": cm.tolist(), "rows": rows},
                ensure_ascii=False,
                indent=2,
            )
        )

    figsize = (7, 6) if len(labels) <= 2 else (11, 9)
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": 12 if len(labels) <= 2 else 9},
    )
    plt.title(title)
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "confusion_matrix.png"))
    plt.close()

    print("Confusion matrix data:")
    print("actual/predicted\t" + "\t".join(labels))
    for label, row in zip(labels, cm):
        print(label + "\t" + "\t".join(str(int(value)) for value in row))
    print(f"Saved confusion matrix data to: {csv_path}")
    print(f"Saved confusion matrix JSON to: {json_path}")

    return cm


def generate_markdown_report(
    history,
    y_true,
    y_pred,
    report_dir,
    target_names=None,
    stage_name="Stage 1",
):
    target_names = target_names or ["NonVPN", "VPN"]
    label_indices = list(range(len(target_names)))
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=label_indices,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    report_path = os.path.join(report_dir, "experiment_report.md")

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(f"# Traffic Classification Report ({stage_name})\n\n")
        handle.write("## Setup\n")
        handle.write(f"- Task: {stage_name}\n")
        handle.write("- Model: 1D-CNN + Transformer with flow statistics fusion\n")
        handle.write("- Flow timeout: 15 seconds\n\n")

        handle.write("## Summary Metrics\n")
        handle.write("| Metric | Value |\n")
        handle.write("| :--- | :--- |\n")
        handle.write(f"| Accuracy | {report_dict['accuracy']:.4f} |\n")
        handle.write(f"| Macro F1 | {report_dict['macro avg']['f1-score']:.4f} |\n")
        if "VPN" in target_names and len(target_names) == 2:
            handle.write(f"| VPN Recall | {report_dict['VPN']['recall']:.4f} |\n")
        handle.write(f"| Final Validation Loss | {history['val_loss'][-1]:.4f} |\n\n")

        if len(target_names) > 2:
            handle.write("## Per-Class Metrics\n")
            handle.write("| Class | Precision | Recall | F1-score | Support |\n")
            handle.write("| :--- | :--- | :--- | :--- | :--- |\n")
            for name in target_names:
                metrics = report_dict[name]
                handle.write(
                    f"| {name} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
                    f"{metrics['f1-score']:.4f} | {metrics['support']} |\n"
                )
            handle.write("\n")

        handle.write("## Detailed Report\n")
        handle.write("```text\n")
        handle.write(
            classification_report(
                y_true,
                y_pred,
                labels=label_indices,
                target_names=target_names,
                zero_division=0,
            )
        )
        handle.write("\n```\n\n")

        handle.write("## Figures\n")
        handle.write("### Learning Curves\n")
        handle.write("![Learning Curves](./learning_curves.png)\n\n")
        handle.write("### Confusion Matrix\n")
        handle.write("![Confusion Matrix](./confusion_matrix.png)\n")
        handle.write("\n")
        handle.write("### Confusion Matrix Data\n")
        handle.write("CSV: `./confusion_matrix.csv`\n\n")
        handle.write("JSON: `./confusion_matrix.json`\n")

    print(f"Saved experiment report to: {report_path}")
