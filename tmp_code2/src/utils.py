import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from sklearn.metrics import confusion_matrix, classification_report

def plot_metrics(history, report_dir):
    """绘制损失和精度曲线"""
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('BCE Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['acc'], label='Validation Accuracy')
    plt.title('Accuracy Trend')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'learning_curves.png'))
    plt.close()

def save_confusion_matrix(y_true, y_pred, report_dir):
    """生成混淆矩阵图"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['NonVPN', 'VPN'], yticklabels=['NonVPN', 'VPN'])
    plt.title('Stage 1: VPN vs Non-VPN Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(report_dir, 'confusion_matrix.png'))
    plt.close()

def generate_markdown_report(history, y_true, y_pred, report_dir):
    """
    自动生成实验报告 Markdown 文件
    包含：核心指标、分类报告、学习曲线和混淆矩阵 [cite: 579, 610-615]
    """
    report_dict = classification_report(y_true, y_pred, target_names=['NonVPN', 'VPN'], output_dict=True)
    report_path = os.path.join(report_dir, "experiment_report.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 🔬 流量识别实验报告 - Scenario A (Stage 1)\n\n")
        f.write("## 1. 实验环境与配置\n")
        f.write("- **任务**: 区分 VPN 与 Non-VPN 流量 [cite: 131, 135]\n")
        f.write("- **架构**: 1D-CNN + Transformer 融合网络\n")
        f.write("- **流超时 (Timeout)**: 15 Seconds [cite: 207, 209]\n\n")
        
        f.write("## 2. 核心性能指标\n")
        f.write("| 指标 (Metric) | 数值 (Value) |\n")
        f.write("| :--- | :--- |\n")
        f.write(f"| **准确率 (Accuracy)** | {report_dict['accuracy']:.4f} |\n")
        f.write(f"| **宏平均 F1 (Macro F1)** | {report_dict['macro avg']['f1-score']:.4f} |\n")
        f.write(f"| **VPN 识别率 (Recall)** | {report_dict['VPN']['recall']:.4f} |\n")
        f.write(f"| **最终验证集 Loss** | {history['val_loss'][-1]:.4f} |\n\n")
        
        f.write("## 3. 详细分类评估\n")
        f.write("```text\n")
        f.write(classification_report(y_true, y_pred, target_names=['NonVPN', 'VPN']))
        f.write("\n```\n\n")
        
        f.write("## 4. 数据可视化\n")
        f.write("### 4.1 学习曲线\n")
        f.write("![Learning Curves](./learning_curves.png)\n\n")
        f.write("### 4.2 混淆矩阵\n")
        f.write("![Confusion Matrix](./confusion_matrix.png)\n")
        
    print(f"📄 实验报告已保存至: {report_path}")