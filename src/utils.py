import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Stage 2 标准应用标签顺序 (必须与 Dataset 中的映射一致)
#STAGE2_LABELS = ['Web Browsing', 'Email', 'Chat', 'Streaming', 'File Transfer', 'VoIP', 'P2P']
# 修改为真实的 5 类
STAGE2_LABELS = ['File_Transfer', 'Streaming', 'VoIP', 'Email', 'Chat']
def plot_metrics(history, report_dir):
    """绘制损失和精度曲线 (兼容所有阶段)"""
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Model Loss Curve')
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

def save_confusion_matrix(y_true, y_pred, report_dir, labels=['NonVPN', 'VPN'], title='Confusion Matrix'):
    """
    生成混淆矩阵图
    - labels: 标签文本列表
    - title: 可自定义标题
    """
    # [修复] 强制指定 labels 索引范围，确保矩阵始终为 len(labels) x len(labels)
    label_indices = list(range(len(labels)))
    cm = confusion_matrix(y_true, y_pred, labels=label_indices)
    
    figsize = (7, 6) if len(labels) <= 2 else (11, 9)
    plt.figure(figsize=figsize)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                annot_kws={"size": 12 if len(labels) <= 2 else 9})
    
    plt.title(title)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'confusion_matrix.png'))
    plt.close()

def generate_markdown_report(history, y_true, y_pred, report_dir, target_names=['NonVPN', 'VPN'], stage_name="Stage 1"):
    """
    自动生成实验报告 Markdown 文件
    """
    # [修复] 显式指定 labels 索引，防止样本缺失导致的维度不匹配
    label_indices = list(range(len(target_names)))
    
    report_dict = classification_report(
        y_true, y_pred, 
        labels=label_indices,      # 核心修复点
        target_names=target_names, 
        output_dict=True, 
        zero_division=0
    )
    
    report_path = os.path.join(report_dir, "experiment_report.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# 🔬 流量识别实验报告 - Scenario A ({stage_name})\n\n")
        
        f.write("## 1. 实验环境与配置\n")
        f.write(f"- **任务**: {stage_name} 流量分类\n")
        f.write("- **架构**: 1D-CNN + Transformer 融合网络 (双轨特征)\n")
        f.write("- **流超时 (Timeout)**: 15 Seconds\n\n")
        
        f.write("## 2. 核心性能指标\n")
        f.write("| 指标 (Metric) | 数值 (Value) |\n")
        f.write("| :--- | :--- |\n")
        f.write(f"| **准确率 (Accuracy)** | {report_dict['accuracy']:.4f} |\n")
        f.write(f"| **宏平均 F1 (Macro F1)** | {report_dict['macro avg']['f1-score']:.4f} |\n")
        
        # 针对 Stage 1 的特殊显示
        if 'VPN' in target_names and len(target_names) == 2:
            f.write(f"| **VPN 识别率 (Recall)** | {report_dict['VPN']['recall']:.4f} |\n")
            
        f.write(f"| **最终验证集 Loss** | {history['val_loss'][-1]:.4f} |\n\n")
        
        # 多分类明细
        if len(target_names) > 2:
            f.write("## 3. 各类别性能明细\n")
            f.write("| 类别 | Precision | Recall | F1-score | Support |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- |\n")
            for name in target_names:
                m = report_dict[name]
                f.write(f"| {name} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1-score']:.4f} | {m['support']} |\n")
            f.write("\n")

        f.write("## 4. 详细分类评估文本\n")
        f.write("```text\n")
        # 此处也需同步显式指定 labels
        f.write(classification_report(y_true, y_pred, labels=label_indices, target_names=target_names, zero_division=0))
        f.write("\n```\n\n")
        
        f.write("## 5. 数据可视化\n")
        f.write("### 5.1 学习曲线\n")
        f.write("![Learning Curves](./learning_curves.png)\n\n")
        f.write("### 5.2 混淆矩阵\n")
        f.write("![Confusion Matrix](./confusion_matrix.png)\n")
        
    print(f"📄 实验报告 ({stage_name}) 已保存至: {report_path}")