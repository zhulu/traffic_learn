# 🔬 流量识别实验报告 - Scenario A (Stage 1)

## 1. 实验环境与配置
- **任务**: 区分 VPN 与 Non-VPN 流量 [cite: 131, 135]
- **架构**: 1D-CNN + Transformer 融合网络
- **流超时 (Timeout)**: 15 Seconds [cite: 207, 209]

## 2. 核心性能指标
| 指标 (Metric) | 数值 (Value) |
| :--- | :--- |
| **准确率 (Accuracy)** | 0.9330 |
| **宏平均 F1 (Macro F1)** | 0.8965 |
| **VPN 识别率 (Recall)** | 0.7488 |
| **最终验证集 Loss** | 0.5207 |

## 3. 详细分类评估
```text
              precision    recall  f1-score   support

      NonVPN       0.93      0.99      0.96      1375
         VPN       0.94      0.75      0.83       402

    accuracy                           0.93      1777
   macro avg       0.94      0.87      0.90      1777
weighted avg       0.93      0.93      0.93      1777

```

## 4. 数据可视化
### 4.1 学习曲线
![Learning Curves](./learning_curves.png)

### 4.2 混淆矩阵
![Confusion Matrix](./confusion_matrix.png)
