# traffic_learn

基于 `ISCX_VPN-NonVPN` 数据集的流量识别实验工程。

当前工程包含三层标签设计：

- `label1`: `VPN` / `NonVPN`
- `label2`: 业务分类，如 `Chat`、`Streaming`、`File Transfer`
- `label3`: 应用分类，如 `skype`、`facebook`、`youtube`

模型结构目前使用 `1D-CNN + Transformer`，训练流程按阶段拆分：

- Stage 1: `label1` 二分类
- Stage 2: 在 `VPN` 或 `NonVPN` 分支内做 `label2` 业务分类
- Stage 3: 在 `label1 + label2` 分支内做 `label3` 应用分类

## Environment

建议使用 Python 3.11。

先升级 `pip`：

```bash
python3.11 -m pip install --upgrade pip
```

安装你指定的基础依赖：

```bash
pip install scikit-learn pandas matplotlib seaborn
```

这个工程实际运行还需要以下依赖：

```bash
pip install numpy torch scapy tqdm
```

如果需要从原始 pcap/pcapng 解析五元组应用流，还需要本机安装 `tshark` 并确保可从命令行直接调用。

## Project Layout

```text
data/
  app/                  tshark 解析出的每个 pcap 的流级 JSON
  app_label.json        流筛选文件
  label.json            文件级真值标签配置
  process/              预处理后的特征文件
scripts/
  main_preprocess.py    生成 data/process 和 samples.npz
  train_stage1.py       Stage 1: VPN / NonVPN
  train_stage2_nonvpn.py
                        Stage 2: 按 label1 做业务分类
  train_stage3.py       Stage 3: 按 label1 + label2 做应用分类
src/
  TrafficDataFactory.py 预处理与样本索引生成
  dataset.py            Stage1/2/3 数据集定义
  models.py             CNN + Transformer 模型
samples.npz             样本索引
```

## Data Pipeline

当前标签语义已经调整为：

- `data/label.json` 是文件级真值来源
- `data/app_label.json` 只用于流筛选，不再覆盖真值标签

如果你修改了流筛选或标签文件，建议重新生成样本：

```bash
python scripts/main_preprocess.py
```

如果只想修复现有 `samples.npz` 中的标签一致性：

```bash
python scripts/relabel_data.py
```

检查 `samples.npz` 与文件真值是否一致：

```bash
python scripts/test_label.py
```

## Training

### Stage 1

```bash
python scripts/train_stage1.py
```

### Stage 2

对 `NonVPN` 分支训练业务分类：

```bash
python scripts/train_stage2_nonvpn.py --label1 NonVPN
```

对 `VPN` 分支训练业务分类：

```bash
python scripts/train_stage2_nonvpn.py --label1 VPN
```

### Stage 3

训练全部可用的 `label1 + label2 -> label3` 专家模型：

```bash
python scripts/train_stage3.py
```

只训练某一个分支，例如 `NonVPN / Chat`：

```bash
python scripts/train_stage3.py --label1 NonVPN --label2 Chat
```

## Notes

- Stage 2 当前训练的是 `label2` 业务分类，不是应用分类。
- Stage 3 不是单一全局应用分类器，而是按 `label1 + label2` 分组训练专家模型。
- 如果原始数据或 `tshark` 识别结果存在噪声，Stage 3 的效果会直接受到影响。
