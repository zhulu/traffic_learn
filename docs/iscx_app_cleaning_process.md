# ISCX_VPN-NonVPN 应用协议清洗与五元组重标注记录

## 1. 任务背景

ISCX_VPN-NonVPN 数据集的官方 `label2` 并没有给出严格、稳定的五元组级标注。  
如果只依赖原始 pcap/pcapng 文件名进行分类，会把大量背景流量、系统流量、辅助流量一起误归入主业务类别，带来明显噪声。

因此，本轮清洗的目标是：

1. 先基于 `tshark` 解析得到的 `application` 信息，尽量在五元组级别做更细的判别。
2. 当 `application` 无法明确映射到目标类别时，再退回到 pcap 文件名进行兜底分类。
3. 清除明显不属于目标七类任务语义的协议噪声。
4. 将结果整理为“一个 pcap 文件下可能包含多个 label，每个 label 对应一组五元组”的结构。

目标类别沿用 `data/label.json` 中的二级类别定义：

- `Web Browsing`
- `Email`
- `Chat`
- `Streaming`
- `File Transfer`
- `VoIP`
- `P2P`

## 2. 原始输入

本轮处理涉及以下输入：

- `data/app/*.json`
  - 每个 pcap 对应一个 JSON 文件
  - 每个 JSON 内包含多个五元组条目
  - 每个条目至少包含：
    - `five_tuple`
    - `application`
    - `packet_count`
- `data/label.json`
  - 用于提供原始类别定义和文件名语义参考

示例条目结构如下：

```json
{
  "five_tuple": {
    "src": "131.202.240.150:49242",
    "dst": "216.58.219.238:80",
    "proto": "TCP"
  },
  "application": ["TCP", "OCSP"],
  "packet_count": 39
}
```

## 3. 中间产物与规则演进

### 3.1 `app_total.json`

首先对 `data/app` 下全部 JSON 做了文件级汇总：

- 读取每个 JSON 文件中的所有 `application`
- 去重后形成该 pcap 的 `app` 列表
- 用编号后的文件名部分作为 key
- 形成总协议列表 `application_total`

例如：

```json
"vpn_skype_chat1a": {
  "app": ["TCP", "TLSv1", "UDP", "DNS", "ICMP"]
}
```

这个阶段的目的主要是：

- 看每个 pcap 中大致出现了哪些协议
- 评估协议和文件名描述的一致性
- 为后续制定分类规则提供依据

### 3.2 第一版 `app_label.json`

之后基于文件级 `app` 并集进行了第一次分类尝试：

1. 优先按 `app` 判定类别
2. `app` 无法明确分类时，按文件名兜底
3. 剔除明显噪声协议

但这一版存在明显问题：

- 它是“文件级 app 并集判别”，不是“五元组级判别”
- 一个文件中只要出现少量 `STUN`、`RTCP`、`LSD` 等协议，整个文件就可能被整体改类
- 会把 `Streaming` 文件整体抢成 `VoIP`
- 会把 `File Transfer` 文件整体抢成 `P2P` 或 `VoIP`

因此这一版只作为规则探索，不作为最终结果。

### 3.3 最终版 `app_label.json`

最终决定回到五元组粒度重新处理：

1. 重新直接读取 `data/app/*.json`
2. 对每个五元组单独判断类别
3. 一个 pcap 文件中允许同时存在多个 label
4. 每个 label 下保存一组五元组

这也是当前最终采用的版本。

## 4. 最终分类规则

最终规则写在：

- `scripts/build_app_label.py`

### 4.1 总体优先级

分类优先级如下：

1. 优先看单个五元组的 `application`
2. 如果 `application` 能明确映射到唯一类别，则直接使用该类别
3. 如果 `application` 过弱、冲突、或无法唯一映射，则按 pcap 文件名兜底
4. 若五元组只包含被排除协议，直接丢弃

### 4.2 明确协议到类别的映射

当前五元组级显式映射如下：

- `Email`
  - `SMTP`, `SMTPS`, `POP3`, `POP3S`, `IMAP`, `IMAPS`
- `P2P`
  - `BitTorrent`, `LSD`, `NAT-PMP`
- `VoIP`
  - `STUN`, `DTLS`, `DTLSv1.0`, `RTCP`, `SIP`
- `File Transfer`
  - `SSH`, `SSHv2`, `FTP`
- `Chat`
  - `XMPP/XML`
- `Streaming`
  - `RTMP`

说明：

- 这些协议被认为具有较强的业务语义
- 只有当单个五元组匹配到唯一类别时，才直接采用 `app` 结果
- 如果同一个五元组同时匹配多个强类别，则视为冲突，退回文件名

### 4.3 文件名兜底规则

当单个五元组的 `application` 无法明确分类时，按 pcap 文件名进行兜底：

- 包含 `bittorrent` 或 `torrent` -> `P2P`
- 包含 `email` -> `Email`
- 包含 `scp`、`sftp`、`ftps`、`file` -> `File Transfer`
- 包含 `voipbuster`、`audio` -> `VoIP`
- 包含 `chat`、`aim`、`icq`、`gmailchat` -> `Chat`
- 包含 `video`、`netflix`、`youtube`、`vimeo`、`spotify` -> `Streaming`
- 包含 `firefox`、`chrome`、`browser`、`web` -> `Web Browsing`
- 若以上都不满足，则该五元组不进入最终结果

### 4.4 排除协议

以下协议不参与最终分类，视为背景噪声、系统流量或与目标 7 类语义弱相关协议：

- `DNS`
- `SMB`
- `SMB2`
- `NBNS`
- `LLMNR`
- `MDNS`
- `ICMP`
- `ICMPv6`
- `OCSP`
- `SSDP`
- `SNMP`
- `DHCP`
- `SRVLOC`
- `DB-LSP-DISC/JSON`
- `DCERPC`
- `DCP-AF`
- `DCP-PFT`
- `THRIFT`
- `R-GOOSE`
- `Pathport`
- `VNC`
- `WireGuard`
- `X11`
- 以及同类其他明显非目标业务协议

完整排除列表见：

- `scripts/build_app_label.py`

### 4.5 流量强度过滤

为减少极短流带来的噪声，增加了包数过滤条件：

- `packet_count <= 5` 的五元组直接剔除

这样可以去掉大量偶发握手流、广播流和边缘短流。

## 5. 最终输出格式

最终结果写入：

- `data/app_label.json`

结构为：

```json
{
  "summary": {
    "file_count": 140,
    "multi_label_file_count": 21,
    "selected_flow_count": 7725,
    "label_file_counts": {
      "Chat": 30,
      "Email": 6,
      "VoIP": 48,
      "Streaming": 35,
      "File Transfer": 38,
      "P2P": 4
    },
    "label_flow_counts": {
      "Chat": 672,
      "Email": 453,
      "VoIP": 2130,
      "Streaming": 1980,
      "File Transfer": 2238,
      "P2P": 252
    }
  },
  "files": {
    "facebook_video1a": {
      "Streaming": [
        {"five_tuple": {...}}
      ],
      "VoIP": [
        {"five_tuple": {...}}
      ]
    }
  }
}
```

也就是说：

- 最外层是 `summary` 和 `files`
- `files` 下面每个 key 是一个 pcap 文件 stem
- 每个 pcap 下可能同时存在多个 label
- 每个 label 下是一组五元组列表

## 6. 当前结果概览

当前版本 `data/app_label.json` 的统计如下：

- `file_count = 140`
- `multi_label_file_count = 21`
- `selected_flow_count = 7725`

按“包含该 label 的文件数”统计：

- `Chat = 30`
- `Email = 6`
- `VoIP = 48`
- `Streaming = 35`
- `File Transfer = 38`
- `P2P = 4`

按“保留下来的五元组数量”统计：

- `Chat = 672`
- `Email = 453`
- `VoIP = 2130`
- `Streaming = 1980`
- `File Transfer = 2238`
- `P2P = 252`

## 7. 典型多标签示例

以下 pcap 文件在最终结果中表现为多标签：

- `facebook_video1a`
  - `Streaming = 30`
  - `VoIP = 27`
- `hangouts_video1b`
  - `Streaming = 21`
  - `VoIP = 2`
- `skype_file3`
  - `File Transfer = 7`
  - `VoIP = 1`
- `ftps_down_1a`
  - `File Transfer = 120`
  - `P2P = 1`
- `voipbuster1b`
  - `VoIP = 64`
  - `P2P = 2`

这些例子说明：

- 单个 pcap 中确实可能混入多个业务子流
- 如果强行给整个 pcap 只保留一个标签，会丢失不少真实结构信息
- 五元组级分组比文件级单标签更适合后续进一步训练或分析

## 8. 当前实现文件

本轮最终相关文件如下：

- 原始五元组协议文件：`data/app/*.json`
- 文件级协议汇总：`data/app_total.json`
- 最终多标签五元组结果：`data/app_label.json`
- 构建脚本：`scripts/build_app_label.py`

## 9. 已知限制

当前版本仍然有一些限制，需要在后续使用中注意：

1. `application` 来自 `tshark` 协议识别，本身可能存在误识别。
2. 某些业务类别缺少足够强的显式协议特征，仍需依赖文件名兜底。
3. `Web Browsing` 在当前版本中几乎没有形成有效样本，说明该类在数据中较弱，或者当前规则对它较为保守。
4. 一些协议如 `GQUIC`、`HTTP/JSON`、`HTTP/XML`、`BROWSER` 目前没有被直接强判为某一类，而是更多依赖文件名语义。

## 10. 后续可选改进

后续如果继续优化，可以考虑以下方向：

1. 对 `GQUIC`、`HTTP/JSON`、`HTTP/XML`、`BROWSER` 做更细的上下文区分。
2. 在五元组之外，引入方向性、持续时间、包长分布等统计特征辅助判别。
3. 将当前 `app_label.json` 继续展开为扁平训练样本格式：
   - `pcap`
   - `five_tuple`
   - `label`
4. 对多标签文件做人工抽样审核，进一步修正规则。

