# IDSAVG GNN —— 独立 per-gate ids_avg 预测模型（DelayGNN 复刻改造）

> 版本记录首条：**17.0.0（2026-09-04）**。判定唯一尺度 = **per-gate 真实 `ids_avg` 预测准确率**（R²/Spearman），与 delay 无关。
> 状态：本地受控对比**已验证**；服务器全量（含 m4）测试**待跑**。

## 1. 目的与定位

项目的 delay 粗筛模型预测的是"路径/门延迟"；serve 端还有一路 GBDT15/线性 `ids_avg` 近似作粗筛候选预打分。
本文件记录一个**独立建模方向**：能否用 GNN 直接在**电路图结构**上预测**每个门**的真实 `ids_avg`
（对 serve 是零仿真可算的物理量，与 per-gate delay 类似但按门计），超越当前近似的准确率天花板。

对比基准（历史实测，电路级切分）：
- 线性近似 per-gate `ids_avg` R² ≈ 0.655
- GBDT15（15 特征，部署同款）≈ 0.68（本 17.0.0 复现 = **0.6740**）

## 2. 架构 = DelayGNN 骨架复刻 + 刺激源锚定

不 import `pyg`、不改 `src/model.py`（生产 delayGNN 原样保留）；用**纯 torch** 复刻 `torch_geometric.GraphConv`：

```
x_i' = W1·x_i + W2·Σ_{j→i} x_j          # 有向 driver→receiver 边; index_add_ 实现
```

- **gate 类型 Embedding**(type_idx→16) concat 连续特征 → K=3 层 GraphConv；每层 LayerNorm+ReLU+Dropout，第 0 层后残差。
- **连续特征** = 静态 7 列 z-score + 7 行条件通道：slew/load 广播、corner slew/load、direction code、**src_anch**、**out_anch**。
- **关键新机制（刺激源锚定）**：行条件 `slew_s` **只**落在 `switching_pin` 节点（`src_anch`），`output_load_f` **只**落在 output 节点（`out_anch`），
  沿有向边传播 ⇒ 模型能感知"刺激源在哪、该门是否在 fanout cone 内、离源多远"。
  这是 GBDT15（slew/load 对**所有**门均匀广播）**无法表达**的结构信号——本次实验要验证的正是它。
- **读出/监督**：每节点单值头 → `log1p(mean 真实 ids_avg)`；只监督 wave 里 real>0 的门（与 GBDT15 实筛同口径）。
- 图：`build_static_graph`（I/O pin 也是节点；门名大小写归一）。

## 3. 数据与评估协议

- **本地受控**：`batch_v2_full` 采样 1500 电路（RandomState(42)），**电路级切分** 1200/150/150（RandomState(7)；
  test = 训练从未见过的电路，serve 真实口径）。80 epochs，AdamW lr 3e-3 wd 1e-4，cosine，best-val 选优。
- **对照同图**：A GBDT15（15 特征部署同款）；V gnn（有向边）；W nograph（同特征无边 = per-node MLP，**隔离"消息传递"本身**）。
- **服务器全量（待跑）**：`_fit_idsavg_gnn_server.py`，`DATA_BATCHES` 默认 full+rest+m4（同 config；rest 自动认 `*_partN.parquet`），
  电路级切分，**test 按批次来源分桶**（full/rest/m4）专看 m4（V3.2 五形状）泛化。

## 4. 结果（本地受控，已验证）

| 模型 | test R²（未见电路） | Spearman | 说明 |
|---|---|---|---|
| A GBDT15（15 特征，部署同款） | **0.6740** | 0.621 | 精确复现既有近似天花板 |
| V gnn（DelayGNN 复刻+锚定+有向边） | **0.7697** | 0.790 | train 0.7777≈test → 几乎不过拟合 |
| W nograph（同特征无边 MLP） | 0.6773 | 0.603 | ≈GBDT15（同"广播特征"天花板） |

**归因**：
- 消息传递自身贡献 = **+0.092** R²（0.6773→0.7697，特征/模型/数据逐项一致，只差"有没有边"）。
- gnn 超 GBDT15 = **+0.096**；Spearman 0.79 亦显著更高。
- ⇒ 有向图传播 + 刺激源锚定携带 GBDT15 均匀广播**看不见**的 per-gate ids_avg 信号 → **方向成立，值得做 serve 端真模型**。
- W ≈ A 印证"广播特征天花板 ~0.67"，只有图结构能突破。

## 5. 文件与复现

| 文件 | 用途 |
|---|---|
| `scripts/diag/_fit_idsavg_gnn.py` | 本地受控基准复现（1500c·80ep，A/V/W 三项） |
| `scripts/diag/_fit_idsavg_gnn_server.py` | 服务器全量（full+rest+m4，test 按批次分桶）；`N_CAP`/`EPOCHS`/`NO_NOGRAPH` 可调 |

复现（本地或服务器 ~/-project）：
```
DATA_BATCHES='batch_v2_full,batch_v2_rest,batch_v2_m4' OMP_NUM_THREADS=6 python scripts/diag/_fit_idsavg_gnn_server.py
```

## 6. 已知注意 / 待办

- ⚠ **GSZ>1 拼接提速为实验性**：block-diagonal 拼接前向与逐电路数值等价（实测差 ~3e-7），但 AdamW 训练大拼接不稳（val 崩）；**默认 GSZ=1 逐电路，勿开 GSZ>1**。
- 本地无 `pyg`（import 崩）→ 纯 torch 复刻即为此；服务器若可用 pyg 亦无需换。
- **待办 = 服务器全量（含 m4）测试**：跑 `_fit_idsavg_gnn_server.py` 后，把 A/V/W 判定 + 分桶（尤其 m4 桶 gnn vs GBDT15）贴回，据此看 m4 未见形态泛化再定 serve 端是否推进。
- 结果以本文件 + PROJECT_LOG 17.0.0 为准；数据相关引用仍以 `docs/GNN_RUST_DATA_DIFF.md` §14 审计标注为基准。
