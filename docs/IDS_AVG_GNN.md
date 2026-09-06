# IDSAVG GNN —— 独立 per-gate ids_avg 预测模型（DelayGNN 复刻改造）

> 版本记录：**17.0.0（2026-09-04，本地受控）→ 17.0.1（2026-09-04，服务器全量基线）**。判定唯一尺度 = **per-gate 真实 `ids_avg` 预测准确率**（R²/Spearman），与 delay 无关。
> 状态：服务器全量基线（17.0.1）、容量+锥体 A2/B2（17.0.6/17.0.7）、**长程对照 B2long/B2v2long（17.0.10）均完成判定**，见 §4.4/§4.5。

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
- **服务器全量（17.0.1 已完成，见 §4.2）**：`_fit_idsavg_gnn_server.py`，`DATA_BATCHES` 默认 full+rest+m4（同 config；rest 自动认 `*_partN.parquet`），
  电路级切分，**test 按批次来源分桶**（full/rest/m4）专看 m4（V3.2 五形状）泛化。

## 4. 结果

### 4.1 本地受控（17.0.0 · 1500c 子集 · 已验证）

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

### 4.2 服务器全量（17.0.1 · full+rest+m4 全域 · 电路级切分 · 已验证）

`_fit_idsavg_gnn_server.py`，EPOCHS=45、NO_NOGRAPH=1（W 无边对照本地已定论，不重跑）；test 按批次来源分桶。

| test 分桶（未见电路） | n（row,gate 样本） | GNN R² | GBDT15 R² | Δ |
|---|---|---|---|---|
| batch_v2_full | 48,905 | **0.7131** | 0.6363 | +0.077 |
| batch_v2_rest | 386,194 | **0.6954** | 0.5492 | +0.146 |
| **batch_v2_m4（五形状）** | 125,143 | **0.6583** | **0.2391** | **+0.419** |

- 总体：GNN test R²=**0.7012** / Spearman 0.7105 / best_val 0.6959；GBDT15 整体 ≈0.49（test 样本量加权粗估，精确值分桶见表）。
- **m4 桶 = 关键读数**：未见电路（V3.2 五形状）上 GBDT15 均匀广播近似**近失效（0.24）**，结构 GNN 仍稳（0.66）→ **图结构跨形态泛化、广播特征近似不能，获全量证实**；serve 端真模型方向的支持度远强于本地子集（全域 +0.21 / m4 +0.42 vs 本地 +0.096）。
- ⚠ 全量 GNN test **0.7012 < 本地子集 0.7697**：分布更广更难 + 只 45ep（本地 80ep）+ 模型按子集取偏小（K3/h96）；best_val≈test **无过拟合、偏欠拟合** ⇒ 层数/宽度/EPOCHS/patience 放宽有实证依据，列为待办对照（未跑）。

### 4.3 服务器容量放宽 A/B 首轮 = 无效对照（17.0.5 · LR/schedule 错配 · 结论作废）

env 旋钮（K/HID/EMB/DROPOUT/LR/PATIENCE/CONE_FEAT）17.0.5 入 `_fit_idsavg_gnn_server.py`。首轮 A（容量放宽：K5/HID160/EMB32、EPOCHS220、PATIENCE40、**LR 3e-3 误保留**）/ B（同上 + `CONE_FEAT=1` 锥体/距离 3 通道），2026-09-04 服务器并行跑。

| run | best_val | test R² | Spearman | early stop | m4 桶 GNN |
|---|---|---|---|---|---|
| A 容量放宽 | 0.3571（@ep1） | 0.3612 | 0.684 | @ep45（pat40） | 0.6343 |
| B +锥体 | 0.4257（@ep5） | 0.4324 | 0.778 | @ep45（pat40） | **0.7579** |
| 17.0.1 基线 | 0.6959 | 0.7012 | 0.7105 | 45ep 自然 | 0.6583 |

**首轮无效**：A/B 均 train_loss 自 ep1 冻在 ~0.0005（train R²≈val≈0.36 = 冻结预测均值）、best_val 落在 ep1/5、patience40 掐死在 ep45——**双双低于 17.0.1 且 full/rest 桶败给 GBDT15**。归因 = **LR 3e-3 对 K5/H160/E32（参数 ~4-5×）过高 + cosine T_max=220 前 45ep 几乎不退火（ep45 时 LR 仍 ~2.4e-3）→ 优化停滞；patience 恰在退火生效前掐死**。**非容量方向被判死、非代码回归**（A/B 文件完整执行到出桶表；默认路径与 17.0.1 逐位等价已本地验）。**唯一正信号 = 坏训练下 B 的 m4 桶 0.7579 仍 > 基线 0.6583 / A 0.6343、Sp 0.778** → 锥体结构信息对 m4 泛化有真实增益，但被污染不能作结论。

**修正（A2/B2，再跑）**：LR→1e-3（≈AdamW 默认，~3× 降）、PATIENCE→0（纯 best-val 择优，回到 17.0.1 选优行为）、EPOCHS→80（cosine 前段即可退火）、保留 K5/H160/EMB32 ± `CONE_FEAT=1`；**不开 N_CAP**（该路径在服务器 3 并发下出现过停滞，探针作罢，与代码正确性无关）。判定：A2 vs 0.7012 = 容量效应；B2 vs A2 = 锥体效应（重点 m4 桶）。（干净裁决见 §4.4）

### 4.4 服务器 A2/B2 干净裁决（17.0.6 · 容量+锥体双生效 · 已收敛）

A2（K5/H160/EMB32、LR1e-3、80ep、PATIENCE=0）/ B2（同 + `CONE_FEAT=1`），全量 45,662 电路跑满 80ep，数据/切分/口径同 §4.2。**首轮失败（§4.3）已证纯 LR/schedule 错配，本次配置修正后干净复现**。

| run | best_val | test R² | Spearman | full 桶 | rest 桶 | m4 桶 |
|---|---|---|---|---|---|---|
| 17.0.1 基线 | 0.6959 | 0.7012 | 0.7105 | 0.7131 | 0.6954 | 0.6583 |
| **A2** 容量放宽 | 0.7358 | **0.7424** | 0.7385 | 0.7558 | 0.7376 | 0.7016 |
| **B2** 容量+锥体 | 0.7810 | **0.7866** | **0.8150** | 0.7580 | **0.7812** | **0.8076** |

- **容量效应 A2−基线 = test +0.041**：各桶均匀 +0.042~0.043；best_val@ep80、train 0.7428≈test 0.7424（无过拟合）→ **"全量偏欠拟合 → 参数放宽"假设实证**（17.0.1 的 §4.2 ⚠ 待办解除）。
- **锥体效应 B2−A2 = test +0.044 / Spearman +0.077**，且增益**集中在 rest(+0.044)/m4(+0.106)，full≈0(+0.002)** → 锥体/距离通道是**跨形态结构泛化**特征（下游 BFS、驱动/输出深度），非拟合容量；**m4 0.8076 为历史最高**（GBDT15 0.2391，差 **+0.568**），§4.3 首轮坏训练 B m4 0.7579 信号被干净复验并放大。
- best_val 恰落 ep80（仍微涨但 train≈test 已收敛）→ 80ep 足够，追长 epoch 边际很小。

**待定（方向性利好）**：serve 端若采用 B2 需 Rust 侧补锥体/距离特征（下游 BFS + 输出深度），成本待估（§13 冲突清单 O 系列可行性）；m4/rest 桶的大优 = serve 端真结构模型收益上限显著上调，是否推进另议。

### 4.5 长程对照 B2long / B2v2long（2026-09-05 启动 · 09-06 完成 · 已判定）

**动机**：§4.4 A2/B2 best_val 恰落 ep80、末 5ep 增益最大、train_loss 尾段仍升 → 追问"80ep 硬停是否截早"。回应：**不加新硬上限，而是取消硬停作裁决**——EPOCHS=220 仅作天花板、PATIENCE=60 由 val 平台自动早停决定收尾（round-1 的"patience 失败"实为 LR 3e-3 冻优化器，非 patience 本身）。
**配置**：沿用 A2/B2（K5/H160/E32/LR1e-3/DROPOUT0.25）+ NO_NOGRAPH=1 + 220ep 长程。B2long = `CONE_FEAT=1`（锥体 v1）；B2v2long = `CONE_V2=1`（17.0.8 锥体 v2 = v1 3 通道 + 主通路 d1∩d2 / 锥内扇出 / 锥内扇入，N_EXTRA 10→13）。同图同批次同切分 → 同口径对判。
**阻塞与根因修复（17.0.9）**：两 run 首启卡在数据切分 `set(ddf['circuit_id'])` 30min+（py-spy 定位：pandas **pyarrow-backed** 字符串列逐元素 `arrow.array.__iter__` 病态慢，746k 行；A2/B2 与历史 N_CAP 探针"停滞"同源，A2/B2 当年在此磨 ~30-45min 未被察觉）。修复 = 先 `ddf['circuit_id'].to_numpy()` C 速转 object 再建 set，语义逐位不变；重启后 2-3min 即越过装配进训练（不再 30min+）。教训入 §6。
**结果（220ep 均跑满；PATIENCE=60 全程未触发 → best_val 落最后一 ep）**：

| run | best_val | test R² | Spearman | full 桶 | rest 桶 | m4 桶 |
|---|---|---|---|---|---|---|
| B2（v1·80ep，§4.4） | 0.7810 | 0.7866 | 0.8150 | 0.7580 | 0.7812 | 0.8076 |
| B2long（v1·220ep） | 0.7833 | 0.7882 | 0.8166 | 0.7641 | 0.7820 | 0.8143 |
| **B2v2long（v2·220ep）** | 0.7841 | **0.7893** | 0.8168 | 0.7625 | 0.7826 | **0.8235** |

**判定① "80ep 截早?" → 基本否（收益噪声级）**：B2→B2long test **+0.0016** / best_val +0.0023 / m4 +0.0067。追长到 220ep 几乎无 gain → §4.4"80ep 足够、追长边际很小"获干净反证。**方法论教训**：长程 val 在 cosine 尾段（ep200-220：0.741→0.783）仍陡升 → PATIENCE 全程未触发、跑满 220；cosine 硬 T_max 尾段 val 一路爬、patience 等不到平台期，**EPOCHS 上限实际就是裁决者**。而 ep80→220 的 test 增益 <0.002 证明尾段爬升对 test 是"清噪声"非真信号 → **未来跑 80-100ep 即够，不必长程**。
**判定② 锥体 v1→v2（同 220ep 口径）= m4 特化 +0.009，总体≈0**：m4 .8143→**.8235**（历史最高，GBDT15 0.2391 差 **+0.584**）；rest +0.0006、full **−0.0016**（唯一退步桶）、overall test +0.0011。v2 的 3 通道（主通路 d1∩d2/锥内扇出/扇入）增益**全集中于 m4 五形状**（比 v1 的 rest/m4 结构泛化主题更窄）。**serve 建议：若采用锥体，v1（3 通道）已拿走全数据几乎全部价值；v2 额外 Rust 成本只换 m4 桶 +0.009，仅当 m4 形态鲁棒是 serve 目标才值得**。
**累计**：全量最佳 = **B2v2long test 0.7893 / m4 0.8235**（GBDT15 全域 ≈0.49）。R² 已到 0.79 高区；剩余可挖 = serve 端锥体可行化 + GBDT15 混合/多 seed 集成（需 ckpt 落盘），另议。

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
- ✅ **服务器全量（含 m4）基线已完成（17.0.1）**：判定 + 分桶见 §4.2——每桶 GNN 均胜，m4 桶 GBDT15 0.24 vs GNN 0.66。
- ✅ **参数放宽对照已定论（17.0.6 A2/B2，§4.4）**：容量 K5/H160/E32+LR1e-3+80ep = test +0.041（0.7012→0.7424）；**叠加锥体 = 0.7866、m4 桶 0.8076**。全量欠拟合假设证实（§4.2 ⚠ 解除）。**未决：serve 端若采用锥体需 Rust 补特征，成本/收益另议**（方向性利好）。
- ✅ **长程对照已判定（17.0.10，§4.5）**：80ep 只轻微截早（追 220ep test +0.0016、噪声级；patience 因 cosine 尾段 val 爬升不触发，跑满上限）；锥体 v1→v2 = m4 特化 +0.009（0.8143→0.8235 历史最高）、overall +0.001、full −0.002 → **serve 用 v1 即可，v2 仅 m4 鲁棒才值**。全量最佳 = B2v2long test 0.7893 / m4 0.8235。
- ⚠ **服务器 pandas pyarrow-backed 字符串列病态慢（17.0.9 定位修复）**：read_parquet 后 `circuit_id` 为 Arrow-backed string，`set()`/`list()` 逐元素迭代走 `arrow.array.__iter__`，746k 行实测 30min+（py-spy 实锤；A2/B2 与历史 N_CAP 探针停滞同源，均在此磨 30min+ 未被察觉）。任何脚本对同源 parquet 列做全量 Python 级迭代前，**先 `col.to_numpy()` C 速转 object 再做 set/list**。`_fit_idsavg_gnn_server.py` 已修（L104-105 `_cid_set`）。
- 结果以本文件 + PROJECT_LOG 17.0.1 为准；数据相关引用仍以 `docs/GNN_RUST_DATA_DIFF.md` §14 审计标注为基准。
