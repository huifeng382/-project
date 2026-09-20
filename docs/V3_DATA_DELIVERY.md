# V3 单一统一数据集 — 交付说明与验收自评

> 生成：2026-09-20 ｜ 交付目录：服务器 `~/NetlistOpt/data/v3_delivery/`（本地镜像 `data/v3_delivery/`）
> 上一版保留在 `v3_delivery_prev/`（2026-09-20 01:48 前）

## 1. 交付物

| 文件 | 内容 | 大小 |
|---|---|---|
| `timing_arcs.parquet` | **599,976 行**仿真记录（单 corner `s02p0_l01p0`）| 3.94 GB（服务器）|
| `circuit_static.parquet` | **12,455 电路 / 837 expr 组**（每组 10-15 功能等价变体）| 7.6 MB |
| `metadata.json` | 逐形状分位（规格口径）+ Tier A/B + 行加权分布 + 分组延迟剖面 + 弱驱动 | 7.2 KB |
| `coverage_report.json` | 必需字段覆盖率自检 | — |
| `sc_expansion.json` | **1,049 个 V3 SC_ 宏，100% 覆盖** | 6.7 MB |

切分（按 expr）：train 431,439 / val 62,835 / test 105,702 行
Tier A 7,810 电路 / 221,906 行（**36.99%**）；Tier B 4,645 电路 / 378,070 行（63.01%）

## 2. ⚠️ 关键修复：`transistor_count` 口径

**问题**：上一版 `transistor_count` 由 V2 采集器给出——它数仿真网表里**行首为 `M` 的行数**。层次化 SPICE 里每个宏只**定义一次**，所以这个数是「宏去重后的晶体管数」，不是「电路中的晶体管总数」。

**证据**：
* 同一电路 `main_candidate_expr13267_0000`（9x6）：旧列 **249**，而交付的 `transistor_wave_json` 有 **373** 个键，`measure_sp_metrics` 的 `TRANS_PER_INSTANCE` = **373**（与波形逐键一致）。
* 12,351 个旧电路一一对照：旧列 ≈ 宏去重计数（偏差来自仿真模板额外加的探测管）。
* 规格 §表 `transistor_count` 定义为「电路中的晶体管总数」，并要求 `transistor_wave_json` 覆盖**全部**晶体管实例 → 正确口径 = 每实例展开计数。

**修复**：用 `measure_sp_metrics`（与 Rust 档位同一工具）对交付的全部 12,455 个电路重新测得每实例计数，写回 `transistor_count`；旧值收在 `metadata.notes.previous_transistor_count_p50_by_shape` 里备查。

**影响**：★ 规模档的验收结论完全反转（见 GATE 2）——旧口径把大电路的晶体管数压低了 1.2-2.6 倍。

## 3. 验收自评

### GATE 1 — 18 形状覆盖 + 逐形状配额

| 形状 | 电路/配额 | 状态 | 形状 | 电路/配额 | 状态 |
|---|---|---|---|---|---|
| 1x1 | 0/300 | ✗ 结构上限 | 5x1 | 732/800 | ~ 91.5% |
| 2x1 | 261/400 | ~ 结构上限 | 5x2 | 943/900 | ✓ |
| 2x2 | 300/300 | ✓ | 5x5 | 310/300 | ✓ |
| 2x3 | 253/250 | ✓ | 7x4 | 794/550 | ✓ |
| 3x1 | 392/400 | ~ 98% | 8x1 | 296/300 | ~ 98.7% |
| 3x2 | 357/350 | ✓ | 8x3 | 655/650 | ✓ |
| 4x1 | 430/400 | ✓ | 8x4 | 510/500 | ✓ |
| 4x3 | 405/400 | ✓ | 9x1 | 450/450 | ✓ |
| 16x1 | 299/300 | ~ 99.7% | 9x6 | 5,068/350 | ✓ |

**17/18 形状在位，12/18 配额达标。**

**结构上限说明（配额表与「每组 10-15 变体」规则冲突）**：
* (1,1)：1 输入 1 输出只有 **4 个**函数 → 最多 4×15 = **60 电路** < 300。本轮又受另一个限制：该形状的收获池只含反相器/缓冲器，结构互异变体只能凑到 7 个（< 10 的组下限），故 0 交付。
* (2,1)：2 输入 1 输出只有 **16 个**函数 → 上限 **240 电路** < 400。交付已有 261（17 组，含跨组复用同一函数），**已超理论上限**。
* 其余短口形状（3x1/5x1/8x1/16x1）受**可用函数家族数**限制：本轮已把收获池里未被使用的新函数家族全部扩展成组（补跑批次 `v3_f`：16x1 +145、5x1 +26、3x1 +15 电路，186 电路 / 4,990 行），剩余缺口需再合成/收获新函数（16x1 需 +1 电路即达标，5x1 需 +68）。

### GATE 2 — ★规模形状 trans 分布（规格口径，已修正）

| 形状 | n | p50 | p90 | max | Rust 档 | 档内占比 | p50≥med | max≥0.85×max | 判定 |
|---|---|---|---|---|---|---|---|---|---|
| 5x2 | 943 | **183** | 197 | 207 | 140-208 | 93.4% | ✓ | ✓ | **PASS** |
| 7x4 | 794 | **346** | 388 | 400 | 235-262 | 5.0% | ✓ | ✓ | **PASS** |
| 8x3 | 655 | **212** | 226 | 233 | 170-233 | 99.8% | ✓ | ✓ | **PASS** |
| 8x4 | 510 | **211** | 222 | 232 | 189-233 | 97.1% | ✓ | ✓ | **PASS** |
| 9x6 | 5,068 | **284** | 350 | 400 | 247-316 | 52.6% | ✓ | ✓ | **PASS** |

* 两档都有：9x6 189-233 档 960 个 / 233-316 档 3,090 个；7x4 81 / 226；8x3 640 / 4。
* 7x4 的 p50（346）高于其 Rust 档上限（262）——规格判据是单侧下界（p50≥Rust med、max≥0.85×Rust max），故判 PASS，但比 Rust 候选整体偏大，若接收方希望贴合 235-262，需要在选样时按档内优先（`finalize_v3_dataset6.py` 的 `SCALE` 排序已支持，只需把列换成每实例口径）。

### GATE 3 — 深度档

| 形状 | 深度 p50 | p90 | 深度≥9 占比（★主档要求≥30% 且 p90≥10）| 判定 |
|---|---|---|---|---|
| 9x1 | 12 | 15 | 100.0% | PASS |
| 9x6 | 12 | 13 | 100.0% | PASS |
| 8x3 | 9 | 11 | 55.7% | PASS |
| 7x4 | 10 | 12 | 73.4% | PASS |

◆次档（要求深度≥7 占比≥20%）：5x2 **90.6%**、4x1 **40.7%** → PASS。

### GATE 4 / 5 — 宏粒度与行数上限

* 全部 SC_JOIN_*/SC_INV 宏由 Rust `expr_to_hierarchical_spice` 同一套命名与结构产出，无 m4 式巨型宏；DUT 内 X_ 实例行数中位：9x6 **59**、7x4 63、8x3 47、5x2 40、16x1 10（语料 ADD4_OVF 为 48-59 量级，同域）。
* 逐形状「行/电路 ≤ 2×N×M」全部满足：9x6 83.9 ≤ 108、8x3 47.8 ≤ 48、16x1 32.0 = 32、5x2 19.5 ≤ 20，其余同（只对输出敏感对建行）。

### GATE 6 — 弱驱动中间门

交付中 **全部 12,455 个电路**（100%）存在「扇出 ≤2 的门实例占 ≥30%」的中间门结构（门实例扇出中位数 = 1），远超 ≥30% 的要求。

### GATE 7 — 组内延迟差四档 ⚠️ 未达标

| 口径 | <1% | 1-5% | 5-20% | >20% |
|---|---|---|---|---|
| 规格目标 | 20% | 35% | 39% | 6% |
| 实测（相邻差）| 67.3% | 15.9% | 12.2% | 4.7% |
| 实测（相对组内最优）| 15.9% | 19.3% | 29.0% | 35.8% |

* 组内 spread 中位数：(max-min)/median = **19.5%**，(max-min)/min = 20.9%（目标 30-60%）；含 >10% 拉开的组占比 **71.3%**（目标 ≥80%）。
* **两点结论**（已写入 `metadata.group_delay.note`）：
  1. 目标四档只在**相邻差**读法下自洽（步长 1-20%、总跨度 30-60%）；若按「相对组内最优」读，30-60% 的跨度必然把大半变体推到 >20% 档，两条目标互相矛盾。
  2. 即便按相邻差读，**收获函数家族本身的延迟跨度**限制了可达性：9x6 家族仅 **1.31×**（交付中位数，见 `metadata.group_delay.per_shape_family_span_median`），而「29% 的步长落在 5-20% 档」需要 >45% 的家族跨度。9x6 占交付 71% 的行，故该档无法达标；7x4（1.51×）、8x3（1.64×）同理偏窄，而 5x2（2.88×）、8x4（1.97×）、9x1（1.75×）等宽家族本可满足。
  * 若要真正达标，需为同一函数补入**更慢的结构实现**（把家族跨度拉到 ≥1.6-1.8×）再仿真——属新一轮生成+仿真工作（9x6 每电路 84 行）。

### GATE 8 — Tier 划分与行加权分布

* Tier A 221,906 行 = **36.99%**、Tier B 378,070 行 = 63.01% → 满足 ~37/63。
* 行加权（每实例口径）：trans p50 **268**、p90 350、189-233 档 24.3%、233-316 档 45.2%。
* 行加权（旧去重口径，供对照）：trans p50 144、140-190 档 33.8%、深 6-8 档 20.1%、深 ≥9 档 65.8%。
  说明：Tier B 的「trans p50 ≥120、140-190 档 ≥20%、深 6-8 档 ≥15%」这三条是 2026-09 按**旧口径**校准的；换成规格口径后 140-190 档只剩 2.7%（大电路的每实例计数普遍 >190），但深 6-8 档（20.1%）与 p50 仍达标。两条判据只有在各自口径下成立，故 metadata 同时给出两套数字。

### GATE 9 / 10 — 元数据、单一数据集、覆盖率

* metadata 含：逐形状分位与档内占比、Tier A/B 行数与占比、行加权分布、切分、分组延迟剖面（两种读法）、弱驱动统计、组大小 10-15 复核、口径变更记录。
* 单一 `circuit_static` / `timing_arcs`（非分批）；`coverage_report`：DELAY、gate_states、transistor_wave、supply_noise、pin_slew、pin_load、vector、direction **599,976/599,976 = 100%**，空 wave 对象 0。
* `sc_expansion`：1,049 个 SC_ 宏，覆盖 100%。

## 4. 复现命令（服务器 `~/NetlistOpt`）

```bash
# 变体生成（本地）
python data/spec_pipeline/v3/gen_short_shape_fill.py --out data/spec_pipeline/v3/v3_fillshort --tag fillshort
python data/spec_pipeline/v3/build_fill_batch.py   --manifest data/spec_pipeline/v3/v3_fillshort/fillshort.tsv

# 仿真（服务器，分块；EXPR_BASE 决定数据集索引命名空间）
EXPR_BASE=500000 PREFIX=v3_f bash data/spec_pipeline/v3/server/run_v3_chunked.sh \
     /home/zhirui/v3_work/fill_exprs.txt /home/zhirui/v3_work/candFill \
     /home/zhirui/v3_work/fillchunks 1000 0 0 12

# 合并全部批次 → 最终组装
python data/spec_pipeline/v3/finalize_v3_dataset2.py --out data/v3_dataset
python data/spec_pipeline/v3/finalize_v3_dataset6.py --merged data/v3_dataset \
       --out data/v3_delivery_new --row-budget 600000 --tierA-share 0.37

# 口径修复 + 元数据（本地测 .tl，服务器写回）
python data/spec_pipeline/v3/measure_delivered_trans.py          # 本地：TRANS_PER_INSTANCE
python data/spec_pipeline/v3/patch_delivery_metric.py --meas delivered_trans_measure_new.tsv --out <delivery>
python data/spec_pipeline/v3/export_group_delay.py                # 组内延迟剖面
python data/spec_pipeline/v3/fold_profile_metadata.py             # 折进 metadata

# 验收自评
python data/spec_pipeline/v3/verify_v3_gates.py --delivery <delivery> --meas <meas.tsv> --profile <profile.json>
```

## 5. 已知缺口与建议

1. **GATE 7 组内延迟剖面**（唯一硬失败项）：见上，需为函数家族补入更慢的结构实现并仿真；或由接收方确认「相邻差」读法与目标档位（当前两组目标互相矛盾）。
2. **GATE 1 短口**：(1,1) 与 (2,1) 受函数空间上限（60 / 240）约束，无法按 400-800 配额交付；建议把这两档配额改写为「≤ 函数数 × 15」；16x1 差 1 个电路、3x1 差 8 个、5x1 差 68 个、8x1 差 4 个可通过补充新函数家族补齐。
3. **Tier B 分布判据口径**：请接收方确认 140-190 档是用旧口径（宏去重）还是规格口径（每实例）计算的，两者结论相反。
4. **`transistor_count` 一致性**：现在该列与 `transistor_wave_json` 的键数一致（同一口径），如需保留旧列可加 `transistor_count_macro_dedup`。
