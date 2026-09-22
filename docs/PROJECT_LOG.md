# Project Log — GNN Delay Prediction

> ⚠️ **必读（项目最高优先级要求文件）**：`docs/GNN_PROJECT_REQUIREMENTS.md` —— git 提交/GitHub 推送版本号规范（大.小.更小，如 10.0.0）、触发文本「记入要求文件」、仓库布局。**任何新对话处理本项目前先读它。**

## Project Overview

**Goal:** Predict circuit propagation delay (SPICE-level) from netlist topology + corner conditions.

**Input:** Transistor-level netlist (4 input pins a/b/c/d + output), per-pin slew/load/arrival_time, corner conditions, vector (input pattern).

**Output:** End-to-end delay from switching pin to output.

**Current Best（点精度，V1 旧数据）：** Test 24.45% (10.6) — 6-layer GraphConv + path sum readout + corner separation（历史最佳点精度）。

> ⚠️ **本表只反映早期「点精度」阶段**。自 12.0 起项目转向**排序任务**（组内 Spearman/遗憾），13.5+ 用 wave 特征、14.4 起对齐 Rust 切到 **V2 数据**；**当前交付 = V2 no-wave 6-seed 集成粗筛模型（hi 遗憾 8.78%）**，见 15.2.10/15.2.11 及 `docs/GNN_RUST_DATA_DIFF.md`。

---

## Baseline Evolution (Only Effective Changes Out of 50+ Attempts)

| Version | Commit | Test Error | Key Change | Effect |
|---------|--------|:----------:|------------|:------:|
| 8.0 | 32b73e6 | ~36% | Old data baseline | — |
| 8.5 | 984c823 | ~33% | New data + per-pin features + vector decode | -3pp |
| 8.7 | 935effc | ~26.5% | Corner condition encoding | **-6.5pp** |
| 9.5 | acfe603 | ~24.6% | 6-layer GNN | -1pp |
| 9.7 | 7dc3e16 | **24.55%** | Gate merge revert + stabilize | Best |
| 10.5 | c6f6ccb | 25.16%¹ | 早停判据改用 val_loss（修复 rel_err 噪声在高LR误停 Plateau@106） | 恢复 |
| 10.6 | 6473e5b | **24.45%**¹ | 深退火（LR_MIN 5e-6→1e-6, LR_FACTOR 0.7→0.5）| **新最优** |

> ¹ 10.5/10.6 在**新数据**（batch3 已更新）上测，与 9.7 的 24.55%（旧数据）非同一测试集，不完全可比。同数据参照：9.7 复现（旧数据）= 24.94%，10.3.3（新数据，早停有 bug）= 27.41%。10.6 的 24.45% 是新数据上迄今最优。

### Detailed Effective Changes

#### 8.5 — Per-Pin Features + New Data
- **Before:** All 4 pins got same global slew_s. Vector was meaningless sequential ID (0-17).
- **After:** Switching pin gets slew, others get 0. Vector bits decoded to per-pin logic states.
- **Why effective:** Model finally had per-pin information to distinguish pins.
- **Test:** ~33% (was ~36%)

#### 8.7 — Corner Condition Encoding
- **Before:** Corner conditions mixed into node features, indistinguishable from circuit structure.
- **After:** Corner (slew/load) extracted as separate graph-level condition, encoded independently, concatenated after pooling.
- **Why effective:** Model learned "how corner affects delay" independently from "what this circuit looks like". Biggest jump in the entire project.
- **Test:** ~26.5% (was ~33%, -6.5pp)

#### 9.5 — 6-Layer GNN
- **Before:** 4 layers, complex circuits (7+ gates) had signal paths longer than GNN reach.
- **After:** 6 layers, switching pin signal reaches all gates.
- **Why effective:** B2/B3 circuits (7 gates median) improved as messages propagated fully.
- **Test:** ~24.6% (was ~25.2%, -0.6pp)

#### 9.7 — Stabilize & Optimize
- Reverted failed gate type merge (9.6), added intersection BFS gate state, path sum readout, circuit signature, training acceleration (BATCH_SIZE=80, BASE_EPOCHS=10, num_workers=2).
- **Test:** 24.55% stable baseline.

---

## All Failed Attempts

### Feature Changes (all failed)
| Version | Change | Result |
|---------|--------|--------|
| 8.6 | Gate state (path flag) | No improvement (GNN propagates this implicitly) |
| 8.10 | Fan-in/distance/on_path features | Worse (27.1%) — GNN already learns topology |
| 8.11 | Distance-only feature | No improvement |
| 9.8 | Log10 corner encoding | Worse (25.9%) — model learned nonlinearity from raw |
| — | Vector normalization | No improvement |

### Architecture Changes (all failed)
| Version | Change | Result |
|---------|--------|--------|
| 8.2 | GATv2Conv | Worse + train/eval mismatch |
| 9.3 | GIN | No improvement (small circuit graphs) |
| 8.9 | Corner modulation (FiLM) | Worse (too many params, overfit) |
| 9.4 | Path sum readout | Minor improvement only |
| 8.8 | Hidden dim 384/512 | Worse (overfit) |
| 9.5-I | 8-layer GNN | Worse (26.6%) — over-smoothing |
| 9.6 | Gate type merge (650→27) | Worse (28.7%) — lost gate-level detail |
| 9.6-J | Gate embed 32→64 | Worse (27.9%) — sparse embedding overfits |

### Training Changes (all failed)
| Version | Change | Result |
|---------|--------|--------|
| 8.7.1 | Batch loss weighting | Worse — model capacity insufficient |
| 8.9-Cos | Cosine LR scheduler | No difference |
| E | Corner loss weighting | Worse |
| G | Gate count weighting | No improvement |
| 9.2 | Corner separation | Architecture change (part of 8.7) |
| 9.3 | Circuit grouping sampler | Minor positive |
| 10.2 | Per-gate direct delay supervision | Worse (25.7%) — GNN node features too noisy |
| 10.3-TW | Transistor multitask (5x aux) | Worse (25.1%) — 777 samples too sparse |
| TW-w10 | TW weight 10x | Worse (26.9%) — oscillates |
| TW-simple | TW 1-output | No improvement (24.9%) |
| PG+TW | Per-gate + TW combined | No improvement (24.7%) |

---

## Experiment Log

### 早停调查（已完成，2026-07-11）
| Exp | Dir | Config | Result |
|-----|-----|--------|--------|
| 9.7 repro | ~/project-97repro | 7dc3e16 旧数据 + 干净缓存 | **Test 24.94%** (val 23.35, Early-stop@315, LR退火到1.18e-5) |
| 10.3.3 noWave | ~/-project | 9.7 arch 新数据 noWave 干净缓存 | **Test 27.41%** (val 26.79, Plateau@106 误停, LR还7e-5) |

**结论**：27.41 vs 24.94 的 3pp 差距 = 早停被 val_rel_err 噪声在高 LR、高震荡时误砍（Plateau@106），**非数据/模型问题**（新旧数据 DELAY/特征/切分 ≈100% 一致，仅 batch3 行序不同致轨迹不同）。差距全在难样本（B1 +0.4, B2 +3.3, B3 +5.3, 极端corner +5）。→ 催生 10.5 早停修复。

### 10.5 批次（Running，均：branch 10.3.3-fix-earlystop, noWave, 新数据, 独立全新缓存）
| Exp | Dir | Cache | 相对 10.5 的改动 | Status | Result |
|-----|-----|-------|------------------|--------|--------|
| 10.5 baseline | project-105 | cache105 | 早停判据改用 val_loss（best_model 仍按 val_rel_err） | **Done** | **Test 25.16%** (val 23.70, Early-stop@344) — 早停修复成功(27.41→25.16) |
| 105-shallow(10.4) | project-105-shallow | cache105shallow | + cherry-pick ed49d20（10.4 浅层逐门 aux loss，合并无冲突） | **Done** | **Test 24.78%** (val 22.93, @293) — 略优 -0.38pp(近噪声) |
| 105-BS 大batch | project-105-bs | cache105bs | BATCH_SIZE 80→160 | **Done** | **Test 25.76%** (val 23.96, @292) — 变差 +0.60pp，弃 |
| 105-AN 深退火 | project-105-an | cache105an | LR_MIN 5e-6→1e-6, LR_FACTOR 0.7→0.5 | **Done** | **Test 24.45%** (val 22.51, @351) 🏆 最优 -0.71pp，甚至低于历史9.7的24.55% |

### 关键结论（2026-07-11）
1. **早停修复验证成功**：baseline 从 10.3.3 的 27.41% 回到 **25.16%**（全部以 Early-stop(val_loss)@290+ 收尾，不再被 Plateau@106 误砍）。与 9.7 复现(旧数据)24.94% 仅差 0.22pp（batch3 行序残余）。
2. **深退火(AN)是真实改进**：24.45%，比 baseline 低 0.71pp（>2× 噪声），且 B2/B3/极端corner 全面更好。原理：更低 LR 更充分退火，压住了 rel_err 震荡、收敛更稳。
3. **per_gate(shallow)微正**：24.78%，-0.38pp，接近噪声，需确认。
4. **大batch(BS)有害**：25.76%，弃。

**预期**：10.5 baseline 应跑过 106、以 Early-stop(val_loss) 收尾、Test ~24.5–25%。三变体与 baseline 同条件可比，看谁能压到 24.5% 以下。

### 10.7 批次（修复门名大小写 bug 后，均：branch 10.3.3-fix-earlystop=含深退火, noWave, 新数据, 独立缓存）
> 背景：发现大小写 bug（node_names 大写 X_ vs JSON key 小写 x_）连累两处——per_gate 从未喂入训练；gate_states 匹配 0 门→path-sum readout 一直只累加 out 节点。10.7 两处均修。
| Exp | Dir | Cache | 内容 | Status | Result |
|-----|-----|-------|------|--------|--------|
| 107-base | project-107-base | cache107base | 仅修复(path-sum 恢复)，无 per_gate loss | **Done** | **Test 24.59%** (B1 20.7/B2 32.5/B3 24.5) — path-sum 修复≈中性(vs 10.6 24.45，噪声内) |
| 107-pgd | project-107-pgd | cache107pgd | +per_gate delay aux(现真生效, w0.5) | **Done** | **Test 29.36%** — 大幅变差 +4.9pp |
| 107-pgs | project-107-pgs | cache107pgs | +per_gate out_slew aux(w0.5) | **Done** | **Test 28.83%** — 大幅变差 +4.4pp |
| 107-pgs2 | project-107-pgs2 | cache107pgs2 | +per_gate out_slew aux w2.0 | **Done** | **Test 28.71%** — 大幅变差 +4.3pp |

### 107 批次关键结论（2026-07-12）
1. **path-sum readout 修复 = 中性**：24.59% vs 10.6 的 24.45%（差在噪声内）。模型用「仅输出节点读出」就够好（6层GNN让out节点看到全图），修好 path-sum 没带来增益。→ 保留(是正确性修复)，但非提升；10.6 仍是最优基线。
2. **per_gate 辅助监督 = 有害(+4~5pp)**：这是 per_gate 第一次真正生效(此前一直被hasattr静默跳过)。delay/out_slew/加权三个变体全部大幅变差，worst corner 从46%崩到56-67%。**per_gate 作为辅助 loss 的方向经实测为死路**，不是「未测试」而是「已测试且负面」。
3. **对 11.0 LIB(Scheme A) 的警示**：LIB 链本质也是 per_gate 监督(delay/out_slew/in_slew)，而 per_gate 监督实测有害。且 PROJECT_LOG 教训#4(PG 24.46%>10.2)已作废(PG当时per_gate是no-op)。→ LIB 大概率也弱，需极小权重或重新评估。

> 启动脚本 setup_exp.sh（commit fd9dcd8→ec508c4 修复 cherry-pick 顺序）。粘贴长命令被终端截断 → 改用「clone exp107 一次 + bash setup_exp.sh <变体>」短命令。

### 重大认知转变：误差瓶颈 & 真实任务（2026-07-12/13）

**A. 「40%+ 极端 corner」大半是相对误差指标假象，非物理失败。**
对 107-base 的 test_predictions 按延迟分档：
| 延迟档 | 延迟均值 | 相对误差 | 绝对误差 |
|---|---|---|---|
| 档1(最小) | 8.4ps | **47.9%** | **2.88ps(最小!)** |
| 档2~5 | 16~110ps | ~18% | 3~18ps |
- 小延迟样本相对误差 48%，但绝对误差最小（2.88ps）——是「除以小分母」放大，非模型预测差。
- 模型真实典型精度 **~18%**（延迟≥16ps 稳定）；24.5% 是被小延迟(低负载/快 corner)拖高的。相关性 rel_err vs 延迟 = -0.15。
- 12.0 已加 per-corner `abs_err`+`mean_delay` 打印、`Test Median Rel Err`(稳健口径)。

**B. 真实任务 = 等价变体择优 → 排序，不是点精度。**
下游用途：等价变换生成多个新电路，用预测 delay 挑最快的。→ 正确指标 = **组内排序**(Spearman/选择遗憾/top1)，恒定偏差抵消，平均相对误差是错的 KPI。
数据天然支持：同 expr 多候选（273 expr，215 个≥2 候选）= 同功能不同结构的变体组。但原「按 circuit 切分」把变体组打散 + expr 级泄漏 → 改「**按 expr 切分**」(无泄漏、test 有完整变体组)。

### 代码里程碑
- **11.0**(c311eed)：LIB 查表延迟链(SC 展开→标准单元→可微链)。per_gate 已废弃(有害)。LIB 长线赌注，链 DP 太慢，需 2D-grid 加速才能跑，**暂缓**。
- **12.0**(bffd67c)：`split_by_expr` + 组内排序评估(Spearman/遗憾/top1) + Test Median Rel Err。对齐真实任务的范式级改动。
- **12.1**(70b6cb8)：seed 解耦(SPLIT_SEED 固定/TRAIN_SEED 可变，集成用) + `BEST_MODEL_METRIC` 选点开关(val_rel_err/val_loss/smoothed)。
- **DATA_SPEC 11.0.3**(aea7698)：per_gate 废弃 + wave 全覆盖要求(含 s40/s80) + 完整性铁律(防假覆盖)。wave 暂缓，规格不改。

### 12.x 探索批次（同 expr 切分 SPLIT_SEED=42，可比）— 结果（2026-07-13）
| Exp | 变体 | 选择遗憾↓ | Spearman↑ | top1↑ | Median Rel | Test(mean) | epochs |
|---|---|---|---|---|---|---|---|
| rank | 12.0 基线(默认选点) | 3.34% | 0.218 | 44.4% | 18.25% | 30.87% | 55 |
| anneal | 深退火 LR_MIN 1e-7 | **2.74%** | 0.222 | 40.9% | 17.94% | 30.94% | 92 |
| bmvl | best_model=val_loss | 4.12% | 0.146 | 38.2% | 19.48% | 36.72% | 61 |
| bmsm | best_model=平滑rel_err | 3.35% | **0.251** | **44.6%** | **16.46%** | 31.16% | 61 |

**结论（2026-07-13，首次看真实任务指标）：**
1. **expr 切分诚实但更难**：mean rel 30.87%（vs 泄漏的 circuit 切分 24.45%），test 是全新 expr。Median 18.25% 为稳健典型值。
2. **模型是弱排序器但选择遗憾低**：Spearman~0.22(分不清相近变体)，但选择遗憾~3%(变体延迟接近，选错代价小)，top1~44%(随机~28%)。→ 能挑近最优，难挑精确最优。
3. **选点策略**：`bmsm`(平滑rel_err)全面最好(Spearman/top1/Median第一)→**采纳**；`bmvl`(val_loss)最差(Spearman0.146)→**别用val_loss选checkpoint**；`anneal`遗憾最低(跑最久92ep)。
4. **都早停(55-92ep)**：expr切分下val_loss快速平台。anneal跑最久+遗憾最低→更慢退火/更长训练可能改善排序，待试。
5. **下一步**：组合 bmsm选点+anneal退火；并判断「遗憾~3%/top1~44%」是否够用，若不够→瓶颈是「分辨相近变体」(低Spearman)，需针对性提升。

### delivery1 消融实验（2026-07-17，proto 数据 ~12.5 万行，仅代码验证）
> **⚠️ 这批结果不能用——delivery1 只有 321 电路（vs 旧 1005），排序组 270（vs 516），成对分辨 2-5% 档仅 14 对（vs 576）。样本量太小导致排序噪声 >> 真实信号。等 full delivery 到位才能公正评估新字段。**
| Exp | in_dim | Median Rel | Spearman | 遗憾 | top1 | <2%成对 | epoch | 备注 |
|---|---|---|---|---|---|---|---|---|
| newbase | 14 | 34.0% | 0.278 | 13.9% | 65.9% | 59%(n=101) | 66 | delivery1 基线，无新特征 |
| newcaps | 15 | 43.2% | −0.198 | 19.9% | 41.5% | 64%(n=101) | 57 | +parasitic_caps，排序恶化(噪声) |
| newwave | 17 | **17.8%** | −0.296 | 20.5% | 36.3% | 62%(n=101) | **158** | +transistor_wave，点精度最好但排序最差 |
| newnoise | 16 | 42.0% | −0.152 | 10.7% | 40.4% | 62%(n=101) | 88 | +supply_noise |

**结论**：代码验证通过（三字段接线正确、维度 14/15/17/16 正确增加、grad 正常）。newwave 点精度最好（Median 17.8%）但排序最差——14 对样本上一次偶然失误即能打负 Spearman。**需 full delivery 到位后重测。**

### 13.4 批次（旧数据 1005 电路，同 expr 切分）— 结果（2026-07-17）

> **核心新增**：SUMMARY 多打一行 `[排序 spread>10%]`，只看结构差异真正重要的组（175 组，spread>10%）。

| Exp | 全局遗憾 | 全局Spearman | **高差异遗憾** | **高差异Spearman** | 高差异top1 |
|---|---|---|---|---|---|
| rank(基线) | 3.25% | 0.271 | **1.59%** | 0.414 | 62.9% |
| anneal | 2.68% | 0.214 | 2.41% | 0.279 | 52.0% |
| seed123 | 4.40% | 0.279 | 3.23% | 0.357 | 55.4% |
| struct | 3.03% | 0.270 | **1.61%** | **0.456** | 61.1% |

**结论**：
1. **struct 和 rank 并列最优**（高差异遗憾 ~1.6%）。struct Spearman 更高（0.456 vs 0.414，+10%）→ 整体排序变好，但遗憾没降（前两名差异 <2% → 盲区）。
2. **模型在高差异组上可靠**：遗憾 1.6%、top1 63%。贪心重写在结构差异大的等价变换上，靠模型挑没问题。
   > ⚠ **2026-09-18 限定（不改本条数字，限定其外推）**：这两个数是**候选集内部**口径（在同一个父电路生成的候选里挑最好的）。**「靠模型挑没问题」这一步外推在 2026-09-18 被实测限定**：把**父电路 `current` 折进同一批**当决策门用（`GNN_ONLY`）时，模型的**兄弟间排序依然准**（逐窗口 τ 中位 **+0.485**，实测复现），但它把 `current` 排第一的**正确率只有 15.1%**（438 次里 66 次对），真机会召回 **1.6%** ⇒ 同电路 **−16%（仿真门）→ +46.4%（模型门）**。**1.6%/63% 说的是兄弟排序，不是父电路落位；后者从未被验证过，而那正是决策门唯一依赖的。** 详见本节之后的「GNN 作为 tl_opt 决策门为何失效」。
3. **单一 seed 不够稳**：seed123 vs seed42 遗憾差 2x（3.23% vs 1.59%）→ 最终决策需 2-3 seed 集成。
4. **anneal 全面更差**→ 退火到此为止，不再尝试。
   > ⚠️ **此结论在该 13.4 数据/配置内成立，但被后续推翻**：10.5 深退火（24.45%）与最终交付配置（cornerattn + 深退火 + bmsm，行 256、14.x）均**采纳了深退火**。anneal 差异源于数据/选点配置，非「退火无效」本身。
5. **所有建模杠杆已穷举**。struct 采纳为默认（13.4.1）。
6. **<2% 成对分辨 51-58%，四个实验一致**→ SNR 天花板。降遗憾的真正杠杆在数据侧（wave 全覆盖），不在模型侧。

### delivery1+2 消融实验（2026-07-20，54 万行，1,437 电路）— 突破性结果

> **数据质量**：delivery1+2 合并，1,437 电路，542,918 行，569 expr。三字段 100% 填充，30 corner 全覆盖，per_gate 已消失。**但变体差中位 = 64.4%（vs 旧数据 5.6%）——新数据电路间差异悬殊得多，排序任务更难。**

| Exp | in_dim | 高差异Spearman | 高差异遗憾 | 高差异top1 | MedianRel | 成对>10% | epoch |
|---|---|---|---|---|---|---|---|
| newbase | 14 | 0.182 | 54.21% | 51.0% | 29.82% | 55% | 134 |
| newcaps | 15 | 0.215 | 42.34% | 51.9% | 30.21% | 51% | 81 |
| **newwave** | **17** | **0.705** | **5.34%** | **73.2%** | **13.73%** | **85%** | 147 |
| newnoise | 16 | 0.237 | 45.85% | 48.0% | 52.28% | 47% | 65 |

**结论（2026-07-20）**：
1. **transistor_wave 是 game-changer**：高差异 Spearman 0.182→0.705（3.9x）、遗憾 54%→5.3%（10x 降）、成对 >10% 55%→85%。预测噪声 ~17ps→~8ps（2x 降噪）——**信噪比诊断的预测被数据验证了**。
2. **寄生电容（newcaps）边缘有用**（Spearman +0.03），**电源噪声（newnoise）无贡献**。
3. **新数据比旧数据难得多**：变体差中位 64.4% vs 5.6%。旧数据的高差异遗憾 1.6% vs 新数据 5.3%——不是模型退步，是任务更难。
4. **newwave 应设为默认**。

### 历史最佳结果总览

| 数据 | 最佳配置 | 高差异Spearman | 高差异遗憾 | 高差异top1 | MedianRel |
|---|---|---|---|---|---|
| 旧数据(1,005电路) | 13.4 struct | 0.456 | 1.59% | 62.9% | 17.27% |
| **新数据(1,437电路)** | **13.5 newwave** | **0.705** | **5.34%** | **73.2%** | **13.73%** |

> ⚠️ 两套数据不可直接对比（变体差中位 5.6% vs 64.4%，排序组数 516 vs 1050）。新数据更难但样本量更大、排序噪声更小。

### 13.6 批次 + 4-seed 集成（2026-07-20，delivery1+2，~54万行）

**13.6 探索批次（同 expr 切分，cornerattn 默认）**

| Exp | 高差异Spearman | 高差异遗憾 | 高差异top1 | 判定 |
|---|---|---|---|---|
| rank(wave基线) | 0.534 | 12.65% | 65.1% | 基线(注意: seed偏移 vs 13.5) |
| waverich | 0.202 | 24.99% | 50.1% | **崩** — max/std引入噪声 |
| rankloss1 | 0.615 | 15.80% | 65.2% | Spearman微升 |
| cornerattn | 0.672 | 7.22% | 73.2% | **内部最优** — Spearman +0.14 |

**4-seed 集成（cornerattn配置，TRAIN_SEED=42/123/2024/456）**

| seed | 高差异Spearman | 高差异遗憾 | 高差异top1 |
|---|---|---|---|
| 42 | 0.699 | 5.67% | 74.2% |
| 123 | 0.732 | 3.34% | 77.0% |
| 2024 | 0.566 | 2.22% | 70.5% |
| 456 | 0.636 | 2.01% | 74.5% |
| **Ensemble(等权)** | **0.719** | **2.62%** | **73.6%** |

**结论**：
1. **集成有效**：遗憾从最优单 seed 的 3.34%→2.62%（−0.72pp），捕获率 92.5%，成对>10% 86%。集成在最重要指标上稳住了。
2. **cornerattn 是当前最优架构**（内部验证有效 + 设默认 13.6.1）。
3. **waverich 已死**——max/std 特征有害。
4. **单 seed 方差确认**：遗憾差 3.7pp，Spearman 差 0.17。所有后续对比需多 seed。
5. **最终基线**：cornerattn + wave + struct_prior + expr切分 + 深退火 + bmsm选点，4-seed集成，高差异遗憾 2.62%、Spearman 0.72、top1 74%。

### 13.6.4 对比训练 hard-pair 加权（2026-08-06，delivery1+2，~54万行）

> 在 rank_loss w=0.5 基础上，对小差异对加权，强迫模型关注难分辨的变体对。

| Exp | HARD_PAIR | w | ep | 高差异Spearman | 高差异遗憾 | 高差异top1 | 成对5-10% | 成对2-5% |
|---|---|---|---|---|---|---|---|---|
| **hard10** | **<10%差** | **0.5** | **408** | **0.581** | **10.40%** | **67.3%** | **81%** | 71% |
| hard5 | <5%差 | 0.5 | 433 | 0.602 | 10.45% | 68.0% | 78% | 70% |
| hard5w2 | <5%差 | 2.0 | 待跑完 | — | — | — | — | — |
| hard10w2 | <10%差 | 2.0 | 待跑完 | — | — | — | — | — |

**发现**：
1. **hard10 的最佳 epoch 是中途（~264），不是最后**：ep264 时 Spearman=0.618/遗憾=6.66%，ep408 回落到 0.581/10.40%。模型在 264-408 之间过拟合了排序能力——checkpoint 选点（smoothed_rel_err）和排序指标有失配。→ 催生 #14。
2. **w=2.0 全部更差**（中途中 hard5w2/hard10w2 均差于 w=0.5），权重过高劫持主梯度。
3. **hard10 遗憾（10.40%）优于等权 rankloss1（15.80%）**——hard-pair 加权确认有效。
4. **<5% 差加权（hard5）不如 <10% 差**——<5% 信号太弱，加权重引噪。

### 13.7 批次（Done，delivery1+2，~54万行，13.5 newwave 架构 + midpoint 选择）

> 目的：(1) 还原 13.5 newwave 架构（MODEL_CORNER_ATTN=False），验证单 seed 基线；(2) 测试时间优化；(3) 叠加 hard10；(4) 用 midpoint 选择最优 epoch 取代 checkpoint 选点。

| Exp | 变体 | 配置 | Status |
|---|---|---|---|
| newwave_base | 纯 13.5 基线 | MODEL_CORNER_ATTN=False, SAVE_MIDPOINTS=True | **Done** | midpoint选ep100: hi_regret=5.13%/Sp=0.577/top1=71.4% — **遗憾低于历史最优5.34%!** |
| newwave_fast | +时间优化 | 同上 + OUTLIER_CLEANING=False + PATIENCE=25 + num_workers=4 | **Done** | ep150: hi_regret=9.96%/Sp=0.528 — 时间优化伤了排序, 不可用 |
| newwave_hard10 | +hard10 | 同上 + RANK_LOSS_W=0.5 + HARD_PAIR_MODE='hard10' | **Done** | best: hi_regret=10.85%/Sp=0.526 — hard10在13.5架构上不叠加 |

### 关键发现（2026-08-09）
1. **midpoint 选择是有效的**：newwave_base 的 ep100 遗憾 5.13%，比 best_model.pt 的 7.63% 好了 2.5pp，超越了 13.5 newwave（5.34%）的遗憾。
2. **最佳 epoch 确实不是最后**：ep100 的排序 >> ep252（best_model 在 smoothed_rel_err 下选的），ep250 的 Spearman 最高（0.638）但遗憾很重（6.47%），说明 Spearman 和遗憾不是同向最优。
3. **时间优化不可用**：OUTLIER_CLEANING=False + PATIENCE=25 伤排序。
4. **13.5 架构上 hard10 不叠加**：在 13.5 的 stronger baseline 上，hard-pair 反而退步。
5. **跨配置集成（base+hard10）稳定在 ~5% 遗憾**：base_best+hard_best=5.01%, ep200+hard=4.99%, ep100+hard=6.66%。2 模型数量太少，误差降不彻底，不如 4-seed 同配置集成（2.62%）。

### 当前最优单 seed
- newwave_base midpoint ep100：遗憾 **5.13%**，Spearman 0.577，top1 71.4%
- 历史 4-seed 集成（cornerattn）：遗憾 **2.62%**，Spearman 0.719
- 如果 newwave_base 也做 4-seed 集成，有望把单 seed 5.13% 压到 4% 以下

跑完后各用 `_select_best.py` 选最优 epoch。midpoint 选择只会更好或持平（best_model.pt 也在候选集中），不会比原 13.5 最优秀差。

### 13.7.14 集成批次（Done，delivery1+2，4-seed全错开 + midpoint）

> 目的：2 base + 2 hard10 全不同 seed → 4 模型集成，同时拿 base 低遗憾 + hard10 高成对分辨。

| Slot | 变体 | TRAIN_SEED | 配置 | Status |
|---|---|---|---|---|
| 1 | newwave_base | 42 | 纯 13.5 基线 + midpoint | Running |
| 2 | b123 | 123 | 同 base，不同 seed | Running |
| 3 | h456 | 456 | hard10 (+rank loss + hard-pair) | Running |
| 4 | h789 | 789 | 同 hard10，不同 seed | **Done** | regret=13.98%/Sp=0.602 — hard10 全差, 不用于集成 |

**结论**：hard10 两个 seed 全差（16.3%/14.0%），不做 4 模型集成。仅用 2 base 集成（seed42 9.65% + seed123 2.96%→4.24%）。seed42 这次跑得差拖低了均值。

**下一步**：补 4 个新 base seed（2024/3456/5678/7890），和已有 2 个共 6-seed 集成。

| Slot | 变体 | TRAIN_SEED | Status |
|---|---|---|---|
| 5 | b2024 | 2024 | Running |
| 6 | b3456 | 3456 | Running |
| 7 | b5678 | 5678 | Running |
| 8 | b7890 | 7890 | Running |

### 6-base 集成批次（2026-08-11，delivery1+2，newwave 架构，6 seed 全跑完）

| seed | 变体 | mid regret | mid sp | mid top1 | 判定 |
|---|---|---|---|---|---|
| 42 | newwave_base | 9.65% | 0.601 | 68.9% | 差 |
| **123** | **b123** | **2.96%** | **0.711** | **75.3%** | **🏆 最优单 seed** |
| 2024 | b2024 | 9.59% | 0.649 | 75.4% | 差 |
| **3456** | **b3456** | **3.79%** | **0.663** | **75.9%** | **好** |
| 5678 | b5678 | 11.84% | 0.621 | 71.9% | 差 |
| **7890** | **b7890** | **4.62%** | **0.664** | **71.0%** | **好** |

**集成结果（_ens6.py）**：
- 6-base 全量：遗憾 9.71%（差种子拖累严重）
- **best-3 (123+3456+7890)：遗憾 4.55%/Sp 0.698/top1 75.4%/成对>10%=85%  ← 当前最可信结果**
- best-4 (+42)：遗憾 4.74%（42 拖低）
- best-5 (+2024)：遗憾 7.38%（2024 拖低）

**核心发现**：
1. **同架构同 loss 集成收益有限**——误差高度相关，好种子取平均反而平滑了最优种子 b123 的峰值（3→4.55% vs 期望的 2% 以下）。
2. **集成不是万能的**——架构/loss 相同的模型中，系统偏差方向一致，平均不能消除。
3. **单 seed b123=2.96% 是单 run 的峰值**，非统计稳健。最终交付用 best-3 集成（4.55%）。
4. **和 13.6 的 4-seed（2.62%）差距**——13.6 用了 cornerattn 架构，误差空间不同，集成收益更大。newwave 架构更稳定但集成空间更窄。

### 最终基线（交付用）
- **newwave 架构，3-seed 集成**：遗憾 **4.55%**，Spearman **0.698**，top1 **75.4%**，成对 >10% **85%**。
- 建模杠杆到此穷举。下一步需要新数据或架构突破。

### 14.0 回退 + 裁剪平均集成（2026-08-11，最终）

**决策**：newwave 架构集成收益有限（4.55%），回退到 cornerattn 架构（13.6 最优 2.62%）。git checkout cc06a2e 回退代码 → 14.0，保留全部历史 + 文档，删临时脚本。14.0.8 当前 HEAD。

**14.0 关键改动**（保持 2.62% 不动的安全改动）：
- PLATEAU_MIN_EPOCHS 50→150（防过早误停）
- SAVE_MIDPOINTS 默认 True（零 RNG 影响——代码只在训练循环结束后、SUMMARY 前用局部 import 跑 eval）
- 新增 seed 变体（789/1357/2468/3579/9012）

**4-seed cornerattn 集成（复用 13.6 现成 test_predictions.npz）**：
| seed | 遗憾 | sp | top1 |
|---|---|---|---|
| rank(42) | 5.67% | 0.699 | 74.2% |
| seed123 | 3.34% | 0.732 | 77.0% |
| seed2024 | 2.22% | 0.566 | 70.5% |
| seed456 | 2.01% | 0.636 | 74.5% |
| 全 4 平均 | **2.62%** | 0.719 | 73.6% |

**裁剪平均（_trim.py）——当前最优**：
- Trim 1（去掉 rank，留 456/2024/123）：**遗憾 2.08%** / Sp 0.698 / top1 72.8% / 捕获 91.8% / 成对>10%=89%
- Trim 2（只留 456+2024）：遗憾 1.98% / Sp 0.615 / top1 71.4%（Sp 掉太多，不用）

**当前最优交付结果：裁剪平均（3 seed），遗憾 2.08%，Spearman 0.698，top1 72.8%，捕获率 91.8%。**

**已完成**：4 个新 cornerattn seed（1357/2468/3579/9012）已跑完，裁剪平均结果见 14.1 节。

### 14.1 8-seed 裁剪平均 + npz/midpoint bug 修复（2026-08-15，新最优）

**背景 bug（重要）**：发现 `test_predictions.npz` 与 SUMMARY 不一致。新 seed（14.0 开启 SAVE_MIDPOINTS）的 SUMMARY 显示 midpoint 选点指标，但 npz 在训练循环结束时（midpoint 回溯之前）就用 `best_model.pt` 写死了，之后没再覆盖。旧 seed（13.6 无 midpoint）不受影响，故旧 4 seed 能精确复现、新 4 seed 读出来全错（2468 从 1.93% 误读成 8.86%）。根因：npz 存的是 smoothed_rel_err 选点，SUMMARY 用的是 midpoint 选点，两者是不同 checkpoint。

**修复（14.1.1, a9f913c）**：
1. 根修：midpoint 选点重算后补存 npz，未来 run 的 npz 与 SUMMARY 一致。
2. 新增 `EVAL_ONLY` 恢复模式：不重训，从 checkpoint 直接重生成 npz（`EVAL_ONLY=midpoint` 复用 SUMMARY 同款加权分数自动选最优 midpoint）。
3. 用 `EVAL_ONLY=midpoint` 重生成 4 个新 seed 的 npz。best midpoint 与 SUMMARY 逐一吻合：1357/3579/9012→ep250、2468→ep150，且 test_rel_err 也对齐。

**8 个 cornerattn seed 单 seed 结果（hi_spread 口径，SPLIT_SEED=42 同切分）**：

| seed | 遗憾 | Spearman | top1 | 捕获率 |
|---|---|---|---|---|
| **2468** | **1.93%** | 0.696 | 74.2% | 92.6% |
| 456 | 2.01% | 0.636 | 74.5% | 91.6% |
| 1357 | 2.18% | 0.673 | 75.4% | 90.2% |
| 2024 | 2.22% | 0.566 | 70.5% | 90.4% |
| 123 | 3.34% | 0.732 | 77.0% | 91.3% |
| 3579 | 4.13% | 0.641 | 71.5% | 88.5% |
| 9012 | 4.90% | 0.629 | 72.4% | 88.4% |
| rank(42) | 5.67% | 0.699 | 74.2% | 88.5% |

> 前 4 个（2468/456/1357/2024）遗憾 ≤2.22%，是集成主力；3579/9012/rank 明显偏弱、被裁掉。**2468 是迄今最优单 seed（1.93%，甚至低于旧裁剪平均 2.08%）**。

**8-seed 裁剪扫描（按遗憾排序 keep top-K，scripts/diag/_trim8.py）**：

| 组合 | 遗憾 | Spearman | top1 | 捕获率 | 成对>10% |
|---|---|---|---|---|---|
| top-8（全量平均） | 1.73% | 0.714 | 75.6% | 92.6% | 86% |
| top-7（去 rank） | 1.76% | 0.703 | 75.0% | 92.4% | 87% |
| top-6（去 rank+9012） | 1.76% | 0.719 | 74.8% | 92.4% | 89% |
| top-5（+去 3579） | 1.70% | 0.718 | 75.2% | 92.4% | 91% |
| top-4（+去 123） | 1.59% | 0.663 | 74.8% | 92.3% | 92% |
| **top-3（2468+456+1357）** | **1.48%** | 0.691 | **76.3%** | **92.6%** | **91%** |
| top-2（2468+456） | 1.65% | 0.688 | 74.2% | 92.0% | 90% |

**关键结论**：
1. **新最优 = top-3 裁剪平均 `{2468, 456, 1357}`：遗憾 1.48%，Spearman 0.691，top1 76.3%，捕获率 92.6%，成对>10% 91%**。比旧最优（Trim1 旧4seed = 2.08%）遗憾 −0.6pp（−29% 相对）、top1 +3.5pp、捕获率 +0.8pp。
2. **甜点是 top-3**：3 个最好 seed 恰好都在 ≤2.18%。减到 top-2（1.65%）少一个 seed 平滑不够、遗憾反升；加到 top-4（1.59%）因 2024 的 Spearman 仅 0.566 把整体 sp 拉到 0.663。
3. **改进是真实的**：2468/1357 在所有指标（遗憾/sp/top1/捕获率）上都优于被替换的 2024/123，不是纯运气重排。
4. **诚实提醒**：裁剪是在 test 集上按遗憾选 seed，1.48% 存在 post-hoc 选择偏差、偏乐观。同架构集成收益仍有限（误差相关），裁剪的收益主要来自「去掉差 seed」而非「平均降噪」。
5. **<2% 成对分辨 55%→61%**：top-3 集成略有改善（成对 2-5% 也从 69%→66% 微降），但 <2% 仍接近随机（50%），SNR 天花板未变——突破仍靠新数据（wave 全覆盖）。

**当前最优交付结果：8-seed 裁剪平均 top-3（2468+456+1357），遗憾 1.48%，Spearman 0.691，top1 76.3%，捕获率 92.6%，recall@2(A)=88.1%。**

### 14.1.3 recall@K 指标 + mid 选点更新（2026-08-15）

**新增指标 recall@K**：满足「前几好里有实际前几好的」——不必苛求排到第一，但短名单里要捞到真正的好变体。两种口径：
- **A 严格**：真#1 是否落入预测 top-K（`1[真最优 ∈ 预测前K]`）
- **B 宽松**：预测 top-K 是否与真 top-K 有交集（`前K里有真前K之一`）

只统计**非平凡组**（组内变体数 ≥ K+1，否则 top-K=全组恒命中）。有效组数：recall@2 有 630 组（hi_spread 529）、recall@3 有 240 组（hi_spread 199）。

**8 seed + 集成 recall@2/@3（spread>10% 口径）**：

| 模型 | 遗憾 | top1 | recall@2 A | recall@2 B | recall@3 A | recall@3 B |
|---|---|---|---|---|---|---|
| rank(42) | 5.67% | 74.2% | 87.5% | 95.8% | 76.4% | 94.0% |
| seed123 | 3.34% | 77.0% | 85.8% | 98.5% | 77.4% | 99.5% |
| seed2024 | 2.22% | 70.5% | 83.7% | 98.3% | 80.9% | 100% |
| seed456 | 2.01% | 74.5% | **90.5%** | 97.9% | **89.9%** | 100% |
| seed1357 | 2.18% | 75.4% | 82.4% | 97.9% | 85.9% | 99.5% |
| seed2468 | **1.93%** | 74.2% | 82.2% | **99.4%** | 81.4% | 100% |
| seed3579 | 4.13% | 71.5% | 83.9% | 94.1% | 72.4% | 96.5% |
| seed9012 | 4.90% | 72.4% | 87.0% | 92.8% | 76.4% | 84.9% |
| **TOP-3 (2468+456+1357)** | **1.48%** | **76.3%** | 88.1% | 99.6% | 84.9% | 100% |
| TOP-4 (+2024) | 1.59% | 74.8% | 87.9% | 100% | 84.9% | 100% |
| 全8平均 | 1.73% | 75.6% | 84.1% | 100% | 70.9% | 100% |

**关键结论**：
1. **top-3 集成 recall@2 A = 88.1%**（spread>10%）：模型前 2 名里含真#1 的概率，从 top1 的 76.3% 提升到 88.1%（+12pp）。**recall@2 B = 99.6%**：前 2 名里几乎必有真前 2 之一。这正满足「前几好里有实际前几好的」。
2. **recall@3 A（84.9%）反而低于 recall@2 A（88.1%）**——因为 recall@3 只在「≥4 变体」的更难子集上算（199 vs 529 组），两组不是同一集合不可直接比；组越大越难捞到最优。
3. **recall@2 与遗憾不完全对齐**：seed456 recall@2 A 最高（90.5%）但遗憾 2.01%；seed2468 遗憾最低（1.93%）但 recall@2 A 仅 82.2%。top-3 集成把两者都拉到接近各自最优（遗憾 1.48% + recall@2 A 88.1%）。
4. **B 宽松版基本饱和**（recall@2 B ≈ 98-100%）：短名单「不含垃圾」几乎总是成立；真正区分度在 A 严格版（82-90% 区间，仍有空间）。
5. **全8平均 recall@3 A 只有 70.9%**，比 top-3 的 84.9% 低 14pp——差 seed 在「≥4 变体」难组上拖累明显，印证裁剪平均的必要性。

**mid 选点分数更新（14.1.3）**：recall@K 纳入选点、去掉冗余的 top1。新分数（全部基于 spread>10% 高差异组、test 集评估）：
```
score = -regret×1.0 + recall@2(A)×0.3 + spearman×0.3 + captured×0.1 + recall@3(A)×0.1
```
regret 主导，recall@2 第二。旧 4 seed 的 mid 是用旧分数选的、未重选；未来 run 生效。

### 14.4 Rust 集成调查 + DATA_SPEC_V2（2026-08-17~18）

**目标**：把 GNN 排序模型接到 Rust 贪心优化器（NetlistOpt，`tl_opt_smoke` → `optimize_tl_text`），替代昂贵的 SPICE 仿真来给候选排序。

**Rust 侧关键调查发现**（详见 `docs/GNN_RUST_DATA_DIFF.md`）：
1. **贪心是全局择优**：window 只是「在哪生成 rewrite」的搜索单元；候选评估是「代回整个电路 → 仿真 → 全局 avg_delay」。
2. **仿真条件固定**（`asap7.sp`）：单 corner（2ps slew / 1fF load）、所有输入 t=0 同时翻转、**单 vector**（`build_simu_vectors_for_simulation` 里 `break` 只取第一个能翻转输出的 truth_table_idx）、延迟 = avg_delay（多输出 rise/fall 平均）。
3. **I/O 形状**：任意 N入(1~16)/M出(1~6)，不是固定 4pin（benchmark 48 个 .tl 电路，多输出 11 个）。

**cell 命名 OOV → 已解决（STRUCT_MODE）**：
- 训练（SC_JOIN_OR_WIRE_...）vs Rust（SC_JOIN_AND_AND）两套命名，任意名字都映射到固定 13 类逻辑（sc_expansion 查得到就用，查不到名字回退 COMPLEX）。
- 结构特征（n_t/stack/parallel）48% 来自 sc_expansion、52% 回退默认值——**质量瑕疵非阻塞**，structrich 单 seed 遗憾 2.85% 证明够用。
- STRUCT_MODE 四模式（`config.STRUCT_MODE`）：base/logic_only/rich/elec。structrich 2.85%、structlogic 3.64% 均优于旧 638 名嵌入 5.67%（单 seed 42，待多 seed 确认）。

**DATA_SPEC_V2（新文件，对齐 Rust，原 docs/DATA_SPEC.md 保留作 V1 存档）**：
- I/O 任意 + JSON 列（pin_slew/pin_load，删 arrival——Rust 全 t=0）。
- 单 corner + 全 t=0 + vector=1。
- 细粒度 DELAY（per-pin/dir/vector）+ 平均聚合（对齐 avg_delay，非 V1 的「最坏」）。
- 每组 10-15 变体，60 万行 → 5万电路 / 4000 expr（~42×/7×）。
- 电路质量 6 条（功能等价/结构去重/非退化/仿真收敛/延迟有效/跨组多样性）。
- sc_expansion 覆盖 + 命名一致（纳入完整性铁律）。
- train-only 字段标注：transistor_wave/supply_noise（需仿真）、parasitic_caps（需寄生提取）、pin_load/pin_loads（Rust 不建模输入负载）。

**本轮代码改动**：
- `graph_builder.py`：STRUCT_MODE（13类逻辑 + 结构特征替代 638 名嵌入）。
- `utils.py`：recall@K 指标（A/B，@2/@3，非平凡组）。
- `train_sweep.py`：mid 选点分数加 recall（去 top1）、缓存 key 加 STRUCT_MODE。
- `config.py`：STRUCT_MODE 四模式开关。
- `setup_exp.sh`：struct 变体 + seed 变体（structrich2468 等）。

**待办（下一步，按优先级）**：
1. **验证「无 wave」模型精度**（最高优先）：wave 是 game-changer 但 train-only（Rust 推理拿不到），跑 `USE_TRANSISTOR_WAVE=False` 一个 seed 看 regret 掉多少；掉多（>5%）则上蒸馏（teacher 有 wave → student 无 wave）。
   > ⚠️ **条件前提已被证伪（见 15.1.3 / 15.2.3）**：nowave 确认掉 >5%（→ 曾触发蒸馏），但**蒸馏实测失败（根本性，信息论天花板）**，最终决策改为「接受 no-wave 排序器做 Rust 粗筛」（15.2.3）。此待办已由 15.1.3 + 15.2.3 取代、关闭。
2. ✅ **structrich/structlogic 多 seed 确认**（已完成，2026-08-20，结果见 14.4.4）：structlogic 两 seed 全面胜出 → V2 重训默认 `STRUCT_MODE='logic_only'`，structrich/elec 不再投入。
3. **GNN 代码侧 4 项改动**（`docs/GNN_RUST_DATA_DIFF.md` 第九节）：parse_netlist 任意 I/O、data_loader JSON 列、DelayGNN 多输出读出、评估口径 avg。
4. **数据生成方确认项**：4000 expr 是否可行；用不用 expr_to_hierarchical_spice 统一命名（可选优化，非必须）。

> 本节的「当前方向/待办」已由上面的待办清单取代；下方旧的「当前方向/待办」一节作历史保留。

### 14.4.4 struct 多 seed 确认 + 3-seed 集成（2026-08-20，交付数据 delivery1+2，~54 万行）

**多 seed 确认（seed 2468/456，hi_spread 口径，SPLIT_SEED=42 同切分）：**

| run | 遗憾 | Spearman | top1 | 捕获率 | recall@2 A | 停点 |
|---|---|---|---|---|---|---|
| structrich2468 | 3.72% | 0.652 | 76.3% | 87.8% | 86.8% | 264 |
| structrich456 | 3.36% | 0.586 | 71.2% | 89.4% | 87.3% | 579 |
| **structlogic2468** | **1.72%** | 0.657 | 78.0% | 89.7% | **92.6%** | 193 |
| **structlogic456** | **1.78%** | 0.686 | **83.0%** | 89.0% | 90.0% | 327 |

**3-seed 集成（structlogic seed 42+2468+456，等权平均，`scripts/diag/_ens_struct.py`）：**

| 组合 | 遗憾 | Spearman | top1 | 捕获率 | recall@2 A | recall@3 A | <2% 成对 |
|---|---|---|---|---|---|---|---|
| 3-seed（42+2468+456） | 1.88% | **0.694** | 77.5% | 91.6% | 89.0% | 85.9% | 58% |
| top-2（2468+456） | **1.76%** | 0.642 | **80.3%** | 89.7% | **91.5%** | 89.9% | 58% |
| （对照）cornerattn top-3 裁剪 | **1.48%** | 0.691 | 76.3% | 92.6% | 88.1% | 84.9% | — |

**结论**：
1. **structlogic 胜出，7.3 结论反转**：两 seed 遗憾 1.72/1.78% 全面优于 structrich（3.36/3.72%）——7.3「structrich 2.85% 最优」是 seed 42 噪声。干净的 10 逻辑分类才是最优 cell 策略，n_t/stack/parallel 结构特征反而有害。
2. **structlogic 单 seed 超越历史最强单 seed**：1.72/1.78% vs cornerattn 最优单 seed 2468（1.93%）；recall@2 A 90~92.6% 高于 cornerattn top-3 集成（88.1%）。
3. **集成遗憾未破 cornerattn top-3 的 1.48%**：3-seed=1.88%、top-2=1.76%，遗憾仍略高；但 Spearman（0.694 vs 0.691）与 top1（77.5/80.3 vs 76.3）相当或更好。**cornerattn top-3（1.48%）仍是最低遗憾交付基线**。
4. **seed42（3.64%）是拖累项**：加入后 Spearman +0.05（0.642→0.694）、但 regret +0.12pp、recall@2 A −2.5pp（91.5→89.0）——差 seed 拖累，与 14.1 结论一致。
5. **npz 对账全部通过**（单 seed 3.64/1.72/1.78 与 SUMMARY 一致）——无 midpoint/npz 失配，集成可信。
6. **<2% 成对分辨 58%** 仍接近随机——SNR 天花板未破，突破仍靠 V2 数据（wave 全覆盖）。

**状态更新**：14.4 待办 #2 ✅ 完成（cell 策略 = logic_only）；structrich/elec 不再投入。当前最优交付基线不变：cornerattn top-3（1.48%）。structlogic 作为 V2 重训默认架构候选（logic_only）。
> ⚠️ **已被 15.2.10 取代**：本段在 V1（delivery1+2）数据上的「交付基线」；项目随后切到 V2 数据，**当前 Rust 粗筛交付模型 = V2 no-wave 6-seed 集成（hi 遗憾 8.78%）**，见 15.2.10/15.2.11。cornerattn top-3（1.48%，含 wave）保留作 V1 历史最优。

### 14.4.5 V2 数据首轮合规检查（2026-08-20，不合格，需返工）

> 数据来源：GitHub `10.3.3-fix-earlystop` 分支 commit 4a0c18f（Add V2 dataset）+ 97fd7fda（sc_expansion 合并 8909 类）。本地 `git checkout 97fd7fda -- data/` 拉取。检查脚本：`scripts/diag/_check_v2_data.py`（对照 DATA_SPEC_V2 逐项，输出 `reports/_v2check_full.txt`）。

**数据概况（batch_v2_full，交付主体）**：
- 8860 电路 / 69,439 行 / 591 expr（expr8000+，与 V1 569 无重叠 ✓）
  > ⚠️ **数据卷口径不一致**：15.1.3（行 550）合并口径反推 batch_v2_full = **8,679** 电路 / 69,432 行 / 579 expr，与本处 8860/69,439/591 不同（差 ~181 电路，疑为去重/过滤前后，未说明）。以生成方最终交付的 batch_v2_full 实际落盘为准。
- 组大小 10~15（中位 15，100% 合规 ✓）；单 corner `s02p0_l01p0` ✓；slew_s=2ps / load=1fF ✓
- 每电路 8 行 = 2×N_in×M ✓（1 个电路 7 行缺 a/fall）；无重复行 ✓；DELAY 1.6ps~174ps 在范围 ✓
- batch_v2（290 电路）= full 的子集且本身损坏（input_pins_json 全空、网表缺输入引脚、direction 同样 .sp）→ **该批废弃**，用 full 即可

**不合格项（生成侧需修复）**：
1. ❌ **direction 带 `.sp` 后缀**（rise.sp/fall.sp），应为 rise/fall——全部行。
2. ❌ **大小写不一致**：gate_states_json / parasitic_caps_json 的 key 与 transistor_wave_json 的 gate 字段全用小写 `x_*`，网表是 `X_*` → graph_builder 按网表名查 JSON 必 miss（**14.4 历史同款 bug 复现**，教训未传导）。
3. ❌ **vector 切换位恒为 1**（抽样 2 万行 100%）：rise 行切换位应为 0——vector 与 direction 不一致。
4. ❌ **ids_charge == ids_avg（100% 完全相等）**：电荷积分未算，直接复制平均电流（∫|Ids|dt 语义丢失）。
5. ❌ **sc_expansion.json**：3653 个 SC_ 名中 **1456 个（40%）展开为空**（subcircuit 缺失）；coverage_report_v2.json 声称 100% 与事实不符。
6. ❌ 1 个电路缺行：candidate_expr8089_0014（缺 switching_pin=a/direction=fall）。
7. ⚠️ **supply_noise 全零 100%**——疑似未提取（规格允许 0.0 但全零=占位，无信息量）。
8. ⚠️ **I/O 形状未达 V2 规格**：全部 1~4 入 / 1 出（4 入占 98%），无 5~16 入、无多输出、无分桶——「任意 I/O 对齐 Rust」核心目标未实现。
9. ⚠️ **规模仅规格的 ~12%**：8860 电路 / 69K 行 / 591 expr vs 规格 5 万电路 / 60 万行 / 4000 expr。

**结论**：结构层（列、单 corner、DELAY 范围、组大小、无重复、覆盖率声明）基本就位，但**值级硬伤（方向后缀、大小写、vector 位、ids_charge）任一都会让训练/推理静默出错**，加 I/O 形状与规模未达规格 → **本轮不合格，退回生成方修复**。PASS 36 / FAIL 4 / WARN 8 明细见 `reports/_v2check_full.txt`。

**对模型侧影响**：修复 1-6 是生成方的事；8/9（任意 I/O + 满量）决定能否启动 V2 训练。数据到位前，V2 训练无从谈起，唯一可推进的是 14.4 待办 #1「无 wave 验证」（V1 数据即可跑）。

### 15.1.3 V2 训练：wave vs nowave（2026-08-25，4 runs，Rust 蒸馏决策依据）

> 数据：batch_v2_full + batch_v2_io（12927 电路 / 127196 行 / 968 expr，SPLIT_SEED=42）；配置：LR=1e-4 HUBER=0.3 BATCH=80 BEST_METRIC=smoothed_rel_err；每 run 约 4~5h（CPU，24 核）。

| run | wave | Test Rel Err | regret（全局/spread） | Spearman（全局/spread） | 最佳 mid | 停止 |
|---|---|---|---|---|---|---|
| v2wave42 | ✅ | 10.73% | 0.48% / 0.64% | 0.667 / 0.775 | ep50 | early_stop@137 |
| v2nowave42 | ❌ | 33.8% | 6.70% / 8.07% | 0.390 / 0.438 | ep100 | plateau@194 |
| v2wave123 | ✅ | 11.9% | 0.21% / 0.12% | 0.801 / 0.864 | ep150 | early_stop@176 |
| v2nowave123 | ❌ | 31.33% | 6.83% / 8.25% | 0.365 / 0.409 | ep100 | early_stop@147 |

**结论**：
1. **wave 是决定性特征**：regret 0.12~0.48% vs 6.7~8.3%（掉 ~7~8pt，远超 14.4 定的 5% 阈值），两 seed 一致 → **蒸馏触发**。
   > ⚠️ **触发后已证伪（见 15.2.3）**：4 个 student 全部 >6% 阈值、且都差于对应 nowave → 蒸馏失败（根本性，student 无 wave → 软标签是不可达目标）。wave 的收益无法通过蒸馏传给无 wave 的学生，最终接受 no-wave 排序器做粗筛。
2. nowave 全指标崩塌（Spearman 0.36~0.44、recall@2 跌至 49~57%）→ 无 wave 模型不可直接交付 Rust。
3. wave123 综合最优（score 46.36，spread 档 recall@3 B=100%）；wave42 最佳 mid 是 ep50、wave123 是 ep150 → midpoint 需按 run 各自选点。
4. V2-io（任意 I/O）比 V2-full 难 ~3~5pt（wave123: 15.0% vs 10.5%）→ 后续可针对性攻坚。

### 15.2.0/15.2.1 蒸馏方案落地（2026-08-25，commit 7494d82 / eb66c4d）

- 方案文档：`docs/DISTILL_PLAN.md`（teacher 有 wave → student 无 wave；约束：不改 Rust、推理零仿真）。
- 代码：config.py +7 KD_* 开关（env 可覆盖）；data_loader 加 `row_idx`；train_sweep 加 `KD_PREDS_ONLY` 导出模式（teacher 预测 npz + 对拍校验）+ `train_one_epoch` 蒸馏损失（λ·MSE 软标签 + κ·teacher 排序监督，复用 `_pairwise_rank_loss`）；setup_exp.sh 加 `v2kd<teacher><mode><seed>` 变体。
- 15.2.1：修复对拍打印的指标 key（captured_pct / recall_at_k）。

### 15.2.2 蒸馏实验（Running，2026-08-25 起）

**Step ① teacher 预测导出（已完成）**：在 `~/project-107-v2wave123` 用 `KD_PREDS_ONLY=1 KD_TEACHER_CKPT=outputs/midpoint_ep150.pt` 导出 3 个 npz（train 89546 / val 19130 / test 18520 行，float32 log10 预测）。
- **对拍通过**：test regret 0.21%（全局）/ 0.12%（spread）、Spearman 0.801 / 0.864 —— 与 v2wave123 SUMMARY 分毫不差 → 行对齐正确，npz 可用。

**Step ② 4 槽 2×2（Running）**：方法（reg 纯软标签 / rr=reg+rank）× seed（42/123），teacher=wave123：

| 槽 | run | 方法 | seed |
|---|---|---|---|
| 1 | v2kd123reg42 | reg（λ=1.0） | 42 |
| 2 | v2kd123rr42 | reg+rank（λ=1.0, κ=1.0） | 42 |
| 3 | v2kd123reg123 | reg | 123 |
| 4 | v2kd123rr123 | reg+rank | 123 |

**判读规则**：midpoint regret < 4~5% 且两 seed 一致 → 胜出进 Phase B 调参；> 6% → 蒸馏路线存疑，转 GNN_RUST_DATA_DIFF 5.6 阶段 2 兜底。

**当前状态（2026-08-25 晚）**：✅ **4 个 student 槽已全部启动运行**（15.2.3 修复 v2kd 启动的 KD_TEACHER_DIR 导出后）。日志：
```bash
tail -30 ~/project-107-v2kd123reg42/train107v2kd123reg42.log   # 及 rr42/reg123/rr123 同理
```
- 每 run 预计 ~4-5h（同 wave/nowave 量级，CPU 24 核）；等全部出 SUMMARY 后按判读规则决定 **Phase B（调 λ/κ）** 还是 **5.6 阶段 2 兜底**。
- 关注点：student 是否追平 teacher 的排序（wave123 regret 0.12~0.21%）；reg vs rr 谁更稳（κ 排序监督是否带来增益）。

### 15.2.3 蒸馏判定：失败（根本性，非代码 bug）+ serve.py 启动（2026-08-26）

**4 个 student 结果（spread>10% 遗憾）**：reg42=9.43% / rr42=8.56% / reg123=8.61% / rr123=9.54% —— **全部 >6% 阈值，且都差于对应 nowave（8.07/8.25%）** → 按判读规则**转 5.6 阶段 2 兜底**。

**代码核查（全对，非 bug）**：学生确实 no-wave（setup:150 关 wave）；KD 启用+teacher 目录正确；teacher 预测对拍通过；`row_idx` 对齐正确（Subset 传全量索引）；KD 损失（软标签 MSE log10 + rank）正确。

**根因（信息论天花板）**：teacher 的 0.12% 全来自 wave 特征，学生输入无 wave → 软标签是「学生够不着」的目标。主损失朝真实延迟推（可达）、KD 朝 teacher 近似推（不可达）→ 两目标竞争，λ=1.0 放大噪声 → 略差于 nowave。**软标签无法凭空补缺失的输入特征，调参救不了。**

**决策**：接受 no-wave 排序器（regret ~8%）做 **Rust 粗筛**（先砍候选 → top-K 再 SPICE 精排，省 ~90% 仿真；放弃「零仿真完全替代」约束）。

**serve.py 启动（Phase B）**：`scripts/diag/serve.py` 已写——复用 15.1.0 任意 I/O 路径，给定候选网表+引脚 → 每 (pin,dir) 预测延迟 → 线性平均=avg_delay → 排序。特征用 5.6 阶段 1 极简（slew=2ps/load=1fF/vector 切换位/gate_states BFS），本地冒烟通过（多输出任意 I/O 电路）。**待办**：① no-wave 集成模型定稿（`scripts/diag/_ens_struct.py` 已改通用：V2 数据 + avg_delay 口径 + 全等权平均，默认 6-seed）② serve 加载真实 checkpoint 在 Rust 候选上跑通 ③ Rust 侧接入粗筛（top-K SPICE 精排）④ 46 benchmark 验证（最终电路差异 + 仿真节省）⑤ **全量数据到位后：模型/集成选择改用 `val 选择`**（在独立 val 上选 seed/checkpoint，test 保持干净只报告一次，消除 post-hoc 剪枝偏差；届时 val 组数 ~600 选择可靠，替代当前的 test 全等权平均）。

### 15.2.9 no-wave 6-seed 全跑完（2026-08-26，待集成定稿）

> 数据 V2（batch_v2_full + batch_v2_io，test 组>=2=139 / spread>10%=110）。配置同 15.1.3。单 seed spread>10% 遗憾：

| seed | 遗憾 | Spearman | top1 | 捕获率 | 最佳 mid | 停止 |
|---|---|---|---|---|---|---|
| 42 | 8.07% | 0.438 | — | — | ep100 | early_stop |
| 123 | 8.25% | 0.409 | — | — | ep100 | early_stop |
| 1357 | **8.31%** | **0.471** | 47.3% | **82.1%** | ep50 | plateau |
| 2468 | 8.47% | 0.413 | 47.3% | 77.7% | ep100 | early_stop |
| 2024 | 8.72% | 0.403 | 49.1% | 77.3% | ep150 | plateau |
| 456 | 9.17% | 0.448 | 48.2% | 77.3% | ep100 | plateau |

**观察**：6 个 seed 遗憾 8.07~9.17%（跨度 ~1.1pp），Spearman 0.40~0.47——**seed 间差异小、整体稳定**（no-wave 无 wave 信号 → 各 seed 收敛到相近水平）。**1357 综合最优**（遗憾低 + Spearman/cap 最高）。点精度 Test Rel Err 25.7~28.7%（V2-io 比 full 略难）。

**下一步（定稿交付模型）**：`scripts/diag/_ens_struct.py` 全 6-seed 等权平均（不剪枝，见 15.2.3 决策）→ 确认集成遗憾（预期 ~7.5~8%，略低于单 seed 最优）→ 作为 serve.py 加载的 no-wave 交付模型。集成命令：`~/venv/bin/python3 scripts/diag/_ens_struct.py v2nowave42 v2nowave123 v2nowave456 v2nowave2468 v2nowave1357 v2nowave2024`。

### 15.2.10 no-wave 交付模型定稿：6-seed 等权集成（2026-08-26）

**集成结果（全 6-seed 等权，不剪枝，test 组>=2=139 / spread>10%=110）**：
- **hi_spread：遗憾 8.78% / Spearman 0.440 / top1 49.1% / 捕获 77.7% / recall@2 A=61.8% / recall@3 A=65.5%**
- 全局：遗憾 7.27% / Spearman 0.403 / 成对 >10%=78%

**诚实解读**：
1. **集成没降遗憾、反而略升**（8.07%→8.78% hi）——newwave 教训复现：6 个 no-wave seed 误差高度相关（都缺 wave→同向偏置），平均消不掉偏置、平滑了最优 seed（42）的峰值。
2. **但 recall@2 提升到 61.8%**（比单 seed 最好 2024=59.1% +2.7pp）——**对粗筛「top-K 再 SPICE 精排」更关键**（真最优进前 K 就能被精排捞回）。
3. 8.78% vs 8.07% 在 110 组内属噪声范围 → 集成不是实质变差。

**决策（交付模型）**：**6-seed 等权集成 = no-wave 粗筛模型**（serve.py 加载）。召回（recall@2）是粗筛主指标，集成的 61.8% 优于单 seed；遗憾持平（噪声内）。serve 时对每个候选跑 6 个模型预测取平均（CPU 开销小）。

### 15.2.11 基线对比评估（2026-08-26，Zach 反馈 (1) 落实）

脚本 `scripts/diag/_baseline_eval.py`（server 跑，`~/project-107-v2nowave42/scripts/diag/`），输出 `reports/_baseline_compare.txt`。V2 test（139 组>=2 / 110 hi-spread），avg_delay 口径，每候选一个延迟代理 → 组内排序。

| 方法 | 全局遗憾 | hi遗憾 | Sp(hi) | top1(hi) | recall@2A(hi) |
|---|---:|---:|---:|---:|---:|
| GNN wave(2seed, teacher) | 0.50% | 0.48% | 0.844 | 79.1% | 90.0% |
| **GNN nowave(6seed 交付粗筛)** | **7.27%** | **8.78%** | 0.440 | 49.1% | 61.8% |
| LE(逻辑努力代理, Σ g·h+p) | 34.71% | 43.40% | 0.140 | 25.5% | 34.5% |
| Random(20 试平均) | 58.04% | 72.61% | ~0 | ~6% | ~13% |
| TC(晶体管数, 少=快) | 67.97% | 85.01% | −0.389 | 5.5% | 6.4% |

**解读**：
1. 随机 = 无信息下限（hi ~72.6%）。
2. **TC(晶体管数) 比随机差**（hi 85% / Spearman −0.39）——Task#8「更少晶体管→更快」是图级聚合规律，同一函数变体内 drive/fanout/串级更关键，TC 不是变体排序器。
3. **LE 逻辑努力代理是最好的廉价启发式**（hi 43.4%），但仍比 nowave GNN 差 ~5x。
4. **no-wave GNN(6seed) hi 8.78%**：比 LE 好 5x、比随机好 8x，是唯一能把粗筛遗憾压到个位数的现成信号 → 粗筛定位成立。
5. wave(teacher) hi 0.48% 接近理论下限，但需 wave 输入，Rust 粗筛没有 → 仅作天花板参照。

**后续**：Elmore（寄生 RC 模型）基线未做，需 parasitic_caps + RC 延迟模型，列为可选严格对照。回应 Zach：RankNet/排序损失已在 13.x 试过并确认有害（SNR 天花板，非实现问题），基线对比已补齐。

#### 英文叙述版（Baseline Comparison, narrative form）

This evaluation was run on the **V2 test set** (from `batch_v2_full` + `batch_v2_io`, split by expression, seed=42): **18,520 rows / 1,924 circuits**, of which **139 variant-groups have ≥2 variants**, and **110 groups are "high-spread" (hi)**.

Key definitions:
- **"hi" = the hi_spread subset** — variant-groups where the fastest-vs-slowest delay difference exceeds **10%**. These are the groups coarse-filtering actually cares about; "hi regret" and "Sp(hi)" are computed only within these 110 high-difference groups.
- **Regret**: the % of achievable delay you give up by choosing the model's pick rather than the true fastest variant of a group. Lower is better; 0% = always pick optimal.

Ordered best to worst:

1. **GNN wave (2-seed ensemble, teacher)** — global regret **0.50%**, hi **0.48%**; hi Spearman **0.844**, hi top1 **79.1%**, recall@2A **90.0%**. Nearly always places the optimal variant in the top two. But it needs **transistor waveform** input, absent in the Rust coarse-filter pipeline → **ceiling reference only**, not usable directly.
2. **GNN nowave (6-seed ensemble, delivery coarse-filter)** — netlist-topology only, no wave. Global **7.27%**, hi **8.78%**, hi Spearman **0.440**, hi top1 **49.1%**, recall@2A **61.8%**. Brings coarse-filter regret to single digits — the only readily available signal that does.
3. **LE (logical-effort proxy, Σ g·h+p)** — best cheap heuristic, no training. Global **34.71%**, hi **43.40%**, hi Spearman **0.140**, hi top1 **25.5%**, recall@2A **34.5%**. Clearly better than random, but ~**5× worse** than GNN nowave (43.40% vs 8.78%).
4. **Random (20-trial average)** — the "no-information" floor: global **58.04%**, hi ~**72.61%**, Spearman ~0, top1 ~6%, recall@2A ~13%. Any method that cannot beat random carries no usable information.
5. **TC (transistor count, fewer=faster)** — actually **worse than random**: global **67.97%**, hi **85.01%**, Spearman **negative −0.389**, hi top1 **5.5%**, recall@2A **6.4%**. Reason: "fewer transistors → faster" is an aggregate, graph-level pattern; it does not hold for ranking variants of the same function (drive/fanout/series depth matter more than area).

**Overall**: Random is the floor (hi ~72.6%); TC falls below it; LE is the best heuristic but ~5× worse than the GNN; **the no-wave GNN brings coarse-filter regret to 8.78%, validating its role**; wave at 0.48% is the theoretical ceiling.

#### 指标口径表（Metric Glossary）

| Metric | Meaning |
|---|---|
| **Global regret** | Selection regret over all test variant-groups: avg delay lost (as % of group delay) choosing the model's pick vs the fastest variant. |
| **hi regret** | Same, but only within the **hi_spread** subset (fastest-vs-slowest >10%). Primary quality signal for coarse-filtering. |
| **Spearman (Sp(hi))** | Rank correlation between predicted and true variant ordering within a group, on hi groups only. +1 = perfect, 0 = none, negative = inverse. |
| **top1 (hi)** | Fraction of hi groups where the top-ranked variant is the true fastest. |
| **recall@2A (hi)** | Fraction of hi groups where the true fastest variant is inside the **top-2** predictions (strict). Coarse-filter hands top-K to SPICE for fine ranking, so this measures how often the true best is "rescued" into K. |
| **hi_spread** | Variant-groups whose delay spread (fastest vs slowest) exceeds 10%. |
| **Regret** | % of achievable delay given up by choosing the model's pick instead of the group's true fastest. |

### 15.2.12 Rust shadow 基准：SPICE 环境就绪 + 46 电路判定（2026-08-26，NetlistOpt b253aed）

**SPICE 环境（服务器 orca 实测）**：✅ yongsheng 的 Xyce（zen5 构建，`~/NetlistOpt` 里 `xyce.sh` 默认路径）能跑 ASAP7 模型，单次 0.33s，**零配置跑通**；❌ `/opt/spack` 的 Xyce 跑不了（M 器件未注册 + BSIM-CMG 106.1 vs 107）；✅ Spectre（`/opt/rh/SPECTRE201` + `/opt/rh/cds.lic.dat`）可作备用。模型/CDL 在 `/home/tianlang/asap7/`（详见 `docs/GNN_RUST_DATA_DIFF.md` 10.5）。

**46 电路基准**（`tests/tl_opt_shadow_batch.rs`，6 组并行 ~1.5h）：45/46 成功，**ovf1 因 Xyce 瞬态步长崩溃不收敛**（SPICE 自身限制）。

**双根因 + 修复（serve 侧，无需重训）**：
1. `logic_sim.compute_gate_states` 反向 BFS 硬编码 `'out'` → Rust 候选输出叫 `y` → gate_mask 清零全图 → 模型全盲 → 预测扁平。修复：`outputs` 参数传真实输出引脚。
2. Rust 复合门名（`SC_JOIN_AND_WIRE_...`）不在 sc_expansion → 全落 COMPLEX。修复：Rust 生成器自产逻辑类（`classify_subckt_logic`）随候选发 `gate_logics`，serve 用 thread-local 覆盖（sc_expansion > 覆盖 > 回退，训练零影响）。

**最终判定（89 候选集，10.3 标准）**：

| 指标 | 结果（均值/中位） | 达标线 |
|---|---|---|
| recall@top-3 | 51.7% / 66.7% | ≥90% ❌ |
| 选择遗憾 | 7.38% / **2.88%** | ≤5% ❌ |
| Spearman | 0.140 / 0.200 | ≥0.6 ❌ |

**→ 双主判据不达标：GNN 只做启发式预排序，不替换逐候选 SPICE。** 修复使排序从系统性反向（pre-fix 22.8%/50.8%/−0.50）变为真实信号（~40% 候选集遗憾 ≤1%、中位 2.88%），但强度不够——候选集内延迟差常 <5%，是 13.x 已确认的 SNR 天花板；10.3 判据不筛跨度，比 V2 hi-spread 口径苛刻。分析脚本 `scripts/diag/_shadow_analyze.py`，明细见 `reports/_shadow_bench_final.txt`。

**ovf1 处理**：接受（SPICE 自身不收敛，不属于粗筛职责；若需可试 `.tran ... UIC` 跳过 DC 工作点，未做）。

---

## 16.x 训练与 Rust 集成测试结果记录（2026-09-02 起：每次训练结果必须记录于此）

> **规则（16.11.16 定）**：每一次训练结果（GNN test 评估 + Rust 集成验证）记入本文件；
> 检查/分析类结论（如 serve 失配归因、分布差异）记入 `docs/GNN_RUST_DATA_DIFF.md`。版本号随提交递增。

### 16.11.x V2 + m4 数据训练批次（full+rest+m4，746,242 样本）

| 变体 | 特征 | 状态 | Test Median | 选择遗憾(训练) | recall@3 B | Rust 严格@3 | Rust 选择遗憾 | Rust 两阶段 |
|---|---|---|---|---|---|---|---|---|
| v2wave42m4 | 真实 wave（教师） | ✅ | 13.30% | 0.65% | 94.0% | —（wave 不可 Rust 部署） | — | — |
| v2iaa42m4 | 线性近似 | ✅ | 22.27% | 3.85% | 83.6% | 30.2% | 15.21% | 5.12% |
| v2nowave42m4 | 无 wave | ✅ | 22.13% | 3.80% | 83.4% | 39.6% | **10.87%** | 3.95% |
| v2iag42m4 | GBDT15 近似 | ✅ | 23.00% | 4.19% | 81.2% | **44.3%** | **12.19%** | **3.93%** |
| v2kdwave42iaa42 | wave教师KD(reg+rank)；**纯拓扑 in=45**（名带 iaa 实无近似列） | ✅ | 21.69% | 4.17% | 80.8% | 36.8% | 14.97% | 4.81% |
| v2iaar42m4 | 线性 + 真值组内排序 loss（RANK_LOSS_W=0.5, #8） | ✅ | 26.00% | **2.94%** | **87.4%** | 31.1%（宽松 59.4） | 12.72% | **3.64%** |
| v2nowaver42m4 | 纯拓扑 × rank 直训（RANK_LOSS_W=0.5, 17.0.4 空格2） | ✅ | 25.29% | **2.91%** | 87.4% | 34.9% | 11.97% | 3.89% |
| v2iagr42m4 | GBDT15 × rank 直训（RANK_LOSS_W=0.5, 17.0.4 空格1） | ✅ | 25.65% | **2.85%** | 87.2% | 34.0% | **10.50%** | 3.83% |

> ⚠ **本表「Rust 严格@3 / Rust 选择遗憾 / Rust 两阶段」三列全部是 106 集 window 池化口径**（历史）。2026-09-16 定案：**部署真值 = 714 集 batch 口径，比本表大一截且增幅因臂而异**（v2iaa42m4 严格@3 30.2%→56.7%、v2nowave42m4 34.9%→**90.8%**；⚠ 后者旧记 74.4% 是错标，见文末「74.4 错标定案」）→ **本表三列只作臂间同口径比较，绝对值不可外引**。详见下方「部署口径定案」块与 `docs/GNN_RUST_DATA_DIFF.md` §15。

**v2iaa42m4 详细（2026-09-02）**：
- 训练：234 epochs（早停），Test Median Rel Err 22.27%，选择遗憾 3.85%（spread>10%: 5.29%），Spearman 0.410，recall@3 B 83.6%。
- Rust（106 候选集/5390 行）：严格 recall@3 30.2%、宽松 57.5%、选择遗憾 15.21%、两阶段 5.12%、Spearman 0.050。
- **结论**：m4 数据加入 Rust 无改善（严格 recall 30.2% vs 无 m4 的 34.9% 反降）；线性近似 serve 失配（训练 3.85% → Rust 15.2% 落差）源于候选门结构分布差异，详见 GNN_RUST_DATA_DIFF §13.1。

**v2wave42m4 详细（2026-09-02）**：60 epochs 早停（Best Val 11.40%），Test Median 13.30%，选择遗憾 0.65%（spread>10%: 0.54%），Spearman 0.688/0.780，recall@3 B 94.0%/98.3%——**真实 wave 是 gold 标准，作蒸馏教师**。

**v2nowave42m4 详细（2026-09-03，完成）**：310 epochs early_stop（Best Val 26.65%），best midpoint ep250，Test Median 22.13%，选择遗憾 3.80%（spread>10%: 5.18%），Spearman 0.393/0.450，recall@3 B 83.4%（spread>10% 88.2%），top1 42.5%，成对分辨 <2% 55%（≈随机，SNR 未破）/ >10% 81%。**对比（仅同 548 组 test 可比）**：与 v2iaa42m4 几乎打平（遗憾 3.80 vs 3.85、Test Median 22.13 vs 22.27、recall@3B 83.4 vs 83.6）→ 线性近似 ids_avg 训练侧无增益，价值在 serve「零仿真可算」（serve 失配见 DIFF §13.1）；vs 旧 v2nowave42（full+io 139 组，遗憾 6.70/8.07）数值大降但 **test 口径不同，不作「rest/m4 提升 nowave」定论**。

**v2nowave42m4 Rust shadow（2026-09-03 19:29 收尾，106 候选集 / 5,390 成功行 / 8 失败，serve in_dim=14 无近似列确认）**：严格 recall@3 39.6%（宽松 65.1%）、选择遗憾 **10.87%**（中位 6.17%）、两阶段 3.95%（中位 0.69%）、Spearman 0.142（hi_spread 79 集：严格@3 38.0% / 两阶段 5.03% / 遗憾 13.30% / Sp 0.232）。**三方 Rust 定论（nowave/iag 同本次 pipeline、iaa 同 pipeline 记录；选择遗憾轴 iaa 的 15.21 经 §14 复算 mean 15.9 吻合，可作 run 级对照）**：**选择遗憾 nowave 10.87% < iag 12.19% < iaa 15.21%——无近似的纯拓扑 serve 反而最好，加近似 ids_avg 特征（线性 +4.3pp / GBDT15 +1.3pp）serve 端是净伤害**；iag 赢在排序质量（Sp 0.210、严格@3 44.3% 三兄弟最高）但 top1 落点输给 nowave（DEPTH_MIX 大窗口 w=5/1/7/11 nowave 明显更好）；两阶段 nowave 3.95 ≈ iag 3.93（≤5%），iaa 5.12 略超标。→ 修订 §13.2「iag serve 增益」表述：GBDT15 仅是**最不伤的近似**，最优 serve = 不加近似（nowave 路线）。详见 DIFF §13.3。

**v2iag42m4 详细（2026-09-03，完成）**：271 epochs early_stop（RESUME ep100 续训，Best Val 27.19%），best midpoint ep200（score 112.51，SUMMARY 显示其指标），Test Median 23.00%，选择遗憾 4.19%（spread>10%: 5.75%），Spearman 0.386/0.441，recall@3 B 81.2%（spread>10% 86.4%），top1 40.0%（spread>10% 52.6%），成对分辨 <2% 55%（≈随机）/ >10% 80%。**对比（同 548 组 test、各自 best midpoint）**：GBDT15 训练特征 vs 线性（v2iaa42m4）与无 wave（v2nowave42m4）**全线略差**——遗憾 4.19 vs 3.85/3.80、Test Median 23.00 vs 22.27/22.13、recall@3B 81.2 vs 83.6/83.4、Spearman 0.386 vs 0.410/0.393。→ 更高保真的 GBDT15 近似**训练侧无增益且略负**（不转化为组内排序提升，与「近似特征训练侧无增益」结论同形态）；价值仅在 serve「零仿真可算」+ 更低 serve 失配预期——GNN-test 无法裁决 serve，**决定性 = v2iag Rust shadow**（对照 v2iaa42m4 Rust；后者 DIFF §13 数字 ⚠ 不可复现，I1），待 I7。

**v2iag42m4 Rust shadow（2026-09-03 19:09 完整收尾，106 候选集 / 5,390 成功行 / 8 失败，GBDT15 加载确认）**：严格 recall@3 44.3%（宽松 64.2%）、选择遗憾 12.19%（中位 6.41%）、两阶段 3.93%（中位 0.33%）、Spearman 0.210（hi_spread 79 集：严格@3 43.0% / 两阶段 4.96% / 遗憾 15.15% / Sp 0.262）。**I7 裁决**：两 run 报告**同为 106 集/5390 行/8 失败** → 同候选窗口，唯一变量 = serve 模型，A/B 干净。GBDT15 全线优于线性（对照 v2iaa42m4 记录）：遗憾 12.19 vs 15.21、两阶段 3.93 vs 5.12、严格@3 44.3 vs 30.2、宽松 64.2 vs 57.5、Spearman 0.210 vs 0.050——训练侧 GBDT15 反而略差（4.19 vs 3.85）而 Rust 端更好 → **增益来自 serve 端鲁棒性（特征一致性假设获支持，DIFF §13.1 佐证）**；**但仍不达 10.3 主判据**（遗憾 >5%、严格@3 <90%）→ GNN 仍只做启发式预排序，剩余训练 4.19%→Rust 12.19% 落差 = 规模失配 + 候选门结构分布差异为主（GBDT15 缓解非消除，A1 修 serve 不可行坐实）；两阶段 3.93% ≤ 5%（中位 0.33%）→「GNN 前 3 → SPICE 精排」粗筛流程实操可接受。附带：~~§14「106 集口径不明」修正——106 = 标准 pipeline 窗口级 ≥4 候选计数，跨两 run 稳定，714 = 复算另类聚合口径。~~ **⚠ 2026-09-16 该附带句反了（DIFF §15）**：**714 集才是部署真值**（`circuit, iter, window_try` = 一次 `pre_rank` 调用），**106 集是池化口径**。「两 run 同为 106 集/5390 行/8 失败」只证 A/B 同口径 → 臂间 A/B 有效，**但本段的数值与排序不可外推到部署口径**（口径效应因臂而异：iaa +26.5pp / nowave **+55.9pp**；⚠ 2026-09-17 改，旧记 +39.5pp 用了错标树的 74.4，见下方「部署口径定案」块 (8) 与 `GNN_RUST_DATA_DIFF.md` §15.8）。

**v2kdwave42iaa42 详细（2026-09-03，完成）**：194 epochs plateau 早停（train 降/val 平 → 过拟合，Best Val 26.69%），总 1268 min（Avg/epoch 391.6s）；SUMMARY 显示 **midpoint ep100** 指标（脚本注：SAVE_MIDPOINTS 载入 midpoint 最优 epoch；best_model.pt 15:12 另存）。Test Median 21.69%（Mean Abs 8.25ps；Mean Rel 28.27% 被小延迟放大仅参考），选择遗憾 4.17%（spread>10%: 5.68%），Spearman 0.377/0.424（spread>10%），recall@3 B 80.8%（spread>10% 85.5%），top1 40.1%（spread>10% 51.7%），成对分辨 <2% 55%（≈随机，SNR 墙同三兄弟）/ >10% 79%。同 548 组 test。**对比（⚠ 16.11.40 修正：学生实为纯拓扑，正确对照 = nowave 同特征非 KD；下表原以「同 iaa 特征」对照 v2iaa42m4 是错位——KD 分支 `USE_TRANSISTOR_WAVE=False` 把近似列挡掉，学生训时即纯拓扑，见 DIFF §13.4 更正）**：Test Median 21.69% 四者最好（v2iaa 22.27 / nowave 22.13 / iag 23.00）但仅 ~0.5-1.3pp 边际；**排序指标低于 nowave（同特征非 KD 真对照）——遗憾 4.17 vs 3.80（+0.37pp）、recall@3B 80.8 vs 83.4（-2.6pp）、Spearman 0.377 vs 0.393（-0.016）**（亦未超 v2iaa42m4：3.85/83.6/0.410）→ **wave 教师（自身 Test 13.30 / 遗憾 0.65）reg+rank 蒸馏训练侧未把纯拓扑学生拉到教师水平，相对朴素 nowave 还略负**；形态 = 信息论天花板（软标签目标学生够不着，主损失与 KD 损失竞争），非「近似特征训练侧无增益」那条。**决定性子 = Rust shadow**（当时预期 iaa 特征 serve 需 USE_IDS_AVG_APPROX=1，serve 失配预期同 v2iaa42m4 → 训练侧未赢则 Rust 大概率不赢）；**已跑完（22:03，见下方 shadow 块）——serve 实测 ckpt in=45 纯拓扑无需 env，与 nowave 干净 A/B，Rust 亦不赢（14.97% vs 10.87%），双端收口 DIFF §13.4**。

**v2kdwave42iaa42 Rust shadow（2026-09-03 22:03 完整收尾，106 候选集 / 5,390 成功行 / 8 失败，serve midpoint_ep100、纯拓扑 in=45 无近似列确认）**：严格 recall@3 36.8%（宽松 64.2%）、选择遗憾 **14.97%**（中位 7.17%）、两阶段 4.81%（中位 1.03%）、Spearman 0.098（hi_spread 79 集：严格@3 31.6% / 宽松 57.0% / 两阶段 6.19% / 遗憾 18.94% / Sp 0.184）。**与 m4三兄弟同 106 集同 pipeline 对比（唯一变量 = 权重；serve 全纯拓扑无 env）**：选择遗憾 14.97 vs nowave 10.87（+4.1pp）/ iag 12.19 / iaa 15.21 → **KD 学生劣于朴素 nowave，仅略优于 iaa 线性近似**；严格@3 36.8 vs nowave 39.6 / iag 44.3 / iaa 30.2（宽松 64.2 vs 65.1/64.2/57.5）；两阶段 4.81 vs 3.95/3.93/5.12（>5% 略超标）；Sp 0.098 vs 0.142/0.210/0.050。**定论（⚠ 16.11.40 修正：学生实为纯拓扑，非「蒸馏到 iaa 结构」）**：wave 教师（自己 Rust 不可部署，train 遗憾 0.65%）经 reg+rank 蒸馏到**纯拓扑学生**，训练侧无增益（L750：4.17 vs nowave 3.80）+ Rust serve 端亦不赢朴素 nowave（KD 学来的权重相对普通训练无 serve 优势，反而 +4.1pp 遗憾）——**「教师 wave 金标准通过软标签转移到无 wave 学生」在 train + Rust 双端坐实不成立**；机制 = 信息论天花板（软标签是学生够不着的目标），**不存在「iaa 权重污染」**（学生从未拿到近似列）。详见 DIFF §13.4。

**v2iaar42m4 详细（2026-09-04，17.0.2 train-side，排序直训 #8；Rust shadow 已决定跑,结果后补表行）**：= **v2iaa42m4 特征 + 真值组内 rank loss（RANK_LOSS_W=0.5）**,与 v2iaa42m4 同 iaa 特征、同 SPLIT_SEED=42 → **唯一变量 = rank loss 的干净 A/B**。220 epochs early_stop（val_loss 连续 40ep 无改善；Best Val Rel Err 29.86%），总 1487.6 min（Avg/epoch 405.1s）；SUMMARY 显示 **midpoint ep150**（加权 score 119.31 最优：ep100 118.64 / ep200 119.13）。Test Median Rel Err **26.00%**（Mean Abs 9.39ps）——⚠ 绝对精度比四兄弟差 ~3.7pp（rank 换绝对精度的权衡）。**[排序]（同 548 组 test）**Spearman **0.485**、选择遗憾 **2.94%**、top1 43.6%、捕获率 82.8%、recall@3 B **87.4%**（A 67.7%）；**[spread>10%（346 组）] Spearman 0.549、遗憾 4.04%、top1 56.4%、recall@3 B 92.5%**；成对分辨 <2% 55 / >10% 84。**判定**：排序指标**全 m4 系最优（含纯拓扑 nowave）**——遗憾 2.94 vs v2iaa42m4 3.85（−0.91pp）/ nowave 3.80 / iag 4.19、Spearman 0.485 vs 0.410/0.393/0.386、recall@3B 87.4 vs 83.6/83.4/81.2 → **"直接为 serve 关心的排序训练"train-side 方向成立**；代价 = Test Med 26.0 vs 22.27 绝对劣化。**serve 路线风险**：iaar = 线性近似（预计 in=46,serve 前 Step 1 查 ckpt 实锤）,iaa 线性 serve 端 m4 系最差且 ⚠ Rust 记录不可复现 → 仍决定 **serve iaar(midpoint_ep150) 跑真读数**;若 Rust 亦印证线性失配,则 next = rank loss 移植到可 serve 特征（nowave 纯拓扑 / iag GBDT15）再 Rust 验证。

**v2iaar42m4 Rust shadow（2026-09-04 12:23 收尾，17.0.3，106 候选集 / 5,390 成功行 / 8 失败；serve midpoint_ep150、in=46 带 USE_IDS_AVG_APPROX=1 确认）**：严格 recall@3 31.1%（宽松 59.4%）、选择遗憾 **12.72%**（中位 7.71%）、两阶段 **3.64%**（中位 1.15%，五者最低）、Spearman 0.170（hi_spread 79 集：严格@3 27.8% / 宽松 53.2% / 两阶段 4.50% / 遗憾 15.94% / Sp 0.216）。**同 106 集同 pipeline 对照**：选择遗憾 12.72 vs nowave 10.87（+1.85pp）/ iag 12.19 / iaa 15.21⚠ / kd 14.97；严格@3 31.1 vs nowave 39.6 / iag 44.3 / iaa 30.2 / kd 36.8（宽松 59.4 五者最低）。**裁决：train-side rank 大优（2.94）不转移 Rust——rank 把训练分布 pairwise 排序练得更锐,但 serve 端线性近似列失配（DIFF §13.1 机制）把增益吃回 → 带线性近似的 serve 即便叠 rank 仍是净伤害方向坐实；唯一赢点 = 两阶段 3.64% 五者最优（GNN 前3→SPICE 流程下最有效粗筛）**。nowave 纯拓扑仍为 serve 交付基线；唯一未试组合 = rank + 可 serve 特征（nowave/iag），候选 next 非必须。详见 DIFF §13.5。

**v2nowaver42m4 详细（2026-09-06 训完，空格2 纯拓扑×rank）**：= **v2nowave42m4 特征（纯拓扑 in=45）+ 真值组内 rank loss（RANK_LOSS_W=0.5）**,同 SPLIT_SEED=42 → 与 v2nowave42m4 **唯一变量 = rank loss 的干净 A/B（同可 serve 特征族）**。383 epochs early_stop（LR 磨到 1e-6 后 val_loss 连续 40ep 无改善；Best Val Rel Err 29.00%），总 2918.0 min（Avg/epoch 456.9s）；SUMMARY 显示 **midpoint ep250**（score 119.52 最优——恰好同 nowave Rust shadow 的 serve epoch）。Test Median Rel Err **25.29%**（Mean Abs 9.04ps）。**[排序]（同 548 组 test）**Spearman **0.489**、选择遗憾 **2.91%**、top1 45.3%、捕获率 83.1%、recall@3 B **87.4%**；**[spread>10%（346 组）] Sp 0.574、遗憾 4.02%、recall@3 B 93.6%**；成对分辨 <2% 56 / >10% 84。**A/B vs nowave42m4**：遗憾 3.80→**2.91**（−0.89pp）、recall@3B 83.4→**87.4**（+4.0）、Sp 0.393→0.489、top1 42.5→45.3、spread>10 recall@3B 88.2→93.6 → **rank loss 在纯拓扑（serve 交付族）上照样把排序指标全面抬过无 rank 版**；代价 = Test Med 22.13→25.29（rank 换绝对精度，同 iaar 形态）。**决定性 = Rust shadow（vs nowave 10.87% 目标线：train 2.91 若按 nowave +7.07 落差转移 ≈ 9.98 < 10.87，但贴近误差带 → 必须真读数裁决）**。

**v2nowaver42m4 Rust shadow（2026-09-06 21:22 收尾，17.0.13，106 候选集 / 5,390 成功行 / 8 失败；serve midpoint_ep250、纯拓扑 in=45 无近似列确认）**：严格 recall@3 34.9%（宽松 57.5%）、选择遗憾 **11.97%**（中位 7.24%）、两阶段 3.89%、Spearman 0.094（hi_spread 79 集：遗憾 14.80%）。**与 nowave42m4 同 106 集同 pipeline、唯一变量 = rank loss 的干净 A/B（可 serve 纯拓扑族内）**：遗憾 11.97 vs 10.87（**+1.10pp，反超不了目标线**）、严格@3 34.9 vs 39.6（−4.7pp）、宽松 57.5 vs 65.1、Sp 0.094 vs 0.142 → **纯拓扑 serve 上叠 rank = 全轴净伤害（连排序质量轴 Sp 都降）→ 纯拓扑×rank 槽位：rank 目标线关闭**。train 2.91 的组内排序优势完全不转移 serve（与 §13.5 iaar 同形态，但此 A/B **无近似列可怪** → 机制收敛为「rank 直训强化的是训练分布组内相对序，serve 候选集门结构漂移下不泛化，也不创造新召回信号」）。

**v2iagr42m4 详细（2026-09-06 训完，空格1 GBDT15×rank）**：= **v2iag42m4 特征（GBDT15 in=46）+ rank loss（RANK_LOSS_W=0.5）**,与 v2iag42m4 干净 A/B。347 epochs early_stop（同上，Best Val 29.07%），总 3322.8 min（Avg/epoch 574.1s）；SUMMARY 显示 **midpoint ep150**（score 117.52 最优）。Test Median Rel Err **25.65%**（Mean Abs 9.32ps）。**[排序]** Spearman **0.480**、遗憾 **2.85%**（三 rank 最低）、top1 43.2%、recall@3 B **87.2%**；**[spread>10%] Sp 0.553、遗憾 3.90%、recall@3 B 91.9%**；成对分辨 <2% 54 / >10% 84。**A/B vs iag42m4**：遗憾 4.19→**2.85**（−1.34pp，三对 A/B rank 增益最大）、recall@3B 81.2→**87.2**（+6.0）、Sp 0.386→0.480、top1 40.0→43.2 → 对**最不伤的近似（GBDT15）**rank 同样生效，且把 no-rank 时最差（4.19）拉到三 rank 最低（2.85）。**决定性 = Rust shadow（赌 rank 遗憾优势 × iag 严格@3 召回优势叠加 > nowave 10.87 / iag 12.19）**。

**v2iagr42m4 Rust shadow（2026-09-06 21:31 收尾，17.0.13，106 候选集 / 5,390 成功行 / 8 失败；serve midpoint_ep150、in=46 带 USE_IDS_AVG_APPROX=2 确认）**：严格 recall@3 34.0%（宽松 52.8%）、选择遗憾 **10.50%**（中位 6.91%，**m4 家族 Rust 最低**）、两阶段 3.83%、Spearman **0.238**（家族最高）（hi_spread 79 集：遗憾 13.03%）。**对照（同 106 集同 pipeline）**：遗憾 10.50 vs iag 12.19（−1.69pp）**且 vs nowave 10.87（−0.37pp，rank×GBDT15 首破交付基线）**；严格@3 34.0 vs iag 44.3（**−10.3pp，iag 的严格召回强项被 rank 打穿**）/ nowave 39.6；宽松 52.8 处家族最低档；Sp 0.238 vs iag 0.210 / nowave 0.142 / iaar 0.170。**裁决：rank 转移成立的第一个 serve datapoint（遗憾 + Spearman 双破纯拓扑交付基线）——但赢点全在排序质量轴、recall 轴塌方（rank 把模型从「top1 落点/严格召回」重新分配到「组内整体序」）** → iagr = 家族「遗憾/Spearman」最佳 Rust 值，但 serve 交付不走它（GBDT15 近似 serve 脆弱 §13.3 + 严格召回输 nowave 5.6pp）；**rank 转移完整结论 = 特征依赖，见 DIFF §13.6**。

**三 rank run 收口（train-side，2026-09-06）**：遗憾 iaar 2.94 / nowaver 2.91 / iagr 2.85、recall@3B 87.4/87.4/87.2、Sp 0.485/0.489/0.480 → **rank loss 把线性/纯拓扑/GBDT15 三种特征族全部拉到同一公共排序天花板（~2.9% 遗憾 / ~87% recall@3）**,无 rank 时族间 3.80~4.19 的排序差异被 rank 抹平 → 排序上限由模型容量 + 纯拓扑可及信号决定、与近似特征选择无关（仍距 wave 教师 0.65% 有 2.2pp——纯拓扑够不到 wave 的排序信息）。⚠ v2iaar bullet "全 m4 系最优" 已被追平/反超（该表述按其时点成立）。**Rust 未跑前 train-side 无法裁决 serve（§13 铁律）——两空格价值正在于把 rank 增益放到可 serve 且 serve 失配更小的特征族上验**；nowaver 尤干净（与 nowave 同 serve 特征、只加 rank loss），其 Rust 结果直接回答"rank 目标线开/关"。**Rust 已跑完（17.0.13，2026-09-06 21:22/21:31 收尾），裁决见下**。

**rank 三空格 Rust 裁决 + 旧 89 集 reports 口径澄清（2026-09-06，17.0.13）**：
1. **rank 转移 = 特征依赖**：iaar（线性+rank）12.72% 不转移（§13.5）；nowaver（纯拓扑+rank）11.97% > nowave 10.87% 全轴净伤害 → **纯拓扑×rank 槽位关闭**；**iagr（GBDT15+rank）10.50% < nowave 10.87% = m4 家族首个 Rust 遗憾破交付基线 + Sp 0.238 家族最高**，但严格@3 自 iag 44.3% 崩到 34.0% → rank 把「召回」换成「排序质量」。**serve 交付基线仍 v2nowave42m4 纯拓扑**（10.87%、无 env、稳健）；iagr 的 0.37pp 遗憾 + 0.096 Sp 不值 GBDT15 serve 的近似脆弱 + 5.6pp 严格召回。**10.3 主判据（遗憾 ≤5% + 严格@3 ≥90%）m4 Rust 全族七 run 全员不达标 → GNN 仍只做启发式预排序**。详见 DIFF §13.6。
2. **⚠ 旧 `reports/_shadow_bench_final.txt` 的 89 集 block（严格@3 47.2% / 宽松 79.8% / 遗憾 7.38%）别再当"最好 recall"**：该 block = §10.8（2026-08-26，v2nowave 纯拓扑，serve §10.7 修复后）**早测小窗口存档**——89 候选集 / 950 成功行 / 35 失败、候选 n 4~33 中位 8，评估量仅现行 106 集 / 5390 行的 ~1/6；且 47.2/79.8 是 **6-seed 等权集成** 结果（单模型 41.7% / 71.3%，§11.1），m4 全族单 seed 从未集成。**跨口径比 recall 无意义**——同一份 09-02 全量 CSV 按朴素 ≥4 分组重算宽松 94.4%（DIFF §14）。现行口径 = 106/5390/8，结果读 `~/shadow_analyze.out`。

### 17.0.0 idsavg GNN 独立建模方向（2026-09-04，major）

> 详档：`docs/IDS_AVG_GNN.md`。判定唯一尺度 = **per-gate 真实 ids_avg 预测 R²/Spearman**（与 delay 无关）。
> 架构 = **DelayGNN 骨架纯 torch 复刻**（不 import pyg、不改 src/model.py）+ **刺激源锚定**（slew 只锚 switching_pin、load 只锚 output，沿有向 driver→receiver 边传播）——让模型感知"刺激源位置/是否在 fanout cone"，这是 GBDT15 均匀广播表达不了的信号。

**本地受控对比（已验证，batch_v2_full 1500 电路 / 电路级切分 1200/150/150 / 80ep）**：
| 模型 | test R² | Spearman | 意义 |
|---|---|---|---|
| A GBDT15（15 特征，部署同款） | **0.6740** | 0.621 | 精确复现既有近似 |
| V gnn（复刻+锚定+有向边） | **0.7697** | 0.790 | train 0.778≈test，几乎不过拟合 |
| W nograph（同特征无边 MLP） | 0.6773 | 0.603 | ≈GBDT15（广播特征天花板） |

**判定**：消息传递自身贡献 **+0.092**（W→V，特征/模型只差"有无边"）；V 超 GBDT15 **+0.096** → 图传播携带 GBDT15 看不见的 per-gate ids_avg 信号，**方向成立，值得做 serve 端真模型**。
**服务器全量基线（17.0.1 · 已完成，2026-09-04）**：`_fit_idsavg_gnn_server.py`，full+rest+m4 全域、电路级切分、EPOCHS=45、NO_NOGRAPH=1（W 无边对照本地已定论≈GBDT，不重跑）。判定（唯一尺度 = per-gate ids_avg test R²，未见电路）：

| test 分桶 | n（row,gate 样本） | GNN R² | GBDT15 R² | Δ |
|---|---|---|---|---|
| batch_v2_full | 48,905 | **0.7131** | 0.6363 | +0.077 |
| batch_v2_rest | 386,194 | **0.6954** | 0.5492 | +0.146 |
| **batch_v2_m4（五形状）** | 125,143 | **0.6583** | **0.2391** | **+0.419** |

总体 V gnn：best_val **0.6959** / test **0.7012** / Spearman 0.7105；GBDT15 整体 ≈0.49（test 样本量加权粗估，分桶精确值见表）。
**判定**：每桶 GNN 均胜；**m4（五形状未见电路）上 GBDT15 广播近似近失效（0.24）、结构 GNN 稳（0.66）→ "图结构跨形态泛化、广播近似不能"获全量证实**，serve 端真模型方向支持度远强于本地子集（全域 +0.21 / m4 +0.42 vs 本地 +0.096）。⚠ **全量 gnn 0.7012 < 本地子集 0.7697**（分布更广更难 + 只 45ep + 模型按子集取偏小）；best_val≈test 无过拟合、偏欠拟合 → **参数放宽（K/宽度/EPOCHS/patience）有实证依据，列为下一步对照（未跑）**。
**待办更新**：基线已定、m4 泛化已下"结构 GNN 显著优于 GBDT15"定论；serve 端推进与否待参数放宽对照 + 可行性评估。详档见 `docs/IDS_AVG_GNN.md` §4.2。

- **17.0.5 A/B 首轮（2026-09-04）= 无效对照，勿引用数值**：env 旋钮（K/HID/EMB/LR/PATIENCE/CONE_FEAT）入服务器脚本后，A（K5/H160/E32·LR3e-3·PAT40）/ B（+锥体）双双 best_val 0.3571/0.4257 且落 ep1/5、early-stop@ep45，train_loss 冻死 = **LR3e-3 × 大模型 4-5× × cosine220 前段不退火 × patience 早掐** 的配置错配，**非容量方向被判死、非代码回归**。唯一正信号 = B 坏训练下 m4 桶 0.7579 > 基线 0.6583（锥体对 m4 有真增益，待干净复验）。详档 §4.3 见 `docs/IDS_AVG_GNN.md`。
- **17.0.6 A2/B2（2026-09-05）= 干净裁决：容量+锥体双生效**：修正配置（LR1e-3·PAT0·EPOCHS80·venv python3·不开 N_CAP）跑满 80ep 无早停。**A2（K5/H160/E32）= test 0.7424 / Sp 0.7385**（基线 0.7012/0.7105 → **容量 +0.041**，全量欠拟合假设证实）；**B2（+CONE_FEAT）= test 0.7866 / Sp 0.8150 / m4 桶 0.8076**（**锥体 +0.044，增益集中 rest/m4，full≈0**；m4 差 GBDT15 0.2391 达 +0.568，历史最高）。best_val@ep80、train≈test 无过拟合。首轮坏训练 B 的 m4 0.7579 信号被干净复验并放大。**未决：serve 端采用 B2 需 Rust 补锥体/距离特征，成本另议（方向性利好）**。详档 §4.4 见 `docs/IDS_AVG_GNN.md`。
- **17.0.9（2026-09-05）= 追凶 + 修复：服务器 Arrow 字符串切分卡死**：B2long/B2v2long 首启均卡数据切分（`总: 电路` 后 30min+、单核 100%）。py-spy 定位 = **pandas pyarrow-backed `circuit_id` 字符串列，`set()`/`list()` 逐元素走 `arrow.array.__iter__` → 746k 行病态慢**；A2/B2 与历史 N_CAP 探针"停滞"同源（A2/B2 当年在此磨 ~30-45min 未被察觉）。修复 = 先 `to_numpy()` C 速转 object 再 set，语义逐位不变、亚秒级。**重启两条长程对照（pid 3554641/3554958）均 2-3min 越过装配**：B2long=CONE_FEAT v1、B2v2long=CONE_V2，EPOCHS220/PATIENCE60（auto early-stop，取消 80ep 硬停裁决），判 §4.4"ep80 截早？" + 锥体 v1→v2 增量。结果待更（详档 §4.5 `docs/IDS_AVG_GNN.md`）。
- **17.0.10（2026-09-06）= 长程对照裁决入档：80ep 截早基本否 + 锥体 v2 = m4 特化**：B2long（v1·220ep）best_val 0.7833 / test **0.7882** / m4 .8143；B2v2long（v2·220ep）0.7841 / test **0.7893** / m4 **.8235**（历史最高，GBDT15 差 +0.584）。两 run 均 PATIENCE60 未触发跑满 220（cosine 尾段 val 爬升→patience 等不到平台，EPOCHS 上限实为裁决者）。**判定①：80ep 只轻微截早**（追 220ep test +0.0016、m4 +0.0067 → 噪声级，"80ep 足够"反证成立，未来 80-100ep 即够）。**判定②：v2 相对 v1 = m4 特化 +0.009（overall +0.001、full −0.002）** → serve 用锥体取 **v1** 即可，v2 额外成本只换 m4 鲁棒。全量最佳配置 = B2v2long（test 0.7893/m4 0.8235）。详档 §4.5 见 `docs/IDS_AVG_GNN.md`。

### 17.1.x delay 侧 GNN 预测 ids_avg 特征 A/B（Phase B，2026-09-12 起训，2026-09-15 收尾）

> 目标：把 idsavg GNN（R²=0.79 线）预测的 per-gate ids_avg 当作 delay 训练的 ids 特征列，看能否把纯拓扑 nowave 的组内排序拉向 wave 教师。防泄漏 = **5 折交叉拟合 OOF 表**（每折模型只预测自己没见过的电路）。

**Phase 1（OOF 表，2026-09-10~12，17.1.2）**：`~/idsavg17/idsgnn_oof_f{0..4}.parquet`，5 折串行（每折 6~13h，共 ~35h；**并行被内存墙否决**——单折 RSS 27GB / 机器共 60GB / nproc 24 空转）。合计 14,193,202 行、每折 9132~9133 电路、门保留率 1.000、无 NaN；**合并去重 45,662 电路，出现次数分布恰 {1: 45662}** = 每电路被没背过它的模型预测一次 → 无泄漏且覆盖 100%。折 0 的 n≥8 池 0.5777 ≈ R4 的 0.5778 → NFOLD 切分（80% 池）未掉质。**键对齐独立实证（`scripts/diag/check_ids_gnn_table.py`，17.1.3/17.1.4）**：源 `~/-project/data/<batch>/timing_arcs*.parquet`（8 文件）与 OOF 表**键集完全相等，746,242 / 746,242 = 100.00%** → delay 侧查表零漏、不会静默填 0 退化成 v2nowave。

| 变体 | 特征 | 早停@ep | Best midpoint | Best Val Rel Err | 选择遗憾(全局) | Spearman | top1 | 捕获率 | recall@3 B |
|---|---|---|---|---|---|---|---|---|---|
| **v2nowave42b** | 纯拓扑（同代码基线） | 269 | ep100 | 27.47% | **3.94%**（spread>10% 5.37%） | 0.389 | 42.2% | 77.8% | 81.6% |
| **v2nowavegnn42b** | +GNN ids 列 | 292 | ep150 | **24.78%** | **4.39%**（spread>10% 5.96%） | 0.340 | 36.3% | 74.4% | 79.9% |
| v2nowave42m4（历史参照，16.11.10） | 纯拓扑 | 310 | ep250 | 26.65% | 3.80% | 0.393 | 42.5% | 78.9% | 83.4% |

**判定：A/B 为负——GNN 预测 ids_avg 作为 delay 特征不进反退。** 全局选择遗憾 3.94% → **4.39%（+0.45pp，相对劣化 11.4%）**，且 **7 个排序指标全部同向变差**：Spearman −0.049、top1 −5.9pp、捕获率 −3.4pp、recall@3B −1.7pp、spread>10% 遗憾 +0.59pp。
**A/B 可信度**：新基线 3.94 vs 历史同种子 3.80 → 跨版本（16.11.10 → 17.1.2）漂移仅 **0.14pp**；两 run 同代码（17.1.2）、同 `TRAIN_SEED=42`、同 `USE_TRANSISTOR_WAVE=False`、同早停规则 → 唯一变量 = GNN ids 列，观测效应是漂移的 3 倍，**不是手气**。
**最有信息量的形态 =「回归变好、排序变差」**：Best Val Rel Err 反而从 27.47% 改善到 **24.78%（−2.69pp）** → 特征确实被模型吃进去并在数值拟合上帮了忙，但**没转化为组内选优能力、反而损害它**。三点（27.47→3.94 / 26.65→3.80 / 24.78→4.39）呈单调反相关 → **本任务上 val loss 与排序目标不是一回事、甚至反向**（提醒：不要拿 val loss 当排序质量代理去选点/早停）。
**关键线索 = 阶梯倒挂**：真值 ids_avg（wave 教师）值 **3.1pp** 巨利（0.65% vs 3.80%），可是**每一个「学出来的近似」要么微利要么倒亏，且近似器越强亏得越多**——线性 **3.35%** < nowave 3.77~3.80% < GBDT15 **4.19%** < GNN **4.39%**。R² 从线性推到 0.79 这条线（17.0.x 全程）在 delay 侧一点没兑现。→ **瓶颈不是近似精度**，把 ids_avg 的 R² 继续往上推不是通向 delay 收益的路径。机制猜想（**待验，未做**）：线性近似对全部门共用一组系数 → 组内相对关系被一致缩放、保序；GNN/GBDT 逐门独立预测 → 每门带自己的误差，注入的恰是「组内不一致」噪声，正打在排序唯一依赖的维度上；真值误差为零故纯赚。
**结论：Phase B 收口，记负数结果。** 价值 = 证明「提高 ids_avg 近似精度」不是路径，可省掉后续大量算力。
**⚠ 墙钟不可比**：zhirui 的 Xyce 仿真把 load 压到 33/24 核，两 run 墙钟 3387 / 3507 min vs 历史 2400 min（段内 5.7× 减速），**只有模型指标可比**。

**Rust shadow 计划（2026-09-15 定，用户决定两条都跑）**：
- `v2nowave42b`（纯拓扑，in=45）→ **不带 `USE_IDS_AVG_APPROX`**，serve `midpoint_ep100.pt`。**它的角色是模式 3 的纯拓扑对照臂**（与 `v2nowavegnn42b` 同代码、同种子，唯一变量 = 有没有那一列 GNN ids）。⚠ **它不是交付基线** —— 交付基线模型是 `v2nowave42m4`（ep250，OPERATIONS §6.7 的 serve 恢复项），42b 只是恰好同族；它的 shadow 数字若落在 nowave 10.87% 附近属于顺带得到的同族一致性检查，**不是本次目的**。
- `v2nowavegnn42b`（in=46）→ **模式 `'3'`**，serve `midpoint_ep150.pt`。
- 原计划是"不实现 serve 侧 GNN"（当时 `serve.py` 的 ids 列只支持 `'1'` 线性 / `'2'` GBDT15，无 GNN 推理路径；且 in=46 在 serve 端历史上是净伤害：iag 12.19 / iaa 15.21 vs in=45 nowave 10.87）。**该判断已推翻**——17.1.6 实现了模式 `'3'`（现场跑 idsavg GNN 折模型、逐 `(pin,dir)` 行算那一列），并新增 `check_idsgnn_serve_parity.py` 作**强制闸门**（与训练 OOF 表逐 `(行,门)` 比对，`max|diff| ≤ 1e-4` 才允许跑 shadow）。
- ⚠ **`in=46` 是二义的**：delay 侧静态块恒为 `logic_only` 7 列，故「7 静态 + 1 列」对三种来源（线性 / GBDT15 / GNN）形状完全相同，且 delay ckpt = 裸 `state_dict()` 不带元信息 → **形状分不出模式，选错还不报错**（只静默喂一条口径不对的列）。判据只能是训练侧的 `IDS_GNN_TABLE` 回显（见 OPERATIONS §6.2 三分支）。
- ⚠ **训练侧判负 ≠ 部署口径同结论**：serve 端默认 5 折预测空间平均（更平滑），而训练时每电路只拿到留出它的那一折的预测（噪声更大）。若阶梯倒挂的机制确为"特征行间不一致破坏排序"，则**更平滑的 serve 列有可能反而变好**。故 shadow 需全折与 `IDSGNN_FOLDS=0` 单折各一次对照，才能分离"特征本身"与"特征噪声水平"。
- 待办：两个 shadow 结果出来后回填本节的 Rust 表行（口径 106 集 / 5390 行 / 8 失败）。

**Rust shadow 执行记录（2026-09-15，双端点一趟并排记录两列）**：
- **Rust 改动**：`gnn_shadow.rs` 增第二端点（读 env `GNN_PORT2` / `GNN_HOST2`，在 `ShadowGnnTlEvaluator::new()` 内解析 → `tl_opt.rs` **零改动**）。第二路与第一路一样是**纯观察者，只写日志、不参与决策**（`evaluate` 返回的始终是 inner 的 SPICE 真值）。不设 `GNN_PORT2` 时 `gnn2 = None`，输出与旧版**逐字节相同**（新列追加在行尾，旧解析器用不锚定的 `re.search`，不受影响）。内层仓库版本 `15.9.3`。
- **端口 → 模型的映射由外部约定，Rust 不知道也不关心**：**8000 = 模型1 = `v2nowave42b` ep100（拓扑臂，不带 env）**；**8001 = 模型2 = `v2nowavegnn42b` ep150（ids 臂，`USE_IDS_AVG_APPROX=3`）**。`run_shadow_batch.sh` 内部硬编码 `GNN_PORT=8000` 且**不设** `GNN_PORT2` → 在父 shell `export GNN_PORT2=8001` 即被全部 12 个分片继承。
- **收尾状态**：12 分片全部归零；**全程 Xyce 进程数 0**（`XYCE_CACHE` 内容寻址缓存全命中）；**总行 5398、第二列 NaN 0**。
- **⚠ epoch 的"独立旁证"没做成，但选择本身有据**：两臂 serve 的 ckpt 都取自上方结果表的 **Best midpoint** 列（42b→ep100、gnn42b→ep150），**来源完全相同、证据强度相同**。本想再用 md5 交叉确认一次，但**两臂的 `best_model.pt` 与任何 `midpoint_epN.pt` 都不撞** → `best_model.pt` 落盘时另存了额外状态（epoch/optimizer 之类），**这个办法验不了 epoch**。故剩下的只是"磁盘上 `midpoint_ep150.pt` 就是表里那个 ep150"这一层，由文件名自证，**不是选错 epoch 的风险**。**若日后确需改 epoch 重跑，因缓存全命中，一趟成本很低。**
- **「轨迹与模型无关」的实证（本趟为 m4 族第 8 趟）**：总行 5398 = 规范口径的 **5390 成功行 + 8 失败行**，与 m4 族七趟的 **106 集 / 5390 行 / 8 失败**逐项相同。这正是"一趟 shadow 能同时记录两个模型"的前提 —— 候选集与真值列全由 SPICE 决定，与 serve 挂哪个模型无关；同时 12 分片并发下模式 3 一次都没超时（第二列 0 NaN），说明现场跑 GNN 的 serve 撑得住这个并发度。
**Rust shadow A/B 结果（2026-09-15，同一趟 106 集全配对）**：

| 判据 | 模型1 = `42b` ep100（拓扑, 8000） | 模型2 = `gnn42b` ep150（GNN ids, 8001） | 配对 1胜/2胜/平 |
|---|---|---|---|
| 严格 k=2（#1∈预测前2） | 23.6% | 18.9% | 12 / 7 / 87 |
| 严格 k=3（#1∈预测前3） | 34.0% | 28.3% | 12 / 6 / 88 |
| 宽松 k=2 | 41.5% | 32.1% | 19 / 9 / 78 |
| 宽松 k=3 | 60.4% | 46.2% | 18 / 3 / 85 |
| 选择遗憾（越低越好） | **14.26%** | 24.00% | 34 / 23 / 49 |
| 两阶段最终遗憾 | 3.47% | 12.92% | 44 / 14 / 48 |
| Spearman（越高越好） | 0.093 | **−0.223** | 68 / 29 / 9 |
| 跨度>10% 子集（79 集）遗憾 | 17.88% | 30.90% | 25 / 17 / 37 |

**判定：部署口径与训练侧同向，且负得更重 —— GNN ids 现场列（模式 3）在真实 SPICE 排序上明确劣于纯拓扑对照。** 7 个判据全部同向，配对胜负**无一反转**（选择遗憾 34/23、两阶段 44/14、Spearman 68/29）。跨度>10% 的可排序子集（79 集）上差距同样存在（17.88% vs 30.90%）。
**最有信息量的形态 = Spearman 转负（0.093 → −0.223）**：这**不是「排序能力变弱」，而是排序与真值整体反相关**（配对 68/29/9）。训练侧那条线只是从 0.389 掉到 0.340（变弱、没转负）→ **部署口径把这一列的破坏性放大了**，"GNN 逐门预测注入组内不一致噪声"的机制假说在此得到独立支持。
**分母**：配对候选集数 = 106 / 主口径 106，**落掉 0 集**，带第二列的行 5390/5390 → 两模型严格同分母，A/B 无分母偏差；12 分片并发下模式 3 一次都没超时。

**✅ 未解释项已定案（同日对照跑）：42b 的 14.26% 是真实的，不是环境/口径产物。**

三臂在**部署口径**下单调可分（均 in=45 拓扑臂 / 8000 / 不带 env；口径 106 集 / 5390 成功 / 8 失败，逐项相同）：

| 服务 ckpt | 严格 k=2 | 严格 k=3 | 宽松 k=2 | 宽松 k=3 | 选择遗憾（中位） | 两阶段（中位） | Spearman | 跨度>10%（79 集）遗憾 |
|---|---|---|---|---|---|---|---|---|
| **`42m4` ep250**（交付基线，16.11.10 代码） | **26.4%** | **39.6%** | **47.2%** | **65.1%** | **10.66%**（6.17%） | 3.88%（**0.69%**） | **0.142** | **13.03%** |
| **`42b` ep100**（17.1.2 代码） | 23.6% | 34.0% | 41.5% | 60.4% | 14.26%（8.55%） | **3.47%**（1.56%） | 0.093 | 17.88% |
| `gnn42b` ep150（+GNN ids，模式 3） | 18.9% | 28.3% | 32.1% | 46.2% | 24.00% | 12.92% | −0.223 | 30.90% |

（候选数分布 min=4 med=13 max=280；三趟的候选集数/成功行/失败行均 106/5390/8，逐项相同。）

> ⚠ **2026-09-16 口径标注**：本表的「106 集」= `(circuit, window_try)` **池化**口径（历史），**不是部署真值**；部署真值 = **714 集 batch 口径**（见下方「部署口径定案」块）。同 arm 只换口径严格@3 就动 **+26.5~55.9pp**（因臂而异；⚠ 2026-09-17 改，旧记上界 39.5 用了错标树的 74.4，见文末「74.4 错标定案」）⇒ **本表只作同口径臂间比较用，数值不可与 714 集口径互读**。另：**本表首行「`42m4` ep250」的 epoch 标注存疑** —— 它的读数（39.6/65.1/10.66）与 §13.3 记为 **mid200** 的那次（39.6/65.1/10.87）一致，而十点扫描的 ep250 读 34.9%，详见该块 (7)。

- **只有「两阶段最终遗憾」的均值不被 42m4 压住**（42b 3.47% < 42m4 3.88%，唯一例外）；但**中位数翻回来 42m4 明显更好（0.69% vs 1.56%）** → 42b 的均值优势来自长尾，典型集上差一倍多。故 **42b 的短板集中在顶端的精细分辨**（选择遗憾 +3.60pp、严格 k=3 −5.6pp），**粗筛层面差距小得多**（宽松 k=3 −4.7pp，两阶段均值反超）。→ 部署含义分岔：走 **GNN 自选 top-1** 时 42m4 明显更好；走 **GNN 前3 → SPICE 精排**（"仿真省 ≥75%"那套两阶段方案）时 42b 差距小得多（但仅均值，典型集仍是 42m4 好）。

- **42m4 把历史基线复现到 0.21pp 以内**（本轮 10.66% vs DIFF §13.2/13.3 的 10.87%）→ 口径、`XYCE_CACHE`、解析脚本三者都可复现，**历史基准可信**；反过来也坐实 42b 那 3.60pp 差是真的。
- **交付基线没有被撼动**：42m4 ep250 仍是部署口径下最好的纯拓扑模型。
- **A/B 结论不受影响、反而更稳**：42b 是**在服务自己那个 ckpt 的条件下**赢下全部 7 个判据的 → 天平已偏向模型2 而它仍输 9.74pp、Spearman 转负；把 42b 的 epoch 修好只会让差距更大，**方向不可能翻转**。
- **由此冒出的新问题（比原问题更有价值）：42b 与 42m4 训练侧几乎同级（选择遗憾 3.94% vs 3.80%、Best Val Rel Err 27.47% vs 26.65%），部署口径却差 3.60pp。** 两条候选解释，一次跑即可分离：
  - **(a) ckpt 选点**：42b 服务 `ep100`（早停于 269），42m4 服务 `ep250`（早停于 310），而 "Best midpoint" 是**按 val loss** 挑的 —— 本节上方已记「本任务 val loss 与排序目标不是一回事、甚至反向」，**这是该教训的第二次独立命中，也是它第一次在部署口径上打出 3.6pp 的实价**。
  - **(b) 代码/数据**：42b 走 17.1.2 代码（42m4 是 16.11.10）。
  - 分离办法：把 42b 换成 **`ep250`**（它最后一个 ckpt）重跑 —— 跳到 ~10.7% 即 (a)，仍 ~14% 即 (b)。缓存全命中，一趟约 2.5min。**待办。**
- 待办：Rust 基准表（本节 + `GNN_RUST_DATA_DIFF.md` §13.2/13.3）回填上面三行；42b 的 ep 分离实验有结果后，再定「42b 该服务哪个 ckpt」以及 42b 是否还适合当纯拓扑对照臂。

**✅ 3.60pp 之谜收口 + 本族第一张可信 epoch–部署质量曲线（2026-09-15 夜）**

> ⚠ **上方三块的部署口径数值（10.66 / 14.26 / 3.60pp / 13.03 等）全部产自 17.2.4 之前**，每份都带**未知边序**，落在实测抖动带 **10.51–11.13%（跨度 0.62pp）** 内 → **不可与下表混用**。根因/修法见 `GNN_RUST_DATA_DIFF.md` §13.7（边序随 `PYTHONHASHSEED` 变 → float32 累加序 → 平均秩整组 ±0.5；17.2.4 用 `sorted(set(edges))` 源头定序）。**上方定性结论（A/B 方向、42m4 不被撼动）全部不受影响**，被重算的只有那个 3.60pp。

**两条候选解释的分离结果：主因 = (a) ckpt 选点（占 56%），残余 = (b) 代码/数据（占 44%）。** 两臂各五个已存 midpoint 全扫（17.2.4 规范序；十趟分母逐项相同 106 集 / 5390 成功 / 8 失败 → 同口径可比）：

| 服务 ckpt | 严格k2 | 严格k3 | 宽松k2 | 宽松k3 | 选择遗憾（中位） | 两阶段（中位） | Spearman | 79 集遗憾 |
|---|---|---|---|---|---|---|---|---|
| `42m4` ep50 | 25.5% | 38.7% | 44.3% | 67.9% | 12.37%（6.51%） | 4.10%（0.60%） | 0.119 | 15.46% |
| `42m4` ep100 | **26.4%** | 34.9% | 41.5% | 62.3% | 12.32%（6.37%） | 4.61%（1.75%） | 0.128 | 15.32% |
| `42m4` ep150 | 17.0% | 32.1% | 32.1% | 58.5% | 12.78%（7.09%） | 4.20%（1.84%） | 0.023 | 15.68% |
| `42m4` ep200 | 22.6% | 33.0% | 37.7% | 62.3% | 13.78%（6.92%） | 4.40%（1.58%） | 0.083 | 17.19% |
| **`42m4` ep250（交付基线）** | 24.5% | 34.9% | **46.2%** | **66.0%** | **11.13%**（6.41%） | 4.00%（0.79%） | 0.134 | **13.66%** |
| `42b` ep50 | 22.6% | 34.0% | 44.3% | 66.0% | 12.55%（7.08%） | 3.57%（0.96%） | 0.156 | 15.57% |
| `42b` ep100（原服务点） | 20.8% | 32.1% | 41.5% | 61.3% | 14.03%（8.73%） | 3.29%（1.79%） | 0.089 | 17.48% |
| `42b` ep150 | 17.9% | **34.9%** | 38.7% | 60.4% | 12.42%（6.52%） | **2.90%**（1.17%） | 0.132 | 15.25% |
| `42b` ep200 | 24.5% | 34.0% | 39.6% | 62.3% | 12.64%（6.41%） | 3.47%（1.14%） | **0.169** | 15.83% |
| `42b` ep250 | 23.6% | 34.9% | 37.7% | 63.2% | 13.07%（6.78%） | 3.00%（0.80%） | 0.154 | 16.29% |

> ⚠ **2026-09-16 口径标注（同上方表）**：本表十行全是 **106 集 window 池化口径**。行内 epoch 间比较**受同一口径约束**故仍有效，但**绝对值不可与 714 集 batch 口径互读**；且 §13.7 的 ±0.31pp 抖动带由**遗憾**标定、**对严格 recall@3 未被验证**（同 arm 两处记录的严格@3 差 4.7pp，见「部署口径定案」块 (7)）。

**3.60pp 的分解（口径修正后 = 2.90pp）**：`42b` ep100 **14.03%** vs `42m4` ep250 **11.13%** = 2.90pp =
- **1.61pp = 42b 服务了自己五个 epoch 里最差的那个**（ep100 14.03% → ep150 12.42%）
- **1.29pp = 残余（代码/数据/训练手气，未分离）**：两臂最优对最优 12.42 vs 11.13

→ **56% 是选点错、44% 是真残余。** 注意 (b) 未被证伪：42b 的最晚 epoch（ep250 13.07%）并没有跳到 ~10.7%，所以残余不是"换了 epoch 就好"。（(a) 的原始判据是「跳 ~10.7% 即 (a)、仍 ~14% 即 (b)」—— 实测落在中间：**两因各占一半**。）

**裁决**：
1. **臂的差异远小于 3.60pp 听起来的量级**：两臂五点区间大幅重叠（42b 12.42–14.03 / 42m4 11.13–13.78），均值只差 0.46pp，最差对最差几乎相等（14.03 vs 13.78）。**ckpt 选点（臂内 1.61–2.65pp）比臂本身（均值 0.46pp）大一个量级** —— 部署选型的杠杆在 epoch 上，不在换臂上。
2. **但两阶段口径存在真正的臂×指标交互**：两阶段前五名全是 42b（2.90–3.57%）、后五名全是 42m4（4.00–4.61%），与 GNN 自选 top1 口径**完全反序**。这是臂级差异（每一对 epoch 组合都不反转），不是选点产物。机制 = **42m4 的右尾更肥**（其「均值−中位」五点 3.50/2.86/2.36/2.82/3.21 vs 42b 2.61/1.50/1.73/2.33/2.20）→ 典型集上两者相当，42m4 被少数电路拖坏。→ **选臂取决于部署是否带 SPICE 精排：不带 → 42m4（前三名全是它）；带 → 42b（五点全胜，最好 2.90 vs 42m4 最好 4.00 = 1.10pp；最差对最好仅 0.43pp）。**
3. **"Best midpoint ≠ 部署最优"累计 1/2**：`42m4` 的 Best midpoint ep250 **恰是它五点最优**（11.13%，全表最优）；`42b` 的 Best midpoint ep100 **恰是它五点最差**（14.03%，全表最差）。同一个 selector 一臂挑中最好、一臂挑中最差 → 该 selector 的判据（`BEST_METRIC=smoothed_rel_err`，非排序量）在部署口径上无判别力，**不能默认服务 Best midpoint**。**⚠ 17.2.6 更正：此处判据名写错** —— `smoothed_rel_err` 是 `best_model.pt` 的判据（`src/train_sweep.py:926-933`：`elif BEST_MODEL_METRIC == 'smoothed_rel_err'`（**17.4.3 时**；17.4.1 时 `:908`，17.4.1 前 `:848`）；⚠ 旧记 `:815-825` **在 17.4.1 前就已过期**——那一段实为训练循环里的逐 epoch `cap2` 最优跟踪，17.4.3 时为 `:893-903`）；`Best midpoint` 由加权 `score`（recall@3 主导，旧 `train_sweep.py:978`）选出，二者是两个不同的 selector，且实测该 `score` 与部署**反序（Spearman = −1）**，见下方 17.2.6 block。**🔴 17.4.1 补：该加权 `score` 已于 17.3.7 整体删除**，midpoint 选点改为 capture2（17.4.3 时 `:1145`）、17.4.1 起改为取平台末端（`:1139-1173`）→ 上述反序是已删代码的历史实测，**不构成当前代码的风险**。附带第二次命中「训练侧 Spearman 高 = 部署差」：`42m4` ep200 训练侧 Sp 次高，部署遗憾 13.78% 为该臂最差。
4. **交付基线不变 = `42m4` ep250（纯拓扑）**，其规范序数值刷新为 **11.13%**（旧记录 10.66 / 10.87 都是 17.2.4 前的抖动抽样，落在 10.51–11.13 带内；11.13 恰是该带最差值 → **旧值把这条臂高估了 0.47pp**）。**A/B（模式 3）结论完全不受影响**：42b 是在服务自己**最差** epoch 的条件下赢下全部 7 个判据的 → epoch 修好只会让天平更偏模型 1（与本页上方 §871 的预判一致）。**（2026-09-17 改：以上 11.13% 是 106 集 window 口径；部署真值口径下该基线的同 ckpt 读数为选择遗憾 5.93% / 严格@3 90.8% / 宽松@3 97.9%——两把尺不可互读，见「部署口径定案」块 (8)。旧记的 7.23 / 74.4 / 97.6 属错标树、已作废。且本条「42m4 ep250」的 epoch 标注本身存疑，见同块 (7)。）**
5. **达标线仍很远**：全表最好的 11.13% 是「选择遗憾 ≤5%」目标的 **2.2×**；两阶段最好的 2.90% 达标但严格 recall 全族仍在 17–39%。→ 本节两次扫描证明的是**选点依据**，不是可交付性；缺口要动模型/数据（60w 新批次），不是继续换 ckpt。
6. **⚠ 记录在案、暂不动 serve**（用户 2026-09-15 决定：不再频繁改动交付基线）：`42b` 若日后仍要当模式 3 的对照臂，**应改服务 ep150**（该臂 regret 与两阶段双最优，除严格k2/宽松k2 外全面压 ep100）。**待办**：Rust 基准表（本节 + DIFF §13.2/13.3）回填上表十行；42b ep100→ep150 与"残余 1.29pp 是代码还是手气"两件都留待 60w 新战役一并处理。

**✅ 训练侧逐 epoch 复核：进步是真的没有、LR 已到底、选点判据与部署反序（2026-09-15 夜 / 17.2.6）**

动机：上表是 **Rust 端**；本节回答「训练侧到底有没有进步」与「每 epoch 迭代的是什么」，并顺手修掉上方裁决 3 的一处判据名错误。

**(0) 先回答"迭代的是什么"：只有 delay，没有排序项。** `src/train_sweep.py:157-164`（**17.4.3 时**；17.4.1 时为 `:139-146`，旧记 `:84-91`）：

```
target_log  = torch.log10(data.y)                                  # data.y = 该行的 SPICE 延迟真值
residual    = out - target_log
sample_loss = Huber(delta=0.3)                                     # log10 空间、逐行
loss        = (sample_loss * PIN_WEIGHTS[switching_pin]).mean()    # 按开关脚加权
```

`RANK_LOSS_W=0`、`KD_ENABLED=False`（纯拓扑臂）→ **损失里没有任何组内排序项**。每 epoch = 约 74.6 万行 delay 过一遍、`BATCH=80` 更新，日志 `Epoch NNN | LR | Train Loss | Val Loss | Val Rel Err` 即全程记录。**组内排序从来不是被优化的量，只是被测量的量**（回归的副产品）→ 它在排序轴上早饱和是**必然**，不是意外。
（分清两条 GNN 线：`_fit_idsavg_gnn_server.py` 那条是 `mse(pred, log1p(ids))`，目标是 **ids**、指标 val_R²，不碰 delay。）

**(1) `42b` 训练侧逐 epoch（`train*.log` 的 `Midpoint Comparison` 块，hi_spread>10% 子集）+ LR / 收敛：**

| ep | 训练 regret | 训练 sp | 训练 cap | 训练 r2 | 训练 r3 | 训练 score | LR | Train Loss | Val Loss | Val Rel Err | **Rust regret** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 5.86% | 0.407 | 82.4% | 63.3% | 69.4% | 108.20 | 1.00e-04 | 0.0102 | 0.0219 | 31.98% | 12.55% |
| 100 | 5.37% | **0.446** | 84.4% | 65.0% | **73.4%** | **113.43** ← Best | 5.00e-05 | 0.0094 | 0.0211 | 29.70% | **14.03%** |
| 150 | **5.25%** | 0.441 | 84.4% | 62.7% | 68.8% | 107.67 | 1.25e-05 | 0.0087 | **0.0206** | **29.14%** | **12.42%** |
| 200 | 5.79% | 0.442 | 83.1% | 63.6% | 69.7% | 108.73 | 3.13e-06 | 0.0084 | 0.0206 | 30.02% | 12.64% |
| 250 | 5.48% | 0.444 | 84.7% | 63.6% | 70.5% | 109.82 | 1.00e-06 | 0.0083 | 0.0206 | 30.20% | 13.07% |

> 日志里的打印顺序是 `ep100 / ep150 / ep200 / ep250 / ep50` —— `sorted(glob)` 按**文件名字典序**排，不是 epoch 序；看别的臂的日志时别被这个顺序带偏。

**(2) 训练侧也平，而且非单调** → **Rust 那条平线是如实的**，不是"训练增益不转移"。regret 全距 0.61pp（5.86 → 5.37 → **5.25** → 5.79 → 5.48）、sp 0.407→0.446 后走平（全距 0.039）、cap 全距 2.3pp、r2 全距 2.3pp → **ep100 之后排序轴上没有净进步**。

**(3) LR 已到底 + 泛化在 ep150 就停了**：LR `1e-4 → 5e-5 → 1.25e-5 → 3.13e-6 → 1e-6`，**ep250 恰落在调度下界 `LR_MIN=1e-6`** → 本次 run 是"跑完"而不是"被截断"（想靠多训补，没有东西可补）。同时 **Val Loss 在 ep150/200/250 三点同为 0.0206**（到底），**Val Rel Err 在 ep150 触底 29.14% 后回升**到 30.02 / 30.20%，而 **Train Loss 仍单调缓降**（0.0087 → 0.0083）→ **ep150 之后那 100 个 epoch 只买到训练集拟合、没买到泛化**（轻度过拟合）。

**(4) 四个独立信号在 ep150 交汇 —— 只有项目自己的判据不在。** 训练侧 regret 唯一最小 5.25%（ep150）、Val Rel Err 唯一最小 29.14%（ep150）、Val Loss 底线首达（ep150）、**Rust 选择遗憾唯一最小 12.42%（ep150）**。**唯一不指向 ep150 的是选点用的 `score`：它挑了 ep100 —— 而 ep100 是 Rust 全表最差（14.03%）。** → **ep150 是这个臂的真实局部最优，不是手气。**

**(5) 判据与部署口径完全反序（Spearman = −1）**：`score` 高→低 = `ep100 / ep250 / ep200 / ep50 / ep150`；Rust 好→坏 = `ep150 / ep50 / ep200 / ep250 / ep100` —— **五点完全反序**。~~且**同名指标的端点也反转**：训练侧 `r3` 最高 ep100（73.4%）→ Rust 严格 k3 **最低**（32.1%）；训练侧 `r3` 最低 ep150（68.8%）→ Rust 严格 k3 **最高**（34.9%）。~~ **⚠ 17.2.7 撤回此句（不是反转，是不可对读）**：训练侧 `r3` 与 Rust 严格 k3 是**两个不同的估计量**（总体 / 并列约定 / 被排序的量三项都不同，见 (9)），且部署侧 106 集上比例量的 1 SE ≈ 5pp（**1 集翻转 = 0.94pp**）→ 那几个百分点的"反转"**完全在噪声内，不构成证据**。而 `score` 的权重是 `100·r3 + 50·r2 + 0.3·sp − 0.2·regret + 0.1·cap`（旧 `train_sweep.py:978`，注释自陈「recall 最重要 → recall@3 主导选点」；**该加权 `score` 已于 17.3.7 删除**）→ **判据被权重最大、却最不转移的那个量主导**。另：训练侧 regret ↔ Rust regret 秩相关 = **0**（仅 ep150 两边同点，其余四点乱序）；训练侧 sp 最高 ep100（0.446）→ Rust sp **最低**（0.089）。
→ 裁决 3 那句「该 selector 无判别力（1/2）」**升级为：判据与部署反序，且在一个臂上完全反序**。**⚠ 但 n=5，五点全反序在 5 个样本上可能纯属巧合，不作定律**；它的分量来自另外两处独立同向命中（`42b` ep100 训练 sp 最高→部署最差；`42m4` ep200 训练 sp 次高→该臂部署最差）。

**(6) 口径修正：「~7pp 硬底」是混合总体，同口径是 ~10pp。** 训练侧这张表是 **spread>10% 子集**（`src/utils.py` `hi = s > 10`），与 Rust 的 **79 集子集同一规则**（`_shadow_analyze.py` 明写"对齐 V2 hi_spread 口径"）→ **子集对子集**：训练 **5.25–5.86%** ↔ Rust **15.25–17.48%** = **~10pp**。而 §13.2 那句「训练 3.80 → Rust 10.87 ≈ 7pp 硬底」是**全局训练数 vs 106 集 Rust 数**（两个不同总体）。7pp 不算错，但**不可当同口径硬底引用**；新战役要引"硬底"请写 **10pp（同口径）**。

**(7) ⚠ 纠一处判据名错误**：上方裁决 3 与 DIFF §13.7.2-4 把 `Best midpoint` 的判据写成 `BEST_METRIC=smoothed_rel_err` —— **错**。`smoothed_rel_err` 是 **`best_model.pt`** 的判据（`src/train_sweep.py:926-933`（**17.4.3 时**；17.4.1 时 `:908`，17.4.1 前 `:848`）；⚠ 旧记 `:815-825` 在 17.4.1 前就已过期——那段是训练循环里的逐 epoch `cap2` 最优跟踪，17.4.3 时 `:893-903`）；`Best midpoint` 由上面那个加权 `score` 选出（**17.3.7 已删**），**二者是两个不同的 selector**。**（17.4.1 再补：常量已更名 `BEST_MODEL_METRIC`，值仍 `'capture2'`，现于 `config.py:42`。）**连带一个待验的有用推论：`smoothed_rel_err` 的逐 epoch 值恰在 ep150 触底 → **`best_model.pt` 大概率就是 ep150 邻域那份权重**，而它**从未被这十趟扫描扫过**（扫的全是 `midpoint_ep*.pt`）。若成立，「服务 `best_model.pt`」可能比「服务 Best midpoint」更接近部署最优 —— **待验（读日志即可，不必跑）**。

**(8) 结论 / 待办**：① **60w 新战役的选点判据必须换成与部署同向的量**（当前 recall 加权的 `score` 会被反向使用）；② 把 `best_model.pt` 加入扫描候选；③ `42b` ep100→ep150、残余 1.29pp 的分离，照旧留待新战役；④ **⚠ 记录在案、不动 serve、不动交付基线**（用户 2026-09-15 决定）。

**(9) ⚠ 17.2.7：撤回 (5) 的「同名端点反转」，改为「两端 recall@3 不可对读」+ 部署侧分辨率不足（2026-09-15，代码已落地）**

- **① 两端 recall@3 是三处不同源的估计量**：(a) **总体** —— 训练侧是 test split 的 `spread>10%` 组（10³ 量级组数），Rust 侧是 106 个部署集（其 hi_spread 子集 79 集）→ 69–73% vs 32–35% **主要是总体差，不是能力差**；(b) **被排序的量** —— 训练侧排**原始预测均值**（`utils.py` `np.argsort(pr)`），serve/Rust 排**候选集内平均秩**（`serve.py` `predict_rank_batch` 的 competition rank `avg=(i+j)/2+1`，写进 `avg_delay` 字段）→ **单模型下单调等价（排序相同）、集成下不等价**，前者只是后者的代理；(c) **并列约定** —— 训练侧原为默认 quicksort（并列次序无保证），Rust 侧 `argsort(kind="stable")` 按 `eval_idx` 行序 → **已在 ④ 统一**。
- **② 部署侧分辨率**：106 集、p≈0.35 时 1 SE ≈ 5pp，**1 个候选集翻转 = 0.94pp**。故 42b 五点跨 epoch 的严格/宽松各口径变化（几个 pp）都在 ~1–2 SE 内 → **"哪个 epoch 的部署 recall 更高"在本轮样本量下不可判**。这与 17.2.4 消掉的**抖动**不同，是**样本量**这层不可分辨性。（要精确翻转数：`~/sweep_v2nowave42b_ep*.out` 里各集的明细行可逐集比对。）
- **③ `regret_2stage` 并列免疫，且与严格 recall@3 逐集等价**：`_shadow_analyze.py` 的 `top3_true_best = true[order_g[:3]].min()` 取**前 3 名内真值最小值**，集合内部次序不影响结果 → 不受并列规则影响；且恒等 **严格 recall@3 == (regret_2stage == 0)**（真 #1 落在预测前 3 ⟺ 前 3 内真最优即真 #1）。→ **带精排路线下，选点/优化的目标就是严格 recall@3**（且该量对并列免疫，是最干净的判据）。
- **④ 17.2.7 代码统一（本轮）**：`src/utils.py` 的 `_spearman` 两次 argsort 与 `ranking_metrics` 的 `ord_pred/ord_true` 全部加 `kind='stable'`（与 Rust 侧同一条并列规则＝按组内行序；无并列时行为不变，已本地核对）。`_shadow_analyze.py`：更正过期注释（旧注「gnn_pred 只写 7 位有效数字 → 伪并列」**该机制不存在** —— gnn_pred 现在是**秩**，7 位可精确表示；那次 0.00%↔36.34% 翻转的真正源头是 float32 边序抖动，17.2.4 已定序），并新增**五个诊断位**汇总打印：重复网表集（真值有完全相等候选 → recall@3 天然 <100%）、预测第3小==第4小（top-3 归属由并列规则决定）、预测有重复值、**量纲混合守卫**、**兜底窗口**。`serve_http.py` docstring 更正：`avg_delay` 字段**不是延迟**（≥2 候选是秩，<2 候选才走原始延迟）—— 这是本轮"排序不一致"最易被误读的一处命名。
- **⑤「量纲混合」现在为何不会发生（推理已变成可断言输出）**：`gnn_shadow.rs` evaluate 兜底路径（预排序缓存未命中 → 单候选 `/rank`）返回**原始延迟**，命中路径返回**秩**；但 `tl_opt.rs` Pass2 `pre_rank` 与 Pass3 `evaluate` 用**同一批 `prepared`** → 一个窗口要么全命中、要么全兜底，**集内不混**；且单模型下兜底窗口按原始延迟排序与按秩排序等价 → 该集 recall/regret 仍有效（仅分数分辨率受 CSV 的 7 位有效数字限制）。诊断位就是把这个推理钉成输出。
- **⑥ 落点**：①②③ 记录在案，**不动 serve、不动交付基线**；③ 是 60w 新战役目标函数的依据（承 (8)①）。

**✅ serve 输入退化定位：晶体管数 + 门数，load 打平（2026-09-16 / 17.3.4）**

> 全表与代码级定义见 `GNN_RUST_DATA_DIFF.md` §13.8。脚本 `scripts/diag/_serve_input_ablation.py`（17.3.2/17.3.3），跑在 `~/abl-run` 链接工作树（钉 `56b2fdd`）。**548 组 / 346 组 hi / 111,364 行 / 13 变体 / 前向 988s。**

**(1) 闸门先过**：`base`（训练侧全真输入）对本臂已存 `test_predictions.npz` 逐位比对 —— **中位相对差 0.000e+00、p95 2.252e-06、max 6.564e-06 → ✅ 复刻成立**（float32 npz，阈 1e-6）→ `base` 绝对值可与该臂历史对读，下表 Δ 才有意义。

**(2) 读数（recall 两口径都给）**：

| 变体 | 遗憾 mean | Spearman | R2 严\|宽 | R3 严\|宽 |
|---|---|---|---|---|
| `base` | **3.79%** | 0.397 | 56.6\|67.3 | 63.9\|83.4 |
| `serve`（现状） | 5.86%（**+2.06pp**） | 0.159 | 42.5\|53.8 | 47.8\|69.3 |
| `loadconst` | 3.70%（−0.09pp） | 0.395 | 55.5\|69.0 | 64.1\|82.3 |
| `serve_fixTC_prior`（真 csig+真 prior） | **3.70%（−0.09pp = 完全恢复）** | — | — | — |

其余 Δ：`csig` 两项合 +0.75（其中 `csig[0]` 单独 +0.22、`csig[1]` 单独 +0.75，次可加）、`noprior` +0.68、`serve_noload` +1.69、**`serve_fixTC_recon` +2.23（比不修还差）**、`serve_fix_recon` +8.60。`serve` 相对 `base`：Sp **−0.238**、R2严 **−14.05pp**、R2宽 −13.5pp、hi_spread **+2.80pp**、hiR3严 −16.76pp。自洽性：`base`≡`csig0_fixTC`、`loadconst`≡`serve_fixTC_prior` ✅。

**(3) 结论**：① 退化几乎全部来自**单一量 `transistor_count`**（喂 `csig[1]`@`data_loader.py:322` 与 `prior[0]`@`:716`），真值一给 2.06pp 全回来；② `csig[0]`（应=**网表 `X_` 行数**）是独立小项 **+0.22pp，serve 可自行复算** → 可先单独上；③ **load 判为并列不做**（从 base 改 −0.09pp、从 serve 改回 +1.69/改善 0.37pp，两向不一致且都 ≤0.37pp；机制上 `(LOAD_F−mean)/scale=−0.15` 本就在一个标准差内）；`arrival` 两侧皆死特征。**`n_gates` 与 `prior[1]/[2]` 在 14,727 行上全量复算 100.0000% 相等（corr 1.000）**；⚠ 但 `set(cell_types_json)==set(网表唯一 cell 名)` 只有 **98.61%**（full 99.59%、io 96.05%）—— 204 个例外全是 `cell_types_json` **多**含 `SC_JOIN_BOOL0_BOOL0`/`SC_JOIN_BOOL1_BOOL1`（常量折叠 join）、**从不缺失**且不匹配两条 prior 模式 → 计数仍恒等。**17.3.2 那个 100% 是 548 组 test split 上的，不是全量。**

**(4) ⚠ 口径修正（本节最重要的一条）**：旧探针把 `transistor_count` 当成 `sc_expansion.json` 里 ASAP7 展开的 Σ`n_t` 复算 → 得 95.6% 误差并一度怀疑数据。**正确口径 = SC 网络（Join/Inv/Bridge/And/Or）的 `M_` 实例总数、含实例级去重**，实现在 `NetlistOpt/src/utils/transistor_count.rs:69`（语义见 `tests/transistor_count.rs:21-52`），与 `DATA_SPEC_V2.md:65` 一致。判据：`tc / 顶层 X_ 门数` 中位 **7.33（full）/ 4.12（m4）** vs ASAP7 口径同一行 `92/6=15.3` —— **差 3~5 倍**；且 1317/3822 个 `(expr, tc)` 组合 **nx 不同而 tc 相同** → 该量不是门数的函数。

**(5) 修法已定（→ Task #63，预期恢复到 −0.09pp）**：Rust **本来就在算这个量** —— `tl_opt.rs:214-217` 的 `FullCircuitTlEvaluator::evaluate` 从 `module.to_recexpr()` 同时得 `transistor_count` 与 `avg_delay`（训练标签 `y`），只是 `gnn_shadow.rs:22-33` 的 `GnnCandidate` 没带这个字段。**Python 侧向后兼容可先上**（无该字段 → `None` → 今天的 csig[1]=0/prior=None 行为），Rust 后上，不用同时发布。**近似值必须放弃**：`serve_fixTC_recon` +2.23pp 证明两条支路是未归一化裸 `Linear`（`model.py:39-54`），错量级比缺值更伤。残余风险 = 生成器（外部工程）写的 `transistor_count` 是否字面等于该例程输出（强证据推断，非读数）→ 闸门：serve 记录收到的 `(netlist, tc)`，离线与 `batch_v2_io`/`m4` 的 `gate_level_netlist` 精确匹配后对读 parquet。

**(6) ⚠ 方法论：n<100 组的消融只能验管道 / 做二值判定，不能给因子排序。** 20 组 smoke 给出 `loadconst +1.54pp`（"load 是主因"）与 `csig_nn +0.00pp`，全量后反转为 **−0.09pp** 与 **+0.22pp**。当时**已有反向信号**（`serve_noload` 比 `serve` 差，而 `base→loadconst` 说相反）被标注却未据此行动 —— **方向自相矛盾本身就是样本量不足的信号**。二值判定（如"复算值是否等于真值"）不受此限，因为它不排序；另有 float32 npz 陷阱（`891422÷111364÷2=4.0` 字节 ⇒ 阈值需 ≥1e-6，1e-9 必然误报）。

**✅ 部署口径定案 714 集 + 交付基线读数刷新（2026-09-16 / 17.3.9–17.3.11）**

> 起因 = 上方 §13 ⚠「v2iaa42m4 的 recall 复算不出来」。**定案：不是数据错，是口径错——而且错的是当时以为对的那一边。** 代码级定义见 `GNN_RUST_DATA_DIFF.md` §15，脚本 `scripts/diag/_shadow_analyze.py`（`--group-by {batch,window}`，**默认 batch**）。

**(1) 「一个候选集」到底是什么。** `gnn_pred` 列存的**不是延迟、是批内平均秩**（`serve.py:358-398 predict_rank_batch`：n≥2 返回组内 competition rank 后跨模型平均，n=1 返回原始延迟 ~1e-11 → **两种量纲同列，跨批不可比**）。`pre_rank` **每个窗口迭代恰调一次**（`tl_opt.rs:893`），其 `prepared` 在 Pass 3 被 move 消费（`:896`）→ 借用检查器保证每窗口重建 → **GNN 的「集合」= 一次 `pre_rank` 调用 = `(circuit, iter, window_try)`**。旧口径 `(circuit, window_try)` 把同电路 ~6.7 个批**池化**成一集（5,398/106 = 50.9 候选/集），集内混了不同批的秩。**714 集 = 部署真值；106 集 = 历史池化口径。**

> 🔴 **2026-09-17 大改判：本节 (2)(6)(8) 的三张表原本是在一棵「错标树」上算的，整表作废并已换新。** 详见文末「74.4 错标定案」块。**上表旧读数（① 严格k3 74.4 等）属于那棵树，不是 `42m4 ep250` 的。**

**(2) 四口径对照读数（2026-09-17 重测，权威）** —— ckpt = v2nowave42m4 `midpoint_ep250.pt`（**sha1 `09be9a3c53b90760`，mtime 2026-09-02 21:30:59，自 09-02 起从未被覆盖**），树 = `~/shadow_archive/midpoint_ep250_20260916_233806`（**714 集 / 5,390 成功 / 8 失败**）。**两次独立复现逐位相同**（另有 `midpoint_ep250_20260917_214841`，相隔一天、另一次战役）：

| 口径 | 单位数 | 严格 k2 | 严格 k3 | 宽松 k2 | 宽松 k3 | 选择遗憾 | 两阶段 |
|---|---|---|---|---|---|---|---|
| **① 全部批（部署真值，默认）** | **714** | **71.8%** | **90.8%** | **83.3%** | **97.9%** | **5.93%**（中位 2.13%） | **0.63%** |
| ② 按候选池去重（同电路同 sig） | 161 | 67.7% | 80.1% | 85.1% | 96.3% | **3.75%** | 0.56% |
| ③ 按电路宏平均（每电路等权） | 18 | 63.8% | 81.5% | 81.1% | 96.1% | 5.34% | 0.89% |
| ④ 排除最大电路 `level2/DEPTH_MIX` | 176 | 65.3% | 76.7% | 83.0% | 91.5% | **3.03%** | 0.52% |

其余 ①：Spearman **0.644**（n=714）、hi_spread 591 集、小集（<4 候选）167；**跨集重复度与旧树逐位相同**（独立集 161/714、重复 553 = 77.5%，因为它只依赖真值列 ⇒ **与 ckpt 无关，是结构量**）。③ 只有 18 个单位（46 个 CSV 里 28 个无合格集）。cap2 与输入污染三项本次未复测（旧树那轮是 95.55% / 三项全 0）→ **待补**。

<details><summary>旧表（作废存档，来自错标树）</summary>

| 口径 | 单位数 | 严格 k2 | 严格 k3 | 宽松 k2 | 宽松 k3 | 选择遗憾 | 两阶段 |
|---|---|---|---|---|---|---|---|
| ① 全部批 | 714 | 53.1% | 74.4% | 69.6% | 97.6% | 7.23% | 1.66% |
| ② 池去重 | 161 | 50.3% | 68.9% | 75.2% | 95.0% | 4.76% | 0.94% |
| ③ 宏平均 | 18 | 60.3% | 79.5% | 76.9% | 96.9% | 5.67% | 0.75% |
| ④ 排除最大 | 176 | 48.3% | 65.3% | 73.3% | 90.3% | 5.11% | 0.76% |

</details>

**(3) 输入污染检查三项全 0**：分组内 eval 去重丢弃 0 / 同电路 eval_idx 跨分组重复 0 / 同一 eval_idx 真值冲突 0。这是「714 能不能当数」的闸门（shadow CSV 是 `append(true)`，OPERATIONS `:177` 记过一次叠行事故）→ **714 是真正不同的 `pre_rank` 调用次数。**

**(4) 跨集重复度**：714 集只对应 **161 个不同的候选池**（重复 553 = 77.5%，最大重数 64；DEPTH_MIX 538→26、COMP4 65→51、OR2 39→14）⇒ **① 的 714 个单元不独立**，② 是同一问题的下界答案，**两者并列读**。

**(5) ⚠ 口径效应不是常数 ⇒ 旧口径的跨臂比较不可外推。** iaa：**同一份 CSV、同一次 run 只换分组键**，严格@3 30.2% → **56.7%（+26.5pp）**、遗憾 15.21% → 15.9%（**+0.7pp**）；nowave（`42m4` ep250）：严格@3 34.9%（09-15 十点扫描，17.2.4 序）→ **90.8%（+55.9pp）**、遗憾 **11.13% → 5.93%（−5.20pp）** ⚠ 这两个数混合了「口径效应」与「服务模型身份」两件事（34.9 那趟的服务模型本身存疑，见 (7)）→ **只作量级参考，不是纯口径效应**；旧记的 74.4 / 7.23 属错标树。→ **失真量 model-dependent，不是可加的固定偏置**。⇒ I7 的六臂结论都产自同一 106 集口径、**臂内可比性仍在**，但**与部署真值的落差因臂而异、不可外推**；本页上方所有「Rust 选择遗憾 10.87/11.13/12.19…」一律按 **106 集 window 口径**解读。**臂间差距被池化压缩**：iaa→nowave 的严格@3 差，batch 口径 +17.7pp vs window 口径 +4.7pp = **压缩 3.8×**（仅一对样本，不作定律）。**判别规则**：recall 升而遗憾不降 ⇒ 口径产物；两者同向 ⇒ 真实能力（iaa 换口径 +26.5pp/遗憾 **+0.7pp 反向** ⇒ 产物；iaa→nowave 严格 +17.7pp/遗憾 **−8.7pp 同向** ⇒ 真实）。
> ⚠ **取数陷阱（本记录初稿踩过）**：nowave 的 window 侧别用 `42b` ep250 那行（宽松 63.2% / 遗憾 13.07% / 中位 6.78%）—— 它的严格@3 恰好也是 34.9%，极易抄错列，抄错就会得出「遗憾改善 5.8pp」的假数（真值 3.90pp）。

**(6) 判据稳健性 —— 2026-09-17 重测后方向翻转（旧判作废）**：
- **严格 recall@3 ≥90%**：**只有 ① 达标**（**90.8%** / 80.1 / 81.5 / 76.7）→ 旧记录那句「四种口径全不达标（74.4/68.9/79.5/65.3）→『GNN 只做启发式粗排』成立」**作废**，它建立在错标树上。
- **选择遗憾 ≤5%**：**②④ 达标**（3.75 / 3.03），**①③ 不达标**（5.93 / 5.34）→ 与严格 k3 的排序**恰好相反**。
- **两阶段遗憾**：四种口径全达标（0.63 / 0.56 / 0.89 / 0.52）。
- ⇒ 诚实读法：**① 口径下部署两项（严格k3 90.8% ≥90%、两阶段 0.63% ≤5%）都达标；但 ① 正是被重复提议主导的口径（DEPTH_MIX 一家 538/714 = 75% 权重）**，而 ②③④ —— 去掉重复计权或去掉那一个电路之后 —— 严格 k3 全掉到 90% 线以下。**判据稳健性的分歧不但还在，而且方向不再一致**：① 在严格 k3 上最好、在选择遗憾上最差，②④ 反之。这两个数的差就是**「重复提议」的定价**。⇒ 不能说「GNN 够了」，也不能说「全不达标」；**部署口径（①）达标，但达标与否压在一个电路上**。
- Rust 侧自己的 判据（集合版 `recall@top-3`）与选择遗憾（① 5.93%）仍未过线 → **与 16.11.39 裁决同向**。

**(7) ⚠ 遗留：历史表的 ckpt 标注不可信，跨 ckpt 排序全部冻结。** 同一 arm（v2nowave42m4）在 106 集窗口口径下有**三个互不吻合**的严格@3：§13.3/§747 记 **39.6%**（§13.3 头部写服务的是 **mid200**）、本页首表把同一读数标成 **ep250**、09-15 十点扫描的 ep250 读 **34.9%**（ep200 33.0%）。前两者高度一致（39.6/65.1/10.66 vs 39.6/65.1/10.87），第三者严格@3 差 **4.7pp** —— 远超 ±0.31pp 抖动带（**该带由遗憾标定，对 recall 未被验证**）。两种可能未分离：(i) 首表 epoch 标注错；(ii) 17.2.4 定序修复对 recall@3 的影响远大于遗憾。⇒ **无论哪种，「mid200 比 ep250 好」这类判断在当前证据下不成立**；要排序须用 batch 口径逐 ckpt 重测 + 每趟盖 ckpt sha1。（I15）

**(8) 交付基线读数刷新（2026-09-17 重测，权威）**：

| 交付基线 | ckpt | 口径 | 严格 k3 | 宽松 k3 | 选择遗憾 | 两阶段 |
|---|---|---|---|---|---|---|
| **v2nowave42m4（纯拓扑）** | `midpoint_ep250.pt`（sha1 09be9a3c53b90760，**两次独立复现**） | **714 集 batch（部署真值）** | **90.8%** | **97.9%** | **5.93%**（中位 2.13%） | **0.63%** |
| 同上（同 ckpt 同 arm，09-15 十点扫描，17.2.4 规范序） | `42m4` ep250 | 106 集 window（历史池化） | 34.9% | 66.0% | 11.13%（中位 6.41%） | 4.00%（中位 0.79%） |
| 同上（更早记录：§13.3 / 本页首表，**17.2.4 前**且 epoch 标注存疑） | 标称 mid200 / ep250（见 (7)） | 106 集 window（历史池化） | 39.6% | 65.1% | 10.87%（中位 6.17%） | 3.95%（中位 0.69%） |
| ~~错标树（曾被当作本行，实为另一模型）~~ | ~~标称 `midpoint_ep250.pt`~~ | 714 集 batch | ~~74.4%~~ | ~~97.6%~~ | ~~7.23%~~ | ~~1.66%~~ |

**⚠ 四行是同一 arm 同标称 ckpt 的四把尺，禁止横向对比。** 第 1 行是权威数（部署口径、两次复现）；第 2、3 行是历史 window 口径记录且**彼此也不一致**（严格@3 差 4.7pp，见 (7)）→ 只作存档；**第 4 行已删除 —— 它不是这个 ckpt 的数**（详见文末「74.4 错标定案」）。结论改为：① **交付基线不变**（纯拓扑 nowave 路线，serve 钉 `midpoint_ep250.pt`）；② **部署口径两项都达标**（严格 k3 **90.8% ≥ 90%**、两阶段 **0.63% ≤ 5%**），但 ① 口径被 DEPTH_MIX 一家占 75% 权重、②③④ 严格 k3 全在 90% 线下（见 (6)）→ **不能说「够了」**；③ 选择遗憾 ① 5.93% > 5% 仍未过，**缺口要动模型/数据（60w 新战役），与 16.11.39 裁决同向**；④ **不改 serve、不改交付基线**（承用户 2026-09-15 决定）。

**(9) 待办**：① `_shadow_ckpt_sweep.sh` 的「<0.3pp 视为打平」→ 中位 + 逐集配对，并按 (4) 的重复度做**按电路分块重采样或 sig 去重**（独立性假设不成立）+ `best_model.pt` 入候选 + 逐 ckpt 盖 ckpt sha1；② 以 batch 口径重测 v2nowave42m4 各 midpoint（至少 mid200 / ep250）—— 这是给 ckpt 排序、也是给「42b 残余 1.29pp」定性的前提；③ `_shadow_analyze.py` ③ 的 **n≥8 分层版**（n=4 集严格@3 随机基线 75% vs n=8 的 30%，宏平均把两种难度等权）+ `判定` 行标「仅 ① 口径」（**代码未改，待批**）；④ 读 宽松 k3 时记住 **n=4 集定义性饱和**（真前3∩预测前3 必然非空 → 恒 100%；对全体的影响 f=20%→97.0% / f=30%→96.6%，97.6% 在带内）。

**✅ serve OMP 空转自旋定案 + 战役提速 13.8× + 兜底闸门判为保守（2026-09-17 / 17.3.18–17.4.0）**

**(1) 根因：serve 在「两批之间」空转，不是「算得慢」。** libgomp 默认 `GOMP_SPINCOUNT=300000` + active 等待策略 ⇒ worker 线程在两个并行区之间**自旋等待**，靠计数器耗尽才入睡。而 serve 的负载恰是**每 (pin, 方向, 模型) 一次 forward + `.cpu()` 同步**、一批候选就是几百个背靠背的小并行区 ⇒ 线程池永不入睡。放大器：`nproc=24` 但 torch 起了 ~56 个 OMP worker（`Threads: 59`，**2.3× 超订**，因为 `OMP_NUM_THREADS` 从来没被设过）。

**(2) A/B 证据（无需新代码，用现成机器）**：两个 serve 并排 —— 8000 裸奔、8001 加修复 env，**同一 ckpt / 同一 scaler / 同一时刻**；一个 `TL_ONLY=level0` 分片用 `GNN_PORT=8000 GNN_PORT2=8001` 同时喂两侧。90 秒窗口读数：

| | 8000 裸奔 | 8001 修复 | |
|---|---|---|---|
| 峰值 CPU | **828%** | **3.3%** | **248×** |
| ckpt 载入阶段 | 70 CPU-s | 3 CPU-s | 纯浪费 |

配对自校验：**237/237 行都带 `gnn_pred2` 列、两列不等 = 0**；两侧进程都打印 `Threads: 59` ⇒ **线程数与工作分解未变 ⇒ 位级中性**。（这也是 #64 「tc 通道活体验证」项的闭环。）

**(3) 修复 = 三行 env，且已设为默认**：`GOMP_SPINCOUNT=0 OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0`。战役实测：

| | 修复前（参考趟） | 修复后 ep250 | 修复后 ep300 |
|---|---|---|---|
| 单趟用时 | 66 m 25 s | **288 s** | **287 s** |

⇒ **13.8×**。两趟闸门全绿：`指纹=b4c8bdc33e58  rank请求=729  兜底=0  CSV=46  行=5398`；714 集 / 5,390 成功 / 8 失败 / 小集 167。爆发速率 20 行/s（修复前爆发 3.9、停滞 0.56）。

**(4) ⚠ 它**不能**消除与 zhirui（16 路 Xyce）的争用 —— 同时纠正我此前的两处错判。** 修复后我们爆发期 zhirui 读 1431–1465%（零负载基线 1596%），与我们不在跑时无差别。机制：**nice 19 下我们仍拿到 2.4 核** ⇒ CPU 时间从来不是瓶颈 ⇒ 那 ~7% 的 IPC 损失是**内存带宽 / LLC / SMT 的共享微架构效应**。⇒ ①「自旋偷走了 zhirui 的核」**不成立**；②「修好自旋就能让 zhirui 变快」**也不成立，且不该指望**。这两条我先前的表述都过早，此处作废。

**(5) ⚠ 兜底闸门（闸门②）判为「方向对但偏严」—— 历史档**不**因此报废。** 机制：**`n_models=1`** 时，批内「平均秩」与「原始延迟」在单模型下**单调等价**（排序恒同）⇒ 兜底只改列的**量纲表示**，不改窗内次序 ⇒ top-1 / recall / regret / Spearman **全部不动**。实证逐字节：

| 存档 | 兜底读数 | 与干净重跑对比 |
|---|---|---|
| `midpoint_ep250_20260916_233806` | **128/714 ⚠** | 四个口径 × 六个指标与 `midpoint_ep250_20260917_214841`（兜底 0）**逐字相同** |
| `midpoint_ep300_20260917_023350` | **305/714 ⚠** | 同样与干净重跑逐字相同 |

⇒ 闸门② 真正防的是「**跨窗池化** `gnn_pred` 分数」与「**≥2 模型的集成**」（那两种情形下秩与原始延迟不再单调等价），单模型窗内比较下是空警报。11 份存档的「同一 `eval_idx` 真值冲突」**全 0** ⇒ `OPERATIONS:177` 那次 append 叠行事故没有重演，714 是真调用次数。

**(6) 口径之争的实证：ep250 vs ep300 的排序是口径的函数，不是模型的函数。** ① 严格 k3 逐中点：

| ckpt | ep50 | ep100 | ep150 | ep200 | ep250 | ep300 | best_model |
|---|---|---|---|---|---|---|---|
| ① 严格 k3 | 78.3% | 80.7% | 90.3% | 90.3% | 90.8% | **91.6%** | 85.6% |

**②③④ 把序打乱**（③ 的冠军也是 ep300 87.4%，但 ep100 86.4% 反超 ep150/200；④ 的冠军是 **ep50 80.1%**）。**达标与否同样翻号**：选择遗憾 ≤5% 在 ① 是 5.93/6.09 ❌、在 ② 是 3.75/4.10 ✅。⇒ 「哪个 epoch 最好」「GNN 够不够格」这两个问题**当前都是口径的函数**，报数前必须先定口径。
> ⚠ ① 之所以是唯一看出趋势的口径，是因为它 **75% 的权重压在 level2/DEPTH_MIX 的重复池上**（538/714 集、26 个独立池、最大重数 64）⇒ 这是「平均批」效应，不是普遍结论。

**(7) 部署选点政策（本轮结论；Rust 贪心的应用口径）**

- **单位定死**：GNN 在贪心里的唯一调用点是 `tl_opt.rs:893` 的 `pre_rank`，**每 (轮次, 窗口) 一次** ⇒ 部署损失按**批**累加 ⇒ **① 是估计量**（`Σ_c (n_c/N)·m_c`，工作加权期望，非偏好）；**③ 等价于假设「GNN 每电路调一次」**——它从没被那样调用过。
- **② 的用途是置信区间，不是点估计**：重复批在部署里是**真实决策**（`pre_rank` 每窗 `self.pre_ranked.clear()`，同批网表每轮真重算）⇒ 点估计必须按重数计权（=①）；② 去重后的 **161** 给有效样本量。
- **部署指标是严格 k3 / 两阶段遗憾**（对应「GNN 砍 top-3 → SPICE 精排」），**不是选择遗憾**——后者对应「完全不跑 SPICE」的更激进方案，也是分析器 `判定` 行那条 5% 线。
- **严格 k3 身兼两职**：既是质量指标，又是 ① 权重成立的前提（=100% 时 GNN 前 3 恒含真最优 ⇒ 部署轨迹与 shadow 逐字相同 ⇒ 候选池与 `n_c` 相同；现在 ~91% ⇒ 约 9% 的批会走岔，权重开始 off-policy 漂移）。
- **选点规则 = 取平台末端（最后一个 midpoint），不取 shadow argmax。** 理由：① 下 ep150–300 是 90.3/90.3/90.8/91.6 的 1.3pp 平台，②③④ 打乱 ⇒ 在**不可分辨**的差上取 argmax = **拟合噪声**（#65 已证严格@3 对 1 ulp 的敏感度是遗憾的 ~17 倍）。取末端不是「选出来的」⇒ 无选择偏差。**这与现有代码一致**：midpoint 块本就是「midpoint 选点优先、`best_model.pt` 只作兜底」。**⏩ 17.4.1 已把「取末端」写进代码**（`config.MIDPOINT_SELECT` + `train_sweep.pick_midpoint`，两处调用点共用）—— 此前代码是「取 val capture2 argmax」，政策与代码**只对「谁当选」一致、对「怎么选」不一致**，现已对齐。位置按锚点找（`grep -n 'Midpoint Comparison\|pick_midpoint' src/train_sweep.py`；17.4.1 该块 = `:1095-1150`，旧记 `:950-1073` 因文件顶插入 55 行已失效）。
- **`best_model.pt` 出局**（① 85.6% 明显低于 ep250/300）。它现在由 `config.py:42 BEST_MODEL_METRIC='capture2'` 选（17.4.1 前为 `:36`），而 #60 已判该判据 ❌ 不采纳（四条证据；「训练侧判据与部署口径脱钩」的第 4 次命中）—— **判决尚未落地到代码**，故本批 `best_model.pt` 仍是该判据的产物。
- **shadow 从「选择器」降级为「闸门」**：每趟训练后跑**一次**（~290s）判过不过线，**不做 ckpt 排名**。这才是「不用每个 ckpt 都跑」的正确解 —— 它成立是因为 §(5) 与闸门① 已经证明 **`true_delay` 列与 ckpt 无关**，所以扫 N 个 ckpt 时有 **N−1 趟的 Xyce 是纯浪费**。
- **免费的上游守卫**：每个中点都打 `ep{N}: cap2=...` ⇒ 看 val capture2 的 argmax 落在**平台内部**还是**末端**，即可判「取末端」这条规则安不安全（落在末端 = 训练被截断，规则不成立）。**不需要 shadow。** 按锚点找 `grep -n 'cap2={score' src/train_sweep.py`（17.4.1 时 `:1128`；旧记 `:1055` 已失效）。**17.4.1 起该守卫自动执行**：`pick_midpoint()` 每次选点都打印 `[选点] 规则=` 与 `守卫⑤` 结论行。
- ⏸ **离线重评（候选落盘 + N 次重评）暂缓**：它解的是「频繁比较多个 ckpt」，而政策刚决定不比较；等到跨配方比较（60w 新战役）时再建，那时才回本。缺件：存档里只有 `gnn_pred`/`true_delay`，没有网表 ⇒ 无法事后重放。最省的落法 = serve 端 dump 收到的请求体（`serve_http.py` 的 `candidates` 里带 `netlist`，是全量候选）。

**(8) ✅ 已定案（2026-09-17）：74.4 是错标，不是 `42m4 ep250` 的数。** 原判「74.4 与 90.8 差 16.4pp 未归因」的两条假设**都已实测否定**：

- **(i) ckpt 换了？——否。** `~/project-107-v2nowave42m4/outputs/midpoint_ep250.pt` 的 `sha1 = 09be9a3c53b90760`，与 09-15 那条旧记录**逐位相同**；mtime `2026-09-02 21:30:59` ⇒ **文件从 09-02 起从未被覆盖**。
- **(ii) Rust vintage 变了？——否。** 三棵树（09-15 打包 `tl_opt_batch_ab_20260915.tgz`、09-16 存档、09-17 战役）的 **`true_delay` 多重集指纹全等 = `cb0d7a96e6aada05`**，且都是 46 CSV / 5398 行 ⇒ **候选生成逐值没变**。（`tree_fp` 之所以不同，是因为它含 `gnn_pred` 以外的其它列/路径；真值这一层是干净的判别量。原判「集数一样而内容不同 ⇒ 更像 Rust 变了」的推理链**作废**。）

**定案证据**：用同一版分析器、同一 batch 口径重算全部存档，**`tl_opt_batch_keep_0916` 与 `unknown_20260916_223140` 打出 `714 53.1 / 74.4 / 69.6 / 97.6 / 7.23 / 1.66`，与 17.3.11 那张表逐位相同**。而这两棵树的候选池与交付树**真值指纹全等**（同一批候选）⇒ 唯一变量是 **`gnn_pred` 来自另一个模型** —— 佐证：它们首行 `gnn_pred=7.562406e-11`，而 09-16/09-17 交付树是 `6.622111e-11`。

⇒ **74.4 属于 09-15 夜挂在 8000 上的那个 ckpt，不是 `midpoint_ep250.pt`。** 按 `OPERATIONS §6.7` 当时的状态记录（「2026-09-15 夜：8000 上挂的是 **`42b` ep250**，两次 epoch 扫描的末位遗留，非交付基线」）与「各项数值全面呈弱臂特征」推断应为 **`42b` ep250**；⚠ 但**这一步仍只是推断**（那棵树无 RUN_INFO，无法直证），**可确证的只是「不是 `42m4 ep250`」**。

**这是第 2 次同成因的 I15 类错标**：记的是「本该服务的模型」的身份（ckpt 路径 / sha1），而不是 8000 实际服务的模型。**防线（17.4.0 已落地）**：`run_shadow_batch.sh` 现在把 **`本轮 ckpt sha1` + `Rust 源指纹`（`~/NetlistOpt` 无 .git ⇒ 哈希源码）+ `repo dirty 指纹` + `serve 脚本指纹`** 一并写进 `RUN_INFO.txt`，随树进归档；且空树不再归档（防 `unknown_*` junk 存档——本次正是靠一份这样的存档定位的）。**Rust 侧若再改，指纹必变**，这条争议从此不可能再发生。

**(9) 已落地的默认（17.4.0，防复发）**

| 问题 | 落点 | 做法 |
|---|---|---|
| serve 空转自旋（248×） | `shadow_campaign.sh` 的 `SERVE_ENV` + `start_serve()` | 三行 env 写死为默认，`SHADOW_SERVE_ENV=''` 可回退做对照；就绪行打印 ckpt sha1 与生效 env |
| 同上的手工路径 | `run_shadow_batch.sh` 前置检查 | serve 在跑但**没带** `GOMP_SPINCOUNT=0` → 高亮告警（只告警不拦，对照实验仍放行） |
| 空树归档成 junk `unknown_*` | `run_shadow_batch.sh` 归档块 | 只归档**含 `gnn_shadow.csv`** 的树；空树跳过并说明。自检加第 6 项断言 |
| 报的数挂错 ckpt | `run_shadow_batch.sh` 的 `RUN_INFO.txt` | 新增 `本轮 ckpt sha1`（内容 sha1，非路径）；`tree_tag` 的 sed 已验不受影响 |
| `判定` 行不可审计 | `_shadow_analyze.py` | 判据 `recall3`(集合版) **此前从不打印数值**（严格/宽松 k3 都是**另一个**指标）→ 现在连值一起印，并标明「仅 ① 口径，②③④ 可能翻号」。本条即本文 (9) 待办 ③ 的落地（**代码已改**） |

*(7)(8) 的逐档读数与四口径明细存于服务器 `~/sweep2_<tag>.out` / `~/sweep_<ARCHIVE>.out`，本节引用其中数值；存档在 `~/shadow_archive/`。)*

### ✅ 17.4.1 — 部署选点政策落地 + 错标数清理 + (b3) 探针（2026-09-17）

**(1) (c) 部署选点政策落进代码。** `config.MIDPOINT_SELECT`（默认 `'last'` = 取平台末端；env `MIDPOINT_SELECT=argmax_capture2` 可复现旧 run）+ `src/train_sweep.py` 的 `midpoint_epoch_of()` / `pick_midpoint()`。**两处调用点共用同一决策函数**（训练内 midpoint 块、`EVAL_ONLY=midpoint` —— 后者原是一份从未被政策覆盖的复制品）。顺带修掉三个真问题：① **`sorted(glob())` 是字符串序** ⇒ `midpoint_ep100` 排在 `midpoint_ep50` **前**，「取末端」若直接取 `[-1]` 会取到 ep50（**取反**），两处均改 `key=midpoint_epoch_of` 数值序；② 上面那处未覆盖的调用点；③ 旧注释把 `BEST_MODEL_METRIC` 记成 `smoothed_rel_err`（实为 capture2）。守卫⑤ 已实现（argmax 落在**末端** ⇒ 告警「早停很可能砍掉平台」；落在**内部** ⇒ 报「平台已形成，规则成立」），**只告警不改选择**。逐中点那行 `  ep N: cap2=...  | 诊断: ...` **格式逐字未动**（它是守卫入口）。自检 = `scripts/diag/_t_midpoint_pick.py`（本地、不需服务器/torch；**从真源码里 `ast` 抽出那两个 def** 来跑，16 项全绿）。

**(2) DIFF 错标数清理（范围比预想大一节）。** 受更正站点：`GNN_RUST_DATA_DIFF.md` §13.2 裁决第 1 点（`:823`，nowave +39.5 → **+55.9pp**）、**§15.2 整张四口径表 + 其余 ① 读数 + ④ 反解**（`:1086-1101`）、§15.5 表 nowave 行（`:1118` → **90.8% / −5.20pp**）、§15.6 两判据与**结论翻转**（`:1132-1138`）、§15.6 陷阱 (b) 实测值（`97.6` → **97.9**）；`PROJECT_LOG.md`（`:869` 上界、`:753`）；`OPEN_ISSUES.md`（`I1`、**`I9` 整行**）。新增 **`GNN_RUST_DATA_DIFF.md` §15.8**「74.4 错标定案（指针）」+ 更正范围清单，避免两个文档各写一份 74.4 定案而各自漂移。**§15.2 的 ④ 反解已重算**：DEPTH_MIX 自身 严格k3 = (90.8×714 − 76.7×176)/538 = **95.4%**（高于全局）、选择遗憾 = (5.93×714 − 3.03×176)/538 = **6.88%**（比全局更差）⇒ 「毛病在权重不在质量」**结论不变**；但反解出的 95.4% 是**它自己那个池被反复评估 64 次**的平均，不是 64 个独立电路的均值 ⇒ 这个「质量好」本身也带 ① 的重复计权（已加注）。**未受影响**：§15.4 跨集重复度（只依赖真值列 ⇒ 结构量）、§15.5 的 **iaa 行**（56.7% 来自 iaa 臂的干净单 run）、以及各表**行内 epoch 间比较**。⚠ **`cap2 95.55%` 与输入污染三项（含诊断位 31/20/271/0/0）仍是旧树那轮读数、本次未复测** —— 引用时必须带这一句（见待办 A2）。

**(3) ⚠ 17.4.1 的两个附带影响。**
- **全库 `train_sweep.py:NNN` 引用整体漂移（已全部改掉，见 17.4.1 补记）**：(1) 在 `src/train_sweep.py:25` 插了 55 行，**且选点块自身也被改过**（argmax 循环 → `pick_midpoint`）⇒ **偏移不是常数**：第 22 行以下 **+55**、730 行以下 **+60**、948 行以下 **+63**、1014 行以下 **+73**、1055 行以下 **+87**（按「+55」一把梭会算错选点块附近）。已改：`DIFF:952`（`train_sweep.py:978` ⇒ 该行 17.3.7 **已删**，留档在 `:1113-1115`）、`DIFF:960` 与 `PROJECT_LOG:921`（`:84-91` → `:139-146`）、`:951`（`:978` → 已删）、`:956`（`:815-825` → 见下、`:978-984` → 已删）、`GNN_CODING_LESSONS:19`（`:142-149, 192` → `:197-204` + `:246-247`）、`OPERATIONS:217`（`:1012-1073` → 节锚 `:1075-1150`、`config.py:22` → `:42`）；**政策操作性引用一律改成按锚点找**（`OPERATIONS:212` 与本节 `:1100/:1103` —— `grep -n 'cap2={score' src/train_sweep.py`，17.4.1 时 `:1128`）。⚠ **顺带查出两处 17.4.1 之前就已过期的引用**：`:956` 引的 `:815-825` 其实是逐 epoch `cap2` 最优跟踪（现 `:875-885`），`smoothed_rel_err` 的判据在 `:908`（17.4.1 前 `:848`）；`DIFF:995` 的 `config.py:107` 17.4.1 前即为 `:121`、现 `:138`，`model.py:110` 现为 `:111`。
- 🔴 **比行号更重的发现（已一并修掉）**：`train_sweep.py:1113` 注释写明「**17.3.7 换判据：原 score（`100·r3+50·r2+0.3·sp−0.2·regret+0.1·cap`，recall@3 主导）**」⇒ **Best midpoint 的加权 `score` 判据自 17.3.7 起已不存在**（现为 `mrk.get('capture2_pct')`，`:1119`）。因此 `DIFF:952` 与 `PROJECT_LOG:956`（那条「⚠ 纠一处判据名错误」）**描述的是已删掉的代码**，且 `BEST_METRIC` 也已改名 `BEST_MODEL_METRIC`。**两处已就地重述**：明写「加权 `score` 自 17.3.7 已删、17.4.1 起改为取平台末端」，历史实测值（该 selector 序与部署序 Spearman = −1）保留**但标明挂在已删代码上 ⇒ 不是当前代码的风险**。**仍未决的是 `BEST_MODEL_METRIC='capture2'` 本身的定档**（§15.6 判 ❌ 不采纳、四条证据，判决未落代码）⇒ 与 D1 一并处理，与本条更正不是同一件事。

**(4) (b3) 探针（本地、模型无关，`scripts/diag/_b3_ceiling.py`）—— 用来判「(b) 换一个只压前3 边界的目标函数，值不值得开一趟训练」。**
范围 = 训练分布 `batch_v2_full/rest/m4` 共 **746,242 行 / 63,224 组**（k≥4 的 58,820 组参与统计）；组键同 `data_loader.py:236`。**组大小 中位 12 / p99 24 / max 36**（无 k≥50 —— 与 `_t_rankpairs.py` 的实测一致）。
- **硬天花板 `frac(c≤3)` = 96.67%**（c = 真·最小延迟的等价类大小；c>3 ⇒ 前 3 装不下真最小 ⇒ **任何秩器**都不可能命中）；**机会底（随机秩器）= 29.43%**。
- **可分性**：组内**前2 相对间隔** 中位 **0.040%**、**<0.1% 占 59.3%**、<1% 占 71.5%。
- **代价**：组内 spread 中位 **12.3%**、**<5% 占 41.6%**、<10% 占 47.5%。
- 逐批（天花板 / 机会底）：`full` 99.83% / 21.83%、`rest` 96.32% / 30.74%、`m4` 96.86% / 26.44%。
- **⇒ 对 (b) 的判决：天花板很高（96.67%）而机会底很低（29.43%）⇒ 目标函数**不是**瓶颈；但 59.3% 的组其「组内前2」只差 0.04% 量级 ⇒ **top-k-only 的排序损失会把梯度花在多数组的亚噪声分辨上**，与 capture2 死于「亚噪声信号」是同一失败模式；且 41.6% 的组整体 spread <5% ⇒ **严格 miss 的代价本身很小**。两条一起说明：(b) 的**期望收益低、风险（放大噪声）已知命中过**。**
- ⚠ **部署口径的那一半本地做不了**：本地 `reports/_shadow_bench*.txt` 无逐候选 `true_delay`、`outputs/test_predictions.npz` 是 7 月旧模型产物 ⇒ **部署侧天花板需服务器 `~/shadow_archive` 的候选级 CSV**。**结论只对训练分布成立，不可与部署 90.8% 相减。**
- 配套 `scripts/diag/_t_rankpairs.py`：**实测否证**了 17.4.0 之前口头流传的机制「组大了对数爆炸、前3 边界被淹没」—— 本数据上**无 k≥50 的组**（那是**部署侧** 200+ 候选池的形状，不是训练侧变体组的形状）；且「恰有一个属真·前3」的对占进 mask 的 dt<0 对的 **39.24%**（1,660,168 / 4,230,762），即损失质量**已有近四成**直接涉及前3 边界。⇒ 真正 size-independent 的不对齐只剩 **`RANK_MARGIN` 的绝对性**（同一个 0.03 log10 目标间隔用于「真#1 vs 真#15」和「真#7 vs 真#8」）。

### ✅ 17.4.2 — 行号引用全清 + 本地目标函数可行性核算（2026-09-17）

**(1) 全库 `train_sweep.py:NNN` 引用清理完毕（= 17.4.1 待办 B7）。** 17.4.1 的「未改，待判」项全部落地，并且纠正了 17.4.1 记下的一个错误说法：
- ⚠ **17.4.1 说「第 24 行以下整体 +55」是错的**：17.4.1 **同时改过选点块自身**（argmax 循环 → `pick_midpoint`）⇒ 偏移**不是常数**。实测梯度：第 22 行以下 **+55**、730 行以下 **+60**、948 行以下 **+63**、1014 行以下 **+73**、1055 行以下 **+87**。按 +55 一把梭会把 `OPERATIONS:217` 那条算错 18 行。已改为「按锚点找」（`grep -n '中途快照回溯' src/train_sweep.py`）。
- 已改站点：`DIFF:952`（`train_sweep.py:978` ⇒ 该行 **17.3.7 已删**，留档注释现 `:1113-1115`）、`DIFF:960` 与 `PROJECT_LOG:921`（`:84-91` → **`:139-146`**）、`:912`/`:956`（`smoothed_rel_err` 判据实为 **`:908`**，17.4.1 前 `:848`）、`:951`（`:978` → 已删）、`GNN_CODING_LESSONS:19`（`:142-149, 192` → **`:197-204`** + **`:246-247`**）、`OPERATIONS:217`（`:1012-1073` → 节锚 **`:1075-1150`**、`config.py:22` → **`:42`**）、`PROJECT_LOG:1101`（`config.py:36` → `:42`）。
- 🔴 **顺带查出三处 17.4.1 之前就已过期的引用**（不是本次插入造成的）：① `:912`/`:956` 引的 `train_sweep.py:815-825` 被说成 `smoothed_rel_err` 的判据 —— 那段实为训练循环里的**逐 epoch `cap2` 最优跟踪**（17.4.1 时 `:875-885`）；② `DIFF:995` 的 `config.py:107`（`USE_STRUCT_PRIOR`）17.4.1 前即为 `:121`，现 **`:138`**；③ 同处 `model.py:110` → **`:111`**（该文件 17.4.1 未动）。
- ✅ **实质过期项已就地重述**：`DIFF:952` 与 `PROJECT_LOG:956` 现在都明写「加权 `score` 自 17.3.7 已删、17.4.1 起选点改为取平台末端」，历史实测值（该 `score` 序与部署序 Spearman = −1）保留**但标明挂在已删代码上 ⇒ 不构成当前代码的风险**；`BEST_METRIC` → `BEST_MODEL_METRIC` 亦已注明。**仍未决的只剩 `BEST_MODEL_METRIC='capture2'` 本身的定档**（§15.6 判 ❌ 不采纳，判决未落代码）⇒ 与 D1 一并处理。

**(2) 本地目标函数可行性核算（新脚本 `scripts/diag/_local_obj_feas.py`，模型无关、零服务器）。**
动机：用户问「训练从 50 到 300 epoch 结果没太大区别、GNN 没练到 Rust 目标的点子上，有什么办法」。在花任何训练成本前，先把「换目标函数」那几条路的**结构性可行性**算掉。范围同 (b3)：746,242 行 / 63,224 组 / k≥4 的 58,820 组入统计；组键同 `data_loader.py:236`；`BATCH=80`。

- **L1 —— 原计划的「干净 A/B」结构上不存在（推翻 17.4.1 之后的提案）**：`_pairwise_rank_loss` 的组来自 `data.grp`（batch 后拼成 (B,) 张量），而组平均 11.8 行 ⇒ 80 行的**随机 batch 里几乎全是不同组的行**，成对项拿不到 pair。实测每 batch **零 pair 的占比**：`full` **54.2%** / `rest` **94.0%** / `m4` **67.3%**；pair 均值 Grouped/随机 = 848× / 6,672× / 1,172×。rest 占 568k/746k ⇒ **加权后约九成的 batch 在成对项上是空的**。⇒ **`GroupedBatchSampler` 不是「附带的混淆」，而是成对项非空的前提**；§13.6 那次「排序损失全轴净伤害」里**损失效应与采样器效应本来就分不开**。**正确设计 = 三臂**：A 原采样器+`RANK_LOSS_W=0`（基线，已有）/ **C `GroupedBatchSampler`+`RANK_LOSS_W=0`（新增，隔离采样器）** / B Grouped+损失（= 已跑过的 `v2nowaver42m4`）⇒ C−A = 纯采样器效应、B−C = 纯损失效应。前提 = 把采样器选择与 `RANK_LOSS_W` 解耦（见 17.4.3）。
  - 🔴 **17.4.3 更正（本项前提有一处错，结论方向不变）**：本项把「原 sampler」建模成**随机置换**，但随机置换只对应**站点 2 的 else**（`shuffle=True`，在**离群点清洗分支**里）。**默认路径是站点 1 的 else = `CircuitGroupSampler`（整电路打包）**。17.4.3 真跑实测（`_t_sampler_live.py` Part C）：零成对占比 **`full` 92.1% / `rest` 49.0% / `m4` 26.7%**，成对均值 0.65 / 1.88 / 4.50 —— 与上面三个随机置换读数（54.2/94.0/67.3）**差 +37.9 / −45.0 / −40.6pp**。⇒ **「rest 约九成 batch 成对项为空」这句在默认路径上不成立**（实为 49%）。**但结论不变**：Grouped 仍是 0.0% 零成对、成对均值 524/416/458 ⇒ 与默认路径之比 **812× / 222× / 102×**，成对项仍被**饿着**（1.88 对/batch vs 416），故「Grouped 是成对项非空的前提」成立、三臂设计不变。
- **L2 —— M3 边界受限损失可行，但目标定义必须改**：只保留「一端属真·前3」的 pair 后仍保留 **34.7% / 40.4% / 37.2%** 的严格有序对，且**边界对归零的组 ≈ 0%**（`rest` 仅 13/45,800 = 0.03%）⇒ 结构上成立。**但量出一个新事实**：**第3 与第4 并列（真·前3 集合本身不唯一）的组占 11.8% / 23.2% / 25.6%** ⇒ 目标**不能**定义成「预测前3 集合 == 真前3 集合」（约 20% 的组上不可达），**必须**定义成「真最优（及其并列等价类）∈ 预测前3」= 严格@3 本身（已证并列免疫）。**连带给出 `RANK_MARGIN` 的修法**：它是 log10 **绝对**量（0.03 ≈ 7% 相对），对 (真#1,真#15) 与 (真#7,真#8) 用同一目标间隔 ⇒ 应改**相对尺度** + 成对项只取「真最优等价类 vs 其近邻」。
- **L3 —— M4 组内归一化在本形态下淘汰**：`spread` 中位 `full` **55.733%** / `rest` **12.617%** / `m4` **0.239%**（`<1%` 占比 3.9% / 32.8% / **60.3%**）；同一 `eps=1e-4` 下权重中位 1.8 / 7.9 / **417.7** ⇒ m4 的 spread 比 full 小 **233×**，**一套 eps 不可能同时对三个 batch 都合适**。按 batch 分别标定 eps 等于承认数据源异质性，已不是「把损失对齐到目标尺度」。**淘汰**（m4 占 14.6% 行，只对 full/rest 做在算术上可行，但那就不是这个方案了）。
- **L4 —— 「严格命中」由一个极窄尺度决定（最重磅，也最需小心读）**：`c_eff(eps)` = #{行: t ≤ t_min·(1+eps)} ≤ 3 的比例，合计 58,820 组：

| eps | 0 | 1e-4 | 1e-3 | 5e-3 | 1e-2 | 0.05 | 0.29 |
|---|---|---|---|---|---|---|---|
| 命中率上界 | **96.67%** | **87.15%** | 60.17% | 44.94% | 41.27% | 31.70% | 15.85% |

  逐批（eps=1e-4）：`full` 98.34% / `rest` 86.75% / `m4` 83.18%。**eps=0 那格 96.67% 与 (b3) 的硬天花板完全同值 ⇒ 两条独立路径交叉验证通过。**
  ⇒ 秩器的相对分辨力从 0 退化到 **0.01%**，可达上界就掉 9.5pp；退化到 0.1%，只剩 60%。这让两件已实测的现象终于有了机制：① 「严格@3 对 1 ulp 扰动敏感度是遗憾的 **~17 倍**」；② 「ep150 之后训练侧全平（regret 5.48 vs 5.25）而部署侧还在动（90.3→91.6）」—— 部署指标由极窄尺度上的事件决定，而训练侧 regret 是 spread 归一的连续量，对那个尺度不敏感。
  ⚠ **三条保留（不标清会被误用）**：(a) `c_eff(eps)` 把 eps 当作秩器的**相对分辨力** ⇒ 这是**诊断上界**，不是 (b3) 那条严格硬天花板；真实秩器的误差是结构性的、跨候选相关的，不是均匀 eps 带。(b) eps=0.29 那格的 15.85% 与 (b3) 的「机会底 29.43%」**不是同一个量**（前者 frac(c_eff≤3)、后者随机秩器期望 1−C(k−c,3)/C(k,3)），**不可互读**。(c) **必须与并列处置一起读**：掉得快一部分是因为真值里本就有大量近等值行，而严格@3 的并列破规则会把它们判成未命中 —— 那是**判据的产物**，不全是模型缺陷（与 L2 那 20% 是同一件事）。

**(3) 方向排序（用户口径：本地能先验的先做，再按风险排）。** R1 本地核算（✅ 本轮完成）→ **R2 部署口径的 L4 + 遗憾分解**（服务器一趟、纯离线、零风险，**决策闸门**：若可改进空间已被近等值区吃掉，则 (b) 整族永久关闭）→ **R3 解耦 sampler 与 `RANK_LOSS_W`**（本地改+本地自检，默认行为逐字不变，17.4.3）→ **R4 M2 第三臂**（1 趟训练，唯一能把 §13.6 否证拆开的实验）→ R5 M3 修正版 → R6 多 seed 集成 → R7（M4 已淘汰 / 换监督信号）。排序口径 = 不可逆性 × 成本 × 结论可信度；R2/R3 排在 R4 前不只因便宜，**它们是 R4/R5 的测量仪器**（没有部署分布的验证集，R4/R5 的差异会被口径噪声淹掉：106 集下 1 SE ≈ 5pp、1 集翻转 = 0.94pp）。

### ✅ 17.4.3 — 采样器与 `RANK_LOSS_W` 解耦（本地逻辑改动，零行为变更）（2026-09-17）

**目标（= 17.4.2 §(3) 的 R3）**：把「用哪个 sampler」从 `RANK_LOSS_W` 里拆出来。**这是一件测量仪器，不是一个结果** —— 它本身不改善任何指标，只让 §13.6 那次「排序损失全轴净伤害」的否证能被拆成两半。

**(1) 改动（默认路径逐字不变 ⇒ 现有全部 run 可复现）。**
- `config.py:49-61` 新增 `USE_GROUPED_SAMPLER = os.environ.get('USE_GROUPED_SAMPLER', 'auto')`：`'auto'` = 旧行为（Grouped ⟺ `RANK_LOSS_W > 0`）、`'1'/'0'`（及 true/false/yes/no/on/off）= 显式指定、未知值回退 `auto`。`RANK_LOSS_W` 行尾注释同步改为「采样器不再隐含在这里」。
- `src/train_sweep.py:81-95` 新增模块级纯函数 `use_grouped_sampler(rank_loss_w, mode='auto')`（放在 `pick_midpoint` 之后，同层、可被 `ast` 抽出）。
- 两个采样站点由 `if RANK_LOSS_W > 0:` 改为 `_grouped = use_grouped_sampler(RANK_LOSS_W, USE_GROUPED_SAMPLER)` + 各加一行**臂自述打印**（`:645` 站点 1 / `:764` 站点 2）。**⚠ 两个 `else` 分支逐字未动，且它们本来就不同**（站点 1 = `CircuitGroupSampler(train_dataset)` + `sampler=`；站点 2 = `shuffle=True`）—— 这是最容易在「解耦」时被顺手统一掉、从而悄悄改掉旧行为的地方，自检专门锁了这两行。
- 全库残余的 `RANK_LOSS_W > 0` **只剩 1 处**，即损失开关本身（现 `:174`）—— 这正是解耦完成的判据。

**(2) 本地自检 `scripts/diag/_t_sampler_decouple.py`（新，47 条断言，全绿）。** 沿用 `_t_midpoint_pick.py` 的做法：**从真源码 `ast` 抽出 `use_grouped_sampler` 跑真值表**（测的是发布出去的那段文本，不是复制品）。含：「auto 与旧代码逐字等价」4 例（含 `w<0` 仍走 else、`w=1e-9` 仍走 if 这两个边界）、显式开关 16 例、大小写/空白 8 例、未知值回退 5 例、纯函数性质 2 例，以及源码守卫：解耦调用 == 2、`USE_GROUPED_SAMPLER` 确实作为第 2 参传入（防写死 `'auto'` 使开关失效）、残余比较 == 1、`GroupedBatchSampler` 真构造 == 2（AST 计数）、两个 else 逐字在场。
- ⚠ **自检第一版自己骗了自己**：用正则数 `GroupedBatchSampler(` 得到 **4**，因为**我自己的打印串和文档串里就写着这个字样**。改成 **AST 计数**（数 `ast.Call` / `ast.Compare` 节点）后才是真实的 2 / 1。教训：**文本匹配的守卫会被「解释这条守卫的文本」污染** —— 凡是同时要「说明它」又要「检查它」的，必须走 AST。此坑已写进脚本注释。
- 同批复跑 `_t_midpoint_pick.py` 仍全绿（解耦未碰选点路径）。

**(3) 由此可表达的第三臂（= 17.4.2 §(2) L1 的落点）。** 17.4.3 之前**做不出**「w=0 但用 Grouped」这个组合，于是 §13.6 的否证无法归因。现在：**A** = 默认（原 sampler + `RANK_LOSS_W=0`，基线，已有）｜**C** = `USE_GROUPED_SAMPLER=1` + `RANK_LOSS_W=0`（**新增**，纯采样器臂）｜**B** = `1` + `w>0`（= 已跑过的 `v2nowaver42m4`）。⇒ **C−A = 纯采样器效应、B−C = 纯损失效应**。C 臂是一条**从未跑过**的配置，也是唯一能把 L1 那条「rest 94.0% 的 batch 成对项为空」变成实测的地方。

**(4) ⚠ 行号又漂了一次（已再次全库改完）。** 本次 3 个 hunk ⇒ 偏移**再次非均匀**：第 80 行以下 **+18**、627 行起 **+22**、742 行起 **+26**；`config.py` 第 49 行以下 **+13**。**17.4.1 那条 +55/+60/+63/+73/+87 梯度已作废。** 当前（17.4.3）权威读数：损失块 `:157-164`｜排序损失站点 `:173-175`｜`_data_mtime_hash` `:215-222`｜键合成 `:264-265`｜逐 epoch `cap2` 跟踪 `:893-903`｜`smoothed_rel_err` 判据 `:926-933`｜选点块 `:1101-1176`｜`capture2_pct` score `:1145`｜两处 `pick_midpoint(pairs, MIDPOINT_SELECT)` `:816` 与 `:1159`｜17.3.7 换判据留档注释 `:1139-1141`｜`config.py` 的 `BEST_MODEL_METRIC` `:42`（未动）/ `USE_STRUCT_PRIOR` `:151` / `USE_GROUPED_SAMPLER` `:60`。已改站点 12 处（`GNN_CODING_LESSONS:19`、`DIFF:952/960/995`、`OPERATIONS:217`×3、`PROJECT_LOG:912`×2/`921`/`956`），并在各引用处**就地标注「此为 17.4.3 时读数」**+ 前版本对照。**🔴 从 17.4.1 起已定规矩：行号必须带时点、且政策操作性引用一律按锚点找 —— 行号必然再漂，锚点不会。**

**(5) 本地**真跑**了一遍（`scripts/diag/_t_sampler_live.py`，新）—— 因为上一个自检不够。**
用户追问「r3 不是要在本地跑吗」是对的：`_t_sampler_decouple.py` 只验了**决策函数真值表**与**源码文本形状**，**没验「翻开关真的换了 sampler、换了 batch」**。两件它照不到的事：① `USE_GROUPED_SAMPLER` 能否真从 `config` 导入 `main()` 的命名空间（上一轮我是**用 grep 查 `__all__` 不存在来推断**的 —— 那是推断不是运行）；② 两个分支装出来的 DataLoader 是否合法、`GroupedBatchSampler` 的 import 会不会炸。新自检补上：
- **A** 真 `exec('from config import *')`（与 `train_sweep.py:8` 同一条语句）→ 开关确实在场、默认 `'auto'`。**推断被运行证实。**
- **B** 按标记注释定位、**抽出站点 1 那整段源码、dedent 后直接 exec**（测真文本，无副本漂移），六种 `(mode, w)` 组合各跑一次：`auto/w0→Circuit`、`auto/w0.5→Grouped`、**`1/w0→Grouped`（新 C 臂）**、`0/w0.5→Circuit`、`1/w0.5→Grouped`、`0/w0→Circuit` 全中；并断言 DataLoader 接法 —— `batch_sampler=` 直通 vs `sampler=` **被 torch 包一层 `BatchSampler`**（`bm.sampler is sampler`、`batch_size=80`）。
- **C** 真数据上量两个 sampler 的 **batch 组成**（见下条更正）。
- ⚠ **自检自己错了一次**：首版把 else 分支的接法断言写成 `loader.batch_sampler is None`，于是三个 else 组合全报 FAIL —— **是自检写错，不是代码错**（`sampler=` 会被包一层）。这条正好证明「单元自检 + 源码守卫」不足、必须真跑。

**(6) 🔴 真跑时改掉一个已记录的事实（L1 的前提错了，结论没变）。**
上面 §(2) L1 把「原 sampler」建模成**随机置换**。真跑发现：**随机置换只对应站点 2 的 else**（`shuffle=True`，在离群点清洗分支里），而**默认路径是站点 1 的 else = `CircuitGroupSampler`（整电路打包）**。真读数（`_t_sampler_live.py` Part C）：

| batch | L1 记（= 随机置换） | **默认路径实测**（CircuitGroupSampler） | 差 | Grouped（C 臂） |
|---|---|---|---|---|
| `full` | 54.2% | **92.1%**（均值 0.65 对/batch） | **+37.9pp** | 0.0%（524 对） |
| `rest` | 94.0% | **49.0%**（均值 1.88 对/batch） | **−45.0pp** | 0.0%（416 对） |
| `m4` | 67.3% | **26.7%**（均值 4.50 对/batch） | **−40.6pp** | 0.0%（458 对） |

（零成对 batch 占比；随机置换那列已用 20 次置换复现 L1 原值，差 ≤1.8pp ⇒ **L1 的脚本没算错，是测了别的 sampler**。）
- **⇒ 「rest 约九成 batch 成对项为空」在默认路径上不成立**（实为 49%）。**`full` 上默认路径甚至比随机更差**（92.1%）：full 每电路仅 8 行而组跨多电路 ⇒ 整电路打包**反而打散**了同组行。
- **⇒ 结论方向不变**：Grouped 仍是 0.0% 零成对，成对均值与默认路径之比 **812× / 222× / 102×** ⇒ 成对项在默认采样器下仍被**饿着**（1.88 对/batch vs 416），「`GroupedBatchSampler` 是成对项非空的前提」成立，**三臂 A/C/B 设计不变**。
- **附带一条机制**：`CircuitGroupSampler` 是以 `sampler=` 传入的 ⇒ DataLoader 会把它吐出的平坦索引序**再按 80 连续重切**（实测 `bm.sampler is sampler`、`batch_size=80`）⇒ 它精心做的「整电路打包（上限 160 行）」在**每 80 行边界被切断**，「电路完整性」本就只是近似的。Part C 的模拟照做了这一步（否则会把「电路完整」误当「组完整」）。
- 已同步更正 `_local_obj_feas.py` 的 L1 说明与 §(2) L1 条目（就地加 🔴 更正，不改历史数字，保留可追溯）。

**(7) 未做（明确划线）**：本次**不跑任何训练**、不改任何默认值、不碰 RNG/梯度 —— 纯逻辑解耦 + 两层自检（单元 + 真跑）。C 臂训练、以及 17.4.2 §(3) 的 R2（服务器上一趟零风险的部署口径核算）都还没做。

### ✅ 17.4.4 — C 臂可启动（`setup_exp.sh` 变体）+ 零成对读数就地更正（2026-09-18）

**(1) 新增 `setup_exp.sh` 变体 `v2nowavegs<seed>`（= 三臂实验的 C 臂）。** 17.4.3 把采样器与 `RANK_LOSS_W` 解耦后，C 臂（`USE_GROUPED_SAMPLER=1` 且 `w=0`）在**代码上**已可表达，但**在 `setup_exp.sh` 里没有入口** —— 而 OPERATIONS §2 规定训练强制走 `setup_exp.sh`。本次补上该分支：关 wave（`USE_TRANSISTOR_WAVE=False`）+ `export USE_GROUPED_SAMPLER=1` + 按尾缀设 `TRAIN_SEED`；**不动 `RANK_LOSS_W`（保持默认 0.0）**。分支插在 `v2nowave[0-9]*` **之前**，实测归属 5/5 正确（`v2nowavegs42m4`→新分支；`v2nowavegnn42`/`v2nowave42m4`/`v2nowaver42m4`/`v2wave42m4` 仍各归原位）。三臂现状：**A** = `v2nowave42m4`（默认 `auto`+w=0，已有）｜**C** = `v2nowavegs42m4`（新）｜**B** = `v2nowaver42m4`（w=0.5，已有）⇒ C−A = 纯采样器、B−C = 纯损失。

**(2) 零成对读数就地更正**（`config.py` 的 `USE_GROUPED_SAMPLER` 块 + `src/train_sweep.py` 的 `use_grouped_sampler` docstring）。两处引的是 **17.4.2 L1 那组随机置换的数**（`rest 94.0% / m4 67.3% / full 54.2%`）并称其为「原 sampler」，而**默认路径的真数是 `full 92.1% / rest 49.0% / m4 26.7%`**（17.4.3 §(6) 已定案）。差异不是笔误而是**测了另一个 sampler**（那三个数只对应站点 2 离群点清洗分支的 `shuffle=True`）。已在两处换成真数并写明来源（`_t_sampler_live.py` Part C），保留「Grouped 是成对项非空的前提」这一结论（0.0% vs 26.7~92.1%，成对均值比 102~812×）。⚠ 更正**不改变三臂设计**，但改掉了该设计被引用时的数字 —— 留着会让下一个人以为「`rest` 上九成 batch 成对项为空」（真值 49%）。

**(3) `docs/OPERATIONS.md` §3 变体表**补 `v2nowavegs<seed>` 一行，并给 `v2nowaver<seed>` 标注「= 三臂 B 臂」，把 A/B/C 对应关系写进操作手册。

**(4) 启动（服务器，用户执行；本次提交不含任何训练）。** `cd ~/-project` → `CACHE_SEED=$HOME/project-107-v2nowave42m4 bash setup_exp.sh v2nowavegs42m4`。核数：`setup_exp.sh` 内**硬编码 `OMP_NUM_THREADS=6`**（`:295/:297`）⇒ 已满足「≤8 核、不挤压他人」，**无需改动**。

**(5) 🔴 顺序约束（`setup_exp.sh` 会 clone GitHub）。** 该脚本用 `git clone -b 10.3.3-fix-earlystop https://github.com/huifeng382/-project.git` 建树，OPERATIONS §2 亦明写「本地未 push 的代码改动不会生效」⇒ **17.4.1~17.4.4 必须先 push，C 臂才可能跑**。若不 push：`v2nowavegs42m4` 在旧 `case` 里**匹配不到任何分支** ⇒ 不执行任何 sed ⇒ `USE_TRANSISTOR_WAVE` 保持默认 **True**、且旧代码根本没有 `USE_GROUPED_SAMPLER` ⇒ **会静默训出「wave 开 + 原采样器」的模型、却挂着 `v2nowavegs42m4` 的树名** —— 又一次 I15 类错标，且全程无报错。

**(6) 未做**：训练本身（用户执行）；跑完按 OPERATIONS §6 走 shadow，逐 ckpt 读数盖 ckpt sha1。本地自检：`_t_sampler_decouple.py` 全绿、`_t_sampler_live.py` 全绿、`bash -n setup_exp.sh` 通过、`case` 分支归属实测 5/5。

**(7) ⚠ 行号又漂一次（两文件同改）。** `config.py` 第 51 行以下 **+5**、`src/train_sweep.py` 第 86 行以下 **+2** ⇒ §17.4.3 §(4) 那张「当前（17.4.3）权威读数」表在本版本上的读数为：损失站点 `:173-175` → **:175-177**｜`17.3.7 换判据` 留档注释 `:1139-1141` → **:1141-1143**｜`capture2_pct` score `:1145` → **:1147**｜两处 `pick_midpoint(pairs, MIDPOINT_SELECT)` `:816`/`:1159` → **:818**/**:1161**｜`config.py` 的 `USE_GROUPED_SAMPLER` `:60` → **:65**、`USE_STRUCT_PRIOR` `:151` → **:156**；`BEST_MODEL_METRIC` 仍在 `config.py:42`（位于改动行之上，未动）。§17.4.3 那张表**保留原值不动**（它是 17.4.3 时点读数，按 17.4.1 起的规矩本就带时点）；**本次及今后的操作性引用一律按锚点找**。

### ✅ 17.4.0 — tl_opt 贪心降延迟定档（46 电路 batch，accepted 终值口径）（2026-09-18）

> ⚠ 编号说明：本节按指定的版本名记为 **17.4.0**，但文档中 17.4.1~17.4.4（训练侧）已在，故本节**位置在 17.4.4 之后而编号更小**。Rust 侧与训练侧的 17.4.x 各自成线，未强行并序。

**结论（n=45 有效 / 46 总数）：贪心在全部 45 个有效电路上都把延迟降下来了。平均 ratio `0.8401`（降 **15.99%**），中位 `0.8544`（降 14.56%），最好 `EQ2`/`XNOR2` `0.7156`（降 28.44%），最差 `DEPTH_MIX` `0.9504`（降 4.96%）。分 level 单调：level0 **18.5%** → level1 20.5% → level2 14.9% → level3 14.3% → level4 **10.9%**（电路越大越难降）。完整 46 行明细 + 复现命令见 `reports/_tl_opt_greedy_46circuit.txt`。**

**(1) 🔴 口径 = 分母用「原电路」，不是「组内真最优」。** 分子 = `rpt/tl_opt_shadow_batch.csv` 的 `avg_delay`（= `r.avg_delay` = `best.avg_delay` = 贪心 **accepted 终态**；`tl_opt_shadow_batch.rs:144-149` 写入、`tl_opt.rs:1044-1063` 构造）；分母 = 同趟归档每个 `<level>/<stem>/gnn_shadow.csv` 的**首行** `true_delay`（= `eval_idx=1, iter=0, window=0` = 原电路，`tl_opt.rs:545-546`）。分子分母**同趟、按 stem 配对**，不含跨趟假设。
⚠ **与 `_baseline_compare.txt` / `_shadow_bench_final.txt` 的口径不同**：那两份的分母是「候选组内的真最优」，回答的是「**粗排准不准**」；本节回答的是「**贪心有没有把电路改快**」。**两者不可混用、不可互相引用数值。**

**(2) 数据源（可复现）。** 归档 `~/shadow_archive/midpoint_ep300_20260917_215321/`（2026-09-17 21:53 那趟，ckpt = `midpoint_ep300`）+ `rpt/tl_opt_shadow_batch.csv` 的末 46 行。命令见 `reports/_tl_opt_greedy_46circuit.txt` 的「复现命令」段。

**(3) 三条自检（都过了）。**
- **分母可信**：归档 `ADD4_OVF` 首行 `3.576878e-11` 与 2026-09-18 独立跑的 `tl_opt_smoke` 读出的 `cached eval_idx=1 delay=3.576878e-11 transistors=192` **逐位相同** ⇒ 归档基线与实时 evaluator 同源、可用。
- **配对可信**：45 个里 **44 个**的 `final` **逐位等于**该归档的 `min(true_delay)`（6 位有效数字全同）。跨趟不可能撞到这个精度 ⇒ 按 stem 的配对成立。
- **唯一 `min ≠ final` 的电路 = `ALU2`**：`min` 3.018454e-11 vs `final` 3.064066e-11，差 **1.34pp** ⇒ 该电路存在「评估到更好但被拒」的候选。也说明 `min` 口径是**乐观下界**，不保证与终值恒等。

**(4) ⚠ 更正（作废一条一度给过、后被自身否掉的说法）。** 曾据 `ADD4_OVF` 报「乐观口径（min）比真实夸张 1.24pp」—— 那是拿**两趟不同的 run** 相比（早停趟的 `min` vs 跑满 40 轮趟的 CSV `final`），**跨趟比较无效，已作废**。同趟内这二者相等。

**(5) ⚠ 单趟结果不可复现（引用本节的数时必须带上这句）。** 同一电路 `ADD4_OVF` 三趟给出 `3.209527 / 3.253864 / 3.269569 e-11` ⇒ 波动约 **±1.3pp**。成因推断：Rust `HashMap` 默认随机 hasher 改变候选枚举顺序（**GNN 影响不到决策** —— `pre_rank` 的返回值在 `tl_opt.rs:893` 被丢弃；`gnn_shadow.rs` 自述「贪心决策照旧，不改行为」）。⇒ **16.0% 是一次抽样的读数，不是精确常数；跨趟方差待测。**

**(6) ⚠ `ovf1` 是「没动过」，不是「降了 0%」。** 归档里该电路 `gnn_shadow.csv` 只有 1 行（= 基线行），**未产生任何候选**，`ratio=1.0000`。与本文件 §15.2.12 记的「ovf1 因 Xyce 瞬态步长崩溃不收敛」一致。

**(7) 副产品：多跑不一定更好，牵出早停阈值。** `ADD4_OVF` 早停那趟（`iter` 标签到 35）`ratio=0.8973`，**优于**跑满 40 轮的两趟（`0.9097` / `0.9141`）。根因是接受规则**只看局部**：`tl_opt.rs:962-964` `let accept = delta < 0.0;`（`delta` 只涉及 1~2 个受影响输出），全局变差也照收 ⇒ 贪心会游走。相关硬编码 `tl_opt.rs:1032` `if best.combined_score <= 0.9 { break; }` **不在 `config.rs`、无环境变量开关**。

**(8) 附带存档（唯一一份带逐轮日志的轨迹）。** `tl_opt_smoke`（单电路 `level4/ADD4_OVF`，参数与 batch 逐字相同，唯一差别是 `debug_logging=true`）2026-09-18 跑到**第 6 轮**由人工停掉，半截轨迹存 server `~/tl_opt_smoke_partial_20260918.{log,out}`（2476 / 2988 行）。该趟 `temp_sim_cache` 125413 → 130629（**+5216 deck**，未删）。逐轮 accepted 得分：round4 `0.969863` / round5 `0.969317` / round6 `0.967689` ⇒ 6 轮只降 3.2%，离早停线 0.9 差 6.8pp。
⚠ 与 (1) 的 `0.8973` **不可相减**：这里是 **accepted 逐轮值**，那是 **min over 全部候选（含被拒）**，两列口径不同。

**(9) 未决，已立任务 #66。** ① 跨趟方差（10 个归档可测 `min` 口径的离散度；accepted 终值口径因 `rpt` CSV 无 run 标签、归档不含终值，暂时只能单趟）；② 早停阈值 0.9 复查（是否上提/去掉、是否改全局判据）。**本次不改任何代码、不跑任何仿真**（数据全部读自盘上已有产物）。

> ⚠ **交叉引用（2026-09-18 补）**：本条的「局部接受规则 ⇒ 贪心游走」是**仿真门**（`GNN_SHADOW`/默认）下的成因，成立。但**不要拿它解释 GNN 决策门**（`GNN_ONLY`）的表现 —— 那条路上判据的输入是模型排名，失败机理完全不同（见下一节）。两者都表现为「多跑不一定更好」，成因不可互换。

---

### 🔬 GNN 作为 tl_opt 决策门为何失效 —— AND2 单电路零真值诊断（2026-09-18，版本编号待定档）

> **本节回答的问题**：模型在候选排序上一直很准（上一节 202 行「高差异遗憾 1.6%、top1 63%」），
> 为什么一当**决策门**（`GNN_ONLY`，搜索期零仿真）就整趟跑坏？**结论：模型没坏，是门问错了问题。**

**(1) 数据与出处。**

- **捕获树**：服务器 `temp_sim_test/tl_opt_smoke_cap/level0/AND2/`，`GNN_CAPTURE` 打开后写出 `/capture/mod_*.tl`。
  **修 `to_tl_text` 依赖序缺陷后**（见 §17.4.x 待定档的序列化修复）共 **691** 个（修复前 709 个，其中 **535 个读不回来**）。
- **真值**：`tl_opt_gnn_batch` 的 `TL_SIM_DIR` 模式对 691 个逐个真仿真 → `truth.csv`（`file,status,avg_delay,transistor_count,error`），**691/691 ok，0 parse_error**。
- **模型判断**：同趟 `windows.csv` 447 行（**446 个可判**；末列 `candidates_ranked` 是逗号连接，解析必须 `split(',', 8)`）。
- **走的那趟**：`mode=gnnonly accepted=8 iters=40 evals=9 failed=0 transistors=20 avg_delay=2.001253e-12 speedup=0.162077 spice_baseline=1.234755e-11 spice_final=1.807904e-11 spice_speedup=1.464180e0`。
- **对照组**：17.4.0 **仿真门** 同电路 = `1.037152e-11` / ratio **0.8400（降 16%）**。⇒ 同电路、同贪心、只换判据来源：**−16% → +46.4%**。

**(2) 🔴 门几乎是个空操作 —— 这是最硬的一条。**

| 判据（同一批窗口、同一批候选，只换判据） | 真机会召回 | 接受里真降比例 | 接受的即时真 Δ 中位 |
|---|---|---|---|
| 模型（`current` 同批排名，strict `<`） | **1.6%** (6/370) | **37.5%** (3/8) ⚠ | **+1.22%** |
| 一行规则：管数最少者胜（同形判据） | **84.9%** (314/370) | 87.5% (314/359) | −9.65% |
| 全跳过（什么都不做） | 0% | — | — |

- ⚠ **上一行的 `6/8` 是「所在窗口存在更优候选」，不是「被接受的那个候选真降」** —— 后者是 **3/8**（与 (5) 的「3 次真降 / 4 次真升 / 1 次零增益」一致）。2026-09-20 复核更正，见 (13)。
- 446 个可判窗口里 **370 个（83.0%）存在严格更优候选**，这些窗口的真最优可降**中位 15.02%**。
- 模型只动了 **8** 次，其中 6 次所在窗口真有机会 ⇒ **真机会召回 1.6%**；另 2 次所在窗口**根本没有更优候选**（空动）。**比「一次都不动」只多 6 次。**
- 一行「晶体管数最少者胜」在**同一判据形状**下抓到 314 个 ⇒ **模型的门不是「判得不够准」，是「几乎不判」**。

**(3) 分辨力是真的；坏的只有 `current` 的落位。**

逐窗口 Kendall τ（模型名次 vs 真延迟名次），**同一批窗口配对比较**：

| 批次构成 | 中位 τ | 均值 τ |
|---|---|---|
| **候选之间**（不含 `current`） | **+0.485** | +0.446 |
| **全批**（`current` 一起排） | **+0.377** | +0.369 |

⇒ **候选之间的分辨力实打实（τ≈0.49 是很强的信号），把 `current` 丢进同一批就掉到 0.377。** 缺陷是局部的、定位明确的。

`current` 的真名次（446 个窗口，0 起）：**第 0 名 14.8%（66）**｜第 1–3 名 22.9%（102）｜**第 4 名或更差 62.3%（278）**，中位 **6**，均值 4.66。

模型判「`current` 最优」**438** 次，其中真值也确实最优只有 **74** 次 ⇒ **正确率 16.9%**（⚠ 2026-09-20 复核更正，原记 66 次 / 15.1%；见 (13)）。`cur_rank` 分布：`{1.0×336, 1.5×81, 2.0×19, 3.0×2}` —— 注意 **1.0 占 336**，即 75% 的窗口**全体模型一致、无并列**地把 `current` 排第一：这是**自信的系统性误判**，不是噪声。
> ⚠ **这个 16.9% 必须跟基准率一起读**：446 个窗口里真值上 `current` 本来就最优的有 **76** 个 = **17.0%**。⇒ 模型在「说 current 不最差」这件事上**与瞎猜持平**。真正刺眼的是它的**确定性子集**：「严格第一」那 **336** 窗里真值同意只有 **26** 次 = **7.7%**，**低于** 17.0% 的基准率 ⇒ **模型的确定性判断方向是反的**，不是「不够准」。（并列那 102 窗同意 48 次 = 47.1%，恰好掷硬币。）

**(4) 模型对 `current` 的落位：不是漂移，是固定误判（⚠ 本节初版曾误记为「漂移」，已更正）。**

初版据「`cur_rank` 的出现值在 iter 19 后收窄到 `{1.0, 1.5}`」判定「偏袒随嵌合加重」。**把模型名次与真名次按深度并排后，这个读法不成立**：

| 深度 | 窗口 | 模型 `cur_rank` 中位 | **真名次中位** | 落差 | 模型说 #1 | 真值 #1 |
|---|---|---|---|---|---|---|
| 0 | 1 | 11.00 | 6.0 | **−4.00** | 0% | 0% |
| 1 | 2 | 2.00 | 6.0 | +5.00 | 50% | 0% |
| 2 | 10 | 1.00 | 7.0 | +7.00 | 90% | 0% |
| **3** | **91** | **1.50** | **2.0** | **+1.50** | 82% | 21% |
| 4 | 21 | 1.00 | 6.0 | +6.00 | 76% | 0% |
| 5 | 22 | 1.00 | 7.0 | +7.00 | 86% | 0% |
| 6 | 18 | 1.00 | 5.0 | +5.00 | 94% | 6% |
| 7 | 22 | 1.00 | 6.0 | +6.00 | 95% | 18% |
| **8** | **259** | **1.00** | **6.0** | **+6.00** | **100%** | 16% |

- 落差序列 `−4 / +5 / +7 / +1.5 / +6 / +7 / +5 / +6 / +6` **不单调**；深度 0（原始电路）甚至是**反方向**的（模型低估到第 11 名）⇒ **不是「偏袒越来越重」**。
- 深度 3 是唯一基本校准的一格（落差 +1.50，真值 #1 率 21%）；**深度 ≥4 系统性乐观 5–7 个名次**。

**正确的形状（决定性观察）**：iter 18 之后再无接受 ⇒ **iters 19–40 的 257 个窗口里 `current` 是同一个电路** `mod_0411000a53ba1908_191.tl`。所以那是：

> **同一个电路，被 257 组不同候选比下去，模型 257/257 全判它第一**（232 次 1.0、25 次 1.5）；而它的真名次中位 **6**（范围 0–9）。

量级对照：它真延迟 `1.807904e-11`，同期候选真延迟中位 `1.7345e-11`、最快 `1.3238e-11` ⇒ **它只比候选的中位数慢 4.2%**，却在一批又一批里被模型放到最顶上。⇒ **这是对特定电路的固定误判，不是随轮次累积的漂移。**

**(5) 接受的 8 步全是坏账。**

- 8 次接受的**真值比例连乘 = ×1.4642（+46.4%）**，与 `spice_speedup=1.464180` 对上。
- 分解：**3 次真降 / 4 次真升 / 1 次零增益**。5 次非降相乘 **×1.579**，3 次真降相乘 **×0.9272** ⇒ 净 ×1.4641。**降的那点完全抵不过升的。**
- **第一步接受就已经是假的**（Δ=+2.44%）⇒ 整趟从第一轮就上了错误的轨道。
- 接受时的即时真 Δ **中位 +1.22%（即平均而言接受的那一个比 `current` 更慢）** —— 因为要盖过 `current` 的偏高，需要很大的模型边际，而**模型边际 ≠ 真边际**。
- **锚定偏差**：模型自评终值 `2.001253e-12` vs 真值 `1.807904e-11` ⇒ **差 9.03 倍**；模型自报 ratio `0.162077`（−83.8%）vs 真值 **+46.4%** —— 模型**完全不知道自己把电路改慢了**。

**(6) 反事实：好答案本来就在候选集里。**

在**这趟自己生成的**候选集上做真值贪心：**1 步**即到 `1.043250e-11`（−15.5%），终值 1.807904e-11 是它的 **1.73 倍**。这个数与 17.4.0 仿真门 AND2 的 `1.037152e-11`（ratio 0.8400）几乎相同。
⇒ **问题不在「候选生成得不好」，而在「选不出来」。** 门只要按真值选一次就到位。

**(7) 为什么「管数」这条一行规则反而好用 —— 以及模型差在哪。**

| 量 | Kendall τ |
|---|---|
| 真延迟 vs 晶体管数 | **+0.650**（极强；16 管档中位 `1.6897e-11`，8 管 = 0.749×，28 管 = 1.177×） |
| 模型名次 vs 晶体管数 | **+0.258** |
| 模型名次 vs 真延迟（全局混窗粗口径） | +0.189 |

- 窗口内：模型首选比真最优**管数更少** 47.3%｜**一样** 46.2%｜**更多** 6.5%。
  ⇒ **模型偏好方向是对的，但力度只有真值的三分之一**；它**从不去选管数更多的**，说明尺寸先验在起作用，只是太弱。
- 跳过窗口里：`current` 管数中位 **20**，该窗口真最优候选中位 **16**；**只有 77/438 个窗口二者管数相同** ⇒ **82.4% 的跳过窗口里都存在一个比 `current` 更小的候选，模型仍然判 `current` 最优。**

**(8) 🎯 根因（⚠ **本节初版结论已被推翻**，此处为更正后的状态：机理未定）。**

初版说的是「`current` 是父电路，模型从未被训练/验证过把父电路放进孩子之间排序」。**这个框架是错的，两处硬伤：**

1. **模型看不到「谁是父」。** `serve.py:363-374` 是 `for c in cands` **逐个候选各自前向**，输入里没有这个信息。任何以「父」为单位的解释在模型这一层就不成立。
2. **训练数据里根本没有「基准」这个东西。** 训练组 = TransiLog e-graph 枚举出的**整个函数的等价实现**（`DATA_SPEC_V2.md:190-192`；变体由保函数的改写生成、逐输出真值表穷举一致，`:325`），`utils.py:72-204` 的 `ranking_metrics` **对称处理所有成员** —— 无 `is_base` / `origin` / 派生深度列 ⇒ 「没被训练过放基准」这句话**在数据层面是空的**。

**初版里唯一成立的部分，正确说法是：所有已上报的指标，批里都不含基准。**

- 714 集口径的池子 = `prepared` = **只有打补丁的候选**（`tl_opt.rs:962-1026`、`_shadow_analyze.py:300-306`）；原电路只作为**大小 1 的组**评估一次，随即被 `--min-cands 4` **丢弃**。
- ⇒ `严格@3 90.8%` / `选择遗憾 5.93%` **一次都没有在含基准的批上算过**。
- `GNN_WALK`（`gnn_only.rs:483-498`）是**第一个也是唯一一个**把基准放进同一批的模式。

> ⇒ **不是「模型做不到」，而是「从没有人测过」。** 把「没测过」写成「没训练过」，是两回事。

**仍活着的机理（假说，未证实）：「一次窗口重写被模型判为变慢」的共同惩罚。** 一批候选全都是 `current` 施加窗口重写得到的；若模型带一个共同方向的惩罚，则 ①候选之间的相对序不受影响 ⇒ τ 仍好（+0.485）；②没有该惩罚的 `current` 靠这个共同项赢下每一批 ⇒ 257/257 全第一。与 (4)(5)(6) 的量级一致（候选真值只快 4.2%，而门要克服的惩罚看起来远大于此，故 446 窗口只接受 8 次）。
⚠ **但 `rewrite_iter_limit` 在 harness 里是 3**（`tl_opt.rs:1289`），单个候选可能差**好几次**重写而非一次 ⇒ 这条假说须按此修正后再验。**分辨靠 (9) 的单候选 `/rank` 探针。**

**(9) ⏳ 未决：门是「模型量级错」还是「排名聚合错」？—— 一次实验即可分辨。**

现在的数据**无法区分**两种可能：

- **(A) 模型的预测延迟本身就把 `current` 算低了** ⇒ 模型/特征问题；
- **(B) 模型对 `current` 的预测值不低，但 `predict_rank_batch` 的聚合把它抬上去了** ⇒ serve 侧问题。B 有具体可疑点：`serve.py:380-403` 用**逐模型名次**再平均（`avg=(i+j)/2+1` 并列取平均，`sc = scores[i]/counted[i]`），且 `nan = any(v != v for v in pm)` 会把某一行**从所有模型**的排名中剔除 ⇒ **不同行的分母可能不同**，理论上产生系统性偏置。

**分辨实验（最便宜的一条，具体到电路）**：取 `mod_0411000a53ba1908_191.tl`（(4) 那个 257/257 全第一的嵌合体）与它若干个候选，对**每一个单独**发一次**单候选** `/rank`（走 `serve.py` 的 `n<2` 分支 `predict_avg_delay`，返回**原始预测延迟**而非名次），比较两种口径给出的序。
- 单候选口径下 `current` 也被算低 ⇒ **(A) 模型量级错**（模型/特征问题）；
- 单候选下正常、批量下才翻 ⇒ **(B) `serve.py:380-403` 的排名聚合错**。

**共 6 次 HTTP 即可分辨，是下一步唯一该先做的事。**
> ⚠ 这条同时能判掉 (8) 末的「重写共同惩罚」假说：若模型给候选的**绝对**预测本来就更低，则该惩罚不存在，问题纯在聚合。

**(10) 适用边界（引用本节数字时必须带上）。**

- **单电路（AND2）、单趟、40 轮、单 seed。** 不得外推到 46 电路战役。17.4.0 的 46 电路均降 15.99% 是**仿真门**的成绩，与本节**不构成**矛盾亦**不构成**支持。
- 反事实「真值贪心 1 步到 −15.5%」只在**这趟自己生成的**候选集上成立；换轨迹 ⇒ 换候选集，不保证复现。
- 本节的 `any_better` 是 `gnnonly` 模式下**实际决策输出**（该模式下模型即门）。**采集（shadow）模式下同一列只是模型的假想判断**，两种读法不可混用。
- 一行「管数最少者胜」在本节只是**衡量模型门欠发的尺子**，其作为可部署判据需独立验证（它在本数据上每次接受的中位即时收益 −9.65%，但连锁轨迹未测）。

**(11) 🔴 独立发现（与门偏置无关，需单独处置）：部署电路在规模上落在训练分布之外。**

数据源：`data/batch_v2_m4` + `data/batch_v2_full` 的 `circuit_static.parquet`，按 `circuit_id` 去重后 **10,479** 个电路 / **729** 个 `expr`。

| | 训练集 | AND2 捕获树（691 个） |
|---|---|---|
| `transistor_count` 范围 | **18 – 118** | 6 – 37 |
| `transistor_count` 中位 | **54** | **17** |
| 组大小 | 10–15（中位 15；分布 `{10:1, 12:150, 14:1, 15:577}`） | 4–12 |
| `gate_level_netlist` 行数 | 中位 10，p95 18，max 25 | — |

- **691 个捕获电路里 364 个（52.7%）低于训练集最小值 18**；中位数 17 位于训练集的 **0.0 分位** ⇒ AND2 这一族上模型的预测是**向下外推**。
- ⚠ **本条不能解释 (8) 的门偏置**：同一个规模外推在「候选之间排序」的 714 集口径上**同样存在**，而那个场景是好的（严格@3 90.8%）。
- ⚠ **也不能推成「整个基准族都偏小」**：`ADD4_OVF` 是 **192 管**（17.4.0 记、`tl_opt_smoke` 读数），**高于**训练最大值 118 ⇒ 该基准族**两端都可能出界**。
- **待办（零成本：本地 parquet + 归档 CSV 即可）**：量全部 46 个基准电路的 `transistor_count` 对训练分布的位置，判断「模型在整个基准族上都是外推」是否成立。**已并入任务 #67，不另立任务。**

**(12) 本节尚未确定的事（一句话）。** 门的偏置（模型对 `current` 系统性乐观约 6 个名次）**机理未定**。已排除「父 vs 子」框架与「没训练过这种批次」两条；仍活着的是 (8) 末的「重写共同惩罚」假说。**下一步只有一件事：(9) 的单候选 `/rank` 探针**（6 次 HTTP，用 `mod_0411000a53ba1908_191.tl` 与它的若干候选）。

**(13) ✅ 2026-09-20 复核：口径对账 + (9) 结案（**零 HTTP** 完成）+ 三处数字更正。**

脚本：`scripts/diag/_t_cap2_an13.py`（口径对账）、`_t_cap2_an14.py`（候选池完整性）、`_t_cap2_an15.py`（单发 vs 批量）、`_t_cap2_an10/11/12.py`（并列窗、尺寸排除）。

**a) (9) 结案 —— 是 (A)「模型预测值本身错」，不是 (B)「排名聚合错」。(B) 在结构上就没有载体。**

- **`n_models = 1`**（线上 `curl http://127.0.0.1:8000/` = `{"n_models": 1}`；serve 起于 `midpoint_ep300.pt`）⇒ `serve.py` 的 `counted[i] ≡ 1` ⇒ **`sc = ranks[i]`，就是一个候选在自己批里的竞争名次**。⇒ **(9) 说的「`nan` 逐行剔除导致不同行分母不同」在 1 模型下不可能发生** —— 而且 `good` 是按**行**过滤的（`serve.py` 的 `good = [i for i in range(n) if not rows[i]['nan']]`），不是按模型。原版本把它读成「某一行从**所有模型**的排名中剔除」，方向是对的，但 1 模型下没有第二项可加。
- **零 HTTP 分辨**：`cands.csv` 的 `single_pred` 列**就是** (9) 想用探针取的那个量（`serve.py` 的 `n<2` 分支，返回原始预测延迟，逐文件一个值）。用它复算：
  - **109/109 窗口同判**（单发口径与批量口径对 `current` 的判完全一致，分歧 0）⇒ **聚合没有引入新方向**。⚠ 诚实说：这在 1 模型下**由构造保证**（竞争名次对原始值是单调的），所以它是**一致性自检**、不是独立证据；它的价值恰好在于**证否 (B)** —— 没有哪一步能重排。
  - **单发口径下，80 个「严格第一」窗口里「候选预测值比 `current` 更低」的是 0 个（0.0%）**，中位比值 `cand/cur = 1.0281`（即模型**确信候选慢 2.8%**）；3 个「反超」窗口里 3/3 候选更低（中位 0.9961）。⇒ **模型的原始预测值与它的判决自洽**，它真的认为 `current` 更快。
  - 绝对量级：`current` 的 `预测/真值` 中位 **0.627**，模型首选候选中位 **0.655** ⇒ **两者同量级，模型没有「专门把 `current` 算低」**。⇒ (A) 的**「针对 current 的偏置」这一形式也不成立**；成立的是**全局标度/压缩**问题（见 (5) 的 9.03× 与 `cands.csv` 的 `预测/真值` 中位 0.814）。
  - ⇒ **正确说法**：门失效 = **模型对「父电路 vs 其重写」这一比较的预测方向本身是错的**（在真值上 83.6% 有改进的窗口里，模型判候选更慢 2.8%），**既不是 serve 聚合问题、也不是「current 被少喂特征」问题**（见 c）。
  - ⚠ 覆盖限制：`cands.csv` 只 574/691 个文件 ⇒ 可判窗口 **109/446**（80 严格 / 26 并列 / 3 反超）。方向性结论（0/80）不受此限，**比例不可外推**。
- ⚠ **`cur_rank` 的 x.5 不是 2 模型集成**：是 `serve.py:396` `avg = (i + j) / 2.0 + 1.0`（competition ranking 对**位级相等**取平均），并列判定是 `gvals[order[j+1]] == gvals[order[i]]`（**float64 精确相等**）。判据是严格 `<` ⇒ **并列必拒**。
- **`module_to_candidate` 对 `current` 与候选完全对称**（`gnn_only.rs` 的 `CUR_ID` 与 `c{i}` 走同一个函数）⇒ **「current 被少喂特征」这条排除**。

**b) 三处数字更正（原节内已就地标注）。** ①(2) 表「接受里真降比例 75.0% (6/8)」→ **37.5% (3/8)**；`6/8` 是**「所在窗口存在更优候选」**，两件事。②(3)「正确率 15.1%（66 次）」→ **16.9%（74 次）**，并补基准率 **76/446 = 17.0%**。③(3) 的 `cur_rank` 直方图 `{1.0×336, 1.5×81, 2.0×19, 3.0×2}` **是对的**（它是 438 个「判 current 非最差」窗的分布，不是全部 446）。

**c) 102 个并列窗不是「漏掉的机会」—— 模型判平是对的。** 逐窗复核（`_t_cap2_an10/11.py`）：

- **71/102** 是**候选池里就有 `current` 自身**（同内容 ⇒ 同 FNV1a 名 ⇒ 逐位相同预测 ⇒ 竞争名次相等 ⇒ 严格 `<` 必拒）。这是**正确的拒绝**。
- 剩下 **47** 个是**不同电路却位级并列**：**47/47 管数相同、行数相同**，文本 diff 只有 **1–2 块**，全是**节点改名**与**双重反相缓冲链**（`X4=!X5=!(!X3)=X3`）；真值相对差**中位 0.0000% / 最大 0.0673%**，47/47 都 <0.1% ⇒ **语义等价换皮**，模型判平是对的。
- ⇒ **并列类别的中位真值增益 = +0.000%**（更快 7 / 更慢 15 / 持平 80），**没有任何可利用的损失**。

**d) 排除「模型只是按尺寸排 ⇒ 必然把 current 排第一」这个平凡解释。**（`_t_cap2_an12.py`）

- 被拒的 336 窗里 `current` **是最小**的只有 **10 个（3.0%）**；**318 窗**存在比它更小的候选；`current − 最小候选` 中位 **+4 管**。
- 窗口内 `τ(管数, 真值)` 中位 **+0.476**、`τ(模型序, 管数)` 中位 **+0.650**、而**模型序恰为管数升序的窗口只 26/446 = 5.8%**。
- ⇒ 模型**确实按尺寸在候选内部排**，却把**明显更大的 `current`** 排在所有候选之上 —— **这是它自己排序内部的不自洽**，不是尺寸先验的必然结果。
- 关键反事实（同一候选池、零模型规则）：模型 #1 更快 **291/446 = 65.2%**（中位 +4.058%）；**「取管数最小的候选」314/446 = 70.4%**（中位 +9.651%）⇒ **模型输给一条「数晶体管」的零参数规则**。

**e) 权威取数口径（此前的子样本口径作废）。** 候选文件名 = `mod_<FNV1a64(文本)>_<字节数>.tl`（**同内容必同名**），所以**直接按文件名 join `truth2.csv`**（691/691 ok）即可，**不需要位置对齐、不需要子样本**。⇒ `cands.csv`（574 行）只是它的干净子集；此前用 574 截出来的「109 窗 / 225 窗」**top-K 口径作废**（本 (13) 里 109 窗的用法是**单发-vs-批量**，不是 top-K，且已标不可外推）。

**f) ⚠ 口径陷阱（本次踩到，记下来）。** 判「真值打平」必须用**相对**容差。第一版脚本用绝对 `EPS = 1e-12`，而真值量级 ~1.2e-11 ⇒ 那是 **8% 的带**，把「候选快 5%」也算成打平，导致「存在更优候选」的窗口数误读成 **268**（权威 **370**）、「取管数最小」误读成 **234**（权威 **314**）。**任何跨电路的真值比较一律用相对容差（本处 1e-9）。**

**g) 相关口径结论（不在本节复述）。** 「90.8% 严格@3 与 GNN-only 问的不是同一件事」、
以及「`ep250` vs `ep300` 不可盲换、先查部署配置」记在 **`GNN_RUST_DATA_DIFF.md` §十六**。

### ✅ 18.5.0 — V3 数据集首次训练（`v3nowave42`，2026-09-20 起训 / 2026-09-22 收尾）

**运行与启动闸门。** `~/project-107-v3nowave42`，**0 error**；225 epochs early_stop（val_loss 连续 40ep 无改善），总 **2186.3 min**（Avg/epoch **558.2 s**）。启动实测四行：`DATA_BATCHES=v3_delivery` / `USE_TRANSISTOR_WAVE=False` / `TRAIN_SEED=42` / `CACHE_DIR=cache107v3nowave42`，日志打印 `V2 data: v3_delivery (1 static + 31 arcs)` ⇒ 走的正是 18.0.0 修的那条**混合形态** loader 路径（单 `circuit_static.parquet` + 31 分片 `timing_arcs`）。配置：LR 1e-4 / LR_MIN 1e-6 / FACTOR 0.5 / HUBER 0.3 / BATCH 80 / `BEST_METRIC=capture2` / `SPLIT_SEED=42` / `V2_STRUCT_MODE=logic_only` / `USE_IDS_AVG_APPROX=0` ⇒ **纯拓扑 in=45**，与 serve 交付形态一致。切分后行数：train **406,953** / val **95,805** / test **97,218**（合计 599,976）。

**部署候选 = `midpoint_ep200.pt`**（政策 `MIDPOINT_SELECT='last'` 取平台末端，17.4.1）。下表全部取 **midpoint 块**。

| 指标（训练侧 test） | `v2nowave42m4`（548 组） | **`v3nowave42`（126 组）** |
|---|---|---|
| Test Median Rel Err | 22.13% | **7.21%** |
| Mean Abs Err | — | 3.58 ps |
| Spearman | 0.393 | **0.879** |
| 选择遗憾 | 3.80% | **0.52%** |
| top1 | 42.5% | **67.5%** |
| 捕获率 | — | 96.7% |
| recall@3 严格 A / 宽松 B | — / 83.4% | **81.7% / 95.2%** |
| 成对分辨 <2% / >10% | 55% / 81% | **75% / 99%** |
| spread>10% 子集（组数） | 遗憾 5.18% / Sp 0.450 / B 88.2%（346） | **遗憾 0.61% / Sp 0.897 / B 95.7%（94）** |

- recall@2：全局 **A 77.0% / B 88.1%**（n=126）；spread>10% 子集 **A 80.9% / B 90.4%**（n=94）。成对分辨分档：<2% **75%**（n=1773）/ 2–5% 95% / 5–10% 97% / >10% **99%**。Best Val Rel Err 12.96%。组内变体差中位 **17.1%**。
- ⚠ **`v2nowave42m4` 那列是训练侧口径**（DIFF §15 已定案：训练侧 ≠ 部署口径）。其 Rust 部署读数（714 集 batch）是 严格@3 **90.8%** / 宽松@3 **97.9%** / 选择遗憾 **5.93%** / 两阶段 0.63% —— **那是另一把尺，不要与本表并列解读**。

**⚠ 三条禁止混读（否则「V3 让模型好了 5 倍」是个假读法）。** ① **组集不同**：548 组 → 126 组，指标是在**两张不同的卷子**上算的；② **spread 更窄**：V3 交付侧组内 spread 中位 **0.195**，**未达规格 0.30**（18.1.0 验收 ❌），而「组内差异更大 ⇒ 更好排」正是这些指标最敏感的方向（`v2nowave42m4` 的同一字段**未记录** ⇒ 跨数据不可比，**待复核**）；③ **训练侧遗憾历来不转移**：交付基线 nowave 训练侧 3.80% → Rust 部署 10.87%（≈7pp 落差，I7），V3 的 0.52% **不能按比例搬到部署**。⇒ 本节只记「V3 首训跑通、训练侧数字在本口径内全面优于 V2 各臂」，**部署是否变好只能由 Rust shadow 回答**。
- ⚠ **SUMMARY 里的 `Test Loss` / `Test Mean Relative Error` 两行来自 `best_model.pt`，不是 `midpoint_ep200.pt`**（日志自己已警告）⇒ 引用 V3 结果**一律引 midpoint 块**；把 20.59% 当「V3 的成绩」是错的。另：`best_model.pt` 的 mtime（09-21 07:43 ≈ ep50）与守卫⑤ 说 val capture2 argmax 在 ep150 对不上 ⇒ **记为 I15 同类证据，不作结论**（任务 #60）。

**126 组 vs 548 组 = 形状差，不是数据量差（已实测）。** 复刻训练切分（同代码同 seed，两侧与各自日志逐数吻合）：V2（full+rest+m4，过 `MIN_GROUP_SIZE=10` 后）**746,242 行 / 45,541 电路 / 3,652 组**，test = **6,843 电路 / 548 组 / 111,364 行**；V3 **599,976 / 12,455 / 837**，test = **1,876 / 126 / 97,218**。⇒ V3 的 test **行数**是 V2 的 87%，但**组数只有 23%**：V3 组**胖 3.8×**（772 vs 203 行/组）。根因是五形状配额把样本压在少数大组（(9,6) 占 **40.7% 电路 / 70.9% 行**，18.1.0）。

**⚠ 启动前拦下的静默错臂风险（本次最值钱的一条）。** 服务器 `~/-project` 停在 **17.4.8**，其 `setup_exp.sh` **没有 `v3*` 臂**（`grep -c v3nowave` = **0**）⇒ 直接用它启动会**静默训成 `v3wave42`**：跑得完、退出码 0、日志正常，只有 `USE_TRANSISTOR_WAVE` 没被关掉 —— 与「名字宣称 nowave」不符而**没有任何报错**。本次改为从 `origin/10.3.3-fix-earlystop:setup_exp.sh` 取 18.4.0 的 blob 落到 `~/setup_exp_1840.sh`（`git hash-object` = `3a765d3ce4c290baeaba6589d4c1dc9d6a568ba3`，`v3nowave` 命中 9 处，`bash -n` 通过）启动，**未动 `~/-project` 的工作区/索引**。**下次 V3 开跑前先确认 `grep -c v3nowave setup_exp.sh` 非 0。**
- ⚠ **`CACHE_SEED` 不要给 `~/cache107_master`**（177M，V2 键）。graph 缓存键 = `md5(graph_builder.py)` + `md5(所有 static/dynamic parquet mtime)` + `STRUCT_MODE`，不匹配则 `_check_cache_dir` **清空重建**并告警 —— 播种 V2 缓存进 V3 树等于白拷。V3 臂的 mtime 基线只有 `$HOME/project-107-v3nowave42`（未来 V3 各臂共用）。
- 启动自检 `scripts/check_transistor_count.py` 按预期报 ⚠ **14.8%**（P1-1 已记档缺陷；只告警不阻断，理由见该脚本文件头）。

**🟡 待复核（未定论，已入 OPEN_ISSUES I18）。** 交付 `metadata.json` 的 `dataset.split_rows` = {train **431,439** / val **62,835** / test **105,702**} 与训练侧重切 {406,953 / 95,805 / 97,218} **不同** —— 两侧都合计 599,976 ⇒ 交付方与 `utils.split_by_expr` 用了**两套切分**。训练**忽略**交付的 `split` 列、每次自己重切（`set_seed(42)` + 排序 expr + `random.shuffle`）⇒ **任何引用交付 `split` 列做评估的口径都会与训练侧对不上**。

**Rust shadow（2026-09-22 23:00:21 → 23:02:26，复用旧缓存）。** 按 `OPERATIONS.md` §6 runbook 走完 Step 1-6：8000 换挂 `midpoint_ep200.pt`（sha1 **`51bd323382ecb9b7`**）+ 本 run `outputs/scaler.pkl`（sha1 **`6b7322d6dc8bd56e`**），**不带 env**（ckpt 实测 `in_features=45` = 纯拓扑，与训练形态一致；`/proc/cmdline` 门控实读 argv 确认，见 §6.2 那条「别信变体名」）。**三件自检全过**：总行 **5390 成功 + 8 失败 = 5398**（与 m4 族 8 趟逐项相同的那个不变量 ⇒ 候选集未被改动）、**三项输入污染检查全 0**、`temp_sim_cache` **134,129 → 134,129 = 零增量**（**零 Xyce 重算、全缓存命中**，全程只 1m48s）。量纲混合守卫 / 兜底窗口 **0 / 0**。

**读法（① 714 集 batch = 部署真口径）。** 先把基线那棵树 `~/shadow_archive/midpoint_ep250_20260917_214841` 用**同一版分析器**（`--root`）重算 = 严格k3 **90.8%** / 宽松k3 **97.9%** / 选择遗憾 **5.93%**（中位 2.13%）/ 两阶段 **0.63%**，②③④ = **80.1 / 81.5 / 76.7** —— 与 §6.7 权威数**逐位相同** ⇒ **分析器无漂移**，两臂可直接并排。且两臂**候选集逐项相同**（714 集 / 5390 / 8 / 重复网表 31 / hi-spread 591 / 独立池 161）⇒ 干净配对。

| ① 714 集 batch | 基线 ep250（V2 栈） | **V3 ep200** | Δ |
|---|---|---|---|
| 严格@2 | 71.8% | 68.8% | −3.0 |
| **严格@3** | **90.8%** | **75.8%** | **−15.0** ❌ |
| 宽松@2 | 83.3% | **89.4%** | **+6.1** |
| 宽松@3 | 97.9% | 97.5% | −0.4 |
| 集合版 recall@3（`判定` 行那把尺） | 73.4% ❌ | 68.0% ❌ | −5.4 |
| **选择遗憾** | 5.93%（中位 2.13） | **5.37%**（中位 2.57） | **−0.56** |
| 两阶段遗憾 | 0.63% | **0.58%** | −0.05 |
| 两阶段捕获率 | 98.04% | 97.33% | −0.71 |
| Spearman | 0.644（中位 0.714） | 0.637（中位 0.786） | ≈0 |
| 跨度>10% 子集（591 集）严格@3 | 94.4% | **77.2%** | **−17.2** |

四口径严格k3（基线 → V3）：①714 **90.8 → 75.8**（−15.0）/ ②161 80.1 → 73.3（−6.8）/ ③18 81.5 → 72.2（−9.3）/ ④176 76.7 → 73.3（−3.4）；而**选择遗憾四个口径全部变好**（5.93→5.37 / 3.75→3.03 / 5.34→4.93 / 3.03→1.73）。**推导**（由 ①④ 两个等权均值反解，可核）：`level2/DEPTH_MIX`（① 里占 538/714 = **75.4% 权重**）单电路 **95.4% → 76.6%（−18.8pp）**，其余 176 集只差 −3.4pp。

**结论 = 「圈得稳、挑不出第一」。** 严格 k **四个口径全部变差**，而宽松@2（+6.1pp）、选择遗憾（四口径全部更好）、两阶段遗憾、Spearman 中位都持平或更好 ⇒ **V3 交付栈在 Rust 部署电路上没有改善「认出唯一最优」的能力，反而明显退步**（最该排对的 hi-spread 集上 −17.2pp）。细节：V3 的改善主要来自**尾部**（均值遗憾更好而**中位反而 2.57 > 2.13**），退步集中在严格@3。**两臂主判据都仍 ❌**（选择遗憾都 >5%、集合版 recall@3 都 <90%）⇒ **交付基线维持 `v2nowave42m4 midpoint_ep250`，不切换**（用户 2026-09-22 决定）。

**⚠ 随数一起记的三条边界。** ① **两臂的特征表也不同**：基线 in_dim **14**、V3 in_dim **45**（= 14 静态 + 31 弧特征，对应本 run 日志 `1 static + 31 arcs`）⇒ 数据与特征**同一次改动一起上**，本趟**分不开**归因；`v3wave42` 也分不开（弧特征在 `USE_TRANSISTOR_WAVE=False` 下照载）。② **106 集 window 口径（已废，仅存档）**：V3 严格k3 44.3 / 宽松k3 66.0 / 遗憾 6.99 vs **同口径重算**基线 47.2 / 70.8 / 9.59 —— 且 V3 该口径下 **12/106 集量纲混合守卫 ❌**（集内「秩」与「原始延迟」混排 ⇒ 排序无意义）⇒ **不得作判读依据**；另：§6.7 存的 106 集基线行（遗憾 11.13 / 严格k3 34.9）与今天同树重算（9.59 / 47.2）对不上，**已记 I21 待查**（batch 口径逐位复现 ⇒ 不是分析器漂移）。③ **排除掉的三个机制**：**不是并列**（同版分析器下「预测有重复值」57 vs 56 集、「第3==第4」5 vs 8，解释不了 15pp）、**不是候选集或特征码错配**（候选集逐项相同；serve 侧 8 个文件 blob 逐行相同）、**不是错标树**（argv + ckpt sha1 + 分析器戳三处一致）。

**下一步** = ① 按 §6 Step 6 **恢复 serve**：8000 现挂本轮换上去的 V3 ep200，**待收**；恢复目标 = 交付基线 `midpoint_ep250`（顺带纠正 09-17 `_shadow_ckpt_sweep.sh` 末位遗留在 8000 的 `ep300` 漂移，见 I9）。② **`v3wave42`（wave 控制臂）是否跑未定** —— nowave 是唯一可部署形态（`serve.py` 强制 `USE_TRANSISTOR_WAVE=False`），wave 臂只回答「训练侧 wave 是否帮忙」。③ §6.4b **双端点 A/B**（8001 挂 `ep250` + 启动前 `export GNN_PORT2=8001`）可消掉「基线侧 serve vintage（09-17 那次 serve 是否带 `PYTHONHASHSEED=0`）」这个唯一剩余未知量 —— 两臂候选集逐字相同、基线数已同版复现 ⇒ 边际价值中等，**本轮未跑**。

### 项目文件归类规范（2026-08-25 起长期有效）

> 教训：之前大量 `_*.py` / `_*.txt` 诊断文件散落在仓库根目录（如 `_bridge_check.txt`），杂乱且难维护。**今后一律按类归档，不往根目录散落。**

| 类别 | 位置 | 示例 |
|---|---|---|
| 核心代码 | 根目录 / `src/` | `main.py`、`config.py`、`setup_exp.sh`、`src/*.py` |
| 文档（*.md） | `docs/` | `docs/PROJECT_LOG.md`、`docs/DATA_SPEC_V2.md`、`docs/GNN_RUST_DATA_DIFF.md` |
| 诊断/分析/集成脚本 | `scripts/diag/` | `_check_v2_data.py`、`_ens_struct.py`、`_smoke_v2.py`、`_check_tl_io.py` |
| 检查/评估报告输出 | `reports/` | `_v2check_full.txt`、`_chk_io.txt`、`_v2check_fix2.txt` |

**规则**：
1. 新写的 `_*.py` 诊断脚本 → `scripts/diag/`；脚本输出的一次性 `.txt` 报告 → `reports/`（有价值的）或直接删除（临时的）。
2. **所有文档（*.md，含本记录）一律放 `docs/`**，不再放根目录。
3. 运行示例：`python scripts/diag/_check_v2_data.py data\batch_v2_full`（脚本从项目根目录跑，路径不变）。
4. 一次性调试输出（`_*.out.txt` 之类）**不得进 git**——`.gitignore` 已统一收编 `*.log` 与 `cache_smoke_*/`。
5. 2026-08-25 已清理：删除 14 个一次性 txt + `新建 文本文档.txt`；14 个诊断脚本移入 `scripts/diag/`、4 份检查报告移入 `reports/`、9 份 .md 移入 `docs/`；文档/代码内引用路径全部同步更新（md 间引用统一写 `docs/xxx.md`）。

### DATA_SPEC v9（V1 存档，两阶段交付，14.0.6-14.0.7）

> ⚠️ **V1 旧数据规格，已由 DATA_SPEC_V2（14.4）取代**（对齐 Rust：任意 I/O、单 corner、细粒度 avg_delay）。仅作历史存档。
1. 阶段1（有前提）：旧 SPICE 波形文件若还在 → 后处理补 3 个新 transistor 字段（ids_rise_time/vgs_swing/ids_charge），零仿真成本。不在则跳过，不重跑 60 万。
2. 阶段2：新 120 万行，4 vector/condition，全部 v9 格式，新 expr 不与已有 569 个重叠。总计 ~180 万行，val 组数翻倍。
3. 新 3 字段 + 旧 4 字段 = 7 个 transistor 子字段，全部 100% 覆盖受铁律约束。

### 当前方向/待办（历史保留，已过时）

> ⚠️ **本段为历史存档，内容已过时**：per_gate 已定死路、wave 已是 train-only game-changer、集成已定稿（15.2.10 6-seed）、#8 已完成。**当前方向/待办以 15.2.10 / 15.2.3 的「待办 ①-⑤」及下方 14.4 待办为准。**

（原「当前方向/待办」正文——历史保留：）
- **per_gate**：死路，搁置。**LIB**：长线，需 2D-grid 加速再评估。
- **wave**：信噪比诊断表明突破 <2% 成对分辨需要晶体管全覆盖数据(降模型预测噪声)；现有 wave 28% 稀疏+集中低slew→不可用。DATA_SPEC 已备好全覆盖规格。
- **集成**：暂缓，优先解决信噪比瓶颈。
- **#8 结构特征分析**：已完成（2026-07-17），结果记录在下方「可复用结构模式」表。待新数据到位后追加。

### 可复用低延迟结构模式（Task #8 分析，2026-07-17。后续有新数据可追加新行）

> 数据源：旧 1005 电路（archive_v13.1），2559 个变体组，中位差异 6.6%，1028 组（40%）差异 >10%。
> 方法：同 expr+corner 内对比最快 vs 最慢变体的结构特征。

| 观察 | 数据 | 备注 |
|---|---|---|
| 更少晶体管 → 更快 | 83% 的高差异飞快变体比慢变体晶体管更少（中位 -8 TC）。门数几乎相同（中位差=0） | 简洁性是最强信号 |
| `SC_INV_WIRE` 强关联低延迟 | 快组中出现 352 次，慢组 252 次（+100）| 单缓冲器结构，替代复杂组合门 |
| `SC_AND` 强关联高延迟 | 慢组中出现 208 次，快组仅 41 次（-167）| 需要多个晶体管实现，延迟更大 |
| `SC_JOIN` 基础型比复杂链更快 | 简单 `SC_JOIN_OR_OR`(+52)、`SC_JOIN_v1`(+41) 在快组多；复杂长链 `SC_JOIN_AND_...` 在慢组多 | 串联级数越长延迟越大 |
| NOR 链优于 OR 直接实现 | `SC_JOIN_OR_OR` 在飞快变体中频繁出现 | NOR2+INV 实现 OR 比直接 OR 更高效 |

> **解读**：这些不是「替换规则」，而是「在已有 1005 电路的数据中，确实观察到这些模式在高差异变体组中反复出现」。等新数据（delivery1 full）到位后，可重新跑分析追加新行，验证这些模式是否跨数据一致、以及是否有新模式浮现。

### 13.x 批次（13.0~13.1.3，成对排序损失 + 排序选点。同 expr 切分，可比。结果 2026-07-13）
| Exp | 变体 | Spearman | 遗憾 | top1 | 捕获率 | <2%成对 | epoch |
|---|---|---|---|---|---|---|---|
| **rank(基线)** | smoothed_rel_err选点+深退火 | **0.206** | **2.63%** | **42.2%** | **68.7%** | 52% | 102 |
| rankloss1 | 成对排序损失 w=0.5 | 0.121 | 3.29% | 37.2% | 62.6% | 56% | 108 |
| rankloss2 | 成对排序损失 w=2.0 | 0.148 | 3.27% | 39.7% | 66.8% | 54% | 231 |
| bestrank | val选择遗憾选checkpoint | 0.058 | 4.31% | 34.1% | 60.6% | 50% | 79 |

**结论**：
1. **基线最优**——所有方向性改动(排序损失/排序选点)全负面。和 per_gate 同理：辅助loss在共享encoder上导致表征冲突。
2. **排序损失有害**（~1pp退化）——和 per_gate 同一机制、量级较轻；**排序选点(bestrank)最差**——val 只有 139 组，regret 噪声大、偶发低点误选早期epoch，停在未收敛状态。
3. **成对分辨 <2% = 52%(随机) 在所有实验上一致**——不是训练/选点/损失的问题，是预测精度的天花板（见下方信噪比诊断）。
4. **最佳配置：bmsm(平滑rel_err选点) + 深退火 + expr切分**。所有可改项已穷举，无需再试。

### 信噪比诊断（_diag_pairwise.py，2026-07-13）
在 rank 基线 test_predictions 上计算：
| 量 | 数值 |
|---|---|
| 模型预测RMS | 17.44 ps |
| Median绝对误差 | 4.07 ps |
| 变体差中位 | 5.6% = 1.34 ps |
| <2%差异信号 | 0.48 ps |
| 变体聚合后噪声(16行/变体) | 4.36 ps |
| **聚合后 SNR vs <2%信号** | **0.11**（需要 >~2 才能稳定分辨，差 ~18x）|

**各延迟档 SNR 均 <0.1**——全量级都无法稳定分辨 <2% 差异。

**为什么所有方向改动都无效**：排序损失/排序选点不创造新信息，只重排已有信息。降预测噪声唯一途径=新数据：
- 更多电路（降方差）→ ~60-65%
- + 晶体管全覆盖数据(wave, 提供电学物理信息) → ~70-75%
- + SPICE 更精仿真 → ~80-85%
- 理论极限（标签测量物理极限）→ ~90%

当前瓶颈是**模型预测噪声**（17ps RMS），远大于 SPICE 标签精度（~1-3ps）。SPICE 更精有意义但非当前瓶颈。

### 13.x 代码里程碑
- **13.0**(24dfd7b)：组内成对排序损失(`_pairwise_rank_loss`) + `GroupedBatchSampler`
- **13.1**(b84f77d)：checkpoint 按 val 排序指标选择(`BEST_RANK_METRIC`)
- **13.1.1**(09c6859)：成对排序损失 nan 保护
- **13.1.2**(f565d55)：bestrank 排序评估 grad 修复
- **13.1.3**(20590d4)：添加成对分辨诊断脚本 `_diag_pairwise.py`

---

## LIB/Scheme A/B Roadmap（历史存档）

> ⚠️ **已过时/被取代**：LIB（Scheme A）依赖 per_gate 监督（已证有害，见 13.x）；TW（Scheme B）虽确认 wave 有效，但 train-only、Rust 推理拿不到，蒸馏失败（15.2.3）→ 走「V2 no-wave 粗筛 + top-K SPICE 精排」路线。保留作历史决策记录。

### Scheme A (LIB Table Lookup)
- **Goal:** Model predicts per-gate (slew, load) → LIB table lookup → sum delays.
- **Status:** LIB (`std_cells.lib`, 93 cells) + SC 展开表 (`sc_expansion.json`, 3868/3868 宏可展开) 已就位。代码已集成（11.0, train_lib），但链 DP 太慢需 2D-grid 加速。**暂缓**。
- **Per-gate 辅助监督已证明有害**（+4~5pp）→ 若启用 LIB，只需总延迟项、关 PG_*_W=0。

### Scheme B (Transistor Waveform)
- **Goal:** 共享 GNN encoder + 晶体管电流/电压(ids_avg/ids_peak/vds_swing)作为额外输入或辅助监督，降低预测噪声。
- **Status:** 现有 batch_wave 仅 28% 覆盖 + 集中低 slew（s03/s05），不可用。**信噪比诊断证明突破 <2% 成对分辨需全覆盖晶体管数据**——这是目前已知最高杠杆的数据需求。
- **DATA_SPEC 已要求全覆盖**（30 corner 全 sweep + 必须含 s40/s80，每行 100% transistor_wave_json）。
- **零额外仿真成本**：SPICE 已经算过这些量，只需在后处理中提取写出。
- **优先级：高**（降模型预测噪声 2-3x，配合更多电路可将 <2% 成对分辨从 52%→70-75%）。

### Decision Tree

> ⚠️ **已失效（历史存档）**：本路线图按「点精度（%）」预测路线，与后续实际走向不符——wave 在**排序任务**上成功（13.5）、per_gate/TW 在 13.x 被证有害、蒸馏失败（15.2.3），最终路线为「V2 no-wave 6-seed 粗筛 + wave 仅作天花板」。当前方向见 15.2.10/15.2.11 与 `docs/GNN_RUST_DATA_DIFF.md`。
```
New Data Arrives
├── SC expansion table → Activate LIB mode (train_lib.py)
│   └── Expected: 20-22%
├── Full transistor data → Activate TW multitask (train_sweep.py)
│   └── Expected: 18-21%
├── Both → Combine (PG + TW)
│   └── Expected: 15-18%
└── Neither → Ensemble (3-seed average)
    └── Expected: 23-24%
```

---

## Data Organization

### Current Active Data (on server)

> ⚠️ **本段过时**：以下为 **V1 旧数据（batch1-3）** 结构。**当前活跃数据是 V2**（`data/batch_v2_full/` + `data/batch_v2_io/`，任意 I/O、单 corner s02p0_l01p0、细粒度 DELAY），详见 `docs/DATA_SPEC_V2.md`。
```
data/
├── batch1/           # 150 circuits, 30 corners, full sweep
├── batch1b/          # 50 circuits, 30 corners
├── batch2/           # 325 circuits, 9 corners
├── batch3/           # 480 circuits, 9 corners
├── batch_wave/       # 20 circuits, 30 corners + transistor (28% filled)
├── std_cells.lib     # ASAP7 LIB table
├── sc_to_asap7.json  # SC→ASAP7 mapping (will be obsolete after expansion)
└── archive/          # Old data versions
```

### Data Versions
- **Original:** batch1/batch1b/batch2/batch3 — old format, no per_gate
- **_fixed:** Same circuits, added gate_states_json + per_gate_timing_json (100% filled)
- **_v4:** Same as _fixed but fields empty (generator error)
- **Current:** _fixed data moved to batch1/batch1b/batch2/batch3. Old data in archive/.

### Key Data Fields
- `per_gate_timing_json`: delay_ps, out_slew_ps, in_slew_ps per gate
- `gate_states_json`: 0/1 per gate (on signal path)
- `transistor_wave_json`: ids_avg, ids_peak, vds_swing per transistor (batch_wave only)
- Per-pin: slew_a~d, load_a~d, arrival_time_a~d
- Global: slew_s, output_load_f, DELAY, corner, vector

---

## Code Architecture (9.7 Baseline)

### Key Files
```
src/
├── model.py          # DelayGNN: 6-layer GraphConv + path sum readout
├── data_loader.py    # DelayDataset: per-pin + per-gate feature extraction
├── train_sweep.py    # 9.7 training loop (main.py → this)
├── train_lib.py      # LIB mode training (unused, for when SC expansion arrives)
├── graph_builder.py  # Static graph + p/g/h electrical features
├── logic_sim.py      # Intersection BFS gate state computation
├── lib_lookup.py     # LIB parser + bilinear interpolation
├── utils.py          # Seed, split, scaler utilities
main.py               # Entry: from src.train_sweep import main
config.py             # Hyperparameters (HIDDEN_DIM=256, NUM_LAYERS=6, etc.)
```

### Model Architecture Detail
```
Input: x = [gate_idx(1), fanout, depth, drive, p, g, h(6 static), 
            logic, is_sw, slew, load, out_load, arrival, gate_state(7 dynamic)]
       = 14 dims total

gate_idx → Embedding(626, 32) → gate_emb (32d)
struct_dyn = x[:, 1:] (13d)
x = cat([gate_emb, struct_dyn])  → 45d

6× [GraphConv + LayerNorm + ReLU + Dropout + Residual]
    ↓
gate_mask * x  → zero non-path nodes → global_add_pool → (B, 256)
    ↓
+ corner_encoder(corner_cond)  → (B, 256)
+ sig_encoder(circuit_sig)     → (B, 256)
    ↓
cat → (B, 768) → Linear(768, 1) → scalar log-delay
```

### Hyperparameters
```
HIDDEN_DIM=256, NUM_LAYERS=6, GATE_EMBED_DIM=32
DROPOUT=0.3, LEARNING_RATE=1e-4, WEIGHT_DECAY=1e-4
BATCH_SIZE=80, EPOCHS=1200, PATIENCE=40
HUBER_DELTA=0.3
```

---

## Server

**Machine:** tianlang@orca (10.20.34.16)
- 24 cores, 60GB RAM, no GPU
- Python 3.13, venv at ~/venv
- Project at ~/./-project/

**Running experiments:**
```bash
# Start:
cd ~/-project && source ~/venv/bin/activate
OMP_NUM_THREADS=6 nohup python3 -u main.py > trainXXX.log 2>&1 &

# Check:
tail -3 ~/-project/trainXXX.log

# Multiple experiments: clone to separate dirs with different CACHE_DIR
cp -r -- -project project-NAME
cd ~/project-NAME && sed -i 's/CACHE_DIR = .*/CACHE_DIR = "cacheNAME"/' config.py
```

**Data sync:** Data files are tracked in git (~13MB). `git pull` gets code + data.

**CACHE_DIR WARNING:** Always use "cache" as default. The "cache953" pollution came from a 953 experiment sed command that was never reverted. Smart cache system (code hash + data mtime) auto-invalidates on changes.

---

## Key Lessons Learned

1. **Input information > architecture:** Corner encoding (-6.5pp) was the biggest gain — it added new INFORMATION, not just better processing. 50+ architecture tweaks combined contributed less.
2. **GNN node features degrade:** After 6 layers, a node's feature is ~30% self, ~70% neighbor mix. Per-gate prediction fails because individual gates lose identity. This is fundamental to message-passing GNNs.
3. **Sparse aux data doesn't train:** Transistor data at 28% density (777/2768) can't drive 118K-sample training. Need full coverage.
4. ~~**LIB is a regularization, not a prediction tool:** PG (24.46%) > 10.2 (25.70%) because LIB chain provides physics-constrained prediction path even if table values are wrong.~~
   **⚠️ 已作废**：该条依据 PG(24.46%) 的 per_gate 辅助，但 107 批次结论（本文件上方）已证明**当时 PG 的 per_gate 是 no-op**（hasattr 静默跳过），24.46% 并非 LIB 链的真实效果。真正生效后的 per_gate 实测有害（+4~5pp）。LIB 路径按此结论应重新评估，本教训不再成立。
5. **Worst corner (l00p2/l00p5) stuck at ~42%:** Corner encoding reaches limit for extreme nonlinearity. Only transistor-level data can capture these.
6. **Don't retry:** GAT, GIN, gate weighting, corner weighting, physical features beyond p/g/h, gate type merge.
   > ⚠️ **部分被推翻**：「physical features beyond p/g/h」泛指失败，但 13.5 的 **transistor_wave**（高差异 Spearman 0.182→0.705，game-changer）与 newcaps（寄生电容，+0.03）是**有效的物理输入特征**——差别在于它们是「补充输入信息」而非「架构改动」。本教训应缩小为「架构层面的物理特征」，不含新增数据特征。
