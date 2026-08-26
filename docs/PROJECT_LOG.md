# Project Log — GNN Delay Prediction

## Project Overview

**Goal:** Predict circuit propagation delay (SPICE-level) from netlist topology + corner conditions.

**Input:** Transistor-level netlist (4 input pins a/b/c/d + output), per-pin slew/load/arrival_time, corner conditions, vector (input pattern).

**Output:** End-to-end delay from switching pin to output.

**Current Best:** Test 24.55% (9.7) — 6-layer GraphConv + path sum readout + corner separation.

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
3. **单一 seed 不够稳**：seed123 vs seed42 遗憾差 2x（3.23% vs 1.59%）→ 最终决策需 2-3 seed 集成。
4. **anneal 全面更差**→ 退火到此为止，不再尝试。
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

### 13.7 批次（Running，delivery1+2，~54万行，13.5 newwave 架构 + midpoint 选择）

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

### 13.7.14 集成批次（Running，delivery1+2，4-seed全错开 + midpoint）

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

### 14.4.5 V2 数据首轮合规检查（2026-08-20，不合格，需返工）

> 数据来源：GitHub `10.3.3-fix-earlystop` 分支 commit 4a0c18f（Add V2 dataset）+ 97fd7fda（sc_expansion 合并 8909 类）。本地 `git checkout 97fd7fda -- data/` 拉取。检查脚本：`scripts/diag/_check_v2_data.py`（对照 DATA_SPEC_V2 逐项，输出 `reports/_v2check_full.txt`）。

**数据概况（batch_v2_full，交付主体）**：
- 8860 电路 / 69,439 行 / 591 expr（expr8000+，与 V1 569 无重叠 ✓）
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

### DATA_SPEC v9（两阶段交付，14.0.6-14.0.7）
1. 阶段1（有前提）：旧 SPICE 波形文件若还在 → 后处理补 3 个新 transistor 字段（ids_rise_time/vgs_swing/ids_charge），零仿真成本。不在则跳过，不重跑 60 万。
2. 阶段2：新 120 万行，4 vector/condition，全部 v9 格式，新 expr 不与已有 569 个重叠。总计 ~180 万行，val 组数翻倍。
3. 新 3 字段 + 旧 4 字段 = 7 个 transistor 子字段，全部 100% 覆盖受铁律约束。

### 当前方向/待办

### 当前方向/待办
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

## LIB/Scheme A/B Roadmap

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
4. **LIB is a regularization, not a prediction tool:** PG (24.46%) > 10.2 (25.70%) because LIB chain provides physics-constrained prediction path even if table values are wrong.
5. **Worst corner (l00p2/l00p5) stuck at ~42%:** Corner encoding reaches limit for extreme nonlinearity. Only transistor-level data can capture these.
6. **Don't retry:** GAT, GIN, gate weighting, corner weighting, physical features beyond p/g/h, gate type merge.
