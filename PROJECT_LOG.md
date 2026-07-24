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
