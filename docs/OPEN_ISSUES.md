# 未决问题 / 待复核 / 风险清单（OPEN ISSUES）

> 集中散落在各文档的「⚠ / 待复核 / 未决」项。**接手新任务或下结论前先扫本表**；状态更新时同步改这里。
> 记录风格遵循版本记录式（REQUIREMENTS §4）。最后更新：2026-09-16（17.3.12）。

| ID | 状态 | 内容 | 位置 / 影响 |
|---|---|---|---|
| I1 | ✅ 已销案（2026-09-16） | **§13 v2iaa42m4 Rust 的 recall「不可复现」= 口径差，不是数据错——且原判反了**：**714 集（`circuit, iter, window_try`）= 一次 `pre_rank` 调用 = 部署真值**，106 集 = `(circuit, window_try)` **池化**口径（历史）。同 CSV 只换键：严格@3 30.2% → 56.7%（+26.5pp），遗憾对口径不敏感（+0.7pp）。⚠ **连带两条**：(a) 口径效应 model-dependent（nowave **+55.9pp**；⚠ 2026-09-17 改，旧记 +39.5pp 用了错标树的 74.4，见 DIFF §15.8）→ **旧 106 集口径的跨臂结论不可外推**（臂内可比性仍在）；(b) **历史表的 ckpt epoch 标注至少有一处错**（同 arm 同标称 ckpt 严格@3 39.6 vs 34.9 = 4.7pp）→ 跨 ckpt 排序冻结待 batch 口径重测。**结论基线 = v2iag42m4 这个裁决产自旧口径，其臂间排序需按 batch 口径重验** | DIFF §15 / §13 / §14⑥ |
| I2 | ✅ 已定案 | DEPTH_MIX「纯深度 OOD」撤销（候选 X_ 链 0%>6、spread≈全体、训练 (5,1) spread 361% 覆盖足）；偏高 2.8pp 遗憾原因未深挖 | DIFF §12.2d；V3 spec (5,1) 已按 spread 大处理 |
| I3 | ⚠ 待核 | §12.3「temp_sim_test 可恢复 dut = 1,253」vs 实测 `dut_expr_*.sp` = **1,135**（差 118） | DIFF §12.3/§14 |
| I4 | 🔶 未决 | V3 数据 `sc_expansion.json` 格式：宏集预计数百-数千种（名称编码结构可自动推导），具体格式待与训练端 STRUCT_MODE 用法对齐 | DATA_SPEC_V2 P15 注 / R4 |
| I5 | 🔶 未决 | **V3 生成方能力**：e-graph 能否产出 ★ 档合格样本（(9,6) 350 个 trans≥247/深≥9 等）——Rust 数据无法回答；靠升级预检（5-8 电路含 ★ 档）+ ★ 弹性条款兜底 | DATA_SPEC_V2 锚定小节 / R6 |
| I6 | ⚠ 待核 | vector 语义已确认（per-(output,pin) break、rise/fall 同 vector 双激励）；但训练数据行数 = 全量 2×N_in×M（m4 实测 108），spec 已改「≤」（敏感对）；生成方实际是否跳过不敏感对待 V3 交付验证 | DATA_SPEC_V2 R5/C1；DIFF §12.2c/12.2e |
| I7 | 🔶 部分进行中 | **m4 系 Rust shadow 全跑完并定论**（iaa 9-2 记录⚠ / iag 19:09 / nowave 19:29 / **kd 22:03**，均同 106 集/5390 行 pipeline；记录 PROJECT_LOG + DIFF §13.2-13.4）：**选择遗憾 nowave 10.87% < iag 12.19% < kd 14.97% < iaa 15.21%——无近似纯拓扑 serve 最优，近似特征 serve 净伤害；KD 学生亦不赢朴素 nowave（train + Rust 双端无增益，蒸馏不转移）**。⚠ **KD 学生真相（16.11.40 修正）：名带 iaa、setup 设 USE_IDS_AVG_APPROX=1，但 KD 分支 `USE_TRANSISTOR_WAVE=False` 把近似列门控挡掉（近似列代码在 wave 块内 data_loader L507/L510）→ 学生训时即纯拓扑 in=45，「iaa 学生 + wave 教师 KD」这格从未真正测过；本次实跑 = 纯拓扑学生 + 软标签，train 对照 = nowave（3.80）非 iaa**。**仍全不达 10.3 主判据**（遗憾 >5%、严格@3 <90%）→ GNN = 启发式预排序，规模失配是硬底（nowave 训练 3.80→Rust 10.87 ≈7pp）。**交付含义：Rust 粗筛走纯拓扑 nowave 路线，USE_IDS_AVG_APPROX 对 serve 负贡献**（§13.4 收口）；serve 现仍挂 kd（恢复 nowave 命令见 OPERATIONS §6 Step 6）。剩 **v2iaar42m4（排序 loss）训练中**，训完照 OPERATIONS §6 runbook 验 Rust | OPERATIONS §5/§6；DIFF §13.2-13.4 |
| I8 | 🔶 未决 | **60 万行 V3 新数据尚未生成**：DATA_SPEC_V2（V3 单一数据集 + 形状-规模-深度锚定 + Tier A/B）待交付生成方；泛化闸门（旧数据固定 test 集 ≥ v2wave42m4 基线）执行时定 | DATA_SPEC_V2 |
| I9 | 🔶 低优先 | serve 8000 端口现挂 **v2nowave42m4 midpoint_ep250**（无近似，in_dim=14；**ckpt sha1 = 09be9a3c53b90760**，17.3.11 `run_stamp` 实读）；占 ~1.1GB 内存，下次换模型时一并停旧起新。其**部署口径（714 集 batch）**读数 = 严格@3 **90.8%** / 宽松@3 **97.9%** / 选择遗憾 **5.93%**（中位 2.13%） / 两阶段 **0.63%**（**与历史 106 集口径的 39.6/65.1/10.87 不是同一把尺**）。⚠ **2026-09-17 改**：旧记 74.4 / 97.6 / 7.23 / 1.66 来自**错标树**（非本 ckpt 的数），见 DIFF §15.8 | OPERATIONS §5；DIFF §15.2/§15.5 |
| I15 | 🔶 待核 | **历史 shadow 表的 ckpt epoch 标注至少有一处错**（v2nowave42m4：§13.3/PROJECT_LOG§747 记 mid200 严格@3 39.6%、首表把同一读数标 ep250、十点扫描 ep250 读 34.9%）→ 两种可能未分离：(i) 标注错；(ii) 17.2.4 定序修复对 recall@3 的影响远大于遗憾（±0.31pp 带只由遗憾标定）。**影响：跨 ckpt 的选点结论（含 42b ep100→ep150、残余 1.29pp 定性）全部挂起** | DIFF §15.5；PROJECT_LOG 16.x 十七点 |
| I16 | 🔶 规划 | `_shadow_analyze.py` **③ 宏平均混 n 不同两类集**（n=4 严格@3 随机基线 75% vs n=8 的 30%）→ 需 n≥8 分层版；`判定` 行应标「仅 ① 口径」。**代码未改，待批**（17.3.12 候选） | DIFF §15.6 |
| I10 | ✅ 已解决 | 弱驱动中间门验收判据已定（★ 档含弱驱动门电路 ≥30%；扇出≤2 或寄生≥2×中位）——数值是否需按 serve 候选校准，待 V3 实际数据验证 | DATA_SPEC_V2 P12/L306 |
| I11 | 🔶 低优先 | 方案 A（LIB/sc_expansion→ASAP7 标准单元）前提与 Rust 宏（晶体管级复合）语义是否一致——方案 A 未启用，不影响交付 | DATA_SPEC_V2 §六 P15 |
| I12 | 🔶 未决 | 粗仿真死因链已补 **warm-start 一条（16.11.37）**：M1 `.nodeset` 完美自暖 ≈ 0%（15 轮交错钉死）→ 生产近孪生 ≤0；M2 `.ic+uic` 跳 DCOP 仅省 ~9% 且需「自己精确 DCOP」当 warm（生产不可行）+ 延迟误差 0.41% > reltol 0.39% 基准 → **方向关闭**。粗仿真死因主线（自适应步长不省时间 + 放宽误差失真/崩）见 WAVE_ABLATION 16.9.5/16.10.2 | git 16.11.0 / _corr_idsavg.py / WAVE_ABLATION 16.11.37 |
| I13 | 🔶 规划 | 6-seed 集成（V2+m4 系现全只有 seed42）：v2iag42m4 Rust 结果后决定铺哪些 seed | DIFF §12.5 |
| I14 | ✅ 已做 | push：16.11.18-32 已全部推送 GitHub（2026-09-03）；NetlistOpt 仅本地（按规则） | — |

> 原则（REQUIREMENTS §5，2026-09-03）：数据/分析结论先完整验证再下结论；证据不足标「待复核」不下定论；数据相关问题以 DIFF 为参考（§14 审计标注为准）。
