# 未决问题 / 待复核 / 风险清单（OPEN ISSUES）

> 集中散落在各文档的「⚠ / 待复核 / 未决」项。**接手新任务或下结论前先扫本表**；状态更新时同步改这里。
> 记录风格遵循版本记录式（REQUIREMENTS §4）。最后更新：2026-09-03（16.11.34）。

| ID | 状态 | 内容 | 位置 / 影响 |
|---|---|---|---|
| I1 | ✅ 基线已切换 | **§13 v2iaa42m4 Rust 表内 recall/Spearman 仍存档不可独立复现**（复算 714 集口径系统性 ≠ pipeline）；**但 106 集口径已澄清** = 标准 shadow pipeline 窗口级 ≥4 候选计数——v2iag42m4 新 run（§13.2）复现同样 106 集/5390 行/8 失败，**跨 run 稳定**（714 = 复算另类聚合）。**结论基线 = v2iag42m4（GBDT15，同 pipeline 实测，I7 已裁决）**，取代 v2iaa ⚠ 数字作量化依据 | DIFF §13/13.2/§14 |
| I2 | ✅ 已定案 | DEPTH_MIX「纯深度 OOD」撤销（候选 X_ 链 0%>6、spread≈全体、训练 (5,1) spread 361% 覆盖足）；偏高 2.8pp 遗憾原因未深挖 | DIFF §12.2d；V3 spec (5,1) 已按 spread 大处理 |
| I3 | ⚠ 待核 | §12.3「temp_sim_test 可恢复 dut = 1,253」vs 实测 `dut_expr_*.sp` = **1,135**（差 118） | DIFF §12.3/§14 |
| I4 | 🔶 未决 | V3 数据 `sc_expansion.json` 格式：宏集预计数百-数千种（名称编码结构可自动推导），具体格式待与训练端 STRUCT_MODE 用法对齐 | DATA_SPEC_V2 P15 注 / R4 |
| I5 | 🔶 未决 | **V3 生成方能力**：e-graph 能否产出 ★ 档合格样本（(9,6) 350 个 trans≥247/深≥9 等）——Rust 数据无法回答；靠升级预检（5-8 电路含 ★ 档）+ ★ 弹性条款兜底 | DATA_SPEC_V2 锚定小节 / R6 |
| I6 | ⚠ 待核 | vector 语义已确认（per-(output,pin) break、rise/fall 同 vector 双激励）；但训练数据行数 = 全量 2×N_in×M（m4 实测 108），spec 已改「≤」（敏感对）；生成方实际是否跳过不敏感对待 V3 交付验证 | DATA_SPEC_V2 R5/C1；DIFF §12.2c/12.2e |
| I7 | 🔶 部分进行中 | **v2iag42m4 Rust shadow 已完成并裁决**（2026-09-03 19:09，106 集/5390 行，GBDT15 加载确认；记录 PROJECT_LOG + DIFF §13.2）：vs v2iaa42m4 记录**全线更优**（遗憾 12.19 vs 15.21 / 严格@3 44.3 vs 30.2 / 两阶段 3.93 vs 5.12 / Sp 0.210 vs 0.050，同候选窗口 A/B）→ **特征一致性增益来自 serve 端鲁棒性**；但**仍不达 10.3 主判据**（遗憾 >5%、严格@3 <90%）→ GNN = 启发式预排序，剩余落差 = 规模失配为主（§13.1 A1 不可行坐实）；两阶段 3.93% ≤5% 达标（前 3→SPICE 粗筛实操可用）。**剩 v2iaar42m4（排序 loss）/ v2kdwave42iaa42（iaa 蒸馏）训练中**——训练完 → 记 PROJECT_LOG → 各自 Rust shadow | OPERATIONS §5；DIFF §13.2 |
| I8 | 🔶 未决 | **60 万行 V3 新数据尚未生成**：DATA_SPEC_V2（V3 单一数据集 + 形状-规模-深度锚定 + Tier A/B）待交付生成方；泛化闸门（旧数据固定 test 集 ≥ v2wave42m4 基线）执行时定 | DATA_SPEC_V2 |
| I9 | 🔶 低优先 | serve 8000 端口现挂 **v2iag42m4 mid200**（`USE_IDS_AVG_APPROX=2`，GBDT15 已加载，刚跑完 v2iag shadow = 当前参考模型，暂无需换）；占 ~1.1GB 内存，下次换模型时一并停旧起新 | OPERATIONS §5 |
| I10 | ✅ 已解决 | 弱驱动中间门验收判据已定（★ 档含弱驱动门电路 ≥30%；扇出≤2 或寄生≥2×中位）——数值是否需按 serve 候选校准，待 V3 实际数据验证 | DATA_SPEC_V2 P12/L306 |
| I11 | 🔶 低优先 | 方案 A（LIB/sc_expansion→ASAP7 标准单元）前提与 Rust 宏（晶体管级复合）语义是否一致——方案 A 未启用，不影响交付 | DATA_SPEC_V2 §六 P15 |
| I12 | 🔶 未决 | 粗仿真（直接测 ids_avg）死因已记：Xyce 自适应步长不省时间 + 调整步长/放宽误差后 ids_avg 失真或收敛崩（细节未留独立记录，机制推断）——如需补充实验记录 | git 16.11.0 / _corr_idsavg.py |
| I13 | 🔶 规划 | 6-seed 集成（V2+m4 系现全只有 seed42）：v2iag42m4 Rust 结果后决定铺哪些 seed | DIFF §12.5 |
| I14 | ✅ 已做 | push：16.11.18-32 已全部推送 GitHub（2026-09-03）；NetlistOpt 仅本地（按规则） | — |

> 原则（REQUIREMENTS §5，2026-09-03）：数据/分析结论先完整验证再下结论；证据不足标「待复核」不下定论；数据相关问题以 DIFF 为参考（§14 审计标注为准）。
