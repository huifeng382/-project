# DATA_SPEC_V2 数据合规问题清单（首轮检查，2026-08-20）

> **结论：不合格，需返工。** 检查对象为 GitHub `10.3.3-fix-earlystop` @ `97fd7fda` 交付的 V2 数据
> （`data/batch_v2` + `data/batch_v2_full`）。检查方法：`_check_v2_data.py` 逐项对照
> `DATA_SPEC_V2.md`（定稿）自动校验，明细见 `_v2check_full.txt`。
>
> **范围说明**：生成**数量**问题（规格 ~5 万电路 / 60 万行 / 4000 expr，实际 8860 / 69,439 / 591，
> 约 12%）按 2026-08-20 讨论**暂缓**，不在本清单展开；本清单只列值级与结构性问题。

---

## 一、数据概况（batch_v2_full，交付主体）

| 项 | 实际 | 规格要求 | 判定 |
|---|---|---|---|
| 电路 / 行 / expr | 8,860 / 69,439 / 591（expr8000+） | 见「范围说明」 | ⏸ 暂缓 |
| I/O 形状 | 全部 1~4 入 / 1 出（4 入占 98%） | 任意 N(1~16)/M(1~6) + 分桶 + 多输出 ≥20% | ❌ P9 |
| corner | 单 `s02p0_l01p0` ✓ | 单 corner（2ps/1fF） | ✅ |
| slew_s / output_load_f | 2e-12 / 1e-15（恒定）✓ | 2ps / 1fF | ✅ |
| 组大小（变体/expr） | 10~15，中位 15，100% 合规 ✓ | 10~15 | ✅ |
| 每电路行数 | 8 = 2×N_in×M（1 个电路 7 行） | 2×N_in×M | ⚠️ P6 |
| vector | 每 (circuit,pin,dir) 恰 1 个 ✓ | 1（对齐 Rust break） | ✅（但值错，见 P3） |
| 重复行 | 0 ✓ | 无 | ✅ |
| DELAY | 1.6ps ~ 174ps，在 (1e-12, 1e-8) ✓ | 同上 | ✅ |
| expr 与 V1 | 不重叠（expr8000+）✓ | 不与 V1 569 重叠 | ✅ |
| 必需列 | 静态 10 列 / 动态 15 列齐全 ✓ | 见规格 | ✅ |
| coverage_report_v2.json | 存在且声称 100% | 随数据交付 | ✅（但值级错误测不出，见各 P） |
| transistor_wave / gate_states / supply_noise / parasitic_caps | 列非空 100% ✓ | 完整性铁律 | ✅（值级问题见 P1/P5/P7） |

- **batch_v2（290 电路）= batch_v2_full 的子集且本身损坏**（`input_pins_json` 全空、网表 `.SUBCKT DUT` 无输入引脚）→ **该批废弃**，用 full 即可（P8）。

---

## 二、问题清单（按严重度排序，每条含「问题所在 → Rust 集成影响 → 修复要求」）

### P1 🔴 大小写不一致：JSON key `x_*` vs 网表 `X_*`（最致命）

- **现象/证据**：`gate_states_json` / `parasitic_caps_json` 的 key 与 `transistor_wave_json` 的 `gate` 子字段
  全部为小写 `x_1`，而 `gate_level_netlist` 中实例名为大写 `X_1`。抽样 3000 电路 / 2000 行，**全部不一致**。
- **问题所在**：生成侧 JSON 序列化时把实例名小写化，与网表生成路径（大写 `X_N`）未对齐。**14.4 历史同款 bug 复现**
  （当时 `node_names` 大写 vs JSON key 小写 → per_gate 从未喂入训练、gate_states 匹配 0 门 → path-sum readout 只累加 out 节点）。
- **Rust 集成影响**：`graph_builder.parse_netlist` 按网表名（`X_1`）查 JSON key（`x_1`）→ **全部 miss**：
  gate_states 特征恒为默认值、parasitic_caps 全丢、transistor_wave 的「晶体管→门」聚合键指向不存在的门。
  Rust 侧 `expr_to_hierarchical_spice` 输出就是大写 `X_1`，serve.py 按 Rust 网表名构造特征——
  **训练数据 key 小写、推理时大写，特征名空间系统性错位，模型静默退化成无路径/无结构信息版**。
- **修复要求**：JSON 中所有实例名 key（gate_states / parasitic_caps / transistor_wave.gate / pin_loads 等）与网表
  `X_*` 大小写完全一致。

### P2 🔴 direction 带 `.sp` 后缀（`rise.sp` / `fall.sp`）

- **现象/证据**：`timing_arcs.direction` 全部行为 `rise.sp` / `fall.sp`（两批均如此）。
- **问题所在**：生成脚本把 SPICE 波形文件名（`rise.sp`/`fall.sp`）当 direction 值写入，未剥离后缀。
- **Rust 集成影响**：Rust 侧 `SimuVector.timing_sense` 是纯 `rise`/`fall`。GNN 训练学到的是 `rise.sp` 脏值；
  serve.py 推理时从 Rust 拿到真 `rise` → **推理值在训练分布之外（OOV）**，direction 特征编码失效。
  且 V2 规格明确 `direction ∈ {rise, fall}`，9.2 data_loader 按 direction 做特征与 avg 聚合，脏值污染口径。
- **修复要求**：`direction` 值去掉 `.sp` 后缀（一行 `replace('.sp','')` 级别）。

### P3 🔴 vector 切换位恒为 1（rise 行应为 0）

- **现象/证据**：抽样 2 万行，切换引脚对应位 **100% 为 1**（含 rise 行）。按规格「rise→0, fall→1」校验仅一半行通过。
- **问题所在**：vector 编码方向反了——写入的是翻转后稳态电平（或取错了 truth_table 行），未按 V2 规格
  §三「switching_pin 对应位与 direction 一致」生成。因所有行错得一致，『每 (circuit,pin,dir) 恰 1 vector』检查通过，掩盖了问题。
- **Rust 集成影响**：Rust vector 语义 = **初始逻辑电平**（`SimuVector` truth_table_idx 决定非切换引脚，
  切换引脚初始态 rise→0 / fall→1）。GNN 自 8.5 起把 vector 解码为 per-pin 逻辑状态特征，
  训练学到的映射与 Rust 推理喂来的初始电平**语义相反** → rise 场景排序直接错。
- **修复要求**：vector 按初始电平生成（rise 行切换位=0，fall 行=1），非切换位按 truth_table_idx。

### P4 🟠 sc_expansion.json：1456/3653 个 SC_ 名展开为空（40%）

- **现象/证据**：V2 数据 `cell_types_json` 共 3653 个 SC_ 名，`sc_expansion.json`（8909 条）中 **0 个缺失，但 1456 个 subcircuit 为空**；
  且 `coverage_report_v2.json` 声称 100% —— **报告与事实不符**。
- **问题所在**：97fd7fda 合并 V1+V2 宏时只加了名字条目，subcircuit 未填（或 V2 新宏尚未展开）。
- **Rust 集成影响**：STRUCT_MODE 结构特征（n_t/stack/parallel）48% 靠 sc_expansion、52% 回退默认；
  40% V2 名展开为空 → 回退率大幅上升，structrich/structbase 变体退化（structlogic 纯 10 逻辑受影响小）。
  **Rust 的 cell 命名是另一套**（`SC_JOIN_AND_AND` 等，7.2 已证 5/7 OOV）——V2 自己的名字都查不到展开，
  Rust 的名字更查不到 → 结构特征全回退。对应 `GNN_RUST_DATA_DIFF.md` 9.3 待办「OOV 名结构精度」：
  要么补 sc_expansion 覆盖，要么改走路线 A 完整版（从 Rust `.sp` 的 `M_` 行直接解析晶体管结构）。
- **修复要求**：所有出现过的 SC_ 名 subcircuit 非空可展开；coverage_report 如实报告。

### P5 🟠 ids_charge == ids_avg（100% 完全相等）

- **现象/证据**：抽样 136,228 个晶体管，`ids_charge / ids_avg` 中位比值 = 1.0000，**100% 完全相等**。
- **问题所在**：后处理提取 ids_charge 时直接复制了 ids_avg，∫|Ids|dt 电荷积分未做（量纲都丢了：fC vs μA）。
- **Rust 集成影响**：wave 是排序 game-changer（newwave：Spearman 0.705、遗憾 54%→5.3%），
  `ids_charge` 是规格标注的「直接决定延迟的物理量」。7 个子字段 1 个是假值 → 模型学到「电荷=平均电流」的错误相关性。
  wave 是 **train-only**（Rust 推理拿不到），计划走蒸馏（teacher 有 wave → student 无 wave）——
  假 ids_charge 让 teacher 物理信号带噪，蒸馏收益打折；若 Rust 侧以后真算 ids_charge（正确量纲），分布立刻漂移。
- **修复要求**：ids_charge 按翻转窗口 ∫|Ids|dt 重算，不得等于 ids_avg。

### P6 🟡 缺行：`candidate_expr8089_0014` 只有 7 行（缺 a/fall）

- **现象/证据**：8,679/8,680 个电路恰 8 行（2×N_in×M），1 个电路 7 行，缺 `(a, fall)` 组合。
- **问题所在**：该 (pin,dir) 仿真行缺失——仿真失败被过滤或输出不翻转被跳过。
- **Rust 集成影响**：破坏行数公式 → 该电路 avg_delay 聚合（对齐 Rust avg_delay）少一个样本，
  **聚合口径与其它电路不一致**，组内排序比较产生系统性偏差。Rust 侧 `simulate_all_outputs_for_expr` 每 (output,pin) 都有 arc。
- **修复要求**：补全该行；或生成侧加「行数完整性」自检（每电路 = 2×N_in×M）。

### P7 🟡 supply_noise 全零（100%）

- **现象/证据**：抽样 5,000 行，`vdd_droop_mV` 与 `gnd_bounce_mV` **全部为 0.0**。
- **问题所在**：提取逻辑未接入（或翻转窗口内未测到），直接写 0.0 占位。规格允许 0.0 为有效值，但全零 = 字段无信息量。
- **Rust 集成影响**：13.5 消融已证 supply_noise 无贡献（Spearman 0.237），且为 train-only → **对当前集成影响最小**。
  隐患：训练学到「噪声恒 0」偏置，若 Rust 阶段 2 真提取噪声喂入，分布立刻漂移。
- **修复要求**：要么真实提取翻转窗口内的 droop/bounce，要么在规格中将该字段标注为「当前交付为占位」。

### P8 🟡 batch_v2 损坏子集（input_pins_json 全空、网表无输入引脚）

- **现象/证据**：`data/batch_v2`（290 电路）的 `input_pins_json` 全部为 `[]`，网表 `.SUBCKT DUT out vdd gnd`
  无输入引脚，而动态表 switching_pin 却用 a/b/c/d。
- **问题所在**：该批走了与 full 不同的生成路径，静态表未写输入引脚。与 full 的 290 个电路完全重复。
- **Rust 集成影响**：误用则 `parse_netlist` 拿空输入 → 图构建失败；与 full 混用不 dedup 则双重计数
  （检查脚本首版即踩坑：16 行/重复行全是它造成的假象）。Rust 侧任意 I/O 的引脚来自 INORDER 列表，该批连列表都没有。
- **修复要求**：**废弃该批**（full 已含全部电路），或修复后交付。

### P9 🟠 I/O 形状未达 V2 规格（结构性缺口，非值级 bug）

- **现象/证据**：全部电路 1~4 入 / 1 出（4 入占 98%）；输入分桶 1~2:45 / 3~4:8,815 / 5~8:0 / 9~16:0；多输出 0。
- **问题所在**：生成批次仍是 V1 时代的 4-pin 枚举，未按 V2 规格做任意 N(1~16)/M(1~6) + 等比例分桶。
- **Rust 集成影响**：Rust benchmark 是任意 N 入（ADD4_OVF = 9 入 6 出；简单 baseline 输出名都是 `y`）。
  训练数据全是 4-pin → **GNN 代码侧 9.1~9.3（parse_netlist 任意 I/O、data_loader JSON pin 列、多输出读出）
  改了也没数据可训**；serve.py 对 Rust 的 9 入 6 出电路只能硬扛 OOD。**该条是 9.1-9.3 代码改动能落地的唯一前提。**
  若按 2026-08-20 讨论与数量问题一并暂缓，则模型侧 V2 训练停摆，只能先推进「无 wave」验证（V1 数据）。
- **修复要求**：按 V2 规格分桶生成（输入 1~2/3~4/5~8/9~16 各 ~25%，多输出 ≥20%）。

---

## 三、给生成方的修复优先级清单

| 优先级 | 问题 | 修复动作 | 验证 |
|---|---|---|---|
| P0 | P1 大小写 | JSON 实例名 key 与网表 `X_*` 一致 | `_check_v2_data.py` 大小写检查 100% |
| P0 | P2 direction | 去 `.sp` 后缀 | direction ∈ {rise, fall} |
| P0 | P3 vector | 切换位 rise→0 / fall→1 | vector 位校验 100% |
| P1 | P4 sc_expansion | 1456 个空 subcircuit 补齐 | sc_expansion 覆盖 100% 可展开 |
| P1 | P5 ids_charge | 重算 ∫|Ids|dt | ids_charge ≠ ids_avg（比值≠1） |
| P2 | P6 缺行 | 补行 + 行数自检 | 每电路 = 2×N_in×M |
| P2 | P7 supply_noise | 真提取或标注占位 | 非全零 或 规格标注 |
| P2 | P8 batch_v2 | 废弃 | 交付仅 full |
| 暂缓 | P9 I/O 形状 + 数量 | 任意 N/M 分桶 + 满量 | 分桶分布 + 总量达标 |

**复检**：修复后重新跑 `python _check_v2_data.py data\batch_v2_full`，目标：无 FAIL（WARN 需人工确认）。
脚本见仓库根目录 `_check_v2_data.py`；首轮明细 `_v2check_full.txt`；检查记录见 `PROJECT_LOG.md` 14.4.5 节。
