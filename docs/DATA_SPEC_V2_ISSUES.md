# DATA_SPEC_V2 数据合规问题清单（版本记录式）

> 检查方法：`scripts/diag/_check_v2_data.py` 逐项对照 `docs/DATA_SPEC_V2.md`（定稿）自动校验；检查对象为 GitHub
> `10.3.3-fix-earlystop` 分支交付的 V2 数据（`data/batch_v2_full` + `data/batch_v2_io`）。
> 明细：首轮 `reports/_v2check_full.txt`、复检 `reports/_v2check_fix2.txt`、V3.0 `reports/_chk_full3.txt`/`reports/_chk_io.txt`；检查记录见 `docs/PROJECT_LOG.md` 14.4.5 节。
>
> **范围说明**：生成**数量**问题在 V1.0/V2.0 按 2026-08-20 讨论暂缓；**V3.0/V3.1（2026-08-25）起数量与 I/O 覆盖绑定、列为正式要求**（见第四节）；
> **V3.1 决策：现有 `batch_v2_full` + `batch_v2_io`（968 expr）不计入总数，正式交付重新生成完整 ~4,000 expr / ~60 万行（见 4.3）**。

---

## 版本记录

| 版本 | 日期 | 状态 | 变更摘要 |
|---|---|---|---|
| **V1.0** | 2026-08-20 | ❌ **不合格，需返工** | 首轮全量检查（commit `97fd7fda` 交付）：发现 **P1~P10 共 10 项问题**——值级硬伤 6 项（P1 大小写 / P2 direction / P3 vector / P4 sc_expansion / P5 ids_charge / P6 缺行）+ 说明类 2 项（P7 supply_noise / P8 batch_v2）+ 结构性 2 项（P9 I/O 形状 / P10 coverage_report）。退回生成方返工。 |
| **V2.0** | 2026-08-24 | 🟡 **基本达标，仅剩小项** | 生成方提交修复 commit `a2b5d36` 后复检（44 PASS / 2 FAIL / 1 WARN）：**P1/P2/P3/P6/P8 已修复、P7 已说明、P4/P5/P10 大部分修复**；**本轮仅剩 3 个小项需修改**（R1 sc_expansion 12 名 null / R2 parasitic 12 电路缺 in_* / R3 ids_charge 11 个复制残留，均小量级）。**新增 P9 I/O 实证**：Rust benchmark 实测 46 电路 18 种形状，当前数据仅覆盖 ~22% → P9 升级为 Rust 集成最高阻塞。 |
| **V3.0** | 2026-08-25 | 🟡 **值级清零，剩数量 / I-O 覆盖缺口** | 生成方提交 `8d47b2d`（修 R1/R2/P10）+ `8ca3004`（交付任意 I/O 批 `batch_v2_io`）后复检：**batch_v2_full = 46 PASS / 0 FAIL**（R1/R2/P10 全修 ✅）；**batch_v2_io = 45 PASS / 1 FAIL / 1 WARN**——P9 主体达成（25 种形状、多输出 20.4%、分桶大致均衡），**但缺 5 个 benchmark 形状（含 9入6出 ADD4_OVF，≥4 输出零覆盖）、36 个退化组、分桶 9~16 偏低**；**总量 12,927 电路 / 968 expr 仅规格 ~1/4 → 续产 ~3,032 expr 必须带 I/O 覆盖（见第四节）**。 |
| **V3.1** | 2026-08-25 | 🔴 **口径变更：现有数据不计入总数** | 决策：`batch_v2_full` + `batch_v2_io`（968 expr）**不计入**规格总量，仅作前期开发/验证数据；正式交付**从零重新生成完整 ~4,000 expr / ~60 万行**，按 4.4 I/O 分配策略分桶（20/25/25/30、多输出 ≥20%、M≥4 每个 ≥30 组、补 G1 缺失形状、无退化组），消除「full 计入总数导致 3~4 桶 70%」的口径矛盾。 |
| **V3.2** | 2026-08-28 | 🟡 **满量已交付，剩 5 形状缺口** | 满量数据到位（`batch_v2_rest` 35,183 电路 / 568,090 行，三批合计 48,110 电路 / 695,286 行 / **3,981 expr**，数量达标：~4,000 expr / ~5 万电路 / ~60 万行 ✓；输入分桶合并后 24.6/25.6/23.7/26.1% ✓、多输出 23.1% ✓）。**唯一实质缺口 = Rust benchmark 5 个形状零覆盖**（9入6出 / 8入4出 / 7入4出 / 5入5出 / 4入3出，M≥4=0，且 4入3出 也是缺的）。**补充要求见第五节：只生成这 5 个形状，~10.9 万行，与现有 3,981 expr 不重复**。 |

---

## 一、V1.0 首轮问题清单（2026-08-20，10 项）

> 首轮数据：batch_v2（290 电路，损坏子集）+ batch_v2_full（8,860 电路 / 69,439 行）。下表为总览，「状态」列是 V2.0 复检结果；详细条目见下。

| # | 问题 | 严重度 | V2.0 状态 |
|---|---|---|---|
| P1 | 大小写不一致：JSON key `x_*` vs 网表 `X_*` | 🔴 致命 | ✅ 已修复 |
| P2 | direction 带 `.sp` 后缀（`rise.sp`/`fall.sp`） | 🔴 高 | ✅ 已修复 |
| P3 | vector 切换位恒为 1（rise 行应为 0） | 🔴 高 | ✅ 已修复 |
| P4 | sc_expansion.json 2716/8909 条目 null（30.5%，几乎全是 JOIN） | 🟠 中高 | 🟡 大部分（剩 12 名 = R1） |
| P5 | ids_charge == ids_avg（100% 复制） | 🟠 中 | 🟡 基本（剩 0.005% = R3） |
| P6 | 缺行：`candidate_expr8089_0014` 只有 7 行 | 🟡 低 | ✅ 已修复（剔除 181 个不完整电路） |
| P7 | supply_noise 全零（100%） | 🟡 低 | ✅ 已说明（ideal-supply 合法） |
| P8 | batch_v2 损坏子集（input_pins_json 全空） | 🟡 低 | ✅ 已废弃（文件删除） |
| P9 | I/O 形状未达任意 N(1~16)/M(1~6)（全 4 入 1 出） | 🟠 结构性 | 🟡 主体已交付（V3.0 batch_v2_io），剩 G1 缺 5 形状 / G2 退化组 / G3 分桶 |
| P10 | coverage_report_v2.json 自身问题（假 100%） | 🟠 中 | 🟡 大部分（值级项已加，2 处口径残留） |

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
  `X_*` 大小写完全一致。✅ **V2.0 已修**（复检 3000/3000 一致）。

### P2 🔴 direction 带 `.sp` 后缀（`rise.sp` / `fall.sp`）

- **现象/证据**：`timing_arcs.direction` 全部行为 `rise.sp` / `fall.sp`（两批均如此）。
- **问题所在**：生成脚本把 SPICE 波形文件名（`rise.sp`/`fall.sp`）当 direction 值写入，未剥离后缀。
- **Rust 集成影响**：Rust 侧 `SimuVector.timing_sense` 是纯 `rise`/`fall`。GNN 训练学到的是 `rise.sp` 脏值；
  serve.py 推理时从 Rust 拿到真 `rise` → **推理值在训练分布之外（OOV）**，direction 特征编码失效。
  且 V2 规格明确 `direction ∈ {rise, fall}`，9.2 data_loader 按 direction 做特征与 avg 聚合，脏值污染口径。
- **修复要求**：`direction` 值去掉 `.sp` 后缀（一行 `replace('.sp','')` 级别）。✅ **V2.0 已修**（原始值 = {rise, fall}）。

### P3 🔴 vector 切换位恒为 1（rise 行应为 0）

- **现象/证据**：抽样 2 万行，切换引脚对应位 **100% 为 1**（含 rise 行）。按规格「rise→0, fall→1」校验仅一半行通过。
- **问题所在**：vector 编码方向反了——写入的是翻转后稳态电平（或取错了 truth_table 行），未按 V2 规格
  §三「switching_pin 对应位与 direction 一致」生成。因所有行错得一致，『每 (circuit,pin,dir) 恰 1 vector』检查通过，掩盖了问题。
- **Rust 集成影响**：Rust vector 语义 = **初始逻辑电平**（`SimuVector` truth_table_idx 决定非切换引脚，
  切换引脚初始态 rise→0 / fall→1）。GNN 自 8.5 起把 vector 解码为 per-pin 逻辑状态特征，
  训练学到的映射与 Rust 推理喂来的初始电平**语义相反** → rise 场景排序直接错。
- **修复要求**：vector 按初始电平生成（rise 行切换位=0，fall 行=1），非切换位按 truth_table_idx。✅ **V2.0 已修**（20000/20000 一致）。

### P4 🟠 sc_expansion.json：2716/8909 条目值为 null（30.5%），几乎全是 SC_JOIN_*

- **现象/证据**：`sc_expansion.json`（8909 条）中 **2716 条（30.5%）值为 null（无 subcircuit）**，前缀分布 JOIN 2701 / BRIDGE 9 / AND 3 / OR 3。
  V2 数据 `cell_types_json` 用到的 3653 个名字中 **1456 个（39.9%）不可展开**，含高频名 `SC_BRIDGE`（x708）、`SC_JOIN_WIRE_WIRE`（x401）、`SC_JOIN_BRIDGE__BRIDGE`（x183）。
  可展开条目质量 OK：124,579 个内部实例无缺字段，引用 11 个标准单元全部存在于 `std_cells.lib`（93 单元）——**问题集中在 JOIN 类宏没填**。
  且 `coverage_report_v2.json` 声称 100% —— 只查名字存在、不查可展开，**报告与事实不符（见 P10）**。
- **问题所在**：97fd7fda 合并 V1+V2 宏时只加了名字条目，subcircuit 未填（或 V2 新宏尚未展开）。
- **Rust 集成影响**：STRUCT_MODE 结构特征（n_t/stack/parallel）48% 靠 sc_expansion、52% 回退默认；
  40% V2 名展开为空 → 回退率大幅上升，structrich/structbase 变体退化（structlogic 纯 10 逻辑受影响小）。
  **Rust 的 cell 命名是另一套**（`SC_JOIN_AND_AND` 等，7.2 已证 5/7 OOV）——V2 自己的名字都查不到展开，
  Rust 的名字更查不到 → 结构特征全回退。对应 `docs/GNN_RUST_DATA_DIFF.md` 9.3 待办「OOV 名结构精度」：
  要么补 sc_expansion 覆盖，要么改走路线 A 完整版（从 Rust `.sp` 的 `M_` 行直接解析晶体管结构）。
- **修复要求**：所有出现过的 SC_ 名 subcircuit 非空可展开；coverage_report 如实报告。🟡 **V2.0 大部分**（null 30.5%→0.28%，剩 12 名 = 本轮 R1）。

### P5 🟠 ids_charge == ids_avg（100% 完全相等）

- **现象/证据**：抽样 136,228 个晶体管，`ids_charge / ids_avg` 中位比值 = 1.0000，**100% 完全相等**。
- **问题所在**：后处理提取 ids_charge 时直接复制了 ids_avg，∫|Ids|dt 电荷积分未做（量纲都丢了：fC vs μA）。
- **Rust 集成影响**：wave 是排序 game-changer（newwave：Spearman 0.705、遗憾 54%→5.3%），
  `ids_charge` 是规格标注的「直接决定延迟的物理量」。7 个子字段 1 个是假值 → 模型学到「电荷=平均电流」的错误相关性。
  wave 是 **train-only**（Rust 推理拿不到），计划走蒸馏（teacher 有 wave → student 无 wave）——
  假 ids_charge 让 teacher 物理信号带噪，蒸馏收益打折；若 Rust 侧以后真算 ids_charge（正确量纲），分布立刻漂移。
- **修复要求**：ids_charge 按翻转窗口 ∫|Ids|dt 重算，不得等于 ids_avg。🟡 **V2.0 基本**（改为 peak×rise_time/1000 fC，剩 0.005% = 本轮 R3）。

### P6 🟡 缺行：`candidate_expr8089_0014` 只有 7 行（缺 a/fall）

- **现象/证据**：8,679/8,680 个电路恰 8 行（2×N_in×M），1 个电路 7 行，缺 `(a, fall)` 组合。
- **问题所在**：该 (pin,dir) 仿真行缺失——仿真失败被过滤或输出不翻转被跳过。
- **Rust 集成影响**：破坏行数公式 → 该电路 avg_delay 聚合（对齐 Rust avg_delay）少一个样本，
  **聚合口径与其它电路不一致**，组内排序比较产生系统性偏差。Rust 侧 `simulate_all_outputs_for_expr` 每 (output,pin) 都有 arc。
- **修复要求**：补全该行；或生成侧加「行数完整性」自检（每电路 = 2×N_in×M）。✅ **V2.0 已修**（剔除 181 个不完整电路，0 不匹配）。

### P7 🟡 supply_noise 全零（100%）

- **现象/证据**：抽样 5,000 行，`vdd_droop_mV` 与 `gnd_bounce_mV` **全部为 0.0**。
- **问题所在**：提取逻辑未接入（或翻转窗口内未测到），直接写 0.0 占位。规格允许 0.0 为有效值，但全零 = 字段无信息量。
- **Rust 集成影响**：13.5 消融已证 supply_noise 无贡献（Spearman 0.237），且为 train-only → **对当前集成影响最小**。
  隐患：训练学到「噪声恒 0」偏置，若 Rust 阶段 2 真提取噪声喂入，分布立刻漂移。
- **修复要求**：要么真实提取翻转窗口内的 droop/bounce，要么在规格中将该字段标注为「当前交付为占位」。✅ **V2.0 已说明**（ideal-supply，VDD/GND 理想源物理上即 0，符合规格 §六 方案 D）。

### P8 🟡 batch_v2 损坏子集（input_pins_json 全空、网表无输入引脚）

- **现象/证据**：`data/batch_v2`（290 电路）的 `input_pins_json` 全部为 `[]`，网表 `.SUBCKT DUT out vdd gnd`
  无输入引脚，而动态表 switching_pin 却用 a/b/c/d。
- **问题所在**：该批走了与 full 不同的生成路径，静态表未写输入引脚。与 full 的 290 个电路完全重复。
- **Rust 集成影响**：误用则 `parse_netlist` 拿空输入 → 图构建失败；与 full 混用不 dedup 则双重计数
  （检查脚本首版即踩坑：16 行/重复行全是它造成的假象）。Rust 侧任意 I/O 的引脚来自 INORDER 列表，该批连列表都没有。
- **修复要求**：**废弃该批**（full 已含全部电路），或修复后交付。✅ **V2.0 已废弃**（文件已删除，仅交付 full）。

### P9 🔴 I/O 形状未达 V2 规格（结构性缺口，非值级 bug）—— 实证见 V2.0 第 2.4 节

- **现象/证据（V1.0 首轮）**：全部电路 1~4 入 / 1 出（4 入占 98%）；输入分桶 1~2:45 / 3~4:8,815 / 5~8:0 / 9~16:0；多输出 0。
- **问题所在**：生成批次仍是 V1 时代的 4-pin 枚举，未按 V2 规格做任意 N(1~16)/M(1~6) + 等比例分桶。
- **Rust 集成影响**：Rust benchmark 是任意 N 入（ADD4_OVF = 9 入 6 出；简单 baseline 输出名都是 `y`）。
  训练数据全是 4-pin → **GNN 代码侧 9.1~9.3（parse_netlist 任意 I/O、data_loader JSON pin 列、多输出读出）改了也没数据可训**；
  serve.py 对 Rust 的 9 入 6 出电路只能硬扛 OOD。**该条是 9.1-9.3 代码改动能落地的唯一前提。**
- **修复要求**：按 V2 规格分桶生成（输入 1~2/3~4/5~8/9~16 各 ~25%，多输出 ≥20%）。❌ **未修**（V2.0 已用 Rust benchmark 实测实证，见 2.4）。

### P10 🟠 coverage_report_v2.json 自身问题（随数据交付的覆盖率报告不可信）

- **现象/证据**：报告声称全部 100%，但三处与事实不符或无效：
  1. `sc_expansion.coverage = 3653/3653 (100%)` —— 只检查「名字存在于 sc_expansion.json」，**不检查 subcircuit 是否可展开**（实际 1456/3653 值为 null，见 P4）；
  2. `parasitic_caps_json.gate_key_match = 8860/8860 (100%)` —— **假阳性**：生成方匹配口径未与网表 `X_*`（大写）对齐（数据里 key 是小写 `x_*`），graph_builder 按网表名查必 miss（与 P1 同源）；
  3. `vector.one_per_group = OK (100%)` —— **假通过**：所有行切换位都错得一致（P3），「每 (circuit,pin,dir) 恰 1 vector」恒成立，掩盖了值错误。
- **问题所在**：报告只做「列非空 + 结构存在」类检查，**没有任何值级校验**（direction 值、vector 位、ids_charge 语义都查不到）。
- **Rust 集成影响**：交付方凭该报告自认为合格 → 值级错误（P2/P3/P5）被「100% 覆盖」掩盖，直接进训练/推理会静默出错；接收方也无法用它做验收。
- **修复要求**：① sc_expansion 覆盖改为「可展开率」口径；② gate_key_match 与网表 `X_*` 大小写敏感比对；③ 增加值级检查（direction ∈ {rise,fall}、vector 切换位、ids_charge≠ids_avg）；④ 或直接改用 `scripts/diag/_check_v2_data.py` 作为验收脚本。
- 注：报告做得好的部分——transistor_wave 7 子字段逐个报告、supply_noise 2 字段、DELAY 范围、per_gate 缺席确认；null 覆盖数据为真（100% 非空是真的，只是「非空 ≠ 值对」）。
- 🟡 **V2.0 大部分**：已加 `direction.valid_values` / `sc_expansion.expandable 99.7%` / `ids_charge.not_equal_avg 75.2%`（诚实上报）；残留 2 处口径（batch_v2 旧段未清、`subfields_valid` 仍漏 in_* 缺失，见本轮 R2）。

---

## 二、V2.0 修复版复检（2026-08-24，commit `a2b5d36` "Fix V2 data compliance (P1-P10)"）

> **本轮结论：基本达标，只需修改小的方面。** 复检结果：**44 PASS / 2 FAIL / 1 WARN**（首轮 36/4/8）。
> 数据：8,679 电路 / 69,432 行 / 579 expr（剔除 181 个不完整电路），全部 4 入 1 出、单 corner `s02p0_l01p0`。

### 2.1 已修复 / 已说明（对照首轮）

| 原问题 | 状态 | 复检证据 |
|---|---|---|
| P1 大小写 | ✅ 已修复 | gate_states / parasitic_caps key 与网表 `X_*` 一致（3000/3000）；transistor_wave.gate 大写（2000/2000） |
| P2 direction | ✅ 已修复 | 原始值 = {rise, fall}（`.sp` 后缀已剥） |
| P3 vector | ✅ 已修复 | 切换位 20000/20000 与 direction 一致（rise→0 / fall→1 各半） |
| P6 缺行 | ✅ 已修复 | 每电路 8 行，0 不匹配（181 个不完整电路已剔除） |
| P7 supply_noise | ✅ 已说明 | 全零 + coverage_report 注明 ideal-supply（物理上即 0，符合规格 §六 方案 D） |
| P8 batch_v2 | ✅ 已废弃 | 文件已删除，仅交付 batch_v2_full |
| P10 coverage_report | 🟡 大部分 | 值级项已加（direction / expandable / ids_charge 诚实上报）；残留 2 处口径（见本轮 R2 附注） |

### 2.2 本轮需修改的小项（3 项）

- **R1（FAIL）sc_expansion 12 个 SC_ 名仍为 null**（V2 用到的 3586 名中 0.33%，全文件 25 个 null）：
  `SC_BRIDGE`（使用 686 次）、`SC_JOIN_WIRE_WIRE`（378）、`SC_JOIN_BRIDGE__BRIDGE`（170）、`SC_JOIN_BRIDGE`（81）等，
  总计 **1414 次 cell 使用**落到 STRUCT_MODE 默认特征回退。修复：补 25 个 null 条目（BRIDGE / JOIN-WIRE 类宏展开）。
- **R2（FAIL）parasitic_caps 12 个电路（抽样 3000 中 12 个）某门缺 `in_*` 子字段**（如 `X_51: {"out": 0.2}` 缺 in_a）——
  与 R1 的 BRIDGE 单输入门疑似同源；coverage_report `subfields_valid` 却声称 100%（口径只数「已有子字段非空」，漏「必需子字段缺失」）。
- **R3（WARN）ids_charge 激活管 0.72% == ids_avg**：核实 1706 个中 1695 个是**2 位小数取整巧合**
  （charge = ids_peak × ids_rise_time / 1000 与 avg 同取整），**真复制残留仅 11 / 23 万激活管（0.005%）**，可忽略。
  新公式物理量级合理（例：21μA × 20ps → 0.42 fC）✅。

### 2.3 本轮新增实证：P9 I/O 形状（Rust benchmark 实测，2026-08-24）

> 数据来源：`NetlistOpt/testbench/tl_cells/`（46 个 .tl，level0~4），逐一解析 INORDER/OUTORDER 实测。

**三组对照**：
1. **V2 规格要求**（`docs/DATA_SPEC_V2.md` L4/L28-29/L172/L190）：任意 N-in（1~16）/ M-out（1~6），输入分桶 1~2/3~4/5~8/9~16 **各 ~25%**，输出 1/2/3+ 三档，**多输出 ≥20%**。
2. **Rust benchmark 实测**：**18 种不同 I/O 形状**，输入 1~16、输出 1~6，**多输出 11 个（~24%）**；`tl_opt_smoke` 测试实际跑 `level4/ADD4_OVF.tl`（**9 入 6 出**）。→ **规格与 Rust 实测一致，问题完全在生成侧。**
3. **当前交付 batch_v2_full**：修复版（8679 电路）**全部 4 入 1 出**（分桶 3~4 档 100%，5~8 / 9~16 档 0，多输出 0）→ **只覆盖 benchmark 46 个电路中的 10 个（~22%）**。

**Rust benchmark I/O 形状实测分布（46 个 .tl）**：

| n_in | n_out | 电路数 | 例子 |
|---|---|---|---|
| 1 | 1 | 1 | INV1 |
| 2 | 1 | 7 | AND2、XOR2、OR2、EQ2… |
| 3 | 1 | 9 | MUX2、OAI21、FA_COUT、MAJ3… |
| **4** | **1** | **10** | AOI22、EQ4、SHARE1/2、SEL_AND… ← 唯一被当前数据覆盖的形状 |
| 5 | 1 | 3 | AOI221、OAI221、DEPTH_MIX |
| 8 | 1 | 2 | AND8、OR8 |
| 9 | 1 | 2 | OVF、ovf1 |
| 16 | 1 | 1 | AND16 |
| 2/3/5 | 2 | 4 | HA、FA、CSA_3_2、ALU_SLICE_SMALL |
| 2/4/8 | 3 | 3 | COMP1、ENC4、COMP4 |
| 7/8 | 4 | 2 | ALU2、ENC8 |
| 5 | 5 | 1 | SHIFTER4 |
| 9 | 6 | 1 | **ADD4_OVF（tl_opt_smoke 实测电路）** |

**影响**：Rust 贪心评估的是**整电路全局延迟**（任意 I/O，含 9 入 6 出 ADD4_OVF）。训练数据全是 4-pin →
GNN 代码侧 9.1~9.3（parse_netlist 任意 I/O、data_loader JSON pin 列、多输出读出）**改了也没数据可训**；
serve.py 对 benchmark 46 个中 36 个（含全部 11 个多输出）只能硬扛 OOD。**P9 是 Rust 集成路径上最大的阻塞项。**

---

## 三、当前待生成方处理的清单（截至 V3.0）

> 值级硬伤（P1/P2/P3/P6/P8）与残留（R1/R2/P10）已清零；**当前需处理：I/O 覆盖缺口（G1/G2/G3）+ 数量（带 I/O 绑定）**。

| 优先级 | 项 | 内容 | 验证 |
|---|---|---|---|
| ✅ 已完成 | R1/R2/P10（8d47b2d） | sc_expansion null 补齐、parasitic in_* 补齐、报告清 batch_v2 段 | batch_v2_full 复检 46 PASS / 0 FAIL |
| 小项 P2 | R3（P5 残留） | ids_charge 真复制残留（两批「相等」中 99% 为取整巧合，真残留 ~0.01%） | 可忽略，不处理也行 |
| **大项 P0** | **G1 缺 benchmark 形状** | **补 5 个缺失形状：9入6出（ADD4_OVF，tl_opt_smoke 实测电路）、8入4出（ENC8）、7入4出（ALU2）、5入5出（SHIFTER4）、4入3出（ENC4）——当前 ≥4 输出形状零覆盖** | batch_v2_io 覆盖全部 18 种 benchmark 形状 |
| **大项 P0** | **G2 退化/小组** | batch_v2_io 36 个单变体组 + 3 个小组（共 39 组，10%）不合 10~15 规格；退化组应剔除或补变体 | 全部组 ≥10 变体 |
| 中项 P1 | G3 分桶 9~16 偏低 | batch_v2_io 分桶 1~2:23.4% / 3~4:24.3% / 5~8:35.4% / 9~16:16.8%（9~16 应 ~25%） | 分桶各 ~25% |
| **大项 ⏸** | **完整数据重新生成（V3.1）** | **全新生成 ~4,000 expr / ~60 万行（不计入现有 968），按 4.4 分桶（20/25/25/30）+ 多输出 ≥20% + M≥4 每个 ≥30 组 + 补 G1 缺失形状 + 无退化组；expr 编号与现有不重叠** | `scripts/diag/_check_v2_data.py` 全绿 + 分桶分布达标 |

**复检方法**：`python scripts/diag/_check_v2_data.py data\batch_v2_full` / `python scripts/diag/_check_v2_data.py data\batch_v2_io`，目标：无 FAIL（WARN 需人工确认）。
脚本见仓库根目录 `scripts/diag/_check_v2_data.py`（V3.0 起含多输出行身份扩展）；明细 `reports/_v2check_full.txt`、`reports/_v2check_fix2.txt`、`reports/_chk_full3.txt`、`reports/_chk_io.txt`；I/O 实证脚本 `scripts/diag/_check_tl_io.py`；检查记录见 `docs/PROJECT_LOG.md` 14.4.5 节。

---

## 四、V3.0 复检与新问题（2026-08-25，commits `8d47b2d` + `8ca3004`）

### 4.1 复检结果

| 批次 | 结果 | 说明 |
|---|---|---|
| batch_v2_full | **46 PASS / 0 FAIL / 1 WARN** | R1（sc_expansion 12 名 null）/ R2（parasitic in_*）/ P10（报告清段）全部修复 ✅；仅剩 ids_charge 取整巧合 WARN（可忽略） |
| batch_v2_io | **45 PASS / 1 FAIL / 1 WARN** | P9 主体达成（25 种形状 / 多输出 20.4% / 分桶大致均衡）；FAIL = ids_charge 阈值口径（真残留仅 0.009%）；WARN = 组大小（G2） |

### 4.2 当前问题清单（V3.0 新增）

- **G1（大项）5 个 benchmark 形状缺失，≥4 输出零覆盖**：9入6出 **ADD4_OVF（tl_opt_smoke 实测电路）**、8入4出 ENC8、7入4出 ALU2、5入5出 SHIFTER4、4入3出 ENC4。
  commit `8ca3004` 声称「18 I/O shapes incl. 9-in 6-out」，**实际交付没有 9入6出**——数据最大输出仅 3（9入3出 19 个 / 16入3出 5 个），多输出读出（GNN 9.3）最需要的 M≥4 场景零覆盖。
- **G2（大项）36 个退化组 + 3 个小组**：batch_v2_io 共 389 组，中位 12，但 36 组仅 1 变体（无组内排序价值）+ 3 组 3~5 变体，共 39 组（10%）不合 10~15 规格。
- **G3（中项）分桶 9~16 偏低**：1~2:23.4% / 3~4:24.3% / 5~8:35.4% / 9~16:16.8%（目标各 ~25%，9~16 差 ~8pp）。
- **G4（小项）ids_charge 检查口径**：io 批 1.02%「相等」中 99% 为 2 位小数取整巧合（charge=peak×rise_time/1000 与 avg 同取整），**真复制残留 26/27.5 万激活管（0.009%）**——`scripts/diag/_check_v2_data.py` ≥1% 阈值触发 FAIL 是口径问题，非数据问题。
- **数量（大项）总量仅规格 ~1/4**：见 4.3。

### 4.3 总量口径（V3.1 决策：现有数据不计入总数，重新生成完整数据）

> **决策（2026-08-25）**：`batch_v2_full`（579 expr）+ `batch_v2_io`（389 expr）= 968 expr **不计入**规格总量；
> 现有数据仅用于**前期开发/验证**（4-pin 训练冒烟、GNN 9.1~9.3 代码验证、pipeline 检查）。
> 正式交付**从零重新生成完整数据**，消除「现有 full 计入总数导致 3~4 桶 70%」的口径矛盾。

**正式交付口径：**

1. **总量**：~4,000 expr / ~5 万电路 / ~60 万行（**全新生成，不包含**现有 968 expr）。
2. **I/O 组成**：按 4.4 分配策略分桶——输入 1~2:20% / 3~4:25% / 5~8:25% / 9~16:30%、输出 1/2/3+ 三档、
   多输出 ≥20%、**M≥4 每个形状 ≥30 组**、**覆盖全部 18 种 benchmark 形状（含 G1 缺失的 9入6出 等）**、
   全部组 ≥10 变体（**无退化组**）。
3. **expr 编号**：与现有 968 expr **不重叠**（建议从 expr10000+ 起或生成方自定义新段），避免误混。
4. **值级标准**：沿用 V2.0/V3.0 已验证的检查项（`scripts/diag/_check_v2_data.py` 全绿：大小写 / direction / vector / ids_charge /
   sc_expansion / parasitic_caps / 行数公式等）。
5. **现有数据去向**：保留作开发集；交付时可选随正式数据附送作 bonus（不计入正式总量与分桶统计）。

### 4.4 I/O 分配策略（正式数据生成指南，2026-08-25）

> 原则：**不是纯均分，也不是纯按 Rust benchmark 频率**——「每个形状保证最小可用量」+「难例 / 关键形状偏重」，
> 训练分布贴近推理分布，同时把样本预算花在收益最高的地方（模型误差集中在难例：B2/B3/大 I/O ≫ B1 小电路）。

| 维度 | 建议 | 理由 |
|---|---|---|
| 输入分桶 | 基线各 ~25%，调整为 **1~2: ~20% / 3~4: ~25% / 5~8: ~25% / 9~16: ~30%** | 9~16 入是 Rust 最需要模型帮忙的难例区（上浮）；1~2 入简单电路样本饱和快（下浮） |
| 输出档 | 多输出 ≥20%；**M≥4 形状单独保证最小量（每个 ≥30 组）** | G1 缺口恰在 ≥4 输出（9入6出 ADD4_OVF / 8入4出 ENC8 / 7入4出 ALU2 / 5入5出 SHIFTER4 / 4入3出 ENC4），多输出读出（GNN 9.3）依赖这些样本 |
| 形状粒度 | **不要求 18 种 benchmark 形状等量**；每种形状保证最小样本量（≥30 组），组内 10~15 变体 | 全覆盖 + 难例偏重，避免极端形状（9入6出）被频率淹没 |
| 组大小 | 全部组 ≥10 变体；**剔除 / 补足退化组（G2，当前 36 个单变体组）** | 排序任务需要组内对比，单变体组无价值 |
| 权重依据 | 难度（大 I/O / 多输出排序误差大）+ **Rust 真实调用频率**（后续打点统计校准） | 46 个 benchmark 太小，不能当唯一频率锚点 |

**锚点补充**：建议后续从 `NetlistOpt` 给 `optimize_tl_text` / `simulate_all_outputs_for_expr` 打点，
统计贪心优化**真实评估的 I/O 形状分布**（window 提取 + 整电路评估的混合），据此校准分桶权重——这是比 46 个 benchmark 更准的推理分布依据。

---

## 五、V3.2 补充要求：只生成缺失的 5 个形状（2026-08-28 定稿）

> 背景：满量数据已交付且数量达标（3,981 expr / 48,110 电路 / 69.5 万行，输入分桶 24.6/25.6/23.7/26.1%、多输出 23.1%）。
> 用 `scripts/diag/_check_v3_data.py` 的 **Rust 形状覆盖视图**实测：46 个 benchmark 的 18 种 I/O 形状中
> **仅 5 个缺失**（其余 13 种全部已覆盖）。因此**只补充这 5 个形状**，其余不再生成，避免数量超发。

### 5.1 缺失形状与目标量（每组 10~15 变体，取中位 12）

| 形状 | Rust 电路 | 组数 | 电路（≈组×12） | 行/电路（2×N×M） | 行数 |
|---|---|---|---|---|---|
| 9入6出 | ADD4_OVF（tl_opt_smoke 实测） | ≥30 | 360 | 108 | 38,880 |
| 8入4出 | ENC8 | ≥30 | 360 | 64 | 23,040 |
| 7入4出 | ALU2 | ≥30 | 360 | 56 | 20,160 |
| 5入5出 | SHIFTER4 | ≥30 | 360 | 50 | 18,000 |
| 4入3出 | ENC4 | ≥30 | 360 | 24 | 8,640 |
| **合计** | | **≥150 组** | **~1,800 电路** | | **~108,720 行（~10.9 万）** |

### 5.2 沿用原有全部要求（与 V3.1 相同标准）

1. **单 corner** `s02p0_l01p0`（2ps slew / 1fF load），全 t=0、vector=1。
2. **组内功能等价**（真值表完全一致）+ **结构去重**（去重后仍 ≥10 变体）。
3. **全部组 ≥10 变体（10~15）**，**无退化组**（单变体/小组不要）。
4. **行数完整**：每电路恰 `2×N_in×M_out` 行（含全部 (pin,dir) 组合）。
5. **值级标准**（`scripts/diag/_check_v2_data.py` 全绿）：大小写一致（`X_*`）、`direction ∈ {rise,fall}`（无 `.sp` 后缀）、vector 切换位与 direction 一致、`ids_charge ≠ ids_avg`（4 位小数重收）、sc_expansion 可展开、transistor_wave 字段齐全。
6. **不与其他数据重复**：expr 编号与现有 **3,981 expr**（full 579 + io 389 + rest 3,013）**不重叠**（建议从 expr10000+ 起）；电路结构也不与现有变体重叠。

### 5.3 验收

- `scripts/diag/_check_v2_data.py` 对新增批次全绿（无 FAIL）；
- `scripts/diag/_check_v3_data.py` 的 Rust 形状覆盖视图 → 5 个形状全部 `OK`；
- 交付文件按 `batch_v2_m4/` 或并入 rest 的 part 拆分（GitHub 100MB 限制），`circuit_static` + `timing_arcs` 同 schema。
