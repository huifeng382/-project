# 数据规格说明书 V2（Rust 对齐版）

> **V2 相对 V1 的三大变更**（对齐 Rust 贪心优化器 `NetlistOpt` 的实际使用方式）：
> 1. **I/O 形状**：输入从「恰好 4 个（a,b,c,d）」改为「任意 N 个（1~16）」；输出从「恰好 1 个（out）」改为「任意 M 个（1~6）」。对齐 Rust benchmark 的 I/O 范围。
> 2. **corner**：从「30 corner（6 slew × 5 load 全交叉）」改为「**单 corner（2ps slew / 1fF load）**」。对齐 Rust `asap7.sp` 模板的固定仿真条件。
> 3. **每组变体数**：从「2~6 个」改为「**10~15 个**」。对齐 Rust 贪心每 window 的候选数（4~12）。
>
> **数据总量保持 ~60 万行不变**（单 corner 省下的仿真预算，转成 ~42× 的电路数量）。

## 输出文件

生成 Parquet 文件（每行均须含 `transistor_wave_json`）。V2 改为单 corner + 任意 I/O + 每组 10-15 变体，批次按 e-graph 枚举分组（总量见第四节，~5 万电路 / ~60 万行），示例：

```
data/batch4/circuit_static.parquet    # e-graph 枚举批次
data/batch4/timing_arcs.parquet
data/batch5/circuit_static.parquet
data/batch5/timing_arcs.parquet
...
```

---

## 一、通用约束

所有电路必须满足：

- 输入引脚：**任意 N 个（1~16）**，命名对齐 Rust benchmark（如 `a_0`, `b_0`, `cin`），不再固定 `a,b,c,d`
- 输出引脚：**任意 M 个（1~6）**，命名对齐 Rust（如 `sum_0`, `cout`, `ovf`），不再固定 `out`
- 电源/地引脚：`vdd`, `gnd`（不出现在输入输出引脚列表中）
- 所有物理量统一使用国际单位制（SI）：
  - 延迟 `DELAY`：秒（s）
  - 负载电容 `pin_load_json` `output_load_f`：法拉（F）
  - slew `slew_s` `pin_slew_json`：秒（s）
- corner 标签中的数值用于标识测试条件，单位：slew（ps），load（fF）

---

## 二、静态数据 `circuit_static.parquet`

每行一个电路。列定义：

| 列名 | 类型 | 示例值 | 说明 |
|------|------|--------|------|
| `circuit_id` | str | `"candidate_expr0001_0005"` | 唯一标识，格式 `candidate_{expr}_{idx}` |
| `expr` | str | `"expr0001"` | 电路所属的实验批次编号 |
| `candidate_idx` | int | `5` | 批次内的序号 |
| `transistor_count` | int | `44` | 电路中的晶体管总数 |
| `gate_level_netlist` | str | 见下方 | SPICE子电路网表 |
| `cell_types_json` | str | `["SC_AND","SC_INV_WIRE"]` | JSON数组，网表中出现的所有门类型名 |
| `input_pins_json` | str | `["a_0","b_0","cin"]` | JSON数组，输入引脚名（任意 N 个，对齐 Rust INORDER） |
| `output_pins_json` | str | `["sum_0","cout"]` | JSON数组，输出引脚名（任意 M 个，对齐 Rust OUTORDER） |
| `pin_loads_json` | str | `{"a_0":4.5e-16,"b_0":3.2e-16,"out":0.0}` | JSON对象，每个引脚的负载电容（F） |
| `parasitic_caps_json` | str | `{"X_1":{"in_a":0.8,"in_b":0.82,"out":1.2},"X_2":{...}}` | **【必须，100%覆盖】** 每个门实例各输入/输出引脚的对地寄生电容（fF）。受完整性铁律约束。详见§六方案C |

### `gate_level_netlist` 格式

**层次化 SPICE**（对齐 Rust `expr_to_hierarchical_spice` 的输出），数据生成器直接用 `expr_to_hierarchical_spice` 产出。GNN 的 `parse_netlist` 只抽 `.SUBCKT DUT` 里的 `X_` 行、自动跳过嵌套 `.SUBCKT` 定义和晶体管 `M_` 行，所以层次化可直接喂。示例：

```
.SUBCKT DUT a_0 b_0 cin sum_0 cout vdd gnd
X_1 a_0 wire_1 SC_AND
X_2 b_0 wire_1 wire_2 SC_NAND
X_3 wire_2 cin sum_0 SC_NOR
X_4 wire_1 wire_2 cout SC_INV_WIRE
.ENDS DUT
```

规则：
- 第一行 `.SUBCKT DUT` 后跟引脚列表（任意输入...任意输出... vdd gnd，对齐 Rust INORDER/OUTORDER）
- 以 `.ENDS DUT` 结尾
- 每个门实例占一行，格式 `X_{序号} {输入网表名}... {输出网表名} {门类型}`
- 最后一个token是门类型名称（如SC_AND、SC_NOR、INVx1_ASAP7_75t_R等）
- 门类型名称需与 `cell_types_json` 中的一致

---

## 三、动态数据 `timing_arcs.parquet`

每行一个仿真样本。列定义：

| 列名 | 类型 | 示例值 | 说明 |
|------|------|--------|------|
| `circuit_id` | str | `"candidate_expr0001_0005"` | 对应静态数据中的电路ID |
| `corner` | str | `"s02p0_l01p0"` | 仿真corner标签（V2 单 corner：2ps/1fF），格式见下方 |
| `switching_pin` | str | `"a_0"` | 发生电平翻转的输入引脚，取值 = `input_pins_json` 中的某个引脚名 |
| `direction` | str | `"rise"` | 翻转方向，`rise`（0→1）或 `fall`（1→0）。若推理时无此列，可从 vector 中 switching_pin 对应位推导（0→rise, 1→fall） |
| `expr` | str | `"expr0001"` | 电路所属批次 |
| `candidate_idx` | int | `5` | 批次内序号 |
| `vector` | str | `"000"` | N位字符串，输入引脚逻辑值，格式见下方 |
| `slew_s` | float | `5.0e-12` | 切换引脚的输入slew（秒） |
| `output_load_f` | float | `1.0e-15` | 输出端负载电容（法拉） |
| `DELAY` | float | `3.304e-11` | 该 timing arc 的传播延迟（秒），从 switching_pin 翻转 50% 到 output 翻转 50%（per-switching_pin/direction/vector 的端到端延迟）。多输出电路为 per-output（每个输出一行） |
| `pin_slew_json` | str | `{"a_0":2.0e-12,"b_0":0.0}` | **【任意 I/O，JSON】** 每个输入引脚的输入 slew（秒）。切换引脚填实际值，非切换引脚填 0。key 集合 = `input_pins_json`。受完整性铁律约束 |
| `pin_load_json` | str | `{"a_0":4.5e-16,"b_0":3.2e-16}` | **【任意 I/O，JSON】** 每个输入引脚的负载电容（法拉）。key 集合 = `input_pins_json`。受完整性铁律约束 |
| `gate_states_json` | str | `{"X_1":0,"X_2":1,"X_3":1}` | **【必须，100%覆盖】** 该vector下各门实例翻转状态，1=翻转，0=静态（SPICE实测，推理时缺失可BFS推算）。受完整性铁律约束 |
| `transistor_wave_json` | str | `{"M1":{"gate":"X_2","ids_avg":12.3,...}}` | **【必须，100%覆盖】** 该仿真行每个晶体管的波形数据（gate/ids_avg/ids_peak/vds_swing）。受完整性铁律约束，所有批次每行必须非空。详见§六方案B |
| `supply_noise_json` | str | `{"vdd_droop_mV":12.3,"gnd_bounce_mV":5.1}` | **【必须，100%覆盖】** 该仿真行翻转窗口内的电源/地噪声。受完整性铁律约束，所有批次每行必须非空。详见§六方案D |
| `per_gate_timing_json` | str | — | **【已废弃，不需要生成】** 逐门过渡时间。实测表明逐门辅助监督对模型有害（见下方废弃说明），不再生成此列 |

> **I/O 列结构说明（方案 X：JSON）**：任意 I/O 下，per-pin 特征用 JSON 列（`pin_slew_json` / `pin_load_json`）而非固定列。原因：(1) JSON 天然变长，无需零填充；(2) 避免「填充 0 被当成真实非切换 pin」的静默坑——填充 pin 和真实非切换 pin 的 slew 都是 0，靠零值无法区分；(3) 与现有 JSON 为主的数据设计一致（gate_states / transistor_wave / pin_loads 均为 JSON）。若用固定列（如 slew_0..slew_15）填零，则 scaler 和数据加载必须严格按 `input_pins_json` 只吃真实 pin，否则会污染特征、损害排序指标。

### `per_gate_timing_json`（已废弃，不需要生成）

> **本字段不再需要，请勿生成。** 逐门 delay/slew 曾作为辅助监督信号，但多组实验实测证明：在共享 GNN encoder 上叠加逐门监督会**显著损害总延迟预测**（测试误差 +4~5pp）。根因是结构性的，与覆盖率无关：
> 1. 辅助 loss 量级远大于主 loss，劫持梯度，模型转去优化逐门量而非总延迟；
> 2. 「池化成总延迟（要混合节点）」与「逐门预测（要保留节点身份）」表征冲突；
> 3. 逐门延迟是总延迟的**冗余分解**，不是新信息。
>
> 补到 100% 覆盖也救不了——已用 100% 覆盖的 `out_slew` 变体验证，同样 +4pp。因此**跳过此字段**，节省仿真记录开销。注意：`gate_states_json`（门翻转状态，见下）**仍然需要**，它是模型的路径特征输入，用途不同。

### `corner` 命名规则

格式：`s{XX}p{Y}_l{ZZ}p{W}`

- `s` = slew条件，`{XX}`=整数部分，`p{Y}`=小数部分（p=小数点）
- `l` = load条件，同上
- 例：slew=2.0ps, load=1.0fF → `s02p0_l01p0`

**V2 统一使用单 corner（对齐 Rust `asap7.sp` 模板的固定仿真条件）：**

| 项目 | 值 | 来源 |
|---|---|---|
| slew | **2 ps** | Rust `VSTIM_RISE ... PWL(0 0 20p 0 22p {VDD} ...)` |
| load | **1 fF** | Rust `Cload_out __PIN_OUT__ 0 1f` |
| corner 标签 | **`s02p0_l01p0`** | — |
| 输入翻转 | **所有输入 t=0 同时翻转** | Rust `VSTIM_RISE ... PWL(0 0 ...)`（PWL 从 0 开始，无到达时间偏移） |

**所有批次统一使用上方单 corner（2ps / 1fF）。** 不再有 30 corner 全交叉规格。

### `vector` 编码规则

N 位字符串，每位表示一个输入引脚在仿真开始时的初始逻辑电平（N = 该电路的输入引脚数）：

- 第 i 位 → 第 i 个输入引脚（按 INORDER 顺序）

取值：`0` = 低电平(0V)，`1` = 高电平(VDD)

要求：
- switching_pin 对应的位必须与 direction 一致：direction=rise 时该位为 0，direction=fall 时该位为 1。
- 每个 (circuit, corner, switching_pin, direction) 组合下固定生成 **1 个 vector**（对齐 Rust `build_simu_vectors_for_simulation` 的 `break`：每个 (output, pin) 只取第一个能让输出翻转的 truth_table_idx）。
- 非切换引脚的电平由该 truth_table_idx 决定（Rust 取「第一个能翻转输出的真值表行」，非任意选）。

### `gate_states_json` 编码规则

JSON对象，key为网表中的门实例名（`X_1`, `X_2`, ...），value为翻转状态：

- `1`：该门在此vector下至少有一个输入发生翻转，输出信号正在传播
- `0`：该门所有输入保持静态，输出不变

示例（对应上方网表，vector="10100"，switching_pin="b"，direction="rise"）：
```
gate_states_json = {"X_1":0,"X_2":1,"X_3":1,"X_4":1}
```
含义：a=1, b=0, c=1, d=0。b从0翻到1（rise），信号经X_2→X_3/X_4传到out。X_1输入a保持1不变，未翻转。

要求：
- key集合必须与网表中所有门实例名完全一致（不含输入/输出引脚）
- 每个vector行必须提供对应的gate_states_json，不得部分行有、部分行无
- 翻转状态通过SPICE仿真中的节点电压波形判定：输出电压摆幅超过VDD的20%即视为翻转
- 推理时缺失可BFS推算

---

## 四、批次要求（V2：单 corner + 任意 I/O + 每组 10-15 变体）

> 相对 V1 的核心变化：单 corner（2ps/1fF）省下仿真预算 → 换成 ~42× 电路数量；I/O 任意；每组变体 10-15 个。

| 项目 | 规格 |
|------|------|
| 电路来源 | TransiLog e-graph 枚举（等价变体组） |
| I/O 形状 | 任意 N-in（1~16）/ M-out（1~6），对齐 Rust benchmark |
| corner 数 | **1**（2ps slew / 1fF load，标签 `s02p0_l01p0`） |
| 每输入引脚 | 2 个方向（rise / fall） |
| vector 数 | 1（对齐 Rust `break`：每个 (output, pin) 只取第一个能翻转输出的 truth_table_idx） |
| 每电路行数 | N_in × 2 dir × 1 vector × M output = 2 × N_in × M 行（N_in = 输入引脚数，M = 输出引脚数） |
| **每组变体数** | **10~15 个**（对齐 Rust 贪心候选数 4~12） |
| expr 编号 | 新 expr 不与 V1 的 569 个重叠 |
| 总量 | **~60 万行** |

**隐含数量**（按平均 N_in ≈ 6 估算，来自下方「I/O 等比例分桶」1~2/3~4/5~8/9~16 各 25%，加权平均 = (1.5+3.5+6.5+12.5)/4 = 6）：
- 每电路 ≈ 12 行（6 pin × 2 dir × 1 vector × M output，此处按 M=1 估算、未计多输出）。
- 电路数 ≈ 60万 / 12 ≈ **5 万**。
- 组数（expr）≈ 5万 / 12.5 ≈ **4000 组**。

> 对比 V1：1200 电路 → ~5万 电路（~42×）；569 expr → ~4000 expr（~7×）。核心收益 = 拓扑多样性暴涨，对齐 Rust 贪心排「任意 I/O 整电路、单 corner、10-15 候选」的场景。

> **聚合方式（对齐 Rust avg_delay）**：评估/排序时，对每个电路所有行（pin/dir/vector/output）的 DELAY 取**平均** = Rust 的 `avg_delay`。注意：V1 用的是「最坏情况（max）」，V2 改成「平均（mean）」——两者训练标签粒度相同（都是 per-pin/dir/vector 端到端延迟），只是聚合方式不同。

> **I/O 多样性要求（避免小 I/O 独大）**：e-graph 枚举天然偏小电路（输入少 → 功能简单 → 可枚举变体多），若不约束，2~4 输入的电路会占绝对多数、大 I/O 复杂电路被忽视——而大 I/O 恰是 Rust 贪心最需要模型帮忙的难例。因此**按输入数分桶等比例**，每桶电路数大致均衡，避免偏斜。建议分桶：输入 1~2 / 3~4 / 5~8 / 9~16 各约 25%；输出 1 / 2 / 3+ 三档，多输出（≥2 输出）占比不低于 ~20%。

### 电路质量要求

每个交付电路必须满足：

1. **功能等价（组内）**：同一 expr 的所有变体必须功能等价（真值表完全一致）。否则「排序择优」排的不是同一功能的不同实现，任务无意义。生成时对每个变体验证真值表一致。
2. **结构去重**：组内变体之间必须结构不同（门类型组成 / 连接 / 拓扑至少一处实质差异）。e-graph 枚举可能产出重复结构或「仅改名」的伪变体，需去重；去重后每组仍需满足 10-15 个变体。
3. **非退化过滤**：剔除退化电路——输出恒为常数（与输入无关）、悬空节点（未连到输出路径的门）、纯缓冲器链（无逻辑功能）等无意义实现。
4. **仿真可收敛**：每个电路必须能成功完成 SPICE 仿真、产出有效 DELAY。仿真失败（DC 不收敛 / 悬空 / 超时）的电路剔除，不计入交付。
5. **延迟有效**：DELAY 在 1e-12 ~ 1e-8 秒（沿用通用质量规则第 1 条），超出视为无效。
6. **跨组拓扑多样性**：不同 expr 覆盖不同门类型组成、拓扑深度、扇出分布（配合等比例分桶 + 结构去重），避免大量相似电路。

## 五、数据质量规则

### ⚠️ 完整性铁律（最重要，必读，历史踩过坑）

> 历史教训：曾出现「列存在、但内部值大量为空/null」的**假覆盖**——`per_gate_timing_json` 实际只填了 **60%**、`transistor_wave_json` 只填了 **28%**，导致这些字段在训练中完全不可用、白白浪费。**本次绝不允许再发生。**

凡本文档标注为「**必须**」的字段，交付时必须满足以下全部条件：

1. **列存在 ≠ 完成。** 每个「必须」字段，必须在**每一个适用行**都填入**有效非空值**：非 `null`、非空字符串 `""`、非空 JSON、非空 dict/list；数值型必须非 `NaN`、非 `None`，且符合各字段的取值约束（如延迟 > 0）。
2. **JSON 内部子字段同样受约束。** 例如 `xxx_json = {"X_1": {"delay_ps": ...}}`，则每个 key 下的**每个子字段**都必须有有效值。**严禁**出现 `{"X_1": {"delay_ps": null}}` 这种「外壳在、内部空」，也严禁 `{}` 空对象占位。
3. **严禁部分行有、部分行无。** 同一「必须」字段不得「一部分行填了、另一部分行留空」。
4. **必须随数据附覆盖率报告 `data/coverage_report.json`**（见本节末格式）。对每个「必须」字段报告：① 列非空行数/总行数；② JSON 内部每个子字段的**有效值行数/总行数**。**任何「必须」字段的任一层覆盖率 < 100%，即视为交付不合格。**
5. 若某字段确实无法做到 100% 覆盖，**必须在交付前主动说明，并将其在本规格中显式改为「可选」**，而不是交付一个「假装完整、实则残缺」的字段。

**`data/coverage_report.json` 格式示例：**
```json
{
  "batch1/timing_arcs": {
    "total_rows": 96000,
    "fields": {
      "transistor_wave_json.column_nonnull": "96000/96000 (100%)",
      "transistor_wave_json.ids_avg": "518400/518400 (100%)",
      "transistor_wave_json.ids_peak": "518400/518400 (100%)",
      "transistor_wave_json.vds_swing": "518400/518400 (100%)",
      "transistor_wave_json.ids_rise_time": "518400/518400 (100%)",
      "transistor_wave_json.vgs_swing": "518400/518400 (100%)",
      "transistor_wave_json.ids_charge": "518400/518400 (100%)"
    }
  },
  "batch1/circuit_static": {
    "total_rows": 200,
    "fields": {
      "parasitic_caps_json.column_nonnull": "200/200 (100%)",
      "parasitic_caps_json.total_gate_keys": "matched to netlist"
    }
  },
  "batch1/timing_arcs": {
    "total_rows": 96000,
    "fields": {
      "supply_noise_json.column_nonnull": "96000/96000 (100%)",
      "supply_noise_json.vdd_droop_mV": "96000/96000 (100%)",
      "supply_noise_json.gnd_bounce_mV": "96000/96000 (100%)"
    }
  },
  "sc_expansion": {
    "coverage": "N/N (100%)"
  }
}
```

### 通用质量规则

1. `DELAY` 值范围：1e-12 < DELAY < 1e-8（超出此范围的视为物理不可行数据，剔除）
2. `slew_s` 不得为0或NaN
3. `output_load_f` 不得为0或NaN
4. `pin_slew_json` 必须覆盖 `input_pins_json` 中所有输入引脚，每个值非 NaN。切换引脚的值等于 `slew_s`，非切换引脚填 0.0
5. `pin_load_json` 必须覆盖 `input_pins_json` 中所有输入引脚，每个值非 NaN。值与 `pin_loads_json`（静态列）一致即可
6. 同一 `(circuit_id, corner, switching_pin, direction, vector)` 组合不得出现重复行
7. `cell_types_json` 中的门类型名称与网表中的门类型名称完全一致
8. `input_pins_json` 为任意 N 个输入引脚名（对齐 Rust INORDER）
9. `pin_loads_json` 必须包含所有输入引脚 + 所有输出引脚的负载值
10. `slew_s` 和 `output_load_f` 是 SPICE 仿真测得的**实际值**，corner 标签中的 S/L 是设定的**测试条件**，两者可能不同。不要用 corner 条件值直接填充实测值列
11. `gate_states_json` **【必须，100%覆盖】** 必须覆盖网表中所有门实例，不得遗漏。翻转判定阈值：输出摆幅 > VDD × 20%。受完整性铁律约束
12. ~~`per_gate_timing_json` 必须覆盖网表中所有门实例~~ **【已废弃，不需要生成，见第三节废弃说明】**
13. `parasitic_caps_json` **【必须，100%覆盖】** 必须覆盖网表中所有门实例（key 集合一致）。每个门必须含 `in_*` 和 `out` 字段。受完整性铁律约束，详见§六方案C
14. `supply_noise_json` **【必须，100%覆盖】** 每一行必须包含 `vdd_droop_mV` 和 `gnd_bounce_mV` 两个字段，值 ≥ 0。受完整性铁律约束，详见§六方案D
15. `sc_expansion.json` **【必须，100%覆盖】** 必须覆盖训练数据 `cell_types_json` 中出现的所有 SC_ cell 名，且每个条目的 `subcircuit` 非 null、能展开为 ASAP7 单元。覆盖率报告加一项 `sc_expansion.coverage`。否则 STRUCT_MODE 提不到结构特征、回退到默认值。
16. **命名一致**：`sc_expansion.json` 的 key 命名必须与训练数据 `cell_types_json` / `gate_level_netlist` 中的 cell 名**完全同一套**（不能用两套命名）。这是规则 15 成立的前提——命名不一致会导致「查不到展开」。

## 六、高级物理数据（突破排序瓶颈）

> **V2 训练策略（重要）**：以下详细物理特征在**数据生成时全部照常生成**（零额外仿真成本——SPICE 已算过，只需后处理写出），但**训练时按需选用**，因为 Rust 推理时拿不到（或依赖仿真）。**train-only 字段**：`transistor_wave_json`（需仿真）、`supply_noise_json`（需仿真）、`parasitic_caps_json`（需寄生提取）、`pin_load_json`/`pin_loads_json`（Rust 不建模输入负载）。首版先试「无 wave」模型看排序指标，不够再蒸馏。**数据生成全要，免得以后重生成。**

> **当前模型排序指标**：Spearman ≈ 0.21，选择遗憾 ≈ 2.6%，top1 ≈ 42%。
> **核心瓶颈**（信噪比诊断实证）：成对分辨 <2% 真实延迟差 = **52%（= 随机）**。
> 根因 = 模型预测噪声 RMS ≈ 17ps >> <2% 信号 ≈ 0.48ps（信噪比 0.11）。
> 降预测噪声的唯一途径 = **新信息 + 更大量/更多样的数据**。本节两项均为降噪杠杆。

### 方案 A：标准单元库 LIB 查找表（已提供，可叠加）

> LIB 为门级延迟提供物理约束表面，降低模型对每门延迟的自由度 → 降预测噪声。
> **数据已就位**：`data/std_cells.lib`（93 cells）+ `data/sc_expansion.json`（3868/3868 宏可展开）。
> 代码已集成（11.0, train_lib），待 2D-grid 加速后启用。**当前优先级低于方案 B。**

**LIB 表内容说明：**

对每种门，LIB 文件含一个 7×7 的二维表：

- 行（索引轴1）：输入 slew（ps），7 个点，典型范围 1~500ps
- 列（索引轴2）：输出负载电容（fF），7 个点，典型范围 0.5~50fF
- 单元格值：该条件下门的传播延迟（ps）

现有数据中 27 种门类型（归一化后），每种都需要对应的 LIB 表条目。对于非标准门（SC_JOIN、SC_BRIDGE、WIRE 类），不需要 LIB 表——模型仍从 GNN 预测。

**格式要求：**

- 文件命名：`std_cells.lib`
- 放在 `data/` 根目录下，与各 batch 目录并列
- Liberty 标准格式，含 `cell()` 条目和 `timing()` 表
- 至少覆盖 INV, NAND, NOR, AND, OR, BUF, XOR 七种基础门类型

**关键前置条件：SC_ 宏展开为标准单元**

网表中的 SC_ 门类型（如 `SC_AND`、`SC_JOIN`）是 TransiLog 合成工具生成的宏单元，**每个 SC_ 宏内部由多个 ASAP7 标准单元互联组成**。LIB 表只能用于标准单元，必须先展开 SC_ 宏才能使用。

**必须提供 SC_ 宏展开表 `data/sc_expansion.json`：**

```json
{
  "SC_AND": {
    "subcircuit": [
      {"inst": "X_A1", "cell": "NAND2x2_ASAP7_75t_R", "inputs": ["A", "B"], "output": "wire_nand"},
      {"inst": "X_A2", "cell": "INVx1_ASAP7_75t_R", "inputs": ["wire_nand"], "output": "Y"}
    ]
  },
  "SC_INV": {
    "subcircuit": [
      {"inst": "X_I1", "cell": "INVx1_ASAP7_75t_R", "inputs": ["A"], "output": "Y"}
    ]
  },
  "SC_JOIN": {
    "subcircuit": [
      {"inst": "X_J1", "cell": "BUFx1_ASAP7_75t_R", "inputs": ["A"], "output": "Y"}
    ]
  }
}
```

格式规则：
- **key**：网表中出现的每个 SC_ 门类型名（来自 `cell_types_json`），每个都必须有对应条目
- **`subcircuit`**：该 SC_ 宏的内部标准单元列表，按从左到右（输入到输出）排列
- **`inst`**：内部实例名，全局唯一（已在前缀中编码了宏名，不同宏之间不会冲突）
- **`cell`**：LIB 文件中的标准单元名（`cell()` 条目名），取值必须是 `std_cells.lib` 中存在的条目
- **`inputs`**：该内部实例的输入网表名列表，可以是宏的输入引脚（A/B/C/D）或前级内部实例的输出
- **`output`**：该内部实例的输出网表名，宏的最后一个内部实例的 output 即为宏的对外输出 Y

**展开后效果：**

原始网表：
```
X_1 a wire_1 SC_AND
X_2 wire_1 out SC_INV
```

展开后网表（SC_AND 展开为 NAND2x2 + INVx1）：
```
X_1_A1 a b wire_nand1 NAND2x2_ASAP7_75t_R
X_1_A2 wire_nand1 wire_1 INVx1_ASAP7_75t_R
X_2_I1 wire_1 out INVx1_ASAP7_75t_R
```

展开后所有 SC_ 宏被替换，网表 100% 为标准单元，LIB 表 100% 可用，`sc_to_asap7.json` 映射表不再需要。

---

### 方案 B：晶体管波形数据 **【当前最高优先级——降预测噪声的核心杠杆】**

> **目的**：模型预测噪声 RMS ≈ 17ps >> <2% 变体差异信号 ≈ 0.48ps（信噪比 0.11）→ 成对分辨 <2% = 52%（随机）。晶体管波形（每管电流/电压）是**唯一**能直接提供器件级物理信息、降低每样本预测误差的信号。**信噪比诊断估算：若全部样本加上晶体管波形，<2% 成对分辨可从 52% 提升至 70-75%。**
>
> **零额外仿真成本**：SPICE 在跑瞬态仿真时已经算出每个器件的 I_ds、V_ds 波形，目前只是没写进输出。只需在后处理阶段把每个晶体管在翻转窗口内的 avg/peak/swing 提取写出即可，**无需任何额外仿真**。
>
> **历史教训（必须避免）**：上一版此方案只在「高/低负载 2 个 corner」采波形，导致实际只有 28% 覆盖、且集中在低 slew（s03/s05）——恰好避开了最需要的高 slew 难 corner，数据基本无用。**本版要求所有批次、每一行都 100% 包含此字段，受第七节完整性铁律约束。**

**`transistor_wave_json` 是一项新的标准列（必须，100% 覆盖所有批次每一行）：**

不再独立为 `data/batch_wave/`。**所有批次（batch1/1b/2/3）在重新生成时，每一行的 `timing_arcs.parquet` 都必须包含此列。**

| 列名 | 类型 | 说�� |
|------|------|------|
| `transistor_wave_json` | str | 该仿真行**每个晶体管**的波形数据。JSON 对象。**每行必须非空** |

**内部结构**：key = SPICE 网表中的晶体管实例名，value = 含以下**全部**字段的对象（每个都必须有效非空）：

| 子字段 | 类型 | 含义 | 单位 | 测量标准 |
|------|------|------|------|------|
| `gate` | str | 该晶体管所属的门级实例名（对应 `gate_level_netlist` 中的 `X_N`），供模型做「晶体管→门」聚合 | — | 网表映射 |
| `ids_avg` | float | 翻转期间平均漏极电流 | μA | 翻转窗口内 \|I_ds\| 的时间平均 |
| `ids_peak` | float | 翻转期间峰值漏极电流 | μA | 翻转窗口内 \|I_ds\| 的最大值 |
| `vds_swing` | float | 翻转期间漏-源电压摆幅 | V | V_ds 最大值 − 最小值 |
| `ids_rise_time` | float | 漏极电流 10%→90% 上升时间 | ps | 翻转窗口内 \|I_ds\| 从峰值的 10% 升至 90% 的时间。越短延迟越小 |
| `vgs_swing` | float | 栅-源电压摆幅 | V | V_gs 最大值 − 最小值。越低开不彻底，延迟越大 |
| `ids_charge` | float | 翻转窗口内漏极电流积分（总电荷） | fC | ∫ \|I_ds\| dt，直接决定延迟的物理量 |

示例：
```json
{"M1": {"gate": "X_2", "ids_avg": 12.3, "ids_peak": 25.1, "vds_swing": 0.72,
        "ids_rise_time": 8.5, "vgs_swing": 0.68, "ids_charge": 45.2},
 "M2": {"gate": "X_2", "ids_avg": 8.7,  "ids_peak": 18.4, "vds_swing": 0.68}}
```

**覆盖率要求（受第七节「完整性铁律」约束，强制 100%）：**
1. **所有批次的每一行**都必须有非空 `transistor_wave_json`——**不允许「只在部分 corner 采样」或「只在单独 batch 提供」**。
2. 每行的 JSON 必须包含该电路**全部晶体管实例**（key 集合 = 该电路 SPICE 网表中所有晶体管），不得遗漏。
3. 每个晶体管的 7 个子字段（`gate`/`ids_avg`/`ids_peak`/`vds_swing`/`ids_rise_time`/`vgs_swing`/`ids_charge`）都必须有效（数值非 NaN；`ids_*` ≥ 0；`vds_swing` ≥ 0；`vgs_swing` ≥ 0；`ids_rise_time` ≥ 0；`ids_charge` ≥ 0；`gate` 为有效 `X_N`）。
4. 交付时在 `data/coverage_report.json` 中报告各批次列非空率、以及 7 个子字段各自的内部非空率，**均须 100%**。

---

### 方案 C：每节点寄生电容 **【推荐——零成本，门级负载直达】**

> **目的**：SPICE 的寄生提取（parasitic extraction）已算出每个门的输入/输出引脚对地电容（fF）。这些是电路结构的固有属性、不依赖 switching scenario。**当前 `pin_loads_json` 只有顶层引脚 a/b/c/d/out 的负载，无法反映内部节点看到的真实 RC 环境。** 寄生电容补上后，GNN 能直接从节点特征看到「这个门驱动多大电容」→ 延迟预测更准。这是**静态特征**——每个电路只写一次、零仿真成本。

**`parasitic_caps_json` 加入 `circuit_static.parquet`（必须，100% 覆盖所有电路）：**

| 列名 | 类型 | 说明 |
|------|------|------|
| `parasitic_caps_json` | str | 每个门实例的对地寄生电容（fF）。JSON 对象，每个电路一行 |

**内部结构**：key = 网表中的门实例名（`X_1`, `X_2`, ...），key 集合必须与 `gate_level_netlist` 中的门实例名完全一致。value 含：

| 子字段 | 类型 | 含义 | 单位 |
|------|------|------|------|
| `in_{pin}` | float | 该门指定输入引脚的对地寄生电容（来自 SPICE 寄生提取），如 `in_a`、`in_b` | fF |
| `out` | float | 该门输出引脚的对地寄生电容 | fF |

**输入引脚名的确定**：按该门在网表中的 `inputs` 顺序，依次记为 a, b, c...（与宏的输入引脚命名一致）。例如 NAND2 有两输入 → `in_a`、`in_b`。**所有值必须 > 0**；SPICE 中不可能出现真正的零寄生电容。

示例（对应网表）：
```json
{"X_1": {"in_a": 0.82, "in_b": 0.85, "out": 1.21},
 "X_2": {"in_a": 0.78, "in_b": 0.81, "out": 1.15},
 "X_3": {"in_a": 0.79, "in_b": 0.83, "out": 1.18},
 "X_4": {"in_a": 0.61, "out": 0.95}}
```

**覆盖率要求（受第七节「完整性铁律」约束，强制 100%）：**
1. 所有电路必须包含此列，列非空 100%。
2. 每电路的 JSON key 集合 = 该电路网表中所有门实例名，不得遗漏。
3. 每门的所有输入引脚 + 输出引脚子字段都必须有效（非 NaN，> 0）。
4. 零额外仿真成本——SPICE 寄生提取已算过，只差写出。

---

### 方案 D：电源/地噪声 **【推荐——极低成本，解释不可建模误差】**

> **目的**：多门同时翻转时 VDD/GND 节点会有瞬态电压波动（droop/bounce），**直接拖慢共享同一电源轨的所有门的延迟**。这恰好是模型当前 <2% 分辨失败的部分原因——不可建模的电源噪声被误当成「预测随机误差」。加入这两个标量，模型就能学会「高噪声 → 延迟偏高」的模式，降剩余预测噪声。
>
> **成本**：SPICE 在跑瞬态仿真时已算出 VDD 和 GND 节点的完整电压波形。只需在后处理中提取翻转窗口内的 VDD 最小值与 GND 最大值即可——单次扫描、极低成本。

**`supply_noise_json` 加入 `timing_arcs.parquet`（必须，100% 覆盖所有批次每一行）：**

| 列名 | 类型 | 说明 |
|------|------|------|
| `supply_noise_json` | str | 该仿真行翻转窗口内的电源/地噪声。JSON 对象，每行一个 |

**内部结构**（两个标量字段，每个都必须有效）：

| 子字段 | 类型 | 含义 | 单位 | 测量标准 |
|------|------|------|------|------|
| `vdd_droop_mV` | float | 翻转窗口内 VDD 节点相对标称值的最大下掉幅度 | mV | nom_VDD − min(V_dd(t))，t ∈ 翻转窗口。≥ 0 |
| `gnd_bounce_mV` | float | 翻转窗口内 GND 节点相对零电位的最大上弹幅度 | mV | max(V_gnd(t)) − 0，t ∈ 翻转窗口。≥ 0 |

**翻转窗口**定义：从 switching_pin 信号越过 50% VDD 开始，到 output 信号越过 50% VDD 结束（与 DELAY 的测量窗口一致）。

示例：
```json
{"vdd_droop_mV": 12.3, "gnd_bounce_mV": 5.1}
```

**覆盖率要求（受第七节「完整性铁律」约束，强制 100%）：**
1. 所有批次每一行都必须包含此列，列非空 100%。
2. 两个子字段都必须有效（非 NaN，≥ 0）。
3. 如果某行恰巧 VDD/GND 无波动（比如只有极少数门翻转），允许值 = 0.0——但这仍然是一个有效值，不是空。

---

## 七、交付计划（单批 ~60 万行）

> V2 改为单批 ~60 万行（单 corner + 任意 I/O + 每组 10-15 变体），批次与数量见第四节。V1 的「两阶段（60 万升级 + 120 万新增）」已废弃。

### 通用验证

- `gate_states_json`：必须，100% 覆盖（`per_gate_timing_json` 已废弃）。
- `transistor_wave_json`（方案 B）、`parasitic_caps_json`（方案 C）、`supply_noise_json`（方案 D）：所有批次每行 100% 覆盖，受完整性铁律约束。
- 方案 A（LIB 查表）数据已提供，优先度低于方案 B/C/D。
- 任何批次交付必须附 `coverage_report.json`。

### ⚠️ 强烈建议：分步交付与预检（非强制，但强烈推荐——零额外成本、可大幅避免返工）

> **动机**：历史教训——上一版 60 万行数据生成后才发现 `per_gate_timing_json` 内部 60% 空、`transistor_wave_json` 实际 28% 覆盖，**全部返工**。本次新增 3 个 JSON 字段（transistor_wave / parasitic_caps / supply_noise），各有子字段约束，出错率较高。以下步骤可**与大规模生成并行、提前暴露规格理解偏差**，无需任何额外仿真。

**步骤 1：小批量预检（先跑 2-3 个电路，验证格式 100% 合规再放量）**

生成方在启动 60 万行全量前，**先完成 2-3 个电路（建议涵盖不同门类型和拓扑深度）的单 corner（2ps/1fF）仿真 + 全部列输出**，并附 `data/coverage_report.json`。接收方立即逐字段核对：

- 列级：`transistor_wave_json`、`parasitic_caps_json`、`supply_noise_json` 是否所有行非空？
- 子字段级：每个 JSON 内部的 key 集合是否齐全？子字段是否全部非 NaN？`gate` 是否映射到正确的 `X_N`？
- 值级：`ids_*` 是否 ≥0？`parasitic_caps` 是否 >0？`vdd_droop_mV`/`gnd_bounce_mV` 是否 ≥0？

**2-3 个电路的预检通过后，再启动全量生成。** 这解决了「60 万行全错 → 全返工」的灾难。

> 预检步骤**不是强制要求**，但强烈推荐——唯一「成本」是生成方多交 2-3 个预检电路的覆盖率报告，这笔时间远小于返工的代价。

---

