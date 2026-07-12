# 数据规格说明书

## 输出文件

生成8个Parquet文件：

```
# 第一批：手选电路全 sweep
data/batch1/circuit_static.parquet    # 150个电路
data/batch1/timing_arcs.parquet

# 第一批追加：手选电路全 sweep
data/batch1b/circuit_static.parquet   # 50个电路
data/batch1b/timing_arcs.parquet

# 第二批：e-graph 稀疏 sweep
data/batch2/circuit_static.parquet    # 325个电路
data/batch2/timing_arcs.parquet

# 第三批：e-graph 稀疏 sweep
data/batch3/circuit_static.parquet    # 480个电路
data/batch3/timing_arcs.parquet
```

---

## 一、通用约束

所有电路必须满足：

- 输入引脚：**恰好4个**，命名为 `a`, `b`, `c`, `d`
- 输出引脚：**恰好1个**，命名为 `out`
- 电源/地引脚：`vdd`, `gnd`（不出现在输入输出引脚列表中）
- 所有物理量统一使用国际单位制（SI）：
  - 延迟 `DELAY`：秒（s）
  - 负载电容 `load_*` `output_load_f`：法拉（F）
  - slew `slew_*`：秒（s）
  - 到达时间 `arrival_time_*`：秒（s）
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
| `input_pins_json` | str | `["a","b","c","d"]` | JSON数组，输入引脚名（统一为a,b,c,d） |
| `output_pins_json` | str | `["out"]` | JSON数组，输出引脚名 |
| `pin_loads_json` | str | `{"a":4.5e-16,"b":3.2e-16,"c":6.1e-16,"d":5.0e-16,"out":0.0}` | JSON对象，每个引脚的负载电容（F） |

### `gate_level_netlist` 格式

SPICE子电路格式，设备行为从网表推断。示例：

```
.SUBCKT DUT a b c d out vdd gnd
X_1 a wire_1 SC_AND
X_2 b wire_1 wire_2 SC_NAND
X_3 wire_2 c out SC_NOR
X_4 wire_1 wire_2 SC_INV_WIRE
.ENDS DUT
```

规则：
- 第一行 `.SUBCKT DUT` 后跟引脚列表（a b c d out vdd gnd）
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
| `corner` | str | `"s05p0_l01p0"` | 仿真corner标签，格式见下方 |
| `switching_pin` | str | `"a"` | 发生电平翻转的输入引脚，取值a/b/c/d。若推理时无此列，可从 slew_a~d 中非零引脚推导 |
| `direction` | str | `"rise"` | 翻转方向，`rise`（0→1）或 `fall`（1→0）。若推理时无此列，可从 vector 中 switching_pin 对应位推导（0→rise, 1→fall） |
| `expr` | str | `"expr0001"` | 电路所属批次 |
| `candidate_idx` | int | `5` | 批次内序号 |
| `vector` | str | `"00101"` | 5位字符串，输入引脚逻辑值，格式见下方 |
| `slew_s` | float | `5.0e-12` | 切换引脚的输入slew（秒） |
| `output_load_f` | float | `1.0e-15` | 输出端负载电容（法拉） |
| `DELAY` | float | `3.304e-11` | 该timing arc的传播延迟（秒），从 switching_pin 翻转50%到 output 翻转50% |
| `slew_a` | float | `5.0e-12` | 引脚a的输入slew（秒）。非切换引脚填0.0 |
| `slew_b` | float | `0.0` | 引脚b的输入slew（秒） |
| `slew_c` | float | `0.0` | 引脚c的输入slew（秒） |
| `slew_d` | float | `0.0` | 引脚d的输入slew（秒） |
| `load_a` | float | `4.5e-16` | 引脚a的负载电容（法拉） |
| `load_b` | float | `3.2e-16` | 引脚b的负载电容（法拉） |
| `load_c` | float | `6.1e-16` | 引脚c的负载电容（法拉） |
| `load_d` | float | `5.0e-16` | 引脚d的负载电容（法拉） |
| `arrival_time_a` | float | `0.0` | 引脚a信号到达时间（秒）。若为最早到达的引脚则填0.0 |
| `arrival_time_b` | float | `5.0e-12` | 引脚b信号到达时间（秒）。相对于最早到达引脚的偏移量 |
| `arrival_time_c` | float | `0.0` | 引脚c信号到达时间（秒） |
| `arrival_time_d` | float | `8.0e-12` | 引脚d信号到达时间（秒） |
| `gate_states_json` | str | `{"X_1":0,"X_2":1,"X_3":1}` | 该vector下各门实例翻转状态，1=翻转，0=静态（SPICE实测，推理时缺失可BFS推算） |
| `per_gate_timing_json` | str | — | **【已废弃，不需要生成】** 逐门过渡时间。实测表明逐门辅助监督对模型有害（见下方废弃说明），不再生成此列 |

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
- 例：slew=5.0ps, load=1.0fF → `s05p0_l01p0`
- 例：slew=0.2ps, load=10.0fF → `s00p2_l10p0`

**第一批使用以下30个corner（6 slew × 5 load 全交叉）：**

slew取值(ps)：3, 5, 10, 20, 40, 80
load取值(fF)：0.2, 0.5, 1, 3, 10

| | l00p2 | l00p5 | l01p0 | l03p0 | l10p0 |
|---|---|---|---|---|---|
| s03p0 | s03p0_l00p2 | s03p0_l00p5 | s03p0_l01p0 | s03p0_l03p0 | s03p0_l10p0 |
| s05p0 | s05p0_l00p2 | s05p0_l00p5 | s05p0_l01p0 | s05p0_l03p0 | s05p0_l10p0 |
| s10p0 | s10p0_l00p2 | s10p0_l00p5 | s10p0_l01p0 | s10p0_l03p0 | s10p0_l10p0 |
| s20p0 | s20p0_l00p2 | s20p0_l00p5 | s20p0_l01p0 | s20p0_l03p0 | s20p0_l10p0 |
| s40p0 | s40p0_l00p2 | s40p0_l00p5 | s40p0_l01p0 | s40p0_l03p0 | s40p0_l10p0 |
| s80p0 | s80p0_l00p2 | s80p0_l00p5 | s80p0_l01p0 | s80p0_l03p0 | s80p0_l10p0 |

**第二批使用以下9个corner（3 slew × 3 load 全交叉）：**

slew取值(ps)：5, 20, 80
load取值(fF)：0.2, 1, 10

| | l00p2 | l01p0 | l10p0 |
|---|---|---|---|
| s05p0 | s05p0_l00p2 | s05p0_l01p0 | s05p0_l10p0 |
| s20p0 | s20p0_l00p2 | s20p0_l01p0 | s20p0_l10p0 |
| s80p0 | s80p0_l00p2 | s80p0_l01p0 | s80p0_l10p0 |

### `vector` 编码规则

5位字符串 `"abcde"`，每位表示一个输入引脚在仿真开始时的初始逻辑电平：

- 第1位 → 引脚a
- 第2位 → 引脚b
- 第3位 → 引脚c
- 第4位 → 引脚d
- 第5位 → 保留（填0）

取值：`0` = 低电平(0V)，`1` = 高电平(VDD)

示例：`vector="00110"` 表示 a=0, b=0, c=1, d=1（第5位为0）

要求：
- switching_pin对应的位必须与direction一致：direction=rise时该位为0（从0翻到1），direction=fall时该位为1（从1翻到0）
- 每个(circuit, corner, switching_pin, direction)组合下固定生成**2个vector**
- 这2个vector的切换引脚位相同（由direction决定），但**非切换引脚位的组合必须不同**，以覆盖不同输入模式对路径选择的影响
- 示例（switching_pin="b", direction="rise"）：
  - vector1: `"00000"`（a=0, b=0, c=0, d=0）—所有非切换引脚为0
  - vector2: `"10100"`（a=1, b=0, c=1, d=0）—非切换引脚取不同值

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

## 四、第一批电路要求（约6万样本）

| 项目 | 规格 |
|------|------|
| 电路数量 | 180-200个 |
| 电路来源 | 手选 |
| expr编号范围 | `expr0000` ~ `expr0149`，不得与第二批重叠 |
| 门类型覆盖 | AND、OR、NAND、NOR、INV、BUF、XOR等，每种门至少5个不同拓扑结构的电路 |
| 拓扑深度 | 浅（2-4级门）、中（5-8级）、深（9+级）各占约1/3 |
| 晶体管数范围 | 10-100 |
| 每电路corner数 | 30（6 slew × 5 load 全交叉） |
| 每corner组合 | 4个引脚 × 2个方向 × 2个vector = 16行 |
| 每电路总行数 | 30 × 16 = 480行 |
| 第一批总行数 | 约57,600-72,000行 |

## 五、第二批电路要求（约4万样本）

| 项目 | 规格 |
|------|------|
| 电路数量 | 325个 |
| 电路来源 | TransiLog e-graph枚举 |
| expr编号范围 | `expr0200` ~ `expr0549` |
| 筛选要求 | 按结构特征去重 |
| 引脚要求 | 全部为4引脚（a,b,c,d） |
| 每电路corner数 | 9（3 slew × 3 load 全交叉） |
| 每corner组合 | 4个引脚 × 2个方向 × 2个vector = 16行 |
| 每电路总行数 | 9 × 16 = 144行 |
| 第二批总行数 | 约44,000行 |

## 六、第三批电路要求（约3.5万样本）

| 项目 | 规格 |
|------|------|
| 电路数量 | 400-500个 |
| 电路来源 | TransiLog e-graph枚举 |
| expr编号范围 | `expr1000` ~ `expr1499`，不得与前两批重叠 |
| 筛选要求 | 按结构特征去重（门类型组成+深度+扇出分布），与第二批的拓扑互补，避免相似 |
| 引脚要求 | 全部为4引脚（a,b,c,d） |
| 每电路corner数 | 9（3 slew × 3 load 全交叉） |
| 每corner组合（基础） | 4个引脚 × 2个方向 × 1个vector = 8行 |
| 每corner组合（增强） | 随机抽取**5个corner**使用2个vector = 额外5行 |
| 每电路总行数 | 9×8 + 5 = **77行** |
| 第三批总行数 | 约30,800-38,500行 |
| 增强corner选择 | 优先极端 corner（s80p0_l00p2、s80p0_l00p5），这些 corner 模型误差最大 |

### 三批合计：约 164,000-181,300 行 ≈ 17万样本

## 七、数据质量规则

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
  "batch_wave/timing_arcs": {
    "total_rows": 9600,
    "fields": {
      "transistor_wave_json.column_nonnull": "9600/9600 (100%)",
      "transistor_wave_json.ids_avg": "518400/518400 (100%)",
      "transistor_wave_json.ids_peak": "518400/518400 (100%)",
      "transistor_wave_json.vds_swing": "518400/518400 (100%)"
    }
  }
}
```

### 通用质量规则

1. `DELAY` 值范围：1e-12 < DELAY < 1e-8（超出此范围的视为物理不可行数据，剔除）
2. `slew_s` 不得为0或NaN
3. `output_load_f` 不得为0或NaN
4. `slew_a/b/c/d` 必须全部非NaN。切换引脚的值等于 `slew_s`，非切换引脚填 0.0
5. `load_a/b/c/d` 必须全部非NaN。值与 `pin_loads_json`（静态列）一致即可，但仍需逐行填入
6. `arrival_time_a/b/c/d` 必须全部非NaN。最早到达的引脚填 0.0，其余引脚填入相对偏移（秒）。不同引脚应有不同的 arrival time 值，不要全部填 0
7. 同一 `(circuit_id, corner, switching_pin, direction, vector)` 组合不得出现重复行
8. `arrival_time_*` 不得全为同一常数。不同电路、不同 corner、不同 vector 应有不同值
9. `cell_types_json` 中的门类型名称与网表中的门类型名称完全一致
10. `input_pins_json` 统一为 `["a","b","c","d"]`
11. `pin_loads_json` 必须包含 a, b, c, d, out 五个引脚的负载值
12. `slew_s` 和 `output_load_f` 是 SPICE 仿真测得的**实际值**，corner 标签中的 S/L 是设定的**测试条件**，两者可能不同。不要用 corner 条件值直接填充实测值列
13. `gate_states_json` 必须覆盖网表中所有门实例，不得遗漏。翻转判定阈值：输出摆幅 > VDD × 20%
14. ~~`per_gate_timing_json` 必须覆盖网表中所有门实例~~ **【已废弃，不需要生成，见第三节废弃说明】**

## 八、高级物理数据（突破 15% 误差瓶颈）

> 当前最优模型误差 ~24.5%。以下两项任选其一可将误差降至 5-15%，**优先使用方案 A**。

### 方案 A：标准单元库 LIB 查找表（优先）

> 如果生成电路时使用了某个 PDK（如 ASAP7），该 PDK 自带标准单元的 .lib 时序库文件。提供该文件即可，无需额外 SPICE 仿真。

**需要提供的文件：**

一个 `.lib` 文件（Liberty 格式），内含标准单元（INV, NAND2, NOR2, AND2, OR2, BUF 等）的时序查找表。

**LIB 表内容说明：**

对每种门，LIB 文件含一个 7×7 的二维表：

- 行（索引轴1）：输入 slew（ps），7 个点，典型范围 1~500ps
- 列（索引轴2）：输出负载电容（fF），7 个点，典型范围 0.5~50fF
- 单元格值：该条件下门的传播延迟（ps）

现有数据中 27 种门类型（归一化后），每种都需要对应的 LIB 表条目。对于非标准门（SC_JOIN、SC_BRIDGE、WIRE 类），不需要 LIB 表——模型仍从 GNN 预测。

**使用方式：**

训练时 GNN 预测每个门看到的"输入 slew"和"输出负载"，然后从 LIB 表双线性插值查出门延迟。所有门延迟求和 = 总 DELAY。

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

**如果无法提供 SC_ 展开表，使用下方方案 B。**

---

### 方案 B：晶体管波形数据（当前重点方向）

> **目的**：模型在**高 slew 极端 corner（s40p0 / s80p0，尤其 l00p2/l00p5）误差最大（42~46%）**，这是门级抽象无法捕捉的晶体管级非线性。晶体管波形（电流/电压）是**唯一**能刻画这一非线性的信号。
>
> **历史教训（必须避免）**：上一版此方案只在「高/低负载 2 个 corner」采波形（原文「其他 corner 不需要波形」），导致实际只有 **28% 覆盖，且集中在低 slew（s03/s05）——恰好避开了最需要的高 slew 难 corner**，数据基本无用。**本版要求全覆盖，且必须覆盖 s40/s80 难 corner。**

**电路数量：20 个新增电路**（独立生成，放 `data/batch_wave/`，不改现有 968 电路）。字段结构与主数据完全一致，**额外**增加 `transistor_wave_json` 一列。

**电路选择要求：**
- 覆盖全部归一化门类型，每种门至少出现 2-3 次
- 同时覆盖简单电路（4 门）与复杂电路（7+ 门）
- 若能增到 40-60 个电路更好（20 个电路多样性偏低）；但每个电路必须满足下方全覆盖要求

**Corner 与行数（全 sweep，不得裁剪）：**
- **30 corner 全交叉**（6 slew × 5 load，与第一批完全相同，见第三节 corner 表）
- **必须包含 s40p0_*、s80p0_* 全部高 slew corner**——这是本方案的核心价值所在，绝不能只采低 slew
- 每 corner：4 引脚 × 2 方向 × 2 vector = 16 行
- 每电路：30 × 16 = 480 行；20 电路共 9,600 行

**新增列 `transistor_wave_json`（必须，100% 覆盖）：**

| 列名 | 类型 | 说明 |
|------|------|------|
| `transistor_wave_json` | str | 该仿真行**每个晶体管**的波形数据。JSON 对象 |

**`transistor_wave_json` 内部结构**：key 为 SPICE 网表中的晶体管实例名，value 为对象，含以下**全部**字段（每个都必须有效非空）：

| 子字段 | 类型 | 含义 | 单位 | 测量标准 |
|------|------|------|------|------|
| `gate` | str | 该晶体管所属的门级实例名（对应 `gate_level_netlist` 中的 `X_N`），供模型做「晶体管→门」聚合 | — | 网表映射 |
| `ids_avg` | float | 翻转期间平均漏极电流 | μA | 翻转窗口内 \|I_ds\| 的时间平均 |
| `ids_peak` | float | 翻转期间峰值漏极电流 | μA | 翻转窗口内 \|I_ds\| 的最大值 |
| `vds_swing` | float | 翻转期间漏-源电压摆幅 | V | V_ds 最大值 − 最小值 |

示例：
```json
{"M1": {"gate": "X_2", "ids_avg": 12.3, "ids_peak": 25.1, "vds_swing": 0.72},
 "M2": {"gate": "X_2", "ids_avg": 8.7,  "ids_peak": 18.4, "vds_swing": 0.68}}
```

**覆盖率要求（受第七节「完整性铁律」约束，强制 100%）：**
1. `data/batch_wave/` 的**每一行**（全部 9,600 行）都必须有非空 `transistor_wave_json`——**不允许「只在部分 corner 采样」**。
2. 每行的 JSON 必须包含该电路**全部晶体管实例**（key 集合 = 该电路 SPICE 网表中所有晶体管），不得遗漏。
3. 每个晶体管的 4 个子字段（`gate`/`ids_avg`/`ids_peak`/`vds_swing`）都必须有效（数值非 NaN；`ids_*` ≥ 0；`vds_swing` ≥ 0；`gate` 为有效 `X_N`）。
4. 交付时在 `data/coverage_report.json` 中报告：列非空率、以及 ids_avg/ids_peak/vds_swing/gate 各自的内部非空率，**均须 100%**。
5. **零额外仿真成本说明**：这些量 SPICE 在跑瞬态仿真时已经算出（每个器件的 I_ds、V_ds 波形本就在结果里），只是之前没写进输出。只需在后处理阶段把每个晶体管在翻转窗口内的 avg/peak/swing 提取写出即可，**无需任何额外仿真**。所以全 corner 全覆盖是可行且低成本的。

---

## 九、验证顺序

1. 全部数据生成，包含 `gate_states_json` 字段（**`per_gate_timing_json` 不再需要**，见第三节废弃说明）
2. **方案 A（LIB 查表）数据已提供并集成**：`data/std_cells.lib` + `data/sc_expansion.json` 已就位，模型已能展开 SC_ 宏并查表
3. **方案 B（晶体管波形）是当前重点数据需求**：按上方要求生成 `data/batch_wave/`，**全 corner 全覆盖、必须含 s40/s80 难 corner、每行 100% 有 `transistor_wave_json`**
4. 两方案**不互斥、可叠加**（LIB 提供门级物理先验，晶体管波形提供极端 corner 的底层非线性）
5. 交付任何批次数据，都必须附 `data/coverage_report.json`（第七节完整性铁律）

---

## 九、版本记录

### v7（当前版本）

| 项目 | v6 | v7 | 原因 |
|------|------|------|------|
| `per_gate_timing_json` | 必须（全门 100%） | **废弃，不需要生成** | 实测逐门辅助监督有害（+4~5pp），结构性问题，补到 100% 也救不了（100% out_slew 已验证同样差） |
| 完整性铁律 | 无 | **新增（第七节置顶）** | 历史假覆盖：per_gate 只 60%、wave 只 28%，字段白废；强制 100% 有效值 + 覆盖率报告 |
| 方案 B 晶体管波形 | 20 电路、仅 2 个 corner 采样（28% 覆盖、集中低 slew） | **全 30 corner 全覆盖、必须含 s40/s80 难 corner、每行 100%、加 `gate` 映射字段** | 旧规格自身导致稀疏且避开难 corner，数据无用；本方案是攻克极端 corner 的重点 |
| 方案优先级 | A/B 互斥、优先 A | **A（已提供并集成）+ B 可叠加，B 为当前重点数据需求** | A 的 LIB+展开表已就位；B 全覆盖后才有意义 |

### v6

| 项目 | v5 | v6 | 原因 |
|------|------|------|------|
| SC_ 映射 | 一对一映射表 | **SC_ 宏展开表** | SC_ 是组合门，一对一对不上；展开为标准单元后 LIB 全覆盖 |

### v5

| 项目 | v4 | v5 | 原因 |
|------|------|------|------|
| 方案 A | — | **新增** LIB 查找表 | 查表替代模型预测 |
| 方案 B | — | **新增** 晶体管波形 | 底层电学数据 |

### v4

| 项目 | v3 | v4 | 原因 |
|------|------|------|------|
| gate_states_json | 可选 | **必须** | SPICE实测，精度提高 |
| 动态字段 | — | **新增** per_gate_timing_json | 门级延迟分解信号，辅助训练 |
| 数据生成 | 增量追加 | **全部重新生成** | 结构简化，所有字段统一规格 |

### v3

| 项目 | v2 | v3 | 原因 |
|------|------|------|------|
| 一批 | 120-150电路 | 150已有(batch1) + 30-50追加(batch1b) | 强化corner物理学习 |
| 二批 | 600-700电路 | 保留已有306电路 | 已有数据输入模式覆盖好 |
| 三批 | 无 | **新增** 400-500电路 | 补充拓扑多样性 |
| 三批向量 | — | 基础1个 + 5个极端corner用2个 | 关键corner不缺输入模式 |
| 文件数 | 4个 | 8个 | 新增batch1b + batch3 |
| 总样本数 | ≈10万 | ≈17万 | 四批合并 |

### v2

| 项目 | 初版 | v2 |
|------|------|------|
| 二批电路数 | 300-350 | 600-700 |
| 二批每corner向量数 | 2 | 1 |
| 二批每电路行数 | 144 | 72 |
| 文件目录 | batch1_30k/batch2_70k | batch1/batch2 |

### v1（初版）

| 项目 | 规格 |
|------|------|
| 一批 | 150电路(已有batch1) + 30-50追加(batch1b)，480行/电路 |
| 二批 | 300-350电路，144行/电路 |
| 向量 | 每条件 2 个 |
| 总样本 | ≈10万 |
