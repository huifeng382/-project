# 数据规格说明书

## 输出文件

生成6个Parquet文件（三批数据合并使用）：

```
# 第一批：手选电路全 sweep（已有，保留不变）
data/batch1/circuit_static.parquet
data/batch1/timing_arcs.parquet

# 第一批追加：手选电路全 sweep（新增，独立文件）
data/batch1b/circuit_static.parquet
data/batch1b/timing_arcs.parquet

# 第二批：e-graph 稀疏 sweep（已有，保留不变）
data/batch2/circuit_static.parquet
data/batch2/timing_arcs.parquet

# 第三批：e-graph 稀疏 sweep（新增）
data/batch3/circuit_static.parquet
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
| `gate_states_json` | str | `{"X_1":0,"X_2":1,"X_3":1}` | （可选）该vector下各门实例翻转状态，1=翻转，0=静态 |

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

要求（该字段为可选，不提供不影响基础训练）：
- key集合必须与网表中所有门实例名完全一致（不含输入/输出引脚）
- 若提供，每个vector行必须提供对应的gate_states_json，不得部分行有、部分行无
- 翻转状态通过SPICE仿真中的节点电压波形判定：输出电压摆幅超过VDD的20%即视为翻转

---

## 四、第一批电路要求（已有，约6万样本）

| 项目 | 规格 |
|------|------|
| 电路数量 | 已有150个 + **追加30-50个**（合计180-200） |
| 电路来源 | 手选。已有电路保留不变，新增电路与原有规格一致 |
| expr编号范围 | `expr0000` ~ `expr0149`，不得与第二批重叠 |
| 门类型覆盖 | AND、OR、NAND、NOR、INV、BUF、XOR等，每种门至少5个不同拓扑结构的电路 |
| 拓扑深度 | 浅（2-4级门）、中（5-8级）、深（9+级）各占约1/3 |
| 晶体管数范围 | 10-100 |
| 每电路corner数 | 30（6 slew × 5 load 全交叉） |
| 每corner组合 | 4个引脚 × 2个方向 × 2个vector = 16行 |
| 每电路总行数 | 30 × 16 = 480行 |
| 第一批总行数 | 约57,600-72,000行 |

## 五、第二批电路要求（已有，约4万样本）

| 项目 | 规格 |
|------|------|
| 电路数量 | 306个（已有，保留不变） |
| 电路来源 | TransiLog e-graph枚举 |
| expr编号范围 | `expr0200` ~ `expr0549` |
| 筛选要求 | 按结构特征去重 |
| 引脚要求 | 全部为4引脚（a,b,c,d） |
| 每电路corner数 | 9（3 slew × 3 load 全交叉） |
| 每corner组合 | 4个引脚 × 2个方向 × 2个vector = 16行 |
| 每电路总行数 | 9 × 16 = 144行 |
| 第二批总行数 | 约44,000行 |

## 六、第三批电路要求（新增，约3.5万样本）

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

### 三批合计：约 160,800-172,500 行 ≈ 16-17万样本

## 七、数据质量规则

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
13. `gate_states_json` 为**可选字段**。若提供，必须覆盖网表中所有门实例，不得遗漏。翻转判定阈值：输出摆幅 > VDD × 20%。不提供不影响基础训练

## 八、本次生成任务

> 已有 `data/batch1/` 和 `data/batch2/` 共4个文件保留不变。本次只生成以下4个新文件，格式与已有数据完全一致。

### 需要生成的文件

```
data/batch1b/circuit_static.parquet  # 新建
data/batch1b/timing_arcs.parquet     # 新建
data/batch3/circuit_static.parquet   # 新建
data/batch3/timing_arcs.parquet      # 新建
```

### 格式要求

batch1b 的列和格式**与 batch1 完全相同**（参见第二节静态数据、第三节动态数据）。
batch3 的列和格式**与 batch2 完全相同**（同上）。

重点确认：
- 列名、类型、单位与已有文件一致
- corner 命名遵循第三节 corner 命名规则（如 `s05p0_l10p0`）
- vector 编码遵循第三节 vector 编码规则（5位，第1位→a，取值0/1）
- 数据质量符合第七节全部13条规则
- circuit_id 格式 `candidate_{expr}_{idx}`，expr 不与已有数据重复

### 一批追加（batch1b，约1.5-2.4万样本）

| 项目 | 规格 |
|------|------|
| 电路数 | 30-50个 |
| 来源 | 手选，4引脚（a,b,c,d），与已有150个互补门类型和拓扑 |
| expr范围 | 避免与已有 batch1（expr0000~expr0149）、batch2（expr0200~expr0549）重复 |
| corner | 30个（slew 3/5/10/20/40/80 ps × load 0.2/0.5/1/3/10 fF 全交叉） |
| 每电路行数 | 4引脚 × 2方向 × 30corner × 2vector = 480行 |
| 输入引脚 | a, b, c, d（全部4个） |
| 字段 | 静态含 gate_level_netlist、cell_types_json、input_pins_json、output_pins_json、pin_loads_json；动态含全部 per-pin 列（slew_a~d、load_a~d、arrival_time_a~d）及 gate_states_json（可选） |

### 三批新建（batch3，约3.1-3.9万样本）

| 项目 | 规格 |
|------|------|
| 电路数 | 400-500个 |
| 来源 | TransiLog e-graph枚举，4引脚（a,b,c,d），与二批拓扑互补 |
| expr范围 | `expr1000` ~ `expr1499` |
| corner | 9个（slew 5/20/80 ps × load 0.2/1/10 fF 全交叉），完整列表见第三节 |
| 基础行数 | 4引脚 × 2方向 × 9corner × 1vector = 72行/电路 |
| 增强行数 | 额外5行：随机选5个corner各增加1个vector（优先 s80p0_l00p2、s80p0_l00p5） |
| 每电路总行数 | 72 + 5 = 77行 |
| 字段 | 与 batch2 完全一致（含全部 per-pin 列及 gate_states_json 可选） |

### 生成后验证

1. 静态文件和已有 batch1/batch2 的列完全一致
2. 动态文件和已有 batch1/batch2 的列完全一致
3. 无重复 circuit_id
4. 全部13条质量规则通过

---

## 九、验证顺序（已有数据+新数据合并后）

1. 已有数据（一批+二批）继续训练，确保第一批 corner 响应曲线和二批输入模式覆盖
2. 生成第三批数据，三批合并训练，验证拓扑多样性对泛化的提升
3. 训练代码需添加三批数据路径，其余逻辑不变

---

## 十、版本记录

### v3（当前版本）

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
