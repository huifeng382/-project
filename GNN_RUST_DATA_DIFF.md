# GNN 模型数据要求 vs Rust 优化器数据 —— 差异记录

> 目的：把「GNN 排序模型能吃什么数据」和「Rust 贪心优化器（NetlistOpt）实际产出/使用什么数据」之间的差异，完整、可核查地记录下来，作为后续集成的依据。
>
> 创建：2026-08-15
> Rust 仓库：`TransiLog-share/NetlistOpt`（`tl_opt_smoke` / `optimize_tl_text`）

---

## 一句话结论

**GNN 模型和 Rust 优化器工作在两个不同的数据域上，不能直接对接。** 差异分两类：

- **软差异**（改生成脚本/补字段可解决）：网表格式、延迟口径、corner 条件、per-pin 特征、候选粒度。
- **硬差异**（必须选一边对齐，否则无法复用现有 checkpoint）：**cell 类型命名** + **电路 I/O 形状**。

---

## 一、两套数据各自的全貌

### 1.1 GNN 侧：训练/输入数据要求（来自 DATA_SPEC v9）

**电路形状**：铁律——输入**恰好 4 个**（`a,b,c,d`）、输出**恰好 1 个**（`out`）。

**网表格式**：扁平 `gate_level_netlist`，只有 `.SUBCKT DUT` + `X_N` 实例行，**没有晶体管级结构**：

```
.SUBCKT DUT a b c d out vdd gnd
X_2 d wire_2 SC_JOIN
X_3 wire_2 wire_3 SC_INV_WIRE
X_6 a wire_6 SC_JOIN
X_12 wire_3 wire_6 b c wire_12 SC_JOIN_OR_WIRE_AND_WIRE_AND_OR_WIRE_AND_WIRE_AND
X_13 wire_12 out SC_INV_WIRE
.ENDS DUT
```

**cell 类型**：638 类（来自训练数据 `cell_types_json`），例如 `SC_INV_WIRE`、`SC_JOIN`、`SC_JOIN_OR_WIRE_AND_WIRE_AND_OR_WIRE_AND_WIRE_AND`、`SC_AND` 等。模型用 `gate_idx`（cell 名）→ `Embedding(638, 32)` 做类别嵌入。

**延迟口径**：每 `(corner, switching_pin, direction, vector)` 一个端到端 `DELAY`（翻转 50% 到输出 50%）。

**corner 条件**：30 个（6 slew × 5 load）。

**输入特征**：per-pin slew/load/arrival_time、vector、gate_states_json、transistor_wave_json（wave 特征）、parasitic_caps_json、supply_noise_json。

**解析器硬编码**（`graph_builder.py:parse_netlist`）：
- 输出 pin **硬编码为 `out`**（`nodes['out'] = OUTPUT_PIN`）。
- 输入 pin 从 `.SUBCKT DUT` 提取，排除 `{vdd, gnd, vss, out}`。
- 只认 `X_` 开头的实例行（`gtype = tokens[-1]`），其它行（含嵌套 `.SUBCKT`、`M_` 晶体管行）全部跳过。

### 1.2 Rust 侧：`expr_to_hierarchical_spice` 产出 + 优化器使用

**电路形状**：任意 N 输入 / M 输出。`ADD4_OVF` = 9 输入（`a_0..a_3 b_0..b_3 cin`）/ 6 输出（`sum_0..3 cout ovf`）；简单 baseline 也是 4 入 1 出但输出名是 **`y`** 而非 `out`。

**网表格式**：层次化 SPICE，含嵌套 `.SUBCKT` 定义 + 晶体管体（`M_` 行，**有晶体管级结构**）：

```
* Corrected Hierarchical SPICE (True Deduplication)
.global vdd gnd

.SUBCKT SC_INV p0 out
M_1 out p0 vdd vdd pmos_lvt l=20n nfin=2
M_2 out p0 gnd gnd nmos_lvt l=20n nfin=2
.ENDS

.SUBCKT SC_JOIN_AND_WIRE_WIRE_AND_WIRE_WIRE p0 p1 out
* PUN (Pull-Up)
M_1 out p0 vdd vdd pmos_lvt l=20n nfin=2
M_2 out p1 vdd vdd pmos_lvt l=20n nfin=2
* PDN (Pull-Down)
M_3 out p0 n1 gnd nmos_lvt l=20n nfin=2
M_4 n1 p1 gnd gnd nmos_lvt l=20n nfin=2
.ENDS

.SUBCKT DUT a b c d y vdd gnd
X_17 a b wire_17 SC_JOIN_AND_WIRE_WIRE_AND_WIRE_WIRE
X_13 a wire_13 SC_INV
...
X_22 wire_21 y SC_INV
.ENDS DUT
```

**cell 类型**：`SC_INV`、`SC_JOIN`、`SC_JOIN_AND_AND`、`SC_JOIN_AND_WIRE_WIRE_AND_WIRE_WIRE`、`SC_JOIN_AND_AND_WIRE_WIRE_AND_AND_WIRE_WIRE` 等——**与训练数据是两套不同的 `semantic_name`**。

**延迟口径**：每电路一个 `avg_delay = (avg_rise + avg_fall) / 2` 再对多输出平均（`simulate_all_outputs_for_expr`）。无 corner/vector 概念，用 `SimuVector`（`pin_index` / `timing_sense` / `truth_table_idx`）。

**仿真条件**：固定模板（`asap7.sp`），无显式 slew/load corner。

**候选粒度**：大电路里抽出的一个 **window**（子电路），`TlWindowParams { max_x_nodes: 6, max_boundary_inputs: 8, max_boundary_outputs: 2, max_flatten_nodes: 120 }`。

---

## 二、差异总表

| 维度 | GNN 数据要求 | Rust 优化器实际用 | 差异性质 |
|---|---|---|---|
| 电路 I/O 形状 | 恰好 4 入 1 出（a,b,c,d→out） | 任意 N 入 M 出（ADD4_OVF=9入6出） | ❌ 结构性 |
| 网表格式 | 扁平，只有 `X_` 行 | 层次化，嵌套 `.SUBCKT` + 晶体管体 | ⚠️ parser 跳过嵌套，可缓解；但见下「输出 pin」 |
| 输出 pin 命名 | 硬编码 `out` | `y` / `sum_0..cout..ovf` | ❌ parse 会误判 |
| cell 类型命名 | 638 类（`SC_INV_WIRE` / `SC_JOIN_OR_WIRE_AND_WIRE_AND_...`） | `SC_INV` / `SC_JOIN_AND_AND` / `SC_JOIN_AND_WIRE_WIRE_...` | ❌ 两套签名，5/7 OOV |
| 延迟口径 | 每 (corner,pin,dir,vector) 端到端 DELAY | 每电路 avg_delay（rise/fall+多输出平均） | ❌ 粒度不同 |
| corner 条件 | 30 corner（6 slew×5 load） | 固定条件，无显式 slew/load | ❌ GNN 要 corner，Rust 没有 |
| 输入特征 | slew/load/arrival/vector/gate_states/wave/... | 只有网表 + 真仿真延迟 | ❌ 特征缺失 |
| 候选粒度 | 完整 4-pin 等价变体电路 | 大电路里的一个 window | ⚠️ 作用范围不同 |

---

## 三、逐条差异详述（含证据）

### 3.1 cell 类型命名（最关键，硬差异）

训练数据 638 类 vs Rust 当前 7 类的比对结果：

| Rust cell 类型 | 在训练集里？ |
|---|---|
| `SC_INV` | ✅ OK |
| `SC_JOIN` | ✅ OK |
| `SC_JOIN_AND_AND` | ❌ OOV |
| `SC_JOIN_AND_WIRE_WIRE_AND_WIRE_WIRE` | ❌ OOV |
| `SC_JOIN_AND_AND_WIRE_WIRE_AND_AND_WIRE_WIRE` | ❌ OOV |
| `SC_JOIN_AND_AND_AND_WIRE_WIRE_AND_AND_AND_WIRE_WIRE` | ❌ OOV |
| `SC_JOIN_OR_WIRE_AND_AND_AND_AND_WIRE_WIRE_WIRE_WIRE_WIRE_OR_WIRE_AND_AND_AND_AND_WIRE_WIRE_WIRE_WIRE_WIRE` | ❌ OOV |

**命名体系根本不同**：
- 训练用：`SC_JOIN_OR_WIRE_AND_WIRE_AND_OR_WIRE_AND_WIRE_AND`（`OR_WIRE` / `AND_WIRE` 交错）
- Rust 用：`SC_JOIN_AND_WIRE_WIRE_AND_WIRE_WIRE`、`SC_JOIN_AND_AND`（`AND_AND` / `WIRE_WIRE` 连续）

**根因**：这些名字是 `compute_signature`/`semantic_name` 对「某个具体晶体管结构（PUN/PDN 串并联排布）」做的确定性序列化。特点是：
1. 同一结构在不同版本代码里会得到不同字符串（已观察到两套）。
2. 名字本身不含任何电学量，本质是"这个 exact 结构"的哈希。
3. **OOV 是结构性的**：贪心优化每次探索新结构 → 生成新签名 → 新名字 → 对训练词表永远是 OOV。统一命名只治标不治本。

### 3.2 电路 I/O 形状（硬差异）

- GNN 的 `parse_netlist` 硬编码 `out` 为输出、且训练数据铁律"4 入 1 出"。
- Rust 的 `ADD4_OVF` 是 9 入 6 出，输出名 `sum_0..cout..ovf`；简单 baseline 输出名也是 `y`。
- 后果：即便 cell 名对上了，`parse_netlist` 也会把 `y`/`sum_0` 误当输入 pin、把输出硬指到不存在的 `out`。

### 3.3 网表格式（软差异，但有隐含硬点）

- GNN 期望扁平 `gate_level_netlist`；Rust 产出层次化 SPICE。
- `parse_netlist` 只认 `X_` 行，会**自动跳过**嵌套 `.SUBCKT` 定义和 `M_` 晶体管行——所以"层次化 vs 扁平"本身可缓解。
- 但隐含硬点在「输出 pin 命名」和「多输出」，见 3.2。

### 3.4 延迟口径（软差异）

- GNN：单 (corner,pin,dir,vector) 端到端 50% 延迟。
- Rust：每电路一个 `avg_delay`（rise/fall 平均 + 多输出平均）。
- 对接需统一：要么 Rust 产出 per-(pin,dir) 延迟，要么 GNN 改学"每电路 avg_delay"。

### 3.5 corner 条件（软差异）

- GNN 依赖 30 corner 的 slew/load。
- Rust 仿真模板（asap7.sp）里其实有固定 slew/load，只是没提取成显式特征。
- 对接需：固定一个 corner 喂 GNN，或从模板提取 slew/load。

### 3.6 输入特征（软差异）

- GNN 需要 slew/load/arrival/vector/gate_states/transistor_wave 等。
- Rust 目前只产出网表 + 真仿真延迟，这些特征都缺。
- 对接需在数据生成侧补齐（或先做"只给 corner + 网表"的极简版验证）。

---

## 四、结论与可选路径

### 核心结论

**cell 名当类别特征是死路**（OOV 永续），应改成：**从晶体管级网表提取稳定结构/电学特征 + 一个 ~10 类的固定逻辑类别**，GNN 和 Rust 共用这套提取逻辑，词表固定、永不 OOV。

### 结构特征方向：两套可提取的特征

| 取代 | 更通用的提取方式 | 来源 |
|---|---|---|
| 638 类 `gate_idx` 嵌入 | 逻辑函数（INV/NAND/NOR/AND/OR/AOI/OAI/XOR ≈10 类）+ 数值结构特征 | 晶体管网表（M_ 行）拓扑推导 |
| `p/g/h`（名字正则，对 SC_JOIN_* 基本失效） | 逻辑努力/寄生延迟从 PUN/PDN 串并联结构算 | 晶体管串并联拓扑 |
| `drive`（名字正则 `x\d+`） | 晶体管宽度 / nfin / 并联数 | `M_` 行的 `nfin=` / `l=` |
| （缺失） | 晶体管数、串联深度（stack height）、并联支路数 | 晶体管网表 |

### Rust 侧改动（两条路线）

- **路线 A：Rust 不改**。Python 解析 `.sp`，从 `.SUBCKT SC_*` 定义 + M_ 行提结构特征。Rust 只需继续出 `.sp`（已有）。
- **路线 B：Rust 当结构真值源**。新增函数从 `RecExpr<StructuralLogic>` 直接算每门结构特征，序列化 JSON 发给 serve.py。改动点在 `StructuralLogic`（Inv/Join/Bridge → 逻辑函数）和 `generate_subckt_definition`（生成 M_ 行处顺手算串并联结构/nfin）。

### 数据生成侧（改动更大，不在 Rust）

要重训"吃结构特征"的 GNN，训练数据得先有结构特征。现在 `gate_level_netlist` 只有 cell 名、无晶体管结构。数据生成器要改成：对每个 SC_ 门输出晶体管级结构（或直接输出结构特征）——这正是 `expr_to_hierarchical_spice` 已经会生成的东西。本质是让数据生成器和 Rust 用同一套结构来源。

---

## 五、待决问题

1. **方向验证**：先离线用现有 54 万数据（或其晶体管结构来源）+ 结构特征，验证"结构特征替代 gate_idx 嵌入"排序指标掉不掉（9.6 gate-merge 650→27 曾失败 24.55%→28.7%，必须实测）。
2. **I/O 是否仍锁 4-pin**：若锁，Rust 只能对能规约成 4-pin/1-out 的窗口用模型；若放开，模型输入侧（parse_netlist 硬编码 out、输入 pin 提取、DelayGNN 维度）都要改。
3. **cell 命名到底哪套**：确认训练数据 638 类 vs `expr_to_hierarchical_spice` 的 `semantic_name` 是否同一逻辑的不同版本，判断能不能做确定性映射（但结论上已倾向"放弃名字、用结构特征"）。
4. **延迟口径统一方向**：GNN 学 per-(pin,dir) 延迟，还是退到"每电路 avg_delay"。
