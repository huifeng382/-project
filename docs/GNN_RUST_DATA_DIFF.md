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

## 五、各维度解决方案

> 每个维度：改哪里 / 怎么改 / 是否要重训 / 是否要改 Rust。

### 5.1 电路 I/O 形状（+ 输出 pin 命名）

- **短期（不改模型、不重训）**：只对 Rust 里「≤4 输入且恰好 1 输出」的 window 调 GNN。窗口边界由 `TlWindowParams` 控制，再加一层过滤即可。
- **代码改点**：`parse_netlist` 把 `input_pins`/`output_pins` 改成从调用方显式传入（读 Rust JSON 的 `input_pins`/`output_pins`），删掉 `.SUBCKT DUT` 猜测和 `out` 硬编码。输出名 `y`/`sum_0` 就不再是问题。
- **长期（放开 4-pin）**：`DelayGNN` 的「固定 4 pin per-pin 特征」改成「变长 pin 逐 pin 编码后池化」，DATA_SPEC 铁律放宽 → 必须重训。
- **是否重训**：锁 4-pin = 否；放开 = 是。**是否改 Rust**：窗口过滤小改。

### 5.2 网表格式

- **基本不用改，反而利用层次化**。`parse_netlist` 现在就是 `if not stripped.startswith('X_'): continue`，会自动跳过嵌套 `.SUBCKT` 和 `M_` 行，X_ 行正常抽出。
- 层次化网表里的 `.SUBCKT SC_*` 定义 + `M_` 行是维度 5.3「结构特征」的原料，**不要扁平化**，让 parser 在遇到 `.SUBCKT SC_*` 时额外解析 M_ 行提结构特征。
- **是否重训**：否。**是否改 Rust**：否。

### 5.3 cell 类型命名（最关键）

**放弃 cell 名当类别，改成「固定逻辑类别 + 晶体管级结构特征」。** 三处改：

1. `graph_builder.parse_netlist`：解析 X_ 行时不再把 cell 名直接当 `gate_idx`，而是找到该 SC_ 门对应的 `.SUBCKT` 定义 → 解析 `M_` 行 → 算结构特征；逻辑类别用固定规则归到 ~10 类。
2. `model.DelayGNN`：`Embedding(638, 32)` 改成 `Embedding(~10, 16)`，结构特征拼进 node features。
3. 数据生成侧（重训必需）：对每个 SC_ 门输出结构特征字段，替代 `cell_types_json` 名字。

**固定特征清单**（稳定、通用、永不 OOV）：
```
logic_type ∈ {INV, NAND, NOR, AND, OR, AOI, OAI, XOR, BUF, COMPLEX}
num_inputs, n_transistors, stack_height(串联深度), parallel_width(并联支路数), nfin(驱动)
```

**是否重训**：是。**是否改 Rust**：路线 A 不改（Python 解析 .sp）；路线 B 改（Rust 从 `StructuralLogic` 出特征 JSON）。

### 5.4 延迟口径

**统一到「每电路最坏情况延迟」**（GNN 评估本来就是这个口径）。

- GNN 侧：评估已是「每变体取 max over (pin/dir/vector) 最坏延迟」，直接用。
- Rust 侧：`simulate_all_outputs_for_expr` 里已有每个 output 的 `avg_rise`/`avg_fall`，把「多输出平均」改成「多输出取最坏」（或保持平均，只要两边一致）。
- 推理时 GNN 在固定 corner 下预测每个 (pin,dir) 延迟取 max 作为排序分数。

**是否重训**：用最坏口径 = 否。**是否改 Rust**：`avg_delay` 改成取 max（小改）。

### 5.5 corner 条件

**提取 Rust 仿真实际 slew/load，固定成一个 corner 喂 GNN。**

- 查 `asap7.sp` 模板的输入 slew 和负载设置 → 映射到 GNN 30 corner 里最接近的一个（如 `s05p0_l01p0`）。
- serve.py 推理时写死这个 corner，不做 corner 泛化。

**是否重训**：否。**是否改 Rust**：否（只需把 asap7.sp 的 slew/load 值告诉 Python 侧）。

### 5.6 输入特征

**分阶段降级。**

- **阶段 1（验证，不改 Rust、不重训）**：只给「网表 + 固定 corner + 默认特征」。slew/load=固定值、arrival=0、vector=默认、gate_states 用 `logic_sim.py` BFS 推算、transistor_wave/parasitic_caps 缺省（设 0 或去维度）。跑通看排序掉多少。
- **阶段 2（Rust 补齐）**：gate_states 用 Rust truth_table 算；vector 从 `SimuVector.truth_table_idx` 映射；slew/load 从 asap7.sp 提取；transistor_wave 在 Rust 真仿真时顺手提取。

**是否重训**：阶段 1 够用 = 否；不够 = 是。**是否改 Rust**：阶段 1 否，阶段 2 是。

### 汇总

| 维度 | 是否重训 | 是否改 Rust | 优先级 |
|---|---|---|---|
| 5.2 网表格式 | 否 | 否 | 无需处理 |
| 5.5 corner | 否 | 否 | 低 |
| 5.4 延迟口径 | 否（最坏口径） | 小改 | 中 |
| 5.6 输入特征 | 阶段1否/阶段2是 | 阶段1否 | 中 |
| 5.1 I/O 形状 | 锁4pin否/放开是 | 窗口过滤小改 | 中 |
| 5.3 cell 命名 | **是** | 路线A否/路线B是 | **最高** |

**核心结论**：5.2/5.4/5.5 不阻塞；5.1/5.6 短期可绕过（锁 4-pin + 极简特征）；**5.3（cell 命名）是唯一必须动模型 + 重训的硬骨头**。

验证顺序：先做 5.2/5.4/5.5/5.6-阶段1 的「极简接线」，在不重训前提下把 Rust 候选喂现有 GNN（cell 名先用「名字→~10 类逻辑类别」粗映射代替），看排序是否有意义；若粗映射就不行，再上 5.3 的结构特征重训。

---

## 六、待决问题（更新）

1. ~~方向验证~~ ✅ **已完成**：结构特征方向成立，见第七节（rich 2.85%、logic 3.64% 均优于旧 638 名嵌入 5.67%）。
2. **I/O 是否仍锁 4-pin**：若锁，Rust 只能对能规约成 4-pin/1-out 的窗口用模型；若放开，模型输入侧（parse_netlist 硬编码 out、输入 pin 提取、DelayGNN 维度）都要改。
3. ~~cell 命名到底哪套~~ ✅ **已解决**：放弃名字，用固定 10 类逻辑 + 结构特征（STRUCT_MODE），任意名字（含 Rust）都不 OOV，见第七节。
4. **延迟口径统一方向**：GNN 学 per-(pin,dir) 延迟，还是退到"每电路 avg_delay"。

---

## 七、问题3（cell 命名 OOV）解决记录 + 结构特征实验

### 7.1 落地实现（14.2.2~14.2.4）

- `graph_builder.py` 新增 `gate_struct(name)`：优先用 `sc_expansion.json` 的 ASAP7 展开算 `(logic_type, n_transistors, drive, stack, parallel)`，查不到回退名字关键字（JOIN/BRIDGE/WIRE → COMPLEX 等）。
- `rebuild_gate_types` 从「动态 638 类名」改成「固定 10 类逻辑 + 保留类」。
- `build_static_graph` 的 `gate_idx` 改用逻辑类别；按 `STRUCT_MODE` 决定拼哪些结构特征列。
- `config.STRUCT_MODE` 四模式：`base`（10逻辑+n_t）/ `logic_only`（只10逻辑）/ `rich`（+stack+parallel）/ `elec`（p/g/drive 从 ASAP7 算）。
- 缓存 key 加 `STRUCT_MODE`（防止脏缓存）。

### 7.2 OOV 彻底解决（已验证）

Rust 侧实际出现的 7 个 cell 名全部映射成功，无一 OOV：

| Rust cell 名 | 逻辑类别 | idx | n_t 来源 |
|---|---|---|---|
| SC_INV | INV | 0 | sc_expansion |
| SC_JOIN | INV | 0 | sc_expansion |
| SC_JOIN_AND_AND | COMPLEX | 9 | 回退（默认6.0） |
| SC_JOIN_AND_WIRE_WIRE_AND_WIRE_WIRE | COMPLEX | 9 | sc_expansion（10） |
| 其余复杂 SC_JOIN_* | COMPLEX | 9 | 回退 |

用这些名字构造的网表跑 `build_static_graph`：gate_idx 全部合法（0-9 逻辑 / 10 INPUT_PIN / 11 OUTPUT_PIN），**无越界、无崩溃、无 UNKNOWN_GATE**。词表固定 13 类，Rust 生成再多新结构也不 OOV。

### 7.3 结构特征实验（单 seed=42，spread>10% 遗憾）

| 变体 | 遗憾 | Spearman | top1 | 捕获 | recall@2 A | 停点 |
|---|---|---|---|---|---|---|
| 旧 rank(42)（638名嵌入） | 5.67% | 0.699 | 74.2% | 88.5% | — | ~300 |
| structbase（10逻辑+n_t） | 6.05% | 0.646 | 74.8% | 83.1% | 82.0% | **160** ⚠️ |
| structlogic（只10逻辑） | 3.64% | 0.692 | 75.7% | 90.9% | **89.2%** | 298 |
| structrich（+stack+parallel） | **2.85%** 🏆 | 0.619 | 73.5% | 86.8% | 85.8% | 311 |
| structelec（p/g/drive修） | 3.78% | 0.529 | 71.1% | 89.0% | 86.2% | 377 |

### 7.4 结论

> ⚠️ **本节 7.3/7.4 为单 seed=42 的历史记录，结论已被 7.6 多 seed 确认推翻（2026-08-20）：真正最优是 structlogic（logic_only），不是 structrich。**

1. **方向成立**：3/4 变体遗憾优于旧 638 名嵌入（5.67%）。rich=2.85%、logic=3.64%、elec=3.78%。
2. **干净的逻辑分类本身有信号**：`structlogic`（纯 10 逻辑，无结构特征）3.64% > 旧 5.67%。9.6 的「650→27 任意聚类」失败是因为聚的是任意类；「INV/NAND/NOR/AND/OR/…」这套有物理意义的分类比任意名字哈希更 informative。
3. ~~**rich 最好（2.85%）**：stack+parallel 有真实增益。~~ ❌ 单 seed 噪声（见 7.6）。
4. **structbase 的 6.05% 被早停污染**：160 epoch 触发 plateau（过拟合）早停，比其它三个少跑一半。是 `n_t` 48% 真实 / 52% 默认 6.0 造成噪声的早过拟合信号，不能当「n_t 有害」的干净证据。

### 7.5 待办

- ✅ **多 seed 确认**（已完成，2026-08-20，见 7.6）：structlogic 两 seed 全面胜出 → 默认 `STRUCT_MODE='logic_only'`。
- **OOV 名结构精度**：52% 回退默认 n_t=6.0，需补 sc_expansion 覆盖或从 Rust .sp 直接解析晶体管结构（路线 A 完整版）。——随 structrich 降级为低优先（logic_only 不依赖 n_t）。

### 7.6 多 seed 确认（2026-08-20）：结论反转，structlogic 胜出

**seed 2468/456 两批（hi_spread 口径，SPLIT_SEED=42 同切分）：**

| run | 遗憾 | Spearman | top1 | 捕获率 | recall@2 A | 停点 |
|---|---|---|---|---|---|---|
| structrich2468 | 3.72% | 0.652 | 76.3% | 87.8% | 86.8% | 264 |
| structrich456 | 3.36% | 0.586 | 71.2% | 89.4% | 87.3% | 579 |
| **structlogic2468** | **1.72%** | 0.657 | 78.0% | 89.7% | **92.6%** | 193 |
| **structlogic456** | **1.78%** | 0.686 | **83.0%** | 89.0% | 90.0% | 327 |

**3-seed 集成（seed 42+2468+456，等权平均）：**

| 组合 | 遗憾 | Spearman | top1 | 捕获率 | recall@2 A |
|---|---|---|---|---|---|
| 3-seed（42+2468+456） | 1.88% | **0.694** | 77.5% | 91.6% | 89.0% |
| top-2（2468+456） | **1.76%** | 0.642 | **80.3%** | 89.7% | **91.5%** |

**结论**：
1. **structlogic 是最优 cell 策略**：两 seed 遗憾 1.72/1.78% 全面优于 structrich（3.36/3.72%）。7.3 的「rich 最好」是 seed 42 噪声。结构特征（n_t/stack/parallel）在 logic_only 之上**有害**（噪声/过拟合），不是「更多特征更好」。
2. **structlogic 单 seed 已超越历史最强单 seed（cornerattn 2468 = 1.93%）**，recall@2 A 90~92.6% 甚至高于 cornerattn top-3 集成（88.1%）。
3. **集成遗憾未破 cornerattn top-3 的 1.48%**（3-seed=1.88%、top-2=1.76%），但 Spearman/top1 相当或更好。cornerattn top-3 仍是最低遗憾交付基线；structlogic（logic_only）为 V2 重训默认。
4. **seed42 是拖累项**：加入集成 Spearman +0.05、regret +0.12pp、recall@2 A −2.5pp——差 seed 拖累，与 14.1 一致。后续 structlogic 集成建议从强 seed（2468/456/1357 系）起步。
5. **<2% 成对分辨 58%** 仍接近随机——SNR 天花板未破，突破靠 V2 数据（wave 全覆盖）。

---

## 八、贪心评估全局性（修正）+ 电气条件 + 收敛问题清单

### 8.1 修正：贪心是全局择优，不是 window 孤立仿真

> 之前「候选粒度 = window 子电路」的表述不准确，本节修正。

- window 只是**局部搜索单元**（在哪生成 rewrite）；**候选评估的对象是「代回 rewrite 的整个电路」**。
- 代码证据（`tl_opt.rs`）：
  - `evaluate()` 用 `module.to_recexpr()` 取**整个电路** expr → `simulate_all_outputs_for_expr` 仿真整个电路。
  - `avg_delay` = 对所有输出平均 = 全局延迟。
  - 接受条件 = `global_delta = global_score - current.combined_score`（全局 score 改善）。
- 因此 GNN 应预测**整个电路的全局延迟**，而非孤立 window。这与 GNN 训练数据（整电路）反而更对齐。

### 8.2 Rust 仿真的电气条件是固定的（asap7.sp 硬编码）

- 输入 slew = **2ps**（`VSTIM_RISE ... PWL(0 0 20p 0 22p {VDD} 1n {VDD})`）。
- 输出 load = **1fF**（`Cload_out __PIN_OUT__ 0 1f`）。
- ~~所有输入 **t=0 同时翻转**~~〔2026-09-03 修正标注：tb_*.sp 实测为**仅 switching pin 翻转**（接 VSTIM_RISE/FALL），其余输入按 vector 固定 vdd/gnd 常量；翻转沿在 20-22ps 非 t=0，见 §12.2d〕、单 vector、非切换输入接 vdd/gnd。
- **vector 数 = 1（不是 2）**：`design.rs:build_simu_vectors_for_simulation` 里对每个 `(output, pin)` 用 `break` 只取第一个能让输出翻转的 truth_table_idx，不覆盖多路径。`direction`（rise/fall）= 2。所以每电路行数 = N_in × 2 dir × 1 vector × M output。
- 对应 GNN 一个固定 corner（≈s03p0_l01p0；2ps 比 GNN 最小 slew corner 3ps 还低）。

### 8.3 收敛后的问题清单（vs GNN）

| # | 问题 | GNN 侧 | Rust 侧 | 性质 |
|---|---|---|---|---|
| 1 | I/O 形状 | 恰好 4入1出 | 任意 N入M出（1~16入 / 1~6出） | 🔴 硬阻塞 |
| 2 | 延迟口径 | per-(pin,dir) 端到端 DELAY（评估取最坏） | avg_delay=(rise+fall)/2 多输出平均 | 🟡 中 |
| 3 | 电气条件 | 30 corner + per-pin arrival/vector | 固定 2ps slew / 1fF load / 单 vector | 🟡 中 |

> 已消除：cell 命名（✅ 13 类逻辑）、候选粒度（✅ 澄清：贪心评估全局）、输出 pin 命名（并入 #1）。网表格式非阻塞（parser 跳过嵌套）。

### 8.4 问题 2/3 的潜在解决方案

**#2 延迟口径**：
- 方案 A（推荐）：统一到「最坏情况延迟」。GNN 评估已用最坏（max over pin/dir/vector）；Rust 把 `avg_delay` 改成 max over outputs 的 max(rise,fall)。只改 Rust 一行聚合，GNN 不重训。
- 方案 B：GNN 重训成「每电路 avg_delay」口径，保持 Rust 不动。丢 per-pin 信息 + 重训成本。

**#3 电气条件**：
- 方案 A（推荐）：GNN 固定到 Rust 的单一 corner。Rust 固定 2ps/1fF → GNN 补一个 s02p0_l01p0（或 s03p0 近似），per-pin 特征固定值合成（slew=2ps 切换/0 其余、load=固定、arrival=0、vector=默认）。Rust 不改。
- 方案 B：Rust 补齐多 corner + per-pin arrival/vector（改仿真模板 + SimuVector，成本高）。
- 方案 C：GNN 降级成无-corner 模型（丢 corner 信息，8.7 教训风险高）。

---

## 九、GNN 代码侧修改计划（4 项，spec 已定、代码待改）

> 前提：`DATA_SPEC_V2` 已把 I/O（任意 N/M + JSON 列）、corner（单）、延迟口径（avg_delay）、网表（层次化）定稿。以下 4 项是 GNN 代码侧要跟上 spec 的改动。依赖关系：①→②③（先让 `parse_netlist` 支持任意 I/O，再改 data_loader 和模型读出），④ 独立。

### 9.1 `parse_netlist` 支持任意 I/O（去 `out` 硬编码）

- **现状**（`src/graph_builder.py:parse_netlist`）：从 `.SUBCKT DUT` 行猜输入 pin（排除 vdd/gnd/vss/out），并硬编码 `nodes['out']` 为唯一输出节点。
- **目标**：输入/输出 pin 由显式列表传入，支持任意 N 输入 / M 输出。
- **改动**：
  1. `parse_netlist(netlist_str, input_pins, output_pins)` 增加两个参数。
  2. 输入节点 = `input_pins` 列表；输出节点 = `output_pins` 列表（删掉硬编码 `nodes['out']`）。
  3. `build_static_graph` 调用处从 `circuit_static` 的 `input_pins_json` / `output_pins_json` 读列表传入。

### 9.2 `data_loader` 读 JSON pin 列（替代固定 4-pin 列）

- **现状**（`src/data_loader.py:_get_dynamic_features`）：读 `slew_a/b/c/d`、`load_a/b/c/d`、`arrival_time_a/b/c/d` 固定列。
- **目标**：读 `pin_slew_json` / `pin_load_json` / `pin_arrival_json`（dict keyed by pin 名）。
- **改动**：
  1. 解析三个 JSON 列，得到 per-pin 的 slew/load/arrival dict。
  2. `self.pins` 从 `input_pins_json` 读（不再固定 `['a','b','c','d']`）。
  3. 按 `input_pins_json` 顺序取每 pin 特征，广播到对应输入节点。

### 9.3 `DelayGNN` 多输出读出（预测 avg_delay）

- **现状**（`src/model.py:forward`）：`gate_mask * x` → `global_add_pool` → 单 scalar，隐含单输出 `out`。
- **目标**：支持任意 M 输出，读出层对 M 个输出节点取平均，预测单个 avg_delay。
- **改动**：
  1. 输出节点来自 `output_pins_json`（M 个），不再硬编码 `out`。
  2. 读出层对 M 个输出节点池化（先做简单版：对输出节点特征取平均 → 预测单个 avg_delay，对齐 V2 的 DELAY=avg_delay）。
  3. 后续可选：每个输出节点预测一个延迟再平均（更细粒度，但先不做）。

### 9.4 评估口径「最坏」→「avg_delay」

- **现状**（`src/utils.py:ranking_metrics`）：按 `(expr, corner)` 分组，每变体取 `max over (pin/dir/vector)` 最坏延迟。
- **目标**：按 `expr` 分组（单 corner），每变体直接用单个 avg_delay。
- **改动**：
  1. 分组 key 从 `(expr, corner)` 改成 `expr`（单 corner 下 corner 维度无意义）。
  2. 每变体延迟直接取 avg_delay（不再 max over pin/dir/vector；V2 里同一电路所有行 DELAY 相同，max 退化为单值）。
  3. regret/Spearman/top1/recall 等指标计算逻辑不变。

### 9.5 前置验证（先做，再动上面 4 项代码）

- **「无 wave」模型 regret**：跑 `USE_TRANSISTOR_WAVE=False`（+ 关 supply_noise），看 regret 掉多少，决定要不要蒸馏。
- **「任意 I/O + 多输出 + avg_delay」能否训练**：用 V2 数据（单 corner + 任意 I/O + avg_delay）训一个 seed，确认 9.1~9.4 改完后能正常收敛、排序指标合理。

> 注：本节 9.4 同时把 8.4 里「#2 延迟口径」的决策从「最坏情况」明确为「avg_delay」（对齐 V2 spec 的 DELAY 定义）。

---

## 十、Rust 粗筛接入（shadow 并行模式）+ 验证标准（2026-08-26）

> 决策（15.2.3）：蒸馏失败（wave 缺失根本性），接受 **no-wave 6-seed 集成模型**做 Rust 粗筛。
> 粗筛 = 用 GNN 给候选排序，SPICE 只对 top-K 精排。先跑 **shadow 并行**（GNN + SPICE 都跑，SPICE 仍做决策），
> 积累 GNN vs SPICE 对照数据，按标准判定「替换」还是「预排序」。

### 10.1 no-wave 粗筛模型（serve.py，已定稿）

- 模型：**6-seed no-wave 等权集成**（42/123/456/2468/1357/2024），hi_spread 遗憾 8.78% / recall@2 61.8%。
- 服务：`scripts/diag/serve_http.py`（跑 orca，`POST /rank`），6 checkpoint 集成预测 avg_delay。
- 输入：候选 `{id, netlist(SPICE), input_pins, output_pins}`；输出：`{ranked:[{id, avg_delay}]}` 升序。
- **绝对延迟被低估 ~2x**（no-wave 点误差特性）——粗筛只看排序，不依赖绝对量。

### 10.2 Rust 接入点（shadow 并行，不改贪心决策）

**核心文件**：`NetlistOpt/src/tl_opt.rs` 的 `FullCircuitTlEvaluator::evaluate()`（每个候选在此跑 SPICE）。

**做法**：包一层 `ShadowGnnTlEvaluator`（包住现有 `FullCircuitTlEvaluator`）：
1. `evaluate(candidate)`：先调 GNN `/rank` 得 `GNN_pred`；再调内部 SPICE 得 `true_avg_delay`（决策照旧）；
2. 记录 `(candidate_id, GNN_pred, true_avg_delay, 所在 window)` 到 CSV；
3. 返回 true metrics —— **贪心行为完全不变**（接受/拒绝仍由 SPICE 真值决定）。

**GNN 客户端**（Rust，零依赖）：
- `std::net::TcpStream` 手写 HTTP POST `/rank`（Cargo 无 reqwest）；
- 候选转 GNN 输入：`module.inorder` → input_pins、`module.outorder` → output_pins、
  `crate::spice::expr_to_hierarchical_spice(expr, &module.outorder)` → netlist；
- 开关 `GNN_SHADOW`（env 或 TlSearchParams 字段）启用，默认关。

### 10.3 验证标准（跑完 shadow 后判定）

在 46 benchmark（或子集）跑完后，对**每个 window 候选集**（≥4）计算并汇总。

**recall 判断标准（2026-08-26 用户定稿，k=2/3 双口径；粗筛下 recall 是最重要指标）**：

| 口径 | 定义（按候选集计数取比例） | k=2 | k=3 |
|---|---|---|---|
| **严格** | 前 k 名中出现**实际第 1 名**的概率（真最优存活率） | recall@2 严格 | recall@3 严格 |
| **宽松** | 前 k 名中出现**实际前 k 名之一**的概率（出现一个就算） | recall@2 宽松 | recall@3 宽松 |

其它指标与达标线：

| 指标 | 定义 | 达标线 |
|---|---|---|
| **选择遗憾** | (GNN 选最优 − SPICE 真最优)/真最优 | **≤ 5%**（主判据） |
| **两阶段最终遗憾** | GNN 前 3 → SPICE 精排 → 选前 3 内真最优的遗憾（真#1 在则 = 0） | 参考（粗筛实际收益） |
| Spearman | GNN 排序 vs SPICE 真序秩相关 | ≥ 0.6（次判据） |
| 仿真节省 | 若 GNN 只排 top-K 能省的 SPICE 比例 | ≥ 75%（收益项） |

**判定**：
- 主判据达标（recall 严格口径达标 + 遗憾 ≤5%）→ **GNN 替换逐候选 SPICE 排序**：优化改成「GNN 排全部候选 → top-K 才 SPICE 精排 → 接受第一个改善的」，仿真省 ≥75%。
- 主判据不达标 → 保留 SPICE 全排序，GNN 只做**启发式预排序**（先粗排再 SPICE 精排 top-N，仍省部分仿真）。

> 关键：贪心每 window 只接受**第一个**改善候选（`break`），粗筛价值 = 让 GNN 把「该接受的候选」排到前面 → SPICE 只验证 top-K，不丢最优又省仿真。

### 10.4 参考实现

`NetlistOpt/src/gnn_shadow.rs`（已直接写入 NetlistOpt 独立仓库；`Cargo.toml` 加 serde/serde_json、`lib.rs` 挂 `mod gnn_shadow`、`tl_opt.rs` 挂载已就位）：
- `GnnClient`：`std::net::TcpStream` HTTP POST `/rank`（零依赖），解析 ranked JSON。
- `ShadowGnnTlEvaluator`：包 `FullCircuitTlEvaluator`（`new` 接收已配好 simulation_cfg 的 inner，因 `design_template` 是 tl_opt 私有字段），调 GNN + SPICE，写 CSV。
- 挂载：`optimize_tl_text` 里 env `GNN_SHADOW=1` 时换用 Shadow 版本（host/port 可 `GNN_HOST`/`GNN_PORT` 覆盖，默认 10.20.34.16:8000）。

### 10.5 服务器仿真环境就绪（2026-08-26，实测）

**结论：SPICE 后端用 yongsheng 的 Xyce（zen5 构建），服务器上零配置跑通。**

- ✅ **yongsheng 的 Xyce**：`/home/yongsheng/Apps/Spack/opt/spack/linux-zen5/xyce-7.10.0-sikno3voapvu34qym5o7gbiysestqq4y/bin/Xyce`（就是本地 `xyce.sh` 的默认路径）——**能跑 ASAP7 BSIM-CMG 模型**，单次瞬态 0.33s。
- ❌ **/opt/spack 的 Xyce**（`/usr/local/bin/Xyce`，linux-x86_64_v4）：能 load 但**跑不了 ASAP7 模型**——原生 M(MOSFET) 器件未注册（level-1 都报 "no valid model card"）、BSIM-CMG 版本 106.1 落后于模型 107。**排除**。
- ✅ **Cadence Spectre**：`/opt/rh/SPECTRE201/.../spectre`（ver 20.1）+ license `/opt/rh/cds.lic.dat`——sibo 生产流实际用它，能跑 ASAP7（实测 delay 5.04ns / slew 13.6ps）。作为 Xyce 之外的备用后端。
- ✅ **模型与 CDL**：`/home/tianlang/asap7/asap7_pdk_r1p7/models/hspice/7nm_TT_160803.pm` + `/home/tianlang/asap7/asap7sc7p5t_28/CDL/LVS/asap7sc7p5t_28_R.cdl`。测试平台把模型**自动拷贝到每个 sim 目录**，无硬编码路径问题。
- 注意：ASAP7 模型是 Spectre/HSPICE 原生（BSIM-CMG 107）；Xyce 仅 yongsheng 的 zen5 构建可用。

### 10.6 46 电路 shadow 基准（2026-08-26，NetlistOpt commit b253aed）

- 入口：`tests/tl_opt_shadow_batch.rs`（`#[ignore]`，需 `-- --ignored`），遍历 `testbench/tl_cells/level{0..4}/*.tl` 共 46 电路。
- env：`GNN_SHADOW=1 GNN_HOST=127.0.0.1 GNN_PORT=8000 SPICEVIZ_OFF=1 TL_MAX_ITERS=<n> TL_ONLY=<分片>`；6 组并行（level 分片）24 核服务器，全程 ~1.5h。
- `SPICEVIZ_OFF=1`：跳过 spiceviz.py 后台 PNG 渲染（debug 可视化，每 DUT 占一核）。
- 每个电路 `gnn_shadow.csv` 记 `eval_idx, iter, window, gnn_pred, true_delay, transistors`（window = window_try，10.3 按 (电路, window) 分组）。
- 结果：**45/46 成功**；`ovf1`（9入1出溢出标志）Xyce 瞬态步长崩溃不收敛（"Step size reached minimum step size bound"，初始设计即失败）——SPICE 自身限制，非代码问题，记为已知失败。

### 10.7 两个推理侧根因 + 修复（2026-08-26，关键）

初跑发现 no-wave GNN 在 Rust 候选上**系统性反向**（Spearman −0.50，recall@top-3 23%，遗憾 51%）。双根因，均已在 serve 侧修复（无需重训）：

1. **gate_states BFS 输出节点硬编码 `'out'`**（`src/logic_sim.py::compute_gate_states`）：Rust 候选输出叫 `y`（output_pins），反向 BFS 从 `'out'` 出发 → 交集空 → **gate_mask 清零全图** → 模型只看 corner+签名，预测全扁平。修复：`outputs` 参数（默认 `['out']` 向后兼容 V1），serve 传真实输出引脚。
2. **Rust 复合门名域外**（`SC_JOIN_AND_WIRE_...` 不在 sc_expansion.json）→ 全落 COMPLEX 兜底类（n_t/drive/p_g_h 全默认）→ 候选间差异被抹平。修复：Rust 生成器**自产逻辑类**——`spice.rs::classify_subckt_logic`（按表达式结构分类 10 类，保守规则）→ `expr_to_hierarchical_spice_with_logics` 返回 `{subckt名->逻辑类}` → `GnnCandidate.gate_logics` 随候选发出 → serve 用 **thread-local 覆盖**（`graph_builder.set_gate_logic_overrides`，ThreadingHTTPServer 多线程安全；优先级 sc_expansion > 覆盖 > 名字回退，训练路径零影响）。

效果：pre-fix 22.8%/50.8%/−0.50 → post-fix 51.7%/7.38%/+0.14（部分数据 31 电路时 51.1%/10.8%/+0.31）。

### 10.8 最终判定（46 电路全量，2026-08-26）

**统计**：89 个合格候选集（≥4 候选；n 分布 4~33，中位 8），成功行 950、失败行 35。

> **结论摘要（用户定稿文字）**：前 3 名中出现实际第 1 名的概率（严格）是 47.2%；前 3 名中出现实际前 3 名之一的概率（宽松，出现一个就算）是 79.8%；两阶段最终遗憾仅 2.02%。两阶段最终遗憾指的是：先由 GNN 选出前 3 名候选，再对这 3 个候选用 SPICE 精排、从中选出最优，最终损失的延迟占该组真正最优延迟的比例（中位数 0%）。也就是说，即使 GNN 有时没能保住精确的最快变体，前 3 名里通常也包含着非常接近最快的候选，经过 SPICE 精排后几乎不会损失延迟。

**recall 判断标准（k=2/3 双口径，用户定稿；粗筛下 recall 是最重要指标）**：

| 口径 | k=2 | k=3 |
|---|---|---|
| **严格**（实际第1 ∈ 预测前k） | **29.2%** | **47.2%** |
| **宽松**（预测前k 含 实际前k 之一） | **57.3%** | **79.8%** |
| 跨度>10% 子集（54 集）：严格 | 33.3% | 46.3% |
| 跨度>10% 子集：宽松 | 57.4% | 77.8% |

其它指标（89 集）：

| 指标 | 均值 | 中位 |
|---|---|---|
| 选择遗憾（GNN 自选 top1） | 7.38% | 2.88% |
| **两阶段最终遗憾（前3→SPICE 精排）** | **2.02%** | **0.00%** |
| Spearman | 0.140 | 0.200 |

**判定：严格口径 recall（k=2 29.2% / k=3 47.2%）远低于达标线 → GNN 只做启发式预排序（SPICE 全排序 + GNN 先粗排 top-N），不替换逐候选 SPICE。**

解读：
1. **recall 严格口径低**：真 #1 只有约一半概率进入预测前 3——GNN 无法稳定保住真最优。
2. **但两阶段最终遗憾仅 2.02%（中位 0%）**：即使真 #1 不在前 3，前 3 内真最优通常也接近真最优（候选集内延迟差本来就小）——粗筛「丢失最优」的实际延迟代价很小。**严格 recall 反映「能否找到精确最优」，两阶段遗憾反映「丢失多少延迟」**——两个视角结论不同。
3. 粗筛定位：GNN 前 3 → SPICE 精排方案在**延迟损失**上可接受（2.02%），但**精确最优保住率**不足（47%）——若要求「每次找到真最快」则不达标；若只要求「延迟损失小」则可行。
4. 本质：no-wave 模型在候选集内（延迟差常 <5%）分辨不了精确最优——13.x 已确认的 SNR 天花板；10.3 判据按全部 window 集汇总（不筛跨度），比 V2 hi-spread 口径苛刻。
5. 分析脚本：`scripts/diag/_shadow_analyze.py`（分组/过滤/判定），输出 `reports/_shadow_bench_final.txt`。

> ⚠ **17.0.13 口径澄清（2026-09-06 补，勿再引本节省 recall 当 serve 真值）**：本节 89 集 / 950 行 / 35 失败、候选 n 4~33 中位 8 = **SPICE 刚就绪第一轮的早测小口径**（评估量仅现行 106 集 / 5,390 行 / 8 失败的 ~1/6，窗口偏小偏易）→ **47.2% / 79.8% 不是模型间的可比最优**。三要点：① 47.2/79.8/7.38 = **6-seed 等权集成**（§11.1：单 nowave 41.7% / 71.3% / 9.60% → 集成 47.2% / 79.8% / 7.38%），m4 全族均单 seed、从未集成；② 跨口径比 recall 无意义——同一份 09-02 全量 CSV（5,398 行）按朴素 ≥4 分组重算宽松 **94.4%** / 严格 56.7%（§14），口径摆动远超模型差距；③ 本节存档文件 `reports/_shadow_bench_final.txt`（mtime 2026-08-29）现行 pipeline 从不重写 → **现行口径 = 106/5390/8，结果读 `~/shadow_analyze.out`**。

---

## 十一、提高 Rust 成绩的方向清单（16.11.4，2026-08-31 记录，后续可选做）

> 现状基线（16.9.3 nowave 单模型）：严格@3 41.7%、宽松 71.3%、选择遗憾 9.60%、两阶段 2.02%、Spearman 0.141。主判据未达标（严格 recall 需 ~90%）。
> 本文档 10.x 已判：GNN = 启发式预排序（两阶段遗憾可接受 2.02%，但精确最优保住率不足）。
> 以下方向按「对症度 × 成本」排列，均为**后续可选**，未承诺实施。

### 🟢 已验证有效（立即可做）

| # | 方向 | 依据 | 成本 |
|---|---|---|---|
| 1 | **多 seed 等权集成** | 16.9.3：6-seed 集成遗憾 9.60%→7.38%、严格@3 41.7%→47.2%。现有 6 个 nowave seed（42/123/456/2024/1357/2468），serve `--ckpt` 已支持多 checkpoint | 分钟级（缓存全暖） |
| 2 | **两阶段流程调优**（top-K 大小 / 精排候选数） | 两阶段遗憾 2.02%（中位 0%）已可操作，可微调 K | 分钟级 |

### 🟡 进行中（等结果）

| # | 方向 | 依据 | 状态 |
|---|---|---|---|
| 3 | **新数据 m4（5 形状 OOD 补充）** | 16.9.x 发现 Rust 候选部分形状 OOD → m4 补 9入6出/8入4出/7入4出/5入5出/4入3出 | v2nowave42m4 / v2iaa42m4 训练中 |
| 4 | **近似 ids_avg 特征（线性）** | 16.11.0：R²=0.64/Spearman 0.54；v2iaa42 实测遗憾 3.35%（vs nowave 3.77%） | v2iaa42m4 训练中 |

### 🔵 新方向（尚未做）

| # | 方向 | 对症度 | 说明 | 成本 |
|---|---|---|---|---|
| 5 | **Rust 候选分布微调（domain adaptation）** | **最高** | 16.9.3 发现「Rust 候选近相同（延迟差极小）+ 部分形状 OOD」→ 训练分布 ≠ Rust 候选分布。用 shadow 已积累的 Rust 候选 + SPICE 真值（gnn_shadow.csv，几千条）微调/重训现有模型，直接对 Rust 分布优化 | 中（需微调脚本） |
| 6 | **serve 端 gate_states 对齐验证** | 中高 | serve 用 BFS 逻辑仿真推 gate_states（logic_sim.py），训练用真实 gate_states_json——若 BFS 与真实不一致则 serve 特征带噪（10.7 已修过 'out' 硬编码，但需全面验证） | 低（验证脚本） |
| 7 | **绝对延迟刻度校准** | 中 | 16.10.1：no-wave 预测被低估 ~2x（99.2% 离谱）；两阶段精排用 SPICE 真值不受影响，但校准 scale 可提升粗筛 top-K 质量 | 低 |
| 8 | **排序损失直接优化（非蒸馏版）** | 中 | 蒸馏的 rr(reg+rank) 模式证明排序损失可行（随蒸馏失败被弃）；可试「真值 huber + rank 辅助 loss」直接训练 | 中 |
| 9 | **门类型词表对齐验证** | 中 | serve 从候选 netlist 重建门类型词表，训练用全局词表——若 Rust 候选含词表外门型则 embedding OOV/随机；sc_expansion merge 24,625 后应覆盖，需验证 | 低 |
| 10 | **Rust 候选数据回流（自举）** | 中 | shadow 已产出「候选 → 真值延迟」数据，可回流训练集，GNN 越用越准（与 #5 类似但为持续机制） | 中 |
| 11 | **GBDT15 非线性近似 ids_avg** | 中 | 16.11.2：GBDT15 R²=0.682 / Spearman 0.653（vs 线性 0.543，+20% 排序一致性），需导出 sklearn 树到 Rust | 中-高 |
| 12 | **wave 蒸馏 iaa（补特征后蒸馏）** | ~~中~~ → **已裁决不跑** | ⚠ §13.4 更正（16.11.40）：**这格从未真正测过**——v2kdwave42iaa42 本意即此（iaa 学生），但 KD 分支 `USE_TRANSISTOR_WAVE=False` 把近似列挡掉（data_loader L507/L510），学生训时纯拓扑。且裁决③证明此前提也站不住：近似列 serve 端净伤（iaa 15.21/iag 12.19 均劣 nowave 10.87）+ 纯拓扑 KD 双端不赢 → 即便真给 iaa 学生补上近似列再蒸，也是「给净伤特征加 KD」，期望负 | 已跑 v2kdwave42iaa42（实为纯拓扑）；无需再跑 |

### 执行建议

1. **现在**：跑 6-seed nowave 集成验证（#1，分钟级确定性提升）
2. **等**：v2nowave42m4 / v2iaa42m4 结果 → 判断 m4 是否解决 OOD（#3）
3. **优先验证**：#5（Rust 候选分布微调）——唯一直接以 Rust 候选分布为目标的方向，先做分布差异检查（shadow CSV vs 训练数据形状/大小/延迟范围）确认值得
4. **顺手**：#6（gate_states 对齐）成本最低，可能是 serve 侧隐藏噪声源

---

## 十二、训练数据 vs Rust tl_opt 候选差异实测（16.11.14，2026-09-02）

> **核心发现（16.11.18 修正）**：训练数据与 Rust 贪心候选的差异**需按晶体管看（门数口径误导——训练用巨型复合门压缩顶层 X_ 数）**。真实差异：① 训练缺晶体管 >184 的超大电路（Rust ENC8/ALU2/ADD4_OVF 到 189-316）；② 更根本的是**深度/结构 OOD**（训练深度全 ≤5,Rust 深电路 8-12）——见 §12.4。

### 12.1 电路规模差异（按晶体管口径，门数有误导）

> ⚠️ **门数口径警告（16.11.18）**：训练数据用**巨型复合门 SC_JOIN_***（名字超长 = 内部大量逻辑），顶层 `X_` 门数（4-15）**严重低估真实规模**（m4 9入6出顶层 21 门但晶体管 110）。**真实规模必须看 `transistor_count`**。下表门数列仅作参考，晶体管列才是真实对比。

| 来源 | 顶层门数中位 | 晶体管中位 | 晶体管范围 |
|---|---|---|---|
| Rust tl_opt 候选（全体） | — | 跨度 4~247 | 4~316 |
| Rust DEPTH_MIX | — | 39 | —（深度 OOD 的代表——〔16.11.24 待复核：候选 X_ 链实测 0% >6，见 §12.2b〕） |
| Rust ADD4_OVF | 59 | **247** | 191~316 |
| batch_v2_full | 7 | 52 | 18~96 |
| batch_v2_rest | 4 | 58 | 18~184 |
| batch_v2_m4 | 15 | 58 | 18~118 |

- **真实规模缺口**：训练晶体管上限 184（rest），Rust ENC8(189)/ALU2(235)/ADD4_OVF(247, max 316) **超出**——训练几乎无 >184 管样本。
- **DEPTH_MIX（39 管）规模不 OOD**——它在训练范围内（52-58 中位）但 recall 低 → OOD 在**深度/结构**（§12.4 缺陷 1），非规模。〔16.11.24 待复核：候选 X_ 链实测 med 4/max 6（0% >6），OOD 未必在深度，可能是结构/spread 或口径差异，见 §12.2b〕
- **下一次生成数据要求**：见 §12.4 缺陷 2（补 180~350+ 管、显式多小门、与深度解耦）。

### 12.2 形状覆盖（已验证 OK，非瓶颈）

- Rust 46 电路共 18 种 I/O 形状，在 full+rest+io+m4 **全量中 100% 覆盖**（无缺失）。
- ~~m4 补的 5 种多输出形状 (4,3)(5,5)(7,4)(8,4)(9,6) **不是 Rust 主战场**——Rust 主要是单输出 (1,1)(2,1)(3,1)(8,1)(9,1)(16,1) 等（rest 早已覆盖）。~~ →〔⚠ **2026-09-03（16.11.36）修正：此归因错误，撤回**，见下方 16.11.36 实测〕
- **结论（修正后）：m4 对 Rust 成绩无改善的原因不是「多输出非主战场」，而是「形状对、规模缩水」（下方 16.11.23/16.11.36 实测）；形状不是瓶颈，规模才是。**

- **16.11.36 修正（shadow 候选集/候选行实测，推翻「Rust 主战场单输出」归因）**：
  - 46 电路按**电路数**确为单输出为主（35/46=76%，11 多输出=24%）——但这不等于 serve 面对的主战场。
  - 按 shadow **候选集**计（逐集明细实测，见 reports/_shadow_bench_nowave_fullrest.txt & _shadow_bench_final.txt）：**多输出电路占 57%（115 集报告）/ 52%（89 集报告）**；按**候选数 Σn**（≈serve 需评估的行）计：**多输出 52%**。→ 多输出是 Rust serve 的半壁江山，非边缘。
  - 权重最大的几个电路正是**多输出大电路**：ADD4_OVF (9,6) 366 候选（全 shadow 单电路最大来源）、COMP4 (8,3) 277；DEPTH_MIX (5,1) 336 / OR2 308 是单输出里的前排。
  - **含义**：m4 选的 5 种多输出形状 (4,3)(5,5)(7,4)(8,4)(9,6) 恰好对上了 Rust serve 权重最大的 ADD4_OVF/ALU2/ENC8/SHIFTER4/ENC4——形状方向没错；**m4 无效的根因是形状内规模缩水**（下方 16.11.23：trans 只有 Rust 的 1/2~1/4），不是选错了形状。
  - 注：本修正用 115/89 集存档逐集明细；106 集时代（v2nowave42m4 等）服务器最新报告的逐集多输出占比待 `_shadow_analyze.py` 输出核对（结构应一致）。

- **16.11.23 更新（实测 m4 形状-规模缩水，触发生成规格修订为「形状-规模锚定」）**：
  - m4 特意选的 5 种多输出形状 (4,3)(5,5)(7,4)(8,4)(9,6) **形状名全在，但形状内实现全是「最小规模缩水版」**（实测）：
    - (4,3) trans 26-30 vs Rust ENC4 56-69（~2×）；(7,4) 58-66 vs ALU2 235-262（~4×）；(8,4) 66-74 vs ENC8 189-233（~3×）；(9,6) 110-118 vs ADD4_OVF 247-316（~2.2×）；仅 (5,5) 52-56 vs SHIFTER4 48-60 达标。
    - rest 同病：(8,3) max 182 < COMP4 233；(5,2) max 158 < ALU_SLICE_SMALL 208。
  - **网表证据（m4 (7,4) 抽样）**：trans 68 / 顶层仅 15 门，全 SC_JOIN_* 巨复合门拼形状——而 Rust ALU2 是 55 表达式节点的显式展开。→「拼形状」而非按 Rust 功能族生成。
  - **含义**：「形状 100% 覆盖」只是形状名不缺；**同形状内没有达到 Rust 档位的实例 → Rust 深/大电路照样 OOD**（形状是规模/深度的载体，光有形状名不够）。
  - **V3 修订（16.11.23）**：DATA_SPEC_V2 第四节新增「### I/O 形状锚定」——18 形状全覆盖 + 每形状最低电路数配额 + ★ 五大规模形状 (5,2)(7,4)(8,3)(8,4)(9,6) 强制 trans p50 ≥ Rust med / max ≥ Rust max×0.85 + (5,1) 深度档 + 功能锚定（按 Rust 功能族生成，禁止拼形状）+ 多输出形状宽松下限 p50 ≥ Rust med×0.7。验收改 metadata 逐形状核对，废弃「多输出 ≥20%」比例口径。

### 12.2b Rust 候选深度档实测（X_ 宏级链，2026-09-03）——规模/深度配额二次细化依据（DATA_SPEC_V2 二版）

- **方法**：解析 temp_sim_test/tl_opt_batch/**/dut_expr_*.sp（1135 个 Rust 候选）的 .SUBCKT DUT 内 X_ 行；口径 = **X_ 宏级最长链**（SC_JOIN_* 宏算 1 层），与训练 gate_level_netlist 同口径、与 .tl 模板的 X 级联不同（宏打包会压平树）。脚本：scripts/diag/_rust_dut_depth.py。
- **实测**：候选深 >6 的只有 6 个电路族——
  - ★深主档（9-16）：OVF 13/15/15（100% >9）、ADD4_OVF 12/14/16、COMP4 10/10/12、ALU2 9/9/10；
  - ◆深次档（7-9）：ALU_SLICE_SMALL 6/8/11（46% >6）、PARITY4 7/8/8（67% >6）；
  - 其余 36 模板候选全部 ≤6。
- **三处归因修正（相对 16.11.23 shape 锚定表）**：
  1. **(9,1) OVF 是最深模板（13-15）但中规模（104-146 管）**——上版标「中规模·建议含深链」非强制，漏标；已升 ★ 强制深度档。
  2. **(5,1) DEPTH_MIX 候选 0% >6**（med 4/max 6）——上版按「深度 OOD」标 ★ 深度档（p90≥10）**依据错误，已撤**；旧「DEPTH_MIX 纯深度 OOD」归因（§12.1/缺陷 1）基于 .tl 级联/模板口径，与候选 X_ 链实测不符，**待复核**（可能是结构/spread OOD 或口径差异）。
  3. **(8,4) ENC8 大而浅**（189-233 管但候选深仅 4，SC_JOIN 宏压平树）——只锚规模档，**不要求深**。
- **结论**：① 深度与规模**正交**（大而浅 ENC8 / 中而深 OVF / 小而深 PARITY4 并存），必须双档独立锚定、双档验收；② 深度 >9 只来自**链式输出依赖**（逐位进位 / 比较 / 算术级联），不能靠做大或串缓冲伪造；③ 旧 spec「≥10% 180~400+」「≥10% >6」全局占比可堆低端（180-190 管 / 7 层），且两维互相漏（浅大、小深都不受约束）。
- **V3 二版（DATA_SPEC_V2 §四「### 形状-规模-深度锚定」，2026-09-03）**：Tier A ≈37% 行档位锚定（18 形状配额 ~7900 电路 ≈22.2 万行；形状 × 规模档 × 深度档三元组）+ Tier B ≈63% 行 **Rust 邻域多样化**（不脱离 Rust 使用域：1-16 入/1-6 出、trans ≤400、深度 ≤18，替代旧全域随机枚举）；规模拆 189-233 / 233-316 两档（删 400+ 主档）；深度主档 9-16 / 次档 7-9；**结构模式锚定**（功能任选 + 链式/级联 + 实测达标，非复刻 Rust 电路）；**泛化闸门**（新模型在旧数据固定 test 集上排序指标 ≥ v2wave42m4 基线——跨分布口径，2026-09-03 修正）。

### 12.2c Rust vector 语义验证 + 组内延迟差实测（2026-09-03，V3 规格 R5/R2 依据）

- **vector 语义（对照 Rust 源码 design.rs uild_simu_vectors_for_simulation + simulation.rs）**：Rust 按 **per-(output, pin)** 各取真值表第一个让该输出翻转的 break 行（truth_table_idx，该 pin 位=0）；rise/fall = **同一 break vector 的双向激励**（simulation.rs：rise 用 idx 行、fall 用 idx | (1<<pin) 行）。m4 训练数据实测 rise/fall vector 关系与此**完全一致**（11880/11880 组，fall = rise 置 pin 位）→ **训练数据已对齐 Rust 语义**；spec 原文字「每 (pin,dir) 固定 1 个 vector」错误（实际每 output 独立 break、同 (pin,dir) 下 M 行 vector 各异），V3 规格已修正（R5）。
- **Rust 714 有效候选集（≥4 候选）组内真延迟分布（2026-09-03，脚本 _rust_group_delay_dist.py）**：组内 spread med **48%**（p75 58% / p90 72%）；「含 ≥2 个差 >10% 候选」的组占 **81%**（578/714；≥1 个 83%、0 个 17%）；组内相邻差四档全局分布 **<1%:20% / 1-5%:35% / 5-20%:39% / >20%:6%**。
- **含义（V3 规格 R2 修正）**：旧要求「每组 10-15 变体含 2-3 个差 >10%」略强于 Rust 现实（19% 近差功能组天然达不到）；改为统计验收线对齐 Rust 实测——**≥80% 的组含 ≥2 个差 >10% 变体 + 相邻差四档全局非零**。防「尽力不管」漏覆盖，又不逼生成方硬造不真实变体。

### 12.2d Rust 宏层 / 仿真条件 / DEPTH_MIX 归因定案（2026-09-03，V3 规格校准依据）

- **宏层实测（1135 个 dut .sp，脚本 _rust_macro_stats.py）**：SC_JOIN_* 宏 **1064 种**（名称编码逻辑结构）；宏内部 M_（trans）med 5 / p90 14 / max 84（2-6 管小宏为主 + 8-84 管多输入复合宏）；DUT 内 X_ 数 med 32 / p90 57 / max 73。→ 校准 V3「宏粒度」：不是「每宏 2-8 逻辑/4-8 管」，而是「沿用 Rust 生成器同套宏 + X_ 数 20-73（med ~32）」，禁 m4 式 ~15 超大宏。
- **仿真条件（asap7.sp + tb_*.sp 实测）**：slew 2ps（翻转沿在 **20-22ps**，非 t=0）；Cload 1fF；.tran 1p 1n；**仅 switching pin 翻转**（接 VSTIM_RISE/FALL），其余输入按 vector 固定 vdd/gnd 常量不翻转（XUUT 实例化证实）；**每 (output) 单独 tb、仅被测输出接 1fF**。→ 修正 spec「所有输入 t=0 同时翻转」旧表述。
- **DEPTH_MIX 归因定案（原「待复核」）**：候选 X_ 链 0% >6（排除深度 OOD）；538 组遗憾 med 7.65% vs 全体 714 组 4.89%（高 2.8pp，但 p75/p90 与全体重合）；spread 51% ≈ 全体 48%；训练 rest (5,1) spread med 361%（覆盖充足，非缺大 spread）→ **「DEPTH_MIX = 纯深度 OOD 代表」正式撤销**；偏高遗憾疑为具体结构覆盖或统计，不再深挖；V3 (5,1) 已有 800+ spread 大变体覆盖（trans 40-120）。
- **Rust 候选 trans 上限**：全体 max = 316（ADD4_OVF 级）→ Tier B 上限表述改为「≤316 为主、316-400 扩展 ≤10%」。

### 12.2e 16.11.28 Rust 实测 vs V3 生成要求：冲突与可优化清单（2026-09-03）

**C1. 行数公式与 Rust 敏感对语义的缺口（冲突，建议修 spec）**
- Rust 实测：仿真 tb = **仅「该输出对该输入敏感（能翻转）」的 (output, pin) 对**——多数电路 100%（全量），但输出-输入非全连通的多输出大电路 <100%：ADD4_OVF **78%** / ENC8 78% / ENC4 75% / ALU2 93% / ALU_SLICE_SMALL 99% / **SHIFTER4 52%**。
- spec R5 已写「每 (output, pin) 取第一个让该输出翻转的 break（不敏感则无翻转行）」→ 隐含行数 ≤ 2×N_in×M；但第四节行数公式仍写死「= 2 × N_in × M」（全量）。**矛盾**：严格按 R5 语义，不敏感 (out,pin) 无行，行数 < 2×N_in×M；m4 实测每电路恰为全量 108 行（含对输出不敏感的行，且都有有效 DELAY）——生成方实际未跳过。
- **建议**：行数公式改「**≤ 2×N_in×M**（仅生成能翻转该输出的 (out,pin) 对，对齐 Rust break；预算按全量上限核算）」，使训练 avg_delay 与 Rust 同口径、避免不敏感对混入。

**O1. 组内相邻差各档占比参考区间（可优化，防「象征性非零」）**
- Rust 相邻差四档实测：<1% 20% / 1-5% 35% / 5-20% 39% / >20% 6%（中间档为主）。spec 现仅要求「四档全局非零 + ≥80% 组 ≥2 个差>10%」——生成方可「每组 2-3 个拉开 + 其余全 <1%」凑数，中间档仍空。
- **建议**：加各档占比参考区间（对齐 Rust ±10pp）：<1% 10-30% / 1-5% 25-45% / 5-20% 30-50% / >20% 3-10%。

**O2. 训练组 spread 形态对齐 Rust（可优化）**
- 实测对比：训练 rest (5,1) 组 spread med **361%**（旧形态 = 相邻差 med 0.1% + 少数极慢 outlier 拉高）；Rust 组 spread med **48%**（p75 58% / p90 72%，差异分布更均匀）。训练组排序难度形态与 Rust 评估明显不同。
- **建议**：V3 组 spread 参考中位 ~30-60%（配合 O1 各档占比），让训练排序信号分布对齐 Rust 评估分布。

**O3. sc_expansion 生成方式提示（可优化）**
- Rust 宏名编码逻辑结构（SC_JOIN_AND_AND = AND2 等，1064 种）→ V3 宏集预计数百~数千种；宏展开表可由**宏名结构自动推导**，不必手写。建议给生成方提示以降交付成本。

**已处理项**：L19 摘要「每组 2-3 个 >10%」残留 vs R2 统计线——16.11.28 已同步。

### 12.3 Rust 候选微调数据可行性（方向 #5 前置，已验证充足）

| 指标 | 值 |
|---|---|
| shadow CSV 总行 | 5,391 |
| 电路数 | 46 |
| 总候选集（window） | 881 |
| **≥4 候选有效集** | **714**（覆盖 18 电路） |
| ≥8 候选集 | 556 |
| 微调可用样本（有效集内行） | ~5,160 |
| temp_sim_test 可恢复 dut netlist | 1,253 个（2026-09-03 复核：实测 dut_expr_*.sp = **1,135**，差 118 待核） |

- **数据量足够做「Rust 候选分布微调」（#5）**：714 有效集 / 5160 样本 > 500 门槛。
- 覆盖 18/46 电路——其余 28 个电路候选集 <4（排序意义小）；微调聚焦大电路（ADD4_OVF/OVF 等）正合适。
- **主要工程**：shadow CSV 没存候选网表，需从 temp_sim_test 的 dut 目录按 (circ/expr/iter/window) 映射，或改 Rust 记录网表后重跑。

### 12.4 给生成方 / 下一次数据的要求清单（16.11.17 更新：基于 V1-V7 差距归因；16.11.22 改 V3 单一数据集）

> **V3：60w 新数据 = 单一统一数据集**（不再分 full/rest/m4 批次，内部按 expr 切 train/val/test）。
> 当前数据缺陷 → 新数据改进点如下（按优先级）。**生成要求已写入 `docs/DATA_SPEC_V2.md`（头部 V3 完整区别表 + 第四节规模/深度要求）**，本清单为分析依据。

#### 缺陷 1：深度 OOD（最严重，recall 下降主因）
- **实测**：训练数据（full/rest/m4）深度**全部 ≤5（>6 占比 0%）**；Rust 深电路（ADD4_OVF/OVF 表达式深度 8-12）recall@3 仅 12-20%；recall@3 随规模递减（小 64% > 中 58% > 大 52%）。〔16.11.24 二版口径：Rust 深度按候选 X_ 宏级链实测 9-16（ADD4_OVF 12-16 / OVF 13-15），本节 8-12 为 .tl 模板级联口径，见 §12.2b〕
- **要求**：新数据必须含**深链电路**——**门数不多但深度大**（如 15-25 门、深度 8-15 的串联链），不是加门数，是**加深**。深度分布覆盖 1~15，>6 占比 ≥10%。
- 说明：6 层 GNN 消息传播深度有限，训练从未见过深电路 → 深电路排序失败。这是**补深链数据**最对症。

#### 缺陷 2：超大电路覆盖不足（晶体管口径，修正门数误导）
- **口径修正（16.11.18）**：训练用**巨型复合门**（SC_JOIN_* 内部含大量逻辑），顶层 `X_` 门数严重低估真实规模——**真实规模看 `transistor_count`**，非门数。
- **实测（晶体管）**：训练 full med 52/max 96、rest med 58/max 184、m4 med 58/max 118 → **训练上限 184**。Rust 电路分化：DEPTH_MIX(39)/COMP4(170)/OVF(131) 在训练范围内；**ENC8(189)/ALU2(235)/ADD4_OVF(247, max 316) 超出训练上限**。
- **要求**：① 补充**晶体管 180~350+ 的超大电路**（对齐 ENC8/ALU2/ADD4_OVF 级），训练上限从 184 提到 350+；② **不要用巨型复合门压缩门数**——用**显式多小门**搭建（对齐 Rust 的展开式表达，门数/晶体管都真实反映结构）；③ 与深度解耦（宽而浅 + 窄而深都要）。
- **说明**：DEPTH_MIX(39 管)规模在训练范围内但 recall 低 → 其 OOD 在**深度/结构**（缺陷 1），不在规模。〔16.11.24 待复核：候选 X_ 链实测 0% >6，见 §12.2b〕

#### 缺陷 3：结构多样性 / 中间门特征
- **实测**：serve 对 Rust 候选算近似 ids_avg 出现 **6.1% 负值**（弱驱动门），训练数据 0%——Rust 候选含训练没有的**弱驱动中间门/长链中间节点**。
- **要求**：生成的电路应含**驱动强度差异大的中间门**（扇出小/驱动弱/寄生大的节点），覆盖这类门结构。

#### 缺陷 4：组内变体延迟差「多档覆盖」（16.11.20 修正——原「缺近延迟对」不成立）
- **实测修正**：训练组内**相邻变体延迟差 med 0.1%、93% <5%**（full/rest）——**不缺近延迟对**（甚至比 Rust 更密：Rust 候选 med 4.4%、55% <5%）。之前误判「缺 <10% 对」。
- **真正问题**：训练组内变体**全挤在 <1% 超近差**（无区分度梯度）+ 少数极端慢拉高 spread——模型没从「4.4% 量级差异」学到可分辨排序（recall@1 仅 11%）。这是**训练目标/模型**问题为主，数据为辅。
- **要求**：组内变体延迟差**覆盖多档**（<1% / 1-5% / 5-20% / >20% 都有），每组含至少 2-3 个差 >10% 的「拉开距离」变体——保证组内排序任务有可学信号梯度。
- **关键**：训练评估与 Rust 一致的**组内排序任务**（不是全数据集延迟回归）——此点仍是核心（需配合排序 loss 而非纯 huber）。

#### 缺陷 5：I/O 形状（16.11.23 修订——「已覆盖」修正为「形状名覆盖但形状内规模缩水」，V3 已改为形状-规模锚定）
- Rust 46 电路 18 种形状在 full+rest+io+m4 中形状名 100% 覆盖（非形状缺失）——旧判断「非瓶颈」**只对形状名成立**。
- **16.11.23 实测修正**：m4 的 5 种多输出形状 (4,3)(5,5)(7,4)(8,4)(9,6) 形状内 trans 全部缩水（(7,4) 66 vs Rust ALU2 262 等，见 §12.2 更新）——「形状存在」≠「形状内达到 Rust 档位」；「多输出 ≥20%」比例约束防不住生成方每形状取最简实现。
- **要求（V3 已写入 DATA_SPEC_V2 §四「### I/O 形状锚定」）**：18 形状全覆盖 + 每形状最低电路数配额 + ★ 大形状 (5,2)(7,4)(8,3)(8,4)(9,6) 强制规模档（p50 ≥ Rust med、max ≥ Rust max×0.85）+ (5,1) 深度档 + 功能锚定（按 Rust 功能族生成、禁止 SC_JOIN 拼形状）+ metadata 逐形状验收。

#### 缺陷 6：样本量与分布
- 现 full+rest 69,432 行但晶体管集中在 <184(小/中电路密集);m4 只 108,720 行 5 形状(晶体管到 118)。
- **要求（60w）**：重点投在**深电路（缺陷 1）+ 超大电路（缺陷 2, 180~350+ 管）**区间,而非继续铺小电路;各 expr 组 ≥10 变体（保持 MIN_GROUP_SIZE 过滤后可用）。

#### 交付配套（避免重蹈 m4 覆辙）
- 交付含 **metadata.json**（形状/深度/规模分布统计）便于验收。
- **深度字段**：若生成管线能算，直接给每电路「深度/最长路径」字段（否则我方从 netlist 算）。

> **📍 生成要求已写入 `docs/DATA_SPEC_V2.md`**（16.11.20）：头部加「V3 相对上一次生成修改点清单」6 条 + 第四节「规模/深度多样性要求」5 条约束——给生成方以此为准，本 §12.4 为分析依据/背景。

### 12.5 关联方向更新（§十一清单）

- **#5（Rust 候选分布微调）**：可行性已验证 ✓ → 从「对症最高但未做」升级为**推荐下一步**（等 v2iag42m4 训练完，用其权重 + Rust 候选微调）。
- **#3（m4 数据）**：形状覆盖已证非瓶颈，m4 价值存疑——待 v2nowave42m4/v2iag42m4 Rust 验证后定论。
- **#9（词表）**：✅ 已验证（Rust classify 输出 7 类 INV/NAND/NOR/AND/OR/BUF/COMPLEX，全在训练 10 类词表内）——**关闭**。
- **#7（刻度校准）**：✅ 已验证——serve 输出**排名**非延迟（med 4.0 vs true 1.8e-11，r=0.03），校准对排名不适用——**关闭（当前 serve 设计下）**。
- **#6（gate_states 对齐）**：✅ 已验证 96.7% 一致——**关闭**。

---

## 十三、v2iaa42m4（线性近似 + m4）Rust 验证（16.11.15，2026-09-02）

**结果（106 候选集 / 5,390 成功行 / 8 失败行，全量 46 电路）**：〔✅ **2026-09-16 定案（§15）：不是数据错，是口径错——而且错的是当时以为对的那一边。** 旧表 106 集 = `(circuit, window_try)` **池化**口径（把 ~6.7 个批并成一个集），复算的 714 集 = `(circuit, iter, window_try)` = **一次 `pre_rank` 调用 = 部署真值**（`tl_opt.rs:893`；Pass 3 的 `prepared` 被 move 消费 → 借用检查器保证每窗口重建）。同一份 CSV（9-2 19:14，行数吻合）只换分组键：严格@3 30.2% → **56.7%**、宽松 57.5% → 94.4%，遗憾 mean 15.9%（旧记 15.21，**遗憾对口径不敏感**：±0.7pp）。→ **本表数值有效、可复现，只是尺子不同**；§14 的 ⚠ 销案。基线切换（v2iag42m4）是当时基于旧口径的裁决，其臂间排序按 §15.5 需在 batch 口径重测。〕

| 指标 | **v2iaa42m4**(线性近似+m4) | v2iaa42(线性近似,无m4) | nowave 基线(16.9.3) |
|---|---|---|---|
| 严格 recall@3 | **30.2%** | 34.9% | 41.7% |
| 宽松 recall@3 | **57.5%** | 55.7% | 71.3% |
| 选择遗憾 | **15.21%** | 16.30% | 9.60% |
| 两阶段遗憾 | **5.12%** | 4.24% | 2.02% |
| Spearman | 0.050 | 0.087 | 0.141 |

**结论**：
1. **m4 数据加入后 Rust 成绩无改善（严格 recall 30.2% vs 34.9% 反而略降）**——〔⚠ 2026-09-03（16.11.36）修正归因：原「m4 补的多输出形状不是 Rust 主战场（Rust 主要是单输出）」**不成立已撤**——按 shadow 候选集实测多输出电路占 52~57%、恰含权重最大的 ADD4_OVF (9,6) 等（§12.2 16.11.36 修正块）〕。m4 无帮助的正确归因 = §12.2 16.11.23 **「形状对、规模缩水」**：m4 的 5 个多输出形状 trans 只有 Rust 真身的 1/2~1/4，serve 面对的真大电路（如 ADD4_OVF 316 trans）照样 OOD → 规模失配是根（见结论 3）。
2. **线性近似（USE_IDS_AVG_APPROX=1）serve 失配依旧**（训练特征 = 训练行 slew/load 分布，serve = 固定 2ps/1fF + 候选门结构不同）→ 选择遗憾 15.2% 远差于 nowave 9.6%。
3. **规模失配（§12.1，36 门 vs 4-15）是根本**——m4 未解决。
4. **训练侧 v2iaa42m4 遗憾 3.85%（§v2iaa42m4 训练结果）但 Rust 端 15.2%**——巨大落差完全来自 serve 特征不一致 + 分布差异。

**待对比**：v2iag42m4（GBDT15，serve 特征与训练一致）训完跑 Rust——若显著优于 v2iaa42m4，证明「特征一致性」是关键；若仍差，规模失配是主因（需 #5 微调或大电路数据）。

**下一步**：wave 蒸馏（教师 v2wave42m4 已完成 → 学生用 GBDT15 近似特征 + 教师软标签）。

### 13.1 serve vs 训练近似特征失配归因（16.11.15，2026-09-02 量化）

**动机**：v2iaa42m4 训练侧遗憾 3.85% vs Rust 15.2%——量化落差多少来自 serve 特征失配 vs 分布差异。

**量化结果（脚本 `_quant_serve_approx_mismatch.py`）**：

| 指标 | **Rust 候选（serve 侧近似）** | **训练数据近似** |
|---|---|---|
| 样本 | 7,105 门（500 dut） | 3,685 门（300 电路） |
| 近似值范围 | -0.38 ~ 15.57 | 0.04 ~ 11.77 |
| **负值比例** | **6.1%** | **0.0%** |
| 正值中位（log10） | -0.405（0.39） | -0.284（0.52） |
| KS 检验 | p=3e-41（分布显著不同） | |

**归因结论**：
1. **serve 公式无 bug**——slew/load 固定值与训练行一致（训练 slew 全 2e-12、corner 全 s02p0_l01p0）；差异不在公式。
2. **差异来源 = Rust 候选门结构**：6.1% 门算出负近似（弱驱动门 → lg<0 → expm1 负），训练数据 0% 此类门；正近似分布也整体偏小（KS p=3e-41）。
3. **Rust 候选 = 大电路（36 门中位, §12.1）含更多弱驱动中间门** → 线性近似（R²=0.64）在这些门上退化更严重。
4. **结论**：训练 3.85% → Rust 15.2% 落差主要来自**候选门结构分布差异（规模失配根本）**，非 serve 实现错误 → **A1（修 serve）不可行**；线性近似在大电路候选上不可靠 → 寄望 GBDT15（能建模非线性）→ v2iag42m4 Rust 验证是决定性对比。

### 13.2 v2iag42m4（GBDT15）Rust 重验 = 决定性对比落地（16.11.34，2026-09-03 19:09）

> 对 §13 ⚠「待 v2iag42m4/新 Rust run 重验为准」的落地。serve 换 v2iag42m4 mid200 checkpoint（`USE_IDS_AVG_APPROX=2`，GBDT15 `idsavg_gbdt15.joblib` 加载确认）；完整报告（106 候选集 / 5,390 成功行 / 8 失败，全量 46 电路，19:09 收尾）。

| 指标 | **v2iag42m4**（GBDT15，本次实测） | v2iaa42m4（线性，§13 记录⚠） |
|---|---|---|
| 严格 recall@3 | **44.3%**（宽松 64.2%） | 30.2%（宽松 57.5%） |
| 选择遗憾 | **12.19%**（中位 6.41%） | 15.21% |
| 两阶段遗憾 | **3.93%**（中位 0.33%） | 5.12% |
| Spearman | **0.210**（中位 0.286） | 0.050 |

（hi_spread >10% 子集 79/106：严格@3 43.0% / 宽松 62.0% / 两阶段 4.96% / 遗憾 15.15% / Sp 0.262。）

**裁决**：
1. **A/B 干净**：两 run（v2iaa 9-2、v2iag 9-3）报告**同为 106 集 / 5390 行 / 8 失败** → 同候选窗口，唯一变量 = serve 模型。⚠ **2026-09-16 更正（§15）**：本节原写「106 = 标准 pipeline、714 = 另类聚合」——**倒置了**。**106 = `(circuit, window_try)` 池化口径，714 = `(circuit, iter, window_try)` = 一次 `pre_rank` 调用 = 部署真值**。上句「A/B 干净」本身**仍成立**（两 run 同用池化口径、同候选窗口 → 臂间比较有效），但**本节第 2、3 点的数值与排序不可外推到部署口径**（口径效应因臂而异：iaa +26.5pp / nowave **+55.9pp**；⚠ 2026-09-17 改，旧记 nowave +39.5pp 是拿错标树的 74.4 算的，见 §15.2/15.5 与 §15.8）。
2. **GBDT15 全线优于线性（§13.1「寄望 GBDT15」获实测支持）**：遗憾 12.19 vs 15.21、两阶段 3.93 vs 5.12、严格@3 44.3 vs 30.2、宽松 64.2 vs 57.5、Sp 0.210 vs 0.050。注意训练侧 GBDT15 反而略差（4.19 vs 3.85，PROJECT_LOG）而 Rust 端更好 → **增益来自 serve 端鲁棒性**（对 serve 固定 2ps/1fF + 大候选门结构偏移更稳），非训练端精度提升。
3. **仍不达 10.3 主判据**（遗憾 12.19% > 5%、严格@3 44.3% < 90%）→ **GNN 仍只做启发式预排序**；剩余训练 4.19% → Rust 12.19% 落差 = **规模失配 + 候选门结构分布差异为主**（§13.1-4「A1 修 serve 不可行」坐实；GBDT15 缓解非消除）。
4. **两阶段遗憾 3.93% ≤ 5%（中位 0.33%）** →「GNN 粗排前 3 → SPICE 精排」流程实操可接受，粗筛定位成立（回 §10.8/13.x 两阶段结论）。

> 〔⚠ 2026-09-03 16.11.35 修订〕第 2 点「增益来自 serve 端鲁棒性」只对了一半（相对 iaa 成立）；**同 bench 无近似对照（§13.3 v2nowave42m4）表明近似整体净伤害 serve，GBDT15 仅是最不伤的近似**。结论以 §13.3 为准。

### 13.3 v2nowave42m4（纯拓扑无近似）Rust 对照 = 近似 serve 净伤害定案（16.11.35，2026-09-03 19:29）

> 补 §13.2 缺的「无近似」对照：serve 换 v2nowave42m4 mid200 checkpoint（in_dim=14，无近似列确认），完整报告（106 候选集 / 5,390 成功行 / 8 失败，19:29 收尾）。

| 指标 | **v2nowave42m4**（纯拓扑，本次实测） | v2iag42m4（GBDT15，§13.2） | v2iaa42m4（线性，§13 记录⚠） |
|---|---|---|---|
| 严格 recall@3 | 39.6% | **44.3%** | 30.2% |
| 宽松 recall@3 | **65.1%** | 64.2% | 57.5% |
| 选择遗憾 | **10.87%**（中位 6.17%） | 12.19% | 15.21% |
| 两阶段遗憾 | 3.95%（中位 0.69%） | **3.93%**（中位 0.33%） | 5.12% |
| Spearman | 0.142 | **0.210** | 0.050 |

（hi_spread 79/106：nowave 严格@3 38.0% / 宽松 55.7% / 两阶段 5.03% / 遗憾 13.30% / Sp 0.232。）

**定论**（三 run 同 106 集同 pipeline；iaa 选择遗憾 15.21 已由 §14 复算 mean 15.9 吻合，可作 run 级对照）：
1. **选择遗憾 nowave 10.87% < iag 12.19% < iaa 15.21%——无近似的纯拓扑 serve 反而最好**。加近似 ids_avg 特征（无论线性/GBDT15）serve 端是**净伤害**：线性 +4.3pp、GBDT15 +1.3pp vs 纯拓扑。§13/13.1/13.2 的「线性 serve 失配 15.2%」归因修正：iaa 15.21 ≈ 4pp 可避免的近似 serve 噪声 + ~7pp 真·规模失配（nowave 训练 3.80 → Rust 10.87 ≈ 7pp 硬底；§13.1「规模失配是根本」仍成立且被无近似对照坐实）。**⚠ 17.2.6 补**：此 7pp 是**混合总体**口径（全局训练数 vs 106 集 Rust 数）；**同口径（hi_spread 子集 ↔ 79 集）为 ~10pp**，见 §13.7.3-6。
2. **iag 的价值 = 最不伤的近似**，不是「胜过无近似」：排序质量确实三兄弟最高（Spearman 0.210、严格@3 44.3%），但 top1 自选落点输给 nowave（DEPTH_MIX 大候选窗口 w=5/1/7/11 等 nowave 明显更好，如 w=5 n=272：nowave 0.00% vs iag 46.50%）。
3. **两阶段 nowave 3.95 ≈ iag 3.93（均 ≤5%），iaa 5.12 略超标** →「前 3 → SPICE 精排」流程下 nowave/iag 等效。
4. **交付含义**：Rust 粗筛模型应走**纯拓扑 no-wave（nowave）路线**；USE_IDS_AVG_APPROX 特征线对 serve 粗筛整体负贡献（训练侧也白搭，见 PROJECT_LOG 三兄弟训练对比）。牵连：v2kdwave42iaa42 蒸馏学生（§13.4 已单验收口）。

### 13.4 v2kdwave42iaa42（wave 教师 KD 学生）Rust 验证 = 蒸馏转移双端收口（16.11.39，2026-09-03 22:03）

> §13.3 第 4 点「KD 学生 Rust 待单验」落地。**serve 发现（§13.3 之外的重要事实）：v2kdwave42iaa42 虽名带 iaa，ckpt `convs.0.lin_rel.weight` in_features=45 → 无近似列 → 按纯拓扑 serve（不带 USE_IDS_AVG_APPROX）**，与 nowave 同特征布局、唯一变量 = KD 学来的权重 → 是「KD 是否转移 serve 优势」的干净 A/B。完整报告（106 候选集 / 5,390 成功行 / 8 失败，22:03 收尾）。

> **⚠ 更正（根因，非仅 serve 层，2026-09-03）**：学生为何 in=45 —— setup_exp.sh KD 分支设 `USE_TRANSISTOR_WAVE=False`，而**近似 ids_avg 列只在 `if config.USE_TRANSISTOR_WAVE:` 块内构造**（data_loader.py L507 门控 → L510 近似分支；L600 `extra_feats.append(tw_feat)` 亦在块内），wave 一关近似列**训练时就不生成** → 学生实为纯拓扑（= v2nowave42m4 特征布局）。故 **「iaa 学生 + wave 教师 KD」这一格从未真正测过**：名带 iaa、`USE_IDS_AVG_APPROX=1` 均未生效。本次实跑 = **纯拓扑学生 + wave 教师软标签（reg+rank）**，其 A/B 对照 = v2nowave42m4（同特征、唯一变量 = KD 权重），**不是 v2iaa42m4**。

| 指标 | v2kdwave42iaa42（KD 学生，本次） | v2nowave42m4（纯拓扑，§13.3） | v2iag42m4（GBDT15，§13.2） | v2iaa42m4（线性，⚠） |
|---|---|---|---|---|
| 严格 recall@3 | 36.8%（宽松 64.2%） | 39.6%（65.1%） | **44.3%**（64.2%） | 30.2%（57.5%） |
| 选择遗憾 | 14.97%（中位 7.17%） | **10.87%** | 12.19% | 15.21% |
| 两阶段遗憾 | 4.81%（中位 1.03%） | 3.95% | **3.93%** | 5.12% |
| Spearman | 0.098 | 0.142 | **0.210** | 0.050 |

（hi_spread 79/106：kd 严格@3 31.6% / 宽松 57.0% / 两阶段 6.19% / 遗憾 18.94% / Sp 0.184。）

**裁决**：
1. **同 106 集同 pipeline，四 run 唯一变量 = serve 权重/特征布局**（iaa 为记录⚠对照）→ A/B 干净。
2. **KD 学生 Rust 劣于朴素 nowave（14.97 vs 10.87，+4.1pp），仅略优于 iaa 线性（15.21）**；严格@3 36.8 介于 nowave 39.6 与 iaa 30.2 之间、输 iag 44.3。train 侧亦无增益（PROJECT_LOG L750：遗憾 4.17 vs nowave **3.80** —— 因学生实为纯拓扑，正确 train 对照 = nowave，非 iaa）→ **wave 教师（自身 train 遗憾 0.65% 但 Rust 不可部署）经 reg+rank 蒸馏到纯拓扑学生，serve 优势不转移——双端（train + Rust）均不赢朴素纯拓扑训练**。
3. **形态解释（修正）**：学生从头到尾就是纯拓扑（近似列被 `USE_TRANSISTOR_WAVE=False` 门控挡住，训练时未构造），**不存在「iaa 结构污染」**。正确机制 = 信息论天花板（同旧 KD 结论 PROJECT_LOG L609）：wave 教师的全部优势来自学生输入里没有的 wave 特征 → 软标签是「学生够不着」的目标，主损失推真实延迟（可达）、KD 损失拉教师 log10（不可达），两目标竞争 → 双端略负。**软标签无法凭空补缺失输入特征，换教师/调 λκτ 均救不了** → 支持 §13.3 定论「Rust 粗筛走朴素纯拓扑 nowave，蒸馏线对 serve 净伤害」。
4. **两阶段 4.81% 略 >5%**（nowave/iag ~3.9）→ KD 在前 3→SPICE 精排流程下也不占优。
5. **收敛**：m4 四模型 Rust 全部验完，**交付 = v2nowave42m4 纯拓扑**。四兄弟 train 侧排序近似平手但 Rust 分层明显 → **Rust serve 端才是裁决场**（再次印证 §13.3 第 4 点）。

### 13.5 v2iaar42m4（线性近似 + 真值组内 rank 排序直训 #8）Rust = rank 杠杆 train 生效但 serve 不转移（17.0.3，2026-09-04 12:23）

> ⚠ **17.0.13 更新**：本节「rank serve 不转移」结论基于 **iaar 单点（线性近似）**。rank×可 serve 特征两空格（v2nowaver42m4 纯拓扑×rank、v2iagr42m4 GBDT15×rank）已跑完 → **rank 转移 = 特征依赖**：纯拓扑叠 rank 全轴净伤害（11.97 > 10.87，槽位关闭），GBDT15 叠 rank 遗憾破交付基线但严格召回塌方（10.50 < 10.87、严格@3 44.3→34.0）。serve 交付基线不变 = v2nowave42m4 纯拓扑。详见 §13.6。

> §13.4 收敛后 m4 第 5 个 Rust datapoint。#8 = v2iaa42m4 特征 + `RANK_LOSS_W=0.5` 排序直训。serve ckpt midpoint_ep150，`convs.0.lin_rel.weight` **in=46** → 近似列确在，serve 带 `USE_IDS_AVG_APPROX=1`（与 iaa 同特征布局、唯一变量 = rank 权重 → vs v2iaa42m4 是干净 A/B，但 iaa Rust 记录 ⚠ 不可复现，主对照取 nowave）。完整报告（106 候选集 / 5,390 成功行 / 8 失败，12:23 收尾）。

| 指标 | v2iaar42m4（线性+rank，本次） | v2nowave42m4（纯拓扑） | v2iag42m4（GBDT15） | v2iaa42m4（线性，⚠） | v2kdwave42iaa42（纯拓扑KD） |
|---|---|---|---|---|---|
| 严格 recall@3 | 31.1%（宽松 59.4%） | 39.6%（65.1%） | **44.3%**（64.2%） | 30.2%（57.5%） | 36.8%（64.2%） |
| 选择遗憾 | 12.72%（中位 7.71%） | **10.87%** | 12.19% | 15.21% | 14.97% |
| 两阶段遗憾 | **3.64%**（中位 1.15%） | 3.95% | 3.93% | 5.12% | 4.81% |
| Spearman | 0.170 | 0.142 | **0.210** | 0.050 | 0.098 |

（hi_spread 79/106：iaar 严格@3 27.8% / 宽松 53.2% / 两阶段 4.50% / 遗憾 15.94% / Sp 0.216。）

**裁决**：
1. **train-side rank 大优不转移 serve**：train 遗憾 2.94%（PROJECT_LOG 17.0.2，m4 系 train 最优）→ Rust **12.72%，仍输朴素纯拓扑 nowave（10.87，+1.85pp），仅 ≈ iag 12.19**；严格@3 31.1%（宽松 59.4% 五者最低）远逊 nowave 39.6 / iag 44.3。
2. **机制** = §13.1 线性近似列 serve 失配未因 rank 修复：rank 把训练分布上 pairwise 排序练得更锐，但 serve 端候选门结构分布漂移（GNN 粗筛面对的正是候选集）把增益吃回 → **给 serve-hostile 特征加训练强度救不了 serve 失配**。
3. **两阶段 3.64% 五者最低**（hi_spread 4.50 vs nowave 5.03 / iag 4.96）→ "GNN 前3→SPICE 精排"流程下 iaar 是 5 个里最有效的粗筛；但 serve 主判据按 GNN 自选 top1 遗憾（12.72 > 5%）仍不达标。
4. **收敛强化**：m4 五模型 Rust 全验完，**加近似 ids_avg 特征（线性/GBDT15）serve 端即便叠 rank/KD 均赢不了朴素纯拓扑 → 交付基线锁定 v2nowave42m4**。若仍想验证 rank 杠杆能否在 serve 兑现，唯一未试 = **rank loss + 可 serve 特征（nowave 纯拓扑 或 iag GBDT15）**——候选 next、非必须（§十三已收敛，边际预期低）；**17.0.4 已起跑该两空格：`v2nowaver42m4`（纯拓扑×rank）+ `v2iagr42m4`（GBDT15×rank），见 OPERATIONS §5**。

---

### 13.6 rank × 可 serve 特征两空格 Rust：rank 转移 = 特征依赖（17.0.13，2026-09-06 21:22 / 21:31）

> §13.5 基于 iaar 单点曾断「rank 杠杆 train 生效 serve 不转移」。17.0.4 起跑两空格，把 rank 放到 serve 失配更小的**可 serve 特征族**上验（纯拓扑 in=45 / GBDT15 in=46），m4 第 6/7 个 Rust datapoint。两 run 完整报告均 106 候选集 / 5,390 成功行 / 8 失败，A/B 干净。

| 指标 | **v2nowaver42m4**（纯拓扑+rank） | **v2iagr42m4**（GBDT15+rank） | v2nowave42m4（纯拓扑） | v2iag42m4（GBDT15） | v2iaar42m4（线性+rank） |
|---|---|---|---|---|---|
| 严格 recall@3 | 34.9%（宽松 57.5%） | 34.0%（52.8%） | 39.6%（65.1%） | **44.3%**（64.2%） | 31.1%（59.4%） |
| 选择遗憾 | 11.97%（中位 7.24） | **10.50%**（中位 6.91） | 10.87% | 12.19% | 12.72% |
| 两阶段遗憾 | 3.89% | 3.83% | 3.95% | 3.93% | **3.64%**（七者最低） |
| Spearman | 0.094 | **0.238** | 0.142 | 0.210 | 0.170 |

（hi_spread 79/106：nowaver 遗憾 14.80%；iagr 遗憾 13.03%。）

**裁决**：
1. **rank 转移 = 特征依赖，§13.5「不转移」须修订为分族结论**：
   - 线性 iaa + rank（iaar）：12.72%，不转移（§13.5，线性 serve 失配根因 §13.1）。
   - **纯拓扑 + rank（nowaver）：11.97% > nowave 10.87% → 全轴净伤害**（严格 34.9 vs 39.6、宽松 57.5 vs 65.1、Sp 0.094 vs 0.142）——同特征、只加 rank loss 的干净 A/B，train 排序优势零转移。
   - **GBDT15 + rank（iagr）：10.50% < nowave 10.87% = 家族首个 Rust 遗憾破交付基线 + Sp 0.238 家族最高** —— rank 转移成立的第一个 serve datapoint，恰发生在「最不伤近似」上。
2. **但 iagr 的赢点全在排序质量轴、recall 轴塌方**：严格@3 34.0 vs iag 44.3（**−10.3pp，iag 的严格召回强项被 rank 打穿**）、宽松 52.8 处家族最低档。→ rank 把模型从「top1 落点 / 严格召回」重新分配到「组内整体序」；**serve 主判据若含严格召回，rank 是净负**。
3. **机制统一**：rank 直训强化训练分布上的 pairwise 组内序 → 是否转移 serve 取决于该特征族能否吸收 serve 候选门结构漂移。线性（§13.1）与纯拓扑吸收不了（nowaver 全轴负）；GBDT15 的特征一致性（§13.2 佐证）让排序增益部分存活——**但只救排序、不救召回**。
4. **serve 交付基线不变 = v2nowave42m4 纯拓扑**（10.87%、无 env、稳健）。iagr 的 0.37pp 遗憾 + 0.096 Sp 需 GBDT15 serve 的近似脆弱（§13.3）换，还输 5.6pp 严格召回 → 不作交付。**10.3 主判据（遗憾 ≤5% + 严格@3 ≥90%）m4 Rust 全族七 run 全员不达标 → GNN 仍只做启发式预排序**。

---

### 13.7 serve 进程间抖动定因 + 修法 + 两臂 epoch 扫描（17.2.4，2026-09-15）

> 本节是**本文档全部部署口径数字的适用性前提**。此前各节的 Rust 数值（§13.2/13.3/13.4/13.5/13.6、§10.8）都产自 17.2.4 之前，**每份都带未知边序**，请按下方的抖动带解读。

#### 13.7.1 根因：边序随哈希种子变（**不是**并列判定、不是线程、不是精度截断）

同一 ckpt、同一 `XYCE_CACHE`、同一解析脚本，连测多次得 **10.51 / 10.60 / 10.66 / 10.86 / 11.13%** —— 跨度 **0.62pp**。定因链（全部双向对照，非相关）：

| 步骤 | 操作 | 结果 |
|---|---|---|
| 1 | 同一 serve 进程连跑两趟 CSV | **逐字节一致** → 分析器对同一 CSV 是纯函数 |
| 2 | **只换进程身份**（同 ckpt 同 env） | 239/4190 行变化，**全是平均秩 ±0.5 的整跳** |
| 3 | 钉 `PYTHONHASHSEED` 后两个新进程 | **逐字节一致** |
| 4 | 三个线程变量（`OMP`/`MKL_NUM_THREADS`/`MKL_DYNAMIC`）钉与不钉 | **零差别** → 与线程无关 |

机制：`src/graph_builder.py parse_netlist` 原为 `edges = list(set(edges))` —— **字符串元组的集合迭代序是 `PYTHONHASHSEED` 的函数**（钉住则该种子下固定，不钉则每进程不同）→ `edge_index` 行序变（`node_names.index(u)` 建行）→ 进 `GCNConv` 与 `global_add_pool` 的 **float32 累加序**变 → 同一候选跨进程预测差 **float32 的 1 ulp（~1e-7 相对）**。大量候选本就是同一份网表（float32 下位级相等），被这点差撬开 → 平均秩 `(i+j)/2+1` 整组 ±0.5 → top-1 翻转 → 该集 0.00% ↔ 19.92%。

**为什么训练侧一直没暴露**：`data_loader.py` 把 `(node_names, node_static, edge_index)` 落盘到 `cache/graphs/<cid>_graph.pt` 复用 → **边序在第一次建缓存时就冻结在磁盘上**（是意外，不是设计）。serve 无图缓存、每进程重建 → 抖动只在 serve 端显形。

**修法（17.2.4）**：`edges = sorted(set(edges))`（去重内容不变、顺序对任何进程/机器/种子唯一）+ 启动行加 `PYTHONHASHSEED=0` 兜底。**同时撤销 17.2.3 的相对容差**（`TIE_RTOL=1e-9`）—— 它瞄的量级比真实抖动**小两个量级**（1e-9 vs float32 的 1.2e-7），该合的合不上，实测无效（同一 ckpt 两进程仍差 478~814 行）；并列判据已回到历史口径的**位级相等 `==`**。

**验收（不是推理，是实验）**：两个**全新 serve 进程、刻意不钉任何 env**，同 ckpt / scaler / Rust / 输入，各跑一趟 level2 分片 → 9 份 CSV / 4234 行，**md5 清单逐字节相同**。唯一改变过的变量就是那条边序语句 → 通道确认堵住。

> **三条作废的旧解释（勿再引用）**：(a)「`gnn_pred` 只存 7 位有效数字 → 伪并列」—— 它是**平均秩**，值就是 2.5/3.0 这种半整数，7 位一位没丢；(b)「噪声底 0.3pp / 上限 0.38pp」—— 实测跨度 0.62pp，且 0.3/0.38 这两个数及其"单集最大遗憾/106"的推导前提都不成立；(c)「17.2.3 的相对容差已修」—— 见上，量级错。
> **旧数据的读法**：<1pp 当打平、1.5pp 以上才是真差；**17.2.4 之后每个 ckpt 一次即可**（同 ckpt 重跑逐位一致）。

#### 13.7.2 两臂 × 五 epoch 全扫（十趟分母逐项相同 106 集 / 5390 成功 / 8 失败）

表见 `docs/PROJECT_LOG.md` §17.1.x「3.60pp 之谜收口」（十行全表含严格/宽松两口径、选择遗憾与两阶段的中位、79 集子集）。要点：

1. **3.60pp（口径修正后 2.90pp）分解 = 1.61pp 选点错 + 1.29pp 残余**。`42b` 服务的 ep100 **是它五个 epoch 里最差的**（14.03% vs 该臂最优 ep150 12.42%）；`42m4` 服务的 ep250 恰是它五点最优（11.13%）。残余 1.29pp（两臂最优对最优）**未分离**，且 (b) 未被证伪：42b 最晚的 ep250（13.07%）并未跳到 ~10.7%。
2. **臂内选点（1.61–2.65pp）比臂间差异（均值 0.46pp）大一个量级**，两臂区间大幅重叠（12.42–14.03 vs 11.13–13.78）→ **部署选型的杠杆在 epoch，不在换臂**。
3. **两阶段口径有真正的臂×指标交互（反序）**：两阶段前五名全是 42b（2.90–3.57%）、后五名全是 42m4（4.00–4.61%），与 GNN 自选 top1 完全反序，**每对 epoch 组合都不反转** → 是臂级差异。机制 = **42m4 右尾更肥**（「均值−中位」3.50/2.86/2.36/2.82/3.21 vs 42b 2.61/1.50/1.73/2.33/2.20），中位数是 42m4 胜 1 / 42b 胜 2 / 平 2。→ **带 SPICE 精排才选 42b，不带选 42m4。**
4. **`Best midpoint` 无判别力，且判据与部署反序（⚠ 17.2.6 更正判据名）**：`Best midpoint` 由加权 `score`（旧 `train_sweep.py:978`，`100·r3+50·r2+0.3·sp−0.2·regret+0.1·cap`，recall@3 主导）选出，**不是** `BEST_METRIC=smoothed_rel_err`（后者是 **`best_model.pt`** 的判据，两者是不同 selector）。**🔴 17.4.1 补：这套加权 `score` 已于 17.3.7 整体删除**（`train_sweep.py:1113` 的注释即记此换判据），midpoint 选点改为 `score = mrk.get('capture2_pct', …)`（`:1119`）；**17.4.1 起更改为不取 argmax 而取平台末端**（`MIDPOINT_SELECT`，`:1113-1147`）。故本节 4/5 两条的「判据与部署反序（Spearman = −1）」是**已删代码**上的历史实测，**不再是当前代码的风险**。`BEST_METRIC` 亦已更名 `BEST_MODEL_METRIC`。同一 selector 在 42m4 挑中五点最优、在 42b 挑中五点最差（1/2）；进一步查 42b 训练侧逐 epoch（§13.7.3）得**该 `score` 序与 Rust 部署序 Spearman = −1（五点完全反序）**。**不得默认服务 Best midpoint。** 附第二次命中「训练侧 Spearman 高 = 部署差」（42m4 ep200 训练侧 Sp 次高 → 部署 13.78% 该臂最差）。
5. **交付基线不变 = `42m4` ep250（纯拓扑）**；规范序数值 **11.13%**（旧 10.66/10.87 均为 17.2.4 前抽样，带内最差值为 11.13 → 旧值高估 0.47pp）。§13.2/13.3 各行的 10.87/12.19/15.21 等按「17.2.4 前、±0.31pp 不确定」解读，未重测。
6. **达标线仍远**：最好的 11.13% 是「选择遗憾 ≤5%」的 2.2×。本节证明的是**选点依据**，不是可交付性。

#### 13.7.3 训练侧逐 epoch 复核：进步是真的没有 + LR 到底 + 选点判据与部署反序（17.2.6，2026-09-15）

全表与逐条论证见 `docs/PROJECT_LOG.md` §17.1.x「训练侧逐 epoch 复核（17.2.6）」。要点：

1. **目标是 delay，损失里没有排序项**：`src/train_sweep.py:139-146`（17.4.1 起；旧记 `:84-91` —— 17.4.1 在文件顶插 55 行，第 24 行以下行号整体 +55） —— `log10(data.y)` 逐行、Huber(delta=0.3)、按开关脚 `PIN_WEIGHTS` 加权取均值；`RANK_LOSS_W=0`、KD 关闭（纯拓扑臂）。**组内排序只是被测量的量，不是被优化的量**（回归副产品）→ 它在排序轴上早饱和是**必然而非意外**。（另一条 idsavg GNN 线目标是 **ids**、指标 val_R²，与本节的 delay GNN 无关。）
2. **训练侧也平且非单调** → §13.7.2 那条 Rust 平线**是如实的**，不是"训练增益不转移"。`42b` 训练侧 regret 5.86 / 5.37 / 5.25 / 5.79 / 5.48（全距 0.61pp），**ep100 之后排序轴零净进步**。
3. **LR 到 ep250 恰在 `LR_MIN=1e-6`，Val Loss 在 ep150/200/250 三点同为 0.0206，Val Rel Err 在 ep150 触底（29.14%）后回升** → **泛化在 ep150 停止，之后是轻度过拟合**（Train Loss 仍缓降至 0.0083）；本次 run 是跑完、不是被截断。
4. **四个独立信号在 ep150 交汇**：训练侧 regret 唯一最小 5.25% / Val Rel Err 唯一最小 29.14% / Val Loss 底线首达 / **Rust 选择遗憾唯一最小 12.42%**。**唯一不指向 ep150 的是选点 `score` —— 它挑了 ep100，而 ep100 是 Rust 全表最差（14.03%）** → ep150 是该臂真实局部最优。
5. **判据与部署口径反序**：`score` 序 vs Rust 序 **Spearman = −1**（五点完全反序）。~~同名端点也反转（训练 r3 最高 ep100 → Rust 严格 k3 最低 32.1%；训练 r3 最低 ep150 → Rust 严格 k3 最高 34.9%）~~ **⚠ 17.2.7 撤回此句**：两端 recall@3 是**两个不同源的估计量**（总体 / 并列约定 / 被排序的量三项都不同），且 106 集下 1 SE ≈ 5pp（1 集翻转 = 0.94pp）→ 那点"反转"在噪声内，**不构成证据**；见 §13.7.4。机制 = `score` 被权重最大、却最不转移的 recall 主导。**⚠ n=5，五点全反序可能纯属巧合，不作定律**；另有 `42b` ep100 / `42m4` ep200 两处独立同向命中。
6. **同口径硬底是 ~10pp，不是 ~7pp**：训练 `hi_spread` 子集 5.25–5.86% ↔ Rust 79 集 15.25–17.48%。§13.2 的 7pp 是"全局训练数 vs 106 集 Rust 数"（混合总体）—— 不算错，但不可当同口径硬底引用。
7. **⚠ 更正 §13.7.2-4 的判据名**：`smoothed_rel_err` 选的是 `best_model.pt`，`Best midpoint` 是那套加权 `score`（两个 selector）。推论：`best_model.pt` 大概率落在 **ep150 邻域**且**从未被任何一趟扫描扫过**（扫的全是 `midpoint_ep*.pt`）→ 值得列入候选（待验）。

#### 13.7.4 两端 recall@3 不可对读 + 判别量并列免疫（17.2.7，2026-09-15，代码已落地）

**动机**：§13.7.3-5 撤回了「同名端点反转」的过度断言后，必须回答"那两端 recall 到底差在哪、哪个量才该拿来选点"。逐条查代码后的结论：

1. **三处不同源**（这就是"不可对读"的全部内容）：
   - **总体**：训练侧 = test split 的 `spread>10%` 组（10³ 量级组数）；Rust 侧 = 106 个部署集（hi_spread 子集 79 集）。69–73% vs 32–35% **主要是总体差，不是能力差**。
   - **被排序的量**：训练侧排**原始预测均值**（`utils.py` `np.argsort(pr)`）；serve/Rust 排**候选集内平均秩**（`serve.py` `predict_rank_batch` 的 competition rank `avg=(i+j)/2+1`，写进 `avg_delay` 字段）。**单模型下单调等价、集成下不等价** → 前者只是后者的代理。
   - **并列约定**：训练侧原为默认 quicksort（无序保证），Rust 侧 `argsort(kind="stable")` 按 `eval_idx` 行序 → **已统一（见 3）**。
2. **部署侧分辨率**：106 集、p≈0.35 时 1 SE ≈ 5pp，**1 集翻转 = 0.94pp** → 跨 epoch 的几个 pp 差异在 ~1–2 SE 内，**"哪个 epoch 的部署 recall 更高"本轮样本量下不可判**（与 17.2.4 消掉的抖动不同，这是样本量）。
3. **`regret_2stage` 并列免疫，且与严格 recall@3 逐集恒等**：`_shadow_analyze.py` 的 `top3_true_best = true[order_g[:3]].min()` 取前 3 名内真值最小值 → 集合内部次序不影响结果；且 **严格 recall@3 == (regret_2stage == 0)**。→ **带精排路线下，选点/优化目标 = 严格 recall@3**（该量对并列免疫，是最干净的判据）。
4. **17.2.7 代码统一**：`src/utils.py` 的 `_spearman` 与 `ranking_metrics`（`ord_pred`/`ord_true`）全部加 `kind='stable'`（同一条并列规则＝组内行序；无并列时行为不变，已本地核对）。`_shadow_analyze.py` 更正过期注释（「gnn_pred 只写 7 位有效数字 → 伪并列」**该机制不存在**：gnn_pred 现在是**秩**，7 位可精确表示；0.00%↔36.34% 翻转的真正源头是 float32 边序抖动，17.2.4 已定序），并新增**五个诊断位**：重复网表集 / 预测第3小==第4小 / 预测有重复值 / **量纲混合守卫** / **兜底窗口**。`serve_http.py` docstring 更正：`avg_delay` 字段**不是延迟**。
5. **「量纲混合」为何现在不会发生**：兜底路径返回**原始延迟**、命中路径返回**秩**，但 `tl_opt.rs` Pass2 `pre_rank` 与 Pass3 `evaluate` 用**同一批 `prepared`** → 一个窗口要么全命中要么全兜底，**集内不混**；单模型下兜底窗口按原始延迟排序与按秩排序等价 → 该集 recall/regret 仍有效（仅分辨率受 CSV 7 位有效数字限）。诊断位把该推理钉成可断言输出。
6. **验证**：`scripts/diag/_t_shadow_analyze_2col.py` 双列回归通过；另用构造数据（并列/兜底/混合/重复四类）确认五个诊断位逐个按预期触发（探针脚本跑完即删，未入库）。

### 13.8 serve 输入退化定位：晶体管数 + 门数，load 打平（17.3.4，2026-09-16）

**动机**：§10.7/§13.1 只做到"serve 与训练的特征失配存在"，没量化过**每一处失配各值多少**。本节用一次 13 变体消融把它拆开，并给出退化量的正确口径（修正本节旧探针的错误口径）。

**方法**：`scripts/diag/_serve_input_ablation.py`（17.3.2/17.3.3 交付，跑在 `~/abl-run` 链接工作树，钉 `56b2fdd`）。同一份 `midpoint_ep250.pt`、同一 scaler、同一份 `test_df`，逐行重建 serve 侧输入 → 前向 → 用训练侧 `ranking_metrics` 口径读数。**548 组 / 346 组 hi_spread / 111,364 行 / 13 变体 / 前向 988s。**

**闸门（先过再读）**：`base` 变体（= 训练侧全真输入）对本臂已存的 `test_predictions.npz` 逐位比对 —— **中位相对差 0.000e+00、p95 2.252e-06、max 6.564e-06 → ✅ 复刻成立**（float32 npz，阈值 1e-6）。即 `base` 的绝对值可与该臂历史数字直接对读，下表 Δ 才有意义。

#### 13.8.1 三处退化的定义（代码级）

| # | 量 | 训练侧（口径真值） | serve 侧（现状） |
|---|---|---|---|
| ① | `circuit_sig` | `[#X_ 网表行数, transistor_count, #input_pins]`（`data_loader.py:314-325`，逐电路算） | `[num_nodes, 0.0, len(pins)]`（`serve.py:281`）—— **两项都错**：[0] 是图节点数≠`X_` 行数（可复算，见 ①② 分解）；[1] 恒 0 |
| ② | `struct_prior` | `[transistor_count, #{g∈ct: 'SC_AND'∈g ∧ 'SC_AND_'∉g}, #{g∈ct: 'SC_INV_WIRE'∈g}]`，`ct=cell_types_json`（`data_loader.py:710-719`，`USE_STRUCT_PRIOR=True` @ `config.py:138`；17.4.1 前为 `:121`） | **整个传 `None`**（`serve.py:286`）→ `model.py:111` 的 `if struct_prior is not None` 为假 → 结构残差整块静默跳过 |
| ③ | 动态 `load` | 逐行 `pin_load_json`（中位 1.2e-15） | 常量 `LOAD_F=1e-15`（`serve.py:117 load = global_load`） |

**①② 的放大器**：`sig_encoder`/`struct_encoder` 都是**未归一化**的裸 `Linear(3, hidden//4)`（`model.py:39-54`），量级 44–132 → **错量级比缺值伤得多**（见 13.8.3 的 recon 行）。

#### 13.8.2 全量读数（548 组；recall 两口径都给）

| 变体 | 选择遗憾 mean | Spearman | R2 严\|宽 | R3 严\|宽 | hi_spread 遗憾 | Δ遗憾 |
|---|---|---|---|---|---|---|
| `base`（全真输入） | **3.79%** | 0.397 | 56.6\|67.3 | 63.9\|83.4 | 5.17% | — |
| `serve`（现状三处全退化） | 5.86% | 0.159 | 42.5\|53.8 | 47.8\|69.3 | 7.98% | **+2.06pp** |
| `loadconst`（只改 load=常量） | 3.70% | 0.395 | 55.5\|69.0 | 64.1\|82.3 | — | **−0.09pp** |
| `csig0`（只 `csig[1]`=0） | — | — | — | — | — | +0.75pp |
| `csig_nn`（只 `csig[0]`=num_nodes） | — | — | — | — | — | +0.22pp |
| `csig_serve`（`csig` 两项都按 serve） | — | — | — | — | — | +0.75pp |
| `noprior`（只 `struct_prior`=None） | — | — | — | — | — | +0.68pp |
| `serve_noload`（serve 但 load 用真值） | — | — | — | — | — | +1.69pp |
| `serve_fixTC_prior`（真 csig + 真 prior，**load 仍常量**） | **3.70%** | — | — | — | — | **−0.09pp** |
| `serve_fixTC_recon`（csig/prior 用**网表复算**的 tc） | — | — | — | — | — | +2.23pp |
| `serve_fix_recon`（csig/prior/load 全用复算值） | — | — | — | — | — | +8.60pp |

`serve` 相对 `base` 的其余分项：**Spearman −0.238、R2严 −14.05pp、R2宽 −13.5pp、hi_spread +2.80pp、hiR3严 −16.76pp。**

内部自洽性（同配置必须同数）：`base` ≡ `csig0_fixTC`、`loadconst` ≡ `serve_fixTC_prior` —— 两组逐项相同 ✅。

#### 13.8.3 读数

1. **退化几乎全部由单一量 `transistor_count` 造成**，它喂两个地方：`csig[1]`（`data_loader.py:322`）与 `prior[0]`（`data_loader.py:716`）。**`serve_fixTC_prior` = −0.09pp（与 `base` 打平）= 完全恢复** —— 真 csig + 真 prior 一给，2.06pp 全回来。单因子分解：`csig` 两项合起来 +0.75pp（其中 `csig[0]` 单独 +0.22、`csig[1]` 单独 +0.75 → **次可加**，两项重叠）、`prior=None` 单独 +0.68pp，两项相加 1.43pp **< 总 2.06pp**，差的 ~0.63pp 来自交互 —— 即 `struct_prior` 残差只在 `csig` 也对时才完全启用（`model.py:49-54` 两条支路同源进残差）。
2. **`csig[0]` 是独立的小项且 serve 可复算**：训练口径就是**网表 `X_` 行数**（`data_loader.py:318` 的 `n_gates`），serve 只要数行即可，值 **+0.22pp**，不需要 `transistor_count`、不需要 Rust 改动 → **可以先单独上**。
3. **`load` 是打平，不需要动**：从 `base` 侧单独改 load 是 **−0.09pp**（略好）；从 `serve` 侧单独把 load 换回真值是 **+1.69 → 相对 serve 改善 0.37pp**。**两个方向不一致，且都在 ≤0.37pp** → 判为并列，不做工程。机制上本应如此：`(LOAD_F − mean)/scale = −0.15`，**在训练分布的一个标准差内**。`arrival` 两侧都是死特征（scaler `mean 0`/`scale_=1.0`，`arrival_time_*` 列根本不存在），同样不用管。
4. **⚠ 近似值比缺失更糟（`serve_fixTC_recon` = +2.23pp，比不修的 `serve` 还差）**：网表复算的"晶体管数"相对真值**中位相对误差 95.6%**（n=111,364）。因为 ①②两条支路都是未归一化的裸 `Linear`，喂一个量级错的数比喂 0 / 跳过**更伤**。→ **在拿到真值之前，`struct_prior` 必须保持 `None`，不要用任何估计值填。**
5. **旧探针的口径错误（本节修正，勿再引用）**：17.3.2/17.3.3 的探针把 `transistor_count` 当成"`sc_expansion.json` 里 ASAP7 展开的 Σ `n_t`"去复算 → 得到 95.6% 误差并一度怀疑数据。**正确口径是 SC 网络（Join/Inv/Bridge/And/Or）的 `M_` 实例总数、含实例级去重**，Rust 侧就在 `NetlistOpt/src/utils/transistor_count.rs:69 count_transistors_instance_on_egraph`（语义见 `tests/transistor_count.rs:21-52`：`(join a b)`→2、`(! (join a b))`→4、`(+ a b)`→2、`(bridge×5)`→5），与 `DATA_SPEC_V2.md:65` 的定义一致。**判据**：数据侧 `tc / 顶层 X_ 门数` 中位 **7.33（batch_v2_full）/ 4.12（batch_v2_m4）**，而 ASAP7 展开口径同一行是 `92/6 = 15.3` —— **差 3~5 倍**，数据集用的绝不是 ASAP7 口径。且 1317/3822 个 `(expr, tc)` 组合在 **nx 不同时 tc 相同** → 该量不是门数的函数，与去重感知的结构计数相符。
6. **可复算的两项已全量核对（17.3.5 闸门，n=14,727 = full 8679 + m4 1800 + io 4248）**：用训练原样字符串模式在网表唯一 cell 名上复算 `prior[1]/[2]` → **100.0000% 相等**（corr 1.000，且 ~92% 为 0）；`n_gates`（`X_` 行数）同样 **100.0000%**。⚠ **但 `set(cell_types_json) == set(网表唯一 cell 名)` 只有 98.6148%**（full 99.5852% / m4 100% / io 96.0452%）—— 17.3.2 的 100% 是 **548 组 test split** 上的，不是全量。204 个例外**全部是同一方向且无害**：`cell_types_json` **多**含一个 `SC_JOIN_BOOL0_BOOL0` / `SC_JOIN_BOOL1_BOOL1`（常量折叠出的 join，网表 `X_` 行里不出现该名），**从不缺失**，且该多出的名字**不匹配两条 prior 模式** → 所以计数相等在 14,727 行上仍成立。**结论**：这两项 serve 端可直接复算，无需 Rust 配合；`prior[1]/[2]` 的 100% 是**实测事实**而非结构性保证（若生成方日后产出匹配模式的额外类型，需重测）。
7. **⚠ 方法论：n<100 组的消融只能验管道 / 做二值判定，不能给因子排序。** 20 组 smoke 曾给出 `loadconst +1.54pp`（"load 是主因"）与 `csig_nn +0.00pp`，全量后反转成 **−0.09pp** 与 **+0.22pp**。当时已有反向信号（`serve_noload` 比 `serve` 差，而 `base→loadconst` 说相反）**被我标注但没据此行动** —— 方向自相矛盾本身就是样本量不足的信号。二值判定（如"网表复算值是否等于真值"）不受此限，因为它不排序。

#### 13.8.4 落地（对应 §11 清单外，按用户指定纳入）

根因是 **Rust→serve 的 payload 少了一个字段**，而 Rust 侧**本来就在算它**：`tl_opt.rs:214-217` 的 `FullCircuitTlEvaluator::evaluate` 从 `module.to_recexpr()` **同时**算出 `transistor_count` 与 `avg_delay`（= 训练标签 `y`），两者同在一个 `TlEvalMetrics` 里。`gnn_shadow.rs:22-33` 的 `GnnCandidate` 只发了 `id/netlist/input_pins/output_pins/gate_logics`。

- **Rust**：`utils/transistor_count.rs` 加 pub 包装 → `gnn_shadow.rs` 的 `GnnCandidate` 加 `transistor_count: Option<u64>`，在 `module_to_candidate` 里填（一处覆盖 `pre_rank` 批量与 `evaluate` 兜底两路）；算失败给 `None`（**不是 0**）。
- **Python**：`serve_http.py:54` 透传；`serve.py:281` 的 `csig[0]` 改数网表 `X_` 行数、`csig[1]` 用收到的 tc；`serve.py:286` 有 tc 才传 `struct_prior`，`prior[1]/[2]` 按 13.8.3-6 复算。
- **向后兼容、可解耦上线**：payload 无该字段 → `None` → `csig[1]=0`/`prior=None` = 今天行为。**Python 先上（含白拿的 +0.22pp），Rust 后上。**
- **预期**：恢复到 `serve_fixTC_prior` 的 **−0.09pp**。
- **残余风险 + 闸门**：生成器（外部工程，只以二进制提交数据，见 `V2_DATA_FULL_DELIVERY.md`）写出的 `transistor_count` 是否**字面等于**上述例程输出，本仓库读不到其代码，属**强证据推断**。间接证据：`tests/transistor_count.rs:105 aligns_with_fixed_point_joinlike_on_singleton_egraph` 说明提取器 cost 与该例程同口径，而生成器走的就是提取器；§12.1 那张表已把两批数字按同口径横向比过。**闸门**：serve 记录收到的 `(netlist, tc)`，离线与 `batch_v2_io`/`batch_v2_m4` 的 `gate_level_netlist` 精确匹配（这两批即由该套 testbench `.tl` 家族生成）后与 parquet 的 `transistor_count` 对读。

---

## 十四、内容数据支持审计（2026-09-03，逐节核对）

> 按「先完整验证再下结论」规则，对本文档各节声明做了可复现性核查：✓ = 保留数据/存档可复现；⚠ = 与现数据不符或不可独立复现，需标注/复核；历史节（run 产物 CSV 已被后续覆盖）标「历史，不可复现但同源印证」。

| 节 | 声明 | 验证方式 | 状态 |
|---|---|---|---|
| §1-3 | cell 638 类 / Rust 7 类两套命名等早期快照 | 历史（2026-08），§7 已解决 | 历史 ✓ |
| §7.2-7.6 | 结构特征实验（logic 1.72/1.78% 等） | PROJECT_LOG/REQUIREMENTS 同源（STRUCT_MODE=logic_only 默认） | 同源 ✓（未逐表复验） |
| §8.2 | 2ps slew / 1fF load / 每 (out,pin) 1 vector | asap7.sp + design.rs 源码实测（16.11.28） | ✓ |
| §8.2 | **「所有输入 t=0 同时翻转」** | tb_*.sp 实测：**仅 switching pin 翻转**（接 VSTIM），其余固定 vdd/gnd；翻转沿 20-22ps | ⚠ **与实测矛盾，应标注修正**（spec 已修，本行未标） |
| §10.8 | 89 集 / 严格@3 47.2% / 宽松 79.8% / 两阶段 2.02% / Spearman 0.140 | reports/_shadow_bench_final.txt（存档完整） | ✓ 完整复现 |
| §12.1 | 训练 trans（full 52/96、rest 58/184、m4 58/118） | 本地 parquet 复算 | ✓ 完全一致 |
| §12.1 | ADD4_OVF 顶层门数 59 | dut 实测 X_ med 57 | ≈ ✓（口径差） |
| §12.2 开头「多输出非主战场/单输出为主」 | **归因错误（2026-09-03 16.11.36 修正）** | 46 电路实测 35 单/11 多；但按 shadow 候选集多输出占 57%（115 集）/52%（89 集）、候选数 52% → 多输出是 serve 半壁江山，且恰含权重最大 ADD4_OVF (9,6) | ⚠ 已撤原句，正因 = 「形状对、规模缩水」 |
| §12.2/12.2b-e | m4 缩水 / 深度档 / vector / 宏层 / 敏感对 | 16.11.23-29 实测（同源脚本） | ✓ |
| §12.3 | **temp_sim_test 可恢复 dut = 1,253** | 实测 dut_expr_*.sp = **1,135** | ⚠ 差 118，待核 |
| §12.3 | 881 组 / 714 有效 / 556(≥8) / ~5,391 行 | 当前 CSV 复算（881/714/556；行 5,391→5,398 微差） | ✓ |
| §12.4 缺陷1 | 训练深度全 ≤5 | **本地首次实测**：full/rest/m4 深度 max 4-5、0% >6（io max 13 但非训练） | ✓ 实测确认 |
| §12.4 缺陷4 | Rust 相邻差 <5% 占 55% / med 4.4% | 12.2c 四档 20/35/39/6（<5%=55%） | ✓ |
| §12.5 #7 | serve 输出排名非延迟（med 4.0 vs true 1.8e-11） | CSV gnn_pred 实测为排名（1.0-10 含 .5 秩平均） | ✓ 方向确认 |
| **§13** | **v2iaa42m4 Rust：106 集 / 严格@3 30.2% / 宽松 57.5% / 遗憾 15.21% / 两阶段 5.12% / Spearman 0.050** | 当前 CSV（=该 run，5,398 行 9-2 19:14 未被覆盖）换分组键复算：714 集 严格@3 **56.7%** / 宽松 **94.4%** / 遗憾 mean 15.9%（吻合 15.21） | ✅ **2026-09-16 销案（原判 ⚠ 反了）**：**106 集 = `(circuit, window_try)` 池化口径（历史）**，**714 集 = `(circuit, iter, window_try)` = 部署真值**（§15.1 代码级定义）。旧判「106 = 标准 pipeline、714 = 另类聚合」是**倒置**：池化把 ~6.7 个批并成一集、集内混不同批的秩（`gnn_pred` 是**批内平均秩**，`serve.py:358-398`）→ 池化侧才系统性失真。**recall 差 26.5pp 是口径差，不是记录错**；遗憾两口径差 0.7pp（对口径不敏感）。 |
| §13.2 | **v2iag42m4（GBDT15）Rust：106 集 / 严格@3 44.3% / 宽松 64.2% / 遗憾 12.19% / 两阶段 3.93% / Spearman 0.210**（GBDT15 serve 加载确认，19:09 完整报告） | 本次 pipeline 实测；与 v2iaa 记录同 106 集/5390 行/8 失败 → A/B 唯一变量 = serve 模型 | ✓ run 级对照有效（裁决见 §13.2，第 2 点经 §13.3 修订） |
| §13.3 | **v2nowave42m4（纯拓扑无近似）Rust：106 集 / 严格@3 39.6% / 宽松 65.1% / 遗憾 10.87% / 两阶段 3.95% / Sp 0.142**（serve in_dim=14 无近似列确认，19:29 完整报告） | 本次 pipeline 实测；同 106 集同 pipeline → 与 §13.2 iag 干净对照 | ✓ run 级对照有效（裁决见 §13.3：无近似 serve 选择遗憾最优） |
| §13.4 | **v2kdwave42iaa42（KD 学生，名带 iaa 实纯拓扑）Rust：106 集 / 严格@3 36.8% / 宽松 64.2% / 遗憾 14.97% / 两阶段 4.81% / Sp 0.098**（serve ckpt in_features=45 无近似列确认，22:03 完整报告） | 本次 pipeline 实测；同 106 集同 pipeline、serve 全纯拓扑 → 与 §13.3 nowave 干净 A/B（唯一变量 = KD 权重；KD 分支 `USE_TRANSISTOR_WAVE=False` 把近似列也挡掉，学生训时即纯拓扑，见 §13.4 更正） | ✓ run 级对照有效（裁决见 §13.4：KD 双端不赢朴素 nowave） |
| §13.1 | 7,105 门(500 dut) / 3,685 门(300 电路) / 负值 6.1% vs 0% / KS p=3e-41 | 脚本 _quant_serve_approx_mismatch.py 存在，输出未存档 | 待核（重跑脚本可验，与缺陷 3 自洽） |
| **§13.7**（2026-09-15 补核） | **抖动根因 = 边序随 `PYTHONHASHSEED` 变；修法 `sorted(set(edges))`；两臂十点扫描** | ① 三组双向对照（同进程逐字节一致 / 只换进程 239 行 ±0.5 整跳 / 钉种子后一致 / 线程变量零差别）；② 验收实验：两个不钉 env 的新进程 level2 产出 9 CSV/4234 行 **md5 清单逐字节相同**；③ 十趟分母逐项相同（106/5390/8） | ✓ 已定因并验收（`scripts/diag/_t_serve_repro.sh` 可复跑）；**由此 §13.2/13.3/13.4/13.5/13.6 与 §10.8 的数值一律按「17.2.4 前、±0.31pp 不确定」解读**，本行即该标注的落点 |

**修正动作**：① §8.2「t=0 同时翻转」加矛盾标注（指向 12.2d）；② §12.3 dut 计数 1,253→实测 1,135 标注；③ §13 表格加「⚠ 无法从保留 CSV 复现，待 v2iag42m4 Rust 重验」标注；④ §13.4 新增 KD 学生 Rust（蒸馏转移双端收口）；⑤ §13.4 更正根因：KD 分支 `USE_TRANSISTOR_WAVE=False` 把近似列门控挡掉 → 学生训时即纯拓扑（in=45），「iaa 学生+KD」这格从未真正测过；train 对照改为 nowave（3.80），裁决③机制从「iaa 污染」改为「信息论天花板」（软标签不可达）。（16.11.40）

**⑥ 2026-09-16（§15）：③ 那条 ⚠ 销案，但结论反了** —— §13/§14 与 OPEN_ISSUES I1 原判「106 集 = 标准 pipeline、714 = 复算另类聚合」，实为**倒置**：714 集才是部署真值，106 集是池化口径。连带两条新标注：**（a）** §13.7 的「±0.31pp 不确定带」由**遗憾**标定，**对严格 recall@3 未被验证**（同 arm 同标称 ckpt 的两处记录差 4.7pp，见 §15.5，成因未收口）；**（b）** 历史表的 epoch 标注至少有一处错（同 §15.5）→ 跨 ckpt 的结论一律冻结待 batch 口径重测。

---

## 十五、部署口径定案与 17.3.11 复算（2026-09-16）

> 起因 = §14 里那条 ⚠「§13 v2iaa42m4 的 recall/Spearman 无法复现」。**定案：不是数据错，是口径错——而且错的是当时以为对的那一边。** 代码动作 17.3.9（分组键）/ 17.3.10（跨集重复度）/ 17.3.11（四口径对照表 + 输入污染检查），脚本 `scripts/diag/_shadow_analyze.py`（`--group-by {batch,window}`，**默认 batch**）；本次读数 = v2nowave42m4，serve 8000，46 个 CSV 快照（09-15 21:03–21:05），`run_stamp` 打印 `midpoint_ep250.pt sha1=09be9a3c53b90760`。

### 15.1 分组键：「一个候选集」= 一次 `pre_rank` 调用

- **`gnn_pred` 列存的不是延迟，是批内平均秩。** `serve.py:358-398 predict_rank_batch`：n≥2 返回组内 competition rank（1 基、并列取平均）再跨模型平均；n=1 返回**原始延迟**（~1e-11）。**两种量纲写进同一列** → 秩只在它自己的批内可比。
- **`pre_rank` 每个窗口迭代恰调一次**：`tl_opt.rs:893 evaluator.pre_rank(&mods, eval_idx)`；`prepared` 在 Pass 1 累积、Pass 3 被 `for (candidate_try_idx, patched, fragment) in prepared`（`:896`）**move 消费** → 借用检查器保证每窗口迭代重建。
- ∴ **GNN 的「集合」= `(circuit, iter, window_try)`**（`window` = `window_try_idx`，重试计数、每轮从 0 重开，`gnn_shadow.rs:269/277`；`iter` = `total_iters`，`tl_opt.rs:577/581/582`）= `--group-by batch`。旧键 `(circuit, window_try)` = `--group-by window`，是**历史池化口径**：把同电路的 ~6.7 个批并成一集（5,398/106 = **50.9 候选/集**），集内混了不同批的秩。
- 单位数：**714 集（batch）/ 106 集（window）**。

### 15.2 四口径对照读数（17.3.11；**17.4.1 错标数更正**）

> ⚠ **2026-09-17 更正**：本节初版整张表（①–④ 六列）取自**错标树** `tl_opt_batch_keep_0916`（= `~/shadow_archive/unknown_20260916_223140`，服务的是**另一个模型**），非 `42m4 midpoint_ep250.pt` 的数。下表已换成**新树复测**（PROJECT_LOG「部署口径定案 714 集 + 交付基线读数刷新」块，同一 ckpt sha1 `09be9a3c53b90760`、`true_delay` 指纹 `cb0d7a96e6aada05` 三树全等）。旧值：① 53.1/**74.4**/69.6/97.6/**7.23**(3.88)/1.66。**行内 epoch 间比较不受影响**（同一棵树内），受影响的只有这些**绝对值**。

714 集 / 5,390 成功行 / 8 失败行 / 小集（<4 候选）167 —— 前三项与 §13.2–13.6 各 run **逐项相同**（跨 5 个模型 run 稳定）。① 是本报告此后默认口径：

| 口径 | 单位数 | 严格 k2 | 严格 k3 | 宽松 k2 | 宽松 k3 | 选择遗憾 | 两阶段 |
|---|---|---|---|---|---|---|---|
| **① 全部批（部署真值）** | **714** | **71.8%** | **90.8%** | **83.3%** | **97.9%** | **5.93%**（中位 2.13%） | **0.63%** |
| ② 按候选池去重（同电路同 sig） | 161 | 67.7% | 80.1% | 85.1% | 96.3% | **3.75%** | 0.56% |
| ③ 按电路宏平均（每电路等权） | 18 | 63.8% | 81.5% | 81.1% | 96.1% | 5.34% | 0.89% |
| ④ 排除最大电路 `level2/DEPTH_MIX` | 176 | 65.3% | 76.7% | 83.0% | 91.5% | **3.03%** | 0.52% |

其余 ① 读数：**Spearman 0.644（n=714）**、hi_spread 591 集、小集（<4 候选）167；**跨集重复度与旧树逐位相同**（独立集 161/714、重复 553 = 77.5%，它只依赖真值列 ⇒ **与 ckpt 无关，是结构量**）。⚠ **cap2 95.55% 与输入污染三项（含诊断位 31/20/271/0/0、hi 子集 14/4/0/0）本次 `17.4.1` 未复测** —— 它们是**旧树那轮**的读数，`PROJECT_LOG` 已标「待补」，引用时须注明来源未复测。其中**量纲混合守卫 0/714**——守卫比对的是「一个集内秩 vs 原始延迟」的混装，而**集内不可能混**（一个窗口要么全 pre_rank 命中、要么全 evaluate 兜底），故它 0 是正确读数，**不构成对池化失真的检测**。

③ 只有 **18 个单位**（不是 46）：46 个 CSV 里 28 个一个合格集都没有，且 ③ 的冠军是 **ep50**（见 `PROJECT_LOG.md`「(6) 口径之争的实证」十点表 —— 该表**是正确树**的数，① ep250 = 90.8% 与本节吻合；③ 冠军 ep300 87.4%、④ 冠军 ep50 80.1%，与 ① 的单调趋势不一致）。④ 与 ① 可反解最大电路自身质量：严格 k3 = (90.8×714 − 76.7×176)/538 = **95.4%**（高于全局 90.8%），选择遗憾 = (5.93×714 − 3.03×176)/538 = **6.88%**（比全局 5.93% **更差**）→ **DEPTH_MIX 的问题在「权重」不在「质量」**（538/714 = 75% 权重压在 1/18 个电路上）。⚠ 但注意反解出的 95.4% 是**它自己那个池被反复评估 64 次**的平均，不是 64 个独立电路的均值 ⇒ 这个「质量好」本身也带 ① 的重复计权。

### 15.3 输入污染检查：三项全 0

分组内 eval 去重丢弃行 **0** / 同电路 eval_idx 跨分组重复行 **0** / 同一 eval_idx 真值冲突 **0**。这是「714 能不能当数」的闸门：shadow CSV 是 `append(true)` 打开（OPERATIONS `:177` 记过一次叠行事故），若快照被叠过行，714 就只是重复计数。**全 0 ⇒ 714 是真正不同的 `pre_rank` 调用次数**，可当调用频次用。

### 15.4 跨集重复度（17.3.10）：714 集只有 161 个不同的候选池

独立集 **161/714**、重复集 553（**77.5%**），最大重数 **64**；真包含子集 1。重灾区：`DEPTH_MIX` 538→26、`COMP4` 65→51、`OR2` 39→14、`ENC8` 12→10。判同用**真值列** `(true_delay, transistors)` 多重集逐字节比对（模型无关：同 netlist → 同预测，也同 Xyce 结果 → 行逐字节相同）。

⇒ **① 的 714 个单元不独立**（同一池被反复评估），任何按集计的比例量都带着这个加权。② 就是给这个问题的下界答案，**两者须并列读**。

### 15.5 ⚠ 口径效应不是常数 ⇒ 旧口径的跨臂比较不可外推

同一 arm、同一 ckpt，**换分组键**（iaa 行是同一次 run 只换键；nowave 行两侧来自两次 run，见表下取数说明）：

| 臂 | window 口径 严格@3 | batch 口径 严格@3 | Δ | 选择遗憾 Δ |
|---|---|---|---|---|
| v2iaa42m4 | 30.2% | **56.7%** | **+26.5pp** | +0.7pp |
| v2nowave42m4 | 34.9% | **90.8%** | **+55.9pp** | **−5.20pp** |

> ⚠ **2026-09-17 更正**：nowave 行的 batch 侧原记 **74.4% / −3.90pp**，是**错标树**的数（见 §15.2 头注与 §15.8）→ 已换为 **90.8% / −5.20pp**（同 ckpt `midpoint_ep250.pt` sha1 `09be9a3c53b90760`）。**「口径效应因臂而异、且 nowave 远大于 iaa」这个结论不变，而且更强**（+55.9 vs +26.5）。**注意方向**：nowave 换到部署口径后严格@3 涨 55.9pp 而遗憾同时**改善** 5.20pp ⇒ 按下方判别规则是**真实能力**，不是口径产物。

> 表的取数说明（**别抄错**）：iaa 行是**同一份 CSV、同一次 run** 只换分组键（30.2% 记于 §13，56.7% 由 §14 复算），干净的单变量。nowave 行的 window 侧 **34.9% / 11.13%** 取自 **09-15 十点扫描**（`42m4` ep250，17.2.4 规范序，PROJECT_LOG 十点表），batch 侧取自 **09-15/16 本次 run** —— 是**同 ckpt 两次 run、两种分组**，不是同一次 run 的两次分组；且该 ckpt 的 epoch 标注本身存疑（§15.5 下段）。**注意别用 `42b` ep250 那行（宽松 63.2% / 遗憾 13.07% / 中位 6.78%）来算 nowave 的差** —— 严格@3 恰好同为 34.9%，极易抄错列。

**失真量 model-dependent，不是可加的固定偏置。** 两层后果：**（a）** I7 的六臂结论（iaa/iag/nowave/kd/iaar/iagr）都产自同一 106 集口径，**臂间内部可比性仍在**，但其与部署真值的落差**因臂而异** → 不可外推到 batch 口径；**（b）** §13.2–13.6 与 §10.8 的全部 recall/Spearman 数值一律标「106 集 window 口径」。

**判别规则（本次总结）**：recall 升而遗憾不降（或反向）⇒ 量纲/口径产物；两者同向改善 ⇒ 真实能力。iaa 换口径 严格 +26.5pp / 遗憾 **+0.7pp（反向）** ⇒ 口径产物；nowave 相对 iaa 严格 +17.7pp / 遗憾 **−8.7pp** ⇒ 真实。

**⚠ 连带发现：历史表的 ckpt 标注不可信。** 同一 arm（v2nowave42m4）在 106 集 window 口径下有**三个互不吻合**的严格@3：§13.3 与 PROJECT_LOG §747 记 **39.6%**（§13.3 头部写服务的是 **mid200**）、PROJECT_LOG 首表把同一读数标成 **ep250**、09-15 十点扫描的 ep250 读 **34.9%**（ep200 33.0%）。前两者数值高度一致（39.6/65.1/10.66 vs 39.6/65.1/10.87）而第三者严格@3 差 **4.7pp** —— 远超 ±0.31pp 抖动带（该带由**遗憾**标定，**对 recall 未被验证**）。两种可能未分离：**（i）** 首表的 epoch 标注错（前两者同源，都是 mid200）；**（ii）** 17.2.4 定序修复对严格 recall@3 的影响远大于对遗憾的影响。⇒ **无论哪种，「mid200 比 ep250 好」这类跨 ckpt 判断在当前证据下都不成立**；要排序必须用 batch 口径逐 ckpt 重测，且每趟盖 ckpt sha1（17.3.11 的 `run_stamp` 已能打印）。

### 15.6 判据的稳健性：**两个判据的方向不再一致**（17.4.1 错标数更正后**结论翻转**）

> ⚠ **2026-09-17 更正**：本节初版写「严格 recall 稳、选择遗憾不稳，且四种口径全不达标」——那套数（74.4/68.9/79.5/65.3 与 4.76/7.23/5.67/5.11）**全部来自错标树**。换回新树复测后**两个判据各自内部就不一致**，且**方向恰好相反**（见下）。**旧判「与口径之争无关，成立」作废。**

**严格 recall@3 ≥90%：只有 ① 达标**（**90.8%** / 80.1 / 81.5 / 76.7；宽松 **97.9%** / 96.3 / 96.1 / 91.5）→ 旧记录那句「四种口径全不达标 →『GNN 只做启发式粗排』成立」**作废**，它建立在错标树上。**但 ① 正是被重复提议主导的口径**（DEPTH_MIX 一家 538/714 = 75% 权重）⇒ 达标的那个口径恰好是最不可外推的那个。
**选择遗憾 ≤5%：②④ 达标**（3.75 / 3.03），**①③ 不达标**（5.93 / 5.34）→ 与严格 k3 的排序**恰好相反**（① 在严格 k3 上最好、在选择遗憾上最差）。
⇒ 诚实读法：**不能说「GNN 够了」，也不能说「全不达标」**；两个判据的差就是**「重复提议」的定价**。**① 口径下两项都达标**（严格 k3 90.8% ≥ 90%、两阶段 0.63% ≤ 5%），**但达标与否压在一个电路上**。

两个已知的读表陷阱：**（a）③ 宏平均混了 n 不同两类集** —— n=4 的集其严格@3 随机基线是 75%，n=8 的是 30%，宏平均把两种难度等权；需 n≥8 分层版（**代码未改，待批**）。**（b）宽松 k3 在 n=4 集上定义性饱和** —— 真前3 与预测前3 各占 4 中的 3 → 必然相交 → 恒 100%；对全体 宽松@3 的影响小（f=20% → 97.0%、f=30% → 96.6%，实测 **97.9%** 在带内），但**读这个数时必须记住**。

### 15.7 待办

1. `_shadow_ckpt_sweep.sh`：`<0.3pp 视为打平` 换成中位 + 逐集配对；并按 §15.4 的重复度做**按电路分块重采样或 sig 去重**（独立性假设不成立）+ `best_model.pt` 入候选 + 逐 ckpt 盖 ckpt sha1。
2. 以 batch 口径重测 v2nowave42m4 的各 midpoint（至少 mid200 / ep250）—— 这是给 ckpt 排序、也是给「42b 残余 1.29pp」定性的前提。
3. `_shadow_analyze.py` ③ 的 n≥8 分层版 + `判定` 行标注「仅 ① 口径」。
4. 上述都属「选点/口径」工作，**不改变 16.11.39 的裁决**：缺口要动模型/数据（60w 新战役），不是继续换 ckpt。

### 15.8 74.4 错标定案（指针）+ 本报告 §15 的错标数更正范围（17.4.1）

**定案本体记在 `PROJECT_LOG.md`「(8) ✅ 已定案（2026-09-17）：74.4 是错标」**，此处只留指针与本次更正的**清单**，避免两个文档各写一份会各自漂移。要点：

- 那棵树的路径 = `tl_opt_batch_keep_0916` = `~/shadow_archive/unknown_20260916_223140`，**服务的是另一个模型**（不是 `42m4 midpoint_ep250.pt`）。
- **ckpt 没换、Rust 没变**：同 ckpt sha1 `09be9a3c53b90760` 自 2026-09-02 起逐位不变；三棵树的 `true_delay` 多重集指纹全等 = `cb0d7a96e6aada05`（46 CSV / 5398 行）。
- **这是第 2 次同成因的 I15 类错标**：记的是「本该服务的模型」的路径/sha1，不是 8000 实际服务的模型。防线已落地（17.4.0）：`run_shadow_batch.sh` 把 ckpt sha1 + Rust 源指纹 + repo dirty 指纹 + serve 脚本指纹写进 `RUN_INFO.txt` 随树归档，且空树不再归档 —— 本次正是靠一份这样的存档定位的。
- ⚠ **`cap2 95.55%` 与输入污染三项（含诊断位 31/20/271/0/0）目前仍是旧树那轮的读数、未复测**（`PROJECT_LOG` 已标「待补」）⇒ 本报告引用它们时必须带这一句，**不要**当成 `42m4 ep250` 的已确认值。

**⚠ 附带影响（17.4.1）：全库 `train_sweep.py:NNN` 引用整体漂移（已全部改掉）。** 17.4.1 在 `src/train_sweep.py:25` 处插入了 `midpoint_epoch_of`/`pick_midpoint` 共 **55 行**，**且选点块自身也被改过**（argmax 循环 → `pick_midpoint`）⇒ **偏移不是常数**：第 22 行以下 **+55**、730 行以下 **+60**、948 行以下 **+63**、1014 行以下 **+73**、1055 行以下 **+87**。⚠ 照「+55」一把梭会算错选点块附近的引用。已逐处核对并改：本报告 `:952`（`train_sweep.py:978` ⇒ 该行 **17.3.7 已删**，留档注释现于 `:1113-1115`）与 `:960`（`:84-91` → `:139-146`）；`PROJECT_LOG.md:921`（`:84-91` → `:139-146`）、`:951`（`:978` → 已删）、`:956`（`:815-825` → 见下、`:978-984` → 已删）；`GNN_CODING_LESSONS.md:19`（`:142-149, 192` → `:197-204`（`_data_mtime_hash`）+ `:246-247`（键合成））；`OPERATIONS.md:217`（`:1012-1073` → 节锚 `:1075-1150`、`config.py:22` → `:42`）。**政策操作性引用一律改成按锚点找**（`OPERATIONS.md:212` 与 `PROJECT_LOG.md:1100/1103` 的 `grep -n 'cap2={score' src/train_sweep.py`，17.4.1 时 `:1128`）—— 行号必然再漂，锚点不会。
- ⚠ **顺带查出两处 17.4.1 之前就已过期的引用**（不是本次插入造成的）：① 本条所引 `train_sweep.py:815-825` 说成 `smoothed_rel_err` 的判据 —— 那段实为训练循环里的**逐 epoch `cap2` 最优跟踪**（17.4.1 时 `:875-885`），`smoothed_rel_err` 的判据在 `:908`（17.4.1 前 `:848`）；② `:995` 的 `config.py:107`（`USE_STRUCT_PRIOR`）17.4.1 前即为 `:121`，现 `:138`；`model.py:110` 现为 `:111`（该文件 17.4.1 未动）。

**🔴 另有比行号更重的发现（17.4.1 内已一并修掉）**：`train_sweep.py:1113` 的注释写着「**17.3.7 换判据：原 score（100·r3+50·r2+0.3·sp−0.2·regret+0.1·cap，recall@3 主导）**」—— 即 **Best midpoint 的加权 `score` 判据在 17.3.7 就已被 capture2 取代**（现 `score = mrk.get('capture2_pct', ...)`，`:1119`）。因此本报告 `:952` 与 `PROJECT_LOG.md:956`（那条「⚠ 纠一处判据名错误 / 加权 score 选出」的更正）**描述的是已不存在的代码**。**已就地重述**：这两处现在都明写「加权 `score` 自 17.3.7 起已删 / 17.4.1 起选点改为取平台末端」，并保留历史实测值（该 `score` 序与部署序 Spearman = −1）**但标明它挂在已删代码上 ⇒ 不构成当前代码的风险**；`BEST_METRIC` 改名 `BEST_MODEL_METRIC` 也已在两处注明。
- **遗留一件仍需用户定夺的**：`BEST_MODEL_METRIC='capture2'` 本身在 §15.6 被判 **❌ 不采纳**（四条证据），但**判决尚未落进代码**（现 `config.py:42`）—— 那是 `BEST_MODEL_METRIC` 的定档（B7/D1），与本处「行号/判据名」的更正不是同一件事。

**本报告被更正的站点（17.4.1）**：§13.2 裁决第 1 点（`:823`，nowave +39.5 → **+55.9pp**）、§15.2 整表 + 其余 ① 读数 + ④ 反解（`:1086-1099`）、§15.5 表 nowave 行（`:1118` → **90.8% / −5.20pp**）、§15.6 两判据与结论翻转（`:1132-1138`）、§15.6 读表陷阱 (b) 的实测值（`:1140`，97.6 → **97.9**）。**未受影响**：§15.4 跨集重复度（只依赖真值列 ⇒ 结构量）、§15.5 表 **iaa 行**（56.7% 来自 iaa 臂的干净单 run，不是这棵树）、以及各表**行内 epoch 间比较**（同一棵树内 ⇒ 相对排序有效）。
