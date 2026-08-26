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
- 所有输入 **t=0 同时翻转**、单 vector、非切换输入接 vdd/gnd。
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

在 46 benchmark（或子集）跑完后，对**每个 window 候选集**（≥4）计算并汇总：

| 指标 | 定义 | 达标线 |
|---|---|---|
| **recall@top-3** | GNN top-3 含 SPICE 真 top-3 的比例 | **≥ 90%**（主判据） |
| **选择遗憾** | (GNN 选最优 − SPICE 真最优)/真最优 | **≤ 5%**（主判据） |
| Spearman | GNN 排序 vs SPICE 真序秩相关 | ≥ 0.6（次判据） |
| 仿真节省 | 若 GNN 只排 top-K 能省的 SPICE 比例 | ≥ 75%（收益项） |

**判定**：
- 两项主判据（recall@top-3 ≥90% 且 遗憾 ≤5%）→ **GNN 替换逐候选 SPICE 排序**：优化改成「GNN 排全部候选 → top-K 才 SPICE 精排 → 接受第一个改善的」，仿真省 ≥75%。
- 任一主判据不达标 → 保留 SPICE 全排序，GNN 只做**启发式预排序**（先粗排再 SPICE 精排 top-N，仍省部分仿真）。

> 关键：贪心每 window 只接受**第一个**改善候选（`break`），粗筛价值 = 让 GNN 把「该接受的候选」排到前面 → SPICE 只验证 top-K，不丢最优又省仿真。

### 10.4 参考实现

`NetlistOpt/src/gnn_shadow.rs`（已直接写入 NetlistOpt 独立仓库；`Cargo.toml` 加 serde/serde_json、`lib.rs` 挂 `mod gnn_shadow`、`tl_opt.rs` 挂载已就位）：
- `GnnClient`：`std::net::TcpStream` HTTP POST `/rank`（零依赖），解析 ranked JSON。
- `ShadowGnnTlEvaluator`：包 `FullCircuitTlEvaluator`（`new` 接收已配好 simulation_cfg 的 inner，因 `design_template` 是 tl_opt 私有字段），调 GNN + SPICE，写 CSV。
- 挂载：`optimize_tl_text` 里 env `GNN_SHADOW=1` 时换用 Shadow 版本（host/port 可 `GNN_HOST`/`GNN_PORT` 覆盖，默认 10.20.34.16:8000）。
