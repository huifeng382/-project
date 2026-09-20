# V3 生成 — 规模/深度档位校准结论（round 2, 2026-09 续；round 3..10 增补）

## round 10 增补（口径修复 + 补量批次 + 交付定稿，2026-09-20）

**1. 发现的根因级问题：`transistor_count` 口径错了。**
V2 采集器 `spec_pipeline/common/spice_utils.py::extract_static_features` 用
`len(re.findall(r'^M', text, re.MULTILINE))` 数晶体管。层次化 SPICE 里每个宏**只定义一次**，
所以得到的是「宏去重计数」，而规格 §表把 `transistor_count` 定义为「电路中的晶体管总数」，
并要求 `transistor_wave_json` 覆盖全部晶体管实例。
* 实证：`main_candidate_expr13267_0000`（9x6）旧列 249，`transistor_wave_json` 键数 373，
  `measure_sp_metrics` 的 `TRANS_PER_INSTANCE` = 373（与波形一致）。
* 后果：之前所有「★ 档不达标」的判断都是**误判**——用错口径把大电路算小了 1.2-2.6 倍。
  换回每实例口径后 5 个 ★ 形状全部达标（9x6 p50 284 vs 档 247-316 等，见交付文档）。
* 处理：不改动 V2 采集器（会影响历史数据），改为交付前用 `measure_delivered_trans.py`
  （本地 `measure_sp_metrics` 逐 .tl 测）+ `patch_delivery_metric.py`（服务器写回 + 重算元数据）修正。

**2. 短口形状补量（gate ①）**：新写 `gen_short_shape_fill.py`
* 综合小规模 SOP 家族（16 输入 5 项积之和 = 5 个 AND 宏 + 4 个 OR 宏 = 9 X_ 行 ≈ 36-48 管，
  正好落在语料 AND16 邻域；必须覆盖全部输入，否则生成网表的端口/形状会缩水——这是踩过的坑）；
* 收获池未被使用的新函数家族 + 结构 profile 去重回填（小电路的改写空间有限，凑不满 10 变体时
  直接用同家族其他收获电路补齐）。
* 产出 186 电路 / 13 组 → 走 `build_fill_batch.py`（A7xxxxx 组名）+ 新批次前缀 `f`
  （`finalize_v3_dataset2.py` 的 BATCH_PREFIXES 已加 `f`，`finalize_v3_dataset6.py` 的 lookup
  已认 `f`/`fill` 域），一次分块仿真（12 路，约 3 分钟）收 4,990 行。
* 结果：3x1 392/400、5x1 732/800、8x1 296/300、16x1 299/300；(1,1) 与 (2,1) 受函数空间上限
  约束无法达标（4×15=60 / 16×15=240），已写入交付文档。

**3. 组内延迟剖面（gate ⑦）无法达标的原因量化**：目标 20/35/39/6 只在**相邻差**读法下自洽；
即便按该读法，9x6 家族延迟跨度只有 **1.31×**（占交付 71% 的行），而「29% 步长落在 5-20% 档」
需要 >45% 家族跨度 → 结构上不可达，除非为同一函数补入更慢的实现再仿真。
（详见 `docs/V3_DATA_DELIVERY.md` §GATE 7 与 `metadata.group_delay.note`。）

**4. 交付定稿**：12,455 电路 / 599,976 行 / 837 组；Tier A 36.99%；覆盖率 100%；sc_expansion 1,049 宏 100%。
回归验收脚本 `verify_v3_gates.py`（本地，读交付 parquet + 测量 TSV + 剖面 JSON 打印十项 gate）。


## round 9 增补（选样定稿 + 数据集仿真已启动）

**选样**：变体倍增器产出 21,306 候选 → 去重 17,661 → 按形状选最优（`v3_combine_manifests.py`）：
- **Tier A 8 个形状达标**：(9,6) n437 p50 305、(7,4) n550 p50 299、(8,4) n500 p50 219、(8,3) n650 p50 218、
  (5,2) n900 p50 190、(9,1) n450 p50 169、(4,1) n400 p50 67、(5,5) n300 p50 82；
  未达标：小/普通形状（1x1,2x1,2x2,2x3,3x1,3x2,4x3,8x1,16x1）数量不足 + (5,1) 473/800。
- **Tier B**：8,876 电路，行加权 trans p50 **150**（≥120 ✓）、140-190 档 **60.7%**（≥20% ✓）、深 6-8 档 **52.5%**（≥15% ✓）。
- 600k 行预算下裁到 8,094 电路（Tier A 5,094 + Tier B 3,000），物化 `data/spec_pipeline/v3/v3_sel/`（sidecar: expr_idx→shape/fp/tier）。

**服务器管线（新建并验证）**：`gen_v3_candidates.rs`（按 expr 精确产出网表，端口命名沿用 V2 约定）、
`run_v3_pipeline.sh`（templates→measures→sim→collect）、`run_v3_chunked.sh`（分块仿真→收集→删 scratch，控磁盘峰值）、
`restart_v3_run.sh`（干净重启）、`kill_stale_sims.sh`/`sim_inventory.sh`（按 cwd 精确清理）。
小批预检（40 电路）**全链路通过**：3,139 行、transistor_wave/supply_noise **100% 覆盖**、行数 84<108 符合「仅敏感对」；
collect 输出实际在 `data/dataset_v3_pilot_v2/`（不是 v3_pilot/）。

**成本实测**：单次仿真 ≈3s（1s@49KB tb，10s@476KB，与规模成正比，模型解析可忽略）→ 全量 ~60 万次 ≈ 580 CPU-h。
**服务器与接收方 GNN 训练共用**（tianlang 的 main.py ×2 各占 ~5 核）→ 有效并行 ~14-16。
→ ETA：Tier A（chunk 0-5）≈24-26h；全量 ≈40-48h；接收方训练暂停可近乎减半。

**当前**：`restart_v3_run.sh 16` 已从 chunk 0 干净重启（8094 电路/9 块/16 路），日志 `/home/zhirui/v3_ds.log`；收获任务已停以让出算力。

## round 8 增补（Tier B 路线 + 两个新工具；收获继续放量）

**收获进度**：phase 1 **80/80 完成**（retry 用 ulimit 修复补掉了 11 个 stack overflow 失败）；
phase 1b 36/68（剩余是 9_6 i45 长 run）；**phase 1c_cheap 并行启动**（162 任务 × PAR=10，8_4/9_1/5_2 多 seed + 5_5/4_1/5_1）；池内 **885 个收获电路**。
链式脚本 `run_chain.sh`（retry → 1c_cheap → 1c_heavy → phase2）在服务器后台保底执行，与并行启动的阶段互为幂等（driver 按 TSV 跳过已完成 run）。

**新工具 1：`sp_metrics.py`（Python netlist 层指标）** —— 与 Rust `measure_sp_metrics` **逐位一致**
（枚举产物实测 XROWS=48 / TRANS=192 / 深=13，与语料 ADD4_OVF 相同）。用途：批量测 `.sp` 产物（枚举/模板管线），不再依赖 .tl 才能测量。

**新工具 2：`v3_variants_from_tl.py`（模块级保功能结构变体生成器）** —— 零搜索开销扩池，供 Tier B 体量与组内变体：
- 三条改写：R1 join 体重组（≥3 因子乘积/≥3 项和的括号重排）、R3 复制-内联（对多用户 X_ 选一用户内联其体为新建 X_，抵消共享 → X_ 行数与 trans 上升）、R5 输出反相配对；
- **每个变体都用 `tl_tt --fp` 验证与基准功能指纹一致**，并用 netlist 指标测量；实测一个 9_6 收获电路（base trans 424/89 行/深 11）→ 8 个等价变体（trans 422-432、X_ 88-92、深 11、宏 29-30）。
- 注意：当前改写只做小幅扰动（变体分布很窄）；需要更强杠杆（全用户复制、De Morgan 相位翻转、join 体拆分）才能显著改变 trans。
- 修了两个 bug：join 体逗号切分（`, ` 无前导空格）、`next_x` 的 `int(x)` → `int(x[1:])`。

**Tier B 策略定案（本地实测依据）**：
- 大模板枚举代价高、变体极少：对 ADD4_OVF 做 `max_depth=10` 枚举只出 **1 个候选**，其 RecExpr 达 **97,597 节点**（`join(X,X)` 自折叠写法撑爆表达式）→ 大形状**不靠枚举**。
- 因此 Tier B = ①大/中形状：**规模化收获**（Rust 邻域天然吻合）+ **变体倍增**（新工具）；②小/中形状：沿用 V2 枚举（便宜、候选多，V2 曾一次产 3.5 万电路）。
- 分发口径：Tier A 锚定 ★/◆ 与 18 形状配额；Tier B 满足分布判据（trans p50 ≥120、140-190 档 ≥20%、深 6-8 档 ≥15% 行）——注意这些是**行占比**，由大电路行数加权，因此 Tier B 需要足量中/大电路而非小电路铺量。

## round 7 增补（phase 1 收获体检：★档实测数据 + 放量策略修正）

**Phase 1 网格**（9 模板 × seeds × 30 iters，18 路并发）跑到 66/80 run 的实测（459 个收获电路，本地 netlist 层测量）：

| shape | 收获数 | trans p50/p90/max | 深 p50/p90 | ≥9 占比 | spec 档 | 判定 |
|---|---|---|---|---|---|---|
| (9,6) | 53 | 238/291/294（全池 192-319） | 10/11 | 1.00 | 247-316 / 深12-16 | 深/规模主体达标，p50 需再抬 |
| (7,4) | 123 | 267/319/348 | 9/10 | 0.77 | 235-262 / 深9-10 | **已达标** |
| (8,3) | 45 | 166/180/188 | 8/9 | 0.20 | 170-233 / 深10-12 | 规模不足（max 188<198）、深不足 |
| (8,4) | 58 | 140/180/200 | 7/7 | 0.00 | 189-233 / 深4 | **规模不足**（需提高 iters 继续 delay 下降） |
| (5,2) | 77 | 129/174/190 | 6/10 | 0.13 | 140-208 / 深6-11 | 规模偏小 + ◆深 13%<20% |
| (9,1) | 41 | 134/176/199 | 10/12 | 1.00 | 104-146 / 深13-15 | 档次达标，缺数量（配额 450） |
| (5,5) | 32 | 54/74/74 | 3/4 | 0 | 48-60 | 达标 |
| (4,1) | 14 | 52/89/89 | 6/7 | 0 | 12-52 | ◆深 0%<20% |
| (5,1) | 16 | 44/44/44 | 2/2 | 0 | 14-117 | DEPTH_MIX 产出极少（1/run），需换 AOI221/OAI221 |

**关键结论与策略修正**：
1. ★scale/深档可由真实 tl_opt 收获满足（(9,6)/(7,4) 实证），但每 run 收获 3-17 个；配额（Tier A ~7900 电路）无法只靠收获。
   单 run 实测成本（18 路并发含争用）：8_4 ≈12 min、9_1 ≈15 min、5_2 ≈27 min、8_3 ≈3.6 h、9_6 ≈13 h。
2. **Tier B（63% 行）走 e-graph 枚举**（V2 老管线 batch_v2_rest 曾一次产出 35,183 电路）——收获供 Tier A 锚定与 Rust 邻域样本，枚举供体量，最后合入同一数据集。
3. 组结构：**expr = 单次 tl_opt run 的收获链**（同 run 功能等价、结构各异；指纹与语料一致已验证）→ 形状配额靠 run 数，不靠多造功能。
4. 选样剔除域外：trans>400 或 depth>18（spec 要求域外 <5% 行）；(7,4) 池有 trans 到 458 的须滤掉。
5. 失败 11/80 run 全为 stack overflow（未定位具体函数；规避方式=换 seed 重跑）。

**Phase 1b（已启动，PAR=6，68 run）**：8_4 提到 60/90 iters（ENC8 单 run 便宜，靠 iters 抬 trans）、9_6 45 iters、5_2/9_1/5_5/4_1 加量、5_1 换 AOI221/OAI221。
**Phase 2 网格已备**（`grid_phase2.txt`，37 模板 × 20 iters × 6-8 seeds）→ 非 ★ 形状配额 + Tier B 邻域池。

**存档**：`reports/pool_screen_round6.tsv`（459 电路逐条指标）、`reports/tierA_precheck_round6.json`、`reports/harvest_phase1_round6.log`。

## round 6 增补（服务器打通，Tier A 收获实跑，spec 档位首次实证）

**链路**：VPN 恢复 → 装 ed25519 免密（本地 `~/.ssh/id_ed25519_dsh`）→ 服务器 `~/NetlistOpt`（V2 时期旧快照）补传：
`src/tl_opt.rs`(收获插桩)、`src/utils/tl_graph.rs`(拓扑修复)、`Cargo.toml`、`data/script/tl_opt_candidates.rs`、
`testbench/tl_cells/`(46 语料)、`data/spec_pipeline/v3/server/*`。
两个环境坑：(1) 服务器缺语料；(2) `src/process_template/xyce.sh` 硬编码 `/home/cao/spack/.../Xyce` → 已改 `/usr/local/bin/Xyce`。

**关键 bug（已修，影响所有收获产物）**：`TlModule::to_tl_text()` 按 X id 顺序输出，而 tl_opt patch 后 id 非拓扑序 → 收获 .tl
「引用先于定义」，`parse_text` 报 unknown signal（pilot 7 个产物全部不可解析）。
修复：`to_tl_text()` 改按依赖拓扑序输出（新增 `TlModule::topological_x_order()`，用现成 `collect_x_names_from_root` 建边 + Kahn，环回退声明序）。
本地验证：新收获文本可解析（TRANS 188 / XROWS 45 / 深 11）。历史产物用 `data/spec_pipeline/v3/tl_topo_repair.py <in> <out>` 抢救（pilot 8 文件已修复）。

**Pilot（ADD4_OVF seed1 × 8iters，真实 Xyce）实测**：117 候选仿真、7 accepted、0 失败；收获链 netlist 层
trans **204→212→212→234→236→250→252**、X_ 行 47-49、链深 11-12、宏种类 6→14；**功能指纹与语料 ADD4_OVF 完全一致**（组内功能等价 ✓）。
对照 spec (9,6)：trans 档 247-316（8 iters 已 2/8 入档）、深 ≥9 ✓、X_ med 32/p90 57 ✓、非 m4 巨型宏 ✓。
→ **★scale/深档可由真实 tl_opt 收获直接满足，机制与流水线双验证通过。**

**成本/放量**：单候选 ≈ 84 次 Xyce；8 iters ≈ 117 候选 ≈ 30-40 min；30 iters ≈ 440 候选 ≈ 2h/run。
`server/run_tierA_harvest_v2.sh`（18 路并发、每 run 独立子目录、跑完清理 expr_* scratch、断点续跑）已 nohup+setsid 启动：
**80 run（9 模板 × seeds 6-12）× 30 iters** → `/home/zhirui/v3_pool/pool/<tag>/i30_s<k>/harvest_*.tl`。
待办：其余 9 形状（1x1,2x1,2x2,2x3,3x1,3x2,4x3,8x1,16x1）模板的 phase 2 收获（小电路、单 run 快）。

**新增本地工具**：`v3_screen_pool.py`（池→netlist 层指标+功能指纹 sha1/分组）、
`v3_pick_tierA.py`（按组选样 ≤15 变体 + 逐形状 spec 验收预检 → report.json）、`tl_topo_repair.py`（旧收获文本抢修）。

## round 5 增补（服务器恢复即执行；一键 stage 脚本就绪）

- 用户已去重连 VPN（2026-09 会话内）。恢复后**一条命令**完成全部服务器侧准备：
  `data/spec_pipeline/v3/server/stage_harvest.ps1`（自动探测服务器 crate → scp 改动的
  src/tl_opt.rs / Cargo.toml / data/script/tl_opt_candidates.rs / run_tierA_harvest.sh →
  cargo build --release --example tl_opt_candidates → 单跑 pilot（ADD4_OVF s1 i8，看每 run
  收获量与耗时）→ nohup 全网格（9 模板 × seeds × iters 30）。
- 收获池在服务器 /home/zhirui/v3_pool/pool/<shape_tag>/（harvest_*.tl + TSV + run 日志），
  回传本地后 `python v3_screen_pool.py <pool_dir> <out.tsv> <crate>` 筛/分组。
- 备选本地路线（WSL Xyce）用户暂不采用；ngspice 已装但不能直接替代 Xyce 输出解析。

## round 4 增补（本地流水线就绪；服务器仍不可达）

- 服务器 VPN 依旧不可达（ping+ssh 全超时）——Tier A 收获与全部 Xyce 仿真仍是硬依赖。
- **组结构定案**：每组 expr = 单次 tl_opt run 的收获链（accepted+best，截 10-15）；配额电路数 = run 次数 × 每 run 收获数。★ 大形状 350-550 电路 ≈ 25-40 run/模板即可，不必每种函数都合成。
- **本地工具补齐（均验证）**：
  - `data/script/tl_tt.rs --fp`：打印全真值表指纹 `TT_FP=n_in_n_out_hex`（无 ref 文件即可做功能分组；COMP4 语料与本地 _cmp4b 指纹一致验证通过）。
  - `data/spec_pipeline/v3/v3_screen_pool.py`：对收获池逐 .tl 跑 measure_sp_metrics + tl_tt --fp → TSV + 按 (shape, 功能) 分组报告，含每形状 in-band（对照 SHAPE_RULES 的 Rust 档）。语料 46 文件 + tl_seeds 全部跑通，数字与直测一致（如 ADD4_OVF 192/48/深13；(9,6) seeds trans 180-218/深最大24）。
  - `data/spec_pipeline/v3/server/run_tierA_harvest.sh`：服务器主跑脚本（模板×(seed 1..N × iters) → 收获池 + TSV + 日志）。
- 数值佐证：tl_opt 纯 delay 下降会把过深种子（ripple 深 20-24）压回 ~12-16 Rust 区间 → 种子深度不必预调。
- 服务器恢复后执行序：run_tierA_harvest.sh → 池回传 → v3_screen_pool.py 筛/分组 → 按配额挑 expr（每 run 组 10-15）→ 既有 V2 服务器 sim 管线 → 单数据集 finalize（V3 版）。

## round 3 增补（结构代理跑真实 tl_opt 验证通胀机制）

- **机制定案**：`src/tl_opt.rs::combined_score = delay/delay0`（纯 delay 最小化，trans 不进目标）。候选比语料大 = delay 下降副作用抬高 trans。
- **本地结构代理实验**（`examples/tl_opt_local_structure.rs`，无 Xyce）：delay 代理 = 每输出「加权 X_ 链长」（hop 权重随该 X_ 的 join 树尺寸增长），trans 用 canonical egraph 计数；直接调 `optimize_tl_module`（对 TlEvaluator 泛型）跑真实窗口搜索。
- **结果（ADD4_OVF，16 iters，seed1）**：accepted=5 harvested=5 best_trans=**247**（语料基线 192 → 抬升 28%，恰落 Rust 档 247-316 下缘）、best_delay 14.96、best_xdefs 58。→ **服务器跑真实 tl_opt（Xyce delay）收获必然能复现 Rust 候选档**，每 run 约 5+ 候选。
- 教训：整数链长作 delay 时全部平局 → 几乎零接受；必须实值（权重）才有接受。真实 Xyce delay 天然实值，服务器 run 无此问题。
- 注意：`to_tl_text()` 对 apply_patch 后的 module 可能输出「引用先于定义」顺序，`TlModule::parse_text` 再解析会失败（'unknown signal'）——收获消费方不要 reparse 文本数 X 定义，按行文本计数（见 example 内 `count_xdefs_text`）；tl_opt_candidates.rs 写盘不受影响（只写不读回）。
- 运行很慢且 extractor 的 `[fixed-concat-sampler]` 刷屏（无条件 println）。仅作本地 smoke/机制验证用，正式收获走服务器 tl_opt_candidates。

## 结论（round 2 定案）

1. **★scale 形状的 trans 档位 = Rust tl_opt 候选实测，不是语料模板基线，也不是任何功能重综合能复现的。**
   - 语料模板按「每实例 M_ 口径」实测（netlist 层）：
     ADD4_OVF(9,6) **192** / 48 X_ / 5 宏 / 链深 13
     ALU2(7,4) 186 / 42 / 5 / 9；COMP4(8,3) 116 / 28 / 6 / 10；ENC8(8,4) 120 / 30 / 7 / 7；OVF(9,1) 116 / 30 / 3 / 13
   - Spec 档位（Rust 候选实测）：(9,6) 247-316 深12-16；(7,4) 235-262 深9-10；(8,3) 170-233 深10-12；(8,4) 189-233 深4；(9,1) 104-146 深13-15。
   - tl_opt 候选比语料模板**更大**（delay 优化 trade trans up），X_ 行 32-73、每实例 trans 高 ~1.3-1.6×。
2. **合成加法器家族不能填 ★ 档**：ripple 218/深20、naive 202/深24、KS 180/深20（超 Rust max16 属域外 >18 风险）、CLA 208/深6（不及 9-16 深档）→ 功能重综合无法命中 (trans, 深度) 二元组。唯一正路 = 服务器跑真实 tl_opt 收获候选。
3. **测量口径已本地忠实复刻**：
   - `count_transistors_instance_on_egraph`（Rust 同款，纯本地）= netlist「每实例 M_ 口径」一致（ADD4_OVF 均 192）。
   - `examples/measure_rust_metrics.rs`：.tl → (TRANS_RUST, X_ 级链深, 宏分布)（无 sim）。
   - `examples/measure_sp_metrics.rs`：.tl → expr_to_hierarchical_spice → 按接收方脚本口径数 DUT X_ 行链深 / 每实例 trans / 宏实例分布（无 sim）。
   - 两工具都已注册进 Cargo.toml，`--release` 构建通过。
4. **tl_opt 收获已插桩**：src/tl_opt.rs `TlSearchParams.harvest_candidates=true` 时把每个 accepted 中间态（功能等价、结构各异）连同 (trans, avg_delay) 存入 `TlOptimizationResult.harvested`；`examples/tl_opt_candidates`（data/script/tl_opt_candidates.rs）会把全部 harvested 写成 `harvest_<tpl>_i<iters>_s<seed>_n<k>.tl` + TSV manifest。本地构建通过。← 服务器用此跑。
5. **COMP4 形状 (8,3) 已解决**：语料 OUTORDER = `lt eq gt`（不是 gt/eq/lt）。本地 `_cmp4b.tl`（lt/eq/gt 4bit 比较器）`MATCH_REF=true`；netlist：126 trans / 34 X_ / 5 宏 / 深 8。

## 服务器恢复后的执行序列（Tier A 引擎）

1. 上传最新代码 + 在服务器 /home/zhirui 的 NetlistOpt 副本 cargo build --release --example tl_opt_candidates。
2. 对 ★ 模板逐条跑（多 seed × iters）收获候选 .tl：
   - (9,6) ADD4_OVF.tl； (7,4) ALU2.tl； (8,3) COMP4.tl； (8,4) ENC8.tl； (9,1) OVF.tl；
   - ◆/深次 (5,2) ALU_SLICE_SMALL.tl； (4,1) PARITY4.tl（+普通形状模板跑 Tier B 邻域池）。
   - 预算参考：接收方 1135 个 dut 网表即此类收获。每模板 ≥10-30 run（seed/iters 组合），把 accepted 链全落盘。
3. 候选 .tl 回传本地 → measure_sp_metrics/measure_rust_metrics 逐条测 (trans/X_/深/宏) → 按形状筛入档（p50≥Rust med、max≥Rust max×0.85、189-233 与 233-316 两档都有、深档 ≥30%≥9 等）→ 真值表分组（功能等价 = 同 expr 组）→ 每组取 10-15 结构去重变体。
4. Tier A 配额不足时：用更多 (9,6)/(7,4)... 同功能不同语料模板？语料每形状通常仅 1 模板 → 靠多 seed 变体数 + (功能任选) 额外合成链式功能族（比较/加法/饱和等）再 tl_opt。
5. 之后走既有 V2 服务器管线（候选→sim→collect→finalize）产出单数据集 + metadata + coverage + sc_expansion（V3 宏集）。

## 当前本地工具清单（F:\TransiLog-share-delay-opt-branch\NetlistOpt）
- examples/measure_rust_metrics.rs、examples/measure_sp_metrics.rs（本地标定）
- data/script/tl_opt_candidates.rs（服务器收获；需 Xyce）
- data/script/tl_tt.rs（真值表 MATCH_REF 验证）
- data/spec_pipeline/v3/synth_v3_ripple.py（参数化 .tl 合成器，可生成其他链式功能家族种子）
