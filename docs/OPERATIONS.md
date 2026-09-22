# 运行手册（OPERATIONS）——服务器 / 变体 / 开关 / 现状 / serve / git

> 与 `TESTING_GUIDE.md`（测试流程）、`GNN_PROJECT_REQUIREMENTS.md` §4/§6.1（启动/命令安全规范）配套。
> 最后更新：2026-09-04（17.0.4）。

## 1. 环境与目录布局

- **服务器**：`gnn-dev` / orca（24 核 / 60GB RAM / Python 3.13 venv `~/venv`）；训练/仿真只在服务器跑。
- **本地**：仅代码编辑、只读分析、git 提交/推送。
- **服务器目录**：
  - `~/-project/`：**分析工作区**（setup_exp.sh、scripts/diag、data/ 数据源、serve log）——⚠ 它的 git HEAD 落后本地（见 §6），分析脚本以本地为准、训练以 GitHub clone 为准；
  - `~/project-107-<V>/`：每个训练 run 一个目录（由 `setup_exp.sh` **clone GitHub 分支**得到，含 config/data/cache/日志/checkpoint）；
  - `~/NetlistOpt/`：Rust 仓库（独立，服务器端源码由 tar 整体同步过，含 testbench/temp_sim_test/src）。
- **本地数据源**：`data/`（batch_v2_full/rest/m4/io、delivery*、archive_v13.1、sc_expansion.json、std_cells.lib…）。

## 2. 训练启动（强制走 setup_exp.sh，REQUIREMENTS §4）

```bash
cd ~/-project
# 全新 run：clone GitHub 分支 → 套变体 → 数据/缓存种子 → nohup 训练
CACHE_SEED=$HOME/project-107-<同特征旧run> bash setup_exp.sh <variant>
# RESUME 续训（保留缓存增量）：目录已存在时
RESUME=1 bash setup_exp.sh <原变体名>
```

- 每 run `OMP_NUM_THREADS=6`；**4 run 并行 = 24 核满**（不要超）。
- 数据默认 = `batch_v2_full + batch_v2_rest + batch_v2_m4`（config `DATA_BATCHES`，m4 自 2026-08-31 默认并入）；`batch_v2_io` 作验证集不入默认训练。
- 缓存：`cache107$V` 目录；**缓存键含数据文件 mtime** → 复用缓存必须 `cp -a` 保 mtime（setup_exp.sh 的 CACHE_SEED 已处理）；特征一致的 run 可复用旧 run 缓存（如 v2iaa42m4 → v2iaar42m4 / v2kdwave42iaa42）。
- 分布式注意：新 run clone 自 GitHub 分支（训练代码版本 = GitHub 最新 push；本地未 push 的代码改动不会生效）。

## 3. 变体字典（setup_exp.sh 命名解码）

| 变体 | 作用 | 备注 |
|---|---|---|
| `v2wave<seed>` | wave 全字段训练 | V2 数据；seed 尾缀 |
| `v2nowave<seed>` | 关 wave（Rust 推理拿不到 wave 的验证） | 16.9.3 基线系 |
| `v2ia<seed>` / `v2cov25<seed>` | 只 ids_avg 单字段 / 25% 行覆盖率 | 消融 |
| `v2iaa<seed>` | ids_avg + **线性拟合近似**(`USE_IDS_AVG_APPROX=1`) | 零仿真可算特征 |
| `v2iag<seed>` | ids_avg + **GBDT15 近似**(`USE_IDS_AVG_APPROX=2`) | 需 `outputs/idsavg_gbdt15.joblib` |
| `v2iaar<seed>` | v2iaa + **真值排序 loss**(`RANK_LOSS_W=0.5`) | #8 排序直训（2026-09-03） |
| `v2iagr<seed>` | v2iag(GBDT15近似) + **真值排序 loss**(`RANK_LOSS_W=0.5`) | 17.0.4 空格1 = rank×最不伤近似（补 §13.5 矩阵空格） |
| `v2nowaver<seed>` | v2nowave(纯拓扑) + **真值排序 loss**(`RANK_LOSS_W=0.5`) | 17.0.4 空格2 = rank×最佳 serve 特征（= 三臂实验的 **B 臂**） |
| `v2nowavegs<seed>` | v2nowave(纯拓扑) + **GroupedBatchSampler 单独打开**(`USE_GROUPED_SAMPLER=1`)，排序损失仍关(w=0) | 17.4.4 三臂实验 **C 臂**：与 A(=默认 `auto`+w=0，即 `v2nowave42m4`) 之差 = **纯采样器效应**，与 B 之差 = 纯损失效应 |
| `v2nowavegnn<seed>` | v2nowave(纯拓扑) + **GNN 预测 ids_avg 列**(`IDS_GNN_TABLE=<OOF 表>`) | 17.1.2；表由 `scripts/diag/_fit_idsavg_gnn_server.py` 的 `NFOLD/FOLD_IDX` + `INFER_CKPT/PRED_OUT` 交叉拟合产出（OOF 无泄漏）；缺表 → 该列全 0 = 退化成 v2nowave。**起训练前先跑 `scripts/diag/check_ids_gnn_table.py` 验键对齐**（17.1.3），行键命中须 ≈100% |
| `v2kdwave42iaa<seed>` | **v2wave42m4 教师蒸馏**(reg+rank)；⚠ 名带 iaa 但学生**实为纯拓扑**——KD 分支 `USE_TRANSISTOR_WAVE=False` 把近似列门控挡掉（data_loader L507/L510），`USE_IDS_AVG_APPROX=1` 未生效 | #12；教师软标签在教师 outputs；「iaa 学生+KD」格从未真正测过（DIFF §13.4 更正） |
| `v2kd<teacher><mode><seed>` | 旧蒸馏（teacher=123/ENS；mode=reg/rr） | 15.2；学生无近似特征（已弃路线） |
| `rankloss1/2`、`bmsm`、`es`、`anneal`、`bestrank`、`seed*` | base 系调参/选点/种子 | V1 时代为主 |
| `struct*`（structlogic/rich/elec…） | STRUCT_MODE 消融 | 默认 logic_only |

- 命名规则：变体名 = 特征/目标前缀 + seed；尾部 `m4` 仅是历史标识（数据默认已含 m4）。
- **教师/学生关系（16.11 系）**：v2wave42m4 = wave 教师（Test 13.3% / 遗憾 0.65% / Spearman 0.688）；v2nowave42m4 = 纯拓扑无近似（**serve 交付走此路线**；部署口径 = 714 集 batch，严格@3 **90.8%** / 宽松@3 **97.9%** / 选择遗憾 **5.93%** / 两阶段 **0.63%**，DIFF §15.2；⚠ 旧记 74.4/97.6/7.23/1.66 **属错标树**，2026-09-17 已改判，见下 §6.7 与 PROJECT_LOG「74.4 错标定案」）；v2iaa42m4 = 线性近似（纯 huber，serve 净伤最大；原「Rust 记录 ⚠ 不可复现」**已销案 = 口径差**，DIFF §15.1）；v2iag42m4 = GBDT15（Test 23.00%——**最不伤的近似**，排序质量最高但 top1 落点输 nowave）；v2iaar42m4 = 线性+rank（已双端验，DIFF §13.5）；v2kdwave42iaa42 = wave 教师 KD 学生（**已验 Rust：名带 iaa 实 in=45 纯拓扑，双端不赢，DIFF §13.4**）。

> ⚠ **下表/上文所有 Rust 数值凡出自 16.11–17.0 记录者，一律是 106 集 `window` 池化口径**（历史）。**2026-09-16 定案：部署真值 = 714 集 `batch` 口径**（`(circuit, iter, window_try)` = 一次 `pre_rank` 调用），换口径对严格@3 的增幅 **+26.5pp（iaa）～+55.9pp（nowave，2026-09-17 改；旧记 +39.5 用了错标树的 74.4）——因臂而异** ⇒ **旧口径的跨臂排序不可外推**（臂内同口径比较仍有效）。选型/选点前请先按 batch 口径重测。详见 `GNN_RUST_DATA_DIFF.md` §15。

## 4. 关键开关（config.py，多数可 env 覆盖）

| 开关 | 取值 | 含义 |
|---|---|---|
| `SPLIT_SEED` / `TRAIN_SEED` | 42（切分固定）/ env | 切分与训练解耦（同切分可集成） |
| `STRUCT_MODE` | logic_only（默认） | 门特征模式（base/logic_only/rich/elec） |
| `USE_TRANSISTOR_WAVE` | True/False | 是否用晶体管波形 |
| `WAVE_FIELDS` | env | 波形字段子集（如 `ids_avg`） |
| `USE_IDS_AVG_APPROX` | env '0'/'1'/'2' | 0=无近似；1=线性；2=GBDT15（加载 `outputs/idsavg_gbdt15.joblib`，优先 `~/-project/outputs/`） |
| `RANK_LOSS_W` | 0.0(默认)/0.5/2.0 | >0 启用真值组内 pairwise rank loss（GroupedBatchSampler） |
| `KD_ENABLED` / `KD_MODE` / `KD_TEACHER_DIR` / `KD_LAMBDA` / `KD_RANK_W` | env | 蒸馏：reg / rank / reg+rank；教师预测 `kd_teacher_preds_{train,val,test}.npy` |
| `KD_PREDS_ONLY` + `KD_TEACHER_CKPT` | env '1' | 一次性导出教师预测（在教师 run 目录跑） |
| `USE_CORNER_ATTN` / `USE_PARASITIC_CAPS` / `USE_SUPPLY_NOISE` / `USE_STRUCT_PRIOR` | 布尔 | 消融开关 |
| `BEST_MODEL_METRIC` | smoothed_rel_err（默认） | checkpoint 选点 |

## 5. 当前状态（2026-09-06，17.0.13 记录后）

> ⚠ **2026-09-16 口径标注（适用于本节所有 Rust 数值）**：本节各行的遗憾/严格@3 全部是 **106 集 `window` 池化口径**（历史）。**部署真值 = 714 集 `batch` 口径**；同 arm 只换口径严格@3 动 **+26.5~55.9pp 且因臂而异**（2026-09-17 改）⇒ **本节的臂间排序不可外推到部署口径**（臂内同口径比较仍有效）。交付基线的部署口径权威数见 §6.7。详见 `GNN_RUST_DATA_DIFF.md` §15。

**rank 两空格 Rust shadow 已跑完（2026-09-06 21:22/21:31，17.0.13 记录；PROJECT_LOG 表行 + DIFF §13.6）**：
- `~/project-107-v2nowaver42m4`（空格2 纯拓扑×rank）：Rust（serve midpoint_ep250、in=45 无 env）遗憾 **11.97%**（严格@3 34.9% / 宽松 57.5% / Sp 0.094）→ **> nowave 10.87% 全轴净伤害，纯拓扑×rank 槽位关闭（rank 目标线关）**
- `~/project-107-v2iagr42m4`（空格1 GBDT15×rank）：Rust（serve midpoint_ep150、in=46 env=2）遗憾 **10.50%**（严格@3 34.0% / 宽松 52.8% / Sp 0.238）→ **首破交付基线 nowave 10.87%、Sp 家族最高，但严格召回自 iag 44.3% 崩到 34.0%；非交付**（GBDT15 serve 脆弱 + 输 5.6pp 严格召回）。serve 交付基线不变 = v2nowave42m4。

idsavg diag：服务器全量已完成 → IDS_AVG_GNN.md §4.2（参数放宽对照待办）。

**已训完（2026-09-04 更新）**：
- `~/project-107-v2iaar42m4`（排序 loss #8 = iaa+rank）：09-04 收尾；**train-side + Rust 均已记**（PROJECT_LOG 17.0.2/17.0.3、DIFF §13.5）——Rust 选择遗憾 12.72%（中位 7.71）仍输 nowave 10.87，**两阶段 3.64% 五者最低**；nowave 纯拓扑交付基线不变；serve 已停（09-04 用户决定不常驻，要用再起，见 §5 其他现场）。
- `v2kdwave42iaa42`（wave 教师 KD #12）——194 epochs plateau 早停；train-side ≈ v2iaa42m4、**无 KD 增益**；**Rust shadow 已跑完**（22:03，106 集，选择遗憾 14.97% vs nowave 10.87%，双端无增益，DIFF §13.4 收口）。

**其他现场**：
- serve：**8000 已于 2026-09-22 23:1x 恢复挂交付基线 `v2nowave42m4 midpoint_ep250`**（PID 1434028；就绪看端口：`{"status":"ok","n_models":1}`；身份按 `/proc/PID/cmdline` 实测 `--ckpt /home/tianlang/project-107-v2nowave42m4/outputs/midpoint_ep250.pt`、ckpt sha1 `09be9a3c53b90760`；log 落 `~/-project/serve_v2nowave42m4.log`）。时间线：`midpoint_ep300`（09-17 ckpt-sweep 末位遗留，非交付基线，见 I9）→ V3 `v3nowave42 midpoint_ep200`（本轮 V3 shadow 用）→ **ep250（现）**。**交付基线（用户 2026-09-22 决定维持）= `v2nowave42m4/outputs/midpoint_ep250.pt`**（`in_features` **45**，探针实测；**不带 `USE_IDS_AVG_APPROX`**，但 §6.3 那三行 + `PYTHONHASHSEED=0` 必带；命令见 §6 Step 2/Step 6）；09-04 起「serve 不常驻」政策不变——确认无后续 shadow 需求可 `pkill -f 'serve_htt[p].py'` 停。
- 教师软标签：`~/project-107-v2wave42m4/outputs/kd_teacher_preds_{train,val,test}.npy`（已产出，train 514,494 行，对拍通过 regret 0.51%/Spearman 0.696）。
- 已训完模型（m4 Rust 三兄弟全跑完，定论 DIFF §13.3）：v2wave42m4（教师）；**v2nowave42m4 = 纯拓扑，Rust 遗憾 10.87% 最优（serve 交付走此路线）**；v2iaa42m4 = 线性，Rust 记录 ⚠ 不可复现（遗憾 15.21%，serve 净伤最大）；v2iag42m4 = GBDT15，Rust 严格@3 44.3% / 遗憾 12.19%（最不伤近似）。

## 6. serve / Rust shadow 标准流程（runbook：训完一个模型 → 换 serve → 跑 shadow → 判收尾 → 记录）

> 每次一个新变体训完要跑 Rust shadow，**照此流程执行**（v2kdwave42iaa42 = 第一次按它走）。全程在服务器，命令直接复制。三处判据坑已固化：**①serve 该不该带 `USE_IDS_AVG_APPROX` 看 ckpt 真实 in_features，别信变体名；②serve 就绪判端口在听，别判 log（无 `-u` 块缓冲，log 空是正常的）；③shadow 收尾判首行时间戳被替换，别 grep `全部分片结束`（旧文件残留会误判）**。

### 6.1 Step 0 — 确认训完 + 记 train-side（本地，先做）
- 训完判据：ps 里训练进程消失 / `outputs/` 有新 `test_predictions.npz` + SUMMARY 停掉。
- 记 PROJECT_LOG（train-side：epoch/stop 原因/Test Median/遗憾/Spearman/recall@3B/成对分辨，跟 m4三兄弟同表对比）+ 同步 OPERATIONS §5 状态 + OPEN_ISSUES I7 + 版本化 commit（16.11.N，≤20 字）。
- ⚠ PROJECT_LOG 训练侧展示的 checkpoint 常是 SUMMARY 的 midpoint，跟 Rust 要 serve 的 ckpt **可以不同**——serve 选哪个另行定（见 Step 1 命令里的 `<ckpt>`）。

### 6.2 Step 1 — 定 serve 特征布局：查 ckpt，不猜变体名
```bash
~/venv/bin/python3 -c "import torch,os; ck=torch.load(os.path.expanduser('~/project-107-<V>/outputs/<ckpt>.pt'),map_location='cpu',weights_only=False); sd=ck.get('state_dict',ck); print('in_features', tuple(sd['convs.0.lin_rel.weight'].shape)[1])"
```
（⚠ 用 `os.path.expanduser` —— Python 不展开 `~`，直接抄字符串会 FileNotFoundError。）

- **in_features = 45 → 无 ids 列（纯拓扑，nowave 风格）→ serve 不带 `USE_IDS_AVG_APPROX`**
- **in_features = 46 → 有 1 列 ids 特征 → 按下面 `struct_mode`/训练配置再分三种模式**

**⚠ 46 是二义的，形状分不出来**（延迟侧静态块恒为 `logic_only` 7 列，故 7+7+1=15→46 对三种来源都一样）；
且 delay ckpt = **裸 `state_dict()`**（train_sweep L824/829），**不带任何元信息**，所以只能从训练配置反推：

| ids 列来源 | serve 模式 | 怎么确认 |
|---|---|---|
| 无（纯拓扑） | 不带 env（in=45） | 形状即证 |
| 线性系数近似 | `USE_IDS_AVG_APPROX=1` | 训练跑的是 `v2iaa*` 臂（`config.IDS_AVG_APPROX_COEF`） |
| GBDT15 近似 | `USE_IDS_AVG_APPROX=2` | 训练跑的是 `v2iag*` 臂 |
| **idsavg GNN 现场预测（17.1.5 新增）** | **`USE_IDS_AVG_APPROX=3`** | 训练跑的是 `v2nowavegnn*` 臂 |

判据用**训练那次 setup_exp.sh 的回显**（`grep -m1 IDS_GNN_TABLE` 该次的启动/训练 log）——`v2nowavegnn[0-9]*`
臂会 `export IDS_GNN_TABLE=...` 并回显 `IDS_GNN_TABLE=<表路径>`，只有它设；`v2iaa*`/`v2iag*` 走 `USE_IDS_AVG_APPROX`。
（`v2nowavegnn42b` 的训练 log 里 `Version: 64dcdcf 17.1.2 - GNN预测ids入delay特征+交叉拟合` 也印证这一点。）
- ⚠ 教训：`v2kdwave42iaa42` 名字带 "iaa" 但 ckpt 实测 **45 → 纯拓扑、不带 env**。盲带 `=1` 会启动失败 `size mismatch ... [256, 45] from checkpoint ... [256, 46]`（serve.py L316-317 按 env 加 extra_dim）。
- ⚠ **模式选错不会报错**（45↔46 才报错；1/2/3 之间形状相同）→ 只会静默喂进一条口径不对的列。

### 6.2b Step 1b — 模式 3 必过的 parity 闸门（不过就别跑 shadow）
模式 3 的那一列是 serve 端**现场**跑 idsavg GNN 算的，必须先证明它 == 训练时喂给 delay 模型的那一列，
否则 shadow 数字测的是"另一条特征路径 + delay 模型"的混合效果，得不出结论。
```bash
cd ~/-project && ~/venv/bin/python3 scripts/diag/check_idsgnn_serve_parity.py
```
- 做法：第 i 折 ckpt ↔ 第 i 折 OOF 表配对，抽电路用表内各行的真实 corner/slew/load 重建 serve 侧输入 → 前向
  → 与表里 `pred_log1p` 逐 (行,门) 对比。**同时验证"逐行前向 == 训练侧 R×N 拼块前向"的等价性**（本地已用
  AST 抽训练侧原函数比对，ΔT=0）。
- 通过判据：全局 `max|diff|` ≤ `--tol`（默认 1e-4），脚本自己 exit 非 0 就是失败。
- ⚠ ckpt 缺 `sta_mean`/`sta_std` 会直接炸（17.1.2 起落盘）——serve 端没有训练折数据，**无法"回退重算"**，
  这是故意设计的硬失败，不要绕过。

### 6.3 Step 2 — 换 serve（停旧起新）
```bash
pkill -f 'serve_htt[p].py'; sleep 1
cd ~/-project && PYTHONHASHSEED=0 \
  GOMP_SPINCOUNT=0 OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0 \
  [USE_IDS_AVG_APPROX=<按Step1> ]nohup ~/venv/bin/python3 scripts/diag/serve_http.py \
  --ckpt ~/project-107-<V>/outputs/<ckpt>.pt \
  --scaler ~/project-107-<V>/outputs/scaler.pkl --port 8000 > serve_<V>.log 2>&1 &
```
- **中间那三行是必带项（17.4.0 起）**：不带则 libgomp 让 worker 线程在**两个并行区之间空转自旋**
  （默认 `GOMP_SPINCOUNT=300000` + active 策略），而 serve 的负载是每 (pin, 方向, 模型) 一次
  forward + `.cpu()` 同步、一批候选几百个背靠背小并行区 ⇒ 线程池永不入睡。**实测同一批候选、
  同一时刻：裸奔 828% CPU vs 加这三行 3.3% CPU = 248×**（ckpt 载入阶段 70 vs 3 CPU-s）。
  这是**等待策略**，不改线程数、不改工作分解 ⇒ **位级中性**（A/B 两侧都打印 `Threads: 59`）；
  与 `torch.set_num_threads` 完全不同，那个会改 float 归约顺序、重演 17.2.4 刚定序掉的边序抖动。
  ⚠ **它不会消除与 zhirui（16 路 Xyce）的争用**：nice 19 下我们仍拿到 2.4 核 ⇒ CPU 时间从来不是
  瓶颈 ⇒ 那 ~7% IPC 损失是内存带宽/LLC/SMT 的共享微架构效应，这组 env 不改变它。
  `shadow_campaign.sh` 已把它写死为默认（`SHADOW_SERVE_ENV=''` 可回退做对照）；`run_shadow_batch.sh`
  起步会检查在跑的 serve 有没有带 `GOMP_SPINCOUNT=0`，没带就高亮告警（只告警不拦）。
- **`PYTHONHASHSEED=0` 是必带项（17.2.4 起）**：serve 的进程间数值抖动根因 = 边序随哈希种子变
  （`parse_netlist` 原 `list(set(edges))` → `edge_index` 行序 → float32 累加序 → 同一候选跨进程
  差 ~1e-7 相对 → 近并列候选互换名次 → 平均秩 ±0.5 → 选择遗憾两跑跨度 0.62pp）。根已由
  `sorted(set(edges))` 定序，这一项是兜底（防我尚未发现的其它哈希序依赖），**漏了不报错、只让
  数字悄悄变**，所以写进 runbook。线程变量（`OMP/MKL_NUM_THREADS`、`MKL_DYNAMIC`）实测零影响，
  不必钉；但 `_shadow_analyze.py` 的运行配置戳会记录这四个变量，便于事后对账。
- log 落在 `~/-project/serve_<V>.log`（先 cd 再重定向）→ tail 用全路径，别去 scripts/diag 下找。
- **模式 3 额外需要 idsavg GNN 折 ckpt**：默认 glob `~/idsavg17/idsgnn_fold*.pt`，可用 `IDSGNN_CKPT` 覆盖
  （如 `IDSGNN_CKPT='~/idsavg17/idsgnn_fold*.pt'`）。**glob 无命中 / ckpt 缺 `sta_mean` 一律启动即炸**（不静默退化）。
  多折默认**在预测空间等权平均**（= OOF 集成的行平均口径）。启动日志会打印折数、结构配置与各折 val_R²。
- **模式 3 的 `IDSGNN_FOLDS`（默认全折集成）**：逗号分隔折下标，如 `IDSGNN_FOLDS=0` 只用第 0 折。
  ⚠ 这不是纯省时旋钮 —— delay 模型**训练时**每张电路只拿到"留出它的那一折"的预测（单折、噪声大），
  而 serve 默认多折平均（更平滑）。Phase B 判负的机制假说正是"特征行间不一致破坏排序"，故**更平滑的
  serve 侧特征有可能反而变好**（shadow 与训练侧结论分叉）。要分离"特征本身"与"特征噪声水平"，
  全折与 `IDSGNN_FOLDS=0` 各跑一次 shadow 对照。

### 6.4 Step 3 — 判 serve 就绪：端口在听，不看 log
```bash
ss -ltnp | grep :8000    # 见到 LISTEN + pid 即就绪
ps -p <pid> -o pid,stat,%cpu,etime    # Sl 存活
```
- python 无 `-u` 时 stdout 重定向到文件是块缓冲 → **log 空正常**，别等 log 出字。若想看启动日志用 `python3 -u` 或另开窗口 tail。

### 6.4b Step 3b — 双端点 A/B：一趟 shadow 并排记录两个模型（17.1.7 / 内层 15.9.3）

> 前提：Rust 侧已装双端点版 `gnn_shadow.rs`（新增 env `GNN_PORT2` / `GNN_HOST2`）。**能省一趟的全部依据 = GNN 在 shadow 里是纯观察者**：`evaluate` 返回的始终是 inner 的 SPICE 真值，候选集与真值列全由 SPICE 决定 → **搜到的轨迹与 serve 挂哪个模型无关**，故一次 pass 就能并排记录两个模型的预测，两列天然同分母（这是 A/B 干净配对的前提，不是巧合）。

- **端口 → 模型的映射纯属外部约定**，Rust 不知道也不关心：**模型1 恒为 `GNN_PORT`(8000)，模型2 恒为 `GNN_PORT2`(8001)**。记录时必须写清哪个模型挂哪个端口，否则数字没法解读。
- 起第二个 serve = 照 §6.3 换端口 + 换 ckpt/env，两者都按 §6.4 判端口在听。
- `run_shadow_batch.sh` 内部**硬编码 `GNN_PORT=8000`、且不设 `GNN_PORT2`** → 只需在父 shell `export GNN_PORT2=8001`，12 个分片全部继承。
- ⚠ **别忘 export，且它不会报错**：忘了则 CSV 行尾没有 `gnn_pred2=`，解析脚本走"无第二列"分支、**不输出 A/B 节**（与旧版输出逐字节相同）——看起来像"跑了但没结果"。
- ⚠ **脚本的 serve 存活检查有漏洞**：`run_shadow_batch.sh:21` 的 `pgrep -f 'serve_htt[p]'` 只确认"**有** serve 在跑"，**不校验端口** → 8001 没起来、或起在了别的端口，脚本不会拦，只会让第二列整列缺失/NaN。**跑前自己按端口逐个确认**：`ss -ltnp | grep -E ':(8000|8001)'`。

### 6.5 Step 4 — 跑 shadow（46 电路，NetlistOpt 内）
```bash
bash ~/-project/scripts/diag/run_shadow_batch.sh   # 内部:rm 旧 CSV → 并行分片 → cargo test tl_opt_shadow_batch
# 另开窗口看进度：
tail -f ~/shadow_analyze.out
```
- shadow CSV 路径：`~/NetlistOpt/temp_sim_test/tl_opt_batch/**/gnn_shadow.csv`（每行 = 一次候选评估：eval_idx/iter/window/gnn_pred/true_delay/transistors[，gnn_pred2=…]）。**跑前 run_shadow 会把整个旧树 mv 归档**（17.3.16 起；此前是 `rm -rf`，等于每跑一趟就毁掉上一趟的原始数据）到 `~/shadow_archive/<TAG>_<时间戳>/`，标签取自**被归档树自己的** `RUN_INFO.txt`。⚠ 该归档**只在脚本内**——直接手敲 `cargo test` 会绕过它，追加写进上一趟的旧行，第二列会出现"前一半没有、后一半有"的半列假象（2026-09-15 实际踩到过；`docs/GNN_CODING_LESSONS.md` §7.2 有记）。**空树（无 `gnn_shadow.csv`）不归档**（17.4.0）——它只证明「这个目录被 mkdir 过」，归档出来就是一份看着像数据的 junk `unknown_*` 存档（2026-09-17 踩过）。
- ⚠ **三个跨 run 不变量**（破坏则 A/B 与历史口径都不可比）：
  - **① 不要删 `~/NetlistOpt/temp_sim_cache/`** —— 那是 `XYCE_CACHE` 的内容寻址延迟缓存，根在 `CARGO_MANIFEST_DIR/temp_sim_cache`，**不在** `temp_sim_test/` 下，故 `run_shadow_batch.sh` 把 `temp_sim_test/tl_opt_batch` 整体 **mv 归档**时**不会**误清它。缓存命中会跳过 Xyce（2026-09-15 那趟全程 Xyce 进程数 0），删了就白跑。
  - **② 两趟之间不要升级 Xyce** —— 缓存键 = tb 文件 + 每个 `.include` 的 DUT/model 文件哈希（`compute_deck_hash`），**不含 Xyce 版本** → 升级后旧缓存照样命中，会**静默**把两个版本的延迟真值混进同一张表。
  - **③ `SIM_OPTIONS` / `TL_MAX_ITERS` 两趟保持完全一致**。
- ⚠ **模式 3 的 NaN 会改变分母**：候选集要求"GNN 与 SPICE 都成功"（`true`/`gnn` 非 None）且 ≥4 候选；第二列 NaN 的行会被从 A/B 里剔掉 → **某些集可能因第二列 NaN 而落出 A/B**（脚本在 A/B 节头显式报出落掉的集数，并警告两节分母不同，跨节比指标要留神）。模式 3 每次 `/rank` 都要现场跑一遍 idsavg GNN，`GnnClient` 的空闲超时**17.3.19 起默认 300s**（`GNN_TIMEOUT_MS` 可覆盖；此前硬编码 15s）——15s 在大候选池被多分片争用时会被打穿，`read_to_end` 报错 → 整窗未命中 → 退到逐候选请求（serve 侧候选数 <2 时改走 `predict_avg_delay`）→ **同一个 `gnn_pred` 列里混进另一种量纲**，同时请求数 ×4.5、该分片从 ~99s 拖到 ~9928s。**跑完先看第二列 NaN 计数是否为 0**；非 0 则 A/B 的集数与主口径对不上。战役的**闸门②（兜底窗口 = 0）**就是拦这件事的。

### 6.6 Step 5 — 判收尾：首行时间戳被替换，不 grep 关键词
```bash
head -1 ~/shadow_analyze.out    # 首行 [date] 时间戳 = 本次启动时间 → 才是本轮完成
```
- ⚠ 判收尾看**首行时间戳被替换**（run_shadow 的 waiter 每轮先盖首行再写 `全部分片结束` + 调 `_shadow_analyze.py`）。**别 grep `全部分片结束`**——旧文件残留尾部也会命中，会误判成已完成。
- 完成标志：首行时间戳刷新 + 后面跟着本次聚合结果（recall/遗憾/Spearman 表）。

### 6.7 Step 6 — 对齐口径 + 记录 + 恢复 serve（本地）
- 读结果：`reports/_shadow_bench_final.txt`（服务器）+ `~/shadow_analyze.out`。**⚠ 2026-09-16 口径定案（DIFF §15）：默认口径 = `--group-by batch` = 714 集 = 部署真值**（`(circuit, iter, window_try)` = 一次 `pre_rank` 调用）；`--group-by window` = **106 集 = 历史池化口径**（把同电路 ~6.7 个批并成一集，集内混不同批的秩 → 系统性失真）。**两个口径都打出来、记录时都写并标清口径，禁止混读**（同 arm 只换口径严格@3 动 **+26.5~55.9pp**，且因臂而异；2026-09-17 改，旧记 +39.5 用了错标树的 74.4）。**别拿 PROJECT_LOG 训练侧 548/714 组混着比。全表总行 = 5398**（5390 成功 + 8 失败），m4 族 8 趟逐项相同 —— 这个数正是"轨迹与模型无关"的实证，也用来快速自检：拿到 5398 就说明候选集没被改动过。**另必看「输入污染检查」三项须全 0**（分组内 eval 去重丢弃 / 同电路 eval_idx 跨分组重复 / 同一 eval_idx 真值冲突）—— shadow CSV 是 `append(true)`，非 0 就说明快照被叠过行、集数是重复计数（OPERATIONS `:177` 记过一次真事故）。
- 同口径基准（m4三兄弟 Rust，DIFF §13.2/13.3）：纯拓扑 **nowave 遗憾 10.87%**（serve 交付基线）；iag GBDT15 严格@3 44.3% / 遗憾 12.19%；iaa ⚠ 15.21%。→ ⚠ **这三行都是 17.2.4 前的数**（边序未知，±0.31pp 带内），见下方权威表。
  - 📌 **交付基线的权威数（2026-09-17 重测，两次独立复现）= v2nowave42m4 `midpoint_ep250.pt`（sha1 09be9a3c53b90760，mtime 09-02 起未变）在 714 集 batch 口径下：严格@3 90.8% / 宽松@3 97.9% / 选择遗憾 5.93%（中位 2.13%）/ 两阶段 0.63%**（两次复现 = `~/shadow_archive/midpoint_ep250_20260916_233806` 与 `..._20260917_214841`，逐位相同；四口径全表见 DIFF §15.2 / PROJECT_LOG「部署口径定案」(2)）。**部署两项（严格k3 ≥90%、两阶段 ≤5%）都达标**；但 ① 口径被 `level2/DEPTH_MIX` 一家占 75% 权重，**②③④ 去重 / 去主导后严格 k3 全掉到 90% 线下**（80.1 / 81.5 / 76.7）→ **不能说「够了」**；选择遗憾 ① 5.93% 仍 > 5%（② 3.75%、④ 3.03% 已过线）→ 诚实读法 = **「部署口径达标，但达标压在一个电路上，且按池计权仍未过 5%」**。详见 DIFF §15.6。
  - 📌 **V3 交付栈部署读数（2026-09-22，18.5.1）= `v3nowave42 midpoint_ep200`（sha1 `51bd323382ecb9b7`，in=45 不带 env）在 714 集 batch 口径下：严格@3 75.8% / 宽松@3 97.5% / 选择遗憾 5.37%（中位 2.57%）/ 两阶段 0.58% / Sp 0.637 / 捕获率 97.33%**（hi-spread 591 集：严格@3 77.2% / 宽松@3 98.0% / 遗憾 6.24%）。对照交付基线 ep250（90.8 / 97.9 / 5.93 / 0.63）⇒ **严格@3 −15.0pp（hi-spread 集 −17.2pp）、宽松@3 持平、选择遗憾与两阶段小幅变好（四口径遗憾全部变好）** ⇒ **用户 2026-09-22 决定：交付基线维持 ep250，不切换**。⚠ **18.5.2 修正：两臂 `in_features` 实测都是 45**（纯拓扑、不带 env）⇒ **不掺特征表变更**，−15pp 归因到「V3 这一版训练出的模型」；未排除的只有单 ckpt 采样（ep200 vs ep250）。⚠ 106 集 window 口径下 V3 有 **12/106 量纲混合 ❌** ⇒ 该口径数不得引用。详见 DIFF §21。
  - ⚠ **用 `--root` 分析归档树时，戳里的 serve 身份是「当下在跑的」不是「那棵树历史上的」**（I20）：`分析根目录`/`CSV 快照` 来自 `--root`，其余（`serve ckpt`/`scaler`/`端口`/`环境`/`repo`/`脚本 sha1`）读活进程与当前工作区 ⇒ 两棵树戳的这几行**逐字相同也不代表**那棵树是本轮模型跑的。
  - 🆕 **✅ 74.4 错标定案（2026-09-17 / 17.4.0）—— 上面那条旧记录整行作废。** `74.4 / 97.6 / 7.23 / 1.66` **不是 `midpoint_ep250.pt` 的数**，它出自分片树 `~/tl_opt_batch_keep_0916` = `~/shadow_archive/unknown_20260916_223140`（用同一版分析器、同一 batch 口径重算，**与 17.3.11 那张表逐位相同**）。两条原假设**都已实测否定**：**(i) ckpt 没换** —— sha1 `09be9a3c53b90760` 与旧记逐位相同、mtime `2026-09-02 21:30:59` 起从未被覆盖；**(ii) Rust 没变** —— 三棵树（09-15 打包 / 09-16 / 09-17）的 **`true_delay` 多重集指纹全等 `cb0d7a96e6aada05`**、都是 46 CSV / 5398 行 ⇒ 候选生成逐值不变（`tree_fp` 不同是因为它含 `gnn_pred` 以外的列/路径，真值层是干净判别量）。**唯一变量是 `gnn_pred` 来自另一个模型**（佐证：那棵老树首行 `gnn_pred=7.562406e-11`，交付树为 `6.622111e-11`）⇒ 按 `§6.7:227`「2026-09-15 夜 8000 上挂的是 **`42b` ep250**」推断为 `42b`（⚠ 该步**仍是推断**，那棵树无 RUN_INFO；**可确证的只是「不是 `42m4` ep250」**）。**这是第 2 次同成因的 I15 类错标**：记的是「本该服务的模型」的身份，不是 8000 实际服务的模型。**防线已落地（17.4.0）**：`RUN_INFO.txt` 现在连同树记录 **`本轮 ckpt sha1` + `Rust 源指纹`（`~/NetlistOpt` 无 .git ⇒ 哈希源码）+ `repo dirty 指纹` + `serve 脚本指纹`**，且空树不再归档（本次正是靠这样一份存档定位的）。
  - 🆕 **部署选点政策（2026-09-17 定）：取平台末端（最后一个 midpoint），不取 shadow argmax；`best_model.pt` 不进部署候选。** ✅ **已落地（17.4.1）**：`config.py` `MIDPOINT_SELECT`（默认 `'last'`；env `MIDPOINT_SELECT=argmax_capture2` 可复现旧 run）+ `src/train_sweep.py` `pick_midpoint()`，**两处调用点共用**（训练内 midpoint 块 / `EVAL_ONLY=midpoint` —— 后者曾是一份未同步的复制）；排序同时改为**数值序**（`sorted(glob())` 是字符串序，`midpoint_ep100` 排在 `midpoint_ep50` **前** ⇒ 直接取 `[-1]` 会把「取末端」取反）。自检：`python scripts/diag/_t_midpoint_pick.py`（16 项）。四口径实证与推导见 PROJECT_LOG §17.4.0。要点：① GNN 在贪心里的唯一调用点是 `pre_rank`、**每 (轮次, 窗口) 一次** ⇒ 部署估计量是 **① 全部批**（工作加权期望），③ 等价于假设「GNN 每电路调一次」——它从没被那样调用过；② ② 的 161 个独立池只用于**配置信区间**（有效样本量），不替换点估计；③ 部署指标是**严格 k3 / 两阶段遗憾**，**不是** `判定` 行的选择遗憾（那个对应「完全不跑 SPICE」的更激进方案）；④ **shadow 从「选择器」降级为「闸门」** —— 每趟训练跑**一次**（~290s）判过不过线，**不做 ckpt 排名**（这才是不必每个 ckpt 都跑的正确解，依据是「`true_delay` 列与 ckpt 无关」已被闸门① 与 n_models=1 单调等价证明）；⑤ **免费的上游守卫** = 看每中点那行 `ep{N}: cap2=...`，val capture2 的 argmax 落在**平台内部**（安全）还是**末端**（训练被截断、规则不成立）。**按锚点找、别按行号**：`grep -n 'cap2={score' src/train_sweep.py`（17.4.1 时为 `:1128`；17.4.1 在文件顶插入 `pick_midpoint` 后本文件**下游行号整体 +55**，旧记的 `:1055` 已失效）。**该守卫已落进代码**：`pick_midpoint()` 每次选点都打印 `[选点] 规则=...` 与 `守卫⑤` 结论行（✓ 平台已形成 / ⚠⚠ argmax 落在末端）。
    > 下面那条「部署选型的杠杆在 epoch，不在换臂」**仍然成立，但答案变了**：臂内选点确实比臂间差异大一个量级 —— 正因如此**才不能靠 shadow 排名去选**（在不可分辨的差上取 argmax = 拟合噪声；已证严格@3 对 1 ulp 的敏感度是遗憾的 ~17 倍）。选点改由**不依赖 shadow 的确定性规则**给出，shadow 只当闸门。
  - 📌 **上一版的权威数（106 集 window 口径，保留存档）= `42m4` ep250 选择遗憾 11.13%**（严格k2 24.5 / 严格k3 34.9 / 宽松k2 46.2 / 宽松k3 66.0 / 两阶段 4.00%（中位 0.79%）/ Sp 0.134 / 79 集 13.66%），17.2.4 规范边序。旧记的 10.87% 与 10.66% 都是抖动抽样，**11.13% 恰是该带（10.51–11.13）最差值 → 旧值高估 0.47pp**。十点全表见 PROJECT_LOG §17.1.x、定因见 DIFF §13.7。**⚠ 该行的 epoch 标注本身存疑**（同 arm 另两处记录把同一读数标成 mid200，严格@3 差 4.7pp，见 DIFF §15.5）→ 跨 ckpt 选点结论冻结待 batch 口径重测。
  - ~~✅ 已定案（2026-09-15 同日对照跑）~~：`42b`（in=45, ep100）复跑 **14.26%**，同口径重跑交付基线 `42m4` ep250 得 **10.66%** → 历史 10.87% 被复现到 0.21pp 以内…**42b 那 3.60pp 差是真的**。部署口径三臂单调可分：`42m4` **10.66** < `42b` **14.26** < `gnn42b` **24.00** → **交付基线 42m4 未被撼动**。
    > ⚠ **本节数值已被 17.2.4 重算（2026-09-15 夜）**：三个数都带未知边序。规范序下 **`42m4` ep250 = 11.13%、`42b` ep100 = 14.03% → 差 2.90pp**（`gnn42b` 24.00 未重测，其结论方向不受影响）。**且那 2.90pp 有 1.61pp 是「42b 服务了自己五个 epoch 里最差的 ep100」**（该臂 ep150 = 12.42%），残余仅 1.29pp。"3.60pp 是臂间真差"这个读法**作废** —— 是「最差 epoch vs 最优 epoch」的对比。**A/B（模式 3）结论不受影响**：42b 是在服务自己最差 epoch 的条件下赢下全部 7 个判据的。
  - ⚠ **新教训（原因已分离，2026-09-15）**：**「服务 Best midpoint」不可默认 = 部署最优 —— 且判据与部署反序**：`Best midpoint` 的选点分数**在 17.3.7 已换**（选点块 = 「中途快照回溯」节，**17.4.3 时为 `src/train_sweep.py:1101-1176`**，用 `grep -n '中途快照回溯' src/train_sweep.py` 定位；⚠ **行号随版本漂移，别做常数换算**（17.4.1 时该块为 `:1075-1150`，17.4.1 前旧记 `:1012-1073`）—— 现在是**逐个 midpoint 在 val 上算 `capture2_pct`**（`BEST_MODEL_METRIC='capture2'`，`config.py:42`；理由是 capture2 直接对应部署口径），**选点规则 17.4.1 起改为政策性的「取平台末端」、不再取 argmax**（17.4.3 时 `score` 见 `:1145`，`pick_midpoint(pairs, MIDPOINT_SELECT)` 见 `:1159`；详见本节 ⑤ 与 DIFF §15.8）。~~旧分数 `100·r3+50·r2+0.3·sp−0.2·regret+0.1·cap`（原 `train_sweep.py:978`，recall@3 主导）~~ **已作废（该行 17.3.7 已删）**——它正是下面那条「与部署反序」的来源（留档注释现于 `:1139-1141`）。两个 selector 仍要分清：`best_model.pt` 走 `smoothed_rel_err`，midpoint 走 capture2（17.2.6 更正）。实测该 `score` 序与 Rust 部署序 **Spearman = −1**（42b 五点完全反序）：`42m4` 挑中五点最优（ep250 11.13%），`42b` 挑中五点**最差**（ep100 14.03%；而该臂训练侧唯一最优 ep150 恰是部署最优 12.42%）。**训完先别信 Best midpoint 这行**。**臂内选点（1.61–2.65pp）比臂间差异（均值 0.46pp）大一个量级** → 部署选型的杠杆在 epoch 上，不在换臂上。换 ckpt 前后各扫一趟很便宜（`_shadow_ckpt_sweep.sh`，缓存全命中，单点 ~3min）。⚠ **但 17.2.4 之前的"每个 ckpt 测多次取中"是错的**（那是在测抖动，不是测模型）——**现在每 ckpt 一次即可**。详见 PROJECT_LOG §17.1.x + DIFF §13.7。
  - 🆕 **17.2.7 必看：分析器新增「并列/一致性诊断」节**（在「跨度>10% 子集」之后；需先 `git pull` 到 17.2.7 才出现）：
    ① **量纲混合守卫必须为 0** —— 非 0 表示该集 `gnn_pred` 混了「候选集内秩」与「原始延迟」两种量纲，排序无意义，先查 serve 预排序整窗是否命中；
    ② **兜底窗口 >0** 表示该批 serve 预排序整窗未命中（集内排序仍有效，仅分数分辨率受 CSV 7 位有效数字限）；
    ③ **重复网表集** = 真值内有完全相等候选 → recall@3 天然 <100%，是**任务天花板**，别算作模型缺陷；
    ④ **预测第3小==第4小** 的集数 = 并列打破规则的**影响面**（训练侧与 Rust 侧现已同一条规则：stable + 组内行序）。
    另外两条口径：**带精排路线下选点看「严格 recall@3」** —— 它逐集恒等于 `regret_2stage == 0`，且对并列免疫（取前 3 内真值最小值，集合内部次序不影响）；**两端 recall@3 不可对读**（总体 / 被排序的量 / 并列约定三处不同源，106 集 1 SE ≈ 5pp），详见 DIFF §13.7.4。
  - **无 ids 列(in=45)的模型与此对比才干净**；带 ids 列(in=46) serve 端本身就净伤害 —— ⚠ 该结论来自**近似**列（iag GBDT15 / iaa 线性，两个都是 serve 端净伤害的近似），**不能外推到模式 3（GNN 现场列）**：模式 3 的那一列本身是准的（R²≈0.79），它测的是"GNN ids 特征入 delay"在部署口径下的效果，是一个独立问题（17.1.5 Phase B 训练侧已判负，Rust 复验是为了确认部署口径同结论，**现已确认：部署口径同向且负得更重，见上**）。
- 记 PROJECT_LOG（Rust 表行）+ 分析结论 → DIFF + 同步 I7 + 版本化 commit；push 仅按要求。
- **恢复 serve**：非交付路线的模型验完，按 Step 2 换回 v2nowave42m4（`--ckpt ~/project-107-v2nowave42m4/outputs/midpoint_ep250.pt`；ckpt sha1 应为 **`09be9a3c53b90760`**、`in_features` 应为 **45**（2026-09-22 实测；**不是旧记的 14**）、**不带 `USE_IDS_AVG_APPROX`**）。⚠ **「不带 env」这句已作废（17.4.0）** —— §6.3 那三行（`GOMP_SPINCOUNT=0 OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0`）**加上 `PYTHONHASHSEED=0`** 现在是**每一条** serve 启动命令的必带项，交付基线的 serve 也一样。
  - ⚠ **当前状态（2026-09-22）**：8000 上挂的是 **V3 `v3nowave42` ep200**（本轮 V3 shadow 换上去的，**待收**）。**恢复前先按 Step 2 换回 `midpoint_ep250`**，并顺手确认换上的不是 `ep300`（09-17 ckpt-sweep 末位遗留过 `ep300`，见 I9）；换完按 §6.3 用 `tr '\0' ' ' < /proc/$PID/cmdline` 复核 argv 里的 `--ckpt`（`ps -o args` 会被列宽截断）。

### 6.8 分析脚本 / 存档位置
- 分析脚本在 `~/-project/scripts/diag/`（`_` 前缀，与本地 scripts/diag/ 双向同步）；关键结果存档 `reports/_shadow_bench_final.txt`。
- **`_shadow_analyze.py`（17.1.7 起支持双列）**：CSV 里**没有** `gnn_pred2=` 时，输出与旧版**逐字节相同**；有则文末多一节「两列并排 A/B」—— 同一候选集、同一批行上分别用两列各算一遍指标 + 逐集配对胜负（严格同分母）。第一列恒为 `gnn_pred`(8000)，第二列 `gnn_pred2`(8001)。取 A/B 结果时的命令同 §6.5 的收尾命令，只是**必须先把 17.1.7 拉下来**（`cd ~/-project && git pull`），否则跑的还是旧脚本、A/B 节不会出现。
- **双列解析的本地回归测试 = `scripts/diag/_shadow_analyze_2col.py`**（跑在本地，**不需要服务器**）：造三棵同数据树 —— 旧格式 / 第二列 `2-pred`（排序反转）并埋 NaN / 第二列 = 真值（oracle）—— 断言 ①旧格式无 A/B 节且主口径与明细逐字节不变；②NaN 集只从 A/B 剔除、计数如实上报；③用另一份 numpy 实现独立复算每集遗憾与配对胜负并对账；④oracle 树模型2 遗憾恒 0、严格 k=3 = 100%。
- **`_shadow_ckpt_sweep.sh`（17.2.0，17.2.1 补宽松 recall 显示；17.2.4 起启动行钉 `PYTHONHASHSEED=0`）**：逐个换 serve 扫一个臂的**多个已存 midpoint**，每 ckpt 一份 `~/sweep_<TAG>_ep<N>.out` + 文末一张汇总表。用法 `bash ~/-project/scripts/diag/_shadow_ckpt_sweep.sh ARM_DIR ARM_TAG [EPOCHS...]`（省略 EPOCHS 默认 `50 100 150 200 250`；单点 ~3min，缓存全命中）。⚠ **末位 epoch 会留在 8000** —— 若不是交付基线（`42m4` ep250），跑完按 §6.7 换回。⚠ 汇总行的「<0.3pp 视为打平」是**抖动期的陈旧阈值**（0.3 是当时估的噪声底，该说法已作废），17.2.4 后无抖动，判读应看**中位数与逐集配对**（均值差在 106 集上以 0.94pp/集为粒度，尾部几个集就能撬动）。
- **`_t_serve_repro.sh`（17.2.4，判决实验）**：serve 进程间可复现性验收 —— 两个**全新 serve 进程、刻意不钉任何 env**，各跑一趟 level2 分片，逐字节比对 md5 清单。通过 = 边序不再依赖哈希种子。用法 `bash ~/-project/scripts/diag/_t_serve_repro.sh CKPT SCALER [LEVEL]`（~3min；**不碰 `~/NetlistOpt/temp_sim_cache/`**）。

## 7. Git 同步现状（⚠ 三处代码源不同步）

| 源 | HEAD | 说明 |
|---|---|---|
| 本地 project（本机） | 17.2.4（已 push GitHub） | 最新；docs/setup_exp.sh/src 均在此 |
| GitHub `10.3.3-fix-earlystop` | 17.2.4 | 训练代码源（服务器 setup_exp.sh clone 用它） |
| 服务器 `~/-project` | HEAD `ca07e94`(16.10.0) + 逐文件替换 | **分析工作区，落后 79、领先 0**；setup_exp.sh 是手动上传的最新版（16.11.32 两个新变体已在）；数据源 data/ 全。**17.0.0 起：新分析脚本改走 GitHub 分支 git 拉取，不再 scp 单文件**（见下行规则）。⚠ **2026-09-15 起这棵树是混源的**：17.2.4 的六个文件已逐个替换进去（`serve.py` / `graph_builder.py` / `_t_serve_repro.sh` / `_shadow_analyze.py` / `_shadow_ckpt_sweep.sh` / `OPERATIONS.md`，各自 blob sha1 已对上），但 HEAD 与索引仍是 16.10.0 → **"某个文件是哪版"只能靠 `sha1sum` 对 `git show <rev>:<path> \| sha1sum` 判**（`git status`/HEAD 不是代理，`serve.py` 曾是 16.11.34 而 `graph_builder.py` 是 `ca07e94` 那版）。归一见 #58。 |

- 规则：**改训练代码 → 本地提交 push GitHub → 新 run clone 生效**；**分析脚本/新能力（含 17.0.0 起的 diag 模型脚本）→ 一律随本地 commit 进 GitHub 分支，服务器用 git 获取（`git clone -b 10.3.3-fix-earlystop …` 或 `git pull`），不再 ssh_upload/scp 单文件同步**（2026-09-04，用户定；§4 16.2.1 路径推广到分析脚本）；改 setup_exp.sh → 顺手 `ssh_upload` 到服务器 `~/-project/`（仅脚本层即时项，不 push 也可）。
- NetlistOpt（Rust）：本地与服务器各自独立，服务器源码曾落后 → 整体 `tar` 同步过；改动 Rust 需手动同步服务器（**无 git 远端**，内层 HEAD 现为 `15.9.3`）。⚠ 本地**编译不了**（rustup 只装了 `x86_64-pc-windows-msvc`，无 VS toolchain / gcc / clang；Git Bash 的 coreutils `link.exe` 还会冒充链接器报 `link: extra operand`）→ **服务器 `cargo build --release --tests` 是唯一编译关**，本地只做语法/逻辑自检。

## 8. 数据 / 缓存要点

- 训练数据在服务器 `~/project-107-<V>/data/`（setup clone 后由 CACHE_SEED 或 `~/-project/data` 种子）；**别手动改 mtime**（缓存键依赖）。
- `cache107$V/graphs/` 图缓存键 = 数据 mtime + STRUCT_MODE + 特征字段；特征不变的 run 可跨目录复用（cp -a）。
- 大表：GBDT15 近似批量向量化后峰值 RSS ~2.46GB；训练内存历史教训见 CODING_LESSONS（swap 40G 事故）。
