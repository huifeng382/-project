# 运行手册（OPERATIONS）——服务器 / 变体 / 开关 / 现状 / serve / git

> 与 `TESTING_GUIDE.md`（测试流程）、`GNN_PROJECT_REQUIREMENTS.md` §4/§6.1（启动/命令安全规范）配套。
> 最后更新：2026-09-03（16.11.35）。

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
| `v2kdwave42iaa<seed>` | **v2wave42m4 教师蒸馏**(reg+rank)；⚠ 名带 iaa 但学生**实为纯拓扑**——KD 分支 `USE_TRANSISTOR_WAVE=False` 把近似列门控挡掉（data_loader L507/L510），`USE_IDS_AVG_APPROX=1` 未生效 | #12；教师软标签在教师 outputs；「iaa 学生+KD」格从未真正测过（DIFF §13.4 更正） |
| `v2kd<teacher><mode><seed>` | 旧蒸馏（teacher=123/ENS；mode=reg/rr） | 15.2；学生无近似特征（已弃路线） |
| `rankloss1/2`、`bmsm`、`es`、`anneal`、`bestrank`、`seed*` | base 系调参/选点/种子 | V1 时代为主 |
| `struct*`（structlogic/rich/elec…） | STRUCT_MODE 消融 | 默认 logic_only |

- 命名规则：变体名 = 特征/目标前缀 + seed；尾部 `m4` 仅是历史标识（数据默认已含 m4）。
- **教师/学生关系（16.11 系）**：v2wave42m4 = wave 教师（Test 13.3% / 遗憾 0.65% / Spearman 0.688）；v2nowave42m4 = 纯拓扑无近似（**Rust 选择遗憾 10.87% = m4 系最优，serve 交付走此路线**，DIFF §13.3）；v2iaa42m4 = 线性近似（纯 huber，Rust 记录 ⚠ 不可复现、serve 净伤最大）；v2iag42m4 = GBDT15（Test 23.00%，Rust 严格@3 44.3%/遗憾 12.19%——**最不伤的近似**，排序质量最高但 top1 落点输 nowave）；v2iaar42m4 = 线性+rank（训练中）；v2kdwave42iaa42 = wave 教师 KD 学生（**已验 Rust：名带 iaa 实 in=45 纯拓扑，遗憾 14.97% vs nowave 10.87% 双端不赢，DIFF §13.4**）。

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

## 5. 当前状态（2026-09-03，16.11.38）

**正在跑（1 run）**：

| run | 内容 | 状态 |
|---|---|---|
| `~/project-107-v2iaar42m4` | 排序 loss #8 | 训练中（9-3 启动，epoch ~179） |

**已训完（2026-09-03）**：`v2kdwave42iaa42`（wave 教师 KD #12）——194 epochs plateau 早停；train-side ≈ v2iaa42m4、**无 KD 增益**；**Rust shadow 已跑完**（22:03，106 集，选择遗憾 14.97% vs nowave 10.87%，双端无增益，DIFF §13.4 收口）。

**其他现场**：
- serve：8000 端口现挂 **v2kdwave42iaa42 midpoint_ep100**（纯拓扑 in=45，无 env；serve log `~/-project/serve_v2kdwave42iaa42.log`）——其 Rust shadow 已跑完（22:03，遗憾 14.97%，DIFF §13.4，非交付路线）。**待换回交付基线 v2nowave42m4**（midpoint_ep250，命令见 §6 Step 6）。
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
~/venv/bin/python3 - <<'PY'
import torch
sd = torch.load("~/project-107-<V>/outputs/<ckpt>.pt", map_location="cpu", weights_only=False)
print("convs.0.lin_rel.weight:", tuple(sd["convs.0.lin_rel.weight"].shape))
PY
```
- **in_features = 45 → 无近似列（纯拓扑，nowave 风格）→ serve 不带 `USE_IDS_AVG_APPROX`**
- **in_features = 46 → 有 ids_avg 近似列 → serve 带 `USE_IDS_AVG_APPROX`（线性近似=1，GBDT15=2，对齐训练 config）**
- ⚠ 教训：`v2kdwave42iaa42` 名字带 "iaa" 但 ckpt 实测 **45 → 纯拓扑、不带 env**。盲带 `=1` 会启动失败 `size mismatch ... [256, 45] from checkpoint ... [256, 46]`（serve.py L316-317 按 env 加 extra_dim）。

### 6.3 Step 2 — 换 serve（停旧起新）
```bash
pkill -f 'serve_htt[p].py'; sleep 1
cd ~/-project && [USE_IDS_AVG_APPROX=<按Step1> ]nohup ~/venv/bin/python3 scripts/diag/serve_http.py \
  --ckpt ~/project-107-<V>/outputs/<ckpt>.pt \
  --scaler ~/project-107-<V>/outputs/scaler.pkl --port 8000 > serve_<V>.log 2>&1 &
```
- log 落在 `~/-project/serve_<V>.log`（先 cd 再重定向）→ tail 用全路径，别去 scripts/diag 下找。

### 6.4 Step 3 — 判 serve 就绪：端口在听，不看 log
```bash
ss -ltnp | grep :8000    # 见到 LISTEN + pid 即就绪
ps -p <pid> -o pid,stat,%cpu,etime    # Sl 存活
```
- python 无 `-u` 时 stdout 重定向到文件是块缓冲 → **log 空正常**，别等 log 出字。若想看启动日志用 `python3 -u` 或另开窗口 tail。

### 6.5 Step 4 — 跑 shadow（46 电路，NetlistOpt 内）
```bash
bash ~/-project/scripts/diag/run_shadow_batch.sh   # 内部:rm 旧 CSV → 并行分片 → cargo test tl_opt_shadow_batch
# 另开窗口看进度：
tail -f ~/shadow_analyze.out
```
- shadow CSV 路径：`~/NetlistOpt/temp_sim_test/tl_opt_batch/**/gnn_shadow.csv`（每行 = 一次候选评估：eval_idx/iter/window/gnn_pred/true_delay/transistors）。**跑前 run_shadow 会整体清空该目录**（CSV append 会混旧行，勿手动残留）。

### 6.6 Step 5 — 判收尾：首行时间戳被替换，不 grep 关键词
```bash
head -1 ~/shadow_analyze.out    # 首行 [date] 时间戳 = 本次启动时间 → 才是本轮完成
```
- ⚠ 判收尾看**首行时间戳被替换**（run_shadow 的 waiter 每轮先盖首行再写 `全部分片结束` + 调 `_shadow_analyze.py`）。**别 grep `全部分片结束`**——旧文件残留尾部也会命中，会误判成已完成。
- 完成标志：首行时间戳刷新 + 后面跟着本次聚合结果（recall/遗憾/Spearman 表）。

### 6.7 Step 6 — 对齐口径 + 记录 + 恢复 serve（本地）
- 读结果：`reports/_shadow_bench_final.txt`（服务器）+ `~/shadow_analyze.out`。**口径固定 = 106 集 / 5390 行 / 8 失败**（pipeline 窗口级 ≥4 候选计数），别拿 PROJECT_LOG 训练侧 714/548 集混着比。
- 同口径基准（m4三兄弟 Rust，DIFF §13.2/13.3）：纯拓扑 **nowave 遗憾 10.87%**（serve 交付基线）；iag GBDT15 严格@3 44.3% / 遗憾 12.19%；iaa ⚠ 15.21%。**无近似列(in=45)的模型与此对比才干净**；带近似列(in=46) serve 端本身就净伤害。
- 记 PROJECT_LOG（Rust 表行）+ 分析结论 → DIFF + 同步 I7 + 版本化 commit；push 仅按要求。
- **恢复 serve**：若非交付路线的模型验完，按 Step 2 换回 v2nowave42m4（`--ckpt ~/project-107-v2nowave42m4/outputs/midpoint_ep250.pt`，不带 env）。

### 6.8 分析脚本 / 存档位置
- 分析脚本在 `~/-project/scripts/diag/`（`_` 前缀，与本地 scripts/diag/ 双向同步）；关键结果存档 `reports/_shadow_bench_final.txt`。

## 7. Git 同步现状（⚠ 三处代码源不同步）

| 源 | HEAD | 说明 |
|---|---|---|
| 本地 project（本机） | 16.11.32（已 push GitHub） | 最新；docs/setup_exp.sh/src 均在此 |
| GitHub `10.3.3-fix-earlystop` | 16.11.32 | 训练代码源（服务器 setup_exp.sh clone 用它） |
| 服务器 `~/-project` | 16.10.0 + 本地改动 | **分析工作区，落后**；setup_exp.sh 是手动上传的最新版（16.11.32 两个新变体已在）；数据源 data/ 全 |

- 规则：**改训练代码 → 本地提交 push GitHub → 新 run clone 生效**；改 setup_exp.sh → 顺手 `ssh_upload` 到服务器 `~/-project/`（不必等 push）；分析脚本 → 本地 scripts/diag 与服务器双向同步。
- NetlistOpt（Rust）：本地与服务器各自独立，服务器源码曾落后 → 整体 `tar` 同步过；改动 Rust 需手动同步服务器（无 git 远端）。

## 8. 数据 / 缓存要点

- 训练数据在服务器 `~/project-107-<V>/data/`（setup clone 后由 CACHE_SEED 或 `~/-project/data` 种子）；**别手动改 mtime**（缓存键依赖）。
- `cache107$V/graphs/` 图缓存键 = 数据 mtime + STRUCT_MODE + 特征字段；特征不变的 run 可跨目录复用（cp -a）。
- 大表：GBDT15 近似批量向量化后峰值 RSS ~2.46GB；训练内存历史教训见 CODING_LESSONS（swap 40G 事故）。
