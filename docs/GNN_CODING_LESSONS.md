# GNN 项目编码踩坑实录（CODING LESSONS）

> 本文件记录本项目**写代码 / 调试 / 部署过程中踩过的坑、定位方法和验证手段**，与 `GNN_PROJECT_REQUIREMENTS.md`（要求/规范）分开管理。
> **何时必读**：改数据加载 / 缓存 / 训练管线、写诊断脚本、部署到服务器前，先通读相关小节（特别是 §1 内存与缓存、§4 验证方法）。
> 格式：`版本 - 现象/原因 - 正确做法`。教训按主题归类，不按时间。

---

## 1. 内存与缓存（本项目的头号坑区）

### 1.1 wave 列是内存真凶（16.4.0）
- **现象**：单 run 稳态 RSS 高达 ~41GB；4 run 并发 OOM 连环杀；LRU 图缓存修了没用。
- **原因**：`transistor_wave_json` 一个列占动态 df **93% 内存**（7.22GB / 7.76GB，每行 ~16KB 波形 JSON）。更糟的是 `DelayDataset` 三个实例（train/val/test）**各自重读全部 parquet** → wave 字符串被持有 5 份 + 3 次全量重读的堆残留 ≈ 41GB/run。图本身只有 ~87MB。
- **正确做法**：① `DelayDataset.__init__` 支持传入已过滤 df（`prefiltered=True` 直接引用不复制），主进程只读一次；② wave OFF 时**按列读取**（`pd.read_parquet(columns=...)` 排除 wave 列，7.76GB→0.54GB）；③ 用完后 `del` 全量 df + `gc.collect()` + `malloc_trim(0)`（glibc 才归还堆）。
- **教训**：先做内存账目（df.memory_usage(deep=True) / smaps）再动手修，别修假想敌。

### 1.2 缓存键 = 数据文件 mtime（16.3.1）
- **现象**：复制了完整缓存但新 run 还是「Graph cache: outdated, clearing」从头重建。
- **原因**：图缓存键 = `hash(graph_builder.py) + md5(各parquet 的 int(mtime)) + STRUCT_MODE`（train_sweep.py:142-149, 192）。新 clone 的数据 mtime ≠ 旧数据 mtime → 键变 → 清空重建。
- **正确做法**：复用缓存必须**连数据一起 `cp -a` 复制**（保 mtime）；跨目录/重拷贝后 mtime 变 → 自动重建（`.version` 机制是安全网，不是 bug）。
- **教训**：`_check_cache_dir` 按 `.version` 文件比对，**只查键、不查完整性**——残缺缓存（构建中断留下的）可能被当成有效。

### 1.3 图缓存键还含 gb 哈希（16.4.0）
- **现象**：`CACHE_SEED` 连数据保 mtime 复制了，缓存仍被拒（`outdated, clearing`）。
- **原因**：键还含 `graph_builder.py` 内容哈希；旧 run 目录的代码与当前代码不同（gb 哈希 57640802 → e18e16cd）→ 旧图结构不兼容。
- **正确做法**：`.version` 拒绝是**安全行为**，说明旧缓存本来就不可用；重建即可。判断键差异：`cat <cachedir>/graphs/.version` 与当前 run 的对比，前 8 位=gb 哈希、中 8 位=数据 mtime 哈希、尾部=STRUCT_MODE 可读串。
- **注意**：离群点掩码键**不含 gb 哈希**（含 train_ids+配置+TRAIN_SEED+数据 mtime）→ 同 seed 同配置下掩码可跨代码版本复用；多 seed 各算各的（文件名键控，共存不互清）。

### 1.4 ps RSS 重复计数 COW 共享页（16.4.0）
- **现象**：`ps -eo pid,rss` 各进程 36-38GB，总和 138GB > 60GB 物理内存，看着像灾难。
- **原因**：worker 从主进程 fork（COW），共享页在每个进程的 RSS 里**各计一次**；实际唯一占用 ≈ `free -g` 的 used。
- **正确做法**：看 `grep -E 'Private|Shared' /proc/<pid>/smaps_rollup` 区分——`Shared_Dirty` 大 = fork 共享（一份）；`Private_Dirty` 大 = 真独有（真凶）。真实页数 = `VmRSS + VmSwap`（两个进程 VmData 相同但 RSS/swap 分摊不同是分页状态差异，不是配置差异）。

### 1.5 OOM 连环杀与错峰（16.4.0）
- **现象**：4 run 并发，几个 run 死在不同阶段（模型初始化 / 离群点基训 / 训练），无 traceback。
- **原因**：单 run 稳态 ~20GB × 4 ≈ 80GB > 60GB → OOM 杀手逐个清，谁在重阶段谁先死。
- **正确做法**：① 查日志**死点** + `free -g` + dmesg；② 幸存者往往是因为跳过了重阶段（如离群点缓存命中）——这是线索；③ **启动必须错峰**：setup 峰值（读全量+建子集+建图 ≈ 16-20GB/run，~10 分钟）4 个同时仍会爆，稳态 4×10-13GB（16.4.0 后）才安全。错峰 ~15 分钟/个。

## 2. 进程与命令陷阱

### 2.1 pkill/pgrep 自匹配自杀（§6.1 已有，重申）
- `pkill -f 'xxx'` 会匹配执行命令的 shell 自己（命令行里含 'xxx'）→ 断连。用 `pkill -9 -x <进程名>` 或括号技巧 `pkill -9 -f 'main[.]py'`。
- `pgrep -f` 只接受**一个** pattern（多个 `-f` 报错）。
- 看内存用 `ps -eo pid,rss,cmd | grep 'python3 -u main[.]py' | grep -v grep`（ps 自带列过滤，不用二次 grep 也行：`ps -eo pid,rss,cmd -C python3`）。

### 2.2 日志文件名与变量前缀（16.4.0）
- 变体名 `$d=v2wave42` 已含 `v2` 前缀时，日志是 `train107$d.log`（=`train107v2wave42.log`），**不是** `train107v2$d.log`——拼错过一次导致 tail 全空，白查一轮。复制命令前先 `ls` 确认实际文件名。

### 2.3 长命令/管道陷阱
- 管道会吞掉前段命令的退出码和 stderr：`dmesg 2>/dev/null | tail` 在 dmesg 无权限时静默无输出（应 `dmesg 2>&1 | tail`）。
- 服务器粘贴超长命令可能被截断 → 拆短命令或写脚本上传。

## 3. 本地开发环境（Windows）陷阱

### 3.1 PowerShell
- `string.Replace([char]0x2713, 'OK')` 会选中 **char 重载**报「字符串长度只能为一个字符」→ 用 `[string][char]0x2713`。
- git push 成功但 PowerShell 报 `exit code: 1`（stderr 被当错误流）——看输出里的 `.. -> ..` 行确认成功，别被退出码误导。
- 中文/✓ 在 GBK 控制台 `UnicodeEncodeError`——脚本 print 避免非 ASCII，或 `$env:PYTHONIOENCODING='utf-8'`。

### 3.2 一律先写脚本文件再跑
- PowerShell 不支持 heredoc（`python - <<'PY'` 直接 ParserError）→ 长逻辑先 `write` 成 `.py` 文件再执行。

## 4. 验证方法（重构数据/缓存管线的标准流程）

### 4.1 等价重构先验数据边界（16.5.0）
- 改数据路径前，先确认数据里没有落在过滤边界上的样本：实测 DELAY 全部在 (1e-12, 1e-8) 内（0 行越界）→ 主进程过滤与 dataset 重读过滤**逐位等价**，改动才安全。任何「行为不变的重构」都要先做这个检查。

### 4.2 双路径 sanity（16.4.0，脚本 `scripts/diag/_sanity_memfix.py`）
- 新旧两条路径（重读 parquet vs 传入子集）对比：**行集 / 行序 / DELAY / `__getitem__` 输出（x、edge_index、y、switching_pin、corner_cond）逐位一致**才算通过。Data 对象字段名是 `x`（不是 node_static）。

### 4.3 启动流程验证（16.6.0）
- 缓存是否命中：日志**无**「Graph cache: initializing/outdated」+ 有「seeded cache」行。
- 离群点是否命中：日志直接「加载离群点清洗缓存: outlier_keep_<hash>.npy」，**无**「Training base model on ...」。

## 5. 版本与提交

### 5.1 merge 提交也要带版本号（16.2.2）
- 曾把 merge 提交命名为 `merge: ...`（无版本号）违反命名规范，已推送无法改名 → 教训：**所有提交（含 merge）一律 `大.小.更小 - 描述`**；版本单调递增，绝不回退。

### 5.2 提交前自查
- `git log --oneline -3` 看当前版本；`git status --short` 别把临时 tar 包/缓存带进提交。

## 6. RESUME 续跑模式（16.8.0）的隐蔽坑

- **env 变体的配置不在 config.py 里**：`v2ia`/`v2cov25` 靠 `export WAVE_FIELDS/WAVE_COVERAGE` 生效（不是 sed）——续跑若跳过变体块，env 丢失 → 配置**静默**变回全 wave。对策：RESUME 仍执行变体块（sed 幂等 + export 必须重跑），只跳过 rm/clone。
- **数据种子会破坏缓存键**：重跑数据种子覆盖目录数据 mtime → 图缓存键变 → 已建的缓存白重建。对策：RESUME 且目录已有 `data/batch_v2_rest` → 跳过数据种子。
- **缓存种子覆盖残留**：master 的 `.version` 覆盖目录 → 残留旧键 .pt 可能被静默使用。对策：RESUME 且目录已有 `graphs/` → 跳过缓存种子（增量续建）。
- **重复启动同一目录**：RESUME 前检查 `/proc/*/cwd` == run 目录的 main.py，有则报错退出（防双写 checkpoint）。
- **变体名必须与目录一致**：RESUME 传错变体名 → 配置/日志/缓存目录全错 → 启动时校验并退出。
- **RESUME 日志用 `>>` 追加**（保留失败现场），fresh 用 `>` 截断。
