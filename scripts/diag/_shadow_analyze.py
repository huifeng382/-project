"""10.3 判定分析：从 shadow 批量输出聚合 recall@top-3 / 选择遗憾 / Spearman。

输入：~NetlistOpt/temp_sim_test/tl_opt_batch/**/gnn_shadow.csv（每行 = 一个候选评估）：
  eval_idx=N, iter=I, window=W, gnn_pred=..., true_delay=...[, transistors=...][, gnn_pred2=...]
第二列 gnn_pred2 由 Rust 侧 env GNN_PORT2 启用（同一次 run 并排记录第二个模型）；
**缺列时本脚本输出与旧版完全一致**，多出的是文末「两列并排 A/B」一节。

统计单位（17.3.9 起默认 = --group-by batch）：
  每个 (电路, iter, window_try) 的候选集 —— **即 tl_opt.rs 的 pre_rank 收到的那一批**。
  为什么这是唯一正确的口径（三处源码证据）：
  ① gnn_pred 是**批内平均秩**而非延迟：serve.py predict_rank_batch 在 n>=2 时做
     competition ranking（1-based、并列取平均秩），n==1 时才写原始延迟（~1e-11），
     两者塞进同一个 avg_delay 字段 → 跨批的 gnn_pred **不在同一把尺子上**。
  ② GNN 的「集合」= pre_rank 收到的 mods = 某一个 (轮次, 窗口) 的全部候选：
     tl_opt.rs Pass2 `evaluator.pre_rank(&mods, eval_idx)`，mods 来自 prepared，而
     prepared 在 Pass3 被 `for .. in prepared` 移动消费 → 每个窗口迭代重建。
  ③ CSV 的 window 列 = window_try_idx（gnn_shadow.rs 的 format 串），是
     0..neighbor_retries 的**重试计数器、每轮从 0 重数**；iter 列 = total_iters
     （tl_opt.rs 的 `while total_iters < max_iters`）。两者都是计数器 →
     同一个 (电路, window_try) 必然横跨多轮。

  旧口径 --group-by window = 每个 (电路, window_try) 的池化集【仅用于复现 17.3.9 之前的数】。
  它把 ~6.7 个批池在一起，而每批恰有一个候选秩=1.0 → 池化后大量候选并列在 1.0；
  per_window_metrics 用 argsort(kind="stable")、且行序已按 eval_idx 升序定序 →
  「GNN 前3」实为**最早那 3 批各自的第 1 名**，故其 recall/遗憾不是部署性能。
  同理，本文件各条守卫（量纲混合 / 兜底窗口 / 并列影响面）都预设「集 = 一次 pre_rank 的批」，
  池化会破坏该前提 → window 模式下这几行读数不可信（脚本会在该处显式警告）。

跨集重复度（17.3.10）：
  batch 口径把集数从 106 抬到 714，但同一电路的不同 (iter, window_try) 会出现**逐字节相同**的
  指标行（regret/cap2/spread 全等）→ 小子电路的搜索在几轮内走完候选空间、之后每轮重新提出
  同一批网表（gnn_shadow.rs 的缓存键 = m.to_tl_text()，同网表 → 同预测；同网表 → 同 Xyce
  结果 → 整行一致）。若成立，714 就不是 714 个独立样本：重复观测不带来信息、却加重权重，
  并让「逐集配对检验」的独立性假设失效。
  文末「跨集重复度」块给出 独立集数 / 重数分布 / 真包含子集。判同用**真值列**
  (true_delay, transistors) 的原始文本 → 与模型无关，换 ckpt 不改变判同结果。
  ⚠ 该块**只计数，不改上面任何指标**（去重重算另立版本）。

交付口径对照 + 输入污染检查（17.3.11）：
  跨集重复度暴露了一件事：714 是按**批**等权，而批被重复提议主导（一份实测快照里
  level2/DEPTH_MIX 一家占 538/714，却只有 26 个互异的池）→「平均批」不是部署总体。
  故新增「交付口径对照」块，把 ① 全部批等权 / ② 按候选池去重 / ③ 按电路宏平均 /
  ④ 排除最大电路 四种口径并排。**脚本只给数，交付报哪个由口径决定，不自作主张。**
  另新增「输入污染检查」：暴露原本**静默**的 eval 去重丢弃行数，并给出
  「同一 eval_idx 却对应两个不同 true_delay」的处数 —— 后者 >0 即 append 叠行（见
  OPERATIONS:177 记的那次事故）的硬证据，届时本快照所有数字须重估。

用法（server 端）：
  ~/venv/bin/python3 scripts/diag/_shadow_analyze.py [--min-cands 4] [--group-by batch|window]
"""
import argparse, glob, hashlib, os, re, shutil, statistics, subprocess, sys, time
from collections import Counter
import numpy as np

# 17.3.9 附带：本文件有 ✅/❌/⚠ 等非 GBK 字符，Windows 下（默认 GBK）必然 UnicodeEncodeError。
# 服务器（Linux/UTF-8）是 no-op，输出不变；加它是为了让本脚本**能本地跑**（本地即可验证口径）。
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

CSV_RE = re.compile(
    r"eval_idx=(\d+), iter=(\d+), window=(\d+), gnn_pred=([0-9.eE+-]+|nan), "
    r"true_delay=([0-9.eE+-]+|NA)"
)
# 第二列单查（不并入 CSV_RE：旧 CSV 没有这一列，并入会让 CSV_RE 失配 → 全表读不出）
CSV2_RE = re.compile(r"gnn_pred2=([0-9.eE+-]+|nan)")
# 17.3.10: transistors 同样单查（同 CSV2_RE 的理由：旧 CSV 可能整列缺席）
CSV_NT_RE = re.compile(r"transistors=(\d+)")
# 17.6.0: 失败行的 error 列同样单查（旧 CSV 里 NA 行**没有** error 列 ⇒ 并入 CSV_RE 会全表失配）。
# 只用于**读数**（数一数有多少 NA 行是超时），不参与任何判据 —— 故截到第一个逗号就够，
# 不必处理引号转义（超时那条消息本身不含逗号，见 simulation.rs 里 Timeout 的构造）。
CSV_ERR_RE = re.compile(r"error=([^,]*)")

def parse_row(line):
    m = CSV_RE.search(line)
    if not m:
        return None
    try:
        g5 = m.group(5)
        g4 = m.group(4)
        m2 = CSV2_RE.search(line)
        g2 = m2.group(1) if m2 else None
        m3 = CSV_NT_RE.search(line)
        m4 = CSV_ERR_RE.search(line)
        return {
            "eval": int(m.group(1)),
            "iter": int(m.group(2)),
            "window": int(m.group(3)),
            "gnn": None if g4 == "nan" else float(g4),
            "true": None if g5 == "NA" else float(g5),
            # c2 = 该行**文本上**带第二列（区分「没开 GNN_PORT2」与「开了但预测 NaN」）
            "c2": m2 is not None,
            "gnn2": None if (g2 is None or g2 == "nan") else float(g2),
            # 17.3.10 跨集判同用：true_delay 的**原始文本**（逐字节判同，不经 float 往返）
            # 与 transistors 文本。两者都取自真值列 → 判同结果不随 ckpt 变。
            "ttxt": g5,
            "nt": m3.group(1) if m3 else "",
            # 17.6.0：失败行的 error 文本（成功行没有这一列 ⇒ 空串）。只给「超时行数」那条读数用。
            "err": m4.group(1) if m4 else "",
        }
    except (ValueError, TypeError):
        return None   # 16.11.6: 损坏行（并发写 CSV 交错）跳过，不崩溃

def per_window_metrics(rows, key="gnn"):
    """rows: 该候选集的全部候选（成功行）。返回 (recall@3, regret, spearman) 或 None。
    key: 取哪一列当预测（"gnn" = gnn_pred， "gnn2" = gnn_pred2）——同一函数两列复用。"""
    n = len(rows)
    if n < 2:
        return None
    gnn = np.array([r[key] for r in rows], dtype=np.float64)
    true = np.array([r["true"] for r in rows], dtype=np.float64)
    # kind="stable"：并列时按输入下标（= 调用方已按 eval_idx 定序的行序）打破，不能用默认
    # quicksort —— 它不保证并列的相对次序，会让结果依赖 CSV 行序（见 main 里 rows.sort 的说明）。
    order_g = np.argsort(gnn, kind="stable")
    order_t = np.argsort(true, kind="stable")
    top3_g = set(order_g[: min(3, n)].tolist())
    top3_t = set(order_t[: min(3, n)].tolist())
    top2_g = set(order_g[: min(2, n)].tolist())
    top2_t = set(order_t[: min(2, n)].tolist())
    recall3 = len(top3_g & top3_t) / len(top3_t)
    gnn_best_true = true[order_g[0]]
    true_best = true[order_t[0]]
    regret = (gnn_best_true - true_best) / true_best
    spread_pct = (true.max() - true.min()) / true.min() * 100.0
    # —— 粗筛视角（recall 是粗筛最重要指标；k=2/3 双口径）——
    # 严格（实际第1名是否出现在预测前k）：k=2、k=3
    recall2_strict = 1.0 if order_t[0] in top2_g else 0.0
    recall3_strict = 1.0 if order_t[0] in top3_g else 0.0
    # 宽松（预测前k 含 实际前k 之一，出现一个就算）：k=2、k=3
    recall2_len = 1.0 if (top2_g & top2_t) else 0.0
    recall3_len = 1.0 if (top3_g & top3_t) else 0.0
    # ⚠ 域依赖（与训练侧的已知差异，2026-09-16 核对）：训练侧 utils.ranking_metrics 的 recall@K
    #   带「非平凡组 m >= K+1 才算」约束（否则 top-K=全组、恒命中，记 NaN）；**本函数没有这条**。
    #   它靠 main 里 `len(rows) < args.min_cands`（默认 4）在聚合前把小集滤掉来保证与训练侧同域 ——
    #   也就是说「本函数的 recall 能与训练侧对读」是**管线不变量**，不是函数自身性质。
    #   若有人用 --min-cands 2/3 跑，n<=3 的集会被记成必命中、把读数抬高。
    #   （本文件新增的 capture2 不依赖此约定：它自带 n>=4 守卫，见 per_window_metrics 内注释。）
    # 两阶段最终遗憾：GNN 前3 → SPICE 精排 → 选前3内真最优（真#1 在则 = 0）
    top3_true_best = true[order_g[: min(3, n)]].min()
    regret_2stage = (top3_true_best - true_best) / true_best
    # 两阶段捕获率（**与训练侧 src/utils.py 的 cap2 同一公式，两侧统一口径**）：
    #   (真最差 − 前3内真最优) / (真最差 − 真最优)
    # 分母是「可改进空间」而不是真最优 → **随 spread 归一**，跨集可比。regret 以真最优为分母，
    # 不具此性质：spread 5% 的集里 2% 遗憾 = 几乎全丢，spread 50% 的集里 2% 则微不足道，
    # 逐集取均值会被 spread 分布直接扭曲。本量量的就是「该拿的拿到了几成」。
    # 非平凡性：n<=3 时「前3」= 全集 → 恒 100%，故与 recall@3 同采 n>=4 才算（否则记 NaN）。
    # ⚠ 标度：本侧记**分数**(0~1)以配合下面 {:.1%}/{:.2%} 格式；训练侧同式记**百分数**。
    _cap2_rng = true[order_t[-1]] - true_best
    capture2 = (((true[order_t[-1]] - top3_true_best) / _cap2_rng)
                if (n >= 4 and _cap2_rng > 0) else float("nan"))
    # Spearman（n>=3 才可靠，n=2 时退化为 ±1，不统计）
    if n >= 3:
        rg = np.empty(n); rt = np.empty(n)
        rg[order_g] = np.arange(n); rt[order_t] = np.arange(n)
        sp = statistics.correlation(rg, rt)
    else:
        sp = None
    # —— 并列/一致性诊断（17.2.7）——
    # 1) 重复网表天花板：真值完全相等的候选数 —— 重复/对称结构会让 recall@3 天然 <100%，
    #    这是「部署侧 recall 上限」的一部分，必须与模型的排序能力分开算。
    # 2) 打破规则的影响面：预测第3小 == 第4小时，top-3 的归属完全由并列打破规则决定；
    #    训练侧（utils.ranking_metrics）与 Rust 侧（本文件）已统一为 stable+组内行序。
    # 3) 量纲混合守卫：同集内同时出现「秩」（~1..n）与「原始延迟」（~1e-11）→ 排序无意义。
    #    按 tl_opt.rs 的 pre_rank→evaluate 同批结构不应发生；把它变成可断言的输出，而非口头约定。
    dup_true = n - int(np.unique(true).size)
    dup_g = n - int(np.unique(gnn).size)
    tie_g_k3 = False
    if n > 3:
        _part = np.partition(gnn, 3)
        tie_g_k3 = bool(_part[2] == _part[3])
    _gpos = gnn[gnn > 0]
    mixed = bool(_gpos.size and np.max(_gpos) / np.min(_gpos) > 1e6)
    # 4) 兜底窗口识别：整集 gnn_pred 都是原始延迟（~1e-11，serve_http 在候选数<2 时走
    #    predict_avg_delay），而非秩。集内排序**仍然有效**（单模型下与秩序单调等价），
    #    但分辨率变粗：原始延迟写 {:.6e}，相对分辨率 ~1e-7，与 float32 的 1 ulp 同量级
    #    → 近邻对会被记成真并列，top-k 归属转由打破规则决定。秩模式下不会（1/n 分辨率）。
    raw_mode = bool(_gpos.size and np.max(_gpos) < 1e-3)
    return {"recall3": recall3, "regret": regret, "spearman": sp, "n": n,
            "spread_pct": spread_pct,
            "recall2_strict": recall2_strict, "recall3_strict": recall3_strict,
            "recall2_len": recall2_len, "recall3_len": recall3_len,
            "regret_2stage": regret_2stage,
            "capture2": capture2,
            "dup_true": dup_true, "dup_g": dup_g, "tie_g_k3": tie_g_k3,
            "mixed": mixed, "raw_mode": raw_mode}

# —— 运行配置戳 ——
# 2026-09-15 起：sweep_ep250(10.51%) 与 repA(10.86%) 差 0.35pp，事后翻结果文件无法回答
# 「这两趟到底哪一项设置不同」——ckpt、SIM_OPTIONS、TL_MAX_ITERS、XYCE_CACHE、Xyce 版本、
# 脚本版本，一个都没记。排查只能靠重跑 + 猜。这里把每一项都写进结果文件。
# 原则：只记「事后能拿来回放的量」，不记推断。
# 18.7.0 补 XYCE_CACHE_DIR：缓存**落点**是运行身份的一部分 ——「不读旧缓存、这趟另建新缓存」
# 的那趟与既有趟正是靠它区分（XYCE_CACHE=0 是读写全禁，做不到「不读旧、写新的」）。
# 不记的话，事后看到两份同 ckpt 的结果有差异无法归因 —— 那就是下一个 I20。
# 17.5.0 再补 TL_BESTFIRST / TL_MAX_STEPS：前者换的是**搜索算法本身**（全组仿真 + 最好者胜 +
# 栈式回溯），后者换的是步数预算。二者都直接改轨迹 ⇒「同 ckpt、同 seed、结果不同」的最可能
# 原因就是它们；不进戳等于下一个 I20。
# 17.6.0 再补 TL_MAX_WALL_S / XYCE_TIMEOUT_S：这两个直接决定「**哪些候选没被仿真**」——
# 前者到点会让这一趟提前收工（DONE 行 stopped=wall、total_iters < 预算），后者会让单次仿真作废
# （CSV 里落 true_delay=NA）。它们不同的两趟，同一 ckpt / 同一 seed 也会有不同轨迹，
# 而差异**长得像**「模型换了」⇒ 不进戳就是下一个 I20。
ENV_KEYS = ("SIM_OPTIONS", "SIM_TRAN", "TL_MAX_ITERS", "TL_BESTFIRST", "TL_MAX_STEPS",
            "TL_MAX_WALL_S", "XYCE_TIMEOUT_S",
            "XYCE_CACHE", "XYCE_CACHE_DIR", "GNN_HOST",
            "GNN_PORT", "GNN_PORT2", "USE_IDS_AVG_APPROX", "IDSGNN_CKPT",
            # 17.2.4 补：进程间数值复现性直接受这几个变量影响（PYTHONHASHSEED 是 serve
            # 抖动根因、已由边序定序修掉，钉它是兜底；线程变量实测零影响，但既然会影响
            # 浮点归约序，就该进戳 —— 不记的话事后无法回答「这两趟到底哪项设置不同」）
            "PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "MKL_DYNAMIC")

def _sha16(path):
    try:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except (OSError, TypeError):
        return "NA"

def _serve_cmdline():
    """正在跑的 serve 进程完整命令行（第一行）；没有 serve 则 None。"""
    try:
        p = subprocess.run(["pgrep", "-af", "serve_htt[p].py"],
                           capture_output=True, text=True, timeout=5)
    except Exception:
        return None
    lines = [l for l in p.stdout.splitlines() if l.strip()]
    return lines[0] if lines else None

def _flag(cmd, name):
    """从命令行取 --name VALUE；取不到返回 None。"""
    if not cmd:
        return None
    parts = cmd.split()
    for i, t in enumerate(parts[:-1]):
        if t == name:
            return parts[i + 1]
    return None

def _pad(s, w):
    """按**显示宽度**右侧补空格（CJK 记 2 列）—— 口径标签是中英混排，
    直接用 f"{s:<w}" 会因双宽字符而错位。

    ⚠ 补到 max(w, d+1)：**超宽时也至少留 1 个空格**。否则标签一旦长过列宽
    （如 ④ 行的电路名带 gnn_shadow.csv 后缀，49 列 > 32）就与下一列**首字符粘连**，
    读出来是 "…DEPTH_MIX/gnn_shadow.csv1"、且按空白切列会切错。"""
    d = sum(2 if ord(c) > 0x2E80 else 1 for c in s)
    return s + " " * max(1, w - d)

def run_stamp(root):
    print("=== 运行配置戳 ===")
    print(f"  时间        : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  分析根目录  : {root}")
    cmd = _serve_cmdline()
    if cmd:
        ck, sc, pt = _flag(cmd, "--ckpt"), _flag(cmd, "--scaler"), _flag(cmd, "--port")
        print(f"  serve ckpt  : {ck}  sha1={_sha16(ck)}")
        print(f"  serve scaler: {sc}  sha1={_sha16(sc)}")
        print(f"  serve 端口  : {pt}")
    else:
        print("  serve       : (没有 serve 在跑 —— 本次服务的 ckpt 无法记录)")
    print("  环境        : " + "  ".join(
        f"{k}={os.environ.get(k, '(未设)')}" for k in ENV_KEYS))
    xy = shutil.which("Xyce") or shutil.which("xyce")
    print(f"  Xyce        : {os.path.realpath(xy) if xy else '(PATH 里找不到)'}")
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print("  脚本 sha1   : " + "  ".join(
        f"{n}={_sha16(os.path.join(repo, 'scripts', 'diag', n))}"
        for n in ("_shadow_analyze.py", "serve_http.py", "run_shadow_batch.sh")))
    try:
        rev = subprocess.run(["git", "-C", repo, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        dirt = subprocess.run(["git", "-C", repo, "status", "--porcelain"],
                              capture_output=True, text=True, timeout=5).stdout.strip()
        print(f"  repo        : {rev or 'NA'}  ({'有未提交改动' if dirt else '干净'})")
    except Exception:
        print("  repo        : NA")
    csvs = sorted(glob.glob(os.path.join(root, "*", "*", "gnn_shadow.csv")))
    if csvs:
        mt = [os.path.getmtime(c) for c in csvs]
        fmt = lambda t: time.strftime("%m-%d %H:%M:%S", time.localtime(t))
        print(f"  CSV 快照    : {len(csvs)} 个, mtime {fmt(min(mt))} ~ {fmt(max(mt))}")
    else:
        print("  CSV 快照    : 0 个（root 下没有 gnn_shadow.csv）")
    print("=== 戳结束 ===")

# ————————————————————————————————————————————————————————————————————————
# 以 current 为锚的 GNN 判据质量（18.6.0 新增 / 18.6.1 修正配对校验）
# ————————————————————————————————————————————————————————————————————————
# 动机：本文件其余所有指标都把 GNN 放在**候选集内部**评分（「这批里谁最快」），参照系里
#   **没有 current 这个角色**。而部署时贪心的实际判据是「按生成序试 current+窗口衍生出来的
#   候选，**第一个 delta<0 的收下**」（tl_opt.rs:1157-1191）⇒ 真正该问的是：**GNN 会不会把
#   一个其实比 current 差的候选判成「比 current 好」**（= 它掌判据时的「误收」）。
# 数据：gnn_shadow.rs 的 rank_window 覆写把 `[current] + 本窗口全部候选` 放进**同一批**
#   /rank，秩落进同目录的 gnn_current.csv（10 列，见该文件模块头）。
#
# ⚠ 口径①（两把尺，禁混读）：本节的 rank 是「**含 current 批**」的秩；上面主口径的 gnn_pred
#   是「**纯候选批**」的秩（pre_rank 另发的一次请求）。批内名次是批内归一化的 ⇒ 同一个候选
#   在两处的秩**不同**。**本节任何数字都不得与上面的 recall@k / 选择遗憾 / Spearman 对读。**
# ⚠ 口径②（相关系数的约定不同）：本节的 Spearman 用 serve 返回的**平均秩**对真值侧的
#   **平均秩**（tie-corrected）算 Pearson；主口径 per_window_metrics 用 argsort 的**序数**秩、
#   无并列修正（:145-147）——**不是同一个约定**，两节的 Spearman 值不可对读。
# ⚠ 口径③（「劣于」的判据尺，**随变体而异**）：本节一律用 avg_delay（**全部输出口**均值）
#   判「劣于 current」。
#   · **bestfirst 变体**（17.5.0 起：全组仿真 + 最好者胜 + 栈式回溯）的搜索判据**就是**
#     avg_delay（`combined_score(metrics.avg_delay, …, delay0, …)`，见 `tl_opt.rs` 的
#     `optimize_tl_module_bestfirst` 口径 ①）⇒ 与本节**同一把尺**、无代理误差：R1/R2 直接
#     读作「GNN 与搜索判据**同判**的组比例」（= GNN 掌判据时会漏掉多少改进）。
#   · **老贪心**（shadow / gnn 变体）吃的是 affected_delay（`window.affected_outputs` 子集均值，
#     tl_opt.rs:908-909/:1127）⇒ 只在 `affected_outputs == 全输出` 时恒等。「35/46 电路单输出」
#     是**按电路数**，而多输出电路占**候选集的 52~57%**（GNN_RUST_DATA_DIFF §12.2 的 16.11.36
#     修正）⇒ 在按候选集加权的分母上「两者等价」**不成立**。故**那些树上**本节的误报率是
#     部署口径的必要不充分代理（被判误报的候选仍可能在 affected_outputs 上真优于 current =
#     贪心会正当接受它），且**单/多输出分层是解释它的必需项**。
#   ⚠ 变体无法从 CSV 数据判定（两变体的 CSV 同格式）⇒ 读本节前先确认这一趟跑的是哪个变体。

def _finite(v):
    """非 None 且非 NaN/Inf。"""
    return v is not None and not (isinstance(v, float) and not np.isfinite(v))

def _avg_ranks(vals):
    """平均秩（并列取平均，1-based）。tie-corrected 相关用。"""
    n = len(vals)
    order = sorted(range(n), key=lambda i: vals[i])
    r = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r

def _kendall_tau_b(x, y):
    """Kendall τ-b（并列修正）。n 很小（≤ 十几）⇒ O(n²) 数对，不引 scipy。
    与 Spearman 并列时的差别：τ-b 把并列对从分母里扣掉，不靠平均秩的连续性假设。"""
    n = len(x)
    if n < 2:
        return None
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]; dy = y[i] - y[j]
            if dx == 0 and dy == 0:
                tx += 1; ty += 1
            elif dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif (dx > 0) == (dy > 0):
                conc += 1
            else:
                disc += 1
    n0 = n * (n - 1) / 2.0
    den = ((n0 - tx) * (n0 - ty)) ** 0.5
    return (conc - disc) / den if den > 0 else None

def _spearman_tie(pred_ranks, true_vals):
    """含 current 批内的 Spearman（真值侧 tie-corrected）。
    pred 侧**直接用 serve 返回的秩**：那本身就是预测值的平均秩（competition ranking 并列取
    平均）⇒ 等价于对 pred 取平均秩，不需要再变换。任一侧恒定（方差 0）⇒ 返回 None（
    statistics.correlation 会抛 StatisticsError，这里挡住）。n<3 不算（与既有守卫一致）。"""
    if len(pred_ranks) < 3:
        return None
    rt = _avg_ranks(true_vals)
    if len(set(pred_ranks)) < 2 or len(set(rt)) < 2:
        return None
    try:
        return statistics.correlation([float(v) for v in pred_ranks], rt)
    except statistics.StatisticsError:
        return None

CUR_HEADER_PREFIX = "iter,window,eval_idx,pos,id,rank,is_current,true_avg,tc,nout"

def load_current(csvp):
    """读 gnn_current.csv → (rows, diag)。rows 每项 = 一个窗口的一行。
    diag: bad=列数/字段坏行数（原本会静默丢），header=表头出现次数（>1 = 同目录 append 叠趟）。"""
    rows = []
    bad = n_head = 0
    with open(csvp, encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("iter,window,"):
                n_head += 1
                continue
            p = s.split(",")
            if len(p) != 10:
                bad += 1
                continue
            try:
                rows.append({
                    "iter": int(p[0]), "window": int(p[1]),
                    "eval": None if p[2] == "NA" else int(p[2]),
                    "pos": None if p[3] == "NA" else int(p[3]),
                    "id": p[4],
                    "rank": None if p[5] == "NA" else float(p[5]),
                    "is_cur": p[6] == "1",
                    "true": None if p[7] == "NA" else float(p[7]),
                    "tc": None if p[8] == "NA" else int(p[8]),
                    "nout": None if p[9] == "NA" else int(p[9]),
                })
            except (ValueError, TypeError):
                bad += 1
                continue
    return rows, {"bad": bad, "header": n_head}

def cur_metrics(cur_rank, cur_true, cands):
    """cands = [(rank_i, t_i)]，只含 rank 与真值都齐的候选。定义不出的量记 nan / None。
    严格 `<` 与贪心的 `delta < 0.0`、`any_better` 同调 ⇒ **并列自动不算改进**：
    serve 给并列同一个秩 ⇒ 并列候选既不计入 S（分母），也就不会被判误报。"""
    S = FP = P = FN = 0
    for r, t in cands:
        if r < cur_rank:
            S += 1
            if t >= cur_true:
                FP += 1
        if t < cur_true:
            P += 1
            if r >= cur_rank:
                FN += 1
    pred = [cur_rank] + [r for r, _ in cands]
    true = [cur_true] + [t for _, t in cands]
    pmin = [i for i, v in enumerate(pred) if v == min(pred)]
    tmin = [i for i, v in enumerate(true) if v == min(true)]
    return {
        "n_cands": len(cands), "S": S, "FP": FP, "P": P, "FN": FN,
        # 参考口径（逐候选）误报率：GNN 说「比 current 好」的那些里，真值其实**不更好**（>= 含相等）的占比
        "fp_rate": (FP / S) if S > 0 else float("nan"),
        # 伴随 = 漏报率：真有改进机会的里面，GNN 判成「不如 current」的占比
        "fn_rate": (FN / P) if P > 0 else float("nan"),
        # top-1：严格 = 真最优落在预测最优集合内；宽松 = 两侧最优集合相交（k=1 版，与既有
        # per_window_metrics 的严格/宽松约定同构）
        "top1_strict": 1.0 if tmin[0] in pmin else 0.0,
        "top1_len": 1.0 if set(pmin) & set(tmin) else 0.0,
        "sp": _spearman_tie(pred, true),
        "kt": _kendall_tau_b(pred, true),
    }

def _agg_keys(ss, keys, macro=False):
    """与 _agg 同构但**键可传**：`_agg` 的键写死了 6 个既有指标，改它会动既有对照表的输出。
    macro=True：先按电路求均值再跨电路求均值。含 nan/None 的集在该键上被跳过（空口径回 nan）。"""
    if macro:
        g = {}
        for s in ss:
            g.setdefault(s["circuit"], []).append(s)
        units = list(g.values())
    else:
        units = [ss]
    units = [u for u in units if u]
    if not units:
        return tuple(float("nan") for _ in keys)
    out = []
    for k in keys:
        per_unit = []
        for u in units:
            vals = [x[k] for x in u if _finite(x.get(k))]
            if vals:
                per_unit.append(statistics.mean(vals))
        out.append(statistics.mean(per_unit) if per_unit else float("nan"))
    return tuple(out)

def current_anchor_report(root, min_cands, main_sets, dom_circ):
    """以 current 为锚的判据质量报告（18.6.0 新增；18.6.1 修正配对校验 + 加核心指标）。
    18.6.1 核心（用户口径）：**只看 GNN 排第一的那个候选是否真负**，不再逐项看所有排在
    current 之前的候选。窗级：严格 = 首推组全部 t1 < t_cur（与 `delta<0` 同调）、宽松 =
    全部 t1 <= t_cur（不劣）；并列取同秩整组，组内真假不一时按**保守**判错并单列 t1_mix；
    首推候选无真值（落在未评估后缀）⇒ 单列 n_t1_nodata 并剔出分母；分母缺口恒等式
    n_zero_S + n_t1_nodata + n_t1_win == len(recs)。同时报**镜像漏推率**（GNN 首推 current
    的窗里其实存在更优候选的比例），防「永远说 current 最好」把首推率衬托得好看。
    旧的逐候选 误报率/漏报率/top-1/秩相关一律降级为**参考口径**。
    main_sets = 主口径候选集列表（仅用于并排报覆盖率）；dom_circ = 主口径 ④ 排除的电路。
    18.6.1 修正（首跑实测倒逼）：18.6.0 要求「每个候选行的推断 eval_idx 都必须在 shadow 里存在」，
    实测 134 个窗口不满足 ⇒ 整窗被剔并报「配对不可信 → 本节数字作废」。但那些「缺」是**预期结构**：
    Pass 3 遇到第一个 accept 就 break（tl_opt.rs:1191）⇒ 该窗尾部候选永不评估、永无 eval_idx 行，
    而本文件的候选行来自 `mods`（= `prepared` 全表，:1024-1026）⇒ present 只能是**前缀**。
    故校验改为「present 必须是前缀 {0..k-1}」+「shadow 行集恰为 {base+1..base+k}」+ 逐行 tc 相符，
    缺的后缀单列计数（它还会让 S 低估，故另报其中「秩高于 current」的项数）。
    18.7.0 加**机会集口径**（见下面「★ 机会集口径 R1/R2」块）：D3/D4 = 组内真值上存在更好/不劣
    候选的组数，N1/N2 = 其中 GNN 仍把 current 排第 1 的组数，R1 = N1/D3、R2 = N2/D4。
    它需要**完整真值**（全组仿真的树）故自带更严的前置校验（pos 集必须是 {0..n-1} 全集）——
    既有那条前缀校验与它下面的所有读数一字未动。
    18.7.4 加**最优先判据「机会把握率」**（打印在本节最前面）：严格 = H3/D3、宽松 = H4/D4，
    分子 = GNN 排第一的那个**真的可用**（严格 t₁ < t_cur；宽松 = 首推就是 current 本身，或 t₁ ≤ t_cur）。
    ⚠ 与 R1/R2 互补而非互读：R1 记「仍首推 current」（越大越坏），把握率记「抓住了机会」（越大越好），
    且 1 − R1 把「守住 current」与「推错对象」并成一格 ⇒ **不能当把握率用**。"""
    print("\n=== 以 current 为锚的 GNN 判据质量（18.6.1；核心 = 首推真负率）===")
    cur_files = sorted(glob.glob(os.path.join(root, "*", "*", "gnn_current.csv")))
    if not cur_files:
        # ⚠ 必须是**常量**：_t_shadow_tiebreak.py 断言剥戳后两侧逐字节相同，而它的 fixture 树
        # 都没有本文件 —— 带上路径/计数/时间就不再是常量，那条断言会红。
        print("  （本树无 gnn_current.csv，跳过：本树不是 18.6.0 之后的 shadow 跑的）")
        return
    recs = []
    n_win_total = n_no_shadow = n_mismatch = n_ev_missing = n_tc_mismatch = 0
    n_empty_cand = n_dim = n_dup_eval = n_dup_win = 0
    n_uneval = n_uneval_above = 0     # 结构性未评估（贪心提前 accept 的尾部候选，见下面配对校验）
    n_t1_nodata = 0                   # 核心口径：首推候选无真值（判不了）的窗数
    # 18.7.0 机会集口径（D3/D4 → R1/R2）：见函数 docstring 与下面「★ 机会集口径」打印块。
    # 这四个是**本块专属**的前置校验失败数，与上面那条既有前缀校验互不影响（既有读数一字未动）。
    n_opp_pair = n_opp_cur_na = n_opp_rank_na = n_opp_true_na = 0
    arecs = []                        # 18.7.0：真值全到位的组（= R1/R2 的候选分母池）
    n_cur_rank_na = n_cur_true_na = n_true_na = n_rank_na = 0
    n_timeout_na = 0                  # 17.6.0：NA 行里由超时造成的那部分（见下面 diag 行）
    n_header_dup = n_bad_row = n_win_unclaimed = 0
    for curp in cur_files:
        shp = os.path.join(os.path.dirname(curp), "gnn_shadow.csv")
        # ⚠ circuit 键必须与主口径**同型**：主口径是 `os.path.relpath(gnn_shadow.csv, args.root)`
        # —— **含文件名**（:701）。这里若写成 dirname(...) 就会少一级，与主口径的 dom_circ 比不上
        # （④ 会永远排除不掉任何电路，而 ③ 的分组数看起来还是对的 ⇒ 静默错）。
        circ = os.path.relpath(shp, root)
        rows_c, diag = load_current(curp)
        n_header_dup += max(0, diag["header"] - 1)
        n_bad_row += diag["bad"]
        if not os.path.isfile(shp):
            n_no_shadow += 1
            continue
        by_win, ev_seen = {}, {}
        with open(shp, encoding="utf-8", errors="replace") as f:
            for line in f:
                r = parse_row(line)
                if not r:
                    continue
                ev_seen[r["eval"]] = ev_seen.get(r["eval"], 0) + 1
                by_win.setdefault((r["iter"], r["window"]), []).append(r)
        n_dup_eval += sum(ev_seen.values()) - len(ev_seen)
        wins = {}
        for r in rows_c:
            wins.setdefault((r["iter"], r["window"]), []).append(r)
        n_win_total += len(wins)
        if len(wins) != len(rows_c) - sum(1 for r in rows_c if r["is_cur"]):
            pass  # 每窗行数由下面的逐窗检查负责，这里不做全局推断
        n_win_unclaimed += sum(1 for k in by_win if k not in wins)
        for (it, w), wr in sorted(wins.items()):
            cur_rows = [r for r in wr if r["is_cur"]]
            cand_rows = [r for r in wr if not r["is_cur"]]
            # 只有 CURRENT 行的窗口**本不该被写出来**（rank_window 里 candidates.is_empty()
            # 已挡；若出现就是那条守卫漏了，此时批次只剩 current、serve 回的是原始延迟）
            if len(cur_rows) != 1 or not cand_rows:
                n_empty_cand += 1
                continue
            cur = cur_rows[0]
            # 量纲守卫（本节自带，不能靠既有那条：既有守的是 gnn_shadow 的值，而本文件是
            # **另一次请求**，serve_http.py:13-17 的「同批不会混」不变量在此不适用）：
            #   ① 秩必在 [1, n]（n = 本窗行数）；② 秩必为整数或半整数（competition ranking）。
            #   这两条正是「把秒当秩读」的检测器（原始延迟 ~1e-11 两条都过不了）。
            nwin = len(wr)
            rk = [r["rank"] for r in wr if r["rank"] is not None]
            if rk and (min(rk) < 1.0 or max(rk) > nwin
                       or any(abs(2.0 * x - round(2.0 * x)) > 1e-9 for x in rk)):
                n_dim += 1
                continue
            sh = by_win.get((it, w), [])
            sh_by_ev, dup = {}, False
            for r in sh:
                if r["eval"] in sh_by_ev:
                    dup = True
                sh_by_ev[r["eval"]] = r
            if dup:
                n_dup_win += 1
                continue
            # —— 配对校验（18.6.1 修正）：present 必须是**前缀**，缺的尾部 = 结构性未评估 ——
            # 代码依据（tl_opt.rs:1156-1191）：Pass 3 按 `prepared` 顺序逐个 evaluate，撞上第一个
            # `accept`（delta<0）就 `break` ⇒ 排在其后的候选**永不评估、永无 eval_idx 行**。而本文件
            # 的候选行出自 `mods`，`mods` 由 `prepared` 现取（:1024-1026）⇒ 本文件 = prepared 全表、
            # shadow 的评估行 = 它的**前缀**。故「缺」是预期结构（贪心接受后就没再试），不是错位。
            # 校验强度不减：① pos 集必须恰为 {0..k-1}（有洞/零个 ⇒ 推断失效，真错位）；② 该窗 shadow
            # 行集必须恰为 {base+1..base+k}（len 相等，无多余行）；③ 逐行 tc 相符；④ 窗内 eval 不重复（上）。
            # —— 18.7.0 机会集口径（D3/D4 → R1/R2）的取数：**独立于**下面那条前缀校验，故放在它之前 ——
            # 为什么要另立一套校验：D3/D4 问的是「该组真值上存在机会吗」，必须有**完整真值**才成立；
            # 而下面那条校验为兼容贪心树只要求 pos 是**前缀**（尾部结构性未评估是预期的）。本块反过来
            # 要求 pos 集恰为 {0..n-1} + 每条 eval 都在 gnn_shadow 里 + 逐行 tc 相符 + 真值不缺 ——
            # 任一条不过就计一类失败、不进 arecs（fail-loud：这些计数必须全 0，否则说明这棵树不是
            # 「全组仿真」（TL_BESTFIRST=1）的，R1/R2 的分母只是下界）。既有那条前缀校验与它下面的
            # 所有读数**一字未动**（三个既有分析器自测靠逐字节比对，动不得）。
            pos_full = (len(cand_rows) > 0
                        and all(c["pos"] is not None and c["eval"] is not None for c in cand_rows)
                        and sorted(c["pos"] for c in cand_rows) == list(range(len(cand_rows)))
                        and len(sh_by_ev) == len(cand_rows)
                        and all(c["eval"] in sh_by_ev for c in cand_rows))
            if not pos_full:
                n_opp_pair += 1
            elif cur["rank"] is None or cur["true"] is None:
                n_opp_cur_na += 1
            elif any(c["rank"] is None for c in cand_rows):
                n_opp_rank_na += 1
            else:
                rs = [c["rank"] for c in cand_rows]
                ts = [sh_by_ev[c["eval"]]["true"] for c in cand_rows]
                tc_ok = all(c["tc"] is None or not sh_by_ev[c["eval"]]["nt"]
                            or c["tc"] == int(sh_by_ev[c["eval"]]["nt"]) for c in cand_rows)
                if not tc_ok or any(t is None for t in ts):
                    n_opp_true_na += 1          # tc 不等 = 位置↔真值错配，与真值缺同类
                else:
                    cur_first = min(rs) >= cur["rank"]     # 没有候选严格排在 current 之前
                    d3 = any(t < cur["true"] for t in ts)
                    d4 = any(t <= cur["true"] for t in ts)
                    # 18.7.4 「机会把握率」两格（本文件**最优先判据**，用户口径）：
                    #   严格 = GNN 排第一的那个候选真值**严格优于** current（t₁ < t_cur）
                    #   宽松 = GNN 排第一的是 **current 本身**（= 没动、不劣），**或**其真值
                    #          **不劣于** current（t₁ ≤ t_cur，含真并列）
                    # 「排第一」与既有口径**同一判据**：min(候选秩) < rank(current) 才算「首推候选」，
                    # 否则首推就是 current（competition ranking，= cur_first 的另一面）；并列首推要求
                    # **整撮都满足**（保守 —— 与 18.6.1 首推真负率的并列规则同一套 ⇒ 三个口径的并列
                    # 语义一致，不许各写一份）。
                    rmin = min(rs)
                    cand_first = rmin < cur["rank"]
                    picks = [t for r, t in zip(rs, ts) if r == rmin] if cand_first else []
                    h3 = 1.0 if (cand_first and all(t < cur["true"] for t in picks)) else 0.0
                    h4 = 1.0 if ((not cand_first) or all(t <= cur["true"] for t in picks)) else 0.0
                    cm = cur_metrics(cur["rank"], cur["true"], list(zip(rs, ts)))
                    arecs.append({
                        "circuit": circ, "iter": it, "window": w,
                        "nout": cur["nout"] if cur["nout"] is not None else nwin - 1,
                        "n_cands": len(cand_rows),
                        "d3": 1.0 if d3 else 0.0, "d4": 1.0 if d4 else 0.0,
                        "n1": 1.0 if (d3 and cur_first) else 0.0,
                        "n2": 1.0 if (d4 and cur_first) else 0.0,
                        # 18.7.4 机会把握率：分子只在**该口径的机会组**里数（见 _opp 那条注释）
                        "h3": h3, "h4": h4,
                        "cur_first": 1.0 if cur_first else 0.0,
                        # 「同秩」= 存在候选的秩**恰等于** current 的秩（competition ranking 下 = 并列第 1）。
                        # ⚠ 不能用 `min(rs) <= cur["rank"]`：那还会把「候选秩**严格优于** current」算进来
                        # （例：cur 秩 5、某候选秩 1 ⇒ 也成立），而那种组**不是**并列、是 cur 根本没被
                        # 排第一（已由 cur_first 表达）⇒ 语义与打印的「并列第 1」不符。自测
                        # _t_shadow_current_anchor.py 的 (a) 伴随计数专门钉了这条。
                        "tie_rank": 1.0 if any(r == cur["rank"] for r in rs) else 0.0,
                        "tie_true": 1.0 if any(t == cur["true"] for t in ts) else 0.0,
                        "top1_strict": cm["top1_strict"], "top1_len": cm["top1_len"],
                    })
            if any(c["pos"] is None for c in cand_rows):
                n_ev_missing += 1; continue
            pres = [c for c in cand_rows if c["eval"] is not None and c["eval"] in sh_by_ev]
            k = len(pres)
            if k == 0 or sorted(c["pos"] for c in pres) != list(range(k)):
                n_ev_missing += 1; continue
            if len(sh_by_ev) != k:
                n_mismatch += 1; continue
            # 计数放在校验通过之后（只统计**被本节采用**的窗）：未评估的那些里「GNN 秩高于
            # current」的个数尤其要紧 —— 它们无真值 ⇒ 分子分母都进不去，**S 因此被低估这么多**，
            # 必须可见（否则读者会以为 S 覆盖了该窗全部候选）
            pres_ids = {id(c) for c in pres}
            uneval = [c for c in cand_rows if id(c) not in pres_ids]
            n_uneval += len(uneval)
            if cur["rank"] is not None:
                n_uneval_above += sum(1 for c in uneval
                                      if c["rank"] is not None and c["rank"] < cur["rank"])
            pairs, broken = [], False
            for c in pres:
                sr = sh_by_ev[c["eval"]]
                # tc 校验（与 shadow 的 transistors 同源同值）：抓「位置↔真值」错配
                tc_s = int(sr["nt"]) if sr["nt"] else None
                if c["tc"] is not None and tc_s is not None and c["tc"] != tc_s:
                    n_tc_mismatch += 1; broken = True; break
                if sr["true"] is None:
                    n_true_na += 1
                    # 17.6.0：其中有多少是**超时**（`XYCE_TIMEOUT_S` 把这一刀掐了）——
                    # 这是「两道保险到底有没有触发」的唯一直接读数。两个 marker 都认：搜索层的
                    # `TlOptError::Timeout`（"simulation timeout: …"）与仿真层的
                    # `SimulationError::Timeout`（"Simulator timed out: …"）—— 后者出现在
                    # 「超时被折进 Evaluation」的错误接线下，认它就等于给那条退化路径留了灯。
                    if "timeout" in (sr["err"] or "").lower():
                        n_timeout_na += 1
                    continue
                if c["rank"] is None:
                    n_rank_na += 1
                    continue
                pairs.append((c["rank"], sr["true"]))
            if broken:
                continue
            if cur["rank"] is None:
                n_cur_rank_na += 1
                continue
            if cur["true"] is None:
                n_cur_true_na += 1
                continue
            if len(pairs) < min_cands:
                continue
            m = cur_metrics(cur["rank"], cur["true"], pairs)
            # —— 核心口径（18.6.1）：首推 = 秩最小的候选，只看它**是否真负**（真改进）——
            # 为什么另立一条：上面那条把「所有排在 current 之前的候选」逐项汇总，一个窗里 GNN 把几个
            # 候选都排在 current 之前时，摊到每项上的说法与部署无关；部署上真正发生的是「贪心照 GNN 的
            # 名次**先试第一名**」⇒ 只问那一个：一试就中，还是空跑。
            # 严格 = 真改进（t₁ < t_cur，与 `delta < 0.0` 同调）；宽松 = 不劣（t₁ ≤ t_cur，含相等）。
            # 并列：serve 给并列同一（平均）秩 ⇒ 首推可能是一组，**要求全组都真负**才算严格正确（保守），
            # 组内真假不一致的单列计数。首推候选若落在未评估后缀（或那条 eval 失败）⇒ 无真值、判不了，
            # 剔出分母并单列 —— 这条不能省：部署上贪心先试的正是首推那个。
            rk_c = [c["rank"] for c in cand_rows if c["rank"] is not None]
            m["t1_win"] = 0.0                     # 该窗是否进核心分母（首推了候选且有真值）
            m["t1_neg"] = m["t1_nn"] = float("nan")
            m["t1_tie"] = m["t1_mix"] = 0.0
            # ⚠ 镜像判据**不能**写成 S == 0：S 只数「rank 与真值都齐」的候选，而首推只看秩 ——
            # 若首推那个候选的真值缺失（未评估后缀），S 仍可为 0，于是同一个窗会被同时算进
            # 「首推 current」与「首推无真值」两个桶 ⇒ 下面那条缺口恒等式破。故这里改用**秩**判：
            # 三个桶（首推 current / 首推无真值 / 进分母）互斥且穷尽（cur["rank"] 为 None 的窗、
            # 空候选窗、len(pairs)<min_cands 的窗都已在此之前 continue 掉）。
            _t1_top = bool(rk_c) and min(rk_c) < cur["rank"]
            m["s0"] = 0.0 if _t1_top else 1.0      # 镜像：GNN 首推 current（无名次更优的候选）
            m["s0_miss"] = 1.0 if (not _t1_top and m["P"] > 0) else 0.0   # 镜像：其实有更优候选
            if _t1_top:
                _mr = min(rk_c)
                picks = [c for c in cand_rows if c["rank"] is not None and c["rank"] == _mr]
                tt = [sh_by_ev[c["eval"]]["true"] if c["eval"] in sh_by_ev else None for c in picks]
                if any(x is None for x in tt):
                    n_t1_nodata += 1              # 首推无真值（未评估后缀 / eval 失败）
                else:
                    m["t1_win"] = 1.0
                    m["t1_neg"] = 1.0 if all(x < cur["true"] for x in tt) else 0.0
                    m["t1_nn"] = 1.0 if all(x <= cur["true"] for x in tt) else 0.0
                    m["t1_tie"] = 1.0 if len(picks) > 1 else 0.0
                    if len(picks) > 1 and len({x < cur["true"] for x in tt}) > 1:
                        m["t1_mix"] = 1.0
            m["circuit"] = circ; m["iter"] = it; m["window"] = w
            m["nout"] = cur["nout"] if cur["nout"] is not None else nwin - 1
            # sig 与主口径**同一配方**（:315-333）：先剔 true/gnn 缺的行，再按 eval 去重，
            # 再取 (true_delay 文本, transistors 文本) 的多重集 ⇒ ② 口径才可比。
            sig_rows = {}
            for r in sh:
                if r["true"] is None or r["gnn"] is None:
                    continue
                sig_rows.setdefault(r["eval"], r)
            m["sig"] = tuple(sorted((r["ttxt"], r["nt"]) for r in sig_rows.values()))
            recs.append(m)

    print(f"  数据源: {len(cur_files)} 个 gnn_current.csv（每窗 1 行 CURRENT + n 行候选）"
          f"；本节合格集 {len(recs)} / 主口径 {len(main_sets)} 集（同一个 --min-cands {min_cands}）")
    # ================= 18.7.0 机会集口径：D3/D4 分母 + N1/N2 分子 → R1/R2 =================
    # 用户口径：一个窗口的衍生电路 + 该窗 current 构成**一组**；只有「组内真值上存在比 current
    # 更好/不劣的候选」的组才是有意义的组（分母），看 GNN 有没有在这些组里**仍然把 current 排第 1**。
    #   D3 = #{组: ∃i t_i <  t_cur}        D4 = #{组: ∃i t_i <= t_cur}（D4−D3 = 只有真并列的组）
    #   N1 = #{组: current 排第 1 ∧ ∃i t_i <  t_cur}      N2 同理用 <=
    #   R1 = N1/D3（严格机会口径）          R2 = N2/D4（宽松机会口径：更好或相等都算机会）
    # 18.7.4 在同一批 arecs 上再加**最优先判据「机会把握率」**（打印在本节最前面，见下面 ★★★ 块）：
    #   严格 H3/D3、宽松 H4/D4 —— 分子 = GNN 第一名真的可用（严格 = t₁ < t_cur；宽松 = 首推就是
    #   current 本身、或 t₁ ≤ t_cur）。它与 R1/R2 **互补**：R1 问「有机会却仍首推 current」，
    #   把握率问「机会有没有被抓住」。⚠ 1 − R1 ≠ 把握率（R1 把「守住 current」与「推错对象」并成一格）。
    # ⚠ 越大越坏（有机可乘的组里仍首推 current = 把机会漏了）；分母为 0 的组单列、不入比值。
    # ⚠ 前提：只有**全组仿真**的树（TL_BESTFIRST=1，Pass 3′ 不再撞上首个 accept 就 break）才有
    # 完整真值；贪心树里 current 之后的候选尾部永不评估 ⇒ 分母只是下界。本块的合格性由下面
    # 「机会集诊断」四项**全为 0** 担保（那是它自带的前置校验，见循环里那段独立取数）。
    # ⚠ 「排第 1」按 competition ranking 判：没有候选秩**严格小于** current 即算并列第 1（同秩的
    # 组单列计数）——与上面核心口径的 s0（镜像）同一判据，两者必须一致。
    def _opp(ss):
        d3 = int(sum(x["d3"] for x in ss)); n1 = int(sum(x["n1"] for x in ss))
        d4 = int(sum(x["d4"] for x in ss)); n2 = int(sum(x["n2"] for x in ss))
        # 18.7.4 机会把握率：分子**只在该口径的机会组里数** ——
        # ⚠ h4 的「首推 current ⇒ 不劣」在 D4 之外的组也成立（整组候选都更差、GNN 守住 current），
        #   所以 h4 必须限定在 `x["d4"]` 里，否则分子外溢到分母之外 ⇒ 比值可以 >100%。
        h3 = int(sum(x["h3"] for x in ss if x["d3"]))
        h4 = int(sum(x["h4"] for x in ss if x["d4"]))
        return (d3, d4, n1, n2, (n1 / d3 if d3 else float("nan")), (n2 / d4 if d4 else float("nan")),
                h3, h4, (h3 / d3 if d3 else float("nan")), (h4 / d4 if d4 else float("nan")))

    if not arecs:
        print("  ★ 机会集口径 R1/R2: 无「真值全到位」的组（见下面诊断计数），不报数")
    else:
        # ★★★ 18.7.4 最优先判据：机会把握率（用户口径）★★★ —— 打在本节**最前面**。
        # 与下面 R1/R2 表**共用同一批 arecs 与同一批谓词**（不另起一遍取数 ⇒ 不可能口径漂移）。
        _o = _opp(arecs)
        _d3, _d4, _n1, _h3, _h4 = _o[0], _o[1], _o[2], _o[6], _o[7]
        _wrong3 = _d3 - _n1 - _h3      # 机会组里「首推了候选、但它比 current 差」
        _wrong4 = _d4 - _h4            # 宽松口径下唯一算「没抓住」的情形（推错对象）

        def _pc(r):
            return f"{r*100:6.2f}%" if _finite(r) else "    NA"

        print("\n  ★★★ 最优先判据 · 机会把握率（18.7.4）: 分子 = GNN 第一名真的可用的组，"
              "分母 = 真值上存在机会的组 ★★★")
        print(f"    严格（GNN 第一名**严格优于** current，t₁ < t_cur）: "
              f"{_pc(_o[8])}   = {_h3}/{_d3} 组")
        print(f"    宽松（GNN 第一名**是 current 本身**，或其真值**不劣于** current，t₁ ≤ t_cur）: "
              f"{_pc(_o[9])}   = {_h4}/{_d4} 组")
        print(f"    机会组三分（严格 D3={_d3}，按 GNN 第一名）: 抓住(候选且真改进) {_h3} ｜ "
              f"守住 current {_n1} ｜ 推错对象(候选但更差) {_wrong3}"
              f"   ⇒ 严格只认「抓住」；宽松把「守住 current」也算通过")
        print(f"    ⚠ 两口径**分母不同、禁互读**：严格 = 存在严格更优候选的组（D3={_d3}）；"
              f"宽松 = 存在更优**或并列**候选的组（D4={_d4}，本树多 {_d4-_d3} 组「只有真并列」）。"
              f"宽松 ≡ 1 − 推错率（{_wrong4}/{_d4}）⇒ 它是「**没犯错**」读数，"
              f"要问「**抓住机会**」只看严格那格。")
        print(f"    ⚠ 1 − R1 **不是**把握率：R1 只记「首推仍是 current」（= 下面 R1/R2 表的 N1={_n1}），"
              f"它把「守住 current」与「推错对象」并成同一格 ⇒ 两者必须分开读（见上面三分）。")
        k4 = {(x["circuit"], x["iter"], x["window"]) for x in recs}
        print(f"\n  ★ 机会集口径 R1/R2（18.7.0；分母 = 真值上存在机会的组，"
              f"分子 = 其中 GNN 把 current 排第 1 的组）:")
        print("    " + _pad("口径", 34) + _pad("组数", 7) + _pad("D3", 7) + _pad("D4", 7)
              + _pad("N1", 7) + _pad("N2", 7) + _pad("R1严格", 9) + "R2宽松")
        for lab, ss in (("全部组（min_cands=1）", arecs),
                        (f"其中 ≥{min_cands} 候选（对齐既有口径）",
                         [x for x in arecs if (x["circuit"], x["iter"], x["window"]) in k4]),
                        ("单输出层（nout=1）", [x for x in arecs if x["nout"] == 1]),
                        ("多输出层（nout>=2）", [x for x in arecs if x["nout"] >= 2])):
            # ⚠ _opp 现在返回 10 元组（18.7.4 加了 h3/h4 与两格把握率），本表只用前 6 格。
            #   写成 `= _opp(ss)` 会 ValueError（too many values to unpack）—— 别改回去。
            d3, d4, n1, n2, r1, r2 = _opp(ss)[:6]
            print("    " + _pad(lab, 34) + _pad(str(len(ss)), 7) + _pad(str(d3), 7) + _pad(str(d4), 7)
                  + _pad(str(n1), 7) + _pad(str(n2), 7)
                  + _pad(f"{r1*100:.2f}%" if _finite(r1) else "NA", 9)
                  + (f"{r2*100:.2f}%" if _finite(r2) else "NA"))
        d3a, d4a = _opp(arecs)[:2]
        print(f"    ⚠ 分母为 0 的组不入比值：D3=0 的组 {len(arecs)-d3a} 个（无可厚非：GNN 首推 current 是对的）"
              f"   D4=0 的组 {len(arecs)-d4a} 个   D4−D3 = {d4a-d3a} 个组只有真并列的候选")
        print(f"    伴随 top-1 命中（含 current 同排）: 严格 "
              f"{statistics.mean([x['top1_strict'] for x in arecs])*100:.1f}%  宽松 "
              f"{statistics.mean([x['top1_len'] for x in arecs])*100:.1f}%"
              f"   （本块候选集 = **全部**候选，不受 min_cands 截断 ⇒ 与既有 top-1 分母不同）")
        print(f"    current 与某候选同秩（并列第 1）的组 {int(sum(x['tie_rank'] for x in arecs))} 个"
              f"   组内存在与 current **真值相等**的候选的组 {int(sum(x['tie_true'] for x in arecs))} 个")
        print("    ⚠ 判据尺：本节判「更好」用 avg_delay（全输出口均值）。**bestfirst 变体**的搜索"
              "判据就是 avg_delay ⇒ 与本节同尺、无代理误差（R1/R2 = GNN 与搜索判据**同判**的组"
              "比例）；**老贪心**（shadow/gnn）判据是 affected_delay（本窗受影响口）⇒ 那些树上本节"
              "含代理误差（偏悲观、是上界），单/多输出分层是解释它的必需项。"
              "上面四行分母各不相同（min_cands 与分层），**禁互读**。")
        print("    ⚠ 与 18.6.1 的 Sp/τ 不可比：本节秩出自「[current]+全部候选」同批（tie-corrected），"
              "且 R1/R2 是**组级条件概率**（分母 = 有机会的组），不是相关系数。")

    if not recs:
        # 18.7.0：这里原先直接 `return` ⇒ 上面两处「见下面诊断计数」都成了**空指**（诊断块在 return
        # 之后），而「一个合格集都没有」恰恰是最需要看诊断的时候。诊断块全是纯计数器、不依赖 recs，
        # 且下方所有 recs 相关读数都自带有空集守卫（空层打 NA/`（无集）`，不是崩）⇒ 取消早退，
        # 让它照常走到诊断计数块；本节各表则一律 NA/nan（消息改成与实情一致，不再说「不报数」）。
        print("  ⚠ 本节无合格集 ⇒ 下面各表一律 NA，只有诊断计数值得看（本节数字作废）")
    main_keys = {(s["circuit"], s["iter"], s["window"]) for s in main_sets}
    cur_keys = {(s["circuit"], s["iter"], s["window"]) for s in recs}
    print(f"  覆盖率: 仅主口径有 {len(main_keys - cur_keys)} 集 ｜ 仅本节有 "
          f"{len(cur_keys - main_keys)} 集（两节分母不同时，跨节比较指标要先看这两项）")
    print("  ⚠ 两把尺：本节的秩出自「[current]+全部候选」同批，主口径 gnn_pred 出自「纯候选批」"
          "——**禁混读、禁跨节比较秩与相关系数**（本节 Sp 还是 tie-corrected，见函数头）")
    print("  ⚠ 判据尺：「劣于」用 avg_delay（全输出口均值）。bestfirst 变体的搜索判据就是它"
          "⇒ 同尺、无代理误差；老贪心（shadow/gnn）吃 affected_delay（本窗受影响口）⇒ 那些树上"
          "本节误报率是部署口径的**必要不充分代理**（被判误报者仍可能被贪心正当接受）")

    def _mean(ss, k):
        v = [x[k] for x in ss if _finite(x.get(k))]
        return statistics.mean(v) if v else float("nan")

    S_tot = sum(x["S"] for x in recs); FP_tot = sum(x["FP"] for x in recs)
    P_tot = sum(x["P"] for x in recs); FN_tot = sum(x["FN"] for x in recs)
    n_zero_S = sum(1 for x in recs if x["S"] == 0)
    n_zero_P = sum(1 for x in recs if x["P"] == 0)
    # —— ★ 核心：只看首推那一个候选「是否真负」 ——
    n_t1_win = sum(1 for x in recs if x["t1_win"] == 1.0)
    t1neg = sum(x["t1_neg"] for x in recs if _finite(x.get("t1_neg")))
    t1nn = sum(x["t1_nn"] for x in recs if _finite(x.get("t1_nn")))
    n_tie = sum(1 for x in recs if x["t1_tie"] == 1.0)
    n_mix = sum(1 for x in recs if x["t1_mix"] == 1.0)
    print(f"\n  ★ 核心 首推真负率（只看 GNN 排第一的那个候选**是否真改进**，"
          f"不再逐项看所有排在 current 之前的）:")
    print(f"      严格（t₁ < t_cur，与 `delta<0` 同调）: {_mean(recs, 't1_neg')*100:6.2f}%   "
          f"宽松（t₁ ≤ t_cur，不劣）: {_mean(recs, 't1_nn')*100:6.2f}%   "
          f"（每格 = 窗均；窗是单位）")
    print(f"      分母 = GNN 严格首推了候选**且首推有真值**的窗 {n_t1_win} 个"
          f"（占合格集 {len(recs)} 的 {n_t1_win/len(recs)*100:.1f}%）；"
          f"池化 Σ真负/Σ窗 = {t1neg/n_t1_win*100:.2f}% / 宽松 {t1nn/n_t1_win*100:.2f}%"
          if n_t1_win else "      分母 = 0 个窗（无可用集）")
    # s0t = 「GNN 首推 current」的窗数（= 上面分母缺口的第一个桶，两者必须是同一个集合）
    s0t = int(sum(x["s0"] for x in recs)); s0m = int(sum(x["s0_miss"] for x in recs))
    print(f"      分母缺口（三项相加必须 = 合格集 {len(recs)}）：GNN 首推 current 的窗 {s0t} 个"
          f" ＋ 首推候选**无真值**的窗 {n_t1_nodata} 个（落在未评估后缀 ⇒ 判不了，"
          f"而部署上贪心先试的恰是它）＋ 进分母 {n_t1_win} 个")
    print(f"      镜像 漏推率（防「永远说 current 最好」好看）：GNN 首推 current 的 {s0t} 个窗里，"
          f"其实**存在更优候选**的 {s0m} 个 ⇒ {s0m/s0t*100 if s0t else float('nan'):6.2f}%"
          f"（下界：未评估后缀里的更优候选无从知道）")
    print(f"      首推并列的窗 {n_tie} 个（并列要求**全组都真负**才算严格正确；"
          f"其中组内真假不一 {n_mix} 个 ⇒ 按保守判错）")
    print(f"\n  —— 以下为**逐候选口径（参考）**：把「所有排在 current 之前的候选」逐项汇总，"
          f"分母是候选而非窗；核心结论请引上面的首推率 ——")
    print(f"  头号 误报率（GNN 把**真不优于** current 的排到了 current 之前）: "
          f"{_mean(recs, 'fp_rate')*100:6.2f}%   分母 ΣS={S_tot} 项 / {len(recs)-n_zero_S} 集")
    print(f"  伴随 漏报率（真有更好却被 GNN 判为不如 current）:                 "
          f"{_mean(recs, 'fn_rate')*100:6.2f}%   分母 ΣP={P_tot} 项 / {len(recs)-n_zero_P} 集")
    print(f"  池化（ΣFP/ΣS / ΣFN/ΣP，不给小分母集加权）: "
          f"误报 {FP_tot/S_tot*100 if S_tot else float('nan'):6.2f}%   "
          f"漏报 {FN_tot/P_tot*100 if P_tot else float('nan'):6.2f}%")
    print(f"  top-1 命中（含 current 一起排）: 严格 {_mean(recs, 'top1_strict')*100:6.1f}%   "
          f"宽松 {_mean(recs, 'top1_len')*100:6.1f}%")
    print(f"  含 current 批内相关: Spearman(修正) {_mean(recs, 'sp'):6.3f}   "
          f"Kendall τ-b {_mean(recs, 'kt'):6.3f}   "
          f"(n={sum(1 for x in recs if _finite(x.get('sp')))} 集；与主口径的 Spearman 约定不同)")
    # 18.7.0：recs 空时 min()/median()/max() 会 ValueError/StatisticsError（原先靠上面的 return
    # 挡住；取消早退后必须自带守卫，否则「一个合格集都没有」的树直接崩在报错路径上）
    if recs:
        print(f"  集内候选数: min={min(x['n_cands'] for x in recs)} "
              f"med={statistics.median([x['n_cands'] for x in recs]):.0f} "
              f"max={max(x['n_cands'] for x in recs)}")
    else:
        print("  集内候选数: （无合格集）")
    print(f"  ⚠ |S|=0 的集 {n_zero_S} 个（该窗口**有真值的**候选里没有一个被排在 current 之前；"
          f"注意与核心口径的缺口行不是同一个集合——那边按秩判，不看真值有没有）"
          f"，⇒ 误报率**无定义**、已从均值与分母里剔除；|P|=0 的集 {n_zero_P} 个同理。"
          f"不剔除的话「谁都不排在 current 之前 ⇒ 误报率 0%」会是假好")

    # —— 本节指标的四口径（复用主表的规格；③ 按电路、④ 排除主表那个最大电路）——
    _seen = set(); dedup = []
    for s in recs:
        k = (s["circuit"], s["sig"])
        if k in _seen:
            continue
        _seen.add(k); dedup.append(s)
    # ★ 核心 推严/推宽 也在里面 —— 四口径与分层表都据此多出一列（列序见表头）
    keys = ("t1_neg", "t1_nn", "fp_rate", "fn_rate", "top1_strict", "top1_len", "sp", "kt")
    # ⚠ 标签必须**前有字**（`口径①` 而非 `①`）：_t_shadow_groupby.py:127 的 table_units() 用
    # `strip().startswith("①②③④")` 认表、toks[-7] 当单位数（int()）—— 本节的行若以圈码起头会被
    # 它当成口径对照表的行**覆盖**既有读数，且末 7 个 token 是百分数 ⇒ `int()` 直接抛异常。
    specs = (
        ("口径① 全部批（等权）", recs, False, len(recs)),
        ("口径② 按候选池去重（同电路同 sig）", dedup, False, len(dedup)),
        ("口径③ 按电路宏平均（每电路等权）", recs, True, len({s["circuit"] for s in recs})),
        (f"口径④ 排除最大电路 {os.path.dirname(dom_circ).replace(os.sep, '/') or dom_circ}",
         [s for s in recs if s["circuit"] != dom_circ], False,
         sum(1 for s in recs if s["circuit"] != dom_circ)),
    )
    print("\n  本节指标的四口径（与上面主表同规格；每格 nan = 该口径无可用集）：")
    print("    " + _pad("口径", 32) + _pad("单位数", 8) + _pad("★推严", 9) + _pad("★推宽", 9)
          + _pad("误报率", 10) + _pad("漏报率", 10) + _pad("top1严格", 10) + _pad("top1宽松", 10)
          + _pad("Sp(修正)", 10) + "Kendall")   # ★ = 核心（首推真负率）两口径
    for label, ss, macro, n_units in specs:
        a = _agg_keys(ss, keys, macro)
        print("    " + _pad(label, 32) + _pad(str(n_units), 8)
              + _pad(f"{a[0]*100:.2f}%" if _finite(a[0]) else "NA", 9)
              + _pad(f"{a[1]*100:.2f}%" if _finite(a[1]) else "NA", 9)
              + _pad(f"{a[2]*100:.2f}%" if _finite(a[2]) else "NA", 10)
              + _pad(f"{a[3]*100:.2f}%" if _finite(a[3]) else "NA", 10)
              + _pad(f"{a[4]*100:.1f}%" if _finite(a[4]) else "NA", 10)
              + _pad(f"{a[5]*100:.1f}%" if _finite(a[5]) else "NA", 10)
              + _pad(f"{a[6]:.3f}" if _finite(a[6]) else "NA", 10)
              + (f"{a[7]:.3f}" if _finite(a[7]) else "NA"))

    # —— 单/多输出分层（**老贪心下是必需项**：口径③的代理误差只在多输出层出现。bestfirst 下
    #    两把尺同尺 ⇒ 分层退化为「看两层覆盖/难度差异」的辅助项，仍打印，便于两变体对比）——
    print("\n  单/多输出分层（解释上面读数用；多输出电路占候选集 52~57%）：")
    print("    " + _pad("层", 20) + _pad("集数", 7) + _pad("★推严", 9) + _pad("★推宽", 9)
          + _pad("误报率", 10) + _pad("漏报率", 10) + _pad("top1严格", 10) + _pad("Sp(修正)", 10)
          + "池化误报")
    for lab, ss in (("单输出(nout=1)", [s for s in recs if s["nout"] == 1]),
                    ("多输出(nout>=2)", [s for s in recs if s["nout"] >= 2])):
        if not ss:
            print("    " + _pad(lab, 20) + "（无集）")
            continue
        St = sum(x["S"] for x in ss); Ft = sum(x["FP"] for x in ss)
        print("    " + _pad(lab, 20) + _pad(str(len(ss)), 7)
              + _pad(f"{_mean(ss, 't1_neg')*100:.2f}%" if _finite(_mean(ss, 't1_neg')) else "NA", 9)
              + _pad(f"{_mean(ss, 't1_nn')*100:.2f}%" if _finite(_mean(ss, 't1_nn')) else "NA", 9)
              + _pad(f"{_mean(ss, 'fp_rate')*100:.2f}%", 10)
              + _pad(f"{_mean(ss, 'fn_rate')*100:.2f}%", 10)
              + _pad(f"{_mean(ss, 'top1_strict')*100:.1f}%", 10)
              + _pad(f"{_mean(ss, 'sp'):.3f}", 10)
              + (f"{Ft/St*100:.2f}%" if St else "NA"))
    print("    ⚠ 单输出层里 avg_delay ≡ affected_delay（与老贪心判据同口径）；多输出层里两者只在"
          "「本窗受影响口 = 全输出口」时相等 ⇒ **老贪心（shadow/gnn）树上**多输出层的误报率含"
          "代理误差，解读时先看分层；bestfirst 变体判据本就是 avg_delay ⇒ 分层只作覆盖差异参考")

    # —— 诊断计数（fail-loud：配对类必须全 0，否则本节数字不可用）——
    print("\n  诊断计数:")
    print(f"    gnn_current 窗口数（全部, 含未成集）: {n_win_total}"
          f"   有 gnn_current 无 gnn_shadow（半趟）: {n_no_shadow}")
    # 窗内 eval_idx 重复**属于配对失败**（该窗的 eval_idx↔行 对应关系有歧义 ⇒ 该窗已被剔出
    # recs）—— 故与双射/缺 eval/tc 同列，不放进下面那组 ⚠（否则「被剔了但没报警」）。
    print(f"    配对失败 双射集合不等: {n_mismatch}   推断位置非前缀(pos 缺失): {n_ev_missing}"
          f"   tc 不等: {n_tc_mismatch}   窗内 eval_idx 重复: {n_dup_win}"
          f"   {'✅' if (n_mismatch + n_ev_missing + n_tc_mismatch + n_dup_win) == 0 else '❌ 配对不可信 → 本节数字作废'}")
    print(f"    只有 CURRENT 行的窗口（rank_window 的空候选守卫漏了）: {n_empty_cand}"
          f"   {'✅' if n_empty_cand == 0 else '❌ 量纲混合：批次只剩 current，serve 回的是原始延迟'}")
    print(f"    量纲守卫（秩越界 / 非整数或半整数）: {n_dim}"
          f"   {'✅' if n_dim == 0 else '❌ 有窗口把「秒」当「秩」写了'}")
    print(f"    同电路 eval_idx 重复行: {n_dup_eval}"
          f"   表头重复: {n_header_dup}   坏行: {n_bad_row}"
          f"   {'✅' if (n_dup_eval + n_header_dup + n_bad_row) == 0 else '⚠ 疑 append 叠趟/损坏行'}")
    print(f"    候选真值缺失(true_delay=NA，已剔出该集): {n_true_na}"
          f"   候选秩缺失/NaN(已剔): {n_rank_na}")
    # 17.6.0：把上面那批 NA 行**按成因**分出一支 —— 单次仿真被 `XYCE_TIMEOUT_S` 掐掉的
    # 有多少。判据 = 该行 error 列含 timeout（见 parse_row 的 CSV_ERR_RE 与上面累加处）。
    # 读法：>0 说明保险**真的触发过**（组内允许：该候选作废、落 NA 行、不写缓存）；
    #       结构性未评估（下一行）则**必须**为 0，两者是不同性质的东西，别混读。
    print(f"    其中由超时造成(单次仿真被 XYCE_TIMEOUT_S 掐掉): {n_timeout_na} 个"
          f"   {'✅ 本趟没有候选被掐' if n_timeout_na == 0 else '⚠ 保险已触发 —— 组内允许；连续 5 次会中止整个电路（DONE 行 error）'}"
          f"   （其余 {n_true_na - n_timeout_na} 个 NA 行是别的失败；本行只看 error 列，不参与任何判据）")
    print(f"    结构性未评估候选（贪心该窗提前 accept 并 break ⇒ 尾部候选永不评估、无真值）: "
          f"{n_uneval} 项，其中「秩高于 current」{n_uneval_above} 项 ⇒ 这 {n_uneval_above} 项"
          f"**计入不了 S 的分母**，S 与误报率因此只覆盖「贪心真试过」的那部分候选")
    print(f"    current 秩缺失/NaN 的集 {n_cur_rank_na}（= 模型评估不了 current，"
          f"**不是**「GNN 一个都没看上」）   current 真值缺失的集 {n_cur_true_na}")
    # 18.7.0：机会集口径（R1/R2）自己的前置校验 —— 四项全 0 才有资格读那个分母
    print(f"    机会集口径（18.7.0）前置校验失败: 配对/覆盖(pos 集非全集) {n_opp_pair}"
          f"   current 秩或真值缺 {n_opp_cur_na}   候选秩缺 {n_opp_rank_na}"
          f"   候选真值缺/tc 不等 {n_opp_true_na}"
          f"   {'✅' if (n_opp_pair + n_opp_cur_na + n_opp_rank_na + n_opp_true_na) == 0 else '❌ 该树不是全组仿真的 ⇒ R1/R2 的分母只是下界'}"
          f"   （另有量纲守卫 {n_dim} / 空候选窗 {n_empty_cand} / 窗内 eval 重复 {n_dup_win} 个窗在本块之外）")
    print(f"    gnn_shadow 有、gnn_current 无的窗口: {n_win_unclaimed}"
          f"（预期每电路约 1 个 = 初始 current 那次 eval_idx=1；候选全空（`prepared` 空）的窗"
          f"不写本文件 —— 注意 lift/patch 失败的候选**根本进不了** `mods`，故不会出现在这里）")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-cands", type=int, default=4, help="候选集最少候选数（10.3 要求 ≥4）")
    ap.add_argument("--root", default=os.path.expanduser("~/NetlistOpt/temp_sim_test/tl_opt_batch"),
                    help="批量输出根目录")
    # 17.3.9: 分组键。默认 batch = 部署真口径（见文件头「统计单位」三条源码证据）。
    ap.add_argument("--group-by", choices=("batch", "window"), default="batch",
                    help="候选集分组键：batch=(电路,iter,window_try)=pre_rank 的单批（部署真口径，默认）；"
                         "window=(电路,window_try)=跨轮池化（仅用于复现 17.3.9 之前的数）")
    ap.add_argument("--detail-max", type=int, default=-1,
                    help="文末逐集明细最多列几集（按遗憾降序，列最差的）；-1=不列（18.7.4 起默认："
                         "该块是最大的一段而信息密度低），0=全列，N>0=列最差的 N 集")
    ap.add_argument("--full", action="store_true",
                    help="打印全部小节。18.7.4 起默认只打**核心读数**：最优先判据（机会把握率）＋ "
                         "R1/R2、首推真负率、recall、10.3 判定、交付口径、fail-loud 计数；省略的是"
                         "纯诊断块（跨度>10% 子集 / 并列・一致性诊断 / 每候选集明细），本开关可随时"
                         "复现（离线秒级，无需重跑仿真）")
    args = ap.parse_args()

    run_stamp(args.root)
    print(f"候选集分组键: --group-by {args.group_by}  " + (
        "(电路, iter, window_try) = 一次 pre_rank 的批【部署真口径】"
        if args.group_by == "batch" else
        "(电路, window_try) = 跨轮池化【历史口径，仅供复现 17.3.9 之前的数】"))
    if args.group_by == "window":
        print("  ⚠ 历史口径把 ~6.7 个批池在一起，而 gnn_pred 是批内秩、跨批不可比 →"
              "「GNN 前3」实为最早 3 批各自的第 1 名；下面的数不是部署性能。")

    sets = []          # 全部候选集（≥min-cands）
    ab_sets = []       # 两列并排：**同一批行、同一候选集**上分别用两列各算一遍
    small_sets = 0
    ok_rows = 0; fail_rows = 0
    c2_rows = 0                       # 文本上带第二列的行数
    ab_dropped_sets = 0               # 主口径合格、但因第二列 NaN 落掉的集
    # 17.3.11 输入污染检查（见文末「输入污染检查」块）
    dedup_dropped = 0                 # 同一 (it,w) 分组内因 eval_idx 重复被丢的行（原本静默）
    eval_dup_rows = 0                 # 同电路内 eval_idx 出现多次的「多余行数」
    eval_conflict = 0                 # 同一 eval_idx 对应多于一个 true_delay 的处数（叠行硬证据）
    # 17.3.9: 分组键 —— batch = (iter, window_try) = 一次 pre_rank 的批（部署真口径）；
    #   window = 历史口径，池化跨轮的批，仅供复现旧数。见文件头「统计单位」。
    def _group_key(r):
        return (r["iter"], r["window"]) if args.group_by == "batch" else (None, r["window"])

    for csvp in sorted(glob.glob(os.path.join(args.root, "*", "*", "gnn_shadow.csv"))):
        circ = os.path.relpath(csvp, args.root)
        by_key = {}
        ev_count = {}      # 17.3.11: eval_idx -> 出现次数（本电路内）
        ev_true = {}       # 17.3.11: eval_idx -> 见过的 true_delay 文本集合
        with open(csvp) as f:
            for line in f:
                r = parse_row(line)
                if not r:
                    continue
                if r["true"] is None or r["gnn"] is None:
                    fail_rows += 1
                    continue
                ok_rows += 1
                if r["c2"]:
                    c2_rows += 1
                ev_count[r["eval"]] = ev_count.get(r["eval"], 0) + 1
                ev_true.setdefault(r["eval"], set()).add(r["ttxt"])
                by_key.setdefault(_group_key(r), []).append(r)
        eval_dup_rows += sum(ev_count.values()) - len(ev_count)
        eval_conflict += sum(1 for v in ev_true.values() if len(v) > 1)
        for key, rows in by_key.items():
            it_idx, w = key
            # 同 window 内按 eval_idx 去重（同一候选可能被缓存复用？以 eval_idx 唯一为准）
            seen = {}
            for r in rows:
                seen.setdefault(r["eval"], r)
            dedup_dropped += len(rows) - len(seen)
            rows = list(seen.values())
            # 按 eval_idx 定序：并列的打破**不能**依赖 CSV 行序（CSV 由并行分片 append 写，
            # 见 parse_row 里「并发写 CSV 交错」；行序不确定）。
            #
            # 17.2.7 更正旧注：旧注说「gnn_pred 只写 7 位有效数字 → 大量伪并列」——**该机制已不存在**。
            # gnn_pred 写的是 `/rank` 返回的 avg_delay 字段，而 serve.predict_rank_batch 往里放的是
            # **候选集内的平均秩**（competition ranking，小整数/半整数），7 位有效数字可精确表示，
            # 不存在截断伪并列。只有原始延迟（~1e-11）才会被 {:.6e} 截断，而那条路径只在预排序
            # 缓存未命中时逐候选兜底（gnn_shadow.rs evaluate），且 pre_rank→evaluate 是同一批
            # prepared（tl_opt.rs Pass2/Pass3）→ **集内要么全秩、要么全原始延迟，不会混**。
            # 那次 0.00% ↔ 36.34% 翻转的真正源头后来定在 float32 边序抖动
            # （graph_builder.parse_netlist 的 list(set(edges))），17.2.4 已用 sorted(set(edges))
            # 定序（证据与验收见 scripts/diag/serve.py:307-313 与 _t_serve_repro.sh）。
            # 这里的定序保留为**廉价确定性保险**，不是那次翻转的修复。
            rows.sort(key=lambda r: r["eval"])
            if len(rows) < args.min_cands:
                small_sets += 1
                continue
            m = per_window_metrics(rows)
            if m is None:
                continue
            m["circuit"] = circ; m["window"] = w; m["iter"] = it_idx
            # 17.3.10: 跨集判同指纹 = 该集 (true_delay, transistors) 的**多重集**（排序后元组）。
            # 用多重集而非集合：候选被重复提出但少了一个时，n 不同 → 不该判同（那属「真包含子集」，
            # 由下面的诊断单独数）。取自真值列 → 与模型无关。
            m["sig"] = tuple(sorted((r["ttxt"], r["nt"]) for r in rows))
            sets.append(m)
            # —— 第二列：只用**两列都非 NaN**的行，两个模型跑在同一行集上 ——
            # 这样 A/B 是严格同分母的配对比较；被 NaN 挤掉的集单独计数上报，不静默消失。
            if c2_rows:
                rows2 = [r for r in rows if r["gnn2"] is not None]
                if len(rows2) < args.min_cands:
                    ab_dropped_sets += 1
                    continue
                a = per_window_metrics(rows2)
                b = per_window_metrics(rows2, "gnn2")
                if a is None or b is None:
                    ab_dropped_sets += 1
                    continue
                for mm in (a, b):
                    mm["circuit"] = circ; mm["window"] = w; mm["iter"] = it_idx
                ab_sets.append((a, b))

    if not sets:
        print(f"无合格候选集（需要 ≥{args.min_cands} 候选）。成功行={ok_rows} 失败行={fail_rows} 小集={small_sets}")
        return

    # —— 18.7.4 两节共用的前置（提到各节之前）——
    # by_circ / _dom_circ 原先在「跨集重复度」块里现算（只为「交付口径对照 ④」与「以 current 为锚」
    # 两处服务）；提上来是为了把**最优先判据**那节挪到文首（见下面的调用），让读数第一屏就能看到 ——
    # 此前它在 20 KB 之后，重点被淹没。计算式与输入一字未改。
    by_circ = {}
    for s in sets:
        by_circ.setdefault(s["circuit"], []).append(s)
    _dom_circ = max(by_circ, key=lambda c: len(by_circ[c]))

    # —— 以 current 为锚的 GNN 判据质量（18.6.0 新增 / 18.6.1 修正配对校验）——
    # ⚠ 位置被两处约束夹住，**只能上提、不要下挪**：
    #  (a) 必须在「=== 每候选集明细」之**前**：_t_shadow_analyze_2col.py 的 head_meta() 把
    #      DETAIL_MARK 之前的一切逐字节跨树比较 ⇒ 本节在无 gnn_current.csv 的树上必须输出
    #      **常量**（已如此，见 current_anchor_report 的早退分支）。
    #  (b) 必须在「=== 两列并排 A/B」之**前**：sweep 的 pick() 取文件序首处，且让本节留在
    #      主口径头里 = 让 (a) 那条逐字节断言顺带守住本节的缺文件路径。
    #  18.7.4：从「交付口径对照」之后**提到**此处（两条约束都是上界，上提不冲突）⇒
    #  ★★★ 最优先判据（机会把握率）落在第一屏。
    current_anchor_report(args.root, args.min_cands, sets, _dom_circ)

    r3 = [s["recall3"] for s in sets]
    rg = [s["regret"] for s in sets]
    sp = [s["spearman"] for s in sets if s["spearman"] is not None]
    n_cands = [s["n"] for s in sets]
    r2s = [s["recall2_strict"] for s in sets]
    r3s = [s["recall3_strict"] for s in sets]
    r2l = [s["recall2_len"] for s in sets]
    r3l = [s["recall3_len"] for s in sets]
    r2st = [s["regret_2stage"] for s in sets]
    c2s = [s["capture2"] for s in sets if not np.isnan(s["capture2"])]

    print(f"候选集数（≥{args.min_cands} 候选）: {len(sets)}   成功行={ok_rows} 失败行={fail_rows} 小集={small_sets}"
          f"   [--group-by {args.group_by}]")
    if args.group_by == "batch":
        print(f"  （batch 口径下小集天然多：多数批只提 1~3 个候选 → 不足 {args.min_cands} 个的被滤掉；"
              f"它们正是「候选太少、无需 GNN 粗筛」的那部分，不构成 recall 的分母）")
    print(f"候选数分布: min={min(n_cands)} med={statistics.median(n_cands):.0f} max={max(n_cands)}")

    # —— 输入污染检查（17.3.11）——
    # 动机：OPERATIONS:177 记过 append 事故 —— 手敲 cargo test 绕过脚本内的清理，把上一趟的旧行
    #   追加进同一个 gnn_shadow.csv。若真发生，本快照就不是单趟数据，所有读数都要重估。
    # 三个数：① 原本**静默**的 eval 去重丢弃（同一 (it,w) 内 eval_idx 重复）；
    #   ② 同一 eval_idx 落在不同 (it,w) 的重复行；③ 同一 eval_idx 却对应两个不同 true_delay
    #   —— 同一次评估不可能有两个真值，故 ③>0 即叠行的硬证据。
    # ⚠ 本块按**电路自比**（每个 CSV 内比）：若实际上 eval_idx 是跨电路全局唯一的，则跨电路的
    #   重复不会被这里发现。这里不预设它们的关系，只报测到的。
    print("\n=== 输入污染检查（17.3.11，按电路自比）===")
    print(f"  分组内 eval 去重丢弃行（原本静默）: {dedup_dropped} "
          f"{'✅' if dedup_dropped == 0 else '⚠ 同一 (it,w) 里有重复 eval_idx'}")
    print(f"  同电路 eval_idx 跨分组重复行:       {eval_dup_rows - dedup_dropped} "
          f"{'✅' if eval_dup_rows == dedup_dropped else '⚠ 同一 eval_idx 落在不同 (it,w)'}")
    print(f"  同一 eval_idx 真值冲突:             {eval_conflict} 处 "
          f"{'✅ 每个 eval 只对应一个真值' if eval_conflict == 0 else '❌ 同一次评估有两个真值 → 两趟的行叠在一起，本快照所有数字须重估'}")
    if dedup_dropped == 0 and eval_dup_rows == 0 and eval_conflict == 0:
        print("  → ✅ 三项全 0：未见 append 叠行迹象，下面的读数按「单趟数据」解释")

    print(f"\n=== recall 判断标准（k=2/3 双口径，粗筛最重要指标）===")
    print(f"  前k名中出现实际第1名（严格）:  k=2 {statistics.mean(r2s)*100:6.1f}%   k=3 {statistics.mean(r3s)*100:6.1f}%")
    print(f"  前k名中出现实际前k之一（宽松）: k=2 {statistics.mean(r2l)*100:6.1f}%   k=3 {statistics.mean(r3l)*100:6.1f}%")
    print(f"\n=== 10.3 判定指标（按候选集汇总）===")
    print(f"  选择遗憾（GNN自选top1）: {statistics.mean(rg)*100:6.2f}%   (达标线 ≤5%)   "
          f"中位 {statistics.median(rg)*100:.2f}%")
    print(f"  两阶段最终遗憾（前3→SPICE精排）: {statistics.mean(r2st)*100:6.2f}%   "
          f"中位 {statistics.median(r2st)*100:.2f}%")
    if c2s:
        print(f"  两阶段捕获率（前3→精排，spread 归一，越高越好）: {statistics.mean(c2s)*100:6.2f}%   "
              f"中位 {statistics.median(c2s)*100:.2f}%  (n={len(c2s)}/{len(sets)} 集，需 n>=4 候选)")
    if sp:
        print(f"  Spearman:       {statistics.mean(sp):6.3f}   (次判据 ≥0.6)   "
              f"中位 {statistics.median(sp):.3f}  (n={len(sp)} 集)")
    # 17.4.0：原来只打 ✅/❌ 不打值 —— 于是「判定」这一行**不可审计**：判据是 recall3(集合版)，
    # 而全文没有任何地方印过这个量的数值（严格/宽松 k3 都是**另一个**指标），读者只能看到结论。
    # 现在把参与判定的两个均值一并印出，并标明它只对应 ① 口径（②③④ 见文末对照表，可能翻号）。
    _r3m, _rgm = statistics.mean(r3), statistics.mean(rg)
    r3_ok = _r3m >= 0.90
    rg_ok = _rgm <= 0.05
    print(f"\n判定(旧口径参考，仅 ① 全部批口径): "
          f"recall@top-3(集合版) {_r3m*100:.1f}% {'✅≥90%' if r3_ok else '❌<90%'}  "
          f"选择遗憾 {_rgm*100:.2f}% {'✅≤5%' if rg_ok else '❌>5%'}")
    print("  ↑ 判据是**集合版 recall@3**（真前3∩预测前3 / 3），不是上面那两行严格/宽松 k3。"
          "口径换成 ②③④ 结论可能翻号 → 报数前先定口径，别只看这一行。")
    if r3_ok and rg_ok:
        print("→ **两项主判据达标：GNN 可替换逐候选 SPICE 排序**（top-K 精排，仿真省 ≥75%）")
    else:
        print("→ **主判据未达标：GNN 只做启发式预排序**（SPICE 全排序 + GNN 先粗排）")

    # 跨度过滤视图：只统计 (max_true-min_true)/min_true > 10% 的「可排序集」（对齐 V2 hi_spread 口径）
    hi = [s for s in sets if s["spread_pct"] > 10.0]
    if args.full and hi:               # 18.7.4：纯诊断块，默认不打（--full 复现）
        hr3 = [s["recall3"] for s in hi]
        hrg = [s["regret"] for s in hi]
        hsp = [s["spearman"] for s in hi if s["spearman"] is not None]
        hr2s = [s["recall2_strict"] for s in hi]
        hr3s = [s["recall3_strict"] for s in hi]
        hr2l = [s["recall2_len"] for s in hi]
        hr3l = [s["recall3_len"] for s in hi]
        hr2st = [s["regret_2stage"] for s in hi]
        hc2 = [s["capture2"] for s in hi if not np.isnan(s["capture2"])]
        print(f"\n=== 跨度>10% 子集（{len(hi)}/{len(sets)} 集，对齐 V2 hi_spread）===")
        print(f"  严格(实际第1∈预测前k):  k=2 {statistics.mean(hr2s)*100:6.1f}%   k=3 {statistics.mean(hr3s)*100:6.1f}%")
        print(f"  宽松(前k含实际前k之一): k=2 {statistics.mean(hr2l)*100:6.1f}%   k=3 {statistics.mean(hr3l)*100:6.1f}%")
        print(f"  两阶段最终遗憾: {statistics.mean(hr2st)*100:6.2f}%   "
              f"选择遗憾(GNN自选): {statistics.mean(hrg)*100:6.2f}%   Spearman: {statistics.mean(hsp):.3f} (n={len(hsp)})")
        if hc2:
            print(f"  两阶段捕获率: {statistics.mean(hc2)*100:6.2f}%   "
                  f"中位 {statistics.median(hc2)*100:.2f}%  (n={len(hc2)}/{len(hi)} 集)")
    elif args.full:
        print("\n（无跨度>10% 的候选集）")

    # —— 并列/一致性诊断（17.2.7）——
    # 目的：把「并列打破规则的影响面」「重复网表天花板」「量纲混合」三件事变成数字，
    # 这样「训练侧 recall 与部署侧 recall 能不能对读」就不用靠推理，直接看这三行。
    n_dup = sum(1 for s in sets if s["dup_true"] > 0)
    n_tie = sum(1 for s in sets if s["tie_g_k3"])
    n_mix = sum(1 for s in sets if s["mixed"])
    n_raw = sum(1 for s in sets if s["raw_mode"])
    n_dupg = sum(1 for s in sets if s["dup_g"] > 0)
    # 18.7.4：本节是纯诊断（三件事的计数），默认不打；--full 或下面那一行提示可复现。
    if args.full:
        print(f"\n=== 并列/一致性诊断（17.2.7）===")
        print(f"  重复网表集（真值内有完全相等候选 → recall@3 天然 <100%）: {n_dup}/{len(sets)} 集")
        print(f"  预测第3小==第4小（top-3 归属由并列规则决定）:            {n_tie}/{len(sets)} 集")
        print(f"  预测有重复值（并列打破规则的影响面，全集口径）:          {n_dupg}/{len(sets)} 集")
        print(f"  量纲混合守卫（同集混「秩」与「原始延迟」→ 排序无意义）:   {n_mix}/{len(sets)} 集 "
              f"{'✅ 无' if n_mix == 0 else '❌ 有 → 该集排序不可信，先查 serve 预排序缓存命中'}")
        print(f"  兜底窗口（整集 gnn_pred 都是原始延迟，非秩）:             {n_raw}/{len(sets)} 集 "
              f"{'✅ 无（全走秩聚合）' if n_raw == 0 else '⚠ 该批 serve 预排序整窗未命中；集内排序仍有效，但分数分辨率受 CSV 的 7 位有效数字限制'}")
        if args.group_by == "window":
            print("  ⚠ 以上并列/一致性诊断在 --group-by window（跨轮池化）下**前提被破坏**："
                  "各条守卫都假定「集 = 一次 pre_rank 的批」（见 per_window_metrics 里「量纲混合守卫」"
                  "「兜底窗口识别」两段原注，以及 main 里 rows.sort 那段「pre_rank→evaluate 是同一批 prepared」），"
                  "池化把不同批的秩与 n==1 批的原始延迟混进同一集 → 读数仅作对照，不作判据。")
        if hi:
            h_dup = sum(1 for s in hi if s["dup_true"] > 0)
            h_tie = sum(1 for s in hi if s["tie_g_k3"])
            h_mix = sum(1 for s in hi if s["mixed"])
            h_raw = sum(1 for s in hi if s["raw_mode"])
            print(f"  （跨度>10% 子集 {len(hi)} 集: 重复网表 {h_dup}   第3==第4 {h_tie}   "
                  f"量纲混合 {h_mix}   兜底窗口 {h_raw}）")
    else:
        # 18.7.4：提示按**实际省了哪些**生成（--detail-max 0/N 时明细是打出来的，别把它也列进「已省略」）。
        # 本 else 只在 `not args.full` 时进来 ⇒ 两块诊断确实都没打。
        _skip = ["跨度>10% 子集", "并列・一致性诊断"]
        _how = ["--full 打这两块"]
        if args.detail_max < 0:
            _skip.append("每候选集明细")
            _how.append("--detail-max N 打明细 N 集（0=全列）")
        print("\n（已省略：" + " / ".join(_skip) + " —— " + "；".join(_how) + "）")

    # —— 跨集重复度（17.3.10）——
    # 上面那行「重复网表集」数的是**集内**真值相等的候选（天花板）；本块数的是**跨集**：
    # 同一电路的不同 (iter, window_try) 提出了**同一个候选池**。两者是不同的事，别混。
    # 只计数，不改上面任何指标。（by_circ 已在 main 开头建好 —— 18.7.4 提上去与「以 current 为锚」
    # 节共用，别在这里再建一份：两份会有漂移风险，而 ④ 与那节必须同分母。）
    n_uniq = 0                      # 互异候选池数
    mult = Counter()                # 重数 -> 具有该重数的独立集数
    worst = []                      # (重复集数, 电路, 总集数, 独立集数, 最大重数)
    n_sub = 0                       # 真包含子集：小集是大集的多重子集（相关但不同池）
    for circ, ss in by_circ.items():
        c = Counter(s["sig"] for s in ss)
        n_uniq += len(c)
        for k in c.values():
            mult[k] += 1
        if len(ss) > len(c):
            worst.append((len(ss) - len(c), circ, len(ss), len(c), max(c.values())))
        # 多重集包含：Counter 相减为空 ⟺ ⊆；再要求候选数严格更少 → 真子集。
        # 延迟是真值（连续量、CSV 写足有效位）→ 偶然命中的概率≈0，命中即真信号。
        cnts = [(Counter(s["sig"]), s["n"]) for s in ss]
        for i, (ci, ni) in enumerate(cnts):
            for j, (cj, nj) in enumerate(cnts):
                if i != j and ni < nj and not (ci - cj):
                    n_sub += 1
                    break
    n_dup_sets = len(sets) - n_uniq
    print(f"\n=== 跨集重复度（17.3.10，[--group-by {args.group_by}]）===")
    print("  判同口径: 同电路内，集的 (true_delay, transistors) 多重集**逐字节**相同 → 同一候选池")
    print(f"  独立集: {n_uniq}/{len(sets)}   重复集: {n_dup_sets}  "
          f"（{n_dup_sets/len(sets)*100:.1f}% 的集是同一池的重复观测）")
    print(f"  重数分布（重数×独立集数）: "
          + "  ".join(f"{k}×{mult[k]}" for k in sorted(mult))
          + f"   最大重数 {max(mult)}")
    if worst:
        print("  重复最多的电路（前 5，按重复集数降序）:")
        for d, circ, tot, uq, mx in sorted(worst, reverse=True)[:5]:
            print(f"    {circ:<45s} {tot:3d} 集 → {uq:3d} 独立（重复 {d}，最大重数 {mx}）")
    print(f"  真包含子集（同电路内小集是大集的多重子集，**未**计入上面的去重）: {n_sub} 集")
    if n_dup_sets or n_sub:
        print("  ⚠ 含义：上表把每集等权平均 → 被重复提出的池会按重数**加倍计权**，"
              "且这些集彼此不独立（同一批真值、同一批 gnn_pred）")
        print("     → 「逐集配对检验 / 符号检验 / bootstrap」的独立性假设在 batch 口径下不成立；"
              "需按电路做 block 重采样，或先按 sig 去重再统计")
    else:
        print("  ✅ 无跨集重复：候选池层面互异（独立集数 = 集数）")

    # —— 交付口径对照（17.3.11）——
    # 上面已证：714 个集是按**批**等权，而批被重复提议主导（一个电路可占 75%）。
    #   于是「平均批」不等于「部署时面对的总体」。本块把四种口径并排，值本身不判断对错，
    #   判断留给使用者 —— 交付数报哪个是口径决定，脚本不自作主张。
    #   ② 按候选池去重：同电路同 sig 只留一个 → 去掉重复观测的加倍计权
    #   ③ 按电路宏平均：先电路内平均、再跨电路平均 → 每个电路等权，消除「谁的批多谁说话」
    #   ④ 排除最大电路：即把「一家独大」整个拿掉，看剩下的是否同向
    def _agg(ss, macro=False):
        """返回 (严格k2, 严格k3, 宽松k2, 宽松k3, 选择遗憾, 两阶段遗憾)。
        macro=True：先按电路求均值，再跨电路求均值（每电路等权）。

        ⚠ 空口径返回全 nan，**不是** StatisticsError：④ 是「排除最大电路」，若快照里
        只有那一个电路（`--root` 指到单电路目录、或小快照），排除后就没东西可平均。
        真快照（46 个电路）不会触发，但为了脚本可本地/小样本试跑，这里必须挡住 ——
        否则前面几百行数字都算对了，最后一行却崩掉。"""
        if macro:
            g = {}
            for s in ss:
                g.setdefault(s["circuit"], []).append(s)
            units = list(g.values())
        else:
            units = [ss]
        units = [u for u in units if u]
        if not units:
            return (float("nan"),) * 6
        return tuple(statistics.mean(statistics.mean(x[k] for x in u) for u in units)
                     for k in ("recall2_strict", "recall3_strict", "recall2_len",
                               "recall3_len", "regret", "regret_2stage"))

    _seen_sig = set()
    dedup_sets = []
    for s in sets:
        k = (s["circuit"], s["sig"])
        if k in _seen_sig:
            continue
        _seen_sig.add(k)
        dedup_sets.append(s)
    # _dom_circ 已在 main 开头算好（18.7.4 提上去，供本节 ④ 与「以 current 为锚」节共用同一个值）
    # 显示名 = 目录部分（去掉 gnn_shadow.csv），且分隔符统一成 / ——
    # circuit 键是 os.path.relpath 的结果，Windows 下是 \ 分隔：硬写 split("/") 会整个失配，
    # 把含文件名的 49 列长串塞进 32 列的标签位。用 os.sep 才跨平台一致。
    _dom_name = os.path.dirname(_dom_circ).replace(os.sep, "/") or _dom_circ
    _spec = (
        ("① 全部批（现状，等权）", sets, False, len(sets)),
        ("② 按候选池去重（同电路同 sig）", dedup_sets, False, len(dedup_sets)),
        ("③ 按电路宏平均（每电路等权）", sets, True, len(by_circ)),
        (f"④ 排除最大电路 {_dom_name}", [s for s in sets if s["circuit"] != _dom_circ],
         False, len(sets) - len(by_circ[_dom_circ])),
    )
    print("\n=== 交付口径对照（17.3.11，同一份快照）===")
    print("  " + _pad("口径", 32) + _pad("单位数", 8) + _pad("严格k2", 9) + _pad("严格k3", 9)
          + _pad("宽松k2", 9) + _pad("宽松k3", 9) + _pad("选择遗憾", 11) + "两阶段遗憾")
    for label, ss, macro, n_units in _spec:
        a2, a3, b2, b3, rg, st = _agg(ss, macro)
        print("  " + _pad(label, 32) + _pad(str(n_units), 8)
              + _pad(f"{a2*100:.1f}%", 9) + _pad(f"{a3*100:.1f}%", 9)
              + _pad(f"{b2*100:.1f}%", 9) + _pad(f"{b3*100:.1f}%", 9)
              + _pad(f"{rg*100:.2f}%", 11) + f"{st*100:.2f}%")
    print("  ⚠ ① 度量的是「平均批」，而批被重复提议主导（见上一块）→ 不是部署总体；"
          "③ 度量的是「平均电路」，部署面对的是电路。两者差多少，就是「重复提议」的定价。")
    print("  ⚠ 宽松 k3 在 n=4 的集上**定义性饱和**（真前3 与预测前3 各占 4 选 3，必然相交 → 恒 100%）"
          "→ 该列含白送分；严格列不受此影响。")

    # —— 以 current 为锚的 GNN 判据质量（18.6.0 新增 / 18.6.1 修正配对校验）——
    # ⚠ 18.7.4 起本节的调用**提到 main 开头**（紧接「无合格候选集」守卫之后）：
    #   理由 = 让 ★★★ 最优先判据（机会把握率）落在第一屏。两条位置约束（见该处注释）都是上界，上提不冲突。

    # —— 两列并排 A/B（仅当 CSV 带 gnn_pred2 列时出现）——
    if ab_sets:
        def col(k, key):
            return [a[key] if k == 0 else b[key] for a, b in ab_sets if (a if k == 0 else b)[key] is not None]

        def paired(key, lower_is_better):
            """同候选集配对胜负（用两列都有效的行集算，严格同分母）。"""
            d = [a[key] - b[key] for a, b in ab_sets
                 if a[key] is not None and b[key] is not None]
            w1 = sum(1 for x in d if (x < 0 if lower_is_better else x > 0))
            w2 = sum(1 for x in d if (x > 0 if lower_is_better else x < 0))
            return w1, w2, len(d) - w1 - w2

        print(f"\n=== 两列并排 A/B（gnn_pred vs gnn_pred2，同一候选集同分母）===")
        print(f"  配对候选集数 = {len(ab_sets)}（主口径 {len(sets)} 集；"
              f"因第二列缺/NaN 落掉 {ab_dropped_sets} 集）  带第二列的行 = {c2_rows}/{ok_rows}")
        if ab_dropped_sets:
            print(f"  ⚠ 落掉的 {ab_dropped_sets} 集**只在 A/B 里不算**，主口径那一节仍是全集 —— "
                  f"两节的分母不同，跨节比较指标时注意")
        print(f"  模型1 = gnn_pred（GNN_PORT，serve 当前默认模式）；模型2 = gnn_pred2（GNN_PORT2）")
        print(f"  {'判据':<26}{'模型1':>10}{'模型2':>10}   {'配对 1胜/2胜/平'}")

        def line(name, key, fmt, lower=None):
            v1 = statistics.mean(col(0, key)); v2 = statistics.mean(col(1, key))
            tail = ""
            if lower is not None:
                w1, w2, t = paired(key, lower)
                tail = f"   {w1:4d}/{w2:4d}/{t:4d}"
            print(f"  {name:<26}{fmt.format(v1):>10}{fmt.format(v2):>10}{tail}")

        line("严格 k=2（#1∈预测前2）", "recall2_strict", "{:.1%}", lower=False)
        line("严格 k=3（#1∈预测前3）", "recall3_strict", "{:.1%}", lower=False)
        line("宽松 k=2", "recall2_len", "{:.1%}", lower=False)
        line("宽松 k=3", "recall3_len", "{:.1%}", lower=False)
        line("选择遗憾（越低越好）", "regret", "{:.2%}", lower=True)
        line("两阶段最终遗憾", "regret_2stage", "{:.2%}", lower=True)
        line("两阶段捕获率（越高越好）", "capture2", "{:.1%}", lower=False)
        line("Spearman（越高越好）", "spearman", "{:.3f}", lower=False)
        ab_hi = [(a, b) for a, b in ab_sets if a["spread_pct"] > 10.0]
        if ab_hi:
            d = [a["regret"] - b["regret"] for a, b in ab_hi]
            w1 = sum(1 for x in d if x < 0); w2 = sum(1 for x in d if x > 0)
            m1 = statistics.mean(a["regret"] for a, _ in ab_hi)
            m2 = statistics.mean(b["regret"] for _, b in ab_hi)
            print(f"  跨度>10% 子集（{len(ab_hi)} 集）遗憾: 模型1 {m1:.2%}  模型2 {m2:.2%}   "
                  f"配对 {w1}/{w2}/{len(d)-w1-w2}")
        print("  ↑ 两模型的 true_delay 列完全相同（GNN 是纯观察者，仿真轨迹与模型无关），"
              "故这里比的是**同一批候选上的排序质量**，差异只来自模型本身")

    # 明细：按遗憾**降序**列出每集（17.3.9：batch 口径下集数可达数百，故默认只列最差的几集）
    srt = sorted(sets, key=lambda x: -x["regret"])
    # 18.7.4：默认**不列**（-1）—— 该块是全文最大的一段，而信息密度最低（只列「遗憾最大的几集」）。
    # 0 仍 = 全列（既有自测 _t_shadow_groupby.py 传的就是 --detail-max 0），N>0 = 列最差的 N 集。
    shown = [] if args.detail_max < 0 else (
        srt if (args.detail_max == 0 or len(srt) <= args.detail_max) else srt[:args.detail_max])
    print("\n=== 每候选集明细（按遗憾降序；#1∈前k=严格, 前k∩真前k=宽松）===")
    if not shown:
        print(f"  （默认不列（本次 {len(srt)} 集）；--detail-max N 列遗憾最大的 N 集，0=全列）")
    elif len(shown) < len(srt):
        print(f"  （只列遗憾最大的 {len(shown)}/{len(srt)} 集；--detail-max 0 可列全部）")
    for s in shown:
        sps = f"{s['spearman']:.2f}" if s["spearman"] is not None else "-"
        wid = (f"it={s['iter']:3d} w={s['window']:2d}" if s["iter"] is not None
               else f"w={s['window']:2d}       ")
        print(f"  {s['circuit']:45s} {wid} n={s['n']:2d} "
              f"#1∈前2={'Y' if s['recall2_strict'] else 'n'} 前2∩真={ 'Y' if s['recall2_len'] else 'n'} "
              f"#1∈前3={'Y' if s['recall3_strict'] else 'n'} 前3∩真={ 'Y' if s['recall3_len'] else 'n'} "
              f"regret={s['regret']*100:7.2f}% 2stage={s['regret_2stage']*100:6.2f}% "
              f"cap2={s['capture2']*100:6.2f}% sp={sps}")

if __name__ == "__main__":
    main()
