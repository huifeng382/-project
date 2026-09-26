#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""_t_shadow_current_anchor.py — 18.7.0「★ 机会集口径 R1/R2」自测（本地 temp 树，**不入库**）。

用法：python scripts/diag/_t_shadow_current_anchor.py
（就地调同目录的 _shadow_analyze.py，只看它新加的那一节；不需要 Xyce / serve / 服务器。

覆盖（对应计划「验证」第 3 条）：
  (a) 手造 6 组小树：用 numpy **独立复算** D3/D4/N1/N2/R1/R2（不复用分析器任何函数），与打印的
      四行逐格对账；并**埋一个「current 被 GNN 排第一、而真值上确有更优候选」的组**，另起一棵
      只含它的树，断言它**恰好**落进 N1 且分母 == 1（分母正确性单独隔离验证）。
  (a2) 18.7.4 ★★★ 最优先判据「机会把握率」两格：同样独立复算 (H3,D3)/(H4,D4) 与打印对账，
      并断言 H4/D4 >= H3/D3（宽松含「首推就是 current 本身」）；两棵隔离树分别钉住
      「两口径都抓住」（⑥ ⇒ 1/1 与 1/1）与「只有真并列」（③ ⇒ 严格 0/0、宽松 1/1）。
  (b) D3=0 的组（真值上没机会 / 只有真值持平）⇒ 单列计数、**不入 R1 分母**（分子分母都在打印里对账）。
      另用 --min-cands 6 把「其中 ≥N 候选」那一行打成空集 ⇒ 断言 R1/R2 打成 NA（除零不崩、不伪装成 0%）。
  (c) rank(cur)=NA、true(cur)=NA、候选 rank=NA、候选 true=NA、候选 tc 与 gnn_shadow 的 transistors
      不等 —— 各造一组 ⇒ 计数如实、不崩，且这棵树**不进** arecs（打「无组」行）。
  (d) **负例**：把一个候选的 eval_idx 挪一位 ⇒ 必须报「配对/覆盖(pos 集非全集) 1」并把这组剔出，
      而不是静默算一个数（这正是本块另立更严校验的理由）。
  (e) 删 gnn_current.csv ⇒ 本节**只多那一行常量跳过语**（连带把 18.6.1 的三个既有自测的
      「常量行」约定一起钉住）。

纪律（本文件是 _t_ 前缀的临时自测，不入库）：
  · 所有断言都来自**独立复算**或**打印文本的显式解析**，不看分析器内部变量。
  · 打印纪律（_t_shadow_groupby.py:127 的 startswith 认表 / 既有 grep -m1 标签）不在本文件职责内，
    那是既有三个自测的事；本文件只保证新增块本身自洽。
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

# 17.3.9 同款：本文件有 ✅/❌，Windows 默认 GBK 会 UnicodeEncodeError，先钉 utf-8
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYZE = os.path.join(HERE, "_shadow_analyze.py")
MIN_C = 4                    # 既有「对齐口径」默认值（_shadow_analyze.py:887）
CUR_HDR = "iter,window,eval_idx,pos,id,rank,is_current,true_avg,tc,nout"
NT_DEF = 12                  # gnn_shadow 的 transistors 默认值；tc 与之相等才算「配对得上」
# 无 gnn_current.csv 时该节的**唯一**输出（_shadow_analyze.py:470；常量行，逐字节钉住）
SKIP_LINE = "（本树无 gnn_current.csv，跳过：本树不是 18.6.0 之后的 shadow 跑的）"
fails = []


def check(ok, msg):
    print(("  ✅ " if ok else "  ❌ ") + msg)
    if not ok:
        fails.append(msg)


# ────────────────────────── fixture：手造小树 ──────────────────────────
# 组 = 一个窗口的「current + 全部衍生电路」；cur=(rank, true, nout)；cands=[(pos, rank, true)]
# 每组独占一段 eval 编号（base = 1 + 10*k）⇒ 与 tl_opt 的 `eval_idx == base+1+pos` 不变量同型，
# 且各窗编号不交叉（窗内重复行才会触发 n_dup_win，跨窗重复只会脏 n_dup_eval）。
G = [
    # ① 真值上**有**更优候选，但 GNN 把某个候选排到了 current 前面 ⇒ cur 没被排第一
    #    ⇒ 计入 D3/D4 分母，**不计入** N1/N2。
    dict(tag="①cur未被排第一", cur=(5.0, 100.0, 1),
         cands=[(0, 2.0, 90.0), (1, 1.0, 110.0), (2, 3.0, 100.0), (3, 4.0, 120.0)]),
    # ② 【正例】GNN 把 current 排第一（rank=1），而真值上**确有**更优候选（95 < 100）
    #    ⇒ 必须落进 N1（严格）与 N2（宽松）。
    dict(tag="②cur第一但真有更优", cur=(1.0, 100.0, 1),
         cands=[(0, 2.0, 95.0), (1, 3.0, 105.0), (2, 4.0, 130.0), (3, 5.0, 140.0)]),
    # ③ 只有**真值持平**（无更优）⇒ D3=0 而 D4=1：正好是「D4−D3」那一类；cur 排第一也只进 N2。
    dict(tag="③只有真值持平", cur=(1.0, 100.0, 1),
         cands=[(0, 2.0, 100.0), (1, 3.0, 110.0), (2, 4.0, 120.0), (3, 5.0, 130.0)]),
    # ④ 真值上**毫无机会**（全部更差）⇒ D3=D4=0：不入任何分母；cur 排第一也无过。
    dict(tag="④无机会", cur=(1.0, 100.0, 1),
         cands=[(0, 2.0, 110.0), (1, 3.0, 120.0), (2, 4.0, 130.0), (3, 5.0, 140.0)]),
    # ⑤ **秩并列**（候选与 current 同秩 1）+ 真值上更优 ⇒ cur_first 仍成立（competition ranking：
    #    没有候选秩**严格小于** current）⇒ 进 N1；且是「同秩」计数器的唯一命中组。
    dict(tag="⑤秩并列", cur=(1.0, 100.0, 1),
         cands=[(0, 1.0, 98.0), (1, 3.0, 105.0), (2, 4.0, 120.0), (3, 5.0, 130.0)]),
    # ⑥ 只有 2 个候选（< min_cands=4 ⇒ 被「对齐既有口径」那行截掉，只在「全部组」里）+ nout=2
    #    ⇒ 单输出/多输出分层的唯一多输出组。
    dict(tag="⑥两候选+多输出", cur=(3.0, 100.0, 2),
         cands=[(0, 1.0, 80.0), (1, 2.0, 200.0)]),
]

# (c) 各种「真值/秩不全」的组：每类各一组 ⇒ 四个计数各 +1（tc 那组算进 true_na 一类）
G_NA = [
    dict(tag="cur秩NA", cur=(None, 100.0, 1),
         cands=[(0, 2.0, 90.0), (1, 1.0, 110.0), (2, 3.0, 100.0), (3, 4.0, 120.0)]),
    dict(tag="cur真值NA", cur=(1.0, None, 1),
         cands=[(0, 2.0, 90.0), (1, 1.0, 110.0), (2, 3.0, 100.0), (3, 4.0, 120.0)]),
    dict(tag="候选秩NA", cur=(1.0, 100.0, 1),
         cands=[(0, 2.0, 90.0), (1, None, 110.0), (2, 3.0, 100.0), (3, 4.0, 120.0)]),
    dict(tag="候选真值NA", cur=(1.0, 100.0, 1),
         cands=[(0, 2.0, 90.0), (1, 3.0, 110.0), (2, 3.0, None), (3, 4.0, 120.0)]),
    # tc 不等只改 **gnn_current** 那一侧（shadow 的 transistors 保持真实值）：两文件对同一
    # eval_idx 的电路说法不一致 = 错配 ⇒ 与「真值缺」同类处理。
    dict(tag="候选tc不等", cur=(1.0, 100.0, 1),
         cands=[(0, 2.0, 90.0), (1, 3.0, 110.0), (2, 3.0, 100.0), (3, 4.0, 120.0)], tc_cur_bad_pos=3),
]


def fnum(v, sci=False):
    if v is None:
        return "NA"
    return f"{v:.6e}" if sci else f"{v:g}"


def write_tree(root, groups, no_current=False):
    """写 root/<A>/<B>/gnn_{shadow,current}.csv（A/B 两层是因为分析器按 `*/*/` glob）。
    ⚠ 不写初始 current 那次 eval（gnn_shadow 无「未认领窗口」行）⇒ 让诊断里的
      `gnn_shadow 有、gnn_current 无的窗口` 保持 0，把断言面缩小到本块自己的计数。"""
    d = os.path.join(root, "tree0", "circ_fixture")
    os.makedirs(d, exist_ok=True)
    shp = os.path.join(d, "gnn_shadow.csv")
    cup = os.path.join(d, "gnn_current.csv")
    with open(shp, "w", encoding="utf-8") as fs, open(cup, "w", encoding="utf-8") as fc:
        fc.write(CUR_HDR + "\n")
        for k, g in enumerate(groups):
            base = 1 + 10 * k
            it = k + 1
            cr, ct, nout = g["cur"]
            for (pos, rank, true) in g["cands"]:
                ev = base + 1 + pos
                ev_cur = ev + (1 if g.get("shift_cur_pos") == pos else 0)   # (d) 只挪 gnn_current 那侧
                tc = NT_DEF                                    # shadow 侧：真实晶体管数
                tc_cur = 999 if g.get("tc_cur_bad_pos") == pos else NT_DEF   # current 侧：(c) 才故意不等
                # ⚠ gnn_pred 恒写**有限数**：gnn_shadow.csv 里没有秩列（秩只落在 gnn_current.csv），
                #    而 CSV_RE 不接受 `gnn_pred=NA`（只有 true_delay 允许 NA）⇒ 写 NA 会让整行读不出，
                #    连带把「候选秩 NA」误报成「配对失败」（首版自测就是这么红出来的）。
                gs = fnum((true if true is not None else 1.0) * 1e-13, True)
                fs.write(f"eval_idx={ev}, iter={it}, window=0, gnn_pred={gs}, "
                         f"true_delay={fnum(true, True)}, transistors={tc}\n")
                fc.write(f"{it},0,{ev_cur},{pos},cand{pos},{fnum(rank)},{0},{fnum(true, True)},{tc_cur},{nout}\n")
            if not no_current:
                fc.write(f"{it},0,{base},NA,CUR,{fnum(cr)},{1},{fnum(ct, True)},NA,{nout}\n")
    if no_current:
        os.remove(cup)
    return root


def run_analyze(root, min_c=MIN_C):
    p = subprocess.run([sys.executable, ANALYZE, "--root", root, "--min-cands", str(min_c),
                        "--detail-max", "0"],
                       capture_output=True, encoding="utf-8", errors="replace")
    return p


# ────────────────────────── 独立复算（numpy，零复用） ──────────────────────────
def indep(groups):
    """从 fixture 原始数据独立复算每组的判据。与分析器同**语义**、不同**实现**：
    cur_first ⟺ 没有候选秩严格小于 current（competition ranking，并列算首推）；
    d3 ⟺ ∃ 候选真值严格更小；d4 ⟺ ∃ 候选真值 ≤ current 真值。
    18.7.4 加 h3/h4（机会把握率）：首推 = 秩最小的候选，且**只有** min(候选秩) < cur 秩时才算
    「首推候选」（否则首推就是 current）；并列首推要求**整撮都满足**。
      h3（严格）= 首推候选且全部 t₁ < t_cur；h4（宽松）= 首推就是 current 本身，或全部 t₁ ≤ t_cur。"""
    out = []
    for g in groups:
        cr, ct, nout = g["cur"]
        if cr is None or ct is None or any(c[1] is None or c[2] is None for c in g["cands"]):
            continue                      # (c) 那类：与「真值不全」同待遇，不进分母池
        rk = np.array([cr] + [c[1] for c in g["cands"]], dtype=float)
        tt = np.array([ct] + [c[2] for c in g["cands"]], dtype=float)
        rmin = rk[1:].min()                       # 候选秩的最小值（cur 不在内）
        cand_first = bool(rmin < rk[0])           # 存在候选严格排在 current 之前
        picks = tt[1:][rk[1:] == rmin] if cand_first else np.array([])
        out.append({
            "tag": g["tag"], "nout": nout, "n_cands": len(g["cands"]),
            "cur_first": bool((rk[1:] >= rk[0]).all()),
            "d3": bool((tt[1:] < tt[0]).any()),
            "d4": bool((tt[1:] <= tt[0]).any()),
            "h3": bool(cand_first and (picks < tt[0]).all()),
            "h4": bool((not cand_first) or (picks <= tt[0]).all()),
            "tie_rank": bool((rk[1:] == rk[0]).any()),
            "tie_true": bool((tt[1:] == tt[0]).any()),
        })
    return out


def pct(a, b):
    return f"{a / b * 100:.2f}%" if b else "NA"


def agg(rows):
    d3 = sum(1 for r in rows if r["d3"]); d4 = sum(1 for r in rows if r["d4"])
    n1 = sum(1 for r in rows if r["d3"] and r["cur_first"])
    n2 = sum(1 for r in rows if r["d4"] and r["cur_first"])
    return (str(len(rows)), str(d3), str(d4), str(n1), str(n2), pct(n1, d3), pct(n2, d4))


# ────────────────────────── 打印文本解析 ──────────────────────────
ROW_KEYS = (("全部组", "all"), ("其中 ≥", "k4"), ("单输出层", "single"), ("多输出层", "multi"))


def parse_rows(out):
    got = {}
    for line in out.splitlines():
        for lab, key in ROW_KEYS:
            if line.strip().startswith(lab):
                got[key] = tuple(line.split()[-7:])    # 组数 D3 D4 N1 N2 R1 R2
    return got


OPP_LABELS = ("配对/覆盖(pos 集非全集)", "current 秩或真值缺", "候选秩缺", "候选真值缺/tc 不等")

# 18.7.4 ★★★ 最优先判据「机会把握率」两行：`… :  25.92%   = 239/922 组`
# ⚠ 分子/分母从**行尾的 a/b**取（`组`前面那个分数），不从百分号取 —— 百分号在分母为 0 时打 NA，
#   而 a/b 恒在（0/0），这样「分母 0 也得报 0/0」这件事本身也被钉住。
CORE_RE = re.compile(r"=\s*(\d+)/(\d+)\s*组")


def parse_core(out):
    """→ ((H3, D3), (H4, D4))；两行任一缺失返回 None（用于断言「必须打出来」）。"""
    got = {}
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("严格（GNN 第一名"):
            m = CORE_RE.search(s)
            got["s"] = (int(m.group(1)), int(m.group(2))) if m else None
        elif s.startswith("宽松（GNN 第一名"):
            m = CORE_RE.search(s)
            got["l"] = (int(m.group(1)), int(m.group(2))) if m else None
    return (got.get("s"), got.get("l")) if len(got) == 2 else None


def parse_opp_diag(out):
    pat = "|".join(re.escape(x) for x in OPP_LABELS)
    for line in out.splitlines():
        if "机会集口径（18.7.0）前置校验失败" in line:
            return dict(re.findall(rf"({pat}) (\d+)", line)), ("✅" in line)
    return None, None


def show(p, tail=26):
    print("    ---- 分析器 rc=%d stderr=%s ----" % (p.returncode, (p.stderr or "").strip()[:200]))
    for l in (p.stdout or "").splitlines()[-tail:]:
        print("    | " + l)


def section_of(out):
    """取「以 current 为锚」那一节的**正文**（丢掉标题行的剩余部分；(e) 断言只剩一行常量跳过语）。"""
    k = "=== 以 current 为锚的 GNN 判据质量"
    if k not in out:
        return "<<无该节>>"
    tail = out.split(k, 1)[1]
    return tail.split("\n", 1)[1] if "\n" in tail else ""


# ────────────────────────── 主流程 ──────────────────────────
def main():
    root = tempfile.mkdtemp(prefix="t_opp_")
    try:
        # ===== 树 T1：6 组 =====
        write_tree(os.path.join(root, "t1"), G)
        p = run_analyze(os.path.join(root, "t1"))
        out = p.stdout or ""
        if p.returncode != 0 or "★ 机会集口径 R1/R2（" not in out:
            show(p, 40)
            check(False, "T1 跑通并打出「★ 机会集口径 R1/R2」块")
            return
        ind = indep(G)
        got = parse_rows(out)
        exp_all = agg(ind)
        exp_k4 = agg([r for r in ind if r["n_cands"] >= MIN_C])
        exp_s1 = agg([r for r in ind if r["nout"] == 1])
        exp_m2 = agg([r for r in ind if r["nout"] >= 2])
        print("  (a) 独立复算 vs 打印（组数/D3/D4/N1/N2/R1严格/R2宽松）：")
        for key, name, exp in (("all", "全部组(min_cands=1)", exp_all),
                               ("k4", f"其中 ≥{MIN_C} 候选", exp_k4),
                               ("single", "单输出层(nout=1)", exp_s1),
                               ("multi", "多输出层(nout>=2)", exp_m2)):
            check(got.get(key) == exp, f"{name}: 打印 {got.get(key)} == 独立复算 {exp}")

        # (a) 正例隔离：只留 ② 一棵树 ⇒ N1 必须恰好是 1、分母恰好是 1、R1 == 100%
        write_tree(os.path.join(root, "t3"), [G[1]])
        p3 = run_analyze(os.path.join(root, "t3"))
        g3 = parse_rows(p3.stdout or "").get("all")
        check(g3 == ("1", "1", "1", "1", "1", "100.00%", "100.00%"),
              f"(a) 隔离正例：只有「cur 排第一却真有更优候选」那一组时 N1/D3 == 1/1、R1 == 100%"
              f"（打印 {g3}）")
        # 反向隔离：只留 ①（同样有更优候选，但 cur 没被排第一）⇒ N1 必须为 0（区别只在秩）
        write_tree(os.path.join(root, "t3b"), [G[0]])
        g3b = parse_rows(run_analyze(os.path.join(root, "t3b")).stdout or "").get("all")
        check(g3b == ("1", "1", "1", "0", "0", "0.00%", "0.00%"),
              f"(a) 反向隔离：① 组同样有更优候选但 cur 未排第一 ⇒ N1 = N2 = 0（打印 {g3b}）")

        # (a2) 18.7.4 ★★★ 最优先判据「机会把握率」：打印的 (H3,D3)/(H4,D4) 与独立复算逐格对账。
        #      两行**必须存在**（parse_core 返回 None 即失败）——这是「最优先判据要专门打出来」的钉子。
        #      ⚠ 分子**必须限定在各自的分母里**数：④ 组（无机会：候选真值全都更差）在宽松谓词下
        #      h4 也为真（首推就是 current）——不限定就会数出一个 > 分母的分子（本例 5/4）。
        exp_s = (sum(1 for r in ind if r["h3"] and r["d3"]), sum(1 for r in ind if r["d3"]))
        exp_l = (sum(1 for r in ind if r["h4"] and r["d4"]), sum(1 for r in ind if r["d4"]))
        raw_l = sum(1 for r in ind if r["h4"])
        core = parse_core(out)
        check(core == (exp_s, exp_l),
              f"(a2) ★★★ 机会把握率：打印 {core} == 独立复算 严格{exp_s}/宽松{exp_l}")
        check(raw_l > exp_l[0] and exp_l[0] <= exp_l[1],
              f"(a2) 分母外溢防线：不限定分母的宽松分子是 {raw_l} > H4 {exp_l[0]}（④ 组该类）"
              f"⇒ 打印取的是**限定后**的 {exp_l[0]}，恒 <= D4 {exp_l[1]}")
        # 宽松分子含「首推就是 current 本身」⇒ 恒有 H4 ≥ H3、D4 ≥ D3（这正是宽松率可能**更低**的来源）
        check(exp_l[0] >= exp_s[0] and exp_l[1] >= exp_s[1],
              f"(a2) 单调性：H4/D4 = {exp_l} >= H3/D3 = {exp_s}（宽松含「首推=current」那类）")
        # 隔离正例：⑥ 组 = 唯一一组两口径都抓住（首推是候选、且真值严格更优）⇒ 严格 1/1、宽松 1/1
        #   ⚠ ⑥ 只有 2 个候选 ⇒ 必须 min_c=2：`sets`（= 逐集明细那条路）是按 min_cands 过滤的，
        #     min_c=4 时这棵树连 sets 都建不起来，报告在 main 顶部就早退了（本节根本不打印）。
        write_tree(os.path.join(root, "t3c"), [G[5]])
        c3c = parse_core(run_analyze(os.path.join(root, "t3c"), min_c=2).stdout or "")
        check(c3c == ((1, 1), (1, 1)),
              f"(a2) 隔离：⑥ 组（首推候选且真改进）⇒ 严格 1/1、宽松 1/1（打印 {c3c}）")
        # 隔离：③ 组只有**真并列**（没有更优）⇒ 严格分母 0（打 0/0 而非 NA 以外的任何数）、宽松 1/1
        #   ⚠ 这一棵同时把「分母 0 时分子分母仍要如实打 a/b」这条钉住（百分号那格才是 NA）
        write_tree(os.path.join(root, "t3d"), [G[2]])
        c3d = parse_core(run_analyze(os.path.join(root, "t3d")).stdout or "")
        check(c3d == ((0, 0), (1, 1)),
              f"(a2) 隔离：③ 组只有真并列 ⇒ 严格分母 0（打 0/0）、宽松 1/1（打印 {c3d}）")

        # (b) 分母为 0 的组单列、不入比值；且「≥N 候选」行是空集时打 NA 而不是 0%
        n_d3_0 = sum(1 for r in ind if not r["d3"])
        n_d4_0 = sum(1 for r in ind if not r["d4"])
        line = next((l for l in out.splitlines() if "分母为 0 的组不入比值" in l), "")
        want = (f"D3=0 的组 {n_d3_0} 个", f"D4=0 的组 {n_d4_0} 个",
                f"D4−D3 = {sum(1 for r in ind if r['d4']) - sum(1 for r in ind if r['d3'])} 个组只有真并列的候选")
        check(all(w in line for w in want), f"(b) 分母 0 的组单列且不入均值：{want} 都在（行= {line.strip()[:90]}…）")
        # (b2) 分母为 0 的**空层**渲染成 NA 而不是 0.00%（真实可达路径：全单输出的树 ⇒ 多输出层空）
        #      ⚠ 不能用「--min-cands 大到该行空集」来造：sets 一空，整节报告会在
        #      _shadow_analyze.py:993 早退（连本节都不打印），那就测不到渲染分支了。
        write_tree(os.path.join(root, "t7"), [G[1], G[2]])       # 两组都 nout=1
        g7 = parse_rows(run_analyze(os.path.join(root, "t7")).stdout or "")
        check(g7.get("multi") == ("0", "0", "0", "0", "0", "NA", "NA"),
              f"(b) 多输出层空集 ⇒ 组数 0 且 R1/R2 打 NA（不是 0.00%）（打印 {g7.get('multi')}）")
        check(g7.get("single") == agg(indep([G[1], G[2]])),
              f"(b) 同一树单输出层照常报数（打印 {g7.get('single')}）")
        # (b3) min_cands 只影响「对齐既有口径」那行，不动「全部组」（两个分母并排的前提）
        p2 = run_analyze(os.path.join(root, "t1"), min_c=2)
        g2c = parse_rows(p2.stdout or "")
        check(g2c.get("k4") == exp_all and g2c.get("all") == exp_all,
              f"(b) --min-cands 2 时「其中 ≥2」== 「全部组」（打印 {g2c.get('k4')}）"
              f"，且「全部组」与 min_cands=4 时逐格相同")

        # 同秩 / 真值并列两个伴随计数
        tl = next((l for l in out.splitlines() if "current 与某候选同秩" in l), "")
        exp_tr = sum(1 for r in ind if r["tie_rank"]); exp_tt = sum(1 for r in ind if r["tie_true"])
        check(f"同秩（并列第 1）的组 {exp_tr} 个" in tl and f"真值相等**的候选的组 {exp_tt} 个" in tl,
              f"(a) 伴随计数：同秩 {exp_tr} 组、真值并列 {exp_tt} 组（行= {tl.strip()[:100]}…）")
        check(not ind[0]["tie_rank"],
              "(a) 「同秩」只数真并列：① 组里有候选秩**严格优于** current（1 < 5）但秩不等 ⇒ 不算同秩")

        # (d) 负例：把 ① 组 pos=3 候选在 **gnn_current.csv** 里的 eval_idx 挪一位（gnn_shadow 不动）
        #      ⇒ 该组的秩↔真值 join 断掉，必须报「配对失败 1」并剔出分母池，而不是静默算一个数。
        t4 = [dict(x) for x in G]
        t4[0] = dict(G[0], shift_cur_pos=3)
        write_tree(os.path.join(root, "t4"), t4)
        p4 = run_analyze(os.path.join(root, "t4"))
        cnt4, ok4 = parse_opp_diag(p4.stdout or "")
        if not (cnt4 is not None and cnt4.get("配对/覆盖(pos 集非全集)") == "1" and ok4 is False):
            show(p4, 34)
        check(cnt4 is not None and cnt4.get("配对/覆盖(pos 集非全集)") == "1" and ok4 is False,
              f"(d) 负例：eval_idx 错位 ⇒ 必须报「配对/覆盖(pos 集非全集) 1」且打 ❌"
              f"（实得 {cnt4} ✅={ok4}）")
        check(parse_rows(p4.stdout or "").get("all", ("",))[0] == str(len(ind) - 1),
              f"(d) 负例：该组被剔出分母池 ⇒ 组数 {len(ind)} → {len(ind)-1}"
              f"（打印 {parse_rows(p4.stdout or '').get('all')}）")
        cnt1, ok1 = parse_opp_diag(out)
        check(cnt1 == {k: "0" for k in OPP_LABELS} and ok1 is True,
              f"(a) 正树前置校验四项全 0 且打 ✅（实得 {cnt1} ✅={ok1}）")

        # (c) 秩/真值不全：计数如实、不进 arecs、不崩
        write_tree(os.path.join(root, "t5"), G_NA)
        p5 = run_analyze(os.path.join(root, "t5"))
        out5 = p5.stdout or ""
        cnt5, ok5 = parse_opp_diag(out5)
        want5 = {"配对/覆盖(pos 集非全集)": "0", "current 秩或真值缺": "2",
                 "候选秩缺": "1", "候选真值缺/tc 不等": "2"}
        if not (p5.returncode == 0 and cnt5 == want5 and ok5 is False):
            show(p5, 34)
        check(p5.returncode == 0 and cnt5 == want5 and ok5 is False,
              f"(c) 秩/真值缺 + tc 不等：计数 {want5}、打 ❌、不崩（实得 {cnt5} ✅={ok5} rc={p5.returncode}）")
        check("★ 机会集口径 R1/R2: 无「真值全到位」的组" in out5,
              "(c) 该树不进分母池 ⇒ 打「无组」行而不是硬凑一个比值")

        # (e) 无 gnn_current.csv ⇒ 该节只剩一行常量跳过语
        write_tree(os.path.join(root, "t6"), G, no_current=True)
        p6b = run_analyze(os.path.join(root, "t6"))
        rest = section_of(p6b.stdout or "").splitlines()
        check(rest and rest[0].strip() == SKIP_LINE and not any("机会集" in l for l in rest),
              f"(e) 无 gnn_current.csv ⇒ 该节只剩那一行常量跳过语（实得首行 {rest[0].strip()[:60]!r}；"
              f"节内其余行提及「机会集」{sum(1 for l in rest if '机会集' in l)} 次）")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print()
    if fails:
        print(f"❌ 失败 {len(fails)} 项：")
        for f in fails:
            print("   - " + f)
        return 1
    print("✅ 全部断言通过（机会集口径 R1/R2：独立复算对账 + 正/反向隔离 + 分母 0 + 秩真值缺 + 错位负例 + 常量行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
