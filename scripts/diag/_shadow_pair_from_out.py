"""_shadow_pair_from_out.py — 从历史 `~/sweep_*.out` 的**逐集明细行**做跨 ckpt 配对比较。

## 为什么存在

`run_shadow_batch.sh:30` 每趟跑前 `rm -rf temp_sim_test/tl_opt_batch`，而 CSV 路径固定跨
run 复用 → **历史各趟的 CSV 树已被逐趟销毁**（2026-09-16 实查：服务器上只剩两份 = 最后一趟的
活树 + 一份逐字节归档 `~/tl_opt_batch_keep_0916`，内容指纹相同）。所以「零仿真重算历史各臂」
这条路对 CSV 是不通的。

**但 15 份 `~/sweep_*.out` 都还在，且每份含全部 106 集的逐集明细行**（不是默认
`--detail-max 30` 的 30 条；该批输出实际列了全量）→ **跨 epoch 的配对可以在零重算下做**。
这正是 DIFF §13.7.4-2 留下的话头：「要精确翻转数：`~/sweep_v2nowave42b_ep*.out` 里各集的
明细行可逐集比对。」

## 为什么配对很重要

106 集 window 口径下，严格@3 的 1 SE ≈ 5pp、1 集翻转 = 0.94pp（DIFF §13.7.4-2）。
拿两个**均值**相减，任何 <5pp 的 epoch 差异都判不出来。配对把「集间方差」整个消掉
（同一批集、同一批真值、同一份 Rust 轨迹），只看逐集翻转 → 灵敏度高一个量级。

## ⚠ 口径（读之前必须知道）

这批 .out 产自 **17.3.9 之前** → 是 **106 集 `window` 池化口径**（组键 = `(电路, w)`）。
池化会让「预测前 3」带并列 artifact（大量候选在池化后秩相同，`argsort(kind="stable")`
落到最早几批的各自第 1 名），而**该 artifact 随模型变**（不同 ckpt 的秩不同 → 并列结构
不同）→ 配对时它是一份**额外噪声，不是共模**。

所以本脚本给的是「比两个均值灵敏得多、但仍带 window 口径 artifact」的读数 ——
**不能替代 batch 口径重扫**，两者要一起看。

## 用法

    # 1) 先量噪声底：同 ckpt、两个进程
    python3 _shadow_pair_from_out.py --ref ~/sweep_v2nowave42m4repA_ep250.out \
                                     --against ~/sweep_v2nowave42m4repB_ep250.out

    # 2) 逐点对基线：42m4 六点全比 ep250
    python3 _shadow_pair_from_out.py --ref ~/sweep_v2nowave42m4_ep250.out \
            --against ~/sweep_v2nowave42m4_ep{50,100,150,200,300}.out

    # 3) 单份的 n 直方图 + 随机基线（口径讨论用）
    python3 _shadow_pair_from_out.py --nhist ~/sweep_v2nowave42b_ep100.out

只读，不写任何文件。输出用 `sys.stdout.reconfigure` 保证中文在重定向下不乱码。
"""
import argparse
import math
import re
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')   # 报错走 stderr；Windows 默认 cp936 会把中文打成乱码

# 逐集明细行（`_shadow_analyze.py` 的 :660-664 打印格式）。
# window 口径渲染成 `w= NN       `（无 it=）；batch 口径渲染成 `it=  N w= NN`。
# 本脚本同时吃两种，但历史 .out 都是前者。
# ⚠ `cap2=` 必须是**可选**段：现存 .out 是 17.3.9 之前的分析器产的（`2stage=…% sp=…`），
#   而 `_shadow_analyze.py` 现行版本在两者之间多印了 `cap2=…%`。若写成必需段，
#   老档会**整行不匹配 → 静默丢集**，而分母看着还挺正常 —— 配对分析最致命的错法。
DETAIL_RE = re.compile(
    r"^\s+(?P<circuit>\S+)\s+"
    r"(?:it=\s*(?P<it>\d+)\s+)?"
    r"w=\s*(?P<w>\d+)\s+"
    r"n=\s*(?P<n>\d+)\s+"
    r"#1∈前2=(?P<s2>[Yn])\s+前2∩真=(?P<l2>[Yn])\s+"
    r"#1∈前3=(?P<s3>[Yn])\s+前3∩真=(?P<l3>[Yn])\s+"
    r"regret=\s*(?P<regret>[\d.]+)%\s+"
    r"2stage=\s*(?P<stage>[\d.]+)%\s+"
    r"(?:cap2=\s*(?P<cap2>[\d.]+)%\s+)?"
    r"sp=(?P<sp>[\d.]+|-)"
)
# 「长得像明细行但没解析出来」的兜底探测：漏解析即报错而非跳过。
# ⚠ 必须锚 **恰好两空格缩进**：明细行是 `print(f"  {circuit:45s} …")`；
#   而明细区标题 `=== 每候选集明细（… #1∈前k=严格 …）===` 顶格、也含 `#1∈前`，
#   只按关键字判会**误报整档**（本题第一次自检就栽在这）。
DETAIL_SMELL = re.compile(r"^ {2}\S.*#1∈前")
META_RE = re.compile(
    r"候选集数（[^）]*）[：:]\s*(?P<sets>\d+)\s+成功行=(?P<ok>\d+)\s+失败行=(?P<fail>\d+)\s+小集=(?P<small>\d+)"
)


def parse(path):
    """→ (meta, {(circuit, w): rec})。**任何丢集都报错**，不静默降级。"""
    meta, recs = {}, {}
    dup = 0
    bad = []
    with open(path, encoding='utf-8', errors='replace') as f:
        for i, ln in enumerate(f, 1):
            if not meta:
                m = META_RE.search(ln)
                if m:
                    meta = {k: int(v) for k, v in m.groupdict().items()}
                    continue
            m = DETAIL_RE.match(ln)
            if m:
                key = (m.group('circuit'), int(m.group('w')))
                if key in recs:
                    dup += 1
                recs[key] = dict(
                    n=int(m.group('n')),
                    s2=m.group('s2') == 'Y', l2=m.group('l2') == 'Y',
                    s3=m.group('s3') == 'Y', l3=m.group('l3') == 'Y',
                    regret=float(m.group('regret')) / 100.0,
                    stage=float(m.group('stage')) / 100.0,
                    cap2=None if m.group('cap2') is None else float(m.group('cap2')) / 100.0,
                    sp=None if m.group('sp') == '-' else float(m.group('sp')),
                )
            elif DETAIL_SMELL.search(ln):
                bad.append((i, ln.rstrip()))
    if dup:
        raise SystemExit(f"{path}: 同一 (电路,w) 出现 {dup} 次 —— 该档不是单趟数据，读数不可用")
    if bad:
        # 解析器旧/新格式不一致时**在这里炸**，而不是悄悄少几集。
        raise SystemExit(
            f"{path}: {len(bad)} 行像明细行但未解析（格式与 DETAIL_RE 不符）—— 拒绝静默丢集。"
            f"\n  首行 line {bad[0][0]}: {bad[0][1][:150]}")
    if meta and recs and meta.get('sets') != len(recs):
        print(f"  ⚠ {path.split('/')[-1]}: 摘要称候选集 {meta['sets']}，实解析 {len(recs)} 行"
              f" —— 明细可能被 --detail-max 截断过，配对基数以下面的交集为准")
    return meta, recs


def base_strict3(n):
    """随机排序下「真第1名落进预测前3」的概率 = 3/n。"""
    return min(1.0, 3.0 / n)


def base_loose3(n):
    """随机排序下「预测前3 ∩ 真前3 ≠ ∅」的概率 = 1 − C(n−3,3)/C(n,3)（n≤5 恒 1）。"""
    if n <= 5:
        return 1.0
    return 1.0 - math.comb(n - 3, 3) / math.comb(n, 3)


def binom_two_sided(b, c):
    """符号检验 / McNemar 精确双侧 p：H0 = 两个方向等概率。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def boot_ci(d, n_boot=20000, seed=0):
    """配对差均值的 bootstrap 95% CI（重采样「集」，不是重采样行）。"""
    arr = np.asarray(d, float)
    if arr.size == 0:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def nhist(recs, label):
    ns = sorted(r['n'] for r in recs.values())
    print(f"\n=== n 直方图（{label}，{len(ns)} 集）===")
    counts = {}
    for n in ns:
        counts[n] = counts.get(n, 0) + 1
    for n in sorted(counts):
        bar = '#' * min(60, counts[n])
        print(f"  n={n:3d}  {counts[n]:4d} 集  {bar}")
    bs = [base_strict3(n) for n in ns]
    bl = [base_loose3(n) for n in ns]
    print(f"  集大小: min={min(ns)} 中位={int(np.median(ns))} max={max(ns)}  均值={np.mean(ns):.1f}")
    print(f"  随机基线（按本直方图加权）: 严格@3 = {np.mean(bs):6.2%}   宽松@3 = {np.mean(bl):6.2%}")
    print("  ⚠ n≤5 的集「宽松@3」恒 100%（真前3∩预测前3 必然非空）→ 该档不携带信息，"
          "上面这个加权基线已把它的贡献算进去")


def compare(ref_path, tgt_path, topk):
    m_ref, ref = parse(ref_path)
    m_tgt, tgt = parse(tgt_path)
    nr, nt = len(ref), len(tgt)
    print(f"\n{'=' * 78}")
    print(f"  A = {ref_path.split('/')[-1]}   ({nr} 集)")
    print(f"  B = {tgt_path.split('/')[-1]}   ({nt} 集)")
    if m_ref != m_tgt:
        print(f"  ⚠ 分母不同: A={m_ref} B={m_tgt} —— 跨档比要留神")
    print('=' * 78)

    # —— 完整性闸门 ——
    only_a = sorted(set(ref) - set(tgt))
    only_b = sorted(set(tgt) - set(ref))
    common = sorted(set(ref) & set(tgt))
    if only_a or only_b:
        print(f"  ❌ 配对键不齐: 只在 A {len(only_a)} 个 / 只在 B {len(only_b)} 个")
        for k in (only_a[:3] + only_b[:3]):
            print(f"       {k}")
        print("     → 两档不是同一批集，下面的读数无意义。停止。")
        return
    n_mismatch = [(k, ref[k]['n'], tgt[k]['n']) for k in common if ref[k]['n'] != tgt[k]['n']]
    if n_mismatch:
        print(f"  ❌ 配对键齐但 n 不同: {len(n_mismatch)} 集（候选池变了 → 不是同一批）")
        for k, a, b in n_mismatch[:5]:
            print(f"       {k}  A.n={a}  B.n={b}")
        print("     → 停止。")
        return
    print(f"  ✅ 闸门通过: 配对键 {len(common)} 集完全对齐，n 逐集相同")

    # —— 选择遗憾（连续量，配对差）——
    d = [tgt[k]['regret'] - ref[k]['regret'] for k in common]
    win = sum(1 for x in d if x < 0)      # B 更小 = B 更好
    loss = sum(1 for x in d if x > 0)
    tie = len(d) - win - loss
    lo, hi = boot_ci(d)
    print(f"\n  ── 选择遗憾（越低越好；Δ = B − A，负 = B 更好）──")
    print(f"  A 均值 {np.mean([ref[k]['regret'] for k in common]):7.3%}"
          f"   中位 {np.median([ref[k]['regret'] for k in common]):7.3%}")
    print(f"  B 均值 {np.mean([tgt[k]['regret'] for k in common]):7.3%}"
          f"   中位 {np.median([tgt[k]['regret'] for k in common]):7.3%}")
    print(f"  Δ 均值 {np.mean(d):+7.3%}   中位 {np.median(d):+7.3%}"
          f"   bootstrap 95% CI [{lo:+.3%}, {hi:+.3%}]")
    print(f"  逐集: B 更好 {win} / A 更好 {loss} / 打平 {tie}"
          f"   （符号检验双侧 p = {binom_two_sided(win, loss):.4f}）")
    print(f"  ⚠ 逐集遗憾按 0.01% 打印 → 低于该粒度的真实差看不出（会记成打平）")

    # —— 严格 / 宽松 recall（逐集 0/1 翻转）——
    for k, tag in (('s3', '严格@3'), ('l3', '宽松@3'), ('s2', '严格@2'), ('l2', '宽松@2')):
        a = [ref[x][k] for x in common]
        b = [tgt[x][k] for x in common]
        up = sum(1 for x, y in zip(a, b) if y and not x)     # B 命中、A 没命中
        dn = sum(1 for x, y in zip(a, b) if x and not y)
        print(f"\n  ── {tag}：A {np.mean(a):6.2%}  B {np.mean(b):6.2%}"
              f"  Δ {(np.mean(b) - np.mean(a)):+.2%} ──")
        print(f"  逐集翻转: B 由不中→中 {up} 集 / 由中→不中 {dn} 集"
              f"   净 {up - dn:+d} 集 = {(up - dn) / len(common):+.2%}"
              f"   （McNemar 精确双侧 p = {binom_two_sided(up, dn):.4f}）")

    # —— 两阶段 ——
    d2 = [tgt[k]['stage'] - ref[k]['stage'] for k in common]
    lo2, hi2 = boot_ci(d2)
    print(f"\n  ── 两阶段最终遗憾（越低越好）──")
    print(f"  A {np.mean([ref[k]['stage'] for k in common]):7.3%}"
          f"   B {np.mean([tgt[k]['stage'] for k in common]):7.3%}"
          f"   Δ {np.mean(d2):+7.3%}  CI [{lo2:+.3%}, {hi2:+.3%}]")

    # —— 差异最大的几集（便于定位是哪些电路在撬动）——
    if topk > 0:
        srt = sorted(common, key=lambda x: -abs(tgt[x]['regret'] - ref[x]['regret']))[:topk]
        print(f"\n  ── |Δ遗憾| 最大的 {len(srt)} 集 ──")
        for k in srt:
            print(f"  {k[0]:45s} w={k[1]:2d} n={ref[k]['n']:2d}"
                  f"  A {ref[k]['regret']:7.2%} → B {tgt[k]['regret']:7.2%}"
                  f"   Δ {tgt[k]['regret'] - ref[k]['regret']:+7.2%}")


def main():
    ap = argparse.ArgumentParser(description="历史 sweep .out 的逐集配对比较（window 口径）")
    ap.add_argument("--ref", help="基线档（A）")
    ap.add_argument("--against", nargs='+', default=[], help="对比档（B），可多个")
    ap.add_argument("--nhist", help="只打印该档的 n 直方图 + 按直方图加权的随机基线")
    ap.add_argument("--topk", type=int, default=5, help="列出 |Δ遗憾| 最大的几集（0 = 不列）")
    args = ap.parse_args()

    if not args.ref and not args.nhist:
        ap.error("至少给 --ref 或 --nhist")

    print("⚠ 口径：本脚本读的是 17.3.9 之前的 sweep 输出 → 106 集 **window 池化**口径。")
    print("  池化的并列 artifact 随模型变 → 配对时是额外噪声、不是共模。结论须与 batch 口径重扫互校。")

    if args.nhist:
        _, recs = parse(args.nhist)
        nhist(recs, args.nhist.split('/')[-1])
    if args.ref:
        for t in args.against:
            compare(args.ref, t, args.topk)


if __name__ == '__main__':
    main()
