"""10.3 判定分析：从 shadow 批量输出聚合 recall@top-3 / 选择遗憾 / Spearman。

输入：~NetlistOpt/temp_sim_test/tl_opt_batch/**/gnn_shadow.csv（每行 = 一个候选评估）：
  eval_idx=N, iter=I, window=W, gnn_pred=..., true_delay=...[, transistors=...]
统计单位：每个 (电路, window) 的候选集（GNN 与 SPICE 都成功的候选 ≥4 才算）。

用法（server 端）：
  ~/venv/bin/python3 scripts/diag/_shadow_analyze.py [--min-cands 4]
"""
import argparse, glob, os, re, statistics, sys
import numpy as np

CSV_RE = re.compile(
    r"eval_idx=(\d+), iter=(\d+), window=(\d+), gnn_pred=([0-9.eE+-]+|nan), "
    r"true_delay=([0-9.eE+-]+|NA)"
)

def parse_row(line):
    m = CSV_RE.search(line)
    if not m:
        return None
    return {
        "eval": int(m.group(1)),
        "iter": int(m.group(2)),
        "window": int(m.group(3)),
        "gnn": None if m.group(4) == "nan" else float(m.group(4)),
        "true": None if m.group(5) == "NA" else float(m.group(5)),
    }

def per_window_metrics(rows):
    """rows: 该候选集的全部候选（成功行）。返回 (recall@3, regret, spearman) 或 None。"""
    n = len(rows)
    if n < 2:
        return None
    gnn = np.array([r["gnn"] for r in rows], dtype=np.float64)
    true = np.array([r["true"] for r in rows], dtype=np.float64)
    order_g = np.argsort(gnn)
    order_t = np.argsort(true)
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
    # 两阶段最终遗憾：GNN 前3 → SPICE 精排 → 选前3内真最优（真#1 在则 = 0）
    top3_true_best = true[order_g[: min(3, n)]].min()
    regret_2stage = (top3_true_best - true_best) / true_best
    # Spearman（n>=3 才可靠，n=2 时退化为 ±1，不统计）
    if n >= 3:
        rg = np.empty(n); rt = np.empty(n)
        rg[order_g] = np.arange(n); rt[order_t] = np.arange(n)
        sp = statistics.correlation(rg, rt)
    else:
        sp = None
    return {"recall3": recall3, "regret": regret, "spearman": sp, "n": n,
            "spread_pct": spread_pct,
            "recall2_strict": recall2_strict, "recall3_strict": recall3_strict,
            "recall2_len": recall2_len, "recall3_len": recall3_len,
            "regret_2stage": regret_2stage}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-cands", type=int, default=4, help="候选集最少候选数（10.3 要求 ≥4）")
    ap.add_argument("--root", default=os.path.expanduser("~/NetlistOpt/temp_sim_test/tl_opt_batch"),
                    help="批量输出根目录")
    args = ap.parse_args()

    sets = []          # 全部候选集（≥min-cands）
    small_sets = 0
    ok_rows = 0; fail_rows = 0
    for csvp in sorted(glob.glob(os.path.join(args.root, "*", "*", "gnn_shadow.csv"))):
        circ = os.path.relpath(csvp, args.root)
        by_window = {}
        with open(csvp) as f:
            for line in f:
                r = parse_row(line)
                if not r:
                    continue
                if r["true"] is None or r["gnn"] is None:
                    fail_rows += 1
                    continue
                ok_rows += 1
                by_window.setdefault(r["window"], []).append(r)
        for w, rows in by_window.items():
            # 同 window 内按 eval_idx 去重（同一候选可能被缓存复用？以 eval_idx 唯一为准）
            seen = {}
            for r in rows:
                seen.setdefault(r["eval"], r)
            rows = list(seen.values())
            if len(rows) < args.min_cands:
                small_sets += 1
                continue
            m = per_window_metrics(rows)
            if m is None:
                continue
            m["circuit"] = circ; m["window"] = w
            sets.append(m)

    if not sets:
        print(f"无合格候选集（需要 ≥{args.min_cands} 候选）。成功行={ok_rows} 失败行={fail_rows} 小集={small_sets}")
        return

    r3 = [s["recall3"] for s in sets]
    rg = [s["regret"] for s in sets]
    sp = [s["spearman"] for s in sets if s["spearman"] is not None]
    n_cands = [s["n"] for s in sets]
    r2s = [s["recall2_strict"] for s in sets]
    r3s = [s["recall3_strict"] for s in sets]
    r2l = [s["recall2_len"] for s in sets]
    r3l = [s["recall3_len"] for s in sets]
    r2st = [s["regret_2stage"] for s in sets]

    print(f"候选集数（≥{args.min_cands} 候选）: {len(sets)}   成功行={ok_rows} 失败行={fail_rows} 小集={small_sets}")
    print(f"候选数分布: min={min(n_cands)} med={statistics.median(n_cands):.0f} max={max(n_cands)}")
    print(f"\n=== recall 判断标准（k=2/3 双口径，粗筛最重要指标）===")
    print(f"  前k名中出现实际第1名（严格）:  k=2 {statistics.mean(r2s)*100:6.1f}%   k=3 {statistics.mean(r3s)*100:6.1f}%")
    print(f"  前k名中出现实际前k之一（宽松）: k=2 {statistics.mean(r2l)*100:6.1f}%   k=3 {statistics.mean(r3l)*100:6.1f}%")
    print(f"\n=== 10.3 判定指标（按候选集汇总）===")
    print(f"  选择遗憾（GNN自选top1）: {statistics.mean(rg)*100:6.2f}%   (达标线 ≤5%)   "
          f"中位 {statistics.median(rg)*100:.2f}%")
    print(f"  两阶段最终遗憾（前3→SPICE精排）: {statistics.mean(r2st)*100:6.2f}%   "
          f"中位 {statistics.median(r2st)*100:.2f}%")
    if sp:
        print(f"  Spearman:       {statistics.mean(sp):6.3f}   (次判据 ≥0.6)   "
              f"中位 {statistics.median(sp):.3f}  (n={len(sp)} 集)")
    r3_ok = statistics.mean(r3) >= 0.90
    rg_ok = statistics.mean(rg) <= 0.05
    print(f"\n判定(旧口径参考): recall@top-3(集合版) {'✅≥90%' if r3_ok else '❌<90%'}  "
          f"遗憾 {'✅≤5%' if rg_ok else '❌>5%'}")
    if r3_ok and rg_ok:
        print("→ **两项主判据达标：GNN 可替换逐候选 SPICE 排序**（top-K 精排，仿真省 ≥75%）")
    else:
        print("→ **主判据未达标：GNN 只做启发式预排序**（SPICE 全排序 + GNN 先粗排）")

    # 跨度过滤视图：只统计 (max_true-min_true)/min_true > 10% 的「可排序集」（对齐 V2 hi_spread 口径）
    hi = [s for s in sets if s["spread_pct"] > 10.0]
    if hi:
        hr3 = [s["recall3"] for s in hi]
        hrg = [s["regret"] for s in hi]
        hsp = [s["spearman"] for s in hi if s["spearman"] is not None]
        hr2s = [s["recall2_strict"] for s in hi]
        hr3s = [s["recall3_strict"] for s in hi]
        hr2l = [s["recall2_len"] for s in hi]
        hr3l = [s["recall3_len"] for s in hi]
        hr2st = [s["regret_2stage"] for s in hi]
        print(f"\n=== 跨度>10% 子集（{len(hi)}/{len(sets)} 集，对齐 V2 hi_spread）===")
        print(f"  严格(实际第1∈预测前k):  k=2 {statistics.mean(hr2s)*100:6.1f}%   k=3 {statistics.mean(hr3s)*100:6.1f}%")
        print(f"  宽松(前k含实际前k之一): k=2 {statistics.mean(hr2l)*100:6.1f}%   k=3 {statistics.mean(hr3l)*100:6.1f}%")
        print(f"  两阶段最终遗憾: {statistics.mean(hr2st)*100:6.2f}%   "
              f"选择遗憾(GNN自选): {statistics.mean(hrg)*100:6.2f}%   Spearman: {statistics.mean(hsp):.3f} (n={len(hsp)})")
    else:
        print("\n（无跨度>10% 的候选集）")

    # 明细：按遗憾排序列出每集
    print("\n=== 每候选集明细（按遗憾升序；#1∈前k=严格, 前k∩真前k=宽松）===")
    for s in sorted(sets, key=lambda x: x["regret"]):
        sps = f"{s['spearman']:.2f}" if s["spearman"] is not None else "-"
        print(f"  {s['circuit']:45s} w={s['window']:3d} n={s['n']:2d} "
              f"#1∈前2={'Y' if s['recall2_strict'] else 'n'} 前2∩真={ 'Y' if s['recall2_len'] else 'n'} "
              f"#1∈前3={'Y' if s['recall3_strict'] else 'n'} 前3∩真={ 'Y' if s['recall3_len'] else 'n'} "
              f"regret={s['regret']*100:7.2f}% 2stage={s['regret_2stage']*100:6.2f}% sp={sps}")

if __name__ == "__main__":
    main()
