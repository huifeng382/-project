"""V1-V4: Rust 差距归因分析 (16.11.17)
V1: Rust 候选集 true_delay spread 分布(判断任务难度/SNR 天花板)
V2: 真值排序上界(即使 GNN 完美, 两阶段/选择的遗憾下限)
V3: 训练数据 vs Rust 候选组内延迟差对比
V4: 候选规模(transistors) vs 排序难度
用法: python scripts/diag/_gap_attribution.py
"""
import re, glob, os, sys
import numpy as np
from collections import defaultdict

CSV_RE = re.compile(
    r'eval_idx=(\d+), iter=(\d+), window=(\d+), gnn_pred=([0-9.eE+-]+|nan), '
    r'true_delay=([0-9.eE+-]+|NA)(?:, transistors=(\d+))?')

def load_shadow():
    """返回 {(circ, iter, window): [(true_delay, transistors, gnn)]}"""
    sets = defaultdict(list)
    for f in glob.glob('/home/tianlang/NetlistOpt/temp_sim_test/tl_opt_batch/**/gnn_shadow.csv',
                       recursive=True):
        parts = f.split('/tl_opt_batch/')[-1].split('/')
        circ = parts[1] if len(parts) >= 2 else '?'
        for line in open(f):
            m = CSV_RE.search(line)
            if not m:
                continue
            try:
                t = float(m.group(5))
            except (ValueError, TypeError):
                continue
            g = float(m.group(4)) if m.group(4) != 'nan' else float('nan')
            tr = int(m.group(6)) if m.group(6) else -1
            sets[(circ, int(m.group(2)), int(m.group(3)))].append((t, tr, g))
    return sets

def main():
    sets = load_shadow()
    valid = {k: v for k, v in sets.items() if len(v) >= 4}
    print(f'总候选集: {len(sets)}, >=4候选有效集: {len(valid)}')

    # ---------- V1: Rust 候选集 true_delay spread ----------
    print('\n===== V1: Rust 候选集延迟差分布(任务难度) =====')
    spreads = []
    for k, rows in valid.items():
        ts = sorted(r[0] for r in rows)
        spread = (ts[-1] - ts[0]) / np.median(ts) * 100
        spreads.append(spread)
    s = np.array(spreads)
    print(f'有效集数={len(s)}')
    print(f'spread(组内最大-最小 / 中位): min={s.min():.2f}% med={np.median(s):.2f}% '
          f'p75={np.percentile(s,75):.2f}% max={s.max():.2f}%')
    for th in [1, 2, 5, 10, 20]:
        print(f'  spread<{th}% 的集占比: {(s<th).mean():.1%}')

    # ---------- V2: 真值排序上界 ----------
    print('\n===== V2: 真值排序上界(完美 GNN 的下限) =====')
    sel_regrets = []   # 选择真最快(遗憾=0 因为是真值排序)
    two_stage = []     # 真值 top-3 内最优 vs 真全局最优
    for k, rows in valid.items():
        ts = np.array(sorted(r[0] for r in rows))
        best = ts[0]
        # 真值排序选 top1 = 最优, 遗憾 0
        sel_regrets.append(0.0)
        # 两阶段: 真值 top-3 选最优(就是 ts[0]), 遗憾也 0
        two_stage.append(0.0)
    print(f'真值排序: 选择遗憾 = 0(恒等), 两阶段遗憾 = 0(恒等)')
    print('→ 若用 SPICE 真值排序, 遗憾为 0: 差距全在 GNN 排序质量, 非任务上限')
    # 但候选集内延迟差小 -> 即使排序错, 实际延迟损失?
    print('\n 候选集内「选错代价」: 若 GNN 选到次优(top2), 损失多少延迟?')
    costs = []
    for k, rows in valid.items():
        ts = sorted(r[0] for r in rows)
        if len(ts) >= 2:
            costs.append((ts[1] - ts[0]) / ts[0] * 100)  # 次优 vs 最优
    c = np.array(costs)
    print(f'  选次优的延迟代价: med={np.median(c):.2f}%  p75={np.percentile(c,75):.2f}%')

    # ---------- V3: 训练 vs Rust 组内延迟差 ----------
    print('\n===== V3: 训练数据 vs Rust 组内延迟差 =====')
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    import pandas as pd
    import glob as gl
    dp = gl.glob('/home/tianlang/-project/data/batch_v2_full/timing_arcs*.parquet')
    ddf = pd.concat([pd.read_parquet(p, columns=['expr', 'DELAY']) for p in dp], ignore_index=True)
    tr_spreads = []
    for e, g in ddf.groupby('expr'):
        if len(g) >= 4:
            tr_spreads.append((g['DELAY'].max() - g['DELAY'].min()) / g['DELAY'].median() * 100)
    ts_tr = np.array(tr_spreads)
    print(f'训练(按expr组): spread med={np.median(ts_tr):.1f}%  <5%占比={(ts_tr<5).mean():.1%}')
    print(f'Rust 候选集:     spread med={np.median(s):.2f}%  <5%占比={(s<5).mean():.1%}')
    print(f'→ 训练组内差({np.median(ts_tr):.0f}%) >> Rust({np.median(s):.1f}%): '
          f'模型从没学过细粒度分辨(<{np.median(s):.0f}% 量级)')

    # ---------- V4: 候选规模 vs 排序难度 ----------
    print('\n===== V4: 候选规模 vs GNN 排序难度 =====')
    # 用 gnn 排序算遗憾(有 gnn_pred 的行)
    size_buckets = defaultdict(list)
    for k, rows in valid.items():
        tr = max(r[1] for r in rows)
        if tr < 0:
            continue
        # GNN 遗憾: 用 gnn_pred 排, 选 top1 的 true_delay / 真最优
        ok = [r for r in rows if not np.isnan(r[2])]
        if len(ok) < 4:
            continue
        ts = np.array([r[0] for r in ok])
        gs = np.array([r[2] for r in ok])
        best_i = int(np.argmin(ts))
        gnn_i = int(np.argmin(gs))
        regret = (ts[gnn_i] - ts[best_i]) / ts[best_i] * 100 if ts[best_i] > 0 else 0
        if tr < 30:
            b = '小(<30管)'
        elif tr < 100:
            b = '中(30-100管)'
        else:
            b = '大(>100管)'
        size_buckets[b].append(regret)
    for b in ['小(<30管)', '中(30-100管)', '大(>100管)']:
        arr = np.array(size_buckets.get(b, []))
        if len(arr):
            print(f'  {b}: n={len(arr)}  GNN选择遗憾 med={np.median(arr):.2f}%')

if __name__ == '__main__':
    main()
