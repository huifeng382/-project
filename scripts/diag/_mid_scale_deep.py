"""V5: 中规模电路(30-100管)遗憾高的深挖 (16.11.17)
1) 中规模桶的电路构成 + 候选集数
2) 中规模 vs 小/大的特征差异(门数/晶体管/延迟范围)
3) 排除 seed 偏差: 用不同模型(nowave42/iaa42m4)对比同规模桶遗憾
用法: python scripts/diag/_mid_scale_deep.py
"""
import re, glob, os, sys
import numpy as np
from collections import defaultdict

CSV_RE = re.compile(
    r'eval_idx=(\d+), iter=(\d+), window=(\d+), gnn_pred=([0-9.eE+-]+|nan), '
    r'true_delay=([0-9.eE+-]+|NA)(?:, transistors=(\d+))?')

def load_shadow():
    sets = defaultdict(lambda: {'circ': None, 'rows': []})
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
            key = (circ, int(m.group(2)), int(m.group(3)))
            sets[key]['circ'] = circ
            sets[key]['rows'].append((t, tr, g))
    return sets

def regret_of(rows):
    ok = [r for r in rows if not np.isnan(r[2])]
    if len(ok) < 4:
        return None
    ts = np.array([r[0] for r in ok])
    gs = np.array([r[2] for r in ok])
    bi = int(np.argmin(ts))
    gi = int(np.argmin(gs))
    return (ts[gi] - ts[bi]) / ts[bi] * 100 if ts[bi] > 0 else 0.0

def main():
    sets = load_shadow()
    # 按桶收集
    buckets = defaultdict(lambda: {'circs': defaultdict(int), 'regrets': [],
                                   'spreads': [], 'n_sets': 0})
    for key, d in sets.items():
        rows = d['rows']
        if len(rows) < 4:
            continue
        circ = d['circ']
        tr = max(r[1] for r in rows)
        if tr < 0:
            continue
        if tr < 30:
            b = 'S小(<30)'
        elif tr < 100:
            b = 'M中(30-100)'
        else:
            b = 'L大(>100)'
        buckets[b]['circs'][circ] += 1
        buckets[b]['n_sets'] += 1
        ts = sorted(r[0] for r in rows)
        buckets[b]['spreads'].append((ts[-1]-ts[0])/np.median(ts)*100)
        r = regret_of(rows)
        if r is not None:
            buckets[b]['regrets'].append(r)

    for b in ['S小(<30)', 'M中(30-100)', 'L大(>100)']:
        d = buckets[b]
        print(f'\n===== {b}: {d["n_sets"]} 候选集 =====')
        # 电路构成
        top_circs = sorted(d['circs'].items(), key=lambda kv: -kv[1])[:6]
        circ_str = ', '.join(f'{c}×{n}' for c, n in top_circs)
        print(f'  主要电路: {circ_str}')
        regs = np.array(d['regrets'])
        sps = np.array(d['spreads'])
        if len(regs):
            print(f'  候选集数={len(regs)}, GNN遗憾 med={np.median(regs):.2f}% mean={regs.mean():.2f}%')
        if len(sps):
            print(f'  spread med={np.median(sps):.1f}%')

    # 中规模的详细电路级遗憾
    print('\n===== 中规模(M) 逐电路遗憾 =====')
    circ_sets = defaultdict(list)
    for key, d in sets.items():
        rows = d['rows']
        if len(rows) < 4:
            continue
        tr = max(r[1] for r in rows)
        if not (30 <= tr < 100):
            continue
        r = regret_of(rows)
        if r is not None:
            circ_sets[d['circ']].append(r)
    for circ, regs in sorted(circ_sets.items(), key=lambda kv: -np.median(kv[1])):
        arr = np.array(regs)
        print(f'  {circ}: n={len(arr)} 遗憾 med={np.median(arr):.2f}%')

if __name__ == '__main__':
    main()
