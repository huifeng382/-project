"""V6: 用 recall 重新分析——聚焦 recall 低的候选集 (16.11.17)
遗憾被 spread 主导(见 V1-V5); recall@1(选对真最快)反映纯排序质量。
分析: 1) 各电路 recall@1 2) recall 低的集的特征(spread/规模/深度代理) 3) recall vs spread 关系
用法: python scripts/diag/_recall_focus.py
"""
import re, glob, os, sys
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr

CSV_RE = re.compile(
    r'eval_idx=(\d+), iter=(\d+), window=(\d+), gnn_pred=([0-9.eE+-]+|nan), '
    r'true_delay=([0-9.eE+-]+|NA)(?:, transistors=(\d+))?')

def main():
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
            sets[key]['rows'].append((t, g, tr))

    # 每候选集: recall@1(选对真最快) + spread + spearman
    results = []
    for key, d in sets.items():
        ok = [r for r in d['rows'] if not np.isnan(r[1])]
        if len(ok) < 4:
            continue
        ts = np.array([r[0] for r in ok])
        gs = np.array([r[1] for r in ok])
        bi = int(np.argmin(ts))
        gi = int(np.argmin(gs))
        hit = 1 if gi == bi else 0
        spread = (ts.max() - ts.min()) / ts.min() * 100
        rho, _ = spearmanr(ts, gs)
        trs = [r[2] for r in ok if r[2] > 0]
        results.append({'circ': d['circ'], 'hit': hit, 'spread': spread,
                        'rho': rho, 'n': len(ok),
                        'tr_med': np.median(trs) if trs else 0})

    # 1) 每电路 recall@1
    print('===== 各电路 recall@1(选对真最快比例) =====')
    circ_stats = defaultdict(list)
    for r in results:
        circ_stats[r['circ']].append(r)
    for circ, rs in sorted(circ_stats.items(), key=lambda kv: -sum(x['hit'] for x in kv[1]) / len(kv[1])):
        hits = sum(x['hit'] for x in rs)
        n = len(rs)
        sps = np.array([x['spread'] for x in rs])
        print(f'  {circ:<14} n={n:<4} recall@1={hits/n*100:5.1f}%  spread med={np.median(sps):5.1f}%')

    # 2) recall@1 vs spread 分箱
    print('\n===== recall@1 vs spread 分箱 =====')
    for lo, hi in [(0, 10), (10, 30), (30, 60), (60, 200)]:
        sub = [r for r in results if lo <= r['spread'] < hi]
        if sub:
            hits = sum(x['hit'] for x in sub)
            print(f'  spread {lo}-{hi}%: n={len(sub)} recall@1={hits/len(sub)*100:.1f}%')

    # 3) recall 低的集长什么样(选错真最快的)
    print('\n===== recall 低(选错)的集分析 =====')
    miss = [r for r in results if r['hit'] == 0]
    hit_r = [r for r in results if r['hit'] == 1]
    print(f'  总集: {len(results)}, 选错: {len(miss)} ({len(miss)/len(results)*100:.1f}%)')
    if miss and hit_r:
        ms = np.array([x['spread'] for x in miss])
        hs = np.array([x['spread'] for x in hit_r])
        print(f'  选错集 spread med={np.median(ms):.1f}% vs 选对集 spread med={np.median(hs):.1f}%')
        mr = np.array([x['rho'] for x in miss])
        hr = np.array([x['rho'] for x in hit_r])
        print(f'  选错集 Spearman med={np.median(mr):.3f} vs 选对集 med={np.median(hr):.3f}')
        # 选错时 GNN 选的是第几快
        # 需要重算: GNN 选的位置
        print(f'  选错集中, GNN 的 Spearman<0(反向)占比: {(mr<0).mean():.1%}')

    # 4) 候选数 vs recall
    print('\n===== 候选数(n) vs recall@1 =====')
    for nn_lo, nn_hi in [(4, 6), (6, 10), (10, 20), (20, 100)]:
        sub = [r for r in results if nn_lo <= r['n'] < nn_hi]
        if sub:
            hits = sum(x['hit'] for x in sub)
            print(f'  n={nn_lo}-{nn_hi}: n集={len(sub)} recall@1={hits/len(sub)*100:.1f}%')

if __name__ == '__main__':
    main()
