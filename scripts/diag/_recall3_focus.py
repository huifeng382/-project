"""V7: recall@3 聚焦分析 (16.11.17)
recall@3 = 真#1 是否在 GNN 前3(两阶段流程的关键指标)。
聚焦 recall@3 低的候选集: 分布、特征(spread/规模/电路)、与 recall@1 的关系。
用法: python scripts/diag/_recall3_focus.py
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

    results = []
    for key, d in sets.items():
        ok = [r for r in d['rows'] if not np.isnan(r[1])]
        if len(ok) < 4:
            continue
        ts = np.array([r[0] for r in ok])
        gs = np.array([r[1] for r in ok])
        bi = int(np.argmin(ts))
        g_order = np.argsort(gs)
        rank_of_best = int(np.where(g_order == bi)[0][0]) + 1  # 1-based
        hit3 = 1 if rank_of_best <= 3 else 0
        spread = (ts.max() - ts.min()) / ts.min() * 100
        rho, _ = spearmanr(ts, gs)
        trs = [r[2] for r in ok if r[2] > 0]
        # 真前3 内延迟差(两阶段在该集的最优可能)
        ts_sorted = np.sort(ts)
        top3_spread = (ts_sorted[2] - ts_sorted[0]) / ts_sorted[0] * 100 if len(ts_sorted) >= 3 else 0
        results.append({'circ': d['circ'], 'hit3': hit3, 'rank': rank_of_best,
                        'spread': spread, 'rho': rho, 'n': len(ok),
                        'tr_med': np.median(trs) if trs else 0,
                        'top3_spread': top3_spread})

    total = len(results)
    hit3_total = sum(r['hit3'] for r in results)
    print(f'总候选集: {total}, recall@3(真#1∈GNN前3): {hit3_total/total*100:.1f}%')

    # 1) 每电路 recall@3
    print('\n===== 各电路 recall@3 =====')
    circ_stats = defaultdict(list)
    for r in results:
        circ_stats[r['circ']].append(r)
    for circ, rs in sorted(circ_stats.items(), key=lambda kv: -sum(x['hit3'] for x in kv[1]) / len(kv[1])):
        hits = sum(x['hit3'] for x in rs)
        n = len(rs)
        sps = np.array([x['spread'] for x in rs])
        print(f'  {circ:<14} n={n:<4} recall@3={hits/n*100:5.1f}%  spread med={np.median(sps):5.1f}%')

    # 2) recall@3 失败集特征 vs 成功集
    fail = [r for r in results if r['hit3'] == 0]
    succ = [r for r in results if r['hit3'] == 1]
    print(f'\n失败集: {len(fail)} ({len(fail)/total*100:.1f}%), 成功集: {len(succ)}')
    for name, grp in [('失败(真#1不在前3)', fail), ('成功(真#1在前3)', succ)]:
        sps = np.array([x['spread'] for x in grp])
        rhos = np.array([x['rho'] for x in grp])
        ns = np.array([x['n'] for x in grp])
        trs = np.array([x['tr_med'] for x in grp])
        print(f'  {name}: n={len(grp)} spread med={np.median(sps):.1f}% '
              f'Spearman med={np.median(rhos):.3f} 候选数 med={np.median(ns):.0f} 管数 med={np.median(trs):.0f}')

    # 3) 失败集的电路分布
    print('\n失败集电路分布(Top):')
    fc = defaultdict(int)
    for r in fail:
        fc[r['circ']] += 1
    for circ, c in sorted(fc.items(), key=lambda kv: -kv[1])[:8]:
        print(f'  {circ}: {c} ({c/len(fail)*100:.1f}%)')

    # 4) spread 分箱 vs recall@3
    print('\n===== recall@3 vs spread 分箱 =====')
    for lo, hi in [(0, 10), (10, 30), (30, 60), (60, 200)]:
        sub = [r for r in results if lo <= r['spread'] < hi]
        if sub:
            hits = sum(x['hit3'] for x in sub)
            print(f'  spread {lo}-{hi}%: n={len(sub)} recall@3={hits/len(sub)*100:.1f}%')

    # 5) 真#1 掉出前3 时, GNN 前3 是否有「接近最优」的替代(top3_spread 小=两阶段仍可救)
    print('\n===== 失败集的可救性(两阶段视角) =====')
    if fail:
        t3 = np.array([x['top3_spread'] for x in fail])
        # 真前3内 spread: 若小, 即使真#1掉出, 前3里也有接近的
        print(f'失败集: 真前3内 spread med={np.median(t3):.1f}%')
        print(f'  真前3 spread<10% 的失败集占比: {(t3<10).mean():.1%} (这些两阶段仍接近最优)')
        print(f'  真前3 spread>20% 的失败集占比: {(t3>20).mean():.1%} (这些两阶段会真损失)')

if __name__ == '__main__':
    main()
