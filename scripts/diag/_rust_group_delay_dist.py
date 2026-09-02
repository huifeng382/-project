"""Rust 贪心组内(候选集)真延迟差分布: 判断 V3 '每组2-3个>10%' 要求是否贴近 Rust 现实。
按 (circ, iter, window) 分组(>=4 候选), 统计: spread / 与组内最优差>10%的候选数 / 相邻差各档占比。
"""
import glob, re
from collections import defaultdict
import numpy as np

CSV_RE = re.compile(r'eval_idx=(\d+), iter=(\d+), window=(\d+), gnn_pred=([0-9.eE+-]+|nan), true_delay=([0-9.eE+-]+|NA)(?:, transistors=(\d+))?')

sets = defaultdict(list)
for f in glob.glob('/home/tianlang/NetlistOpt/temp_sim_test/tl_opt_batch/**/gnn_shadow.csv', recursive=True):
    parts = f.split('/tl_opt_batch/')[-1].split('/')
    circ = parts[1] if len(parts) >= 2 else '?'
    for line in open(f):
        m = CSV_RE.search(line)
        if not m: continue
        try:
            t = float(m.group(5))
        except (ValueError, TypeError):
            continue
        sets[(circ, int(m.group(2)), int(m.group(3)))].append(t)

valid = [v for v in sets.values() if len(v) >= 4]
print('有效候选集(>=4):', len(valid), '/', len(sets))

spreads = []
n_gt10 = []       # 与最优差>10% 的候选数
n_gt10_frac = []  # 该占比
frac_gt5 = []
has2gt10 = 0
for v in valid:
    ts = np.array(sorted(v))
    best = ts.min()
    rel = (ts - best) / best * 100
    spreads.append((ts.max()-ts.min())/np.median(ts)*100)
    k = int((rel > 10).sum())
    n_gt10.append(k)
    n_gt10_frac.append(k / len(ts))
    has2gt10 += (k >= 2)

print('组内 spread 分布: med=%.0f%% p75=%.0f%% p90=%.0f%%' % tuple(np.percentile(spreads, [50,75,90])))
a = np.array(n_gt10)
print('组内差>10%%候选数: 0个组占 %.0f%%, >=1个 %.0f%%, >=2个 %.0f%%, >=3个 %.0f%%' % (
    100*np.mean(a==0), 100*np.mean(a>=1), 100*np.mean(a>=2), 100*np.mean(a>=3)))
print('组内差>5%%候选占比 med=%.0f%%' % np.median(100*np.array(n_gt10_frac)))
print('若要求每组>=2个差>10%%: 满足率 = %.0f%% (%d/%d)' % (100*has2gt10/len(valid), has2gt10, len(valid)))

# 相邻差档(排序后相邻候选差) 全局分布
adj = []
for v in valid:
    ts = sorted(v)
    for i in range(len(ts)-1):
        if ts[i] > 0:
            adj.append((ts[i+1]-ts[i])/ts[i]*100)
adj = np.array(adj)
print('相邻差 n=%d: <1%%: %.0f%%  1-5%%: %.0f%%  5-20%%: %.0f%%  >20%%: %.0f%%' % (
    len(adj), 100*np.mean(adj<1), 100*np.mean((adj>=1)&(adj<5)), 100*np.mean((adj>=5)&(adj<20)), 100*np.mean(adj>=20)))
