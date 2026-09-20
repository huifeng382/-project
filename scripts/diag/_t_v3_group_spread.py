"""V3 组内延迟跨度实测 —— 独立复核交付方的 GATE 7（2026-09-20）。

为什么必须自己算一遍：GNN 的任务本体就是**同一 expr 组内功能等价变体之间排序**。
若组内变体的延迟本身几乎一样，那么「排序」在物理上就没有可学的信号，
模型的失败会被误读成「模型不行」而不是「数据没给出区分度」。
交付方自评 GATE 7 **硬失败**（相邻差 67.3% 落在 <1% 档），这个数必须独立复现。

口径（写死在代码里，避免读法漂移）：
  * 每个**电路**的标量延迟 = 该电路全部行的 DELAY 均值（与 truth2.csv 的 avg_delay 同口径）；
  * 组内按该标量升序排序，相邻差 gap_i = (d[i+1]-d[i]) / d[i]（相对前一档的步长）；
  * 组内跨度 spread = (max-min) / median。
⚠ 逐电路取的是 84~108 条弧的**均值** ⇒ 单条弧的仿真噪声被平均掉，
  所以「跨度窄」是真实性质，不是噪声（这一点与单次仿真不可比）。

严格/宽松两口径都给（沿用本仓库的 recall 双口径习惯）：
  宽松 = 步长 >1%；严格 = 步长 >5%。

只读 data/，不修改任何文件，不跑训练。
"""
import glob
import os
import sys
from collections import Counter, defaultdict

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd                                          # noqa: E402
import pyarrow.parquet as pq                                 # noqa: E402

BUCKETS = [('<1%', 0.0, 0.01), ('1-5%', 0.01, 0.05),
           ('5-20%', 0.05, 0.20), ('>20%', 0.20, float('inf'))]


def load_batch(bdir, with_shape=False):
    """返回 DataFrame[circuit_id, expr, delay_per_circuit(, shape)]，逐电路均值。"""
    fps = sorted(glob.glob(os.path.join(bdir, 'timing_arcs*.parquet')))
    if not fps:
        return None
    cols = ['circuit_id', 'expr', 'DELAY']
    df = pd.concat([pq.read_table(p, columns=cols).to_pandas() for p in fps],
                   ignore_index=True)
    df['circuit_id'] = df['circuit_id'].astype(str)
    per = df.groupby('circuit_id').agg(expr=('expr', 'first'),
                                       d=('DELAY', 'mean')).reset_index()
    per['expr'] = per['expr'].astype(str)
    if with_shape:
        sp = os.path.join(bdir, 'circuit_static.parquet')
        if os.path.exists(sp):
            st = pq.read_table(sp, columns=['circuit_id', 'shape']).to_pandas()
            st['circuit_id'] = st['circuit_id'].astype(str)
            per = per.merge(st, on='circuit_id', how='left')
    return per


def analyze(per, label):
    print()
    print('=' * 100)
    print(f'{label}   电路 {len(per)}  组 {per["expr"].nunique()}')
    print('=' * 100)
    gaps = []
    spreads = []
    headroom = []
    for e, g in per.groupby('expr'):
        v = sorted(g['d'].tolist())
        if len(v) < 2:
            continue
        med = v[len(v) // 2]
        if med <= 0:
            continue
        spreads.append((v[-1] - v[0]) / med)
        headroom.append((med - v[0]) / med)         # 完美排序者相对中位的可降幅
        for i in range(len(v) - 1):
            if v[i] > 0:
                gaps.append((v[i + 1] - v[i]) / v[i])
    n = len(gaps)
    print(f'  相邻步长 n={n}')
    for nm, lo, hi in BUCKETS:
        c = sum(1 for x in gaps if lo <= x < hi)
        print(f'    {nm:<7s} {c:>7d}  {c / n:>6.1%}')
    s = sorted(gaps)
    print(f'    步长分位 p10={s[n // 10]:.4f} p50={s[n // 2]:.4f} p90={s[9 * n // 10]:.4f}')
    print(f'  宽松口径(步长>1%): {sum(1 for x in gaps if x > 0.01) / n:.1%}   '
          f'严格口径(>5%): {sum(1 for x in gaps if x > 0.05) / n:.1%}')
    sp = sorted(spreads)
    hr = sorted(headroom)
    m = len(sp)
    print(f'  组内跨度 (max-min)/median : 中位 {sp[m // 2]:.3f}  '
          f'p10 {sp[m // 10]:.3f}  p90 {sp[9 * m // 10]:.3f}   组数 {m}')
    print(f'  完美排序者相对中位可降幅   : 中位 {hr[m // 2]:.1%}  '
          f'p90 {hr[9 * m // 10]:.1%}')
    for thr in (0.05, 0.10, 0.20, 0.30):
        c = sum(1 for x in sp if x > thr)
        print(f'    跨度 >{thr:.0%} 的组: {c:>4d} / {m}  {c / m:>6.1%}')
    return gaps, spreads


print('读取 V3（31 分片）...')
v3 = load_batch(os.path.join(ROOT, 'data', 'v3_delivery'), with_shape=True)
g3, s3 = analyze(v3, 'V3  data/v3_delivery（单一数据集）')

print()
print('--- V3 逐形状（9x6 占交付 71% 的行，交付方称其家族跨度只 1.31×）---')
if 'shape' in v3.columns:
    rows = []
    for sh, g in v3.groupby('shape'):
        sp = []
        for e, gg in g.groupby('expr'):
            v = sorted(gg['d'].tolist())
            if len(v) >= 2 and v[len(v) // 2] > 0:
                sp.append((v[-1] - v[0]) / v[len(v) // 2])
        if sp:
            sp.sort()
            rows.append((str(sh), len(g), len(sp), sp[len(sp) // 2]))
    for sh, nc, ng, med in sorted(rows, key=lambda r: -r[1]):
        print(f'    shape {sh:<6s} 电路 {nc:>6d}  组 {ng:>4d}   跨度中位 {med:.3f}')

for b in ('batch_v2_full', 'batch_v2_rest', 'batch_v2_m4'):
    bd = os.path.join(ROOT, 'data', b)
    if not os.path.isdir(bd):
        continue
    print(f'\n读取 V2 {b} ...')
    pv = load_batch(bd)
    if pv is not None:
        analyze(pv, f'V2  {b}（对照）')

print()
print('完成（本脚本未修改任何文件）')
