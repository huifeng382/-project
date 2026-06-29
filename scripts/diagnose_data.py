"""
诊断数据质量：检查 delay 分布、corner 差异、数据异常
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import json

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

for name in ['batch1_30k', 'batch2_70k']:
    d = pd.read_parquet(os.path.join(DATA_DIR, name, 'timing_arcs.parquet'))
    d = d.rename(columns={'candidate_id': 'circuit_id', 'delay_s': 'DELAY'})
    d = d[d['DELAY'] > 0]
    d['circuit_id'] = d['circuit_id'].astype(str)

    print(f"\n{'='*70}")
    print(f"  {name}: {len(d):,} samples, {d['circuit_id'].nunique()} circuits")
    print(f"{'='*70}")

    # 1. 全局 delay 分布
    print(f"\n  Delay 分布 (ps):")
    delays_ps = d['DELAY'] * 1e12
    for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"    P{q:2d}: {np.percentile(delays_ps, q):.2f} ps")
    print(f"    min: {delays_ps.min():.4f} ps, max: {delays_ps.max():.2f} ps")

    # 2. 每个 corner 的 delay 分布
    print(f"\n  Per-corner delay (ps):")
    print(f"    {'corner':<18s} {'n':>7s} {'mean':>10s} {'std':>10s} {'min':>10s} {'max':>10s} {'CV%':>8s}")
    print(f"    {'-'*63}")
    for c in sorted(d['corner'].unique()):
        subset = d[d['corner'] == c]['DELAY'] * 1e12
        cv = subset.std() / subset.mean() * 100 if subset.mean() > 0 else 0
        print(f"    {c:<18s} {len(subset):>7,d} {subset.mean():>10.4f} {subset.std():>10.4f} "
              f"{subset.min():>10.4f} {subset.max():>10.2f} {cv:>7.1f}%")

    # 3. 检查负值和零值
    n_neg = (d['DELAY'] <= 0).sum()
    n_near_zero = (d['DELAY'] < 1e-12).sum()
    if n_neg or n_near_zero:
        print(f"\n  ⚠ 负值: {n_neg}, 接近零(<1e-12): {n_near_zero}")

    # 4. 每个电路在不同 corner 下的 delay 变化（取前3个电路）
    print(f"\n  同一电路跨 corner 的 delay 变化（前3个电路）:")
    for cid in sorted(d['circuit_id'].unique())[:3]:
        sub = d[d['circuit_id'] == cid]
        # 找同一 pin+dir 在不同 corner 下的 delay
        for (pin, direction), grp in sub.groupby(['switching_pin', 'direction']):
            if len(grp) >= 3:
                delays = grp['DELAY'] * 1e12
                print(f"    {cid} pin={pin} dir={direction}: "
                      f"delay={delays.min():.3f}~{delays.max():.3f}ps, "
                      f"ratio={delays.max()/delays.min():.1f}x, n_corners={len(grp)}")
                break  # 只打印一个 combo

    # 5. 引脚级别的 delay 差异
    print(f"\n  Per-pin delay 统计:")
    for pin in sorted(d['switching_pin'].unique()):
        sub = d[d['switching_pin'] == pin]['DELAY'] * 1e12
        print(f"    {pin}: n={len(sub):>7,d}  mean={sub.mean():.2f}ps  median={sub.median():.2f}ps  "
              f"std={sub.std():.2f}ps  skew={sub.skew():.2f}")

    # 6. direction 差异
    print(f"\n  Direction 差异:")
    for direc in ['rise', 'fall']:
        sub = d[d['direction'] == direc]['DELAY'] * 1e12
        print(f"    {direc}: n={len(sub):>7,d}  mean={sub.mean():.2f}ps  median={sub.median():.2f}ps")

    # 7. slew 和 load 的简单回归检查
    if name == 'batch1_30k':
        print(f"\n  Slew/Load 对 delay 的影响 (batch1 全 corner):")
        for load_val in sorted(d['output_load_f'].unique()):
            sub = d[d['output_load_f'] == load_val]
            delays = sub['DELAY'] * 1e12
            slews = sub['slew_s'] * 1e12
            if len(sub) > 100:
                # 计算平均趋势
                slew_bins = pd.cut(slews, bins=6)
                trend = sub.groupby(slew_bins)['DELAY'].mean() * 1e12
                print(f"    load={load_val*1e15:.1f}fF: delay range={delays.min():.2f}~{delays.max():.2f}ps "
                      f"(n={len(sub):,})")

print(f"\n{'='*70}")
print("  诊断完成")
print(f"{'='*70}")
