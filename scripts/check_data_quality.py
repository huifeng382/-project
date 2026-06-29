"""
检查 batch_05 数据质量：延迟分布、角落一致性、电路质量、异常值等。

用法:
    python src/check_data_quality.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import json

DATA_DIR = "data/batch_05"
TIMING_PATH = os.path.join(DATA_DIR, "timing_arcs.parquet")
STATIC_PATH = os.path.join(DATA_DIR, "circuit_static.parquet")

# ───────────────────── 工具函数 ─────────────────────

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ───────────────────── 主流程 ─────────────────────

def main():
    timing = pd.read_parquet(TIMING_PATH)
    static = pd.read_parquet(STATIC_PATH)

    # 列名规范化
    for col in ['candidate', 'candidate_id']:
        if col in timing.columns and 'circuit_id' not in timing.columns:
            timing = timing.rename(columns={col: 'circuit_id'})
    timing['circuit_id'] = timing['circuit_id'].astype(str)
    if 'DELAY' not in timing.columns:
        for col in ['delay_s', 'delay']:
            if col in timing.columns:
                timing = timing.rename(columns={col: 'DELAY'})
                break

    # ── 1. 总体概览 ──
    print_section("1. 总体概览")
    print(f"  总样本: {len(timing)}")
    print(f"  电路数: {timing['circuit_id'].nunique()}")
    print(f"  列: {timing.columns.tolist()}")
    print(f"  缺失值:\n{timing.isnull().sum()[timing.isnull().sum() > 0]}")

    # ── 2. 延迟分布 ──
    print_section("2. DELAY 分布")
    delays = timing['DELAY']
    print(f"  min:    {delays.min():.3e}")
    print(f"  max:    {delays.max():.3e}")
    print(f"  mean:   {delays.mean():.3e}")
    print(f"  median: {delays.median():.3e}")
    print(f"  std:    {delays.std():.3e}")
    print(f"  <= 0:   {(delays <= 0).sum()}")
    print(f"  <= 1e-12: {(delays <= 1e-12).sum()}")

    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"  分位数:")
    for p in percentiles:
        print(f"    p{p:02d}: {np.percentile(delays, p):.3e}")

    # ── 3. 各角落延迟分布 ──
    print_section("3. 各 Corner 延迟分布")
    corners = timing['corner'].unique()
    for c in sorted(corners):
        c_delays = timing[timing['corner'] == c]['DELAY']
        # 解析 corner 名: s{ slew_ps }_l{ load_ff }
        parts = c.split('_')
        slew_str = parts[0].replace('s', '').replace('p', '.')  # "05p0" → "05.0"
        load_str = parts[1].replace('l', '').replace('p', '.')  # "l00p5" → "00.5"
        print(f"  {c:16s} (slew={slew_str}ps, load={load_str}fF): "
              f"n={len(c_delays):5d}, median={c_delays.median():.3e}, "
              f"mean={c_delays.mean():.3e}, std={c_delays.std():.3e}")

    # ── 4. 各引脚延迟 ──
    print_section("4. 各 Switching Pin 延迟")
    for pin in sorted(timing['switching_pin'].unique()):
        p_delays = timing[timing['switching_pin'] == pin]['DELAY']
        print(f"  pin {pin}: n={len(p_delays):5d}, median={p_delays.median():.3e}, "
              f"mean={p_delays.mean():.3e}, std={p_delays.std():.3e}")

    # ── 5. 各方向延迟 ──
    print_section("5. Rise vs Fall")
    for d in ['rise', 'fall']:
        d_delays = timing[timing['direction'] == d]['DELAY']
        print(f"  {d:5s}: n={len(d_delays):5d}, median={d_delays.median():.3e}, "
              f"mean={d_delays.mean():.3e}")

    # ── 6. 电路级质量 ──
    print_section("6. 电路级延迟统计")
    ckt_stats = timing.groupby('circuit_id')['DELAY'].agg(['mean', 'std', 'min', 'max', 'count'])
    ckt_stats['cv'] = ckt_stats['std'] / ckt_stats['mean']  # 变异系数

    print(f"  每电路样本数: min={ckt_stats['count'].min()}, max={ckt_stats['count'].max()}, mean={ckt_stats['count'].mean():.0f}")
    print(f"  电路延迟均值分布:")
    print(f"    min={ckt_stats['mean'].min():.3e}, max={ckt_stats['mean'].max():.3e}, "
          f"median={ckt_stats['mean'].median():.3e}")
    print(f"  变异系数 (CV):")
    print(f"    min={ckt_stats['cv'].min():.3f}, max={ckt_stats['cv'].max():.3f}, "
          f"median={ckt_stats['cv'].median():.3f}")

    # 找出延迟最高/最低/最分散的电路
    print(f"  延迟最低的 3 个电路: {ckt_stats['mean'].nsmallest(3).index.tolist()} → {[f'{v:.3e}' for v in ckt_stats['mean'].nsmallest(3)]}")
    print(f"  延迟最高的 3 个电路: {ckt_stats['mean'].nlargest(3).index.tolist()} → {[f'{v:.3e}' for v in ckt_stats['mean'].nlargest(3)]}")
    print(f"  延迟最分散的 3 个电路 (高CV): {ckt_stats['cv'].nlargest(3).index.tolist()} → {[f'{v:.3f}' for v in ckt_stats['cv'].nlargest(3)]}")

    # ── 7. 延迟-负载-Slew 关系 ──
    print_section("7. 延迟 vs Slew/Load 关系")
    for corner_group, label in [
        (['s05p0_l00p5', 's20p0_l00p5', 's50p0_l00p5'], "固定 load=0.5fF, 不同 slew"),
        (['s05p0_l02p0', 's20p0_l02p0', 's50p0_l02p0'], "固定 load=2.0fF, 不同 slew"),
        (['s05p0_l10p0', 's20p0_l10p0', 's50p0_l10p0'], "固定 load=10fF, 不同 slew"),
        (['s05p0_l00p5', 's05p0_l02p0', 's05p0_l10p0'], "固定 slew=5ps, 不同 load"),
        (['s20p0_l00p5', 's20p0_l02p0', 's20p0_l10p0'], "固定 slew=20ps, 不同 load"),
        (['s50p0_l00p5', 's50p0_l02p0', 's50p0_l10p0'], "固定 slew=50ps, 不同 load"),
    ]:
        print(f"\n  [{label}]")
        for c in corner_group:
            if c in corners:
                med = timing[timing['corner'] == c]['DELAY'].median()
                print(f"    {c}: median delay = {med:.3e}")

    # ── 8. 异常值检测 ──
    print_section("8. 潜在异常值")
    # IQR 方法
    q1 = delays.quantile(0.25)
    q3 = delays.quantile(0.75)
    iqr = q3 - q1
    outliers_iqr = delays[(delays < q1 - 3 * iqr) | (delays > q3 + 3 * iqr)]
    print(f"  IQR 异常值 (3xIQR): {len(outliers_iqr)} 条 ({len(outliers_iqr)/len(delays)*100:.2f}%)")

    # 极端值 (log10 域内偏离 > 2 std)
    log_delays = np.log10(delays)
    log_std = log_delays.std()
    log_mean = log_delays.mean()
    extreme = delays[abs(log_delays - log_mean) > 3 * log_std]
    print(f"  log10 极端值 (3σ): {len(extreme)} 条 ({len(extreme)/len(delays)*100:.2f}%)")

    # ── 9. Train/Val/Test 分布一致性 ──
    print_section("9. Train/Val/Test 划分检查 (模拟)")
    from src.utils import split_by_circuit
    circuit_ids = timing['circuit_id'].unique().tolist()
    train_ids, val_ids, test_ids = split_by_circuit(circuit_ids, seed=42)

    for name, ids in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
        subset = timing[timing['circuit_id'].isin(ids)]['DELAY']
        print(f"  {name}: n={len(subset):5d}, median={subset.median():.3e}, "
              f"mean={subset.mean():.3e}, std={subset.std():.3e}")

    # 角落分布在三组中的一致性
    print(f"\n  角落分布:")
    for c in sorted(corners):
        total_pct = len(timing[timing['corner'] == c]) / len(timing) * 100
        train_pct = len(timing[(timing['corner'] == c) & (timing['circuit_id'].isin(train_ids))]) / len(timing[timing['circuit_id'].isin(train_ids)]) * 100
        test_pct = len(timing[(timing['corner'] == c) & (timing['circuit_id'].isin(test_ids))]) / len(timing[timing['circuit_id'].isin(test_ids)]) * 100
        print(f"    {c:16s}: total={total_pct:5.1f}%, train={train_pct:5.1f}%, test={test_pct:5.1f}%")

    # ── 10. 被清洗掉的样本 ──
    print_section("10. 被 train_sweep.py 清洗掉的样本")
    null_mask = timing['DELAY'].isna() | timing['circuit_id'].isna()
    zero_mask = timing['DELAY'] <= 1e-12
    large_mask = timing['DELAY'] >= 1e-8
    print(f"  DELAY NaN/circuit NaN: {null_mask.sum()} 条")
    print(f"  DELAY <= 1e-12: {zero_mask.sum()} 条")
    print(f"  DELAY >= 1e-8:  {large_mask.sum()} 条")
    print(f"  清洗后剩余: {len(timing) - (null_mask | zero_mask | large_mask).sum()} 条")

    # ── 11. 静态数据检查 ──
    print_section("11. 静态数据检查")
    for col in ['candidate', 'candidate_id']:
        if col in static.columns and 'circuit_id' not in static.columns:
            static = static.rename(columns={col: 'circuit_id'})
    static['circuit_id'] = static['circuit_id'].astype(str)

    print(f"  电路数: {len(static)}")
    print(f"  列: {static.columns.tolist()}")

    # 网表门数分布
    if 'gate_level_netlist' in static.columns:
        netlist_lines = static['gate_level_netlist'].apply(lambda s: len(str(s).split('\n')))
        print(f"  网表行数: min={netlist_lines.min()}, max={netlist_lines.max()}, mean={netlist_lines.mean():.0f}")

    # pin_loads 范围
    if 'pin_loads_json' in static.columns:
        all_loads = []
        for loads_json in static['pin_loads_json']:
            try:
                all_loads.extend(json.loads(loads_json).values())
            except:
                pass
        if all_loads:
            print(f"  Pin loads: min={min(all_loads):.3e}, max={max(all_loads):.3e}, "
                  f"mean={np.mean(all_loads):.3e}")

    # 静态数据 vs 动态数据 电路匹配
    static_cids = set(static['circuit_id'])
    dynamic_cids = set(timing['circuit_id'])
    if static_cids != dynamic_cids:
        print(f"  ⚠ 静态 vs 动态电路不匹配!")
        print(f"    仅静态有: {static_cids - dynamic_cids}")
        print(f"    仅动态有: {dynamic_cids - static_cids}")
    else:
        print(f"  静态与动态电路完全匹配 ✓")

    print("\n" + "=" * 60)
    print("  分析完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
