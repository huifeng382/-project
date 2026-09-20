#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V3 组内重复候选 → 组内延迟差（缺陷 4）的根因分解（只读）。

背景：`_t_v3_group_integrity.py` 实测 **37.2% 的电路在**自己的 expr 组**内**存在
静态输入全签名完全相同的孪生**（净list 层 40.3%）。若这些孪生连**时序数据**也相同，
它们就是同一个设计的重复副本 —— 那么「组内延迟差太小」（缺陷 4）就不（只）是
「采集不到差异大的实现」，而是**候选集里有大量重复点**。这两件事的修法完全不同。

本脚本做三件事：
  1. 每电路的标量延迟 = 其全部行 DELAY 的均值（与 truth2.csv 的 avg_delay 同口径）。
  2. 孪生判定升级为**设计级**：静态全签名相同 **且** 逐弧签名（switching_pin×direction×
     output×DELAY 的排序 md5）相同 → 判为「真重复」；只有静态相同而无时序相同 → 「同构异值」。
  3. 相邻差分解：把 `<1%` 档拆成 **恰好 0** 与 **(0,1%)**；并给出**去重后**的组内跨度/相邻差，
     回答「缺陷 4 是重复造成的还是采集极限造成的」。

用法：python scripts/diag/_t_v3_dup_delay.py
"""
import os
import glob
import hashlib
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
VDIR = os.path.join(ROOT, 'data', 'v3_delivery')
ARCS = sorted(glob.glob(os.path.join(VDIR, 'timing_arcs_part*.parquet')))
SIG_COLS = ['gate_level_netlist', 'cell_types_json', 'input_pins_json',
            'output_pins_json', 'pin_loads_json', 'parasitic_caps_json',
            'transistor_count']


def hdr(t):
    print('\n' + '=' * 78)
    print(t)
    print('=' * 78)


def load_arcs():
    print('读 31 片 arcs（circuit_id/switching_pin/direction/output/DELAY）...')
    parts = []
    for p in ARCS:
        d = pd.read_parquet(p, columns=['circuit_id', 'switching_pin',
                                        'direction', 'output', 'DELAY'])
        parts.append(d)
        print('  %s  %d 行' % (os.path.basename(p), len(d)), end='\r')
    df = pd.concat(parts, ignore_index=True)
    print('\n  合计 %d 行 / %d 电路' % (len(df), df['circuit_id'].nunique()))
    return df


def arc_sig(g):
    s = g[['switching_pin', 'direction', 'output', 'DELAY']].astype(str)
    s = s.sort_values(['switching_pin', 'direction', 'output', 'DELAY'])
    h = hashlib.md5('|'.join(s.agg(', '.join, axis=1)).encode()).hexdigest()[:10]
    return pd.Series({'mean_delay': float(g['DELAY'].mean()), 'n_arc': len(g), 'arc_sig': h})


def buckets(v):
    v = np.asarray(v, dtype=float)
    if len(v) == 0:
        return None
    return {
        '== 0': float((v == 0).mean()),
        '(0,1%)': float(((v > 0) & (v < 0.01)).mean()),
        '1-5%': float(((v >= 0.01) & (v < 0.05)).mean()),
        '5-20%': float(((v >= 0.05) & (v < 0.20)).mean()),
        '>=20%': float((v >= 0.20).mean()),
        'n': len(v),
    }


def crit1(dfx):
    """缺陷 4 ①：组内「与组内最优差 >10%」的变体数 ≥2 的组占比。"""
    n_ok = n = 0
    for _, g in dfx.groupby('expr'):
        d = g['mean_delay'].to_numpy()
        if len(d) < 2:
            continue
        n += 1
        best = d.min()
        if int((d > 1.10 * best).sum()) >= 2:
            n_ok += 1
    return n_ok / max(n, 1)


QUOTA = {'1x1': 300, '2x1': 400, '2x2': 300, '2x3': 250, '3x1': 400, '3x2': 350,
         '4x1': 400, '4x3': 400, '5x1': 800, '5x2': 900, '5x5': 300, '7x4': 550,
         '8x1': 300, '8x3': 650, '8x4': 500, '9x1': 450, '9x6': 350, '16x1': 300}


def sec7_quota(m, ded):
    """形状配额：名义口径 vs 「可分设计」口径（规格 quota 见 DATA_SPEC_V2 §四）。"""
    hdr('7. 形状配额在两种口径下的读数')
    a = m.groupby('shape').size()
    b = ded.groupby('shape').size()
    print('  %-7s %8s %10s %10s %8s %s' % ('shape', '配额', '交付', '有效设计', '交付判定', '有效判定'))
    nf_raw = nf_ded = 0
    for k, q in QUOTA.items():
        na, nb = int(a.get(k, 0)), int(b.get(k, 0))
        ja, jb = na >= q, nb >= q
        nf_raw += (not ja)
        nf_ded += (not jb)
        print('  %-7s %8d %10d %10d %8s %s' % (
            k, q, na, nb, '✅' if ja else '❌ -%d' % (q - na),
            '✅' if jb else '❌ -%d' % (q - nb)))
    print('  ⇒ 不合格形状数：名义口径 %d/18；「可分设计」口径 %d/18'
          % (nf_raw, nf_ded))
    print('  ⚠ 1x1 在两种口径下都是 0（形状完全缺失）')


def _sec5(m, ded):
    e = ded.groupby('expr').size()
    n_all = m.groupby('expr').size()
    print('  原始组大小:   %s' % dict(n_all.value_counts().sort_index()))
    print('  去重后组大小: %s' % dict(e.value_counts().sort_index()))
    print('  去重后 <10 变体的组 = %d/%d' % (int((e < 10).sum()), len(e)))
    print('  平均有效候选数 = %.2f（名义 %.2f）' % (e.mean(), n_all.mean()))
    print('\n（只读 data/，不写文件、不跑训练）')


def sec6_v2_contrast():
    """V2 三批是否也有「组内静态签名重复」—— 判「V3 引入」还是「这条线一直如此」。"""
    hdr('6. V2 对照：组内静态签名重复率（只读 static，无需 arcs）')
    print('  %-16s %7s %7s %9s %9s %10s' % ('数据集', '组', '电路', '净list重复',
                                             '签名重复', '真有效候选'))
    for d in ['batch_v2_full', 'batch_v2_rest', 'batch_v2_m4']:
        p = os.path.join(ROOT, 'data', d, 'circuit_static.parquet')
        if not os.path.exists(p):
            p = os.path.join(ROOT, 'data', d, 'circuit_static_part1.parquet')
        if not os.path.exists(p):
            print('  %-16s 缺 static' % d)
            continue
        s = pd.read_parquet(p)
        if 'expr' not in s.columns or 'gate_level_netlist' not in s.columns:
            print('  %-16s 无 expr/gate_level_netlist 列' % d)
            continue
        cols = [c for c in SIG_COLS if c in s.columns]
        s['sig'] = s[cols].astype(str).agg('|'.join, axis=1)
        ncirc = len(s)
        dup_nl = ncirc - s.groupby(['expr', 'gate_level_netlist']).ngroups
        dup_sig = ncirc - s.groupby(['expr', 'sig']).ngroups
        eff = s.drop_duplicates(subset=['expr', 'sig']).groupby('expr').size()
        print('  %-16s %7d %7d %8.1f%% %8.1f%% %10.2f'
              % (d, s['expr'].nunique(), ncirc,
                 100.0 * dup_nl / ncirc, 100.0 * dup_sig / ncirc, eff.mean()))
    print('  （「真有效候选」= 组内按静态全签名去重后的平均组大小）')


def main():
    st = pd.read_parquet(os.path.join(VDIR, 'circuit_static.parquet'))
    st['sig'] = st[SIG_COLS].astype(str).agg('|'.join, axis=1)
    st['cid'] = st['circuit_id'].astype(str)
    arcs = load_arcs()
    arcs['cid'] = arcs['circuit_id'].astype(str)

    hdr('1. 逐电路标量延迟 + 逐弧签名')
    per = arcs.groupby('cid', sort=False).apply(arc_sig, include_groups=False)
    print('  每电路行数分布: %s' % dict(per['n_arc'].value_counts().head(6)))
    m = st.merge(per, left_on='cid', right_index=True, how='left')
    miss = int(m['mean_delay'].isna().sum())
    print('  静态有、arcs 无的电路 = %d（应 0）' % miss)
    m = m.dropna(subset=['mean_delay'])

    hdr('2. 组内孪生：设计级 vs 同构异值')
    rows = []
    for expr, g in m.groupby('expr'):
        for sig, gg in g.groupby('sig'):
            if len(gg) < 2:
                continue
            d = gg['mean_delay'].to_numpy()
            rows.append({
                'expr': expr, 'n': len(gg),
                'n_distinct_arc': gg['arc_sig'].nunique(),
                'delay_spread_abs': float(d.max() - d.min()),
                'delay_spread_rel': float((d.max() - d.min()) / np.median(d)),
                'all_same_delay': bool(np.allclose(d, d[0], rtol=0, atol=0)),
            })
    t = pd.DataFrame(rows)
    n_twin_circ = int(t['n'].sum()) if len(t) else 0
    tot_circ = len(m)
    print('  孪生簇 %d 个，涉及电路 %d/%d = %.1f%%' % (len(t), n_twin_circ, tot_circ,
                                                      100.0 * n_twin_circ / tot_circ))
    if len(t):
        pure = t[(t['n_distinct_arc'] == 1) & t['all_same_delay']]
        same_arc = t[t['n_distinct_arc'] == 1]
        print('  其中 **真重复**（弧签名全同 且 延迟逐位相同）簇 %d 个，涉及电路 %d = %.1f%%'
              % (len(pure), int(pure['n'].sum()), 100.0 * int(pure['n'].sum()) / tot_circ))
        print('  弧签名全同（但延迟可能微差）     簇 %d 个，涉及电路 %d = %.1f%%'
              % (len(same_arc), int(same_arc['n'].sum()),
                 100.0 * int(same_arc['n'].sum()) / tot_circ))
        print('  孪生簇内延迟跨度的中位/均值 = %.3g / %.3g（绝对值，秒）'
              % (t['delay_spread_rel'].median(), t['delay_spread_rel'].mean()))
        print('  孪生簇内**延迟完全为 0 差**的簇占比 = %.1f%%'
              % (100.0 * t['all_same_delay'].mean()))

    hdr('3. 相邻差分解：恰好 0 vs (0,1%)')
    def adj_and_span(df_circ):
        """df_circ: 每行 = 一个电路（需含 expr / mean_delay）。返回相邻差列表与组跨度列表。"""
        A, S = [], []
        for _, g in df_circ.groupby('expr'):
            d = np.sort(g['mean_delay'].to_numpy())
            if len(d) < 2:
                continue
            A.extend(((d[1:] - d[:-1]) / np.maximum(d[:-1], 1e-30)).tolist())
            med = np.median(d)
            if med > 0:
                S.append((d[-1] - d[0]) / med)
        return np.array(A), np.array(S)

    a_raw, s_raw = adj_and_span(m)
    print('  全量 12,455 电路：')
    print('    相邻差 %s' % {k: (round(v, 4) if k != 'n' else v)
                             for k, v in buckets(a_raw).items()})
    print('    组内跨度中位 = %.4f；跨度 >10%% 的组 = %.1f%%'
          % (np.median(s_raw), 100.0 * (s_raw > 0.10).mean()))

    # 去重：每组内同静态签名只留一个代表
    ded = m.drop_duplicates(subset=['expr', 'sig'])
    a_ded, s_ded = adj_and_span(ded)
    print('  组内按静态签名去重后（%d 电路）：' % len(ded))
    print('    相邻差 %s' % {k: (round(v, 4) if k != 'n' else v)
                             for k, v in buckets(a_ded).items()})
    print('    组内跨度中位 = %.4f；跨度 >10%% 的组 = %.1f%%'
          % (np.median(s_ded), 100.0 * (s_ded > 0.10).mean()))

    hdr('4. 规格判据在两种口径下的读数')
    for lab, a, s, dfx in [('原始（含重复）', a_raw, s_raw, m), ('去重后', a_ded, s_ded, ded)]:
        b = buckets(a)
        c1 = crit1(dfx)
        print('  %-14s  <1%%=%.1f%%（其中恰好 0 = %.1f%%）  1-5%%=%.1f%%  5-20%%=%.1f%%  >20%%=%.1f%%  跨度中位=%.3f  ①>10%%差≥2个的组=%.1f%%'
              % (lab, 100 * (b['== 0'] + b['(0,1%)']), 100 * b['== 0'],
                 100 * b['1-5%'], 100 * b['5-20%'], 100 * b['>=20%'], np.median(s),
                 100.0 * c1))
    print('  规格目标：<1%=10-30%  1-5%=25-45%  5-20%=30-50%  >20%=3-10%  跨度中位=0.30-0.60  ①≥80%')

    hdr('5. 组内「有效候选数」（去重后）分布')
    _sec5(m, ded)
    sec7_quota(m, ded)
    sec6_v2_contrast()


if __name__ == '__main__':
    main()
