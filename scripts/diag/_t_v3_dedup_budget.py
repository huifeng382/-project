#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""「直接删掉 V3 组内重复」够不够？—— 去重 × MIN_GROUP_SIZE 过滤的联立后果（只读）。

背景：V3_ISSUES P0-1 说 37.2% 的电路是组内重复副本。最直觉的修法是「把重复行删掉」。
但训练侧有一条组大小过滤（`train_sweep.py:417-420`，口径 = `groupby('expr')['circuit_id'].nunique()`
≥ `MIN_GROUP_SIZE`，默认 10），它按**电路口径**数。删重复会同时把组变小 ⇒ 落到阈值以下的组
**整组被丢掉**。本脚本量出这一步的净后果。

同时做两个敏感性检查：
  ① 签名里含 / 不含 `transistor_count`。P1-1 已证该列与 netlist 不自洽（14.8% 的电路有假差异），
     **含它就等于放过了一批真重复** ⇒ 含它的读数（37.2%）是**下界**。
  ② 删重复后组大小分布 vs 阈值 10 的关系。

用法：python scripts/diag/_t_v3_dedup_budget.py
"""
import os
import glob
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
VDIR = os.path.join(ROOT, 'data', 'v3_delivery')
ARCS = sorted(glob.glob(os.path.join(VDIR, 'timing_arcs_part*.parquet')))

BASE = ['gate_level_netlist', 'cell_types_json', 'input_pins_json',
        'output_pins_json', 'pin_loads_json', 'parasitic_caps_json']
MIN_GROUP = int(os.environ.get('MIN_GROUP_SIZE', '10'))


def hdr(t):
    print('\n' + '=' * 78)
    print(t)
    print('=' * 78)


def rows_per_circuit():
    cnt = Counter()
    for p in ARCS:
        for b in pq.ParquetFile(p).iter_batches(batch_size=200000, columns=['circuit_id']):
            cnt.update(b.column('circuit_id').to_pandas().astype(str).value_counts().to_dict())
    return cnt


def report(st, rpc, cols, label):
    st = st.copy()
    st['cid'] = st['circuit_id'].astype(str)
    st['sig'] = st[cols].astype(str).agg('|'.join, axis=1)
    st['rows'] = st['cid'].map(rpc).fillna(0).astype(int)

    ded = st.drop_duplicates(subset=['expr', 'sig'])
    gs = ded.groupby('expr').size()
    keep = gs[gs >= MIN_GROUP].index
    ded_f = ded[ded['expr'].isin(keep)]

    hdr('%s  （签名 = %d 列%s）' % (label, len(cols),
                                   '' if 'transistor_count' in cols else '，不含 transistor_count'))
    n_rows_all, n_rows_ded, n_rows_f = int(st['rows'].sum()), int(ded['rows'].sum()), int(ded_f['rows'].sum())
    print('  %-22s %8s %9s %9s' % ('', '电路', '组', '行'))
    print('  %-22s %8d %9d %9d   （交付口径）' % ('① 交付原样', len(st), st['expr'].nunique(), n_rows_all))
    print('  %-22s %8d %9d %9d   （删重复后）' % ('② 去重', len(ded), ded['expr'].nunique(), n_rows_ded))
    print('  %-22s %8d %9d %9d   （再套 MIN_GROUP>=%d）' % (
        '③ 去重 + 过滤', len(ded_f), ded_f['expr'].nunique(), n_rows_f, MIN_GROUP))
    print('\n  ⇒ 只删重复：电路 %.1f%%、行 %.1f%% 留下来；组 %.1f%%'
          % (100.0 * len(ded) / len(st), 100.0 * n_rows_ded / n_rows_all,
             100.0 * ded['expr'].nunique() / st['expr'].nunique()))
    print('  ⇒ 再套组大小过滤：**组只剩 %.1f%%**（%d/%d 组被整组丢弃），行只剩 %.1f%%'
          % (100.0 * ded_f['expr'].nunique() / st['expr'].nunique(),
             st['expr'].nunique() - ded_f['expr'].nunique(), st['expr'].nunique(),
             100.0 * n_rows_f / n_rows_all))
    print('\n  去重后组大小分布: %s' % dict(sorted(Counter(gs).items())))
    print('  低于阈值 %d 的组 = %d/%d = %.1f%%；组大小中位 %.1f（名义 %.1f）'
          % (MIN_GROUP, int((gs < MIN_GROUP).sum()), len(gs),
             100.0 * (gs < MIN_GROUP).mean(), gs.median(),
             st.groupby('expr').size().median()))
    return ded, ded_f


def main():
    print('V3 去重预算体检（只读 data/，不写文件、不跑训练）')
    rpc = rows_per_circuit()
    print('  arcs 行数合计 = %d' % sum(rpc.values()))
    st = pd.read_parquet(os.path.join(VDIR, 'circuit_static.parquet'))

    d1, f1 = report(st, rpc, BASE + ['transistor_count'], '含 transistor_count（本次 V3_ISSUES 的口径）')
    d2, f2 = report(st, rpc, BASE, '不含 transistor_count（真设计同构）')

    hdr('对比：两套签名的差别')
    print('  含 transistor_count   ：去重后 %d 电路（重复 %d）' % (len(d1), len(st) - len(d1)))
    print('  不含 transistor_count ：去重后 %d 电路（重复 %d）' % (len(d2), len(st) - len(d2)))
    print('  ⇒ 差 %d 个电路 = %.1f%% —— 这批是「同设计被 P1-1 的 transistor_count 假差异拆开」的'
          % (len(d1) - len(d2), 100.0 * (len(d1) - len(d2)) / len(st)))
    print('     即 V3_ISSUES 里 37.2%% 是**下界**，真重复率是 %.1f%%'
          % (100.0 * (len(st) - len(d2)) / len(st)))
    print('\n（只读 data/，不写文件、不跑训练）')


if __name__ == '__main__':
    main()
