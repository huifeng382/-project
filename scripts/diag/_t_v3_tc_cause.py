#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P1-1 归因探针：同一 `gate_level_netlist` 为何会对应多个 `transistor_count`？只读。

背景：`scripts/check_transistor_count.py` 实测 V3 371/5900 种 netlist 不自洽（14.8% 电路），
而 V2 两批全自洽。`docs/V3_RECONCILE_NOTES.md` round 10 明写生成方**已知**该列口径错过、
并在交付前用 `measure_delivered_trans.py`（逐 .tl 测）+ `patch_delivery_metric.py`
（服务器写回）修正过 ⇒ 本探针要看的是：**修完之后为什么还自相矛盾**。

三个竞争假设，各自给判据：
  H1 **宏体不在 netlist 文本里**：netlist 只写宏名，宏体在 `sc_expansion.json`/`cell_types_json`。
     ⇒ 同一文本、不同 `cell_types_json` ⇒ 真电路不同、transistor_count 不同是**对的**，
        错的是「netlist 文本被当成电路唯一标识」这个用法（我方签名/去重/门数都依赖它）。
  H2 **patch 未全量生效**：旧口径值与新口径值混在同一列。
     ⇒ 同一文本的多个值会呈**离散两簇**，且两簇比值接近旧/新口径的已知倍数。
  H3 **容差/取整**：值只差 1-2。
     ⇒ 极差分布集中在极小值。

判据全部基于 `data/v3_delivery/circuit_static.parquet`，无需仿真。

用法: python scripts/diag/_t_v3_tc_cause.py
"""
import os
import json
from collections import defaultdict, Counter

import pandas as pd
import pyarrow.parquet as pq

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PATH = os.path.join(ROOT, 'data', 'v3_delivery', 'circuit_static.parquet')


def macros_of(nl):
    """netlist 文本里出现的宏名集合（SC_ 开头），用来看文本是否只引用宏名、不带宏体。"""
    out = set()
    for tok in str(nl).replace('(', ' ').replace(')', ' ').replace(',', ' ').split():
        if tok.startswith('SC_'):
            out.add(tok)
    return out


def main():
    names = pq.ParquetFile(PATH).schema_arrow.names
    cols = [c for c in ['circuit_id', 'expr', 'candidate_idx', 'transistor_count',
                        'gate_level_netlist', 'cell_types_json', 'shape', 'tier']
            if c in names]
    df = pd.read_parquet(PATH, columns=cols)
    df['cid'] = df['circuit_id'].astype(str)
    nl = 'gate_level_netlist'
    print('列 = %s；行 = %d' % (', '.join(cols), len(df)))

    g = df.groupby(nl)
    nuniq = g['transistor_count'].nunique()
    bad = nuniq[nuniq > 1].index
    print('\n不自洽 netlist = %d / %d' % (len(bad), len(nuniq)))
    sub = df[df[nl].isin(bad)].copy()
    print('涉及电路 = %d / %d' % (sub['cid'].nunique(), df['cid'].nunique()))

    # ---------- H2/H3：极差分布 ----------
    spread = (g['transistor_count'].max() - g['transistor_count'].min())[bad]
    print('\n[H3] 极差分布：', ' '.join(
        '%s=%d' % (k, v) for k, v in
        Counter(spread.clip(upper=5)).most_common()))
    print('     极差 == 1 的 netlist = %d (%.1f%%)；<= 2 的 = %d (%.1f%%)'
          % ((spread == 1).sum(), 100.0 * (spread == 1).mean(),
             (spread <= 2).sum(), 100.0 * (spread <= 2).mean()))

    # ---------- H1：同一 netlist 内 cell_types_json / expr 是否也不同 ----------
    gs = df.groupby(nl)
    n_ct = gs['cell_types_json'].nunique()
    n_ex = gs['expr'].nunique()
    print('\n[H1] 不自洽 netlist 里：')
    print('     cell_types_json 也不同 = %d / %d (%.1f%%)'
          % ((n_ct[bad] > 1).sum(), len(bad), 100.0 * (n_ct[bad] > 1).mean()))
    print('     expr 也不同           = %d / %d (%.1f%%)'
          % ((n_ex[bad] > 1).sum(), len(bad), 100.0 * (n_ex[bad] > 1).mean()))
    print('     对照：自洽 netlist 里 expr 也不同 = %d / %d (%.1f%%)'
          % ((n_ex[nuniq == 1] > 1).sum(), (nuniq == 1).sum(),
             100.0 * (n_ex[nuniq == 1] > 1).mean()))

    # ---------- 文本是否含宏体 ----------
    one = df.drop_duplicates(nl)
    hit = one[nl].map(lambda s: len(macros_of(s)) > 0)
    print('\n[文本形态] 含 SC_ 宏名的 netlist = %d / %d (%.1f%%)'
          % (hit.sum(), len(one), 100.0 * hit.mean()))
    if hit.any():
        s = one.loc[hit.idxmax(), nl]
        print('     样例（前 400 字符）：')
        print('     ' + repr(s[:400]))

    # ---------- 逐条看几个不自洽样例 ----------
    print('\n[H1 逐条] 前 3 个不自洽 netlist 的 (expr, cand, tc, shape, cell_types) 组合：')
    for k, kk in enumerate(bad[:3]):
        d = df[df[nl] == kk]
        print('  --- netlist #%d（%d 电路）' % (k + 1, len(d)))
        for _, r in d.head(6).iterrows():
            try:
                ct = json.loads(r['cell_types_json'])
                ct = '%d 种: %s' % (len(ct), ','.join(sorted(ct)[:6]))
            except Exception:                                       # noqa: BLE001
                ct = str(r['cell_types_json'])[:60]
            print('     %-14s %-6s %-4s tc=%-5s shape=%-6s %s'
                  % (r['expr'], r['candidate_idx'], r['tier'], r['transistor_count'],
                     r['shape'], ct))

    # ---------- 反向：同 expr 内 tc 是否唯一 ----------
    ge = df.groupby('expr')['transistor_count'].nunique()
    print('\n[对照] 按 expr 分组：多值 = %d / %d' % ((ge > 1).sum(), len(ge)))
    gn = df.groupby(nl)['transistor_count'].nunique()
    print('[对照] 按 netlist 分组：多值 = %d / %d' % ((gn > 1).sum(), len(gn)))


if __name__ == '__main__':
    main()
