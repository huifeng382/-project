#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V3 组完整性与连接键体检（只读）。

两件事，都是「静默退化」类：

  1. **组内静态输入重复** —— 同一 expr 组内若两个变体的模型输入完全相同，它们
     之间的延迟差就是**不可学**的（模型无从区分），既白占行、又拉低可学信号。
     本段按「netlist 文本」和「静态输入全签名」两个粒度各测一次（精确，全量 12,455 行）。

  2. **wave → 节点 的连接键** —— `transistor_wave_json` 按 `gate` 字段（小写门实例名）
     聚合后贴到 `node_names` 上（`data_loader.py` 的 `gate_agg` / `gkey`）。若两侧名域
     对不上，特征**整块静默变 0**（该分支的 `except: pass` 什么都不会说）。
     同时测 transistor_wave 的内层键（`ids_avg` / `ids_peak` / `vds_swing`）是否真有值。

  3. **V2 对照** —— `slew_s` / `output_load_f` / `supply_noise_json` 在 V3 上是常量，
     测 V2 是否也如此（判「V3 引入」还是「这条数据线一直如此」）。

用法：python scripts/diag/_t_v3_group_integrity.py [--sample 300]
"""
import os
import sys
import json
import glob
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
VDIR = os.path.join(ROOT, 'data', 'v3_delivery')
STATIC = os.path.join(VDIR, 'circuit_static.parquet')
ARCS = sorted(glob.glob(os.path.join(VDIR, 'timing_arcs_part*.parquet')))

SAMPLE = 300
if '--sample' in sys.argv:
    SAMPLE = int(sys.argv[sys.argv.index('--sample') + 1])

SIG_COLS = ['gate_level_netlist', 'cell_types_json', 'input_pins_json',
            'output_pins_json', 'pin_loads_json', 'parasitic_caps_json',
            'transistor_count']


def hdr(t):
    print('\n' + '=' * 78)
    print(t)
    print('=' * 78)


def dut_instances(nl):
    """`.SUBCKT DUT … .ENDS` 块内 X_ 行的实例名（小写）—— 与 wave 的 gate 键同域。

    规格：跳过嵌套 .SUBCKT 定义与 M_ 行（它们不属于 DUT 主体）。"""
    names = []
    indut = False
    for raw in str(nl).replace('\r', '\n').split('\n'):
        s = raw.strip()
        if not s:
            continue
        u = s.upper()
        if u.startswith('.SUBCKT'):
            t = s.split()
            indut = len(t) > 1 and t[1].upper() == 'DUT'
            continue
        if u.startswith('.ENDS'):
            if indut:
                break
            continue
        if not indut or not u.startswith('X'):
            continue
        t = s.split()
        if len(t) >= 3:
            names.append(t[0].lower())
    return names


# ------------------------------------------------------- 1. 组内静态输入重复
def sec1(df):
    hdr('1. 组内静态输入重复（全量 12,455 行）')
    print('  判等粒度 ① netlist 文本  ② 静态输入全签名 %s' % (SIG_COLS,))
    df = df.copy()
    df['sig'] = df[SIG_COLS].astype(str).agg('|'.join, axis=1)

    rows = []
    for expr, g in df.groupby('expr'):
        n = len(g)
        rows.append((expr, n,
                     g['gate_level_netlist'].nunique(),
                     g['sig'].nunique()))
    r = pd.DataFrame(rows, columns=['expr', 'n', 'n_nl', 'n_sig'])
    print('\n  组数 = %d；总电路 = %d' % (len(r), r['n'].sum()))
    for col, lab in [('n_nl', 'netlist 去重后'), ('n_sig', '全签名去重后')]:
        d = r['n'] - r[col]
        dup_groups = int((d > 0).sum())
        dup_circ = int(d.sum())
        print('  %-16s: 有重复的组 %d/%d (%.1f%%)；多出的重复电路 %d/%d (%.1f%% 的电路有同组孪生)'
              % (lab, dup_groups, len(r), 100.0 * dup_groups / len(r),
                 dup_circ, r['n'].sum(), 100.0 * dup_circ / r['n'].sum()))
    # 最极端的组
    r['dup_nl'] = r['n'] - r['n_nl']
    r['dup_sig'] = r['n'] - r['n_sig']
    print('\n  重复最多的 8 个组（按全签名重复数）:')
    print(r.sort_values('dup_sig', ascending=False).head(8).to_string(index=False))
    print('\n  全签名重复数分布(dup_sig -> 组数): %s'
          % dict(sorted(Counter(r['dup_sig']).items())))

    # 跨 expr 的 netlist 复用（不同功能用了同一拓扑）
    x = df.groupby('gate_level_netlist')['expr'].nunique()
    print('\n  跨 expr 复用的 netlist：%d 种 netlist 出现在 >1 个 expr（占 %d 种 netlist 的 %.1f%%）'
          % (int((x > 1).sum()), df['gate_level_netlist'].nunique(),
             100.0 * float((x > 1).sum()) / df['gate_level_netlist'].nunique()))
    # circuit_id 是否跨组
    y = df.groupby('circuit_id')['expr'].nunique()
    print('  circuit_id 落在多个 expr 的个数 = %d（应 0）' % int((y > 1).sum()))
    return df


# ------------------------------------------- 2. wave 内层键 + 连接键名域
def read_arcs_sample(cols, n=SAMPLE):
    """每片在**整片内等距**取 n 行，凑够 n 行左右。

    ⚠ 不能取头部行：分片按电路顺序写，头部几十行常只属于一两路电路，
    同一列 head 抽样与等距抽样实测能差 50 个百分点。
    """
    out = []
    for p in ARCS:
        pf = pq.ParquetFile(p)
        step = max(1, pf.metadata.num_rows // max(n, 1))
        for b in pf.iter_batches(batch_size=step, columns=cols):
            out.append(b.to_pandas().iloc[[0]])
    return pd.concat(out, ignore_index=True)


def sec2(dfstatic):
    hdr('2. wave 内层键 + wave→节点 连接键（等距抽样）')
    cols = ['circuit_id', 'transistor_wave_json', 'slew_s', 'output_load_f',
            'supply_noise_json', 'corner']
    d = read_arcs_sample(cols)
    print('  样本行数 = %d（每片 %d 行等距 × %d 片）' % (len(d), SAMPLE, len(ARCS)))

    inner = defaultdict(list)
    inner_cov = Counter()
    n_entries = 0
    n_nondict = 0
    gate_hit = gate_miss = 0
    gate_miss_examples = []
    nl_of = dict(zip(dfstatic['circuit_id'].astype(str), dfstatic['gate_level_netlist']))
    inst_cache = {}
    for _, row in d.iterrows():
        try:
            tw = json.loads(row['transistor_wave_json'])
        except Exception:                                          # noqa: BLE001
            continue
        if not isinstance(tw, dict):
            continue
        cid = str(row['circuit_id'])
        if cid not in inst_cache:
            inst_cache[cid] = set(dut_instances(nl_of.get(cid, '')))
        inst = inst_cache[cid]
        for _, td in tw.items():
            n_entries += 1
            if not isinstance(td, dict):
                n_nondict += 1
                continue
            for k, v in td.items():
                inner_cov[k] += 1
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    inner[k].append(float(v))
            g = str(td.get('gate', '')).lower()
            if not g:
                continue
            if g in inst:
                gate_hit += 1
            else:
                gate_miss += 1
                if len(gate_miss_examples) < 5:
                    gate_miss_examples.append((cid, g))

    print('\n  transistor_wave_json 条目总数 = %d（非 dict 值 %d）' % (n_entries, n_nondict))
    print('  内层键覆盖: %s' % dict(inner_cov))
    print('  %-12s %9s %12s %12s %12s %9s' % ('内层键', '样本', 'min', 'median', 'max', '零占比'))
    for k, v in sorted(inner.items(), key=lambda kv: -len(kv[1])):
        a = np.array(v)
        print('  %-12s %9d %12.4g %12.4g %12.4g %8.1f%%'
              % (k, len(a), a.min(), np.median(a), a.max(), 100.0 * (a == 0).mean()))

    tot = gate_hit + gate_miss
    print('\n  ★ 连接键 `gate` vs netlist 实例名: 命中 %d / 落空 %d = **命中率 %.2f%%**'
          % (gate_hit, gate_miss, 100.0 * gate_hit / max(tot, 1)))
    if gate_miss_examples:
        print('    落空示例: %s' % (gate_miss_examples,))

    print('\n  V3 标量列常量性（同批样本）:')
    for c in ['slew_s', 'output_load_f']:
        v = pd.to_numeric(d[c], errors='coerce').dropna()
        print('    %-16s distinct=%d  min=%s max=%s' % (c, v.nunique(), v.min(), v.max()))
    sn = d['supply_noise_json'].dropna().astype(str)
    print('    %-16s distinct=%d  取值=%s' % ('supply_noise_json', sn.nunique(),
                                              list(sn.unique())[:3]))


# ---------------------------------------------------------------- 3. V2 对照
def sec3():
    hdr('3. V2 对照：slew_s / output_load_f / supply_noise 是否也恒定')
    targets = [('batch_v2_full', 'timing_arcs.parquet'),
               ('batch_v2_m4', 'timing_arcs.parquet'),
               ('batch_v2_rest', 'timing_arcs_part1.parquet')]
    for d, f in targets:
        p = os.path.join(ROOT, 'data', d, f)
        if not os.path.exists(p):
            print('  %-14s 缺文件 %s' % (d, f))
            continue
        want = ['slew_s', 'output_load_f', 'supply_noise_json',
                'direction', 'switching_pin']
        pf = pq.ParquetFile(p)
        names = pf.schema_arrow.names
        cols = [c for c in want if c in names]
        d2 = pd.concat([b.to_pandas() for b in
                        pf.iter_batches(batch_size=500, columns=cols)][:1],
                       ignore_index=True)
        print('  --- %s (%s，读 %d 行) ---' % (d, f, len(d2)))
        for c in ['slew_s', 'output_load_f']:
            if c in d2.columns:
                v = pd.to_numeric(d2[c], errors='coerce').dropna()
                print('    %-18s distinct=%d  min=%s max=%s  取值=%s'
                      % (c, v.nunique(), v.min(), v.max(), list(v.unique())[:4]))
        if 'supply_noise_json' in d2.columns:
            sn = d2['supply_noise_json'].dropna().astype(str)
            print('    %-18s distinct=%d  取值=%s' % ('supply_noise_json', sn.nunique(),
                                                      list(sn.unique())[:2]))


def main():
    print('V3 目录: %s' % VDIR)
    df = pd.read_parquet(STATIC)
    sec1(df)
    sec2(df)
    sec3()
    print('\n（本脚本只读 data/，不写文件、不跑训练）')


if __name__ == '__main__':
    main()
