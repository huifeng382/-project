#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V3 列健康度体检（只读）。

目的：把「某列在 V3 上整体为 0 / 恒定 / 全空」这类**静默退化**一次列全，供
docs/V3_ISSUES.md 引用。静默退化的危害在于：不报错、不告警，只是某一维特征
永远学不到东西 —— 事后归因时会被误读成「模型不行」。

分五段：
  A  static 全列：空值、空容器（[]/{}）、常数、数值零占比。（static 只 7.6 MB，整表读）
  B  static 语义列：tier / shape / split / candidate_idx / expr 的取值分布
  C  arcs 标量列：用 parquet 行组统计**精确**判「全 0 / 恒定」（不读数据，扫全 31 片）
  D  arcs JSON 列：抽样逐键统计（键数、零值占比、取值范围）—— 其中 transistor_wave_json
     的 ids_avg/ids_peak/vds_swing 是**默认开启**的节点特征（USE_TRANSISTOR_WAVE=1）
  E  arcs 31 片 schema 一致性

⚠ 口径：D 段是**抽样**（每片 SAMPLE 行），不是全量；C 段是全量（行组统计）。
   抽样结论一律标注样本量，不当全量结论用。

用法：python scripts/diag/_t_v3_col_health.py [--sample 400]
"""
import os
import sys
import json
import glob
from collections import defaultdict

import pandas as pd
import pyarrow.parquet as pq

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
VDIR = os.path.join(ROOT, 'data', 'v3_delivery')
STATIC = os.path.join(VDIR, 'circuit_static.parquet')
ARCS = sorted(glob.glob(os.path.join(VDIR, 'timing_arcs_part*.parquet')))

SAMPLE = 400
if '--sample' in sys.argv:
    SAMPLE = int(sys.argv[sys.argv.index('--sample') + 1])

EMPTYISH = {'', '[]', '{}', 'null', 'None', '[{}]', '[[]]'}


def hdr(t):
    print('\n' + '=' * 78)
    print(t)
    print('=' * 78)


def sample_rows(p, cols, n):
    """在**整片内等距**抽 n 行。

    ⚠ 不能用 `head(n)`：分片是按电路顺序写的，头部几十行常常只属于一两路电路，
    实测同一列用 head 抽样与等距抽样能差出 50 个百分点（ids_avg 零占比 57.5% vs 0.0%）。
    """
    pf = pq.ParquetFile(p)
    step = max(1, pf.metadata.num_rows // max(n, 1))
    out = []
    for b in pf.iter_batches(batch_size=step, columns=cols):
        out.append(b.to_pandas().iloc[[0]])
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- A / B
def part_a_b():
    df = pd.read_parquet(STATIC)
    hdr('A. static 全列体检（%d 行 × %d 列，整表读）' % (len(df), len(df.columns)))
    print('%-24s %-10s %8s %9s %9s  %s' % ('列', 'dtype', '空值', '空容器', '常数?', '零/说明'))
    for c in df.columns:
        s = df[c]
        n_null = int(s.isna().sum())
        kind = str(s.dtype)
        empty = -1
        note = ''
        if s.dtype == object:
            ss = s.dropna().astype(str)
            empty = int(ss.isin(EMPTYISH).sum())
            nu = ss.nunique()
            if nu == 1:
                note = '常量=%r' % (ss.iloc[0][:40],)
            else:
                note = 'distinct=%d' % nu
            mean_len = float(ss.str.len().mean()) if len(ss) else 0.0
            note += '  平均长=%.0f' % mean_len
        else:
            nun = int(s.nunique(dropna=True))
            z = float((s == 0).sum()) / max(len(s), 1)
            if nun == 1:
                note = '★常量=%r' % (s.dropna().iloc[0] if s.notna().any() else None,)
            else:
                note = 'distinct=%d  min=%s max=%s  零占比=%.1f%%' % (
                    nun, s.min(), s.max(), 100 * z)
        print('%-24s %-10s %8d %9s %9s  %s' % (
            c, kind[:10], n_null, ('' if empty < 0 else empty), '', note))

    hdr('B. static 语义列取值分布')
    for c in ['tier', 'shape', 'split', 'candidate_idx', 'expr']:
        if c not in df.columns:
            print('  %-14s 缺列' % c)
            continue
        s = df[c]
        if c == 'expr':
            print('  %-14s 不同 expr=%d；每 expr 电路数 min/med/max = %d/%d/%d' % (
                c, s.nunique(), *[int(x) for x in
                                  s.value_counts().agg(['min', 'median', 'max']).tolist()]))
        elif c == 'candidate_idx':
            print('  %-14s min=%s max=%s 每电路候选数(按 circuit_id 数): %s' % (
                c, s.min(), s.max(),
                df.groupby('circuit_id').size().value_counts().to_dict()))
        else:
            vc = s.value_counts(dropna=False)
            print('  %-14s %s' % (c, vc.to_dict()))

    hdr('B2. 交叉：tier × 电路数 / 行数')
    if 'tier' in df.columns:
        g = df.groupby('tier').agg(电路=('circuit_id', 'nunique'), 行=('circuit_id', 'size'))
        g['行占比'] = (g['行'] / g['行'].sum() * 100).round(1)
        print(g.to_string())
    hdr('B3. 交叉：split × 电路数 / 行数（交付列，训练不使用，见 DIFF §17.8）')
    if 'split' in df.columns:
        g = df.groupby('split').agg(电路=('circuit_id', 'nunique'), 行=('circuit_id', 'size'))
        g['行占比'] = (g['行'] / g['行'].sum() * 100).round(1)
        print(g.to_string())
        # 与 metadata 自评对照
        try:
            meta = json.load(open(os.path.join(VDIR, 'metadata.json')))
            sr = meta.get('dataset', {}).get('split_rows', {})
            print('  metadata 自评 split_rows =', sr, '（行数合计 %d）' % sum(sr.values()))
        except Exception as e:                                    # noqa: BLE001
            print('  (metadata 读取失败: %s)' % e)
    return df


# ---------------------------------------------------------------- C
def part_c():
    hdr('C. arcs 标量列：全量行组统计（31 片，精确判「全 0 / 恒定 / 有 null」）')
    agg = {}          # col -> dict
    for p in ARCS:
        pf = pq.ParquetFile(p)
        names = pf.schema_arrow.names
        md = pf.metadata
        for rg in range(md.num_row_groups):
            row = md.row_group(rg)
            for j in range(row.num_columns):
                col = row.column(j)
                st = col.statistics
                name = names[j]
                a = agg.setdefault(name, {'min': None, 'max': None, 'nulls': 0,
                                          'rows': 0, 'nopred': 0})
                a['rows'] += col.num_values
                a['nulls'] += (st.null_count if (st and st.has_null_count) else 0)
                if st is None or not st.has_min_max:
                    a['nopred'] += 1
                    continue
                mn, mx = st.min, st.max
                try:
                    if a['min'] is None or mn < a['min']:
                        a['min'] = mn
                    if a['max'] is None or mx > a['max']:
                        a['max'] = mx
                except TypeError:
                    a['nopred'] += 1
    print('%-24s %10s %9s %-22s %-22s %s' % ('列', '行', '空值', '全量 min', '全量 max', '判定'))
    for name, a in agg.items():
        mn, mx = a['min'], a['max']
        verdict = ''
        if mn is None:
            verdict = '（无统计可用，须实读）'
        else:
            try:
                if mn == mx:
                    verdict = '★常量 %r%s' % (mn, ' ← 全 0' if mn in (0, 0.0, b'') else '')
                elif isinstance(mn, (int, float)) and mn == 0 and mx == 0:
                    verdict = '★全 0'
            except TypeError:
                pass
        smn = repr(mn)[:20] if mn is not None else 'None'
        smx = repr(mx)[:20] if mx is not None else 'None'
        print('%-24s %10d %9d %-22s %-22s %s' % (name, a['rows'], a['nulls'], smn, smx, verdict))


# ---------------------------------------------------------------- D
def part_d():
    hdr('D. arcs JSON 列：等距抽样逐键统计（每片 %d 行，跨整片）' % SAMPLE)
    json_cols = ['transistor_wave_json', 'pin_slew_json', 'pin_load_json',
                 'supply_noise_json', 'gate_states_json']
    avail = pq.ParquetFile(ARCS[0]).schema_arrow.names
    json_cols = [c for c in json_cols if c in avail]
    inner = {c: defaultdict(list) for c in json_cols}   # 数值键值（含内层）
    nkeys = {c: defaultdict(int) for c in json_cols}
    nrow = 0
    for p in ARCS:
        df = sample_rows(p, json_cols, SAMPLE)
        nrow += len(df)
        for c in json_cols:
            for v in df[c]:
                if v is None:
                    continue
                try:
                    d = json.loads(v) if isinstance(v, str) else v
                except Exception:                                  # noqa: BLE001
                    continue
                if not isinstance(d, dict):
                    continue
                for k, val in d.items():
                    if isinstance(val, dict):
                        # 二层（transistor_wave_json 的条目是 per-transistor dict）：
                        # 统计按**内层字段名**汇总，否则键会变成 120 万个晶体管名。
                        for k2, v2 in val.items():
                            nkeys[c][k2] += 1
                            if isinstance(v2, (int, float)) and not isinstance(v2, bool):
                                inner[c][k2].append(float(v2))
                    else:
                        nkeys[c][k] += 1
                        if isinstance(val, (int, float)) and not isinstance(val, bool):
                            inner[c][k].append(float(val))
    print('样本行数 = %d（每片 %d 行等距 × %d 片）\n' % (nrow, SAMPLE, len(ARCS)))
    for c in json_cols:
        print('--- %s ---' % c)
        if not nkeys[c]:
            print('    （无可用样本）')
            continue
        ks = sorted(nkeys[c], key=lambda k: -nkeys[c][k])
        print('    %-38s %10s %10s %12s %12s %12s %9s' % (
            '键', '出现', '覆盖%', 'min', 'median', 'max', '零占比'))
        for k in ks[:10]:
            v = inner[c].get(k, [])
            if v:
                s = pd.Series(v)
                z = 100.0 * float((s == 0).sum()) / len(s)
                print('    %-38s %10d %9.1f%% %12.4g %12.4g %12.4g %8.1f%%' % (
                    k[:38], nkeys[c][k], 100.0 * nkeys[c][k] / nrow,
                    s.min(), s.median(), s.max(), z))
            else:
                print('    %-38s %10d %9.1f%% %12s' % (
                    k[:38], nkeys[c][k], 100.0 * nkeys[c][k] / nrow, '(非数值)'))
        if len(ks) > 10:
            print('    … 其余 %d 个键: %s' % (len(ks) - 10, ', '.join(k[:26] for k in ks[10:16])))


# ---------------------------------------------------------------- E
def part_e():
    hdr('E. arcs 31 片 schema 一致性')
    base = None
    bad = 0
    for p in ARCS:
        sch = pq.ParquetFile(p).schema_arrow
        sig = [(f.name, str(f.type)) for f in sch]
        if base is None:
            base = sig
            print('  基准 = %s' % os.path.basename(p))
            print('  列数 %d: %s' % (len(sig), ', '.join(n for n, _ in sig)))
        elif sig != base:
            bad += 1
            print('  ★不一致: %s' % os.path.basename(p))
    print('  共 %d 片；列名/类型不一致的片数 = %d' % (len(ARCS), bad))


def main():
    print('V3 目录: %s' % VDIR)
    print('static : %s' % os.path.basename(STATIC))
    print('arcs   : %d 片' % len(ARCS))
    part_a_b()
    part_c()
    part_d()
    part_e()
    print('\n（本脚本只读 data/，不写文件、不跑训练）')


if __name__ == '__main__':
    main()
