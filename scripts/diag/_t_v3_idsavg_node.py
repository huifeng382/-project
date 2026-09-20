#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P1-4 定论探针：`ids_avg` 的**节点级**零占比（V3 与 V2 同口径对照）。只读。

背景（docs/V3_ISSUES.md P1-4）：已测的是**条目级** —— V3 part01 全量 49.0%、part16 57.0%、
等距抽样 31 片 54.7%；V2 对照 batch_v2_full 26.2% / batch_v2_m4 49.6%。
但 data_loader **不是**按条目喂特征，而是**按 gate 取该门全部晶体管的均值**
（src/data_loader.py:643-665）：

    g = str(td.get('gate','')).lower()      # td = 一个晶体管的波形 dict
    gate_agg[g][f].append(float(td[f]))
    ...
    tw_feat[i, fi] = arr.mean()             # 节点 i 对应门 g

⇒ 一个门的 `ids_avg` **只在它全部晶体管都为 0 时才为 0**，所以节点级零占比必然 ≤ 条目级。
原文档 L510 原话：「本报告**没有测节点级**……在测出节点级零占比、并与 V2 结点级对照之前，
**不把它写成「特征死了」**」。本脚本补的就是这个数。

★ 四个口径必须分清，**不可互读**：
  ① 条目级 = 全部晶体管条目里 `ids_avg == 0` 的比例（= 已有读数，本脚本同样本复算作对照）
  ② 节点级 = 「一个有 wave 数据的门」其全部晶体管 `ids_avg` **均值 == 0** 的比例
  ③ 行级   = 一行（circuit, switching_pin, direction, output, corner）里**所有门**都为 0 的比例
  ④ 特征级 = 非零节点数 / **电路总门数**。这才是模型真正看到的那一维的零占比 ——
     ② 只在一件事成立时才等于 ④：**wave 覆盖了电路全部门**。
     本脚本用 `circuit_static.gate_level_netlist` 数门（口径同 `data_loader.py:319`：
     `len([l for l in nl.split('\n') if l.strip().startswith('X_')])`），实测覆盖率并给出 ④。

抽样：每片**整片等距**抽 S 行。⚠ 绝不能用 `head(n)` —— 见 V3_ISSUES §3 自查记录 1：
分片按电路顺序写，头部几十行常只属于一两路电路，实测同一列 head 与等距抽样差过 50 个百分点。
样本量随每一条结论一并打印，**本脚本全部是抽样口径，不是全量**。

用法：
  python scripts/diag/_t_v3_idsavg_node.py --probe      # 只打印一行 wave 的结构（先跑这个）
  python scripts/diag/_t_v3_idsavg_node.py --sample 400 # 正式读数
"""
import os
import sys
import json
import glob
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# config.WAVE_FIELDS 的默认值（本探针只关心 ids_avg，其余两个顺带报，便于交叉印证）
FIELDS = ['ids_avg', 'ids_peak', 'vds_swing']
FOCUS = 'ids_avg'

DATASETS = [
    ('V3 v3_delivery', os.path.join('data', 'v3_delivery'), 'timing_arcs_part*.parquet'),
    ('V2 batch_v2_full', os.path.join('data', 'batch_v2_full'), 'timing_arcs.parquet'),
    ('V2 batch_v2_m4', os.path.join('data', 'batch_v2_m4'), 'timing_arcs.parquet'),
]

SAMPLE = 400
PROBE = '--probe' in sys.argv
if '--sample' in sys.argv:
    SAMPLE = int(sys.argv[sys.argv.index('--sample') + 1])

_NGATES = {}


def hdr(t):
    print('\n' + '=' * 78)
    print(t)
    print('=' * 78)


def load_ngates(ddir):
    """cid -> 电路总门数。口径 = data_loader.py:319（数以 'X_' 开头的行）。"""
    if ddir in _NGATES:
        return _NGATES[ddir]
    path = os.path.join(ROOT, ddir, 'circuit_static.parquet')
    out = {}
    if os.path.exists(path):
        names = pq.ParquetFile(path).schema_arrow.names
        # 与 data_loader.py:80-82 一致：优先标准化网表
        col = 'gate_level_netlist_std' if 'gate_level_netlist_std' in names \
            else ('gate_level_netlist' if 'gate_level_netlist' in names else None)
        if col:
            df = pd.read_parquet(path, columns=['circuit_id', col])
            for cid, nl in zip(df['circuit_id'].astype(str), df[col]):
                out[cid] = len([l for l in str(nl).split('\n')
                                if l.strip().startswith('X_')])
    _NGATES[ddir] = out
    return out


def sample_rows(p, cols, n):
    """在**整片内等距**抽 n 行（口径同 _t_v3_col_health.py:49-60）。"""
    pf = pq.ParquetFile(p)
    step = max(1, pf.metadata.num_rows // max(n, 1))
    out = []
    for b in pf.iter_batches(batch_size=step, columns=cols):
        out.append(b.to_pandas().iloc[[0]])
    return pd.concat(out, ignore_index=True)


def parse(tw):
    """→ (条目级 {field: [值]}, 节点级 {gate: {field: 均值}})。

    严格照抄 loader 的聚合：分组键 = `str(td['gate']).lower()`；门均值 = 该门全部晶体管
    **非 None 值**的算术平均（loader :656 `if v is not None` ⇒ None 条目不入列表；
    loader :663 `if not vals: continue` ⇒ 一个都没取到的门不写特征、保持 0）。
    """
    if isinstance(tw, str):
        try:
            tw = json.loads(tw)
        except Exception:                                          # noqa: BLE001
            return {}, {}
    if not isinstance(tw, dict):
        return {}, {}
    item = defaultdict(list)
    gate = defaultdict(lambda: defaultdict(list))
    for _, td in tw.items():
        if not isinstance(td, dict):
            continue
        g = str(td.get('gate', '')).lower()
        for f in FIELDS:
            v = td.get(f)
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            item[f].append(fv)
            if g:
                gate[g][f].append(fv)
    node = {}
    for g, d in gate.items():
        node[g] = {f: float(np.mean(v)) for f, v in d.items() if v}
    return item, node


def run(label, ddir, pat, sample):
    files = sorted(glob.glob(os.path.join(ROOT, ddir, pat)))
    if not files:
        print('\n--- %s ---  ⚠ 找不到文件（%s/%s），跳过' % (label, ddir, pat))
        return None
    hdr('%s   （%d 片，每片等距 %d 行）' % (label, len(files), sample))

    ngates = load_ngates(ddir)
    item_n = defaultdict(int)
    item_z = defaultdict(int)
    node_n = defaultdict(int)
    node_z = defaultdict(int)
    node_vals = defaultdict(list)          # 节点级非零值，看分布（尤其 vds_swing 的集中度）
    gate_per_row = []
    nrow = 0
    nrow_nowave = 0                        # 有行但 wave 空/解析不出 → 该行全部节点槽位为 0
    row_allzero = 0
    tot_slots = 0                          # Σ 电路总门数（模型看到的节点槽位总数）
    tot_wave_gates = 0                     # Σ 有 wave 的门数
    cover_pairs = []                       # (总门数, 有 wave 门数) 用于覆盖率配对比较

    for p in files:
        names = pq.ParquetFile(p).schema_arrow.names
        if 'transistor_wave_json' not in names:
            print('  ⚠ %s 无 transistor_wave_json 列，跳过' % os.path.basename(p))
            continue
        cols = ['circuit_id', 'transistor_wave_json']
        df = sample_rows(p, [c for c in cols if c in names], sample)
        for cid, v in zip(df['circuit_id'].astype(str), df['transistor_wave_json']):
            n_g = ngates.get(cid)
            if n_g is None:
                continue
            nrow += 1
            tot_slots += n_g
            item, node = parse(v)
            if not node:
                nrow_nowave += 1
                continue
            gate_per_row.append(len(node))
            tot_wave_gates += len(node)
            cover_pairs.append((n_g, len(node)))
            for f in FIELDS:
                vals = item.get(f, [])
                item_n[f] += len(vals)
                item_z[f] += int(sum(1 for x in vals if x == 0))
            for g, d in node.items():
                for f, mv in d.items():
                    node_n[f] += 1
                    if mv == 0:
                        node_z[f] += 1
                    else:
                        node_vals[f].append(mv)
            if 'ids_avg' in node and all(
                    d.get(FOCUS, 0.0) == 0.0 for d in node.values()):
                row_allzero += 1

    if nrow == 0:
        print('  ⚠ 无有效样本')
        return None

    print('  有效行 = %d（其中 wave 空/解析不出 %d 行）；总节点槽位 Σ门数 = %d'
          % (nrow, nrow_nowave, tot_slots))
    print('  每行「有 wave 数据的门数」 min/med/max = %s'
          % ('-' if not gate_per_row else '%d/%d/%d' % (
              int(np.min(gate_per_row)), int(np.median(gate_per_row)),
              int(np.max(gate_per_row)))))
    eq = sum(1 for a, b in cover_pairs if a == b)
    print('  ★ **覆盖率**：有 wave 的门数 == 电路总门数 的行 = %d/%d = %.2f%%'
          % (eq, len(cover_pairs), 100.0 * eq / max(len(cover_pairs), 1)))
    print('     Σ有 wave 门数 / Σ总门数 = %d / %d = **%.2f%%**'
          % (tot_wave_gates, tot_slots, 100.0 * tot_wave_gates / max(tot_slots, 1)))
    print()
    print('  %-11s %14s %14s %14s %14s' % ('口径', '① 条目级', '② 节点级', '③ 行级全零', '④ 特征级'))
    for f in FIELDS:
        iz = 100.0 * item_z[f] / item_n[f] if item_n[f] else float('nan')
        nz = 100.0 * node_z[f] / node_n[f] if node_n[f] else float('nan')
        # ④ 特征级：分母换成**全部节点槽位**；wave 空的行整行为 0，自然计入
        nz_all = 100.0 * (node_z[f] + (tot_slots - node_n[f])) / max(tot_slots, 1)
        mark = '  ← P1-4 的主数' if f == FOCUS else ''
        print('  %-11s %13.1f%% %13.1f%% %13s %13.1f%%%s'
              % (f, iz, nz,
                 ('%.1f%%' % (100.0 * row_allzero / nrow)) if f == FOCUS else '-',
                 nz_all, mark))
    print()
    print('  ★ ② 节点级 vs ① 条目级：`%s` **%.1f%% → %.1f%%**（降 %.1f 个百分点，比值 %.2f×）'
          % (FOCUS, 100.0 * item_z[FOCUS] / max(item_n[FOCUS], 1),
             100.0 * node_z[FOCUS] / max(node_n[FOCUS], 1),
             100.0 * (item_z[FOCUS] / max(item_n[FOCUS], 1)
                      - node_z[FOCUS] / max(node_n[FOCUS], 1)),
             (item_z[FOCUS] / max(item_n[FOCUS], 1))
             / max(node_z[FOCUS] / max(node_n[FOCUS], 1), 1e-9)))
    print('  ★ ④ 特征级 = ② 再加上「缺 wave 的门槽位」—— 覆盖率 100%% 时两者相等')

    out = {'label': label, 'nrow': nrow, 'tot_slots': tot_slots,
           'tot_wave_gates': tot_wave_gates, 'cover_eq': eq,
           'item': {f: (item_z[f], item_n[f]) for f in FIELDS},
           'node': {f: (node_z[f], node_n[f]) for f in FIELDS},
           'row_allzero': row_allzero}
    for f in FIELDS:
        v = node_vals.get(f, [])
        if v:
            s = pd.Series(v)
            print('  %s 节点级**非零**取值: n=%d  min=%.4g  p10=%.4g  median=%.4g  '
                  'mean=%.4g  p90=%.4g  max=%.4g'
                  % (f, len(v), s.min(), s.quantile(.10), s.median(), s.mean(),
                     s.quantile(.90), s.max()))
            out.setdefault('dist', {})[f] = (len(v), float(s.median()), float(s.mean()))
    return out


def main():
    print('P1-4 节点级探针（只读 data/，不写文件、不跑训练）')
    print('聚合口径 = src/data_loader.py:643-665（按 gate 取该门全部晶体管 ids_avg 的均值）')
    print('门数口径 = src/data_loader.py:319（数 netlist 里以 X_ 开头的行）')

    if PROBE:
        p = sorted(glob.glob(os.path.join(
            ROOT, 'data', 'v3_delivery', 'timing_arcs_part*.parquet')))[0]
        df = sample_rows(p, ['transistor_wave_json'], 3)
        for i, v in enumerate(df['transistor_wave_json']):
            d = json.loads(v) if isinstance(v, str) else v
            print('\n--- 样本 %d：顶层类型 %s，条目数 %d' % (i, type(d).__name__, len(d)))
            if isinstance(d, dict):
                for j, (k, td) in enumerate(d.items()):
                    print('  键 %r → %s' % (k, json.dumps(td, ensure_ascii=False)[:160]))
                    if j >= 2:
                        break
        return

    res = []
    for label, ddir, pat in DATASETS:
        r = run(label, ddir, pat, SAMPLE)
        if r:
            res.append(r)

    hdr('★ 结论表：`ids_avg` 四级零占比（同一抽样口径下的横向对照）')
    print('  %-18s %8s %10s %10s %10s %10s' % (
        '数据集', '有效行', '①条目级', '②节点级', '③行全零', '④特征级'))
    for r in res:
        iz, iN = r['item'][FOCUS]
        nz, nN = r['node'][FOCUS]
        nzall = 100.0 * (nz + (r['tot_slots'] - nN)) / max(r['tot_slots'], 1)
        print('  %-18s %8d %9.1f%% %9.1f%% %9.1f%% %9.1f%%'
              % (r['label'], r['nrow'], 100.0 * iz / max(iN, 1),
                 100.0 * nz / max(nN, 1),
                 100.0 * r['row_allzero'] / r['nrow'], nzall))
    print()
    for r in res:
        print('  %-18s wave 覆盖率 %.2f%%（有 wave 门数==总门数 的行 %d/%d）'
              % (r['label'],
                 100.0 * r['tot_wave_gates'] / max(r['tot_slots'], 1),
                 r['cover_eq'], r['nrow']))
    print('\n  ⚠ 全部为**抽样**口径（每片等距 %d 行）；① 条目级一列应与 V3_ISSUES P1-4 的'
          ' 54.7%%（等距 31 片）在同一量级 —— 用来确认本探针没跑偏。' % SAMPLE)
    print('  ⚠ ①②③ 的分母是「有 wave 的门」；④ 的分母是「电路总门数」，含没有 wave 的门槽位。')
    print('     **模型看到的是 ④**；只有当 wave 覆盖全部门（本例实测如此）时 ② == ④。')
    print('\n（只读 data/，不写文件、不跑训练）')


if __name__ == '__main__':
    main()
