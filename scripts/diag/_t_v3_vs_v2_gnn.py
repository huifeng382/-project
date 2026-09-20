#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V3 vs V2：**跑 GNN 的层面**到底变了什么（只读）。

回答的问题：换到 V3 之后，喂进 GNN 的东西（图、样本、标签对）与 V2 相比变了什么。
不比缺陷，比「形状」：图多大/多深、每电路几个训练样本、组内有多少对可比的标签。

四个数据集各算一遍，**同一套口径**：
  v3_delivery / batch_v2_full / batch_v2_rest / batch_v2_m4

算六个量（全部与 GNN 直接相关）：
  ① 图规模：X_ 门实例数（≈节点数）、每条 X_ 行的输入脚数之和（≈边数）
  ② 图深度：DUT 内 X_ 宏级最长链（= 规格 §四口径去掉「只看到输出门」那一步，
     故跨数据集可比；V3 另给规格口径做对照）
  ③ 每电路行数（arcs 行数）—— 决定「一个电路贡献几个训练样本」
  ④ 组：expr 组数、组大小 —— 决定「一个组贡献几对可比的排序标签」
  ⑤ 形状分布 (N_in, N_out) —— 决定 batch 内 padding 与形状不平衡
  ⑥ 组内静态签名重复 —— 决定这些排序标签里有多少是「自己跟自己比」

用法：python scripts/diag/_t_v3_vs_v2_gnn.py
"""
import os
import json
import glob
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SIG_COLS = ['gate_level_netlist', 'cell_types_json', 'input_pins_json',
            'output_pins_json', 'pin_loads_json', 'parasitic_caps_json',
            'transistor_count']

DS = [
    ('v3_delivery',     'circuit_static.parquet',           'timing_arcs_part*.parquet'),
    ('batch_v2_full',   'circuit_static.parquet',           'timing_arcs.parquet'),
    ('batch_v2_rest',   'circuit_static_part1.parquet',     'timing_arcs_part1.parquet'),
    ('batch_v2_m4',     'circuit_static.parquet',           'timing_arcs.parquet'),
]


def rule(t):
    print('\n' + '=' * 78)
    print(t)
    print('=' * 78)


def pctl(v, q):
    return float(np.percentile(np.asarray(v, dtype=float), q))


def parse_dut(nl):
    """`.SUBCKT DUT … .ENDS` 块内的 X_ 行 → (out_net, in_nets, cell)。跳过嵌套定义与 M_。"""
    gates = []
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
            gates.append((t[-2], t[1:-2], t[-1]))
    return gates


def max_chain(gates):
    """最长 X_ 链（对**所有**门取 max，不限定在输出门上 ⇒ 跨数据集可比）。

    返回 (全门最长链, 到「真输出门」的最长链, 环计数)。"""
    prod = defaultdict(list)
    for i, (o, _, _) in enumerate(gates):
        prod[o].append(i)
    memo, cyc = {}, [0]

    def lvl(i, stack):
        if i in memo:
            return memo[i]
        if i in stack:
            cyc[0] += 1
            return 0
        stack.add(i)
        m = 0
        for n in gates[i][1]:
            for p in prod.get(n, ()):
                m = max(m, lvl(p, stack))
        stack.discard(i)
        memo[i] = m + 1
        return m + 1

    allv = [lvl(i, set()) for i in range(len(gates))]
    return (max(allv) if allv else 0), allv, cyc[0], prod


def load_rows(d, pat):
    """每电路 arcs 行数（只读 circuit_id 列）。"""
    cnt = Counter()
    n = 0
    for p in sorted(glob.glob(os.path.join(ROOT, 'data', d, pat))):
        for b in pq.ParquetFile(p).iter_batches(batch_size=200000,
                                                columns=['circuit_id']):
            s = b.column('circuit_id').to_pandas().astype(str)
            cnt.update(s.value_counts().to_dict())
            n += len(s)
    return cnt, n


def one(d, spath, apat):
    rule('%s' % d)
    sp = os.path.join(ROOT, 'data', d, spath)
    if not os.path.exists(sp):
        print('  缺 static: %s' % sp)
        return None
    st = pd.read_parquet(sp)
    print('  static: %s  %d 行 × %d 列' % (spath, len(st), len(st.columns)))
    has_expr = 'expr' in st.columns
    has_nl = 'gate_level_netlist' in st.columns
    print('  有 expr=%s  gate_level_netlist=%s  output_pins_json=%s'
          % (has_expr, has_nl, 'output_pins_json' in st.columns))
    if not has_nl:
        return None

    rows_per = {}
    try:
        rows_per, nrows = load_rows(d, apat)
        print('  arcs: %d 行（%s）' % (nrows, apat))
    except Exception as e:                                        # noqa: BLE001
        print('  arcs 读取失败（行数/电路数相关结论跳过）: %s' % e)

    recs = []
    for r in st.itertuples(index=False):
        d_ = r._asdict() if hasattr(r, '_asdict') else None
        nl = getattr(r, 'gate_level_netlist')
        g = parse_dut(nl)
        dall, allv, cyc, _ = max_chain(g)
        outs = set()
        if 'output_pins_json' in st.columns:
            try:
                outs = set(json.loads(getattr(r, 'output_pins_json')))
            except Exception:                                      # noqa: BLE001
                outs = set()
        to_out = [allv[i] for i, (o, _, _) in enumerate(g) if o in outs]
        dspec = max(to_out) if to_out else dall
        nedge = sum(len(x[1]) for x in g)
        nin = len(json.loads(getattr(r, 'input_pins_json'))) if 'input_pins_json' in st.columns else -1
        nout = len(outs)
        recs.append(dict(
            cid=str(getattr(r, 'circuit_id')),
            expr=str(getattr(r, 'expr')) if has_expr else '?',
            nx=len(g), nedge=nedge, depth=dall, depth_spec=dspec,
            trans=int(getattr(r, 'transistor_count')) if 'transistor_count' in st.columns else -1,
            shape='%dx%d' % (nin, nout) if nin >= 0 and outs else '?',
            rows=rows_per.get(str(getattr(r, 'circuit_id')), 0),
            cyc=cyc))
    if not recs:
        return None
    df = pd.DataFrame(recs)
    df = df[df['nx'] > 0]
    print('  解析出 X_ 门的电路 = %d（解析失败 %d）' % (len(df), len(recs) - len(df)))

    # ① 图规模
    print('\n  ① 图规模（GNN 的节点数 / 边数）')
    for c, lab in [('nx', 'X_ 门实例数（≈节点数）'), ('nedge', 'X_ 输入脚总数（≈边数）')]:
        v = df[c].to_numpy()
        print('     %-24s med %4.0f  p90 %5.0f  max %5d   (中位电路)' % (
            lab, np.median(v), pctl(v, 90), v.max()))
    # ② 深度
    print('\n  ② 图深度（X_ 宏级最长链 ⇒ GNN 感受野要几跳才覆盖全网）')
    for c, lab in [('depth', '最长链（全门 max，跨集可比）'),
                   ('depth_spec', '规格口径（限定到真输出门）')]:
        v = df[c].to_numpy()
        print('     %-28s p10 %2.0f  p50 %2.0f  p90 %2.0f  max %2d   占比>6 %5.1f%%  占比>3 %5.1f%%'
              % (lab, pctl(v, 10), pctl(v, 50), pctl(v, 90), v.max(),
                 100 * (v > 6).mean(), 100 * (v > 3).mean()))
    print('     环 = %d 处' % int(df['cyc'].sum()))
    # ③ 每电路行数
    if rows_per:
        rz = df[df['rows'] > 0]
        print('\n  ③ 每电路训练样本数（arcs 行数；时序弧口径）')
        print('     中位 %4.0f   p90 %5.0f   max %5d   min %3.0f   '
              'max/min 比 = %.0f×' % (
                  np.median(rz['rows']), pctl(rz['rows'], 90), rz['rows'].max(),
                  rz['rows'].min(), rz['rows'].max() / max(rz['rows'].min(), 1)))
        print('     总行数 = %d' % int(rz['rows'].sum()))
    # ④ 组
    if has_expr:
        gs = df.groupby('expr').size()
        print('\n  ④ 组（expr）：排序任务的分母单位')
        print('     组数 %d；组大小 中位 %.1f  min %d  max %d' % (
            len(gs), np.median(gs), gs.min(), gs.max()))
        # ⑤ 形状
        print('\n  ⑤ 形状分布 (N_in × N_out) —— 决定 batch padding 与形状不平衡')
        vc = df['shape'].value_counts()
        print('     形状数 %d；Top6: %s' % (
            len(vc), ', '.join('%s:%.1f%%' % (k, 100.0 * v / len(df)) for k, v in vc.head(6).items())))
        # ⑥ 组内重复
        cols = [c for c in SIG_COLS if c in st.columns]
        st2 = st.copy()
        st2['sig'] = st2[cols].astype(str).agg('|'.join, axis=1)
        st2['cid'] = st2['circuit_id'].astype(str)
        st2 = st2[st2['cid'].isin(set(df['cid']))]
        if has_expr:
            dup = len(st2) - st2.groupby(['expr', 'sig']).ngroups
            eff = st2.drop_duplicates(subset=['expr', 'sig']).groupby('expr').size()
            print('\n  ⑥ 组内静态签名重复（决定排序标签里有多少是「自己跟自己比」）')
            print('     重复电路 %d/%d = %.1f%%；去重后平均组大小 %.2f（名义 %.2f）'
                  % (dup, len(st2), 100.0 * dup / len(st2), eff.mean(),
                     st2.groupby('expr').size().mean()))
    return df


def main():
    print('V3 vs V2：跑 GNN 的层面（只读 data/，不写文件、不跑训练）')
    out = {}
    for d, s, a in DS:
        out[d] = one(d, s, a)

    rule('横向汇总')
    print('  %-15s %7s %6s %7s %7s %8s %9s %8s' % (
        '数据集', '电路', '组数', '节点med', '深度p50', '深度>6', '每电路行', '组内重复'))
    for d, _, _ in DS:
        x = out.get(d)
        if x is None:
            print('  %-15s （跳过）' % d)
            continue
        nrow = x['rows'].median() if x['rows'].max() > 0 else 0
        print('  %-15s %7d %6d %7.0f %7.0f %8.1f%% %8.0f %7s' % (
            d, len(x), x['expr'].nunique(), np.median(x['nx']),
            np.median(x['depth']), 100 * (x['depth'] > 6).mean(), nrow, '—'))


if __name__ == '__main__':
    main()
