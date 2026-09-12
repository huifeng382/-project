#!/usr/bin/env python3
"""17.1.3 诊断：校验 idsavg GNN 的 OOF 表 与 delay 侧数据的查表键是否对得上。

背景：delay 侧 data_loader 按 (circuit_id, switching_pin, direction, output, corner) 查表，
键对不上时该特征列大面积填 0 → 静默退化成 v2nowave，跑再久也读不出真结论。
本脚本独立于训练流程，直接对源 parquet 建键集求交集，用来在起训练前先排掉这个坑。

用法:
  python3 scripts/diag/check_ids_gnn_table.py                    # 自动找表与数据
  python3 scripts/diag/check_ids_gnn_table.py <表glob> <数据根目录>
"""
import collections
import glob
import os
import sys

import pandas as pd

HOME = os.path.expanduser('~')
COLS = ['circuit_id', 'switching_pin', 'direction', 'output', 'corner']


def keys_of(files, label):
    ks = set()
    for f in files:
        try:
            d = pd.read_parquet(f, columns=COLS)
        except Exception as e:
            print(f'  ! 跳过 {f}: {e}')
            continue
        arrs = [d[c].astype(str).to_numpy(dtype=object) for c in COLS]
        k = arrs[0]
        for a in arrs[1:]:
            k = k + '|' + a
        ks.update(k.tolist())
        print(f'  {label} {os.path.basename(f)}: {len(d)} 行')
    return ks


def main():
    tbl_glob = sys.argv[1] if len(sys.argv) > 1 else f'{HOME}/idsavg17/idsgnn_oof_f*.parquet'
    data_root = sys.argv[2] if len(sys.argv) > 2 else f'{HOME}/-project/data'

    tf = sorted(glob.glob(tbl_glob))
    print(f'[表] glob={tbl_glob} -> {len(tf)} 个文件')
    if not tf:
        sys.exit('未找到 OOF 表')
    oof = keys_of(tf, '表')

    # 数据按批次目录组织，规则同 _fit_idsavg_gnn_server.py L157-162:
    #   <root>/<batch>/timing_arcs.parquet  或  <root>/<batch>/timing_arcs_part*.parquet
    src = []
    for b in ('batch_v2_full', 'batch_v2_rest', 'batch_v2_m4'):
        d = os.path.join(data_root, b)
        one = os.path.join(d, 'timing_arcs.parquet')
        if os.path.exists(one):
            src.append(one)
        src += sorted(glob.glob(os.path.join(d, 'timing_arcs_part*.parquet')))
    print(f'[数据] root={data_root} -> {len(src)} 个 parquet')
    if not src:
        cand = sorted(glob.glob(os.path.join(data_root, '*', 'timing_arcs*.parquet')))
        print(f'  {data_root}/*/timing_arcs*.parquet 候选: {cand[:20]}')
        sys.exit('未找到 batch_v2_full/rest/m4 的 timing_arcs parquet（用第二个参数指定数据根目录）')
    dl = keys_of(src, '数据')

    miss = dl - oof
    hit = len(dl) - len(miss)
    print(f'\nOOF 去重键  = {len(oof)}')
    print(f'数据 去重键 = {len(dl)}')
    print(f'命中 = {hit} / {len(dl)} = {100.0 * hit / max(len(dl), 1):.2f}%   未命中 = {len(miss)}')
    if miss:
        for i, c in enumerate(COLS):
            top = collections.Counter(k.split('|')[i] for k in miss).most_common(5)
            print(f'  未命中按 {c} 分布 (top5): {dict(top)}')
        print(f'  样例: {list(miss)[:3]}')
    else:
        print('  OK 键完全对齐（数据键集 ⊆ 表键集）')


if __name__ == '__main__':
    main()
