#!/usr/bin/env python3
"""check_idsgnn_serve_parity.py — 校验 serve 端 idsavg GNN（模式 '3'）与训练时 dump 的 OOF 表
是否逐行一致。**这是跑 v2nowavegnn* 的 Rust shadow 前必须过的闸门**：

shadow 的数字只有在"serve 现场算出的那一列 == 训练时喂给 delay 模型的那一列"时才可解释；
否则测到的是"另一条特征路径 + delay 模型"的混合效果，得不出任何结论。

做法：对第 i 折的 ckpt 与第 i 折的 OOF 表，抽若干电路（默认 3 个），用**表内各行的真实 corner/
slew/load** 重建 serve 侧单行输入 → 前向 → 与表里的 pred_log1p 逐 (行,门) 对比。
（训练侧 dump 是 R×N 拼块前向；本模块是逐行前向 —— 这条对比同时验证"逐行 == 拼块"的等价性。）

用法：
  python3 scripts/diag/check_idsgnn_serve_parity.py                 # 5 折各抽 3 个电路
  python3 scripts/diag/check_idsgnn_serve_parity.py --ncirc 5 --tol 1e-4
  python3 scripts/diag/check_idsgnn_serve_parity.py --fold 2        # 只查第 2 折
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _idsgnn_serve import IdsGnnServer, DEFAULT_CKPT_GLOB   # noqa: E402

HOME = os.path.expanduser('~')
KEY = ['circuit_id', 'switching_pin', 'direction', 'output', 'corner']
ARC_COLS = KEY + ['slew_s', 'output_load_f']


def parse_corner(corner):
    """与 _fit_idsavg_gnn_server.parse_corner 逐字一致。"""
    try:
        s, l = str(corner).split('_')[:2]
        return float(s[1:].replace('p', '.')), float(l[1:].replace('p', '.'))
    except Exception:
        return 5.0, 10.0


def load_static(data_root, batches):
    """{circuit_id: (netlist, input_pins, output_pins)}（只取需要的三列）。"""
    frames = []
    for b in batches:
        p = os.path.join(data_root, b, 'circuit_static.parquet')
        if os.path.exists(p):
            frames.append(pd.read_parquet(p, columns=['circuit_id', 'gate_level_netlist',
                                                      'input_pins_json', 'output_pins_json']))
    if not frames:
        sys.exit(f'未找到任何 circuit_static.parquet（root={data_root}, batches={batches}）')
    d = pd.concat(frames, ignore_index=True).drop_duplicates('circuit_id').set_index('circuit_id')
    out = {}
    for cid, r in d.iterrows():
        try:
            ip = json.loads(r['input_pins_json']) if isinstance(r['input_pins_json'], str) else r['input_pins_json']
            op = json.loads(r['output_pins_json']) if isinstance(r['output_pins_json'], str) else r['output_pins_json']
        except Exception:
            continue
        out[cid] = (r['gate_level_netlist'], ip or [], op or [])
    return out


def arcs_for(data_root, batches, cid, keys):
    """该电路的 timing_arcs 行（slew_s/output_load_f），按 5 元键索引。"""
    got = {}
    want = set(keys)
    for b in batches:
        d0 = os.path.join(data_root, b)
        files = []
        one = os.path.join(d0, 'timing_arcs.parquet')
        if os.path.exists(one):
            files.append(one)
        files += sorted(glob.glob(os.path.join(d0, 'timing_arcs_part*.parquet')))
        for f in files:
            try:
                d = pd.read_parquet(f, columns=ARC_COLS, filters=[('circuit_id', '==', cid)])
            except Exception:
                continue
            if not len(d):
                continue
            arrs = [d[c].astype(str).to_numpy(dtype=object) for c in KEY]
            k = arrs[0]
            for a in arrs[1:]:
                k = k + '|' + a
            for kk, s, l in zip(k.tolist(), d['slew_s'].to_numpy(), d['output_load_f'].to_numpy()):
                if kk in want:
                    got[kk] = (float(s or 0), float(l or 0))
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-glob', default=os.environ.get('IDSGNN_CKPT', DEFAULT_CKPT_GLOB))
    ap.add_argument('--oof-glob', default=os.path.join(HOME, 'idsavg17', 'idsgnn_oof_f*.parquet'))
    ap.add_argument('--data-root', default=os.path.join(HOME, '-project', 'data'))
    ap.add_argument('--batches', default='batch_v2_full,batch_v2_rest,batch_v2_m4')
    ap.add_argument('--fold', type=int, default=-1, help='只查该折（默认全部）')
    ap.add_argument('--ncirc', type=int, default=3, help='每折抽几个电路')
    ap.add_argument('--tol', type=float, default=1e-4, help='逐 (行,门) |diff| 上限')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    ckpts = sorted(glob.glob(args.ckpt_glob))
    oofs = sorted(glob.glob(args.oof_glob))
    if not ckpts:
        sys.exit(f'ckpt glob 无命中: {args.ckpt_glob}')
    if not oofs:
        sys.exit(f'OOF glob 无命中: {args.oof_glob}')
    nfold = min(len(ckpts), len(oofs))
    print(f'[parity] ckpt {len(ckpts)} 个 / OOF {len(oofs)} 个 → 逐折配对比对前 {nfold} 折')

    batches = [b for b in args.batches.split(',') if b]
    static = load_static(args.data_root, batches)
    print(f'[parity] circuit_static 载入 {len(static)} 个电路 (root={args.data_root})')

    folds = range(nfold) if args.fold < 0 else [args.fold]
    rng = np.random.RandomState(args.seed)
    worst = 0.0
    total = 0
    failed = []
    for i in folds:
        srv = IdsGnnServer([ckpts[i]], device='cpu', verbose=False)
        d = pd.read_parquet(oofs[i], columns=KEY + ['gate', 'pred_log1p'])
        cids = d['circuit_id'].astype(str).unique()
        pick = rng.choice(cids, size=min(args.ncirc, len(cids)), replace=False).tolist()
        d = d[d['circuit_id'].astype(str).isin(pick)]
        n_cmp = 0
        f_max = 0.0
        for cid in pick:
            if cid not in static:
                print(f'  fold{i} {cid}: 静态表缺失，跳过')
                continue
            netlist, ip, op = static[cid]
            sub = d[d['circuit_id'].astype(str) == cid]
            keys = ['|'.join(str(x) for x in t) for t in zip(*[sub[c].astype(str) for c in KEY])]
            arcs = arcs_for(args.data_root, batches, cid, set(keys))
            # 用表内各行真实 corner/slew/load 组行（训练口径：slew_s>0 用其值，否则回落到 corner）
            rows = []
            for t in zip(*[sub[c].astype(str) for c in KEY]):
                c_slew, c_load = parse_corner(t[4])
                s, l = arcs.get('|'.join(t), (0.0, 0.0))
                rows.append({'switching_pin': t[1], 'direction': t[2], 'output': t[3],
                             'row_slew': (s if s > 0 else c_slew * 1e-12),
                             'row_load': (l if l > 0 else c_load * 1e-15),
                             'c_slew': c_slew, 'c_load': c_load})
            if not rows:
                continue
            names, ns, adj, radj, edges = srv.build_graph(netlist, ip, op)
            per = srv.ids_avg_rows(ns, adj, radj, edges, names, rows)
            idx = {str(n).lower(): j for j, n in enumerate(names)}
            ref = np.array(sub['pred_log1p'].to_numpy(dtype=np.float64))
            gates = [str(g).lower() for g in sub['gate'].tolist()]
            pred = np.array([per[k][idx[g]] for k, g in enumerate(gates)], dtype=np.float64) \
                if all(g in idx for g in gates) else None
            if pred is None:
                miss = [g for g in gates if g not in idx]
                print(f'  fold{i} {cid}: 表内门名不在图里 {miss[:3]}，跳过')
                continue
            diff = np.abs(pred - ref)
            f_max = max(f_max, float(diff.max()))
            n_cmp += len(diff)
            if diff.max() > args.tol:
                failed.append((i, cid, float(diff.max())))
        total += n_cmp
        worst = max(worst, f_max)
        print(f'  fold{i}: {len(pick)} 电路 / {n_cmp} 个 (行,门) 对比 | max|diff| = {f_max:.3e} '
              f'{"OK" if f_max <= args.tol else "**超限**"}')

    print(f'\n[parity] 合计 {total} 个 (行,门) 对比，全局 max|diff| = {worst:.3e}（上限 {args.tol:g}）')
    if failed or worst > args.tol:
        for i, cid, mx in failed[:5]:
            print(f'  ✗ fold{i} {cid}: max|diff|={mx:.3e}')
        sys.exit('parity 失败 → serve 端特征构造与训练不一致，shadow 数字不可解释（先修，别跑 shadow）')
    print('  OK —— serve 端 idsavg GNN 与训练 OOF 表逐位一致，模式 \'3\' 可用于 Rust shadow')


if __name__ == '__main__':
    main()
