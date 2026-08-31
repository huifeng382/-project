"""③ ids_avg GBDT15 全量拟合 + 导出（16.11.4，方向 #11）
在全量数据(8679电路×多行)上拟合 GBDT15(15 特征),评估 R^2/Pearson/Spearman,
导出 joblib 模型供 data_loader 训练特征 + Rust 端树遍历。

用法: python scripts/diag/_fit_idsavg_gbdt15.py [--sample 1.0] [--out outputs/idsavg_gbdt15.joblib]
"""
import sys
import os
import glob
import json
import math
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
import joblib
from src.graph_builder import build_static_graph, rebuild_gate_types

DATA = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(DATA, 'outputs')

# 特征顺序(与 data_loader/serve 一致)
FEAT_NAMES = [
    'log1p(slew_ps)', 'log1p(load_fF)', 'log1p(drive)', 'log1p(parasitic)',
    'log1p(fanout)', 'log1p(h)', 'log1p(corner_slew)', 'log1p(corner_load)',
    'log1p(drive)*log1p(load)', 'log1p(drive)*log1p(h)',
    'log1p(fanout)*log1p(parasitic)', 'log1p(slew)*log1p(load)',
    'log1p(tau_rc_ps)', 'log1p(parasitic*fanout)', 'log1p(1/(1+fanout))',
]


def parse_corner(corner):
    try:
        s, l = str(corner).split('_')[:2]
        return float(s[1:].replace('p', '.')), float(l[1:].replace('p', '.'))
    except Exception:
        return 5.0, 10.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default=os.path.join(OUT, 'idsavg_gbdt15.joblib'))
    ap.add_argument('--test-frac', type=float, default=0.1)
    args = ap.parse_args()

    sp = os.path.join(DATA, 'data/batch_v2_full/circuit_static.parquet')
    sparts = sorted(glob.glob(os.path.join(DATA, 'data/batch_v2_full/circuit_static_part*.parquet')))
    sdf = pd.concat([pd.read_parquet(p) for p in ([sp] if os.path.exists(sp) else sparts)], ignore_index=True)
    sdf['circuit_id'] = sdf['circuit_id'].astype(str)
    sdf = sdf.drop_duplicates('circuit_id').set_index('circuit_id')

    dp = os.path.join(DATA, 'data/batch_v2_full/timing_arcs.parquet')
    dparts = sorted(glob.glob(os.path.join(DATA, 'data/batch_v2_full/timing_arcs_part*.parquet')))
    cols = ['circuit_id', 'transistor_wave_json', 'slew_s', 'output_load_f', 'corner']
    ddf = pd.concat([pd.read_parquet(p, columns=cols) for p in ([dp] if os.path.exists(dp) else dparts)], ignore_index=True)
    ddf['circuit_id'] = ddf['circuit_id'].astype(str)
    ddf = ddf[ddf['transistor_wave_json'].notna()]
    if args.sample < 1.0:
        rng = np.random.RandomState(args.seed)
        sel = set(rng.choice(ddf['circuit_id'].drop_duplicates().tolist(),
                             size=int(len(ddf['circuit_id'].unique()) * args.sample), replace=False))
        ddf = ddf[ddf['circuit_id'].isin(sel)]

    X, y = [], []
    t0 = time.time()
    for k, (_, r) in enumerate(ddf.iterrows()):
        cid = r['circuit_id']
        if cid not in sdf.index:
            continue
        try:
            wave = json.loads(r['transistor_wave_json']) if isinstance(r['transistor_wave_json'], str) else {}
        except Exception:
            continue
        if not isinstance(wave, dict) or not wave:
            continue
        gate_avg = {}
        for tv in wave.values():
            if not isinstance(tv, dict):
                continue
            g, v = tv.get('gate'), tv.get('ids_avg')
            if g is None or v is None:
                continue
            gate_avg.setdefault(str(g).lower(), []).append(float(v))
        if not gate_avg:
            continue
        try:
            srow = sdf.loc[cid]
            nl = srow['gate_level_netlist']
            ip = json.loads(srow['input_pins_json']) if isinstance(srow['input_pins_json'], str) else srow['input_pins_json']
            op = json.loads(srow['output_pins_json']) if isinstance(srow['output_pins_json'], str) else srow['output_pins_json']
            rebuild_gate_types(_cell_types(nl))
            node_names, ns, _ = build_static_graph(cid, nl, ip or None, op or None)
            pl_json = srow.get('parasitic_caps_json')
            pl = json.loads(pl_json) if isinstance(pl_json, str) else (pl_json or {})
            pl = {str(k).lower(): v for k, v in pl.items()} if isinstance(pl, dict) else {}
        except Exception:
            continue
        slew_s = float(r.get('slew_s', 0) or 0)
        load_f = float(r.get('output_load_f', 0) or 0)
        c_slew, c_load = parse_corner(r.get('corner'))
        row_slew = (slew_s if slew_s > 0 else c_slew * 1e-12)
        row_load = (load_f if load_f > 0 else c_load * 1e-15)
        f_slew = math.log1p(row_slew * 1e12)
        f_load = math.log1p(row_load * 1e15)
        f_cslew = math.log1p(c_slew)
        f_cload = math.log1p(c_load)
        for i, n in enumerate(node_names):
            gkey = str(n).lower()
            if gkey not in gate_avg:
                continue
            real = float(np.mean(gate_avg[gkey]))
            if real <= 0:
                continue
            drive = float(ns[i, 3])
            parasitic = float(ns[i, 4])
            fanout = float(np.expm1(ns[i, 1]))
            h = float(np.expm1(ns[i, 6]))
            f_drive = math.log1p(drive)
            f_par = math.log1p(parasitic)
            f_fan = math.log1p(fanout)
            f_h = math.log1p(h)
            r_on = 1.0 / max(drive, 1e-6)
            c_l = max(parasitic * 1e-15, 1e-18)
            tau_rc = r_on * c_l
            gate_par = 0.0
            pv = pl.get(gkey)
            if isinstance(pv, dict):
                gate_par = sum(float(x) for x in pv.values() if x is not None)
            elif pv is not None:
                try:
                    gate_par = float(pv)
                except Exception:
                    gate_par = 0.0
            X.append([
                f_slew, f_load, f_drive, f_par, f_fan, f_h, f_cslew, f_cload,
                f_drive * f_load, f_drive * f_h, f_fan * f_par, f_slew * f_load,
                math.log1p(tau_rc * 1e15), math.log1p(max(parasitic * fanout, 1e-9)),
                math.log1p(max(1.0 / (1.0 + fanout), 1e-9)),
            ])
            y.append(math.log1p(real))
        if (k + 1) % 2000 == 0:
            print(f'  行 {k+1}: 样本 {len(y)}, {time.time()-t0:.0f}s', flush=True)

    X, y = np.array(X), np.array(y)
    print(f'\n样本: {len(y)} (row,gate) 对', flush=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_frac, random_state=args.seed)

    print('拟合 GBDT15...', flush=True)
    gb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                       random_state=args.seed, validation_fraction=0.1,
                                       early_stopping=True, n_iter_no_change=40)
    gb.fit(Xtr, ytr)
    p = gb.predict(Xte)
    r2 = 1 - np.sum((yte - p) ** 2) / np.sum((yte - yte.mean()) ** 2)
    r = float(np.corrcoef(yte, p)[0, 1])
    rho, _ = spearmanr(yte, p)
    print(f'\nGBDT15 测试集 (n={len(yte)}):')
    print(f'  R^2 = {r2:.4f}  Pearson = {r:.4f}  Spearman = {rho:.4f}')
    print(f'  实际迭代数 = {gb.n_iter_}')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump({'model': gb, 'feat_names': FEAT_NAMES, 'r2': float(r2),
                 'pearson': r, 'spearman': float(rho), 'n_train': len(ytr),
                 'date': time.strftime('%Y-%m-%d')}, args.out)
    print(f'\n模型已存: {args.out}')
    print('FEAT_NAMES=' + json.dumps(FEAT_NAMES))


def _cell_types(nl):
    types = set()
    for line in (nl or '').split('\n'):
        s = line.strip()
        if s.startswith('X_') and len(s.split()) >= 3:
            types.add(s.split()[-1])
    return types


if __name__ == '__main__':
    main()
