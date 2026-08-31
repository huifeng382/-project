"""② ids_avg 近似升级评估（16.11.2）
对比线性 vs 非线性（GBDT/MLP）+ 特征扩充,在全量数据(8679电路×多行)上评估
R^2 / Pearson r,目标: 找更贴近真实 ids_avg 的非仿真拟合。

用法: python scripts/diag/_eval_idsavg_nonlinear.py [--sample 0.3]
"""
import sys
import os
import glob
import json
import math
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd
from src.graph_builder import build_static_graph

DATA = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VDD = 1.8


def parse_corner(corner):
    """s05p0_l10p0 -> (slew_s, load_f)"""
    try:
        s, l = str(corner).split('_')[:2]
        return float(s[1:].replace('p', '.')), float(l[1:].replace('p', '.'))
    except Exception:
        return 5.0, 10.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=float, default=0.3, help='电路采样比例(默认0.3≈2600电路)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    # ---- 静态 ----
    sp = os.path.join(DATA, 'data/batch_v2_full/circuit_static.parquet')
    sparts = sorted(glob.glob(os.path.join(DATA, 'data/batch_v2_full/circuit_static_part*.parquet')))
    sdf = pd.concat([pd.read_parquet(p) for p in ([sp] if os.path.exists(sp) else sparts)], ignore_index=True)
    sdf['circuit_id'] = sdf['circuit_id'].astype(str)
    sdf = sdf.drop_duplicates('circuit_id').set_index('circuit_id')
    print(f'静态: {len(sdf)} 电路', flush=True)

    # ---- 动态行（按电路采样）----
    dp = os.path.join(DATA, 'data/batch_v2_full/timing_arcs.parquet')
    dparts = sorted(glob.glob(os.path.join(DATA, 'data/batch_v2_full/timing_arcs_part*.parquet')))
    cols = ['circuit_id', 'transistor_wave_json', 'slew_s', 'output_load_f', 'corner']
    ddf = pd.concat([pd.read_parquet(p, columns=cols) for p in ([dp] if os.path.exists(dp) else dparts)], ignore_index=True)
    ddf['circuit_id'] = ddf['circuit_id'].astype(str)
    ddf = ddf[ddf['transistor_wave_json'].notna()]
    circuits = ddf['circuit_id'].drop_duplicates()
    rng = np.random.RandomState(args.seed)
    n_circ = int(len(circuits) * args.sample)
    sel = set(rng.choice(circuits.tolist(), size=n_circ, replace=False))
    ddf = ddf[ddf['circuit_id'].isin(sel)]
    print(f'动态: {len(ddf)} 行 / {len(sel)} 电路', flush=True)

    X, y = [], []          # 8 特征线性（与 16.9.4 对齐）
    Xe, ye = [], []        # 特征扩充(交互项)
    n_gates = 0
    n_skip = 0
    for _, r in ddf.iterrows():
        cid = r['circuit_id']
        if cid not in sdf.index:
            n_skip += 1
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
            node_names, node_static, _ = build_static_graph(cid, nl, ip or None, op or None)
            # 16.11.2: 新便宜特征源
            pl_json = srow.get('parasitic_caps_json')
            pl = json.loads(pl_json) if isinstance(pl_json, str) else (pl_json or {})
            pl = {str(k).lower(): v for k, v in pl.items()} if isinstance(pl, dict) else {}
            ct_json = srow.get('cell_types_json')
            ct = json.loads(ct_json) if isinstance(ct_json, str) else (ct_json or [])
            ct_set = set(ct) if isinstance(ct, list) else set()
        except Exception:
            continue
        slew_s = float(r.get('slew_s', 0) or 0)
        load_f = float(r.get('output_load_f', 0) or 0)
        c_slew, c_load = parse_corner(r.get('corner'))
        row_slew = (slew_s if slew_s > 0 else c_slew * 1e-12)
        row_load = (load_f if load_f > 0 else c_load * 1e-15)

        for i, n in enumerate(node_names):
            gkey = str(n).lower()
            if gkey not in gate_avg:
                continue
            real = float(np.mean(gate_avg[gkey]))
            if real <= 0:
                continue
            n_gates += 1
            drive = float(node_static[i, 3])
            parasitic = float(node_static[i, 4])
            fanout = float(np.expm1(node_static[i, 1]))
            h = float(np.expm1(node_static[i, 6]))
            slew_ps = row_slew * 1e12
            load_fF = row_load * 1e15

            f_slew = math.log1p(slew_ps)
            f_load = math.log1p(load_fF)
            f_drive = math.log1p(drive)
            f_par = math.log1p(parasitic)
            f_fan = math.log1p(fanout)
            f_h = math.log1p(h)
            f_cslew = math.log1p(c_slew)
            f_cload = math.log1p(c_load)
            yv = math.log1p(real)

            # 8 特征（原）
            X.append([f_slew, f_load, f_drive, f_par, f_fan, f_h, f_cslew, f_cload])
            y.append(yv)

            # 扩充: 交互项 + RC 时间常数 + 深度的代理
            r_on = 1.0 / max(drive, 1e-6)
            c_l = max(parasitic * 1e-15, 1e-18)
            tau_rc = r_on * c_l
            # 门级真实寄生电容（pl 里该门各引脚之和, 已含 fF 量级）
            gate_par = 0.0
            pv = pl.get(gkey)
            if isinstance(pv, dict):
                gate_par = sum(float(x) for x in pv.values() if x is not None)
            elif pv is not None:
                try:
                    gate_par = float(pv)
                except Exception:
                    gate_par = 0.0
            Xe.append([
                f_slew, f_load, f_drive, f_par, f_fan, f_h, f_cslew, f_cload,
                f_drive * f_load,        # 驱动×负载交互
                f_drive * f_h,           # 驱动×电努力
                f_fan * f_par,           # 扇出×寄生
                f_slew * f_load,         # slew×负载
                math.log1p(tau_rc * 1e15),  # RC 时间常数(ps)
                math.log1p(max(parasitic * fanout, 1e-9)),  # 总负载代理
                math.log1p(max(1.0 / (1.0 + fanout), 1e-9)),  # 1/(1+fanout)
                math.log1p(gate_par),    # 门级真实寄生电容(新)
                math.log1p(max(len(gkey), 1)),  # 门名长度代理复杂度(新)
            ])
            ye.append(yv)

    X, y = np.array(X), np.array(y)
    Xe, ye = np.array(Xe), np.array(ye)
    print(f"样本: {len(y)} (row,gate) 对, {n_gates} 门, 跳过 {n_skip}", flush=True)
    if len(y) < 500:
        print("样本太少")
        return

    # 8:2 划分（按行随机）
    rng2 = np.random.RandomState(7)
    idx = rng2.permutation(len(y))
    n_tr = int(len(idx) * 0.8)
    tr, te = idx[:n_tr], idx[n_tr:]

    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    def evalm(name, model, Xtr, ytr, Xte, yte):
        model.fit(Xtr, ytr)
        p = model.predict(Xte)
        ss_res = float(np.sum((yte - p) ** 2))
        ss_tot = float(np.sum((yte - yte.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot
        r = float(np.corrcoef(yte, p)[0, 1])
        # 原始尺度误差（expm1 还原）
        y_orig = np.expm1(yte)
        p_orig = np.expm1(p)
        rel = float(np.median(np.abs(p_orig - y_orig) / np.maximum(y_orig, 1e-12)))
        print(f"  {name:28s} R^2={r2:.4f}  r={r:.4f}  中位相对误差={rel:.1%}", flush=True)
        return r2, r

    print("\n===== 8 特征（原线性特征集）=====", flush=True)
    evalm('线性(8)', LinearRegression(), X[tr], y[tr], X[te], y[te])
    evalm('GBDT(8)', HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=args.seed), X[tr], y[tr], X[te], y[te])
    evalm('RF(8)', RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=args.seed), X[tr], y[tr], X[te], y[te])

    print("\n===== 扩充特征(15)=====", flush=True)
    evalm('线性(15)', LinearRegression(), Xe[tr], ye[tr], Xe[te], ye[te])
    evalm('GBDT(15)', HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=args.seed), Xe[tr], ye[tr], Xe[te], ye[te])
    evalm('RF(15)', RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=args.seed), Xe[tr], ye[tr], Xe[te], ye[te])
    # MLP
    sc = StandardScaler().fit(Xe[tr])
    evalm('MLP(15)', MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=args.seed), sc.transform(Xe[tr]), ye[tr], sc.transform(Xe[te]), ye[te])


if __name__ == '__main__':
    main()
