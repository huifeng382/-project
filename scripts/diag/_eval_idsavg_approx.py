"""① ids_avg 近似公式评估（16.9.4）
目标：判断「便宜量能否近似 ids_avg」——决定值不值得训一版近似特征模型。
- 回归 R^2（log-log）：slew/load/drive/parasitic/fanout/h 能解释 ids_avg 多少方差 = 任何公式的天花板
- 物理公式（C_L·ΔV/T_sw）相关性：候选公式的实际表现
用法: python scripts/diag/_eval_idsavg_approx.py
"""
import sys
import os
import glob
import json
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd
import torch
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
    # ---- 静态 ----
    sp = os.path.join(DATA, 'data/batch_v2_full/circuit_static.parquet')
    sparts = sorted(glob.glob(os.path.join(DATA, 'data/batch_v2_full/circuit_static_part*.parquet')))
    sdf = pd.concat([pd.read_parquet(p) for p in ([sp] if os.path.exists(sp) else sparts)], ignore_index=True)
    sdf['circuit_id'] = sdf['circuit_id'].astype(str)
    sdf = sdf.drop_duplicates('circuit_id').set_index('circuit_id')

    # ---- 动态行（采样前 150 电路 × 每电路 1 行）----
    dp = os.path.join(DATA, 'data/batch_v2_full/timing_arcs.parquet')
    dparts = sorted(glob.glob(os.path.join(DATA, 'data/batch_v2_full/timing_arcs_part*.parquet')))
    cols = ['circuit_id', 'transistor_wave_json', 'slew_s', 'output_load_f', 'corner']
    ddf = pd.concat([pd.read_parquet(p, columns=cols) for p in ([dp] if os.path.exists(dp) else dparts)], ignore_index=True)
    ddf['circuit_id'] = ddf['circuit_id'].astype(str)
    ddf = ddf[ddf['transistor_wave_json'].notna()]
    circuits = ddf['circuit_id'].drop_duplicates().tolist()[:150]
    rows = ddf[ddf['circuit_id'].isin(circuits)].groupby('circuit_id').first().reset_index()

    X, y = [], []          # 回归特征 / 目标
    Xf, yf = [], []        # 物理公式近似 / 目标
    n_gates = 0
    for _, r in rows.iterrows():
        cid = r['circuit_id']
        if cid not in sdf.index:
            continue
        try:
            wave = json.loads(r['transistor_wave_json']) if isinstance(r['transistor_wave_json'], str) else {}
        except Exception:
            continue
        if not isinstance(wave, dict) or not wave:
            continue
        # 每门真实 ids_avg（跨晶体管均值，对齐 GNN 特征聚合）
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
        # 静态图（门级特征）
        try:
            srow = sdf.loc[cid]
            nl = srow['gate_level_netlist']
            ip = json.loads(srow['input_pins_json']) if isinstance(srow['input_pins_json'], str) else srow['input_pins_json']
            op = json.loads(srow['output_pins_json']) if isinstance(srow['output_pins_json'], str) else srow['output_pins_json']
            node_names, node_static, _ = build_static_graph(cid, nl, ip or None, op or None)
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
                continue  # 非开关门（零电流）不计入
            n_gates += 1
            drive = float(node_static[i, 3])
            parasitic = float(node_static[i, 4])          # 寄生电容（fF 量级）
            fanout = float(np.expm1(node_static[i, 1]))
            h = float(np.expm1(node_static[i, 6]))        # 电努力 = out_load/input_cap

            # 回归特征（行级 slew/load + 门级 drive/parasitic/fanout/h + corner）
            X.append([
                math.log1p(row_slew * 1e12), math.log1p(row_load * 1e15),
                math.log1p(drive), math.log1p(parasitic),
                math.log1p(fanout), math.log1p(h),
                math.log1p(c_slew), math.log1p(c_load),
            ])
            y.append(math.log1p(real))

            # 物理公式：ids_avg ≈ C_L·ΔV / T_sw
            r_on = 1.0 / max(drive, 1e-6)                   # 导通电阻 ∝ 1/drive
            c_l = max(parasitic * 1e-15, 1e-18)             # 门级负载代理（寄生电容）
            t_sw = row_slew + r_on * c_l                    # 开关时间 = 输入斜率 + RC
            swing = VDD * (1.0 - math.exp(-t_sw / max(r_on * c_l, 1e-24)))
            approx = c_l * swing / max(t_sw, 1e-18)
            Xf.append(math.log1p(approx))
            yf.append(math.log1p(real))

    X, y = np.array(X), np.array(y)
    Xf, yf = np.array(Xf), np.array(yf)
    print(f"样本: {len(y)} (row,gate) 对, {n_gates} 门, {len(rows)} 电路行")
    if len(y) < 100:
        print("样本太少，无法评估")
        return

    # ---- 1) 回归 R^2（天花板）----
    A = np.column_stack([X, np.ones(len(y))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2_reg = 1 - ss_res / ss_tot
    corr_reg = float(np.corrcoef(y, pred)[0, 1])
    print(f"\n[回归天花板] log-log 线性 R^2 = {r2_reg:.3f}  Pearson r = {corr_reg:.3f}")

    # ---- 2) 物理公式（校准后 R^2）----
    B = np.column_stack([Xf, np.ones(len(yf))])
    cf, *_ = np.linalg.lstsq(B, yf, rcond=None)
    pf = B @ cf
    ss_res_f = float(np.sum((yf - pf) ** 2))
    r2_f = 1 - ss_res_f / float(np.sum((yf - yf.mean()) ** 2))
    corr_f = float(np.corrcoef(yf, pf)[0, 1])
    corr_raw = float(np.corrcoef(Xf, yf)[0, 1])
    print(f"[物理公式] 原始 log 相关 r = {corr_raw:.3f}   校准后 R^2 = {r2_f:.3f}")

    # ---- 3) 判定 ----
    print("\n判定参考：")
    print(f"  回归 R^2 {r2_reg:.2f}: {'>=0.5 → 便宜量解释力强，近似公式值得做' if r2_reg >= 0.5 else '<0.5 → 便宜量解释力有限，近似收益存疑'}")
    print(f"  物理公式 R^2 {r2_f:.2f}: {'>=0.3 → 候选公式可用' if r2_f >= 0.3 else '<0.3 → 公式太粗糙，需拟合更复杂的映射'}")

    # ---- 4) 输出拟合系数（供 data_loader USE_IDS_AVG_APPROX=1 使用）----
    print("\n=== 拟合系数（log-log 线性，特征顺序同特征表）===")
    feat_names = ['log1p(slew_p)', 'log1p(load_f)', 'log1p(drive)', 'log1p(parasitic)',
                  'log1p(fanout)', 'log1p(h)', 'log1p(corner_slew)', 'log1p(corner_load)']
    for i, n in enumerate(feat_names):
        print(f"  {n}: {coef[i]:+.6f}")
    print(f"  intercept: {coef[-1]:+.6f}")
    print("COEFS_JSON=" + __import__('json').dumps({'coef': [round(float(c), 6) for c in coef],
                                                     'feat_names': feat_names}))


if __name__ == '__main__':
    main()
