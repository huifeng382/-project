"""gate_states 对齐验证（方向 #6，16.11.4）
serve 推理时用 BFS 逻辑仿真推 gate_states（logic_sim.compute_gate_states），
训练用真实 gate_states_json。本脚本对比两者一致性，判断 serve 特征是否带噪。

用法: python scripts/diag/_check_gate_states_align.py [--n 200]
输出: 逐电路匹配率统计 + 平均一致率
"""
import sys
import os
import glob
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd
from src.graph_builder import build_static_graph, rebuild_gate_types
import src.graph_builder as gb
from src.logic_sim import compute_gate_states

DATA = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=200, help='采样电路行数')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    # 静态表
    sp = os.path.join(DATA, 'data/batch_v2_full/circuit_static.parquet')
    sparts = sorted(glob.glob(os.path.join(DATA, 'data/batch_v2_full/circuit_static_part*.parquet')))
    sdf = pd.concat([pd.read_parquet(p) for p in ([sp] if os.path.exists(sp) else sparts)], ignore_index=True)
    sdf['circuit_id'] = sdf['circuit_id'].astype(str)
    sdf = sdf.drop_duplicates('circuit_id').set_index('circuit_id')

    # 动态表（含真实 gate_states_json）
    dp = os.path.join(DATA, 'data/batch_v2_full/timing_arcs.parquet')
    dparts = sorted(glob.glob(os.path.join(DATA, 'data/batch_v2_full/timing_arcs_part*.parquet')))
    cols = ['circuit_id', 'switching_pin', 'direction', 'vector', 'gate_states_json',
            'output']
    ddf = pd.concat([pd.read_parquet(p, columns=cols) for p in ([dp] if os.path.exists(dp) else dparts)],
                    ignore_index=True)
    ddf['circuit_id'] = ddf['circuit_id'].astype(str)
    ddf = ddf[ddf['gate_states_json'].notna()]
    rng = np.random.RandomState(args.seed)
    if len(ddf) > args.n:
        ddf = ddf.sample(args.n, random_state=rng)

    n_total = 0
    n_match = 0
    n_circ = 0
    circ_rates = []
    for _, r in ddf.iterrows():
        cid = r['circuit_id']
        if cid not in sdf.index:
            continue
        try:
            srow = sdf.loc[cid]
            nl = srow['gate_level_netlist']
            ip = json.loads(srow['input_pins_json']) if isinstance(srow['input_pins_json'], str) else srow['input_pins_json']
            op = json.loads(srow['output_pins_json']) if isinstance(srow['output_pins_json'], str) else srow['output_pins_json']
            # 真实 gate_states
            gs_true = json.loads(r['gate_states_json']) if isinstance(r['gate_states_json'], str) else r['gate_states_json']
            gs_true = {str(k).lower(): int(v) for k, v in gs_true.items()} if isinstance(gs_true, dict) else {}
        except Exception:
            continue
        if not gs_true:
            continue
        try:
            rebuild_gate_types(_load_cell_types_from_netlist(nl))
            node_names, node_static, edge_index = build_static_graph(cid, nl, list(ip or []), list(op or []))
        except Exception:
            continue
        # serve 的 BFS 路径（同 serve.py _gate_states_bfs）
        vector_str = str(r.get('vector', ''))
        switching = r['switching_pin']
        node_types = {}
        for j, n in enumerate(node_names):
            ti = int(node_static[j, 0].item())
            node_types[n] = gb.GATE_TYPES[ti] if ti < len(gb.GATE_TYPES) else 'UNKNOWN'
        try:
            gs_bfs = compute_gate_states(node_names, node_types, edge_index, vector_str,
                                         list(ip or []), switching, outputs=list(op or []))
        except Exception:
            continue
        # 对比共同键
        common = [n for n in node_names if str(n).lower() in gs_true]
        if not common:
            continue
        n_circ += 1
        match = sum(1 for n in common
                    if gs_bfs.get(n) == gs_true.get(str(n).lower()))
        n_total += len(common)
        n_match += match
        circ_rates.append(match / len(common))

    if n_circ == 0:
        print('无有效样本')
        return
    rates = np.array(circ_rates)
    print(f'电路数={n_circ}  门总数={n_total}')
    print(f'门级一致率: {n_match/n_total:.4f}')
    print(f'电路级一致率: min={rates.min():.3f} med={np.median(rates):.3f} '
          f'mean={rates.mean():.3f} max={rates.max():.3f}')
    print(f'完全一致(100%)的电路: {(rates==1.0).sum()}/{len(rates)}')
    print(f'一致率<0.5 的电路: {(rates<0.5).sum()}/{len(rates)}')


def _load_cell_types_from_netlist(netlist):
    types = set()
    for line in (netlist or '').split('\n'):
        s = line.strip()
        if s.startswith('X_') and len(s.split()) >= 3:
            types.add(s.split()[-1])
    return types


if __name__ == '__main__':
    main()
