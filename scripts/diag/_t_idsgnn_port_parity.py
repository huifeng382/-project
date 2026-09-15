"""_t_idsgnn_port_parity.py — 本地等价性自检：serve 端 assemble_row 是否与训练侧 assemble 逐位一致。

做法：从 scripts/diag/_fit_idsavg_gnn_server.py **原文件里按 AST 抽出** `_cone_dists` 与 `assemble`
的源码并 exec（不是照抄 —— 照抄就证明不了任何东西），对同一张合成电路、同一批行各跑一遍，
逐列比对 extras 布局 / 锥体通道 / 边表。跑通才能说 serve 端那一列的口径没漂。

用法: python3 scripts/diag/_t_idsgnn_port_parity.py
"""
import ast
import os
import sys
from collections import deque

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'diag'))

import config                                     # noqa: E402
from src.graph_builder import build_static_graph, rebuild_gate_types   # noqa: E402
from _idsgnn_serve import IdsGnnServer, cell_types                    # noqa: E402

# R4 regime（= 5 折 idsavg GNN 实际配置）
STRUCT_MODE, N_CONT_BASE = 'base', 7
CONE_FEAT, CONE_V2, CONE_PIN = False, True, True
CONE_N = 6 if CONE_V2 else (3 if CONE_FEAT else 0)
PIN_N = 3 if CONE_PIN else 0
N_EXTRA = 7 + CONE_N + PIN_N
N_CONT = N_CONT_BASE + N_EXTRA

NETLIST = '\n'.join([
    'X_1 in1 in2 n1 NAND2x1',
    'X_2 in2 in3 n2 NOR2x1',
    'X_3 in1 n1 n3 AND2x1',
    'X_4 n1 n2 n4 XOR2x1',
    'X_5 n3 n4 n5 NAND3x1',
    'X_6 n2 n5 n6 INVx2',
    'X_7 n5 n6 out1 BUFx1',
    'X_8 n5 n6 out2 NAND2x2',
])
IPS = ['in1', 'in2', 'in3']
OPS = ['out1', 'out2']


def extract(path, names):
    """从原文件按 AST 抽出若干顶层函数源码并 exec，返回命名空间。"""
    src = open(path, encoding='utf-8').read()
    tree = ast.parse(src)
    segs = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            segs.append(ast.get_source_segment(src, node))
    missing = set(names) - {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    if missing:
        sys.exit(f'原文件里找不到 {missing} —— 训练侧代码改名了，本自检需同步')
    ns = {'np': np, 'torch': torch, 'deque': deque,
          'N_CONT_BASE': N_CONT_BASE, 'N_EXTRA': N_EXTRA, 'CONE_N': CONE_N,
          'CONE_V2': CONE_V2, 'CONE_FEAT': CONE_FEAT, 'CONE_PIN': CONE_PIN, 'PIN_N': PIN_N,
          '_tier_ok': lambda nt: True}
    exec('\n\n'.join(segs), ns)
    return ns


def main():
    ref_ns = extract(os.path.join(ROOT, 'scripts', 'diag', '_fit_idsavg_gnn_server.py'),
                     ['_cone_dists', 'assemble'])
    print('[port] 已从训练侧原文件抽出 _cone_dists / assemble')

    rebuild_gate_types(cell_types(NETLIST))
    node_names, ns_t, ei = build_static_graph('t', NETLIST, IPS, OPS, mode=STRUCT_MODE)
    assert isinstance(ns_t, torch.Tensor), type(ns_t)
    ns = ns_t.numpy()
    N = ns.shape[0]
    # ns 总宽 = 1(type 列) + N_CONT_BASE（训练侧取 ns[:, 1:1+N_CONT_BASE]）
    assert ns.shape[1] == N_CONT_BASE + 1, f'静态列宽 {ns.shape[1]} != {N_CONT_BASE + 1}'
    edges = [(int(a), int(b)) for a, b in ei.t().tolist() if a != b]
    adj, radj = {}, {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        radj.setdefault(b, []).append(a)
    n2i = {n.lower(): i for i, n in enumerate(node_names)}
    print(f'[port] 电路 N={N} 边={len(edges)} 节点={node_names}')

    sta_mean = ns[:, 1:1 + N_CONT_BASE].mean(0).astype(np.float32)
    sta_std = ns[:, 1:1 + N_CONT_BASE].std(0).astype(np.float32) + 1e-6

    specs = []       # (src_pin, dir, out_pin) —— 含一行 out 缺失，走 None 分支
    for sw in IPS:
        for d in ('rise', 'fall'):
            for o in OPS:
                specs.append((sw, d, o))
    specs.append((IPS[0], 'rise', None))

    def row_of(sw, d, o, rk):
        c_slew, c_load = 2.0, 1.0
        s, l = (2.0 + 0.1 * rk) * 1e-12, (1.0 + 0.05 * rk) * 1e-15
        return {'edges': edges,
                'f_slew': float(np.log1p(s * 1e12)), 'f_load': float(np.log1p(l * 1e15)),
                'f_cs': float(np.log1p(c_slew)), 'f_cl': float(np.log1p(c_load)),
                'dir_code': 1.0 if d == 'rise' else 0.0,
                'src_i': n2i.get(str(sw).lower()),
                'out_i': (n2i.get(str(o).lower()) if o is not None else None),
                'sup': [(i, float(i)) for i in range(N)]}

    rows = [row_of(sw, d, o, rk) for rk, (sw, d, o) in enumerate(specs)]
    blocks = {'c0': {'ns': ns.astype(np.float32), 'rows': rows, 'adj': adj, 'radj': radj,
                     'node_names': node_names, 'edges': edges}}

    ref_ns['blocks'] = blocks
    ref_ns['_sta_mean'] = sta_mean
    ref_ns['_sta_std'] = sta_std
    ty_r, T_r, ei_r, s_i_r, s_y_r = ref_ns['assemble'](['c0'], use_edges=True)[0]
    T_r = T_r.numpy()
    R = len(rows)
    print(f'[port] 训练侧 assemble: T={tuple(T_r.shape)} ei={tuple(ei_r.shape)} '
          f'（R={R}, N={N}, n_cont={N_CONT}）')
    assert T_r.shape == (R * N, N_CONT), T_r.shape

    srv = object.__new__(IdsGnnServer)
    srv.n_cont_base, srv.n_extra = N_CONT_BASE, N_EXTRA
    srv.cone_n, srv.pin_n = CONE_N, PIN_N
    srv.cone_v2, srv.cone_feat, srv.cone_pin = CONE_V2, CONE_FEAT, CONE_PIN
    srv.sta_mean, srv.sta_std = sta_mean, sta_std

    worst_T = worst_Ty = 0.0
    E = len(edges)
    for rk, (sw, d, o) in enumerate(specs):
        r = rows[rk]
        ty_m, T_m, ei_m = srv.assemble_row(ns.astype(np.float32), adj, radj, edges,
                                          r['f_slew'], r['f_load'], r['f_cs'], r['f_cl'],
                                          r['dir_code'], r['src_i'], r['out_i'])
        T_o = T_r[rk * N:(rk + 1) * N]
        ty_o = ty_r[rk * N:(rk + 1) * N].numpy()
        dT = float(np.abs(T_m.numpy() - T_o).max())
        dTy = float(np.abs(ty_m.numpy() - ty_o).max()) if N else 0.0
        # 训练侧把每行的边整体平移 rk*N（R×N 拼块）；serve 端不复制图 → 去掉该平移再比
        ei_o = ei_r[:, rk * E:(rk + 1) * E].numpy() - rk * N
        ok_ei = (ei_m.numpy().shape == ei_o.shape and np.array_equal(ei_m.numpy(), ei_o))
        worst_T, worst_Ty = max(worst_T, dT), max(worst_Ty, dTy)
        flag = 'OK' if (dT == 0 and dTy == 0 and ok_ei) else '**不一致**'
        print(f'  行{rk:2d} {sw}/{d}/{o}: ΔT={dT:.3e} ΔTy={dTy:.3e} 边表{"同" if ok_ei else "异"} {flag}')

    # 通道非零性自检：锥体/输入脚通道真的有值（全 0 说明 BFS 或列偏移错了，但差值仍可能为 0）
    nz = {c: int((np.abs(T_r[:, c]) > 0).sum()) for c in range(N_CONT_BASE, N_CONT)}
    print('[port] extras 各通道非零计数: ' + ', '.join(f'+{c - N_CONT_BASE}={v}' for c, v in nz.items()))
    assert nz[N_CONT_BASE + 7] > 0, '锥体 in-cone 通道全 0 —— BFS/列偏移有问题'
    assert nz[N_CONT_BASE + 7 + CONE_N] > 0, '输入脚 count 通道全 0'

    print(f'\n[port] 全局 max|ΔT| = {worst_T:.3e}, max|ΔTy| = {worst_Ty:.3e}')
    if worst_T != 0 or worst_Ty != 0:
        sys.exit('[port] 失败：serve 端 assemble_row 与训练侧 assemble 不一致')
    print('[port] OK —— serve 端单行组块与训练侧拼块逐位一致')


if __name__ == '__main__':
    main()
