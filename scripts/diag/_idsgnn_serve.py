"""_idsgnn_serve.py — idsavg GNN 的 serve 端推理（17.1.5，serve 近似模式 '3'）。

背景：v2nowavegnn* 系 delay 模型比纯拓扑多一列 = per-gate ids_avg（训练时由交叉拟合的 idsavg GNN
预测值查表填入）。serve 端原本只有 '1'（线性系数）/ '2'（GBDT15 joblib）两条近似路径，都算不出这一列
（那一列是 GNN 现场推的）→ 本模块补上 '3' = 在 serve 端直接跑 idsavg GNN。

口径来源：scripts/diag/_fit_idsavg_gnn_server.py（R4 = STRUCT_MODE=base + CONE_V2=1 + CONE_PIN=1）。
本模块复用其全部特征构造（同一 build_static_graph、同一 static 归一化、同一 extras 布局），
唯一差异：对**单条 (switching_pin, direction, output, corner) 行**建 N 节点图并前向 —— 训练时是
R×N 拼块，但块内各行特征互不跨行（edges 按行复制、无跨行边），故逐行前向与拼块前向逐位等价。

⚠ 硬性前提：ckpt 必须自带 sta_mean/sta_std（17.1.2 起落盘）。缺失即 raise —— serve 端没有训练折
数据，"回退重算"不可能；静默换用别的统计量只会产出一条看似正常、实则系统性偏移的列。

⚠ in=46 二义性：`convs.0.lin_rel.weight` 的 in_features=46 有两种来源 ——
  (a) STRUCT_MODE=logic_only(7 静态列) + 1 列 ids 近似  → 用模式 '1'/'2'
  (b) STRUCT_MODE=base(8 静态列, 含 n_t) + 1 列 ids GNN → 用模式 '3'
  两种都恰好 46，load_state_dict 不会报错（形状相同）→ **必须查 ckpt 的 struct_mode / IDS_GNN_TABLE，
  不能只看 45/46**。见 docs/OPERATIONS.md §6.2 与 §6.3。
"""
import math
import os
import sys
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.graph_builder import build_static_graph, rebuild_gate_types

DEFAULT_CKPT_GLOB = os.path.join(os.path.expanduser('~'), 'idsavg17', 'idsgnn_fold*.pt')


def cell_types(nl):
    """与 _fit_idsavg_gnn_server.cell_types 逐字一致（网表 X_ 行的末列 = 门类型）。"""
    return {ln.split()[-1] for ln in (nl or '').split('\n')
            if ln.strip().startswith('X_') and len(ln.split()) >= 3}


class GraphConvL(nn.Module):
    """逐位复刻 _fit_idsavg_gnn_server.GraphConvL（勿改；改则 ckpt 权重对不上）。"""

    def __init__(self, fin, fout, bias=True):
        super().__init__()
        self.W1 = nn.Linear(fin, fout, bias=bias)
        self.W2 = nn.Linear(fin, fout, bias=False)

    def forward(self, x, edge_index, n_nodes):
        out = self.W1(x)
        if edge_index.numel():
            src, dst = edge_index[0], edge_index[1]
            agg = torch.zeros(n_nodes, self.W2.out_features, device=x.device, dtype=x.dtype)
            agg.index_add_(0, dst, self.W2(x[src]))
            out = out + agg
        return out


class IdsAvgGNN(nn.Module):
    """逐位复刻 _fit_idsavg_gnn_server.IdsAvgGNN。"""

    def __init__(self, num_types, n_cont, emb, hid, K, dropout):
        super().__init__()
        self.gate_embed = nn.Embedding(num_types, emb)
        actual_in = emb + n_cont
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.convs.append(GraphConvL(actual_in, hid))
        self.norms.append(nn.LayerNorm(hid))
        for _ in range(K - 1):
            self.convs.append(GraphConvL(hid, hid))
            self.norms.append(nn.LayerNorm(hid))
        self.head = nn.Linear(hid, 1)
        self.dropout = dropout

    def forward(self, type_idx, cont, edge_index, n_nodes):
        x = torch.cat([self.gate_embed(type_idx), cont], dim=-1)
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            residual = x
            x = conv(x, edge_index, n_nodes)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if i > 0 and residual.shape == x.shape:
                x = x + residual
        return self.head(x).squeeze(-1)


def cone_dists(start, adj):
    """BFS 沿有向边 (driver->receiver) 从 start 出发: {节点: 跳数}。同训练侧 _cone_dists。"""
    dist = {start: 0}
    dq = deque([start])
    while dq:
        u = dq.popleft()
        for v in adj.get(u, ()):
            if v not in dist:
                dist[v] = dist[u] + 1
                dq.append(v)
    return dist


class IdsGnnServer:
    """载入 idsavg GNN 折模型（等权集成），为候选电路现场算 per-gate ids_avg 特征列。

    结构超参/锥体开关/归一化统计一律从 ckpt 元信息读（不写死），避免与本仓 config 漂移。"""

    def __init__(self, ckpt_paths, device='cpu', verbose=True):
        if not ckpt_paths:
            raise SystemExit('[idsgnn] 未给定任何 idsavg GNN ckpt')
        self.device = device
        meta = None
        self.models = []
        self.val_r2 = []
        keys_cfg = ('num_types', 'n_cont', 'n_cont_base', 'struct_mode', 'cone_v2', 'cone_feat',
                    'cone_pin', 'emb', 'hid', 'k')
        for p in ckpt_paths:
            if not os.path.exists(p):
                raise SystemExit(f'[idsgnn] ckpt 不存在: {p}')
            ck = torch.load(p, map_location='cpu', weights_only=False)
            for k in ('num_types', 'n_cont', 'n_cont_base', 'emb', 'hid', 'k'):
                if k not in ck:
                    raise SystemExit(f'[idsgnn] ckpt {p} 缺元信息 {k}（非 _fit_idsavg_gnn_server 产物?）')
            if 'sta_mean' not in ck or 'sta_std' not in ck:
                raise SystemExit(
                    f'[idsgnn] ckpt {p} 缺 sta_mean/sta_std。\n'
                    f'  serve 端没有训练折数据，无法"回退重算"归一化统计 → 拒绝静默降级。\n'
                    f'  修法：用 17.1.2 及以后代码重跑该折，或从同一次训练的产物里取（ckpt 由 run_variant 落盘即自带）。')
            if meta is None:
                meta = ck
            else:
                for k in keys_cfg:
                    if ck.get(k) != meta.get(k):
                        raise SystemExit(f'[idsgnn] ckpt 配置不一致: {os.path.basename(p)} 的 {k}='
                                         f'{ck.get(k)} != {meta.get(k)}（勿混用不同折/不同 regime 的 ckpt）')
            m = IdsAvgGNN(int(ck['num_types']), int(ck['n_cont']), emb=int(ck['emb']),
                          hid=int(ck['hid']), K=int(ck['k']), dropout=float(ck.get('dropout', 0.0)))
            m.load_state_dict(ck['state_dict'])
            m.eval()
            self.models.append(m.to(device))
            self.val_r2.append(float(ck.get('val_r2', float('nan'))))
        # 由元信息推导块宽（与 _fit_idsavg_gnn_server L85-88 同公式）
        self.struct_mode = str(meta.get('struct_mode', 'base'))
        self.n_cont_base = int(meta['n_cont_base'])
        self.cone_v2 = bool(meta.get('cone_v2', False))
        self.cone_feat = bool(meta.get('cone_feat', False))
        self.cone_pin = bool(meta.get('cone_pin', False))
        self.cone_n = 6 if self.cone_v2 else (3 if self.cone_feat else 0)
        self.pin_n = 3 if self.cone_pin else 0
        self.n_extra = 7 + self.cone_n + self.pin_n
        if self.n_cont_base + self.n_extra != int(meta['n_cont']):
            raise SystemExit(f'[idsgnn] 块宽自洽性失败: n_cont_base({self.n_cont_base}) + n_extra'
                             f'({self.n_extra}) != ckpt n_cont({meta["n_cont"]})')
        self.sta_mean = np.asarray(meta['sta_mean'], dtype=np.float32)
        self.sta_std = np.asarray(meta['sta_std'], dtype=np.float32)
        if self.sta_mean.shape[0] != self.n_cont_base:
            raise SystemExit(f'[idsgnn] sta_mean 宽度 {self.sta_mean.shape} != n_cont_base {self.n_cont_base}')
        if verbose:
            r2 = ', '.join(f'{v:.4f}' for v in self.val_r2)
            print(f'[idsgnn] {len(self.models)} 折载入 | struct_mode={self.struct_mode} '
                  f'n_cont={self.n_cont_base}+{self.n_extra} cone_v2={int(self.cone_v2)} '
                  f'cone_feat={int(self.cone_feat)} cone_pin={int(self.cone_pin)} | fold val_R²={r2}')

    # ---------------------------------------------------------------- 建图
    def build_graph(self, netlist, input_pins, output_pins):
        """返回 (node_names, ns, adj, radj, edges)。ns 按 ckpt 记录的结构模式构建
        （显式传 mode，不改全局 config.STRUCT_MODE —— serve 是多线程 HTTP，改全局会串）。"""
        rebuild_gate_types(cell_types(netlist))
        node_names, ns, ei = build_static_graph('serve', netlist, list(input_pins), list(output_pins),
                                                mode=self.struct_mode)
        ns = ns.detach().cpu().numpy() if hasattr(ns, 'detach') else np.asarray(ns)
        # 与训练侧一致：邻接表与 GNN 边表都剔除自环（build_static_graph 无边时补的自环不带入 GNN）
        edges = [(int(a), int(b)) for a, b in ei.t().tolist() if a != b]
        adj, radj = {}, {}
        for a, b in edges:
            adj.setdefault(a, []).append(b)
            radj.setdefault(b, []).append(a)
        return node_names, ns, adj, radj, edges

    # ---------------------------------------------------------------- 单行组块
    def assemble_row(self, ns, adj, radj, edges, f_slew, f_load, f_cs, f_cl, dir_code, src_i, out_i):
        """单条 timing arc → (Ty[N], T[N, n_cont], edge_index[2,E])。
        与训练侧 assemble() 的块内单行逐位一致（只去掉了 rk*N 的行偏移）。"""
        N = ns.shape[0]
        nb = self.n_cont_base
        static = (ns[:, 1:1 + nb].astype(np.float32) - self.sta_mean) / self.sta_std
        T = np.zeros((N, nb + self.n_extra), dtype=np.float32)
        T[:, :nb] = static
        T[:, nb + 0] = f_slew
        T[:, nb + 1] = f_load
        T[:, nb + 2] = f_cs
        T[:, nb + 3] = f_cl
        T[:, nb + 4] = dir_code
        if src_i is not None:
            T[src_i, nb + 5] = f_slew
        if out_i is not None:
            T[out_i, nb + 6] = f_load
        if self.cone_n or self.pin_n:
            Wc = self.cone_n + self.pin_n
            co = np.zeros((N, Wc), dtype=np.float32)
            d1, d2 = {}, {}
            if src_i is not None:
                d1 = cone_dists(src_i, adj)
            if self.cone_n:
                if d1:
                    md1 = max(d1.values())
                    for g, dd in d1.items():
                        co[g, 0] = 1.0
                        co[g, 1] = dd / max(md1, 1)
                if out_i is not None:
                    d2 = cone_dists(out_i, radj)
                    if d2:
                        md2 = max(d2.values())
                        for g, dd in d2.items():
                            co[g, 2] = dd / max(md2, 1)
            if self.cone_v2:
                if src_i is not None and out_i is not None and d1 and d2:
                    for g in d1:
                        if g in d2:
                            co[g, 3] = 1.0
                if d1:
                    _mxo = _mxi = 0
                    for g in d1:
                        _fo = sum(1 for c in adj.get(g, ()) if c in d1)
                        _fi = sum(1 for p in radj.get(g, ()) if p in d1)
                        co[g, 4] = _fo
                        co[g, 5] = _fi
                        if _fo > _mxo:
                            _mxo = _fo
                        if _fi > _mxi:
                            _mxi = _fi
                    if _mxo > 0:
                        for g in d1:
                            co[g, 4] /= _mxo
                    if _mxi > 0:
                        for g in d1:
                            co[g, 5] /= _mxi
            if self.pin_n and d1:
                for g in d1:
                    _drv = radj.get(g, ())
                    _nt = len(_drv)
                    if not _nt:
                        continue
                    _cnt = sum(1 for _p in _drv if _p in d1)
                    co[g, self.cone_n + 0] = float(_cnt)
                    co[g, self.cone_n + 1] = _cnt / _nt
                    co[g, self.cone_n + 2] = 1.0 if _cnt >= 2 else 0.0
            T[:, nb + 7: nb + 7 + Wc] = co
        Ty = ns[:, 0].astype(np.int64)
        ei = (torch.tensor(edges, dtype=torch.long).t().contiguous() if edges
              else torch.zeros(2, 0, dtype=torch.long))
        return torch.from_numpy(Ty), torch.from_numpy(T), ei

    # ---------------------------------------------------------------- 前向
    def forward_row(self, ty, T, ei):
        """单行前向，返回 log1p 空间 per-node 预测（多折在**预测空间**等权平均 = OOF 集成的行平均口径）。"""
        ty = ty.to(self.device)
        T = T.to(self.device)
        ei = ei.to(self.device)
        acc = None
        with torch.no_grad():
            for m in self.models:
                p = m(ty, T, ei, ty.shape[0])
                acc = p if acc is None else acc + p
        return ((acc / len(self.models)).cpu().numpy() if acc is not None
                else np.zeros(T.shape[0], dtype=np.float32))

    def ids_avg_rows(self, ns, adj, radj, edges, node_names, rows):
        """rows: [{'switching_pin','direction','output','row_slew','row_load','c_slew','c_load'}]
        → {index: np.ndarray[N]}（expm1 后，单位同 ids_avg）。用于 parity 复验（按表内各行的真实 corner）。"""
        n2i = {str(n).lower(): i for i, n in enumerate(node_names)}
        out = {}
        for i, r in enumerate(rows):
            ty, T, ei = self.assemble_row(
                ns, adj, radj, edges,
                math.log1p(float(r['row_slew']) * 1e12), math.log1p(float(r['row_load']) * 1e15),
                math.log1p(float(r['c_slew'])), math.log1p(float(r['c_load'])),
                1.0 if str(r['direction']) == 'rise' else 0.0,
                n2i.get(str(r['switching_pin']).lower()), n2i.get(str(r['output']).lower()))
            out[i] = np.expm1(self.forward_row(ty, T, ei))
        return out

    def ids_avg_by_pin_dir(self, netlist, input_pins, output_pins, slew_s=2e-12, load_f=1e-15,
                           corner=(2.0, 1.0)):
        """serve 口径：延迟按 (pin, dir) 聚合、对**全部 output** 取平均（对齐 Rust
        `simulate_all_outputs_for_expr`）→ ids 特征亦在 log1p 空间对全部 output 平均后再 expm1。
        返回 ({('pin', 'rise'|'fall'): np.ndarray[N]}, node_names)。"""
        node_names, ns, adj, radj, edges = self.build_graph(netlist, input_pins, output_pins)
        rows, keys = [], []
        for sw in input_pins:
            for d in ('rise', 'fall'):
                keys.append((str(sw).lower(), d))
                for o in (output_pins or [None]):
                    rows.append({'switching_pin': sw, 'direction': d, 'output': o,
                                 'row_slew': slew_s, 'row_load': load_f,
                                 'c_slew': corner[0], 'c_load': corner[1]})
        per = self.ids_avg_rows(ns, adj, radj, edges, node_names, rows)
        n_out = max(len(output_pins or []), 1)
        tbl = {}
        for j, k in enumerate(keys):
            lg = np.mean([np.log1p(per[j * n_out + m]) for m in range(n_out)], axis=0)
            tbl[k] = np.expm1(lg)
        return tbl, node_names


_PREPARED = {}


def prepare(ckpt_glob=None, device='cpu', verbose=True):
    """按 glob 载入折模型（进程内缓存）。glob 无命中即 raise（不静默退化）。

    IDSGNN_FOLDS：逗号分隔的折下标（按 glob 排序后的序号），默认全部。
    ⚠ 为什么要这个旋钮：delay 模型**训练时**每张电路只拿到"留出它的那一折"的预测（单折、噪声较大），
    而 serve 端默认多折平均（更平滑）。训练侧 Phase B 判负的机制假说正是"特征的行间不一致破坏排序"
    （见 PROJECT_LOG 17.1.x 阶梯倒挂段）—— 那么**更平滑的 serve 侧特征有可能反而变好**，
    即 shadow 与训练侧结论分叉。要分离"特征本身"与"特征噪声水平"，用 IDSGNN_FOLDS=0 跑一次单折对照。"""
    g = ckpt_glob or os.environ.get('IDSGNN_CKPT') or DEFAULT_CKPT_GLOB
    sel = os.environ.get('IDSGNN_FOLDS', '').strip()
    key = (g, sel)
    if key in _PREPARED:
        return _PREPARED[key]
    import glob as _g
    paths = sorted(_g.glob(g))
    if not paths:
        raise SystemExit(f'[idsgnn] IDSGNN_CKPT glob 无命中: {g}\n'
                         f'  用 IDSGNN_CKPT 指定折 ckpt（如 ~/idsavg17/idsgnn_fold*.pt）')
    if sel:
        want = [int(x) for x in sel.split(',') if x.strip() != '']
        bad = [i for i in want if not (0 <= i < len(paths))]
        if bad:
            raise SystemExit(f'[idsgnn] IDSGNN_FOLDS={sel} 越界（glob 命中 {len(paths)} 个折）')
        paths = [paths[i] for i in want]
        print(f'[idsgnn] IDSGNN_FOLDS={sel} → 只用 {[os.path.basename(p) for p in paths]}（单折=贴近训练侧噪声水平）')
    _PREPARED[key] = IdsGnnServer(paths, device=device, verbose=verbose)
    return _PREPARED[key]
