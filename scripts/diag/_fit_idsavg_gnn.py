"""复刻 DelayGNN → 独立 per-gate ids_avg 回归器 (受控对比, 判断唯一尺度 = ids_avg R^2/Spearman)

架构 = DelayGNN 骨架的纯 torch 复刻 (不 import pyg / 不改 src/model.py):
  - gate 类型 Embedding (type_idx)  concat 连续特征 → K 层 GraphConv(W1 自身 + W2 邻居求和)
  - 每层 LayerNorm + ReLU + Dropout, 第 0 层后残差 (对齐 src/model.py DelayGNN.convs 循环)
  - row 条件 (slew/load/corner/direction) 烘焙进节点特征, 参与消息传递 (对齐 delayGNN 的 delay 数据流)
  - 关键差异点 (物理动机): slew_s 只锚定 switching_pin 节点、output_load_f 只锚定 output 节点,
    沿有向 driver->receiver 边传播 => 模型能感知「刺激源位置 + 门是否在 cone 内 / 离源距离」,
    这是 GBDT15 (slew/load 对所有门均匀广播) 无法表达的信号.
  - 读出: 每节点单值头 → log1p(mean 真实 ids_avg); 只监督 wave 里 real>0 的门 (与 _fit_idsavg_gbdt15 同口径)

对照 (同 1500 电路 / 同 target / 同电路级切分 train1200/val150/test150):
  A. GBDT15 (部署同款, 15 特征)                     [复现 0.674]
  V_gnn     : 本模型 + 有向边消息传递
  V_nograph : 同特征, 无边 (等价 per-node MLP)      [隔离「消息传递」贡献, 防架构错觉]
若 V_gnn 显著 > V_nograph 且 >= GBDT15 => 图传播携带 ids_avg 信号, 方向成立.
判断: 一切以 ids_avg 准确率为准, 与 delay 无关.
"""
import sys, os, json, math, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import spearmanr
from src.graph_builder import build_static_graph, rebuild_gate_types

torch.manual_seed(0); np.random.seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'DEV={DEV}', flush=True)

DATA = 'data/batch_v2_full'
N_CIRCUITS = 1500
CIRC_SPLIT = (0.8, 0.1, 0.1)          # 电路级切分: train=未见电路 (serve 真实口径)
# ---------------- GNN 超参 (DelayGNN 骨架复刻) ----------------
EMB = 16            # gate type embedding dim
HID = 96            # hidden dim
K = 3               # GraphConv 层数
DROPOUT = 0.25
LR = 3e-3
EPOCHS = 80
N_CONT_BASE = 7     # ns 静态连续列: fanout_log, depth_log, drive, p, g, h_log, n_t
# 每节点额外连续通道:
#   [0] slew_ps_log(广播) [1] load_ff_log(广播) [2] c_slew_log [3] c_load_log [4] dir_code
#   [5] src_anch: log1p(row_slew*1e12) 只落在 switching_pin 节点, 其余 0
#   [6] out_anch: log1p(row_load*1e15) 只落在 output 节点,   其余 0
N_EXTRA = 7

def parse_corner(corner):
    try:
        s, l = str(corner).split('_')[:2]
        return float(s[1:].replace('p', '.')), float(l[1:].replace('p', '.'))
    except Exception:
        return 5.0, 10.0

def cell_types(nl):
    return {ln.split()[-1] for ln in (nl or '').split('\n')
            if ln.strip().startswith('X_') and len(ln.split()) >= 3}

# ============================================================ 载入 + 建块
sdf = pd.read_parquet(os.path.join(DATA, 'circuit_static.parquet'))
sdf['circuit_id'] = sdf['circuit_id'].astype(str)
sdf = sdf.drop_duplicates('circuit_id').set_index('circuit_id')
cols = ['circuit_id', 'transistor_wave_json', 'slew_s', 'output_load_f', 'corner', 'direction', 'switching_pin', 'output']
ddf = pd.read_parquet(os.path.join(DATA, 'timing_arcs.parquet'), columns=cols)
ddf['circuit_id'] = ddf['circuit_id'].astype(str)
ddf = ddf[ddf['transistor_wave_json'].notna()]
circ_all = ddf['circuit_id'].drop_duplicates().tolist()
sel = np.random.RandomState(42).choice(circ_all, size=min(N_CIRCUITS, len(circ_all)), replace=False)
ddf = ddf[ddf['circuit_id'].isin(sel)]
print(f'采样 {len(sel)} 电路 / {len(ddf)} 行', flush=True)

order = np.asarray(list(sel)); rp = np.random.RandomState(7).permutation(len(order))
n1 = int(len(order)*CIRC_SPLIT[0]); n2 = n1 + int(len(order)*CIRC_SPLIT[1])
tr_c = set(order[rp[:n1]]); va_c = set(order[rp[n1:n2]]); te_c = set(order[rp[n2:]])
print(f'电路级: train {len(tr_c)} / val {len(va_c)} / test {len(te_c)}', flush=True)

Xs, ys, row_m = [], [], []            # GBDT 特征收集 (复刻 _fit_idsavg_gbdt15)
blocks = {}                           # cid -> 训练块
gidx_cache = {}                       # type idx 全局注册
max_type = 0
t0 = time.time()
for ci, cid in enumerate(sel):
    srow = sdf.loc[cid]
    nl = srow['gate_level_netlist']
    try:
        ip = json.loads(srow['input_pins_json']) if isinstance(srow['input_pins_json'], str) else srow['input_pins_json']
        op = json.loads(srow['output_pins_json']) if isinstance(srow['output_pins_json'], str) else srow['output_pins_json']
        rebuild_gate_types(cell_types(nl))
        node_names, ns, ei = build_static_graph(cid, nl, ip or None, op or None)
    except Exception:
        continue
    ns = ns.numpy()
    max_type = max(max_type, int(ns[:, 0].max()))
    n2i = {n: i for i, n in enumerate(node_names)}
    edges = [(int(a), int(b)) for a, b in ei.t().tolist() if a != b]   # 有向 driver->receiver
    rows = ddf[ddf['circuit_id'] == cid]
    blk = []
    for _, r in rows.iterrows():
        try:
            wave = json.loads(r['transistor_wave_json']) if isinstance(r['transistor_wave_json'], str) else {}
        except Exception:
            continue
        if not isinstance(wave, dict) or not wave:
            continue
        gate_avg = {}
        for tv in wave.values():
            if not isinstance(tv, dict): continue
            g, v = tv.get('gate'), tv.get('ids_avg')
            if g is None or v is None: continue
            gate_avg.setdefault(str(g).lower(), []).append(float(v))
        if not gate_avg: continue
        slew_s = float(r.get('slew_s', 0) or 0); load_f = float(r.get('output_load_f', 0) or 0)
        c_slew, c_load = parse_corner(r.get('corner'))
        row_slew = (slew_s if slew_s > 0 else c_slew*1e-12)
        row_load = (load_f if load_f > 0 else c_load*1e-15)
        f_slew = math.log1p(row_slew*1e12); f_load = math.log1p(row_load*1e15)
        f_cs = math.log1p(c_slew); f_cl = math.log1p(c_load)
        dir_code = 1.0 if str(r.get('direction')) == 'rise' else 0.0
        src_i = n2i.get(str(r.get('switching_pin', '')).lower())
        out_i = n2i.get(str(r.get('output', '')).lower())
        sup = []          # (node_idx, y)
        pairs_gbdt = []
        for i, n in enumerate(node_names):
            gk = str(n).lower()
            if gk not in gate_avg: continue
            real = float(np.mean(gate_avg[gk]))
            if real <= 0: continue
            y = math.log1p(real)
            sup.append((i, y))
            # GBDT 特征 (逐列复刻; depth/g 在末两列用于 17 特征, GBDT15 只取前 15)
            drive, par = float(ns[i, 3]), float(ns[i, 4])
            fan, hh = float(np.expm1(ns[i, 1])), float(np.expm1(ns[i, 6]))
            dep, g = float(ns[i, 2]), float(ns[i, 5])
            f_d, f_p = math.log1p(drive), math.log1p(par)
            f_fan, f_h = math.log1p(fan), math.log1p(hh)
            r_on = 1.0/max(drive, 1e-6); c_l = max(par*1e-15, 1e-18)
            Xs.append([f_slew, f_load, f_d, f_p, f_fan, f_h, f_cs, f_cl,
                       f_d*f_load, f_d*f_h, f_fan*f_p, f_slew*f_load,
                       math.log1p(r_on*c_l*1e15), math.log1p(max(par*fan, 1e-9)),
                       math.log1p(max(1.0/(1.0+fan), 1e-9)),
                       dep, g])
            ys.append(y); row_m.append(cid)
        if not sup: continue
        blk.append({'n2i': n2i, 'edges': edges, 'f_slew': f_slew, 'f_load': f_load,
                    'f_cs': f_cs, 'f_cl': f_cl, 'dir_code': dir_code,
                    'src_i': src_i, 'out_i': out_i, 'sup': sup})
    if blk:
        blocks[cid] = {'ns': ns, 'node_names': node_names, 'rows': blk}
    if (ci+1) % 400 == 0:
        print(f'  电路 {ci+1}/{len(sel)}: GBDT样本 {len(ys)} / 建块 {len(blocks)}, {time.time()-t0:.0f}s', flush=True)

Xs = np.array(Xs); ys = np.array(ys)
NUM_TYPES = max_type + 2   # +type pad margin
print(f'\n样本 {len(ys)} (row,gate); 电路块 {len(blocks)}; max_type={max_type}', flush=True)

# ---- 静态 7 列 z-score (训练集统计; n_t 可达 80+, 直接进卷积会爆 nan) ----
_tr_ns = np.vstack([blocks[c]['ns'][:, 1:1+N_CONT_BASE] for c in tr_c if c in blocks])
_sta_mean = _tr_ns.mean(0)
_sta_std = _tr_ns.std(0) + 1e-6

def masks(cids):
    m = np.array([c in cids for c in row_m])
    return m
mtr = masks(tr_c); mva = masks(va_c); mte = masks(te_c)

# ============================================================ 对照 A: GBDT15 (电路级切分)
print('\n===== A. GBDT15 (15特征, 部署同款, 电路级切分) =====', flush=True)
def gbdt_r2(name, Xtr, ytr, Xte, yte):
    gb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, random_state=42,
                                       validation_fraction=0.1, early_stopping=True, n_iter_no_change=40)
    gb.fit(Xtr, ytr)
    p = gb.predict(Xte)
    r2 = 1-np.sum((yte-p)**2)/np.sum((yte-yte.mean())**2)
    rho, _ = spearmanr(yte, p)
    print(f'  {name:34s} R^2={r2:.4f}  Spearman={rho:.4f}  n_iter={gb.n_iter_}', flush=True)
    return r2, rho
gbdt_r2('A. GBDT15', Xs[mtr][:, :15], ys[mtr], Xs[mte][:, :15], ys[mte])

# ============================================================ DelayGNN 复刻骨架
class GraphConvL(nn.Module):
    """纯 torch 复刻 torch_geometric GraphConv: x_i' = W1 x_i + W2 * sum_{j->i} x_j (+bias)
    edge_index: [2, E] 有向, src=driver, dst=receiver。"""
    def __init__(self, fin, fout, bias=True):
        super().__init__()
        self.W1 = nn.Linear(fin, fout, bias=bias)
        self.W2 = nn.Linear(fin, fout, bias=False)
    def forward(self, x, edge_index, n_nodes):
        out = self.W1(x)
        if edge_index.numel():
            src, dst = edge_index[0], edge_index[1]
            agg = torch.zeros(n_nodes, self.W2.out_features, device=x.device, dtype=x.dtype)
            msg = self.W2(x[src])
            agg.index_add_(0, dst, msg)
            out = out + agg
        return out

class IdsAvgGNN(nn.Module):
    """DelayGNN 骨架复刻 + per-node ids_avg 头 (log1p空间)."""
    def __init__(self, num_types, n_cont, emb=EMB, hid=HID, K=K, dropout=DROPOUT):
        super().__init__()
        self.gate_embed = nn.Embedding(num_types, emb)
        actual_in = emb + n_cont
        self.convs = nn.ModuleList(); self.norms = nn.ModuleList()
        self.convs.append(GraphConvL(actual_in, hid)); self.norms.append(nn.LayerNorm(hid))
        for _ in range(K - 1):
            self.convs.append(GraphConvL(hid, hid)); self.norms.append(nn.LayerNorm(hid))
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

# ============================================================ 组装训练块 (per circuit 整块批量: 所有行 block-diag)
def assemble(cids, use_edges=True):
    """返回 (type_idx, cont, edge_index, sup_idx, sup_y), 每条电路拼成一个块(R*N 节点)."""
    out = []
    for cid in cids:
        b = blocks.get(cid)
        if not b: continue
        ns = b['ns']; N = ns.shape[0]; rows = b['rows']; R = len(rows)
        # 每行块: 静态列(7) + extra(7). type 单独.
        static = (ns[:, 1:1+N_CONT_BASE].astype(np.float32) - _sta_mean) / _sta_std   # (N,7) z-score
        typ = ns[:, 0].astype(np.int64)                          # (N,)
        T = np.zeros((R*N, N_CONT_BASE + N_EXTRA), dtype=np.float32)
        Ty = np.empty(R*N, dtype=np.int64)
        for rk, row in enumerate(rows):
            s = rk*N; e = s+N
            T[s:e, :N_CONT_BASE] = static
            T[s:e, N_CONT_BASE+0] = row['f_slew']
            T[s:e, N_CONT_BASE+1] = row['f_load']
            T[s:e, N_CONT_BASE+2] = row['f_cs']
            T[s:e, N_CONT_BASE+3] = row['f_cl']
            T[s:e, N_CONT_BASE+4] = row['dir_code']
            if row['src_i'] is not None:
                T[s+row['src_i'], N_CONT_BASE+5] = row['f_slew']
            if row['out_i'] is not None:
                T[s+row['out_i'], N_CONT_BASE+6] = row['f_load']
            Ty[s:e] = typ
        edges_t = []
        if use_edges:
            for rk in range(R):
                off = rk*N
                edges_t += [(a+off, b+off) for a, b in rows[0]['edges']]
        ei = torch.tensor(edges_t, dtype=torch.long).t().contiguous() if edges_t else torch.zeros(2, 0, dtype=torch.long)
        sup_i = [rk*N + gi for rk, row in enumerate(rows) for gi, _ in row['sup']]
        sup_y = [y for row in rows for _, y in row['sup']]
        out.append((torch.tensor(Ty), torch.tensor(T), ei,
                    torch.tensor(sup_i, dtype=torch.long), torch.tensor(sup_y, dtype=torch.float32)))
    return out

n_cont = N_CONT_BASE + N_EXTRA
tr_blk = assemble(tr_c, use_edges=True)
va_blk = assemble(va_c, use_edges=True)
te_blk = assemble(te_c, use_edges=True)

def run_variant(name, tr_data, va_data, te_data):
    model = IdsAvgGNN(NUM_TYPES, n_cont).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_va = -1e9; best_sd = None

    def eval_blocks(datas):
        model.eval()
        ps, ts = [], []
        with torch.no_grad():
            for ty, co, ei, s_i, s_y in datas:
                ty = ty.to(DEV); co = co.to(DEV); ei = ei.to(DEV); s_i = s_i.to(DEV)
                pred = model(ty, co, ei, ty.shape[0])[s_i]
                ps.append(pred.cpu().numpy()); ts.append(s_y.numpy())
        if not ps: return -9, -9, 0
        ps = np.concatenate(ps); ts = np.concatenate(ts)
        r2 = 1-np.sum((ts-ps)**2)/np.sum((ts-ts.mean())**2)
        rho, _ = spearmanr(ts, ps)
        return r2, rho, len(ts)

    for ep in range(EPOCHS):
        model.train()
        el, nb = 0.0, 0
        for ty, co, ei, s_i, s_y in tr_data:
            ty = ty.to(DEV); co = co.to(DEV); ei = ei.to(DEV)
            s_i = s_i.to(DEV); s_y = s_y.to(DEV)
            pred = model(ty, co, ei, ty.shape[0])[s_i]
            loss = F.mse_loss(pred, s_y)
            opt.zero_grad(); loss.backward(); opt.step()
            el += loss.item(); nb += s_y.shape[0]
        sched.step()
        if (ep+1) % 4 == 0 or ep == 0:
            vr2, vrho, n = eval_blocks(va_data)
            print(f'  [{name}] ep {ep+1:2d}  train_loss={el/max(nb,1):.5f}  val_R^2={vr2:.4f} (n={n})', flush=True)
            if vr2 > best_va:
                best_va = vr2
                best_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_sd)
    tr2, trho, tn = eval_blocks(tr_data)
    te2, teho, ten = eval_blocks(te_data)
    print(f'  [{name}] train R^2={tr2:.4f} / test(未见电路) R^2={te2:.4f} Spearman={teho:.4f} (n={ten})', flush=True)
    return te2, teho

print('\n===== V. DelayGNN 复刻 idsavg GNN =====', flush=True)
g_te, g_rho = run_variant('V_gnn', tr_blk, va_blk, te_blk)

# 无边对照: 同特征同模型, 仅去掉消息传递 (per-node MLP, 检验「图传播」本身的价值)
print('\n===== W. 无边对照 (同特征, per-node MLP) =====', flush=True)
w_tr = assemble(tr_c, use_edges=False); w_va = assemble(va_c, use_edges=False)
w_te = assemble(te_c, use_edges=False)
te_te2, te_rho2 = run_variant('V_nograph', w_tr, w_va, w_te)

print('\n判定 (唯一尺度 ids_avg R^2, 电路级切分):')
print(f'  A  GBDT15       = 0.6740 (参考)')
print(f'  V  gnn          = {g_te:.4f}')
print(f'  W  nograph(MLP) = {te_te2:.4f}')
print('  若 gnn 显著>nograph => 图传播携带 ids_avg 信号; 若 gnn>=GBDT15 => 值得做 serve 端真模型.')
