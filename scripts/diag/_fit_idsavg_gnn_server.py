"""服务器全量版: 复刻 DelayGNN → 独立 per-gate ids_avg 回归 (DATA_BATCHES = full+rest+m4 同 delayGNN 训练域)

与 _fit_idsavg_gnn.py 同架构/同口径 (电路级切分, 唯一尺度 = ids_avg R^2/Spearman), 差异:
  - 数据 = DATA_BATCHES (默认 batch_v2_full,batch_v2_rest,batch_v2_m4; 与 config.DATA_BATCHES 一致)
    * rest 批次为 *_partN.parquet 分片, 自动 glob
  - 行按电路预分组 (groupby), 避免 45k 电路逐行过滤
  - 测试按批次来源分桶 (full/rest/m4), 验证 m4 等未见形态的泛化

本地受控对照 (1500 电路, batch_v2_full): GBDT15=0.6740 / gnn=0.7697 / nograph=0.6773
  => 图传播携带 per-gate ids_avg 信号; 本脚本在全量数据 (含 m4) 上复核.

用法 (服务器, 训练在服务器跑 = 用户执行):
  DATA_BATCHES='batch_v2_full,batch_v2_rest,batch_v2_m4' python scripts/diag/_fit_idsavg_gnn_server.py
  可选: N_CAP=<电路数上限> 快速验证; EPOCHS=<n> 默认 45; NO_NOGRAPH=1 跳过无边对照(省~一半时间)
  17.0.5: K/HID/EMB/DROPOUT/LR/PATIENCE 可 env 覆盖(0=不早停); CONE_FEAT=1 追加锥体/距离通道(见 17.0.5 记录)
"""
import sys, os, json, math, time, glob as _glob
from collections import deque
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

DATA_ROOT = os.environ.get('DATA_ROOT', 'data')
DATA_BATCHES = os.environ.get('DATA_BATCHES', 'batch_v2_full,batch_v2_rest,batch_v2_m4')
N_CAP = int(os.environ.get('N_CAP', '0') or 0)     # 0=全部电路
EPOCHS = int(os.environ.get('EPOCHS', '45'))
NO_NOGRAPH = os.environ.get('NO_NOGRAPH', '0') == '1'
CIRC_SPLIT = (0.85, 0.05, 0.10)
# GNN 超参 (同 DelayGNN 复刻); 17.0.5 起全部可 env 覆盖 (A/B 对照用)
EMB = int(os.environ.get('EMB', '16')); HID = int(os.environ.get('HID', '96'))
K = int(os.environ.get('K', '3')); DROPOUT = float(os.environ.get('DROPOUT', '0.25'))
LR = float(os.environ.get('LR', '3e-3'))
PATIENCE = int(os.environ.get('PATIENCE', '0') or 0)   # 0 = 不早停 (沿用历史行为)
CONE_FEAT = os.environ.get('CONE_FEAT', '0') == '1'     # B: 追加锥体/距离通道 (N_EXTRA 7->10)
N_CONT_BASE = 7; N_EXTRA = 10 if CONE_FEAT else 7

def parse_corner(corner):
    try:
        s, l = str(corner).split('_')[:2]
        return float(s[1:].replace('p', '.')), float(l[1:].replace('p', '.'))
    except Exception:
        return 5.0, 10.0

def cell_types(nl):
    return {ln.split()[-1] for ln in (nl or '').split('\n')
            if ln.strip().startswith('X_') and len(ln.split()) >= 3}

def resolve_parquets(batch):
    """返回 (static_paths, dynamic_paths), 支持 *_partN.parquet 分片 (对齐 train_sweep 逻辑)."""
    sp = os.path.join(DATA_ROOT, f'{batch}/circuit_static.parquet')
    dp = os.path.join(DATA_ROOT, f'{batch}/timing_arcs.parquet')
    if os.path.exists(sp) and os.path.exists(dp):
        return [sp], [dp]
    sparts = sorted(_glob.glob(os.path.join(DATA_ROOT, f'{batch}/circuit_static_part*.parquet')))
    dparts = sorted(_glob.glob(os.path.join(DATA_ROOT, f'{batch}/timing_arcs_part*.parquet')))
    if sparts and dparts:
        return sparts, dparts
    raise FileNotFoundError(f'batch {batch}: no static/dynamic parquet found under {DATA_ROOT}')

# ============================================================ 载入多批次
_batch_names = [b.strip() for b in DATA_BATCHES.split(',') if b.strip()]
static_dfs, dyn_dfs = [], []
batch_of_circ = {}     # circuit_id -> 来源批次 (第一命中为准; 与 drop_duplicates 序一致)
for batch in _batch_names:
    try:
        sps, dps = resolve_parquets(batch)
    except FileNotFoundError as e:
        print(f'WARN: {e}, skip'); continue
    s = pd.read_parquet(sps[0] if len(sps) == 1 else sps) if len(sps) == 1 else pd.concat([pd.read_parquet(p) for p in sps])
    d = pd.concat([pd.read_parquet(p, columns=['circuit_id', 'transistor_wave_json', 'slew_s',
                                               'output_load_f', 'corner', 'direction',
                                               'switching_pin', 'output']) for p in dps], ignore_index=True)
    static_dfs.append(s); dyn_dfs.append(d)
    for c in s['circuit_id'].astype(str):
        batch_of_circ.setdefault(c, batch)
    print(f'batch {batch}: static={len(s)} circuits / dynamic={len(d)} rows', flush=True)

sdf = pd.concat(static_dfs).drop_duplicates('circuit_id')
sdf['circuit_id'] = sdf['circuit_id'].astype(str)
sdf = sdf.set_index('circuit_id')
ddf = pd.concat(dyn_dfs, ignore_index=True)
ddf['circuit_id'] = ddf['circuit_id'].astype(str)
ddf = ddf[ddf['transistor_wave_json'].notna()]
print(f'总: 电路 {len(sdf)} / 行 {len(ddf)}', flush=True)

# 电路级切分
circ_all = [c for c in sdf.index if c in set(ddf['circuit_id'])]
if N_CAP > 0:
    circ_all = np.random.RandomState(42).choice(circ_all, size=min(N_CAP, len(circ_all)), replace=False).tolist()
circ_all = list(circ_all)
order = np.asarray(circ_all); rp = np.random.RandomState(7).permutation(len(order))
n1 = int(len(order)*CIRC_SPLIT[0]); n2 = n1 + int(len(order)*CIRC_SPLIT[1])
tr_c = set(order[rp[:n1]].tolist()); va_c = set(order[rp[n1:n2]].tolist()); te_c = set(order[rp[n2:]].tolist())
# 有序列表 (set 迭代序不确定, assemble/分桶一律用有序列表保证块顺序可复现)
tr_l = [c for c in circ_all if c in tr_c]
va_l = [c for c in circ_all if c in va_c]
te_l = [c for c in circ_all if c in te_c]
print(f'电路级: train {len(tr_l)} / val {len(va_l)} / test {len(te_l)}', flush=True)

# 行预分组 (只保留需要用到的电路)
use = tr_c | va_c | te_c
ddf = ddf[ddf['circuit_id'].isin(use)]
rowg = {cid: g for cid, g in ddf.groupby('circuit_id')}
del ddf

# ============================================================ 逐电路建图 + 收集样本/块
Xs, ys, row_m = [], [], []
blocks = {}; max_type = 0

t0 = time.time()
for ci, cid in enumerate(circ_all):
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
    edges = [(int(a), int(b)) for a, b in ei.t().tolist() if a != b]
    rows_df = rowg.get(cid)
    if rows_df is None:
        continue
    blk = []
    for _, r in rows_df.iterrows():
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
        sup = []
        for i, n in enumerate(node_names):
            gk = str(n).lower()
            if gk not in gate_avg: continue
            real = float(np.mean(gate_avg[gk]))
            if real <= 0: continue
            y = math.log1p(real)
            sup.append((i, y))
            drive, par = float(ns[i, 3]), float(ns[i, 4])
            fan, hh = float(np.expm1(ns[i, 1])), float(np.expm1(ns[i, 6]))
            dep, g = float(ns[i, 2]), float(ns[i, 5])
            f_d, f_p = math.log1p(drive), math.log1p(par)
            f_fan, f_h = math.log1p(fan), math.log1p(hh)
            r_on = 1.0/max(drive, 1e-6); c_l = max(par*1e-15, 1e-18)
            Xs.append([f_slew, f_load, f_d, f_p, f_fan, f_h, f_cs, f_cl,
                       f_d*f_load, f_d*f_h, f_fan*f_p, f_slew*f_load,
                       math.log1p(r_on*c_l*1e15), math.log1p(max(par*fan, 1e-9)),
                       math.log1p(max(1.0/(1.0+fan), 1e-9)), dep, g])
            ys.append(y); row_m.append(cid)
        if not sup: continue
        blk.append({'edges': edges, 'f_slew': f_slew, 'f_load': f_load, 'f_cs': f_cs,
                    'f_cl': f_cl, 'dir_code': dir_code, 'src_i': src_i, 'out_i': out_i, 'sup': sup})
    if blk:
        _be = {'ns': ns, 'rows': blk, 'batch': batch_of_circ.get(cid, '?')}
        if CONE_FEAT and edges:      # B: 每电路邻接表一次建好, assemble 三份 (tr/va/te) 复用
            _adj = {}; _radj = {}
            for _a, _b in edges:
                _adj.setdefault(_a, []).append(_b); _radj.setdefault(_b, []).append(_a)
            _be['adj'] = _adj; _be['radj'] = _radj
        blocks[cid] = _be
    if (ci+1) % 5000 == 0:
        print(f'  电路 {ci+1}/{len(circ_all)}: GBDT样本 {len(ys)} / 块 {len(blocks)}, {time.time()-t0:.0f}s', flush=True)

Xs = np.array(Xs); ys = np.array(ys)
NUM_TYPES = max_type + 2
print(f'\n样本 {len(ys)} (row,gate); 电路块 {len(blocks)}; max_type={max_type}', flush=True)

_tr_ns = np.vstack([blocks[c]['ns'][:, 1:1+N_CONT_BASE] for c in tr_c if c in blocks])
_sta_mean = _tr_ns.mean(0); _sta_std = _tr_ns.std(0) + 1e-6

def masks(cids):
    m = np.array([c in cids for c in row_m])
    return m
mtr = masks(tr_c); mva = masks(va_c); mte = masks(te_c)
print('GBDT 样本量 train/val/test:', mtr.sum(), mva.sum(), mte.sum(), flush=True)

# ============================================================ A. GBDT15 基线
print('\n===== A. GBDT15 (15特征, 部署同款, 电路级切分) =====', flush=True)
def gbdt_r2(name, Xtr, ytr, Xte, yte):
    gb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, random_state=42,
                                       validation_fraction=0.1, early_stopping=True, n_iter_no_change=40)
    gb.fit(Xtr, ytr)
    p = gb.predict(Xte)
    r2 = 1-np.sum((yte-p)**2)/np.sum((yte-yte.mean())**2)
    rho, _ = spearmanr(yte, p)
    print(f'  {name:34s} R^2={r2:.4f}  Spearman={rho:.4f}  n_iter={gb.n_iter_}', flush=True)
    return gb, p
gb, pte = gbdt_r2('A. GBDT15', Xs[mtr][:, :15], ys[mtr], Xs[mte][:, :15], ys[mte])

# test 每样本 GBDT 预测按电路聚合 (分桶复用; 避免逐电路 O(N) 扫描)
_carr = np.array(row_m)
gb_by_cid = {}
for j, pos in enumerate(np.where(mte)[0]):
    gb_by_cid.setdefault(_carr[pos], []).append(pte[j])

# ============================================================ DelayGNN 复刻骨架
class GraphConvL(nn.Module):
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

def _cone_dists(start, adj):
    """BFS 沿有向边 (driver->receiver) 从 start 出发: 返回 {节点: 跳数}. 无边时仅含 start."""
    dist = {start: 0}; dq = deque([start])
    while dq:
        u = dq.popleft()
        for v in adj.get(u, ()):
            if v not in dist:
                dist[v] = dist[u] + 1; dq.append(v)
    return dist

def assemble(cids, use_edges=True):
    out = []
    for cid in cids:
        b = blocks.get(cid)
        if not b: continue
        ns = b['ns']; N = ns.shape[0]; rows = b['rows']; R = len(rows)
        static = (ns[:, 1:1+N_CONT_BASE].astype(np.float32) - _sta_mean) / _sta_std
        typ = ns[:, 0].astype(np.int64)
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
            if row['src_i'] is not None: T[s+row['src_i'], N_CONT_BASE+5] = row['f_slew']
            if row['out_i'] is not None: T[s+row['out_i'], N_CONT_BASE+6] = row['f_load']
            if CONE_FEAT:
                # B: 锥体/距离通道 (N_EXTRA 已扩到 10, 原锚位 +5/+6 不变):
                #   +7 在 src 扇出锥内(含 src) | +8 沿有向边下游深度 (src=0, 最深=1)
                #   +9 反向(朝 out)上游深度 = 喂到该输出的路径深度; src_i/out_i=None => 全 0
                co = np.zeros((N, 3), dtype=np.float32)
                if row['src_i'] is not None:
                    d1 = _cone_dists(row['src_i'], b.get('adj', {}))
                    md1 = max(d1.values()) if d1 else 0
                    for g, dd in d1.items():
                        co[g, 0] = 1.0; co[g, 1] = dd / max(md1, 1)
                if row['out_i'] is not None:
                    d2 = _cone_dists(row['out_i'], b.get('radj', {}))
                    md2 = max(d2.values()) if d2 else 0
                    for g, dd in d2.items():
                        co[g, 2] = dd / max(md2, 1)
                T[s:e, N_CONT_BASE+7:N_CONT_BASE+10] = co
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
GSZ = int(os.environ.get('GSZ', '1'))      # ⚠ 默认 1 = 逐电路步进(已验证稳定)。GSZ>1 拼接提速为实验性(见 run_full_local 记录),暂勿开
def concat_group(group):
    tys, cos, eis, sis, sys_, n_off, has_e = [], [], [], [], [], 0, False
    for ty, co, ei, s_i, s_y in group:
        N = ty.shape[0]
        tys.append(ty); cos.append(co)
        if ei.numel():
            has_e = True; eis.append(ei + n_off)
        sis.append(s_i + n_off); sys_.append(s_y)
        n_off += N
    Ty = torch.cat(tys); Co = torch.cat(cos); Si = torch.cat(sis); Sy = torch.cat(sys_)
    E = torch.cat(eis, dim=1) if has_e else torch.zeros(2, 0, dtype=torch.long)
    return Ty, Co, E, Si, Sy

def run_variant(name, tr_data, va_data, te_data, out_ckpt=None):
    model = IdsAvgGNN(NUM_TYPES, n_cont).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_va = -1e9; best_sd = None; best_ep = 0
    def eval_blocks(datas):
        model.eval()
        ps, ts = [], []
        with torch.no_grad():
            for ty, co, ei, s_i, s_y in datas:
                ty = ty.to(DEV); co = co.to(DEV); ei = ei.to(DEV); s_i = s_i.to(DEV)
                ps.append(model(ty, co, ei, ty.shape[0])[s_i].cpu().numpy()); ts.append(s_y.numpy())
        if not ps: return -9, -9, 0
        ps = np.concatenate(ps); ts = np.concatenate(ts)
        r2 = 1-np.sum((ts-ps)**2)/np.sum((ts-ts.mean())**2)
        rho, _ = spearmanr(ts, ps)
        if os.environ.get('DEBUG') == '1':
            print(f'      [dbg] pred mean={ps.mean():.4f} std={ps.std():.4f} min={ps.min():.4f} max={ps.max():.4f} | y mean={ts.mean():.4f} std={ts.std():.4f} min={ts.min():.4f} max={ts.max():.4f}', flush=True)
        return r2, rho, len(ts)
    for ep in range(EPOCHS):
        model.train()
        el, nb = 0.0, 0
        for i in range(0, len(tr_data), GSZ):
            ty, co, ei, s_i, s_y = concat_group(tr_data[i:i+GSZ])
            ty = ty.to(DEV); co = co.to(DEV); ei = ei.to(DEV)
            s_i = s_i.to(DEV); s_y = s_y.to(DEV)
            pred = model(ty, co, ei, ty.shape[0])[s_i]
            loss = F.mse_loss(pred, s_y)
            opt.zero_grad(); loss.backward(); opt.step()
            el += loss.item(); nb += s_y.shape[0]
        sched.step()
        if (ep+1) % 5 == 0 or ep == 0:
            vr2, vrho, n = eval_blocks(va_data)
            print(f'  [{name}] ep {ep+1:2d}  train_loss={el/max(nb,1):.5f}  val_R^2={vr2:.4f} (n={n})', flush=True)
            if vr2 > best_va:
                best_va = vr2; best_ep = ep+1
                best_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
            elif PATIENCE > 0 and (ep+1) - best_ep >= PATIENCE:
                print(f'  [{name}] early stop @ ep {ep+1} (patience {PATIENCE}; best val_R^2={best_va:.4f} @ ep {best_ep})', flush=True)
                break
    model.load_state_dict(best_sd)
    tr2, trho, tn = eval_blocks(tr_data)
    te2, teho, ten = eval_blocks(te_data)
    print(f'  [{name}] best_val_R^2={best_va:.4f}  train R^2={tr2:.4f} / test(未见电路) R^2={te2:.4f} Spearman={teho:.4f} (n={ten})', flush=True)
    return model, te2, teho, best_va

print('\n===== V. DelayGNN 复刻 idsavg GNN (有向边消息传递) =====', flush=True)
tr_blk = assemble(tr_l, use_edges=True); va_blk = assemble(va_l, use_edges=True); te_blk = assemble(te_l, use_edges=True)
mdl, g_te, g_rho, g_bv = run_variant('V_gnn', tr_blk, va_blk, te_blk)

# 无边对照 (同特征 per-node MLP, 隔离「消息传递」贡献; 本地已证明, 全量可跳过省时)
w_te2 = w_rho2 = w_bv = None
if not NO_NOGRAPH:
    print('\n===== W. 无边对照 (同特征, 无边=per-node MLP) =====', flush=True)
    w_tr = assemble(tr_l, use_edges=False); w_va = assemble(va_l, use_edges=False); w_te = assemble(te_l, use_edges=False)
    _, w_te2, w_rho2, w_bv = run_variant('V_nograph', w_tr, w_va, w_te)

# ============================================================ 汇总 + 测试按批次分桶
print('\n===== 判定 (唯一尺度 ids_avg R^2, 电路级切分, 未见电路) =====', flush=True)
print(f'  A  GBDT15 (15特征, 部署同款)  test R^2 (上表)  -- 参考: batch_v2_full 电路级 0.674')
print(f'  V  gnn           best_val R^2={g_bv:.4f}  test R^2={g_te:.4f}  Spearman={g_rho:.4f}')
if w_te2 is not None:
    print(f'  W  nograph (MLP) best_val R^2={w_bv:.4f}  test R^2={w_te2:.4f}  Spearman={w_rho2:.4f}')
else:
    print('  W  nograph (MLP) skipped (NO_NOGRAPH=1; 本地对照已证 gnn-nograph=+0.09)')

# 测试按批次分桶 (GNN 与 GBDT15 同电路同序: te_blk 按 te_l 建, gnn 预测序 = 电路内 sup 序 = gb_by_cid)
import collections
per_b = collections.defaultdict(lambda: {'p_g': [], 'p_gb': [], 't': []})
with torch.no_grad():
    for cid, (ty, co, ei, s_i, s_y) in zip([c for c in te_l if c in blocks], te_blk):
        btag = blocks[cid]['batch']
        ty = ty.to(DEV); co = co.to(DEV); ei = ei.to(DEV); s_i = s_i.to(DEV)
        pred = mdl(ty, co, ei, ty.shape[0])[s_i].cpu().numpy()
        per_b[btag]['p_g'].extend(pred.tolist()); per_b[btag]['t'].extend(s_y.numpy().tolist())
        per_b[btag]['p_gb'].extend(gb_by_cid.get(cid, []))
print('\n  --- test 按批次来源分桶 (R^2 GNN vs GBDT15) ---')
for btag, d in per_b.items():
    if len(d['t']) < 10: continue
    t = np.array(d['t']); pg = np.array(d['p_g']); pgb = np.array(d['p_gb'])
    rg = 1-np.sum((t-pg)**2)/np.sum((t-t.mean())**2)
    rg2 = 1-np.sum((t-pgb)**2)/np.sum((t-t.mean())**2)
    print(f'    {btag:16s} n={len(t):6d}  GNN R^2={rg:.4f} | GBDT15 R^2={rg2:.4f}', flush=True)
