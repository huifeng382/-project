"""serve.py — GNN 推理服务：为 Rust 候选电路排序（no-wave 粗筛）。

设计：复用 15.1.0 的任意 I/O 图构建 + JSON pin 特征逻辑。给定一个候选电路
（网表 + 输入/输出引脚），为每个 (切换引脚, direction) 生成一行特征，逐行预测延迟，
对电路取平均 = avg_delay（对齐 Rust `simulate_all_outputs_for_expr`），据此给候选排序。

Rust 粗筛流程（5.6 阶段 1，极简特征、不依赖 Rust 额外输出）：
  - corner 固定 s02p0_l01p0（2ps slew / 1fF load）
  - 每 (pin, dir) 一行，vector 默认全 0、切换位按 direction（rise→0 / fall→1）
  - gate_states 用逻辑仿真 BFS 推算（推理时 Rust 无此列）

用法：
  python scripts/diag/serve.py --ckpt <model.pt> --scaler outputs/scaler.pkl \
      --in candidates.json --out ranked.json
candidates.json: [{"id": "...", "netlist": "...", "input_pins": [...], "output_pins": [...]}]
ranked.json:     [{"id": "...", "avg_delay": 3.2e-11}, ...] 按 avg_delay 升序
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch

# 强制对齐 no-wave 推理配置（serve 加载的是 no-wave 模型，不能开 wave/supply_noise）
import config
config.USE_TRANSISTOR_WAVE = False
config.USE_SUPPLY_NOISE = False
config.USE_PARASITIC_CAPS = False
config.USE_V2 = True
# 16.11.3/16.11.35: 近似 ids_avg 特征。v2iaa 系 checkpoint 用 USE_IDS_AVG_APPROX=1（线性系数）、
# v2iag 系用 =2（GBDT15 joblib）训练。推理须按同一模式算每门近似 ids_avg 拼进 x 的绝对末维，
# in_dim 才能对齐（46 = base+7+1）。config.USE_IDS_AVG_APPROX 作布尔（是否启用近似特征列）；
# 具体模式存模块级 _APPROX_MODE（'0'/'1'/'2'）。
_APPROX_MODE = os.environ.get('USE_IDS_AVG_APPROX', '0')
config.USE_IDS_AVG_APPROX = _APPROX_MODE != '0'
# STRUCT_MODE 必须与 checkpoint 训练一致（no-wave 交付用 logic_only，in_dim 才能对上）；
# 可被 STRUCT_MODE 环境变量覆盖（服务其它 STRUCT_MODE 的 checkpoint 时）
config.STRUCT_MODE = os.environ.get('STRUCT_MODE', 'logic_only')

from src.model import DelayGNN
from src.graph_builder import build_static_graph, rebuild_gate_types
import src.graph_builder as gb
from src.utils import load_scaler

SLEW_S = 2e-12      # Rust asap7.sp 固定 2ps
LOAD_F = 1e-15      # 1fF
CORNER = (2.0, 1.0) # (slew_cond, load_cond) -> corner_cond


def set_gate_logic_overrides(mapping):
    """设置 {门名 -> 逻辑类} 覆盖（Rust 侧随候选传入；thread-local，多线程安全）。"""
    gb.set_gate_logic_overrides(mapping)


def _load_cell_types_from_netlist(netlist):
    """从网表 X_ 行收集门类型（用于重建词表；STRUCT_MODE 下实际用逻辑类别，兼容性调用）。"""
    types = set()
    for line in (netlist or '').split('\n'):
        s = line.strip()
        if s.startswith('X_') and len(s.split()) >= 3:
            types.add(s.split()[-1])
    return types


def build_candidate_tensors(netlist, input_pins, output_pins, scaler, vector=None):
    """建单个候选电路的静态图 + 图级特征。返回 (node_names, node_static, edge_index, input_pins, output_pins)。"""
    rebuild_gate_types(_load_cell_types_from_netlist(netlist))
    node_names, node_static, edge_index = build_static_graph(
        'serve', netlist, list(input_pins), list(output_pins))
    return node_names, node_static, edge_index, list(input_pins), list(output_pins)


def row_dynamic_features(pins, switching, direction, global_slew, global_load, scaler, vector_str=None):
    """镜像 data_loader._get_dynamic_features 的 per-pin 特征（无 pin_slew_json，用固定值回退）。"""
    switching_before = 0.0 if direction == 'rise' else 1.0
    if vector_str is None:
        v = ['0'] * len(pins)
        v[pins.index(switching)] = '1' if direction == 'fall' else '0'
        vector_str = ''.join(v)
    dyn = {}
    for pin in pins:
        slew = global_slew if pin == switching else 0.0
        load = global_load
        arrival = 0.0
        if pin == switching:
            logic = switching_before
        else:
            try:
                bi = pins.index(pin)
                logic = float(vector_str[bi]) if bi < len(vector_str) else 0.5
            except (ValueError, IndexError):
                logic = 0.5
        feat = [logic, 1.0 if pin == switching else 0.0, slew, load, global_load, arrival, 0.0]
        if scaler is not None:
            cont = np.array([feat[2], feat[3], feat[4], feat[5]]).reshape(1, -1)
            sc = scaler.transform(cont)[0]
            feat[2], feat[3], feat[4], feat[5] = sc[0], sc[1], sc[2], sc[3]
        dyn[pin] = feat
    return dyn, vector_str


def _gate_states_bfs(node_names, node_static, edge_index, vector_str, pins, switching, outs):
    """推理时无 gate_states_json：用逻辑仿真 BFS 推算（与训练 fallback 一致）。
    outs：输出节点名（Rust 候选用真实输出引脚，非固定 'out'）。"""
    from src.logic_sim import compute_gate_states
    from src.graph_builder import GATE_TYPES
    node_types = {}
    for j, n in enumerate(node_names):
        ti = int(node_static[j, 0].item())
        node_types[n] = GATE_TYPES[ti] if ti < len(GATE_TYPES) else 'UNKNOWN'
    try:
        gs = compute_gate_states(node_names, node_types, edge_index, vector_str, list(pins), switching,
                                 outputs=list(outs))
        return gs if isinstance(gs, dict) else {}
    except Exception:
        return {}


_GBDT15 = None


def _load_gbdt15():
    """加载 GBDT15 ids_avg 模型（USE_IDS_AVG_APPROX=2 推理用）。路径解析与 data_loader 一致
    （优先 ~/-project/outputs/，再回退 serve 仓 outputs/）。缺失/失败返回 None（上层回退线性近似）。"""
    global _GBDT15
    if _GBDT15 is not None:
        return _GBDT15
    try:
        import joblib
        _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        _cands = [
            os.path.join(os.path.expanduser('~'), '-project', 'outputs', 'idsavg_gbdt15.joblib'),
            os.path.join(_root, 'outputs', 'idsavg_gbdt15.joblib'),
        ]
        _p = next((p for p in _cands if os.path.exists(p)), None)
        if _p is None:
            print('[serve] WARN: idsavg_gbdt15.joblib 缺失，回退线性近似')
            return None
        _d = joblib.load(_p)
        _GBDT15 = _d['model'] if isinstance(_d, dict) else _d
        print(f'[serve] GBDT15 ids_avg 模型已加载: {_p}')
        return _GBDT15
    except Exception as e:
        print(f'[serve] WARN: GBDT15 加载失败 ({e})，回退线性近似')
        return None


def _approx_ids_avg_vector(node_static):
    """每节点绝对末维近似 ids_avg（零仿真）。Rust shadow 固定 corner s02p0_l01p0（2ps slew/1fF load）。
    mode '2' = GBDT15（15 维特征布局与 data_loader L532-556 一致）；'1'（或 '2' 但 joblib 缺失）= 线性系数。
    node_static: [N, base] 静态特征（列序与训练 build_static_graph 一致）。"""
    import math
    import numpy as _np
    if hasattr(node_static, 'detach'):
        node_static = node_static.detach().cpu().numpy()
    else:
        node_static = _np.asarray(node_static)
    n = node_static.shape[0]
    f_slew = math.log1p(SLEW_S * 1e12)     # 2ps
    f_load = math.log1p(LOAD_F * 1e15)     # 1fF
    f_cslew = math.log1p(CORNER[0])        # 2.0
    f_cload = math.log1p(CORNER[1])        # 1.0
    if _APPROX_MODE == '2' and _load_gbdt15() is not None:
        gb = _GBDT15
        xv = _np.zeros((n, 15), dtype=_np.float64)
        xv[:, 0] = f_slew; xv[:, 1] = f_load
        xv[:, 6] = f_cslew; xv[:, 7] = f_cload
        xv[:, 3] = _np.array([math.log1p(float(node_static[i, 4])) for i in range(n)])   # par 已 log1p? 否→log1p
        xv[:, 2] = _np.array([math.log1p(float(node_static[i, 3])) for i in range(n)])   # drive
        xv[:, 4] = _np.array([float(node_static[i, 1]) for i in range(n)])               # fan(已 log1p)
        xv[:, 5] = _np.array([float(node_static[i, 6]) for i in range(n)])               # h(已 log1p)
        xv[:, 8] = xv[:, 2] * xv[:, 1]
        xv[:, 9] = xv[:, 2] * xv[:, 5]
        xv[:, 10] = xv[:, 4] * xv[:, 3]
        xv[:, 11] = xv[:, 0] * xv[:, 1]
        _fan = _np.expm1(xv[:, 4])
        _drv = _np.maximum(_np.expm1(xv[:, 2]), 1e-6)
        _par_f = _np.expm1(xv[:, 3])
        _tau = (1.0 / _drv) * _np.maximum(_par_f * 1e-15, 1e-18)
        xv[:, 12] = _np.log1p(_tau * 1e15)
        xv[:, 13] = _np.log1p(_np.maximum(_par_f * _fan, 1e-9))
        xv[:, 14] = _np.log1p(_np.maximum(1.0 / (1.0 + _fan), 1e-9))
        return _np.expm1(gb.predict(xv))
    # '1' 线性（或 '2' 但 joblib 缺失 → 回退线性）
    C = config.IDS_AVG_APPROX_COEF
    out = _np.zeros(n, dtype=_np.float64)
    for i in range(n):
        ns = node_static[i]
        f_drive = math.log1p(float(ns[3]))
        f_par = math.log1p(float(ns[4]))
        f_fan = float(ns[1])             # 已 log1p(fanout)
        f_h = float(ns[6])               # 已 log1p(电努力)
        lg = (C[0] * f_slew + C[1] * f_load + C[2] * f_drive + C[3] * f_par
              + C[4] * f_fan + C[5] * f_h + C[6] * f_cslew + C[7] * f_cload + C[8])
        out[i] = math.expm1(lg)
    return out


def _per_model_avg_delays(models, node_names, node_static, edge_index, pins, outs, scaler, device):
    """单候选、共享静态图：返回每模型的 avg_delay（list，长度 = len(models)）。
    每 (pin,dir) 行对每模型做一次前向，行内多模型求均值 → 电路级平均。
    16.11.35: 近似 ids_avg（mode '1'=线性 / '2'=GBDT15）在 x 绝对末维，in_dim = base+7+1，与训练一致。"""
    num_nodes = len(node_names)
    num_dyn = 7                        # 动态特征固定 7 维（含 gate_states 在动态末位）
    extra_dim = 1 if config.USE_IDS_AVG_APPROX else 0
    base = node_static.shape[1]
    # 末维近似 ids_avg 对同一候选所有 (pin,dir) 行相同（Rust corner 固定 2ps/1fF）→ 预算一次
    approx_ids = _approx_ids_avg_vector(node_static) if config.USE_IDS_AVG_APPROX else None
    if approx_ids is not None:
        approx_ids = torch.as_tensor(approx_ids, dtype=node_static.dtype, device=device)
    node_static = node_static.to(device)
    edge_index = edge_index.to(device)
    acc = np.zeros(len(models))
    n_rows = 0
    for switching in pins:
        for direction in ('rise', 'fall'):
            dyn, vector_str = row_dynamic_features(pins, switching, direction, SLEW_S, LOAD_F, scaler)
            x = torch.zeros((num_nodes, base + num_dyn + extra_dim), device=device)
            x[:, :base] = node_static
            for i, n in enumerate(node_names):
                if n in dyn:
                    x[i, base:base + num_dyn] = torch.tensor(dyn[n], dtype=torch.float, device=device)
            # 绝对末维 = 近似 ids_avg（零仿真，_approx_ids_avg_vector 按 _APPROX_MODE 算好）
            if approx_ids is not None:
                x[:, -1] = approx_ids
            gs = _gate_states_bfs(node_names, node_static, edge_index, vector_str, pins, switching, outs)
            gs_lc = {str(k).lower(): v for k, v in gs.items()}
            # gate_states 写在动态 7 维的末位（与训练一致）
            dyn_last = base + num_dyn - 1
            for i, n in enumerate(node_names):
                if str(n).lower() in gs_lc:
                    x[i, dyn_last] = float(gs_lc[str(n).lower()])
                elif n in outs:
                    x[i, dyn_last] = 1.0
            with torch.no_grad():
                corner = torch.tensor([CORNER], dtype=torch.float, device=device)
                csig = torch.tensor([[float(num_nodes), 0.0, float(len(pins))]],
                                    dtype=torch.float, device=device)
                for mi, m in enumerate(models):
                    m.eval()
                    out, _ = m(x, edge_index, torch.zeros(num_nodes, dtype=torch.long, device=device),
                               corner, csig, None)
                    acc[mi] += float((10 ** out.cpu()).clamp(1e-12, 1e-8).item())
            n_rows += 1
    if n_rows == 0:
        return [float('nan')] * len(models)
    return (acc / n_rows).tolist()


def predict_avg_delay(models, netlist, input_pins, output_pins, scaler, device,
                      gate_logics=None):
    """预测单个候选电路的 avg_delay（每 (pin,dir) 行延迟的线性平均，对齐 Rust）。
    models：list[DelayGNN] —— 多模型（集成）时对每行预测取平均。
    gate_logics：可选 {门名 -> 逻辑类}，Rust 侧传入时覆盖 sc_expansion 查不到的门的逻辑类。"""
    set_gate_logic_overrides(gate_logics)
    node_names, node_static, edge_index, pins, outs = build_candidate_tensors(
        netlist, input_pins, output_pins, scaler)
    pm = _per_model_avg_delays(models, node_names, node_static, edge_index, pins, outs, scaler, device)
    return float(np.mean(pm)) if pm else float('nan')


def predict_rank_batch(models, cands, scaler, device):
    """批量候选秩聚合（方案 1，15.3.x）：每候选先得每模型 avg_delay，再在候选集内按模型排名，
    取平均秩（competition ranking，并列取平均）作为得分——得分越低 = 共识越快。
    返回 [{id, avg_delay: 平均秩得分}]，按得分升序（快→慢）。"""
    rows = []
    for c in cands:
        set_gate_logic_overrides(c.get('gate_logics'))
        try:
            nn, ns, ei, pins, outs = build_candidate_tensors(
                c.get('netlist', ''), c.get('input_pins', []), c.get('output_pins', []), scaler)
            pm = _per_model_avg_delays(models, nn, ns, ei, pins, outs, scaler, device)
        except Exception:
            pm = [float('nan')] * len(models)
        rows.append({'id': c.get('id', '?'), 'pm': pm, 'nan': any(v != v for v in pm)})
    n = len(rows)
    if n == 0:
        return []
    if n == 1:
        return [{'id': r['id'], 'avg_delay': float(np.mean(r['pm']))} for r in rows]
    # 秩聚合：逐模型排名 → 平均秩（低 = 快）
    scores = np.zeros(n)
    counted = np.zeros(n)
    for mi in range(len(models)):
        vals = np.array([r['pm'][mi] for r in rows], dtype=np.float64)
        good = [i for i in range(n) if not rows[i]['nan']]
        if not good:
            continue
        gvals = vals[good]
        order = np.argsort(gvals, kind='mergesort')
        ranks = np.empty(len(good))
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and gvals[order[j + 1]] == gvals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        for idx, rk in zip(good, ranks):
            scores[idx] += rk
            counted[idx] += 1
    out = []
    for i, r in enumerate(rows):
        sc = scores[i] / counted[i] if counted[i] > 0 else float('nan')
        out.append({'id': r['id'], 'avg_delay': float(sc)})
    out.sort(key=lambda x: (x['avg_delay'] != x['avg_delay'], x['avg_delay']))
    return out


def load_models(ckpt_paths, scaler, sample_netlist, sample_pins, sample_outs):
    """加载 N 个 checkpoint（等权集成）。返回 (models, in_dim)。in_dim 由 sample 图维度决定。"""
    rebuild_gate_types(_load_cell_types_from_netlist(sample_netlist))
    _, ns, _, _, _ = build_candidate_tensors(sample_netlist, sample_pins, sample_outs, scaler)
    import src.graph_builder as gb
    extra_dim = 1 if config.USE_IDS_AVG_APPROX else 0   # 16.11.3: v2iaa 近似 ids_avg 维
    in_dim = ns.shape[1] + 7 + extra_dim   # 7 动态特征 (+近似 ids_avg)
    models = []
    for ck in ckpt_paths:
        m = DelayGNN(in_dim=in_dim, hidden_dim=config.HIDDEN_DIM, num_layers=config.NUM_LAYERS,
                     dropout=config.DROPOUT, num_gate_types=len(gb.GATE_TYPES),
                     gate_embed_dim=config.GATE_EMBED_DIM)
        m.load_state_dict(torch.load(ck, map_location='cpu', weights_only=False))
        models.append(m)
    return models, in_dim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', nargs='+', required=True,
                    help='no-wave checkpoint(s)（.pt，可多个做等权集成，如 6-seed）')
    ap.add_argument('--scaler', default=os.path.join('outputs', 'scaler.pkl'), help='scaler.pkl 路径')
    ap.add_argument('--in', dest='inp', required=True, help='候选 JSON 列表')
    ap.add_argument('--out', default='ranked.json', help='输出排序 JSON')
    args = ap.parse_args()

    with open(args.inp, encoding='utf-8') as f:
        cands = json.load(f)
    scaler = load_scaler(args.scaler) if os.path.exists(args.scaler) else None

    # 用第一个候选的图维度确定 in_dim（STRUCT_MODE 需与 checkpoint 一致）
    c0 = cands[0]
    models, in_dim = load_models(args.ckpt, scaler, c0.get('netlist'),
                                 c0.get('input_pins', []), c0.get('output_pins', []))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for m in models:
        m.to(device)
    print(f"[serve] {len(models)} 个模型等权集成, in_dim={in_dim}, STRUCT_MODE={config.STRUCT_MODE}")

    results = []
    for c in cands:
        try:
            ad = predict_avg_delay(models, c['netlist'], c.get('input_pins', []),
                                   c.get('output_pins', []), scaler, device)
            results.append({'id': c.get('id', c.get('circuit_id', '?')), 'avg_delay': ad})
        except Exception as e:
            results.append({'id': c.get('id', '?'), 'avg_delay': None, 'error': str(e)})
    results.sort(key=lambda r: (r['avg_delay'] is None, r['avg_delay']))
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    for r in results:
        print(f"  {r['id']}: avg_delay={r['avg_delay']}")
    print(f"[serve] 已写入 {args.out}")


if __name__ == '__main__':
    main()
