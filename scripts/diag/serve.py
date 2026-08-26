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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

# 强制对齐 no-wave 推理配置（serve 加载的是 no-wave 模型，不能开 wave/supply_noise）
import config
config.USE_TRANSISTOR_WAVE = False
config.USE_SUPPLY_NOISE = False
config.USE_PARASITIC_CAPS = False
config.USE_V2 = True
# STRUCT_MODE 必须与 checkpoint 训练一致（no-wave 交付用 logic_only，in_dim 才能对上）；
# 可被 STRUCT_MODE 环境变量覆盖（服务其它 STRUCT_MODE 的 checkpoint 时）
config.STRUCT_MODE = os.environ.get('STRUCT_MODE', 'logic_only')

from src.model import DelayGNN
from src.graph_builder import build_static_graph, rebuild_gate_types
from src.utils import load_scaler

SLEW_S = 2e-12      # Rust asap7.sp 固定 2ps
LOAD_F = 1e-15      # 1fF
CORNER = (2.0, 1.0) # (slew_cond, load_cond) -> corner_cond


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


def _gate_states_bfs(node_names, node_static, edge_index, vector_str, pins, switching):
    """推理时无 gate_states_json：用逻辑仿真 BFS 推算（与训练 fallback 一致）。"""
    from src.logic_sim import compute_gate_states
    from src.graph_builder import GATE_TYPES
    node_types = {}
    for j, n in enumerate(node_names):
        ti = int(node_static[j, 0].item())
        node_types[n] = GATE_TYPES[ti] if ti < len(GATE_TYPES) else 'UNKNOWN'
    try:
        gs = compute_gate_states(node_names, node_types, edge_index, vector_str, list(pins), switching)
        return gs if isinstance(gs, dict) else {}
    except Exception:
        return {}


def predict_avg_delay(models, netlist, input_pins, output_pins, scaler, device):
    """预测单个候选电路的 avg_delay（每 (pin,dir) 行延迟的线性平均，对齐 Rust）。
    models：list[DelayGNN] —— 多模型（集成）时对每行预测取平均。"""
    node_names, node_static, edge_index, pins, outs = build_candidate_tensors(
        netlist, input_pins, output_pins, scaler)
    node_static = node_static.to(device)
    edge_index = edge_index.to(device)
    num_nodes = len(node_names)
    num_dyn = 7
    base = node_static.shape[1]
    preds_lin = []
    for switching in pins:
        for direction in ('rise', 'fall'):
            dyn, vector_str = row_dynamic_features(pins, switching, direction, SLEW_S, LOAD_F, scaler)
            x = torch.zeros((num_nodes, base + num_dyn), device=device)
            x[:, :base] = node_static
            for i, n in enumerate(node_names):
                if n in dyn:
                    x[i, -num_dyn:] = torch.tensor(dyn[n], dtype=torch.float, device=device)
            gs = _gate_states_bfs(node_names, node_static, edge_index, vector_str, pins, switching)
            gs_lc = {str(k).lower(): v for k, v in gs.items()}
            for i, n in enumerate(node_names):
                if str(n).lower() in gs_lc:
                    x[i, -1] = float(gs_lc[str(n).lower()])
                elif n in outs:
                    x[i, -1] = 1.0
            with torch.no_grad():
                corner = torch.tensor([CORNER], dtype=torch.float, device=device)
                csig = torch.tensor([[float(len(node_names)), 0.0, float(len(pins))]],
                                    dtype=torch.float, device=device)
                row_preds = []
                for m in models:
                    m.eval()
                    out, _ = m(x, edge_index, torch.zeros(num_nodes, dtype=torch.long, device=device),
                               corner, csig, None)
                    row_preds.append(float((10 ** out.cpu()).clamp(1e-12, 1e-8).item()))
                preds_lin.append(float(np.mean(row_preds)))
    return float(np.mean(preds_lin)) if preds_lin else float('nan')


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
    rebuild_gate_types(_load_cell_types_from_netlist(c0.get('netlist')))
    node_names, node_static, _, _, _ = build_candidate_tensors(
        c0['netlist'], c0.get('input_pins', []), c0.get('output_pins', []), scaler)
    import src.graph_builder as gb
    in_dim = node_static.shape[1] + 7   # 7 动态特征
    models = []
    for ck in args.ckpt:
        m = DelayGNN(in_dim=in_dim, hidden_dim=config.HIDDEN_DIM, num_layers=config.NUM_LAYERS,
                     dropout=config.DROPOUT, num_gate_types=len(gb.GATE_TYPES),
                     gate_embed_dim=config.GATE_EMBED_DIM)
        m.load_state_dict(torch.load(ck, map_location='cpu', weights_only=False))
        models.append(m)
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
