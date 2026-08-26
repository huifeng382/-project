import torch
import numpy as np
from collections import deque
import re
import config

# 定义门类型映射（根据实际网表中出现的门类型扩充）
GATE_TYPES = [
    # 原有 batch01 的门类型
    'SC_AND', 'SC_AND_v1',
    'SC_JOIN_BRIDGE_WIRE_WIRE_WIRE_WIRE_WIRE_BRIDGE_WIRE_WIRE_WIRE_WIRE_WIRE',
    'SC_JOIN_BRIDGE_WIRE_WIRE_WIRE_WIRE_BRIDGE_WIRE_WIRE_WIRE_WIRE',
    'SC_JOIN_BRIDGE_WIRE_WIRE_WIRE_BRIDGE_WIRE_WIRE_WIRE',
    'SC_INV_WIRE', 'INPUT_PIN', 'OUTPUT_PIN',
    'SC_JOIN_BRIDGE__BRIDGE', 
    # batch02 新增的门类型
    'AND2x2_ASAP7_75t_R',
    'AND3x2_ASAP7_75t_R',
    'BUFx1_ASAP7_75t_R',
    'INVx1_ASAP7_75t_R',
    'NAND3x1_ASAP7_75t_R',
    'NAND5x1_ASAP7_75t_R',
    'NOR2x1_ASAP7_75t_R',
    'NOR3x1_ASAP7_75t_R',
    'NOR4x1_ASAP7_75t_R',
    'NOR5x1_ASAP7_75t_R',
    'OR2x2_ASAP7_75t_R',
    'OR3x2_ASAP7_75t_R',
    'TIEHIx1_ASAP7_75t_R',
    'TIELOx1_ASAP7_75t_R',
    # batch06 新增的 ASAP7 标准单元
    'AND8x2_ASAP7_75t_R',
    'NAND4x1_ASAP7_75t_R',
    'OR8x2_ASAP7_75t_R',
    'UNKNOWN_GATE',
]
GATE_TO_IDX = {gt: i for i, gt in enumerate(GATE_TYPES)}

# ---- 结构特征：固定逻辑类别替代 638 类 cell 名嵌入（通用、永不 OOV）----
LOGIC_TYPES = ['INV', 'NAND', 'NOR', 'AND', 'OR', 'AOI', 'OAI', 'XOR', 'BUF', 'COMPLEX']
LOGIC_TO_IDX = {t: i for i, t in enumerate(LOGIC_TYPES)}

import os as _os
import json as _json
import threading as _threading

_SC_EXPANSION = None

# Rust 侧门逻辑覆盖（serve 端按候选设置；thread-local 保证多线程 serve 下各请求互不干扰，
# 训练路径从不设置 -> 完全不受影响）。
_gate_logic_override = _threading.local()


def set_gate_logic_overrides(mapping):
    """serve 端设置 {门名 -> 逻辑类(10类)} 覆盖；mapping=None/{} 清空。"""
    _gate_logic_override.map = dict(mapping) if mapping else {}


def _load_sc_expansion():
    global _SC_EXPANSION
    if _SC_EXPANSION is None:
        _p = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                           'data', 'sc_expansion.json')
        try:
            _SC_EXPANSION = _json.load(open(_p, encoding='utf-8'))
        except Exception:
            _SC_EXPANSION = {}
    return _SC_EXPANSION


def _asap7_cell_feat(cell):
    """ASAP7 单元名 -> (逻辑函数, 输入数, 驱动, 晶体管数)。NAND2x1 -> ('NAND',2,1,4)。"""
    m = re.match(r'^([A-Z]+)(\d*)x(\d+)', cell)
    if not m:
        return None
    func, nin_s, drive_s = m.group(1), m.group(2), m.group(3)
    nin = int(nin_s) if nin_s else 1
    drive = int(drive_s)
    if func == 'INV':
        n_t = 2
    elif func == 'BUF':
        n_t = 4
    elif func in ('NAND', 'NOR'):
        n_t = 2 * nin
    elif func in ('AND', 'OR'):
        n_t = 2 * nin + 2
    elif func in ('XOR', 'XNOR'):
        n_t = 12
    else:
        n_t = 2 * max(nin, 1)
    return func, nin, drive, n_t


def _compose_logic(funcs):
    if not funcs:
        return 'COMPLEX'
    if len(funcs) == 1:
        f = funcs[0]
        return {'INV': 'INV', 'BUF': 'BUF', 'NAND': 'NAND', 'NOR': 'NOR',
                'XOR': 'XOR', 'XNOR': 'XOR'}.get(f, 'COMPLEX')
    if len(funcs) == 2 and funcs[-1] in ('INV', 'BUF'):
        if funcs[0] == 'NAND':
            return 'AND'
        if funcs[0] == 'NOR':
            return 'OR'
        if funcs[0] == 'INV':
            return 'BUF'
    return 'COMPLEX'


def _name_fallback_logic(name):
    gt = name.upper()
    # SC_JOIN_*/SC_BRIDGE_*/SC_*_WIRE_* 都是复杂门，归 COMPLEX
    if 'JOIN' in gt or 'BRIDGE' in gt or 'WIRE' in gt:
        return 'COMPLEX'
    if 'NAND' in gt:
        return 'NAND'
    if 'NOR' in gt:
        return 'NOR'
    if 'XOR' in gt or 'XNOR' in gt:
        return 'XOR'
    if 'INV' in gt:
        return 'INV'
    if 'BUF' in gt:
        return 'BUF'
    if 'AND' in gt:
        return 'AND'
    if 'OR' in gt:
        return 'OR'
    return 'COMPLEX'


def gate_struct(name):
    """cell 名 -> dict{logic, n_t, drive, stack, parallel}。
    优先用 sc_expansion.json 的 ASAP7 展开；查不到再查 Rust 侧覆盖（serve 端注入）；
    最后回退名字关键字。"""
    d = {'logic': 'COMPLEX', 'n_t': 6.0, 'drive': 1.0, 'stack': 1.0, 'parallel': 1.0}
    exp = _load_sc_expansion().get(name)
    if isinstance(exp, dict):
        sub = exp.get('subcircuit')
        if sub:
            feats = [f for f in (_asap7_cell_feat(x.get('cell', '')) for x in sub) if f is not None]
            if feats:
                d['logic'] = _compose_logic([f[0] for f in feats])
                d['n_t'] = float(sum(f[3] for f in feats))
                d['drive'] = float(feats[-1][2])
                d['stack'] = float(max(f[1] for f in feats))
                d['parallel'] = float(len(sub))
                return d
    # sc_expansion 无/空展开 → Rust 侧覆盖（只作用于 serve 端注入的名字，训练不受影响）
    ov = getattr(_gate_logic_override, 'map', None) or {}
    if name in ov:
        d['logic'] = ov[name]
        return d
    d['logic'] = _name_fallback_logic(name)
    return d


def _logic_p_g(logic_type, num_inputs):
    """从逻辑类别 + 输入数算逻辑努力参数 (p, g)。"""
    gt = logic_type.upper()
    n = max(int(num_inputs), 1)
    if gt == 'INV':
        p, g = 1.0, 1.0
    elif gt == 'NAND':
        p, g = float(n), (n + 2) / 3
    elif gt == 'NOR':
        p, g = float(n), (2 * n + 1) / 3
    elif gt == 'AND':
        p, g = float(n + 1), (n + 2) / 3
    elif gt == 'OR':
        p, g = float(n + 1), (2 * n + 1) / 3
    elif gt == 'XOR':
        p, g = 2.0 * n, 4.0
    elif gt == 'BUF':
        p, g = 1.0, 1.0
    else:
        p, g = 1.0, 1.0
    return p, g


def rebuild_gate_types(cell_types):
    """
    从实际数据中动态构建门类型映射。
    cell_types: 所有出现的 cell 类型字符串的可迭代集合。
    调用后 GATE_TYPES 和 GATE_TO_IDX 会被重建，包含所有 found types +
    INPUT_PIN / OUTPUT_PIN / UNKNOWN_GATE。
    """
    global GATE_TYPES, GATE_TO_IDX
    reserved = ['INPUT_PIN', 'OUTPUT_PIN', 'UNKNOWN_GATE']
    # 用固定逻辑类别（10 类）替代动态 cell 名词表；cell_types 参数保留（兼容调用）但不再用于构建词表
    GATE_TYPES = LOGIC_TYPES + reserved
    GATE_TO_IDX = {gt: i for i, gt in enumerate(GATE_TYPES)}

def parse_netlist(netlist_str, input_pins=None, output_pins=None):
    """解析网表 → (nodes, edges)。

    input_pins / output_pins：显式引脚列表（V2 从 circuit_static 的
    input_pins_json / output_pins_json 传入，支持任意 N 入 / M 出，含多输出）。
    缺省时从网表推导：输入取 .SUBCKT DUT 头（排除 vdd/gnd/vss），
    输出取「是某门输出、但从未被任何门当输入」的 sink net（V1 的 'out' 即此）。
    """
    lines = netlist_str.strip().split('\n')
    gates = {}
    wire_to_driver = {}

    for line in lines:
        stripped = line.strip()
        if not stripped.startswith('X_'):
            continue
        tokens = stripped.split()
        if len(tokens) < 3:
            continue
        inst = tokens[0]
        gtype = tokens[-1]
        io_tokens = tokens[1:-1]
        if len(io_tokens) < 1:
            continue
        output = io_tokens[-1]
        inputs = io_tokens[:-1]
        gates[inst] = {'type': gtype, 'inputs': inputs, 'output': output}
        wire_to_driver[output] = inst

    if input_pins is None or output_pins is None:
        header_pins = []
        for line in lines:
            s = line.strip()
            if s.upper().startswith('.SUBCKT DUT'):
                parts = s.split()
                if len(parts) >= 3:
                    header_pins = [p for p in parts[2:] if p.lower() not in ('vdd', 'gnd', 'vss')]
                break
        if output_pins is None:
            consumed = {inp for info in gates.values() for inp in info['inputs']}
            output_pins = [o for o in wire_to_driver if o not in consumed]
        if input_pins is None:
            input_pins = [p for p in header_pins if p not in output_pins]

    nodes = {}
    for inst, info in gates.items():
        nodes[inst] = {'type': info['type'], 'is_input': False, 'is_output': False}
    for pin in input_pins:
        nodes[pin] = {'type': 'INPUT_PIN', 'is_input': True, 'is_output': False}
    for op in output_pins:
        nodes[op] = {'type': 'OUTPUT_PIN', 'is_input': False, 'is_output': True}

    edges = []
    for inst, info in gates.items():
        for inp in info['inputs']:
            if inp in input_pins:
                edges.append((inp, inst))
    for inst, info in gates.items():
        for inp in info['inputs']:
            if inp in wire_to_driver and wire_to_driver[inp] != inst:
                edges.append((wire_to_driver[inp], inst))
    # 门输出 → 输出引脚节点（多输出：每个输出引脚一条边）
    for op in output_pins:
        if op in wire_to_driver:
            edges.append((wire_to_driver[op], op))

    edges = list(set(edges))
    return nodes, edges

def build_static_graph(circuit_id, netlist_str, input_pins=None, output_pins=None):
    nodes, edges = parse_netlist(netlist_str, input_pins, output_pins)
    node_names = list(nodes.keys())
    
    # 如果没有边，添加自环边
    if len(edges) == 0:
        edges = [(n, n) for n in node_names]
        print(f"WARNING: circuit {circuit_id} had no edges, added self-loops")
    
    # 构建 edge_index
    edge_index = []
    for u, v in edges:
        if u in node_names and v in node_names:
            edge_index.append([node_names.index(u), node_names.index(v)])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # 节点门类型索引 + 结构特征（按 STRUCT_MODE 决定用哪些）
    mode = getattr(config, 'STRUCT_MODE', 'base')
    node_type_idx = []
    struct_info = {}
    for n in node_names:
        gt = nodes[n]['type']
        if gt in ('INPUT_PIN', 'OUTPUT_PIN', 'UNKNOWN_GATE'):
            idx = GATE_TO_IDX.get(gt, GATE_TO_IDX['UNKNOWN_GATE'])
            struct_info[n] = {'logic': gt, 'n_t': 0.0, 'drive': 0.0, 'stack': 0.0, 'parallel': 0.0}
        else:
            si = gate_struct(gt)
            idx = GATE_TO_IDX.get(si['logic'], GATE_TO_IDX['UNKNOWN_GATE'])
            struct_info[n] = si
        node_type_idx.append([float(idx)])
    node_type_idx = torch.tensor(node_type_idx, dtype=torch.float)
    
    # 1. 扇出数（出度）
    out_degree = {n: 0 for n in node_names}
    for u, v in edges:
        out_degree[u] += 1
    
    # 2. 逻辑深度（最长路径，拓扑排序）
    adj = {n: [] for n in node_names}
    indeg = {n: 0 for n in node_names}
    for u, v in edges:
        adj[u].append(v)
        indeg[v] += 1
    q = deque([n for n in node_names if indeg[n] == 0])
    depth = {n: 0 for n in node_names}
    while q:
        u = q.popleft()
        for v in adj[u]:
            depth[v] = max(depth[v], depth[u] + 1)
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    
    # 3. 驱动强度
    def get_drive_strength(gate_type):
        match = re.search(r'x(\d+)', gate_type, re.IGNORECASE)
        if match:
            return float(match.group(1))
        match = re.search(r'AND(\d+)x', gate_type, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return 1.0

    # 4. 逻辑努力参数（寄生延迟 p, 逻辑努力 g, 输入电容 C_in）
    def get_logic_params(gate_type, num_inputs):
        gt = gate_type.upper()
        n = max(num_inputs, 1)
        # 寄生延迟 p
        if 'INV' in gt or 'NOT' in gt:    p = 1.0
        elif 'NAND' in gt:                p = n
        elif 'NOR' in gt:                 p = n
        elif 'AND' in gt:                 p = n + 1   # AND = NAND + INV
        elif 'OR' in gt:                  p = n + 1   # OR = NOR + INV
        elif 'XOR' in gt:                 p = 2 * n
        elif 'BUF' in gt:                 p = 1.0
        elif 'JOIN' in gt or 'BRIDGE' in gt or 'WIRE' in gt: p = 0.5
        else:                             p = 1.0
        # 逻辑努力 g
        if 'INV' in gt or 'NOT' in gt:    g = 1.0
        elif 'NAND' in gt:                g = (n + 2) / 3
        elif 'NOR' in gt:                 g = (2 * n + 1) / 3
        elif 'AND' in gt:                 g = (n + 2) / 3   # NAND part
        elif 'OR' in gt:                  g = (2 * n + 1) / 3  # NOR part
        elif 'XOR' in gt:                 g = 4.0
        elif 'BUF' in gt:                 g = 1.0
        elif 'JOIN' in gt or 'BRIDGE' in gt or 'WIRE' in gt: g = 0.5
        else:                             g = 1.0
        # 输入电容 C_in (从门类型名提取驱动等级)
        match = re.search(r'x(\d+)', gate_type, re.IGNORECASE)
        cin = float(match.group(1)) if match else 1.0
        return p, g, cin

    drive_strength = []
    parasitic = []
    logic_effort = []
    input_cap = []
    fanin_count = []
    for n in node_names:
        if n.startswith('X_'):
            gt = nodes[n]['type']
            num_in = len([e for e in edges if e[1] == n])
            ds = get_drive_strength(gt)
            p, g, cin = get_logic_params(gt, num_in)
            if mode == 'elec':
                # 电学特征(drive/p/g)从结构(ASAP7)算，替代名字正则（发现1第二处修复）
                si = struct_info[n]
                ds = si['drive']
                p, g = _logic_p_g(si['logic'], num_in)
        else:
            ds = 0.0
            num_in = 0
            p, g, cin = 0.0, 0.0, 0.0
        drive_strength.append(ds)
        parasitic.append(p)
        logic_effort.append(g)
        input_cap.append(cin)
        fanin_count.append(num_in)

    # 电努力 h = 输出负载 / 输入电容（仅门节点，引脚为0）
    out_load = [0.0] * len(node_names)
    for u, v in edges:
        ui = node_names.index(u)
        vi = node_names.index(v)
        out_load[ui] += input_cap[vi]
    electrical_effort = []
    for i, n in enumerate(node_names):
        if n.startswith('X_'):
            h = out_load[i] / max(input_cap[i], 0.01)
        else:
            h = 0.0
        electrical_effort.append(h)

    # 将特征转换为张量（使用 log1p 平滑）
    fanout_feat  = torch.tensor([[np.log1p(out_degree[n])] for n in node_names], dtype=torch.float)
    depth_feat   = torch.tensor([[np.log1p(depth[n])] for n in node_names], dtype=torch.float)
    drive_feat   = torch.tensor([[ds] for ds in drive_strength], dtype=torch.float)
    p_feat       = torch.tensor([[p] for p in parasitic], dtype=torch.float)
    g_feat       = torch.tensor([[g] for g in logic_effort], dtype=torch.float)
    h_feat       = torch.tensor([[np.log1p(h)] for h in electrical_effort], dtype=torch.float)

    # 结构特征（按 STRUCT_MODE）
    struct_feats = []
    if mode in ('base', 'rich', 'elec'):
        struct_feats.append(torch.tensor([[struct_info[n]['n_t']] for n in node_names], dtype=torch.float))
    if mode == 'rich':
        struct_feats.append(torch.tensor([[struct_info[n]['stack']] for n in node_names], dtype=torch.float))
        struct_feats.append(torch.tensor([[struct_info[n]['parallel']] for n in node_names], dtype=torch.float))

    # 合并静态特征：门类型索引 + 扇出 + 深度 + 驱动 + 寄生延迟 + 逻辑努力 + 电努力 + 结构特征
    node_static = torch.cat([node_type_idx, fanout_feat, depth_feat, drive_feat,
                              p_feat, g_feat, h_feat] + struct_feats, dim=1)

    return node_names, node_static, edge_index