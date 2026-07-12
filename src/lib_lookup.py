"""
标准单元库 LIB 表解析和双线性插值查表。
使用 NLDM 7×7 表格，输入 slew → 负载 → 门延迟。
"""
import re, json, os
import numpy as np


def parse_lib(lib_path):
    """
    解析 Liberty .lib 文件，提取每个标准单元的 NLDM 查找表。
    返回: dict, key=cell_name, value=dict with 'rise'/'fall' tables
    """
    with open(lib_path, 'r') as f:
        content = f.read()

    cells = {}
    # 用 cell ( 分割，而非嵌套大括号的正则
    blocks = content.split('cell (')
    for block in blocks[1:]:  # 第一个是文件头，跳过
        # 提取 cell name
        name_end = block.index(')')
        cell_name = block[:name_end].strip()
        # 提取 cell body（大括号匹配）
        body_start = block.index('{', name_end) + 1
        depth = 1
        i = body_start
        while i < len(block) and depth > 0:
            if block[i] == '{': depth += 1
            elif block[i] == '}': depth -= 1
            i += 1
        body = block[body_start:i-1]
        cells[cell_name] = _parse_cell_body(body)

    return cells


def _parse_cell_body(body):
    """解析一个 cell：四张 7×7 表 + 输入引脚电容。
    返回 dict:
      'rise'/'fall'             -> {'idx1','idx2','table'}  (延迟, ps)
      'rise_trans'/'fall_trans' -> {'idx1','idx2','table'}  (输出 slew, ps)
      'in_cap'                  -> {pin_name: fF}           (仅 input 方向引脚)
    每种表取 cell 内第一次出现的 arc 作代表。"""
    result = {}
    for key, tbl in [('rise', 'cell_rise'), ('fall', 'cell_fall'),
                     ('rise_trans', 'rise_transition'), ('fall_trans', 'fall_transition')]:
        m = re.search(
            tbl + r'\s*\([^)]*\)\s*\{[^}]*?'
            r'index_1\s*\(\s*"([^"]*)"\s*\)[^}]*?'
            r'index_2\s*\(\s*"([^"]*)"\s*\)[^}]*?'
            r'values\s*\(\s*(.*?)\)\s*;', body, re.DOTALL)
        if not m:
            continue
        idx1 = [float(x.strip()) for x in m.group(1).split(',')]
        idx2 = [float(x.strip()) for x in m.group(2).split(',')]
        vals_str = m.group(3).replace('\\', '').replace('\n', ' ').replace('"', '')
        values = [float(x.strip()) for x in vals_str.split(',') if x.strip()]
        if len(values) == len(idx1) * len(idx2):
            result[key] = {'idx1': np.array(idx1), 'idx2': np.array(idx2),
                           'table': np.array(values).reshape(len(idx1), len(idx2))}

    # 输入引脚电容：pin(NAME){ direction:input ... capacitance : X }
    in_cap = {}
    pin_starts = [pm for pm in re.finditer(r'pin\s*\(\s*([A-Za-z0-9_]+)\s*\)\s*\{', body)]
    for i, pm in enumerate(pin_starts):
        name = pm.group(1)
        end = pin_starts[i + 1].start() if i + 1 < len(pin_starts) else len(body)
        seg = body[pm.end():end]
        if re.search(r'direction\s*:\s*input', seg):
            cm = re.search(r'\bcapacitance\s*:\s*([\d.]+)', seg)
            if cm:
                in_cap[name] = float(cm.group(1))
    if in_cap:
        result['in_cap'] = in_cap
    return result


def load_mapping(mapping_path):
    """加载 SC_→ASAP7 映射表"""
    with open(mapping_path, 'r') as f:
        return json.load(f)


def build_lib_tensors(lib_cells, mapping):
    """
    将 LIB 表转为 PyTorch tensor，支持可微双线性插值。
    返回: (gate_names, idx1_t, idx2_t, tables_t)
           gate_names: list of 27 gate type names
           idx1_t: (27, 7) slew index
           idx2_t: (27, 7) load index
           tables_t: (27, 2, 7, 7) delay tables [gate, rise/fall, slew_idx, load_idx]
    """
    import torch
    gate_list = sorted(set(v for v in mapping.values() if v is not None and v in lib_cells))
    N = len(gate_list)
    idx1_t = torch.zeros(N, 7)
    idx2_t = torch.zeros(N, 7)
    tables_t = torch.zeros(N, 2, 7, 7)  # 2 = rise, fall

    for gi, gname in enumerate(gate_list):
        cell = lib_cells[gname]
        for si, sense in enumerate(['rise', 'fall']):
            if sense in cell:
                t = cell[sense]
                idx1_t[gi] = torch.tensor(t['idx1'][:7], dtype=torch.float)
                idx2_t[gi] = torch.tensor(t['idx2'][:7], dtype=torch.float)
                tables_t[gi, si] = torch.tensor(t['table'][:7, :7], dtype=torch.float)

    return gate_list, idx1_t, idx2_t, tables_t


def lib_batch_lookup(sc_gate_names, slew_ps, load_ff, gate_list, idx1_t, idx2_t,
                      tables_t, mapping, default=10.0):
    """
    PyTorch 可微批量 LIB 查表。
    sc_gate_names: list of str, SC_ 门类型名
    slew_ps: (N,) tensor, 输入 slew (ps)
    load_ff: (N,) tensor, 输出负载 (fF)
    返回: (N,) tensor, 门延迟 (ps)
    """
    import torch
    N = len(sc_gate_names)
    delays = torch.full((N,), default, dtype=torch.float)

    for i in range(N):
        sc_name = sc_gate_names[i]
        if sc_name not in mapping or mapping[sc_name] is None:
            continue
        asap7 = mapping[sc_name]
        if asap7 not in gate_list:
            continue
        gi = gate_list.index(asap7)
        # 用 rise 表（简化：后续可扩展方向判断）
        table = tables_t[gi, 0]  # (7, 7)
        idx1 = idx1_t[gi]  # (7,)
        idx2 = idx2_t[gi]  # (7,)

        x = torch.clamp(slew_ps[i], idx1[0], idx1[-1])
        y = torch.clamp(load_ff[i], idx2[0], idx2[-1])

        # searchsorted 在 PyTorch 中不可微，用 clamp+linear 近似
        # 用 torch.where 计算插值权重
        xi = torch.searchsorted(idx1, x).clamp(1, 6)
        x0, x1 = idx1[xi-1], idx1[xi]
        wx = (x - x0) / (x1 - x0 + 1e-10)

        yi = torch.searchsorted(idx2, y).clamp(1, 6)
        y0, y1 = idx2[yi-1], idx2[yi]
        wy = (y - y0) / (y1 - y0 + 1e-10)

        # 双线性插值（可微）
        q00 = table[xi-1, yi-1]
        q10 = table[xi, yi-1]
        q01 = table[xi-1, yi]
        q11 = table[xi, yi]
        d = (q00 * (1-wx) * (1-wy) + q10 * wx * (1-wy) +
             q01 * (1-wx) * wy + q11 * wx * wy)
        delays[i] = d

    return delays


def lookup_delay(cell_name, input_slew_ps, output_load_ff, lib_cells, mapping,
                 sense='rise', default_delay=10.0):
    """
    查表获取门延迟。
    cell_name: SC_ 门类型名
    input_slew_ps: 输入 slew (ps)
    output_load_ff: 输出负载 (fF)
    lib_cells: parse_lib() 返回的结果
    mapping: load_mapping() 返回的结果
    sense: 'rise' 或 'fall'
    """
    if cell_name not in mapping:
        return default_delay
    asap7_name = mapping[cell_name]
    if asap7_name is None or asap7_name not in lib_cells:
        return default_delay

    cell_data = lib_cells[asap7_name]
    if sense not in cell_data:
        return default_delay

    t = cell_data[sense]
    idx1, idx2, table = t['idx1'], t['idx2'], t['table']

    # 双线性插值
    x = np.clip(input_slew_ps, idx1[0], idx1[-1])
    y = np.clip(output_load_ff, idx2[0], idx2[-1])

    i = np.searchsorted(idx1, x) - 1
    j = np.searchsorted(idx2, y) - 1
    i = max(0, min(i, len(idx1) - 2))
    j = max(0, min(j, len(idx2) - 2))

    x1, x2 = idx1[i], idx1[i+1]
    y1, y2 = idx2[j], idx2[j+1]
    q11, q12 = table[i, j], table[i, j+1]
    q21, q22 = table[i+1, j], table[i+1, j+1]

    return (q11 * (x2-x) * (y2-y) + q21 * (x-x1) * (y2-y) +
            q12 * (x2-x) * (y-y1) + q22 * (x-x1) * (y-y1)) / ((x2-x1) * (y2-y1))


# ==================== SC 宏展开 → LIB 延迟链（Scheme A） ====================

def load_expansion(path):
    """加载 sc_expansion.json，丢弃 null（未展开的 SC_BRIDGE / 超长 SC_JOIN）。
    返回 dict[sc_name -> {'subcircuit': [{inst,cell,inputs,output}, ...]}]。"""
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        if isinstance(v, dict) and v.get('subcircuit'):
            out[k] = v
    return out


def _incap_list(lib_cell):
    """按引脚名(A,B,C..)排序的输入电容列表(fF)。"""
    ic = lib_cell.get('in_cap', {}) if lib_cell else {}
    return [ic[k] for k in sorted(ic.keys())]


def precompute_macro_chains(expansion, lib_cells, default_cap=1.0):
    """为每个可展开的 SC 宏预计算延迟链计划（一次性、非可微、可缓存）。
    连接靠线名匹配；内部扇出=1，非 sink 单元负载=其唯一下游输入引脚电容(fF)。
    每单元的延迟/输出slew表取 rise/fall 均值(避开极性追踪)。
    返回 dict[sc_name -> plan]；plan 各字段为 numpy/py 原生，便于缓存。"""
    plans = {}
    for sc_name, entry in expansion.items():
        sub = entry['subcircuit']
        n = len(sub)
        cell_names = [c['cell'] for c in sub]
        inputs = [list(c.get('inputs', [])) for c in sub]
        outputs = [c['output'] for c in sub]
        # 所有单元 cell 必须在 lib 中
        if any(cn not in lib_cells for cn in cell_names):
            continue
        driver = {outputs[i]: i for i in range(n)}          # net -> 产出它的单元
        all_out = set(outputs)
        all_in = set(w for ins in inputs for w in ins)
        ext_inputs = all_in - all_out                        # 外部输入(字母)
        # sink：输出为 Y 的单元
        sink_idx = driver.get('Y', None)
        if sink_idx is None:
            continue
        # 拓扑序(Kahn)：单元就绪 = 其所有输入net 是外部输入或已产出
        indeg = [sum(1 for w in inputs[i] if w in driver) for i in range(n)]
        from collections import deque
        q = deque([i for i in range(n) if indeg[i] == 0])
        order = []
        # 下游依赖：net -> 消费它的单元们(此处扇出=1)
        consumers = {}
        for j in range(n):
            for w in inputs[j]:
                consumers.setdefault(w, []).append(j)
        indeg_work = indeg[:]
        while q:
            i = q.popleft(); order.append(i)
            for j in consumers.get(outputs[i], []):
                indeg_work[j] -= 1
                if indeg_work[j] == 0:
                    q.append(j)
        if len(order) != n:      # 有环/异常，跳过
            continue
        # 静态负载：非 sink 单元 = 所有下游单元对应输入引脚电容之和（处理扇出>1）
        static_load = [0.0] * n
        for i in range(n):
            if i == sink_idx:
                static_load[i] = -1.0          # 动态(外部负载)
                continue
            tot = 0.0
            for j in consumers.get(outputs[i], []):
                caps = _incap_list(lib_cells[cell_names[j]])
                for pos, w in enumerate(inputs[j]):
                    if w == outputs[i]:
                        tot += caps[pos] if pos < len(caps) else default_cap
            static_load[i] = tot if tot > 0 else default_cap
        # 每单元的均值延迟/输出slew表 + 各自的 idx 轴（不同门类型的负载/slew 轴不同！）
        d_tabs, s_tabs, idx1s, idx2s, is_const = [], [], [], [], []
        ok = True
        for i in range(n):
            lc = lib_cells[cell_names[i]]
            is_const.append(len(inputs[i]) == 0)             # TIE 常量源
            rise = lc.get('rise'); fall = lc.get('fall', rise)
            rt = lc.get('rise_trans'); ft = lc.get('fall_trans', rt)
            if rise is None or rt is None:
                if is_const[-1]:                              # TIE 无 timing，占位
                    d_tabs.append(np.zeros((7, 7))); s_tabs.append(np.zeros((7, 7)))
                    idx1s.append(None); idx2s.append(None)
                    continue
                ok = False; break
            d_tabs.append(0.5 * (rise['table'] + fall['table']))
            s_tabs.append(0.5 * (rt['table'] + ft['table']))
            idx1s.append(rise['idx1']); idx2s.append(rise['idx2'])
        if not ok:
            continue
        plans[sc_name] = {
            'order': order, 'inputs': inputs, 'outputs': outputs,
            'sink_idx': sink_idx, 'ext_inputs': ext_inputs, 'is_const': is_const,
            'static_load': static_load, 'idx1s': idx1s, 'idx2s': idx2s,
            'd_tabs': d_tabs, 's_tabs': s_tabs,
        }
    return plans


def _to_torch_plan(plan, device):
    """把 numpy plan 转成 torch 张量(缓存于 plan['_t']，按 device 校验)。"""
    import torch
    if plan.get('_t_dev') == str(device):
        return plan['_t']
    def _mk(a):
        return None if a is None else torch.tensor(a, dtype=torch.float, device=device)
    t = {
        'idx1s': [_mk(a) for a in plan['idx1s']],
        'idx2s': [_mk(a) for a in plan['idx2s']],
        'd_tabs': [torch.tensor(d, dtype=torch.float, device=device) for d in plan['d_tabs']],
        's_tabs': [torch.tensor(s, dtype=torch.float, device=device) for s in plan['s_tabs']],
        'static_load': [torch.tensor(float(l), device=device) for l in plan['static_load']],
    }
    plan['_t'] = t
    plan['_t_dev'] = str(device)
    return t


def _bilinear_t(table, idx1, idx2, x, y):
    """可微双线性插值(torch)。table:(7,7), idx1/idx2:(7,), x/y:标量。"""
    import torch
    x = torch.clamp(x, idx1[0], idx1[-1])
    y = torch.clamp(y, idx2[0], idx2[-1])
    xi = torch.searchsorted(idx1, x.reshape(1)).clamp(1, len(idx1) - 1)[0]
    yi = torch.searchsorted(idx2, y.reshape(1)).clamp(1, len(idx2) - 1)[0]
    x0, x1 = idx1[xi - 1], idx1[xi]
    y0, y1 = idx2[yi - 1], idx2[yi]
    wx = (x - x0) / (x1 - x0 + 1e-10)
    wy = (y - y0) / (y1 - y0 + 1e-10)
    q00 = table[xi - 1, yi - 1]; q10 = table[xi, yi - 1]
    q01 = table[xi - 1, yi]; q11 = table[xi, yi]
    return (q00 * (1 - wx) * (1 - wy) + q10 * wx * (1 - wy) +
            q01 * (1 - wx) * wy + q11 * wx * wy)


def macro_delay(plan, in_slew, ext_load):
    """可微：给定宏计划 + 模型预测的(宏输入slew ps, 外部负载 fF)，
    沿内部链做到达时间 DP，返回 (宏延迟 ps, 宏输出slew ps)。"""
    import torch
    t = _to_torch_plan(plan, in_slew.device)
    arrival, slew = {}, {}
    zero = in_slew * 0.0
    for w in plan['ext_inputs']:
        arrival[w] = zero
        slew[w] = in_slew
    for i in plan['order']:
        out_net = plan['outputs'][i]
        if plan['is_const'][i]:                # TIE 常量源：arrival 0，slew 取最小(5ps)
            arrival[out_net] = zero
            slew[out_net] = 5.0 + zero
            continue
        idx1 = t['idx1s'][i]; idx2 = t['idx2s'][i]
        load = ext_load if i == plan['sink_idx'] else t['static_load'][i]
        cands, sins = [], []
        for w in plan['inputs'][i]:
            s_in = slew.get(w, in_slew)
            a_in = arrival.get(w, zero)
            d = _bilinear_t(t['d_tabs'][i], idx1, idx2, s_in, load)
            cands.append(a_in + d)
            sins.append(s_in)
        cand = torch.stack(cands)
        k = int(torch.argmax(cand).item())
        arrival[out_net] = cand[k]
        slew[out_net] = _bilinear_t(t['s_tabs'][i], idx1, idx2, sins[k], load)
    return arrival['Y'], slew['Y']


def macro_batch_delay(sc_names, in_slew, ext_load, plans, default=0.0):
    """对一批门(宏)算延迟+输出slew+valid掩码。未知/未展开宏 -> (default, 0, False)。
    返回 (delays_ps (N,), out_slews_ps (N,), valid (N,) bool)，可微。"""
    import torch
    dev = in_slew.device
    N = len(sc_names)
    if N == 0:
        z = in_slew.new_zeros(0)
        return z, z, torch.zeros(0, dtype=torch.bool, device=dev)
    delays = [None] * N
    slews = [None] * N
    valid = [False] * N
    for i in range(N):
        plan = plans.get(sc_names[i])
        if plan is None:                        # 未展开(布线类SC_JOIN/BRIDGE等) -> 0 延迟, 不监督
            delays[i] = torch.tensor(float(default), device=dev)
            slews[i] = torch.tensor(0.0, device=dev)
        else:
            d, s = macro_delay(plan, in_slew[i], ext_load[i])
            delays[i] = d; slews[i] = s; valid[i] = True
    return (torch.stack(delays), torch.stack(slews),
            torch.tensor(valid, dtype=torch.bool, device=dev))
