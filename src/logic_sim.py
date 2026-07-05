"""
轻量级逻辑仿真：不依赖 SPICE，基于网表和 vector 推算门的翻转状态。
用于替代 gate_states_json，标记信号路径上的门。
"""

def evaluate_gate(gate_type, input_vals):
    """
    根据门类型和输入值计算输出。
    gate_type: str, 门类型名（如 SC_AND, INVx1_ASAP7_75t_R）
    input_vals: list of int (0/1)，门的输入值列表
    返回: int (0/1)
    """
    gt = gate_type.upper()
    valid = [v for v in input_vals if v is not None]
    if not valid:
        return 0

    if 'NAND' in gt:
        return 0 if all(v == 1 for v in valid) else 1
    elif 'NOR' in gt:
        return 1 if all(v == 0 for v in valid) else 0
    elif 'AND' in gt:
        return 1 if all(v == 1 for v in valid) else 0
    elif 'OR' in gt:
        return 1 if any(v == 1 for v in valid) else 0
    elif 'XOR' in gt:
        return sum(valid) % 2
    elif 'INV' in gt or 'NOT' in gt:
        return 1 - valid[0]
    elif 'BUF' in gt:
        return valid[0]
    elif 'BRIDGE' in gt or 'JOIN' in gt or 'WIRE' in gt:
        return valid[0]
    elif 'TIEHI' in gt:
        return 1
    elif 'TIELO' in gt:
        return 0
    elif 'INPUT_PIN' in gt or 'OUTPUT_PIN' in gt:
        return valid[0]
    else:
        return valid[0]


def simulate_circuit(node_names, node_types, edge_index, input_values):
    """
    对电路做一次逻辑传播。
    node_names: list of str, 节点名
    node_types: dict name->gate_type_str
    edge_index: torch.Tensor (2, E), 有向边 u->v
    input_values: dict name->int (0/1), 输入引脚的值
    返回: dict name->int (0/1), 所有节点的输出值
    """
    # 构建邻接表（每个门有哪些输入节点）
    in_edges = {n: [] for n in node_names}
    out_edges = {n: [] for n in node_names}
    for i in range(edge_index.shape[1]):
        u = node_names[edge_index[0, i].item()]
        v = node_names[edge_index[1, i].item()]
        out_edges[u].append(v)
        in_edges[v].append(u)

    # 拓扑排序（BFS 从输入引脚出发）
    values = {}
    indeg = {n: len(in_edges[n]) for n in node_names}
    queue = [n for n in node_names if indeg[n] == 0]

    # 输入引脚：直接用 input_values
    for n in node_names:
        if n in input_values:
            values[n] = input_values[n]
            indeg[n] = 0  # 确保在队列中
            if n not in queue:
                queue.append(n)

    while queue:
        u = queue.pop(0)
        if u not in values:
            # 这个节点没有输入值（不应发生），设为 0
            values[u] = 0
        for v in out_edges[u]:
            indeg[v] -= 1
            if indeg[v] <= 0:
                # 收集所有输入值
                inputs = [values.get(w, 0) for w in in_edges[v]]
                gate_type = node_types.get(v, 'UNKNOWN')
                values[v] = evaluate_gate(gate_type, inputs)
                if v not in queue:
                    queue.append(v)

    # 处理环（如果有）：对剩余节点设默认值
    for n in node_names:
        if n not in values:
            values[n] = 0

    return values


def compute_gate_states(node_names, node_types, edge_index, vector_str, pins, switching_pin):
    """
    通过正反两向 BFS 取交集确定信号路径上的门。
    - 正向：从 switching_pin 可达（信号源）
    - 反向：可达 output（信号宿）
    - 交集 = 实际信号路径（排除死分支）

    返回: dict, 门名->state (1=在路径上, 0=不在)
    """
    # 构建正向和反向邻接表
    forward_adj = {n: [] for n in node_names}
    reverse_adj = {n: [] for n in node_names}
    for i in range(edge_index.shape[1]):
        u = node_names[edge_index[0, i].item()]
        v = node_names[edge_index[1, i].item()]
        forward_adj[u].append(v)
        reverse_adj[v].append(u)

    from collections import deque

    # 正向 BFS：从 switching_pin 可达的节点
    fwd_visited = {switching_pin}
    q = deque([switching_pin])
    while q:
        u = q.popleft()
        for v in forward_adj.get(u, []):
            if v not in fwd_visited:
                fwd_visited.add(v)
                q.append(v)

    # 反向 BFS：可达 output 的节点
    rev_visited = set()
    if 'out' in node_names:
        rev_visited.add('out')
        q = deque(['out'])
        while q:
            u = q.popleft()
            for v in reverse_adj.get(u, []):
                if v not in rev_visited:
                    rev_visited.add(v)
                    q.append(v)

    # 交集 = 信号路径
    gate_states = {n: 0 for n in node_names}
    gate_states['out'] = 1
    for n in node_names:
        if n in pins:
            gate_states[n] = 0  # 输入引脚不参与 sum
        elif n != 'out' and n in fwd_visited and n in rev_visited:
            gate_states[n] = 1  # 在信号路径上的门

    return gate_states
