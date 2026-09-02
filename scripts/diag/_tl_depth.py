"""解析 Rust tl_cells 46 模板: IO / X节点数 / X级联深度(输入=0, 每X=1+max(依赖X))"""
import glob, re, os
from collections import defaultdict

def parse_tl(path):
    txt = open(path).read()
    mi = re.search(r'INORDER\s*=\s*([^;]+);', txt)
    mo = re.search(r'OUTORDER\s*=\s*([^;]+);', txt)
    ins = mi.group(1).split() if mi else []
    outs = mo.group(1).split() if mo else []
    # X_i = expr; 且 y = X_i (输出别名)
    assigns = {}
    alias = {}  # out -> assigned node
    for m in re.finditer(r'^\s*([A-Za-z_][\w\[\]]*)\s*=\s*(.+?)\s*;', txt, re.M):
        lhs, rhs = m.group(1), m.group(2)
        rhs2 = re.sub(r'^join\(|\)$', '', rhs.strip()) if rhs.strip().startswith('join(') else rhs.strip()
        # rhs 引用的 X 节点
        refs = set(re.findall(r'X\d+', rhs2))
        assigns[lhs] = refs
        # 输出可能直接是表达式行如 ovf = X34; 或 y = X1; 已被上面捕获
    depth = {}
    def dep(name):
        if name in depth:
            return depth[name]
        if name not in assigns:   # 输入
            return 0
        refs = assigns[name]
        d = 0
        for r in refs:
            d = max(d, dep(r))
        depth[name] = d + 1
        return depth[name]
    outd = []
    for o in outs:
        # 输出可能带 [k] 索引或直接名; tl 中 sum[0]..sum[3] 各行? OUTORDER = sum[0] sum[1]...
        # 输出节点可能是 sum[0] 等名, 需从 assigns 找(可能有 sum[0] = ... 行)
        if o in assigns:
            outd.append(dep(o))
        else:
            # 输出名可能引用了某 X
            found = None
            for lhs, refs in assigns.items():
                if o == lhs or (o.split('[')[0] == lhs.split('[')[0]):
                    found = dep(lhs)
            outd.append(found if found is not None else -1)
    # 所有节点深度(含内部中间)
    all_d = []
    for lhs in assigns:
        all_d.append(dep(lhs))
    return ins, outs, len(assigns), (max(outd) if outd else -1), (max(all_d) if all_d else -1)

rows = []
for f in sorted(glob.glob('/home/tianlang/NetlistOpt/testbench/tl_cells/**/*.tl', recursive=True)):
    name = os.path.basename(f).replace('.tl', '')
    ins, outs, nn, out_depth, all_depth = parse_tl(f)
    rows.append((name, len(ins), len(outs), nn, out_depth, all_depth))

print('%-14s %4s %4s %6s %8s %8s' % ('circ', 'n_in', 'n_out', 'X_nodes', 'out_depth', 'max_depth'))
for r in sorted(rows, key=lambda x: -x[5]):
    print('%-14s %4d %4d %6d %8d %8d' % r)
deep = [r for r in rows if r[5] > 6]
print('\n深度>6 模板:', len(deep), [r[0] for r in deep])
