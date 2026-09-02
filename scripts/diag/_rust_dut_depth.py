"""Rust 候选 dut .sp 宏级深度(X_ 最长链)聚合——与训练 gate_level_netlist 同口径。
对 temp_sim_test/tl_opt_batch/**/dut_expr_*.sp 解析 .SUBCKT DUT 内 X_ 行。
"""
import glob, re, os
from collections import defaultdict
import numpy as np

def depth_of_file(path, ins, outs):
    txt = open(path, encoding='utf-8', errors='replace').read()
    # 找 .SUBCKT DUT 段
    m = re.search(r'\.SUBCKT\s+DUT\b([^\n]*)\n(.*?)\.ENDS', txt, re.S)
    if not m:
        return None
    ports = m.group(1).split()
    body = m.group(2)
    in_set = set(ins)
    out_set = set(outs)
    # X_ 行: X_N <in nets...> <out net> <cell>
    xrows = []
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith('X_'):
            continue
        toks = line.split()
        if len(toks) < 4:
            continue
        nets, cell = toks[1:-1], toks[-1]
        xrows.append(nets)  # 最后一个是输出
    if not xrows:
        return None
    # net -> depth (X_ 输出); DUT 输入 net depth 0
    depth = {}
    for nets in xrows:
        outn = nets[-1]
        din = 0
        for n in nets[:-1]:
            d = depth.get(n, 0)
            if d > din:
                din = d
        depth[outn] = din + 1
    # DUT 输出深度 = max over out nets(在 depth 中或为输入)
    od = 0
    for o in out_set:
        d = depth.get(o, 0)
        if d > od:
            od = d
    return len(xrows), od, len(depth)

def main():
    # circ -> (ins, outs) from tl files
    circ_io = {}
    for f in glob.glob('/home/tianlang/NetlistOpt/testbench/tl_cells/**/*.tl', recursive=True):
        txt = open(f).read()
        mi = re.search(r'INORDER\s*=\s*([^;]+);', txt)
        mo = re.search(r'OUTORDER\s*=\s*([^;]+);', txt)
        if mi and mo:
            nm = os.path.basename(f).replace('.tl', '')
            circ_io[nm] = (mi.group(1).split(), mo.group(1).split())
    agg = defaultdict(lambda: {'xn': [], 'dep': []})
    files = sorted(glob.glob('/home/tianlang/NetlistOpt/temp_sim_test/tl_opt_batch/**/dut_expr_*.sp', recursive=True))
    print('dut files:', len(files))
    for f in files:
        parts = f.split('/tl_opt_batch/')[-1].split('/')
        circ = parts[1] if len(parts) >= 3 else '?'
        io = circ_io.get(circ)
        if not io:
            continue
        r = depth_of_file(f, io[0], io[1])
        if r:
            agg[circ]['xn'].append(r[0])
            agg[circ]['dep'].append(r[1])
    print('%-14s %5s %8s %8s %8s %8s %8s %8s' % ('circ', 'n', 'X_med', 'dep_med', 'dep_p90', 'dep_max', 'dep>6%', 'dep>9%'))
    for circ in sorted(agg, key=lambda c: -np.percentile(agg[c]['dep'], 90)):
        d = agg[circ]
        xn = np.array(d['xn']); dep = np.array(d['dep'])
        print('%-14s %5d %8.0f %8.0f %8.0f %8d %8.0f %8.0f' % (
            circ, len(dep), np.median(xn), np.median(dep),
            np.percentile(dep, 90), dep.max(),
            100*np.mean(dep > 6), 100*np.mean(dep > 9)))

if __name__ == '__main__':
    main()
