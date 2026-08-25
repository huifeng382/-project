"""解析 Rust NetlistOpt benchmark .tl 电路的 I/O 形状"""
import os, re, glob, sys
from collections import Counter
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

base = r'\\wsl.localhost\Ubuntu\home\huifeng\test\TransiLog-share\NetlistOpt\testbench\tl_cells'
files = sorted(glob.glob(os.path.join(base, 'level*', '*.tl')))
shapes = Counter()
details = []
for f in files:
    with open(f, encoding='utf-8') as fh:
        txt = fh.read()
    m_in = re.search(r'INORDER\s*=\s*([^;]+);', txt)
    m_out = re.search(r'OUTORDER\s*=\s*([^;]+);', txt)
    nin = len(m_in.group(1).split()) if m_in else -1
    nout = len(m_out.group(1).split()) if m_out else -1
    level = os.path.basename(os.path.dirname(f))
    name = os.path.basename(f)
    shapes[(nin, nout)] += 1
    details.append((name, level, nin, nout, (m_in.group(1).strip() if m_in else '?')[:70], (m_out.group(1).strip() if m_out else '?')[:70]))

print(f'文件总数: {len(files)}')
print('\nI/O 形状分布 (n_in, n_out) -> 电路数:')
for k, v in sorted(shapes.items(), key=lambda x: (x[0][1], x[0][0])):
    print(f'  {k[0]:>2} 入 / {k[1]} 出 : {v} 个')
multi_out = sum(v for (n, m), v in shapes.items() if m > 1)
print(f'\n不同形状数: {len(shapes)} | 多输出(>=2出)电路: {multi_out} 个')
nin_vals = [d[2] for d in details]; nout_vals = [d[3] for d in details]
print(f'输入数范围: {min(nin_vals)}~{max(nin_vals)} | 输出数范围: {min(nout_vals)}~{max(nout_vals)}')
print('\n明细 (name, level, n_in, n_out):')
for name, level, nin, nout, i, o in details:
    print(f'  {level:8s} {name:22s} {nin:>2} 入 / {nout} 出  IN=[{i}]  OUT=[{o}]')
