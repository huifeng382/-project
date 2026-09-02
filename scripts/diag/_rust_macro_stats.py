"""Rust dut .sp 宏层统计: 全部 .SUBCKT(非 DUT) 的 端口数/内部M_数; X_ cell 名分布; DUT trans 全体max; ★五形状 trans med 复核。
"""
import glob, re, os
from collections import Counter, defaultdict
import numpy as np

duts = sorted(glob.glob('/home/tianlang/NetlistOpt/temp_sim_test/tl_opt_batch/**/dut_expr_*.sp', recursive=True))
macro_ports = defaultdict(list)   # name -> [n_ports]
macro_M = defaultdict(list)       # name -> [n_M]
xcell = Counter()
dut_trans = []   # per-dut total M_
print('dut 文件数:', len(duts))
for f in duts:
    txt = open(f, encoding='utf-8', errors='replace').read()
    # 按 .SUBCKT 分段
    subs = re.findall(r'\.SUBCKT\s+(\S+)\s+([^\n]*)\n(.*?)(?=\.SUBCKT|\.ENDS\s|$)', txt, re.S)
    # 更稳: 找所有 .SUBCKT 名与 .ENDS
    names = re.findall(r'\.SUBCKT\s+(\S+)\s+([^\n]*)', txt)
    # 每段内 M_ 数: 用段落切
    segs = re.split(r'\.ENDS', txt)
    idx = 0
    n_dut_M = 0
    for s in segs:
        m = re.search(r'\.SUBCKT\s+(\S+)\s+([^\n]*)', s)
        if not m:
            continue
        nm = m.group(1)
        ports = m.group(2).split()
        nM = len(re.findall(r'^\s*M_\d+\s', s, re.M))
        if nm == 'DUT':
            n_dut_M = nM
        else:
            macro_ports[nm].append(len(ports))
            macro_M[nm].append(nM)
        # X_ 实例行在该段?X_ 在 DUT 段
        for xm in re.finditer(r'^\s*(X_\d+)\s+(.+?)\s+(\S+)\s*$', s, re.M):
            xcell[xm.group(3)] += 1
    dut_trans.append(n_dut_M)

print('\n== DUT trans 全体分布 ==')
dt = np.array(dut_trans)
print('max=%d p99=%d p95=%d p90=%d' % (dt.max(), np.percentile(dt,99), np.percentile(dt,95), np.percentile(dt,90)))

print('\n== 宏种类数:', len(macro_ports), '; 宏实例数总和:', sum(xcell.values()))
# 宏 trans(M_数) 分布
allM = [v for vs in macro_M.values() for v in vs]
aM = np.array(allM)
print('宏内部 M_ 数: med=%d p90=%d max=%d (n=%d)' % (np.median(aM), np.percentile(aM,90), aM.max(), len(aM)))
print('宏端口数: med=%d p90=%d max=%d' % (np.median([v for vs in macro_ports.values() for v in vs]),
      np.percentile([v for vs in macro_ports.values() for v in vs],90),
      max(v for vs in macro_ports.values() for v in vs)))
print('\n== 最常见 20 个 X_ cell ==')
for nm, c in xcell.most_common(20):
    pm = macro_ports.get(nm, ['?'])
    pM = macro_M.get(nm, ['?'])
    print('  %-46s 实例%5d 端口%s M_%s' % (nm[:46], c, sorted(set(pm))[:4], sorted(set(pM))[:4]))
print('\n== DUT 内 X_ 数分布 ==')
xn = []
for f in duts:
    txt = open(f, encoding='utf-8', errors='replace').read()
    segs = re.split(r'\.ENDS', txt)
    for s in segs:
        if '.SUBCKT  DUT' in s or '.SUBCKT DUT' in s:
            xn.append(len(re.findall(r'^\s*X_\d+\s', s, re.M)))
            break
ax = np.array(xn)
print('X_ 数: med=%.0f p90=%.0f max=%d' % (np.median(ax), np.percentile(ax,90), ax.max()))
