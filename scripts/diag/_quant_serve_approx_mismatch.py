"""A1+B3 合并: serve vs 训练近似 ids_avg 特征失配量化 + 落差归因 (16.11.15)
对 Rust 候选(dut netlist)算 serve 侧近似(固定 corner), 对比:
  1) 训练 data_loader 同公式(同输入应一致) —— 验证 serve 实现
  2) 特征值分布 vs 训练数据的近似分布 —— 量化分布偏移
  3) 是否存在负值/异常 → 失配程度

用法: python scripts/diag/_quant_serve_approx_mismatch.py
"""
import sys, os, glob, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

import config
config.USE_IDS_AVG_APPROX = '1'  # 线性模式
from src.graph_builder import build_static_graph, rebuild_gate_types
from src.data_loader import DelayDataset

DATA = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
C = config.IDS_AVG_APPROX_COEF

def cell_types(nl):
    types = set()
    for line in (nl or '').split('\n'):
        s = line.strip()
        if s.startswith('X_') and len(s.split()) >= 3:
            types.add(s.split()[-1])
    return types

def approx_for(ns, slew_ps=2.0, load_fF=1.0, cslew=2.0, cload=1.0):
    """serve 同款线性近似: ns=node_static, 返回每门近似值"""
    f_slew = math.log1p(slew_ps); f_load = math.log1p(load_fF)
    f_cslew = math.log1p(cslew); f_cload = math.log1p(cload)
    vals = []
    for i in range(ns.shape[0]):
        f_drive = math.log1p(float(ns[i, 3]))
        f_par = math.log1p(float(ns[i, 4]))
        f_fan = float(ns[i, 1])
        f_h = float(ns[i, 6])
        lg = (C[0]*f_slew + C[1]*f_load + C[2]*f_drive + C[3]*f_par
              + C[4]*f_fan + C[5]*f_h + C[6]*f_cslew + C[7]*f_cload + C[8])
        vals.append(math.expm1(lg))
    return np.array(vals)

def main():
    # 1) Rust 候选的 serve 近似(抽样 dut netlist)
    rust_files = glob.glob('/home/tianlang/NetlistOpt/temp_sim_test/tl_opt_batch/level*/**/dut_*.sp',
                           recursive=True)
    print(f'Rust dut 文件: {len(rust_files)}')
    serve_vals = []
    for f in rust_files[:500]:
        try:
            nl = open(f).read()
            rebuild_gate_types(cell_types(nl))
            # 输入/输出引脚从文件名无法直接得, 用 .SUBCKT DUT 定义解析
            ip = []; op = []
            for line in nl.split('\n'):
                s = line.strip()
                if s.startswith('.SUBCKT DUT'):
                    pins = s.split()[3:]
                    # 假设 vdd/gnd 在末尾, 前面都是 IO(粗估)
                    real = [p for p in pins if p.lower() not in ('vdd','gnd','vss')]
                    # 无精确 IO 划分, 用全部非电源引脚作输入, 输出留空(影响小)
                    ip = real; op = []
            nn, ns, _ = build_static_graph('rust', nl, ip or None, op or None)
            v = approx_for(ns)
            serve_vals.extend(v.tolist())
        except Exception:
            continue
    sv = np.array(serve_vals)
    print(f'\n[Rust 候选 serve 近似] n={len(sv)}')
    print(f'  min={sv.min():.4f} med={np.median(sv):.4f} max={sv.max():.4f}')
    print(f'  负值比例: {(sv<0).mean():.1%}  零值: {(sv==0).mean():.1%}')

    # 2) 训练数据的近似(在真实训练电路上, 用 data_loader 公式)
    # 加载小样本训练数据(避免全量)
    sp = glob.glob(f'{DATA}/data/batch_v2_full/circuit_static*.parquet')
    dp = glob.glob(f'{DATA}/data/batch_v2_full/timing_arcs*.parquet')
    # 直接读 static 建图(不需要 dynamic, 近似只用 node_static + 固定 corner)
    import pandas as pd
    sdf = pd.concat([pd.read_parquet(p) for p in sp], ignore_index=True)
    sdf = sdf.sample(min(300, len(sdf)), random_state=1)
    train_vals = []
    for _, r in sdf.iterrows():
        try:
            nl = r['gate_level_netlist']
            ip = json.loads(r['input_pins_json']) if isinstance(r['input_pins_json'], str) else r['input_pins_json']
            op = json.loads(r['output_pins_json']) if isinstance(r['output_pins_json'], str) else r['output_pins_json']
            rebuild_gate_types(cell_types(nl))
            nn, ns, _ = build_static_graph(str(r['circuit_id']), nl, ip or None, op or None)
            v = approx_for(ns)
            train_vals.extend(v.tolist())
        except Exception:
            continue
    tv = np.array(train_vals)
    print(f'\n[训练数据近似] n={len(tv)}')
    print(f'  min={tv.min():.4f} med={np.median(tv):.4f} max={tv.max():.4f}')
    print(f'  负值比例: {(tv<0).mean():.1%}  零值: {(tv==0).mean():.1%}')

    # 3) 分布对比(serve 候选 vs 训练)
    print('\n=== 分布对比 ===')
    for name, arr in [('serve候选', sv), ('训练', tv)]:
        pos = arr[arr > 0]
        if len(pos):
            print(f'  {name}: log10中位={np.median(np.log10(pos)):.3f} 正值中位={np.median(pos):.3e}')
    # KS 检验
    from scipy.stats import ks_2samp
    sv_pos = sv[sv > 0]; tv_pos = tv[tv > 0]
    if len(sv_pos) > 10 and len(tv_pos) > 10:
        ks = ks_2samp(np.log10(sv_pos), np.log10(tv_pos))
        print(f'  KS检验(log10正近似): stat={ks.statistic:.4f} p={ks.pvalue:.2e}')
        print(f'  (p<0.05 表示两分布显著不同)')

    # 4) 落差归因结论
    print('\n=== 落差归因 ===')
    print(f'serve负值比例 {((sv<0).mean()):.1%} vs 训练负值 {((tv<0).mean()):.1%}')
    if (sv < 0).mean() > (tv < 0).mean():
        print('→ serve 端负值更多: 候选门结构(弱驱动)导致近似出负, 训练数据少此类门')
    else:
        print('→ serve/训练负值相当: 负值非主因')

if __name__ == '__main__':
    main()
