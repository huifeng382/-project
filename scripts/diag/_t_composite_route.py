"""#34 复合路由指标 (零训练): 条件专家 full test R^2.

部署形态 = 按 n_t 路由: n_t<8 用共享模型 pred, n_t>=8 用专化模型 pred。
先对每个 resid 复算 full/ge8 池 R^2 核对落盘 log (R1 full~0.7861 ge8~0.5689; R4 0.7883/0.5778; R5 0.6053; R6 0.6102),
通过后出复合数。复合与共享 full 用同一批行/同一均值基线 => 差值 = 纯 ge8 换专化 pred 的 headline 收益。

用法 (在 ~/idsavg17 下): ~/venv/bin/python3 scripts/diag/_t_composite_route.py
"""
import pandas as pd, numpy as np

FILES = {
    'R1': 'resid_R1_base_17015.parquet',
    'R4': 'resid_R4_pin_17016.parquet',
    'R5': 'resid_R5_nt8_17019.parquet',
    'R6': 'resid_R6_nt8pin_17019.parquet',
}

def r2(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)

def load(p):
    return pd.read_parquet(p, columns=['y', 'pred', 'n_t'])

D = {k: load(v) for k, v in FILES.items()}
for k, df in D.items():
    ge = df[df['n_t'] >= 8]
    print(f"[{k}] rows={len(df)}  full R^2={r2(df['y'], df['pred']):.4f}  "
          f"ge8 rows={len(ge)}  ge8 R^2={r2(ge['y'], ge['pred']):.4f}")
print()

def composite(sh, sp, tag):
    """sh=共享全池文件(供 n_t<8), sp=专化文件(供 n_t>=8)。ge8 行以专化为准, 其余用共享。"""
    s = D[sh]; g = D[sp]
    lt = s[s['n_t'] < 8]
    ge = g[g['n_t'] >= 8]
    y = np.concatenate([lt['y'].to_numpy(), ge['y'].to_numpy()])
    p = np.concatenate([lt['pred'].to_numpy(), ge['pred'].to_numpy()])
    base = D[sh]  # 同一批行下共享模型自己的 full 就是复合的基线
    return tag, len(y), r2(y, p), r2(base['y'], base['pred']), len(lt), len(ge)

print('===== 复合路由 full test R^2 (同行同基线; 末列=对照共享模型自身 full) =====')
for (sh, sp, tag) in [
    ('R1', 'R5', 'A  R1共享(n<8) + R5专化(ge8)'),
    ('R1', 'R6', 'D  R1共享(n<8) + R6专化+腿(ge8)'),
    ('R4', 'R5', 'C  R4共享+腿(n<8) + R5专化(ge8)'),
    ('R4', 'R6', 'B  R4共享+腿(n<8) + R6专化+腿(ge8)'),
]:
    tag, n, rc, rb, nlt, nge = composite(sh, sp, tag)
    print(f"  {tag:34s} n={n:7d} (n<8 {nlt:6d} | ge8 {nge:6d})  复合 R^2={rc:.4f}   对照共享 R^2={rb:.4f}   Δ={rc-rb:+.4f}")
