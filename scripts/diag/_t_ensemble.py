"""#35 多 seed 集成分析: B 复合 (R4 共享+腿 n<8 + R6 专化+腿 ge8) 的散布 + 集成数 (零训练)。

对每侧自动收集: seed0 = 现有 resid, 新 seed = resid_{R4e,R6e}<n>_pin_17021.parquet (glob 探测, 按序号升序)。
同侧文件须行序逐位对齐 (同切分/同 sup 序, 仅 torch 初始化不同) → 校验 len/n_t/y 一致后按行平均。
输出: 每侧单模型 R^2 散布 / SxS 全配对复合 / 集成复合。

用法 (在 ~/idsavg17 下): ~/venv/bin/python3 scripts/diag/_t_ensemble.py
"""
import glob, re, pandas as pd, numpy as np

def r2(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)

def load(p):
    return pd.read_parquet(p, columns=['y', 'pred', 'n_t'])

def collect(pattern0, pattens):
    """seed0 文件 + 序号式新 seed 文件, 按 seed 升序返回 [(seed, df), ...]."""
    out = []
    df0 = load(pattern0); out.append((0, df0))
    for p in sorted(glob.glob(pattens), key=lambda x: int(re.search(r'e(\d+)_', x).group(1))):
        out.append((int(re.search(r'e(\d+)_', p).group(1)), load(p)))
    return out

def check_align(lst, side):
    """同侧各 seed 必须同 len/n_t/y (行序逐位对齐)."""
    ref = lst[0][1]
    for s, df in lst[1:]:
        ok = (len(df) == len(ref) and np.array_equal(df['n_t'].to_numpy(), ref['n_t'].to_numpy())
              and np.allclose(df['y'].to_numpy(), ref['y'].to_numpy()))
        if not ok:
            raise SystemExit(f'[align] {side} seed{s} 与 seed0 行序/标签不一致, 不可按行平均!')
    print(f'[align] {side}: {len(lst)} seeds x {len(ref)} 行, 逐位对齐 OK')

SH0  = 'resid_R4_pin_17016.parquet'      # 共享+腿 seed0
SHe  = 'resid_R4e*_pin_17021.parquet'
SP0  = 'resid_R6_nt8pin_17019.parquet'   # 专化+腿 seed0
SPe  = 'resid_R6e*_pin_17021.parquet'

sh = collect(SH0, SHe)   # [(seed, df_全test行), ...]
sp = collect(SP0, SPe)   # [(seed, df_ge8test行), ...]
check_align(sh, '共享+腿'); check_align(sp, '专化+腿')

# 每侧行切片
sh_lt = [(s, df[df['n_t'] < 8]) for s, df in sh]          # 供 n<8
sp_ge = [(s, df[df['n_t'] >= 8]) for s, df in sp]          # 供 ge8 (应全行)
y_sh = sh_lt[0][1]['y'].to_numpy(); y_sp = sp_ge[0][1]['y'].to_numpy()
n_sh, n_sp = len(y_sh), len(y_sp)
print(f'\n侧: 共享+腿 n<8 行={n_sh} | 专化+腿 ge8 行={n_sp}')

# (a) 每侧单模型散布 (各自全量口径)
print('\n===== (a) 每 seed 单模型 R^2 (散布) =====')
rsh = [r2(df['y'], df['pred']) for _, df in sh]     # 共享全池(全test行)
rge = [r2(df['y'], df['pred']) for _, df in sp]     # 专化 ge8 池
for (s, _), a, b in zip(sh, rsh, rge):
    print(f'  seed {s}: 共享+腿 full R^2={a:.4f} | 专化+腿 ge8 R^2={b:.4f}')
a = np.array(rsh); b = np.array(rge)
print(f'  共享+腿 full: mean={a.mean():.4f} sd={a.std():.4f}  min={a.min():.4f} max={a.max():.4f}')
print(f'  专化+腿 ge8:  mean={b.mean():.4f} sd={b.std():.4f}  min={b.min():.4f} max={b.max():.4f}')

# (b) SxS 全配对复合 (共享_i n<8 + 专化_j ge8) → 分布
ys = np.concatenate([y_sh, y_sp]); base_mean = ys.mean()
def comp_r2(p_sh, p_sp):
    p = np.concatenate([p_sh, p_sp])
    return 1 - np.sum((ys - p) ** 2) / np.sum((ys - base_mean) ** 2)
pairs = np.array([[comp_r2(sh_lt[i][1]['pred'].to_numpy(), sp_ge[j][1]['pred'].to_numpy())
                   for j in range(len(sp_ge))] for i in range(len(sh_lt))])
print(f'\n===== (b) SxS={len(sh)}x{len(sp)} 全配对复合 full R^2 =====')
print(f'  配对 mean={pairs.mean():.4f} sd={pairs.std():.4f}  min={pairs.min():.4f} max={pairs.max():.4f}  (seed0+seed0={pairs[0,0]:.4f})')
diag = np.array([pairs[i, i] for i in range(min(len(sh), len(sp)))])
print(f'  对角(同 seed 配对) mean={diag.mean():.4f} sd={diag.std():.4f}')

# (c) 集成复合 (每侧按行平均 pred)
p_sh_ens = np.mean([df['pred'].to_numpy() for _, df in sh_lt], axis=0)
p_sp_ens = np.mean([df['pred'].to_numpy() for _, df in sp_ge], axis=0)
r_ens = comp_r2(p_sh_ens, p_sp_ens)
print(f'\n===== (c) 集成复合 (每侧 {len(sh)}-seed 平均) =====')
print(f'  集成复合 full R^2 = {r_ens:.4f}   (基线: 共享+腿 seed0 单模型 full = {rsh[0]:.4f}, Δ={r_ens-rsh[0]:+.4f})')
