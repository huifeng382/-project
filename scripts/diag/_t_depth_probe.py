"""n<=2 桶残差按 逻辑深度(depth=距主输入级数, 最长路径) 分层: 检验"浅门 0.80 是被到达 slew/load
幅度代理粗(埋深门拿不到沿链真实幅度)压住"的假设。零训练, 纯读 R4 resid (每行已 dump depth)。

判定:
  - 残差集中高 depth (近输入门 depth 0-1 的 R2 已经 ~0.9) => 传播假设成立 -> logical-effort 级间
    幅度特征有据可攻, 0.85 是真目标。
  - depth 0-1 也压不动 (~0.8) => 0.80 不是代理粗, 是更底层的量(如切换脉冲波形) => 别在 n<=2 豪赌。

用法 (在 ~/idsavg17 下): ~/venv/bin/python3 scripts/diag/_t_depth_probe.py
"""
import collections
import pandas as pd, numpy as np

df = pd.read_parquet('resid_R4_pin_17016.parquet', columns=['y', 'pred', 'n_t', 'depth'])
df = df[df['n_t'] <= 2]
y = df['y'].to_numpy(); p = df['pred'].to_numpy(); d = df['depth'].to_numpy().astype(int)
def r2(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return 1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum()
print(f'n<=2: rows={len(y)}  R2={r2(y, p):.4f}  mean_y={y.mean():.3f}')
sse_n2 = ((y - p) ** 2).sum()

lv = collections.defaultdict(list)
for yy, pp, dd in zip(y, p, d):
    lv[dd].append((yy, pp))
print('\n--- 按 depth 逐级 (每级行数>=30 才列) ---')
print('depth  rows    mean_y   R2      SSE占n2%')
for dd in sorted(lv):
    a = np.array([t[0] for t in lv[dd]]); b = np.array([t[1] for t in lv[dd]])
    if len(a) < 30:
        continue
    print(f'{dd:4d}  {len(a):6d}  {a.mean():.3f}  {r2(a, b):.4f}  '
          f'{((a - b) ** 2).sum() / sse_n2 * 100:6.2f}%')

print('\n--- 粗分带 ---')
print('band  rows    R2       SSE占n2%')
for lo, hi, lbl in [(0, 1, '0-1'), (2, 3, '2-3'), (4, 5, '4-5'), (6, 999, '6+')]:
    m = (d >= lo) & (d <= hi)
    if m.sum() < 30:
        continue
    a = y[m]; b = p[m]
    print(f'{lbl:5s} {m.sum():6d}  {r2(a, b):.4f}  {((a - b) ** 2).sum() / sse_n2 * 100:6.2f}%')
