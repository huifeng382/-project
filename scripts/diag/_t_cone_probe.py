"""n<=2 桶修正轴探针: incone(是否在本次切换 src 锥内) × cone_down(距 src 归一深)。

背景修正: 原 _t_depth_probe 用静态 depth(主输入逻辑深)当"埋没度"= 错轴; 本任务一行 = 单次开关事件,
  正确传播轴 = 距本次 src 的距离 (cone_down)。n<=2 的"低电流群/高电流群"大概率 = 不在锥/在锥。
  且 assemble 里 f_slew 是 src 的 slew 广播给锥内所有门 → 锥内门没有自己的到达 slew (只有归一 cone_down)。

判定:
  - 锥内(incone=1)n<=2 门 R^2 随 cone_down 增大明显衰减 => 传播假设成立: 缺"per-gate 到达 slew/load"
    (logical-effort 沿锥传播, sim-free/STA/Rust 可行) -> 0.85 的实特征杠杆。
  - 锥内近 src(cone_down~0)也已 ~0.6 且不随距离变 => 均匀墙成立, 别赌传播特征。

用法 (在 ~/idsavg17 下): ~/venv/bin/python3 scripts/diag/_t_cone_probe.py
"""
import pandas as pd, numpy as np

df = pd.read_parquet('resid_R4_pin_17016.parquet',
                     columns=['y', 'pred', 'n_t', 'incone', 'cone_down', 'depth'])
df = df[df['n_t'] <= 2]
y = df['y'].to_numpy(); p = df['pred'].to_numpy()
def r2(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return 1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum()
print(f'n<=2 total: rows={len(y)}  R2={r2(y, p):.4f}  mean_y={y.mean():.3f}')
sse_n2 = ((y - p) ** 2).sum()

incone = (df['incone'].to_numpy() > 0)
for lbl, m in [('incone=0 不在切换锥', ~incone), ('incone=1 在切换锥内', incone)]:
    a = y[m]; b = p[m]
    print(f'{lbl}: rows={m.sum():7d}  R2={r2(a, b):.4f}  mean_y={a.mean():.3f}  '
          f'SSE占n2%={((a - b) ** 2).sum() / sse_n2 * 100:.2f}%')

print('\n--- depth(静态) x incone 交叉行数 (解释旧探针两群) ---')
print(pd.crosstab(df['depth'].astype(int), incone.astype(int)))

mc = incone
dd = df.loc[mc, 'cone_down'].to_numpy()
print('\n--- 锥内 n<=2 按 cone_down(距src, 归一) 分层 ---')
print('cone_down  rows    R2       SSE占锥内%  mean_y')
sse_in = ((y[mc] - p[mc]) ** 2).sum()
for lo, hi in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.001)]:
    mm = mc & (dd >= lo) & (dd < hi)
    if mm.sum() < 30:
        continue
    a = y[mm]; b = p[mm]
    print(f'{lo:.1f}-{hi:.1f}  {mm.sum():6d}  {r2(a, b):.4f}  '
          f'{((a - b) ** 2).sum() / sse_in * 100:6.2f}%  {a.mean():.3f}')

res2 = (y[mc] - p[mc]) ** 2
print(f'\n锥内 |残差|^2 与 cone_down 相关: {np.corrcoef(res2, dd)[0, 1]:+.3f}')
print(f'锥内 |残差|^2 与 depth(静态) 相关: '
      f'{np.corrcoef(res2, df.loc[mc, "depth"].to_numpy())[0, 1]:+.3f}')
