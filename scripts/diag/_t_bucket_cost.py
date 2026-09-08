"""桶成本分解: "0.85 需要动哪个桶、动多少"? (零训练, 跑现有 resid)

对 R4 单模型 与 B 复合 (R4 n<8 + R6 ge8) 各算: 每桶 SSE 占全局 SSE 的份额
= 若该桶残差归零, headline R^2 能抬多少。回答"墙在哪一桶、值不值得攻"。

用法 (在 ~/idsavg17 下): ~/venv/bin/python3 scripts/diag/_t_bucket_cost.py
"""
import pandas as pd, numpy as np

R4 = pd.read_parquet('resid_R4_pin_17016.parquet', columns=['y', 'pred', 'n_t'])   # 全 test 行, 共享+腿
R6 = pd.read_parquet('resid_R6_nt8pin_17019.parquet', columns=['y', 'pred', 'n_t']) # ge8 行, 专化+腿
ge = (R4['n_t'] >= 8).to_numpy()
assert len(R6) == ge.sum() and np.array_equal(R6['y'].to_numpy(), R4['y'].to_numpy()[ge]), 'R6 ge8 与 R4 的 ge8 行未对齐'
print(f'[对齐] R6 {len(R6)} 行 = R4 ge8 行, y 逐位一致 OK')

y = R4['y'].to_numpy(); nt = R4['n_t'].to_numpy()
SStot = ((y - y.mean()) ** 2).sum()
BUCKETS = [(0, 2, 'n<=2 '), (3, 7, 'n3-7 '), (8, 999, 'n>=8 ')]

def show(pred, name):
    full = 1 - ((y - pred) ** 2).sum() / SStot
    print(f'\n===== {name}: full test R^2 = {full:.4f} =====')
    for lo, hi, lbl in BUCKETS:
        m = (nt >= lo) & (nt <= hi)
        SSE = ((y[m] - pred[m]) ** 2).sum()
        gain = SSE / SStot                      # 若此桶归零 headline 抬升
        print(f'  {lbl} rows={m.sum():6d} ({m.mean()*100:4.1f}%)  mean_y={y[m].mean():.3f}  '
              f'headline杠杆(SSE/全局SS)={SSE/SStot*100:5.2f}%  → 桶归零则 headline {full+gain:.4f}')

pB = R4['pred'].to_numpy().copy(); pB[ge] = R6['pred'].to_numpy()
show(R4['pred'].to_numpy(), 'R4 共享+腿 单模型')   # 应≈0.7883
show(pB, 'B 复合 (R4 n<8 + R6 ge8)')               # 应≈0.7912, 验证对齐
