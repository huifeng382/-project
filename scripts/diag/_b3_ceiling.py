"""_b3_ceiling.py — (b3) 探针（本地、模型无关）：严格 recall@3 在**训练分布**上的
可达天花板 / 机会底 / 代价结构。

答案服务于一个决策：(b)「换一个只压前3 边界的目标函数」值不值得开一趟训练？
  若天花板本来就低（大片组**任何秩器**都不可能命中）⇒ 换目标函数无用，是数据/任务的问题。
  若天花板高而机会底很低 ⇒ 目标是瓶颈，值得动。
  若大量组的组内次序**低于可分性/噪声底** ⇒ 那些组上的前3 训练信号是噪声，
     换目标函数会把噪声也一起放大（严格@3 对 1 ulp 的敏感度是遗憾的 ~17 倍，已实测）。

四个量（全部只从 data/*/timing_arcs*.parquet 的 DELAY 算出，不需要模型/服务器）：
  A 机会集：行 / 组 / 组大小分布
  B **硬天花板**：真·最小延迟的等价类大小 c。命中条件是「预测前3 里含真最小」，
     而任何秩器最多放 3 个 ⇒ **c>3 的组永远不可能命中** ⇒ 天花板 = frac(c<=3)。
     同时给出把 c>3 的组排除后的等价读数。
  C **机会底**：随机秩器的严格 recall@3 = 1 - C(k-c,3)/C(k,3)（按组求期望）。
  D **可分性**：组内前2 的相对间隔 (t2-t1)/t1 —— 间隔极小 ⇒「前3 内真最优」退化为
     近乎掷硬币，训练信号是噪声。同时给 spread=(max-min)/min 的分布（miss 的代价）。

⚠ 口径提醒：这是**训练分布**（变体组，k=10~15）。**部署口径**是 714 个池、k 可达 200+，
   两者形状不同 ⇒ 本脚本的天花板**不能直接与部署 90.8% 相减**。部署侧天花板要另算
   （需要 ~/shadow_archive 里的候选级 CSV，本地没有）。

用法（本地 Windows）：python scripts/diag/_b3_ceiling.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BATCHES = ['batch_v2_full', 'batch_v2_rest', 'batch_v2_m4']
KEY = ['expr', 'corner', 'switching_pin', 'direction', 'vector']

print(f"(b3) 训练分布严格 recall@3 天花板/机会底  |  范围 = {','.join(BATCHES)}")
print(f"组键 = {KEY}（同 src/data_loader.py:236）\n")

TOT = dict(rows=0, groups=0, c_ok=0, c_bad=0, chance_hit=0.0, chance_n=0,
           r12=[], spread=[], kk=[])

for b in BATCHES:
    parts = sorted(glob.glob(os.path.join(ROOT, 'data', b, 'timing_arcs*.parquet')))
    if not parts:
        print(f"[{b}] 缺文件，跳过")
        continue
    df = pd.concat([pd.read_parquet(p, columns=['DELAY'] + KEY) for p in parts], ignore_index=True) \
        if len(parts) > 1 else pd.read_parquet(parts[0], columns=['DELAY'] + KEY)
    df = df.dropna(subset=['DELAY'])
    df = df[df['DELAY'] > 1e-12].reset_index(drop=True)

    cols = []
    for name in KEY:
        cols.append(df[name].astype(str).to_numpy() if name in df.columns
                    else np.full(len(df), '', dtype=object))
    gk = cols[0]
    for c in cols[1:]:
        gk = np.char.add(np.char.add(gk, '|'), c)
    codes = pd.factorize(gk)[0]
    dly = df['DELAY'].to_numpy(dtype=float)

    # 按组切片：排序一次，O(n log n)，避开「每组扫全表」的 O(n·g)
    order = np.argsort(codes, kind='stable')
    sc = codes[order]
    st = np.searchsorted(sc, np.arange(sc[-1] + 1), side='left')
    en = np.searchsorted(sc, np.arange(sc[-1] + 1), side='right')
    ts = dly[order]

    n_g = len(st)
    c_hist = np.zeros(8, dtype=int)          # c=1..7, 8=8+
    c_ok = 0
    chance = 0.0
    chance_n = 0
    r12 = []
    spread = []
    kk = []
    for i in range(n_g):
        t = ts[st[i]:en[i]]
        k = t.size
        if k < 4:
            continue
        kk.append(k)
        tmin = t.min()
        c = int((t == tmin).sum())
        c_hist[min(c, 8) - 1] += 1
        if c <= 3:
            c_ok += 1
        # 机会底：P(前3 命中) = 1 - C(k-c,3)/C(k,3)
        from math import comb
        p = 1.0 - (comb(k - c, 3) / comb(k, 3) if k - c >= 3 else 0.0)
        chance += p
        chance_n += 1
        # 前2 相对间隔
        t2 = np.partition(t, 1)[:2]
        lo, hi = float(t2.min()), float(t2.max())
        if lo > 0:
            r12.append((hi - lo) / lo)
        spread.append((float(t.max()) - tmin) / tmin if tmin > 0 else np.nan)

    print(f"[{b}] 组 {n_g:,}（k>=4 的 {chance_n:,}）")
    print(f"      c 分布(前8档 c=1..7,8+): {c_hist.tolist()}")
    print(f"      硬天花板 frac(c<=3) = {100.0*c_ok/max(chance_n,1):.2f}%"
          f"  |  机会底(随机秩器) = {100.0*chance/max(chance_n,1):.2f}%")
    r12 = np.array(r12); spread = np.array(spread)
    for lbl, q in [('前2 相对间隔', r12), ('spread (max-min)/min', spread)]:
        q = q[np.isfinite(q)]
        print(f"      {lbl}: 中位 {100*np.median(q):.3f}% | "
              f"<0.1% 占 {100*(q<0.001).mean():.1f}% | <1% 占 {100*(q<0.01).mean():.1f}% | "
              f"<5% 占 {100*(q<0.05).mean():.1f}% | <10% 占 {100*(q<0.10).mean():.1f}%")

    TOT['rows'] += len(df)
    TOT['groups'] += n_g
    TOT['c_ok'] += c_ok
    TOT['chance_hit'] += chance
    TOT['chance_n'] += chance_n
    TOT['r12'].append(r12)
    TOT['spread'].append(spread)
    TOT['kk'].append(np.array(kk))

r12 = np.concatenate(TOT['r12']); spread = np.concatenate(TOT['spread']); kk = np.concatenate(TOT['kk'])
print("\n" + "=" * 68)
print(f"合计：行 {TOT['rows']:,} | 组 {TOT['groups']:,} | 参与统计的组(k>=4) {TOT['chance_n']:,}")
print(f"组大小: 中位 {np.median(kk):.0f} / p99 {np.percentile(kk,99):.0f} / max {kk.max()}")
print(f"**硬天花板 frac(c<=3) = {100.0*TOT['c_ok']/TOT['chance_n']:.2f}%**"
      f"   ← 任何秩器都不可能更高（c>3 的组前3 装不下真最小值）")
print(f"**机会底  随机秩器     = {100.0*TOT['chance_hit']/TOT['chance_n']:.2f}%**")
r12i = r12[np.isfinite(r12)]
print(f"可分性  前2 相对间隔: 中位 {100*np.median(r12i):.3f}% | <0.1% 占 {100*(r12i<0.001).mean():.1f}%"
      f" | <1% 占 {100*(r12i<0.01).mean():.1f}%")
si = spread[np.isfinite(spread)]
print(f"代价    spread: 中位 {100*np.median(si):.1f}% | <5% 占 {100*(si<0.05).mean():.1f}%"
      f" | <10% 占 {100*(si<0.10).mean():.1f}%")
print("=" * 68)
