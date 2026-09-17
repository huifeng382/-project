"""_t_rankpairs.py — 本地测算：训练侧「变体组」大小分布 & 成对排序损失里
「涉及真·前3」的损失质量占比。

为什么测这个：(b) 方案（top-k 加权排序损失）的**动机强度**完全取决于组大小 k。
  _pairwise_rank_loss 对每组取组内**全部** dt<0 有序对的 relu(dp+margin) 的均值，
  而 margin 是 log10 空间里的**绝对**间隔（3% ≈ 7% 相对），与这对在名次里的位置无关
  ⇒ 名次中/低的那些对和「真·前3 vs 其余」的对拿同样的目标间隔。
  涉及真·前3 的对数 ≈ 3(k-3)，全部 dt<0 对数 = k(k-1)/2 ⇒ 占比 ≈ 6/k。
  k=10 → ~60%（不饥饿，top-k 加权几乎无增量）
  k=200 → ~3%（严重饥饿）
本脚本给出**按对数量加权**的实测占比 —— 不是 6/k 的均值近似，是真实聚合值。

组键与 src/data_loader.py:236 完全一致：expr|corner|switching_pin|direction|vector
取数范围与 config.DATA_BATCHES 默认一致：batch_v2_full,batch_v2_rest,batch_v2_m4

用法（本地 Windows）：python scripts/diag/_t_rankpairs.py
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

print(f"范围 = config.DATA_BATCHES 默认 = {','.join(BATCHES)}")
print(f"组键 = {KEY}（同 src/data_loader.py:236）")
print("⚠ batch_v2_rest 是**分片**的（timing_arcs_part1..6.parquet）—— 与 train_sweep.py:236-255 同法全读\n")

size_all, pairs_all, top3_pairs_all, rows_all = [], 0, 0, 0

for b in BATCHES:
    # 分片感知：timing_arcs.parquet 与 timing_arcs_partN.parquet 都收
    parts = sorted(glob.glob(os.path.join(ROOT, 'data', b, 'timing_arcs*.parquet')))
    if not parts:
        print(f"[{b}] 缺文件，跳过: data/{b}/timing_arcs*.parquet")
        continue
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    if len(parts) > 1:
        print(f"[{b}] 读了 {len(parts)} 个分片: {[os.path.basename(p) for p in parts]}")
    n0 = len(df)
    # 与 data_loader:199-200 一致：丢掉 DELAY 非正/缺失
    if 'DELAY' in df.columns:
        df = df.dropna(subset=['DELAY'])
        df = df[df['DELAY'] > 1e-12]
    # 与 data_loader:233-235 同法取列（缺失填空串），并转成 Python 层可迭代的 numpy
    cols = []
    for name in KEY:
        if name in df.columns:
            cols.append(df[name].astype(str).to_numpy())
        else:
            cols.append(np.full(len(df), '', dtype=object))
    gk = cols[0]
    for c in cols[1:]:
        gk = np.char.add(np.char.add(gk, '|'), c)
    codes, uniq = pd.factorize(gk)
    sizes = np.bincount(codes)
    sizes = sizes[sizes >= 1]

    # 每组【精确】统计（不用 k(k-1)/2 近似 —— 并列 dt==0 的对**不进** mask，会被高估）
    #   n_dt_ord  = 该组进入 `mask = dt<0` 的对数 = 目标值不等的无序对数
    #   n_t3_ord  = 其中「恰有一个元素属于真·前3」的对数 = 真正对齐部署口径的那部分
    dly = df['DELAY'].to_numpy(dtype=float)
    t3 = 0
    n_dt_ord = 0
    ties = 0
    allpairs = 0
    for c in range(len(uniq)):
        m = codes == c
        k = int(m.sum())
        if k < 2:
            continue
        t = dly[m]
        allpairs += k * (k - 1) // 2
        if k < 4:
            continue
        # 真前3 = 该组最小的 3 个（并列时按 stable 行序取 —— 见文末 caveat）
        top3 = np.zeros(k, dtype=bool)
        top3[np.argsort(t, kind='stable')[:3]] = True
        # dt = t_i - t_j for i<j；用广播算 (k,k) 下三角
        dt = t[:, None] - t[None, :]
        iu = np.triu_indices(k, k=1)
        dv = dt[iu]                                   # 无序对 (i,j), i<j
        nz = dv != 0
        n_dt_ord += int(nz.sum())                     # 每对不等 → 恰贡献 1 个 dt<0
        ties += int((~nz).sum())
        # 恰有一个在真前3 的对
        one_t3 = top3[iu[0]] ^ top3[iu[1]]
        t3 += int((one_t3 & nz).sum())

    k = sizes.astype(np.int64)
    print(f"[{b}] 行 {n0:,} → 有效 {len(df):,} | 组 {len(uniq):,}")
    print(f"      组大小: min {k.min()} / p25 {np.percentile(k,25):.0f} / 中位 {np.median(k):.0f} / "
          f"p75 {np.percentile(k,75):.0f} / p90 {np.percentile(k,90):.0f} / "
          f"p99 {np.percentile(k,99):.0f} / max {k.max()}")
    print(f"      行占比: k>=50 的组占 {100.0*k[k>=50].sum()/k.sum():.1f}% 行 | "
          f"k>=200 占 {100.0*k[k>=200].sum()/k.sum():.1f}% 行 | k<4 的组 {int((k<4).sum()):,} 个")
    print(f"      无序对 {allpairs:,} | 其中**并列**(dt==0) {ties:,} = {100.0*ties/max(allpairs,1):.1f}% "
          f"| 进 mask 的 dt<0 对 {n_dt_ord:,}")
    size_all.append(k)
    pairs_all += n_dt_ord
    top3_pairs_all += t3
    rows_all += len(df)

print("\n" + "=" * 62)
print(f"合计：行 {rows_all:,} | 进 mask 的 dt<0 有序对 {pairs_all:,}")
frac = 100.0 * top3_pairs_all / max(pairs_all, 1)
print(f"**「恰有一个属真·前3」的对 / 进 mask 的 dt<0 对 = {top3_pairs_all:,} / {pairs_all:,} = {frac:.2f}%**")
print(f"  （逐组近似 6/k：k=15 → 3·15-6=39 对 / 105 对 = 37%；并行加权的实测聚合值见上）")
print("=" * 62)
print("""
读法：
  · 该占比 = 现损失里「真正对齐部署口径（真·前3 落在预测前3）」的那部分梯度质量。
    其余 (100-该值)% 花在「名次中/低的对」上 —— 它们对严格@3 几乎无影响，却同样拿 margin。
  · **训练侧组极小且整齐**（无 k≥50 的组）⇒「组大了对数爆炸、前3 边界被淹没」这个机制
    **在本数据上不成立**（那是部署侧 200+ 候选池的形状，不是训练侧变体组的形状）。
    这是对 17.4.0 之前口头机制的一次**实测否证** —— top-k 加权的增量应按实测占比估，
    不能按「k 很大」的直觉估。
  · 真正 size-independent 的不对齐在**margin 的绝对性**：`RANK_MARGIN=0.03`（log10）对
    「真#1 vs 真#15」和「真#7 vs 真#8」是同一个目标间隔，而部署只消费前 3 边界。
  · caveat：真前3 用 argsort 取，并列时按行序。并列占比已实测（见上）；部署侧严格 recall@3
    对并列免疫（取预测前3 内真值最小值），故并列不改结论方向。""")
