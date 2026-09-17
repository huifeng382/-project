"""_local_obj_feas.py — 本地、模型无关：把「换目标函数」那几条路先做**结构性可行性**核算。

目的：在花任何训练成本之前，先淘汰掉「结构上就不成立」的方案。四个问题：

  L1 **M2 的干净 A/B 到底存不存在？** `_pairwise_rank_loss` 的组来自 `data.grp`
     （= 行级组ID，batch 后拼成 (B,) 张量），而组本身是 (expr,corner,switching_pin,
     direction,vector) 的变体集。若一个 80 行的 batch 里几乎全是不同组的行，成对项拿到的
     pair 数 ≈ 0 ⇒ 「保留原 sampler、只加损失项」这条干净 A/B **在结构上是空的**，不是效果差。
     ⇒ 核算：两种 sampler 下每个 batch 的 (pair 数, 组数) 分布。

     🔴 **17.4.3 更正（本项的前提有一处错）**：本项把「原 sampler」建模成了**随机置换**，
     而随机置换只对应**站点 2 的 else**（`DataLoader(..., shuffle=True)`，在**离群点清洗
     分支**里）。**默认路径是站点 1 的 else = `CircuitGroupSampler`（整电路打包）**，实测
     零成对占比与随机置换差很多：`full` 92.1% / `rest` 49.0% / `m4` 26.7%（随机置换分别为
     54.2 / 94.0 / 67.3）⇒ **本项记的那三个数是「另一个 sampler」的读数**。
     结论方向不变（Grouped 仍是成对项非空的前提：0.0% vs 49~92%，成对均值 102~812×），
     但**「rest 九成 batch 成对项为空」这句在默认路径上不成立**（实为 49%、均值 1.88 对/batch）。
     真读数见 `scripts/diag/_t_sampler_live.py` 的 Part C（它直接 exec 站点真源码 + 用真
     `CircuitGroupSampler` 类跑真数据）。

  L2 **M3 边界受限损失可行否？** 只保留「一端属真·前3」的 pair 后，还剩多少 pair、
     有多少组归零（归零的组拿不到任何梯度）。

  L3 **M4 组内归一化可行否？** w_g = 1/max(spread, eps) 的分布；eps 取多少才不炸。
     另给「最大 1% 的组占了多少权重」的集中度。

  L4 **M0 的训练分布那一半**：把「真值近邻竞争数」c_eff(eps) = #{行: t <= t_min*(1+eps)}
     扫一遍 eps，看「允许 eps 相对误差的秩器」能到多少。
     ⚠ 这是**诊断上界**（把 eps 当成秩器的相对分辨力），不是 (b3) 那条严格硬天花板。

用法（本地 Windows）：python scripts/diag/_local_obj_feas.py
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
BATCH = 80                     # config.BATCH_SIZE
RNG = np.random.default_rng(20260917)
EPS = [0.0, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.9e-1]


def load(b):
    parts = sorted(glob.glob(os.path.join(ROOT, 'data', b, 'timing_arcs*.parquet')))
    if not parts:
        return None
    df = pd.concat([pd.read_parquet(p, columns=['DELAY'] + KEY) for p in parts],
                   ignore_index=True) if len(parts) > 1 \
        else pd.read_parquet(parts[0], columns=['DELAY'] + KEY)
    df = df.dropna(subset=['DELAY'])
    df = df[df['DELAY'] > 1e-12].reset_index(drop=True)
    gk = None
    for name in KEY:
        c = df[name].astype(str).to_numpy() if name in df.columns \
            else np.full(len(df), '', dtype=object)
        gk = c if gk is None else np.char.add(np.char.add(gk, '|'), c)
    return pd.factorize(gk)[0], df['DELAY'].to_numpy(dtype=float)


def group_slice(codes):
    """按 codes 排序后，每组在排序数组里连续 ⇒ 返回 (顺序, 组起点, 组终点)。"""
    order = np.argsort(codes, kind='stable')
    _, st = np.unique(codes[order], return_index=True)
    en = np.append(st[1:], len(codes))
    return order, st, en


def pairs_of(m):
    return m * (m - 1) // 2


def l1(codes):
    """两种 sampler 下每 batch 的 (pair 数, 组数)。"""
    n = len(codes)
    nb = n // BATCH
    perm = RNG.permutation(n)[:nb * BATCH].reshape(nb, BATCH)
    allc = np.sort(codes[perm], axis=1)

    # A) 随机 shuffle：batch 内同组的行对才会产生 pair
    pA = np.zeros(nb, dtype=np.int64)
    for i in range(BATCH - 1):
        pA += (allc[:, i + 1:] == allc[:, [i]]).sum(axis=1)
    gA = 1 + (allc[:, 1:] != allc[:, :-1]).sum(axis=1)

    # B) GroupedBatchSampler：整组打包；组大于 batch_size 时单独成 batch（照 src/utils.py:207）
    _, st, en = group_slice(codes)
    size = (en - st).astype(np.int64)
    pB, gB = [], []
    cur_p, cur_rows, cur_groups = 0, 0, 0
    for gi in RNG.permutation(len(size)):
        m = int(size[gi])
        if m > BATCH:
            if cur_groups:
                pB.append(cur_p); gB.append(cur_groups)
                cur_p, cur_rows, cur_groups = 0, 0, 0
            pB.append(pairs_of(m)); gB.append(1)
            continue
        if cur_groups and cur_rows + m > BATCH:
            pB.append(cur_p); gB.append(cur_groups)
            cur_p, cur_rows, cur_groups = 0, 0, 0
        cur_p += pairs_of(m)
        cur_rows += m
        cur_groups += 1
    if cur_groups:
        pB.append(cur_p); gB.append(cur_groups)
    return pA, gA, np.array(pB, dtype=np.int64), np.array(gB, dtype=np.int64)


def scan(dly_s, st, en):
    """一趟扫完 L2(spread/边界对) 与 L4(c_eff 扫描)。"""
    ce_ok = np.zeros(len(EPS), dtype=np.int64)
    n_used = allp_tot = bp_tot = bp_zero = amb = 0
    spreads = []
    for i in range(len(st)):
        t = dly_s[st[i]:en[i]]
        k = t.size
        if k < 4:
            continue
        n_used += 1
        s = np.sort(t)
        t1 = float(s[0])
        if t1 > 0:
            spreads.append((float(s[-1]) - t1) / t1)
        if s[2] == s[3]:
            amb += 1                       # 第3与第4并列 ⇒ 前3 集合本身不唯一
        # 真·前3 掩码（按行序稳定取最小的 3 行）
        idx3 = np.argsort(t, kind='stable')[:3]
        maskB = np.zeros(k, dtype=bool)
        maskB[idx3] = True
        bp = int((t[maskB][:, None] < t[~maskB][None, :]).sum())
        bp_tot += bp
        if bp == 0:
            bp_zero += 1
        # 全部严格有序对 = Σ_{a<b} cnt_a·cnt_b
        _, cnt = np.unique(s, return_counts=True)
        cum = np.cumsum(cnt)
        allp_tot += int((cnt * (cum - cnt)).sum())
        for j, e in enumerate(EPS):
            if int((s <= t1 * (1.0 + e)).sum()) <= 3:
                ce_ok[j] += 1
    return n_used, allp_tot, bp_tot, bp_zero, amb, np.array(spreads), ce_ok


print("本地目标函数可行性核算（模型无关）")
print(f"组键 = {KEY}（同 src/data_loader.py:236）；BATCH={BATCH}\n")

grand_rows = 0
ALL_CE = np.zeros(len(EPS), dtype=np.int64)
ALL_N = 0

for b in BATCHES:
    got = load(b)
    if got is None:
        print(f"[{b}] 缺文件，跳过")
        continue
    codes, dly = got
    grand_rows += len(codes)
    print(f"[{b}] 行 {len(codes):,}")

    pA, gA, pB, gB = l1(codes)
    print("  L1 每 batch：")
    print(f"     随机 shuffle : pair 中位 {np.median(pA):.0f} / 均值 {pA.mean():.1f} / "
          f"p90 {np.percentile(pA,90):.0f} | 组数中位 {np.median(gA):.0f} | "
          f"**零 pair 的 batch 占 {100.0*(pA==0).mean():.1f}%**")
    print(f"     Grouped      : pair 中位 {np.median(pB):.0f} / 均值 {pB.mean():.1f} / "
          f"p90 {np.percentile(pB,90):.0f} | 组数中位 {np.median(gB):.0f} | "
          f"零 pair 的 batch 占 {100.0*(pB==0).mean():.1f}%")
    print(f"     ⇒ 均值之比 Grouped / 随机 = {pB.mean()/max(pA.mean(),1e-9):,.0f}×")

    order, st, en = group_slice(codes)
    n_used, allp, bp, bpz, amb, sp, ce = scan(dly[order], st, en)
    ALL_CE += ce
    ALL_N += n_used

    print(f"  L2 边界受限（M3）：参与组 {n_used:,}")
    print(f"     全部严格有序对 {allp:,} → 边界对 {bp:,} = **{100.0*bp/max(allp,1):.1f}%**"
          f"（保留 {100.0*bp/max(allp,1):.1f}% 的 pair）")
    print(f"     边界对归零的组 {bpz:,} = **{100.0*bpz/max(n_used,1):.2f}%**（这些组无梯度）")
    print(f"     第3与第4 并列（前3 集合不唯一）的组 {amb:,} = {100.0*amb/max(n_used,1):.1f}%")

    si = sp[np.isfinite(sp) & (sp > 0)]
    print(f"  L3 组内归一（M4）：spread 中位 {100*np.median(si):.3f}% | "
          f"<0.1% 占 {100*(si<1e-3).mean():.1f}% | <1% 占 {100*(si<1e-2).mean():.1f}%")
    for e in [1e-4, 1e-3, 1e-2, 5e-2]:
        w = np.sort(1.0 / np.maximum(si, e))[::-1]
        top1 = w[:max(1, len(w)//100)].sum() / w.sum()
        print(f"     eps={e:<7g} 权重中位 {np.median(w):>9.1f} / max {w.max():>11.1f} | "
              f"**最大 1% 的组占总权重 {100*top1:5.1f}%**")

    line = "  L4 c_eff(eps) 诊断上界："
    print(line)
    for j, e in enumerate(EPS):
        print(f"     eps={e:<8g} 命中率上界 {100.0*ce[j]/max(n_used,1):6.2f}%")
    print()

print("=" * 72)
print(f"合计 行 {grand_rows:,} | 参与统计组 {ALL_N:,}")
print("  L4 eps=0（严格）：" + f"{100.0*ALL_CE[0]/max(ALL_N,1):.2f}%")
for j, e in enumerate(EPS):
    print(f"  L4 eps={e:<8g} {100.0*ALL_CE[j]/max(ALL_N,1):6.2f}%")
print("=" * 72)
