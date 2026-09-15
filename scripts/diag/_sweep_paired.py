"""_sweep_paired.py — 多 epoch 扫描结果的**逐集配对**检验（17.2.7 新增）。

为什么需要它：`_shadow_ckpt_sweep.sh` 的汇总行比的是**各 epoch 的独立均值**
（"选择遗憾 x% vs y%"），但 GNN 是纯观察者 → 搜索轨迹与模型无关
→ **每个 epoch 看到的是同一批候选集、同一批真值**（_shadow_analyze.py:387 已注明）。
所以正确统计量是**逐集差值**；独立均值比较把配对结构丢掉了，方差被高估。

⚠ 17.2.7 二次修正（首版读数不可用）：首版按"逐集最优次数"排名，但
  `两阶段最终遗憾`/`严格k3` 这类量有**巨大的点质量**（真值最优落进 GNN 前 3
  → 遗憾恒为 0；中位数 0.0% 就是证据）。平局时首版把票投给**列表里排第一的
  epoch**，于是 ep50 拿到"77/106 最优"这种假冠军；而"优集占比"用 1-P(d>0)
  **把平局算进了更优**。两个读数都被平局污染。
  本版：①平局单独报数；②胜负只算**唯一最小值**；③所有排名与检验只在
  **有区分度的集**（五 epoch 不全等）上做——全等的集对排序不提供任何信息，
  留着只稀释样本、把 MDD 撑大，还会让人误以为"分辨率不够"。

输入：`_shadow_ckpt_sweep.sh` 产出的 `~/sweep_<TAG>_ep<N>.out`（一个 ckpt 一份）。
      解析其中的「每候选集明细」段（_shadow_analyze.py:391-397）。

用法（服务器，终端 = 服务器）：
  ~/venv/bin/python3 scripts/diag/_sweep_paired.py --glob '~/sweep_v2nowave42b_ep*.out'
  自检（不需要数据）：~/venv/bin/python3 scripts/diag/_sweep_paired.py --selftest

输出：
  1) 每判据的平局结构：n_总 / n_全等（无区分度）/ n_有效
  2) 有效子集上各 epoch 的 均值 / 中位 / p90 / **严格最优次数**
  3) 配对差：严格更优 / 打平 / 更差 三分数 + SE + t + **MDD**
     MDD = 2.80·SE（80% power，双侧 5%）；观测差 ≤ MDD → **不可分辨**
     （注意：不可分辨 ≠ 打平。打平=两端一样好；不可分辨=样本量不足以判断。
      两者对"要不要改判据"的结论相反。）
"""
import argparse, glob, math, os, re, statistics, sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8')
    except (AttributeError, OSError):
        pass

DETAIL_MARK = '每候选集明细'   # _shadow_analyze.py:391 的段头子串
ROW_RE = re.compile(
    r'^\s*(?P<circ>\S+)\s+w=\s*(?P<w>\d+)\s+n=\s*(?P<n>\d+)\s+'
    r'#1∈前2=(?P<r2s>[Yn])\s+前2∩真=(?P<r2l>[Yn])\s+'
    r'#1∈前3=(?P<r3s>[Yn])\s+前3∩真=(?P<r3l>[Yn])\s+'
    r'regret=\s*(?P<regret>[\d.]+)%\s+2stage=\s*(?P<stage>[\d.]+)%\s+sp=(?P<sp>\S+)'
)
SP_NULL = ('-', '', 'nan', 'NaN', 'N/A', 'None')

METRICS = {
    'stage':   ('两阶段最终遗憾【交付口径】', lambda r: r['stage'],  True,  '{:.2f}%'),
    'regret':  ('选择遗憾(GNN自选)',      lambda r: r['regret'], True,  '{:.2f}%'),
    'strict3': ('严格 k=3',             lambda r: r['r3s'],    False, '{:.1%}'),
    'len3':    ('宽松 k=3',             lambda r: r['r3l'],    False, '{:.1%}'),
    'strict2': ('严格 k=2',             lambda r: r['r2s'],    False, '{:.1%}'),
    'len2':    ('宽松 k=2',             lambda r: r['r2l'],    False, '{:.1%}'),
    'sp':      ('Spearman',            lambda r: r['sp'],     False, '{:.3f}'),
}


def parse_out(path):
    """→ {(circ, window): row}。只取「每候选集明细」段之后的行。"""
    rows, in_detail = {}, False
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if DETAIL_MARK in line:
                in_detail = True
                continue
            if not in_detail:
                continue
            m = ROW_RE.match(line)
            if not m:
                continue
            d = m.groupdict()
            rows[(d['circ'], int(d['w']))] = {
                'n': int(d['n']),
                'r2s': 1.0 if d['r2s'] == 'Y' else 0.0,
                'r2l': 1.0 if d['r2l'] == 'Y' else 0.0,
                'r3s': 1.0 if d['r3s'] == 'Y' else 0.0,
                'r3l': 1.0 if d['r3l'] == 'Y' else 0.0,
                'regret': float(d['regret']),
                'stage': float(d['stage']),
                'sp': (None if d['sp'] in SP_NULL else float(d['sp'])),
            }
    return rows


def label(path):
    m = re.search(r'ep(\d+)', os.path.basename(path))
    return f"ep{m.group(1)}" if m else os.path.basename(path)


def pct(vals, q):
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    i = q * (len(s) - 1)
    lo = int(i)
    if lo + 1 >= len(s):
        return s[-1]
    return s[lo] * (1 - (i - lo)) + s[lo + 1] * (i - lo)


def mdd(se):
    """最小可辨差（80% power，双侧 5%）：(1.96+0.84)·SE。"""
    return 2.80 * se


def paired(a, b):
    """a=对比, b=基准（同键序）→ 逐集差 d=a-b 的统计量与三分数。"""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    if n < 2:
        return None
    md = statistics.mean(d)
    se = statistics.stdev(d) / math.sqrt(n)
    nlt = sum(1 for x in d if x < 0)     # a 更小
    ngt = sum(1 for x in d if x > 0)     # a 更大
    return {'md': md, 'se': se, 'n': n,
            't': (md / se if se > 0 else float('nan')),
            'lt': nlt / n, 'eq': (n - nlt - ngt) / n, 'gt': ngt / n}


def report(files):
    table = {label(p): parse_out(p) for p in files}
    order = [label(p) for p in files]
    print("=== 配对检验 ===")
    for p, k in zip(files, order):
        print(f"  {k:>7s}: {len(table[k]):3d} 集   {os.path.basename(p)}")

    keys = set.intersection(*[set(t.keys()) for t in table.values()])
    miss = {k: len(table[k]) - len(keys) for k in order}
    print(f"  配对键 = (circuit, window)：交集 {len(keys)} 集；各文件未配对 {miss}")
    if any(v > 0 for v in miss.values()):
        print(f"  ⚠ 有未配对集 → 检查各次扫描的 --min-cands 是否一致（_shadow_analyze.py:239）")
    bad = [k for k in keys if len({table[e][k]['n'] for e in order}) > 1]
    if bad:
        print(f"  ❌ 配对可能失效：{len(bad)} 个键的 n 不一致（例：{bad[:3]}）→ 结论仅供参照")
    else:
        print(f"  ✅ 配对有效：{len(keys)} 个键的候选数 n 在所有 epoch 一致（轨迹与模型无关）")

    summary, tied_report = [], []
    for mk, (name, get, lower, fmt) in METRICS.items():
        ks_all = sorted(k for k in keys if all(get(table[e][k]) is not None for e in order))
        if not ks_all:
            continue
        allv = {e: [get(table[e][k]) for k in ks_all] for e in order}
        tied = [k for k in ks_all if len({get(table[e][k]) for e in order}) == 1]
        ts = set(tied)
        ks = [k for k in ks_all if k not in ts]
        print(f"\n--- 判据：{name}（{'越低越好' if lower else '越高越好'}）---")
        print(f"  平局结构：总 {len(ks_all)} 集；五 epoch 全等 {len(tied)} 集 "
              f"({len(tied)/len(ks_all):.0%}，对排序无信息) → **有效集 {len(ks)}**")
        if not ks:
            print(f"  ⚠ 全部集在所有 epoch 上完全相同 → 该判据对本批数据无区分度，跳过")
            summary.append((name, '(无区分度)', 0.0, 0, len(ks_all)))
            tied_report.append((name, len(tied), len(ks_all)))
            continue
        v = {e: [get(table[e][k]) for k in ks] for e in order}
        means = {e: statistics.mean(v[e]) for e in order}
        meds = {e: statistics.median(v[e]) for e in order}
        best_m = min(means, key=means.get) if lower else max(means, key=means.get)
        best_d = min(meds, key=meds.get) if lower else max(meds, key=meds.get)

        # 严格最优次数（唯一最小值；平局不记给任何人）
        win = {e: 0 for e in order}
        ntie = 0
        for k in ks:
            c = [(e, get(table[e][k])) for e in order]
            m = (min(c, key=lambda t: t[1]) if lower else max(c, key=lambda t: t[1]))[1]
            w = [e for e, x in c if x == m]
            if len(w) == 1:
                win[w[0]] += 1
            else:
                ntie += 1

        print(f"  {'epoch':>7s}{'均值':>11s}{'中位':>11s}{'p90':>11s}{'严格最优':>10s}")
        for e in order:
            print(f"  {e:>7s}{fmt.format(means[e]):>11s}{fmt.format(meds[e]):>11s}"
                  f"{fmt.format(pct(v[e], 0.90)):>11s}{win[e]:>10d}")
        print(f"  有效集内并列最优 {ntie} 集；均值最优={best_m}，中位最优={best_d}"
              + ("  ⚠ 两者不一致 → 分布重尾，均值被少数集主导" if best_m != best_d else ""))

        print(f"  {'对比':>16s}{'均值差':>10s}{'SE':>8s}{'t':>7s}{'更优':>7s}{'打平':>7s}{'更差':>7s}{'MDD':>8s}")
        unres = []
        for e in order:
            if e == best_m:
                continue
            st = paired(v[e], v[best_m])
            if st is None:
                continue
            fav = st['lt'] if lower else st['gt']      # best_m 更好的集占比
            opp = st['gt'] if lower else st['lt']
            print(f"  {best_m + ' vs ' + e:>16s}{st['md']:>10.3f}{st['se']:>8.3f}"
                  f"{st['t']:>7.2f}{fav:>6.0%}{st['eq']:>7.0%}{opp:>7.0%}{mdd(st['se']):>8.3f}")
            if abs(st['md']) <= mdd(st['se']):
                unres.append((f"{best_m} vs {e}", abs(st['md']), mdd(st['se'])))
        if unres:
            print(f"  ⚠ 观测差 ≤ MDD → 不可分辨（≠打平）：")
            for nm, d0, m0 in unres:
                why = ("有效集内差恒为 0" if m0 == 0 else "样本量不足以分辨")
                print(f"      {nm}: 观测差 {d0:.3f} ≤ MDD {m0:.3f}   [{why}]")
        else:
            print(f"  ✅ 全部 ≥ MDD → 该判据下 {best_m} 与其余可分")
        summary.append((name, best_m, win[best_m] / max(1, sum(win.values())), len(unres),
                        len(ks_all)))
        tied_report.append((name, len(tied), len(ks_all)))

    print("\n=== 结论摘要 ===")
    print(f"  {'判据':<26}{'均值最优':>10}{'严格最优占比':>14}{'不可分辨数':>12}")
    for name, b, share, nr, ntot in summary:
        print(f"  {name:<26}{b:>10}{share:>13.0%}{nr:>12}")
    print("\n  平局结构（平局多 = 该判据分辨率低，不是样本量问题）：")
    for name, nt, ntot in tied_report:
        print(f"    {name:<26} 全等 {nt:>3d}/{ntot:<4d} = {nt/max(1,ntot):>4.0%}")
    return 0


def _emit(path, keys, rows):
    with open(path, 'w', encoding='utf-8') as f:
        f.write("=== 每候选集明细（按遗憾升序；#1∈前k=严格, 前k∩真前k=宽松）===\n")
        for k in keys:
            n, rg, st, r3, r2 = rows[k]
            f.write(f"  {k[0]:<45s} w={k[1]:3d} n={n:2d} "
                    f"#1∈前2={'Y' if r2 else 'n'} 前2∩真={'Y' if r2 else 'n'} "
                    f"#1∈前3={'Y' if r3 else 'n'} 前3∩真={'Y' if r3 else 'n'} "
                    f"regret={rg:7.2f}% 2stage={st:6.2f}% sp=0.50\n")


def selftest():
    """四棵树，固定算术、不用随机。"""
    import tempfile
    d = tempfile.mkdtemp(prefix='_sweep_paired_')
    keys = [(f"level0/C{i}/w0/gnn_shadow.csv", 0) for i in range(20)]

    def build(tag, fn):
        for i in range(2):
            ep = (i + 1) * 50
            _emit(os.path.join(d, f"{tag}_ep{ep}.out"), keys,
                  {k: fn(j, i) for j, k in enumerate(keys)})
        return [os.path.join(d, f"{tag}_ep{ep}.out") for ep in (50, 100)]

    # 1) 一致赢家：每个集 ep100 都低 1.0
    f1 = build('consistent', lambda j, i: (6, 10.0 - i, 0.0, True, True))
    # 2) 掷硬币：偶数集 ep50 好、奇数集 ep100 好，幅度相同
    f2 = build('coin', lambda j, i: (6, 5.0 + (0.0 if (i == 0) == (j % 2 == 0) else 2.0),
                                     0.0, True, True))
    # 3) 配对失效：第二个 epoch 的 n 不同
    f3 = build('broken', lambda j, i: (6 if i == 0 else 5, 5.0, 0.0, True, True))
    # 4) 平局主导：20 集里 14 集两 epoch 相等；3 集 ep100 好 1.0；3 集 ep50 好 1.0
    #    期望：全等 14/20 = 70%，有效 6，严格最优 **各 3**，均值差 0 → 不可分辨
    f4 = build('tied', lambda j, i: (
        6,
        (9.0 if j < 14 else (8.0 if i == 1 else 9.0)) if j < 17 else (8.0 if i == 0 else 9.0),
        0.0, True, True))

    print("### 自检 1：一致赢家（应 ✅ 可分、严格最优 20/0）")
    r = report(f1)
    print("\n### 自检 2：掷硬币（应 ⚠ 不可分辨、更优 50%）")
    r |= report(f2)
    print("\n### 自检 3：配对失效（应 ❌ n 不一致）")
    r |= report(f3)
    print("\n### 自检 4：平局主导（应报 全等 70%、有效 6、严格最优 3/3、不可分辨）")
    r |= report(f4)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='*', help='sweep_<TAG>_ep<N>.out，按 epoch 升序给')
    ap.add_argument('--glob', dest='pat', default=None, help="如 '~/sweep_v2nowave42b_ep*.out'")
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    files = list(a.files)
    if a.pat:
        def _ep(p):
            m = re.search(r'ep(\d+)', p)
            return int(m.group(1)) if m else 0
        files = sorted(glob.glob(os.path.expanduser(a.pat)), key=_ep)
    if len(files) < 2:
        print("ERROR: 至少给两个 .out（不同 epoch）"); return 2
    for p in files:
        if not os.path.exists(p):
            print(f"ERROR: 找不到 {p}"); return 2
    return report(files)


if __name__ == '__main__':
    sys.exit(main())
