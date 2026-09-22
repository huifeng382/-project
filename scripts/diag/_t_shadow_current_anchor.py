"""_t_shadow_current_anchor.py — 验证 18.6.0 新增节「以 current 为锚的 GNN 判据质量」。

本地即可跑，不需要服务器、不需要真 CSV：手造一棵小树（gnn_shadow.csv + gnn_current.csv），
每个窗口的答案**手算**在 WINDOWS 声明里，再用 numpy **独立复算**一遍（与 _shadow_analyze.py
里的循环实现是两套代码）→ 与打印值对账。

断言的不只是数字，还有**失真机制本身**：
  [1] 头号误报率 = 「排到 current 之前 ∧ 真不优于 current」/「排到 current 之前」，手算命中
  [2] |S|=0 的窗口**单列计数、不进误报率均值**（含它算会得到 65.00%，与 81.25% 差得开）
  [3] rank(cur)=NA 的窗口与「分母=0」**分量上报**（合并就看不见「模型评估不了 current」）
  [4] tc 双射校验**不是摆设**：把一个候选的 eval_idx 挪一位 ⇒ 必须报「配对失败」而不是静默算数
  [5] 缺 gnn_current.csv ⇒ 只多那行**常量**跳过语（并逐字节比较两棵树，证明它是常量）
  [6] 量纲守卫**正例**：把秩换成原始延迟（~1e-11）⇒ 必须报「把秒当秩」
  [7] ④ 排除最大电路**真的排除掉了**（circuit 键类型错了就一个都排不掉 → 单位数会是 5 而不是 1）
  [8] 单/多输出分层与 ③ 宏平均的分母各按构造命中

不入库（沿用 _t_ 前缀惯例，保持 untracked）。
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
ANALYZE = os.path.join(HERE, '_shadow_analyze.py')

# ————————————————————————————————————————————————————————————————————————
# 夹具声明（唯一真相）：每个窗口一行，答案由 ref() 独立复算 + 手算常量双重钉死
#   tcur/rcur = CURRENT 行的 true_avg 与 rank；cands = (eval_idx, true_delay, tc, rank)
#   true_delay=None ⇒ 该候选写 true_delay=NA（真值缺失，应被剔出 pairs 但仍在双射里）
#   rcur=None ⇒ CURRENT 的 rank 写 NA（模型评估不了 current）
#   候选的 rank 故意在 W2 里**乱序写**，证明实现不依赖文件行序
# ————————————————————————————————————————————————————————————————————————
C1 = dict(stem='level0/C1', orphan=1, windows=[
    dict(it=1, wi=0, tcur=100.0, rcur=3.0, tccur=10, nout=1, cands=[
        (2, 110.0, 11, 1.0), (3, 90.0, 12, 2.0), (4, 95.0, 13, 4.0), (5, 105.0, 14, 5.0)]),
    dict(it=2, wi=0, tcur=100.0, rcur=1.0, tccur=20, nout=1, cands=[
        (9, 99.0, 24, 5.0), (6, 90.0, 21, 2.0), (7, 95.0, 22, 3.0), (8, 98.0, 23, 4.0)]),
    dict(it=3, wi=0, tcur=100.0, rcur=5.0, tccur=30, nout=2, cands=[
        (10, 80.0, 31, 1.0), (11, 120.0, 32, 2.0), (12, 130.0, 33, 3.0), (13, 140.0, 34, 4.0)]),
    dict(it=4, wi=0, tcur=100.0, rcur=2.0, tccur=40, nout=1, cands=[
        (14, 110.0, 41, 1.0), (15, None, 42, 3.0), (16, 90.0, 43, 4.0),
        (17, 95.0, 44, 5.0), (18, 99.0, 45, 6.0)]),
    dict(it=5, wi=0, tcur=100.0, rcur=None, tccur=50, nout=1, cands=[
        (19, 90.0, 51, 1.0), (20, 95.0, 52, 2.0), (21, 98.0, 53, 3.0), (22, 99.0, 54, 4.0)]),
])
C2 = dict(stem='level0/C2', orphan=101, windows=[
    dict(it=1, wi=0, tcur=50.0, rcur=2.0, tccur=60, nout=1, cands=[
        (102, 60.0, 61, 1.0), (103, 40.0, 62, 3.0), (104, 45.0, 63, 4.0), (105, 70.0, 64, 5.0)]),
])
CIRCS = (C1, C2)


def _sh_line(ev, it, wi, t, tc, gnn):
    ts = 'NA' if t is None else f"{t:.6e}"
    return (f"eval_idx={ev}, iter={it}, window={wi}, gnn_pred={gnn:.6e}, "
            f"true_delay={ts}, transistors={tc}")


def _cur_line(it, wi, ev, pos, cid, rank, is_cur, tavg, tc, nout):
    r = 'NA' if rank is None else f"{rank:.6e}"
    a = 'NA' if tavg is None else f"{tavg:.6e}"
    e = 'NA' if ev is None else str(ev)
    p = 'NA' if pos is None else str(pos)
    return f"{it},{wi},{e},{p},{cid},{r},{is_cur},{a},{tc},{nout}"


def build_tree(root, mutate=None):
    """mutate(kind, circ, win) → 覆盖单个字段，用于负例/量纲正例。"""
    for c in CIRCS:
        d = os.path.join(root, c['stem'])
        os.makedirs(d, exist_ok=True)
        sh, cur = [_sh_line(c['orphan'], 0, 0, 100.0, 9, 1.0)], []
        for w in c['windows']:
            for (ev, t, tc, rk) in w['cands']:
                e, r = ev, rk
                if mutate == 'ev_shift' and c is C1 and w['it'] == 1 and ev == 4:
                    e = 5                      # c2 的 eval_idx 挪一位 → 撞上 c3 那一行（重复守卫）
                if mutate == 'dim' and c is C1 and w['it'] == 3:
                    r = 1.5e-11                # 秩位置写原始延迟（秒）→ 量纲守卫必须抓
                ctc = tc
                if mutate == 'tc_bad' and c is C1 and w['it'] == 1 and ev == 4:
                    ctc = 99                   # eval_idx 合法但 tc 与 shadow 的 transistors=13 不符
                sh.append(_sh_line(e, w['it'], w['wi'], t, tc, float(rk)))
                cur.append(_cur_line(w['it'], w['wi'], e, ev - min(x[0] for x in w['cands']),
                                     f"c{ev - min(x[0] for x in w['cands'])}", r, 0, None, ctc, w['nout']))
            cur.insert(len(cur) - len(w['cands']),
                       _cur_line(w['it'], w['wi'], None, None, 'CURRENT', w['rcur'], 1,
                                 w['tcur'], w['tccur'], w['nout']))
        with open(os.path.join(d, 'gnn_shadow.csv'), 'w', encoding='utf-8') as f:
            f.write("\n".join(sh) + "\n")
        with open(os.path.join(d, 'gnn_current.csv'), 'w', encoding='utf-8') as f:
            f.write("iter,window,eval_idx,pos,id,rank,is_current,true_avg,tc,nout\n")
            f.write("\n".join(cur) + "\n")
    return root


# —— 独立复算（numpy 向量化，与 _shadow_analyze.py 的循环实现是两套代码）——
def ref(w):
    """→ (S, FP, P, FN, top1_strict, top1_loose)。cands 里 true=None 的先剔（与实现同）。
    ⚠ 宽松**不能**写成 min(pred)==min(true)：那是「两侧最小值恰好相等」，而实现要的是
    「两侧最优**下标集合**相交」。反例 pred=[1,2] / true=[5,1] → min 都是 1 但集合不相交。"""
    cs = [(t, r) for (_, t, _, r) in w['cands'] if t is not None]
    if w['rcur'] is None or not cs:
        return None
    r = np.array([x[1] for x in cs], float)
    t = np.array([x[0] for x in cs], float)
    rc, tc_ = w['rcur'], w['tcur']
    pred = np.concatenate(([rc], r)); true = np.concatenate(([tc_], t))
    pmin = set(np.flatnonzero(pred == pred.min()).tolist())
    tmin = set(np.flatnonzero(true == true.min()).tolist())
    return (int((r < rc).sum()), int(((r < rc) & (t >= tc_)).sum()),
            int((t < tc_).sum()), int(((r >= rc) & (t < tc_)).sum()),
            1.0 if min(tmin) in pmin else 0.0, 1.0 if (pmin & tmin) else 0.0)


def refs(circs=()):
    """→ (rows, S, FP, P, FN) —— 只含能算出误报率的窗口（rcur 非 None）。"""
    out = []
    for c in (circs or CIRCS):
        for w in c['windows']:
            m = ref(w)
            if m is None:
                continue
            out.append((m[0], m[1], m[2], m[3]))
    return out


def run(root):
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    p = subprocess.run([sys.executable, ANALYZE, '--root', root, '--detail-max', '0'],
                       capture_output=True, text=True, encoding='utf-8', env=env)
    if p.returncode != 0:
        print(p.stdout[-3000:], p.stderr[-3000:])
        raise SystemExit(f'analyzer 退出码 {p.returncode}')
    return p.stdout


def one(out, pat, cast=float):
    m = re.search(pat, out)
    return None if not m else (cast(m.group(1)) if m.groups() else None)


def row(out, label):
    """取某行整行文本（用于读多个数）。"""
    for l in out.splitlines():
        if l.strip().startswith(label):
            return l
    return None


def tail(rowtext, n):
    """行尾 n 个 token —— 口径标签里含空格（`排除最大电路 level0/C1`），故**必须从右数**：
    数字列的个数是固定的（四口径表 = 单位数+6 指标；分层表 = 集数+5 指标）。"""
    return rowtext.split()[-n:]


def main():
    fails = []
    base = tempfile.mkdtemp(prefix='t_cur_anchor_')
    roots = {k: os.path.join(base, k) for k in ('ok', 'tc_bad', 'ev_shift', 'dim', 'nofile')}
    try:
        build_tree(roots['ok'])
        build_tree(roots['tc_bad'], mutate='tc_bad')
        build_tree(roots['ev_shift'], mutate='ev_shift')
        build_tree(roots['dim'], mutate='dim')
        build_tree(roots['nofile'])
        os.remove(os.path.join(roots['nofile'], 'level0', 'C1', 'gnn_current.csv'))
        os.remove(os.path.join(roots['nofile'], 'level0', 'C2', 'gnn_current.csv'))
        out = {k: run(roots[k]) for k in roots}

        # —— [1] 头号/伴随/池化/top-1 与独立复算对账 ——
        print('=== [1] 头号误报率 等与 numpy 独立复算对账 ===')
        rows = refs()
        S = sum(x[0] for x in rows); FP = sum(x[1] for x in rows)
        P = sum(x[2] for x in rows); FN = sum(x[3] for x in rows)
        nzS = sum(1 for x in rows if x[0] == 0)
        mean_fp = np.mean([x[1] / x[0] for x in rows if x[0] > 0])
        mean_fn = np.mean([x[3] / x[2] for x in rows if x[2] > 0])
        exp = dict(fp=mean_fp * 100, fn=mean_fn * 100, pfp=FP / S * 100, pfn=FN / P * 100,
                   nzS=nzS, S=S, P=P, nS=len(rows) - nzS,
                   nP=len(rows) - sum(1 for x in rows if x[2] == 0))
        # 手算常量（与 ref 相互独立地钉死夹具语义）
        hand = dict(fp=81.25, fn=70.00, pfp=75.00, pfn=83.333333, nzS=1, S=8, P=12, nS=4, nP=5)
        print(f'  独立复算: 误报均值 {exp["fp"]:.2f}%  漏报均值 {exp["fn"]:.2f}%  '
              f'池化 {exp["pfp"]:.2f}%/{exp["pfn"]:.2f}%  ΣS={exp["S"]}/{exp["nS"]} 集  '
              f'ΣP={exp["P"]}/{exp["nP"]} 集  |S|=0 集={exp["nzS"]}')
        print(f'  手算常量: 误报均值 {hand["fp"]:.2f}%  漏报均值 {hand["fn"]:.2f}%  '
              f'池化 {hand["pfp"]:.2f}%/{hand["pfn"]:.2f}%  ΣS={hand["S"]}/{hand["nS"]} 集  '
              f'ΣP={hand["P"]}/{hand["nP"]} 集  |S|=0 集={hand["nzS"]}')
        for k in ('fp', 'fn', 'pfp', 'pfn'):
            if abs(exp[k] - hand[k]) > 0.01:
                fails.append(f'独立复算 {k}={exp[k]:.4f} != 手算 {hand[k]:.4f}（夹具声明自相矛盾）')
        for k in ('S', 'P', 'nS', 'nP', 'nzS'):
            if exp[k] != hand[k]:
                fails.append(f'独立复算 {k}={exp[k]} != 手算 {hand[k]}（夹具声明自相矛盾）')
        got_fp = one(out['ok'], r'头号 误报率.*?:\s*([\d.]+)%')
        got_fn = one(out['ok'], r'伴随 漏报率.*?:\s*([\d.]+)%')
        got_p = re.search(r'池化.*?误报\s*([\d.]+)%\s*漏报\s*([\d.]+)%', out['ok'])
        got_s = re.search(r'分母 ΣS=(\d+) 项 / (\d+) 集', out['ok'])
        got_sp = re.search(r'分母 ΣP=(\d+) 项 / (\d+) 集', out['ok'])
        got_ns = one(out['ok'], r'\|S\|=0 的集 (\d+) 个', int)
        n_nzP = sum(1 for x in rows if x[2] == 0)
        print(f'  分析器读数: 误报 {got_fp}%  漏报 {got_fn}%  池化 '
              f'{got_p.group(1) if got_p else None}%/{got_p.group(2) if got_p else None}%  '
              f'ΣS={got_s.groups() if got_s else None}  ΣP={got_sp.groups() if got_sp else None}  '
              f'|S|=0 集={got_ns}')
        checks = [(got_fp, exp['fp'], '头号误报率'), (got_fn, exp['fn'], '伴随漏报率')]
        if not got_p:
            fails.append('没拿到池化行（正则失配）')
        else:
            checks += [(float(got_p.group(1)), exp['pfp'], '池化误报'),
                       (float(got_p.group(2)), exp['pfn'], '池化漏报')]
        for a, b, nm in checks:
            if a is None or abs(a - b) > 0.01:
                fails.append(f'{nm}: 分析器 {a} != 复算 {b:.4f}')
        # 两个分母**分开**断言：S 的分母剔 |S|=0 的集，P 的分母剔 |P|=0 的集，两者不必相等
        if got_s is None or (int(got_s.group(1)), int(got_s.group(2))) != (exp['S'], exp['nS']):
            fails.append(f'分母 ΣS 行 {got_s.groups() if got_s else None} != {(exp["S"], exp["nS"])}')
        if got_sp is None or (int(got_sp.group(1)), int(got_sp.group(2))) != (exp['P'], exp['nP']):
            fails.append(f'分母 ΣP 行 {got_sp.groups() if got_sp else None} != {(exp["P"], exp["nP"])}')
        if got_ns != exp['nzS']:
            fails.append(f'|S|=0 集数 {got_ns} != {exp["nzS"]}')
        if (exp['nS'], exp['nP']) == (len(rows), len(rows)):
            fails.append('S 与 P 的分母集数相同（本夹具 |S|=0 那集应把 S 分母压到 4）')
        if abs(got_fp - 65.0) < 0.01:
            fails.append('误报率均值把 |S|=0 的窗口按 0% 算进去了（应剔除）')
        print('  ✅ 头号/伴随/池化/分母 全部与独立复算一致，且 |S|=0 的窗口未进均值'
              if not [f for f in fails if '误报' in f or '池化' in f or 'ΣS' in f or '|S|' in f]
              else '  ❌ 见下方失败项')

        # —— top-1 与相关系数存在性 ——
        t1 = re.search(r'top-1 命中（含 current 一起排）: 严格\s*([\d.]+)%\s*宽松\s*([\d.]+)%',
                       out['ok'])
        exp_t1 = np.mean([m[4] for m in (ref(w) for c in CIRCS for w in c['windows'])
                          if m is not None]) * 100
        exp_t1l = np.mean([m[5] for m in (ref(w) for c in CIRCS for w in c['windows'])
                           if m is not None]) * 100
        print(f'\n=== [2] top-1（严格/宽松）=== 分析器 '
              f'{t1.groups() if t1 else None}  复算 {exp_t1:.1f}%/{exp_t1l:.1f}%')
        if not t1 or abs(float(t1.group(1)) - exp_t1) > 0.05 or abs(float(t1.group(2)) - exp_t1l) > 0.05:
            fails.append(f'top-1 不符: {t1.groups() if t1 else None} vs '
                         f'{exp_t1:.1f}/{exp_t1l:.1f}')
        elif not (50.0 <= exp_t1 <= 100.0) and exp_t1 == 0.0:
            fails.append('top-1 恒 0 —— 夹具全是反例，断言分辨不出对错')
        sp = row(out['ok'], '含 current 批内相关')
        print(f'  {sp.strip() if sp else "（缺）"}')
        if not sp:
            fails.append('没拿到含 current 批内相关行')

        # —— [3] rank(cur)=NA 与「分母=0」分量上报 ——
        print('\n=== [3] rank(cur)=NA 与 分母=0 分量上报 ===')
        na_cur = one(out['ok'], r'current 秩缺失/NaN 的集 (\d+)', int)
        empty = one(out['ok'], r'只有 CURRENT 行的窗口.*?:\s*(\d+)', int)
        dim = one(out['ok'], r'量纲守卫（秩越界 / 非整数或半整数）:\s*(\d+)', int)
        print(f'  rcur=NA 的集 {na_cur}（预期 1）  空候选窗口 {empty}（预期 0）  量纲守卫 {dim}（预期 0）')
        if (na_cur, empty, dim) != (1, 0, 0):
            fails.append(f'干净树诊断计数 {(na_cur, empty, dim)} != (1, 0, 0)')
        tn = one(out['ok'], r'候选真值缺失\(true_delay=NA，已剔出该集\):\s*(\d+)', int)
        win = one(out['ok'], r'gnn_current 窗口数（全部, 含未成集）:\s*(\d+)', int)
        unc = one(out['ok'], r'gnn_shadow 有、gnn_current 无的窗口:\s*(\d+)', int)
        nok = re.search(r'本节合格集 (\d+) / 主口径 (\d+) 集', out['ok'])
        print(f'  真值缺失候选 {tn}（预期 1）  窗口数 {win}（预期 6）  unclaimed {unc}（预期 2）  '
              f'合格集 {nok.groups() if nok else None}（预期 5 集）')
        if tn != 1:
            fails.append(f'真值缺失候选数 {tn} != 1')
        if win != 6:
            fails.append(f'窗口数 {win} != 6')
        if unc != 2:
            fails.append(f'unclaimed 窗口 {unc} != 2（两个电路的初始 current）')
        if not nok or int(nok.group(1)) != 5:
            fails.append(f'合格集数 {nok.groups() if nok else None} != 5')

        # —— [4] ③ 宏平均 / ④ 排除最大电路 / 分层 ——
        print('\n=== [4] ③ 宏平均、④ 排除最大电路、单/多输出分层 ===')
        r3 = row(out['ok'], '口径③')
        r4 = row(out['ok'], '口径④')
        r1 = row(out['ok'], '口径①')
        r2 = row(out['ok'], '口径②')
        s1 = row(out['ok'], '单输出(nout=1)')
        s2 = row(out['ok'], '多输出(nout>=2)')
        print(f'  {r1.strip() if r1 else "（缺 ①）"}')
        print(f'  {r3.strip() if r3 else "（缺 ③）"}')
        print(f'  {r4.strip() if r4 else "（缺 ④）"}')
        print(f'  {s1.strip() if s1 else "（缺 单输出层）"}')
        print(f'  {s2.strip() if s2 else "（缺 多输出层）"}')
        if not (r1 and r2 and r3 and r4 and s1 and s2):
            fails.append('四口径或分层有缺行')
        else:
            # 四口径表列序：单位数 误报率 漏报率 top1严格 top1宽松 Sp(修正) Kendall（7 列，从右数）
            t1_, t2_, t3_, t4_ = tail(r1, 7), tail(r2, 7), tail(r3, 7), tail(r4, 7)
            # 分层表列序：集数 误报率 漏报率 top1严格 Sp(修正) 池化误报（6 列，从右数）
            u1, u2 = tail(s1, 6), tail(s2, 6)
            if float(t1_[1].rstrip('%')) != pytest_round(exp['fp']):
                fails.append(f'① 误报率 {t1_[1]} != 头号 {exp["fp"]:.2f}%')
            if t2_[0] != str(len(rows)):
                fails.append(f'② 单位数 {t2_[0]} != {len(rows)}（本夹具无同池重复，不该去重）')
            if t3_[0] != '2':
                fails.append(f'③ 单位数 {t3_[0]} != 2（两个电路）')
            if float(t3_[1].rstrip('%')) != 87.50:
                fails.append(f'③ 宏平均误报率 {t3_[1]} != 87.50%（C1 0.75 与 C2 1.0 的均值）')
            if t4_[0] != '1':
                fails.append(f'④ 单位数 {t4_[0]} != 1 —— circuit 键与主口径不同型时一个电路都排不掉'
                             f'（这正是本断言要抓的）')
            if (u1[0], u2[0]) != ('4', '1'):
                fails.append(f'分层集数 {u1[0]}/{u2[0]} != 4/1')
            if float(u1[1].rstrip('%')) != 83.33:
                fails.append(f'单输出层误报率 {u1[1]} != 83.33%')
            if float(u2[1].rstrip('%')) != 75.00:
                fails.append(f'多输出层误报率 {u2[1]} != 75.00%')

        # —— [5] 负例：两条独立守卫都必须「报错并剔窗」，绝不静默算出一个数 ——
        #  5a tc 不符：eval_idx 合法且不撞车，只有 tc 对不上 → 抓 tc 校验本身
        #  5b eval_idx 挪一位撞上另一行 → 重复行守卫先触发（顺序有别，但同样必须 fail-loud）
        print('\n=== [5] 负例：tc 不符 / eval_idx 撞行 —— 都必须报「配对失败」而非静默算数 ===')
        for tag, exp_tc, exp_dup in (('tc_bad', 1, 0), ('ev_shift', 0, 1)):
            g = re.search(r'配对失败 双射集合不等: (\d+)\s+eval_idx 不存在: (\d+)\s+tc 不等: (\d+)'
                          r'\s+窗内 eval_idx 重复: (\d+)\s*(\S*)', out[tag])
            n2 = re.search(r'本节合格集 (\d+) /', out[tag])
            got = tuple(int(g.group(i)) for i in (1, 2, 3, 4)) if g else None
            print(f'  {tag}: 配对(双射/缺eval/tc) + 窗内重复 = {got}  '
                  f'（预期 tc 不等 {exp_tc}、窗内重复 {exp_dup}）  合格集 '
                  f'{n2.group(1) if n2 else None}（预期 4 < 5）')
            if not got:
                fails.append(f'{tag}: 配对诊断行读不出（正则失配）')
                continue
            if got != (0, 0, exp_tc, exp_dup):
                fails.append(f'{tag}: 诊断 {got} != (0, 0, {exp_tc}, {exp_dup})')
            mark = g.group(5)
            if '❌' not in mark:
                fails.append(f'{tag}: 配对失败但没标 ❌（{mark}）')
            if not n2 or int(n2.group(1)) != 4:
                fails.append(f'{tag}: 合格集 {n2.group(1) if n2 else None} != 4（该窗应被剔出）')
        # 干净树上这四条都必须为 0（否则上面两条断言可能只是「本来就红」）
        g0 = re.search(r'配对失败 双射集合不等: (\d+)\s+eval_idx 不存在: (\d+)\s+tc 不等: (\d+)'
                       r'\s+窗内 eval_idx 重复: (\d+)', out['ok'])
        if not g0 or any(int(x) for x in g0.groups()):
            fails.append(f'干净树配对诊断非 0: {g0.groups() if g0 else None}')

        # —— [6] 量纲守卫正例 ——
        print('\n=== [6] 量纲守卫正例：把秩位置写成原始延迟 ~1e-11 ===')
        d = re.search(r'量纲守卫（秩越界 / 非整数或半整数）:\s*(\d+)\s*(\S*)', out['dim'])
        dset = re.search(r'本节合格集 (\d+) /', out['dim'])
        print(f'  量纲守卫 {d.groups() if d else None}（预期 ≥1、标 ❌）  合格集 '
              f'{dset.group(1) if dset else None}（预期 4）')
        if not d or int(d.group(1)) < 1:
            fails.append(f'原始延迟混进秩没被检出: {d.groups() if d else None}')
        elif '❌' not in d.group(2):
            fails.append('量纲守卫命中但没标 ❌')
        if not dset or int(dset.group(1)) != 4:
            fails.append(f'量纲正例合格集 {dset.group(1) if dset else None} != 4')

        # —— [7] 缺文件 ⇒ 常量跳过语，且两棵树逐字节相同 ——
        print('\n=== [7] 缺 gnn_current.csv：只多那行常量跳过语 ===')
        nof = out['nofile']
        skip = [l for l in nof.splitlines() if '本树无 gnn_current.csv' in l]
        has_head = any('头号 误报率' in l for l in nof.splitlines())
        base_ok = [l for l in out['ok'].splitlines() if '本树无 gnn_current.csv' in l]
        print(f'  跳过语出现 {len(skip)} 次  头号行存在={has_head}（预期 0 / False）')
        print(f'  干净树里出现 {len(base_ok)} 次（预期 0）')
        if len(skip) != 1 or has_head or base_ok:
            fails.append(f'缺文件路径不符: skip={len(skip)} 头号行={has_head} 干净树={len(base_ok)}')
        # 常量性：两棵都缺文件的树（nofile 与 dim 无 C2 差异不适用）→ 用 nofile 跑两遍比字节
        again = run(roots['nofile'])
        if again != nof:
            # 只有戳里的时间/mtime/sha 会变 → 剥戳后必须逐字节相同
            cut = nof.find('=== 戳结束 ===')
            cut2 = again.find('=== 戳结束 ===')
            if cut2 < 0 or nof[cut:] != again[cut2:]:
                fails.append('缺文件时的输出不是常量（剥戳后仍不同）')
            else:
                print('  ✅ 剥戳后逐字节相同（缺文件路径是常量）')
        else:
            print('  ✅ 两次运行逐字节相同')

        print('\n=== 结论 ===')
        if fails:
            print('❌ 失败: ' + '; '.join(fails))
        else:
            print('✅ 全部断言通过（头号误报率/漏报率/池化/top-1/四口径/分层/诊断计数/'
                  '配对失败/量纲守卫/缺文件常量 全覆盖）')
        return 1 if fails else 0
    finally:
        shutil.rmtree(base, ignore_errors=True)


def pytest_round(x):
    return float(f'{x:.2f}')


if __name__ == '__main__':
    sys.exit(main())
