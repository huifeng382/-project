"""_t_shadow_current_anchor.py — 验证 18.6.0 新增、18.6.1（配对改前缀口径 + ★首推真负率）的节"
「以 current 为锚的 GNN 判据质量」。

本地即可跑，不需要服务器、不需要真 CSV：手造一棵小树（gnn_shadow.csv + gnn_current.csv），
每个窗口的答案**手算**在 WINDOWS 声明里，再用 numpy **独立复算**一遍（与 _shadow_analyze.py
里的循环实现是两套代码）→ 与打印值对账。

断言的不只是数字，还有**失真机制本身**：
  [1] 头号误报率 = 「排到 current 之前 ∧ 真不优于 current」/「排到 current 之前」，手算命中
      （同时断言 |S|=0 的窗口单列计数、不进均值；含它算会得到 65.00%，与 81.25% 差得开）
      18.6.1 另加 **★核心 首推真负率**（只看 GNN 排第一的那个候选是否真负）：用 ref_core()
      独立复算严格/宽松/分母/池化/镜像/并列，并断言**缺口恒等式**
      （首推 current + 首推无真值 + 进分母 = 合格集）—— 这条能抓住「镜像写成 S==0」这类
      分桶重叠的错（见 [9] topund）
  [2] top-1 严格/宽松与 numpy 独立复算一致；[3] rank(cur)=NA 与「分母=0」**分量上报**
      （合并就看不见「模型评估不了 current」）；[4] 四口径 + ③ 宏平均 + ④ 排除最大电路真的排掉了
      （circuit 键类型错了就一个都排不掉 → 单位数会是 5 而不是 1）+ 单/多输出分层
  [5] 配对失败**不是摆设**：tc 不符、eval_idx 撞行、推断位置有洞（非前缀）三种都必须报「配对失败」
      并标 ❌、剔窗，而不是静默算出一个数
  [6] 量纲守卫**正例**：把秩换成原始延迟（~1e-11）⇒ 必须报「把秒当秩」
  [7] 18.6.1 修正：**后缀**缺 eval_idx（= 贪心在该窗提前 accept 并 break，tl_opt.rs:1191）是
      **预期结构**，必须照常成集、不报 ❌、且那行不入分子分母（头号/池化/分母与干净树逐字相同）；
      同一棵树上把缺的位置换成「中间有洞」⇒ 必须回到 ❌
  [8] 缺 gnn_current.csv ⇒ 只多那行**常量**跳过语（并逐字节比较两棵树，证明它是常量）
  [9] 核心口径的四个边界：非首推的未评估行（rank 3.5 > rcur 3.0）不得改变任何核心读数
      （与干净树逐字相同）；并列组内真假不一（C1 it=3 的 rank 并列 1.0，t=80 真负 / t=120
      非负）⇒ 严格与宽松都判错、并列 1 / 混 1；首推落在未评估后缀（C1 it=4 插入 rank 1.0 的
      无真值行，同时把 c0 降成 2.0 与 current 并列）⇒ 该窗让出镜像桶、剔出分母，且**该窗 S
      同时为 0** —— 这一格专抓「镜像写成 S==0」的分桶重叠（那样缺口三项会多算一个窗）；
      与真首推并列的无真值行（uneval）⇒ 分母 4→3。⚠ 插入行的 rank 不能取 0.5：竞争秩的最小
      值是 1.0，0.5 会被本节自带的量纲守卫当成「把秒当秩」而整窗剔掉（试过，见 [6]）

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

# 额外候选行（**只在 gnn_current 侧**插；shadow 无此行 ⇒ 推断位置落在未评估后缀）：
#   extra_uneval → W1(rcur=3.0) rank 1.0 = 与真首推**并列** ⇒ 首推组含无真值项 ⇒「首推无真值」桶 +1
#   nontop       → W1 rank 3.5（> rcur）⇒ **不是首推**，核心口径必须与干净树逐字相同
#   topund       → W4 rank 1.0（< rcur 2.0）⇒ 首推无真值；且该窗的 c0 被降成 2.0（与 current 并列）
#                  ⇒ S 归 0。这正是「镜像写成 S==0」会重复计数的那一格（缺口恒等式当场破）
EXTRA = {'extra_uneval': (1, 1.0), 'nontop': (1, 3.5), 'topund': (4, 1.0)}


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
                e, r, ce = ev, rk, ev
                if mutate == 'ev_shift' and c is C1 and w['it'] == 1 and ev == 4:
                    e = 5; ce = 5              # c2 的 eval_idx 挪一位 → 撞上 c3 那一行（重复守卫）
                if mutate == 'hole' and c is C1 and w['it'] == 1 and ev == 3:
                    ce = 999                   # **只改 gnn_current 侧** ⇒ 推断位置落空且非前缀（真错位）
                if mutate == 'dim' and c is C1 and w['it'] == 3:
                    r = 1.5e-11                # 秩位置写原始延迟（秒）→ 量纲守卫必须抓
                ctc = tc
                if mutate == 'tc_bad' and c is C1 and w['it'] == 1 and ev == 4:
                    ctc = 99                   # eval_idx 合法但 tc 与 shadow 的 transistors=13 不符
                if mutate == 'topund' and c is C1 and w['it'] == 4 and ev == 14:
                    r = 2.0                    # 与 current(2.0) 并列 ⇒ 不再 < rcur ⇒ 该窗 S 归 0；
                    #                            配合下面插入的无真值首推行，正好造出「首推无真值 ∧ S=0」
                if mutate == 'tie' and c is C1 and w['it'] == 3 and ev == 11:
                    r = 1.0                    # 与首推 c0(rank 1.0, t=80) **并列** ⇒ 首推变成一组、
                    #                            组内真假不一（80 真负 / 120 非负）
                sh.append(_sh_line(e, w['it'], w['wi'], t, tc, float(rk)))
                cur.append(_cur_line(w['it'], w['wi'], ce, ev - min(x[0] for x in w['cands']),
                                     f"c{ev - min(x[0] for x in w['cands'])}", r, 0, None, ctc, w['nout']))
            ins = len(cur) - len(w['cands'])
            ex = EXTRA.get(mutate)
            if ex and c is C1 and w['it'] == ex[0]:
                # 结构性未评估：gnn_current 多一行候选（pos=4、eval 写 999 = shadow 里不存在），
                # 这正是「贪心在该窗提前 accept 并 break」的产物（tl_opt.rs:1191）—— 本文件的候选行
                # 出自 mods(=prepared 全表)，而 shadow 只有它的**前缀** ⇒ 必须照常成集、不报 ❌
                cur.insert(ins, _cur_line(w['it'], w['wi'], 999, 4, 'c4', ex[1], 0, None, 20, w['nout']))
                ins += 1
            cur.insert(ins,
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


def ref_core(w):
    """独立复算**核心口径**（首推真负率）→ dict；与 _shadow_analyze.py 的窗口循环是两套代码。
    ⚠ 与 ref() 的关键差别：这里用**全部候选行**（含真值 NA 的）算「谁排第一」—— 实现里 rk_c 就是
    全部候选行的秩，真值只在取 t1 时才查 ⇒ 首推完全可能是个**没有真值**的候选（未评估后缀）。
    top=False ⇒ GNN 首推 current；win=0.0 且 top=True ⇒ 首推了候选但它无真值（判不了、剔出分母）。"""
    if w['rcur'] is None:
        return None
    rk = [r for (_, _, _, r) in w['cands'] if r is not None]
    if not rk or min(rk) >= w['rcur']:
        return dict(top=False, win=0.0, neg=None, nn=None, tie=0, mix=0)
    tt = [t for (_, t, _, r) in w['cands'] if r is not None and r == min(rk)]
    if any(t is None for t in tt):
        return dict(top=True, win=0.0, neg=None, nn=None, tie=0, mix=0)
    return dict(top=True, win=1.0,
                neg=1.0 if all(t < w['tcur'] for t in tt) else 0.0,
                nn=1.0 if all(t <= w['tcur'] for t in tt) else 0.0,
                tie=1 if len(tt) > 1 else 0,
                mix=1 if (len(tt) > 1 and len(set(t < w['tcur'] for t in tt)) > 1) else 0)


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
    roots = {k: os.path.join(base, k)
             for k in ('ok', 'tc_bad', 'ev_shift', 'dim', 'nofile', 'uneval', 'hole',
                       'nontop', 'tie', 'topund')}
    try:
        build_tree(roots['ok'])
        build_tree(roots['tc_bad'], mutate='tc_bad')
        build_tree(roots['ev_shift'], mutate='ev_shift')
        build_tree(roots['dim'], mutate='dim')
        build_tree(roots['nofile'])
        build_tree(roots['uneval'], mutate='extra_uneval')
        build_tree(roots['hole'], mutate='hole')
        build_tree(roots['nontop'], mutate='nontop')
        build_tree(roots['tie'], mutate='tie')
        build_tree(roots['topund'], mutate='topund')
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

        # —— ★ 核心口径（首推真负率，18.6.1）与独立复算对账 ——
        #   三个桶必须互斥且穷尽：首推 current / 首推了候选但无真值 / 进分母
        core = [x for x in (ref_core(w) for c in CIRCS for w in c['windows']) if x is not None]
        c_win = [x for x in core if x['win'] == 1.0]
        c_cur = sum(1 for x in core if not x['top'])
        c_nod = sum(1 for x in core if x['top'] and x['win'] == 0.0)
        c_mis = 0
        for c in CIRCS:
            for w in c['windows']:
                if w['rcur'] is None:
                    continue
                rk = [r for (_, _, _, r) in w['cands'] if r is not None]
                if rk and min(rk) < w['rcur']:
                    continue                      # 首推了候选 ⇒ 不在镜像分母里
                if any(t is not None and t < w['tcur'] for (_, t, _, _) in w['cands']):
                    c_mis += 1
        e_neg = np.mean([x['neg'] for x in c_win]) * 100
        e_nn = np.mean([x['nn'] for x in c_win]) * 100
        e_tie = sum(x['tie'] for x in core); e_mix = sum(x['mix'] for x in core)
        g1 = re.search(r'严格（t₁ < t_cur[^:]*:\s*([\d.]+)%\s+宽松（t₁ ≤ t_cur[^:]*:\s*([\d.]+)%',
                       out['ok'])
        g2 = re.search(r'且首推有真值\*\*的窗 (\d+) 个（占合格集 (\d+) 的 [\d.]+%）；'
                       r'池化 Σ真负/Σ窗 = ([\d.]+)% / 宽松 ([\d.]+)%', out['ok'])
        g3 = re.search(r'首推 current 的窗 (\d+) 个 ＋ 首推候选\*\*无真值\*\*的窗 (\d+) 个'
                       r'.*?＋ 进分母 (\d+) 个', out['ok'])
        # 盘化：`\S+` 会把后面的「（下界：…）」整段吞掉（同一行无空格）⇒ 只取数字或 nan
        g4 = re.search(r'的 (\d+) 个窗里，其实\*\*存在更优候选\*\*的 (\d+) 个 ⇒\s*([\d.]+|nan)%',
                       out['ok'])
        g5 = re.search(r'首推并列的窗 (\d+) 个.*?组内真假不一 (\d+) 个', out['ok'])
        print(f'  ★核心复算: 严格 {e_neg:.2f}% / 宽松 {e_nn:.2f}%  分母 {len(c_win)} 个窗'
              f'（首推current {c_cur} + 无真值 {c_nod} + 进分母 {len(c_win)} = {len(core)}）'
              f'  镜像 {c_mis}/{c_cur}  并列 {e_tie} 混 {e_mix}')
        print(f'  ★核心读数: 严格/宽松 {g1.groups() if g1 else None}  分母/池化 '
              f'{(g2.group(1), g2.group(3), g2.group(4)) if g2 else None}  缺口 '
              f'{g3.groups() if g3 else None}  镜像 {g4.groups() if g4 else None}  并列 '
              f'{g5.groups() if g5 else None}')
        if not g1 or abs(float(g1.group(1)) - e_neg) > 0.01 or abs(float(g1.group(2)) - e_nn) > 0.01:
            fails.append(f'首推率 {g1.groups() if g1 else None} != 复算 {e_neg:.2f}/{e_nn:.2f}')
        if g3 is None or tuple(int(x) for x in g3.groups()) != (c_cur, c_nod, len(c_win)):
            fails.append(f'缺口三桶 {g3.groups() if g3 else None} != {(c_cur, c_nod, len(c_win))}')
        elif sum(int(x) for x in g3.groups()) != len(core):
            fails.append('缺口恒等式不成立（三桶相加 != 合格集）')
        if (g2 is None or int(g2.group(1)) != len(c_win)
                or abs(float(g2.group(3)) - e_neg) > 0.01 or abs(float(g2.group(4)) - e_nn) > 0.01):
            fails.append(f'核心分母/池化 {g2.groups() if g2 else None} != '
                         f'{(len(c_win), e_neg, e_nn)}')
        if g4 is None or (int(g4.group(1)), int(g4.group(2))) != (c_cur, c_mis):
            fails.append(f'镜像 {g4.groups() if g4 else None} != {(c_cur, c_mis)}')
        if g5 is None or tuple(int(x) for x in g5.groups()) != (e_tie, e_mix):
            fails.append(f'并列计数 {g5.groups() if g5 else None} != {(e_tie, e_mix)}')
        if not [f for f in fails if '首推率' in f or '缺口' in f or '核心分母' in f or '镜像' in f
                or '并列计数' in f]:
            print('  ✅ 核心口径（严格/宽松/分母/缺口恒等式/镜像/并列）与独立复算一致')


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
            # 四口径表列序：单位数 ★推严 ★推宽 误报率 漏报率 top1严格 top1宽松 Sp(修正)
            #             Kendall（9 列，从右数）；★ = 核心口径，故下标 [1]/[2]、误报率在 [3]
            t1_, t2_, t3_, t4_ = tail(r1, 9), tail(r2, 9), tail(r3, 9), tail(r4, 9)
            # 分层表列序：集数 ★推严 ★推宽 误报率 漏报率 top1严格 Sp(修正) 池化误报
            #            （8 列，从右数）
            u1, u2 = tail(s1, 8), tail(s2, 8)
            if float(t1_[3].rstrip('%')) != pytest_round(exp['fp']):
                fails.append(f'① 误报率 {t1_[3]} != 头号 {exp["fp"]:.2f}%')
            # ★ 两列 = 核心口径（窗均），必须与 [1] 独立复算 + 手算常量一致
            if float(t1_[1].rstrip('%')) != 25.00 or float(t1_[2].rstrip('%')) != 25.00:
                fails.append(f'① ★推严/推宽 {t1_[1]}/{t1_[2]} != 25.00%/25.00%')
            if t2_[0] != str(len(rows)):
                fails.append(f'② 单位数 {t2_[0]} != {len(rows)}（本夹具无同池重复，不该去重）')
            if t3_[0] != '2':
                fails.append(f'③ 单位数 {t3_[0]} != 2（两个电路）')
            if float(t3_[3].rstrip('%')) != 87.50:
                fails.append(f'③ 宏平均误报率 {t3_[3]} != 87.50%（C1 0.75 与 C2 1.0 的均值）')
            if float(t3_[1].rstrip('%')) != 16.67:
                fails.append(f'③ 宏平均★推严 {t3_[1]} != 16.67%（C1 1/3 与 C2 0 的均值）')
            if t4_[0] != '1':
                fails.append(f'④ 单位数 {t4_[0]} != 1 —— circuit 键与主口径不同型时一个电路都排不掉'
                             f'（这正是本断言要抓的）')
            if (u1[0], u2[0]) != ('4', '1'):
                fails.append(f'分层集数 {u1[0]}/{u2[0]} != 4/1')
            if float(u1[3].rstrip('%')) != 83.33:
                fails.append(f'单输出层误报率 {u1[3]} != 83.33%')
            if float(u2[3].rstrip('%')) != 75.00:
                fails.append(f'多输出层误报率 {u2[3]} != 75.00%')
            # 分层 ★推严：单输出层 4 个窗里进分母的 3 个全判错（0%）；多输出层只有 1 个窗，
            # 它首推的 t=80 真改进（100%）—— 分层差异正是这条读数的解释入口
            if float(u1[1].rstrip('%')) != 0.00:
                fails.append(f'单输出层★推严 {u1[1]} != 0.00%')
            if float(u2[1].rstrip('%')) != 100.00:
                fails.append(f'多输出层★推严 {u2[1]} != 100.00%')

        # —— [5] 负例：两条独立守卫都必须「报错并剔窗」，绝不静默算出一个数 ——
        #  5a tc 不符：eval_idx 合法且不撞车，只有 tc 对不上 → 抓 tc 校验本身
        #  5b eval_idx 挪一位撞上另一行 → 重复行守卫先触发（顺序有别，但同样必须 fail-loud）
        print('\n=== [5] 负例：tc 不符 / eval_idx 撞行 —— 都必须报「配对失败」而非静默算数 ===')
        for tag, exp_tc, exp_dup in (('tc_bad', 1, 0), ('ev_shift', 0, 1)):
            g = re.search(r'配对失败 双射集合不等: (\d+)\s+推断位置非前缀\(pos 缺失\): (\d+)\s+tc 不等: (\d+)'
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
        g0 = re.search(r'配对失败 双射集合不等: (\d+)\s+推断位置非前缀\(pos 缺失\): (\d+)\s+tc 不等: (\d+)'
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

        # —— [8] 18.6.1 修正：结构性未评估（后缀缺 eval_idx）照常成集；有洞才是真错位 ——
        #   8a extra_uneval：C1 it=1 在 gnn_current 多一行候选（eval_idx=6、rank 1.0 < rcur 3.0），
        #      shadow 无此行 = 贪心在该窗提前 accept 并 break 的产物 ⇒ 不得报 ❌、集数不减、
        #      且**头号/池化/ΣS 必须与干净树逐字相同**（证明这行既不入分子也不入分母）
        #   8b hole：同一窗的 pos=1 在 gnn_current 侧被改成 shadow 里不存在的 eval_idx=999
        #      ⇒ present 位置集 {0,2,3} 非前缀 ⇒ 必须报配对失败、标 ❌、该窗剔出（合格集 5→4）
        print('\n=== [7] 结构性未评估（后缀缺）vs 真错位（有洞）===')
        P4 = (r'配对失败 双射集合不等: (\d+)\s+推断位置非前缀\(pos 缺失\): (\d+)\s+tc 不等: (\d+)'
              r'\s+窗内 eval_idx 重复: (\d+)\s*(\S*)')
        gu = re.search(P4, out['uneval'])
        un = re.search(r'结构性未评估候选.*?:\s*(\d+) 项，其中「秩高于 current」(\d+) 项', out['uneval'])
        nu = re.search(r'本节合格集 (\d+) /', out['uneval'])
        gu4 = tuple(int(gu.group(i)) for i in (1, 2, 3, 4)) if gu else None
        print(f'  uneval: 配对诊断 {gu4}（预期 (0,0,0,0)、✅）  合格集 {nu.group(1) if nu else None}'
              f'（预期 5 = 干净树）  未评估计数 {un.groups() if un else None}（预期 (1, 1)）')
        if gu4 != (0, 0, 0, 0):
            fails.append(f'uneval: 后缀缺 eval_idx 被当成配对失败: {gu4}')
        elif '❌' in gu.group(5):
            fails.append('uneval: 结构性未评估却标了 ❌')
        if not nu or int(nu.group(1)) != 5:
            fails.append(f'uneval: 合格集 {nu.group(1) if nu else None} != 5（该窗不该被剔）')
        if not un or un.groups() != ('1', '1'):
            fails.append(f'uneval: 未评估计数 {un.groups() if un else None} != (1, 1)')
        # 逐字对账：未评估行的 rank 1.0 < rcur 3.0，若被算进 S 则误报率与 ΣS 都会变
        same = []
        for pat in (r'头号 误报率.*?:\s*([\d.]+)%', r'池化.*?误报\s*([\d.]+)%\s*漏报\s*([\d.]+)%',
                    r'分母 ΣS=(\d+) 项 / (\d+) 集', r'分母 ΣP=(\d+) 项 / (\d+) 集'):
            a = re.search(pat, out['ok']); b = re.search(pat, out['uneval'])
            same.append(a is not None and b is not None and a.groups() == b.groups())
        if not all(same):
            fails.append('uneval: 未评估的那行进了统计（头号/池化/分母与干净树不一致）')
        else:
            print('  ✅ 头号/池化/两个分母与干净树逐字相同（该行未入分子分母）、集数不减、无 ❌')
        gh = re.search(P4, out['hole'])
        nh = re.search(r'本节合格集 (\d+) /', out['hole'])
        gh4 = tuple(int(gh.group(i)) for i in (1, 2, 3, 4)) if gh else None
        print(f'  hole:   配对诊断 {gh4}（预期 非前缀≥1）  合格集 {nh.group(1) if nh else None}（预期 4）')
        if not gh4 or (gh4[0] + gh4[1]) < 1:
            fails.append(f'hole: 非前缀的推断位置没被检出: {gh4}')
        elif '❌' not in gh.group(5):
            fails.append('hole: 真错位但没标 ❌')
        if not nh or int(nh.group(1)) != 4:
            fails.append(f'hole: 合格集 {nh.group(1) if nh else None} != 4（该窗应被剔出）')

        # —— [7] 缺文件 ⇒ 常量跳过语，且两棵树逐字节相同 ——
        print('\n=== [8] 缺 gnn_current.csv：只多那行常量跳过语 ===')
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

        # —— [9] 核心口径的三个边界 ——
        #   ok/nontop/tie/topund/uneval 五棵树各钉一个边界：正常 / 非首推的未评估行 / 并列组 /
        #   首推落在未评估后缀（且该窗 S 仍为 0）/ 与真首推并列的无真值行
        print('\n=== [9] 核心口径边界：非首推行 / 并列组真假不一 / 首推落在未评估后缀 ===')
        # 列序：严格% 宽松% (缺口 首推cur,无真值,进分母=核心分母) **镜像分母**(首推current 的窗数)
        #       镜像命中 池化严格% 并列 混 未评估项 秩高于cur
        #   ⚠ 镜像分母**不是**核心分母：前者数「GNN 首推 current」的窗，后者数「首推了候选且有
        #     真值」的窗 —— 两桶互补，恰好就是上面缺口的前两项（本夹具里都是 1 个窗）
        exp9 = {
            'ok':     (25.00, 25.00, (1, 0, 4), 1, 1, 25.00, 0, 0, 0, 0),
            'nontop': (25.00, 25.00, (1, 0, 4), 1, 1, 25.00, 0, 0, 1, 0),
            'tie':    (0.00,  0.00,  (1, 0, 4), 1, 1, 0.00,  1, 1, 0, 0),
            'topund': (33.33, 33.33, (1, 1, 3), 1, 1, 33.33, 0, 0, 1, 1),
            'uneval': (33.33, 33.33, (1, 1, 3), 1, 1, 33.33, 0, 0, 1, 1),
        }
        for tag, (en, el, gap, den, mis, pn, tie, mix, ue, ua) in exp9.items():
            o = out[tag]
            g1 = re.search(r'严格（t₁ < t_cur[^:]*:\s*([\d.]+)%\s+宽松（t₁ ≤ t_cur[^:]*:\s*([\d.]+)%', o)
            g2 = re.search(r'池化 Σ真负/Σ窗 = ([\d.]+)% / 宽松 ([\d.]+)%', o)
            g3 = re.search(r'首推 current 的窗 (\d+) 个 ＋ 首推候选\*\*无真值\*\*的窗 (\d+) 个'
                           r'.*?＋ 进分母 (\d+) 个', o)
            g4 = re.search(r'的 (\d+) 个窗里，其实\*\*存在更优候选\*\*的 (\d+) 个 ⇒\s*([\d.]+|nan)%', o)
            g5 = re.search(r'首推并列的窗 (\d+) 个.*?组内真假不一 (\d+) 个', o)
            g6 = re.search(r'结构性未评估候选.*?:\s*(\d+) 项，其中「秩高于 current」(\d+) 项', o)
            g7 = re.search(r'本节合格集 (\d+) /', o)
            got = (float(g1.group(1)), float(g1.group(2))) if g1 else None
            got3 = tuple(int(x) for x in g3.groups()) if g3 else None
            got4 = (int(g4.group(1)), int(g4.group(2)), g4.group(3)) if g4 else None
            got5 = tuple(int(x) for x in g5.groups()) if g5 else None
            got6 = tuple(int(x) for x in g6.groups()) if g6 else None
            nrec = int(g7.group(1)) if g7 else None
            print(f'  {tag:8s} 严格/宽松 {got}  缺口 {got3}（合格集 {nrec}）  池化 '
                  f'{g2.groups() if g2 else None}  镜像(分母,命中,率) {got4}  并列 {got5}  '
                  f'未评估 {got6}')
            if got is None or abs(got[0] - en) > 0.01 or abs(got[1] - el) > 0.01:
                fails.append(f'[9]{tag}: 首推率 {got} != {en}/{el}')
            if got3 != gap or nrec is None or sum(got3) != nrec:
                fails.append(f'[9]{tag}: 缺口 {got3} != {gap}（或三桶相加 != 合格集 {nrec}）')
            if not g2 or abs(float(g2.group(1)) - pn) > 0.01 or float(g2.group(2)) != pn:
                fails.append(f'[9]{tag}: 池化 {g2.groups() if g2 else None} != {pn}')
            if got4 is None or got4[:2] != (den, mis):
                fails.append(f'[9]{tag}: 镜像 {got4} != {(den, mis)}')
            elif den == 0:
                if got4[2] != 'nan':
                    fails.append(f'[9]{tag}: 镜像分母为 0 却没退化成 nan（打印 {got4[2]}）')
            elif abs(float(got4[2]) - mis / den * 100) > 0.01:
                fails.append(f'[9]{tag}: 镜像率 {got4[2]} != {mis / den * 100:.2f}%')
            if got5 != (tie, mix):
                fails.append(f'[9]{tag}: 并列/混 {got5} != {(tie, mix)}')
            if got6 != (ue, ua):
                fails.append(f'[9]{tag}: 未评估计数 {got6} != {(ue, ua)}')
        # nontop 的关键断言：那条 rank 3.5 的行既要进「未评估」计数，又**绝不能**变成首推
        # （若实现按行序或按「最后一行」取首推，这里立刻分叉）
        same9 = []
        for key in (r'严格（t₁ < t_cur[^:]*:\s*[\d.]+%\s+宽松（t₁ ≤ t_cur[^:]*:\s*[\d.]+%',
                    r'池化 Σ真负/Σ窗 = [\d.]+% / 宽松 [\d.]+%',
                    r'首推 current 的窗 \d+ 个 ＋ 首推候选\*\*无真值\*\*的窗 \d+ 个.*?＋ 进分母 \d+ 个',
                    r'首推并列的窗 \d+ 个.*?组内真假不一 \d+ 个'):
            a = re.search(key, out['ok']); b = re.search(key, out['nontop'])
            same9.append(a is not None and b is not None and a.group(0) == b.group(0))
        if not all(same9):
            fails.append('nontop: 非首推的未评估行被当成了首推（核心行与干净树不一致）')
        else:
            print('  ✅ nontop: 未评估行（rank 3.5 > rcur 3.0）未被当成首推，核心读数与干净树逐字相同')

        print('\n=== 结论 ===')
        if fails:
            print('❌ 失败: ' + '; '.join(fails))
        else:
            print('✅ 全部断言通过（★核心首推真负率（严格/宽松/分母/池化/镜像/并列/缺口恒等式）'
                  '头号误报率/漏报率/池化/top-1/四口径/分层/诊断计数/'
                  '配对失败/结构性未评估/量纲守卫/缺文件常量 全覆盖）')
        return 1 if fails else 0
    finally:
        shutil.rmtree(base, ignore_errors=True)


def pytest_round(x):
    return float(f'{x:.2f}')


if __name__ == '__main__':
    sys.exit(main())
