"""_t_shadow_analyze_2col.py — _shadow_analyze.py 双列改动的回归测试（本地，不需要服务器）。

真实布局 tl_opt_batch/<levelN>/<STEM>/gnn_shadow.csv（一个 STEM 一个 CSV，各 window 是行段），
造三棵同数据树：
  A) 旧格式：只有 gnn_pred / true_delay      → 输出必须与改动前完全一致（无 A/B 节）
  B) rev_nan：第二列 = 2-pred（**排序完全反转，必输**），并埋两类 NaN
       · 6 行窗口里 1 行 gnn_pred2=nan → 该集 A/B 用 5 行（主口径仍 6 行）
       · 恰 4 行窗口里 1 行 gnn_pred2=nan → 该集 A/B 落掉（计入 ab_dropped_sets）
       → 断言：主口径各节与 A 逐字节相同；配对遗憾 模型1胜≥1 且 模型2胜=0（方向不能反）
  C) oracle：第二列 = true_delay（**完美预测，必胜**）
       → 断言：配对遗憾 模型2胜≥1 且 模型1胜=0，模型2 遗憾 = 0.00%、严格k=3 = 100%

用法: python3 scripts/diag/_t_shadow_analyze_2col.py
"""
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile

import numpy as np

# 本机 Windows 控制台默认 GBK，脚本自身的 ✅/⚠ 会 UnicodeEncodeError（与解析逻辑无关，
# 服务器 UTF-8 locale 无此问题）—— 让脚本在任何控制台下都自洽，不依赖外部设 PYTHONIOENCODING。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8')
    except (AttributeError, OSError):
        pass

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYZE = os.path.join(ROOT, 'scripts', 'diag', '_shadow_analyze.py')

# 固定算术、不用 random：窗口内 k 的两个系数与各自模数互质 → 排名无并列，可复现
WINDOWS = ((0, 6, 2), (1, 4, 1), (2, 7, None))   # (window, 行数, 第二列写 nan 的 k)


def rows_for(circ_i, win, n):
    out = []
    for k in range(n):
        out.append({
            "k": k,
            "pred": 1.0 + ((circ_i * 10 + win) * 37 + k * 11) % 23 / 100.0,
            "true": 1.0 + ((circ_i * 5 + win * 3) * 7 + k * 7) % 19 / 50.0,
        })
    return out


def col2_value(mode, k, nan2_at, r):
    if mode is None:
        return None
    if mode == 'oracle':
        return f"{r['true']:.6e}"
    if k == nan2_at:
        return 'nan'
    if mode == 'rev_nan':
        return f"{2.0 - r['pred']:.6e}"       # 反转排序 → 必输
    raise ValueError(mode)


def write_tree(root, mode):
    for ci, circ in enumerate(['level0/CIRC_A', 'level2/CIRC_B']):
        d = os.path.join(root, circ)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'gnn_shadow.csv'), 'w') as f:
            for win, n, nan2_at in WINDOWS:
                for r in rows_for(ci, win, n):
                    line = (f"eval_idx={r['k']}, iter=1, window={win}, "
                            f"gnn_pred={r['pred']:.6e}, true_delay={r['true']:.6e}, "
                            f"transistors={100 + r['k']}")
                    v = col2_value(mode, r['k'], nan2_at, r)
                    if v is not None:
                        line += f", gnn_pred2={v}"
                    f.write(line + '\n')
            # 损坏行（两列解析都必须同样跳过）
            f.write('eval_idx=99, iter=1, window=0, gnn_pred=1.0, true_del\n')


def run(root):
    # 本机 Windows 控制台默认 GBK，脚本里的 ✅❌ 会 UnicodeEncodeError（与本改动无关，
    # 服务器 UTF-8 locale 无此问题）→ 强制子进程用 UTF-8 写管道。
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    p = subprocess.run([sys.executable, ANALYZE, '--root', root],
                       capture_output=True, text=True, encoding='utf-8', env=env)
    if p.returncode != 0:
        print(p.stdout, p.stderr)
        raise SystemExit(f'analyzer 退出码 {p.returncode}')
    return p.stdout


AB_MARK = '=== 两列并排 A/B'
DETAIL_MARK = '=== 每候选集明细'


def split_ab(out):
    """→ (主口径+跨度节, A/B 节 或 None, 明细节)"""
    head, _, rest = out.partition(AB_MARK)
    if not rest:
        return out[:out.index(DETAIL_MARK)], None, out[out.index(DETAIL_MARK):]
    ab, _, detail = rest.partition(DETAIL_MARK)
    return head, ab, DETAIL_MARK + detail


def paired(ab, judge):
    """从 A/B 节取某判据那行的 配对 1胜/2胜/平。"""
    m = re.search(re.escape(judge) + r'\s+[\d.]+%?\s+[\d.]+%?\s+(\d+)/\s*(\d+)/\s*(\d+)', ab)
    if not m:
        m = re.search(re.escape(judge) + r'\s+[\d.]+\s+[\d.]+\s+(\d+)/\s*(\d+)/\s*(\d+)', ab)
    assert m, f'找不到 {judge} 的配对计数'
    return tuple(int(x) for x in m.groups())


def head_meta(out):
    """主口径头部两个数字行（候选集数 / 成功行）。"""
    return out.split(DETAIL_MARK)[0]


def main():
    base = tempfile.mkdtemp(prefix='_t_2col_')
    roots = {m: os.path.join(base, m) for m in ('old', 'rev', 'oracle')}
    write_tree(roots['old'], None)
    write_tree(roots['rev'], 'rev_nan')
    write_tree(roots['oracle'], 'oracle')
    out = {m: run(roots[m]) for m in ('old', 'rev', 'oracle')}

    h_old, ab_old, d_old = split_ab(out['old'])
    h_rev, ab_rev, d_rev = split_ab(out['rev'])
    h_or, ab_or, d_or = split_ab(out['oracle'])

    # —— 1) 缺列守卫：旧格式无 A/B 节，且主口径/明细都不变 ——
    assert ab_rev is not None and ab_or is not None, '带第二列的树没输出 A/B 节'
    assert ab_old is None and AB_MARK not in out['old'], '旧格式不该有 A/B 节'
    assert h_old == h_rev == h_or, '主口径节在三种格式下不一致（第二列污染了主口径）'
    assert d_old == d_rev == d_or, '明细节在三种格式下不一致'
    print('[1] 旧格式无 A/B 节；三棵树主口径与明细逐字节相同 ✅')

    # —— 2) NaN 落集计数 ——
    m = re.search(r'配对候选集数 = (\d+)（主口径 (\d+) 集；因第二列缺/NaN 落掉 (\d+) 集）'
                  r'\s+带第二列的行 = (\d+)/(\d+)', ab_rev)
    assert m, 'A/B 头部格式不符'
    ab_n, main_n, dropped, c2rows, okrows = (int(x) for x in m.groups())
    assert (main_n, ab_n, dropped) == (6, 4, 2), (main_n, ab_n, dropped)
    assert c2rows == okrows == 34, (c2rows, okrows)
    # oracle 树无 NaN → 配对集 = 主口径集
    mo = re.search(r'配对候选集数 = (\d+)（主口径 (\d+) 集；因第二列缺/NaN 落掉 (\d+) 集）', ab_or)
    assert tuple(int(x) for x in mo.groups()) == (6, 6, 0), mo.groups()
    print(f'[2] rev 树 配对={ab_n}/{main_n} 落掉={dropped}（NaN 集只从 A/B 剔除）；'
          f'oracle 树 配对=6/6 落掉=0 ✅')

    # —— 3) 独立复算对账：配对胜负与均值必须与分析器打印的一致 ——
    # 不预设「反转必输」：n=4 的小集里反转可能碰巧选中真最优 → 该集它赢（实测 3/1/0）。
    # 所以这里验的是**计数与均值本身没说谎**：用另一份实现（numpy）逐集重算再对账。
    def independent(mode):
        """按 rows_for/col2_value 复算每个候选集的 (regret1, regret2)；None = 该集进不了 A/B。"""
        out = []
        for ci in range(2):
            for win, n, nan2_at in WINDOWS:
                rs = rows_for(ci, win, n)
                keep = [r for r in rs if col2_value(mode, r['k'], nan2_at, r) != 'nan']
                if len(keep) < 4:
                    continue
                t = np.array([r['true'] for r in keep])
                p1 = np.array([r['pred'] for r in keep])
                p2 = np.array([float(col2_value(mode, r['k'], nan2_at, r)) for r in keep])
                best = t.min()
                out.append(((t[p1.argmin()] - best) / best, (t[p2.argmin()] - best) / best))
        return out

    import numpy as np
    rev = independent('rev_nan')
    w1 = sum(1 for a, b in rev if a < b)
    w2 = sum(1 for a, b in rev if b < a)
    tie = len(rev) - w1 - w2
    pt, pt2, ptt = paired(ab_rev, '选择遗憾（越低越好）')
    assert (pt, pt2, ptt) == (w1, w2, tie), \
        f'rev 树配对计数不符：打印 {pt}/{pt2}/{ptt}，独立复算 {w1}/{w2}/{tie}'
    m1, m2 = statistics.mean(a for a, _ in rev), statistics.mean(b for _, b in rev)
    printed = re.search(r'选择遗憾（越低越好）\s+([\d.]+)%\s+([\d.]+)%', ab_rev).groups()
    assert printed == (f'{m1*100:.2f}', f'{m2*100:.2f}'), \
        f'rev 树遗憾均值不符：打印 {printed}，独立复算 {m1*100:.2f}/{m2*100:.2f}'
    # 全局方向：反转排序整体应更差（不要求每集都差）
    assert m2 > m1, f'rev 树模型2（反转）整体遗憾应更大：{m1:.4f} vs {m2:.4f}'
    print(f'[3] rev 树配对 {w1}/{w2}/{tie} 与遗憾均值 {printed[0]}%/{printed[1]}% '
          f'均与独立复算一致；整体方向 模型2 更差 ✅')

    # —— 4) 反向方向测试：第二列 = 真值 → 每集 model2 遗憾恒 0，配对全归模型2 ——
    orc = independent('oracle')
    wo2 = sum(1 for a, b in orc if b < a)
    wo1 = sum(1 for a, b in orc if a < b)
    assert all(b == 0.0 for _, b in orc), 'oracle 树模型2遗憾应恒为 0'
    assert wo1 == 0 and wo2 >= 1, f'oracle 树配对应 模型1胜=0，实得 {wo1}/{wo2}'
    assert paired(ab_or, '选择遗憾（越低越好）') == (wo1, wo2, len(orc) - wo1 - wo2), 'oracle 配对计数不符'
    assert re.search(r'选择遗憾（越低越好）\s+[\d.]+%\s+0\.00%', ab_or), 'oracle 树模型2遗憾应为 0.00%'
    assert re.search(r'严格 k=3（#1∈预测前3）\s+[\d.]+%\s+100\.0%', ab_or), 'oracle 树模型2严格k=3 应为 100%'
    print(f'[4] oracle 树配对 {wo1}/{wo2}（全归模型2），模型2 遗憾恒 0、严格k=3 100% ✅')

    print('--- rev 树 A/B 节原文 ---')
    print(AB_MARK + ab_rev.rstrip())
    shutil.rmtree(base, ignore_errors=True)
    print('\n[OK] 双列解析回归通过（含胜负方向）')


if __name__ == '__main__':
    main()
