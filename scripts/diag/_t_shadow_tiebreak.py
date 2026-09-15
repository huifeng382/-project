"""并列打破回归测试：同一批候选行、两种 CSV 行序，分析器必须给同一结果。

复刻 2026-09-15 实测到的那个坑：gnn_pred 只写 7 位有效数字（gnn_shadow.rs 的 {:.6e}），
第 8 位起的差异被抹平 → 伪并列 → argsort 按 CSV 行序打破（CSV 由并行分片 append 写，行序不定）。
真值侧故意让伪并列的两行差 40%（= 实测 DEPTH_MIX w=7 那集 0.00% ↔ 36.34% 的形状）。

用法: python3 scripts/diag/_t_shadow_tiebreak.py
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYZE = os.path.join(ROOT, "scripts", "diag", "_shadow_analyze.py")

# (eval_idx, gnn_pred, true_delay)；e0/e1 预测伪并列，真值差 40%
ROWS = [(0, 1.000000e+00, 1.00), (1, 1.000000e+00, 1.40)] + [
    (k, 1.0 + k / 100.0, 1.0 + k / 100.0) for k in range(2, 24)]

ORD_A = list(range(len(ROWS)))              # 真最优(e0)在前
ORD_B = [1, 0] + list(range(2, len(ROWS)))  # 伪并列的另一半(e1)在前


def write(root, order):
    d = os.path.join(root, "level2", "DEPTH_MIX")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "gnn_shadow.csv"), "w") as f:
        for k in order:
            e, p, t = ROWS[k]
            f.write(f"eval_idx={e}, iter=1, window=0, gnn_pred={p:.6e}, "
                    f"true_delay={t:.6e}, transistors=100\n")


def run(root):
    p = subprocess.run([sys.executable, ANALYZE, "--root", root],
                       capture_output=True, text=True, encoding="utf-8",
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    return p.stdout, p.stderr, p.returncode


def pick(out, tag):
    for l in out.splitlines():
        if tag in l:
            return l.strip()
    return None


# —— 0) 先证明测试不是空的：旧逻辑（默认 argsort + 不定序）两种行序必然不同 ——
def old_regret(order):
    g = np.array([ROWS[k][1] for k in order])
    t = np.array([ROWS[k][2] for k in order])
    og = np.argsort(g)                       # 旧：quicksort，无 kind="stable"
    return (t[og[0]] - t.min()) / t.min()


r_a, r_b = old_regret(ORD_A), old_regret(ORD_B)
assert r_a != r_b, f"测试无效：旧逻辑在两种行序下结果相同({r_a:.4f})，没复现出坑"
print(f"[0] 旧逻辑确实受行序影响：regret {r_a:.2%} vs {r_b:.2%} ✅（测试有效）")

# —— 1) 新逻辑：两种行序必须逐字节同结果 ——
base = tempfile.mkdtemp(prefix="_t_tie_")
pa, pb = os.path.join(base, "a"), os.path.join(base, "b")
write(pa, ORD_A)
write(pb, ORD_B)
oa, ea, ca = run(pa)
ob, eb, cb = run(pb)
assert ca == 0 and cb == 0, f"分析器退出码 {ca}/{cb}\n{ea}\n{eb}"

TAIL_A = oa.split("=== 戳结束 ===")[-1]
TAIL_B = ob.split("=== 戳结束 ===")[-1]
assert TAIL_A == TAIL_B, "行序仍影响结果：两种行序的分析正文不一致"
print("[1] 两种行序结果正文逐字节相同 ✅")

# —— 2) 两个口径都在（宽松是必需项）——
for tag in ("前k名中出现实际第1名（严格）", "前k名中出现实际前k之一（宽松）"):
    assert tag in oa, f"缺口径：{tag}"
print("[2] 严格 + 宽松两口径都在 ✅")
print("    " + pick(oa, "前k名中出现实际第1名（严格）"))
print("    " + pick(oa, "前k名中出现实际前k之一（宽松）"))

# —— 3) 定序后取到的是 eval_idx 小的那个（= 0.00%，不再随机翻转）——
sel = pick(oa, "选择遗憾（GNN自选top1）") or ""
assert "0.00%" in sel, sel
print("[3] 并列按 eval_idx 打破，稳定取到 e0 → " + sel + " ✅")

# —— 4) 运行配置戳落在输出里 ——
# 本机（Windows）没有 pgrep / Xyce，走的是「没有 serve 在跑」分支；服务器上会打 ckpt 行。
# 所以只断言戳头 + 环境行必须在，serve 分支两种都接受。
assert "=== 运行配置戳 ===" in oa and "=== 戳结束 ===" in oa, "戳没打出来"
assert pick(oa, "  环境        :"), "戳里缺环境行"
print("[4] 运行配置戳已打头 ✅")
print("    " + (pick(oa, "  serve") or "  serve: -"))
print("    " + (pick(oa, "  Xyce") or "  Xyce: -"))
print("    " + (pick(oa, "  CSV 快照") or "  CSV 快照: -"))

print("\n[OK] 并列打破 + 配置戳回归通过")
