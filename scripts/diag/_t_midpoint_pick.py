"""_t_midpoint_pick.py — 本地自检 17.4.1 的 midpoint 选点政策（不需要服务器/训练/torch）。

做两件事：
  ① 从 src/train_sweep.py **抽出** midpoint_epoch_of / pick_midpoint 两个 def 直接跑
     （测的是发布出去的那段文本本身，不是它的复制品）；
  ② 复现「字符串序 vs 数值序」那个坑 —— 这是「取平台末端」最容易被写反的地方。

用法（本地 Windows）：python scripts/diag/_t_midpoint_pick.py
"""
import ast
import os
import re
import sys

import numpy as np

# 控制台默认是 GBK，打不出 ⚠/✓ → 强制 UTF-8（否则诊断信息自己会崩，掩盖真实结论）
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, 'src', 'train_sweep.py')

FAIL = 0


def ok(msg):
    print(f"  OK   {msg}")


def no(msg):
    global FAIL
    FAIL += 1
    print(f"  FAIL {msg}")


# ---------- ① 从真源码里抽出这两个函数 ----------
def load_funcs():
    text = open(SRC, encoding='utf-8').read()
    tree = ast.parse(text)
    want = {'midpoint_epoch_of', 'pick_midpoint'}
    picked = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in want:
            picked.append(node)
    names = {n.name for n in picked}
    if names != want:
        print(f"FAIL 抽不到函数，得到 {names}")
        sys.exit(1)
    mod = ast.Module(body=picked, type_ignores=[])
    ns = {'os': os, 'np': np}
    exec(compile(mod, SRC, 'exec'), ns)
    return ns['midpoint_epoch_of'], ns['pick_midpoint']


midpoint_epoch_of, pick_midpoint = load_funcs()
print(f"抽取到 midpoint_epoch_of / pick_midpoint（源: {os.path.relpath(SRC, ROOT)}）")


# ---------- ② 字符串序 vs 数值序 ----------
print("--- 1) 字符串序陷阱：sorted(glob()) 会把 ep100 排在 ep50 前面 ---")
names = ['midpoint_ep50.pt', 'midpoint_ep100.pt', 'midpoint_ep150.pt',
         'midpoint_ep200.pt', 'midpoint_ep250.pt']
str_order = sorted(names)
num_order = sorted(names, key=midpoint_epoch_of)
print(f"      字符串序: {[midpoint_epoch_of(n) for n in str_order]}")
print(f"      数值序  : {[midpoint_epoch_of(n) for n in num_order]}")
if midpoint_epoch_of(str_order[-1]) == 50:
    ok("确认字符串序下 [-1] 是 ep50（若不修就会把「取末端」取反）")
else:
    no(f"字符串序假设不成立，末位是 ep{midpoint_epoch_of(str_order[-1])}")
if [midpoint_epoch_of(n) for n in num_order] == [50, 100, 150, 200, 250]:
    ok("数值序排序正确")
else:
    no("数值序排序错误")
if midpoint_epoch_of('midpoint_ep250.pt') == 250 and midpoint_epoch_of('junk.pt') == -1:
    ok("解析：正常名→int，垃圾名→-1")
else:
    no("解析函数行为不对")

# ---------- 选点行为 ----------
# 「平台已形成」形状：argmax **严格在内部**（ep200），末端是 ep250 —— 对应 42b 那类
#   「训练侧最优点落在中点区间内部」的实测形态（config.py:51 记的正是这个）。
P_INNER = [(50, 61.0), (100, 66.0), (150, 74.0), (200, 74.4), (250, 74.2)]
# 被截断形状：曲线到末端仍在爬（argmax == 末位）
P_TRUNC = [(50, 61.0), (100, 66.0), (150, 70.0), (200, 72.0), (250, 75.0)]
# 末位 NaN
P_NANLAST = [(50, 61.0), (100, 66.0), (150, 74.0), (200, 73.5), (250, float('nan'))]

print("--- 2) mode='last'：取 epoch 最大的**有效**中点 ---")
ep, notes = pick_midpoint(P_INNER, 'last')
if ep == 250:
    ok("last → ep250（末端）")
else:
    no(f"last 应为 250，得到 {ep}")
if any('守卫⑤' in n and '✓' in n for n in notes):
    ok("argmax 在内部(ep200) → 守卫⑤ 报「平台已形成，规则成立」")
else:
    no(f"守卫⑤ 未正确判内部: {notes}")
if any('⚠' in n for n in notes):
    no("argmax 在内部时不该告警")
else:
    ok("argmax 在内部时不告警")

print("--- 3) 守卫⑤ 触发：曲线到末端仍在爬（早停砍掉平台）---")
ep, notes = pick_midpoint(P_TRUNC, 'last')
if ep == 250:
    ok("last 仍选 ep250（守门只告警、不改变选择）")
else:
    no(f"last 应为 250，得到 {ep}")
if any('守卫⑤' in n and '⚠' in n and '末端' in n for n in notes):
    ok("末端 argmax → 守卫⑤ 正确告警")
else:
    no(f"守卫⑤ 未告警: {notes}")

print("--- 4) 末位 NaN：取最后一个**有效**中点，不选 NaN ---")
ep, notes = pick_midpoint(P_NANLAST, 'last')
if ep == 200:
    ok("末位 NaN → 退到 ep200（不拿 NaN 当平台末端）")
else:
    no(f"应为 200，得到 {ep}")

print("--- 5) mode='argmax_capture2'：旧行为可复现，且与 last 确实不同 ---")
ep_old, notes_old = pick_midpoint(P_INNER, 'argmax_capture2')
ep_new, _ = pick_midpoint(P_INNER, 'last')
if ep_old == 200 and ep_new == 250:
    ok("同一曲线上：旧模式选 argmax=ep200，新模式选末端=ep250（可复现旧行为、且默认确实变了）")
else:
    no(f"模式区分失败：argmax={ep_old} last={ep_new}")
if any('仅供复现' in n for n in notes_old):
    ok("旧模式打印里标明了「仅供复现」")
else:
    no(f"旧模式未标注用途: {notes_old}")

print("--- 6) 退化与确定性 ---")
ep, notes = pick_midpoint([(50, float('nan')), (100, float('nan'))], 'last')
if ep is None and notes == []:
    ok("全 NaN → 返回 (None, [])（调用点各自走退化路径）")
else:
    no(f"全 NaN 应返回 (None, [])，得到 ({ep}, {notes})")
P_TIE = [(50, 70.0), (100, 74.0), (150, 74.0), (200, 74.0)]
# 同分（74.0 三个）→ 取**最早** ep100。与旧代码在字符串序下 `>` 的严格比较结果一致
#   （旧序 100 先于 150/200 被看到）⇒ 换成数值序后同分行为不变，复现旧 run 不受影响。
if pick_midpoint(P_TIE, 'argmax_capture2')[0] == 100 and pick_midpoint(P_TIE, 'last')[0] == 200:
    ok("同分取最早 ep100（确定性、且与旧代码一致）；last 不受同分影响")
else:
    no(f"同分行为不对：argmax={pick_midpoint(P_TIE, 'argmax_capture2')[0]}")
# 幂等：同输入两次调用同结果
if pick_midpoint(P_INNER, 'last')[0] == pick_midpoint(P_INNER, 'last')[0]:
    ok("同输入同输出（无隐藏状态）")
else:
    no("非确定性")

print("--- 7) 源码守卫：两处调用点都必须走 pick_midpoint，且排序必须带 key= ---")
raw = open(SRC, encoding='utf-8').read()
n_call = len(re.findall(r'pick_midpoint\(pairs, MIDPOINT_SELECT\)', raw))
n_glob = len(re.findall(r"glob\(os\.path\.join\(OUTPUT_DIR, 'midpoint_ep\*\.pt'\)\)", raw))
n_keyed = len(re.findall(r"glob\(os\.path\.join\(OUTPUT_DIR, 'midpoint_ep\*\.pt'\)\), key=midpoint_epoch_of\)", raw))
n_argmax = len(re.findall(r'best_score, best_ep = score, ep_num', raw))
print(f"      pick_midpoint 调用={n_call}  glob 点={n_glob}  带 key= 的={n_keyed}  残留 argmax 循环={n_argmax}")
if n_call == 2:
    ok("两处调用点都用同一决策函数")
else:
    no(f"应有 2 处调用 pick_midpoint，得到 {n_call}（政策会落一半）")
if n_glob == n_keyed:
    ok("所有 mid_files 排序都带 key=midpoint_epoch_of（无字符串序残留）")
else:
    no(f"有 {n_glob - n_keyed} 处排序漏了 key=")
if n_argmax == 0:
    ok("无残留的 argmax 选点循环")
else:
    no(f"仍有 {n_argmax} 处旧 argmax 选点代码")

print("==================================================")
if FAIL:
    print(f"FAIL 共 {FAIL} 项")
    sys.exit(1)
print("OK 全部通过")
