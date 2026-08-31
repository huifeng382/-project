#!/usr/bin/env python3
"""
_corr_idsavg.py — 粗 ids_avg vs 全精度 ids_avg 相关性实验
============================================================
目的：回答「粗仿真能否为 GNN 提供与全精度足够相关的 ids_avg 特征」。
方法：对同一批 deck（tb 文件），生成全精度(.tran 1p 1n)与粗精度(.tran 10p 200p)
      两份测试台，自动为 DUT 内每个 MOSFET 生成逐管 .measure tran AVG ID(...)，
      跑 Xyce，解析 .mt0，得到每个 deck 的逐管 ids_avg 向量，
      对全/粗两向量计算 Pearson r（逐管级 + 逐门级聚合），并输出分档耗时。

用法（在服务器上）：
  python3 _corr_idsavg.py \
      --deck-list <文件:每行一个 tb 绝对路径> \
      --xyce /usr/local/bin/Xyce \
      --out <结果CSV输出前缀> \
      --coarse-tran "10p 200p" \
      [--full-tran "1p 1n"] \
      [--jobs N]            # Xyce 并发数（默认 4）
  （deck 会先被复制到 --work 目录，不污染原始目录）

输出：
  <prefix>_perdevice.csv  逐管级 ids_avg 与 r
  <prefix>_pergate.csv    逐门级聚合 ids_avg 与 r
  <prefix>_summary.csv    每个 deck 的 r、两精度耗时、电路规模
"""

import argparse
import csv
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

MEAS_RE = re.compile(r'^(\w+)\s*=\s*([-\d.eE+]+)\s*$')


def extract_measures(mt0_path):
    """解析 Xyce .mt0 文件 → {measure名: 值}"""
    out = {}
    if not os.path.exists(mt0_path):
        return out
    with open(mt0_path, 'r', errors='replace') as f:
        for line in f:
            line = line.strip()
            m = MEAS_RE.match(line)
            if m:
                try:
                    out[m.group(1)] = float(m.group(2))
                except ValueError:
                    pass
    return out


def list_mosfets(dut_sp_path):
    """扫描 DUT 子电路定义 → [(门实例名, 子电路类型, 内部管数)] 及每类门管数。
    只统计 DUT 顶层实例化的 X 门（如 X_6 ... SC_INV）。"""
    subckt_mos = {}   # 子电路名 -> 管数（M_ 行数）
    cur = None
    with open(dut_sp_path, 'r', errors='replace') as f:
        for line in f:
            s = line.strip()
            if s.startswith('.SUBCKT'):
                parts = s.split()
                cur = parts[1] if len(parts) > 1 else None
                subckt_mos[cur] = 0
            elif s.startswith('M_') and cur is not None:
                subckt_mos[cur] = subckt_mos.get(cur, 0) + 1
            elif s.startswith('.ENDS') and cur is not None:
                cur = None
    gates = []
    with open(dut_sp_path, 'r', errors='replace') as f:
        for line in f:
            s = line.strip()
            if s.startswith('X_') and not s.startswith('.SUBCKT'):
                parts = s.split()
                if len(parts) >= 3:
                    gates.append((parts[0], parts[-1]))  # (X_6, SC_INV)
    return gates, subckt_mos


def build_measure_lines(gates, subckt_mos):
    """为每个门实例的每根管生成 .measure tran AVG ID(XUUT:X_6:M_1)"""
    lines = []
    for gate_name, subckt in gates:
        n = subckt_mos.get(subckt, 0)
        for i in range(1, n + 1):
            lines.append(
                f'.measure tran ids_{gate_name}_m{i} AVG ID(XUUT:{gate_name}:M_{i})')
    return lines


def make_deck(tb_path, work_dir, tran_line, tag):
    """把 tb 复制到 work_dir/<tag>_<hash>/,改写 .tran 与 .include 相对路径,
    返回 (deck_sp 路径, dut_sp 路径)。"""
    h = hashlib.md5(tb_path.encode()).hexdigest()[:10]
    d = os.path.join(work_dir, f'{tag}_{h}')
    os.makedirs(d, exist_ok=True)
    # 复制 tb + 其 .include 引用的所有文件
    tb_dir = os.path.dirname(tb_path)
    deps = [tb_path]
    with open(tb_path, 'r', errors='replace') as f:
        for line in f:
            m = re.match(r'\.include\s+"([^"]+)"', line.strip())
            if m:
                deps.append(m.group(1))
    new_names = {}
    for src in deps:
        base = os.path.basename(src)
        dst = os.path.join(d, base)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        new_names[src] = os.path.join(d, base)
    tb_dst = os.path.join(d, os.path.basename(tb_path))
    # 重写 tb：.include 绝对路径 -> 新位置；.tran -> 指定精度
    with open(tb_path, 'r', errors='replace') as f:
        content = f.read()
    for src, dst in new_names.items():
        if src != tb_path:
            content = content.replace(src, dst)
    # .tran 行替换（精确替换首个 .tran 1p 1n）
    content = re.sub(r'\.tran\s+\S+\s+\S+', f'.tran {tran_line}', content, count=1)
    # 找到 .end,在它前面插入 .measure 行
    dut_path = None
    for src, dst in new_names.items():
        base = os.path.basename(src)
        if base.startswith('dut_') or 'dut_' in base:
            dut_path = dst
    with open(tb_dst, 'w') as f:
        f.write(content)
    return tb_dst, dut_path


def run_one_deck(tb_path, work_dir, xyce, full_tran, coarse_trans, timeout_s):
    """对单个 deck 跑全精度 + 每组粗精度,返回逐管 ids_avg 与耗时
    coarse_trans: list of (tag, tran_line)"""
    result = {'tb': tb_path, 'full': None,
              'full_s': None, 'gates': 0, 'mos': 0}
    for tag, _ in coarse_trans:
        result[tag] = None
        result[f'{tag}_s'] = None
    full_tb, dut = make_deck(tb_path, work_dir, full_tran, 'full')
    if dut is None:
        return result
    gates, subckt_mos = list_mosfets(dut)
    result['gates'] = len(gates)
    result['mos'] = sum(subckt_mos.get(s, 0) for _, s in gates)
    meas = build_measure_lines(gates, subckt_mos)
    # 跑全精度
    full_mt0 = full_tb + '.mt0'
    t0 = time.time()
    ok_f = run_xyce(full_tb, xyce, meas, timeout_s)
    result['full_s'] = time.time() - t0
    result['full'] = extract_measures(full_mt0) if ok_f else {}
    # 跑每组粗精度
    for tag, tran_line in coarse_trans:
        coarse_tb, _ = make_deck(tb_path, work_dir, tran_line, tag)
        coarse_mt0 = coarse_tb + '.mt0'
        t0 = time.time()
        ok_c = run_xyce(coarse_tb, xyce, meas, timeout_s)
        result[f'{tag}_s'] = time.time() - t0
        result[tag] = extract_measures(coarse_mt0) if ok_c else {}
    return result


def run_xyce(tb_path, xyce, measure_lines, timeout_s):
    """在 tb 的 .end 前插入 measure 行并跑 Xyce"""
    with open(tb_path, 'r', errors='replace') as f:
        content = f.read()
    if '.end' in content.lower():
        content = re.sub(r'(?i)\.end\s*$',
                         '\n'.join(measure_lines) + '\n.end\n', content, count=1)
    with open(tb_path, 'w') as f:
        f.write(content)
    try:
        subprocess.run([xyce, os.path.basename(tb_path)],
                       cwd=os.path.dirname(tb_path),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=timeout_s)
        return True
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def pearson(xs, ys):
    n = len(xs)
    if n == 0:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    vx = sum((a - mx) ** 2 for a in xs)
    vy = sum((b - my) ** 2 for b in ys)
    if vx == 0 or vy == 0:
        return None
    return cov / (vx * vy) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--deck-list', required=True)
    ap.add_argument('--xyce', default='/usr/local/bin/Xyce')
    ap.add_argument('--out', default='/home/tianlang/corr_test/results')
    ap.add_argument('--work', default='/home/tianlang/corr_test/work')
    ap.add_argument('--full-tran', default='1p 1n')
    ap.add_argument('--coarse-tran', default='10p 200p',
                    help='逗号分隔多组: tag=tran_line,tag2=tran_line2')
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--timeout', type=int, default=1200)
    ap.add_argument('--limit', type=int, default=0, help='只处理前 N 个 deck(0=全部)')
    args = ap.parse_args()

    coarse_trans = []
    for item in args.coarse_tran.split(','):
        item = item.strip()
        if '=' in item:
            tag, tran = item.split('=', 1)
            coarse_trans.append((tag.strip(), tran.strip()))
        else:
            coarse_trans.append((f'c{len(coarse_trans)}', item))
    if not coarse_trans:
        coarse_trans = [('coarse', '10p 200p')]

    with open(args.deck_list) as f:
        decks = [l.strip() for l in f if l.strip()]
    if args.limit:
        decks = decks[:args.limit]
    os.makedirs(args.work, exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f'[corr] {len(decks)} decks, full={args.full_tran} '
          f'coarse={[(t, l) for t, l in coarse_trans]} '
          f'jobs={args.jobs}', flush=True)

    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run_one_deck, d, args.work, args.xyce,
                          args.full_tran, coarse_trans, args.timeout): d
                for d in decks}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            n_full = len([v for v in r['full'].values() if v is not None])
            tags = ' '.join(f'{t}={len([v for v in (r[t] or {}).values() if v is not None])}'
                            for t, _ in coarse_trans)
            print(f'  [{i}/{len(decks)}] {os.path.basename(r["tb"])[:50]} '
                  f'mos={r["mos"]} full={n_full} {tags} '
                  f'({r["full_s"]:.1f}s/'
                  f'{" ".join(f"{r[t+chr(95)+chr(115)]:.1f}s" for t, _ in coarse_trans)})',
                  flush=True)

    # ---- 逐管级 ----
    dev_rows = []
    gate_rows = []
    sum_rows = []
    for r in results:
        full, coarse = r['full'], r['full']
        # 对每组粗精度各算一次
        common = [k for k in full
                  if k in coarse and k.upper().startswith('IDS_')]
        fv = [full[k] for k in common]
        base = {'tb': r['tb'], 'gates': r['gates'], 'mos': r['mos'],
                'devices': len(fv),
                'full_s': r['full_s']}
        for tag, _ in coarse_trans:
            cv = [r[tag][k] for k in common if k in r[tag]]
            r_dev = pearson(fv, cv) if len(fv) >= 2 and len(cv) == len(fv) else None
            row = dict(base)
            row[f'r_{tag}'] = r_dev
            row[f'{tag}_s'] = r.get(f'{tag}_s')
            row[f'speedup_{tag}'] = (r['full_s'] / r[f'{tag}_s']) \
                if r.get(f'{tag}_s') else None
            sum_rows.append(row)
            for k in common:
                if k in r[tag]:
                    dev_rows.append({'tb': os.path.basename(r['tb']),
                                     'device': k, 'full': full[k],
                                     f'{tag}': r[tag][k]})
            # 逐门聚合
            gate_map = {}
            for k in common:
                m = re.match(r'ids_(X_\d+)_m(\d+)', k, re.IGNORECASE)
                if m and k in r[tag]:
                    gate_map.setdefault(m.group(1), []).append(
                        (full[k], r[tag][k]))
            for g, pairs in gate_map.items():
                fa = sum(p[0] for p in pairs) / len(pairs)
                ca = sum(p[1] for p in pairs) / len(pairs)
                gate_rows.append({'tb': os.path.basename(r['tb']),
                                  'gate': g, 'tag': tag,
                                  'full_avg': fa, 'coarse_avg': ca})

    dev_fields = ['tb', 'device', 'full'] + [t for t, _ in coarse_trans]
    with open(args.out + '_perdevice.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=dev_fields)
        w.writeheader()
        w.writerows(dev_rows)
    with open(args.out + '_pergate.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['tb', 'gate', 'tag',
                                          'full_avg', 'coarse_avg'])
        w.writeheader()
        w.writerows(gate_rows)
    sum_fields = ['tb', 'gates', 'mos', 'devices', 'full_s']
    for tag, _ in coarse_trans:
        sum_fields += [f'r_{tag}', f'{tag}_s', f'speedup_{tag}']
    with open(args.out + '_summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=sum_fields)
        w.writeheader()
        w.writerows(sum_rows)

    # ---- 汇总 ----
    for tag, _ in coarse_trans:
        all_fv, all_cv = [], []
        for row in dev_rows:
            if tag in row:
                all_fv.append(row['full'])
                all_cv.append(row[tag])
        r_all = pearson(all_fv, all_cv)
        rs = [r[f'r_{tag}'] for r in sum_rows if r.get(f'r_{tag}') is not None]
        print(f'\n===== 粗精度 [{tag}] 汇总 =====', flush=True)
        print(f'逐管总样本: {len(all_fv)}  r = {r_all:.4f}' if r_all
              else f'[{tag}] 逐管样本不足', flush=True)
        if rs:
            print(f'逐 deck r: min={min(rs):.4f} median={sorted(rs)[len(rs)//2]:.4f} '
                  f'max={max(rs):.4f}  (n={len(rs)} decks)', flush=True)
        fs = [r['full_s'] for r in sum_rows if r.get('full_s')]
        cs = [r[f'{tag}_s'] for r in sum_rows if r.get(f'{tag}_s')]
        if fs and cs:
            print(f'平均耗时: full={sum(fs)/len(fs):.1f}s '
                  f'{tag}={sum(cs)/len(cs):.1f}s '
                  f'加速比={sum(fs)/sum(cs):.2f}x', flush=True)
    print(f'结果已写: {args.out}_*.csv', flush=True)


if __name__ == '__main__':
    main()
