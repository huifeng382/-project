#!/usr/bin/env python3
"""auto_launch.py — 多 run 并行启动自动编排（16.7.0）

一条命令并行启动 N 个实验，自动处理：
  1) master 缓存（图缓存 + 离群点掩码）：未就绪则由第一个 run（builder）顺带构建
  2) 轮询 builder 日志到 "Start training..."（= 图缓存建完 + 掩码算完）→ 自动复制进 master
  3) 其余 run 从 master 种子启动（跳过建图 ~8min + 离群点基训 ~30-40min），自动错峰（--stagger）
  4) 启动阶段（未到 Start training... 前）死掉的 run 自动重启（--retry 次）
  5) 防呆：启动前检查该 run 目录是否已有在跑进程（避免 rm -rf 掉正在训练的 run）

用法（服务器，nohup 后台，跑完即退、训练进程独立存活）：
  cd ~/-project && git pull
  nohup ~/venv/bin/python3 scripts/diag/auto_launch.py \
      v2wave42 v2nowave42 v2ia42 v2cov2542 \
      --stagger 900 --retry 2 > ~/auto_launch.log 2>&1 &

前置：数据在 ~/-project/data（mtime 稳定）；磁盘有余量；不要在已有 run 训练时传入同名 variant。
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import time

HOME = os.path.expanduser('~')
START_MARKER = 'Start training...'
GRAPH_MIN = 30000          # 图缓存完整性的下限（当前数据 ~43.7k）
BUILD_TIMEOUT = 3 * 3600   # builder 构建超时


def log(msg):
    print(f'[{time.strftime("%F %T")}] {msg}', flush=True)


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def master_ready(master):
    g = os.path.join(master, 'graphs')
    if not os.path.exists(os.path.join(g, '.version')):
        return False
    if len(glob.glob(os.path.join(g, '*_graph.pt'))) < GRAPH_MIN:
        return False
    if not glob.glob(os.path.join(master, 'outlier', 'outlier_keep_*.npy')):
        return False
    return True


def alive(pid):
    if pid is None:
        return True  # 无法确认时假定存活
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def running_run_dirs():
    """返回当前在跑的 main.py 进程的工作目录（= run 目录）。"""
    out = sh(['pgrep', '-f', r'main[.]py'])
    dirs = set()
    for pid in out.stdout.split():
        try:
            dirs.add(os.path.realpath(os.readlink(f'/proc/{pid}/cwd')))
        except OSError:
            pass
    return dirs


def paused_run_dir(variant):
    """目录存在（代码完整）但无进程在跑 = 暂停态，可 RESUME。"""
    d = os.path.join(HOME, f'project-107-{variant}')
    if not (os.path.exists(os.path.join(d, 'main.py')) and os.path.exists(os.path.join(d, 'src'))):
        return False
    return os.path.realpath(d) not in running_run_dirs()


def launch(setup, variant, seed, resume=False):
    run_dir = os.path.join(HOME, f'project-107-{variant}')
    if os.path.realpath(run_dir) in running_run_dirs():
        log(f'SKIP {variant}: 目录 {run_dir} 已有进程在跑（避免误杀）')
        return None, 'skip'
    env = dict(os.environ, CACHE_SEED=seed)
    if resume:
        env['RESUME'] = '1'
    log(f'launch {variant} (CACHE_SEED={seed}{", RESUME=1" if resume else ""})')
    r = sh(['bash', setup, variant], env=env)
    out = (r.stdout or '') + (r.stderr or '')
    m = re.search(r'pid=(\d+)', out)
    pid = int(m.group(1)) if m else None
    tail = [l for l in out.strip().splitlines() if l.strip()][-1:] or ['(no output)']
    log('  ' + tail[0][:160])
    return pid, 'ok'


def wait_start(run_dir, pid, timeout, label):
    logs = glob.glob(os.path.join(run_dir, 'train107*.log'))
    t0 = time.time()
    while time.time() - t0 < timeout:
        if logs and os.path.exists(logs[0]):
            try:
                txt = open(logs[0], errors='ignore').read()
            except OSError:
                txt = ''
            if START_MARKER in txt:
                log(f'{label} 到达 {START_MARKER!r}')
                return 'ok'
        if not alive(pid):
            return 'dead'
        time.sleep(20)
    return 'timeout'


def copy_cache_to_master(builder_dir, master):
    srcs = glob.glob(os.path.join(builder_dir, 'cache107*'))
    if not srcs:
        log(f'!! {builder_dir} 下没有缓存目录')
        return False
    os.makedirs(master, exist_ok=True)
    r = sh(['cp', '-a', srcs[0] + '/.', master + '/'])
    log(f'复制缓存 {srcs[0]} -> {master} {"OK" if r.returncode == 0 else "FAILED"}')
    return r.returncode == 0


def ensure_run(setup, variant, seed, timeout, label):
    """启动一个 run 并等到 Start training...，失败自动 RESUME 重启（--retry 次）。
    首次若检测到暂停态目录（有代码无进程）→ 直接 RESUME（保留半成品缓存/掩码）。"""
    first_resume = paused_run_dir(variant)
    if first_resume:
        log(f'{label}: 检测到暂停态目录（无进程），首启即 RESUME')
    for attempt in range(1, args.retry + 2):
        pid, st = launch(setup, variant, seed, resume=(first_resume or attempt > 1))
        if st == 'skip':
            return 'skipped'
        run_dir = os.path.join(HOME, f'project-107-{variant}')
        st = wait_start(run_dir, pid, timeout, f'{label}[{attempt}]')
        if st == 'ok':
            return 'ok'
        log(f'{label} 第 {attempt} 次尝试 -> {st}' +
            ('，将以 RESUME 模式重试' if attempt <= args.retry else '，放弃'))
    return 'failed'


def main():
    global args
    ap = argparse.ArgumentParser()
    ap.add_argument('variants', nargs='+')
    ap.add_argument('--stagger', type=int, default=900, help='错峰秒数（默认 900）')
    ap.add_argument('--retry', type=int, default=2)
    ap.add_argument('--master', default=os.path.join(HOME, 'cache107_master'))
    ap.add_argument('--setup', default=os.path.join(HOME, '-project', 'setup_exp.sh'))
    args = ap.parse_args()

    variants = args.variants
    log(f'variants={variants} stagger={args.stagger}s retry={args.retry} master={args.master}')

    # ---- 阶段 1：确保 master 缓存就绪（builder 顺带构建）----
    if master_ready(args.master):
        log('master 缓存已就绪，跳过 builder 阶段')
    else:
        builder = variants[0]
        log(f'master 未就绪 -> builder: {builder}（构建图缓存 + 离群点掩码，~40-50 分钟）')
        st = ensure_run(args.setup, builder, args.master, BUILD_TIMEOUT, 'builder')
        if st != 'ok':
            log('ABORT: master 缓存构建失败')
            sys.exit(1)
        if not copy_cache_to_master(os.path.join(HOME, f'project-107-{builder}'), args.master):
            log('ABORT: 缓存复制失败')
            sys.exit(1)

    # ---- 阶段 2：其余 run 错峰启动 ----
    for i, v in enumerate(variants[1:], start=1):
        if i == 1:
            log('master 就绪，60s 后启动第 1 个后续 run')
            time.sleep(60)
        else:
            log(f'错峰 {args.stagger}s 后启动 {v}')
            time.sleep(args.stagger)
        st = ensure_run(args.setup, v, args.master, BUILD_TIMEOUT, v)
        log(f'{v}: {st}')

    # ---- 阶段 3：收尾状态 ----
    log('=== 编排完成，各 run 状态 ===')
    for v in variants:
        logf = glob.glob(os.path.join(HOME, f'project-107-{v}', 'train107*.log'))
        last = ''
        if logf and os.path.exists(logf[0]):
            try:
                lines = open(logf[0], errors='ignore').read().strip().splitlines()
            except OSError:
                lines = []
            last = lines[-1][:120] if lines else ''
        log(f'{v}: {"训练中/已启动" if last else "无日志"} | {last}')
    log('监控: pgrep -af "main[.]py" ; tail -5 ~/project-107-*/train107*.log')


if __name__ == '__main__':
    main()
