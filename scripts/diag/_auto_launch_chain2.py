#!/usr/bin/env python3
"""_auto_launch_chain2.py — 训练链编排 v2（16.11.11，防误杀版）

链：等 v2iag42m4 开始训练（建图完成、内存回落）→ 启动 v2wave42m4（wave 教师）→ 错峰后启动 v2nowave42m4 重训
安全：所有进程操作前 readlink /proc/<pid>/cwd 确认归属（§8.2 教训）；
      只按 cwd 精确匹配目标 run 目录，绝不猜 PID。

用法（服务器 nohup 后台）：
  nohup ~/venv/bin/python3 scripts/diag/_auto_launch_chain2.py > ~/chain2.log 2>&1 &
"""
import os
import re
import subprocess
import time

HOME = os.path.expanduser('~')
LOG = os.path.join(HOME, 'chain2.log')
DATA_BATCHES = 'batch_v2_full,batch_v2_rest,batch_v2_m4'

def log(msg):
    line = f"[{time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')

def run(cmd, timeout=120):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() + r.stderr.strip()
    except Exception as e:
        return f'ERR {e}'

def procs_by_cwd(name):
    """返回 cwd 匹配 <name> 的所有 PID（安全，逐进程查 cwd）"""
    pids = []
    try:
        out = run("ps -eo pid --user $(whoami) | tail -n +2")
        for line in out.splitlines():
            pid = line.strip()
            if not pid.isdigit():
                continue
            cwd = os.path.realpath(f'/proc/{pid}/cwd')
            if name in cwd:
                pids.append(pid)
    except Exception:
        pass
    return pids

def training_started(name):
    p = os.path.join(HOME, f'project-107-{name}', f'train107{name}.log')
    if not os.path.exists(p):
        return False
    try:
        with open(p, errors='replace') as f:
            t = f.read()
    except Exception:
        return False
    return ('Start training' in t) or bool(re.search(r'Epoch \d+', t))

def mem_avail_gb():
    try:
        out = run("free -g | awk 'NR==2{print $7}'")
        return int(float(out.strip()))
    except Exception:
        return 99

def launch(name):
    log(f'启动 {name} ...')
    # 先确认无同目录残留进程（防双写）
    existing = procs_by_cwd(name)
    if existing:
        log(f'WARN: {name} 已有进程 {existing}，跳过启动')
        return False
    run(f"cd {HOME}/-project && DATA_BATCHES='{DATA_BATCHES}' nohup bash setup_exp.sh {name} > {HOME}/setup_{name}.log 2>&1 &", timeout=15)
    # 确认启动（最多等 5 分钟 clone）
    for _ in range(30):
        time.sleep(10)
        if procs_by_cwd(name):
            log(f'{name} 进程已确认（cwd 匹配）')
            return True
    log(f'WARN: {name} 5 分钟内未见进程')
    return False

def main():
    log('=== 训练链编排 v2 启动（防误杀版）===')
    log('链: v2iag42m4 训练中 → v2wave42m4 → v2nowave42m4 重训')

    # 阶段1: 等 v2iag42m4 开始训练（建图完成、内存回落）
    log('阶段1: 等 v2iag42m4 开始训练(建图完成)...')
    while not training_started('v2iag42m4'):
        time.sleep(120)
        log(f'  等待 v2iag42m4（内存可用 {mem_avail_gb()}G）')
    log('v2iag42m4 已开始训练')

    # 阶段2: 错峰后启动 v2wave42m4
    log('阶段2: 错峰 15 分钟')
    time.sleep(900)
    if mem_avail_gb() < 20:
        log('WARN: 内存 <20G，再等 10 分钟')
        time.sleep(600)
    launch('v2wave42m4')

    # 阶段3: 错峰后启动 v2nowave42m4 重训
    log('阶段3: 错峰 20 分钟（等 v2wave42m4 建图峰值过）')
    time.sleep(1200)
    if mem_avail_gb() < 15:
        log('WARN: 内存 <15G，再等 15 分钟')
        time.sleep(900)
    log('重建 v2nowave42m4（重训）...')
    # 重训 = fresh（删旧目录，保留数据种子从 ~/-project 重拷）
    run(f"rm -rf {HOME}/project-107-v2nowave42m4", timeout=60)
    launch('v2nowave42m4')

    log('=== 编排完成: v2iag42m4 + v2wave42m4 + v2nowave42m4 应均运行 ===')

if __name__ == '__main__':
    main()
