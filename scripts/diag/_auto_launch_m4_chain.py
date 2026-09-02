#!/usr/bin/env python3
"""_auto_launch_m4_chain.py — m4 训练链自动编排（16.11.9）

链：等 v2iaa42m4 建图完成 → 启动 v2iag42m4（GBDT15，fresh）→ 错峰后启动 v2wave42m4（wave 教师，为 ia 蒸馏准备）
条件：内存/CPU 错峰（建图峰值错开），磁盘余量检查。

用法（服务器 nohup 后台）：
  nohup ~/venv/bin/python3 scripts/diag/_auto_launch_m4_chain.py > ~/m4_chain.log 2>&1 &
"""
import os
import re
import subprocess
import sys
import time

HOME = os.path.expanduser('~')
LOG = os.path.join(HOME, 'm4_chain.log')
DATA_BATCHES = 'batch_v2_full,batch_v2_rest,batch_v2_m4'

def log(msg):
    line = f"[{time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')

def run(cmd, timeout=300):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() + r.stderr.strip()
    except Exception as e:
        return f'ERR {e}'

def dir_training_started(name):
    """run 目录是否已开始训练(出现 Start training 或 Epoch)"""
    p = os.path.join(HOME, f'project-107-{name}', f'train107{name}.log')
    if not os.path.exists(p):
        return False
    with open(p, errors='replace') as f:
        t = f.read()
    return ('Start training' in t) or ('Epoch 0' in t) or ('Epoch 1' in t) or ('Epoch 00' in t)

def mem_ok(min_gb=20):
    """可用内存 >= min_gb"""
    try:
        out = run("free -g | awk 'NR==2{print $7}'")
        return int(float(out.strip())) >= min_gb
    except Exception:
        return True

def disk_ok(min_gb=15):
    try:
        out = run("df -h / | tail -1 | awk '{print $4}'")
        s = out.strip()
        gb = float(re.sub(r'[Gg]', '', s))
        return gb >= min_gb
    except Exception:
        return True

def launch(name):
    """用 setup_exp.sh fresh 启动"""
    log(f'启动 {name} ...')
    out = run(f"cd {HOME}/-project && DATA_BATCHES='{DATA_BATCHES}' nohup bash setup_exp.sh {name} > {HOME}/setup_{name}.log 2>&1 &", timeout=10)
    log(f'{name} 启动命令已发出: {out[:100]}')
    # 确认 main.py 进程出现
    for _ in range(12):
        time.sleep(10)
        c = run(f"ps -eo pid,cmd --user $(whoami) | grep '[m]ain.py' | grep '{name}' | wc -l")
        if c.strip() and int(c.strip()) > 0:
            log(f'{name} 进程已确认')
            return True
    log(f'{name} 进程未确认(可能还在 clone)')
    return False


def main():
    log('=== m4 训练链编排启动 ===')
    log(f'目标链: 等 v2iaa42m4 建图 → v2iag42m4 → v2wave42m4')

    # 阶段1: 等 v2iaa42m4 建图完成(训练开始)
    log('阶段1: 等 v2iaa42m4 开始训练(建图完成)...')
    waited = 0
    while not dir_training_started('v2iaa42m4'):
        time.sleep(60)
        waited += 1
        if waited % 30 == 0:
            log(f'  仍等待 v2iaa42m4 建图 ({waited*60//3600}h)')
        if waited > 180:  # 3h 上限
            log('WARN: v2iaa42m4 3h 未开始训练, 继续等待...')
            waited = 0
    log('v2iaa42m4 已开始训练')

    # 阶段2: 错峰后启动 v2iag42m4
    log('阶段2: 错峰 10 分钟(等 v2iaa42m4 建图峰值过)')
    time.sleep(600)
    if not mem_ok(20):
        log('WARN: 内存不足 20G, 再等 5 分钟')
        time.sleep(300)
    launch('v2iag42m4')

    # 阶段3: 错峰后启动 v2wave42m4
    log('阶段3: 错峰 20 分钟(等 v2iag42m4 建图峰值过)')
    time.sleep(1200)
    if not mem_ok(20):
        log('WARN: 内存不足 20G, 再等 10 分钟')
        time.sleep(600)
    if not disk_ok(15):
        log('WARN: 磁盘不足 15G, 停止(避免 OOM)')
        return
    launch('v2wave42m4')

    log('=== 链编排完成: v2iaa42m4 + v2iag42m4 + v2wave42m4 均应运行 ===')


if __name__ == '__main__':
    main()
