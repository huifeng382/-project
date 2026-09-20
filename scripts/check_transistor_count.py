#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""启动前自检：`transistor_count` 是否与 netlist 自洽（docs/V3_ISSUES.md P1-1）。只读。

**为什么要这条**：`transistor_count` 同时是**输入特征**（`src/data_loader.py:322`）与
`struct_prior[0]`（`:716`），而且 `struct_encoder` **没有任何输入归一化**
（`src/model.py:50-54`）。V3 实测：371 种 netlist（占 5,900 种的 6.3%）**同一个 netlist
文本被赋予了多个不同的 `transistor_count`**，涉及 1,847 电路 = **14.8%**，同 netlist 内
极差中位 6 / p90 16 / max 32。⇒ 等于给模型注入「两个延迟相同、图结构相同的样本，尺寸却不同」
的噪声，直接对抗排序目标。

**口径与 data_loader 一致**：网表列优先取 `gate_level_netlist_std`（`data_loader.py:80-82`
会 drop 掉原始 `gate_level_netlist` 并改名），这样本检查看到的就是**模型看到的那张网表**。

⚠ **默认只告警、不阻断（退出码恒为 0）**，这是刻意的：
  当前 V3 数据**确实命中 14.8%**（已记档的缺陷）。写成硬失败会把**每一次 V3 运行**都挡在
  门外，而它挡下的是**已经知道的事**，不是新信息 —— 检查的价值在于「下一版别再犯」和
  「将来某次运行突然变好/变坏要看得见」，不在于拦住今天这一次。
  生成方给出该列的确切定义、使其成为电路内容的函数之后，本检查应当**恒静默**；
  到那时（且只有在到那时）才考虑 `--strict`。

用法（一般由 setup_exp.sh 调用）：
  python scripts/check_transistor_count.py --batches v3_delivery
  python scripts/check_transistor_count.py --batches batch_v2_full,batch_v2_m4
  python scripts/check_transistor_count.py --batches v3_delivery --strict   # 命中即退出 1
"""
import os
import sys
import argparse
from collections import defaultdict

import pandas as pd
import pyarrow.parquet as pq

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def pick_netlist_col(names):
    """口径同 data_loader.py:80-82。"""
    if 'gate_level_netlist_std' in names:
        return 'gate_level_netlist_std'
    if 'gate_level_netlist' in names:
        return 'gate_level_netlist'
    return None


def check_batch(batch):
    path = os.path.join(ROOT, 'data', batch, 'circuit_static.parquet')
    if not os.path.exists(path):
        print('  %-16s ⚠ 缺 circuit_static.parquet，跳过' % batch)
        return None
    names = pq.ParquetFile(path).schema_arrow.names
    col = pick_netlist_col(names)
    if col is None:
        print('  %-16s ⚠ 无网表列（%s），跳过' % (batch, ', '.join(names[:6])))
        return None
    if 'transistor_count' not in names:
        print('  %-16s ⚠ 无 transistor_count 列，跳过' % batch)
        return None

    df = pd.read_parquet(path, columns=['circuit_id', col, 'transistor_count'])
    df['cid'] = df['circuit_id'].astype(str)
    df['tc'] = pd.to_numeric(df['transistor_count'], errors='coerce')

    g = df.groupby(col)['tc']
    nuniq = g.nunique()
    spread = g.max() - g.min()
    n_circ_all = df['cid'].nunique()

    bad = nuniq[nuniq > 1].index
    if len(bad) == 0:
        print('  %-16s ✅ 自洽：%d 种 netlist / %d 电路，无一种 netlist 对应多个 '
              'transistor_count' % (batch, len(nuniq), n_circ_all))
        return {'batch': batch, 'col': col, 'bad_nl': 0, 'bad_circ': 0,
                'n_nl': len(nuniq), 'n_circ': n_circ_all}

    bad_df = df[df[col].isin(bad)]
    n_bad_circ = bad_df['cid'].nunique()
    sp = spread[nuniq > 1]
    print('  %-16s ⚠ **不自洽**：%d/%d 种 netlist（%.1f%%）对应多个 transistor_count；'
          '涉及 %d/%d 电路 = **%.1f%%**'
          % (batch, len(bad), len(nuniq), 100.0 * len(bad) / max(len(nuniq), 1),
             n_bad_circ, n_circ_all, 100.0 * n_bad_circ / max(n_circ_all, 1)))
    print('  %-16s    同 netlist 极差：中位 %.0f / p90 %.0f / max %.0f（列 = %s）'
          % ('', sp.median(), sp.quantile(.90), sp.max(), col))
    # 「该文本被几个电路共用」—— 用来一眼分清两种病（见 V3_ISSUES P1-1 新证据二）：
    #   · 共用数大、延迟全同 ⇒ 错在**这一列**（本脚本能看出来）
    #   · 另有少量「六列静态全同、延迟不同」⇒ 错在**交付记录缺维度**（要看时序弧，本脚本不看）
    share = bad_df.groupby(col)['cid'].nunique()
    print('  %-16s    命中 netlist 被共用的电路数：中位 %.0f / p90 %.0f / max %.0f'
          % ('', share.median(), share.quantile(.90), share.max()))
    print('  %-16s    ⚠ 其中「静态六列全同、时序弧 DELAY 却不同」的那一类**本脚本看不出来**，'
          % '')
    print('  %-16s      要跑 `scripts/diag/_t_v3_tc_cause.py`（P1-1 新证据二）' % '')
    print('  %-16s    ⚠ 该列同时是输入特征与 struct_prior[0]，且 struct_encoder 无归一化'
          % '')
    print('  %-16s    （V3_ISSUES P1-1；只告警不阻断）' % '')
    return {'batch': batch, 'col': col, 'bad_nl': len(bad), 'bad_circ': n_bad_circ,
            'n_nl': len(nuniq), 'n_circ': n_circ_all}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batches', default=os.environ.get('DATA_BATCHES', 'v3_delivery'),
                    help='逗号分隔的数据集名（默认取 $DATA_BATCHES）')
    ap.add_argument('--strict', action='store_true',
                    help='命中即退出码 1（默认只告警；见文件头「为什么默认不阻断」）')
    a = ap.parse_args()

    batches = [b.strip() for b in a.batches.split(',') if b.strip()]
    print('[check_transistor_count] P1-1 自检：%s' % ', '.join(batches))
    hits = 0
    for b in batches:
        r = check_batch(b)
        if r and r['bad_nl']:
            hits += 1
    if hits == 0:
        print('[check_transistor_count] ✅ 全部自洽（或全部跳过）')
    else:
        print('[check_transistor_count] ⚠ %d 个数据集命中不自洽 —— 只告警，训练继续'
              '（理由见 scripts/check_transistor_count.py 文件头）' % hits)
    if a.strict and hits:
        sys.exit(1)
    sys.exit(0)


if __name__ == '__main__':
    main()
