"""把多个 sc_expansion 宏表**合并**成一张 —— 不是覆盖。

## 为什么不能覆盖（2026-09-20 实测，`_t_v3_loader_check.py` §5）

两张表的 cell 名域**近乎互斥**：

| 表 | 对 V3 名域（1,049 个） | 对 V2 名域（3,586 个） |
|---|---|---|
| V3 自带表（1,049 条） | **100.0%** | **0.3%** |
| 共享表（24,625 条） | 8.3% | **100.0%** |

训练读的是**硬编码共享路径** `data/sc_expansion.json`（`graph_builder._load_sc_expansion`），
于是两条路都走不通：

* **只覆盖** ⇒ V2 数据加载被打崩（可展开率 100% → 0.3%），V2 复现性毁掉；
* **只保留** ⇒ V3 有 91.7% 的 SC_ 宏落进 `gate_struct` 的兜底
  （`logic='COMPLEX', n_t=6.0, …`），`STRUCT_MODE='base'` 的 n_transistors 近乎常数。

⇒ 取**并集**，两边都 100%。

## 用法

    # 只报数，不写任何文件（默认）
    python3 scripts/merge_sc_expansion.py --base data/sc_expansion.json \
                                          --add  data/v3_delivery/sc_expansion.json

    # 写到新文件
    ... --out data/sc_expansion_merged.json

    # 原地覆盖 base（**训练树里由 setup_exp.sh 调用**；仓库里不要这么干）
    ... --inplace

⚠ 仓库里**不要**提交合并后的表：union ≈ 96 MiB，逼近 GitHub 100 MiB 硬限，
且任一来源表更新都得重合并。共享表保持 V2 内容（V2 运行逐字节不变），
V3 运行在 `setup_exp.sh` 里当场合并。

幂等：`--inplace` 重复跑是安全的（dict 更新，并集不变）。
"""
import argparse
import json
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


def load(path):
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, help='基准表（后加的只补不删）')
    ap.add_argument('--add', action='append', required=True,
                    help='要并入的表；可重复给多个')
    ap.add_argument('--out', help='写到这个路径')
    ap.add_argument('--inplace', action='store_true', help='原地覆盖 base')
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()
    if a.out and a.inplace:
        print('ERROR: --out 与 --inplace 只能给一个')
        return 2

    base = load(a.base)
    print(f'base  {a.base}: {len(base)} 条')
    merged = dict(base)
    total_conflict = 0
    for p in a.add:
        add = load(p)
        common = [k for k in add if k in merged]
        conflict = [k for k in common if merged[k] != add[k]]
        new = len(add) - len(common)
        merged.update(add)                      # 新表获胜（更新的口径）
        print(f'add   {p}: {len(add)} 条  →  新增 {new}  同名 {len(common)}  '
              f'其中内容不同 {len(conflict)}')
        if conflict:
            total_conflict += len(conflict)
            print(f'      ⚠ 同名但内容不同，已取 add 的版本: {conflict[:5]}'
                  f'{" …" if len(conflict) > 5 else ""}')
    print(f'union: {len(merged)} 条（base {len(base)} + 净增 {len(merged) - len(base)}）')
    print(f'同名冲突合计 {total_conflict} 处')

    if a.inplace:
        dst = a.base
    elif a.out:
        dst = a.out
    else:
        print('（未指定 --out/--inplace，只报数不写）')
        return 0
    with open(dst, 'w', encoding='utf-8') as fh:
        json.dump(merged, fh, ensure_ascii=False)
    print(f'已写 {dst}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
