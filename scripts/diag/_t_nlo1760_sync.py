"""_t_nlo1760_sync.py — 打 NetlistOpt 17.6.0 的传输包 + 全树 sha1 清单（本地临时，不入库）。

三件产物（与 17.5.0 那版同构，只换了文件集与命名）：
  1) tar  = 17.6.0 相对服务器现状**改动的 5 个文件**（增量同步，不动服务器上其余文件）；
  2) sha1 = **整棵共享树**（`src/` `tests/` `Cargo.toml`，即驱动脚本 RUST_FP 的那 55 个文件）的
     期望校验码清单，供服务器 `sha1sum -c` 一条命令既验同步、又**报漂移**；
  3) 打印本地侧 RUST_FP（= 驱动 RUN_INFO 里那个 12 位指纹），同步后与服务器算出的比对。

⚠ 17.6.0 相对 17.5.0 **多带一个非 Rust 文件**：`src/process_template/xyce.sh`。
   单次仿真超时（`XYCE_TIMEOUT_S` ⇒ 退出码 42）就加在这个模板里，它是 `RUST_FP` 那 55 个文件之一；
   **漏传它就等于只改 Rust 不改脚本 ⇒ 超时静默失效**（Rust 侧永远收不到 42）。

为什么要绕这一下：本地工作区是 CRLF（core.autocrlf=true），直接 tar 会把 CRLF 带到服务器。
本脚本把内容规范成 **LF**，使包内每个成员的 sha1 字面等于 `git show <rev>:<path> | sha1sum`
（**内容 sha1**，不是 `git hash-object` 的对象 id —— 拿后者当期望值会得出"全不符"的假警报）。

⚠ 别用 `tr -d '\r'` 做归一化：本机 Git Bash 下实测是空操作，会静默产出 CRLF 包。
⚠ 必须在**内层 commit 之后**跑：FILES 里任何一个还没入库（`git show` 取不到 blob）⇒ 期望值缺失，
   本脚本会直接判 FAIL 并中止（不给「未验证成员」发车）。
"""
import hashlib
import os
import subprocess
import sys
import tarfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NLO = os.path.join(REPO, 'NetlistOpt')
REV = 'HEAD'
# 17.6.0 的改动集：4 个 Rust/模板文件 + 1 个既有测试（它带 17.5.0 + 17.6.0 两轮改动，
# 因为服务器还没吃过 17.5.0 的包）。增量同步，不整体覆盖服务器树。
FILES = [
    'src/simulation.rs',
    'src/tl_opt.rs',
    'src/process_template/xyce.sh',
    'tests/xyce_timeout.rs',
    'tests/tl_opt_gnn_batch.rs',
]
STAGE = os.path.join(os.environ.get('TEMP', 'C:/Windows/Temp'), 'nlo1760_lf')
# 产物落到**项目内**的 sync_out/（已进 .gitignore）。原先硬编码在桌面，用户要求别再往桌面丢东西。
# 放项目内还有个理由：PreToolUse 的 fence-project.ps1 本就禁止在项目目录外写文件，桌面是绕过它。
OUT_DIR = os.path.join(REPO, 'sync_out')
os.makedirs(OUT_DIR, exist_ok=True)
OUT_TAR = os.path.join(OUT_DIR, 'nlo1760.tar.gz')
OUT_SHA = os.path.join(OUT_DIR, 'nlo1760.sha1')


def lf(path):
    return open(path, 'rb').read().replace(b'\r\n', b'\n')


# 本机 Windows 控制台默认 GBK，直接 print 中文会变成乱码（17.5.0 那版就是这样）。
# 显式把 stdout 定成 UTF-8：Git Bash / WSL 侧照常显示，重定向进文件也不会掺 GBK。
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def blob_sha(rev, path):
    """`git show <rev>:<path> | sha1sum` == 内容 sha1（LF）。取不到返回 None。"""
    p = subprocess.run(['git', 'show', f'{rev}:{path}'], cwd=NLO,
                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return hashlib.sha1(p.stdout).hexdigest() if p.returncode == 0 else None


def tree_files():
    """复刻驱动的 `find src tests Cargo.toml -type f | LC_ALL=C sort`（按字节序）。"""
    out = []
    for root in ('src', 'tests'):
        for d, _, fs in os.walk(os.path.join(NLO, root)):
            for f in fs:
                rel = os.path.relpath(os.path.join(d, f), NLO).replace(os.sep, '/')
                out.append(rel)
    out.append('Cargo.toml')
    return sorted(out, key=lambda p: p.encode())


def fp_of(lines):
    """复刻 `... | xargs sha1sum | sha1sum | cut -c1-12`。"""
    return hashlib.sha1(''.join(lines).encode()).hexdigest()[:12]


def main():
    os.makedirs(STAGE, exist_ok=True)
    os.chdir(NLO)
    rev = subprocess.run(['git', 'rev-parse', '--short', REV], cwd=NLO, stdout=subprocess.PIPE,
                         text=True).stdout.strip()
    print(f'[1] 内层 rev = {rev}')

    # --- 0) 前置：FILES 必须全部已入库（否则期望值缺失 = 无法验证，不发车）---
    missing = [f for f in FILES if blob_sha(REV, f) is None]
    if missing:
        print(f'\n[0] FAIL —— 这些文件还没进 {rev}（先 commit 内层再跑本脚本）：')
        for f in missing:
            print(f'    {f}')
        sys.exit(1)
    print(f'[0] FILES 全部已入库（{len(FILES)} 个）')

    # --- 1) 增量包：FILES，LF 归一化，成员逐个验 ---
    print('\n[2] 增量包成员（期望值取 git blob 内容 sha1）')
    expect = {}
    for f in FILES:
        raw = lf(f)
        dst = os.path.join(STAGE, f)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        open(dst, 'wb').write(raw)
        got, want = hashlib.sha1(raw).hexdigest(), blob_sha(REV, f)
        expect[f] = want
        flag = 'OK' if got == want else 'MISMATCH'
        print(f'    {got}  {f}  [{flag}]  (git blob {want})')
    with tarfile.open(OUT_TAR, 'w:gz') as tar:
        for f in FILES:
            tar.add(os.path.join(STAGE, f), arcname='./' + f)
    print(f'    包 {OUT_TAR}  ({os.path.getsize(OUT_TAR)} bytes)')
    with tarfile.open(OUT_TAR) as tar:
        bad = [m.name for m in tar.getmembers() if m.isfile()
               and hashlib.sha1(tar.extractfile(m).read()).hexdigest()
               != expect.get(m.name.lstrip('./'))]
    print(f'    包内复验：{"全部 OK" if not bad else "不符 " + str(bad)}')
    if bad:
        sys.exit(1)

    # --- 2) 全树清单：既验同步也报漂移 ---
    files = tree_files()
    lines, drift, untracked = [], [], []
    for rel in files:
        h = hashlib.sha1(lf(rel)).hexdigest()
        lines.append(f'{h}  {rel}\n')
        b = blob_sha(REV, rel)
        if b is None:
            untracked.append(rel)
        elif b != h:
            drift.append(rel)
    open(OUT_SHA, 'w', newline='\n').write(''.join(lines))
    print(f'\n[3] 全树清单 {OUT_SHA}：{len(files)} 个文件')
    print(f'    工作区 != {rev} 的文件：{drift if drift else "（无 —— 清单即该 rev 的快照）"}')
    print(f'    未入库（工作区有、{rev} 无）：{untracked if untracked else "（无）"}')

    # --- 3) 本地 RUST_FP（LF 口径）---
    print(f'\n[4] 本地 RUST_FP（LF 口径，与驱动 RUN_INFO 同法）= {fp_of(lines)}')
    print('    同步后请在服务器跑同一条 find|sha1sum|sha1sum，两值必须逐位相同。')


if __name__ == '__main__':
    main()
