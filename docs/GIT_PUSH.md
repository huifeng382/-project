# Git 推送方法（长期有效，2026-08-26 实测定稿）

## 仓库布局

| 仓库 | 位置 | 远端 | 规则 |
|---|---|---|---|
| **project**（GNN 项目） | 工作区根目录 | `https://github.com/huifeng382/-project.git`，分支 `10.3.3-fix-earlystop` | **本地 + GitHub** |
| **NetlistOpt** | `NetlistOpt/`（独立 git 仓库；父仓库 `.gitignore` 已忽略） | **无** | **仅本地 git，不推远端**（2026-08-26 用户明确要求） |

## 推送方法（实测可用）

```bash
# 1) 推送前确认待推内容
git rev-list --left-right --count origin/<branch>...HEAD     # 输出 "0  <N>" 表示 N 个待推

# 2) 推送（必须带 openssl 后端）
git -c http.sslBackend=openssl push origin <branch>

# 3) 推送后验证同步
git rev-list --left-right --count origin/<branch>...HEAD     # 应为 "0  0"
git status --short                                            # 应为空（clean）
```

## 关键约束与踩坑记录

1. **必须用 openssl 后端**：Windows 默认 `schannel` 不稳定（SSL 报 `SEC_E_NO_CREDENTIALS`、连接被重置、crates.io/github 均踩过）。一律 `-c http.sslBackend=openssl`。

2. **凭据来源**：Windows 凭据管理器存有 `git:https://github.com`（用户名 huifeng382 + PAT）。正常情况 git 的 `manager` 助手自动提供，**不要把 token 写进 URL/命令/文件**（会留在进程列表与日志）。

3. **沙箱限制（本机 dsh 环境）**：沙箱禁命名管道 → Git for Windows 的 cygwin 助手（signal pipe）和凭据提示脚本无法启动，报错特征：
   - `couldn't create signal pipe, Win32 error 5`
   - `failed to execute prompt script (exit code 66)`
   - `could not read Username for 'https://github.com'`
   **解法**：用 `danger-full-access` 权限重试同一条 `git -c http.sslBackend=openssl push ...` 命令即可（无沙箱限制时凭据管理器正常工作，token 不暴露）。勿改用 CredRead P/Invoke 或 `git credential fill` 手撸凭据——沙箱下不可靠。

4. **GitHub 网络本身不稳定**：失败就重试（幂等），不要改命令。大批量推送前先看 `rev-list` 确认。

5. **沙箱自杀陷阱**：远程/服务器上 `pkill -f 'xxx'` / `pgrep -f 'xxx'` 会匹配到自己命令行（含 xxx 字样）→ 自杀断连。用 `pkill -9 -x <进程名>` 精确匹配，或 `[x]xx` 正则技巧。

## 提交规范（配合 15.2.5 文件归类）

- 文档一律 `docs/`，诊断脚本 `scripts/diag/`，报告 `reports/`，一次性输出不提交。
- NetlistOpt 与 project 是两个独立 git 仓库，各自 commit、互不包含。
