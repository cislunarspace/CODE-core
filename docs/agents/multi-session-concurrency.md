# 多会话并发与评审取修订的防护约定

同一 clone 被多个会话（agent 与 agent，或人与 agent）并发驱动时的操作约定。来源：issue #699，记录自 #681/#682/#683 合并期间的实测事故。

## 适用场景

以下任一条成立即适用本约定：

- 同一个 clone / worktree 由多个 agent 会话（或人与 agent）同时驱动。
- 会话之间会切换分支、`git fetch`、提交或推送。
- 一个会话在评审某分支，同时别的会话可能改动同一 clone 的 HEAD 或工作区。

## 实测事故

| 事故 | 现象 | 后果 |
| --- | --- | --- |
| 评审取错修订 | re-review 的 agent 读的是**本地工作区文件**（停在旧修订 `fa16a0d`，不含被审改动），却按 `origin/<branch>`（`47062ac`）的 diff 出报告 | 得出「声称的修复不存在」的错误结论，白跑一轮评审；另一 agent 读 ref 后结论截然相反 |
| 跨会话分支改写 | 同一 clone 内另一会话把被审分支从当前分支切出并在其上提交；本会话的 `git commit --amend` 落到对方那次提交（`e10b578`）上 | 不属于本会话的文件被混入；随后只能用 plumbing（`git update-ref` + `read-tree`/`write-tree`/`commit-tree`）复原对方提交并把本次修复另立提交 |
| 推送竞态 | `gh stack link` 首次以「非快进」失败（本地落后远端 1 分钟前的新 tip） | 需先 `git fetch`，再重试同一命令才成功 |

## 根因

- 会话的**工作区 / HEAD** 与**被审或被改的修订**不是同一对象：存在多 worktree，分支会被并发切换。
- `--amend` / `reset` / `rebase` / `push --force*` / `gh stack *` 默认作用于**当前 HEAD**，而 HEAD 可能已被其它会话改写。
- 评审提示词未强制「从 ref 读取」，agent 会顺手读工作区文件。

## 约定

### 1. 评审/审查以 ref 为准

- 下发评审任务时显式给出修订 ref（如 `origin/<branch>` 或具体 commit），并注明「工作区不等于被审修订，禁止读工作区文件」。
- 只允许通过 ref 读取：

  ```
  git show <ref>:<path>
  git diff <base>...<head>
  git log <base>..<head>
  ```

- 需要 checkout 才能跑验证时，用独立 worktree（`git worktree add <dir> <ref>`），不切当前 clone 的分支。

### 2. 改写分支前确认归属

- 改写（`--amend` / `reset` / `rebase` / `push --force*`）前先 `git fetch`，再确认 HEAD 就是目标提交：

  ```
  git reflog -5
  git log -1
  ```

- 若 `git log -1` 或 `reflog` 显示 HEAD 与预期不符，停止改写，先查明是哪个会话切了分支。
- 远端同步只允许 `--force-with-lease`，禁止裸 `--force`。

### 3. 共享 clone 的并发写走 plumbing

- 在可能被并发切分支的 clone 上，优先用 plumbing 提交：`git read-tree` / `git write-tree` / `git commit-tree` / `git update-ref`，在不切分支、不动工作区的前提下落提交。
- 避免在共享 clone 上直接 `--amend`；若必须改写历史，先确认没有其它会话依赖当前分支。
- plumbing 写完后，用 `git log -1 <branch>` 核对新 tip 指向预期提交。

### 4. 脏工作区不共享

- 被多会话复用的 worktree 不留长驻未提交改动；交还前提交或暂存（`git commit` 或 `git stash`）。
- 交接时在任务说明里写清当前分支、HEAD 与是否有未提交改动。

## 提交/评审前自检清单

- [ ] 评审任务已给出明确 ref，并注明「从 ref 读取，禁止读工作区」。
- [ ] 评审结论来自 `git show` / `git diff <base>...<head>` / `git log <base>..<head>`，不是工作区文件。
- [ ] 改写历史前已 `git fetch`，并用 `git reflog -5` 与 `git log -1` 确认 HEAD 归属。
- [ ] 推送用 `--force-with-lease`，未使用裸 `--force`。
- [ ] 共享 clone 的写入走 plumbing，未在并发环境直接 `--amend`。
- [ ] 交还 worktree 前工作区干净（已提交或已暂存）。

## 验收

约定写入仓库文档；下一次多会话并发中不再出现「评审取错修订」或「amend 误落他人提交」。相关来源：issue #699。
