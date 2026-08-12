# 会话交接：物理常数独立管理（#377）+ 禁止静默回退（#378）

日期：2026-08-10 → 2026-08-11
分支：master
仓库：/home/ouyangjiahong/codes/e2m2e

## 一句话现状

#377 阶段 1/2/3 已提交，**阶段 4（μ 切 DE421）改动已全部做完但尚未提交**——正处于"验证测试全绿后提交"这一步被打断。

## 环境前置（每次跑 Rust/测试前必须）

```bash
cd /home/ouyangjiahong/codes/e2m2e
export CSPICE_DIR=$(python3 scripts/download_cspice.py --print-cspice-dir)
export LIBCLANG_PATH=/usr/lib/llvm-21/lib
# 或直接走 make（Makefile 会自动配这两个变量）：make dev / make check
```

libclang 在 `/usr/lib/llvm-21/lib`（系统是 libclang-21.so，非 libclang.so，Makefile 已改为兼容探测——这是阶段 2 的改动之一）。

## 已提交（4 个 commit，按序）

| commit | 内容 |
|---|---|
| 41515db | docs(adr): ADR 0022 物理常数独立管理 |
| 67e2c1c | feat: #377 阶段1 新增 data/constants 常数层 |
| f0c9c26 | feat: #377 阶段2 Rust/Python 同源（constants.toml + build.rs 生成） |
| 825f149 | refactor: #377 阶段3 收编算法层散落常数到 data/constants |

## 未提交：阶段 4（μ 切 DE421）——下一步就是验证并提交它

**已完成的改动**（工作区，未提交）：
- 源码默认 μ 全切 `Datum.DE421.mu`（= 0.012150585350562453）：`data/templates/seed.py`、`station_keeping/monte_carlo.py`、`dynamics/bcr4bp_system.py`、`dynamics/cr3bp_system.py`、`tests/conftest.py` 等。
- 37 个测试文件的断言已按新 μ 更新。
- `grep 0.0121506683` 无代码残留（仅 bcr4bp_system.py docstring 说明文字）。

**阶段 4 提交前必做**：
1. **跑测试确认全绿**。`tests/numerical` 已验证 609 passed（builder 报告）。**还需确认**：`tests/algorithm`、`tests/api`、`tests/tools` 等目录。注意：整套 `pytest tests/` 会因某个 slow/长积分测试卡住超时，**分目录跑、各加 `timeout`**：
   ```bash
   uv run pytest tests/algorithm -q -p no:cacheprovider
   uv run pytest tests/api tests/tools -q -p no:cacheprovider
   ```
   （numerical 约 173s，algorithm 各子目录单独跑。）
2. **补 CHANGELOG.md**：在 Changed 段加一条——默认地月 μ 从 1965 旧值 0.0121506683 切换到 DE421 0.012150585350562453，CR3BP 数值结果会变。（builder 漏了这条，必须补。）
3. **提交时排除两类文件**：
   - **CLAUDE.md**：用户手动加的 `## Python`（uv 管理虚拟环境）段，不是本任务内容，**不纳入提交**，保留。
   - **LPO 工作线遗留**（会话开始前就存在，与 #377 无关，**绝不提交**）：
     `crates/e2m2e-integrators/src/{lib,design_lpo,lpo_correction,lpo_family}.rs`、`e2m2e/integrators.py`、`e2m2e/algorithm/family/cr3bp_orbits.py`、`crates/e2m2e-integrators/abi-version.txt`、`archive/plans/lpo-rust-rayon.md`、`tests/algorithm/family/test_lpo_rust_equivalence.py`

## 两条已定决策（贯穿后续）

1. **μ 全部统一切 DE421，不留旧值**（一刀切，不提供 legacy 复现选项）。
2. **BCR4BP 太阳参数（MU_SUN/SUN_DISTANCE/SUN_OMEGA 无量纲量）保留 Topputo 文献约定不动**，只切 μ；`normal_form/constants.py` 的 qiao μ（0.012150585609624）是独立模型约定，不动。

## 待办：#378（禁止静默回退）——#377 提交后开工

- issue #378 已开（ready-for-agent）：**必须确保 Rust 扩展可用，不允许在需要用 Rust 时回退 Python**。spice 默认启用下，扩展不可用应显式报错（含 `make dev` 指引），不静默降级；测试环境同样禁止静默回退。
- **背景**：现有 14 处 `except ImportError → _HAS_RUST_*=False` 静默降级（bcr4bp_dynamics、dynamics、force_model 等），外加力模型 `compute_acceleration` 的 Python 回退警告。ADR 0002 line 97 曾把防御性回退记为"有意保留"，本决策改变这点。
- **issue 里有 3 个待澄清点**（实现前需用户定）：① Python 参考实现去留（倾向保留为显式对照、移出默认路径）；② 无 spice 极简环境是否还支持；③ ADR 0002 有意保留的 scipy 事件检测路径如何与静默降级区分。
- **注意**：#378 会改 force_model/dynamics 分发逻辑，与 #377 阶段 4 改的测试有交集，故排在 #377 提交之后，不要并行。

## 工作方法（loop-go）

- 本任务用 builder（写代码）+ checker（跑检查）循环驱动，每阶段一个 commit。红线：不弱化/删除/跳过测试、不碰 LPO 遗留、checker 报告原样转发 builder。
- 检查命令集（来自 CLAUDE.md 提交前检查）：`uv run ruff check e2m2e/ tests/`、`uv run ruff format --check ...`、`uv run mypy e2m2e/ --ignore-missing-imports`、`cargo fmt --all -- --check`、`cargo clippy --workspace -- -D warnings`、pytest。
- **教训**：跑测试务必分目录+超时，整套 `pytest tests/` 会卡死；builder 自报结果要用 checker 独立复核（本次多次靠 git status/diff 甄别出 builder 误报）。

## 关键背景（防误读）

- **"走 Python 回退路径，应优先走 Rust 编译路径"是既有 DeprecationWarning**，出现在测试直接调 `force.compute_acceleration()` 时——`ForceModel.propagate()` 才有 Rust 分发。这不是 bug、不是 git 回退，是 #378 要处理的设计问题。
- **GitHub issue 用 `gh` CLI**，仓库 cislunarspace/e2m2e；#377=物理常数，#378=禁止静默回退。
