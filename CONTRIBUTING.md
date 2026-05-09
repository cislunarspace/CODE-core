# e2m2e 贡献指南

感谢您对 e2m2e 项目的关注！本指南涵盖贡献者工作流和开发者内部指南两部分内容。

---

## 第一部分：贡献者工作流

### 行为准则

在参与本项目时，请保持尊重和礼貌。我们欢迎各种形式的贡献：

- 报告 bug
- 提出新功能建议
- 改进文档
- 提交代码改进

详细行为准则请参阅 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)。

### 报告问题

#### 报告 Bug

1. **检查是否已存在**：在 [Issues](https://github.com/cislunarspace/e2m2e/issues) 中搜索是否已有类似问题
2. **创建新 Issue**：使用 Bug Report 模板
3. **提供详细信息**：
   - 清晰的标题
   - 详细的问题描述
   - 重现步骤
   - 预期行为 vs 实际行为
   - 环境信息（Python 版本、操作系统等）
   - 相关代码片段或截图

#### 建议新功能

1. **检查是否已建议**：在 Issues 中搜索是否已有类似建议
2. **创建新 Issue**：使用 Feature Request 模板
3. **详细说明**：功能描述、使用场景、可能的实现方案

### 提交代码

#### 准备工作

1. **Fork 仓库**：点击 GitHub 页面的 Fork 按钮
2. **克隆仓库**（替换 `<your-username>` 为你的 GitHub 用户名）：

   ```bash
   git clone https://github.com/<your-username>/e2m2e.git
   cd e2m2e
   ```

3. **添加上游仓库**：

   ```bash
   git remote add upstream https://github.com/cislunarspace/e2m2e.git
   ```

#### 开发流程

1. **同步最新代码**：

   ```bash
   git fetch upstream
   git checkout master
   git merge upstream/master
   ```

2. **创建功能分支**：

   ```bash
   git checkout -b feature/your-branch-name
   # 或修复 bug：git checkout -b fix/issue-number-description
   ```

3. **安装开发环境**：

   ```bash
   uv sync --group dev
   ```

4. **进行修改**：按照编码规范进行开发

5. **运行测试**：

   ```bash
   uv run pytest tests/ -v
   ```

6. **代码格式化**：

   ```bash
   uv run ruff check --fix .
   uv run ruff format .
   ```

7. **类型检查**：

   ```bash
   uv run mypy e2m2e/ --ignore-missing-imports
   ```

8. **提交更改**：遵循 [约定式提交](https://www.conventionalcommits.org/zh-hans/)格式：

   ```bash
   git add .
   git commit -m "feat: 添加 XXX 功能"
   # 或
   git commit -m "fix: 修复 XXX 问题"
   # 或
   git commit -m "docs: 更新 XXX 文档"
   ```

9. **推送到 GitHub**：

   ```bash
   git push origin feature/your-branch-name
   ```

10. **创建 Pull Request**：
    - 访问你的 GitHub 仓库页面
    - 点击 "Compare & pull request"
    - 填写清晰的 PR 描述
    - 链接相关 Issue（如有）

### 编码规范

- 遵循 [PEP 8](https://www.python.org/dev/peps/pep-0008/) 规范
- 使用有意义的变量和函数名
- 为所有公共函数和类添加文档字符串
- 保持函数简洁专注（单一职责原则）
- 导入顺序：标准库 → 第三方库 → 本地模块
- Ruff 作为代码检查和格式化工具，行长度限制为 100 字符

### 测试要求

- 为新功能编写测试（测试覆盖率要求 ≥ 80%）
- 确保所有现有测试通过
- 测试名称应清晰描述测试内容
- SPICE 依赖测试使用 `@pytest.mark.spice` 标记

### 代码审查流程

1. **自动检查**：GitHub Actions 自动运行 lint、类型检查和测试
2. **人工审查**：维护者审查代码
3. **反馈循环**：根据反馈进行修改
4. **合并**：审查通过后合并到主分支

审查重点：代码正确性、性能影响、可读性和可维护性、测试覆盖、文档完整性。

---

## 第二部分：开发者内部指南

### 库目录结构与职责

```text
e2m2e/
├── core/           ← 数据结构和基础物理模型（修改需格外谨慎）
├── algorithms/     ← 数值算法（最常扩展）
├── transfer/       ← 转移轨道设计方案
├── visualization/  ← 绘图工具
├── mbse/           ← MBSE 基础设施：Protocol 接口、Pydantic 模型、需求数据库
└── __init__.py     ← 公共 API 注册入口
```

**核心原则：`core/` 是基础，`algorithms/` 和 `transfer/` 是上层建筑。** 修改 core 影响整个库，扩展 algorithms/transfer 相对独立。

### 常见修改场景

#### 场景 A：为现有类添加字段/方法

直接在对应文件中添加。注意：如果新字段影响 `save_to_file()` / `load_from_file()`，需同步更新序列化逻辑。

#### 场景 B：添加新算法

1. 在 `e2m2e/algorithms/` 下创建新文件
2. 在 `algorithms/__init__.py` 中导出
3. 在顶层 `e2m2e/__init__.py` 中注册公共 API 并加入 `__all__`

#### 场景 C：添加新的校正策略

在 `algorithms/strategies/` 下创建新文件，参考现有的 `halo.py`、`symmetric_2d.py` 等。

#### 场景 D：添加新模块

1. 在相应子包中创建文件
2. 在子包 `__init__.py` 中导出
3. 在顶层 `e2m2e/__init__.py` 中注册并加入 `__all__`
4. 如需新依赖，更新 `pyproject.toml` 后运行 `uv sync`
5. 如为新组件，在 `e2m2e/mbse/architecture/` 中注册

### 关键注意事项

#### 1. 维护接口稳定性（最重要）

公共方法签名不能随意更改。如必须修改，通过添加带默认值的新参数保持向后兼容。

#### 2. uv 可编辑安装

使用 `uv sync` 安装后，修改 `e2m2e/` 源码立即生效，无需重新安装。唯一例外：修改 `pyproject.toml` 中的依赖后需重新运行 `uv sync`。

#### 3. 核心类依赖关系

```text
CR3BP_System  ←─ CR3BP_Dynamics  ←─ DifferentialCorrection
                       ↑                      ↑
                     Orbit           Continuation, StabilityAnalysis
                       ↑
               CoordinateTransformation
```

修改 `CR3BP_Dynamics` 时特别注意：

- `equations_of_motion(t, state)` 的签名被所有算法调用
- `propagate()` 返回的字典键（`'states'`、`'time'`、`'stm'`、`'jacobi_error'`）被多处依赖
- 添加新动力学模型时，创建新子类而非修改基类

#### 4. 数值精度敏感性

- 积分器始终使用 `rtol=atol=1e-12` 或更高精度
- 不要随意增大有限差分步长（差分校正中的 `eps`）
- 状态向量顺序 `[x, y, z, vx, vy, vz]` 的改变会导致全局崩溃

#### 5. 状态向量与矩阵形状约定

- 状态向量：`[x, y, z, vx, vy, vz]`
- States 形状：`(n_points, 6)`
- STM 形状：`(n_points, 6, 6)`

#### 6. MBSE 协议一致性

新类应满足 `e2m2e.mbse.architecture.ports` 中定义的相应 Protocol 接口。

### 快速参考：添加新内容检查表

| 操作 | 需修改的文件 |
| --- | --- |
| 为现有类添加方法 | 对应模块文件 |
| 添加新算法类 | 新文件 + 子包 `__init__.py` + 顶层 `__init__.py` |
| 添加新依赖 | `pyproject.toml` → 重新运行 `uv sync` |
| 修改公共接口 | 对应模块 + 测试 + 验证外部调用兼容性 |
| 添加新子包 | 新目录 + `__init__.py` + 顶层注册 |
| 添加新组件 | 在 `mbse/architecture/` 注册 + 在 `mbse/requirements/` 添加需求 |

---

## 发布流程

### 版本号规范

遵循 [语义化版本](https://semver.org/lang/zh-CN/)：

```text
4.0.0 → 4.0.1  Bug 修复 / 微调
4.0.0 → 4.1.0  新功能模块
4.0.0 → 5.0.0  破坏性接口变更
```

版本号定义在 `pyproject.toml` 中，`__init__.py` 通过 `importlib.metadata` 读取。

### 发布步骤

1. 更新 `pyproject.toml` 中的版本号
2. 更新 `CHANGELOG.md`
3. 提交并打标签：

   ```bash
   git add pyproject.toml CHANGELOG.md
   git commit -m "chore: bump version to 4.x.x"
   git tag v4.x.x
   git push origin master --tags
   ```

4. GitHub Actions 自动运行 CI、创建 Release、发布到 PyPI

### CI/CD 工作流

| 工作流 | 触发条件 | 功能 |
| --- | --- | --- |
| `ci.yml` | Push/PR 到 master | Lint + 类型检查 + 测试（3 OS × 4 Python 版本） |
| `release.yml` | 推送 `v*` 标签 | 构建 + GitHub Release + PyPI 发布 |
| `deploy-docs.yml` | Push 到 master | Sphinx 文档构建并部署到 GitHub Pages |

---

## 常见问题

### 如何开始贡献？

从简单的任务开始：修复文档中的错别字、改进现有测试、添加使用示例。

### 遇到问题怎么办？

查看现有文档和 Issues，在相关 Issue 中提问，或联系维护者。

### 贡献会被接受吗？

只要符合项目目标和质量标准，所有有价值的贡献都会被考虑。即使没有被接受，我们也会提供建设性反馈。
