# e2m2e 贡献指南（中文版）

感谢您对 e2m2e 项目的关注！本指南将帮助您了解如何为项目做出贡献。

## 📋 行为准则

在参与本项目时，请保持尊重和礼貌。我们欢迎各种形式的贡献，包括但不限于：
- 报告 bug
- 提出新功能建议
- 改进文档
- 提交代码改进

## 🐛 报告问题

### 报告 Bug

1. **检查是否已存在**：在 Issues 中搜索是否已有类似问题
2. **创建新 Issue**：如果不存在，创建新 Issue
3. **提供详细信息**：
   - 清晰的标题
   - 详细的问题描述
   - 重现步骤
   - 预期行为 vs 实际行为
   - 环境信息（Python 版本、操作系统等）
   - 相关代码片段或截图

### 建议新功能

1. **检查是否已建议**：在 Issues 中搜索是否已有类似建议
2. **创建新 Issue**：如果不存在，创建新 Issue
3. **详细说明**：
   - 功能的具体描述
   - 使用场景和优势
   - 可能的实现方案
   - 相关参考资料（如有）

## 💻 提交代码

### 准备工作

1. **Fork 仓库**：点击 GitHub 页面的 Fork 按钮
2. **克隆仓库**：
   ```bash
   git clone https://github.com/your-username/e2m2e.git
   cd e2m2e
   ```
3. **添加上游仓库**：
   ```bash
   git remote add upstream https://github.com/original-username/e2m2e.git
   ```

### 开发流程

1. **同步最新代码**：
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **创建功能分支**：
   ```bash
   git checkout -b feature/your-feature-name
   # 或修复 bug：git checkout -b fix/issue-number-description
   ```

3. **安装开发环境**：
   ```bash
   pip install -e ".[dev]"
   ```

4. **进行修改**：按照编码规范进行开发

5. **运行测试**：
   ```bash
   pytest tests/
   ```

6. **代码格式化**：
   ```bash
   ruff check --fix .
   ruff format .
   ```

7. **提交更改**：
   ```bash
   git add .
   git commit -m "描述性提交信息"
   ```

8. **推送到 GitHub**：
   ```bash
   git push origin feature/your-feature-name
   ```

9. **创建 Pull Request**：
   - 访问你的 GitHub 仓库页面
   - 点击 "Compare & pull request"
   - 填写清晰的 PR 描述
   - 链接相关 Issue（如有）

## 🎨 编码规范

### Python 代码风格

- 遵循 [PEP 8](https://www.python.org/dev/peps/pep-0008/) 规范
- 使用有意义的变量和函数名
- 为所有公共函数和类添加文档字符串（docstrings）
- 保持函数简洁专注（单一职责原则）

### 文档字符串格式

```python
def function_name(param1, param2):
    """函数功能的简要描述
    
    参数的详细说明和使用示例。
    
    参数：
    - param1: 参数1的描述
    - param2: 参数2的描述
    
    返回：
    - 返回值的描述
    
    示例：
    >>> function_name(value1, value2)
    expected_output
    """
```

### 导入顺序

1. 标准库导入
2. 第三方库导入
3. 本地应用/库导入

### 测试要求

- 为新功能编写测试
- 确保所有现有测试通过
- 测试覆盖率应保持在合理水平
- 测试名称应清晰描述测试内容

## 📝 文档要求

### 更新 README

如果添加了新功能或修改了现有功能，请相应更新：
- README.md 中的功能描述
- 使用示例
- API 文档

### 代码注释

- 复杂的算法需要详细注释
- 不明显的代码逻辑需要解释
- 使用 TODO、FIXME 等标记需要改进的地方

## 🔍 代码审查流程

1. **自动检查**：GitHub Actions 会自动运行测试和代码检查
2. **人工审查**：维护者会审查代码
3. **反馈循环**：根据反馈进行修改
4. **合并**：审查通过后合并到主分支

### 审查重点

- 代码正确性
- 性能影响
- 可读性和可维护性
- 测试覆盖
- 文档完整性

## 🚀 发布流程

### 版本号规范

遵循 [语义化版本](https://semver.org/lang/zh-CN/)：
- **主版本号**：不兼容的 API 修改
- **次版本号**：向下兼容的功能性新增
- **修订号**：向下兼容的问题修正

### 发布步骤

1. 更新 `pyproject.toml` 中的版本号
2. 更新 CHANGELOG.md（如有）
3. 创建发布标签
4. 构建并上传到 PyPI

## ❓ 常见问题

### 如何开始贡献？

从简单的任务开始，如：
- 修复文档中的错别字
- 改进现有测试
- 添加使用示例

### 遇到问题怎么办？

- 查看现有文档和 Issues
- 在相关 Issue 中提问
- 联系维护者

### 贡献会被接受吗？

只要符合项目目标和质量标准，所有有价值的贡献都会被考虑。即使没有被接受，我们也会提供建设性反馈。

## 🙌 致谢

感谢所有贡献者的时间和努力！您的贡献让这个项目变得更好。

---

**Happy Contributing!** 🎉