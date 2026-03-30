# 优化 e2m2e/algorithms 代码注释

## 目标
将 e2m2e/algorithms 下所有源文件的 docstring 统一为 Google 风格（中文），删除纯冗余内联注释，与 core/ 已完成的优化保持一致。

## 任务列表
- [x] 1. 检查并优化 `__init__.py` 的模块 docstring（已符合规范无需改动）
- [x] 2. 优化 `differential_correction.py`：docstring 转 Google 风格 + 清理纯冗余内联注释
- [x] 3. 优化 `continuation.py`：docstring 转 Google 风格 + 清理纯冗余内联注释
- [x] 4. 优化 `stability.py`：docstring 转 Google 风格 + 清理纯冗余内联注释
- [x] 5. 运行测试验证无回归（用户跳过）

## 备注
- Google 风格 docstring 规范：
  - 函数参数用 `Args:` 标签（非 `参数：`）
  - 返回值用 `Returns:` 标签（非 `返回：`）
  - 异常用 `Raises:` 标签（非 `抛出：`）
  - 类属性用 `Attributes:` 标签（非 `属性：`）
  - 附加说明用 `Note:` 标签
- 内联注释保留原则：解释"为什么"的保留，仅重复代码"做什么"的删除
- 保留算法步骤编号（如 `# 1. 带STM传播...`）和数学说明注释
- 不修改任何代码逻辑
