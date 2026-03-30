# 优化 e2m2e/core 代码注释

## 目标
将 e2m2e/core 下所有源文件的 docstring 统一为 Google 风格（中文），删除冗余内联注释，处理 TODO 项。

## 任务列表
- [x] 1. 在项目根目录创建 TODO.md，将 system.py 中的 TODO 迁移过去，并从源码中移除原注释
- [x] 2. 优化 `__init__.py` 的模块 docstring（如需要）
- [x] 3. 优化 `system.py`：docstring 转 Google 风格 + 清理冗余内联注释
- [x] 4. 优化 `orbit.py`：docstring 转 Google 风格 + 清理冗余内联注释 + 删除类 docstring 中冗余的方法列表
- [x] 5. 优化 `dynamics.py`：docstring 转 Google 风格 + 清理冗余内联注释 + 去除子类重复基类属性的 docstring
- [x] 6. 优化 `coordinate.py`：docstring 转 Google 风格 + 清理冗余内联注释
- [x] 7. 运行测试验证无回归（193 passed，1 个预存 bug：`is_initialized` 未在 `__init__` 中初始化）

## 备注
- Google 风格 docstring 示例：
  ```
  Args:
      mu: 质量参数 μ = m2/(m1+m2)
      primary: 主天体名称
  
  Returns:
      平动点位置字典
  ```
- 内联注释保留原则：解释"为什么"的保留，仅重复代码"做什么"的删除
- TODO.md 用于集中管理项目待办事项
