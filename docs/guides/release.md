# 发布指南

本文档介绍如何将 e2m2e 发布到 PyPI。

## 准备工作

### 1. 安装构建工具

```bash
pip install build twine
```

### 2. 获取 PyPI API Token

- **TestPyPI**：前往 [https://test.pypi.org](https://test.pypi.org) 注册账号，生成 API Token
- **正式 PyPI**：前往 [https://pypi.org](https://pypi.org) 注册账号，生成 API Token

配置 token：

```bash
# ~/.pypirc 文件内容
[pypi]
username = __token__
password = pypi-xxxxxxxxxxxx

[testpypi]
username = __token__
password = pypi-xxxxxxxxxxxx
```

## 发布流程

### 1. 确认版本号

在 `pyproject.toml` 中更新版本号（必须高于上一个版本）：

```toml
[project]
version = "3.1.12"
```

### 2. 提交 Git 快照

```bash
cd /path/to/e2m2e
git add .
git commit -m "描述修改内容"
git tag v3.1.12
git push origin master --tags
```

### 3. 构建分发包

```bash
rm -rf dist/ build/ *.egg-info/
python -m build
```

### 4. 上传到 TestPyPI（推荐先行测试）

```bash
twine upload --repository testpypi dist/*
```

验证安装：

```bash
pip install --index-url https://test.pypi.org/simple/ e2m2e==3.1.12
```

### 5. 上传到正式 PyPI

```bash
twine upload --repository pypi dist/*
```

## 常用命令汇总

```bash
# 完整发布流程
rm -rf dist/ build/ *.egg-info/
python -m build
twine upload --repository testpypi dist/*    # 测试环境
twine upload --repository pypi dist/*         # 正式环境

# 仅安装正式版本
pip install e2m2e

# 安装指定版本
pip install e2m2e==3.1.12

# 从 TestPyPI 安装
pip install --index-url https://test.pypi.org/simple/ e2m2e==3.1.11
```

## 注意事项

1. **版本号必须递增**：PyPI 不允许上传相同或更低版本号的包
2. **先测 TestPyPI**：每次发布前先在 TestPyPI 验证包是否正常
3. **保持一致性**：Git tag 与 `pyproject.toml` 中的版本号应保持一致
4. **license 字段**：确保 `pyproject.toml` 中 `license = {text = "Apache-2.0"}` 与 LICENSE 文件一致
