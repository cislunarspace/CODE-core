# CI 与 Release 工作流优化

## 目标
全面优化 CI 速度、Release 流程自动化程度、以及代码质量/安全检查

## 背景
现有 ci.yml 和 publish.yml 功能基本完备但存在优化空间：
- CI 无缓存，12 个 test matrix 全量安装依赖，耗时长
- release 和 publish 分散在两个 workflow，流程可合并
- 缺少 type check、依赖安全扫描
- coverage 已配置 fail_under=80 但 CI 中未强制

## 约束与风险
- SPICE kernel 相关测试需要跳过（已有 pytest.mark.spice 标记）
- PyPI 发布需要 Trusted Publisher（id-token: write）已配置，勿改动
- 保持向后兼容：现有 trigger 不变（push master / PR master / tag v*）

## 任务列表

- [x] 1. **CI 速度优化：添加缓存和依赖安装加速** `.github/workflows/ci.yml`
  - lint/typecheck/security job: 使用 `actions/setup-python` 内置 pip cache (`cache: pip`)
  - test job: 同上，每个 matrix job 均启用 pip cache
  - 结果：所有 job 均配置 `cache: pip`

- [x] 2. **CI 速度优化：精简 test matrix** `.github/workflows/ci.yml`
  - PR 触发只跑 ubuntu-latest + Python 3.12（1 个 test job）
  - push master 跑完整 3 OS × 4 Python（12 个 test job）
  - 使用 `fromJson()` 动态生成 matrix

- [x] 3. **添加质量检查 job** `.github/workflows/ci.yml`
  - 新增 `typecheck` job：mypy --ignore-missing-imports
  - 新增 `security` job：pip-audit
  - lint、typecheck、security 并行运行，test 依赖全部通过

- [x] 4. **Coverage 门控强化** `.github/workflows/ci.yml`
  - test job 添加 `--cov-fail-under=80` 参数
  - codecov upload 保留仅在 ubuntu+3.12 执行

- [x] 5. **统一 Release + PyPI 发布流程** `.github/workflows/release.yml`
  - 合并旧 ci.yml release job 和 publish.yml 为单一 release.yml
  - 流程：lint → test (ubuntu+3.12) → build → github-release → publish-pypi
  - 使用 softprops/action-gh-release，自动生成 release notes + 上传 dist
  - build 前验证 git tag 版本与 pyproject.toml 一致
  - 已删除旧 publish.yml

- [x] 6. **PyPI 发布验证与安全** `.github/workflows/release.yml`
  - build job 中添加 `twine check dist/*` 验证产物完整性
  - publish-pypi job 配置 `environment: pypi` + `permissions: id-token: write`
  - 使用 `pypa/gh-action-pypi-publish@release/v1`（Trusted Publisher 模式）

## 备注
- mypy 初期使用 `--ignore-missing-imports` 宽松模式，后续可逐步收紧
- ci.yml 已移除 `tags: ["v*"]` trigger，release 事件由 release.yml 独立处理
- PyPI 已有 e2m2e 包发布，Trusted Publisher 配置勿动
