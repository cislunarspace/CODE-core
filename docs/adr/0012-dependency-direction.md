# ADR 0012：依赖方向规则与 CI import 检查

**状态**：已采纳（已实施）
**日期**：2026-07-31
**关联**：ADR 0011（五层架构）

## 背景

五层架构（ADR 0011）的可行性依赖依赖方向被真实执行。若无强制手段，层间 import 会随时间漂移——算法层悄悄 import 接口层、数据层依赖算法类型。参考书结论："架构靠评审和基线，不靠自觉"。

## 决策

依赖方向规则（硬规则）：

```
api/ → algorithm/ + data/
algorithm/ → data/ + _integrators
data/ → 仅外部库（SPICE/r2s2/numpy）
integrators.py → _integrators
tools/ → 任意（辅助，核心不 import tools/）
```

**两条硬边界**：
1. 算法层不 import api/（Pydantic 只在 api/ 边界，算法层用 numpy/dataclass）。
2. 数据层不 import algorithm/（类型 Orbit/EphemerisTable 在 data/types/，是数据层自产）。

CI 跑 import 检查强制（自定义脚本或 lint 规则，检查 `algorithm/` 不 import `api/`、`tools/` 等）。

## 理由

1. **Pydantic 只在 api/ 边界**：算法层保持 numpy + 异常，避免 156 个测试因 Pydantic 对象改签名。
2. **数据层自足**：data/ 只依赖外部库，可独立测试、可独立替换数据源。
3. **核心不依赖工具**：tools/ 是辅助层，核心库不 import 它，保持核心纯净。

## 结果

- CI 新增 import 检查步骤。
- 新代码遵循依赖方向，旧代码迁移时同步修正。
