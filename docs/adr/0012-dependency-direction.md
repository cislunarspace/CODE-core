# ADR 0012: Dependency-direction rules with CI import checks / 依赖方向规则与 CI import 检查

[English](#adr-0012-dependency-direction-rules-with-ci-import-checks) | [简体中文](#中文)

## English

**Status**: Adopted (implemented)
**Date**: 2026-07-31
**Related**: ADR 0011 (five-layer architecture)

### Context

The five-layer architecture (ADR 0011) is only viable if dependency direction
is genuinely enforced. Without enforcement, inter-layer imports drift over
time: the algorithm layer quietly imports the interface layer, the data layer
starts depending on algorithm types. The reference literature's conclusion:
architecture rests on reviews and baselines, not good intentions.

### Decision

Dependency-direction rules (hard rules):

```
api/ → algorithm/ + data/
algorithm/ → data/ + _integrators
data/ → external libraries only (SPICE/r2s2/numpy)
integrators.py → _integrators
tools/ → anything (auxiliary; core never imports tools/)
```

**Two hard boundaries**:
1. The algorithm layer does not import `api/` (Pydantic lives only at the
   `api/` boundary; the algorithm layer uses numpy/dataclasses).
2. The data layer does not import `algorithm/` (types Orbit/EphemerisTable
   live in `data/types/` — produced by the data layer itself).

CI runs import checks for enforcement (custom script or lint rules checking,
e.g., that `algorithm/` does not import `api/` or `tools/`).

### Rationale

1. **Pydantic only at the `api/` boundary**: keeps the algorithm layer on
   numpy + exceptions, avoiding 156 tests breaking over Pydantic object
   signature changes.
2. **Self-sufficient data layer**: `data/` depends only on external libraries;
   independently testable and swappable data sources.
3. **Core never depends on tools**: `tools/` is auxiliary; the core library
   doesn't import it, keeping core pure.

### Consequences

- CI gains an import-check step.
- New code follows dependency direction; legacy code gets fixed while
  migrating.

## 中文

**状态**：已采纳（已实施）
**日期**：2026-07-31
**关联**：ADR 0011（五层架构）

### 背景

五层架构（ADR 0011）的可行性依赖依赖方向被真实执行。若无强制手段，层间 import 会随时间漂移：算法层悄悄 import 接口层、数据层依赖算法类型。参考书的结论是，架构靠评审和基线，不靠自觉。

### 决策

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

### 理由

1. **Pydantic 只在 api/ 边界**：算法层保持 numpy + 异常，避免 156 个测试因 Pydantic 对象改签名。
2. **数据层自足**：data/ 只依赖外部库，可独立测试、可独立替换数据源。
3. **核心不依赖工具**：tools/ 是辅助层，核心库不 import 它，保持核心纯净。

### 结果

- CI 新增 import 检查步骤。
- 新代码遵循依赖方向，旧代码迁移时同步修正。
