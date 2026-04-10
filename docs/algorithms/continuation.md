# Continuation（轨道族延拓）

**文件**: `e2m2e/algorithms/continuation.py`

**类签名**:
```python
class Continuation:
    """轨道族延拓：自然参数、伪弧长（XZ 对称）、Halo 种子与族生成"""
```

## 设计原理

延拓从已知周期轨道（种子）出发，沿参数或伪弧长逐步生成轨道族。自然参数延拓在参数单调时简单有效；伪弧长延拓在自由变量 \(\mathbf{X}=[r_x,r_z,\dot y,T/2]\) 上引入切向与约束，用于跟踪含折返的族曲线。

## 核心方法（节选）

| 方法 | 说明 |
|------|------|
| `natural_continuation(...)` | 自然参数延拓 |
| `pseudo_arclength_continuation(seed_orbit, ...)` | XZ 对称伪弧长延拓（单方向 `positive` / `negative`） |
| `generate_halo_seed_orbit(...)` | Halo 种子轨道 |
| `generate_halo_family(...)` | 按 `amplitude_z` 步进的 Halo 族（独立 Richardson 初值） |
| `halo_pseudo_arclength_continuation(...)` | Halo 专用：双向支、步长与 MATLAB 脚本可对齐 |

**Halo 初值、PAL 细节、脚本入口与 MATLAB 对照**见 **[Halo 轨道算法文档](halo.md)**。

## 使用示例

```python
from e2m2e.algorithms.continuation import Continuation

continuation = Continuation(corrector, step=0.01)
family = continuation.natural_continuation(
    seed_orbit=seed_orbit,
    param_range=(0.8, 1.0),
    step_size=0.01,
    verbose=True,
)
```

伪弧长与 Halo 族示例见 [Halo](halo.md) 与 [轨道生成指南 - Halo](../guides/orbit-generation.md)。
