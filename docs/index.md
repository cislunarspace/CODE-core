# E2M2E 文档

> Earth-to-Moon-to-Earth (E2M2E) 轨道力学库的完整文档

## 文档结构

文档组织与代码模块结构一一对应：

```
docs/
├── index.md              # 本文档
├── core/                 # 核心模块
│   ├── system.md         # CR3BP_System - 系统参数
│   ├── dynamics.md       # CR3BP_Dynamics - 动力学
│   ├── orbit.md          # Orbit, OrbitFamily - 轨道
│   └── coordinate.md     # CoordinateTransformation - 坐标变换
├── algorithms/          # 算法模块
│   ├── continuation.md       # 延拓法
│   ├── differential_correction.md  # 微分修正
│   └── stability.md          # 稳定性分析
├── transfer/            # 转移模块
│   ├── inter_orbit.md       # DROROTransferSearch - 轨道间转移
│   ├── earth_moon.md        # EarthMoonTransfer - 地月转移
│   └── moon_earth.md        # MoonEarthTransfer - 月地转移
└── visualization/       # 可视化模块
    └── plotting.md          # 绘图函数
```

## 模块索引

### 核心模块 (core)

| 类/模块 | 文件 | 说明 |
|---------|------|------|
| `CR3BP_System` | [core/system.md](core/system.md) | 系统参数与平动点计算 |
| `CR3BP_Dynamics` | [core/dynamics.md](core/dynamics.md) | 运动方程与数值积分 |
| `Orbit` | [core/orbit.md](core/orbit.md) | 轨道数据管理 |
| `OrbitFamily` | [core/orbit.md](core/orbit.md) | 轨道族管理 |
| `CoordinateTransformation` | [core/coordinate.md](core/coordinate.md) | 坐标系变换 |

### 算法模块 (algorithms)

| 类/模块 | 文件 | 说明 |
|---------|------|------|
| `ContinuationMethod` | [algorithms/continuation.md](algorithms/continuation.md) | 弧长延拓法 |
| `DifferentialCorrection` | [algorithms/differential_correction.md](algorithms/differential_correction.md) | 周期轨道修正 |
| `StabilityAnalysis` | [algorithms/stability.md](algorithms/stability.md) | Floquet稳定性分析 |

### 转移模块 (transfer)

| 类/模块 | 文件 | 说明 |
|---------|------|------|
| `InterOrbitTransfer` | [transfer/inter_orbit.md](transfer/inter_orbit.md) | 轨道间转移设计 |
| `DROROTransferSearch` | [transfer/inter_orbit.md](transfer/inter_orbit.md) | DRO→RO转移搜索 (见下方注意事项) |
| `EarthMoonTransfer` | [transfer/earth_moon.md](transfer/earth_moon.md) | 地月转移设计 |
| `MoonEarthTransfer` | [transfer/moon_earth.md](transfer/moon_earth.md) | 月地返回设计 |

> **注意**: `DROROTransferSearch` 类实际位于 `e2m2e/transfer/dro_ro_search.py`（原始版本）和 `e2m2e/transfer/dro_ro_search_v2.py`（修复bug版本）。`inter_orbit.md` 文档描述的是该类的使用方法。

### 可视化模块 (visualization)

| 函数 | 文件 | 说明 |
|------|------|------|
| `plot_orbit_2d/3d` | [visualization/plotting.md](visualization/plotting.md) | 轨道绘图 |
| `plot_transfer_2d/3d` | [visualization/plotting.md](visualization/plotting.md) | 转移轨迹绘图 |
| `plot_system_geometry` | [visualization/plotting.md](visualization/plotting.md) | 系统几何绘图 |

## 快速开始

```python
from e2m2e.core.system import CR3BP_System
from e2m2e.core.dynamics import CR3BP_Dynamics
from e2m2e.core.orbit import Orbit

# 创建系统
system = CR3BP_System.from_known_system("earth_moon")

# 创建动力学模型
dynamics = CR3BP_Dynamics(system)

# 传播轨道
result = dynamics.propagate(initial_state=state, t_span=(0, 10.0))
```

## 资源

- [API 参考](../e2m2e/) - 代码中的 docstring
- [示例代码](../examples/) - 实际使用示例
- [测试用例](../tests/) - 单元测试

### 常用任务

1. **设计DRO轨道** → 参考 [轨道生成 - DRO](guides/orbit-generation.md#distant-retrograde-orbit-dro)
2. **设计Halo轨道** → 参考 [轨道生成 - Halo](guides/orbit-generation.md#halo轨道)
3. **生成轨道族** → 参考 [轨道族延拓](reference/algorithms.md#5-轨道族延拓算法)
4. **分析稳定性** → 参考 [稳定性分析](reference/algorithms.md#7-稳定性分析)

## 物理背景

E2M2E 基于**圆型限制性三体问题 (CR3BP)** 实现轨道设计。地月系统中：
- 质量参数 $\mu \approx 0.01215$
- 特征距离：384,400 km（地月距离）
- 特征周期：27.32 天

详见 [CR3BP理论](reference/algorithms.md#1-概述) 和 [系统概述](guides/system-overview.md)。
