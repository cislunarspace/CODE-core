# E2M2E 文档

> Earth-to-Moon-to-Earth (E2M2E) 轨道力学库的完整文档

## 文档结构

```
docs/
├── index.md              # 本文档（文档索引）
├── guides/               # 使用指南
│   ├── system-overview.md    # 系统架构与设计
│   ├── orbit-generation.md  # 轨道生成教程
│   ├── visualization-guide.md # 可视化教程
│   └── release.md            # PyPI 发布指南
├── core/                 # 核心模块
│   ├── system.md         # CR3BP_System - 系统参数
│   ├── dynamics.md       # CR3BP_Dynamics - 动力学
│   ├── orbit.md          # Orbit, OrbitFamily - 轨道
│   ├── coordinate.md     # CoordinateTransformation - 坐标变换
│   ├── ephemeris_system.md      # EphemerisSystem - 星历系统
│   ├── ephemeris_dynamics.md    # EphemerisDynamics - 星历动力学
│   └── spice.md                 # SPICE 内核管理
├── algorithms/           # 算法模块
│   ├── continuation.md       # 延拓法
│   ├── halo.md               # Halo 轨道与伪弧长轨道族
│   ├── differential_correction.md  # 微分修正
│   ├── stability.md          # 稳定性分析
│   └── multiple_shooting.md  # 多重打靶法
├── visualization/       # 可视化模块
│   └── plotting.md           # 绘图功能（向后兼容）
└── reference/           # 技术参考
    ├── api-reference.md     # 完整API文档
    └── algorithms.md        # 算法技术细节
```

## 快速导航

### 使用指南 (guides)

| 文档 | 说明 |
|------|------|
| [系统总览](guides/system-overview.md) | 架构设计、模块职责、数据流、典型工作流 |
| [轨道生成](guides/orbit-generation.md) | DRO、Halo、Lissajous 轨道生成教程 |
| [可视化指南](guides/visualization-guide.md) | 绘图功能详解、2D/3D 可视化 |
| [发布指南](guides/release.md) | PyPI 发布流程、版本管理 |

### 核心模块 (core)

| 类/模块 | 文件 | 说明 |
|---------|------|------|
| `System` | [core/system.md](core/system.md) | 天体系统基类 |
| `CR3BP_System` | [core/system.md](core/system.md) | 系统参数与平动点计算 |
| `EphemerisSystem` | [core/ephemeris_system.md](core/ephemeris_system.md) | 星历系统定义 |
| `Dynamics` | [core/dynamics.md](core/dynamics.md) | 动力学基类 |
| `CR3BP_Dynamics` | [core/dynamics.md](core/dynamics.md) | 运动方程与数值积分 |
| `EphemerisDynamics` | [core/ephemeris_dynamics.md](core/ephemeris_dynamics.md) | 星历动力学 |
| `Orbit` | [core/orbit.md](core/orbit.md) | 轨道数据管理 |
| `OrbitFamily` | [core/orbit.md](core/orbit.md) | 轨道族管理 |
| `CoordinateTransformation` | [core/coordinate.md](core/coordinate.md) | 坐标系变换 |
| `SPICEManager` | [core/spice.md](core/spice.md) | SPICE 内核管理 |

### 算法模块 (algorithms)

| 类/模块 | 文件 | 说明 |
|---------|------|------|
| `Continuation` | [algorithms/continuation.md](algorithms/continuation.md) | 自然/伪弧长延拓 |
| Halo / PAL 轨道族 | [algorithms/halo.md](algorithms/halo.md) | Halo 初值、伪弧长延拓、脚本与 MATLAB 对照 |
| `DifferentialCorrection` | [algorithms/differential_correction.md](algorithms/differential_correction.md) | 周期轨道修正 |
| `StabilityAnalysis` | [algorithms/stability.md](algorithms/stability.md) | Floquet稳定性分析 |
| `MultipleShooting` | [algorithms/multiple_shooting.md](algorithms/multiple_shooting.md) | 多重打靶法修正器 |

### 可视化模块 (visualization)

| 类/函数 | 文件 | 说明 |
|---------|------|------|
| `plotting` 模块 | [visualization/plotting.md](visualization/plotting.md) | 绘图功能（向后兼容） |

### 转移模块 (transfer)

| 类/函数 | 文件 | 说明 |
|---------|------|------|
| `DROTransferSearch` | [reference/api-reference.md](reference/api-reference.md) | DRO到RO转移网格搜索 |
| `DROTRONLPOptimizer` | [reference/api-reference.md](reference/api-reference.md) | 两脉冲转移NLP优化 |
| `load_orbit_from_json` | [reference/api-reference.md](reference/api-reference.md) | 从JSON加载轨道数据 |
| `save_search_results` | [reference/api-reference.md](reference/api-reference.md) | 保存搜索结果 |

### 技术参考 (reference)

| 文档 | 说明 |
|------|------|
| [API 参考](reference/api-reference.md) | 完整 API 文档、类与方法说明 |
| [算法参考](reference/algorithms.md) | CR3BP 理论、算法数学基础 |

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

- [API 参考](reference/api-reference.md) - 完整 API 文档
- [示例代码](../examples/) - 实际使用示例
- [测试用例](../tests/) - 单元测试
- [Halo 后续开发路线图](ways-of-work/plan/halo-roadmap_zh.md) - PAL / 轨道族演进计划

### 常用任务

1. **设计DRO轨道** → 参考 [轨道生成 - DRO](guides/orbit-generation.md#distant-retrograde-orbit-dro)
2. **设计Halo轨道** → 参考 [轨道生成 - Halo](guides/orbit-generation.md)
3. **生成轨道族** → 参考 [轨道族延拓](reference/algorithms.md)
4. **分析稳定性** → 参考 [稳定性分析](reference/algorithms.md)

## 物理背景

E2M2E 基于**圆型限制性三体问题 (CR3BP)** 实现轨道设计。地月系统中：
- 质量参数 $\mu \approx 0.01215$
- 特征距离：384,400 km（地月距离）
- 特征周期：27.32 天

详见 [CR3BP理论](reference/algorithms.md) 和 [系统概述](guides/system-overview.md)。
