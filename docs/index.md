# E2M2E — 地月转移轨道设计库

基于圆型限制性三体问题 (CR3BP) 的轨道力学工具，用于设计地月空间的周期轨道和转移轨道。

## 30 秒上手

```bash
pip install e2m2e
```

```python
from e2m2e.core import CR3BP_System, CR3BP_Dynamics
from e2m2e.algorithms import DifferentialCorrection
import numpy as np

# 创建地月系统，设计一条 DRO 轨道
system = CR3BP_System.from_known_system("earth_moon")
system.set_characteristic_scales(distance=384400, period=27.32*86400)
system.compute_libration_points()

dynamics = CR3BP_Dynamics(system)
dc = DifferentialCorrection(dynamics)
dc.setup_2D_symmetric_x_fixed_x0(x0=0.8)

initial_state = np.array([0.8, 0.0, 0.0, 0.0, 0.5, 0.0])
orbit, result = dc.iterate_correction(initial_state, t_half=1.6)

print(f"轨道周期: {orbit.period:.4f}")
print(f"Jacobi 常数: {orbit.jacobi_constant:.6f}")
```

## 你能用它做什么

### 设计周期轨道

DRO、Halo、Lyapunov — 从初始猜测到收敛轨道，再到整族轨道的延拓。

- [轨道生成教程](guides/orbit-generation.md) — DRO / Halo / Lyapunov 生成流程
- [微分修正](algorithms/differential_correction.md) — 怎么修正出精确的周期轨道
- [轨道族延拓](algorithms/continuation.md) — 怎么生成一族周期轨道
- [Halo 轨道](algorithms/halo.md) — Richardson 初值、伪弧长延拓、命令行脚本

### 分析轨道稳定性

Floquet 乘子、分岔检测、稳定性指数 — 理解轨道的动力学特性。

- [稳定性分析](algorithms/stability.md) — 怎么判断轨道是否稳定、怎么找分岔点

### 可视化轨道族和转移轨迹

2D/3D 投影、Jacobi 着色、稳定性图 — 生成高质量的轨道力学图。

- [可视化指南](guides/visualization-guide.md) — 从单条轨道到轨道族的可视化全流程

## 库的结构

```
core/           基础层 — 系统定义、动力学方程、轨道数据
  ↓
algorithms/     算法层 — 微分修正、延拓、稳定性分析、多重打靶
  ↓
transfer/       设计层 — DRO→RO 转移搜索、NLP 优化
  ↓
visualization/  展示层 — 轨道族绘图、转移轨迹可视化
```

| 层 | 做什么 | 入口类 |
|----|--------|--------|
| `core` | 定义天体系统、积分运动方程、管理轨道数据 | `CR3BP_System`, `CR3BP_Dynamics`, `Orbit` |
| `algorithms` | 修正周期轨道、延拓轨道族、分析稳定性 | `DifferentialCorrection`, `Continuation`, `StabilityAnalysis` |
| `transfer` | 搜索和优化轨道转移 | `Transfer`, `DROTransferSearch` |
| `visualization` | 绘制轨道、族、转移轨迹 | `OrbitVisualizer`, `FamilyPlotter`, `TransferPlotter` |

## 典型工作流

```python
from e2m2e.core import CR3BP_System, CR3BP_Dynamics, Orbit
from e2m2e.algorithms import DifferentialCorrection, Continuation
from e2m2e.visualization import OrbitVisualizer

# 1. 定义系统
system = CR3BP_System.from_known_system("earth_moon")
system.set_characteristic_scales(distance=384400, period=27.32*86400)
system.compute_libration_points()

# 2. 创建动力学模型
dynamics = CR3BP_Dynamics(system)

# 3. 设计一条周期轨道
dc = DifferentialCorrection(dynamics)
dc.setup_2D_symmetric_x_fixed_x0(x0=0.8)
orbit, result = dc.iterate_correction(
    np.array([0.8, 0.0, 0.0, 0.0, 0.5, 0.0]), t_half=1.6
)

# 4. 延拓得到轨道族
cont = Continuation(corrector=dc, step=0.01)
family = cont.natural_continuation(
    seed_orbit=orbit, param_range=(0.8, 0.95), step_size=0.01
)

# 5. 可视化
viz = OrbitVisualizer(system)
for orb in family:
    viz.plot_2d_projection(orb, plane="xy")
viz.show()
```

## 进一步了解

| 你可能想知道 | 去哪里看 |
|-------------|---------|
| CR3BP 的物理背景和数学公式 | [算法细节](reference/algorithms.md) |
| 所有类和方法的完整列表 | [API 参考](reference/api-reference.md) |
| 坐标变换（旋转系 ↔ 惯性系） | [坐标变换](core/coordinate.md) |
| 星历动力学和 SPICE 管理 | [星历系统](core/ephemeris_system.md) |
| 多重打靶法 | [多重打靶](algorithms/multiple_shooting.md) |
| 项目架构设计细节 | [系统总览](guides/system-overview.md) |
