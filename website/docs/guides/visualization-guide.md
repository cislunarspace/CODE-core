---
title: e2m2e 可视化模块使用指南
---

# e2m2e 可视化模块使用指南

## 概述

`e2m2e.visualization` 模块提供了圆形限制性三体问题（CR3BP）中轨道的各种可视化功能。模块采用分层架构，将可视化功能拆分为独立的组件，便于维护和扩展。

### 模块结构

```
e2m2e/visualization/
├── __init__.py        # 公共 API 导出
├── config.py          # PlotConfig 数据类 — 所有样式/图形设置
├── base.py            # OrbitVisualizer 基类 — 原子化绑图操作
├── family.py          # FamilyPlotter — 轨道族高级可视化
├── transfer.py        # TransferPlotter — 转移轨道可视化
├── stability.py       # compute_stability_for_family — 并行稳定性计算
└── plotting.py        # 向后兼容的重导出 & configure_academic_fonts()
```

### 类层次结构

```
PlotConfig (dataclass)
    ↓ 传入
OrbitVisualizer (base.py)
    ├── FamilyPlotter (family.py)
    └── TransferPlotter (transfer.py)

compute_stability_for_family (stability.py)  — 独立函数
```

## 目录

1. [快速开始](#快速开始)
2. [核心类与配置](#核心类与配置)
3. [基本可视化功能（OrbitVisualizer）](#基本可视化功能orbitvisualizer)
4. [轨道族可视化（FamilyPlotter）](#轨道族可视化familyplotter)
5. [转移轨道可视化（TransferPlotter）](#转移轨道可视化transferplotter)
6. [稳定性计算](#稳定性计算)
7. [自定义设置（PlotConfig）](#自定义设置plotconfig)
8. [常见问题](#常见问题)
9. [示例代码](#示例代码)

## 快速开始

### 安装依赖

```bash
pip install numpy matplotlib
```

### 基本使用示例

```python
import numpy as np
from e2m2e.core.system import CR3BP_System
from e2m2e.visualization import OrbitVisualizer

# 1. 创建地月系统
system = CR3BP_System.from_known_system("earth_moon")
system.set_characteristic_scales(distance=384400, period=27.32 * 86400)
system.compute_libration_points()

# 2. 创建可视化器
viz = OrbitVisualizer(system)

# 3. 创建示例轨道数据（这里使用简单的圆形轨道作为示例）
n_points = 100
t = np.linspace(0, 2*np.pi, n_points)
x = 0.8 + 0.1 * np.cos(t)
y = 0.1 * np.sin(t)
z = np.zeros_like(t)
vx = -0.1 * np.sin(t)
vy = 0.1 * np.cos(t)
vz = np.zeros_like(t)

orbit_states = np.column_stack([x, y, z, vx, vy, vz])

# 4. 绘制2D投影
viz.plot_2d_projection(orbit_states, plane='xy', color='blue', label='Test Orbit')
viz.plot_primary_bodies()
viz.plot_libration_points()
viz.show()
```

## 核心类与配置

### PlotConfig — 样式配置数据类

`PlotConfig` 是一个 `dataclass`，集中管理所有图形样式参数。可以直接构造并传入任何可视化器。

```python
from e2m2e.visualization import PlotConfig

# 使用默认配置
config = PlotConfig()

# 自定义配置
config = PlotConfig(
    colormap="plasma",
    orbit_linewidth=2.0,
    orbit_alpha=0.9,
    figsize_2d=(14, 10),
    figsize_3d=(16, 10),
    dpi=150,
    primary_body_color="blue",
    primary_body_size=200,
    secondary_body_color="silver",
    secondary_body_size=100,
)

# 应用全局字体设置
config.apply_rcparams()
```

**PlotConfig 字段一览：**

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `colormap` | `str` | `"coolwarm"` | 轨道族色谱 |
| `orbit_linewidth` | `float` | `1.5` | 轨道线宽 |
| `orbit_alpha` | `float` | `0.8` | 轨道透明度 |
| `figsize_2d` | `tuple` | `(12, 10)` | 2D 图形大小 |
| `figsize_3d` | `tuple` | `(14, 10)` | 3D 图形大小 |
| `figsize_dual` | `tuple` | `(12, 7)` | 双轴图图形大小 |
| `figsize_overview` | `tuple` | `(18, 14)` | 概览图图形大小 |
| `dpi` | `int` | `100` | 分辨率 |
| `primary_body_color` | `str` | `"blue"` | 主天体颜色 |
| `primary_body_size` | `int` | `200` | 主天体标记大小 |
| `secondary_body_color` | `str` | `"silver"` | 次天体颜色 |
| `secondary_body_size` | `int` | `100` | 次天体标记大小 |
| `lp_colors` | `List[str]` | `["gray"]*5` | 平动点颜色 |
| `lp_markers` | `List[str]` | `["^"]*5` | 平动点标记 |
| `lp_sizes` | `List[int]` | `[60]*5` | 平动点标记大小 |
| `title` / `label` / `tick` / `legend` | `float` | 各级字号 | 字体大小控制 |

### OrbitVisualizer — 基类

```python
from e2m2e.visualization import OrbitVisualizer, PlotConfig

config = PlotConfig(dpi=150)
viz = OrbitVisualizer(system, config=config)
```

**参数：**
- `system`: CR3BP_System 对象，必需。用于获取系统参数和平动点位置。
- `config`: PlotConfig 对象，可选。不传则使用默认配置。

### FamilyPlotter — 轨道族可视化

```python
from e2m2e.visualization import FamilyPlotter, PlotConfig

config = PlotConfig(colormap="viridis")
plotter = FamilyPlotter(system, config=config)
```

继承自 `OrbitVisualizer`，额外提供 `plot_family_2d`、`plot_family_3d`、`plot_jacobi_period_stability`、`plot_family_overview` 等高级方法。

### TransferPlotter — 转移轨道可视化

```python
from e2m2e.visualization import TransferPlotter

plotter = TransferPlotter(system)
```

继承自 `OrbitVisualizer`，提供 `plot_solution_plane`、`plot_transfer_orbit` 等转移轨道专用方法。

### 方法总览

| 方法 | 所属类 | 描述 | 常用参数 |
|------|--------|------|----------|
| `plot_3d_orbit()` | OrbitVisualizer | 绘制3D轨道 | `orbit, color, label, ax, show_start` |
| `plot_2d_projection()` | OrbitVisualizer | 绘制2D投影 | `orbit, plane, color, label, ax, show_start` |
| `plot_libration_points()` | OrbitVisualizer | 绘制平动点 | `ax, show_labels, is_3d` |
| `plot_primary_bodies()` | OrbitVisualizer | 绘制天体 | `ax, is_3d` |
| `plot_family_2d()` | FamilyPlotter | 绘制轨道族2D视图 | `family_result, jacobi_values, title, plane, ...` |
| `plot_family_3d()` | FamilyPlotter | 绘制轨道族3D视图 | `family_result, jacobi_values, title, center, ...` |
| `plot_jacobi_period_stability()` | FamilyPlotter | 绘制Jacobi-周期-稳定性图 | `jacobi_values, periods, stability_values, ...` |
| `plot_family_overview()` | FamilyPlotter | 绘制轨道族四子图概览 | `family_result, jacobi_values, periods, stability_values, ...` |
| `plot_solution_plane()` | TransferPlotter | 绘制解空间散点图 | `results, color_by, ax, ...` |
| `plot_transfer_orbit()` | TransferPlotter | 绘制转移轨道3D图 | `departure_orbit, arrival_orbit, transfer_trajectory, ...` |
| `show()` | OrbitVisualizer | 显示图形 | 无 |
| `save()` | OrbitVisualizer | 保存图形 | `filename, dpi` |

## 基本可视化功能（OrbitVisualizer）

### 1. 3D轨道可视化

```python
from e2m2e.visualization import OrbitVisualizer

viz = OrbitVisualizer(system)
ax = viz.plot_3d_orbit(orbit, color='blue', label='3D Orbit')

viz.plot_primary_bodies(ax=ax, is_3d=True)
viz.plot_libration_points(ax=ax, is_3d=True)

ax.legend()
viz.show()
```

### 2. 2D投影

```python
# XY平面投影
viz.plot_2d_projection(orbit, plane='xy', color='red', label='XY Projection')
viz.plot_primary_bodies()
viz.plot_libration_points()
viz.show()

# XZ平面投影
viz.plot_2d_projection(orbit, plane='xz', color='green', label='XZ Projection')
viz.show()

# YZ平面投影
viz.plot_2d_projection(orbit, plane='yz', color='purple', label='YZ Projection')
viz.show()
```

### 3. 天体和平动点

```python
viz.plot_primary_bodies()
viz.plot_libration_points()
viz.show()
```

## 轨道族可视化（FamilyPlotter）

`FamilyPlotter` 继承 `OrbitVisualizer`，提供一键式轨道族可视化，自动按 Jacobi 常数着色。

### 1. 轨道族2D视图

```python
from e2m2e.visualization import FamilyPlotter, PlotConfig

plotter = FamilyPlotter(system, config=PlotConfig(colormap="viridis"))

# 计算 Jacobi 常数列表
jacobi_values = [orbit.jacobi_constant for orbit in family_result]

fig, ax = plotter.plot_family_2d(
    family_result,
    jacobi_values,
    title="L1 Lyapunov Family",
    plane="xy",
    show_bodies=True,
    show_libration=True,
    show_colorbar=True,
    save_path="family_2d.png",
)
```

### 2. 轨道族3D视图

```python
fig, ax = plotter.plot_family_3d(
    family_result,
    jacobi_values,
    title="L1 Lyapunov Family (3D)",
    center=(0.5, 0.0, 0.0),
    radius=0.65,
    elev=20,
    azim=-60,
    save_path="family_3d.png",
)
```

### 3. Jacobi-周期-稳定性图

```python
from e2m2e.visualization import FamilyPlotter, compute_stability_for_family

plotter = FamilyPlotter(system)
periods = [orbit.period for orbit in family_result]
stability_values = compute_stability_for_family(family_result, system)

fig, ax = plotter.plot_jacobi_period_stability(
    jacobi_values,
    periods,
    stability_values,
    title="Period & Stability vs Jacobi Constant",
    target_period=6.0,
    save_path="jacobi_period_stability.png",
)
```

### 4. 轨道族综合概览图

`plot_family_overview` 一键生成包含四个子图的概览：全局2D视图、缩放2D视图、Jacobi-周期-稳定性图、3D视图。

```python
fig = plotter.plot_family_overview(
    family_result,
    jacobi_values,
    periods,
    stability_values,
    suptitle="L1 Lyapunov Family Overview",
    plane="xy",
    zoom_xlim=(0.4, 0.6),
    zoom_ylim=(-0.15, 0.15),
    center_3d=(0.5, 0.0, 0.0),
    radius_3d=0.3,
    target_period=6.0,
    save_path="family_overview.png",
)
```

## 转移轨道可视化（TransferPlotter）

### 1. 解空间散点图

```python
from e2m2e.visualization import TransferPlotter

transfer_plotter = TransferPlotter(system)

# 按转移类型着色
ax = transfer_plotter.plot_solution_plane(
    results,
    color_by="transfer_type",
)
```

### 2. 转移轨道3D图

```python
ax = transfer_plotter.plot_transfer_orbit(
    departure_orbit=dro,
    arrival_orbit=ro,
    transfer_trajectory=transfer_states,
    departure_state=dep_state,
    insertion_state=ins_state,
    label="Transfer",
    color="red",
)
```

## 稳定性计算

`compute_stability_for_family` 是独立函数，使用多进程并行计算轨道族中每条轨道的稳定性指数（特征值最大模）。

```python
from e2m2e.visualization import compute_stability_for_family

# 自动使用所有 CPU 核心
stability_values = compute_stability_for_family(family_result, system)

# 限制并行数
stability_values = compute_stability_for_family(family_result, system, max_workers=4)
```

**参数：**
- `family_result`: 轨道族结果列表
- `system`: CR3BP_System 对象
- `max_workers`: 最大并行进程数，默认为 `min(cpu_count, len(family))`

**返回：** `List[float]` — 每条轨道的稳定性指数

## 自定义设置（PlotConfig）

### 使用 PlotConfig 数据类

```python
from e2m2e.visualization import PlotConfig, FamilyPlotter

config = PlotConfig(
    # 图形尺寸
    figsize_2d=(14, 10),
    figsize_3d=(16, 10),
    dpi=150,

    # 轨道样式
    orbit_linewidth=2.0,
    orbit_alpha=0.9,

    # 色谱
    colormap="plasma",

    # 天体样式
    primary_body_color="blue",
    primary_body_size=300,
    secondary_body_color="gray",
    secondary_body_size=150,

    # 平动点样式
    lp_colors=["darkred", "darkblue", "darkgreen", "darkviolet", "darkorange"],
    lp_markers=["^", "s", "D", "o", "v"],
    lp_sizes=[120, 120, 120, 180, 180],
)

plotter = FamilyPlotter(system, config=config)
```

### 应用全局字体设置

```python
config = PlotConfig()
config.apply_rcparams()  # 设置 Times New Roman + STIX 数学字体
```

### 使用现有坐标轴

```python
import matplotlib.pyplot as plt
from e2m2e.visualization import OrbitVisualizer

viz = OrbitVisualizer(system)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

viz.plot_2d_projection(orbit1, plane='xy', color='blue', ax=ax1, label='Orbit 1')
viz.plot_primary_bodies(ax=ax1)
ax1.set_title('Orbit 1 - XY Projection')

viz.plot_2d_projection(orbit2, plane='xy', color='red', ax=ax2, label='Orbit 2')
viz.plot_primary_bodies(ax=ax2)
ax2.set_title('Orbit 2 - XY Projection')

plt.tight_layout()
plt.show()
```

## 常见问题

### 1. 图形不显示

**问题：** 调用 `viz.show()` 后图形不显示。

**解决方案：**
```python
# 在脚本中使用
viz.show()

# 在Jupyter notebook中使用
%matplotlib inline
viz.show()

# 或者使用交互式模式
%matplotlib notebook
viz.show()
```

### 2. 平动点不显示

**问题：** 平动点没有出现在图形中。

**解决方案：** 确保系统已经计算了平动点。
```python
system.compute_libration_points()
viz = OrbitVisualizer(system)
```

### 3. 坐标轴比例不正确

**问题：** 2D投影的坐标轴比例不是1:1。

**解决方案：** `plot_2d_projection` 会自动设置等比例坐标轴。如果手动修改了坐标轴，可以重新设置：
```python
ax.set_aspect('equal')
```

### 4. 保存图形质量差

**问题：** 保存的图形分辨率低。

**解决方案：** 增加dpi参数。
```python
viz.save('orbit.png', dpi=300)  # 高分辨率
viz.save('orbit.pdf')           # 矢量图，无限分辨率
```

### 5. 多个轨道叠加

**问题：** 如何在同一个图形上绘制多个轨道。

**解决方案：** 多次调用绘图函数。
```python
viz.plot_2d_projection(orbit1, plane='xy', color='blue', label='Orbit 1')
viz.plot_2d_projection(orbit2, plane='xy', color='red', label='Orbit 2')

viz.plot_primary_bodies()
viz.plot_libration_points()

viz.axes.legend()
viz.show()
```

## 示例代码

### 完整示例：地月系统Lyapunov轨道可视化

```python
import numpy as np
from e2m2e.core.system import CR3BP_System
from e2m2e.core.dynamics import CR3BP_Dynamics
from e2m2e.visualization import PlotConfig, OrbitVisualizer

def visualize_lyapunov_orbit():
    """可视化地月系统的Lyapunov轨道"""

    # 1. 创建地月系统
    print("创建地月系统...")
    system = CR3BP_System.from_known_system("earth_moon")
    system.set_characteristic_scales(384400, 27.32 * 86400)
    system.compute_libration_points()

    # 2. 生成示例轨道数据
    print("生成示例轨道数据...")
    dynamics = CR3BP_Dynamics(system)

    n_points = 200
    t = np.linspace(0, 2*np.pi, n_points)
    amplitude = 0.02

    x0 = system.L1[0]
    x = x0 + amplitude * np.cos(t)
    y = amplitude * np.sin(t)
    z = np.zeros_like(t)
    vx = -amplitude * np.sin(t)
    vy = amplitude * np.cos(t)
    vz = np.zeros_like(t)

    orbit_states = np.column_stack([x, y, z, vx, vy, vz])

    # 3. 使用自定义配置创建可视化器
    print("创建可视化器...")
    config = PlotConfig(dpi=150, orbit_linewidth=2.0)
    viz = OrbitVisualizer(system, config=config)

    # 4. 绘制XY投影
    print("生成XY投影图...")
    viz.plot_2d_projection(orbit_states, plane='xy', color='blue', label='Lyapunov Orbit')
    viz.plot_primary_bodies()
    viz.plot_libration_points()
    viz.axes.legend()
    viz.save('lyapunov_orbit_xy.png', dpi=300)
    viz.show()

    # 5. 绘制3D视图
    print("生成3D视图...")
    viz.plot_3d_orbit(orbit_states, color='red', label='3D View')
    viz.plot_primary_bodies(ax=viz.axes_3d, is_3d=True)
    viz.plot_libration_points(ax=viz.axes_3d, is_3d=True)
    viz.axes_3d.legend()
    viz.save('lyapunov_orbit_3d.png', dpi=300)
    viz.show()

    print("可视化完成！")

if __name__ == "__main__":
    visualize_lyapunov_orbit()
```

### 完整示例：轨道族全流程可视化

```python
from e2m2e.core.system import CR3BP_System
from e2m2e.visualization import (
    PlotConfig, FamilyPlotter, compute_stability_for_family,
)

def visualize_orbit_family():
    """轨道族全流程可视化：2D、3D、稳定性、概览"""

    system = CR3BP_System.from_known_system("earth_moon")
    system.compute_libration_points()

    # family_result 来自 Continuation 算法
    # ... 此处省略续算法调用 ...

    # 准备数据
    jacobi_values = [orbit.jacobi_constant for orbit in family_result]
    periods = [orbit.period for orbit in family_result]
    stability_values = compute_stability_for_family(family_result, system)

    # 创建 FamilyPlotter
    config = PlotConfig(colormap="coolwarm", dpi=150)
    plotter = FamilyPlotter(system, config=config)

    # 2D 轨道族视图
    plotter.plot_family_2d(
        family_result, jacobi_values,
        title="L1 Lyapunov Family",
        plane="xy",
        save_path="family_2d.png",
    )

    # 3D 轨道族视图
    plotter.plot_family_3d(
        family_result, jacobi_values,
        title="L1 Lyapunov Family (3D)",
        elev=20, azim=-60,
        save_path="family_3d.png",
    )

    # Jacobi-周期-稳定性图
    plotter.plot_jacobi_period_stability(
        jacobi_values, periods, stability_values,
        title="Period & Stability vs Jacobi Constant",
        target_period=6.0,
        save_path="stability.png",
    )

    # 一键四子图概览
    plotter.plot_family_overview(
        family_result, jacobi_values, periods, stability_values,
        suptitle="L1 Lyapunov Family Overview",
        zoom_xlim=(0.4, 0.6),
        zoom_ylim=(-0.15, 0.15),
        save_path="family_overview.png",
    )

if __name__ == "__main__":
    visualize_orbit_family()
```

### 示例：比较多个轨道

```python
import numpy as np
from e2m2e.core.system import CR3BP_System
from e2m2e.visualization import OrbitVisualizer

def compare_orbits():
    """比较多个轨道"""

    system = CR3BP_System.from_known_system("earth_moon")
    system.set_characteristic_scales(384400, 27.32 * 86400)
    system.compute_libration_points()

    viz = OrbitVisualizer(system)

    n_points = 100
    t = np.linspace(0, 2*np.pi, n_points)

    # 轨道1：小振幅
    amp1 = 0.01
    x1 = system.L1[0] + amp1 * np.cos(t)
    y1 = amp1 * np.sin(t)
    orbit1 = np.column_stack([x1, y1, np.zeros(n_points),
                              -amp1*np.sin(t), amp1*np.cos(t), np.zeros(n_points)])

    # 轨道2：中振幅
    amp2 = 0.02
    x2 = system.L1[0] + amp2 * np.cos(t)
    y2 = amp2 * np.sin(t)
    orbit2 = np.column_stack([x2, y2, np.zeros(n_points),
                              -amp2*np.sin(t), amp2*np.cos(t), np.zeros(n_points)])

    # 轨道3：大振幅
    amp3 = 0.03
    x3 = system.L1[0] + amp3 * np.cos(t)
    y3 = amp3 * np.sin(t)
    orbit3 = np.column_stack([x3, y3, np.zeros(n_points),
                              -amp3*np.sin(t), amp3*np.cos(t), np.zeros(n_points)])

    viz.plot_2d_projection(orbit1, plane='xy', color='blue', label=f'Amplitude={amp1}')
    viz.plot_2d_projection(orbit2, plane='xy', color='green', label=f'Amplitude={amp2}')
    viz.plot_2d_projection(orbit3, plane='xy', color='red', label=f'Amplitude={amp3}')

    viz.plot_primary_bodies()
    viz.plot_libration_points()

    viz.axes.legend(title='Lyapunov Orbits')
    viz.axes.set_title('Comparison of Lyapunov Orbits with Different Amplitudes')

    viz.show()
    viz.save('orbit_comparison.png', dpi=300)
```

## 总结

`e2m2e.visualization` 模块采用分层架构，提供强大的轨道可视化功能：

1. **PlotConfig**：集中管理所有样式参数的数据类
2. **OrbitVisualizer**：基类，提供 3D 轨道、2D 投影、天体、平动点等原子化绑图操作
3. **FamilyPlotter**：轨道族专用，一键生成 2D/3D/稳定性/概览图
4. **TransferPlotter**：转移轨道专用，解空间散点图和转移轨迹3D图
5. **compute_stability_for_family**：多进程并行稳定性计算

通过合理使用这些功能，您可以：
- 快速验证轨道设计的正确性
- 直观理解轨道动力学特性
- 生成高质量的学术图表
- 进行轨道族的系统性分析

如有更多问题，请参考模块源代码中的文档字符串或联系项目维护者。
