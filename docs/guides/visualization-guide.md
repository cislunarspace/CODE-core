# e2m2e 可视化模块使用指南

## 概述

`e2m2e.visualization.plotting` 模块提供了圆形限制性三体问题（CR3BP）中轨道的各种可视化功能。本指南将详细介绍如何使用这些功能。

## 目录

1. [快速开始](#快速开始)
2. [核心类：OrbitVisualizer](#核心类orbitvisualizer)
3. [基本可视化功能](#基本可视化功能)
4. [高级可视化功能](#高级可视化功能)
5. [自定义设置](#自定义设置)
6. [常见问题](#常见问题)
7. [示例代码](#示例代码)

## 快速开始

### 安装依赖

```bash
pip install numpy matplotlib
```

### 基本使用示例

```python
import numpy as np
from e2m2e.core.system import CR3BP_System
from e2m2e.visualization.plotting import OrbitVisualizer

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

## 核心类：OrbitVisualizer

### 初始化

```python
viz = OrbitVisualizer(system)
```

**参数：**
- `system`: CR3BP_System对象，必需。用于获取系统参数和平动点位置。

**属性：**
- `figsize`: 图形大小，默认 `(12, 8)`
- `dpi`: 分辨率，默认 `100`
- `orbit_linewidth`: 轨道线宽，默认 `1.5`
- `orbit_alpha`: 轨道透明度，默认 `0.8`
- `primary_body_color`: 主天体颜色，默认 `"gold"`
- `secondary_body_color`: 次天体颜色，默认 `"silver"`

### 方法概览

| 方法 | 描述 | 常用参数 |
|------|------|----------|
| `plot_3d_orbit()` | 绘制3D轨道 | `orbit, color, label, ax, show_start` |
| `plot_2d_projection()` | 绘制2D投影 | `orbit, plane, color, label, ax, show_start` |
| `plot_libration_points()` | 绘制平动点 | `ax, show_labels, is_3d` |
| `plot_primary_bodies()` | 绘制天体 | `ax, is_3d` |
| `plot_orbit_family()` | 绘制轨道族 | `family_result, plane, colormap, ax` |
| `plot_poincare_section()` | 绘制庞加莱截面 | `orbits, plane, value, ax` |
| `plot_jacobi_constant()` | 绘制Jacobi常数 | `orbit, ax` |
| `plot_stability_diagram()` | 绘制稳定性图 | `family_result, ax` |
| `create_overview_plot()` | 创建概览图 | `orbit` |
| `show()` | 显示图形 | 无 |
| `save()` | 保存图形 | `filename, dpi` |

## 基本可视化功能

### 1. 3D轨道可视化

```python
# 绘制3D轨道
ax = viz.plot_3d_orbit(orbit, color='blue', label='3D Orbit')

# 添加天体和平动点
viz.plot_primary_bodies(ax=ax, is_3d=True)
viz.plot_libration_points(ax=ax, is_3d=True)

# 添加图例和显示
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
# 只绘制天体
viz.plot_primary_bodies()
viz.show()

# 只绘制平动点
viz.plot_libration_points()
viz.show()

# 同时绘制天和平动点
viz.plot_primary_bodies()
viz.plot_libration_points()
viz.show()
```

## 高级可视化功能

### 1. 轨道族可视化

```python
# 假设 family_result 是 Continuation 算法返回的结果
viz.plot_orbit_family(family_result, plane='xy', colormap='viridis')
viz.show()
```

### 2. 庞加莱截面

```python
# 绘制 y=0 的庞加莱截面
viz.plot_poincare_section(orbit, plane='y', value=0.0)
viz.show()

# 绘制 x=0.5 的庞加莱截面
viz.plot_poincare_section(orbit, plane='x', value=0.5)
viz.show()
```

### 3. Jacobi常数变化

```python
# 绘制Jacobi常数随时间变化
viz.plot_jacobi_constant(orbit)
viz.show()
```

### 4. 稳定性分析

```python
# 绘制轨道族周期变化图
viz.plot_stability_diagram(family_result)
viz.show()
```

### 5. 综合概览图

```python
# 创建包含四个子图的概览图
fig = viz.create_overview_plot(orbit)
viz.show()

# 保存概览图
viz.save('orbit_overview.png', dpi=300)
```

## 自定义设置

### 修改图形样式

```python
# 修改图形大小和分辨率
viz.figsize = (10, 6)
viz.dpi = 150

# 修改轨道样式
viz.orbit_linewidth = 2.0
viz.orbit_alpha = 0.9

# 修改天体样式
viz.primary_body_color = 'orange'
viz.primary_body_size = 300
viz.secondary_body_color = 'gray'
viz.secondary_body_size = 150

# 修改平动点样式
viz.libration_point_colors = ['darkred', 'darkblue', 'darkgreen', 'darkviolet', 'darkorange']
viz.libration_point_sizes = [120, 120, 120, 180, 180]
```

### 使用现有坐标轴

```python
import matplotlib.pyplot as plt

# 创建自定义图形布局
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 在第一个坐标轴上绘制XY投影
viz.plot_2d_projection(orbit1, plane='xy', color='blue', ax=ax1, label='Orbit 1')
viz.plot_primary_bodies(ax=ax1)
ax1.set_title('Orbit 1 - XY Projection')

# 在第二个坐标轴上绘制XY投影
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
# 绘制第一个轨道
viz.plot_2d_projection(orbit1, plane='xy', color='blue', label='Orbit 1')

# 绘制第二个轨道
viz.plot_2d_projection(orbit2, plane='xy', color='red', label='Orbit 2')

# 添加天和平动点
viz.plot_primary_bodies()
viz.plot_libration_points()

# 显示图例和图形
viz.axes.legend()
viz.show()
```

## 示例代码

### 完整示例：地月系统Lyapunov轨道可视化

```python
import numpy as np
from e2m2e.core.system import CR3BP_System
from e2m2e.core.dynamics import CR3BP_Dynamics
from e2m2e.algorithms.differential_correction import DifferentialCorrection
from e2m2e.visualization.plotting import OrbitVisualizer

def visualize_lyapunov_orbit():
    """可视化地月系统的Lyapunov轨道"""
    
    # 1. 创建地月系统
    print("创建地月系统...")
    system = CR3BP_System.from_known_system("earth_moon")
    system.set_characteristic_scales(384400, 27.32 * 86400)
    system.compute_libration_points()
    
    # 2. 设计Lyapunov轨道（简化示例）
    print("生成示例轨道数据...")
    dynamics = CR3BP_Dynamics(system)
    
    # 创建示例轨道（围绕L1点的近似Lyapunov轨道）
    n_points = 200
    t = np.linspace(0, 2*np.pi, n_points)
    amplitude = 0.02
    
    # 轨道参数
    x0 = system.L1[0]  # L1点的x坐标
    y_amplitude = amplitude
    z_amplitude = 0.0  # 2D轨道
    
    # 生成轨道状态
    x = x0 + amplitude * np.cos(t)
    y = y_amplitude * np.sin(t)
    z = np.zeros_like(t)
    vx = -amplitude * np.sin(t)
    vy = y_amplitude * np.cos(t)
    vz = np.zeros_like(t)
    
    orbit_states = np.column_stack([x, y, z, vx, vy, vz])
    
    # 3. 创建可视化器
    print("创建可视化器...")
    viz = OrbitVisualizer(system)
    
    # 4. 创建概览图
    print("生成概览图...")
    viz.create_overview_plot(orbit_states)
    viz.save('lyapunov_orbit_overview.png', dpi=300)
    viz.show()
    
    # 5. 单独绘制XY投影
    print("生成XY投影图...")
    viz.plot_2d_projection(orbit_states, plane='xy', color='blue', label='Lyapunov Orbit')
    viz.plot_primary_bodies()
    viz.plot_libration_points()
    viz.axes.legend()
    viz.save('lyapunov_orbit_xy.png', dpi=300)
    viz.show()
    
    # 6. 绘制3D视图
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

### 示例：比较多个轨道

```python
def compare_orbits():
    """比较多个轨道"""
    
    # 创建系统
    system = CR3BP_System.from_known_system("earth_moon")
    system.set_characteristic_scales(384400, 27.32 * 86400)
    system.compute_libration_points()
    
    # 创建可视化器
    viz = OrbitVisualizer(system)
    
    # 生成多个示例轨道
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
    
    # 绘制所有轨道
    viz.plot_2d_projection(orbit1, plane='xy', color='blue', label=f'Amplitude={amp1}')
    viz.plot_2d_projection(orbit2, plane='xy', color='green', label=f'Amplitude={amp2}')
    viz.plot_2d_projection(orbit3, plane='xy', color='red', label=f'Amplitude={amp3}')
    
    # 添加天和平动点
    viz.plot_primary_bodies()
    viz.plot_libration_points()
    
    # 添加图例和标题
    viz.axes.legend(title='Lyapunov Orbits')
    viz.axes.set_title('Comparison of Lyapunov Orbits with Different Amplitudes')
    
    # 显示和保存
    viz.show()
    viz.save('orbit_comparison.png', dpi=300)
```

## 总结

`e2m2e.visualization.plotting` 模块提供了强大的轨道可视化功能，包括：

1. **基本可视化**：3D轨道、2D投影、天体和平动点
2. **高级分析**：轨道族、庞加莱截面、Jacobi常数、稳定性
3. **自定义功能**：灵活的样式设置、坐标轴控制、图形保存
4. **易用性**：清晰的API、详细的错误提示、丰富的示例

通过合理使用这些功能，您可以：
- 快速验证轨道设计的正确性
- 直观理解轨道动力学特性
- 生成高质量的学术图表
- 进行轨道族的系统性分析

如有更多问题，请参考模块源代码中的文档字符串或联系项目维护者。