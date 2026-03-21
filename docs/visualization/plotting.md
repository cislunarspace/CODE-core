# PlottingFunctions

**文件**: `e2m2e/visualization/plotting.py`

## 设计原理

提供轨道和转移轨迹的可视化函数，支持 2D 和 3D 绘图。

## 核心绘图函数

### 轨道绘图

| 函数 | 说明 |
|------|------|
| `plot_orbit_2d(orbit, ax, **kwargs)` | 2D 轨道绘图 |
| `plot_orbit_3d(orbit_or_family, ax, **kwargs)` | 3D 轨道绘图 |
| `plot_orbit_family(family, ax, **kwargs)` | 绘制轨道族 |
| `plot_poincare_section(family, section, ax)` | Poincaré 截面 |

### 转移轨迹绘图

| 函数 | 说明 |
|------|------|
| `plot_transfer_2d(transfer, ax, **kwargs)` | 2D 转移轨迹 |
| `plot_transfer_3d(transfer, ax, **kwargs)` | 3D 转移轨迹 |
| `plot_transfer_trajectory(transfer, ax)` | 绘制转移路径 |
| `plot_delta_v_budget(transfer, ax)` | $\Delta v$ 预算图 |

### 系统绘图

| 函数 | 说明 |
|------|------|
| `plot_system_geometry(system, ax, ...)` | 系统几何结构 |
| `plot_libration_points(system, ax)` | 绘制平动点 |
| `plot_lagrange_surfaces(system, ax)` | 绘制 Lagrange 等势面 |

## 绘图样式配置

```python
# 颜色配置
COLORS = {
    "earth": "#1E90FF",      # 蓝色
    "moon": "#808080",       # 灰色
    "dro": "#FF6B6B",        # 红色
    "ro": "#4ECDC4",         # 青色
    "transfer": "#FFE66D",   # 黄色
}

# 线型配置
LINE_STYLES = {
    "stable": "-",           # 实线
    "unstable": "--",        # 虚线
}
```

## 使用示例

```python
import matplotlib.pyplot as plt
from e2m2e.visualization.plotting import plot_orbit_3d, plot_transfer_3d

# 创建 3D 图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制 DRO
plot_orbit_3d(dro_orbit, ax, color='red', label='DRO')

# 绘制 RO
plot_orbit_3d(ro_orbit, ax, color='cyan', label='RO')

# 绘制转移轨迹
plot_transfer_3d(transfer, ax, color='yellow', label='Transfer')

ax.legend()
plt.show()
```

## 子模块索引

| 子模块 | 文件 | 主要类/函数 |
|--------|------|-------------|
| `core.system` | `core/system.py` | `CR3BP_System` |
| `core.dynamics` | `core/dynamics.py` | `CR3BP_Dynamics` |
| `core.orbit` | `core/orbit.py` | `Orbit`, `OrbitFamily` |
| `core.coordinate` | `core/coordinate.py` | `CoordinateTransformation` |
| `algorithms.continuation` | `algorithms/continuation.py` | `ContinuationMethod` |
| `algorithms.differential_correction` | `algorithms/differential_correction.py` | `DifferentialCorrection` |
| `algorithms.stability` | `algorithms/stability.py` | `StabilityAnalysis` |
| `transfer.inter_orbit` | `transfer/inter_orbit.py` | `DROROTransferSearch` |
| `transfer.earth_moon` | `transfer/earth_moon.py` | `EarthMoonTransfer` |
| `transfer.moon_earth` | `transfer/moon_earth.py` | `MoonEarthTransfer` |
| `visualization.plotting` | `visualization/plotting.py` | 绘图函数 |
