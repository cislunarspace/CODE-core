# Visualization 子包

**目录**: `e2m2e/visualization/`

原单体文件 `plotting.py`（1650 行）已重构为 6 个模块的子包结构。

## 模块结构

```
visualization/
├── __init__.py        # 公共导出
├── config.py          # PlotConfig 配置数据类
├── base.py            # OrbitVisualizer 基类
├── family.py          # FamilyPlotter 轨道族绘图
├── transfer.py        # TransferPlotter 转移轨迹绘图
├── stability.py       # 稳定性计算（并行）
└── plotting.py        # 向后兼容重导出垫片
```

## 模块详情

### `config.py` — 绘图配置

`PlotConfig` 数据类，集中管理所有可视化参数：

| 属性 | 说明 |
|------|------|
| `title_fontsize` / `label_fontsize` / `tick_fontsize` | 标题、轴标签、刻度字号 |
| `legend_fontsize` / `colorbar_fontsize` | 图例、colorbar 字号 |
| `suptitle_fontsize` / `lp_label_fontsize` | 总标题、平动点标签字号 |
| `colormap` | 颜色映射名称 |
| `body_colors` / `body_sizes` | 天体颜色与大小 |
| `figure_sizes` | 2D / 3D 图形尺寸 |
| `orbit_linewidth` / `orbit_alpha` | 轨道线宽与透明度 |
| `title_y_offsets` | 标题 y 偏移量 |

**方法：**

| 方法 | 说明 |
|------|------|
| `apply_rcparams()` | 应用全部 matplotlib rc 设置（含学术字体 Times New Roman、stix math） |
| `get_cmap()` | 获取当前配置的 colormap 实例 |

### `base.py` — 可视化基类

`OrbitVisualizer` 基类 + `ProjectionPlane` 枚举。

**`ProjectionPlane` 枚举值：** `XY`、`XZ`、`YZ`

**`OrbitVisualizer` 方法：**

| 方法 | 说明 |
|------|------|
| `plot_2d_projection(orbit, plane, ax)` | 2D 投影绘图 |
| `plot_3d_orbit(orbit, ax)` | 3D 轨道绘图 |
| `plot_primary_bodies(system, ax)` | 绘制主天体 |
| `plot_libration_points(system, ax)` | 绘制平动点 |
| `show()` | 显示图形 |
| `save(path)` | 保存图形 |
| `_extract_states(orbit)` | 从轨道对象提取状态数组 |
| `_sort_points_by_nearest_neighbor(points)` | 最近邻排序（消除折线乱序） |

### `family.py` — 轨道族绘图

`FamilyPlotter(OrbitVisualizer)`，负责轨道族的完整可视化。

**公共方法：**

| 方法 | 说明 |
|------|------|
| `plot_family_2d(family, plane, ax)` | 轨道族 2D 投影 |
| `plot_family_3d(family, ax)` | 轨道族 3D 绘图 |
| `plot_jacobi_period_stability(family, ax)` | Jacobi 常数 / 周期 / 稳定性综合图 |
| `plot_family_overview(family)` | 轨道族总览图 |

**内部辅助方法：**

| 方法 | 说明 |
|------|------|
| `_draw_orbit_loop_2d(orbit, plane, ax)` | 绘制单条轨道 2D 闭环 |
| `_draw_orbit_loop_3d(orbit, ax)` | 绘制单条轨道 3D 闭环 |
| `_add_colorbar(mappable, ax)` | 添加 colorbar |
| `_style_2d_ax(ax, plane)` | 设置 2D 坐标轴样式 |
| `_style_3d_ax(ax)` | 设置 3D 坐标轴样式 |
| `_get_jacobi_norm(family)` | 获取 Jacobi 常数归一化数组 |

### `transfer.py` — 转移轨迹绘图

`TransferPlotter(OrbitVisualizer)`，负责转移轨道的可视化。

| 方法 | 说明 |
|------|------|
| `plot_solution_plane(transfer, ax)` | 绘制解平面图 |
| `plot_transfer_orbit(transfer, ax)` | 绘制转移轨道 |

### `stability.py` — 稳定性计算

| 函数 | 说明 |
|------|------|
| `compute_stability_for_family(family)` | 使用 `ProcessPoolExecutor` 并行计算轨道族稳定性指数 |

### `plotting.py` — 向后兼容垫片

提供旧 API 的重导出以保持向后兼容，并暴露：

| 函数 | 说明 |
|------|------|
| `configure_academic_fonts()` | 配置学术出版级字体（Times New Roman + stix math） |

### `__init__.py` — 公共导出

```python
from e2m2e.visualization import (
    PlotConfig,
    OrbitVisualizer,
    ProjectionPlane,
    FamilyPlotter,
    TransferPlotter,
    compute_stability_for_family,
)
```

## 使用示例

```python
from e2m2e.visualization import PlotConfig, FamilyPlotter, TransferPlotter

# 应用学术字体与样式
config = PlotConfig()
config.apply_rcparams()

# 轨道族可视化
fp = FamilyPlotter(config=config)
fp.plot_family_overview(family)
fp.plot_jacobi_period_stability(family)
fp.show()

# 转移轨迹可视化
tp = TransferPlotter(config=config)
tp.plot_transfer_orbit(transfer)
tp.save("transfer.png")
```

## 向后兼容

旧的 `from e2m2e.visualization.plotting import ...` 导入路径仍然可用，`plotting.py` 垫片会自动重导出到新模块。
