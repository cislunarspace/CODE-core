可视化
======

e2m2e 的可视化模块提供轨道族和转移轨迹的绘图工具，基于 matplotlib。

核心类
------

- **PlotConfig** — 统一的绘图配置（字体、颜色、尺寸、DPI 缩放、天体图标）
- **OrbitVisualizer** — 可视化基类，定义 ``plot()`` 接口
- **FamilyPlotter** — 轨道族可视化：2D/3D 绘图、Jacobi-周期-稳定性分析图
- **TransferPlotter** — 转移轨迹可视化

PlotConfig
----------

.. code-block:: python

   from e2m2e.visualization import PlotConfig

   # 默认配置
   config = PlotConfig()
   config.apply_rcparams()

   # 自定义字体大小
   config = PlotConfig(title=32, label=28, tick=20)
   config.apply_rcparams()

   # 从环境变量读取（仅图标路径与缩放两个变量）
   config = PlotConfig.from_env()

环境变量：

- ``E2M2E_BODY_ICON_PATH`` — 天体图标目录
- ``E2M2E_BODY_ICON_SCALE`` — 天体图标缩放系数

高 DPI 屏幕的缩放独立于环境变量，调用 ``configure_dpi_scaling()`` 自动检测并调整。

FamilyPlotter
-------------

.. code-block:: python

   from e2m2e.visualization import FamilyPlotter, PlotConfig

   config = PlotConfig(title=32, label=28)
   config.apply_rcparams()

   plotter = FamilyPlotter(system, config)

   # 2D 轨道族（按 Jacobi 常数着色）
   plotter.plot_family_2d(family, jacobi_values, title="DRO Family")

   # 3D 轨道族
   plotter.plot_family_3d(family, jacobi_values, title="DRO Family 3D")

   # Jacobi-周期-稳定性组合分析图
   plotter.plot_jacobi_period_stability(
       jacobi_values, periods, stability_values,
       title="DRO Family 分析",
   )

TransferPlotter
---------------

.. code-block:: python

   from e2m2e.visualization import TransferPlotter

   plotter = TransferPlotter(system, config)

   # 绘制转移轨迹与出发/到达轨道
   plotter.plot_transfer_orbit(
       departure_orbit=dro_orbit,
       arrival_orbit=ro_orbit,
       transfer_trajectory=result.transfer_trajectory,
       departure_state=result.departure_state,
       insertion_state=result.insertion_state,
       label="DRO-RO Transfer",
   )

投影平面
--------

``ProjectionPlane`` 枚举指定 2D 绘图的投影平面：

- ``XY`` — x-y 平面（默认）
- ``XZ`` — x-z 平面
- ``YZ`` — y-z 平面

可运行示例见 ``examples/visualization_example.py``。
