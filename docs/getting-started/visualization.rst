可视化
======

e2m2e 提供轨道族和转移轨迹的绘图工具，基于 matplotlib。

核心类
------

- **PlotConfig** — 统一的绘图配置（字体、颜色、尺寸、DPI 缩放）
- **FamilyPlotter** — 轨道族可视化：2D/3D 绘图、Jacobi-周期-稳定性分析图
- **TransferPlotter** — 转移轨迹可视化
- **OrbitVisualizer** — 可视化基类

配置绘图参数
------------

.. code-block:: python

   from e2m2e.tools.viz import PlotConfig

   # 使用默认配置
   config = PlotConfig()
   config.apply_rcparams()

   # 自定义字体大小
   config = PlotConfig(title=32, label=28, tick=20)
   config.apply_rcparams()

   # 从环境变量读取（支持高 DPI 缩放）
   config = PlotConfig.from_env()

绘制轨道族
----------

.. code-block:: python

   from e2m2e.tools.viz import PlotConfig, FamilyPlotter

   config = PlotConfig(title=32, label=28)
   config.apply_rcparams()

   plotter = FamilyPlotter(system, config)

   # 2D 轨道族（按 Jacobi 常数着色）
   plotter.plot_family_2d(family, jacobi_values, title="DRO Family")

   # 3D 轨道族
   plotter.plot_family_3d(family, jacobi_values, title="DRO Family 3D")

   # Jacobi-周期-稳定性组合图（三个序列与 family 中轨道一一对应）
   from e2m2e.algorithm.stability import StabilityAnalysis

   periods = [orb.period for orb in family]
   stability_values = [
       StabilityAnalysis(orb, dynamics).classify_orbit()["max_eigenvalue_magnitude"]
       for orb in family
   ]
   plotter.plot_jacobi_period_stability(
       jacobi_values, periods, stability_values, title="DRO Family Analysis"
   )

绘制转移轨迹
------------

.. code-block:: python

   from e2m2e.tools.viz import TransferPlotter

   plotter = TransferPlotter(system, config)

   # 绘制转移轨迹与出发/到达轨道（3D 视图）
   plotter.plot_transfer_orbit(
       departure_orbit=dro_orbit,
       arrival_orbit=ro_orbit,
       transfer_trajectory=result.transfer_trajectory,
       departure_state=result.departure_state,
       insertion_state=result.insertion_state,
       label="DRO-RO Transfer",
   )

天体图标
--------

可视化支持在轨道图上绘制天体图标（地球、月球）。图标路径通过环境变量
``E2M2E_BODY_ICON_PATH`` 配置，缩放系数通过 ``E2M2E_BODY_ICON_SCALE`` 配置。
