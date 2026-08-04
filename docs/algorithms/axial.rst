Axial 轨道族
============

Axial 轨道族是 CR3BP 共线平动点附近的另一类周期轨道族，
基于 Gómez Type B 分岔（Gómez et al., 2001）实现。

与 Halo 轨道（从 L1/L2 平动点的特征值分岔产生）不同，
Axial 轨道的分岔机制来自平动点附近动力学的 Type B 不稳定性。

理论背景
--------

Gómez et al. (2001) 证明共线平动点附近的周期轨道存在多种分岔族：

- **Type A**：标准 Halo 分岔（面外振幅从零增长）
- **Type B**：Axial 分岔，轨道形态与 Halo 不同，主要沿轴向运动

Axial 轨道在某些参数范围内可提供不同于 Halo 的覆盖特性，
适合特定任务需求下的轨道选择。

使用方式
--------

通过 ``Facade`` 一档接口设计：

.. code-block:: python

   from e2m2e.api import Facade

   facade = Facade()
   result = facade.design_orbit(
       orbit_type="Axial",
       collinear_point=2,
       amplitude=20000.0,  # km
       epoch=[2024, 1, 1, 0, 0, 0.0],
       duration=1.0,  # 年
   )

或使用底层 API：

.. code-block:: python

   from e2m2e.algorithm.design import design_orbit

   result = design_orbit(
       orbit_type="Axial",
       collinear_point=2,
       amplitude=20000.0,
       epoch=[2024, 1, 1, 0, 0, 0.0],
   )

参考文献
--------

- Gómez, G., Llibre, J., Martínez, R., & Simó, C. (2001). *Dynamics and Mission Design Near Libration Points*. World Scientific.
