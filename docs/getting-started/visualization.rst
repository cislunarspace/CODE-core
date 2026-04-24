可视化
======

e2m2e 的轨道可视化功能。

3D 轨道绘制
-----------

.. code-block:: python

   from e2m2e.visualization import plot_orbit_3d

   # 绘制轨道
   fig = plot_orbit_3d(orbit)
   fig.show()

动画生成
--------

.. code-block:: python

   from e2m2e.visualization import animate_orbit

   # 生成轨道动画
   anim = animate_orbit(orbit, duration=10.0)
