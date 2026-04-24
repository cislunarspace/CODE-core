轨道
====

CR3BP 轨道类型与计算方法。

轨道类型
--------

- **周期轨道**: halo 轨道、Lyapunov 轨道、Lissajous 轨道
- **转移轨道**: 低能转移、弱稳定边界转移
- **拟周期轨道**: 围绕周期轨道的拟周期运动

轨道表示
--------

轨道用 6 维状态向量表示：位置 (x, y, z) 和速度 (ẋ, ẏ, ż)。

.. code-block:: python

   import numpy as np

   # 状态向量
   state = np.array([x, y, z, vx, vy, vz])
