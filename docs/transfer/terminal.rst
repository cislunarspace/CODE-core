端点条件
========

端点条件定义转移设计中出发和到达必须满足的条件。

TerminalCondition
-----------------

:class:`~e2m2e.transfer.terminal.TerminalCondition` 是抽象基类，定义出发状态与到达状态的获取契约：

.. code-block:: python

   from e2m2e.transfer.terminal import TerminalCondition

   class TerminalCondition(ABC):
       def get_initial_state(self) -> np.ndarray:
           """返回出发状态 [x, y, z, vx, vy, vz]"""
           ...

       def get_arrival_state(self, t_transfer: float) -> np.ndarray:
           """返回到达状态（可能依赖传播）"""
           ...

OrbitTerminal
-------------

:class:`~e2m2e.transfer.terminal.OrbitTerminal` 位于某条 ``Orbit`` 上：

- 出发状态取轨道首点
- 到达状态通过动力学传播获取

.. code-block:: python

   from e2m2e.transfer.terminal import OrbitTerminal

   departure = OrbitTerminal(orbit=dro_orbit, dynamics=dynamics, is_departure=True)
   arrival = OrbitTerminal(orbit=ro_orbit, dynamics=dynamics, is_departure=False)

StateTerminal
-------------

:class:`~e2m2e.transfer.terminal.StateTerminal` 固定状态与时间：

- 出发状态与到达状态均为固定值
- 不依赖动力学传播

.. code-block:: python

   from e2m2e.transfer.terminal import StateTerminal

   departure = StateTerminal(state=np.array([r, 0, 0, 0, v, 0]), time=0.0)

与优化器的配合
--------------

端点条件通过 ``Transfer.set_orbit()`` 注入优化器：

.. code-block:: python

   from e2m2e.transfer import Transfer

   transfer = Transfer(dynamics)
   transfer.set_orbit(start=dro_orbit, end=ro_orbit)

   # 内部创建 OrbitTerminal 作为出发和到达条件
