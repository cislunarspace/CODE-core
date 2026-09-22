示例脚本
========

仓库 ``examples/`` 下的四个脚本演示 **完整任务链路** ，与教程的差异：

- 走 algorithm 层细粒度 API（``e2m2e.algorithm.design``、
  ``e2m2e.algorithm.station_keeping`` 等）并直接使用 ``DesignOrbitRequest``
  等请求模型，不用 ``Facade``；
- 直接可跑、产出 PNG 图（共用 ``_plot_setup.py`` 做跨平台 CJK 字体配置）；
- 统一 CLI：``--save`` 存图（无头服务器可用），``--log-level`` 控制日志。

运行方式（在仓库根，需先 :doc:`/install` 完成 ``make dev``）：

.. code-block:: bash

   uv run --no-sync python examples/main_design.py --save

.. toctree::
   :maxdepth: 1

   design
   propagate
   control
   transfer
