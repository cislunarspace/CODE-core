教程
====

教程面向 **调用方** ：全部走接口层暴露类（``Facade`` / ``Catalog`` /
``Spatiography``），每篇给一段可直接运行的完整代码、返回对象的解读方式，以及
CLI／MCP 等价入口。示例脚本（``examples/``）走 algorithm 层细粒度 API，见
:doc:`/examples/index`。

前置约定：代码在装好 e2m2e 的环境里运行（见 :doc:`/install`）；涉及星历的任务
要求 SPICE 内核在仓库根 ``kernels/`` 或 ``$SPICE_KERNEL_DIR`` 指向的目录。
交互式探索建议 ``make dev-release`` 构建 Rust 扩展——debug 构建慢约一个数量级
（如 Halo 设计 release 约 1 秒、debug 约 12 秒）。

.. toctree::
   :maxdepth: 1

   concepts
   orbit-design
   orbit-propagation
   station-keeping
   transfer-design
   spacetime-transform
   orbit-catalog
   spatiography
   integration
