e2m2e 文档
==========

e2m2e（Earth to Moon, Moon to Earth）是地月空间 **算法工具集** ：圆型限制性三体问题
（CR3BP）转移轨道设计、任务轨道设计与保持、轨道预报、时空坐标转换、轨道库与地月
空间分区分析。数值内核用 Rust 实现（PyO3 扩展），Python 侧提供算法编排与接口层。

同一套能力有三条调用链：

- **进程内 import**：``from e2m2e.api import Facade``，Python 直接调用；
- **MCP**：``e2m2e mcp-serve``，在 MCP 客户端里用自然语言驱动；
- **CLI**：``e2m2e <工具名>``，命令行调用。

GUI 经 sidecar stdio 协议（``e2m2e serve-stdio``）使用同一执行入口。

.. toctree::
   :maxdepth: 2
   :caption: 目录:

   install
   tutorials/index
   examples/index
   api/index
   contrib
   glossary

引用
----

如果 e2m2e 对你的研究有帮助，请引用：

.. code-block:: bibtex

   @software{e2m2e,
     title = {e2m2e: Earth to Moon, Moon to Earth Transfer Orbit Design Library},
     author = {ouyangjiahong},
     email = {ouyangjiahong22@nudt.edu.cn},
     url = {https://github.com/cislunarspace/CODE-core},
     version = {5.9.4},
     year = {2026},
   }

许可证
------

e2m2e 以 `Apache-2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_ 发布。
二进制发行版静态链接的第三方组件见仓库 `NOTICE
<https://github.com/cislunarspace/CODE-core/blob/main/NOTICE>`_：

- CSPICE Toolkit——NASA JPL／NAIF 开发并提供；
- Lambert 求解器——移植自 Fortran-Astrodynamics-Toolkit（BSD-3-Clause）。

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
