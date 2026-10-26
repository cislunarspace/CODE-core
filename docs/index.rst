e2m2e 文档
==========

e2m2e（Earth to Moon, Moon to Earth）是地月空间算法工具集，覆盖 CR3BP
任务轨道与轨道族、星历传播、转移设计、轨道保持、时空坐标转换、轨道库与
地月空间分区分析。数值内核用 Rust 实现（PyO3 扩展），Python 侧负责算法
编排与接口层。

调用方通过四条调用链使用同一套能力：

- 进程内 import：``from e2m2e.api import Facade``，Python 直接调用。
- CLI：``e2m2e <工具名>``，命令行调用。
- MCP：``e2m2e mcp-serve``，把工具注册到 MCP 客户端。
- GUI sidecar：``e2m2e serve-stdio``，按 stdio JSON 行与二进制帧协议接入
  同一执行入口。

四条调用链共用同一份工具清单。工具名与参数模型以
``tool_inventory(Facade())`` 为准，MCP schema 与 CLI 参数都从请求模型生成。

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
     version = {5.9.11},
     year = {2026},
   }

许可证
------

e2m2e 以 `Apache-2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_ 发布。
二进制发行版静态链接的第三方组件见仓库 `NOTICE
<https://github.com/cislunarspace/CODE-core/blob/main/NOTICE>`_：

- CSPICE Toolkit，NASA JPL／NAIF 开发并提供。
- Lambert 求解器，移植自 Fortran-Astrodynamics-Toolkit（BSD-3-Clause）。

索引与表格
==========

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
