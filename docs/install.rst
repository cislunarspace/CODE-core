安装
====

从 PyPI 安装
------------

用 `uv <https://docs.astral.sh/uv/>`_ 安装：

.. code-block:: bash

   uv pip install e2m2e

MCP 接口与 CLI 随主包装好；如需 MCP 协议层：

.. code-block:: bash

   uv pip install "e2m2e[mcp]"

SPICE 星历内核不随包分发：``make dev``（源码开发）会自动从仓库 GitHub Release
的 ``kernels-v1`` 下载到仓库根 ``kernels/``；也可以手动下载解压到该目录，或用
环境变量 ``SPICE_KERNEL_DIR`` 指向任意位置。

从源码开发
----------

.. code-block:: bash

   git clone https://github.com/cislunarspace/CODE-core.git
   cd CODE-core
   make dev

``make dev`` 是唯一开发入口，一次完成：同步 Python 依赖、拉取 CSPICE 编译包与
SPICE 内核（幂等）、用 maturin 以 debug 模式构建并安装 Rust 扩展
``e2m2e._integrators``。

构建前置（``make dev`` 自动处理，也可单独执行）：

- CSPICE 编译包：``make cspice`` 运行 ``scripts/download_cspice.py`` 下载预编译包
  （不要从 NAIF 官网手动下载）；
- ``LIBCLANG_PATH``：Rust 绑定生成（bindgen）需要 libclang，Makefile 自动探测导出；
- SPICE 内核：``make kernels`` 单独拉取，或 ``make setup`` 一次完成 CSPICE + 内核。

性能基准或长期预报场景用 ``make dev-release`` 以 ``--release`` 构建扩展。

Windows
-------

Windows 默认没有 ``make``，可用
`Scoop <https://scoop.sh/>`_ 安装：

.. code-block:: powershell

   Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
   irm get.scoop.sh | iex
   scoop install make

其余注意点：

- ``make PYTHON=python`` 覆盖解释器（Windows 官方 Python 只装 ``python.exe``）；
- mypy / pytest 一律经 ``python -m`` 调用，避免 uv 垫片路径问题。

运行配置环境变量
----------------

``Facade`` 的运行配置由 ``e2m2e.api.config.Config`` 承载，构造时注入
（``Facade(config=Config(...))``），缺省从环境变量读取：

===========================  ====================================================
环境变量                      含义
===========================  ====================================================
``SPICE_KERNEL_DIR``          SPICE 内核目录，默认仓库根 ``kernels/``
``E2M2E_CATALOG_DIR``         轨道库目录；未设置时库操作报
                              ``CATALOG_NOT_CONFIGURED``，不建目录
``E2M2E_CATALOG_ENABLED``     产物成功后是否自动入库（``1``/``true``/``yes``），
                              默认关闭——入库是调用方的显式决定
===========================  ====================================================

另有个别环境开关不进 ``Config``：``E2M2E_WSB_PARALLEL``、
``E2M2E_LOW_ENERGY_PARALLEL``、``E2M2E_SEARCH_PARALLEL`` 控制对应 Rust 搜索的
多线程开关，``RAYON_NUM_THREADS`` 控制 rayon 线程数。
