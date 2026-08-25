安装
====

e2m2e 支持 uv（推荐）、conda、pip 三种安装方式。

依赖要求
--------

- Python >= 3.10
- Rust 工具链（从源码安装时需要，用于构建积分器内核）

推荐工具：`uv <https://docs.astral.sh/uv/>`__ 是推荐的包管理器，仅 uv 安装
路径需要；走 conda/pip 路径则不必安装。

uv（推荐）
----------

`uv <https://docs.astral.sh/uv/>`_ 是本项目推荐的包管理器，速度快、依赖解析可靠。

.. code-block:: bash

   uv pip install e2m2e

在自己的项目中使用：

.. code-block:: bash

   uv init my-project && cd my-project
   uv add e2m2e

conda
-----

e2m2e 没有 conda-forge 包，conda 用来创建和管理 Python 环境，环境内仍用 pip 安装：

.. code-block:: bash

   conda create -n e2m2e python=3.12
   conda activate e2m2e
   pip install e2m2e

pip 安装
--------

.. code-block:: bash

   pip install e2m2e

可选依赖：

.. code-block:: bash

   pip install e2m2e[normal-form]   # 标准形化简
   pip install "e2m2e[mcp]"         # MCP 服务器（e2m2e mcp-serve，见「通过 MCP 使用 e2m2e」）

发布的 wheel 内嵌 CSPICE（静态链接，署名见仓库 NOTICE），STM 传播、打靶、
第三体引力等 Rust 快速路径开箱即用。

从源码安装
----------

从源码安装需要 `Rust 工具链 <https://www.rust-lang.org/tools/install>`_，因为积分器核心
由 Rust 实现（PyO3 绑定），通过 `maturin <https://www.maturin.rs/>`_ 构建。

源码开发只有一条入口命令，它依次完成依赖同步、数据拉取与扩展构建（均幂等，
可重复运行）：

.. code-block:: bash

   git clone https://github.com/cislunarspace/e2m2e.git
   cd e2m2e
   make dev         # 唯一入口：uv sync 装依赖（不构建扩展）+ 拉取 CSPICE 编译包与
                    # SPICE 内核 + maturin develop 构建安装 Rust 扩展（spice 默认开启，debug）
   make dev-release # 同 dev 但 --release（性能基准 / 长期预报用）

只需拉取数据、暂不构建时（如 CI 只跑 lint），可用 ``make setup`` （CSPICE 编译包
+ SPICE 内核，幂等）。

.. warning::

   不要裸跑 ``uv sync`` 。e2m2e 在 ``uv.lock`` 中是 editable 包，裸 ``uv sync``
   会当场以 maturin 构建 Rust 扩展：一方面构建需要 ``CSPICE_DIR`` （未导出时
   cspice-sys 直接报错 ``CSPICE_DIR environment variable was not provided`` ），
   另一方面与 ``make dev`` 的构建重复。看到上述报错时，改用 ``make dev`` 。

构建统一走 `Makefile`：它自动从 ``scripts/download_cspice.py`` 解析并
``export CSPICE_DIR`` 。CSPICE 一律取 GitHub release 预编译包；缺
``CSPICE_DIR`` 时构建直接报错（不启用 ``cspice-sys`` 的 ``downloadcspice`` ，
杜绝走国内不可达的 NAIF 源码下载）。裸 ``cargo`` / ``maturin`` 命令需自行
``export CSPICE_DIR=$(python3 scripts/download_cspice.py --print-cspice-dir)`` 。

.. note::

   Windows 上运行 Rust 测试请使用 ``make test-rust`` 。测试二进制依赖
   ``python3.dll`` ；该 DLL 位于 Python 安装根目录（``sys.base_prefix`` ），不在
   虚拟环境 ``Scripts/`` 。Makefile 会自动探测并把该目录加入测试进程 PATH；
   若自动探测失败，可显式执行 ``make test-rust PYTHON_DLL_DIR=<含 python*.dll 的目录>`` 。
   手工排查 ``0xc0000135`` 时，先对失败测试 EXE 执行 ``dumpbin /DEPENDENTS``
   或 ``dumpbin /IMPORTS`` 确认实际缺失的 DLL，不要把 CSPICE 的 ``lib/`` （静态库
   目录）加入 PATH。

spice 是默认 feature（``crates/*/Cargo.toml default=["spice"]`` +
``pyproject features=["spice"]`` 双保险），不再产无 spice 子集；Rust 扩展
不可用时显式报错，不静默降级到纯 Python 路径（ADR 0020）。

SPICE 内核
----------

星历动力学需要 NASA SPICE 内核文件，放置在 ``kernels/`` 目录或 ``$SPICE_KERNEL_DIR``
指定的路径。

国内用户推荐从项目的 `GitHub Release <https://github.com/cislunarspace/e2m2e/releases>`_
下载：``kernels-v1`` 中打包了全部必需内核，下载后放入 ``kernels/`` 目录即可：

- ``de430.bsp`` 、``de440s.bsp``：JPL 行星星历
- ``earth_latest_high_prec.bpc`` 、``SPICEEarthPredictedKernel.bpc``：地球自转（ITRF93 高精度）
- ``SPICELunaCurrentKernel.bpc`` 、``SPICELunaFrameKernel.tf``：月球姿态与坐标架（MOON_PA）
- ``naif0011.tls`` 、``naif0012.tls``：闰秒
- ``pck00010.tpc``：行星常数

官方来源（网络可达时）：`NASA NAIF <https://naif.jpl.nasa.gov/naif/data.html>`_

验证安装
--------

.. code-block:: python

   import e2m2e
   print(e2m2e.__version__)

   # 验证 Rust 积分器可用
   from e2m2e.integrators import rk_step, RkMethod
   print("Rust 积分器加载成功")
