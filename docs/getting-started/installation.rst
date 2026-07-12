安装
====

e2m2e 支持 pip 安装和从源码安装。

依赖要求
--------

- Python >= 3.10
- Rust 工具链（从源码安装时需要，用于构建积分器内核）
- [uv](https://docs.astral.sh/uv/)（推荐的包管理器）

pip 安装
--------

.. code-block:: bash

   pip install e2m2e

可选依赖（标准形化简）：

.. code-block:: bash

   pip install e2m2e[normal-form]

从源码安装
----------

.. code-block:: bash

   git clone https://github.com/cislunarspace/e2m2e.git
   cd e2m2e
   uv sync

开发依赖：

.. code-block:: bash

   uv sync --group dev

从源码安装需要 `Rust 工具链 <https://www.rust-lang.org/tools/install>`_，因为积分器核心
由 Rust 实现（PyO3 绑定），通过 `maturin <https://www.maturin.rs/>`_ 构建。

SPICE 内核
----------

星历动力学需要 NASA SPICE 内核文件，放置在 ``kernels/`` 目录或 ``$SPICE_KERNEL_DIR``
指定的路径。

常用内核：

- `de440.bsp <https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/>`_ — JPL DE440 行星星历（推荐，覆盖 1550–2650 年）
- ``moon_pa_de440_200625.bsp`` — 月球姿态
- `pck00011.tpc <https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/>`_ — 行星常数

内核下载：`NASA NAIF <https://naif.jpl.nasa.gov/naif/data.html>`_

验证安装
--------

.. code-block:: python

   import e2m2e
   print(e2m2e.__version__)

   # 验证 Rust 积分器可用
   from e2m2e.integrators import rk_step, RkMethod
   print("Rust 积分器加载成功")
