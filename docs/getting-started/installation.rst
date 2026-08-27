Installation
============

e2m2e installs with uv (recommended), conda, or pip.

Requirements:

- Python >= 3.10
- Rust toolchain (only needed for source builds, to compile the integrator kernel)

The recommended package manager is `uv <https://docs.astral.sh/uv/>`__; it is
required only on the uv install path — conda/pip users don't need it.

Install with uv (recommended):

.. code-block:: bash

   uv pip install e2m2e

To use in your own project:

.. code-block:: bash

   uv init my-project && cd my-project
   uv add e2m2e

conda: there is no conda-forge package; use conda only to create and manage the
Python environment, installing e2m2e with pip inside it.

.. code-block:: bash

   conda create -n e2m2e python=3.12
   conda activate e2m2e
   pip install e2m2e

pip:

.. code-block:: bash

   pip install e2m2e

Optional extras:

.. code-block:: bash

   pip install e2m2e[normal-form]   # normal-form reduction
   pip install "e2m2e[mcp]"         # MCP server (e2m2e mcp-serve; see "Using e2m2e through MCP")

Released wheels embed CSPICE (statically linked; attribution in the repo NOTICE),
so Rust fast paths — STM propagation, shooting, third-body gravity — work out of
the box.

Building from source requires the `Rust toolchain
<https://www.rust-lang.org/tools/install>`_ because the integrator core is Rust
(PyO3 bindings) built via `maturin <https://www.maturin.rs/>`_.

Source development has a single entry command that sequentially syncs deps,
fetches data, and builds the extension (all idempotent):

.. code-block:: bash

   git clone https://github.com/cislunarspace/e2m2e.git
   cd e2m2e
   make dev         # Single entry: uv sync deps (no extension build) + fetch CSPICE build
                    # package & SPICE kernels + maturin develop build/install of the Rust
                    # extension (spice enabled by default, debug)
   make dev-release # Same as dev but --release (for benchmarks / long-range propagation)

To fetch data without building (e.g., lint-only CI), run ``make setup`` (CSPICE
build package + SPICE kernels, idempotent).

.. warning::

   Do not run bare ``uv sync``. In ``uv.lock`` e2m2e is an editable package;
   bare ``uv sync`` triggers maturin to build the Rust extension immediately:
   the build needs ``CSPICE_DIR`` (otherwise cspice-sys fails outright with
   ``CSPICE_DIR environment variable was not provided``), and it duplicates what
   ``make dev`` does anyway. On this error, switch to ``make dev``.

Builds go through the `Makefile`, which resolves and exports ``CSPICE_DIR``
automatically via ``scripts/download_cspice.py``. CSPICE always comes from GitHub
release prebuilt packages; missing ``CSPICE_DIR`` fails the build outright (the
``cspice-sys`` ``downloadcspice`` feature is not used, avoiding unreachable NAIF
source downloads from domestic networks). Bare ``cargo`` / ``maturin`` commands
need you to export
``CSPICE_DIR=$(python3 scripts/download_cspice.py --print-cspice-dir)`` yourself.

.. note::

   On Windows, run Rust tests with ``make test-rust``. The test binary depends on
   ``python3.dll``, which lives at the Python installation root
   (``sys.base_prefix``), not inside the venv's ``Scripts/``. The Makefile
   auto-detects and adds it to the test process PATH; if detection fails, pass
   ``make test-rust PYTHON_DLL_DIR=<directory containing python*.dll>`` explicitly.
   When debugging ``0xc0000135`` by hand, run ``dumpbin /DEPENDENTS`` or
   ``dumpbin /IMPORTS`` on the failing test EXE first to identify the actually
   missing DLL; never put CSPICE's ``lib/`` (a static-library directory) on PATH.

spice is the default feature (`crates/*/Cargo.toml default=["spice"]` +
``pyproject features=["spice"]`` as double insurance); no no-spice subset is
produced; when the Rust extension is unavailable the library raises explicitly
instead of silently degrading to pure-Python paths (ADR 0020).

SPICE kernels: ephemeris dynamics need NASA SPICE kernel files placed under
``kernels/`` or a path given by ``$SPICE_KERNEL_DIR``.

Users behind domestic networks should download from the project's
`GitHub Release <https://github.com/cislunarspace/e2m2e/releases>`_: the
``kernels-v1`` asset packages all required kernels; extract into ``kernels/``:

- ``de430.bsp``, ``de440s.bsp``: JPL planetary ephemerides
- ``earth_latest_high_prec.bpc``, ``SPICEEarthPredictedKernel.bpc``: Earth rotation (ITRF93 high precision)
- ``SPICELunaCurrentKernel.bpc``, ``SPICELunaFrameKernel.tf``: lunar attitude & frames (MOON_PA)
- ``naif0011.tls``, ``naif0012.tls``: leap seconds
- ``pck00010.tpc``: planetary constants

Official source (when reachable): `NASA NAIF
<https://naif.jpl.nasa.gov/naif/data.html>`_

Verify the install:

.. code-block:: python

   import e2m2e
   print(e2m2e.__version__)

   # Verify the Rust integrator loads
   from e2m2e.integrators import rk_step, RkMethod
   print("Rust integrator loaded successfully")
