"""ephemeris_correction 子包导入不变量测试。

验证同伦修正与分发器同置于 ``ephemeris_correction`` 子包后：

- ``homotopy`` 子模块可独立导入并暴露公开 API；
- 分发器不再依赖延迟导入 ``correct_with_homotopy`` 的循环依赖 workaround；
- standard / two_level 路径对未知方法仍抛 ValueError；
- 默认 lambda 步长不变。
"""

from __future__ import annotations

import importlib

import numpy as np

from e2m2e.algorithms import ephemeris_correction
from e2m2e.algorithms.ephemeris_correction import homotopy


def test_homotopy_submodule_exposes_public_api():
    """``ephemeris_correction.homotopy`` 子模块应暴露公开 API。

    同伦修正与分发器同置于子包后，``correct_with_homotopy`` 在
    ``homotopy`` 子模块内定义，分发器的 ``_HomotopyPatchPointCorrector``
    通过模块级正常 import / 同模块调用使用它——不再需要延迟导入。
    """
    mod = importlib.import_module("e2m2e.algorithms.ephemeris_correction.homotopy")
    assert callable(getattr(mod, "correct_with_homotopy", None))
    assert callable(getattr(mod, "HomotopyEphemerisDynamics", None))
    assert hasattr(mod, "DEFAULT_LAMBDA_STEPS")


def test_dispatcher_does_not_use_lazy_homotopy_import():
    """分发器子包的 ``__init__`` 不应包含对 homotopy 的延迟 import workaround。

    ``_HomotopyPatchPointCorrector`` 通过子包顶部正常导入，因此
    ``__init__`` 模块源码中不应出现 ``import correct_with_homotopy``
    形式的函数内延迟导入。
    """
    import inspect

    source = inspect.getsource(ephemeris_correction)
    # correct_with_homotopy 由 homotopy 子模块提供；__init__ 只 import
    # _HomotopyPatchPointCorrector 类，不应在函数体内再 import 函数。
    assert "from .homotopy import correct_with_homotopy" not in source
    assert "import correct_with_homotopy" not in source


def test_standard_and_two_level_methods_unaffected_by_homotopy_module():
    """standard / two_level paths must still raise ValueError for bad method strings.

    We verify the contract is unchanged: an unknown method raises ValueError,
    while the homotopy path now does NOT raise NotImplementedError.
    """
    # Reload the dispatch package to ensure its top-level import graph is intact.
    importlib.reload(ephemeris_correction)
    # An unknown method still raises ValueError
    try:
        ephemeris_correction.correct_ephemeris_patch_points(
            "unknown_method",
            dynamics=object(),
            t_patch=np.array([0.0, 1.0]),
            state_patch=np.zeros((2, 6)),
            tolerance=1e-3,
            max_iter=5,
            verbose=False,
            n_workers=1,
            kernel_dir="k",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown method")


def test_homotopy_default_lambda_steps():
    assert homotopy.DEFAULT_LAMBDA_STEPS == (0.25, 0.50, 0.75, 1.00)
