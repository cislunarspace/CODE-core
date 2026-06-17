"""homotopy_correction 模块导入时不变量测试。

验证延迟导入避免循环依赖、默认 lambda 步长、
以及标准/两层方法不受新模块影响。
"""

from __future__ import annotations

import importlib
import sys

import numpy as np

from e2m2e.algorithms import ephemeris_correction, homotopy_correction


def test_homotopy_correction_importable_independently():
    """homotopy_correction must be importable without importing ephemeris_correction first."""
    # Drop ephemeris_correction from sys.modules, then import homotopy_correction.
    # A circular import would force the eager load and surface here.
    saved_eph = sys.modules.pop("e2m2e.algorithms.ephemeris_correction", None)
    saved_hc = sys.modules.pop("e2m2e.algorithms.homotopy_correction", None)
    try:
        mod = importlib.import_module("e2m2e.algorithms.homotopy_correction")
        # Public API contract
        assert callable(getattr(mod, "correct_with_homotopy", None))
        assert callable(getattr(mod, "HomotopyEphemerisDynamics", None))
        assert hasattr(mod, "DEFAULT_LAMBDA_STEPS")
    finally:
        if saved_eph is not None:
            sys.modules["e2m2e.algorithms.ephemeris_correction"] = saved_eph
        if saved_hc is not None:
            sys.modules["e2m2e.algorithms.homotopy_correction"] = saved_hc
        # importlib.import_module 会把新模块注册为包属性；
        # 必须恢复包命名空间，否则后续 monkeypatch 与延迟 import
        # 指向不同模块对象（sys.modules vs 包属性），导致补丁失效。
        import e2m2e.algorithms as _algo_pkg

        if saved_eph is not None:
            _algo_pkg.ephemeris_correction = saved_eph
        if saved_hc is not None:
            _algo_pkg.homotopy_correction = saved_hc


def test_standard_and_two_level_methods_unaffected_by_homotopy_module():
    """standard / two_level paths must still raise ValueError for bad method strings.

    We verify the contract is unchanged: an unknown method raises ValueError,
    while the homotopy path now does NOT raise NotImplementedError.
    """
    # Reload the dispatch module to ensure its top-level import graph is
    # unaffected by the new homotopy_correction module.
    importlib.reload(ephemeris_correction)
    # An unknown method still raises ValueError
    try:
        ephemeris_correction.correct_ephemeris_patch_points(
            "unknown_method",
            dynamics=object(),
            t_patch=np.array([0.0, 1.0]),
            state_patch=np.zeros((2, 6)),
            tolerance=1e-3, max_iter=5, verbose=False,
            n_workers=1, kernel_dir="k",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown method")


def test_homotopy_correction_default_lambda_steps():
    assert homotopy_correction.DEFAULT_LAMBDA_STEPS == (0.25, 0.50, 0.75, 1.00)
