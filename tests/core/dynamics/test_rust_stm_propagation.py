"""Rust propagate_with_stm_py 对接测试。

验证 EphemerisDynamics 的 Rust 快速路径与 SciPy 路径
在 LEO 一个周期内的 STM 一致性。
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

pytestmark = pytest.mark.spice


@pytest.fixture
def leo_state():
    """近地轨道初始状态 (J2000, km, km/s)"""
    r = 6778  # 地球半径 + 400 km
    v = np.sqrt(398600.436 / r)  # 圆轨道速度
    return np.array([r, 0, 0, 0, v, 0])


@pytest.fixture
def reference_et(spice_manager, reference_epoch):
    """参考历元的 ET 秒数"""
    return spice_manager.utc_to_et(reference_epoch)


class TestRustStmAvailability:
    """测试 Rust STM 模块是否可用"""

    def test_rust_stm_import_flag(self):
        """应能检测到 propagate_with_stm_py 是否可用"""
        from e2m2e.algorithm.dynamics.ephemeris_dynamics import _HAS_RUST_STM

        # 无论是否安装了 spice wheel，flag 都应该是 bool
        assert isinstance(_HAS_RUST_STM, bool)


class TestRustStmPropagation:
    """测试 Rust STM 传播的正确性"""

    def test_rust_propagate_with_stm_shape(self, spice_eph_dynamics, reference_et, leo_state):
        """Rust 路径应返回正确形状的 states/stm/time"""
        from e2m2e.algorithm.dynamics.ephemeris_dynamics import _HAS_RUST_STM

        if not _HAS_RUST_STM:
            pytest.skip("propagate_with_stm_py 不可用（未安装 spice wheel）")

        t_span = (reference_et, reference_et + 5400)  # 1.5 小时
        t_eval = np.linspace(t_span[0], t_span[1], 50)
        result = spice_eph_dynamics._propagate_with_stm_rust(
            leo_state, t_span, t_eval, max_step=60.0
        )

        assert result["states"].shape == (50, 6)
        assert result["stm"].shape == (50, 6, 6)
        assert result["time"].shape == (50,)

    def test_rust_stm_initial_is_identity(self, spice_eph_dynamics, reference_et, leo_state):
        """Rust 路径的初始 STM 应为单位矩阵"""
        from e2m2e.algorithm.dynamics.ephemeris_dynamics import _HAS_RUST_STM

        if not _HAS_RUST_STM:
            pytest.skip("propagate_with_stm_py 不可用（未安装 spice wheel）")

        t_span = (reference_et, reference_et + 3600)
        t_eval = np.linspace(t_span[0], t_span[1], 100)
        result = spice_eph_dynamics._propagate_with_stm_rust(
            leo_state, t_span, t_eval, max_step=60.0
        )

        assert_allclose(result["stm"][0], np.eye(6), atol=1e-6)

    def test_rust_vs_python_stm_leo_one_period(self, spice_eph_dynamics, reference_et, leo_state):
        """Rust 与 Python STM 在 LEO 一个周期内一致性"""
        from e2m2e.algorithm.dynamics.ephemeris_dynamics import _HAS_RUST_STM

        if not _HAS_RUST_STM:
            pytest.skip("propagate_with_stm_py 不可用（未安装 spice wheel）")

        # LEO 约 90 分钟
        period = 2 * np.pi * np.sqrt(6778**3 / 398600.436)
        t_span = (reference_et, reference_et + period)
        t_eval = np.linspace(t_span[0], t_span[1], 20)
        max_step = spice_eph_dynamics._get_max_step(t_span)

        # Rust 路径
        rust_result = spice_eph_dynamics._propagate_with_stm_rust(
            leo_state, t_span, t_eval, max_step
        )

        # Python 路径（基类 SciPy）
        from e2m2e.core.dynamics import Dynamics

        python_result = Dynamics._propagate_with_stm(
            spice_eph_dynamics,
            leo_state,
            t_span,
            t_eval,
            max_step,
            with_jacobi=False,
        )

        # 状态对比：用 atol 1e-3 km（1 米），因为接近零的分量相对误差无意义
        assert_allclose(
            rust_result["states"],
            python_result["states"],
            atol=1e-3,
            err_msg="Rust 与 Python 状态传播结果不一致",
        )

        # STM 对比：独立 DOP853 实现累积差异，max abs diff ~1e-4
        assert_allclose(
            rust_result["stm"],
            python_result["stm"],
            atol=1e-3,
            err_msg="Rust 与 Python STM 结果不一致",
        )

    def test_rust_propagate_api_entry(self, spice_eph_dynamics, reference_et, leo_state):
        """通过 propagate(with_stm=True) 入口应自动走 Rust 路径"""
        from e2m2e.algorithm.dynamics.ephemeris_dynamics import _HAS_RUST_STM

        if not _HAS_RUST_STM:
            pytest.skip("propagate_with_stm_py 不可用（未安装 spice wheel）")

        t_span = (reference_et, reference_et + 3600)
        result = spice_eph_dynamics.propagate(leo_state, t_span, with_stm=True)

        assert "stm" in result
        assert result["stm"].shape[1:] == (6, 6)
        assert np.all(np.isfinite(result["states"]))
        assert np.all(np.isfinite(result["stm"]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
