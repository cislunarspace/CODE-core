"""星历动力学 Rust 绑定的实现等价性测试。

Rust/SciPy 对照只验证两个实现路径没有漂移，不承担动力学物理正确性的
最终证明；后者由 tests/algorithm/dynamics 的定义性测试负责。
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from e2m2e.data.constants import Datum
from e2m2e.exceptions import RustExtensionUnavailableError

pytestmark = [
    pytest.mark.integrator,
    pytest.mark.spice,
]


@pytest.fixture
def leo_state():
    """近地轨道初始状态 (J2000, km, km/s)"""
    r = Datum.WGS84.earth_radius_km + 400.0
    v = np.sqrt(Datum.DE440.earth_gm / r)
    return np.array([r, 0, 0, 0, v, 0])


@pytest.fixture
def reference_et(spice_manager, reference_epoch):
    """参考历元的 ET 秒数"""
    return spice_manager.utc_to_et(reference_epoch)


class TestRustStmAvailability:
    """测试 Rust STM 模块的能力错误。"""

    def test_rust_stm_missing_extension_raises_capability_error(self, monkeypatch):
        """扩展缺失时在使用处给出构建指引，不影响模块导入。"""
        import e2m2e.integrators as integrators

        monkeypatch.setattr(integrators, "_rust_extension", None)
        monkeypatch.setattr(integrators, "_abi_ok", False)
        monkeypatch.setattr(integrators, "propagate_with_stm_py", None)

        with pytest.raises(RustExtensionUnavailableError, match="make dev"):
            integrators.require_rust_extension("propagate_with_stm_py")


class TestRustStmPropagation:
    """测试 Rust STM 传播的正确性"""

    def test_rust_propagate_with_stm_shape(self, spice_eph_dynamics, reference_et, leo_state):
        """Rust 路径应返回正确形状的 states/stm/time"""
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
        t_span = (reference_et, reference_et + 3600)
        t_eval = np.linspace(t_span[0], t_span[1], 100)
        result = spice_eph_dynamics._propagate_with_stm_rust(
            leo_state, t_span, t_eval, max_step=60.0
        )

        assert_allclose(result["stm"][0], np.eye(6), atol=1e-6)

    def test_rust_vs_python_stm_leo_one_period(self, spice_eph_dynamics, reference_et, leo_state):
        """Rust 与 Python STM 在 LEO 一个周期内一致性"""
        # LEO 约 90 分钟
        period = 2.0 * np.pi * np.sqrt(leo_state[0] ** 3 / Datum.DE440.earth_gm)
        t_span = (reference_et, reference_et + period)
        t_eval = np.linspace(t_span[0], t_span[1], 20)
        max_step = spice_eph_dynamics._get_max_step(t_span)

        # Rust 路径
        rust_result = spice_eph_dynamics._propagate_with_stm_rust(
            leo_state, t_span, t_eval, max_step
        )

        # Python 路径（基类 SciPy）
        from e2m2e.algorithm.dynamics.dynamics import Dynamics

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
        t_span = (reference_et, reference_et + 3600)
        result = spice_eph_dynamics.propagate(leo_state, t_span, with_stm=True)

        assert "stm" in result
        assert result["stm"].shape[1:] == (6, 6)
        assert np.all(np.isfinite(result["states"]))
        assert np.all(np.isfinite(result["stm"]))


class TestRustStatePropagation:
    """测试 Rust 纯状态传播（propagate_with_state_py）与 parity。"""

    def test_state_vs_stm_parity(self, spice_eph_dynamics, reference_et, leo_state):
        """纯状态 Rust vs STM Rust：states 前 6 维逐位相等（parity）。

        两条路径同用 solve_ivp_capped（同一 PD78 积分循环、同一 error_dim=6
        误差统计）。星历传播 km 量级状态下，自适应初始步长（~1e9）远大于
        max_step（~540s），初始步长被 max_step 钳位为同一值，故步长序列
        完全一致，states 前 6 维逐位相等。这是"两条路径等价"的直接证据。
        """
        t_span = (reference_et, reference_et + 5400)
        t_eval = np.linspace(t_span[0], t_span[1], 50)
        max_step = spice_eph_dynamics._get_max_step(t_span)

        rust_stm = spice_eph_dynamics._propagate_with_stm_rust(leo_state, t_span, t_eval, max_step)
        rust_state = spice_eph_dynamics._propagate_state_rust(leo_state, t_span, t_eval, max_step)

        # atol=0：逐位相等（步长序列一致的直接推论）
        assert_allclose(
            rust_state["states"],
            rust_stm["states"],
            atol=0.0,
            err_msg="纯状态与 STM 路径 states 不逐位相等（parity 破坏）",
        )

    def test_state_rust_vs_scipy(self, spice_eph_dynamics, reference_et, leo_state):
        """纯状态 Rust vs SciPy 基类：states 一致（独立 DOP853 实现累积差异）。"""
        t_span = (reference_et, reference_et + 5400)
        t_eval = np.linspace(t_span[0], t_span[1], 50)
        max_step = spice_eph_dynamics._get_max_step(t_span)

        rust_state = spice_eph_dynamics._propagate_state_rust(leo_state, t_span, t_eval, max_step)

        # SciPy 基类路径（保 super() 兜底分支做对照）
        from e2m2e.algorithm.dynamics.dynamics import Dynamics

        scipy_result = Dynamics._propagate_state_only(
            spice_eph_dynamics,
            leo_state,
            t_span,
            t_eval,
            max_step,
            with_jacobi=False,
        )

        # atol=1e-3 km（1 米）：与 STM 对照测试同量级
        assert_allclose(
            rust_state["states"],
            scipy_result["states"],
            atol=1e-3,
            err_msg="Rust 纯状态与 SciPy 状态传播结果不一致",
        )

    def test_state_rust_backward(self, spice_eph_dynamics, reference_et, leo_state):
        """纯状态 Rust 应支持双向积分（t_span.1 < t_span.0）。

        与 ``test_propagate_backward_time`` 同链路：覆写 _propagate_state_only 后，
        backward 传播走 Rust 纯状态路径（solve_ivp_capped 的 dir 逻辑）。
        """
        t_span = (reference_et, reference_et - 3600)
        result = spice_eph_dynamics._propagate_state_rust(
            leo_state, t_span, None, spice_eph_dynamics._get_max_step(t_span)
        )

        assert np.all(np.isfinite(result["states"]))
        # t_eval 默认 [t0, t1]，backward 时 t1 < t0，时间应递减
        assert result["time"][0] > result["time"][-1]
        # 首点应等于初值
        assert_allclose(result["states"][0], leo_state, atol=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
