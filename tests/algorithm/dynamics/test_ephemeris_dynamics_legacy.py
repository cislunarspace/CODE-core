"""EphemerisDynamics 遗留消费者所需的最小接口契约。"""

import re

import numpy as np
import pytest
from numpy.testing import assert_allclose

from e2m2e.algorithm.dynamics import Dynamics, EphemerisDynamics
from e2m2e.data.constants import Datum

pytestmark = [pytest.mark.interface, pytest.mark.spice]


@pytest.fixture
def leo_state():
    radius_km = Datum.WGS84.earth_radius_km + 400.0
    circular_speed_km_s = np.sqrt(Datum.DE440.earth_gm / radius_km)
    return np.array([radius_km, 0.0, 0.0, 0.0, circular_speed_km_s, 0.0])


@pytest.fixture
def reference_et(spice_manager, reference_epoch):
    return spice_manager.utc_to_et(reference_epoch)


def test_legacy_dynamics_keeps_the_dynamics_interface(spice_eph_dynamics, spice_eph_system):
    assert isinstance(spice_eph_dynamics, Dynamics)
    assert isinstance(spice_eph_dynamics, EphemerisDynamics)
    assert spice_eph_dynamics.system is spice_eph_system


def test_legacy_equations_preserve_state_vector_layout(spice_eph_dynamics, reference_et, leo_state):
    derivative = spice_eph_dynamics.equations_of_motion(reference_et, leo_state)
    assert derivative.shape == (6,)
    assert_allclose(derivative[:3], leo_state[3:])


def test_legacy_stm_equation_starts_from_a_times_identity(
    spice_eph_dynamics, reference_et, leo_state
):
    augmented = np.concatenate([leo_state, np.eye(6).ravel()])
    derivative = spice_eph_dynamics.equations_with_stm(reference_et, augmented)
    assert derivative.shape == (42,)
    assert_allclose(
        derivative[6:].reshape(6, 6),
        spice_eph_dynamics.compute_jacobian_A(reference_et, leo_state),
    )


def test_legacy_propagation_returns_aligned_state_and_stm_histories(
    spice_eph_dynamics, reference_et, leo_state
):
    t_span = (reference_et, reference_et + 3_600.0)
    t_eval = np.linspace(*t_span, 10)
    result = spice_eph_dynamics.propagate(leo_state, t_span, t_eval=t_eval, with_stm=True)
    assert result["time"].shape == (10,)
    assert result["states"].shape == (10, 6)
    assert result["stm"].shape == (10, 6, 6)
    assert_allclose(result["states"][0], leo_state, atol=1e-9)
    assert_allclose(result["stm"][0], np.eye(6), atol=1e-9)


def test_legacy_propagation_supports_zero_and_backward_duration(
    spice_eph_dynamics, reference_et, leo_state
):
    zero = spice_eph_dynamics.propagate(leo_state, (reference_et, reference_et))
    backward = spice_eph_dynamics.propagate(leo_state, (reference_et, reference_et - 3_600.0))
    assert_allclose(zero["states"][0], leo_state, atol=1e-9)
    assert backward["time"][0] > backward["time"][-1]


def test_legacy_max_step_is_capped_by_short_propagation_duration(spice_eph_dynamics):
    assert spice_eph_dynamics._get_max_step((0.0, 1_000.0)) == pytest.approx(100.0)
    assert spice_eph_dynamics._get_max_step((0.0, 100.0)) == pytest.approx(10.0)
    assert spice_eph_dynamics._get_max_step((100.0, 0.0)) == pytest.approx(10.0)


def test_legacy_dynamics_rejects_events_explicitly(spice_eph_dynamics, reference_et, leo_state):
    def event(time, state):  # noqa: ARG001
        return float(state[0])

    with pytest.raises(ValueError, match="必须显式指定 backend"):
        spice_eph_dynamics.propagate(leo_state, (reference_et, reference_et + 100.0), events=event)
    with pytest.raises(NotImplementedError, match="事件检测"):
        spice_eph_dynamics.propagate(
            leo_state,
            (reference_et, reference_et + 100.0),
            events=event,
            backend="scipy",
        )


def _kernel_coverage_end_et(mgr, target: str = "MOON", observer: str = "EARTH") -> float:
    """二分定位 ``(target, observer)`` 的 SPK 覆盖末端 et（秒）。

    不硬编码日历日期：不同环境加载的 DE 内核（de440 / de440s / de430）覆盖
    区间不同，硬编码要么让用例退化成初值预检失败，要么根本触发不了越界。
    J2000 必在覆盖内，1e11 秒（约 5138 年）必在任何 DE 内核覆盖外。
    """

    def queryable(et: float) -> bool:
        try:
            mgr.get_body_state(target, et, "J2000", observer)
        except Exception:  # noqa: BLE001 - 任何 SPICE 失败都视为覆盖外
            return False
        return True

    assert queryable(0.0), "J2000 处必须能查到月星历"
    assert not queryable(1.0e11), "1e11 秒处不应有月星历"
    lo, hi = 0.0, 1.0e11
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if queryable(mid):
            lo = mid
        else:
            hi = mid
    return lo


def test_legacy_propagation_beyond_kernel_coverage_reports_real_cause(
    spice_eph_dynamics, spice_manager, leo_state
):
    """越出内核覆盖时错误消息须给出真实 cause，而非通用截断文案（#677）。"""
    wall = _kernel_coverage_end_et(spice_manager)
    t0 = wall - 1.0e5
    t_end = wall + 1.0e5

    with pytest.raises(RuntimeError) as excinfo:
        spice_eph_dynamics.propagate(leo_state, (t0, t_end), t_eval=np.array([t0, t_end]))

    message = str(excinfo.value)
    assert "cause:" in message
    assert "insufficient kernel coverage" in message
    assert "likely cause" not in message
    assert "step size collapsed" not in message


def test_legacy_initial_rhs_failure_reports_real_cause(
    spice_eph_dynamics, spice_manager, leo_state
):
    """初值时刻已在覆盖外时走预检出口，消息须同样带 cause 段（#677）。"""
    wall = _kernel_coverage_end_et(spice_manager)
    t0 = wall + 1.0e5  # 初值已在覆盖外，积分不启动
    t_span = (t0, t0 + 1.0e3)

    with pytest.raises(RuntimeError) as excinfo:
        spice_eph_dynamics.propagate(leo_state, t_span, t_eval=np.array(t_span))

    message = str(excinfo.value)
    assert "initial RHS evaluation failed at t=" in message
    assert "cause:" in message
    assert "insufficient kernel coverage" in message
    assert "likely cause" not in message


@pytest.fixture
def narrow_rust_ephem_cache(reference_et):
    """启用只覆盖 ``[reference_et, reference_et + 3600]`` 的 Rust 星历缓存。"""
    from e2m2e.integrators import disable_ephem_cache, enable_ephem_cache

    enable_ephem_cache(
        targets=[("MOON", "EARTH"), ("SUN", "EARTH")],
        frame_pairs=[],
        et_start=reference_et,
        et_end=reference_et + 3600.0,
        dt=600.0,
    )
    yield
    disable_ephem_cache()


def test_legacy_propagation_outside_cache_window_reports_window(
    spice_eph_dynamics, reference_et, leo_state, narrow_rust_ephem_cache
):
    """越出星历缓存窗口时 cause 须区分窗口外并携带窗口与查询时刻（#677）。"""
    t_span = (reference_et, reference_et + 7200.0)

    with pytest.raises(RuntimeError) as excinfo:
        spice_eph_dynamics.propagate(leo_state, t_span, t_eval=np.array(t_span))

    message = str(excinfo.value)
    assert "EPHEM_CACHE_MISS" in message
    assert "outside cached window" in message
    # 查询时刻与缓存窗口都来自进程状态，不是照抄底层文本：两者须以数值出现。
    assert re.search(r"et -?\d+\.\d+, window \[-?\d+\.\d+, -?\d+\.\d+\]", message)
