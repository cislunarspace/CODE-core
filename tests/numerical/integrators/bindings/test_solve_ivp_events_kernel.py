"""``solve_ivp_events`` 内核分派（``RustEomKernel``）的绑定契约测试。

EOM 内核标识（``cr3bp`` / ``cr3bp-with-stm`` / ``bcr4bp`` / ``bcr4bp-with-stm``）
使每步 RHS 求值留在 Rust 内（复用 ``e2m2e-forces`` 的 CR3BP/BCR4BP EOM/STM
内核，issue #594）；事件函数仍为 Python 回调。数值等价的参照实现 =
同一通用积分器 + Python EOM 回调（两侧喂同一组参数）。

经 Dynamics 编排层的等价测试见
``tests/algorithm/dynamics/test_rust_events_kernel.py``。
"""

import numpy as np
import numpy.typing as npt
import pytest

from e2m2e.integrators import RkMethod, RustEomKernel, solve_ivp_events

pytestmark = pytest.mark.integrator

# 参照实现与内核标识喂同一组参数——本文件验证的是两条路线的数值一致性，
# 参数本身只需物理上合理，不绑定仓库系统常量。
MU = 0.01215058560962404
MU_SUN = 328900.5614
SUN_DISTANCE = 389.1709
SUN_ANGULAR_RATE = -0.9251959665512
SUN_PHASE0 = 0.3

Y0_6D = [0.8, 0.05, 0.01, 0.0, -0.1, 0.02]
T_SPAN = (0.0, 3.0)
T_EVAL = np.linspace(0.0, 3.0, 31)
RTOL = 1e-12
ATOL = 1e-12
MAX_STEP = 0.05


def _cr3bp_parts(t, y):
    """CR3BP 右端与雅可比 A（numpy 参照，dynamics.py/potential.py 的移植）。"""
    x, y_, z, vx, vy, vz = y[:6]
    r1 = max(np.sqrt((x + MU) ** 2 + y_**2 + z**2), 1e-10)
    r2 = max(np.sqrt((x - 1 + MU) ** 2 + y_**2 + z**2), 1e-10)
    ax = 2 * vy + x - (1 - MU) * (x + MU) / r1**3 - MU * (x - 1 + MU) / r2**3
    ay = -2 * vx + y_ - (1 - MU) * y_ / r1**3 - MU * y_ / r2**3
    az = -(1 - MU) * z / r1**3 - MU * z / r2**3
    eom = np.array([vx, vy, vz, ax, ay, az])
    ir1_3, ir2_3 = 1.0 / r1**3, 1.0 / r2**3
    ir1_5, ir2_5 = ir1_3 / r1**2, ir2_3 / r2**2
    xm, dx1, dx2 = 1.0 - MU, x + MU, x - 1.0 + MU
    u_xx = 1.0 - xm * (ir1_3 - 3 * dx1**2 * ir1_5) - MU * (ir2_3 - 3 * dx2**2 * ir2_5)
    u_yy = 1.0 - xm * (ir1_3 - 3 * y_**2 * ir1_5) - MU * (ir2_3 - 3 * y_**2 * ir2_5)
    u_zz = -xm * (ir1_3 - 3 * z**2 * ir1_5) - MU * (ir2_3 - 3 * z**2 * ir2_5)
    u_xy = 3 * xm * dx1 * y_ * ir1_5 + 3 * MU * dx2 * y_ * ir2_5
    u_xz = 3 * xm * dx1 * z * ir1_5 + 3 * MU * dx2 * z * ir2_5
    u_yz = 3 * xm * y_ * z * ir1_5 + 3 * MU * y_ * z * ir2_5
    a = np.zeros((6, 6))
    a[:3, 3:] = np.eye(3)
    a[3:, :3] = [[u_xx, u_xy, u_xz], [u_xy, u_yy, u_yz], [u_xz, u_yz, u_zz]]
    a[3, 4] = 2.0
    a[4, 3] = -2.0
    return eom, a


def _cr3bp_ref(with_stm: bool):
    """CR3BP 参照右端（numpy），with_stm 时输出 42 维。"""

    def f(t: float, y: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        eom, a = _cr3bp_parts(t, y)
        if not with_stm:
            return eom
        stm = y[6:].reshape(6, 6)
        return np.concatenate([eom, (a @ stm).ravel()])

    return f


def _bcr4bp_parts(t, y):
    """BCR4BP 右端与雅可比 A(t)（numpy 参照，bcr4bp_dynamics.py 的移植）。"""
    eom, a = _cr3bp_parts(t, y)
    theta = SUN_PHASE0 + SUN_ANGULAR_RATE * t
    r_s = np.array([SUN_DISTANCE * np.cos(theta), SUN_DISTANCE * np.sin(theta), 0.0])
    d = y[:3] - r_s
    d_norm = max(float(np.linalg.norm(d)), 1e-10)
    eom[3:] += -MU_SUN * (d / d_norm**3 + r_s / np.linalg.norm(r_s) ** 3)
    a[3:, :3] += -MU_SUN / d_norm**3 * (np.eye(3) - 3.0 * np.outer(d, d) / d_norm**2)
    return eom, a


def _bcr4bp_ref(with_stm: bool):
    """BCR4BP 参照右端（numpy），with_stm 时输出 42 维。"""

    def f(t: float, y: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        eom, a = _bcr4bp_parts(t, y)
        if not with_stm:
            return eom
        stm = y[6:].reshape(6, 6)
        return np.concatenate([eom, (a @ stm).ravel()])

    return f


def _crossing_y(t: float, y: npt.NDArray[np.floating]) -> float:
    """y = 0 截面（42 维增广态下仍取前 6 维的 y 分量）。"""
    return y[1]


def _run(f, events, y0, state_error_dim=None):
    return solve_ivp_events(
        T_SPAN,
        y0,
        T_EVAL,
        RTOL,
        ATOL,
        f,
        events,
        method=RkMethod.PD78,
        max_step=MAX_STEP,
        state_error_dim=state_error_dim,
    )


def test_cr3bp_kernel_matches_python_callback_route():
    """cr3bp 内核与 Python 回调参照：states 与事件时刻在容差内一致。"""
    kernel = RustEomKernel("cr3bp", {"mu": MU})
    ref = _run(_cr3bp_ref(False), [(_crossing_y, False, -1.0)], Y0_6D)
    got = _run(kernel, [(_crossing_y, False, -1.0)], Y0_6D)

    np.testing.assert_allclose(got["states"], ref["states"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(got["t_events"][0], ref["t_events"][0], rtol=0, atol=1e-8)
    assert len(got["t_events"][0]) > 0


def test_cr3bp_with_stm_kernel_matches_python_callback_route():
    """cr3bp-with-stm 内核：states/STM/事件态均与参照一致。"""
    kernel = RustEomKernel("cr3bp-with-stm", {"mu": MU})
    y0 = Y0_6D + np.eye(6).ravel().tolist()
    ref = _run(_cr3bp_ref(True), [(_crossing_y, False, -1.0)], y0, state_error_dim=6)
    got = _run(kernel, [(_crossing_y, False, -1.0)], y0, state_error_dim=6)

    got_states = np.asarray(got["states"])
    ref_states = np.asarray(ref["states"])
    got_stm = got_states[:, 6:].reshape(-1, 6, 6)
    ref_stm = ref_states[:, 6:].reshape(-1, 6, 6)
    np.testing.assert_allclose(got_states[:, :6], ref_states[:, :6], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(got_stm, ref_stm, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(got["t_events"][0], ref["t_events"][0], rtol=0, atol=1e-8)


def test_bcr4bp_kernel_matches_python_callback_route():
    """bcr4bp 内核（显式含时）与 Python 回调参照一致。"""
    kernel = RustEomKernel(
        "bcr4bp",
        {
            "mu": MU,
            "mu_sun": MU_SUN,
            "sun_distance": SUN_DISTANCE,
            "sun_angular_rate": SUN_ANGULAR_RATE,
            "sun_phase0": SUN_PHASE0,
        },
    )
    ref = _run(_bcr4bp_ref(False), [(_crossing_y, False, -1.0)], Y0_6D)
    got = _run(kernel, [(_crossing_y, False, -1.0)], Y0_6D)

    np.testing.assert_allclose(got["states"], ref["states"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(got["t_events"][0], ref["t_events"][0], rtol=0, atol=1e-8)
    assert len(got["t_events"][0]) > 0


def test_bcr4bp_with_stm_kernel_matches_python_callback_route():
    """bcr4bp-with-stm 内核：A(t) 显式含时，states/STM 与参照一致。"""
    kernel = RustEomKernel(
        "bcr4bp-with-stm",
        {
            "mu": MU,
            "mu_sun": MU_SUN,
            "sun_distance": SUN_DISTANCE,
            "sun_angular_rate": SUN_ANGULAR_RATE,
            "sun_phase0": SUN_PHASE0,
        },
    )
    y0 = Y0_6D + np.eye(6).ravel().tolist()
    ref = _run(_bcr4bp_ref(True), [(_crossing_y, False, -1.0)], y0, state_error_dim=6)
    got = _run(kernel, [(_crossing_y, False, -1.0)], y0, state_error_dim=6)

    got_states = np.asarray(got["states"])
    ref_states = np.asarray(ref["states"])
    got_stm = got_states[:, 6:].reshape(-1, 6, 6)
    ref_stm = ref_states[:, 6:].reshape(-1, 6, 6)
    np.testing.assert_allclose(got_states[:, :6], ref_states[:, :6], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(got_stm, ref_stm, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(got["t_events"][0], ref["t_events"][0], rtol=0, atol=1e-8)


def test_kernel_terminal_event_contract():
    """内核路径的 terminal 事件契约：截断末点 = 求精事件点，事件索引回传。"""

    def crossing_x(t, y):
        return y[0] - 0.75

    kernel = RustEomKernel("cr3bp", {"mu": MU})
    got = _run(
        kernel,
        [(crossing_x, False, 0.0), (_crossing_y, True, -1.0)],
        Y0_6D,
    )
    assert got["terminal_event"] == 1
    assert got["time"][-1] == got["t_events"][1][-1]
    assert got["time"][-1] < T_SPAN[1]
    assert len(got["t_events"][0]) > 0
    assert abs(got["y_events"][1][0][1]) < 1e-6


def test_kernel_with_empty_events_behaves_like_plain_integration():
    """events=[] 时内核路径等价普通积分。"""
    kernel = RustEomKernel("cr3bp", {"mu": MU})
    got = _run(kernel, [], Y0_6D)
    assert got["terminal_event"] is None
    assert len(got["t_events"]) == 0
    assert got["time"][-1] == pytest.approx(T_SPAN[1])
    ref = _run(_cr3bp_ref(False), [], Y0_6D)
    np.testing.assert_allclose(got["states"], ref["states"], rtol=1e-9, atol=1e-12)


def test_unknown_kernel_is_rejected_at_construction():
    """未知内核标识在 RustEomKernel 构造时即报错（Rust 侧兜底见 Rust 单测）。"""
    with pytest.raises(ValueError, match="kernel"):
        RustEomKernel("ephemeris", {"mu": MU})


def test_kernel_missing_params_are_rejected():
    with pytest.raises(ValueError, match="mu"):
        _run(RustEomKernel("cr3bp", {}), [], Y0_6D)
    with pytest.raises(ValueError, match="sun_distance"):
        _run(RustEomKernel("bcr4bp", {"mu": MU, "mu_sun": MU_SUN}), [], Y0_6D)


def test_kernel_state_length_mismatch_is_rejected():
    """with-stm 内核要求 42 维增广初值，6 维初值必须报错。"""
    kernel = RustEomKernel("cr3bp-with-stm", {"mu": MU})
    with pytest.raises(ValueError, match="42"):
        _run(kernel, [], Y0_6D)


def test_rust_eom_kernel_validates_identifier_and_coerces_params():
    with pytest.raises(ValueError, match="kernel"):
        RustEomKernel("nope", {"mu": MU})
    kernel = RustEomKernel("cr3bp", {"mu": "0.012"})  # str → float 强制转换
    assert kernel.params == {"mu": 0.012}
