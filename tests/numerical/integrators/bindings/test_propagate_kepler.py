"""``propagate_kepler_py`` 绑定行为测试（issue #739）。

UV 二体 Kepler 封闭解内核与 ``propagate_two_body`` 的 DOP853 数值积分是
两条独立实现路径，互为交叉验证（非 oracle 复制）；解析 STM 的正确性由
辛性（det Φ = 1，机器精度）与中心差分对照共同钉住。
"""

import numpy as np
import pytest

pytest.importorskip("e2m2e._integrators")

from e2m2e.algorithm.transfer.multi_impulse import propagate_two_body
from e2m2e.integrators import propagate_kepler_py

pytestmark = pytest.mark.integrator


if propagate_kepler_py is None:
    pytest.skip("propagate_kepler_py 需要 make dev 构建", allow_module_level=True)

# 地球 GM（km³/s²），固定值便于复算。
MU = 398600.4418

# 椭圆（|v0| ≈ 7.57 < v_esc ≈ 10.67，e ≈ 0.0075）与双曲（v0 > v_esc）两组初态。
ELLIPTIC = [7000.0, 0.0, 0.0, 0.0, 7.5, 1.0]
HYPERBOLIC = [7000.0, 0.0, 0.0, 0.0, 11.0, 0.0]


def test_zero_dt_identity():
    """dt = 0 直接返回初值，STM 为单位阵。"""
    out = propagate_kepler_py(ELLIPTIC, [0.0], MU, with_stm=True)
    assert list(out["states"][0]) == ELLIPTIC
    np.testing.assert_allclose(np.asarray(out["stm"][0]).reshape(6, 6), np.eye(6), atol=1e-15)


def test_against_dop853():
    """椭圆与双曲，与 DOP853 数值积分对照（含负 dt 后向分支）。

    闭式解为定义性公式（Vallado 2013 §2-5），DOP853 参照的积分容差为
    multi_impulse 内部常量（rtol 1e-11 / atol 1e-12），两路径独立。
    注意参照实现 ``propagate_two_body`` 的语义是 state0 位于 ``t_eval[0]``
    并沿单调时间轴向前积分；本内核语义是 state0 位于历元、t_eval 为
    逐点流逝秒（可负）。正向对照用非负轴；后向分支从 −3600 s 历元状态
    续推参照积分，独立核对负 dt 的正确性。
    """
    for state0 in (ELLIPTIC, HYPERBOLIC):
        # 正向：state0 @ 0，对照 1800/3600/5400/7200 s。
        t_fwd = [0.0, 1800.0, 3600.0, 5400.0, 7200.0]
        out = propagate_kepler_py(state0, t_fwd, MU)
        ref = propagate_two_body(state0, t_fwd, MU)
        states = np.asarray(out["states"])
        np.testing.assert_allclose(
            states[:, :3],
            ref["states"][:, :3],
            atol=1e-6,
            err_msg=f"位置与 DOP853 不一致: {state0}",
        )
        np.testing.assert_allclose(
            states[:, 3:],
            ref["states"][:, 3:],
            atol=1e-9,
            err_msg=f"速度与 DOP853 不一致: {state0}",
        )
        np.testing.assert_allclose(out["time"], ref["time"])

        # 后向：内核在 −3600/−1800/0 的状态，对照从 −3600 状态续推的
        # DOP853（参照轴 0/1800/3600 对应内核轴 −3600/−1800/0）。
        back0 = propagate_kepler_py(state0, [-3600.0], MU)["states"][0]
        out_neg = np.asarray(propagate_kepler_py(state0, [-3600.0, -1800.0, 0.0], MU)["states"])
        ref_neg = propagate_two_body(back0, [0.0, 1800.0, 3600.0], MU)["states"]
        np.testing.assert_allclose(
            out_neg[1:, :3],
            ref_neg[1:, :3],
            atol=1e-6,
            err_msg=f"后向位置与 DOP853 不一致: {state0}",
        )
        np.testing.assert_allclose(
            out_neg[1:, 3:],
            ref_neg[1:, 3:],
            atol=1e-9,
            err_msg=f"后向速度与 DOP853 不一致: {state0}",
        )


def test_circular_orbit_closure():
    """圆轨道一个周期后位置回到初值（封闭解定义性验收）。"""
    r = 8000.0
    v = float(np.sqrt(MU / r))
    state0 = [r, 0.0, 0.0, 0.0, v, 0.0]
    period = 2.0 * np.pi * np.sqrt(r**3 / MU)
    out = propagate_kepler_py(state0, [period], MU)
    np.testing.assert_allclose(out["states"][0][:3], state0[:3], atol=1e-6)


def test_stm_symplectic():
    """解析 STM 辛性：椭圆与双曲、正负多个 Δt，det Φ = 1（机器精度）。"""
    cases = [
        (ELLIPTIC, [600.0, -1800.0, 3600.0]),
        (HYPERBOLIC, [600.0, -1800.0, 1800.0]),
    ]
    for state0, t_eval in cases:
        out = propagate_kepler_py(state0, t_eval, MU, with_stm=True)
        for row in out["stm"]:
            det = float(np.linalg.det(np.asarray(row).reshape(6, 6)))
            assert abs(det - 1.0) < 1e-12, f"det Φ − 1 = {det - 1.0}"


def test_stm_vs_finite_difference():
    """解析 STM 各列与初始状态中心差分对照（椭圆弧单 Δt）。

    差分步长按量纲取：位置 1e-3 km、速度 1e-6 km/s；Δt 取短弧压低
    中心差分三阶截断误差，使差分自身精度高于 1e-5 断言门限。
    """
    state0 = np.asarray(ELLIPTIC, dtype=float)
    dt = 300.0
    out = propagate_kepler_py(state0.tolist(), [dt], MU, with_stm=True)
    stm = np.asarray(out["stm"][0]).reshape(6, 6)
    steps = np.array([1e-3] * 3 + [1e-6] * 3)
    for j in range(6):
        y_plus = state0.copy()
        y_minus = state0.copy()
        y_plus[j] += steps[j]
        y_minus[j] -= steps[j]
        s_plus = np.asarray(propagate_kepler_py(y_plus.tolist(), [dt], MU)["states"][0])
        s_minus = np.asarray(propagate_kepler_py(y_minus.tolist(), [dt], MU)["states"][0])
        fd = (s_plus - s_minus) / (2.0 * steps[j])
        np.testing.assert_allclose(fd, stm[:, j], atol=1e-5, err_msg=f"STM 第 {j} 列与中心差分不符")


def test_energy_angular_momentum_conservation():
    """椭圆弧多点：比能量与角动量矢量守恒（相对偏差 < 1e-12）。"""
    state0 = np.asarray(ELLIPTIC, dtype=float)
    eps0 = float(state0[3:] @ state0[3:] / 2.0 - MU / np.linalg.norm(state0[:3]))
    h0 = np.cross(state0[:3], state0[3:])
    t_eval = [300.0, 1000.0, 2000.0, 3500.0, 5000.0]
    out = propagate_kepler_py(state0.tolist(), t_eval, MU)
    for row in out["states"]:
        s = np.asarray(row, dtype=float)
        r = float(np.linalg.norm(s[:3]))
        eps = float(s[3:] @ s[3:] / 2.0 - MU / r)
        assert abs((eps - eps0) / eps0) < 1e-12
        h = np.cross(s[:3], s[3:])
        np.testing.assert_allclose(h, h0, rtol=0.0, atol=1e-12 * float(np.linalg.norm(h0)))


def test_invalid_inputs():
    """参数校验：空 t_eval、state0 维数、非正 mu、零 r0 各抛 ValueError。"""
    with pytest.raises(ValueError, match="t_eval 不能为空"):
        propagate_kepler_py(ELLIPTIC, [], MU)
    with pytest.raises(ValueError, match="state0 必须为 6 维"):
        propagate_kepler_py([7000.0, 0.0, 0.0, 0.0, 7.5], [60.0], MU)
    with pytest.raises(ValueError, match="mu 必须为正的有限值"):
        propagate_kepler_py(ELLIPTIC, [60.0], 0.0)
    with pytest.raises(ValueError, match="r0 不能为零向量"):
        propagate_kepler_py([0.0] * 6, [60.0], MU)


def test_no_stm_key_by_default():
    """with_stm=False 返回 dict 不含 stm 键；True 时含 36 维行主序 STM。"""
    out = propagate_kepler_py(ELLIPTIC, [60.0], MU)
    assert "stm" not in out
    assert set(out) == {"time", "states"}
    out_stm = propagate_kepler_py(ELLIPTIC, [60.0], MU, with_stm=True)
    assert "stm" in out_stm
    assert len(out_stm["stm"][0]) == 36
