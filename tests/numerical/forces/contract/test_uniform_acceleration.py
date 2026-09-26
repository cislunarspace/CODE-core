"""UniformAcceleration（RTN 常值加速度）编译传播契约。

覆盖：RTN 三分量到惯性轴的映射、沿迹轴垂直径向、径向分量无长期漂移、退化状态
（|r|≈0、|v|≈0、r∥v）显式报错与全零豁免、常值横向加速度下的半长轴闭式漂移、
法向加速度的面外运动闭式解、STM 与初态中心差分一致性、构造校验。
"""

import numpy as np
import pytest

from e2m2e.algorithm.forces import ForceModel, PointMassGravity, UniformAcceleration
from tests.numerical.forces.conftest import EARTH_MU, FakeSystem, semi_major_axis

pytestmark = pytest.mark.force

# 赤道圆轨道基准态：R̂=x̂、T̂=ŷ、N̂=ẑ。
_R0 = 7000.0
_CIRCULAR_Y0 = np.array([_R0, 0.0, 0.0, 0.0, np.sqrt(EARTH_MU / _R0), 0.0])


def _propagate(accel_rtn, state, t_span, *, forces_extra=(), with_stm=False):
    """以 ``UniformAcceleration``（可叠加 ``forces_extra``）传播，返回结果字典。"""
    fm = ForceModel(FakeSystem(), [*forces_extra, UniformAcceleration(accel_rtn)])
    return fm.propagate(
        np.asarray(state, dtype=float), t_span, t_eval=np.array(t_span), with_stm=with_stm
    )


def _delta_v(accel_rtn, state, t_span, *, forces_extra=()):
    """传播后的速度增量 Δv = v_end − v0。"""
    y0 = np.asarray(state, dtype=float)
    result = _propagate(accel_rtn, y0, t_span, forces_extra=forces_extra)
    return result["states"][-1, 3:6] - y0[3:6]


@pytest.mark.parametrize(
    ("accel_rtn", "expected"),
    [
        ([1e-6, 0.0, 0.0], [1e-6, 0.0, 0.0]),
        ([0.0, 1e-6, 0.0], [0.0, 1e-6, 0.0]),
        ([0.0, 0.0, 1e-6], [0.0, 0.0, 1e-6]),
        ([1e-6, 2e-6, 3e-6], [1e-6, 2e-6, 3e-6]),
    ],
)
def test_rtn_component_direction_mapping(accel_rtn, expected):
    """RTN 三分量在 R̂=x̂、T̂=ŷ、N̂=ẑ 的基准态上逐一映射到惯性轴。

    容差取 2e-9：1 s 内位置漂移 v·t=7.5 km 使 RTN 基绕 ẑ 转 θ≈1.07e-3 rad，
    a_T 项因此产生一阶偏差 a_T·θ/2≈1.07e-9（组合向量最大分量），叠加上积分器
    噪声 ~1e-11；a_R（二阶 ~1e-12）与 a_N（基倾斜 ~6e-13）无此偏差。各轴符号
    由单轴用例强校验（符号翻转即偏离 2e-6）。
    """
    state = [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0]
    np.testing.assert_allclose(_delta_v(accel_rtn, state, (0.0, 1.0)), expected, atol=2e-9)


def test_rtn_along_track_perpendicular_to_radial():
    """速度含径向分量时，T 仍取 N×R（垂直于径向），不随速度方向偏斜。"""
    state = [7000.0, 0.0, 0.0, 1.0, 7.5, 0.0]
    delta_v = _delta_v([0.0, 1e-6, 0.0], state, (0.0, 1.0))
    assert abs(delta_v[0]) < 1e-8
    assert delta_v[1] > 0.99e-6


def test_zero_acceleration_on_degenerate_state():
    """三分量全零等价于无该力，退化状态（v=0）下仍正常传播且 Δv=0。"""
    delta_v = _delta_v([0.0, 0.0, 0.0], [7000.0, 0.0, 0.0, 0.0, 0.0, 0.0], (0.0, 1.0))
    np.testing.assert_allclose(delta_v, np.zeros(3), atol=1e-15)


@pytest.mark.parametrize(
    ("accel_rtn", "state", "error"),
    [
        ([1e-6, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 7.5, 0.0], "non-zero position"),
        ([1e-6, 0.0, 0.0], [7000.0, 0.0, 0.0, 0.0, 0.0, 0.0], "non-zero velocity"),
        (
            [1e-6, 2e-6, 3e-6],
            [7000.0, 0.0, 0.0, 7.5, 0.0, 0.0],
            "non-zero angular momentum",
        ),
        (
            [1e-6, 0.0, 0.0],
            [7000.0, 0.0, 0.0, 7.5, 0.0, 0.0],
            "non-zero angular momentum",
        ),
    ],
)
def test_degenerate_state_raises(accel_rtn, state, error):
    """任一非零分量在退化状态（|r|≈0、|v|≈0 或 r∥v）下明确失败、不丢分量。

    措辞与既有 VNB/LVLH 退化报错同风格；共线时 N 轴与沿迹轴均无定义，故连
    仅有径向分量的加速度也报错。
    """
    with pytest.raises(RuntimeError, match=error):
        _propagate(accel_rtn, state, (0.0, 1.0))


def test_zero_acceleration_on_collinear_state():
    """三分量全零在共线状态（|r×v|≈0）下仍等价于无该力、不报错。"""
    delta_v = _delta_v([0.0, 0.0, 0.0], [7000.0, 0.0, 0.0, 7.5, 0.0, 0.0], (0.0, 1.0))
    np.testing.assert_allclose(delta_v, np.zeros(3), atol=1e-15)


def test_radial_component_no_secular_semimajor_axis_drift():
    """径向（R）分量在一个圆轨道周期上不产生半长轴长期漂移。

    径向分量对能量的一个周期平均贡献为零，半长轴只有周期振荡而无长期项：
    实测一个周期（5828.5 s）末漂移 +6.5e-11 km，而同幅值横向分量同弧段为
    +1.08 km。阈值 1e-6 km 距实测 4 个量级以上、距横向信号 6 个量级以下。
    """
    a_r = 1e-7
    n = np.sqrt(EARTH_MU / _R0**3)
    result = _propagate(
        [a_r, 0.0, 0.0],
        _CIRCULAR_Y0,
        (0.0, 2.0 * np.pi / n),
        forces_extra=[PointMassGravity("EARTH", mu=EARTH_MU)],
    )
    drift = abs(semi_major_axis(result["states"][-1], EARTH_MU) - _R0)
    assert drift < 1e-6


def test_transverse_semimajor_axis_drift_matches_closed_form():
    """常值横向加速度下的半长轴漂移与 Gauss 行星方程闭式解一致。

    a⁻¹ᐟ²(t) = a₀⁻¹ᐟ² − a_T·t/√μ（Gauss 行星方程 da/dt = 2a^{3/2}a_T/√μ 的积分）。
    """
    a_t = 1e-7
    t_end = 600.0
    result = _propagate(
        [0.0, a_t, 0.0],
        _CIRCULAR_Y0,
        (0.0, t_end),
        forces_extra=[PointMassGravity("EARTH", mu=EARTH_MU)],
    )
    inv_sqrt_a = 1.0 / np.sqrt(_R0) - a_t * t_end / np.sqrt(EARTH_MU)
    expected = inv_sqrt_a**-2
    got = semi_major_axis(result["states"][-1], EARTH_MU)
    assert got > _R0
    np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_normal_component_out_of_plane_motion():
    """常值法向加速度产生的面外运动与解析解 z(t) = a_N(1−cos nt)/n² 一致。"""
    a_n = 1e-6
    t_end = 60.0
    n = np.sqrt(EARTH_MU / _R0**3)
    result = _propagate(
        [0.0, 0.0, a_n],
        _CIRCULAR_Y0,
        (0.0, t_end),
        forces_extra=[PointMassGravity("EARTH", mu=EARTH_MU)],
    )
    expected = a_n * (1.0 - np.cos(n * t_end)) / n**2
    got = result["states"][-1, 2]
    assert got > 0
    np.testing.assert_allclose(got, expected, rtol=1e-4)


@pytest.mark.parametrize("column", range(6))
def test_with_stm_matches_initial_state_finite_difference(column):
    """公开 STM 与独立初态中心差分逐列一致（含 ∂a/∂r 与 ∂a/∂v）。

    扰动步长取 1e-4 km：位置行元素（~1e-4）的量级下，1e-6 km 步长的两点差被
    状态量（~7e3 km）的双精度相消噪声（~7e-9 km 差 / 2e-6 ≈ 4e-3）淹没；
    1e-4 km 处截断误差 ~2e-9 且噪声 ~8e-9 都在容差内。
    """
    t_end = 60.0
    forces = [PointMassGravity("EARTH", mu=EARTH_MU)]
    fm = ForceModel(FakeSystem(), [*forces, UniformAcceleration([0.0, 1e-7, 0.0])])
    stm_result = fm.propagate(_CIRCULAR_Y0, (0.0, t_end), t_eval=[0.0, t_end], with_stm=True)

    perturbation = 1e-4
    plus = _CIRCULAR_Y0.copy()
    minus = _CIRCULAR_Y0.copy()
    plus[column] += perturbation
    minus[column] -= perturbation
    final_plus = fm.propagate(plus, (0.0, t_end), t_eval=[0.0, t_end])["states"][-1]
    final_minus = fm.propagate(minus, (0.0, t_end), t_eval=[0.0, t_end])["states"][-1]
    finite_difference = (final_plus - final_minus) / (2.0 * perturbation)

    np.testing.assert_allclose(
        stm_result["stm"][-1, :, column], finite_difference, rtol=1e-4, atol=1e-7
    )


def test_constructor_validation():
    """构造期校验：加速度形状、有限性与方向帧标签。"""
    with pytest.raises(ValueError, match="acceleration_rtn"):
        UniformAcceleration([1e-6, 2e-6])
    with pytest.raises(ValueError, match="acceleration_rtn"):
        UniformAcceleration([1e-6, np.nan, 3e-6])
    with pytest.raises(ValueError, match="direction_frame"):
        UniformAcceleration([1e-6, 0.0, 0.0], direction_frame="LVLH")
