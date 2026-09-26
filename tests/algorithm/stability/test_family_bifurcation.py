"""族级分岔扫描测试（#688）。

合成族经 ``multiplier_fn`` 注入脚本乘子，覆盖三类穿越（saddle-node /
period-doubling / torus）、平凡乘子对排除、跳支区间、成员分析失败、
粗扫与精化两种模式、参数校验与 ``find_nearest_bifurcation``；末尾一条
最小真实调用（ADR 0037）守护真实单值矩阵链路上的"平凡对不误报"。

脚本族的物理口径：垂直临界穿越 vt = 1 对应面外乘子对 ν = 2·vt 穿过 +2
（``_quadratic_pair`` 给出 |vt|<1 的单位圆共轭对与 |vt|>1 的实数倒数对，
与真实族的连续过渡一致）。
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from kernel_helpers import requires_native_symbols

from e2m2e.algorithm.dynamics import CR3BP_Dynamics
from e2m2e.algorithm.family.cr3bp_orbits import _correct_dro
from e2m2e.algorithm.stability import (
    BifurcationType,
    FamilyBifurcationPoint,
    FamilyBifurcationScan,
    StabilityAnalysis,
)
from e2m2e.data.templates.seed import _DRO_SEED_X0
from e2m2e.data.types.orbit import Orbit

pytestmark = pytest.mark.theory

#: 脚本族参数网格（默认：穿越区间落在 [0.4, 0.6]）
_GRID = (0.2, 0.4, 0.6, 0.8)


def _fake_orbit(parameter: float, system) -> Orbit:
    """把族参数编进 ``states[0][0]`` 的合成族成员（period 固定 1.0）。"""
    orbit = Orbit(
        states=np.array([[parameter, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        times=np.array([0.0]),
        system=system,
    )
    orbit.period = 1.0
    return orbit


def _read_parameter(orbit: Orbit) -> float:
    return float(orbit.states[0][0])


def _trivial_pair() -> tuple[complex, complex]:
    """自治 Hamilton 流的时间平移/能量平凡乘子对（λ≈1，ν≈2）。"""
    return (1.0 + 1e-10, 1.0 - 1e-10)


def _quadratic_pair(vt: float) -> tuple[complex, complex]:
    """μ² − 2·vt·μ + 1 = 0 的根对（ν = 2·vt）。

    |vt| < 1 时为单位圆上的共轭对（ν ∈ (−2, 2)），|vt| > 1 时为实数倒数对
    （ν > 2）——垂直临界 vt = +1 两侧的乘子形态。
    """
    root = np.sqrt(complex(vt * vt - 1.0))
    return (complex(vt) + root, complex(vt) - root)


def _on_circle_pair(nu: float) -> tuple[complex, complex]:
    """单位圆共轭对，稳定性指数恰为 ``nu``（ν = 2cosθ）。"""
    theta = np.arccos(nu / 2.0)
    return (complex(np.exp(1j * theta)), complex(np.exp(-1j * theta)))


def _multipliers(
    tested: tuple[complex, complex], tail: tuple[complex, complex] | None = None
) -> np.ndarray:
    """6 乘子：平凡对 + 被测对 + 第三对（缺省为固定单位圆对 e^{±1.1i}）。"""
    trivial = _trivial_pair()
    third = _on_circle_pair(2.0 * np.cos(1.1)) if tail is None else tail
    return np.array(
        [trivial[0], trivial[1], tested[0], tested[1], third[0], third[1]], dtype=complex
    )


def _scripted_multiplier_fn(mode: str):
    """按族参数脚本化乘子的 ``multiplier_fn``。"""

    def multiplier_fn(orbit: Orbit) -> np.ndarray:
        parameter = _read_parameter(orbit)
        if mode == "saddle_node":
            tested = _quadratic_pair(1.0 + 0.5 * (parameter - 0.5))
        elif mode == "period_doubling":
            tested = _quadratic_pair(-1.0 - 0.5 * (parameter - 0.5))
        elif mode == "torus":
            theta = 0.7
            radius = 1.0 + 0.2 * (parameter - 0.5)
            tested = (complex(radius * np.exp(1j * theta)), complex(np.exp(-1j * theta) / radius))
        elif mode == "drifting_in_plane":
            # 面内强不稳定模态（ν ≈ 1e3、随族参数平滑漂移 Δν ≈ 8/区间）＋面外穿越；
            # 口径来自 L2 Lyapunov 族实测（面内 ν ≈ 1448 → 789）
            tested = _quadratic_pair(1.0 + 0.5 * (parameter - 0.5))
            return _multipliers(tested, tail=_quadratic_pair(0.5 * (1000.0 + 40.0 * parameter)))
        elif mode == "trivial":
            tested = _trivial_pair()
        elif mode == "jump":
            # [0.4, 0.6] 区间内 ν 从 0 跳到 0.5（|Δν| = 0.5 > 0.3）
            tested = _on_circle_pair(0.0 if parameter < 0.5 else 0.5)
        else:  # pragma: no cover - 仅防护脚本名拼错
            raise AssertionError(f"未知脚本模式 {mode}")
        return _multipliers(tested)

    return multiplier_fn


def _failing_multiplier_fn(mode: str, fail_at: float):
    """在参数 ``fail_at`` 处抛错的乘子脚本（模拟成员分析失败）。"""
    base = _scripted_multiplier_fn(mode)

    def multiplier_fn(orbit: Orbit) -> np.ndarray:
        if _read_parameter(orbit) == fail_at:
            raise RuntimeError("注入的乘子分析失败")
        return base(orbit)

    return multiplier_fn


def _pair_for_nu(nu: float) -> tuple[complex, complex]:
    """按稳定性指数构造一个倒数对：|ν| > 2 用实倒数对（vt = ν/2），否则用圆上共轭对。"""
    if abs(nu) > 2.0:
        return _quadratic_pair(nu / 2.0)
    return _on_circle_pair(nu)


def _triplet_multiplier_fn(values: dict[float, tuple[float, float, float]]):
    """按族参数脚本化三条 ν（ν≈2 用平凡对）的 ``multiplier_fn``。

    显式给出三条 ν 才能构造「缺平凡对」的成员：三条 ν 都不落在 2±1e-6 内时
    ``_drop_trivial_pair`` 不剔除任何一对，该成员留给跟踪 3 条 ν。
    """

    def multiplier_fn(orbit: Orbit) -> np.ndarray:
        pairs = [
            (1.0 + 1e-10, 1.0 - 1e-10) if abs(nu - 2.0) < 1e-12 else _pair_for_nu(nu)
            for nu in values[_read_parameter(orbit)]
        ]
        return np.array([value for pair in pairs for value in pair], dtype=complex)

    return multiplier_fn


def _scan(
    mode: str,
    system,
    *,
    grid: tuple[float, ...] = _GRID,
    refine: bool = True,
    multiplier_fn=None,
    parameter_name: str = "x0",
) -> FamilyBifurcationScan:
    orbits = [_fake_orbit(parameter, system) for parameter in grid]
    return StabilityAnalysis.detect_bifurcation_in_family(
        orbits,
        grid,
        parameter_name=parameter_name,
        member_at=(lambda parameter: _fake_orbit(parameter, system)) if refine else None,
        multiplier_fn=multiplier_fn if multiplier_fn is not None else _scripted_multiplier_fn(mode),
    )


class TestCrossingTypes:
    """三类穿越判据与精化。"""

    def test_saddle_node_crossing_is_located(self, earth_moon_system):
        scan = _scan("saddle_node", earth_moon_system)

        assert len(scan.points) == 1
        point = scan.points[0]
        assert isinstance(point, FamilyBifurcationPoint)
        assert point.type is BifurcationType.SADDLE_NODE
        assert point.parameter_name == "x0"
        assert point.parameter == pytest.approx(0.5, abs=1e-5)
        assert point.residual <= 1e-6
        assert point.multipliers.shape == (6,)
        assert _read_parameter(point.orbit) == pytest.approx(point.parameter, abs=1e-5)
        assert scan.branch_jumps == ()
        assert scan.failures == ()

    def test_period_doubling_crossing_is_located(self, earth_moon_system):
        scan = _scan("period_doubling", earth_moon_system)

        assert len(scan.points) == 1
        point = scan.points[0]
        assert point.type is BifurcationType.PERIOD_DOUBLING
        assert point.parameter == pytest.approx(0.5, abs=1e-5)
        assert point.residual <= 1e-6

    def test_torus_onset_is_located(self, earth_moon_system):
        # p=0.5 处复四元组恰在单位圆上（|Im ν| = 0），越过即 onset
        scan = _scan(
            "torus",
            earth_moon_system,
            grid=(0.2, 0.5, 0.65, 0.8),
            parameter_name="z0",
        )

        assert len(scan.points) == 1
        point = scan.points[0]
        assert point.type is BifurcationType.TORUS
        assert point.parameter_name == "z0"
        assert point.parameter == pytest.approx(0.5, abs=1e-4)
        assert point.residual <= 1e-6


class TestTrivialAndJumpHandling:
    """平凡对排除与跳支区间。"""

    def test_trivial_pair_only_family_reports_nothing(self, earth_moon_system):
        scan = _scan("trivial", earth_moon_system)

        assert scan.points == ()
        assert scan.branch_jumps == ()
        assert scan.failures == ()

    def test_drifting_large_index_track_does_not_mask_crossing(self, earth_moon_system):
        """量级 1e3 的平滑漂移轨迹不得被判为跳支而屏蔽同区间内的真实穿越。"""
        scan = _scan("drifting_in_plane", earth_moon_system)

        assert scan.branch_jumps == ()
        assert len(scan.points) == 1
        assert scan.points[0].type is BifurcationType.SADDLE_NODE
        assert scan.points[0].parameter == pytest.approx(0.5, abs=1e-5)

    def test_branch_jump_is_reported_instead_of_crossing(self, earth_moon_system):
        scan = _scan("jump", earth_moon_system, refine=False)

        assert scan.points == ()
        assert len(scan.branch_jumps) == 1
        jump = scan.branch_jumps[0]
        assert jump.parameter_lo == pytest.approx(0.4)
        assert jump.parameter_hi == pytest.approx(0.6)
        assert jump.max_displacement == pytest.approx(0.5, abs=1e-6)


class TestMemberFailures:
    """成员分析失败：该成员两侧区间不参与判据。"""

    def test_failing_member_is_recorded_and_other_intervals_survive(self, earth_moon_system):
        grid = (0.2, 0.4, 0.6, 0.8, 1.0)
        scan = _scan(
            "saddle_node",
            earth_moon_system,
            grid=grid,
            multiplier_fn=_failing_multiplier_fn("saddle_node", 0.8),
        )

        assert [failure.parameter for failure in scan.failures] == [0.8]
        assert "注入的乘子分析失败" in scan.failures[0].message
        assert len(scan.points) == 1
        assert scan.points[0].type is BifurcationType.SADDLE_NODE
        assert scan.points[0].parameter == pytest.approx(0.5, abs=1e-5)


class TestDegenerateMembers:
    """退化成员：判据恰为零的节点、缺平凡对的成员、多穿越排序、精化失败原因。"""

    def test_crossing_at_grid_node_is_located_once(self, earth_moon_system):
        """判据恰为零的网格节点（ν = 2）须被认领一次：既不漏报也不重复报。"""
        scan = _scan("saddle_node", earth_moon_system, grid=(0.2, 0.4, 0.5, 0.6, 0.8))

        assert len(scan.points) == 1
        assert scan.points[0].type is BifurcationType.SADDLE_NODE
        assert scan.points[0].parameter == pytest.approx(0.5, abs=1e-3)
        assert scan.failures == ()

    def test_member_without_trivial_pair_keeps_one_point_per_crossing(self, earth_moon_system):
        """成员无平凡对（逐成员剔除后剩 3 条 ν）不得使同一穿越重复报点或丢点。"""
        grid = (0.0, 0.5, 1.0)
        # 轨迹 A：1.90 → 2.10 跨 +2；mid 成员三条 ν 都不在 2±1e-6 内 → 不剔除任何一对
        flipping = {
            0.0: (2.0, 1.90, 1.50),
            0.5: (2.10, 1.55, 1.60),
            1.0: (2.20, 1.60, 1.65),
        }
        control = {
            0.0: (2.0, 1.90, 1.50),
            0.5: (2.0, 2.10, 1.55),
            1.0: (2.20, 1.60, 1.65),
        }

        for name, values in (("含缺平凡对成员", flipping), ("对照", control)):
            scan = _scan(
                "saddle_node",
                earth_moon_system,
                grid=grid,
                refine=False,
                multiplier_fn=_triplet_multiplier_fn(values),
            )
            parameters = [point.parameter for point in scan.points]

            assert len(scan.points) == 1, name
            assert scan.points[0].type is BifurcationType.SADDLE_NODE, name
            assert scan.branch_jumps == (), name
            assert len(set(parameters)) == len(parameters), name

    def test_multiple_crossings_are_strictly_ascending(self, earth_moon_system):
        """同一族报两个穿越（+2 与 −2）：族参数严格升序，每个穿越各一条。"""
        grid = (0.2, 0.4, 0.6, 0.8)
        values = {
            0.2: (2.0, 1.90, -1.90),
            0.4: (2.0, 2.10, -1.95),  # 轨迹 A 跨 +2 → saddle_node
            0.6: (2.0, 2.20, -2.10),  # 轨迹 C 跨 −2 → period_doubling
            0.8: (2.0, 2.30, -2.20),
        }
        scan = _scan(
            "saddle_node",
            earth_moon_system,
            grid=grid,
            refine=False,
            multiplier_fn=_triplet_multiplier_fn(values),
        )

        assert [point.type for point in scan.points] == [
            BifurcationType.SADDLE_NODE,
            BifurcationType.PERIOD_DOUBLING,
        ]
        parameters = [point.parameter for point in scan.points]
        assert parameters == sorted(parameters)
        assert all(later > earlier for earlier, later in pairwise(parameters))

    def test_refinement_failure_keeps_underlying_cause(self, earth_moon_system):
        """精化采样全失败时失败记录须带底层原因（与成员循环同口径）。"""

        def member_at(parameter: float) -> Orbit:
            raise RuntimeError("注入的精化修正失败")

        orbits = [_fake_orbit(parameter, earth_moon_system) for parameter in _GRID]
        scan = StabilityAnalysis.detect_bifurcation_in_family(
            orbits,
            _GRID,
            parameter_name="x0",
            member_at=member_at,
            multiplier_fn=_scripted_multiplier_fn("saddle_node"),
        )

        assert len(scan.failures) == 1
        assert "注入的精化修正失败" in scan.failures[0].message
        assert scan.points == ()


class TestCoarseScanMode:
    """粗扫模式（member_at=None）：点仍报出，参数为过零插值。"""

    def test_coarse_scan_interpolates_crossing(self, earth_moon_system):
        scan = _scan("saddle_node", earth_moon_system, refine=False)

        assert len(scan.points) == 1
        point = scan.points[0]
        assert point.type is BifurcationType.SADDLE_NODE
        assert point.parameter == pytest.approx(0.5, abs=1e-12)
        assert point.residual == pytest.approx(0.1, rel=1e-6)
        assert point.multipliers.shape == (6,)
        assert _read_parameter(point.orbit) in (0.4, 0.6)


class TestFindNearestBifurcation:
    """按族参数取最近分岔点。"""

    def test_empty_scan_returns_none(self):
        scan = FamilyBifurcationScan(points=(), branch_jumps=(), failures=())

        assert StabilityAnalysis.find_nearest_bifurcation(scan, target=0.5) is None

    def test_picks_nearest_by_parameter(self, earth_moon_system):
        first = FamilyBifurcationPoint(
            parameter=1.0,
            parameter_name="x0",
            type=BifurcationType.SADDLE_NODE,
            orbit=_fake_orbit(1.0, earth_moon_system),
            multipliers=_multipliers(_trivial_pair()),
            residual=0.0,
        )
        second = FamilyBifurcationPoint(
            parameter=5.0,
            parameter_name="x0",
            type=BifurcationType.PERIOD_DOUBLING,
            orbit=_fake_orbit(5.0, earth_moon_system),
            multipliers=_multipliers(_trivial_pair()),
            residual=0.0,
        )
        scan = FamilyBifurcationScan(points=(first, second), branch_jumps=(), failures=())

        assert StabilityAnalysis.find_nearest_bifurcation(scan, target=4.2) is second
        assert StabilityAnalysis.find_nearest_bifurcation(scan, target=0.1) is first


class TestValidation:
    """输入校验。"""

    def test_length_mismatch_is_rejected(self, earth_moon_system):
        orbits = [_fake_orbit(0.2, earth_moon_system), _fake_orbit(0.4, earth_moon_system)]

        with pytest.raises(ValueError, match="长度必须一致"):
            StabilityAnalysis.detect_bifurcation_in_family(orbits, (0.2, 0.4, 0.6))

    def test_non_increasing_parameters_are_rejected(self, earth_moon_system):
        grid = (0.4, 0.2)
        orbits = [_fake_orbit(parameter, earth_moon_system) for parameter in grid]

        with pytest.raises(ValueError, match="严格递增"):
            StabilityAnalysis.detect_bifurcation_in_family(orbits, grid)

    def test_single_member_is_rejected(self, earth_moon_system):
        orbits = [_fake_orbit(0.2, earth_moon_system)]

        with pytest.raises(ValueError, match="至少需要 2 个族成员"):
            StabilityAnalysis.detect_bifurcation_in_family(orbits, (0.2,))


class TestRealDroFamily:
    """最小真实调用（ADR 0037）：真实 Rust 单值矩阵链路上平凡对不误报。"""

    @requires_native_symbols("propagate_cr3bp_stm_py", "differential_correction_cr3bp_py")
    def test_planar_dro_family_reports_no_false_positive(self, earth_moon_system):
        dynamics = CR3BP_Dynamics(earth_moon_system)
        orbit = _correct_dro(dynamics, _DRO_SEED_X0, None)
        grid = (_DRO_SEED_X0, _DRO_SEED_X0 + 1e-4, _DRO_SEED_X0 + 2e-4)
        # 同一轨道复制成 3 成员假族：ν 恒定，任何穿越报告都是误报
        scan = StabilityAnalysis.detect_bifurcation_in_family(
            [orbit, orbit, orbit], grid, parameter_name="x0"
        )

        assert scan.points == ()
        assert scan.branch_jumps == ()
        assert scan.failures == ()
