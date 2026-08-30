"""orbit_taxonomy 分类器测试：baseline 数据集回归 + 合成判据用例。

随包 baseline（ADR 0036）是判据的回归锚点：13 个周期族全体成员必须
命中预期标签（映射约定为 unclassified 的族除外）。butterfly/dragonfly/
vertical/lyapunov 无 baseline 锚点，用**判据级合成轨迹**覆盖（合成
曲线验证穿越几何判据本身，不是动力学解——ADR 0042 验证状态表）。
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pytest

from e2m2e.algorithm.orbit_taxonomy import classify_orbit

pytestmark = pytest.mark.theory

#: 随包 baseline 目录。
BASELINE_DIR = Path(__file__).resolve().parents[3] / "e2m2e" / "data" / "catalog_baseline"

#: baseline 族 → 预期标签集合（成员 primary 必须落在集合内）。
#: lissajous 是拟周期采样（成员不闭合）；horseshoe 不在分类学内
#: （ADR 0042 映射表：入库按设计侧覆写为空标签）——两者不进本表。
#: dpo 全体成员应为顺行（#587 修复后族行走不落在逆行支）；小振幅
#: 近月成员按 ρ_max 分入 low_prograde_*，大振幅成员为 distant_prograde。
EXPECTED_FAMILY_LABELS = {
    "axial-l1": {"axial_l1"},
    "axial-l2": {"axial_l2"},
    "dpo": {
        "distant_prograde",
        "low_prograde_eastern",
        "low_prograde_western",
    },
    "dro": {"distant_retrograde"},
    "halo-l1": {"halo_l1_northern"},
    "halo-l2": {"halo_l2_northern"},
    "lpo-l4": {"longperiod_l4"},
    "nrho-l1": {"halo_l1_southern"},
    "nrho-l2": {"halo_l2_southern"},
    "spo-l4": {"shortperiod_l4"},
}


def _load_family(name: str) -> tuple[list[np.ndarray], list[float], str]:
    """读 baseline 族：返回（成员单状态列表，成员周期列表，periodicity）。"""
    meta = json.loads((BASELINE_DIR / f"baseline-{name}.json").read_text(encoding="utf-8"))
    bundle = np.load(BASELINE_DIR / f"baseline-{name}.npz")
    states = [bundle[f"cr3bp/members/{i:04d}/states"][0] for i in range(len(meta["members"]))]
    periods = [member["period"] for member in meta["members"]]
    periodicity = meta["scalars"].get("periodicity", "periodic")
    return states, periods, periodicity


@pytest.mark.parametrize(
    "family,expected",
    sorted(EXPECTED_FAMILY_LABELS.items()),
)
def test_baseline_family_members_classify(family: str, expected: set[str]):
    """baseline 每个周期族全体成员的 primary 落在预期标签集合内。"""
    states, periods, periodicity = _load_family(family)
    assert periodicity == "periodic"
    actual: set[str] = set()
    for state, period in zip(states, periods, strict=True):
        result = classify_orbit(state[None, :], period=period)
        assert result.status.value == "converged", result.message
        assert result.primary is not None, (family, result.diagnostics)
        actual.add(result.primary.canonical)
    assert actual <= expected, f"{family} 多出预期外的标签：{actual - expected}"
    assert actual, family


def test_baseline_dro_all_members_retrograde():
    """DRO 全体成员（含深近月小成员）都判 distant_retrograde。"""
    states, periods, _ = _load_family("dro")
    for state, period in zip(states, periods, strict=True):
        result = classify_orbit(state[None, :], period=period)
        assert result.primary is not None
        assert result.primary.canonical == "distant_retrograde", result.diagnostics


def test_baseline_halo_hemisphere_matches_halo_class():
    """halo/nrho 族的南北判定与记录参数 halo_class 一致（0=北/1=南）。"""
    for family, point in (("halo-l1", 1), ("halo-l2", 2), ("nrho-l1", 1), ("nrho-l2", 2)):
        meta = json.loads((BASELINE_DIR / f"baseline-{family}.json").read_text(encoding="utf-8"))
        bundle = np.load(BASELINE_DIR / f"baseline-{family}.npz")
        for i, member in enumerate(meta["members"]):
            state = bundle[f"cr3bp/members/{i:04d}/states"][0]
            result = classify_orbit(state[None, :], period=member["period"])
            assert result.primary is not None, (family, i, result.diagnostics)
            # 南北判定以轨迹几何为准（ADR 0042 约定：vy<0 穿越点 z 符号）；
            # #586 修复后几何与记录参数 halo_class 一致，两侧同时断言。
            seed_z, seed_vy = float(state[2]), float(state[4])
            geometric_expected = (
                f"halo_l{point}_northern" if seed_z * seed_vy < 0 else f"halo_l{point}_southern"
            )
            assert result.primary.canonical == geometric_expected, (
                family,
                i,
                member["parameters"]["halo_class"],
                result.diagnostics,
            )
            class_expected = (
                f"halo_l{point}_northern"
                if member["parameters"]["halo_class"] == 0
                else f"halo_l{point}_southern"
            )
            assert geometric_expected == class_expected, (family, i)


def test_baseline_lissajous_quasi_periodic():
    """lissajous 族按 periodicity 标注直接判 unclassified。"""
    states, periods, periodicity = _load_family("lissajous-l1")
    assert periodicity == "quasi-periodic"
    result = classify_orbit(states[0][None, :], period=periods[0], periodicity=periodicity)
    assert result.labels == ()
    assert result.unclassified_reason == "quasi_periodic"


def test_non_periodic_arc_unclassified():
    """不闭合的轨迹（转移弧）判 unclassified，不报错。"""
    t = np.linspace(0.0, 2.0, 200)
    # 会合系中一段不闭合的漂移弧（几何构造，非动力学解）
    states = np.column_stack(
        [
            0.7 + 0.1 * t,
            0.05 * np.sin(t),
            np.zeros_like(t),
            np.zeros_like(t),
            0.05 * np.cos(t),
            np.zeros_like(t),
        ]
    )
    result = classify_orbit(states, t)
    assert result.status.value == "converged"
    assert result.labels == ()
    assert result.unclassified_reason == "non_periodic"


def _closed_curve(components, period: float, n: int = 2880) -> tuple[np.ndarray, np.ndarray]:
    """按参数式构造闭合轨迹，供判据级合成用例。

    ``components`` 为 (x, y, z, vx, vy, vz) 六个 ``t → 值`` 的可调用
    （按状态列序），首末点（含速度）解析闭合，避开数值差分的端点泄漏。
    """
    t = np.linspace(0.0, period, n)
    states = np.zeros((n, 6))
    for column, fn in enumerate(components):
        states[:, column] = fn(t)
    return states, t


def test_synthetic_resonant_interior_and_exterior():
    """判据级：绕地环绕闭合曲线按 T/T☾ 命中共振比（p:q = 卫星:月球）。"""
    # 内共振 2:1：T = T☾/2 = π，半径 0.55 的绕地圆（不触月心/平动点分支）
    states, t = _closed_curve(
        (
            lambda t: 0.55 * np.cos(2 * t),
            lambda t: 0.55 * np.sin(2 * t),
            lambda t: np.zeros_like(t),
            lambda t: -1.1 * np.sin(2 * t),
            lambda t: 1.1 * np.cos(2 * t),
            lambda t: np.zeros_like(t),
        ),
        period=np.pi,
    )
    result = classify_orbit(states, t)
    assert result.primary is not None and result.primary.canonical == "resonant_2_1", (
        result.diagnostics
    )

    # 外共振 1:2：T = 2·T☾ = 4π，远距绕地圆（把月与平动点圈在内但均不局域）
    states, t = _closed_curve(
        (
            lambda t: 1.6 * np.cos(0.5 * t),
            lambda t: 1.6 * np.sin(0.5 * t),
            lambda t: np.zeros_like(t),
            lambda t: -0.8 * np.sin(0.5 * t),
            lambda t: 0.8 * np.cos(0.5 * t),
            lambda t: np.zeros_like(t),
        ),
        period=4 * np.pi,
    )
    result = classify_orbit(states, t)
    assert result.primary is not None and result.primary.canonical == "resonant_1_2", (
        result.diagnostics
    )


def test_synthetic_moon_centered_multilabel_with_resonance():
    """判据级：绕月闭合曲线周期通约时多标签（月心族 + resonant）。"""
    mu = 0.012150585350562453
    moon_x = 1 - mu
    # 逆行绕月圆，角速率 0.75 → T = (4/3)·T☾，一周期恰绕月 2π
    r = 0.05
    w = -0.75
    states, t = _closed_curve(
        (
            lambda t: moon_x + r * np.cos(w * t),
            lambda t: r * np.sin(w * t),
            lambda t: np.zeros_like(t),
            lambda t: -r * w * np.sin(w * t),
            lambda t: r * w * np.cos(w * t),
            lambda t: np.zeros_like(t),
        ),
        period=(4.0 / 3.0) * 2 * np.pi,
    )
    result = classify_orbit(states, t)
    assert result.canonical_labels == ("distant_retrograde", "resonant_3_4"), result.diagnostics


def test_synthetic_low_prograde_east_west():
    """判据级：低月顺行圆按近月点方向半平面分东西。"""
    mu = 0.012150585350562453
    moon_x = 1 - mu
    # 月心小圆（ρ_max = 偏置+r ≈0.018 < low/distant 分界，月在圆内），
    # 圆心 −y 偏置 → 近月点落在 +y（东）半平面，反之西
    for offset_y, expected in ((-0.006, "low_prograde_eastern"), (0.006, "low_prograde_western")):
        r = 0.012
        w = 2 * np.pi / 2.2
        states, t = _closed_curve(
            (
                lambda t, cx=moon_x, rr=r, ww=w: cx + rr * np.cos(ww * t),
                lambda t, oy=offset_y, rr=r, ww=w: oy + rr * np.sin(ww * t),
                lambda t: np.zeros_like(t),
                lambda t, rr=r, ww=w: -rr * ww * np.sin(ww * t),
                lambda t, rr=r, ww=w: rr * ww * np.cos(ww * t),
                lambda t: np.zeros_like(t),
            ),
            period=2.2,
        )
        result = classify_orbit(states, t)
        assert result.primary is not None and result.primary.canonical == expected, (
            expected,
            result.diagnostics,
        )


def test_synthetic_butterfly_dragonfly_vertical_crossing_patterns():
    """判据级：穿越计数/形态命中 butterfly/dragonfly/vertical。"""
    l1 = 0.8369
    # butterfly：每周期 4 次垂直 y=0 穿越的三维闭合曲线（绕 L1）
    states, t = _closed_curve(
        (
            lambda t: l1 + 0.02 * np.cos(4 * t),
            lambda t: 0.02 * np.sin(4 * t),
            lambda t: 0.01 * np.cos(4 * t),
            lambda t: -0.08 * np.sin(4 * t),
            lambda t: 0.08 * np.cos(4 * t),
            lambda t: -0.04 * np.sin(4 * t),
        ),
        period=np.pi,
    )
    result = classify_orbit(states, t)
    assert result.primary is not None and result.primary.canonical in (
        "butterfly_northern",
        "butterfly_southern",
    ), result.diagnostics

    # dragonfly：6 次垂直穿越
    states, t = _closed_curve(
        (
            lambda t: l1 + 0.02 * np.cos(6 * t),
            lambda t: 0.02 * np.sin(6 * t),
            lambda t: 0.01 * np.cos(6 * t),
            lambda t: -0.12 * np.sin(6 * t),
            lambda t: 0.12 * np.cos(6 * t),
            lambda t: -0.06 * np.sin(6 * t),
        ),
        period=np.pi,
    )
    result = classify_orbit(states, t)
    assert result.primary is not None and result.primary.canonical in (
        "dragonfly_northern",
        "dragonfly_southern",
    ), result.diagnostics

    # vertical：无 y=0 穿越、有垂直 z=0 穿越的三维闭合曲线（L1 邻域）
    states, t = _closed_curve(
        (
            lambda t: l1 + 0.015 * (1 + np.cos(4 * t)),
            lambda t: 0.05 + 0.015 * (1 - np.cos(4 * t)),
            lambda t: 0.02 * np.sin(2 * t),
            lambda t: -0.06 * np.sin(4 * t),
            lambda t: 0.06 * np.sin(4 * t),
            lambda t: 0.04 * np.cos(2 * t),
        ),
        period=np.pi,
    )
    result = classify_orbit(states, t)
    assert result.primary is not None and result.primary.canonical == "vertical_l1", (
        result.diagnostics
    )


def test_invalid_inputs():
    """非法输入走 FAILED/INVALID_INPUT。"""
    bad_shape = classify_orbit(np.zeros((3, 5)))
    assert bad_shape.status.value == "failed" and bad_shape.cause.value == "invalid_input"

    no_period = classify_orbit(np.zeros((1, 6)))
    assert no_period.cause.value == "invalid_input"

    bad_period = classify_orbit(np.zeros((1, 6)), period=-1.0)
    assert bad_period.cause.value == "invalid_input"


def test_baseline_files_present():
    """基线目录文件齐备（防止测试环境缺数据静默跳过）。"""
    jsons = {os.path.basename(p) for p in glob.glob(str(BASELINE_DIR / "*.json"))}
    for family in EXPECTED_FAMILY_LABELS:
        assert f"baseline-{family}.json" in jsons, family
