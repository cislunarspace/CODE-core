"""NRLMSISE00Atmosphere 大气密度模型测试。

覆盖：默认参数与 Ap 史规范化、构造校验、配置 round-trip、``DragModel.to_rust_spec``
的 NRLMSISE-00 元组契约，以及与 nyx 公开验证夹具的对拍（需 SPICE 内核与
``nrlmsise00_density_py`` 符号）。
"""

from __future__ import annotations

import numpy as np
import pytest
from kernel_helpers import requires_native_symbols

from e2m2e.algorithm.forces import DragModel, ForceModel
from e2m2e.algorithm.forces.atmosphere import ExponentialAtmosphere, NRLMSISE00Atmosphere
from e2m2e.algorithm.forces.force_config import build_force, serialize_force
from tests.numerical.forces.conftest import FakeSystem

pytestmark = pytest.mark.force


# --- 默认参数与 Ap 史规范化 ---


def test_defaults_are_mid_solar_and_geomagnetic_activity():
    """默认 f107=150/150、Ap=15（中间活动水平），ap 恒为 7 元。"""
    atm = NRLMSISE00Atmosphere()
    assert atm.f107_daily == 150.0
    assert atm.f107_avg == 150.0
    assert atm.ap == (15.0,) * 7


def test_f107_avg_defaults_to_daily():
    """f107_avg 缺省时取 f107_daily。"""
    atm = NRLMSISE00Atmosphere(f107_daily=180.0)
    assert atm.f107_avg == 180.0
    assert NRLMSISE00Atmosphere(f107_daily=180.0, f107_avg=200.0).f107_avg == 200.0


def test_scalar_ap_broadcasts_to_seven():
    """标量 Ap 广播成 7 元全等史（静态空间天气）。"""
    atm = NRLMSISE00Atmosphere(ap=42.0)
    assert atm.ap == (42.0,) * 7


def test_seven_element_ap_is_kept_verbatim():
    """7 元 Ap 史原样保留（非全等 = 暴时历史）。"""
    history = (15.0, 130.0, 150.0, 50.0, 20.0, 15.0, 15.0)
    atm = NRLMSISE00Atmosphere(ap=history)
    assert atm.ap == history


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"f107_daily": 0.0}, "f107_daily"),
        ({"f107_daily": -1.0}, "f107_daily"),
        ({"f107_daily": float("nan")}, "f107_daily"),
        ({"f107_avg": 0.0}, "f107_avg"),
        ({"ap": [15.0] * 3}, "ap"),
        ({"ap": [15.0] * 8}, "ap"),
        ({"ap": -1.0}, "ap"),
        ({"ap": [15.0] * 6 + [float("inf")]}, "ap"),
    ],
)
def test_invalid_parameters_raise(kwargs, match):
    """f107 非正/非有限、ap 形状错误或含负值/非有限值都抛 ValueError。"""
    with pytest.raises(ValueError, match=match):
        NRLMSISE00Atmosphere(**kwargs)


# --- ExponentialAtmosphere 的接口统一 ---


def test_exponential_density_accepts_and_ignores_epoch_arguments():
    """指数大气接受 NRLMSISE-00 的历元/坐标参数并忽略，数值与位置调用完全一致。"""
    atm = ExponentialAtmosphere(f107=200.0, ap=50.0)
    baseline = atm.density(400.0)
    assert (
        atm.density(400.0, epoch_et=1234.5, geodetic_lat_deg=45.0, geodetic_lon_deg=120.0)
        == baseline
    )


# --- 配置序列化 / 构造 ---


def test_serialize_build_round_trip_scalar_ap():
    """NRLMSISE-00 阻力经 serialize_force → build_force 无损往返（标量 Ap）。"""
    force = DragModel(
        atmosphere=NRLMSISE00Atmosphere(f107_daily=180.0, f107_avg=170.0, ap=25.0),
        area=5.0,
        mass=1000.0,
        cd=2.2,
    )
    config = serialize_force(force)
    assert config == {
        "type": "DragModel",
        "params": {
            "area": 5.0,
            "mass": 1000.0,
            "body": "EARTH",
            "cd": 2.2,
            "atmosphere": {
                "type": "NRLMSISE00Atmosphere",
                "params": {"f107_daily": 180.0, "f107_avg": 170.0, "ap": 25.0},
            },
        },
    }
    rebuilt = build_force(config["type"], config["params"])
    assert serialize_force(rebuilt) == config


def test_serialize_build_round_trip_ap_history():
    """非全等 7 元 Ap 史序列化为 7 元列表并往返一致。"""
    history = [15.0, 130.0, 150.0, 50.0, 20.0, 15.0, 15.0]
    force = DragModel(
        atmosphere=NRLMSISE00Atmosphere(ap=history),
        area=5.0,
        mass=1000.0,
    )
    config = serialize_force(force)
    atmo = config["params"]["atmosphere"]
    assert atmo["type"] == "NRLMSISE00Atmosphere"
    assert atmo["params"]["ap"] == history
    assert isinstance(atmo["params"]["ap"], list)
    rebuilt = build_force(config["type"], config["params"])
    assert serialize_force(rebuilt) == config


def test_serialize_flattens_uniform_ap_history():
    """7 元全等史（等价于静态输入）压成标量，避免配置里出现同值列表。"""
    force = DragModel(atmosphere=NRLMSISE00Atmosphere(ap=[20.0] * 7), area=5.0, mass=1000.0)
    config = serialize_force(force)
    assert config["params"]["atmosphere"]["params"]["ap"] == 20.0


def test_build_unknown_atmosphere_type_raises():
    """未知 atmosphere type 抛 ValueError，并列出已知类型。"""
    with pytest.raises(ValueError, match="unknown atmosphere type"):
        build_force(
            "DragModel",
            {
                "area": 5.0,
                "mass": 1000.0,
                "cd": 2.2,
                "atmosphere": {"type": "USSA76", "params": {}},
            },
        )


def test_nrlmsise00_drag_config_round_trip():
    """ForceModel 级配置 round-trip：NRLMSISE-00 大气递归进入 {type, params}。"""
    system = FakeSystem()
    fm = ForceModel(system)
    fm.add_force(
        DragModel(
            atmosphere=NRLMSISE00Atmosphere(f107_daily=200.0, f107_avg=180.0, ap=30.0),
            area=5.0,
            mass=1000.0,
            cd=2.2,
            body="EARTH",
        ),
        name="drag",
    )

    config = ForceModel.to_config(fm)
    fm2 = ForceModel.from_config(config, system)

    entry = config["forces"][0]
    assert entry["params"]["atmosphere"] == {
        "type": "NRLMSISE00Atmosphere",
        "params": {"f107_daily": 200.0, "f107_avg": 180.0, "ap": 30.0},
    }
    assert ForceModel.to_config(fm2) == config


# --- to_rust_spec 元组契约 ---


def test_to_rust_spec_carries_nrlmsise00_tuple():
    """NRLMSISE-00 大气序列化为 ``("drag_nrlmsise00", area, mass, cd, frame, ...)``。"""
    history = (15.0, 130.0, 150.0, 50.0, 20.0, 15.0, 15.0)
    drag = DragModel(
        atmosphere=NRLMSISE00Atmosphere(f107_daily=200.0, f107_avg=180.0, ap=history),
        area=10.0,
        mass=1000.0,
        cd=2.2,
    )
    spec = drag.to_rust_spec(FakeSystem())
    assert spec is not None
    assert spec[0] == "drag_nrlmsise00"
    assert spec[1:5] == (10.0, 1000.0, 2.2, "J2000")
    assert spec[5] == pytest.approx(200.0)
    assert spec[6] == pytest.approx(180.0)
    assert tuple(spec[7]) == history


def test_to_rust_spec_nrlmsise00_none_without_spice():
    """system 无 spice 时 NRLMSISE-00 阻力同样返回 None。"""
    drag = DragModel(atmosphere=NRLMSISE00Atmosphere(), area=10.0, mass=1000.0)
    assert drag.to_rust_spec(FakeSystem(spice=None)) is None


# --- 与 nyx 公开验证夹具对拍 ---
#
# 期望值取自 nyx-space/nyx 的 data/03_tests/nrlmsise00_validation.json
# （仓库内转录为 crates/e2m2e-forces/tests/data/nrlmsise00_validation.csv），
# 均为 doy=1（2000-01-01）、f107_daily=f107_avg=150 sfu 的工况。
# 夹具是 f32 精度产物，故容差取 1e-5（Rust 侧对拍实测最大偏差 ~1.4e-6）。
_NYX_CASES = [
    # (CSV 记录序号, utc, alt_km, lat_deg, lon_deg, ap, 密度, 温度)
    (
        0,  # 记录 0：100 km、赤道、0600 UT、暴时 Ap 史
        "2000-01-01T06:00:00",
        100.0,
        0.0,
        0.0,
        (15, 130, 150, 50, 20, 15, 15),
        4.70059489998675417e-07,
        1.88600997924804688e02,
    ),
    (
        16,  # 记录 16：483 km、赤道、暴时 Ap 史
        "2000-01-01T06:00:00",
        483.0,
        0.0,
        0.0,
        (15, 130, 150, 50, 20, 15, 15),
        1.13102745485232914e-12,
        9.75571594238281250e02,
    ),
    (
        17,  # 记录 17：483 km、赤道、静态 Ap
        "2000-01-01T06:00:00",
        483.0,
        0.0,
        0.0,
        (15, 15, 15, 15, 15, 15, 15),
        6.68412211430463588e-13,
        9.03853881835937500e02,
    ),
    (
        47,  # 记录 47：483 km、80° 纬度、静态 Ap
        "2000-01-01T12:00:00",
        483.0,
        80.0,
        0.0,
        (15, 15, 15, 15, 15, 15, 15),
        8.47542901746128896e-13,
        1.00122430419921875e03,
    ),
]


@requires_native_symbols("nrlmsise00_density_py")
@pytest.mark.spice
@pytest.mark.parametrize("case", _NYX_CASES, ids=lambda c: f"nyx_row{c[0]}")
def test_density_matches_nyx_validation(case, earth_icrf_system):
    """密度与温度对拍 nyx 公开夹具（相对误差 ≤ 1e-5，夹具为 f32 精度）。"""
    _, utc, alt_km, lat_deg, lon_deg, ap, want_rho, want_temp = case
    spice = earth_icrf_system.spice
    epoch_et = spice.utc_to_et(utc)

    atm = NRLMSISE00Atmosphere(f107_daily=150.0, f107_avg=150.0, ap=[float(v) for v in ap])
    rho = atm.density(alt_km, epoch_et=epoch_et, geodetic_lat_deg=lat_deg, geodetic_lon_deg=lon_deg)
    assert rho == pytest.approx(want_rho, rel=1e-5)

    from e2m2e.integrators import nrlmsise00_density_py, require_rust_extension

    require_rust_extension("nrlmsise00_density_py")
    rho_direct, temp = nrlmsise00_density_py(
        epoch_et, alt_km, lat_deg, lon_deg, 150.0, 150.0, [float(v) for v in ap]
    )
    assert rho_direct == pytest.approx(rho, rel=1e-15)
    assert temp == pytest.approx(want_temp, rel=1e-5)


@requires_native_symbols("nrlmsise00_density_py")
@pytest.mark.spice
def test_density_above_ceiling_is_zero(earth_icrf_system):
    """1000 km 及以上返回零密度（与指数模型的上限约定一致）。"""
    spice = earth_icrf_system.spice
    epoch_et = spice.utc_to_et("2000-01-01T06:00:00")
    atm = NRLMSISE00Atmosphere()
    assert atm.density(1000.0, epoch_et=epoch_et, geodetic_lat_deg=0.0, geodetic_lon_deg=0.0) == 0.0
    assert atm.density(2000.0, epoch_et=epoch_et, geodetic_lat_deg=0.0, geodetic_lon_deg=0.0) == 0.0


@requires_native_symbols("nrlmsise00_density_py")
@pytest.mark.spice
def test_ap_history_increases_density_over_static(earth_icrf_system):
    """同一历元下 Ap 史（暴时）比静态 Ap 给出更高密度（磁活动加热效应）。"""
    spice = earth_icrf_system.spice
    epoch_et = spice.utc_to_et("2000-01-01T06:00:00")
    common = {"epoch_et": epoch_et, "geodetic_lat_deg": 0.0, "geodetic_lon_deg": 0.0}

    static = NRLMSISE00Atmosphere(ap=15.0).density(483.0, **common)
    storm = NRLMSISE00Atmosphere(ap=[15.0, 130.0, 150.0, 50.0, 20.0, 15.0, 15.0]).density(
        483.0, **common
    )
    assert storm > static
    assert np.isfinite(storm) and np.isfinite(static)
