"""forces 测试共享基准物理。

- 常量：近地场景默认基准取 ``Datum.WGS84``（ADR 0022），与 integrators
  测试一致；各力模型的解析/文件自带常量（EGM96、GRGM900C 等）保留在
  各自测试文件内并标注来源。
- 工具：``keplerian_to_cartesian`` / ``semi_major_axis`` 供 LEO/地月
  传播场景共用（原先各文件重复抄写，收敛到本模块）。
- fixture：``point_mass_force``（点质量力）、``earth_icrf_system``
  （地心 ICRF 星历系统，加载地月日三星历 + body-fixed 内核）。
"""

import numpy as np
import pytest

from e2m2e.algorithm.coordinate.coordinate_system import CoordinateSystem
from e2m2e.algorithm.coordinate.standard_axes import ICRSAxes
from e2m2e.algorithm.coordinate.standard_origins import CelestialBodyOrigin
from e2m2e.algorithm.dynamics.ephemeris_system import EphemerisSystem
from e2m2e.algorithm.forces import PointMassGravity
from e2m2e.data.constants import Datum
from e2m2e.data.kernels.manager import SPICEManager

# 地球引力参数与赤道半径：WGS-84 基准（近地场景默认基准）。
EARTH_MU = Datum.WGS84.earth_gm  # km³/s²
EARTH_RE = Datum.WGS84.earth_radius_km  # km
# 月球引力参数：DE421 基准（地月系统默认基准）。
MOON_MU = Datum.DE421.moon_gm  # km³/s²

# 哨兵：区分"未传 spice"（默认有 SPICE）与"显式传 None"（模拟资源缺失）。
_SPICE_UNSET = object()


class FakeSystem:
    """最小 System 桩，供不需要 SPICE 的契约/传播测试使用。

    - ``coordinate_system``：默认占位对象，可用 ``has_coordinate_system=False`` 关闭；
    - ``spice``：默认占位对象（模拟有 SPICE：力无 Rust spec 属能力缺失）；
      传 ``spice=None`` 模拟资源缺失（ADR 0020 分流）；
    - ``gravitational_parameter``：EARTH 用 ``Datum.WGS84.earth_gm``（近地场景
      默认基准）、MOON 用 ``Datum.DE421.moon_gm``（地月系统默认基准，ADR 0022）。
    """

    def __init__(
        self,
        has_coordinate_system: bool = True,
        spice: object | None = _SPICE_UNSET,
    ):
        self.coordinate_system = object() if has_coordinate_system else None
        self.origin = "EARTH"
        self.spice = object() if spice is _SPICE_UNSET else spice

    @property
    def frame(self):
        from e2m2e.data.templates.enums import ReferenceFrame

        return ReferenceFrame.J2000

    @property
    def unit_system(self):
        from e2m2e.data.templates.enums import UnitSystem

        return UnitSystem.SI

    def gravitational_parameter(self, body: str) -> float:
        b = body.upper()
        if b == "EARTH":
            return EARTH_MU
        if b == "MOON":
            return MOON_MU
        raise ValueError(f"unknown body {body}")


def keplerian_to_cartesian(a, e, i, raan, argp, nu, mu):
    """开普勒根数（角度制）→ J2000 笛卡尔状态 [r, v]（km, km/s）。"""
    p = a * (1 - e**2)
    r = p / (1 + e * np.cos(nu))

    i = np.radians(i)
    raan = np.radians(raan)
    argp = np.radians(argp)
    nu = np.radians(nu)

    r_pqw = np.array([r * np.cos(nu), r * np.sin(nu), 0.0])
    v_pqw = np.array(
        [
            -np.sqrt(mu / p) * np.sin(nu),
            np.sqrt(mu / p) * (e + np.cos(nu)),
            0.0,
        ]
    )

    R3_raan = np.array(
        [
            [np.cos(raan), -np.sin(raan), 0.0],
            [np.sin(raan), np.cos(raan), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    R1_i = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(i), -np.sin(i)],
            [0.0, np.sin(i), np.cos(i)],
        ]
    )
    R3_argp = np.array(
        [
            [np.cos(argp), -np.sin(argp), 0.0],
            [np.sin(argp), np.cos(argp), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    R = R3_raan @ R1_i @ R3_argp

    r_eci = R @ r_pqw
    v_eci = R @ v_pqw
    return np.concatenate([r_eci, v_eci])


def semi_major_axis(state, mu):
    """由状态向量用能量公式计算半长轴。"""
    r = np.linalg.norm(state[:3])
    v = np.linalg.norm(state[3:6])
    energy = v**2 / 2.0 - mu / r
    return -mu / (2.0 * energy)


@pytest.fixture
def point_mass_force():
    """地球引力参数下的点质量力 fixture。"""
    return PointMassGravity("EARTH", mu=EARTH_MU)


@pytest.fixture
def earth_icrf_system(spice_kernel_path):
    """地心 ICRF 星历系统（加载地月日三星历 + body-fixed 内核）。

    body-fixed 内核（ITRF93/MOON_PA，``kernel_helpers.BODY_FIXED_KERNELS``）
    文件缺失时静默跳过；阻力/潮汐类测试在需要处自行探测可用性。
    """
    from kernel_helpers import load_body_fixed_kernels, unload_kernels

    spice = SPICEManager()
    spice.load_kernel(spice_kernel_path)
    bf_kernels = load_body_fixed_kernels(spice)
    try:
        system = EphemerisSystem(
            bodies=["EARTH", "MOON", "SUN"],
            spice=spice,
            origin="EARTH",
        )
        system.coordinate_system = CoordinateSystem(
            axes=ICRSAxes(),
            origin=CelestialBodyOrigin(body="EARTH", spice=spice),
        )
        yield system
    finally:
        unload_kernels(spice, bf_kernels)
        spice.unload_kernel(spice_kernel_path)
