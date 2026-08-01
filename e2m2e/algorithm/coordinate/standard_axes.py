"""常用坐标轴实现。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from ...data.frames.eop import CoordinateDataError, EopFile
from ...data.frames.gmat_fixture import gmat_fixture_path
from ...data.frames.leap_seconds import TaiUtcTable
from ...data.kernels._spice_loader import get_spiceypy
from .axes import Axes
from .coordinate_system import CoordinateSystem
from .gmat_itrf import GmatItrfReduction
from .gmat_time import TimeSystemConverter
from .iau_2006 import iau2000eq_matrix
from .standard_origins import InertialOrigin
from .xys import ErfaXysProvider, XysProvider


class ICRSAxes(Axes):
    """国际天球参考系（ICRF）坐标轴。"""

    def rotation_matrix(self, et: float) -> npt.NDArray[np.floating]:
        return np.eye(3)

    def rotation_and_rate(
        self, et: float
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        return np.eye(3), np.zeros((3, 3))


class IAU2000EqAxes(Axes):
    """基于简化 IAU 2000/2006 岁差章动模型的近似惯性轴。"""

    _DEFAULT_TIME_STEP = 1.0

    def __init__(self, time_step: float = _DEFAULT_TIME_STEP) -> None:
        self._time_step = time_step

    def rotation_matrix(self, et: float) -> npt.NDArray[np.floating]:
        return iau2000eq_matrix(et)


class ITRFSpiceAxes(Axes):
    """SPICE-backed 高精度 ITRF 坐标轴，默认使用 ``ITRF93``。"""

    _DEFAULT_FRAME = "ITRF93"

    def __init__(self, frame: str = _DEFAULT_FRAME) -> None:
        self._frame = frame

    @property
    def frame(self) -> str:
        return self._frame

    def rotation_matrix(self, et: float) -> npt.NDArray[np.floating]:
        try:
            return np.array(get_spiceypy().pxform(self._frame, "J2000", et))
        except Exception as exc:  # pragma: no cover
            raise CoordinateDataError(
                "SPICE ITRF transform unavailable. Load an LSK, text PCK, and Earth binary PCK "
                f"that define {self._frame}; no fallback to IAU_EARTH is performed."
            ) from exc

    def rotation_and_rate(
        self, et: float
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        try:
            transform = np.array(get_spiceypy().sxform(self._frame, "J2000", et))
        except Exception as exc:  # pragma: no cover
            raise CoordinateDataError(
                "SPICE ITRF state transform unavailable. Load an LSK, text PCK, and Earth binary "
                f"PCK that define {self._frame}; no fallback to IAU_EARTH is performed."
            ) from exc
        return transform[:3, :3], transform[3:, :3]


class ITRFAxes(ITRFSpiceAxes):
    """向后兼容别名：默认精确 ITRF 为 SPICE-backed ``ITRF93``。"""


class GMATITRFAxes(Axes):
    """显式 opt-in 的第一阶段 GMAT-compatible ITRF 坐标轴。"""

    def __init__(
        self,
        *,
        tai_utc_path: str | Path | None = None,
        eop_path: str | Path | None = None,
        xys_provider: XysProvider | None = None,
        eop_extrapolation: str = "raise",
        compatibility: str | None = None,
    ) -> None:
        if compatibility == "gmat" and eop_extrapolation == "raise":
            eop_extrapolation = "clamp"
        if eop_extrapolation not in {"raise", "clamp"}:
            raise ValueError("eop_extrapolation must be 'raise' or 'clamp'")
        tai_path = (
            Path(tai_utc_path) if tai_utc_path is not None else gmat_fixture_path("tai-utc.dat")
        )
        eop_fixture_path = (
            Path(eop_path)
            if eop_path is not None
            else gmat_fixture_path("eopc04_08.62-now.trimmed")
        )
        self._reduction = GmatItrfReduction(
            TimeSystemConverter(TaiUtcTable.from_file(tai_path)),
            EopFile.from_file(eop_fixture_path),
            xys_provider or ErfaXysProvider(),
            eop_extrapolation=eop_extrapolation,
        )

    def rotation_matrix(self, et: float) -> npt.NDArray[np.floating]:
        rotation, _rate = self.rotation_and_rate(et)
        return rotation

    def rotation_and_rate(
        self, et: float
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        return self._reduction.rotation_and_rate(et)


class ITRFApproxAxes(Axes):
    """低精度/教学用途的近似 ITRF 坐标轴。"""

    _DEFAULT_TIME_STEP = 1.0

    def __init__(self, time_step: float = _DEFAULT_TIME_STEP) -> None:
        self._iau2000eq = IAU2000EqAxes(time_step=time_step)
        self._time_step = time_step

    def rotation_matrix(self, et: float) -> npt.NDArray[np.floating]:
        precession = self._iau2000eq.rotation_matrix(et)
        gast = _greenwich_apparent_sidereal_time(et)
        return precession @ _rotation3(gast)


def standard_itrf() -> ITRFSpiceAxes:
    """返回公共默认 ITRF：SPICE-backed ``ITRF93``。"""
    return ITRFSpiceAxes()


def standard_icrf() -> CoordinateSystem:
    """返回 ICRF 标准坐标系预设。

    ICRF = ICRSAxes(恒等旋转,与 ICRF/J2000 同向)+ InertialOrigin(太阳系
    质心,无平移)。这是地心 ITRF 转换的惯性别,常作为 ``CoordinateSystem``
    组合的一端。

    与 ``standard_itrf`` 不同,这里直接返回完整 ``CoordinateSystem``——
    ICRF 预设在调用方通常直接用作转换源/目标,不需再自行拼 Axes + Origin。
    """
    return CoordinateSystem(axes=ICRSAxes(), origin=InertialOrigin())


def _greenwich_apparent_sidereal_time(et: float) -> float:
    from .iau_2006 import _seconds_to_julian_centuries, nutation_angles

    t = _seconds_to_julian_centuries(et)
    days = et / 86400.0
    gmst_deg = 280.46061837 + 360.98564736629 * days + 0.000387933 * t**2 - t**3 / 38710000.0
    gmst = np.deg2rad(gmst_deg % 360.0)
    dpsi, _deps, eps0 = nutation_angles(t)
    return gmst + dpsi * np.cos(eps0)


def _rotation3(angle: float) -> npt.NDArray[np.floating]:
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return np.array([[cos_angle, -sin_angle, 0.0], [sin_angle, cos_angle, 0.0], [0.0, 0.0, 1.0]])
