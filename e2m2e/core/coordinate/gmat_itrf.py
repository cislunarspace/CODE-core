"""GMAT 风格 ITRF reduction stages。"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .gmat_eop import EopFile, EopSample
from .gmat_time import TimeSystemConverter
from .xys import XysProvider

SECONDS_PER_DAY = 86400.0
J2000_JD = 2451545.0
J2000_MJD = 51544.5
DAYS_PER_JULIAN_CENTURY = 36525.0
EARTH_ROTATION_RATE = 7.292115146706979e-5
ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)


def rotation1(angle: float) -> npt.NDArray[np.floating]:
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, cos_angle, sin_angle], [0.0, -sin_angle, cos_angle]])


def rotation2(angle: float) -> npt.NDArray[np.floating]:
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return np.array([[cos_angle, 0.0, -sin_angle], [0.0, 1.0, 0.0], [sin_angle, 0.0, cos_angle]])


def rotation3(angle: float) -> npt.NDArray[np.floating]:
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return np.array([[cos_angle, sin_angle, 0.0], [-sin_angle, cos_angle, 0.0], [0.0, 0.0, 1.0]])


def skew(vector: npt.ArrayLike) -> npt.NDArray[np.floating]:
    x, y, z = np.asarray(vector, dtype=float)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def celestial_intermediate_matrix(x: float, y: float, s: float) -> npt.NDArray[np.floating]:
    root = np.sqrt(1.0 - x * x - y * y)
    b = 1.0 / (1.0 + root)
    matrix = np.array(
        [
            [1.0 - b * x * x, -b * x * y, x],
            [-b * x * y, 1.0 - b * y * y, y],
            [-x, -y, 1.0 - b * (x * x + y * y)],
        ]
    )
    return matrix @ rotation3(s)


class GmatItrfReduction:
    """GMAT 现代 ITRF reduction 的 first-phase Python 实现。"""

    def __init__(
        self,
        time_converter: TimeSystemConverter,
        eop: EopFile,
        xys_provider: XysProvider,
        *,
        eop_extrapolation: str = "raise",
    ) -> None:
        self._time_converter = time_converter
        self._eop = eop
        self._xys_provider = xys_provider
        self._eop_extrapolation = eop_extrapolation

    def eop_sample_for_et(self, et: float) -> EopSample:
        utc_mjd = self._time_converter.et_to_utc_mjd(et)
        return self._eop.at_utc_mjd(utc_mjd, extrapolation=self._eop_extrapolation)

    def rotation_and_rate(
        self, et: float
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        utc_mjd = self._time_converter.et_to_utc_mjd(et)
        tt_mjd = self._time_converter.et_to_tt_mjd(et)
        eop_sample = self._eop.at_utc_mjd(utc_mjd, extrapolation=self._eop_extrapolation)
        ut1_mjd = utc_mjd + eop_sample.ut1_utc / SECONDS_PER_DAY
        jd_ut1 = ut1_mjd + 2400000.5
        t_tt = (tt_mjd - J2000_MJD) / DAYS_PER_JULIAN_CENTURY

        s_prime = -0.000047 * ARCSEC_TO_RAD * t_tt
        polar_motion = (
            rotation3(-s_prime) @ rotation2(eop_sample.x_rad) @ rotation1(eop_sample.y_rad)
        )
        theta = np.fmod(
            2.0 * np.pi * (0.7790572732640 + 1.00273781191135448 * (jd_ut1 - J2000_JD)),
            2.0 * np.pi,
        )
        x, y, s = self._xys_provider.xys(tt_mjd)
        celestial = celestial_intermediate_matrix(x, y, s)
        rotation = celestial @ rotation3(-theta) @ polar_motion

        omega_earth = EARTH_ROTATION_RATE * (1.0 - eop_sample.lod / SECONDS_PER_DAY)
        rate = celestial @ rotation3(-theta) @ skew([0.0, 0.0, omega_earth]) @ polar_motion
        return rotation, rate
