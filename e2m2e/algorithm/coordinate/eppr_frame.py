"""星历脉动旋转坐标系（Earth-Moon EPPR）的帧量推导。

EPPR（Ephemeris Pulsing Rotating frame）：x̂ 沿地球→月球瞬时连线、ẑ 沿瞬时
地月轨道角动量、ŷ 右手补全；长度尺度取瞬时地月距离 d(t)（脉动）。无量纲时间
与 CR3BP 约定一致（t_c = ``Datum.DE421.char_time_s``，lstar =
``Datum.DE421.char_length_km``）；地球固定在 EPPR 无量纲 (−μ, 0, 0)，月球固定在
(1−μ, 0, 0)。

``EPPRFrameState`` 汇集某时刻的全部帧量，``EPPRFrameModel`` 负责从唯一星历
查询路径（``SPICEManager.get_body_state``，cache-aware）解析推导并缓存。轴向
约定与 ``SynodicAxes`` 同定义：``r_icrf = R @ r_eppr_axes``。

公式沿用并三维推广 ``crates/e2m2e-hjb-dynamics/src/ephemeris.rs`` 的
``EphemerisPlanar::frame``（d、ḋ、d̈ 与角速度解析式）；角速度的解析形式（而非
``SynodicAxes.rotation_and_rate`` 的中心差分）是必需的——Ẇ 需要在 ω 上再差分
一次，数值 Ṙ 的舍入基底（~1e-16/s）经 t_c² ≈ 1.4e11 放大到 ~1e-5 无量纲，会
污染 Euler 项；解析 ω 的舍入基底差分后仅 ~1e-7 无量纲。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .synodic_axes import SynodicAxes


@dataclass(frozen=True)
class EPPRFrameState:
    """某时刻的 EPPR 帧量快照（长度 km、时间 s）。

    Attributes:
        et: 快照对应的 SPICE 历书时（秒）。
        distance: 瞬时地月距离 d(t)，km。
        distance_rate: ḋ，km/s。
        distance_accel: d̈，km/s²（速度中心差分求取）。
        rotation: R (3, 3)，``r_icrf = R @ r_eppr_axes``。
        rotation_rate: Ṙ (3, 3)，1/s（解析，非数值微分）。
        angular_velocity: W (3,)，帧分量角速度，1/s（``skew(W) = RᵀṘ``）。
        angular_acceleration: Ẇ (3,)，帧分量角加速度，1/s²。
        barycenter_accel: a_bc (3,)，地月质心平动加速度，J2000 惯性分量 km/s²。
    """

    et: float
    distance: float
    distance_rate: float
    distance_accel: float
    rotation: npt.NDArray[np.floating]
    rotation_rate: npt.NDArray[np.floating]
    angular_velocity: npt.NDArray[np.floating]
    angular_acceleration: npt.NDArray[np.floating]
    barycenter_accel: npt.NDArray[np.floating]


class EPPRFrameModel:
    """由 SPICE 星历推导 EPPR 帧量并缓存。

    ``frame_state(et)`` 的推导只依赖 ``_moon_rel_state`` / ``_earth_abs_state``
    两个查询 seam：测试可用合成星历子类化这两个方法（无需 SPICE）。
    """

    #: 速度中心差分求加速度的步长（秒）。与 ``SynodicAxes._DEFAULT_RATE_STEP``
    #: 同口径：1.0s 处于舍入-截断平衡平台。
    _ACCEL_STEP = 1.0
    #: ω 中心差分求 Ẇ 的步长（秒）。100s 使 ω 的二阶差商误差低于 Ẇ 的目标
    #: 量级（~1e-7 无量纲）。
    _OMEGA_RATE_STEP = 100.0
    #: ``frame_state`` 缓存容量（FIFO 淘汰）。
    _CACHE_CAPACITY = 256

    def __init__(self, spice, mu: float) -> None:
        self._spice = spice
        self._mu = float(mu)
        self._cache: dict[float, EPPRFrameState] = {}

    # ---- 星历查询 seam（合成星历测试子类化用） ----

    def _moon_rel_state(self, et: float) -> npt.NDArray[np.floating]:
        """月球相对地球的 J2000 状态 (6,)：[km, km/s]。"""
        return np.asarray(self._spice.get_body_state("MOON", et, "J2000", "EARTH"), dtype=float)

    def _earth_abs_state(self, et: float) -> npt.NDArray[np.floating]:
        """地球相对太阳系质心的 J2000 状态 (6,)：[km, km/s]。"""
        return np.asarray(
            self._spice.get_body_state("EARTH", et, "J2000", "SOLAR SYSTEM BARYCENTER"),
            dtype=float,
        )

    # ---- 帧量推导 ----

    def _frame_basis(
        self, et: float
    ) -> tuple[
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
        float,
        float,
        float,
        npt.NDArray[np.floating],
    ]:
        """返回 ``(R, Ṙ, d, ḋ, d̈, 月球相对地球加速度 a)``。"""
        moon_state = self._moon_rel_state(et)
        r = moon_state[:3]
        v = moon_state[3:]
        step = self._ACCEL_STEP
        a = (self._moon_rel_state(et + step)[3:] - self._moon_rel_state(et - step)[3:]) / (
            2.0 * step
        )

        d = float(np.linalg.norm(r))
        h = np.cross(r, v)
        h_norm = float(np.linalg.norm(h))
        if d <= 0.0 or h_norm == 0.0:
            raise ValueError("退化星历：地月距离为零或月球轨道角动量为零，无法定义 EPPR 帧")

        d_dot = float(np.dot(r, v)) / d
        d_ddot = (float(np.dot(v, v)) + float(np.dot(r, a))) / d - d_dot * d_dot / d

        e3 = h / h_norm
        e1 = r / d
        e1_rate = v / d - r * (float(np.dot(r, v)) / d**3)
        h_rate = np.cross(r, a)
        e3_rate = h_rate / h_norm - h * (float(np.dot(h, h_rate)) / h_norm**3)
        e2_rate = np.cross(e3_rate, e1) + np.cross(e3, e1_rate)

        # 基与 SynodicAxes 同定义（同一静态构造），只额外给出解析速率。
        rotation = SynodicAxes._build_rotation_matrix(r, v)
        rotation_rate = np.column_stack([e1_rate, e2_rate, e3_rate])
        return rotation, rotation_rate, d, d_dot, d_ddot, a

    @staticmethod
    def _omega_from_rate(
        rotation: npt.NDArray[np.floating], rotation_rate: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """由 ``(R, Ṙ)`` 取帧分量角速度向量（``skew(W) = RᵀṘ``）。"""
        skew = rotation.T @ rotation_rate
        return np.array([skew[2, 1], skew[0, 2], skew[1, 0]])

    def _omega(self, et: float) -> npt.NDArray[np.floating]:
        rotation, rotation_rate, *_ = self._frame_basis(et)
        return self._omega_from_rate(rotation, rotation_rate)

    def _derive(self, et: float) -> EPPRFrameState:
        rotation, rotation_rate, d, d_dot, d_ddot, a_rel = self._frame_basis(et)
        omega = self._omega_from_rate(rotation, rotation_rate)
        omega_step = self._OMEGA_RATE_STEP
        omega_dot = (self._omega(et + omega_step) - self._omega(et - omega_step)) / (
            2.0 * omega_step
        )

        step = self._ACCEL_STEP
        a_earth = (self._earth_abs_state(et + step)[3:] - self._earth_abs_state(et - step)[3:]) / (
            2.0 * step
        )
        # a_bc = (1−μ)Ẍ_E + μ(Ẍ_E + r̈_rel) = Ẍ_E + μ·r̈_rel
        barycenter_accel = a_earth + self._mu * a_rel

        return EPPRFrameState(
            et=et,
            distance=d,
            distance_rate=d_dot,
            distance_accel=d_ddot,
            rotation=rotation,
            rotation_rate=rotation_rate,
            angular_velocity=omega,
            angular_acceleration=omega_dot,
            barycenter_accel=barycenter_accel,
        )

    def _evict_oldest(self) -> None:
        if len(self._cache) >= self._CACHE_CAPACITY:
            self._cache.pop(next(iter(self._cache)))

    def frame_state(self, et: float) -> EPPRFrameState:
        """返回 ``et`` 时刻的 EPPR 帧量（命中缓存则复用）。"""
        cached = self._cache.get(et)
        if cached is not None:
            return cached
        state = self._derive(et)
        self._evict_oldest()
        self._cache[et] = state
        return state
