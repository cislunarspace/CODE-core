"""双圆限制性四体问题（BCR4BP）系统模块。

包含 ``BCR4BPSystem`` 类：在 ``CR3BP_System`` （地月会合旋转系）之上叠加
太阳质点摄动。双圆近似下，地月绕公共质心作圆周运动（CR3BP 假设），太阳
也在会合系中绕质心作共面圆周运动，其位置是时间 t 的解析函数，无需星历。

无量纲约定与 CR3BP 一致：距离单位 DU = 地月距离，时间单位 TU 使地月
会合系角速度为 1，总质量（地+月）为 1。太阳参数（质量、距离、角速度）
均在此约定下无量纲化。
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ...data.templates.systems import AU as _AU_KM
from .cr3bp_system import CR3BP_System


class BCR4BPSystem(CR3BP_System):
    """双圆限制性四体问题（Bicircular Restricted Four-Body Problem）系统

    地月会合旋转系 + 太阳质点摄动。太阳在会合系中作共面圆周运动：

        r_s(t) = a_s · (cos θ(t), sin θ(t), 0),   θ(t) = θ0 + ω_s · t

    其中 ω_s = n_s - 1 < 0：n_s 是太阳公转的无量纲角速度（惯性系），
    减去会合系自身角速度 1 后即太阳在会合系中的（逆行）角速度。
    系统是时间周期的，周期 ``T = 2π/|ω_s|``，约一个会合月。

    Attributes:
        sun_mass: 太阳无量纲质量 m_s = GM_sun / (GM_earth + GM_moon)
        sun_distance: 太阳圆周轨道半径 a_s（无量纲，日地平均距离 / DU）
        sun_angular_rate: 太阳在会合系中的角速度 ω_s（无量纲，负值表示逆行）
        sun_phase0: t = 0 时刻的太阳相位角 θ0（弧度）

    其余属性（mu、特征尺度、平动点等）继承自 ``CR3BP_System``。
    注意：BCR4BP 无 Jacobi 积分，``compute_libration_points`` 给出的是
    对应 CR3BP 的平动点，仅作参考位置使用。
    """

    # 太阳参数取值来源（DE440，与 e2m2e/core/spice.py 的 _GM_VALUES 一致）
    SUN_GM_KM3_S2 = 1.32712440018e11  # 太阳 GM (km^3/s^2), DE440
    EARTH_MOON_GM_KM3_S2 = 403503.235502  # 地月质心 GM (km^3/s^2), DE440
    SUN_EARTH_DISTANCE_KM = _AU_KM  # 日地平均距离 (km), GMAT nominalSun

    def __init__(
        self,
        mu: float,
        primary: str,
        secondary: str,
        sun_mass: float | None = None,
        sun_distance: float | None = None,
        sun_angular_rate: float | None = None,
        sun_phase0: float = 0.0,
    ) -> None:
        """初始化 BCR4BP 系统

        Args:
            mu: 质量参数 μ = m2/(m1+m2)
            primary: 主天体名称
            secondary: 次天体名称
            sun_mass: 太阳无量纲质量；None 时按地月系默认值
                （GM_sun / GM_EMB，DE440）
            sun_distance: 太阳圆周轨道半径（无量纲）；None 时取
                日地平均距离 / 地月距离
            sun_angular_rate: 太阳会合系角速度 ω_s；None 时按地月系默认值
                （需在 ``set_characteristic_scales`` 之后由
                ``earth_moon`` 类方法设置，直接构造时请显式给出）
            sun_phase0: t = 0 时刻的太阳相位角（弧度）
        """
        super().__init__(mu=mu, primary=primary, secondary=secondary)

        if sun_mass is None:
            sun_mass = self.SUN_GM_KM3_S2 / self.EARTH_MOON_GM_KM3_S2
        if sun_distance is None:
            sun_distance = self.SUN_EARTH_DISTANCE_KM / self.EARTH_MOON_DISTANCE_KM

        if sun_mass < 0:
            raise ValueError(f"sun_mass must be non-negative, got {sun_mass}")
        if sun_distance <= 0:
            raise ValueError(f"sun_distance must be positive, got {sun_distance}")

        self.sun_mass: float = float(sun_mass)
        self.sun_distance: float = float(sun_distance)
        # sun_angular_rate 为 None 时暂存 None，由 set_characteristic_scales
        # 按特征时间推导（见该方法注释）。
        self._sun_angular_rate: float | None = (
            float(sun_angular_rate) if sun_angular_rate is not None else None
        )
        self.sun_phase0: float = float(sun_phase0)

    @classmethod
    def earth_moon(
        cls,
        mu: float = 0.0121506683,
        sun_phase0: float = 0.0,
    ) -> BCR4BPSystem:
        """构造标准地月 BCR4BP 系统（含默认特征尺度）

        特征尺度与 ``CR3BP_System._with_default_scales`` 的地月分支一致
        （DU = 384405 km，周期 27.32 天），太阳参数取 DE440 / 日地平均
        距离推导的无量纲值。

        Args:
            mu: 地月质量参数，默认 0.0121506683（与 conftest 地月系统一致）
            sun_phase0: t = 0 时刻的太阳相位角（弧度）

        Returns:
            已初始化的 BCR4BPSystem
        """
        system = cls(mu=mu, primary="Earth", secondary="Moon", sun_phase0=sun_phase0)
        system.set_characteristic_scales(
            distance=cls.EARTH_MOON_DISTANCE_KM,
            period=27.32 * cls.DAY,
        )
        return system

    @property
    def sun_angular_rate(self) -> float:
        """太阳在会合系中的角速度 ω_s（无量纲，负值表示逆行）"""
        if self._sun_angular_rate is None:
            raise ValueError(
                "sun_angular_rate 未设置：请在构造时显式传入，"
                "或使用 BCR4BPSystem.earth_moon() / set_characteristic_scales()"
            )
        return self._sun_angular_rate

    @sun_angular_rate.setter
    def sun_angular_rate(self, value: float) -> None:
        self._sun_angular_rate = float(value)

    def set_characteristic_scales(self, distance: float, period: float) -> None:
        """设置特征尺度，并按特征时间推导默认太阳角速度

        若构造时未显式给出 ``sun_angular_rate``，则按太阳公转周期
        （儒略年，``CR3BP_System.YEAR``）推导：

            n_s = 2π·t* / P_year = P_em / P_year,   ω_s = n_s - 1

        其中 t* = period/(2π) 为特征时间，P_em 为地月轨道周期。
        显式传入的 ``sun_angular_rate`` 不受本方法影响。
        """
        super().set_characteristic_scales(distance=distance, period=period)
        if self._sun_angular_rate is None:
            self._sun_angular_rate = period / self.YEAR - 1.0

    def sun_position(self, t: float) -> npt.NDArray[np.floating]:
        """太阳在会合系中的解析位置（无量纲）

        双圆近似：太阳在会合系中作共面圆周运动，

            r_s(t) = a_s · (cos θ, sin θ, 0),   θ = θ0 + ω_s·t

        Args:
            t: 无量纲时间（以地月会合系特征时间计）

        Returns:
            太阳位置向量，形状 (3,)
        """
        theta = self.sun_phase0 + self.sun_angular_rate * t
        return np.array(
            [
                self.sun_distance * np.cos(theta),
                self.sun_distance * np.sin(theta),
                0.0,
            ]
        )

    def gravitational_parameter(self, body: str) -> float:
        """获取天体的无量纲引力参数

        在 CR3BP 约定（"primary"/"secondary"）之上额外接受 "sun"，
        返回太阳无量纲质量 m_s。
        """
        if body.lower() == "sun":
            return self.sun_mass
        return super().gravitational_parameter(body)

    def __str__(self):
        """字符串表示"""
        return (
            f"BCR4BPSystem(mu={self.mu}, "
            f"primary='{self.primary_body}', secondary='{self.secondary_body}', "
            f"sun_mass={self.sun_mass:.4f}, sun_distance={self.sun_distance:.4f})"
        )

    def __repr__(self):
        """详细表示"""
        return (
            f"BCR4BPSystem(mu={self.mu}, "
            f"primary='{self.primary_body}', secondary='{self.secondary_body}', "
            f"sun_mass={self.sun_mass:.6f}, sun_distance={self.sun_distance:.6f}, "
            f"sun_angular_rate={self._sun_angular_rate}, "
            f"sun_phase0={self.sun_phase0}, initialized={self.is_initialized})"
        )

