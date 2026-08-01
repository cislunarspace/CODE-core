"""标准形化简上下文对象 ``NormalFormContext``。

集中存放一条标准形化简流水线所需的全部静态/派生数据：
归一化常量、平动点几何、基础频率、中心流形频率、特征指数、用户传入的
历元与展开阶数。本切片只交付构造与读取；后续切片在该对象上调用具体
化简器（DynamicalSubstituteCorrector、QuasiFloquetReducer 等）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ..dynamics import LibrationPoint, System
from .constants import (
    BASE_FREQUENCIES,
    JD0_J2000,
    LU_KM,
    MU,
    MU_E,
    MU_M,
    MU_S,
    TU_S,
    central_frequencies,
    characteristic_exponent,
    compute_libration_position,
    libration_gamma,
)

if TYPE_CHECKING:
    from datetime import datetime


class NormalFormContext:
    """标准形化简流水线上下文。

    构造时由 ``System`` 提供质量比与归一化尺度（若有），并由 ``LibrationPoint``
    选择平动点；其余 qiao 归一化常量（LU、TU、mu_e/m/s、JD0、基础频率等）
    采用固化值，与 qiao ``Global_File.py`` 保持一致。

    Attributes:
        system: 关联的 e2m2e ``System``（一般 ``CR3BP_System``）。
        libration_point: 选定平动点。
        epoch: 参考历元儒略日。
        order: 法型展开阶数。
        LU: 归一化长度（km）。
        TU: 归一化时间（s）。
        VU: 归一化速度（km/s），由 ``LU/TU`` 推导。
        mu: 系统质量比。
        mu_e: 归一化地球引力常数（无量纲）。
        mu_m: 归一化月球引力常数（无量纲）。
        mu_s: 归一化太阳引力常数（无量纲）。
        jd0: 参考历元儒略日。
        libration_position: 平动点在无量纲会合系下的 ``(3,)`` 坐标。
        gamma: 共线平动点的 γ 值；L4/L5 为 ``None``。
        base_frequencies: 基础频率 ``(omega_1, ..., omega_4)``。
        central_frequencies: 中心流形频率 ``(nu_1, nu_2)``。
        characteristic_exponent: 特征指数 λ。
    """

    def __init__(
        self,
        system: System,
        libration_point: LibrationPoint,
        epoch: float | datetime,
        order: int,
        *,
        LU: float | None = None,
        TU: float | None = None,
        mu: float | None = None,
        mu_e: float | None = None,
        mu_m: float | None = None,
        mu_s: float | None = None,
        frequency_scale: float = 1.0,
    ) -> None:
        """构造上下文。

        Args:
            system: 关联 ``System``。``CR3BP_System`` 提供 ``mu``；
                其他系统如不显式传入 ``mu`` 参数则取 ``0.0``。
            libration_point: 平动点枚举（L1–L5）。
            epoch: 历元，可为儒略日 ``float`` 或 ``datetime``。
            order: 法型展开阶数，必须为正整数。
            LU: 覆盖默认 LU（km）。仅供测试/非地月系统使用。
            TU: 覆盖默认 TU（s）。仅供测试/非地月系统使用。
            mu: 覆盖系统/默认质量比。
            mu_e: 覆盖默认地球引力常数。
            mu_m: 覆盖默认月球引力常数。
            mu_s: 覆盖默认太阳引力常数。

        Raises:
            ValueError: ``order`` 非正整数；``libration_point`` 非法。
        """
        if not isinstance(order, int) or order <= 0:
            raise ValueError(f"order 必须为正整数，得到 {order!r}")

        # 历元归一化为儒略日 float
        if hasattr(epoch, "timestamp"):  # datetime 兼容
            # datetime.timestamp 使用 POSIX 历元；转儒略日需 +2440587.5
            epoch_jd: float = float(epoch.timestamp()) / 86400.0 + 2440587.5
        else:
            epoch_jd = float(epoch)

        self.system: System = system
        self.libration_point: LibrationPoint = librationPoint_normalize(libration_point)
        self.epoch: float = epoch_jd
        self.order: int = order

        # 归一化常量（默认取 qiao 值；显式覆盖优先）
        self.LU: float = float(LU) if LU is not None else LU_KM
        self.TU: float = float(TU) if TU is not None else TU_S
        self.VU: float = self.LU / self.TU

        if mu is not None:
            self.mu: float = float(mu)
        else:
            self.mu = _system_mu(system, fallback=MU)

        self.mu_e: float = float(mu_e) if mu_e is not None else MU_E
        self.mu_m: float = float(mu_m) if mu_m is not None else MU_M
        self.mu_s: float = float(mu_s) if mu_s is not None else MU_S
        self.jd0: float = JD0_J2000

        # 平动点几何
        if self.libration_point in (LibrationPoint.L4, LibrationPoint.L5):
            self.gamma: float | None = None
        else:
            self.gamma = libration_gamma(self.libration_point)
        self.libration_position: npt.NDArray[np.floating] = compute_libration_position(
            self.libration_point, self.mu, gamma=self.gamma
        )

        # 频率体系（可选 frequency_scale 缩放，用于 γ 缩放坐标：t'=t/γ^{3/2}
        # 使频率变为 ω·γ^{3/2}，让 Hamiltonian 高阶系数 c_n=O(1)）
        self.frequency_scale: float = float(frequency_scale)
        self.base_frequencies: npt.NDArray[np.floating] = (
            np.array(BASE_FREQUENCIES, dtype=float) * self.frequency_scale
        )
        nu1, nu2 = central_frequencies(self.libration_point)
        self.central_frequencies: tuple[float, float] = (
            nu1 * self.frequency_scale,
            nu2 * self.frequency_scale,
        )
        self.characteristic_exponent: float = (
            characteristic_exponent(self.libration_point) * self.frequency_scale
        )

    # ------------------------------------------------------------------
    # 时间转换
    # ------------------------------------------------------------------

    def seconds_to_tu(self, t_seconds: float) -> float:
        """SI 秒 → 归一化 TU。"""
        return float(t_seconds) / self.TU

    def tu_to_seconds(self, t_tu: float) -> float:
        """归一化 TU → SI 秒。"""
        return float(t_tu) * self.TU

    # ------------------------------------------------------------------
    # 表示
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"NormalFormContext(point={self.libration_point.name}, order={self.order}, "
            f"mu={self.mu:.6e}, LU={self.LU:.3f} km, TU={self.TU:.3f} s, "
            f"lambda={self.characteristic_exponent:.6f})"
        )

    def __str__(self) -> str:
        return (
            f"NormalFormContext[{self.libration_point.name}, order={self.order}, "
            f"LU={self.LU:.3f} km, TU={self.TU:.3f} s]"
        )


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _system_mu(system: System, fallback: float) -> float:
    """从 System 提取质量比。

    CR3BP_System 暴露 ``mu`` 属性；其他系统若未实现则回退到 ``fallback``。
    """
    mu = getattr(system, "mu", None)
    if isinstance(mu, (int, float)) and float(mu) > 0:
        return float(mu)
    return float(fallback)


def librationPoint_normalize(point: LibrationPoint | int | str) -> LibrationPoint:
    """把 ``int``/``str`` 形式的平动点规范化为 ``LibrationPoint`` 枚举。

    Args:
        point: 平动点，可为枚举、1–5 整数或 ``"L1"``–``"L5"`` 字符串。

    Returns:
        对应 ``LibrationPoint`` 枚举值。

    Raises:
        ValueError: 无法识别。
    """
    if isinstance(point, LibrationPoint):
        return point
    if isinstance(point, int):
        for lp in LibrationPoint:
            if lp.value == point:
                return lp
    if isinstance(point, str):
        try:
            return LibrationPoint[point.upper()]
        except KeyError:
            pass
    raise ValueError(f"无法识别的平动点: {point!r}")


__all__ = ["NormalFormContext"]
