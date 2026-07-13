"""广义标准形化简上下文对象 ``GeneralizedNormalFormContext``。

扩展 ``NormalFormContext`` 以支持任意参考中心（周期轨道），
不再局限于平动点。这是"分区正规化 + 全局拼接"理论框架的基础。

核心创新：
1. 参考中心可以是任意周期轨道（不限于平动点）
2. 谱结构自适应（支持 0 双曲、1 双曲、2 双曲等）
3. 标准形矩阵根据实际谱结构构造

与 ``NormalFormContext`` 的关系：
- ``GeneralizedNormalFormContext`` 是 ``NormalFormContext`` 的推广
- 当参考中心是平动点时，退化为 ``NormalFormContext``
- 保持向后兼容，现有代码继续工作
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from e2m2e.core import System

from .constants import (
    BASE_FREQUENCIES,
    JD0_J2000,
    LU_KM,
    MU,
    MU_E,
    MU_M,
    MU_S,
    TU_S,
)

if TYPE_CHECKING:
    from datetime import datetime


# ---------------------------------------------------------------------------
# 谱结构分类
# ---------------------------------------------------------------------------


class SpectralType(Enum):
    """Monodromy 矩阵的谱结构类型。"""

    ONE_HYPERBOLIC_TWO_CENTER = "1H+2C"  # 共线平动点（现有方法）
    THREE_CENTER = "0H+3C"  # DRO（无双曲方向）
    TWO_HYPERBOLIC_ONE_CENTER = "2H+1C"  # 共振轨道
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SpectralStructure:
    """Monodromy 矩阵的谱结构信息。

    Attributes:
        spectral_type: 谱结构类型
        n_hyperbolic: 双曲方向数量
        n_center: 中心方向数量
        hyperbolic_exponents: 双曲特征指数（正值，按降序排列）
        center_frequencies: 中心频率（正值，按降序排列）
        eigenvalues: 原始特征值
    """

    spectral_type: SpectralType
    n_hyperbolic: int
    n_center: int
    hyperbolic_exponents: npt.NDArray[np.floating] = field(default_factory=lambda: np.array([]))
    center_frequencies: npt.NDArray[np.floating] = field(default_factory=lambda: np.array([]))
    eigenvalues: npt.NDArray[np.floating] = field(default_factory=lambda: np.array([]))


def classify_spectrum(
    monodromy_matrix: npt.NDArray[np.floating],
    tol: float = 1e-10,
) -> SpectralStructure:
    """分类 Monodromy 矩阵的谱结构。

    Args:
        monodromy_matrix: Monodromy 矩阵 (6, 6)
        tol: 特征值分类的容差

    Returns:
        SpectralStructure 对象
    """
    eigenvalues = np.linalg.eigvals(monodromy_matrix)

    hyperbolic = []
    center = []

    for lam in eigenvalues:
        mag = abs(lam)
        if abs(mag - 1.0) > tol:
            # 双曲方向
            if mag > 1.0:
                hyperbolic.append(np.log(mag))
        else:
            # 中心方向
            if np.imag(lam) > tol:
                center.append(np.abs(np.angle(lam)))

    n_hyp = len(hyperbolic)
    n_center = len(center)

    # 分类
    if n_hyp == 1 and n_center == 2:
        spectral_type = SpectralType.ONE_HYPERBOLIC_TWO_CENTER
    elif n_hyp == 0 and n_center == 3:
        spectral_type = SpectralType.THREE_CENTER
    elif n_hyp == 2 and n_center == 1:
        spectral_type = SpectralType.TWO_HYPERBOLIC_ONE_CENTER
    else:
        spectral_type = SpectralType.UNKNOWN

    return SpectralStructure(
        spectral_type=spectral_type,
        n_hyperbolic=n_hyp,
        n_center=n_center,
        hyperbolic_exponents=np.sort(np.array(hyperbolic))[::-1],  # 降序
        center_frequencies=np.sort(np.array(center))[::-1],  # 降序
        eigenvalues=eigenvalues,
    )


# ---------------------------------------------------------------------------
# 参考轨道
# ---------------------------------------------------------------------------


@dataclass
class ReferenceOrbit:
    """参考轨道（可以是周期轨道或拟周期轨道）。

    Attributes:
        times: 时间数组 (n,)
        states: 状态数组 (n, 6)
        period: 轨道周期（如果是周期轨道）
        is_periodic: 是否是周期轨道
        monodromy_matrix: Monodromy 矩阵（如果是周期轨道）
        spectral_structure: 谱结构信息
    """

    times: npt.NDArray[np.floating]
    states: npt.NDArray[np.floating]
    period: float | None = None
    is_periodic: bool = False
    monodromy_matrix: npt.NDArray[np.floating] | None = None
    spectral_structure: SpectralStructure | None = None

    def __post_init__(self) -> None:
        """验证输入。"""
        if self.times.ndim != 1:
            raise ValueError(f"times 必须是一维数组，得到 {self.times.ndim} 维")
        if self.states.ndim != 2 or self.states.shape[1] != 6:
            raise ValueError(f"states 必须是 (n, 6) 数组，得到 {self.states.shape}")
        if len(self.times) != len(self.states):
            raise ValueError(
                f"times 和 states 长度不一致：{len(self.times)} vs {len(self.states)}"
            )

    @property
    def n_samples(self) -> int:
        """采样点数量。"""
        return len(self.times)

    def state_at(self, t: float) -> npt.NDArray[np.floating]:
        """在时刻 t 线性插值状态。

        Args:
            t: 时间

        Returns:
            状态 (6,)
        """
        return np.array(
            [float(np.interp(t, self.times, self.states[:, k])) for k in range(6)],
            dtype=float,
        )


# ---------------------------------------------------------------------------
# 广义上下文
# ---------------------------------------------------------------------------


class GeneralizedNormalFormContext:
    """广义标准形化简上下文。

    扩展 ``NormalFormContext`` 以支持任意参考中心（周期轨道）。

    Attributes:
        system: 关联的 e2m2e ``System``
        reference_orbit: 参考轨道（可以是周期轨道或平动点）
        epoch: 参考历元儒略日
        order: 法型展开阶数
        LU: 归一化长度（km）
        TU: 归一化时间（s）
        VU: 归一化速度（km/s）
        mu: 系统质量比
        mu_e: 归一化地球引力常数
        mu_m: 归一化月球引力常数
        mu_s: 归一化太阳引力常数
        jd0: 参考历元儒略日
        base_frequencies: 基础频率 (omega_1, ..., omega_4)
        spectral_structure: 谱结构信息
        n_hyperbolic: 双曲方向数量
        n_center: 中心方向数量
    """

    def __init__(
        self,
        system: System,
        reference_orbit: ReferenceOrbit,
        epoch: float | datetime,
        order: int,
        *,
        LU: float | None = None,
        TU: float | None = None,
        mu: float | None = None,
        mu_e: float | None = None,
        mu_m: float | None = None,
        mu_s: float | None = None,
    ) -> None:
        """构造广义上下文。

        Args:
            system: 关联 ``System``
            reference_orbit: 参考轨道（可以是周期轨道或平动点）
            epoch: 历元，可为儒略日 ``float`` 或 ``datetime``
            order: 法型展开阶数，必须为正整数
            LU: 覆盖默认 LU（km）
            TU: 覆盖默认 TU（s）
            mu: 覆盖系统/默认质量比
            mu_e: 覆盖默认地球引力常数
            mu_m: 覆盖默认月球引力常数
            mu_s: 覆盖默认太阳引力常数

        Raises:
            ValueError: ``order`` 非正整数
        """
        if not isinstance(order, int) or order <= 0:
            raise ValueError(f"order 必须为正整数，得到 {order!r}")

        # 历元归一化为儒略日 float
        if hasattr(epoch, "timestamp"):
            epoch_jd: float = float(epoch.timestamp()) / 86400.0 + 2440587.5
        else:
            epoch_jd = float(epoch)

        self.system: System = system
        self.reference_orbit: ReferenceOrbit = reference_orbit
        self.epoch: float = epoch_jd
        self.order: int = order

        # 归一化常量
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

        # 基础频率
        self.base_frequencies: npt.NDArray[np.floating] = np.array(BASE_FREQUENCIES, dtype=float)

        # 谱结构
        if reference_orbit.spectral_structure is not None:
            self.spectral_structure: SpectralStructure = reference_orbit.spectral_structure
        elif reference_orbit.monodromy_matrix is not None:
            self.spectral_structure = classify_spectrum(reference_orbit.monodromy_matrix)
        else:
            # 默认假设 1 双曲 + 2 中心（向后兼容）
            self.spectral_structure = SpectralStructure(
                spectral_type=SpectralType.ONE_HYPERBOLIC_TWO_CENTER,
                n_hyperbolic=1,
                n_center=2,
            )

        self.n_hyperbolic: int = self.spectral_structure.n_hyperbolic
        self.n_center: int = self.spectral_structure.n_center

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
    # 标准形矩阵
    # ------------------------------------------------------------------

    def build_normal_form_matrix(self) -> npt.NDArray[np.floating]:
        """根据谱结构构造标准形矩阵 D。

        Returns:
            (6, 6) 标准形矩阵
        """
        D = np.zeros((6, 6), dtype=float)

        if self.spectral_structure.spectral_type == SpectralType.ONE_HYPERBOLIC_TWO_CENTER:
            # 共线平动点：1 双曲 + 2 中心
            lam = self.spectral_structure.hyperbolic_exponents[0]
            wp, wv = self.spectral_structure.center_frequencies[:2]
            D[0, 0] = lam
            D[3, 3] = -lam
            D[1, 4] = wp
            D[4, 1] = -wp
            D[2, 5] = wv
            D[5, 2] = -wv

        elif self.spectral_structure.spectral_type == SpectralType.THREE_CENTER:
            # DRO：0 双曲 + 3 中心
            w1, w2, w3 = self.spectral_structure.center_frequencies[:3]
            D[0, 1] = w1
            D[1, 0] = -w1
            D[2, 3] = w2
            D[3, 2] = -w2
            D[4, 5] = w3
            D[5, 4] = -w3

        elif self.spectral_structure.spectral_type == SpectralType.TWO_HYPERBOLIC_ONE_CENTER:
            # 共振轨道：2 双曲 + 1 中心
            lam1, lam2 = self.spectral_structure.hyperbolic_exponents[:2]
            w = self.spectral_structure.center_frequencies[0]
            D[0, 0] = lam1
            D[3, 3] = -lam1
            D[1, 1] = lam2
            D[4, 4] = -lam2
            D[2, 5] = w
            D[5, 2] = -w

        else:
            raise ValueError(
                f"未知的谱结构类型：{self.spectral_structure.spectral_type}"
            )

        return D

    # ------------------------------------------------------------------
    # 表示
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"GeneralizedNormalFormContext("
            f"type={self.spectral_structure.spectral_type.value}, "
            f"order={self.order}, "
            f"n_hyp={self.n_hyperbolic}, "
            f"n_center={self.n_center}, "
            f"mu={self.mu:.6e})"
        )

    def __str__(self) -> str:
        return (
            f"GeneralizedNormalFormContext["
            f"{self.spectral_structure.spectral_type.value}, "
            f"order={self.order}]"
        )


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _system_mu(system: System, fallback: float) -> float:
    """从 System 提取质量比。"""
    mu = getattr(system, "mu", None)
    if isinstance(mu, (int, float)) and float(mu) > 0:
        return float(mu)
    return float(fallback)


__all__ = [
    "GeneralizedNormalFormContext",
    "ReferenceOrbit",
    "SpectralStructure",
    "SpectralType",
    "classify_spectrum",
]
