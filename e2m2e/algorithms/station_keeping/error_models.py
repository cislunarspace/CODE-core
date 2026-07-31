"""站保误差模型：采样、测定轨扰动、分段控制误差、光压弧段误差。

算法依据《控制方案.md》（hybrid_auto 版）§1.5：

- §1.5.1 轨道误差仿真：式 5.37-5.39 的测定轨高斯扰动（位置/速度 1-sigma
  可配），标准正态样本用式 5.37 下方的 Marsaglia 极坐标 Box-Muller 构造
- §1.5.2 控制误差仿真：式 5.40 分段模型——``Δv < Δv_min`` 不开机；
  ``Δv_min ≤ Δv < Δv_mid`` 绝对误差（大小 1-sigma + 球面角 α/β 误差）；
  ``Δv_mid ≤ Δv ≤ Δv_max`` 相对误差；``Δv > Δv_max`` 判控制失败
- §1.5.3 力模型误差仿真：真实轨道与控制轨道的光压差异每控制弧段随机
  生成、弧段内固定（表 5-3 脚注）

单位约定：轨道状态为 GCRS（位置 km、速度 km/s），与传播器一致；误差
参数沿用 DFH 惯例（位置精度 m、速度精度 m/s、脉冲大小 m/s）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    "BoxMullerSampler",
    "NavigationErrorModel",
    "ThrustExecutionError",
    "SrpErrorModel",
]


class BoxMullerSampler:
    """Marsaglia 极坐标 Box-Muller 标准正态采样器（§1.5.1 构造）。

    ``u1, u2 ∈ (0,1)`` → ``v = 2u - 1``，``r² = v1² + v2² < 1`` 时
    ``z1 = v1·sqrt(-2·ln(r²)/r²)``、``z2 = v2·sqrt(-2·ln(r²)/r²)`` 为
    两个独立标准正态数。可设种子复现（蒙特卡洛回归要求同种子同结果）。
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def standard_normal(self, size: int = 1) -> npt.NDArray[np.floating]:
        """生成 ``size`` 个独立标准正态样本。"""
        out = np.empty(size, dtype=float)
        filled = 0
        while filled < size:
            n = size - filled
            v = self._rng.uniform(0.0, 1.0, size=(2 * n,)) * 2.0 - 1.0
            v1, v2 = v[::2], v[1::2]
            r2 = v1 * v1 + v2 * v2
            ok = r2 < 1.0
            if not np.any(ok):
                continue
            scale = np.sqrt(-2.0 * np.log(r2[ok]) / r2[ok])
            z1 = v1[ok] * scale
            z2 = v2[ok] * scale
            take = min(ok.sum(), n)
            out[filled : filled + take] = np.concatenate([z1, z2])[:take]
            filled += take
        return out


@dataclass
class NavigationErrorModel:
    """测定轨误差模型（§1.5.1）。

    Attributes:
        position_sigma_m: 位置 1-sigma（m），DFH 默认 1500
        velocity_sigma_mps: 速度 1-sigma（m/s），DFH 默认 0.002
    """

    position_sigma_m: float = 1500.0
    velocity_sigma_mps: float = 0.002

    def perturb(
        self, state: npt.ArrayLike, sampler: BoxMullerSampler
    ) -> npt.NDArray[np.floating]:
        """叠加一次测定轨高斯扰动（式 5.37/5.38/5.39）。

        Args:
            state: GCRS 状态 ``[x, y, z, vx, vy, vz]``（km, km/s）
            sampler: 标准正态采样器

        Returns:
            扰动后的状态（km, km/s）
        """
        state = np.asarray(state, dtype=float)
        z = sampler.standard_normal(6)
        dz = np.empty(6)
        dz[:3] = z[:3] * (self.position_sigma_m / 1000.0)
        dz[3:] = z[3:] * (self.velocity_sigma_mps / 1000.0)
        return state + dz


@dataclass
class ThrustExecutionError:
    """轨控发动机执行误差模型（式 5.40 分段）。

    Attributes:
        dv_min: 最小开机速度增量（m/s）；``Δv < dv_min`` 完全不开机
        dv_mid: 中点值（m/s）；小量段（绝对误差）与大量段（相对误差）分界
        dv_max: 最大开机速度增量（m/s）；``Δv > dv_max`` 判控制失败
        abs_sigma_mps: 小量段大小 1-sigma（m/s，DFH 默认 0.033）
        rel_sigma: 大量段相对 1-sigma（DFH 默认 0.003）
        angle_sigma_deg: 推力方向球面角 1-sigma（deg，DFH 默认 0.333）
    """

    dv_min: float = 0.1
    dv_mid: float = 10.0
    dv_max: float = 100.0
    abs_sigma_mps: float = 0.033
    rel_sigma: float = 0.003
    angle_sigma_deg: float = 0.333

    def apply(
        self, dv_c: npt.ArrayLike, sampler: BoxMullerSampler
    ) -> tuple[npt.NDArray[np.floating] | None, bool]:
        """施加执行误差，返回（实际 Δv 矢量 m/s，是否控制失败）。

        理论控制量 ``Δv_c`` 为 3 矢量（m/s）。结果约定：

        - 不开机（``|Δv_c| < dv_min``）：返回 ``None``（零矢量），非失败
        - 绝对误差段：大小 ``Δv + N(0, abs_sigma)``、方向角加高斯误差
        - 相对误差段：大小 ``Δv·(1 + N(0, rel_sigma))``、方向角加高斯误差
        - 失败（``|Δv_c| > dv_max``）：返回 ``(None, True)``

        方向误差按式 5.40 的球面角模型：``α``（方位角）与 ``β``（仰角）
        各加高斯误差，实际方向为 (cosβ'cosα', cosβ'sinα', sinβ')。
        """
        dv_c = np.asarray(dv_c, dtype=float)
        mag = float(np.linalg.norm(dv_c))
        if mag < self.dv_min:
            return None, False
        if mag > self.dv_max:
            return None, True

        z = sampler.standard_normal(3)
        # 理论方向球面角（β 仰角、α 方位角）
        beta = np.arcsin(np.clip(dv_c[2] / mag, -1.0, 1.0))
        alpha = np.arctan2(dv_c[1], dv_c[0])
        ang_sigma = np.deg2rad(self.angle_sigma_deg)
        beta_p = beta + ang_sigma * z[1]
        alpha_p = alpha + ang_sigma * z[2]

        if mag < self.dv_mid:
            # 绝对误差段：大小 = Δv + N(0, abs_sigma)
            mag_p = mag + self.abs_sigma_mps * z[0]
        else:
            # 相对误差段：大小 = Δv·(1 + N(0, rel_sigma))
            mag_p = mag * (1.0 + self.rel_sigma * z[0])

        dv_r = np.array(
            [
                mag_p * np.cos(beta_p) * np.cos(alpha_p),
                mag_p * np.cos(beta_p) * np.sin(alpha_p),
                mag_p * np.sin(beta_p),
            ]
        )
        return dv_r, False


@dataclass
class SrpErrorModel:
    """光压弧段随机误差（表 5-3 脚注）。

    真实轨道与控制轨道的光压模型差异每控制弧段随机生成、弧段内固定。
    实现为真实轨道光压系数乘子 ``1 + error_level·z``，每弧段抽一次。
    """

    error_level: float = 0.10

    def sample_cr_scale(self, sampler: BoxMullerSampler) -> float:
        """抽样当前弧段的光压系数乘子。"""
        return 1.0 + self.error_level * float(sampler.standard_normal(1)[0])
