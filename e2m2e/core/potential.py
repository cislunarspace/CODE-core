"""CR3BP 伪势能函数

包含伪势能 Ω 的 Hessian 矩阵计算，供动力学方程和稳定性分析共用。
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


_MIN_DISTANCE = 1e-10


def pseudo_potential_hessian(
    mu: float,
    x: float,
    y: float,
    z: float,
    min_distance: float = _MIN_DISTANCE,
) -> npt.NDArray[np.float64]:
    """计算 CR3BP 伪势能 Ω 的 Hessian 矩阵 U_ij = ∂²Ω/∂rᵢ∂rⱼ

    Args:
        mu: 质量参数
        x: 旋转系 x 坐标（无量纲）
        y: 旋转系 y 坐标（无量纲）
        z: 旋转系 z 坐标（无量纲）
        min_distance: 最小距离钳位，防止在天体位置处除零

    Returns:
        3×3 对称 Hessian 矩阵
    """
    r1 = max(np.sqrt((x + mu) ** 2 + y**2 + z**2), min_distance)
    r2 = max(np.sqrt((x - 1 + mu) ** 2 + y**2 + z**2), min_distance)

    # Hessian 元素由伪势能 Ω = (x²+y²)/2 + (1-μ)/r₁ + μ/r₂ 对各坐标求二阶偏导得到。
    # 以引力项为例，∂²/∂x²[(1-μ)/r₁] = (1-μ)·(3(x+μ)²/r₁⁵ - 1/r₁³)，
    # 离心力项 ∂²/∂x²[(x²+y²)/2] = 1，组合即得 U_xx。
    # z 方向无离心力贡献，故 U_zz 缺少前导常数项 1。
    U_xx = (
        1
        - (1 - mu) * (1 / r1**3 - 3 * (x + mu) ** 2 / r1**5)
        - mu * (1 / r2**3 - 3 * (x - 1 + mu) ** 2 / r2**5)
    )
    U_yy = (
        1
        - (1 - mu) * (1 / r1**3 - 3 * y**2 / r1**5)
        - mu * (1 / r2**3 - 3 * y**2 / r2**5)
    )
    U_zz = -(1 - mu) * (1 / r1**3 - 3 * z**2 / r1**5) - mu * (1 / r2**3 - 3 * z**2 / r2**5)
    # 混合偏导 ∂²Ω/∂x∂y：离心力项 ∂²/∂x∂y[(x²+y²)/2] = 0，仅保留引力交叉项
    U_xy = 3 * (1 - mu) * (x + mu) * y / r1**5 + 3 * mu * (x - 1 + mu) * y / r2**5
    U_xz = 3 * (1 - mu) * (x + mu) * z / r1**5 + 3 * mu * (x - 1 + mu) * z / r2**5
    U_yz = 3 * (1 - mu) * y * z / r1**5 + 3 * mu * y * z / r2**5

    return np.array(
        [
            [U_xx, U_xy, U_xz],
            [U_xy, U_yy, U_yz],
            [U_xz, U_yz, U_zz],
        ]
    )
