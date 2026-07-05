"""共享领域枚举

存放被 core / algorithms / transfer / visualization 等多层引用的基础领域枚举。
仅 mbse 内部使用的追溯性枚举仍保留在 ``e2m2e.mbse.data.enums``。

此处定义的枚举也经 ``e2m2e.mbse.data.enums`` 重导出，以保持向后兼容。
"""

from __future__ import annotations

import enum


class ReferenceFrame(enum.Enum):
    """参考坐标系"""

    ROTATING = "rotating"  # CR3BP 旋转坐标系
    INERTIAL = "inertial"  # 惯性坐标系
    BARYCENTRIC = "barycentric"  # 质心坐标系
    PRIMARY_CENTERED = "primary_centered"  # 主天体中心坐标系
    SECONDARY_CENTERED = "secondary_centered"  # 次天体中心坐标系
    SYNODIC = "synodic"  # 会合坐标系
    J2000 = "J2000"  # J2000 惯性系


class UnitSystem(enum.Enum):
    """单位系统"""

    DIMENSIONLESS = "dimensionless"  # 无量纲单位（如 CR3BP）
    SI = "si"  # 国际单位制（km, s, km/s）


class ProjectionPlane(enum.Enum):
    """投影平面"""

    XY = "xy"
    XZ = "xz"
    YZ = "yz"


class TransferType(enum.Enum):
    """转移类型"""

    DIRECT = "direct"
    LGA = "lga"  # Lunar Gravity Assist
    EXTERNAL = "external"


class BoundaryMode(enum.Enum):
    """两层多重打靶的边界条件。"""

    FIXED_ENDPOINTS = "fixed_endpoints"


class TwoLevelMultipleShootingStatus(enum.Enum):
    """两层多重打靶的结果状态。"""

    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    LEVEL1_FAILED = "level1_failed"


class ConvergenceState(enum.Enum):
    """算法收敛状态（用于状态机图）"""

    ITERATING = "iterating"
    CONVERGED = "converged"
    DIVERGED = "diverged"
    STAGNATED = "stagnated"
    MAX_ITERATIONS = "max_iterations"
