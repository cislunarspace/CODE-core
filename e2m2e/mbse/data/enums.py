"""共享枚举定义

统一管理 e2m2e 系统中所有枚举类型，替代分散在各个模块中的局部枚举。
"""

from __future__ import annotations

import enum


class OrbitFamilyType(enum.Enum):
    """轨道族类型"""

    HALO = "halo"
    LYAPUNOV = "lyapunov"
    VERTICAL = "vertical"
    AXIAL = "axial"
    BUTTERFLY = "butterfly"
    DRAGONFLY = "dragonfly"
    DRO = "dro"  # Distant Retrograde Orbit
    NRHO = "nrho"  # Near Rectilinear Halo Orbit
    LYO = "lyo"  # Lissajous Orbit


class StabilityLabel(enum.Enum):
    """轨道稳定性标签"""

    STABLE = "stable"
    UNSTABLE = "unstable"
    MARGINALLY_STABLE = "marginally_stable"
    HYPERBOLIC = "hyperbolic"
    ELLIPTIC = "elliptic"
    PARABOLIC = "parabolic"


class BifurcationLabel(enum.Enum):
    """分岔类型标签"""

    NONE = "none"
    PERIOD_DOUBLING = "period_doubling"
    SADDLE_NODE = "saddle_node"
    TORUS = "torus"
    PITCHFORK = "pitchfork"
    TRANSCRITICAL = "transcritical"
    SECONDARY_HOPF = "secondary_hopf"


class ConvergenceState(enum.Enum):
    """算法收敛状态（用于状态机图）"""

    ITERATING = "iterating"
    CONVERGED = "converged"
    DIVERGED = "diverged"
    STAGNATED = "stagnated"
    MAX_ITERATIONS = "max_iterations"


class TransferPhase(enum.Enum):
    """转移设计阶段（用于状态机图）"""

    CONFIGURED = "configured"
    SEARCHING = "searching"
    CANDIDATES_FOUND = "candidates_found"
    OPTIMIZING = "optimizing"
    COMPLETE = "complete"
    FAILED = "failed"


class ReferenceFrame(enum.Enum):
    """参考坐标系"""

    ROTATING = "rotating"  # CR3BP 旋转坐标系
    INERTIAL = "inertial"  # 惯性坐标系
    BARYCENTRIC = "barycentric"  # 质心坐标系
    PRIMARY_CENTERED = "primary_centered"  # 主天体中心坐标系
    SECONDARY_CENTERED = "secondary_centered"  # 次天体中心坐标系
    SYNODIC = "synodic"  # 会合坐标系
    J2000 = "j2000"  # J2000 惯性系


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


class UnitSystem(enum.Enum):
    """单位系统"""

    DIMENSIONLESS = "dimensionless"  # 无量纲单位（如 CR3BP）
    SI = "si"  # 国际单位制（km, s, km/s）
