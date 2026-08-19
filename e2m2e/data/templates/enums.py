"""领域枚举：轨道族类型、参考系、单位系统等。

枚举是数据，归 data/templates/（ADR 0011 迁移，源：``core/enums.py`` +
``mbse/data/enums.py``）。算法层/接口层引用此处；旧路径已删除。
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


class LibrationPoint(enum.Enum):
    """CR3BP 模型的五个平动点。"""

    L1 = 1
    L2 = 2
    L3 = 3
    L4 = 4
    L5 = 5


class ProjectionPlane(enum.Enum):
    """投影平面"""

    XY = "xy"
    XZ = "xz"
    YZ = "yz"


class TransferType(enum.Enum):
    """转移类型"""

    DIRECT = "direct"
    LGA = "lga"  # Lunar Gravity Assist
    WSB = "wsb"  # Weak Stability Boundary (sun-perturbed indirect transfer)
    EXTERNAL = "external"
    UNKNOWN = "unknown"  # 传播失败/轨迹为空，无法分类（#352，不再假 DIRECT）


class BoundaryMode(enum.Enum):
    """两层多重打靶的边界条件。"""

    FIXED_ENDPOINTS = "fixed_endpoints"


class ConvergenceState(enum.Enum):
    """算法最终状态。"""

    ITERATING = "iterating"
    CONVERGED = "converged"
    DIVERGED = "diverged"
    STAGNATED = "stagnated"
    MAX_ITERATIONS = "max_iterations"
    INFEASIBLE = "infeasible"
    COLLISION = "collision"
    FAILED = "failed"


class FailureCause(enum.Enum):
    """算法最终结局的稳定原因码。"""

    NONE = "none"
    INTEGRATION_FAILED = "integration_failed"
    SINGULAR_JACOBIAN = "singular_jacobian"
    INVALID_PERIOD = "invalid_period"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"
    STAGNATION_DETECTED = "stagnation_detected"
    DIVERGENCE_DETECTED = "divergence_detected"
    NO_INTERSECTION = "no_intersection"
    CONSTRAINT_VIOLATION = "constraint_violation"
    BODY_COLLISION = "body_collision"
    LEVEL1_CORRECTION_FAILED = "level1_correction_failed"
    BACKEND_FAILURE = "backend_failure"
    INVALID_INPUT = "invalid_input"
    UNKNOWN = "unknown"


class OrbitFamilyType(enum.Enum):
    """轨道族类型"""

    HALO = "halo"
    LYAPUNOV = "lyapunov"
    VERTICAL = "vertical"  # 待实现
    AXIAL = "axial"
    BUTTERFLY = "butterfly"  # 待实现
    DRAGONFLY = "dragonfly"  # 待实现
    DRO = "dro"  # Distant Retrograde Orbit
    DPO = "dpo"  # Direct Prograde Orbit
    SPO = "spo"  # Short Period Orbit
    LPO = "lpo"  # Long Period Orbit
    TADPOLE = "tadpole"  # 待实现
    HORSESHOE = "horseshoe"
    RO = "ro"  # Resonant Orbit
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


class TransferPhase(enum.Enum):
    """转移设计阶段（用于状态机图）"""

    CONFIGURED = "configured"
    SEARCHING = "searching"
    CANDIDATES_FOUND = "candidates_found"
    OPTIMIZING = "optimizing"
    COMPLETE = "complete"
    FAILED = "failed"
