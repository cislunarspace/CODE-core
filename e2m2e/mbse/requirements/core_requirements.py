"""核心层需求定义。

定义 e2m2e 的系统建模、动力学传播、轨道数据和坐标变换等基础需求。
需求 ID 范围：REQ-001 ~ REQ-030。
"""

from .base import Requirement, RequirementCategory, RequirementPriority

CORE_REQUIREMENTS = [
    Requirement(
        id="REQ-001",
        title="State vector ordering",
        category=RequirementCategory.INTERFACE,
        description="状态向量必须按 [x, y, z, vx, vy, vz] 顺序排列，即前 3 分量为位置，后 3 分量为速度。",  # noqa: E501
        priority=RequirementPriority.SHALL,
        verification_method="test",
        linked_code=["e2m2e.data.types.orbit", "e2m2e.algorithm.dynamics.dynamics"],
        linked_tests=[
            "tests/data/types/test_orbit.py",
            "tests/algorithm/dynamics/test_dynamics_contract.py",
        ],
    ),
    Requirement(
        id="REQ-002",
        title="Shape of propagated states",
        category=RequirementCategory.INTERFACE,
        description="所有 Dynamics 子类的 propagate() 方法返回的 states 必须为 (n_points, 6) 形状。",  # noqa: E501
        priority=RequirementPriority.SHALL,
        verification_method="test",
        linked_code=[
            "e2m2e.algorithm.dynamics.dynamics",
            "e2m2e.algorithm.dynamics.ephemeris_dynamics",
        ],
        linked_tests=[
            "tests/algorithm/dynamics/test_dynamics_contract.py",
            "tests/algorithm/dynamics/test_ephemeris_dynamics_legacy.py",
        ],
    ),
    Requirement(
        id="REQ-003",
        title="Jacobi constant drift tolerance",
        category=RequirementCategory.PERFORMANCE,
        description="沿一个轨道周期积分后，Jacobi 常数最大漂移不超过 1e-10。",
        priority=RequirementPriority.SHALL,
        verification_method="test",
        linked_code=["e2m2e.algorithm.dynamics.dynamics"],
        linked_tests=["tests/algorithm/dynamics/test_cr3bp_model.py"],
    ),
    Requirement(
        id="REQ-004",
        title="Analytic Jacobian for the STM",
        category=RequirementCategory.FUNCTIONAL,
        description=(
            "状态转移矩阵必须通过解析 Jacobian（compute_jacobian_A）计算；"
            "解析 Jacobian 应与运动方程的有限差分一致。"
        ),
        priority=RequirementPriority.SHALL,
        verification_method="test",
        linked_code=["e2m2e.algorithm.dynamics.dynamics"],
        linked_tests=["tests/algorithm/dynamics/test_cr3bp_variational.py"],
    ),
    Requirement(
        id="REQ-005",
        title="Dynamics subclasses call super().__init__()",
        category=RequirementCategory.INTERFACE,
        description="所有 Dynamics 子类必须调用 super().__init__()，确保基类属性正确初始化。",
        priority=RequirementPriority.SHALL,
        verification_method="inspection",
        linked_code=[
            "e2m2e.algorithm.dynamics.dynamics",
            "e2m2e.algorithm.dynamics.ephemeris_dynamics",
        ],
        linked_tests=["tests/algorithm/dynamics/test_ephemeris_dynamics_legacy.py"],
    ),
    Requirement(
        id="REQ-006",
        title="Coordinate transforms are mutually inverse",
        category=RequirementCategory.FUNCTIONAL,
        description="坐标变换与其逆变换的组合必须恢复原始坐标（在数值精度范围内）。",
        priority=RequirementPriority.SHALL,
        verification_method="test",
        linked_code=["e2m2e.algorithm.coordinate.coordinate_system", "e2m2e.algorithm.coordinate"],
        linked_tests=["tests/algorithm/coordinate/test_synodic_j2000.py"],
    ),
    Requirement(
        id="REQ-010",
        title="Libration point position accuracy",
        category=RequirementCategory.PERFORMANCE,
        description="平动点 L1-L5 必须满足对应的 CR3BP 平衡方程与三角几何关系。",
        priority=RequirementPriority.SHALL,
        verification_method="test",
        linked_code=["e2m2e.algorithm.dynamics.cr3bp_system"],
        linked_tests=["tests/algorithm/dynamics/test_cr3bp_system.py"],
    ),
    Requirement(
        id="REQ-011",
        title="Characteristic scales precondition",
        category=RequirementCategory.FUNCTIONAL,
        description="在物理单位转换前必须调用 set_characteristic_scales() 设置特征尺度。",
        priority=RequirementPriority.SHOULD,
        verification_method="test",
        linked_code=["e2m2e.algorithm.dynamics.cr3bp_system"],
        linked_tests=["tests/algorithm/dynamics/test_cr3bp_system.py"],
    ),
    Requirement(
        id="REQ-012",
        title="Default integration tolerances",
        category=RequirementCategory.PERFORMANCE,
        description="数值积分的相对容差和绝对容差默认值均为 1e-12（双精度机器精度量级）。",
        priority=RequirementPriority.SHALL,
        verification_method="inspection",
        linked_code=["e2m2e.algorithm.dynamics.dynamics"],
        linked_tests=["tests/algorithm/dynamics/test_dynamics_contract.py"],
    ),
    Requirement(
        id="REQ-020",
        title="Orbit serialization compatibility",
        category=RequirementCategory.INTERFACE,
        description="Orbit.save_to_file / load_from_file 必须兼容 v3 JSON 格式。",
        priority=RequirementPriority.SHALL,
        verification_method="test",
        linked_code=["e2m2e.data.types.orbit"],
        linked_tests=["tests/data/types/test_orbit_io.py"],
    ),
    Requirement(
        id="REQ-021",
        title="Orbit period estimation",
        category=RequirementCategory.FUNCTIONAL,
        description="Orbit 在初始化时自动通过零交叉检测估计轨道周期。",
        priority=RequirementPriority.SHOULD,
        verification_method="test",
        linked_code=["e2m2e.data.types.orbit"],
        linked_tests=["tests/data/types/test_orbit.py"],
    ),
    Requirement(
        id="REQ-022",
        title="OrbitFamily aggregation",
        category=RequirementCategory.FUNCTIONAL,
        description="OrbitFamily 提供统一的聚合接口：states、periods 属性与 get_jacobi_constants() 方法。",  # noqa: E501
        priority=RequirementPriority.SHALL,
        verification_method="test",
        linked_code=["e2m2e.data.types.orbit"],
        linked_tests=["tests/data/types/test_orbit_family.py"],
    ),
    Requirement(
        id="REQ-025",
        title="EphemerisDynamics unified interface (legacy)",
        category=RequirementCategory.INTERFACE,
        description=(
            "EphemerisDynamics 必须与 CR3BP_Dynamics 共享 Dynamics 基类接口。"
            "（遗留实现，仅供 multiple_shooting 内部使用；新代码用 ForceModel。）"
        ),
        priority=RequirementPriority.SHALL,
        verification_method="test",
        linked_code=["e2m2e.algorithm.dynamics.ephemeris_dynamics"],
        linked_tests=["tests/algorithm/dynamics/test_ephemeris_dynamics_legacy.py"],
    ),
    Requirement(
        id="REQ-026",
        title="EphemerisDynamics adaptive step size (legacy)",
        category=RequirementCategory.FUNCTIONAL,
        description=(
            "EphemerisDynamics 根据传播时长自适应调整最大步长"
            "（max_step = min(60s, duration/10)）。"
            "（遗留实现，新代码用 ForceModel。）"
        ),
        priority=RequirementPriority.SHOULD,
        verification_method="test",
        linked_code=["e2m2e.algorithm.dynamics.ephemeris_dynamics"],
        linked_tests=["tests/algorithm/dynamics/test_ephemeris_dynamics_legacy.py"],
    ),
]
