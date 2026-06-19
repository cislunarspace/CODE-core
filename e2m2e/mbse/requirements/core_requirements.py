"""Core 层需求定义

定义 e2m2e 核心层（core/）的系统需求，覆盖系统建模、动力学传播、轨道数据、
坐标变换等基础功能。需求 ID 范围：REQ-001 ~ REQ-030。
"""

from .base import Requirement, RequirementCategory, RequirementPriority

CORE_REQUIREMENTS = [
    # ---- 状态向量与数据格式 ----
    Requirement(
        id="REQ-001",
        title="状态向量顺序",
        category=RequirementCategory.INTERFACE,
        description=(
            "状态向量必须按 [x, y, z, vx, vy, vz] 顺序排列，即前 3 分量为位置，后 3 分量为速度。"
        ),
        priority=RequirementPriority.SHALL,
        verification_method="test",
        linked_code=["e2m2e.core.orbit", "e2m2e.core.dynamics"],
        linked_tests=["tests/core/test_orbit.py", "tests/core/test_dynamics.py"],
    ),
    Requirement(
        id="REQ-002",
        title="传播结果 states 形状",
        category=RequirementCategory.INTERFACE,
        description=(
            "所有 Dynamics 子类的 propagate() 方法返回的 states 必须为 (n_points, 6) 形状。"
        ),
        priority=RequirementPriority.SHALL,
        verification_method="test",
        linked_code=["e2m2e.core.dynamics", "e2m2e.core.ephemeris_dynamics"],
        linked_tests=[
            "tests/core/test_dynamics.py",
            "tests/core/dynamics/test_ephemeris_dynamics.py",
        ],
    ),
    Requirement(
        id="REQ-003",
        title="Jacobi 常数漂移容限",
        category=RequirementCategory.PERFORMANCE,
        description="沿一个轨道周期积分后，Jacobi 常数最大漂移不超过 1e-10。",
        priority=RequirementPriority.SHALL,
        verification_method="test",
        linked_code=["e2m2e.core.dynamics"],
        linked_tests=["tests/core/test_dynamics.py"],
    ),
    Requirement(
        id="REQ-004",
        title="STM 解析 Jacobian",
        category=RequirementCategory.FUNCTIONAL,
        description=(
            "状态转移矩阵必须通过解析 Jacobian（compute_jacobian_A）计算，不得使用有限差分。"
        ),
        priority=RequirementPriority.SHALL,
        verification_method="analysis",
        linked_code=["e2m2e.core.dynamics"],
        linked_tests=["tests/core/test_dynamics.py"],
    ),
    # ---- 继承与接口 ----
    Requirement(
        id="REQ-005",
        title="Dynamics 子类调用 super().__init__()",
        category=RequirementCategory.INTERFACE,
        description="所有 Dynamics 子类必须调用 super().__init__()，确保基类属性正确初始化。",
        priority=RequirementPriority.SHALL,
        verification_method="inspection",
        linked_code=["e2m2e.core.dynamics", "e2m2e.core.ephemeris_dynamics"],
        linked_tests=["tests/core/dynamics/test_ephemeris_dynamics.py"],
    ),
    Requirement(
        id="REQ-006",
        title="坐标变换互逆一致",
        category=RequirementCategory.FUNCTIONAL,
        description="坐标变换与其逆变换的组合必须恢复原始坐标（在数值精度范围内）。",
        priority=RequirementPriority.SHALL,
        verification_method="test",
        linked_code=["e2m2e.core.coordinate"],
        linked_tests=["tests/core/test_coordinate.py", "tests/core/coordinate/test_coordinate.py"],
    ),
    # ---- 物理模型精度 ----
    Requirement(
        id="REQ-010",
        title="平动点位置精度",
        category=RequirementCategory.PERFORMANCE,
        description="平动点 L1-L5 的位置计算与解析解的误差须小于 1e-12。",
        priority=RequirementPriority.SHALL,
        verification_method="test",
        linked_code=["e2m2e.core.system"],
        linked_tests=["tests/core/test_system.py"],
    ),
    Requirement(
        id="REQ-011",
        title="特征尺度设置前置条件",
        category=RequirementCategory.FUNCTIONAL,
        description="在物理单位转换前必须调用 set_characteristic_scales() 设置特征尺度。",
        priority=RequirementPriority.SHOULD,
        verification_method="test",
        linked_code=["e2m2e.core.system"],
        linked_tests=["tests/core/test_system.py"],
    ),
    Requirement(
        id="REQ-012",
        title="积分容差默认值",
        category=RequirementCategory.PERFORMANCE,
        description="数值积分的相对容差和绝对容差默认值均为 1e-12（双精度机器精度量级）。",
        priority=RequirementPriority.SHALL,
        verification_method="inspection",
        linked_code=["e2m2e.core.dynamics"],
        linked_tests=["tests/core/test_dynamics.py"],
    ),
    # ---- 轨道数据 ----
    Requirement(
        id="REQ-020",
        title="Orbit 序列化兼容性",
        category=RequirementCategory.INTERFACE,
        description="Orbit.save_to_file / load_from_file 必须兼容 v3 JSON 格式。",
        priority=RequirementPriority.SHALL,
        verification_method="test",
        linked_code=["e2m2e.core.orbit"],
        linked_tests=["tests/core/orbit/test_orbit_io.py"],
    ),
    Requirement(
        id="REQ-021",
        title="Orbit 周期估计",
        category=RequirementCategory.FUNCTIONAL,
        description="Orbit 在初始化时自动通过零交叉检测估计轨道周期。",
        priority=RequirementPriority.SHOULD,
        verification_method="test",
        linked_code=["e2m2e.core.orbit"],
        linked_tests=["tests/core/test_orbit.py"],
    ),
    Requirement(
        id="REQ-022",
        title="OrbitFamily 聚合",
        category=RequirementCategory.FUNCTIONAL,
        description=(
            "OrbitFamily 提供统一的聚合接口：states、periods 属性与 get_jacobi_constants() 方法。"
        ),
        priority=RequirementPriority.SHALL,
        verification_method="test",
        linked_code=["e2m2e.core.orbit"],
        linked_tests=["tests/core/orbit/test_orbit_family.py"],
    ),
    # ---- 星历动力学 ----
    Requirement(
        id="REQ-025",
        title="EphemerisDynamics 统一接口",
        category=RequirementCategory.INTERFACE,
        description="EphemerisDynamics 必须与 CR3BP_Dynamics 共享 Dynamics 基类接口。",
        priority=RequirementPriority.SHALL,
        verification_method="test",
        linked_code=["e2m2e.core.ephemeris_dynamics"],
        linked_tests=["tests/core/dynamics/test_ephemeris_dynamics.py"],
    ),
    Requirement(
        id="REQ-026",
        title="EphemerisDynamics 自适应步长",
        category=RequirementCategory.FUNCTIONAL,
        description=(
            "EphemerisDynamics 根据传播时长自适应调整最大步长（max_step = min(60s, duration/10)）。"
        ),
        priority=RequirementPriority.SHOULD,
        verification_method="test",
        linked_code=["e2m2e.core.ephemeris_dynamics"],
        linked_tests=["tests/core/dynamics/test_ephemeris_dynamics.py"],
    ),
]
