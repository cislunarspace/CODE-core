"""Algorithms 层需求定义

定义 e2m2e 算法层（algorithms/）的系统需求，覆盖微分修正、延拓、
稳定性分析、多点射击法等核心数值算法。需求 ID 范围：REQ-100 ~ REQ-119。
"""

from .base import Requirement, RequirementCategory, RequirementPriority

ALGORITHMS_REQUIREMENTS = [
    # ---- 微分修正 ----
    Requirement(
        id="REQ-100",
        title="微分修正 50 次迭代内收敛",
        category=RequirementCategory.PERFORMANCE,
        description="DifferentialCorrection 在默认容差 1e-12 下应在 50 次迭代内收敛。",
        priority=RequirementPriority.SHALL,
        verification_method="test",
        linked_code=["e2m2e.algorithm.solver.differential_correction"],
        linked_tests=["tests/algorithm/solver/test_differential_correction.py"],
    ),
    Requirement(
        id="REQ-101",
        title="收敛容差默认 1e-12",
        category=RequirementCategory.PERFORMANCE,
        description="DifferentialCorrection 默认容差为 1e-12。",
        priority=RequirementPriority.SHALL,
        verification_method="inspection",
        linked_code=["e2m2e.algorithm.solver.differential_correction"],
        linked_tests=["tests/algorithm/solver/test_differential_correction.py"],
    ),
    Requirement(
        id="REQ-102",
        title="策略模式分离配置与迭代",
        category=RequirementCategory.INTERFACE,
        description=(
            "DifferentialCorrection 使用 CorrectionConfig 策略对象"
            " 将修正配置与 Newton 迭代循环分离。"
        ),
        priority=RequirementPriority.SHOULD,
        verification_method="inspection",
        linked_code=[
            "e2m2e.algorithm.family.strategies",
            "e2m2e.algorithm.solver.differential_correction",
        ],
        linked_tests=["tests/algorithm/solver/test_differential_correction.py"],
    ),
    Requirement(
        id="REQ-103",
        title="Continuation 不重复 CR3BP 物理",
        category=RequirementCategory.INTERFACE,
        description=(
            "Continuation 模块的 compute_F_and_dF_symmetric_xz_plane"
            " 必须通过 CR3BP_Dynamics 实例调用运动方程和 Jacobian，"
            "不得本地复制物理公式。"
        ),
        priority=RequirementPriority.SHALL,
        verification_method="inspection",
        linked_code=["e2m2e.algorithm.solver.continuation"],
        linked_tests=["tests/algorithm/solver/test_continuation.py"],
    ),
    Requirement(
        id="REQ-104",
        title="算法层 STM 解析计算",
        category=RequirementCategory.FUNCTIONAL,
        description=(
            "DifferentialCorrection 的 iterate_correction 必须使用解析 STM"
            "（来自 propagate(with_stm=True)），默认不使用有限差分。"
        ),
        priority=RequirementPriority.SHALL,
        verification_method="inspection",
        linked_code=["e2m2e.algorithm.solver.differential_correction"],
        linked_tests=["tests/algorithm/solver/test_differential_correction.py"],
    ),
    Requirement(
        id="REQ-105",
        title="Richardson 三阶近似精度",
        category=RequirementCategory.FUNCTIONAL,
        description=(
            "compute_halo_initial_guess 生成的初始猜测经过微分修正后能收敛到 Halo 周期轨道。"
        ),
        priority=RequirementPriority.SHOULD,
        verification_method="test",
        linked_code=["e2m2e.algorithm.solver.differential_correction"],
        linked_tests=["tests/algorithm/solver/test_differential_correction.py"],
    ),
    # ---- 稳定性分析 ----
    Requirement(
        id="REQ-110",
        title="稳定性指标满足 v1*v2 = 1",
        category=RequirementCategory.PERFORMANCE,
        description="对于保守系统 CR3BP，单周期轨道的 Floquet 乘子乘积 v1*v2 = 1（辛条件）。",
        priority=RequirementPriority.SHOULD,
        verification_method="test",
        linked_code=["e2m2e.algorithm.stability"],
        linked_tests=["tests/algorithm/stability/test_stability.py"],
    ),
    # ---- 多点射击 ----
    Requirement(
        id="REQ-111",
        title="MultipleShooting 并行传播",
        category=RequirementCategory.FUNCTIONAL,
        description="MultipleShooting 支持通过 n_workers 参数进行并行传播，结果与串行一致。",
        priority=RequirementPriority.SHOULD,
        verification_method="test",
        linked_code=["e2m2e.algorithm.solver.multiple_shooting"],
        linked_tests=["tests/algorithm/solver/test_multiple_shooting.py"],
    ),
    # ---- 延拓 ----
    Requirement(
        id="REQ-112",
        title="延拓步长自适应",
        category=RequirementCategory.FUNCTIONAL,
        description=(
            "Continuation 在修正成功时增大步长、失败时减小步长，步长范围 [min_step, max_step]。"
        ),
        priority=RequirementPriority.SHOULD,
        verification_method="test",
        linked_code=["e2m2e.algorithm.solver.continuation"],
        linked_tests=["tests/algorithm/solver/test_continuation.py"],
    ),
    Requirement(
        id="REQ-113",
        title="伪弧长延拓切向量计算",
        category=RequirementCategory.FUNCTIONAL,
        description="伪弧长延拓使用 SVD 计算 Jacobian 零空间作为切向量（预测方向）。",
        priority=RequirementPriority.SHALL,
        verification_method="inspection",
        linked_code=["e2m2e.algorithm.solver.continuation"],
        linked_tests=["tests/algorithm/solver/test_continuation.py"],
    ),
]
