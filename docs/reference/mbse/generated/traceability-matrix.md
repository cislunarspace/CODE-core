---
title: 需求追溯矩阵
---

# 需求追溯矩阵

| 需求 ID | 标题 | 类别 | 优先级 | 验证方法 | 关联代码 | 关联测试 |
|---------|------|------|--------|----------|----------|----------|
| REQ-001 | 状态向量顺序 | interface | shall | test | e2m2e.data.types.orbit<br>e2m2e.algorithm.dynamics.dynamics | tests/data/types/test_orbit.py<br>tests/algorithm/dynamics/test_dynamics_contract.py |
| REQ-002 | 传播结果 states 形状 | interface | shall | test | e2m2e.algorithm.dynamics.dynamics<br>e2m2e.algorithm.dynamics.ephemeris_dynamics | tests/algorithm/dynamics/test_dynamics_contract.py<br>tests/algorithm/dynamics/test_ephemeris_dynamics_legacy.py |
| REQ-003 | Jacobi 常数漂移容限 | performance | shall | test | e2m2e.algorithm.dynamics.dynamics | tests/algorithm/dynamics/test_cr3bp_model.py |
| REQ-004 | STM 解析 Jacobian | functional | shall | test | e2m2e.algorithm.dynamics.dynamics | tests/algorithm/dynamics/test_cr3bp_variational.py |
| REQ-005 | Dynamics 子类调用 super().__init__() | interface | shall | inspection | e2m2e.algorithm.dynamics.dynamics<br>e2m2e.algorithm.dynamics.ephemeris_dynamics | tests/algorithm/dynamics/test_ephemeris_dynamics_legacy.py |
| REQ-006 | 坐标变换互逆一致 | functional | shall | test | e2m2e.algorithm.coordinate.coordinate_system<br>e2m2e.algorithm.coordinate | tests/algorithm/coordinate/test_synodic_j2000.py |
| REQ-010 | 平动点位置精度 | performance | shall | test | e2m2e.algorithm.dynamics.cr3bp_system | tests/algorithm/dynamics/test_cr3bp_system.py |
| REQ-011 | 特征尺度设置前置条件 | functional | should | test | e2m2e.algorithm.dynamics.cr3bp_system | tests/algorithm/dynamics/test_cr3bp_system.py |
| REQ-012 | 积分容差默认值 | performance | shall | inspection | e2m2e.algorithm.dynamics.dynamics | tests/algorithm/dynamics/test_dynamics_contract.py |
| REQ-020 | Orbit 序列化兼容性 | interface | shall | test | e2m2e.data.types.orbit | tests/data/types/test_orbit_io.py |
| REQ-021 | Orbit 周期估计 | functional | should | test | e2m2e.data.types.orbit | tests/data/types/test_orbit.py |
| REQ-022 | OrbitFamily 聚合 | functional | shall | test | e2m2e.data.types.orbit | tests/data/types/test_orbit_family.py |
| REQ-025 | EphemerisDynamics 统一接口（遗留） | interface | shall | test | e2m2e.algorithm.dynamics.ephemeris_dynamics | tests/algorithm/dynamics/test_ephemeris_dynamics_legacy.py |
| REQ-026 | EphemerisDynamics 自适应步长（遗留） | functional | should | test | e2m2e.algorithm.dynamics.ephemeris_dynamics | tests/algorithm/dynamics/test_ephemeris_dynamics_legacy.py |
| REQ-100 | 微分修正 50 次迭代内收敛 | performance | shall | test | e2m2e.algorithm.solver.differential_correction | tests/algorithm/correction/test_differential_correction.py |
| REQ-101 | 收敛容差默认 1e-12 | performance | shall | inspection | e2m2e.algorithm.solver.differential_correction | tests/algorithm/correction/test_differential_correction.py |
| REQ-102 | 策略模式分离配置与迭代 | interface | should | inspection | e2m2e.algorithm.family.strategies<br>e2m2e.algorithm.solver.differential_correction | tests/algorithm/correction/test_differential_correction.py |
| REQ-103 | Continuation 不重复 CR3BP 物理 | interface | shall | inspection | e2m2e.algorithm.solver.continuation | tests/algorithm/correction/test_continuation.py |
| REQ-104 | 算法层 STM 解析计算 | functional | shall | inspection | e2m2e.algorithm.solver.differential_correction | tests/algorithm/correction/test_differential_correction.py |
| REQ-105 | Richardson 三阶近似精度 | functional | should | test | e2m2e.algorithm.solver.differential_correction | tests/algorithm/correction/test_differential_correction.py |
| REQ-110 | 稳定性指标满足 v1*v2 = 1 | performance | should | test | e2m2e.algorithm.stability | tests/algorithm/stability/test_stability.py |
| REQ-111 | MultipleShooting 并行传播 | functional | should | test | e2m2e.algorithm.solver.multiple_shooting | tests/algorithm/correction/test_multiple_shooting.py |
| REQ-112 | 延拓步长自适应 | functional | should | test | e2m2e.algorithm.solver.continuation | tests/algorithm/correction/test_continuation.py |
| REQ-113 | 伪弧长延拓切向量计算 | functional | shall | inspection | e2m2e.algorithm.solver.continuation | tests/algorithm/correction/test_continuation.py |
