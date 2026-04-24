---
title: 需求追溯矩阵
---

# 需求追溯矩阵

## 覆盖率: 100.0% (24/24)

| 需求 ID | 标题 | 类别 | 优先级 | 验证方法 | 关联代码 | 关联测试 |
|---------|------|------|--------|----------|----------|----------|
| REQ-001 | 状态向量顺序 | interface | shall | test | orbit, dynamics | test_orbit.py, test_dynamics.py |
| REQ-002 | 传播结果 states 形状 | interface | shall | test | dynamics, ephemeris_dynamics | test_dynamics.py, test_ephemeris_dynamics.py |
| REQ-003 | Jacobi 常数漂移容限 | performance | shall | test | dynamics | test_dynamics.py |
| REQ-004 | STM 解析 Jacobian | functional | shall | analysis | dynamics | test_dynamics.py |
| REQ-005 | Dynamics 子类调用 super().__init__() | interface | shall | inspection | dynamics, ephemeris_dynamics | test_ephemeris_dynamics.py |
| REQ-006 | 坐标变换互逆一致 | functional | shall | test | coordinate | test_coordinate.py, test_coordinate.py |
| REQ-010 | 平动点位置精度 | performance | shall | test | system | test_system.py |
| REQ-011 | 特征尺度设置前置条件 | functional | should | test | system | test_system.py |
| REQ-012 | 积分容差默认值 | performance | shall | inspection | dynamics | test_dynamics.py |
| REQ-020 | Orbit 序列化兼容性 | interface | shall | test | orbit | test_orbit_io.py |
| REQ-021 | Orbit 周期估计 | functional | should | test | orbit | test_orbit.py |
| REQ-022 | OrbitFamily 聚合 | functional | shall | test | orbit | test_orbit_family.py |
| REQ-025 | EphemerisDynamics 统一接口 | interface | shall | test | ephemeris_dynamics | test_ephemeris_dynamics.py |
| REQ-026 | EphemerisDynamics 自适应步长 | functional | should | test | ephemeris_dynamics | test_ephemeris_dynamics.py |
| REQ-100 | 微分修正 50 次迭代内收敛 | performance | shall | test | differential_correction | test_differential_correction.py |
| REQ-101 | 收敛容差默认 1e-12 | performance | shall | inspection | differential_correction | test_differential_correction.py |
| REQ-102 | 策略模式分离配置与迭代 | interface | should | inspection | strategies, differential_correction | test_differential_correction.py |
| REQ-103 | Continuation 不重复 CR3BP 物理 | interface | shall | inspection | continuation | test_continuation.py |
| REQ-104 | 算法层 STM 解析计算 | functional | shall | inspection | differential_correction | test_differential_correction.py |
| REQ-105 | Richardson 三阶近似精度 | functional | should | test | differential_correction | test_differential_correction.py |
| REQ-110 | 稳定性指标满足 v1*v2 = 1 | performance | should | test | stability | test_stability.py |
| REQ-111 | MultipleShooting 并行传播 | functional | should | test | multiple_shooting | test_multiple_shooting.py |
| REQ-112 | 延拓步长自适应 | functional | should | test | continuation | test_continuation.py |
| REQ-113 | 伪弧长延拓切向量计算 | functional | shall | inspection | continuation | test_continuation.py |

## 按层统计

| 层 | 需求数量 | 需求 ID 范围 |
|------|----------|--------------|
| Core | 14 | REQ-001 ~ REQ-026 |
| Algorithms | 10 | REQ-100 ~ REQ-113 |
| **总计** | **24** | |

## 按验证方法统计

- **test**: 16 条需求
- **analysis**: 1 条需求
- **inspection**: 7 条需求

## 按优先级统计

- **shall**: 16 条需求
- **should**: 8 条需求