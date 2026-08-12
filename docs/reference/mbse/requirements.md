---
title: 功能需求
---

# 功能需求

```mermaid
requirementDiagram
    requirement REQ_001 {
        title: 状态向量顺序
        type: interface
        risk: shall
    }
    e2m2e_core_orbit -satisfies-> REQ_001
    e2m2e_core_dynamics -satisfies-> REQ_001
    requirement REQ_002 {
        title: 传播结果 states 形状
        type: interface
        risk: shall
    }
    e2m2e_core_dynamics -satisfies-> REQ_002
    e2m2e_core_ephemeris_dynamics -satisfies-> REQ_002
    requirement REQ_003 {
        title: Jacobi 常数漂移容限
        type: performance
        risk: shall
    }
    e2m2e_core_dynamics -satisfies-> REQ_003
    requirement REQ_004 {
        title: STM 解析 Jacobian
        type: functional
        risk: shall
    }
    e2m2e_core_dynamics -satisfies-> REQ_004
    requirement REQ_005 {
        title: Dynamics 子类调用 super().__init__()
        type: interface
        risk: shall
    }
    e2m2e_core_dynamics -satisfies-> REQ_005
    e2m2e_core_ephemeris_dynamics -satisfies-> REQ_005
    requirement REQ_006 {
        title: 坐标变换互逆一致
        type: functional
        risk: shall
    }
    e2m2e_core_coordinate -satisfies-> REQ_006
    requirement REQ_010 {
        title: 平动点位置精度
        type: performance
        risk: shall
    }
    e2m2e_core_system -satisfies-> REQ_010
    requirement REQ_011 {
        title: 特征尺度设置前置条件
        type: functional
        risk: should
    }
    e2m2e_core_system -satisfies-> REQ_011
    requirement REQ_012 {
        title: 积分容差默认值
        type: performance
        risk: shall
    }
    e2m2e_core_dynamics -satisfies-> REQ_012
    requirement REQ_020 {
        title: Orbit 序列化兼容性
        type: interface
        risk: shall
    }
    e2m2e_core_orbit -satisfies-> REQ_020
    requirement REQ_021 {
        title: Orbit 周期估计
        type: functional
        risk: should
    }
    e2m2e_core_orbit -satisfies-> REQ_021
    requirement REQ_022 {
        title: OrbitFamily 聚合
        type: functional
        risk: shall
    }
    e2m2e_core_orbit -satisfies-> REQ_022
    requirement REQ_025 {
        title: EphemerisDynamics 统一接口
        type: interface
        risk: shall
    }
    e2m2e_core_ephemeris_dynamics -satisfies-> REQ_025
    requirement REQ_026 {
        title: EphemerisDynamics 自适应步长
        type: functional
        risk: should
    }
    e2m2e_core_ephemeris_dynamics -satisfies-> REQ_026
    requirement REQ_100 {
        title: 微分修正 50 次迭代内收敛
        type: performance
        risk: shall
    }
    e2m2e_algorithms_differential_correction -satisfies-> REQ_100
    requirement REQ_101 {
        title: 收敛容差默认 1e-12
        type: performance
        risk: shall
    }
    e2m2e_algorithms_differential_correction -satisfies-> REQ_101
    requirement REQ_102 {
        title: 策略模式分离配置与迭代
        type: interface
        risk: should
    }
    e2m2e_algorithms_strategies -satisfies-> REQ_102
    e2m2e_algorithms_differential_correction -satisfies-> REQ_102
    requirement REQ_103 {
        title: Continuation 不重复 CR3BP 物理
        type: interface
        risk: shall
    }
    e2m2e_algorithms_continuation -satisfies-> REQ_103
    requirement REQ_104 {
        title: 算法层 STM 解析计算
        type: functional
        risk: shall
    }
    e2m2e_algorithms_differential_correction -satisfies-> REQ_104
    requirement REQ_105 {
        title: Richardson 三阶近似精度
        type: functional
        risk: should
    }
    e2m2e_algorithms_differential_correction -satisfies-> REQ_105
    requirement REQ_110 {
        title: 稳定性指标满足 v1*v2 = 1
        type: performance
        risk: should
    }
    e2m2e_algorithms_stability -satisfies-> REQ_110
    requirement REQ_111 {
        title: MultipleShooting 并行传播
        type: functional
        risk: should
    }
    e2m2e_algorithms_multiple_shooting -satisfies-> REQ_111
    requirement REQ_112 {
        title: 延拓步长自适应
        type: functional
        risk: should
    }
    e2m2e_algorithms_continuation -satisfies-> REQ_112
    requirement REQ_113 {
        title: 伪弧长延拓切向量计算
        type: functional
        risk: shall
    }
    e2m2e_algorithms_continuation -satisfies-> REQ_113
```
