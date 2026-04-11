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
```
