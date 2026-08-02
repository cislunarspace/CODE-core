# DFH 对齐需求规格（PRD）

> 本文档是 e2m2e 与 DFH 功能对齐的需求基线。各 FR 对应 e2m2e 架构设计的五大核心能力
> （见 `docs/architecture/architecture.md`）。验证策略遵循 ADR 0013：按物理定义完成任务，
> 不用黄金样本、不与其他软件强制对比。

## FR1：任务轨道设计

**目标**：从任务参数（轨道类型、高度、振幅、历元、力模型开关）产出标称轨道。

**轨道类型**：DRO、NRHO、Halo、Lissajous、L4、L5

**实现状态**：✅ 已完成（#254）

**入口**：`e2m2e.algorithm.design.design_orbit.design_orbit(orbit_type, ...)`

**输出**：`Orbit` 数据容器（states/times/metadata）

**关联**：
- issue #254（CLOSED）
- ADR 0015（NominalOrbit 坐标系——FR1↔FR2 数据契约）

---

## FR2：轨道保持

**目标**：沿标称轨道施加脉冲控制，仿真测定轨与控制误差，蒙特卡洛评估。

**控制模式**（以《控制方案.md》hybrid_auto 版为准）：
1. 宽松目标点（Loose）——二次型成本函数解析最优 Δv*（Q=R=I、S=1e-2·I）
2. 严格目标点（Tight）——位置重合微分修正（STM 子矩阵牛顿迭代）
3. 特征点（Special）——x-z 平面穿越约束 + 最小范数解

**误差模型**：
- 测定轨扰动（Box-Muller 高斯采样，位置/速度 1-sigma 可配）
- 推力执行误差（分段：Δv_min 不开机 / 绝对误差 / 相对误差 / Δv_max 失败）
- 光压弧段随机误差（弧段内固定、弧段间重采样）

**实现状态**：✅ 主体完成（#257），遗留项：
- #280：TIGHT/SPECIAL 量级对齐（TIGHT 偏低 3x、SPECIAL 偏高 16x）
- 角动量管理（模式 4-6）在 #261

**入口**：`e2m2e.algorithm.station_keeping.controller.control_orbit(input_ephemeris, control_mode=1/2/3, ...)`

**输出**：`ControlOrbitResult`（SK_STATISTIC、MANEUVERS、受控星历）

**验收标准**：
- [x] 三种控制模式各有端到端算例，全自动跑通
- [x] 蒙特卡洛统计输出 SK_STATISTIC/MANEUVERS
- [x] Rust 批量 STM 传播接口可用，蒙特卡洛 100 样本规模可接受耗时
- [x] 误差模型参数可配且有默认值
- [ ] TIGHT/SPECIAL 量级对齐（#280）

**关联**：
- issue #257（CLOSED）
- issue #280（OPEN：TIGHT/SPECIAL 量级对齐）
- issue #279（OPEN：4 个传播器 bug 回归测试）
- issue #261（OPEN：角动量管理）
- ADR 0015（NominalOrbit 数据契约）

---

## FR3：转移轨道设计

**目标**：从出发条件到目标轨道的转移路径（脉冲/小推力/低能量）。

**实现状态**：🔧 部分完成

**入口**：`e2m2e.algorithm.transfer.transfer_orbit.transfer_design(transfer_type, ...)`

**转移类型**：
- 脉冲（Lambert、三体 Lambert、多脉冲）
- 自然动力学（低能量、流形）
- 低推力
- 任务层搜索/优化（NSGA-II、porkchop）

**关联**：
- 架构设计 `docs/architecture/architecture.md` 第 3 层算法层

---

## FR4：轨道预报

**目标**：给定初值与力模型的高精度数值外推。

**实现状态**：✅ 已完成

**入口**：`e2m2e.algorithm.propagation`（薄壳）→ Rust `propagate_compiled_py` / `propagate_compiled_stm_py`

**关联**：
- ADR 0002（Rust 积分器核心）
- issue #279（h_init 步长上限 bug 回归测试）

---

## FR5：时空坐标转换

**目标**：TDT+GCRS ↔ TDB+EBCRS 等参考系与时间尺度转换。

**实现状态**：✅ 已完成（#252）

**入口**：`e2m2e.algorithm.coordinate.coordinate_system.CoordinateSystem`

**关联**：
- issue #252（CLOSED）
- ADR 0010（r2s2 GCRS-EBCRS 时空转换）
- ADR 0015（NominalOrbit 坐标系）

---

## 验证策略

遵循 ADR 0013：

1. **正确性由物理定义裁决**：解析解对照 + 物理不变量
2. **测试标准允许文献公式/解析值，不允许其他软件运行输出**
3. **不使用黄金样本对照**（输入格式规格测试除外——验证序列化格式，非算法正确性）
4. **DFH 仅作开发期交叉参考**（本地手动跑，诊断量级/系统性偏差；脚本放 `scripts/`，不进 CI）

## 未实现能力

| 能力 | Issue | 状态 |
|---|---|---|
| 角动量管理（控制模式 4-6） | #261 | 未实现 |
| ECOM 9 系数光压 | #253 | 部分完成（PR1） |
| LGA/WSB 低能量转移 | — | 未实现 |
| Facade/MCP 接口层 | — | 未实现 |
