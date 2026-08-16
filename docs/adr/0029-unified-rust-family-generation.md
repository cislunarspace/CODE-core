# ADR 0029：轨道族生成采用统一 Rust 深模块

**状态**：已采纳（已实施）
**日期**：2026-08-16
**关联 Issue**：#428
**关联**：ADR 0011（数值与编排分工）、ADR 0014（Facade 响应）、ADR 0024（状态三元组）、ADR 0028（平面全周期 PAL，本篇局部修订其接缝）

## 背景

`FamilyGenerationRequest` 已登记 Halo、NRHO、Axial、Lissajous、SPO、LPO 和 Horseshoe，但 Facade 只有 Halo 族生成。现有数值能力分散在 Python 族行走、Rust 微分修正、Halo PAL 和平面全周期 PAL 中；如果 Facade 逐条循环单轨入口，步长、筛选和失败语义会泄漏到接口层。

Lissajous 还带来第二个问题：它是双频拟周期有界轨迹，不满足周期闭合。单轨 `design_lissajous` 已使用高阶 normal-form 中心流形；族生成又要求数值核心全在 Rust，不能从 Python 循环调用该入口。

## 决策

### 1. 七族共用一个 Rust 生成接口

`e2m2e-integrators::family_generation` 提供一个带标签规格的纯 Rust 模块，经 `generate_cr3bp_family_py` 暴露。一次调用完成种子构造、传播、STM、微分修正、PAL、步长控制、成员筛选、几何度量和结构化终止。Python 只校验请求、选择族规格并把原始成员重包为领域对象；没有 Python 数值回退，也不逐成员跨 FFI。

内部规格按七族分型。固定的采样规则和延拓方向不做数值配置：NRHO L1 使用单条 Rust Halo PAL，L2 从 DE421 地月折叠后标定成员固定 x0 行走；Axial 从 DE421 垂直临界轨道标定种子固定 vz0 行走；Horseshoe 复用 LPO 链。

### 2. Facade 返回专属 Pydantic 响应并保持 OrbitFamily 兼容

`FamilyGenerationResponse` 以 Pydantic 模型直接承载 `status/cause/message`、请求/生成成员数和族成员，同时继承 `OrbitFamily` 的读取接口。成功和软失败使用同一响应；软失败保留已完成成员。算法层继续使用 `FamilyGenerationResult`，数据层 `OrbitFamily` 不承载算法状态。

这满足 ADR 0014/0024 的接口状态契约，也保持 #428 要求的成功结果可按 `OrbitFamily` 迭代、索引和读取周期语义。

### 3. Lissajous 族使用 Rust 非线性中心约化流

族采样在共线点四维中心子空间内参数化状态。Rust 用完整 CR3BP 非线性势梯度计算约化右端，以 RK4 推进面内和面外两个中心自由度；状态重建始终排除线性双曲方向，因此结果按构造有界并保留非线性频率/振幅耦合。结果标记 `periodicity="quasi-periodic"`，`period` 仅为面内名义周期。

该约化流不宣称等同于单轨入口的高阶 normal-form 展开。`design_lissajous` 保持原实现和精度；两条接口共享振幅、相位、有界和拟周期语义，但服务不同：单轨高阶设计与族参数扫描。

### 4. 区分 PAL trace 与公开振幅窗口

ADR 0028 的 `PlanarPalRustResult` 仍始终包含种子和全部已完成 PAL trace，软失败诊断不变。#428 的 SPO/LPO/Horseshoe 结果是该 trace 上的公开振幅窗口筛选：1000 km 数值种子不在请求窗口时，不进入最终 `OrbitFamily`，否则会违反请求范围。

筛选后的每个成员保留其 Newton 次数、切向/增广系统有效秩与条件数、实际步长、闭合误差和 Jacobi 漂移。因而公开族不冒充原始 PAL trace，也不丢失所返回成员的数值诊断。

## 理由

单入口把七族共同的状态、失败和 FFI 契约集中在一处，同时把族差异藏在 Rust 枚举后，接口比七个扁平数值函数更深。数值依赖均为进程内 Rust 计算，不需要端口或回调。

Facade 若直接返回算法 dataclass，会违反接口层 Pydantic 响应约定；若把状态字段塞入 `OrbitFamily`，会污染数据层职责。Pydantic 响应继承既有读取接口，是同时满足两项约束的最小兼容方案。

Lissajous 直接传播完整 CR3BP 会放大残余双曲分量；纯线性轨迹又没有非线性耦合。中心约化右端在不引入 Python normal-form 的前提下保留非线性，并从状态空间中排除发散方向。

## 结果

PyO3 ABI 升至 v15。七族成功与软失败共享状态三元组；`n_orbits` 始终是最终成员数上限。周期族成员验证完整周期闭合、Jacobi 守恒和适用对称性；Lissajous 验证多点、有界、有限和严格递增时间。

NRHO/Axial 的标定种子当前限 DE421 地月上下文，Rust 对其他质量参数显式拒绝，不静默套用。一般 CR3BP 的标定扩展需要新的数值证据和决策。
