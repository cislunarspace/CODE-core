# ADR 0028：平面三角平动点族采用全周期伪弧长延拓

**状态**：已采纳（#428 的统一 Rust 接缝经 ADR 0029 修订并实施）
**日期**：2026-08-16
**关联 Issue**：#436、#428、#435、#451
**关联**：ADR 0024（统一算法结果状态契约）、ADR 0013（按定义验证）

## 背景

SPO、LPO 与 Horseshoe 是 L4/L5 附近的平面周期轨道。它们没有 Halo 的 x 轴或 XZ 镜面对称，因而必须按完整周期闭合。现有 `DifferentialCorrection` 已能在固定 `x0` 时做三残差全周期修正；`Continuation.natural_continuation()` 和既有 PAL 则都依赖半周期对称约束，不能承载这两类问题。

固定 `x0` 的自然参数延拓在局部可行，却不能作为 LPO 全局参数化：振幅和周期会转向，且 Horseshoe 是 LPO 的大振幅成员，不应再建一套独立延拓方程。目标是决定正式的算法形状，而不是在本 ADR 中修改 Facade 或公开请求模型。

## 数值证据

实验脚本为 `scripts/research_issue_436_full_period_pal.py`（使命完成后已移除，可在下述固定提交的 git 历史中取得），固定提交 `14025914956faa90a1d3e24019db5ca6c33647af`、标准地月 CR3BP 质量比 `0.01215058560962404`、L4、小振幅线性化种子、归一化弧长步长 `0.01`。运行前通过 `make setup && make dev` 配置 CSPICE、内核和 Rust 扩展。

- LPO 向 `x0` 减小方向走 60 步，记录在 `docs/research/issue-436-lpo-pal-long.json`。L4 的轨道几何振幅度量从 586 km 到 238,833 km；最大完整平面闭合无穷范数为 `3.30e-10`，最大 Jacobi 漂移为 `3.11e-15`。第 9 至 13 步周期先减后增，但链连续，未发生跳支或修正失败。
- L5 LPO 同方向走 20 步，记录在 `docs/research/issue-436-l5-lpo-pal-extended.json`。其轨道几何振幅度量越过 110,000 km 到 138,526 km；最大完整平面闭合无穷范数为 `8.17e-10`，最大 Jacobi 漂移为 `1.78e-15`。
- SPO 同一方向走 5 步，记录在 `docs/research/issue-436-spo-pal-local-baseline.json`。最大完整平面闭合无穷范数为 `1.58e-11`，最大 Jacobi 漂移为 `1.33e-15`。
- LPO 的完整闭合加相位条件在相对奇异值阈值 `1e-8` 下有效秩为 4；加伪弧长条件后的增广系统有效秩为 5。积分和 STM 误差会把自治系统理论零空间抬升，不能以机器精度阈值判断秩。

这些结果证明该公式能连续追踪 L4 长程分支和 L5 的对应扩展，并越过当前 110,000 km 的搜索声明范围；它们不证明物理振幅上限，也不足以改变 #435 的公开范围。反向分支、碰撞边界及其他归一化/步长仍需在后续实现验收中复测。

## 决策

### 1. 采用平面全周期 PAL

令平面初态为 `s=(x0,y0,vx0,vy0)`，未知量为 `q=(s,T)`。正式算法使用：

```text
R(q) = Pi(phi_T(s)) - s = 0
h(q; qk) = (s - sk) dot f(sk) / ||f(sk)|| = 0
g(q; qk, tk, ds) = ((q - qk) / scales) dot tk - ds = 0
```

其中 `Pi` 取平面四个分量，`qk` 是上一条已收敛轨道，`tk` 是其切向量。`R` 检查全部四个平面闭合分量；相位条件消除自治系统的相位自由度；伪弧长条件选择相邻的族成员。切向量由 `[dR/dq; dh/dq]` 的零空间取得并与上一切向量同向化。预测后对 `[R; h; g]` 的 6 行、5 列增广系统作最小二乘 Newton 修正。

首版固定 `scales=(1,1,1,1,10)`，与实验一致。实现必须记录归一化并在条件数恶化时缩小步长；不把 `x0`、振幅、周期或 Jacobi 常数强制为单调参数。初始 `x0` 方向只决定第一步切向量符号，穿过折叠后继续沿弧长方向。

### 2. 使用专用深模块，不泛化现有 Halo PAL

在 `crates/e2m2e-integrators` 新增平面全周期 PAL 数值内核；Python 面向调用方的接口保持为一个族生成入口：

```python
generate_planar_periodic_family(
    dynamics,
    seed_orbit,
    *,
    family_type,
    libration_point,
    n_orbits,
    step_size,
    initial_direction,
)
```

Rust 内部封装 STM 闭合 Jacobian、相位规范、SVD 切向量、固定阻尼的 SVD 最小二乘 Newton、线搜索、步长收缩与有效秩检查。Python 只从 `CR3BP_Dynamics` 读取质量参数与积分配置、调用 Rust 内核，再把原始成员解释为 `OrbitFamily`；返回 ADR 0024 定义的 `ContinuationResult`，其中 `family` 总是包含种子和已完成的部分轨道。调用方不传残差索引、相位函数或 Jacobian 形状。

SPO、LPO 与 Horseshoe 共用这个模块；族名称、L4/L5 标签和振幅测量留在 `algorithm.family` 编排层。Horseshoe 是 LPO 的成员分类，不获得第二套求解器。

既有 `Continuation.pseudo_arclength_continuation()` 保持 Halo 专用。它的 XZ 对称自由变量和物理范围不能通过参数开关兼容此问题。当前只有这一种非对称平面适配器，故不引入“任意周期问题适配器”这一抽象；当三维全周期族出现第二个实际适配器时，再抽取共享 PAL 数值内核。

复用现有多重打靶的正规方程高斯消元不满足该问题：PAL 必须识别闭合加相位矩阵的秩亏零空间，并求解病态的增广最小二乘。Rust 内核使用 `nalgebra` 的 SVD 与秩判定，避免把该数值责任复制到 Python。

### 3. 明确失败与精度契约

成功成员必须满足完整六维闭合无穷范数不大于 `1e-8`、平面约束 `z=vz=0`，并通过 CR3BP Jacobi 漂移检查。增广系统无法保持有效秩 5 时返回 `FAILED/SINGULAR_JACOBIAN`；步长缩小至下限仍不能得到收敛成员时返回 `STAGNATED/STAGNATION_DETECTED`。任何软失败保留已生成的 `OrbitFamily`，不以裸 `None` 或隐式回退表达。每个成员的有效秩、条件数、Newton 迭代数和实际弧长步长保留在 Rust 边界结果与数值诊断入口中；数据层的 `OrbitFamily` 只记录族语义，不扩展为诊断容器。

## 取舍

将 `iterate_full_period_correction()` 塞入现有自然延拓，改动小但只能继续固定 `x0`，不能处理参数折叠。保留网格搜索可继续按振幅找单条轨道，却不能产生连续族或说明失败位置。二者可保留为单轨设计基线，但不是 #428 的族生成方法。

直接做“全 6D 状态加周期”的通用 PAL，会同时决定三维残差独立性、相位规范和新接口，尚无第二个调用方验证其抽象价值。首版限于平面问题，换来更小接口和更集中的数值验证；代价是三维非对称周期族出现时需要一项新的设计决策。

实验已超过 110,000 km，但只覆盖一条 L4 单向分支和一种步长。立刻放宽公开振幅范围会把方法可行性误写成物理可达性，因此拒绝。

## 结果

后续 #428 实现 SPO/LPO/Horseshoe 族分派时，以本 ADR 的 Rust 内核和 Python 族生成接缝为前提。实现验收至少覆盖 L4/L5、两个初始方向、局部 SPO/LPO 链、LPO 周期转向区、结构化失败和既有 Halo PAL 回归；大振幅扫描继续由可复跑脚本记录，不把长时数值实验塞入常规 pytest。

## 修订（2026-08-16，#451）

决策 2 初稿将平面全周期 PAL 数值模块写在 Python `solver` 包。对照五层架构后修订为：数值迭代位于 Rust 数值层，`algorithm.family` 保留问题构造与结果解释的族生成接缝；公开方程、结果状态契约和不泛化 Halo PAL 的决定不变。实现引入了 `nalgebra`，原因是既有 Rust 打靶的正规方程高斯消元无法提供 PAL 所需的秩亏零空间、有效秩和病态最小二乘诊断。

#451 已在当前分支完成首个纵向切片：Rust 内核、PyO3 ABI、Python 族生成接缝、L4/L5 的 SPO/LPO 接缝测试、LPO 周期转向用例、部分族保留与既有逐族延拓回归均已通过。反向初始方向和更长程分支扫描仍按本 ADR 的验收边界继续补齐。

## 修订（2026-08-16，ADR 0029、#428）

#428 将本 ADR 决策 2 的 Python 族生成接缝收进统一 Rust 族生成模块。底层 `PlanarPalRustResult` 仍按决策 3 始终保留种子和完整已收敛 trace；Facade 返回的是按请求振幅窗口筛选后的领域族，数值种子不在窗口时不进入公开成员。筛选后的成员继续保留有效秩、条件数、Newton 次数、实际步长、闭合误差和 Jacobi 漂移。
