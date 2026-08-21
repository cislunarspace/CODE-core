# e2m2e-levelset

水平集方法与 Hamilton-Jacobi（HJ）可达性求解器，自 UBC Ian M. Mitchell
的 [Toolbox of Level Set Methods](https://www.cs.ubc.ca/~mitchell/ToolboxLS/)
（ToolboxLS 1.1，MATLAB）移植。求解含时 HJ PDE `D_t φ = -H(x, t, φ, ∇φ)`，
为低推力轨迹的可达集、最短到达时间（TTR）与追逃博弈计算提供网格 PDE 内核。

与 e2m2e 其他数值 crate 的分工：e2m2e-propagation / e2m2e-forces 沿**单条轨迹**
积分 ODE，本 crate 在**结构网格**上演化 PDE；动力学通过
[`Hamiltonian`](src/hamiltonian.rs) trait 进入（如 CR3BP 相对运动的哈密顿量），
不依赖 e2m2e 的积分器。纯数学 crate：无 SPICE、无 pyo3；对外暴露经
e2m2e-integrators 按 ABI 戳流程统一进行。

## 许可证（重要）

本 crate 派生自 ToolboxLS，原作采用 ACM 非商业许可（全文见
`LICENSE-ToolboxLS`）。学术与科研使用免版税，但**分发必须保留原版权与
许可条款**，且不得用于商业产品。因此本 crate 的 `license-file` 不随
workspace 的 Apache-2.0。若未来 e2m2e 发布包含本 crate 的二进制，需在
发布说明中注明此差异；商业使用需联系原作者（mitchell@cs.ubc.ca）。

## 模块对照

| Rust 模块 | ToolboxLS 来源 | 说明 |
|---|---|---|
| `grid` | `Grids/processGrid.m` | 网格结构体，完整实现 |
| `boundary` | `BoundaryCondition/addGhost*.m` | 鬼单元填充（5 种边界条件） |
| `derivative` | `SpatialDerivative/UpwindFirst/upwindFirst*.m` | 一阶 / ENO2 / ENO3 / WENO5 迎风格式 |
| `hamiltonian` | 用户回调 `hamFunc` / `partialFunc` | 两个函数句柄合并为一个 trait |
| `dissipation` | `Dissipation/artificialDissipationGLF/LLF/LLLF.m` | LF 耗散系数与 CFL 步长上界 |
| `term` | `Term/termLaxFriedrichs/Normal/Reinit/Sum.m` | HJ 时间项 |
| `integrator` | `Integrators/odeCFL1/2/3.m` + `odeCFLset.m` | TVD Runge-Kutta + CFL 步长控制 |
| `integrator`（Post 模块） | `Helper/PostTimestep/postTimestepMask/Reinit/TTR.m` | 每步后处理钩子 |
| `integrator`（TerminalEvent） | `Helper/TerminalEvent/terminalEventConverge.m` | 收敛终止事件 |
| `shape` | `InitialConditions/BasicShapes` + `SetOperations` | 隐式形状初值与集合运算 |
| `signed_distance` | `Helper/SignedDistance/signedDistanceIterative.m` | 迭代法符号距离函数 |

不移植：向量水平集（`Examples/Vector`，多分量 cell）、绘图动画
（`Helper/Visualization` 与各 `animate*` 示例）、`termConvection` 等纯对流
快捷项（可用 LF 项 + 线性哈密顿量等价表达）。

## 协议映射（MATLAB → Rust 的核心设计决策）

ToolboxLS 的内核是一个函数句柄协议网，`schemeData` 弱类型结构体把网格、
格式选择和用户参数一起塞进每个回调。Rust 侧的做法是**把每个"协议角色"
物化为一个 trait，把 `schemeData` 的字段物化为结构体字段**：

| MATLAB 协议 | Rust 对应 | 备注 |
|---|---|---|
| `schemeData.derivFunc` 函数句柄 | [`UpwindDerivative`](src/derivative.rs) trait | `[derivL, derivR] = derivFunc(grid, data, dim)` 逐维求值保留 |
| `schemeData.hamFunc` + `partialFunc` | [`Hamiltonian`](src/hamiltonian.rs) trait（两方法） | 两个回调本就成对出现，合并避免实现半个协议 |
| `schemeData.dissFunc` | [`Dissipation`](src/dissipation.rs) trait | 返回 `(diss, stepBound)` 二元组，同 MATLAB |
| `schemeFunc(t, y, schemeData)` | [`Term`](src/term.rs) trait | `ydot`/`stepBound` 装进 `TermRhs` 结构体 |
| `hamFunc` 经 `nargout` 探测原位改 `schemeData` | 取消 | 动力学参数放实现结构体的字段，`&self` 求值；`Term::rhs(&mut self)` 保留可演化状态 |
| `schemeData` 的用户自由字段 | `Hamiltonian` 实现的自有字段 | air3D 等示例的转向率界等参数去向 |
| `grid.bdry{dim}` 函数句柄 + `bdryData` | [`BoundaryCondition`](src/grid.rs) 枚举 | 有限五种，枚举比闭包更可检验 |
| `grid` 结构体的 `xs`/`shape` 派生字段 | `Grid::axis()`/`shape()` 方法 | 不落盘冗余数据 |
| cell 数组（每维一项） | `Vec<ArrayD<f64>>` | 向量水平集（cell of cell）除外 |
| `y(:)` 向量化 / `reshape` 往返 | 全程 `ArrayD<f64>` | 只在与 Python 交互时序列化 |
| MATLAB 1-based 维号 | 0-based | 全库统一 |

数据约定：网格函数形状 = 各维节点数 `n`；节点为单元中心
`min + (i + 0.5) * dx`（与原版一致，`dx = (max - min) / n`）。

## 实现与验证状态

四个阶段全部实现完毕，验证不依赖 MATLAB 基准数据，改用解析解与
收敛阶两类自足门控（22 项测试用例全过：单测 10 + 集成测试 12）：

| 阶段 | 内容 | 验证门控（实测） |
|---|---|---|
| 1 | 鬼单元 ×5、一阶迎风、`ode_cfl1` | 周期回卷/常值/斜率/外推逐位断言；平流圆质心误差 < 0.02、面积比偏差 < 5%（81² 网格） |
| 2 | ENO2/ENO3/WENO5、GLF/LLF/LLLF、`termLaxFriedrichs`、`ode_cfl2/3` | sin 场导数收敛阶断言门限 0.8/1.8/2.5/4.0（实测 1.00/2.00/3.00/5.00，见 `derivative.rs` 单测）；Burgers 方程 t=0.5 与 Hopf–Lax 精确解 L∞ 误差 < 0.02（ENO2+GLF，N=401）且加密网格收敛 |
| 3 | `ReinitTerm`（含 Russo-Smereka 亚网格修正）、`signed_distance_iterative`、全部形状与集合运算 | 0.3 倍距离函数重初始化回真距离：最大误差 < 2.5 dx、平均 < 0.5 dx（81²）；Zalesak 圆盘缺口拓扑保持 |
| 4 | `RestrictUpdateTerm`、`PostTimestepTtrRecorder`、`TerminalEvent`、双积分器 TTR | 可达集边界与解析解错分 < 2%（101²）；TTR 最大误差 < 0.45 且分辨率 51→101 明显下降；收敛事件提前终止 |

TTR 的 0.45 量级是 LF 类格式的固有耗散（双积分器的 ∂H/∂p 与 p 无关，
GLF/LLF/LLLF 等价）：解析 TTR 的梯度模约 2–3，耗散滞后 ∫diss·dt 累积
即此量级，加倍网格实测降至 0.28，与 MATLAB 原版同分辨率行为一致。
WENO5/odeCFL3 组合可显著压低该误差，代价是计算量。

移植中修正过的两处易错点（回归防线在测试里）：
- 三阶均差的尺度因子是 `Δ²D2 / (3·dx)`，不是 `dx·Δ²D2 / 3`（在 h=1
  网格上两者 coincidentally 相等，收敛阶测试可捕获）；
- `termReinit` 组装的是 δ，ODE 右端项要取负（漏掉会指数发散）。

## 集成路径

- Python 暴露：经 `e2m2e-integrators` 的 pyo3 cdylib，新增子模块
  （如 `_levelset`），随 `abi-version.txt` 递增 ABI 戳；大数组（100³ 网格
  ≈ 8 MB/分量）以扁平 `Vec<f64>` + shape 跨界，与现有 `_integrators` 一致，
  不引入 pyo3-numpy。
- 可视化：Python 侧 matplotlib 等值线，不进 Rust。
