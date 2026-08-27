# e2m2e-levelset

[English](#english) | [简体中文](#简体中文)

## English

Level-set methods and Hamilton-Jacobi (HJ) reachability solvers, ported from
Ian M. Mitchell's (UBC) [Toolbox of Level Set
Methods](https://www.cs.ubc.ca/~mitchell/ToolboxLS/) (ToolboxLS 1.1, MATLAB).
It solves time-dependent HJ PDEs `D_t φ = -H(x, t, φ, ∇φ)`, providing a grid
PDE kernel for reachability sets, shortest time-to-reach (TTR), and
pursuit-evasion game computations for low-thrust trajectories.

Division of labor with e2m2e's other numerical crates: e2m2e-propagation /
e2m2e-forces integrate ODEs along a **single trajectory**, while this crate
evolves PDEs on **structured grids**; dynamics enter via the
[`Hamiltonian`](src/hamiltonian.rs) trait (e.g. the CR3BP relative-motion
Hamiltonian) without depending on e2m2e's integrators. Pure math crate: no
SPICE, no pyo3; external exposure goes through e2m2e-integrators under the
unified ABI-stamp process.

### License (important)

This crate derives from ToolboxLS, whose original license is the ACM
non-commercial license (full text in `LICENSE-ToolboxLS`). Academic and
research use is royalty-free, but **redistribution must retain the original
copyright and license terms**, and commercial products are not permitted.
Hence this crate's `license-file` deviates from the workspace Apache-2.0. If
a future e2m2e release ships binaries including this crate, release notes
must state this difference; commercial use requires contacting the original
author (mitchell@cs.ubc.ca).

### Module mapping

| Rust module | ToolboxLS source | Notes |
|---|---|---|
| `grid` | `Grids/processGrid.m` | Grid struct, complete |
| `boundary` | `BoundaryCondition/addGhost*.m` | Ghost-cell fill (5 boundary conditions) |
| `derivative` | `SpatialDerivative/UpwindFirst/upwindFirst*.m` | First-order / ENO2 / ENO3 / WENO5 upwind schemes |
| `hamiltonian` | user callbacks `hamFunc` / `partialFunc` | Two function handles merged into one trait |
| `dissipation` | `Dissipation/artificialDissipationGLF/LLF/LLLF.m` | LF dissipation coefficients and CFL step bounds |
| `term` | `Term/termLaxFriedrichs/Normal/Reinit/Sum.m` | HJ temporal term |
| `integrator` | `Integrators/odeCFL1/2/3.m` + `odeCFLset.m` | TVD Runge-Kutta + CFL step control |
| `integrator` (Post modules) | `Helper/PostTimestep/postTimestepMask/Reinit/TTR.m` | Per-step post-processing hooks |
| `integrator` (TerminalEvent) | `Helper/TerminalEvent/terminalEventConverge.m` | Convergence terminal event |
| `shape` | `InitialConditions/BasicShapes` + `SetOperations` | Implicit shape initial data and set operations |
| `signed_distance` | `Helper/SignedDistance/signedDistanceIterative.m` | Iterative signed distance function |

Not ported: vector level sets (`Examples/Vector`, multi-component cells),
plotting/animation (`Helper/Visualization` and the various `animate*`
examples), and pure-convection shortcuts such as `termConvection` (expressible
equivalently with an LF term plus linear Hamiltonians).

### Protocol mapping (core MATLAB → Rust design decisions)

ToolboxLS's kernel is a web of function-handle protocols; the weakly-typed
`schemeData` struct stuffs grids, scheme choices, and user parameters into
every callback. On the Rust side, **each "protocol role" is materialized as a
trait, and `schemeData` fields become struct fields**:

| MATLAB protocol | Rust counterpart | Notes |
|---|---|---|
| `schemeData.derivFunc` function handle | [`UpwindDerivative`](src/derivative.rs) trait | `[derivL, derivR] = derivFunc(grid, data, dim)` evaluated per dimension, preserved |
| `schemeData.hamFunc` + `partialFunc` | [`Hamiltonian`](src/hamiltonian.rs) trait (two methods) | The two callbacks always appear as a pair; merging avoids implementing half a protocol |
| `schemeData.dissFunc` | [`Dissipation`](src/dissipation.rs) trait | Returns `(diss, stepBound)`, same as MATLAB |
| `schemeFunc(t, y, schemeData)` | [`Term`](src/term.rs) trait | `ydot`/`stepBound` packed into a `TermRhs` struct |
| `hamFunc` mutating `schemeData` in place probed via `nargout` | removed | Dynamics parameters live in fields of the implementing struct; evaluated with `&self`; `Term::rhs(&mut self)` keeps evolvable state |
| Free-form user fields of `schemeData` | own fields of the `Hamiltonian` impl | Where parameters like air3D turning-rate bounds go |
| `grid.bdry{dim}` function handle + `bdryData` | [`BoundaryCondition`](src/grid.rs) enum | A finite set of five; enums are more testable than closures |
| Derived fields `xs`/`shape` of the `grid` struct | `Grid::axis()`/`shape()` methods | No redundant serialization |
| Cell arrays (one entry per dim) | `Vec<ArrayD<f64>>` | Vector level sets (cell of cell) excepted |
| `y(:)` vectorization / `reshape` round trips | `ArrayD<f64>` throughout | Serialized only when crossing to Python |
| MATLAB 1-based dimension indices | 0-based | Uniform across the library |

Data conventions: grid function shape = node counts `n` per dimension; nodes
are cell centers `min + (i + 0.5) * dx` (as in the original,
`dx = (max - min) / n`).

### Implementation and verification status

All four phases are implemented; verification does not rely on MATLAB
reference data but uses two self-contained gates — analytic solutions and
convergence orders (22 test cases all passing: 10 unit + 12 integration):

| Phase | Content | Verification gates (measured) |
|---|---|---|
| 1 | Ghost cells ×5, first-order upwind, `ode_cfl1` | Bit-exact assertions for periodic wrap / constants / slopes / extrapolation; advecting-circle centroid error < 0.02 and area-ratio deviation < 5% (81² grid) |
| 2 | ENO2/ENO3/WENO5, GLF/LLF/LLLF, `termLaxFriedrichs`, `ode_cfl2/3` | Sin-field derivative convergence-order gates at 0.8/1.8/2.5/4.0 (measured 1.00/2.00/3.00/5.00, see `derivative.rs` unit tests); Burgers equation t=0.5 and Hopf–Lax exact solution L∞ error < 0.02 (ENO2+GLF, N=401) with convergence under refinement |
| 3 | `ReinitTerm` (with Russo-Smereka subgrid fix), `signed_distance_iterative`, all shapes and set operations | Reinitialization by 0.3× the distance function returns true distances with max error < 2.5 dx and mean < 0.5 dx (81²); Zalesak disk slit topology preserved |
| 4 | `RestrictUpdateTerm`, `PostTimestepTtrRecorder`, `TerminalEvent`, dual-integrator TTR | Reachable-set boundary misclassification vs analytic solution < 2% (101²); TTR max error < 0.45, visibly reduced at 51→101 resolution; convergence events terminate early |

The TTR's order-of-0.45 magnitude is inherent LF-family dissipation (the dual
integrator's ∂H/∂p is p-independent, making GLF/LLF/LLLF equivalent): the
analytic TTR has gradient magnitude about 2–3, so dissipation lag
∫diss·dt accumulates to exactly this scale; doubling the grid brings it down
to 0.28, consistent with the MATLAB original at equal resolution. The
WENO5/odeCFL3 combination can substantially reduce this error at higher cost.

Two error-prone spots fixed during the port (regression-guarded in tests):
- The third-order mixed-difference scale factor is `Δ²D2 / (3·dx)`, not
  `dx·Δ²D2 / 3` (coincidentally equal on h=1 grids; convergence-order tests
  catch this);
- `termReinit` assembles δ, and the ODE right-hand side must take the sign
  flip (missing it diverges exponentially).

### Integration path

- Python exposure: through e2m2e-integrators' pyo3 cdylib with a new submodule
  (e.g. `_levelset`), ABI stamp incremented in `abi-version.txt`; large arrays
  (100³ grid ≈ 8 MB/component) cross as flat `Vec<f64>` + shape, consistent
  with `_integrators`, without pyo3-numpy.
- Visualization: Python-side matplotlib contours, kept out of Rust.

## 简体中文

水平集方法与 Hamilton-Jacobi（HJ）可达性求解器，自 UBC Ian M. Mitchell
的 [Toolbox of Level Set Methods](https://www.cs.ubc.ca/~mitchell/ToolboxLS/)
（ToolboxLS 1.1，MATLAB）移植。求解含时 HJ PDE `D_t φ = -H(x, t, φ, ∇φ)`，
为低推力轨迹的可达集、最短到达时间（TTR）与追逃博弈计算提供网格 PDE 内核。

与 e2m2e 其他数值 crate 的分工：e2m2e-propagation / e2m2e-forces 沿**单条轨迹**
积分 ODE，本 crate 在**结构网格**上演化 PDE；动力学通过
[`Hamiltonian`](src/hamiltonian.rs) trait 进入（如 CR3BP 相对运动的哈密顿量），
不依赖 e2m2e 的积分器。纯数学 crate：无 SPICE、无 pyo3；对外暴露经
e2m2e-integrators 按 ABI 戳流程统一进行。

### 许可证（重要）

本 crate 派生自 ToolboxLS，原作采用 ACM 非商业许可（全文见
`LICENSE-ToolboxLS`）。学术与科研使用免版税，但**分发必须保留原版权与
许可条款**，且不得用于商业产品。因此本 crate 的 `license-file` 不随
workspace 的 Apache-2.0。若未来 e2m2e 发布包含本 crate 的二进制，需在
发布说明中注明此差异；商业使用需联系原作者（mitchell@cs.ubc.ca）。

### 模块对照

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

### 协议映射（MATLAB → Rust 的核心设计决策）

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

### 实现与验证状态

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

### 集成路径

- Python 暴露：经 `e2m2e-integrators` 的 pyo3 cdylib，新增子模块
  （如 `_levelset`），随 `abi-version.txt` 递增 ABI 戳；大数组（100³ 网格
  ≈ 8 MB/分量）以扁平 `Vec<f64>` + shape 跨界，与现有 `_integrators` 一致，
  不引入 pyo3-numpy。
- 可视化：Python 侧 matplotlib 等值线，不进 Rust。
