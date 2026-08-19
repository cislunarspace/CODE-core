# algorithm 层数值内核迁移状态清单

ADR 0011 决策：部分计算功能由 Python 执行，正在逐步迁移至 Rust 计算核心。
本文按 `e2m2e/algorithm/` 的子模块逐项登记迁移状态，供审计直接引用，避免把
"过渡状态"误读成"放错层"（误判先例与裁决见 ADR 0026）。

每个登记项回答三个问题：**数值内核在哪**（Rust crate / Python）、**迁移状态**
（已下沉 / 迁移中 / 有意留 Python）、**为什么**（有意留 Python 必写理由；
迁移中必附工作 issue）。

状态词是固定的三个，全文可 grep：

- `已下沉` — 数值内核在 Rust，Python 侧是薄封装或编排
- `迁移中` — 有明确的下沉工作项，见对应 issue
- `有意留 Python` — 有决策依据地保留在 Python

## 速查表

### 已下沉

| 模块 | 数值内核 | 工作 issue |
|---|---|---|
| `algorithm/dynamics`（传播） | Rust（`e2m2e-integrators`） | — |
| `algorithm/forces`（数值） | Rust（`e2m2e-forces`） | — |
| `algorithm/transfer/lambert.py` | Rust（`e2m2e-propagation`） | — |
| `algorithm/transfer/search_parallel.py`（网格搜索） | Rust（`e2m2e-integrators`） | — |
| `algorithm/transfer/wsb.py`（WSB 网格候选评估） | Rust（`e2m2e-forces` + `e2m2e-integrators`） | #447 |
| `algorithm/transfer/low_energy.py`（流形截面态配对） | Rust（`e2m2e-forces` + `e2m2e-integrators`） | #447 |
| `algorithm/solver/differential_correction.py`（CR3BP 数值内核） | Rust（`e2m2e-integrators`） | #441 |
| `algorithm/solver`（星历修正路径） | Rust（`e2m2e-integrators`） | — |
| `algorithm/solver/continuation.py`（PAL 数值内核） | Rust（`e2m2e-forces`） | #443 |
| `algorithm/family`（#428 轨道族数值内核） | Rust（`e2m2e-forces` + `e2m2e-integrators`） | #428 |
| `algorithm/transfer/qlaw.py`（反馈积分与 Q 函数；Python 仅组装初猜） | Rust（`e2m2e-forces` + `e2m2e-integrators`） | #442 |
| `algorithm/transfer/nsga2.py`（演化算子；Python 保留评估与编排） | Rust（`e2m2e-integrators`） | #444 |
| `algorithm/transfer/lowthrust_shooting.py`、`lowthrust_collocation.py`（直接法数值评估） | Rust（`e2m2e-integrators`；SLSQP 编排留 Python） | #445 |
| `algorithm/transfer/porkchop.py`（网格评估：终端传播 + Lambert + ΔV） | Rust（`e2m2e-forces` + `e2m2e-integrators`；Python 仅问题构造与存档/查询） | #446 |
| `algorithm/design`（打靶/传播路径） | Rust（`e2m2e-integrators`） | — |
| `algorithm/coordinate/synodic_j2000.py`（批量转换） | Rust（`e2m2e-integrators`） | — |
| `algorithm/proximity/relative_dynamics.py`（传播） | Rust（`e2m2e-integrators`） | — |
| `algorithm/station_keeping/monte_carlo.py`（传播） | Rust（`e2m2e-integrators`） | — |
| `algorithm/normal_form`（积分路径） | Rust（`e2m2e-integrators`） | #336/#340 |
| `algorithm/normal_form`（CR3BP Hamiltonian 数值构造） | Rust（`e2m2e-integrators`） | — |
| `algorithm/normal_form`（H→QF 标量多项式投影） | Rust（`e2m2e-integrators`） | — |
| `algorithm/normal_form`（数值多项式核） | Rust（`e2m2e-integrators`） | #464 |
| `algorithm/normal_form`（复值积分 + QF↔CM Lie 流） | Rust（`e2m2e-integrators`，12 实维分裂） | #465 |
| `algorithm/normal_form`（中心流形化简） | Rust（`e2m2e-integrators`） | #466 |
| `algorithm/manifold/manifolds.py`（种子生成与批量传播） | Rust（`e2m2e-forces` + `e2m2e-integrators`） | #448 |

### 迁移中

| 模块 | 数值内核 | 工作 issue |
|---|---|---|
| `algorithm/solver/MultipleShooting` 类（transfer/hohmann） | Python | 待单独迁移 |

### 有意留 Python

| 模块 | 数值内核 | 工作 issue |
|---|---|---|
| `algorithm/transfer/nlp_*`、`transfer_optimization.py`（NLP 优化与编排） | Python | — |
| `algorithm/family/*_initial_guess.py`、`strategies/`、`cr3bp_orbits.py`（问题构造与族编排） | Python（数值已 Rust） | — |
| `algorithm/family/halo_family.py`（族延拓编排） | Python（数值已 Rust） | — |
| `algorithm/transfer` 二体/解析与编排模块 | Python（Lambert 已 Rust） | — |
| `algorithm/transfer/search_geometry.py`、`search_progress.py`、`solution_database.py`（搜索辅助） | Python | — |
| `algorithm/manifold/sections.py`（截面事件函数） | Python | — |
| `algorithm/normal_form`（符号 Legendre/星历 H、NAFF、pipeline 编排） | Python | #449 |
| `algorithm/stability.py` | Python | — |
| `algorithm/station_keeping`（控制律） | Python（传播已 Rust） | — |
| `algorithm/coordinate`（单次转换） | Python | — |
| `algorithm/design`（编排） | Python（数值已 Rust） | — |
| `algorithm/design/frozen_orbit.py`（ELFO 辅助） | Python | — |
| `algorithm/dynamics/potential.py`（伪势能 Hessian） | Python | — |
| `algorithm/propagation.py` | Python（传播已 Rust） | — |
| `algorithm/proximity`（编排） | Python（传播已 Rust） | — |
| `algorithm/nominal_orbit/` | Python（占位） | — |

## 已下沉

数值内核在 Rust，Python 侧只构造问题、传参过 FFI、解释结果。审计时若看到
Python 文件里有数值循环，先查它是否只是薄封装。

**`algorithm/dynamics`（传播）。** CR3BP/BCR4BP/星历路径的传播数值在
`e2m2e-integrators` crate（`propagate_cr3bp_py`、`propagate_bcr4bp_py`、
`propagate_compiled_py` 等）。无事件时 CR3BP 直接走 Rust 快速路径，
`dynamics.py` 只做问题构造与结果解释。同包 `potential.py` 的伪势能
Hessian 是 numpy 实现（供非传播路径共用），见有意留 Python 节。
依据 ADR 0002。

**`algorithm/forces`（数值）。** 力模型数值（球谐、潮汐、SRP、三体、大气）
在 `e2m2e-forces` crate；`force_model.py` 等 Python 文件是"参数验证 +
to_rust_spec 序列化"配置面，源码留在 algorithm 层（ADR 0030）：Python 是
编排侧配置面，不是数值核；不新建 Python 数值目录。层级裁决不改变
"数值已下沉"的登记。

**`algorithm/transfer/lambert.py`。** 二体 Lambert（Izzo）在
`e2m2e-propagation` crate（`lambert.rs`），本文件是薄封装
（`lambert_izzo_py` / `lambert_batch_py`）。

**`algorithm/transfer/search_parallel.py`（网格搜索）。** 搜索阶段的评估单元
与网格分发在 `e2m2e-integrators`（`transfer_grid_search`，Rayon 并行）。
Python 侧保留 `TransferSearch` 编排、后端分发与 6 个几何 thin-wrapper
（monkeypatch 缝，见 ADR 0017）。

**`algorithm/transfer/wsb.py`（WSB 网格候选评估）。** TLI 参数化、BCR4BP
传播、近月点检测与插值、H2、到达态/Delta-v 计算和候选筛选在
`e2m2e-forces` 的纯 Rust 核执行，经 `e2m2e-integrators` 用 Rayon 分发。
Python 保留系统/参数校验和领域结果组装；默认 Rust，Python 仅作显式等价性
对照，绝不自动回退。工作项：#447。

**`algorithm/transfer/low_energy.py`（流形截面态配对）。** 两组截面态的
笛卡尔积、位置/速度范数、加权拼接代价和稳定排序在 `e2m2e-forces` 的
纯 Rust 核执行，经 `e2m2e-integrators` 暴露。流形管管理、四分支编排和
ThreeBodyLambert 闭合仍在 Python；流形种子、STM 转运和管传播已由 #448 下沉。
默认 Rust，Python 仅作显式等价性对照。工作项：#447。

**`algorithm/transfer/qlaw.py`。** Q-law 反馈律积分、开普勒根数转换、Gauss
方程、Q 函数与推力方向在 Rust 内核完成，经 `qlaw_propagate_py` 与
`qlaw_segment_direction_py` 暴露。Python 侧只解析动力学参数、从连续轨迹
重采样并组装 `LowThrustSegment`；独立公开的 `rv_to_keplerian` 保持既有兼容
行为，不构成反馈积分降级路径。

**`algorithm/transfer/lowthrust_shooting.py`、`lowthrust_collocation.py`。**
低推力直接打靶的多段受控传播、灵敏度链式组装，以及 Hermite-Simpson 配点的
批量缺陷求值在 `e2m2e-integrators` 的 Rust 入口完成。Python 侧保留问题构造、
SLSQP 外层编排、初猜与结果解释；`backend="python"` 提供原有实现作为等价性
对照和降级路径。对照测试见
`tests/algorithm/transfer/test_lowthrust_rust_backend.py`。迁移完成于 #445。

**`algorithm/transfer/porkchop.py`。** 网格评估（终端传播 + Lambert +
ΔV 组装 + Rayon 分发）在 `e2m2e-forces` 的 Rust 核完成，经
`e2m2e-integrators` 暴露。Python 只做问题构造与结果解释：内置终端
（`OrbitTerminal`/`StateTerminal` + 未 patch 的 `CR3BP_Dynamics`）把终端
规格直接交给 Rust（轨道终端传播同步下沉）；自定义终端或 patch 场景经
`get_arrival_state` 协议提取状态网格后交同一 Rust 核，无 Python 数值
回退（#378）。SQLite 存档、插值查询、Pareto 前沿留 Python。对照测试见
`tests/algorithm/transfer/test_porkchop_rust_backend.py`。迁移完成于 #446。

**`algorithm/solver`（星历修正路径）。** 多重打靶迭代
`multiple_shooting_correct_py` 在 Rust，segmented 与稳定轨道修正默认走它。
`MultipleShooting` 类（transfer/hohmann 仍使用）本身还是 Python 实现，见
迁移中（待单独立项）。

**`algorithm/solver/differential_correction.py`（CR3BP 数值内核）。** 对称性
策略与 Orbit 编排留在 Python；半周期对称和全周期闭合的残差、STM 雅可比、
Newton 修正、线性求解与收敛状态机经
`differential_correction_cr3bp_py` 在 Rust 执行。Python 不再保留微分修正
数值后端；CR3BP 修正入口统一走 Rust。

**`algorithm/transfer/nsga2.py`。** 约束非支配排序、拥挤度距离、锦标赛和
环境选择、SBX 交叉及多项式变异在 `e2m2e-integrators` 的 `nsga2` 模块，经
`nsga2_*_py` 暴露。Python 保留目标函数回调、`ProcessPoolExecutor` 并行评估、
NumPy 随机数生成、逐代评估与 `NSGA2Result` 组装；`backend="python"` 保留原
实现作对照与降级。随机抽样由 Python 按既有条件分支顺序生成并交给 Rust，因此
两后端同种子演化等价。工作项：#444。

**`algorithm/solver/continuation.py`（PAL 数值内核）。** 伪弧长延拓的 XZ
对称约束 F/dF 组装、切向量（零空间）与 PAL 牛顿迭代在 `e2m2e-forces`
crate（`pal_continuation` 模块），经 `pal_f_df_tangent_py` /
`pal_newton_step_py` 暴露。`pseudo_arclength_continuation` 双后端：默认
rust，`backend="python"` 走 numpy 参照路径（对照与降级；等价性对照见
`tests/algorithm/design/continuation/test_halo_pal_rust_equivalence.py`）。
初始切向量两后端统一由 Python 参照计算（零空间符号约定在 SVD 与 Rust
广义叉积间无保证，首步延拓方向须由同一实现锁定）。外层逐轨编排（物理合理性
检查、方向反馈、停滞检测）留 Python，微分修正数值内核已下沉，见已下沉
#441。`family/halo_family.py` 是纯编排，无独立数值内核，登记在有意留
Python 节。

**`algorithm/family`（#428 轨道族数值内核）。** 七族统一走
`generate_cr3bp_family_py`：一次 Rust 调用完成种子构造、CR3BP 修正、PAL、
步长控制、成员筛选、结构化终止和部分族保留。其内部复用
`differential_correction_cr3bp_py` 与 `planar_full_period_pal_py` 背后的纯
Rust 实现；共线点求根、线性中心模态、Lissajous 非线性中心约化多点轨迹，
以及近月距、
L4/L5 径向振幅与面外振幅扫描也在同一模块内执行。Python 不逐成员调用
数值原子，不保留数值回退，只负责请求校验、领域分派和 `OrbitFamily` 重包。
工作项：#428。

**`algorithm/design`（打靶/传播路径）。** 分段修正、多重打靶、段传播、
时间转换走 Rust（`segmented_shooting_correct_py`、
`multiple_shooting_correct_py`、`propagate_segments_py`、
`batch_et_to_utc_py`）；`design_orbit.py` 是任务编排。

**`algorithm/coordinate/synodic_j2000.py`（批量转换）。** 会合系↔J2000
批量转换走 Rust（`batch_synodic_to_j2000_py`、`batch_j2000_to_synodic_py`）。

**`algorithm/proximity/relative_dynamics.py`（传播）。** 相对动力学传播走
Rust（`solve_ivp_events`）。

**`algorithm/station_keeping/monte_carlo.py`（传播）。** 蒙特卡洛采样的
传播走 Rust（`propagate_compiled_stm_py`）。

**`algorithm/normal_form`（积分路径）。** 传播积分走 Rust（`solve_ivp_rust`，
见 #336/#340）。QF↔CM 高阶 Lie 流走 `qf_to_cm_py` / `cm_to_qf_py`（#465，
12 实维分裂复积分）；`backend="python"` 仅作显式对照。

**`algorithm/normal_form`（CR3BP Hamiltonian 数值构造）。** 共线点
`build_cr3bp_hamiltonian` 走 `build_cr3bp_hamiltonian_py`（JM `c_n` 形式，
ms 级）；三角点无该输入语义，保留 sympy 符号路径。

**`algorithm/normal_form`（H→QF 标量多项式投影）。** CR3BP 标量系数的
`project_hamiltonian_to_qf` 走 `project_hamiltonian_qf_py`（multinomial
数值展开）；星历时间序列系数回退 sympy。

**`algorithm/normal_form`（数值多项式核）。** `poly_poisson` /
`poly_simplify` / `polylist_simplify` 及核内幂次工具（`keys_by_order` /
`trim_degree`）走 `e2m2e-integrators` 的 `poly_*_py` 绑定；完整支持标量与
一维时间序列、实/复系数。Python 侧为薄封装，默认 `backend='rust'`，
`backend='python'` 仅作显式等价性对照（不静默回退）。sympy 符号系数路径
仍留 Python。工作项：#464。

**`algorithm/normal_form`（复值积分 + QF↔CM Lie 流）。** `qf_to_cm` /
`cm_to_qf` 默认走 `qf_to_cm_py` / `cm_to_qf_py`（12 实维分裂 + DOP853，
完整实↔复基底与全阶 Lie）；`backend="python"` 仅作显式对照。工作项：#465。

**`algorithm/normal_form`（中心流形化简）。** `CenterManifoldReducer.reduce`
默认走 `center_manifold_reduce_py`：两步 Lie 同调（invariant / center）、
实/复两套频域 W、MAD 抑制、`list_deriv`、全阶 Poisson 链与虚/实基底变换
完整在 Rust；路径内嵌 Poisson 链所用多项式运算，并与 #464 包级 `poly_*`
并存。`backend="python"` 仅显式对照，禁止静默降级（ADR 0020）。工作项：
#466。

**`algorithm/manifold/manifolds.py`（种子生成与批量传播）。** 单值矩阵特征
分解、STM 转运、±ε 种子与批量弧传播调度在 `e2m2e-forces` 的 `manifold`
模块，经 `manifold_seeds_py` / `manifold_propagate_py` 暴露。Python 只做
参数校验、`ManifoldTube`/`Orbit` 组装与可选的事后截面截断；不保留 Python
数值回退。工作项：#448。

## 迁移中

ADR 0011 明示的过渡状态，每个条目有独立工作项。`MultipleShooting` 是支持多种
动力学模型的泛型 Python 类，不能复用 CR3BP 微分修正内核；后续需单独评估和迁移。

**`algorithm/solver/MultipleShooting` 类。** transfer/hohmann 使用的多重
打靶类仍是泛型 Python 实现，后续需单独评估和迁移。

后置未派发（#449 评估）：quasi-Floquet 全矩阵法（P5）、多重打靶 Newton
壳（P6）；独立 FFT 产品化（P2）——#466 已内嵌特解所需 FFT，可不单开。

## 有意留 Python

有决策依据地保留在 Python，不是技术债。审计时看到这些模块的 Python 数值，
按对应理由判定，不误报"放错层"。

**`algorithm/transfer/nlp_core.py`、`nlp_scipy.py`、`nlp_copt.py`、
`transfer_optimization.py`（NLP 优化与编排）。** 理由：SLSQP/COPT 串行迭代是
Python 强项（早期架构讨论共识，ADR 0017 边界固化）；
`transfer_optimization.py` 是"搜索-优化"两步法优化阶段的高层编排（构造优化器、
计算目标/约束），属 NLP 范畴。这是默认求解器所在，不是迁移目标。

**`algorithm/family/*_initial_guess.py`、`strategies/`、`cr3bp_orbits.py`
（问题构造与族编排）。** 理由：选择轨道族、校验固定方向/采样规则并把
Rust 结果重包为领域对象属于编排模块职责（architecture.md 第 3 节）。
#428 的种子、传播、修正、PAL、步长、筛选、中心模态、Lissajous 采样和族
度量均已收进单次 Rust 调用；Python 不提供数值回退。

**`algorithm/family/halo_family.py`（族延拓编排）。** 理由：种子生成、
逐轨微分修正调用、方向反馈、停滞检测与族组装是编排职责
（architecture.md 第 3 节）；调用的 PAL 数值内核经 `continuation.py`
已下沉（见已下沉 #443），本文件自身无数值迭代。

**`algorithm/transfer` 二体/解析与编排模块。** `hohmann.py`、`multi_impulse.py`、
`lga.py`、`three_body_lambert.py`、`mission_assessment.py`、`cost.py`、
`propulsion.py`、`terminal.py`、`transfer.py`（NLP 编排）、`config.py`。
理由：解析公式或编排，无"喂进数字就迭代"的热路径；其中 Lambert 求解已
Rust。多脉冲 NLP 的节点优化属 NLP 范畴，见上。

**`algorithm/transfer/search_geometry.py`、`search_progress.py`、
`solution_database.py`（搜索辅助）。** 理由：`search_geometry.py` 是几何核的
numpy 纯函数实现，ADR 0017 边界固化的 thin-wrapper / numpy 对照基准
（monkeypatch 缝），必须与 Rust 几何核并存；`search_progress.py` 是 tqdm
进度封装；`solution_database.py` 是解库查询与筛选（数据管理性质）。

**`algorithm/stability.py`。** 理由：单值矩阵/Floquet 乘子分析是 numpy 特征
分解，无性能热路径；传播经 `dynamics` 已 Rust。未列入迁移清单。

**`algorithm/manifold/sections.py`（截面事件函数）。** 理由：庞加莱截面事件
函数定义（穿越检测交给 `Dynamics.propagate(events=...)`，由积分器步内
定位，传播已 Rust），本身无独立数值迭代。

**`algorithm/normal_form`（符号 Legendre/星历 H、NAFF、pipeline 编排）。**
理由（#449 评估）：

- **符号构造不是数值热路径。** sympy Legendre / 星历 `build_hamiltonian`
  属 CAS；共线点 CR3BP 数值构造已 Rust。无对等「下沉收益」。
- **NAFF** 是外部可执行文件封装，不进 Rust crate。
- **pipeline / catalog 编排** 是串联与结果组装；多项式核、QF↔CM 与
  中心流形化简已下沉（#464/#465/#466），不单独迁编排层。
- 数值主链三项均已下沉；quasi-Floquet 全矩阵法与 MS Newton 壳后置
  未派发。正确性不以 qiao 为 oracle（#426）。

**`algorithm/design/frozen_orbit.py`（ELFO 辅助）。** 理由：经典根数↔笛卡尔
转换是解析公式，漂移统计是编排；传播已 Rust（查询已下沉 Rust 实例）。

**`algorithm/dynamics/potential.py`（伪势能 Hessian）。** 理由：解析导数公式
的 numpy 实现，供动力学方程与稳定性分析等非传播路径共用，单次调用无热
路径；传播路径的加速度计算已 Rust。

**`algorithm/station_keeping`（控制律）。** `controller.py`、
`momentum_management.py`、`special_point.py`、`target_point.py`、
`error_models.py`。理由：控制律与编排，无热路径数值迭代；传播/STM 已 Rust
（`monte_carlo.py` 用 `propagate_compiled_stm_py`）。

**`algorithm/coordinate`（单次转换）。** 理由：单次标量转换无性能热路径；
批量路径已 Rust（`synodic_j2000.py`）。coordinate 的层级归属由 ADR 0026
决策 1 裁决（留在 algorithm 层），与本清单无关。

**`algorithm/design`（编排）。** 理由：任务编排职责；数值（打靶/传播）已
Rust，见上。

**`algorithm/propagation.py`。** 理由：单段预报编排（ADR 0011：不建独立
编排器），配 `ForceModel` + 调传播 + 输出 `EphemerisTable`；传播数值已 Rust。

**`algorithm/proximity`（编排）。** `phasing.py`、`safety.py`。理由：编排；
相对动力学传播已 Rust（`relative_dynamics.py`）。

**`algorithm/nominal_orbit/`。** 理由：当前为占位实现（插值器待 FR1 落地，
包 docstring 明示），无实际数值可下沉。

## 维护说明

- **状态词固定**：新登记项只能用"已下沉 / 迁移中 / 有意留 Python"三词，
  保证全文 grep 可枚举。
- **粒度**：登记粒度到文件/路径。同一模块可拆多条（如 `algorithm/design`：
  打靶/传播路径已下沉、编排有意留 Python；`algorithm/normal_form`：积分/
  CR3BP Hamiltonian/H→QF/多项式核/QF↔CM/中心流形已下沉，符号与
  NAFF/编排有意留 Python），以速查表路径为准。
- **迁移中条目**：issue 关闭（下沉完成或改判）时，把条目移到对应状态节，
  保留 issue 编号作为历史指针。
- **有意留 Python 条目**：必须带理由，理由应引用 ADR 或文档决策，不写
  "暂时不迁"这类临时话。
- **新模块**：`e2m2e/algorithm/` 下出现含数值实现的新子模块时，在本清单
  登记后再合入。
- 关联 ADR：0011（五层架构）、0017（网格搜索下沉 Rayon）、0026（测试
  套件层级澄清，后续工作第三条即本清单的由来）。
