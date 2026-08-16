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
| `algorithm/solver`（星历修正路径） | Rust（`e2m2e-integrators`） | — |
| `algorithm/solver/continuation.py`（PAL 数值内核） | Rust（`e2m2e-forces`） | #443 |
| `algorithm/design`（打靶/传播路径） | Rust（`e2m2e-integrators`） | — |
| `algorithm/coordinate/synodic_j2000.py`（批量转换） | Rust（`e2m2e-integrators`） | — |
| `algorithm/proximity/relative_dynamics.py`（传播） | Rust（`e2m2e-integrators`） | — |
| `algorithm/station_keeping/monte_carlo.py`（传播） | Rust（`e2m2e-integrators`） | — |
| `algorithm/normal_form`（积分路径） | Rust（`e2m2e-integrators`） | — |

### 迁移中

| 模块 | 数值内核 | 工作 issue |
|---|---|---|
| `algorithm/solver/differential_correction.py`、`MultipleShooting` 类 | Python | #441 |
| `algorithm/transfer/qlaw.py` | Python（步进已 Rust） | #442 |
| `algorithm/transfer/nsga2.py` | Python | #444 |
| `algorithm/transfer/lowthrust_shooting.py`、`lowthrust_collocation.py` | Python（传播已 Rust） | #445 |
| `algorithm/transfer/porkchop.py` | Python（Lambert 已 Rust） | #446 |
| `algorithm/transfer/wsb.py`、`low_energy.py`（WSB/低能转移） | Python（传播已 Rust） | #447 |
| `algorithm/manifold/manifolds.py` | Python | #448 |
| `algorithm/normal_form`（FFT/多项式/化简、多重打靶 Newton） | Python（积分已 Rust） | #449 |

### 有意留 Python

| 模块 | 数值内核 | 工作 issue |
|---|---|---|
| `algorithm/transfer/nlp_*`、`transfer_optimization.py`（NLP 优化与编排） | Python | — |
| `algorithm/family/*_initial_guess.py`、`strategies/`、`cr3bp_orbits.py`（初猜生成） | Python | — |
| `algorithm/family/halo_family.py`（族延拓编排） | Python | — |
| `algorithm/transfer` 二体/解析与编排模块 | Python（Lambert 已 Rust） | — |
| `algorithm/transfer/search_geometry.py`、`search_progress.py`、`solution_database.py`（搜索辅助） | Python | — |
| `algorithm/manifold/sections.py`（截面事件函数） | Python | — |
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
to_rust_spec 序列化"配置面。`forces` 的层级归属（数值层配置面还是算法层
力模型定义）由 #429 独立评估——那是层级议题，不改变"数值已下沉"的
登记。

**`algorithm/transfer/lambert.py`。** 二体 Lambert（Izzo）在
`e2m2e-propagation` crate（`lambert.rs`），本文件是薄封装
（`lambert_izzo_py` / `lambert_batch_py`）。

**`algorithm/transfer/search_parallel.py`（网格搜索）。** 搜索阶段的评估单元
与网格分发在 `e2m2e-integrators`（`transfer_grid_search`，Rayon 并行）。
Python 侧保留 `TransferSearch` 编排、后端分发与 6 个几何 thin-wrapper
（monkeypatch 缝，见 ADR 0017）。

**`algorithm/solver`（星历修正路径）。** 多重打靶迭代
`multiple_shooting_correct_py` 在 Rust，segmented 与稳定轨道修正默认走它。
注意 `MultipleShooting` 类（transfer/hohmann 仍使用）本身还是 Python
实现，见迁移中 #441。

**`algorithm/solver/continuation.py`（PAL 数值内核）。** 伪弧长延拓的 XZ
对称约束 F/dF 组装、切向量（零空间）与 PAL 牛顿迭代在 `e2m2e-forces`
crate（`pal_continuation` 模块），经 `pal_f_df_tangent_py` /
`pal_newton_step_py` 暴露。`pseudo_arclength_continuation` 双后端：默认
rust，`backend="python"` 走 numpy 参照路径（对照与降级；等价性对照见
`tests/algorithm/design/continuation/test_halo_pal_rust_equivalence.py`）。
初始切向量两后端统一由 Python 参照计算（零空间符号约定在 SVD 与 Rust
广义叉积间无保证，首步延拓方向须由同一实现锁定）。外层逐轨编排（微分
修正、物理合理性检查、方向反馈、停滞检测）留 Python；自然参数延拓的
步进循环本身是编排，其热路径为微分修正，见迁移中 #441。
`family/halo_family.py` 是纯编排，无独立数值内核，登记在有意留
Python 节。

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
见 #336/#340）；FFT/多项式/化简、多重打靶 Newton 迭代仍是 Python，见
迁移中 #449。

## 迁移中

ADR 0011 明示的过渡状态，每个条目有独立工作 issue，下沉方式照 ADR 0017
先例（双后端共存 + 等价性对照，Python 路径保留作对照与降级）。

**`algorithm/solver/differential_correction.py` 与 `MultipleShooting` 类。**
单段微分修正与 `MultipleShooting` 类是纯 Python；星历修正路径已有 Rust 版
（见上），单段修正可评估与 `multiple_shooting_correct_py` 共用基础设施。
工作项：#441。

**`algorithm/transfer/qlaw.py`。** Q-law 反馈律的积分步进已 Rust
（`rk_step`），Q 函数评估与反馈循环仍是 Python。ADR 0011 明示"Q-law"仍由
Python 执行。工作项：#442。

**`algorithm/transfer/nsga2.py`。** 非支配排序、拥挤度、演化循环纯 Python。
ADR 0017 边界明确 nsga2"后续单独迁移"。工作项：#444。

**`algorithm/transfer/lowthrust_shooting.py`、`lowthrust_collocation.py`。**
低推力打靶/配点的迭代求解是 Python；底层 7D 可变质量传播已 Rust
（`propagate_compiled_lowthrust`、`augmented_eom_7d_py`）。ADR 0011 明示
"低推力打靶"仍由 Python 执行；ADR 0017 边界列 low-thrust 后续迁移。
工作项：#445。

**`algorithm/transfer/porkchop.py`。** 网格循环与几何评估是 Python；
网格上的 Lambert 求解已 Rust（`lambert_batch_py`）。ADR 0017 边界列
porkchop 后续迁移，可复用 `transfer_grid_search` 的 Rayon 分发范式。
工作项：#446。

**`algorithm/transfer/wsb.py`、`low_energy.py`（WSB/低能转移）。**
`wsb.py` 的 BCR4BP 弹道搜索（sun_phase × departure_phase × tof 三维网格，
ProcessPoolExecutor 并行）与 `low_energy.py` 的流形管配对、拼接代价评估
是 Python；底层传播已 Rust。ADR 0017 边界列 wsb 后续迁移，可复用
`transfer_grid_search` 的 Rayon 分发范式。工作项：#447。

**`algorithm/manifold/manifolds.py`。** 单值矩阵特征分解、STM 转运、种子
生成纯 Python（numpy）。ADR 0026 后续工作第三条点名的过渡状态之一。
工作项：#448。

**`algorithm/normal_form`（FFT/多项式/化简、多重打靶 Newton）。** 频率提取、
Legendre 系数、多项式环、Hamiltonian 化简、`multiple_shooting.py` 的块三对角
Newton 迭代等是 Python（numpy/sympy 惰性导入）；积分路径已 Rust（见上）。
是否下沉需先评估（研究性算法链，受 #426"qiao 对拍不在范围"约束），故为
评估任务。工作项：#449。

## 有意留 Python

有决策依据地保留在 Python，不是技术债。审计时看到这些模块的 Python 数值，
按对应理由判定，不误报"放错层"。

**`algorithm/transfer/nlp_core.py`、`nlp_scipy.py`、`nlp_copt.py`、
`transfer_optimization.py`（NLP 优化与编排）。** 理由：SLSQP/COPT 串行迭代是
Python 强项（`architecture-design-discussion.md` 共识，ADR 0017 边界固化）；
`transfer_optimization.py` 是"搜索-优化"两步法优化阶段的高层编排（构造优化器、
计算目标/约束），属 NLP 范畴。这是默认求解器所在，不是迁移目标。

**`algorithm/family/*_initial_guess.py`、`strategies/`、`cr3bp_orbits.py`
（初猜生成）。** 理由：初猜是领域决策，天天在变，属"编排模块"职责
（architecture.md 第 3 节）；Richardson 三阶等解析近似、`cr3bp_orbits.py` 的
族行走割线法都是设计链路初猜段的单次标量迭代，无热路径。族延拓（PAL
数值内核）已下沉，见已下沉 #443。

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
  打靶/传播路径已下沉、编排有意留 Python；`algorithm/normal_form`：积分
  路径已下沉、算法链迁移中），以速查表路径为准。
- **迁移中条目**：issue 关闭（下沉完成或改判）时，把条目移到对应状态节，
  保留 issue 编号作为历史指针。
- **有意留 Python 条目**：必须带理由，理由应引用 ADR 或文档决策，不写
  "暂时不迁"这类临时话。
- **新模块**：`e2m2e/algorithm/` 下出现含数值实现的新子模块时，在本清单
  登记后再合入。
- 关联 ADR：0011（五层架构）、0017（网格搜索下沉 Rayon）、0026（测试
  套件层级澄清，后续工作第三条即本清单的由来）。
