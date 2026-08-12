# ADR 0020：失败处理策略——确定性失败抛异常，搜索不可行带标记，禁止隐式降级

**状态**：已采纳（决策 3 经 ADR 0024 修订）
**日期**：2026-08-09
**关联**：ADR 0002（多处 scipy 回退被本 ADR 修订）、ADR 0003（坐标层"绝不自动降精度"——本 ADR 的直接前身）、ADR 0009、ADR 0014（决策 4 错误翻译）、ADR 0016、ADR 0017、ADR 0019

## 背景

一次跨 `algorithm/` + `data/` + `api/` 的健壮性盘点（四路并行扫描）找出约 139 处"失败发生时，做了抛异常或带标记返回以外的事"的代码——静默返回近似值、自动换后端、放宽容差并报告成功、把失败藏进成功统计。它们分散在各层，但同源：**把"失败"当成"一种可接受的备选结果"，而不是"需要上抛或显式标记的事件"**。

几条最典型的（完整清单见 `archive/plans/robustness-cleanup.md`）：

- **步长塌缩被静默吞掉**：`dynamics.py:611-633` 捕获 Rust 的 "step size collapsed" 错误（靠 `dynamics.py:54` 的字符串匹配），返回空 states；`propagate_orbit_state_at_time`（`dynamics.py:688-699`）拿到空 states 后退回轨道自身数据的插值，当成功结果返回。
- **谎报收敛**：`differential_correction.py:730-744`，牛顿修正停滞（修正量 < 1e-14）但残差仍有 1e-8 时，直接标 `converged=True`——配置容差是 1e-12，等于静默放宽容差 4 个数量级。
- **失败藏进成功统计**：`monte_carlo.py:484-511`，控制律返回 None（没收敛/没找到穿越点）时，样本 `failed_k = False`，保持策略的 Δv 统计系统性偏低。
- **六种互不兼容的失败标记方言**：`MultipleShootingResult.converged` / `TransferSolution.converged` / `TransferOptimizationResult.success` / `Orbit.correction_success`（bool|None 三态）/ DC 的 implicit None / 网格搜索的 `success:bool + 自由字符串 status`。后果：`search_parallel.py:189-198` 网格搜索碰撞格 `success` 硬编码 True，碰撞格被当有效解画进 Δv-Time 图（`tools/viz/transfer.py:78`）。
- **资源降级链**：`spice_optional=True` 三级链（`normal_form/pipeline` → `dynamical_substitution` → `quasi_floquet`）在 SPICE 不可用时静默把物理模型从完整星历换成纯 CR3BP；`nlp_copt.py` COPT 不可用自动回退 SLSQP；`_HAS_RUST_*` 导入门控在 Rust 不可用时回退 scipy。

航天轨道力学是确定性的：同样的初值和力模型，结果唯一。这些"健壮性"代码的代价是——**跑出来的结果可能被悄悄改过，而调用方无信号**。

## 统一原则

**行为由显式输入决定，不隐式降级。** 失败要么上抛（确定性过程），要么带统一标记返回（搜索过程）；不存在"失败了一种、返回了另一种、调用方看不出区别"的中间态。

## 决策

### 决策 1：失败分三类，处置不同

| 类别 | 含义 | 处置 |
|---|---|---|
| 确定性传播失败 | 积分发散、步长塌缩到机器精度地板、雅可比算不出 | 抛 `PropagationFailure`（决策 2） |
| 搜索/优化不可行 | 网格搜索某格发散、NLP 候选不可行、微分修正单步不收敛 | 带统一标记返回（决策 3） |
| 红线（任何类都禁） | 谎报成功、把失败藏进成功统计、静默换物理模型 | 一律改掉，无例外 |

判别关键不是"代码在哪一层"，而是**调用方是否能把"我得到了想要的结果"和"我没得到"区分开**。能区分的带标记返回是合规的；不能区分的（谎报 `success=True`、返回近似值无标志、implicit None 原因丢失）是红线。

### 决策 2：确定性传播失败的语境化语义

"步长塌缩→抛异常"是一条过于粗糙的规则。按字面它会禁掉自适应积分的标准行为、杀掉合法的引力辅助轨迹。精确化分三级：

1. **步拒绝**（error > tol，但 h 仍大于机器精度地板）：自适应控制器的标准行为——拒绝该步、缩小 h、重试（`cr3bp.rs:260-277`、`solve_ivp.rs:251-303`、`force_model.py` 的 RK 循环）。**不是失败，不报告，不计为"回退"。**
2. **步塌缩到机器精度地板 / 雅可比算不出 / 真发散**：传播器抛 **`PropagationFailure`**（新建类型异常，见"结果"）。取代当前 `dynamics.py:54` 靠字符串匹配 `"step size collapsed"` 的脆弱 catch——消息改一个字就断。
3. **语境决定汇报语义**：同一次 `PropagationFailure`，用户直接调 `propagate` → 上抛；搜索/优化 wrapper（网格搜索、NLP、多重打靶段积分）→ catch，转成 `status=INFEASIBLE/DIVERGED` 带标记返回。**传播器自身不假设语境**——由调用方决定 catch 还是 re-raise。

机器精度地板（`MIN_STEP = 1e-12·span`、`10·EPSILON·(1+|t|)`）必须保留并显式承认：它是循环守卫（防真发散时 h→0 死循环），不是隐瞒失败。地板值须可观测、进结果对象。**禁止的是物理量级地板**——如 `qlaw.py:379-380` 的 `if h<1e-6: h=step` 把被拒步长重置回原步长（触发后空转到 200 万步几乎不推进 t，最后用未经验收的中间态拼出控制律返回），或 `force_model.py:804` 的 `h=max(h,min_step)` 强行接受。删除物理量级地板后，Python 路径与 Rust `solve_ivp.rs:244-248` 的范式对齐（min_step 只用作失败判定，绝不做抬升）。

> `dynamics.py:610-633` 与 `transfer_grid_search.rs:157-187` 在搜索语境 catch 步塌缩转 infeasible，是合规的；要改的不是"别 catch"，而是把返回值从"空 states 靠 `len==0` 嗅探"改成"带 failure 标记的结构化结果"。

### 决策 3：搜索不可行的统一失败标记

"带 converged=False 返回"不够——必须钉死标记的形状，否则各模块各搞方言，调用方被 None/False/缺标志坑。现有六种方言（见背景）已经让碰撞格谎报 success=True。

以现有 `ConvergenceState` 枚举（`e2m2e/data/templates/enums.py`，含 ITERATING/CONVERGED/DIVERGED/STAGNATED/MAX_ITERATIONS）为锚，**扩充 `INFEASIBLE` 与 `COLLISION`**，规定：

- 所有搜索/修正结果暴露同名的 `status: ConvergenceState` 字段 + `cause: str`（失败原因）。
- **失败时也必须返回结果对象**（携带 status + cause）。禁止"成功返回对象、失败返回 None"的不对称签名——`DifferentialCorrection.iterate_correction` 的 `Orbit | None` 改为始终返回携带 status 的结果（Orbit 作为其字段），`termination_reason` 跟对象走而非留在 solver 上。
- 碰撞是独立的 status 枚举值（COLLISION），不是 `success=True + status 字符串`。`search_parallel.py:189-198` 的碰撞格必须进失败侧。
- 废除 `bool | None` 三态、自由字符串 status、implicit None。`converged`/`success`/`correction_success` 作为 `status == CONVERGED` 的派生属性可兼容保留，但不得是唯一信号。

先例：`design_orbit.py`（所有不收敛路径抛 `DesignNotConvergedError`）、`multiple_shooting.py`（`MultipleShootingResult(converged=False, status=...)`）已是仓内正确范式，本决策推广之。

> **修订（2026-08-11，ADR 0024）**：布尔兼容投影不再保留——`success`/`converged`/`correction_success` 一律移除，不设运行时兼容层；`ConvergenceState` 在本决策的 `INFEASIBLE`/`COLLISION` 之外另增 `FAILED`。上文“作为派生属性可兼容保留”一句作废，其余条款不变。

### 决策 4：禁止隐式资源降级，区分两种不可用

资源（SPICE/Rust/COPT）不可用时"自动换后端"是本 ADR 要消除的核心模式。但不可用有两种，处置不同：

- **资源缺失**（没装/没构建）：**报错**。spice 现为默认 feature、`make dev` 标准化、release wheel 已带 spice——正常操作中这些资源恒在，缺失即环境没搭好，不是"悄悄换慢路径"的理由。修订：ADR 0002 的"Dynamics 基类 Rust 不可用回退 scipy""COPT 不可用回退 SLSQP""无 spice 静默降级"、ADR 0009 的 release try/except、ADR 0016 的缓存 miss 回退 cspice、ADR 0017 的"显式 rust 不可用回退 processes"、ADR 0019 的 SPICE 缺失降 ITRFApproxAxes——全改报错。
- **能力缺失**（后端在，但某功能未实现/语义未对齐）：**显式 `backend="scipy"` / `backend="rust"` 参数**，二选一；不传则报错（迁移期可给 deprecation warning，下个 major 移除）。**不允许 `backend="auto"`**——auto 仍是代码替用户决定后端，属于隐式。典型：CR3BP/BCR4BP 事件检测 Rust 语义未对齐 scipy（ADR 0002 "事件检测"条），属能力缺失，走显式 backend。

**测试注入缝豁免**：ADR 0017 的 monkeypatch 回退（测试 `setattr` 注入合成轨迹时回退 Python，让注入生效）是测试基础设施，不是生产降级，不在禁止之列——但须限定在测试路径（`_geometry_methods_monkeypatched` 检测），生产路径不触发。

### 决策 5：奇点正则化与碰撞终止分离

"不做距离钳位"会把两件不相关的事一起删掉，后果是 Hessian 爆炸。精确化：

- **机器精度正则化保留**：`MIN_DISTANCE ≈ 1e-10 LU`（≈3.8cm，远在任何天体半径内）防引力 1/rⁿ 奇点的除零 NaN，存在于 `potential.py:11`、`dynamics.py:84`、`cr3bp.rs:19`、`bcr4bp.rs:22`、`nbody_stm.rs:27`。Hessian 含 1/r⁵ 项（`potential.py:42-50`），删了它接近主天体就 inf/NaN。这是数值守卫，不是物理谎言。**全部保留。**
- **物理量级钳位改碰撞终止**：撞天体半径（地球 R≈6378km、月球 R≈1737km）→ 事件检测 `g = |r| - R_body`，`terminal=True`，或抛异常。`transfer_geometry.rs:211` 的 `check_collision` 已有 post-hoc 扫描；core propagation 需补 event-based 版本。
- CR3BP 是质点模型，没有内禀天体半径——碰撞终止需要**从外部注入 body-radius 配置**。这是新功能，不是删旧行为。
- 措辞从"不做距离钳位"改为"**不做天体半径以内的距离钳位**"。

## 既有 ADR 的修订

| ADR | 原决策 | 改为 | 类别 |
|---|---|---|---|
| 0002 | Dynamics 基类 Rust 不可用回退 scipy（"构建失败降级路径"） | Rust 不可用即报错 | 资源缺失 |
| 0002 | COPT 不可用回退 SLSQP | 报错，NLP 后端显式指定 | 资源缺失 |
| 0002 | 无 spice 静默降级慢路径 | 报错 | 资源缺失 |
| 0002 | CR3BP/BCR4BP events 传 scipy 回退 | 显式 `backend="scipy"/"rust"`，不 auto | 能力缺失 |
| 0009 | release 不带 spice，try/except 静默降级 | 报错（release 已带 spice，降级机制删除） | 资源缺失 |
| 0016 | 缓存 miss 静默回退 cspice FFI | 报错或显式指定（Strict 模式从"并行专用"推广为默认） | 资源缺失 |
| 0017 | 显式选 rust 但 Rust 不可用回退 processes | 报错（测试 monkeypatch 缝豁免） | 资源缺失 |
| 0019 | SPICE 缺失 drag 帧旋转降 ITRFApproxAxes | 报错或显式指定低精度后端 | 资源缺失 |

> 注：ADR 0002 第 96 行原称 BCR4BP 传 events 抛 NotImplementedError（#333），实际代码（`bcr4bp_dynamics.py:204-212`）已改为 warn + 回退 scipy——ADR 与代码已不一致，本 ADR 一并澄清为"能力缺失，走显式 backend"。

## 理由

1. **方向有先例，不是凭空而来。** ADR 0003 第 7 条「错误明确，绝不自动降精度；钳位需显式选项」早在坐标层确立了"缺失即报错、钳位需显式"的范式——本 ADR 是把它推广到全局。ADR 0004「不可序列化大声报错」、0018「雅可比接口强制三元组，让静默出错变编译不过」是决策 4（禁止谎报）的先例；ADR 0014 决策 4「异常在 api/ 翻译成 OrbitError(code/message/details)」是决策 2（抛异常）的下游出口；ADR 0016 Strict 模式「miss 硬失败」是决策 4（资源缺失报错）的先例。
2. **粗措辞会杀合法行为——对抗验证排除的三个反面。**
   - "步长塌缩→抛异常，不回退不设地板"按字面会禁掉自适应步拒绝-缩步-重试（标准 RK 行为），且与"去钳位"叠加后，月球低高度掠过（r₂≈1e-3、距月心 384km、未撞月面的合法 gravity assist）会被报成积分失败。决策 2 的三级化排除了它。
   - "不做距离钳位"删掉 1e-10 LU 正则化会让 Hessian（含 1/r⁵）在近天体处 inf/NaN。决策 5 的拆分排除了它。
   - "带 converged=False 返回"不钉死形状，已让网格搜索碰撞格谎报 success=True（`tools/viz/transfer.py:78` 把碰撞格画进有效解）。决策 3 的统一枚举排除了它。
3. **确定性是领域要求。** 航天轨道力学确定性传播，同初值同模型结果唯一。"跑出来的结果被悄悄改过、调用方无信号"违背这一性质。隐式降级（换物理模型、换精度档、换后端）的最坏后果不是慢，是**错而不自知**——`spice_optional` 链换物理模型、`ITRFApproxAxes` 降精度档、DC 停滞放宽容差，都是"结果数值变了，调用方以为没变"。
4. **成本可控。** 决策 5 的精确化使原盘点 ~30 处 MIN_DISTANCE 钳位绝大多数判为"机器精度正则化保留"，迁移面大幅缩小。真正的删除集中在决策 4 的资源降级（8 处 ADR 修订）和决策 1 的红线（谎报/藏失败，约 36 处）。

## 结果

### 新增

- `PropagationFailure(E2M2EError)` 类型异常（`e2m2e/exceptions.py`），取代 `dynamics.py:54` 的字符串匹配 catch。
- `ConvergenceState` 扩充 `INFEASIBLE`、`COLLISION`；规定搜索/修正结果统一 `status: ConvergenceState` + `cause: str` 规范。
- 碰撞终止能力：CR3BP/BCR4BP body-radius 配置注入 + propagation 内事件检测（`g=|r|-R_body, terminal=True`）。
- 能力缺失场景的显式 `backend="scipy"/"rust"` 参数（事件检测等），无 `auto`。

### 变更（迁移顺序，详见 `archive/plans/robustness-cleanup.md`）

1. 加 `PropagationFailure` 类型异常（零测试破坏，地基）。
2. 决策 3：统一 `ConvergenceState` status 规范，各搜索结果对象对齐（多数测试断言 happy path，破坏小）。
3. 决策 1 红线：修谎报/藏失败（DC 停滞短路、MC 控制器 None 当成功、`propagate_orbit_state_at_time` 空states退回插值、网格搜索碰撞格 success=True、qlaw 步塌缩空转、qlaw `_resolve_mu` 静默地球 μ）。
4. 决策 2：`_propagate_state_only` 空states改带 failure 标记；同步改 `transfer_optimization.py` 的 `len==0` 嗅探与 NLP `dv=1e10` 双惩罚（去目标惩罚，留约束冲突标记）。
5. 决策 4：移除资源降级（8 处 ADR 修订）；事件检测加显式 backend，不 auto。
6. 决策 5：碰撞终止 + body-radius 注入（最高风险，影响力求值/STM，须先确保碰撞事件终止再动任何物理量级钳位）。

### 不变

- 机器精度正则化（MIN_DISTANCE ≈ 1e-10 LU，防 NaN）。
- 自适应积分的步拒绝-缩步-重试标准行为。
- 机器精度步长地板（循环守卫）。
- 测试注入缝（ADR 0017 monkeypatch 回退，限测试路径）。
- `design_orbit.py` / `multiple_shooting.py` / `homotopy.py` 的抛异常 + 带标记返回范式（已是合规先例）。
- IEEE 754 浮点定义域防护（如 arccos 前 clip 到 [-1,1]）。

