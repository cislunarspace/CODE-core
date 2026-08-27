# ADR 0020: Failure handling policy — deterministic failures raise, infeasible searches return flags, no implicit degradation / 失败处理策略：确定性失败抛异常，搜索不可行带标记，禁止隐式降级

[English](#adr-0020-failure-handling-policy--deterministic-failures-raise-infeasible-searches-return-flags-no-implicit-degradation) | [简体中文](#中文)

## English

**Status**: Adopted (decision 3 revised by ADR 0024)
**Date**: 2026-08-09
**Related**: ADR 0002 (multiple scipy fallbacks revised by this ADR), ADR 0003
(the frame layer's never-auto-degrade principle — direct precursor), ADR 0009,
ADR 0014 (decision 4 error translation), ADR 0016, ADR 0017, ADR 0019

### Context

A robustness audit across `algorithm/` + `data/` + `api/` (four parallel
scans) found ~139 sites doing something other than raising or flagging on
failure: silently returning approximations, auto-switching backends, loosening
tolerances while reporting success, hiding failures inside success statistics.
They spread across layers but share one root: **treating failure as an
acceptable alternative result rather than an event to raise or explicitly
flag.**

The most typical:

- **Step-size collapse silently swallowed**: `dynamics.py:611-633` catches
  Rust's "step size collapsed" error (via string matching at `dynamics.py:54`)
  and returns empty states; `propagate_orbit_state_at_time`
  (`dynamics.py:688-699`) then interpolates from the orbit's own data on empty
  states and returns it as a successful result.
- **Lying about convergence**: `differential_correction.py:730-744` marks
  `converged=True` when Newton stalls (corrections < 1e-14) with residuals
  still at 1e-8, while configured tolerance is 1e-12 — a silent four-order-of-
  magnitude relaxation.
- **Failures hidden inside success statistics**: `monte_carlo.py:484-511`, when
  the control law returns None (no convergence / no crossing found), records
  `failed_k = False`; station-keeping Δv statistics skew systematically low.
- **Six incompatible failure-flag dialects**:
  `MultipleShootingResult.converged` /
  `TransferSolution.converged` /
  `TransferOptimizationResult.success` /
  `Orbit.correction_success` (bool|None tri-state) / DC's implicit None /
  grid search's `success:bool + free-string status`. Consequence:
  `search_parallel.py:189-198` hardcodes collision cells' `success=True`,
  and collision cells get plotted as valid solutions on the Δv-Time chart
  (`tools/viz/transfer.py:78`).
- **Resource degradation chains**: the `spice_optional=True` three-tier chain
  (`normal_form/pipeline` → `dynamical_substitution` → `quasi_floquet`)
  silently swaps physics from full ephemeris to pure CR3BP when SPICE is
  missing; `nlp_copt.py` auto-falls-back SLSQP when COPT is unavailable;
  `_HAS_RUST_*` import gates fall back to scipy when Rust is unavailable.

Astrodynamics is deterministic: same initial values + force models → unique
results. These robustness snippets cost: **computed results may have been
quietly altered with no signal to callers.**

### Unifying principle

**Behavior is decided by explicit inputs; no implicit degradation.**
Failure either raises (deterministic processes) or returns with unified flags
(search processes); no intermediate state exists where one thing failed,
another was returned, and callers cannot tell.

### Decision

#### Decision 1: three failure classes, three treatments

| Class | Meaning | Treatment |
|---|---|---|
| Deterministic propagation failure | integration divergence, step collapse to machine floor, unavailable Jacobian | raise `PropagationFailure` (decision 2) |
| Search/optimization infeasibility | grid cell diverges, NLP candidate infeasible, single DC step fails | return with unified flags (decision 3) |
| Red lines (forbidden for all classes) | lying about success, hiding failures in success stats, silently swapping physics | fix always, no exceptions |

The discriminator isn't which layer code sits in but **whether callers can
distinguish "got what I wanted" from "didn't"**. Flagged returns that preserve
that distinction are compliant; those destroying it (`success=True` lies,
approximate values without flags, implicit None losing causes) are red lines.

#### Decision 2: contextualized semantics of deterministic propagation failures

Raising on any step collapse is too coarse; literally applied it would ban
adaptive integrators' standard behavior and kill legitimate gravity assists.
Refined into three tiers:

1. **Step rejection** (error > tol but h still above machine floor):
   standard adaptive-controller behavior — reject, shrink h, retry
   (`cr3bp.rs:260-277`, `solve_ivp.rs:251-303`, `force_model.py` RK loop).
   **Not failure; not reported; not counted as fallback.**
2. **Collapse to machine floor / unavailable Jacobian / true divergence**:
   propagators raise **`PropagationFailure`** (new typed exception, see
   Consequences). Replaces the fragile catch matching `"step size collapsed"`
   via string at `dynamics.py:54` (one wording change breaks it).
3. **Context decides reporting semantics**: the same `PropagationFailure`
   raised to users calling `propagate` directly; search/optimization wrappers
   (grid search, NLP, multiple-shooting segments) catch and convert to
   `status=INFEASIBLE/DIVERGED` flagged returns. **Propagators themselves don't
   assume context**; callers decide whether to catch or re-raise.

Machine-precision floors (`MIN_STEP = 1e-12·span`,
`10·EPSILON·(1+|t|)`) must remain, explicitly acknowledged: they are loop
guards (preventing h→0 infinite loops on true divergence), not concealment.
Floors must be observable and enter result objects. What's forbidden are
**physical-magnitude floors**, e.g. `qlaw.py:379-380` resetting rejected steps
back to original step length (triggering ~2 million idle steps barely advancing
t before assembling a control law from unaccepted intermediate states), or
`force_model.py:804` forcing acceptance via `h=max(h,min_step)`. Removing
physical floors aligns the Python path with Rust's pattern
(`solve_ivp.rs:244-248`: min_step used solely for failure detection, never
lifts).

> Catching step-collapse in search contexts and converting to infeasible
> (`dynamics.py:610-633`, `transfer_grid_search.rs:157-187`) is compliant;
> what changes isn't the catching but replacing empty-states-with-len==0
> sniffing with structured results carrying failure flags.

#### Decision 3: unified failure flags for infeasible searches

Returning converged=False isn't enough: the flag's shape must be pinned, or
each module grows its own dialect and callers get bitten by None/False/missing
flags. The six dialects already let collision cells lie success=True.

Anchored on the existing `ConvergenceState` enum
(`e2m2e/data/templates/enums.py`: ITERATING/CONVERGED/DIVERGED/STAGNATED/
MAX_ITERATIONS), **extend with `INFEASIBLE` and `COLLISION`**, and require:

- All search/correction results expose identically named
  `status: ConvergenceState` field + `cause: str`.
- **Result objects must be returned even on failure** (carrying status +
  cause). Asymmetric signatures returning objects on success and None on
  failure are banned: `DifferentialCorrection.iterate_correction`'s
  `Orbit | None` becomes always-a-result-carrying-status (Orbit as its field);
  `termination_reason` travels with the object instead of lingering on the
  solver.
- Collision is its own status enum value (COLLISION), not
  `success=True + status string`. The collision cells at
  `search_parallel.py:189-198` must move to the failure side.
- Abolish `bool | None` tri-states, free-string statuses, implicit None.
  `converged`/`success`/`correction_success` may survive compatibly as derived
  properties of `status == CONVERGED`, but must not be the only signal.

Precedents: `design_orbit.py` (raises `DesignNotConvergedError` on all
non-convergent paths), `multiple_shooting.py`
(`MultipleShootingResult(converged=False, status=...)`) are already correct
in-repo patterns; this decision generalizes them.

> **Revision (2026-08-11, ADR 0024)**: boolean compatibility projections are
> dropped: `success`/`converged`/`correction_success` are removed outright
> without a runtime compatibility layer; `ConvergenceState` gains `FAILED`
> beyond this decision's `INFEASIBLE`/`COLLISION`. The "may survive as derived
> properties" clause above is void; all other clauses stand.

#### Decision 4: no implicit resource degradation; two kinds of unavailability distinguished

Auto-switching backends when resources (SPICE/Rust/COPT) are unavailable is
this ADR's core elimination target. But two kinds exist, treated differently:

- **Resource missing** (not installed/not built): **error out**. Spice is now a
  default feature, standardized by `make dev`, shipped in release wheels —
  these resources are constant in normal operation; absence means environment
  misconfiguration, never a reason to quietly switch to slow paths. Revised:
  ADR 0002's Dynamics-base scipy fallback on missing Rust, COPT→SLSQP
  fallback, silent sans-spice degradation; ADR 0009's release try/except;
  ADR 0016's cache-miss cspice fallback; ADR 0017's rust-unavailable
  processes fallback; ADR 0019's SPICE-missing ITRFApproxAxes degradation —
  all become errors.
- **Capability missing** (backend present, feature unimplemented/semantics
  unaligned): **explicit `backend="scipy"` / `backend="rust"` parameter**,
  one or the other; omitting raises (deprecation warning acceptable during
  migration, removed next major). **No `backend="auto"`** — auto still lets
  code decide backends for users, i.e., implicit. Typical case: CR3BP/BCR4BP
  event detection's Rust semantics not yet aligned with scipy (ADR 0002 event
  clause) — capability missing, explicit backend required.

**Test-injection seam exemption**: ADR 0017's monkeypatch fallback (fall back
to Python when tests inject synthetic trajectories, so injections take effect)
is test infrastructure, not production degradation — not banned, but confined
to test paths (`_geometry_methods_monkeypatched` detection); production paths
never trigger it.

#### Decision 5: separate singularity regularization from collision termination

Naively removing distance clamps would delete two unrelated things together,
blowing up Hessians. Refined:

- **Machine-precision regularization kept**:
  `MIN_DISTANCE ≈ 1e-10 LU` (≈3.8 cm, far inside any body radius) prevents
  divide-by-zero NaNs at gravity's 1/rⁿ singularity — present in
  `potential.py:11`, `dynamics.py:76`, `cr3bp.rs:19`, `bcr4bp.rs:22`
  (all 1e-10 nondimensional), and `nbody_stm.rs:27` at 1e-6 km for the same
  purpose. Hessians contain 1/r⁵ terms (`potential.py:42-50`); deleting these
  yields inf/NaN near primaries. Numerical guards, not physical falsehoods.
  **All retained.**
- **Physical-magnitude clamps become collision termination**: intersecting a
  body radius (Earth ≈6378 km, Moon ≈1737 km) → event detection
  `g = |r| - R_body`, `terminal=True`, or raise. `transfer_geometry.rs:211`'s
  `check_collision` already post-hoc scans; core propagation needs the
  event-based version.
- CR3BP is a point-mass model with no intrinsic body radii — collision
  termination needs **external body-radius configuration injection**. New
  feature, not removal of old behavior.
- Wording shifts from "no distance clamping" to "**no clamping within body
  radii**".

### Revisions to existing ADRs

| ADR | Original decision | Changed to | Class |
|---|---|---|---|
| 0002 | Dynamics-base scipy fallback on missing Rust | error on missing Rust | resource missing |
| 0002 | COPT→SLSQP fallback | error; NLP backend explicit | resource missing |
| 0002 | Silent sans-spice slow-path degradation | error | resource missing |
| 0002 | CR3BP/BCR4BP events passed to scipy fallback | explicit `backend="scipy"/"rust"`, no auto | capability missing |
| 0009 | Release without spice; try/except silent degradation | error (releases ship spice; mechanism removed) | resource missing |
| 0016 | Cache miss silently falls back to cspice FFI | error or explicit selection (Strict mode generalized from parallel-only to default) | resource missing |
| 0017 | Explicitly chosen rust falls back to processes when Rust missing | error (test monkeypatch seam exempt) | resource missing |
| 0019 | SPICE missing degrades drag rotation to ITRFApproxAxes | error or explicit low-precision backend choice | resource missing |

> Note: ADR 0002 line 96 originally stated BCR4BP events raise
> NotImplementedError (#333); actual code (`bcr4bp_dynamics.py:204-212`) had
> changed to warn + scipy fallback — doc/code mismatch resolved here as
> capability missing with explicit backend.

### Rationale

1. **Direction has precedent, not invented wholesale.** ADR 0003 item 7
   (explicit errors, never auto precision degradation; clamping behind explicit
   options) established error-on-missing at the coordinate layer long ago;
   this ADR generalizes it repo-wide. ADR 0004's loud non-serializable errors
   and ADR 0018's mandatory triples turning silent corruption into compile
   failures are precedents for decision 4 (no lying); ADR 0014 decision 4's
   exception translation at api/ into `OrbitError(code/message/details)` is the
   downstream exit for decision 2 (raising); ADR 0016 Strict mode's hard-fail
   misses precede decision 4 (resource-missing ⇒ error).
2. **Coarse wording kills legitimate behavior — three counterexamples excluded
   via adversarial verification.**
   - Raise-on-any-collapse would ban adaptive reject-shrink-retry (standard RK)
     and, combined with de-clamping, legitimate low-altitude lunar flybys
     (r₂≈1e-3, 384 km from lunar center, missing the surface — a legal gravity
     assist) would be reported as integration failure. Decision 2's tiering
     excludes it.
   - Deleting 1e-10 LU regularization makes Hessians (1/r⁵ terms) inf/NaN near
     bodies. Decision 5's split excludes it.
   - Flagless converged=False returns already let grid-search collision cells
     lie success=True (`tools/viz/transfer.py:78` plotting collisions as valid
     solutions). Decision 3's unified enum excludes it.
3. **Determinism is a domain requirement.** Astrodynamical propagation is
   deterministic — same initial state and model, unique outcome. Results being
   quietly altered with no caller signal violates it. Implicit degradation's
   worst consequence isn't slowness but **wrongness without awareness**:
   `spice_optional` chains swapping physics, `ITRFApproxAxes` dropping accuracy
   tiers, DC stalling loosening tolerances — all change numbers while callers
   assume nothing changed.
4. **Cost is controlled.** Decision 5's refinement reclassifies most of the
   audit's ~30 MIN_DISTANCE clamps as machine-precision regularization
   (retained); migration scope shrinks drastically. Real deletions concentrate
   in decision 4's resource degradations (8 ADR revisions) and decision 1's
   red lines (~36 lying/hiding sites).

### Consequences

#### Added

- `PropagationFailure(E2M2EError)` typed exception (`e2m2e/exceptions.py`),
  replacing the string-matching catch at `dynamics.py:54`.
- `ConvergenceState` extended with `INFEASIBLE`, `COLLISION`; unified
  `status: ConvergenceState` + `cause: str` norm for search/correction results.
- Collision termination: CR3BP/BCR4BP body-radius config injection +
  event-based termination in propagation (`g=|r|-R_body, terminal=True`).
- Explicit `backend="scipy"/"rust"` parameters for capability-missing scenarios
  (event detection etc.), no `auto`.

#### Changes (migration order)

1. Add `PropagationFailure` typed exception (zero test breakage; foundation).
2. Decision 3: unify `ConvergenceState` status norms across search results
  (most tests assert happy paths; small breakage).
3. Decision 1 red lines: fix lying/hiding (DC stall shortcut, MC controller
  None treated as success, `propagate_orbit_state_at_time` empty-states
  interpolation retreat, grid-search collision success=True, qlaw idle spin,
  qlaw `_resolve_mu` silent Earth μ).
4. Decision 2: `_propagate_state_only` empty states → flagged failure; sync
  `transfer_optimization.py`'s len==0 sniffing and NLP's dv=1e10 double
  penalty (drop objective penalty, keep constraint-conflict flags).
5. Decision 4: remove resource degradations (8 ADR revisions); event detection
  gains explicit backend, no auto.
6. Decision 5: collision termination + body-radius injection (highest risk —
  touches force evaluation/STM; ensure collision-event termination before
  touching any physical-magnitude clamps).

#### Unchanged

- Machine-precision regularization (MIN_DISTANCE ≈ 1e-10 LU, NaN guards).
- Adaptive integration's reject-shrink-retry standard behavior.
- Machine-precision step floors (loop guards).
- Test injection seam (ADR 0017 monkeypatch fallback, test paths only).
- `design_orbit.py` / `multiple_shooting.py` / `homotopy.py`'s raise + flag
  paradigm (already-compliant precedents).
- IEEE 754 domain protection (e.g., clip to [-1,1] before arccos).

## 中文

**状态**：已采纳（决策 3 经 ADR 0024 修订）
**日期**：2026-08-09
**关联**：ADR 0002（多处 scipy 回退被本 ADR 修订）、ADR 0003（坐标层绝不自动降精度原则，本 ADR 的直接前身）、ADR 0009、ADR 0014（决策 4 错误翻译）、ADR 0016、ADR 0017、ADR 0019

### 背景

一次跨 `algorithm/` + `data/` + `api/` 的健壮性盘点（四路并行扫描）找出约 139 处失败发生时做了抛异常或带标记返回以外的事的代码：静默返回近似值、自动换后端、放宽容差并报告成功、把失败藏进成功统计。它们分散在各层，但同源：**把失败当成一种可接受的备选结果，而不是需要上抛或显式标记的事件**。

几条最典型的：

- **步长塌缩被静默吞掉**：`dynamics.py:611-633` 捕获 Rust 的 "step size collapsed" 错误（靠 `dynamics.py:54` 的字符串匹配），返回空 states；`propagate_orbit_state_at_time`（`dynamics.py:688-699`）拿到空 states 后退回轨道自身数据的插值，当成功结果返回。
- **谎报收敛**：`differential_correction.py:730-744`，牛顿修正停滞（修正量 < 1e-14）但残差仍有 1e-8 时，直接标 `converged=True`；配置容差是 1e-12，等于静默放宽容差 4 个数量级。
- **失败藏进成功统计**：`monte_carlo.py:484-511`，控制律返回 None（没收敛/没找到穿越点）时，样本 `failed_k = False`，保持策略的 Δv 统计系统性偏低。
- **六种互不兼容的失败标记方言**：`MultipleShootingResult.converged` / `TransferSolution.converged` / `TransferOptimizationResult.success` / `Orbit.correction_success`（bool|None 三态）/ DC 的 implicit None / 网格搜索的 `success:bool + 自由字符串 status`。后果：`search_parallel.py:189-198` 网格搜索碰撞格 `success` 硬编码 True，碰撞格被当有效解画进 Δv-Time 图（`tools/viz/transfer.py:78`）。
- **资源降级链**：`spice_optional=True` 三级链（`normal_form/pipeline` → `dynamical_substitution` → `quasi_floquet`）在 SPICE 不可用时静默把物理模型从完整星历换成纯 CR3BP；`nlp_copt.py` COPT 不可用自动回退 SLSQP；`_HAS_RUST_*` 导入门控在 Rust 不可用时回退 scipy。

航天轨道力学是确定性的：同样的初值和力模型，结果唯一。这些健壮性代码的代价是：**跑出来的结果可能被悄悄改过，而调用方无信号**。

### 统一原则

**行为由显式输入决定，不隐式降级。** 失败要么上抛（确定性过程），要么带统一标记返回（搜索过程）；不存在失败了一种、返回了另一种、调用方看不出区别的中间态。

### 决策

### 决策 1：失败分三类，处置不同

| 类别 | 含义 | 处置 |
|---|---|---|
| 确定性传播失败 | 积分发散、步长塌缩到机器精度地板、雅可比算不出 | 抛 `PropagationFailure`（决策 2） |
| 搜索/优化不可行 | 网格搜索某格发散、NLP 候选不可行、微分修正单步不收敛 | 带统一标记返回（决策 3） |
| 红线（任何类都禁） | 谎报成功、把失败藏进成功统计、静默换物理模型 | 一律改掉，无例外 |

判别关键不是代码在哪一层，而是**调用方是否能把得到了想要的结果和没得到区分开**。能区分的带标记返回是合规的；不能区分的（谎报 `success=True`、返回近似值无标志、implicit None 原因丢失）是红线。

### 决策 2：确定性传播失败的语境化语义

步长塌缩即抛异常是一条过于粗糙的规则。按字面它会禁掉自适应积分的标准行为、杀掉合法的引力辅助轨迹。精确化分三级：

1. **步拒绝**（error > tol，但 h 仍大于机器精度地板）：自适应控制器的标准行为，拒绝该步、缩小 h、重试（`cr3bp.rs:260-277`、`solve_ivp.rs:251-303`、`force_model.py` 的 RK 循环）。**不是失败，不报告，不计为回退。**
2. **步塌缩到机器精度地板 / 雅可比算不出 / 真发散**：传播器抛 **`PropagationFailure`**（新建类型异常，见结果一节）。取代当前 `dynamics.py:54` 靠字符串匹配 `"step size collapsed"` 的脆弱 catch（消息改一个字就断）。
3. **语境决定汇报语义**：同一次 `PropagationFailure`，用户直接调 `propagate` → 上抛；搜索/优化 wrapper（网格搜索、NLP、多重打靶段积分）→ catch，转成 `status=INFEASIBLE/DIVERGED` 带标记返回。**传播器自身不假设语境**，由调用方决定 catch 还是 re-raise。

机器精度地板（`MIN_STEP = 1e-12·span`、`10·EPSILON·(1+|t|)`）必须保留并显式承认：它是循环守卫（防真发散时 h→0 死循环），不是隐瞒失败。地板值须可观测、进结果对象。**禁止的是物理量级地板**，如 `qlaw.py:379-380` 的 `if h<1e-6: h=step` 把被拒步长重置回原步长（触发后空转到 200 万步几乎不推进 t，最后用未经验收的中间态拼出控制律返回），或 `force_model.py:804` 的 `h=max(h,min_step)` 强行接受。删除物理量级地板后，Python 路径与 Rust `solve_ivp.rs:244-248` 的范式对齐（min_step 只用作失败判定，绝不做抬升）。

> `dynamics.py:610-633` 与 `transfer_grid_search.rs:157-187` 在搜索语境 catch 步塌缩转 infeasible，是合规的；要改的不是别 catch，而是把返回值从空 states 靠 `len==0` 嗅探改成带 failure 标记的结构化结果。

### 决策 3：搜索不可行的统一失败标记

带 converged=False 返回不够：必须钉死标记的形状，否则各模块各搞方言，调用方被 None/False/缺标志坑。现有六种方言（见背景）已经让碰撞格谎报 success=True。

以现有 `ConvergenceState` 枚举（`e2m2e/data/templates/enums.py`，含 ITERATING/CONVERGED/DIVERGED/STAGNATED/MAX_ITERATIONS）为锚，**扩充 `INFEASIBLE` 与 `COLLISION`**，规定：

- 所有搜索/修正结果暴露同名的 `status: ConvergenceState` 字段 + `cause: str`（失败原因）。
- **失败时也必须返回结果对象**（携带 status + cause）。禁止成功返回对象、失败返回 None 的不对称签名：`DifferentialCorrection.iterate_correction` 的 `Orbit | None` 改为始终返回携带 status 的结果（Orbit 作为其字段），`termination_reason` 跟对象走而非留在 solver 上。
- 碰撞是独立的 status 枚举值（COLLISION），不是 `success=True + status 字符串`。`search_parallel.py:189-198` 的碰撞格必须进失败侧。
- 废除 `bool | None` 三态、自由字符串 status、implicit None。`converged`/`success`/`correction_success` 作为 `status == CONVERGED` 的派生属性可兼容保留，但不得是唯一信号。

先例：`design_orbit.py`（所有不收敛路径抛 `DesignNotConvergedError`）、`multiple_shooting.py`（`MultipleShootingResult(converged=False, status=...)`）已是仓内正确范式，本决策推广之。

> **修订（2026-08-11，ADR 0024）**：布尔兼容投影不再保留：`success`/`converged`/`correction_success` 一律移除，不设运行时兼容层；`ConvergenceState` 在本决策的 `INFEASIBLE`/`COLLISION` 之外另增 `FAILED`。上文作为派生属性可兼容保留一句作废，其余条款不变。

### 决策 4：禁止隐式资源降级，区分两种不可用

资源（SPICE/Rust/COPT）不可用时自动换后端是本 ADR 要消除的核心模式。但不可用有两种，处置不同：

- **资源缺失**（没装/没构建）：**报错**。spice 现为默认 feature、`make dev` 标准化、release wheel 已带 spice，正常操作中这些资源恒在，缺失即环境没搭好，不是悄悄换慢路径的理由。修订：ADR 0002 的 Dynamics 基类 Rust 不可用回退 scipy、COPT 不可用回退 SLSQP、无 spice 静默降级，ADR 0009 的 release try/except，ADR 0016 的缓存 miss 回退 cspice，ADR 0017 的显式 rust 不可用回退 processes，ADR 0019 的 SPICE 缺失降 ITRFApproxAxes，全改报错。
- **能力缺失**（后端在，但某功能未实现/语义未对齐）：**显式 `backend="scipy"` / `backend="rust"` 参数**，二选一；不传则报错（迁移期可给 deprecation warning，下个 major 移除）。**不允许 `backend="auto"`**，auto 仍是代码替用户决定后端，属于隐式。典型：CR3BP/BCR4BP 事件检测 Rust 语义未对齐 scipy（ADR 0002 事件检测条），属能力缺失，走显式 backend。

**测试注入缝豁免**：ADR 0017 的 monkeypatch 回退（测试 `setattr` 注入合成轨迹时回退 Python，让注入生效）是测试基础设施，不是生产降级，不在禁止之列，但须限定在测试路径（`_geometry_methods_monkeypatched` 检测），生产路径不触发。

### 决策 5：奇点正则化与碰撞终止分离

不做距离钳位会把两件不相关的事一起删掉，后果是 Hessian 爆炸。精确化：

- **机器精度正则化保留**：`MIN_DISTANCE ≈ 1e-10 LU`（≈3.8cm，远在任何天体半径内）防引力 1/rⁿ 奇点的除零 NaN，存在于 `potential.py:11`、`dynamics.py:76`、`cr3bp.rs:19`、`bcr4bp.rs:22`（均 1e-10，无量纲），`nbody_stm.rs:27` 同用途但取 1e-6 km。Hessian 含 1/r⁵ 项（`potential.py:42-50`），删了它接近主天体就 inf/NaN。这是数值守卫，不是物理谎言。**全部保留。**
- **物理量级钳位改碰撞终止**：撞天体半径（地球 R≈6378km、月球 R≈1737km）→ 事件检测 `g = |r| - R_body`，`terminal=True`，或抛异常。`transfer_geometry.rs:211` 的 `check_collision` 已有 post-hoc 扫描；core propagation 需补 event-based 版本。
- CR3BP 是质点模型，没有内禀天体半径，碰撞终止需要**从外部注入 body-radius 配置**。这是新功能，不是删旧行为。
- 措辞从不做距离钳位改为**不做天体半径以内的距离钳位**。

### 既有 ADR 的修订

| ADR | 原决策 | 改为 | 类别 |
|---|---|---|---|
| 0002 | Dynamics 基类 Rust 不可用回退 scipy（构建失败降级路径） | Rust 不可用即报错 | 资源缺失 |
| 0002 | COPT 不可用回退 SLSQP | 报错，NLP 后端显式指定 | 资源缺失 |
| 0002 | 无 spice 静默降级慢路径 | 报错 | 资源缺失 |
| 0002 | CR3BP/BCR4BP events 传 scipy 回退 | 显式 `backend="scipy"/"rust"`，不 auto | 能力缺失 |
| 0009 | release 不带 spice，try/except 静默降级 | 报错（release 已带 spice，降级机制删除） | 资源缺失 |
| 0016 | 缓存 miss 静默回退 cspice FFI | 报错或显式指定（Strict 模式从并行专用推广为默认） | 资源缺失 |
| 0017 | 显式选 rust 但 Rust 不可用回退 processes | 报错（测试 monkeypatch 缝豁免） | 资源缺失 |
| 0019 | SPICE 缺失 drag 帧旋转降 ITRFApproxAxes | 报错或显式指定低精度后端 | 资源缺失 |

> 注：ADR 0002 第 96 行原称 BCR4BP 传 events 抛 NotImplementedError（#333），实际代码（`bcr4bp_dynamics.py:204-212`）已改为 warn + 回退 scipy，ADR 与代码已不一致，本 ADR 一并澄清为能力缺失，走显式 backend。

### 理由

1. **方向有先例，不是凭空而来。** ADR 0003 第 7 条（错误明确，绝不自动降精度；钳位需显式选项）早在坐标层确立了缺失即报错、钳位需显式的范式，本 ADR 是把它推广到全局。ADR 0004 的不可序列化大声报错、0018 的雅可比接口强制三元组让静默出错变编译不过，是决策 4（禁止谎报）的先例；ADR 0014 决策 4 的异常在 api/ 翻译成 `OrbitError(code/message/details)` 是决策 2（抛异常）的下游出口；ADR 0016 Strict 模式的 miss 硬失败是决策 4（资源缺失报错）的先例。
2. **粗措辞会杀合法行为：对抗验证排除的三个反面。**
   - 步长塌缩即抛异常、不回退不设地板，按字面会禁掉自适应步拒绝-缩步-重试（标准 RK 行为），且与去钳位叠加后，月球低高度掠过（r₂≈1e-3、距月心 384km、未撞月面的合法 gravity assist）会被报成积分失败。决策 2 的三级化排除了它。
   - 不做距离钳位删掉 1e-10 LU 正则化会让 Hessian（含 1/r⁵）在近天体处 inf/NaN。决策 5 的拆分排除了它。
   - 带 converged=False 返回不钉死形状，已让网格搜索碰撞格谎报 success=True（`tools/viz/transfer.py:78` 把碰撞格画进有效解）。决策 3 的统一枚举排除了它。
3. **确定性是领域要求。** 航天轨道力学确定性传播，同初值同模型结果唯一。跑出来的结果被悄悄改过、调用方无信号，违背这一性质。隐式降级（换物理模型、换精度档、换后端）的最坏后果不是慢，是**错而不自知**：`spice_optional` 链换物理模型、`ITRFApproxAxes` 降精度档、DC 停滞放宽容差，都是结果数值变了、调用方以为没变。
4. **成本可控。** 决策 5 的精确化使原盘点 ~30 处 MIN_DISTANCE 钳位绝大多数判为机器精度正则化保留，迁移面大幅缩小。真正的删除集中在决策 4 的资源降级（8 处 ADR 修订）和决策 1 的红线（谎报/藏失败，约 36 处）。

### 结果

### 新增

- `PropagationFailure(E2M2EError)` 类型异常（`e2m2e/exceptions.py`），取代 `dynamics.py:54` 的字符串匹配 catch。
- `ConvergenceState` 扩充 `INFEASIBLE`、`COLLISION`；规定搜索/修正结果统一 `status: ConvergenceState` + `cause: str` 规范。
- 碰撞终止能力：CR3BP/BCR4BP body-radius 配置注入 + propagation 内事件检测（`g=|r|-R_body, terminal=True`）。
- 能力缺失场景的显式 `backend="scipy"/"rust"` 参数（事件检测等），无 `auto`。

### 变更（迁移顺序）

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
