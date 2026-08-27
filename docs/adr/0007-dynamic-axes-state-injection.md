# ADR 0007: Dynamic-axes state injection scheme / 动态坐标轴状态注入方案

[English](#adr-0007-dynamic-axes-state-injection-scheme) | [简体中文](#中文)

## English

**Status**: Adopted
**Date**: 2026-06-15
**Related Issue**: Slice 12 (dynamic axes and VNB/LVLH maneuver support)

### Context

Slice 12 needs two dynamic axes types — VNB (Velocity-Normal-Binormal) and
LVLH (Local Vertical Local Horizontal) — for:
1. Specifying `FiniteBurn` thrust direction in the VNB/LVLH frame;
2. Specifying `ImpulsiveBurn` Δv in VNB/LVLH before conversion to inertial.

The fundamental difference between dynamic and static axes (ICRS, ITRF,
etc.): their rotation matrices depend not only on epoch `et` but also on
spacecraft `state` (position + velocity). VNB's x-axis lies along velocity, z
along angular momentum, y completes by cross product — all changing with
`state` in real time.

The existing `Axes` interface offers only `rotation_matrix(et)` and
`rotation_and_rate(et)`, with no `state` in signatures. The decision: modify
the `Axes` interface, or find another way while keeping current signatures?

### Alternatives compared

#### Option A: modify the `Axes` interface

Add a `rotation_matrix(et, state)` overload to the `Axes` base class, or make
`state` an optional parameter. All existing subclasses (ICRS, ITRF, IAU2000Eq,
etc.) would need adaptation.

Problems:
1. **Static axes forced to accept irrelevant parameters.** ICRS is always the
   identity matrix; passing it `state` is noise.
2. **Breaks existing callers.** `CoordinateSystem.transform_state`, all tests,
   and all force-model coordinate conversions would need signature changes or
   branching.
3. **Conflates two concepts.** Static axes are pure functions of time;
   dynamic axes are functions of state + time. Stuffing both into one
   interface blurs the distinction.

#### Option B: keep `Axes` signatures; add a `DynamicAxes.update` method

Leave the `Axes` interface untouched. Add abstract base class `DynamicAxes`
extending `Axes` with an `update(t, state)` method. Callers `update` first,
then take matrices when needed.

```python
class DynamicAxes(Axes):
    """State-dependent dynamic axes."""

    @abc.abstractmethod
    def update(self, t: float, state: npt.NDArray[np.floating]) -> None:
        """Refresh internal direction cache from current state."""
        raise NotImplementedError

    def rotation_matrix(self, et: float) -> npt.NDArray[np.floating]:
        """Return rotation matrix from most recent update."""
        # subclass caches R during update; return cache here
        ...
```

VNB and LVLH are implemented as `DynamicAxes` subclasses. Static axes remain
untouched.

### Decision

**Option B is chosen.** Concretely:

1. **The `Axes` interface stays unchanged.** Static axes (ICRS, ITRF,
   IAU2000Eq etc.) fully unaffected. Callers like
   `CoordinateSystem.transform_state` need no changes.

2. **New `DynamicAxes` abstract base class.** Extends `Axes`, adds abstract
   `update(t, state)`. Subclasses compute direction vectors from state inside
   `update` and cache them; `rotation_matrix` returns cached values.
   `rotation_and_rate` likewise.

3. **The state-injection points live in `ForceModel.propagate` and
   `System.update_coordinate_systems`.**
   - `ForceModel.propagate`, before each integration step, if a `DynamicAxes`
     is detected, calls `axes.update(t, y)` to refresh directions.
   - `System.update_coordinate_systems` serves as the system-level unified
     entry: before propagation starts or any computation needing dynamic axes,
     the system injects current state into all registered dynamic axes.
   The system-level entry was chosen over per-force-model injection because
   dynamic axes may be shared across force models (VNB used for both drag and
   thrust directions simultaneously); updating at system level avoids duplicate
   computation and inconsistent states.

4. **VNB axis definitions (revised 2026-08-23).**
   - x-axis (Velocity): along `v / |v|`.
   - y-axis (Normal): along orbital angular momentum `h = r × v`,
     `h / |h|`.
   - z-axis (Binormal): `x × y`, ensuring right-handedness.
   The original text placed Normal on z and wrote Binormal as `z × x`,
   contradicting implementation and tests; corrected here against
   `VNBAxes` in `standard_dynamic_axes.py`. Literature disagrees on VNB's
   y/z naming; e2m2e takes this entry as normative.

5. **LVLH axis definitions (revised 2026-08-23).**
   - x-axis (Radial): along position `r / |r|` (radially outward).
   - z-axis (Cross-track): along orbital angular momentum `h / |h|`.
   - y-axis (In-track/Local Horizontal): `z × x`, ensuring right-handedness.
   The original text had z pointing Earthward (`-r`) and y along negative
   angular momentum, contradicting implementation and tests; corrected here
   against `LVLHAxes`. LVLH conventions vary in literature (some take z as
   `-r`, Earth-pointing); e2m2e adopts this entry — the RSW convention:
   radially outward, in-track, orbit-normal.

6. **`ObjectReferencedAxes` deferred.** `ObjectReferencedAxes` (relative axes
   centered on a celestial body, e.g. Moon-centered VNB) requires body state
   queries via `System`'s ephemeris interfaces. Slice 12 only supports VNB/
   LVLH based on spacecraft's own state; `ObjectReferencedAxes` awaits a later
   slice, which will extend the `DynamicAxes.update` signature to accept body
   states.

### Rationale

1. **Why leave the `Axes` interface alone.** Static axes are the overwhelming
   majority. Changing all static axes and all callers for a few dynamic ones
   pays general costs for special-case benefits; `DynamicAxes` as a subclass
   extends generality with special cases — open-closed principle.

2. **Why update-and-cache rather than recompute every call.** `rotation_matrix`
   may be called multiple times per state (`transform_vector` and
   `transform_state` each call once). `update` concentrates the state→direction
   computation at one point; subsequent matrix reads are O(1) cache lookups.
   The `update` semantics are also explicit: "I'm about to use this state;
   prepare."

3. **Why inject state at system level.** Dynamic axes are a coordinate-system-
   layer concept, not any single force model's private property. If both
   `DragModel` and `FiniteBurn` use VNB, calling `update` inside each
   `compute_acceleration` causes:
   - duplicated direction computation within one integration step;
   - direction inconsistency if two models interpret state differently (raw
     state vs interpolated midstep state).
   Updating once at `System.update_coordinate_systems` /
   `ForceModel.propagate` guarantees all force models see the same direction
   within a step.

4. **Why fix written definitions rather than inherit literature ambiguity.**
   VNB/LVLH axis conventions differ across sources (some define LVLH z as
   `+r`, others `-r`). Decisions 4–5 pin definitions into this ADR because:
   - axis inconsistency directly changes a maneuver's physical meaning; there
     must be one written baseline;
   - tests assert against decisions 4–5's definitions, locking implementation,
     tests, and docs together;
   - users migrating missions from other software can check this entry instead
     of guessing.

5. **Why defer `ObjectReferencedAxes`.** It adds two complexities:
   - querying a body's state at epoch `et` (`System.get_body_state`),
     introducing ephemeris dependency;
   - defining relative-state semantics (spacecraft state minus body state,
     then VNB/LVLH).
   Slice 12's use case (spacecraft-own VNB/LVLH maneuvers) needs neither.
   Deferral avoids premature generalization.

### Consequences

#### Added

- `e2m2e/algorithm/coordinate/dynamic_axes.py`: `DynamicAxes` abstract base.
- `e2m2e/algorithm/coordinate/standard_dynamic_axes.py`: `VNBAxes` and
  `LVLHAxes`.
- `tests/algorithm/coordinate/test_dynamic_axes.py`: direction correctness
  tests (asserting axes per decisions 4–5).

#### Changed

- `ForceModel.propagate`: after propagation moved to the Rust compiled path,
  Python no longer calls `DynamicAxes.update` step-by-step; state injection of
  dynamic axes rests with the system-level entry.
- `EphemerisSystem`: adds `update_coordinate_systems(t, state)`, calling
  `update` when `coordinate_system.axes` is a `DynamicAxes`. The `System` base
  class does not define it.

#### Unchanged

- `Axes` interface signatures (`rotation_matrix(et)`,
  `rotation_and_rate(et)`).
- All static axes implementations (ICRS, ITRF, IAU2000Eq, GMATITRF).
- `CoordinateSystem.transform_state` / `transform_vector` signatures and
  behavior.
- Existing behavior of `FiniteBurn`/`ImpulsiveBurn` (`FiniteBurn` gains VNB/
  LVLH support via a new `direction_frame` field without breaking existing
  APIs).

#### Follow-up work

- `ObjectReferencedAxes`: implement once body-relative frames are needed;
  extends `DynamicAxes.update` to accept reference-body states.
- `FiniteBurn` config DSL extension: new `direction_kind: "vnb" | "lvlh"` plus
  `direction_vector` (components in the dynamic frame); see ADR 0004 follow-ups.
- `ImpulsiveBurn.frame` field: accept `"vnb"`, `"lvlh"`, converting via
  `CoordinateSystem.transform_vector` (matching GMAT
  `Burn::ConvertDeltaVToInertial`'s `coincident=true` pure rotation).

## 中文

**状态**：已采纳
**日期**：2026-06-15
**关联 Issue**：Slice 12（动态坐标轴与 VNB/LVLH 机动支持）

### 背景

Slice 12 需要支持 VNB（Velocity-Normal-Binormal）和 LVLH（Local Vertical Local Horizontal）两种动态坐标轴，用于：
1. `FiniteBurn` 的推力方向在 VNB/LVLH 系中指定；
2. `ImpulsiveBurn` 的 Δv 在 VNB/LVLH 系中指定后转换到惯性系。

动态坐标轴与静态坐标轴（ICRS、ITRF 等）的根本区别：它们的旋转矩阵不仅依赖历元 `et`，还依赖航天器状态 `state`（位置+速度）。例如 VNB 的 x 轴沿速度方向，z 轴沿角动量方向，y 轴由叉积补全，这些方向都随 `state` 实时变化。

现有 `Axes` 接口只有 `rotation_matrix(et)` 和 `rotation_and_rate(et)`，签名中不含 `state`。需要决定：是改 `Axes` 接口，还是在保持现有签名的前提下另辟蹊径。

### 方案对比

### 方案 A：修改 `Axes` 接口

在 `Axes` 基类上增加 `rotation_matrix(et, state)` 重载，或把 `state` 作为可选参数。所有现有子类（ICRS、ITRF、IAU2000Eq 等）都需要适配。

问题：
1. **静态坐标轴被迫接受无关参数。** ICRS 永远是单位阵，给它传 `state` 是噪音。
2. **破坏现有调用方。** `CoordinateSystem.transform_state`、所有测试、所有力模型中的坐标变换调用，都需要改签名或做分支判断。
3. **混淆两种概念。** 静态坐标轴是纯时间函数，动态坐标轴是状态+时间的函数，硬塞进同一接口会模糊这一区别。

### 方案 B：保持 `Axes` 签名，新增 `DynamicAxes.update` 方法

不动 `Axes` 接口。新增 `DynamicAxes` 抽象基类，继承 `Axes`，增加 `update(t, state)` 方法。调用方在需要时先 `update` 再取矩阵。

```python
class DynamicAxes(Axes):
    """状态依赖的动态坐标轴。"""

    @abc.abstractmethod
    def update(self, t: float, state: npt.NDArray[np.floating]) -> None:
        """用当前状态更新内部方向缓存。"""
        raise NotImplementedError

    def rotation_matrix(self, et: float) -> npt.NDArray[np.floating]:
        """返回最近一次 update 后的旋转矩阵。"""
        # 子类在 update 中缓存 R，此处直接返回
        ...
```

VNB、LVLH 作为 `DynamicAxes` 子类实现。静态坐标轴不受影响。

### 决策

选择 **方案 B**。具体决策如下：

1. **`Axes` 接口不变。** 静态坐标轴（ICRS、ITRF、IAU2000Eq 等）完全不受影响。`CoordinateSystem.transform_state` 等调用方无需改动。

2. **新增 `DynamicAxes` 抽象基类。** 继承 `Axes`，增加 `update(t, state)` 抽象方法。子类在 `update` 中根据状态计算方向向量并缓存，`rotation_matrix` 返回缓存值。`rotation_and_rate` 同理。

3. **状态注入点放在 `ForceModel.propagate` 和 `System.update_coordinate_systems`。**
   - `ForceModel.propagate` 在每一步积分前，若检测到 `DynamicAxes`，调用 `axes.update(t, y)` 更新方向。
   - `System.update_coordinate_systems` 作为系统级统一入口：在传播开始前、或在任何需要动态坐标轴的运算前，由系统负责把当前状态注入到所有已注册的动态坐标轴中。
   选择系统级入口而非力模型级入口，是因为动态坐标轴可能被多个力模型共享（如 VNB 同时用于阻力和推力方向），在系统层面统一更新避免重复计算和状态不一致。

4. **VNB 轴向定义（修订，2026-08-23）。**
   - x 轴（Velocity）：沿速度方向 `v / |v|`。
   - y 轴（Normal）：沿轨道角动量方向 `h = r × v`，`h / |h|`。
   - z 轴（Binormal）：`x × y`，保证右手系。
   原文把 Normal 写在 z 轴、Binormal 写作 `z × x`，与实现和测试不符，
   此处按 `standard_dynamic_axes.py` 的 `VNBAxes` 勘正。不同文献对
   VNB 的 y/z 命名有分歧，e2m2e 以本条定义为准。

5. **LVLH 轴向定义（修订，2026-08-23）。**
   - x 轴（Radial）：沿位置方向 `r / |r|`（径向向外）。
   - z 轴（Cross-track）：沿轨道角动量方向 `h / |h|`。
   - y 轴（In-track/Local Horizontal）：`z × x`，保证右手系。
   原文写的是 z 轴指向地心（`-r`）、y 轴沿负角动量的另一套约定，
   与实现和测试不符，此处按 `LVLHAxes` 勘正。LVLH 的轴向约定在
   文献中不统一（有的 z 轴取 `-r` 指向地心），e2m2e 以本条定义
   为准，即径向向外、沿迹、轨道面法向的 RSW 口径。

6. **`ObjectReferencedAxes` 推迟实现。** `ObjectReferencedAxes`（以某天体为原点的相对坐标轴，如以月球为中心的 VNB）需要天体状态查询，依赖 `System` 的星历接口。当前 Slice 12 只支持以航天器自身状态为基准的 VNB/LVLH；`ObjectReferencedAxes` 留待后续 Slice 处理，届时需要扩展 `DynamicAxes.update` 签名以接受天体状态。

### 理由

1. **为什么不动 `Axes` 接口。** 静态坐标轴是绝大多数。为少数动态坐标轴改动所有静态坐标轴和所有调用方，是用普遍改特殊，代价远大于收益。`DynamicAxes` 作为子类，是用特殊扩展普遍，符合开闭原则。

2. **为什么用 `update` 缓存而非每次重新计算。** `rotation_matrix` 可能在同一状态被多次调用（如 `transform_vector` 和 `transform_state` 各调一次）。`update` 把状态到方向的计算集中在一点，后续取矩阵是 O(1) 查缓存。`update` 语义也明确，即我要用这个状态了，请准备好。

3. **为什么状态注入点在系统层。** 动态坐标轴是坐标系层面的概念，不是某个力模型的私有属性。如果 `DragModel` 和 `FiniteBurn` 都使用 VNB，各自在 `compute_acceleration` 里调用 `update` 会导致：
   - 同一积分步内重复计算方向向量；
   - 若两个力模型对状态的解释不同（如一个用原始状态、一个用中间插值状态），方向不一致。
   在 `System.update_coordinate_systems` 或 `ForceModel.propagate` 的统一位置更新，保证同一步所有力模型看到同一方向。

4. **为什么固定书面定义而非沿用文献歧义。** VNB/LVLH 的轴向定义在文献中并不统一（有些把 LVLH 的 z 轴定义为 `+r`，有些定义为 `-r`）。决策 4、5 把定义写死在 ADR 里，是因为：
   - 轴向定义不一致会直接改变机动的物理含义，必须有一处书面基准；
   - 测试以决策 4、5 的定义为断言依据，实现、测试、文档三者锁死；
   - 用户从其他软件迁移任务时，对照本条即可确认约定，无需猜。

5. **为什么推迟 `ObjectReferencedAxes`。** 它涉及两个额外复杂度：
   - 需要查询天体在历元 `et` 的状态（`System.get_body_state`），引入星历依赖；
   - 需要定义相对状态的语义（航天器状态减天体状态，再算 VNB/LVLH）。
   当前 Slice 12 的用例（航天器自身 VNB/LVLH 机动）不需要这些。推迟避免过早泛化。

### 结果

### 新增

- `e2m2e/algorithm/coordinate/dynamic_axes.py`：`DynamicAxes` 抽象基类。
- `e2m2e/algorithm/coordinate/standard_dynamic_axes.py`：`VNBAxes` 与 `LVLHAxes` 实现。
- `tests/algorithm/coordinate/test_dynamic_axes.py`：VNB/LVLH 方向正确性测试（按决策 4、5 的定义断言轴向）。

### 变更

- `ForceModel.propagate`：传播改走 Rust 编译路径后，Python 侧不逐步回调 `DynamicAxes.update`；动态坐标轴的状态注入由系统层入口承担。
- `EphemerisSystem`：新增 `update_coordinate_systems(t, state)`，当 `coordinate_system.axes` 为 `DynamicAxes` 实例时调用其 `update`。`System` 基类未定义该方法。

### 不变

- `Axes` 接口签名（`rotation_matrix(et)`、`rotation_and_rate(et)`）。
- 所有静态坐标轴实现（ICRS、ITRF、IAU2000Eq、GMATITRF）。
- `CoordinateSystem.transform_state` / `transform_vector` 签名与行为。
- `FiniteBurn` 和 `ImpulsiveBurn` 的现有行为（`FiniteBurn` 的 VNB/LVLH 支持通过新增 `direction_frame` 字段扩展，不破坏现有 API）。

### 后续工作

- `ObjectReferencedAxes`：待天体相对坐标系需求明确后实现，需扩展 `DynamicAxes.update` 以接受参考天体状态。
- `FiniteBurn` 配置 DSL 扩展：新增 `direction_kind: "vnb" | "lvlh"` 与 `direction_vector`（在动态坐标轴中的分量），见 ADR 0004 的后续工作列表。
- `ImpulsiveBurn` 的 `frame` 字段：支持 `"vnb"`、`"lvlh"` 值，转换走 `CoordinateSystem.transform_vector`（对应 GMAT `Burn::ConvertDeltaVToInertial` 的 `coincident=true` 纯旋转）。
