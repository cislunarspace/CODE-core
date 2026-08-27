# ADR 0008: Revoke runtime freezing of Axes / Origin / CoordinateSystem / 撤销 Axes / Origin / CoordinateSystem 运行时冻结

[English](#adr-0008-revoke-runtime-freezing-of-axes--origin--coordinatesystem) | [简体中文](#中文)

## English

**Status**: Adopted (freezing mechanism rejected; landed freezing reverted)
**Date**: 2026-07-17
**Related Issues**: #76, #121, #122

### Context

Concrete subclasses of `Axes`, `Origin`, and `CoordinateSystem` were at one
point slated for runtime freezing (`@dataclass(frozen=True)` or equivalent).
The freezing effort went through:

- **#76** introduced `CoordinateSystem` freezing, converting it to
  `@dataclass(frozen=True)`; landed and CLOSED.
- **#121** followed up, planning to freeze static `Axes` / `Origin` and all
  their concrete subclasses.
- **#122** designed the boundary between `DynamicAxes` and frozen base classes
  (dependent on #121).

After landing, re-evaluation judged freezing **unnecessary**. This ADR records
the reversal: the freezing mechanism is rejected, `CoordinateSystem` has been
reverted to a plain class, and #121/#122 are closed as wontfix.

The problem freezing targeted — runtime swapping of `axes`/`origin` components
causing inconsistent transformation results — never actually occurred. These
classes' instances follow a read-only convention after construction: callers
should not swap `axes`/`origin` components at runtime, but the application
layer does not enforce it.

### Decision

**Reject the freezing mechanism and revert the landed #76 freeze.**

Concretely:

1. `CoordinateSystem` reverts from `@dataclass(frozen=True)` to a plain class
   (commit `62f7308`).
2. #121 (freezing static `Axes` / `Origin` and concrete subclasses) will not
   proceed; wontfix.
3. #122 (boundary design for `DynamicAxes` vs frozen base classes) is wontfix
   with #121.
4. None of these are introduced: `CoordinateSystem` freezing,
   `Axes`/`Origin` base `__init_subclass__` hooks, dataclass conversion of
   concrete subclasses.

### Rationale

1. **No real bug drives it.** The problem freezing was meant to solve —
   runtime component swaps yielding inconsistent transformations — never
   appeared in actual code. After `CoordinateSystem` freezing shipped,
   patterns like `cs.axes = X` appeared neither in core nor in forces /
   transfer / algorithms layers. The problem was imagined, not observed.

2. **Tamper protection is not the application layer's job.** If someone swaps
   `cs.axes` to another instance at runtime, that's a scenario where the
   program itself has been tampered with. Users obtain verified code copies
   from GitHub and can trust their runtime. Application-layer freezing cannot
   stop the root cause of program tampering; it merely moves checkpoints to
   runtime.

3. **Freezing carries non-trivial engineering cost.** Concrete subclasses like
   `IAU2000EqAxes` have derived fields (e.g. internal state derived from
   `time_step`); the dataclass form needs boilerplate such as
   `field(default_factory=...)` + `__post_init__` + `object.__setattr__`; the
   boundary between `DynamicAxes` (deliberately mutable per ADR 0007) and
   frozen bases requires dedicated design (composition / exemption hooks /
   parallel inheritance trees — #122 existed entirely for this). That
   complexity serves a nonexistent bug; the cost exceeds the benefit.

4. **YAGNI.** At the code-style level, "authors shouldn't write mutation" is
   already covered by the grep gatekeeper in
   `tests/algorithm/coordinate/test_coordinate_immutability.py` (scanning
   `e2m2e/algorithm/coordinate/` for assignments like `cs.axes = X`). Runtime
   swap protection differs from that and was never an application-layer
   concern anyway.

### Consequences

#### Reverted / not proceeding

- `CoordinateSystem`'s `@dataclass(frozen=True)`: reverted to plain class
  (commit `62f7308`).
- Three tests in `TestCoordinateSystemFrozen`: deleted.
- #121 (full freezing of static `Axes` / `Origin` and subclasses): wontfix.
- #122 (`DynamicAxes` boundary design): wontfix.

#### Kept

- `TestCoordinateSystemOrthogonality`,
  `TestCoordinateSystemTransformVector` (zero vector): unrelated to freezing;
  kept.
- The grep gatekeeper in
  `tests/algorithm/coordinate/test_coordinate_immutability.py`: kept. It
  guards against authors writing mutation (static code style), not runtime
  swapping (dynamic protection). Different responsibilities; the former has
  nothing to do with freezing.

#### Impact on upstream decisions

| Issue | Original decision | State after this ADR |
|-------|--------|----------------|
| #76 | `CoordinateSystem` freezing (was CLOSED) | Reverted; original decision overturned |
| #121 | Freeze all static `Axes` / `Origin` | wontfix |
| #122 | `DynamicAxes` vs frozen-base boundary | wontfix (depended on #121) |

## 中文

**状态**：已采纳（冻结机制被拒绝，已落地的冻结已回退）
**日期**：2026-07-17
**关联 Issue**：#76、#121、#122

### 背景

`Axes`、`Origin`、`CoordinateSystem` 的具体子类一度计划做运行时冻结
（`@dataclass(frozen=True)` 或等价机制）。冻结机制经历了如下历程：

- **#76** 引入 `CoordinateSystem` 冻结，将 `CoordinateSystem` 改为
  `@dataclass(frozen=True)`，落地并 CLOSED。
- **#121** 跟进，计划对静态 `Axes` / `Origin` 及其全部具体子类做冻结。
- **#122** 设计 `DynamicAxes` 与冻结基类的边界（依赖 #121）。

冻结落地后，重新评估判定冻结**不必要**。本 ADR 记录这一回退决策：
冻结机制被拒绝，`CoordinateSystem` 已回退为普通类，#121 / #122 以
wontfix 关闭。

冻结所针对的运行时偷换 `axes` / `origin` 组件导致变换结果不一致的问题
并未真实发生。这些类的实例构造后视为只读约定：调用方不应在运行时换
`axes` / `origin` 组件，但应用层不强制。

### 决策

**拒绝冻结机制，并回退已落地的 #76 冻结。**

具体动作：

1. `CoordinateSystem` 由 `@dataclass(frozen=True)` 回退为普通类
   （commit `62f7308`）。
2. #121（静态 `Axes` / `Origin` 及其具体子类冻结）不再推进，wontfix。
3. #122（`DynamicAxes` 与冻结基类的边界设计）随 #121 一并 wontfix。
4. 不引入 `CoordinateSystem` 冻结、`Axes` / `Origin` 基类
   `__init_subclass__` 钩子、具体子类 dataclass 改造中的任何一项。

### 理由

1. **没有真实 bug 驱动。** 冻结要解决的运行时偷换组件导致变换结果
   不一致问题，在实际代码中从未出现。`CoordinateSystem` 冻结上线后，
   `cs.axes = X` 这类写法既不在 core 也不在 forces / transfer /
   algorithms 层出现。问题是想象出来的，不是观察到的。

2. **篡改防护不在应用层职责。** 若有人在运行时把 `cs.axes` 换成另一个
   实例，那是程序本身被篡改的场景。用户从 GitHub 获取经验证的代码
   副本即可信任运行环境。应用层冻结防不住程序被改的根因，只是把
   检查点挪到运行时。

3. **冻结的工程成本不低。** `IAU2000EqAxes` 等具体子类有派生字段
   （如由 `time_step` 派生的内部 `IAU2000EqAxes`），dataclass 形态需
   `field(default_factory=...)` + `__post_init__` +
   `object.__setattr__` 等样板；`DynamicAxes`（故意需要可变状态，
   ADR 0007 锁定）与冻结基类的边界要专门设计（组合 / 豁免钩子 /
   平行继承树三选一，#122 全是为此）。这些复杂度为一个不存在的 bug
   服务，得不偿失。

4. **YAGNI。** 代码风格层面，作者不应主动写 mutate 这一点已由
   `tests/algorithm/coordinate/test_coordinate_immutability.py` 的 grep
   守门员覆盖（扫描 `e2m2e/algorithm/coordinate/` 内的 `cs.axes = X` 这类赋值）。
   运行时偷换防护与之不同，本就不在应用层。

### 结果

### 已回退 / 不推进

- `CoordinateSystem` 的 `@dataclass(frozen=True)`：已回退为普通类
  （commit `62f7308`）。
- `TestCoordinateSystemFrozen` 的三个测试：已删除。
- #121（静态 `Axes` / `Origin` 及具体子类全量冻结）：wontfix。
- #122（`DynamicAxes` 与冻结基类边界设计）：wontfix。

### 保留

- `TestCoordinateSystemOrthogonality`、`TestCoordinateSystemTransformVector`
  （零向量）：与冻结无关，保留。
- `tests/algorithm/coordinate/test_coordinate_immutability.py` 的 grep
  守门员：保留。它防的是代码作者主动写 mutate（静态代码风格），
  不是运行时偷换（动态防护）。两者职责不同，前者与冻结无关。

### 对上游决策的影响

| Issue | 原决策 | 本 ADR 后状态 |
|-------|--------|----------------|
| #76 | `CoordinateSystem` 冻结（已 CLOSED） | 回退，原决策被推翻 |
| #121 | 静态 `Axes` / `Origin` 全量冻结 | wontfix |
| #122 | `DynamicAxes` 与冻结基类边界设计 | wontfix（依赖 #121） |
