# ADR 0003: Axes, ITRF93 defaults, GMAT-compatible Earth orientation / 坐标轴、ITRF93 默认值与 GMAT 兼容的地球定向

[English](#adr-0003-axes-itrf93-defaults-gmat-compatible-earth-orientation) | [简体中文](#中文)

## English

**Status**: Adopted
**Date**: 2026-06-12
**Related Issues**: #62, #80

### Context

Issue #62 introduces GMAT-style coordinate systems, composed from independent
axes and origins. The original request mixes a simplified IAU 2006
implementation with SPICE validation at extremely tight tolerances. That is
not a stable contract for the Earth-fixed frame: a simplified native model can
neither independently match SPICE's high-precision Earth orientation at
`1e-12` nor account for EOP data.

GMAT R2026a uses `ITRF93` as its high-precision Earth SPICE frame and treats
`IAU_EARTH` as low precision. GMAT's native ITRF path also has several
compatibility-specific behaviors: A1MJD inputs, C04 EOP parsing, linear
interpolation of `UT1-UTC` and polar motion, non-interpolated `LOD`, and
optional clamping outside EOP coverage.

### Decision

1. **The default high-precision ITRF is SPICE-backed `ITRF93`.**
   - Public/default ITRF factories and compatibility `ITRFAxes` point to
     `ITRFSpiceAxes(frame="ITRF93")`.
   - `IAU_EARTH` never acts as a silent fallback for high-precision ITRF.
   - `ITRFApproxAxes` remains low-precision/educational only.

2. **GMAT-compatible native ITRF is opt-in.**
   - `GMATITRFAxes` is an independent axes implementation.
   - First-phase native reduction uses a pyerfa/SOFA-based `ErfaXysProvider`
     (an implementation of `XysProvider`) supplying IAU `X, Y, s`.
   - If exact table-level agreement with GMAT is later required, a
     `GMATXysProvider` can replace that source.

3. **State transformation is rotation-rate based.**
   - `Axes.rotation_matrix(et)` returns `R`, with the convention
     `r_icrf = R @ r_axes`.
   - `Axes.rotation_and_rate(et)` returns `(R, Rdot)`, with
     `v_icrf = R @ v_axes + Rdot @ r_axes`.
   - `Axes.state_transform_matrix(et)` derives from `(R, Rdot)`.
   - `CoordinateSystem.transform_state()` prefers `(R, Rdot)` before falling
     back to any angular-velocity compatible path.

4. **Public coordinate epoch inputs remain ET seconds.**
   - Public axes and coordinate-system APIs take SPICE ET seconds.
   - GMAT A1MJD support is lower-level and test-facing, used only for
     consistency checks.

5. **Verification is two-tiered.**
   - `ITRFSpiceAxes("ITRF93")` verifies against `spiceypy.pxform/sxform` to
     `<1e-12` when high-precision kernels are available.
   - `GMATITRFAxes` initially sanity-checks against SPICE `ITRF93` at about
     `1e-7`; main native-chain verification is carried by parser/time/EOP/
     per-stage tests.

6. **Test fixtures and data strategy are explicit.**
   - Committed slim text fixtures cover J2000, the 2017 leap-second boundary,
     and a 2026-06-12 window.
   - Full GMAT data is opt-in via `GMAT_DATA_DIR`.
   - Optional high-precision Earth BPC checks skip explicitly when
     unavailable.
   - Missing required committed fixtures or kernels raise clear errors.

7. **Errors are explicit; precision never silently degrades.**
   - A missing `ITRF93` kernel raises a coordinate error with actionable hints.
   - Missing GMAT native data raises a data error at construction or query
     time.
   - Out-of-range EOP raises by default.
   - GMAT-style clamping exists only behind an explicit compatibility option.

8. **System integration offers no frame-conversion shortcuts.**
   - `System.coordinate_system` holds an optional coordinate system so force
     models can query which frame input states are in.
   - Conversion entry points live at the `CoordinateSystem` layer: callers use
     `system.coordinate_system.transform_state()` / `transform_vector()`
     directly or construct their own `CoordinateSystem`.
   - `System` does **not** provide a `transform()` shortcut.

   > **Revision note (2026-06-15, issue #79)**: original decision 8 specified
   > `System.transform()` as a thin delegate over
   > `CoordinateSystem.transform_state()`. After landing, real production code
   > (drag, gravity, thrust, SRP models) all bypassed `System.transform()` and
   > used `system.coordinate_system.transform_*` directly. Re-discussion
   > concluded the thin delegation was needless indirection; the shortcut is
   > unnecessary. #79 removed `System.transform()` and this clause was revised
   > accordingly. The original `System.transform()` landed in #90 on an
   > under-considered design call now reversed.

### Consequences

#### Added

- A stable contract covering SPICE-backed defaults, GMAT-compatible native
  behaviors, fixtures, and tolerances.
- A rotation-rate-based axes abstraction serving both SPICE `sxform` and GMAT
  native `Rdot`.

#### Unchanged

- Public coordinate APIs keep taking ET seconds.
- Approximate ITRF remains explicitly labeled low-precision.

#### Follow-up work

- If exact agreement with GMAT XYS interpolation is required later, implement
  a GMAT-table-backed `XysProvider`.
- Tighten native-vs-SPICE tolerances only after equivalence of data sources,
  interpolation, and model versions is proven.

## 中文

**状态**：已采纳
**日期**：2026-06-12
**关联 Issue**：#62, #80

### 背景

Issue #62 引入 GMAT 风格的坐标系，由独立的坐标轴与原点组合而成。原始请求把一个简化的 IAU 2006 实现与 SPICE 在极紧容差下的验证混在一起。这对地固系而言不是一个稳定的契约：一个简化的原生模型既无法在 `1e-12` 量级上独立匹配 SPICE 高精度地球定向，又省略了 EOP 数据。

GMAT R2026a 用 `ITRF93` 作为高精度地球 SPICE 帧，把 `IAU_EARTH` 当作低精度。GMAT 的原生 ITRF 路径还有若干兼容性特有的行为：A1MJD 输入、C04 EOP 解析、`UT1-UTC` 与极移的线性插值、不插值的 `LOD`，以及在 EOP 覆盖范围之外可选的钳位（clamping）。

### 决策

1. **默认的高精度 ITRF 采用 SPICE 支撑的 `ITRF93`。**
   - 公开/默认的 ITRF 工厂与兼容性 `ITRFAxes` 指向 `ITRFSpiceAxes(frame="ITRF93")`。
   - `IAU_EARTH` 绝不作为高精度 ITRF 的静默回退。
   - `ITRFApproxAxes` 仍仅用于低精度/教学。

2. **GMAT 兼容的原生 ITRF 需显式选择。**
   - `GMATITRFAxes` 是一个独立的坐标轴实现。
   - 第一阶段的原生归约使用基于 pyerfa/SOFA 的 `ErfaXysProvider`（`XysProvider` 的实现）提供 IAU 的 `X, Y, s`。
   - 若日后要求与 GMAT 表格精确一致，可用一个 `GMATXysProvider` 替换该来源。

3. **状态变换基于旋转率。**
   - `Axes.rotation_matrix(et)` 返回 `R`，约定 `r_icrf = R @ r_axes`。
   - `Axes.rotation_and_rate(et)` 返回 `(R, Rdot)`，约定 `v_icrf = R @ v_axes + Rdot @ r_axes`。
   - `Axes.state_transform_matrix(et)` 由 `(R, Rdot)` 推导。
   - `CoordinateSystem.transform_state()` 优先使用 `(R, Rdot)`，再回退到任何角速度兼容路径。

4. **公开的坐标历元输入仍为 ET 秒。**
   - 公开的坐标轴与坐标系 API 接收 SPICE ET 秒。
   - GMAT A1MJD 支持是底层、面向测试的，仅用于一致性核对。

5. **验证分两级。**
   - `ITRFSpiceAxes("ITRF93")` 在高精度内核可用时对照 `spiceypy.pxform/sxform` 验证至 `<1e-12`。
   - `GMATITRFAxes` 起初对照 SPICE `ITRF93` 在约 `1e-7` 量级做合理性核对，主要的原生链验证由解析器/时间/EOP/各阶段测试承担。

6. **测试夹具与数据策略显式化。**
   - 提交精简的文本夹具，覆盖 J2000、2017 闰秒边界、2026-06-12 窗口。
   - 完整 GMAT 数据通过 `GMAT_DATA_DIR` 按需启用。
   - 可选的高精度地球 BPC 校验在不可用时明确跳过。
   - 缺失必需的已提交夹具或内核时抛出明确错误。

7. **错误明确，绝不自动降精度。**
   - 缺失 `ITRF93` 内核时抛出坐标错误，并给出可操作的提示。
   - 缺失 GMAT 原生数据时，在构造或查询阶段抛出数据错误。
   - EOP 越界默认抛错。
   - GMAT 风格的钳位只在显式兼容选项下可用。

8. **系统集成不提供坐标转换快捷方式。**
   - `System.coordinate_system` 存放一个可选的坐标系，供力模型查询输入状态位于哪个坐标系。
   - 坐标转换入口在 `CoordinateSystem` 层：调用方直接用 `system.coordinate_system.transform_state()` / `transform_vector()`，或自行构造 `CoordinateSystem` 实例调用。
   - `System` **不**提供 `transform()` 快捷方式。

   > **修订记录（2026-06-15，issue #79）**：原决策第 8 条写 `System.transform()` 薄委托给 `CoordinateSystem.transform_state()`。落地后实际生产代码（阻力、重力、推力、光压模型）全部绕过 `System.transform()`，直接用 `system.coordinate_system.transform_*`。经重新讨论判定：薄委托层只是多余间接，快捷方式不必要。#79 移除 `System.transform()` 方法，本条同步修订。原 `System.transform()` 落地于 #90，未经深思熟虑的设计判断现已反转。

### 结果

### 新增

- 一份关于 SPICE 支撑默认值、GMAT 兼容原生行为、夹具与容差的稳定契约。
- 一个基于旋转率的坐标轴抽象，同时适用于 SPICE `sxform` 与 GMAT 原生 `Rdot`。

### 不变

- 公开坐标 API 继续接收 ET 秒。
- 近似 ITRF 仍明确标注为低精度。

### 后续工作

- 若日后要求与 GMAT XYS 插值精确一致，再实现一个 GMAT 表格支撑的 `XysProvider`。
- 只有在数据来源、插值与模型版本被证明等价之后，才收紧原生与 SPICE 之间的容差。
