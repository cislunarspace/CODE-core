# ADR 0019: Drag Rust port uses ITRF93 pxform frame rotation / drag Rust 移植用 ITRF93 pxform 帧旋转（替 ITRFApproxAxes）

[English](#adr-0019-drag-rust-port-uses-itrf93-pxform-frame-rotation-replacing-itrfapproxaxes) | [简体中文](#中文)

## English

**Status**: Adopted
**Date**: 2026-08-07
**Related**: ADR 0003 (ITRF93 defaults & GMAT compatibility), ADR 0018
(Jacobian ∂a/∂v), issue #315, issue #317
**Related code**: `crates/e2m2e-forces/src/forces/drag.rs`,
`e2m2e/algorithm/forces/drag.py`

### Context

Drag physics is computed in the **Earth-fixed frame** (ITRF): the atmosphere
co-rotates with Earth, so relative velocity = spacecraft ITRF velocity. The
drag pipeline is thus a round trip of frame rotations: J2000 propagation frame
→ ITRF → drag formula → J2000.

Python's ``DragModel.compute_acceleration`` (`drag.py`) uses
``ITRFApproxAxes`` for this rotation. ADR 0003 decision 1 already labels
``ITRFApproxAxes`` as low-precision/educational with SPICE-backed `ITRF93` as
the high-precision default. Python drag keeping approximate axes is legacy:
drag was long a teaching/LEO-scale example where ``ITRFApproxAxes`` sufficed.

The drag Rust port (issue #315 series) brings `DragModel` into compiled force
models sharing the same SPICE FFI as `GravityField` (``pxform`` / kernel pool).
Drag's frame rotation then had two options:

- **A. Copy Python**: reimplement the ``ITRFApproxAxes`` approximate reduction
  in Rust (simplified IAU 2006 + linear LOD etc.), solely for bit-level parity
  with Python.
- **B. Align with ADR 0003 defaults**: use `ITRF93`'s ``pxform`` directly
  (same rotation path as `GravityField`).

`drag.rs`'s header comment calls this "decision 1b", but it had no documented
home (issue #317 item 1.2 noted this). This ADR provides one.

### Decision

**Option B**: the drag Rust path uses `ITRF93` ``pxform``; no reimplementation
of ``ITRFApproxAxes``.

1. **`drag_accel` rotation via ITRF93.**
   - `lookup_frame_matrix("ITRF93", propagation_frame, et)` prefers the
     ephemeris pre-sampling cache (ADR 0016), falling back to
     ``pxform("ITRF93", propagation_frame, et)``.
   - Same pattern as `GravityField`'s `pxform` usage
     (``r_itrf = Rᵀ·r_j2000``, ``a_j2000 = R·a_itrf``).

2. **Python path unchanged.**
   - `DragModel.compute_acceleration` keeps `ITRFApproxAxes`. Under current
     reality — drag needing SPICE always goes Rust — the two paths don't
     coexist: `DragModel.to_rust_spec` returns None when `system.spice` is
     missing, `ForceModel` falls back to the Python path, and without SPICE
     there is no `ITRF93`; `ITRFApproxAxes` is a reasonable degradation.

3. **`∂a/∂v` is orthogonal to frames.**
   - Drag's velocity dependence (``a ∝ |v|·v``) is physics, independent of the
     rotation choice. The triple-extension interface lives in ADR 0018, out of
     scope here.

### Rationale

1. **Avoid re-implementing an explicitly low-precision frame inside Rust.**
   ADR 0003 confines `ITRFApproxAxes` to low-precision/education. The Rust
   compiled path is the high-precision production path; porting a reduction our
   own ADR classifies as low precision is self-contradictory, and maintaining
   two Earth orientations (ITRF93 + approximation) doubles the burden.

2. **Shares the rotation path with `GravityField`.** Both are body-fixed
   forces (density vs spherical harmonics), both need J2000↔ITRF rotations.
   Using the same `pxform` path shares cache hits (ADR 0016) and reduces SPICE
   FFI calls. Forking would waste landed caching infrastructure.

3. **Bit-level parity with Python was never the goal.** The Rust compiled path
   aims to be faster and no worse than Python (ADR 0002 revision 2), not a bit-
   for-bit clone. `ITRF93` beats `ITRFApproxAxes` in accuracy; the divergence
   direction is Rust closer to truth — consistent with ADR 0003's accuracy
   ladder.

4. **Degradation semantics self-consistent.** The Python path only engages
   without SPICE; without SPICE, `ITRF93` is unavailable anyway (ADR 0003 item
   7 raises on missing kernels, never silently degrading precision).
   `ITRFApproxAxes` is then the only available option — not violating ADR 0003;
   it exists precisely as the sans-SPICE low-precision fallback.

### Boundaries

- **Drag only.** Other body-fixed forces (`GravityField`, `EarthTide`) follow
  ADR 0003 on their own; unaffected here.
- **No change to Python `DragModel`.** Its `ITRFApproxAxes` stays as sans-SPICE
  degradation. If Python drag later switches to `ITRF93`, evaluate separately
  (at that point both paths match in accuracy; this ADR's dual-path notes could
  be removed).
- **No GMAT-native ITRF.** ADR 0003 item 2's `GMATITRFAxes` is an independent
  GMAT-compatible frame; drag doesn't use it.

### Consequences

#### Added

- Drag Rust path uses `ITRF93` `pxform`, aligned with ADR 0003 defaults.
- Decision 1b has a documented home (this ADR); `drag.rs` comments no longer
  dangle.

#### Unchanged

- Python `DragModel.compute_acceleration`'s `ITRFApproxAxes` path (sans-SPICE
  degradation).
- Drag physics formulas, density interface, `∂a/∂v` Jacobian (ADR 0018).
- All ADR 0003 decisions (this ADR specializes them for drag without revising
  the parent).

#### Trade-offs

- **Dual-path accuracy divergence.** With SPICE present, Rust (`ITRF93`) vs
  Python (`ITRFApproxAxes`) drag accelerations differ slightly in theory.
  Negligible at LEO scale (the approximate axes' rotation error is
  higher-order small for drag magnitudes); moreover the paths never coexist
  now (SPICE ⇒ Rust). If Python-path ITRF93 accuracy is ever needed, a separate
  change follows.

### Revision (2026-08-12, ADR 0020 decision 4)

**Missing SPICE no longer degrades to ITRFApproxAxes**: when `system.spice`
is absent, `DragModel.to_rust_spec` returns None and `ForceModel` explicitly
raises a capability error (issue #378) instead of silently falling back to the
Python slow path (decision 2's claim that ITRFApproxAxes was a reasonable
sans-SPICE degradation is void). The Python reference implementation was
removed wholesale by issue #378 — `DragModel` no longer carries
`compute_acceleration` (decision 2 and the keep-Python-path clauses lapse);
`ITRFApproxAxes` remains only at the coordinate layer
(`e2m2e/algorithm/coordinate/standard_axes.py`) for low-precision/teaching per
ADR 0003, absent from every drag propagation path.

## 中文

**状态**：已采纳
**日期**：2026-08-07
**关联**：ADR 0003（ITRF93 默认值与 GMAT 兼容）、ADR 0018（Jacobian ∂a/∂v）、issue #315、issue #317
**关联代码**：`crates/e2m2e-forces/src/forces/drag.rs`、`e2m2e/algorithm/forces/drag.py`

### 背景

大气阻力的物理在**地固系**（ITRF）中计算：大气随地球自转，相对速度 = 航天器 ITRF 速度。阻力管线因此是一次 J2000 传播系 → ITRF → 阻力公式 → J2000 的帧旋转往返。

Python ``DragModel.compute_acceleration``（`drag.py`）用 ``ITRFApproxAxes`` 做这次旋转。ADR 0003 第 1 条已把 ``ITRFApproxAxes`` 明确标注为低精度/教学，高精度默认是 SPICE 支撑的 ``ITRF93``。Python drag 沿用近似轴是历史遗留：drag 长期是教学/LEO 量级示例，``ITRFApproxAxes`` 够用。

drag Rust 移植（issue #315 系列）把 ``DragModel`` 纳入 compiled 力模型，与 ``GravityField`` 共享同一套 SPICE FFI（``pxform`` / 内核池）。此时 drag 的帧旋转有两个选择：

- **A. 照搬 Python**：在 Rust 侧重新实现 ``ITRFApproxAxes`` 的近似归约（IAU 2006 简化 + 线性 LOD 等），仅为与 Python 逐位一致。
- **B. 对齐 ADR 0003 默认**：drag 直接用 ``ITRF93`` 的 ``pxform``（与 ``GravityField`` 同一帧旋转路径）。

`drag.rs` 顶部注释称此为决策 1b，但该决策此前无文档落点（issue #317 第 1.2 项指出）。本 ADR 补上。

### 决策

**选 B**：drag Rust 路径用 ``ITRF93`` ``pxform`` 帧旋转，不重新实现 ``ITRFApproxAxes``。

1. **``drag_accel`` 帧旋转走 ITRF93。**
   - ``lookup_frame_matrix("ITRF93", propagation_frame, et)`` 优先走星历预采样缓存（ADR 0016），未命中走 ``pxform("ITRF93", propagation_frame, et)``。
   - 与 ``GravityField`` 的 ``pxform`` 模式一致（``r_itrf = Rᵀ·r_j2000``，``a_j2000 = R·a_itrf``）。

2. **Python 路径不改。**
   - ``DragModel.compute_acceleration`` 仍用 ``ITRFApproxAxes``。两条路径在 drag 需 SPICE 必走 Rust 的现状下不并存：``DragModel.to_rust_spec`` 在 ``system.spice`` 缺失时返回 ``None``，``ForceModel`` 回退 Python 路径，此时没有 SPICE 也谈不上 ``ITRF93``，``ITRFApproxAxes`` 是合理的降级。

3. **``∂a/∂v`` 与坐标系正交。**
   - drag 的速度依赖（``a ∝ |v|·v``）是物理，与帧旋转选择无关。``∂a/∂v`` 接口扩三元组见 ADR 0018，不在本 ADR 范围。

### 理由

1. **避免在 Rust 里重实现一个明确标注低精度的轴系。** ADR 0003 已定 ``ITRFApproxAxes`` 仅用于低精度/教学。Rust compiled 路径是高精度生产路径，重新移植一个被自家 ADR 划为低精度的归约，方向自相矛盾；维护两套地球定向（ITRF93 + 近似）也加倍负担。

2. **与 ``GravityField`` 共享帧旋转路径。** drag 与 ``GravityField`` 都是 body-fixed 力（一个用密度，一个用球谐），都需 J2000↔ITRF 旋转。两者用同一 ``pxform`` 路径，共享星历缓存命中（ADR 0016），减少 SPICE FFI 调用。分叉反而浪费已落地的缓存基础设施。

3. **Python 逐位一致并非目标。** Rust compiled 路径的定位是更快且不劣于 Python（ADR 0002 修订 2），不是 Python 的逐位复刻。``ITRF93`` 比 ``ITRFApproxAxes`` 更准，分歧方向是 Rust 更接近真值，符合 ADR 0003 的精度阶梯。

4. **降级语义自洽。** Python 路径只在无 SPICE 时启用；无 SPICE 时 ``ITRF93`` 本就不可用（ADR 0003 第 7 条缺失 ITRF93 内核抛坐标错误，绝不自动降精度）。``ITRFApproxAxes`` 在这条降级路径上是唯一可用选项，不违反 ADR 0003：它本就是为无 SPICE 保留的低精度退路。

### 边界

- **仅 drag。** 本决策只针对 drag 的帧旋转。其他 body-fixed 力（``GravityField``、``EarthTide``）的坐标系选择各自遵循 ADR 0003，不受本 ADR 影响。
- **不改 Python ``DragModel``。** Python 路径的 ``ITRFApproxAxes`` 保留作无 SPICE 降级。若日后 Python drag 也切 ``ITRF93``，需单独评估（届时两条路径精度一致，可考虑移除本 ADR 的双路径说明）。
- **不引入 GMAT 原生 ITRF。** ADR 0003 第 2 条的 ``GMATITRFAxes`` 是独立的 GMAT 兼容轴系，drag 不用它。

### 结果

### 新增

- drag Rust 路径用 ``ITRF93`` ``pxform``，与 ADR 0003 默认对齐。
- 决策 1b 有了文档落点（本 ADR），`drag.rs` 注释引用不再悬空。

### 不变

- Python ``DragModel.compute_acceleration`` 的 ``ITRFApproxAxes`` 路径（无 SPICE 降级）。
- drag 的物理公式、大气密度接口、``∂a/∂v`` 雅可比（属 ADR 0018）。
- ADR 0003 全部决策（本 ADR 是其在 drag 上的具体化，不修订母决策）。

### 取舍

- **双路径精度分歧。** Rust（``ITRF93``）与 Python（``ITRFApproxAxes``）drag 加速度在有 SPICE 时理论上略有差异。在 LEO 量级可忽略（``ITRFApproxAxes`` 的旋转误差对阻力量级是高阶小量）；且现状下两者不并存（有 SPICE 必走 Rust）。若未来需要 Python 路径也达 ``ITRF93`` 精度，再开单独改动。

### 修订（2026-08-12，ADR 0020 决策 4）

**SPICE 缺失不再降 ITRFApproxAxes**：`DragModel.to_rust_spec` 在 `system.spice` 缺失时返回 `None`，`ForceModel` 显式抛能力错误（issue #378），不再静默回退 Python 慢路径（原决策 2 中无 SPICE 时 ITRFApproxAxes 是合理降级的说法作废）。Python 侧参考实现已随 issue #378 整体移除，`DragModel` 不再保留 `compute_acceleration`（决策 2 与结果/不变中 Python 路径保留的条目随之失效）；`ITRFApproxAxes` 仅留在坐标层（`e2m2e/algorithm/coordinate/standard_axes.py`）作低精度/教学用途（ADR 0003），不再出现在 drag 的任何传播路径上。
