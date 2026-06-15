# ADR 0003：坐标轴、ITRF93 默认值与 GMAT 兼容的地球定向

**状态**：已采纳
**日期**：2026-06-12
**关联 Issue**：#62, #80

## 背景

Issue #62 引入 GMAT 风格的坐标系，由独立的坐标轴与原点组合而成。原始请求把一个简化的 IAU 2006 实现与 SPICE 在极紧容差下的验证混在一起。这对地固系而言不是一个稳定的契约：一个简化的原生模型既无法在 `1e-12` 量级上独立匹配 SPICE 高精度地球定向，同时又省略 EOP 数据。

GMAT R2026a 用 `ITRF93` 作为高精度地球 SPICE 帧，把 `IAU_EARTH` 当作低精度。GMAT 的原生 ITRF 路径还有若干兼容性特有的行为：A1MJD 输入、C04 EOP 解析、`UT1-UTC` 与极移的线性插值、不插值的 `LOD`，以及在 EOP 覆盖范围之外可选的钳位（clamping）。

## 决策

1. **默认的高精度 ITRF 采用 SPICE 支撑的 `ITRF93`。**
   - 公开/默认的 ITRF 工厂与兼容性 `ITRFAxes` 指向 `ITRFSpiceAxes(frame="ITRF93")`。
   - `IAU_EARTH` 绝不作为高精度 ITRF 的静默回退。
   - `ITRFApproxAxes` 仍仅用于低精度/教学。

2. **GMAT 兼容的原生 ITRF 需显式选择。**
   - `GMATITRFAxes` 是一个独立的坐标轴实现。
   - 第一阶段的原生归约使用基于 pyerfa/SOFA 的 `XysProvider` 提供 IAU 的 `X, Y, s`。
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

8. **系统集成是薄委托。**
   - `System.coordinate_system` 存放一个可选的坐标系。
   - `System.transform()` 委托给 `CoordinateSystem.transform_state()`，不重复坐标数学。

## 结果

### 新增

- 一份关于 SPICE 支撑默认值、GMAT 兼容原生行为、夹具与容差的稳定契约。
- 一个基于旋转率的坐标轴抽象，同时适用于 SPICE `sxform` 与 GMAT 原生 `Rdot`。

### 不变

- 公开坐标 API 继续接收 ET 秒。
- 近似 ITRF 仍明确标注为低精度。

### 后续工作

- 若日后要求与 GMAT XYS 插值精确一致，再实现一个 GMAT 表格支撑的 `XysProvider`。
- 只有在数据来源、插值与模型版本被证明等价之后，才收紧原生与 SPICE 之间的容差。
