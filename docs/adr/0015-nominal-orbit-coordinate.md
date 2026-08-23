# ADR 0015：NominalOrbit 名义轨道契约与坐标转换抽象

**状态**：已采纳（已实施）
**日期**：2026-07-31
**关联**：ADR 0011（五层架构）、Gómez《平动点任务设计》vol I §8.2.3、Soffel《时空参考系》

## 背景

任务轨道设计（FR1）与轨道保持（FR2）之间需要一个数据契约：设计产出的标称轨道如何被保持控制消费。Gómez vol I §8.2.3 给出规范：**名义轨道 = 等间距历元状态表 + Floquet 基 + 投影因子表 + 高次插值（Lagrange r=5~6）**。

另一问题：时空参考系。现有坐标转换类接口各异（SynodicJ2000System/GCRSEBCRSSystem/Axes），草案曾提议新增 Frame 抽象统一之。Soffel 结论：转换链算法归内核、数据（EOP/闰秒/历表）归数据层。

## 决策

1. **NominalOrbit 是 FR1↔FR2 数据契约**，归 `data/types/trajectory.py`：等间距历元状态表 + Floquet 基 + 投影因子表 + 高次插值器。**Floquet 基 + 投影因子由 FR1 预计算**（design_orbit 产出自带），控制全程插值不复算。control_orbit 控制律从现算 STM 演进为插值投影因子。
2. **时空参考系 = 强化现有 Axes/Origin/CoordinateSystem 抽象**（不新增 Frame 抽象）。所有坐标系（含 synodic↔J2000、GCRS↔EBCRS）表达为 Axes + Origin + CoordinateSystem；时空间联合转换（同时换时间尺度）作为 CoordinateSystem 扩展方法。要补：时间尺度作为参考系的一部分统一。
3. **时间尺度并入 EphemerisProvider**（不单独 TimeSystem 类）。TDB 作动力学统一时间：算法层/数值层内部统一用 ET(TDB) 或 JD_TDB；只有接口边界才转 UTC。
4. **转换算法归 `algorithm/coordinate/`**，`frames/` 只留数据（EOP、闰秒、历表句柄）。转换算法最终留 Python（Axes 子类方法），Rust 下沉是后续性能优化。

## 理由

1. **预计算投影因子**：站保控制每次控制时刻都要用投影因子，预计算一次、控制全程插值，明显更优（Gómez 8.2.3）。
2. **不新增 Frame 抽象**：ADR 0003/0007 已投资 Axes/Origin/CoordinateSystem，再引入平行 Frame 抽象会重复，避免过度抽象（重复比错误抽象便宜）。
3. **TDB 统一**：TCB 有 0.47s/年漂移，TDB 只剩 <2ms 周期项（Soffel），动力学时间统一用 TDB。

## 结果

- `data/types/trajectory.py` 定义 NominalOrbit（含 Floquet 基 + 投影因子 + 插值器）。
- `data/kernels/provider.py` 的 EphemerisProvider 含时间/状态/帧三类方法，单点 + 批量。
- `data/frames/` 只留数据，转换算法在 `algorithm/coordinate/`。
