# ADR 0005：TwoLevelMultipleShooting 作为独立算法

**状态**：已采纳
**日期**：2026-05-13
**编号说明**：本篇为后补的历史决策，实际决策时间介于 ADR 0001 与 0002 之间。

把 `TwoLevelMultipleShooting` 作为 `e2m2e.algorithms` 中的一个独立算法加入，而不是扩展或继承现有的 `MultipleShooting`。两层修正与现有的全状态多重打靶求解器在自由变量、残差、雅可比结构和结果诊断上都不同；把它分开，既保住了通用求解器更简单的 API，又给转移设计代码提供了一组稳定的 API，用来承载原有的两层星历修正语义。

## 修订（2026-08-13，关联 ADR 0006 修订）

`TwoLevelMultipleShooting` 连同 `ephemeris_correction` 分发子包已删除：设计链路
统一走 Rust 多重打靶（``multiple_shooting_correct_py``，segmented 与稳定轨道
默认路径），速度连续由 ``vel_weight`` 加权在 Rust 侧一并收敛。本 ADR 所定
的独立算法安排已无消费者，决策对象撤销；`MultipleShooting` 本身保留（transfer/hohmann
等非设计链路仍使用）。
