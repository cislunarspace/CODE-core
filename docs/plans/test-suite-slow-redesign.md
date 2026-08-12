# 测试套件重设计记录

**立项**：#361
**关联**：ADR 0021（测试套件按功能类目组织）、ADR 0013（验证策略）、#359（标记重组落地）
**日期**：2026-08-09

## 决策

测试按它证明的事实划分为 `theory`、`integrator`、`force`、`data`、
`orchestration`、`interface`、`aux` 七类；不再按运行速度维护 `slow` 套件。

运行时间不是正确性类别。一个测试若只能证明字段契约，应下沉为直接构造的
`data`/`interface` 测试；若验证算法编排或物理结论，保留对应功能类测试，
但不以 `slow` 标记把它隔离成另一条测试门。未完成能力仍可使用专门 marker
控制默认测试门，例如 `low_thrust`。

## 已完成处置

- 原 Lissajous/Triangular、segmented、PAL、normal-form 有界性、homotopy
  真实内核、WSB 真搜索与 FiniteBurn 长弧传播的 slow 用例已删除。
- Frozen orbit 的外部工具漂移 oracle 已删除；短弧结果结构与传播链路由现有
  集成和单元测试覆盖。
- `WsbTransferDetails` 字段契约已下沉到 `tests/api/test_facade.py`，不再依赖
  WSB 搜索。
- `compute_lissajous_bounded_trajectory` 的返回结构仍由快速契约测试覆盖。
- homotopy、低推力等未完成功能保留各自的输入校验和快速行为测试，不保留
  端到端长弧测试。
- `slow` marker、默认 `not slow` 过滤以及独立 slow 套件均已移除；默认测试门
  只排除 `low_thrust`。

## 验收

- 每个保留测试可按功能类说明其证明的物理或契约事实。
- 默认测试不再按运行速度过滤。
- 运行时间回归用 `pytest --durations` 定位到具体测试处理，不重建速度分层。
- `grep -rn 'L3\|三层分层\|scenarios，端到端' tests/` 无残留。
