# ADR 0011：五层架构与激进式全量重命名

**状态**：已采纳（已实施）
**日期**：2026-07-31
**关联**：`docs/architecture/architecture.md`

## 背景

e2m2e 现有顶层结构是历史演进的结果：`core`（系统/坐标系/力模型）、`algorithms`（数值算法）、`transfer`（转移）、`dfh`（DFH 对齐层）、`io`（DFH 格式）、`visualization`、`mbse`、`proximity`、`integrators`。功能已落地（FR1-FR5），但职责边界靠约定不靠结构：

- `dfh/` 是DFH 对齐层，按来源组织而非按领域组织，一个功能层横跨数据、算法、数值三层。
- `io/` 夹在数据层与 DFH 专用之间，定位模糊。
- 没有统一入口（Facade），用户要自己拼 `CR3BP_System → Dynamics → DifferentialCorrection`。
- 没有 MCP，README 的 LLM+Agent 可调用愿景未落地。

需求：设计最终形态的软件系统架构，代码留模板供后续实现，未实现功能在文档和 README 说明。

## 决策

**五层架构**：数据层 `data/` → 数值层 `crates/` → 算法层 `algorithm/` → 接口层 `api/` → 工具层 `tools/`。依赖方向内层不感知外层。

**激进式全量重命名**：现有 `core/algorithms/transfer/dfh/io/visualization` 全部迁入新五层，不保留旧包。`dfh/` 拆散（五个能力归各自领域），`core` 拆散（顶层无 core），`algorithms` → `algorithm`（单数），`io/` 最终不进 e2m2e（DFH 格式是临时脚本）。

**过渡策略**：`sys.modules` 别名保留旧路径；按依赖序分批重命名（先 data → 数值/算法 → api/tools），每批一 commit、跑通测试再动下一批。现有 HEAD 是产品基线。

## 理由

1. **结构强制取代约定**：分层靠目录结构 + 依赖规则（ADR 0012）强制，不靠自觉。
2. **dfh/ 拆散**：轨道设计/保持/转移/预报/时空转换是 e2m2e 自己的领域能力，不是为了对齐 DFH 而存在。DFH 只是开发期参考（ADR 0013）。
3. **io/ 不进 e2m2e**：DFH 格式互操作是开发期临时脚本，最终形态 e2m2e 是独立库（ADR 0013）。
4. **Facade/MCP/CLI**：把库变成可被 LLM 和人类同时调用的工具集，兑现 README 愿景。

## 结果

- 新建 `docs/architecture/architecture.md` 描述最终形态。
- 顶层结构：data/algorithm/api/tools + integrators.py + mbse + _integrators。
- 迁移按依赖序分批进行，每批回归。
