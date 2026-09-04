# ADR 0047: 轨道库归属——库不维护数据库，提供建库基础设施

**状态**：已采纳（已实现）
**日期**：2026-09-04
**相关 issue**：#632
**相关**：ADR 0031（决策 5、8 在此收紧）、ADR 0036（决策 5 废止；数据集本身
延续）、ADR 0045（决策 8 的随包分发废止；记录粒度不变）、ADR 0024（错误
契约，新增 `CATALOG_NOT_CONFIGURED`）、#625（症状之鉴）、#601（Config
载荷契约）。

## 背景

e2m2e 是库，却在运行期持有并维护持久状态，共三处：

1. **隐式默认库目录**：`catalog_dir` 缺省 `./catalog`——相对当前工作目录，
   跑哪建哪，用户往往不知道它存在；
2. **首用静默导入基线**：第一次碰库就把包内 592 条基线成员物化进用户库
   （1184 个文件），并跨版本维护基线版本对齐（删旧族重展开）；
3. **计算任务默认自动入库**：design/control/transfer 成功即写库，
   `catalog_enabled` 默认开。

持有状态的直接代价：schema 演进变成库的迁移责任。ADR 0045 把记录升到
v2 时，持有 v1 旧库的用户任何轨道库操作都裸崩——#625 修的是报错 UX，
根因是库不该拥有用户库的生命周期。

基础设施其实已经齐备：`CatalogStore` 引擎可对任意目录显式建库，记录
文件是事实来源、索引永远可重建；分发包展开（`bundle.py`）与导入
（`baseline.py`）本是纯工具，只是被"首用自动执行"包成了维护行为。

## 决策

### 1. 库的职责边界 = 基础设施

保留并继续维护：引擎（`CatalogStore`）、格式（v2 记录、v1 束传输格式）、
查询索引、导入导出、显式入库接缝（`auto_ingest` 系）。不再做：默认目录、
默认导入、默认写入、用户库 schema 迁移（跨 schema 旧产物仍是"删除后
重算"，ADR 0045）。

### 2. 自动入库默认关

`catalog_enabled` 默认 `False`，环境变量 `$E2M2E_CATALOG_ENABLED` 可显式
开启。计算结果只随响应返回；入库是调用方的显式决定。未开启时响应的
`record_id` / `family_id` 为 `None`。design→control（`input_record_id`）
链路要求调用方先开启入库。

### 3. 无隐式库目录

`catalog_dir` 默认 `None`（`$E2M2E_CATALOG_DIR` 仍是显式指定通道之一）。
未指定目录时，一切库操作（查询、入库、取记录）抛
`CATALOG_NOT_CONFIGURED`（ADR 0024 契约内，`INVALID_INPUT`），消息写明
两种指定方式，不建目录、不猜路径。

### 4. 基线数据集出包，Release 分发

wheel 不再携带 `catalog_baseline/`。数据集以 GitHub Release 资产（束
文件 zip）分发；调用方下载后解压，经 `import_baseline(store, source_dir)`
显式导入建库。`import_baseline` 的源参数变为必填，源目录缺失报
`FileNotFoundError` 而非静默跳过。仓库内保留该数据集，作为分类学全成员
回归（ADR 0042）的夹具与 Release 资产的生成源（`make catalog-baseline`）。

### 5. Config 载荷契约变更（#601）

删 `catalog_baseline_import`；`catalog_dir` 变 `str | None`；
`catalog_enabled` 默认 `False`。`from_payload` 未知字段照旧报错——跨
版本序列化的配置不兼容是预期行为，旧 worker 载荷（含
`catalog_baseline_import`）会被拒绝。

## 备选

- **彻底移除自动入库接缝**（只留显式 put 工具）：更纯，但
  design→control 链路要求调用方手工入库，MCP/GUI 改动面大；接缝保留、
  默认关，两全。
- **包内只读库**（查询直接打开包资源，免物化）：需引擎支持只读打开包
  资源、束/库双形态查询合并，复杂度高；Release 资产 + 显式导入达到同一
  自由度（数据要不要、放哪、何时导，调用方定）。
- **静默跳过或自动删旧库记录**（#625 的另一修法）：ADR 0045 实现注已
  否决，维持。

## 后果

### 新增

- `CATALOG_NOT_CONFIGURED` 错误码；`$E2M2E_CATALOG_ENABLED` 环境开关；
  基线 Release 资产管线（`make catalog-baseline` 生成 + zip 上传）。

### 变更

- 默认行为三处：不自动入库、无默认目录、不自动导基线（breaking）。
- wheel 不再含基线数据集（瘦身约 3.5 MB）。
- ADR 0031 决策 5（自动入库）的默认值、决策 8 的无条件入库；ADR 0036
  决策 5（首用导入）；ADR 0045 决策 8 的"随包分发"表述——均废止，以本
  ADR 为准。
- Config 载荷字段集（决策 5）。

### 不变

- `CatalogStore` 引擎与存储布局；记录 schema v2；束 v1 传输格式；
  显式开启后的 `auto_ingest` 行为；`import_baseline` 的幂等与基线版本
  对齐语义（对显式导入仍生效）；打开即失败的旧产物处置（ADR 0045）。

### 代价

- 升级是 breaking：依赖默认入库的调用方升级后 `record_id` 为 `None`、
  未设目录的库操作报错；迁移步骤进 CHANGELOG 升级注意。
- 基线查询多一步（下载 Release 资产 + 显式导入），换来库不再替用户
  决定磁盘上有什么。
