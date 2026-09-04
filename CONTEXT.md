# 统一领域模型术语表

本文档记录 e2m2e 项目的规范领域术语与统一语言（Ubiquitous Language），
遵循 ADR 0011（五层架构）与 ADR 0021（测试套件功能类目）。

## 测试与质量保证

### 功能类标记（Functional Marker）
每个测试用例**恰好拥有一个**主分类标记，描述**验证什么**（封闭 7 类：`theory`、`integrator`、`force`、`data`、`orchestration`、`interface`、`aux`），正交标记（`spice`、`low_thrust`）按需附加。废除速度分层与 e2e 类别（ADR 0021、ADR 0025）。

### 真调用最小覆盖（Minimal Real-Call Coverage）
编排器与接口层（`orchestration`/`interface`）的 API 正确性天然需要一次不 mock 的真实调用（ADR 0013 反 mock 决策）。**最小覆盖**要求每个编排入口恰好对应一条小规模、确定性、耗时有界的冒烟用例；超出最小覆盖的遍历性网格与大振幅长弧验证不进入默认 pytest，归属 benchmark 或手工诊断脚本（ADR 0037）。

### 测试套件时间预算（Test Suite Time Budget）
- 单测试用例墙钟时间上限：**10 秒**
- 单测试文件墙钟时间上限：**60 秒**
超出预算的测试不得进入默认 pytest 集（ADR 0037）。

### 研究级容差 vs 筛选级容差（Research vs Screening Tolerance）
- **研究级容差**：动力学链路默认 `DEFAULT_TOLERANCE = 1e-12`（dynamics、force_model、differential_correction 三处同名常量），用于动力学基准积分、辛结构保真与高精度轨线精化。库内其他模块另有各自容差常量，不以本条为准。
- **筛选级容差**：`rtol/atol ≈ 1e-9 ~ 1e-10`，用于初猜网格搜索、多初值筛选与测试套件。测试套件不得对全网格使用研究级容差（ADR 0021 修订 #420、issue #536）。

## 架构分层

### 五层架构（Five-Layer Architecture）
源码由底至顶严格单向依赖（ADR 0011、ADR 0012）：
1. `data`：常数、历表、坐标系与数据模型
2. `numerical`：底层数值方法与积分器核心，以 Rust crate 落地（`crates/`），Python 侧入口为 `e2m2e.integrators`
3. `algorithm`：动力学与力模型编排、轨道族、微分修正、转移等算法
4. `api`：Facade 门面与 MCP 协议接口
5. `tools`：日志、格式化与可视化辅助

`mbse` 为独立顶层子系统（系统工程模型、需求追溯与架构图），位于依赖链之外，仅被测试、脚本与文档消费。

### 门面（Facade）
接口层的任务级入口类：五个任务级能力。与轨道库类、spatiography 类共同构成暴露的接口类集合（ADR 0014，经 ADR 0043 分家）。

## 语言约定

仓库拥有者在评审、issue 与 agent 会话中的固定用词，记录以保证沟通单一口径。

### 术语清单（Terminology list）
字段或过滤条件允许取值的封闭集合（分类学标签、记录侧族名、转移类型），随包版本冻结。
_避免_：词表、vocabulary

### 调用方（Caller）
使用本仓库接口的下游程序：GUI、MCP Agent、CLI。
_避免_：消费方、consumer

### 版本（术语清单的）
术语清单只随包发布变更的保证；调用方升级时刷新。
_避免_：契约、contract

### 接口（Interface）
两个部分相接、可独立变化的边界。
_避免_：缝、seam

### 调用链（Call chain）
调用方调起本仓库的路径：进程内 import、MCP、CLI、stdio sidecar 协议。
_避免_：通道、channel、消费、consume

## 轨道库数据模型

### 轨道记录（Orbit record）
一条原子目录记录只载一条轨迹：任务轨道或转移轨道。绝不打捆。
_避免_：族记录、捆绑记录

### 任务轨道（Mission orbit）
航天器驻留或受控沿行的轨迹：设计产物、族成员、受控维持产物。

### 转移轨道（Transfer orbit）
从一条轨道或状态到另一条轨道或状态的轨迹；以非空 transfer_type 判定。

### 族（标签而非容器）
轨道记录上的 orbit_family 族名标签 + 生成批次 family_id；族是可查询的记录分组。
_避免_：族记录、容器、打捆

### 分发包（Distribution bundle）
基线数据集的打包传输形态：GitHub Release 资产内打捆，调用方下载解压后
显式导入展开为轨道记录（ADR 0047）。不是轨道记录。
