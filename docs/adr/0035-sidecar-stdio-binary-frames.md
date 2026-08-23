# ADR 0035：GUI sidecar stdio 协议：共享 Facade 信封，大数组走二进制帧

**状态**：已采纳
**日期**：2026-08-22
**关联**：ADR 0014（接口层 Facade/MCP/CLI）、issue #518

## 背景

tod（transfer-orbit-design）正把 UI 迁移到 Tauri 架构（Rust 壳 + Web 前端）。e2m2e 是 Python 库，Rust 无法进程内调用，需要以常驻子进程（sidecar）+ stdio 消息协议的方式使用 Facade。

量测（tod 仓库 `tools/bench_serialization.py`）表明：全量 halo 族（26882 成员，32M 浮点）JSON 编码需 17.9 s、体积 687 MB，比计算本身还慢；而原始 f64 二进制约 98 ms / 258 MB。结论：控制消息可以走 JSON 文本行，大数组必须走二进制帧。

## 决策

新增 `e2m2e serve-stdio` CLI 子命令（与 `mcp-serve` 同级），作为 GUI sidecar 入口。协议分两层：

### 1. 控制层：JSON 行，复用统一信封

请求、响应、进度事件都是一行一个 JSON 对象。响应复用 ADR 0014 §4 的统一信封 `{status, data, error, meta}`，信封定义在 `e2m2e/api/mcp/envelope.py`，单一来源，MCP 与 sidecar 两个传输层共享，协议演进只维护一处。工具面 = Facade 上 `mcp_exposed=True` 的方法（纯派生，不新增业务逻辑）。

### 2. 进度行

不复用 MCP notification（其语义服务于 LLM 客户端）。GUI 侧只需要任务未完成期间的可丢弃进度行，消费端是 Rust 壳的一个 match：

```json
{"status": "progress", "data": null, "error": null, "meta": {"job_id": "...", "percent": 0.42, "message": "..."}}
```

`percent` 为 0–1 浮点；消费端对 `status` 未知的行必须跳过不报错。

### 3. 二进制帧：跨仓库持久契约

JSON 行末尾带 `"binary_frames": N` 字段（N 缺省为 0）时，该行换行符之后紧跟 N 个二进制帧顺序排列，帧后恢复 JSON 行流。**帧格式是 tod ↔ e2m2e 的跨仓库持久契约**，逐字段定义如下（所有多字节整数小端）：

```
偏移  长度  类型   含义
0     4    u32   magic = 0x324D3245（ASCII "E2M2" 的小端 u32；字节序 45 32 4D 32）
4     1    u8    dtype：0 = float32，1 = float64
5     1    u8    ndim：数组维度数，≥ 1
6     4·ndim  u32[]  shape：各维元素数（不是字节数），小端
6+4·ndim  —   —    原始数组字节：C 连续（行主序）、小端，长度 = prod(shape) · 元素宽度
```

- f32 与 f64 都支持，由请求方在请求参数中声明 dtype。tod 画布渲染用 f32（Three.js `BufferAttribute` 本就是 f32，f64 进浏览器也得截断）；计算中间量、要落盘复算的量保持 f64。
- 没有 version 字段：magic 兼职版本锚点，不兼容改动时换 magic，新旧 magic 共存即共存两个协议版本。
- 不搞多数组打包头：N 个独立帧顺序排，数量由 JSON 行的 `binary_frames` 声明。
- JSON 行中对应字段（如状态的展平列表）置为 `null` 占位或直接省略，二进制帧是唯一真身；具体哪个字段对应第几个帧，属于各工具的响应模型约定，随模型演进。

## 理由

1. **信封单一来源**：`envelope.py` 现实现为普通 dict + 纯函数，无 MCP 专属耦合，跨传输层复用零改动。
2. **帧格式一次定死**：跨语言 ABI 定错难改，所以字段能少则少：magic、dtype、shape、字节，共 4 项。复杂布局（打包头、version、对齐填充）全部不要。
3. **进度行走信封**：避免为 GUI 引入第二套事件机制；`status` 是既有字段，扩展取值即可。

## 结果

- `e2m2e serve-stdio` 子命令 + sidecar 协议模块（帧编解码独立于传输，可单测）。
- tod Rust 壳按本 ADR §3 实现对端解码；帧格式变更必须走新 ADR。
