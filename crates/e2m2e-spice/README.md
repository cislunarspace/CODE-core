# e2m2e-spice

SPICE 能力 crate，维护 native 与 cspice vendor 两条栈的分工。e2m2e 的星历/
帧/时间查询生产路径是纯 Rust native 栈，CSPICE FFI 只作对拍 oracle 与兜底校验。

## 双栈分工

| 栈 | 模块 | 角色 |
|---|---|---|
| native（生产路径，纯 Rust，零 CSPICE FFI） | `native_spk`（ADR 0051，DAF/SPK Type 2 星历读取与求值，无条件编译） | 星历状态查询的运行时后端 |
| native（生产路径，纯 Rust） | `native_frame` / `native_time`（ADR 0056，BPC 二进制与 FK/文本 PCK 帧旋转、LSK 时间系统） | 帧旋转与时间转换的运行时后端 |
| cspice vendor（对拍 oracle） | `spice_ffi.rs`，经 [cislunarspace/cspice-rs](https://github.com/cislunarspace/cspice-rs) git 依赖（ADR 0057） | `#[doc(hidden)] ffi_oracle` 供 native 栈对拍测试；`spice_ffi::spkezr` / `pxform` 生产路径内部直调 native 后端 |

`ephem_cache` 在积分前把天体状态与帧旋转矩阵在均匀网格上预采样、建三次样条
装入进程级缓存，此后力模型每步查表、不再跨 FFI；`spk_accel` 在 SPK 链上
求值第三体/间接项加速度（含解析雅可比）。

## furnish_kernel 双注册语义

[`furnish_kernel`](src/lib.rs) 按文件头识别并双登记：前 8 字节 `DAF/` 的二进制
内核进 `native_spk` 注册表；`KPL/FK` / `KPL/PCK` / `KPL/LSK` 文本内核分派到
`native_frame` / `native_time` 文本池（ADR 0056）；随后一律 furnsh 到 CSPICE
内核池。native 识别不了的文本不进 native 池，但 CSPICE 侧照常 furnsh 校验
兜底；任一真实错误上抛并回滚 native 侧登记，不静默丢内核。

## feature 门控

- `native_spk` 无条件编译（生产星历路径不依赖 CSPICE）。
- `native_frame` / `native_time` / `spice_ffi` / `spk_accel` / `ephem_cache`
  门控在 `spice` feature（默认开启）。

构建 CSPICE 来自 GitHub Release `cspice-v1` 预编译包（`CSPICE_DIR`，由
`scripts/download_cspice.py` 提供），不启用 cspice-rs 的官网下载路径（ADR 0057）。
