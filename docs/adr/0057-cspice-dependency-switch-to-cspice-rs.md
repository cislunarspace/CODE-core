# ADR 0057：cspice 依赖从本地 vendor 切换至 cislunarspace/cspice-rs

- 状态：已接受
- 日期：2026-10-24
- 关联 issue：#717（上游化尝试，已关闭）、#756（本切换）
- 前置：ADR 0051（vendor 的由来与纯 Rust 后端）

## 背景

`cspice` 0.1.0 / `cspice-sys` 1.0.4（jacob-pro/cspice-rs）在 aarch64 上编译失败（E0606），且 LP64 平台三处硬编码 `i32` 导致 Linux/macOS 从未可编译。上游失修约三年，#717 提交的 PR #12/#13 无维护者响应，遂建立组织内独立维护仓库 [cislunarspace/cspice-rs](https://github.com/cislunarspace/cspice-rs)（决策记录见该仓库 ADR 0001-0003）：

- crate 更名 `cspice-rs` / `cspice-rs-sys`（crates.io 原名被上游占用），版本线 0.1.0 重新起步；
- 基线含 PR #12/#13 等价内容 + 4 处 LP64 修复；
- CI 覆盖三平台测试与 aarch64-linux 交叉检查（run 36401328560 全绿）；
- LGPL-3.0 与原作者署名链在该仓库 README/LICENSE 保留。

## 决策

CODE-core 删除 `[patch.crates-io]` vendor（`crates/cspice/`），改以 git 依赖（tag `v0.1.0`）使用独立仓库：

- workspace 依赖名 `cspice`/`cspice-sys` → `cspice-rs`/`cspice-rs-sys`，源码 `use` 路径随之 `cspice_rs`/`cspice_rs_sys`；`spice_ffi.rs` 继续直调 sys 层，与 `cspice::ffi` 的去重留待后续 issue。
- `CSPICE_DIR` 契约不变（scripts/download_cspice.py + cspice-v1 release），不启用 downloadcspice。
- 后续 crates.io 发布 0.1.0 完成浸泡验证后，可再评估切 registry 版本号依赖（届时另开 issue，不阻塞本决策）。

## 后果

- 正面：vendor 目录（约 5 千行第三方代码）出库，LP64/aarch64 修复由上游（本组织仓库）统一维护，三平台 CI 门禁不再依赖本仓库。
- 负面：git 依赖在 crates.io 发布前是唯一渠道（离线构建需 `cargo vendor` 或缓存）；依赖名变更波及三个 crate 的 feature 图与源码 use 路径。
- 风险控制：tag `v0.1.0` 固定版本；独立仓库分支保护与 CI 已就绪，后续升级走显式 tag 变更。

## 修订记录

- 2026-10-24：首次记录。
