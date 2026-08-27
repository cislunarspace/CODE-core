# ADR 0009: Enable spice feature for release wheels / release wheel 启用 spice feature

[English](#adr-0009-enable-spice-feature-for-release-wheels) | [简体中文](#中文)

## English

**Status**: Adopted (implemented)
**Date**: 2026-07-29
**Related**: ADR 0002 (Rust integrator core), issue #246

### Context

The `spice` feature gates Rust-side fast paths — third-body gravity, compiled
STM propagation, shooting, etc. (roughly half the Rust code). Until now,
release wheels uniformly shipped without spice: cspice-sys builds needed a
local CSPICE or `CSPICE_DIR`; CI/release never configured either, so enabling
it failed the build. Python-side code try/excepts missing spice bindings and
silently degrades to Python slow paths — functionally correct, but users lose
the fast paths without knowing. This ADR evaluates whether shipping wheels
with spice is feasible.

### Evaluation

#### Licensing

NAIF's CSPICE redistribution terms (Toolkit Redistribution clause on the
Toolkit page):

- Mirroring/forwarding the entire Toolkit: requires NAIF's prior written
  permission; do not do it unilaterally.
- **Shipping SPICE Toolkit library modules as part of a package supporting
  self-built SPICE tools: "entirely appropriate".** JPL Document Review
  issued Clearance CL05-2438 permitting SPICE products distribution via NAIF
  servers.

e2m2e wheels embedding statically linked CSPICE fall under the second case
(the wheel is a self-built SPICE tool; CSPICE is one of its components) — no
licensing obstacle. The obligation is attribution: add NOTICE in repo and
wheel metadata stating CSPICE comes from NASA/JPL NAIF.

#### Size

Measured on a real server with release builds
(`libe2m2e_integrators.so`): without spice 832 KB, with spice 2681 KB —
**about +1.8 MB per wheel**. Acceptable.

#### Build reliability

cspice-sys's `downloadcspice` downloads Toolkit sources from
naif.jpl.nasa.gov at build time and compiles them in place (reqwest download +
gzip/tar unpack; all work inside manylinux containers). CI has exercised this
mechanism successfully (PR #248). A pre-download + actions/cache + `CSPICE_DIR`
scheme for the release runner was considered but rejected: manylinux builds run
inside Docker where passing host cache dirs and `CSPICE_DIR` into the container
has no clean path — high complexity, little benefit. Release uses the same
downloadcspice mechanism as CI (which exercises it every run); NAIF's realtime
reachability is recorded as accepted risk, to be hardened into a caching scheme
only if it becomes a problem.

#### Platform matrix

cspice-sys compiles CSPICE from source via cc, depending on no prebuilt
libraries:

- manylinux_2_17 (glibc 2.17): source build, no glibc version issues.
- Windows MSVC: officially supported by cspice-sys (downloadcspice fetches the
  PC_Windows_VisualC_64bit package).
- macOS: not in the current release matrix; not evaluated.

### Consequences

Licensing, size, and platforms all clear — **release enables spice**
(implemented):

1. release.yml's maturin build adds `--features spice`, built via
   downloadcspice (same mechanism as CI).
2. NOTICE file added at repo root attributing CSPICE to NASA/JPL NAIF;
   packaged in sdist too.
3. Installation docs clarified: wheels carry Rust fast paths; source builds
   also default to spice (Cargo default feature, see ADR 0002's 2026-08
   revision).

### Revision (2026-08: build-mechanism change)

The downloadcspice mechanism described above (build-reliability section,
platform matrix, and consequence item 1) is deprecated:

- cspice-sys removed its downloadcspice feature: builds now fail outright when
  `CSPICE_DIR` is missing instead of downloading sources from naif.jpl.nasa.gov.
  CSPICE always comes from GitHub `cspice-v1` release prebuilt packages via
  `scripts/download_cspice.py`, pointed at by `CSPICE_DIR` (annotated in root
  Cargo.toml and release.yml).
- The aarch64 prebuilt package is built from NAIF sources on native arm64
  runners by `cspice-aarch64-build.yml` and published to `cspice-v1`.
- Spice is now a default feature (`crates/*/Cargo.toml default=["spice"]`,
  declared again in pyproject `[tool.maturin]`); release.yml's maturin args no
  longer need explicit `--features spice`.
- Licensing conclusions, NOTICE attribution (item 2), and the enable-spice-for-
  releases decision are unchanged.

### Revision (2026-08-12, ADR 0020 decision 4)

After releases began shipping spice, the Python-side try/except silent
degradation for missing spice bindings was removed: without spice (environment
not set up) the library raises (issue #378) instead of silently falling back to
slow paths. Corresponding tests' `importorskip` semantics were adjusted
likewise.

## 中文

**状态**：已采纳（已实施）
**日期**：2026-07-29
**关联**：ADR 0002（Rust 积分器内核）、issue #246

### 背景

`spice` feature 控制 Rust 侧第三体引力、compiled STM 传播、打靶等快速路径
（约占 Rust 代码一半）。此前 release wheel 一律不带 spice：cspice-sys 构建
需要本机 CSPICE 或 `CSPICE_DIR`，CI/release 从未配置，开了就构建失败。
Python 侧对缺失的 spice 绑定全部 try/except 静默降级到 Python 慢路径，功能
正确但用户无感知地失去快速路径。本次评估发布带 spice 的 wheel 是否可行。

### 评估

### 许可

NAIF 对 CSPICE 的再分发条款（Toolkit 页面的 Toolkit Redistribution 条款）：

- 简单镜像转发整个 Toolkit：需 NAIF 事先书面许可，禁止擅自做。
- **把 SPICE Toolkit 库模块作为支持自建 SPICE 工具的软件包的一部分发布：
  完全合适（entirely appropriate）。** JPL Document Review 已发
  Clearance CL05-2438 准许 SPICE 产品经 NAIF 服务器分发。

e2m2e wheel 内嵌静态链接的 CSPICE 属于第二种情形（wheel 是自建 SPICE
工具，CSPICE 是其组成部分），许可上无障碍。义务是署名：在仓库与 wheel
元数据中加 NOTICE，注明 CSPICE 来自 NASA/JPL NAIF。

### 体积

服务器实测 release 构建（`libe2m2e_integrators.so`）：不带 spice 832 KB，
带 spice 2681 KB，**增量约 1.8 MB / wheel**。可接受。

### 构建可靠性

`cspice-sys` 的 `downloadcspice` 在构建时从 naif.jpl.nasa.gov 下载 Toolkit
源码并就地编译（reqwest 下载 + gzip/tar 解包，manylinux 容器内均可运行），
CI 已用该机制跑通（PR #248）。曾考虑 release runner 预下载 + actions/cache
+ `CSPICE_DIR` 的方案，但 manylinux 构建在 Docker 容器内进行，宿主缓存目录
与 `CSPICE_DIR` 环境变量传入容器没有干净的路径，复杂度高而收益小。
release 采用与 CI 相同的 downloadcspice 机制（每次 CI 都在验证它）；NAIF
实时可达性作为已接受的风险记录在案，若日后成为问题再加固为缓存方案。

### 平台矩阵

cspice-sys 用 cc 从源码编译 CSPICE，不依赖预编译库：

- manylinux_2_17（glibc 2.17）：源码编译，无 glibc 版本问题。
- Windows MSVC：cspice-sys 官方支持路径（downloadcspice 拉的就是
  PC_Windows_VisualC_64bit 包）。
- macOS：当前 release 矩阵不含，暂不评估。

### 结果

许可、体积、平台均无障碍，**release 启用 spice**（已实施）：

1. release.yml 的 maturin build 加 `--features spice`，经 downloadcspice
   构建（与 CI 同机制）。
2. 仓库根加 NOTICE 文件，注明 CSPICE 归属 NASA/JPL NAIF；sdist 一并打包。
3. 安装文档补充说明：wheel 自带 Rust 快速路径；源码构建也默认带 spice（Cargo default feature，见 ADR 0002 2026-08 修订）。

### 修订（2026-08，构建机制变更）

上文构建可靠性、平台矩阵两节及结果第 1 条描述的 downloadcspice 机制已废弃：

- cspice-sys 已去除 downloadcspice feature：缺 `CSPICE_DIR` 时构建直接报错，
  不再从 naif.jpl.nasa.gov 下载源码就地编译。CSPICE 一律经
  `scripts/download_cspice.py` 取 GitHub `cspice-v1` release 预编译包，
  由 `CSPICE_DIR` 指向（根 Cargo.toml 与 release.yml 均有注释说明）。
- aarch64 预编译包由 `cspice-aarch64-build.yml` 在原生 arm64 runner 上从
  NAIF 源码编译并发布到 `cspice-v1`。
- spice 已升为默认 feature（`crates/*/Cargo.toml default=["spice"]`，
  pyproject `[tool.maturin]` 再声明一次），release.yml 的 maturin args
  无需再显式传 `--features spice`。
- 许可结论、NOTICE 署名（第 2 条）与 release 启用 spice 的决策不变。

### 修订（2026-08-12，ADR 0020 决策 4）

release 已带 spice 后，Python 侧对缺失 spice 绑定的 try/except 静默降级机制已删除：
无 spice（环境没搭好）即报错（issue #378），不再静默回退慢路径。对应测试的
`importorskip` 语义同步调整。
