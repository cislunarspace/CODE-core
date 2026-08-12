# ADR 0009：release wheel 启用 spice feature

**状态**：已采纳（已实施）
**日期**：2026-07-29
**关联**：ADR 0002（Rust 积分器内核）、issue #246

## 背景

`spice` feature 控制 Rust 侧第三体引力、compiled STM 传播、打靶等快速路径
（约占 Rust 代码一半）。此前 release wheel 一律不带 spice：cspice-sys 构建
需要本机 CSPICE 或 `CSPICE_DIR`，CI/release 从未配置，开了就构建失败。
Python 侧对缺失的 spice 绑定全部 try/except 静默降级到 Python 慢路径，功能
正确但用户无感知地失去快速路径。本次评估发布带 spice 的 wheel 是否可行。

## 评估

### 许可

NAIF 对 CSPICE 的再分发条款（Toolkit 页面 "Toolkit Redistribution"）：

- 简单镜像转发整个 Toolkit：需 NAIF 事先书面许可，禁止擅自做。
- **把 SPICE Toolkit 库模块作为支持自建 SPICE 工具的软件包的一部分发布：
  完全合适（"entirely appropriate"）。** JPL Document Review 已发
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

## 结果

许可、体积、平台均无障碍，**release 启用 spice**（已实施）：

1. release.yml 的 maturin build 加 `--features spice`，经 downloadcspice
   构建（与 CI 同机制）。
2. 仓库根加 NOTICE 文件，注明 CSPICE 归属 NASA/JPL NAIF；sdist 一并打包。
3. 安装文档补充说明：wheel 自带 Rust 快速路径；源码构建也默认带 spice（Cargo default feature，见 ADR 0002 2026-08 修订）。

## 修订（2026-08-12，ADR 0020 决策 4）

release 已带 spice 后，Python 侧对缺失 spice 绑定的 try/except 静默降级机制已删除：
无 spice（环境没搭好）即报错（issue #378），不再静默回退慢路径。对应测试的
`importorskip` 语义同步调整。
