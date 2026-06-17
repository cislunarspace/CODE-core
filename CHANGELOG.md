# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [4.2.1] - 2026-05-25

### Fixed
- 修正 `e2m2e/core/orbit.py` 导入块排序（ruff I001）

## [4.2.0] - 2026-05-24

### Added
- 新增两级多重打靶求解器
- 新增同伦星历修正分发器
- 新增 `PlotConfig.from_env`，支持从环境变量读取绘图配置
- 可视化模块支持 3D 天体 Billboard PNG 图标
- `generate_halo_family` 增加 `z_range` 与全局 `n_orbits` 上限
- 转移搜索记录首次可行解耗时

### Changed
- 微分修正迭代增加 `callback` 参数，支持实时迭代监控
- 稳定性分析补充缺失的动力学回退
- 简化 CI 测试矩阵为 Python 3.13 on Ubuntu / Windows
- 测试覆盖率阈值从 60% 降至 55%

### Fixed
- 修正 `Orbit` 类中未使用代码
- 修正坐标、稳定性、GEO、配置、SPICE 等模块分支覆盖
- 修正 CI lint 与 typecheck 失败

## [4.1.0] - 2026-04-24

### Added
- 新增 `CR3BP_SRP_Dynamics`，支持圆型限制性三体问题下的太阳光压动力学
- 新增 Sphinx 文档站，替换原有 Docusaurus 站点
- 新增 core、ephemeris、transfer 用户文档

### Changed
- README 重写，提升准确性与简洁性
- 文档依赖与 CI 流程适配 Sphinx 迁移
- 移除过期计划文件

### Fixed
- 修正 SRP 动力学中基类返回值被原地修改的问题

## [4.0.0] - 2026-04-14

### Added
- MBSE：带 `@runtime_checkable` 的 Protocol 接口（`SystemModel`、`EOMProvider`、`Propagator`、`OrbitContainer`、`CorrectorStrategy`、`Optimizer`、`Visualizer`）
- Pydantic 数据模型：`PropagationResult`、`OrbitProperties`、`OrbitStability`、`JacobiResult`
- 基于策略的微分修正：`halo_fixed_z0`、`halo_fixed_x0`、`symmetric_2d_fixed_x0` 等
- 需求注册表与 Mermaid 图表自动生成
- `MultipleShooting` 类，支持多进程并行传播
- `GeoTransferSearch`：DRO→GEO 并行网格搜索
- `BodyName` 常量类，含太阳系 GM 数据
- `HomotopyEphemerisDynamics`：同伦星历动力学

### Changed
- 重构星历层，改进数值稳健性
- 将单体微分修正拆解为策略模式
- `convert_to_j2000` 参数由 `tu_seconds` 改为 `tu_days`
- 可视化重构，统一用 `PlotConfig` 配置

### Fixed
- `EphemerisDynamics` 方法签名与 `Dynamics` 基类对齐
- 修正 SPICE `bodvrd` 返回值索引与 Pylance 类型错误
- Jacobi 常数族绘图排序，消除错位伪影
- 修复 core/algorithms 22 项、可视化 32 项测试

## [3.2.0] - 2025-12-01

### Added
- 星历动力学（`EphemerisDynamics`），支持 SPICE 内核
- `EphemerisSystem`：多体引力建模
- SPICE 内核管理工具
- 同伦动力学：跨解分支延拓

### Changed
- 文档改为任务导向风格（中英）

### Fixed
- 修复文档中 90 处失效锚点警告

## [0.1.0] - 2025-06-21

### Added
- CR3BP 系统建模（`CR3BP_System`、`CR3BP_Dynamics`）
- 轨道数据结构，含状态向量与周期记录
- 2D/3D 对称轨道微分修正
- 自然延拓与伪弧长延拓
- 稳定性分析（Floquet 乘子）
- 转移轨迹设计：网格搜索 + NLP 优化
- 2D/3D 轨道可视化
