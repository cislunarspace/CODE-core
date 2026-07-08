# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [5.1.0] - 2026-07-08

### Added
- Hamiltonian 正规化流水线 `e2m2e.algorithms.normal_form`（#168–#176）：把 CR3BP 平动点附近的非线性动力学逐层化简为少数表征参数
  - `NormalFormPipeline` 一键式流水线：动力学替代 → quasi-Floquet 变换 → 中心流形化简 → rho↔param 坐标变换
  - `NormalFormContext`/`NormalFormResult`、`DynamicalSubstituteCorrector`、`QuasiFloquetReducer`、`CenterManifoldReducer`、`LibrationCatalogTransformer`
  - NAFF 频率分析（不可用时降级 FFT）、块三对角多重打靶、函数式坐标变换链 `coord_trans`
  - `[normal-form]` 可选依赖（sympy、joblib），惰性导入不阻塞基础包加载
  - 文档 `docs/algorithms/normal-form.rst`、示例 `examples/normal_form_example.py`
  - `CONTEXT.md` 与 `docs/reference/glossary.rst` 新增「Hamiltonian 正规化」术语小节

### Fixed
- 修正 quasi-Floquet Lie 代数法的 dexp 公式误差：原实现假设 `d/dt exp(ξ)=exp(ξ)·ξ̇`（仅 [ξ,ξ̇]=0 成立），改为 commutator-free 4 阶 Lie group RK4
- 修正中心流形 Step 2（center）的 W 归零 bug：MAD 离群抑制在常系数输入下把 W 缩到 0，且 `reduce()` 取实部丢弃了纯虚 W
- 修正 rho↔param 坐标变换的插值时间网格缺陷：原用 `dt=0.1` 兜底网格，改为取 `qf_result.tlist` 的真实采样点

## [5.0.0] - 2026-06-19

### Added
- 力模型体系：球谐重力场（J2）、指数大气阻力、太阳光压与圆锥阴影、脉冲/有限推力、相对论修正
- 潮汐模型：固体潮 Step1/Step2 修正、极潮（含 Desai 海洋极潮）、永久潮汐 tide-free/zero-tide 约定、`GravityField` 潮汐集成
- 积分器族：Rust 工作空间 + maturin 绑定，新增 PD45、RK89、PD78、Cowell 8 阶、ABM 4 阶步进器
- 坐标系：ITRF 框架、`standard_icrf()` 工厂、ICRF↔ITRF 端到端测试、IAU 2006 简化模型 pyerfa 黄金参考对比
- 转移轨道：推进模型与终端条件抽象、SciPy/COPT adapter 化、二层/标准多重打靶固定步长同伦星历转换
- System 抽象：`CR3BP_System` 与 `EphemerisSystem` 实现 `System` ABC，`UnitSystem` 与 `ReferenceFrame.J2000`
- 数据模型：`BoundaryMode`、`TwoLevelMultipleShootingStatus`、`OrbitFamilyType`（15 个轨道族）
- 动态坐标轴 VNB/LVLH 与 `FiniteBurn` 方向帧
- Sphinx 文档补齐：力模型、积分器、大气、SRP、策略、转移工作流
- 真实 SPICE 小样例验证、月影真实几何测试、LEO 潮汐端到端自洽验证、GMAT LEO 参考轨道对比工具链

### Changed
- **BREAKING**: `Dynamics` 签名改为接收 `system: System`
- **BREAKING**: `ForceModel` 改为配置驱动，容器按名管理；命名与移除接口对齐 #69 关闭共识
- **BREAKING**: 星历修正从字符串分发改为 `PatchPointCorrector` 接缝与注册表
- **BREAKING**: 统一枚举定义中心到 `mbse/data/enums.py`
- **BREAKING**: 二层多重打靶接口与收敛语义重构，Level 2 修正接口与回溯逻辑简化
- `MultipleShootingResult` 升级为 frozen dataclass
- 回退 `CoordinateSystem` 冻结（#76 决策反转）
- `CONTEXT.md` 重写术语表，ADR 全量译为中文，传播/坐标术语对齐
- CLAUDE.md 增补写作要求段
- CI 依赖升级

### Fixed
- 修正 `ITRFApproxAxes` GAST 旋转符号
- 延迟加载 `spiceypy`，避免顶层导入强制加载
- 修正 Halo PAL 延拓折叠振荡与南 Halo 分支方向
- 清理 `Orbit` 派生属性边界
- Sphinx 构建零警告收口（#123-#129）
- 同伦动力学权重、失败路径与残差可观测性补强
- 统一结果类型验证与收尾

### Removed
- **BREAKING**: 删除 `from_known_system` 及迁移调用点
- **BREAKING**: 移除 `System.transform()` 快捷方式
- 删除冗余 `test_basic.py`
- 删除浅层 MBSE 元数据测试
- 移除 `AGENTS.md` 与 `CONTRIBUTING.md`

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
