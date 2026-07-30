# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- 二体 Lambert 求解器（`e2m2e.transfer.lambert`）：Izzo 算法 Rust 内核（`e2m2e-propagation` crate），`solve_lambert`/`solve_lambert_batch` 支持短程/长程与多圈解，基准对齐 Vallado 文献值
- porkchop 扫描（`e2m2e.transfer.porkchop`）：出发时间 × 飞行时间网格的双脉冲 ΔV 扫描，终端状态经 `TerminalCondition` 接口提取
- 不变流形与庞加莱截面（`e2m2e.algorithms.manifolds`/`sections`）：`InvariantManifold` 种子生成与批量传播；`PoincareSection` 平面/近拱点截面，事后检测 + Brent 插值求精
- 三体打靶与低能转移（`e2m2e.transfer.three_body_lambert`/`low_energy`）：`ThreeBodyLambert` 以二体解为初猜在 CR3BP 下 Newton 打靶；`patch_manifolds` 截面拼接、`design_low_energy_transfer` 低能转移流水线
- 事件检测：`Dynamics.propagate(events=...)` 透传 scipy 事件语义（terminal/direction），`PoincareSection.event(direction, terminal)` 生成截面穿越事件函数；Rust `solve_ivp_events_py`（薄封装 `e2m2e.integrators.solve_ivp_events`）在积分内循环完成事件检测与二分求精，`ForceModel.propagate` 事件走此快速路径
- BCR4BP 双圆限制性四体模型（`e2m2e.core.bcr4bp_system`/`bcr4bp_dynamics`）：`BCR4BPSystem` 在 CR3BP 会合系上叠加太阳解析圆轨道摄动（m_s=328900.56、a_s=389.17、ω_s=−0.9252，DE440/GMAT 来源），`BCR4BP_Dynamics` 含太阳直接/间接项与 STM；无 Jacobi 积分，与星历 1 天外推误差约 1e3 km（主误差来自月球圆轨道近似）
- 多脉冲转移与主矢量检验（`e2m2e.transfer.multi_impulse`）：`MultiImpulseTransfer` 以中途节点 `[t_i, r_i]` 为决策变量、弧段 Lambert 封闭、scipy SLSQP 最小化总 ΔV；`check_primer_vector` 实现 Lawden 必要条件检验与 Lion & Handelsman 中途脉冲插入准则

## [5.3.1] - 2026-07-29

### Added
- `e2m2e.algorithms` 导出 `sample_patch_points_perilune_clustered`（multiple_shooting.py:551），与 `sample_patch_points` 并列，面向 NRHO 近月点欠约束场景；此前需从 `e2m2e.algorithms.multiple_shooting` 模块路径导入

### Fixed
- `ForceModel` 类 docstring 仍自述「不支持 STM」，与 `propagate(with_stm=True)` 现状矛盾；改为以代码实际为准（支持 STM，不支持 Jacobi）

### Changed
- 发版前 GitHub Pages 文档全面审查与更新（PR #250）：50+ 处问题修复
  - 硬错误：quickstart 力模型组合三连错、stability 整节旧 API 重写、TransferConfig 9 个旧字段改 `nlp_*`、terminal/propulsion 虚构签名、visualization 两个不存在的绘图方法、orbit 的 `save/load`、dynamics/ephemeris 的 dict 属性访问、coordinate transform 参数序与 `UnitSystem.CR3BP`、halo 族编排四文件签名、两层打靶参数/字段名
  - 过时描述：Orbit 显式字段、System 基类三问题、halo_class 南北族（3 文件写反）、EphemerisDynamics 内部实现地位、ITRFSpiceAxes 标注、ReferenceFrame 枚举成员、内核下载指向 GitHub Release（kernels-v1）等
  - 新功能文档：ForceModel STM 与 propagate_compiled 快速路径、`rk_step` `state_error_dim`、SPICEManager 星历缓存、STM 截断抛错契约、`NormalFormResult.save/load`、近月点加密采样；API 页补 `indirect_term`/`relativistic_correction`/`ephemeris_correction`/`normal_form`/`two_level_multiple_shooting` 等模块
  - 审查中追加：halo 种子示例 `amplitude_z` 由 0.01 改 0.001（Richardson 近似在 0.01 不收敛、与 `examples/halo_orbit_design.py` 一致）；stability 分岔判据措辞由「穿过」改为「接近」（与代码 `abs(lam±1.0) < tol` 一致）
- 源码 docstring RST 排版修复以支撑 Sphinx 构建零警告：normal_form 11 个模块、ephemeris_correction、physical_model、visualization/config、force_model 等 docstring 解析干净；`api/e2m2e.mbse.rst` 与 `api/e2m2e.visualization.rst` 用 `:exclude-members:` 消解 ProjectionPlane/PlotConfig 的重复对象描述与交叉引用歧义

## [5.3.0] - 2026-07-10

### Added
- rho↔ECI 坐标桥接（#193）：`rho_to_eci`/`eci_to_rho` 把 qiao 的 rho 无量纲状态接入 `ForceModel`（ECI 积分），往返误差 < 1e-9
- `NormalFormResult.save/load`（#192）：npz 序列化，含 W_series 三层嵌套复值 dict，正规化结果可预计算反复用
- `ForceModel.propagate` 开放积分器选择（#191）：`method` 参数支持 PD45/PD78/RK89，PD78 与 qiao Rust / GMAT 逐字一致

### Fixed
- **rho↔ECI 速度变换 bug**：`v_eci` 公式末项误旋转已是 J2000 速度的 `v_LP`（多乘 `C`），致 ECI 积分 72h 轨迹发散 120,065 km。修正后 python-e2m2e 星历链 vs qiao Rust 72h 差异 0.035 km
- `rho_bridge._jd_to_et` 时间格式：`spice.str2et(f"JD {jd}")` 把 TDB 当 UTC，差 ~68 秒；改用 `f"{jd:.20f} JDTDB"`
- `SynodicAxes` 数值微分步长 1e-5s→1.0s：原步长严重吃有效数字，Cdot 相对误差 1.36e-3；1.0s 处于舍入-截断平衡，误差降到 ~1e-7

### Changed
- **坐标系族提为 `core/coordinate/` 子目录**（#197）：12 文件从 core 根层收入子包，与 tests/core/coordinate/ 对齐；`sys.modules` 别名保持 `from e2m2e.core.axes import` 等旧路径零破坏
- **ForceModel 移除对 Dynamics 的假继承**（#140）：原继承只为复用几个数据属性却全部重写 propagate、对 STM/Jacobi 抛 NotImplementedError（LSP 违反）；改为独立类自持属性
- 折叠单实现 ABC（#196/#203）：删 PropulsionModel、AtmosphereModel、ShadowModel（各仅一个实现，死灵活性）；XysProvider 按 ADR 0003 保留（GMATXysProvider 是真实需求）
- `rho_bridge` 用 Protocol 消除 core→algorithms 反向依赖（#198）
- `MultipleShooting` 三并行后端的段收集逻辑统一（#143），消除 ~30 行重复
- `DifferentialCorrection` 清理死代码（#141）：`_compute_error_vector` + 3 个从不读取的实例属性
- `NormalFormResult` 删两个从不填充的幽灵字段（#201）；`_wrap_orbit` 移除过宽 `except Exception`（#204）
- `propagate` 的 `_prepare_t_eval`/`_estimate_initial_step` 标为 staticmethod（#147）
- 删废弃 `SearchConfig` 别名与死常量（#202）；修正示例失效导入（#199）；`.gitignore` 修 `*.pyd`/`scripts/*.png` 粘连 bug（#200）

### Removed
- `EphemerisDynamics` 从 `e2m2e.core` 公开导出移除（降级为内部实现，子模块路径 `e2m2e.core.ephemeris_dynamics` 仍可用）
- `PropulsionModel`/`AtmosphereModel`/`ShadowModel` ABC、`SearchConfig` 别名（见 Changed）

## [5.2.0] - 2026-07-09

### Added
- 第三体引力 `ThirdBodyGravity`（#181–#183）：力分解路径补上缺失的第三体点质量引力，使 `ForceModel` 能外推 cislunar 轨道。每个摄动天体一个实例，含直接项与间接项，与 `EphemerisDynamics` 的解析 N 体公式物理等价（自洽性测试 < 1 mm）
- 月球非球形引力（#185–#189）：`GravityField` 改造为天体无关，支持任意天体的球谐引力场
  - 按 `body` 自动切换 body-fixed 轴：地球 ITRF93、月球 MOON_PA（DE421 principal axes）
  - COF 格式引力场文件解析（移植 GMAT `LM_LoadCof`），引入月球 GRGM900C（360×360，GRAIL）
  - 固体潮 Step1 重构为天体无关（扰动体列表 + Love number），地球 Step2/极潮保留专用；月球固体潮（k₂=0.024116）
  - 引入完整 EGM96（360×360），替换此前只有 n≤2 的截断文件，地球引力场真正支持 10×10
  - DFH "非球形-大天体耦合项"即固体潮，`tide_mode="solid"` 启用
- `PointMassGravity` 配置序列化（补历史欠账，#183）
- SPICE 内核：地球 ITRF93（`earth_latest_high_prec.bpc` + `SPICEEarthPredictedKernel.bpc`）、月球 MOON_PA（`SPICELunaCurrentKernel.bpc` + `SPICELunaFrameKernel.tf`）

### Changed
- 地球 body-fixed 从低精度 `ITRFApproxAxes`（岁差+GAST）升级到 SPICE ITRF93（IAU 2000 CIO 方法）
- `solid_tide_step1` 签名重构为接收扰动体列表 + Love number 表（天体无关），地球潮汐回归逐字一致（误差 < 1e-19）

### Fixed
- COF 解析器注释跳过：GMAT EGM96.cof 用 `CCCCC` 边框注释，原逻辑只跳 `COMMENT`/`C ` 开头

### 验证
- e2m2e vs DFH 满配（无 SRP）力模型对齐：Halo 7 天 88 米、Lissajous 7 天 201 米
- SRP 诊断结论：残余差异 97% 来自 ECOM vs cannonball 光压模型差异，力模型基线已对齐到亚百米级

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
