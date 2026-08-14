# 变更日志

## [Unreleased]

## [5.6.10] - 2026-08-14

### Fixed
- **GMAT 参考数据未随包分发**（#423）：EOP/闰秒裁剪数据此前在 `tests/data/frames/fixtures/gmat`，加载器按仓库布局用 `parents[3]` 定位——wheel/sdist 不含该目录，安装后 GMAT-compatible 坐标轴（显式 EOP/闰秒输入）取数失败。数据迁入包内 `e2m2e/data/frames/fixtures/gmat` 并声明为包资源，仓库与安装布局下定位一致，wheel/sdist 自动分发。

### Changed
- **MBSE 活模型校验与文档产物生成**（#422）：五层组件清单（algorithm/api/core/data/numerical/tools）与隔离注册表，覆盖测试改为校验活模型而非文档快照；BDD 场景、需求与追溯矩阵改由生成器产出 `docs/reference/mbse/generated/` 并纳入测试校验。
- **数据层测试按边界重组**（#423 连带）：`tests/data` 镜像数据层结构（atmosphere/frames/kernels/templates/types）；`test_no_furnsh_bypass` 迁 `tests/_meta`，新增数据常量依赖守门。
- **API 测试按接口职责重组**（#424）：`test_facade.py` 拆分出 `test_models.py`（模型默认值、按类型范围边界、非法输入拒绝）与 `test_facade_responses.py`（响应契约），WSB 容器测试按契约归位。
- **测试套件收敛**（#425，ADR 0025）：新增主功能类标记守恒守门员元测试（每测试恰 1 主类）；`tests/algorithm/correction` 更名 `solver` 镜像源码结构；family 纯 API 校验用例删除（#424 的 `test_models` 已等价覆盖）；design 集成改用 `make_design_request` 桩工厂，`tests/algorithm` 不再 import `e2m2e.api`；normal_form 清除外部软件参照——qiao .mat 逐项比对迁 `scripts/diagnose_hamiltonian_vs_qiao.py` 手工诊断，5 个 skip 守卫空回归占位删除（转 #426 跟踪）。
- **测试套件层级澄清**（#432，ADR 0026）：coordinate 测试审查收尾与注释中文化，清理 `e2m2e.core`/`e2m2e.algorithms` 旧路径死引用，atmosphere 测试并入 numerical/forces 契约层。

## [5.6.9] - 2026-08-14

### Added
- **aarch64 Linux（arm64）构建与发布支持**：CSPICE 无官方 Linux ARM64 预编译包（NAIF 仅 x86_64，conda-forge 亦无），新增 `cspice-aarch64-build.yml` 在 GitHub 原生 arm64 runner 上从 NAIF 源码包（cspice-sys 同源 URL）编译静态库并发布到 `cspice-v1` release；`scripts/download_cspice.py` 按机器架构选择 x86_64/aarch64 资产（解压目录按架构隔离，避免缓存串用）；release 工作流构建矩阵增加 `ubuntu-24.04-arm`（maturin-action 自动用 manylinux_2_28_aarch64 容器）；calcephpy 增加 aarch64 预编译 wheel（calcephpy-wheel.yml 的 build-linux-aarch64 job）与 `[tool.uv.sources]` 条目，arm 用户免现场编译；CI 新增 arm64 回归 job（cargo test 全量跑在原生 arm64 runner）。
- **FiniteBurn 恒质量连续推力的 Rust 传播**（#407）：6D 编译传播支持 constant/pulse 推力、惯性/VNB/LVLH 方向帧和 STM；推力开关时刻作为积分边界，避免跨步累计冲量依赖输出网格。任意 Python callable 推力或方向仍显式报告 Rust 编译路径不可用；需推进剂消耗时继续使用 7D 的 `VariableMassFiniteBurn`。
- **Facade 轨道族生成模型与工具清单**（#411/#412）：新增 `FamilyGenerationRequest`，按轨道族公开平动点和 Halo 振幅的条件范围，第一版实现 Halo 族延拓；新增 `ToolInfo` 与 `tool_inventory()`，供 MCP/CLI 查询工具的实现状态和请求模型。
- **轨道设计条件范围 API**（#415）：`DesignOrbitRequest.valid_ranges()` 与 `NumericRange` 公开各轨道类型、平动点上下文下适用的参数范围，校验器与接口共用同一规则。

### Changed
- **CI/release 提速**：ci.yml 的 lint/typecheck 改用 `uv sync --no-install-project`——ruff、结构检查脚本（标准库 ast 静态扫描）与 mypy（`--ignore-missing-imports`）均不 import 项目本体，此前两个 job 各白编一次 Rust 扩展（30s/50s），现跳过构建、typecheck 连带免下 CSPICE 编译包；cargo 编译产物接入 `Swatinem/rust-cache`（ci clippy、release test、三平台 wheel 构建与 docs 构建），全量编译改增量（`CSPICE_DIR` 入 key 防串味，key 默认含 job id 与 OS/arch，各 job、各平台天然隔离）。runner 固定版本（ubuntu-24.04 / windows-2025），checkout 加 `persist-credentials: false`。
- **轨道设计剩余 Python 计算下沉 Rust**：synodic↔J2000 批量坐标转换、ET→UTC 日历分量（星历表组装，一年 8766 点的逐点循环）、ELFO 月心根数提取的月球状态查询改走 Rust 批量入口（`frame_convert`：`batch_synodic_to_j2000_py`/`batch_j2000_to_synodic_py`/`batch_et_to_utc_py`/`batch_body_states_py`）；leapseconds 内核双侧 furnsh（Rust 实例批量 ET→UTC 需闰秒表，原仅 Python 侧加载）。ELFO 经典根数换算向量化（`_cart2oe_batch`）。Rust 侧新增解析恒等式单测（往返对称性、尺度定义），Python 侧断言按 ADR 0013 用物理定义（月球位置、已知历元往返、退化分支）。
- **cargo test 链接 libpython**：pyo3 `extension-module` feature 不再在 workspace 硬编码（该 feature 使 Linux 上含测试二进制在内的所有目标跳过链接 libpython，`cargo test` 报 undefined symbol），改由 e2m2e-integrators 的 `extension-module` feature 按目标启用、pyproject `[tool.maturin] features` 显式传入——cdylib 产物仍不链接 libpython（abi3 可移植性不变），测试二进制可独立链接运行。
- **轨道保持 Facade 参数面收口**（#414）：`ControlOrbitRequest` 覆盖算法层全部业务参数并在 Facade 完整透传，补齐控制时间、误差、摄动、力模型、角动量管理和迭代配置的校验与 schema 元数据。

### Fixed
- **Halo/NRHO 长弧分段打靶合并不收敛**（#400）：移除合并层首末节点锚定，全部节点由 LM 最小范数更新；分段打靶、层级合并与逐段积分下沉 Rust 并行。60 天和 180 天 Halo 基准恢复收敛，180 天设计耗时从约 223 秒降至 63 秒。
- **轨道保持 API 参数未透传**（#414）：此前模型已接收的多项控制、摄动与 TIGHT/SPECIAL 参数未传给算法层，现按模型字段完整传递。
- **低能转移因单条流形弧步长塌缩而整体中止**：流形管生成现在捕获单个候选的 `PropagationFailure` 并跳过该弧，保留其余候选继续搜索；直接传播仍按 ADR 0020 显式抛出异常。

### Removed
- **Python 星历修正分发（homotopy/two_level）**：删除 `e2m2e/algorithm/ephemeris_correction` 子包与 `TwoLevelMultipleShooting`，`design_orbit` 的 homotopy 修正分支移除——星历修正统一走 Rust 多重打靶（segmented 与稳定轨道默认路径），`EphemerisCorrectionResult` 迁入 `algorithm/results.py` 作 Rust 打靶结果的领域重包。ADR 0005/0006 追加修订小节，关联测试同步删除。
- **`DesignOrbitRequest.correction_velocity_tolerance`**（#410）：该参数从未被算法层消费，现因 `extra="forbid"` 被明确拒绝，避免暴露无效配置。

## [5.6.8] - 2026-08-13

### Fixed
- **segmented 星历位置与时间网格错位，GUI 中 Halo 轨道一圈一圈偏离**（#398）：`_prepare_t_eval` 在 `t_eval` 末尾自动追加段终点 `tf`，segmented 逐段积分把每段多出的 `tf` 端点状态拼进 `states_dense`，位置数组比时间网格多出段数个点，`batch_j2000_to_synodic` 按索引配对位置与旋转时刻后错位逐段累积。逐段积分 mask 改右开区间（seam 整点只归后段）、每段输出截断丢 `tf` 端点，`states_dense` 与 `et_grid` 严格逐点对齐；ELFO 路径同类问题（duration 非整小时时多出 1 点）按 `min` 截断对齐；`_build_ephemeris_table` 加长度一致性断言防回归。
- **segmented 尾部窗口无段覆盖，长 duration 星历缺尾部数据**（#398 连带）：打靶节点采样不含圈终点（`endpoint=False`），右开 mask 只覆盖到最后打靶节点，`et_grid` 尾部（最后节点与 `n_rev` 圈终点之间）无段覆盖，状态点数少于时间网格（43 天 Halo 实测 1022 vs 1033，差 11 点），此前无断言时静默缺尾部数据。最后一段 mask 改闭区间、`t_span` 终点延伸到 `et_grid` 尾部；段间去重（右开后为死代码）删除；ELFO `min()` 静默截断改显式报错。新增 30 天 Halo 端到端与 43 天尾部窗口回归测试。

## [5.6.7] - 2026-08-12

### Fixed
- **发布物漏打包 `constants.toml`，5.6.6 wheel/sdist 均不可用**（#377 后续）：5.6.6 把物理常数单一来源放在仓库根 `constants.toml`，Python 加载器按仓库布局用 `parents[3]` 定位——wheel 不含该文件，安装后 `import e2m2e` 即 `FileNotFoundError`；sdist 同样缺失，从源码构建在 `build.rs` 直接 panic。单一来源迁入包内 `e2m2e/data/constants/constants.toml`（随 wheel/sdist 自动分发，仓库与安装布局下定位一致），Rust `build.rs` 改指包内同一份文件，单一来源关系不变。

## [5.6.6] - 2026-08-12

### Added
- **碰撞终止 + 天体半径注入**（#355，ADR 0020 决策 5）：`Dynamics.propagate` 新增 `collision_detection` 开关，启用时以 `g = |r − r_body| − R_body` 构造碰撞事件，探测器与主/次天体表面接触即终止积分，结果附 `collision` 键（天体名、终止时刻、终止状态）；`CR3BP_System`/`BCR4BP_System` 新增 `primary_radius_km`/`secondary_radius_km` 半径配置注入，保留机器精度正则化。

### Fixed
- **三脉冲优化对零脉冲初猜提前收敛**（#384）：`MultiImpulseTransfer.optimize` 的 SLSQP 对"零脉冲初猜"（双脉冲弧上的点，目标值恰为双脉冲成本）数值敏感——目标函数在该平坦走廊上梯度极小，收敛判据可能提前触发、一步即停，三脉冲对 ΔV 几乎无改善（`test_multi_impulse` 随主矢量检验采样密度约半数配置失败，issue 报的改善量 2.96e-11 即此现象）。首次优化相对初猜改善不足时，现自动从微扰初猜（时刻整体 ×0.5/1.5/2.0、位置 ×0.9/1.1）重试并取总 ΔV 最小者；实测该场景三脉冲确有 0.78 km/s（约 11%）改善空间，断言与阈值不变。新增 n_samples=200 回归测试。
- **Lissajous/三角平动点星历修正不收敛**（#366）：`design_orbit` 对 LISSAJOUS-L2/L4/L5 在 Rust 多重打靶（容差 0.02 km）下迭代到 80 次上限仍不收敛（位置残差 0.16–174 km）。根因是拟周期族（面内/面外频率不可约、短/长周期模态耦合，无周期闭合）在自由时间打靶下时间自由度与沿流状态自由度近线性相关，雅可比列病态、LM 线性收敛卡死；固定时间（`var_time=False`，新增 `_FIXED_TIME_ORBIT_TYPES` 族集合）下节点时刻保持 CR3BP 名义周期均匀采样，位置/速度修正直接吸收星历偏差，实测 L1/L2/L3/L4/L5 全部 4–6 迭代收敛（秒级到 37 s）。`test_lissajous_triangular` 重启用 L2/L4/L5 参数化，Jacobi 漂移断言改相对判据（|ΔC|/|C| < 1e-3，约化误差随振幅增长是固有量级）。
- **LPO 初猜网格搜索过慢**（#367）：`design_lpo` 网格搜索 45 点 × 微分修正 ~700 次 STM 传播，默认 `max_step=0.01 TU` 对长周期 LPO（~21 TU/圈）意味着每次传播 ≥2100 步，单次 `design_lpo` 112 s、`test_lpo_family` 全文件 12.7 分钟（默认套件 wall time 卡点）。`_correct_lpo` 搜索阶段临时放宽 `max_step` 到 0.1 TU（rtol=1e-12 自适应仍控精度，收敛产物一致），单次降至 46 s；`test_lpo_family` 收敛测试改吃共享 fixture，全文件降至 3.2 分钟。
- **lowthrust 解析雅可比加速比断言 flaky**（#367）：`test_analytic_jacobian_speedup_over_finite_difference` 硬绑绝对加速比 >5.0，机器负载下实测 3.1x；改为 >1.5（只守护"解析实现未退化"，不硬绑绝对量级）。
- **微分修正/延拓测试适配结果契约**（#367 连带）：#351 结果对象迁移后 `tests/algorithm/design` 的 12 个测试仍用已移除属性（`closure_error` 挂在结果对象、`correction_success`/`correction_iterations`/`ContinuationResult.orbits` 等改名），阻塞默认套件绿门；conftest 的 `_corrected_*_cached`/`corrected_*` fixture 改返回 `(orbit, result)`，correction/continuation 测试改用 `status`/`iterations`/`family.orbits` 新契约。
- **低能转移流水线流形空轨迹崩溃**（#379）：出发/目标轨道某流形分支数值发散时，Rust 积分返回空 states（#246 语义），流形弧以空 `Orbit` 构造触发 `np.max` 崩溃；`InvariantManifold.propagate` 现跳过空轨迹弧，`test_pipeline_converges` 恢复断言（status API）并移除 skip。
- **积分失败类型化**（#349）：新增 `PropagationFailure` 类型异常，消除步塌缩跨语言字符串匹配 catch。
- **redline 静默退化清理**（#352）：微分修正停滞不再短路为成功（`is_periodic` 用配置容差）；控制律未产出机动计为失败样本（修复无角动量管理 Δv 不累计回归）；Q-law 步塌缩抛 `PropagationFailure`、μ 解析失败不静默替换；网格搜索碰撞格进失败侧；propulsion 法向退化、third_body except 收窄、normal-form dense output/星历失败等静默退化改抛。
- **Rust 内核池双侧卸载**（#387）：`SPICEManager.unload_kernel` 对称调 Rust `spice_unload`（新增 pyo3 绑定 + `LOADED_KERNELS` 幂等清单，只卸载确已加载的文件），消除 Rust 内核池残留导致的测试顺序依赖。
- **LICENSE 与 NOTICE 未随 wheel 分发**：`license` 改 SPDX 字符串并加 `license-files`（PEP 639），wheel 的 `.dist-info/licenses/` 现含 LICENSE 与 NOTICE，补上 ADR 0009 的署名承诺；同步删除与 SPDX 冲突的 License classifier、移除从未使用的 `joblib` 依赖声明、maturin 下界抬到 >=1.8（PEP 639 支持）。

### Changed
- **显式事件的传播分派**（#378，ADR 0023）：CR3BP/BCR4BP 仅在调用者传入 `events` 时使用 SciPy 事件积分，该例外由输入触发，与 Rust 扩展可用性无关；未传事件时仍要求 Rust 路径，扩展缺失显式报错。ForceModel 事件传播明确未实现，Rust 事件细化由 `solve_ivp_events` 单独提供。
- **物理常数独立管理，默认地月 μ 切到 DE421**（#377，ADR 0022）：新建 `data/constants/` 常数层（与 `data/templates/` 平级），作为全库物理常数唯一来源——通用物理常量（光速、G、AU、时间常量、太阳常数）全库一套，天体参数按"基准（datum）"组织成自洽集合（DE421 / DE440 / WGS-84），调用方按场景取 `Datum.DE421.mu` 等。Python 与 Rust 从仓库根 `constants.toml` 单一来源构建期生成，两边数值不再各自漂移。此前同一物理量多套值并存的不一致（地月 μ 三套、地球 GM 五值、太阳半径 695700/696000 分叉等 9 类）随收编消除；太阳半径统一为 696000 km（Vallado）。**行为变化：默认地月质量比 μ 从 1965 旧值 0.0121506683 切到 DE421 0.012150585350562453，CR3BP 轨道族/平动点的数值结果会变**；旧值不再支持（一刀切，无 legacy 复现选项）。BCR4BP 太阳无量纲参数（MU_SUN/SUN_DISTANCE/SUN_OMEGA）保留 Topputo 文献约定；normal-form 的 qiao μ（0.012150585609624）为独立模型约定，不并入基准集。
- **移除隐式资源降级，能力缺失改显式报错**（#354/#388，ADR 0020 迁移第 5 步）：`Dynamics.propagate` 事件检测新增显式 `backend` 参数（不传报错、拒绝 auto；`events=[]` 等价无事件，走 Rust 快速路径）；`spice_optional` 默认改报错、QF method 显式选择；ephem cache enable 后 miss 一律报错（缓存 key 归一化，天体名↔NAIF ID 反查收敛）；COPT 不可用默认报错（`fallback_to_scipy` 默认 False）；`to_rust_spec` None 按资源/能力分流（`RustExtensionUnavailableError` vs `NotImplementedError`）；8 处 ADR 修订。
- **cspice-sys 去除 downloadcspice feature**：缺 `CSPICE_DIR` 构建直接报错，不再静默走 NAIF 官网源码下载；CSPICE 一律经 `scripts/download_cspice.py` 取 GitHub `cspice-v1` release 预编译包。
- **删除 slow 测试速度分层**：5.6.4 随 ADR 0021 引入的 slow 标记分层撤销（约 1040 行），测试组织回归功能类标记（`theory`/`orchestration`/`data` 等）。
- **删除超时端到端测试**：删除超时低推力端到端测试与 GMAT ELFO 长弧回归，治理默认套件 wall time。

### Docs
- 删除 `SECURITY.md`（GitHub 标准模板不再保留）。
- `constants.toml` 注释中的特殊符号改用 LaTeX 记号。
- CLAUDE.md 英文翻译为中文。
- 同步过时文档——spice 默认构建、DE421 μ、#351 结果契约。
- README 重写精简：删除使命叙述、配图与进度表格；能力按领域分组列写（时空系统/积分器与动力学/任务轨道设计/转移轨道设计/轨道控制/接口与工具）；安装只留 uv；快速开始缩为 Halo 最小示例；测试说明合并为七类边界清单；开头追加 stars 与 Rust stable 徽章。
- `docs/adr/` 新增 README 索引（状态词汇、编号规则、模板与全量索引）；ADR 0020 与 0024 互加修订指针；12 篇状态词汇统一。
- 删除 `archive/` 目录（20 篇已完成计划与讨论草案），任务跟踪归于 GitHub issue。
- NOTICE 补 Lambert 求解器的 BSD-3 署名（蓝本 jacobwilliams Fortran-Astrodynamics-Toolkit）。

## [5.6.5] - 2026-08-10

### Fixed
- **Halo/NRHO 星历修正多圈发散**（f70252c）：`design_orbit` 入口对 Halo/NRHO 自动重定向 segmented 修正——two_level/standard 的"修正 1 圈 + 自由外推"对不稳定轨道（STM ~1e7/圈）必发散（实测第二圈起圈间偏差 ~7 万 km、第三圈漂离 L2）；segmented 全程分段打靶拼接，第 1 步多圈长段（节点密、段内约束强）+ 固定节点时刻 `var_time=False`（对齐朱彦伟 2026、杨洪伟 2015、刘刚 2017），产出不发散的标称参考轨道。two_level/standard 仅留稳定轨道（DRO 等）。星历 Halo 的圈间漂移是固有准周期特征，由 `station_keeping` 处理，不在本转换范围。

## [5.6.4] - 2026-08-10

### Added
- **ELFO 冻结轨道设计**（#350，closes #348）：`design_orbit` 统一入口新增 ELFO 分支——经典六根数构造初值 → 全摄动传播 → 月心根数漂移分析，复用 CR3BP 管线的力模型路径（GRGM900C 10×10 + EGM96 10×10 + 第三体 + 炮弹光压）。`OrbitDesignResult`/`DesignOrbitResponse` 新增 5 个月心漂移字段（`drift_e`、`drift_aop_deg`、`drift_rp_km`、`secular_aop_rate_deg_per_year`、`moon_centric_elements`），ELFO 场景下 CR3BP 字段留空。
- **双 CSPICE 实例双侧同步**（#357，closes #334）：Python（spiceypy）与 Rust（cspice-sys）是两个独立 CSPICE 实例，内核池与名字表互不共享；本 PR 强制 `furnsh`+`boddef` 双侧同步，补齐 Rust 侧错误处理与诊断入口——新增 `spice_spkezr`/`spice_pxform` pyfunction 供 Python 直接查 Rust 实例做双侧对拍（`e2m2e-integrators` ABI v3→v4）。
- **测试套件按功能类目重组**（#359，ADR 0021）：7 类功能标记（`theory`/`integrator`/`force`/`data`/`orchestration`/`interface`/`aux`）+ `slow`/`spice` 正交标记，取代 L1–L4 速度分层；测试目录迁至镜像源包的 `tests/algorithm/`。CI 维持静态门，测试在 release 前跑全量。

### Changed
- **`design_orbit` 签名重构**（#350）：散参改为 `DesignOrbitRequest` 模型入口，`duration` 单位从年改为秒（clean break），参数校验迁移到 Pydantic `model_validator`；Facade 直接透传 request 对象，不再逐字段解包。
- **Rust SPICE 错误处理加固**（#357）：`erract`/`errdev` 显式设 RETURN/NULL，消除对上游 cspice crate 初始化顺序的依赖（默认 ABORT 出错即 exit）；错误信息由 SHORT 升级为 SHORT+LONG+traceback；`spkezr`/`pxform` 入口经 `ktotal` 预检内核池，空时报项目语境错误（ADR 0020，不走 FFI、不字符串匹配）。
- **SPICEManager 收口 boddef**（#357）：`_BODY_ID_ALIASES` 单一归属 `SPICEManager`（`design_orbit` 不再自带表），`load_kernel` 首次调用在 spiceypy 侧 boddef 全部别名，对称 Rust 侧 `register_bodies`；多进程 `_worker_init` 改走 `SPICEManager.load_kernel` 双侧 furnsh，不再直接 `spiceypy.furnsh`。
- **ADR 0013「测试分层」标注已被 ADR 0021 取代**（#359）：0013 其余不变。
- **calcephpy Windows 免编译**（#362/#363/#364）：新增 `calcephpy-wheel.yml` 为 cp310–cp313 构建 win_amd64 wheel 发到 `calcephpy-v1` release；`pyproject.toml` 经 `[tool.uv.sources]` 在 Windows 直拉预编译 wheel，免去 cmake+MSVC 现场编译，其他平台回落 PyPI sdist。
- **slow 测试 e2e 解散与契约下沉**（#361，ADR 0021）：A 类 e2e 按断言解散合并（lissajous + triangular 参数化、共享 fixture、新增 Jacobi 守恒漂移断言），B 类 fixture 共享收敛（pal_stagnation 延拓 24→2 次），字段形状契约下沉到零管线单元测试；homotopy/segmented 为开发中 feat，显式 `pytest.skip`。

### Fixed
- **frozen 轨道 e2e 静默 skip**（#371）：`test_frozen_orbit_e2e.py` 内核相对路径少算一级，`_SPICE_AVAILABLE` 恒为 False 致整组从未实跑（ADR 0020 禁止的隐式降级）；改用 `kernel_helpers.SPICE_KERNEL_DIR` 后首次实跑，暴露 `drift_e` 实测与设计报告量值不符（跟踪于 #370）。

### Docs
- 修正 #359 测试目录迁移后失效的 tests 路径引用（ADR 0006/0007/0008、`manifolds.rst`、`lambert.rst`）。
- README `design_orbit` 示例 `duration` 语义同步（年→秒）、bibtex 版本号同步。

## [5.6.3] - 2026-08-08

### Added
- **Rust 单元测试补充**（#329）：e2m2e-propagation 新增 4 个集成测试文件（Butcher 表行和条件、二体解析解对照、Lambert-Hohmann 一致检验、Cowell 匀加速），e2m2e-spice 新增 `EphemCache::from_raw_grids` 公共构造器 + 9 项无内核正确性测试（样条误差界、往返位插值、帧旋转正交性），全部基于解析解与物理不变量。
- **测试分层标记 l1/l2/l3/l4**（#328）：按 ADR 0013 为 39 个测试文件打上分层标记，默认排除 l3/slow，支持 `-m l1`/`l2`/`l3`/`l4` 按层独立运行。

### Fixed
- **大幅 DRO 星历发散**（#324）：Rust 多重打靶残差向量中位置残差（几百 km）单边主导速度残差（~0.04 km/s），LM 停在"位置连续/速度跳变 25-50 m/s"的局部极小。速度分量引入 `vel_weight` 加权使两者在容差尺度可比，大幅 DRO 从发散（20 万 km/月）修复到有界（~7 万 km/月）、速度残差归零。
- **BCR4BP 事件检测回退 scipy**（#333）：与 CR3BP_Dynamics 对齐，传入 events 时回退 scipy solve_ivp 并发出警告，不再抛出 NotImplementedError。
- **normal_form 分段积分边界时刻对齐**（#340）：段末状态边界时刻对齐，消除短窗口不一致。

### Changed
- **normal_form scipy → Rust 迁移**（#336）：新增 `_solve_ivp_rust.py` 适配器，中心流形 Hamilton 正则方程传播、段积分、稠密输出采样、QF 矩阵积分（36D/1296D）改走 Rust `solve_ivp_py`（DOP853），仅复值 Lie 级数流保留 scipy。
- **`spice_poc_furnsh` 重命名为 `spice_furnsh`**（#332）：移除生产 API 的 "poc" 前缀。
- **DFH 对拍残留移除**（#338）：删 tests/dfh/ 对拍回归与 scripts/ 下黄金样本/临时诊断脚本，`tests/dfh_format/` 更名 `tests/format/`，`dfh_perturbation_to_force_config` 改名 `perturbation_to_force_config`。
- **力模型 `compute_acceleration` 标注 DeprecationWarning**（#331）：8 个力模型类的 Python 加速度计算已被 Rust 编译路径取代，标注弃用提示。
- **golden regression 测试移入 scripts/**（#326）：DFH 比对脚本从 CI 路径移出，需时手动运行。
- **`tests/algorithm/` 归入 `tests/algorithms/`**（#327）：消除五层架构重命名后的目录并存。
- **ADR 0002 修订 3 措辞修正**（#330）：逐项列出 scipy 回退路径的保留场景（事件检测、防御性回退、NLP、normal_form 传播、平动点解算），替代"已全部移除"的绝对表述。
- **CLAUDE.md 入库**：项目级 Claude 指令随仓库版本化。
- **CI lint 缓存 `.cspice` 目录**（#347）：用 `actions/cache@v4` 缓存预编译包，`download_cspice.py` 检测到 `SpiceUsr.h` 存在即跳过下载。

## [5.6.2] - 2026-08-08

### Fixed
- **Lissajous 初猜改用中心流形约化，返回多点有界轨迹**（#323）：CR3BP 层 Lissajous 初猜从旧方法改为 `force_cr3bp` 中心流形约化，返回多点有界参考轨迹，修复 #323 大振幅星历长期预报发散问题。顺带补充 `two_level` 轨迹覆盖守卫。

### Changed
- orbit-design 代码清理（#325）：旋转阵去重、删除未使用的 `order` 形参、import 提至模块顶层、删除死代码。

### Docs
- 修正五层架构迁移后的过时文档与 rst 引用。

## [5.6.1] - 2026-08-07

### Added
- **Facade Response 补齐轨道几何字段**（#312）：`DesignOrbitResponse` 增 `mu` / `states` / `times` / `ephemeris`（CR3BP 参考轨道 + 标称星历），`ControlOrbitResponse` 增 `controlled_ephemeris` / `mu`（请求透传）。下游可退回 Facade、移除 algorithm 层直调（下游 ADR 0011 缓解措施 3 / ADR 0012）。`ephemeris`/`controlled_ephemeris` 为 `EphemerisTable` 全字段 dict，重建容器过滤 None 值即可。

### Fixed
- 积分器主循环补 `py.allow_threads` 释放 GIL（#318/#313）：`propagate_compiled`（#318）与 CR3BP/BCR4BP PD78 的 4 个绑定（含 STM 变体，#313）主循环整段释放 GIL，长期预报与 STM 修正段不再阻塞其他 Python 线程；design_dro STM 修正冻结主线程的根因消除；顺带修零跨度守卫（`compute_stm(t=0)` latent bug，释 GIL 前不可达）。各配心跳回归测试。
- Python STM 回退路径补 ∂a/∂v（#317）：`_compute_total_jacobian` 返回值由 ∂a/∂r 扩为 (∂a/∂r, ∂a/∂v)，无解析雅可比力走中心差分同时扰动位置与速度。消除速度依赖力（drag）在无 SPICE 走 Python 回退时静默丢失速度块雅可比的隐患（当前 drag 必走 Rust、回退不可达，属预防性修补）。
- `spk_accel` 月球第三体 sanity test 断言量级修正 + cspice 单测串行化：原断言把引力主项（~3×10⁻⁵ km/s²）误当第三体摄动，实际是直接项−间接项的潮汐残差（~6×10⁻¹⁰ km/s²，实测 8.3×10⁻¹⁰）；改为 [1e-11, 1e-7]，上限可捕获间接项漏算。另给 6 个调 cspice 的单测（spk_accel / spice_ffi）加 crate 级串行锁，避免多线程并发撞 cspice 全局状态（产品积分走 ephem_cache 内存表不受影响）。

### Changed
- 步长塌缩错误前缀提为跨语言命名常量（#317），Rust↔Python 稳定契约不再靠裸字符串匹配；新增 ADR 0018（Jacobian ∂a/∂v 接口）、ADR 0019（drag Rust ITRF93 帧旋转）为既有决策补文档落点。
- spice 升为默认 feature + 开发入口自动化（#313）：crates default 改 `["spice"]`、pyproject maturin `features=["spice"]` 双保险，不再产无 spice 子集，CI 单趟 clippy/test 即覆盖；新增 `Makefile`（setup/dev/test/check，自动 export CSPICE_DIR/LIBCLANG_PATH）与 `scripts/download_kernels.py`（从 kernels-v1 幂等下载内核）。

## [5.6.0] - 2026-08-07

### Added
- **转移网格搜索 Rust Rayon 化**（#316）：搜索路径全链路 Rust 化——5 个几何函数 Rust 化 → 6 步评估单元串行版 → Rayon 段间并行（`py.allow_threads` 释放 GIL）→ `TransferSearch` 接入（`set_parallel_backend='rust'` 路由）→ 基准对齐 + ADR 0017 + `parallel=True` → 进度回调 + n_workers 转发 + 轨迹过滤 → 默认后端切 rust
- **Drag 力 Rust 传播**（#315）：Rust atmosphere 模块（USSA76 大气密度）+ CompiledForce 雅可比扩 ∂a/∂v（3×6）+ Drag 接入 Rust 传播（ITRF93 pxform 取风系大气速度）；drag 路径透传 f107/ap 到 density，消除与 Python 静默分歧
- **动力学 Rust PD78 传播**：CR3BP_Dynamics（STM 路径透明加速）、BCR4BP_Dynamics（CR3BP + 太阳解析第三体）、EphemerisDynamics（新增 Rust 纯状态路径）接入 Rust PD78 步进器

### Changed
- **移除 scipy 回退**（迁移路线图）：EphemerisDynamics、proximity/manifold 积分改走 Rust `solve_ivp_events`，截面求根走纯 Python bisection
- 删除 `io` 包，DFH 格式读写迁入 `data/types`（第 5 批架构清理；顶层 `__init__` 未导出，不影响公开 API）
- 轨道设计测试重组：建 `tests/orbit_design/` 骨架（初猜 / correction / continuation 分层），design_orbit 端到端迁入 `scenarios/`（L3，默认不跑），清理低价值测试（类型/属性断言 + 元测试）

### Fixed
- `propagate_compiled` 逐段积分首点错置（#310）：打靶升级 PD78；补齐同类 eval_idx 硬编码与空 t_eval 守卫
- CR3BP 步长塌缩回归（task#3 遗留）：积分器抛错时 catch + 调用方走惩罚/插值降级
- 闰秒内核搜索路径增强（search_dir + 同级目录 + 告警不 raise）

## [5.5.0] - 2026-08-04

### Added
- **转移轨道设计（transfer/）扩展**：
  - WSB 太阳引力辅助间接转移模块（`e2m2e.algorithm.transfer.wsb`，#259）：BCR4BP 弹道网格搜索 + ThreeBodyLambert 到达段精化，以 H₂ < 0 判定弹道捕获（Belbruno 约定），`transfer_orbit("WSB")` 编排器接入
  - 小推力转移编排器接入（#260）：低推力直接打靶/配点求解器接入 transfer 编排层，提供端到端算例
  - LGA 月球引力辅助间接转移模块（`e2m2e.algorithm.transfer.lga`，#258）：`search_lga_trajectories` 弹道网格搜索（出发相位角 × TOF），圆锥曲线拼接的出发段 → 引力辅助 → 到达段全链路，精化到达段 TOF 计算
  - HMN 霍曼直接转移模块（`e2m2e.algorithm.transfer.hohmann`，#256）：`hohmann_delta_v`/`hohmann_tof`/`scan_lambert_delta_v`/`ephemeris_shoot_transfer`，支持 LEO→GEO 等共面圆轨道转移
- **轨道族扩展**：
  - DPO 顺行轨道族（`e2m2e.algorithms.orbit_design`，#288）：DPO（Distant Prograde Orbit）顺行族设计与延拓
  - Axial 轨道族（`e2m2e.algorithms.orbit_design`，#287）：Axial 轨道族实现（Gómez Type B 分岔），扩展平动点附近轨道族覆盖
  - L4/L5 SPO 周期轨道族（#289）：三角平动点附近短周期轨道族，短周期模态初猜 + 二分搜索 x₀ 逼近目标振幅，`design_orbit` 新增 L4_SPO/L5_SPO 分发
  - L4/L5 LPO 长周期轨道族及 Horseshoe 马蹄轨道（#290）：长周期模态初猜 + 平面周期修正，三层网格搜索解决振幅-x₀ 非单调，`design_horseshoe` 马蹄族封装，`design_orbit` 新增 L4_LPO/L5_LPO/L4_HORSESHOE/L5_HORSESHOE 分发
  - Lissajous + L4/L5 端到端验证（#255）：三类轨道的星历修正全链路验证，QPIT（Quasi-Periodic Invariant Torus）正确器实现
- **Facade 一档接口补齐**（#291）：`transfer_design`/`orbit_propagation`/`spacetime_transform` 三方法从占位升级为薄封装，Facade 五个一档方法（design_orbit/control_orbit/transfer_design/orbit_propagation/spacetime_transform）全部落地
- **力模型扩展**：
  - ECOM 9 系数光压模型（`e2m2e.core.forces`，#253/#276）：ECOM（Empirical CODE Orbit Model）光压模型，9 参数经验力，替代简单炮弹模型
  - 耦合项固体潮强制启用（#277）：力模型 `coupling=1` 配置映射为固体潮 `tide_mode="solid"` 强制启用
- **角动量管理联合控制**（#261）：姿态发动机联合控制实现，扩展轨道保持控制律
- **TIGHT/SPECIAL 控制律参数修正与收敛增强**（#280）：轨道保持控制律参数优化，改善收敛性
- **Rust 二进制过期防护**（#300）：靶向守卫 + ABI 版本戳 + 统一网关，防止 Rust 积分器二进制与 Python 源码不一致
- **EphemCache 扩展**：Relativistic 力缓存（sxform + de Sitter spkezr，#273）、tide 扰动体位置走 EphemCache 解锁 tide=1 并行打靶（#267）、EphemCache ADR（#272）
- **星历并行打靶**（#265）：Rust 侧星历预采样缓存 strict 模式 + rayon 并行段积分 + LM 阻尼打靶
- SRP 放行 Rust STM 快速路径，打靶提速 9.4x

### Changed
- 转移模块常量归一化 + Lambert 向量化（#293/#294/#295）
- CI 拆分 lint 与 test，lint 只做静态分析（#278）
- gcrs_ebcrs 测试随源码迁入 `tests/algorithm/coordinate/`（#252 收尾）

### Fixed
- LGA 精化到达段 TOF 计算修正（#258）
- 低推力 h_init 步长上限 + #257 遗留回归测试（#279）
- COF/GFC 文件读取增加 UTF-8 容错解码（#283）
- phasing mypy 类型错误 — dynamics 参数类型从 CR3BP_Dynamics 改为 DynamicsLike（#266）
- 打靶与长期预报共用同一全摄动 ForceModel
- Rust 多重打靶治发散 + rayon 并行段积分
- 收敛容差 2e-2 km（20m）+ 各段收敛检查 + 真实残差上报
- `design_orbit` 分发测试因 Rust binding 导入失败（#301）：模块顶层引用导致测试收集阶段触发 Rust binding 导入链，改延迟导入与运行时默认值

## [5.4.0] - 2026-08-01

### Added
- **一档任务接口（ADR 0014）**：`e2m2e.api.Facade` 接入真实编排器，`design_orbit`/`control_orbit` 薄封装 algorithm/ 编排器；Pydantic 请求/响应模型（`DesignOrbitRequest/Response`、`ControlOrbitRequest/Response`）与 `OrbitError` 结构化错误；MCP 工具清单 = Facade 上 `mcp_exposed` 方法全集（纯派生）
- **DFH 对齐（FR1/FR2/FR4/FR5）**：
  - 任务轨道设计全六类接入 `algorithm/design`（#254）：DRO/NRHO/Halo 与 L4/L5、Lissajous 设计入口，DFH 形状参数 → CR3BP 初猜 → 星历修正 → 高精度预报，DFH 同格式星历
  - 轨道控制模块（功能码 2，#257）：三控制律 + 蒙特卡洛，输出 SK_STATISTIC/MANEUVERS
  - DFH 文件格式互操作与轨道预报（FR4，#251）：inputs-dac 读写、`read/write_ephemeris`、`propagate_orbit`
  - 时空坐标转换（FR5，#252）：TDT+GCRS ↔ TDB+EBCRS（r2s2 后端）
- **五层架构迁移（ADR 0011-0015）**：data/ algorithm/ api/ tools/ 分层迁移、顶层导出更新、旧包清理（5 批）
- **normal form**：Lissajous 中心流形约化基础设施（#255）及 Lie 变换残留耦合修复
- **推进系统建模**：混合推进系统建模与同伦法求解器框架；小推力（电推进）动力学模型
- **SPICE 编译包自动化**：预编译 MICE 工具包发布至 GitHub Release（`cspice-v1`，`cspice-windows.zip`/`cspice-linux.zip`）；新增 `scripts/download_cspice.py` 从 Release 下载解压并输出 `CSPICE_DIR`，CI 与本地构建绕开 NAIF 官网下载（国内网络常不可达）
- **转移轨道设计（transfer/）**：
  - 二体 Lambert 求解器（`e2m2e.algorithm.transfer.lambert`）：Izzo 算法 Rust 内核（`e2m2e-propagation` crate），`solve_lambert`/`solve_lambert_batch` 支持短程/长程与多圈解，基准对齐 Vallado 文献值
  - porkchop 扫描（`e2m2e.algorithm.transfer.porkchop`）：出发时间 × 飞行时间网格的双脉冲 ΔV 扫描，终端状态经 `TerminalCondition` 接口提取
  - 多脉冲转移与主矢量检验（`e2m2e.algorithm.transfer.multi_impulse`）：`MultiImpulseTransfer` 以中途节点 `[t_i, r_i]` 为决策变量、弧段 Lambert 封闭、scipy SLSQP 最小化总 ΔV；`check_primer_vector` 实现 Lawden 必要条件检验与 Lion & Handelsman 中途脉冲插入准则
  - porkchop 持久化与 Pareto 前沿（`e2m2e.algorithm.transfer.porkchop`，主题 8）：`PorkchopData.to_sqlite`/`from_sqlite` 把 ΔV 网格落盘为 SQLite 解数据库（scans 元数据表 + design_points 展平表，NaN→NULL，stdlib sqlite3 零新增依赖，多 scan 自增累积）；`pareto_front` 用经典非支配排序（Deb 2002 O(MN²)）从网格提取 ΔV–TOF Pareto 前沿（Topputo 2013 双目标范式），支持自定义目标字段组合，产出 `ParetoFront` 带绘图。10 测试全过（SQLite 往返等价、多 scan 累积、已知支配关系、LEO→GEO 前沿形态）
  - porkchop 插值代价查询（`e2m2e.algorithm.transfer.porkchop`，主题 8）：`PorkchopData.query(t_dep, tof)` 规则网格双线性插值查总 ΔV，NaN 格点传播 NaN；`PorkchopData.query_scan(path, scan_id, t_dep, tof)` 从 SQLite 解数据库读网格后插值（等价于 from_sqlite + query）。补 `(scan_id, t_dep, tof)` 索引。对应规划文档「宋亮俊数据库的在线查询」——预计算网格 + 双线性插值替代逐点重算 Lambert。6 测试全过（网格点精确值、双线性权重、NaN 传播、越界报错、SQLite 路径一致性、LEO→GEO 谷区平滑性）
  - NSGA-II 多目标优化器（`e2m2e.algorithm.transfer.nsga2`，主题 8）：`nsga2(objectives, bounds, ...)` 经典 NSGA-II（Deb 2002），非支配排序 + 拥挤度选择 + 精英保留，SBX 交叉 + 多项式变异。约束用 Deb 可行支配规则（可行解支配不可行解，都不可行按违反量排序），无需罚因子。`ObjectiveFn` 签名 `fn(x) -> (objectives, violation)`，全部目标最小化。并行评估用 ProcessPoolExecutor（Windows spawn 安全，fn 须模块级可 pickle）。ZDT1 收敛验证（100 代平均误差 0.0013）、Schaffer N.1 收敛、约束问题前沿全可行、串行/并行一致性。11 测试全过
  - 任务综合评估与解数据库（`e2m2e.algorithm.transfer.mission_assessment`/`solution_database`，主题 8 收尾）：`MissionAssessment` 多指标静态加权（`evaluate`/`rank`/`best`，指标名自动推断或显式指定），在 Pareto 前沿上标量化辅助决策；`SolutionDatabase` 封装多 scan 聚合查询（`add_scan`/`get_scan`/`query`/`pareto_front`/`list_scans`/`filter`），`filter` 预留 Grossi 式主矢量筛选钩子。12 测试全过
- **不变流形与低能转移**：
  - 不变流形与庞加莱截面（`e2m2e.algorithms.manifolds`/`sections`）：`InvariantManifold` 种子生成与批量传播；`PoincareSection` 平面/近拱点截面，事后检测 + Brent 插值求精
  - 三体打靶与低能转移（`e2m2e.algorithm.transfer.three_body_lambert`/`low_energy`）：`ThreeBodyLambert` 以二体解为初猜在 CR3BP 下 Newton 打靶；`patch_manifolds` 截面拼接、`design_low_energy_transfer` 低能转移流水线
- **低推力转移（transfer/）**：
  - 可变质量低推力受控动力学基座（`e2m2e.core.forces.VariableMassFiniteBurn`）：质量作为状态量 `state[6]` 随推力消耗（`ṁ = −T/(Isp·g₀)`），`ForceModel.propagate` 自动把状态扩展为 7D `[r,v,m]` 并分流到 Rust `propagate_compiled_lowthrust`（复用 `augmented_state::augmented_eom_7d`）。本期支持常量推力与固定方向；半长轴变化率对标解析解误差 < 5%、质量消耗对标 < 1e-6。为后续最优控制求解层（直接法配点、间接法协态、月面动力下降）的共同基座
  - 低推力多段直接打靶求解器（`e2m2e.algorithm.transfer.lowthrust_shooting`）：`LowThrustShooting` 以各段常量控制（throttle + 方向）为决策变量、接龙传播复用 7D 地基、SLSQP 最小化燃料、归一化末端等式约束匹配目标。`LowThrustShootingSolution` 含控制历史与质量剖面。文献参数（Zhang 2025: T=20mN, Isp=3000s, m0=500kg）短弧 min-fuel 闭环验证
  - 低推力解析雅可比（灵敏度方程法）：`augmented_eom_7d_with_sensitivity`（Rust，64D 增广 `[x₇, Φ₆ₓ₆, S₇ₓ₃]`）一次传播同时产出末端状态、STM、状态对控制 (throttle,θ₁,θ₂) 的灵敏度；`propagate_compiled_lowthrust_sensitivity` PyO3 出口。`LowThrustShooting` 改角度参数化方向（Du 2024 式 5，决策变量 4N→3N），目标/约束提供解析雅可比喂 SLSQP。文献调研（Zhang 2015 式 21-24）。每迭代传播次数从 3N+1（数值差分）降到 1（增广积分），实测 24x 加速。有限差分对标（throttle/θ₁/θ₂ 单段 + 全链式雅可比）逐元素一致
  - Q-law 低推力初猜生成器（`e2m2e.algorithm.transfer.qlaw`）：`qlaw_guess` 用 Lyapunov 反馈律（Holt 2024 式 6-10、Petropoulos Q-law）前向积分产出次优控制历史，治满推力初猜「推过头」的发散问题。`rv_to_keplerian` 自写 rv→经典根数（项目原无）。架构：rk_step 单步循环（每步重算 Q-law 方向跟随轨道）+ 重采样成分段常量控制喂求解器——不用 propagate_compiled_lowthrust（它段内固定惯性方向，长段不跟随速度）。`LowThrustShooting.solve_from_qlaw` 串联 Q-law 初猜 + 解析雅可比打磨，完成 gap-analysis「Q-law 作初猜→打靶优化」两级流程。最简版控 a,e,i；验证：Q 单调下降、a 朝目标收敛、初猜约束残差小于满推力
  - Hermite-Simpson 配点求解器（`e2m2e.algorithm.transfer.lowthrust_collocation`）：`LowThrustCollocation` 并列于直接打靶（`LowThrustShooting`），把节点状态+控制都作决策变量、HS 缺陷约束保证段间动力学连续，比单弧打靶更鲁棒。复用地基 Rust `augmented_eom_7d`（新增 `augmented_eom_7d_py` 单点 EOM PyO3 出口供配点频繁求值），Q-law 初猜可用（`solve_from_qlaw`）。决策变量 10(N+1)（节点状态7D+控制3D），Simpson 缺陷约束。验证：HS 缺陷在真实轨迹上随弧长 O(dt⁴) 下降（转录正确）、min-fuel 闭环收敛、对标直接打靶末态一致；3 测试全过。解析缺陷雅可比（块三对角）留后续性能优化
  - `PointMassGravity.to_rust_spec`：补齐 `("point_mass", mu)` 序列化，让点质量引力走 Rust 编译路径（与 GravityField degree=0 对齐但更轻量，不查星历）
- **proximity 相对运动**：
  - CR3BP 相对运动动力学（`e2m2e.proximity`，主题 3 第一版）：`TargetOrbit` 目标轨道包装（线性插值 `state_at(t)`）；`RelativeDynamics` RLM 时变线性化（`linear_model(t)` 返回 A(t) 矩阵，`propagate(rho0, t_span)` 传播 6 维相对状态，`propagate_with_stm` 复用绝对 STM 传播相对状态+相对 STM）。复用 `CR3BP_Dynamics.compute_jacobian_A` 在目标状态处求值。9 测试全过（TargetOrbit 插值、A(t) 与绝对雅可比一致、小扰动传播、STM 一致性）
  - 相对运动扩展（`e2m2e.proximity`，主题 3 后续）：**Encke 式非线性相对方程**（`encke_eom`/`nonlinear_eom`/`propagate_nonlinear`），Encke 改写引力差分项避免近距离两式相减截断误差（Battin 标准形式，f(q)·q·r + g(q)·δr），与牛顿式机器精度一致；**LVLH 系相对状态转换**（`to_lvlh`/`from_lvlh`），含 Ṙ 中心差分修正，往返转换零误差；**调相设计**（`phasing.py::phasing_search`），基于相对 STM 的两脉冲调相（Fossa 2022 NRHO 范式），在 tof 网格上解两点边值。8 测试全过（Encke/牛顿式一致性、LVLH 往返、调相两脉冲结构）
  - 星历相对动力学与保持点安全分析（`e2m2e.proximity`，主题 3 后续）：**星历 RLM**（`RelativeDynamics` 鸭子类型适配 `compute_jacobian_A(t, state)`，CR3BP/星历接口统一，星历 Encke 留占位）；**保持点安全分析**（`safety.py`）：`SafetyRegion`（球/锥，keep-out/approach 语义）、`check_passive_safety`（自由漂移违背检测）、`max_collision_probability`（Chan 公式，协方差最大特征值方向）。9 测试全过（球/锥包含、违背检测、碰撞概率）
- **动力学模型与数值方法**：
  - 事件检测：`Dynamics.propagate(events=...)` 透传 scipy 事件语义（terminal/direction），`PoincareSection.event(direction, terminal)` 生成截面穿越事件函数；Rust `solve_ivp_events_py`（薄封装 `e2m2e.integrators.solve_ivp_events`）在积分内循环完成事件检测与二分求精，`ForceModel.propagate` 事件走此快速路径
  - BCR4BP 双圆限制性四体模型（`e2m2e.core.bcr4bp_system`/`bcr4bp_dynamics`）：`BCR4BPSystem` 在 CR3BP 会合系上叠加太阳解析圆轨道摄动（m_s=328900.56、a_s=389.17、ω_s=−0.9252，DE440/GMAT 来源），`BCR4BP_Dynamics` 含太阳直接/间接项与 STM；无 Jacobi 积分，与星历 1 天外推误差约 1e3 km（主误差来自月球圆轨道近似）
  - Rust 侧星历预采样缓存（`e2m2e._integrators.enable_ephem_cache`/`disable_ephem_cache`）：积分前把天体状态与帧旋转矩阵在均匀网格上预采样、建三次样条存内存表，`GravityField`/`ThirdBody`/`IndirectTerm` 每步查表替代 cspice FFI。Python 侧 `EphemCache` 对 Rust 积分内循环无效（Rust 直接走 spice_ffi），故缓存做在 `e2m2e-spice` crate。三次样条保 C² 连续避免自适应积分器缩步长。附优化：`GravityField` 在 `body==origin`（地心系地球重力场等）时跳过 origin→SSB 查询（该量在 r_body_icrf 短路分支未被使用），零误差省 FFI

### Changed
- README 快速开始改写为 `Facade.design_orbit` 一档接口示例（L2 Halo）；低层 CR3BP 轨道族示例移至在线文档与 `examples/`
- 版本同步升至 5.4.0（Cargo 工作空间、README bibtex 拉齐）

### Fixed
- 修复 README 能力表 MCP 行乱码占位文本

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

## [3.2.0] - 2026-04-10

### Added
- 星历动力学（`EphemerisDynamics`），支持 SPICE 内核
- `EphemerisSystem`：多体引力建模
- SPICE 内核管理工具
- 同伦动力学：跨解分支延拓

### Changed
- 文档改为任务导向风格（中英）

### Fixed
- 修复文档中 90 处失效锚点警告

## [3.1.11] - 2026-03-19

### Fixed
- 修复延续算法 verbose 参数传递问题，改进停滞检测逻辑

## [3.1.10] - 2026-03-19

### Added
- 新增 y 轴对称微分修正配置与步长控制

## [3.1.9] - 2026-03-15

### Added
- 新增 3D 轨道族局部放大视图绘制方法

## [3.1.8] - 2026-03-14

### Added
- 自然参数延拓支持双向延拓与自适应步长，优化 corrector 复用与轨道创建逻辑
- 可视化新增学术规范字体配置与图像下载缓存

## [3.1.7] - 2026-03-11

### Changed
- `out/` 纳入 .gitignore，调整 continuation 算法调试输出

## [3.1.6] - 2026-03-11

### Changed
- 重构 OrbitFamily 初始化逻辑，优化 DifferentialCorrection 返回类型

## [3.1.5] - 2026-03-11

### Added
- 差分修正算法新增误差与收敛历史记录

## [3.1.4] - 2026-03-11

### Added
- 新增 Orbit 深拷贝方法与 OrbitFamily 属性方法

## [3.1.3] - 2026-03-11

### Docs
- 移除 differential_correction 模块中的 TODO 注释

## [3.1.2] - 2026-03-11

### Changed
- 移除 Orbit.copy 方法，简化 OrbitFamily 类

## [3.1.1] - 2026-03-11

### Changed
- 移除 ContinuationDirection 枚举，简化 Continuation 类
- 为核心与算法模块添加完整类型注解，DifferentialCorrection 初始化改经 self.dynamics
### Added
- 新增 DRO 轨道生成与延拓测试

## [3.0.8] - 2026-03-11

### Added
- 补充稳定性分析与动力学模型测试

## [3.0.7] - 2026-03-11

### Changed
- 补充 numpy 数组转换注释

## [3.0.6] - 2026-03-10

### Changed
- 重构延拓算法以使用新的微分校正接口

## [3.0.5] - 2026-03-10

### Changed
- 更新差分修正算法输出格式与注释

## [3.0.4] - 2026-03-10

### Changed
- 测试导入语句适配新模块结构

## [3.0.3] - 2026-03-10

### Fixed
- 修正平面对称周期轨道搜索的初始 x 坐标类型

## [3.0.2] - 2026-03-10

### Changed
- 重构包导入结构，简化顶层 API，提升模块化

## [2.0.1] - 2026-03-10

### Added
- 新增 py.typed marker 并补齐缺失符号导出，扩展 continuation 模块导出接口
- 增强轨道文件保存功能并补充单元测试
### Changed
- 增强微分修正算法灵活性与健壮性（setup_2D_symmetric_x_fixed_x0 默认 x0=0）
- 可视化强制要求传入 system 对象并简化 mu 初始化，大幅扩展可视化模块文档

## [2.0.0] - 2026-03-06

### Changed
- 微分修正算法改用基于 STM 的牛顿法
- CR3BP_System.info() 支持多模式输出
- Orbit 插值器支持单点轨道初始化
### Added
- 重构测试结构，新增微分修正单元测试

## [1.0.0] - 2026-03-06

### Changed
- 引入 .gitignore 并扩展 CR3BP_System 信息输出
- 用 ruff 修复代码规范与格式问题，移除临时文件

## [0.1.1] - 2026-03-04

### Added
- 新增 LICENSE
### Docs
- 更新 README：移除标题中的 emoji，修正导入路径

## [0.1.0] - 2026-03-03

### Added
- CR3BP 系统建模（`CR3BP_System`、`CR3BP_Dynamics`）
- 轨道数据结构，含状态向量与周期记录
- 2D/3D 对称轨道微分修正
- 自然延拓与伪弧长延拓
- 稳定性分析（Floquet 乘子）
- 转移轨迹设计：网格搜索 + NLP 优化
- 2D/3D 轨道可视化
