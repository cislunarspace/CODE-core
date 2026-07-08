# PRD：将 qiao Hamiltonian 正规化能力集成到 e2m2e

## 问题陈述

e2m2e 目前已经能生成 CR3BP 轨道族、在星历模型下传播轨道、并用多重打靶修正 patch points，但缺少对**星历模型下平动点附近可积结构**的利用。用户无法把一条星历轨道转换成统一的作用量-角变量/表征参数，也无法在指定历元生成动力学替代轨道作为高保真参考解。

qiao 仓库已经实现了完整的 Hamiltonian 正规化流水线。本 PRD 要求把这套能力以原生 e2m2e 模块的形式引入，让 e2m2e 用户在保留现有工作流的同时，获得平动点附近的高阶解析工具。

## 方案

新增 `e2m2e.algorithms.normal_form` 模块，提供：

1. 从 `e2m2e.System` 构造平动点附近的 Hamiltonian 正规化上下文；
2. 一键式 `NormalFormPipeline`，把星历轨道还原为动力学替代轨道、QF 变换、中心流形化简，并输出编目参数；
3. 分步 reducer，供研究者调试和替换中间步骤；
4. 函数式坐标变换链（`rho ↔ EM ↔ DS ↔ QF ↔ CM ↔ param`）；
5. 自动处理 e2m2e SI 单位与 qiao 归一化单位之间的转换。

qiao 的数学代码不作为“外来遗留代码”隔离，而是作为 `normal_form` 的正式子模块公开复用。

## 用户故事

1. 作为**轨道设计师**，我想要把一条地月 `L2` 附近的星历轨道转换成表征参数，以便识别它属于 Halo、Lissajous 还是 quasi-Halo 族。
2. 作为**任务分析师**，我想要在指定历元生成 `L1` 动力学替代轨道，以便把它作为后续星历修正的参考初值。
3. 作为**研究人员**，我想要分步执行 Legendre 展开、动力学替代、QF 变换、中心流形化简，以便检查每一步的收敛性与中间结果。
4. 作为**下游工具开发者**，我想要一个稳定的 `NormalFormResult` 句柄，以便在不重新跑完整流水线的情况下反复调用 `rho_to_param`。

## 实现决策

- **新增模块**：`e2m2e.algorithms.normal_form`，与现有 `differential_correction`、`continuation` 等算法并列。
- **上下文对象**：`NormalFormContext` 负责把 `e2m2e.System` + 平动点 + 历元 + 展开阶数翻译为 qiao 所需的归一化常数、基础频率、中心流形频率和特征指数，并自动处理单位转换。
- **高层接口**：`NormalFormPipeline(context).reduce(orbit)` 返回 `NormalFormResult`，后者提供 `catalog_transformer`、`substitute_orbit`、`qf_matrix(t)` 等封装访问器。
- **分步接口**：`DynamicalSubstituteCorrector`、`QuasiFloquetReducer`、`CenterManifoldReducer`、`LibrationCatalogTransformer` 可独立使用。
- **坐标变换**：以**函数式接口**为主，链式函数位于 `normal_form.coord_trans` 子包中。
- **单位约定**：public 接口使用 e2m2e 的 SI / J2000 ET；内部数学实现继续运行在 qiao 归一化单位下。转换逻辑集中在 `NormalFormContext` 与 `units` 模块，不允许 qiao 代码直接读取 e2m2e 原始状态。
- **平动点支持**：`L1`–`L5`。`L1`–`L3` 使用共线 `gamma` 参数；`L4`/`L5` 使用旋转矩阵方法。
- **频率分析**：NAFF 为首选，但核心功能不得依赖 NAFF；NAFF 不可用时降级到 FFT。
- **依赖策略**：`sympy` 与 `joblib` 放入 `normal-form` optional dependency group，避免污染核心安装。
- **结果序列化**：`NormalFormResult` 支持保存/加载，但 7 GB 级大系数表允许懒加载或由用户显式提供路径。
- **不暴露原始 qiao 数据结构**：`NormalFormResult` 不直接暴露 `W_poly`、`QFtrans_mat` 等内部结构，只通过封装访问器提供结果。

## 测试决策

- **测试切面**：单一高层面 `tests/algorithms/normal_form/test_pipeline.py`，对 `L1_Halo_Large` 跑一次完整 `NormalFormPipeline`，并验证 `rho_to_param` / `param_to_rho` 往返误差在 qiao 等价结果容差范围内。
- **测试原则**：只测外部行为（输入 `Orbit` / `System`，输出 `param` 与还原误差），不测实现细节（如 Poisson 括号内部系数）。
- **先例**：`tests/algorithms/test_dro_ephemeris_correction.py` 等已有算法测试采用“输入-输出”回归模式，可直接复用。
- **补充**：在实现过程中可新增针对 `NormalFormContext` 单位转换、各 reducer 收敛性的单元测试，但这些属于开发期测试，不占用主要测试切面。

## 不在范围内

- qiao 的 Monte-Carlo 实验框架。
- qiao 的 Rust 性能路径（后续可作为独立优化跟进）。
- qiao 的绘图 CLI。
- 替换 e2m2e 现有的 CR3BP 轨道族生成器；normal_form 是新增垂直切片。
- 真实任务级的轨道识别 UI 或数据库化编目。

## 补充说明

- 本次迁移基于用户确认：qiao 代码同样由本仓库维护者编写，因此不作为“私有移植代码”隔离，而是作为 `normal_form` 的公开子模块复用。
- 三个待决定问题可在实现过程中逐步澄清：
  1. `NormalFormResult` 是否额外暴露原始 qiao 数据结构给高级用户？
  2. `normal-form` 依赖是保持 optional 还是提升为 main dependencies？
  3. 回归测试 fixtures 是否使用 Git LFS 管理？
