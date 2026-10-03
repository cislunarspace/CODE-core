# ADR 0059：CR3BP 多项式流——DA 算术传播内核

- 状态：已采纳（已实现）
- 日期：2026-10-03
- 关联 issue：#786
- 相关：ADR 0058（forces/dyn 拆分）、ADR 0002（Rust 数值内核）、ADR 0055（验收 oracle 类别）

## 背景

issue #784 落地了 dace-rs（DACE 2.1 截断 Taylor 多项式的纯 Rust 实现）的
最小原语面（`Da`/`CompiledDa` 与上下文管理入口）。issue #786 要求在其上
实现 CR3BP 多项式流：初值六分量偏差作为 DA 变量，积分器在 DA 算术下运转，
各求值时刻的状态输出为初值偏差的 k 阶截断 Taylor 多项式，Python 侧经编译
求值接口在任意偏差处快速求值。这是分区判据区间化与后续自动域分裂的
数值底座。

## 决策

1. **组合 dace-rs，不重写算术**。多项式算术、初等函数、编译求值全部来自
   dace-rs 0.1.0（crates.io），`e2m2e-da` 只固定上下文生命周期与 panic
   边界约定，保持零算法。
2. **内核落 `e2m2e-dyn`（`polynomial_flow` 模块），不放 `e2m2e-da`**。
   沿 MEGNO 先例：dyn 收 CR3BP 之上的传播内核；da 保持纯封装。dyn 依赖
   da 取用 `initialized`/`max_order`/`max_variables` 校验调用方上下文。
3. **积分器为经典固定步长 RK4，不做 DA 自适应、不做 Rayon 并行**。单条
   多项式流本征串行；自适应与域分裂留给后续工单。骨架同 Losacco 等人
   2022 年的 DA 多项式流做法：一条名义轨迹加高阶偏差映射。
4. **DA 版 EOM 不设 `MIN_DISTANCE` 钳制**（f64 版 `cr3bp_eom` 有）。多项式
   的求值域不可先验钳制，近碰撞轨迹在 DA 算术下表现为域错误（r1 常数部分
   过零触发 645/641），按 `Runtime` 错误上抛；这是口径差异而非缺陷。
5. **上下文归调用方**。内核只校验（已初始化、变量数 ≥ 6、截断阶在
   1..=上下文阶数内——dace 对超界变量静默给零多项式、对超高截断阶静默
   钳制，必须前置挡住）并临时切换线程局部截断阶（save/restore），绝不
   自行 re-init，避免跨代次混用 panic。
6. **Python 侧返回既有 `Da` pyclass 列表**，系数读取复用
   `cons`/`linear`/`coefficient`/`monomials`，编译求值复用 `CompiledDa`，
   不新建机制。绑定 `propagate_cr3bp_da_py` 注册于 e2m2e-integrators，
   ABI 升至 v29（v28 已被先落地 master 的 `da_ads_split` 入口占用）。

## 后果

- 正面：一条调用链拿到任意阶截断的初值偏差映射；一阶截断与既有 STM 互为
  对拍（研究级 1e-10）、二阶系数与中心差分 Hessian 抽查一致、Jacobi 常数
  DA 展开余项只剩积分误差，三条验收研究级断言进 Rust 测试；Python 侧另有
  返回契约与错误路径的行为测试。
- 正面：NRHO 短弧的一致半径-阶数曲线（`scripts/da_uniform_radius_curve.py`
  → `datasets/da_flow/uniform_radius_nrho_l2.csv`）给出阶数选择的量化依据，
  曲线随阶数单调不降；研究脚本不进默认 pytest。issue 验收原文写脚本与数据
  落 scripts/，数据实际按仓库 datasets/ 数据留档惯例改放 datasets/da_flow/
  并提交入库，脚本仍在 scripts/。
- 负面：固定步长 RK4 在长弧上步数开销线性于弧长×精度要求；阶数 8 的单次
  传播在 debug 构建下明显偏慢，脚本引导 `make dev-release`。
- 风险控制：截断阶 save/restore 保证内核不污染调用方的线程局部设置；
  `allow_threads` 不换线程，DA 上下文进程级 + 截断阶线程局部的组合安全。

## 修订记录

- 2026-10-03：首次记录。
