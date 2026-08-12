# 健壮性清理迁移清单（ADR 0020 执行跟踪）

**关联**：[ADR 0020](../adr/0020-failure-handling-policy.md)
**状态**：初始工作清单。条目取自 2026-08-09 四路并行盘点，行号迁移时逐条核实。

## 分类口径

| 类别 | 含义 | 动作 |
|---|---|---|
| A | 掩盖真实失败（谎报成功、静默返回近似值、implicit None 原因丢失） | 改报错或带标记返回 |
| B | 搜索/优化不可行（网格某格发散、NLP 候选不可行、DC 单步不收敛） | 带统一 `status` 标记返回（决策 3） |
| C | 多后端/资源降级（SPICE/Rust/COPT 不可用自动换） | 资源缺失改报错；能力缺失改显式 backend（决策 4） |
| D→保留 | 机器精度正则化（MIN_DISTANCE≈1e-10、arccos clip、步长地板作失败判定） | 保留 |
| D→改 | 物理量级钳位 / 步长地板作抬升 | 改碰撞终止 / 删抬升（决策 5） |

## 全量统计

四路盘点约 139 条：A ~36、B ~33、C ~36、D ~30（D 中绝大多数为机器精度正则化→保留，仅物理量级少数→改碰撞终止）。unclear ~4。

## 条目清单（按盘点分区）

### 分区 1：dynamics + integrators

- **[A]** `propagate_orbit_state_at_time` 空states退回 np.interp 当成功 — `dynamics.py:688-699` — 改：空 states 时带 failure 标记/抛，不退回插值
- **[A]** `get_jacobi_constant` 奇点返回 nan — `cr3bp_system.py:326-332` — 改：抛（奇点是真实退化，不是合法值）
- **[B]** 步长塌缩 catch → 空 states — `dynamics.py:48-54, 611-633` — 改：抛 `PropagationFailure`，搜索 wrapper catch 转 infeasible；返回带标记，不靠 `len==0` 嗅探
- **[C]** `_HAS_RUST_CR3BP_STM` 等导入门控回退 scipy — `dynamics.py:38-46`、`integrators.py:11-128` — 改：Rust 不可用报错（资源缺失）
- **[C]** events 传入时回退 scipy — `dynamics.py:521,596`、`bcr4bp_dynamics.py:204-212` — 改：显式 `backend` 参数（能力缺失），不 auto
- **[D→保留]** MIN_DISTANCE 奇点钳位 (1e-10) — `dynamics.py:437-439`、`bcr4bp_dynamics.py:93,116,156`、`ephemeris_dynamics.py:282-306` — 保留（机器精度正则化）；`ephemeris_dynamics` 的 1e-6 km 钳位核实是否物理量级

### 分区 2：coordinate + data + normal_form

- **[A]** `_dense_output` 静默 2 点线性回退（污染下游 FFT） — `dynamical_substitution.py:319-323` — 改：抛
- **[A]** `_failure()` 吞 5 阶段异常返 `success=False` 丢回溯 — `pipeline.py:134-189, 232-251` — 改：raise，不吞
- **[A]** `_bdot2a` `except Exception` → 常数 CR3BP 矩阵 — `dynamical_substitution.py:472-504` — 改：抛
- **[A]** `_eval_coef` 求不出返 0.0（哈密顿量静默错） — `hamiltonian.py:380-410` — 改：抛
- **[A]** `enable_ephem_cache` broad except 藏 body 名 typo — `kernels/manager.py:292-335` — 改：收窄异常
- **[A]** EOP `at_utc_mjd` clamp 越界返边界值 — `frames/eop.py:95-105` — 改：越界抛（物理量级钳位）
- **[A]** `compatibility="gmat"` 静默切 `eop_extrapolation="clamp"` — `coordinate/standard_axes.py:94-100` — 改：显式
- **[C]** `spice_optional=True` 三级链换物理模型 — `pipeline.py:131` + `dynamical_substitution.py:266-279` + `quasi_floquet.py:986-996` — 改：SPICE 不可用报错；QF method 不静默覆盖
- **[C]** Rust→sympy 回退（仅性能） — `hamiltonian.py:502-519`、`qf_projection.py:71-88` — 改：Rust 不可用报错
- **[C]** NAFF→FFT 回退（精度降级） — `normal_form/fft.py:536-554` — 改：显式选 method
- **[D]** `_system_mu` 静默替换地球月球 MU — `context.py:195-203` — 改：显式注入或报错

### 分区 3：solver + station_keeping + manifold + design

- **[A] 红线** DC-4/12 停滞 1e-8 短路标 `converged=True` — `differential_correction.py:730-744, 951-962` — 改：删短路，走已有 else（带标记失败）
- **[A]** DC-1/2/3/5/7 六条 implicit None（积分异常/发散/雅可比奇异/max_iter/周期无效） — `differential_correction.py:617-793` — 改：带 `status` 结构化结果，`termination_reason` 透传到对象
- **[A]** DC-8 闭合误差事后速度半误差修补 — `differential_correction.py:1042-1069` — 改：报失败，不修补
- **[A]** CT-4 PAL 雅可比奇异转发陈旧 Xnew — `continuation.py:600-605` — 改：带标记
- **[A]** CT-6 PAL 超范围静默换欧拉预测 — `continuation.py:627-642` — 改：带标记
- **[A]** CT-9 单次修正失败静默终止族 — `continuation.py:750-754` — 改：带标记透传到 OrbitFamily
- **[A] 红线** MC-2/3 控制器 None 当成功（统计偏差） — `monte_carlo.py:484-511, 534-541` — 改：`None → failed_k=True`
- **[A]** SP-1/2 无穿越/不收敛返 None — `special_point.py:234-235, 278-279` — 改：带标记
- **[A]** TP-1 max_iter 耗尽返无标志 Δv — `target_point.py:90-103` — 改：带收敛标志
- **[A]** SC-1 二分 max_iter 返中点（不可达） — `sections.py:73-82` — 改：带标记（低优）
- **[B]** DC-6/10 时间钳位、CT-2 retry/5/7/8、TL-1/3、EC-2/4、SP-3/4、TP-2 — 各搜索 retry/clamp/同伦策略 — 保留（算法策略），对齐统一 `status`
- **[C]** DO-2 body-fixed 内核静默跳过 — `design_orbit.py:208-212` — 改：报错
- **[C]** MF-1 `n_workers!=1` 静默串行 — `manifolds.py:165-166` — 改：报错或显式
- **[D]** CT-1/3、MS-1、TL-2、DO-1、SF-1、PH-1/2 — 输入校验/上限带标记 — 保留（决策 3 flagged 范式）

### 分区 4：forces + transfer

- **[A]** `check_collision` 空传播返 (False,False) 假阴性 — `transfer_optimization.py:412-448` — 改：空传播 → 不可行标记
- **[A]** `_classify_transfer` 空传播返 DIRECT — `transfer_optimization.py:579-580` — 改：不可行标记
- **[A]** `propulsion` 共面退化 → 任意 [1,0,0] 法向 — `propulsion.py:67` — 改：抛
- **[A]** `multi_impulse` 初猜 except → 线性插值 — `multi_impulse.py:446-455` — 改：抛/带标记
- **[A]** `qlaw._resolve_mu` 查询失败 → 地球 μ — `qlaw.py:415-429` — 改：抛
- **[A]** `qlaw` 步塌缩 `h<1e-6: h=step`（与 dynamics 同类，空转到 200 万步） — `qlaw.py:379-380` — 改：抛 `PropagationFailure` + 连续拒绝计数保险
- **[A]** `lowthrust_shooting` SLSQP 油门静默 clip [0,1] — `lowthrust_shooting.py:297-299` — 改：报约束违反
- **[A]** `third_body._name_or_id` bods2c except 吞 — `third_body_gravity.py:51-59` — 改：收窄
- **[A→删]** `force_model:804` `h=max(h,min_step)` 地板 — `force_model.py:804` — 删（保留 `min_step` 作失败判定阈值 `:853`，对齐 Rust `solve_ivp.rs:244-248`）
- **[B] 红线** NLP `dv=1e10` 目标+约束双惩罚 — `transfer_optimization.py:222-304` — 改：去目标惩罚（`objective=2e10`），留约束冲突标记（`pos_violation`/`vel_constraint`）
- **[B]** `nlp_scipy`/`nlp_copt` try-except 返初始猜 — `nlp_scipy.py:127-145`、`nlp_copt.py:277-298` — 对齐统一 `status`
- **[B]** `multi_impulse._CLOSURE_PENALTY=1e3` — `multi_impulse.py:43-45, 491-500` — 对齐
- **[B]** `search_single_departure` `integration_failed` 标记 — `search_parallel.py:126-151` — 保留范式，对齐 `status` 枚举
- **[B]** `three_body_lambert` converged=False / `wsb` / `lga` / `porkchop` NaN / `hohmann` / `nsga2` Deb 规则 — 各文件 — 保留（搜索不可行），对齐统一 `status`
- **[C]** `force_model` Rust 路径/STM/事件门控 — `force_model.py:346-394, 768-779` — 改：Rust 不可用报错
- **[C]** `to_rust_spec` 返 None（drag/gravity_field/thrust/physical_model） — 各力文件 — 改：报错或显式 backend（注：部分是"力无 Rust 实现"=能力缺失 → 显式 backend；非资源缺失）
- **[C]** `_default_parallel_backend` import 回退 — `transfer_search.py:39-50` — 改：报错
- **[C]** `search_parallel` monkeypatch 缝 + Rust 缺失回退 — `search_parallel.py:300-314, 405-443` — monkeypatch 缝保留（测试豁免）；Rust 缺失回退改报错
- **[C]** `nlp_copt` import stub + `optimize_with_copt` 回退 — `nlp_copt.py:29-39, 330-412` — 改：报错，显式后端
- **[D→保留]** arccos/arcsin 前 clip [-1,1] — `qlaw.py:68-258` — 保留（IEEE 754 防护）
- **[D→保留]** `point_mass`/`indirect_term` r<1e-15 → zeros — `point_mass_gravity.py:62-63`、`indirect_term.py:102-103` — 保留（机器精度正则化）
- **[D→保留]** `third_body` MIN_DISTANCE 钳位 + warn — `third_body_gravity.py:135-144` — 保留
- **[D→保留]** `_estimate_h` clamp / nsga2 SBX clip — `qlaw.py:432-434`、`lowthrust_shooting.py:402-405`、`nsga2.py:374-375,413` — 保留（数值防护）
- **[D→改]** `three_body_lambert` `phi_rv` 奇异 lstsq — `three_body_lambert.py:147-148` — 改：LinAlgError 带标记返回（边界验证确认 A，非 D）

## 迁移顺序（同 ADR 0020 结果-变更）

1. **地基**：加 `PropagationFailure` 类型异常（`e2m2e/exceptions.py`），零测试破坏。
2. **决策 3**：统一 `ConvergenceState` status 规范，各搜索结果对象对齐。多数测试断言 happy path，破坏小。
3. **决策 1 红线**：修谎报/藏失败（DC 停滞短路、MC 控制器 None 当成功、`propagate_orbit_state_at_time` 空states退回插值、网格搜索碰撞格 success=True、qlaw 步塌缩空转、qlaw `_resolve_mu` 静默地球 μ）。
4. **决策 2**：`_propagate_state_only` 空states改带 failure 标记；同步改 `transfer_optimization.py` 的 `len==0` 嗅探与 NLP 双惩罚。
5. **决策 4**：移除资源降级（8 处 ADR 修订）；事件检测加显式 backend，不 auto。
6. **决策 5**：碰撞终止 + body-radius 注入（最高风险，须先确保碰撞事件终止再动任何物理量级钳位）。

## 跟踪

迁移推进时逐条勾选并补 commit。每步建议独立 PR，配回归测试（先写复现旧行为的测试，改后更新断言）。
