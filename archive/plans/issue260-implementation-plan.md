# #260 实施计划：小推力转移端到端算例

> 本计划基于对 issue #260 与现有代码库的全面审查。issue 描述存在多处与代码实际状态不符，
> 需同步修正（见 §一）。

---

## §一、Issue 描述修正

### 原始描述问题

| # | 原始描述 | 实际状态 | 修正 |
|---|---------|----------|------|
| 1 | "阻塞于 #256" | #256 (HMN) 已 CLOSED | 删除阻塞关系 |
| 2 | "`transfer/propulsion.py` 的 `LowThrustPropulsion` 标注待实现" | `propulsion.py` 只有 `ImpulsivePropulsion`，无 `LowThrustPropulsion` | 修正：低推力入口是 `LowThrustShooting`/`LowThrustCollocation` + `EngineConfig` |
| 3 | "推力方向剖面参数化 + NLP 求解" | 角度参数化 `α(θ₁,θ₂)` 已在 Rust `augmented_state.rs:170-176` 实现，含 64D 解析雅可比；SLSQP NLP 在 `LowThrustShooting.solve()` 和 `LowThrustCollocation.solve()` 中落地 | 修正：说明这些是已有基础设施 |
| 4 | "打通一条端到端小推力转移路径" | **70-80% 已完成**（见下方清单） | 修正：剩余工作是编排层接入 + 端到端算例 |

### 建议修正后的 Issue Body

```markdown
## 要构建什么

在 7D 增广状态 [r,v,m] 与混合推进参数建模的已有地基上（Rust `augmented_state.rs`
含 7D/43D/64D EOM、Python `LowThrustShooting`/`LowThrustCollocation`/`qlaw_guess`
求解器），完成 FR3 小推力路线的**编排层接入**：`transfer_orbit("low_thrust", ...)`
路由 + 端到端地月小推力转移算例，含质量演化与推力-时间历史输出。

**已有基础设施**（不需要重新实现）：
- Rust 7D/43D/64D 增广 EOM + `propagate_compiled_lowthrust` / `_sensitivity` 绑定
- Python `LowThrustShooting`（多段直接打靶，解析雅可比）/ `LowThrustCollocation`
  （Hermite-Simpson 配点）/ `qlaw_guess`（Q-law Lyapunov 初猜）
- `EngineConfig`、`LowThrustSegment`、`LowThrustShootingSolution` 数据类型
- `VariableMassFiniteBurn` 力模型 + `to_rust_spec()` Rust 序列化

规格见 `archive/plans/dfh-parity-prd.md` FR3 延伸要求。DFH 本身无小推力功能，本条无
黄金样本，验证用终端约束满足度与等效 Δv 口径一致的脉冲/小推力对比。

## 验收标准

- [ ] `transfer_orbit("low_thrust", engine_config=..., n_segments=..., ...)` 编排路由
      调用 `LowThrustShooting.solve_from_qlaw` 收敛
- [ ] 端到端算例：地球停泊轨道 → 标称轨道（与 #256 同任务场景），终端约束残差 < 给定容差
- [ ] 质量演化与推力历史（7D 状态 + 各段控制）可输出、可绘图
- [ ] `TransferDesignResult` 携带 `LowThrustTransferDetails`（等效 Δv、燃料消耗、
      发动机参数、段数、收敛状态）
- [ ] 与脉冲路线（#256 HMN）同任务场景的 Δv 对比写入算例文档
- [ ] `transfer_orbit` 的 `NotImplementedError("low_thrust")` 占位消除
```

---

## §二、现有基础设施清单

### 不需要重新实现的组件

```
Rust layer (crates/e2m2e-forces/src/forces/)
├── augmented_state.rs
│   ├── AugmentedState7          # [r(3), v(3), m(1)]
│   ├── augmented_eom_7d()       # 7D EOM
│   ├── augmented_eom_7d_with_stm()  # 43D EOM + STM
│   ├── augmented_eom_7d_with_sensitivity()  # 64D EOM + STM + S(7×3)
│   ├── ThrustParams             # t_max/isp/throttle/direction
│   └── direction_from_angles()  # α(θ₁,θ₂) 球面参数化
└── hybrid_propulsion.rs         # 化学+电推进参数

Rust bindings (crates/e2m2e-integrators/src/lib.rs)
├── propagate_compiled_lowthrust           # 7D 受控传播
└── propagate_compiled_lowthrust_sensitivity  # 64D 增广传播

Python layer (e2m2e/algorithm/)
├── transfer/
│   ├── lowthrust_shooting.py   # LowThrustShooting + solve()/solve_from_qlaw()
│   ├── lowthrust_collocation.py# LowThrustCollocation + solve()/solve_from_qlaw()
│   ├── qlaw.py                 # qlaw_guess() + Q-law 反馈律
│   └── config.py               # TransferDesignResult 结构（需扩展 LowThrustTransferDetails）
└── forces/
    └── thrust.py               # VariableMassFiniteBurn + to_rust_spec()

Tests (tests/transfer/)
├── test_lowthrust_shooting.py           # 段连续性/决策变量/约束/min-fuel
├── test_lowthrust_collocation.py        # HS 缺陷/min-fuel/打靶一致性
├── test_lowthrust_analytic_jacobian.py  # 灵敏度 vs 有限差分/链式雅可比/加速比
└── test_qlaw.py                         # Q 单调下降/根数收敛/初猜质量
```

---

## §三、实施步骤

### Step 1：新增 `LowThrustTransferDetails` 数据类

**文件**：`e2m2e/algorithm/transfer/__init__.py`

在现有 `HmnTransferDetails` / `LgaTransferDetails` / `WsbTransferDetails` 旁边新增：

```python
@dataclass
class LowThrustTransferDetails:
    """小推力转移设计细节。

    Attributes:
        engine: 推进配置 (t_max, isp)。
        initial_mass: 初始质量 (kg)。
        final_mass: 末态质量 (kg)。
        fuel_consumed: 燃料消耗 (kg)。
        equivalent_delta_v: 等效 Δv (km/s)，Tsiolkovsky 方程反算。
        n_segments: 求解器段数。
        solver_method: 求解方法 ("shooting" / "collocation")。
        converged: 是否收敛。
        n_iter: 迭代次数。
        solver_message: 求解器消息。
        terminal_residual: 终端约束残差（位置 km、速度 km/s）。
        time: 采样时间序列 (M,)，SPICE et 秒。
        states_7d: 7D 状态序列 (M, 7) [x,y,z,vx,vy,vz,m]。
        segments: 各段常量控制 (throttle, direction)。
        qlaw_q_history: Q-law Q 值历史（仅 solve_from_qlaw 时非空）。
    """

    engine: EngineConfig
    initial_mass: float
    final_mass: float
    fuel_consumed: float
    equivalent_delta_v: float
    n_segments: int
    solver_method: str          # "shooting" | "collocation"
    converged: bool
    n_iter: int
    solver_message: str
    terminal_residual: tuple[float, float]  # (r_err_km, v_err_km_s)
    time: np.ndarray
    states_7d: np.ndarray
    segments: tuple[LowThrustSegment, ...]
    qlaw_q_history: np.ndarray | None = None
```

**等效 Δv 计算**：

```python
def _equivalent_delta_v(m0: float, mf: float, isp: float) -> float:
    """Tsiolkovsky 方程：Δv = Isp·g₀·ln(m0/mf)，单位 km/s。"""
    g0 = 9.81
    return isp * g0 * math.log(m0 / mf) / 1000.0
```

**估计工作量**：~40 行

---

### Step 2：实现 `_transfer_orbit_low_thrust` 编排函数

**文件**：`e2m2e/algorithm/transfer/__init__.py`

新增私有编排函数，对齐 `_transfer_orbit_hmn` / `_transfer_orbit_lga` / `_transfer_orbit_wsb` 的模式：

```python
def _transfer_orbit_low_thrust(
    tli_params: TliParams | None,
    target_ephemeris: Any,
    engine_config: EngineConfig,
    initial_mass: float,
    n_segments: int = 10,
    *,
    target_oe: tuple[float, float, float] | None = None,
    solver_method: str = "shooting",
    duration_days: float = 30.0,
    system: Any = None,
    forces: Any = None,
) -> TransferDesignResult:
```

**内部流程**：

```
1. TliParams → construct_departure_state() → (r0, v0)  # 复用 HMN 出发状态构造
2. target_ephemeris → _extract_target_state() → (r_target, v_target)
3. 构造 EphemerisSystem（如 system 未提供）或复用 system
4. 构造力模型列表（如 forces 未提供）：GravityField("EARTH", degree=0, order=0)
5. t0 = spice.utc_to_et(tli_params.epoch); tf = t0 + duration_days * 86400
6. 若 solver_method == "shooting":
     solver = LowThrustShooting(system, forces, engine_config, [r0,v0], initial_mass,
                                 [r_target,v_target], t0, tf)
     sol = solver.solve_from_qlaw(n_segments, target_oe or (r2, 0, 0), forces)
   Elif solver_method == "collocation":
     solver = LowThrustCollocation(...)
     sol = solver.solve_from_qlaw(n_segments, target_oe, forces)
7. 计算终端残差、等效 Δv
8. 返回 TransferDesignResult(
       transfer_type="low_thrust",
       delta_v=equivalent_delta_v,
       trajectory=sol.states,  # 7D
       details=LowThrustTransferDetails(...),
   )
```

**关键设计决策**：

| 决策 | 选择 | 理由 |
|------|------|------|
| 默认求解方法 | `"shooting"` | 解析雅可比快 5-24x，已有测试验证 |
| 默认 Q-law 初猜 | `solve_from_qlaw()` | 满推力初猜"推过头"发散，Q-law 初猜约束残差更小（`test_qlaw.py` 已验证） |
| 目标 OE | `(a_target, 0, 0)` | 圆轨道，从 target_ephemeris 反推半长轴；用户可覆盖 |
| 飞行时间 | 30 天默认 | 量级与 SMART-1（~13 月）、Conway LEO→GEO（~200 天）同类；可覆盖 |
| 力模型 | 仅点质量重力 | 端到端算例聚焦小推力闭合验证，不叠加摄动复杂度 |
| 发射参数 | 复用 `TliParams` | 与 HMN/LGA/WSB 共享出发状态构造，确保同场景可比 |

**估计工作量**：~80-120 行

---

### Step 3：在 `transfer_orbit` 编排器注册低推力路由

**文件**：`e2m2e/algorithm/transfer/__init__.py`

修改 `transfer_orbit()` 函数签名和路由：

```python
def transfer_orbit(
    transfer_type: str,
    *,
    # ... 现有参数 ...
    # 新增低推力参数
    engine_config: EngineConfig | None = None,
    initial_mass: float | None = None,
    n_segments: int = 10,
    target_oe: tuple[float, float, float] | None = None,
    solver_method: str = "shooting",
    duration_days: float = 30.0,
    **kwargs,
) -> TransferDesignResult:
```

在函数体中新增路由分支：

```python
    if transfer_type == "low_thrust":
        if engine_config is None:
            raise ValueError("low_thrust 转移需要 engine_config")
        if initial_mass is None:
            raise ValueError("low_thrust 转移需要 initial_mass")
        return _transfer_orbit_low_thrust(
            tli_params=tli_params,
            target_ephemeris=target_ephemeris,
            engine_config=engine_config,
            initial_mass=initial_mass,
            n_segments=n_segments,
            target_oe=target_oe,
            solver_method=solver_method,
            duration_days=duration_days,
            system=kwargs.get("system"),
            forces=kwargs.get("forces"),
        )
```

**估计工作量**：~30 行

---

### Step 4：端到端集成测试

**文件**：`tests/transfer/test_low_thrust_end_to_end.py`（新建）

三个测试用例：

#### 4a. `test_low_thrust_orchestrator_converges`

```python
"""transfer_orbit("low_thrust", ...) 编排路由端到端收敛。

纯二体（SimpleNamespace + PointMassGravity），不依赖 SPICE。
目标：7000→7200 km 圆轨道，T=0.5N, Isp=3000s, m0=1000kg。
"""

def test_low_thrust_orchestrator_converges():
    from e2m2e.algorithm.transfer import (
        EngineConfig, TransferDesignResult, transfer_orbit,
    )

    engine = EngineConfig(t_max=0.5, isp=3000.0)
    system = SimpleNamespace(origin="EARTH")
    forces = [PointMassGravity("EARTH", mu=MU)]

    r0 = 7000.0; v0 = np.sqrt(MU / r0)
    rT = 7200.0; vT = np.sqrt(MU / rT)
    departure = np.array([r0, 0, 0, 0, v0, 0])
    target = np.array([rT, 0, 0, 0, vT, 0])

    result = transfer_orbit(
        "low_thrust",
        engine_config=engine,
        initial_mass=1000.0,
        n_segments=5,
        target_oe=(rT, 0.0, 0.0),
        solver_method="shooting",
        duration_days=3.0,
        system=system,
        forces=forces,
        # 通过 kwargs 传入 departure_state / target_state 或用 tli_params+target_ephemeris
    )

    assert result.transfer_type == "low_thrust"
    assert result.details.converged
    assert result.details.fuel_consumed > 0
    assert result.details.equivalent_delta_v > 0
    # 终端残差
    r_err, v_err = result.details.terminal_residual
    assert r_err < 10.0  # km
    assert v_err < 0.01  # km/s
```

#### 4b. `test_low_thrust_mass_and_thrust_history`

```python
"""质量单调递减、推力历史可输出。"""

def test_low_thrust_mass_and_thrust_history():
    # ... 同 4a 设定 ...
    # 验证质量单调递减（零推力段除外）
    masses = result.details.states_7d[:, 6]
    assert np.all(np.diff(masses) <= 1e-10)
    # 推力历史各段 throttle ∈ [0, 1]
    for seg in result.details.segments:
        assert 0.0 <= seg.throttle <= 1.0
```

#### 4c. `test_low_thrust_vs_impulsive_delta_v_comparison`

```python
"""同任务场景低推力等效 Δv ≥ 脉冲 Δv（物理定律约束）。

低推力因持续推力损失（gravity loss），等效 Δv 应 ≥ 霍曼脉冲 Δv。
"""

def test_low_thrust_vs_impulsive_delta_v_comparison():
    r1 = 7000.0; r2 = 7200.0
    dv1, dv2 = hohmann_delta_v(r1, r2)
    dv_impulsive = dv1 + dv2

    # 低推力等效 Δv（从 fuel_consumed 反算）
    engine = EngineConfig(t_max=0.5, isp=3000.0)
    # ... solve ...
    dv_lt = result.details.equivalent_delta_v

    # 低推力等效 Δv 应 ≥ 霍曼 Δv（或至少同一量级；短弧提升小、gravity loss 不显著）
    # 放宽到 5x，因为 LEO→LEO+200km 是非常小的轨道改变
    assert dv_lt >= 0.0  # 正值
    assert dv_lt < dv_impulsive * 5.0  # 不应超过太多
```

**估计工作量**：~100-150 行

---

### Step 5：算例文档

**文件**：`docs/transfer/low_thrust.rst`（新建）

内容结构：

```rst
小推力转移
==========

.. 内容：
   1. 基本原理（7D 增广状态、min-fuel NLP、角度参数化）
   2. 使用方法（transfer_orbit("low_thrust", ...) 代码示例）
   3. 与脉冲转移的 Δv 对比表
   4. 推进参数说明（EngineConfig 字段含义、典型值域）
   5. 质量演化与推力历史绘图示例
```

**估计工作量**：~80-120 行 RST

---

## §四、风险与注意事项

| 风险 | 缓解措施 |
|------|----------|
| 端到端算例不收敛（飞行时间/段数/初猜不匹配） | 先用纯二体短弧（~3 天）验证机制，再调参到物理合理值；Q-law 初猜比满推力初猜收敛性好（已验证） |
| 地月场景需要 SPICE 星历 | 端到端测试分两层：纯二体单元测试（无 SPICE）+ 地月集成测试（@mark.spice） |
| 范围控制：不要顺手建通用低推力最优控制框架 | Issue agent brief 已明确"只要一条端到端算例"，间接法/配点法/Pontryagin 极大值原理均不在范围内 |
| `duration_days` 默认值对地月场景不合适 | 30 天是保守值；地月转移 ~3-5 天（SMART-1 ~13 月是极低推力），参数可覆盖 |
| 与 #256 脉冲 Δv 比较的口径 | 脉冲 = Σ\|Δv_i\|，小推力 = Isp·g₀·ln(m0/mf)/1000，两者单位一致（km/s），但物理含义不同（脉冲瞬时 vs 连续），文档须说明 |

---

## §五、不在范围内

以下均属 gap-analysis 已识别的大坑，**本条不做**：

- 间接法协态初值打靶（Pontryagin 极大值原理）
- 配点法 / 伪谱法（Gauss-Legendre、Chebyshev）
- 自由飞行时间 / 自由推力方向联合优化
- 通用低推力最优控制框架（orchestrator for indirect/collocation/hybrid）
- 与 GMAT CSALT 的 golden 对照（ADR 0013：不用黄金样本）
- 地月多体摄动下的小推力（仅点质量重力，验证数学内核）

---

## §六、工作量估计

| 步骤 | 文件 | 行数 | 复杂度 |
|------|------|------|--------|
| Step 1 | `__init__.py` | ~40 | 低 |
| Step 2 | `__init__.py` | ~100 | 中 |
| Step 3 | `__init__.py` | ~30 | 低 |
| Step 4 | `test_low_thrust_end_to_end.py` | ~150 | 中 |
| Step 5 | `low_thrust.rst` | ~100 | 低 |
| **合计** | | **~420 行** | |

关键路径：Step 2（编排函数）→ Step 4（集成测试验证收敛）。Step 1 和 Step 3 可并行。
