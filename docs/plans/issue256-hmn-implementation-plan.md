# Issue #256 实施方案：HMN 霍曼直接转移（FR3 第一部分）

> 供审查。实施前须修订 issue #256 验收标准（ADR 0013 对齐），见第 0 节。
> 公式依据见第 9 节文献引用表。

---

## 0. 验收标准修订（ADR 0013 对齐）

**原 issue #256 验收标准存在 ADR 0013 硬冲突**：要求 "与 DFH RESULTS_HMN 黄金样本对比，容差进 tests/dfh/ 回归"，而 ADR 0013 明确规定：

> - 不使用黄金样本（golden file）对照
> - DFH 仅作开发期交叉参考，脚本放 `scripts/`，不进 CI
> - 正确性由物理定义裁决——霍曼转移 Δv 匹配理论值

**修订后验收标准：**

1. **端到端 HMN 算例**：TLI 参数（高度/倾角/航迹角）+ 目标标称星历 → 转移星历 + 设计结果汇总（Δv、弹道参数）
2. **物理定义验证**（ADR 0013 第 1 条）：
   - 二体模型下 Δv₁、Δv₂ 与经典霍曼转移理论公式误差 < 1%（解析对照）
   - 能量守恒验证：转移弧段始末比机械能差 < 容差
   - 角动量守恒验证（共面情形）
3. **星历模型收敛**：最终以星历模型打靶收敛解为准，非二体解析解
4. **DFH 交叉参考**（开发期，不进 CI）：与 RESULTS_HMN.TXT 的对比脚本放 `scripts/`，供本地手动诊断

---

## 1. 现状分析

### 1.1 可复用组件（已就位）

| 组件 | 位置 | 状态 |
|------|------|------|
| Lambert 求解器 | `e2m2e/algorithm/transfer/lambert.py` (`solve_lambert`, `solve_lambert_batch`) | ✅ 封装完成，Rust Izzo 2015 内核 |
| 多脉冲框架 | `e2m2e/algorithm/transfer/multi_impulse.py` (`MultiImpulseTransfer`) | ✅ 支持 Lambert 闭合的 N 脉冲优化 |
| 端点条件 | `e2m2e/algorithm/transfer/terminal.py` (`StateTerminal`, `OrbitTerminal`) | ✅ StateTerminal 可表示固定状态 |
| NLP 后端 | `e2m2e/algorithm/transfer/nlp_scipy.py` | ✅ 可用，但当前绑定 DRO-RO 特定优化器 |
| 多重打靶 | `algorithms/multiple_shooting.py` + Rust 编译版 | ✅ 星历模型打靶就位 |
| 星历输出 | `#251` 的 io 模块 | ✅ |
| HMN 结果解析器 | `e2m2e/io/results.py` (`parse_results_hmn`, `HmnResult`) | ✅ 已实现 |
| 编排器占位 | `e2m2e/algorithm/transfer/__init__.py` (`transfer_orbit`) | 🔧 占位（`NotImplementedError`） |

### 1.2 需新增/修改

| 缺失组件 | 说明 |
|----------|------|
| **TLI 参数 → 出发状态构造** | TLI 高度/倾角/航迹角 → ECI 出发状态向量。参照 MATLAB `fmt_inputs_transfer.m` 语义 |
| **地球停泊轨道端点** | 当前 `StateTerminal`/`OrbitTerminal` 不处理地球中心惯性系；需补停泊轨道表示 |
| **transfer_orbit("HMN") 编排体** | 当前是 `NotImplementedError`，需实现完整管线 |
| **TransferDesignResult 扩展** | 当前 `details: dict[str, Any]` 无结构化定义；HMN 需要结构化弹道参数汇总 |
| **HMN 端到端测试** | 物理定义验证（解析对照 + 守恒量） |

---

## 2. 架构设计

### 2.1 数据流

```
TLI 参数 (高度 h, 倾角 i, 航迹角 γ, 历元 t0)
    │
    ▼
[Step 1] construct_departure_state()     ← 新函数
    │  输出: ECI 状态 r0, v0 (km, km/s)
    ▼
[Step 2] Lambert 初猜                    ← 复用 solve_lambert
    │  输入: r0, r_target(t0+tof), tof, μ_earth
    │  输出: v0_Lambert, vf_Lambert
    ▼
[Step 3] 星历模型多重打靶收敛             ← 复用 multiple_shooting + Rust
    │  输入: Lambert 初猜轨迹
    │  输出: 收敛的星历模型转移轨迹
    ▼
[Step 4] 结果汇总                         ← 新函数
    │  输出: TransferDesignResult(transfer_type="HMN", delta_v, trajectory, details)
    │         + EphemerisTable (转移星历)
    ▼
[验证] 物理定义对照                       ← 测试
    - Δv vs 霍曼理论公式
    - 能量/角动量守恒
```

### 2.2 模块组织（遵循 ADR 0011 五层架构 + ADR 0012 依赖方向）

```
e2m2e/algorithm/transfer/
├── __init__.py          # 修改: transfer_orbit() 编排器实现 HMN 分支
├── hohmann.py           # 新增: TLI 构造 + 霍曼转移专用逻辑
├── lambert.py           # 现有，不修改
├── terminal.py          # 现有，可能需补 LEO 端点类
└── ...

tests/transfer/
├── test_hohmann.py      # 新增: HMN 端到端测试（物理定义验证）
└── ...

scripts/
├── dfh_hmn_compare.py   # 新增: DFH RESULTS_HMN 交叉对比（开发期，不进 CI）
```

### 2.3 关键设计决策

| 决策 | 方案 | 理由 |
|------|------|------|
| TLI 构造在 Python 算法层 | ✅ 是 | 需要领域知识（开普勒根数→笛卡尔状态），不属于"喂进数字就迭代"的数值层 |
| 星历打靶用现有 Rust 后端 | ✅ 是 | 复用已有 multiple_shooting + propagate_compiled，不新开 Rust 代码 |
| 二体解析解用于验证 | ✅ 是 | ADR 0013 明确：正确性由物理定义裁决。霍曼转移 Δv 有闭式公式 |
| DFH 对比不进 CI | ✅ 是 | ADR 0013 第 4 条 |
| transfer_orbit 编排器保持 flat 函数 | ✅ 是 | 遵循 design_orbit 的编排器模式（flat function, not class） |

---

## 3. 实施步骤

### Step 1：TLI 参数→出发状态（`hohmann.py`）

**文件：`e2m2e/algorithm/transfer/hohmann.py`（新增）**

#### 1.1 数据类型

```python
@dataclass(frozen=True)
class TliParams:
    """Trans-Lunar Injection 出发参数。

    语义参照 Curtis (2008) Algorithm 4.2 + Lu et al. (2021) Eqs. 1-10。

    Attributes:
        parking_alt_km: 停泊轨道高度 (km)，圆轨道假设 (γ=0)
        inclination_deg: 停泊轨道倾角 (deg)
        flight_path_angle_deg: 航迹角 (deg)，霍曼转移出发条件为 0
        raan_deg: 升交点赤经 (deg)
        arg_perigee_deg: 近地点幅角 (deg)
        epoch: 出发历元（UTC 字符串或 JD_TDB 浮点数）
    """
    parking_alt_km: float
    inclination_deg: float
    flight_path_angle_deg: float = 0.0
    raan_deg: float = 0.0
    arg_perigee_deg: float = 0.0
    epoch: float | str = 0.0
```

#### 1.2 核心算法：开普勒根数→笛卡尔状态

算法来源：Curtis (2008) *Orbital Mechanics for Engineering Students*, Algorithm 4.2。

**输入**：半长轴 `a`、偏心率 `e`、倾角 `i`、升交点赤经 `Ω`、近地点幅角 `ω`、真近点角 `ν`

**Step A**：轨道平面内位置与速度（perifocal frame）

```
p = a(1 - e²)                          # 半通径
r = p / (1 + e·cos ν)                  # 距焦点距离

r_pf = r·[cos ν, sin ν, 0]ᵀ           # perifocal 位置
v_pf = √(μ/p)·[-sin ν, (e+cos ν), 0]ᵀ # perifocal 速度
```

**Step B**：旋转到 ECI（地心惯性系）

```
R = R₃(-Ω)·R₁(-i)·R₃(-ω)

其中：
R₃(θ) = [[ cos θ, sin θ, 0],
          [-sin θ, cos θ, 0],
          [     0,     0, 1]]

R₁(θ) = [[1,     0,     0],
          [0, cos θ, sin θ],
          [0,-sin θ, cos θ]]

r_ECI = R · r_pf
v_ECI = R · v_pf
```

**Step C**：霍曼转移出发的特殊化（γ=0，圆轨道）

当 `flight_path_angle_deg = 0` 时，停泊轨道为圆轨道，`e=0`，`ν=0`：

```
r_park = r_earth + parking_alt_km
v_park = √(μ_earth / r_park)           # 圆轨道速度

r_pf = [r_park, 0, 0]ᵀ
v_pf = [0, v_park, 0]ᵀ
```

再经 Step B 旋转到 ECI。此时 `a = r_park`，`e = 0`，`ω` 和 `M` 无定义（圆轨道），但 `Ω` 和 `i` 仍有效。

**Step D**：霍曼转移 Δv 与速度方向

出发点速度增量（Curtis Ch. 6 / Vallado）：

```
Δv₁ = √(μ/r₁) · (√(2r₂/(r₁+r₂)) - 1)    # 出发点切向加速
Δv₂ = √(μ/r₂) · (1 - √(2r₁/(r₁+r₂)))    # 到达点切向减速
```

其中 `r₁ = r_park`（地球停泊轨道半径），`r₂` = 目标轨道半径（月球附近）。

#### 1.3 完整函数签名

```python
def keplerian_to_cartesian(
    a_km: float, e: float, i_deg: float,
    omega_deg: float, raan_deg: float, nu_deg: float,
    mu: float = 398600.4418,
) -> tuple[np.ndarray, np.ndarray]:
    """Curtis (2008) Algorithm 4.2: 开普勒根数 → ECI 笛卡尔状态。

    Args:
        a_km: 半长轴 (km)
        e: 偏心率
        i_deg: 倾角 (deg)
        omega_deg: 近地点幅角 (deg)
        raan_deg: 升交点赤经 (deg)
        nu_deg: 真近点角 (deg)
        mu: 引力参数 (km³/s²)

    Returns:
        (r_eci, v_eci) in (km, km/s)，shape 均为 (3,)
    """

def construct_departure_state(
    params: TliParams,
    mu_earth: float = 398600.4418,
    r_earth: float = 6378.137,
) -> tuple[np.ndarray, np.ndarray]:
    """TLI 参数 → ECI 出发状态向量。

    简化路径（γ=0，圆停泊轨道）：
    1. r_park = r_earth + parking_alt_km
    2. v_park = √(mu_earth / r_park)
    3. r_pf = [r_park, 0, 0]ᵀ;  v_pf = [0, v_park, 0]ᵀ
    4. R = R₃(-Ω)·R₁(-i)·R₃(-ω)
    5. r_eci = R·r_pf;  v_eci = R·v_pf

    非零航迹角路径（γ≠0，椭圆停泊轨道）：
    调用 keplerian_to_cartesian(a, e, i, ω, Ω, ν=0) 其中：
    a = r_park / (1 - e), e 由航迹角推导。
    """
```

**关键约束**：航迹角 (flight path angle) 的语义**必须照 MATLAB `fmt_inputs_transfer.m` 实现**。
issue brief 明确警告 "别自己发明"。若 MATLAB 文件不可得（外部，不在 repo）：
- 零航迹角路径有闭式解（圆轨道，上述 Step C），是霍曼转移的标准出发条件
- 非零航迹角参 Curtis (2008) Eq. 2.132: `tan γ = (e·sin ν) / (1 + e·cos ν)` 反解 `e`

**估计工作量**：~100 行 Python。

### Step 2：transfer_orbit("HMN") 编排器

**文件：`e2m2e/algorithm/transfer/__init__.py`（修改）**

#### 2.1 编排器整体流程

```python
def transfer_orbit(
    transfer_type: str,
    *,
    target_ephemeris: Any = None,
    tli_params: TliParams | None = None,
    tof_range: tuple[float, float] | None = None,
    **kwargs,
) -> TransferDesignResult:
    if transfer_type != "HMN":
        raise NotImplementedError(f"transfer_orbit('{transfer_type}') 实现未完成")

    # Step 1: TLI → 出发状态（ECI）
    r0, v0_park = construct_departure_state(tli_params)

    # Step 2: 目标轨道 → 到达位置（ECI）
    # 从目标标称星历插值获取到达位置
    t_arrival = tli_params.epoch + tof
    r_target = interpolate_target(target_ephemeris, t_arrival)

    # Step 3: Lambert 初猜（二体模型）
    sol = solve_lambert(r0, r_target, tof, mu_earth, direction="short")
    # → v0_lambert: 出发速度, vf_lambert: 到达速度

    # Step 4: 出发 Δv
    delta_v1 = norm(sol.v0 - v0_park)  # km/s

    # Step 5: 到达 Δv
    # 到达速度需匹配目标轨道速度
    v_target = interpolate_target_velocity(target_ephemeris, t_arrival)
    delta_v2 = norm(sol.vf - v_target)  # km/s

    # Step 6: 星历模型打靶收敛（可选，视精度需求）
    # 将 Lambert 二体解作为初猜，传入星历模型多重打靶
    # ephem_trajectory = ephemeris_shooting(r0, sol.v0, tof, ...)
    ephem_trajectory = None  # 初版可跳过星历打靶，仅用 Lambert

    # Step 7: 结果汇总
    return TransferDesignResult(
        transfer_type="HMN",
        delta_v=delta_v1 + delta_v2,
        trajectory=ephem_trajectory or make_lambert_trajectory(sol, tof),
        details=HmnTransferDetails(...),
    )
```

#### 2.2 关键子问题与文献公式

**子问题 A：目标星历到达位置插值**

输入 `target_ephemeris`（FR1 产物）可能是 `NominalOrbit` 或 `EphemerisTable`。

- `NominalOrbit` 有 `epochs` + `states` 数组；`state_at(t)` 当前是 stub
- 需实现 Lagrange 插值（r=5-6 阶），参照 Gomez (2001) Vol. I §8.2.3 的 NominalOrbit 契约

若 `NominalOrbit.state_at()` 不可用，退路：直接用 numpy 线性插值（精度足够用于 Lambert 初猜）。

**子问题 B：飞行时间 tof 选择**

文献参考（Zhang et al. 2023）：地月直接转移典型飞行时间 3-6 天，最省能约 4.5 天。

| 策略 | 说明 | 适用 |
|------|------|------|
| 固定中值 | `tof = (tof_min + tof_max) / 2` | 初版最小闭环 |
| Δv 最小扫描 | `solve_lambert_batch` over tof grid, 选 min(Δv₁+Δv₂) | 正式版 |
| 共轭点法 | Lawden (1962) 最优转移时间条件 | 后续迭代 |

推荐：初版用固定中值（4.5 天默认），后续迭代到批量扫描。

**子问题 C：星历模型打靶**

Liu et al. (2008) 提出三阶段收敛策略（与本方案一致）：

1. 解析初猜（Patched Conic / Lambert 二体解）
2. 简单力模型数值修正（地球 J2 + 月球三体）
3. 高精度星历模型最终修正（DE430 + 高阶引力 + 太阳引力 + 光压）

微分修正方程（Liu et al. 2008 Eq. 5）：

```
X^(k+1) = X^k + (A^T A)^{-1} A^T [Y - F(X^k)]
```

其中 `A` 为雅可比矩阵，通过有限差分（步长 1-2%）计算（Eq. 6）。

初版可仅实现阶段 1（Lambert 二体解），星历打靶作为后续增强。

**子问题 D：坐标系**

| 阶段 | 坐标系 | 说明 |
|------|--------|------|
| TLI 构造 | ECI (J2000) | 地心惯性系 |
| Lambert 求解 | ECI (J2000) | 位置/速度在同一天球参考系 |
| 目标星历 | GCRS → ECI | ADR 0010: GCRS ≈ ICRF，偏差 ~20mas |
| 星历打靶 | ET(TDB) | ADR 0010: 统一动力学时间 |

GCRS ↔ ECI 的差异（~20mas 帧偏差）对 Lambert 初猜精度无实质影响，初版可忽略。

**关键问题需决定**：

1. **目标星历格式**：确认 `NominalOrbit.state_at()` 状态；若未实现，HMN 自行做 Lagrange 插值。
2. **初版是否跳过星历打靶**：建议初版仅用 Lambert 二体解 + 物理验证，星历打靶作后续 PR。
3. **TransferType 枚举**：现有 `TransferType.DIRECT` 是否对应 HMN？若是，编排器用枚举而非字符串。

**估计工作量**：~150 行 Python 编排逻辑（含插值和结果格式化）。

### Step 3：结果汇总与弹道参数

**扩展 `TransferDesignResult.details` 结构**：

```python
@dataclass
class HmnTransferDetails:
    """HMN 转移设计详细结果。"""
    tli_epoch: float           # TLI 历元
    noi_epoch: float           # 近月插入历元
    tof_day: float             # 飞行时间 (天)
    tli_state: np.ndarray      # TLI 状态 (6,)
    noi_state: np.ndarray      # NOI 状态 (6,)
    tli_elements: np.ndarray   # TLI 开普勒根数 (6,) [a,e,i,Ω,ω,M]
    noi_elements: np.ndarray   # NOI 开普勒根数 (6,)
    delta_v1: float            # 出发 Δv (km/s)
    delta_v2: float            # 到达 Δv (km/s)
    converged: bool            # 打靶是否收敛
    n_iterations: int          # 打靶迭代次数
```

此结构与现有 `HmnResult`（io/results.py 解析器输出）对称，但这是**自产**结果而非 DFH 输入的解析。

**估计工作量**：~40 行。

### Step 4：端到端测试

**文件：`tests/transfer/test_hohmann.py`（新增）**

测试策略遵循 ADR 0013：按物理定义验证，不用黄金样本。

#### 4.1 验证公式完备性

| 验证项 | 公式 | 文献来源 | 容差 |
|--------|------|---------|------|
| 圆轨道速度 | `v = √(μ/r_park)` | Curtis (2008) §2.3 | < 0.01% |
| 霍曼 Δv₁ | `Δv₁ = √(μ/r₁)·(√(2r₂/(r₁+r₂)) - 1)` | Curtis (2008) §6.1 / Vallado | < 1% |
| 霍曼 Δv₂ | `Δv₂ = √(μ/r₂)·(1 - √(2r₁/(r₁+r₂)))` | 同上 | < 1% |
| 比机械能守恒 | `ε = v²/2 - μ/r = const` | 轨道力学定义 | < 1e-6 km²/s² |
| 比角动量守恒（共面） | `h = |r × v| = const` | 轨道力学定义 | < 1e-3 km²/s |
| 转移半长轴 | `a_t = (r₁+r₂)/2` | 霍曼转移定义 | < 0.1% |
| 飞行时间 | `TOF = π·√(a_t³/μ)` | 半椭圆周期 | < 1% |

#### 4.2 典型测试参数

参考 Chen et al. (2023) Table 1 数值和 Zhang et al. (2023) 综述：

```python
# 测试用例：LEO 200km → 月球附近
MU_EARTH = 398600.4418   # km³/s²
R_EARTH = 6378.137       # km
R_MOON_ORBIT = 384405.0  # km（地月平均距离，Cui et al. 2025）

r1 = R_EARTH + 200.0     # = 6578.137 km
r2 = R_MOON_ORBIT        # = 384405.0 km

# 理论 Δv
dv1_theory = sqrt(MU_EARTH/r1) * (sqrt(2*r2/(r1+r2)) - 1)  # ≈ 3.13 km/s
dv2_theory = sqrt(MU_EARTH/r2) * (1 - sqrt(2*r1/(r1+r2)))   # ≈ 0.83 km/s

# 理论飞行时间
a_t = (r1 + r2) / 2
tof_theory = pi * sqrt(a_t**3 / MU_EARTH)  # ≈ 4.43 天

# 理论转移轨道能量
energy_departure = -MU_EARTH / (2 * a_t)  # 比机械能
```

#### 4.3 测试函数

```python
class TestTliStateConstruction:
    """TLI 参数→出发状态 单元测试。"""

    def test_circular_orbit_speed(self):
        """圆轨道速度 = √(μ/r_park)，误差 < 0.01%。"""

    def test_rotation_matrix_orthogonal(self):
        """R₃(-Ω)·R₁(-i)·R₃(-ω) 为正交矩阵，det(R) = 1。"""

    def test_inclination_effect(self):
        """倾角 = 90° 时，速度 z 分量最大；倾角 = 0° 时 z 分量为 0。"""


class TestHmnTransfer:
    """HMN 霍曼直接转移端到端测试。"""

    def test_hohmann_dv_matches_analytical(self):
        """二体模型下 Δv₁、Δv₂ 与经典公式误差 < 1%。

        验证方法（ADR 0013 第 1 条）：
        Δv₁ = √(μ/r₁)·(√(2r₂/(r₁+r₂)) - 1)
        Δv₂ = √(μ/r₂)·(1 - √(2r₁/(r₁+r₂)))
        """

    def test_transfer_semi_major_axis(self):
        """转移轨道半长轴 a_t = (r₁+r₂)/2，误差 < 0.1%。"""

    def test_flight_time_matches_half_period(self):
        """飞行时间 = π·√(a_t³/μ)，误差 < 1%。"""

    def test_energy_conservation(self):
        """转移弧段始末比机械能 ε = v²/2 - μ/r 守恒，差 < 1e-6 km²/s²。"""

    def test_angular_momentum_conservation(self):
        """共面转移比角动量 h = |r×v| 守恒，差 < 1e-3 km²/s。"""

    def test_end_to_end_hmn(self):
        """完整 HMN 算例：TliParams + 目标星历 → TransferDesignResult。

        验证：
        - 返回 TransferDesignResult，transfer_type == "HMN"
        - delta_v₁ ≈ 3.13 km/s (LEO 200km → 月球)
        - delta_v₂ ≈ 0.83 km/s
        - TOF ≈ 4.4 天
        - trajectory 非空
        - details 包含所有 HmnTransferDetails 字段
        """
```

**估计工作量**：~200 行测试代码（比原计划多 ~50 行，因新增半长轴、飞行时间、角动量验证）。

### Step 5（可选）：DFH 交叉参考脚本

**文件：`scripts/dfh_hmn_compare.py`（新增，不进 CI）**

```python
"""DFH RESULTS_HMN 交叉对比脚本（开发期诊断，不进 CI）。

用法：python scripts/dfh_hmn_compare.py

参照 ADR 0013 第 4 条：
- 脚本放 scripts/，不进 CI、不进发布包
- 用于诊断量级/系统性偏差
"""
from e2m2e.io.results import read_results_hmn
from e2m2e.algorithm.transfer import transfer_orbit, TliParams

def compare():
    golden = read_results_hmn("tests/io/fixtures/RESULTS_HMN.TXT")
    result = transfer_orbit("HMN", tli_params=..., target_ephemeris=...)
    # 输出对比表：Δv 偏差、TOF 偏差、弹道参数偏差
    ...

if __name__ == "__main__":
    compare()
```

**注意**：此脚本需要能从 `RESULTS_HMN.TXT` 反推 `TliParams` 输入参数（TLI 高度/倾角/航迹角），这可能需要从 TLI 状态向量反算开普勒根数。

---

## 4. 风险与缓解

| 风险 | 等级 | 缓解 | 文献依据 |
|------|------|------|---------|
| MATLAB `fmt_inputs_transfer.m` 不在 repo | 中 | 零航迹角路径有闭式解（圆轨道），非零航迹角参 Curtis Eq. 2.132 | Curtis (2008) |
| `NominalOrbit.state_at()` 未实现 | 中 | 退路：numpy 线性/Lagrange 插值，精度对 Lambert 初猜足够 | Gomez (2001) §8.2.3 |
| 星历打靶接受 Lambert 初猜的接口未验证 | 中 | 初版可跳过星历打靶，仅用 Lambert 二体解验证 | Liu et al. (2008) |
| 坐标系混用（ECI vs GCRS vs synodic） | 低 | GCRS ≈ ICRF 偏差 ~20mas，对 Lambert 无实质影响；明确标注各阶段坐标系 | ADR 0010 / Soffel |
| 霍曼转移严格共面假设 vs 真实地月转移的非共面性 | 低 | 初版接受小倾角差；后续引入 Lambert 批量扫描 | Zhang et al. (2023) |
| 目标标称轨道历元与 TLI 历元的时间对齐 | 低 | 从目标星历插值，不强求目标轨道周期整数倍 | — |

---

## 5. 工作量估计

| Step | 新增/修改代码 | 估计行数 |
|------|-------------|---------|
| Step 1: TLI 构造 | `hohmann.py` 新增（含 keplerian_to_cartesian） | ~100 行 |
| Step 2: 编排器 | `__init__.py` 修改 | ~150 行 |
| Step 3: 结果汇总 | `hohmann.py` 或 `__init__.py` | ~40 行 |
| Step 4: 测试 | `test_hohmann.py` 新增 | ~200 行 |
| Step 5: DFH 脚本 | `scripts/` 新增 | ~60 行 |
| **合计** | | **~550 行** |

---

## 6. 实施顺序与依赖

```
Step 1 (hohmann.py: TliParams + construct_departure_state)
    │
    ├── Step 4a (test_tli_state_construction — 可立即写，TDD RED)
    │
    ▼
Step 2 (编排器: transfer_orbit("HMN"))
    │
    ├── Step 4b (test_end_to_end_hmn — 端到端)
    │
    ▼
Step 3 (结果汇总: HmnTransferDetails)
    │
    ▼
Step 4c (test_hohmann_dv_matches_analytical + test_energy_conservation)
    │
    ▼
Step 5 (DFH 交叉参考，可选，不阻塞 PR)
```

Step 1-2 是核心路径。Step 5 不阻塞合并。

---

## 7. 与现有 issue/PR 的关系

| Issue/PR | 关系 |
|----------|------|
| #254 (FR1) | ✅ CLOSED，其产物 `NominalOrbit` 是 HMN 的目标输入 |
| #251 (io) | ✅ 已完成，EphemerisTable 输出格式可用 |
| #280 (TIGHT/SPECIAL) | 无直接关系，可并行开发 |
| #261 (角动量管理) | 无直接关系，可并行开发 |
| lambert.rs 工作区 | ⚠️ brief 提示 "未提交状态"，实施前需确认是否已合入 master |

---

## 8. 待确认事项（实施前需要决定）

1. **TLI 航迹角语义**：MATLAB `fmt_inputs_transfer.m` 不在 repo 中。是否能获取？若不能，以 Vallado 公式为准。
2. **目标星历插值**：`NominalOrbit.state_at()` 是否已实现？若未实现，HMN 需要自己从 `epochs`/`states` 做线性/Lagrange 插值。
3. **星历打靶初猜格式**：`multiple_shooting` 接受什么格式的初始猜测？需要先确认接口。
4. **TransferType 枚举**：现有 `TransferType.DIRECT` 是否对应 HMN？若是，transfer_orbit 编排器应使用该枚举而非字符串 `"HMN"`。
5. **tof 选择策略**：初版用中值 vs Lambert 批量扫描？建议先中值。

---

## 9. 文献引用表

### 直接引用（方案公式来源）

| 编号 | 文献 | 方案中引用位置 | 关键内容 |
|------|------|---------------|---------|
| [C08] | Curtis, H. (2008). *Orbital Mechanics for Engineering Students*. Butterworth-Heinemann. | Step 1 §1.2, Step 4 §4.1 | Algorithm 4.2（开普勒→笛卡尔）；§6.1（霍曼转移 Δv 公式）；§2.3（圆轨道速度） |
| [Vallado] | Vallado, D. A. (2013). *Fundamentals of Astrodynamics and Applications*. 4th ed. Microcosm Press. | Step 1 §1.2, ADR 0013 | 第 2 章（通用椭圆轨道状态构造）；霍曼转移公式；航迹角定义 |
| [Liu08] | 刘磊等 (2008). 多约束条件下的地月转移轨道设计. *宇航学报*. | Step 2 §2.2 子问题 C | 三阶段收敛策略；微分修正方程 (Eq. 5)；雅可比矩阵有限差分 (Eq. 6) |
| [Lu21] | 陆林等 (2021). 载人月球极地探测地月转移轨道设计. *中国科学*. | Step 1 §1.2 | 近月点状态构造 (Eqs. 1-4)；坐标旋转到 J2000 (Eqs. 5-10) |
| [Zhang23] | Zhang et al. (2023). Overview of Earth-moon transfer trajectory modeling and design. *Astrodynamics*. | Step 2 §2.2 子问题 B, 风险表 | 转移方法分类；典型飞行时间 3-6 天；Δv 范围 3.5-4 km/s；CRTBP 方程 (Eqs. 2-5) |
| [Chen23] | 陈天冀等 (2023). 考虑环月交会约束的地月转移轨道设计. *宇航学报*. | Step 4 §4.2 | TLI 数值算例（Table 1）；Δv ≈ 845 m/s（NOI 部分）；4.8 天飞行时间 |
| [Izzo15] | Izzo, D. (2015). Revisiting Lambert's problem. *Celestial Mechanics and Dynamical Astronomy*. | 已有实现 | Lambert 求解器算法（已实现在 `lambert.rs`） |
| [Peng16] | 彭祺擘, 张海联 (2016). 载人登月地月转移轨道方案综述. *载人航天*. | 风险表 | 短程/长程到达模式；能量消耗对比；出发约束 |
| [Lawden62] | Lawden, D. F. (1962). Impulsive transfer between elliptical orbits. | 后续迭代 | 最优两脉冲转移条件；切向推力原则 |
| [Prussing19] | Prussing, J. E. (2019). *Optimal Spacecraft Trajectories*. SIAM. | 已有多脉冲实现 | Primer vector 理论（Ch. 3-5）；Lawden 必要条件 |

### 已有代码中的文献引用（本方案复用）

| 文献 | 代码位置 | 复用组件 |
|------|---------|---------|
| Izzo (2015) | `lambert.rs`, `lambert.py` | Lambert 求解器 |
| Cui et al. (2025) | `transfer_search.py`, `transfer_optimization.py` | DRO-RO 转移框架；地月常数 |
| Prussing (2019) | `multi_impulse.py` | 多脉冲优化框架 |
| Battin | `ephemeris_dynamics.py`, `lambert.rs` | N 体动力学；Lambert 超几何级数 |
| Gomez (2001) | `architecture-design-discussion.md` | NominalOrbit 契约；打靶收敛参数 |
| Soffel | ADR 0010 | 时间尺度（TDB）；参考系转换 |

### 文献库目录（供后续深入参考）

完整文献库位于 `C:\baidunetdiskdownload\地月空间相关md\output\`（354 篇），与本方案直接相关的子集：

- `1962 - Impulsive transfer between elliptical orbits/` — Lawden 经典脉冲转移理论
- `Zhang 等 - 2023 - Overview of Earth-moon transfer trajectory modeling and design/` — 地月转移方法综述
- `彭祺擘和张海联 - 2016 - 载人登月地月转移轨道方案综述/` — 霍曼转移参数与约束
- `刘磊 等 - 2008 - 多约束条件下的地月转移轨道设计/` — 微分修正方法
- `陆林 等 - 2021 - 载人月球极地探测地月转移轨道设计/` — TLI 状态构造公式
- `陈天冀 等 - 2023 - 考虑环月交会约束的地月转移轨道设计/` — 数值算例
- `Shi 等 - 2025 - Review of cislunar space transfer trajectory design based on impulsive/` — 脉冲转移综述
- `Curtis - 2008 - Orbital mechanics for engineering students/` — 标准教科书（Algorithm 4.2, 5.2）
