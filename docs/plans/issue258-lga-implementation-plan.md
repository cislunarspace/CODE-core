# Issue #258 实施方案：LGA 月球引力辅助间接转移（FR3 第二部分）

> 供审查。验收标准已按 ADR 0013 修订（第 0 节）。
> 公式依据见第 9 节文献引用表。

---

## 0. 验收标准修订（ADR 0013 对齐）

**原 issue #258 验收标准存在 ADR 0013 硬冲突**：要求"与 DFH RESULTS_LGA 黄金样本对比，容差进 tests/dfh/ 回归"，而 ADR 0013 明确规定：

> - 不使用黄金样本（golden file）对照
> - DFH 仅作开发期交叉参考，脚本放 `scripts/`，不进 CI
> - 正确性由物理定义裁决

**修订后验收标准：**

1. **端到端 LGA 算例**：TLI 参数 + 目标星历 → 转移星历 + 设计结果汇总（各段 Δv、弹道参数、近月点高度/速度）
2. **物理定义验证**（ADR 0013 第 1 条）：
   - 近月飞越后能量变化量与引力辅助理论预测一致：双曲线超速 v∞ 不变（进出 SOI 速度模相等）、近月点速度 > 月球逃逸速度
   - 轨道连续性：三段弹道在拼接点位置/速度无跳变（残差 < 1e-6 无量纲）
   - Jacobi 常数在 CR3BP 模型下守恒（飞越前后差 < 1e-8 无量纲）
   - 近月点检测精度：半径极值与 Brent 法根定位一致（残差 < 1e-6 DU）
3. **飞越段轨道连续且精度满足回归容差**：事件定位方案（Python scipy.brentq 根定位 + PoincareSection.periapsis 复用）在实施说明中记录
4. **DFH 交叉参考**（开发期，不进 CI）：与 RESULTS_LGA.TXT 的对比脚本放 `scripts/`

---

## 1. 现状分析

### 1.1 LGA 物理模型概述

月球引力辅助（Lunar Gravity Assist）转移的核心是利用航天器飞越月球时的引力效应改变速度方向和大小（**无动力 LGA**，不消耗燃料在飞越段），从而以更低的总 Δv 到达目标轨道（Shi et al. 2025；Qi & Xu 2015）。总 Δv 仅来自出发段（TLI）和到达段（LOI）的脉冲，飞越段 Δv = 0。典型 TOF 15-45 天，总 Δv ~250 m/s（单次 LGA）或 ~220 m/s（双次 LGA）。搜索空间为 **出发窗口（TLI 历元相位）× 飞行时间（TOF）**；近月点高度由传播自然决定，不作为独立搜索变量（Parker & Anderson 2014 §3.4）。

CR3BP 框架下的 LGA 搜索简化为：

```
出发段：地球停泊轨道 → 月球影响球（SOI）入口
    │  用 CR3BP 前向传播 + Lambert 初猜
    ▼
飞越段：月球 SOI 内双曲线飞越
    │  以月球为中心的二体双曲线几何
    │  近月点高度 perilune_h 和 B-plane 角 θ_B 参数化
    ▼
到达段：月球 SOI 出口 → 目标轨道
    │  CR3BP 后续传播 + 打靶闭合
    ▼
筛选：总 Δv 最小的可行解
```

### 1.2 可复用组件（已就位）

| 组件 | 位置 | 状态 | LGA 复用方式 |
|------|------|------|-------------|
| **TLI 出发构造** | `hohmann.py` (TliParams, construct_departure_state) | ✅ | 完全复用 |
| **Lambert 求解器** | `lambert.py` (solve_lambert, solve_lambert_batch) | ✅ | 完全复用，CR3BP 初猜 |
| **ThreeBodyLambert BVP** | `three_body_lambert.py` | ✅ | 完全复用，到达段打靶闭合 |
| **CR3BP 动力学** | `dynamics/` (CR3BP_System, CR3BP_Dynamics) | ✅ | 完全复用 |
| **庞加莱截面** | `manifold/sections.py` (PoincareSection, detect_crossings) | ✅ | 完全复用——近月点检测（r·v=0）已实现且含 Brent 求精 |
| **流形拼接框架** | `low_energy.py` (patch_manifolds, design_low_energy_transfer) | ⚠️ | 架构参考——LGA 搜索空间不同（出发窗口×TOF，非流形管），但 PatchCandidate 数据结构和排序逻辑可复用 |
| **TransferSearch 搜索框架** | `transfer_search.py` + `search_parallel.py` | ⚠️ | 架构参考——DRO-RO 专用搜索的编排模式（网格 → 前向积分 → 可行性筛选）可参考 |
| **TransferType.LGA 枚举** | `data/templates/enums.py` | ✅ | 已定义 |
| **TransferDesignResult** | `__init__.py` | ✅ | 扩展 LGA 分支 |
| **ephemeris_shoot_transfer** | `hohmann.py` | ✅ | 星历打靶修正（Phase 3 增强） |

### 1.3 需新增

| 缺失组件 | 说明 |
|----------|------|
| **lga.py（核心）** | LGA 弹道搜索算法：出发窗口×TOF 网格扫描 + 近月飞越处理 + CR3BP 传播 |
| **LgaTransferDetails** | LGA 转移结果数据结构 |
| **transfer_orbit("LGA") 分支** | 编排器扩展 |
| **test_lga.py** | 物理定义验证 + 端到端测试 |
| **scripts/dfh_lga_compare.py** | DFH 交叉参考脚本（不进 CI） |

### 1.4 事件检测现状评估

**结论：无需 Rust 化，现有 Python 基础设施足够。**

`manifold/sections.py` 的 `PoincareSection.periapsis(center="moon", system)` 已实现：
- 截面函数 `s(state) = r · v`（相对月球的位置·速度点积），零点即近月点
- `detect_crossings()` 符号变化检测 + `_refine_crossing()` Brent 法求精（xtol=1e-14）
- 残差可达 1e-10 以下

LGA 近月点检测直接复用此机制。SOI 边界在 patched conic 方法中用 Laplace SOI（r ≈ 66,300 km，Zhang et al. 2023 Eq. 1），但方案采用 CR3BP 直接传播，不手动切换 SOI——引力效应由 CR3BP 动力学自然包含。SOI 概念仅用于物理验证（验收标准中"进出 SOI 速度模相等"）。

---

## 2. 架构设计

### 2.1 数据流

```
TLI 参数 (h, i, γ, epoch) + 目标轨道 + 搜索参数
    │
    ▼
[Step 1] construct_departure_state()              ← 复用 hohmann.py
    │  输出: ECI 出发态 r0, v0 (km, km/s)
    ▼
[Step 2] 无量纲化 → CR3BP 出发态                 ← 复用 CR3BP_System
    │  输入: r0, v0 → x0 (无量纲)
    ▼
[Step 3] LGA 弹道网格搜索                          ← 新: lga.py
    │  for each (departure_phase, tof):
    │    3a. CR3BP 前向传播到近月点截面
    │    3b. 近月点状态提取（Brent 求精）
    │    3c. 飞越后速度（双曲线几何 or 直接传播穿越月球 SOI）
    │    3d. CR3BP 后续传播到目标位置
    │    3e. 计算总 Δv = |Δv_dep| + |Δv_arr|
    │  输出: 候选解列表 sorted by Δv
    ▼
[Step 4] 最优候选 → ThreeBodyLambert 打靶精化     ← 复用 three_body_lambert.py
    │  输入: 近月点状态 + 目标终端
    │  输出: 收敛的三段弧转移轨迹
    ▼
[Step 5] 结果汇总                                  ← 新: lga.py
    │  输出: TransferDesignResult(transfer_type="LGA", ...)
    ▼
[验证] 物理定义对照                                ← test_lga.py
```

### 2.2 模块组织

```
e2m2e/algorithm/transfer/
├── __init__.py          # 修改: transfer_orbit() 加 LGA 分支
├── lga.py               # 新增: LGA 弹道搜索 + 近月飞越处理
├── hohmann.py           # 现有，不修改
├── three_body_lambert.py# 现有，不修改
├── low_energy.py        # 现有，不修改（架构参考）
└── ...

tests/transfer/
├── test_lga.py          # 新增: LGA 端到端测试（物理定义验证）

scripts/
├── dfh_lga_compare.py   # 新增: DFH RESULTS_LGA 交叉对比（不进 CI）
```

### 2.3 关键设计决策

| 决策 | 方案 | 理由 |
|------|------|------|
| 搜索在 CR3BP 框架内 | ✅ 是 | 与 TransferSearch、ThreeBodyLambert 统一；星历修正作后续增强 |
| 近月飞越用传播穿越（非双曲线拼接） | ✅ 是 | CR3BP 传播自然包含月球引力效应，无需手动切换二体模型；近月点检测用 PoincareSection.periapsis("moon") |
| 出发窗口参数化用相位角 | ✅ 是 | departure_phase ∈ [0, 2π)，与现有 TransferSearch 的 α 参数精神一致 |
| 事件检测用 Python | ✅ 是 | PoincareSection + Brent 已满足精度（1e-10），无需 Rust 化（拆独立 issue） |
| ThreeBodyLambert 打靶用于精化 | ✅ 是 | 最优候选的出发/到达速度作为 ThreeBodyLambert 的初猜，在 CR3BP 下 Newton 迭代精化 |
| 与 #259（WSB）共享搜索底座 | 延迟共享 | 先在 lga.py 内实现完整搜索；#259 实施时再审视抽取共享层 |

---

## 3. 实施步骤

### Step 1：LGA 搜索参数与数据结构（`lga.py`）

**文件：`e2m2e/algorithm/transfer/lga.py`（新增）**

#### 1.1 搜索配置

```python
@dataclass(frozen=True)
class LgaSearchParams:
    """LGA 弹道搜索参数。

    搜索空间：出发相位角 × 飞行时间（TOF）。
    近月点高度由传播自然决定，不作为独立搜索变量，
    仅用于筛选可行候选（Parker & Anderson 2014 §3.4）。

    Attributes:
        departure_phase_range: 出发相位角范围 (min, max)，弧度，[0, 2π)
        n_departure_phase: 出发相位角网格点数
        tof_range: 飞行时间范围 (min, max)，天（LGA 典型 15-45 天，Shi et al. 2025）
        n_tof: TOF 网格点数
        perilune_alt_min: 近月点高度下限 (km)，低于此值的候选丢弃（避免撞击月面）
        perilune_alt_max: 近月点高度上限 (km)，高于此值的候选丢弃（飞越不够近）
        max_total_dv: 最大总 Δv 筛选阈值，km/s（超过的候选丢弃）
    """
    departure_phase_range: tuple[float, float] = (0.0, 2.0 * math.pi)
    n_departure_phase: int = 50
    tof_range: tuple[float, float] = (15.0, 45.0)  # 天（Shi et al. 2025 Table 1）
    n_tof: int = 50
    perilune_alt_min: float = 100.0   # km（避免撞击）
    perilune_alt_max: float = 10000.0 # km（飞越有效性）
    max_total_dv: float = 5.0  # km/s
```

#### 1.2 LGA 搜索结果

```python
@dataclass
class LgaCandidate:
    """单个 LGA 候选解（无动力 LGA）。

    飞越段 Δv = 0（仅利用月球引力），总 Δv = 出发脉冲 + 到达脉冲。
    """
    departure_phase: float      # 出发相位角 (rad)
    tof_sec: float              # 飞行时间 (s)
    departure_state: np.ndarray # 出发态 (6,) km, km/s
    perilune_state: np.ndarray  # 近月点态 (6,) 相对月心，km, km/s
    perilune_alt_km: float      # 近月点高度 (km)
    arrival_state: np.ndarray   # 到达态 (6,) km, km/s
    dv_departure: float         # 出发脉冲 (km/s)
    dv_arrival: float           # 到达脉冲 (km/s)
    total_dv: float             # 总 Δv = dv_departure + dv_arrival (km/s)
    jacobi_departure: float     # 出发侧 Jacobi 常数
    jacobi_arrival: float       # 到达侧 Jacobi 常数
    converged: bool             # ThreeBodyLambert 打靶是否收敛
```

**估计工作量**：~60 行。

### Step 2：CR3BP 弹道传播 + 近月点检测

**文件：`e2m2e/algorithm/transfer/lga.py`**

#### 2.1 核心搜索函数

算法来源：Parker & Anderson (2014) *Low-Energy Lunar Trajectory Design* §4.3 + Cui et al. (2025) 搜索编排。

```python
def search_lga_trajectories(
    departure_state: np.ndarray,
    target_state: np.ndarray,
    system: CR3BP_System,
    dynamics: CR3BP_Dynamics,
    params: LgaSearchParams = LgaSearchParams(),
) -> list[LgaCandidate]:
    """LGA 弹道网格搜索。

    对每个 (departure_phase, tof) 组合：
    1. 从出发态按 phase 偏移出发时刻（CR3BP 旋转系中的时间偏移）
    2. CR3BP 前向传播，用 PoincareSection.periapsis("moon") 检测近月点
    3. 近月点高度在 perilune_alt_range 内的保留
    4. 继续传播到 tof 结束，计算到达位置与目标的距离
    5. Δv_dep = |v_departure - v_parking|，Δv_arr = |v_arrival - v_target|
    6. 总 Δv < max_total_dv 的保留为候选

    Args:
        departure_state: CR3BP 无量纲出发态 (6,)
        target_state: CR3BP 无量纲目标态 (6,)
        system: CR3BP 系统
        dynamics: CR3BP 动力学
        params: 搜索参数

    Returns:
        按 total_dv 升序排列的候选列表
    """
```

#### 2.2 近月点检测复用

```python
from ..manifold.sections import PoincareSection, detect_crossings

# 近月点截面：r·v = 0（相对月心位置与速度的点积为零）
periapsis_section = PoincareSection.periapsis("moon", system)

# 传播完成后用 detect_crossings 提取近月点
crossings = detect_crossings(times, states, periapsis_section)
if crossings:
    t_peri, state_peri, _ = crossings[0]  # 首次近月点
    r_peri = np.linalg.norm(state_peri[:3] - moon_center)
    alt_peri = r_peri * DU - R_MOON  # 换算为 km
```

**估计工作量**：~150 行。

### Step 3：飞越段处理

**文件：`e2m2e/algorithm/transfer/lga.py`**

#### 3.1 方案选择：直接传播穿越（非双曲线拼接）

**理由**：CR3BP 动力学自然包含月球引力效应——航天器接近月球时自动受到月球引力加速/减速。无需手动切换到以月球为中心的二体双曲线模型。这与 `TransferSearch` 的前向积分模式一致。

直接传播的优势：
- 无需 SOI 边界匹配（CR3BP 无 SOI 概念，引力连续变化）
- 无需 B-plane 参数化（近月点高度由传播自然决定）
- 代码更简单，与现有搜索框架一致

#### 3.2 近月段加密采样

```python
def _propagate_with_periapsis_refinement(
    dynamics: CR3BP_Dynamics,
    x0: np.ndarray,
    t_span: tuple[float, float],
    section: PoincareSection,
    n_samples: int = 500,
) -> dict:
    """传播并在近月点附近加密采样。

    1. 粗传播（n_samples/5 步）定位近月点区间
    2. 近月点区间 ±0.1 无量纲时间内加密到 n_samples 步
    3. 返回完整轨迹 + 近月点状态
    """
```

**估计工作量**：~80 行。

### Step 4：ThreeBodyLambert 打靶精化

**文件：`e2m2e/algorithm/transfer/lga.py`**

对搜索得到的最优候选，用 ThreeBodyLambert 在 CR3BP 下做精确打靶：

```python
def _refine_lga_candidate(
    candidate: LgaCandidate,
    system: CR3BP_System,
    dynamics: CR3BP_Dynamics,
    target_state: np.ndarray,
) -> LgaCandidate:
    """用 ThreeBodyLambert 打靶精化 LGA 候选。

    分两段打靶：
    1. 出发段：departure → perilune（ThreeBodyLambert 打靶修正出发速度）
    2. 到达段：perilune → target（ThreeBodyLambert 打靶修正到达速度）

    打靶后的出发/到达速度更新 Δv 计算。
    """
```

**估计工作量**：~60 行。

### Step 5：transfer_orbit("LGA") 编排器

**文件：`e2m2e/algorithm/transfer/__init__.py`（修改）**

#### 5.1 LGA 结果数据结构

```python
@dataclass
class LgaTransferDetails:
    """LGA 月球引力辅助转移设计细节。"""
    tli_epoch: float | str           # 出发历元
    tof_sec: float                   # 飞行时间 (s)
    perilune_alt_km: float           # 近月点高度 (km)
    perilune_vel_km_s: float         # 近月点速度 (km/s)
    perilune_state: np.ndarray       # 近月点状态 (6,) 相对月心 km, km/s
    dv_departure_km_s: float         # 出发脉冲 (km/s)
    dv_arrival_km_s: float           # 到达脉冲 (km/s)
    jacobi_departure: float          # 出发侧 Jacobi 常数
    jacobi_arrival: float            # 到达侧 Jacobi 常数
    n_candidates_searched: int       # 搜索候选总数
    n_candidates_feasible: int       # 可行候选数
    converged: bool                  # 打靶是否收敛
    search_params: LgaSearchParams   # 搜索参数快照
```

#### 5.2 编排器扩展

```python
def transfer_orbit(
    transfer_type: str,
    *,
    target_ephemeris: Any = None,
    tli_params: TliParams | None = None,
    tof_range: tuple[float, float] | None = None,
    target_orbit_radius_km: float | None = None,
    dynamics: Any = None,
    lga_search_params: LgaSearchParams | None = None,
    **kwargs,
) -> TransferDesignResult:
    if transfer_type == "HMN":
        return _transfer_orbit_hmn(...)
    if transfer_type == "LGA":
        return _transfer_orbit_lga(
            tli_params=tli_params,
            target_ephemeris=target_ephemeris,
            search_params=lga_search_params,
            dynamics=dynamics,
        )
    raise NotImplementedError(...)
```

#### 5.3 LGA 编排体

```python
def _transfer_orbit_lga(
    tli_params: TliParams | None,
    target_ephemeris: Any,
    search_params: LgaSearchParams | None,
    dynamics: Any = None,
) -> TransferDesignResult:
    """LGA 月球引力辅助转移编排。

    流程：
    1. TliParams → ECI 出发态
    2. 目标星历 → 目标态
    3. ECI → CR3BP 无量纲
    4. search_lga_trajectories() 网格搜索
    5. 取最优候选
    6. _refine_lga_candidate() ThreeBodyLambert 打靶精化
    7. 物理单位换算 + 结果汇总
    """
```

**估计工作量**：~120 行（含 __init__.py 修改）。

### Step 6：端到端测试

**文件：`tests/transfer/test_lga.py`（新增）**

测试策略遵循 ADR 0013：按物理定义验证，不用黄金样本。

#### 6.1 测试类

```python
class TestLgaSearchParams:
    """搜索参数验证。"""

    def test_default_params_valid(self):
        """默认参数范围合理：tof 3-7天，perilune 100-10000km。"""

    def test_invalid_phase_range_raises(self):
        """相位角范围超出 [0, 2π) 报错。"""


class TestLgaSearch:
    """LGA 弹道搜索单元测试。"""

    def test_search_returns_candidates(self):
        """给定典型 LEO→月球参数，搜索应返回非空候选列表。"""

    def test_candidates_sorted_by_dv(self):
        """候选按 total_dv 升序排列。"""

    def test_periapsis_detected_within_range(self):
        """可行候选的近月点高度在 perilune_alt_range 内。"""

    def test_jacobi_conservation(self):
        """CR3BP Jacobi 常数在飞越前后守恒（差 < 1e-8）。

        验证方法（ADR 0013）：
        C = 2·U(x,y,z) - (vx²+vy²+vz²)
        其中 U 为 CR3BP 伪势。飞越前后 C 值之差应 < 1e-8。
        """

    def test_trajectory_continuity_at_perilune(self):
        """飞越段在近月点处轨道连续：前段末态 ≈ 后段初态。"""


class TestLgaTransferOrbit:
    """transfer_orbit("LGA") 端到端测试。"""

    def test_returns_transfer_design_result(self):
        """transfer_orbit("LGA", ...) 返回 TransferDesignResult，transfer_type == "LGA"。"""

    def test_lga_details_populated(self):
        """details 包含 LgaTransferDetails 全部字段。"""

    def test_total_dv_positive(self):
        """总 Δv > 0（有出发和到达脉冲）。"""

    def test_periapsis_alt_in_range(self):
        """近月点高度在合理范围内（100-10000 km）。"""

    def test_energy_gain_demonstrated(self):
        """LGA 的总 Δv < 同等条件直飞 HMN 的 Δv。

        这是引力辅助的核心价值——以更少的燃料消耗实现转移。
        注：在某些相位角下 LGA 可能不如 HMN，此测试用最优候选。"""


class TestLgaPhysics:
    """LGA 物理不变量验证。"""

    def test_jacobi_constant_conservation(self):
        """Jacobi 常数沿整个转移轨迹守恒（CR3BP 模型下）。"""

    def test_perilune_speed_exceeds_escape(self):
        """近月点速度 > 月球逃逸速度（确认飞越，非捕获）。

        月球逃逸速度 v_esc = √(2μ_moon/r_peri)
        在 CR3BP 中等价于：近月点处相对月心速度 > √(2μ_moon/r_peri)。
        """

    def test_departure_arrival_state_continuity(self):
        """出发段末态与飞越段初态位置连续（拼接残差 < 1e-6 无量纲）。"""
```

#### 6.2 测试参数

```python
# 典型 LGA 算例参数
MU_CR3BP = 1.21506683e-2  # 地月 CR3BP 质量比
DU = 384405.0             # km（地月平均距离）
TU = 375196.0             # s（CR3BP 时间单位）
VU = DU / TU              # km/s（速度单位）
R_MOON_SOI = 66300.0      # km（Laplace SOI，Zhang et al. 2023）

# LEO 停泊轨道
PARKING_ALT_KM = 200.0
R_PARKING = R_EARTH + PARKING_ALT_KM  # km

# 月球参数
R_MOON = 1737.4           # km
PERILUNE_ALT = 500.0      # km（典型近月飞越高度）

# LGA 典型参数（Shi et al. 2025 Table 1）
LGA_TOF_MIN_DAYS = 15.0   # 最短飞行时间
LGA_TOF_MAX_DAYS = 45.0   # 最长飞行时间
LGA_DV_TYPICAL = 0.250    # km/s（单次 LGA 典型总 Δv）
```

**估计工作量**：~250 行。

### Step 7（可选）：DFH 交叉参考脚本

**文件：`scripts/dfh_lga_compare.py`（新增，不进 CI）**

```python
"""DFH RESULTS_LGA 交叉对比脚本（开发期诊断，不进 CI）。

用法：python scripts/dfh_lga_compare.py

参照 ADR 0013 第 4 条：
- 脚本放 scripts/，不进 CI、不进发布包
- 用于诊断量级/系统性偏差
"""
```

**估计工作量**：~60 行。

---

## 4. Jacobi 常数验证公式

CR3BP Jacobi 常数（守恒量）：

```
C = 2·U(x, y, z) - (ẋ² + ẏ² + ż²)

其中伪势：
U(x, y, z) = ½(x² + y²) + (1-μ)/r₁ + μ/r₂

r₁ = √((x+μ)² + y² + z²)    （相对主天体距离）
r₂ = √((x-1+μ)² + y² + z²)  （相对次天体距离）
```

飞越前后 C 值之差应 < 1e-8（无量纲），对应物理量级 ~0.01 m²/s²。

---

## 5. 风险与缓解

| 风险 | 等级 | 缓解 | 文献依据 |
|------|------|------|---------|
| 搜索网格太粗，漏掉最优解 | 中 | 初版 50×50 网格（2500 组合），后续可加粗或用 continuation 细化 | Parker & Anderson (2014) §4.3 |
| 近月点检测在高倾角飞越时遗漏 | 低 | PoincareSection.periapsis 用 r·v 零点检测，与倾角无关；Brent 求精残差 1e-14 | sections.py 已验证 |
| CR3BP 近似 vs 星历模型偏差 | 中 | 初版 CR3BP-only；星历修正作 Phase 3 增强（复用 ephemeris_shoot_transfer） | Liu et al. (2008) |
| 搜索结果多解——不同相位角给出不同飞越几何 | 低 | 取 total_dv 最小候选；不试图覆盖全部解族 | — |
| ThreeBodyLambert 打靶对飞越段初猜敏感 | 中 | 用搜索阶段的传播态作为初猜（比 Lambert 初猜更接近解）；阻尼 Newton 已内置 | three_body_lambert.py 已验证 |
| 与 #259 共用底座接口过早固化 | 低 | 初版不抽取共享层；#259 实施时再审视 | issue brief 建议 |

---

## 6. 工作量估计

| Step | 新增/修改代码 | 估计行数 |
|------|-------------|---------|
| Step 1: 搜索参数与数据结构 | `lga.py` 新增 | ~60 行 |
| Step 2: CR3BP 弹道搜索 | `lga.py` 新增 | ~150 行 |
| Step 3: 飞越段处理 | `lga.py` 新增 | ~80 行 |
| Step 4: ThreeBodyLambert 精化 | `lga.py` 新增 | ~60 行 |
| Step 5: 编排器 | `__init__.py` 修改 + `lga.py` | ~120 行 |
| Step 6: 测试 | `test_lga.py` 新增 | ~250 行 |
| Step 7: DFH 脚本 | `scripts/` 新增 | ~60 行 |
| **合计** | | **~780 行** |

---

## 7. 实施顺序与依赖

```
Step 1 (lga.py: LgaSearchParams + LgaCandidate)
    │
    ├── Step 6a (test_lga_search_params — 可立即写，TDD RED)
    │
    ▼
Step 2 (弹道搜索: search_lga_trajectories)
    │
    ├── Step 6b (test_lga_search 单元测试)
    │
    ▼
Step 3 (飞越段: _propagate_with_periapsis_refinement)
    │
    ├── Step 6c (test_periapsis_detected_within_range)
    │
    ▼
Step 4 (ThreeBodyLambert 精化: _refine_lga_candidate)
    │
    ▼
Step 5 (编排器: transfer_orbit("LGA") + LgaTransferDetails)
    │
    ├── Step 6d (test_transfer_orbit_lga 端到端)
    │
    ▼
Step 6e (test_lga_physics 物理不变量)
    │
    ▼
Step 7 (DFH 交叉参考脚本，可选，不阻塞 PR)
```

Step 1-5 是核心路径。Step 7 不阻塞合并。

---

## 8. 与现有 issue/PR 的关系

| Issue/PR | 关系 |
|----------|------|
| #256 (HMN) | ✅ CLOSED，其 TliParams + construct_departure_state 完全复用 |
| #259 (WSB) | 后续，共用三体弹道搜索底座（本 PR 先实现 LGA，#259 再审视共享） |
| #279 (h_init bug) | 无直接关系，可并行 |
| #280 (TIGHT/SPECIAL) | 无直接关系，可并行 |
| #261 (角动量管理) | 无直接关系，可并行 |

---

## 9. 文献引用表

### 直接引用（方案公式来源）

| 编号 | 文献 | 方案中引用位置 | 关键内容 |
|------|------|---------------|---------|
| [PA14] | Parker, J. S. & Anderson, R. L. (2014). *Low-Energy Lunar Trajectory Design*. JPL DESCANSO/Wiley. | Step 2 §搜索算法, §10 公式验证 | §2.5.1.3 Jacobi 常数公式（Eq. 2.6-2.8）；§2.5.2 Patched Three-Body Model（Eq. 2.9 3BSOI）；§3.4 低能转移构造（6 参数法）；§4.3 引力辅助轨道搜索 |
| [Cui25] | Cui et al. (2025). | Step 2 §搜索编排 | 搜索-优化两步法；DRO-RO 转移框架中的网格搜索模式 |
| [Shi25] | Shi et al. (2025). Review of cislunar space transfer trajectory design based on impulsive maneuvers. | §1 LGA 分类, §10.3 飞越物理 | LGA 分类（unpowered/powered）；TOF 15-45天/Δv ~250m/s 统计；Three-Body Lambert 求解步骤 |
| [Zhang23] | Zhang et al. (2023). Overview of Earth-moon transfer trajectory modeling and design. *Astrodynamics*. | §10.1-10.2 公式验证 | Eq. 3 Ω 定义（含常数项）；Eq. 1 Laplace SOI（66,300 km）；§2.1 Patched Conic 方法；SOI 边界速度组合效应 |
| [Qi15] | Qi, Y. & Xu, S. (2015). Mechanical analysis of lunar gravity assist in the earth-moon system. *Astrophysics and Space Science*, 360, 55. | §10.3 LGA 力学 | LGA 力学分析：速度方向/大小变化机理 |
| [Qi17] | Qi, Y., Xu, S., & Qi, R. (2017). Transfer from earth to libration point orbit using lunar gravity assist. *Acta Astronautica*, 133, 145-157. | 参考 | LGA + 流形拼接转移设计 |
| [Wilson98] | Wilson, R. & Howell, K. (1998). Trajectory design in the sun-earth-moon system using lunar gravity assists. *J. Spacecraft and Rockets*, 35, 191-198. | 参考 | LGA 轨道设计经典文献（有动力 LGA） |
| [Liu08] | 刘磊等 (2008). 多约束条件下的地月转移轨道设计. *宇航学报*. | Step 3, 风险表 | 三阶段收敛策略；微分修正方程 |
| [Gómez01] | Gómez, G. et al. (2001). *Dynamics and Mission Design Near Libration Points*. Vol. I. | Jacobi 常数公式 | CR3BP 伪势与 Jacobi 常数守恒（§2.2） |
| [Vallado] | Vallado, D. A. (2013). *Fundamentals of Astrodynamics*. 4th ed. | 飞越段物理 | 引力辅助 Δv 关系；双曲线超速守恒 |
| [Curtis08] | Curtis, H. (2008). *Orbital Mechanics for Engineering Students*. | TLI 构造 | Algorithm 4.2（已实现） |

### 已有代码中的文献引用（本方案复用）

| 文献 | 代码位置 | 复用组件 |
|------|---------|---------|
| Izzo (2015) | `lambert.rs`, `lambert.py` | Lambert 求解器 |
| Battin | `ephemeris_dynamics.py`, `lambert.rs` | N 体动力学；Lambert 超几何级数 |
| Gómez (2001) | `manifolds.py`, `sections.py` | 流形计算；近月点截面检测 |

---

## 10. 文献公式验证

> 以下验证基于本地文献库（`C:\baidunetdiskdownload\地月空间相关md\output\`）。

### 10.1 Jacobi 常数公式 ✅ 正确

**实施方案第 4 节**：

```
C = 2·U(x,y,z) - (ẋ² + ẏ² + ż²)
U = ½(x² + y²) + (1-μ)/r₁ + μ/r₂
```

**文献对照**：

| 文献 | 公式 | 一致性 |
|------|------|--------|
| Parker & Anderson (2014) Eq. 2.6-2.8 | C = 2U - V², U = ½(x²+y²) + (1-μ)/r₁ + μ/r₂ | ✅ 完全一致 |
| Zhang et al. (2023) Eq. 3, 6 | C = 2Ω - (Ẋ²+Ẏ²+Ż²), Ω = ½(X²+Y²) + (1-μ)/R₁ + μ/R₂ + ½μ(1-μ) | ⚠️ 差一个常数 |

**差异说明**：Zhang 的 Ω 比 Parker 的 U 多了常数项 ½μ(1-μ)。这是因为 Zhang 用质心坐标系（原点在质心），Parker 用主天体坐标系（原点在主天体）。常数项在梯度运算中消失（不影响运动方程），只改变 Jacobi 常数的绝对值。**对守恒量验证无影响**——飞越前后 C 的差值在两种约定下相同。

**建议**：统一采用 Parker 约定（与现有 CR3BP_System 实现一致）。

### 10.2 SOI 半径公式 ⚠️ 需区分两个概念

**实施方案第 3 节**提到"SOI 进出事件"时给出 r_SOI ≈ 66,200 km。

**文献中有两个不同的 SOI 概念**：

| SOI 类型 | 公式 | 半径 | 文献 |
|----------|------|------|------|
| **Laplace SOI**（二体） | R = D(m/M)^(2/5) | ~66,300 km | Zhang et al. (2023) Eq. 1 |
| **3BSOI**（三体） | r = a(m_Moon/m_Sun)^(2/5) | ~159,200 km | Parker & Anderson (2014) Eq. 2.9 |

其中 D = 地月平均距离，m/M = 月球/地球质量比（Laplace），a ≈ 1 AU = 日月平均距离，m_Moon/m_Sun = 月球/太阳质量比（3BSOI）。

**关键差异**：
- Laplace SOI（66,300 km）：传统的二体影响球，用于 patched conic 方法
- 3BSOI（159,200 km）：三体模型的切换边界，包含 L₁ 和 L₂ 点，用于 patched three-body model

**对实施方案的影响**：方案采用 CR3BP 直接传播穿越（非 patched conic），不手动切换 SOI。SOI 概念仅用于验收标准中"双曲线超速 v∞ 不变"的物理验证。此处应明确使用 **Laplace SOI**（66,300 km），因为双曲线超速守恒是二体 SOI 内的性质。

**修正**：验收标准第 2 条的"进出 SOI 速度模相等"应改为"在 Laplace SOI 边界处（r ≈ 66,300 km），进出速度相对月心的模相等（无动力飞越）"。

### 10.3 LGA 飞越物理描述 ⚠️ 需修正

**实施方案第 1.1 节**描述为"利用航天器飞越月球时的引力效应改变速度方向和大小"。

**文献修正**：

Shi et al. (2025) Table 1 明确区分了两种 LGA：
- **无动力 LGA**（unpowered LGA）：航天器仅利用月球引力场改变速度方向和大小，**不消耗燃料**
- **有动力 LGA**（powered LGA）：在飞越点施加脉冲机动，实现更灵活的轨迹拼接

QI and XU (2015) "Mechanical analysis of lunar gravity assist in the earth-moon system" 给出了 LGA 的力学分析。

**关键物理约束**（Shi et al. 2025）：
- LGA 典型 TOF：15-45 天（单次 LGA ~250 m/s，双次 LGA ~220 m/s）
- LGA + 流形拼接：20-50 天，~460 m/s
- WSB + LGA：70-120 天，~104 m/s

**修正**：方案描述应改为"利用航天器飞越月球时的引力效应改变速度方向和大小（无动力 LGA），不消耗燃料在飞越段"。总 Δv 仅来自出发段和到达段的脉冲。

### 10.4 搜索方法的文献依据 ⚠️ 需补充

**实施方案**提出"出发窗口×TOF 网格搜索"。

**文献中有两种 LGA 设计方法**：

1. **Patched Conic + 网格搜索**（传统方法）：
   - Zhang et al. (2023) §2.1：将轨迹分为地心段（椭圆）和月心段（双曲线），在 SOI 边界拼接
   - 网格变量：出发时间 + TOF + 近月点高度
   - 优点：解析公式可用，计算快
   - 缺点：忽略三体耦合效应

2. **CR3BP 流形 + Poincaré 截面**（Parker & Anderson 2014 §3.4）：
   - 用 6 个参数定义转移：[F, C, θ, τ, p, Δtₘ]
   - 通过 Poincaré 截面（月球近拱点截面）识别可行转移
   - 优点：利用动力学系统结构，搜索空间更高效
   - 缺点：需要流形计算基础设施

3. **直接数值搜索**（实施方案采用）：
   - 在 CR3BP 中直接网格搜索出发相位×TOF
   - 近月点用 PoincareSection.periapsis 检测
   - 优点：实现简单，复用现有 CR3BP 基础设施
   - 缺点：搜索空间可能遗漏（无动力学结构引导）

**评估**：方案的直接数值搜索方法是合理的初始实现策略。与 Parker 的参数化方法相比，实现更简单但搜索效率较低。后续可增强为 Parker 参数化方法（复用流形基础设施）。

### 10.5 Patched Three-Body Model 的 SOI 切换 ✅ 方案正确回避

**Parker & Anderson (2014) §2.5.2** 描述了 patched three-body model：航天器在月球附近用 EM-CR3BP，远离月球用 SE-CR3BP，边界为 3BSOI。

**实施方案**采用单一 EM-CR3BP 模型全程传播。这在以下条件下有效：
- 转移时间 < 1 月（太阳摄动可忽略）
- LGA 转移典型 TOF 15-45 天（Shi et al. 2025），在单一 EM-CR3BP 内合理
- 但超过 ~30 天时，太阳摄动会显著影响轨迹精度

**建议**：初版用 EM-CR3BP；后续增强可引入 patched three-body model（复用 Sun-Earth CR3BP）或直接用星历模型。

### 10.6 文献补充引用

原方案遗漏以下关键文献：

| 编号 | 文献 | 重要性 |
|------|------|--------|
| [Shi25] | Shi et al. (2025). Review of cislunar space transfer trajectory design based on impulsive maneuvers. | LGA 分类（unpowered/powered）、TOF/Δv 统计、文献综述 |
| [Qi15] | Qi, Y. & Xu, S. (2015). Mechanical analysis of lunar gravity assist in the earth-moon system. *Astrophysics and Space Science*, 360, 55. | LGA 力学分析 |
| [Qi17] | Qi, Y., Xu, S., & Qi, R. (2017). Transfer from earth to libration point orbit using lunar gravity assist. *Acta Astronautica*, 133, 145-157. | LGA + 流形拼接 |
| [Wilson98] | Wilson, R. & Howell, K. (1998). Trajectory design in the sun-earth-moon system using lunar gravity assists. *Journal of Spacecraft and Rockets*, 35, 191-198. | LGA 轨道设计经典文献 |
| [TanXX] | Tan et al. 低能双脉冲转移（用周期轨道做 lunar flyby）| 有动力 LGA 方法 |

建议在方案文献引用表中补充以上文献。

---

## 11. 待确认事项（实施前需要决定）

1. **目标星历格式**：确认 `target_ephemeris` 接口——是 `NominalOrbit`（有 `.states` + `.times`）、`EphemerisTable`（有 `.position_km` + `.velocity_mps`）、还是 `np.ndarray (n, 6)`？`_extract_target_state()` 已处理三种格式，但 LGA 需要目标轨道的**多个状态点**（不止最后一行）用于到达段匹配。建议：目标轨道用 `Orbit` 类型（states/times）。

2. **CR3BP 单位换算**：出发态从 ECI (km) 换算到 CR3BP 无量纲，需要 CR3BP_System 初始化特征尺度。当前 `CR3BP_System(mu=MU)._with_default_scales()` 可用，但需确认 `DU = 384405 km` 和 `TU = 375196 s` 的值与系统一致。

3. **搜索网格分辨率**：初版 50×50 是否够用？对于 LGA 问题，出发相位角的敏感度取决于目标轨道类型（DRO vs Halo vs NRHO）。建议：50×50 作为默认，`LgaSearchParams` 允许用户覆盖。

4. **与 #259 共享层的范围**：issue brief 提到"间接转移搜索抽成共用层"。本方案先在 `lga.py` 内实现完整搜索，#259 实施时再审视共享。是否接受此策略？
