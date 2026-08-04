# Issue #259 实施方案：WSB 太阳引力辅助间接转移（FR3 第三部分）

> 供审查。验收标准已按 ADR 0013 修订（第 0 节）。
> 公式依据见第 9 节文献引用表，核对结果见第 10 节。
> 与 #258 LGA 方案的共享决策见第 2.3 节。

---

## 0. 验收标准修订（ADR 0013 对齐）

**原 issue #259 验收标准存在 ADR 0013 硬冲突**：issue brief 第 5 步要求"与 DFH RESULTS_WSB 黄金样本对比，容差进 tests/dfh/ 回归"，而 ADR 0013 明确规定：

> - 不使用黄金样本（golden file）对照
> - DFH 仅作开发期交叉参考，脚本放 `scripts/`，不进 CI
> - 正确性由物理定义裁决

**修订后验收标准：**

1. **端到端 WSB 算例**：TLI 参数 + 目标星历 → 转移星历 + 设计结果汇总（各段 Δv、弹道参数、近月点高度/速度、Kepler 能量 H₂）
2. **物理定义验证**（ADR 0013 第 1 条）：
   - 弹道捕获判据：H₂ < 0（相对月球 Kepler 能量为负——无需制动脉冲即被捕获）
   - 轨道连续性：拼接点处位置/速度无跳变（残差 < 1e-6 无量纲）
   - 太阳摄动有效性：BCR4BP 结果与 CR3BP 结果有显著差异（证明太阳摄动在 WSB 机制中起关键作用）
   - 近月点检测精度：半径极值与 Brent 法根定位一致（残差 < 1e-6 DU）
3. **弹道捕获段轨道连续且精度满足回归容差**：事件定位方案（PoincareSection.periapsis + Brent 求精）在实施说明中记录
4. **DFH 交叉参考**（开发期，不进 CI）：与 RESULTS_WSB.TXT 的对比脚本放 `scripts/`

---

## 1. 现状分析

### 1.1 WSB 物理模型概述

弱稳定边界（Weak Stability Boundary）转移利用太阳引力摄动实现月球弹道捕获。核心机制（Belbruno & Miller 1993）：

1. 航天器从地球出发，飞越月球获取引力辅助能量增益
2. 飞出至 WSB(Earth)（距地球约 1.5×10⁶ km，~4 倍地月距离）
3. 在 WSB 处，太阳 + 地球引力几乎抵消月球引力——航天器的开普勒能量 H₂ 从零变为负
4. 无需制动脉冲即被月球"捕获"（弹道捕获，Δv_capture = 0）
5. 捕获是临时的（不稳定），可由小量圆化脉冲稳定

**BCR4BP 框架下的 WSB 搜索**：

```
出发段：地球停泊轨道 → BCR4BP 前向传播
    │  太阳相位角 × TOF 网格扫描
    ▼
传播段：BCR4BP 全程传播（含太阳直接项 + 间接项摄动）
    │  太阳引力自然创造 WSB 区域
    ▼
筛选段：近月点检测 + H₂ < 0 筛选
    │  PoincareSection.periapsis("moon") + Kepler 能量
    ▼
精化段：最优候选 → ThreeBodyLambert 打靶精化
    │  CR3BP 下 Newton 迭代精化
    ▼
验证段：物理定义对照（ADR 0013）
```

**与 LGA 的关键区别**：

| 特征 | LGA（#258） | WSB（#259） |
|------|------------|------------|
| 动力学模型 | CR3BP（三体） | BCR4BP（四体 + 太阳摄动） |
| 搜索变量 | 出发相位角 × TOF | 太阳相位角 × TOF |
| 筛选准则 | 近月点高度 ∈ [min, max] | **H₂ < 0**（弹道捕获） + 近月点高度 ∈ [min, max] |
| 飞行时间 | 15-45 天 | **90-150 天**（3-5 月） |
| 捕获机制 | 月球引力偏转（飞越） | **太阳摄动创造 WSB 区域**（弹道捕获） |
| 到达 Δv | 需 LOI 脉冲 | **≈ 0**（弹道捕获） |
| Jacobi 常数 | 守恒（CR3BP） | **不守恒**（BCR4BP 时间周期系统） |

**物理特征**（Belbruno & Miller 1993 Table 3）：

- 比 Hohmann 省 ~18% Δv
- 中段修正 ~0.029 km/s + 圆化 ~0.648 km/s
- **捕获 Δv = 0**（弹道捕获——核心优势）
- TOF: 3-5 个月（主要劣势）

### 1.2 可复用组件

| 组件 | 位置 | 状态 | WSB 复用方式 |
|------|------|------|-------------|
| **BCR4BP 系统** | `bcr4bp_system.py` (BCR4BPSystem) | ✅ | 完全复用——太阳参数、太阳位置解析、`earth_moon()` 工厂 |
| **BCR4BP 动力学** | `bcr4bp_dynamics.py` (BCR4BP_Dynamics) | ✅ | 完全复用——运动方程、雅可比 A(t)、42 维 STM |
| **TLI 出发构造** | `hohmann.py` (TliParams, construct_departure_state) | ✅ | 完全复用 |
| **ThreeBodyLambert** | `three_body_lambert.py` | ✅ | 完全复用——到达段打靶精化 |
| **庞加莱截面** | `manifold/sections.py` (PoincareSection, detect_crossings) | ✅ | 完全复用——近月点检测（r·v=0 + Brent 求精） |
| **EBCRS 时空转换** | `coordinate/gcrs_ebcrs.py` (GCRSEBCRSSystem) | ✅ (#252 CLOSED) | EBCRS 精化阶段（Phase 3 增强） |
| **星历打靶** | `hohmann.py` (ephemeris_shoot_transfer) | ✅ | 星历模型精化（Phase 3 增强） |
| **CR3BP 系统** | `cr3bp_system.py` (CR3BP_System) | ✅ | 无量纲化/有量纲化转换（BCR4BP 共用约定） |
| **Lambert 求解器** | `lambert.py` (solve_lambert, solve_lambert_batch) | ✅ | CR3BP 初猜（备选） |
| **TransferDesignResult** | `__init__.py` | ✅ | 扩展 WSB 分支 |
| **TransferType.WSB** | `data/templates/enums.py` | ⚠️ 待确认 | 需确认枚举是否已定义 |

### 1.3 需新增

| 缺失组件 | 说明 |
|----------|------|
| **wsb.py（核心）** | WSB 弹道搜索算法：BCR4BP 前向传播 + H₂ 筛选 + 近月点检测 |
| **compute_kepler_energy_moon()** | 旋转系中相对月球 Kepler 能量计算（H₂ 旋转系 → 惯性系修正） |
| **WsbTransferDetails** | WSB 转移结果数据结构 |
| **transfer_orbit("WSB") 分支** | 编排器扩展 |
| **test_wsb.py** | 物理定义验证 + 端到端测试 |
| **scripts/dfh_wsb_compare.py** | DFH 交叉参考脚本（不进 CI） |

### 1.4 BCR4BP 动力学现状评估

**结论：BCR4BP 基础设施已就位，经文献验证公式正确。无需新增 Rust 代码。**

`BCR4BPSystem` + `BCR4BP_Dynamics` 提供：
- 完整运动方程（CR3BP + 太阳直接项 + 间接项）——经 Belbruno 1993 Eq(1') 验证 ✓
- 雅可比矩阵 A(t)（6×6，显式含时）——经标准第三体雅可比公式验证 ✓
- 42 维增广状态（状态 + STM）——支持打靶精化
- 太阳位置解析 `r_s(t) = a_s·(cos θ, sin θ, 0)`——标准双圆近似 ✓
- `earth_moon()` 工厂方法（默认地月参数，DE440 常数）

**注意**：BCR4BP 无 Jacobi 积分（时间周期系统），`compute_jacobi_constant()` 抛 NotImplementedError。WSB 用 **Kepler 能量 H₂** 替代 Jacobi 作为弹道捕获判据。

**PoincareSection 兼容性**：`detect_crossings(times, states, section)` 接受轨迹数据（times, states），不绑定特定动力学模型。BCR4BP 传播结果可直接传入，无需修改。

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
[Step 2] 无量纲化 → BCR4BP 出发态                ← 复用 CR3BP_System
    │  r0, v0 → x0 (无量纲)，BCR4BP 共用 CR3BP 无量纲约定
    ▼
[Step 3] BCR4BP 弹道网格搜索                       ← 新: wsb.py
    │  for each (sun_phase, tof):
    │    3a. 设置 system.sun_phase0 = sun_phase
    │    3b. BCR4BP 前向传播 tof 时间
    │    3c. PoincareSection.periapsis("moon") 检测近月点
    │    3d. 计算 H₂（Kepler 能量相对月球，旋转系→惯性系修正）
    │    3e. 筛选: H₂ < 0 AND perilune_alt ∈ [min, max]
    │    3f. 计算总 Δv = |Δv_dep| + |Δv_arr|
    │  输出: 候选解列表 sorted by Δv
    ▼
[Step 4] 最优候选 → ThreeBodyLambert 打靶精化     ← 复用 three_body_lambert.py
    │  输入: 近月点状态 + 目标终端
    │  输出: 收敛的转移轨迹
    ▼
[Step 5] 结果汇总                                  ← 新: wsb.py
    │  输出: TransferDesignResult(transfer_type="WSB", ...)
    ▼
[验证] 物理定义对照                                ← test_wsb.py
```

### 2.2 模块组织

```
e2m2e/algorithm/transfer/
├── __init__.py          # 修改: transfer_orbit() 加 WSB 分支 + WsbSearchParams/WsbCandidate 导入
├── wsb.py               # 新增: BCR4BP 弹道搜索 + H₂ 筛选 + Kepler 能量计算
├── lga.py               # 现有，不修改
├── hohmann.py           # 现有，不修改
├── three_body_lambert.py# 现有，不修改
└── ...

tests/transfer/
├── test_wsb.py          # 新增: WSB 端到端测试（物理定义验证）

scripts/
├── dfh_wsb_compare.py   # 新增: DFH RESULTS_WSB 交叉对比（不进 CI）
```

### 2.3 关键设计决策

| 决策 | 方案 | 理由 |
|------|------|------|
| 搜索动力学模型 | BCR4BP | WSB 本质是太阳摄动效应，必须用四体模型；BCR4BPSystem/BCR4BP_Dynamics 已就位 |
| 搜索方法 | 前向传播 + H₂ 筛选 | 复用现有 BCR4BP_Dynamics + PoincareSection 基础设施；与 LGA 搜索范式一致；实现简单 |
| 弹道捕获判据 | H₂ < 0（Kepler 能量） | Belbruno 定义：H₂ 从零变负即为弹道捕获（Belbruno 2010 Eq 2.8） |
| 精化方法 | ThreeBodyLambert + 星历打靶 | 与 LGA 统一；ThreeBodyLambert 在 CR3BP 下精化，星历打靶在真实力模型下精化 |
| 与 LGA 共享底座 | **不抽取共享搜索层**，共享编排器基础设施 | 搜索算法差异大（CR3BP 网格 vs BCR4BP 传播 + H₂）；强行抽象是过早优化；共享 TLI 构造、_extract_target_state、transfer_orbit 分发 |
| 搜索变量 | 太阳相位角 × TOF | BCR4BP 中太阳相位决定初始日-地-月几何；TOF 决定飞行时间（WSB 典型 90-150 天） |
| Belbruno 反向积分 | **作为增强方案记录**（不阻塞 MVP） | 正向搜索更简单、复用更多基础设施；反向积分更高效但需捕获态参数化 + 反向传播基础设施 |
| 搜索并行化 | `concurrent.futures.ProcessPoolExecutor` | 每个 (sun_phase, tof) 组合独立（embarrassingly parallel）；使用 `os.cpu_count()` 个工作进程充分利用多核 CPU |

---

## 3. 实施步骤

### Step 1：WSB 搜索参数与数据结构（`wsb.py`）

**文件：`e2m2e/algorithm/transfer/wsb.py`（新增）**

#### 1.1 搜索配置

```python
@dataclass(frozen=True)
class WsbSearchParams:
    """WSB 弹道搜索参数。

    搜索空间：太阳相位角 × 飞行时间（TOF）。
    近月点高度由传播自然决定，不作为独立搜索变量。
    弹道捕获由 H₂ < 0 判定（Belbruno & Miller 1993）。

    Attributes:
        sun_phase_range: 太阳相位角范围 (min, max)，弧度，[0, 2π)
        n_sun_phase: 太阳相位角网格点数
        tof_range: 飞行时间范围 (min, max)，天（WSB 典型 90-150 天，Belbruno 1993）
        n_tof: TOF 网格点数
        perilune_alt_min: 近月点高度下限 (km)，低于此值的候选丢弃（避免撞击月面）
        perilune_alt_max: 近月点高度上限 (km)，高于此值的候选丢弃
        max_total_dv: 最大总 Δv 筛选阈值 (km/s)
        h2_threshold: H₂ 筛选阈值（默认 0.0；H₂ < 0 即弹道捕获）
    """
    sun_phase_range: tuple[float, float] = (0.0, 2.0 * math.pi)
    n_sun_phase: int = 50
    tof_range: tuple[float, float] = (90.0, 150.0)  # 天（Belbruno 1993: 3-5 月）
    n_tof: int = 50
    perilune_alt_min: float = 100.0   # km（避免撞击）
    perilune_alt_max: float = 10000.0 # km
    max_total_dv: float = 5.0  # km/s
    h2_threshold: float = 0.0  # H₂ < 0 弹道捕获
```

#### 1.2 Kepler 能量计算

```python
def compute_kepler_energy_moon(
    state: npt.NDArray[np.floating],
    mu: float,
) -> float:
    """计算相对月球的 Kepler 能量 H₂（旋转系中，含惯性系速度修正）。

    H₂ = ½|v_rel|² - μ/|r_rel|

    旋转系 → 惯性系相对速度推导：
        v_sc_inertial = v_rot + ω × r,       ω = (0, 0, 1)
        v_moon_inertial = ω × r_moon = (0, 1-μ, 0)
        v_rel = v_sc_inertial - v_moon_inertial
            = (vx - y,  vy + x - (1-μ),  vz)

    Args:
        state: 旋转系状态 [x, y, z, vx, vy, vz]（无量纲）
        mu: CR3BP 质量参数

    Returns:
        H₂ 值（无量纲）。H₂ < 0: 弹道捕获；H₂ = 0: WSB 边界；H₂ > 0: 双曲飞越
    """
    x, y, z, vx, vy, vz = np.asarray(state, dtype=float)
    x_moon = 1.0 - mu

    # 相对位置
    dx, dy, dz = x - x_moon, y, z
    r_rel = np.sqrt(dx**2 + dy**2 + dz**2)
    if r_rel < 1e-12:
        return float("nan")

    # 旋转系 → 惯性系相对速度
    vx_rel = vx - y
    vy_rel = vy + x - x_moon   # = vy + x - (1-μ)
    vz_rel = vz

    v_rel_sq = vx_rel**2 + vy_rel**2 + vz_rel**2
    return 0.5 * v_rel_sq - mu / r_rel
```

#### 1.3 WSB 搜索结果

```python
@dataclass
class WsbCandidate:
    """单个 WSB 候选解（弹道捕获）。

    捕获段 Δv = 0（弹道捕获——太阳摄动使 H₂ < 0），总 Δv = 出发脉冲 + 圆化脉冲。
    """
    sun_phase: float           # 太阳相位角 (rad)
    tof_sec: float             # 飞行时间 (s)
    departure_state: np.ndarray # 出发态 (6,) km, km/s
    perilune_state: np.ndarray  # 近月点态 (6,) 相对月心，km, km/s
    perilune_alt_km: float      # 近月点高度 (km)
    h2_kepler: float            # Kepler 能量 H₂（弹道捕获判据）
    dv_departure: float         # 出发脉冲 (km/s)
    dv_arrival: float           # 到达脉冲 (km/s)，WSB 应为 ~0
    total_dv: float             # 总 Δv = dv_departure + dv_arrival (km/s)
    converged: bool             # ThreeBodyLambert 打靶是否收敛
```

**估计工作量**：~120 行（含 H₂ 计算）。

### Step 2：BCR4BP 弹道传播 + H₂ 筛选

**文件：`e2m2e/algorithm/transfer/wsb.py`**

#### 2.1 核心搜索函数

算法来源：Belbruno & Miller (1993) WSB 搜索 + Parker & Anderson (2014) §3.4 流形思想。
本方案采用前向传播（实现简单、复用基础设施），Belbruno 反向积分作为增强方案记录。

```python
import itertools
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

def search_wsb_trajectories(
    departure_state: np.ndarray,
    target_state: np.ndarray,
    system: BCR4BPSystem,
    dynamics: BCR4BP_Dynamics,
    params: WsbSearchParams = WsbSearchParams(),
) -> list[WsbCandidate]:
    """WSB 弹道网格搜索（BCR4BP 前向传播 + H₂ 筛选）。

    对每个 (sun_phase, tof) 组合：
    1. 设置 system.sun_phase0 = sun_phase（初始日-地-月几何）
    2. BCR4BP 前向传播 tof 时间
    3. PoincareSection.periapsis("moon") 检测近月点
    4. compute_kepler_energy_moon() 计算 H₂
    5. 筛选: H₂ < h2_threshold AND perilune_alt ∈ [min, max]
    6. 计算总 Δv = |Δv_dep| + |Δv_arr|
    7. max_total_dv 筛选

    Args:
        departure_state: BCR4BP 无量纲出发态 (6,)
        target_state: BCR4BP 无量纲目标态 (6,)
        system: BCR4BP 系统
        dynamics: BCR4BP 动力学
        params: 搜索参数

    Returns:
        按 total_dv 升序排列的候选列表
    """

    # 并行搜索：每个 (sun_phase, tof) 组合独立，embarrassingly parallel
    combos = list(itertools.product(sun_phase_grid, tof_grid))

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {
            executor.submit(_evaluate_wsb_combo, sp, tof, departure_state,
                           target_state, system, params): (sp, tof)
            for sp, tof in combos
        }
        results = []
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)

    return sorted(results, key=lambda c: c.total_dv)


def _evaluate_wsb_combo(
    sun_phase: float,
    tof_sec: float,
    departure_state: np.ndarray,
    target_state: np.ndarray,
    system_template: BCR4BPSystem,
    params: WsbSearchParams,
) -> WsbCandidate | None:
    """评估单个 (sun_phase, tof) 组合（供 ProcessPoolExecutor 调用）。

    1. 克隆 system 并设置 sun_phase0
    2. BCR4BP 前向传播
    3. 近月点检测 + H₂ 筛选
    4. 返回 WsbCandidate 或 None
    """
```

#### 2.2 近月点检测复用

```python
from ..manifold.sections import PoincareSection, detect_crossings

# 近月点截面：r·v = 0（相对月心位置与速度的点积为零）
periapsis_section = PoincareSection.periapsis("moon", system)

# BCR4BP 传播完成后用 detect_crossings 提取近月点
# （PoincareSection 不绑定特定动力学模型，接受轨迹数据）
crossings = detect_crossings(times, states, periapsis_section)
if crossings:
    t_peri, state_peri, _ = crossings[0]
    # 计算 H₂ 和近月点高度
    h2 = compute_kepler_energy_moon(state_peri, system.mu)
    r_peri = np.linalg.norm(state_peri[:3] - np.array([1 - system.mu, 0.0, 0.0]))
    alt_peri = r_peri * DU - R_MOON  # 换算为 km
```

#### 2.3 搜索空间说明

BCR4BP 中太阳相位角 `sun_phase0` 决定 t=0 时刻太阳在旋转系中的位置，对应不同的出发日期（日-地-月几何）。搜索遍历不同太阳相位 × TOF 组合，找到弹道捕获走廊。

- 太阳相位周期 ≈ 2π/|ω_s| ≈ 29.5 天（会合月），50 点覆盖一个完整周期
- TOF 90-150 天覆盖 Belbruno 1993 报告的 WSB 典型飞行时间
- 总搜索量 50×50 = 2500 组合，与 LGA 同量级

**估计工作量**：~180 行。

### Step 3：ThreeBodyLambert 打靶精化

**文件：`e2m2e/algorithm/transfer/wsb.py`**

对搜索得到的最优候选，用 ThreeBodyLambert 在 CR3BP 下做精确打靶（与 LGA 精化逻辑一致）：

```python
def _refine_wsb_candidate(
    candidate: WsbCandidate,
    system: CR3BP_System,
    dynamics: CR3BP_Dynamics,
    target_state: np.ndarray,
) -> WsbCandidate:
    """用 ThreeBodyLambert 打靶精化 WSB 候选。

    分两段打靶（与 LGA _refine_lga_candidate 同构）：
    1. 出发段：departure → perilune（ThreeBodyLambert 打靶修正出发速度）
    2. 到达段：perilune → target（ThreeBodyLambert 打靶修正到达速度）

    WSB 特性：到达段 Δv 应接近 0（弹道捕获），精化应保持此性质。
    打靶后的出发/到达速度更新 Δv 和 H₂ 计算。
    """
```

**估计工作量**：~60 行（可参考 `lga.py` `_refine_lga_candidate`）。

### Step 4：transfer_orbit("WSB") 编排器

**文件：`e2m2e/algorithm/transfer/__init__.py`（修改）**

#### 4.1 WSB 结果数据结构

```python
@dataclass
class WsbTransferDetails:
    """WSB 太阳引力辅助转移设计细节。"""
    tli_epoch: float | str           # 出发历元
    tof_sec: float                   # 飞行时间 (s)
    perilune_alt_km: float           # 近月点高度 (km)
    perilune_vel_km_s: float         # 近月点速度 (km/s)
    perilune_state: np.ndarray       # 近月点状态 (6,) 相对月心 km, km/s
    dv_departure_km_s: float         # 出发脉冲 (km/s)
    dv_arrival_km_s: float           # 到达脉冲 (km/s)，WSB 应为 ~0
    h2_kepler: float                 # Kepler 能量 H₂（弹道捕获判据）
    n_candidates_searched: int       # 搜索候选总数
    n_candidates_feasible: int       # 可行候选数（H₂ < 0）
    converged: bool                  # 打靶是否收敛
    search_params: WsbSearchParams   # 搜索参数快照
```

#### 4.2 编排器扩展

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
    wsb_search_params: WsbSearchParams | None = None,  # 新增
    **kwargs,
) -> TransferDesignResult:
    if transfer_type == "HMN":
        return _transfer_orbit_hmn(...)
    if transfer_type == "LGA":
        return _transfer_orbit_lga(...)
    if transfer_type == "WSB":
        return _transfer_orbit_wsb(
            tli_params=tli_params,
            target_ephemeris=target_ephemeris,
            search_params=wsb_search_params,
            dynamics=dynamics,
        )
    raise NotImplementedError(f"transfer_orbit('{transfer_type}') 实现未完成（能力在规划中）")
```

#### 4.3 WSB 编排体

```python
def _transfer_orbit_wsb(
    tli_params: TliParams | None,
    target_ephemeris: Any,
    search_params: WsbSearchParams | None,
    dynamics: Any = None,
) -> TransferDesignResult:
    """WSB 太阳引力辅助转移编排。

    流程：
    1. TliParams → ECI 出发态
    2. 目标星历 → 目标态
    3. ECI → BCR4BP 无量纲（复用 CR3BP_System 转换）
    4. search_wsb_trajectories() BCR4BP 网格搜索
    5. 取最优候选
    6. _refine_wsb_candidate() ThreeBodyLambert 打靶精化
    7. 物理单位换算 + 结果汇总

    与 _transfer_orbit_lga 的主要区别：
    - 使用 BCR4BPSystem + BCR4BP_Dynamics（而非 CR3BP）
    - 搜索变量为太阳相位角 × TOF（而非出发相位角 × TOF）
    - 筛选准则为 H₂ < 0（弹道捕获）+ 近月点高度范围
    """
```

**估计工作量**：~130 行（含 `__init__.py` 修改）。

### Step 5：端到端测试

**文件：`tests/transfer/test_wsb.py`（新增）**

测试策略遵循 ADR 0013：按物理定义验证，不用黄金样本。

#### 5.1 测试类

```python
class TestKeplerEnergy:
    """H₂ Kepler 能量计算验证。"""

    def test_h2_negative_for_bound_orbit(self):
        """圆轨道 H₂ < 0（相对月球引力束缚）。"""
        # 在月球附近构造圆轨道状态，验证 H₂ < 0

    def test_h2_positive_for_hyperbolic(self):
        """高速飞越 H₂ > 0（双曲轨道）。"""
        # 构造高速飞越状态，验证 H₂ > 0

    def test_h2_rotation_correction(self):
        """旋转系 → 惯性系速度修正正确。
        验证：v_rel = (vx - y, vy + x - (1-μ), vz)。
        构造已知旋转系状态，手算惯性系速度，对比函数输出。"""


class TestWsbSearchParams:
    """搜索参数验证。"""

    def test_default_params_valid(self):
        """默认参数范围合理：tof 90-150 天，perilune 100-10000km。"""

    def test_invalid_phase_range_raises(self):
        """相位角范围超出 [0, 2π) 报错。"""


class TestWsbSearch:
    """WSB 弹道搜索单元测试。"""

    def test_search_returns_candidates(self):
        """给定典型 LEO→月球参数，BCR4BP 搜索应返回非空候选列表。"""

    def test_candidates_sorted_by_dv(self):
        """候选按 total_dv 升序排列。"""

    def test_h2_negative_for_wsb_candidates(self):
        """WSB 候选的 H₂ < 0（弹道捕获判据）。"""

    def test_perilune_detected(self):
        """可行候选的近月点高度在 perilune_alt_range 内。"""


class TestWsbTransferOrbit:
    """transfer_orbit("WSB") 端到端测试。"""

    def test_returns_transfer_design_result(self):
        """transfer_orbit("WSB", ...) 返回 TransferDesignResult，transfer_type == "WSB"。"""

    def test_wsb_details_populated(self):
        """details 包含 WsbTransferDetails 全部字段。"""

    def test_total_dv_positive(self):
        """总 Δv > 0（有出发脉冲）。"""

    def test_ballistic_capture_h2_negative(self):
        """H₂ < 0——弹道捕获判据成立。"""

    def test_capture_dv_near_zero(self):
        """到达段 Δv ≈ 0（弹道捕获，无制动脉冲）。
        容差：dv_arrival < 0.1 km/s（Belbruno 1993: 中段修正 ~0.029 km/s）。"""


class TestWsbPhysics:
    """WSB 物理不变量验证。"""

    def test_solar_perturbation_effective(self):
        """BCR4BP 结果与 CR3BP 结果有显著差异。
        方法：用相同初始条件分别在 BCR4BP 和 CR3BP 下搜索，
        验证 BCR4BP 能找到 H₂ < 0 的弹道捕获候选而 CR3BP 不能。"""

    def test_trajectory_continuity(self):
        """拼接点处轨道连续（残差 < 1e-6 无量纲）。"""

    def test_wsb_saves_dv_vs_hohmann(self):
        """WSB 最优候选的总 Δv < 同等条件直飞 HMN 的 Δv。
        Belbruno 1993 Table 3: WSB 比 Hohmann 省 ~18%。"""
```

#### 5.2 测试参数

```python
# WSB 典型算例参数
MU_CR3BP = 1.21506683e-2  # 地月 CR3BP 质量比
DU = 384405.0             # km（地月平均距离）
TU = 375196.0             # s（CR3BP 时间单位）
R_MOON = 1737.4           # km

# LEO 停泊轨道
PARKING_ALT_KM = 200.0
R_PARKING = R_EARTH + PARKING_ALT_KM  # km

# WSB 典型参数（Belbruno 1993）
WSB_TOF_MIN_DAYS = 90.0   # 最短飞行时间（3 月）
WSB_TOF_MAX_DAYS = 150.0  # 最长飞行时间（5 月）
WSB_DV_TYPICAL = 0.680    # km/s（中段修正 + 圆化，Belbruno 1993）
```

#### 5.3 测试策略（网格分辨率 + CI/release 区分）

| 场景 | 网格分辨率 | 执行时机 |
|------|-----------|---------|
| 本地开发（提交前） | 细网格（50×50） | 每次提交前必须通过 |
| Release | 粗网格（10×10） | release CI 流水线 |
| CI（常规推送） | **不执行 WSB 测试** | — |

标记方式：WSB 端到端测试用 `@pytest.mark.slow` 标记，CI 配置排除 slow 标记。

**估计工作量**：~280 行。

### Step 6（可选）：DFH 交叉参考脚本

**文件：`scripts/dfh_wsb_compare.py`（新增，不进 CI）**

```python
"""DFH RESULTS_WSB 交叉对比脚本（开发期诊断，不进 CI）。

用法：python scripts/dfh_wsb_compare.py

参照 ADR 0013 第 4 条：
- 脚本放 scripts/，不进 CI、不进发布包
- 用于诊断量级/系统性偏差
"""
```

**估计工作量**：~60 行。

---

## 4. Kepler 能量 H₂ 验证公式

### 4.1 旋转系中的 H₂ 计算

```
H₂ = ½|v_rel|² - μ/|r_rel|

其中：
r_rel = (x - (1-μ), y, z)              # 航天器相对月球位置（旋转系）
v_rel = (vx - y, vy + x - (1-μ), vz)    # 航天器相对月球惯性系速度

旋转系 → 惯性系速度推导：
  v_sc_inertial = v_rot + ω × r,         ω = (0, 0, 1)
  v_moon_inertial = ω × r_moon = (0, 1-μ, 0)
  v_rel = v_sc_inertial - v_moon_inertial
```

### 4.2 弹道捕获判据

| H₂ 值 | 物理含义 | 筛选动作 |
|--------|---------|---------|
| H₂ < 0 | 开普勒椭圆（弹道捕获） | **保留**——WSB 候选 |
| H₂ = 0 | 抛物线（WSB 边界） | 边界情况，保留 |
| H₂ > 0 | 双曲轨道（飞越） | **丢弃**——非 WSB |

### 4.3 文献对照

| 文献 | 公式 | 一致性 |
|------|------|--------|
| Belbruno 2010 Eq 2.8 | H₂ = ½v² - μ/r | ✅ 一致（旋转系中的 v 需修正为惯性系相对速度） |
| Belbruno 2010 Eq 2.9 | H₂ = (e-1)μ/(2r)（在近月点） | ✅ 等价（e < 1 → H₂ < 0） |
| Belbruno 1993 Eq 1' | 四体运动方程中的 Kepler 能量 | ✅ H₂ 是四体框架下的局部 Kepler 能量 |

### 4.4 数值示例

在月球近表面（r_rel ≈ R_MOON/DU ≈ 0.00452），圆轨道速度（v_circ ≈ √(μ/r_rel)）：

```
μ = 0.01215, r_rel = 0.00452
v_circ = √(0.01215 / 0.00452) ≈ 1.64（无量纲）
H₂ = ½ × 1.64² - 0.01215 / 0.00452 ≈ 1.345 - 2.688 ≈ -1.343 < 0 ✓（束缚轨道）
```

---

## 5. 风险与缓解

| 风险 | 等级 | 缓解 | 文献依据 |
|------|------|------|---------|
| BCR4BP 双圆近似 vs 真实星历偏差 | 中 | 初版 BCR4BP-only；星历精化作 Phase 3 增强（复用 ephemeris_shoot_transfer） | Belbruno 1993 最终用 14 阶积分器 + 星历精化 |
| WSB 轨道对初始条件敏感（混沌） | 中 | 粗搜索后加细搜索（在最优候选附近加密网格） | Belbruno 2010: WSB 是分形/Cantor 集 |
| 搜索网格可能漏掉 WSB 走廊 | 中 | 初版 50×50；后续可加密或引入反向积分方法 | Belbruno 方法用反向积分避免此问题 |
| 飞行时间长（3-5 月）→ 测试慢 | 中 | CI 用粗网格/小样本；精细网格放 optional | — |
| H₂ 在旋转系中是近似值 | 低 | 旋转系 H₂ 是 Kepler 能量的良好近似（旋转角速度量级 ω ≈ 2.7×10⁻⁶ rad/s，Coriolis 修正量级小）；精化阶段用星历模型验证 | — |
| 与 LGA 不共用搜索底座→代码重复 | 低 | 搜索算法差异大，强行抽象是过早优化；共享编排器基础设施 | #258 方案: "先在 lga.py 内实现完整搜索；#259 实施时再审视" |

---

## 6. 工作量估计

| Step | 新增/修改代码 | 估计行数 |
|------|-------------|---------|
| Step 1: 搜索参数 + H₂ 计算 + 数据结构 | `wsb.py` 新增 | ~120 行 |
| Step 2: BCR4BP 弹道搜索 | `wsb.py` 新增 | ~180 行 |
| Step 3: ThreeBodyLambert 精化 | `wsb.py` 新增 | ~60 行 |
| Step 4: 编排器 | `__init__.py` 修改 + `wsb.py` | ~130 行 |
| Step 5: 测试 | `test_wsb.py` 新增 | ~280 行 |
| Step 6: DFH 脚本 | `scripts/` 新增 | ~60 行 |
| **合计** | | **~830 行** |

---

## 7. 实施顺序与依赖

```
Step 1 (wsb.py: WsbSearchParams + WsbCandidate + compute_kepler_energy_moon)
    │
    ├── Step 5a (test_wsb_search_params + test_kepler_energy — 可立即写，TDD RED)
    │
    ▼
Step 2 (BCR4BP 弹道搜索: search_wsb_trajectories)
    │
    ├── Step 5b (test_wsb_search 单元测试)
    │
    ▼
Step 3 (ThreeBodyLambert 精化: _refine_wsb_candidate)
    │
    ▼
Step 4 (编排器: transfer_orbit("WSB") + WsbTransferDetails)
    │
    ├── Step 5c (test_transfer_orbit_wsb 端到端)
    │
    ▼
Step 5d (test_wsb_physics 物理不变量)
    │
    ▼
Step 6 (DFH 交叉参考脚本，可选，不阻塞 PR)
```

Step 1-4 是核心路径。Step 6 不阻塞合并。

---

## 8. 与现有 issue/PR 的关系

| Issue/PR | 关系 |
|----------|------|
| #252 (FR5 EBCRS) | ✅ CLOSED，GCRSEBCRSSystem 可用于 EBCRS 精化（Phase 3 增强） |
| #256 (HMN) | ✅ CLOSED，TliParams + construct_departure_state 完全复用 |
| #258 (LGA) | ✅ 已完成，编排器基础设施共享；搜索算法独立 |
| #279 (h_init bug) | 无直接关系，可并行 |
| #280 (TIGHT/SPECIAL) | 无直接关系，可并行 |
| #261 (角动量管理) | 无直接关系，可并行 |

---

## 9. 文献引用表

### 直接引用（方案公式来源）

| 编号 | 文献 | 方案中引用位置 | 关键内容 |
|------|------|---------------|---------|
| [BM93] | Belbruno, E. & Miller, J. K. (1993). Sun-perturbed earth-to-moon transfers with ballistic capture. *JGCD*, 16(4), 770-775. | §1.1 WSB 物理, §4 H₂ 公式 | WSB 定义、弹道捕获机制、四体方程 Eq(1/1')、Hiten 算例、TOF/Δv 统计（Table 3） |
| [BGT10] | Belbruno, E., Gidea, M. & Topputo, F. (2010). Weak stability boundary and invariant manifolds. *SIAM J. Appl. Dyn. Syst.*, 9(3), 1061-1089. | §1.1 WSB 定义, §4.1 H₂ 公式 | WSB 严格定义（Def 3.1: n-稳定/不稳定边界）、H₂ 公式 Eq 2.8/2.9、Jacobi 常数 Eq 2.4/2.7、流形-WSB 对应 Theorem 4.1 |
| [Gom01] | Gómez, G. et al. (2001). *Dynamics and Mission Design Near Libration Points*. Vol. I. World Scientific. | §10 公式验证 | CR3BP 运动方程 §1.1.2、伪势 Convention A/B §2.1.1、Jacobi 常数、Hessian 矩阵 |
| [PA14] | Parker, J. S. & Anderson, R. L. (2014). *Low-Energy Lunar Trajectory Design*. JPL DESCANSO/Wiley. | §10 公式验证 | Jacobi Eq 2.6-2.8、3BSOI Eq 2.9、流形搜索 §3.4、参考值 Table 2-4 |
| [B04] | Belbruno, E. (2004). *Capture Dynamics and Chaotic Motions in Celestial Mechanics*. Princeton. | §1.4 BCR4BP 与 WSB | WSB 在 BCR4BP 中的存在性（非自治系统不变流形） |

### 已有代码中的文献引用（本方案复用）

| 文献 | 代码位置 | 复用组件 |
|------|---------|---------|
| Izzo (2015) | `lambert.rs`, `lambert.py` | Lambert 求解器 |
| Gómez (2001) | `manifolds.py`, `sections.py` | 流形计算；近月点截面检测 |

---

## 10. 文献公式验证

> 以下验证基于本地文献库（`C:\baidunetdiskdownload\地月空间相关md\output\`）与代码交叉核对。

### 10.1 CR3BP 运动方程 ✅ 完全一致

| 公式 | 文献 | 代码 | 一致性 |
|------|------|------|--------|
| ẍ - 2ẏ = x - (1-μ)(x+μ)/r₁³ - μ(x-1+μ)/r₂³ | Parker Eq 2.1 | `bcr4bp_dynamics.py:108` | ✅ |
| ÿ + 2ẋ = y - (1-μ)y/r₁³ - μy/r₂³ | Parker Eq 2.2 | `bcr4bp_dynamics.py:109` | ✅ |
| z̈ = -(1-μ)z/r₁³ - μz/r₂³ | Parker Eq 2.3 | `bcr4bp_dynamics.py:110` | ✅ |
| r₁² = (x+μ)² + y² + z² | Parker Eq 2.4 | `dynamics.py:396` | ✅ |
| r₂² = (x-1+μ)² + y² + z² | Parker Eq 2.5 | `dynamics.py:396` | ✅ |

### 10.2 Jacobi 常数 ⚠️ 约定差异（已知、已文档化）

| 约定 | 公式 | 文献 | 代码 |
|------|------|------|------|
| Parker（无常数） | U = ½(x²+y²) + (1-μ)/r₁ + μ/r₂ | Parker Eq 2.7 | `cr3bp_system.py:327` ✅ |
| Gómez/Belbruno（含常数） | Ω = U + ½μ(1-μ) | Gómez §2.1.1, Belbruno Eq 2.2 | 不适用（代码用 Parker） |

差值：`C_Belbruno = C_Parker + μ(1-μ) ≈ 0.012`。常数项不影响运动方程（梯度为零）。

参考值验证：
- Parker Table 2-4: C(LL1) = 3.18834129
- Belbruno 2010: C₁ = 3.2003449098
- 差值 = 0.012004 ≈ μ(1-μ) = 0.012003 ✓

**注意**：BCR4BP 无 Jacobi 积分（时间周期系统，`compute_jacobi_constant()` 抛 NotImplementedError）。WSB 用 **H₂** 替代 Jacobi 作为能量准则，此约定差异对 WSB 实现影响极小。

**纠正 #258 方案 §10.1 的解释**：该节称"Zhang 用质心坐标系（原点在质心），Parker 用主天体坐标系（原点在主天体）"——**此解释有误**。两者都使用质心旋转系（原点在地月质心），差异纯粹是伪势定义是否包含常数项 ½μ(1-μ)。Gómez 2001 在同一本书中明确记录了两种约定（Ch.1 Convention A 无常数、Ch.2 Convention B 有常数），坐标系完全相同。

### 10.3 BCR4BP 太阳摄动 ✅ 完全一致

| 公式 | 文献 | 代码 | 一致性 |
|------|------|------|--------|
| a_sun = -m_s·[(r-r_s)/|r-r_s|³ + r_s/|r_s|³] | Belbruno 1993 Eq(1') | `bcr4bp_dynamics.py:78-84` | ✅ |
| J_sun = -m_s·(I/|d|³ - 3·d·d^T/|d|⁵) | 标准第三体雅可比 | `bcr4bp_dynamics.py:146` | ✅ |
| r_s(t) = a_s·(cos θ, sin θ, 0) | BCR4BP 双圆近似 | `bcr4bp_system.py:149-169` | ✅ |
| 间接项 -m_s·r_s/|r_s|³ | Belbruno 1993 Eq(1') 第二求和项 | `bcr4bp_dynamics.py:84` | ✅ |

### 10.4 SOI 定义 ⚠️ Laplace SOI 来源需注意

| SOI 类型 | 公式 | 来源 | WSB 依赖 |
|----------|------|------|---------|
| 3BSOI | r = a·(m_Moon/m_Sun)^(2/5) ≈ 159,200 km | Parker Eq 2.9 ✅ | WSB 不直接使用（BCR4BP 全程传播，无需 SOI 切换） |
| Laplace SOI | r = D·(m_Moon/m_Earth)^(2/5) ≈ 66,300 km | Zhang et al. 2023 Eq.1（本次未独立验证） | 仅用于物理验证（H₂ 在 Laplace SOI 边界的值） |

**注意**：Parker 中**未讨论** Laplace SOI，仅有 3BSOI（Eq 2.9）。#258 方案 §10.2 将 Laplace SOI 归于 Zhang et al. 2023，本次未独立验证。

### 10.5 Hessian 矩阵 ✅ 完全一致

| 公式 | 文献 | 代码 | 一致性 |
|------|------|------|--------|
| U_xx = 1 - (1-μ)(1/r₁³ - 3(x+μ)²/r₁⁵) - μ(1/r₂³ - 3(x-1+μ)²/r₂⁵) | Gómez §1.1.2 | `potential.py:40-44` | ✅ |
| U_zz = -(1-μ)(1/r₁³ - 3z²/r₁⁵) - μ(1/r₂³ - 3z²/r₂⁵) | Gómez §1.1.2 | `potential.py:46`（无前导 1，z 方向无离心力） | ✅ |
| 共线平动点: U_xx = 1 - 2U_zz, U_yy = 1 + 2U_zz | Gómez §1.1.3 | 代入 y=z=0 从代码公式可推导 ✓ | ✅ |

### 10.6 Kepler 能量 H₂（本方案新增，旋转系推导）

**推导**：旋转系中航天器和月球都静止（在旋转坐标的意义下），但 Kepler 能量使用惯性系速度。需要从旋转系速度和位置推导惯性系相对速度。

```
月球旋转系位置: r_moon = (1-μ, 0, 0)（静止）
月球惯性系速度: v_moon = ω × r_moon = (0, 1-μ, 0)
航天器惯性系速度: v_sc = v_rot + ω × r = (vx - y, vy + x, vz)
相对惯性系速度: v_rel = v_sc - v_moon = (vx - y, vy + x - (1-μ), vz)
```

**文献对照**：Belbruno 2010 Eq 2.8 `H₂ = ½v² - μ/r`，此处 v 为航天器相对月球的惯性系速度。从旋转系推导得上式。

**数值验证**：月球近表面圆轨道（r_rel ≈ 0.00452 DU）：
- v_circ = √(μ/r_rel) = √(0.01215/0.00452) ≈ 1.64
- H₂ = ½ × 1.64² - 0.01215/0.00452 ≈ -1.34 < 0 ✓（束缚轨道）

---

## 11. 待确认事项

> ✅ 已确认，❌ 需修改，⏳ 待后续

1. ✅ **搜索方法**：接受 BCR4BP 前向传播 + H₂ 筛选作为 MVP，Belbruno 反向积分作为后续增强。
2. ✅ **与 LGA 共享层**：接受不抽取共享搜索层，共享编排器基础设施。
3. ✅ **搜索网格与并行化**：需更细网格或自适应搜索；搜索**必须并行计算**（`ProcessPoolExecutor`，充分利用多核 CPU）。
4. ✅ **测试策略**：CI 不执行 test；release 用粗网格（10×10）test；提交前在本机完成完整 test（细网格）。
5. ✅ **TransferType.WSB 枚举**：已添加至 `data/templates/enums.py`（`WSB = "wsb"`）。
6. ✅ **优先级排序**：接受排在 propagate_with_stm 和 #280/#279 之后。
7. ✅ **Jacobi 常数约定文档**：已在 `cr3bp_system.py` 的 `get_jacobi_constant` docstring 中标注 Parker 约定及与 Belbruno/Gómez 的偏移量。
8. ✅ **#258 方案 §10.1 修正**：已修正 Jacobi 约定差异的解释（从错误的"坐标系不同"改为正确的"伪势定义约定不同"）。

---

## 附录 A：增强方案——Belbruno 反向积分搜索（记录，不阻塞 MVP）

作为后续增强，可实现 Belbruno 1993 的原始反向积分方法：

```
1. 定义捕获态：月球近月点（高度 h，偏心率 e < 1）
2. 增加偏心率 e → e + δ
3. 从捕获态反向积分（BCR4BP，负时间方向）
4. 检测地球近地点穿越（PoincareSection.periapsis("earth")）
5. 计算出发 Δv（近地点速度 vs 停泊轨道速度）
6. 最小化 Δv（调整 h, e, δ, TOF, 太阳相位）
```

优势：更高效地定位 WSB 走廊（从捕获态出发，动力学自然找到走廊）。
代价：需新增捕获态参数化 + 反向传播基础设施。

**建议**：MVP 用前向搜索验证 WSB 功能；后续版本替换为反向积分以提升搜索效率。
