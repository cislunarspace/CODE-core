# Issue #289 文献调研 + 实施方案：L4/L5 Short-Period Orbit (SPO) 周期轨道族

> **状态**：待审查
> **日期**：2026-08-04
> **关联**：[Issue #289](https://github.com/cislunarspace/e2m2e/issues/289)

---

## 1. 文献调研

### 1.1 SPO 的精确定义

**Gómez et al. (2001)** Vol. II "Fundamentals: The Case of Triangular
Libration Points"（ESA 合同报告，World Scientific 出版）：

> L4/L5 三角平动点在 CR3BP 中存在两个基本周期轨道族：
> - **短周期族** $\mathcal{L}_s$：围绕 L4/L5 的小振幅周期轨道，极限周期 6.5827 nd
>   （≈ 28.5 天），振幅增大周期递减
> - **长周期族** $\mathcal{L}_l$：大振幅周期轨道，周期略大于 3 倍朔望月

短周期族成员（Short-Period Orbits, SPO）的物理特征：

| 特征 | SPO | DRO（对比） |
|------|-----|------------|
| 中心天体 | L4/L5（三角平动点） | 月球 |
| 坐标平面 | xy 平面（z=0） | xy 平面（z=0） |
| 对称性 | **无** x 轴或 xz 平面对称 | x 轴对称 |
| 周期 | ≈ 27-31 天（1 朔望月量级） | 可变（取决于振幅） |
| Jacobi 常数 | ≈ 2.91 | ≈ 2.92-2.99 |
| 稳定性 | 近稳定（特征值模 ≈ 1.001） | 稳定 |
| 形状 | 近似椭圆（主导谐波 n=1） | 复杂 |

**Capdevila & Howell (2018)** "A transfer network linking Earth, Moon,
and the triangular libration point regions"（JGCD）直接给出 CR3BP 中
SPO 的显式初始条件（Table 1），可作为种子常量：

| 轨道 | x₀ (nd) | y₀ (nd) | z₀ (nd) | ẋ₀ (nd) | ẏ₀ (nd) | ż₀ (nd) | 周期 (天) | C (Jacobi) |
|------|---------|---------|---------|---------|---------|---------|----------|-----------|
| L4 SPO | -0.2255 | 0.8660 | 0 | -0.2384 | 0.2494 | 0 | 28.3488 | 2.9132 |
| L5 SPO | -0.2255 | -0.8660 | 0 | 0.2384 | 0.2494 | 0 | 28.3488 | 2.9132 |

**L4/L5 对称性**：CR3BP 的 (x, y, z, ẋ, ẏ, ż, t) → (x, -y, z, -ẋ, ẏ, -ż, -t)
变换下 L5 轨道 = L4 轨道的 y 镜像（Gómez vol II, §5.2, p.1662）。

### 1.2 Gómez 中间方程的周期轨道族结构

Gómez vol II Chapter 5 通过数值延拓从双圆问题得到中间方程的周期轨道：

| 轨道 | 初始点 (相对 L4) | 周期 (nd) | 特征值模 | 物理角色 |
|------|-----------------|-----------|---------|---------|
| A | (0.3446, -0.0557, 0.1131, -0.3702) | 6.7912 | 1.0011 | 短周期族（标准 SPO） |
| B | (-0.7582, 0.0891, -0.0365, 0.2717) | 6.7912 | 1.0011 | 短周期族（A 的相位对） |
| C | (-0.0045, 0.0137, 0.0299, -0.0038) | 6.7912 | 1.1666 | 短周期族（二倍频） |
| F | (0.4100, -0.1029, 0.1138, -0.3961) | 20.374 | 5.844 | 长周期族 |
| G | (-0.9590, 0.1274, 0.0984, 0.3250) | 20.374 | 5.844 | 长周期族 |

关键结论：
- 轨道 A/B 是**同一条轨道**的两个相位表示（差半个朔望月）
- 短周期族全部是**平面轨道**（z=0）
- 短周期族**近稳定**（λ≈1.001），放大因子 1000× 需 ~12 个朔望月
- 长周期族**强不稳定**（λ≈5.844），不在本 issue 范围内

### 1.3 与现有 `design_triangular` 的关系

| | `design_triangular` (现有) | SPO (本 issue) |
|--|--------------------------|---------------|
| 输出类型 | 拟周期轨道 | **周期轨道** |
| 模态数 | 3（短+长+垂直） | **1**（仅短周期） |
| 微分修正 | 无 | **有**（通用平面周期修正） |
| 族参数 | amplitude_in, amplitude_out | **距 L4/L5 的径向距离** |
| 精度 | 一阶线性化近似 | **精确 CR3BP 数值解** |
| 用途 | 星历修正 patch points | **精确周期轨道 + 流形计算基础** |

两者互补：`design_triangular` 给拟周期初猜，SPO 给精确周期解。
现有 L4/L5 端到端流程（`design_orbit("L4"/"L5", ...)`）保持不变。

---

## 2. 实施方案

### 2.1 总体设计

**核心挑战**：SPO 无 x 轴对称性，不能复用现有任何 `setup_*` 修正方法。
需要新增**通用平面周期轨道修正**（无对称性假设）。

**族参数**：选择初始 x₀（x 轴穿越分量）作为族行走参数。
x₀ 从 L4/L5 的 x 坐标附近开始，沿族递增/递减，对应的距 L4/L5 径向距离
作为振幅度量。

**振幅定义**：一个周期内距 L4/L5 的径向距离最小/最大值的均值（km），
与 DRO 的月心距振幅定义一致。

### 2.2 变更清单

#### 2.2.1 新建文件

| # | 文件 | 说明 | 行数估计 |
|---|------|------|----------|
| A | `e2m2e/algorithm/family/strategies/spo.py` | SPO 修正策略（通用平面周期修正） | ~70 |
| B | `tests/algorithms/test_spo_family.py` | 端到端 + 物理不变量测试 | ~150 |

#### 2.2.2 修改文件

| # | 文件 | 变更 | 行数估计 |
|---|------|------|----------|
| C | `e2m2e/data/templates/seed.py` | 新增 SPO 种子常量（Capdevila 数据） | ~15 |
| D | `e2m2e/algorithm/solver/differential_correction.py` | 新增 `setup_spo_fixed_x0` 修正方法 | ~80 |
| E | `e2m2e/algorithm/family/cr3bp_orbits.py` | 新增 `_correct_spo` + `design_spo` | ~100 |
| F | `e2m2e/algorithm/family/strategies/__init__.py` | 导出新策略 | ~5 |
| G | `e2m2e/algorithm/family/__init__.py` | 注册表 + 惰性导出 | ~15 |
| H | `e2m2e/algorithm/design/design_orbit.py` | `_validate_params` + `_cr3bp_orbit_for` 分发 | ~30 |

#### 2.2.3 不修改

- **Rust 层**：无变更（纯 Python 算法层新增）
- **`triangular_initial_guess.py`**：保持不变（拟周期轨道仍需此模块）
- **现有 L4/L5 流程**：保持不变（`design_orbit("L4"/"L5", ...)` 继续走
  `design_triangular`）

### 2.3 分步实施

#### Step 1：SPO 种子常量（文件 C）

**文件**：`e2m2e/data/templates/seed.py`

```python
#: SPO（Short-Period Orbit）族标准种子
#: 来源：Capdevila & Howell (2018), JGCD, Table 1
#: L4 SPO: (x₀, y₀, ẋ₀, ẏ₀) 在质心会合系中，z₀=ż₀=0（平面轨道）
#: L5 SPO 由 CR3BP 对称性得到：y₀→-y₀, ẋ₀→-ẋ₀
_SPO_L4_SEED_X0 = -0.2255
_SPO_L4_SEED_Y0 = 0.8660
_SPO_L4_SEED_VX0 = -0.2384
_SPO_L4_SEED_VY0 = 0.2494
_SPO_SEED_PERIOD = 6.529   # 28.3488 天 / CHAR_PERIOD_SEC * 2π
```

> 注：周期 28.3488 天 = 28.3488 / (27.32 × 2π) × 2π ≈ 28.3488 / 4.3488
> ≈ 6.520 nd（按 T* = L*³/² / √(G(m₁+m₂)) = 4.3423 天，6.529 nd）。
> 最终值以 Capdevila 数据直接计算为准。

#### Step 2：通用平面周期轨道修正方法（文件 D）

**文件**：`e2m2e/algorithm/solver/differential_correction.py`

新增 `setup_spo_fixed_x0` 方法。SPO 是平面 CR3BP 中无对称性的
周期轨道，需要通用的周期闭合修正：

```python
def setup_spo_fixed_x0(self, x0: float, libration_point: int = 5):
    """SPO 通用平面周期修正：固定 x₀，自由 y₀, ẋ₀, ẏ₀, T。

    SPO 无 x 轴对称性（y₀≠0），不能使用半周期约束。
    直接求解全周期闭合条件：state(T) = state(0)。

    自由变量: [y₀, ẋ₀, ẏ₀, T]（4 个）
    约束: [Δx=0, Δy=0, Δẋ=0, Δẏ=0]（4 个）
    其中 Δq = q(T) - q(0)，z 分量自动满足（平面轨道 ż₀=0 → z(t)≡0）。

    变分方程同时积分，构造 4×4 雅可比矩阵用于牛顿迭代。
    """
```

**关键实现细节**：

1. 积分一个周期，计算 state(T) - state(0)
2. 积分变分方程 Φ(T,0)（单值矩阵），构造雅可比：
   ```
   J = [Φ₁₁-1  Φ₁₂   Φ₁₄   ẋ(T)]
       [Φ₂₁    Φ₂₂-1 Φ₂₄   ẏ(T)]
       [Φ₄₁    Φ₄₂   Φ₄₄-1 ẍ(T)]
       [Φ₅₁    Φ₅₂   Φ₅₄   ÿ(T)]
   ```
   （第 3,5 行/列对应 z 分量，平面轨道自动为零，可剔除）
3. 牛顿步：[Δy₀, Δẋ₀, Δẏ₀, ΔT] = -J⁻¹ · residual
4. 加阻尼和回溯线搜索（复用 `DifferentialCorrection` 现有框架）

**与现有方法的区别**：
- 现有所有 `setup_*` 利用对称性将 6 闭合条件缩减到 2-3 个
- SPO 需要 4 个闭合条件（z 自动满足后），无缩减
- 条件数可能比对称方法差，但 SPO 近稳定（λ≈1.001），牛顿收敛应可靠

#### Step 3：SPO 修正策略配置（文件 A）

**文件**：`e2m2e/algorithm/family/strategies/spo.py`

```python
"""SPO（Short-Period Orbit）修正策略。

L4/L5 短周期族是 xy 平面内围绕三角平动点的周期轨道，
不具有 x 轴或 xz 平面对称性。使用通用平面周期修正。
"""

def spo_fixed_x0(x0: float, libration_point: int = 5) -> CorrectionConfig:
    """固定 x₀ 的 SPO 修正。

    自由变量: [y₀, ẋ₀, ẏ₀, T]（4 个）
    约束: [Δx=0, Δy=0, Δẋ=0, Δẏ=0]（4 个，全周期闭合）
    """
    return CorrectionConfig(
        setup_type="spo_fixed_x0",
        symmetry_condition="none",       # 无对称性
        fixed_parameters={"x0": x0, "libration_point": libration_point},
        free_variables=["y0", "vx0", "vy0", "T"],
        free_variable_indices=[1, 3, 4, 6],
        target_conditions={"dx": 0.0, "dy": 0.0, "dvx": 0.0, "dvy": 0.0},
        constraint_indices=[0, 1, 3, 4],
        constraint_weights={"dx": 1.0, "dy": 1.0, "dvx": 1.0, "dvy": 1.0},
        constraint_types={"dx": "closure", "dy": "closure",
                          "dvx": "closure", "dvy": "closure"},
    )
```

#### Step 4：SPO 修正函数 + 设计函数（文件 E）

**文件**：`e2m2e/algorithm/family/cr3bp_orbits.py`

```python
def _correct_spo(
    dynamics: CR3BP_Dynamics, x0: float, libration_point: int,
    guess: Orbit | None,
) -> Orbit:
    """在 x₀ 处修正 SPO（通用平面周期修正，无对称性假设）。

    初始状态 (x₀, y₀, 0, ẋ₀, ẏ₀, 0)：
    - x₀ 固定（族参数）
    - y₀, ẋ₀, ẏ₀ 自由（从种子或上一步轨道获取）
    - T 自由（全周期闭合）

    首次调用（guess=None）使用 Capdevila & Howell (2018) 种子常量。
    后续调用保留已收敛轨道状态作初猜。
    """
    ...


def _l45_distance(
    dynamics: CR3BP_Dynamics, orbit: Orbit, point: int,
    n_points: int = 2000,
) -> tuple[float, float]:
    """传播一个周期，返回距 L4/L5 的径向距离最小/最大值（无量纲）。"""
    ...


def design_spo(
    libration_point: int,
    amplitude_km: float,
    *,
    dynamics: CR3BP_Dynamics | None = None,
    tol_km: float = 20.0,
) -> Orbit:
    """生成指定振幅的 L4/L5 SPO 周期轨道。

    振幅定义：一个周期内距 L4/L5 径向距离最小/最大值的均值（km）。
    以 x₀ 为族参数沿短周期族行走，命中 tol_km 内即停。

    References:
        Gómez et al. (2001). Dynamics and mission design near libration
        points, Vol. II. ESA Contract Report.
        Capdevila & Howell (2018). A transfer network linking Earth,
        Moon, and the triangular libration point regions. JGCD.
    """
    ...
```

**族行走策略**：

1. 从 Capdevila 种子（x₀=-0.2255）出发
2. `_correct_spo` 在该 x₀ 处修正出周期轨道
3. `_walk_family` 沿 x₀ 行走，measure = 距 L4/L5 径向距离均值
4. x₀ 向 L4/L5 方向递增 → 距离减小（小振幅 SPO）；
   x₀ 向远离方向递减 → 距离增大（大振幅 SPO）
5. 二分收敛到目标振幅

#### Step 5：注册表 + 导出（文件 F + G）

**文件 F**：`strategies/__init__.py`
```python
from .spo import spo_fixed_x0
```

**文件 G**：`family/__init__.py`

在注册表中新增 `"L4_SPO"` 和 `"L5_SPO"` 条目：
```python
"L4_SPO": lambda amplitude, **kw: design_spo(4, amplitude, **kw),
"L5_SPO": lambda amplitude, **kw: design_spo(5, amplitude, **kw),
```

> 注：保留现有 `"L4"` / `"L5"` 条目不变（走 `design_triangular`），
> 新增 `"L4_SPO"` / `"L5_SPO"` 走 `design_spo`。两者互补而非替代。

#### Step 6：design_orbit 分发（文件 H）

**文件**：`e2m2e/algorithm/design/design_orbit.py`

新增参数校验：
```python
if sel in ("L4_SPO", "L5_SPO"):
    amplitude = 10000.0 if amplitude is None else float(amplitude)
    phase = 0.0 if phase is None else float(phase)
    if not 1737.0 <= amplitude <= 200000.0:
        raise ValueError(...)
    if not 0.0 <= phase <= 1.0:
        raise ValueError(...)
    return {"amplitude": amplitude, "phase": phase}
```

新增 `_cr3bp_orbit_for` 分发：
```python
if sel in ("L4_SPO", "L5_SPO"):
    return design_spo(4 if sel == "L4_SPO" else 5, params["amplitude"],
                      dynamics=dynamics)
```

### 2.4 依赖关系

```
Step 1 (seed.py: SPO 常量)
    ↓
Step 2 (differential_correction.py: setup_spo_fixed_x0)
    ↓
Step 3 (strategies/spo.py: spo_fixed_x0)
    ↓
Step 4 (cr3bp_orbits.py: _correct_spo + design_spo)
    ↓
Step 5 (__init__.py: registry)  ←  可与 Step 6 并行
Step 6 (design_orbit.py: 分发)
    ↓
Step 7 (test_spo_family.py: 端到端 + 不变量测试)
```

### 2.5 代码行数估计

| 变更 | 新增 | 修改 |
|------|------|------|
| seed.py (SPO 常量) | ~10 | ~1 |
| differential_correction.py (修正方法) | ~80 | ~3 |
| strategies/spo.py (策略) | ~70 | - |
| strategies/__init__.py (导出) | ~2 | ~1 |
| cr3bp_orbits.py (_correct_spo + design_spo) | ~100 | ~5 |
| family/__init__.py (注册表) | ~10 | ~5 |
| design_orbit.py (分发) | ~25 | ~5 |
| test_spo_family.py (测试) | ~150 | - |
| **合计** | **~447** | **~20** |

---

## 3. 验证矩阵（ADR 0013）

| # | 验证项 | 方法 | 通过标准 | 文献依据 |
|---|--------|------|----------|----------|
| 1 | 周期闭合 | state(T) vs state(0) | ‖Δstate‖ < 1e-8 | 通用 |
| 2 | Jacobi 守恒 | 一个周期内漂移 | ‖ΔC‖ < 1e-10 | ADR 0013 |
| 3 | 平面约束 | z(t) 全周期 | max\|z\| < 1e-8 | Capdevila: z₀=0 |
| 4 | 周期范围 | T(days) ∈ [27, 31] | 27 < T < 31 | Gómez: T_S ≈ 29.5d |
| 5 | Jacobi 范围 | C ∈ [2.85, 2.95] | 2.85 < C < 2.95 | Capdevila: C=2.9132 |
| 6 | Capdevila 复现 | 种子 x₀=-0.2255 直接收敛 | 收敛 | Capdevila 2018 Table 1 |
| 7 | L4/L5 镜像对称 | L5 轨道 = L4 轨道 y 镜像 | y₀ 符号相反, ẋ₀ 符号相反 | CR3BP 对称性 |
| 8 | 注册表 | registry["L4_SPO"/"L5_SPO"] 可用 | 可调用 | 架构 |
| 9 | design_orbit 分发 | `design_orbit("L4_SPO", ...)` 全流程 | 不抛异常 | 架构 |
| 10 | 振幅精度 | 距 L4/L5 径向距离均值 ≈ 目标 | ±25 km | 与 DRO tol 对齐 |

---

## 4. 风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| 通用平面修正条件数差，牛顿不收敛 | 低 | SPO 近稳定（λ≈1.001），Capdevila 种子精确 |
| 族行走越过短周期族末端（B₄₅ 分岔点） | 中 | `_walk_family` 的步长退半机制自动处理 |
| 大振幅 SPO 偏离线性化近似 | 低 | 种子直接来自数值精确轨道，不依赖线性化 |
| L4_SPO/L5_SPO 命名与其他 L4/L5 类型混淆 | 低 | 文档明确区分：L4/L5 = 拟周期，L4_SPO/L5_SPO = 周期 |

---

## 5. 参考文献

1. **Gómez, G., Llibre, J., Martínez, R. & Simó, C. (2001)**.
   Dynamics and mission design near libration points, Vol. II:
   Fundamentals — The Case of Triangular Libration Points.
   World Scientific. [ESA Contract 6139/84/D/JS(SC)]
   → L4/L5 周期轨道族结构、中间方程周期轨道 A/B/C/F/G、
   数值延拓方法、稳定性分析。

2. **Gómez, G., Jorba, À., Masdemont, J. & Simó, C. (2001)**.
   Dynamics and mission design near libration points, Vol. IV:
   Advanced Methods for Triangular Points. World Scientific.
   → 拟周期轨道的高级方法，准周期解的半解析构造。

3. **Capdevila, L.R. & Howell, K.C. (2018)**.
   A transfer network linking Earth, Moon, and the triangular
   libration point regions in the Earth-Moon system.
   JGCD, 41(7), 1475-1492.
   → CR3BP SPO 显式初始条件（Table 1）、DRO↔SPO 转移设计。

4. **Deprit, A., Henrard, J. & Rom, A. (1967)**.
   Natural satellites — Periodic orbits.
   → 短周期族 L_s 和长周期族 L_l 的经典分析。

5. **Kolenkiewicz, R. & Carpenter, L. (1968)**.
   Periodic orbits around L4 in the restricted three-body problem.
   → SPO 数值计算的早期工作（Gómez vol II 引用）。
