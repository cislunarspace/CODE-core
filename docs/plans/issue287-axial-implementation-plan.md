# Issue #287 实施计划：Axial 轨道族（修订版）

> **状态**：待审查
> **日期**：2026-08-03
> **关联**：[Issue #287](https://github.com/cislunarspace/e2m2e/issues/287)

---

## 1. 背景与物理定义

### 1.1 Axial 轨道 = Gómez Type B 分岔族

Axial 轨道是从 planar Lyapunov 轨道通过 **Type B pitchfork 分岔**产生的
3D 周期轨道族（Gómez vol I, line 665）：

- **对称性**：关于 x₁ 轴对称
- **分岔条件**：a_v=1, b_v=0（垂直临界轨道）
- **存在平动点**：所有 L_i（L1-L5）
- **Jacobi 范围**：窄（L1: 2.991-3.021, L2: 2.967-3.014）
- **稳定性**：全部不稳定，无准周期邻域

### 1.2 与 Halo 的关系

| | Halo (Type A) | Axial (Type B) |
|--|---------------|----------------|
| 对称面 | (x₁, x₃) 平面 | x₁ 轴 |
| 分岔条件 | a_v=1, c_v=0 | a_v=1, b_v=0 |
| 初始条件 | (x₀, 0, z₀, 0, ẏ₀, 0) | (x₀, 0, z₀, 0, ẏ₀, ż₀) |
| 半周期条件 | y=0, ẋ=0, ż=0 | y=0, ẋ=0, z→-z, ż→-ż |

---

## 2. 变更清单

### 2.1 新建文件

| # | 文件 | 说明 | 行数估计 |
|---|------|------|----------|
| A | `e2m2e/algorithm/family/axial_initial_guess.py` | Axial 初猜模块 | ~100 |
| B | `tests/algorithms/test_axial_initial_guess.py` | 初猜单元测试 | ~100 |
| C | `tests/algorithms/test_axial_family.py` | 端到端 + 物理不变量测试 | ~120 |

### 2.2 修改文件

| # | 文件 | 变更 | 行数估计 |
|---|------|------|----------|
| D | `e2m2e/algorithm/family/cr3bp_orbits.py` | 新增 `_correct_axial()` + `design_axial()` | ~100 |
| E | `e2m2e/algorithm/family/strategies/axial.py` | Axial 修正策略（新策略文件） | ~60 |
| F | `e2m2e/algorithm/family/strategies/__init__.py` | 导出新策略 | ~5 |
| G | `e2m2e/algorithm/family/__init__.py` | 注册表 + 导出 | ~15 |
| H | `tests/architecture/test_placeholder.py` | 更新 placeholder | ~5 |

---

## 3. 分步实施

### Step 1：Axial 修正策略（文件 E+F）

**文件**：`e2m2e/algorithm/family/strategies/axial.py`

Axial 轨道关于 x 轴对称（Type B），需要新的修正策略：

```python
"""Axial 轨道修正策略（Type B 对称性）。"""

def axial_fixed_z0(z0: float, libration_point: int = 1) -> CorrectionConfig:
    """固定 z0 的 Axial 轨道修正。

    Axial 轨道关于 x 轴对称（Type B）：
    - 初始条件：(x₀, 0, z₀, 0, ẏ₀, ż₀)
    - 半周期后：y=0, ẋ=0（z 和 ż 自动反号）

    自由变量：[x₀, ẏ₀, ż₀, T_half]（4 个）
    约束：[y(T/2)=0, ẋ(T/2)=0]（2 个）
    → 2 维族（固定 z₀ 后）
    """
    return CorrectionConfig(
        setup_type="axial_fixed_z0",
        symmetry_condition="x_axis",
        fixed_parameters={"z0": z0, "libration_point": libration_point},
        free_variables=["x0", "y_dot0", "z_dot0", "T_half"],
        free_variable_indices=[0, 4, 5, 6],
        target_conditions={"y": 0.0, "x_dot": 0.0},
        constraint_indices=[1, 3],
        constraint_weights={"y": 1.0, "x_dot": 1.0},
        constraint_types={"y": "equality", "x_dot": "equality"},
    )
```

**注意**：4 自由变量 vs 2 约束 = 欠定系统。需要用最小二乘或额外约束。
可能需要固定更多参数（如固定 x₀ 和 z₀，只自由 ẏ₀, ż₀, T_half）。

### Step 2：Axial 初猜模块（文件 A）

**文件**：`e2m2e/algorithm/family/axial_initial_guess.py`

从 planar Lyapunov 轨道的 Type B 垂直临界点出发：

```python
"""Axial 轨道初猜模块。

从 planar Lyapunov 轨道的 Type B 垂直临界点出发，
施加 z 方向扰动构造 Axial 轨道初猜。

Gómez vol I, line 665: Type B = 关于 x₁ 轴对称，
分岔条件 a_v=1, b_v=0。
"""
```

**方法**：
1. 用 `_linear_modes()` 获取共线点的面内/面外模态
2. 从 Lyapunov 种子出发，沿族行走找到垂直临界轨道（a_v=1, b_v=0）
3. 在垂直临界轨道上施加小 z 扰动（含 ż₀）
4. 作为 Axial 族的种子

### Step 3：设计函数（文件 D）

**文件**：`e2m2e/algorithm/family/cr3bp_orbits.py`

```python
def design_axial(
    collinear_point: int,
    amplitude_z_km: float,
    *,
    dynamics: CR3BP_Dynamics | None = None,
) -> Orbit:
    """生成指定 z 振幅的 Axial 周期轨道。

    Axial 轨道从 planar Lyapunov 的 Type B 垂直临界点分岔。
    z 振幅定义为轨道在 z 方向上的最大偏移（km）。
    """
```

**实现思路**：
1. 构造地月系统
2. 用 `axial_initial_guess` 找到 Type B 分岔种子
3. 用 `_walk_family` 或 PAL 延拓沿族走到目标 z 振幅
4. 注册到 registry

### Step 4：注册表与导出（文件 G）

同原计划：`"AXIAL": design_axial` 加入 `_build_registry`。

### Step 5：测试（文件 B+C+H）

- 初猜单元测试：对称性验证、z 振幅一致性
- 端到端测试：convergence、period > 0
- 物理不变量：Jacobi 守恒、x 轴对称、周期闭合
- 注册表测试

---

## 4. 风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| Type B 分岔点难以精确找到 | 中 | 先用 PAL 延拓扫描 a_v，找到 a_v≈1 且 b_v≈0 的轨道 |
| 修正策略欠定（4 自由变量 vs 2 约束） | 高 | 固定 x₀ 和 z₀，只自由 ẏ₀, ż₀, T_half |
| Jacobi 范围窄，延拓步长需很小 | 中 | 用小步长 PAL（step_size=0.001） |
| L3/L4/L5 Axial 的 Jacobi 范围未知 | 低 | 首期只做 L1/L2 |

---

## 5. 验证矩阵

| 验证项 | 方法 | 通过标准 |
|--------|------|----------|
| 初猜对称性 | 单元测试 | y(0)≈0, ẋ(0)≈0 |
| 微分修正收敛 | 端到端 | 小 z 振幅收敛 |
| Jacobi 守恒 | 物理不变量 | 漂移 < 1e-10 |
| x 轴对称 | 物理不变量 | state(T/2) 满足 z→-z, ż→-ż |
| 周期闭合 | 物理不变量 | 闭合误差 < 1e-8 |
| 注册表 | 集成测试 | `registry["AXIAL"]` 可用 |

---

## 6. 依赖关系

```
Step 1 (axial strategy: axial.py)
    ↓
Step 2 (axial_initial_guess.py)
    ↓
Step 3 (cr3bp_orbits.py: design_axial)
    ↓
Step 4 (__init__.py: registry)
    ↓
Step 5 (tests)
```

---

## 7. 代码行数估计

| 变更 | 新增 | 修改 |
|------|------|------|
| axial.py (策略) | ~60 | - |
| axial_initial_guess.py | ~100 | - |
| cr3bp_orbits.py | ~100 | ~2 |
| strategies/__init__.py | ~5 | ~2 |
| family/__init__.py | ~10 | ~5 |
| test_axial_initial_guess.py | ~100 | - |
| test_axial_family.py | ~120 | - |
| test_placeholder.py | ~3 | ~2 |
| **合计** | **~498** | **~11** |
