---
title: E2M2E 技术文档
---

# E2M2E 技术文档

> 本文档为 Earth-to-Moon-to-Earth (E2M2E) 轨道力学库的完整技术文档，涵盖所有核心类的设计原理、数学基础、API接口和使用指南。

---

---

## 1. 核心模块

### 1.1 CR3BP_System

**文件**: `e2m2e/core/system.py`

**类签名**:
```python
class CR3BP_System:
    """圆型限制性三体问题系统参数"""
```

#### 设计原理

`CR3BP_System` 类封装了圆型限制性三体问题 (Circular Restricted Three-Body Problem) 的系统参数。在CR3BP模型中：
- 两个大质量天体（主天体 $m_1$ 和次天体 $m_2$）在它们相互的引力作用下围绕共同的质心做圆轨道运动
- 一个小质量天体（探测器）在上述两个大天体的引力场中运动，其质量对两个大天体的运动没有影响

质量参数定义为：
$$\mu = \frac{m_2}{m_1 + m_2}$$

地月系统的 $\mu \approx 0.01215$

#### 数学基础

**平动点（Libration Points）计算**：
平动点是相对于两个大天体保持静止的特殊点，满足：
$$\nabla U(\mathbf{r}) = 0$$

其中 $U$ 是有效势函数：
$$U = \frac{x^2 + y^2}{2} + \frac{1-\mu}{r_1} + \frac{\mu}{r_2}$$

**Jacobi常数**：
$$C = 2U - v^2 = x^2 + y^2 + \frac{2(1-\mu)}{r_1} + \frac{2\mu}{r_2} - (v_x^2 + v_y^2 + v_z^2)$$

#### 属性

| 属性名 | 类型 | 说明 |
|--------|------|------|
| `mu` | `float` | 质量参数 $\mu = m_2/(m_1+m_2)$ |
| `primary_body` | `str` | 主天体名称 |
| `secondary_body` | `str` | 次天体名称 |
| `L1-L5` | `np.ndarray` | 五个平动点的坐标 |
| `characteristic_length` | `float` | 特征长度（两天体间距离） |
| `characteristic_time` | `float` | 特征时间 |
| `characteristic_velocity` | `float` | 特征速度 |

#### 核心方法

| 方法 | 说明 |
|------|------|
| `compute_libration_points()` | 计算五个平动点位置 |
| `get_libration_point(point)` | 获取指定平动点坐标 |
| `get_jacobi_constant(state)` | 计算Jacobi常数 |
| `dimensionless_to_physical(state)` | 无量纲→物理单位 |
| `physical_to_dimensionless(state)` | 物理单位→无量纲 |
| `compute_stability_index(L_point)` | 计算平动点稳定性指标 |

#### 使用示例

```python
from e2m2e.core.system import CR3BP_System, LibrationPoint

# 从已知系统创建
system = CR3BP_System.from_known_system("earth_moon")
system.set_characteristic_scales(distance=384400, period=27.32*86400)
system.compute_libration_points()

# 获取平动点
L1 = system.get_libration_point(LibrationPoint.L1)
print(f"L1位置: {L1}")

# 计算Jacobi常数
state = np.array([0.8, 0, 0, 0, 1.5, 0])
C = system.get_jacobi_constant(state)
```

---

### 1.2 CR3BP_Dynamics

**文件**: `e2m2e/core/dynamics.py`

**类签名**:
```python
class CR3BP_Dynamics:
    """CR3BP动力学方程"""
```

#### 设计原理

`CR3BP_Dynamics` 类封装了CR3BP的运动方程和数值积分方法。运动方程在旋转坐标系下写为（无量纲形式）：

$$\begin{cases}
\dot{x} = v_x \\
\dot{y} = v_y \\
\dot{z} = v_z \\
\dot{v}_x = 2v_y + x - \frac{(1-\mu)(x+\mu)}{r_1^3} - \frac{\mu(x-1+\mu)}{r_2^3} \\
\dot{v}_y = -2v_x + y - \frac{(1-\mu)y}{r_1^3} - \frac{\mu y}{r_2^3} \\
\dot{v}_z = -\frac{(1-\mu)z}{r_1^3} - \frac{\mu z}{r_2^3}
\end{cases}$$

其中：
$$r_1 = \sqrt{(x+\mu)^2 + y^2 + z^2}, \quad r_2 = \sqrt{(x-1+\mu)^2 + y^2 + z^2}$$

#### 核心功能

1. **状态传播**: `propagate()` 方法使用 scipy 的 `solve_ivp` 进行数值积分
2. **状态转移矩阵 (STM)**: 通过 `equations_with_stm()` 同时积分42维增广状态
3. **Jacobi常数监控**: 实时计算 Jacobi 常数用于精度检验

#### 核心方法

| 方法 | 说明 |
|------|------|
| `equations_of_motion(t, state)` | 6维运动方程 |
| `equations_with_stm(t, augmented_state)` | 42维增广运动方程（含STM） |
| `propagate(initial_state, t_span, with_stm=False)` | 传播轨迹 |
| `compute_state_transition_matrix(initial_state, t)` | 计算STM |
| `compute_jacobi_constant(state)` | 计算Jacobi常数 |
| `check_cross_section(state, plane, value)` | 检测截面穿越 |

#### 使用示例

```python
from e2m2e.core.dynamics import CR3BP_Dynamics

dynamics = CR3BP_Dynamics(system)

# 简单传播
result = dynamics.propagate(initial_state, t_span=(0, 10), t_eval=np.linspace(0, 10, 1000))

# 带STM传播（用于微分修正）
result_with_stm = dynamics.propagate(initial_state, t_span=(0, T), with_stm=True)
stm = result_with_stm['stm'][-1]  # 最终时刻的STM
```

---

### 1.3 Orbit

**文件**: `e2m2e/core/orbit.py`

**类签名**:
```python
class Orbit:
    """轨道数据和处理"""
```

#### 设计原理

`Orbit` 类表示CR3BP中的一条完整轨道。轨道数据包括：
- **states**: 状态序列，形状为 `(n, 6)`，每行为 `[x, y, z, vx, vy, vz]`
- **times**: 对应的时间序列

#### 轨道周期检测

轨道周期通过检测 x 分量的过零点来估计：

1. 找到 x 坐标相对于轨道中心 $\bar{x}$ 的符号变化点
2. 使用相邻两个过零点的时间差 $\Delta t$ 估计半周期：$T_{half} = \Delta t$
3. 完整周期：$T = 2 \times T_{half}$

#### 单值矩阵与稳定性

**单值矩阵 (Monodromy Matrix)**: 沿闭合轨道积分一个周期得到的状态转移矩阵：
$$M = \Phi(T, 0)$$
其中 $\Phi$ 是状态转移矩阵。

**稳定性判定**：
- 若所有Floquet乘子（STM特征值）$\lambda_i$ 满足 $|\lambda_i| = 1$，轨道线性稳定
- 若存在 $|\lambda_i| > 1$，轨道不稳定

#### 核心属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `states` | `np.ndarray` | 状态序列 (n, 6) |
| `times` | `np.ndarray` | 时间序列 (n,) |
| `period` | `float` | 轨道周期 |
| `amplitudes` | `dict` | 各方向振幅 |
| `monodromy_matrix` | `np.ndarray` | 单值矩阵 (6, 6) |
| `eigenvalues` | `np.ndarray` | 特征值 |
| `stability` | `str` | 稳定性标签 |
| `is_periodic` | `bool` | 是否为周期轨道 |

#### 核心方法

| 方法 | 说明 |
|------|------|
| `compute_monodromy_matrix(dynamics)` | 计算单值矩阵 |
| `compute_stability(dynamics)` | 计算稳定性 |
| `interpolate_at_time(t)` | 时间插值 |
| `get_amplitude(direction)` | 获取振幅 |
| `save_to_file(filename)` | 保存到文件 |
| `load_from_file(filename)` | 从文件加载 |

---

### 1.4 OrbitFamily

**文件**: `e2m2e/core/orbit.py`

**类签名**:
```python
class OrbitFamily:
    """轨道族容器"""
```

#### 设计原理

`OrbitFamily` 用于存储和管理一族相关的轨道（如同一族Halo轨道、Lyaounov轨道等）。支持批量操作如获取所有初始状态、周期、Jacobi常数等。

#### 核心属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `orbits` | `List[Orbit]` | Orbit对象列表 |
| `family_type` | `str` | 轨道族类型 |
| `system` | `CR3BP_System` | 关联的系统 |
| `states` | `np.ndarray` | property: 所有初始状态 (n, 6) |
| `periods` | `np.ndarray` | property: 所有周期 (n,) |

#### 核心方法

| 方法 | 说明 |
|------|------|
| `add_orbit(orbit)` | 添加轨道 |
| `get_states()` | 获取初始状态数组 |
| `get_periods()` | 获取周期数组 |
| `get_jacobi_constants()` | 获取Jacobi常数数组 |
| `save_to_file(filename)` | 保存轨道族 |
| `load_from_file(filename)` | 加载轨道族 |

---

### 1.5 CoordinateTransformation & ReferenceFrame

**文件**: `e2m2e/core/coordinate.py`

**类签名**:
```python
class ReferenceFrame(Enum):
    ROTATING = "rotating"      # 旋转系
    INERTIAL = "inertial"     # 惯性系
    BARYCENTRIC = "barycentric"  # 质心系
    PRIMARY_CENTERED = "primary_centered"  # 主天体中心系
    SECONDARY_CENTERED = "secondary_centered"  # 次天体中心系
    SYNODIC = "synodic"        # 会合系

class CoordinateTransformation:
    """坐标系变换"""
```

#### 旋转系↔惯性系变换

旋转系（会合坐标系）到惯性系的变换矩阵为：

$$\mathbf{r}_{inertial} = R_z(\theta)^T \mathbf{r}_{rotating}$$

其中 $R_z(\theta)$ 是绕 z 轴的旋转矩阵：
$$R_z(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{pmatrix}$$

速度变换包含科里奥利项：
$$\mathbf{v}_{inertial} = R_z(\theta)^T \mathbf{v}_{rotating} + \dot{R}_z(\theta)^T \mathbf{r}_{rotating}$$

#### 核心方法

| 方法 | 说明 |
|------|------|
| `rotating_to_inertial(state, time)` | 旋转系→惯性系 |
| `inertial_to_rotating(state, time)` | 惯性系→旋转系 |
| `barycentric_to_primary(state)` | 质心系→主天体中心 |
| `primary_to_barycentric(state)` | 主天体中心→质心系 |
| `compute_rotation_matrix(time)` | 计算旋转矩阵 |

---

## 2. 算法模块

### 2.1 DifferentialCorrection

**文件**: `e2m2e/algorithms/differential_correction.py`

**类签名**:
```python
class DifferentialCorrection:
    """微分修正算法"""
```

#### 设计原理

微分修正算法用于求解周期轨道的初始条件问题。核心思想是：
1. 假设初始条件 $\mathbf{x}_0$ 和半周期 $T/2$ 作为自由变量
2. 积分得到终点状态 $\mathbf{x}(T/2)$
3. 构造约束方程 $\mathbf{F}(\mathbf{x}_0, T/2) = 0$
4. 使用牛顿迭代求解：
$$\begin{pmatrix} \delta\mathbf{x}_0 \\ \delta(T/2) \end{pmatrix} = -J^{-1} \mathbf{F}$$

其中 $J$ 是约束方程对自由变量的雅可比矩阵。

#### 支持的对称性配置

| 配置类型 | 自由变量 | 目标约束 |
|---------|---------|---------|
| `2D_symmetric_x_fixed_x0` | $[v_{y0}, T_{half}]$ | $y=0, \dot{x}=0$ |
| `2D_symmetric_x_fixed_t` | $[x_0, v_{y0}]$ | $y=0, \dot{x}=0$ |
| `2D_symmetric_y_fixed_y0` | $[\dot{x}_0, T_{half}]$ | $x=0, \dot{x}=0$ |
| `3D_symmetric_x_fixed_x0` | $[z_0, v_{y0}, T_{half}]$ | $y=0, \dot{x}=0, \dot{z}=0$ |
| `3D_symmetric_xz_fixed_x0` | $[z_0, v_{y0}, T_{half}]$ | $y=0, \dot{x}=0, \dot{z}=0$ |
| `3D_symmetric_xz_fixed_z0` | $[x_0, v_{y0}, T_{half}]$ | $y=0, \dot{x}=0, \dot{z}=0$ |

#### 配置方法

```python
# 2D对称x轴固定x0配置
corrector.setup_2D_symmetric_x_fixed_x0(x0=0.8)

# 3D Halo轨道配置
corrector.setup_3D_symmetric_x_fixed_x0(x0=0.8)
```

#### 核心方法

| 方法 | 说明 |
|------|------|
| `setup_2D_symmetric_x_fixed_x0(x0)` | 配置2D对称x轴搜索 |
| `setup_2D_symmetric_x_fixed_t(t_half)` | 配置2D固定周期搜索 |
| `setup_3D_symmetric_x_fixed_x0(x0)` | 配置3D对称搜索 |
| `iterate_correction(initial_guess, t_half, verbose)` | 执行迭代修正 |
| `_compute_jacobian_finite_diff()` | 有限差分计算雅可比 |

---

### 2.2 Continuation

**文件**: `e2m2e/algorithms/continuation.py`

**类签名**:
```python
class Continuation:
    """轨道族延拓"""
```

#### 自然参数延拓 (Natural Continuation)

沿选定的参数（如 $x_0$, $z_0$, 周期等）逐步延拓：

1. 从种子轨道出发
2. 在参数方向施加步长 $\Delta s$
3. 使用前一条轨道状态作为初始猜测
4. 调用微分修正器求解
5. 重复直到参数范围边界

#### 伪弧长延拓 (Pseudo-Arclength Continuation)

当轨道族出现折返（fold）时，自然延拓会失效。伪弧长方法引入弧长参数 $s$：

$$\frac{d\mathbf{u}}{ds} = \frac{\mathbf{t}}{\|\mathbf{t}\|}$$

其中 $\mathbf{t}$ 是切向量，$\mathbf{u} = [\mathbf{x}; T/2]$ 是状态向量。

#### 核心方法

| 方法 | 说明 |
|------|------|
| `natural_continuation(seed_orbit, param_range, step_size, verbose)` | 自然参数延拓 |
| `pseudo_arclength_continuation(seed_orbit, n_orbits, step_size, direction, ..., dc_scheme, ...)` | XZ 对称伪弧长延拓（`direction`: `positive` / `negative`） |
| `generate_halo_seed_orbit(libration_point, amplitude_z, halo_class, ...)` | Halo 种子轨道 |
| `generate_halo_family(seed_orbit, n_orbits, direction, step_size)` | 按 `amplitude_z` 的自然参数式族生成 |
| `halo_pseudo_arclength_continuation(seed_orbit, n_orbits, direction, step_size, step_size_negative, ...)` | Halo 伪弧长族（双向支、可选 MATLAB 对齐参数） |

详见 [Halo 算法文档](../algorithms/halo.md)。

#### 使用示例

```python
from e2m2e.algorithms.continuation import Continuation

# 创建延拓器
continuation = Continuation(corrector, step=0.01)

# 自然参数延拓
family = continuation.natural_continuation(
    seed_orbit=seed_orbit,
    param_range=(0.8, 1.2),
    step_size=0.01,
    verbose=True
)
```

---

### 2.3 StabilityAnalysis, StabilityType & BifurcationType

**文件**: `e2m2e/algorithms/stability.py`

**类签名**:
```python
class StabilityType(Enum):
    STABLE = "stable"
    UNSTABLE = "unstable"
    MARGINALLY_STABLE = "marginally_stable"
    HYPERBOLIC = "hyperbolic"
    ELLIPTIC = "elliptic"
    PARABOLIC = "parabolic"

class BifurcationType(Enum):
    NONE = "none"
    PERIOD_DOUBLING = "period_doubling"
    SADDLE_NODE = "saddle_node"
    TORUS = "torus"
    PITCHFORK = "pitchfork"
    TRANSCRITICAL = "transcritical"
    SECONDARY_HOPF = "secondary_hopf"

class StabilityAnalysis:
    """轨道稳定性分析"""
```

#### 稳定性分析数学基础

对于周期轨道，单值矩阵 $M$ 的特征值（Floquet乘子）$\lambda_i$ 满足：
- $|\lambda_i| = 1$: 中性稳定（椭圆轨道）
- $|\lambda_i| < 1$: 渐近稳定
- $|\lambda_i| > 1$: 不稳定（双曲轨道）

由于辛矩阵的性质，特征值成倒数对出现：
$$\lambda_1 \lambda_2 = 1, \quad \lambda_3 \lambda_4 = 1$$

#### 稳定性指数

常用的稳定性指数定义：

**Broucke稳定性指数**:
$$\nu = \frac{|\lambda_1| + |\lambda_2| + |\lambda_3| + |\lambda_4|}{4}$$

对于稳定轨道，$\nu = 1$。

#### 核心方法

| 方法 | 说明 |
|------|------|
| `compute_monodromy()` | 计算单值矩阵 |
| `compute_floquet_multipliers()` | 计算Floquet乘子 |
| `compute_stability_index()` | 计算稳定性指数 (nu1, nu2, nu3, broucke) |
| `classify_orbit()` | 分析稳定性类型 |
| `analyze_bifurcation()` | 检测分岔类型 |
| `full_analysis()` | 运行完整分析 |
| `detect_bifurcation_in_family(orbits, dynamics)` | 静态方法：检测轨道族中的分岔 |
| `find_nearest_bifurcation(orbits, dynamics, target_x0)` | 静态方法：查找最近分岔点 |

---

### 2.4 CorrectionConfig & 策略函数

**文件**: `e2m2e/algorithms/strategies/`

v3.2 引入的策略模式将微分修正的配置逻辑从 `DifferentialCorrection` 类中分离为独立的不可变配置和策略函数。

#### CorrectionConfig

**类签名**:
```python
@dataclass(frozen=True)
class CorrectionConfig:
    """微分修正策略的不可变配置"""
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `setup_type` | `str` | 修正配置类型标识 |
| `symmetry_condition` | `str` | 利用的对称性（如 `'x_axis'`） |
| `fixed_parameters` | `Dict[str, float]` | 修正期间保持不变的参数 |
| `free_variables` | `List[str]` | Newton 求解器调整的变量名 |
| `free_variable_indices` | `List[int]` | 自由变量对应的状态向量索引 |
| `target_conditions` | `Dict[str, float]` | 约束名称到目标值的映射 |
| `constraint_indices` | `List[int]` | 约束评估对应的状态向量索引 |
| `constraint_weights` | `Dict[str, float]` | 雅可比矩阵的逐约束权重 |
| `constraint_types` | `Dict[str, str]` | 逐约束分类（如 `'equality'`） |

#### 策略函数

| 函数 | 对称性 | 自由变量 | 目标约束 |
|------|--------|---------|---------|
| `symmetric_2d_fixed_x0(x0)` | x 轴 | $[v_{y0}, T_{half}]$ | $y=0, \dot{x}=0$ |
| `symmetric_2d_fixed_t(t_half)` | x 轴 | $[x_0, v_{y0}]$ | $y=0, \dot{x}=0$ |
| `symmetric_2d_fixed_y0(y0)` | y 轴 | $[\dot{x}_0, T_{half}]$ | $x=0, \dot{y}=0$ |
| `symmetric_3d_fixed_x0(x0)` | x 轴 | $[z_0, v_{y0}, T_{half}]$ | $y=0, \dot{x}=0, \dot{z}=0$ |
| `symmetric_xz_fixed_x0(x0)` | xz 平面 | $[z_0, v_{y0}, T_{half}]$ | $y=0, \dot{x}=0, \dot{z}=0$ |
| `symmetric_xz_fixed_z0(z0)` | xz 平面 | $[x_0, v_{y0}, T_{half}]$ | $y=0, \dot{x}=0, \dot{z}=0$ |
| `halo_fixed_x0(x0, libration_point)` | xz 平面 | $[z_0, v_{y0}, T_{half}]$ | $y=0, \dot{x}=0, \dot{z}=0$ |
| `halo_fixed_z0(z0, libration_point)` | xz 平面 | $[x_0, v_{y0}, T_{half}]$ | $y=0, \dot{x}=0, \dot{z}=0$ |

#### 使用示例

```python
from e2m2e.algorithms.strategies import CorrectionConfig, halo_fixed_z0

# 使用策略函数生成配置
config = halo_fixed_z0(z0=0.1, libration_point=1)
# config 是一个不可变的 CorrectionConfig 实例

# 传给 DifferentialCorrection 使用
corrector = DifferentialCorrection(dynamics)
corrector.setup_halo_orbit_fixed_z0(z0=0.1, libration_point=1)
```

---

## 3. 转移模块

### 3.1 TransferSearch / DROTransferSearch

**文件**: `e2m2e/transfer/transfer_search.py`

**类签名**:
```python
class TransferSearch:
    """DRO到RO平面转移轨道的网格搜索"""

DROTransferSearch = TransferSearch   # 别名
DROROTransferSearch = TransferSearch # 别名
```

#### 设计原理

`TransferSearch` 实现从远距离逆行轨道（DRO）到共振轨道（RO）的两脉冲转移的网格搜索阶段：

1. 在出发 DRO 上采样多个出发点
2. 对每个出发点，沿切向施加不同速度比 $\alpha$ 进行前向积分
3. 检测轨迹是否与目标 RO 相交或距离局部最小
4. 标记可行候选解

#### 搜索参数（直接在实例上设置）

|| 属性 | 类型 | 默认值 | 说明 |
||------|------|--------|------|
|| `alpha_min` | float | 0.5 | 切向速度比下界 |
|| `alpha_max` | float | 2.5 | 切向速度比上界 |
|| `n_alpha` | int | 101 | $\alpha$ 方向网格点数 |
|| `n_departure` | int | 200 | 出发点采样数 |
|| `max_transfer_time` | float | 200/TU | 最大转移时间（无量纲） |
|| `intersection_threshold` | float | 0.001 | 相交判定阈值（DU） |
|| `min_distance_threshold` | float | 100/DU | 最小距离阈值 |
|| `collision_earth_radius` | float | 200/DU | 地球碰撞半径 |
|| `collision_moon_radius` | float | 100/DU | 月球碰撞半径 |
|| `integration_dt` | float | 1/(24·TU) | 积分步长 |

#### 核心方法

|| 方法 | 说明 |
||------|------|
|| `search(*, alpha_min, alpha_max, n_alpha, n_departure, max_transfer_time, departure_orbit, arrival_orbit, ...)` | 执行网格搜索 |
|| `get_feasible_results()` | 获取最近一次搜索的可行结果 |
|| `optimize(initial_guess)` | 用 NLP 优化最佳搜索结果 |
|| `set_verbose(verbose)` | 设置详细输出（可链式调用） |
|| `set_n_workers(n_workers)` | 设置并行工作数（可链式调用） |
|| `set_parallel_backend(backend)` | 设置后端：`"threads"` / `"processes"`（可链式调用） |

#### 使用示例

```python
from e2m2e.transfer import TransferSearch

transfer_search = TransferSearch(dynamics=dynamics)
results = transfer_search.search(
    alpha_min=0.5, alpha_max=2.5,
    n_alpha=101, n_departure=200,
    max_transfer_time=200.0,
    departure_orbit=dro_orbit,
    arrival_orbit=ro_orbit,
)
```

---

### 3.2 DROTRONLPOptimizer

**文件**: `e2m2e/transfer/transfer_optimization.py`

**类签名**:
```python
class DROTRONLPOptimizer:
    """DRO到RO两脉冲转移的NLP优化器"""
```

#### 设计原理

将两脉冲转移问题转化为非线性规划（NLP）问题：

- **优化变量**: $y = \{\alpha, T, t_{ins}\}$
- **目标函数**: $J(y) = \Delta v_1 + \Delta v_2$
- **约束条件**: 位置连续、速度平行、撞星约束

默认使用 SciPy `minimize(method="SLSQP")` 求解。

#### TransferType 枚举

|| 值 | 说明 |
||------|------|
|| `DIRECT` | 直接转移 |
|| `LGA` | 月球借力转移 |
|| `EXTERNAL` | 外部转移 |

#### 核心方法

|| 方法 | 说明 |
||------|------|
|| `optimize(initial_variables, ...)` | 执行NLP优化 |
|| `optimize_with_copt(...)` | 使用COPT求解器（需安装coptpy） |

#### 使用示例

```python
from e2m2e.transfer import DROTRONLPOptimizer, NLPOptimizationVariables

optimizer = DROTRONLPOptimizer(
    system=system, dynamics=dynamics,
    departure_orbit=dro_orbit, arrival_orbit=ro_orbit,
    departure_state=dro_orbit.states[0],
)

initial_vars = NLPOptimizationVariables(alpha=1.0, transfer_time=5.0, t_ins=3.0)
result = optimizer.optimize(initial_guess=initial_vars)
```

---

### 3.3 SearchConfig

**文件**: `e2m2e/transfer/search_config.py`

**类签名**:
```python
@dataclass
class SearchConfig:
    """TransferSearch 网格搜索配置"""
```

`SearchConfig` 将搜索和优化参数集中管理为可复用的 dataclass，便于序列化和类型检查。

#### 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `alpha_min` | `float` | α（切向速度比）下界 |
| `alpha_max` | `float` | α 上界 |
| `n_alpha` | `int` | α 方向网格点数 |
| `n_departure` | `int` | 出发点采样数量 |
| `max_transfer_time` | `float` | 最大转移时间（无量纲） |
| `intersection_threshold` | `float` | 相交判定距离阈值 |
| `min_distance_threshold` | `float` | 候选解最小距离阈值 |
| `collision_earth_radius` | `float` | 地球碰撞检测半径（无量纲） |
| `collision_moon_radius` | `float` | 月球碰撞检测半径（无量纲） |
| `integration_dt` | `float` | 积分时间步长（无量纲） |
| `alpha_range` | `Tuple[float, float]` | 优化阶段 α 搜索范围 |
| `transfer_time_range` | `Tuple[float, float]` | 优化阶段转移时间范围 |
| `t_ins_range` | `Tuple[float, float]` | 优化阶段插入时间范围 |
| `velocity_angle_tolerance` | `float` | 速度平行性容差（弧度） |

---

### 3.4 Transfer（简化 API）

**文件**: `e2m2e/transfer/transfer.py`

**类签名**:
```python
class Transfer:
    """DRO-RO 转移轨迹优化的简化链式接口"""
```

`Transfer` 封装了 `DROTRONLPOptimizer`，提供流畅的链式调用风格。

#### 使用示例

```python
from e2m2e.transfer import Transfer

transfer = Transfer(dynamics)
result = transfer.set_orbit(start=dro_orbit, end=ro_orbit).optimize(
    initial_guess={"alpha": 1.0, "transfer_time": 15.0, "t_ins": 5.0},
    alpha_range=(0.5, 2.5),
)
```

---

### 3.5 工具函数

```python
from e2m2e.transfer import load_orbit_from_json, optimize_transfer, optimize_with_copt

# 从JSON加载轨道
orbit = load_orbit_from_json("path/to/orbit.json")

# 便捷优化函数
result = optimize_transfer(system, dynamics, departure_orbit, arrival_orbit, departure_state)

# 使用COPT求解器（需安装coptpy）
result = optimize_with_copt(optimizer, initial_guess, fallback_to_scipy=True)
```

---

## 4. 可视化模块

> `plotting.py` 已拆分为 `config.py`、`base.py`、`family.py`、`transfer.py`、`stability.py`，原路径仍作为重导出兼容层可用。

### 4.1 PlotConfig

**文件**: `e2m2e/visualization/config.py`

**类签名**:
```python
@dataclass
class PlotConfig:
    """可视化全局配置"""
```

#### 主要字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `figsize` | `tuple` | 图像尺寸 (width, height) |
| `dpi` | `int` | 分辨率 |
| `style` | `str` | matplotlib 风格 |
| `color_scheme` | `str` | 配色方案 |
| `show_grid` | `bool` | 是否显示网格 |
| `show_legend` | `bool` | 是否显示图例 |
| `save_format` | `str` | 保存格式 (png/pdf/svg) |

#### 使用示例

```python
from e2m2e.visualization import PlotConfig

config = PlotConfig(figsize=(12, 8), dpi=150, style="dark_background")
```

---

### 4.2 OrbitVisualizer & ProjectionPlane

**文件**: `e2m2e/visualization/base.py`

**类签名**:
```python
class ProjectionPlane(Enum):
    XY = "xy"
    XZ = "xz"
    YZ = "yz"

class OrbitVisualizer:
    """轨道可视化器"""
```

#### 功能列表

| 功能 | 方法 |
|------|------|
| 3D轨道绘制 | `plot_3d_orbit()` |
| 2D投影 | `plot_2d_projection()` |
| 主/次天体 | `plot_primary_bodies()` |
| 平动点标注 | `plot_libration_points()` |
| 庞加莱截面 | `plot_poincare_section()` |
| Jacobi常数 | `plot_jacobi_constant()` |
| 稳定性图 | `plot_stability_diagram()` |
| 概览图 | `create_overview_plot()` |

#### 使用示例

```python
from e2m2e.visualization.base import OrbitVisualizer

viz = OrbitVisualizer(system)

# 创建概览图
fig = viz.create_overview_plot(orbit)

# 保存
viz.save('orbit.png', dpi=300)
```

---

### 4.3 FamilyPlotter

**文件**: `e2m2e/visualization/family.py`

**类签名**:
```python
class FamilyPlotter:
    """轨道族可视化器"""
```

#### 核心方法

| 方法 | 说明 |
|------|------|
| `plot_family_2d()` | 绘制轨道族 2D 投影图 |
| `plot_family_3d()` | 绘制轨道族 3D 图 |
| `plot_jacobi_period_stability()` | 绘制 Jacobi 常数–周期–稳定性关系图 |
| `plot_family_overview()` | 绘制轨道族综合概览图 |

#### 使用示例

```python
from e2m2e.visualization import FamilyPlotter

plotter = FamilyPlotter(system)
plotter.plot_family_2d(family)
plotter.plot_family_overview(family)
```

---

### 4.4 TransferPlotter

**文件**: `e2m2e/visualization/transfer.py`

**类签名**:
```python
class TransferPlotter:
    """转移轨道可视化器"""
```

#### 核心方法

| 方法 | 说明 |
|------|------|
| `plot_solution_plane()` | 绘制解平面图 |
| `plot_transfer_orbit()` | 绘制转移轨迹图 |

#### 使用示例

```python
from e2m2e.visualization import TransferPlotter

plotter = TransferPlotter(system)
plotter.plot_solution_plane(search_results)
plotter.plot_transfer_orbit(transfer_result)
```

---

### 4.5 compute_stability_for_family

**文件**: `e2m2e/visualization/stability.py`

**函数签名**:
```python
def compute_stability_for_family(family_result, system) -> list:
    """计算轨道族的稳定性指数"""
```

#### 功能说明

对轨道族中每条轨道计算稳定性指数：
1. 计算单值矩阵 $M$
2. 求特征值 $\lambda_i$
3. 取最大模长 $\nu = \max|\lambda_i|$

稳定性判定：
- $\nu = 1$: 中性稳定
- $\nu < 1$: 渐近稳定
- $\nu > 1$: 不稳定

---

## 附录

### A. 物理常数

| 常数 | 值 | 单位 |
|------|-----|------|
| $G$ | $6.67430 \times 10^{-20}$ | km³/kg/s² |
| AU | $149,597,870.7$ | km |
| 地月距离 | 384,400 | km |
| 地月周期 | 27.32 | day |

### B. 已知系统预设

```python
KNOWN_SYSTEMS = {
    "earth_moon": {"mu": 0.01215, "distance": 384400, "period": 27.32*86400},
    "sun_earth": {"mu": 3.0039e-6, "distance": 1*AU, "period": 365.25*86400},
    "sun_jupiter": {"mu": 0.0009535, "distance": 5.2*AU, "period": 11.86*365.25*86400}
}
```

### C. 状态向量索引

| 索引 | 分量 | 说明 |
|------|------|------|
| 0 | x | x坐标 |
| 1 | y | y坐标 |
| 2 | z | z坐标 |
| 3 | $\dot{x}$ | x方向速度 |
| 4 | $\dot{y}$ | y方向速度 |
| 5 | $\dot{z}$ | z方向速度 |
