# 轨道生成指南

## 概述

本指南介绍如何使用 E2M2E 生成各类周期轨道：DRO、Halo、Lyapunov 等。

## 通用流程

所有轨道生成的通用步骤：

```python
import numpy as np
from e2m2e.core.system import CR3BP_System
from e2m2e.core.dynamics import CR3BP_Dynamics
from e2m2e.algorithms.differential_correction import DifferentialCorrection

# 1. 初始化系统
system = CR3BP_System.from_known_system("earth_moon")
system.set_characteristic_scales(distance=384400, period=27.32*86400)
system.compute_libration_points()

# 2. 创建动力学模型
dynamics = CR3BP_Dynamics(system)

# 3. 创建微分修正器
dc = DifferentialCorrection(dynamics)

# 4. 配置并求解
dc.setup_2D_symmetric_x_fixed_x0(x0=0.8)  # 或其他配置
orbit, result = dc.iterate_correction(initial_guess, t_half_guess)
```

---

## Distant Retrograde Orbit (DRO)

### 理论背景

DRO 是围绕月球的大幅值逆行轨道，在旋转系中闭合呈泪滴形。其特点：
- 关于 x 轴对称
- 初始条件：$[x_0, 0, 0, 0, \dot{y}_0, 0]$
- 半周期条件：$y=0$, $\dot{x}=0$

### 生成步骤

```python
# 配置 2D 对称 x 轴固定 x0
dc.setup_2D_symmetric_x_fixed_x0(x0=0.8)

# 初始猜测（x0=0.8 附近）
initial_state = np.array([0.8, 0.0, 0.0, 0.0, 0.5, 0.0])
t_half_guess = 1.6  # 半周期猜测

# 迭代修正
orbit, result = dc.iterate_correction(initial_state, t_half_guess, verbose=True)
```

### 参数范围

| 参数 | 典型范围 | 说明 |
|------|----------|------|
| $x_0$ | 0.6 - 0.95 | 初始 x 坐标（相对于 L1/L2） |
| $\dot{y}_0$ | 0.3 - 0.8 | 初始 y 方向速度 |
| $T/2$ | 1.4 - 1.8 | 半周期（无量纲） |

### 完整示例

```python
def generate_dro(x0=0.8, y_dot_guess=0.5, t_half_guess=1.6):
    """生成 DRO 轨道"""
    system = CR3BP_System.from_known_system("earth_moon")
    system.compute_libration_points()
    dynamics = CR3BP_Dynamics(system)
    dc = DifferentialCorrection(dynamics)
    
    dc.setup_2D_symmetric_x_fixed_x0(x0=x0)
    initial_state = np.array([x0, 0.0, 0.0, 0.0, y_dot_guess, 0.0])
    
    orbit, result = dc.iterate_correction(initial_state, t_half_guess)
    return orbit, result, system
```

---

## Halo 轨道 {#halo-轨道}

### 理论背景

Halo 轨道是围绕平动点（L1 或 L2）的三维周期轨道，呈现扭曲的"8"字形或马蹄形。

**完整说明（API、伪弧长延拓、脚本、与 MATLAB 对照）见 [算法文档：Halo](../algorithms/halo.md)。**

### 库内推荐流程

1. **单条轨道**：Richardson 初值由 `compute_halo_initial_guess` 提供，`DifferentialCorrection.setup_halo_orbit_fixed_z0`（或 `fixed_x0`）后调用 `iterate_correction`。
2. **种子 + 轨道族**：`Continuation.generate_halo_seed_orbit` 得到收敛种子，再 `halo_pseudo_arclength_continuation` 沿伪弧长生成族（默认 `dc_scheme='adaptive'`，步长可与 `CR3BP_MATLAB_Library/examples/FAMILY_L1Halo_North.m` 对齐）。

```python
from e2m2e.algorithms import Continuation, DifferentialCorrection

dc = DifferentialCorrection(dynamic=dynamics)
cont = Continuation(corrector=dc)

seed = cont.generate_halo_seed_orbit(
    libration_point=1,
    amplitude_z=0.23,
    halo_class=0,
    verbose=False,
)
family = cont.halo_pseudo_arclength_continuation(
    seed_orbit=seed,
    n_orbits=10,
    direction="both",
    step_size=0.0045,
    step_size_negative=0.009,
    verbose=True,
)
```

### 命令行脚本

| 脚本 | 作用 |
|------|------|
| `scripts/generate/generate_halo_orbit.py` | 生成单条 Halo 并写入 `output/halo/` |
| `scripts/generate/generate_halo_family.py` | 种子 + 伪弧长轨道族 JSON |
| `scripts/plot/plot_halo_orbit.py` | 单轨/多轨绘图 |
| `scripts/plot/plot_halo_family.py` | 轨道族概览与稳定性图 |

### 3D 对称配置（底层）

```python
# 配置 3D 对称 x 轴固定 x0
dc.setup_3D_symmetric_x_fixed_x0(x0=0.8)

t_half_guess = 1.6
initial_state = np.array([0.8, 0.0, 0.1, 0.0, 0.5, 0.0])
orbit = Orbit(states=[initial_state], times=[0.0])
orbit.period = 2 * t_half_guess

orbit_result = dc.iterate_correction(orbit, verbose=True)
```

### L1 vs L2 Halo

| 特性 | L1 Halo | L2 Halo |
|------|---------|---------|
| 位置 | L1 点附近 | L2 点附近 |
| $x_0$ 范围 | $0.8 < x_0 < 1.0$ | $1.0 < x_0 < 1.2$ |
| 振幅 | 通常较小 | 通常较大 |

```python
# L1 Halo
L1_x0 = system.L1[0] + 0.01  # L1 右侧
dc.setup_3D_symmetric_x_fixed_x0(x0=L1_x0)

# L2 Halo
L2_x0 = system.L2[0] - 0.01  # L2 左侧
dc.setup_3D_symmetric_x_fixed_x0(x0=L2_x0)
```

---

## Lyapunov 轨道

### 理论背景

Lyapunov 轨道是位于平动点平面（z=0）内的二维周期轨道，呈现椭圆形或香蕉形。

### 配置

```python
# 使用 2D 对称配置
dc.setup_2D_symmetric_x_fixed_x0(x0=0.8)

# 初始猜测
initial_state = np.array([0.8, 0.0, 0.0, 0.0, 0.3, 0.0])
t_half_guess = 1.5
```

### 区别于 DRO

| 特性 | Lyapunov | DRO |
|------|----------|-----|
| 位置 | L1/L2 附近 | 月球附近 |
| z 振幅 | 0 | > 0（3D）|
| 周期 | 较短 | 较长 |

---

## 轨道族延拓

### 自然参数延拓

```python
from e2m2e.algorithms.continuation import Continuation

# 创建延拓器
continuation = Continuation(dc, step=0.005)

# 从种子轨道开始延拓
family = continuation.natural_continuation(
    seed_orbit=seed_orbit,
    param_range=(0.6, 0.95),  # x0 范围
    step_size=0.005,
    verbose=True
)
```

### 双向延拓

```python
# 正向延拓（x0 增大）
family_forward = continuation.natural_continuation(
    seed_orbit=seed_orbit,
    param_range=(seed_x0, 0.95),
    step_size=0.005
)

# 反向延拓（x0 减小）
family_backward = continuation.natural_continuation(
    seed_orbit=seed_orbit,
    param_range=(seed_x0, 0.6),
    step_size=-0.005  # 负步长
)
```

### 伪弧长延拓（绕过拐点）

```python
# 当自然延拓在拐点处失效时使用
family = continuation.pseudo_arclength_continuation(
    seed_state=seed_state,
    seed_t_half=seed_t_half,
    n_orbits=100,
    verbose=True
)
```

---

## 保存与加载

### 保存轨道族

```python
# 保存到 JSON 文件
family.save_to_file("output/dro_family.json")

# 或单个轨道
orbit.save_to_file("output/dro_single.json")
```

### 加载轨道族

```python
from e2m2e.core.orbit import OrbitFamily, Orbit

# 加载轨道族
family = OrbitFamily.load_from_file("output/dro_family.json")

# 加载单个轨道
orbit = Orbit.load_from_file("output/dro_single.json")
```

---

## 常见问题

### 1. 初始猜测如何确定？

**经验法则**：
- $x_0$：从目标区域中心开始，如 L1+0.01
- $\dot{y}_0$：从 0.5 开始，根据误差调整
- $T/2$：从 1.5 开始，根据周期估计调整

**调试技巧**：
- 先传播初始猜测，检查是否接近周期
- 观察 $y$ 和 $\dot{x}$ 的最终误差

### 2. 迭代不收敛怎么办？

**检查项**：
1. 初始猜测是否在合理范围
2. 积分器容差是否足够（当前 1e-12）
3. 尝试不同的初始猜测

**自适应阻尼**：
算法内置自适应阻尼，如仍不收敛，可手动减小步长。

### 3. 如何生成大幅值轨道？

**策略**：
1. 从小幅值轨道开始
2. 使用伪弧长延拓穿过拐点
3. 逐步增大振幅

---

## 参考

- 详见 [API 参考 - DifferentialCorrection](../reference/api-reference.md#21-differentialcorrection)
- 详见 [算法参考 - 轨道族延拓](../reference/algorithms.md)
