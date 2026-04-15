---
title: 轨道生成指南
---

# 轨道生成指南

> 从初值猜测到收敛的周期轨道，再到完整的轨道族。

## 通用流程

所有周期轨道的生成都遵循相同的步骤：

```python
import numpy as np
from e2m2e.core import CR3BP_System, CR3BP_Dynamics
from e2m2e.algorithms import DifferentialCorrection

# 1. 创建系统
system = CR3BP_System.from_known_system("earth_moon")
system.set_characteristic_scales(distance=384400, period=27.32*86400)
system.compute_libration_points()

# 2. 创建动力学模型
dynamics = CR3BP_Dynamics(system)

# 3. 创建修正器，选择对称性配置
dc = DifferentialCorrection(dynamic=dynamics)
```

下一步取决于你要设计的轨道类型。继续阅读选择对应章节。

---

## DRO (Distant Retrograde Orbit)

DRO 是围绕月球的大幅值逆行轨道，在旋转系中闭合呈泪滴形。

**特征**：关于 x 轴对称，初始条件 $[x_0, 0, 0, 0, \dot{y}_0, 0]$，半周期处 $y=0, \dot{x}=0$。

```python
dc.setup_2D_symmetric_x_fixed_x0(x0=0.8)

initial_state = np.array([0.8, 0.0, 0.0, 0.0, 0.5, 0.0])
t_half_guess = 1.6

orbit, result = dc.iterate_correction(initial_state, t_half=t_half_guess)
print(f"周期: {orbit.period:.4f}, Jacobi: {orbit.jacobi_constant:.6f}")
```

### 参数范围参考

| 参数 | 典型范围 | 说明 |
|------|----------|------|
| $x_0$ | 0.6 - 0.95 | 初始 x 坐标（相对于 L1/L2） |
| $\dot{y}_0$ | 0.3 - 0.8 | 初始 y 方向速度 |
| $T/2$ | 1.4 - 1.8 | 半周期（无量纲） |

### 固定周期的 DRO

如果需要精确指定周期，用 `setup_2D_symmetric_x_fixed_t`：

```python
dc.setup_2D_symmetric_x_fixed_t(t_half=1.6)
# 此时 x0 和 vy 由修正器自动求解
```

→ 微分修正配置详解见 [微分修正](../algorithms/differential_correction.md)

---

## Halo 轨道

Halo 是围绕 L1/L2 平动点的三维周期轨道。

### 用 Richardson 初值生成

```python
from e2m2e.algorithms import compute_halo_initial_guess

initial_state, t_half = compute_halo_initial_guess(
    system=system, libration_point=1, amplitude_z=0.1
)

dc.setup_halo_orbit_fixed_z0(z0=initial_state[2], libration_point=1)
orbit, result = dc.iterate_correction(initial_state, t_half=t_half)
```

### 生成 Halo 轨道族

```python
from e2m2e.algorithms import Continuation

cont = Continuation(corrector=dc)
seed = cont.generate_halo_seed_orbit(
    libration_point=1, amplitude_z=0.23, halo_class=0,
)
family = cont.halo_pseudo_arclength_continuation(
    seed_orbit=seed, n_orbits=10, direction="both",
    step_size=0.0045, verbose=True,
)
```

### L1 vs L2

| 特性 | L1 Halo | L2 Halo |
|------|---------|---------|
| 位置 | L1 点附近 | L2 点附近 |
| $x_0$ 范围 | $0.8 < x_0 < 1.0$ | $1.0 < x_0 < 1.2$ |
| 振幅 | 通常较小 | 通常较大 |

→ Halo 详细文档（Richardson 初值、PAL 实现、MATLAB 对照、命令行脚本）见 [Halo 轨道](../algorithms/halo.md)

---

## Lyapunov 轨道

Lyapunov 轨道是位于平动点平面（$z=0$）内的二维周期轨道。

```python
dc.setup_2D_symmetric_x_fixed_x0(x0=system.L1[0] + 0.01)

initial_state = np.array([system.L1[0] + 0.01, 0.0, 0.0, 0.0, 0.3, 0.0])
orbit, result = dc.iterate_correction(initial_state, t_half=1.5)
```

Lyapunov 和 DRO 的区别：Lyapunov 在 L1/L2 附近、周期较短；DRO 在月球附近、周期较长。

---

## 轨道族延拓

从一条收敛的种子轨道出发，生成一族轨道：

```python
from e2m2e.algorithms import Continuation

cont = Continuation(corrector=dc, step=0.01)

# 自然延拓（参数单调变化时使用）
family = cont.natural_continuation(
    seed_orbit=orbit,
    param_range=(0.8, 0.95),
    step_size=0.01,
)
```

**什么时候用自然延拓、什么时候用伪弧长延拓**，以及参数详解，见 [轨道族延拓](../algorithms/continuation.md)。

---

## 保存与加载

```python
# 保存
family.save_to_file("output/dro_family.json")
orbit.save_to_file("output/dro_single.json")

# 加载
from e2m2e.core.orbit import OrbitFamily, Orbit
family = OrbitFamily.load_from_file("output/dro_family.json", system=system)
orbit = Orbit.load_from_file("output/dro_single.json", system=system)
```

---

## 常见问题

### 初值不收敛

- 先传播初值，检查是否接近周期（$y$ 和 $\dot{x}$ 在半周期处应接近 0）
- 从小幅值/短周期开始，逐步延拓
- 对 Halo，用 `compute_halo_initial_guess` 而非手动构造

### 需要大幅值轨道

从小幅值种子出发，用 [延拓算法](../algorithms/continuation.md) 逐步增大振幅。自然延拓在族曲线的转向点处会失效，此时切换到伪弧长延拓。

---

## 参考

- [微分修正](../algorithms/differential_correction.md) — 对称性配置选择、收敛问题排查
- [轨道族延拓](../algorithms/continuation.md) — 自然 vs 伪弧长延拓
- [Halo 轨道](../algorithms/halo.md) — Richardson 初值、PAL、MATLAB 对照
- [稳定性分析](../algorithms/stability.md) — Floquet 分析、分岔检测
- [可视化指南](visualization-guide.md) — 轨道族和转移轨迹可视化
