---
title: 周期轨道修正：DifferentialCorrection
---

# 周期轨道修正：DifferentialCorrection

> **文件**: `e2m2e/algorithms/differential_correction.py`

微分修正通过线化周期条件，用 Newton-Raphson 迭代将一个"粗略猜测"收敛为精确的周期轨道。

## 怎么修正出一条精确的周期轨道

端到端流程：

```python
import numpy as np
from e2m2e.core import CR3BP_System, CR3BP_Dynamics
from e2m2e.algorithms import DifferentialCorrection

# 1. 初始化
system = CR3BP_System.from_known_system("earth_moon")
system.set_characteristic_scales(distance=384400, period=27.32*86400)
system.compute_libration_points()
dynamics = CR3BP_Dynamics(system)

# 2. 创建修正器，选择对称性配置
dc = DifferentialCorrection(dynamic=dynamics)
dc.setup_2D_symmetric_x_fixed_x0(x0=0.8)

# 3. 提供初值猜测并迭代
initial_state = np.array([0.8, 0.0, 0.0, 0.0, 0.5, 0.0])
orbit, result = dc.iterate_correction(initial_state, t_half=1.6)

print(f"收敛: {result['converged']}, 迭代: {result['iterations']}, 误差: {result['error']:.2e}")
```

## 选哪个对称性配置？

不同的轨道类型需要不同的对称性配置。选错了会导致不收敛或得到错误的轨道。

| 你想做什么 | 用哪个配置 | 说明 |
|-----------|-----------|------|
| 设计 DRO 或 Lyapunov（固定 x0） | `setup_2D_symmetric_x_fixed_x0(x0)` | 固定初始 x 坐标，求解 vy 和半周期 |
| 设计 DRO（固定周期） | `setup_2D_symmetric_x_fixed_t(t_half)` | 固定半周期，求解 x0 和 vy |
| 设计 DRO（固定 y0） | `setup_2D_symmetric_y_fixed_y0(y0)` | 固定初始 y 坐标 |
| 设计 Halo（固定 z0 振幅） | `setup_halo_orbit_fixed_z0(z0, lp)` | 最常用的 Halo 配置 |
| 设计 Halo（固定 x0） | `setup_halo_orbit_fixed_x0(x0, lp)` | 另一种 Halo 配置 |
| 一般 3D XZ 对称轨道 | `setup_3D_symmetric_xz_fixed_x0(x0)` | 通用 3D 对称 |

**选择经验**：

- **2D vs 3D**：如果轨道在 $z=0$ 平面内（DRO、Lyapunov），用 2D 配置；如果轨道有 $z$ 方向分量（Halo），用 3D 配置。
- **固定哪个参数**：取决于你对哪个量有好的先验估计。比如 Richardson 近似给出 $z_0$ 的估计，就用 `fixed_z0`。

## 初值猜测怎么来

好的初值是收敛的关键。几种获取方式：

### Richardson 三阶近似（Halo 专用）

```python
from e2m2e.algorithms import compute_halo_initial_guess

initial_state, t_half = compute_halo_initial_guess(
    system=system, libration_point=1, amplitude_z=0.1
)
```

这是基于 Lindstedt-Poincaré 方法的解析近似，对小幅值 Halo 非常准确。

### 从已有轨道延拓

如果已经有一条收敛的轨道，可以用 [Continuation](continuation.md) 沿族曲线小幅移动，得到新轨道的初值。

### 手动构造

对于 DRO，典型的初值结构：

```python
# DRO 初值：[x0, 0, 0, 0, vy0, 0]
initial_state = np.array([x0, 0.0, 0.0, 0.0, vy_guess, 0.0])
t_half = period_guess  # 半周期
```

## 收敛失败怎么办

| 现象 | 可能原因 | 对策 |
|------|---------|------|
| 迭代发散 | 初值离真实轨道太远 | 从更接近的已知轨道出发，或用 Richardson 近似 |
| 振荡不收敛 | 步长过大 | 检查 `max_iter`（默认 50），增大迭代次数 |
| 收敛到错误轨道 | 对称性配置不匹配 | 检查轨道类型是否与 `setup_*` 方法匹配 |
| Jacobi 常数异常 | 积分精度不够 | 确认 `rtol=atol=1e-12`，不要增大步长 |

## 数学原理

周期条件要求 $\mathbf{x}(T) - \mathbf{x}(0) = \mathbf{0}$。通过加入相位条件 $\phi$，构建校正方程：

$$\mathbf{F}(\mathbf{x}, \lambda) = \begin{pmatrix} \mathbf{x}(T; \mathbf{x}_0, \lambda) - \mathbf{x}_0 \\ \phi(\mathbf{x}_0, \lambda) \end{pmatrix} = \mathbf{0}$$

用 Newton-Raphson 迭代求解：

$$\mathbf{J} \Delta \mathbf{x} = -\mathbf{F}$$

其中 $\mathbf{J}$ 是通过 STM（状态转移矩阵）构建的 Jacobian 矩阵。

## API 速查

| 方法 | 说明 |
|------|------|
| `setup_2D_symmetric_x_fixed_x0(x0)` | 2D 对称，固定 x0 |
| `setup_2D_symmetric_x_fixed_t(t_half)` | 2D 对称，固定半周期 |
| `setup_2D_symmetric_y_fixed_y0(y0)` | 2D 对称，固定 y0 |
| `setup_3D_symmetric_x_fixed_x0(x0)` | 3D 对称，固定 x0 |
| `setup_3D_symmetric_xz_fixed_x0(x0)` | 3D XZ 对称，固定 x0 |
| `setup_3D_symmetric_xz_fixed_z0(z0)` | 3D XZ 对称，固定 z0 |
| `setup_halo_orbit_fixed_z0(z0, lp)` | Halo 专用，固定 z0 |
| `setup_halo_orbit_fixed_x0(x0, lp)` | Halo 专用，固定 x0 |
| `iterate_correction(initial_guess, t_half)` | 执行迭代修正，返回 `(Orbit, dict)` |

完整 API 文档见 [API 参考](../reference/api-reference.md)。
