# 动力学积分：CR3BP_Dynamics

> **文件**: `e2m2e/core/dynamics.py`

`CR3BP_Dynamics` 实现了 CR3BP 的运动方程和数值积分。它是所有轨道计算的核心引擎——微分修正、轨道传播、状态转移矩阵计算都依赖它。

## 怎么创建动力学模型

```python
from e2m2e.core import CR3BP_System, CR3BP_Dynamics

system = CR3BP_System.from_known_system("earth_moon")
dynamics = CR3BP_Dynamics(system)
```

一个 `CR3BP_Dynamics` 对象绑定一个 `CR3BP_System`，后续所有传播都使用该系统的参数。

## 怎么传播轨道

### 基本传播

```python
import numpy as np

initial_state = np.array([0.8, 0.0, 0.0, 0.0, 1.0, 0.0])
result = dynamics.propagate(initial_state, t_span=(0, 10.0))

states = result["states"]       # [n, 6] 状态数组
times = result["times"]         # [n] 时间数组
jacobi = result["jacobi"]       # [n] Jacobi 常数数组
```

积分器使用 scipy `solve_ivp`，精度为 `rtol=atol=1e-12`。

### 带 STM 的传播

当需要计算状态转移矩阵（微分修正、稳定性分析等算法需要）时，设置 `with_stm=True`：

```python
result = dynamics.propagate(initial_state, t_span=(0, 10.0), with_stm=True)

stm = result["stm"]  # [n, 6, 6] 状态转移矩阵数组
```

这会同时积分 42 维增广状态（6 维状态 + 36 维 STM 分量）。

### 监控 Jacobi 常数

Jacobi 常数在 CR3BP 中是守恒量。如果传播过程中 Jacobi 常数显著变化，说明积分精度不够：

```python
result = dynamics.propagate(initial_state, t_span=(0, 100.0))
jacobi_drift = abs(result["jacobi"][-1] - result["jacobi"][0])
print(f"Jacobi 漂移: {jacobi_drift:.2e}")
# 正常情况应 < 1e-10
```

## 怎么计算状态转移矩阵

单独计算某一时刻的 STM：

```python
stm = dynamics.compute_state_transition_matrix(initial_state, t=5.0)
print(stm.shape)  # (6, 6)
```

STM 是微分修正的核心工具——它告诉你"初始状态的微小变化如何影响最终状态"。

→ 详见 [微分修正 - 怎么修正出精确的周期轨道](../algorithms/differential_correction.md)

## API 速查

| 方法 | 说明 |
|------|------|
| `propagate(initial_state, t_span, with_stm, with_jacobi)` | 积分轨道（核心方法） |
| `compute_state_transition_matrix(initial_state, t)` | 计算 6×6 状态转移矩阵 |
| `equations_of_motion(t, state)` | 6 维运动方程（可重写） |
| `equations_with_stm(t, augmented_state)` | 42 维增广方程 |
| `compute_jacobi_constant(state)` | 计算 Jacobi 常数 |
| `check_cross_section(state, plane, value)` | 检测 Poincaré 截面穿越 |

完整 API 文档见 [API 参考](../reference/api-reference.md)。

## 运动方程

CR3BP 在旋转坐标系下的无量纲运动方程：

$$\begin{cases}
\dot{x} = v_x \\
\dot{y} = v_y \\
\dot{z} = v_z \\
\dot{v}_x = 2v_y + x - \dfrac{(1-\mu)(x+\mu)}{r_1^3} - \dfrac{\mu(x-1+\mu)}{r_2^3} \\
\dot{v}_y = -2v_x + y - \dfrac{(1-\mu)y}{r_1^3} - \dfrac{\mu y}{r_2^3} \\
\dot{v}_z = -\dfrac{(1-\mu)z}{r_1^3} - \dfrac{\mu z}{r_2^3}
\end{cases}$$

其中 $r_1 = \sqrt{(x+\mu)^2 + y^2 + z^2}$，$r_2 = \sqrt{(x-1+\mu)^2 + y^2 + z^2}$。

加速度项包含：科氏力（$\pm 2v$）、离心力（$x, y$）、两个天体的引力。
