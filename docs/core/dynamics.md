# CR3BP_Dynamics

**文件**: `e2m2e/core/dynamics.py`

**类签名**:
```python
class CR3BP_Dynamics:
    """CR3BP动力学方程"""
```

## 设计原理

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

## 核心功能

1. **状态传播**: `propagate()` 方法使用 scipy 的 `solve_ivp` 进行数值积分
2. **状态转移矩阵 (STM)**: 通过 `equations_with_stm()` 同时积分42维增广状态
3. **Jacobi常数监控**: 实时计算 Jacobi 常数用于精度检验

## 核心方法

| 方法 | 说明 |
|------|------|
| `equations_of_motion(t, state)` | 6维运动方程 |
| `equations_with_stm(t, augmented_state)` | 42维增广运动方程（含STM） |
| `propagate(initial_state, t_span, with_stm=False)` | 传播轨迹 |
| `compute_state_transition_matrix(initial_state, t)` | 计算STM |
| `compute_jacobi_constant(state)` | 计算Jacobi常数 |
| `check_cross_section(state, plane, value)` | 检测截面穿越 |

## 使用示例

```python
from e2m2e.core.dynamics import CR3BP_Dynamics

# 创建动力学模型
dynamics = CR3BP_Dynamics(system)

# 传播轨迹
result = dynamics.propagate(
    initial_state=np.array([0.8, 0, 0, 0, 1.0, 0]),
    t_span=(0, 10.0)
)

print(f"轨迹点数: {len(result['states'])}")
print(f"最终状态: {result['states'][-1]}")
```
