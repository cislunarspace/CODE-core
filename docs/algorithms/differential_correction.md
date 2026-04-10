# DifferentialCorrection

**文件**: `e2m2e/algorithms/differential_correction.py`

**类签名**:
```python
class DifferentialCorrection:
    """微分修正算法"""
```

## 设计原理

微分修正通过线性化周期条件来精确找到周期轨道。在周期轨道问题中，状态需要满足周期条件：
$$\mathbf{x}(T) - \mathbf{x}(0) = \mathbf{0}$$

## 单参数修正法

适用于族延拓中修正单参数（如 $C_J$）的情况：

1. **构建校正方程**
$$\mathbf{F}(\mathbf{x}, \lambda) = \begin{pmatrix} \mathbf{x}(T; \mathbf{x}_0, \lambda) - \mathbf{x}_0 \\ \phi(\mathbf{x}_0, \lambda) \end{pmatrix} = \mathbf{0}$$

其中 $\phi$ 是人为添加的相位条件。

2. **求解校正方程**
使用 Newton-Raphson 迭代：
$$\mathbf{J} \Delta \mathbf{x} = -\mathbf{F}$$

## 核心方法

| 方法 | 说明 |
|------|------|
| `setup_2D_symmetric_x_fixed_x0(x0)` | 配置 2D 对称、固定 x0 的搜索 |
| `setup_2D_symmetric_x_fixed_t(t_half)` | 配置 2D 对称、固定周期的搜索 |
| `setup_3D_symmetric_x_fixed_x0(x0)` | 配置 3D 对称搜索 |
| `iterate_correction(initial_guess, ...)` | 执行迭代修正 |

## 周期轨道检测

周期检测条件：
$$\|\mathbf{x}(T) - \mathbf{x}(0)\| < \epsilon_{period}$$

## 使用示例

```python
from e2m2e.core import CR3BP_System, CR3BP_Dynamics, Orbit
from e2m2e.algorithms import DifferentialCorrection

system = CR3BP_System.from_known_system("earth_moon")
dynamics = CR3BP_Dynamics(system)

corrector = DifferentialCorrection(dynamic=dynamics)
corrector.setup_2D_symmetric_x_fixed_x0(x0=0.8)

seed_orbit = Orbit(states=[initial_state], times=[0])
seed_orbit.period = 3.0

corrected_orbit = corrector.iterate_correction(initial_guess=seed_orbit)
```
