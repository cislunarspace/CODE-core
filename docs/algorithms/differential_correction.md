# DifferentialCorrection

**文件**: `e2m2e/algorithms/differential_correction.py`

**类签名**:
```python
class DifferentialCorrection:
    """微分修正算法"""
```

## 设计原理

微分修正通过线性化 Poincaré 映射来精确找到周期轨道。在周期轨道问题中，状态需要满足周期条件：
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
| `correct_period(orbit, target_state)` | 周期轨道修正 |
| `correct_poincare(state, section)` | Poincaré 截面修正 |
| `compute_poincare_map(state, section)` | 计算 Poincaré 映射 |
| `compute_monodromy(state)` | 计算单值矩阵 |

## 周期轨道检测

周期检测条件：
$$\|\mathbf{x}(T) - \mathbf{x}(0)\| < \epsilon_{period}$$

## 使用示例

```python
from e2m2e.algorithms.differential_correction import DifferentialCorrection

corrector = DifferentialCorrection(system, dynamics)

# 修正周期轨道
corrected_orbit = corrector.correct_period(
    initial_guess=orbit,
    max_iterations=50,
    tolerance=1e-10
)
```
