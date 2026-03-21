# ContinuationMethod

**文件**: `e2m2e/algorithms/continuation.py`

**类签名**:
```python
class ContinuationMethod(Enum):
    NATURAL = "natural"           # 自然延拓
    PREDICTOR_CORRECTOR = "predictor_corrector"  # 预估校正
    ARC_LENGTH = "arc_length"     # 弧长延拓
```

## 设计原理

延拓法（Continuation Method）用于追踪参数空间中的解曲线，特别适合处理解的分叉和转向问题。

## 弧长延拓法 (Arc-Length Continuation)

核心思想：将参数 $\lambda$ 作为弧长 $s$ 的函数，通过预估-校正步骤沿解曲线前进。

### 预估步骤 (Predictor)
$$\begin{pmatrix} \mathbf{f}(\mathbf{x}_k) \\ \mathbf{g}(\mathbf{x}_k, s_k) \end{pmatrix} = \mathbf{0}$$

其中 $\mathbf{g}$ 是弧长约束条件。

### 校正步骤 (Corrector)
使用 Newton-Raphson 迭代求解：
$$\mathbf{J} \Delta \mathbf{x} = -\mathbf{f}$$

其中 $\mathbf{J}$ 是扩展 Jacobian 矩阵。

## 核心方法

| 方法 | 说明 |
|------|------|
| `predict(state, tangent, ds)` | 预估下一个点 |
| `correct(state, constraints)` | 校正求解 |
| `compute_tangent(jacobian)` | 计算切向量 |
| `find_bifurcation(points)` | 检测分叉点 |

## 使用示例

```python
from e2m2e.algorithms.continuation import ContinuationMethod

# 沿参数曲线延拓
continuer = ArcLengthContinuation(f, jacobian)
curve = continuer.continue_curve(x0, lambda_range, ds=0.01)
```
