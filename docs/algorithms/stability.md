# StabilityAnalysis

**文件**: `e2m2e/algorithms/stability.py`

**类签名**:
```python
class StabilityAnalysis:
    """轨道稳定性分析"""
```

## Floquet 理论

对于周期轨道 $\mathbf{x}(t)$，其附近的小扰动满足：
$$\Delta \dot{\mathbf{x}}(t) = \mathbf{A}(t) \Delta \mathbf{x}(t)$$

其中 $\mathbf{A}(t)$ 是状态转移矩阵，周期性：$\mathbf{A}(t+T) = \mathbf{A}(t)$。

### Floquet 乘子
$$\boldsymbol{\Phi}(T) \mathbf{v} = \lambda \mathbf{v}$$

$\boldsymbol{\Phi}(T)$ 是单值矩阵，$\lambda$ 是 Floquet 乘子。

## 稳定性分类

| 稳定性类型 | 乘子特征 | 轨道性质 |
|------------|----------|----------|
| 稳定 (Stable) | 所有 $\|\lambda\| = 1$ | Lyapunov 稳定 |
| 不稳定 (Unstable) | 存在 $\|\lambda\| > 1$ | 指数发散 |
| 椭圆型 (Elliptic) | 乘子在单位圆上 | KAM 适用 |
| 抛物型 (Parabolic) | 乘子 $= 1$ | 临界情况 |

## 核心方法

| 方法 | 说明 |
|------|------|
| `compute_floquet_multipliers(orbit)` | 计算 Floquet 乘子 |
| `classify_stability(multipliers)` | 分类稳定性 |
| `compute_stability_index(multipliers)` | 计算稳定性指数 |
| `analyze_lyapunov(orbit, dt, n_orbits)` | Lyapunov 指数分析 |

## 稳定性指数

$$\nu = \frac{1}{n} \sum_{i=1}^{n} \ln |\lambda_i|$$

其中 $\lambda_i$ 是 Floquet 乘子。

## 使用示例

```python
from e2m2e.algorithms.stability import StabilityAnalysis

analyzer = StabilityAnalysis(system, dynamics)

# 分析轨道稳定性
multipliers = analyzer.compute_floquet_multipliers(orbit)
stability_type, index = analyzer.classify_stability(multipliers)

print(f"稳定性: {stability_type}, 指数: {index}")
```
