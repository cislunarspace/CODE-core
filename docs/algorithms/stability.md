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

## 枚举类型

### StabilityType（稳定性类型）

| 值 | 说明 |
|----|------|
| `STABLE` | 所有 $\|\lambda\| = 1$，Lyapunov 稳定 |
| `UNSTABLE` | 存在 $\|\lambda\| > 1$，指数发散 |
| `MARGINALLY_STABLE` | 临界稳定 |
| `HYPERBOLIC` | 双曲型 |
| `ELLIPTIC` | 椭圆型，乘子在单位圆上 |
| `PARABOLIC` | 抛物型，乘子 $= 1$ |

### BifurcationType（分岔类型）

| 值 | 说明 |
|----|------|
| `NONE` | 无分岔 |
| `PERIOD_DOUBLING` | 倍周期分岔 |
| `SADDLE_NODE` | 鞍结分岔 |
| `TORUS` | 环面分岔 |
| `PITCHFORK` | 叉形分岔 |
| `TRANSCRITICAL` | 跨临界分岔 |
| `SECONDARY_HOPF` | 次 Hopf 分岔 |

## 核心方法

| 方法 | 说明 |
|------|------|
| `analyze()` | 执行完整稳定性分析，返回 `StabilityType` 和分岔信息 |
| `compute_stability_index()` | 计算最大 Floquet 乘子模 |
| `classify_stability()` | 分类稳定性类型 |
| `detect_bifurcation()` | 检测分岔类型 |

## 稳定性指数

$$\nu = \frac{1}{n} \sum_{i=1}^{n} \ln |\lambda_i|$$

其中 $\lambda_i$ 是 Floquet 乘子。

## 使用示例

```python
from e2m2e.core import CR3BP_System, CR3BP_Dynamics, Orbit
from e2m2e.algorithms import StabilityAnalysis, StabilityType, BifurcationType

system = CR3BP_System.from_known_system("earth_moon")
dynamics = CR3BP_Dynamics(system)

# 创建轨道对象
orbit = Orbit(states=[initial_state], times=[0])
orbit.period = 3.0
orbit.system = system

# 创建分析器
analyzer = StabilityAnalysis(orbit=orbit, dynamics=dynamics)

# 执行稳定性分析
result = analyzer.analyze()
print(f"稳定性类型: {result.stability_type}")
print(f"最大乘子模: {result.max_multiplier_magnitude}")

if result.bifurcation_type != BifurcationType.NONE:
    print(f"检测到分岔: {result.bifurcation_type}")
```
