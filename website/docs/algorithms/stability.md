---
title: 稳定性分析：StabilityAnalysis
---

# 稳定性分析：StabilityAnalysis

> **文件**: `e2m2e/algorithms/stability.py`

稳定性分析通过 Floquet 乘子判断周期轨道的局部稳定性，检测分岔点。这是理解轨道族拓扑结构的关键工具。

## 怎么判断一条轨道是否稳定

```python
from e2m2e.core import CR3BP_System, CR3BP_Dynamics, Orbit
from e2m2e.algorithms import StabilityAnalysis

system = CR3BP_System.from_known_system("earth_moon")
dynamics = CR3BP_Dynamics(system)

# 假设 orbit 是已有的周期轨道
analyzer = StabilityAnalysis(orbit=orbit, dynamics=dynamics)
result = analyzer.analyze()

print(f"稳定性类型: {result.stability_type}")
print(f"最大乘子模: {result.max_multiplier_magnitude:.6f}")
```

**怎么解读**：

- 最大乘子模 $\approx 1.0$：轨道稳定（乘子在单位圆上）
- 最大乘子模 $> 1.0$：轨道不稳定（存在指数发散方向）
- 乘子恰好穿过单位圆：可能有分岔

## 怎么在轨道族中找分岔点

分岔点是轨道族的"拐点"——在这些点上轨道的拓扑结构发生变化（例如从平面轨道分岔出 3D 轨道）。

```python
from e2m2e.algorithms import StabilityAnalysis, BifurcationType

# 对轨道族中的每条轨道分析稳定性
for orbit in family:
    analyzer = StabilityAnalysis(orbit=orbit, dynamics=dynamics)
    result = analyzer.analyze()

    if result.bifurcation_type != BifurcationType.NONE:
        print(f"Jacobi={orbit.jacobi_constant:.4f}: 检测到 {result.bifurcation_type}")
```

### 常见分岔类型

| 分岔类型 | 物理含义 | 对延拓的影响 |
|---------|---------|-------------|
| `SADDLE_NODE` | 鞍结分岔：族曲线的端点 | 延拓在此终止，需换方向 |
| `PERIOD_DOUBLING` | 倍周期分岔：周期翻倍 | 产生新的倍周期族 |
| `PITCHFORK` | 叉形分岔：对称性破缺 | 产生对称的两条新族 |
| `TORUS` | 环面分岔：乘子离开单位圆 | 产生准周期运动 |

### 稳定性类型速查

| 类型 | 含义 |
|------|------|
| `STABLE` | 所有乘子在单位圆上，Lyapunov 稳定 |
| `UNSTABLE` | 存在单位圆外的乘子 |
| `HYPERBOLIC` | 双曲型：乘子不在单位圆上 |
| `ELLIPTIC` | 椭圆型：乘子均在单位圆上 |
| `MARGINALLY_STABLE` | 临界稳定 |
| `PARABOLIC` | 抛物型：乘子恰好为 1 |

## 批量稳定性计算

对轨道族批量计算稳定性（可视化模块提供并行版本）：

```python
from e2m2e.visualization import compute_stability_for_family

stability_values = compute_stability_for_family(family, system)
# 返回每条轨道的最大乘子模
```

→ 详见 [可视化指南](../guides/visualization-guide.md#稳定性计算)

## API 速查

| 方法 | 说明 |
|------|------|
| `analyze()` | 完整分析：稳定性类型 + 分岔检测 |
| `compute_stability_index()` | 计算稳定性指数（最大乘子模） |
| `classify_stability()` | 分类稳定性类型 |
| `detect_bifurcation()` | 检测分岔类型 |

完整 API 文档见 [API 参考](../reference/api-reference.md)。

## 数学背景

### Floquet 理论

对于周期轨道 $\mathbf{x}(t)$，其附近的小扰动满足：

$$\Delta \dot{\mathbf{x}}(t) = \mathbf{A}(t) \Delta \mathbf{x}(t)$$

其中 $\mathbf{A}(t)$ 是周期系数矩阵。Monodromy 矩阵 $\boldsymbol{\Phi}(T)$ 的特征值就是 Floquet 乘子：

$$\boldsymbol{\Phi}(T) \mathbf{v} = \lambda \mathbf{v}$$

乘子在单位圆内/外/上分别对应稳定/不稳定/临界状态。

### 稳定性指数

$$\nu = \max_i |\lambda_i|$$

其中 $\lambda_i$ 是 Floquet 乘子。$\nu > 1$ 表示不稳定。
