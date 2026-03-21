# DROROTransferSearch

**文件**: `e2m2e/transfer/inter_orbit.py`

**类签名**:
```python
class DROROTransferSearch:
    """DRO到RO转移轨道搜索"""
```

## 设计原理

DRO（远距离逆行轨道）到 RO（共振轨道）的转移轨道搜索模块。使用网格搜索和 SQP 优化算法寻找可行的两脉冲转移轨道。

## 算法流程

### 第一阶段：网格搜索 (Grid Search)

1. 在出发参数空间 $(v_\infty, \alpha, \beta)$ 离散采样
2. 对每个网格点，计算两脉冲转移的 $\Delta v$
3. 筛选满足约束的可行解

### 第二阶段：NLP优化 (SQP)

对可行解进行序列二次规划（SQP）优化：
$$\min \Delta v_{total}$$

$$\text{s.t.} \quad \mathbf{h}(\mathbf{x}) = \mathbf{0}$$

## 出发参数定义

| 参数 | 说明 |
|------|------|
| $\alpha$ | 方位角 (Azimuth angle) |
| $\beta$ | 高低角 (Elevation angle) |
| $v_\infty$ | 无限远速度 |

## 转移几何

```
DRO ──[脉冲1: 出发]──→ 转移轨道 ──[脉冲2: 到达]──→ RO
```

## 核心方法

| 方法 | 说明 |
|------|------|
| `grid_search(departure_orbit, arrival_orbit, ...)` | 网格搜索可行解 |
| `optimize_transfer(transfer_problem)` | SQP 优化 |
| `validate_transfer(transfer)` | 验证转移可行性 |

## 使用示例

```python
from e2m2e.transfer.inter_orbit import DROROTransferSearch

# 创建搜索器
searcher = DROROTransferSearch(
    system=system,
    dynamics=dynamics,
    max_transfer_time=5.0
)

# 网格搜索
feasible_solutions = searcher.grid_search(
    departure_orbit=dro_orbit,
    arrival_orbit=ro_orbit,
    alpha_range=(-30, 30),
    beta_range=(-10, 10),
    n_alpha=21,
    n_beta=21,
    n_departure=50
)

# 优化最优解
optimal_transfer = searcher.optimize_transfer(
    feasible_solutions[0]
)
```

## 输出格式

搜索结果：
```python
{
    "alpha": float,      # 方位角 (deg)
    "beta": float,       # 高低角 (deg)
    "departure_time": float,  # 出发时间
    "delta_v1": float,   # 第一次脉冲
    "delta_v2": float,   # 第二次脉冲
    "total_delta_v": float,
    "transfer_time": float
}
```
