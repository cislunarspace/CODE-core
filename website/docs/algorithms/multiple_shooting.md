---
title: 多重打靶法 MultipleShooting
---

# 多重打靶法（MultipleShooting）

`MultipleShooting` 类实现多重打靶法（Multiple Shooting）数值修正器，适用于复杂约束和长周期轨道的数值修正。

## 概述

多重打靶法将一条轨迹分为 N 个节点、n_seg = N-1 段弧段，对每段独立积分后，通过匹配相邻段端点状态来构建残差向量，再利用雅可比矩阵（含状态转移矩阵）进行最小二乘修正，反复迭代直到残差满足容差。

### 算法特点
- **分段积分**: 将长周期轨道分为多个短弧段，提高数值稳定性
- **边界匹配**: 通过匹配相邻段端点状态构建约束条件
- **自由时间支持**: 可选是否将时间节点作为自由变量
- **状态转移矩阵**: 利用 STM 构建雅可比矩阵，提高收敛速度

## 类定义

```python
class MultipleShooting:
    """多重打靶法（Multiple Shooting）修正器。
    
    将一条轨迹分为 N 个节点、n_seg = N-1 段弧段，对每段独立积分后，
    通过匹配相邻段端点状态来构建残差向量，再利用雅可比矩阵（含 STM）
    进行最小二乘修正，反复迭代直到残差满足容差。
    
    当 var_time=True 时，时间节点也作为自由变量参与修正（适用于自由时间问题）。
    """
```

## 辅助类

### `MultipleShootingResult`
修正结果的容器类。

**属性**:
- `t_patch`: 修正后的时间节点数组，形状 (N,)
- `state_patch`: 修正后的状态量数组，形状 (N, 6)
- `converged`: 是否在最大迭代次数内收敛
- `iterations`: 实际迭代次数
- `max_residual`: 最终迭代的最大残差
- `residual_history`: 每次迭代最大残差的历史记录

## 主要方法

### `__init__(dynamics)`
初始化多重打靶法修正器。

**参数**:
- `dynamics`: 动力学模型对象，需提供以下接口：
  - `propagate(state, time_span, with_stm=True)`: 积分传播，返回含 "states" 和 "stm" 的字典
  - `equations_of_motion(t, state)`: 计算状态导数（右端函数值）

### `correct(t_patch, state_patch, max_iter=100, tol=1e-10, var_time=False, verbose=False)`
执行多重打靶法修正。

**参数**:
- `t_patch`: 初始时间节点数组，形状 (N,)
- `state_patch`: 初始状态量数组，形状 (N, 6)
- `max_iter`: 最大迭代次数
- `tol`: 收敛容差
- `var_time`: 是否将时间节点作为自由变量
- `verbose`: 是否打印迭代信息

**返回**:
- `MultipleShootingResult`: 修正结果

### `compute_residual(t_patch, state_patch)`
计算当前打靶点的残差向量。

**参数**:
- `t_patch`: 时间节点数组
- `state_patch`: 状态量数组

**返回**:
- `np.ndarray`: 残差向量

### `compute_jacobian(t_patch, state_patch, var_time=False)`
计算雅可比矩阵。

**参数**:
- `t_patch`: 时间节点数组
- `state_patch`: 状态量数组
- `var_time`: 是否包含时间变量的导数

**返回**:
- `np.ndarray`: 雅可比矩阵

## 工具函数

### `sample_patch_points(orbit, n_segments, method='uniform')`
从轨道中均匀采样打靶点。

**参数**:
- `orbit`: `Orbit` 对象
- `n_segments`: 弧段数量（节点数 = n_segments + 1）
- `method`: 采样方法，'uniform'（均匀）或 'adaptive'（自适应）

**返回**:
- `tuple`: (t_patch, state_patch)

### `convert_to_j2000(state_patch, system)`
将状态转换到 J2000 惯性系。

**参数**:
- `state_patch`: 状态量数组，形状 (N, 6)
- `system`: `CR3BP_System` 或 `EphemerisSystem` 对象

**返回**:
- `np.ndarray`: J2000 惯性系中的状态量

## 使用示例

### 基本使用
```python
from e2m2e.algorithms import MultipleShooting, sample_patch_points
from e2m2e.core import CR3BP_System, CR3BP_Dynamics, Orbit

# 创建系统和动力学
system = CR3BP_System.from_known_system("earth_moon")
dynamics = CR3BP_Dynamics(system=system)

# 创建初始轨道（示例）
initial_state = [0.8, 0, 0, 0, 0.5, 0]
orbit = Orbit(states=[initial_state], times=[0])
orbit.period = 3.0

# 采样打靶点
t_patch, state_patch = sample_patch_points(
    orbit=orbit,
    n_segments=5  # 分为5段弧段
)

# 创建多重打靶法修正器
multiple_shooting = MultipleShooting(dynamics=dynamics)

# 执行修正
result = multiple_shooting.correct(
    t_patch=t_patch,
    state_patch=state_patch,
    max_iter=50,
    tol=1e-10,
    var_time=True,  # 允许时间节点变化
    verbose=True
)

# 检查结果
if result.converged:
    print(f"收敛于 {result.iterations} 次迭代")
    print(f"最大残差: {result.max_residual}")
    print(f"修正后时间节点: {result.t_patch}")
    print(f"修正后状态: {result.state_patch}")
else:
    print("未收敛")
    print(f"最终残差: {result.max_residual}")
```

### 与星历动力学结合
```python
from e2m2e.core import EphemerisSystem, EphemerisDynamics
from e2m2e.core.spice import SPICEManager

# 初始化 SPICE
spice_manager = SPICEManager()
spice_manager.load_kernels_from_directory("./kernels/")

# 创建星历系统和动力学
ephemeris_system = EphemerisSystem(
    bodies=["EARTH", "MOON", "SUN"],
    reference_epoch="2025-06-21T11:00:06"
)
ephemeris_dynamics = EphemerisDynamics(system=ephemeris_system)

# 使用多重打靶法修正星历轨道
multiple_shooting = MultipleShooting(dynamics=ephemeris_dynamics)

# 执行修正（固定时间）
result = multiple_shooting.correct(
    t_patch=t_patch,
    state_patch=state_patch,
    max_iter=100,
    tol=1e-12,
    var_time=False,  # 固定时间节点
    verbose=True
)
```

### 转换到 J2000 惯性系
```python
from e2m2e.algorithms import convert_to_j2000

# 将修正后的状态转换到 J2000 惯性系
state_j2000 = convert_to_j2000(
    state_patch=result.state_patch,
    system=system  # 或 ephemeris_system
)

print(f"J2000 惯性系状态形状: {state_j2000.shape}")
```

## 算法原理

### 1. 问题描述
对于轨道修正问题，需要找到满足边界条件的轨道。多重打靶法将连续问题离散化为多个弧段。

### 2. 变量定义
- 时间节点: $t_0, t_1, \dots, t_N$
- 状态节点: $\mathbf{x}_0, \mathbf{x}_1, \dots, \mathbf{x}_N$
- 弧段数量: $N$（节点数 = $N+1$）

### 3. 约束条件
对于每个弧段 $i$，从 $\mathbf{x}_i$ 积分到 $\mathbf{x}_{i+1}$ 应满足动力学方程：
$$\mathbf{F}_i = \Phi(t_i, t_{i+1}; \mathbf{x}_i) - \mathbf{x}_{i+1} = \mathbf{0}$$
其中 $\Phi$ 是流映射（flow map）。

### 4. 雅可比矩阵
雅可比矩阵由状态转移矩阵构成：
$$J = \begin{bmatrix}
\Phi_1 & -I & 0 & \cdots & 0 \\
0 & \Phi_2 & -I & \cdots & 0 \\
\vdots & \vdots & \ddots & \ddots & \vdots \\
0 & 0 & \cdots & \Phi_N & -I
\end{bmatrix}$$
其中 $\Phi_i$ 是第 $i$ 段弧的状态转移矩阵。

### 5. 迭代修正
使用牛顿法迭代：
$$\Delta \mathbf{X} = -J^{-1} \mathbf{F}$$
$$\mathbf{X}^{(k+1)} = \mathbf{X}^{(k)} + \Delta \mathbf{X}$$

## 参数选择建议

### 1. 弧段数量
- **短周期轨道**: 3-5 个弧段
- **中周期轨道**: 5-10 个弧段  
- **长周期轨道**: 10-20 个弧段
- **非常长轨道**: 20+ 个弧段

### 2. 收敛容差
- **一般精度**: `tol=1e-8`
- **高精度**: `tol=1e-12`
- **超高精度**: `tol=1e-14`

### 3. 最大迭代次数
- **简单问题**: `max_iter=50`
- **中等问题**: `max_iter=100`
- **困难问题**: `max_iter=200`

### 4. 时间变量
- **固定时间问题**: `var_time=False`
- **自由时间问题**: `var_time=True`
- **混合问题**: 部分时间固定，部分自由

## 应用场景

### 1. 长周期轨道修正
```python
# 对于周期很长的轨道，单次积分可能数值不稳定
# 使用多重打靶法分段积分
result = multiple_shooting.correct(
    t_patch=t_patch,
    state_patch=state_patch,
    max_iter=100,
    tol=1e-10,
    var_time=True
)
```

### 2. 复杂边界条件
```python
# 需要满足多个中间约束的轨道
# 可以在特定节点添加额外约束
def add_intermediate_constraint(state):
    """添加中间点约束"""
    # 例如：要求通过特定位置
    return state[0] - target_x  # x 坐标约束
```

## 性能优化

### 1. 并行计算
```python
# 各弧段积分可以并行执行
from concurrent.futures import ThreadPoolExecutor

def integrate_segment(args):
    """并行积分单个弧段"""
    i, dynamics, state_i, t_span = args
    result = dynamics.propagate(state_i, t_span, with_stm=True)
    return i, result

# 使用线程池并行积分
with ThreadPoolExecutor() as executor:
    futures = []
    for i in range(n_segments):
        args = (i, dynamics, state_patch[i], [t_patch[i], t_patch[i+1]])
        futures.append(executor.submit(integrate_segment, args))
    
    results = [f.result() for f in futures]
```

### 2. 稀疏矩阵
对于大量弧段，雅可比矩阵是稀疏的，可以使用稀疏矩阵存储和求解。

### 3. 初值猜测
好的初值可以显著提高收敛速度：
- 使用解析近似
- 从前一步结果外推
- 使用低精度结果作为高精度初值

## 常见问题

### 1. 不收敛
**可能原因**:
- 初值太差
- 弧段划分不合理
- 容差设置过严
- 动力学模型过于复杂

**解决方案**:
- 改进初值猜测
- 调整弧段数量
- 放宽容差，逐步收紧
- 使用同伦法简化问题

### 2. 数值不稳定
**可能原因**:
- 弧段长度差异太大
- 积分器容差不合适
- 状态转移矩阵计算误差

**解决方案**:
- 均匀划分时间节点
- 调整积分器参数
- 使用更高精度的积分器

### 3. 内存不足
**可能原因**:
- 弧段数量太多
- 状态维度太高
- 保存了完整的历史数据

**解决方案**:
- 减少弧段数量
- 使用稀疏矩阵
- 只保存必要的数据

## 相关算法

- [`DifferentialCorrection`](differential_correction.md): 微分修正法（单次积分）
- [`Continuation`](continuation.md): 轨道族延拓
- [`StabilityAnalysis`](stability.md): 稳定性分析

## 参考文献

1. Stoer, J., & Bulirsch, R. (2002). Introduction to Numerical Analysis. Springer.
2. Betts, J. T. (2010). Practical Methods for Optimal Control and Estimation Using Nonlinear Programming. SIAM.
3. Ascher, U. M., & Petzold, L. R. (1998). Computer Methods for Ordinary Differential Equations and Differential-Algebraic Equations. SIAM.