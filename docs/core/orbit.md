# 轨道数据：Orbit 与 OrbitFamily

> **文件**: `e2m2e/core/orbit.py`

`Orbit` 存储单条轨道的状态序列和属性（周期、Jacobi 常数、稳定性）。`OrbitFamily` 管理一族轨道，通常由延拓算法生成。

## 怎么创建轨道对象

通常不需要手动创建——`DifferentialCorrection.iterate_correction()` 和 `Dynamics.propagate()` 的返回值已经是 `Orbit` 对象。

手动创建：

```python
import numpy as np
from e2m2e.core import CR3BP_System, Orbit

system = CR3BP_System.from_known_system("earth_moon")
states = np.array([[0.8, 0, 0, 0, 0.5, 0], ...])  # [n, 6]
times = np.array([0.0, 0.01, 0.02, ...])

orbit = Orbit(states=states, times=times, system=system)
```

创建时自动计算基本属性（周期、振幅、Jacobi 常数）。

## 怎么分析轨道属性

### 周期和振幅

```python
print(f"周期: {orbit.period:.4f}")
print(f"x 振幅: {orbit.get_amplitude('x'):.6f}")
print(f"y 振幅: {orbit.get_amplitude('y'):.6f}")
```

### 稳定性分析

```python
from e2m2e.core import CR3BP_Dynamics

dynamics = CR3BP_Dynamics(system)

# 计算 Monodromy 矩阵（一个周期的 STM）
orbit.compute_monodromy_matrix(dynamics)

# 计算 Floquet 乘子和稳定性
stability = orbit.compute_stability(dynamics)
print(f"稳定性: {stability['type']}")
print(f"Lyapunov 指数: {stability['lyapunov_exponents']}")
```

→ 详见 [稳定性分析](../algorithms/stability.md)

## 怎么保存和加载轨道

### 单条轨道

```python
orbit.save_to_file("output/my_orbit.json")
loaded = Orbit.load_from_file("output/my_orbit.json", system=system)
```

### 轨道族

```python
# 保存（通常由 Continuation 返回的 OrbitFamily 直接调用）
family.save_to_file("output/dro_family.json")

# 加载
from e2m2e.core.orbit import OrbitFamily
family = OrbitFamily.load_from_file("output/dro_family.json", system=system)
```

JSON 格式包含完整状态序列和元数据，可跨会话复用。

## 怎么管理轨道族

`OrbitFamily` 通常由 [Continuation](../algorithms/continuation.md) 算法自动生成。手动构建：

```python
from e2m2e.core.orbit import OrbitFamily

family = OrbitFamily(orbits=[orbit1, orbit2, orbit3], family_type="DRO", system=system)

# 访问
print(f"轨道数: {len(family)}")
print(f"Jacobi 常数范围: {family.get_jacobi_constants()}")

# 迭代
for orbit in family:
    print(orbit.period)
```

→ 轨道族的延拓生成详见 [轨道族延拓](../algorithms/continuation.md)
→ 轨道族可视化详见 [可视化指南](../guides/visualization-guide.md)

## API 速查

### Orbit

| 方法 | 说明 |
|------|------|
| `get_period()` | 获取轨道周期 |
| `get_amplitude(direction)` | 获取指定方向振幅（"x" / "y" / "z"） |
| `compute_monodromy_matrix(dynamics)` | 计算 Monodromy 矩阵 |
| `compute_stability(dynamics)` | 计算 Floquet 稳定性 |
| `save_to_file(filename)` | 保存到 JSON |
| `load_from_file(filename, system)` | 从 JSON 加载 |
| `copy()` | 深拷贝 |

### OrbitFamily

| 方法 | 说明 |
|------|------|
| `add_orbit(orbit)` | 添加轨道 |
| `get_states()` / `.states` | 所有轨道初始状态 (n, 6) |
| `get_periods()` / `.periods` | 所有轨道周期 |
| `get_jacobi_constants()` | 所有轨道 Jacobi 常数 |
| `save_to_file(filename)` | 保存到 JSON |
| `load_from_file(filename, system)` | 从 JSON 加载 |

完整 API 文档见 [API 参考](../reference/api-reference.md)。
