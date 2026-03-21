# Orbit & OrbitFamily

**文件**: `e2m2e/core/orbit.py`

## 设计原理

`Orbit` 类封装轨道数据管理功能，包括周期检测和稳定性分析。`OrbitFamily` 类管理一族轨道数据。

## Orbit 类

### 核心属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `states` | `np.ndarray` | 轨道状态序列 [n, 6] |
| `times` | `np.ndarray` | 对应时间序列 |
| `period` | `float` | 轨道周期（检测得到） |
| `stability_index` | `float` | 稳定性指数 |
| `jacobi_constant` | `float` | Jacobi常数 |

### 核心方法

| 方法 | 说明 |
|------|------|
| `detect_period()` | 检测轨道周期 |
| `interpolate_at_time(t)` | 时间插值获取状态 |
| `compute_stability()` | 计算Floquet乘子 |
| `compute_jacobi()` | 计算Jacobi常数 |

## OrbitFamily 类

### 核心属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `orbits` | `List[Orbit]` | 轨道列表 |
| `parameter_name` | `str` | 参数名称（如 "x0", "mu"） |
| `parameter_values` | `List[float]` | 参数值序列 |

### 核心方法

| 方法 | 说明 |
|------|------|
| `add_orbit(orbit, param_value)` | 添加轨道到族 |
| `get_orbit_at(param_value)` | 获取指定参数的轨道 |
| `filter_by_stability(stability_type)` | 按稳定性筛选 |

## 使用示例

```python
from e2m2e.core.orbit import Orbit, OrbitFamily

# 创建轨道
orbit = Orbit(states=states, times=times)
orbit.detect_period()

# 创建轨道族
family = OrbitFamily(parameter_name="x0")
family.add_orbit(orbit, param_value=0.8)
```
