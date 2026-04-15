---
title: 系统定义：CR3BP_System
---

# 系统定义：CR3BP_System

> **文件**: `e2m2e/core/system.py`

`CR3BP_System` 封装了圆型限制性三体问题 (CR3BP) 的系统参数——质量参数、特征尺度、平动点位置、Jacobi 常数。它是使用 e2m2e 的第一步：所有动力学模型、轨道算法和可视化都需要一个系统对象。

## 怎么创建一个天体系统

### 内置系统（推荐）

```python
from e2m2e.core import CR3BP_System

system = CR3BP_System.from_known_system("earth_moon")
# 其他可选: "sun_earth", "sun_jupiter"
```

`from_known_system` 会自动设置质量参数 $\mu$ 和天体名称。

### 自定义系统

```python
system = CR3BP_System(mu=0.01215, primary_body="Earth", secondary_body="Moon")
```

### 设置特征尺度

如果需要在无量纲和物理单位之间转换，必须先设置特征尺度：

```python
system.set_characteristic_scales(
    distance=384400,       # 地月距离 (km)
    period=27.32 * 86400,  # 月球轨道周期 (s)
)
```

设置后可以使用 `dimensionless_to_physical()` 和 `physical_to_dimensionless()` 进行单位转换。

## 怎么使用系统信息

### 计算平动点

```python
system.compute_libration_points()

# 获取 L1 点位置
L1 = system.L1  # 或 system.get_libration_point(LibrationPoint.L1)
print(f"L1: {L1}")
```

五个平动点 `L1` ~ `L5` 计算后直接作为属性可用。

### 计算 Jacobi 常数

Jacobi 常数是 CR3BP 中的守恒量，用于衡量轨道的能量水平：

```python
import numpy as np

state = np.array([0.8, 0.1, 0.0, 0.0, 0.2, 0.0])
C = system.get_jacobi_constant(state)
print(f"Jacobi 常数: {C:.6f}")
```

Jacobi 常数越大，轨道能量越低。平动点的 Jacobi 常数是区分不同运动区域的关键阈值。

### 单位转换

```python
# 无量纲 → 物理 (km, km/s)
physical = system.dimensionless_to_physical(state)

# 物理 → 无量纲
dimensionless = system.physical_to_dimensionless(physical)
```

## API 速查

| 方法 | 说明 |
|------|------|
| `from_known_system(name)` | 创建内置系统（"earth_moon" / "sun_earth" / "sun_jupiter"） |
| `set_characteristic_scales(distance, period)` | 设置特征尺度（物理单位转换前置步骤） |
| `compute_libration_points()` | 计算五个平动点位置 |
| `get_libration_point(point)` | 获取指定平动点坐标 |
| `get_jacobi_constant(state)` | 计算 Jacobi 常数 |
| `dimensionless_to_physical(state)` | 无量纲 → 物理单位 |
| `physical_to_dimensionless(state)` | 物理单位 → 无量纲 |
| `compute_stability_index(L_point)` | 计算平动点线性化稳定性指标 |
| `info(mode)` | 打印系统信息 |

完整 API 文档见 [API 参考](../reference/api-reference.md)。

## 数学背景

### 质量参数

$$\mu = \frac{m_2}{m_1 + m_2}$$

地月系统 $\mu \approx 0.01215$。

### 平动点

平动点满足有效势函数梯度为零：$\nabla U(\mathbf{r}) = 0$，其中

$$U = \frac{x^2 + y^2}{2} + \frac{1-\mu}{r_1} + \frac{\mu}{r_2}$$

L1、L2、L3 位于 $x$ 轴上（共线平动点），L4、L5 构成等边三角形。

### Jacobi 常数

$$C = 2U - v^2 = x^2 + y^2 + \frac{2(1-\mu)}{r_1} + \frac{2\mu}{r_2} - (v_x^2 + v_y^2 + v_z^2)$$

在无量纲旋转坐标系中，Jacobi 常数沿轨迹守恒。
