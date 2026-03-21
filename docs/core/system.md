# CR3BP_System

**文件**: `e2m2e/core/system.py`

**类签名**:
```python
class CR3BP_System:
    """圆型限制性三体问题系统参数"""
```

## 设计原理

`CR3BP_System` 类封装了圆型限制性三体问题 (Circular Restricted Three-Body Problem) 的系统参数。在CR3BP模型中：
- 两个大质量天体（主天体 $m_1$ 和次天体 $m_2$）在它们相互的引力作用下围绕共同的质心做圆轨道运动
- 一个小质量天体（探测器）在上述两个大天体的引力场中运动，其质量对两个大天体的运动没有影响

质量参数定义为：
$$\mu = \frac{m_2}{m_1 + m_2}$$

地月系统的 $\mu \approx 0.01215$

## 数学基础

### 平动点（Libration Points）计算
平动点是相对于两个大天体保持静止的特殊点，满足：
$$\nabla U(\mathbf{r}) = 0$$

其中 $U$ 是有效势函数：
$$U = \frac{x^2 + y^2}{2} + \frac{1-\mu}{r_1} + \frac{\mu}{r_2}$$

### Jacobi常数
$$C = 2U - v^2 = x^2 + y^2 + \frac{2(1-\mu)}{r_1} + \frac{2\mu}{r_2} - (v_x^2 + v_y^2 + v_z^2)$$

## 属性

| 属性名 | 类型 | 说明 |
|--------|------|------|
| `mu` | `float` | 质量参数 $\mu = m_2/(m_1+m_2)$ |
| `primary_body` | `str` | 主天体名称 |
| `secondary_body` | `str` | 次天体名称 |
| `L1-L5` | `np.ndarray` | 五个平动点的坐标 |
| `characteristic_length` | `float` | 特征长度（两天体间距离） |
| `characteristic_time` | `float` | 特征时间 |
| `characteristic_velocity` | `float` | 特征速度 |

## 核心方法

| 方法 | 说明 |
|------|------|
| `compute_libration_points()` | 计算五个平动点位置 |
| `get_libration_point(point)` | 获取指定平动点坐标 |
| `get_jacobi_constant(state)` | 计算Jacobi常数 |
| `dimensionless_to_physical(state)` | 无量纲→物理单位 |
| `physical_to_dimensionless(state)` | 物理单位→无量纲 |
| `compute_stability_index(L_point)` | 计算平动点稳定性指标 |

## 使用示例

```python
from e2m2e.core.system import CR3BP_System, LibrationPoint

# 从已知系统创建
system = CR3BP_System.from_known_system("earth_moon")
system.set_characteristic_scales(distance=384400, period=27.32*86400)
system.compute_libration_points()

# 获取平动点
L1 = system.get_libration_point(LibrationPoint.L1)
print(f"L1位置: {L1}")

# 计算Jacobi常数
state = np.array([0.8, 0, 0, 0, 1.5, 0])
C = system.get_jacobi_constant(state)
```
