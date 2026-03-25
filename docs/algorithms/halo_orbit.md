# HaloOrbitGenerator

**文件**: `e2m2e/algorithms/halo_orbit.py`

**类签名**:
```python
class HaloOrbitGenerator:
    """Halo轨道生成器"""
```

## 功能概述

Halo轨道生成器结合Richardson三阶解析近似和微分修正算法，用于生成精确的Halo周期轨道。Halo轨道是一类在拉格朗日点附近的三维周期轨道，因其形状类似光晕而得名。

## 算法原理

### Richardson 三阶近似

Halo轨道的解析近似基于Richardson三阶理论。对于L1/L2点附近的Halo轨道，状态变量在旋转坐标系下可表示为：

$$u = a_{21}A_u^2 + a_{22}A_w^2 - A_u\cos(\tau + \phi) + (a_{23}A_u^2 - a_{24}A_w^2)\cos(2(\tau + \phi)) + a_{31}A_u^3\cos(3(\tau + \phi))$$

$$v = kA_u\sin(\tau + \phi) + (b_{21}A_u^2 - b_{22}A_w^2)\sin(2(\tau + \phi)) + b_{31}A_u^3\sin(3(\tau + \phi))$$

$$w = \delta\left[A_w\cos(\tau + \phi) + d_{21}A_uA_w(\cos(2(\tau + \phi)) - 3) + (d_{32}A_wA_u^2 - d_{31}A_w^3)\cos(3(\tau + \phi))\right]$$

周期时间为：

$$T = 2\pi(1 + \kappa_1A_u^2 + \kappa_2A_w^2)$$

其中 $A_u$ 和 $A_w$ 分别是U方向和W方向的振幅。

### 微分修正

生成的解析近似轨道通过微分修正器迭代优化，以满足精确的周期边界条件：

$$\mathbf{x}(T) - \mathbf{x}(0) = \mathbf{0}$$

## Halo轨道分类

| 类别 | halo_class | 特征 |
|------|------------|------|
| 北Halo (Class I) | 0 | z方向振幅为正 |
| 南Halo (Class II) | 1 | z方向振幅为负 |

## 核心方法

| 方法 | 说明 |
|------|------|
| `generate_seed_orbit(libration_point, amplitude_z, halo_class)` | 生成单条Halo轨道 |
| `generate_family(seed_orbit, n_orbits, direction)` | 生成Halo轨道族 |

## 使用示例

```python
from e2m2e.core import CR3BP_System
from e2m2e.algorithms import HaloOrbitGenerator

system = CR3BP_System.from_known_system("earth_moon")
generator = HaloOrbitGenerator(system)

# 生成L1点北Halo轨道
halo = generator.generate_seed_orbit(
    libration_point=1,
    amplitude_z=0.1,
    halo_class=0
)

# 生成L2点南Halo轨道
halo_l2_south = generator.generate_seed_orbit(
    libration_point=2,
    amplitude_z=0.05,
    halo_class=1
)

# 生成Halo轨道族
family = generator.generate_family(
    seed_orbit=halo,
    n_orbits=50,
    direction="positive",
    step_size=0.001
)
```

## generate_seed_orbit 详解

**签名**:
```python
def generate_seed_orbit(
    self,
    libration_point: int,
    amplitude_z: float,
    halo_class: int = 0,
    verbose: bool = False,
) -> Orbit
```

**参数**:
| 参数 | 类型 | 说明 |
|------|------|------|
| libration_point | int | 拉格朗日点 (1=L1, 2=L2) |
| amplitude_z | float | Z方向振幅 (必须为正数) |
| halo_class | int | 0=北Halo, 1=南Halo |
| verbose | bool | 是否打印迭代详细信息 |

**返回**: `Orbit` - Halo周期轨道对象

## generate_family 详解

**签名**:
```python
def generate_family(
    self,
    seed_orbit: Orbit,
    n_orbits: int = 50,
    direction: str = "positive",
    step_size: float = 0.001,
) -> List[Orbit]
```

**参数**:
| 参数 | 类型 | 说明 |
|------|------|------|
| seed_orbit | Orbit | 种子轨道 |
| n_orbits | int | 目标轨道数量 |
| direction | str | 延拓方向 ("positive", "negative", "both") |
| step_size | float | 参数步长 |

**返回**: `List[Orbit]` - Halo轨道族列表

## 参考资料

- Richardson, D. L. (1980). Analytic construction of periodic orbits about the collinear points. *Celestial Mechanics*, 22(3), 241-253.
- Breakwell, J. V., & Brown, J. V. (1979). The 'halo' family of 3-dimensional periodic orbits in the earth-moon restricted 3-body problem. *Celestial Mechanics*, 20(4), 389-404.
