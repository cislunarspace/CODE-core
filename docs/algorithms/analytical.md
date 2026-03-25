# Analytical Approximation Module

**文件**: `e2m2e/algorithms/analytical.py`

本模块提供Halo轨道的解析近似方法，基于Richardson三阶理论生成高质量的初始猜测。

## compute_halo_coefficients

**签名**:
```python
def compute_halo_coefficients(mu: float, L: int) -> Dict[str, float]
```

计算Halo轨道三阶近似的系数。

**参数**:
| 参数 | 类型 | 说明 |
|------|------|------|
| mu | float | 质量比 (CR3BP系统的小天体质量比) |
| L | int | 拉格朗日点 (1=L1, 2=L2) |

**返回**: `Dict[str, float]` - 包含以下系数的字典:
- `gamma`: L点距离参数
- `c1, c2, c3`: 常数系数
- `a21, a22, a23, a24, a31`: u分量系数
- `b21, b22, b31`: v分量系数
- `d21, d31, d32`: w分量系数
- `k, delta`: 分类参数
- `l1, l2, l3`: 周期修正系数
- `kappa1, kappa2`: 振幅系数

**示例**:
```python
from e2m2e.algorithms.analytical import compute_halo_coefficients

coeffs = compute_halo_coefficients(mu=0.01215, L=1)
print(coeffs["kappa1"])  # 输出: 0.1735...
```

## halo_third_order_approximation

**签名**:
```python
def halo_third_order_approximation(
    mu: float,
    Au: float,
    Aw: float,
    phi: float,
    L: int,
    tf: float,
    N: int,
    halo_class: int = 0,
) -> Tuple[npt.NDArray, npt.NDArray, float]
```

计算Halo轨道三阶解析近似的状态序列。

**参数**:
| 参数 | 类型 | 说明 |
|------|------|------|
| mu | float | 质量比 |
| Au | float | U方向振幅 |
| Aw | float | W方向振幅 |
| phi | float | 相位偏移 (弧度) |
| L | int | 拉格朗日点 (1=L1, 2=L2) |
| tf | float | 终止时间 |
| N | int | 采样点数量 |
| halo_class | int | 0=Class I (北), 1=Class II (南) |

**返回**:
- `SV_uvw`: 状态向量序列 (N, 6)，格式为 `[x, y, z, vx, vy, vz]`
- `t`: 时间序列
- `T`: 周期

**数学公式**:

状态分量在旋转坐标系下表示为：

$$u(\tau) = a_{21}A_u^2 + a_{22}A_w^2 - A_u\cos(\tau + \phi) + (a_{23}A_u^2 - a_{24}A_w^2)\cos(2(\tau + \phi)) + a_{31}A_u^3\cos(3(\tau + \phi))$$

$$v(\tau) = kA_u\sin(\tau + \phi) + (b_{21}A_u^2 - b_{22}A_w^2)\sin(2(\tau + \phi)) + b_{31}A_u^3\sin(3(\tau + \phi))$$

$$w(\tau) = \delta\left[A_w\cos(\tau + \phi) + d_{21}A_uA_w(\cos(2(\tau + \phi)) - 3) + (d_{32}A_wA_u^2 - d_{31}A_w^3)\cos(3(\tau + \phi))\right]$$

速度分量为：

$$u'(\tau) = A_u\sin(\tau + \phi) + 2(a_{23}A_u^2 - a_{24}A_w^2)\sin(2(\tau + \phi))$$

$$v'(\tau) = kA_u\cos(\tau + \phi) + 2(b_{21}A_u^2 - b_{22}A_w^2)\cos(2(\tau + \phi))$$

$$w'(\tau) = -A_w\sin(\tau + \phi) - 2d_{21}A_uA_w\sin(2(\tau + \phi))$$

周期时间为：

$$T = 2\pi(1 + \kappa_1A_u^2 + \kappa_2A_w^2)$$

**示例**:
```python
from e2m2e.algorithms.analytical import halo_third_order_approximation
import numpy as np

mu = 0.01215
Au = 0.01
Aw = 0.1
phi = 0.0

SV, t, T = halo_third_order_approximation(
    mu=mu,
    Au=Au,
    Aw=Aw,
    phi=phi,
    L=1,
    tf=3.0,
    N=100,
    halo_class=0
)

print(f"轨道周期: {T:.4f} 时间单位")
```

## compute_halo_initial_guess

**签名**:
```python
def compute_halo_initial_guess(
    mu: float,
    z_amplitude: float,
    L: int = 1,
    halo_class: int = 0,
) -> Dict[str, float]
```

计算Halo轨道初始猜测参数，用于配合微分修正器生成精确周期轨道。

**参数**:
| 参数 | 类型 | 说明 |
|------|------|------|
| mu | float | 质量比 |
| z_amplitude | float | Z方向振幅 (必须为正数) |
| L | int | 拉格朗日点 (1=L1, 2=L2) |
| halo_class | int | 0=北Halo, 1=南Halo |

**返回**: `Dict[str, float]` - 包含初始猜测参数:
| 键 | 说明 |
|---|------|
| x0 | 初始x坐标 |
| y0 | 初始y坐标 (固定为0) |
| z0 | 初始z坐标 (固定为0) |
| vx0 | 初始vx (固定为0) |
| vy0 | 初始vy |
| vz0 | 初始vz (固定为0) |
| T_half | 半周期 |
| Au | U方向振幅 |
| Aw | W方向振幅 |

**振幅关系**:

Z方向振幅 $A_w$ 与U方向振幅 $A_u$ 的关系为：

$$A_u = \sqrt{\frac{-\kappa_1}{l_1}} A_w$$

**初始状态计算**:

对于L1点：
$$x_0 = L_1 - a_{21}A_u^2 - a_{22}A_w^2 + A_u$$

对于L2点：
$$x_0 = L_2 - a_{21}A_u^2 - a_{22}A_w^2 - A_u$$

$$v_{y0} = -kA_u(1 + l_1A_u^2 + l_2A_w^2)$$

**示例**:
```python
from e2m2e.algorithms.analytical import compute_halo_initial_guess

guess = compute_halo_initial_guess(
    mu=0.01215,
    z_amplitude=0.1,
    L=1,
    halo_class=0
)

print(f"初始状态: x0={guess['x0']:.6f}, vy0={guess['vy0']:.6f}")
print(f"半周期: {guess['T_half']:.4f}")
```

## 模块设计

本模块采用分层设计：

1. **compute_halo_coefficients**: 计算基础系数，仅依赖质量比和L点选择
2. **halo_third_order_approximation**: 基于系数计算完整状态序列
3. **compute_halo_initial_guess**: 封装为微分修正器可直接使用的初始猜测格式

这种设计确保系数计算的一次性，避免重复运算。

## 参考资料

- Richardson, D. L. (1980). Analytic construction of periodic orbits about the collinear points. *Celestial Mechanics*, 22(3), 241-253.
