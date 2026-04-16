---
title: 坐标变换与参考系
---

# 坐标变换与参考系（CoordinateTransformation & ReferenceFrame）

**文件**: `e2m2e/core/coordinate.py`

## 设计原理

提供旋转坐标系（会合坐标系）与惯性坐标系之间的变换。

## 旋转系↔惯性系变换

### 旋转系到惯性系

```python
def rotating_to_inertial(state_rotating: np.ndarray, theta: float) -> np.ndarray:
    """
    旋转系状态转惯性系
    
    参数:
        state_rotating: 旋转系状态 [x, y, z, vx, vy, vz]
        theta: 旋转角度 (rad)
    
    返回:
        惯性系状态
    """
    # 位置变换
    x_rot = state_rotating[0]
    y_rot = state_rotating[1]
    x_inert = x_rot * np.cos(theta) - y_rot * np.sin(theta)
    y_inert = x_rot * np.sin(theta) + y_rot * np.cos(theta)
    z_inert = state_rotating[2]
    
    # 速度变换
    vx_rot = state_rotating[3]
    vy_rot = state_rotating[4]
    vx_inert = vx_rot * np.cos(theta) - vy_rot * np.sin(theta) - y_inert
    vy_inert = vx_rot * np.sin(theta) + vy_rot * np.cos(theta) + x_inert
    vz_inert = state_rotating[5]
    
    return np.array([x_inert, y_inert, z_inert, vx_inert, vy_inert, vz_inert])
```

## 核心方法

| 方法 | 说明 |
|------|------|
| `rotating_to_inertial(state, theta)` | 旋转系→惯性系 |
| `inertial_to_rotating(state, theta)` | 惯性系→旋转系 |
| `compute_rotation_angle(t)` | 计算旋转角度 |

## ReferenceFrame 枚举

```python
class ReferenceFrame(Enum):
    ROTATING = "rotating"      # 旋转系（会合坐标系）
    INERTIAL = "inertial"      # 惯性系
```
