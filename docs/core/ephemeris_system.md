# EphemerisSystem - 星历系统

`EphemerisSystem` 类用于定义基于 NASA SPICE 内核的星历系统，支持多天体引力计算。

## 类定义

```python
class EphemerisSystem:
    """基于 SPICE 内核的星历系统
    
    定义参与引力计算的天体列表和参考历元，支持精确的多天体星历计算。
    
    Args:
        bodies: 天体名称列表，如 ["EARTH", "MOON", "SUN"]
        reference_epoch: 参考历元字符串，格式为 "YYYY-MM-DDTHH:MM:SS"
    """
```

## 主要属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `bodies` | `List[str]` | 参与引力计算的天体名称列表 |
| `reference_epoch` | `str` | 参考历元，用于初始化 SPICE 时间 |
| `epoch_et` | `float` | 参考历元对应的星历时间（秒） |
| `body_ids` | `Dict[str, int]` | 天体名称到 NAIF ID 的映射 |

## 主要方法

### `__init__(bodies, reference_epoch)`
初始化星历系统。

**参数**:
- `bodies`: 天体名称列表，使用标准 NAIF 名称（如 "EARTH", "MOON", "SUN"）
- `reference_epoch`: 参考历元，格式为 ISO 8601 字符串

**示例**:
```python
from e2m2e.core import EphemerisSystem

system = EphemerisSystem(
    bodies=["EARTH", "MOON", "SUN"],
    reference_epoch="2025-06-21T11:00:06"
)
```

### `get_body_position(body_name, et)`
获取指定天体在给定星历时间的位置。

**参数**:
- `body_name`: 天体名称
- `et`: 星历时间（秒）

**返回**:
- `np.ndarray`: 3维位置向量（km），在 J2000 惯性系中

### `get_body_state(body_name, et)`
获取指定天体在给定星历时间的状态（位置和速度）。

**参数**:
- `body_name`: 天体名称
- `et`: 星历时间（秒）

**返回**:
- `np.ndarray`: 6维状态向量 [x, y, z, vx, vy, vz]（km, km/s）

## 使用示例

### 基本使用
```python
from e2m2e.core import EphemerisSystem
from e2m2e.core.spice import SPICEManager

# 初始化 SPICE 管理器并加载内核
spice_manager = SPICEManager()
spice_manager.load_kernels_from_directory("./kernels/")

# 创建星历系统
system = EphemerisSystem(
    bodies=["EARTH", "MOON", "SUN"],
    reference_epoch="2025-06-21T11:00:06"
)

# 获取天体位置
et = system.epoch_et + 86400  # 参考历元后1天
earth_pos = system.get_body_position("EARTH", et)
moon_pos = system.get_body_position("MOON", et)
sun_pos = system.get_body_position("SUN", et)
```

### 与动力学结合使用
```python
from e2m2e.core import EphemerisDynamics

# 创建星历动力学
dynamics = EphemerisDynamics(system=system)

# 传播轨道
initial_state = [384400, 0, 0, 0, 1023, 0]  # 初始状态 (km, km/s)
result = dynamics.propagate(initial_state, time_span=[0, 86400])
```

## 注意事项

1. **SPICE 内核**: 使用前必须加载相应的 SPICE 内核文件
2. **时间系统**: 使用星历时间（Ephemeris Time, ET），单位为秒
3. **坐标系**: 所有位置和速度都在 J2000 惯性系中
4. **单位**: 位置单位为 km，速度单位为 km/s

## 相关类

- [`EphemerisDynamics`](ephemeris_dynamics.md): 星历动力学计算
- [`SPICEManager`](spice.md): SPICE 内核管理