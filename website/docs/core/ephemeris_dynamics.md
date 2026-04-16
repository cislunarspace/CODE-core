---
title: 星历动力学 EphemerisDynamics
---

# 星历动力学（EphemerisDynamics）

`EphemerisDynamics` 类继承自 `Dynamics` 基类，提供基于 NASA SPICE 内核的精确多天体引力计算。

## 类定义

```python
class EphemerisDynamics(Dynamics):
    """基于 SPICE 内核的星历动力学
    
    计算多个天体对航天器的引力加速度，考虑实际天体的非球形引力和精确星历位置。
    
    Args:
        system: EphemerisSystem 对象，定义参与计算的天体
    """
```

## 继承关系

```
Dynamics (基类)
    ├── CR3BP_Dynamics (CR3BP 动力学)
    └── EphemerisDynamics (星历动力学)
```

## 主要方法

### `__init__(system)`
初始化星历动力学。

**参数**:
- `system`: `EphemerisSystem` 对象

### `equations_of_motion(t, state)`
计算状态导数（右端函数值）。

**参数**:
- `t`: 时间（秒），相对于参考历元
- `state`: 6维状态向量 [x, y, z, vx, vy, vz]（km, km/s）

**返回**:
- `np.ndarray`: 6维状态导数 [vx, vy, vz, ax, ay, az]

**物理模型**:
加速度计算公式：
$$a(r, t) = \sum_{i=1}^{N} \mu_i \frac{r_i - r}{\|r_i - r\|^3}$$
其中：
- $r$ 是航天器位置
- $r_i$ 是第 $i$ 个天体的位置
- $\mu_i$ 是第 $i$ 个天体的引力常数

### `propagate(initial_state, time_span, **kwargs)`
传播轨道。

**参数**:
- `initial_state`: 初始状态向量
- `time_span`: 时间区间 [t_start, t_end] 或积分时间列表
- `**kwargs`: 传递给 `solve_ivp` 的额外参数

**返回**:
- `dict`: 包含以下键的字典：
  - `states`: 状态历史，形状 (n_times, 6)
  - `times`: 时间点数组
  - `stm`: 状态转移矩阵历史（如果 requested）

## 使用示例

### 基本使用
```python
from e2m2e.core import EphemerisSystem, EphemerisDynamics
from e2m2e.core.spice import SPICEManager

# 初始化 SPICE
spice_manager = SPICEManager()
spice_manager.load_kernels_from_directory("./kernels/")

# 创建系统
system = EphemerisSystem(
    bodies=["EARTH", "MOON", "SUN"],
    reference_epoch="2025-06-21T11:00:06"
)

# 创建动力学
dynamics = EphemerisDynamics(system=system)

# 定义初始状态（地月 L2 点附近）
initial_state = [1.1556 * 384400, 0, 0, 0, 1.023, 0]  # km, km/s

# 传播轨道
result = dynamics.propagate(
    initial_state=initial_state,
    time_span=[0, 7 * 86400],  # 传播7天
    method='DOP853',
    rtol=1e-12,
    atol=1e-14
)

# 获取结果
states = result["states"]
times = result["times"]
```

### 计算状态转移矩阵
```python
# 传播并计算 STM
result_with_stm = dynamics.propagate(
    initial_state=initial_state,
    time_span=[0, 86400],
    with_stm=True
)

stm_history = result_with_stm["stm"]  # 形状 (n_times, 6, 6)
```

### 与算法结合使用
```python
from e2m2e.algorithms import DifferentialCorrection

# 创建微分修正器
corrector = DifferentialCorrection(dynamic=dynamics)

# 配置对称性（示例）
corrector.setup_2D_symmetric_x_fixed_x0(x0=1.1556)

# 执行修正
initial_guess = Orbit(states=[initial_state], times=[0])
initial_guess.period = 6.8 * 86400  # 估计周期（秒）

corrected_orbit = corrector.iterate_correction(initial_guess=initial_guess)
```

## 性能考虑

1. **计算成本**: 星历动力学计算比 CR3BP 更昂贵，因为需要：
   - 查询每个天体的精确位置
   - 计算每个天体的引力加速度
   - 处理非球形引力项（如果内核支持）

2. **积分器选择**: 推荐使用高阶积分器：
   - `'DOP853'`: 8阶显式 Runge-Kutta，适合高精度需求
   - `'RK45'`: 5阶 Runge-Kutta，平衡精度和速度

3. **容差设置**: 建议使用较小的容差：
   - `rtol=1e-12`: 相对容差
   - `atol=1e-14`: 绝对容差

## 注意事项

1. **SPICE 内核**: 必须提前加载所需的内核文件
2. **时间系统**: 使用星历时间（ET），注意与 UTC 的转换
3. **单位一致性**: 确保所有输入使用一致的单位（km, km/s）
4. **内存使用**: 长时间传播可能产生大量数据，注意内存管理

## 相关类

- [`EphemerisSystem`](ephemeris_system.md): 星历系统定义
- [`CR3BP_Dynamics`](dynamics.md): CR3BP 动力学（简化模型）
- [`Dynamics`](dynamics.md): 动力学基类