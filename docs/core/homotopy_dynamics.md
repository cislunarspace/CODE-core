# HomotopyEphemerisDynamics - 同伦星历动力学

`HomotopyEphemerisDynamics` 类继承自 `EphemerisDynamics`，通过同伦参数 λ 平滑过渡从基础模型到完整星历模型，适用于轨道修正和延续。

## 类定义

```python
class HomotopyEphemerisDynamics(EphemerisDynamics):
    """基于摄动天体逐步引入的同伦星历动力学
    
    将 system.bodies 分为 base_bodies 和 perturbation_bodies 两组。
    base_bodies 的引力始终以满值参与计算，perturbation_bodies 的引力乘以同伦参数 λ。
    
    Args:
        system: EphemerisSystem 对象
        base_bodies: 基础天体列表（如 ["EARTH", "MOON"]），始终满引力
        perturbation_bodies: 摄动天体列表（如 ["SUN"]），引力乘以 λ。
            若为 None，自动取 system.bodies 中不在 base_bodies 的天体。
        homotopy_param: 同伦参数 λ ∈ [0, 1]
    """
```

## 物理模型

### 同伦方程
加速度计算公式：
$$a(r, t, λ) = \sum_{b \in \text{base}} a_b(r, t) + λ \cdot \sum_{p \in \text{perturbation}} a_p(r, t)$$

其中：
- $a_b(r, t)$: 基础天体的引力加速度
- $a_p(r, t)$: 摄动天体的引力加速度
- $λ$: 同伦参数，取值范围 [0, 1]

### 物理含义
- **λ = 0**: 仅基础天体的引力（接近 CRTBP 的星历等效）
- **λ = 1**: 所有天体的完整引力（完整星历模型）
- **0 < λ < 1**: 基础天体满引力 + 摄动天体部分引力

## 主要属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `base_bodies` | `List[str]` | 基础天体列表，始终满引力 |
| `perturbation_bodies` | `List[str]` | 摄动天体列表，引力乘以 λ |
| `homotopy_param` | `float` | 同伦参数 λ |
| `_base_body_ids` | `List[int]` | 基础天体的 NAIF ID 列表 |
| `_perturbation_body_ids` | `List[int]` | 摄动天体的 NAIF ID 列表 |

## 主要方法

### `__init__(system, base_bodies, perturbation_bodies, homotopy_param)`
初始化同伦动力学。

**参数**:
- `system`: `EphemerisSystem` 对象
- `base_bodies`: 基础天体名称列表
- `perturbation_bodies`: 摄动天体名称列表（可选）
- `homotopy_param`: 同伦参数，默认 0.0

### `equations_of_motion(t, state)`
计算状态导数，考虑同伦参数。

**重写自** `EphemerisDynamics.equations_of_motion`

### `set_homotopy_param(lambda_val)`
设置同伦参数值。

**参数**:
- `lambda_val`: 新的同伦参数值，应在 [0, 1] 范围内

### `get_acceleration_breakdown(t, state)`
获取各天体加速度的详细分解。

**参数**:
- `t`: 时间（秒）
- `state`: 6维状态向量

**返回**:
- `dict`: 包含以下键的字典：
  - `total`: 总加速度
  - `base`: 基础天体贡献
  - `perturbation`: 摄动天体贡献（未乘 λ）
  - `scaled_perturbation`: 摄动天体贡献（已乘 λ）

## 使用示例

### 基本使用
```python
from e2m2e.core import EphemerisSystem, HomotopyEphemerisDynamics
from e2m2e.core.spice import SPICEManager

# 初始化 SPICE
spice_manager = SPICEManager()
spice_manager.load_kernels_from_directory("./kernels/")

# 创建系统
system = EphemerisSystem(
    bodies=["EARTH", "MOON", "SUN", "JUPITER BARYCENTER"],
    reference_epoch="2025-06-21T11:00:06"
)

# 创建同伦动力学
dynamics = HomotopyEphemerisDynamics(
    system=system,
    base_bodies=["EARTH", "MOON"],      # 基础天体
    perturbation_bodies=["SUN", "JUPITER BARYCENTER"],  # 摄动天体
    homotopy_param=0.0                  # 初始：仅地月引力
)

# 逐步增加摄动引力
for lambda_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
    dynamics.set_homotopy_param(lambda_val)
    
    # 传播轨道
    result = dynamics.propagate(
        initial_state=initial_state,
        time_span=[0, 86400]
    )
    
    print(f"λ={lambda_val}: 轨道周期 = {compute_period(result['states'])}")
```

### 轨道修正应用
```python
from e2m2e.algorithms import DifferentialCorrection

# 使用同伦法进行轨道修正
corrector = DifferentialCorrection(dynamic=dynamics)

# 从 λ=0 开始（简化模型）
dynamics.set_homotopy_param(0.0)
orbit_0 = corrector.iterate_correction(initial_guess)

# 逐步增加 λ，以前一步结果为初值
for lambda_val in [0.25, 0.5, 0.75, 1.0]:
    dynamics.set_homotopy_param(lambda_val)
    orbit = corrector.iterate_correction(
        initial_guess=orbit_prev,  # 使用前一步结果
        max_iter=100
    )
    orbit_prev = orbit
```

### 加速度分解分析
```python
# 分析各天体贡献
accel_info = dynamics.get_acceleration_breakdown(
    t=0,
    state=initial_state
)

print("基础天体贡献:", np.linalg.norm(accel_info["base"]))
print("摄动天体贡献:", np.linalg.norm(accel_info["perturbation"]))
print("缩放后摄动贡献:", np.linalg.norm(accel_info["scaled_perturbation"]))
print("总加速度:", np.linalg.norm(accel_info["total"]))
```

## 应用场景

### 1. 轨道延续
从简化模型（λ=0）开始，逐步增加摄动引力，平滑过渡到完整模型。

### 2. 收敛性改善
复杂模型可能难以直接收敛，同伦法提供渐进式求解路径。

### 3. 敏感性分析
通过改变 λ 值，分析摄动天体对轨道的影响程度。

### 4. 模型验证
比较 λ=0（简化模型）和 λ=1（完整模型）的结果，验证模型一致性。

## 性能建议

1. **λ 步长**: 建议使用较小的步长（如 0.1-0.25）以确保平滑过渡
2. **收敛检测**: 每个 λ 值下确保轨道修正充分收敛
3. **初值传递**: 使用前一步结果作为下一步初值，提高效率
4. **参数扫描**: 可对 λ 进行参数扫描，研究模型过渡特性

## 注意事项

1. **天体分组**: 合理选择基础天体和摄动天体
2. **参数范围**: λ 应在 [0, 1] 范围内
3. **数值稳定性**: 小 λ 值可能导致数值问题，适当调整容差
4. **物理意义**: 确保分组符合物理实际（如地月为基础，太阳为摄动）

## 相关类

- [`EphemerisDynamics`](ephemeris_dynamics.md): 父类，完整星历动力学
- [`EphemerisSystem`](ephemeris_system.md): 星历系统定义
- [`DifferentialCorrection`](../algorithms/differential_correction.md): 微分修正算法
- [`MultipleShooting`](../algorithms/multiple_shooting.md): 多重打靶法