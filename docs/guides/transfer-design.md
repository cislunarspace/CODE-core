# 转移轨道设计指南

## 概述

E2M2E 提供多种转移轨道设计策略，从简单的直接转移到复杂的不变流形转移。

## 转移策略分类

| 策略 | 复杂度 | ΔV | 适用场景 |
|------|--------|-----|----------|
| 直接转移 | 低 | 高 | 快速转移 |
| 低能转移 | 中 | 中 | 燃料优化 |
| 流形转移 | 高 | 低 | 最优轨道间转移 |
| 异宿连接 | 高 | 极低 | L1↔L2 转移 |

---

## 直接转移

### 地球 → 月球（DRO/Halo）

```python
from e2m2e.transfer.earth_moon import EarthMoonTransfer

# 创建转移器
transfer = EarthMoonTransfer(system, dynamics)

# 设计直接转移
result = transfer.design_direct_transfer(
    r_departure=0.017,    # LEO 轨道半径（无量纲）
    r_arrival=0.25,       # 月球轨道半径（无量纲）
    t_transfer_guess=5.0  # 转移时间猜测（天）
)

print(f"转移时间: {result['transfer_time']:.2f} 天")
print(f"总 ΔV: {result['delta_v_total']:.2f} km/s")
```

### 轨道间直接转移

```python
from e2m2e.transfer.inter_orbit import InterOrbitTransfer

transfer = InterOrbitTransfer(system, dynamics)

# 在两条轨道间搜索最优转移
result = transfer.design_direct_transfer(
    orbit_departure=orbit1,
    orbit_arrival=orbit2,
    n_search=20
)

print(f"最优 ΔV: {result['delta_v']:.3f} km/s")
```

---

## 低能转移

### 经 L1/L2 的低能转移

```python
# 设计经 L1 的低能转移
result = transfer.design_low_energy_transfer(
    target_orbit=target_dro,
    libration_point='L1',
    approach_type='backward'  # 或 'forward'
)

print(f"转移时间: {result['transfer_time']:.1f} 天")
print(f"ΔV 节省: {result['savings']:.1f}%")
```

### 弱稳定边界 (WSB) 转移

```python
# 利用弱稳定边界特性
result = transfer.design_manifold_transfer(
    target_orbit=dro,
    manifold_type='wsb',  # 弱稳定边界流形
    n_manifolds=50
)
```

---

## 不变流形转移

### 同宿转移 (Homoclinic)

连接同一轨道的不稳定流形与稳定流形：

```python
# 设计同宿转移
result = transfer.design_homoclinic_transfer(
    orbit=halo_orbit,
    manifold_direction='unstable',  # 不稳定流形方向
    n_manifolds=30
)

print(f"转移时间: {result['transfer_time']:.1f} 无量纲时间")
print(f"ΔV: {result['delta_v']:.3f}")
```

### 异宿转移 (Heteroclinic)

连接不同轨道（通常 L1↔L2）的流形：

```python
# L1 Halo → L2 Halo 异宿转移
result = transfer.design_heteroclinic_transfer(
    orbit_L1=halo_L1,
    orbit_L2=halo_L2,
    plane='xy',  # 庞加莱截面平面
    verbose=True
)

# 打印结果
print(f"L1 出口状态: {result['departure_state']}")
print(f"L2 入口状态: {result['arrival_state']}")
print(f"转移时间: {result['transfer_time']:.2f} 无量纲时间")
```

### 流形交叉搜索

```python
# 搜索两条轨道流形的交叉点
result = transfer.design_manifold_intersection(
    orbit_departure=orbit1,
    orbit_arrival=orbit2,
    plane='y',
    plane_value=0.0,
    n_samples=100
)

print(f"交叉点距离: {result['closest_distance']:.6f}")
print(f"对应转移弧: {result['transfer_arcs']}")
```

---

## 返回地球

### 直接返回

```python
from e2m2e.transfer.moon_earth import MoonEarthTransfer

return_transfer = MoonEarthTransfer(system, dynamics)

# 设计直接返回
result = return_transfer.design_direct_return(
    departure_orbit=dro,
    r_reentry=0.017,  # LEO 轨道
    n_search_points=20
)

print(f"返回 ΔV: {result['delta_v']:.2f} km/s")
```

### 低能返回

```python
# 设计低能返回（经 L2）
result = return_transfer.design_low_energy_return(
    departure_orbit=dro,
    libration_point='L2',
    approach_direction='forward'
)
```

---

## ΔV 计算

### 基本计算

```python
# 两状态间的 ΔV
delta_v = transfer.compute_delta_v(
    state1=state_departure,
    state2=state_arrival
)
print(f"ΔV = {delta_v:.3f} km/s")
```

### 总 ΔV 统计

```python
# 统计多脉冲转移的总 ΔV
total_dv = sum(pulse['delta_v'] for pulse in result['impulses'])
print(f"总 ΔV: {total_dv:.2f} km/s")

# 各脉冲分解
for i, impulse in enumerate(result['impulses']):
    print(f"脉冲 {i+1}: ΔV = {impulse['delta_v']:.3f} km/s")
```

---

## 庞加莱截面分析

### 截面绘制

```python
from e2m2e.visualization.plotting import OrbitVisualizer

viz = OrbitVisualizer(system)

# 绘制流形截面
viz.plot_poincare_section(
    orbits=[manifold_unstable, manifold_stable],
    plane='y',
    value=0.0,
    ax=None
)

viz.show()
```

### 交叉点检测

```python
# 在截面 y=0 上检测交叉
crossings_unstable = detect_crossings(manifold_unstable, plane='y', value=0.0)
crossings_stable = detect_crossings(manifold_stable, plane='y', value=0.0)

# 寻找最近配对
closest_pair = find_closest_crossings(
    crossings_unstable,
    crossings_stable
)
print(f"最小距离: {closest_pair['distance']:.6f}")
```

---

## 完整示例：L1-L2 异宿转移设计

```python
import numpy as np
from e2m2e.core.system import CR3BP_System
from e2m2e.core.dynamics import CR3BP_Dynamics
from e2m2e.algorithms.differential_correction import DifferentialCorrection
from e2m2e.algorithms.continuation import Continuation
from e2m2e.transfer.inter_orbit import InterOrbitTransfer

# 1. 初始化
system = CR3BP_System.from_known_system("earth_moon")
system.compute_libration_points()
dynamics = CR3BP_Dynamics(system)

# 2. 生成 L1 和 L2 Halo 轨道族
dc = DifferentialCorrection(dynamics)
continuation = Continuation(dc, step=0.005)

# 3. 设计 L1 Halo
dc.setup_3D_symmetric_x_fixed_x0(x0=system.L1[0] + 0.01)
orbit_L1, _ = dc.iterate_correction(l1_guess, t_half=1.6)

# 4. 设计 L2 Halo
dc.setup_3D_symmetric_x_fixed_x0(x0=system.L2[0] - 0.01)
orbit_L2, _ = dc.iterate_correction(l2_guess, t_half=1.6)

# 5. 设计异宿转移
transfer = InterOrbitTransfer(system, dynamics)
result = transfer.design_heteroclinic_transfer(orbit_L1, orbit_L2)

print(f"转移时间: {result['transfer_time']:.2f} 天")
print(f"总 ΔV: {result['delta_v']:.3f} km/s")
```

---

## 常见问题

### 1. 流形不交叉怎么办？

**可能原因**：
- 轨道选取不当
- 截面位置不合适

**解决方案**：
1. 尝试不同截面平面
2. 增加流形采样点
3. 调整轨道振幅

### 2. ΔV 为负值？

**含义**：流形自然演化导致的能量变化

**处理**：
- 检查状态计算
- 确认坐标系一致性

### 3. 转移时间过长？

**优化策略**：
1. 选择更近的出发/到达轨道
2. 使用多脉冲而非单脉冲
3. 考虑中途机动

---

## 参考

- 详见 [技术文档 - 转移模块](e2m2e_technical_documentation.md#3-transfer-module-转移模块)
- 详见 [可视化指南 - 庞加莱截面](visualization_guide.md#庞加莱截面)
