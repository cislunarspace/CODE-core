---
title: 轨道族延拓：Continuation
---

# 轨道族延拓：Continuation

> **文件**: `e2m2e/algorithms/continuation.py`

延拓从一条已收敛的周期轨道（种子）出发，沿参数方向逐步追踪，生成一族轨道。这是系统化探索轨道空间的核心工具。

## 怎么生成一族周期轨道

```python
from e2m2e.core import CR3BP_System, CR3BP_Dynamics
from e2m2e.algorithms import DifferentialCorrection, Continuation

# 1. 先得到一条种子轨道
system = CR3BP_System.from_known_system("earth_moon")
system.set_characteristic_scales(distance=384400, period=27.32*86400)
system.compute_libration_points()
dynamics = CR3BP_Dynamics(system)

dc = DifferentialCorrection(dynamic=dynamics)
dc.setup_2D_symmetric_x_fixed_x0(x0=0.8)
seed_orbit, _ = dc.iterate_correction(
    np.array([0.8, 0.0, 0.0, 0.0, 0.5, 0.0]), t_half=1.6
)

# 2. 沿参数方向延拓
cont = Continuation(corrector=dc, step=0.01)
family = cont.natural_continuation(
    seed_orbit=seed_orbit,
    param_range=(0.8, 0.95),  # x0 的范围
    step_size=0.01,
    verbose=True,
)

# 3. 保存
family.save_to_file("output/dro_family.json")
print(f"生成了 {len(family)} 条轨道")
```

## 自然延拓 vs 伪弧长延拓？

| | 自然延拓 | 伪弧长延拓 |
|---|---------|-----------|
| **原理** | 固定一个参数，逐步变化 | 在参数空间中沿曲线切线方向追踪 |
| **优点** | 简单，参数少 | 能绕过转向点（turning point） |
| **缺点** | 在转向点处失效 | 参数更多，调参更复杂 |
| **适用** | 参数单调变化的族 | 有拐点/折叠的族（如 Halo 族） |

**建议**：先试自然延拓。如果在某个参数值处失败（报错或轨道跳变），换伪弧长延拓。

### 伪弧长延拓

```python
family = cont.pseudo_arclength_continuation(
    seed_orbit=seed_orbit,
    n_orbits=50,
    direction="positive",  # 或 "negative"
    verbose=True,
)
```

关键参数：
- `n_orbits`：生成的轨道数量
- `direction`：延拓方向（"positive" 或 "negative"）
- 初始步长通过 `Continuation(step=...)` 设定，算法会自适应调整

## 怎么生成 Halo 轨道族

Halo 轨道族通常使用伪弧长延拓，因为族曲线包含转向点：

```python
from e2m2e.algorithms import Continuation, DifferentialCorrection

dc = DifferentialCorrection(dynamic=dynamics)
cont = Continuation(corrector=dc)

# 生成种子轨道
seed = cont.generate_halo_seed_orbit(
    libration_point=1,   # L1
    amplitude_z=0.23,    # z 方向振幅
    halo_class=0,        # Northern halo
)

# 伪弧长延拓生成族
family = cont.halo_pseudo_arclength_continuation(
    seed_orbit=seed,
    n_orbits=10,
    direction="both",    # 双向延拓
    step_size=0.0045,
    verbose=True,
)
```

→ 详见 [Halo 轨道](halo.md)（包含 Richardson 初值、PAL 细节、命令行脚本）

## API 速查

| 方法 | 说明 |
|------|------|
| `natural_continuation(seed_orbit, param_range, step_size)` | 自然参数延拓 |
| `pseudo_arclength_continuation(seed_orbit, n_orbits, direction)` | 伪弧长延拓 |
| `generate_halo_seed_orbit(libration_point, amplitude_z, halo_class)` | 生成 Halo 种子轨道 |
| `generate_halo_family(seed_orbit, ...)` | 按振幅步进的 Halo 族 |
| `halo_pseudo_arclength_continuation(seed_orbit, n_orbits, direction)` | Halo 专用伪弧长延拓 |

完整 API 文档见 [API 参考](../reference/api-reference.md)。
