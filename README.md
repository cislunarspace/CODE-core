# e2m2e — Earth to Moon, Moon to Earth

**地月空间转移轨道设计库**

`e2m2e` 是一个基于圆型限制性三体问题（CR3BP）的Python库，专注于设计和分析地月空间转移轨道。

## 主要功能

### 地球到月球转移
- 直接转移轨道设计
- 低能转移轨道（经平动点L1/L2）
- 基于不变流形的转移

### 月球到地球转移
- 直接返回轨道
- 低能返回路径
- 流形辅助返回

### 轨道间转移
- 同族轨道转移（如不同Halo轨道之间）
- 异族轨道转移（如Lyapunov → Halo）
- L1-L2异宿连接
- 同宿轨道转移

### 三体轨道设计
- 平动点轨道（Halo、Lyapunov、Vertical等）
- 微分修正算法（多种对称性配置）
- 轨道族延拓（自然参数/伪弧长）
- 稳定性分析与分岔检测

### 可视化
- 3D轨道绘制
- 2D投影图
- 轨道族演化图
- 庞加莱截面图
- 稳定性图

## 安装

```bash
cd e2m2e
pip install -e .
```

## 快速开始

```python
import e2m2e

# 1. 创建地月系统，为后续计算提供常数接口、数据存储接口等功能
system = e2m2e.core.system.CR3BP_System.from_known_system("earth_moon") # 也可以自定义系统参数
# system = e2m2e.core.system.CR3BP_System(mu=0.0121505856, primary="Earth", secondary="Moon") # 直接指定质量比
system.compute_libration_points() # 计算平动点位置（根据系统初始化的mu来计算）
system.set_characteristic_scales(distance=384400, period=27.32 * 86400)

print(f"地月系统: {system}")
print(f"L1点: {system.L1}")
print(f"L2点: {system.L2}")

# 2. 创建动力学对象
dynamics = e2m2e.core.dynamics.CR3BP_Dynamics(system)

# 3. 设计Lyapunov轨道（微分修正）
dc = e2m2e.algorithms.differential_correction.DifferentialCorrection(dynamics)
dc.setup_2D_symmetric_x_fixed_x0(x0=system.L1[0] + 0.01)

# 初始猜测
initial_state = [system.L1[0] + 0.01, 0, 0, 0, 0.1, 0]
orbit, result = dc.correct_orbit(initial_state, t_half=1.5)

if orbit is not None:
    print(f"Lyapunov轨道周期: {orbit.period:.4f}")

    # 4. 可视化
    viz = e2m2e.visualization.plotting.OrbitVisualizer(system)
    viz.create_overview_plot(orbit)
    viz.show()

# 5. 轨道族延拓
cont = e2m2e.algorithms.continuation.Continuation(dc, param="x0", step=0.001)
family = cont.natural_continuation(
    seed_state=result['state'],
    seed_t_half=result['t_half'],
    n_orbits=20
)

# 6. 转移轨道设计
transfer = e2m2e.transfer.earth_moon.EarthMoonTransfer(system, dynamics) 
# ... 设计具体的转移轨道
```

## 库结构

```
e2m2e/
├── __init__.py              # 主入口
├── core/                    # 核心模块
│   ├── system.py            # CR3BP系统定义
│   ├── dynamics.py          # 动力学方程
│   ├── orbit.py             # 轨道数据结构
│   └── coordinate.py        # 坐标变换
├── algorithms/              # 算法模块
│   ├── differential_correction.py  # 微分修正
│   ├── continuation.py      # 轨道族延拓
│   └── stability.py         # 稳定性分析
├── transfer/                # 转移轨道模块
│   ├── earth_moon.py        # 地球→月球
│   ├── moon_earth.py        # 月球→地球
│   └── inter_orbit.py       # 轨道间转移
└── visualization/           # 可视化模块
    └── plotting.py          # 绘图工具
```

## 理论背景

本库基于**圆型限制性三体问题（CR3BP）**框架，在旋转坐标系中建立运动方程：

$$\ddot{x} - 2\dot{y} = \frac{\partial \Omega}{\partial x}$$

$$\ddot{y} + 2\dot{x} = \frac{\partial \Omega}{\partial y}$$

$$\ddot{z} = \frac{\partial \Omega}{\partial z}$$

其中等效势能为：

$$\Omega = \frac{1}{2}(x^2+y^2) + \frac{1-\mu}{r_1} + \frac{\mu}{r_2}$$

## 依赖

- Python ≥ 3.10
- NumPy ≥ 1.24
- SciPy ≥ 1.10
- Matplotlib ≥ 3.7

## 许可证

MIT License