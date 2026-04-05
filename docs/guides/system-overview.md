# 系统总览

## 架构设计

E2M2E 采用模块化设计，包含四个核心模块：

```
┌─────────────────────────────────────────────────────────────┐
│                         e2m2e                               │
├─────────────┬─────────────┬─────────────┬─────────────────┤
│    core     │ algorithms  │  transfer   │  visualization  │
├─────────────┼─────────────┼─────────────┼─────────────────┤
│ system.py   │ differential │ earth_moon  │   config.py     │
│ dynamics.py │ correction.py│ moon_earth  │   base.py       │
│ orbit.py    │continuation.py│inter_orbit │   family.py     │
│coordinate.py│ stability.py │             │   transfer.py   │
│             │             │             │   stability.py   │
│             │             │             │   plotting.py    │
└─────────────┴─────────────┴─────────────┴─────────────────┘
```

## 模块职责

### Core（核心模块）

提供 CR3BP 轨道力学的基础构建块：

| 文件 | 类/函数 | 职责 |
|------|---------|------|
| `system.py` | `CR3BP_System` | 系统参数、平动点计算、坐标转换 |
| `dynamics.py` | `CR3BP_Dynamics` | 运动方程、数值积分、STM传播 |
| `orbit.py` | `Orbit`, `OrbitFamily` | 轨道数据管理、周期检测、稳定性分析 |
| `coordinate.py` | `CoordinateTransformation` | 坐标系变换（旋转系↔惯性系） |

### Algorithms（算法模块）

实现周期轨道设计所需的数值算法：

| 文件 | 类/函数 | 职责 |
|------|---------|------|
| `differential_correction.py` | `DifferentialCorrection` | 牛顿迭代求解周期轨道 |
| `continuation.py` | `Continuation` | 轨道族延拓（自然/伪弧长） |
| `stability.py` | `StabilityAnalysis` | Floquet乘子、稳定性判定、分岔检测 |

### Transfer（转移模块）

基于核心模块实现轨道转移设计：

| 文件 | 类 | 职责 |
|------|---|------|
| `transfer.py` | `Transfer` | 简化链式 API |
| `transfer_search.py` | `TransferSearch` | DRO→RO 平面转移网格搜索（并行） |
| `transfer_optimization.py` | `DROTRONLPOptimizer` | NLP 优化 |

### Visualization（可视化模块）

| 文件 | 类/函数 | 职责 |
|------|---------|------|
| `config.py` | `PlotConfig` | 可视化配置（颜色、标签、样式） |
| `base.py` | `OrbitVisualizer`, `ProjectionPlane` | 2D/3D轨道绘制、庞加莱截面、概览图 |
| `family.py` | `FamilyPlotter` | 轨道族可视化 |
| `transfer.py` | `TransferPlotter` | 转移轨道可视化 |
| `stability.py` | `compute_stability_for_family` | 轨道族稳定性计算 |
| `plotting.py` | *(re-export shim)* | 向后兼容重导出 |

## 数据流

```
┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│ CR3BP_System │────▶│ CR3BP_Dynamics│────▶│    Orbit     │
└──────────────┘     └───────────────┘     └──────────────┘
                            │                     │
                            ▼                     ▼
                     ┌───────────────┐     ┌──────────────┐
                     │Differential   │     │   Orbit      │
                     │Correction     │────▶│   Family     │
                     └───────────────┘     └──────────────┘
                            │
                            ▼
                     ┌───────────────┐     ┌──────────────┐
                     │ Continuation  │────▶│  Transfer    │
                     └───────────────┘     │  Design      │
                                            └──────────────┘
```

## 典型工作流

### 1. 周期轨道设计

```python
# 1. 创建系统
system = CR3BP_System.from_known_system("earth_moon")
system.compute_libration_points()

# 2. 创建动力学模型
dynamics = CR3BP_Dynamics(system)

# 3. 配置微分修正器
dc = DifferentialCorrection(dynamics)
dc.setup_2D_symmetric_x_fixed_x0(x0=0.8)

# 4. 迭代求解
orbit, result = dc.iterate_correction(initial_state, t_half=1.5)
```

### 2. 轨道族延拓

```python
# 5. 延拓轨道族
continuation = Continuation(dc, step=0.01)
family = continuation.natural_continuation(
    seed_orbit=orbit,
    param_range=(0.8, 1.2),
    step_size=0.01
)
```

### 3. 转移设计

```python
# 6. 设计转移轨道
transfer = InterOrbitTransfer(system, dynamics)
result = transfer.design_heteroclinic_transfer(orbit_L1, orbit_L2)
```

## 设计原则

1. **物理驱动**：所有算法基于CR3BP严格数学模型
2. **模块化**：各模块独立可测试，接口清晰
3. **数值稳健**：使用高阶积分器、自适应步长、收敛检测
4. **用户友好**：丰富的中文注释和错误提示

## 扩展指南

### 添加新轨道类型

1. 在 `differential_correction.py` 中添加新的对称性配置
2. 实现对应的 `setup_*` 方法
3. 添加测试用例

### 添加新转移策略

1. 在 `transfer/` 目录创建新模块
2. 继承基础类实现具体策略
3. 在 `__init__.py` 中导出

详见项目根目录的 CONTRIBUTING.md 文件。
