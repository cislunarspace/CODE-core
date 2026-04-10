# 系统总览

> E2M2E 的架构设计、模块职责和扩展指南。

## 四层架构

```
core/           基础层 — 数据结构和物理模型
  ↓
algorithms/     算法层 — 微分修正、延拓、稳定性、多重打靶
  ↓
transfer/       设计层 — 网格搜索、NLP 优化
  ↓
visualization/  展示层 — 轨道族绘图、转移轨迹可视化
```

层与层之间严格单向依赖：上层可以使用下层的功能，下层不能引用上层。

## 模块职责

### Core（核心模块）

| 文件 | 类 | 做什么 |
|------|-----|--------|
| `system.py` | `CR3BP_System` | 系统参数、平动点、Jacobi 常数、单位转换 |
| `dynamics.py` | `CR3BP_Dynamics` | 运动方程、数值积分、STM 传播 |
| `orbit.py` | `Orbit`, `OrbitFamily` | 轨道数据容器、周期检测、JSON 序列化 |
| `coordinate.py` | `CoordinateTransformation` | 旋转系 ↔ 惯性系坐标变换 |
| `spice.py` | `SPICEManager` | SPICE 内核管理 |
| `ephemeris_system.py` | `EphemerisSystem` | 多天体星历系统 |
| `ephemeris_dynamics.py` | `EphemerisDynamics` | 基于 SPICE 的 N 体动力学 |

### Algorithms（算法模块）

| 文件 | 类 | 做什么 |
|------|-----|--------|
| `differential_correction.py` | `DifferentialCorrection` | Newton-Raphson 迭代求解周期轨道 |
| `continuation.py` | `Continuation` | 自然/伪弧长轨道族延拓 |
| `stability.py` | `StabilityAnalysis` | Floquet 乘子、分岔检测 |
| `multiple_shooting.py` | `MultipleShooting` | 多重打靶法，复杂约束修正 |

### Transfer（转移模块）

| 文件 | 类 | 做什么 |
|------|-----|--------|
| `transfer.py` | `Transfer` | 链式 API：`set_orbit().optimize()` |
| `transfer_search.py` | `DROTransferSearch` | DRO→RO 平面转移网格搜索（并行） |
| `transfer_optimization.py` | `DROTRONLPOptimizer` | NLP 优化（可选 COPT 求解器） |

### Visualization（可视化模块）

| 文件 | 类 | 做什么 |
|------|-----|--------|
| `config.py` | `PlotConfig` | 样式配置（字体、颜色、尺寸） |
| `base.py` | `OrbitVisualizer` | 2D/3D 轨道绘制基类 |
| `family.py` | `FamilyPlotter` | 轨道族可视化（Jacobi 着色） |
| `transfer.py` | `TransferPlotter` | 转移轨迹可视化 |

## 数据流

```
CR3BP_System → CR3BP_Dynamics → Orbit/OrbitFamily
                    ↓                    ↓
          DifferentialCorrection →  Transfer Design
                    ↓
              Continuation → Visualization
```

## 关键约定

- **状态向量顺序**始终为 `[x, y, z, vx, vy, vz]`，全局一致
- **数值精度**：积分器 `rtol=atol=1e-12`，有限差分步长不可增大
- **无量纲单位**：DU（距离）、TU（时间）、VU（速度）；物理计算前必须调用 `set_characteristic_scales()`
- **接口稳定性**：公共方法签名不可破坏向后兼容性；新参数必须有默认值

## 扩展指南

### 添加新的轨道类型

在 `differential_correction.py` 中添加对应的 `setup_*` 方法和对称性配置。

### 添加新的动力学模型

创建 `Dynamics` 的子类，实现 `equations_of_motion()` 和 `propagate()`。不要修改基类。

### 添加新的算法

在 `algorithms/` 目录创建新模块，遵循现有接口设计，在 `__init__.py` 中导出。

→ 各模块的详细用法见对应文档页面。
