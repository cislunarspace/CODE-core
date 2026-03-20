# E2M2E 文档

> Earth-to-Moon-to-Earth (E2M2E) 轨道力学库的完整文档

## 文档结构

```
docs/
├── index.md              # 本文档
├── guides/               # 用户指南
│   ├── system-overview.md
│   ├── orbit-generation.md
│   ├── transfer-design.md
│   └── visualization-guide.md
└── reference/            # 技术参考
    ├── api-reference.md
    └── algorithms.md
```

### 用户指南

| 文档 | 说明 |
|------|------|
| [系统总览](guides/system-overview.md) | 系统架构、模块关系、设计理念 |
| [快速开始](reference/api-reference.md#快速开始) | 5分钟上手指南 |
| [轨道生成](guides/orbit-generation.md) | DRO、Halo、Lyapunov等轨道生成教程 |
| [转移轨道设计](guides/transfer-design.md) | 轨道间转移、低能转移设计指南 |

### 技术参考

| 文档 | 说明 |
|------|------|
| [API 参考](reference/api-reference.md) | 完整API参考、数学基础、设计原理 |
| [CR3BP算法](reference/algorithms.md) | 微分修正、延拓、稳定性分析算法详解 |
| [可视化指南](guides/visualization-guide.md) | 绘图功能、使用示例、自定义设置 |

### 资源

- [API 参考](../e2m2e/) - 代码中的 docstring
- [示例代码](../examples/) - 实际使用示例
- [测试用例](../tests/) - 单元测试，覆盖核心功能

## 快速链接

### 核心概念

- [CR3BP_System](../e2m2e/core/system.py) - 系统参数与平动点
- [CR3BP_Dynamics](../e2m2e/core/dynamics.py) - 动力学方程与数值积分
- [Orbit](../e2m2e/core/orbit.py) - 轨道数据与周期检测
- [DifferentialCorrection](../e2m2e/algorithms/differential_correction.py) - 周期轨道求解

### 常用任务

1. **设计DRO轨道** → 参考 [轨道生成 - DRO](guides/orbit-generation.md#distant-retrograde-orbit-dro)
2. **设计Halo轨道** → 参考 [轨道生成 - Halo](guides/orbit-generation.md#halo轨道)
3. **生成轨道族** → 参考 [轨道族延拓](reference/algorithms.md#5-轨道族延拓算法)
4. **分析稳定性** → 参考 [稳定性分析](reference/algorithms.md#7-稳定性分析)

## 物理背景

E2M2E 基于**圆型限制性三体问题 (CR3BP)** 实现轨道设计。地月系统中：
- 质量参数 $\mu \approx 0.01215$
- 特征距离：384,400 km（地月距离）
- 特征周期：27.32 天

详见 [CR3BP理论](reference/algorithms.md#1-概述) 和 [系统概述](guides/system-overview.md)。
