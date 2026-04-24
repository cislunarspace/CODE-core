# 设计文档：Sphinx/RST 迁移 + SRP 动力学子类

**日期**: 2026-04-24
**状态**: 已批准
**作者**: Claude Code

## 概述

本文档描述两个独立但相关的任务：
1. 将 e2m2e 文档系统从 Docusaurus 迁移到 Sphinx/RST
2. 从 EXOSIMS 导入太阳辐射压（SRP）实现，创建 CR3BP_SRP_Dynamics 子类

## 任务一：Sphinx/RST 迁移

### 目标

- 完全替换 Docusaurus 为 Sphinx/RST
- 保留所有现有 30+ 篇文档内容
- 添加 API 自动生成（从 docstring 提取）
- 支持中文和英文双语

### 架构设计

```
e2m2e/
├── docs/                          # Sphinx 文档根目录
│   ├── conf.py                    # Sphinx 配置
│   ├── index.rst                  # 文档首页
│   ├── Makefile                   # 构建脚本
│   │
│   ├── getting-started/           # 快速入门
│   │   ├── installation.rst
│   │   ├── quickstart.rst
│   │   └── visualization.rst
│   │
│   ├── core/                      # 核心概念
│   │   ├── system.rst
│   │   ├── dynamics.rst
│   │   ├── orbit.rst
│   │   └── coordinate.rst
│   │
│   ├── algorithms/                # 算法详解
│   │   ├── differential-correction.rst
│   │   ├── continuation.rst
│   │   ├── halo.rst
│   │   ├── stability.rst
│   │   └── multiple-shooting.rst
│   │
│   ├── transfer/                  # 转移轨道设计
│   │   ├── overview.rst
│   │   ├── search.rst
│   │   └── optimization.rst
│   │
│   ├── api/                       # API 自动生成
│   │   ├── e2m2e.rst
│   │   ├── e2m2e.core.rst
│   │   ├── e2m2e.algorithms.rst
│   │   ├── e2m2e.transfer.rst
│   │   ├── e2m2e.mbse.rst
│   │   └── e2m2e.visualization.rst
│   │
│   ├── reference/                 # 参考资料
│   │   ├── api-reference.rst
│   │   ├── algorithm-reference.rst
│   │   └── glossary.rst
│   │
│   └── _static/                   # 静态资源
│       └── images/
```

### Sphinx 配置

```python
# conf.py 核心配置
extensions = [
    "sphinx.ext.autodoc",        # 从 docstring 生成 API
    "sphinx.ext.napoleon",       # 支持 Google/NumPy docstring
    "sphinx.ext.mathjax",        # 数学公式渲染
    "sphinx.ext.viewcode",       # 源码链接
    "sphinx.ext.intersphinx",    # 跨项目引用
    "sphinxcontrib.mermaid",     # Mermaid 图表
    "sphinx_copybutton",         # 代码块复制按钮
]

# 中文支持
language = "zh_CN"
locale_dirs = ["locale/"]

# 主题
html_theme = "sphinx_rtd_theme"  # Read the Docs 主题
```

### 迁移步骤

1. 创建 `docs/` 目录结构
2. 配置 `conf.py`（参考 EXOSIMS 配置）
3. 转换 Markdown → RST（使用 pandoc 或手动）
4. 创建 API 文档模板（`automodule` 指令）
5. 更新 CI/CD（替换 Docusaurus 构建为 Sphinx）
6. 更新 `.github/workflows/deploy-docs.yml`

### 依赖变更

```toml
# pyproject.toml [project.optional-dependencies]
docs = [
    "sphinx>=7.0",
    "sphinx-rtd-theme>=2.0",
    "sphinxcontrib-mermaid>=0.9",
    "sphinx-copybutton>=0.5",
]
```

## 任务二：SRP 动力学子类

### 目标

- 创建 `CR3BP_SRP_Dynamics` 子类
- 继承 `CR3BP_Dynamics` 的所有功能
- 添加太阳辐射压扰动力
- 从 EXOSIMS 的 `equationsOfMotion_CRTBP` 导入 SRP 实现

### 类架构

```
Dynamics (base)
    └── CR3BP_Dynamics
            └── CR3BP_SRP_Dynamics (新增)
```

### 核心实现

```python
# e2m2e/core/srp_dynamics.py

class CR3BP_SRP_Dynamics(CR3BP_Dynamics):
    """CR3BP with Solar Radiation Pressure perturbation.
    
    Adds SRP force to the standard CR3BP equations of motion.
    Based on EXOSIMS implementation with optical coefficients.
    """
    
    def __init__(
        self,
        system: CR3BP_System,
        # SRP 参数
        area: float = 1.0,                    # 航天器截面积 (m²)
        mass: float = 1000.0,                  # 航天器质量 (kg)
        Cr: float = 1.5,                       # 反射系数 (1=完全吸收, 2=完全反射)
        # 光学系数（来自 EXOSIMS）
        non_lambertian_front: float = 0.038,
        non_lambertian_back: float = 0.004,
        specular_reflection: float = 0.975,
        nreflection_coeff: float = 0.999,
        emission_front: float = 0.8,
        emission_back: float = 0.2,
        # 太阳辐射压常数
        P_srp: float = 4.56e-6,               # 太阳辐射压 (N/m²) at 1 AU
        **kwargs
    ):
        super().__init__(system, **kwargs)
        self.area = area
        self.mass = mass
        self.Cr = Cr
        
        # 计算光学系数 (来自 EXOSIMS)
        self.b1 = 0.5 * (1.0 - specular_reflection * nreflection_coeff)
        self.b2 = specular_reflection * nreflection_coeff
        self.b3 = 0.5 * (
            non_lambertian_front * (1.0 - specular_reflection) * nreflection_coeff
            + (1.0 - nreflection_coeff) * (
                emission_front * non_lambertian_front 
                - emission_back * non_lambertian_back
            ) / (emission_front + emission_back)
        )
        
        # SRP 加速度系数
        self.beta = self._compute_beta()
    
    def _compute_beta(self) -> float:
        """Compute SRP acceleration coefficient (solar sail parameter)."""
        # β = P_srp * A * Cr / (2 * m)
        return self.P_srp * self.area * self.Cr / (2.0 * self.mass)
    
    def _get_eom_func(self):
        """Override to add SRP perturbation to CR3BP EOM."""
        base_eom = super()._get_eom_func()
        
        def eom_with_srp(t, state):
            # 基础 CR3BP 方程
            ds = base_eom(t, state)
            
            # SRP 扰动加速度
            x, y, z = state[:3]
            mu = self.system.mu
            
            # 航天器到主天体（太阳/地球）的距离
            r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
            
            # 径向单位矢量（从主天体指向航天器）
            ur = np.array([x + mu, y, z]) / r1
            
            # 切向单位矢量（在旋转平面内垂直于径向）
            ut = np.array([-y, x + mu, 0]) / np.sqrt((x + mu)**2 + y**2)
            
            # 径向和切向 SRP 力分量（来自 EXOSIMS）
            F_radial = self.b1 + 0.25 * self.b2 + 0.5 * self.b3
            F_tangential = (np.sqrt(3) * 0.25) * (self.b2 + 2.0 * self.b3)
            
            # SRP 扰动加速度
            a_srp = self.beta * (F_radial * ur + F_tangential * ut)
            
            # 添加到速度导数（加速度分量）
            ds[3] += a_srp[0]  # ẍ
            ds[4] += a_srp[1]  # ÿ
            ds[5] += a_srp[2]  # z̈
            
            return ds
        
        return eom_with_srp
```

### 关键特性

| 特性 | 说明 |
|------|------|
| **继承** | 完全继承 `CR3BP_Dynamics` 的所有功能（STM、传播器等） |
| **参数化** | 支持光学系数、截面积、质量等 SRP 参数 |
| **无量纲化** | 自动处理物理单位到 CR3BP 无量纲单位的转换 |
| **可选启用** | 通过设置 `area=0` 或 `Cr=0` 可禁用 SRP |
| **STM 兼容** | 扰动不影响 STM 计算（数值微分自动适应） |

### 使用示例

```python
from e2m2e.core.system import CR3BP_System
from e2m2e.core.srp_dynamics import CR3BP_SRP_Dynamics

# 创建地月系统
system = CR3BP_System("earth-moon")

# 创建带 SRP 的动力学
dynamics = CR3BP_SRP_Dynamics(
    system,
    area=100.0,      # 100 m² 截面积
    mass=1000.0,     # 1000 kg 质量
    Cr=1.5           # 反射系数
)

# 正常使用（与 CR3BP_Dynamics 接口一致）
orbit = dynamics.propagate(initial_state, t_span)
```

### 测试计划

1. **单元测试**：验证 SRP 力计算正确性
2. **对比测试**：与 EXOSIMS 结果对比
3. **回归测试**：确保 `area=0` 时与纯 CR3BP 一致
4. **集成测试**：在转移轨道设计中使用 SRP 动力学

## 依赖关系

两个任务相互独立，可并行实现：
- 任务一（Sphinx 迁移）不影响代码逻辑
- 任务二（SRP 子类）不影响文档系统

## 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| Markdown → RST 转换丢失格式 | 使用 pandoc 预转换，手动校对关键文档 |
| SRP 单位转换错误 | 与 EXOSIMS 结果对比验证 |
| 现有测试失败 | 迁移前确保所有测试通过 |

## 成功标准

- [ ] Sphinx 文档成功构建并部署
- [ ] 所有现有文档内容保留
- [ ] API 文档自动生成正常
- [ ] CR3BP_SRP_Dynamics 通过所有测试
- [ ] SRP 结果与 EXOSIMS 一致（误差 < 1%）
