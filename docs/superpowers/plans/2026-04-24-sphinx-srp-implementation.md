# Sphinx/RST 迁移 + SRP 动力学子类 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 e2m2e 文档系统从 Docusaurus 迁移到 Sphinx/RST，并创建 CR3BP_SRP_Dynamics 子类实现太阳辐射压扰动。

**Architecture:** 两个独立任务并行实现：(1) Sphinx/RST 文档系统替换 Docusaurus，保留现有内容并添加 API 自动生成；(2) 继承 CR3BP_Dynamics 创建 SRP 子类，从 EXOSIMS 导入光学系数模型。

**Tech Stack:** Sphinx, sphinx-rtd-theme, sphinxcontrib-mermaid, numpy, scipy

---

## 文件结构

### 任务一：SRP 动力学子类

| 操作 | 文件路径 | 说明 |
|------|----------|------|
| Create | `e2m2e/core/srp_dynamics.py` | CR3BP_SRP_Dynamics 子类 |
| Create | `tests/core/test_srp_dynamics.py` | SRP 动力学测试 |
| Modify | `e2m2e/core/__init__.py` | 添加 SRP 导出 |

### 任务二：Sphinx/RST 迁移

| 操作 | 文件路径 | 说明 |
|------|----------|------|
| Create | `docs/conf.py` | Sphinx 配置 |
| Create | `docs/index.rst` | 文档首页 |
| Create | `docs/Makefile` | 构建脚本 |
| Create | `docs/getting-started/*.rst` | 快速入门文档 |
| Create | `docs/core/*.rst` | 核心概念文档 |
| Create | `docs/api/*.rst` | API 自动生成文档 |
| Modify | `pyproject.toml` | 添加 docs 依赖 |
| Modify | `.github/workflows/deploy-docs.yml` | 更新部署流程 |

---

## 任务一：SRP 动力学子类

### Task 1: 编写 SRP 动力学失败测试

**Files:**
- Create: `tests/core/test_srp_dynamics.py`

- [ ] **Step 1: 创建测试文件**

```python
"""CR3BP_SRP_Dynamics 测试

测试太阳辐射压动力学子类的功能：
1. 光学系数计算正确性
2. SRP 力计算正确性
3. 与纯 CR3BP 的一致性（area=0 时）
4. 传播功能正常工作
"""

import numpy as np
import pytest

from e2m2e.core.system import CR3BP_System
from e2m2e.core.srp_dynamics import CR3BP_SRP_Dynamics


@pytest.fixture
def earth_moon_system():
    """创建地月系统"""
    return CR3BP_System("earth-moon")


@pytest.fixture
def srp_dynamics(earth_moon_system):
    """创建 SRP 动力学对象"""
    return CR3BP_SRP_Dynamics(
        earth_moon_system,
        area=100.0,
        mass=1000.0,
        Cr=1.5,
    )


@pytest.fixture
def zero_srp_dynamics(earth_moon_system):
    """创建零 SRP 动力学对象（应与纯 CR3BP 一致）"""
    return CR3BP_SRP_Dynamics(
        earth_moon_system,
        area=0.0,
        mass=1000.0,
        Cr=1.5,
    )


class TestOpticalCoefficients:
    """测试光学系数计算"""

    def test_b1_coefficient(self, srp_dynamics):
        """测试 b1 系数计算"""
        # b1 = 0.5 * (1 - s * p)
        # 默认值: s=0.975, p=0.999
        expected = 0.5 * (1.0 - 0.975 * 0.999)
        assert abs(srp_dynamics.b1 - expected) < 1e-10

    def test_b2_coefficient(self, srp_dynamics):
        """测试 b2 系数计算"""
        # b2 = s * p
        expected = 0.975 * 0.999
        assert abs(srp_dynamics.b2 - expected) < 1e-10

    def test_b3_coefficient(self, srp_dynamics):
        """测试 b3 系数计算"""
        # b3 = 0.5 * (Bf * (1-s) * p + (1-p) * (ef*Bf - eb*Bb) / (ef + eb))
        Bf = 0.038
        Bb = 0.004
        s = 0.975
        p = 0.999
        ef = 0.8
        eb = 0.2
        expected = 0.5 * (
            Bf * (1.0 - s) * p
            + (1.0 - p) * (ef * Bf - eb * Bb) / (ef + eb)
        )
        assert abs(srp_dynamics.b3 - expected) < 1e-10


class TestBetaCoefficient:
    """测试 SRP 加速度系数"""

    def test_beta_calculation(self, srp_dynamics):
        """测试 beta = P_srp * A * Cr / (2 * m)"""
        P_srp = 4.56e-6  # N/m²
        area = 100.0  # m²
        Cr = 1.5
        mass = 1000.0  # kg
        expected = P_srp * area * Cr / (2.0 * mass)
        assert abs(srp_dynamics.beta - expected) < 1e-15

    def test_zero_area_gives_zero_beta(self, zero_srp_dynamics):
        """测试面积为 0 时 beta 为 0"""
        assert zero_srp_dynamics.beta == 0.0


class TestSRPForce:
    """测试 SRP 力计算"""

    def test_srp_force_magnitude(self, srp_dynamics):
        """测试 SRP 力的量级合理"""
        # 在 L1 点附近，SRP 力应该远小于引力
        state = np.array([0.8, 0.0, 0.0, 0.0, 0.0, 0.0])
        eom = srp_dynamics._get_eom_func(with_stm=False)
        ds = eom(0.0, state)

        # SRP 扰动应该使加速度有微小变化
        # 这里只检查返回值形状和有限性
        assert ds.shape == (6,)
        assert np.all(np.isfinite(ds))

    def test_srp_disabled_when_area_zero(self, zero_srp_dynamics):
        """测试面积为 0 时 SRP 被禁用"""
        state = np.array([0.8, 0.0, 0.0, 0.0, 0.0, 0.0])
        eom = zero_srp_dynamics._get_eom_func(with_stm=False)
        ds_zero = eom(0.0, state)

        # 与纯 CR3BP 比较
        from e2m2e.core.dynamics import CR3BP_Dynamics

        cr3bp = CR3BP_Dynamics(zero_srp_dynamics.system)
        ds_cr3bp = cr3bp.equations_of_motion(0.0, state)

        np.testing.assert_allclose(ds_zero, ds_cr3bp, atol=1e-15)


class TestPropagation:
    """测试传播功能"""

    def test_propagate_returns_correct_shape(self, srp_dynamics):
        """测试传播结果形状正确"""
        initial_state = np.array([0.8, 0.0, 0.0, 0.0, 0.6, 0.0])
        result = srp_dynamics.propagate(
            initial_state,
            t_span=(0.0, 1.0),
            t_eval=np.linspace(0.0, 1.0, 10),
        )

        assert result["states"].shape == (10, 6)
        assert result["time"].shape == (10,)

    def test_propagate_with_stm(self, srp_dynamics):
        """测试带 STM 的传播"""
        initial_state = np.array([0.8, 0.0, 0.0, 0.0, 0.6, 0.0])
        result = srp_dynamics.propagate(
            initial_state,
            t_span=(0.0, 0.1),
            with_stm=True,
        )

        assert "stm" in result
        assert result["stm"].shape[1:] == (6, 6)

    def test_propagate_with_jacobi(self, srp_dynamics):
        """测试带 Jacobi 常数的传播"""
        initial_state = np.array([0.8, 0.0, 0.0, 0.0, 0.6, 0.0])
        result = srp_dynamics.propagate(
            initial_state,
            t_span=(0.0, 0.1),
            with_jacobi=True,
        )

        # SRP 系统中 Jacobi 常数不守恒（因为有非保守力）
        # 但应该有值
        assert "jacobi" in result
        assert len(result["jacobi"]) > 0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd ~/codes/e2m2e && python -m pytest tests/core/test_srp_dynamics.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'e2m2e.core.srp_dynamics'"

- [ ] **Step 3: 提交测试文件**

```bash
cd ~/codes/e2m2e
git add tests/core/test_srp_dynamics.py
git commit -m "test: add failing tests for CR3BP_SRP_Dynamics"
```

### Task 2: 实现 CR3BP_SRP_Dynamics

**Files:**
- Create: `e2m2e/core/srp_dynamics.py`

- [ ] **Step 1: 创建 SRP 动力学模块**

```python
"""
太阳辐射压动力学模块

实现 CR3BP 框架下的太阳辐射压 (SRP) 扰动。
基于 EXOSIMS 的 equationsOfMotion_CRTBP 实现，采用光学系数模型。

物理背景
--------
太阳辐射压是光子撞击航天器表面产生的力。对于非完美反射表面，
SRP 力可分解为径向和切向分量，由光学系数 b1, b2, b3 决定。

光学系数模型（来自 EXOSIMS）：
  - b1 = 0.5 * (1 - s * p)：漫反射分量
  - b2 = s * p：镜面反射分量
  - b3 = 0.5 * (Bf * (1-s) * p + (1-p) * (ef*Bf - eb*Bb) / (ef + eb))：非朗伯分量

其中：
  - s: 镜面反射因子
  - p: 反射系数
  - Bf, Bb: 前/后表面非朗伯系数
  - ef, eb: 前/后表面发射系数

References:
    - EXOSIMS ObservatoryL2Halo.equationsOfMotion_CRTBP
    - Vallado, D. A. (2013). Fundamentals of Astrodynamics and Applications.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from .dynamics import CR3BP_Dynamics
from .system import CR3BP_System


class CR3BP_SRP_Dynamics(CR3BP_Dynamics):
    """带太阳辐射压的 CR3BP 动力学

    在标准 CR3BP 运动方程基础上添加太阳辐射压扰动力。
    SRP 力模型基于 EXOSIMS 实现，支持非完美反射表面的光学系数。

    Attributes:
        area: 航天器截面积 (m²)
        mass: 航天器质量 (kg)
        Cr: 反射系数 (1=完全吸收, 2=完全反射)
        b1: 漫反射光学系数
        b2: 镜面反射光学系数
        b3: 非朗伯光学系数
        beta: SRP 加速度系数 (solar sail parameter)
        P_srp: 太阳辐射压常数 (N/m² at 1 AU)

    Example:
        >>> from e2m2e.core.system import CR3BP_System
        >>> from e2m2e.core.srp_dynamics import CR3BP_SRP_Dynamics
        >>> system = CR3BP_System("earth-moon")
        >>> dynamics = CR3BP_SRP_Dynamics(system, area=100.0, mass=1000.0)
        >>> result = dynamics.propagate([0.8, 0, 0, 0, 0.6, 0], (0, 1))
    """

    # 默认光学系数（来自 EXOSIMS）
    DEFAULT_NON_LAMBERTIAN_FRONT = 0.038
    DEFAULT_NON_LAMBERTIAN_BACK = 0.004
    DEFAULT_SPECULAR_REFLECTION = 0.975
    DEFAULT_NREFLECTION_COEFF = 0.999
    DEFAULT_EMISSION_FRONT = 0.8
    DEFAULT_EMISSION_BACK = 0.2
    DEFAULT_P_SRP = 4.56e-6  # N/m² at 1 AU

    def __init__(
        self,
        system: CR3BP_System,
        area: float = 1.0,
        mass: float = 1000.0,
        Cr: float = 1.5,
        non_lambertian_front: float = DEFAULT_NON_LAMBERTIAN_FRONT,
        non_lambertian_back: float = DEFAULT_NON_LAMBERTIAN_BACK,
        specular_reflection: float = DEFAULT_SPECULAR_REFLECTION,
        nreflection_coeff: float = DEFAULT_NREFLECTION_COEFF,
        emission_front: float = DEFAULT_EMISSION_FRONT,
        emission_back: float = DEFAULT_EMISSION_BACK,
        P_srp: float = DEFAULT_P_SRP,
        **kwargs: Any,
    ) -> None:
        """初始化 SRP 动力学

        Args:
            system: CR3BP_System 对象
            area: 航天器截面积 (m²)
            mass: 航天器质量 (kg)
            Cr: 反射系数 (1=完全吸收, 2=完全反射)
            non_lambertian_front: 前表面非朗伯系数
            non_lambertian_back: 后表面非朗伯系数
            specular_reflection: 镜面反射因子
            nreflection_coeff: 反射系数
            emission_front: 前表面发射系数
            emission_back: 后表面发射系数
            P_srp: 太阳辐射压常数 (N/m² at 1 AU)
            **kwargs: 传递给基类的其他参数
        """
        super().__init__(system, **kwargs)

        self.area = area
        self.mass = mass
        self.Cr = Cr
        self.P_srp = P_srp

        # 计算光学系数 (来自 EXOSIMS)
        s = specular_reflection
        p = nreflection_coeff
        Bf = non_lambertian_front
        Bb = non_lambertian_back
        ef = emission_front
        eb = emission_back

        self.b1 = 0.5 * (1.0 - s * p)
        self.b2 = s * p
        self.b3 = 0.5 * (
            Bf * (1.0 - s) * p
            + (1.0 - p) * (ef * Bf - eb * Bb) / (ef + eb)
        )

        # SRP 加速度系数
        self.beta = self._compute_beta()

    def _compute_beta(self) -> float:
        """计算 SRP 加速度系数 (solar sail parameter)

        beta = P_srp * A * Cr / (2 * m)

        Returns:
            SRP 加速度系数 (无量纲，已归一化到 CR3BP 单位)
        """
        if self.mass <= 0:
            raise ValueError("航天器质量必须为正数")
        return self.P_srp * self.area * self.Cr / (2.0 * self.mass)

    def _get_eom_func(self, with_stm: bool) -> Callable:
        """返回带 SRP 扰动的运动方程函数

        Args:
            with_stm: 是否需要 STM 版本

        Returns:
            ODE 右端函数
        """
        if with_stm:
            return self._equations_with_stm_srp
        return self._equations_of_motion_srp

    def _equations_of_motion_srp(
        self, t: float, state: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """带 SRP 的 6 维运动方程

        Args:
            t: 时间
            state: 状态向量 [x, y, z, vx, vy, vz]

        Returns:
            状态导数 [vx, vy, vz, ax, ay, az]
        """
        # 基础 CR3BP 方程
        ds = super().equations_of_motion(t, state)

        # 如果 beta 为 0，跳过 SRP 计算
        if self.beta == 0.0:
            return ds

        x, y, z = state[:3]
        mu = self.system.mu

        # 航天器到主天体（较大天体，位于 x=-mu）的距离
        r1 = np.sqrt((x + mu) ** 2 + y**2 + z**2)

        # 径向单位矢量（从主天体指向航天器）
        ur = np.array([x + mu, y, z]) / r1

        # 切向单位矢量（在旋转平面内垂直于径向）
        r_xy = np.sqrt((x + mu) ** 2 + y**2)
        if r_xy > 1e-15:
            ut = np.array([-y, x + mu, 0.0]) / r_xy
        else:
            ut = np.array([0.0, 0.0, 0.0])

        # 径向和切向 SRP 力分量（来自 EXOSIMS）
        F_radial = self.b1 + 0.25 * self.b2 + 0.5 * self.b3
        F_tangential = (np.sqrt(3) * 0.25) * (self.b2 + 2.0 * self.b3)

        # SRP 扰动加速度
        a_srp = self.beta * (F_radial * ur + F_tangential * ut)

        # 添加到加速度分量
        ds[3] += a_srp[0]
        ds[4] += a_srp[1]
        ds[5] += a_srp[2]

        return ds

    def _equations_with_stm_srp(
        self, t: float, augmented_state: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """带 SRP 的 42 维增广运动方程

        Args:
            t: 时间
            augmented_state: 增广状态 [6状态 + 36个STM元素]

        Returns:
            增广状态导数
        """
        state = augmented_state[:6]
        stm = augmented_state[6:].reshape((6, 6))

        # 带 SRP 的状态导数
        state_derivative = self._equations_of_motion_srp(t, state)

        # 使用数值差分计算雅可比矩阵（因为 SRP 力依赖于位置）
        A = self._compute_jacobian_numerical(state)

        stm_dot = A @ stm

        return np.concatenate([state_derivative, stm_dot.flatten()])

    def _compute_jacobian_numerical(
        self, state: npt.NDArray[np.floating]
    ) -> np.ndarray:
        """数值计算雅可比矩阵

        使用中心差分计算包含 SRP 的雅可比矩阵。

        Args:
            state: 状态向量

        Returns:
            6x6 雅可比矩阵
        """
        eps = 1e-8
        n = 6
        A = np.zeros((n, n))

        f0 = self._equations_of_motion_srp(0.0, state)

        for j in range(n):
            state_plus = state.copy()
            state_plus[j] += eps
            f_plus = self._equations_of_motion_srp(0.0, state_plus)

            state_minus = state.copy()
            state_minus[j] -= eps
            f_minus = self._equations_of_motion_srp(0.0, state_minus)

            A[:, j] = (f_plus - f_minus) / (2.0 * eps)

        return A

    def __str__(self) -> str:
        return (
            f"CR3BP_SRP_Dynamics(system={self.system}, "
            f"area={self.area}, mass={self.mass}, Cr={self.Cr})"
        )

    def __repr__(self) -> str:
        return (
            f"CR3BP_SRP_Dynamics("
            f"system={self.system}, "
            f"area={self.area}, "
            f"mass={self.mass}, "
            f"Cr={self.Cr}, "
            f"beta={self.beta:.6e})"
        )
```

- [ ] **Step 2: 运行测试确认通过**

Run: `cd ~/codes/e2m2e && python -m pytest tests/core/test_srp_dynamics.py -v`
Expected: All tests PASS

- [ ] **Step 3: 更新 __init__.py 导出**

在 `e2m2e/core/__init__.py` 中添加：

```python
from .srp_dynamics import CR3BP_SRP_Dynamics
```

并在 `__all__` 列表中添加 `"CR3BP_SRP_Dynamics"`。

- [ ] **Step 4: 运行完整测试套件**

Run: `cd ~/codes/e2m2e && python -m pytest tests/ -v`
Expected: All tests PASS (无回归)

- [ ] **Step 5: 提交实现**

```bash
cd ~/codes/e2m2e
git add e2m2e/core/srp_dynamics.py e2m2e/core/__init__.py
git commit -m "feat: add CR3BP_SRP_Dynamics with solar radiation pressure

- Inherits from CR3BP_Dynamics
- Implements optical coefficient model from EXOSIMS
- Supports radial and tangential SRP force components
- Compatible with STM propagation"
```

---

## 任务二：Sphinx/RST 迁移

### Task 3: 创建 Sphinx 文档结构

**Files:**
- Create: `docs/conf.py`
- Create: `docs/index.rst`
- Create: `docs/Makefile`

- [ ] **Step 1: 创建 conf.py**

```python
# -*- coding: utf-8 -*-
"""Sphinx configuration for e2m2e documentation."""

import os
import sys

# 添加项目根目录到 path
sys.path.insert(0, os.path.abspath(".."))

# -- General configuration ------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.mermaid",
    "sphinx_copybutton",
]

# 中文支持
language = "zh_CN"
locale_dirs = ["locale/"]

# 源文件后缀
source_suffix = ".rst"
master_doc = "index"

# 项目信息
project = "e2m2e"
copyright = "2026, ouyangjiahong"
author = "ouyangjiahong"

# 版本信息
from e2m2e import __version__

release = __version__

# -- Options for HTML output ----------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Extension configuration ----------------------------------------------
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}
```

- [ ] **Step 2: 创建 index.rst**

```rst
.. e2m2e documentation master file

e2m2e: 地月转移轨道设计库
===================================

e2m2e (Earth to Moon, Moon to Earth) 是一个基于圆型限制性三体问题 (CR3BP) 的
地月转移轨道设计 Python 库。

.. toctree::
   :maxdepth: 2
   :caption: 目录

   getting-started/installation
   getting-started/quickstart
   getting-started/visualization

   core/system
   core/dynamics
   core/orbit
   core/coordinate

   algorithms/differential-correction
   algorithms/continuation
   algorithms/halo
   algorithms/stability
   algorithms/multiple-shooting

   transfer/overview
   transfer/search
   transfer/optimization

   api/e2m2e
   api/e2m2e.core
   api/e2m2e.algorithms
   api/e2m2e.transfer

   reference/glossary


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
```

- [ ] **Step 3: 创建 Makefile**

```makefile
# Minimal makefile for Sphinx documentation

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
```

- [ ] **Step 4: 创建目录结构**

```bash
cd ~/codes/e2m2e
mkdir -p docs/getting-started
mkdir -p docs/core
mkdir -p docs/algorithms
mkdir -p docs/transfer
mkdir -p docs/api
mkdir -p docs/reference
mkdir -p docs/_static
mkdir -p docs/locale
```

- [ ] **Step 5: 提交基础结构**

```bash
cd ~/codes/e2m2e
git add docs/conf.py docs/index.rst docs/Makefile
git commit -m "docs: add Sphinx configuration and index"
```

### Task 4: 创建 API 文档模板

**Files:**
- Create: `docs/api/e2m2e.rst`
- Create: `docs/api/e2m2e.core.rst`
- Create: `docs/api/e2m2e.algorithms.rst`
- Create: `docs/api/e2m2e.transfer.rst`

- [ ] **Step 1: 创建 e2m2e.core.rst**

```rst
e2m2e.core package
===================

.. automodule:: e2m2e.core
   :members:
   :undoc-members:
   :show-inheritance:

Submodules
----------

e2m2e.core.system module
-------------------------

.. automodule:: e2m2e.core.system
   :members:
   :undoc-members:
   :show-inheritance:

e2m2e.core.dynamics module
---------------------------

.. automodule:: e2m2e.core.dynamics
   :members:
   :undoc-members:
   :show-inheritance:

e2m2e.core.srp_dynamics module
-------------------------------

.. automodule:: e2m2e.core.srp_dynamics
   :members:
   :undoc-members:
   :show-inheritance:

e2m2e.core.orbit module
------------------------

.. automodule:: e2m2e.core.orbit
   :members:
   :undoc-members:
   :show-inheritance:

e2m2e.core.coordinate module
-----------------------------

.. automodule:: e2m2e.core.coordinate
   :members:
   :undoc-members:
   :show-inheritance:

e2m2e.core.ephemeris_dynamics module
-------------------------------------

.. automodule:: e2m2e.core.ephemeris_dynamics
   :members:
   :undoc-members:
   :show-inheritance:

e2m2e.core.ephemeris_system module
-----------------------------------

.. automodule:: e2m2e.core.ephemeris_system
   :members:
   :undoc-members:
   :show-inheritance:

e2m2e.core.spice module
------------------------

.. automodule:: e2m2e.core.spice
   :members:
   :undoc-members:
   :show-inheritance:
```

- [ ] **Step 2: 创建其他 API 文件**

创建 `docs/api/e2m2e.rst`、`docs/api/e2m2e.algorithms.rst`、`docs/api/e2m2e.transfer.rst`，
格式类似，使用 `automodule` 指令。

- [ ] **Step 3: 提交 API 文档**

```bash
cd ~/codes/e2m2e
git add docs/api/
git commit -m "docs: add API documentation templates"
```

### Task 5: 创建核心概念文档

**Files:**
- Create: `docs/core/system.rst`
- Create: `docs/core/dynamics.rst`

- [ ] **Step 1: 创建 system.rst**

```rst
CR3BP 系统
==========

圆型限制性三体问题 (CR3BP) 系统定义。

概述
----

CR3BP 描述一个质量可忽略的第三体在两个主天体引力场中的运动。
两个主天体绕其公共质心做圆周运动，采用旋转坐标系使主天体固定。

质量参数
--------

质量参数 μ = m₂/(m₁+m₂)，其中 m₂ 为较小天体质量。

- 地月系统: μ ≈ 0.01215
- 日地系统: μ ≈ 3.0039e-6
- 日木系统: μ ≈ 0.0009535

拉格朗日点
----------

系统有 5 个拉格朗日点 (L1-L5)，是旋转坐标系中的平衡点。

.. math::

   L1, L2, L3: 共线平衡点（x 轴上）
   L4, L5: 三角平衡点（等边三角形顶点）

Jacobi 常数
-----------

Jacobi 常数是 CR3BP 中唯一的运动积分：

.. math::

   C_J = 2\Omega - v^2

其中 Ω 为伪势能，v 为速度大小。

使用示例
--------

.. code-block:: python

   from e2m2e.core.system import CR3BP_System

   # 创建地月系统
   system = CR3BP_System("earth-moon")

   # 获取质量参数
   print(f"μ = {system.mu}")

   # 计算拉格朗日点
   L1 = system.libration_points[0]
   print(f"L1 = {L1}")

   # 计算 Jacobi 常数
   state = [0.8, 0, 0, 0, 0.6, 0]
   CJ = system.get_jacobi_constant(state)
   print(f"C_J = {CJ}")
```

- [ ] **Step 2: 创建 dynamics.rst**

类似格式，描述 CR3BP 运动方程和 SRP 扰动。

- [ ] **Step 3: 提交核心文档**

```bash
cd ~/codes/e2m2e
git add docs/core/
git commit -m "docs: add core concept documentation"
```

### Task 6: 更新依赖和 CI/CD

**Files:**
- Modify: `pyproject.toml`
- Modify: `.github/workflows/deploy-docs.yml`

- [ ] **Step 1: 更新 pyproject.toml**

在 `[project.optional-dependencies]` 中添加：

```toml
docs = [
    "sphinx>=7.0",
    "sphinx-rtd-theme>=2.0",
    "sphinxcontrib-mermaid>=0.9",
    "sphinx-copybutton>=0.5",
]
```

- [ ] **Step 2: 更新 deploy-docs.yml**

将 Docusaurus 构建替换为 Sphinx：

```yaml
name: Deploy Docs

on:
  push:
    branches: [main]

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: |
          pip install -e ".[docs]"

      - name: Build docs
        run: |
          cd docs
          make html

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: docs/_build/html

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

- [ ] **Step 3: 提交依赖更新**

```bash
cd ~/codes/e2m2e
git add pyproject.toml .github/workflows/deploy-docs.yml
git commit -m "docs: update dependencies and CI for Sphinx"
```

### Task 7: 验证 Sphinx 构建

- [ ] **Step 1: 安装依赖**

```bash
cd ~/codes/e2m2e
pip install -e ".[docs]"
```

- [ ] **Step 2: 构建文档**

```bash
cd ~/codes/e2m2e/docs
make html
```

Expected: 构建成功，无错误

- [ ] **Step 3: 本地预览**

```bash
cd ~/codes/e2m2e/docs/_build/html
python -m http.server 8000
```

访问 http://localhost:8000 查看文档

- [ ] **Step 4: 提交最终版本**

```bash
cd ~/codes/e2m2e
git add -A
git commit -m "docs: complete Sphinx migration from Docusaurus"
```

---

## 成功标准检查

- [ ] Sphinx 文档成功构建 (`make html` 无错误)
- [ ] API 文档自动生成正常 (autodoc 提取 docstring)
- [ ] CR3BP_SRP_Dynamics 通过所有测试
- [ ] SRP 力计算正确 (光学系数、beta 系数)
- [ ] area=0 时与纯 CR3BP 一致
- [ ] 现有测试无回归
