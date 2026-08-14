"""算法测试共享 fixtures。

包含跨文件共享的通用 fixture（reference_et、cr3bp_system、cr3bp_dynamics）
以及 DRO 种子轨道修正场景的预配置 fixture。
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_Dynamics
from e2m2e.algorithm.solver.continuation import Continuation
from e2m2e.algorithm.solver.differential_correction import DifferentialCorrection
from e2m2e.data.constants import Datum
from e2m2e.data.types.orbit import Orbit

# DRO 种子参数（Cui et al. 2025）——整个测试套件统一使用该种子。
DRO_X0 = 0.79188556619742
DRO_VY0 = 0.573665890385585
DRO_PERIOD_GUESS = 6.307498


# =============================================================================
# 跨文件共享 fixtures（DRO / NRHO / multiple shooting / patch point 通用）
# =============================================================================
@pytest.fixture
def reference_et(spice_manager, reference_epoch):
    """参考历元 ET"""
    return spice_manager.utc_to_et(reference_epoch)


@pytest.fixture
def cr3bp_system():
    """地月 CR3BP 系统"""
    return _make_earth_moon_system()


@pytest.fixture
def cr3bp_dynamics(cr3bp_system):
    """CR3BP 动力学"""
    return CR3BP_Dynamics(system=cr3bp_system)


@pytest.fixture(scope="session")
def dro_seed_state() -> np.ndarray:
    """DRO 种子状态向量 [x, y, z, vx, vy, vz]，无量纲 CR3BP 单位。"""
    return np.array([DRO_X0, 0.0, 0.0, 0.0, DRO_VY0, 0.0])


@pytest.fixture(scope="session")
def dro_seed_orbit(dro_seed_state) -> Orbit:
    """DRO 种子 Orbit 对象（单点轨道），可直接用于微分修正。"""
    orbit = Orbit(states=[dro_seed_state], times=[0])
    orbit.period = DRO_PERIOD_GUESS
    return orbit


@pytest.fixture(scope="session")
def _corrected_dro_cached(dro_seed_orbit) -> Orbit:
    """每会话计算一次修正后的 DRO；代价高（5–15 次 STM 传播）。

    在此处（而非经 dro_corrector）构造修正器，使缓存结果不依赖
    function 作用域的 fixture。
    """
    system = _make_earth_moon_system()
    dynamics = CR3BP_Dynamics(system)
    corrector = DifferentialCorrection(dynamics)
    corrector.setup_2D_symmetric_x_fixed_x0(DRO_X0)
    return corrector.iterate_correction(dro_seed_orbit, verbose=False).orbit


@pytest.fixture
def corrected_dro(_corrected_dro_cached) -> Orbit:
    """每个测试都拿到修正后 DRO 的全新 deepcopy（可安全修改）。"""
    return copy.deepcopy(_corrected_dro_cached)


@pytest.fixture
def dro_corrector() -> DifferentialCorrection:
    """按标准 DRO 修正配置的全新 DifferentialCorrection。"""
    system = _make_earth_moon_system()
    dynamics = CR3BP_Dynamics(system)
    corrector = DifferentialCorrection(dynamics)
    corrector.setup_2D_symmetric_x_fixed_x0(DRO_X0)
    return corrector


@pytest.fixture
def dro_dynamics() -> CR3BP_Dynamics:
    """DRO 算法测试用的地月 CR3BP 动力学。"""
    system = _make_earth_moon_system()
    return CR3BP_Dynamics(system)


@pytest.fixture
def dro_continuation(dro_corrector) -> Continuation:
    """为 DRO 族生成配置的 Continuation 实例。"""
    return Continuation(corrector=dro_corrector, step=0.001)


def _make_earth_moon_system():
    """构造与现有测试套件一致的地月 CR3BP 系统。

    根 conftest 的 `earth_moon_system` 是 function 作用域的（测试可随意修改）。
    这里的 session 作用域 fixture 需要一套自有的全新系统；
    本辅助函数是这一决策的唯一落脚点。
    """
    from e2m2e.algorithm.dynamics import CR3BP_System

    return CR3BP_System(mu=Datum.DE421.mu, primary="earth", secondary="moon")
