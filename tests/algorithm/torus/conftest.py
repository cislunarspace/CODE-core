"""torus 测试共享夹具。"""

from __future__ import annotations

import pytest

from e2m2e.algorithm.dynamics import CR3BP_Dynamics
from e2m2e.algorithm.family import design_halo
from e2m2e.algorithm.family.orbits.walk import earth_moon_system
from e2m2e.data.types.orbit import Orbit


@pytest.fixture(scope="session")
def halo_seed_raw() -> tuple[Orbit, CR3BP_Dynamics]:
    """L2 北族 Halo（面外振幅 3000 km）与地月动力学，session 级只读缓存。

    该成员单值矩阵恰有一个非平凡单位模复对（面内双曲对 1206/0.0008），
    是环面构造的单中心对形态。平面 DRO 种子有两个单位模对，需要额外
    的面内振幅规范，属 PR2 的已记录限制（ADR 0063）。
    """
    dynamics = CR3BP_Dynamics(earth_moon_system())
    orbit = design_halo(2, 3000.0, dynamics=dynamics)
    return orbit, dynamics


@pytest.fixture
def halo_seed(halo_seed_raw: tuple[Orbit, CR3BP_Dynamics]) -> tuple[Orbit, CR3BP_Dynamics]:
    """L2 Halo 种子与地月动力学，函数级返回轨道副本。"""
    orbit, dynamics = halo_seed_raw
    return orbit.copy(), dynamics


@pytest.fixture(scope="session")
def dro_seed_raw() -> tuple[Orbit, CR3BP_Dynamics]:
    """标准种子 DRO 与地月动力学，session 级只读缓存。

    双中心对形态（面内与法向各一个单位模对）：环面构造丢弃未规范的
    面内振幅方向后旋转数不漂移，但振幅塌缩、构造不收敛，属第二阶段的
    已记录限制（ADR 0063 决策 8）。
    """
    from e2m2e.algorithm.family.orbits.dro import _correct_dro

    dynamics = CR3BP_Dynamics(earth_moon_system())
    orbit = _correct_dro(dynamics, 0.79188556619742, None)
    return orbit, dynamics


@pytest.fixture
def dro_seed(dro_seed_raw: tuple[Orbit, CR3BP_Dynamics]) -> tuple[Orbit, CR3BP_Dynamics]:
    """标准种子 DRO 与地月动力学，函数级返回轨道副本。"""
    orbit, dynamics = dro_seed_raw
    return orbit.copy(), dynamics
