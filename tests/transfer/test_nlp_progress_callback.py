"""
需求: DROTRONLPOptimizer 进度回调

验收标准:
  1. DROTRONLPOptimizer 新增 set_progress_callback(callback) 方法
  2. callback 签名: callback(iteration: int, objective: float,
                             alpha: float, transfer_time: float, t_ins: float) -> None
  3. 不设置 callback 时行为与当前完全一致（不报错、无副作用）
  4. callback 为 None 时等价于未设置
"""

import numpy as np
import pytest

from e2m2e.core import CR3BP_Dynamics, CR3BP_System, Orbit
from e2m2e.transfer import DROTRONLPOptimizer


def _simple_orbit(n: int = 80) -> Orbit:
    t = np.linspace(0, 6.28, n)
    x = 0.9 + 0.08 * np.cos(t)
    y = 0.08 * np.sin(t)
    z = np.zeros_like(t)
    vx = -0.08 * np.sin(t)
    vy = 0.08 * np.cos(t)
    vz = np.zeros_like(t)
    states = np.column_stack([x, y, z, vx, vy, vz])
    orbit = Orbit(states, t)
    orbit.period = float(t[-1])
    return orbit


@pytest.fixture
def system():
    return CR3BP_System(mu=1.21506683e-2, primary="earth", secondary="moon")


@pytest.fixture
def dynamics(system):
    d = CR3BP_Dynamics(system)
    d.integrator = "RK45"
    d.rtol = d.atol = 1e-3
    d.max_step = 2.0
    return d


@pytest.fixture
def optimizer(system, dynamics):
    dro = _simple_orbit(60)
    ro = _simple_orbit(50)
    dep = dro.states[0]
    return DROTRONLPOptimizer(
        system=system,
        dynamics=dynamics,
        departure_orbit=dro,
        arrival_orbit=ro,
        departure_state=dep,
    )


class TestSetProgressCallback:
    def test_method_exists(self, optimizer):
        assert hasattr(optimizer, "set_progress_callback")
        assert callable(optimizer.set_progress_callback)

    def test_accepts_callable(self, optimizer):
        def cb(it, obj, alpha, T, tins):
            pass

        optimizer.set_progress_callback(cb)
        assert optimizer._progress_callback is cb

    def test_accepts_none(self, optimizer):
        optimizer.set_progress_callback(None)
        assert optimizer._progress_callback is None
