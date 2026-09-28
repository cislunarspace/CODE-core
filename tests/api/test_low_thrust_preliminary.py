"""``Facade().low_thrust_preliminary`` 接口契约测试（#725，ADR 0054 决策 2/3）。

覆盖：请求校验 → INVALID_PARAMS 翻译、工具面派生（inventory 携带请求模型、
长任务路由）、罐装解的响应映射与进度容错（无 Rust 依赖）、以及真内核的
单 leg rendezvous 端到端冒烟（复刻 ``test_sims_flanagan._helio_150_problem``
的 1 AU → 1.5 AU Hohmann 定相几何）。纯 conic 档二体，无需 SPICE 内核。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from kernel_helpers import requires_native_symbols

from e2m2e.algorithm.transfer import (
    SimsFlanaganFlybyResult,
    SimsFlanaganLegSolution,
    SimsFlanaganMultiLegSolution,
    SimsFlanaganNode,
    SimsFlanaganPropulsion,
)
from e2m2e.api import Facade
from e2m2e.api.execution import LONG_RUNNING_TOOLS
from e2m2e.api.facade import tool_inventory
from e2m2e.api.models import (
    LowThrustPreliminaryRequest,
    LowThrustPreliminaryResponse,
    OrbitError,
)
from e2m2e.status import ConvergenceState, FailureCause

pytestmark = pytest.mark.interface

MU_SUN = 1.32712440018e11  # 太阳 GM（km³/s²），DE421 口径
AU_KM = 1.495978707e8  # IAU 2012 天文单位（km）

_MISSING = object()  # 参数化中表达“该字段缺省”


def _valid_params(**overrides: Any) -> dict[str, Any]:
    """合法请求参数（1 AU → 1.5 AU Hohmann 定相几何，纯 conic 档不需要内核）。"""
    r1, r2 = AU_KM, 1.5 * AU_KM
    v1, v2 = math.sqrt(MU_SUN / r1), math.sqrt(MU_SUN / r2)
    tof = math.pi * math.sqrt(((r1 + r2) / 2.0) ** 3 / MU_SUN)
    params: dict[str, Any] = {
        "backend": "conic",
        "departure_state": [r1, 0.0, 0.0, 0.0, v1, 0.0],
        "nodes": [{"kind": "rendezvous", "state": [-r2, 0.0, 0.0, 0.0, -v2, 0.0]}],
        "leg_tofs_s": [tof],
        "mu_km3_s2": MU_SUN,
        "initial_mass_kg": 1000.0,
        "propulsion": {"isp_s": 3000.0, "t_max_n": 5.0},
    }
    for key, value in overrides.items():
        if value is _MISSING:
            params.pop(key, None)
        else:
            params[key] = value
    return params


class TestRequestValidation:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"backend": _MISSING},
            {"backend": "ephemeris"},
            {"propulsion": {"isp_s": 3000.0, "t_max_n": 5.0, "p0_w": 40000.0}},
            {"propulsion": {"isp_s": 3000.0}},
            {"nodes": [{"kind": "rendezvous", "state": [1.0] * 5}]},
            {"leg_tofs_s": [1.0, 2.0]},
            {"leg_tofs_s": [-1.0]},
            {"n_segments": 1},
            {"n_segments": [1]},
            {"cost": "min_time"},
            {"cost": "weighted"},
            {"cost": "weighted", "cost_weights": [1.0]},
            {"cost": "weighted", "cost_weights": [1.0, -2.0], "tof_bounds_s": [1.0, 2.0]},
            {"cost": "min_fuel", "cost_weights": [1.0, 1.0]},
            {"cost": "min_fuel", "tof_bounds_s": [1.0, 2.0]},
            {"mu_km3_s2": 0.0},
        ],
    )
    def test_invalid_request_translates_to_invalid_params(self, overrides):
        with pytest.raises(OrbitError) as excinfo:
            Facade().low_thrust_preliminary(**_valid_params(**overrides))
        assert excinfo.value.code == "INVALID_PARAMS"

    def test_ephemeris_backend_reports_not_implemented(self):
        with pytest.raises(OrbitError, match="ephemeris") as excinfo:
            Facade().low_thrust_preliminary(**_valid_params(backend="ephemeris"))
        assert excinfo.value.code == "INVALID_PARAMS"
        assert "ephemeris 档未实现" in str(excinfo.value)


class TestToolDerivation:
    def test_inventory_entry_carries_request_model(self):
        inventory = {info.name: info for info in tool_inventory(Facade())}
        info = inventory["low_thrust_preliminary"]
        assert info.status == "implemented"
        assert info.request_model is LowThrustPreliminaryRequest

    def test_routed_as_long_running_tool(self):
        assert "low_thrust_preliminary" in LONG_RUNNING_TOOLS


class TestResponseMapping:
    """无 Rust 依赖：问题类被替换为罐装解，验证映射与进度容错。"""

    @staticmethod
    def _canned_solution() -> SimsFlanaganMultiLegSolution:
        node_times = np.linspace(0.0, 2.2e7, 7)
        return SimsFlanaganMultiLegSolution(
            legs=[
                SimsFlanaganLegSolution(
                    impulses_km_s=np.tile([0.01, -0.02, 0.0], (6, 1)),
                    impulse_times_s=0.5 * (node_times[:-1] + node_times[1:]),
                    node_times_s=node_times,
                    forward_states=np.arange(42.0).reshape(7, 6),
                    backward_states=np.arange(42.0).reshape(7, 6)[::-1].copy(),
                    matchpoint_residual=np.arange(6.0) * 1e-6,
                )
            ],
            flybys=[
                SimsFlanaganFlybyResult(
                    v_inf_in_km_s=np.array([3.0, 0.0, 0.0]),
                    v_inf_out_km_s=np.array([3.0 * math.cos(0.5), 3.0 * math.sin(0.5), 0.0]),
                    v_inf_km_s=3.0,
                    turn_angle_rad=0.5,
                    pericenter_radius_km=math.inf,  # δ≠0 时实际有限；此处钉 inf→None 映射
                )
            ],
            leg_tofs_s=np.array([2.2e7]),
            delta_v_total_km_s=5.4,
            final_mass_kg=807.2,
            fuel_kg=192.8,
            objective_value=5.4,
            cost="min_fuel",
            n_iter=42,
            status=ConvergenceState.CONVERGED,
            cause=FailureCause.NONE,
            message="罐装解",
        )

    def test_canned_solution_maps_and_progress_failure_swallowed(self, monkeypatch):
        import e2m2e.algorithm.transfer as transfer

        captured: dict[str, Any] = {}

        class FakeProblem:
            def __init__(self, departure, nodes, tofs, propulsion, m0, mu, *, backend):
                captured["init"] = {
                    "departure": departure,
                    "nodes": nodes,
                    "tofs": tofs,
                    "propulsion": propulsion,
                    "m0": m0,
                    "mu": mu,
                    "backend": backend,
                }

            def solve(self, n_segments, **kwargs):
                captured["solve"] = {"n_segments": n_segments, **kwargs}
                return TestResponseMapping._canned_solution()

        monkeypatch.setattr(transfer, "SimsFlanaganMultiLegProblem", FakeProblem)

        def raising_progress(fraction, message=None):
            raise RuntimeError("回调故意失败")

        response = Facade().low_thrust_preliminary(
            **_valid_params(n_segments=6, guess="edelbaum"),
            progress_callback=raising_progress,
        )

        # 组装：backend 透传、出发状态 numpy 化、节点为算法层对象
        assert captured["init"]["backend"] == "conic"
        np.testing.assert_allclose(
            captured["init"]["departure"],
            np.asarray(_valid_params()["departure_state"], dtype=float),
        )
        node = captured["init"]["nodes"][0]
        assert isinstance(node, SimsFlanaganNode)
        assert node.kind == "rendezvous"
        assert captured["init"]["m0"] == 1000.0
        assert captured["init"]["mu"] == MU_SUN
        assert isinstance(captured["init"]["propulsion"], SimsFlanaganPropulsion)
        assert captured["init"]["propulsion"].t_max_n == 5.0
        # solve 透传求解配置（cost_weights None 而非空元组）
        assert captured["solve"] == {
            "n_segments": 6,
            "cost": "min_fuel",
            "cost_weights": None,
            "tof_bounds_s": None,
            "guess": "edelbaum",
            "ftol": 1e-9,
            "maxiter": 200,
            "use_analytic_jac": True,
        }

        # 响应映射：转角换度、inf 近心点 → None（#698 口径）、leg 剖面齐全
        assert response.status is ConvergenceState.CONVERGED
        assert response.message == "罐装解"
        flyby = response.flybys[0]
        assert flyby.turn_angle_deg == pytest.approx(math.degrees(0.5))
        assert flyby.pericenter_radius_km is None
        assert flyby.v_inf_km_s == pytest.approx(3.0)
        assert flyby.v_inf_in_km_s == [3.0, 0.0, 0.0]
        leg = response.legs[0]
        assert len(leg.impulses_km_s) == 6
        assert leg.impulses_km_s[0] == [0.01, -0.02, 0.0]
        assert len(leg.node_times_s) == 7
        assert len(leg.matchpoint_residual) == 6
        assert response.leg_tofs_s == [2.2e7]
        assert response.delta_v_total_km_s == pytest.approx(5.4)
        assert response.n_iter == 42
        # JSON round-trip（allow_inf_nan=False 契约下合法）
        revived = LowThrustPreliminaryResponse.model_validate_json(response.model_dump_json())
        assert revived == response


@requires_native_symbols("propagate_kepler_py")
class TestConicSmoke:
    def test_single_leg_rendezvous_end_to_end(self):
        response = Facade().low_thrust_preliminary(**_valid_params(n_segments=6, guess="edelbaum"))
        assert response.status is ConvergenceState.CONVERGED
        assert response.delta_v_total_km_s > 0
        assert 0.0 < response.final_mass_kg < 1000.0
        assert len(response.legs) == 1
        assert len(response.legs[0].impulses_km_s) == 6
        assert response.flybys == []
        assert response.fuel_kg == pytest.approx(1000.0 - response.final_mass_kg)
