"""``Facade().mission_architecture_search`` 接口契约测试（#723，ADR 0054）。

覆盖：请求校验的 INVALID_PARAMS 翻译、工具面派生（request_model / 长任务路由）、
进度适配器容错与响应映射（无 SPICE，罐装算法结果）、SPICE 冒烟端到端。
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from kernel_helpers import requires_spice

from e2m2e.algorithm.transfer import FlybyEvaluation, MgaCandidate, MgaSearchResult
from e2m2e.api import Facade
from e2m2e.api.execution import LONG_RUNNING_TOOLS
from e2m2e.api.facade import tool_inventory
from e2m2e.api.models import (
    MissionArchitectureSearchRequest,
    MissionArchitectureSearchResponse,
    OrbitError,
)
from e2m2e.status import ConvergenceState, FailureCause

pytestmark = pytest.mark.interface

_DAY = 86400.0
_JD_TDB_AT_J2000 = 2451545.0


def _valid_params(**overrides: Any) -> dict[str, Any]:
    """合法请求参数（纯 JD_TDB 窗口 → 不需要内核）。"""
    params: dict[str, Any] = {
        "body_sequence": ["EARTH", "VENUS"],
        "launch_window": [_JD_TDB_AT_J2000 + 9500.0, _JD_TDB_AT_J2000 + 9502.0],
        "launch_window_step_days": 1.0,
        "leg_tof_ranges": [[100.0, 110.0]],
        "leg_tof_step_days": 5.0,
        "min_flyby_pericenter_km": [],
        "top_n": 3,
    }
    params.update(overrides)
    return params


class TestRequestValidation:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"body_sequence": ["EARTH"]},
            {"leg_tof_ranges": [[100.0, 110.0], [200.0, 210.0]]},
            {"leg_tof_ranges": [[110.0, 100.0]]},
            {"leg_tof_ranges": [[100.0, 100.0]]},
            {"min_flyby_pericenter_km": [1.0]},
            {"min_flyby_pericenter_km": [-1.0]},
            {"launch_window": [2461041.5]},
            {"launch_window": [2461041.5, 2461041.5]},
            {"launch_window": [2461049.5, 2461041.5]},
            {"launch_window": [math.inf, 2461041.5]},
            {"launch_window": ["", "2026-01-10T00:00:00"]},
            {"launch_window_step_days": 0.0},
            {"leg_tof_step_days": -1.0},
            {"v_inf_match_tol_km_s": -0.1},
            {"top_n": 0},
        ],
    )
    def test_invalid_request_translates_to_invalid_params(self, overrides):
        with pytest.raises(OrbitError) as excinfo:
            Facade().mission_architecture_search(**_valid_params(**overrides))
        assert excinfo.value.code == "INVALID_PARAMS"

    def test_missing_required_fields_rejected(self):
        with pytest.raises(OrbitError) as excinfo:
            Facade().mission_architecture_search()
        assert excinfo.value.code == "INVALID_PARAMS"


class TestToolDerivation:
    def test_inventory_entry_carries_request_model(self):
        inventory = {info.name: info for info in tool_inventory(Facade())}
        info = inventory["mission_architecture_search"]
        assert info.status == "implemented"
        assert info.request_model is MissionArchitectureSearchRequest

    def test_routed_as_long_running_tool(self):
        assert "mission_architecture_search" in LONG_RUNNING_TOOLS


class TestProgressAndResponseMapping:
    """无 SPICE：算法层被替换为罐装结果，验证映射与进度容错。"""

    @staticmethod
    def _canned_result() -> MgaSearchResult:
        return MgaSearchResult(
            status=ConvergenceState.CONVERGED,
            cause=FailureCause.NONE,
            message="罐装结果",
            candidates=(
                MgaCandidate(
                    launch_epoch_et=7.0 * _DAY,
                    leg_tofs_days=(105.0, 155.0),
                    departure_v_inf_km_s=3.5,
                    arrival_v_inf_km_s=2.9,
                    total_delta_v_km_s=6.4,
                    flybys=(
                        FlybyEvaluation(
                            body="VENUS",
                            v_inf_km_s=4.0,
                            turn_angle_rad=math.radians(30.0),
                            pericenter_radius_km=8000.0,
                            tisserand_before=2.9,
                            tisserand_after=2.9,
                        ),
                        FlybyEvaluation(
                            body="VENUS",
                            v_inf_km_s=4.0,
                            turn_angle_rad=0.0,
                            pericenter_radius_km=math.inf,
                            tisserand_before=2.9,
                            tisserand_after=2.9,
                        ),
                    ),
                ),
            ),
        )

    def test_progress_failure_is_swallowed_and_response_maps_candidate(self, monkeypatch):
        import e2m2e.algorithm.transfer as transfer

        captured: dict[str, Any] = {}

        def fake_search(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            on_delta = kwargs["progress_callback"]
            if on_delta is not None:
                on_delta(1)
                on_delta(2)
            return self._canned_result()

        monkeypatch.setattr(transfer, "search_mga_chains", fake_search)

        def raising_progress(fraction, message=None):
            raise RuntimeError("回调故意失败")

        response = Facade().mission_architecture_search(
            **_valid_params(
                body_sequence=["EARTH", "VENUS", "EARTH"],
                leg_tof_ranges=[[100.0, 105.0], [150.0, 155.0]],
                min_flyby_pericenter_km=[1.0],
            ),
            progress_callback=raising_progress,
        )

        assert response.status is ConvergenceState.CONVERGED
        assert response.message == "罐装结果"
        # 纯 JD_TDB 窗口：内核未加载（ephemeris 交给算法层缺省路径）
        assert captured["kwargs"]["ephemeris"] is None
        assert captured["args"][0] == ["EARTH", "VENUS", "EARTH"]
        assert len(captured["args"][1]) == 3  # [9500, 9501, 9502] 天（含起点、按步长）
        assert captured["args"][1][0] == pytest.approx(9500.0 * _DAY)
        assert captured["args"][2] == [[100.0, 105.0], [150.0, 155.0]]  # 步长 5 天
        assert captured["args"][3] == [1.0]
        assert captured["kwargs"]["top_n"] == 3

        candidate = response.candidates[0]
        assert candidate.launch_epoch_jd_tdb == pytest.approx(_JD_TDB_AT_J2000 + 7.0)
        assert candidate.leg_tofs_days == [105.0, 155.0]
        assert candidate.total_delta_v_km_s == pytest.approx(6.4)
        assert candidate.flybys[0].turn_angle_deg == pytest.approx(30.0)
        assert candidate.flybys[0].pericenter_radius_km == 8000.0
        # δ == 0 的零退化：非有限 r_p 以 null 表达（#698 口径），JSON 全程合法
        assert candidate.flybys[1].pericenter_radius_km is None
        revived = MissionArchitectureSearchResponse.model_validate_json(response.model_dump_json())
        assert revived == response


@pytest.mark.spice
@requires_spice
class TestSpiceSmoke:
    def test_earth_venus_window_smoke(self):
        """真实星历端到端：窗口/网格展开 → Lambert leg → 候选翻译。"""
        response = Facade().mission_architecture_search(
            body_sequence=["EARTH", "VENUS"],
            launch_window=["2026-01-01T00:00:00", "2026-01-10T00:00:00"],
            launch_window_step_days=5.0,
            leg_tof_ranges=[[100.0, 110.0]],
            leg_tof_step_days=5.0,
            min_flyby_pericenter_km=[],
            top_n=3,
        )
        assert response.status is ConvergenceState.CONVERGED
        assert response.cause is FailureCause.NONE
        assert response.candidates
        totals = [candidate.total_delta_v_km_s for candidate in response.candidates]
        assert totals == sorted(totals)
        for candidate in response.candidates:
            assert candidate.arrival_v_inf_km_s > 0.0
            assert candidate.departure_v_inf_km_s > 0.0
            assert candidate.leg_tofs_days[0] in (100.0, 105.0, 110.0)
            # 窗口两端 UTC 2026-01-01 / 2026-01-10（JD_TDB 相差 ~69 s 的 UTC↔TDB 偏移）
            assert 2461041.4 <= candidate.launch_epoch_jd_tdb <= 2461049.6

    def test_inverted_string_window_rejected(self):
        """字符串窗口终点早于起点：ET 解析后拒绝（INVALID_PARAMS）。"""
        with pytest.raises(OrbitError) as excinfo:
            Facade().mission_architecture_search(
                body_sequence=["EARTH", "VENUS"],
                launch_window=["2026-01-10T00:00:00", "2026-01-01T00:00:00"],
                leg_tof_ranges=[[100.0, 110.0]],
                min_flyby_pericenter_km=[],
            )
        assert excinfo.value.code == "INVALID_PARAMS"
