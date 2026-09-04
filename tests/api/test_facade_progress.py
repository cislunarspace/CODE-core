"""长任务进度回调测试（#576 Phase 1）：Facade 进度形参 + 语义。

进度契约：``progress_callback(fraction, message=None)``，fraction ∈ [0, 1]
单调不减（0 = 开始，1 = 完成）；WSB 后端把既有网格 delta 回调映射到
(0.1, 0.9) 区间；回调异常不得中断计算；不传回调时签名向后兼容。
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from e2m2e.api.facade import Facade
from e2m2e.data.templates import ConvergenceState, FailureCause

pytestmark = pytest.mark.interface


@dataclass
class _FakeDetails:
    tli_epoch: float
    tof_sec: float


def _fake_transfer_result():
    from e2m2e.algorithm.transfer import ManeuverEvent

    return SimpleNamespace(
        status=ConvergenceState.CONVERGED,
        cause=FailureCause.NONE,
        message="任务完成",
        transfer_type="WSB",
        delta_v=3.9,
        trajectory=np.arange(6, dtype=float).reshape(1, 6),
        trajectory_times=np.array([0.0]),
        state_frame="synodic_barycentric_km",
        maneuver_events=(ManeuverEvent(kind="departure", t_sec=0.0, dv_km_s=3.9),),
        details=_FakeDetails(tli_epoch=2460800.5, tof_sec=100.0),
    )


def _patch_transfer_orbit(monkeypatch, deltas):
    """替身 transfer_orbit：记录收到的 progress_callback 并依次喂 delta。"""
    import e2m2e.algorithm.transfer as transfer

    received: list = []

    def fake_transfer_orbit(transfer_type, **kwargs):
        received.append(kwargs.get("progress_callback"))
        callback = kwargs.get("progress_callback")
        if callback is not None:
            for delta in deltas:
                callback(delta)
        return _fake_transfer_result()

    monkeypatch.setattr(transfer, "transfer_orbit", fake_transfer_orbit)
    return received


class TestTransferDesignProgress:
    def test_wsb_search_deltas_map_to_monotone_fractions(self, monkeypatch):
        """WSB 网格 delta → (0.1, 0.9) 区间分数；起止两端必然上报。"""
        received = _patch_transfer_orbit(monkeypatch, deltas=[10, 20])
        seen: list[tuple[float, str | None]] = []

        Facade().transfer_design(
            transfer_type="WSB",
            tli_epoch=2460800.5,
            target_ephemeris=[[1.0] * 6],
            progress_callback=lambda fraction, message=None: seen.append((fraction, message)),
        )

        fractions = [fraction for fraction, _ in seen]
        assert fractions[0] == 0.0
        assert fractions[-1] == 1.0
        assert fractions == sorted(fractions)
        assert any(0.0 < fraction < 1.0 for fraction in fractions)
        # WSB 网格 delta 回调确实穿到了算法层
        assert received and received[0] is not None

    def test_without_callback_is_backward_compatible(self, monkeypatch):
        """不传 progress_callback：算法层收到 None，调用方式与旧版一致。"""
        received = _patch_transfer_orbit(monkeypatch, deltas=[])

        response = Facade().transfer_design(
            transfer_type="HMN", tli_epoch=2460800.5, target_orbit_radius_km=384405.0
        )

        assert response.status is ConvergenceState.CONVERGED
        assert received == [None]

    def test_callback_exception_does_not_break_computation(self, monkeypatch):
        """进度观察者抛异常：计算照常完成（进度失败仅降级为无进度）。"""
        _patch_transfer_orbit(monkeypatch, deltas=[5])

        def raising(fraction, message=None):
            raise RuntimeError("observer boom")

        response = Facade().transfer_design(
            transfer_type="WSB",
            tli_epoch=2460800.5,
            target_ephemeris=[[1.0] * 6],
            progress_callback=raising,
        )
        assert response.status is ConvergenceState.CONVERGED

    def test_hmn_has_only_start_and_end(self, monkeypatch):
        """无搜索进度的后端（HMN）：只有 0.0 与 1.0 两次上报。"""
        received = _patch_transfer_orbit(monkeypatch, deltas=[])
        seen: list[float] = []

        Facade().transfer_design(
            transfer_type="HMN",
            tli_epoch=2460800.5,
            target_orbit_radius_km=384405.0,
            progress_callback=lambda fraction, message=None: seen.append(fraction),
        )

        assert seen == [0.0, 1.0]
        assert received == [None]  # 非 WSB 后端不注入 delta 回调


class _FakeFamily:
    """OrbitFamily 兼容替身（零成员：可迭代、可 len）。"""

    family_type = "dro"
    system = None
    metadata: dict = {}

    def __init__(self) -> None:
        self.orbits: list = []

    def __iter__(self):
        return iter(self.orbits)

    def __len__(self) -> int:
        return len(self.orbits)


class TestFamilyGenerationProgress:
    def test_stage_level_start_and_end(self, monkeypatch):
        """族生成（单次 Rust 调用）：阶段级 0.0/1.0 两次上报。"""
        import e2m2e.algorithm.family as family_pkg

        monkeypatch.setattr(
            family_pkg,
            "design_dro_family",
            lambda *args, **kwargs: _FakeFamily(),
        )
        seen: list[float] = []

        Facade().catalog.orbit_family_generation(
            orbit_type="DRO",
            n_orbits=2,
            progress_callback=lambda fraction, message=None: seen.append(fraction),
        )

        assert seen == [0.0, 1.0]

    def test_without_callback_is_backward_compatible(self, monkeypatch):
        import e2m2e.algorithm.family as family_pkg

        monkeypatch.setattr(
            family_pkg,
            "design_dro_family",
            lambda *args, **kwargs: _FakeFamily(),
        )

        response = Facade().catalog.orbit_family_generation(orbit_type="DRO", n_orbits=2)

        assert response.status is ConvergenceState.CONVERGED
