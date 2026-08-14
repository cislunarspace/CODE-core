"""WSB 搜索结果容器的公开数据契约测试。"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.transfer import WsbSearchParams, WsbTransferDetails
from e2m2e.data.templates import ConvergenceState, FailureCause

# 结果容器的数据结构与默认值契约：按验证对象归 data（ADR 0021 理由 5）；
# 目录镜像源位于 transfer（ADR 0021 决策 2：目录导航、marker 标功能类）。
pytestmark = pytest.mark.data


def test_wsb_transfer_details_preserves_payload_and_status_triplet():
    """结果容器保留数值载荷与统一状态三元组。"""
    details = WsbTransferDetails(
        tli_epoch="2025-01-01T00:00:00",
        tof_sec=1e7,
        perilune_alt_km=100.0,
        perilune_vel_km_s=2.5,
        perilune_state=np.zeros(6),
        h2_kepler=-0.5,
        dv_departure_km_s=3.1,
        dv_arrival_km_s=0.8,
        n_candidates_searched=100,
        n_candidates_feasible=5,
        status=ConvergenceState.CONVERGED,
        cause=FailureCause.NONE,
        message="收敛",
        search_params=WsbSearchParams(),
    )

    assert isinstance(details.tof_sec, float)
    assert isinstance(details.perilune_alt_km, float)
    assert isinstance(details.perilune_vel_km_s, float)
    assert isinstance(details.dv_departure_km_s, float)
    assert isinstance(details.dv_arrival_km_s, float)
    assert isinstance(details.h2_kepler, float)
    assert isinstance(details.n_candidates_searched, int)
    assert isinstance(details.n_candidates_feasible, int)
    assert details.status is ConvergenceState.CONVERGED
    assert details.cause is FailureCause.NONE
    assert details.message == "收敛"
    assert isinstance(details.search_params, WsbSearchParams)
