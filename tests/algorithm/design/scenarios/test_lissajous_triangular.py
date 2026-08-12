"""design_orbit 全链路集成测试（orchestration）。

通过 ``design_orbit(DesignOrbitRequest(orbit_type="LISSAJOUS", ...))`` 全流程
（CR3BP 初猜 → two_level 星历修正 → 高精度预报），验证编排链路收敛、
星历有界、CR3BP 参考轨道 Jacobi 漂移受控。

字段形状契约（``EphemerisTable`` / ``DesignOrbitResponse`` 字段形状与类型）
已下沉到 ``tests/data/test_types_trajectory.py`` 与
``tests/api/test_facade.py``（ADR 0021：字段契约归 data/interface 类，
不靠真传播验证）。本文件只验证编排链路集成行为。

合并自原 ``test_lissajous.py`` + ``test_triangular.py``：原两文件 12 个 slow
测试逐行重复，各跑一次 18 天 ``design_orbit``。本文件用参数化 + 共享 module
fixture 让收敛与有界两个测试复用同一次 ``design_orbit`` 调用，字段形状契约
下沉零管线验证。

.. note::

   拟周期族（Lissajous / 三角平动点）星历修正用固定时间打靶：自由时间
   模式下时间自由度与沿流状态自由度近线性相关、雅可比病态，LM 线性收敛
   卡在 0.5–174 km（#366）；固定时间下 4–6 迭代收敛（秒级到几十秒）。

依赖 SPICE 内核（``design_orbit`` 自动加载 ``kernels/``）。
内核缺失时整组跳过。
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from kernel_helpers import SPICE_KERNEL_DIR

from e2m2e.algorithm.design import design_orbit
from e2m2e.api.models import DesignOrbitRequest
from e2m2e.data.templates import ConvergenceState

_SPICE_AVAILABLE = os.path.isdir(SPICE_KERNEL_DIR) and any(
    f.endswith(".bsp") for f in os.listdir(SPICE_KERNEL_DIR)
)

pytestmark = [
    pytest.mark.orchestration,
    pytest.mark.slow,
    pytest.mark.spice,
    pytest.mark.skipif(not _SPICE_AVAILABLE, reason="SPICE kernels not available"),
]

# 弧段须覆盖 ≥ 1 个 CR3BP 周期：地月 L1 Lissajous 面内周期 ~14 天，短弧
# （< 1 周期）patch points 采样不足，星历修正不收敛。保留 18 天（~1.3 周期），
# 与原 test_lissajous 一致。
DURATION_SEC = 0.05 * 365.25 * 86400

# 集成收敛参数化。覆盖共线 Lissajous（L1/L2）与三角平动点（L4/L5）：
# 拟周期族无周期闭合，星历修正用固定时间打靶（var_time=False，见
# design_orbit._FIXED_TIME_ORBIT_TYPES 与 #366），Gauss-Newton 二次收敛。
# (orbit_type, collinear_point, amplitude_in, amplitude_out)
_ORBIT_CASES = [
    pytest.param(("LISSAJOUS", 1, 500.0, 2000.0), id="LISSAJOUS-L1"),
    pytest.param(("LISSAJOUS", 2, 2500.0, 7500.0), id="LISSAJOUS-L2"),
    pytest.param(("L4", None, 8000.0, 6000.0), id="L4"),
    pytest.param(("L5", None, 8000.0, 6000.0), id="L5"),
]


def _build_request(
    orbit_type: str,
    collinear_point: int | None,
    amplitude_in: float,
    amplitude_out: float,
) -> DesignOrbitRequest:
    kwargs: dict[str, object] = dict(
        orbit_type=orbit_type,
        amplitude_in=amplitude_in,
        amplitude_out=amplitude_out,
        phase_in=0.0,
        phase_out=0.0,
        epoch="2025-01-01T00:00:00",
        duration=DURATION_SEC,
        output_step=3600.0,
    )
    if collinear_point is not None:
        kwargs["collinear_point"] = collinear_point
    return DesignOrbitRequest(**kwargs)  # type: ignore[arg-type]


@pytest.fixture(scope="module", params=_ORBIT_CASES)
def design_case(request: pytest.FixtureRequest):
    """每类型跑一次 design_orbit，module 内复用。

    返回 ``(期望 orbit_type, OrbitDesignResult)``。共享 fixture 让收敛与
    有界两个测试复用同一次 design_orbit 调用。
    """
    orbit_type, collinear_point, amp_in, amp_out = request.param
    result = design_orbit(_build_request(orbit_type, collinear_point, amp_in, amp_out))
    return orbit_type, result


def test_design_orbit_converges(design_case):
    """编排链路收敛：CR3BP 初猜 → two_level 星历修正 → 高精度预报。

    断言修正收敛 + 星历非空 + orbit_type 透传。字段形状契约（position_km /
    velocity_mps / synodic_position / initial_state）已下沉到 data/interface 类。
    """
    expected_type, result = design_case
    assert result.correction is not None
    assert result.correction.status is ConvergenceState.CONVERGED
    assert result.ephemeris is not None
    assert len(result.ephemeris) > 0
    assert result.orbit_type == expected_type


def test_design_orbit_bounded(design_case):
    """星历位置有界 + CR3BP 参考轨道 Jacobi 漂移受控。

    位置范围断言保留原 e2e 测试的物理判据（地月空间合理范围），验证高精度
    预报不爆炸。Jacobi 漂移断言验证轨道能量未发散：LISSAJOUS 的 cr3bp_orbit
    为中心流形约化的多点有界轨迹，沿轨道 Jacobi 漂移在约化精度量级（~1e-4，
    非纯数值积分的 1e-10 守恒级），且随振幅增长（大振幅 L2 实测 ~1e-3）。
    阈值取漂移相对能量面 <0.1%（|ΔC|/|C|），验证轨道未发散到错误能量面。
    """
    _, result = design_case
    pos_norms = np.linalg.norm(result.ephemeris.position_km, axis=1)
    assert pos_norms.max() < 1e6, f"最大地心距 {pos_norms.max():.0f} km 超出地月空间合理范围"
    assert pos_norms.min() > 1e4, f"最小地心距 {pos_norms.min():.0f} km 超出地月空间合理范围"

    states = np.asarray(result.cr3bp_orbit.states)
    if states.shape[0] > 1:
        system = result.cr3bp_orbit.system
        jacobi = np.array([system.get_jacobi_constant(s) for s in states])
        drift = float(jacobi.max() - jacobi.min())
        c_scale = float(max(abs(jacobi.min()), abs(jacobi.max())))
        assert drift < 1e-3 * c_scale, (
            f"CR3BP 参考轨道 Jacobi 漂移 {drift:.2e} 超出中心流形约化精度范围"
            f"（C ∈ [{jacobi.min():.6f}, {jacobi.max():.6f}]）"
        )
