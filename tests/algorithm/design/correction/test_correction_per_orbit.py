"""每类轨道的微分修正收敛测试（阶段 2）。

7 类代表轨道各对应一个 session 缓存 fixture（见 conftest._corrected_*_cached），
本文件只验证收敛行为——闭合误差、周期性、成功标志、迭代次数合理；
不测类型/属性（那是阶段 1 已清掉的废话）。
"""

import pytest

# DifferentialCorrection 默认最大迭代次数，用作迭代次数合理性上界。
from e2m2e.algorithm.solver.differential_correction import DifferentialCorrection

pytestmark = pytest.mark.orchestration


# (orbit_id, session 缓存 fixture 名) —— parametrize 项与缓存一一对应。
CORRECTION_CASES = [
    ("dro", "corrected_dro"),
    ("halo_l1", "corrected_halo_l1"),
    ("halo_l2", "corrected_halo_l2"),
    ("lyapunov_l1", "corrected_lyapunov_l1"),
    ("axial_l1", "corrected_axial_l1"),
    ("dpo", "corrected_dpo"),
    ("triangular_l4", "corrected_triangular_l4"),
]


@pytest.mark.parametrize("orbit_id, fixture_name", CORRECTION_CASES)
def test_correction_converges(orbit_id, fixture_name, request):
    """每类轨道从标准种子修正后应收敛到周期轨道。

    验证四点（真行为，非类型检查）：
    - ``closure_error < 1e-6``：全周期首尾闭合；
    - ``is_periodic is True``：闭合误差低于周期性判据；
    - ``correction_success is True``：修正器自报成功；
    - ``1 <= correction_iterations < max_iterations``：迭代次数合理（非零、未触顶）。
    """
    orbit = request.getfixturevalue(fixture_name)

    assert orbit.closure_error < 1e-6, (
        f"{orbit_id}: closure_error={orbit.closure_error:.3e} 未达 1e-6"
    )
    assert orbit.is_periodic is True, f"{orbit_id}: is_periodic 不为 True"
    assert orbit.correction_success is True, f"{orbit_id}: correction_success 不为 True"
    assert 1 <= orbit.correction_iterations < DifferentialCorrection.DEFAULT_MAX_ITERATIONS, (
        f"{orbit_id}: correction_iterations={orbit.correction_iterations} 不在 "
        f"[1, {DifferentialCorrection.DEFAULT_MAX_ITERATIONS}) 内"
    )
