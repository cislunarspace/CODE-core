"""NRHO 族生成与请求半球的几何一致性（issue #586 / ADR 0042）。

北/南（``north_south`` 1/2）按 ADR 0042 的几何约定验收：成员存储态
是 vy<0 穿越点，该处 z0 符号即半球（北 +、南 −）；并用轨道分类学对
最小形态成员实测，primary 标签必须与请求半球一致。L1/L2 × 北/南 四向
都验——#586 的病灶正是族生成内核的种子相位反号，只验单侧测不出。
"""

from __future__ import annotations

import pytest

from e2m2e.algorithm.family import design_nrho_family
from e2m2e.algorithm.orbit_taxonomy import classify_orbit
from e2m2e.data.templates import ConvergenceState

pytestmark = pytest.mark.orchestration

#: (libration_point, north_south, 期望存储态 z0 符号, 期望分类学 canonical)
_CASES = [
    (1, 1, +1.0, "halo_l1_northern"),
    (1, 2, -1.0, "halo_l1_southern"),
    (2, 1, +1.0, "halo_l2_northern"),
    (2, 2, -1.0, "halo_l2_southern"),
]


@pytest.mark.parametrize("point,north_south,z_sign,canonical", _CASES)
def test_nrho_family_hemisphere_matches_request(
    point: int, north_south: int, z_sign: float, canonical: str
) -> None:
    """族成员存储态与实测分类学的半球必须与 north_south 请求一致。"""
    result = design_nrho_family(point, north_south, 20000.0, n_orbits=3)
    assert result.status is ConvergenceState.CONVERGED
    family = result.family
    assert family.orbits, "NRHO 族零成员"
    for orbit in family.orbits:
        state = orbit.states[0]
        assert state[4] < 0.0, "存储约定：vy<0 穿越点"
        assert state[2] * z_sign > 0.0, "vy<0 穿越点处 z0 符号 = 请求半球"
        assert orbit.period is not None and orbit.period > 0.0

    first = family.orbits[0]
    assert first.period is not None
    measured = classify_orbit(first.states, period=first.period)
    assert measured.status is ConvergenceState.CONVERGED
    assert measured.labels, f"成员实测未分类：{measured.unclassified_reason}"
    assert measured.labels[0].canonical == canonical
