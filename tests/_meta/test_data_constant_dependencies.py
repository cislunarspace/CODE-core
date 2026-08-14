"""数据层单一来源的架构约束测试。"""

import pytest

from e2m2e.data.constants import (
    SECONDS_PER_DAY,
    SECONDS_PER_JULIAN_YEAR,
    SPEED_OF_LIGHT_KMS,
    Datum,
)
from e2m2e.data.constants.bodies import EARTH, SUN
from e2m2e.data.templates import CHAR_LENGTH_KM

pytestmark = pytest.mark.aux


def test_algorithm_defaults_are_sourced_from_data_constants():
    """算法默认值必须从 data.constants 取值，不能在算法层重新定义。"""
    from e2m2e.algorithm.dynamics.bcr4bp_system import BCR4BPSystem
    from e2m2e.algorithm.dynamics.cr3bp_system import CR3BP_System
    from e2m2e.algorithm.family.cr3bp_orbits import CHAR_LENGTH_KM as family_char_length
    from e2m2e.algorithm.forces.force_mapping import perturbation_to_force_config
    from e2m2e.algorithm.forces.gravity_file import _DEFAULT_MU
    from e2m2e.algorithm.propagation import _DEFAULT_FORCE_CONFIG

    assert SUN.gm_by_datum["DE440"] == BCR4BPSystem.SUN_GM_KM3_S2
    assert CR3BP_System.DAY == SECONDS_PER_DAY
    assert CR3BP_System.YEAR == SECONDS_PER_JULIAN_YEAR
    assert EARTH.gm_by_datum["DE421"] == _DEFAULT_MU
    assert family_char_length == CHAR_LENGTH_KM
    assert _DEFAULT_FORCE_CONFIG["forces"][0]["params"]["mu"] == Datum.DE440.earth_gm

    relativity_cfg = perturbation_to_force_config({"relativity": 1})
    relativity = next(
        force for force in relativity_cfg["forces"] if force["type"] == "RelativisticCorrection"
    )
    assert relativity["params"]["c"] == SPEED_OF_LIGHT_KMS
