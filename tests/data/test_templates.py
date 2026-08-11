"""data/templates/：常量、种子、枚举、schema 数据测试。"""

import pytest

from e2m2e.data.constants import (
    AU_KM,
    EARTH_MOON_DISTANCE_KM,
    KM_TO_M,
    SUN,
    Datum,
)
from e2m2e.data.constants import (
    GRAVITATIONAL_CONSTANT as G,
)
from e2m2e.data.constants import (
    SECONDS_PER_DAY as DAY,
)
from e2m2e.data.constants import (
    SECONDS_PER_JULIAN_YEAR as YEAR,
)
from e2m2e.data.constants.bodies import EARTH
from e2m2e.data.templates import (
    CHAR_LENGTH_KM,
    CHAR_PERIOD_SEC,
    DEFAULT_DYB,
    DEFAULT_PERTURBATION,
    EARTH_MOON_MU,
    MOON_RADIUS_KM,
    OrbitFamilyType,
    ReferenceFrame,
    force_config,
    seed,
)

pytestmark = pytest.mark.data

R_EARTH = EARTH.gravity_ref_radius_km


class TestSystemsConstants:
    def test_physical_constants(self):
        assert R_EARTH == 6378.1363
        assert AU_KM == 149597870.7
        assert KM_TO_M == 1000.0

    def test_cr3bp_standard(self):
        assert EARTH_MOON_DISTANCE_KM == 384405.0
        assert G == 6.67430e-20
        assert DAY == 86400
        assert YEAR == 365.25 * 86400

    def test_algorithm_layer_single_source(self):
        """算法层不再硬编码物理常量：常量子来自 data.constants 单一来源。"""
        from e2m2e.algorithm.dynamics.bcr4bp_system import BCR4BPSystem
        from e2m2e.algorithm.dynamics.cr3bp_system import CR3BP_System
        from e2m2e.algorithm.forces.force_mapping import perturbation_to_force_config
        from e2m2e.algorithm.forces.gravity_file import _DEFAULT_MU
        from e2m2e.algorithm.propagation import _DEFAULT_FORCE_CONFIG
        from e2m2e.data.constants import SPEED_OF_LIGHT_KMS

        assert SUN.gm_by_datum["DE440"] == BCR4BPSystem.SUN_GM_KM3_S2
        assert CR3BP_System.DAY == DAY
        assert CR3BP_System.YEAR == YEAR
        assert EARTH.gm_by_datum["DE421"] == _DEFAULT_MU
        assert _DEFAULT_FORCE_CONFIG["forces"][0]["params"]["mu"] == Datum.DE440.earth_gm
        relativity_cfg = perturbation_to_force_config({"relativity": 1})
        relativity = next(
            f for f in relativity_cfg["forces"] if f["type"] == "RelativisticCorrection"
        )
        assert relativity["params"]["c"] == SPEED_OF_LIGHT_KMS


class TestSeedConstants:
    def test_seed_values(self):
        assert Datum.DE421.mu == EARTH_MOON_MU
        assert Datum.DE421.char_length_km == CHAR_LENGTH_KM
        assert 2 * 3.141592653589793 * Datum.DE421.char_time_s == CHAR_PERIOD_SEC
        assert MOON_RADIUS_KM == 1737.4
        assert seed._DRO_SEED_X0 == 0.79188556619742
        assert seed._HALO_SEED_Z0 == 0.001
        assert seed._HALO_FOLD_Z0 == {1: 0.07, 2: 0.15}

    def test_family_uses_data_layer_constants(self):
        """algorithm/family/cr3bp_orbits.py 的常量来自数据层（单一来源）。"""
        from e2m2e.algorithm.family.cr3bp_orbits import CHAR_LENGTH_KM as family_char_length

        assert family_char_length == CHAR_LENGTH_KM


class TestPerturbationDefaults:
    def test_default_perturbation(self):
        assert DEFAULT_PERTURBATION["solar_radiation"] == 2
        assert DEFAULT_PERTURBATION["coupling"] == 1

    def test_default_dyb(self):
        assert DEFAULT_DYB[0] == 0.01
        assert len(DEFAULT_DYB) == 9

    def test_io_uses_data_layer_defaults(self):
        from e2m2e.data.templates.perturbations import DEFAULT_DYB as io_dyb
        from e2m2e.data.templates.perturbations import DEFAULT_PERTURBATION as io_perturb

        assert io_dyb == DEFAULT_DYB
        assert io_perturb == DEFAULT_PERTURBATION


class TestEnums:
    def test_reference_frame_values(self):
        assert ReferenceFrame.SYNODIC.value == "synodic"
        assert ReferenceFrame.J2000.value == "J2000"

    def test_orbit_family_type(self):
        assert OrbitFamilyType.DRO.value == "dro"
        assert OrbitFamilyType.NRHO.value == "nrho"

    def test_old_paths_shim(self):
        from e2m2e.data.templates.enums import ReferenceFrame as CoreReferenceFrame
        from e2m2e.mbse.data.enums import OrbitFamilyType as MbseOrbitFamilyType

        assert CoreReferenceFrame is ReferenceFrame
        assert MbseOrbitFamilyType is OrbitFamilyType


class TestForceConfigSchema:
    def test_schema_shape(self):
        assert force_config.CONFIG_VERSION == 1
        cfg = force_config.ForceConfig(
            version=1,
            forces=[
                force_config.ForceEntry(
                    name="moon", type="PointMassGravity", enabled=True, params={}
                )
            ],
        )
        assert cfg["forces"][0]["type"] == "PointMassGravity"
