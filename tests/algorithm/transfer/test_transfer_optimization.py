"""Transfer 创建与配置契约测试。

测试策略：只保留 Transfer 实例化、轨道设置与配置字段契约。
optimize() 全链 NLP 测试（rtol=1e-12 研究级容差）已按维护决策移除
（全量预算超 ADR 0037 上限）。
"""

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.orchestration


class TestTransferCreation:
    """Test Transfer class instantiation and configuration."""

    def test_transfer_creation_with_dynamics(self, dynamics):
        """Transfer should be created with a dynamics instance."""
        from e2m2e.algorithm.transfer import Transfer

        transfer = Transfer(dynamics)

        assert transfer.dynamics is not None
        assert transfer.departure_orbit is None
        assert transfer.arrival_orbit is None
        assert transfer.result is None

    def test_transfer_set_orbits(self, dynamics, dro_orbit, ro_orbit):
        """Transfer.set_orbit() should accept start (departure) and end (arrival) orbits."""
        from e2m2e.algorithm.transfer import Transfer

        transfer = Transfer(dynamics)
        transfer.set_orbit(start=dro_orbit, end=ro_orbit)

        assert transfer.departure_orbit is dro_orbit
        assert transfer.arrival_orbit is ro_orbit

    def test_transfer_config_fields(self, dynamics):
        """Transfer.config should expose optimization configuration."""
        from e2m2e.algorithm.transfer import Transfer, TransferConfig

        transfer = Transfer(dynamics)

        assert hasattr(transfer, "config")
        assert isinstance(transfer.config, TransferConfig)
        assert transfer.config.nlp_alpha_min == 0.5
        assert transfer.config.nlp_alpha_max == 2.5


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def project_root():
    return Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def dro_file(project_root):
    return project_root / "output/dro/dro_31_3857117441.json"


@pytest.fixture
def ro_file(project_root):
    return project_root / "output/ro/ro_31_3857122799.json"


@pytest.fixture
def dro_orbit(dro_file):
    pytest.importorskip("_dro_data", reason="DRO data file not available")
    if not dro_file.exists():
        pytest.skip("DRO orbit data file not found")
    from e2m2e.algorithm.transfer import load_orbit_from_json

    return load_orbit_from_json(str(dro_file))


@pytest.fixture
def ro_orbit(ro_file):
    if not ro_file.exists():
        pytest.skip("RO orbit data file not found")
    from e2m2e.algorithm.transfer import load_orbit_from_json

    with open(ro_file, encoding="utf-8") as f:
        ro_json = json.load(f)

    orbit = load_orbit_from_json(str(ro_file))
    if "properties" in ro_json and "period" in ro_json["properties"]:
        orbit.period = float(ro_json["properties"]["period"])
    return orbit


@pytest.fixture
def dynamics():
    from e2m2e.algorithm.dynamics import CR3BP_Dynamics, CR3BP_System
    from e2m2e.data.constants import Datum

    system = CR3BP_System(mu=Datum.DE421.mu, primary="earth", secondary="moon")
    dyn = CR3BP_Dynamics(system=system)
    dyn.integrator = "DOP853"
    dyn.rtol = 1e-12
    dyn.atol = 1e-12
    dyn.max_step = 1.0 / (24.0 * 384405.0 / 26970.0 * 2.0 * np.pi / 27.321661)
    return dyn
