"""data/kernels/：SPICEManager 与 EphemerisProvider 抽象测试。"""

import os
from pathlib import Path

import numpy as np
import pytest

from e2m2e.data.kernels.manager import SPICEManager
from e2m2e.data.kernels.provider import EphemerisProvider

pytestmark = [pytest.mark.spice, pytest.mark.l3]


def _kernel_path() -> str | None:
    """返回可用星历内核路径（与 tests/conftest 同优先级）。"""
    search_dir = os.environ.get(
        "SPICE_KERNEL_DIR",
        str(Path(__file__).resolve().parents[2] / "kernels"),
    )
    for name in ("de440.bsp", "de440s.bsp", "de438.bsp", "de435.bsp"):
        path = Path(search_dir) / name
        if path.is_file():
            return str(path)
    return None


@pytest.fixture
def loaded_spice():
    path = _kernel_path()
    if path is None:
        pytest.skip("DE440/DE438/DE435 SPICE kernel not found")
    mgr = SPICEManager()
    mgr.load_kernel(path)
    yield mgr
    mgr.unload_kernel(path)


class TestSPICEManager:
    def test_implements_provider_interface(self):
        assert issubclass(SPICEManager, EphemerisProvider)

    def test_utc_to_tdb_and_roundtrip(self, loaded_spice):
        et = loaded_spice.utc_to_tdb("2025-06-21T11:00:00")
        assert isinstance(et, float)
        assert loaded_spice.et_to_utc(et) == "2025-06-21T11:00:00"

    def test_body_position_state_shapes(self, loaded_spice):
        et = loaded_spice.utc_to_tdb("2025-06-21T11:00:00")
        pos = loaded_spice.body_position("MOON", et)
        state = loaded_spice.body_state("MOON", et)
        assert pos.shape == (3,)
        assert state.shape == (6,)
        assert np.linalg.norm(pos) > 350000

    def test_pxform_identity(self, loaded_spice):
        et = loaded_spice.utc_to_tdb("2025-06-21T11:00:00")
        rot = loaded_spice.pxform("J2000", "J2000", et)
        np.testing.assert_allclose(rot, np.eye(3), atol=1e-12)

    def test_find_kernel_priority(self, tmp_path):
        (tmp_path / "de440.bsp").write_bytes(b"fake")
        (tmp_path / "de438.bsp").write_bytes(b"fake")
        assert SPICEManager().find_ephemeris_kernel(str(tmp_path)).endswith("de440.bsp")


class TestEphemerisProviderAbstract:
    def test_unimplemented_methods_raise(self):
        provider = EphemerisProvider()
        for call in (
            lambda: provider.utc_to_tai("2025-01-01T00:00:00"),
            lambda: provider.body_rotation("MOON", 0.0, "J2000", "MOON_PA"),
        ):
            with pytest.raises(NotImplementedError):
                call()

    def test_interface_documented(self):
        """接口 docstring 明确三类方法（时间/状态/帧）。"""
        doc = EphemerisProvider.__doc__ or ""
        assert "时间" in doc and "状态" in doc and "帧" in doc
