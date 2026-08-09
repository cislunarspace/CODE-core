"""data/types/trajectory.py：EphemerisTable/NominalOrbit 容器测试。"""

import numpy as np
import pytest

from e2m2e.data.types.trajectory import EphemerisTable, NominalOrbit

pytestmark = pytest.mark.data


def _table() -> EphemerisTable:
    return EphemerisTable(
        year=np.array([2024, 2024]),
        month=np.array([1, 1]),
        day=np.array([1, 1]),
        hour=np.array([0, 1]),
        minute=np.array([0, 0]),
        second=np.array([0.0, 0.0]),
        position_km=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        velocity_mps=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        synodic_position=np.array([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]]),
    )


class TestEphemerisTable:
    def test_len_and_columns(self):
        t = _table()
        assert len(t) == 2
        assert t.position_km.shape == (2, 3)
        assert t.velocity_mps.shape == (2, 3)
        assert t.synodic_position.shape == (2, 3)

    def test_raw_text_default_empty(self):
        assert _table().raw_text == ""

    def test_roundtrip_via_io_functions(self, tmp_path):
        """与 io/ 临时脚本的读写函数互通。"""
        from e2m2e.data.types.trajectory import read_ephemeris, write_ephemeris

        t = _table()
        out = tmp_path / "EPHEMERIDES_OUT.TXT"
        write_ephemeris(t, out)
        dst = read_ephemeris(out)
        assert len(dst) == len(t)
        np.testing.assert_allclose(dst.position_km, t.position_km, atol=1e-12)


class TestNominalOrbit:
    def test_state_at_without_interpolator_raises(self):
        orbit = NominalOrbit(
            epochs=np.array([0.0, 1.0]),
            states=np.zeros((2, 6)),
        )
        with pytest.raises(NotImplementedError, match="插值器"):
            orbit.state_at(0.5)
