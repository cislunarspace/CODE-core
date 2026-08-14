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


class TestEphemerisFieldContract:
    """EphemerisTable 字段形状契约（从 design/scenarios e2e 测试下沉）。

    ADR 0021：字段形状契约归 data 类，不靠真 design_orbit 传播验证。
    覆盖原 test_lissajous.py / test_triangular.py 的 output_shape /
    epoch_matches_input 断言。
    """

    def test_position_velocity_synodic_shapes(self):
        """position_km / velocity_mps / synodic_position 形状均为 (n, 3)。"""
        n = 5
        t = EphemerisTable(
            year=np.full(n, 2025, dtype=int),
            month=np.full(n, 1, dtype=int),
            day=np.full(n, 1, dtype=int),
            hour=np.arange(n, dtype=int),
            minute=np.zeros(n, dtype=int),
            second=np.zeros(n, dtype=float),
            position_km=np.arange(n * 3, dtype=float).reshape(n, 3),
            velocity_mps=np.full((n, 3), 1000.0),
            synodic_position=np.full((n, 3), 0.5),
        )
        assert len(t) == n
        assert t.position_km.shape == (n, 3)
        assert t.velocity_mps.shape == (n, 3)
        assert t.synodic_position.shape == (n, 3)

    def test_epoch_first_row_indexable(self):
        """起始历元字段（year/month/day）可索引访问，反映输入 epoch。"""
        t = _table()
        assert t.year[0] == 2024
        assert t.month[0] == 1
        assert t.day[0] == 1
