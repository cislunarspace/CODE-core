"""轨道库 catalog 的 Facade 接缝测试：自动入库、谱系、多维查询、标注、导出、派生清单。

只断外部行为（库中出现什么记录、查询返回什么、错误码是什么），不断
实现细节（SQLite 表结构、JSON 内部布局）。算法层按既有先例用 fake
结果（design/control）或小族真算（family/sweep）。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from e2m2e.api.config import Config
from e2m2e.api.facade import Facade, mcp_tools, tool_inventory
from e2m2e.api.models import (
    CatalogDeleteRequest,
    CatalogExportRequest,
    CatalogGetRequest,
    CatalogPromoteRequest,
    CatalogQueryRequest,
    CatalogSweepRequest,
    CatalogTagRequest,
    OrbitError,
)
from e2m2e.data.constants import Datum
from e2m2e.data.templates import ConvergenceState, FailureCause
from e2m2e.data.types.trajectory import EphemerisTable

pytestmark = pytest.mark.interface

CHAR_LENGTH_KM = 384400.0


def _make_ephemeris(n: int = 3) -> EphemerisTable:
    return EphemerisTable(
        year=np.full(n, 2024, dtype=int),
        month=np.full(n, 1, dtype=int),
        day=np.full(n, 1, dtype=int),
        hour=np.arange(n, dtype=int),
        minute=np.zeros(n, dtype=int),
        second=np.zeros(n, dtype=float),
        position_km=np.arange(n * 3, dtype=float).reshape(n, 3),
        velocity_mps=np.full((n, 3), 1000.0),
        synodic_position=np.full((n, 3), 0.5),
        times_jd_tdb=np.linspace(2460310.0, 2460311.0, n),
    )


def _make_design_result(
    *,
    orbit_type: str = "DRO",
    jacobi: float = 3.16,
    with_cr3bp: bool = True,
) -> SimpleNamespace:
    cr3bp_orbit = None
    if with_cr3bp:
        cr3bp_orbit = SimpleNamespace(
            states=np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.1, 0.1, 0.1]]),
            times=np.array([0.0, 1.234]),
            system=SimpleNamespace(mu=Datum.DE421.mu, characteristic_length=CHAR_LENGTH_KM),
        )
    return SimpleNamespace(
        orbit_type=orbit_type,
        epoch_utc="2024-01-01T00:00:00.000",
        duration_day=365.25,
        output_step_sec=3600.0,
        initial_state=np.zeros(6),
        ephemeris=_make_ephemeris(),
        cr3bp_orbit=cr3bp_orbit,
        cr3bp_jacobi=jacobi if with_cr3bp else float("nan"),
        correction=SimpleNamespace(iterations=4) if with_cr3bp else None,
        correction_method="two_level" if with_cr3bp else None,
        force_config={"sun_body": 1},
        status=ConvergenceState.CONVERGED,
        cause=FailureCause.NONE,
        message="任务完成",
        drift_e=None,
        drift_aop_deg=None,
        drift_rp_km=None,
        secular_aop_rate_deg_per_year=None,
    )


def _make_control_result() -> SimpleNamespace:
    return SimpleNamespace(
        num_failed=1,
        status=ConvergenceState.CONVERGED,
        cause=FailureCause.NONE,
        message="任务完成",
        sk_statistic=SimpleNamespace(rows=np.zeros((2, 3)), num_failed=1),
        maneuvers=SimpleNamespace(mjd_tdb=np.array([60000.0]), delta_v_mps=np.array([1.0, 2.0])),
        controlled_ephemeris=_make_ephemeris(n=2),
    )


def _fake_design(monkeypatch, result):
    import e2m2e.algorithm.design as design

    monkeypatch.setattr(design, "design_orbit", lambda *args, **kwargs: result)


def _fake_control(monkeypatch, captured: dict | None = None):
    import e2m2e.algorithm.station_keeping as station_keeping

    def fake_control(input_ephemeris, **kwargs):
        if captured is not None:
            captured["input_ephemeris"] = input_ephemeris
        return _make_control_result()

    monkeypatch.setattr(station_keeping, "control_orbit", fake_control)


class TestAutoIngest:
    def test_design_orbit_record_lands_in_catalog(self, monkeypatch):
        _fake_design(monkeypatch, _make_design_result(orbit_type="NRHO", jacobi=3.05))
        facade = Facade()
        response = facade.design_orbit(orbit_type="NRHO")

        assert response.record_id is not None
        summaries = facade.catalog_query().records
        assert len(summaries) == 1
        summary = summaries[0]
        assert summary.record_id == response.record_id
        assert summary.orbit_family == "nrho"
        assert summary.libration_point == 2
        assert summary.jacobi == [3.05, 3.05]
        assert summary.has_cr3bp is True
        assert summary.has_ephemeris is True
        assert summary.status is ConvergenceState.CONVERGED

    def test_design_record_keeps_both_segments_and_request_snapshot(self, monkeypatch):
        _fake_design(monkeypatch, _make_design_result())
        facade = Facade()
        response = facade.design_orbit(orbit_type="DRO")

        record = facade.catalog_get(record_id=response.record_id)
        assert "cr3bp/states" in record.arrays
        assert "eph/position_km" in record.arrays
        assert record.request["orbit_type"] == "DRO"
        assert record.scalars["correction_method"] == "two_level"
        assert record.status is ConvergenceState.CONVERGED
        assert record.cause is FailureCause.NONE
        # 主振幅 = 几何半极差最大值 × 特征长度（0.5 × 384400 km）
        assert record.amplitude == [pytest.approx(0.5 * CHAR_LENGTH_KM)] * 2
        table = record.to_ephemeris_table()
        assert table is not None and len(table) == 3
        assert record.to_orbit() is not None

    def test_family_generation_record_is_one_record_with_members(self):
        facade = Facade()
        response = facade.orbit_family_generation(
            orbit_type="HALO", libration_point=1, max_amplitude_km=3000.0, n_orbits=2
        )

        assert response.record_id is not None
        summaries = facade.catalog_query().records
        assert len(summaries) == 1
        assert summaries[0].orbit_family == "halo"
        assert summaries[0].libration_point == 1
        assert summaries[0].member_count == len(response.orbits)
        assert summaries[0].has_cr3bp is True
        assert summaries[0].has_ephemeris is False

        record = facade.catalog_get(record_id=response.record_id)
        assert len(record.members) == len(response.orbits)
        assert "cr3bp/members/0000/states" in record.arrays
        assert record.jacobi is not None

    def test_control_record_points_to_source_record(self, monkeypatch):
        _fake_design(monkeypatch, _make_design_result())
        captured: dict = {}
        _fake_control(monkeypatch, captured)
        facade = Facade()
        design_response = facade.design_orbit(orbit_type="DRO")

        control_response = facade.control_orbit(input_record_id=design_response.record_id)

        # 输入星历经库记录解析
        assert isinstance(captured["input_ephemeris"], EphemerisTable)
        assert control_response.record_id is not None
        record = facade.catalog_get(record_id=control_response.record_id)
        assert record.source_record_id == design_response.record_id
        assert record.source_tool == "control_orbit"
        # 站保产物只含星历段，并继承被控轨道的族分类
        assert record.has_cr3bp is False
        assert record.has_ephemeris is True
        assert record.orbit_family == "dro"
        assert record.to_orbit() is None
        assert record.to_ephemeris_table() is not None

    def test_control_rejects_both_input_sources(self):
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            Facade().control_orbit(input_ephemeris="x", input_record_id="y")

    def test_control_with_unknown_input_record_id(self):
        with pytest.raises(OrbitError, match="RECORD_NOT_FOUND"):
            Facade().control_orbit(input_record_id="no-such-record")

    def test_catalog_disabled_leaves_no_files(self, monkeypatch, tmp_path):
        _fake_design(monkeypatch, _make_design_result())
        catalog_dir = tmp_path / "catalog"
        facade = Facade(Config(catalog_dir=str(catalog_dir), catalog_enabled=False))

        response = facade.design_orbit(orbit_type="DRO")

        assert response.record_id is None
        assert not catalog_dir.exists()


class TestQuery:
    @pytest.fixture
    def facade_with_records(self, monkeypatch):
        facade = Facade()
        _fake_design(monkeypatch, _make_design_result(orbit_type="DRO", jacobi=3.16))
        facade.design_orbit(orbit_type="DRO")
        _fake_design(monkeypatch, _make_design_result(orbit_type="NRHO", jacobi=3.05))
        facade.design_orbit(orbit_type="NRHO")
        _fake_design(monkeypatch, _make_design_result(orbit_type="ELFO", with_cr3bp=False))
        facade.design_orbit(orbit_type="ELFO", semi_major_axis=7000.0)
        return facade

    def test_filter_by_orbit_family(self, facade_with_records):
        records = facade_with_records.catalog_query(orbit_family="nrho").records
        assert len(records) == 1
        assert records[0].orbit_family == "nrho"

    def test_filter_by_libration_point(self, facade_with_records):
        records = facade_with_records.catalog_query(libration_point=2).records
        assert len(records) == 1
        assert records[0].orbit_family == "nrho"

    def test_filter_by_jacobi_range(self, facade_with_records):
        assert len(facade_with_records.catalog_query(jacobi_min=3.1, jacobi_max=3.2).records) == 1
        assert len(facade_with_records.catalog_query(jacobi_min=3.0, jacobi_max=3.1).records) == 1
        assert len(facade_with_records.catalog_query(jacobi_min=2.0, jacobi_max=2.5).records) == 0

    def test_filter_by_segment_presence(self, facade_with_records):
        assert len(facade_with_records.catalog_query(has_ephemeris=True).records) == 3
        assert len(facade_with_records.catalog_query(has_cr3bp=True).records) == 2

    def test_combined_filter(self, facade_with_records):
        records = facade_with_records.catalog_query(
            orbit_family="nrho",
            libration_point=2,
            jacobi_min=3.0,
            jacobi_max=3.1,
            has_ephemeris=True,
        ).records
        assert len(records) == 1

    def test_invalid_range_is_rejected(self, facade_with_records):
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            facade_with_records.catalog_query(jacobi_min=3.2, jacobi_max=3.0)

    def test_summaries_carry_no_arrays(self, facade_with_records):
        records = facade_with_records.catalog_query().records
        assert len(records) == 3
        assert all(not hasattr(record, "arrays") for record in records)


class TestGetDelete:
    def test_get_unknown_record_id_raises_structured_error(self):
        with pytest.raises(OrbitError) as exc_info:
            Facade().catalog_get(record_id="no-such-record")
        assert exc_info.value.code == "RECORD_NOT_FOUND"
        assert exc_info.value.status is ConvergenceState.FAILED

    def test_delete_removes_record(self, monkeypatch):
        _fake_design(monkeypatch, _make_design_result())
        facade = Facade()
        record_id = facade.design_orbit(orbit_type="DRO").record_id

        response = facade.catalog_delete(record_id=record_id)

        assert response.deleted is True
        assert facade.catalog_query().records == []
        with pytest.raises(OrbitError, match="RECORD_NOT_FOUND"):
            facade.catalog_get(record_id=record_id)


class TestTagExport:
    def test_tag_is_visible_in_record(self, monkeypatch):
        _fake_design(monkeypatch, _make_design_result())
        facade = Facade()
        record_id = facade.design_orbit(orbit_type="DRO").record_id

        response = facade.catalog_tag(record_id=record_id, tags=["期中案例"], note="注意近月点高度")

        assert response.record.tags == ["期中案例"]
        record = facade.catalog_get(record_id=record_id)
        assert record.tags == ["期中案例"]
        assert record.note == "注意近月点高度"

    def test_export_package_carries_annotation(self, monkeypatch, tmp_path):
        _fake_design(monkeypatch, _make_design_result())
        facade = Facade()
        record_id = facade.design_orbit(orbit_type="DRO").record_id
        facade.catalog_tag(record_id=record_id, tags=["教学"])
        facade.design_orbit(orbit_type="DRO")
        dest = tmp_path / "案例包"

        response = facade.catalog_export(tags=["教学"], dest=str(dest))

        assert response.exported_count == 1
        assert response.record_ids == [record_id]
        exported = facade.catalog_get(record_id=record_id)
        assert exported.tags == ["教学"]
        assert (dest / "records" / f"{record_id}.json").exists()
        assert (dest / "manifest.json").exists()


class TestPromote:
    def test_promoted_member_points_to_family(self):
        facade = Facade()
        family_response = facade.orbit_family_generation(
            orbit_type="HALO", libration_point=1, max_amplitude_km=3000.0, n_orbits=2
        )

        promoted = facade.catalog_promote(record_id=family_response.record_id, member_index=1)

        record = promoted.record
        assert record.source_record_id == family_response.record_id
        assert record.source_tool == "catalog_promote"
        assert record.orbit_family == "halo"
        assert record.has_cr3bp is True
        assert "cr3bp/states" in record.arrays
        assert len(facade.catalog_query().records) == 2

    def test_promote_bad_member_index_raises_structured_error(self):
        facade = Facade()
        family_response = facade.orbit_family_generation(
            orbit_type="HALO", libration_point=1, max_amplitude_km=3000.0, n_orbits=2
        )
        with pytest.raises(OrbitError, match="RECORD_NOT_FOUND"):
            facade.catalog_promote(record_id=family_response.record_id, member_index=99)


class TestSweep:
    def test_sweep_generates_records_for_grid(self):
        facade = Facade()
        response = facade.catalog_sweep(
            orbit_types=["HALO"],
            libration_points=[1],
            max_amplitudes_km=[2000.0, 3000.0],
            n_orbits=1,
        )

        assert response.succeeded == 2
        assert response.failed == 0
        assert len(response.record_ids) == 2
        assert all(point.record_id is not None for point in response.points)
        assert {point.parameter_km for point in response.points} == {2000.0, 3000.0}
        records = facade.catalog_query(orbit_family="halo").records
        assert len(records) == 2

    def test_sweep_rejects_lissajous_with_one_dimensional_grid(self):
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            Facade().catalog_sweep(orbit_types=["LISSAJOUS"], max_amplitudes_km=[5000.0])
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            Facade().catalog_sweep(orbit_types=["LISSAJOUS"], jacobi_windows=[[3.17, 3.18]])

    def test_sweep_rejects_mutually_exclusive_grid_dimensions(self):
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            Facade().catalog_sweep(
                orbit_types=["HALO"],
                max_amplitudes_km=[2000.0],
                jacobi_windows=[[3.17, 3.18]],
            )
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            Facade().catalog_sweep(
                orbit_types=["HALO"],
                max_amplitudes_km=[2000.0],
                amplitude_ins_km=[1000.0],
                amplitude_outs_km=[3000.0],
            )

    def test_sweep_rejects_invalid_jacobi_windows(self):
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            Facade().catalog_sweep(orbit_types=["HALO"], jacobi_windows=[[3.18, 3.17]])
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            Facade().catalog_sweep(orbit_types=["HALO"], jacobi_windows=[[3.17, 3.18, 3.19]])
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            Facade().catalog_sweep(orbit_types=["HALO"], jacobi_windows=[[3.17]])

    def test_sweep_lissajous_grid_requires_both_amplitude_lists(self):
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            Facade().catalog_sweep(orbit_types=["LISSAJOUS"], amplitude_ins_km=[1000.0])

    def test_sweep_lissajous_grid_creates_records(self):
        facade = Facade()
        response = facade.catalog_sweep(
            orbit_types=["LISSAJOUS"],
            libration_points=[2],
            amplitude_ins_km=[1000.0, 2000.0],
            amplitude_outs_km=[3000.0],
            n_orbits=2,
        )

        assert response.succeeded == 2
        assert response.failed == 0
        assert len(response.record_ids) == 2
        assert {tuple(point.amplitudes_km) for point in response.points} == {
            (1000.0, 3000.0),
            (2000.0, 3000.0),
        }
        assert all(
            point.parameter_km is None and point.jacobi_window is None for point in response.points
        )
        records = facade.catalog_query(orbit_family="lissajous", libration_point=2).records
        assert len(records) == 2
        assert all(record.member_count == 2 for record in records)

    def test_sweep_jacobi_windows_create_layered_records(self):
        facade = Facade()
        # 自校准窗口：先探出族的真实能量范围，窗口取其内部分层
        probe = facade.orbit_family_generation(
            orbit_type="HALO", libration_point=1, max_amplitude_km=3000.0, n_orbits=5
        )
        jacobis = probe.get_jacobi_constants()
        lower, upper = float(jacobis.min()), float(jacobis.max())
        middle = 0.5 * (lower + upper)

        response = facade.catalog_sweep(
            orbit_types=["HALO"],
            libration_points=[1],
            jacobi_windows=[[lower, middle], [middle, upper], [9.9, 9.95]],
            n_orbits=5,
        )

        assert response.succeeded == 2
        assert response.failed == 0
        assert "1 点软失败无成员产出" in response.message
        windows = {tuple(point.jacobi_window) for point in response.points}
        assert windows == {(lower, middle), (middle, upper), (9.9, 9.95)}
        assert all(
            point.parameter_km is None and point.amplitudes_km is None for point in response.points
        )

        for point in response.points[:2]:
            assert point.status is ConvergenceState.CONVERGED
            assert point.record_id is not None
            record = facade.catalog_get(record_id=point.record_id)
            window_lo, window_hi = point.jacobi_window
            assert record.jacobi is not None
            assert record.jacobi[0] >= window_lo
            assert record.jacobi[1] <= window_hi

        empty = response.points[2]
        assert empty.status is ConvergenceState.INFEASIBLE
        assert empty.cause is FailureCause.CONSTRAINT_VIOLATION
        assert empty.record_id is None
        assert empty.generated_members == 0

    def test_sweep_request_model_declares_new_dimensions(self):
        field_names = set(CatalogSweepRequest.model_fields)
        assert {
            "jacobi_windows",
            "amplitude_ins_km",
            "amplitude_outs_km",
        } <= field_names
        # 条件取值域公开且同源（ADR 0014 决策 8）：族 × 可用维度经公开接口给出
        assert CatalogSweepRequest.supported_grid_dimensions("LISSAJOUS") == (
            "amplitude_ins_km",
            "amplitude_outs_km",
        )
        for family in ("HALO", "NRHO", "AXIAL", "SPO", "LPO", "HORSESHOE"):
            dimensions = CatalogSweepRequest.supported_grid_dimensions(family)
            assert "jacobi_windows" in dimensions
            assert "amplitude_ins_km" not in dimensions
        with pytest.raises(ValueError, match="不支持的 orbit_type"):
            CatalogSweepRequest.supported_grid_dimensions("DRO")
        # 新维度已由派生工具清单引用的请求模型携带（CLI/MCP 纯派生，ADR 0014）
        by_name = {tool.name: tool for tool in tool_inventory(Facade())}
        assert by_name["catalog_sweep"].request_model is CatalogSweepRequest

    def test_sweep_requires_grid_for_family(self):
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            Facade().catalog_sweep(orbit_types=["NRHO"], max_amplitudes_km=[5000.0])

    def test_sweep_counts_records_and_failures(self, monkeypatch):
        import e2m2e.algorithm.catalog_sweep as sweep_module
        from e2m2e.algorithm.catalog_sweep import FamilySweepPoint, SweepPointResult
        from e2m2e.algorithm.results import FamilyGenerationResult
        from e2m2e.data.types.orbit import Orbit, OrbitFamily

        def make_point() -> FamilySweepPoint:
            return FamilySweepPoint(orbit_type="HALO", libration_point=1, n_orbits=1, kwargs={})

        orbit = Orbit(states=np.zeros((2, 6)), times=np.array([0.0, 1.0]))
        family = OrbitFamily(
            orbits=[orbit],
            family_type="halo",
            system=SimpleNamespace(characteristic_length=384400.0),
        )
        good = FamilyGenerationResult(
            status=ConvergenceState.CONVERGED,
            cause=FailureCause.NONE,
            message="完成",
            family=family,
            requested_members=1,
            generated_members=1,
        )
        soft_empty = FamilyGenerationResult(
            status=ConvergenceState.STAGNATED,
            cause=FailureCause.STAGNATION_DETECTED,
            message="停滞",
            family=OrbitFamily(orbits=[], family_type="halo"),
            requested_members=5,
            generated_members=0,
        )
        outcomes = [
            SweepPointResult(
                point=make_point(),
                status=good.status,
                cause=good.cause,
                message=good.message,
                result=good,
            ),
            SweepPointResult(
                point=make_point(),
                status=ConvergenceState.FAILED,
                cause=FailureCause.UNKNOWN,
                message="爆炸",
                result=None,
            ),
            SweepPointResult(
                point=make_point(),
                status=soft_empty.status,
                cause=soft_empty.cause,
                message=soft_empty.message,
                result=soft_empty,
            ),
        ]
        monkeypatch.setattr(sweep_module, "run_family_sweep", lambda points: outcomes)

        response = Facade().catalog_sweep(
            orbit_types=["HALO"],
            libration_points=[1],
            max_amplitudes_km=[2000.0, 3000.0, 4000.0],
            n_orbits=1,
        )

        assert response.succeeded == 1
        assert response.failed == 1
        assert len(response.record_ids) == 1
        assert "1 点软失败无成员产出" in response.message
        assert response.points[1].record_id is None
        assert response.points[1].message == "爆炸"
        assert response.points[2].status is ConvergenceState.STAGNATED


class TestCatalogErrors:
    def test_corrupted_index_translates_to_structured_error(self, tmp_path):
        catalog_dir = tmp_path / "catalog"
        (catalog_dir / "records").mkdir(parents=True)
        (catalog_dir / "catalog.db").write_bytes(b"not a sqlite database")
        facade = Facade(Config(catalog_dir=str(catalog_dir)))
        with pytest.raises(OrbitError, match="CATALOG_READ_FAILED"):
            facade.catalog_query()

    def test_path_traversal_record_id_is_record_not_found(self):
        """record_id 拼路径前的形态校验：路径穿越一律 RECORD_NOT_FOUND。"""
        facade = Facade()
        with pytest.raises(OrbitError, match="RECORD_NOT_FOUND"):
            facade.catalog_get(record_id="../../etc/passwd")
        with pytest.raises(OrbitError, match="RECORD_NOT_FOUND"):
            facade.catalog_delete(record_id="../../etc/passwd")


class TestToolInventory:
    def test_catalog_methods_are_in_derived_inventory(self):
        facade = Facade()
        names = set(mcp_tools(facade))
        expected = {
            "catalog_query",
            "catalog_get",
            "catalog_delete",
            "catalog_tag",
            "catalog_promote",
            "catalog_export",
            "catalog_sweep",
        }
        assert expected <= names

        by_name = {tool.name: tool for tool in tool_inventory(facade)}
        request_models = {
            "catalog_query": CatalogQueryRequest,
            "catalog_get": CatalogGetRequest,
            "catalog_delete": CatalogDeleteRequest,
            "catalog_tag": CatalogTagRequest,
            "catalog_promote": CatalogPromoteRequest,
            "catalog_export": CatalogExportRequest,
            "catalog_sweep": CatalogSweepRequest,
        }
        for name, model in request_models.items():
            assert by_name[name].status == "implemented"
            assert by_name[name].request_model is model
