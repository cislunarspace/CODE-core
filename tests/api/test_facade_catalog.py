"""轨道库 catalog 的 Facade 接缝测试：自动入库、谱系、多维查询、标注、导出、派生清单。

只断外部行为（库中出现什么记录、查询返回什么、错误码是什么），不断
实现细节（SQLite 表结构、JSON 内部布局）。算法层按既有先例用 fake
结果（design/control）或小族真算（family/sweep）。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from e2m2e.api.config import Config
from e2m2e.api.facade import Facade, mcp_tools, tool_inventory
from e2m2e.api.models import (
    CatalogDeleteRequest,
    CatalogExportRequest,
    CatalogGetRequest,
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


def _fake_transfer(monkeypatch, result):
    import e2m2e.algorithm.transfer as transfer

    monkeypatch.setattr(transfer, "transfer_orbit", lambda *args, **kwargs: result)


@dataclass
class _FakeImpulsiveDetails:
    """脉冲转移 details 形状（HMN/LGA/WSB dataclass 的 tof_sec 字段）。"""

    tli_epoch: float
    tof_sec: float


@dataclass
class _FakeLowThrustDetails:
    """low_thrust details 形状：无 tof_sec 字段。"""

    equivalent_delta_v: float


def _make_transfer_result(
    *,
    transfer_type: str = "WSB",
    delta_v: float = 3.9,
    with_trajectory: bool = True,
    with_gcrs: bool = True,
    with_candidates: bool = False,
) -> SimpleNamespace:
    from e2m2e.algorithm.transfer import ManeuverEvent, TransferCandidate

    candidates = None
    if with_candidates:
        # top-N 候选（#583）：选中解与顶层同口径，另带一个网格估计候选——
        # 两者 Δv 均不同于顶层值，入库守护据此区分“只 ingest 选中解”
        candidates = (
            TransferCandidate(
                delta_v_km_s=delta_v,
                tli_epoch=2460800.5,
                tof_sec=100.0,
                trajectory=np.arange(12, dtype=float).reshape(2, 6),
                trajectory_times=np.array([0.0, 100.0]),
                state_frame="synodic_barycentric_km",
                selected=True,
                refined=True,
            ),
            TransferCandidate(
                delta_v_km_s=9.9,
                tli_epoch=2460800.5,
                tof_sec=200.0,
                trajectory=np.arange(6, dtype=float).reshape(1, 6),
                trajectory_times=np.array([0.0]),
                state_frame="synodic_barycentric_km",
                selected=False,
                refined=False,
            ),
        )

    return SimpleNamespace(
        status=ConvergenceState.CONVERGED,
        cause=FailureCause.NONE,
        message="任务完成",
        transfer_type=transfer_type,
        delta_v=delta_v,
        trajectory=np.arange(12, dtype=float).reshape(2, 6) if with_trajectory else None,
        trajectory_times=np.array([0.0, 100.0]) if with_trajectory else None,
        trajectory_gcrs_km=(
            np.arange(12, dtype=float).reshape(2, 6) + 100.0
            if with_trajectory and with_gcrs
            else None
        ),
        state_frame="synodic_barycentric_km",
        maneuver_events=(
            ManeuverEvent(kind="departure", t_sec=0.0, dv_km_s=3.5),
            ManeuverEvent(kind="perilune", t_sec=50.0, dv_km_s=0.0),
            ManeuverEvent(kind="arrival", t_sec=100.0, dv_km_s=0.4),
        ),
        details=_FakeImpulsiveDetails(tli_epoch=2460800.5, tof_sec=100.0),
        candidates=candidates if candidates is not None else (),
    )


class TestTransferRecord:
    """transfer_design 产物入库（#574）：transfer record type。"""

    def test_transfer_record_lands_in_catalog(self, monkeypatch, tmp_path):
        _fake_transfer(monkeypatch, _make_transfer_result())
        facade = Facade(Config(catalog_dir=str(tmp_path / "catalog")))

        response = facade.transfer_design(
            transfer_type="WSB", tli_epoch=2460800.5, target_ephemeris=[[1.0] * 6]
        )

        assert response.record_id is not None
        record = facade.catalog.catalog_get(record_id=response.record_id)
        assert record.source_tool == "transfer_design"
        # 元数据：transfer 专属标量走 scalars，SCHEMA_VERSION 不动
        assert record.scalars["transfer_type"] == "WSB"
        assert record.scalars["delta_v_km_s"] == pytest.approx(3.9)
        assert record.scalars["tli_epoch"] == pytest.approx(2460800.5)
        assert record.scalars["tof_sec"] == pytest.approx(100.0)
        assert record.scalars["state_frame"] == "synodic_barycentric_km"
        # 轨道分类 6 键置 None/False（validate_meta 允许）
        assert record.orbit_family is None
        assert record.libration_point is None
        assert record.jacobi is None
        assert record.amplitude is None
        assert record.has_cr3bp is False
        assert record.has_ephemeris is False
        assert record.family_id is None  # 受控产物非族成员（ADR 0045）
        # 二进制段：trajectory (n,6) + trajectory_times (n,) 行数一致
        states = record.arrays["transfer/states"]
        times = record.arrays["transfer/times"]
        assert states.shape == (2, 6)
        assert times.shape == (2,)
        assert np.allclose(times, [0.0, 100.0])
        # details 原样存 + 结构化机动事件随块入 record（catalog_get 露出）
        assert record.details["tof_sec"] == pytest.approx(100.0)
        assert record.details["tli_epoch"] == pytest.approx(2460800.5)
        assert [e["kind"] for e in record.details["maneuver_events"]] == [
            "departure",
            "perilune",
            "arrival",
        ]
        assert record.details["maneuver_events"][1] == {
            "kind": "perilune",
            "t_sec": 50.0,
            "dv_km_s": 0.0,
            "note": None,
        }
        # request 快照可追溯
        assert record.request["transfer_type"] == "WSB"

    def test_transfer_record_carries_gcrs_segment(self, monkeypatch, tmp_path):
        """惯性段入 transfer/ 段（#584）：gcrs 键自带数据系标注，时刻共用不双份。"""
        _fake_transfer(monkeypatch, _make_transfer_result(with_gcrs=True))
        facade = Facade(Config(catalog_dir=str(tmp_path / "catalog")))

        response = facade.transfer_design(
            transfer_type="WSB", tli_epoch=2460800.5, target_ephemeris=[[1.0] * 6]
        )

        assert response.record_id is not None
        record = facade.catalog.catalog_get(record_id=response.record_id)
        # 惯性段：键名内嵌词汇值 gcrs_km（该段的 frame 标注）
        gcrs = record.arrays["transfer/states_gcrs_km"]
        assert gcrs.shape == (2, 6)
        assert np.allclose(gcrs, np.arange(12, dtype=float).reshape(2, 6) + 100.0)
        # 时刻数组共用：全段唯一 times 键，与两份几何同行对齐
        times_keys = {key for key in record.arrays if key.endswith("/times")}
        assert times_keys == {"transfer/times"}
        # 主几何的 state_frame 标注不受并行段影响（仍指 synodic 主段）
        assert record.scalars["state_frame"] == "synodic_barycentric_km"

    def test_transfer_record_without_gcrs_omits_segment(self, monkeypatch, tmp_path):
        """无惯性段（low_thrust/零结果）不落 gcrs 键，旧消费方零感知。"""
        _fake_transfer(monkeypatch, _make_transfer_result(with_gcrs=False))
        facade = Facade(Config(catalog_dir=str(tmp_path / "catalog")))

        response = facade.transfer_design(
            transfer_type="WSB", tli_epoch=2460800.5, target_ephemeris=[[1.0] * 6]
        )

        record = facade.catalog.catalog_get(record_id=response.record_id)
        assert "transfer/states_gcrs_km" not in record.arrays
        assert set(record.arrays) == {"transfer/states", "transfer/times"}

    def test_transfer_without_trajectory_makes_no_record(self, monkeypatch, tmp_path):
        _fake_transfer(monkeypatch, _make_transfer_result(with_trajectory=False))
        facade = Facade(Config(catalog_dir=str(tmp_path / "catalog")))

        response = facade.transfer_design(
            transfer_type="WSB", tli_epoch=2460800.5, target_ephemeris=[[1.0] * 6]
        )

        assert response.record_id is None
        summaries = facade.catalog.catalog_query().records
        assert all(s.source_tool != "transfer_design" for s in summaries)

    def test_catalog_disabled_transfer_record(self, monkeypatch, tmp_path):
        _fake_transfer(monkeypatch, _make_transfer_result())
        catalog_dir = tmp_path / "catalog"
        facade = Facade(Config(catalog_dir=str(catalog_dir), catalog_enabled=False))

        response = facade.transfer_design(
            transfer_type="WSB", tli_epoch=2460800.5, target_ephemeris=[[1.0] * 6]
        )

        assert response.record_id is None
        assert not catalog_dir.exists()

    def test_low_thrust_record_with_null_tof(self, monkeypatch, tmp_path):
        """low_thrust details 无 tof_sec：tof 标量为 None，记录照常落库。"""
        from e2m2e.algorithm.transfer import ManeuverEvent

        result = _make_transfer_result(transfer_type="low_thrust", with_gcrs=False)
        result.state_frame = "force_model_state"
        result.details = _FakeLowThrustDetails(equivalent_delta_v=2.2)  # 无 tof_sec
        result.maneuver_events = (ManeuverEvent(kind="departure", t_sec=0.0, dv_km_s=2.2),)
        _fake_transfer(monkeypatch, result)
        facade = Facade(Config(catalog_dir=str(tmp_path / "catalog")))

        response = facade.transfer_design(
            transfer_type="low_thrust", tli_epoch="2025-06-21T11:00:00"
        )

        assert response.record_id is not None
        record = facade.catalog.catalog_get(record_id=response.record_id)
        assert record.scalars["transfer_type"] == "low_thrust"
        assert record.scalars["tof_sec"] is None
        assert record.scalars["tli_epoch"] == "2025-06-21T11:00:00"  # 原样存档
        assert record.scalars["state_frame"] == "force_model_state"


class TestTransferQuery:
    """catalog_query 的 transfer 过滤维度（#574）。"""

    @pytest.fixture
    def facade_with_transfer(self, monkeypatch, tmp_path):
        facade = Facade(Config(catalog_dir=str(tmp_path / "catalog")))
        _fake_transfer(monkeypatch, _make_transfer_result())
        facade.transfer_design(
            transfer_type="WSB", tli_epoch=2460800.5, target_ephemeris=[[1.0] * 6]
        )
        _fake_design(monkeypatch, _make_design_result(orbit_type="DRO"))
        facade.design_orbit(orbit_type="DRO")
        return facade

    def test_filter_by_transfer_type(self, facade_with_transfer):
        records = facade_with_transfer.catalog.catalog_query(transfer_type="WSB").records
        assert len(records) == 1
        assert records[0].transfer_type == "WSB"
        assert len(facade_with_transfer.catalog.catalog_query(transfer_type="LGA").records) == 0

    def test_filter_by_delta_v_range(self, facade_with_transfer):
        assert (
            len(
                facade_with_transfer.catalog.catalog_query(
                    delta_v_min_km_s=3.0, delta_v_max_km_s=4.0
                ).records
            )
            == 1
        )
        assert (
            len(
                facade_with_transfer.catalog.catalog_query(
                    delta_v_min_km_s=5.0, delta_v_max_km_s=6.0
                ).records
            )
            == 0
        )

    def test_filter_by_tli_epoch_range(self, facade_with_transfer):
        hits = facade_with_transfer.catalog.catalog_query(
            tli_epoch_min=2460800.0, tli_epoch_max=2460801.0
        ).records
        assert len(hits) == 1
        assert hits[0].tli_epoch == pytest.approx(2460800.5)
        assert (
            len(
                facade_with_transfer.catalog.catalog_query(
                    tli_epoch_min=2461000.0, tli_epoch_max=2461001.0
                ).records
            )
            == 0
        )

    def test_transfer_summary_fields_present_orbit_records_none(self, facade_with_transfer):
        summaries = {s.source_tool: s for s in facade_with_transfer.catalog.catalog_query().records}
        assert summaries["transfer_design"].transfer_type == "WSB"
        assert summaries["transfer_design"].delta_v_km_s == pytest.approx(3.9)
        assert summaries["transfer_design"].tli_epoch == pytest.approx(2460800.5)
        assert summaries["design_orbit"].transfer_type is None
        assert summaries["design_orbit"].delta_v_km_s is None
        assert summaries["design_orbit"].tli_epoch is None

    def test_invalid_transfer_ranges_rejected(self, facade_with_transfer):
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            facade_with_transfer.catalog.catalog_query(delta_v_min_km_s=4.0, delta_v_max_km_s=3.0)
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            facade_with_transfer.catalog.catalog_query(
                tli_epoch_min=2460801.0, tli_epoch_max=2460800.0
            )

    def test_combined_transfer_filters(self, facade_with_transfer):
        records = facade_with_transfer.catalog.catalog_query(
            transfer_type="WSB", delta_v_min_km_s=3.0, tli_epoch_max=2460900.0
        ).records
        assert len(records) == 1
        records = facade_with_transfer.catalog.catalog_query(
            transfer_type="WSB", delta_v_max_km_s=1.0
        ).records
        assert len(records) == 0


class TestAutoIngest:
    def test_design_orbit_record_lands_in_catalog(self, monkeypatch):
        _fake_design(monkeypatch, _make_design_result(orbit_type="NRHO", jacobi=3.05))
        facade = Facade()
        response = facade.design_orbit(orbit_type="NRHO")

        assert response.record_id is not None
        summaries = facade.catalog.catalog_query().records
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

        record = facade.catalog.catalog_get(record_id=response.record_id)
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

    def test_family_generation_members_are_individual_records(self):
        """族 = 标签（ADR 0045）：成员逐条入库，整族经 family_id 查询。"""
        facade = Facade()
        response = facade.catalog.orbit_family_generation(
            orbit_type="HALO", libration_point=1, max_amplitude_km=3000.0, n_orbits=2
        )

        assert response.family_id is not None
        summaries = facade.catalog.catalog_query(family_id=response.family_id).records
        assert len(summaries) == len(response.orbits)
        assert summaries[0].orbit_family == "halo"
        assert summaries[0].libration_point == 1
        assert sorted(s.member_index for s in summaries) == list(range(len(response.orbits)))
        assert all(s.has_cr3bp and not s.has_ephemeris for s in summaries)
        assert all(s.family_id == response.family_id for s in summaries)

        record = facade.catalog.catalog_get(record_id=summaries[0].record_id)
        assert "cr3bp/states" in record.arrays
        assert record.jacobi is not None
        assert record.family_id == response.family_id

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
        record = facade.catalog.catalog_get(record_id=control_response.record_id)
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


class TestNoImplicitCatalog:
    """ADR 0047：库不维护数据库——默认不入库、无隐式库目录。"""

    def test_default_config_ingests_nothing(self, monkeypatch):
        monkeypatch.delenv("E2M2E_CATALOG_DIR", raising=False)
        monkeypatch.delenv("E2M2E_CATALOG_ENABLED", raising=False)
        _fake_design(monkeypatch, _make_design_result())
        facade = Facade()

        response = facade.design_orbit(orbit_type="DRO")

        assert response.record_id is None
        with pytest.raises(OrbitError, match="CATALOG_NOT_CONFIGURED"):
            facade.catalog.catalog_query()

    def test_catalog_operation_without_dir_raises_not_configured(self, monkeypatch):
        monkeypatch.delenv("E2M2E_CATALOG_DIR", raising=False)
        facade = Facade(Config(catalog_enabled=True))

        with pytest.raises(OrbitError) as exc_info:
            facade.catalog.catalog_query()

        assert exc_info.value.code == "CATALOG_NOT_CONFIGURED"
        assert "E2M2E_CATALOG_DIR" in str(exc_info.value)


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
        records = facade_with_records.catalog.catalog_query(orbit_family="nrho").records
        assert len(records) == 1
        assert records[0].orbit_family == "nrho"

    def test_filter_by_libration_point(self, facade_with_records):
        records = facade_with_records.catalog.catalog_query(libration_point=2).records
        assert len(records) == 1
        assert records[0].orbit_family == "nrho"

    def test_filter_by_jacobi_range(self, facade_with_records):
        assert (
            len(facade_with_records.catalog.catalog_query(jacobi_min=3.1, jacobi_max=3.2).records)
            == 1
        )
        assert (
            len(facade_with_records.catalog.catalog_query(jacobi_min=3.0, jacobi_max=3.1).records)
            == 1
        )
        assert (
            len(facade_with_records.catalog.catalog_query(jacobi_min=2.0, jacobi_max=2.5).records)
            == 0
        )

    def test_filter_by_segment_presence(self, facade_with_records):
        assert len(facade_with_records.catalog.catalog_query(has_ephemeris=True).records) == 3
        assert len(facade_with_records.catalog.catalog_query(has_cr3bp=True).records) == 2

    def test_combined_filter(self, facade_with_records):
        records = facade_with_records.catalog.catalog_query(
            orbit_family="nrho",
            libration_point=2,
            jacobi_min=3.0,
            jacobi_max=3.1,
            has_ephemeris=True,
        ).records
        assert len(records) == 1

    def test_invalid_range_is_rejected(self, facade_with_records):
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            facade_with_records.catalog.catalog_query(jacobi_min=3.2, jacobi_max=3.0)

    def test_summaries_carry_no_arrays(self, facade_with_records):
        records = facade_with_records.catalog.catalog_query().records
        assert len(records) == 3
        assert all(not hasattr(record, "arrays") for record in records)

    def test_query_across_threads_after_lazy_open(self, monkeypatch, tmp_path):
        """mcp-serve 逐 tools/call 换线程池线程（#559）：惰性 catalog 连接
        绑定首用线程后，其他线程查询不得报 SQLite 跨线程错误。"""
        _fake_design(monkeypatch, _make_design_result())
        facade = Facade(Config(catalog_dir=str(tmp_path / "catalog")))
        facade.design_orbit(orbit_type="DRO")  # 种一条记录：跨线程有一致可见的数据
        expected = len(facade.catalog.catalog_query().records)  # 首用线程惰性建库
        assert expected == 1
        counts: list[int] = []
        errors: list[Exception] = []

        def query_from_pool_thread() -> None:
            try:
                counts.append(len(facade.catalog.catalog_query().records))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        for _ in range(2):
            thread = threading.Thread(target=query_from_pool_thread)
            thread.start()
            thread.join()

        assert errors == []
        assert counts == [expected, expected]


class TestGetDelete:
    def test_get_unknown_record_id_raises_structured_error(self):
        with pytest.raises(OrbitError) as exc_info:
            Facade().catalog.catalog_get(record_id="no-such-record")
        assert exc_info.value.code == "RECORD_NOT_FOUND"
        assert exc_info.value.status is ConvergenceState.FAILED

    def test_delete_removes_record(self, monkeypatch):
        _fake_design(monkeypatch, _make_design_result())
        facade = Facade()
        record_id = facade.design_orbit(orbit_type="DRO").record_id

        response = facade.catalog.catalog_delete(record_id=record_id)

        assert response.deleted is True
        assert facade.catalog.catalog_query().records == []
        with pytest.raises(OrbitError, match="RECORD_NOT_FOUND"):
            facade.catalog.catalog_get(record_id=record_id)


class TestTagExport:
    def test_tag_is_visible_in_record(self, monkeypatch):
        _fake_design(monkeypatch, _make_design_result())
        facade = Facade()
        record_id = facade.design_orbit(orbit_type="DRO").record_id

        response = facade.catalog.catalog_tag(
            record_id=record_id, tags=["期中案例"], note="注意近月点高度"
        )

        assert response.record.tags == ["期中案例"]
        record = facade.catalog.catalog_get(record_id=record_id)
        assert record.tags == ["期中案例"]
        assert record.note == "注意近月点高度"

    def test_export_package_carries_annotation(self, monkeypatch, tmp_path):
        _fake_design(monkeypatch, _make_design_result())
        facade = Facade()
        record_id = facade.design_orbit(orbit_type="DRO").record_id
        facade.catalog.catalog_tag(record_id=record_id, tags=["教学"])
        facade.design_orbit(orbit_type="DRO")
        dest = tmp_path / "案例包"

        response = facade.catalog.catalog_export(tags=["教学"], dest=str(dest))

        assert response.exported_count == 1
        assert response.record_ids == [record_id]
        exported = facade.catalog.catalog_get(record_id=record_id)
        assert exported.tags == ["教学"]
        assert (dest / "records" / f"{record_id}.json").exists()
        assert (dest / "manifest.json").exists()


class TestSweep:
    def test_sweep_generates_records_for_grid(self):
        facade = Facade()
        response = facade.catalog.catalog_sweep(
            orbit_types=["HALO"],
            libration_points=[1],
            max_amplitudes_km=[2000.0, 3000.0],
            n_orbits=1,
        )

        assert response.succeeded == 2
        assert response.failed == 0
        assert len(response.family_ids) == 2
        assert all(point.family_id is not None for point in response.points)
        assert {point.parameter_km for point in response.points} == {2000.0, 3000.0}
        # 每点一族（n_orbits=1），成员逐条入库（ADR 0045）
        records = facade.catalog.catalog_query(orbit_family="halo").records
        assert len(records) == 2
        assert all(r.family_id in set(response.family_ids) for r in records)

    def test_sweep_rejects_lissajous_with_one_dimensional_grid(self):
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            Facade().catalog.catalog_sweep(orbit_types=["LISSAJOUS"], max_amplitudes_km=[5000.0])
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            Facade().catalog.catalog_sweep(orbit_types=["LISSAJOUS"], jacobi_windows=[[3.17, 3.18]])

    def test_sweep_rejects_mutually_exclusive_grid_dimensions(self):
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            Facade().catalog.catalog_sweep(
                orbit_types=["HALO"],
                max_amplitudes_km=[2000.0],
                jacobi_windows=[[3.17, 3.18]],
            )
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            Facade().catalog.catalog_sweep(
                orbit_types=["HALO"],
                max_amplitudes_km=[2000.0],
                amplitude_ins_km=[1000.0],
                amplitude_outs_km=[3000.0],
            )

    def test_sweep_rejects_invalid_jacobi_windows(self):
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            Facade().catalog.catalog_sweep(orbit_types=["HALO"], jacobi_windows=[[3.18, 3.17]])
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            Facade().catalog.catalog_sweep(
                orbit_types=["HALO"], jacobi_windows=[[3.17, 3.18, 3.19]]
            )
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            Facade().catalog.catalog_sweep(orbit_types=["HALO"], jacobi_windows=[[3.17]])

    def test_sweep_lissajous_grid_requires_both_amplitude_lists(self):
        with pytest.raises(OrbitError, match="INVALID_PARAMS"):
            Facade().catalog.catalog_sweep(orbit_types=["LISSAJOUS"], amplitude_ins_km=[1000.0])

    def test_sweep_lissajous_grid_creates_records(self):
        facade = Facade()
        response = facade.catalog.catalog_sweep(
            orbit_types=["LISSAJOUS"],
            libration_points=[2],
            amplitude_ins_km=[1000.0, 2000.0],
            amplitude_outs_km=[3000.0],
            n_orbits=2,
        )

        assert response.succeeded == 2
        assert response.failed == 0
        assert len(response.family_ids) == 2
        assert {tuple(point.amplitudes_km) for point in response.points} == {
            (1000.0, 3000.0),
            (2000.0, 3000.0),
        }
        assert all(
            point.parameter_km is None and point.jacobi_window is None for point in response.points
        )
        records = facade.catalog.catalog_query(orbit_family="lissajous", libration_point=2).records
        assert len(records) == 4  # 2 点 × 2 成员，逐条入库（ADR 0045）
        assert all(r.family_id is not None for r in records)

    # jacobi 窗口 sweep 的 Facade 集成用例已移出默认套件（ADR 0037：窗口
    # 模式共享的 Rust trace 有 200 成员兜底，单次 ≥30s；窗口编排便过语义
    # 由 tests/algorithm/test_catalog_sweep.py 的小规模真实扫描覆盖，
    # Facade 落库粘合由 test_sweep_generates_records_for_grid 覆盖）。

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
            Facade().catalog.catalog_sweep(orbit_types=["NRHO"], max_amplitudes_km=[5000.0])

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

        response = Facade().catalog.catalog_sweep(
            orbit_types=["HALO"],
            libration_points=[1],
            max_amplitudes_km=[2000.0, 3000.0, 4000.0],
            n_orbits=1,
        )

        assert response.succeeded == 1
        assert response.failed == 1
        assert len(response.family_ids) == 1
        assert "1 点软失败无成员产出" in response.message
        assert response.points[1].family_id is None
        assert response.points[1].message == "爆炸"
        assert response.points[2].status is ConvergenceState.STAGNATED


class TestCatalogErrors:
    def test_corrupted_index_translates_to_structured_error(self, tmp_path):
        catalog_dir = tmp_path / "catalog"
        (catalog_dir / "records").mkdir(parents=True)
        (catalog_dir / "catalog.db").write_bytes(b"not a sqlite database")
        facade = Facade(Config(catalog_dir=str(catalog_dir)))
        with pytest.raises(OrbitError, match="CATALOG_READ_FAILED"):
            facade.catalog.catalog_query()

    def test_path_traversal_record_id_is_record_not_found(self):
        """record_id 拼路径前的形态校验：路径穿越一律 RECORD_NOT_FOUND。"""
        facade = Facade()
        with pytest.raises(OrbitError, match="RECORD_NOT_FOUND"):
            facade.catalog.catalog_get(record_id="../../etc/passwd")
        with pytest.raises(OrbitError, match="RECORD_NOT_FOUND"):
            facade.catalog.catalog_delete(record_id="../../etc/passwd")


class TestToolInventory:
    def test_catalog_methods_are_in_derived_inventory(self):
        facade = Facade()
        # 类分家后轨道库方法住在 Catalog 类上，但仍在组合根清单内（ADR 0043）
        names = set(mcp_tools(facade.catalog))
        expected = {
            "catalog_query",
            "catalog_get",
            "catalog_delete",
            "catalog_tag",
            "catalog_export",
            "catalog_sweep",
        }
        assert expected <= names
        # catalog_promote 已随一轨一记录移除（ADR 0045 决策 5）
        assert "catalog_promote" not in names

        by_name = {tool.name: tool for tool in tool_inventory(facade)}
        request_models = {
            "catalog_query": CatalogQueryRequest,
            "catalog_get": CatalogGetRequest,
            "catalog_delete": CatalogDeleteRequest,
            "catalog_tag": CatalogTagRequest,
            "catalog_export": CatalogExportRequest,
            "catalog_sweep": CatalogSweepRequest,
        }
        for name, model in request_models.items():
            assert by_name[name].status == "implemented"
            assert by_name[name].request_model is model

    def test_top_n_run_ingests_selected_solution_only(self, monkeypatch, tmp_path):
        """top-N 运行（#583）入库的仍是选中解单条记录：Δv 为顶层口径，
        候选快照不产生额外段或记录。"""
        _fake_transfer(monkeypatch, _make_transfer_result(with_candidates=True))
        facade = Facade(Config(catalog_dir=str(tmp_path / "catalog")))

        response = facade.transfer_design(
            transfer_type="WSB", tli_epoch=2460800.5, target_ephemeris=[[1.0] * 6]
        )

        assert response.record_id is not None
        record = facade.catalog.catalog_get(record_id=response.record_id)
        # 选中解口径（顶层 delta_v=3.9），非候选网格估计 9.9
        assert record.scalars["delta_v_km_s"] == pytest.approx(3.9)
        assert record.scalars["tof_sec"] == pytest.approx(100.0)
        # 段只有选中解的几何（与默认路径同形状），无候选衍生段
        assert record.arrays["transfer/states"].shape == (2, 6)
        assert not any("candidate" in key for key in record.arrays)
