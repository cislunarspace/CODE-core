"""轨道库接口类：catalog 数据管理与批量生成（ADR 0043 决策 2）。

[English]

``Catalog`` is the interface-layer class holding catalog data management
(query/get/delete/tag/promote/export/sweep) plus family generation
(``orbit_family_generation`` — ``catalog_sweep`` is its batch orchestration;
both call the same Rust family kernel). MCP tools, CLI subcommands and the
sidecar reach these methods through the single tool inventory; in-process
callers obtain the instance from ``Facade().catalog`` or construct it
directly (ADR 0043; the class split of ADR 0014 decision 2).

[简体中文]

``Catalog`` 是接口层的轨道库类：catalog 数据管理（query/get/delete/tag/
promote/export/sweep）与族生成（``orbit_family_generation``——
``catalog_sweep`` 是它的批量编排，两者共用同一个 Rust 族生成内核）。
MCP/CLI/sidecar 经单一工具清单到达这些方法；进程内调用方从
``Facade().catalog`` 取实例或直接构造（ADR 0043，ADR 0014 决策 2 的
类分家）。
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from e2m2e.algorithm.results import FamilyGenerationResult
from e2m2e.data.catalog import (
    CatalogError,
    CatalogFilter,
    CatalogStore,
    RecordNotFoundError,
    ephemeris_from_arrays,
    import_baseline,
    numeric_or_none,
)
from e2m2e.data.templates import ConvergenceState, FailureCause
from e2m2e.data.types.orbit import OrbitFamily

from . import catalog_ingest
from .config import Config
from .facade import ProgressCallback, _emit_progress, _exception_triplet, mcp_exposed
from .models import (
    _FAMILY_DEFAULT_LIBRATION_POINT,
    _FAMILY_LIBRATION_POINT_RANGES,
    CatalogDeleteRequest,
    CatalogDeleteResponse,
    CatalogExportRequest,
    CatalogExportResponse,
    CatalogGetRequest,
    CatalogQueryRequest,
    CatalogQueryResponse,
    CatalogRecordResponse,
    CatalogRecordSummary,
    CatalogSweepPointOutcome,
    CatalogSweepRequest,
    CatalogSweepResponse,
    CatalogTagRequest,
    CatalogTagResponse,
    CatalogTerminologyResponse,
    FamilyGenerationRequest,
    FamilyGenerationResponse,
    OrbitError,
)

__all__ = ["Catalog"]

if TYPE_CHECKING:
    # 仅类型检查用：算法/数据层结果类型（运行时懒加载，见各方法内 import）。
    from e2m2e.data.types.trajectory import EphemerisTable


# ---------------------------------------------------------------------------
# 记录 ↔ 响应翻译、过滤构造、sweep 网格展开
# ---------------------------------------------------------------------------


def _record_not_found(exc: RecordNotFoundError) -> OrbitError:
    return OrbitError(
        "RECORD_NOT_FOUND",
        str(exc),
        status=ConvergenceState.FAILED,
        cause=FailureCause.INVALID_INPUT,
    )


def _catalog_read_failed(exc: Exception) -> OrbitError:
    return OrbitError(
        "CATALOG_READ_FAILED",
        str(exc),
        status=ConvergenceState.FAILED,
        cause=FailureCause.BACKEND_FAILURE,
    )


def _catalog_write_failed(exc: Exception) -> OrbitError:
    return OrbitError(
        "CATALOG_WRITE_FAILED",
        str(exc),
        status=ConvergenceState.FAILED,
        cause=FailureCause.BACKEND_FAILURE,
    )


def _to_catalog_filter(request: CatalogQueryRequest) -> CatalogFilter:
    """把查询请求模型翻译为数据层过滤条件。"""
    return CatalogFilter(
        orbit_family=request.orbit_family,
        family_id=request.family_id,
        libration_point=request.libration_point,
        jacobi_min=request.jacobi_min,
        jacobi_max=request.jacobi_max,
        amplitude_min_km=request.amplitude_min_km,
        amplitude_max_km=request.amplitude_max_km,
        has_cr3bp=request.has_cr3bp,
        has_ephemeris=request.has_ephemeris,
        status=None if request.status is None else request.status.value,
        tags=None if request.tags is None else tuple(request.tags),
        transfer_type=request.transfer_type,
        delta_v_min_km_s=request.delta_v_min_km_s,
        delta_v_max_km_s=request.delta_v_max_km_s,
        tli_epoch_min=request.tli_epoch_min,
        tli_epoch_max=request.tli_epoch_max,
    )


def _summary_kwargs(
    *,
    identity: dict[str, Any],
    classification: dict[str, Any],
) -> dict[str, Any]:
    """摘要公共字段（索引行与记录元数据两种来源共用）。

    transfer 维度（#574）两种来源都归一在 identity 顶层：索引行直接带
    列值；记录元数据经 ``_summary_from_meta`` 从 ``scalars`` 提取后合入。
    分类学标签（#581）随 ``classification`` 走，未打标记录为 None。
    族维度（ADR 0045）：``family_id``/``member_index`` 随 identity 顶层走。
    """
    return {
        "record_id": identity["record_id"],
        "created_at": identity["created_at"],
        "source_tool": identity["source_tool"],
        "source_record_id": identity["source_record_id"],
        "family_id": identity.get("family_id"),
        "member_index": identity.get("member_index"),
        "orbit_family": classification["orbit_family"],
        "libration_point": classification["libration_point"],
        "jacobi": classification["jacobi"],
        "amplitude": classification["amplitude"],
        "has_cr3bp": classification["has_cr3bp"],
        "has_ephemeris": classification["has_ephemeris"],
        "taxonomy_labels": classification.get("taxonomy_labels"),
        "transfer_type": identity.get("transfer_type"),
        "delta_v_km_s": identity.get("delta_v_km_s"),
        "tli_epoch": identity.get("tli_epoch"),
        "status": identity["status"],
        "cause": identity["cause"],
        "message": identity["message"],
        "tags": identity["tags"],
        "note": identity["note"],
    }


def _summary_from_index(summary: dict[str, Any]) -> CatalogRecordSummary:
    """索引行摘要 → 响应模型。"""
    return CatalogRecordSummary(
        **_summary_kwargs(
            identity=summary,
            classification=summary["classification"],
        )
    )


def _summary_from_meta(meta: dict[str, Any]) -> CatalogRecordSummary:
    """记录元数据 → 摘要响应模型（族维度与索引用同一份顶层推导）。

    transfer 维度从 ``scalars`` 提取归一到 identity 顶层（与索引行同形）。
    """
    scalars = meta.get("scalars", {})
    identity = {
        **meta,
        "transfer_type": scalars.get("transfer_type"),
        "delta_v_km_s": numeric_or_none(scalars.get("delta_v_km_s")),
        "tli_epoch": numeric_or_none(scalars.get("tli_epoch")),
    }
    return CatalogRecordSummary(
        **_summary_kwargs(
            identity=identity,
            classification=meta["classification"],
        )
    )


def _record_to_response(record: Any) -> CatalogRecordResponse:
    """完整记录 → 响应模型（含数组段，numpy 值）。"""
    meta = record.meta
    summary = _summary_from_meta(meta)
    return CatalogRecordResponse(
        **summary.model_dump(),
        scalars=meta["scalars"],
        request=meta["request"],
        details=meta.get("details"),
        arrays=record.arrays,
    )


#: sweep 各族一维扫描的主延拓参数字段（FamilyGenerationRequest 术语）。
#: LISSAJOUS 不在表中：其一维扫描不成立，走 amplitude_ins_km ×
#: amplitude_outs_km 二维振幅网格（能量窗口扫描同理不适用）。
_SWEEP_GRID_FIELD = {
    "HALO": "max_amplitude_km",
    "NRHO": "perilune_height_max_km",
    "AXIAL": "max_amplitude_km",
    "SPO": "max_amplitude_km",
    "LPO": "max_amplitude_km",
    "HORSESHOE": "max_amplitude_km",
}


@dataclasses.dataclass(frozen=True)
class _SweepPlanPoint:
    """展开后的扫描参数点：族请求（已校验、含默认值）与主参数标签。"""

    family_request: FamilyGenerationRequest
    libration_point: int
    parameter_km: float | None = None
    jacobi_window: tuple[float, float] | None = None
    amplitudes_km: tuple[float, float] | None = None


def _sweep_grid_mode(request: CatalogSweepRequest, selection: str) -> str:
    """按请求网格字段与族返回扫描主参数维度（维度互斥已在请求模型拒绝）。

    族 × 可用维度的适用性只经 ``CatalogSweepRequest.supported_grid_dimensions``
    判断（条件取值域公开且同源，ADR 0014 决策 8），本函数不另立规则。
    """
    dimensions = CatalogSweepRequest.supported_grid_dimensions(selection)
    if request.amplitude_ins_km is not None:
        if "amplitude_ins_km" not in dimensions:
            raise ValueError("amplitude_ins_km/amplitude_outs_km 二维振幅网格仅 LISSAJOUS 用")
        return "lissajous"
    if "amplitude_ins_km" in dimensions:
        raise ValueError(
            "LISSAJOUS 振幅为面内/面外二维，只支持 amplitude_ins_km × "
            "amplitude_outs_km 二维振幅网格（不支持一维振幅或能量窗口）"
        )
    if request.jacobi_windows is not None:
        if "jacobi_windows" not in dimensions:
            raise ValueError(f"{selection} 不支持 jacobi_windows 能量窗口扫描")
        return "jacobi"
    return "amplitude"


def _expand_sweep_points(request: CatalogSweepRequest) -> list[_SweepPlanPoint]:
    """把扫描参数空间展开为参数点序列（族请求 + 主参数标签）。

    网格 = 族 × 平动点 × 主参数维度：NRHO 一维扫 perilune_heights_max_km，
    其余共线/三角族扫 max_amplitudes_km；能量窗口维扫 jacobi_windows（族
    延拓范围取各族默认值，不叠振幅网格）；LISSAJOUS 扫面内×面外二维振幅
    笛卡尔积（相位取请求默认值）。每点经 ``FamilyGenerationRequest`` 校验，
    非法点在此拒绝（ValueError → 接口层翻译 INVALID_PARAMS）。
    """
    points: list[_SweepPlanPoint] = []
    for orbit_type in request.orbit_types:
        selection = orbit_type.upper()
        try:
            CatalogSweepRequest.supported_grid_dimensions(selection)
        except ValueError as exc:
            raise ValueError(f"catalog_sweep 不支持的 orbit_type: {orbit_type!r}") from exc
        allowed = _FAMILY_LIBRATION_POINT_RANGES[selection]
        libration_points = request.libration_points or [_FAMILY_DEFAULT_LIBRATION_POINT[selection]]
        for libration_point in libration_points:
            if libration_point not in allowed:
                raise ValueError(
                    f"{selection} libration_point 必须为 {sorted(allowed)}，"
                    f"当前 {libration_point!r}"
                )
        mode = _sweep_grid_mode(request, selection)
        for libration_point in libration_points:
            if mode == "jacobi":
                for window in request.jacobi_windows or ():
                    points.append(
                        _SweepPlanPoint(
                            family_request=FamilyGenerationRequest(
                                orbit_type=selection,
                                libration_point=libration_point,
                                n_orbits=request.n_orbits,
                            ),
                            libration_point=libration_point,
                            jacobi_window=(float(window[0]), float(window[1])),
                        )
                    )
                continue
            if mode == "lissajous":
                for amplitude_in in request.amplitude_ins_km or ():
                    for amplitude_out in request.amplitude_outs_km or ():
                        points.append(
                            _SweepPlanPoint(
                                family_request=FamilyGenerationRequest(
                                    orbit_type=selection,
                                    libration_point=libration_point,
                                    n_orbits=request.n_orbits,
                                    amplitude_in_km=amplitude_in,
                                    amplitude_out_km=amplitude_out,
                                ),
                                libration_point=libration_point,
                                amplitudes_km=(float(amplitude_in), float(amplitude_out)),
                            )
                        )
                continue
            grid = (
                request.perilune_heights_max_km
                if selection == "NRHO"
                else request.max_amplitudes_km
            )
            grid_name = "perilune_heights_max_km" if selection == "NRHO" else "max_amplitudes_km"
            if not grid:
                raise ValueError(f"{selection} 扫描需要 {grid_name} 网格")
            field_name = _SWEEP_GRID_FIELD[selection]
            for value in grid:
                overrides: dict[str, Any] = {field_name: value}
                points.append(
                    _SweepPlanPoint(
                        family_request=FamilyGenerationRequest(
                            orbit_type=selection,
                            libration_point=libration_point,
                            n_orbits=request.n_orbits,
                            **overrides,
                        ),
                        libration_point=libration_point,
                        parameter_km=float(value),
                    )
                )
    return points


def _family_entry_kwargs(request: FamilyGenerationRequest) -> dict[str, Any]:
    """族请求模型 → 算法层族生成入口的关键字参数。"""
    selection = request.orbit_type.upper()
    if selection == "HALO":
        return {"max_amplitude_km": request.max_amplitude_km}
    if selection == "NRHO":
        return {
            "north_south": request.north_south,
            "perilune_height_max_km": request.perilune_height_max_km,
            "continuation_direction": request.continuation_direction,
        }
    if selection == "AXIAL":
        return {
            "max_amplitude_km": request.max_amplitude_km,
            "continuation_direction": request.continuation_direction,
        }
    if selection == "LISSAJOUS":
        return {
            "amplitude_in_km": request.amplitude_in_km,
            "amplitude_out_km": request.amplitude_out_km,
            "phase_in": request.phase_in,
            "phase_out": request.phase_out,
        }
    return {
        "min_amplitude_km": request.min_amplitude_km,
        "max_amplitude_km": request.max_amplitude_km,
        "continuation_direction": request.continuation_direction,
        "match_tolerance_km": request.match_tolerance_km,
    }


def _family_generation_payload(
    result: OrbitFamily | FamilyGenerationResult,
    *,
    requested_members: int | None = None,
) -> FamilyGenerationResponse:
    """把算法层成功或软失败统一投影为接口层专属响应。

    分类学标签（#581，ADR 0042）对全体成员实测打标后去重；拟周期
    （lissajous）与不在分类学内的族按映射表为空表。
    """
    from .catalog_ingest import stamp_taxonomy_labels

    if isinstance(result, FamilyGenerationResult):
        family = result.family
        status = result.status
        cause = result.cause
        message = result.message
        requested_members = result.requested_members
        generated_members = result.generated_members
    else:
        family = result
        status = ConvergenceState.CONVERGED
        cause = FailureCause.NONE
        message = "轨道族生成完成"
        requested_members = requested_members or len(family)
        generated_members = len(family)
    taxonomy_labels, _ = stamp_taxonomy_labels(
        family.family_type,
        family.orbits,
        periodicity=family.metadata.get("periodicity", "periodic"),
        context="orbit_family_generation",
    )
    return FamilyGenerationResponse(
        status=status,
        cause=cause,
        message=message,
        orbits=family.orbits,
        family_type=family.family_type,
        system=family.system,
        metadata=family.metadata,
        taxonomy_labels=taxonomy_labels,
        requested_members=requested_members,
        generated_members=generated_members,
    )


class Catalog:
    """轨道库接口类（ADR 0043 决策 2）：数据管理 + 批量生成。

    ``Catalog(config=Config(...))`` 构造注入配置；进程内调用方通常经
    ``Facade().catalog`` 取同一实例。``auto_ingest`` 与
    ``load_record_ephemeris`` 是同包任务方法（Facade 的 design/control/
    transfer 自动入库与 input_record_id 输入解析）的接缝，非 MCP 工具。
    """

    def __init__(self, config: Config | None = None) -> None:
        """构造 Catalog。

        Args:
            config: 运行配置（api/config.py Config），缺省从环境变量读。
        """
        self._config = config or Config()
        self._catalog_store: CatalogStore | None = None

    @property
    def config(self) -> Config:
        """运行配置（只读视图）。"""
        return self._config

    # ---- 库目录与自动入库（ADR 0031；Facade 任务方法的接缝）----

    def _open_catalog(self) -> CatalogStore:
        """懒打开库目录（首次使用时才产生目录副作用）。

        首次打开时做基线首用导入（ADR 0036 决策 5）：用户库缺基线记录
        或版本不一致时从包内复制并重建索引；``catalog_baseline_import``
        可关闭。
        """
        if self._catalog_store is None:
            self._catalog_store = CatalogStore(self._config.catalog_dir)
            if self._config.catalog_baseline_import:
                try:
                    import_baseline(self._catalog_store)
                except Exception as exc:
                    raise _catalog_write_failed(exc) from exc
        return self._catalog_store

    def auto_ingest(self, builder: Callable[[], tuple[dict, dict[str, Any]] | None]) -> str | None:
        """单条产物自动入库（ADR 0031 决策 8；Facade 任务方法经此接缝入库）。

        无产物（站保星历缺失）返回 None；入库失败抛
        ``CATALOG_WRITE_FAILED``，不静默降级、不冒名为计算失败（ADR 0020）。
        """
        if not self._config.catalog_enabled:
            return None
        try:
            built = builder()
            if built is None:
                return None
            meta, arrays = built
            return self._open_catalog().put(meta, arrays)
        except OrbitError:
            raise
        except Exception as exc:
            raise _catalog_write_failed(exc) from exc

    def auto_ingest_family(
        self,
        builder: Callable[[], tuple[str, list[tuple[dict, dict[str, Any]]]] | None],
    ) -> str | None:
        """族产物逐成员自动入库（ADR 0045：一轨一记录），返回 family_id。

        无产物（零成员）返回 None。逐条写入、无跨记录事务（决策 7）：
        中途失败时已写成员保留（诚实的部分结果）并抛
        ``CATALOG_WRITE_FAILED``；运行级溯源随每条成员走，同一 builder
        单点写入，不会漂移（决策 2）。
        """
        if not self._config.catalog_enabled:
            return None
        try:
            built = builder()
            if built is None:
                return None
            family_id, records = built
            for meta, arrays in records:
                self._open_catalog().put(meta, arrays)
            return family_id
        except OrbitError:
            raise
        except Exception as exc:
            raise _catalog_write_failed(exc) from exc

    def load_record_ephemeris(self, record_id: str) -> tuple[dict, EphemerisTable]:
        """取库中记录的元数据与星历段（control_orbit 的 input_record_id 输入源）。"""
        try:
            record = self._open_catalog().get(record_id)
        except RecordNotFoundError as exc:
            raise OrbitError(
                "RECORD_NOT_FOUND",
                str(exc),
                status=ConvergenceState.FAILED,
                cause=FailureCause.INVALID_INPUT,
            ) from exc
        try:
            table = ephemeris_from_arrays(record.arrays)
        except CatalogError as exc:
            raise OrbitError(
                "INVALID_PARAMS",
                f"记录 {record_id} 无星历段，不能作为站保输入：{exc}",
                status=ConvergenceState.FAILED,
                cause=FailureCause.INVALID_INPUT,
            ) from exc
        return record.meta, table

    def _get_record(self, record_id: str) -> Any:
        """取完整记录；不存在抛 ``RECORD_NOT_FOUND``，记录损坏抛 ``CATALOG_READ_FAILED``。"""
        try:
            return self._open_catalog().get(record_id)
        except RecordNotFoundError as exc:
            raise _record_not_found(exc) from exc
        except CatalogError as exc:
            raise OrbitError(
                "CATALOG_READ_FAILED",
                str(exc),
                status=ConvergenceState.FAILED,
                cause=FailureCause.BACKEND_FAILURE,
            ) from exc

    def _ingest_sweep_outcome(self, outcome: Any, family_request: Any) -> str | None:
        """扫描单点结果逐成员入库，返回 family_id；硬失败点（result=None）无记录。"""
        result = outcome.result
        if result is None:
            return None
        return self.auto_ingest_family(
            lambda: catalog_ingest.build_family_records(
                family_request,
                family=result.family,
                status=outcome.status,
                cause=outcome.cause,
                message=outcome.message,
                requested_members=result.requested_members,
                generated_members=result.generated_members,
            )
        )

    # ---- 族生成与轨道库工具（ADR 0043 决策 2，mcp_exposed=True）----

    @mcp_exposed(request_model=FamilyGenerationRequest)
    def orbit_family_generation(
        self, progress_callback: ProgressCallback | None = None, **params
    ) -> FamilyGenerationResponse:
        """Orbit family generation (tier 2). / 轨道族生成（二档）.

        Pydantic 模型校验 → 按 orbit_type 分派到算法层族生成入口 →
        结构化错误。八族均已实现，成功返回统一容器
        ``FamilyGenerationResponse``（兼容 ``OrbitFamily`` 读取接口）；
        Lissajous 是拟周期参数采样，族上显式标注
        ``periodicity=quasi-periodic``。软失败使用同一响应保留部分族。
        ``progress_callback(fraction, message)`` 上报阶段级进度（族生成
        是单次 Rust 调用，仅起止两端；逐成员进度待 Rust 侧通道）。
        """
        try:
            request = FamilyGenerationRequest(**params)
            _emit_progress(progress_callback, 0.0, f"轨道族生成开始：{request.orbit_type}")
            sel = request.orbit_type.upper()
            if sel == "HALO":
                from e2m2e.algorithm.family import design_halo_family

                result = design_halo_family(
                    request.libration_point,
                    request.max_amplitude_km,
                    n_orbits=request.n_orbits,
                )
                response = _family_generation_payload(
                    result,
                    requested_members=request.n_orbits,
                )
            elif sel == "NRHO":
                from e2m2e.algorithm.family import design_nrho_family

                assert request.north_south is not None
                assert request.perilune_height_max_km is not None
                assert request.continuation_direction is not None
                result = design_nrho_family(
                    request.libration_point,
                    request.north_south,
                    request.perilune_height_max_km,
                    n_orbits=request.n_orbits,
                    continuation_direction=request.continuation_direction,
                )
                response = _family_generation_payload(result)
            elif sel == "AXIAL":
                from e2m2e.algorithm.family import design_axial_family

                assert request.max_amplitude_km is not None
                assert request.continuation_direction is not None
                result = design_axial_family(
                    request.libration_point,
                    request.max_amplitude_km,
                    n_orbits=request.n_orbits,
                    continuation_direction=request.continuation_direction,
                )
                response = _family_generation_payload(result)
            elif sel == "LISSAJOUS":
                from e2m2e.algorithm.family import design_lissajous_family

                assert request.amplitude_in_km is not None
                assert request.amplitude_out_km is not None
                assert request.phase_in is not None
                assert request.phase_out is not None
                assert request.sampling_mode is not None
                result = design_lissajous_family(
                    request.libration_point,
                    request.amplitude_in_km,
                    request.amplitude_out_km,
                    request.phase_in,
                    request.phase_out,
                    n_orbits=request.n_orbits,
                    sampling_mode=request.sampling_mode,
                )
                response = _family_generation_payload(result)
            elif sel == "DRO":
                from e2m2e.algorithm.family import design_dro_family

                assert request.min_amplitude_km is not None
                assert request.max_amplitude_km is not None
                result = design_dro_family(
                    request.min_amplitude_km,
                    request.max_amplitude_km,
                    n_orbits=request.n_orbits,
                )
                response = _family_generation_payload(result)
            else:
                from e2m2e.algorithm.family import (
                    design_horseshoe_family,
                    design_lpo_family,
                    design_spo_family,
                )

                triangular_entry = {
                    "SPO": design_spo_family,
                    "LPO": design_lpo_family,
                    "HORSESHOE": design_horseshoe_family,
                }[sel]
                assert request.min_amplitude_km is not None
                assert request.max_amplitude_km is not None
                assert request.continuation_direction is not None
                assert request.match_tolerance_km is not None
                result = triangular_entry(
                    request.libration_point,
                    request.min_amplitude_km,
                    request.max_amplitude_km,
                    n_orbits=request.n_orbits,
                    continuation_direction=request.continuation_direction,
                    match_tolerance_km=request.match_tolerance_km,
                )
                response = _family_generation_payload(result)
        except OrbitError:
            raise
        except (ValueError, TypeError) as exc:
            raise OrbitError(
                "INVALID_PARAMS",
                str(exc),
                status=ConvergenceState.FAILED,
                cause=FailureCause.INVALID_INPUT,
            ) from exc
        except Exception as exc:
            status, cause, message = _exception_triplet(exc)
            raise OrbitError(
                "FAMILY_GENERATION_FAILED", message, status=status, cause=cause
            ) from exc
        _emit_progress(progress_callback, 1.0, "轨道族生成完成")
        response.family_id = self.auto_ingest_family(
            lambda: catalog_ingest.build_family_records(
                request,
                family=response,
                status=response.status,
                cause=response.cause,
                message=response.message,
                requested_members=response.requested_members,
                generated_members=response.generated_members,
            )
        )
        return response

    @mcp_exposed(request_model=CatalogQueryRequest)
    def catalog_query(self, **params) -> CatalogQueryResponse:
        """Multi-dimensional catalog query returning record summaries..

        多维过滤查询，返回摘要列表（不含数组段与请求快照）。"""
        try:
            request = CatalogQueryRequest(**params)
        except (ValueError, TypeError) as exc:
            raise OrbitError(
                "INVALID_PARAMS",
                str(exc),
                status=ConvergenceState.FAILED,
                cause=FailureCause.INVALID_INPUT,
            ) from exc
        try:
            summaries = self._open_catalog().query(_to_catalog_filter(request))
        except CatalogError as exc:
            raise _catalog_read_failed(exc) from exc
        records = [_summary_from_index(summary) for summary in summaries]
        return CatalogQueryResponse(
            status=ConvergenceState.CONVERGED,
            cause=FailureCause.NONE,
            message=f"查询完成：{len(records)} 条记录",
            records=records,
        )

    @mcp_exposed(request_model=CatalogGetRequest)
    def catalog_get(self, **params) -> CatalogRecordResponse:
        """Fetch a full record by ``record_id``..

        按 record_id 取完整记录（含数组段）；不存在抛 ``RECORD_NOT_FOUND``。"""
        try:
            request = CatalogGetRequest(**params)
        except (ValueError, TypeError) as exc:
            raise OrbitError(
                "INVALID_PARAMS",
                str(exc),
                status=ConvergenceState.FAILED,
                cause=FailureCause.INVALID_INPUT,
            ) from exc
        record = self._get_record(request.record_id)
        return _record_to_response(record)

    @mcp_exposed(request_model=CatalogDeleteRequest)
    def catalog_delete(self, **params) -> CatalogDeleteResponse:
        """Delete a record by ``record_id`` — irreversible..

        按 record_id 删除记录（文件与索引条目）；删除不可撤销。"""
        try:
            request = CatalogDeleteRequest(**params)
            self._open_catalog().delete(request.record_id)
        except RecordNotFoundError as exc:
            raise _record_not_found(exc) from exc
        except (CatalogError, OSError) as exc:
            raise _catalog_write_failed(exc) from exc
        except (ValueError, TypeError) as exc:
            raise OrbitError(
                "INVALID_PARAMS",
                str(exc),
                status=ConvergenceState.FAILED,
                cause=FailureCause.INVALID_INPUT,
            ) from exc
        return CatalogDeleteResponse(
            status=ConvergenceState.CONVERGED,
            cause=FailureCause.NONE,
            message=f"记录已删除：{request.record_id}",
            record_id=request.record_id,
            deleted=True,
        )

    @mcp_exposed(request_model=CatalogTagRequest)
    def catalog_tag(self, **params) -> CatalogTagResponse:
        """Write teaching annotations to the JSON record..

        写教学标注入 JSON 记录（随文件走）；tags 整体替换，note=None 保留。"""
        try:
            request = CatalogTagRequest(**params)
            meta = self._open_catalog().tag(request.record_id, request.tags, request.note)
        except RecordNotFoundError as exc:
            raise _record_not_found(exc) from exc
        except (CatalogError, OSError) as exc:
            raise _catalog_write_failed(exc) from exc
        except (ValueError, TypeError) as exc:
            raise OrbitError(
                "INVALID_PARAMS",
                str(exc),
                status=ConvergenceState.FAILED,
                cause=FailureCause.INVALID_INPUT,
            ) from exc
        return CatalogTagResponse(
            status=ConvergenceState.CONVERGED,
            cause=FailureCause.NONE,
            message=f"标注已写入：{request.record_id}",
            record=_summary_from_meta(meta),
        )

    @mcp_exposed(request_model=CatalogExportRequest)
    def catalog_export(self, **params) -> CatalogExportResponse:
        """Package the query result subset for distribution..

        把查询子集打包导出（标注随包）；包可直接作为库打开。"""
        try:
            request = CatalogExportRequest(**params)
        except (ValueError, TypeError) as exc:
            raise OrbitError(
                "INVALID_PARAMS",
                str(exc),
                status=ConvergenceState.FAILED,
                cause=FailureCause.INVALID_INPUT,
            ) from exc
        try:
            record_ids = self._open_catalog().export(_to_catalog_filter(request), request.dest)
        except (CatalogError, OSError) as exc:
            raise _catalog_write_failed(exc) from exc
        return CatalogExportResponse(
            status=ConvergenceState.CONVERGED,
            cause=FailureCause.NONE,
            message=f"导出完成：{len(record_ids)} 条记录 → {request.dest}",
            dest=request.dest,
            record_ids=record_ids,
            exported_count=len(record_ids),
        )

    @mcp_exposed
    def catalog_terminology(self) -> CatalogTerminologyResponse:
        """Closed value sets callers need to render catalog results..

        术语清单（ADR 0044）：分类学标签图例 + orbit_family 闭值集 +
        transfer_type 闭值集，无参数。包版本即术语版本：调用方每会话取
        一次、升级后刷新，未知标签按可读规范串原样渲染。"""
        from e2m2e.data.catalog.terminology import (
            RECORD_ORBIT_FAMILIES,
            TRANSFER_TYPES,
            label_legend,
        )

        legend = label_legend()
        return CatalogTerminologyResponse(
            status=ConvergenceState.CONVERGED,
            cause=FailureCause.NONE,
            message=(
                f"术语清单：{len(legend)} 标签 / "
                f"{len(RECORD_ORBIT_FAMILIES)} 族名 / {len(TRANSFER_TYPES)} 转移类型"
            ),
            taxonomy_labels=legend,
            orbit_families=list(RECORD_ORBIT_FAMILIES),
            transfer_types=list(TRANSFER_TYPES),
        )

    @mcp_exposed(request_model=CatalogSweepRequest)
    def catalog_sweep(self, **params) -> CatalogSweepResponse:
        """Parameter-space sweep batch-generating into the catalog.

        参数空间扫描批量生成并入库（编排复用 ADR 0029 的 Rust 族生成）。

            网格 = 族 × 平动点 × 主参数维度（一维振幅/近月点高度、能量窗口、
            LISSAJOUS 二维振幅，三选一）；部分参数点失败时已产出的记录
            保留，失败原因逐点可查（ADR 0020 软失败语义）。
        """
        try:
            request = CatalogSweepRequest(**params)
            points = _expand_sweep_points(request)
        except (ValueError, TypeError) as exc:
            raise OrbitError(
                "INVALID_PARAMS",
                str(exc),
                status=ConvergenceState.FAILED,
                cause=FailureCause.INVALID_INPUT,
            ) from exc
        from e2m2e.algorithm.catalog_sweep import FamilySweepPoint, run_family_sweep

        sweep_points = [
            FamilySweepPoint(
                orbit_type=plan.family_request.orbit_type.upper(),
                libration_point=plan.libration_point,
                n_orbits=plan.family_request.n_orbits,
                kwargs=_family_entry_kwargs(plan.family_request),
                jacobi_window=plan.jacobi_window,
            )
            for plan in points
        ]
        raw_outcomes = run_family_sweep(sweep_points)

        outcomes: list[CatalogSweepPointOutcome] = []
        family_ids: list[str] = []
        failed = 0
        for plan, outcome in zip(points, raw_outcomes, strict=True):
            family_id = self._ingest_sweep_outcome(outcome, plan.family_request)
            if family_id is not None:
                family_ids.append(family_id)
            if outcome.result is None:
                failed += 1
            outcomes.append(
                CatalogSweepPointOutcome(
                    orbit_type=plan.family_request.orbit_type.upper(),
                    libration_point=plan.libration_point,
                    parameter_km=plan.parameter_km,
                    jacobi_window=(
                        [plan.jacobi_window[0], plan.jacobi_window[1]]
                        if plan.jacobi_window is not None
                        else None
                    ),
                    amplitudes_km=(
                        [plan.amplitudes_km[0], plan.amplitudes_km[1]]
                        if plan.amplitudes_km is not None
                        else None
                    ),
                    status=outcome.status,
                    cause=outcome.cause,
                    message=outcome.message,
                    family_id=family_id,
                    generated_members=(
                        outcome.result.generated_members if outcome.result is not None else 0
                    ),
                )
            )
        succeeded = len(family_ids)
        soft_empty = len(points) - succeeded - failed
        message = (
            f"扫描完成：{succeeded} 个参数点入库（逐成员记录，ADR 0045），"
            f"{failed} 点失败，共 {len(points)} 点"
        )
        if soft_empty:
            message += f"（{soft_empty} 点软失败无成员产出）"
        return CatalogSweepResponse(
            status=ConvergenceState.CONVERGED,
            cause=FailureCause.NONE,
            message=message,
            points=outcomes,
            family_ids=family_ids,
            succeeded=succeeded,
            failed=failed,
        )
