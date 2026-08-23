"""Facade 门面：唯一公开顶级入口，粗粒度任务方法。

两层粒度（ADR 0014）：Facade 暴露粗粒度任务方法（人类/Agent 常用），算法层
保留细粒度 API（专家用）。MCP 工具 = Facade 方法全集（纯派生），方法带
``mcp_exposed`` 元数据控制是否对 MCP 暴露。

实现状态：一档任务已接入 algorithm/ 编排器（design_orbit/control_orbit/
transfer_design/orbit_propagation/spacetime_transform），二档子任务已接入
已有算法（family/stability/proximity）。
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from e2m2e.algorithm.results import FamilyGenerationResult
from e2m2e.data.catalog import (
    CatalogError,
    CatalogFilter,
    CatalogStore,
    RecordNotFoundError,
    ephemeris_from_arrays,
    import_baseline,
)
from e2m2e.data.catalog import member_count as catalog_member_count
from e2m2e.data.constants import SECONDS_PER_DAY
from e2m2e.data.templates import ConvergenceState, FailureCause
from e2m2e.data.types.orbit import OrbitFamily

from . import catalog_ingest
from .config import Config
from .models import (
    _FAMILY_DEFAULT_LIBRATION_POINT,
    _FAMILY_LIBRATION_POINT_RANGES,
    CatalogDeleteRequest,
    CatalogDeleteResponse,
    CatalogExportRequest,
    CatalogExportResponse,
    CatalogGetRequest,
    CatalogPromoteRequest,
    CatalogPromoteResponse,
    CatalogQueryRequest,
    CatalogQueryResponse,
    CatalogRecordResponse,
    CatalogRecordSummary,
    CatalogSweepPointOutcome,
    CatalogSweepRequest,
    CatalogSweepResponse,
    CatalogTagRequest,
    CatalogTagResponse,
    ControlOrbitRequest,
    ControlOrbitResponse,
    DesignOrbitRequest,
    DesignOrbitResponse,
    FamilyGenerationRequest,
    FamilyGenerationResponse,
    OrbitError,
    PropagationRequest,
    PropagationResponse,
    SpacetimeTransformRequest,
    SpacetimeTransformResponse,
    TransferDesignRequest,
    TransferDesignResponse,
)

__all__ = ["Facade", "ToolInfo", "mcp_tools", "tool_inventory"]

if TYPE_CHECKING:
    # 仅类型检查用：算法/数据层结果类型（运行时懒加载，见各 Facade 方法内 import）。
    from e2m2e.algorithm.design import OrbitDesignResult
    from e2m2e.algorithm.station_keeping import ControlOrbitResult
    from e2m2e.data.types.trajectory import EphemerisTable


@dataclasses.dataclass(frozen=True)
class ToolInfo:
    """Facade 工具的机器可读元数据。"""

    name: str
    mcp_exposed: bool
    status: Literal["implemented", "placeholder"]
    request_model: type[Any] | None = None


def mcp_exposed(
    func: Callable[..., Any] | None = None,
    *,
    status: Literal["implemented", "placeholder"] = "implemented",
    request_model: type[Any] | None = None,
) -> Callable[..., Any]:
    """标记 Facade 方法对 MCP 暴露并记录其实现状态和请求模型。"""

    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        function.mcp_exposed = True  # type: ignore[attr-defined]
        function.tool_status = status  # type: ignore[attr-defined]
        function.request_model = request_model  # type: ignore[attr-defined]
        return function

    return decorate(func) if func is not None else decorate


def _result_triplet(result: Any) -> tuple[ConvergenceState, FailureCause, str]:
    """读取算法任务结果的最终状态三元组。"""
    try:
        if isinstance(result, dict):
            triplet = result["status"], result["cause"], result["message"]
        else:
            triplet = result.status, result.cause, result.message
    except (AttributeError, KeyError) as exc:
        raise OrbitError(
            "RESULT_CONTRACT_FAILED",
            "算法任务结果缺少 status/cause/message 契约",
            status=ConvergenceState.FAILED,
            cause=FailureCause.BACKEND_FAILURE,
        ) from exc
    return triplet


def _family_generation_payload(
    result: OrbitFamily | FamilyGenerationResult,
    *,
    requested_members: int | None = None,
) -> FamilyGenerationResponse:
    """把算法层成功或软失败统一投影为 Facade 专属响应。"""
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
    return FamilyGenerationResponse(
        status=status,
        cause=cause,
        message=message,
        orbits=family.orbits,
        family_type=family.family_type,
        system=family.system,
        metadata=family.metadata,
        requested_members=requested_members,
        generated_members=generated_members,
    )


def _exception_triplet(exc: Exception) -> tuple[ConvergenceState, FailureCause, str]:
    """读取算法异常携带的最终状态三元组。

    异常自身携带三元组时原样返回；否则按算法层普通失败处理（保留原始
    诊断信息），不让 Facade 吞掉 message 或误报契约错误。
    """
    status = getattr(exc, "status", None)
    cause = getattr(exc, "cause", None)
    message = getattr(exc, "message", None)
    if status is not None and cause is not None:
        return status, cause, message or str(exc)
    return (
        ConvergenceState.FAILED,
        FailureCause.UNKNOWN,
        str(exc),
    )


def _details_to_dict(details: Any) -> dict[str, Any]:
    """把 details dataclass 转为 JSON 兼容 dict（ndarray → list）。"""
    if details is None:
        return {}
    if isinstance(details, dict):
        return {k: _serialize_value(v) for k, v in details.items()}
    if dataclasses.is_dataclass(details) and not isinstance(details, type):
        return {
            f.name: _serialize_value(getattr(details, f.name)) for f in dataclasses.fields(details)
        }
    return {"value": _serialize_value(details)}


def _serialize_value(value: Any) -> Any:
    """递归序列化 numpy 值为 JSON 兼容类型。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    return value


def _ephemeris_to_dict(ephemeris: EphemerisTable | None) -> dict[str, Any] | None:
    """把 ``EphemerisTable`` 序列化为 JSON 兼容 dict（ndarray → list）。

    迭代 ``dataclasses.fields(EphemerisTable)`` 取全字段（自动跟随容器
    演进），跳过 ``raw_text`` （原始文件文本，程序生成时为空串、读入时为
    大字符串，下游重建容器不需要）。``times_jd_tdb`` 设计链路不填 → None。
    下游过滤 None 值即可重建 ``EphemerisTable``。``None`` 输入返回 ``None``
    （control 全样本失败时受控星历缺失）。
    """
    if ephemeris is None:
        return None
    return {
        f.name: _serialize_value(getattr(ephemeris, f.name))
        for f in dataclasses.fields(ephemeris)
        if f.name != "raw_text"
    }


def _design_result_to_response(result: OrbitDesignResult) -> DesignOrbitResponse:
    """把 ``OrbitDesignResult`` 翻译为 ``DesignOrbitResponse`` （含几何字段，#312）。

    纯翻译、无副作用、不依赖 SPICE。ELFO 场景下 ``cr3bp_orbit`` /
    ``correction`` 为 None，对应字段输出默认值（mu=None、states/times 空、
    correction_iterations=0）。CR3BP 场景下从 ``cr3bp_orbit`` 提取
    ``states`` / ``times`` / ``mu``。
    """
    cr3bp_orbit = result.cr3bp_orbit
    correction = result.correction
    if cr3bp_orbit is not None:
        mu = getattr(cr3bp_orbit.system, "mu", None)
        states = cr3bp_orbit.states.tolist()
        times = cr3bp_orbit.times.tolist()
    else:
        mu = None
        states = []
        times = []
    return DesignOrbitResponse(
        orbit_type=result.orbit_type,
        epoch_utc=result.epoch_utc,
        duration_day=result.duration_day,
        initial_state=result.initial_state.tolist(),
        cr3bp_jacobi=result.cr3bp_jacobi,
        status=result.status,
        cause=result.cause,
        message=result.message,
        correction_iterations=correction.iterations if correction else 0,
        correction_method=result.correction_method,
        force_config=result.force_config,
        mu=mu,
        states=states,
        times=times,
        ephemeris=_ephemeris_to_dict(result.ephemeris),
        drift_e=result.drift_e,
        drift_aop_deg=result.drift_aop_deg,
        drift_rp_km=result.drift_rp_km,
        secular_aop_rate_deg_per_year=result.secular_aop_rate_deg_per_year,
    )


def _control_result_to_response(
    result: ControlOrbitResult, *, mu: float | None
) -> ControlOrbitResponse:
    """把 ``ControlOrbitResult`` 翻译为 ``ControlOrbitResponse`` （含几何字段，#312）。

    ``controlled_ephemeris`` 来自最后一次蒙特卡洛样本（全失败时 None）；
    ``mu`` 由请求透传——算法层不产 mu，design→control 链式时由调用方注入。
    """
    return ControlOrbitResponse(
        status=result.status,
        cause=result.cause,
        message=result.message,
        num_failed=result.num_failed,
        sk_statistic={
            "rows": result.sk_statistic.rows.tolist(),
            "num_failed": result.sk_statistic.num_failed,
        },
        maneuvers={
            "mjd_tdb": result.maneuvers.mjd_tdb.tolist(),
            "delta_v_mps": result.maneuvers.delta_v_mps.tolist(),
        },
        controlled_ephemeris=_ephemeris_to_dict(result.controlled_ephemeris),
        mu=mu,
    )


# ---------------------------------------------------------------------------
# 轨道库 catalog（ADR 0031）：记录 ↔ 响应翻译、过滤构造、sweep 网格展开
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
        libration_point=request.libration_point,
        jacobi_min=request.jacobi_min,
        jacobi_max=request.jacobi_max,
        amplitude_min_km=request.amplitude_min_km,
        amplitude_max_km=request.amplitude_max_km,
        has_cr3bp=request.has_cr3bp,
        has_ephemeris=request.has_ephemeris,
        status=None if request.status is None else request.status.value,
        tags=None if request.tags is None else tuple(request.tags),
    )


def _summary_kwargs(
    *,
    identity: dict[str, Any],
    classification: dict[str, Any],
    member_count: int,
) -> dict[str, Any]:
    """摘要公共字段（索引行与记录元数据两种来源共用）。"""
    return {
        "record_id": identity["record_id"],
        "created_at": identity["created_at"],
        "source_tool": identity["source_tool"],
        "source_record_id": identity["source_record_id"],
        "orbit_family": classification["orbit_family"],
        "libration_point": classification["libration_point"],
        "jacobi": classification["jacobi"],
        "amplitude": classification["amplitude"],
        "has_cr3bp": classification["has_cr3bp"],
        "has_ephemeris": classification["has_ephemeris"],
        "status": identity["status"],
        "cause": identity["cause"],
        "message": identity["message"],
        "member_count": member_count,
        "tags": identity["tags"],
        "note": identity["note"],
    }


def _summary_from_index(summary: dict[str, Any]) -> CatalogRecordSummary:
    """索引行摘要 → 响应模型。"""
    return CatalogRecordSummary(
        **_summary_kwargs(
            identity=summary,
            classification=summary["classification"],
            member_count=summary["member_count"],
        )
    )


def _summary_from_meta(meta: dict[str, Any]) -> CatalogRecordSummary:
    """记录元数据 → 摘要响应模型（member_count 与索引用同一份推导）。"""
    return CatalogRecordSummary(
        **_summary_kwargs(
            identity=meta,
            classification=meta["classification"],
            member_count=catalog_member_count(meta),
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
        members=meta["members"],
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
    非法点在此拒绝（ValueError → Facade 翻译 INVALID_PARAMS）。
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


class Facade:
    """e2m2e 唯一公开入口。

    ``Facade(config=Config(...))`` 构造注入配置（ADR 0014）。方法对应任务级
    能力，一档任务（稳定骨架，会增）：orbit_design / orbit_control /
    transfer_design / orbit_propagation / spacetime_transform。二档子任务
    （会增）标 ``mcp_exposed=True``，三档辅助标 ``False``。
    """

    def __init__(self, config: Config | None = None) -> None:
        """构造 Facade。

        Args:
            config: 运行配置（api/config.py Config），缺省从环境变量读。
        """
        self._config = config or Config()
        self._catalog_store: CatalogStore | None = None

    # ---- 轨道库 catalog 私有设施（ADR 0031）----

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

    def _auto_catalog(
        self, builder: Callable[[], tuple[dict, dict[str, np.ndarray]] | None]
    ) -> str | None:
        """产物自动入库（ADR 0031 决策 8）。

        无产物（族零成员、站保星历缺失）返回 None；入库失败抛
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

    def _load_record_ephemeris(self, record_id: str) -> tuple[dict, EphemerisTable]:
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

    # ---- 一档任务（mcp_exposed=True）----

    @mcp_exposed(request_model=DesignOrbitRequest)
    def design_orbit(self, **params) -> DesignOrbitResponse:
        """任务轨道设计（一档）。

        薄封装 ``algorithm/design/design_orbit``：Pydantic 校验 → 编排 → 结果
        翻译为 Response。算法层异常翻译为 ``OrbitError``。
        """
        try:
            request = DesignOrbitRequest(**params)
            from e2m2e.algorithm.design import design_orbit as _design

            result = _design(request, kernel_dir=self._config.kernel_dir)
            response = _design_result_to_response(result)
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
            raise OrbitError("DESIGN_FAILED", message, status=status, cause=cause) from exc
        response.record_id = self._auto_catalog(
            lambda: catalog_ingest.build_design_record(request, result)
        )
        return response

    @mcp_exposed(request_model=ControlOrbitRequest)
    def control_orbit(self, **params) -> ControlOrbitResponse:
        """轨道保持（一档）。

        薄封装 ``algorithm/station_keeping/control_orbit``。
        """
        try:
            request = ControlOrbitRequest(**params)
            from e2m2e.algorithm.station_keeping import control_orbit as _control

            source_meta: dict[str, Any] | None = None
            input_ephemeris = request.input_ephemeris
            if request.input_record_id is not None:
                source_meta, input_ephemeris = self._load_record_ephemeris(request.input_record_id)

            result = _control(
                input_ephemeris,
                control_mode=request.control_mode,
                is_nrho=request.is_nrho,
                special_mode=request.special_mode,
                control_interval=request.control_interval,
                feedback_arc=request.feedback_arc,
                special_crossings=request.special_crossings,
                num_controls=request.num_controls,
                num_monte_carlo=request.num_monte_carlo,
                output_step=request.output_step,
                position_accuracy=request.position_accuracy,
                velocity_accuracy=request.velocity_accuracy,
                thrust_angle_err=request.thrust_angle_err,
                thrust_mean=request.thrust_mean,
                thrust_rel_err=request.thrust_rel_err,
                thrust_abs_err=request.thrust_abs_err,
                thrust_min=request.thrust_min,
                thrust_max=request.thrust_max,
                thrust_total=request.thrust_total,
                srp_error_level=request.srp_error_level,
                perturbation=request.perturbation,
                dyb=request.dyb,
                earth_degree=request.earth_degree,
                moon_degree=request.moon_degree,
                real_perturbation=request.real_perturbation,
                real_dyb=request.real_dyb,
                real_earth_degree=request.real_earth_degree,
                real_moon_degree=request.real_moon_degree,
                kernel_dir=self._config.kernel_dir,
                engine_layout=request.engine_layout,
                momentum_interval=request.momentum_interval,
                srp_offset_m=request.srp_offset_m,
                spacecraft_mass=request.spacecraft_mass,
                srp_torque=request.srp_torque,
                tight_tolerance_km=request.tight_tolerance_km,
                tight_max_iter=request.tight_max_iter,
                special_damping_factor=request.special_damping_factor,
            )
            response = _control_result_to_response(result, mu=request.mu)
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
            raise OrbitError("CONTROL_FAILED", message, status=status, cause=cause) from exc
        response.record_id = self._auto_catalog(
            lambda: catalog_ingest.build_control_record(request, result, source_meta=source_meta)
        )
        return response

    @mcp_exposed(request_model=TransferDesignRequest)
    def transfer_design(self, **params) -> TransferDesignResponse:
        """转移轨道设计（一档）。

        薄封装 ``algorithm/transfer/transfer_orbit``：Pydantic 校验 → 编排 →
        结果翻译为 Response。
        """
        try:
            request = TransferDesignRequest(**params)
            from e2m2e.algorithm.transfer import EngineConfig, TliParams, transfer_orbit

            tli_params = TliParams(
                epoch=request.tli_epoch,
                parking_alt_km=request.parking_alt_km,
                inclination_deg=request.incl_deg,
                flight_path_angle_deg=request.flight_path_deg,
            )
            tof_range = (
                (float(request.tof_range[0]), float(request.tof_range[1]))
                if request.tof_range
                else None
            )
            engine_config = (
                EngineConfig(**request.engine_config) if request.engine_config is not None else None
            )
            result = transfer_orbit(
                request.transfer_type,
                target_ephemeris=request.target_ephemeris,
                tli_params=tli_params,
                tof_range=tof_range,
                target_orbit_radius_km=request.target_orbit_radius_km,
                lga_search_params=request.lga_search_params,
                wsb_search_params=request.wsb_search_params,
                engine_config=engine_config,
                initial_mass=request.initial_mass,
                n_segments=request.n_segments,
                target_oe=(
                    (request.target_oe[0], request.target_oe[1], request.target_oe[2])
                    if request.target_oe is not None
                    else None
                ),
                solver_method=request.solver_method,
                duration_days=request.duration_days,
                departure_state=(
                    np.asarray(request.departure_state, dtype=np.float64)
                    if request.departure_state is not None
                    else None
                ),
                target_state=(
                    np.asarray(request.target_state, dtype=np.float64)
                    if request.target_state is not None
                    else None
                ),
            )
            trajectory = (
                result.trajectory.tolist()
                if result.trajectory is not None and isinstance(result.trajectory, np.ndarray)
                else result.trajectory
            )
            status, cause, message = _result_triplet(result)
            return TransferDesignResponse(
                status=status,
                cause=cause,
                message=message,
                transfer_type=result.transfer_type,
                delta_v=result.delta_v,
                trajectory=trajectory,
                details=_details_to_dict(result.details),
            )
        except OrbitError:
            raise
        except (ValueError, TypeError) as exc:
            raise OrbitError("INVALID_PARAMS", str(exc)) from exc
        except NotImplementedError as exc:
            raise OrbitError(
                "NOT_IMPLEMENTED",
                str(exc),
                status=ConvergenceState.FAILED,
                cause=FailureCause.BACKEND_FAILURE,
            ) from exc
        except Exception as exc:
            status, cause, message = _exception_triplet(exc)
            raise OrbitError("TRANSFER_FAILED", message, status=status, cause=cause) from exc

    @mcp_exposed(request_model=PropagationRequest)
    def orbit_propagation(self, **params) -> PropagationResponse:
        """轨道预报（一档）。

        薄封装 ``algorithm/propagation/propagate_orbit``：Pydantic 校验 →
        传播 → EphemerisTable 翻译为 Response。
        """
        try:
            request = PropagationRequest(**params)
            from e2m2e.algorithm.propagation import propagate_orbit

            result = propagate_orbit(
                initial_state=request.initial_state,
                epoch=request.epoch,
                duration=request.duration,
                force_config=request.force_config,
                output_step=request.output_step,
                kernel_dir=self._config.kernel_dir,
            )
            ephemeris = result.ephemeris
            times_jd = ephemeris.times_jd_tdb
            assert times_jd is not None  # propagate_orbit 始终填充
            status, cause, message = _result_triplet(result)
            vel_km_s = ephemeris.velocity_mps / 1000.0
            return PropagationResponse(
                status=status,
                cause=cause,
                message=message,
                epoch_utc=str(request.epoch) if isinstance(request.epoch, str) else "",
                duration_sec=float(request.duration),
                output_step=float(request.output_step),
                n_points=len(ephemeris.year),
                time_sec=((times_jd - times_jd[0]) * SECONDS_PER_DAY).tolist(),
                times_jd_tdb=times_jd.tolist(),
                position_km=ephemeris.position_km.tolist(),
                velocity_km_s=vel_km_s.tolist(),
                final_state=ephemeris.position_km[-1].tolist() + vel_km_s[-1].tolist(),
            )
        except OrbitError:
            raise
        except (ValueError, TypeError) as exc:
            raise OrbitError("INVALID_PARAMS", str(exc)) from exc
        except NotImplementedError as exc:
            raise OrbitError(
                "NOT_IMPLEMENTED",
                str(exc),
                status=ConvergenceState.FAILED,
                cause=FailureCause.BACKEND_FAILURE,
            ) from exc
        except Exception as exc:
            status, cause, message = _exception_triplet(exc)
            raise OrbitError("PROPAGATION_FAILED", message, status=status, cause=cause) from exc

    @mcp_exposed(request_model=SpacetimeTransformRequest)
    def spacetime_transform(self, **params) -> SpacetimeTransformResponse:
        """时空坐标转换（一档）。

        薄封装 ``algorithm/coordinate/spacetime_convert``：Pydantic 校验 →
        逐条转换 → 结果翻译为 Response。
        """
        try:
            request = SpacetimeTransformRequest(**params)
            from e2m2e.algorithm.coordinate import spacetime_convert

            if len(request.states) != len(request.times):
                raise ValueError("states 与 times 长度必须一致")

            converted_states: list[list[float]] = []
            converted_times: list[float] = []
            for state, t in zip(request.states, request.times, strict=True):
                result = spacetime_convert(
                    request.transform_type,
                    state,
                    float(t),
                    et0_jd=request.et0_jd,
                    ephemeris_path=request.ephemeris_path,
                    kernel_dir=self._config.kernel_dir,
                )
                converted_states.append(result["state"].tolist())
                converted_times.append(float(result["time"]))

            status, cause, message = _result_triplet(result)
            return SpacetimeTransformResponse(
                status=status,
                cause=cause,
                message=message,
                states=converted_states,
                times=converted_times,
                transform_type=request.transform_type,
                details={
                    "n_states": len(converted_states),
                    "et0_jd": request.et0_jd,
                },
            )
        except OrbitError:
            raise
        except (ValueError, TypeError) as exc:
            raise OrbitError("INVALID_PARAMS", str(exc)) from exc
        except NotImplementedError as exc:
            raise OrbitError(
                "NOT_IMPLEMENTED",
                str(exc),
                status=ConvergenceState.FAILED,
                cause=FailureCause.BACKEND_FAILURE,
            ) from exc
        except Exception as exc:
            status, cause, message = _exception_triplet(exc)
            raise OrbitError("TRANSFORM_FAILED", message, status=status, cause=cause) from exc

    # ---- 二档子任务（mcp_exposed=True）----

    @mcp_exposed(request_model=FamilyGenerationRequest)
    def orbit_family_generation(self, **params) -> FamilyGenerationResponse:
        """轨道族生成（二档）。

        Pydantic 模型校验（#411）→ 按 orbit_type 分派到算法层族生成入
        口（#428、#502）→ 结构化错误。八族均已实现，成功返回统一容器
        ``FamilyGenerationResponse``（兼容 ``OrbitFamily`` 读取接口）；
        Lissajous 是拟周期参数采样，族上显式标注
        ``periodicity=quasi-periodic``。软失败使用同一响应保留部分族。
        """
        try:
            request = FamilyGenerationRequest(**params)
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
        response.record_id = self._auto_catalog(
            lambda: catalog_ingest.build_family_record(
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

    @mcp_exposed
    def orbit_stability(self, **params) -> Any:
        """稳定性分析（二档）：薄封装 algorithm/stability。"""
        from e2m2e.algorithm.stability import StabilityAnalysis

        orbit = params.get("orbit")
        dynamics = params.get("dynamics")
        if orbit is None:
            raise OrbitError("INVALID_PARAMS", "orbit 参数必填")
        return StabilityAnalysis(orbit=orbit, dynamics=dynamics).analyze()

    @mcp_exposed(status="placeholder")
    def transfer_search(self, **params) -> Any:
        """转移网格搜索（二档）。

        实现状态：占位。
        """
        raise NotImplementedError("Facade.transfer_search 待接入 algorithm/transfer/")

    @mcp_exposed(status="placeholder")
    def low_thrust_design(self, **params) -> Any:
        """小推力转移设计（二档）。

        实现状态：占位。
        """
        raise NotImplementedError("Facade.low_thrust_design 待接入 algorithm/transfer/")

    @mcp_exposed(status="placeholder")
    def manifold_analysis(self, **params) -> Any:
        """不变流形分析（二档）。

        实现状态：占位。
        """
        raise NotImplementedError("Facade.manifold_analysis 待接入 algorithm/manifold/")

    @mcp_exposed(status="placeholder")
    def low_energy_transfer(self, **params) -> Any:
        """低能转移（二档）。

        实现状态：占位。
        """
        raise NotImplementedError("Facade.low_energy_transfer 待接入 algorithm/transfer/")

    @mcp_exposed(status="placeholder")
    def relative_motion(self, **params) -> Any:
        """相对运动（二档）。

        实现状态：占位。待接入 algorithm/proximity 的 RelativeDynamics。
        chief/deputy 参数需映射为 TargetOrbit + dynamics 对象后接入。
        """
        raise NotImplementedError("Facade.relative_motion 待接入 algorithm/proximity/")

    # ---- 轨道库 catalog（ADR 0031，mcp_exposed=True）----

    @mcp_exposed(request_model=CatalogQueryRequest)
    def catalog_query(self, **params) -> CatalogQueryResponse:
        """多维过滤查询，返回摘要列表（不含数组段与请求快照）。"""
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
        """按 record_id 取完整记录（含数组段）；不存在抛 ``RECORD_NOT_FOUND``。"""
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
        """按 record_id 删除记录（文件与索引条目）；删除不可撤销。"""
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
        """写教学标注入 JSON 记录（随文件走）；tags 整体替换，note=None 保留。"""
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

    @mcp_exposed(request_model=CatalogPromoteRequest)
    def catalog_promote(self, **params) -> CatalogPromoteResponse:
        """把族成员提升为独立记录（source_record_id 指向所属族）。"""
        try:
            request = CatalogPromoteRequest(**params)
            record = self._open_catalog().promote_member(request.record_id, request.member_index)
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
        return CatalogPromoteResponse(
            status=ConvergenceState.CONVERGED,
            cause=FailureCause.NONE,
            message=f"成员 {request.member_index} 已提升为独立记录",
            record=_record_to_response(record),
        )

    @mcp_exposed(request_model=CatalogExportRequest)
    def catalog_export(self, **params) -> CatalogExportResponse:
        """把查询子集打包导出（标注随包）；包可直接作为库打开。"""
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

    @mcp_exposed(request_model=CatalogSweepRequest)
    def catalog_sweep(self, **params) -> CatalogSweepResponse:
        """参数空间扫描批量生成并入库（编排复用 ADR 0029 的 Rust 族生成）。

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
        record_ids: list[str] = []
        failed = 0
        for plan, outcome in zip(points, raw_outcomes, strict=True):
            record_id = self._ingest_sweep_outcome(outcome, plan.family_request)
            if record_id is not None:
                record_ids.append(record_id)
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
                    record_id=record_id,
                    generated_members=(
                        outcome.result.generated_members if outcome.result is not None else 0
                    ),
                )
            )
        succeeded = len(record_ids)
        soft_empty = len(points) - succeeded - failed
        message = f"扫描完成：{succeeded} 条记录入库，{failed} 点失败，共 {len(points)} 点"
        if soft_empty:
            message += f"（{soft_empty} 点软失败无成员产出）"
        return CatalogSweepResponse(
            status=ConvergenceState.CONVERGED,
            cause=FailureCause.NONE,
            message=message,
            points=outcomes,
            record_ids=record_ids,
            succeeded=succeeded,
            failed=failed,
        )

    def _ingest_sweep_outcome(self, outcome: Any, family_request: Any) -> str | None:
        """扫描单点结果入库；硬失败点（result=None）无记录。"""
        result = outcome.result
        if result is None:
            return None
        return self._auto_catalog(
            lambda: catalog_ingest.build_family_record(
                family_request,
                family=result.family,
                status=outcome.status,
                cause=outcome.cause,
                message=outcome.message,
                requested_members=result.requested_members,
                generated_members=result.generated_members,
            )
        )

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


def mcp_tools(facade: Facade) -> list[str]:
    """返回对 MCP 暴露的 Facade 方法名（纯派生，ADR 0014）。"""
    names: list[str] = []
    for name in dir(facade):
        if name.startswith("_"):
            continue
        attr = getattr(facade, name)
        if callable(attr) and getattr(attr, "mcp_exposed", False):
            names.append(name)
    return names


def tool_inventory(facade: Facade) -> list[ToolInfo]:
    """返回对 MCP 暴露的 Facade 工具及其实现元数据。"""
    inventory: list[ToolInfo] = []
    for name in mcp_tools(facade):
        method = getattr(facade, name)
        inventory.append(
            ToolInfo(
                name=name,
                mcp_exposed=True,
                status=method.tool_status,
                request_model=method.request_model,
            )
        )
    return inventory
