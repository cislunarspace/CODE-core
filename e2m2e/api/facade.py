"""Facade 门面：任务级入口与暴露类组合根（ADR 0043）。

接口层暴露三个类（ADR 0043）：``Facade`` 只留五个任务级能力
（design_orbit/control_orbit/transfer_design/orbit_propagation/
spacetime_transform）；``e2m2e.api.catalog.Catalog`` 承担轨道库数据管理与
族生成；``e2m2e.api.spatiography.Spatiography`` 承担分区分析。
``Facade`` 是组合根：``Facade().catalog`` / ``Facade().spatiography``
向进程内调用方交出另外两类；唯一工具清单扫描 ``Facade().exposed_apis``
——MCP/CLI/sidecar 都从这一份清单派生（ADR 0014 决策 2，扫描根由
ADR 0043 拓宽）。方法带 ``mcp_exposed`` 元数据；算法层保留细粒度 API
（专家用）。
"""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from e2m2e.data.constants import SECONDS_PER_DAY
from e2m2e.data.templates import ConvergenceState, FailureCause

from . import catalog_ingest
from .config import Config
from .models import (
    ControlOrbitRequest,
    ControlOrbitResponse,
    DesignOrbitRequest,
    DesignOrbitResponse,
    ManeuverEvent,
    OrbitError,
    PropagationRequest,
    PropagationResponse,
    SpacetimeTransformRequest,
    SpacetimeTransformResponse,
    TransferCandidate,
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


# ---- 长任务进度（#576 Phase 1）----

#: 长任务 Facade 进度回调形状：``cb(fraction, message=None)``，fraction ∈
#: [0, 1] 单调不减（0.0 = 开始，1.0 = 完成）。MCP 层经 progressToken 把
#: 它桥接到 ``notifications/progress``；进度语义按任务定义：转移搜索按
#: 网格任务（仅 WSB 后端当前暴露 delta 回调），族生成为阶段级（单次
#: Rust 调用，逐成员进度待 Rust 侧通道，见 #576 Phase 2 记录）。
ProgressCallback = Callable[[float, str | None], None]


def _emit_progress(
    callback: ProgressCallback | None, fraction: float, message: str | None = None
) -> None:
    """进度上报安全垫：无回调直通；回调异常吞掉（进度失败不中断计算）。"""
    if callback is None:
        return
    with contextlib.suppress(Exception):
        callback(fraction, message)


def _wsb_search_progress(
    callback: ProgressCallback | None, request: TransferDesignRequest
) -> Callable[[int], None] | None:
    """WSB 网格搜索 delta 回调 → fraction 适配器（#576）。

    WSB 是当前唯一暴露搜索进度回调的转移路径（Rust 侧每完成一个
    ``(sun_phase, tof)`` 网格任务发一次 delta）；映射到 (0.1, 0.9) 区间，
    起止两端由调用方上报。非 WSB 或无回调返回 None（零开销直通）。
    """
    if callback is None or request.transfer_type != "WSB":
        return None
    params = request.wsb_search_params
    if params is None:
        from e2m2e.algorithm.transfer import WsbSearchParams

        params = WsbSearchParams()
    total = max(params.n_sun_phase * params.n_tof, 1)
    seen = [0]

    def on_delta(delta: int) -> None:
        seen[0] += delta
        done = min(seen[0], total)
        _emit_progress(
            callback,
            0.1 + 0.8 * done / total,
            f"WSB 网格搜索 {done}/{total}",
        )

    return on_delta


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
    """把 ``OrbitDesignResult`` 翻译为 ``DesignOrbitResponse``（含几何字段）。

    纯翻译、无副作用、不依赖 SPICE。ELFO 场景下 ``cr3bp_orbit`` /
    ``correction`` 为 None，对应字段输出默认值（mu=None、states/times 空、
    correction_iterations=0）。CR3BP 场景下从 ``cr3bp_orbit`` 提取
    ``states`` / ``times`` / ``mu``，并对参考周期轨道做分类学实测打标
    （#581，ADR 0042；整条轨迹消费，无额外传播）。
    """
    from e2m2e.algorithm.orbit_taxonomy import classify_orbit

    cr3bp_orbit = result.cr3bp_orbit
    correction = result.correction
    taxonomy_labels: list[str] = []
    if cr3bp_orbit is not None:
        mu = getattr(cr3bp_orbit.system, "mu", None)
        states = cr3bp_orbit.states.tolist()
        times = cr3bp_orbit.times.tolist()
        taxonomy_labels = list(
            classify_orbit(cr3bp_orbit.states, cr3bp_orbit.times, mu=mu).canonical_labels
        )
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
        taxonomy_labels=taxonomy_labels,
        ephemeris=_ephemeris_to_dict(result.ephemeris),
        drift_e=result.drift_e,
        drift_aop_deg=result.drift_aop_deg,
        drift_rp_km=result.drift_rp_km,
        secular_aop_rate_deg_per_year=result.secular_aop_rate_deg_per_year,
    )


def _control_result_to_response(
    result: ControlOrbitResult, *, mu: float | None
) -> ControlOrbitResponse:
    """把 ``ControlOrbitResult`` 翻译为 ``ControlOrbitResponse``（含几何字段）。

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


class Facade:
    """任务级入口与暴露类组合根（ADR 0043 决策 1）。

    ``Facade(config=Config(...))`` 构造注入配置（ADR 0014），只承载五个
    任务级方法；轨道库与分区分析分别经 ``self.catalog`` /
    ``self.spatiography`` 暴露（决策 2/3），工具清单经 ``exposed_apis``
    跨类扫描（决策 5）。
    """

    def __init__(self, config: Config | None = None) -> None:
        """构造 Facade。

        Args:
            config: 运行配置（api/config.py Config），缺省从环境变量读。
        """
        self._config = config or Config()
        # 懒导入避免 facade ↔ catalog/spatiography 模块级循环
        # （两类从本模块取 mcp_exposed 与共享翻译小件）。
        from .catalog import Catalog
        from .spatiography import Spatiography

        self.catalog = Catalog(self._config)
        self.spatiography = Spatiography()

    @property
    def config(self) -> Config:
        """运行配置（只读视图）。

        长任务工具经 worker 子进程执行时，配置经
        :meth:`Config.to_payload` 随请求下发（#601），子进程用它重建
        Facade——构造注入对全部工具生效，不再从环境变量静默重建。
        """
        return self._config

    @property
    def exposed_apis(self) -> tuple[Any, ...]:
        """暴露类实例全集（ADR 0043 决策 5；``tool_inventory`` 的扫描根）。"""
        return (self, self.catalog, self.spatiography)

    # ---- 一档任务（mcp_exposed=True）----

    @mcp_exposed(request_model=DesignOrbitRequest)
    def design_orbit(self, **params) -> DesignOrbitResponse:
        """Mission orbit design (tier 1). / 任务轨道设计（一档）。

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
        response.record_id = self.catalog.auto_ingest(
            lambda: catalog_ingest.build_design_record(request, result)
        )
        return response

    @mcp_exposed(request_model=ControlOrbitRequest)
    def control_orbit(self, **params) -> ControlOrbitResponse:
        """Station-keeping Monte Carlo simulation (tier 1). / 轨道保持（一档）。

        薄封装 ``algorithm/station_keeping/control_orbit``。
        """
        try:
            request = ControlOrbitRequest(**params)
            from e2m2e.algorithm.station_keeping import control_orbit as _control

            source_meta: dict[str, Any] | None = None
            input_ephemeris = request.input_ephemeris
            if request.input_record_id is not None:
                source_meta, input_ephemeris = self.catalog.load_record_ephemeris(
                    request.input_record_id
                )

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
        response.record_id = self.catalog.auto_ingest(
            lambda: catalog_ingest.build_control_record(request, result, source_meta=source_meta)
        )
        return response

    @mcp_exposed(request_model=TransferDesignRequest)
    def transfer_design(
        self, progress_callback: ProgressCallback | None = None, **params
    ) -> TransferDesignResponse:
        """Transfer design (tier 1). / 转移轨道设计（一档）。

        薄封装 ``algorithm/transfer/transfer_orbit``：Pydantic 校验 → 编排 →
        结果翻译为 Response。``progress_callback(fraction, message)`` 上报
        长任务进度（#576 Phase 1：WSB 后端映射网格任务，其余后端仅起止）。
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
            _emit_progress(progress_callback, 0.0, f"转移设计开始：{request.transfer_type}")
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
                progress_callback=_wsb_search_progress(progress_callback, request),
                top_n=request.top_n,
            )
            trajectory = (
                result.trajectory.tolist()
                if result.trajectory is not None and isinstance(result.trajectory, np.ndarray)
                else result.trajectory
            )
            trajectory_times = (
                result.trajectory_times.tolist()
                if result.trajectory_times is not None
                and isinstance(result.trajectory_times, np.ndarray)
                else result.trajectory_times
            )
            # 惯性段（#584）：旧结果对象（含测试替身）无该字段时视为缺位
            gcrs_segment = getattr(result, "trajectory_gcrs_km", None)
            trajectory_gcrs_km = (
                gcrs_segment.tolist() if isinstance(gcrs_segment, np.ndarray) else gcrs_segment
            )
            status, cause, message = _result_triplet(result)
            response = TransferDesignResponse(
                status=status,
                cause=cause,
                message=message,
                transfer_type=result.transfer_type,
                delta_v=result.delta_v,
                trajectory=trajectory,
                trajectory_times=trajectory_times,
                trajectory_gcrs_km=trajectory_gcrs_km,
                # __post_init__ 派生保证取值在 Literal 集内；显式覆盖属 ADR 0040
                # 扩展路径，越界值由响应构造期的 pydantic 校验兜底。
                state_frame=cast(
                    Literal["synodic_barycentric_km", "force_model_state"],
                    result.state_frame,
                ),
                maneuver_events=[
                    ManeuverEvent(
                        kind=event.kind,
                        t_sec=event.t_sec,
                        dv_km_s=event.dv_km_s,
                        note=event.note,
                    )
                    for event in result.maneuver_events
                ],
                candidates=[
                    TransferCandidate(
                        delta_v_km_s=cand.delta_v_km_s,
                        tli_epoch=cand.tli_epoch,
                        tof_sec=cand.tof_sec,
                        trajectory=(
                            cand.trajectory.tolist()
                            if isinstance(cand.trajectory, np.ndarray)
                            else cand.trajectory
                        ),
                        trajectory_times=(
                            cand.trajectory_times.tolist()
                            if isinstance(cand.trajectory_times, np.ndarray)
                            else cand.trajectory_times
                        ),
                        state_frame=cand.state_frame,
                        selected=cand.selected,
                        refined=cand.refined,
                    )
                    for cand in (getattr(result, "candidates", ()) or ())
                ]
                or None,
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
        _emit_progress(progress_callback, 1.0, "转移设计完成")
        response.record_id = self.catalog.auto_ingest(
            lambda: catalog_ingest.build_transfer_record(request, result)
        )
        return response

    @mcp_exposed(request_model=PropagationRequest)
    def orbit_propagation(self, **params) -> PropagationResponse:
        """Orbit prediction (tier 1). / 轨道预报（一档）。

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
        """Spacetime coordinate conversion (tier 1). / 时空坐标转换（一档）。

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


def mcp_tools(facade: Any) -> list[str]:
    """返回单个接口类实例上对 MCP 暴露的方法名（纯派生，ADR 0014）。"""
    names: list[str] = []
    for name in dir(facade):
        if name.startswith("_"):
            continue
        attr = getattr(facade, name)
        if callable(attr) and getattr(attr, "mcp_exposed", False):
            names.append(name)
    return names


def tool_inventory(facade: Any) -> list[ToolInfo]:
    """返回对 MCP 暴露的工具及其元数据（多类扫描，ADR 0043 决策 5）。

    扫描根是暴露类实例全集：组合根（``Facade``）经 ``exposed_apis`` 给出
    三类；无该属性的单一对象（测试桩、单独构造的领域类）按自身扫描。
    清单仍单一来源，MCP/CLI/sidecar 消费不变。
    """
    apis = getattr(facade, "exposed_apis", None) or (facade,)
    inventory: list[ToolInfo] = []
    for api in apis:
        for name in mcp_tools(api):
            method = getattr(api, name)
            inventory.append(
                ToolInfo(
                    name=name,
                    mcp_exposed=True,
                    status=method.tool_status,
                    request_model=method.request_model,
                )
            )
    return inventory


def resolve_tool_method(api_root: Any, name: str) -> Any | None:
    """工具名 → 属主实例上的绑定方法（扫描根与 ``tool_inventory`` 同规则）。

    跨暴露类解析（ADR 0043 决策 5）的唯一入口：MCP 规格、CLI 子命令派生
    都经它找方法，不各自再写一份属主搜索。无属主返回 None。
    """
    for api in getattr(api_root, "exposed_apis", None) or (api_root,):
        attr = getattr(api, name, None)
        if attr is not None and getattr(attr, "mcp_exposed", False):
            return attr
    return None
