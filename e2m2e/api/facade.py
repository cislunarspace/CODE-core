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
from typing import TYPE_CHECKING, Any

import numpy as np

from e2m2e.data.constants import SECONDS_PER_DAY
from e2m2e.data.templates import ConvergenceState, FailureCause

from .config import Config
from .models import (
    ControlOrbitRequest,
    ControlOrbitResponse,
    DesignOrbitRequest,
    DesignOrbitResponse,
    OrbitError,
    PropagationRequest,
    PropagationResponse,
    SpacetimeTransformRequest,
    SpacetimeTransformResponse,
    TransferDesignRequest,
    TransferDesignResponse,
)

__all__ = ["Facade", "mcp_tools"]

if TYPE_CHECKING:
    # 仅类型检查用：算法/数据层结果类型（运行时懒加载，见各 Facade 方法内 import）。
    from e2m2e.algorithm.design import OrbitDesignResult
    from e2m2e.algorithm.station_keeping import ControlOrbitResult
    from e2m2e.data.types.trajectory import EphemerisTable


def mcp_exposed(func):
    """标记 Facade 方法对 MCP 暴露（纯派生 + 元数据标记，ADR 0014）。"""
    func.mcp_exposed = True  # type: ignore[attr-defined]
    return func


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
    """读取算法异常携带的最终状态三元组。"""
    try:
        return exc.status, exc.cause, exc.message  # type: ignore[attr-defined]
    except AttributeError as missing:
        raise OrbitError(
            "RESULT_CONTRACT_FAILED",
            "算法异常缺少 status/cause/message 契约",
            status=ConvergenceState.FAILED,
            cause=FailureCause.BACKEND_FAILURE,
        ) from missing


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
    演进），跳过 ``raw_text``（原始文件文本，程序生成时为空串、读入时为
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
    """把 ``OrbitDesignResult`` 翻译为 ``DesignOrbitResponse``（含几何字段，#312）。

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
    """把 ``ControlOrbitResult`` 翻译为 ``ControlOrbitResponse``（含几何字段，#312）。

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

    # ---- 一档任务（mcp_exposed=True）----

    @mcp_exposed
    def design_orbit(self, **params) -> DesignOrbitResponse:
        """任务轨道设计（一档）。

        薄封装 ``algorithm/design/design_orbit``：Pydantic 校验 → 编排 → 结果
        翻译为 Response。算法层异常翻译为 ``OrbitError``。
        """
        try:
            request = DesignOrbitRequest(**params)
            from e2m2e.algorithm.design import design_orbit as _design

            result = _design(request, kernel_dir=self._config.kernel_dir)
            return _design_result_to_response(result)
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

    @mcp_exposed
    def control_orbit(self, **params) -> ControlOrbitResponse:
        """轨道保持（一档）。

        薄封装 ``algorithm/station_keeping/control_orbit``。
        """
        try:
            request = ControlOrbitRequest(**params)
            from e2m2e.algorithm.station_keeping import control_orbit as _control

            result = _control(
                request.input_ephemeris,
                control_mode=request.control_mode,
                is_nrho=request.is_nrho,
                special_mode=request.special_mode,
                num_controls=request.num_controls,
                num_monte_carlo=request.num_monte_carlo,
                output_step=request.output_step,
                kernel_dir=self._config.kernel_dir,
                engine_layout=request.engine_layout,
                momentum_interval=request.momentum_interval,
                srp_offset_m=request.srp_offset_m,
                spacecraft_mass=request.spacecraft_mass,
                srp_torque=request.srp_torque,
            )
            return _control_result_to_response(result, mu=request.mu)
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

    @mcp_exposed
    def transfer_design(self, **params) -> TransferDesignResponse:
        """转移轨道设计（一档）。

        薄封装 ``algorithm/transfer/transfer_orbit``：Pydantic 校验 → 编排 →
        结果翻译为 Response。
        """
        try:
            request = TransferDesignRequest(**params)
            from e2m2e.algorithm.transfer import TliParams, transfer_orbit

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
            result = transfer_orbit(
                request.transfer_type,
                target_ephemeris=request.target_ephemeris,
                tli_params=tli_params,
                tof_range=tof_range,
                target_orbit_radius_km=request.target_orbit_radius_km,
                lga_search_params=request.lga_search_params,
                wsb_search_params=request.wsb_search_params,
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

    @mcp_exposed
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

    @mcp_exposed
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

    @mcp_exposed
    def orbit_family_generation(self, **params) -> Any:
        """轨道族生成（二档）：薄封装 algorithm/family 六类初猜。"""
        from e2m2e.algorithm.family import registry

        orbit_type = params.pop("orbit_type")
        fn = registry.get(orbit_type)
        if fn is None:
            raise OrbitError("INVALID_PARAMS", f"未知轨道族类型: {orbit_type}")
        return fn(**params)

    @mcp_exposed
    def orbit_stability(self, **params) -> Any:
        """稳定性分析（二档）：薄封装 algorithm/stability。"""
        from e2m2e.algorithm.stability import StabilityAnalysis

        orbit = params.get("orbit")
        dynamics = params.get("dynamics")
        if orbit is None:
            raise OrbitError("INVALID_PARAMS", "orbit 参数必填")
        return StabilityAnalysis(orbit=orbit, dynamics=dynamics).analyze()

    @mcp_exposed
    def transfer_search(self, **params) -> Any:
        """转移网格搜索（二档）。

        实现状态：占位。
        """
        raise NotImplementedError("Facade.transfer_search 待接入 algorithm/transfer/")

    @mcp_exposed
    def low_thrust_design(self, **params) -> Any:
        """小推力转移设计（二档）。

        实现状态：占位。
        """
        raise NotImplementedError("Facade.low_thrust_design 待接入 algorithm/transfer/")

    @mcp_exposed
    def manifold_analysis(self, **params) -> Any:
        """不变流形分析（二档）。

        实现状态：占位。
        """
        raise NotImplementedError("Facade.manifold_analysis 待接入 algorithm/manifold/")

    @mcp_exposed
    def low_energy_transfer(self, **params) -> Any:
        """低能转移（二档）。

        实现状态：占位。
        """
        raise NotImplementedError("Facade.low_energy_transfer 待接入 algorithm/transfer/")

    @mcp_exposed
    def relative_motion(self, **params) -> Any:
        """相对运动（二档）。

        实现状态：占位。待接入 algorithm/proximity 的 RelativeDynamics。
        chief/deputy 参数需映射为 TargetOrbit + dynamics 对象后接入。
        """
        raise NotImplementedError("Facade.relative_motion 待接入 algorithm/proximity/")


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
