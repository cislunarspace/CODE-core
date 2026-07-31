"""Facade 门面：唯一公开顶级入口，粗粒度任务方法。

两层粒度（ADR 0014）：Facade 暴露粗粒度任务方法（人类/Agent 常用），算法层
保留细粒度 API（专家用）。MCP 工具 = Facade 方法全集（纯派生），方法带
``mcp_exposed`` 元数据控制是否对 MCP 暴露。

实现状态：一档任务已接入 algorithm/ 编排器（design_orbit/control_orbit），
二档子任务已接入已有算法（family/stability/proximity）；未实现能力
（transfer_design/orbit_propagation/spacetime_transform 等）保持占位。
"""

from __future__ import annotations

from typing import Any

from .config import Config
from .models import (
    ControlOrbitRequest,
    ControlOrbitResponse,
    DesignOrbitRequest,
    DesignOrbitResponse,
    OrbitError,
)

__all__ = ["Facade", "mcp_tools"]


def mcp_exposed(func):
    """标记 Facade 方法对 MCP 暴露（纯派生 + 元数据标记，ADR 0014）。"""
    func.mcp_exposed = True  # type: ignore[attr-defined]
    return func


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

            result = _design(
                request.orbit_type,
                amplitude=request.amplitude,
                phase=request.phase,
                collinear_point=request.collinear_point,
                north_south=request.north_south,
                perilune_height=request.perilune_height,
                amplitude_in=request.amplitude_in,
                amplitude_out=request.amplitude_out,
                phase_in=request.phase_in,
                phase_out=request.phase_out,
                epoch=request.epoch,
                duration=request.duration,
                output_step=request.output_step,
                correction_method=request.correction_method,
                kernel_dir=self._config.kernel_dir,
            )
            return DesignOrbitResponse(
                orbit_type=result.orbit_type,
                epoch_utc=result.epoch_utc,
                duration_day=result.duration_day,
                initial_state=result.initial_state.tolist(),
                cr3bp_jacobi=result.cr3bp_jacobi,
                correction_converged=result.correction.converged,
                correction_iterations=result.correction.iterations,
                force_config=result.force_config,
            )
        except OrbitError:
            raise
        except (ValueError, TypeError) as exc:
            raise OrbitError("INVALID_PARAMS", str(exc)) from exc
        except Exception as exc:
            raise OrbitError("DESIGN_FAILED", str(exc)) from exc

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
            )
            return ControlOrbitResponse(
                num_failed=result.num_failed,
                sk_statistic={
                    "rows": result.sk_statistic.rows.tolist(),
                    "num_failed": result.sk_statistic.num_failed,
                },
                maneuvers={
                    "mjd_tdb": result.maneuvers.mjd_tdb.tolist(),
                    "delta_v_mps": result.maneuvers.delta_v_mps.tolist(),
                },
            )
        except OrbitError:
            raise
        except (ValueError, TypeError) as exc:
            raise OrbitError("INVALID_PARAMS", str(exc)) from exc
        except Exception as exc:
            raise OrbitError("CONTROL_FAILED", str(exc)) from exc

    @mcp_exposed
    def transfer_design(self, **params) -> Any:
        """转移轨道设计（一档）。

        实现状态：占位（编排器 transfer_orbit 能力在规划中）。
        """
        raise NotImplementedError("Facade.transfer_design 待接入 algorithm/transfer/")

    @mcp_exposed
    def orbit_propagation(self, **params) -> Any:
        """轨道预报（一档）。

        实现状态：占位（propagate_orbit 能力在规划中）。
        """
        raise NotImplementedError("Facade.orbit_propagation 待接入 algorithm/propagation.py")

    @mcp_exposed
    def spacetime_transform(self, **params) -> Any:
        """时空坐标转换（一档）。

        实现状态：占位（统一转换入口待接入 algorithm/coordinate/）。
        """
        raise NotImplementedError("Facade.spacetime_transform 待接入 algorithm/coordinate/")

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
        """相对运动（二档）：薄封装 algorithm/proximity。"""
        from e2m2e.algorithm.proximity import RelativeDynamics

        chief = params.get("chief")
        deputy = params.get("deputy")
        if chief is None or deputy is None:
            raise OrbitError("INVALID_PARAMS", "chief/deputy 参数必填")
        return RelativeDynamics(chief=chief, deputy=deputy)


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
