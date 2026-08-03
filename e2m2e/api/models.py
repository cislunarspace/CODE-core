"""公开数据模型（Pydantic，全手写）。

输入/输出/错误模型精雕参数单位、默认值、取值域（ADR 0014）。Pydantic 只在
api/ 边界，算法层用 numpy/dataclass。每个 Facade 方法一个 Request/Response，
外加统一错误模型 ``OrbitError``。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "OrbitError",
    "DesignOrbitRequest",
    "DesignOrbitResponse",
    "ControlOrbitRequest",
    "ControlOrbitResponse",
]


class OrbitError(Exception):
    """结构化错误（api/ 边界翻译，ADR 0014）。

    Attributes:
        code: 错误码（如 "NOT_IMPLEMENTED"/"NOT_CONVERGED"/"INVALID_PARAMS"）。
        message: 可读错误信息。
        details: 附加细节。
    """

    def __init__(
        self,
        code: str = "ERROR",
        message: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


class _ApiModel(BaseModel):
    """api 模型公共配置：允许任意内部字段，输出按声明序列化。"""

    model_config = ConfigDict(extra="forbid")


class DesignOrbitRequest(_ApiModel):
    """任务轨道设计输入（对齐 algorithm/design 的 design_orbit 参数）。"""

    orbit_type: str = Field(description="DRO/NRHO/Halo/Lissajous/L4/L5")
    amplitude: float | None = Field(default=None, ge=1.0, le=110000.0)
    phase: float | None = Field(default=None, ge=0.0, le=1.0)
    collinear_point: int | None = Field(default=None, ge=1, le=3)
    north_south: int | None = Field(default=None, ge=1, le=2)
    perilune_height: float | None = Field(default=None, ge=100.0, le=10000.0)
    amplitude_in: float | None = Field(default=None, gt=0.0, le=100000.0)
    amplitude_out: float | None = Field(default=None, gt=0.0, le=76000.0)
    phase_in: float | None = Field(default=None, ge=0.0, le=1.0)
    phase_out: float | None = Field(default=None, ge=0.0, le=1.0)
    epoch: Any = Field(default=(2024, 1, 1, 0, 0, 0.0), description="[年,月,日,时,分,秒] 或 ISO")
    duration: float = Field(default=1.0, gt=0.0, le=20.0)
    output_step: float = Field(default=3600.0, gt=0.0)
    correction_method: str = Field(default="two_level")


class DesignOrbitResponse(_ApiModel):
    """任务轨道设计输出。"""

    orbit_type: str
    epoch_utc: str
    duration_day: float
    initial_state: list[float]
    cr3bp_jacobi: float
    correction_converged: bool
    correction_iterations: int
    force_config: dict[str, Any]


class ControlOrbitRequest(_ApiModel):
    """轨道保持输入（对齐 algorithm/station_keeping 的 control_orbit 参数）。"""

    input_ephemeris: Any = Field(description="标称轨道星历路径或 EphemerisTable")
    control_mode: int = Field(default=1, ge=1, le=6)
    is_nrho: int = Field(default=0, ge=0, le=1)
    special_mode: int = Field(default=1, ge=1, le=2)
    num_controls: int = Field(default=120, ge=1)
    num_monte_carlo: int = Field(default=5, ge=1)
    output_step: float = Field(default=86400.0, gt=0.0)
    engine_layout: Any = Field(default=None, description="EngineLayout（角动量管理 4-6 必填）")
    momentum_interval: float = Field(default=5.0, gt=0.0, description="角动量卸载间隔（天）")
    srp_offset_m: list[float] | None = Field(default=None, description="SRP 压心偏移 [x,y,z]（m）")
    spacecraft_mass: float = Field(default=1000.0, gt=0.0, description="航天器质量（kg）")
    srp_torque: list[float] | None = Field(default=None, description="常值 SRP 力矩 [τx,τy,τz]（N·m）")


class ControlOrbitResponse(_ApiModel):
    """轨道保持输出。"""

    num_failed: int
    sk_statistic: dict[str, Any]
    maneuvers: dict[str, Any]
