"""力模型容器与 Rust 积分器传播实现。"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NoReturn

import numpy as np
import numpy.typing as npt

from e2m2e.exceptions import RustExtensionUnavailableError
from e2m2e.integrators import RkMethod, require_rust_extension

from .physical_model import PhysicalModel
from .thrust import BurnApplication, ImpulsiveBurn


@dataclass(frozen=True)
class ForceEntry:
    """容器内单个力模型的注册记录。"""

    name: str
    force: PhysicalModel
    enabled: bool = True


class ForceModel:
    """聚合多个 PhysicalModel 并完成传播的动力学容器。

    不继承 ``Dynamics``：``Dynamics`` 是 CR3BP/Ephemeris 的基类，其
    ``propagate()`` 基于 ``scipy.solve_ivp`` 与 STM 模板方法；ForceModel 用
    Rust ``rk_step`` 单步步进器实现自适应传播，支持 ``with_stm=True``
    （各力解析雅可比叠加、无雅可比的力用有限差分兜底），不支持 Jacobi。
    ``Dynamics`` 的模板方法契约无法兑现（``propagate`` 全部重写、STM/Jacobi
    抛 ``NotImplementedError``），形式继承属 LSP 违反（假继承）。
    """

    DEFAULT_TOLERANCE = 1e-12
    DEFAULT_MAX_STEP = 60.0  # 秒，用于物理单位传播
    STATE_DIM = 6  # 状态向量维度 [x, y, z, vx, vy, vz]
    STM_DIM = STATE_DIM + STATE_DIM * STATE_DIM  # 42 = 6 状态 + 36 STM 展平

    def __init__(
        self,
        system: Any,
        forces: list[PhysicalModel] | None = None,
    ) -> None:
        """初始化 ForceModel。

        Args:
            system: 动力学系统，必须提供 ``coordinate_system``。
            forces: 初始力模型列表，默认空列表。
        """
        self.system = system
        if getattr(system, "coordinate_system", None) is None:
            raise ValueError("ForceModel requires system.coordinate_system to be set.")
        # 积分器配置（与 Dynamics 同名属性，供 propagate 与外部配置使用）
        self.rtol: float = self.DEFAULT_TOLERANCE
        self.atol: float = self.DEFAULT_TOLERANCE
        self.max_step: float = self.DEFAULT_MAX_STEP
        self.last_trajectory: tuple[np.ndarray, np.ndarray] | None = None
        self._entries: tuple[ForceEntry, ...] = ()
        if forces is not None:
            for force in forces:
                self.add_force(force)

    @property
    def forces(self) -> tuple[PhysicalModel, ...]:
        """当前聚合的力模型，只读（含已 disable 的项）。"""
        return tuple(entry.force for entry in self._entries)

    def add_force(self, force: PhysicalModel, name: str | None = None) -> None:
        """添加一个力模型。

        Args:
            force: 待添加的力模型。
            name: 力模型的名字。缺省时按类名自动生成，遇同类自动消歧
                （``Foo``、``Foo_2``、``Foo_3``…）。显式给出且与已有名字
                冲突时抛 ``ValueError``。
        """
        if not isinstance(force, PhysicalModel):
            raise TypeError(f"force must be a PhysicalModel, got {type(force).__name__}")
        if name is None:
            name = self._auto_name(force)
        elif any(entry.name == name for entry in self._entries):
            raise ValueError(f"force name {name!r} already exists in ForceModel")
        self._entries = self._entries + (ForceEntry(name, force, True),)

    def _auto_name(self, force: PhysicalModel) -> str:
        """按类名生成唯一名字，遇同类自动加序号后缀。

        若已存在 ``Foo`` 或 ``Foo_N`` 中的任意一个，新名字取该系列最大
        序号加 1，保证整个 ``Foo`` 系列名字连续且唯一。
        """
        base = type(force).__name__
        pattern = re.compile(rf"^{re.escape(base)}(_(\d+))?$")
        max_suffix = 0
        for entry in self._entries:
            match = pattern.match(entry.name)
            if match:
                suffix = int(match.group(2)) if match.group(2) else 1
                max_suffix = max(max_suffix, suffix)
        if max_suffix == 0:
            return base
        return f"{base}_{max_suffix + 1}"

    def get_force(self, name: str) -> PhysicalModel:
        """按名取力模型；不存在抛 ``KeyError``。"""
        for entry in self._entries:
            if entry.name == name:
                return entry.force
        raise KeyError(name)

    def list_forces(self) -> list[ForceEntry]:
        """返回所有力模型的注册记录（含已 disable 的项）。

        与 ``forces`` 属性的区别：本方法暴露 ``name`` 与 ``enabled`` 两个维度。
        """
        return list(self._entries)

    def enable(self, name: str) -> None:
        """按名启用一个力模型；不存在抛 ``KeyError``。"""
        self._set_enabled(name, True)

    def disable(self, name: str) -> None:
        """按名禁用一个力模型（跳过加速度计算，但保留在容器内）。

        不存在抛 ``KeyError``。
        """
        self._set_enabled(name, False)

    def _set_enabled(self, name: str, enabled: bool) -> None:
        """翻转指定名字力模型的 enabled 标志（不可变替换）。"""
        if not any(entry.name == name for entry in self._entries):
            raise KeyError(name)
        self._entries = tuple(
            ForceEntry(entry.name, entry.force, enabled) if entry.name == name else entry
            for entry in self._entries
        )

    # --- 配置驱动（ADR 0004）---

    _CONFIG_VERSION = 1

    def to_config(self) -> dict[str, Any]:
        """序列化为配置字典 ``{version, forces: [...]}``。

        每条力经 ``force_config.serialize_force`` 转 ``{type, params}``，
        容器补 ``name`` 与 ``enabled``。round-trip 契约见 ADR 0004。
        """
        from .force_config import serialize_force

        forces_config: list[dict[str, Any]] = []
        for entry in self._entries:
            single = serialize_force(entry.force)
            forces_config.append(
                {
                    "name": entry.name,
                    "type": single["type"],
                    "enabled": entry.enabled,
                    "params": single["params"],
                }
            )
        return {"version": self._CONFIG_VERSION, "forces": forces_config}

    @classmethod
    def from_config(cls, config: dict[str, Any], system: Any) -> ForceModel:
        """从配置字典构建 ``ForceModel``。

        校验 ``version``，逐条 ``force_config.build_force`` 构造并按 ``name``
        注册；``enabled: false`` 的条目构造后立即 disable。
        """
        from .force_config import build_force

        version = config.get("version")
        if version != cls._CONFIG_VERSION:
            raise ValueError(
                f"unsupported config version {version!r}; expected {cls._CONFIG_VERSION}"
            )
        fm = cls(system)
        for entry in config.get("forces", []):
            force = build_force(entry["type"], entry.get("params", {}))
            fm.add_force(force, name=entry["name"])
            if not entry.get("enabled", True):
                fm.disable(entry["name"])
        return fm

    def remove_force(self, index: int | str) -> None:
        """移除一个力模型（按索引或名字）。

        Args:
            index: 整数索引，或力模型注册名。

        Raises:
            ValueError: 名字不存在时。
            IndexError: 索引越界时。
        """
        entries = list(self._entries)
        if isinstance(index, int):
            del entries[index]
        else:
            for i, entry in enumerate(entries):
                if entry.name == index:
                    del entries[i]
                    break
            else:
                raise ValueError(f"force name {index!r} not found in ForceModel")
        self._entries = tuple(entries)

    # ── Rust propagate_compiled 快速路径（spice feature 启用时） ──

    # Rust STM 路径不支持的力模型类型：`acceleration_and_jacobian` 对
    # RelativisticCorrection 和 VariableMassFiniteBurn 返回 Err（compiled.rs `_ => Err`），
    # 其余力（含 SRP）都有解析或 Rust 内有限差分雅可比。
    _STM_UNSUPPORTED_TYPES = ("RelativisticCorrection", "VariableMassFiniteBurn")

    def _raise_to_rust_spec_none(self, force_name: str) -> NoReturn:
        """``to_rust_spec`` 返回 None 时按原因分流报错（ADR 0020 决策 4）。

        ``system`` 无 ``spice`` （资源缺失，环境没搭好）→
        :class:`RustExtensionUnavailableError`，措辞含修复指引；
        ``system`` 有 ``spice`` 但力仍无 Rust 实现（能力缺失，如非 EARTH
        drag、特定潮汐模式）→ ``NotImplementedError``，措辞指明是模型限制。
        """
        if getattr(self.system, "spice", None) is None:
            raise RustExtensionUnavailableError(
                f"force {force_name} 需要 SPICE（system.spice 缺失）；请 make setup "
                "下载内核或加载 .tls + .bsp（资源缺失，非能力限制）。"
            )
        raise NotImplementedError(
            f"force {force_name} 无 Rust 实现（to_rust_spec 返回 None）；"
            "该力不支持 Rust 编译传播（能力缺失）。"
        )

    def _require_rust_capability(self, *, stm: bool) -> None:
        """校验 Rust 扩展可用且所有启用力模型支持 Rust 编译；不满足即显式报错。

        issue #378：核心传播一律走编译 Rust，扩展不可用（抛
        :class:`RustExtensionUnavailableError`）或某 force 无 ``to_rust_spec``
        （按原因分流：无 spice → ``RustExtensionUnavailableError``；能力缺失
        → ``NotImplementedError``，ADR 0020 决策 4）时不再静默回退
        Python/scipy。

        Args:
            stm: 目标路径是否含 STM（``propagate_compiled_stm_py``）。
        """
        if stm:
            require_rust_extension("propagate_compiled_stm_py")
        else:
            require_rust_extension("propagate_compiled")
        for entry in self._entries:
            if not entry.enabled:
                continue
            force = entry.force
            if stm and type(force).__name__ in self._STM_UNSUPPORTED_TYPES:
                raise NotImplementedError(
                    f"force {type(force).__name__}（name={entry.name!r}）不支持 STM "
                    "传播（Rust acceleration_and_jacobian 对该力返回 Err）。"
                    "issue #378：不再回退 Python FD 路径。"
                )
            if force.to_rust_spec(self.system) is None:
                self._raise_to_rust_spec_none(type(force).__name__)

    def _propagate_via_rust(
        self,
        y0: npt.NDArray[np.floating],
        t0: float,
        tf: float,
        t_eval: npt.ArrayLike,
        tol: float,
        h_init: float,
        max_steps: int,
    ) -> dict[str, Any]:
        """走 Rust propagate_compiled（零跨界）。

        自动序列化所有 force，调 Rust 入口。返回格式与 Python propagate 一致。
        """
        require_rust_extension("propagate_compiled")
        from e2m2e.integrators import RkMethod, propagate_compiled

        forces_py = []
        for entry in self._entries:
            if not entry.enabled:
                continue
            spec = entry.force.to_rust_spec(self.system)
            if spec is None:
                # _require_rust_capability 已过滤，理论不会到这
                self._raise_to_rust_spec_none(entry.force.__class__.__name__)
            forces_py.append(spec)

        observer = getattr(self.system, "origin", "EARTH")
        t_eval_list = [float(x) for x in np.asarray(t_eval).flat]
        # RkMethod：ForceModel.propagate 默认 PD45（见 propagate 签名 method 参数）
        # 这里固定 PD45；如需支持其它 method，可暴露参数
        result = propagate_compiled(
            RkMethod.PD45,
            float(t0),
            [float(x) for x in y0],
            float(h_init),
            float(tol),
            t_eval_list,
            observer,
            forces_py,
            int(max_steps),
        )
        self.last_trajectory = (np.asarray(result["time"]), np.asarray(result["states"]))
        return {
            "time": np.asarray(result["time"]),
            "states": np.asarray(result["states"]),
            "terminal_event_index": None,
            "n_steps": result["n_steps"],
            "n_rejected": result["n_rejected"],
        }

    def _has_variable_mass_thrust(self) -> bool:
        """是否含启用的 :class:`VariableMassFiniteBurn`。"""
        from .thrust import VariableMassFiniteBurn

        return any(
            isinstance(entry.force, VariableMassFiniteBurn)
            for entry in self._entries
            if entry.enabled
        )

    def _propagate_lowthrust(
        self,
        initial_state: npt.ArrayLike,
        t0: float,
        tf: float,
        t_eval: npt.ArrayLike | None,
        with_stm: bool,
        max_steps: int,
    ) -> dict[str, Any]:
        """可变质量低推力 7D 传播 ``[r, v, m]``。

        从 force 列表拆出 :class:`VariableMassFiniteBurn` 的推力规格（throttle/
        方向，本期仅常量），其余 force 收集成非推力 force 列表，调 Rust
        ``propagate_compiled_lowthrust``。``with_stm=True`` 暂不支持（43D 含
        STM 的受控动力学版本已存在于 Rust 侧，留待求解器期次接线）。
        """
        from .thrust import VariableMassFiniteBurn

        if with_stm:
            raise NotImplementedError(
                "VariableMassFiniteBurn propagation does not support with_stm yet"
            )

        y0 = np.asarray(initial_state, dtype=float)
        if y0.shape != (self.STATE_DIM + 1,):
            raise ValueError(
                f"VariableMassFiniteBurn requires a 7D initial_state "
                f"[r, v, m], got shape {y0.shape}"
            )

        # 拆推力 force 与其余 force
        thrust_spec: tuple | None = None
        forces_py: list[tuple] = []
        for entry in self._entries:
            if not entry.enabled:
                continue
            if isinstance(entry.force, VariableMassFiniteBurn):
                spec = entry.force.to_rust_spec(self.system)
                if spec is None:
                    raise NotImplementedError(
                        "VariableMassFiniteBurn with a callable direction cannot use "
                        "the Rust path; use a fixed direction."
                    )
                if thrust_spec is not None:
                    raise ValueError("only one VariableMassFiniteBurn is supported per propagation")
                thrust_spec = spec
            else:
                spec = entry.force.to_rust_spec(self.system)
                if spec is None:
                    self._raise_to_rust_spec_none(entry.force.__class__.__name__)
                forces_py.append(spec)

        if thrust_spec is None:
            raise RuntimeError("VariableMassFiniteBurn not found (should not happen)")

        # thrust_spec = ("low_thrust_variable", t_max, isp, throttle, dx, dy, dz)
        t_max = float(thrust_spec[1])
        isp = float(thrust_spec[2])
        throttle = float(thrust_spec[3])
        direction = (float(thrust_spec[4]), float(thrust_spec[5]), float(thrust_spec[6]))

        from e2m2e.integrators import RkMethod

        require_rust_extension("propagate_compiled_lowthrust")
        from e2m2e.integrators import propagate_compiled_lowthrust

        t_eval_arr = self._prepare_t_eval(t0, tf, t_eval)
        observer = getattr(self.system, "origin", "EARTH")
        tol = float(self.rtol)
        h = self._estimate_initial_step(y0[: self.STATE_DIM], t0, tf)

        result = propagate_compiled_lowthrust(
            RkMethod.PD45,
            float(t0),
            [float(x) for x in y0],
            float(h),
            tol,
            [float(x) for x in np.asarray(t_eval_arr).flat],
            observer,
            forces_py,
            (t_max, isp, throttle, direction[0], direction[1], direction[2]),
            int(max_steps),
        )
        self.last_trajectory = (np.asarray(result["time"]), np.asarray(result["states"]))
        return {
            "time": np.asarray(result["time"]),
            "states": np.asarray(result["states"]),
            "terminal_event_index": None,
            "n_steps": result["n_steps"],
            "n_rejected": result["n_rejected"],
        }

    def _propagate_via_rust_stm(
        self,
        y0: npt.NDArray[np.floating],
        t0: float,
        tf: float,
        t_eval: npt.ArrayLike,
        tol: float,
        max_steps: int,
    ) -> dict[str, Any]:
        """走 Rust propagate_compiled_stm_py（含 STM，消除 cspice 隔离）。

        自动序列化所有 force，调 Rust STM 入口。返回格式与 Python propagate 一致。
        """
        require_rust_extension("propagate_compiled_stm_py")
        from e2m2e.integrators import propagate_compiled_stm_py

        forces_py = []
        for entry in self._entries:
            if not entry.enabled:
                continue
            spec = entry.force.to_rust_spec(self.system)
            if spec is None:
                self._raise_to_rust_spec_none(entry.force.__class__.__name__)
            forces_py.append(spec)

        observer = getattr(self.system, "origin", "EARTH")
        t_eval_list = [float(x) for x in np.asarray(t_eval).flat]

        result = propagate_compiled_stm_py(
            observer,
            forces_py,
            (float(t0), float(tf)),
            t_eval_list,
            [float(x) for x in y0],
            float(tol),
            float(tol),
            float(self.max_step),
            int(max_steps),
        )

        states = np.asarray(result["states"])
        stm_flat = np.asarray(result["stm"])
        stm = stm_flat.reshape(-1, 6, 6)
        self.last_trajectory = (np.asarray(result["time"]), states)
        return {
            "time": np.asarray(result["time"]),
            "states": states,
            "stm": stm,
            "terminal_event_index": None,
            "n_steps": result["n_steps"],
            "n_rejected": result["n_rejected"],
        }

    def propagate(
        self,
        initial_state: npt.ArrayLike,
        t_span: tuple[float, float],
        t_eval: npt.ArrayLike | None = None,
        with_stm: bool = False,
        with_jacobi: bool = False,
        *,
        initial_step: float | None = None,
        events: list[Callable[[float, npt.NDArray[np.floating]], float]] | None = None,
        max_steps: int = 100_000,
        method: RkMethod | None = None,
    ) -> dict[str, Any]:
        """使用 Rust 编译传播轨迹（零跨界）。

        issue #378：默认传播一律走编译 Rust（``propagate_compiled`` /
        ``propagate_compiled_stm_py`` / ``propagate_compiled_lowthrust``）。
        扩展不可用（``RustExtensionUnavailableError``）或力模型无 Rust spec
        （``NotImplementedError`` 能力错误）时显式报错，不再静默回退
        Python/scipy。

        Args:
            initial_state: 初始状态向量，形状 (6,)。
            t_span: 时间区间 [t0, tf]，单位为 SPICE et 秒。
            t_eval: 评估时间点数组，默认 linspace(t0, tf, 100)。
            with_stm: 是否同时积分状态转移矩阵。返回字典额外含 ``stm`` 键，
                形状 (n_points, 6, 6)。STM 不参与步长误差控制（对齐 GMAT）。
            with_jacobi: 不支持，传 True 抛 NotImplementedError。
            initial_step: 初始步长，默认从初始状态估算。
            events: 不支持。ForceModel 事件传播需要 compiled-forces Rust API，
                当前未提供，传 events 抛 NotImplementedError（不能回退 Python
                RHS，issue #378）。
            max_steps: 最大积分步数，默认 100_000。
            method: Runge-Kutta 积分器方法，默认 PD45。

        Returns:
            包含 ``time``、``states`` 和 ``terminal_event_index`` 的字典；
            ``with_stm=True`` 时额外含 ``stm`` 键。
        """
        if method is None:
            method = RkMethod.PD45
        self._raise_for_unsupported(with_stm, with_jacobi)

        if len(t_span) != 2:
            raise ValueError("t_span must be a tuple of (t0, tf)")

        t0, tf = float(t_span[0]), float(t_span[1])
        if tf < t0:
            raise NotImplementedError(
                "ForceModel propagation only supports forward integration (tf >= t0)."
            )

        if events is not None:
            raise NotImplementedError(
                "ForceModel 事件传播需要 compiled-forces Rust API（事件检测与"
                "力求值都在 Rust 内循环完成）；当前未提供，传 events 不支持。"
                "issue #378：不再回退 Python RHS。"
            )

        # ── 可变质量低推力 7D 路径 ──
        # 当 force 列表含 VariableMassFiniteBurn 时，状态为 7D [r, v, m]，质量
        # 随推力消耗，走 Rust propagate_compiled_lowthrust。与 6D 主路径隔离，
        # 不参与 STM/事件分派。详见 docs/plans/lowthrust-foundation-prd.md。
        if self._has_variable_mass_thrust():
            return self._propagate_lowthrust(initial_state, t0, tf, t_eval, with_stm, max_steps)

        y0 = np.asarray(initial_state, dtype=float)
        if y0.shape != (self.STATE_DIM,):
            raise ValueError(f"initial_state must have shape ({self.STATE_DIM},)")

        t_eval = self._prepare_t_eval(t0, tf, t_eval)
        tol = float(self.rtol)

        if initial_step is not None:
            if initial_step <= 0:
                raise ValueError("initial_step must be positive")
            h = float(initial_step)
        else:
            h = self._estimate_initial_step(y0, t0, tf)

        # ── 零时长：先检查扩展/spec，防绕过（issue #378）──
        if tf == t0:
            self._require_rust_capability(stm=with_stm)
            self.last_trajectory = (np.array([t0]), y0.reshape(1, -1))
            out: dict[str, Any] = {
                "time": np.array([t0]),
                "states": y0.reshape(1, -1),
                "terminal_event_index": None,
            }
            if with_stm:
                out["stm"] = np.eye(self.STATE_DIM).reshape(1, self.STATE_DIM, self.STATE_DIM)
            return out

        # ── Rust compiled STM 路径 ──
        if with_stm:
            self._require_rust_capability(stm=True)
            return self._propagate_via_rust_stm(y0, t0, tf, t_eval, tol, max_steps)

        # ── Rust propagate_compiled 快速路径 ──
        self._require_rust_capability(stm=False)
        return self._propagate_via_rust(y0, t0, tf, t_eval, tol, h, max_steps)

    def propagate_maneuvers(
        self,
        initial_state: npt.ArrayLike,
        t_span: tuple[float, float],
        burns: list[ImpulsiveBurn],
        *,
        initial_step: float | None = None,
        max_steps: int = 100_000,
        method: RkMethod | None = None,
    ) -> dict[str, Any]:
        """带脉冲机动的传播：coast 段之间在 burn epoch 处施加 Δv。

        按 epoch 排序 burns，依次 coast → 施加 Δv → 续传。burn epoch 处
        输出行携带 post-burn 速度（丢 pre-burn 行，无重复 epoch）。
        """
        if method is None:
            method = RkMethod.PD45
        t0, tf = float(t_span[0]), float(t_span[1])
        y = np.asarray(initial_state, dtype=float)

        _eps = 1e-12
        for burn in burns:
            if burn.epoch < t0 - _eps or burn.epoch > tf + _eps:
                raise ValueError(f"burn epoch {burn.epoch} outside t_span ({t0}, {tf})")

        sorted_burns = sorted(burns, key=lambda b: b.epoch)

        segments: list[tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]] = []
        burn_meta: list[
            tuple[
                float,
                npt.NDArray[np.floating],
                npt.NDArray[np.floating],
                npt.NDArray[np.floating],
            ]
        ] = []
        current_t = t0
        for burn in sorted_burns:
            seg = self.propagate(
                y,
                (current_t, burn.epoch),
                initial_step=initial_step,
                max_steps=max_steps,
                method=method,
            )
            segments.append((seg["time"], seg["states"]))
            y = np.asarray(seg["states"][-1], dtype=float).copy()
            current_t = burn.epoch
            velocity_before = y[3:6].copy()
            delta_v = np.asarray(burn.delta_v, dtype=float)
            y[3:6] = y[3:6] + delta_v
            velocity_after = y[3:6].copy()
            burn_meta.append((burn.epoch, delta_v, velocity_before, velocity_after))

        seg = self.propagate(
            y,
            (current_t, tf),
            initial_step=initial_step,
            max_steps=max_steps,
            method=method,
        )
        segments.append((seg["time"], seg["states"]))

        # 拼接：首段全留；后续段替换运行结果的末行（pre-burn）为本段首行（post-burn）
        time = segments[0][0]
        states = segments[0][1]
        applied: list[BurnApplication] = []
        for i, (seg_time, seg_states) in enumerate(segments[1:]):
            post_burn_index = len(states) - 1
            epoch_i, dv_i, vb_i, va_i = burn_meta[i]
            applied.append(
                BurnApplication(
                    index=post_burn_index,
                    epoch=epoch_i,
                    delta_v=dv_i,
                    velocity_before=vb_i,
                    velocity_after=va_i,
                )
            )
            time = np.concatenate([time[:-1], seg_time])
            states = np.concatenate([states[:-1], seg_states], axis=0)

        self.last_trajectory = (time, states)
        return {
            "time": time,
            "states": states,
            "burns": applied,
            "terminal_event_index": None,
        }

    @staticmethod
    def _prepare_t_eval(
        t0: float, tf: float, t_eval: npt.ArrayLike | None
    ) -> npt.NDArray[np.floating]:
        """准备并校验 t_eval 数组。"""
        if t_eval is None:
            return np.linspace(t0, tf, 100)

        t_eval = np.asarray(t_eval, dtype=float)
        if t_eval.size == 0:
            return np.linspace(t0, tf, 100)

        if t_eval.ndim != 1:
            raise ValueError("t_eval must be one-dimensional")

        if np.any(t_eval < t0 - 1e-14) or np.any(t_eval > tf + 1e-14):
            raise ValueError("t_eval must be within t_span")

        if not np.all(np.diff(t_eval) >= -1e-14):
            raise ValueError("t_eval must be monotonically increasing")

        # Append tf if not present, then unique
        combined = np.concatenate([t_eval, [tf]])
        t_eval = np.unique(np.round(combined / 1e-14) * 1e-14)
        # Ensure monotonic and within bounds after rounding
        t_eval = np.clip(t_eval, t0, tf)
        return np.asarray(t_eval, dtype=float)

    @staticmethod
    def _estimate_initial_step(y: npt.NDArray[np.floating], t0: float, tf: float) -> float:
        """从初始状态估算初始步长。"""
        r = float(np.linalg.norm(y[:3]))
        v = float(np.linalg.norm(y[3:]))
        if r == 0 or v == 0:
            return 1e-6 * abs(tf - t0)
        # Rough orbital period estimate for central motion: 2*pi*r/v
        period = 2.0 * np.pi * r / v
        return float(period / 100.0)

    def _raise_for_unsupported(self, with_stm: bool, with_jacobi: bool) -> None:
        if with_jacobi:
            raise NotImplementedError("ForceModel does not support Jacobi constant computation.")
