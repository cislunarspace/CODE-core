"""Rust 积分器扩展的公共 Python 适配层。"""
# ruff: noqa: F821, F822

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from e2m2e.exceptions import RustExtensionUnavailableError
from e2m2e.spice_ext import _check_rust_abi, _ensure_symbols
from e2m2e.status import ConvergenceState, FailureCause, ResultStatus

# 扩展符号在运行时逐个装载；静态类型检查将其视为动态对象。
if TYPE_CHECKING:
    augmented_eom_7d_py: Any
    batch_body_states_py: Any
    batch_et_to_utc_py: Any
    batch_j2000_to_synodic_py: Any
    batch_synodic_to_j2000_py: Any
    build_cr3bp_hamiltonian_py: Any
    center_manifold_reduce_py: Any
    check_collision_py: Any
    cm_to_qf_py: Any
    collinear_center_modes_py: Any
    compute_distance_series_py: Any
    compute_min_distance_py: Any
    detect_intersection_py: Any
    generate_cr3bp_family_py: Any
    generate_cr3bp_family_windows_py: Any
    detect_local_minimum_py: Any
    disable_ephem_cache: Any
    enable_ephem_cache: Any
    ephem_ffi_call_count: Any
    hello_integrators: Any
    keys_by_order_py: Any
    lambert_batch_py: Any
    lambert_izzo_py: Any
    low_energy_patch_py: Any
    lowthrust_collocation_defects_py: Any
    lowthrust_discrete_collocation_defects_py: Any
    lowthrust_shooting_evaluate_py: Any
    lowthrust_variable_time_collocation_defects_py: Any
    lissajous_bounded_trajectory_py: Any
    manifold_propagate_py: Any
    manifold_seeds_py: Any
    nsga2_environmental_selection_py: Any
    nsga2_sort_py: Any
    nsga2_tournament_selection_py: Any
    nsga2_variation_py: Any
    orbit_family_metric_py: Any
    pal_f_df_tangent_py: Any
    pal_newton_step_py: Any
    pole_tide: Any
    poly_poisson_py: Any
    poly_simplify_py: Any
    polylist_simplify_py: Any
    porkchop_grid_py: Any
    porkchop_grid_states_py: Any
    project_hamiltonian_qf_py: Any
    qf_to_cm_py: Any
    qlaw_propagate_py: Any
    trim_degree_py: Any
    qlaw_segment_direction_py: Any
    propagate_bcr4bp_py: Any
    propagate_bcr4bp_stm_py: Any
    propagate_compiled: Any
    propagate_compiled_ias15_py: Any
    propagate_compiled_lowthrust: Any
    propagate_compiled_lowthrust_sensitivity: Any
    propagate_compiled_stm_py: Any
    propagate_cr3bp_py: Any
    propagate_cr3bp_stm_py: Any
    propagate_cr3bp_megno_py: Any
    propagate_bcr4bp_megno_py: Any
    propagate_geocentric_fate_map_py: Any
    propagate_segments_py: Any
    propagate_with_state_py: Any
    propagate_with_stm_py: Any
    solid_tide_step1: Any
    solid_tide_step2: Any
    solve_ivp_events_py: Any
    solve_ivp_events_kernel_py: Any
    solve_hjb_py: Any
    solve_planar_lowthrust_hjb_py: Any
    spice_furnsh: Any
    spice_pxform: Any
    spice_spkezr: Any
    spice_unload: Any
    srp_acceleration: Any
    third_body_acceleration: Any
    transfer_grid_search_py: Any
    transfer_grid_search_serial_py: Any
    wsb_search_py: Any
    CowellResult: Any
    MultistepMethod: Any
    MultistepResult: Any
    PlanarPalRustResult: Any
    RkMethod: Any
    LowEnergyPatchCandidate: Any
    TransferPointResult: Any
    WsbCandidate: Any
    _cowell_step: Any
    _multistep_step: Any
    _rk_step: Any

_RUST_SYMBOLS = (
    "CowellResult",
    "MultistepMethod",
    "MultistepResult",
    "PlanarPalRustResult",
    "RkMethod",
    "LowEnergyPatchCandidate",
    "TransferPointResult",
    "WsbCandidate",
    "_cowell_step",
    "_multistep_step",
    "_rk_step",
    "augmented_eom_7d_py",
    "batch_body_states_py",
    "batch_et_to_utc_py",
    "batch_j2000_to_synodic_py",
    "batch_synodic_to_j2000_py",
    "build_cr3bp_hamiltonian_py",
    "center_manifold_reduce_py",
    "check_collision_py",
    "cm_to_qf_py",
    "collinear_center_modes_py",
    "compute_distance_series_py",
    "compute_min_distance_py",
    "differential_correction_cr3bp_py",
    "generate_cr3bp_family_py",
    "generate_cr3bp_family_windows_py",
    "detect_intersection_py",
    "detect_local_minimum_py",
    "disable_ephem_cache",
    "enable_ephem_cache",
    "ephem_ffi_call_count",
    "hello_integrators",
    "indirect_term_acceleration",
    "keys_by_order_py",
    "lambert_batch_py",
    "lambert_izzo_py",
    "low_energy_patch_py",
    "lowthrust_collocation_defects_py",
    "lowthrust_discrete_collocation_defects_py",
    "lowthrust_shooting_evaluate_py",
    "lowthrust_variable_time_collocation_defects_py",
    "lissajous_bounded_trajectory_py",
    "manifold_propagate_py",
    "manifold_seeds_py",
    "multiple_shooting_correct_py",
    "nsga2_environmental_selection_py",
    "nsga2_sort_py",
    "nsga2_tournament_selection_py",
    "nsga2_variation_py",
    "orbit_family_metric_py",
    "pal_f_df_tangent_py",
    "pal_newton_step_py",
    "planar_full_period_pal_py",
    "pole_tide",
    "poly_poisson_py",
    "poly_simplify_py",
    "polylist_simplify_py",
    "porkchop_grid_py",
    "porkchop_grid_states_py",
    "project_hamiltonian_qf_py",
    "qf_to_cm_py",
    "qlaw_propagate_py",
    "qlaw_segment_direction_py",
    "trim_degree_py",
    "propagate_bcr4bp_py",
    "propagate_bcr4bp_stm_py",
    "propagate_compiled",
    "propagate_compiled_ias15_py",
    "propagate_compiled_lowthrust",
    "propagate_compiled_lowthrust_sensitivity",
    "propagate_compiled_stm_py",
    "propagate_cr3bp_py",
    "propagate_cr3bp_stm_py",
    "propagate_cr3bp_megno_py",
    "propagate_bcr4bp_megno_py",
    "propagate_geocentric_fate_map_py",
    "propagate_segments_py",
    "propagate_with_state_py",
    "propagate_with_stm_py",
    "reset_ephem_ffi_call_count",
    "segmented_shooting_correct_py",
    "solid_tide_step1",
    "solid_tide_step2",
    "solve_ivp_events_py",
    "solve_ivp_events_kernel_py",
    "solve_hjb_py",
    "solve_planar_lowthrust_hjb_py",
    "spice_furnsh",
    "spice_pxform",
    "spice_spkezr",
    "spice_unload",
    "spherical_harmonic_accel",
    "srp_acceleration",
    "third_body_acceleration",
    "transfer_grid_search_py",
    "transfer_grid_search_serial_py",
    "wsb_search_py",
)

try:
    _rust_extension: Any = importlib.import_module("e2m2e._integrators")
except ImportError:
    _rust_extension = None

for _symbol in _RUST_SYMBOLS:
    _extension_symbol = _symbol.removeprefix("_")
    globals()[_symbol] = getattr(_rust_extension, _extension_symbol, None)


class _ShootingResult:
    """将 Rust 打靶结果在边界处规范化为领域枚举。"""

    def __init__(self, raw: Any) -> None:
        self._raw = raw
        self.status = ConvergenceState(raw.status)
        self.cause = FailureCause(raw.cause)
        self.message = str(raw.message)
        ResultStatus(self.status, self.cause, self.message)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw, name)


_differential_correction_cr3bp_py_raw: Any = globals()["differential_correction_cr3bp_py"]
_multiple_shooting_correct_py_raw: Any = globals()["multiple_shooting_correct_py"]
_planar_full_period_pal_py_raw: Any = globals()["planar_full_period_pal_py"]
_segmented_shooting_correct_py_raw: Any = globals()["segmented_shooting_correct_py"]


def differential_correction_cr3bp_py(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """调用 Rust CR3BP 微分修正，并返回规范化字典。"""
    require_rust_extension("differential_correction_cr3bp_py")
    if _differential_correction_cr3bp_py_raw is None:
        raise RustExtensionUnavailableError(
            "e2m2e._integrators 缺少所需符号：differential_correction_cr3bp_py。请先重建：make dev"
        )
    return dict(_differential_correction_cr3bp_py_raw(*args, **kwargs))


def multiple_shooting_correct_py(*args: Any, **kwargs: Any) -> _ShootingResult:
    """调用 Rust 多重打靶，并立即校验最终状态三元组。"""
    require_rust_extension("multiple_shooting_correct_py")
    if _multiple_shooting_correct_py_raw is None:
        raise RustExtensionUnavailableError(
            "e2m2e._integrators 缺少所需符号：multiple_shooting_correct_py。请先重建：make dev"
        )
    return _ShootingResult(_multiple_shooting_correct_py_raw(*args, **kwargs))


def planar_full_period_pal_py(*args: Any, **kwargs: Any) -> _ShootingResult:
    """调用 Rust 平面全周期 PAL，并立即校验最终状态三元组。"""
    require_rust_extension("planar_full_period_pal_py")
    if _planar_full_period_pal_py_raw is None:
        raise RustExtensionUnavailableError(
            "e2m2e._integrators 缺少所需符号：planar_full_period_pal_py。请先重建：make dev"
        )
    return _ShootingResult(_planar_full_period_pal_py_raw(*args, **kwargs))


def segmented_shooting_correct_py(*args: Any, **kwargs: Any) -> _ShootingResult:
    """调用 Rust 分段打靶，并立即校验最终状态三元组。"""
    require_rust_extension("segmented_shooting_correct_py")
    if _segmented_shooting_correct_py_raw is None:
        raise RustExtensionUnavailableError(
            "e2m2e._integrators 缺少所需符号：segmented_shooting_correct_py。请先重建：make dev"
        )
    return _ShootingResult(_segmented_shooting_correct_py_raw(*args, **kwargs))


# ---- Python↔Rust ABI 版本校验：核心在共享内核叶 spice_ext（ADR 0039）----
# 门面与 data 层各留一道符号门（符号检查查各自模块的 globals），
# 共享 spice_ext._check_rust_abi 的进程级 ABI 缓存与 _ensure_symbols 的报错文案。


def require_rust_extension(*required_symbols: str) -> None:
    """确保 Rust 扩展可用且指定的模块级符号存在；否则抛 ``RustExtensionUnavailableError``。

    在使用 Rust 扩展符号的每个入口调用。扩展未构建、构建不含 spice
    feature、或符号缺失时，抛带 ``make dev`` 指引的
    :class:`RustExtensionUnavailableError`，不允许静默回退到 Python/scipy。
    ``required_symbols`` 是 ``e2m2e.integrators`` 模块级
    符号名；扩展缺失时符号为 ``None``。

    Example:
        >>> require_rust_extension("propagate_compiled", "spice_furnsh")
    """
    _check_rust_abi()
    _ensure_symbols(globals(), required_symbols)


__all__ = [
    "augmented_eom_7d_py",
    "batch_body_states_py",
    "batch_et_to_utc_py",
    "batch_j2000_to_synodic_py",
    "batch_synodic_to_j2000_py",
    "build_cr3bp_hamiltonian_py",
    "center_manifold_reduce_py",
    "check_collision_py",
    "cm_to_qf_py",
    "collinear_center_modes_py",
    "CowellResult",
    "cowell_step",
    "compute_distance_series_py",
    "compute_min_distance_py",
    "differential_correction_cr3bp_py",
    "generate_cr3bp_family_py",
    "generate_cr3bp_family_windows_py",
    "detect_intersection_py",
    "detect_local_minimum_py",
    "disable_ephem_cache",
    "enable_ephem_cache",
    "ephem_ffi_call_count",
    "grid_search_rust",
    "grid_search_rust_serial",
    "hello_integrators",
    "indirect_term_acceleration",
    "initialize_abm_history",
    "initialize_cowell_history",
    "keys_by_order_py",
    "lambert_batch_py",
    "lambert_izzo_py",
    "low_energy_patch_rust",
    "lowthrust_collocation_defects_py",
    "lowthrust_discrete_collocation_defects_py",
    "lowthrust_shooting_evaluate_py",
    "lowthrust_variable_time_collocation_defects_py",
    "lissajous_bounded_trajectory_py",
    "manifold_propagate_py",
    "manifold_seeds_py",
    "MultistepMethod",
    "MultistepResult",
    "multistep_step",
    "multiple_shooting_correct_py",
    "nsga2_environmental_selection_py",
    "nsga2_sort_py",
    "nsga2_tournament_selection_py",
    "nsga2_variation_py",
    "orbit_family_metric_py",
    "pal_f_df_tangent_py",
    "pal_newton_step_py",
    "planar_full_period_pal_py",
    "pole_tide",
    "poly_poisson_py",
    "poly_simplify_py",
    "polylist_simplify_py",
    "porkchop_grid_py",
    "porkchop_grid_states_py",
    "project_hamiltonian_qf_py",
    "qf_to_cm_py",
    "qlaw_propagate_py",
    "qlaw_segment_direction_py",
    "trim_degree_py",
    "propagate_compiled",
    "propagate_compiled_ias15_py",
    "propagate_compiled_lowthrust",
    "propagate_compiled_lowthrust_sensitivity",
    "propagate_compiled_stm_py",
    "propagate_bcr4bp_py",
    "propagate_bcr4bp_stm_py",
    "propagate_cr3bp_py",
    "propagate_cr3bp_stm_py",
    "propagate_cr3bp_megno_py",
    "propagate_bcr4bp_megno_py",
    "propagate_geocentric_fate_map_py",
    "propagate_segments_py",
    "propagate_with_state_py",
    "propagate_with_stm_py",
    "reset_ephem_ffi_call_count",
    "rk_step",
    "RkMethod",
    "RustEomKernel",
    "require_rust_extension",
    "segmented_shooting_correct_py",
    "solid_tide_step1",
    "solid_tide_step2",
    "solve_ivp_events",
    "solve_ivp_events_py",
    "solve_ivp_events_kernel_py",
    "solve_hjb_py",
    "solve_planar_lowthrust_hjb_py",
    "spice_furnsh",
    "spice_pxform",
    "spice_spkezr",
    "spice_unload",
    "spherical_harmonic_accel",
    "srp_acceleration",
    "third_body_acceleration",
    "wsb_search_rust",
    "LowEnergyPatchCandidate",
    "TransferPointResult",
    "transfer_grid_search_py",
    "transfer_grid_search_serial_py",
]

# 事件路径内核分派（issue #594）支持的动力学标识；与 Rust 侧
# ``build_event_kernel_rhs`` 的 match 分支保持一致（Rust 侧为权威校验）。
_EOM_KERNELS = (
    "cr3bp",
    "cr3bp-with-stm",
    "bcr4bp",
    "bcr4bp-with-stm",
)


class RustEomKernel:
    """Rust 内置 EOM 内核标识，替代 Python RHS 回调传入 :func:`solve_ivp_events`。

    携带动力学标识 ``kernel`` 与无量纲参数表 ``params``；积分器据此把每步
    RHS 求值留在 Rust 内（复用 ``e2m2e-forces`` 的 CR3BP/BCR4BP EOM/STM
    实现，issue #594），事件函数仍为 Python 回调。

    标识与所需参数：

    - ``"cr3bp"``：``{"mu": ...}``，6 维右端
    - ``"cr3bp-with-stm"``：``{"mu": ...}``，42 维增广右端（状态 + STM）
    - ``"bcr4bp"``：``{"mu", "mu_sun", "sun_distance", "sun_angular_rate",
      "sun_phase0"}``，6 维右端（显式含时）
    - ``"bcr4bp-with-stm"``：同 ``"bcr4bp"`` 参数，42 维增广右端

    Args:
        kernel: 动力学标识（上表之一，未知标识在构造时报错）。
        params: 参数表；值经 ``float()`` 强制转换。
    """

    __slots__ = ("kernel", "params")

    kernel: str
    params: dict[str, float]

    def __init__(self, kernel: str, params: dict[str, float]) -> None:
        if kernel not in _EOM_KERNELS:
            raise ValueError(
                f"unknown EOM kernel {kernel!r} (supported: {', '.join(_EOM_KERNELS)})"
            )
        object.__setattr__(self, "kernel", kernel)
        object.__setattr__(self, "params", {str(k): float(v) for k, v in params.items()})

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("RustEomKernel is immutable")

    def __repr__(self) -> str:
        return f"RustEomKernel(kernel={self.kernel!r}, params={self.params!r})"


def rk_step(
    method: RkMethod,
    t: float,
    y: npt.ArrayLike,
    h: float,
    tol: float,
    f: Callable[[float, npt.NDArray[np.floating]], npt.NDArray[np.floating]],
    state_error_dim: int | None = None,
):
    """使用 Rust 积分器内核执行单个 Runge-Kutta 步。

    回调 ``f`` 接收 NumPy ndarray，并须返回同长度数组。返回的 ``StepResult``
    包含 ``y_new``、``error``、``h_next``。

    ``state_error_dim``：步长误差控制只统计前 N 维（``None`` 时统计全部）。
    STM 增广传播时传 6，让状态转移矩阵的 36 个分量不主导步长控制。
    """
    require_rust_extension("_rk_step", "RkMethod")
    y = np.asarray(y, dtype=float)

    def _adapt(t_i: float, y_i: list[float]) -> list[float]:
        y_arr = np.asarray(y_i, dtype=float)
        result = f(t_i, y_arr)
        return np.asarray(result, dtype=float).tolist()

    return _rk_step(method, t, y.tolist(), h, tol, _adapt, state_error_dim)


def multistep_step(
    method: MultistepMethod,
    t: float,
    y: npt.ArrayLike,
    h: float,
    tol: float,
    f: Callable[[float, npt.NDArray[np.floating]], npt.NDArray[np.floating]],
    history: list[npt.ArrayLike],
):
    """执行单个多步预测-校正步。

    ``history`` 须按从旧到新的顺序保存 ``method.steps()`` 个导数样本，每个样本
    与 ``y`` 等长，间隔均为 ``h``。回调 ``f`` 的签名与 :func:`rk_step` 相同。
    返回 ``MultistepResult``，其 ``history`` 是供下一步使用的滚动缓冲区。

    假定步长固定；改变 ``h`` 后须重新初始化 history（见
    :func:`initialize_abm_history`）。
    """
    require_rust_extension("_multistep_step", "MultistepMethod")
    y = np.asarray(y, dtype=float)

    def _adapt(t_i: float, y_i: list[float]) -> list[float]:
        y_arr = np.asarray(y_i, dtype=float)
        result = f(t_i, y_arr)
        return np.asarray(result, dtype=float).tolist()

    hist_lists = [np.asarray(hi, dtype=float).tolist() for hi in history]
    return _multistep_step(method, t, y.tolist(), h, tol, _adapt, hist_lists)


def initialize_abm_history(
    t0: float,
    y0: npt.ArrayLike,
    h: float,
    f: Callable[[float, npt.NDArray[np.floating]], npt.NDArray[np.floating]],
    n_stages: int = 3,
    tol: float = 1e-12,
) -> tuple[float, np.ndarray, list[list[float]]]:
    """以 ``n_stages`` 个 RK89 步启动 ABM history。

    ABM 方法使用 4 个导数样本；默认 ``n_stages=3`` 时返回
    ``(t0 + 3h, y(3h), [f_0, f_1, f_2, f_3])``，其中 history 可直接传给
    :func:`multistep_step`。
    """
    require_rust_extension("RkMethod")
    y = np.asarray(y0, dtype=float).copy()
    t = float(t0)
    history: list[list[float]] = [np.asarray(f(t, y), dtype=float).tolist()]
    for _ in range(n_stages):
        result = rk_step(RkMethod.RK89, t, y, h, tol, f)
        y = np.asarray(result.y_new, dtype=float)
        t += h
        history.append(np.asarray(f(t, y), dtype=float).tolist())
    return t, y, history


def cowell_step(
    t: float,
    h: float,
    tol: float,
    accel: Callable[[float, npt.NDArray[np.floating]], npt.NDArray[np.floating]],
    history: list[npt.ArrayLike],
):
    """对 ``x'' = a(t, x)`` 执行单个 Cowell（Störmer-Cowell）8 阶步。

    ``history`` = ``[x_{n-1}, x_n, a_{n-7}, ..., a_n]`` （10 个向量：2 个位置
    样本与 8 个加速度样本，按从旧到新排列）。``accel(t, x)`` 返回只依赖位置的
    加速度（引力、J2）。输出仅含位置，步长固定。

    返回的 ``CowellResult`` 包含 ``x_new``、``error``、``h_next``、``history``。
    """
    require_rust_extension("_cowell_step")
    hist_lists = [np.asarray(hi, dtype=float).tolist() for hi in history]

    def _adapt(t_i: float, x_i: list[float]) -> list[float]:
        x_arr = np.asarray(x_i, dtype=float)
        result = accel(t_i, x_arr)
        return np.asarray(result, dtype=float).tolist()

    return _cowell_step(t, h, tol, _adapt, hist_lists)


def initialize_cowell_history(
    t0: float,
    x0: npt.ArrayLike,
    v0: npt.ArrayLike,
    h: float,
    accel: Callable[[float, npt.NDArray[np.floating]], npt.NDArray[np.floating]],
    n_startup: int = 7,
    tol: float = 1e-12,
) -> tuple[float, np.ndarray, np.ndarray, list[list[float]]]:
    """以 ``n_startup`` 个 RK89 步启动 8 阶 Cowell history。

    返回 ``(t, x, v, history)``，其中
    ``history = [x_{n-1}, x_n, a_{n-7}, ..., a_n]`` （2 个位置与 8 个加速度，
    可直接传给 :func:`cowell_step`）。``n_startup`` 须不小于 7，以获得最近的
    8 个加速度样本；默认 ``n_startup=7`` 时状态推进至 ``t0 + 7h``。
    """
    require_rust_extension("RkMethod")
    if n_startup < 7:
        raise ValueError(
            f"8th-order Cowell needs n_startup >= 7 (8 acceleration samples), got {n_startup}"
        )
    x = np.asarray(x0, dtype=float).copy()
    v = np.asarray(v0, dtype=float).copy()
    t = float(t0)
    d = len(x)
    xs = [x.copy()]
    accels = [np.asarray(accel(t, x), dtype=float)]

    def _first_order(t_i: float, state: np.ndarray) -> np.ndarray:
        return np.concatenate([state[d:], np.asarray(accel(t_i, state[:d]), dtype=float)])

    for _ in range(n_startup):
        result = rk_step(RkMethod.RK89, t, np.concatenate([x, v]), h, tol, _first_order)
        y = np.asarray(result.y_new, dtype=float)
        x, v = y[:d], y[d:]
        t += h
        xs.append(x.copy())
        accels.append(np.asarray(accel(t, x), dtype=float))

    history = [xs[-2].tolist(), xs[-1].tolist()] + [a.tolist() for a in accels[-8:]]
    return t, x, v, history


def solve_ivp_events(
    t_span: tuple[float, float],
    y0: npt.ArrayLike,
    t_eval: npt.ArrayLike,
    rtol: float,
    atol: float,
    f: Callable[[float, npt.NDArray[np.floating]], npt.NDArray[np.floating]] | RustEomKernel,
    events: list[tuple[Callable[[float, npt.NDArray[np.floating]], float], bool, float]],
    method: RkMethod | None = None,
    max_step: float | None = None,
    max_steps: int | None = None,
    state_error_dim: int | None = None,
) -> dict[str, Any]:
    """带事件检测的 Rust solve_ivp 封装（scipy 事件语义）。

    事件检测在 Rust 积分内循环完成：每个接受步的端点评估事件函数，
    符号变化（经 direction 过滤）时在步内对线性插值态二分求精（无稠密输出）。

    Args:
        t_span: 积分区间 ``(t0, tf)``。
        y0: 初始状态向量。
        t_eval: 输出时间点数组。
        rtol: 相对容差。
        atol: 绝对容差。
        f: ODE 右端，两种形态二选一：Python callable
            ``f(t, y) -> dy/dt``（每步求值跨语言回调），或
            :class:`RustEomKernel` 内核标识（每步 RHS 求值留在 Rust 内，
            复用 ``e2m2e-forces`` 的 EOM/STM 内核，issue #594；事件函数
            仍为 Python 回调）。
        events: ``[(g, terminal, direction), ...]``，``g(t, y) -> float``，
            零点即事件面；``terminal=True`` 触发即停；``direction`` > 0 只记
            上行穿越（g 由负到正）、< 0 只记下行、0 双向。
        method: RK 方法，默认 PD78（DOP853）。
        max_step: 最大步长。求精精度受步内线性插值误差（``~h²/8·|ÿ|``）限制，
            需要更紧的事件时刻时请设小 max_step。
        max_steps: 最大积分步数。
        state_error_dim: 步长误差控制只统计前 N 维（用于 STM 增广传播）。

    Returns:
        dict：``states``/``time`` （t_eval 前缀，terminal 截断时末点为求精后的
        事件点）、``t_events``/``y_events`` （逐事件的触发时刻与状态列表）、
        ``terminal_event`` （触发终止的事件索引或 None）、``n_steps``。
    """
    y0_arr = np.asarray(y0, dtype=float)

    def _adapt_event(
        g: Callable[[float, npt.NDArray[np.floating]], float],
    ) -> Callable[[float, list[float]], float]:
        def _g(t_i: float, y_i: list[float]) -> float:
            return float(g(t_i, np.asarray(y_i, dtype=float)))

        return _g

    event_specs = [
        (_adapt_event(g), terminal, float(direction)) for g, terminal, direction in events
    ]

    args = (
        (float(t_span[0]), float(t_span[1])),
        y0_arr.tolist(),
        [float(t) for t in np.asarray(t_eval, dtype=float).flat],
        float(rtol),
        float(atol),
    )
    if isinstance(f, RustEomKernel):
        # 内核分派：每步 RHS 求值留在 Rust 内（issue #594）。
        require_rust_extension("solve_ivp_events_kernel_py")
        return solve_ivp_events_kernel_py(
            *args,
            f.kernel,
            f.params,
            event_specs,
            method,
            max_step,
            max_steps,
            state_error_dim,
        )

    require_rust_extension("solve_ivp_events_py")

    def _adapt_rhs(t_i: float, y_i: list[float]) -> list[float]:
        return np.asarray(f(t_i, np.asarray(y_i, dtype=float)), dtype=float).tolist()

    return solve_ivp_events_py(
        *args,
        _adapt_rhs,
        event_specs,
        method,
        max_step,
        max_steps,
        state_error_dim,
    )


def wsb_search_rust(
    departure_state: npt.ArrayLike,
    target_state: npt.ArrayLike,
    *,
    mu: float,
    mu_sun: float,
    sun_distance: float,
    sun_angular_rate: float,
    sun_phase_range: tuple[float, float],
    n_sun_phase: int,
    departure_phase_range: tuple[float, float],
    n_departure_phase: int,
    tof_range_sec: tuple[float, float],
    n_tof: int,
    perilune_alt_range_km: tuple[float, float],
    max_total_dv: float,
    h2_energy_threshold: float,
    tli_speed_factor: float,
    n_propagation_samples: int,
    rtol: float,
    atol: float,
    max_step: float,
    max_steps: int,
    secondary_radius_km: float,
    characteristic_length_km: float,
    characteristic_time_sec: float,
    parallel: bool | None = None,
    n_workers: int | None = None,
    progress_callback: Callable[[int], Any] | None = None,
) -> tuple[list[dict[str, Any]], int, int]:
    """WSB 三维网格搜索的 Rust 后端。

    参数全部是已无量纲化的 POD 数值；BCR4BP 传播、截面求精和候选筛选均在
    Rust 内完成。``parallel=None`` 时由 ``E2M2E_WSB_PARALLEL`` 控制 Rayon。
    """
    require_rust_extension("wsb_search_py")
    departure_arr = np.asarray(departure_state, dtype=float).reshape(-1)
    target_arr = np.asarray(target_state, dtype=float).reshape(-1)
    if departure_arr.shape != (6,) or target_arr.shape != (6,):
        raise ValueError("departure_state 与 target_state 必须都是长度 6 的状态")

    raw_candidates, n_propagation_failures, n_perilune_in_window = wsb_search_py(
        departure_arr.tolist(),
        target_arr.tolist(),
        float(mu),
        float(mu_sun),
        float(sun_distance),
        float(sun_angular_rate),
        float(sun_phase_range[0]),
        float(sun_phase_range[1]),
        int(n_sun_phase),
        float(departure_phase_range[0]),
        float(departure_phase_range[1]),
        int(n_departure_phase),
        float(tof_range_sec[0]),
        float(tof_range_sec[1]),
        int(n_tof),
        float(perilune_alt_range_km[0]),
        float(perilune_alt_range_km[1]),
        float(max_total_dv),
        float(h2_energy_threshold),
        float(tli_speed_factor),
        int(n_propagation_samples),
        float(rtol),
        float(atol),
        float(max_step),
        int(max_steps),
        float(secondary_radius_km),
        float(characteristic_length_km),
        float(characteristic_time_sec),
        parallel=parallel,
        n_workers=n_workers,
        progress_callback=progress_callback,
    )
    return (
        [
            {
                "sun_phase0": candidate.sun_phase0,
                "departure_phase": candidate.departure_phase,
                "tof_sec": candidate.tof_sec,
                "departure_state": np.asarray(candidate.departure_state, dtype=float),
                "perilune_state": np.asarray(candidate.perilune_state, dtype=float),
                "perilune_alt_km": candidate.perilune_alt_km,
                "perilune_time_dim": candidate.perilune_time_dim,
                "arrival_state": np.asarray(candidate.arrival_state, dtype=float),
                "h2_kepler": candidate.h2_kepler,
                "dv_departure": candidate.dv_departure,
                "dv_arrival": candidate.dv_arrival,
                "total_dv": candidate.total_dv,
                "arrival_time_dim": candidate.arrival_time_dim,
            }
            for candidate in raw_candidates
        ],
        int(n_propagation_failures),
        int(n_perilune_in_window),
    )


def low_energy_patch_rust(
    states_a: npt.ArrayLike,
    states_b: npt.ArrayLike,
    weights: tuple[float, float],
    *,
    parallel: bool | None = None,
    n_workers: int | None = None,
    progress_callback: Callable[[int], Any] | None = None,
) -> list[dict[str, Any]]:
    """低能转移流形截面态配对的 Rust 后端。

    输入为两组 ``(n, 6)`` 截面态，Rust 完成全部配对、位置/速度范数、
    加权代价和稳定排序。``parallel=None`` 时由
    ``E2M2E_LOW_ENERGY_PARALLEL`` 决定是否使用 Rayon；显式 ``n_workers``
    为本次调用建立一次性线程池。
    """
    require_rust_extension("low_energy_patch_py")
    states_a_arr = np.asarray(states_a, dtype=float)
    states_b_arr = np.asarray(states_b, dtype=float)
    if states_a_arr.ndim != 2 or states_a_arr.shape[1:] != (6,):
        raise ValueError(f"states_a 须为 (n, 6)，得到 {states_a_arr.shape}")
    if states_b_arr.ndim != 2 or states_b_arr.shape[1:] != (6,):
        raise ValueError(f"states_b 须为 (n, 6)，得到 {states_b_arr.shape}")

    weight_r, weight_v = (float(weights[0]), float(weights[1]))
    raw_candidates = low_energy_patch_py(
        states_a_arr.reshape(-1).tolist(),
        states_b_arr.reshape(-1).tolist(),
        weight_r,
        weight_v,
        parallel=parallel,
        n_workers=n_workers,
        progress_callback=progress_callback,
    )
    return [
        {
            "i_a": candidate.i_a,
            "i_b": candidate.i_b,
            "state_a": np.asarray(candidate.state_a, dtype=float),
            "state_b": np.asarray(candidate.state_b, dtype=float),
            "delta_r": candidate.delta_r,
            "delta_v": candidate.delta_v,
            "cost": candidate.cost,
        }
        for candidate in raw_candidates
    ]


def grid_search_rust_serial(
    dep_states: npt.ArrayLike,
    dep_times: npt.ArrayLike,
    alpha_grid: npt.ArrayLike,
    arrival_states: npt.ArrayLike,
    *,
    mu: float,
    max_transfer_time: float,
    integration_dt: float,
    intersection_threshold: float,
    min_distance_threshold: float,
    collision_earth_radius: float,
    collision_moon_radius: float,
    rtol: float,
    atol: float,
    max_step: float,
    progress_callback: Callable[[int], Any] | None = None,
) -> list[dict[str, Any]]:
    """转移网格搜索 Rust 串行后端（阶段 B）。

    展平 POD 输入 → 调 ``transfer_grid_search_serial_py`` → 转 ``list[dict]``。
    返回字段对齐 ``search_parallel.grid_search_sequential``，便于逐候选等价
    对照（整数索引精确相等、浮点 ``allclose``）。

    本 wrapper 只做数组→dict 转换，不依赖 transfer 算法层（分层：算法层调
    数值层合法，数值层不反向依赖）。出发轨道采样（``sample_departure_points``）
    与 Orbit 展平由调用方完成；阶段 D 的 ``grid_search_rust`` 编排器会在此之上
    接入 ``TransferSearch``。

    Args:
        dep_states: ``(n_dep, 6)`` 或展平 ``n_dep*6`` 出发状态。
        dep_times: ``(n_dep,)`` 出发时刻。
        alpha_grid: ``(n_alpha,)`` 切向速度比 α 网格。
        arrival_states: ``(n_arrival, 6)`` 或展平目标轨道状态。
        mu / max_transfer_time / integration_dt / intersection_threshold /
            min_distance_threshold / collision_earth_radius /
            collision_moon_radius: CR3BP 与搜索标量配置。
        rtol / atol / max_step: 积分器容差与最大步长。
        progress_callback: ``cb(delta: int) -> None``，每个 departure 完成
            调一次（出发粒度）；``None`` 不回调。Rust 端走 channel + drainer
            线程，释放 GIL 后实时回调。

    Returns:
        ``list[dict]``，长度 ``n_dep * n_alpha``，顺序为外层 departure、
        内层 alpha（与 ``grid_search_sequential`` 一致）。
    """
    require_rust_extension("transfer_grid_search_serial_py")

    dep_states_arr = np.asarray(dep_states, dtype=float).reshape(-1)
    dep_times_arr = np.asarray(dep_times, dtype=float).reshape(-1)
    alpha_arr = np.asarray(alpha_grid, dtype=float).reshape(-1)
    arrival_arr = np.asarray(arrival_states, dtype=float).reshape(-1)

    results = transfer_grid_search_serial_py(
        dep_states_arr.tolist(),
        dep_times_arr.tolist(),
        alpha_arr.tolist(),
        arrival_arr.tolist(),
        float(mu),
        float(max_transfer_time),
        float(integration_dt),
        float(intersection_threshold),
        float(min_distance_threshold),
        float(collision_earth_radius),
        float(collision_moon_radius),
        float(rtol),
        float(atol),
        float(max_step),
        progress_callback=progress_callback,
    )
    return [_transfer_point_result_to_dict(r) for r in results]


def grid_search_rust(
    dep_states: npt.ArrayLike,
    dep_times: npt.ArrayLike,
    alpha_grid: npt.ArrayLike,
    arrival_states: npt.ArrayLike,
    *,
    mu: float,
    max_transfer_time: float,
    integration_dt: float,
    intersection_threshold: float,
    min_distance_threshold: float,
    collision_earth_radius: float,
    collision_moon_radius: float,
    rtol: float,
    atol: float,
    max_step: float,
    parallel: bool | None = None,
    n_workers: int | None = None,
    progress_callback: Callable[[int], Any] | None = None,
) -> list[dict[str, Any]]:
    """转移网格搜索 Rust 后端（阶段 C，Rayon 并行 + GIL 释放）。

    展平 POD 输入 → 调 ``transfer_grid_search_py`` （``py.allow_threads``
    释放 GIL + Rayon ``par_iter`` 真并行）→ 转 ``list[dict]``。返回字段与
    顺序与 :func:`grid_search_rust_serial` 完全一致——并行与串行逐位相同
    （``par_iter``+``collect`` 保序、``evaluate_point`` 纯函数）。
    其余参数同 :func:`grid_search_rust_serial`。

    Args:
        parallel: ``None`` （默认）时由 ``E2M2E_SEARCH_PARALLEL`` 环境变量决定
            （``"0"``→串行，其余/未设→并行）；显式 ``True``/``False`` 覆盖。
            串/并一致性对照用 ``parallel=False`` 与 ``parallel=True`` 各跑一遍。
        n_workers: ``None`` （默认）时用 Rayon 全局线程池，线程数由
            ``RAYON_NUM_THREADS`` 决定（未设则 cpu 核数）；显式传入时 Rust 端
            建一次性 ``ThreadPoolBuilder`` 限定 ``max(n_workers, 1)`` 个线程并
            ``install`` 本次 compute，覆盖 ``RAYON_NUM_THREADS``。串行模式
            （``parallel=False``）下无线程池，此参数被忽略。
        progress_callback: ``cb(delta: int) -> None``，每个 departure 完成
            调一次（出发粒度）；``None`` 不回调。Rust 端走 channel + drainer
            线程，释放 GIL 后实时回调。

    Returns:
        ``list[dict]``，长度 ``n_dep * n_alpha``，顺序为外层 departure、内层 alpha。
    """
    require_rust_extension("transfer_grid_search_py")

    dep_states_arr = np.asarray(dep_states, dtype=float).reshape(-1)
    dep_times_arr = np.asarray(dep_times, dtype=float).reshape(-1)
    alpha_arr = np.asarray(alpha_grid, dtype=float).reshape(-1)
    arrival_arr = np.asarray(arrival_states, dtype=float).reshape(-1)

    results = transfer_grid_search_py(
        dep_states_arr.tolist(),
        dep_times_arr.tolist(),
        alpha_arr.tolist(),
        arrival_arr.tolist(),
        float(mu),
        float(max_transfer_time),
        float(integration_dt),
        float(intersection_threshold),
        float(min_distance_threshold),
        float(collision_earth_radius),
        float(collision_moon_radius),
        float(rtol),
        float(atol),
        float(max_step),
        parallel=parallel,
        n_workers=n_workers,
        progress_callback=progress_callback,
    )
    return [_transfer_point_result_to_dict(r) for r in results]


def _transfer_point_result_to_dict(r: TransferPointResult) -> dict[str, Any]:
    """``TransferPointResult`` pyclass → dict，字段对齐 ``search_single_departure``。

    数组字段（``transfer_trajectory``/``intersection_point``/``transfer_times``）
    还原为 numpy 数组，与 Python sequential 后端返回类型一致。
    """
    traj = r.transfer_trajectory
    traj_arr = np.asarray(traj, dtype=float).reshape(-1, 6) if traj is not None else None
    times = r.transfer_times
    times_arr = np.asarray(times, dtype=float) if times is not None else None
    int_pt = r.intersection_point
    int_pt_arr = np.asarray(int_pt, dtype=float) if int_pt is not None else None
    status = ConvergenceState(r.status)
    cause_value = r.cause
    if cause_value == "infeasible":
        cause_value = "no_intersection"
    cause = FailureCause(cause_value)
    message = r.message
    ResultStatus(status, cause, message)
    return {
        "status": status,
        "cause": cause,
        "message": message,
        "departure_state": np.asarray(r.departure_state, dtype=float),
        "departure_time": r.departure_time,
        "alpha": r.alpha,
        "transfer_trajectory": traj_arr,
        "transfer_times": times_arr,
        "transfer_time": r.transfer_time,
        "min_distance": r.min_distance,
        "min_distance_idx": r.min_distance_idx,
        "min_distance_orbit_idx": r.min_distance_orbit_idx,
        "dv_departure": r.dv_departure,
        "dv_insertion": r.dv_insertion,
        "intersection_found": r.intersection_found,
        "intersection_point": int_pt_arr,
        "intersection_idx": r.intersection_idx,
        "first_intersection_idx": r.first_intersection_idx,
        "first_intersection_time": r.first_intersection_time,
        "first_min_distance_idx": r.first_min_distance_idx,
        "first_min_distance_time": r.first_min_distance_time,
        "local_minimum_found": r.local_minimum_found,
        "local_minimum_distance": r.local_minimum_distance,
        "local_minimum_idx": r.local_minimum_idx,
        "collision_found": r.collision_found,
        "collision_body": r.collision_body,
        "collision_idx": r.collision_idx,
    }
