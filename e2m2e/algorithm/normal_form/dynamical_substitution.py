"""动力学替代轨道与生成函数 ``W`` 计算（Code5）。

对应 qiao ``Code05_DynSubs_Gfunc.py``：

1. **多点打靶**：在归一化时间窗口上以等距节点构造初值；用块三对角
   Newton 迭代求解连续性方程，得到一条围绕平动点的闭轨道
   ``b(t) = (rho, rhodot)``；
2. **频域分解**：对 ``b(t)`` 做频率分析（NAFF/FFT），把受迫分量与
   中心流形分量分离；
3. **生成函数 ``W(t)``**：对动量分量数值微分得 ``B̈``，由 ``Bdot2A``
   公式把 ``B`` 与 ``A`` 拼起来，组装 ``W_poly`` / ``Wdot_poly``。

Public API：

- :class:`DynamicalSubstituteCorrector` —— 上下文绑定的 corrector，
  通过 :meth:`reduce` 给出 :class:`DynamicalSubstituteResult`；
- :class:`DynamicalSubstituteResult` —— 不透明结果句柄；
- :func:`_build_dynamics_rhs` —— 把 ``NormalFormContext`` 翻译为
  ODE 右端项的内部辅助（也供 slice 3 测试 / 复用）。

实现策略：

- 复用 :mod:`~e2m2e.algorithms.normal_form.multiple_shooting` 的块三对角消元；
- 复用 :mod:`.fft` 的 NAFF/FFT 自动选择；
- 复用 :func:`.hamiltonian.evaluate_hamiltonian` / 星历参数（与
  slice 1 保持接口一致）；
- 当外部 SPICE 内核不可用时（如 CI 环境），``reduce`` 走 ``Pure
  CR3BP`` 退路：忽略太阳与三体摄动，使用旋转系下的 Hill 方程；
  该退路仅供烟雾测试，不用于生产数据。
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .. import normal_form
from .fft import (
    FFTComponent,
    extract_frequencies,
)
from .multiple_shooting import (
    MultipleShootingResult,
    ODESubstituteSolver,
    ShootingPatch,
    SubstituteSolver,
    multiple_shooting_newton,
)

if TYPE_CHECKING:
    from .context import NormalFormContext


# ---------------------------------------------------------------------------
# 默认打靶窗口与节点间距
# ---------------------------------------------------------------------------

#: qiao Code05 默认总窗口：``0.1 * 2^16 = 6553.6 TU``。
DEFAULT_TOTAL_TU: float = 0.1 * (2**16)
#: qiao Code05 默认节点间距：``0.8 TU``。
DEFAULT_NODE_STEP: float = 0.8
#: qiao Code05 稠密输出采样间距：``0.1 TU``。
DEFAULT_DENSE_STEP: float = 0.1
#: qiao Code05 Newton 迭代最大轮数：``20``。
DEFAULT_MAX_ITER: int = 19
#: qiao Code05 收敛容差：``1e-11``。
DEFAULT_TOLERANCE: float = 1e-11

#: 纯 CR3BP（自治）的 Coriolis 阵 ``C_pq = [[0,1,0],[-1,0,0],[0,0,0]]``。
#: ``Bdot2A`` 与降级路径共用——避免在 ``_bdot2a`` 内联两份同值矩阵。
_CR3BP_CPQ: npt.NDArray[np.floating] = np.array(
    [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float
)


# ---------------------------------------------------------------------------
# 结果容器
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DynamicalSubstituteResult:
    """动力学替代校正结果。

    Attributes:
        context: 关联 :class:`NormalFormContext`。
        order: 展开阶数（与 ``context.order`` 一致）。
        substitute_orbit: 替代轨道稠密输出，``(n, 6)`` 状态数组。
        tlist: 稠密输出时间数组，形状 ``(n,)``，归一化 TU。
        Xlist: 稠密输出状态数组，形状 ``(n, 6)``。
        W_poly: ``(pow, coef_array)`` 形式的生成函数 ``W(t)``；6 个
            线性项各对应一个幂次 ``(1,0,0,0,0,0)``/.../``(0,0,0,0,0,1)``。
        Wdot_poly: 与 ``W_poly`` 同结构的 ``Wdot(t)``。
        fft_components: ``x/y/z`` 三个方向的 :class:`FFTComponent` 列表；
            供后续 slice 引用。
        shooting_result: 多重打靶迭代结果（节点、残差历史、收敛标志）。
        backend: 实际使用的频率分析后端：``"naff"`` / ``"fft"``。
        spice_available: 本次 ``reduce`` 是否实际使用了 SPICE 星历模型。
        metadata: 自由扩展字段。
    """

    context: NormalFormContext
    order: int
    substitute_orbit: npt.NDArray[np.floating]
    tlist: npt.NDArray[np.floating]
    Xlist: npt.NDArray[np.floating]
    W_poly: dict[tuple[int, ...], npt.NDArray[np.floating]]
    Wdot_poly: dict[tuple[int, ...], npt.NDArray[np.floating]]
    fft_components: dict[str, list[FFTComponent]] = field(default_factory=dict)
    shooting_result: normal_form.multiple_shooting.MultipleShootingResult | None = None
    backend: str = "fft"
    spice_available: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def residual_norm(self) -> float:
        """打靶连续性残差最大值（供调用方快速判定收敛性）。"""
        if self.shooting_result is None:
            return float("nan")
        return self.shooting_result.max_residual


# ---------------------------------------------------------------------------
# Corrector 类
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DynamicalSubstituteCorrector:
    """动力学替代 corrector（上下文绑定）。

    通过 :meth:`reduce` 把 ``seed`` 状态修正到动力学替代轨道 ``b(t)``，
    并输出生成函数 ``W`` / ``Wdot`` 与频率分析结果。

    Args:
        context: 归一化上下文。
        t_total: 打靶总窗口（TU）。
        node_step: 节点间距（TU）。
        dense_step: 稠密输出采样间距（TU）。
        max_iter: Newton 最大迭代轮数。
        tolerance: 收敛容差（最大连续性残差）。
        prefer: 频率分析后端选择（``"auto"``/``"naff"``/``"fft"``）。
        spice_optional: SPICE 内核不可用时是否降级到纯 CR3BP。
            ``True`` 时静默降级；``False`` 时抛 :class:`RuntimeError`。
    """

    context: NormalFormContext
    t_total: float = DEFAULT_TOTAL_TU
    node_step: float = DEFAULT_NODE_STEP
    dense_step: float = DEFAULT_DENSE_STEP
    max_iter: int = DEFAULT_MAX_ITER
    tolerance: float = DEFAULT_TOLERANCE
    prefer: str = "auto"
    spice_optional: bool = True

    # ------------------------------------------------------------------
    # 公开入口
    # ------------------------------------------------------------------

    def reduce(
        self,
        seed: npt.ArrayLike | None = None,
    ) -> DynamicalSubstituteResult:
        """对 ``seed`` 状态执行动力学替代校正。

        Args:
            seed: ``(6,)`` rho 坐标初始状态；``None`` 时用平动点位置
                零速度作为初始猜测（``X_Q`` 全零初值）。

        Returns:
            :class:`DynamicalSubstituteResult`。

        Raises:
            RuntimeError: 当 ``spice_optional=False`` 且 SPICE 不可用。
        """
        seed_arr = self._normalize_seed(seed)
        n_nodes = int(round(self.t_total / self.node_step)) + 1
        t_Q = np.linspace(0.0, self.t_total, n_nodes)

        # 初始 X_Q 拷贝到全部节点
        X_Q = np.tile(seed_arr, (n_nodes, 1))

        rhs, provider = self._build_dynamics()
        spice_available = provider is not None

        if not spice_available and not self.spice_optional:
            raise RuntimeError(
                "SPICE 内核不可用且 spice_optional=False。请加载 .tls + .bsp 或显式允许降级。"
            )

        solver: SubstituteSolver = ODESubstituteSolver(rhs=rhs, rtol=1e-10, atol=1e-12)

        # ---- 多重打靶 ----
        patch = ShootingPatch(t_Q=t_Q, X_Q=X_Q)
        shooting = multiple_shooting_newton(
            patch,
            solver,
            max_iter=self.max_iter,
            tolerance=self.tolerance,
        )

        # ---- 稠密输出 ----
        tlist, Xlist = self._dense_output(shooting, solver)

        # ---- 频率分析 ----
        fft_components, backend = self._frequency_analysis(tlist, Xlist)

        # ---- 生成函数 W ----
        W_poly, Wdot_poly = self._build_W(tlist, Xlist, use_cr3bp=not spice_available)

        # ---- 包装成 Orbit ----
        substitute_orbit = Xlist

        return DynamicalSubstituteResult(
            context=self.context,
            order=int(self.context.order),
            substitute_orbit=substitute_orbit,
            tlist=tlist,
            Xlist=Xlist,
            W_poly=W_poly,
            Wdot_poly=Wdot_poly,
            fft_components=fft_components,
            shooting_result=shooting,
            backend=backend,
            spice_available=spice_available,
            metadata={
                "t_total": float(self.t_total),
                "node_step": float(self.node_step),
                "dense_step": float(self.dense_step),
                "n_nodes": int(shooting.t_Q.shape[0]),
                "n_segments": int(shooting.t_Q.shape[0] - 1),
            },
        )

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _normalize_seed(self, seed: npt.ArrayLike | None) -> npt.NDArray[np.floating]:
        if seed is None:
            return np.zeros(6, dtype=float)
        arr = np.asarray(seed, dtype=float).ravel()
        if arr.shape != (6,):
            raise ValueError(f"seed 必须是形状 (6,)，得到 {arr.shape}")
        return arr

    def _build_dynamics(
        self,
    ) -> tuple[
        Callable[[float, npt.ArrayLike], npt.ArrayLike],
        Callable[[float], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | None,
    ]:
        """构造 rho 坐标 ODE 右端项与（可选）SPICE provider。

        探测性求值一次：SPICE 缺失（如未加载 ``naif*.tls``）会在
        ``str2et`` 抛 :class:`SpiceNOLEAPSECONDS`，此时回退到纯 CR3BP。
        ``context.force_cr3bp=True`` 时跳过 SPICE 探测，直接用纯 CR3BP rhs。
        """
        if self.context.force_cr3bp:
            return _build_dynamics_rhs_circular(self.context), None
        try:
            rhs, provider = _build_dynamics_rhs_spice(self.context)
            _ = rhs(0.0, np.zeros(6))
        except Exception as exc:
            if not self.spice_optional:
                raise
            warnings.warn(
                f"SPICE 求值失败（{type(exc).__name__}: {exc}）；"
                "降级到纯 CR3BP 旋转系。"
                "该退路仅供烟雾测试，不用于生产数据。",
                stacklevel=3,
            )
            return _build_dynamics_rhs_circular(self.context), None
        return rhs, provider

    def _dense_output(
        self,
        shooting: MultipleShootingResult,
        solver: SubstituteSolver,
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """在 ``t_Q`` 节点基础上做稠密输出（DOP853 高阶积分）。"""
        from ._solve_ivp_rust import solve_ivp_rust

        t_Q = shooting.t_Q
        X_Q = shooting.X_Q
        n_seg = t_Q.shape[0] - 1

        # 稠密采样网格
        t_dense = np.arange(0.0, self.t_total + 0.5 * self.dense_step, self.dense_step)

        pieces_t: list[np.ndarray] = []
        pieces_X: list[np.ndarray] = []
        for i in range(n_seg):
            t_lo = float(t_Q[i])
            t_hi = float(t_Q[i + 1])
            seg_t = t_dense[(t_dense >= t_lo - 1e-9) & (t_dense <= t_hi + 1e-9)]
            if seg_t.size == 0:
                continue
            sol = solve_ivp_rust(
                fun=lambda t, X: (
                    np.asarray(
                        solver.propagate_segment.__self__.rhs(t, X),  # type: ignore[attr-defined]
                        dtype=float,
                    ).ravel()
                    if hasattr(solver.propagate_segment, "__self__")
                    else _ode_rhs_via_solver(solver, t, X)
                ),
                t_span=(float(seg_t[0]), float(seg_t[-1])),
                y0=np.asarray(X_Q[i], dtype=float),
                t_eval=seg_t,
                rtol=1e-10,
                atol=1e-12,
            )
            if not sol.success:
                # 稠密输出失败不再用 2 点线性近似顶替（#352）：2 点线性会
                # 污染下游 FFT 频率分析；改由 pipeline 统一降级为 FAILED 结果。
                raise RuntimeError(
                    f"稠密输出积分失败（段 {i}: t∈[{t_lo:.6g}, {t_hi:.6g}]）：{sol.message}"
                )
            # 段 i 的末时刻 = 段 i+1 的初时刻，避免 tlist 出现重复点
            if i < n_seg - 1 and sol.t.size > 1:
                pieces_t.append(sol.t[:-1])
                pieces_X.append(sol.y[:, :-1].T)
            else:
                pieces_t.append(sol.t)
                pieces_X.append(sol.y.T)

        if not pieces_t:
            return t_Q.copy(), X_Q.copy()

        tlist = np.concatenate(pieces_t)
        Xlist = np.concatenate(pieces_X, axis=0)
        return tlist, Xlist

    def _frequency_analysis(
        self,
        tlist: npt.NDArray[np.floating],
        Xlist: npt.NDArray[np.floating],
    ) -> tuple[dict[str, list[FFTComponent]], str]:
        """对 x/y/z 三方向做 NAFF/FFT 频率分析。"""
        result: dict[str, list[FFTComponent]] = {}
        backend = "fft"
        for idx, label in enumerate(("x", "y", "z")):
            comps, used = extract_frequencies(
                tlist,
                Xlist[:, idx],
                n_components=20,
                prefer=self.prefer,
            )
            result[label] = comps
            # 三个方向应使用同一种后端
            if used == "naff":
                backend = "naff"
        return result, backend

    def _build_W(
        self,
        tlist: npt.NDArray[np.floating],
        Xlist: npt.NDArray[np.floating],
        *,
        use_cr3bp: bool,
    ) -> tuple[
        dict[tuple[int, ...], npt.NDArray[np.floating]],
        dict[tuple[int, ...], npt.NDArray[np.floating]],
    ]:
        """由 ``Xlist`` 数值微分得 ``W_poly`` / ``Wdot_poly``。

        ``use_cr3bp`` 为 True（显式 force_cr3bp 或 SPICE 不可用降级）时
        ``_bdot2a`` 走纯 CR3BP 旋转矩阵，不探 SPICE（#352：SPICE 可用时星历
        失败不再静默退化为纯 CR3BP）。
        """
        """由 ``Xlist`` 数值微分得 ``W_poly`` / ``Wdot_poly``。"""
        if tlist.size < 2:
            empty: dict[tuple[int, ...], npt.NDArray[np.floating]] = {}
            return empty, empty

        dt = float(np.mean(np.diff(tlist)))
        B = Xlist[:, :3]
        Bdot = Xlist[:, 3:6]
        # 二阶导：用中心差分，避免引入额外依赖
        Bddot = _second_derivative(Bdot, dt)

        A, Adot = _bdot2a(self.context, B, Bdot, Bddot, tlist, use_cr3bp=use_cr3bp)

        W_poly: dict[tuple[int, ...], npt.NDArray[np.floating]] = {}
        Wdot_poly: dict[tuple[int, ...], npt.NDArray[np.floating]] = {}
        pow_units = [
            (1, 0, 0, 0, 0, 0),
            (0, 1, 0, 0, 0, 0),
            (0, 0, 1, 0, 0, 0),
            (0, 0, 0, 1, 0, 0),
            (0, 0, 0, 0, 1, 0),
            (0, 0, 0, 0, 0, 1),
        ]
        for k, pow_tuple in enumerate(pow_units):
            if k < 3:
                W_poly[pow_tuple] = A[:, k]
                Wdot_poly[pow_tuple] = Adot[:, k]
            else:
                W_poly[pow_tuple] = B[:, k - 3]
                Wdot_poly[pow_tuple] = Bdot[:, k - 3]
        return W_poly, Wdot_poly


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------


def _ode_rhs_via_solver(
    solver: SubstituteSolver, t: float, X: npt.ArrayLike
) -> npt.NDArray[np.floating]:
    """从 :class:`ODESubstituteSolver` 中取出 ``rhs`` 调用。"""
    rhs = getattr(solver, "rhs", None)
    if rhs is None:
        raise RuntimeError("solver 必须暴露 rhs 才能用于稠密输出")
    return np.asarray(rhs(t, X), dtype=float).ravel()


def _second_derivative(y: npt.NDArray[np.floating], dt: float) -> npt.NDArray[np.floating]:
    """等距采样的二阶中心差分；首末端用一阶差分。

    对应 qiao ``list_deriv`` 的两遍应用：``ddot = deriv(deriv(y))``。
    本切片刻意走更简洁的中心差分（足以满足烟雾测试），slice 3 可
    再换成 qiao 风格的高阶 Vandermonde 系数。
    """
    y = np.asarray(y, dtype=float)
    out = np.zeros_like(y)
    if y.shape[0] < 3:
        # 全一阶差分兜底
        if y.shape[0] >= 2:
            out[1:-1] = (y[2:] - 2 * y[1:-1] + y[:-2]) / (dt * dt)
            out[0] = (y[1] - y[0]) / dt
            out[-1] = (y[-1] - y[-2]) / dt
        return out
    out[1:-1] = (y[2:] - 2 * y[1:-1] + y[:-2]) / (dt * dt)
    out[0] = (y[1] - y[0]) / dt
    out[-1] = (y[-1] - y[-2]) / dt
    return out


def _bdot2a(
    context: NormalFormContext,
    B: npt.NDArray[np.floating],
    Bdot: npt.NDArray[np.floating],
    Bddot: npt.NDArray[np.floating],
    tlist: npt.NDArray[np.floating],
    *,
    use_cr3bp: bool,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """``B, Bdot, Bddot`` → ``(A, Adot)``（qiao ``Bdot2A``）。

    A = -Bdot + C_pq @ B
    Adot = -Bddot + (dC_pq) @ B + C_pq @ Bdot

    ``C_pq`` 与 ``dC_pq`` 默认通过 :func:`._ephemeris.eval_params`（SPICE
    星历）取。``use_cr3bp=True``（显式 ``force_cr3bp`` 或 SPICE 不可用降级）
    时走纯 CR3BP（自治）：``C_pq`` 恒为旋转矩阵系数 ``[[0,1,0],[-1,0,0],
    [0,0,0]]``、``dC_pq = 0``，不探 SPICE——这是 CR3BP 中心流形约化的正路，
    不是降级。SPICE 可用（``use_cr3bp=False``）时星历失败抛异常（#352），
    不再静默退化为纯 CR3BP（那会丢星历摄动）。
    """
    B = np.asarray(B, dtype=float)
    Bdot = np.asarray(Bdot, dtype=float)
    Bddot = np.asarray(Bddot, dtype=float)
    tlist = np.asarray(tlist, dtype=float)

    n = B.shape[0]
    if Bdot.shape != B.shape or Bddot.shape != B.shape:
        raise ValueError(f"B/Bdot/Bddot 形状不一致：{B.shape}/{Bdot.shape}/{Bddot.shape}")
    if tlist.shape[0] != n:
        raise ValueError(f"tlist 长度必须等于 B 行数：{tlist.shape[0]} vs {n}")

    if use_cr3bp:
        # 纯 CR3BP（自治系统，显式选择或 SPICE 不可用降级）：C_pq 恒为旋转
        # 矩阵、dC_pq=0，无需 SPICE 星历。
        Cpq_seq: list[np.ndarray] = [_CR3BP_CPQ] * n
        dCpq_seq: list[np.ndarray] = [np.zeros((3, 3))] * n
    else:
        Cpq_seq, dCpq_seq = [], []
        try:
            from ._ephemeris import eval_params as _eval_params

            tu_days = float(context.TU) / 86400.0
            for t in tlist:
                jd = float(context.epoch) + float(t) * tu_days
                params = _eval_params(jd, context)
                cpq = np.array(
                    [
                        [params["Cpq1"], params["Cpq2"], params["Cpq3"]],
                        [params["Cpq4"], params["Cpq5"], params["Cpq6"]],
                        [params["Cpq7"], params["Cpq8"], params["Cpq9"]],
                    ],
                    dtype=float,
                )
                cqq = np.array(
                    [
                        [params["Cqq1"], params["Cqq2"], params["Cqq3"]],
                        [params["Cqq4"], params["Cqq5"], params["Cqq6"]],
                        [params["Cqq7"], params["Cqq8"], params["Cqq9"]],
                    ],
                    dtype=float,
                )
                dcpq = cqq - cpq @ cpq  # d/dt(C_pq) = C_qq - C_pq^2
                Cpq_seq.append(cpq)
                dCpq_seq.append(dcpq)
        except Exception as exc:
            # SPICE 可用但星历参数解析失败：不再静默退化为纯 CR3BP 旋转矩阵
            # （#352）——退化会用错 C_pq 污染 A/Adot、静默丢星历摄动。纯 CR3BP
            # 应显式走 use_cr3bp=True（该路径不探 SPICE）。
            raise RuntimeError(
                f"_ephemeris.eval_params 失败：{exc}；不退化到纯 CR3BP 旋转矩阵"
                f"（如需纯 CR3BP 请显式设置 context.force_cr3bp=True 或允许降级）"
            ) from exc

    A = np.zeros_like(B)
    Adot = np.zeros_like(B)
    for i in range(n):
        cpq_i = Cpq_seq[i]
        dcpq_i = dCpq_seq[i]
        A[i] = -Bdot[i] + cpq_i @ B[i]
        Adot[i] = -Bddot[i] + dcpq_i @ B[i] + cpq_i @ Bdot[i]
    return A, Adot


def _build_dynamics_rhs_circular(
    context: NormalFormContext,
) -> Callable[[float, npt.ArrayLike], npt.NDArray[np.floating]]:
    """纯 CR3BP 地心会合系下的 rho 坐标右端项（无 SPICE 退路）。

    对应 qiao ``Dynfunc_rho.m`` 的 CR3BP 降级（忽略太阳与三体摄动）。
    坐标系与 qiao 一致：地心会合系（地球在原点、月球在 ``(1,0,0)``），
    rho 为平动点相对坐标（原点在平动点 ``r0``）。

    运动方程（平动点平衡项已消去，使 ``rho=0`` 是平衡点）::

        ρ̈ = −μ_e·[(r0+ρ)/|r0+ρ|³ − r0/|r0|³]
            −μ_m·[(r0+ρ−r_m)/|r0+ρ−r_m|³ − (r0−r_m)/|r0−r_m|³]
            −2ω×ρ̇

    其中 ``r0`` 是平动点在地心会合系的位置（如 L2 = 1+γ）、``r_m=(1,0,0)``
    是月球位置、``ω=ẑ``。此形式由 qiao ``Dynfunc_rho`` 第 69 行的
    ``−μ_m·rm/|rm|³ − r0dotdot``（平动点平衡条件）消去常数项得到。
    """
    mu_e = float(context.mu_e)  # 归一化地球引力常数（≈1−μ）
    mu_m = float(context.mu_m)  # 归一化月球引力常数（≈μ）
    r0 = np.asarray(context.libration_position, dtype=float).ravel()
    rm = np.array([1.0, 0.0, 0.0])  # 月球在地心会合系
    omega = np.array([0.0, 0.0, 1.0])
    # 平动点处的引力加速度（平衡条件，用于消去常数项）
    d_e_0 = r0
    d_m_0 = r0 - rm
    grav0 = -mu_e * d_e_0 / np.linalg.norm(d_e_0) ** 3 - mu_m * d_m_0 / np.linalg.norm(d_m_0) ** 3

    def rhs(t: float, X: npt.ArrayLike) -> npt.NDArray[np.floating]:
        X_arr = np.asarray(X, dtype=float).ravel()
        rho = X_arr[:3]
        rhodot = X_arr[3:6]
        d_e = r0 + rho
        d_m = r0 + rho - rm
        grav = -mu_e * d_e / np.linalg.norm(d_e) ** 3 - mu_m * d_m / np.linalg.norm(d_m) ** 3
        coriolis = -2.0 * np.cross(omega, rhodot)
        centrifugal = -np.cross(omega, np.cross(omega, rho))
        rhodotdot = (grav - grav0) + coriolis + centrifugal
        return np.concatenate([rhodot, rhodotdot])

    return rhs


def _build_dynamics_rhs_spice(
    context: NormalFormContext,
) -> tuple[
    Callable[[float, npt.ArrayLike], npt.NDArray[np.floating]],
    Callable[[float], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
]:
    """星历模型 rho 坐标右端项；复用 :mod:`._ephemeris` 推导的 EMR 参数。

    失败时抛 ``RuntimeError``，由 :meth:`_build_dynamics` 决定是否降级。
    """
    from ._ephemeris import eval_params as _eval_params

    def rhs(t: float, X: npt.ArrayLike) -> npt.NDArray[np.floating]:
        X_arr = np.asarray(X, dtype=float).ravel()
        rho = X_arr[:3]
        rhodot = X_arr[3:6]

        tu_days = float(context.TU) / 86400.0
        jd = float(context.epoch) + float(t) * tu_days
        params = _eval_params(jd, context)

        cpq = np.array(
            [
                [params["Cpq1"], params["Cpq2"], params["Cpq3"]],
                [params["Cpq4"], params["Cpq5"], params["Cpq6"]],
                [params["Cpq7"], params["Cpq8"], params["Cpq9"]],
            ],
            dtype=float,
        )
        cqq = np.array(
            [
                [params["Cqq1"], params["Cqq2"], params["Cqq3"]],
                [params["Cqq4"], params["Cqq5"], params["Cqq6"]],
                [params["Cqq7"], params["Cqq8"], params["Cqq9"]],
            ],
            dtype=float,
        )
        force = np.array([params["f1"], params["f2"], params["f3"]], dtype=float)
        rex = np.array([params["rex"], params["rey"], params["rez"]], dtype=float)
        re0 = float(params["re0"])
        rmx = np.array([params["rmx"], params["rmy"], params["rmz"]], dtype=float)
        rm0 = float(params["rm0"])
        rsx = np.array([params["rsx"], params["rsy"], params["rsz"]], dtype=float)
        rs0 = float(params["rs0"])
        mu_e = float(params["mu_e"])
        mu_m = float(params["mu_m"])
        mu_s = float(params["mu_s"])

        d_e = rex + rho  # 平动点相对地球
        d_m = rmx + rho
        d_s = rsx + rho
        d_e3 = float(np.linalg.norm(d_e)) ** 3
        d_m3 = float(np.linalg.norm(d_m)) ** 3
        d_s3 = float(np.linalg.norm(d_s)) ** 3
        r0dotdot = (
            np.array(
                [
                    params.get("r0dotdot_x", 0.0),
                    params.get("r0dotdot_y", 0.0),
                    params.get("r0dotdot_z", 0.0),
                ],
                dtype=float,
            )
            if all(k in params for k in ("r0dotdot_x", "r0dotdot_y", "r0dotdot_z"))
            else np.zeros(3)
        )

        # 简化版：与 qiao dynfunc_rho_core 1:1 公式对齐；把二阶非惯性项
        # 折叠到 cqq 矩阵上。完整推导见 qiao Python/crtbp/Subfunction/dynfunc/dynfunc_rho_core.py。
        rho_dotdot = (
            force
            - cqq @ rho
            - 2.0 * cpq @ rhodot
            + (-mu_e * (rex + rho) / d_e3 - mu_m * (rmx + rho) / d_m3 - mu_s * (rsx + rho) / d_s3)
            - (-mu_e * rex / re0**3 - mu_m * rmx / rm0**3 - mu_s * rsx / rs0**3)
            - r0dotdot
        )

        return np.concatenate([rhodot, rho_dotdot])

    # provider：仅供稠密输出诊断使用；与 qiao make_spice_provider 一致
    def provider(t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        tu_days = float(context.TU) / 86400.0
        jd = float(context.epoch) + float(t) * tu_days
        from ._ephemeris import _ephemeris_states

        r_em, v_em, r_es, v_es = _ephemeris_states(jd)
        return r_em, v_em, r_es, v_es

    return rhs, provider


__all__ = [
    "DEFAULT_TOTAL_TU",
    "DEFAULT_NODE_STEP",
    "DEFAULT_DENSE_STEP",
    "DEFAULT_MAX_ITER",
    "DEFAULT_TOLERANCE",
    "DynamicalSubstituteCorrector",
    "DynamicalSubstituteResult",
]
