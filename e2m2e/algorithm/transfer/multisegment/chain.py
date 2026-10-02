"""广义多重打靶组装：leg 切分、matchpoint 连续性约束与雅可比拼装（#726）。

框架只做问题组装（ADR 0053）：逐 leg 前向/后向 pass 到本 leg 匹配点、
6 维连续性残差、归一化等式约束雅可比、段中点位置灵敏度与跨 leg 窗口
平移灵敏度。NLP 求解不在本层（消费方走 ``nlp_scipy`` / ``multisegment.solve``）。

pass 与匹配点组装原属 ``sims_flanagan.py`` 的 ``_leg_pass`` / ``_evaluate``
（#740/#741），搬移时数值行为逐行保持；新增的**窗口平移灵敏度**（跨 leg
TOF 星历耦合）见 :func:`evaluate_chain` 的 docstring。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .kernels import LegKernel

__all__ = ["ChainLegEval", "ChainLegRequest", "evaluate_chain"]

#: 冲量对 6 维状态的映射矩阵 [[0₃], [I₃]]（冲量只改速度分量）。
_B_IMPULSE: npt.NDArray[np.floating] = np.zeros((6, 3))
_B_IMPULSE[3:, :] = np.eye(3)


@dataclass(frozen=True)
class ChainLegRequest:
    """单 leg 的组装请求（锚定与解包由调用方传入，框架不自持节点语义）。

    Attributes:
        kernel: 本 leg 传播内核（逐 leg 可混档）。
        n_segments: 段数 ``n ≥ 2``；匹配点取 ``m = n // 2``。
        imp_offset: 本 leg 冲量块在全局决策向量中的起始列。
        dv: 段中冲量 ``(n, 3)``（km/s），按段升序。
        dt: 段长（s）``= 本 leg TOF / n``。
        t_start: 本 leg 起始绝对历元（星历档必填：前向游标自它出发、后向
            游标自 ``t_start + n·dt`` 反向递减；conic 档忽略）。
        tof_col: 本 leg TOF 时长导数落点列（TOF 为决策变量时给定；固定
            TOF 为 ``None``）。
        anchor_f: 前向锚 ``(6,)``（出发态或节点态 + v∞_out）。
        anchor_b: 后向锚 ``(6,)``（到达态或节点态 + v∞_in）。
        anchor_sens_f: 前向锚对全局决策变量的灵敏度 ``(6, n_var)``。
        anchor_sens_b: 后向锚对全局决策变量的灵敏度 ``(6, n_var)``。
        shift_cols_f: 前向 pass 窗口平移灵敏度落点列（前序 leg 的 TOF 列）；
            ``None`` 表示无（conic 档与单 leg 恒为 ``None``）。
        shift_cols_b: 后向 pass 窗口平移灵敏度落点列（前序 leg TOF 列 ∪
            本 leg TOF 列，后向锚随二者平移）；``None`` 表示无。
    """

    kernel: LegKernel
    n_segments: int
    imp_offset: int
    dv: npt.ArrayLike
    dt: float
    anchor_f: npt.ArrayLike
    anchor_b: npt.ArrayLike
    anchor_sens_f: npt.ArrayLike
    anchor_sens_b: npt.ArrayLike
    t_start: float | None = None
    tof_col: int | None = None
    shift_cols_f: npt.ArrayLike | None = None
    shift_cols_b: npt.ArrayLike | None = None


@dataclass(frozen=True)
class ChainLegEval:
    """单 leg 的组装产物（值 + 灵敏度）。

    Attributes:
        fwd_states: 前向节点状态 ``(m+1, 6)``（出发锚…匹配点）。
        bwd_states: 后向节点状态 ``(n-m+1, 6)``（匹配点…到达锚）。
        residual_raw: 匹配点原始残差 ``(6,)`` = ``fwd[-1] − bwd[0]``。
        eq_jac: 归一化等式约束雅可比 ``(6, n_var)``（位置行除以
            ``len_scale``、速度行除以 ``vel_scale`` 的 fwd−bwd 灵敏度差，
            含窗口平移列贡献）；``with_sens=False`` 时为 ``None``。
        mid_pos: 段中点位置 ``(n, 3)``（前向段升序接后向段升序）。
        r_mid: 段中点半径 ``(n,)``。
        spos: 段中点位置灵敏度 ``(n, 3, n_var)``（含窗口平移列贡献）。
        smid_shift: 段中点位置对窗口平移的灵敏度 ``(n, 3)``（无 shift 列
            或 conic 档时为零矩阵）。
    """

    fwd_states: npt.NDArray[np.floating]
    bwd_states: npt.NDArray[np.floating]
    residual_raw: npt.NDArray[np.floating]
    eq_jac: npt.NDArray[np.floating] | None
    mid_pos: npt.NDArray[np.floating]
    r_mid: npt.NDArray[np.floating]
    spos: npt.NDArray[np.floating]
    smid_shift: npt.NDArray[np.floating]


def evaluate_chain(
    legs: Sequence[ChainLegRequest],
    *,
    n_var: int,
    len_scale: float,
    vel_scale: float,
    with_sens: bool,
) -> list[ChainLegEval]:
    """逐 leg 组装多重打靶问题（前向/后向 pass + matchpoint 连续性）。

    每 leg 独立前向/后向 pass 至本 leg 匹配点（leg 间经节点锚定解耦，
    位置匹配由锚定结构性满足）；等式约束为归一化 6 维匹配点残差，雅可比
    为 fwd−bwd 灵敏度差。传播异常（闭式 Kepler 病态能量 ``ValueError``）
    向上抛，由调用方惩罚评估兜底。

    **窗口平移灵敏度（跨 leg TOF 星历耦合，#726）**：``with_sens`` 且内核
    ``absolute_time`` 且 shift 列非空时，在 pass 循环内维护 6 维向量 ``s``
    （当前状态对整个 leg 窗口统一平移 δ 的导数，锚态为固定数据故初值
    为零）。递推：每个半段传播后 ``s ← Φ·s + f(t_out, x_out) − Φ·f(t_in,
    x_in)``（``x_in`` 取该半段输入态；冲量不改变 ``s``；``half < 0`` 后向
    同式成立）。段中点处记录 ``smid_shift = s[:3]``；落列为
    ``eq_jac[:, c] += 归一化(s_f) − 归一化(s_b)``、``spos[:, :, c] +=
    smid_shift``。时间不变动力学下 ``Φ·f_in = f_out`` 恒成立，该项逐位为零
    （退化 parity 不变）；conic 档完全跳过该分支（不引入数值噪声）。TOF
    时长伸缩项（本 leg 段长随 TOF 变化）由 ``tof_col`` 链承担，与窗口平移
    项相加即总导数。

    Args:
        legs: 逐 leg 组装请求（按链序）。
        n_var: 全局决策变量数。
        len_scale: 位置归一化尺度（km）。
        vel_scale: 速度归一化尺度（km/s）。
        with_sens: 是否组装灵敏度（雅可比与段中点灵敏度）。

    Returns:
        逐 leg 的 :class:`ChainLegEval`（与 ``legs`` 等长同序）。

    Raises:
        ValueError: 请求字段形状/取值非法。
    """
    return [
        _evaluate_leg(
            leg,
            n_var=n_var,
            len_scale=len_scale,
            vel_scale=vel_scale,
            with_sens=with_sens,
        )
        for leg in legs
    ]


def _evaluate_leg(
    req: ChainLegRequest,
    *,
    n_var: int,
    len_scale: float,
    vel_scale: float,
    with_sens: bool,
) -> ChainLegEval:
    """单 leg 的前向/后向 pass 与 matchpoint 组装（``_leg_pass`` 的框架化）。"""
    n = int(req.n_segments)
    if n < 2:
        raise ValueError(f"n_segments 必须为 ≥ 2 的整数，得到 {req.n_segments!r}")
    if req.imp_offset < 0 or req.imp_offset + 3 * n > n_var:
        raise ValueError(f"imp_offset={req.imp_offset} 超出决策变量布局 n_var={n_var}")
    if req.tof_col is not None and not 0 <= int(req.tof_col) < n_var:
        raise ValueError(f"tof_col={req.tof_col} 超出决策变量布局 n_var={n_var}")
    if len_scale <= 0.0 or vel_scale <= 0.0:
        raise ValueError(f"len_scale/vel_scale 必须为正，得到 {len_scale}/{vel_scale}")
    if req.kernel.absolute_time and req.t_start is None:
        raise ValueError("星历档 leg 必须提供起始绝对历元 t_start")

    m = n // 2
    fwd = _run_pass(req, forward=True, n_var=n_var, count=m, with_sens=with_sens)
    bwd = _run_pass(req, forward=False, n_var=n_var, count=n - m, with_sens=with_sens)

    eq_jac = None
    if with_sens:
        sens_f, s_shift_f = fwd[0], fwd[2]
        sens_b, s_shift_b = bwd[0], bwd[2]
        eq_jac = np.vstack([sens_f[:3] / len_scale, sens_f[3:] / vel_scale]) - np.vstack(
            [sens_b[:3] / len_scale, sens_b[3:] / vel_scale]
        )
        shift_f = _shift_array(req.shift_cols_f)
        shift_b = _shift_array(req.shift_cols_b)
        if shift_f is not None:
            eq_jac[:, shift_f] += _normalize(s_shift_f, len_scale, vel_scale)[:, None]
        if shift_b is not None:
            eq_jac[:, shift_b] -= _normalize(s_shift_b, len_scale, vel_scale)[:, None]

    return ChainLegEval(
        fwd_states=fwd[3],
        bwd_states=bwd[3],
        residual_raw=fwd[3][-1] - bwd[3][0],
        eq_jac=eq_jac,
        mid_pos=np.concatenate([fwd[4], bwd[4]]),
        r_mid=np.concatenate([fwd[5], bwd[5]]),
        spos=np.concatenate([fwd[1], bwd[1]], axis=0),
        smid_shift=np.concatenate([fwd[6], bwd[6]]),
    )


def _run_pass(
    req: ChainLegRequest,
    *,
    forward: bool,
    n_var: int,
    count: int,
    with_sens: bool,
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
]:
    """单 leg 的前向/后向 pass（Sims-Flanagan ``_leg_pass`` 的框架化）。

    ``forward=True`` 自前向锚接龙段 ``0..m−1``，否则自后向锚反向接龙段
    ``m..n−1``（内核支持负 dt）。返回 ``(匹配节点灵敏度 (6, n_var), 段中点
    位置灵敏度 (count, 3, n_var), 窗口平移终值灵敏度 (6,), 节点状态, 段中点
    位置, 段中点半径, 段中点窗口平移灵敏度 (count, 3))``；后向的节点状态按
    ``[匹配节点..锚]`` 排列、段序数组按段升序翻转。
    """
    n = req.n_segments
    dv = np.asarray(req.dv, dtype=float).reshape(n, 3)
    sign = 1.0 if forward else -1.0
    half = sign * 0.5 * req.dt
    kernel = req.kernel
    shift_cols = _shift_array(req.shift_cols_f if forward else req.shift_cols_b)
    track_shift = with_sens and kernel.absolute_time and shift_cols is not None

    x = np.asarray(req.anchor_f if forward else req.anchor_b, dtype=float).copy()
    sens = np.asarray(req.anchor_sens_f if forward else req.anchor_sens_b, dtype=float).copy()
    if sens.shape != (6, n_var):
        raise ValueError(f"锚灵敏度形状必须为 (6, {n_var})，得到 {sens.shape}")
    s_shift = np.zeros(6)
    seg_ids = range(count) if forward else range(n - 1, n - count - 1, -1)
    states = [x.copy()]
    mid_pos = np.zeros((count, 3))
    r_mid = np.zeros(count)
    spos = np.zeros((count, 3, n_var))
    smid_shift = np.zeros((count, 3))
    t = None if req.t_start is None else (req.t_start if forward else req.t_start + n * req.dt)
    for idx, k in enumerate(seg_ids):
        t_mid = None if t is None else t + half
        x_mid, phi1 = kernel.propagate_half(x, t, half, with_stm=with_sens)
        mid_pos[idx] = x_mid[:3]
        r_mid[idx] = float(np.linalg.norm(x_mid[:3]))
        if with_sens:
            assert phi1 is not None  # with_stm=True 时 propagate_half 恒返回 STM
            s_mid = phi1 @ sens
            rhs_mid = kernel.rhs(t_mid, x_mid) if (req.tof_col is not None or track_shift) else None
            if req.tof_col is not None:
                assert rhs_mid is not None
                # 半段时长 h = ±TOF/(2n)：∂x_out/∂TOF += RHS(x_out)·∂h/∂TOF。
                s_mid[:, req.tof_col] += sign * rhs_mid / (2.0 * n)
            if track_shift:
                assert rhs_mid is not None
                f_in = kernel.rhs(t, x)
                s_shift = phi1 @ s_shift + rhs_mid - phi1 @ f_in
                smid_shift[idx] = s_shift[:3]
            spos[idx] = s_mid[:3, :]
            if track_shift:
                spos[idx][:, shift_cols] += smid_shift[idx][:, None]
            s_mid[:, req.imp_offset + 3 * k : req.imp_offset + 3 * k + 3] += sign * _B_IMPULSE
        x_mid = x_mid + sign * (_B_IMPULSE @ dv[k])
        t_out = None if t_mid is None else t_mid + half  # 传播输出时刻（与 x 同步）
        x, phi2 = kernel.propagate_half(x_mid, t_mid, half, with_stm=with_sens)
        if with_sens:
            assert phi2 is not None
            sens = phi2 @ s_mid
            rhs_out = kernel.rhs(t_out, x) if (req.tof_col is not None or track_shift) else None
            if req.tof_col is not None:
                assert rhs_out is not None
                sens[:, req.tof_col] += sign * rhs_out / (2.0 * n)
            if track_shift:
                assert rhs_out is not None
                f_in = kernel.rhs(t_mid, x_mid)
                # 冲量是窗口相对事件（不随历元平移改变），s_shift 无冲量项。
                s_shift = phi2 @ s_shift + rhs_out - phi2 @ f_in
        states.append(x.copy())
        t = t_out
    if forward:
        return sens, spos, s_shift, np.asarray(states), mid_pos, r_mid, smid_shift
    return (
        sens,
        spos[::-1],
        s_shift,
        np.asarray(states[::-1]),
        mid_pos[::-1],
        r_mid[::-1],
        smid_shift[::-1],
    )


def _shift_array(shift_cols: npt.ArrayLike | None) -> npt.NDArray[np.integer] | None:
    """shift 列归一为 int 数组；``None`` 或空集统一返回 ``None``。"""
    if shift_cols is None:
        return None
    arr = np.asarray(shift_cols, dtype=int).reshape(-1)
    return arr if arr.size > 0 else None


def _normalize(
    s: npt.NDArray[np.floating], len_scale: float, vel_scale: float
) -> npt.NDArray[np.floating]:
    """6 维向量按约束归一化尺度缩放（位置/len、速度/vel）。"""
    return np.concatenate([s[:3] / len_scale, s[3:] / vel_scale])
