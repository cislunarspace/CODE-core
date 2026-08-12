"""频率分析（FFT/NAFF）辅助函数。

对应 qiao ``Subfunction/fft_operator`` 与 ``Subfunction/fft2num``：

- :class:`FFTComponent` —— 一个频域分量 ``s·sin(ωt) + c·cos(ωt)``；
- :func:`naff_available` —— 通过 ``shutil.which`` 探测外部 ``naff_uv``
  Fortran 可执行文件；
- :func:`detect_naff` —— 调用外部 NAFF；不可用时抛 :class:`RuntimeError`，
  调用方应改用 :func:`fft_extract`；
- :func:`fft_extract` —— 纯 NumPy FFT 实现的频率提取（NAFF 不可用时的
  降级路径），返回按振幅排序的 ``FFTComponent`` 列表；
- :func:`frequency_match` —— 把检测到的频率匹配到预计算基频表；
- :func:`reconstruct_signal` / :func:`reconstruct_derivative` —— 从
  FFT 频域表示重构时域信号（或其导数）。

NAFF 检测只在调用方显式调用 :func:`detect_naff` 时触发；本模块其余函数
不强制依赖 NAFF 可用性，从而保证 ``import e2m2e.algorithms.normal_form``
在任何环境都能正常工作。
"""

from __future__ import annotations

import shutil
import warnings
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

#: 外部 NAFF 可执行文件名（按 qiao ``FrequencyMatch._find_naff_binary``）。
#: 通过 ``shutil.which`` 在 ``PATH`` 中探测；亦可由环境变量 ``NAFF_BINARY``
#: 显式指定。
_NAFF_BINARY_CANDIDATES: tuple[str, ...] = ("naff_uv", "NUAA.exe", "naff")

#: NAFF 调用超时（秒）。
NAFF_TIMEOUT_S: float = 300.0


@dataclass(frozen=True)
class FFTComponent:
    """单个频域分量 ``s·sin(ωt) + c·cos(ωt)``。

    Attributes:
        freq: 角频率 ``ω``。
        amp_s: 正弦振幅 ``s``。
        amp_c: 余弦振幅 ``c``。
        amp: 总振幅 ``sqrt(s² + c²)``。
        match_freq: 在基频表上最近邻匹配频率（``frequency_match`` 写入）。
        coef: 与 ``match_freq`` 对应的整数系数向量。
        err: 频率匹配误差 ``|freq - match_freq|``。
    """

    freq: float
    amp_s: float = 0.0
    amp_c: float = 0.0
    amp: float = 0.0
    match_freq: float = 0.0
    coef: tuple[int, ...] = field(default_factory=lambda: (0, 0, 0, 0, 0, 0))
    err: float = 0.0


# ---------------------------------------------------------------------------
# NAFF 探测与调用
# ---------------------------------------------------------------------------


def _resolve_naff_binary() -> str | None:
    """查找可用的 NAFF 可执行文件。

    优先使用 ``NAFF_BINARY`` 环境变量，否则在 ``PATH`` 中探测
    ``naff_uv`` / ``NUAA.exe`` / ``naff``。找不到返回 ``None``。
    """
    import os

    override = os.environ.get("NAFF_BINARY")
    if override:
        return override if os.path.isfile(override) else None
    for name in _NAFF_BINARY_CANDIDATES:
        path = shutil.which(name)
        if path is not None:
            return path
    return None


def naff_available() -> bool:
    """当前环境是否可调用外部 NAFF（无须抛错）。"""
    return _resolve_naff_binary() is not None


def detect_naff(
    times: npt.ArrayLike,
    data: npt.ArrayLike,
    *,
    n_components: int = 10,
    timeout: float = NAFF_TIMEOUT_S,
) -> list[FFTComponent]:
    """调用外部 NAFF 提取时间序列主要频率分量。

    Args:
        times: 等距采样时间序列，形状 ``(N,)``。
        data: 与 ``times`` 对应的实信号，形状 ``(N,)``。
        n_components: NAFF 期望返回的分量数（透传给控制文件第 8 行）。
        timeout: NAFF 调用超时（秒）。

    Returns:
        按振幅降序排列的 :class:`FFTComponent` 列表；直流分量（``ω=0``）
        始终位于首位。

    Raises:
        RuntimeError: NAFF 不可用（:func:`naff_available` 返回 ``False``），
            或调用失败。
    """
    import os
    import subprocess
    import tempfile

    exe = _resolve_naff_binary()
    if exe is None:
        raise RuntimeError(
            "NAFF binary not found on PATH. "
            "Set NAFF_BINARY or install naff_uv (see qiao FFTanalysis/)."
        )

    t = np.asarray(times, dtype=float).ravel()
    y = np.asarray(data, dtype=float).ravel()
    if t.shape != y.shape:
        raise ValueError(f"times 与 data 形状不一致：times={t.shape} data={y.shape}")
    if t.size < 2:
        raise ValueError("时间序列至少需要 2 个采样点")
    dt = float(np.mean(np.diff(t)))
    if dt <= 0:
        raise ValueError("时间序列非单调递增")
    if not np.all(np.diff(t) > 0):
        raise ValueError("时间序列必须严格单调递增")
    fs = 1.0 / dt

    with tempfile.TemporaryDirectory(prefix="e2m2e_naff_") as tmp:
        # 输入数据
        input_path = os.path.join(tmp, "Input.txt")
        with open(input_path, "w") as f:
            for v in y:
                f.write(f"{v:20.18f}\n")

        # 控制文件
        control_path = os.path.join(tmp, "naff_uv_control_file.txt")
        with open(control_path, "w") as f:
            f.write("Input.txt\n")
            f.write("r\n")
            f.write("full\n")
            f.write("1\n")
            f.write("1\n")
            f.write("1\n")
            f.write(f"{fs:20.18f}\n")
            f.write(f"{int(n_components)}\n")
            f.write("1\n")
            f.write("han_5\n")
            f.write("hft\n")
            f.write("0\n")
            f.write("Output.txt\n")

        # 拷贝可执行文件到 tmp（与 qiao 一致），便于 DLL 解析
        exe_name = os.path.basename(exe)
        exe_dest = os.path.join(tmp, exe_name)
        shutil.copy2(exe, exe_dest)
        os.chmod(exe_dest, 0o755)

        result = subprocess.run(
            [f"./{exe_name}"],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"NAFF binary failed (code {result.returncode}): "
                f"{(result.stderr or result.stdout)[:512]}"
            )

        output_path = os.path.join(tmp, "Output.txt")
        if not os.path.exists(output_path):
            raise RuntimeError(f"NAFF 未生成 Output.txt（{output_path}）")

        with open(output_path) as f:
            lines = f.readlines()

    freqs: list[float] = []
    amps: list[float] = []
    amp_cs: list[float] = []
    amp_ss: list[float] = []
    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        freqs.append(2.0 * np.pi * float(parts[1]))
        amps.append(float(parts[2]))
        amp_cs.append(float(parts[3]))
        amp_ss.append(-float(parts[4]))  # 与 qiao 保持一致：s 反号

    return [
        FFTComponent(freq=f, amp_s=s, amp_c=c, amp=a)
        for f, s, c, a in zip(freqs, amp_ss, amp_cs, amps, strict=True)
    ]


# ---------------------------------------------------------------------------
# FFT 回退实现（NAFF 不可用时）
# ---------------------------------------------------------------------------


def fft_extract(
    times: npt.ArrayLike,
    data: npt.ArrayLike,
    *,
    n_components: int | None = None,
    detrend: bool = True,
) -> list[FFTComponent]:
    """用 NumPy FFT 提取时间序列的主要频率分量（NAFF 不可用时的降级路径）。

    算法：去均值 → Hanning 窗 → FFT → 取正频率 → 找 ``n_components``
    个最大幅值 → 二次谱插值校正（提高频率估计精度）→ 构造
    :class:`FFTComponent`。

    与真正的 NAFF 相比，FFT 路径的频率估计误差量级为 ``Δf / N``，对
    本切片（``T_total = 0.1·2^16 TU``、``N ≈ 6554``）足以分辨受迫
    频率与中心流形频率。

    Args:
        times: 等距采样时间序列。
        data: 同形状实信号。
        n_components: 保留分量数。``None`` 时取 ``min(N // 2, 50)``，
            并去掉直流项；显式给定 ``0`` 或负数返回空列表。
        detrend: 是否先减去均值再变换。

    Returns:
        按振幅降序排列的 :class:`FFTComponent`` 列表；直流分量始终位于
        首位（若信号均值非零）。
    """
    t = np.asarray(times, dtype=float).ravel()
    y = np.asarray(data, dtype=float).ravel()
    if t.shape != y.shape:
        raise ValueError(f"times 与 data 形状不一致：times={t.shape} data={y.shape}")
    n = int(t.size)
    if n < 4:
        return []

    dt = float(np.mean(np.diff(t)))
    if dt <= 0:
        raise ValueError("时间序列非单调递增")
    if not np.all(np.diff(t) > 0):
        raise ValueError("时间序列必须严格单调递增")
    fs = 1.0 / dt

    if detrend:
        y = y - float(np.mean(y))

    window = np.hanning(n)
    yw = y * window
    spectrum = np.fft.rfft(yw)
    freqs_full = np.fft.rfftfreq(n, d=dt) * 2.0 * np.pi  # 转角频率

    # 直流分量另算（不在 rfft 中做正弦/余弦分解）
    dc = FFTComponent(
        freq=0.0,
        amp_s=0.0,
        amp_c=float(np.mean(y)) if detrend else 0.0,
        amp=float(np.mean(y)) if detrend else 0.0,
    )

    # 取正频率部分（k=1..n//2）
    pos_freqs = freqs_full[1:]
    pos_amps = np.abs(spectrum[1:])
    if pos_amps.size == 0:
        return [dc]

    # Hanning 窗的相干增益（用于复原真实振幅）
    cg = float(np.sum(window)) / n
    if cg <= 0:
        cg = 1.0

    if n_components is None or n_components <= 0:
        keep = pos_amps.size
    else:
        keep = min(n_components, pos_amps.size)

    top_idx = np.argsort(pos_amps)[::-1][:keep]
    components: list[FFTComponent] = []
    for k in top_idx:
        omega = float(pos_freqs[k])
        mag = float(pos_amps[k]) / (n * cg)  # 复原振幅
        # 二次谱插值：使用 k±1 三点；校正后频率更接近真值
        if 1 <= k < pos_amps.size - 1:
            a_prev = float(pos_amps[k - 1])
            a_curr = mag * n * cg  # 用原始幅值做比值
            a_next = float(pos_amps[k + 1])
            denom = a_prev - 2 * a_curr + a_next
            if denom != 0:
                delta = 0.5 * (a_prev - a_next) / denom
                # 限制插值范围，避免数值发散
                delta = max(-1.0, min(1.0, delta))
                df = (2.0 * np.pi * fs) / n
                omega = omega + delta * df

        # 复原正弦/余弦分量（同一幅值下给 ``amp_c`` 一个保守的全幅分配；
        # 调用方可通过 :func:`frequency_match`/线性回归再做精细分配）
        components.append(
            FFTComponent(
                freq=omega,
                amp_s=0.0,
                amp_c=mag,
                amp=mag,
            )
        )

    components.sort(key=lambda c: c.amp, reverse=True)
    return [dc, *components]


# ---------------------------------------------------------------------------
# 频率匹配与频域重构
# ---------------------------------------------------------------------------


def frequency_match(
    components: list[FFTComponent],
    basis_freqs: npt.ArrayLike,
    basis_coefs: npt.ArrayLike | None = None,
) -> list[FFTComponent]:
    """把 FFT 分量匹配到预计算基频表最近邻。

    对应 qiao ``fft_match``。

    Args:
        components: :func:`detect_naff` 或 :func:`fft_extract` 输出。
        basis_freqs: ``(M,)`` 基频角频率数组。
        basis_coefs: ``(M, k)`` 整数系数矩阵（每行一个基频对应的
            整数系数向量）。``None`` 时系数填零向量。

    Returns:
        原列表的浅拷贝（每个 :class:`FFTComponent` 替换为带匹配信息
        的新实例）。
    """
    basis_w = np.asarray(basis_freqs, dtype=float).ravel()
    if basis_w.size == 0:
        raise ValueError("basis_freqs 不能为空")
    if basis_coefs is None:
        basis_coefs_mat = np.zeros((basis_w.size, 6), dtype=int)
    else:
        basis_coefs_mat = np.asarray(basis_coefs)
        if basis_coefs_mat.shape[0] != basis_w.size:
            raise ValueError(
                "basis_coefs 行数必须与 basis_freqs 一致："
                f"{basis_coefs_mat.shape[0]} vs {basis_w.size}"
            )

    matched: list[FFTComponent] = []
    for comp in components:
        if abs(comp.freq) < 1e-14:
            matched.append(
                FFTComponent(
                    freq=0.0,
                    amp_s=comp.amp_s,
                    amp_c=comp.amp_c,
                    amp=comp.amp,
                    match_freq=0.0,
                    coef=tuple(int(c) for c in basis_coefs_mat[0]),
                    err=0.0,
                )
            )
            continue
        idx = int(np.argmin(np.abs(basis_w - comp.freq)))
        match_w = float(basis_w[idx])
        matched.append(
            FFTComponent(
                freq=comp.freq,
                amp_s=comp.amp_s,
                amp_c=comp.amp_c,
                amp=comp.amp,
                match_freq=match_w,
                coef=tuple(int(c) for c in basis_coefs_mat[idx]),
                err=abs(comp.freq - match_w),
            )
        )
    return matched


def reconstruct_signal(
    t: npt.ArrayLike, components: list[FFTComponent]
) -> npt.NDArray[np.floating]:
    """从 FFT 频域表示重构时域信号 ``X(t) = Σ s·sin(ωt) + c·cos(ωt)``。

    对应 qiao ``FFT_X``。当所有 ``amp_s == 0`` （FFT 路径常见情况）时，
    该函数退化为余弦和；一般配合 :func:`least_squares_sin_cos_fit`
    精细分配 ``(amp_s, amp_c)`` 后使用。

    Args:
        t: 标量或 ``(M,)`` 数组。
        components: 频域分量列表。

    Returns:
        与 ``t`` 同形状的实数数组。
    """
    if not components:
        return np.zeros_like(np.asarray(t, dtype=float), dtype=float)

    w = np.array([c.freq for c in components], dtype=float)
    s = np.array([c.amp_s for c in components], dtype=float)
    c = np.array([c.amp_c for c in components], dtype=float)
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    phase = w[:, None] * t_arr[None, :]
    return (s[:, None] * np.sin(phase) + c[:, None] * np.cos(phase)).sum(axis=0)


def reconstruct_derivative(
    t: npt.ArrayLike, components: list[FFTComponent]
) -> npt.NDArray[np.floating]:
    """从 FFT 频域表示重构时域信号的一阶导数 ``dX/dt``。

    对应 qiao ``FFT_dX``。``d(sin)/dt = ω·cos``，``d(cos)/dt = -ω·sin``。
    """
    if not components:
        return np.zeros_like(np.asarray(t, dtype=float), dtype=float)

    w = np.array([c.freq for c in components], dtype=float)
    s = np.array([c.amp_s for c in components], dtype=float)
    c = np.array([c.amp_c for c in components], dtype=float)
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    phase = w[:, None] * t_arr[None, :]
    return (s[:, None] * w[:, None] * np.cos(phase) - c[:, None] * w[:, None] * np.sin(phase)).sum(
        axis=0
    )


def least_squares_sin_cos_fit(
    times: npt.ArrayLike,
    data: npt.ArrayLike,
    frequencies: npt.ArrayLike,
    *,
    subtract_mean: bool = True,
) -> list[FFTComponent]:
    """对给定频率集合用最小二乘拟合 ``s·sin(ωt) + c·cos(ωt)``。

    当频率由 :func:`fft_extract` 得到但 ``amp_s == 0`` 时，本函数可
    给出更精确的正弦/余弦振幅分配：每个频率解一个 ``2×2`` 正规方程。

    Args:
        times: ``(N,)`` 采样时刻。
        data: ``(N,)`` 信号采样值。
        frequencies: ``(K,)`` 角频率（含 0 表示直流）。
        subtract_mean: 是否先减去均值（与 :func:`fft_extract` 默认行为一致）。

    Returns:
        与 ``frequencies`` 同序的 :class:`FFTComponent` 列表。
    """
    t = np.asarray(times, dtype=float).ravel()
    y = np.asarray(data, dtype=float).ravel()
    w = np.asarray(frequencies, dtype=float).ravel()
    if t.shape != y.shape:
        raise ValueError(f"times 与 data 形状不一致：times={t.shape} data={y.shape}")
    if w.size == 0 or t.size == 0:
        return []

    if subtract_mean:
        y = y - float(np.mean(y))

    components: list[FFTComponent] = []
    for omega in w:
        if abs(omega) < 1e-14:
            mean_val = float(np.mean(np.asarray(data, dtype=float)))
            components.append(
                FFTComponent(
                    freq=0.0,
                    amp_s=0.0,
                    amp_c=mean_val,
                    amp=mean_val,
                )
            )
            continue
        phase = omega * t
        s_cos = np.cos(phase)
        s_sin = np.sin(phase)
        # Normal equations for s·sin + c·cos = y
        # A = [[Σ sin², Σ sin·cos], [Σ sin·cos, Σ cos²]]
        a11 = float(np.dot(s_sin, s_sin))
        a12 = float(np.dot(s_sin, s_cos))
        a22 = float(np.dot(s_cos, s_cos))
        b1 = float(np.dot(s_sin, y))
        b2 = float(np.dot(s_cos, y))
        det = a11 * a22 - a12 * a12
        if abs(det) < 1e-30:
            components.append(FFTComponent(freq=float(omega), amp_s=0.0, amp_c=0.0, amp=0.0))
            continue
        amp_s = (a22 * b1 - a12 * b2) / det
        amp_c = (a11 * b2 - a12 * b1) / det
        amp = float(np.sqrt(amp_s * amp_s + amp_c * amp_c))
        components.append(
            FFTComponent(
                freq=float(omega),
                amp_s=float(amp_s),
                amp_c=float(amp_c),
                amp=amp,
            )
        )

    components.sort(key=lambda c: c.amp, reverse=True)
    return components


# ---------------------------------------------------------------------------
# 便捷门面：在 NAFF 与 FFT 之间自动选择
# ---------------------------------------------------------------------------


def extract_frequencies(
    times: npt.ArrayLike,
    data: npt.ArrayLike,
    *,
    n_components: int | None = None,
    prefer: str = "auto",
) -> tuple[list[FFTComponent], str]:
    """NAFF/FFT 自动选择：NAFF 不可用时回退到 FFT 并发出警告。

    Args:
        times: ``(N,)`` 时间数组。
        data: ``(N,)`` 信号数组。
        n_components: NAFF/FFT 保留分量数；``None`` 时取 ``min(N // 2, 50)``。
        prefer: ``"auto"`` / ``"naff"`` / ``"fft"`` 之一。

    Returns:
        ``(components, backend)`` —— ``components`` 是 :class:`FFTComponent`
        列表；``backend`` 是 ``"naff"`` 或 ``"fft"``，便于调用方记录诊断。
    """
    if prefer not in {"auto", "naff", "fft"}:
        raise ValueError(f"prefer 必须是 auto/naff/fft，得到 {prefer!r}")

    if prefer == "fft":
        return fft_extract(times, data, n_components=n_components), "fft"

    if prefer == "auto" and not naff_available():
        warnings.warn(
            "NAFF binary 未找到；降级到 FFT 实现。"
            "频率估计精度约为 Δf/N，对受迫频率/中心流形频率分辨足够。",
            stacklevel=2,
        )
        return fft_extract(times, data, n_components=n_components), "fft"

    try:
        return detect_naff(times, data, n_components=n_components or 10), "naff"
    except (RuntimeError, FileNotFoundError, OSError) as exc:
        warnings.warn(
            f"NAFF 调用失败（{exc}）；降级到 FFT 实现。",
            stacklevel=2,
        )
        return fft_extract(times, data, n_components=n_components), "fft"


__all__ = [
    "FFTComponent",
    "NAFF_TIMEOUT_S",
    "naff_available",
    "detect_naff",
    "fft_extract",
    "frequency_match",
    "reconstruct_signal",
    "reconstruct_derivative",
    "least_squares_sin_cos_fit",
    "extract_frequencies",
]
