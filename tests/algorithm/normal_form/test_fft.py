"""``normal_form.fft`` 测试。

覆盖：

- :func:`naff_available` 不抛错；
- :func:`fft_extract` 在合成单频信号上恢复频率与振幅；
- :func:`least_squares_sin_cos_fit` 给出正确的正弦/余弦分配；
- :func:`reconstruct_signal` / :func:`reconstruct_derivative` 与
  解析值匹配；
- :func:`frequency_match` 把检测频率匹配到基频表最近邻；
- :func:`extract_frequencies` 在 NAFF 不可用时降级到 FFT 并发出警告。
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from e2m2e.algorithm.normal_form.fft import (
    FFTComponent,
    extract_frequencies,
    fft_extract,
    frequency_match,
    least_squares_sin_cos_fit,
    naff_available,
    reconstruct_derivative,
    reconstruct_signal,
)

pytestmark = pytest.mark.theory


# ---------------------------------------------------------------------------
# 公共合成数据
# ---------------------------------------------------------------------------


def _make_single_tone(
    omega: float, amp_s: float, amp_c: float, n: int = 256, t_end: float = 32.0
) -> tuple[np.ndarray, np.ndarray]:
    """构造 ``s·sin(ωt) + c·cos(ωt)`` 时间序列。"""
    t = np.linspace(0.0, t_end, n, endpoint=False)
    y = amp_s * np.sin(omega * t) + amp_c * np.cos(omega * t)
    return t, y


# ---------------------------------------------------------------------------
# NAFF 探测
# ---------------------------------------------------------------------------


def test_naff_available_does_not_raise():
    """``naff_available`` 总是返回 ``bool``，不抛错。"""
    result = naff_available()
    assert isinstance(result, bool)


def test_extract_frequencies_falls_back_to_fft_when_naff_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    """``extract_frequencies(prefer="auto")`` 在 NAFF 不可用时降级到 FFT 并警告。"""
    monkeypatch.setattr(
        "e2m2e.algorithm.normal_form.fft._resolve_naff_binary",
        lambda: None,
    )
    monkeypatch.setattr(
        "e2m2e.algorithm.normal_form.fft.naff_available",
        lambda: False,
    )

    omega = 1.7
    t, y = _make_single_tone(omega, 0.4, 0.6)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        comps, backend = extract_frequencies(t, y, n_components=5)

    assert backend == "fft"
    assert isinstance(comps[0], FFTComponent)
    # 至少一条警告指出 NAFF 不可用
    assert any("NAFF" in str(w.message) for w in caught), [str(w.message) for w in caught]


def test_extract_frequencies_prefer_naff_falls_back_on_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
):
    """``prefer="naff"`` 但 NAFF 调用失败时也降级到 FFT。"""

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated NAFF crash")

    monkeypatch.setattr(
        "e2m2e.algorithm.normal_form.fft._resolve_naff_binary",
        lambda: "/bin/true",
    )
    monkeypatch.setattr(
        "e2m2e.algorithm.normal_form.fft.naff_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "e2m2e.algorithm.normal_form.fft.detect_naff",
        _boom,
    )

    omega = 1.7
    t, y = _make_single_tone(omega, 0.4, 0.6)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        comps, backend = extract_frequencies(t, y, n_components=5)

    assert backend == "fft"
    assert any("NAFF" in str(w.message) for w in caught)


def test_extract_frequencies_prefer_fft_skips_naff():
    """``prefer="fft"`` 时即使 NAFF 可用也走 FFT。"""
    omega = 1.7
    t, y = _make_single_tone(omega, 0.4, 0.6)
    comps, backend = extract_frequencies(t, y, n_components=3, prefer="fft")
    assert backend == "fft"


def test_extract_frequencies_rejects_bad_prefer():
    """``prefer`` 取值非法时抛 :class:`ValueError`。"""
    t, y = _make_single_tone(1.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="prefer"):
        extract_frequencies(t, y, prefer="nonsense")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# FFT 提取
# ---------------------------------------------------------------------------


def test_fft_extract_recovers_single_tone_frequency():
    """合成单频信号的 FFT 提取应给出与真值接近的频率。"""
    omega = 1.7
    t, y = _make_single_tone(omega, 0.4, 0.6, n=512, t_end=64.0)
    comps = fft_extract(t, y, n_components=5)
    # 直流分量排第一，剩下按振幅降序——主峰应非常接近真频率
    non_dc = [c for c in comps if abs(c.freq) > 1e-6]
    assert non_dc, "至少应有一个非零频率分量"
    top_amp, top_freq = max((c.amp, c.freq) for c in non_dc)
    # Hanning 窗 + 二次谱插值：典型精度 ~0.5% 频率分辨率
    assert top_freq == pytest.approx(omega, rel=5e-3)


def test_fft_extract_handles_short_series():
    """极短序列（< 4 点）应安全返回。"""
    t = np.array([0.0, 1.0])
    y = np.array([0.5, -0.5])
    assert fft_extract(t, y) == []
    t = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 0.0, -1.0])
    comps = fft_extract(t, y, n_components=2)
    assert all(isinstance(c, FFTComponent) for c in comps)


def test_fft_extract_rejects_shape_mismatch():
    """``times`` 与 ``data`` 形状不一致时抛 :class:`ValueError`。"""
    t = np.linspace(0.0, 1.0, 10)
    y = np.zeros(11)
    with pytest.raises(ValueError, match="形状"):
        fft_extract(t, y)


def test_fft_extract_rejects_non_monotonic_time():
    """时间序列非单调递增时抛 :class:`ValueError`。"""
    t = np.array([0.0, 0.5, 0.4, 0.9])
    y = np.zeros(4)
    with pytest.raises(ValueError, match="单调递增"):
        fft_extract(t, y)


# ---------------------------------------------------------------------------
# 最小二乘 sin/cos 拟合
# ---------------------------------------------------------------------------


def test_least_squares_fit_recovers_amp_split():
    """对已知 ``s, c`` 的信号，最小二乘应复原 ``amp_s, amp_c``。"""
    omega = 2.4
    amp_s, amp_c = 0.31, -0.47
    t, y = _make_single_tone(omega, amp_s, amp_c, n=512, t_end=64.0)

    comps = least_squares_sin_cos_fit(t, y, [omega])
    assert len(comps) == 1
    c = comps[0]
    assert c.freq == pytest.approx(omega, abs=1e-12)
    assert c.amp_s == pytest.approx(amp_s, abs=1e-2)
    assert c.amp_c == pytest.approx(amp_c, abs=1e-2)


def test_least_squares_fit_includes_dc_component():
    """频率集合含 0 时应给出直流分量。"""
    t = np.linspace(0.0, 4.0, 64)
    y = np.full_like(t, 1.23)
    comps = least_squares_sin_cos_fit(t, y, [0.0])
    assert comps[0].freq == 0.0
    assert comps[0].amp_c == pytest.approx(1.23, abs=1e-12)


def test_least_squares_fit_rejects_shape_mismatch():
    """形状不一致时抛 :class:`ValueError`。"""
    t = np.zeros(8)
    y = np.zeros(7)
    with pytest.raises(ValueError, match="形状"):
        least_squares_sin_cos_fit(t, y, [1.0])


# ---------------------------------------------------------------------------
# 信号重构
# ---------------------------------------------------------------------------


def test_reconstruct_signal_recovers_synthetic_combo():
    """``reconstruct_signal`` 应回放 ``Σ s·sin + c·cos``。"""
    omega1, omega2 = 1.1, 2.3
    t = np.linspace(0.0, 6.0, 200)

    components = [
        FFTComponent(freq=omega1, amp_s=0.3, amp_c=0.4),
        FFTComponent(freq=omega2, amp_s=-0.2, amp_c=0.5),
    ]

    expected = (
        0.3 * np.sin(omega1 * t)
        + 0.4 * np.cos(omega1 * t)
        + (-0.2) * np.sin(omega2 * t)
        + 0.5 * np.cos(omega2 * t)
    )
    np.testing.assert_allclose(reconstruct_signal(t, components), expected, atol=1e-12)


def test_reconstruct_signal_empty_returns_zero():
    """空频域分量应返回零数组。"""
    t = np.array([0.0, 1.0, 2.0])
    out = reconstruct_signal(t, [])
    np.testing.assert_array_equal(out, np.zeros_like(t))


def test_reconstruct_derivative_matches_analytical():
    """``reconstruct_derivative`` 应解析地对信号求导。"""
    omega = 1.5
    t = np.linspace(0.0, 4.0, 100)
    components = [
        FFTComponent(freq=omega, amp_s=0.3, amp_c=0.4),
    ]
    expected = (
        -0.3 * omega * np.sin(omega * t)
        - 0.4 * omega * np.sin(omega * t) * 0.0
        + (0.3 * omega * np.cos(omega * t) - 0.4 * omega * np.sin(omega * t))
    )
    # 重写 expected = s·ω·cos − c·ω·sin
    s, c = 0.3, 0.4
    expected = s * omega * np.cos(omega * t) - c * omega * np.sin(omega * t)
    np.testing.assert_allclose(reconstruct_derivative(t, components), expected, atol=1e-12)


# ---------------------------------------------------------------------------
# 频率匹配
# ---------------------------------------------------------------------------


def test_fft_extract_detects_suppressed_center_manifold_frequency():
    """验收标准 #3 的检测逻辑：强受迫频率 + 弱中心流形频率时，
    FFT 能定位到中心流形频率且其幅值远低于受迫主峰。

    合成信号 ``cos(ω_forced·t) + ε·cos(ω_cm·t)``，其中 ``ω_cm`` 取
    L1 中心流形频率 ν₁≈2.3377，受迫分量幅值是中心流形的 100 倍。
    FFT 提取后，离 ν₁ 最近的分量幅值应低于主峰的 5%（即中心流形
    频率被压制）。该测试覆盖"幅值低于阈值"的判定逻辑本身；端到端
    的动力学替代轨道 FFT 检查（依赖 SPICE + 完整窗口）见
    ``test_dynamical_substitution``。
    """
    omega_forced = 0.9  # 受迫基频量级
    omega_cm = 2.33774371420711  # L1 中心流形频率 ν₁
    amp_forced, amp_cm = 1.0, 0.01  # 中心流形被压制 100 倍

    t = np.linspace(0.0, 200.0, 4000, endpoint=False)
    y = amp_forced * np.cos(omega_forced * t) + amp_cm * np.cos(omega_cm * t)

    comps = fft_extract(t, y, n_components=8)
    non_dc = [c for c in comps if abs(c.freq) > 1e-6]
    assert non_dc

    max_amp = max(c.amp for c in non_dc)
    nearest_cm = min(non_dc, key=lambda c: abs(c.freq - omega_cm))
    # 检测到的中心流形频率应接近真值
    assert nearest_cm.freq == pytest.approx(omega_cm, abs=0.05)
    # 其幅值应远低于受迫主峰（中心流形被压制）
    assert nearest_cm.amp < 0.05 * max_amp


def test_frequency_match_assigns_nearest_basis():
    """``frequency_match`` 把检测频率匹配到基频表最近邻。"""
    basis_freqs = np.array([0.991548, 0.074801, 0.925199, 1.004022])
    basis_coefs = np.array(
        [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0]],
        dtype=int,
    )
    components = [
        FFTComponent(freq=0.99, amp_s=0.1, amp_c=0.0, amp=0.1),
        FFTComponent(freq=0.075, amp_s=0.0, amp_c=0.2, amp=0.2),
        FFTComponent(freq=0.0, amp_s=0.0, amp_c=0.05, amp=0.05),
    ]
    matched = frequency_match(components, basis_freqs, basis_coefs)
    matched_non_dc = [c for c in matched if c.match_freq != 0.0]
    # 两个非零分量应分别匹配到 0.991548 与 0.074801
    matched_freqs = {round(c.match_freq, 6) for c in matched_non_dc}
    assert 0.991548 in matched_freqs
    assert 0.074801 in matched_freqs


def test_frequency_match_rejects_empty_basis():
    """空基频表必须抛 :class:`ValueError`。"""
    components = [FFTComponent(freq=1.0, amp=1.0)]
    with pytest.raises(ValueError, match="basis_freqs"):
        frequency_match(components, [])


def test_frequency_match_handles_dc_only():
    """仅含直流分量时也返回占位匹配项。"""
    components = [FFTComponent(freq=0.0, amp_s=0.0, amp_c=1.0, amp=1.0)]
    matched = frequency_match(components, [0.0], np.zeros((1, 6), dtype=int))
    assert matched[0].match_freq == 0.0
    assert matched[0].coef == (0, 0, 0, 0, 0, 0)


def test_frequency_match_rejects_coef_row_mismatch():
    """``basis_coefs`` 行数与 ``basis_freqs`` 不一致时报错。"""
    components = [FFTComponent(freq=1.0, amp=1.0)]
    with pytest.raises(ValueError, match="basis_coefs"):
        frequency_match(components, [1.0, 2.0], np.zeros((1, 6), dtype=int))
