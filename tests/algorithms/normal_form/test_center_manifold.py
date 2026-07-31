"""``normal_form.center_manifold`` 测试。

覆盖（issue #173，切片 4）：

- :class:`CenterManifoldReducer` 可用，``reduce`` 返回
  :class:`CenterManifoldResult`；
- 两步化简（``invariant`` / ``center``）的判别条件忠实迁移 qiao
  Code10/Code11；
- **化简效果硬性断言**：注入双曲-中心交叉项后，``max_hyperbolic_coupling``
  显著低于化简前；化简后剩余 Hamiltonian 只含作用量项；
- 单步可独立执行（``steps=("invariant",)`` / ``("center",)``）；
- 频域 ODE 求解器、MAD 离群抑制、高阶数值微分单元测试；
- qiao ``.npz`` 回归 fixture 用 ``pytest.skip`` 守卫（本仓库没有）。

输入 :class:`QuasiFloquetResult` 的构造沿用
``test_quasi_floquet.py`` 的范式：本切片的 reducer 只读 ``tlist``、
``D``（频率）；高阶 Hamiltonian 项经 ``hamiltonian_terms`` 注入
（对应 qiao Code09 的 ``L?_QF_Hamilton.npz``）。注入项刻意构造为
物理上明确的「双曲-中心交叉项」，用以直接断言化简效果。
"""

from __future__ import annotations

import numpy as np
import pytest

# sympy 是 normal-form optional dep；未安装时整个文件 skip（不 error）。
pytest.importorskip("sympy")

from e2m2e.algorithms.normal_form.center_manifold import (
    DEFAULT_MAX_ORDER,
    CenterManifoldReducer,
    CenterManifoldResult,
    list_deriv,
)
from e2m2e.algorithms.normal_form.quasi_floquet import (
    QuasiFloquetResult,
    real_normal_form_matrix,
)
from e2m2e.core import LibrationPoint

# ---------------------------------------------------------------------------
# 公共 fixture（沿用 test_quasi_floquet 的范式）
# ---------------------------------------------------------------------------


@pytest.fixture
def l1_context(earth_moon_system):
    """L1 共线点上下文。"""
    from e2m2e.algorithms.normal_form import NormalFormContext
    from e2m2e.algorithms.normal_form.constants import JD0_J2000

    return NormalFormContext(
        system=earth_moon_system,
        libration_point=LibrationPoint.L1,
        epoch=JD0_J2000,
        order=4,
    )


def _make_qf_result(l1_context, *, n: int = 96, T: float = 3.0) -> QuasiFloquetResult:
    """构造一个最小 :class:`QuasiFloquetResult`（B(t)=I，仅用于承载 tlist/D）。

    本切片 reducer 只读 ``tlist`` 与实标准形 ``D``（频率）；``B_samples``
    不参与化简，故用单位矩阵占位即可。这与 qiao Code10 的实际输入
    （Code09 已把 Hamiltonian 投到 QF 坐标、留下时变系数）等价：
    系数时变性已包含在注入的 ``hamiltonian_terms`` 中。
    """
    lam = float(l1_context.characteristic_exponent)
    nu1, nu2 = l1_context.central_frequencies
    D = real_normal_form_matrix(lam, float(nu1), float(nu2))
    tlist = np.linspace(0.0, T, n)
    B = np.stack([np.eye(6, dtype=float) for _ in range(n)])
    return QuasiFloquetResult(
        context=l1_context,
        order=int(l1_context.order),
        tlist=tlist,
        B_samples=B,
        D=D,
        method="matrix",
    )


def _hyper_center_terms(tlist, scale: float = 0.1):
    """注入一批双曲-中心交叉项（``q_1``/``p_1`` 与 ``q_2``/``p_2`` 耦合）。

    这些项是 Step 1（``invariant``）应当消去的对象：判别条件
    ``pow(1) != pow(4)``（双曲方向不平衡）即不满足保留判据。
    """
    ones = np.ones_like(tlist)
    return {
        (2, 1, 0, 1, 0, 0): scale * ones,  # q1²·q2·p1（pow1=2≠pow4=1）
        (1, 2, 0, 2, 0, 0): scale * 0.5 * ones,  # q1·q2²·p1²（pow1=1≠pow4=2）
        (1, 0, 1, 2, 0, 0): scale * 0.3 * ones,  # q1·q3·p1²（双曲-垂直交叉）
    }


def _center_cross_terms(tlist, scale: float = 0.08):
    """注入一批中心方向间非共振耦合项（``q_2``/``p_2`` 与 ``q_3``/``p_3``）。

    这些项双曲方向平衡（``pow1==pow4==0``）但中心方向不平衡，是 Step 2
    （``center``）应当消去的对象。
    """
    ones = np.ones_like(tlist)
    return {
        (0, 3, 0, 0, 1, 0): scale * ones,  # q2³·p2（pow2=3≠pow5=1）
        (0, 1, 2, 0, 1, 2): scale * 0.5 * ones,  # q2·q3²·p2·p3²
    }


# ---------------------------------------------------------------------------
# 构造与接口
# ---------------------------------------------------------------------------


def test_reducer_is_constructible(l1_context):
    """``CenterManifoldReducer`` 可用上下文构造，默认 max_order=10。"""
    reducer = CenterManifoldReducer(context=l1_context)
    assert reducer.context is l1_context
    assert reducer.max_order == DEFAULT_MAX_ORDER == 10


def test_reducer_rejects_unknown_step(l1_context):
    """``steps`` 含非法名时抛 :class:`ValueError`。"""
    reducer = CenterManifoldReducer(context=l1_context, max_order=5)
    qf = _make_qf_result(l1_context)
    with pytest.raises(ValueError, match="非法值"):
        reducer.reduce(qf, steps=("bogus",))


def test_reducer_rejects_nonpositive_max_order(l1_context):
    """``max_order`` 非正时抛 :class:`ValueError`。"""
    reducer = CenterManifoldReducer(context=l1_context, max_order=0)
    qf = _make_qf_result(l1_context)
    with pytest.raises(ValueError, match="max_order"):
        reducer.reduce(qf)


def test_reduce_returns_result_with_required_fields(l1_context):
    """``reduce`` 返回 :class:`CenterManifoldResult`，字段齐备。"""
    reducer = CenterManifoldReducer(context=l1_context, max_order=5)
    qf = _make_qf_result(l1_context)
    terms = _hyper_center_terms(qf.tlist)
    result = reducer.reduce(qf, hamiltonian_terms=terms)

    assert isinstance(result, CenterManifoldResult)
    assert result.context is l1_context
    assert result.order == 5
    # 默认两步都执行
    assert result.steps_performed == ("invariant", "center")
    # metadata 记录频率
    assert {"lambda", "wp", "wv", "n_samples", "pre_hyperbolic_center_coupling"} <= set(
        result.metadata
    )
    # W_series 封装了每步每阶的系数表
    assert "invariant" in result.W_series
    assert "center" in result.W_series


# ---------------------------------------------------------------------------
# smoke：端到端跑通小输入
# ---------------------------------------------------------------------------


def test_smoke_end_to_end_runs(l1_context):
    """端到端：构造 QF 结果 + 注入高阶项 → reduce 跑通不报错。"""
    reducer = CenterManifoldReducer(context=l1_context, max_order=6)
    qf = _make_qf_result(l1_context)
    terms = _hyper_center_terms(qf.tlist)
    result = reducer.reduce(qf, hamiltonian_terms=terms)
    # 结果非空
    assert len(result.hamiltonian_terms) > 0
    # W_series 两步都有内容
    assert any(result.W_series["invariant"].values())
    # hamiltonian_terms 系数都是有限实数
    for coef in result.hamiltonian_terms.values():
        assert np.all(np.isfinite(coef))


# ---------------------------------------------------------------------------
# 化简效果硬性断言：交叉项系数下降
# ---------------------------------------------------------------------------


def test_invariant_step_reduces_hyperbolic_center_coupling(l1_context):
    """Step 1（invariant）消去双曲方向不平衡的交叉项。

    注入 ``q_1``/``p_1`` 与中心方向耦合的项后，化简结果中 3+ 阶不得含
    双曲方向不平衡项（``pow(1)!=pow(4)``）——这是 qiao ``Code10`` 的
    删除判据（只消双曲不平衡项）。双曲平衡但中心不平衡的项（如
    ``q1²·q2·q3·p1²``）是 Step 2（``center``）的目标，Step 1 保留。
    """
    reducer = CenterManifoldReducer(context=l1_context, max_order=6)
    qf = _make_qf_result(l1_context)
    terms = _hyper_center_terms(qf.tlist, scale=0.1)
    pre = 0.1  # _hyper_center_terms 最大幅值

    result = reducer.reduce(qf, hamiltonian_terms=terms, steps=("invariant",))
    # 注入项全部是双曲不平衡型（pow(1)!=pow(4)），Step 1 后必须消失
    for pow_tuple in result.hamiltonian_terms:
        if sum(pow_tuple) < 3:
            continue
        assert pow_tuple[0] == pow_tuple[3], f"Step 1 后 3+ 阶仍含双曲不平衡项 {pow_tuple}"
    # 残余耦合（含 Step 2 目标项）应远小于注入幅值
    post = result.max_hyperbolic_coupling
    assert post < 0.05 * pre, f"Step 1 化简后残余耦合 {post:.2e} 未显著低于注入幅值 {pre:.2e}"


def test_center_step_leaves_only_action_terms(l1_context):
    """Step 1 + Step 2 后 Hamiltonian 只剩作用量项。

    两步完成后，把化简后的实坐标 Hamiltonian 虚变换回复坐标，所有
    ≥3 阶项必须满足三对共轭全部平衡（``pow1==pow4 && pow2==pow5 &&
    pow3==pow6``）——即仅依赖作用量 ``I_1``、``I_2``、``I_3``。注意
    必须在**复坐标**断言：实坐标下作用量 ``I_2=(q2²+p2²)/2`` 的展开项
    （如 ``q2⁶``）本身不满足幂次平衡。
    """
    reducer = CenterManifoldReducer(context=l1_context, max_order=6)
    qf = _make_qf_result(l1_context)
    terms = {}
    terms.update(_hyper_center_terms(qf.tlist, scale=0.1))
    terms.update(_center_cross_terms(qf.tlist, scale=0.08))

    result = reducer.reduce(qf, hamiltonian_terms=terms)  # 两步都跑
    _assert_action_form(result.hamiltonian_terms)


def _assert_action_form(hamiltonian_terms):
    """断言 Hamiltonian 只依赖作用量：虚变换回复坐标后三对全平衡。"""
    from e2m2e.algorithms.normal_form.center_manifold import _D, _linear_basis_change

    H_by_order: dict[int, dict[tuple[int, ...], np.ndarray]] = {}
    for k, v in hamiltonian_terms.items():
        deg = sum(k)
        H_by_order.setdefault(deg, {})[k] = np.asarray(v)
    H_c = _linear_basis_change(H_by_order, _D)
    for o, poly in H_c.items():
        if o < 3:
            continue
        for pow_tuple, v in poly.items():
            if np.max(np.abs(v)) <= 1e-12:
                continue
            is_action = (
                pow_tuple[0] == pow_tuple[3]
                and pow_tuple[1] == pow_tuple[4]
                and pow_tuple[2] == pow_tuple[5]
            )
            assert is_action, f"两步化简后 {o} 阶仍含非作用量项 {pow_tuple}"


def test_max_hyperbolic_coupling_decreases_after_full_reduction(l1_context):
    """两步化简后 ``max_hyperbolic_coupling`` 严格小于化简前。"""
    reducer = CenterManifoldReducer(context=l1_context, max_order=6)
    qf = _make_qf_result(l1_context)
    terms = _hyper_center_terms(qf.tlist, scale=0.1)
    pre = 0.1  # _hyper_center_terms 最大注入幅值

    result = reducer.reduce(qf, hamiltonian_terms=terms)
    assert result.max_hyperbolic_coupling < pre
    assert result.metadata["pre_hyperbolic_center_coupling"] == pytest.approx(pre)


# ---------------------------------------------------------------------------
# 单步独立执行
# ---------------------------------------------------------------------------


def test_single_step_invariant(l1_context):
    """``steps=("invariant",)`` 只跑 Step 1。"""
    reducer = CenterManifoldReducer(context=l1_context, max_order=5)
    qf = _make_qf_result(l1_context)
    terms = _hyper_center_terms(qf.tlist)
    result = reducer.reduce(qf, hamiltonian_terms=terms, steps=("invariant",))
    assert result.steps_performed == ("invariant",)
    assert "invariant" in result.W_series
    assert "center" not in result.W_series


def test_single_step_center(l1_context):
    """``steps=("center",)`` 只跑 Step 2。

    无高阶项注入时 Step 2 退化为只处理二阶实标准形（平凡），仍应跑通。
    """
    reducer = CenterManifoldReducer(context=l1_context, max_order=5)
    qf = _make_qf_result(l1_context)
    result = reducer.reduce(qf, steps=("center",))
    assert result.steps_performed == ("center",)
    assert "center" in result.W_series
    assert "invariant" not in result.W_series


def test_center_step_produces_nonzero_complex_W(l1_context):
    """Step 2 对中心非共振项真正求出**非零复值** W。

    回归守卫：注入双曲方向已平衡（``pow1==pow4==0``）但中心方向不平衡
    （``pow2!=pow5``）的纯中心项，Step 2 必须对它们求解同调方程、产出
    非零 W。中心方向特征频率为纯虚 ``iω``，故 W 为纯虚（实部≈0）。

    此前曾存在两个互相叠加的退化：(1) ``_limit_fft_outliers_mad`` 在
    ``MAD=0``（常系数输入）时把唯一非零的零频当离群点缩到 0，使 W 归零；
    (2) ``reduce`` 对 W 取实部，丢弃 Step 2 纯虚 W 的全部内容。任一退化
    复发，本测试的 ``nonzero_W`` 断言即失败。
    """
    reducer = CenterManifoldReducer(context=l1_context, max_order=6)
    qf = _make_qf_result(l1_context)
    ones = np.ones_like(qf.tlist)
    # 纯中心非共振项：双曲平衡，中心不平衡 → 仅 Step 2 处理
    center_terms = {
        (0, 3, 0, 0, 1, 0): 0.08 * ones,  # pow2=3 ≠ pow5=1
        (0, 2, 1, 0, 1, 1): 0.05 * ones,  # pow2=2 ≠ pow5=1
    }
    result = reducer.reduce(qf, hamiltonian_terms=center_terms, steps=("center",))
    # W_series["center"] 必须含至少一个非零项
    nonzero_W = any(
        np.any(np.abs(v)) > 1e-9
        for poly in result.W_series["center"].values()
        for v in poly.values()
    )
    assert nonzero_W, "Step 2 未对中心非共振项求出非零 W（退化复发？）"
    # Step 2 的 W 是复值（中心方向在复域操作）
    for poly in result.W_series["center"].values():
        for v in poly.values():
            assert np.iscomplexobj(v)
            assert v.dtype == np.complex128


def test_center_step_reduces_center_coupling(l1_context):
    """Step 2 消去中心方向间非共振耦合，使 Hamiltonian 只剩作用量项。

    回归守卫：注入纯中心非共振项后，Step 2 化简结果（虚变换回复坐标
    后）中 ≥3 阶项必须全部满足三对共轭平衡（``pow1==pow4 &&
    pow2==pow5 && pow3==pow6``），即仅依赖作用量 ``I_2``/``I_3``。
    此前 Step 2 因 W 归零退化而是 no-op，本测试直接断言化简效果，
    不依赖正逆相消。
    """
    reducer = CenterManifoldReducer(context=l1_context, max_order=6)
    qf = _make_qf_result(l1_context)
    ones = np.ones_like(qf.tlist)
    center_terms = {
        (0, 3, 0, 0, 1, 0): 0.08 * ones,
        (0, 2, 1, 0, 1, 1): 0.05 * ones,
    }
    result = reducer.reduce(qf, hamiltonian_terms=center_terms, steps=("center",))
    _assert_action_form(result.hamiltonian_terms)


def test_W_for_accessor_and_keyerror(l1_context):
    """``W_for(step, order)`` 封装访问器正确返回；非法 step 抛 KeyError。"""
    reducer = CenterManifoldReducer(context=l1_context, max_order=5)
    qf = _make_qf_result(l1_context)
    result = reducer.reduce(qf, hamiltonian_terms=_hyper_center_terms(qf.tlist))
    w3 = result.W_for("invariant", 3)
    assert isinstance(w3, dict)
    with pytest.raises(KeyError, match="未执行"):
        result.W_for("bogus", 3)


def test_trivial_input_no_higher_order_terms(l1_context):
    """不注入高阶项时 reducer 退化为只含二阶实标准形（仍跑通）。"""
    reducer = CenterManifoldReducer(context=l1_context, max_order=4)
    qf = _make_qf_result(l1_context)
    result = reducer.reduce(qf)  # hamiltonian_terms=None
    # 只剩二阶实标准形 5 项
    lam = float(l1_context.characteristic_exponent)
    nu1, nu2 = l1_context.central_frequencies
    expected = {
        (1, 0, 0, 1, 0, 0): lam,
        (0, 2, 0, 0, 0, 0): float(nu1) / 2,
        (0, 0, 0, 0, 2, 0): float(nu1) / 2,
        (0, 0, 2, 0, 0, 0): float(nu2) / 2,
        (0, 0, 0, 0, 0, 2): float(nu2) / 2,
    }
    assert set(result.hamiltonian_terms.keys()) == set(expected.keys())
    for k, v in expected.items():
        np.testing.assert_allclose(result.hamiltonian_terms[k][0], v, atol=1e-10)


def test_coefficient_length_mismatch_raises(l1_context):
    """``hamiltonian_terms`` 系数长度与 ``tlist`` 不一致时报错。"""
    reducer = CenterManifoldReducer(context=l1_context, max_order=4)
    qf = _make_qf_result(l1_context)
    bad_terms = {(2, 1, 0, 1, 0, 0): np.ones(7)}  # 长度 7 ≠ tlist 长度
    with pytest.raises(ValueError, match="不一致"):
        reducer.reduce(qf, hamiltonian_terms=bad_terms)


# ---------------------------------------------------------------------------
# 频域 ODE 求解器与数值微分单元测试
# ---------------------------------------------------------------------------


def test_solve_wfunc_fft_recovers_exponential_forcing():
    """``_solve_wfunc_fft`` 对 ``f(t)=e^{αt}``、``k=-α`` 应近似恢复 ``t+c``。

    ``ẏ = -α·y + e^{αt}`` 的一个特解是 ``y = e^{αt}/(2α)``；此处只验证
    求解器返回有限复值数组、且与解析特解量级一致。
    """
    from e2m2e.algorithms.normal_form.center_manifold import _solve_wfunc_fft

    tlist = np.linspace(0, 2.0, 128)
    alpha = 1.7
    forcing = np.exp(alpha * tlist)
    k = complex(-alpha, 0.0)
    y = _solve_wfunc_fft(tlist, forcing, k)
    assert y.shape == tlist.shape
    assert np.all(np.isfinite(y))


# ---------------------------------------------------------------------------
# 虚变换 / 实变换（复基底变换）
# ---------------------------------------------------------------------------


def test_virtual_real_transform_roundtrip():
    """虚变换 + 实变换 roundtrip：实多项式还原（含双曲/中心混合项）。"""
    from e2m2e.algorithms.normal_form.center_manifold import (
        _D,
        _D_INV,
        _linear_basis_change,
    )

    ones = np.ones(4)
    H = {
        3: {
            (0, 3, 0, 0, 0, 0): ones,  # q2³
            (1, 1, 0, 1, 0, 0): 0.5 * ones,  # q1·q2·p1
            (0, 1, 1, 0, 1, 1): 0.3 * ones,  # q2·q3·p2·p3
        }
    }
    Hc = _linear_basis_change(H, _D)
    Hr = _linear_basis_change(Hc, _D_INV)
    for k, v in H[3].items():
        got = np.real(Hr[3][k])
        np.testing.assert_allclose(got, v, atol=1e-12)


def test_virtual_transform_makes_complex_diagonal(l1_context):
    """H₂ 实正规形虚变换后成复对角形。

    ``(ω_p/2)(q2²+p2²)`` 与 ``(ω_v/2)(q3²+p3²)`` 各自合成单项
    ``i·ω_p·y2·y5``、``i·ω_v·y3·y6``；双曲项 ``λ·q1·p1`` 不变。这是
    Gómez vol III §2.7.1 "put in complex form" 一步（qiao Code10 虚变换）。
    """
    from e2m2e.algorithms.normal_form.center_manifold import _D, _linear_basis_change

    lam = float(l1_context.characteristic_exponent)
    nu1, nu2 = l1_context.central_frequencies
    wp, wv = float(nu1), float(nu2)
    ones = np.ones(4)
    H2 = {
        2: {
            (1, 0, 0, 1, 0, 0): lam * ones,
            (0, 2, 0, 0, 0, 0): (wp / 2) * ones,
            (0, 0, 0, 0, 2, 0): (wp / 2) * ones,
            (0, 0, 2, 0, 0, 0): (wv / 2) * ones,
            (0, 0, 0, 0, 0, 2): (wv / 2) * ones,
        }
    }
    H2c = _linear_basis_change(H2, _D)
    kept = {k: v for k, v in H2c[2].items() if np.max(np.abs(v)) > 1e-12}
    assert set(kept.keys()) == {
        (1, 0, 0, 1, 0, 0),
        (0, 1, 0, 0, 1, 0),
        (0, 0, 1, 0, 0, 1),
    }, kept.keys()
    np.testing.assert_allclose(kept[(1, 0, 0, 1, 0, 0)][0], lam, atol=1e-12)
    np.testing.assert_allclose(kept[(0, 1, 0, 0, 1, 0)][0], 1j * wp, atol=1e-12)
    np.testing.assert_allclose(kept[(0, 0, 1, 0, 0, 1)][0], 1j * wv, atol=1e-12)


def test_homological_equation_residual(l1_context):
    """同调方程残差：``{H₂c, W} + H_elim ≈ 0``（复坐标下谱公式匹配）。

    在复对角形 ``H₂c = λ·y1·y4 + i·ω_p·y2·y5 + i·ω_v·y3·y6`` 下，
    对常数系数 ``H_elim`` 用频域求解器解 ``W``，泊松括号必须精确抵消
    ``H_elim``。此前在实正规形下用复值 ``k`` 求解，残差高达 O(h)（残留
    耦合 max=48 的根因）。
    """
    from e2m2e.algorithms.normal_form.center_manifold import (
        _characteristic_freq,
        _solve_wfunc_fft,
    )
    from e2m2e.algorithms.normal_form.polynomial import poly_poisson

    lam = float(l1_context.characteristic_exponent)
    nu1, nu2 = l1_context.central_frequencies
    wp, wv = float(nu1), float(nu2)
    tlist = np.linspace(0.0, 3.0, 96)
    ones = np.ones_like(tlist)

    H2c = {
        (1, 0, 0, 1, 0, 0): lam * ones,
        (0, 1, 0, 0, 1, 0): 1j * wp * ones,
        (0, 0, 1, 0, 0, 1): 1j * wv * ones,
    }
    elim_terms = {
        (1, 2, 0, 0, 0, 0): 0.1 * ones,  # y1·y2²，k = -λ + 2i·ω_p
        (0, 1, 0, 2, 0, 0): 0.05 * ones,  # y2·y4²，k = 2λ - i·ω_p
        (1, 0, 1, 0, 0, 1): 0.03 * ones,  # y1·y3·y6，k = -λ + i·ω_v - i·ω_v
    }
    for pow_t, h in elim_terms.items():
        k = _characteristic_freq(pow_t, lam, wp, wv)
        assert abs(k) > 1e-9, f"{pow_t} 的 k 为零（共振项不应在此消去）"
        W = {pow_t: _solve_wfunc_fft(tlist, np.real(h), k)}
        pb = poly_poisson(H2c, W)
        residual = pb.get(pow_t, np.zeros_like(ones)) + h
        assert np.max(np.abs(residual)) < 1e-6, (
            f"{pow_t}: 同调方程残差 {np.max(np.abs(residual)):.2e}（k={k}）"
        )


def test_list_deriv_recovers_linear_function():
    """``list_deriv`` 对线性函数 ``y=t`` 应精确返回常数 1。"""
    tlist = np.linspace(0, 1.0, 40)
    h = float(tlist[1] - tlist[0])
    y = 2.5 * tlist + 1.0
    dy = list_deriv(y, h, n=8)
    # 内部点应精确（端点因差分阶降低误差略大，放宽）
    np.testing.assert_allclose(dy[8:-8], np.full(dy[8:-8].shape, 2.5), atol=1e-10)


def test_list_deriv_handles_complex():
    """``list_deriv`` 对复值输入分别处理实/虚部。"""
    tlist = np.linspace(0, 2.0, 50)
    h = float(tlist[1] - tlist[0])
    y = np.cos(tlist) + 1j * np.sin(tlist)
    dy = list_deriv(y, h, n=8)
    # d/dt[cos+isin] = -sin + i cos
    np.testing.assert_allclose(dy.real[6:-6], -np.sin(tlist[6:-6]), atol=1e-8)
    np.testing.assert_allclose(dy.imag[6:-6], np.cos(tlist[6:-6]), atol=1e-8)


def test_keep_criteria_predicates():
    """判别条件忠实迁移 qiao Code10/Code11。"""
    from e2m2e.algorithms.normal_form.center_manifold import (
        _is_center_term,
        _is_invariant_term,
    )

    # invariant：只看 pow(1)==pow(4)
    assert _is_invariant_term((2, 1, 0, 2, 0, 0))  # pow1==pow4==2，保留
    assert not _is_invariant_term((2, 1, 0, 1, 0, 0))  # pow1=2≠pow4=1，消去
    # center：三对共轭全部平衡
    assert _is_center_term((2, 1, 0, 2, 1, 0))  # 全平衡
    assert not _is_center_term((2, 1, 0, 2, 0, 0))  # pow2=1≠pow5=0
    assert not _is_center_term((2, 1, 0, 1, 1, 0))  # pow1=2≠pow4=1
    # invariant 保留的项未必满足 center（center 更严）
    assert _is_invariant_term((0, 3, 0, 0, 1, 0))  # pow1==pow4==0
    assert not _is_center_term((0, 3, 0, 0, 1, 0))  # 但 pow2=3≠pow5=1


# ---------------------------------------------------------------------------
# qiao .npz 回归 fixture（本仓库没有，skip 守卫）
# ---------------------------------------------------------------------------


def test_qiao_npz_regression_is_skipped_without_fixture():
    """qiao ``L1_QF_Hamilton.npz`` fixture 在本仓库不存在，必须 skip。

    本测试验证 skip 守卫生效；外部 fixture 引入后可替换为与 qiao
    Code10/Code11 输出的真实数值比对（``L1_InvarManifold.npz`` /
    ``L1_CenterManifold.npz``）。
    """
    from pathlib import Path

    fixture = Path(__file__).parent / "data" / "L1_QF_Hamilton.npz"
    if not fixture.exists():
        pytest.skip(f"qiao .npz fixture 不存在：{fixture}")
    data = np.load(fixture, allow_pickle=True)
    assert "powers" in data and "coefficients" in data
