"""NormalFormResult 序列化辅助（save/load 内部实现）。

把 ``NormalFormResult`` 及其四个子结果扁平化为 ``dict[str, ndarray | str]``,
供 ``np.savez`` 写入单个 ``.npz`` 文件；反向从 ``.npz`` 重建等价对象。

键命名约定：

- ``_ctx_*`` —— NormalFormContext 标量/数组
- ``_ds_*``  —— DynamicalSubstituteResult
- ``_qf_*``  —— QuasiFloquetResult
- ``_cm_*``  —— CenterManifoldResult
- ``_nfr_*`` —— NormalFormResult 顶层字段

tuple 键编码：``p0_p1_p2_p3_p4_p5``（下划线分隔非负整数）。
W_series 三层嵌套键编码：``w__{step}__o{order}__p0_p1_p2_p3_p4_p5``。
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .types import NormalFormResult


# ---------------------------------------------------------------------------
# tuple 键编解码
# ---------------------------------------------------------------------------


def _encode_pow(pow_tuple: tuple[int, ...]) -> str:
    """tuple 键 → 字符串 ``p0_p1_p2_p3_p4_p5``。"""
    return "p" + "_".join(str(int(x)) for x in pow_tuple)


def _decode_pow(s: str) -> tuple[int, ...]:
    """字符串 ``p0_p1_p2_p3_p4_p5`` → tuple 键。"""
    return tuple(int(x) for x in s[1:].split("_"))


def _encode_w_series_key(step: str, order: int, pow_tuple: tuple[int, ...]) -> str:
    """W_series 三层键 → 扁平键 ``w__{step}__o{order}__p...``。"""
    return f"w__{step}__o{order}__{_encode_pow(pow_tuple)}"


_W_KEY_RE = re.compile(r"^w__(\w+)__o(\d+)__(p[\d_]+)$")


def _decode_w_series_key(key: str) -> tuple[str, int, tuple[int, ...]]:
    """扁平键 → (step, order, pow_tuple)。"""
    m = _W_KEY_RE.match(key)
    if m is None:
        raise ValueError(f"无法解析 W_series 键：{key!r}")
    return m.group(1), int(m.group(2)), _decode_pow(m.group(3))


# ---------------------------------------------------------------------------
# Context 序列化
# ---------------------------------------------------------------------------


def _context_to_dict(ctx: Any) -> dict[str, Any]:
    """NormalFormContext → 扁平字典。"""
    d: dict[str, Any] = {}
    d["_ctx_system"] = ctx.system.__class__.__name__
    d["_ctx_point"] = ctx.libration_point.name
    d["_ctx_epoch"] = np.array(ctx.epoch)
    d["_ctx_order"] = np.array(ctx.order)
    d["_ctx_LU"] = np.array(ctx.LU)
    d["_ctx_TU"] = np.array(ctx.TU)
    d["_ctx_mu"] = np.array(ctx.mu)
    d["_ctx_mu_e"] = np.array(ctx.mu_e)
    d["_ctx_mu_m"] = np.array(ctx.mu_m)
    d["_ctx_mu_s"] = np.array(ctx.mu_s)
    d["_ctx_libration_position"] = np.asarray(ctx.libration_position, dtype=float)
    d["_ctx_base_frequencies"] = np.asarray(ctx.base_frequencies, dtype=float)
    d["_ctx_central_nu1"] = np.array(ctx.central_frequencies[0])
    d["_ctx_central_nu2"] = np.array(ctx.central_frequencies[1])
    d["_ctx_characteristic_exp"] = np.array(ctx.characteristic_exponent)
    if ctx.gamma is not None:
        d["_ctx_gamma"] = np.array(ctx.gamma)
    return d


def _context_from_dict(d: dict[str, Any]) -> Any:
    """扁平字典 → NormalFormContext。"""
    from e2m2e.core import CR3BP_System, LibrationPoint

    from .context import NormalFormContext

    point_name = str(d["_ctx_point"])
    point = LibrationPoint[point_name]
    epoch = float(d["_ctx_epoch"])
    order = int(d["_ctx_order"])

    # 重建 system（CR3BP_System 用 mu 构造）
    mu = float(d["_ctx_mu"])
    system_name = str(d["_ctx_system"])
    if system_name == "CR3BP_System":
        system = CR3BP_System(mu=mu, primary="Earth", secondary="Moon")
    else:
        raise ValueError(f"不支持的 system 类型：{system_name}")

    ctx = NormalFormContext(
        system=system,
        libration_point=point,
        epoch=epoch,
        order=order,
        LU=float(d["_ctx_LU"]),
        TU=float(d["_ctx_TU"]),
        mu=mu,
        mu_e=float(d["_ctx_mu_e"]),
        mu_m=float(d["_ctx_mu_m"]),
        mu_s=float(d["_ctx_mu_s"]),
    )
    return ctx


# ---------------------------------------------------------------------------
# DynamicalSubstituteResult 序列化
# ---------------------------------------------------------------------------


def _ds_to_dict(ds: Any) -> dict[str, Any]:
    d: dict[str, Any] = {}
    d["_ds_order"] = np.array(ds.order)
    d["_ds_tlist"] = np.asarray(ds.tlist, dtype=float)
    d["_ds_Xlist"] = np.asarray(ds.Xlist, dtype=float)
    for pow_tuple, arr in ds.W_poly.items():
        d[f"_ds_W_poly_{_encode_pow(pow_tuple)}"] = np.asarray(arr, dtype=float)
    for pow_tuple, arr in ds.Wdot_poly.items():
        d[f"_ds_Wdot_poly_{_encode_pow(pow_tuple)}"] = np.asarray(arr, dtype=float)
    d["_ds_backend"] = ds.backend
    d["_ds_spice_available"] = np.array(int(ds.spice_available))
    # fft_components
    for direction, comps in ds.fft_components.items():
        n = len(comps)
        d[f"_ds_fft_{direction}_n"] = np.array(n)
        if n > 0:
            d[f"_ds_fft_{direction}_freq"] = np.array([c.freq for c in comps])
            d[f"_ds_fft_{direction}_amp_s"] = np.array([c.amp_s for c in comps])
            d[f"_ds_fft_{direction}_amp_c"] = np.array([c.amp_c for c in comps])
            d[f"_ds_fft_{direction}_amp"] = np.array([c.amp for c in comps])
            d[f"_ds_fft_{direction}_match_freq"] = np.array([c.match_freq for c in comps])
            d[f"_ds_fft_{direction}_err"] = np.array([c.err for c in comps])
            # coef 是 tuple[int, ...]；存为 (n, 6) int 数组
            coef_arr = np.array([list(c.coef) for c in comps], dtype=int)
            d[f"_ds_fft_{direction}_coef"] = coef_arr
    # shooting_result
    if ds.shooting_result is not None:
        sr = ds.shooting_result
        d["_ds_sh_t_Q"] = np.asarray(sr.t_Q, dtype=float)
        d["_ds_sh_X_Q"] = np.asarray(sr.X_Q, dtype=float)
        d["_ds_sh_max_residual"] = np.array(sr.max_residual)
        d["_ds_sh_mean_residual"] = np.array(sr.mean_residual)
        d["_ds_sh_iterations"] = np.array(sr.iterations)
        d["_ds_sh_converged"] = np.array(int(sr.converged))
        d["_ds_sh_residual_history"] = np.array(sr.residual_history, dtype=float)
    return d


def _ds_from_dict(d: dict[str, Any], ctx: Any) -> Any:
    from .dynamical_substitution import DynamicalSubstituteResult
    from .fft import FFTComponent
    from .multiple_shooting import MultipleShootingResult

    order = int(d["_ds_order"])
    tlist = d["_ds_tlist"]
    Xlist = d["_ds_Xlist"]

    W_poly = {}
    Wdot_poly = {}
    for key in d:
        if key.startswith("_ds_W_poly_p"):
            pow_tuple = _decode_pow(key.removeprefix("_ds_W_poly_"))
            W_poly[pow_tuple] = d[key]
        elif key.startswith("_ds_Wdot_poly_p"):
            pow_tuple = _decode_pow(key.removeprefix("_ds_Wdot_poly_"))
            Wdot_poly[pow_tuple] = d[key]

    backend = str(d["_ds_backend"])
    spice_available = bool(int(d["_ds_spice_available"]))

    # fft_components
    fft_components: dict[str, list[FFTComponent]] = {}
    for direction in ("x", "y", "z"):
        n_key = f"_ds_fft_{direction}_n"
        if n_key not in d:
            continue
        n = int(d[n_key])
        if n == 0:
            fft_components[direction] = []
            continue
        freqs = d[f"_ds_fft_{direction}_freq"]
        amp_ss = d[f"_ds_fft_{direction}_amp_s"]
        amp_cs = d[f"_ds_fft_{direction}_amp_c"]
        amps = d[f"_ds_fft_{direction}_amp"]
        match_freqs = d[f"_ds_fft_{direction}_match_freq"]
        errs = d[f"_ds_fft_{direction}_err"]
        coefs = d[f"_ds_fft_{direction}_coef"]
        comps = []
        for i in range(n):
            comps.append(
                FFTComponent(
                    freq=float(freqs[i]),
                    amp_s=float(amp_ss[i]),
                    amp_c=float(amp_cs[i]),
                    amp=float(amps[i]),
                    match_freq=float(match_freqs[i]),
                    coef=tuple(int(c) for c in coefs[i]),
                    err=float(errs[i]),
                )
            )
        fft_components[direction] = comps

    # shooting_result
    shooting_result = None
    if "_ds_sh_t_Q" in d:
        shooting_result = MultipleShootingResult(
            t_Q=d["_ds_sh_t_Q"],
            X_Q=d["_ds_sh_X_Q"],
            max_residual=float(d["_ds_sh_max_residual"]),
            mean_residual=float(d["_ds_sh_mean_residual"]),
            iterations=int(d["_ds_sh_iterations"]),
            converged=bool(int(d["_ds_sh_converged"])),
            residual_history=tuple(float(x) for x in d["_ds_sh_residual_history"]),
        )

    return DynamicalSubstituteResult(
        context=ctx,
        order=order,
        substitute_orbit=Xlist,  # 反序列化后用 Xlist 代替 Orbit
        tlist=tlist,
        Xlist=Xlist,
        W_poly=W_poly,
        Wdot_poly=Wdot_poly,
        Kamiltonian=None,
        fft_components=fft_components,
        shooting_result=shooting_result,
        backend=backend,
        spice_available=spice_available,
    )


# ---------------------------------------------------------------------------
# QuasiFloquetResult 序列化
# ---------------------------------------------------------------------------


def _qf_to_dict(qf: Any) -> dict[str, Any]:
    d: dict[str, Any] = {}
    d["_qf_order"] = np.array(qf.order)
    d["_qf_tlist"] = np.asarray(qf.tlist, dtype=float)
    d["_qf_B_samples"] = np.asarray(qf.B_samples, dtype=float)
    d["_qf_D"] = np.asarray(qf.D, dtype=float)
    d["_qf_method"] = qf.method
    if qf.M_samples is not None:
        d["_qf_M_samples"] = np.asarray(qf.M_samples, dtype=float)
    return d


def _qf_from_dict(d: dict[str, Any], ctx: Any) -> Any:
    from .quasi_floquet import QuasiFloquetResult

    return QuasiFloquetResult(
        context=ctx,
        order=int(d["_qf_order"]),
        tlist=d["_qf_tlist"],
        B_samples=d["_qf_B_samples"],
        D=d["_qf_D"],
        method=str(d["_qf_method"]),
        M_samples=d.get("_qf_M_samples"),
    )


# ---------------------------------------------------------------------------
# CenterManifoldResult 序列化
# ---------------------------------------------------------------------------


def _cm_to_dict(cm: Any) -> dict[str, Any]:
    d: dict[str, Any] = {}
    d["_cm_order"] = np.array(cm.order)
    d["_cm_steps"] = "|".join(cm.steps_performed)
    # hamiltonian_terms
    for pow_tuple, arr in cm.hamiltonian_terms.items():
        d[f"_cm_ht_{_encode_pow(pow_tuple)}"] = np.asarray(arr, dtype=float)
    # W_series 三层嵌套
    for step, step_data in cm.W_series.items():
        for order, poly in step_data.items():
            if not poly:
                # 空多项式也需标记，否则反序列化时丢失该 (step, order) 条目
                d[f"_cm_{_encode_w_series_key(step, order, (0, 0, 0, 0, 0, 0))}_empty"] = np.array(
                    1
                )
            for pow_tuple, coef_arr in poly.items():
                key = f"_cm_{_encode_w_series_key(step, order, pow_tuple)}"
                d[key] = np.asarray(coef_arr, dtype=complex)
    return d


def _cm_from_dict(d: dict[str, Any], ctx: Any) -> Any:
    from .center_manifold import CenterManifoldResult

    order = int(d["_cm_order"])
    steps = tuple(d["_cm_steps"].split("|"))

    # hamiltonian_terms
    hamiltonian_terms: dict[tuple[int, ...], Any] = {}
    for key in d:
        if key.startswith("_cm_ht_p"):
            pow_tuple = _decode_pow(key.removeprefix("_cm_ht_"))
            hamiltonian_terms[pow_tuple] = d[key]

    # W_series
    W_series: dict[str, dict[int, dict[tuple[int, ...], Any]]] = {}
    for key in d:
        if key.startswith("_cm_w__"):
            w_key = key.removeprefix("_cm_")
            if w_key.endswith("_empty"):
                # 空多项式标记
                w_key = w_key.removesuffix("_empty")
                step, o, _ = _decode_w_series_key(w_key)
                W_series.setdefault(step, {}).setdefault(o, {})
            else:
                step, o, pow_tuple = _decode_w_series_key(w_key)
                W_series.setdefault(step, {}).setdefault(o, {})[pow_tuple] = d[key]

    return CenterManifoldResult(
        context=ctx,
        order=order,
        W_series=W_series,
        hamiltonian_terms=hamiltonian_terms,
        steps_performed=steps,
    )


# ---------------------------------------------------------------------------
# 顶层 NormalFormResult 序列化
# ---------------------------------------------------------------------------


def _encode_str(s: str) -> np.ndarray:
    """str → bytes ndarray（np.savez 兼容）。"""
    return np.frombuffer(s.encode("utf-8"), dtype=np.uint8)


def _decode_str(arr: np.ndarray) -> str:
    """bytes ndarray → str。"""
    return bytes(arr.astype(np.uint8)).decode("utf-8")


def result_to_npz_dict(result: NormalFormResult) -> dict[str, Any]:
    """NormalFormResult → 扁平字典（供 np.savez 使用）。"""
    d: dict[str, Any] = {}

    # 标记格式版本
    d["_fmt_version"] = np.array(1)

    # context
    d.update(_context_to_dict(result.context))

    # 顶层字段
    d["_nfr_order"] = np.array(result.order)
    d["_nfr_residual"] = np.array(result.substitute_residual)
    d["_nfr_success"] = np.array(int(result.success))
    d["_nfr_message"] = _encode_str(result.message)

    # 子结果
    if result.ds_result is not None:
        d["_has_ds"] = np.array(1)
        d.update(_ds_to_dict(result.ds_result))
    if result.qf_result is not None:
        d["_has_qf"] = np.array(1)
        d.update(_qf_to_dict(result.qf_result))
    if result.cm_result is not None:
        d["_has_cm"] = np.array(1)
        d.update(_cm_to_dict(result.cm_result))

    # 把所有 str 值编码为 bytes ndarray
    for k, v in list(d.items()):
        if isinstance(v, str):
            d[k] = _encode_str(v)

    return d


def result_from_npz_dict(d: dict[str, Any]) -> NormalFormResult:
    """扁平字典 → NormalFormResult。"""
    from .catalog import LibrationCatalogData, LibrationCatalogTransformer
    from .types import NormalFormResult

    # 把 bytes ndarray 解码回 str
    _STR_KEYS = {
        "_ctx_system",
        "_ctx_point",
        "_ds_backend",
        "_qf_method",
        "_cm_steps",
        "_nfr_message",
    }
    for k in _STR_KEYS:
        if k in d:
            d[k] = _decode_str(d[k])

    ctx = _context_from_dict(d)

    order = int(d["_nfr_order"])
    residual = float(d["_nfr_residual"])
    success = bool(int(d["_nfr_success"]))
    message = str(d["_nfr_message"])

    # 子结果
    ds_result = None
    if "_has_ds" in d:
        ds_result = _ds_from_dict(d, ctx)

    qf_result = None
    if "_has_qf" in d:
        qf_result = _qf_from_dict(d, ctx)

    cm_result = None
    if "_has_cm" in d:
        cm_result = _cm_from_dict(d, ctx)

    # 重建 catalog_transformer
    catalog_transformer = None
    if ds_result is not None and qf_result is not None and cm_result is not None:
        catalog_data = LibrationCatalogData(
            context=ctx,
            ds_result=ds_result,
            qf_result=qf_result,
            cm_result=cm_result,
        )
        catalog_transformer = LibrationCatalogTransformer(data=catalog_data)

    return NormalFormResult(
        context=ctx,
        order=order,
        substitute_residual=residual,
        success=success,
        message=message,
        ds_result=ds_result,
        qf_result=qf_result,
        cm_result=cm_result,
        catalog_transformer=catalog_transformer,
    )
