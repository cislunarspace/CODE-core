"""生成 CR3BP 基线轨道族数据集（ADR 0036；ADR 0047 出包）。

九族（HALO/NRHO/AXIAL/LISSAJOUS/DRO/DPO/SPO/HORSESHOE/LPO）× 地月
DE421，用各族规格的默认参数整族生成，产出 ADR 0031 格式的 catalog
记录（一族一条）写入 ``e2m2e/data/catalog_baseline/``：``tags=["baseline"]``、
``scalars.baseline_version`` 取包版本、record_id 确定性命名
（如 ``baseline-halo-l2``），供显式导入按 id 对位。

数据集不随 wheel 分发（ADR 0047）：本目录是仓库内回归夹具与 Release
资产源；分发时 zip 整目录上传 GitHub Release，调用方解压后经
``import_baseline(store, source_dir)`` 显式导入。

直接调算法层 ``design_*_family``（绕过 Facade，避免自动入库副作用）；
参数取 ``FamilyGenerationRequest`` 的族默认值，成员数上限取 100
（ADR 0036 实测口径）。DPO 无族层入口（ADR 0029 只登记八族），按
``design_dpo`` 振幅网格逐条组装族。

内置校验断言（``validate_baseline_record``）：status 收敛、成员数 > 0、
覆盖元数据（振幅区间、成员数、终止三元组、基线版本）完整；任一失败
报错退出、不写包（ADR 0036 决策 7）。

用法（uv run 会因缺 CSPICE_DIR 重构建失败，直接用 venv 解释器）::

    .venv/bin/python scripts/generate_catalog_baseline.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from e2m2e import __version__
from e2m2e.algorithm.family import (
    Cr3bpOrbitError,
    design_axial_family,
    design_dpo,
    design_dro_family,
    design_halo_family,
    design_horseshoe_family,
    design_lissajous_family,
    design_lpo_family,
    design_nrho_family,
    design_spo_family,
)
from e2m2e.algorithm.results import FamilyGenerationResult
from e2m2e.api.catalog_ingest import build_family_bundle
from e2m2e.api.models import FamilyGenerationRequest
from e2m2e.data.catalog import CatalogStore
from e2m2e.data.templates import ConvergenceState, FailureCause
from e2m2e.data.types.orbit import Orbit, OrbitFamily
from e2m2e.integrators import orbit_family_metric_py

#: 基线输出目录（仓库回归夹具与 Release 资产源；不随包分发，ADR 0047）
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "e2m2e" / "data" / "catalog_baseline"

#: 族成员数上限（ADR 0036 实测口径 n_orbits=100）
N_ORBITS = 100

#: DPO 无族层入口，按 design_dpo 振幅网格逐条生成（每条约 1 s）
DPO_N_ORBITS = 20

#: (orbit_type, libration_point)：共线族按 ADR 0036 决策 4 各生成 L1/L2
#: 一份；三角族取默认 L4；DRO/DPO 是月心族不绑点。
BASELINE_SPEC: tuple[tuple[str, int | None], ...] = (
    ("HALO", 1),
    ("HALO", 2),
    ("NRHO", 1),
    ("NRHO", 2),
    ("AXIAL", 1),
    ("AXIAL", 2),
    ("LISSAJOUS", 1),
    ("LISSAJOUS", 2),
    ("DRO", None),
    ("DPO", None),
    ("SPO", 4),
    ("LPO", 4),
    ("HORSESHOE", 4),
)


class _DpoFamilyRequest:
    """DPO 的族请求快照替身：FamilyGenerationRequest 不支持 DPO（无族层入口）。"""

    libration_point = None

    def __init__(self, min_amplitude_km: float, max_amplitude_km: float) -> None:
        self._dump = {
            "orbit_type": "DPO",
            "libration_point": None,
            "n_orbits": DPO_N_ORBITS,
            "min_amplitude_km": min_amplitude_km,
            "max_amplitude_km": max_amplitude_km,
        }

    def model_dump(self) -> dict[str, Any]:
        return dict(self._dump)


def _generate(request: FamilyGenerationRequest | _DpoFamilyRequest) -> tuple[Any, ...]:
    """按 orbit_type 分派到算法层族生成入口（镜像 Facade 分派，无入库）。

    返回 ``(family, status, cause, message, requested, generated)``。
    """
    if isinstance(request, _DpoFamilyRequest):
        return _generate_dpo_family(request)
    sel = request.orbit_type.upper()
    if sel == "HALO":
        result: Any = design_halo_family(
            request.libration_point, request.max_amplitude_km, n_orbits=request.n_orbits
        )
    elif sel == "NRHO":
        result = design_nrho_family(
            request.libration_point,
            request.north_south,
            request.perilune_height_max_km,
            n_orbits=request.n_orbits,
        )
    elif sel == "AXIAL":
        result = design_axial_family(
            request.libration_point,
            request.max_amplitude_km,
            n_orbits=request.n_orbits,
        )
    elif sel == "LISSAJOUS":
        result = design_lissajous_family(
            request.libration_point,
            request.amplitude_in_km,
            request.amplitude_out_km,
            request.phase_in,
            request.phase_out,
            n_orbits=request.n_orbits,
        )
    elif sel == "DRO":
        result = design_dro_family(
            request.min_amplitude_km, request.max_amplitude_km, n_orbits=request.n_orbits
        )
    else:
        entry = {
            "SPO": design_spo_family,
            "LPO": design_lpo_family,
            "HORSESHOE": design_horseshoe_family,
        }[sel]
        result = entry(
            request.libration_point,
            request.min_amplitude_km,
            request.max_amplitude_km,
            n_orbits=request.n_orbits,
        )
    if isinstance(result, FamilyGenerationResult):
        return (
            result.family,
            result.status,
            result.cause,
            result.message,
            result.requested_members,
            result.generated_members,
        )
    # design_halo_family 收敛时直接返回 OrbitFamily；三元组按 Facade 投影补齐
    return (
        result,
        ConvergenceState.CONVERGED,
        FailureCause.NONE,
        "轨道族生成完成",
        request.n_orbits,
        len(result),
    )


def _generate_dpo_family(request: _DpoFamilyRequest) -> tuple[Any, ...]:
    """DPO 族：design_dpo 振幅网格逐条生成（每条从种子独立行走）。

    振幅范围沿用 DRO 族的默认请求窗口；网格上不可达的振幅（族参数域
    边界、共振缝隙）跳过并计入 message，成功侧状态如实收敛。
    """
    amplitudes = np.geomspace(
        request._dump["min_amplitude_km"], request._dump["max_amplitude_km"], DPO_N_ORBITS
    )
    orbits: list[Orbit] = []
    for amplitude in amplitudes:
        try:
            orbit = design_dpo(float(amplitude))
        except Cr3bpOrbitError:
            continue
        orbit.family_type = "dpo"
        orbit.parameters = {"amplitude_target_km": float(amplitude)}
        orbits.append(orbit)
    if not orbits:
        raise SystemExit("DPO 族生成失败：振幅网格无一命中")
    family = OrbitFamily(orbits=orbits, family_type="dpo", system=orbits[0].system)
    family.metadata.update(
        periodicity="periodic",
        backend="python-amplitude-grid",
        amplitude_range_km=[
            request._dump["min_amplitude_km"],
            request._dump["max_amplitude_km"],
        ],
    )
    message = (
        f"DPO 振幅网格生成完成：{len(orbits)}/{DPO_N_ORBITS} 条命中"
        f"（跳过 {DPO_N_ORBITS - len(orbits)} 个不可达振幅）"
    )
    return (
        family,
        ConvergenceState.CONVERGED,
        FailureCause.NONE,
        message,
        DPO_N_ORBITS,
        len(orbits),
    )


def baseline_record_id(family_type: str, libration_point: int | None) -> str:
    """确定性 record_id：文件名与记录 id 对位，显式导入按 id 命中。"""
    suffix = "" if libration_point is None else f"-l{libration_point}"
    return f"baseline-{family_type}{suffix}"


def _member_amplitude_km(family_type: str, orbit: Orbit) -> float | None:
    """成员振幅（km），按各族规格的参数振幅定义取值。

    分类字段 ``classification.amplitude`` 是存储段（多数族仅 (1,6) 初态）
    上的几何主振幅，对初态单点恒为 0；覆盖元数据 ``scalars.
    amplitude_envelope_km`` 改用各族自己的参数振幅，如实反映实际覆盖。
    """
    char_length = orbit.system.characteristic_length if orbit.system is not None else None
    if family_type in ("halo", "nrho"):
        # Halo/NRHO 族参数振幅 = 参考穿越点 z0（DU）× 特征长度
        if char_length is None:
            return None
        return abs(float(orbit.states[0, 2])) * float(char_length)
    parameters = orbit.parameters or {}
    if family_type == "axial":
        value = parameters.get("amplitude_z_km")
    elif family_type == "lissajous":
        value = max(
            parameters.get("amplitude_in_km", 0.0) or 0.0,
            parameters.get("amplitude_out_km", 0.0) or 0.0,
        )
    elif family_type in ("spo", "lpo", "horseshoe"):
        value = parameters.get("amplitude_km")
    elif family_type == "dpo":
        value = parameters.get("amplitude_target_km")
    elif family_type == "dro":
        # DRO 成员不带参数振幅，按族定义现测：距月心距离 min/max 均值
        if orbit.period is None:
            return None
        minimum, maximum = orbit_family_metric_py(
            float(orbit.system.mu),
            "moon-distance",
            0,
            orbit.states[0],
            float(orbit.period),
            sample_count=1000,
        )
        value = 0.5 * (float(minimum) + float(maximum)) * float(char_length)
    else:
        value = None
    return None if value is None else float(value)


def build_baseline_record(
    request: FamilyGenerationRequest | _DpoFamilyRequest,
    orbit_type: str,
    libration_point: int | None,
    generated: tuple[Any, ...],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """族结果 → 基线 catalog 记录（tags/baseline_version/确定性 record_id）。"""
    family, status, cause, message, requested, generated_members = generated
    built = build_family_bundle(
        request,
        family=family,
        status=status,
        cause=cause,
        message=message,
        requested_members=requested,
        generated_members=generated_members,
    )
    if built is None:
        raise SystemExit(f"{orbit_type} 族零成员，不产生基线记录")
    meta, arrays = built
    meta["record_id"] = baseline_record_id(family.family_type, libration_point)
    meta["tags"] = ["baseline"]
    meta["note"] = "CR3BP 基线轨道族（ADR 0036，地月 DE421 默认参数整族）"
    meta["scalars"]["baseline_version"] = __version__
    amplitudes = [
        amplitude
        for amplitude in (
            _member_amplitude_km(family.family_type, orbit) for orbit in family.orbits
        )
        if amplitude is not None
    ]
    if amplitudes:
        meta["scalars"]["amplitude_envelope_km"] = [min(amplitudes), max(amplitudes)]
    return meta, arrays


def validate_baseline_record(meta: dict[str, Any]) -> None:
    """ADR 0036 决策 7 的入库断言；任一失败抛 ValueError，不写包。"""
    label = meta.get("record_id", "<无 record_id>")

    def _fail(reason: str) -> None:
        raise ValueError(f"基线记录 {label} 校验失败：{reason}")

    if meta.get("status") != ConvergenceState.CONVERGED.value:
        _fail(f"status 非成功侧（{meta.get('status')!r}，message={meta.get('message')!r}）")
    members = meta.get("members") or []
    if not members:
        _fail("零成员")
    scalars = meta.get("scalars", {})
    if scalars.get("member_count") != len(members):
        _fail(f"member_count（{scalars.get('member_count')}）与成员数（{len(members)}）不一致")
    for key in ("requested_members", "generated_members"):
        if not isinstance(scalars.get(key), int):
            _fail(f"覆盖元数据缺 {key}")
    if not isinstance(scalars.get("baseline_version"), str) or not scalars["baseline_version"]:
        _fail("缺基线版本号")
    if "baseline" not in meta.get("tags", []):
        _fail("缺 baseline 标签")
    classification = meta.get("classification", {})
    amplitude = classification.get("amplitude")
    if not (isinstance(amplitude, list) and len(amplitude) == 2):
        _fail(f"振幅覆盖区间缺失（{amplitude!r}）")
    envelope = scalars.get("amplitude_envelope_km")
    if not (
        isinstance(envelope, list)
        and len(envelope) == 2
        and envelope[0] <= envelope[1]
        and envelope[1] > 0.0
    ):
        _fail(f"参数振幅覆盖区间缺失或退化（{envelope!r}）")
    if not meta.get("message"):
        _fail("终止原因 message 缺失")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "__init__.py").touch()
    with tempfile.TemporaryDirectory() as tmp:
        store = CatalogStore(tmp)
        rows = []
        for orbit_type, libration_point in BASELINE_SPEC:
            if orbit_type == "DPO":
                request: Any = _DpoFamilyRequest(2000.0, 60000.0)
            else:
                request = FamilyGenerationRequest(
                    orbit_type=orbit_type,
                    libration_point=libration_point,
                    n_orbits=N_ORBITS,
                )
            generated = _generate(request)
            meta, arrays = build_baseline_record(request, orbit_type, libration_point, generated)
            validate_baseline_record(meta)
            store.put(meta, arrays)
            rows.append(meta)
        # 校验全部通过才落包：清掉旧基线文件（保留 __init__.py），复制新记录
        for old in OUTPUT_DIR.glob("*.json"):
            old.unlink()
        for old in OUTPUT_DIR.glob("*.npz"):
            old.unlink()
        for meta in rows:
            record_id = meta["record_id"]
            for suffix in (".json", ".npz"):
                shutil.copy2(store.records_dir / f"{record_id}{suffix}", OUTPUT_DIR)
    total_bytes = sum(
        p.stat().st_size for p in OUTPUT_DIR.iterdir() if p.suffix in (".json", ".npz")
    )
    print(f"基线版本：{__version__}")
    for meta in rows:
        amp = meta["scalars"]["amplitude_envelope_km"]
        print(
            f"  {meta['record_id']:<24} 成员 {meta['scalars']['member_count']:>3} 条"
            f"（请求 {meta['scalars']['requested_members']}）"
            f"  振幅 {amp[0]:.0f}~{amp[1]:.0f} km"
        )
    print(f"共 {len(rows)} 条记录，{total_bytes / 1024:.0f} KB → {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
