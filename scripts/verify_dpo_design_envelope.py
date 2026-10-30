"""DPO 设计包络端到端复核（#841 / ADR 0061）。

用法：
    .venv/bin/python scripts/verify_dpo_design_envelope.py
    .venv/bin/python scripts/verify_dpo_design_envelope.py --probe-failures

默认检查公开设计上界 27000 km 与默认振幅 20000 km：请求走请求模型，
duration 取缺省两圈，检查 segmented 星历修正收敛并打印覆盖天数与耗时。
--probe-failures 绕过请求模型校验探测 27500 km 的两圈请求，记录族行走
可达但星历修正不收敛的证据。

按 ADR 0037 的救济路径，长弧星历回归归 scripts/ 手工诊断，不进默认
pytest。不要在 scripts/ 里用 uv run，它会触发 editable 重建。
"""

from __future__ import annotations

import argparse
import time

from e2m2e.algorithm.design import OrbitDesignResult, design_orbit
from e2m2e.api.models import DesignOrbitRequest
from e2m2e.status import ConvergenceState


def _run(amplitude_km: float, *, bypass_validation: bool) -> tuple[OrbitDesignResult, float]:
    if bypass_validation:
        # 公开包络之外的点只用于诊断，不能走请求模型
        request = DesignOrbitRequest.model_construct(
            orbit_type="DPO",
            amplitude=amplitude_km,
            phase=0.5001,
            epoch=(2024, 1, 1, 0, 0, 0.0),
            duration=None,
            output_step=3600.0,
            correction_method="segmented",
            correction_revolutions=1,
            earth_degree=10,
            moon_degree=10,
        )
    else:
        request = DesignOrbitRequest(orbit_type="DPO", amplitude=amplitude_km)
    start = time.perf_counter()
    result = design_orbit(request)
    return result, time.perf_counter() - start


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-failures",
        action="store_true",
        help="追加 27500 km 的算法层不收敛探测",
    )
    args = parser.parse_args()

    failed = False
    for amplitude_km in (20000.0, 27000.0):
        result, elapsed = _run(amplitude_km, bypass_validation=False)
        correction = result.correction
        correction_status = correction.status if correction is not None else None
        print(
            f"{amplitude_km:.0f} km: status={result.status} correction={correction_status} "
            f"duration_day={result.duration_day:.3f} {elapsed:.1f}s"
        )
        if result.status is not ConvergenceState.CONVERGED:
            failed = True
    if failed:
        print("公开包络内的点未达预期，DPO 轨道设计包络需要重新标定")
        return 1

    if args.probe_failures:
        try:
            probe, elapsed = _run(27500.0, bypass_validation=True)
        except Exception as exc:  # 诊断脚本要如实打印任意失败
            print(f"27500 km: {type(exc).__name__} {exc}")
            return 0
        if probe.status is ConvergenceState.CONVERGED:
            print(f"27500 km: 意外收敛 status={probe.status} {elapsed:.1f}s")
            return 1
        print(f"27500 km: 按预期未收敛 status={probe.status} {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
