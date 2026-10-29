"""DRO 大振幅星历设计诊断（#838 / ADR 0061）。

用法：
    .venv/bin/python scripts/verify_dro_amplitude_envelope.py
    .venv/bin/python scripts/verify_dro_amplitude_envelope.py --probe-failures

默认运行 150000 km 的 design_orbit（请求模型默认一年 duration），预期
CONVERGED 且星历修正收敛。--probe-failures 追加 200000、300000、400000 km
的算法层探测，记录不收敛证据；这些振幅超出 150000 km 的公开包络，脚本
绕过请求模型校验直接调用算法层。带 --probe-failures 的完整运行约十分钟。

不要在 scripts/ 里用 uv run，它会触发 editable 重建。
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
            orbit_type="DRO",
            amplitude=amplitude_km,
            phase=0.5001,
            epoch=(2024, 1, 1, 0, 0, 0.0),
            duration=31557600.0,
            output_step=3600.0,
            correction_method="two_level",
            correction_revolutions=1,
            earth_degree=10,
            moon_degree=10,
        )
    else:
        request = DesignOrbitRequest(orbit_type="DRO", amplitude=amplitude_km)
    start = time.perf_counter()
    result = design_orbit(request)
    return result, time.perf_counter() - start


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-failures",
        action="store_true",
        help="追加 200000、300000、400000 km 的算法层不收敛探测",
    )
    args = parser.parse_args()

    result, elapsed = _run(150000.0, bypass_validation=False)
    correction_status = result.correction.status if result.correction is not None else None
    print(f"150000 km: status={result.status} correction={correction_status} {elapsed:.1f}s")
    if (
        result.status is not ConvergenceState.CONVERGED
        or correction_status is not ConvergenceState.CONVERGED
    ):
        print("150000 km 未达到预期，DRO 轨道设计包络需要重新标定")
        return 1
    print("150000 km 通过")

    unexpected_convergence = False
    if args.probe_failures:
        for amplitude_km in (200000.0, 300000.0, 400000.0):
            start = time.perf_counter()
            try:
                probe, elapsed = _run(amplitude_km, bypass_validation=True)
                if probe.status is ConvergenceState.CONVERGED:
                    print(f"{amplitude_km:.0f} km: 意外收敛 status={probe.status} {elapsed:.1f}s")
                    unexpected_convergence = True
                else:
                    print(
                        f"{amplitude_km:.0f} km: 按预期未收敛 status={probe.status} {elapsed:.1f}s"
                    )
            except Exception as exc:  # 诊断脚本要如实打印任意失败
                elapsed = time.perf_counter() - start
                print(f"{amplitude_km:.0f} km: {type(exc).__name__} {exc} {elapsed:.1f}s")
    return 1 if unexpected_convergence else 0


if __name__ == "__main__":
    raise SystemExit(main())
