#!/usr/bin/env python
"""RO 星历链路手工诊断(#627):``design_orbit(RO)`` 端到端真实调用。

按 ADR 0037 的救济路径归属 ``scripts/``:pytest 放不下这条链路——
``tests/conftest.py`` 把 Rust rayon/OMP/BLAS 钉单线程(确定性所需),
two_level 星历修正的多段传播在单线程下实测 >600 s;本脚本在多线程
独立进程下实测约 219 s(duration=300_000 s、output_step=36_000 s 画像)。

已知问题:duration 缩到 50_000 s(短于修正弧)时修正分段失稳狂奔
(实测 >600 s 未收敛),另行跟进,故本脚本固定用 300_000 s 稳定画像。

用法::

    uv run --no-sync python scripts/design_ro_ephemeris_smoke.py

链路断言与 ``tests/algorithm/design/test_design_orbit_smoke.py`` 的
ELFO 冒烟同口径;任何一条不达标即以非零退出码结束。
"""

from __future__ import annotations

import sys
import time

import numpy as np

from e2m2e.algorithm.design import design_orbit
from e2m2e.api.models import DesignOrbitRequest
from e2m2e.status import ConvergenceState


def main() -> int:
    request = DesignOrbitRequest(
        orbit_type="RO",
        resonance_p=4,
        resonance_q=1,
        phase=0.0,
        duration=300_000.0,  # 3.5 天短弧:实测 219 s 的最便宜稳定画像
        output_step=36_000.0,
    )
    start = time.perf_counter()
    result = design_orbit(request)
    elapsed = time.perf_counter() - start
    correction_status = result.correction.status if result.correction is not None else None
    print(f"elapsed={elapsed:.1f}s status={result.status} corr={correction_status}")

    checks = {
        "orbit_type": result.orbit_type == "RO",
        "status_converged": result.status is ConvergenceState.CONVERGED,
        "correction_converged": (
            result.correction is not None and result.correction.status is ConvergenceState.CONVERGED
        ),
        "ephemeris_nonempty": len(result.ephemeris) > 0,
    }
    orbit = result.cr3bp_orbit
    checks["cr3bp_orbit_present"] = orbit is not None and orbit.period is not None
    if orbit is not None:
        # 闭合残差(无量纲)达标 + 周期精确通约(T = 2πq = 2π)
        checks["closure_within_contract"] = (
            orbit.closure_error is not None and orbit.closure_error < 1e-6
        )
        checks["period_exact_commensurability"] = (
            orbit.period is not None and abs(orbit.period - 2.0 * np.pi) / (2.0 * np.pi) < 1e-9
        )

    for name, passed in checks.items():
        print(f"{'OK  ' if passed else 'FAIL'} {name}")
    ok = all(checks.values())
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
