"""design_orbit 修正方法入口契约。

族→方法的规范化在请求校验层完成（``DesignOrbitRequest``，tests/api 覆盖）；
算法入口只做防御：不稳定族（HALO/NRHO/DPO）携带非 segmented 方法时
fail fast，绝不静默改写。本模块用 duck-typed 请求绕过校验层直达算法
入口，专测该防御检查，无需 SPICE（检查先于任何内核使用）。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from e2m2e.algorithm.design import design_orbit
from tests.algorithm.design.conftest import make_design_request

pytestmark = pytest.mark.orchestration


@pytest.mark.parametrize("method", ["two_level", "standard", "rust"])
def test_unnormalized_unstable_request_fails_fast(method):
    request = make_design_request(orbit_type="HALO", correction_method=method)
    with pytest.raises(ValueError, match="segmented"):
        design_orbit(request, spice=SimpleNamespace())
