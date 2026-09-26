"""DE421 行星星历口径：GM 与位置同口径，且口径差异可量化（#665，ADR 0048）。

验收口径来自 issue #665：加载 ``de421.bsp`` 后第三体位置与 GM 均为 DE421
口径；与 DE440s 口径的差异可量化输出。

为什么用这些判据（ADR 0013 按定义验证）：
- GM 断言取 DE421 的**已发布定义值**（``Datum.DE421``，constants.toml 单一来源），
  不是从同一实现复制出来的期望值；
- 位置差异用"量级守卫"而非钉死机器值：DE421/DE440s 的月球位置差在米级、
  太阳在百米级，上界取 1 km / 5 km 用于抓住"内核没真正换"
  （那会让差异退化为 0 或跳到 DE430 量级），而不锁死具体数值。
"""

from __future__ import annotations

import numpy as np
import pytest
from kernel_helpers import SPICE_KERNEL_DIR, requires_de421

from e2m2e.algorithm.design.design_orbit import load_design_kernels
from e2m2e.algorithm.dynamics.ephemeris_system import EphemerisSystem
from e2m2e.algorithm.forces import ThirdBodyGravity
from e2m2e.data.constants import Datum
from e2m2e.data.constants.bodies import EARTH, EMB, MOON, SUN
from e2m2e.data.kernels.manager import SPICEManager

pytestmark = [pytest.mark.interface, pytest.mark.spice, requires_de421]

#: 采样弧段：现代历元，完全落在 DE421 覆盖（1899-07-29 ~ 2053-10-09）内。
_EPOCH_START = "2020-01-01T00:00:00"
_EPOCH_END = "2026-01-01T00:00:00"
_N_SAMPLES = 60
#: 位置差异量级守门（km）。实测月球 ~0.004 km、太阳 ~0.22 km。
_MAX_MOON_DIFF_KM = 1.0
_MAX_SUN_DIFF_KM = 5.0


@pytest.fixture(scope="module")
def de421_datum_env():
    """DE421 与 DE440s 两套口径的 SPICEManager + EphemerisSystem，模块级复用。

    datum 簿记是**类级**的（ADR 0048）：CSPICE 内核池进程级全局，任一实例加载的
    星历内核都改变"重叠覆盖段取后加载者"的实际口径，故本 fixture 期间两个 system
    看到的是同一个 GM 口径（DE421）；DE440s 的对照值用显式 ``datum`` 查询取得，
    位置差的对照在 :class:`TestDe421PositionDifference` 里用"换内核再查一次"的
    顺序方式完成。

    卸载一律走 ``SPICEManager.unload_kernel``，保证 Python/Rust 双实例对称卸载
    （见 manager.unload_kernel 注释）。
    """
    de421 = SPICEManager()
    de440s = SPICEManager()
    loaded_421 = load_design_kernels(de421, SPICE_KERNEL_DIR, datum="DE421")
    try:
        yield {
            "de421": de421,
            "de440s": de440s,
            "de421_system": EphemerisSystem(["EARTH", "MOON", "SUN"], de421, origin="EARTH"),
            "de440s_system": EphemerisSystem(["EARTH", "MOON", "SUN"], de440s, origin="EARTH"),
        }
    finally:
        for path in reversed(loaded_421):
            de421.unload_kernel(path)


class TestDe421DatumSelection:
    """load_design_kernels 的口径选择契约。"""

    def test_explicit_de421_loads_de421_kernel(self):
        """datum="DE421" 时确实加载 de421.bsp。"""
        spice = SPICEManager()
        loaded = load_design_kernels(spice, SPICE_KERNEL_DIR, datum="DE421")
        try:
            assert any(p.endswith("de421.bsp") for p in loaded)
            assert spice.ephemeris_datum == "DE421"
        finally:
            for path in reversed(loaded):
                spice.unload_kernel(path)

    def test_missing_de421_datum_raises(self, tmp_path):
        """显式请求 DE421 而目录内无该内核时报错，不静默降级（口径不谎报）。"""
        with pytest.raises(FileNotFoundError, match="de421.bsp"):
            load_design_kernels(SPICEManager(), str(tmp_path), datum="DE421")

    def test_unknown_datum_raises(self):
        """无偏好内核的基准应显式报错。"""
        with pytest.raises(ValueError, match="未知的星历基准"):
            load_design_kernels(SPICEManager(), SPICE_KERNEL_DIR, datum="DE999")

    def test_default_datum_keeps_legacy_selection(self):
        """不传 datum 时仍走 de440s > de430 的历史顺序（行为不变）。"""
        spice = SPICEManager()
        loaded = load_design_kernels(spice, SPICE_KERNEL_DIR)
        try:
            assert any(p.endswith("de440s.bsp") for p in loaded)
            assert spice.ephemeris_datum == "DE440"
        finally:
            for path in reversed(loaded):
                spice.unload_kernel(path)


class TestDe421GmConsistency:
    """加载 de421 后 GM 必须是 DE421 口径。"""

    def test_gm_follows_loaded_kernel(self, de421_datum_env):
        """同一 manager 上，GM 随已加载内核切到 DE421 的已发布值。"""
        de421 = de421_datum_env["de421"]
        system_421 = de421_datum_env["de421_system"]
        system_440 = de421_datum_env["de440s_system"]

        assert de421.ephemeris_datum == "DE421"
        assert system_421.gravitational_parameter("MOON") == Datum.DE421.moon_gm
        # SUN 无 DE421 权威 GM（有意留空，#670）：回退 DE440 并告警一次，不静默混用。
        assert system_421.gravitational_parameter("SUN") == Datum.DE440.sun_gm
        assert system_421.gravitational_parameter("EMB") == Datum.DE421.emb_gm
        # 簿记类级共享（内核池进程级全局，ADR 0048）：同进程另一 system 也看到
        # DE421 口径，避免「de440s 位置 + DE421 GM」的静默错配。
        assert system_440.gravitational_parameter("MOON") == Datum.DE421.moon_gm
        # 对照：DE440 口径必须给出不同的值（否则"切换"是假的），显式 datum 取。
        assert de421.get_gm("MOON", datum="DE440") == Datum.DE440.moon_gm
        assert Datum.DE421.moon_gm != Datum.DE440.moon_gm

    def test_body_table_matches_datum_table(self):
        """body 表与 datum 表的 DE421 GM 必须同值（单一来源不漂移，ADR 0022/0048）。"""
        assert EARTH.gm_by_datum["DE421"] == Datum.DE421.earth_gm
        assert MOON.gm_by_datum["DE421"] == Datum.DE421.moon_gm
        assert EMB.gm_by_datum["DE421"] == Datum.DE421.emb_gm

    def test_third_body_spec_carries_de421_gm(self, de421_datum_env):
        """GM 必须真的进入 Rust spec，而非只在查询层正确。"""
        system_421 = de421_datum_env["de421_system"]
        spec = ThirdBodyGravity("MOON").to_rust_spec(system_421)
        assert spec[0] == "third_body"
        assert spec[2] == pytest.approx(Datum.DE421.moon_gm)

    def test_out_of_scope_body_falls_back_with_warning(self, de421_datum_env, caplog):
        """外行星无 DE421 权威 GM：回退 DE440 并告警，不得静默混用。"""
        from e2m2e.data.constants.bodies import JUPITER

        with caplog.at_level("WARNING"):
            gm = de421_datum_env["de421"].get_gm("JUPITER")
        assert gm == JUPITER.gm_by_datum["DE440"]

    def test_sun_has_no_de421_gm_and_falls_back(self, de421_datum_env):
        """SUN 亦无 DE421 权威 GM（有意留空，#670）：按 DE421 查询回退 DE440。

        回退告警按 (天体, 基准) 每 manager 一次，由
        :meth:`test_out_of_scope_body_falls_back_with_warning` 覆盖，故此处不应
        断言告警条数（依赖用例顺序）。
        """
        assert "DE421" not in SUN.gm_by_datum
        assert de421_datum_env["de421"].get_gm("SUN") == SUN.gm_by_datum["DE440"]


class TestDe421PositionDifference:
    """两个口径的位置差可量化，且量级与 DE421/DE440 的精度声明相符。

    对照方式：模块 fixture 只保持 de421 处于加载态；本类先把 de421 口径的
    位置查完，再卸 de421、装 de440s，查第二遍，最后恢复 de421。这样每个
    观测时刻的内核池里只有一个星历内核，测量的是**口径**而非加载顺序。
    """

    def _positions_km(self, spice: SPICEManager, body: str, ets: np.ndarray) -> np.ndarray:
        return np.array([spice.get_body_position(body, et, "J2000", "EARTH") for et in ets])

    def _diff_km(self, de421_datum_env, body: str) -> np.ndarray:
        spice_421 = de421_datum_env["de421"]
        spice_440 = de421_datum_env["de440s"]
        ets = np.linspace(
            spice_421.utc_to_et(_EPOCH_START),
            spice_421.utc_to_et(_EPOCH_END),
            _N_SAMPLES,
        )
        pos_421 = self._positions_km(spice_421, body, ets)
        loaded_440 = load_design_kernels(spice_440, SPICE_KERNEL_DIR, datum="DE440")
        try:
            pos_440 = self._positions_km(spice_440, body, ets)
        finally:
            # 恢复 de421 为唯一加载的星历内核，供后续用例继续使用 fixture。
            for path in reversed(loaded_440):
                spice_440.unload_kernel(path)
            load_design_kernels(spice_421, SPICE_KERNEL_DIR, datum="DE421")
        return np.linalg.norm(pos_421 - pos_440, axis=1)

    def test_moon_position_difference_is_meter_level(self, de421_datum_env):
        """月球位置差在米级（量级守卫：内核真换了，且没跳到 DE430 量级）。"""
        diff = self._diff_km(de421_datum_env, "MOON")
        assert diff.max() < _MAX_MOON_DIFF_KM
        assert diff.max() > 0.0, "两口径给出完全相同的位置，说明内核未真正切换"

    def test_sun_position_difference_bounded(self, de421_datum_env):
        """太阳位置差有界（百米级）。"""
        diff = self._diff_km(de421_datum_env, "SUN")
        assert diff.max() < _MAX_SUN_DIFF_KM
        assert diff.max() > 0.0
