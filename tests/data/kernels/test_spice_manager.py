"""SPICE 星历内核集成测试（Layer 1a）。

覆盖 SPICEManager 初始化、内核加载错误契约、批量 ET→UTC（Rust 闰秒
换算，仅依赖闰秒表 .tls，不 furnsh 星历内核）与星历内核目录搜索。
"""

import datetime
import os

import pytest
from kernel_helpers import SPICE_KERNEL_DIR, de421_kernel_file, requires_de421

from e2m2e.data.constants import Datum
from e2m2e.data.constants.bodies import JUPITER, MOON
from e2m2e.data.kernels.manager import SPICEManager

pytestmark = [
    pytest.mark.data,
    pytest.mark.spice,
]


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def spice_kernel_dir():
    """返回 SPICE 内核文件所在目录，不存在或无内核文件则跳过。"""
    if not os.path.isdir(SPICE_KERNEL_DIR):
        pytest.skip("SPICE kernel directory not found, set SPICE_KERNEL_DIR")
    bsp_files = [f for f in os.listdir(SPICE_KERNEL_DIR) if f.endswith(".bsp")]
    if not bsp_files:
        pytest.skip("No .bsp kernel files found in SPICE kernel directory")
    return SPICE_KERNEL_DIR


@pytest.fixture
def bare_spice_manager():
    """未加载内核的裸 SPICEManager 实例（测类本身 API 用）。"""
    return SPICEManager()


@pytest.fixture
def rust_leapseconds():
    """仅向 Rust CSPICE 内核池 furnsh 闰秒表（.tls），teardown 卸载。

    Rust ``et2utc`` 系入口预检要求内核池非空，且实际换算需要闰秒表；
    批量 ET→UTC 只查 LSK，无需星历内核（de440）。

    须钉死 naif0011.tls：CSPICE 内核池允许多个 LSK，**最后 furnsh 者生效**；
    naif0011 与 naif0012 对 2017-01-01 闰秒的收录不同（37@2017 仅后者有），
    本文件测试的常量 ET 按 naif0011 语义标定，et2utc 必须用同一闰秒表。
    """
    from e2m2e.spice_ext import spice_furnsh, spice_unload

    if spice_furnsh is None or spice_unload is None:
        pytest.skip("Rust spice_ext extension not available")
    path = os.path.join(SPICE_KERNEL_DIR, "naif0011.tls")
    if not os.path.isfile(path):
        pytest.skip(f"leapseconds kernel not found: {path}")
    spice_furnsh(path)
    yield path
    spice_unload(path)


# =============================================================================
# Test SPICEManager 初始化
# =============================================================================
class TestSPICEManagerInit:
    """测试 SPICEManager 创建和基本属性"""

    def test_create_instance(self):
        """应能创建 SPICEManager 实例"""
        manager = SPICEManager()
        assert manager is not None

    def test_has_load_kernel_method(self, bare_spice_manager):
        """SPICEManager 应有 load_kernel 方法"""
        assert hasattr(bare_spice_manager, "load_kernel")
        assert callable(bare_spice_manager.load_kernel)

    def test_has_unload_kernel_method(self, bare_spice_manager):
        """SPICEManager 应有 unload_kernel 方法"""
        assert hasattr(bare_spice_manager, "unload_kernel")

    def test_has_utc_to_et_method(self, bare_spice_manager):
        """SPICEManager 应有 utc_to_et 方法"""
        assert hasattr(bare_spice_manager, "utc_to_et")

    def test_has_get_body_state_method(self, bare_spice_manager):
        """SPICEManager 应有 get_body_state 方法"""
        assert hasattr(bare_spice_manager, "get_body_state")


# =============================================================================
# Test 内核加载
# =============================================================================
class TestSPICEKernelLoading:
    """测试 SPICE 内核加载的错误契约（不 furnsh 真实星历）"""

    def test_load_nonexistent_kernel_raises(self, bare_spice_manager):
        """加载不存在的文件应抛出异常"""
        with pytest.raises((FileNotFoundError, OSError, RuntimeError)):
            bare_spice_manager.load_kernel("/nonexistent/path/de440.bsp")


# =============================================================================
# Test 批量时间转换（Rust 闰秒换算）
# =============================================================================
class TestSPICETimeConversion:
    """测试批量 ET→UTC（Rust 批量闰秒换算，仅依赖 LSK）"""

    def test_batch_et_to_utc_round_trip(self, rust_leapseconds):
        """批量 ET→UTC 与已知历元分量互逆（对称性）。

        输入为常量 ET（SPICE str2et 精确值，TDB 秒 past J2000，与期望
        UTC 分量一一对应），不经 SPICEManager 取值；仅 furnsh 一次闰秒
        表，验证 Rust 批量闰秒换算的数学正确性。
        """
        from e2m2e.integrators import batch_et_to_utc_py, require_rust_extension

        require_rust_extension("batch_et_to_utc_py")
        et_cases = [
            ("2024-01-01T00:00:00", 757339268.1839061),
            ("2025-06-21T11:00:06", 803775674.1843798),
            ("2026-12-31T23:59:59", 852033667.1839125),
            ("2017-01-01T00:00:00", 536500868.1839298),  # 2017-01-01 闰秒生效时刻
        ]
        year, month, day, hour, minute, second = batch_et_to_utc_py([et for _, et in et_cases])
        for k, (utc, _) in enumerate(et_cases):
            dt = datetime.datetime.fromisoformat(utc)
            assert (year[k], month[k], day[k], hour[k], minute[k], second[k]) == (
                dt.year,
                dt.month,
                dt.day,
                dt.hour,
                dt.minute,
                float(dt.second),
            )


# =============================================================================
# Test 星历内核搜索
# =============================================================================
class TestSPICEManagerFindEphemerisKernel:
    """需求: SPICEManager 应提供公开方法在指定目录中搜索星历内核文件。

    背景:
        transfer-orbit-design 的 correct_dro_to_ephemeris.py 中有 find_spice_kernel()
        函数，硬编码了 e2m2e/kernels 路径并按优先级搜索 .bsp 文件。
        此逻辑应属于 e2m2e 的 SPICEManager，使上层脚本无需重复实现。

    接口:
        bare_spice_manager.find_ephemeris_kernel(search_dir: str) -> str
        - search_dir: 要搜索的目录路径
        - 返回: 找到的第一个 .bsp 内核文件的绝对路径
        - 按优先级搜索: de440.bsp > de440s.bsp > de435.bsp > de438.bsp
        - 找不到则抛出 FileNotFoundError
    """

    def test_has_find_ephemeris_kernel_method(self, bare_spice_manager):
        """SPICEManager 应有 find_ephemeris_kernel 方法"""
        assert hasattr(bare_spice_manager, "find_ephemeris_kernel")
        assert callable(bare_spice_manager.find_ephemeris_kernel)

    @pytest.mark.spice
    def test_find_kernel_in_valid_directory(self, bare_spice_manager, spice_kernel_dir):
        """在包含内核文件的目录中应能找到并返回路径"""
        path = bare_spice_manager.find_ephemeris_kernel(spice_kernel_dir)
        assert os.path.exists(path)
        assert path.endswith(".bsp")

    @pytest.mark.spice
    def test_find_kernel_returns_existing_file(self, bare_spice_manager, spice_kernel_dir):
        """返回的路径应指向一个实际存在的文件"""
        path = bare_spice_manager.find_ephemeris_kernel(spice_kernel_dir)
        assert os.path.isfile(path)

    def test_find_kernel_priority_de440_over_de438(self, bare_spice_manager, tmp_path):
        """当 de440 和 de438 同时存在时，应返回 de440"""
        (tmp_path / "de440.bsp").write_bytes(b"fake")
        (tmp_path / "de438.bsp").write_bytes(b"fake")
        path = bare_spice_manager.find_ephemeris_kernel(str(tmp_path))
        assert path.endswith("de440.bsp")

    def test_find_kernel_priority_de440s_over_de435(self, bare_spice_manager, tmp_path):
        """当 de440s 和 de435 同时存在时，应返回 de440s"""
        (tmp_path / "de440s.bsp").write_bytes(b"fake")
        (tmp_path / "de435.bsp").write_bytes(b"fake")
        path = bare_spice_manager.find_ephemeris_kernel(str(tmp_path))
        assert path.endswith("de440s.bsp")

    def test_find_kernel_fallback_to_de435(self, bare_spice_manager, tmp_path):
        """当只有 de435 存在时，应返回 de435"""
        (tmp_path / "de435.bsp").write_bytes(b"fake")
        path = bare_spice_manager.find_ephemeris_kernel(str(tmp_path))
        assert path.endswith("de435.bsp")

    def test_find_kernel_fallback_to_de438(self, bare_spice_manager, tmp_path):
        """当只有 de438 存在时，应返回 de438"""
        (tmp_path / "de438.bsp").write_bytes(b"fake")
        path = bare_spice_manager.find_ephemeris_kernel(str(tmp_path))
        assert path.endswith("de438.bsp")

    def test_find_kernel_raises_when_not_found(self, bare_spice_manager, tmp_path):
        """目录中无内核文件时应抛出 FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            bare_spice_manager.find_ephemeris_kernel(str(tmp_path))

    def test_find_kernel_raises_when_dir_not_exists(self, bare_spice_manager):
        """目录不存在时应抛出 FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            bare_spice_manager.find_ephemeris_kernel("/nonexistent/path/to/kernels")

    @requires_de421
    def test_find_kernel_preferred_de421(self, bare_spice_manager, tmp_path):
        """显式请求 DE421 口径时，de421.bsp 优先于其它 DE 系列（ADR 0048）。"""
        import shutil

        shutil.copy(de421_kernel_file(), tmp_path / "de421.bsp")
        (tmp_path / "de440s.bsp").write_bytes(b"fake")
        path = bare_spice_manager.find_ephemeris_kernel(str(tmp_path), preferred="DE421")
        assert path.endswith("de421.bsp")

    def test_find_kernel_preferred_missing_raises(self, bare_spice_manager, tmp_path):
        """显式请求的口径内核缺失时报错，不静默降级到其它 DE 系列。"""
        (tmp_path / "de435.bsp").write_bytes(b"fake")
        with pytest.raises(FileNotFoundError, match="de421.bsp"):
            bare_spice_manager.find_ephemeris_kernel(str(tmp_path), preferred="DE421")

    def test_find_kernel_unknown_preferred_raises(self, bare_spice_manager, tmp_path):
        """未知基准没有偏好内核，显式请求应报错而非回落到默认优先级。"""
        (tmp_path / "de440s.bsp").write_bytes(b"fake")
        with pytest.raises(ValueError, match="未知的星历基准"):
            bare_spice_manager.find_ephemeris_kernel(str(tmp_path), preferred="DE999")


# =============================================================================
# Test GM 基准与星历内核配对（ADR 0048）
# =============================================================================
class TestEphemerisDatumGM:
    """GM 口径跟随已加载星历内核；缺该基准记录时回退 DE440 并告警。

    背景：SPK 内核本身不携带 GM（``bodvrd(..., "GM")`` 报 KERNELVARNOTFOUND），
    GM 只能来自 constants.toml 的声明式 body 表，故 datum 由内核**文件名**推断。
    """

    def test_default_datum_is_de440(self, bare_spice_manager):
        """未加载任何星历内核时，口径为 ADR 0022 的星历动力学默认 DE440。"""
        assert bare_spice_manager.ephemeris_datum == "DE440"
        assert bare_spice_manager.get_gm("MOON") == MOON.gm_by_datum["DE440"]

    def test_explicit_datum_overrides_active(self, bare_spice_manager):
        """显式 datum 覆盖当前口径；天体名大小写不敏感。"""
        assert bare_spice_manager.get_gm("moon", datum="DE421") == Datum.DE421.moon_gm
        assert bare_spice_manager.get_gm("MOON", datum="de421") == Datum.DE421.moon_gm

    def test_missing_datum_falls_back_to_de440_with_warning(self, bare_spice_manager, caplog):
        """请求基准下无该天体记录时回退 DE440 并告警一次（不静默混用）。"""
        with caplog.at_level("WARNING"):
            gm = bare_spice_manager.get_gm("JUPITER", datum="DE421")
            bare_spice_manager.get_gm("JUPITER", datum="DE421")
        assert gm == JUPITER.gm_by_datum["DE440"]
        warnings = [r for r in caplog.records if "JUPITER" in r.getMessage()]
        assert len(warnings) == 1, "同一 (天体, 基准) 组合只应告警一次"

    @requires_de421
    def test_loading_de421_switches_gm_datum(self, monkeypatch):
        """加载 de421.bsp 后 GM 切到 DE421；卸载后回落默认口径。

        furnsh 被替换为 no-op：只验证 manager 的 datum 簿记，不触碰内核池
        （真实加载路径由 test_de421_datum.py 端到端覆盖）。
        """
        import e2m2e.spice_ext as spice_ext

        monkeypatch.setattr(spice_ext, "spice_furnsh", None, raising=False)
        monkeypatch.setattr(spice_ext, "spice_unload", None, raising=False)
        path = de421_kernel_file()
        mgr = SPICEManager()
        start = mgr.get_gm("MOON")
        mgr.load_kernel(path)
        assert mgr.ephemeris_datum == "DE421"
        assert mgr.get_gm("MOON") == Datum.DE421.moon_gm
        assert mgr.get_gm("MOON") != start
        mgr.unload_kernel(path)
        assert mgr.ephemeris_datum == "DE440"
        assert mgr.get_gm("MOON") == start

    @requires_de421
    def test_last_loaded_ephemeris_wins(self, monkeypatch):
        """同时加载两个星历内核时，口径取后加载者（与 SPICE 优先级规则一致）。"""
        import e2m2e.spice_ext as spice_ext

        monkeypatch.setattr(spice_ext, "spice_furnsh", None, raising=False)
        monkeypatch.setattr(spice_ext, "spice_unload", None, raising=False)
        de440s = os.path.join(SPICE_KERNEL_DIR, "de440s.bsp")
        if not os.path.isfile(de440s):
            pytest.skip("de440s.bsp not available")
        mgr = SPICEManager()
        de421 = de421_kernel_file()
        mgr.load_kernel(de421)
        mgr.load_kernel(de440s)
        assert mgr.ephemeris_datum == "DE440"
        assert mgr.get_gm("MOON") == Datum.DE440.moon_gm
        # 卸载后加载者 → 回落到仍在加载的 de421
        mgr.unload_kernel(de440s)
        assert mgr.ephemeris_datum == "DE421"
        assert mgr.get_gm("MOON") == Datum.DE421.moon_gm


class TestDatumBookkeepingScope:
    """口径簿记与告警的作用域（#683 评审修复，ADR 0048）。

    内核池是 CSPICE 进程级全局的，故 datum 簿记必须**类级共享**；告警去重则按
    **manager 实例**，避免跨测试/跨调用方耦合。
    """

    def test_bookkeeping_is_class_level(self, monkeypatch):
        """任一实例加载的星历内核，对同进程其它实例同样生效（与池同域）。"""
        monkeypatch.setattr(SPICEManager, "_loaded_ephemeris", [("/tmp/de421.bsp", "DE421")])
        assert SPICEManager().ephemeris_datum == "DE421"

    def test_fallback_warning_is_per_instance(self, monkeypatch, caplog):
        """同一回退组合在两个实例上各告警一次（去重按实例，非进程全局）。"""
        monkeypatch.setattr(SPICEManager, "_loaded_ephemeris", [("/tmp/de421.bsp", "DE421")])
        with caplog.at_level("WARNING"):
            SPICEManager().get_gm("JUPITER")
            SPICEManager().get_gm("JUPITER")
        warnings = [r for r in caplog.records if "JUPITER" in r.getMessage()]
        assert len(warnings) == 2

    def test_datum_kernel_name_single_source(self):
        """基准→内核映射由 manager 单源导出（design 链路同引用，无第二份副本）。"""
        assert SPICEManager.datum_kernel_name("de421") == "de421.bsp"
        assert SPICEManager.datum_kernel_name("DE440") == "de440s.bsp"
        assert SPICEManager.datum_kernel_name("DE999") is None

    def test_find_kernel_preferred_de440_matches_design_contract(
        self, bare_spice_manager, tmp_path
    ):
        """``preferred="DE440"`` 与 ``load_design_kernels(datum="DE440")`` 同契约。"""
        (tmp_path / "de440s.bsp").write_bytes(b"fake")
        path = bare_spice_manager.find_ephemeris_kernel(str(tmp_path), preferred="DE440")
        assert path.endswith("de440s.bsp")


class TestKernelDatumWarnings:
    """未收录 / 近似口径内核在加载时告警（落实 ADR 0048「绝不静默混用」）。"""

    def test_approximated_kernel_warns_once(self, bare_spice_manager, caplog):
        """de441 等按 DE440 近似的内核：按内核名告警一次。"""
        with caplog.at_level("WARNING"):
            bare_spice_manager._warn_approximated_kernel("de441", "DE440")
            bare_spice_manager._warn_approximated_kernel("de441", "DE440")
        warnings = [r for r in caplog.records if "de441" in r.getMessage()]
        assert len(warnings) == 1

    def test_authoritative_kernel_does_not_warn(self, bare_spice_manager, caplog):
        """自有权威 GM 表的内核（de421/de440s）不告警。"""
        with caplog.at_level("WARNING"):
            bare_spice_manager._warn_approximated_kernel("de421", "DE421")
            bare_spice_manager._warn_approximated_kernel("de440s", "DE440")
        assert caplog.records == []

    def test_unregistered_kernel_warns(self, bare_spice_manager, caplog):
        """匹配 DE 命名但未收录的内核：不改变口径但必须告警。"""
        with caplog.at_level("WARNING"):
            bare_spice_manager._warn_unregistered_kernel("de423")
        warnings = [r for r in caplog.records if "de423" in r.getMessage()]
        assert len(warnings) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
