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

        只把 Rust 桥（``e2m2e.spice_ext.spice_furnsh``/``spice_unload``）置为
        None，Python 侧 ``spiceypy.furnsh``/``unload`` 仍**真实执行**：本用例验证
        manager 的 datum 簿记与口径切换，故必须保证 de421 一定被卸载（try/finally），
        否则泄漏到同 worker 的后续用例。真实双池加载路径由 test_de421_datum.py
        端到端覆盖。
        """
        import e2m2e.spice_ext as spice_ext

        monkeypatch.setattr(spice_ext, "spice_furnsh", None, raising=False)
        monkeypatch.setattr(spice_ext, "spice_unload", None, raising=False)
        path = de421_kernel_file()
        mgr = SPICEManager()
        start = mgr.get_gm("MOON")
        try:
            mgr.load_kernel(path)
            assert mgr.ephemeris_datum == "DE421"
            assert mgr.get_gm("MOON") == Datum.DE421.moon_gm
            assert mgr.get_gm("MOON") != start
        finally:
            mgr.unload_kernel(path)
        assert mgr.ephemeris_datum == "DE440"
        assert mgr.get_gm("MOON") == start

    @requires_de421
    def test_last_loaded_ephemeris_wins(self, monkeypatch):
        """同时加载两个星历内核时，口径取后加载者（与 SPICE 优先级规则一致）。

        同 ``test_loading_de421_switches_gm_datum``：Rust 桥置 None，Python 侧
        ``spiceypy.furnsh``/``unload`` 真实执行，故两个内核都必须在 try/finally
        中卸载（de440s 在 finally 里重复卸载是幂等 no-op）。
        """
        import e2m2e.spice_ext as spice_ext

        monkeypatch.setattr(spice_ext, "spice_furnsh", None, raising=False)
        monkeypatch.setattr(spice_ext, "spice_unload", None, raising=False)
        de440s = os.path.join(SPICE_KERNEL_DIR, "de440s.bsp")
        if not os.path.isfile(de440s):
            pytest.skip("de440s.bsp not available")
        mgr = SPICEManager()
        de421 = de421_kernel_file()
        try:
            mgr.load_kernel(de421)
            mgr.load_kernel(de440s)
            assert mgr.ephemeris_datum == "DE440"
            assert mgr.get_gm("MOON") == Datum.DE440.moon_gm
            # 卸载后加载者 → 回落到仍在加载的 de421
            mgr.unload_kernel(de440s)
            assert mgr.ephemeris_datum == "DE421"
            assert mgr.get_gm("MOON") == Datum.DE421.moon_gm
        finally:
            # 簿记类级/进程级（ADR 0048）：不卸载会泄漏到同 worker 的后续用例。
            mgr.unload_kernel(de440s)
            mgr.unload_kernel(de421)


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

    def test_datum_kernel_names_single_source(self):
        """基准→内核候选由 manager 单源导出（design 链路同引用，无第二份副本）。"""
        assert SPICEManager.datum_kernel_names("de421") == ("de421.bsp",)
        assert SPICEManager.datum_kernel_names("DE440") == ("de440.bsp", "de440s.bsp")
        assert SPICEManager.datum_kernel_names("DE999") == ()

    def test_find_kernel_preferred_de440_matches_design_contract(
        self, bare_spice_manager, tmp_path
    ):
        """``preferred="DE440"`` 与 ``load_design_kernels(datum="DE440")`` 同契约。

        同基准的多个内核等价：目录只有 ``de440.bsp`` 时也必须接受（否则与
        manager 默认优先级 de440.bsp 优先自相矛盾）。
        """
        (tmp_path / "de440s.bsp").write_bytes(b"fake")
        path = bare_spice_manager.find_ephemeris_kernel(str(tmp_path), preferred="DE440")
        assert path.endswith("de440s.bsp")
        (tmp_path / "de440s.bsp").unlink()
        (tmp_path / "de440.bsp").write_bytes(b"fake")
        path = bare_spice_manager.find_ephemeris_kernel(str(tmp_path), preferred="DE440")
        assert path.endswith("de440.bsp")


class _RecordingSpicePy:
    """spiceypy 桩：记录 furnsh/unload 调用，并记录调用时簿记锁是否已被持有。"""

    def __init__(self) -> None:
        #: (方法名, 绝对路径, 调用时 ``_bookkeeping_lock.locked()``)
        self.calls: list[tuple[str, str, bool]] = []

    def _record(self, method: str, path: str) -> None:
        self.calls.append((method, os.path.abspath(path), SPICEManager._bookkeeping_lock.locked()))

    def furnsh(self, path: str) -> None:
        """记录一次 furnsh。"""
        self._record("furnsh", path)

    def unload(self, path: str) -> None:
        """记录一次 unload。"""
        self._record("unload", path)


class TestKernelBookkeepingAtomicity:
    """furnsh 与簿记同一临界区 + 失败路径回滚（#697，ADR 0048 Revision (c)）。

    全部用例用 monkeypatch 桩替掉 spiceypy 与 Rust 桥，不触碰真实 CSPICE 池，
    故不依赖线程/sleep，也不依赖执行顺序。
    """

    @requires_de421
    def test_python_furnsh_runs_inside_bookkeeping_lock(self, monkeypatch):
        """Python furnsh 与簿记在同一临界区：furnsh 时簿记锁已被持有。

        否则并发加载不同内核时，簿记末位可能晚于池里实际生效的末位，出现
        「位置按新内核、GM 按旧簿记」的静默错配。
        """
        import e2m2e.data.kernels.manager as manager_module
        import e2m2e.spice_ext as spice_ext

        monkeypatch.setattr(SPICEManager, "_leapseconds_loaded", True)
        monkeypatch.setattr(SPICEManager, "_bodies_registered", True)
        monkeypatch.setattr(SPICEManager, "_loaded_ephemeris", [])
        stub = _RecordingSpicePy()
        monkeypatch.setattr(manager_module, "get_spiceypy", lambda: stub)
        monkeypatch.setattr(spice_ext, "spice_furnsh", None, raising=False)
        path = de421_kernel_file()

        SPICEManager().load_kernel(path)

        abspath = os.path.abspath(path)
        assert stub.calls == [("furnsh", abspath, True)]
        assert SPICEManager._loaded_ephemeris == [(abspath, "DE421")]

    def test_load_warning_runs_outside_bookkeeping_lock(self, monkeypatch, tmp_path):
        """加载期告警在临界区之外调用（``_warn_unregistered_kernel`` 读
        ``ephemeris_datum``，会重入同一把非可重入锁 → 放进锁内即死锁）。

        用 spy 替换告警方法：真发生回归时观察到 ``locked() is True`` 即 fail
        fast，而不是把 CI 挂死。
        """
        import e2m2e.data.kernels.manager as manager_module
        import e2m2e.spice_ext as spice_ext

        monkeypatch.setattr(SPICEManager, "_leapseconds_loaded", True)
        monkeypatch.setattr(SPICEManager, "_bodies_registered", True)
        monkeypatch.setattr(SPICEManager, "_loaded_ephemeris", [])
        stub = _RecordingSpicePy()
        monkeypatch.setattr(manager_module, "get_spiceypy", lambda: stub)
        monkeypatch.setattr(spice_ext, "spice_furnsh", None, raising=False)
        observed: list[tuple[str, bool]] = []
        monkeypatch.setattr(
            SPICEManager,
            "_warn_unregistered_kernel",
            lambda _self, name: observed.append((name, SPICEManager._bookkeeping_lock.locked())),
        )
        unregistered = tmp_path / "de423.bsp"
        unregistered.write_bytes(b"")

        SPICEManager().load_kernel(str(unregistered))

        # 未收录内核不登记簿记，告警也必须发生在临界区之外。
        assert observed == [("de423", False)]
        assert SPICEManager._loaded_ephemeris == []

    @requires_de421
    def test_load_rolls_back_python_furnsh_when_rust_fails(self, monkeypatch):
        """Rust furnsh 失败 → 撤销 Python 侧、不登记簿记、原异常上抛。"""
        import e2m2e.data.kernels.manager as manager_module
        import e2m2e.spice_ext as spice_ext

        monkeypatch.setattr(SPICEManager, "_leapseconds_loaded", True)
        monkeypatch.setattr(SPICEManager, "_bodies_registered", True)
        monkeypatch.setattr(SPICEManager, "_loaded_ephemeris", [])
        stub = _RecordingSpicePy()
        monkeypatch.setattr(manager_module, "get_spiceypy", lambda: stub)

        def failing_furnsh(path: str) -> None:
            raise RuntimeError("rust furnsh failed")

        monkeypatch.setattr(spice_ext, "spice_furnsh", failing_furnsh, raising=False)
        path = de421_kernel_file()

        with pytest.raises(RuntimeError, match="rust furnsh failed"):
            SPICEManager().load_kernel(path)

        abspath = os.path.abspath(path)
        assert stub.calls == [("furnsh", abspath, True), ("unload", abspath, True)]
        assert SPICEManager._loaded_ephemeris == []

    @requires_de421
    def test_unload_restores_python_kernel_when_rust_fails(self, monkeypatch):
        """Rust unload 失败 → Python 侧 re-furnsh 恢复、保留簿记、原异常上抛。"""
        import e2m2e.data.kernels.manager as manager_module
        import e2m2e.spice_ext as spice_ext

        path = de421_kernel_file()
        abspath = os.path.abspath(path)
        stub = _RecordingSpicePy()
        monkeypatch.setattr(manager_module, "get_spiceypy", lambda: stub)
        monkeypatch.setattr(SPICEManager, "_loaded_ephemeris", [(abspath, "DE421")])

        def failing_unload(path: str) -> None:
            raise RuntimeError("rust unload failed")

        monkeypatch.setattr(spice_ext, "spice_unload", failing_unload, raising=False)

        with pytest.raises(RuntimeError, match="rust unload failed"):
            SPICEManager().unload_kernel(path)

        assert stub.calls == [("unload", abspath, True), ("furnsh", abspath, True)]
        assert SPICEManager._loaded_ephemeris == [(abspath, "DE421")]

    @requires_de421
    def test_unload_failure_moves_restored_kernel_to_last(self, monkeypatch):
        """被卸载者不是末位内核时，恢复（重新 furnsh）要把它在簿记中挪到末位。

        Python 池内重新 furnsh 使该内核重成「后加载者」（重叠段生效者），若簿记仍
        留在原位置，就会「位置按该内核、GM 按旧末位」地失配（ADR 0048 Revision (c)）。
        """
        import e2m2e.data.kernels.manager as manager_module
        import e2m2e.spice_ext as spice_ext

        path = de421_kernel_file()
        abspath = os.path.abspath(path)
        other = (os.path.join(os.path.dirname(abspath), "de440s.bsp"), "DE440")
        stub = _RecordingSpicePy()
        monkeypatch.setattr(manager_module, "get_spiceypy", lambda: stub)
        monkeypatch.setattr(SPICEManager, "_loaded_ephemeris", [(abspath, "DE421"), other])

        def failing_unload(path: str) -> None:
            raise RuntimeError("rust unload failed")

        monkeypatch.setattr(spice_ext, "spice_unload", failing_unload, raising=False)

        with pytest.raises(RuntimeError, match="rust unload failed"):
            SPICEManager().unload_kernel(path)

        assert stub.calls == [("unload", abspath, True), ("furnsh", abspath, True)]
        # 恢复即重新加载：de421 移到末位，GM 口径回到 DE421（与 Python 池生效末位一致）
        assert SPICEManager._loaded_ephemeris == [other, (abspath, "DE421")]
        assert SPICEManager().ephemeris_datum == "DE421"

    @requires_de421
    def test_reload_failure_does_not_drop_previous_load(self, monkeypatch):
        """重载同一内核时 Rust 失败：回滚恰为本次 furnsh 的逆操作，先前加载与簿记保留。

        生产链路（``load_design_kernels`` 等）会对同一内核重复 ``load_kernel`` 且
        从不卸载；若回滚多卸一条，就会丢掉先前那次加载。桩按 CSPICE 的实例语义
        （``furnsh`` 记一条、``unload`` 撤销一条）记录调用次数，故这里断言本次
        调用只产生一次 furnsh 与一次 unload。
        """
        import e2m2e.data.kernels.manager as manager_module
        import e2m2e.spice_ext as spice_ext

        path = de421_kernel_file()
        abspath = os.path.abspath(path)
        monkeypatch.setattr(SPICEManager, "_leapseconds_loaded", True)
        monkeypatch.setattr(SPICEManager, "_bodies_registered", True)
        monkeypatch.setattr(SPICEManager, "_loaded_ephemeris", [(abspath, "DE421")])
        stub = _RecordingSpicePy()
        monkeypatch.setattr(manager_module, "get_spiceypy", lambda: stub)

        def failing_furnsh(path: str) -> None:
            raise RuntimeError("rust furnsh failed")

        monkeypatch.setattr(spice_ext, "spice_furnsh", failing_furnsh, raising=False)

        with pytest.raises(RuntimeError, match="rust furnsh failed"):
            SPICEManager().load_kernel(path)

        assert stub.calls == [("furnsh", abspath, True), ("unload", abspath, True)]
        assert SPICEManager._loaded_ephemeris == [(abspath, "DE421")]


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
