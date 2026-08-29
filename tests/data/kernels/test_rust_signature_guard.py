"""SPICEManager Rust 扩展签名漂移守卫的单元测试。

当 ``e2m2e._integrators`` 的编译产物（.pyd/.so）落后于源码时，PyO3
在参数绑定阶段抛出毫无指向的 ``TypeError``（如 "got an unexpected keyword
argument 'sxform_pairs'"），栈顶远离调用点，用户无从得知只需重建。``SPICEManager``
经 ``_call_rust_or_compat_error`` 把这类签名漂移转成带"请重建"提示的
``RuntimeError``。

本文件**不**依赖编译扩展或 SPICE 内核——通过注入伪造的"过期"函数覆盖守卫
逻辑，因此不带 ``pytest.mark.spice``，可在无 spice 构建的环境跑。
"""

from __future__ import annotations

import inspect
import sys
import types

import pytest

from e2m2e import spice_ext as gw
from e2m2e.data.kernels.manager import SPICEManager, _call_rust_or_compat_error
from e2m2e.exceptions import RustExtensionUnavailableError

pytestmark = pytest.mark.data


class TestRustExtensionAvailability:
    """扩展缺失与 ABI 过期时的失败契约。"""

    def test_absent_extension_raises_unavailable_error(self, monkeypatch):
        """扩展缺失不得静默降级。"""
        monkeypatch.setitem(sys.modules, "e2m2e._integrators", None)
        monkeypatch.setattr(gw, "_abi_ok", False)

        with pytest.raises(RustExtensionUnavailableError, match="make dev"):
            gw._check_rust_abi()

    def test_import_time_check_catches_stale_binary(self, monkeypatch):
        """已加载的过期扩展在 ABI 检查时明确失败。"""
        monkeypatch.setitem(
            sys.modules,
            "e2m2e._integrators",
            types.SimpleNamespace(_py_abi_version=lambda: 0),
        )
        monkeypatch.setattr(gw, "_abi_ok", False)
        monkeypatch.setattr(gw, "_MIN_REQUIRED_RUST_ABI", 1)

        with pytest.raises(RuntimeError, match="make dev"):
            gw._check_rust_abi()


# =============================================================================
# 守卫函数单元测试（注入伪造 callable，无需编译扩展）
# =============================================================================
class TestCallRustOrCompatError:
    """``_call_rust_or_compat_error`` 的行为契约。"""

    def test_current_signature_passes_through(self):
        """签名齐全时，正常转发调用并返回结果，kwargs 正确下传。"""

        def current_fn(targets, frame_pairs, et_start, et_end, *, dt=3600.0, sxform_pairs=None):
            return {"targets": targets, "sxform_pairs": sxform_pairs, "dt": dt}

        result = _call_rust_or_compat_error(
            current_fn,
            [("EARTH", "EARTH")],
            [("ITRF93", "J2000")],
            0.0,
            100.0,
            fn_name="enable_ephem_cache",
            required_kwargs=("dt", "sxform_pairs"),
            dt=1800.0,
            sxform_pairs=[("ITRF93", "J2000")],
        )
        assert result["sxform_pairs"] == [("ITRF93", "J2000")]
        assert result["dt"] == 1800.0

    def test_stale_signature_missing_kwarg_raises_runtime_error(self):
        """【回归】签名缺 sxform_pairs（模拟过期 .pyd）→ RuntimeError 而非 TypeError。"""

        def stale_fn(targets, frame_pairs, et_start, et_end, *, dt=3600.0):
            raise AssertionError("过期函数不应被调用：预检应先拦截")

        with pytest.raises(RuntimeError, match="make dev") as excinfo:
            _call_rust_or_compat_error(
                stale_fn,
                [],
                [],
                0.0,
                1.0,
                fn_name="enable_ephem_cache",
                required_kwargs=("dt", "sxform_pairs"),
                dt=3600.0,
                sxform_pairs=[],
            )
        msg = str(excinfo.value)
        # 必须指向"重建"并点名缺失参数
        assert "maturin" in msg
        assert "sxform_pairs" in msg

    def test_unrelated_typeerror_propagates_unchanged(self):
        """与签名漂移无关的 TypeError 不得被吞掉或重映射（避免掩盖真实 bug）。"""

        def fn(targets, frame_pairs, et_start, et_end, *, dt=3600.0, sxform_pairs=None):
            raise TypeError("totally unrelated internal error")

        with pytest.raises(TypeError, match="unrelated") as excinfo:
            _call_rust_or_compat_error(
                fn,
                [],
                [],
                0.0,
                1.0,
                fn_name="enable_ephem_cache",
                required_kwargs=("dt", "sxform_pairs"),
                dt=3600.0,
                sxform_pairs=[],
            )
        # 原始异常类型保留，未被升级为 RuntimeError
        assert type(excinfo.value) is TypeError

    def test_legitimate_dt_typeerror_is_not_remapped(self):
        """【M1 回归】dt 参数类型错误（非漂移）不得被误判为"编译产物过期"。"""

        def fn(targets, frame_pairs, et_start, et_end, *, dt=3600.0, sxform_pairs=None):
            # PyO3 对 dt="3600"（str→f64）的真实错误模板
            raise TypeError("argument 'dt': must be real number, not str")

        with pytest.raises(TypeError, match="must be real number") as excinfo:
            _call_rust_or_compat_error(
                fn,
                [],
                [],
                0.0,
                1.0,
                fn_name="enable_ephem_cache",
                required_kwargs=("dt", "sxform_pairs"),
                dt="3600",  # 故意传错类型触发 PyO3 类型检查
                sxform_pairs=[],
            )
        assert type(excinfo.value) is TypeError

    def test_introspection_valueerror_falls_back_to_call(self, monkeypatch):
        """内省抛 ValueError 时（无 __text_signature__），退化为调用点兜底。"""

        class _NoSig:
            """模拟无 __signature__ 且 __call__ 也不可内省的对象（如部分 C 内建）。"""

            def __call__(self, *args, **kwargs):
                raise TypeError(
                    "enable_ephem_cache() got an unexpected keyword argument 'sxform_pairs'"
                )

        _original_sig = inspect.signature

        def _mock_signature(obj, **kw):
            if isinstance(obj, _NoSig):
                raise ValueError("no signature")
            return _original_sig(obj, **kw)

        monkeypatch.setattr(inspect, "signature", _mock_signature)
        with pytest.raises(RuntimeError, match="maturin"):
            _call_rust_or_compat_error(
                _NoSig(),
                fn_name="enable_ephem_cache",
                required_kwargs=("dt", "sxform_pairs"),
                dt=3600.0,
                sxform_pairs=[],
            )

    def test_call_time_drift_typeerror_remapped(self):
        """签名内省通过、但调用时仍抛漂移型 TypeError → 兜底重映射。

        覆盖 PyO3 函数无 ``__text_signature__``、内省取不到签名的情形：
        预检放行，由调用点捕获 TypeError 并按参数名命中重映射。
        """

        def lying_fn(targets, frame_pairs, et_start, et_end, *, dt=3600.0, sxform_pairs=None):
            raise TypeError(
                "enable_ephem_cache() got an unexpected keyword argument 'sxform_pairs'"
            )

        with pytest.raises(RuntimeError, match="maturin"):
            _call_rust_or_compat_error(
                lying_fn,
                [],
                [],
                0.0,
                1.0,
                fn_name="enable_ephem_cache",
                required_kwargs=("dt", "sxform_pairs"),
                dt=3600.0,
                sxform_pairs=[],
            )


# =============================================================================
# 方法级接线测试（注入伪造 e2m2e._integrators，无需编译扩展/内核）
# =============================================================================
class TestEnableEphemCacheWiring:
    """``SPICEManager.enable_ephem_cache`` 是否真的接上守卫。"""

    @pytest.fixture(autouse=True)
    def _stub_python_cache(self, monkeypatch):
        """跳过 Python 侧缓存构建（依赖 SPICE 采样），只测 Rust 调用分支。"""
        monkeypatch.setattr(
            "e2m2e.data.kernels.ephem_cache.build_ephem_cache",
            lambda *args, **kwargs: object(),
        )

    def test_stale_binary_raises_runtime_error(self, monkeypatch):
        """过期二进制下，方法抛 RuntimeError 并提示 maturin 重建。"""

        def _stale_enable(targets, frame_pairs, et_start, et_end, dt=3600.0):
            raise AssertionError("过期函数不应被调用")

        monkeypatch.setattr("e2m2e.spice_ext.enable_ephem_cache", _stale_enable)

        spice = SPICEManager()
        with pytest.raises(RuntimeError, match="maturin"):
            spice.enable_ephem_cache(["EARTH"], 0.0, 100.0, sxform_pairs=[])

    def test_absent_extension_raises_unavailable_error(self, monkeypatch):
        """扩展缺失时，不得仅保留 Python 缓存而继续运行。"""
        monkeypatch.setattr("e2m2e.spice_ext.enable_ephem_cache", None)

        spice = SPICEManager()
        with pytest.raises(RustExtensionUnavailableError, match="make dev"):
            spice.enable_ephem_cache(["EARTH"], 0.0, 100.0, sxform_pairs=[])
