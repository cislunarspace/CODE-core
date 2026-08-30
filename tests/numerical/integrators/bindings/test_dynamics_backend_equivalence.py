"""星历动力学 Rust 绑定的可用性契约测试。

Rust/SciPy 对照与 STM 传播的物理正确性验证已随端到端测试裁剪移除；
本文件仅保留扩展缺失时的能力错误契约，后者由
tests/algorithm/dynamics 的定义性测试与数值抽查覆盖。
"""

import pytest

from e2m2e.exceptions import RustExtensionUnavailableError

pytestmark = [
    pytest.mark.integrator,
    pytest.mark.spice,
]


class TestRustStmAvailability:
    """测试 Rust STM 模块的能力错误。"""

    def test_rust_stm_missing_extension_raises_capability_error(self, monkeypatch):
        """扩展缺失时在使用处给出构建指引，不影响模块导入。"""
        import e2m2e.integrators as integrators

        monkeypatch.setattr(integrators, "_rust_extension", None)
        monkeypatch.setattr("e2m2e.spice_ext._abi_ok", False)
        monkeypatch.setattr(integrators, "propagate_with_stm_py", None)

        with pytest.raises(RustExtensionUnavailableError, match="make dev"):
            integrators.require_rust_extension("propagate_with_stm_py")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
