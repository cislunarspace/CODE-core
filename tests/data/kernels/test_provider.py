"""data/kernels/：SPICEManager 与 EphemerisProvider 抽象测试。"""

import pytest

from e2m2e.data.kernels.manager import SPICEManager
from e2m2e.data.kernels.provider import EphemerisProvider

pytestmark = [
    pytest.mark.data,
    pytest.mark.spice,
]


class TestSPICEManager:
    def test_implements_provider_interface(self):
        assert issubclass(SPICEManager, EphemerisProvider)

    def test_find_kernel_priority(self, tmp_path):
        (tmp_path / "de440.bsp").write_bytes(b"fake")
        (tmp_path / "de438.bsp").write_bytes(b"fake")
        assert SPICEManager().find_ephemeris_kernel(str(tmp_path)).endswith("de440.bsp")


class TestEphemerisProviderAbstract:
    def test_unimplemented_methods_raise(self):
        provider = EphemerisProvider()
        for call in (
            lambda: provider.utc_to_tai("2025-01-01T00:00:00"),
            lambda: provider.body_rotation("MOON", 0.0, "J2000", "MOON_PA"),
        ):
            with pytest.raises(NotImplementedError):
                call()

    def test_interface_documented(self):
        """接口 docstring 明确三类方法（时间/状态/帧）。"""
        doc = EphemerisProvider.__doc__ or ""
        assert "时间" in doc and "状态" in doc and "帧" in doc
