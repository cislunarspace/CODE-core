"""
core/__init__.py 延迟导出测试

验证 e2m2e.core 子包本身对 SPICE 相关符号使用 __getattr__ 按需延迟导出，
使用户只导入非星历符号时不会强制加载 spiceypy。
"""

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest


class TestCoreLazyExports:
    """测试 e2m2e.core 包对 SPICE 相关符号的延迟导出。"""

    @staticmethod
    def _core_package_path() -> Path:
        return Path(__file__).parent.parent.parent / "e2m2e" / "core" / "__init__.py"

    @staticmethod
    def _load_core_directly():
        """绕过 e2m2e/__init__.py，直接加载 e2m2e.core 子包。

        父包会在 slice 3 中处理，此处只验证 core 子包自身的延迟导出行为。
        """
        module_path = TestCoreLazyExports._core_package_path()
        spec = importlib.util.spec_from_file_location(
            "e2m2e.core", module_path, submodule_search_locations=[str(module_path.parent)]
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        # 需要先注册父包占位符，否则子模块相对导入会失败
        core_path = TestCoreLazyExports._core_package_path()
        e2m2e_pkg = types.ModuleType("e2m2e")
        e2m2e_pkg.__path__ = [str(core_path.parent.parent)]
        sys.modules["e2m2e"] = e2m2e_pkg
        sys.modules["e2m2e.core"] = module
        spec.loader.exec_module(module)
        return module

    def test_import_non_spice_symbols_does_not_load_spiceypy(self):
        """从 e2m2e.core 导入非星历符号时不应加载 spiceypy。"""
        code = """
import sys
import types
from pathlib import Path
import importlib.util

module_path = Path(r"%s")
spec = importlib.util.spec_from_file_location(
    "e2m2e.core", module_path, submodule_search_locations=[str(module_path.parent)]
)
module = importlib.util.module_from_spec(spec)
e2m2e_pkg = types.ModuleType("e2m2e")
e2m2e_pkg.__path__ = [str(module_path.parent.parent)]
sys.modules["e2m2e"] = e2m2e_pkg
sys.modules["e2m2e.core"] = module
spec.loader.exec_module(module)

OrbitFamily = module.OrbitFamily
CR3BP_System = module.CR3BP_System
print("spiceypy_loaded:", "spiceypy" in sys.modules)
print("orbit_family:", OrbitFamily)
print("cr3bp_system:", CR3BP_System)
""" % self._core_package_path()
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, f"子进程退出码非零: stderr={result.stderr}"
        assert "spiceypy_loaded: False" in result.stdout, (
            f"导入非星历符号时 spiceypy 被加载: stdout={result.stdout}"
        )

    def test_spice_symbols_are_accessible_through_lazy_export(self):
        """SPICE 相关符号仍可通过 e2m2e.core 访问并正常工作。"""
        code = """
import sys
import types
from pathlib import Path
import importlib.util

module_path = Path(r"%s")
spec = importlib.util.spec_from_file_location(
    "e2m2e.core", module_path, submodule_search_locations=[str(module_path.parent)]
)
module = importlib.util.module_from_spec(spec)
e2m2e_pkg = types.ModuleType("e2m2e")
e2m2e_pkg.__path__ = [str(module_path.parent.parent)]
sys.modules["e2m2e"] = e2m2e_pkg
sys.modules["e2m2e.core"] = module
spec.loader.exec_module(module)

SPICEManager = module.SPICEManager
print("spiceypy_loaded:", "spiceypy" in sys.modules)
print("manager:", SPICEManager)
mgr = SPICEManager()
gm = mgr.get_gm("EARTH")
print("gm:", gm)
""" % self._core_package_path()
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, f"子进程退出码非零: stderr={result.stderr}"
        assert "spiceypy_loaded: False" in result.stdout, (
            f"仅引用 SPICEManager 类时 spiceypy 被加载: stdout={result.stdout}"
        )
        assert "gm: 398600" in result.stdout

    def test_all_public_api_exports_preserved(self):
        """core/__init__.__all__ 中导出的公开 API 仍然可用。"""
        module = self._load_core_directly()
        expected = [
            "SPICEManager",
            "CR3BP_System",
            "OrbitFamily",
            "CoordinateTransformation",
            "ITRFSpiceAxes",
            "CelestialBodyOrigin",
            "standard_itrf",
        ]
        for name in expected:
            assert hasattr(module, name), f"core 缺少导出: {name}"
