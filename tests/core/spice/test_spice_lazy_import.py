"""SPICE 模块延迟加载测试。

验证 e2m2e.core.spice 不强制加载 spiceypy 直到首次调用需要 SPICE 的方法。
"""

import subprocess
import sys
from pathlib import Path


class TestSpiceLazyImport:
    """测试 e2m2e.core.spice 模块的延迟加载行为。"""

    @staticmethod
    def _run_in_subprocess(code: str) -> tuple[int, str, str]:
        """在干净子进程中执行 Python 代码。"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode, result.stdout, result.stderr

    def test_import_spice_module_does_not_load_spiceypy(self):
        """加载 e2m2e.core.spice 模块文件时不应加载 spiceypy。"""
        code = """
import importlib.util
import sys
from pathlib import Path

module_path = Path(r"%s")
spec = importlib.util.spec_from_file_location("e2m2e.core.spice", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print("spiceypy_loaded:", "spiceypy" in sys.modules)
""" % (Path(__file__).parent.parent.parent.parent / "e2m2e" / "core" / "spice.py")
        returncode, stdout, stderr = self._run_in_subprocess(code)

        assert returncode == 0, f"子进程退出码非零: stderr={stderr}"
        assert "spiceypy_loaded: False" in stdout, (
            f"加载 e2m2e.core.spice 时 spiceypy 被加载: stdout={stdout}"
        )

    def test_import_spicemanager_does_not_load_spiceypy(self):
        """从 e2m2e.core.spice 导入 SPICEManager 类不应加载 spiceypy。"""
        code = """
import importlib.util
import sys
from pathlib import Path

module_path = Path(r"%s")
spec = importlib.util.spec_from_file_location("e2m2e.core.spice", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
manager_class = module.SPICEManager
print("spiceypy_loaded:", "spiceypy" in sys.modules)
""" % (Path(__file__).parent.parent.parent.parent / "e2m2e" / "core" / "spice.py")
        returncode, stdout, stderr = self._run_in_subprocess(code)

        assert returncode == 0, f"子进程退出码非零: stderr={stderr}"
        assert "spiceypy_loaded: False" in stdout, (
            f"导入 SPICEManager 时 spiceypy 被加载: stdout={stdout}"
        )

    def test_spicemanager_methods_load_spiceypy_lazily(self):
        """SPICEManager 首次调用需要 SPICE 的方法时才加载 spiceypy。"""
        code = """
import importlib.util
import sys
from pathlib import Path

module_path = Path(r"%s")
spec = importlib.util.spec_from_file_location("e2m2e.core.spice", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
mgr = module.SPICEManager()
print("after_init:", "spiceypy" in sys.modules)
# get_gm 命中本地缓存，不访问 SPICE 内核，因此不应加载 spiceypy
gm = mgr.get_gm("EARTH")
print("after_get_gm_cached:", "spiceypy" in sys.modules)
print("gm:", gm)
# get_gm("UNKNOWN_BODY") 未命中缓存，会调用 bodvrd，此时才加载 spiceypy。
# 由于子进程没有 SPICE 内核，bodvrd 会抛出异常；我们只需验证 spiceypy 被加载。
try:
    mgr.get_gm("UNKNOWN_BODY_FOR_LAZY_LOAD_TEST")
except Exception:
    pass
print("after_get_gm_uncached:", "spiceypy" in sys.modules)
""" % (Path(__file__).parent.parent.parent.parent / "e2m2e" / "core" / "spice.py")
        returncode, stdout, stderr = self._run_in_subprocess(code)

        assert returncode == 0, f"子进程退出码非零: stderr={stderr}"
        assert "after_init: False" in stdout
        assert "after_get_gm_cached: False" in stdout
        assert "gm: 398600" in stdout
        assert "after_get_gm_uncached: True" in stdout
