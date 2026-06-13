"""
algorithms/__init__.py 延迟按需导入测试

验证 import e2m2e.algorithms 时不应强制加载所有算法子模块，
特别是不应加载依赖 spiceypy 的星历修正/多重打靶模块；
按需导入的符号（如 Continuation）仍可正常使用。
"""

import subprocess
import sys
from pathlib import Path

import pytest


class TestAlgorithmsLazyImports:
    """测试 e2m2e.algorithms 包的按需导入行为。"""

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

    def test_import_algorithms_does_not_load_spiceypy(self):
        """导入 e2m2e.algorithms 包时不应加载 spiceypy。"""
        code = """
import sys
import e2m2e.algorithms
print("spiceypy_loaded:", "spiceypy" in sys.modules)
"""
        returncode, stdout, stderr = self._run_in_subprocess(code)

        assert returncode == 0, f"子进程退出码非零: stderr={stderr}"
        assert "spiceypy_loaded: False" in stdout, (
            f"导入 e2m2e.algorithms 时 spiceypy 被加载: stdout={stdout}"
        )

    def test_import_continuation_does_not_load_ephemeris_modules(self):
        """from e2m2e.algorithms import Continuation 不应加载星历相关模块。"""
        code = """
import sys
from e2m2e.algorithms import Continuation
print("spiceypy_loaded:", "spiceypy" in sys.modules)
print("continuation:", Continuation)
print("ephemeris_correction_loaded:", "e2m2e.algorithms.ephemeris_correction" in sys.modules)
print("multiple_shooting_loaded:", "e2m2e.algorithms.multiple_shooting" in sys.modules)
"""
        returncode, stdout, stderr = self._run_in_subprocess(code)

        assert returncode == 0, f"子进程退出码非零: stderr={stderr}"
        assert "spiceypy_loaded: False" in stdout
        assert "continuation: <class 'e2m2e.algorithms.continuation.Continuation'>" in stdout
        assert "ephemeris_correction_loaded: False" in stdout
        assert "multiple_shooting_loaded: False" in stdout

    def test_ephemeris_symbols_still_accessible(self):
        """星历相关算法符号仍可通过 e2m2e.algorithms 按需访问。"""
        code = """
import sys
from e2m2e.algorithms import EphemerisCorrectionResult, MultipleShooting
print("spiceypy_loaded:", "spiceypy" in sys.modules)
print("ephemeris_correction_result:", EphemerisCorrectionResult)
print("multiple_shooting:", MultipleShooting)
"""
        returncode, stdout, stderr = self._run_in_subprocess(code)

        assert returncode == 0, f"子进程退出码非零: stderr={stderr}"
        assert "spiceypy_loaded: False" in stdout
        assert "ephemeris_correction_result: <class" in stdout
        assert "multiple_shooting: <class" in stdout
