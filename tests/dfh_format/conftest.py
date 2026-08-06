"""tests/dfh_format 共享 fixture：DFH 真实样例文件目录。

夹具来自 DFH_DAC 实际运行输出与 MATLAB 库的 _golden 目录，
复制入库以避免引用仓库外路径。
"""

import sys
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"

# 把 scripts/ 加入 sys.path，使 tests/dfh_format 可从 scripts.dfh_* 导入
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


@pytest.fixture
def fixtures_dir():
    return FIXTURES
