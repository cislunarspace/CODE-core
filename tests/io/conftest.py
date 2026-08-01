"""tests/io 共享 fixture：DFH 真实样例文件目录。

夹具来自 DFH_DAC 实际运行输出与 MATLAB 库的 _golden 目录，
复制入库以避免引用仓库外路径。
"""

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir():
    return FIXTURES
