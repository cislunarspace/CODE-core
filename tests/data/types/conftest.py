"""tests/format 共享 fixture：格式测试真实样例文件目录。

夹具来自实际运行输出与既有标定样本，复制入库以避免引用仓库外路径。
"""

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir():
    return FIXTURES
