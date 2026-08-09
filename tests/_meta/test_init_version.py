"""e2m2e 包元数据与子包导入测试。

验证版本号、作者、导出列表与可导入性。
"""

import pytest

import e2m2e

pytestmark = pytest.mark.aux


class TestPackageMetadata:
    def test_version_attribute_exists(self):
        assert hasattr(e2m2e, "__version__")
        assert isinstance(e2m2e.__version__, str)
        assert len(e2m2e.__version__) > 0

    def test_author(self):
        assert e2m2e.__author__ == "天疆说"

    def test_email(self):
        assert e2m2e.__email__ == "ouyangjiahong22@nudt.edu.cn"

    def test_all_exports(self):
        assert "data" in e2m2e.__all__
        assert "algorithm" in e2m2e.__all__
        assert "api" in e2m2e.__all__
        assert "tools" in e2m2e.__all__
        assert "mbse" in e2m2e.__all__
        assert "integrators" in e2m2e.__all__

    def test_subpackages_importable(self):
        assert hasattr(e2m2e, "data")
        assert hasattr(e2m2e, "algorithm")
        assert hasattr(e2m2e, "api")
        assert hasattr(e2m2e, "tools")
        assert hasattr(e2m2e, "mbse")
        assert hasattr(e2m2e, "integrators")
