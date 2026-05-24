"""Cover __init__.py metadata and subpackage imports."""

import e2m2e


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
        assert "core" in e2m2e.__all__
        assert "algorithms" in e2m2e.__all__
        assert "visualization" in e2m2e.__all__
        assert "transfer" in e2m2e.__all__
        assert "mbse" in e2m2e.__all__

    def test_subpackages_importable(self):
        assert hasattr(e2m2e, "core")
        assert hasattr(e2m2e, "algorithms")
        assert hasattr(e2m2e, "visualization")
        assert hasattr(e2m2e, "transfer")
        assert hasattr(e2m2e, "mbse")
