"""SPICEManager 未覆盖分支测试（无需真实内核）。

覆盖内核文件缺失、GM 缓存、闰秒内核搜索。
"""

from unittest.mock import patch

import pytest

from e2m2e.core.spice import _GM_VALUES, SPICEManager


class TestLoadKernelFileNotFound:
    def test_raises_for_missing_file(self):
        mgr = SPICEManager()
        with pytest.raises(FileNotFoundError, match="Kernel file not found"):
            mgr.load_kernel("/nonexistent/path.bsp")


class TestGetGmFromCache:
    def test_earth_gm_from_cache(self):
        mgr = SPICEManager()
        gm = mgr.get_gm("EARTH")
        assert gm == _GM_VALUES["EARTH"]

    def test_moon_gm_from_cache(self):
        mgr = SPICEManager()
        gm = mgr.get_gm("MOON")
        assert gm == _GM_VALUES["MOON"]

    def test_sun_gm_from_cache(self):
        mgr = SPICEManager()
        gm = mgr.get_gm("SUN")
        assert gm == _GM_VALUES["SUN"]

    def test_case_insensitive_lookup(self):
        mgr = SPICEManager()
        assert mgr.get_gm("earth") == _GM_VALUES["EARTH"]


class TestFindEphemerisKernelNonexistent:
    def test_nonexistent_dir_raises(self):
        mgr = SPICEManager()
        with pytest.raises(FileNotFoundError):
            mgr.find_ephemeris_kernel("/nonexistent/dir")


class TestFindLeapsecondsKernel:
    def test_returns_none_for_empty_paths(self):
        from e2m2e.core.spice import _find_leapseconds_kernel

        with patch("e2m2e.core.spice._LEAPSECOND_SEARCH_PATHS", ["", "/nonexistent"]):
            result = _find_leapseconds_kernel()
            assert result is None
