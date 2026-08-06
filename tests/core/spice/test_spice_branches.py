"""SPICEManager 未覆盖分支测试（无需真实内核）。

覆盖内核文件缺失、GM 缓存、闰秒内核搜索。
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from e2m2e.data.kernels.manager import _GM_VALUES, SPICEManager


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
        from e2m2e.data.kernels.manager import _find_leapseconds_kernel

        with patch("e2m2e.data.kernels.manager._LEAPSECOND_SEARCH_PATHS", ["", "/nonexistent"]):
            result = _find_leapseconds_kernel()
            assert result is None


@pytest.fixture
def isolated_leapsecond_state():
    """隔离闰秒加载状态与仓库 kernels/ 目录，避免类变量跨测试串扰。"""
    SPICEManager._leapseconds_loaded = False
    # 把默认搜索路径指向不存在的目录，确保只依赖测试自造的文件。
    with patch(
        "e2m2e.data.kernels.manager._LEAPSECOND_SEARCH_PATHS",
        ["", "/nonexistent/isolated"],
    ):
        yield
    SPICEManager._leapseconds_loaded = False


class TestLoadKernelAutoLoadsLeapseconds:
    """load_kernel 应在被加载 .bsp 的同级目录查找并自动加载 .tls。"""

    def test_sibling_dir_leapseconds_loaded(self, tmp_path, isolated_leapsecond_state):
        tls_path = tmp_path / "naif0012.tls"
        bsp_path = tmp_path / "de440s.bsp"
        tls_path.write_text("fake tls")
        bsp_path.write_bytes(b"fake bsp")

        fake_spice = MagicMock()
        with (
            patch("e2m2e.data.kernels.manager.get_spiceypy", return_value=fake_spice),
            # 屏蔽 Rust cspice furnsh：假 bsp 进 cspice 会报错，且与闰秒逻辑无关。
            patch("e2m2e.integrators.spice_poc_furnsh", None),
        ):
            mgr = SPICEManager()
            mgr.load_kernel(str(bsp_path))

        # 闰秒内核（同级目录）应先于 .bsp 被 furnsh。
        assert SPICEManager._leapseconds_loaded is True
        loaded = [call.args[0] for call in fake_spice.furnsh.call_args_list]
        assert str(tls_path) in loaded
        assert loaded[0] == str(tls_path)
        assert str(bsp_path) in loaded

    def test_missing_leapseconds_warns_without_raising(
        self, tmp_path, isolated_leapsecond_state, caplog
    ):
        # 仅放 .bsp，无任何 .tls。
        bsp_path = tmp_path / "de440s.bsp"
        bsp_path.write_bytes(b"fake bsp")

        fake_spice = MagicMock()
        with (
            patch("e2m2e.data.kernels.manager.get_spiceypy", return_value=fake_spice),
            patch("e2m2e.integrators.spice_poc_furnsh", None),
        ):
            mgr = SPICEManager()
            with caplog.at_level(logging.WARNING, logger="e2m2e.data.kernels.manager"):
                # 不应抛异常：用户仍可自行 furnsh。
                mgr.load_kernel(str(bsp_path))

        # 未找到 .tls：标志保持 False（允许重试），仅 .bsp 被 furnsh。
        assert SPICEManager._leapseconds_loaded is False
        loaded = [call.args[0] for call in fake_spice.furnsh.call_args_list]
        assert loaded == [str(bsp_path)]
        assert any("naif0012.tls" in r.message for r in caplog.records)
        assert any("NOLEAPSECONDS" in r.message for r in caplog.records)
