"""Cover apply_rcparams and get_cmap on PlotConfig."""

import matplotlib

from e2m2e.visualization.config import PlotConfig


class TestApplyRcParams:
    def test_apply_sets_font_params(self):
        config = PlotConfig(title=20, label=18, tick=15, legend=16)
        config.apply_rcparams()
        assert matplotlib.rcParams["axes.titlesize"] == 20
        assert matplotlib.rcParams["axes.labelsize"] == 18
        assert matplotlib.rcParams["xtick.labelsize"] == 15
        assert matplotlib.rcParams["legend.fontsize"] == 16

    def test_apply_sets_serif_font(self):
        config = PlotConfig()
        config.apply_rcparams()
        family = matplotlib.rcParams["font.family"]
        assert "serif" in family
        assert "Times New Roman" in matplotlib.rcParams["font.serif"]


class TestGetCmap:
    def test_default_cmap(self):
        config = PlotConfig()
        cmap = config.get_cmap()
        assert cmap.name == "coolwarm"

    def test_custom_cmap(self):
        config = PlotConfig(colormap="viridis")
        cmap = config.get_cmap()
        assert cmap.name == "viridis"
