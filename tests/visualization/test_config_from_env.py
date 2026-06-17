"""PlotConfig.from_env 类方法测试。

验证环境变量缺失、有效值、非数字/非正回退与 overrides 优先级。
"""

from __future__ import annotations

import pytest

from e2m2e.visualization.config import BODY_ICON_SCALE_ENV, PlotConfig


class TestFromEnvBodyIconScale:
    """测试 BODY_ICON_SCALE_ENV 环境变量解析。"""

    def test_env_missing_uses_field_default(self):
        config = PlotConfig.from_env(env={})

        assert config.primary_body_icon_scale == 1.0
        assert config.secondary_body_icon_scale == 1.0

    def test_env_valid_applies_to_both_bodies(self):
        config = PlotConfig.from_env(env={BODY_ICON_SCALE_ENV: "0.25"})

        assert config.primary_body_icon_scale == 0.25
        assert config.secondary_body_icon_scale == 0.25

    @pytest.mark.parametrize("raw", ["abc", "", "1.0.0", "nan_value"])
    def test_env_non_numeric_falls_back_silently(self, raw: str):
        config = PlotConfig.from_env(env={BODY_ICON_SCALE_ENV: raw})

        assert config.primary_body_icon_scale == 1.0
        assert config.secondary_body_icon_scale == 1.0

    @pytest.mark.parametrize("raw", ["0", "-1", "-0.5", "0.0"])
    def test_env_non_positive_falls_back_silently(self, raw: str):
        config = PlotConfig.from_env(env={BODY_ICON_SCALE_ENV: raw})

        assert config.primary_body_icon_scale == 1.0
        assert config.secondary_body_icon_scale == 1.0


class TestFromEnvOverridesPrecedence:
    """测试显式 overrides 优先级高于环境变量。"""

    def test_overrides_beat_env_for_primary(self):
        config = PlotConfig.from_env(
            env={BODY_ICON_SCALE_ENV: "0.25"},
            primary_body_icon_scale=0.5,
        )

        assert config.primary_body_icon_scale == 0.5
        # 未被 override 的字段仍走 env
        assert config.secondary_body_icon_scale == 0.25

    def test_overrides_beat_env_for_both(self):
        config = PlotConfig.from_env(
            env={BODY_ICON_SCALE_ENV: "0.25"},
            primary_body_icon_scale=0.5,
            secondary_body_icon_scale=0.75,
        )

        assert config.primary_body_icon_scale == 0.5
        assert config.secondary_body_icon_scale == 0.75

    def test_overrides_apply_when_env_missing(self):
        config = PlotConfig.from_env(env={}, primary_body_icon_scale=0.3)

        assert config.primary_body_icon_scale == 0.3
        assert config.secondary_body_icon_scale == 1.0


class TestFromEnvOtherFields:
    """from_env 不应影响无关字段。"""

    def test_other_fields_use_defaults(self):
        config = PlotConfig.from_env(env={BODY_ICON_SCALE_ENV: "0.25"})

        # 抽查几个字段，确保 from_env 没有意外修改
        assert config.title == 16
        assert config.colormap == "coolwarm"
        assert config.dpi == 100

    def test_overrides_can_set_unrelated_fields(self):
        config = PlotConfig.from_env(env={}, title=20, dpi=150)

        assert config.title == 20
        assert config.dpi == 150
