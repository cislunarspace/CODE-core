#!/usr/bin/env python3
"""示例绘图共用配置：跨平台中文字体探测 + rcParams 应用。

matplotlib 默认字体不含中文字符，标题含中文会显示豆腐块。示例脚本统一
从这里探测系统可用的 CJK 字体并加入回退链，本机（SimHei/YaHei/Noto Sans
SC）与服务器（Noto Sans CJK JP/AR PL UKai）都能正确显示中文。

用法（在示例脚本 main() 内、import matplotlib 之后调用）：
    from _plot_setup import setup_cjk_font
    setup_cjk_font()
"""

from __future__ import annotations

import matplotlib

# 常见 CJK 字体族名，按可读性偏好排序；跨平台探测时按此顺序选第一个可用
_CJK_FAMILIES = [
    "Microsoft YaHei",  # Windows
    "SimHei",  # Windows
    "Noto Sans SC",  # Windows/macOS 部分
    "Noto Sans CJK SC",  # Linux
    "Noto Sans CJK JP",  # Linux（部分发行版只有 JP 字面）
    "WenQuanYi Micro Hei",  # Linux
    "AR PL UKai CN",  # Linux
    "PingFang SC",  # macOS
]

_configured = False


def setup_cjk_font() -> None:
    """把系统可用的第一个 CJK 字体加入 matplotlib 回退链（幂等）。"""
    global _configured
    if _configured:
        return

    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for family in _CJK_FAMILIES:
        if family in available:
            # 在现有 sans-serif 链前插 CJK 字体，ASCII 字符仍走原字体
            chain = [family, *matplotlib.rcParams["font.sans-serif"]]
            matplotlib.rcParams["font.sans-serif"] = chain
            matplotlib.rcParams["font.family"] = "sans-serif"
            _configured = True
            return

    # 无可探测字体：静默跳过，保持默认（标题中文可能显示豆腐块）
    _configured = True
