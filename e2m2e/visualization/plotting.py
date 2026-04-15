"""学术绘图工具模块

提供学术论文标准的 matplotlib 字体配置和便捷绘图函数。
"""

from __future__ import annotations


def configure_academic_fonts():
    """配置学术论文标准的 matplotlib 字体。

    设置 Times New Roman 字体族、STIX 数学字体、标准字号和图例样式。
    适用于 IEEE/Elsevier 等学术期刊的插图要求。
    """
    import matplotlib

    # 字体族：Times New Roman（学术标准）
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    # 字号配置（标题 → 轴标 → 刻度标签 → 图例）
    matplotlib.rcParams["font.size"] = 11
    matplotlib.rcParams["axes.labelsize"] = 12
    matplotlib.rcParams["axes.titlesize"] = 13
    matplotlib.rcParams["xtick.labelsize"] = 10
    matplotlib.rcParams["ytick.labelsize"] = 10
    matplotlib.rcParams["legend.fontsize"] = 9
    # 数学文本使用 STIX 字体（与 Times 风格匹配）
    matplotlib.rcParams["mathtext.fontset"] = "stix"
    matplotlib.rcParams["mathtext.rm"] = "serif"
    matplotlib.rcParams["mathtext.it"] = "serif:italic"
    matplotlib.rcParams["mathtext.bf"] = "serif:bold"
    matplotlib.rcParams["axes.unicode_minus"] = False  # 使用 - 而非 Unicode 负号
    # 图例：带边框、半透明、无阴影（学术标准样式）
    matplotlib.rcParams["legend.frameon"] = True
    matplotlib.rcParams["legend.framealpha"] = 0.9
    matplotlib.rcParams["legend.fancybox"] = True
    matplotlib.rcParams["legend.shadow"] = False
