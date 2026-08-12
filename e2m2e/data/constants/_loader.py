"""constants.toml 加载器（单一来源文件路径只维护一处）。"""

from pathlib import Path

import tomllib

# 单一来源文件随包分发（wheel/sdist 均含），包内定位在仓库与安装布局下一致。
_CONSTANTS_TOML = Path(__file__).resolve().parent / "constants.toml"


def _load_section(section: str) -> dict[str, object]:
    if not _CONSTANTS_TOML.is_file():
        raise FileNotFoundError(
            f"物理常数单一来源文件缺失：{_CONSTANTS_TOML}\n"
            f"请确认 e2m2e/data/constants/constants.toml 随包安装，"
            f"它是 Python/Rust 物理常数的唯一来源。"
        )
    with _CONSTANTS_TOML.open("rb") as f:
        return tomllib.load(f)[section]
