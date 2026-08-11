"""constants.toml 加载器（单一来源文件路径只维护一处）。"""

from pathlib import Path

import tomllib

_CONSTANTS_TOML = Path(__file__).resolve().parents[3] / "constants.toml"


def _load_section(section: str) -> dict[str, object]:
    if not _CONSTANTS_TOML.is_file():
        raise FileNotFoundError(
            f"物理常数单一来源文件缺失：{_CONSTANTS_TOML}\n"
            f"请确认仓库根存在 constants.toml，它是 Python/Rust 物理常数的唯一来源。"
        )
    with _CONSTANTS_TOML.open("rb") as f:
        return tomllib.load(f)[section]
