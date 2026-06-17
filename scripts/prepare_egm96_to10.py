"""从完整 EGM96 .gfc 文件生成截断到 degree 10 的包内数据文件。"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Truncate EGM96 .gfc to degree 10")
    parser.add_argument("--input", required=True, type=Path, help="Full EGM96 .gfc file")
    parser.add_argument("--output", required=True, type=Path, help="Output truncated file")
    parser.add_argument("--max-degree", type=int, default=10, help="Maximum degree to keep")
    args = parser.parse_args()

    output_lines: list[str] = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("/*"):
                continue
            parts = stripped.split()
            if parts[0].lower() == "max_degree":
                output_lines.append(f"max_degree {args.max_degree}\n")
                continue
            if parts[0].lower() == "gfc":
                n = int(parts[1])
                if n > args.max_degree:
                    continue
            output_lines.append(line)

    args.output.write_text("".join(output_lines), encoding="utf-8")
    print(f"Wrote {args.output} with max_degree={args.max_degree}")


if __name__ == "__main__":
    main()
