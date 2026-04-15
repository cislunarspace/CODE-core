"""Fix MkDocs-generated directory hrefs for local file:// browsing.

MkDocs with use_directory_urls=true generates links like href="/path/"
which work on web servers but break when opening files directly from the
filesystem. This script replaces all directory-reference hrefs with the
actual index.html paths.
"""

import re
from pathlib import Path


def fix_html_file(filepath: Path) -> int:
    """Fix directory-reference hrefs in a single HTML file.

    Returns the number of replacements made.
    """
    content = filepath.read_text(encoding="utf-8")
    original = content

    # Match href="..." where the value ends with "/" and is a relative/root-relative path
    # Skip: external URLs (http://, https://), anchors (#), mailto:, javascript:, tel:
    def replace_href(match: re.Match) -> str:
        url = match.group(1)  # the URL value inside quotes

        # Skip non-path URLs
        if url.startswith(("#", "mailto:", "javascript:", "tel:", "data:")):
            return match.group(0)
        if url.startswith(("http://", "https://", "//")):
            return match.group(0)
        # Skip root-only href="/"
        if url == "/":
            return match.group(0)

        # Only fix URLs that end with "/"
        if url.endswith("/"):
            new_url = url + "index.html"
            return match.group(0).replace(url, new_url)

        return match.group(0)

    # Match href="..." — capture the URL value (group 1)
    content = re.sub(
        r"""href=["']([^"']*?)["']""",
        replace_href,
        content,
    )

    if content != original:
        filepath.write_text(content, encoding="utf-8")
        return content.count("index.html") - original.count("index.html")

    return 0


def main() -> None:
    site_dir = Path(__file__).resolve().parent.parent / "site"

    if not site_dir.exists():
        print(f"Error: {site_dir} does not exist. Run 'mkdocs build' first.")
        return

    html_files = list(site_dir.rglob("*.html"))
    total_fixes = 0

    for filepath in sorted(html_files):
        fixes = fix_html_file(filepath)
        if fixes > 0:
            rel = filepath.relative_to(site_dir)
            print(f"  Fixed {fixes} links in {rel}")
            total_fixes += fixes

    print(f"\nDone: {total_fixes} hrefs fixed across {len(html_files)} files.")


if __name__ == "__main__":
    main()
