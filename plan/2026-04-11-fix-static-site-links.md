# Plan: Fix Static Site Hyperlinks for Local File Browsing

## Problem

The MkDocs-generated `site/` directory contains ~1,666 hyperlinks across 41 HTML files that reference directories with trailing `/` (e.g., `href="/algorithms/halo/"`) instead of the actual files (`href="/algorithms/halo/index.html"`). These work on web servers (which auto-resolve directory requests to `index.html`) but break when opening files directly from the filesystem (`file://` protocol).

## Root Cause

MkDocs defaults to `use_directory_urls: true`, generating clean URLs like `/path/`. No MkDocs config option produces `/path/index.html` instead.

## Solution

Write a post-build Python script that batch-fixes all directory-reference hrefs in the generated HTML.

## Implementation Steps

### Phase 1: Create the fix script

Create `scripts/fix_local_links.py`:
- Find all `.html` files under `site/`
- For each file, parse HTML and find all `href` attributes ending with `/`
- Replace trailing `/` with `/index.html`
- Skip: `href="/"`, external URLs, anchor links (`#`), `mailto:`, `javascript:`
- Save the modified files in-place

### Phase 2: Add build integration

Add a convenience script or update existing docs build process:
- Run `mkdocs build && python scripts/fix_local_links.py`
- Or add to `pyproject.toml` as a script entry point

### Phase 3: Run and verify

- Run the script against the existing `site/` directory
- Spot-check a few links manually
- Verify no false positives (external links, anchors, etc.)

## Risk Assessment

| Risk | Level | Mitigation |
|------|-------|------------|
| Breaking external links | LOW | Script only modifies hrefs ending with `/` that are relative or root-relative paths |
| Missing some patterns | LOW | Regex covers all observed patterns: `/path/`, `../path/`, `./path/` |
| Re-run needed after each build | MEDIUM | Document in workflow; integrate into build command |

## Complexity: LOW

Single Python script (~40 lines), no architectural changes.
