---
title: Release Guide
---

# Release Guide

This document describes how to publish e2m2e to PyPI.

## Prerequisites

### 1. Install Build Tools

```bash
pip install build twine
```

### 2. Obtain a PyPI API Token

- **TestPyPI**: Go to [https://test.pypi.org](https://test.pypi.org) to register an account and generate an API Token
- **Production PyPI**: Go to [https://pypi.org](https://pypi.org) to register an account and generate an API Token

Configure the token:

```bash
# ~/.pypirc file contents
[pypi]
username = __token__
password = pypi-xxxxxxxxxxxx

[testpypi]
username = __token__
password = pypi-xxxxxxxxxxxx
```

## Release Process

### 1. Confirm Version Number

Update the version number in `pyproject.toml` (must be higher than the previous version):

```toml
[project]
version = "3.1.12"
```

### 2. Commit a Git Snapshot

```bash
cd /path/to/e2m2e
git add .
git commit -m "describe changes"
git tag v3.1.12
git push origin master --tags
```

### 3. Build Distribution Packages

```bash
rm -rf dist/ build/ *.egg-info/
python -m build
```

### 4. Upload to TestPyPI (Recommended for Pre-Release Testing)

```bash
twine upload --repository testpypi dist/*
```

Verify the installation:

```bash
pip install --index-url https://test.pypi.org/simple/ e2m2e==3.1.12
```

### 5. Upload to Production PyPI

```bash
twine upload --repository pypi dist/*
```

## Common Commands Summary

```bash
# Full release workflow
rm -rf dist/ build/ *.egg-info/
python -m build
twine upload --repository testpypi dist/*    # Test environment
twine upload --repository pypi dist/*         # Production environment

# Install production version only
pip install e2m2e

# Install a specific version
pip install e2m2e==3.1.12

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ e2m2e==3.2.0
```

## Notes

1. **Version numbers must be strictly increasing**: PyPI does not allow uploading packages with the same or lower version number
2. **Test on TestPyPI first**: Always verify the package works correctly on TestPyPI before each release
3. **Maintain consistency**: The Git tag and the version in `pyproject.toml` should always match
4. **License field**: Ensure `license = {text = "Apache-2.0"}` in `pyproject.toml` is consistent with the LICENSE file
