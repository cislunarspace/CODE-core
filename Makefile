# e2m2e 开发入口（唯一）。
#
# 源码开发只需 `make dev`：同步 Python 依赖（--no-install-project，不构建扩展）、
# 拉取 CSPICE 编译包与 SPICE 内核（幂等）、maturin develop 构建安装 Rust 扩展。
# 切勿裸跑 `uv sync`：e2m2e 在 uv.lock 中是 editable 包，uv sync 会当场以 maturin
# 构建扩展（需要 CSPICE_DIR），且与 make dev 形成重复构建（issue #478）。

# ---------- Python / uv ----------

# Windows 官方 Python 只装 python.exe（无 python3 命令），Linux 惯例为 python3；
# 均可用命令行 `make PYTHON=...` 覆盖。
ifeq ($(OS),Windows_NT)
PYTHON ?= python
else
PYTHON ?= python3
endif
# --no-sync：uv run 默认先同步环境（触发 editable 构建），与 maturin develop 重复；
# 依赖由 make dev 显式同步，此处一律跳过（与 CI 的 uv run --no-sync 模式一致）。
# 注：mypy/pytest 一律经 `python -m` 调用——Windows 上 uv 的 console-script 垫片
# 可能报 "trampoline failed to canonicalize script path"；ruff 为原生 exe 不受影响。
UV     := uv run --no-sync

# ---------- 环境导出：CSPICE_DIR / LIBCLANG_PATH（cspice-sys 构建依赖） ----------

# CSPICE_DIR：解析编译包目录（未缓存时自动下载）。
ifeq ($(OS),Windows_NT)
CSPICE_DIR := $(CURDIR)/.cspice/mice_windows
else
CSPICE_DIR := $(shell $(PYTHON) scripts/download_cspice.py --print-cspice-dir 2>/dev/null)
endif
export CSPICE_DIR

# LIBCLANG_PATH：bindgen 需要。Windows 取 LLVM 官方安装器默认路径（可覆盖），
# Linux 由 llvm-config / 文件系统探测。
ifeq ($(OS),Windows_NT)
LIBCLANG_PATH ?= C:/Program Files/LLVM/bin
else
LLVM_CONFIG   := $(shell command -v llvm-config 2>/dev/null || ls /usr/bin/llvm-config-* 2>/dev/null | head -1)
LIBCLANG_PATH := $(or $(shell $(LLVM_CONFIG) --libdir 2>/dev/null),$(shell find /usr/lib -maxdepth 4 -name 'libclang*.so*' 2>/dev/null | head -1 | xargs dirname 2>/dev/null),$(LIBCLANG_PATH))
endif
export LIBCLANG_PATH

# ---------- 测试参数与共用命令 ----------

PYTEST_WORKERS ?= auto
PYTEST_DIST ?= loadscope
# dev / dev-release 共用的依赖同步（--no-install-project 理由见文件头，issue #478）。
DEV_SYNC := uv sync --group dev --no-install-project

.DEFAULT_GOAL := help

.PHONY: help setup cspice kernels dev dev-release test test-rust test-python docs check fmt clean clean-tests

help:  ## 显示本帮助
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

cspice:  ## 下载 CSPICE 编译包
	$(PYTHON) scripts/download_cspice.py

kernels:  ## 下载 SPICE 内核
	$(PYTHON) scripts/download_kernels.py

setup: cspice kernels  ## 首次拉取：CSPICE 编译包 + SPICE 内核（kernels/）

catalog-baseline:  ## 重新生成随包分发的 CR3BP 基线轨道族数据集（ADR 0036）
	$(PYTHON) scripts/generate_catalog_baseline.py

dev: setup  ## 唯一开发入口：同步依赖 + 拉数据 + 构建安装 Rust 扩展（debug）
	$(DEV_SYNC)
	$(UV) maturin develop

dev-release: setup  ## 同 dev，以 --release 构建（性能基准 / 长期预报用）
	$(DEV_SYNC)
	$(UV) maturin develop --release

test: test-rust test-python  ## 全量测试（Rust 工作区 + Python）

# cargo test 不启用 extension-module，测试 EXE 需运行时加载 python3.dll
#（issue #495）；虚拟环境 Scripts/ 没有该 DLL，必须由 Python 安装根目录提供。
# 探测放在 recipe 中执行：仅在跑 Rust 测试时求值，失败即终止，不影响其他目标。
# Windows 分支按 POSIX sh 编写（Git Bash/Scoop make）；PATH 采用 Windows 原生
# 路径与分号，MSYS 会在启动原生 cargo 前转换为 Windows loader 可识别的形式。
ifeq ($(OS),Windows_NT)
TEST_RUST = set -e; \
	PYTHON_DLL_DIR="$$($(PYTHON) scripts/python_dll_dir.py $(if $(PYTHON_DLL_DIR),--override '$(PYTHON_DLL_DIR)'))"; \
	export PATH="$$PYTHON_DLL_DIR;$$PATH"; \
	cargo test --workspace -- --test-threads=1
else
TEST_RUST = cargo test --workspace -- --test-threads=1
endif

test-rust:  ## Rust 工作区测试（spice 默认；串行）
	$(TEST_RUST)

test-python:  ## Python 测试（默认 xdist 并行；含 spice-gated，需先 make setup 拉内核）
	$(UV) python -m pytest tests/ -n $(PYTEST_WORKERS) --dist $(PYTEST_DIST)

docs:  ## 构建 Sphinx 文档到 docs/_build/html（独立 docs 依赖组；需先 make dev）
	uv sync --group docs --no-install-project
	$(UV) sphinx-build -b html docs docs/_build/html

# 命令集须与 ci.yml 保持对齐（CI 不调用 make）；改动任一边时同步另一边。
check:  ## 格式 + lint + 类型/层级检查（Rust + Python，与 ci.yml 对齐）
	cargo fmt --all -- --check
	cargo clippy --workspace -- -D warnings
	$(UV) ruff check .
	$(UV) ruff format --check .
	$(UV) python scripts/check_layer_imports.py
	$(UV) python scripts/check_deleted_dir_refs.py
	$(UV) python -m mypy e2m2e/ --ignore-missing-imports

fmt:  ## 就地格式化（Rust + Python）
	cargo fmt --all
	$(UV) ruff format .
	$(UV) ruff check --fix .

clean:  ## 清理 Rust 构建产物（保留 .cspice / kernels 缓存）
	cargo clean

clean-tests:  ## 清理 tests/ 幽灵目录与 __pycache__（#603；配套门禁在 tests/_meta）
	$(PYTHON) scripts/clean_test_residue.py --fix
