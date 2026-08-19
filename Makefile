# e2m2e 开发入口（唯一）。
#
# 源码开发只需 `make dev`：同步 Python 依赖（--no-install-project，不构建扩展）、
# 拉取 CSPICE 编译包与 SPICE 内核（幂等）、maturin develop 构建安装 Rust 扩展。
# 切勿裸跑 `uv sync`：e2m2e 在 uv.lock 中是 editable 包，uv sync 会当场以 maturin
# 构建扩展（需要 CSPICE_DIR），且与 make dev 形成重复构建（issue #478）。

PYTHON := python3
# --no-sync：uv run 默认先同步环境（触发 editable 构建），与 maturin develop 重复。
# 依赖由 make dev 显式同步，此处一律跳过（与 CI 的 uv run --no-sync 模式一致）。
UV     := uv run --no-sync
PYTEST_WORKERS ?= auto
PYTEST_DIST ?= loadscope

# 解析并导出 CSPICE_DIR（未缓存时自动下载）。
CSPICE_DIR := $(shell $(PYTHON) scripts/download_cspice.py --print-cspice-dir 2>/dev/null)
export CSPICE_DIR

# 探测并导出 LIBCLANG_PATH（cspice-sys 的 bindgen 需要）。
LLVM_CONFIG   := $(shell command -v llvm-config 2>/dev/null || ls /usr/bin/llvm-config-* 2>/dev/null | head -1)
LIBCLANG_PATH := $(or $(shell $(LLVM_CONFIG) --libdir 2>/dev/null),$(shell find /usr/lib -maxdepth 4 -name 'libclang*.so*' 2>/dev/null | head -1 | xargs dirname 2>/dev/null),$(LIBCLANG_PATH))
export LIBCLANG_PATH

.DEFAULT_GOAL := help

.PHONY: help setup dev dev-release test test-rust test-python check fmt clean

help:  ## 显示本帮助
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup:  ## 只拉数据不构建：CSPICE 编译包 + SPICE 内核（kernels/），幂等
	$(PYTHON) scripts/download_cspice.py
	$(PYTHON) scripts/download_kernels.py

dev: setup  ## 唯一开发入口：同步依赖 + 拉数据 + 构建安装 Rust 扩展（debug）
	uv sync --group dev --no-install-project
	$(UV) maturin develop

dev-release: setup  ## 同 dev，以 --release 构建（性能基准 / 长期预报用）
	uv sync --group dev --no-install-project
	$(UV) maturin develop --release

test: test-rust test-python  ## 全量测试（Rust 工作区 + Python）

test-rust:  ## Rust 工作区测试（spice 默认；串行）
	cargo test --workspace -- --test-threads=1

test-python:  ## Python 测试（默认 xdist 并行；含 spice-gated，需先 make setup 拉内核）
	$(UV) pytest tests/ -n $(PYTEST_WORKERS) --dist $(PYTEST_DIST)

check:  ## 格式 + lint（Rust + Python）
	cargo fmt --all -- --check
	cargo clippy --workspace -- -D warnings
	$(UV) ruff check .
	$(UV) ruff format --check .

fmt:  ## 就地格式化（Rust + Python）
	cargo fmt --all
	$(UV) ruff format .
	$(UV) ruff check --fix .

clean:  ## 清理 Rust 构建产物（保留 .cspice / kernels 缓存）
	cargo clean
