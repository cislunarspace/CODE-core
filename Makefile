# e2m2e 开发入口（唯一）。
#
# spice 为必选 feature（crates/*/Cargo.toml default=["spice"]），普通 cargo/maturin
# 构建都依赖 CSPICE。CSPICE_DIR 由 scripts/download_cspice.py 解析缓存目录后导出，
# cspice-sys 无 CSPICE_DIR 时直接构建报错（不启用 downloadcspice，杜绝走 NAIF 官网
# 下载，国内网络常不可达）。故裸 cargo / maturin 命令需经本 Makefile，或自行
# `export CSPICE_DIR=$(python3 scripts/download_cspice.py --print-cspice-dir)`。

PYTHON := python3
UV     := uv run
PYTEST_WORKERS ?= auto
PYTEST_DIST ?= loadscope

# 首次即自动落盘并解析路径：已缓存时 download_cspice.py 仅 stat（近乎零开销）。
CSPICE_DIR := $(shell $(PYTHON) scripts/download_cspice.py --print-cspice-dir 2>/dev/null)
export CSPICE_DIR

# libclang（cspice-sys 的 bindgen 需要）：用 := 覆盖环境里可能失效的旧值
# （命令行 make LIBCLANG_PATH=... 仍可覆盖）。优先 llvm-config --libdir（版本无关，
# 其次版本化 llvm-config-*），回退 /usr/lib 探测。
LLVM_CONFIG   := $(shell command -v llvm-config 2>/dev/null || ls /usr/bin/llvm-config-* 2>/dev/null | head -1)
LIBCLANG_PATH := $(or $(LIBCLANG_PATH),$(shell $(LLVM_CONFIG) --libdir 2>/dev/null),$(shell find /usr/lib -maxdepth 4 -name 'libclang*.so*' 2>/dev/null | head -1 | xargs dirname 2>/dev/null))
export LIBCLANG_PATH

.DEFAULT_GOAL := help

.PHONY: help setup dev dev-release test test-rust test-python check fmt clean

help:  ## 显示本帮助
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup:  ## 首次拉取：CSPICE 编译包 + SPICE 内核（kernels/）
	$(PYTHON) scripts/download_cspice.py
	$(PYTHON) scripts/download_kernels.py

dev:  ## 构建并安装开发版扩展（maturin develop，spice 默认）
	$(UV) maturin develop

dev-release:  ## 同 dev 但 --release（性能基准 / 长期预报用）
	$(UV) maturin develop --release

test: test-rust test-python  ## 全量测试（Rust 工作区 + Python）

# --test-threads=1：CSPICE 有全局线程锁，forces/integrators 的 spice-gated
# 测试串行执行避免竞争；非 spice 测试一并串行（略慢但稳妥）。
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
