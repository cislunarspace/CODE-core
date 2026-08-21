# Issue #495：Windows Rust 测试的 DLL 加载失败调研

## 结论

Issue #495 报告的 `0xc0000135` 是 Windows NTSTATUS `STATUS_DLL_NOT_FOUND`。它只能证明测试进程启动时的 DLL 加载链失败，不能从错误码本身判断缺少的是 Python、CSPICE、LLVM 还是 MSVC 运行库。Win32 的 `ERROR_MOD_NOT_FOUND` 是另一个数值 `126 (0x7e)`，不要将两者混写。

结合当前仓库的构建配置，本机诊断已确认缺失项是 Python DLL：测试 EXE 的直接导入表包含 `python3.dll`，默认 PATH 找不到它；把当前解释器的 base prefix 加入 PATH 后，同一测试 EXE 与 `cargo test -p e2m2e-integrators --lib -- normal_form nsga2` 均成功运行。Issue 中把 `.venv/Scripts` 加入 `PATH` 不能解决该问题，因为 Python DLL 位于 Python 安装根目录。CSPICE Windows 包中的 `cspice.lib` 是静态库，`CSPICE_DIR` 主要用于编译期链接和 bindgen；不能把 `lib/` 目录当成 DLL 目录来解决启动期加载问题。

修复 `make test-rust` 前已经取得 PE 依赖清单和 loader 行为证据；其他 Windows 环境若再出现 `0xc0000135`，仍应按下面的诊断流程确认实际缺失 DLL。

## 已确认事实

- Issue 复现的是 `cargo test -p e2m2e-integrators --lib -- normal_form nsga2`：测试二进制编译成功，启动时报 `0xc0000135`；同环境 `cargo check` 通过。
- 当前 `Makefile` 的 `test-rust` 直接执行 `cargo test --workspace -- --test-threads=1`，没有 Windows 专用的运行时 DLL 配置。
- 当前 CI 的 `ci.yml` 只有 Ubuntu lint/typecheck，没有 Windows Rust 测试 job。
- `Cargo.toml` 的 PyO3 workspace feature 只有 `abi3-py310`；`extension-module` 仅由 maturin 构建路径显式启用。`cargo test` 不应使用 `extension-module`，而是由 PyO3 配置链接 Python。
- `cspice-sys` 从 `CSPICE_DIR/lib` 添加 native link search，并链接静态 `cspice` 库。静态库本身不是一个运行时 DLL。
- 历史提交中对 `/NODEFAULTLIB:LIBCMT` 的修改处理的是 Windows 静态 CSPICE 与 PyO3 的 CRT 混链风险，属于链接期问题，不等同于本 issue 的启动期 `STATUS_DLL_NOT_FOUND`。

## 必须先完成的诊断

在 Windows x64 Native Tools Command Prompt 或等价环境中执行：

```powershell
$env:CSPICE_DIR = "$pwd/.cspice/mice_windows"
cargo test -p e2m2e-integrators --lib --no-run --message-format=json
```

从 JSON 输出的 `executable` 字段取得测试 EXE，然后检查其直接依赖：

```powershell
dumpbin /DEPENDENTS path\to\e2m2e_integrators-<hash>.exe
dumpbin /IMPORTS path\to\e2m2e_integrators-<hash>.exe
```

对其中的非系统 DLL 继续执行 `/DEPENDENTS`。同时记录 Python 实际安装位置和 DLL 名称：

```powershell
python -c "import sys,sysconfig; print(sys.executable); print(sys.base_prefix); print(sysconfig.get_config_var('prefix')); print(sysconfig.get_config_var('LIBDIR')); print(sysconfig.get_config_var('LDLIBRARY'))"
where.exe python
Get-ChildItem (python -c "import sys;print(sys.base_prefix)") -Filter '*.dll'
Get-ChildItem .venv/Scripts -Recurse -Filter 'python*.dll'
```

若 `/DEPENDENTS` 仍不能确定具体缺失项，使用 Microsoft Process Monitor 过滤失败 EXE 的 `CreateFile` 操作，查找 DLL 路径返回 `NAME NOT FOUND` 的记录。也可使用 Dependencies 等 PE 依赖查看工具。

还应记录以下信息，以排除环境误判：

```powershell
where.exe dumpbin
where.exe link
where.exe clang
rustc -vV
cargo tree -e features -p e2m2e-integrators
dumpbin /DIRECTIVES .cspice\mice_windows\lib\cspice.lib | findstr /i "DEFAULTLIB LIBCMT MSVCRT"
```

## 方案取舍

### 1. 仅补充文档

明确 Windows 上的 CSPICE、LLVM、Python base prefix 准备步骤，并给出上述依赖诊断命令。风险最低，但每台机器仍需手工设置 PATH，不能保证新贡献者直接成功。

### 2. 修复 Makefile 入口

若诊断确认缺失的是 Python DLL，可让 Windows `test-rust` 自动从当前 Python 解释器计算 base prefix，并把该目录加入当前命令的 PATH；同时保留 `PYTHON_DLL_DIR` 作为显式覆盖。不要把 CSPICE 的 `.lib` 目录加入 DLL PATH。该方案直接修复仓库规定的 `make test-rust` 入口，但必须覆盖 Git Bash、PowerShell、多个 Python 安装和路径含空格等情况。

### 3. 增加 Windows CI 测试

在 Windows runner 上固定 Rust 1.98/MSVC、Python、LLVM，下载并解压 `cspice-windows.zip`，设置 `CSPICE_DIR`，先运行 `e2m2e-integrators` 的最小测试，再决定是否扩大到整个 workspace。它能真实暴露启动链问题并填补 PR 覆盖空档，但需要维护依赖准备、缓存和运行时间。CI 不能替代本地文档或 Makefile 入口。

推荐顺序是：先完成 PE/loader 诊断；确认缺失 DLL 后，优先修复 Makefile 并补文档，再考虑增加 Windows CI 回归任务。

## 来源

一手资料：

- GitHub issue #495：<https://github.com/cislunarspace/e2m2e/issues/495>
- Microsoft 系统错误码：<https://learn.microsoft.com/en-us/windows/win32/debug/system-error-codes--0-499->
- Microsoft DLL 搜索顺序：<https://learn.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-search-order>
- Microsoft DUMPBIN `/DEPENDENTS`：<https://learn.microsoft.com/en-us/cpp/build/reference/dependents?view=msvc-170>
- Microsoft DUMPBIN `/IMPORTS`：<https://learn.microsoft.com/en-us/cpp/build/reference/imports-dumpbin?view=msvc-170>
- Microsoft PE 格式与导入目录：<https://learn.microsoft.com/en-us/windows/win32/debug/pe-format>
- PyO3 0.24 构建与发布指南：<https://pyo3.rs/v0.24.0/building-and-distribution.html>
- PyO3 0.24 构建配置源码：<https://github.com/PyO3/pyo3/tree/v0.24.0/pyo3-build-config>
- cspice-rs `cspice-sys` 构建源码：<https://github.com/jacob-pro/cspice-rs/blob/master/cspice-sys/build.rs>

仓库证据：

- `Makefile`：Windows 环境变量与 `test-rust` 目标
- `Cargo.toml`：PyO3 workspace features 与 CSPICE 依赖
- `crates/e2m2e-integrators/Cargo.toml`：`extension-module` 与 `spice` feature
- `pyproject.toml`：maturin 专用 features
- `.github/workflows/ci.yml`：当前 CI job 范围
- `.cargo/config.toml`：Windows CRT 链接参数
