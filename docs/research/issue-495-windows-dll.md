# Issue #495: Research on Windows Rust test DLL load failures

## Conclusion

The `0xc0000135` reported by issue #495 is Windows NTSTATUS
`STATUS_DLL_NOT_FOUND`. It proves only that the DLL loading chain failed at
test-process startup; from the error code alone you cannot tell whether the
missing piece is Python, CSPICE, LLVM, or an MSVC runtime library. Win32's
`ERROR_MOD_NOT_FOUND` is a different number, `126 (0x7e)` — never conflate the
two.

Combined with this repo's build configuration, local diagnosis confirmed the
missing item is the Python DLL: the test EXE's direct import table contains
`python3.dll`, absent from default PATH; after adding the current interpreter's
base prefix to PATH, both the same test EXE and
`cargo test -p e2m2e-integrators --lib -- normal_form nsga2` ran successfully.
Adding `.venv/Scripts` to `PATH` as suggested in the issue cannot fix this,
because the Python DLL lives in the Python installation root directory.
CSPICE's Windows-package `cspice.lib` is a static library; `CSPICE_DIR`
primarily serves compile-time linking and bindgen — you cannot treat its `lib/`
directory as a DLL directory to solve startup-time loading.

These judgments rest on PE dependency listings and loader-behavior evidence;
`make test-rust` is fixed accordingly (see Option 2). Should `0xc0000135`
reappear in other Windows environments, still walk the diagnostic procedure
below to identify the actually missing DLL.

## Confirmed facts

- The issue reproduces `cargo test -p e2m2e-integrators --lib -- normal_form
  nsga2`: the test binary compiles but fails at startup with `0xc0000135`;
  same-environment `cargo check` passes.
- Current `Makefile`'s `test-rust` on Windows first runs
  `scripts/python_dll_dir.py` probing the DLL directory from the current Python
  interpreter (`PYTHON_DLL_DIR` explicit override supported) into PATH, then runs
  `cargo test --workspace -- --test-threads=1`; other platforms run cargo test
  directly.
- Current CI's `ci.yml` holds only Ubuntu lint/typecheck — no Windows Rust test
  job.
- Root `Cargo.toml`'s PyO3 workspace feature carries only `abi3-py310`;
  `extension-module` gets enabled explicitly only by maturin build paths.
  `cargo test` should not use `extension-module`; PyO3 config links Python
  instead.
- `cspice-sys` adds native link search from `CSPICE_DIR/lib` and links static
  `cspice`. A static library is not a runtime DLL.
- Historical commits touching `/NODEFAULTLIB:LIBCMT` addressed CRT mixed-linking
  risk between Windows static CSPICE and PyO3 — a link-time issue, not
  equivalent to this issue's startup-time `STATUS_DLL_NOT_FOUND`.

## Diagnostics to complete first

In a Windows x64 Native Tools Command Prompt or equivalent:

```powershell
$env:CSPICE_DIR = "$pwd/.cspice/mice_windows"
cargo test -p e2m2e-integrators --lib --no-run --message-format=json
```

Take the test EXE path from JSON's `executable` field, then inspect direct
dependencies:

```powershell
dumpbin /DEPENDENTS path\to\e2m2e_integrators-<hash>.exe
dumpbin /IMPORTS path\to\e2m2e_integrators-<hash>.exe
```

Recursively `/DEPENDENTS` any non-system DLLs found. Also record the actual
Python installation location and DLL names:

```powershell
python -c "import sys,sysconfig; print(sys.executable); print(sys.base_prefix); print(sysconfig.get_config_var('prefix')); print(sysconfig.get_config_var('LIBDIR')); print(sysconfig.get_config_var('LDLIBRARY'))"
where.exe python
Get-ChildItem (python -c "import sys;print(sys.base_prefix)") -Filter '*.dll'
Get-ChildItem .venv/Scripts -Recurse -Filter 'python*.dll'
```

If `/DEPENDENTS` still can't pin down the missing item, use Microsoft Process
Monitor filtering the failing EXE's `CreateFile` operations for DLL paths
returning `NAME NOT FOUND`. PE-dependency viewers like Dependencies also work.

Record further environment details to rule out misjudgment:

```powershell
where.exe dumpbin
where.exe link
where.exe clang
rustc -vV
cargo tree -e features -p e2m2e-integrators
dumpbin /DIRECTIVES .cspice\mice_windows\lib\cspice.lib | findstr /i "DEFAULTLIB LIBCMT MSVCRT"
```

## Options considered

### Option 1: documentation only

Clarify Windows setup steps for CSPICE/LLVM/Python base prefix plus the above
dependency-diagnosis commands. Lowest risk, but every machine still needs manual
PATH configuration; new contributors get no success guarantee.

### Option 2: fix the Makefile entry

If diagnosis confirms the missing piece is the Python DLL, have Windows
`test-rust` compute the interpreter's base prefix automatically and add it to the
current command's PATH, keeping `PYTHON_DLL_DIR` as explicit override. Never put
CSPICE's `.lib` directory onto the DLL PATH. This directly fixes the repo's
designated `make test-rust` entry, but must survive Git Bash, PowerShell,
multiple Python installs, and spaces-in-paths. **Landed** — current `Makefile` +
`scripts/python_dll_dir.py`.

### Option 3: add Windows CI testing

Pin Rust 1.98/MSVC, Python, LLVM on Windows runners; download & unpack
`cspice-windows.zip`; set `CSPICE_DIR`; run minimal e2m2e-integrators tests
first before widening to the workspace. Genuinely surfaces startup-chain problems
and fills PR coverage gaps — at the cost of maintaining dependency prep, caches,
and runtime budgets. CI cannot replace local docs or Makefile entries.

Recommended order: complete PE/loader diagnostics first; after confirming the
missing DLL, prefer fixing Makefile + documenting; consider Windows CI regression
afterward.

## Sources

Primary references:

- GitHub issue #495: <https://github.com/cislunarspace/e2m2e/issues/495>
- Microsoft system error codes: <https://learn.microsoft.com/en-us/windows/win32/debug/system-error-codes--0-499->
- Microsoft DLL search order: <https://learn.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-search-order>
- Microsoft DUMPBIN `/DEPENDENTS`: <https://learn.microsoft.com/en-us/cpp/build/reference/dependents?view=msvc-170>
- Microsoft DUMPBIN `/IMPORTS`: <https://learn.microsoft.com/en-us/cpp/build/reference/imports-dumpbin?view=msvc-170>
- Microsoft PE format & import directory: <https://learn.microsoft.com/en-us/windows/win32/debug/pe-format>
- PyO3 0.24 building & distribution guide: <https://pyo3.rs/v0.24.0/building-and-distribution.html>
- PyO3 0.24 build-config sources: <https://github.com/PyO3/pyo3/tree/v0.24.0/pyo3-build-config>
- cspice-rs `cspice-sys` build sources: <https://github.com/jacob-pro/cspice-rs/blob/master/cspice-sys/build.rs>

Repository evidence:

- `Makefile`: Windows env vars + `test-rust` target
- `scripts/python_dll_dir.py`: Windows Python-DLL-directory detection
- `Cargo.toml`: PyO3 workspace features + CSPICE dependencies
- `crates/e2m2e-integrators/Cargo.toml`: `extension-module` + `spice` features
- `pyproject.toml`: maturin-specific features
- `.github/workflows/ci.yml`: current CI job scope
- `.cargo/config.toml`: Windows CRT link arguments
