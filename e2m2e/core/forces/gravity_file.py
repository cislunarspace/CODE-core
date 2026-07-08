"""重力场文件解析。

支持两种格式:
- ICGEM ``.gfc``（``load_gfc_file``）。
- GMAT ``.cof``（``load_cof_file``），移植 GMAT ``HarmonicGravity.cpp`` 的
  ``LM_LoadCof`` 逻辑。

统一入口 ``load_gravity_file`` 按文件扩展名分发。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from e2m2e.core.constants import R_EARTH

_DEFAULT_MU = 398600.4415  # km^3/s^2
_DEFAULT_RADIUS = R_EARTH  # km
_SECONDS_PER_YEAR = 365.25 * 86400.0  # 儒略年,与 GMAT DAYS_PER_YEAR 一致


@dataclass(frozen=True)
class GravityFileData:
    """解析后的重力场文件数据。"""

    model_name: str
    mu: float
    radius: float
    max_degree: int
    normalized: bool
    C: npt.NDArray[np.floating]
    S: npt.NDArray[np.floating]
    dotC: npt.NDArray[np.floating]
    dotS: npt.NDArray[np.floating]


def _normalize_coefficient(c: float, n: int, m: int) -> float:
    """把非正规化系数转换为完全正规化系数。"""
    # C_nm^norm = C_nm * sqrt((2 - delta_m0) * (2n+1) * (n-m)! / (n+m)!)
    delta_m0 = 1.0 if m == 0 else 0.0
    factorial_ratio = 1.0
    for k in range(1, m + 1):
        factorial_ratio *= (n - k + 1) / (n + k)
    scale = np.sqrt((2.0 - delta_m0) * (2.0 * n + 1.0) * factorial_ratio)
    return c * float(scale)


def load_gfc_file(
    path: str | Path,
    *,
    requested_degree: int | None = None,
    default_mu: float = _DEFAULT_MU,
    default_radius: float = _DEFAULT_RADIUS,
) -> GravityFileData:
    """加载 ICGEM .gfc 格式重力场文件。

    Args:
        path: 文件路径。
        requested_degree: 请求的最大 degree，用于校验。
        default_mu: 文件头缺失 GM 时的默认值，单位 km^3/s^2。
        default_radius: 文件头缺失参考半径时的默认值，单位 km。

    Returns:
        解析后的重力场数据。
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    model_name = "unknown"
    mu = float(default_mu)
    radius = float(default_radius)
    max_degree: int | None = None
    normalized = True

    coefficients: dict[tuple[int, int], tuple[float, float]] = {}
    dot_coefficients: dict[tuple[int, int], tuple[float, float]] = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("/*") or line.upper() == "END":
            continue

        parts = line.split()
        if parts[0].lower() == "modelname":
            model_name = " ".join(parts[1:])
            continue
        if parts[0].lower() == "earth_gravity_constant":
            mu = float(parts[1])
            continue
        if parts[0].lower() == "radius":
            radius = float(parts[1])
            continue
        if parts[0].lower() == "max_degree":
            max_degree = int(parts[1])
            continue
        if parts[0].lower() == "norm":
            norm_value = " ".join(parts[1:]).lower()
            if norm_value == "fully_normalized":
                normalized = True
            elif norm_value in {"unnormalized", "not_normalized"}:
                normalized = False
            else:
                raise ValueError(f"Unknown normalization in .gfc file: {norm_value}")
            continue

        if parts[0].lower() == "gfc":
            if len(parts) < 5:
                raise ValueError(f"Invalid coefficient line: {line}")
            n = int(parts[1])
            m = int(parts[2])
            c_val = float(parts[3])
            s_val = float(parts[4])
            if n < 0 or m < 0 or m > n:
                raise ValueError(f"Invalid degree/order: n={n}, m={m}")
            if (n, m) in coefficients:
                raise ValueError(f"Duplicate coefficient for degree={n}, order={m}")
            if not normalized:
                c_val = _normalize_coefficient(c_val, n, m)
                s_val = _normalize_coefficient(s_val, n, m)
            coefficients[(n, m)] = (c_val, s_val)
            continue

        if parts[0].lower() == "dot":
            if len(parts) < 5:
                raise ValueError(f"Invalid dot line: {line}")
            n = int(parts[1])
            m = int(parts[2])
            dc_val = float(parts[3])
            ds_val = float(parts[4])
            if n < 0 or m < 0 or m > n:
                raise ValueError(f"Invalid degree/order: n={n}, m={m}")
            if (n, m) in dot_coefficients:
                raise ValueError(f"Duplicate dot coefficient for degree={n}, order={m}")
            if not normalized:
                dc_val = _normalize_coefficient(dc_val, n, m)
                ds_val = _normalize_coefficient(ds_val, n, m)
            dot_coefficients[(n, m)] = (dc_val, ds_val)
            continue

    if mu <= 0 or radius <= 0:
        raise ValueError("GM and radius must be positive")

    # C00 is conventionally 1.0 if not present
    if (0, 0) not in coefficients:
        coefficients[(0, 0)] = (1.0, 0.0)

    actual_max_degree = max(n for n, _ in coefficients)
    if max_degree is None:
        max_degree = actual_max_degree
    elif actual_max_degree < max_degree:
        # Allow declared max_degree to exceed present coefficients; missing terms
        # are treated as zero (common for truncated distribution files).
        pass

    if requested_degree is not None and requested_degree > max_degree:
        raise ValueError(
            f"Requested degree {requested_degree} exceeds file max_degree {max_degree}"
        )

    degree = requested_degree if requested_degree is not None else max_degree
    size = degree + 1
    C = np.zeros((size, size), dtype=float)
    S = np.zeros((size, size), dtype=float)
    dotC = np.zeros((size, size), dtype=float)
    dotS = np.zeros((size, size), dtype=float)
    for (n, m), (c_val, s_val) in coefficients.items():
        if n <= degree:
            C[n, m] = c_val
            S[n, m] = s_val
    for (n, m), (dc_val, ds_val) in dot_coefficients.items():
        if n <= degree:
            dotC[n, m] = dc_val
            dotS[n, m] = ds_val

    return GravityFileData(
        model_name=model_name,
        mu=mu,
        radius=radius,
        max_degree=max_degree,
        normalized=True,
        C=C,
        S=S,
        dotC=dotC,
        dotS=dotS,
    )


# ----------------------------------------------------------------------------
# GMAT .cof 格式（移植自 GMAT HarmonicGravity.cpp 的 LM_LoadCof）
# ----------------------------------------------------------------------------


def load_cof_file(
    path: str | Path,
    *,
    requested_degree: int | None = None,
    default_mu: float = _DEFAULT_MU,
    default_radius: float = _DEFAULT_RADIUS,
) -> GravityFileData:
    """加载 GMAT ``.cof`` 格式重力场文件。

    解析逻辑移植自 GMAT ``HarmonicGravity.cpp`` 的 ``LM_LoadCof``。文件结构:

    - 头行 ``POTFIELD<NNN><MMM> <flag> <Mu> <RefRadius> <Normalized>``
      - ``NNN``/``MMM`` 各 3 字符，分别为文件中包含的最大 degree 与 order
        （如 ``POTFIELD360360``）。
      - ``Mu`` 单位 m³/s²×1e9，解析时除以 1e9 得到 km³/s²。
      - ``RefRadius`` 单位 m×1e3，解析时除以 1e3 得到 km。
      - ``Normalized`` 为 1.0 表示系数已完全正规化。
    - 系数行 ``RECOEF <n:3> <m:3> <Cnm:21> <Snm:21>``，按固定列宽解析
      （n=substr(8,3), m=substr(11,3), Cnm=substr(17,21), Snm=substr(38,21)）。
      m=0 时无 Snm 列。
    - 以 ``COMMENT`` 或 ``C `` 开头的行为注释，跳过。

    返回结构与 :func:`load_gfc_file` 完全一致。COF 文件不含 dot 项,
    故 ``dotC``/``dotS`` 全零;COF 通常省略 C₀₀,此处补 1.0。

    Args:
        path: 文件路径。
        requested_degree: 请求的最大 degree,用于截断读取。
        default_mu: 头行缺失 GM 时的默认值,单位 km^3/s^2。
        default_radius: 头行缺失参考半径时的默认值,单位 km。

    Returns:
        解析后的重力场数据。
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    model_name = path.stem
    mu = float(default_mu)
    radius = float(default_radius)
    max_degree: int | None = None
    normalized = True

    coefficients: dict[tuple[int, int], tuple[float, float]] = {}

    header_seen = False
    for raw_line in text.splitlines():
        line = raw_line.rstrip("\n")
        if not line.strip():
            continue
        if line.upper() == "END":
            continue

        upper = line.upper()
        # 注释行:以 "COMMENT" 或 "C "（C 后紧跟空格）开头。
        if upper.startswith("COMMENT") or upper.startswith("C "):
            continue

        if not header_seen:
            # 头行:前 8 字符为 "POTFIELD"，紧接 3 字符 degree 与 3 字符 order。
            if not upper.startswith("POTFIELD"):
                raise ValueError(
                    f"Expected POTFIELD header in .cof file, got: {raw_line!r}"
                )
            if len(line) < 14:
                raise ValueError(f"POTFIELD header too short: {raw_line!r}")
            # GMAT: NN=substr(8,3), MM=substr(11,3), 其余 substr(14) 按空格分隔。
            try:
                header_degree = int(line[8:11])
            except ValueError as exc:
                raise ValueError(
                    f"Cannot parse degree from POTFIELD header: {raw_line!r}"
                ) from exc
            parts = line[14:].split()
            if len(parts) < 4:
                raise ValueError(f"POTFIELD header missing fields: {raw_line!r}")
            # parts = [flag, Mu, RefRadius, Normalized, ...]
            # 单位:Mu 为 m^3/s^2 × 1e9 -> 除以 1e9 得 km^3/s^2;
            #       RefRadius 为 m × 1e3 -> 除以 1e3 得 km。
            mu = float(parts[1]) / 1e9
            radius = float(parts[2]) / 1e3
            normalized = float(parts[3]) > 0.0
            max_degree = header_degree
            header_seen = True
            continue

        if not upper.startswith("RECOEF"):
            # 未知行类型,跳过(与 GMAT 宽容策略一致)。
            continue

        # 固定列:n=substr(8,3), m=substr(11,3)。
        if len(line) < 14:
            raise ValueError(f"RECOEF line too short: {raw_line!r}")
        try:
            n = int(line[8:11])
            m = int(line[11:14])
        except ValueError as exc:
            raise ValueError(f"Cannot parse n/m in RECOEF line: {raw_line!r}") from exc

        # Cnm=substr(17,21)（固定列,C++ substr 越界会返回较短子串,
        # Python 切片同样安全）。字段不足时为错误。
        c_field = line[17:38].strip()
        if not c_field:
            raise ValueError(f"RECOEF line missing Cnm field: {raw_line!r}")
        try:
            c_val = float(c_field)
        except ValueError as exc:
            raise ValueError(f"Cannot parse Cnm {c_field!r}: {raw_line!r}") from exc

        # Snm=substr(38,21),m=0 时通常无此列;按 GMAT 仅在行足够长时读取。
        s_field = line[38:59].strip()
        if s_field:
            try:
                s_val = float(s_field)
            except ValueError as exc:
                raise ValueError(f"Cannot parse Snm {s_field!r}: {raw_line!r}") from exc
        else:
            s_val = 0.0

        if n < 0 or m < 0 or m > n:
            raise ValueError(f"Invalid degree/order: n={n}, m={m}")
        if (n, m) in coefficients:
            raise ValueError(f"Duplicate coefficient for degree={n}, order={m}")
        coefficients[(n, m)] = (c_val, s_val)

    if not header_seen:
        raise ValueError(f"No POTFIELD header found in .cof file: {path}")
    if mu <= 0 or radius <= 0:
        raise ValueError("GM and radius must be positive")

    # C00 约定为 1.0(COF 通常省略 n=0 行)。
    if (0, 0) not in coefficients:
        coefficients[(0, 0)] = (1.0, 0.0)

    actual_max_degree = max(n for n, _ in coefficients)
    if max_degree is None:
        max_degree = actual_max_degree
    elif actual_max_degree < max_degree:
        # 允许声明的 max_degree 超过实际系数(缺项按零处理)。
        pass

    if requested_degree is not None and requested_degree > max_degree:
        raise ValueError(
            f"Requested degree {requested_degree} exceeds file max_degree {max_degree}"
        )

    degree = requested_degree if requested_degree is not None else max_degree
    size = degree + 1
    C = np.zeros((size, size), dtype=float)
    S = np.zeros((size, size), dtype=float)
    dotC = np.zeros((size, size), dtype=float)
    dotS = np.zeros((size, size), dtype=float)
    for (n, m), (c_val, s_val) in coefficients.items():
        if n <= degree:
            C[n, m] = c_val
            S[n, m] = s_val

    return GravityFileData(
        model_name=model_name,
        mu=mu,
        radius=radius,
        max_degree=max_degree,
        normalized=normalized,
        C=C,
        S=S,
        dotC=dotC,
        dotS=dotS,
    )


def load_gravity_file(
    path: str | Path,
    *,
    requested_degree: int | None = None,
    default_mu: float = _DEFAULT_MU,
    default_radius: float = _DEFAULT_RADIUS,
) -> GravityFileData:
    """按文件扩展名分发到对应格式的解析器。

    - ``.gfc`` -> :func:`load_gfc_file`
    - ``.cof`` -> :func:`load_cof_file`

    其它扩展名抛 :class:`ValueError`。

    Args:
        path: 重力场文件路径。
        requested_degree: 请求的最大 degree。
        default_mu: 头部缺失 GM 时的默认值,单位 km^3/s^2。
        default_radius: 头部缺失参考半径时的默认值,单位 km。

    Returns:
        解析后的重力场数据。
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".gfc":
        return load_gfc_file(
            path,
            requested_degree=requested_degree,
            default_mu=default_mu,
            default_radius=default_radius,
        )
    if suffix == ".cof":
        return load_cof_file(
            path,
            requested_degree=requested_degree,
            default_mu=default_mu,
            default_radius=default_radius,
        )
    raise ValueError(
        f"Unsupported gravity file extension {suffix!r} (expected .gfc or .cof): {path}"
    )


def extrapolate_coefficients(
    C: npt.NDArray[np.floating],
    S: npt.NDArray[np.floating],
    dotC: npt.NDArray[np.floating],
    dotS: npt.NDArray[np.floating],
    t: float,
    t0: float,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """按 dot 项(系数长期变化率)外推球谐系数到历元 t。

    dot 单位为 1/年(ICGEM .gfc 标准,与 GMAT DAYS_PER_YEAR 一致);
    t 与 t0 为 SPICE et 秒,差值转换为儒略年。

    Args:
        C, S: 参考历元 t0 的正规化球谐系数。
        dotC, dotS: 系数长期变化率(1/年),形状与 C/S 一致。
        t: 目标历元(SPICE et 秒)。
        t0: 参考历元(SPICE et 秒)。

    Returns:
        外推后的 (C_out, S_out),新数组(不修改输入)。
    """
    dt_years = (t - t0) / _SECONDS_PER_YEAR
    return C + dotC * dt_years, S + dotS * dt_years
