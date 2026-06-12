"""ICGEM .gfc 重力场文件解析。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

_DEFAULT_MU = 398600.4415  # km^3/s^2
_DEFAULT_RADIUS = 6378.1363  # km


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


def _normalize_coefficient(
    c: float, n: int, m: int
) -> float:
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
    for (n, m), (c_val, s_val) in coefficients.items():
        if n <= degree:
            C[n, m] = c_val
            S[n, m] = s_val

    return GravityFileData(
        model_name=model_name,
        mu=mu,
        radius=radius,
        max_degree=max_degree,
        normalized=True,
        C=C,
        S=S,
    )
