from __future__ import annotations

import os
from typing import Dict

import numpy as np
import numpy.typing as npt
import spiceypy


_GM_VALUES: Dict[str, float] = {
    "EARTH": 398600.436,
    "MOON": 4902.8,
    "SUN": 1.32712440018e11,
    "EMB": 398600.436,
}

_NAIF_IDS: Dict[str, int] = {
    "EARTH": 399,
    "MOON": 301,
    "SUN": 10,
    "EMB": 3,
}


_LEAPSECOND_SEARCH_PATHS = [
    os.path.join(os.path.dirname(__file__), "..", "..", "kernels"),
    os.environ.get("SPICE_KERNEL_DIR", ""),
]


def _find_leapseconds_kernel():
    for search_dir in _LEAPSECOND_SEARCH_PATHS:
        if not search_dir or not os.path.isdir(search_dir):
            continue
        for root, dirs, files in os.walk(search_dir):
            for f in files:
                if f.endswith(".tls"):
                    return os.path.join(root, f)
    return None


class SPICEManager:
    def __init__(self) -> None:
        self._leapseconds_loaded = False

    def _ensure_leapseconds(self):
        if self._leapseconds_loaded:
            return
        path = _find_leapseconds_kernel()
        if path:
            spiceypy.furnsh(path)
            self._leapseconds_loaded = True

    def load_kernel(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Kernel file not found: {path}")
        self._ensure_leapseconds()
        spiceypy.furnsh(path)

    def unload_kernel(self, path: str) -> None:
        spiceypy.unload(path)

    def utc_to_et(self, utc_str: str) -> float:
        return float(spiceypy.str2et(utc_str))

    def et_to_utc(self, et: float) -> str:
        return spiceypy.et2utc(et, "ISOC", 0)

    def get_body_state(
        self, target: str, et: float, frame: str, observer: str
    ) -> npt.NDArray[np.floating]:
        state, _lt = spiceypy.spkezr(target, et, frame, "NONE", observer)
        return np.array(state)

    def get_body_position(
        self, target: str, et: float, frame: str, observer: str
    ) -> npt.NDArray[np.floating]:
        position, _lt = spiceypy.spkpos(target, et, frame, "NONE", observer)
        return np.array(position)

    def get_gm(self, body: str) -> float:
        name_upper = body.upper()
        if name_upper in _GM_VALUES:
            return _GM_VALUES[name_upper]
        body_id = _NAIF_IDS.get(name_upper, body)
        vals = spiceypy.bodvrd(body_id, "GM", 1)
        return float(vals[0][0])
