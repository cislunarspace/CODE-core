from __future__ import annotations

from typing import List

from .spice import SPICEManager


class EphemerisSystem:
    def __init__(
        self,
        bodies: List[str],
        spice: SPICEManager,
        origin: str = "EARTH",
        frame: str = "J2000",
    ) -> None:
        self.bodies = list(bodies)
        self.spice = spice
        self.origin = origin
        self.frame = frame

    def get_gm_values(self) -> List[float]:
        return [self.spice.get_gm(body) for body in self.bodies]
