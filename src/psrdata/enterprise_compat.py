"""Enterprise compatibility constants (SPEC §8).

These keep Enterprise's established meanings and quirks so a producer fills
the record the way Enterprise's own loaders would.

"""

from __future__ import annotations

from typing import Mapping

import numpy as np

#: R-8.1: planet index into ``planetssb``'s second axis.
PLANET_SLOTS: Mapping[str, int] = {
    "mercury": 0,
    "venus": 1,
    "earth": 2,
    "mars": 3,
    "jupiter": 4,
    "saturn": 5,
    "uranus": 6,
    "neptune": 7,
    "pluto": 8,
}

#: R-8.2: each nonempty flag in this order overwrites the preceding value.
BACKEND_FLAG_ORDER = ("f", "i", "sys", "g", "group")

#: R-8.3: Enterprise's fallback (distance, uncertainty) in kpc.
DEFAULT_DISTANCE_KPC = (1.0, 0.2)


def backend_flags(flags: Mapping[str, np.ndarray], n: int) -> np.ndarray:
    """Enterprise's backend label for each of ``n`` rows (R-8.2).

    ``fe_be`` is formed first on rows where both ``fe`` and ``be`` are present
    and nonempty; then each nonempty flag in :data:`BACKEND_FLAG_ORDER`
    overwrites it. Rows with none of these get ``""``.

    "Where both are present" is read per row: both flags nonempty on that
    row.
    """
    labels = np.full(n, "", dtype=object)

    def as_rows(key):
        arr = np.asarray(flags[key]).astype("U")
        if arr.shape != (n,):
            raise ValueError(f"flag {key!r} must have shape ({n},), got {arr.shape}")
        return arr

    if "fe" in flags and "be" in flags:
        fe, be = as_rows("fe"), as_rows("be")
        both = (fe != "") & (be != "")
        labels[both] = np.char.add(np.char.add(fe[both], "_"), be[both])
    for key in BACKEND_FLAG_ORDER:
        if key in flags:
            arr = as_rows(key)
            mask = arr != ""
            labels[mask] = arr[mask]
    return labels.astype("U")


__all__ = [
    "PLANET_SLOTS",
    "BACKEND_FLAG_ORDER",
    "DEFAULT_DISTANCE_KPC",
    "backend_flags",
]
