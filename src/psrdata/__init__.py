"""psrdata: the frozen pulsar record, its feather schema, its own linear
engine, the gauge provenance it carries, and the par-text rules.

    from psrdata import PulsarData
    psr = PulsarData.from_feather("J1909-3744.feather")
    engine = psr.linear_engine()          # Δr = −Mmat δ, nltiming's engine shape

Deliberately depends on numpy and pyarrow only. This package sits *below*
every timing package and every likelihood, so it must not import PINT, tempo2,
JAX or astropy -- a frozen array record that needed a timing engine to read
would defeat its own purpose, and so would a linear engine that needed one to
run. ``tests/test_imports.py`` asserts it.
"""

from __future__ import annotations

from . import partext
from .errors import ParTextError
from .gauge import GaugeProvenance
from .linear import (
    LinearContribution,
    LinearModel,
    LinearTimingEngine,
    RecordLinearTimingEngine,
    linear_engine,
)
from .record import (
    BACKEND_FLAG_ORDER,
    DEFAULT_DISTANCE_KPC,
    PLANET_SLOTS,
    SCHEMA,
    PulsarData,
    TOARows,
    backend_flags,
)

try:  # pragma: no cover - packaging detail
    from importlib.metadata import version

    __version__ = version("psrdata")
except Exception:  # pragma: no cover
    __version__ = "0.0.0.dev0"

__all__ = [
    "PulsarData",
    "TOARows",
    "GaugeProvenance",
    "LinearModel",
    "LinearContribution",
    "LinearTimingEngine",
    "RecordLinearTimingEngine",
    "linear_engine",
    "SCHEMA",
    "PLANET_SLOTS",
    "BACKEND_FLAG_ORDER",
    "DEFAULT_DISTANCE_KPC",
    "backend_flags",
    "partext",
    "ParTextError",
    "__version__",
]
