"""psrdata: the shared pulsar-data record, its Feather schema, its own linear
engine and the pure par-text rules.

Runtime dependencies are NumPy and PyArrow only (R-2.1); importing this package
must not import a timing package, JAX, Astropy, SciPy, Enterprise, Discovery,
MetaPulsar or nltiming.
"""

from __future__ import annotations

from . import feather, partext
from .enterprise_compat import (
    BACKEND_FLAG_ORDER,
    DEFAULT_DISTANCE_KPC,
    PLANET_SLOTS,
    backend_flags,
)
from .errors import LinearEngineError, ParTextError, RecordError, SchemaError
from .feather import SCHEMA
from .linear import LinearContribution, LinearTimingEngine, linear_engine
from .record import (
    SINGLE_KEY,
    ParameterFact,
    PulsarData,
    ResidualCentering,
    TOARows,
)

__all__ = [
    "PulsarData",
    "ParameterFact",
    "ResidualCentering",
    "TOARows",
    "SINGLE_KEY",
    "LinearTimingEngine",
    "LinearContribution",
    "linear_engine",
    "SCHEMA",
    "feather",
    "partext",
    "PLANET_SLOTS",
    "BACKEND_FLAG_ORDER",
    "DEFAULT_DISTANCE_KPC",
    "backend_flags",
    "RecordError",
    "SchemaError",
    "LinearEngineError",
    "ParTextError",
]
