"""The public psrdata error categories (SPEC §14).

All four derive from ``ValueError``, so a caller that catches the spec's
category catches the Python category too.
"""

from __future__ import annotations


class RecordError(ValueError):
    """Invalid record shapes, parameter coverage, mappings or phase-offset columns."""


class SchemaError(ValueError):
    """Missing, malformed or unsupported psrdata schema metadata."""


class LinearEngineError(ValueError):
    """An invalid delta or an impossible linear decomposition."""


class ParTextError(ValueError):
    """Malformed par text handled by :mod:`psrdata.partext`."""


__all__ = ["RecordError", "SchemaError", "LinearEngineError", "ParTextError"]
