"""Errors raised by this package."""

from __future__ import annotations


class ParTextError(ValueError):
    """A par file's *text* is malformed in a way no reader can resolve.

    Raised by :mod:`psrdata.partext` only. It is a text error, never a physics
    one: two active ``UNITS`` lines, a non-repeatable parameter given twice
    with different values. Callers wrap it in whatever their own ingest layer
    raises.
    """
