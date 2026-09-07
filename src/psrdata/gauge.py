"""Gauge provenance: what a producer did about the unmeasurable phase offset.

This type is serialized in the record (the ``gauge`` field, one per record or
one per leg of a composite), which is why it lives here and not in the
inference layer that reads it. It used to be defined in nltiming, mirrored
as plain data in vela-jax, and reproduced as a dict literal in MetaPulsar;
the file format's ``gauge`` key was therefore defined one layer up, and a
renamed field there would have made every file on disk unreadable. nltiming
re-exports this class under its old name, so nothing above changes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

_EXPORTS = ("none", "applied-unknown")
_MODES = ("none", "mean", "constant", "unknown")


@dataclass(frozen=True)
class GaugeProvenance:
    """Serializable gauge facts for one record or one leg. Manifest data only.

    ``export``: whether a gauge (mean subtraction or similar) was applied to
    the exported residuals. ``"none"`` means the residual is gauge-free,
    defined only modulo the gauge column.

    ``reference_mode`` / ``reference_weighted``: the gauge the reference
    residual was reduced with; ``reference_weighted`` is required for
    ``"mean"`` and forbidden otherwise.

    ``reporting_mode`` / ``reporting_weighted``: the gauge a consumer would
    see if it reduced these residuals the way the timing package's family
    reports them; the same rule for the weighted flag.
    """

    export: Literal["none", "applied-unknown"]
    reference_mode: Literal["none", "mean", "constant", "unknown"]
    reference_weighted: bool | None = None  # only for reference_mode="mean"
    reporting_mode: Literal["none", "mean", "constant", "unknown"] = "unknown"
    reporting_weighted: bool | None = None

    def __post_init__(self) -> None:
        if self.export not in _EXPORTS:
            raise ValueError(
                f"GaugeProvenance.export must be 'none' or 'applied-unknown'; "
                f"got {self.export!r}"
            )
        if self.reference_mode not in _MODES:
            raise ValueError(
                f"GaugeProvenance.reference_mode invalid: {self.reference_mode!r}"
            )
        if self.reference_mode == "mean":
            if self.reference_weighted is None:
                raise ValueError(
                    "GaugeProvenance(reference_mode='mean') requires "
                    "reference_weighted"
                )
        elif self.reference_weighted is not None:
            raise ValueError(
                "reference_weighted is only allowed when reference_mode='mean'"
            )
        if self.reporting_mode not in _MODES:
            raise ValueError(
                f"GaugeProvenance.reporting_mode invalid: {self.reporting_mode!r}"
            )
        if self.reporting_mode == "mean":
            if self.reporting_weighted is None:
                raise ValueError(
                    "GaugeProvenance(reporting_mode='mean') requires "
                    "reporting_weighted"
                )
        elif self.reporting_weighted is not None:
            raise ValueError(
                "reporting_weighted is only allowed when reporting_mode='mean'"
            )

    @classmethod
    def from_mapping(cls, value: Any) -> "GaugeProvenance":
        """Accept an instance, or a plain mapping of the field names."""
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(**dict(value))
        raise TypeError(
            f"gauge provenance must be a GaugeProvenance or a mapping; "
            f"got {type(value).__name__}"
        )

    def to_dict(self) -> dict[str, Any]:
        """The JSON form: the field names, as the feather block stores them."""
        return asdict(self)


def coerce_gauge(value: Any, *, composite: bool):
    """The record's ``gauge`` field, validated.

    A single-leg record carries one provenance; a composite carries one per
    leg, keyed by leg name. Either form accepts instances or plain mappings.
    """
    if composite:
        if (
            not isinstance(value, Mapping)
            or isinstance(value, GaugeProvenance)
            or "export" in value
        ):
            raise TypeError(
                "a composite record's gauge must map leg -> provenance; got a "
                "single provenance"
            )
        return {str(leg): GaugeProvenance.from_mapping(v) for leg, v in value.items()}
    return GaugeProvenance.from_mapping(value)


def gauge_to_json(value) -> dict[str, Any]:
    """The feather form of a validated ``gauge`` field."""
    if isinstance(value, GaugeProvenance):
        return value.to_dict()
    return {leg: GaugeProvenance.from_mapping(v).to_dict() for leg, v in value.items()}


__all__ = ["GaugeProvenance", "coerce_gauge", "gauge_to_json"]
