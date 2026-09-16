"""The ``PulsarData`` record and its value types (SPEC §3, §4).

A record is the result of one materialized timing calculation at one
reference model: row arrays, residuals, design matrix and the facts needed to
interpret them. psrdata validates what it can observe structurally (§3.2,
R-4.1.1) and never reorders rows (R-1.4). What the values *mean* is the
producer's obligation (§9).

"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from numbers import Real
from typing import Any, Literal, Mapping, NamedTuple

import numpy as np

from .errors import RecordError

#: The reserved data-set key psrdata inserts for scalar standalone metadata.
SINGLE_KEY = "single"

PARTIM_COMPATIBILITIES = ("pint", "tempo2")

CENTERING_LITERALS = ("none", "mean_removed", "constant_removed", "unknown")

#: A phase-offset fit parameter: bare ``Offset``/``PHOFF`` or ``<name>_<key>``.
PHASE_OFFSET_RE = re.compile(r"^(Offset|PHOFF)(?:_(.+))?$")

#: The unit of a phase-offset parameter's delta, by bare name (R-3.2.2).
PHASE_OFFSET_UNITS = {"Offset": "s", "PHOFF": "dimensionless"}

DMX_ENTRY_KEYS = ("DMX", "DMXerr", "DMXR1", "DMXR2", "fit")


# --- value types ------------------------------------------------------------------


@dataclass
class ParameterFact:
    """Value, PINT unit and optional uncertainty of one set parameter (§3.3)."""

    value: str
    units: str | None
    uncertainty: str | None = None


@dataclass
class ResidualCentering:
    """What centering the stored residuals carry, and what the package normally
    applies (§4). Descriptive only: nothing in psrdata branches on it."""

    stored_residuals: Literal["none", "mean_removed", "constant_removed", "unknown"]
    stored_weighted: bool | None = None
    standard_output: Literal["none", "mean_removed", "constant_removed", "unknown"] = (
        "unknown"
    )
    standard_weighted: bool | None = None

    def __post_init__(self):
        validate_residual_centering(self)


def validate_residual_centering(rc: ResidualCentering, where: str = "") -> None:
    """R-4.1.1. Raised as ``RecordError`` at construction and at Feather read."""
    prefix = f"{where}: " if where else ""
    if not isinstance(rc, ResidualCentering):
        raise RecordError(f"{prefix}expected a ResidualCentering, got {rc!r}")
    for kind in ("stored", "standard"):
        literal = getattr(
            rc, "stored_residuals" if kind == "stored" else "standard_output"
        )
        weighted = getattr(rc, f"{kind}_weighted")
        if literal not in CENTERING_LITERALS:
            raise RecordError(
                f"{prefix}{kind} residual centering {literal!r} is not one of "
                f"{CENTERING_LITERALS}"
            )
        if literal == "mean_removed":
            if not isinstance(weighted, bool):
                raise RecordError(
                    f"{prefix}{kind}_weighted must be a bool when the {kind} "
                    f"centering is 'mean_removed', got {weighted!r}"
                )
        elif weighted is not None:
            raise RecordError(
                f"{prefix}{kind}_weighted must be None unless the {kind} "
                f"centering is 'mean_removed', got {weighted!r}"
            )


class TOARows(NamedTuple):
    """Ordered alignment signature for transitional two-read comparison (R-3.8.1)."""

    stoas: np.ndarray
    freqs: np.ndarray
    toaerrs: np.ndarray


# --- the record -------------------------------------------------------------------

_FLOAT_ROW_FIELDS = ("toas", "stoas", "toaerrs", "residuals", "freqs")
_STR_ROW_FIELDS = ("backend_flags", "telescope")


def _as_float_array(name: str, value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype.kind != "f":
        raise RecordError(f"{name} must be a floating array, got dtype {arr.dtype}")
    return arr


def _as_str_array(name: str, value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype.kind == "O":
        if not all(isinstance(x, str) for x in arr.ravel()):
            raise RecordError(f"{name} must contain only strings")
    elif arr.dtype.kind not in "US":
        raise RecordError(f"{name} must be a string array, got dtype {arr.dtype}")
    return arr


def _check_shape(name: str, arr: np.ndarray, shape: tuple[int, ...]) -> None:
    if arr.shape != shape:
        raise RecordError(f"{name} must have shape {shape}, got {arr.shape}")


def _is_finite_decimal(text: str) -> bool:
    try:
        return Decimal(text).is_finite()
    except (InvalidOperation, ValueError, TypeError):
        return False


def _is_decimal_zero(text: str) -> bool:
    try:
        return Decimal(text) == 0
    except (InvalidOperation, ValueError, TypeError):
        return False


def _require_finite(name: str, arr: np.ndarray) -> None:
    if not np.isfinite(arr).all():
        raise RecordError(f"{name} must be finite")


def _require_finite_or_nan(name: str, arr: np.ndarray) -> None:
    if np.isinf(arr).any():
        raise RecordError(f"{name} must be finite or NaN")


@dataclass(frozen=True, eq=False, kw_only=True)
class PulsarData:
    """One materialized timing calculation and the metadata to interpret it (§3.1).

    Frozen at the attribute level only: arrays are stored as given, never
    copied (SPEC-motivation §5). Row ``i`` of every row array is the
    producer's row ``i`` (R-1.4).
    """

    name: str
    setpars: tuple[str, ...]
    fitpars: tuple[str, ...]
    parameters: Mapping[str, ParameterFact]
    toas: np.ndarray
    stoas: np.ndarray
    toaerrs: np.ndarray
    residuals: np.ndarray
    freqs: np.ndarray
    Mmat: np.ndarray
    flags: Mapping[str, np.ndarray]
    backend_flags: np.ndarray
    telescope: np.ndarray
    pos: np.ndarray
    pos_t: np.ndarray
    sunssb: np.ndarray
    planetssb: np.ndarray
    theta: float
    phi: float
    pdist: tuple[float, float]
    dm: float
    dmx: Mapping[str, Mapping[str, Any]] | None
    timing_package: Mapping[str, str]
    partim_compatibility: Mapping[str, Literal["pint", "tempo2"]]
    residual_centering: Mapping[str, ResidualCentering]
    producer: str
    extra: Mapping[str, Any] = field(default_factory=dict)

    # -- construction -----------------------------------------------------------

    def __post_init__(self):
        set_ = object.__setattr__

        if not isinstance(self.name, str) or not self.name:
            raise RecordError(f"name must be a nonempty string, got {self.name!r}")
        if not isinstance(self.producer, str) or not self.producer:
            raise RecordError(
                f"producer must be a nonempty string, got {self.producer!r}"
            )
        if not isinstance(self.extra, Mapping):
            raise RecordError(f"extra must be a mapping, got {type(self.extra)}")

        # row arrays and shapes (R-3.2.1)
        for name in _FLOAT_ROW_FIELDS:
            set_(self, name, _as_float_array(name, getattr(self, name)))
        for name in _STR_ROW_FIELDS:
            set_(self, name, _as_str_array(name, getattr(self, name)))
        if self.toas.ndim != 1:
            raise RecordError(f"toas must have shape (n,), got {self.toas.shape}")
        n = len(self.toas)
        for name in _FLOAT_ROW_FIELDS + _STR_ROW_FIELDS:
            _check_shape(name, getattr(self, name), (n,))

        set_(self, "setpars", tuple(self.setpars))
        set_(self, "fitpars", tuple(self.fitpars))
        for field_name in ("setpars", "fitpars"):
            names = getattr(self, field_name)
            bad = [name for name in names if not isinstance(name, str) or not name]
            if bad:
                raise RecordError(
                    f"{field_name} must contain nonempty strings, got {bad!r}"
                )
        p = len(self.fitpars)
        set_(self, "Mmat", _as_float_array("Mmat", self.Mmat))
        _check_shape("Mmat", self.Mmat, (n, p))
        for name, shape in (
            ("pos", (3,)),
            ("pos_t", (n, 3)),
            ("sunssb", (n, 6)),
            ("planetssb", (n, 9, 6)),
        ):
            arr = _as_float_array(name, getattr(self, name))
            _check_shape(name, arr, shape)
            set_(self, name, arr)

        # flags (R-3.2.1, R-3.2.4)
        if not isinstance(self.flags, Mapping):
            raise RecordError(f"flags must be a mapping, got {type(self.flags)}")
        flags = {}
        for key, value in self.flags.items():
            if not isinstance(key, str) or not key:
                raise RecordError(f"flag keys must be nonempty strings, got {key!r}")
            arr = _as_str_array(f"flag {key!r}", value)
            _check_shape(f"flag {key!r}", arr, (n,))
            flags[key] = arr
        set_(self, "flags", flags)

        # finite values (R-3.2.5)
        for name in ("toas", "stoas", "toaerrs", "residuals", "Mmat", "pos", "pos_t"):
            _require_finite(name, getattr(self, name))
        _require_finite("sunssb position", self.sunssb[..., :3])
        _require_finite_or_nan("sunssb unused velocity", self.sunssb[..., 3:])
        _require_finite_or_nan("planetssb", self.planetssb)
        if np.isnan(self.freqs).any() or np.isneginf(self.freqs).any():
            raise RecordError("freqs may contain +inf but not NaN or -inf")

        # parameter names and facts (R-3.2.2, R-3.3)
        self._validate_parameters()

        # scalars
        for name in ("theta", "phi", "dm"):
            value = getattr(self, name)
            if not isinstance(value, Real):
                raise RecordError(f"{name} must be a float, got {value!r}")
            value = float(value)
            if not np.isfinite(value):
                raise RecordError(f"{name} must be finite")
            set_(self, name, value)
        try:
            d, e = self.pdist
            pdist = (float(d), float(e))
        except (TypeError, ValueError):
            raise RecordError(
                f"pdist must be a (distance, uncertainty) pair, got {self.pdist!r}"
            ) from None
        if not all(np.isfinite(x) for x in pdist):
            raise RecordError("pdist must be finite")
        set_(self, "pdist", pdist)
        self._validate_dmx()

        # data-set mappings (R-3.2.3, R-4.1.1)
        self._normalize_data_set_mappings()

    def _validate_parameters(self) -> None:
        if len(set(self.setpars)) != len(self.setpars):
            raise RecordError("setpars contains duplicate names")
        if len(set(self.fitpars)) != len(self.fitpars):
            raise RecordError("fitpars contains duplicate names")
        setpars = set(self.setpars)
        for name in self.fitpars:
            if name not in setpars:
                raise RecordError(f"fit parameter {name!r} is not in setpars")
        if not isinstance(self.parameters, Mapping):
            raise RecordError("parameters must be a mapping of ParameterFact")
        for name in self.setpars:
            if name not in self.parameters:
                raise RecordError(f"set parameter {name!r} has no ParameterFact")
        # A fact for a name outside setpars is refused: a parameter with a
        # value is by definition set (§0, R-3.3.4).
        for name in self.parameters:
            if name not in setpars:
                raise RecordError(
                    f"parameters has an entry {name!r} that is not in setpars"
                )
        for name, fact in self.parameters.items():
            if not isinstance(fact, ParameterFact):
                raise RecordError(f"parameters[{name!r}] is not a ParameterFact")
            if not isinstance(fact.value, str):
                raise RecordError(f"parameters[{name!r}].value must be a string")
            if fact.units is not None and not isinstance(fact.units, str):
                raise RecordError(f"parameters[{name!r}].units must be str or None")
            if fact.uncertainty is not None and not isinstance(fact.uncertainty, str):
                raise RecordError(
                    f"parameters[{name!r}].uncertainty must be str or None"
                )
            if fact.units is not None and not _is_finite_decimal(fact.value):
                raise RecordError(
                    f"numerical parameter {name!r} has a non-finite or "
                    f"non-decimal value {fact.value!r}"
                )
            if fact.uncertainty is not None and not _is_finite_decimal(
                fact.uncertainty
            ):
                raise RecordError(
                    f"parameter {name!r} has a non-finite or non-decimal "
                    f"uncertainty {fact.uncertainty!r}"
                )
        # A fit parameter is a matrix column, so it is numerical with a unit
        # (R-3.3.2).
        for name in self.fitpars:
            fact = self.parameters[name]
            if fact.units is None:
                raise RecordError(f"fit parameter {name!r} has no unit")
            match = PHASE_OFFSET_RE.match(name)
            if match is None:
                continue
            expected = PHASE_OFFSET_UNITS[match.group(1)]
            if not _is_decimal_zero(fact.value) or fact.uncertainty is not None:
                raise RecordError(
                    f"phase-offset fit parameter {name!r} must have a decimal "
                    f"value of exactly zero and no uncertainty, got {fact!r}"
                )
            if fact.units != expected:
                raise RecordError(
                    f"phase-offset fit parameter {name!r} must have units "
                    f"{expected!r}, got {fact.units!r}"
                )

    def _validate_dmx(self) -> None:
        dmx = self.dmx
        if dmx is None:
            return
        if not isinstance(dmx, Mapping):
            raise RecordError(f"dmx must be a mapping or None, got {type(dmx)}")
        # An empty mapping is refused unconditionally: a model without DMX
        # uses None, and an empty table on a model with DMX would silently
        # disable Enterprise's wideband model (§3.1).
        if not dmx:
            raise RecordError(
                "dmx is an empty mapping; use None for a model without DMX"
            )
        for name, entry in dmx.items():
            if not isinstance(name, str) or not isinstance(entry, Mapping):
                raise RecordError(f"dmx entry {name!r} is malformed")
            missing = [k for k in DMX_ENTRY_KEYS if k not in entry]
            if missing:
                raise RecordError(f"dmx entry {name!r} lacks {missing}")
            for k in ("DMX", "DMXR1", "DMXR2"):
                value = entry[k]
                if (
                    not isinstance(value, Real)
                    or isinstance(value, bool)
                    or not np.isfinite(value)
                ):
                    raise RecordError(
                        f"dmx entry {name!r}[{k!r}] must be a finite float"
                    )
            dmxerr = entry["DMXerr"]
            if dmxerr is not None and (
                not isinstance(dmxerr, Real)
                or isinstance(dmxerr, bool)
                or not np.isfinite(dmxerr)
            ):
                raise RecordError(
                    f"dmx entry {name!r}['DMXerr'] must be a finite float or None"
                )
            if not isinstance(entry["fit"], bool):
                raise RecordError(f"dmx entry {name!r}['fit'] must be a bool")

    def _normalize_data_set_mappings(self) -> None:
        set_ = object.__setattr__
        tp, pc, rc = (
            self.timing_package,
            self.partim_compatibility,
            self.residual_centering,
        )
        scalar = (
            isinstance(tp, str),
            isinstance(pc, str),
            isinstance(rc, ResidualCentering),
        )
        if all(scalar):
            tp = {SINGLE_KEY: tp}
            pc = {SINGLE_KEY: pc}
            rc = {SINGLE_KEY: rc}
        elif any(scalar):
            raise RecordError(
                "timing_package, partim_compatibility and residual_centering must "
                "be supplied either all as scalar single-data-set values or all as "
                "mappings"
            )
        for label, mapping in (
            ("timing_package", tp),
            ("partim_compatibility", pc),
            ("residual_centering", rc),
        ):
            if not isinstance(mapping, Mapping):
                raise RecordError(f"{label} must be a mapping, got {type(mapping)}")
            if not mapping:
                raise RecordError(f"{label} must be a nonempty mapping")
            for key in mapping:
                if not isinstance(key, str) or not key:
                    raise RecordError(f"{label} has a non-string or empty key {key!r}")
        keys = tuple(tp)
        if set(pc) != set(keys) or set(rc) != set(keys):
            raise RecordError(
                "timing_package, partim_compatibility and residual_centering must "
                f"have exactly the same data-set keys; got {sorted(tp)}, "
                f"{sorted(pc)}, {sorted(rc)}"
            )
        for key, value in tp.items():
            if not isinstance(value, str) or not value:
                raise RecordError(
                    f"timing_package[{key!r}] must be a nonempty string, got {value!r}"
                )
        for key, value in pc.items():
            if value not in PARTIM_COMPATIBILITIES:
                raise RecordError(
                    f"partim_compatibility[{key!r}] must be one of "
                    f"{PARTIM_COMPATIBILITIES}, got {value!r}"
                )
        for key, value in rc.items():
            validate_residual_centering(value, f"residual_centering[{key!r}]")
        # The three mappings are re-keyed in ``timing_package``'s insertion
        # order so R-6.2.0's "same keys in the same insertion order" holds for
        # every record.
        set_(self, "timing_package", {k: tp[k] for k in keys})
        set_(self, "partim_compatibility", {k: pc[k] for k in keys})
        set_(self, "residual_centering", {k: rc[k] for k in keys})

    # -- derived views -----------------------------------------------------------

    @property
    def data_set_keys(self) -> tuple[str, ...]:
        """The data sets represented in the record (R-3.7.2)."""
        return tuple(self.timing_package)

    def toa_rows(self) -> TOARows:
        """R-3.8.1."""
        return TOARows(self.stoas, self.freqs, self.toaerrs)

    def linear_engine(self):
        """The complete linear calculation over the record's matrix (§5)."""
        from .linear import LinearTimingEngine

        return LinearTimingEngine.from_pulsar_data(self)

    @classmethod
    def from_feather(cls, path) -> "PulsarData":
        from .feather import read

        return read(path)

    def to_feather(self, path, noisedict=None) -> None:
        from .feather import write

        write(self, path, noisedict=noisedict)

    def __repr__(self) -> str:
        return (
            f"<PulsarData {self.name}: {len(self.toas)} rows, "
            f"{len(self.fitpars)} fitpars, data sets {self.data_set_keys}>"
        )


__all__ = [
    "SINGLE_KEY",
    "PARTIM_COMPATIBILITIES",
    "CENTERING_LITERALS",
    "PHASE_OFFSET_RE",
    "PHASE_OFFSET_UNITS",
    "ParameterFact",
    "ResidualCentering",
    "validate_residual_centering",
    "TOARows",
    "PulsarData",
]
