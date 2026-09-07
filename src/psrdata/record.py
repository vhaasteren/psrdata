"""The pulsar: one frozen record of named arrays.

:class:`PulsarData` is simultaneously three things, with no adapter code
between them:

* nltiming's ``protocols.PulsarData`` plus its ephemeris extras, satisfied
  structurally;
* an Enterprise pulsar in the ``FeatherPulsar`` sense -- plain attributes with
  the names Enterprise binds by ``hasattr``, no properties, no live timing
  object;
* the in-memory form of the feather file (schema v1, :mod:`psrdata.feather`).

It lives here rather than in a timing package because a frozen array record
must not require a JAX engine to exist, and rather than in a protocol package
because protocols are not products. Its dependencies are numpy and pyarrow.

The record is also its own linear engine: :meth:`PulsarData.linear_engine`
returns ``Δr = −Mmat δ`` in the shape nltiming's ``TimingEngine`` protocol
describes (:mod:`psrdata.linear`), so a frozen linear timing analysis needs
this file and nothing else -- no timing package, no inference package.

**Row order is the writer's, and this package never changes it.** Row ``i`` of
every array is row ``i`` as the producing timing package emitted it. A consumer
that wants time order sorts on read: Enterprise already does, at its own
property layer, and both consumers' ECORR quantization groups TOAs by value
rather than adjacency, so neither needs sorted input. A timing package that
sorts the rows its residual, Jacobian and design matrix are all built from
publishes two orders for one freeze, reconciled by a permutation that is the
identity on any already-ordered ``.tim`` and therefore never exercised.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, NamedTuple

import numpy as np

from .gauge import GaugeProvenance, coerce_gauge

if TYPE_CHECKING:  # pragma: no cover
    from .linear import LinearTimingEngine

#: On-disk schema name and version. The columns are exactly the ones
#: Enterprise's ``FeatherPulsar`` and Discovery's ``Pulsar`` already read; the
#: metadata block is additive, so a reader that predates it ignores it.
#:
#: Wideband (schema still v1 until a producer emits it). A wideband TOA is one
#: row with a second measurement, not a second row: PINT forbids mixing
#: narrowband and wideband in one ``TOAs`` object, and Vela.jl's
#: ``WidebandTOA`` is a TOA plus ``DMInfo(value, error)``. Enterprise's
#: ``WidebandTimingModel`` already reads ``flags["pp_dm"]`` / ``flags["pp_dme"]``
#: and ``dmx`` from this record — those flags can land in :attr:`flags` with
#: no schema bump. A PINT/Vela residual engine needs the DM residual and its
#: error as arrays (``dm_residuals``, ``dmerrs``, and a DM design matrix or a
#: stacked ``(2N, n_fit)`` ``Mmat``); that is a new schema version, because
#: old readers would otherwise treat ``Mmat`` as TOA-only. Keep ``DMJUMP`` on
#: the par (it is a DM delay). Drop ``DMEFAC``/``DMEQUAD`` from
#: :data:`psrdata.partext.NOISE_NAMES` once that ingest exists — they scale
#: the DM error the way EFAC/EQUAD scale the TOA error. pyvela refuses ECORR
#: on wideband.
SCHEMA = "pulsardata-feather-v1"

#: Enterprise planet slots: Mercury=0 ... Pluto=8.
#:
#: What the consumers actually read, measured, so that nobody "fixes" the NaNs:
#:
#: ==============================  =========================================
#: reader                          slots
#: ==============================  =========================================
#: ``utils.physical_ephem_delay``  positions of 2 (Earth), 4, 5, 6, 7
#: ``PhysicalEphemerisSignal``     the same
#: Discovery ``solar.py``          slot 2 position, ``sunssb[:, :3]``
#: anything                        **no velocity is read by any consumer**
#: ==============================  =========================================
#:
#: Venus (slot 1) is filled whenever the producer had it: NaN-ing a column we
#: hold, "for parity" with Enterprise's PINT path, would be theater.
PLANET_SLOTS = {
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

#: Enterprise's backend-flag precedence, most specific last
#: (``enterprise/pulsar.py``: ``fe_be`` base, then these overwrite where
#: non-empty). The quirk is copied intact, not improved.
BACKEND_FLAG_ORDER = ("f", "i", "sys", "g", "group")

#: Enterprise's fallback when the par carries no parallax.
DEFAULT_DISTANCE_KPC = (1.0, 0.2)


class TOARows(NamedTuple):
    """The three columns that identify a row.

    Site arrival alone cannot separate simultaneous sub-band TOAs and frequency
    alone cannot separate epochs; together they pin a row. Two codes holding
    what should be the same data compare these to check that they do.
    """

    stoas: np.ndarray  # site arrival, seconds
    freqs: np.ndarray  # MHz; inf where the timing package reported no frequency
    toaerrs: np.ndarray  # seconds


def _readonly(array) -> np.ndarray:
    out = np.asarray(array)
    out.flags.writeable = False
    return out


def backend_flags(flags: Mapping[str, np.ndarray], n: int) -> np.ndarray:
    """Enterprise's ``backend_flags`` recipe, precedence quirk intact.

    ``fe_be`` is the base and the finer flags overwrite it where they are
    non-empty, last one winning (``enterprise/pulsar.py:327``).
    """
    out = np.array([""] * n, dtype=object)
    if "fe" in flags and "be" in flags:
        out[:] = [
            f"{fe}_{be}" if (fe and be) else ""
            for fe, be in zip(flags["fe"], flags["be"])
        ]
    for name in BACKEND_FLAG_ORDER:
        if name in flags:
            out[:] = np.where(np.asarray(flags[name]) == "", out, flags[name])
    return out.astype(str)


@dataclass(frozen=True)
class PulsarData:
    """The pulsar, frozen, in the writer's row order. Read-only array views."""

    name: str
    fitpars: tuple[str, ...]
    setpars: tuple[str, ...]
    toas: np.ndarray  # barycentric arrival, seconds
    stoas: np.ndarray  # site arrival, seconds
    toaerrs: np.ndarray  # seconds
    residuals: np.ndarray  # seconds
    freqs: np.ndarray  # MHz
    Mmat: np.ndarray  # (N, n_fit), fitter sign
    flags: Mapping[str, np.ndarray]
    backend_flags: np.ndarray
    telescope: np.ndarray
    pos: np.ndarray  # (3,) ICRS unit vector
    pos_t: np.ndarray  # (N, 3)
    sunssb: np.ndarray  # (N, 6) lt-s
    planetssb: np.ndarray  # (N, 9, 6); see PLANET_SLOTS
    theta: float
    phi: float
    pdist: tuple[float, float]
    dm: float
    dmx: dict | None
    state_id: str
    #: Which package wrote this record: ``"metapulsar"``, ...
    software: str
    #: Which timing package read the files: ``"pint"``, ``"tempo2"``, or
    #: ``"composite"`` for a record combining legs read by different ones.
    timing_package: str
    #: One :class:`~psrdata.gauge.GaugeProvenance`, or ``{leg: provenance}``
    #: for a composite. A plain mapping of the field names is accepted and
    #: validated on construction, which is also what the feather reader hands
    #: in.
    gauge: GaugeProvenance | Mapping[str, GaugeProvenance]
    reference_theta_exact: Mapping[str, str]
    native_units: Mapping[str, str]
    #: Additive metadata a producer wants to carry (a composite's ``legs``,
    #: its combination strategy, ...). Readers that do not know a key ignore it.
    extra: Mapping[str, Any] = field(default_factory=dict)
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        # Shapes are checked here rather than trusted, because every consumer
        # binds these names by `hasattr` and a wrong-length column becomes a
        # broadcasting bug three packages away.
        n = len(self.toas)
        for name in (
            "stoas",
            "toaerrs",
            "residuals",
            "freqs",
            "backend_flags",
            "telescope",
        ):
            if np.asarray(getattr(self, name)).shape != (n,):
                raise ValueError(f"{name} must have shape ({n},)")
        if np.asarray(self.Mmat).shape != (n, len(self.fitpars)):
            raise ValueError(f"Mmat must have shape ({n}, {len(self.fitpars)})")
        for name, shape in (
            ("pos", (3,)),
            ("pos_t", (n, 3)),
            ("sunssb", (n, 6)),
            ("planetssb", (n, 9, 6)),
        ):
            if np.asarray(getattr(self, name)).shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        for key, value in self.flags.items():
            if np.asarray(value).shape != (n,):
                raise ValueError(f"flags[{key!r}] must have shape ({n},)")
        missing = [f for f in self.fitpars if f not in self.reference_theta_exact]
        if missing:
            raise ValueError(f"reference_theta_exact is missing fitpars {missing}")
        object.__setattr__(
            self,
            "gauge",
            coerce_gauge(self.gauge, composite=self.timing_package == "composite"),
        )

        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, np.ndarray):
                object.__setattr__(self, f.name, _readonly(value))
        object.__setattr__(
            self,
            "flags",
            {k: _readonly(np.asarray(v)) for k, v in self.flags.items()},
        )

    def __len__(self) -> int:
        return len(self.toas)

    def __repr__(self) -> str:
        return (
            f"<psrdata.PulsarData {self.name}: {len(self.toas)} TOAs, "
            f"{len(self.fitpars)} fitpars, software={self.software}, "
            f"timing_package={self.timing_package}>"
        )

    def toa_rows(self) -> TOARows:
        """The three columns that identify a row."""
        return TOARows(self.stoas, self.freqs, self.toaerrs)

    def linear_engine(self) -> "LinearTimingEngine":
        """This record as its own linear engine, ``Δr = −Mmat δ``.

        Single-leg or composite; see :func:`psrdata.linear.linear_engine`.
        """
        from .linear import linear_engine

        return linear_engine(self)

    def to_feather(self, path, *, noisedict=None) -> Path:
        from .feather import write

        return write(self, path, noisedict=noisedict)

    @classmethod
    def from_feather(cls, path) -> "PulsarData":
        from .feather import read

        return read(path)


__all__ = [
    "PulsarData",
    "TOARows",
    "SCHEMA",
    "PLANET_SLOTS",
    "BACKEND_FLAG_ORDER",
    "DEFAULT_DISTANCE_KPC",
    "backend_flags",
]
