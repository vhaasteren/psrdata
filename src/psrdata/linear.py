"""The record's linear engine: ``Δr = -Mmat @ δ`` (SPEC §3.6, §5).

The engine follows nltiming's timing-engine interface structurally, without
importing nltiming (R-5.4.1), and imports no JAX (R-5.2.10). A single-data-set
and a combined record get the same engine; per-data-set structure is read off
the phase-offset columns (R-3.7.3), never from stored partition metadata.

"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Mapping

import numpy as np

from .errors import LinearEngineError
from .record import SINGLE_KEY, ParameterFact, PulsarData, ResidualCentering

ENGINE_NAME = "linear"

PHASE_OFFSET_BARE_NAMES = ("Offset", "PHOFF")


class _LinearBlock:
    """Shared behaviour of the whole-record engine and one data-set contribution."""

    engine_name: str = ENGINE_NAME
    nonlinear_params: None = None

    def __init__(
        self,
        Mmat: np.ndarray,
        fitpars: tuple[str, ...],
        parameters: Mapping[str, ParameterFact],
    ):
        self._Mmat = Mmat
        self._fitpars = tuple(fitpars)
        self._facts = {name: parameters[name] for name in self._fitpars}

    # -- fit coordinates (R-5.2.1 .. R-5.2.4) -----------------------------------

    @property
    def fitpars(self) -> tuple[str, ...]:
        return self._fitpars

    @property
    def native_units(self) -> dict[str, str]:
        """Mapping of each fit parameter to its PINT unit, in fitpar order (R-5.2.2)."""
        return {name: fact.units for name, fact in self._facts.items()}

    def reference_theta_exact(self) -> dict[str, str]:
        """Mapping of each fit parameter to its decimal value string, in fitpar
        order (R-5.2.3). This is the authority for the recorded reference."""
        return {name: fact.value for name, fact in self._facts.items()}

    def reference_theta(self) -> np.ndarray:
        """Derived float64 reference vector (R-5.2.4, R-3.3.5).

        Each entry is the correctly rounded value of the decimal string, with
        no intermediate float round trip. float64 is not the authority for a
        parameter such as ``F0``; read ``reference_theta_exact()`` for the
        recorded digits.
        """
        return np.array(
            [float(Decimal(fact.value)) for fact in self._facts.values()],
            dtype=np.float64,
        )

    @property
    def linear_fitpars(self) -> tuple[str, ...]:
        """Compatibility field carrying the exactly linear parameters (R-5.3.2):
        every fit parameter of a linear block."""
        return self._fitpars

    def identically_linear_fitpars(self) -> tuple[str, ...]:
        """R-5.2.8: every fit parameter."""
        return self._fitpars

    # -- the calculation (R-5.2.5 .. R-5.2.7) -----------------------------------

    def residual_delta(self, delta) -> np.ndarray:
        """``-Mmat @ δ`` for ``δ`` of shape ``(p,)`` in fitpar order and PINT units."""
        delta = np.asarray(delta, dtype=np.float64)
        p = len(self._fitpars)
        if delta.shape != (p,):
            raise LinearEngineError(
                f"delta must have shape ({p},) for fitpars {self._fitpars}, "
                f"got {delta.shape}"
            )
        return -(self._Mmat @ delta)

    def design_matrix(self, params=None) -> np.ndarray:
        """``Mmat``. A linear engine has one design matrix (R-5.2.6)."""
        if params is not None:
            raise LinearEngineError(
                "the linear engine's design matrix is the record's Mmat and takes "
                "no parameter values; pass params=None"
            )
        return self._Mmat

    def residual_jacobian(self, params=None) -> np.ndarray:
        """``-Mmat`` (R-5.2.7)."""
        if params is not None:
            raise LinearEngineError(
                "the linear engine's residual Jacobian is -Mmat and takes no "
                "parameter values; pass params=None"
            )
        return -self._Mmat

    def binary_chart_capability(self, *args: Any, **kwargs: Any) -> None:
        """R-5.2.9: a linear engine has no binary chart."""
        return None


class LinearContribution(_LinearBlock):
    """One data set's block of the linear calculation (§5.3).

    Its rows are the support of its phase-offset column; its fit parameters
    are the columns nonzero on those rows, in global fitpar order.
    """

    def __init__(
        self,
        *,
        key: str,
        rows: np.ndarray,
        column_indices: tuple[int, ...],
        phase_offset: str,
        record: PulsarData,
    ):
        fitpars = tuple(record.fitpars[j] for j in column_indices)
        super().__init__(
            record.Mmat[np.ix_(rows, list(column_indices))], fitpars, record.parameters
        )
        self.key = key
        self.rows = rows
        self.column_indices = column_indices
        self.phase_offset = phase_offset
        self.timing_package: str = record.timing_package[key]
        self.partim_compatibility: str = record.partim_compatibility[key]
        self.residual_centering: ResidualCentering = record.residual_centering[key]
        self.residuals = record.residuals[rows]

    def __repr__(self) -> str:
        return (
            f"<LinearContribution {self.key!r}: {len(self.rows)} rows, "
            f"fitpars {self.fitpars}>"
        )


class LinearTimingEngine(_LinearBlock):
    """The complete linear calculation over a record's matrix (§5)."""

    def __init__(self, record: PulsarData):
        super().__init__(record.Mmat, record.fitpars, record.parameters)
        self.record = record
        self._contributions = _decompose(record)

    @classmethod
    def from_pulsar_data(cls, record: PulsarData) -> "LinearTimingEngine":
        return cls(record)

    @classmethod
    def from_feather(cls, path) -> "LinearTimingEngine":
        from .feather import read

        return cls(read(path))

    # -- per-data-set views (§5.3) ---------------------------------------------

    @property
    def timing_package(self) -> Mapping[str, str]:
        return self.record.timing_package

    @property
    def partim_compatibility(self) -> Mapping[str, str]:
        return self.record.partim_compatibility

    @property
    def residual_centering(self) -> Mapping[str, ResidualCentering]:
        """One entry per data set, for one or several alike (R-5.3.4)."""
        return self.record.residual_centering

    @property
    def data_set_keys(self) -> tuple[str, ...]:
        return tuple(self._contributions)

    def contributions(self) -> dict[str, LinearContribution]:
        """The per-data-set contributions (§5.3), keyed by data-set key in the
        record's data-set order."""
        return dict(self._contributions)

    def __repr__(self) -> str:
        return (
            f"<LinearTimingEngine {self.record.name}: {self._Mmat.shape[0]} rows, "
            f"{len(self.fitpars)} fitpars, data sets {self.data_set_keys}>"
        )


def locate_phase_offset(record: PulsarData, key: str) -> str:
    """The name of data set ``key``'s phase-offset fit parameter (R-3.6.1)."""
    candidates = [f"{bare}_{key}" for bare in PHASE_OFFSET_BARE_NAMES]
    if len(record.timing_package) == 1:
        candidates += list(PHASE_OFFSET_BARE_NAMES)
    present = [name for name in candidates if name in record.fitpars]
    if len(present) != 1:
        raise LinearEngineError(
            f"data set {key!r} needs exactly one phase-offset fit parameter among "
            f"{candidates}; found {present}"
        )
    return present[0]


def _decompose(record: PulsarData) -> dict[str, LinearContribution]:
    """R-5.3.1: rows from phase-offset support, columns from activity."""
    n = record.Mmat.shape[0]
    keys = tuple(record.timing_package)
    names = {key: locate_phase_offset(record, key) for key in keys}
    supports = {}
    for key, name in names.items():
        column = record.Mmat[:, record.fitpars.index(name)]
        supports[key] = np.flatnonzero(column != 0.0)
    membership = np.zeros(n, dtype=int)
    for rows in supports.values():
        membership[rows] += 1
    if (membership != 1).any():
        overlap = np.flatnonzero(membership > 1)
        gap = np.flatnonzero(membership == 0)
        raise LinearEngineError(
            "the phase-offset columns do not partition the rows: "
            f"{len(overlap)} rows in more than one support, {len(gap)} rows in "
            f"none (columns {tuple(names.values())})"
        )
    contributions = {}
    for key in keys:
        rows = supports[key]
        active = np.flatnonzero((record.Mmat[rows] != 0.0).any(axis=0))
        contributions[key] = LinearContribution(
            key=key,
            rows=rows,
            column_indices=tuple(int(j) for j in active),
            phase_offset=names[key],
            record=record,
        )
    return contributions


def linear_engine(record: PulsarData) -> LinearTimingEngine:
    """``psrdata.linear_engine(record)`` (§5.1)."""
    return LinearTimingEngine.from_pulsar_data(record)


__all__ = [
    "ENGINE_NAME",
    "SINGLE_KEY",
    "LinearTimingEngine",
    "LinearContribution",
    "linear_engine",
    "locate_phase_offset",
]
