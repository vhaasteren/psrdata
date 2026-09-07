"""The record's own linear engine: ``Δr = −Mmat δ`` and nothing else.

A record is the image of one evaluation of a timing model, and its
linearization is complete in the record itself: ``Mmat`` in the fitter
sign, the exact reference strings, the units, the per-leg gauge provenance.
This module makes that identity callable, in the shape nltiming's
``TimingEngine`` / ``JacobianTimingEngine`` protocols describe, **without
importing nltiming**: the protocols are structural, this package defines
nothing about live functions, and conformance is asserted by a
``consumers``-marked test that runs nltiming's own ``isinstance`` checks
when nltiming is installed (the same way the feather layout is asserted
against Enterprise and Discovery).

Every optional hook nltiming reads by name has a well-defined linear answer
and is declared here so a reader never has to guess:

==========================  ==============================================
hook                        the linear engine's answer
==========================  ==============================================
``identically_linear_fitpars``  every fitpar
``nonlinear_params``        ``None`` (all axes linear; that is the executed mode)
``binary_chart_capability`` ``None`` (no nonlinear map to chart)
``contributions``           one per leg, derived from the matrix (composite)
``gauge_direction``         the record's own named gauge column
``residual_jacobian``       ``-Mmat``
==========================  ==============================================

There is no ``residual_delta_jax``: this package cannot import JAX. For a
linear model the traced form is the same matrix product, and nltiming's JAX
wrapper supplies it when NUTS needs it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .gauge import GaugeProvenance


@dataclass(frozen=True)
class LinearModel:
    """Linearized residual model around the reference theta."""

    fitpars: tuple[str, ...]
    design: np.ndarray
    theta_exact: Mapping[str, str]
    native_units: Mapping[str, str]

    @classmethod
    def from_design(
        cls,
        *,
        fitpars: tuple[str, ...],
        design: np.ndarray,
        theta_exact: Mapping[str, str] | None = None,
        native_units: Mapping[str, str] | None = None,
    ) -> "LinearModel":
        if theta_exact is None:
            theta_exact = {name: "0.0" for name in fitpars}
        if native_units is None:
            native_units = {name: "native" for name in fitpars}
        return cls(
            fitpars=tuple(fitpars),
            design=np.asarray(design, dtype=float),
            theta_exact=dict(theta_exact),
            native_units=dict(native_units),
        )

    def reference_theta(self) -> np.ndarray:
        with localcontext() as ctx:
            ctx.prec = 50
            return np.asarray(
                [float(Decimal(self.theta_exact[name])) for name in self.fitpars],
                dtype=float,
            )

    def residual_delta(self, delta_theta: np.ndarray) -> np.ndarray:
        delta = np.asarray(delta_theta, dtype=float)
        if delta.shape != (len(self.fitpars),):
            raise ValueError("delta_theta shape mismatch with fitpars")
        # Fitter sign: Δr ≈ -M δ
        return -(self.design @ delta)


@dataclass(frozen=True)
class LinearContribution:
    """One leg of a composite linear engine: its rows and its own linear leaf.

    The attribute names (``name``, ``row_indices``, ``engine``) are the ones
    nltiming reads off a composite's contributions; the two trailing fields
    exist so a reader written against MetaPulsar's ``PtaContribution`` finds
    them.
    """

    name: str
    row_indices: np.ndarray
    engine: "LinearTimingEngine"
    exact_linear_fitpars: frozenset = frozenset()
    fallback_reference_exact: Mapping[str, str] = field(default_factory=dict)


def _record_gauge_column(fitpars: tuple[str, ...], leg: str | None) -> str:
    """The named gauge column of a record, or of one leg of a composite one.

    A leg's column carries the leg's name as its suffix (``Offset_epta``,
    ``PHOFF_epta``); a single-leg record carries the bare name.
    """
    if leg is not None:
        candidates = (f"Offset_{leg}", f"PHOFF_{leg}")
        what = f"leg {leg!r}"
    else:
        candidates = ("Offset", "PHOFF")
        what = "the record"
    found = [name for name in candidates if name in fitpars]
    if len(found) != 1:
        raise ValueError(
            f"{what} must carry exactly one named gauge column among "
            f"{candidates}; found {found} in fitpars {list(fitpars)}"
        )
    return found[0]


class LinearTimingEngine:
    """``-Mmat @ delta`` over a :class:`LinearModel`, in nltiming's engine shape.

    With ``contributions`` it is a composite: ``residual_delta`` is still the
    one matrix product over the whole matrix, and the contributions exist so
    readers that want per-leg facts (nltiming's gauge check, its per-leg
    gauge provenance in the context and the manifest) find them. Such an
    engine has no gauge provenance of its own, exactly like MetaPulsar's live
    composite.
    """

    engine_name = "linear"

    #: The hybrid residual mode this engine executes: every axis linear.
    nonlinear_params = None

    def __init__(
        self,
        model: LinearModel,
        *,
        gauge_provenance: GaugeProvenance | None = None,
        contributions: list[LinearContribution] | None = None,
    ):
        if gauge_provenance is None and not contributions:
            raise TypeError(
                "LinearTimingEngine requires gauge_provenance unless it is a "
                "composite with contributions"
            )
        self._model = model
        self._gauge_provenance = gauge_provenance
        self.contributions = list(contributions) if contributions else None
        self.fitpars = model.fitpars
        self.native_units = dict(model.native_units)

    # --- construction from a record ------------------------------------

    @classmethod
    def from_pulsar_data(cls, record) -> "LinearTimingEngine":
        """The linear engine of a record; see :func:`linear_engine`."""
        return linear_engine(record)

    @classmethod
    def from_feather(cls, path) -> "LinearTimingEngine":
        """The same, from a schema-v1 feather file and nothing else."""
        from .feather import read

        return linear_engine(read(Path(path)))

    # --- TimingEngine ------------------------------------------------------

    def reference_theta(self) -> np.ndarray:
        return self._model.reference_theta()

    def reference_theta_exact(self) -> Mapping[str, str]:
        return dict(self._model.theta_exact)

    def residual_delta(self, delta_theta: np.ndarray) -> np.ndarray:
        return self._model.residual_delta(delta_theta)

    def design_matrix(self, params: Any | None = None) -> np.ndarray:
        _ = params
        return np.asarray(self._model.design, dtype=float)

    def gauge_provenance(self) -> GaugeProvenance:
        if self._gauge_provenance is None:
            raise AttributeError(
                "a composite LinearTimingEngine has no own gauge_provenance; "
                "read it per contribution"
            )
        return self._gauge_provenance

    @property
    def gauge_applied(self) -> bool:
        if self._gauge_provenance is None:
            # OR over leaves, as MetaPulsar's composite does: diagnostic only.
            return any(
                bool(getattr(c.engine, "gauge_applied", False))
                for c in self.contributions or ()
            )
        return self.gauge_provenance().export != "none"

    # --- JacobianTimingEngine ------------------------------------------------

    def residual_jacobian(self) -> np.ndarray:
        """J = -M for the linearized model."""
        return -np.asarray(self._model.design, dtype=float)

    # --- the optional hooks, each with its linear answer -------------------

    def identically_linear_fitpars(self) -> frozenset[str]:
        """A linear model is affine in every delta, so every fitpar qualifies."""
        return frozenset(self.fitpars)

    def binary_chart_capability(self, chart_family: str, suffix: str):
        """A linear model has no nonlinear binary map to chart: never a candidate."""
        _ = (chart_family, suffix)
        return None

    def __repr__(self) -> str:
        legs = "" if self.contributions is None else f", {len(self.contributions)} legs"
        return f"<psrdata.LinearTimingEngine {len(self.fitpars)} fitpars{legs}>"


class RecordLinearTimingEngine(LinearTimingEngine):
    """The linear engine of one record, or of one leg of a composite record.

    Its gauge direction is its own named gauge column. A record's ``Mmat`` is
    the producing engine's design matrix, so the column named ``Offset`` or
    ``PHOFF`` (suffixed on a leg) *is* the direction an unmeasurable phase
    offset moves this record's residual: exactly the constant vector for a
    PINT or tempo2 record, whose matrices divide by the constant ``F0``, and
    ``1/F(t_i)`` for a vela-jax record, whose matrix is ``-J`` of a residual
    divided by the spin Taylor series. The two differ by ``F1 * T / F0``,
    about 4e-8 on B1937+21 over twenty years, above the 1e-8 nltiming's
    gauge check lives at. Declaring the column lets that check test what is
    true of the record rather than the constant; a zero column still fails,
    because a zero direction is refused.
    """

    def __init__(
        self,
        model: LinearModel,
        *,
        gauge_provenance: GaugeProvenance,
        gauge_column: str,
    ):
        super().__init__(model, gauge_provenance=gauge_provenance)
        if gauge_column not in self.fitpars:
            raise ValueError(
                f"gauge column {gauge_column!r} is not among fitpars "
                f"{list(self.fitpars)}"
            )
        self.gauge_column = gauge_column

    def gauge_direction(self) -> np.ndarray:
        column = self.fitpars.index(self.gauge_column)
        return np.asarray(self._model.design[:, column], dtype=float)


def linear_engine(record) -> LinearTimingEngine:
    """The linear engine of a record, single-leg or composite.

    The engine is ``-Mmat @ delta`` over the record's own matrix either way;
    that matrix is all the linearized model is. A composite record
    (``timing_package == "composite"``) carries one gauge provenance per leg,
    keyed by leg name, and the matrix carries the rest: a leg's rows are the
    support of its named gauge column (``Offset_<leg>`` / ``PHOFF_<leg>``),
    and the parameters it owns are the columns nonzero on those rows. Each
    leg becomes a :class:`LinearContribution` with its own
    :class:`RecordLinearTimingEngine` leaf, so nltiming's per-contribution
    readers see the shape MetaPulsar's live composite gives them.
    """
    fitpars = tuple(record.fitpars)
    design = np.asarray(record.Mmat, dtype=float)
    theta_exact = dict(record.reference_theta_exact)
    native_units = dict(record.native_units)
    model = LinearModel.from_design(
        fitpars=fitpars,
        design=design,
        theta_exact=theta_exact,
        native_units=native_units,
    )
    if record.timing_package != "composite":
        return RecordLinearTimingEngine(
            model,
            gauge_provenance=GaugeProvenance.from_mapping(record.gauge),
            gauge_column=_record_gauge_column(fitpars, None),
        )

    n_toa, n_fit = design.shape
    contributions: list[LinearContribution] = []
    for leg, provenance in dict(record.gauge).items():
        column = _record_gauge_column(fitpars, leg)
        rows = np.flatnonzero(design[:, fitpars.index(column)])
        if rows.size == 0:
            raise ValueError(
                f"gauge column {column!r} of leg {leg!r} is numerically zero"
            )
        owned = [j for j in range(n_fit) if np.any(design[rows, j])]
        leaf_fitpars = tuple(fitpars[j] for j in owned)
        leaf = RecordLinearTimingEngine(
            LinearModel.from_design(
                fitpars=leaf_fitpars,
                design=design[np.ix_(rows, owned)],
                theta_exact={name: theta_exact[name] for name in leaf_fitpars},
                native_units={
                    name: native_units.get(name, "native") for name in leaf_fitpars
                },
            ),
            gauge_provenance=GaugeProvenance.from_mapping(provenance),
            gauge_column=column,
        )
        contributions.append(
            LinearContribution(name=str(leg), row_indices=rows, engine=leaf)
        )
    covered = np.sort(np.concatenate([c.row_indices for c in contributions]))
    if not np.array_equal(covered, np.arange(n_toa)):
        raise ValueError(
            "the legs' gauge columns must partition the rows: every row "
            "in exactly one leg's support"
        )
    return LinearTimingEngine(model, contributions=contributions)


__all__ = [
    "LinearModel",
    "LinearContribution",
    "LinearTimingEngine",
    "RecordLinearTimingEngine",
    "linear_engine",
]
