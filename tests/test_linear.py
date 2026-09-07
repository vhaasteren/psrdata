"""The record's own linear engine, single-leg and composite; no timing package.

The engine is ``-Mmat @ delta`` over the record's matrix. A composite record
declares no partition: a leg's rows are the support of its named gauge
column, and the parameters it owns are the columns nonzero on those rows.
The ``consumers`` tests at the end run nltiming's own protocol checks and
its gauge check against the engine, which is how conformance to a protocol
this package does not import is asserted.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from psrdata import (
    GaugeProvenance,
    LinearContribution,
    LinearModel,
    LinearTimingEngine,
    PulsarData,
    RecordLinearTimingEngine,
)

_GAUGE_FREE = {
    "export": "none",
    "reference_mode": "none",
    "reporting_mode": "mean",
    "reporting_weighted": True,
}


def _with_matrix(record, fitpars, design, *, gauge, timing_package):
    """The fixture record with a different matrix, fitpars and gauge."""
    n = design.shape[0]
    rng = np.random.default_rng(1)
    planetssb = np.full((n, 9, 6), np.nan)
    return dataclasses.replace(
        record,
        fitpars=tuple(fitpars),
        toas=np.linspace(5e9, 5e9 + 6e8, n),
        stoas=np.linspace(5e9, 5e9 + 6e8, n),
        toaerrs=np.full(n, 1e-6),
        residuals=rng.normal(size=n) * 1e-6,
        freqs=np.full(n, 1400.0),
        Mmat=np.asarray(design, dtype=float),
        flags={},
        backend_flags=np.array([""] * n),
        telescope=np.array(["ao"] * n),
        pos_t=np.tile([0.6, 0.8, 0.0], (n, 1)),
        sunssb=np.zeros((n, 6)),
        planetssb=planetssb,
        timing_package=timing_package,
        gauge=gauge,
        reference_theta_exact={name: "1.0" for name in fitpars},
        native_units={name: "native" for name in fitpars},
    )


@pytest.fixture
def composite(record):
    """Two legs, one shared F0, one local parameter each, per-leg offsets."""
    n_epta, n_ppta = 6, 4
    n = n_epta + n_ppta
    epta = np.arange(n_epta)
    ppta = np.arange(n_epta, n)
    fitpars = ("F0", "F1_epta", "Offset_epta", "Offset_ppta", "JUMP1_ppta")
    M = np.zeros((n, len(fitpars)))
    M[:, 0] = np.linspace(1.0, 2.0, n)
    M[epta, 1] = np.linspace(0.1, 0.5, n_epta)
    M[epta, 2] = 1.0
    M[ppta, 3] = 1.0
    M[ppta, 4] = 1.0
    gauge = {
        "epta": dict(_GAUGE_FREE),
        "ppta": {**_GAUGE_FREE, "reporting_weighted": False},
    }
    return _with_matrix(record, fitpars, M, gauge=gauge, timing_package="composite")


@pytest.fixture
def single(record):
    """A vela-jax-like record: the PHOFF column is 1/F(t), drifting 1e-7."""
    n = 24
    phoff = 1.0 / (300.0 * (1.0 + 1e-7 * np.linspace(0.0, 1.0, n)))
    M = np.column_stack([np.linspace(1.0, 2.0, n), phoff])
    return _with_matrix(
        record, ("F0", "PHOFF"), M, gauge=dict(_GAUGE_FREE), timing_package="pint"
    )


# --- the linear model ------------------------------------------------------


def test_the_engine_is_minus_m_delta(single):
    engine = single.linear_engine()
    delta = np.array([1e-3, -2e-3])
    np.testing.assert_allclose(engine.residual_delta(delta), -(single.Mmat @ delta))
    np.testing.assert_array_equal(engine.design_matrix(), single.Mmat)
    np.testing.assert_array_equal(engine.residual_jacobian(), -single.Mmat)
    assert engine.residual_delta(np.zeros(2)).max() == 0.0
    np.testing.assert_array_equal(engine.reference_theta(), [1.0, 1.0])
    assert engine.reference_theta_exact() == {"F0": "1.0", "PHOFF": "1.0"}


def test_the_reference_is_parsed_exactly():
    model = LinearModel.from_design(
        fitpars=("F0",),
        design=np.ones((3, 1)),
        theta_exact={"F0": "339.31568728824099254"},
    )
    assert model.reference_theta()[0] == float("339.31568728824099254")


def test_a_wrong_length_delta_is_refused(single):
    with pytest.raises(ValueError, match="shape mismatch"):
        single.linear_engine().residual_delta(np.zeros(3))


def test_every_optional_hook_has_its_linear_answer(single):
    engine = single.linear_engine()
    assert engine.identically_linear_fitpars() == frozenset(single.fitpars)
    assert engine.nonlinear_params is None
    assert engine.binary_chart_capability("kepler_laplace", "") is None
    assert engine.contributions is None
    assert engine.gauge_applied is False
    assert engine.gauge_provenance() == GaugeProvenance(**_GAUGE_FREE)


# --- the gauge direction is the record's own column -------------------------


def test_a_single_leg_record_declares_its_gauge_column(single):
    engine = single.linear_engine()
    assert isinstance(engine, RecordLinearTimingEngine)
    assert engine.gauge_column == "PHOFF"
    np.testing.assert_array_equal(engine.gauge_direction(), single.Mmat[:, 1])


def test_a_record_without_a_named_gauge_column_is_refused(record):
    M = np.column_stack([np.linspace(1.0, 2.0, 5), np.ones(5)])
    broken = _with_matrix(
        record, ("F0", "DM"), M, gauge=dict(_GAUGE_FREE), timing_package="pint"
    )
    with pytest.raises(ValueError, match="exactly one named gauge column"):
        broken.linear_engine()


def test_two_gauge_columns_are_refused(record):
    M = np.ones((5, 3))
    broken = _with_matrix(
        record,
        ("F0", "Offset", "PHOFF"),
        M,
        gauge=dict(_GAUGE_FREE),
        timing_package="pint",
    )
    with pytest.raises(ValueError, match="exactly one named gauge column"):
        broken.linear_engine()


# --- composite: the partition is in the matrix -------------------------------


def test_a_composite_record_is_its_own_linear_engine(composite):
    engine = composite.linear_engine()
    assert isinstance(engine, LinearTimingEngine)
    delta = np.array([1e-3, 2e-3, -1e-3, 4e-3, 5e-3])
    np.testing.assert_allclose(engine.residual_delta(delta), -(composite.Mmat @ delta))

    by_name = {c.name: c for c in engine.contributions}
    assert set(by_name) == {"epta", "ppta"}
    assert all(isinstance(c, LinearContribution) for c in engine.contributions)
    np.testing.assert_array_equal(by_name["epta"].row_indices, np.arange(6))
    np.testing.assert_array_equal(by_name["ppta"].row_indices, np.arange(6, 10))
    # Ownership is read off the matrix: nonzero on the leg's rows.
    assert by_name["epta"].engine.fitpars == ("F0", "F1_epta", "Offset_epta")
    assert by_name["ppta"].engine.fitpars == ("F0", "Offset_ppta", "JUMP1_ppta")
    assert by_name["epta"].engine.gauge_column == "Offset_epta"
    np.testing.assert_array_equal(by_name["ppta"].engine.gauge_direction(), np.ones(4))


def test_a_composite_has_one_provenance_per_leg_and_none_of_its_own(composite):
    engine = composite.linear_engine()
    by_name = {c.name: c.engine.gauge_provenance() for c in engine.contributions}
    assert by_name["epta"].reporting_weighted is True
    assert by_name["ppta"].reporting_weighted is False
    with pytest.raises(AttributeError, match="per contribution"):
        engine.gauge_provenance()
    assert engine.gauge_applied is False


def test_a_leaf_is_the_block_of_the_whole(composite):
    engine = composite.linear_engine()
    delta = np.array([1e-3, 2e-3, -1e-3, 4e-3, 5e-3])
    whole = engine.residual_delta(delta)
    for contribution in engine.contributions:
        leaf = contribution.engine
        idx = [composite.fitpars.index(name) for name in leaf.fitpars]
        np.testing.assert_allclose(
            leaf.residual_delta(delta[idx]), whole[contribution.row_indices]
        )


def test_a_leg_without_its_own_gauge_column_is_refused(composite):
    fitpars = tuple(
        name.replace("Offset_ppta", "Offset_other") for name in composite.fitpars
    )
    broken = dataclasses.replace(
        composite,
        fitpars=fitpars,
        reference_theta_exact={name: "1.0" for name in fitpars},
        native_units={name: "native" for name in fitpars},
    )
    with pytest.raises(ValueError, match="leg 'ppta' must carry exactly one"):
        broken.linear_engine()


def test_legs_whose_gauge_columns_overlap_are_refused(composite):
    M = np.array(composite.Mmat)
    M[:, 3] = 1.0  # Offset_ppta now claims every row
    with pytest.raises(ValueError, match="partition the rows"):
        dataclasses.replace(composite, Mmat=M).linear_engine()


def test_a_zero_gauge_column_is_refused(composite):
    M = np.array(composite.Mmat)
    M[:, 3] = 0.0
    with pytest.raises(ValueError, match="numerically zero"):
        dataclasses.replace(composite, Mmat=M).linear_engine()


def test_the_engine_survives_the_feather(composite, tmp_path):
    path = composite.to_feather(tmp_path / "composite.feather")
    engine = LinearTimingEngine.from_feather(path)
    by_name = {c.name: c for c in engine.contributions}
    np.testing.assert_array_equal(by_name["epta"].row_indices, np.arange(6))
    delta = np.ones(len(composite.fitpars)) * 1e-3
    np.testing.assert_allclose(engine.residual_delta(delta), -(composite.Mmat @ delta))
    assert LinearTimingEngine.from_pulsar_data(
        PulsarData.from_feather(path)
    ).fitpars == (composite.fitpars)


# --- conformance to the consumer's protocol, asserted by the consumer ---------


@pytest.mark.consumers
def test_nltiming_accepts_it_as_a_timing_engine(single, composite):
    """Structural conformance, checked with nltiming's own protocols."""
    protocols = pytest.importorskip("nltiming.protocols")
    for record in (single, composite):
        engine = record.linear_engine()
        assert isinstance(engine, protocols.TimingEngine)
        assert isinstance(engine, protocols.JacobianTimingEngine)
        assert not isinstance(engine, protocols.JaxTimingEngine)
        for contribution in engine.contributions or ():
            assert isinstance(contribution.engine, protocols.JacobianTimingEngine)


@pytest.mark.consumers
def test_nltiming_gauge_check_passes_and_reads_one_provenance_per_leg(
    single, composite
):
    ntm = pytest.importorskip("nltiming.nonlinear_timing_model")
    for record in (single, composite):
        engine = record.linear_engine()
        ntm.assert_gauge_column_present(record, engine, np.asarray(record.Mmat))
    provenance = dict(
        ntm._normalize_gauge_provenance(composite, composite.linear_engine())
    )
    assert set(provenance) == {"epta", "ppta"}

    # The declared direction is what closes the high-F1 case: the same
    # matrix under an engine that declares nothing is held to the constant.
    silent = LinearTimingEngine(
        LinearModel.from_design(fitpars=single.fitpars, design=single.Mmat),
        gauge_provenance=GaugeProvenance(**_GAUGE_FREE),
    )
    with pytest.raises(ntm.GaugeColumnMissingError, match="constant direction"):
        ntm.assert_gauge_column_present(single, silent, np.asarray(single.Mmat))
