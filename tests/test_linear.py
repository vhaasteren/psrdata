"""The record's linear engine: SPEC §3.6, §5 (test obligations 4, 7, 8, 13)."""

from __future__ import annotations

import subprocess
import sys
from decimal import Decimal

import numpy as np
import pytest

import psrdata
from conftest import (
    COMBINED_FITPARS,
    COMBINED_KEYS,
    SINGLE_FITPARS,
    SINGLE_SETPARS,
    combined_kwargs,
    make_combined,
    make_record,
    record_kwargs,
)
from psrdata import (
    LinearEngineError,
    LinearTimingEngine,
    ParameterFact,
    PulsarData,
    linear_engine,
)

# --- §5.1 construction --------------------------------------------------------


def test_the_four_entry_points_agree(record, tmp_path):
    path = tmp_path / "r.feather"
    psrdata.feather.write(record, path)
    engines = [
        record.linear_engine(),
        linear_engine(record),
        LinearTimingEngine.from_pulsar_data(record),
        LinearTimingEngine.from_feather(path),
    ]
    delta = np.arange(len(record.fitpars), dtype=float)
    for engine in engines:
        assert isinstance(engine, LinearTimingEngine)
        assert engine.fitpars == record.fitpars
        assert np.allclose(engine.residual_delta(delta), -record.Mmat @ delta)


# --- §5.2 required behaviour --------------------------------------------------


def test_fitpars_units_and_references_come_from_the_facts(record):
    engine = record.linear_engine()
    assert engine.fitpars == SINGLE_FITPARS
    assert engine.native_units == {
        name: record.parameters[name].units for name in SINGLE_FITPARS
    }
    assert tuple(engine.native_units) == SINGLE_FITPARS
    assert engine.reference_theta_exact() == {
        name: record.parameters[name].value for name in SINGLE_FITPARS
    }
    assert tuple(engine.reference_theta_exact()) == SINGLE_FITPARS


def test_reference_theta_parses_decimals_without_a_float_round_trip():
    """R-5.2.4 / R-3.3.5: the float64 view is the correctly rounded value of
    the decimal string; the exact mapping keeps digits float64 cannot."""
    exact = "339.315687288152034567891234"
    kwargs = record_kwargs()
    parameters = dict(kwargs["parameters"])
    parameters["F0"] = ParameterFact(exact, "Hz")
    engine = make_record(parameters=parameters).linear_engine()
    theta = engine.reference_theta()
    assert theta.dtype == np.float64
    assert theta.shape == (len(SINGLE_FITPARS),)
    assert theta[0] == float(Decimal(exact))
    assert engine.reference_theta_exact()["F0"] == exact
    assert str(theta[0]) != exact


def test_residual_delta_is_minus_mmat_delta(record):
    engine = record.linear_engine()
    delta = np.array([1e-9, 1e-20, 1e-7, 2e-6])
    assert np.allclose(engine.residual_delta(delta), -record.Mmat @ delta)
    assert engine.residual_delta(delta).shape == (len(record.toas),)


@pytest.mark.parametrize("shape", [(3,), (5,), (4, 1), (1, 4), (), (2, 2)])
def test_the_wrong_delta_shape_is_refused(record, shape):
    engine = record.linear_engine()
    with pytest.raises(LinearEngineError):
        engine.residual_delta(np.zeros(shape))


def test_design_matrix_accepts_only_params_none(record):
    engine = record.linear_engine()
    assert engine.design_matrix() is record.Mmat
    assert engine.design_matrix(params=None) is record.Mmat
    with pytest.raises(LinearEngineError):
        engine.design_matrix(params={"F0": 1.0})
    with pytest.raises(LinearEngineError):
        engine.design_matrix(np.zeros(4))


def test_residual_jacobian_is_minus_mmat(record):
    engine = record.linear_engine()
    assert np.array_equal(engine.residual_jacobian(), -record.Mmat)


def test_every_fit_parameter_is_identically_linear(record):
    engine = record.linear_engine()
    assert tuple(engine.identically_linear_fitpars()) == record.fitpars


def test_no_nonlinear_surface(record):
    engine = record.linear_engine()
    assert engine.nonlinear_params is None
    assert engine.binary_chart_capability("ELL1") is None
    assert engine.binary_chart_capability() is None
    assert engine.engine_name == "linear"
    assert engine.timing_package == {"single": "tempo2"}
    assert engine.partim_compatibility == {"single": "tempo2"}


def test_the_engine_imports_no_jax():
    code = (
        "import sys, psrdata.linear, psrdata.feather, psrdata.record; "
        "assert not [m for m in sys.modules if m == 'jax' or m.startswith('jax.')]"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# --- obligation 4: PINT-unit matrix calculations -------------------------------


def test_matrix_columns_are_seconds_per_pint_unit(record):
    """Column j is s per ``parameters[fitpars[j]].units``: a unit delta on
    coordinate j moves the residuals by minus that column, and the unit label
    the engine reports for j is the fact's PINT unit, not a heuristic."""
    engine = record.linear_engine()
    p = len(record.fitpars)
    for j, name in enumerate(record.fitpars):
        e_j = np.zeros(p)
        e_j[j] = 1.0
        assert np.array_equal(engine.residual_delta(e_j), -record.Mmat[:, j])
    assert engine.native_units["F0"] == "Hz"
    assert engine.native_units["F1"] == "Hz / s"
    assert engine.native_units["RAJ"] == "hourangle"
    assert engine.native_units["Offset"] == "s"


# --- §3.6 phase-offset columns ------------------------------------------------


def _single_with_offsets(names, units):
    """A single-data-set record whose phase-offset fit parameters are ``names``."""
    kwargs = record_kwargs()
    base_fit = tuple(p for p in SINGLE_FITPARS if p != "Offset")
    base_set = tuple(p for p in SINGLE_SETPARS if p != "Offset")
    parameters = {k: v for k, v in kwargs["parameters"].items() if k != "Offset"}
    Mmat = kwargs["Mmat"][:, :-1]
    for name, unit in zip(names, units):
        parameters[name] = ParameterFact("0", unit)
        Mmat = np.hstack([Mmat, np.ones((Mmat.shape[0], 1))])
    kwargs.update(
        fitpars=base_fit + tuple(names),
        setpars=base_set + tuple(names),
        parameters=parameters,
        Mmat=Mmat,
    )
    return PulsarData(**kwargs)


@pytest.mark.parametrize(
    "name, unit",
    [
        ("Offset", "s"),
        ("PHOFF", "dimensionless"),
        ("Offset_single", "s"),
        ("PHOFF_single", "dimensionless"),
    ],
)
def test_a_single_data_set_may_use_a_bare_or_suffixed_name(name, unit):
    engine = _single_with_offsets([name], [unit]).linear_engine()
    (contribution,) = engine.contributions().values()
    assert contribution.key == "single"
    assert contribution.phase_offset == name


def test_no_phase_offset_column_is_refused():
    rec = _single_with_offsets([], [])
    with pytest.raises(LinearEngineError, match="phase-offset"):
        rec.linear_engine()


@pytest.mark.parametrize(
    "names, units",
    [
        (("Offset", "PHOFF"), ("s", "dimensionless")),
        (("Offset", "Offset_single"), ("s", "s")),
        (("PHOFF", "PHOFF_single"), ("dimensionless", "dimensionless")),
    ],
)
def test_two_candidates_for_one_data_set_are_refused(names, units):
    rec = _single_with_offsets(names, units)
    with pytest.raises(LinearEngineError, match="phase-offset"):
        rec.linear_engine()


def test_a_combined_record_never_uses_a_bare_name():
    kwargs, _ = combined_kwargs()
    fitpars = tuple("Offset" if p == "Offset_EPTA_DR2" else p for p in COMBINED_FITPARS)
    parameters = dict(kwargs["parameters"])
    parameters["Offset"] = parameters.pop("Offset_EPTA_DR2")
    rec = make_combined(
        fitpars=fitpars,
        setpars=fitpars + ("PB", "PSR"),
        parameters=parameters,
    )
    with pytest.raises(LinearEngineError, match="EPTA_DR2"):
        rec.linear_engine()


def test_a_combined_record_with_both_forms_for_one_key_is_refused():
    kwargs, _ = combined_kwargs()
    Mmat = np.hstack([kwargs["Mmat"], kwargs["Mmat"][:, 2:3]])
    parameters = dict(kwargs["parameters"])
    parameters["PHOFF_EPTA_DR2"] = ParameterFact("0", "dimensionless")
    rec = make_combined(
        fitpars=COMBINED_FITPARS + ("PHOFF_EPTA_DR2",),
        setpars=COMBINED_FITPARS + ("PHOFF_EPTA_DR2", "PB", "PSR"),
        parameters=parameters,
        Mmat=Mmat,
    )
    with pytest.raises(LinearEngineError, match="EPTA_DR2"):
        rec.linear_engine()


def test_overlapping_supports_are_refused():
    kwargs, membership = combined_kwargs()
    Mmat = kwargs["Mmat"].copy()
    row = int(np.flatnonzero(membership == 1)[0])
    Mmat[row, COMBINED_FITPARS.index("Offset_EPTA_DR2")] = 1.0
    with pytest.raises(LinearEngineError, match="partition"):
        make_combined(Mmat=Mmat).linear_engine()


def test_a_gap_in_the_supports_is_refused():
    kwargs, membership = combined_kwargs()
    Mmat = kwargs["Mmat"].copy()
    row = int(np.flatnonzero(membership == 1)[0])
    Mmat[row, COMBINED_FITPARS.index("PHOFF_PPTA_DR3")] = 0.0
    with pytest.raises(LinearEngineError, match="partition"):
        make_combined(Mmat=Mmat).linear_engine()


def test_a_single_record_gap_is_refused():
    kwargs = record_kwargs()
    Mmat = kwargs["Mmat"].copy()
    Mmat[4, SINGLE_FITPARS.index("Offset")] = 0.0
    with pytest.raises(LinearEngineError, match="partition"):
        make_record(Mmat=Mmat).linear_engine()


# --- §5.3 per-data-set contributions -------------------------------------------


def test_contributions_partition_the_rows(combined_membership):
    rec, membership = combined_membership
    contributions = rec.linear_engine().contributions()
    assert tuple(contributions) == COMBINED_KEYS
    for i, key in enumerate(COMBINED_KEYS):
        assert np.array_equal(contributions[key].rows, np.flatnonzero(membership == i))
    all_rows = np.concatenate([c.rows for c in contributions.values()])
    assert np.array_equal(np.sort(all_rows), np.arange(len(rec.toas)))


def test_a_contributions_fit_parameters_are_its_active_columns(combined):
    contributions = combined.linear_engine().contributions()
    epta = contributions["EPTA_DR2"]
    ppta = contributions["PPTA_DR3"]
    assert epta.fitpars == ("F0", "F1", "Offset_EPTA_DR2")
    assert ppta.fitpars == ("F0", "F1", "DM_PPTA_DR3", "PHOFF_PPTA_DR3")
    assert epta.phase_offset == "Offset_EPTA_DR2"
    assert ppta.phase_offset == "PHOFF_PPTA_DR3"
    for c in (epta, ppta):
        assert tuple(c.identically_linear_fitpars()) == c.fitpars
        assert tuple(c.linear_fitpars) == c.fitpars
        assert c.design_matrix().shape == (len(c.rows), len(c.fitpars))
        assert np.array_equal(
            c.design_matrix(), combined.Mmat[np.ix_(c.rows, c.column_indices)]
        )
        assert tuple(c.native_units) == c.fitpars
        assert tuple(c.reference_theta_exact()) == c.fitpars
        assert c.engine_name == "linear"
        assert c.nonlinear_params is None


def test_contributions_reassemble_the_whole_product(combined):
    """R-5.3.3."""
    engine = combined.linear_engine()
    rng = np.random.default_rng(7)
    delta = rng.normal(size=len(combined.fitpars))
    whole = engine.residual_delta(delta)
    assembled = np.full_like(whole, np.nan)
    for c in engine.contributions().values():
        assembled[c.rows] = c.residual_delta(delta[list(c.column_indices)])
    assert np.allclose(assembled, whole)
    assert not np.isnan(assembled).any()


def test_a_single_record_has_one_contribution_over_every_row(record):
    engine = record.linear_engine()
    (c,) = engine.contributions().values()
    assert np.array_equal(c.rows, np.arange(len(record.toas)))
    assert c.fitpars == record.fitpars
    assert np.array_equal(c.design_matrix(), record.Mmat)


def test_residual_centering_is_a_mapping_for_one_or_several(record, combined):
    """R-5.3.4: nothing raises merely because there are several data sets."""
    single = record.linear_engine()
    assert tuple(single.residual_centering) == ("single",)
    several = combined.linear_engine()
    assert tuple(several.residual_centering) == COMBINED_KEYS
    for key, c in several.contributions().items():
        assert c.residual_centering == combined.residual_centering[key]
        assert c.timing_package == combined.timing_package[key]
        assert c.partim_compatibility == combined.partim_compatibility[key]


# --- obligation 13: mixed compatibilities in PINT-unit coordinates ------------


def test_mixed_pint_and_tempo2_data_sets_share_one_pint_unit_coordinate(combined):
    """A PINT-read and a tempo2-read data set meet in one global column whose
    unit is the fact's PINT unit; each contribution reports that same unit and
    the same reference value for the shared parameter."""
    assert combined.partim_compatibility == {"EPTA_DR2": "pint", "PPTA_DR3": "tempo2"}
    engine = combined.linear_engine()
    assert engine.native_units["F0"] == "Hz"
    for c in engine.contributions().values():
        assert c.native_units["F0"] == "Hz"
        assert c.reference_theta_exact()["F0"] == combined.parameters["F0"].value
    j = combined.fitpars.index("F0")
    e = np.zeros(len(combined.fitpars))
    e[j] = 1e-9
    assert np.allclose(engine.residual_delta(e), -1e-9 * combined.Mmat[:, j])
