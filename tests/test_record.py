"""Structural validation of the record: SPEC §3.2, §3.3, §3.4, §3.8.

Every check here is one psrdata can make from the arrays and mappings it is
handed. What the values *mean* is the producer's obligation (§9) and is not
tested here.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import (
    COMBINED_KEYS,
    N_TOAS,
    SINGLE_FITPARS,
    SINGLE_SETPARS,
    combined_kwargs,
    make_combined,
    make_record,
    record_kwargs,
)
from psrdata import ParameterFact, PulsarData, RecordError, ResidualCentering, TOARows

# --- R-3.2.1 shapes ----------------------------------------------------------


@pytest.mark.parametrize("field", ["toas", "stoas", "toaerrs", "residuals", "freqs"])
def test_row_float_fields_must_have_n_rows(field):
    kwargs = record_kwargs()
    with pytest.raises(RecordError, match=field):
        make_record(**{field: kwargs[field][:-1]})
    with pytest.raises(RecordError, match=field):
        make_record(**{field: kwargs[field][:, None]})


@pytest.mark.parametrize("field", ["backend_flags", "telescope"])
def test_row_string_fields_must_have_n_rows(field):
    kwargs = record_kwargs()
    with pytest.raises(RecordError, match=field):
        make_record(**{field: kwargs[field][:-1]})


def test_mmat_must_be_n_by_p():
    kwargs = record_kwargs()
    with pytest.raises(RecordError, match="Mmat"):
        make_record(Mmat=kwargs["Mmat"][:-1])
    with pytest.raises(RecordError, match="Mmat"):
        make_record(Mmat=kwargs["Mmat"][:, :-1])
    with pytest.raises(RecordError, match="Mmat"):
        make_record(Mmat=kwargs["Mmat"].ravel())


@pytest.mark.parametrize(
    "field, bad_shape",
    [
        ("pos", (2,)),
        ("pos", (3, 1)),
        ("pos_t", (N_TOAS, 2)),
        ("pos_t", (N_TOAS - 1, 3)),
        ("sunssb", (N_TOAS, 3)),
        ("sunssb", (N_TOAS + 1, 6)),
        ("planetssb", (N_TOAS, 8, 6)),
        ("planetssb", (N_TOAS, 9, 3)),
        ("planetssb", (N_TOAS, 54)),
    ],
)
def test_vector_and_tensor_shapes(field, bad_shape):
    with pytest.raises(RecordError, match=field):
        make_record(**{field: np.zeros(bad_shape)})


def test_every_flag_value_must_have_n_rows():
    with pytest.raises(RecordError, match="flag"):
        make_record(flags={"f": np.array(["a"] * (N_TOAS - 1))})


# --- R-3.2.2 parameter names --------------------------------------------------


def test_duplicate_setpars_refused():
    with pytest.raises(RecordError, match="setpars"):
        make_record(setpars=SINGLE_SETPARS + ("F0",))


def test_duplicate_fitpars_refused():
    kwargs = record_kwargs()
    fitpars = SINGLE_FITPARS + ("F0",)
    Mmat = np.hstack([kwargs["Mmat"], kwargs["Mmat"][:, :1]])
    with pytest.raises(RecordError, match="fitpars"):
        make_record(fitpars=fitpars, Mmat=Mmat)


@pytest.mark.parametrize("field", ["setpars", "fitpars"])
@pytest.mark.parametrize("bad_name", [7, "", None])
def test_parameter_names_must_be_nonempty_strings(field, bad_name):
    kwargs = record_kwargs()
    names = list(kwargs[field])
    names[0] = bad_name
    updates = {field: tuple(names)}
    if field == "fitpars":
        parameters = dict(kwargs["parameters"])
        parameters[bad_name] = parameters["F0"]
        updates["parameters"] = parameters
    with pytest.raises(RecordError, match=field):
        make_record(**updates)


def test_every_fitpar_must_be_a_setpar():
    with pytest.raises(RecordError, match="F0"):
        make_record(setpars=tuple(p for p in SINGLE_SETPARS if p != "F0"))


def test_every_setpar_needs_a_parameter_fact():
    kwargs = record_kwargs()
    parameters = dict(kwargs["parameters"])
    del parameters["PB"]
    with pytest.raises(RecordError, match="PB"):
        make_record(parameters=parameters)


def test_a_fact_for_a_name_outside_setpars_is_refused():
    """``parameters`` may not carry names that are not set (R-3.3.4)."""
    kwargs = record_kwargs()
    parameters = dict(kwargs["parameters"])
    parameters["DM"] = ParameterFact("10.39", "pc / cm3")
    with pytest.raises(RecordError, match="DM"):
        make_record(parameters=parameters)


@pytest.mark.parametrize(
    "name, fact",
    [
        ("Offset", ParameterFact("0", "dimensionless")),
        ("Offset", ParameterFact("1", "s")),
        ("Offset", ParameterFact("0.0000000000000001", "s")),
        ("Offset", ParameterFact("0", "s", "0.1")),
        ("PHOFF", ParameterFact("0", "s")),
        ("Offset_x", ParameterFact("0", "dimensionless")),
        ("PHOFF_x", ParameterFact("0", "s")),
    ],
)
def test_phase_offset_facts_are_zero_unitted_and_uncertaintyless(name, fact):
    kwargs = record_kwargs()
    parameters = dict(kwargs["parameters"])
    setpars = tuple(p for p in SINGLE_SETPARS if p != "Offset") + (name,)
    fitpars = tuple(p for p in SINGLE_FITPARS if p != "Offset") + (name,)
    del parameters["Offset"]
    parameters[name] = fact
    with pytest.raises(RecordError, match=name):
        make_record(setpars=setpars, fitpars=fitpars, parameters=parameters)


@pytest.mark.parametrize(
    "name, fact",
    [
        ("Offset", ParameterFact("0", "s")),
        ("Offset", ParameterFact("0.0", "s")),
        ("Offset", ParameterFact("0.00", "s")),
        ("Offset", ParameterFact("0E0", "s")),
        ("Offset", ParameterFact("+0", "s")),
        ("PHOFF", ParameterFact("0", "dimensionless")),
        ("PHOFF", ParameterFact("0.0", "dimensionless")),
        ("Offset_EPTA_DR2", ParameterFact("0", "s")),
        ("PHOFF_EPTA_DR2", ParameterFact("0E-10", "dimensionless")),
    ],
)
def test_conforming_phase_offset_facts_are_accepted(name, fact):
    kwargs = record_kwargs()
    parameters = dict(kwargs["parameters"])
    setpars = tuple(p for p in SINGLE_SETPARS if p != "Offset") + (name,)
    fitpars = tuple(p for p in SINGLE_FITPARS if p != "Offset") + (name,)
    del parameters["Offset"]
    parameters[name] = fact
    rec = make_record(setpars=setpars, fitpars=fitpars, parameters=parameters)
    assert rec.parameters[name] == fact
    assert rec.parameters[name].value == fact.value


def test_a_frozen_phase_offset_in_setpars_is_not_required_to_be_zero():
    kwargs = record_kwargs()
    parameters = dict(kwargs["parameters"])
    parameters["PHOFF"] = ParameterFact("0.001", "dimensionless")
    rec = make_record(setpars=SINGLE_SETPARS + ("PHOFF",), parameters=parameters)
    assert rec.parameters["PHOFF"].value == "0.001"
    assert "Offset" in rec.fitpars
    assert "PHOFF" not in rec.fitpars


def test_a_fit_parameter_needs_a_unit_and_a_decimal_value():
    """A fit parameter's column has a unit (R-3.3.2), so a textual or unitless
    fit parameter is refused."""
    kwargs = record_kwargs()
    parameters = dict(kwargs["parameters"])
    parameters["F0"] = ParameterFact("339.3", None)
    with pytest.raises(RecordError, match="F0"):
        make_record(parameters=parameters)
    parameters["F0"] = ParameterFact("three hundred", "Hz")
    with pytest.raises(RecordError, match="F0"):
        make_record(parameters=parameters)


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "not-a-number"])
def test_every_numerical_parameter_value_must_be_a_finite_decimal(value):
    kwargs = record_kwargs()
    parameters = dict(kwargs["parameters"])
    parameters["PB"] = ParameterFact(value, "d")
    with pytest.raises(RecordError, match="PB"):
        make_record(parameters=parameters)


@pytest.mark.parametrize("uncertainty", ["NaN", "Infinity", "not-a-number"])
def test_every_parameter_uncertainty_must_be_a_finite_decimal(uncertainty):
    kwargs = record_kwargs()
    parameters = dict(kwargs["parameters"])
    parameters["PB"] = ParameterFact("1.53", "d", uncertainty)
    with pytest.raises(RecordError, match="PB"):
        make_record(parameters=parameters)


def test_a_textual_set_parameter_may_have_no_unit(record):
    assert record.parameters["PSR"] == ParameterFact("J1909-3744", None, None)


# --- R-3.2.3 mapping keys -----------------------------------------------------


def test_scalars_normalize_to_the_single_key(record):
    assert record.timing_package == {"single": "tempo2"}
    assert record.partim_compatibility == {"single": "tempo2"}
    assert list(record.residual_centering) == ["single"]
    assert isinstance(record.residual_centering["single"], ResidualCentering)


def test_named_mappings_are_kept_as_given(combined):
    assert tuple(combined.timing_package) == COMBINED_KEYS
    assert tuple(combined.partim_compatibility) == COMBINED_KEYS
    assert tuple(combined.residual_centering) == COMBINED_KEYS
    assert combined.timing_package == {"EPTA_DR2": "pint", "PPTA_DR3": "vela_jax"}


def test_a_scalar_for_only_some_of_the_three_is_refused():
    with pytest.raises(RecordError, match="scalar"):
        make_record(timing_package={"single": "tempo2"})
    with pytest.raises(RecordError, match="scalar"):
        make_combined(timing_package="pint")


def test_mapping_keys_must_agree():
    with pytest.raises(RecordError, match="keys"):
        make_combined(timing_package={"EPTA_DR2": "pint", "NANOGrav": "pint"})
    with pytest.raises(RecordError, match="keys"):
        make_combined(partim_compatibility={"EPTA_DR2": "pint"})
    with pytest.raises(RecordError, match="keys"):
        make_combined(
            residual_centering={
                "EPTA_DR2": ResidualCentering("none"),
                "PPTA_DR3": ResidualCentering("none"),
                "extra": ResidualCentering("none"),
            }
        )


def test_empty_mappings_are_refused():
    with pytest.raises(RecordError):
        make_combined(timing_package={}, partim_compatibility={}, residual_centering={})


def test_mapping_values_are_typed():
    with pytest.raises(RecordError, match="partim_compatibility"):
        make_combined(partim_compatibility={"EPTA_DR2": "pint", "PPTA_DR3": "jug"})
    with pytest.raises(RecordError, match="residual_centering"):
        make_combined(
            residual_centering={"EPTA_DR2": "none", "PPTA_DR3": "none"},
        )
    with pytest.raises(RecordError, match="timing_package"):
        make_combined(timing_package={"EPTA_DR2": "", "PPTA_DR3": "vela_jax"})


@pytest.mark.parametrize("value", ["pint", "tempo2"])
def test_partim_compatibility_literals(value):
    assert make_record(partim_compatibility=value).partim_compatibility == {
        "single": value
    }


def test_mappings_are_stored_in_one_shared_key_order():
    """Same keys in a different insertion order are accepted and re-ordered to
    ``timing_package``'s order, so the Feather writer's "same keys in the same
    insertion order" (R-6.2.0) holds by construction."""
    kwargs, _ = combined_kwargs()
    reversed_pc = dict(reversed(list(kwargs["partim_compatibility"].items())))
    rec = make_combined(partim_compatibility=reversed_pc)
    assert tuple(rec.partim_compatibility) == tuple(rec.timing_package)


# --- R-3.2.4 flags ------------------------------------------------------------


def test_flag_keys_are_nonempty_strings():
    with pytest.raises(RecordError, match="flag"):
        make_record(flags={"": np.array(["a"] * N_TOAS)})
    with pytest.raises(RecordError, match="flag"):
        make_record(flags={3: np.array(["a"] * N_TOAS)})


def test_flag_values_are_strings():
    with pytest.raises(RecordError, match="flag"):
        make_record(flags={"f": np.arange(N_TOAS)})
    with pytest.raises(RecordError, match="flag"):
        make_record(flags={"f": np.array([1.0] * N_TOAS, dtype=object)})


def test_object_arrays_of_str_are_accepted_as_flags():
    rec = make_record(flags={"f": np.array(["a"] * N_TOAS, dtype=object)})
    assert rec.flags["f"].tolist() == ["a"] * N_TOAS


def test_backend_flags_and_telescope_are_strings():
    with pytest.raises(RecordError, match="backend_flags"):
        make_record(backend_flags=np.arange(N_TOAS))
    with pytest.raises(RecordError, match="telescope"):
        make_record(telescope=np.arange(N_TOAS, dtype=float))


# --- R-3.2.5 finite values ----------------------------------------------------


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize(
    "field", ["toas", "stoas", "toaerrs", "residuals", "Mmat", "pos", "pos_t"]
)
def test_row_and_geometry_fields_must_be_finite(field, bad):
    kwargs = record_kwargs()
    arr = kwargs[field].copy()
    arr.flat[0] = bad
    with pytest.raises(RecordError, match=field):
        make_record(**{field: arr})


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_sunssb_position_must_be_finite(bad):
    kwargs = record_kwargs()
    sunssb = kwargs["sunssb"].copy()
    sunssb[0, 1] = bad
    with pytest.raises(RecordError, match="sunssb position"):
        make_record(sunssb=sunssb)


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_sunssb_velocity_may_be_nan_but_not_inf(bad):
    kwargs = record_kwargs()
    sunssb = kwargs["sunssb"].copy()
    sunssb[:, 3:] = np.nan
    rec = make_record(sunssb=sunssb)
    assert np.isnan(rec.sunssb[:, 3:]).all()
    sunssb[0, 4] = bad
    with pytest.raises(RecordError, match="sunssb unused velocity"):
        make_record(sunssb=sunssb)


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_planetssb_may_be_nan_but_not_inf(bad):
    kwargs = record_kwargs()
    planetssb = kwargs["planetssb"].copy()
    planetssb[0, 0, 0] = bad
    with pytest.raises(RecordError, match="planetssb"):
        make_record(planetssb=planetssb)


@pytest.mark.parametrize("field", ["theta", "phi", "dm"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_scalar_geometry_must_be_finite(field, bad):
    with pytest.raises(RecordError, match=field):
        make_record(**{field: bad})


@pytest.mark.parametrize("pdist", [(np.nan, 0.2), (1.0, np.inf), (-np.inf, 0.2)])
def test_pdist_must_be_finite(pdist):
    with pytest.raises(RecordError, match="pdist"):
        make_record(pdist=pdist)


@pytest.mark.parametrize("bad", [np.nan, -np.inf])
def test_freqs_refuse_nan_and_negative_inf(bad):
    kwargs = record_kwargs()
    freqs = kwargs["freqs"].copy()
    freqs[2] = bad
    with pytest.raises(RecordError, match="freqs"):
        make_record(freqs=freqs)


def test_freqs_may_be_positive_inf_and_planet_slots_may_be_nan():
    kwargs = record_kwargs()
    freqs = kwargs["freqs"].copy()
    freqs[2] = np.inf
    planetssb = kwargs["planetssb"].copy()
    planetssb[:, 8, :] = np.nan
    planetssb[:, :, 3:] = np.nan
    sunssb = kwargs["sunssb"].copy()
    sunssb[:, 3:] = np.nan
    rec = make_record(freqs=freqs, planetssb=planetssb, sunssb=sunssb)
    assert np.isposinf(rec.freqs[2])
    assert np.isnan(rec.planetssb[:, 8, :]).all()
    assert np.isnan(rec.sunssb[:, 3:]).all()


# --- other fields -------------------------------------------------------------


def test_dmx_none_or_a_nonempty_table():
    assert make_record(dmx=None).dmx is None
    table = {
        "DMX_0001": {
            "DMX": 1e-3,
            "DMXerr": None,
            "DMXR1": 55000.0,
            "DMXR2": 55010.0,
            "fit": True,
        }
    }
    assert make_record(dmx=table).dmx == table
    with pytest.raises(RecordError, match="dmx"):
        make_record(dmx={})
    with pytest.raises(RecordError, match="dmx"):
        make_record(dmx={"DMX_0001": {"DMX": 1e-3}})


@pytest.mark.parametrize("field", ["DMX", "DMXerr", "DMXR1", "DMXR2"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, True])
def test_dmx_numbers_must_be_finite_floats(field, bad):
    entry = {
        "DMX": 1e-3,
        "DMXerr": None,
        "DMXR1": 55000.0,
        "DMXR2": 55010.0,
        "fit": True,
    }
    entry[field] = bad
    with pytest.raises(RecordError, match=field):
        make_record(dmx={"DMX_0001": entry})


def test_pdist_is_a_pair(record):
    assert record.pdist == (1.14, 0.03)
    with pytest.raises(RecordError, match="pdist"):
        make_record(pdist=(1.0,))


def test_producer_and_name_are_nonempty_strings():
    with pytest.raises(RecordError, match="producer"):
        make_record(producer="")
    with pytest.raises(RecordError, match="name"):
        make_record(name=None)


def test_extra_defaults_to_an_empty_mapping():
    kwargs = record_kwargs()
    del kwargs["extra"]
    assert PulsarData(**kwargs).extra == {}


def test_float_row_fields_must_be_floating():
    with pytest.raises(RecordError, match="toaerrs"):
        make_record(toaerrs=np.ones(N_TOAS, dtype=int))


# --- R-1.3, R-1.4, R-3.8.1 ----------------------------------------------------


def test_the_record_is_frozen_and_keeps_the_producers_row_order(record):
    with pytest.raises(AttributeError):
        record.name = "other"
    kwargs = record_kwargs()
    # rows are not in time order in the fixture's ``toas``; they must stay put
    assert np.array_equal(record.toas, kwargs["toas"])
    assert np.array_equal(record.Mmat, kwargs["Mmat"])
    assert record.setpars == SINGLE_SETPARS
    assert record.fitpars == SINGLE_FITPARS


def test_arrays_are_not_copied(record):
    kwargs = record_kwargs()
    rec = PulsarData(**kwargs)
    assert rec.Mmat is kwargs["Mmat"]
    assert rec.toas is kwargs["toas"]


def test_toa_rows_is_the_ordered_alignment_signature(record):
    rows = record.toa_rows()
    assert isinstance(rows, TOARows)
    assert rows.stoas is record.stoas
    assert rows.freqs is record.freqs
    assert rows.toaerrs is record.toaerrs
    assert rows == TOARows(record.stoas, record.freqs, record.toaerrs)


def test_combined_and_single_records_are_one_class(record, combined):
    assert type(record) is type(combined) is PulsarData
    assert len(combined.timing_package) == 2
    assert len(record.timing_package) == 1
