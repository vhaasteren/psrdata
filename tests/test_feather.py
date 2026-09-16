"""Feather schema v1: SPEC §6 (test obligation 9)."""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.feather as pa_feather
import pytest

import psrdata
from conftest import make_combined, make_record, record_kwargs
from psrdata import (
    ParameterFact,
    PulsarData,
    RecordError,
    ResidualCentering,
    SchemaError,
    feather,
)

ARRAY_FIELDS = (
    "toas",
    "stoas",
    "toaerrs",
    "residuals",
    "freqs",
    "Mmat",
    "backend_flags",
    "telescope",
    "pos",
    "pos_t",
    "sunssb",
    "planetssb",
)


def assert_records_equal(a: PulsarData, b: PulsarData):
    for field in ARRAY_FIELDS:
        x, y = getattr(a, field), getattr(b, field)
        equal_nan = x.dtype.kind == "f"
        assert np.array_equal(x, y, equal_nan=equal_nan), field
    assert a.name == b.name
    assert a.setpars == b.setpars
    assert a.fitpars == b.fitpars
    assert a.parameters == b.parameters
    assert tuple(a.parameters) == tuple(b.parameters)
    assert set(a.flags) == set(b.flags)
    for key in a.flags:
        assert np.array_equal(a.flags[key], b.flags[key]), key
    assert a.theta == b.theta
    assert a.phi == b.phi
    assert a.pdist == b.pdist
    assert a.dm == b.dm
    assert a.dmx == b.dmx
    assert a.timing_package == b.timing_package
    assert tuple(a.timing_package) == tuple(b.timing_package)
    assert a.partim_compatibility == b.partim_compatibility
    assert a.residual_centering == b.residual_centering
    assert a.producer == b.producer
    assert a.extra == b.extra


def _meta(path):
    return json.loads(pa_feather.read_table(path).schema.metadata[b"json"])


def _rewrite(path, meta=None, drop_columns=(), rename=None):
    """Write the table at ``path`` back with altered metadata or columns."""
    table = pa_feather.read_table(path)
    metadata = table.schema.metadata
    if drop_columns:
        table = table.drop(list(drop_columns))
    if rename:
        # rename_columns drops the schema metadata; put it back
        table = table.rename_columns([rename.get(c, c) for c in table.column_names])
        table = table.replace_schema_metadata(metadata)
    if meta is not None:
        table = table.replace_schema_metadata({"json": json.dumps(meta)})
    pa_feather.write_feather(table, path)


# --- R-6.3.1 round trip -------------------------------------------------------


def test_single_record_round_trips_field_by_field(tmp_path):
    kwargs = record_kwargs()
    freqs = kwargs["freqs"].copy()
    freqs[1] = np.inf
    planetssb = kwargs["planetssb"].copy()
    planetssb[:, 8, :] = np.nan
    planetssb[:, :, 3:] = np.nan
    flags = dict(kwargs["flags"])
    flags["empty"] = np.array([""] * len(freqs))
    flags["pta_dataset"] = np.array(["PPTA_DR3"] * len(freqs))
    rec = make_record(freqs=freqs, planetssb=planetssb, flags=flags)
    path = tmp_path / "single.feather"
    feather.write(rec, path)
    back = feather.read(path)
    assert_records_equal(rec, back)
    assert np.isinf(back.freqs[1])
    assert np.isnan(back.planetssb[:, 8, :]).all()
    assert back.timing_package == {"single": "tempo2"}


def test_combined_record_round_trips(tmp_path):
    dmx = {
        "DMX_0001": {
            "DMX": 1e-3,
            "DMXerr": None,
            "DMXR1": 55000.0,
            "DMXR2": 55010.0,
            "fit": True,
        },
        "DMX_0002": {
            "DMX": -2e-3,
            "DMXerr": 4e-4,
            "DMXR1": 55010.0,
            "DMXR2": 55020.0,
            "fit": False,
        },
    }
    rec = make_combined(dmx=dmx, extra={"combination": {"strategy": "shared"}})
    path = tmp_path / "combined.feather"
    feather.write(rec, path)
    assert_records_equal(rec, feather.read(path))


def test_record_methods_are_the_same_codec(record, tmp_path):
    path = tmp_path / "m.feather"
    record.to_feather(path)
    assert_records_equal(record, PulsarData.from_feather(path))
    assert_records_equal(record, feather.read(str(path)))


def test_row_order_survives_unsorted(tmp_path):
    kwargs = record_kwargs()
    toas = kwargs["toas"][::-1].copy()
    rec = make_record(toas=toas)
    path = tmp_path / "rev.feather"
    feather.write(rec, path)
    assert np.array_equal(feather.read(path).toas, toas)


def test_many_matrix_columns_keep_their_index_order(tmp_path):
    """``Mmat_10`` must not sort before ``Mmat_2``; both readers rely on it."""
    kwargs = record_kwargs()
    n = len(kwargs["toas"])
    extra = [f"F{i}" for i in range(2, 12)]
    fitpars = tuple(kwargs["fitpars"]) + tuple(extra)
    setpars = tuple(kwargs["setpars"]) + tuple(extra)
    parameters = dict(kwargs["parameters"])
    for name in extra:
        parameters[name] = ParameterFact("0.0", f"Hz / s{name[1:]}")
    Mmat = np.hstack([kwargs["Mmat"], np.random.default_rng(3).normal(size=(n, 10))])
    rec = make_record(
        fitpars=fitpars, setpars=setpars, parameters=parameters, Mmat=Mmat
    )
    path = tmp_path / "wide.feather"
    feather.write(rec, path)
    names = pa_feather.read_table(path).column_names
    mcols = [c for c in names if c.startswith("Mmat")]
    assert mcols == [f"Mmat_{i}" for i in range(len(fitpars))]
    assert np.array_equal(feather.read(path).Mmat, Mmat)


# --- §6.1, §6.2 layout --------------------------------------------------------


def test_the_column_layout_is_enterprises(record, tmp_path):
    path = tmp_path / "cols.feather"
    feather.write(record, path)
    names = pa_feather.read_table(path).column_names
    p = len(record.fitpars)
    expected = (
        ["toas", "stoas", "toaerrs", "residuals", "freqs", "backend_flags", "telescope"]
        + [f"Mmat_{i}" for i in range(p)]
        + [f"sunssb_{i}" for i in range(6)]
        + [f"pos_t_{i}" for i in range(3)]
        + [f"planetssb_{i}_{j}" for i in range(9) for j in range(6)]
        + [f"flags_{k}" for k in record.flags]
    )
    assert set(names) == set(expected)
    assert len(names) == len(expected)


def test_the_metadata_carries_both_key_sets(record, tmp_path):
    path = tmp_path / "meta.feather"
    feather.write(record, path)
    meta = _meta(path)
    for key in (
        "name", "dm", "dmx", "pdist", "_pdist", "pos", "phi", "theta",
        "fitpars", "setpars",
    ):  # fmt: skip
        assert key in meta, key
    for key in (
        "schema", "parameters", "timing_package", "partim_compatibility",
        "residual_centering", "producer", "extra",
    ):  # fmt: skip
        assert key in meta, key
    assert meta["schema"] == feather.SCHEMA == "pulsardata-feather-v1"
    assert meta["pdist"] == meta["_pdist"] == [1.14, 0.03]
    assert meta["pos"] == record.pos.tolist()
    assert meta["fitpars"] == list(record.fitpars)
    assert meta["setpars"] == list(record.setpars)
    assert meta["dmx"] is None
    assert meta["producer"] == "test"
    assert "noisedict" not in meta


def test_the_psrdata_keys_have_the_specified_json_forms(combined, tmp_path):
    """R-6.2.0."""
    path = tmp_path / "forms.feather"
    feather.write(combined, path)
    meta = _meta(path)
    for name, fact in combined.parameters.items():
        entry = meta["parameters"][name]
        assert set(entry) == {"value", "units", "uncertainty"}
        assert isinstance(entry["value"], str)
        assert entry["units"] is None or isinstance(entry["units"], str)
        assert entry["uncertainty"] is None or isinstance(entry["uncertainty"], str)
        assert entry == {
            "value": fact.value,
            "units": fact.units,
            "uncertainty": fact.uncertainty,
        }
    keys = list(combined.timing_package)
    assert list(meta["timing_package"]) == keys
    assert list(meta["partim_compatibility"]) == keys
    assert list(meta["residual_centering"]) == keys
    assert meta["residual_centering"]["PPTA_DR3"] == {
        "stored_residuals": "none",
        "stored_weighted": None,
        "standard_output": "constant_removed",
        "standard_weighted": None,
    }
    assert isinstance(meta["extra"], dict)


def test_noisedict_is_filtered_to_this_pulsar(record, tmp_path):
    path = tmp_path / "nd.feather"
    noisedict = {
        "J1909-3744_efac": 1.1,
        "J1909-3744_log10_ecorr": -7.0,
        "J0437-4715_efac": 0.9,
        "gw_log10_A": -14.5,
    }
    feather.write(record, path, noisedict=noisedict)
    assert _meta(path)["noisedict"] == {
        "J1909-3744_efac": 1.1,
        "J1909-3744_log10_ecorr": -7.0,
    }
    assert feather.read_metadata(path)["noisedict"] == _meta(path)["noisedict"]
    assert_records_equal(record, feather.read(path))


def test_numbers_inside_parameters_are_never_json_numbers(record, tmp_path):
    path = tmp_path / "num.feather"
    feather.write(record, path)
    raw = pa_feather.read_table(path).schema.metadata[b"json"].decode()
    assert '"value": "339.31568728815203"' in raw


# --- R-6.2.2, R-6.4.3 schema refusal ------------------------------------------


def test_a_missing_schema_is_refused(record, tmp_path):
    path = tmp_path / "noschema.feather"
    feather.write(record, path)
    meta = _meta(path)
    del meta["schema"]
    _rewrite(path, meta=meta)
    with pytest.raises(SchemaError, match="schema"):
        feather.read(path)


def test_an_unknown_schema_is_refused(record, tmp_path):
    path = tmp_path / "v2.feather"
    feather.write(record, path)
    meta = _meta(path)
    meta["schema"] = "pulsardata-feather-v2"
    _rewrite(path, meta=meta)
    with pytest.raises(SchemaError, match="pulsardata-feather-v2"):
        feather.read(path)


def test_an_enterprise_only_file_is_refused(record, tmp_path):
    """No ``json`` metadata at all, or Enterprise's keys alone: not psrdata."""
    path = tmp_path / "ent.feather"
    feather.write(record, path)
    table = pa_feather.read_table(path)
    pa_feather.write_feather(table.replace_schema_metadata(None), path)
    with pytest.raises(SchemaError):
        feather.read(path)
    feather.write(record, path)
    meta = {k: v for k, v in _meta(path).items() if not k.startswith("s")}
    _rewrite(path, meta=meta)
    with pytest.raises(SchemaError):
        feather.read(path)


@pytest.mark.parametrize(
    "key",
    [
        "parameters", "timing_package", "partim_compatibility",
        "residual_centering", "producer", "extra", "name", "fitpars",
        "setpars", "pos", "theta", "phi", "pdist", "dm", "dmx",
    ],
)  # fmt: skip
def test_a_missing_required_key_is_refused(record, tmp_path, key):
    path = tmp_path / "missing.feather"
    feather.write(record, path)
    meta = _meta(path)
    del meta[key]
    _rewrite(path, meta=meta)
    with pytest.raises(SchemaError, match=key):
        feather.read(path)


def test_unknown_additive_keys_are_ignored(record, tmp_path):
    path = tmp_path / "additive.feather"
    feather.write(record, path)
    meta = _meta(path)
    meta["future_key"] = {"anything": [1, 2, 3]}
    _rewrite(path, meta=meta)
    assert_records_equal(record, feather.read(path))


def test_malformed_parameter_entries_are_refused(record, tmp_path):
    path = tmp_path / "badpar.feather"
    feather.write(record, path)
    meta = _meta(path)
    meta["parameters"]["F0"]["value"] = 339.3
    _rewrite(path, meta=meta)
    with pytest.raises(SchemaError, match="F0"):
        feather.read(path)
    meta = _meta(path)
    del meta["parameters"]["F0"]["uncertainty"]
    _rewrite(path, meta=meta)
    with pytest.raises(SchemaError, match="F0"):
        feather.read(path)


@pytest.mark.parametrize(
    "key,bad",
    [
        ("setpars", 5),
        ("fitpars", "F0"),
        ("setpars", ["F0", 7]),
        ("fitpars", [""]),
    ],
)
def test_malformed_parameter_name_lists_are_schema_errors(record, tmp_path, key, bad):
    path = tmp_path / "badnames.feather"
    feather.write(record, path)
    meta = _meta(path)
    meta[key] = bad
    _rewrite(path, meta=meta)
    with pytest.raises(SchemaError, match=key):
        feather.read(path)


def test_invalid_centering_at_read_is_a_record_error(record, tmp_path):
    """R-4.1.1: refused at Feather read as a ``RecordError``."""
    path = tmp_path / "badrc.feather"
    feather.write(record, path)
    meta = _meta(path)
    meta["residual_centering"]["single"]["stored_weighted"] = None
    _rewrite(path, meta=meta)
    with pytest.raises(RecordError):
        feather.read(path)


def test_a_missing_column_is_refused(record, tmp_path):
    path = tmp_path / "nocol.feather"
    feather.write(record, path)
    _rewrite(path, drop_columns=["residuals"])
    with pytest.raises(SchemaError, match="residuals"):
        feather.read(path)


def test_a_stray_column_under_a_reserved_prefix_is_refused(record, tmp_path):
    """Enterprise would read ``Mmat_extra`` as one more matrix column."""
    path = tmp_path / "stray.feather"
    feather.write(record, path)
    _rewrite(path, rename={"flags_pta": "Mmat_extra"})
    with pytest.raises(SchemaError, match="Mmat_extra"):
        feather.read(path)


def test_a_non_contiguous_matrix_block_is_refused(record, tmp_path):
    path = tmp_path / "gap.feather"
    feather.write(record, path)
    _rewrite(path, drop_columns=["Mmat_1"])
    with pytest.raises(SchemaError, match="Mmat"):
        feather.read(path)


def test_a_data_set_key_mismatch_in_the_file_is_a_record_error(combined, tmp_path):
    path = tmp_path / "keys.feather"
    feather.write(combined, path)
    meta = _meta(path)
    meta["timing_package"]["NANOGrav_15y"] = meta["timing_package"].pop("PPTA_DR3")
    _rewrite(path, meta=meta)
    with pytest.raises(RecordError, match="keys"):
        feather.read(path)


def test_the_single_key_is_read_back_not_rederived(record, tmp_path):
    path = tmp_path / "single.feather"
    feather.write(record, path)
    meta = _meta(path)
    assert list(meta["timing_package"]) == ["single"]
    meta["timing_package"] = {"PPTA_DR3": "tempo2"}
    meta["partim_compatibility"] = {"PPTA_DR3": "tempo2"}
    meta["residual_centering"] = {"PPTA_DR3": meta["residual_centering"]["single"]}
    _rewrite(path, meta=meta)
    assert list(feather.read(path).timing_package) == ["PPTA_DR3"]


def test_nonserializable_extra_is_a_schema_error(record, tmp_path):
    rec = make_record(extra={"obj": object()})
    with pytest.raises(SchemaError, match="extra"):
        feather.write(rec, tmp_path / "x.feather")


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_json_metadata_is_a_schema_error(tmp_path, bad):
    rec = make_record(extra={"bad": bad})
    with pytest.raises(SchemaError, match="extra"):
        feather.write(rec, tmp_path / "x.feather")


def test_numpy_scalars_in_metadata_are_written(tmp_path):
    rec = make_record(
        theta=np.float64(2.23),
        dm=np.float32(10.39),
        pdist=(np.float64(1.0), np.float64(0.2)),
        extra={"n": np.int64(3), "arr": np.arange(2)},
    )
    path = tmp_path / "np.feather"
    feather.write(rec, path)
    back = feather.read(path)
    assert back.extra == {"n": 3, "arr": [0, 1]}
    assert back.pdist == (1.0, 0.2)


def test_module_exposes_the_schema_constant():
    assert psrdata.SCHEMA == "pulsardata-feather-v1"
    assert isinstance(ResidualCentering("none"), ResidualCentering)
    assert pa.__version__  # the codec is PyArrow's
