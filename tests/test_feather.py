"""Schema v1: lossless round trip, and readable by the consumers unchanged."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from psrdata import SCHEMA, PulsarData


@pytest.fixture
def written(record, tmp_path):
    return record, record.to_feather(tmp_path / "psr.feather")


def test_round_trip_is_lossless(written):
    record, path = written
    back = PulsarData.from_feather(path)
    for f in dataclasses.fields(record):
        mine, theirs = getattr(record, f.name), getattr(back, f.name)
        if isinstance(mine, np.ndarray):
            if mine.dtype.kind in "US":
                assert np.array_equal(
                    mine.astype(str), np.asarray(theirs).astype(str)
                ), f.name
            else:
                assert np.array_equal(mine, theirs, equal_nan=True), f.name
        elif f.name == "flags":
            assert set(mine) == set(theirs)
            for key, values in mine.items():
                assert np.array_equal(np.asarray(values).astype(str), theirs[key])
        else:
            same = type(mine)(theirs) == mine if mine is not None else theirs is None
            assert same, f.name


def test_an_infinite_frequency_survives_the_file(written):
    """A TOA the timing package gave no frequency is ``inf``, not a placeholder."""
    record, path = written
    back = PulsarData.from_feather(path)
    assert np.isinf(back.freqs[7])
    assert np.isfinite(np.delete(back.freqs, 7)).all()


def test_the_metadata_block_is_self_describing(written):
    """Everything a frozen linear analysis needs, with no timing package."""
    import pyarrow.feather

    record, path = written
    meta = json.loads(pyarrow.feather.read_table(str(path)).schema.metadata[b"json"])
    assert meta["schema"] == SCHEMA
    assert meta["software"] == "test" and meta["timing_package"] == "pint"
    assert set(meta["reference_theta_exact"]) == set(record.fitpars)
    assert set(meta["native_units"]) == set(record.fitpars)
    assert meta["gauge"]["export"] == "none"
    # Enterprise's reader looks for both spellings and complains about the
    # one it cannot find.
    assert meta["pdist"] == meta["_pdist"]


def test_extra_metadata_travels(record, tmp_path):
    """A producer's additive keys survive; readers that do not know them ignore them."""
    fields = {f.name: getattr(record, f.name) for f in dataclasses.fields(record)}
    fields["extra"] = {"legs": [{"pta": "epta", "timing_package": "tempo2"}]}
    path = PulsarData(**fields).to_feather(tmp_path / "extra.feather")
    assert PulsarData.from_feather(path).extra["legs"][0]["pta"] == "epta"


def test_a_noisedict_is_filtered_to_this_pulsar(record, tmp_path):
    import pyarrow.feather

    path = record.to_feather(
        tmp_path / "noise.feather",
        noisedict={"J0000+0000_efac": 1.0, "J1111+1111_efac": 2.0},
    )
    meta = json.loads(pyarrow.feather.read_table(str(path)).schema.metadata[b"json"])
    assert meta["noisedict"] == {"J0000+0000_efac": 1.0}


def test_a_foreign_schema_is_refused(written, tmp_path):
    import pyarrow
    import pyarrow.feather

    _, path = written
    table = pyarrow.feather.read_table(str(path))
    meta = json.loads(table.schema.metadata[b"json"])
    meta["schema"] = "something-else-v9"
    other = tmp_path / "foreign.feather"
    pyarrow.feather.write_feather(
        table.replace_schema_metadata({"json": json.dumps(meta)}), str(other)
    )
    with pytest.raises(ValueError, match="schema"):
        PulsarData.from_feather(other)


@pytest.mark.consumers
def test_discovery_reads_it(written):
    """Stock ``discovery.Pulsar.read_feather``, no adapter."""
    discovery = pytest.importorskip("discovery")

    record, path = written
    theirs = discovery.Pulsar.read_feather(str(path))
    assert theirs.name == record.name
    assert np.array_equal(theirs.toas, record.toas)
    assert np.array_equal(theirs.residuals, record.residuals)
    assert np.array_equal(theirs.Mmat, record.Mmat)
    assert list(theirs.fitpars) == list(record.fitpars)
    # Discovery reads slot 2 and sunssb[:, :3].
    assert np.all(np.isfinite(theirs.planetssb[:, 2, :3]))
    assert np.array_equal(theirs.sunssb, record.sunssb)


@pytest.mark.consumers
def test_enterprise_reads_it(written):
    """Stock ``FeatherPulsar.read_feather``, no adapter."""
    pulsar = pytest.importorskip("enterprise.pulsar")

    record, path = written
    theirs = pulsar.FeatherPulsar.read_feather(str(path))
    assert theirs.name == record.name
    assert np.array_equal(theirs.toas, record.toas)
    assert np.array_equal(theirs.Mmat, record.Mmat)
    assert np.array_equal(theirs.telescope.astype(str), np.asarray(record.telescope))
    assert np.array_equal(theirs.planetssb, record.planetssb, equal_nan=True)
    # Enterprise sorts on read, at its own property layer, on its own key.
    # That is its business; the file carries the writer's rows.
    assert sorted(theirs._isort.tolist()) == list(range(len(record.toas)))
