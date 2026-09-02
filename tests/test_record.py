"""The record's invariants: shapes checked, arrays frozen, order untouched."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from psrdata import PulsarData, backend_flags


def test_the_arrays_are_read_only(record):
    """Every consumer binds these by name; none of them may write."""
    with pytest.raises(ValueError):
        record.residuals[0] = 0.0
    with pytest.raises(ValueError):
        record.flags["fe"][0] = "x"
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.name = "other"


@pytest.mark.parametrize(
    "field,bad",
    [
        ("residuals", np.zeros(49)),
        ("freqs", np.zeros(51)),
        ("Mmat", np.zeros((50, 2))),
        ("pos_t", np.zeros((50, 2))),
        ("planetssb", np.zeros((50, 9, 3))),
        ("sunssb", np.zeros((50, 3))),
    ],
)
def test_a_wrong_shape_is_refused_at_construction(record, field, bad):
    """A wrong-length column becomes a broadcasting bug three packages away,
    so it is caught here instead."""
    fields = {f.name: getattr(record, f.name) for f in dataclasses.fields(record)}
    fields[field] = bad
    with pytest.raises(ValueError, match=field):
        PulsarData(**fields)


def test_a_fitpar_without_a_reference_is_refused(record):
    fields = {f.name: getattr(record, f.name) for f in dataclasses.fields(record)}
    fields["reference_theta_exact"] = {"F0": "1.0"}
    with pytest.raises(ValueError, match="reference_theta_exact"):
        PulsarData(**fields)


def test_toa_rows_are_the_three_identifying_columns(record):
    rows = record.toa_rows()
    assert np.array_equal(rows.stoas, record.stoas)
    assert np.array_equal(rows.freqs, record.freqs, equal_nan=True)
    assert np.array_equal(rows.toaerrs, record.toaerrs)


def test_the_record_does_not_reorder_anything(record):
    """There is no sort here, and no permutation to publish.

    A timing package that sorts the rows its residual and design matrix are
    built from publishes two orders for one freeze; the record's contract is
    that row ``i`` is the writer's row ``i``.
    """
    assert np.array_equal(record.toas, np.sort(record.toas))  # this fixture is ordered
    for attribute in ("data_order", "_isort", "_iisort", "sort_data"):
        assert not hasattr(record, attribute)


def test_backend_flags_follow_enterprises_precedence():
    """``fe_be`` is the base; finer flags overwrite where non-empty, last wins.

    Copied from ``enterprise/pulsar.py`` intact, quirk included, because a
    record that disagrees with Enterprise about backend names selects
    different noise parameters.
    """
    flags = {
        "fe": np.array(["L-wide", "L-wide", ""]),
        "be": np.array(["ASP", "ASP", "PUPPI"]),
        "f": np.array(["", "L-wide_PUPPI", ""]),
    }
    assert list(backend_flags(flags, 3)) == ["L-wide_ASP", "L-wide_PUPPI", ""]
