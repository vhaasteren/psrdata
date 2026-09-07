"""``ResidualCentering`` validation: SPEC §4 (R-4.1.1)."""

from __future__ import annotations

import pytest

from conftest import make_record
from psrdata import RecordError, ResidualCentering

LITERALS = ("none", "mean_removed", "constant_removed", "unknown")


@pytest.mark.parametrize("stored", LITERALS)
@pytest.mark.parametrize("standard", LITERALS)
def test_every_literal_pair_with_the_matching_weighted_values(stored, standard):
    rc = ResidualCentering(
        stored_residuals=stored,
        stored_weighted=True if stored == "mean_removed" else None,
        standard_output=standard,
        standard_weighted=False if standard == "mean_removed" else None,
    )
    assert rc.stored_residuals == stored
    assert rc.standard_output == standard


def test_defaults_are_unknown_and_unweighted():
    rc = ResidualCentering("none")
    assert rc.stored_weighted is None
    assert rc.standard_output == "unknown"
    assert rc.standard_weighted is None


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(stored_residuals="centred"),
        dict(stored_residuals="none", standard_output="demeaned"),
        dict(stored_residuals="mean_removed"),  # weighted required
        dict(stored_residuals="none", stored_weighted=True),
        dict(stored_residuals="none", standard_output="mean_removed"),
        dict(stored_residuals="none", standard_output="none", standard_weighted=False),
        dict(stored_residuals="mean_removed", stored_weighted="yes"),
        dict(stored_residuals=None),
    ],
)
def test_invalid_combinations_are_a_record_error(kwargs):
    with pytest.raises(RecordError):
        ResidualCentering(**kwargs)


def test_the_record_refuses_an_invalid_centering_too():
    """R-4.1.1: refused at record construction, not only at type construction.

    The dataclass validates itself, so the only way to reach the record with a
    bad value is to bypass ``__post_init__``; the record checks again.
    """
    rc = ResidualCentering("none")
    object.__setattr__(rc, "stored_residuals", "demeaned")
    with pytest.raises(RecordError):
        make_record(residual_centering=rc)
