"""Gauge provenance: validated on construction, on the record, and on read."""

from __future__ import annotations

import dataclasses

import pytest

from psrdata import GaugeProvenance, PulsarData


def test_a_valid_provenance_round_trips_through_its_dict():
    gp = GaugeProvenance(export="none", reference_mode="mean", reference_weighted=True)
    assert GaugeProvenance.from_mapping(gp.to_dict()) == gp
    assert GaugeProvenance.from_mapping(gp) is gp


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"export": "yes", "reference_mode": "none"}, "export"),
        ({"export": "none", "reference_mode": "sometimes"}, "reference_mode"),
        ({"export": "none", "reference_mode": "mean"}, "requires reference_weighted"),
        (
            {"export": "none", "reference_mode": "none", "reference_weighted": True},
            "only allowed",
        ),
        (
            {"export": "none", "reference_mode": "none", "reporting_mode": "mean"},
            "requires reporting_weighted",
        ),
        (
            {"export": "none", "reference_mode": "none", "reporting_weighted": False},
            "only allowed",
        ),
    ],
)
def test_an_invalid_provenance_is_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        GaugeProvenance(**kwargs)


def test_the_record_coerces_a_mapping_into_the_type(record):
    """The fixture passes a plain dict; the record holds the validated type."""
    assert isinstance(record.gauge, GaugeProvenance)
    assert record.gauge.export == "none"


def test_a_malformed_gauge_is_refused_at_construction(record):
    with pytest.raises(ValueError, match="export"):
        dataclasses.replace(record, gauge={"export": "no", "reference_mode": "none"})
    with pytest.raises(TypeError):
        dataclasses.replace(record, gauge="none")


def test_a_composite_carries_one_provenance_per_leg(record):
    composite = dataclasses.replace(
        record,
        timing_package="composite",
        gauge={
            "epta": {"export": "none", "reference_mode": "none"},
            "ppta": GaugeProvenance(export="none", reference_mode="unknown"),
        },
    )
    assert set(composite.gauge) == {"epta", "ppta"}
    assert all(isinstance(v, GaugeProvenance) for v in composite.gauge.values())
    with pytest.raises(TypeError, match="leg -> provenance"):
        dataclasses.replace(
            record,
            timing_package="composite",
            gauge={"export": "none", "reference_mode": "none"},
        )


def test_the_feather_reader_validates_the_gauge(record, tmp_path):
    path = record.to_feather(tmp_path / "psr.feather")
    back = PulsarData.from_feather(path)
    assert back.gauge == record.gauge
    composite = dataclasses.replace(
        record,
        timing_package="composite",
        gauge={"one": {"export": "none", "reference_mode": "none"}},
    )
    path = composite.to_feather(tmp_path / "composite.feather")
    back = PulsarData.from_feather(path)
    assert back.gauge == composite.gauge
