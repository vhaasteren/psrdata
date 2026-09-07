"""Enterprise compatibility constants: SPEC §8."""

from __future__ import annotations

import numpy as np
import pytest

from psrdata import (
    BACKEND_FLAG_ORDER,
    DEFAULT_DISTANCE_KPC,
    PLANET_SLOTS,
    backend_flags,
)


def test_planet_slots_are_enterprises():
    assert PLANET_SLOTS == {
        "mercury": 0,
        "venus": 1,
        "earth": 2,
        "mars": 3,
        "jupiter": 4,
        "saturn": 5,
        "uranus": 6,
        "neptune": 7,
        "pluto": 8,
    }


def test_backend_flag_order():
    assert BACKEND_FLAG_ORDER == ("f", "i", "sys", "g", "group")


def test_default_distance():
    assert DEFAULT_DISTANCE_KPC == (1.0, 0.2)


def test_fe_be_then_each_flag_overwrites_in_order():
    n = 5
    flags = {
        "fe": np.array(["L", "L", "", "S", "S"]),
        "be": np.array(["ASP", "", "GUPPI", "GUPPI", "GUPPI"]),
        "f": np.array(["", "", "", "S_GUPPI_f", ""]),
        "sys": np.array(["", "", "sys2", "sys3", ""]),
        "group": np.array(["", "", "", "grp3", "grp4"]),
    }
    out = backend_flags(flags, n)
    assert out.tolist() == ["L_ASP", "", "sys2", "grp3", "grp4"]
    assert out.dtype.kind == "U"


def test_missing_flags_give_empty_labels():
    out = backend_flags({}, 3)
    assert out.tolist() == ["", "", ""]


def test_precedence_is_the_listed_order_not_the_mapping_order():
    n = 1
    flags = {
        "group": np.array(["grp"]),
        "f": np.array(["fflag"]),
        "i": np.array(["iflag"]),
    }
    assert backend_flags(flags, n).tolist() == ["grp"]
    flags = {"group": np.array([""]), "f": np.array(["fflag"]), "i": np.array([""])}
    assert backend_flags(flags, n).tolist() == ["fflag"]


def test_wrong_length_flag_is_refused():
    with pytest.raises(ValueError):
        backend_flags({"f": np.array(["a", "b"])}, 3)
