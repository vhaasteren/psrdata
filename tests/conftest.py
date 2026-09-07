"""Builders for valid records, so every test starts from a record that passes.

``record_kwargs`` returns the keyword arguments of a valid single-data-set
record; ``combined_kwargs`` those of a valid two-data-set record. Tests
mutate one field and assert what the constructor does with it.
"""

from __future__ import annotations

import numpy as np
import pytest

from psrdata import ParameterFact, PulsarData, ResidualCentering

N_TOAS = 12

SINGLE_FITPARS = ("F0", "F1", "RAJ", "Offset")
SINGLE_SETPARS = ("F0", "F1", "RAJ", "DECJ", "PB", "PSR", "Offset")

CENTERING = ResidualCentering(
    stored_residuals="mean_removed",
    stored_weighted=True,
    standard_output="mean_removed",
    standard_weighted=True,
)


def _parameters(setpars):
    facts = {
        "F0": ParameterFact("339.31568728815203", "Hz", "1.2e-13"),
        "F1": ParameterFact("-1.614739e-15", "Hz / s", "3e-21"),
        "RAJ": ParameterFact("19.0975", "hourangle", "1e-8"),
        "DECJ": ParameterFact("-37.7369", "deg", None),
        "PB": ParameterFact("1.533449474406", "d", "1e-11"),
        "PSR": ParameterFact("J1909-3744", None, None),
        "Offset": ParameterFact("0", "s", None),
    }
    return {name: facts[name] for name in setpars}


def _rows(n, rng):
    stoas = 55000.0 * 86400 + np.sort(rng.uniform(0, 3.0e8, n))
    return dict(
        toas=stoas + rng.normal(0, 300.0, n),
        stoas=stoas,
        toaerrs=rng.uniform(1e-7, 1e-6, n),
        residuals=rng.normal(0, 1e-6, n),
        freqs=rng.uniform(700.0, 3000.0, n),
        backend_flags=np.array([f"be{i % 2}" for i in range(n)]),
        telescope=np.array(["pks"] * n),
        pos_t=np.tile(np.array([0.1, -0.6, 0.79]), (n, 1)),
        sunssb=rng.normal(0, 500.0, (n, 6)),
        planetssb=rng.normal(0, 3000.0, (n, 9, 6)),
    )


def record_kwargs(n=N_TOAS, seed=0):
    """Keyword arguments of a valid single-data-set record."""
    rng = np.random.default_rng(seed)
    rows = _rows(n, rng)
    Mmat = rng.normal(0, 1.0, (n, len(SINGLE_FITPARS)))
    Mmat[:, SINGLE_FITPARS.index("Offset")] = 1.0
    return dict(
        name="J1909-3744",
        setpars=SINGLE_SETPARS,
        fitpars=SINGLE_FITPARS,
        parameters=_parameters(SINGLE_SETPARS),
        Mmat=Mmat,
        flags={
            "f": np.array([f"be{i % 2}" for i in range(n)]),
            "pta": np.array(["PPTA"] * n),
        },
        pos=np.array([0.1, -0.6, 0.79]),
        theta=2.23,
        phi=5.0,
        pdist=(1.14, 0.03),
        dm=10.39,
        dmx=None,
        timing_package="tempo2",
        partim_compatibility="tempo2",
        residual_centering=CENTERING,
        producer="test",
        extra={"note": "fixture"},
        **rows,
    )


def make_record(**overrides):
    kwargs = record_kwargs()
    kwargs.update(overrides)
    return PulsarData(**kwargs)


COMBINED_KEYS = ("EPTA_DR2", "PPTA_DR3")
COMBINED_FITPARS = (
    "F0",
    "F1",
    "Offset_EPTA_DR2",
    "DM_PPTA_DR3",
    "PHOFF_PPTA_DR3",
)
COMBINED_SETPARS = COMBINED_FITPARS + ("PB", "PSR")


def combined_kwargs(n_per=(5, 7), seed=1):
    """Keyword arguments of a valid two-data-set record.

    Rows are interleaved between the two data sets rather than blocked, so a
    test that relied on contiguous row slices would fail.
    """
    rng = np.random.default_rng(seed)
    n = sum(n_per)
    membership = np.array([0] * n_per[0] + [1] * n_per[1])
    rng.shuffle(membership)
    rows = _rows(n, rng)
    p = len(COMBINED_FITPARS)
    Mmat = rng.normal(0, 1.0, (n, p))
    in_a = membership == 0
    in_b = membership == 1
    Mmat[:, COMBINED_FITPARS.index("Offset_EPTA_DR2")] = np.where(in_a, 1.0, 0.0)
    Mmat[:, COMBINED_FITPARS.index("PHOFF_PPTA_DR3")] = np.where(in_b, -2.9e-3, 0.0)
    Mmat[in_a, COMBINED_FITPARS.index("DM_PPTA_DR3")] = 0.0
    parameters = {
        "F0": ParameterFact("339.31568728815203", "Hz", "1.2e-13"),
        "F1": ParameterFact("-1.614739e-15", "Hz / s", "3e-21"),
        "Offset_EPTA_DR2": ParameterFact("0", "s", None),
        "DM_PPTA_DR3": ParameterFact("10.3912", "pc / cm3", "2e-4"),
        "PHOFF_PPTA_DR3": ParameterFact("0", "dimensionless", None),
        "PB": ParameterFact("1.533449474406", "d", None),
        "PSR": ParameterFact("J1909-3744", None, None),
    }
    keys = COMBINED_KEYS
    return (
        dict(
            name="J1909-3744",
            setpars=COMBINED_SETPARS,
            fitpars=COMBINED_FITPARS,
            parameters=parameters,
            Mmat=Mmat,
            flags={"pta": np.array([keys[m].split("_")[0] for m in membership])},
            pos=np.array([0.1, -0.6, 0.79]),
            theta=2.23,
            phi=5.0,
            pdist=(1.14, 0.03),
            dm=10.39,
            dmx=None,
            timing_package={keys[0]: "pint", keys[1]: "vela_jax"},
            partim_compatibility={keys[0]: "pint", keys[1]: "tempo2"},
            residual_centering={
                keys[0]: ResidualCentering("mean_removed", True, "mean_removed", True),
                keys[1]: ResidualCentering("none", None, "constant_removed", None),
            },
            producer="metapulsar",
            extra={},
            **rows,
        ),
        membership,
    )


def make_combined(**overrides):
    kwargs, _ = combined_kwargs()
    kwargs.update(overrides)
    return PulsarData(**kwargs)


@pytest.fixture
def record():
    return make_record()


@pytest.fixture
def combined():
    return make_combined()


@pytest.fixture
def combined_membership():
    kwargs, membership = combined_kwargs()
    return PulsarData(**kwargs), membership
