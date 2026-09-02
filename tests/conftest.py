"""A synthetic record: enough shape for the schema, no timing package."""

from __future__ import annotations

import numpy as np
import pytest

from psrdata import PulsarData


@pytest.fixture
def record():
    """A 50-row, 3-fitpar record with the shapes the schema requires.

    Synthetic on purpose: this package must be testable with nothing installed
    but numpy and pyarrow, so no fixture here comes from a par file.
    """
    n, npar = 50, 3
    rng = np.random.default_rng(0)
    planetssb = np.full((n, 9, 6), np.nan)
    for slot in (1, 2, 4, 5, 6, 7):
        planetssb[:, slot, :3] = rng.normal(size=(n, 3))
    sunssb = np.zeros((n, 6))
    sunssb[:, :3] = rng.normal(size=(n, 3))
    freqs = np.full(n, 1400.0)
    freqs[7] = np.inf  # a TOA the timing package gave no frequency
    return PulsarData(
        name="J0000+0000",
        fitpars=("F0", "F1", "PHOFF"),
        setpars=("PSR", "PEPOCH"),
        toas=np.linspace(5e9, 5e9 + 1e7, n),
        stoas=np.linspace(5e9, 5e9 + 1e7, n) + 1.0,
        toaerrs=np.full(n, 1e-6),
        residuals=rng.normal(size=n) * 1e-6,
        freqs=freqs,
        Mmat=rng.normal(size=(n, npar)),
        flags={"fe": np.array(["L-wide"] * n), "be": np.array(["ASP"] * n)},
        backend_flags=np.array(["L-wide_ASP"] * n),
        telescope=np.array(["ao"] * n),
        pos=np.array([0.6, 0.8, 0.0]),
        pos_t=np.tile([0.6, 0.8, 0.0], (n, 1)),
        sunssb=sunssb,
        planetssb=planetssb,
        theta=1.2,
        phi=0.3,
        pdist=(1.0, 0.2),
        dm=12.5,
        dmx=None,
        state_id="synthetic-0",
        software="test",
        timing_package="pint",
        gauge={"export": "none", "reference_mode": "none"},
        reference_theta_exact={"F0": "1.0", "F1": "-1e-15", "PHOFF": "0.0"},
        native_units={"F0": "Hz", "F1": "Hz / s", "PHOFF": "1"},
    )
