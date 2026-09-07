"""The property that lets this package sit below every other one."""

from __future__ import annotations

import subprocess
import sys

# R-2.1: PINT, tempo2/libstempo, JAX, Astropy, SciPy, Enterprise, Discovery,
# MetaPulsar and nltiming.
FORBIDDEN = (
    "pint",
    "libstempo",
    "tempo2",
    "jax",
    "astropy",
    "scipy",
    "enterprise",
    "discovery",
    "metapulsar",
    "nltiming",
    "pandas",
)


def test_importing_psrdata_pulls_in_no_timing_stack():
    """numpy and pyarrow, and nothing else.

    A frozen array record that needed PINT or JAX installed to read would
    defeat its own purpose: the point of the file is that a frozen linear
    analysis can be rebuilt from it with no timing package present.
    """
    out = subprocess.run(
        [sys.executable, "-X", "importtime", "-c", "import psrdata"],
        capture_output=True,
        text=True,
        check=True,
    ).stderr.lower()
    offenders = sorted(
        {name for name in FORBIDDEN if f" {name}\n" in out or f" {name}." in out}
    )
    assert not offenders, f"import psrdata pulled in {offenders}"
