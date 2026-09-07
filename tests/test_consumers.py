"""R-6.3.2: stock Enterprise and Discovery readers read psrdata-written files.

Neither reader imports psrdata. They are the compatibility authority for the
column layout, so this is the schema's real gate (test obligation 10).
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_combined, make_record
from psrdata import feather

pytestmark = pytest.mark.consumers


def _check_arrays(psr, rec):
    for field in ("toas", "stoas", "toaerrs", "residuals", "freqs"):
        assert np.array_equal(getattr(psr, field), getattr(rec, field)), field
    assert np.array_equal(psr.backend_flags.astype("U"), rec.backend_flags)
    assert np.array_equal(psr.Mmat, rec.Mmat)
    assert np.array_equal(psr.sunssb, rec.sunssb, equal_nan=True)
    assert np.array_equal(psr.pos_t, rec.pos_t)
    assert np.array_equal(psr.planetssb, rec.planetssb, equal_nan=True)
    assert set(psr.flags) == set(rec.flags)
    for key in rec.flags:
        assert np.array_equal(psr.flags[key], rec.flags[key]), key
    assert psr.name == rec.name
    assert list(psr.fitpars) == list(rec.fitpars)
    assert list(psr.setpars) == list(rec.setpars)
    assert np.allclose(psr.pos, rec.pos)
    assert psr.theta == rec.theta
    assert psr.phi == rec.phi
    assert tuple(psr.pdist) == rec.pdist
    assert psr.dm == rec.dm
    assert psr.dmx == rec.dmx


def _records():
    kwargs_freqs = make_record().freqs.copy()
    kwargs_freqs[0] = np.inf
    single = make_record(freqs=kwargs_freqs)
    combined = make_combined(
        dmx={
            "DMX_0001": {
                "DMX": 1e-3,
                "DMXerr": None,
                "DMXR1": 55000.0,
                "DMXR2": 55010.0,
                "fit": True,
            }
        }
    )
    return {"single": single, "combined": combined}


@pytest.mark.parametrize("which", ["single", "combined"])
def test_enterprise_feather_pulsar_reads_the_file(tmp_path, which):
    enterprise_pulsar = pytest.importorskip("enterprise.pulsar")
    rec = _records()[which]
    path = tmp_path / f"{which}.feather"
    feather.write(rec, path, noisedict={"J1909-3744_efac": 1.0, "J0437-4715_efac": 2})
    psr = enterprise_pulsar.FeatherPulsar.read_feather(str(path))
    _check_arrays(psr, rec)
    assert np.array_equal(psr.telescope.astype("U"), rec.telescope)
    assert psr._pdist == list(rec.pdist)
    assert psr.noisedict == {"J1909-3744_efac": 1.0}
    # the ``Pulsar`` factory dispatches on the extension
    psr2 = enterprise_pulsar.Pulsar(str(path))
    assert np.array_equal(psr2.residuals, rec.residuals)


@pytest.mark.parametrize("which", ["single", "combined"])
def test_discovery_pulsar_reads_the_file(tmp_path, which):
    discovery_pulsar = pytest.importorskip("discovery.pulsar")
    rec = _records()[which]
    path = tmp_path / f"{which}.feather"
    feather.write(rec, path)
    psr = discovery_pulsar.Pulsar.read_feather(str(path))
    _check_arrays(psr, rec)
    assert psr.mintoa == rec.toas.min()
    assert psr.maxtoa == rec.toas.max()


def test_enterprise_can_build_a_timing_model_over_the_file(tmp_path):
    """The record is one pulsar with one ``Mmat`` to Enterprise, single or combined."""
    enterprise_pulsar = pytest.importorskip("enterprise.pulsar")
    gp_signals = pytest.importorskip("enterprise.signals.gp_signals")
    rec = _records()["combined"]
    path = tmp_path / "tm.feather"
    feather.write(rec, path)
    psr = enterprise_pulsar.FeatherPulsar.read_feather(str(path))
    tm = gp_signals.TimingModel()(psr)
    basis, _ = tm.get_basis(), None
    assert basis.shape == rec.Mmat.shape
