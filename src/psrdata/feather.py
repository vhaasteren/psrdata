"""Feather schema v1: the on-disk form of :class:`~psrdata.record.PulsarData`.

The columns are frozen exactly as Enterprise's ``FeatherPulsar`` and
Discovery's ``Pulsar`` already read them, so a file written here is consumable
by both with no adapter. The metadata block is the addition: ``fitpars``,
``Mmat``, the exact reference strings, the units and the gauge provenance all
travel with the file, which is what lets a frozen linear timing analysis be
rebuilt from it alone, with no timing package installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .gauge import gauge_to_json
from .record import SCHEMA, PulsarData

_COLUMNS = (
    "toas",
    "stoas",
    "toaerrs",
    "residuals",
    "freqs",
    "backend_flags",
    "telescope",
)
_VECTORS = ("Mmat", "sunssb", "pos_t")
_TENSORS = ("planetssb",)


def write(record: PulsarData, path, *, noisedict=None) -> Path:
    """Write ``record`` as a schema-v1 feather file."""
    import pyarrow
    import pyarrow.feather

    table: dict[str, Any] = {
        name: np.asarray(getattr(record, name)) for name in _COLUMNS
    }
    for name in _VECTORS:
        block = np.asarray(getattr(record, name))
        table.update({f"{name}_{i}": block[:, i] for i in range(block.shape[1])})
    for name in _TENSORS:
        block = np.asarray(getattr(record, name))
        table.update(
            {
                f"{name}_{i}_{j}": block[:, i, j]
                for i in range(block.shape[1])
                for j in range(block.shape[2])
            }
        )
    table.update({f"flags_{key}": value for key, value in record.flags.items()})

    meta = {
        "name": record.name,
        "dm": record.dm,
        "dmx": record.dmx,
        "pdist": list(record.pdist),
        # Enterprise's FeatherPulsar reads both spellings and prints a
        # complaint for the one it cannot find; writing both keeps a stock
        # reader quiet without adding a field.
        "_pdist": list(record.pdist),
        "pos": np.asarray(record.pos).tolist(),
        "phi": record.phi,
        "theta": record.theta,
        "fitpars": list(record.fitpars),
        "setpars": list(record.setpars),
        "schema": record.schema,
        "state_id": record.state_id,
        "software": record.software,
        "timing_package": record.timing_package,
        "gauge": gauge_to_json(record.gauge),
        "reference_theta_exact": dict(record.reference_theta_exact or {}),
        "native_units": dict(record.native_units or {}),
        "extra": dict(record.extra or {}),
    }
    if noisedict:
        # Enterprise's convention: only this pulsar's entries, keyed by name.
        meta["noisedict"] = {
            key: value
            for key, value in noisedict.items()
            if key.startswith(record.name)
        }
    pyarrow.feather.write_feather(
        pyarrow.Table.from_pydict(table, metadata={"json": json.dumps(meta)}),
        str(path),
    )
    return Path(path)


def read(path) -> PulsarData:
    """Read a schema-v1 feather back into a record, losslessly."""
    import pyarrow.feather

    table = pyarrow.feather.read_table(str(path))
    names = table.column_names
    columns = {name: table[name].to_numpy() for name in _COLUMNS}

    def block(prefix, width):
        return np.stack(
            [table[f"{prefix}_{i}"].to_numpy() for i in range(width)], axis=1
        )

    n_fit = sum(1 for name in names if name.startswith("Mmat_"))
    vectors = {
        "Mmat": block("Mmat", n_fit),
        "sunssb": block("sunssb", 6),
        "pos_t": block("pos_t", 3),
    }
    planetssb = np.stack(
        [
            np.stack([table[f"planetssb_{i}_{j}"].to_numpy() for j in range(6)], axis=1)
            for i in range(9)
        ],
        axis=1,
    )
    flags = {
        name[len("flags_") :]: table[name].to_numpy().astype(str)
        for name in names
        if name.startswith("flags_")
    }
    meta = json.loads(table.schema.metadata[b"json"])
    if meta.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {meta.get('schema')!r}, expected {SCHEMA!r}")
    return PulsarData(
        name=meta["name"],
        fitpars=tuple(meta["fitpars"]),
        setpars=tuple(meta["setpars"]),
        flags=flags,
        pos=np.asarray(meta["pos"], dtype=float),
        theta=float(meta["theta"]),
        phi=float(meta["phi"]),
        pdist=tuple(meta["pdist"]),
        dm=float(meta["dm"]),
        dmx=meta["dmx"],
        state_id=meta["state_id"],
        software=meta["software"],
        timing_package=meta["timing_package"],
        gauge=meta["gauge"],
        reference_theta_exact=meta["reference_theta_exact"],
        native_units=meta["native_units"],
        extra=meta.get("extra", {}),
        schema=meta["schema"],
        planetssb=planetssb,
        **columns,
        **vectors,
    )


__all__ = ["write", "read"]
