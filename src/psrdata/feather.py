"""Feather schema v1, ``pulsardata-feather-v1`` (SPEC §6).

The column layout is Enterprise's, so ``enterprise.pulsar.FeatherPulsar`` and
``discovery.pulsar.Pulsar`` read these files with no adapter (R-6.3.2). The
psrdata record is rebuilt from the additive JSON metadata and nothing else
(R-6.2.0). Both readers pick vector columns up by *column order*, so the
writer emits ``Mmat_0 .. Mmat_{p-1}`` in index order and never lets another
column start with a reserved prefix.

The codec entry points are ``write(record, path, noisedict=None)``,
``read(path)`` and ``read_metadata(path)``; ``PulsarData.from_feather`` /
``to_feather`` and ``LinearTimingEngine.from_feather`` go through them.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.feather as pa_feather

from .errors import SchemaError
from .record import ParameterFact, PulsarData, ResidualCentering

#: R-6.2.1.
SCHEMA = "pulsardata-feather-v1"

SCALAR_COLUMNS = (
    "toas",
    "stoas",
    "toaerrs",
    "residuals",
    "freqs",
    "backend_flags",
    "telescope",
)
FLOAT_COLUMNS = ("toas", "stoas", "toaerrs", "residuals", "freqs")
VECTOR_COLUMNS = ("Mmat", "sunssb", "pos_t")
TENSOR_COLUMNS = ("planetssb",)
FLAG_PREFIX = "flags_"

ENTERPRISE_KEYS = (
    "name",
    "dm",
    "dmx",
    "pdist",
    "_pdist",
    "pos",
    "phi",
    "theta",
    "fitpars",
    "setpars",
)
PSRDATA_KEYS = (
    "schema",
    "parameters",
    "timing_package",
    "partim_compatibility",
    "residual_centering",
    "producer",
    "extra",
)

_VECTOR_RE = re.compile(r"^(Mmat|sunssb|pos_t)_(\d+)$")
_TENSOR_RE = re.compile(r"^(planetssb)_(\d+)_(\d+)$")


# --- writing ------------------------------------------------------------------------


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")


def _metadata(record: PulsarData, noisedict) -> dict[str, Any]:
    pdist = [float(record.pdist[0]), float(record.pdist[1])]
    meta: dict[str, Any] = {
        "name": record.name,
        "dm": float(record.dm),
        "dmx": (
            None if record.dmx is None else {k: dict(v) for k, v in record.dmx.items()}
        ),
        "pdist": pdist,
        # Enterprise's FeatherPulsar reads both ``pdist`` and ``_pdist``; they
        # carry the same pair.
        "_pdist": list(pdist),
        "pos": [float(x) for x in record.pos],
        "phi": float(record.phi),
        "theta": float(record.theta),
        "fitpars": list(record.fitpars),
        "setpars": list(record.setpars),
        "schema": SCHEMA,
        "parameters": {
            name: {
                "value": fact.value,
                "units": fact.units,
                "uncertainty": fact.uncertainty,
            }
            for name, fact in record.parameters.items()
        },
        "timing_package": dict(record.timing_package),
        "partim_compatibility": dict(record.partim_compatibility),
        "residual_centering": {
            key: asdict(rc) for key, rc in record.residual_centering.items()
        },
        "producer": record.producer,
        "extra": dict(record.extra),
    }
    if noisedict:
        # Enterprise's convention: keys that start with the pulsar name.
        meta["noisedict"] = {
            par: val for par, val in noisedict.items() if par.startswith(record.name)
        }
    return meta


def write(record: PulsarData, path, noisedict: Mapping[str, Any] | None = None) -> None:
    """Write ``record`` at ``path`` in schema v1.

    ``noisedict`` is optional and is filtered to this pulsar using Enterprise's
    convention. Everything about the record is written; nothing is sorted.
    """
    columns: dict[str, Any] = {}
    for name in FLOAT_COLUMNS:
        columns[name] = np.asarray(getattr(record, name), dtype=np.float64)
    columns["backend_flags"] = record.backend_flags
    columns["telescope"] = record.telescope
    for name in VECTOR_COLUMNS:
        arr = np.asarray(getattr(record, name), dtype=np.float64)
        for i in range(arr.shape[1]):
            columns[f"{name}_{i}"] = arr[:, i]
    for name in TENSOR_COLUMNS:
        arr = np.asarray(getattr(record, name), dtype=np.float64)
        for i in range(arr.shape[1]):
            for j in range(arr.shape[2]):
                columns[f"{name}_{i}_{j}"] = arr[:, i, j]
    for key, value in record.flags.items():
        columns[f"{FLAG_PREFIX}{key}"] = np.asarray(value).astype("U")

    meta = _metadata(record, noisedict)
    try:
        text = json.dumps(meta, default=_json_default, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise SchemaError(
            f"record metadata is not JSON serializable (check extra/dmx): {exc}"
        ) from exc
    table = pa.Table.from_pydict(columns, metadata={"json": text})
    pa_feather.write_feather(table, str(path))


# --- reading ------------------------------------------------------------------------


def read_metadata(path) -> dict[str, Any]:
    """The JSON metadata object of a psrdata file, schema-checked but otherwise raw.

    This is where an optional key such as ``noisedict`` is found; ``read``
    rebuilds the record from the required keys only.
    """
    return _checked_metadata(pa_feather.read_table(str(path)), str(path))


def _checked_metadata(table: pa.Table, where: str) -> dict[str, Any]:
    raw = (table.schema.metadata or {}).get(b"json")
    if raw is None:
        raise SchemaError(f"{where}: no 'json' schema metadata; not a psrdata file")
    try:
        meta = json.loads(raw)
    except ValueError as exc:
        raise SchemaError(f"{where}: schema metadata is not valid JSON: {exc}") from exc
    if not isinstance(meta, dict):
        raise SchemaError(f"{where}: schema metadata is not a JSON object")
    if "schema" not in meta:
        raise SchemaError(
            f"{where}: no 'schema' key in the metadata; refusing to guess the layout"
        )
    if meta["schema"] != SCHEMA:
        raise SchemaError(
            f"{where}: unsupported schema {meta['schema']!r}; this reader "
            f"understands {SCHEMA!r} only"
        )
    return meta


def _require(meta: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in meta:
        raise SchemaError(f"{where}: required metadata key {key!r} is missing")
    return meta[key]


def _string_tuple(meta: Mapping[str, Any], key: str, where: str) -> tuple[str, ...]:
    value = _require(meta, key, where)
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise SchemaError(f"{where}: {key!r} must be a JSON array of nonempty strings")
    return tuple(value)


def _str_or_none(value, what: str, where: str) -> str | None:
    if value is None or isinstance(value, str):
        return value
    raise SchemaError(f"{where}: {what} must be a string or null, got {value!r}")


def _parameters(meta, where) -> dict[str, ParameterFact]:
    raw = _require(meta, "parameters", where)
    if not isinstance(raw, dict):
        raise SchemaError(f"{where}: 'parameters' must be a JSON object")
    facts = {}
    for name, entry in raw.items():
        if not isinstance(entry, dict) or set(entry) != {
            "value",
            "units",
            "uncertainty",
        }:
            raise SchemaError(
                f"{where}: parameters[{name!r}] must be an object with exactly "
                f"the keys value, units, uncertainty; got {entry!r}"
            )
        if not isinstance(entry["value"], str):
            raise SchemaError(
                f"{where}: parameters[{name!r}].value must be a string (never a "
                f"JSON number), got {entry['value']!r}"
            )
        facts[name] = ParameterFact(
            value=entry["value"],
            units=_str_or_none(entry["units"], f"parameters[{name!r}].units", where),
            uncertainty=_str_or_none(
                entry["uncertainty"], f"parameters[{name!r}].uncertainty", where
            ),
        )
    return facts


def _keyed_object(meta, key, where) -> dict[str, Any]:
    raw = _require(meta, key, where)
    if not isinstance(raw, dict):
        raise SchemaError(f"{where}: {key!r} must be a JSON object keyed by data set")
    return raw


def _residual_centering(meta, where) -> dict[str, ResidualCentering]:
    raw = _keyed_object(meta, "residual_centering", where)
    out = {}
    fields = (
        "stored_residuals",
        "stored_weighted",
        "standard_output",
        "standard_weighted",
    )
    for key, entry in raw.items():
        if not isinstance(entry, dict) or any(f not in entry for f in fields):
            raise SchemaError(
                f"{where}: residual_centering[{key!r}] must be an object with "
                f"the keys {fields}; got {entry!r}"
            )
        # An out-of-contract value is a RecordError here (R-4.1.1).
        out[key] = ResidualCentering(**{f: entry[f] for f in fields})
    return out


def _columns(table: pa.Table, where: str):
    names = table.column_names
    present = set(names)
    for name in SCALAR_COLUMNS:
        if name not in present:
            raise SchemaError(f"{where}: required column {name!r} is missing")

    def column(name):
        return table[name].to_numpy()

    scalars = {name: column(name) for name in FLOAT_COLUMNS}
    scalars["backend_flags"] = column("backend_flags").astype("U")
    scalars["telescope"] = column("telescope").astype("U")

    vectors: dict[str, dict[int, np.ndarray]] = {v: {} for v in VECTOR_COLUMNS}
    tensors: dict[str, dict[tuple[int, int], np.ndarray]] = {
        t: {} for t in TENSOR_COLUMNS
    }
    flags: dict[str, np.ndarray] = {}
    for name in names:
        if name in SCALAR_COLUMNS:
            continue
        if name.startswith(FLAG_PREFIX):
            flags[name[len(FLAG_PREFIX) :]] = column(name).astype("U")
            continue
        m = _VECTOR_RE.match(name)
        if m:
            vectors[m.group(1)][int(m.group(2))] = column(name)
            continue
        m = _TENSOR_RE.match(name)
        if m:
            tensors[m.group(1)][(int(m.group(2)), int(m.group(3)))] = column(name)
            continue
        if name.startswith(VECTOR_COLUMNS + TENSOR_COLUMNS):
            raise SchemaError(
                f"{where}: column {name!r} uses a reserved prefix but is not a "
                "block column; Enterprise would misread it"
            )
        # any other column is not part of the schema and is ignored

    n = len(scalars["toas"])
    arrays = dict(scalars)
    for vname, cols in vectors.items():
        k = len(cols)
        if sorted(cols) != list(range(k)):
            raise SchemaError(
                f"{where}: {vname} block columns are not {vname}_0..{vname}_{k - 1}: "
                f"{sorted(cols)}"
            )
        arrays[vname] = (
            np.stack([cols[i] for i in range(k)], axis=1) if k else np.empty((n, 0))
        )
    for tname, cols in tensors.items():
        if not cols:
            raise SchemaError(f"{where}: no {tname} block columns")
        ni = max(i for i, _ in cols) + 1
        nj = max(j for _, j in cols) + 1
        expected = {(i, j) for i in range(ni) for j in range(nj)}
        if set(cols) != expected:
            raise SchemaError(f"{where}: {tname} block columns are not a full grid")
        arrays[tname] = np.stack(
            [np.stack([cols[(i, j)] for j in range(nj)], axis=1) for i in range(ni)],
            axis=1,
        )
    return arrays, flags


def read(path) -> PulsarData:
    """Rebuild the record written at ``path``.

    Refuses a missing or unknown schema (R-6.2.2) and malformed psrdata
    metadata as ``SchemaError``; a structurally invalid record, including an
    out-of-contract residual centering, as ``RecordError``.
    """
    where = str(path)
    table = pa_feather.read_table(where)
    meta = _checked_metadata(table, where)
    arrays, flags = _columns(table, where)

    for key in ENTERPRISE_KEYS + PSRDATA_KEYS:
        _require(meta, key, where)
    pdist = meta["pdist"]
    try:
        pdist = (float(pdist[0]), float(pdist[1]))
    except (TypeError, ValueError, IndexError) as exc:
        raise SchemaError(f"{where}: 'pdist' must be a pair of numbers") from exc
    for key in ("producer", "name"):
        if not isinstance(meta[key], str):
            raise SchemaError(f"{where}: {key!r} must be a string")
    if not isinstance(meta["extra"], dict):
        raise SchemaError(f"{where}: 'extra' must be a JSON object")

    return PulsarData(
        name=meta["name"],
        setpars=_string_tuple(meta, "setpars", where),
        fitpars=_string_tuple(meta, "fitpars", where),
        parameters=_parameters(meta, where),
        flags=flags,
        pos=np.asarray(meta["pos"], dtype=np.float64),
        theta=meta["theta"],
        phi=meta["phi"],
        pdist=pdist,
        dm=meta["dm"],
        dmx=meta["dmx"],
        timing_package=_keyed_object(meta, "timing_package", where),
        partim_compatibility=_keyed_object(meta, "partim_compatibility", where),
        residual_centering=_residual_centering(meta, where),
        producer=meta["producer"],
        extra=meta["extra"],
        **arrays,
    )


__all__ = ["SCHEMA", "write", "read", "read_metadata"]
