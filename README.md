# psrdata

**The frozen pulsar-data record, its on-disk schema, its own linear engine,
and the pure par-text rules** — the layer every pulsar-timing package in this
stack agrees on, and the only one none of them owns.

```python
from psrdata import PulsarData

psr = PulsarData.from_feather("J1909-3744.feather")
psr.toas, psr.residuals, psr.Mmat, psr.fitpars   # named, frozen, in the writer's row order
engine = psr.linear_engine()                     # Δr = −Mmat δ, in nltiming's engine shape
```

[`SPEC.md`](SPEC.md) is the normative description; [`SPEC-motivation.md`](SPEC-motivation.md)
says why it is designed the way it is.

## Status

This repository is a development preview, not a release. The psrdata-owned
parts of the v1 contract are implemented and tested, including stock
Enterprise and Discovery readers and nltiming's record-engine conformance
suite. Producer migration in MetaPulsar and vela-jax is still coordinated
separately; pre-v1 development Feather files are intentionally not accepted
by the v1 reader.

## What it is

Four things, each of which was being duplicated or defined one layer too high:

1. **`PulsarData`** — a frozen record of named arrays (TOAs, residuals, design
   matrix, flags, ephemeris vectors) plus the metadata that makes it
   self-describing: which software wrote it (`producer`), which timing
   package calculated each data set's residuals and matrix (`timing_package`),
   which of PINT or tempo2 read the files (`partim_compatibility`), the exact
   reference parameter values and PINT units (`parameters`), and the residual
   centering per data set. `feather.write`/`read` are its on-disk form, schema
   `pulsardata-feather-v1`, whose columns are exactly what Enterprise's
   `FeatherPulsar` and Discovery's `Pulsar` already read.
2. **The record's linear engine** — `PulsarData.linear_engine()` returns
   `Δr = −Mmat δ` over the record's own matrix, single-leg or composite, in
   the shape nltiming's `TimingEngine` protocol describes, without importing
   nltiming. A combined record declares no partition: a data set's rows are
   the support of its phase-offset column (`Offset_<key>` or `PHOFF_<key>`),
   and its active linear columns are the ones nonzero on those rows.
3. **`ParameterFact` and `ResidualCentering`** — the validated value types of
   the `parameters` and `residual_centering` fields, here because they are
   serialized in the record.
4. **`partext`** — the par-file rules that are pure text: which lines are
   noise hyperparameters, what `UNITS` means, the two keyword respellings PINT
   and tempo2 disagree about, and collapsing a doubled non-repeatable line.

## What it is not

It does not know about PINT, tempo2, JAX, or sorting, and it defines no
protocol for a live function of the timing parameters. Its runtime
dependencies are `numpy` and `pyarrow`, and a test asserts that importing it
pulls in nothing else — that is the property that lets it sit below everything,
and what lets a frozen linear timing analysis run from the file alone.

**It never reorders rows.** Row `i` of every array is row `i` as the writer
emitted it. A consumer that wants time order sorts when it reads; a timing
package that permutes the rows its residual and design matrix are built from
publishes two orders for one freeze, and the reconciling permutation is the
identity on most real files and therefore never exercised.

## Who produces it

`MetaPulsar` (one record per multi-PTA composite), `vela-jax` (one per pulsar)
and any other timing package that emits the record. Enterprise and Discovery
read the feather with their own readers and never import this package.
`nltiming` consumes the record and its linear engine.
