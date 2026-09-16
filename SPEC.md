# psrdata — specification

This document defines the shared `psrdata` contract for MetaPulsar,
vela-jax and nltiming. The companion
[`SPEC-motivation.md`](SPEC-motivation.md) explains the design choices.

Requirement identifiers (`R-n.m`) are stable handles for tests and reviews.
“Must” states a requirement. “May” states an allowed choice. Each requirement
also says whether psrdata can enforce it directly or whether its producer must
guarantee it.

---

## 0. Vocabulary

| term | meaning |
|---|---|
| **record** | one `psrdata.PulsarData`: materialized timing arrays and the metadata needed to interpret them |
| **producer** | code that constructs a record from a timing calculation |
| **consumer** | code that reads a record or its Feather representation |
| **timing package** | the software that calculated the stored `residuals` and `Mmat`: PINT, tempo2/libstempo, JUG, Vela.jl, vela-jax, or another implementation |
| **par/tim compatibility** | which of the two supported packages, PINT or tempo2, read a data set's par/tim files under its rules and defaults and produced the frozen input quantities the calculation consumed |
| **producer software** | the package that constructed and wrote the record: MetaPulsar, vela-jax, or another producer. Distinct from the timing package when the producer combines or wraps calculations it did not perform |
| **data set** | one independently timed contribution with its own par/tim interpretation and arbitrary phase reference; a single-data-set pulsar has one, a combined pulsar has several |
| **standalone data-set key** | the reserved key `"single"`, inserted by psrdata when a producer supplies scalar metadata for one unnamed data set |
| **combined record** | one record formed from several data sets in one row space and one fit-parameter space |
| **phase-offset column** | the `Offset` or `PHOFF` design-matrix column representing one data set's arbitrary pulse-phase reference |
| **PINT unit** | the default unit assigned to that parameter by PINT; all record parameter values, uncertainties, deltas and design-matrix columns use these units |
| **fitter sign** | `Δr ≈ -Mmat @ δ` |
| **linear engine** | the record's callable linear timing calculation, `Δr = -Mmat @ δ` |
| **schema** | the versioned Feather representation, `pulsardata-feather-v1` |

---

## 1. Identity and scope

**R-1.1 (package identity).** `psrdata` owns:

1. the `PulsarData` record and structural validation of its fields;
2. the reference codec for the Enterprise-compatible Feather layout;
3. serialized value types needed to interpret record fields;
4. the linear timing calculation completely determined by the record;
5. shared par-file text transformations that do not require a loaded timing
   model.

**R-1.2 (outside the package).** `psrdata` does not own:

- nonlinear timing calculations;
- timing-engine protocol definitions;
- selection of timing software;
- parameter priors, sampling choices or physical-coordinate charts;
- PTA likelihoods, noise models or samplers.

nltiming owns the timing-engine protocols it consumes. `psrdata` implements
their linear case structurally and does not import nltiming.

**R-1.3 (materialized calculation).** A record contains the row data, timing
residuals, design matrix and parameter reference used for one materialized
timing calculation. It is not a view that re-queries timing software.

`psrdata` provides no mutation API. Callers are expected to treat a record as
read-only. If a caller modifies supplied arrays or mappings after construction,
the result is outside this contract.

**R-1.4 (no reordering, enforced by psrdata).** `psrdata` never permutes rows.
Row `i` of every row array is the producer's row `i`. A consumer that wants
time order sorts at its own layer.

---

## 2. Dependency position

**R-2.1 (dependency floor, enforced by psrdata).** Runtime dependencies are
NumPy and PyArrow. Importing `psrdata` must not import PINT, tempo2/libstempo,
JAX, Astropy, SciPy, Enterprise, Discovery, MetaPulsar or nltiming.

**R-2.2 (dependency direction).** MetaPulsar, vela-jax and nltiming may depend
on psrdata. psrdata must not import them. Enterprise and Discovery read the
Feather layout through their own readers.

**R-2.3 (consumer-defined engine interface).** nltiming defines the engine
interface. The psrdata linear engine must implement every required operation
for which the linear answer exists. Conformance is tested using nltiming's own
runtime and behavioural checks when nltiming is installed.

---

## 3. The `PulsarData` record

### 3.1 Core fields

With `n = len(toas)` and `p = len(fitpars)`, the record contains:

| field | type / shape | meaning |
|---|---|---|
| `name` | `str` | pulsar name |
| `setpars` | `tuple[str, ...]` | every timing-model parameter that is defined and has a value, whether fixed or fittable |
| `fitpars` | `tuple[str, ...]` | the subset of `setpars` that may be inferred; also the `Mmat` column order |
| `parameters` | `Mapping[str, ParameterFact]` | values, units and optional uncertainties for `setpars` (§3.3) |
| `toas` | float `(n,)`, TDB seconds | barycentric arrival times at PINT's barycentric cutoff (R-3.1.1) |
| `stoas` | float `(n,)`, seconds | site arrival times |
| `toaerrs` | float `(n,)`, seconds | unscaled TOA uncertainties |
| `residuals` | float `(n,)`, seconds | timing residuals at the recorded parameter values |
| `freqs` | float `(n,)`, MHz | barycentric observing frequency; `inf` where no frequency was reported |
| `Mmat` | float `(n, p)` | design matrix in fitter sign and PINT units |
| `flags` | `Mapping[str, str (n,)]` | TOA flags without the leading `-`; a missing per-row value is `""` |
| `backend_flags` | str `(n,)` | Enterprise backend label |
| `telescope` | str `(n,)` | observatory code |
| `pos` | float `(3,)` | ICRS unit vector at the reference epoch |
| `pos_t` | float `(n, 3)` | ICRS unit vector at each TOA |
| `sunssb` | float `(n, 6)`, lt-s | Sun position and, when available, velocity |
| `planetssb` | float `(n, 9, 6)`, lt-s | planet positions and, when available, velocities |
| `theta` | float, radians | colatitude, `π/2 - dec` |
| `phi` | float, radians | right ascension |
| `pdist` | `(float, float)`, kpc | distance and uncertainty; Enterprise fallback when unavailable |
| `dm` | float, pc cm⁻³ | model DM, `0.0` when absent |
| `dmx` | mapping or `None` | Enterprise's DMX table: `{name: {"DMX": float, "DMXerr": float or None, "DMXR1": float, "DMXR2": float, "fit": bool}}` keyed by the `DMX_nnnn` parameter name; `None` when the model has no DMX component. An empty mapping on a model that has DMX is refused, because it silently disables Enterprise's wideband model |
| `timing_package` | `Mapping[str, str]` | timing software used for each data set (§3.4); the constructor also accepts a scalar for a standalone record |
| `partim_compatibility` | `Mapping[str, Literal["pint", "tempo2"]]` | par/tim interpretation used for each data set (§3.4); the constructor also accepts a scalar |
| `residual_centering` | `Mapping[str, ResidualCentering]` | residual-centering information for each data set (§4); the constructor also accepts one scalar value |
| `producer` | `str` | the software that constructed and wrote the record (R-3.1.2) |
| `extra` | `Mapping[str, Any]` | optional additive producer metadata |

`schema` is a codec constant, not scientific record data.

**R-3.1.1 (`toas`, guaranteed by producers).** `toas` contains barycentric
arrival times in TDB seconds, that is TDB MJD multiplied by 86400, evaluated
at PINT's barycentric cutoff: the TDB arrival time minus every delay PINT's
chain applies before its binary component, and none of the delays it applies
from the binary component onward. It is not the fully corrected arrival time;
the two differ by the binary Roemer delay, which is seconds. Every producer
obtains this value from the equivalent point in its own timing chain,
whichever package read the files and whichever package calculated the
residuals. vela-jax's `Engine.reference_barycentric()` is one such
implementation over both par/tim compatibilities.

**R-3.1.2 (`producer`, guaranteed by producers).** `producer` names the
software that constructed and wrote the record. Recommended spellings are
`"metapulsar"` and `"vela_jax"`. It answers a different question from
`timing_package`: MetaPulsar writes records whose blocks were calculated by
other packages, and a later reader must be able to tell that the record is a
MetaPulsar combination without inspecting `extra`.

### 3.2 Structural validation

**R-3.2.1 (shapes, enforced by psrdata).**

- `toas`, `stoas`, `toaerrs`, `residuals`, `freqs`, `backend_flags` and
  `telescope` must have shape `(n,)`;
- `Mmat` must have shape `(n, p)`;
- `pos` must have shape `(3,)`;
- `pos_t` must have shape `(n, 3)`;
- `sunssb` must have shape `(n, 6)`;
- `planetssb` must have shape `(n, 9, 6)`;
- every flag value must have shape `(n,)`.

**R-3.2.2 (parameter names, enforced by psrdata).**

- `setpars` contains no duplicate names;
- `fitpars` contains no duplicate names;
- every `fitpar` occurs in `setpars`;
- every `setpar` has one entry in `parameters`;
- a producer-created fit parameter such as `Offset` or `PHOFF` is also added
  to `setpars` and `parameters`. Its `value` is a decimal representation of
  exactly zero (the token need not be `"0"`: `"0.0"` and other exact-zero
  decimals are accepted and stored as written), it has no `uncertainty`, and
  it uses the unit of its column's delta: `"s"` for `Offset` (a time offset,
  so its column is dimensionless) and `"dimensionless"` for `PHOFF` (a phase
  offset in turns, so its column is seconds per turn); a suffixed form takes
  the unit of its bare name. This zero/units/no-uncertainty check applies to
  phase-offset names in `fitpars`. A frozen `Offset` or `PHOFF` that appears
  only in `setpars` is a model parameter, not the §3.6 column, and is not
  required to be zero;
- a producer may not put two matrix columns under the same fit-parameter name.

**R-3.2.3 (mapping keys, enforced by psrdata).** `timing_package`,
`partim_compatibility` and `residual_centering` are nonempty mappings with
exactly the same data-set keys.

At construction, all three may instead be supplied as scalar single-data-set
values. psrdata normalizes them to one-entry mappings under the reserved key
`"single"`. Supplying a scalar for only some of the three fields is refused.

**R-3.2.4 (flags, enforced by psrdata).** Flag keys are nonempty strings and
flag values are strings.

**R-3.2.5 (finite values, enforced by psrdata).** Every numeric field is
finite, with these exceptions:

- `freqs` may contain `+inf` where no frequency was reported; `NaN` and
  `-inf` are refused, so there is one missing-frequency sentinel;
- `planetssb` may contain `NaN` for missing planet positions and unused
  velocity entries;
- unused Sun velocity columns (`sunssb` columns 3–5) may contain `NaN`.

`Mmat`, `residuals`, `toas`, `stoas`, `toaerrs`, `pos`, `pos_t`, Sun
position (`sunssb` columns 0–2), `theta`, `phi`, `dm` and both entries of
`pdist` are finite. `±inf` is not an allowed missing-value encoding outside
`freqs`.

### 3.3 `ParameterFact` and PINT units

```python
@dataclass
class ParameterFact:
    value: str
    units: str | None
    uncertainty: str | None = None
```

**R-3.3.1 (value).** `value` is the value used for the timing calculation.
For a numerical parameter it is a decimal representation in PINT's default
unit, obtained without a float64 round trip. For a textual parameter it is the
textual value.

The value is not required to preserve the exact spelling of the par-file
token. For example, a sexagesimal position may be converted to a decimal value
in its PINT unit. Such a conversion MUST preserve the full precision of the
reference value the timing package used: the converted decimal MUST round-trip
to the timing package's internal representation of that value, and MUST carry
at least as many significant digits as the par-file token. Decimal arithmetic
MUST use an explicitly set precision of at least 30 significant digits;
mathematical exactness is not required where the converted value has no finite
decimal representation, as when seconds of arc are divided by 3600.

**R-3.3.2 (units).** `units` is the PINT default unit label. It is required for
every numerical fit parameter. Dimensionless parameters use an explicit
dimensionless label. Textual parameters may use `None`.

**R-3.3.3 (uncertainty).** `uncertainty`, when present, is a decimal
representation in the same PINT unit as `value`. It records the uncertainty
reported by the timing solution. psrdata assigns it no inference meaning.

**R-3.3.4 (coverage).** `parameters` contains every name in `setpars`.
`fitpars` identifies which of those parameters form the record's fit
coordinates. No separate fitted/frozen field is stored.

**R-3.3.5 (derived engine views).** The linear engine derives:

- `reference_theta_exact()` from `parameters[name].value` in fitpar order;
- `native_units` from `parameters[name].units` in fitpar order.

Both are mappings keyed by fit-parameter name whose insertion order is
fitpar order.

`reference_theta()` is a derived float64 array of those decimal values,
each entry the correctly rounded value of the decimal string with no
intermediate float round trip. float64 is not the authority for a
parameter such as `F0`, whose decimal needs more digits than a float64
mantissa; a consumer that needs the recorded precision reads
`reference_theta_exact()` (or the facts). The linear calculation itself
is float64 because `Mmat` is float64.

The names are retained on the engine only because nltiming's timing-engine
interface uses them. The record does not duplicate these mappings.

### 3.4 Timing software and par/tim compatibility

**R-3.4.1 (`timing_package`).** For each data-set key,
`timing_package[key]` names the software that calculated that data set's
stored residual block and design-matrix block.

Recommended stable spellings are:

- `"pint"`;
- `"tempo2"` for tempo2/libstempo calculations;
- `"jug"`;
- `"vela"` for Vela.jl;
- `"vela_jax"`.

The vocabulary is open so a future timing implementation can identify itself
without a schema change.

**R-3.4.2 (`partim_compatibility`).** For each data-set key,
`partim_compatibility[key]` is:

- `"pint"` when PINT read, interpreted and completed the par/tim data under
  its rules and defaults;
- `"tempo2"` when tempo2 did so under its rules and defaults.

These are the two par/tim standards supported by this ecosystem. The field
names more than a set of parsing rules: the package that reads the files also
produces the frozen input quantities the calculation consumes, and those are
physical content of the record. They include the clock-corrected and
TT-to-TDB-converted arrival times, the site and solar-system ephemeris
vectors, the barycentric frequency correction, and the pulse-number
assignment (tempo2 honours `-pn` flags only under `TRACK -2`). Two records of
the same files with different `partim_compatibility` differ in `toas`,
`freqs`, `sunssb`, `planetssb` and `pos_t` at the level those packages
disagree, which is nanoseconds for the clock chain and can be a whole turn
for pulse numbers.

This field does not say which software calculated the residuals or matrix.
The two coincide for a pure PINT or a pure tempo2/libstempo record and differ
for every engine that separates the reading from the calculation: vela-jax
over a tempo2 read, JUG over a PINT or tempo2 session, Vela.jl over a PINT
read.

For example, a standalone vela-jax producer supplies:

```python
timing_package="vela_jax"
partim_compatibility="tempo2"
residual_centering=ResidualCentering(...)
```

The constructed record exposes:

```python
timing_package={"single": "vela_jax"}
partim_compatibility={"single": "tempo2"}
residual_centering={"single": ResidualCentering(...)}
```

**R-3.4.3 (standalone and named data sets).** A standalone producer does not
ask its caller for a data-set name. It supplies the three scalar values above,
and psrdata uses `"single"`.

MetaPulsar receives names as the keys of its input dictionaries
(`"EPTA_DR2"`, `"PPTA_DR3"`, and so on). It uses those names as the
keys of all three mappings when constructing a combined record. If MetaPulsar
is given a standalone record, it replaces the internal `"single"` key in the
new combined record with MetaPulsar's own input key.

The key identifies the timing data set, not the pulsar, timing package or
par/tim compatibility. It therefore must not be generated from a string such
as `"J1234+5678_PINT"`, whose meaning would change when the same data were
recalculated by another package.

After construction, every record exposes the same mapping shape: one entry for
a single data set and one entry per constituent data set for a combined
pulsar. There is no `"composite"` sentinel and no different record type.

### 3.5 Design matrix, coordinates and units

**R-3.5.1 (sign).** For `δ` in fitpar order,

```text
Δr ≈ -Mmat @ δ
```

If timing software exposes `J = ∂r/∂θ`, the producer stores `Mmat = -J`.

**R-3.5.2 (record linearization).** The complete linear timing calculation
defined by a record is:

```text
Δr = -Mmat @ δ
```

No timing package is needed to evaluate it.

**R-3.5.3 (PINT units everywhere).** Every timing parameter in a record uses
PINT's default unit:

- numerical `ParameterFact.value`;
- `ParameterFact.uncertainty`;
- each element of `δ`;
- each `Mmat` column scale;
- reference parameter values returned by the linear engine.

Column `j` therefore has units seconds per
`parameters[fitpars[j]].units`.

Any code that reads or evaluates tempo2/libstempo data must convert parameter
values, uncertainties and design-matrix columns to PINT units before creating
the record. The conversion occurs at that package boundary. A consumer never
applies a tempo2 unit heuristic to a record.

**R-3.5.4 (combined fit coordinates).** A combined record defines one global
fit coordinate for every `fitpars[j]`. Every contributing data-set block in
column `j` must be expressed in the same PINT unit and relative to the same
stored value. The combiner applies all name, scale and reference conversions
before inserting the blocks.

A data-set-specific fit parameter may be suffixed by its data-set key. Its
column uses the parameter's PINT default unit.

### 3.6 Phase-offset columns

**R-3.6.1 (one per data set, enforced by the linear engine).** Every data set
has exactly one phase-offset fit parameter:

- a single-data-set record may use bare `Offset` or `PHOFF`, or the suffixed
  form; the automatically inserted `"single"` key does not require an
  `_single` suffix;
- a record with several data sets uses `Offset_<key>` or `PHOFF_<key>` for
  each data-set key.

The linear engine locates a data set's phase-offset column as follows. The
candidates for key `k` are `Offset_<k>` and `PHOFF_<k>`, and, when the record
has one data set, also bare `Offset` and `PHOFF`. Exactly one candidate must
be present in `fitpars`; none or more than one is refused with
`LinearEngineError`. A bare name and a suffixed name for the same data set
therefore never coexist, and a several-data-set record never uses a bare
name.

**R-3.6.2 (row partition, enforced by the linear engine).** A data set's
phase-offset column is finite and nonzero on every row belonging to that data
set and exactly zero elsewhere. The supports of all data-set phase-offset
columns partition the rows without overlap or gaps.

**R-3.6.3 (pulsar-frame residual convention, guaranteed by producers).**
Residuals are formed by dividing phase residuals by pulsar-frame spin
frequency: the spin Taylor series where the timing software uses it, or the
constant `F0` where that is the package convention. A producer must not use
the topocentric Doppler-shifted frequency.

**R-3.6.4 (no separate direction field).** The phase-reference direction is
the named design-matrix column. No duplicate direction vector is stored.

### 3.7 Single and combined records

**R-3.7.1 (one record interface).** Single-data-set and combined records have
the same class, fields and operations. Enterprise and Discovery see either as
one pulsar with one row space and one `Mmat`.

**R-3.7.2 (data-set mappings).** The keys of `timing_package`,
`partim_compatibility` and `residual_centering` define the data sets represented
in the record.

**R-3.7.3 (no partition metadata).** A combined record does not serialize row
slices or a parameter-ownership map. Phase-offset-column support supplies the
row partition. For linear evaluation, columns that are nonzero on those rows
are the data set's active linear columns.

Numerical support is not claimed to express conceptual ownership of a timing
parameter.

**R-3.7.4 (selection flags).** A combined record carries PTA/data-set flags
needed by Enterprise and Discovery selections. Those flags are selection
metadata, not the authoritative row partition.

### 3.8 Row comparison

**R-3.8.1 (`TOARows`).** `record.toa_rows()` returns
`TOARows(stoas, freqs, toaerrs)`. It is an ordered alignment signature for
code that compares two reads of the same data. It is not a persistent
identifier and is not required to be unique. A record–engine pair produced by
one calculation requires no second-read alignment check.

---

## 4. Residual centering

```python
@dataclass
class ResidualCentering:
    stored_residuals: Literal[
        "none", "mean_removed", "constant_removed", "unknown"
    ]
    stored_weighted: bool | None = None
    standard_output: Literal[
        "none", "mean_removed", "constant_removed", "unknown"
    ] = "unknown"
    standard_weighted: bool | None = None
```

**R-4.1 (meaning).**

- `stored_residuals` states what centering was applied to the residual array in
  this record;
- `standard_output` states what the named timing package normally applies when
  reporting residuals;
- the corresponding weighted value is required only for `"mean_removed"` and
  must otherwise be `None`.

**R-4.1.1 (validation, enforced by psrdata).** Construction refuses a
`stored_residuals` or `standard_output` value outside the four literals, a
`stored_weighted` that is `None` when `stored_residuals == "mean_removed"`
or non-`None` otherwise, and the same for the `standard_*` pair. The
refusal is a `RecordError`, at record construction and at Feather read.

**R-4.2 (descriptive only).** Residual-centering information does not alter
`Mmat`, `residual_delta` or a PTA likelihood. The independent phase-offset
columns in §3.6 carry the scientifically required freedom. This field exists
to explain the stored residual convention and differences between timing
packages.

**R-4.3 (uniform mapping).** `residual_centering` always maps data-set key to
`ResidualCentering`: one entry for a single data set and several for a combined
record.

---

## 5. The record's linear engine

### 5.1 Public construction

These entry points are equivalent:

```python
record.linear_engine()
psrdata.linear_engine(record)
LinearTimingEngine.from_pulsar_data(record)
LinearTimingEngine.from_feather(path)
```

They return the complete linear calculation over the record's matrix.

### 5.2 Required behaviour

**R-5.2.1.** `fitpars` is the record's fitpar tuple.

**R-5.2.2.** `native_units` is a mapping from each fit parameter to its PINT
unit, keyed in fitpar order. The attribute implements nltiming's
`native_units` interface; its contents are always PINT units.

**R-5.2.3.** `reference_theta_exact()` is a mapping from each fit parameter
to its decimal `ParameterFact.value` string, keyed in fitpar order. This is
the authority for the recorded reference.

**R-5.2.4.** `reference_theta()` is a float64 array in fitpar order. Each
entry is the correctly rounded value of the corresponding decimal string,
with no intermediate float round trip. It is a derived view for float64
linear algebra, not a second stored reference.

**R-5.2.5.** `residual_delta(δ)` returns `-Mmat @ δ` and rejects the wrong
delta shape.

**R-5.2.6.** `design_matrix()` returns `Mmat`. The linear engine accepts only
`params=None`; another value is rejected.

**R-5.2.7.** `residual_jacobian()` returns `-Mmat`.

**R-5.2.8.** `identically_linear_fitpars()` returns every fit parameter.

**R-5.2.9.** `nonlinear_params` is `None`,
`binary_chart_capability(...)` returns `None`, and `engine_name` is
`"linear"`. The record's `timing_package` mapping retains the names of the
timing packages that calculated its stored residual and matrix blocks.

**R-5.2.10.** The engine does not import or expose JAX. A consumer may wrap
the matrix multiplication in its own array library.

### 5.3 Per-data-set contributions

**R-5.3.1.** For each data-set key, the linear engine:

1. locates the data set's phase-offset column;
2. obtains its rows from that column's support;
3. identifies active linear columns on those rows;
4. constructs one contribution over that matrix block.

The contributions' rows partition the full row space.

**R-5.3.2.** A contribution's fit parameters are its active linear columns.
Its `identically_linear_fitpars()` and any compatibility field carrying exact
linear parameters both report all of those fit parameters.

**R-5.3.3.** Evaluating every contribution and placing the results on its rows
equals the whole-record matrix multiplication.

**R-5.3.4.** Residual-centering information is exposed uniformly as a mapping.
A single-data-set engine returns one entry and a combined engine returns
several. No operation raises merely because the record has several data sets.

### 5.4 nltiming conformance

**R-5.4.1.** The linear engine directly follows nltiming's timing-engine
interface without importing nltiming. When that required interface changes,
psrdata implements the linear answer and its consumer test is updated.

---

## 6. Feather schema v1

### 6.1 Columns

The table columns are Enterprise-compatible:

- scalar columns: `toas`, `stoas`, `toaerrs`, `residuals`, `freqs`,
  `backend_flags`, `telescope`;
- vector blocks: `Mmat_<i>`, `sunssb_<i>`, `pos_t_<i>`;
- tensor block: `planetssb_<i>_<j>`;
- flags: `flags_<key>`.

No other column may use one of those reserved prefixes.

### 6.2 Metadata

The Arrow schema carries a JSON object under `b"json"`.

Required Enterprise-compatible keys:

- `name`, `dm`, `dmx`, `pdist`, `_pdist`, `pos`, `phi`, `theta`, `fitpars`,
  `setpars`.

Required psrdata keys:

- `schema`;
- `parameters`;
- `timing_package`;
- `partim_compatibility`;
- `residual_centering`;
- `producer`;
- `extra`.

Optional key:

- `noisedict`, filtered to this pulsar using Enterprise's convention.

Unknown additive metadata keys are ignored.

**R-6.2.0 (JSON forms of the psrdata keys).** Each key is a JSON value of the
following shape; the reader rebuilds the record from these and nothing else.

| key | JSON form |
|---|---|
| `schema` | string |
| `parameters` | object keyed by parameter name; each value is `{"value": string, "units": string or null, "uncertainty": string or null}`, with `uncertainty` present as `null` when absent |
| `timing_package` | object keyed by data-set key; each value a string |
| `partim_compatibility` | object keyed by data-set key; each value `"pint"` or `"tempo2"` |
| `residual_centering` | object keyed by data-set key; each value `{"stored_residuals": string, "stored_weighted": bool or null, "standard_output": string, "standard_weighted": bool or null}` |
| `producer` | string |
| `extra` | object; contents producer-defined |

The three data-set-keyed objects carry the same keys in the same insertion
order. A standalone record writes its single entry under `"single"`; the
reader does not re-derive the key. Numbers inside `parameters` are strings,
never JSON numbers, so that no float round trip occurs in the codec.

**R-6.2.1.** `schema` equals `pulsardata-feather-v1`.

**R-6.2.2.** psrdata refuses a missing or unknown schema rather than guessing.

### 6.3 Round trip and consumers

**R-6.3.1.** A psrdata record survives write/read field by field, including
parameter facts, per-data-set mappings, `+inf` frequencies, permitted `NaN`
planet slots and unused velocity entries, and flags.

**R-6.3.2.** Stock Enterprise and Discovery readers read psrdata-written
files without importing psrdata or understanding its additive metadata.

### 6.4 Versioning

**R-6.4.1.** Additive metadata does not change the schema when readers for the
schema can safely ignore it. Functionality requiring an optional key checks
for that key explicitly.

**R-6.4.2.** A change to the meaning of an array, field, unit, sign or shape
requires a new schema identifier.

**R-6.4.3.** A reader refuses an unknown schema. No forward-compatibility
guessing is allowed.

---

## 7. Par-text rules

`psrdata.partext` contains shared deterministic par-text transformations. It
does not instantiate a timing model or run timing software.

### 7.1 Active lines

**R-7.1.1.** `is_active_line(line)` is false for blank lines, lines whose
stripped form begins with `#`, and lines whose first token is the tempo2
comment marker `C`. It is true otherwise.

**R-7.1.2.** `active_lines(text)` yields active lines in their original order.

**R-7.1.3.** `line_key(line)` returns the upper-cased first token of an active
line.

### 7.2 Noise lines

**R-7.2.1.** `NOISE_NAMES` is an explicit set of upper-cased first tokens.
`is_noise_line(line)` is true exactly when the active line's key is in that
set. There are no prefix catch-alls.

The set contains the established white-noise, ECORR, red-noise, DM-noise,
chromatic-noise, solar-wind-noise and fit-summary names shared by MetaPulsar
and vela-jax. The exact membership in `psrdata.partext.NOISE_NAMES` is the
contract and is enumerated by its tests.

**R-7.2.2.** `DMJUMP` is not a noise line: it changes the timing model.

**R-7.2.3.** `strip_noise_lines(text)` removes noise lines, preserves the
remaining line order and returns text with a trailing newline. It does not
rewrite a file.

### 7.3 Timescale text

**R-7.3.1.** `effective_units(text)` implements the tempo2 textual rule:

- absent `UNITS` means `"TCB"`;
- `UNITS SI` means `"TCB"`;
- `UNITS TCB` means `"TCB"`;
- `UNITS TDB` means `"TDB"`;
- duplicate, missing or unknown values raise `ParTextError`.

This helper does not decide `partim_compatibility`. Code reading PINT-compatible
files applies PINT's own rules.

### 7.4 Keyword respelling

**R-7.4.1.** `respell_fdjump_for_pint(text)` rewrites tempo2 `FDJUMPn` to
PINT `FDnJUMP`, changing the first token only.

**R-7.4.2.** `respell_clock_for_tempo2(text)` rewrites PINT `CLOCK` to tempo2
`CLK`, changing the first token only.

No general alias table lives in psrdata.

### 7.5 Duplicate non-repeatable lines

**R-7.5.1.** `dedupe_nonrepeatable(text)` retains repeated keys listed in
`REPEATABLE_KEYS` and repeated FDJUMP-family lines. For another key, identical
duplicate values collapse to the first line and conflicting values raise
`ParTextError`.

This exists for known duplicated output from tempo2 transformations; it is not
a timing-model consistency checker.

### 7.6 Errors

**R-7.6.1.** `ParTextError(ValueError)` is the public error for malformed text
handled by this module.

---

## 8. Enterprise compatibility constants

These values retain Enterprise's established meanings and quirks.

**R-8.1 (`PLANET_SLOTS`).** Planet indices are Mercury `0`, Venus `1`, Earth
`2`, Mars `3`, Jupiter `4`, Saturn `5`, Uranus `6`, Neptune `7`, Pluto `8`.

**R-8.2 (`BACKEND_FLAG_ORDER`).** The precedence order is
`("f", "i", "sys", "g", "group")`. `backend_flags(flags, n)` first forms
`fe_be` where both are present and then lets each nonempty flag in that order
overwrite the preceding value.

**R-8.3 (`DEFAULT_DISTANCE_KPC`).** The Enterprise-compatible fallback is
`(1.0, 0.2)` kpc when no usable parallax distance is available.

**R-8.4 (ephemeris blocks).** Producers fill every supported planet position
they have. Missing positions and unused velocity entries may be `NaN`; a
producer does not discard an available value merely to imitate a more limited
reader path.

---

## 9. Producer contract

A conforming producer must:

1. identify the timing package that calculated each residual and matrix block;
2. identify each data set's PINT or tempo2 par/tim compatibility, meaning the
   package that read the files and produced the frozen input quantities;
3. identify itself as the producer that wrote the record;
4. produce residuals and `Mmat` from the same timing calculation;
5. take `toas` from the equivalent of PINT's barycentric cutoff in its own
   chain (R-3.1.1);
6. use identical TOA row order for every row array and matrix block;
7. convert every timing parameter, uncertainty, delta and matrix column to
   PINT units before record construction, preserving the reference precision
   (R-3.3.1);
8. provide one `ParameterFact` for every set parameter;
9. keep `fitpars` as a subset of `setpars`;
10. provide one independent phase-offset column per data set;
11. form residuals using the pulsar-frame convention;
12. describe residual centering for each data set.

psrdata validates the structural consequences it can observe. Producer
packages test the scientific statements that psrdata cannot independently
recalculate.

---

## 10. Record–engine agreement

The concrete record–engine pair lives in timing-software and MetaPulsar
packages, not in psrdata. A conforming pair agrees on:

- TOA row order;
- fitpar order;
- PINT units;
- parameter values;
- residual convention;
- phase-offset columns;
- reference design matrix.

At the recorded reference, the engine's design matrix equals `record.Mmat`.
When it exposes a residual Jacobian, `record.Mmat = -J`.

MetaPulsar selects timing software during construction. It may rename
parameters, convert units and combine blocks once. It must not later replace
the selected timing calculation with unrelated timing software while retaining
the record.

---

## 11. Explicit non-goals

psrdata does not:

- define live timing-engine protocols;
- select timing software;
- evaluate nonlinear delays;
- store timing priors or sampling policy;
- classify parameters into inference roles;
- preserve par-file aliases or original parameter spelling;
- assign persistent row identifiers;
- promise correctness after callers mutate a record;
- distinguish a combined record through a different class or protocol;
- store a duplicate phase-reference direction;
- define wideband timing.

---

## 12. Conformance

A conforming psrdata implementation satisfies the requirements assigned to
psrdata in this document. A conforming producer additionally satisfies §9 and
tests the scientific guarantees that psrdata cannot observe. A conforming
record–engine pair satisfies §10. A conforming consumer interprets the record
fields, units and schema as specified here.

Engine selection belongs to producer construction. nltiming receives the
constructed timing engine and does not choose PINT, tempo2, JUG, Vela.jl or
vela-jax.

---

## 13. Test obligations

A complete implementation must test:

1. dependency-floor import graph;
2. every structural record check in §3.2;
3. `fitpars ⊆ setpars` and complete parameter facts;
4. PINT-unit matrix calculations;
5. timing-package and par/tim-compatibility mapping agreement;
6. `ResidualCentering` validation;
7. `-Mmat @ δ`, exact references and unit views;
8. phase-offset support and per-data-set linear contributions;
9. lossless Feather round trip;
10. stock Enterprise and Discovery readers;
11. nltiming engine conformance;
12. each producer's record–engine agreement;
13. mixed PINT/tempo2 par/tim compatibility combined into PINT-unit matrix
    coordinates.

---

## 14. Errors

The public psrdata error categories are:

- `RecordError`: invalid record shapes, parameter coverage, mappings or
  phase-offset columns;
- `SchemaError`: missing, malformed or unsupported psrdata schema metadata;
- `LinearEngineError`: an invalid delta or impossible linear decomposition;
- `ParTextError`: malformed par text handled by `partext`.

Filesystem and PyArrow exceptions may propagate from file operations.
