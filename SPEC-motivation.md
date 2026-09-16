# psrdata — design motivation

This document explains the decisions in [`SPEC.md`](SPEC.md). The specification
is normative; this document records why its boundaries, fields and seams have
their present shape.

---

## 1. Why this package exists

Three separate concerns meet at this boundary:

1. **A timing-engine interface.** nltiming states which operations it needs,
   and timing software implements those operations structurally.
2. **A file layout.** Enterprise and Discovery consume the same Feather column
   layout. A layout needs a written specification and compatibility tests.
3. **Shared implementation.** MetaPulsar, vela-jax and nltiming need the same
   materialized record, metadata codec, linear matrix calculation and par-text
   rules, but none is an appropriate dependency of the other two.

Only the third concern requires `psrdata` as a package.

`psrdata` therefore does not own the general pulsar abstraction or the live
timing-engine protocols. It owns the concrete data product shared at the
boundary: the record, its Feather representation, the values needed to
interpret it and the linear calculation defined by its matrix.

This is intentionally a modest purpose. It centralizes shared code without
turning psrdata into the vocabulary layer for the entire PTA stack.

---

## 2. Dependency direction

MetaPulsar, vela-jax and nltiming deliberately do not import one another on
their basic construction paths:

- timing software must not depend on an inference package;
- the combiner must not require the inference layer to construct timing data;
- nltiming consumes an engine without knowing which package built it.

Shared code therefore needs a location below all three.

The NumPy/PyArrow dependency floor keeps that location honest. It does not
require an installation to avoid PINT; it ensures that the materialized record
and `-Mmat @ delta` do not themselves acquire timing-software dependencies.

Enterprise and Discovery are deliberately not required to import psrdata.
Their stock readers are compatibility authorities. A psrdata file is useful
only if those readers consume its specified columns.

---

## 3. Record, timing calculation and pair

Three objects are commonly called “the pulsar”:

- The materialized timing data: TOAs, residuals, design matrix, flags and
  parameter values. This is `PulsarData`.
- The callable timing calculation `r(theta)`. This is a timing engine.
- The two together. vela-jax's `TimingPulsar` and MetaPulsar's `TimingLeg`
  are examples.

The record is not a reduced timing package. It is the result of one timing
calculation at one reference model.

The record nevertheless defines one complete callable calculation:

```text
delta residual = -Mmat @ delta parameter
```

That linear calculation requires no clocks, ephemeris files, par parser or
timing implementation. Keeping it with the record makes the matrix convention
executable rather than merely documented.

The nonlinear function remains with the timing software that implements it.

### 3.1 Why the pair must agree

The record's reference residuals and matrix are used by the likelihood, while
the attached engine supplies timing changes away from the reference. If those
two halves come from different calculations, the likelihood can marginalize
one tangent while sampling another function.

The pair must therefore agree on:

- rows;
- fit coordinates;
- parameter values and units;
- residual convention;
- phase-offset columns;
- reference matrix.

Timing-package selection therefore occurs during MetaPulsar construction.
Replacing that timing package after construction would create the same
ambiguity.

There is no additional requirement that repeated engine access return the
same Python object. Scientific agreement of the record and calculation is the
contract.

---

## 4. No state fingerprint

A record contains the state needed for its linear calculation. It
does not change through normal use, and loading another record creates another
object. A partial hash or readable token adds no scientific guarantee.

A `state_id` would serve cache invalidation for mutable live pulsar objects,
not interpretation of a materialized record. It is excluded from psrdata
because:

- it is not a file-integrity checksum;
- it is not proof that two calculations used the same external files;
- it is not needed to evaluate or compare records;
- internal caching of live objects belongs to the package implementing that
  cache.

The record itself is the data product. No surrogate identifier is made part of
its scientific contract.

---

## 5. No immutability contract

The package does not need database-style ownership or mutation guarantees.
Normal PTA workflows construct the record and then read it.

`psrdata` keeps the record a frozen dataclass, because that is free and it is
what makes "no mutation API" true at the attribute level. It does nothing
else: no copying of inputs, no read-only flags on arrays, no recursive
freezing of metadata, no defence against a caller deliberately changing
arrays. There is one reader and no mutation in any workflow this record
serves; the data changes when a new release is published, years apart, at
which point new files are produced.

The simple rule is sufficient:

> psrdata supplies no mutation API; callers that modify a record leave the
> supported contract.

This avoids memory copies and machinery for a failure mode the workflow does
not exercise.

---

## 6. Timing package versus par/tim compatibility

The ecosystem needs two names because two independent questions are asked.

### 6.1 `timing_package`: what calculated the model?

In psrdata, **timing package** means the software that calculated the stored
residuals and design matrix.

Examples are:

- PINT;
- tempo2/libstempo;
- JUG;
- Vela.jl;
- vela-jax.

This is what a researcher needs when asking which timing implementation
produced the numbers in the record.

For a single data set the mapping contains one entry. For a combined record it
contains one entry per data set, because different parts may have been
calculated by different timing packages.

MetaPulsar combines those results; it is not substituted for the package that
calculated each block.

### 6.2 `partim_compatibility`: how were the input files understood?

There are currently two supported par/tim interpretation standards:

- PINT;
- tempo2.

They differ in accepted keywords, aliases, implicit defaults and how absent
values are completed. For example, an absent `UNITS` declaration does not have
the same interpretation on both paths.

`partim_compatibility` records which package read, interpreted and completed
each data set's files under its rules. It does not identify the code that
calculated the final matrix.

The name undersells what the reading package does, and the specification
says so (R-3.4.2). Reading is not parsing alone. The package that opens the
files also produces the frozen input quantities the calculation then
consumes: the clock chain and the TT-to-TDB conversion, the site and
solar-system ephemeris vectors, the barycentric frequency correction, and
the pulse-number assignment. Those quantities are stored in the record as
`toas`, `freqs`, `sunssb`, `planetssb` and `pos_t`, and two reads of the
same files under the two compatibilities differ in them at the level the
packages disagree. The field therefore names the source of physical content,
not a syntax dialect.

The question "is the reader ever not the calculator?" has a definite answer.
For a pure PINT or pure tempo2/libstempo record the two are the same
package. They differ in every engine that separates the two roles:

| configuration | reads the files, forms the frozen inputs | calculates residuals and `Mmat` |
|---|---|---|
| PINT | PINT | PINT |
| tempo2/libstempo | tempo2 | tempo2 |
| vela-jax over `Engine.from_tempo2` | tempo2 | vela-jax |
| JUG | a PINT or tempo2 session | JUG |
| Vela.jl via pyvela | PINT | Vela.jl |

So vela-jax can:

- read files under tempo2 rules, taking tempo2's clocks and pulse numbers;
- calculate residuals and a Jacobian itself;
- expose fit coordinates in PINT units.

No single “timing package” string can communicate all three facts. This is
why a MetaPulsar choice between PINT and tempo2 as the reader populates
`partim_compatibility`, while the software that calculates the residual and
matrix block populates `timing_package`.

### 6.2a Why the writer is recorded separately

`producer` names the software that constructed and wrote the record. It is a
third question. A MetaPulsar record's blocks were calculated by other
packages and read under compatibilities MetaPulsar chose, but the
combination, the parameter renaming, the unit conversion and the file were
MetaPulsar's. A reader that sees `timing_package={"EPTA_DR2": "vela_jax",
"PPTA_DR3": "vela_jax"}` cannot infer from that alone which software combined
the blocks. Reproduction therefore requires `producer` as a separate field.

### 6.3 Why both are mappings

A single-data-set pulsar and a combined pulsar should have the same interface.
MetaPulsar has meaningful data-set names: they are the keys supplied
to `create_metapulsar()` or the direct `MetaPulsar` constructor. Those names
flow through its per-PTA dictionaries and become names such as `"EPTA_DR2"` or
`"PPTA_DR3"`.

Standalone timing software has no equivalent name:

- a PINT model/TOAs pair has none;
- a libstempo pulsar has none;
- `vela_jax.TimingPulsar.from_files()` receives no data-set label;
- MetaPulsar's `TimingLeg` gets its PTA identity from the surrounding
  MetaPulsar dictionary, not from the leg object.

Requiring standalone callers to invent a label would add work without adding
information. psrdata therefore accepts scalar metadata for a standalone
record:

```python
timing_package="vela_jax"
partim_compatibility="tempo2"
residual_centering=ResidualCentering(...)
```

It normalizes those values to:

```python
timing_package={"single": "vela_jax"}
partim_compatibility={"single": "tempo2"}
residual_centering={"single": ResidualCentering(...)}
```

`"single"` is deliberately not a scientific name. A generated label such as
`"J1234+5678_PINT"` would duplicate the pulsar name and compatibility, and it
would change if the same data were recalculated by JUG or vela-jax. That would
incorrectly turn a software choice into data-set identity.

When MetaPulsar combines records, it constructs mappings using its
input keys. The standalone `"single"` key is not retained as a leg name.

After normalization, the public shape is uniform: the mappings have one entry
for one data set and several entries when several data sets are combined. No
`"composite"` sentinel or alternate protocol is needed. A standalone record
may keep a bare `Offset` or `PHOFF`; it does not acquire an `Offset_single` or
`PHOFF_single` name merely because its metadata uses the reserved key.

---

## 7. PINT units are the ecosystem standard

Package-native units would leave a combined fit coordinate ill-defined when
one block comes from PINT-family calculations and another from
tempo2/libstempo. The record therefore adopts one rule:

> Every timing parameter, uncertainty, delta and design-matrix column in a
> psrdata record uses PINT's default unit for that parameter.

This is a deliberate standardization, not a preference for which timing
calculation is physically correct.

PINT has a developed parameter-unit model. tempo2 and libstempo do not expose
equally transparent unit handling through their Python interface. Code that
interacts with tempo2/libstempo must know where conversions are
required. The reliable place to perform them is at that boundary, before the
record is created.

MetaPulsar applies this pattern to design-matrix columns, including
the angular-coordinate scalings where PINT and tempo2 expose different units.
Making the rule universal prevents a consumer from having to infer a column's
scale from the timing package name.

### 7.1 Combined coordinates

A combined matrix has one global delta vector. A shared column cannot use
degrees for one data-set block and radians for another.

The combiner therefore:

1. chooses the record's canonical parameter name;
2. converts each data-set value and matrix block to the PINT unit;
3. verifies the shared reference value;
4. inserts every block into one global column.

File-only linear evaluation needs no record of the source scale factors after
conversion. Their effects are in `Mmat`.

---

## 8. Parameter facts without duplication

nltiming needs several simple timing-solution facts:

- exact parameter values;
- units;
- uncertainties;
- fixed values such as `PB` that may be relevant alongside fitted binary
  parameters.

These are facts about the timing solution, not an inference recipe. They belong
in the materialized record.

The minimal representation is:

```python
ParameterFact(value, units, uncertainty=None)
```

The mapping covers `setpars`. `fitpars` is a subset selecting the parameters
that may be inferred and defining matrix-column order.

This establishes the intended meanings:

- **set parameter:** defined and has a value, whether fixed or fittable;
- **fit parameter:** a set parameter exposed as an inference coordinate.

There is no separate fitted/frozen flag.

### 8.1 Why one unit is enough

All numerical facts use PINT units. The record therefore needs only `units`,
not separate native and display units.

This unit is used for:

- the recorded value;
- the recorded uncertainty;
- the parameter delta;
- the matching `Mmat` column.

Consumers may format values differently for presentation, but that is not a
second scientific unit in the record.

### 8.2 Why derived views are not stored

`reference_theta_exact` duplicated `parameters[name].value` for fit parameters.
`native_units` duplicated `parameters[name].units`.

The record does not store either view. The linear engine derives the methods
and attributes nltiming expects from `parameters`:

```text
reference_theta_exact() = fitpar values
native_units             = fitpar PINT units
```

This leaves one authority for each value.

`reference_theta()` is not that authority. It is a float64 array derived by
correctly rounding each decimal string. float64 is too little precision to
carry a millisecond-pulsar `F0` over a PTA span, which is why the facts
store decimal strings and why a consumer that needs the recorded digits
reads `reference_theta_exact()`. The linear calculation `Δr = -Mmat @ δ`
remains float64 because `Mmat` is float64; a higher-precision reference
vector would not recover those bits.

### 8.3 What is deliberately not stored

The record does not store:

- inference roles or parameter categories;
- aliases or original par spelling;
- attached PINT prior objects;
- a second display-unit system;
- generic syntax trees for repeated par lines;
- nonlinear chart decisions.

Those are unnecessary for interpreting the stored matrix. A consumer may use
the available values, units and uncertainties however it chooses.

---

## 9. Single and combined records are the same object

Enterprise and Discovery consume one pulsar-shaped collection of arrays. They
do not need to know whether the rows originated in one timing data set or
several.

psrdata preserves that property:

- one class;
- one set of fields;
- one matrix;
- one delta vector;
- one `linear_engine()` operation.

Per-data-set metadata is represented by mappings. The one-entry and
several-entry cases have identical APIs.

### 9.1 Why no partition block is stored

Every independent timing data set needs its own arbitrary phase-offset
parameter. Its design-matrix column is nonzero on that data set's rows and zero
elsewhere. Those supports deliberately partition the rows.

That gives the linear engine all structure it needs to expose per-data-set
blocks. A second row-partition structure would duplicate information.

The nonzero support of another parameter is called its active linear support,
not ownership. A derivative can be zero at a particular reference point
without changing which physical model contains that parameter.

---

## 10. Residual centering, not gauge provenance

Independent phase-offset columns are scientifically necessary. They prevent a
combined likelihood from treating the arbitrary phase relation between, for
example, EPTA and PPTA timing solutions as measured information.

Whether a timing package subtracts a mean from the residuals is different. It
changes how the stored residual vector is presented, but it does not change a
likelihood that marginalizes the required phase-offset column.

The record keeps this information because it helps explain differences
between timing packages and stored products. The name `ResidualCentering`
states what the information is without introducing general “provenance” or
gauge terminology into the user-facing API.

The two recorded facts are:

- what centering was applied to the residuals actually stored;
- what the named timing package normally does in its standard output.

Weighted-mean information is recorded where applicable. The values are
descriptive only; no psrdata arithmetic branches on them.

Like timing-package metadata, residual centering is always a mapping by data
set. A combined pulsar therefore needs no different engine protocol and no
singular method that fails when several data sets are present.

---

## 11. Row order and comparison

The producer's row order is authoritative. psrdata never sorts.

`TOARows(stoas, freqs, toaerrs)` is a small ordered signature for code that
compares two reads. It is not a permanent row-identity system.

A timing calculation that emits its record and engine together needs no such
comparison because there is no second read to reconcile.

---

## 12. Structural validation versus timing correctness

psrdata can validate:

- shapes;
- parameter-name uniqueness and coverage;
- `fitpars` being a subset of `setpars`;
- parameter facts existing for all set parameters;
- agreement of data-set mapping keys;
- the phase-offset-column partition;
- numeric fields being finite, except `+inf` in `freqs` and `NaN` in
  `planetssb` and unused velocity columns.

It cannot independently validate:

- which timing package calculated a matrix;
- whether residuals and matrix came from the same calculation;
- whether a tempo2 unit conversion was correct;
- whether exact parameter values match the calculation;
- whether the residual divisor matches the stated convention.

Those are producer obligations. They are tested where the timing calculation
is available: vela-jax, MetaPulsar and each future producer.

This is the seam:

> psrdata validates the record it can see; the producer guarantees the timing
> meaning of the values it supplied.

---

## 13. Feather compatibility and versioning

The array-column layout belongs to the established Enterprise-compatible
format. psrdata cannot rename or reinterpret those columns without breaking
readers that do not import psrdata.

The metadata needed for the psrdata record consists of:

- parameter facts;
- timing-package mapping;
- par/tim-compatibility mapping;
- residual-centering mapping;
- the schema tag.

Enterprise and Discovery ignore these metadata keys.

Additive metadata does not require a schema bump when readers for the schema
can safely ignore it. A change to the meaning of an array, unit, sign or shape
requires a new schema identifier.

Wideband timing changes the measurement representation and lies outside this
contract.

---

## 14. Why protocol definitions stay in nltiming

A timing-engine protocol is a statement by nltiming about what nltiming needs.
It therefore stays in nltiming.

The psrdata linear engine directly implements that interface without importing
its classes. Consumer tests enforce every required operation for which a
linear answer exists.

No additional compatibility version or intermediate capability layer is
needed. All relevant packages are maintained together and tested against the
actual interface.

Combined pulsars require no separate protocol. They perform the same whole-row
operations as single-data-set pulsars. Any information that naturally has one
value per data set is returned as a mapping in both cases.

---

## 15. Why `partext` stays here

The par-text helpers are not conceptually part of the timing-data record. They
belong in psrdata for a practical reason: MetaPulsar and vela-jax require the
same deterministic transformations. One shared implementation keeps those
transformations identical.

They stay narrowly limited to transformations that do not instantiate a timing
model or run timing software. Anything requiring a loaded model remains in the
producer.

---

## 16. Architecture

1. A producer selects a timing package for each data set.
2. The par/tim files are interpreted under declared PINT or tempo2
   compatibility.
3. The timing package calculates residuals and its design matrix.
4. The producer converts every parameter quantity and matrix column to PINT
   units.
5. The producer emits one record and its matching timing engine.
6. MetaPulsar, when used, combines those pairs into one PINT-unit row and
   parameter space.
7. psrdata serializes the record and supplies its exact linear calculation.
8. nltiming consumes the constructed engine and applies inference policy.
9. Enterprise and Discovery consume the established array layout.

The contract has:

- no state fingerprint;
- no deep immutability machinery;
- no separate combined-pulsar protocol;
- no duplicate reference or unit mappings;
- no general parameter taxonomy;
- one timing-package name for the software that calculated each block;
- one producer name for the software that wrote the record;
- one par/tim-compatibility declaration for how each data set was interpreted;
- PINT units throughout the ecosystem.
