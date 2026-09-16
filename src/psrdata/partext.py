"""Par-file rules that are pure text.

Everything here is a function from par text to par text (or to a verdict about
one line), with no subprocess, no PINT, no alias table and no physics. That is
the boundary: the moment a rule needs to know what a parameter *means* it
belongs to a timing package, not here.

These rules were duplicated across timing packages, which is how two noise
classifiers came to disagree — one caught ``TNEFAC`` and missed ``TRES``; the
other the reverse — while a cross-repo byte-identity test asserted they were
the same. :func:`is_noise_line` below is the union, and the union is now the
only copy.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Iterator

from .errors import ParTextError

#: Timescales a par file can be written in.
TIMESCALES = ("TDB", "TCB")

# --- active lines ----------------------------------------------------------


def is_active_line(line: str) -> bool:
    """True for a parameter/directive line: not blank, not a comment.

    Both comment spellings count: ``#`` (PINT) and a leading ``C`` token
    (tempo2 ``readParfile.C``).
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return False
    return stripped.split()[0].upper() != "C"


def active_lines(text: str) -> Iterator[str]:
    """Every active line of ``text``, in order."""
    for line in text.splitlines():
        if is_active_line(line):
            yield line


def line_key(line: str) -> str:
    """Upper-cased first token of an active par line."""
    return line.split()[0].upper()


# --- noise lines -----------------------------------------------------------

#: Exact first-token identity, upper-cased. No prefix catch-alls: those are
#: how a delay keyword gets stripped by accident.
#:
#: The names that actually matter to a residual ingest are the white-noise
#: scalings (``EFAC``/``EQUAD`` and their tempo2/PINT aliases) and ``ECORR``
#: (Vela.jl uses it to decide whether to reorder TOAs). Everything else on
#: this list — power-law hyperparameters, fit-summary lines (``TRES``,
#: ``CHI2``), wideband DM-error scalings (``DMEFAC``/``DMEQUAD``) — is inert
#: in a delay-only engine, but is still stripped so the packages that ingest
#: these objects do not have to ignore them.
#:
#: Wideband (when a producer starts emitting it): stop stripping
#: ``DMEFAC``/``DMEQUAD`` — they are PINT ``ScaleDmError`` / Vela.jl
#: ``DispersionMeasurementNoise``, the DM analogue of EFAC/EQUAD. Keep
#: ``DMJUMP``. PINT forbids mixing narrowband and wideband TOAs in one
#: object; Vela.jl's ``WidebandTOA`` carries ``DMInfo(value, error)`` beside
#: the TOA and ``form_residual`` returns ``(tres, dmres)``. ECORR is refused
#: on wideband in pyvela. See :class:`~psrdata.record.PulsarData` for the
#: array-side of that bump (``flags["pp_dm"]``/``pp_dme`` already satisfy
#: Enterprise's ``WidebandTimingModel``).
# fmt: off
NOISE_NAMES = frozenset({
    # white-noise scaling (PINT ScaleToaError + tempo2 spellings)
    "EFAC", "T2EFAC", "TNEF", "TNEFAC",
    "EQUAD", "T2EQUAD", "TNEQ", "TNEQUAD",
    "TNGLOBALEF", "TNGLOBALEQ",
    # ECORR: Vela.jl sorts TOAs when this is present (narrowband only)
    "ECORR", "TNECORR",
    # wideband DM-error scaling (PINT ScaleDmError)
    "DMEFAC", "DMEQUAD",
    # power-law red / DM / chromatic / solar-wind (inert for residuals)
    "RNAMP", "RNIDX",
    "TNREDAMP", "TNREDGAM", "TNREDC", "TNREDF", "TNREDFC",
    "TNREDFLOG", "TNREDFLOG_FACTOR", "TNREDTSPAN",
    "TNDMAMP", "TNDMGAM", "TNDMC", "TNDMFLOG", "TNDMFLOG_FACTOR", "TNDMTSPAN",
    "TNCHROMAMP", "TNCHROMGAM", "TNCHROMC", "TNCHROMIDX",
    "TNCHROMFLOG", "TNCHROMFLOG_FACTOR", "TNCHROMTSPAN",
    "TNSWAMP", "TNSWGAM", "TNSWC", "TNSWFLOG", "TNSWFLOG_FACTOR",
    "TNGAMMA", "TNAMP",
    "PLREDFREQ", "PLREDAMP",
    # fit summaries (PINT TimingModel; Vela.jl ignores them)
    "TRES", "DMRES", "CHI2", "CHI2R",
})
# fmt: on


def is_noise_line(line: str) -> bool:
    """True when the line's first token is in :data:`NOISE_NAMES`.

    Comments and blanks are not noise. Matching is exact, not a prefix.
    """
    if not is_active_line(line):
        return False
    return line_key(line) in NOISE_NAMES


def strip_noise_lines(text: str) -> str:
    """Remove noise hyperparameter lines. The caller's file is never rewritten."""
    kept = [line for line in text.splitlines() if not is_noise_line(line)]
    return "\n".join(kept) + "\n"


# --- timescale -------------------------------------------------------------


def effective_units(text: str) -> str:
    """The timescale a par file is actually written in.

    Absent ``UNITS`` means TCB, and ``UNITS SI`` is tempo2's synonym for TCB.
    Duplicate active ``UNITS`` lines are refused rather than resolved by a
    silent first- or last-wins rule.
    """
    units = [line for line in active_lines(text) if line_key(line) == "UNITS"]
    if len(units) > 1:
        raise ParTextError(
            f"{len(units)} active UNITS lines in the par file; refusing to guess "
            "which one tempo2 would have used"
        )
    if not units:
        return "TCB"
    tokens = units[0].split()
    if len(tokens) < 2:
        raise ParTextError(f"UNITS line has no value: {units[0]!r}")
    value = tokens[1].upper()
    if value in ("TCB", "SI"):
        return "TCB"
    if value == "TDB":
        return "TDB"
    raise ParTextError(f"unknown UNITS value {tokens[1]!r}; expected TDB, TCB or SI")


# --- keyword respellings ---------------------------------------------------

_FDJUMP_TEMPO2 = re.compile(r"^(\s*)FDJUMP(\d+)(?=\s|$)", re.I)
_CLOCK_PINT = re.compile(r"^(\s*)CLOCK(?=\s|$)", re.I)


def respell_fdjump_for_pint(text: str) -> str:
    """Rewrite tempo2 ``FDJUMPn`` mask parameters to PINT's ``FDnJUMP``.

    Only the first token changes; the rest of the line is kept verbatim.
    """
    out = []
    for line in text.splitlines():
        out.append(
            _FDJUMP_TEMPO2.sub(lambda m: f"{m.group(1)}FD{int(m.group(2))}JUMP", line)
        )
    return "\n".join(out) + "\n"


def respell_clock_for_tempo2(text: str) -> str:
    """Rewrite PINT's ``CLOCK`` keyword to the ``CLK`` tempo2 parses.

    PINT names the parameter ``CLOCK`` with ``CLK`` as an alias; tempo2's
    ``readParfile.C`` knows only ``CLK`` and silently falls back to its default
    realisation for anything else. A par written by PINT therefore pins the
    clock chain for one timing package and not the other -- worth 234 ns of
    PINT-tempo2 residual difference on the Vela.jl fixtures. This is the mirror
    of :func:`respell_fdjump_for_pint`: the same numbers, spelled so both codes
    read them.
    """
    out = []
    for line in text.splitlines():
        out.append(_CLOCK_PINT.sub(r"\1CLK", line))
    return "\n".join(out) + "\n"


# --- duplicate non-repeatable lines ---------------------------------------

#: Parameters a par file may legitimately list more than once, because each
#: line carries its own selector.
# fmt: off
REPEATABLE_KEYS = frozenset({
    "JUMP", "DMJUMP", "EFAC", "EQUAD", "ECORR", "T2EFAC", "T2EQUAD",
    "TNEF", "TNEQ", "TNECORR", "DMEFAC", "DMEQUAD",
})
# fmt: on
_FDJUMP_ANY = re.compile(r"^FD\d*JUMP\d*$")


def _same_value(first: str | None, second: str | None) -> bool:
    if first == second:
        return True
    if first is None or second is None:
        return False
    try:
        return Decimal(first) == Decimal(second)
    except (InvalidOperation, ValueError):
        return False


def dedupe_nonrepeatable(text: str) -> str:
    """Collapse a doubled non-repeatable line, keeping the first.

    Old tempo2 builds emit ``NE_SW`` twice from ``-gr transform``, which PINT
    then refuses. First-token identity, upper-cased, no alias table: this is
    for text tempo2 itself wrote, with canonical keywords. A par that spells
    one parameter two ways (``E`` and ``ECC``) is an ingest problem for the
    caller's PINT-aware sanitizer, not for this function.

    Two lines with genuinely different values are refused rather than resolved:
    silently keeping one of them would change the model.
    """
    seen: dict[str, str | None] = {}
    kept: list[str] = []
    for line in text.splitlines():
        if not is_active_line(line):
            kept.append(line)
            continue
        tokens = line.split()
        key = tokens[0].upper()
        if key in REPEATABLE_KEYS or _FDJUMP_ANY.match(key):
            kept.append(line)
            continue
        value = tokens[1] if len(tokens) > 1 else None
        if key not in seen:
            seen[key] = value
            kept.append(line)
        elif not _same_value(seen[key], value):
            raise ParTextError(
                f"parameter {key} appears twice with different values "
                f"({seen[key]!r}, {value!r})"
            )
    return "\n".join(kept) + "\n"


__all__ = [
    "TIMESCALES",
    "NOISE_NAMES",
    "REPEATABLE_KEYS",
    "is_active_line",
    "active_lines",
    "line_key",
    "is_noise_line",
    "strip_noise_lines",
    "effective_units",
    "respell_fdjump_for_pint",
    "respell_clock_for_tempo2",
    "dedupe_nonrepeatable",
]
