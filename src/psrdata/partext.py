"""Par-file rules that are pure text.

Everything here is a function from par text to par text (or to a verdict about
one line), with no subprocess, no PINT, no alias table and no physics. That is
the boundary: the moment a rule needs to know what a parameter *means* it
belongs to a timing package, not here.

These rules were duplicated across MetaPulsar and vela-jax, which is how the
two noise classifiers came to disagree — MetaPulsar's caught ``TNEFAC`` and
missed ``TRES``; vela-jax's the reverse — while a cross-repo byte-identity test
asserted they were the same. :func:`is_noise_line` below is the union, and the
union is now the only copy.
"""

from __future__ import annotations

import re
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
    upper = stripped.upper()
    return not (upper == "C" or upper.startswith("C "))


def active_lines(text: str) -> Iterator[str]:
    """Every active line of ``text``, in order."""
    for line in text.splitlines():
        if is_active_line(line):
            yield line


def line_key(line: str) -> str:
    """Upper-cased first token of an active par line."""
    return line.split()[0].upper()


# --- noise lines -----------------------------------------------------------

#: The union of MetaPulsar's ``NOISE_PAR_KEYS`` and vela-jax's
#: ``freeze.NOISE_NAMES``. Neither alone is right: MetaPulsar's spells the
#: tempo2 EFAC family ``TNEFAC``/``TNEQUAD`` and would leave vela-jax's
#: ``TNEF``/``TNEQ``, ``TRES``, ``DMRES`` in a stripped par; vela-jax's is the
#: mirror image. Combining them is safe because every key here names a
#: white/red-noise hyperparameter, and none of them is a delay column on a
#: narrowband path.
#:
#: ``DMJUMP``/``DMEFAC``/``DMEQUAD`` apply only to wideband DM *measurements*.
#: They stay in the set so a stripped par drops them the same way until a
#: wideband path exists.
# fmt: off
NOISE_NAMES = frozenset({
    "EFAC", "TNEFAC", "T2EFAC", "EQUAD", "TNEQUAD", "T2EQUAD",
    "ECORR", "TNECORR", "DMEFAC", "DMEQUAD", "DMJUMP",
    "RNAMP", "RNIDX",
    "TNREDAMP", "TNREDGAM", "TNREDC", "TNREDF", "TNREDFC",
    "TNDMAMP", "TNDMGAM",
    "TNCHROMAMP", "TNCHROMGAM", "TNCHROMIDX",
    "TNGAMMA", "TNAMP",
    "TNEF", "TNEQ", "TNGLOBALEF", "TNGLOBALEQ", "TRES", "DMRES",
})
NOISE_PREFIXES = (
    "EFAC", "EQUAD", "ECORR", "T2EFAC", "T2EQUAD", "TNEFAC", "TNEQUAD",
    "DMEFAC", "DMEQUAD", "DMJUMP",
    "TNRED", "TNDM", "TNCHROM", "TNSW", "PLRED", "PLDM", "PLCHROM", "CHI2",
)
# fmt: on

#: Timing keywords that begin ``TN``/``RN`` and are *not* noise. Without these
#: the catch-alls below would strip a par's phase connection (``TRACK``), its
#: TT->TDB ephemeris (``TIMEEPH``), its celestial-frame method (``T2CMETHOD``)
#: or its right ascension (``RA``/``RAJ``).
_TN_EXCEPTIONS = frozenset({"TRACK", "TIMEEPH", "T2CMETHOD"})
_RN_EXCEPTIONS = frozenset({"RA", "RAJ"})


def is_noise_line(line: str) -> bool:
    """True when ``line`` is a white/red-noise hyperparameter, not a delay.

    Comments and blanks are not noise. The ``TN*`` / ``RN*`` catch-alls exist
    because tempo2's noise vocabulary is open-ended; the exception sets above
    are what keeps them from swallowing timing keywords.
    """
    if not is_active_line(line):
        return False
    key = line_key(line)
    if key in NOISE_NAMES:
        return True
    if key.startswith("TN") and key not in _TN_EXCEPTIONS:
        return True
    if key.startswith("RN") and key not in _RN_EXCEPTIONS:
        return True
    return any(key.startswith(prefix) for prefix in NOISE_PREFIXES)


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

_FDJUMP_TEMPO2 = re.compile(r"^FDJUMP(\d+)$", re.I)


def respell_fdjump_for_pint(text: str) -> str:
    """Rewrite tempo2 ``FDJUMPn`` mask parameters to PINT's ``FDnJUMP``."""
    out = []
    for line in text.splitlines():
        head, _, rest = line.partition(" ")
        match = _FDJUMP_TEMPO2.match(head)
        out.append(f"FD{int(match.group(1))}JUMP {rest}" if match else line)
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
        tokens = line.split()
        if tokens and tokens[0].upper() == "CLOCK":
            out.append(" ".join(["CLK"] + tokens[1:]))
        else:
            out.append(line)
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
        return float(first) == float(second)
    except ValueError:
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
    "NOISE_PREFIXES",
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
