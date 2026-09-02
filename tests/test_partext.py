"""The par-text rules: text in, verdict or text out. No PINT anywhere."""

from __future__ import annotations

import pytest

from psrdata import ParTextError
from psrdata.partext import (
    active_lines,
    dedupe_nonrepeatable,
    effective_units,
    is_active_line,
    is_noise_line,
    respell_clock_for_tempo2,
    respell_fdjump_for_pint,
    strip_noise_lines,
)


@pytest.mark.parametrize(
    "line",
    [
        "EFAC -f L-wide_ASP 1.0",
        "T2EQUAD -f L-wide 0.1",
        "ECORR -f L-wide 0.5",
        "TNRedAmp -13.5",
        "TNDMGam 2.0",
        "PLREDFREQ 1.0",
        "RNAMP 1e-14",
        "DMJUMP -fe Rcvr 1e-3",
        "CHI2 1234.5",
        # MetaPulsar's classifier caught these and vela-jax's did not
        "TNEFAC -f L 1.0",
        "TNEQUAD -f L 0.1",
        # vela-jax's caught these and MetaPulsar's did not
        "TRES 1.04",
        "DMRES 0.5",
        "TNEF -f L 1.0",
        "TNEQ -f L -6.0",
        "TNSWAMP -12.0",
        "PLREDAMP -13.0",
    ],
)
def test_the_union_classifier_catches_both_vocabularies(line):
    """Neither original classifier was right on its own.

    MetaPulsar's spelled the tempo2 EFAC family ``TNEFAC``/``TNEQUAD`` and
    would have left ``TNEF``/``TRES`` in a stripped par; vela-jax's was the
    mirror image. A cross-repo byte-identity test asserted they agreed.
    """
    assert is_noise_line(line)


@pytest.mark.parametrize(
    "line",
    [
        "F0 61.485476554371304592 1 1.7e-11",
        "JUMP -fe Rcvr_800 0.1 1",
        "DM 160.0 1",
        "ELAT 5.0 1",
        "FD1 0.001 1",
        "TZRMJD 55000.0",
        # the TN*/RN* catch-alls must not swallow these
        "TRACK -2",
        "TIMEEPH FB90",
        "T2CMETHOD IAU2000B",
        "RAJ 18:53:57.3",
        "RA 18:53:57.3",
    ],
)
def test_timing_keywords_survive(line):
    assert not is_noise_line(line)


@pytest.mark.parametrize("line", ["", "   ", "# comment", "C commented out", "C"])
def test_comments_and_blanks_are_not_noise(line):
    assert not is_noise_line(line)
    assert not is_active_line(line)


def test_strip_keeps_order_and_ends_with_a_newline():
    assert strip_noise_lines("F0 1.0\nEFAC -f x 1.0\nDM 2.0\n") == "F0 1.0\nDM 2.0\n"


def test_active_lines_skips_both_comment_spellings():
    text = "# pint\nC tempo2\nF0 1.0\n\nDM 2.0\n"
    assert list(active_lines(text)) == ["F0 1.0", "DM 2.0"]


@pytest.mark.parametrize(
    "line,expected",
    [("UNITS TDB", "TDB"), ("UNITS TCB", "TCB"), ("UNITS SI", "TCB")],
)
def test_units_values(line, expected):
    assert effective_units(f"F0 1.0\n{line}\n") == expected


def test_an_absent_units_line_means_tcb():
    """A tempo2 par is TCB unless it says otherwise -- the whole reason the
    conversion path exists."""
    assert effective_units("F0 1.0\n") == "TCB"


def test_a_commented_units_line_is_not_a_units_line():
    assert effective_units("C UNITS TDB\nF0 1.0\n") == "TCB"


@pytest.mark.parametrize(
    "text",
    ["UNITS TDB\nUNITS TCB\n", "UNITS\n", "UNITS BARYCENTRIC\n"],
)
def test_ambiguous_or_unknown_units_are_refused(text):
    """Refused rather than resolved by a silent first- or last-wins rule."""
    with pytest.raises(ParTextError):
        effective_units(text)


def test_fdjump_is_respelled_for_pint():
    text = "FDJUMP1 -sys A 1.0 1\nFDJUMP12 -sys B 2.0\nFD1 3.0\n"
    assert (
        respell_fdjump_for_pint(text)
        == "FD1JUMP -sys A 1.0 1\nFD12JUMP -sys B 2.0\nFD1 3.0\n"
    )


def test_clock_is_respelled_for_tempo2():
    """tempo2's ``readParfile.C`` knows only ``CLK``; PINT writes ``CLOCK``."""
    assert respell_clock_for_tempo2("CLOCK  TT(BIPM2023)\nF0 1.0\n") == (
        "CLK TT(BIPM2023)\nF0 1.0\n"
    )
    assert respell_clock_for_tempo2("CLKCORR 1\n") == "CLKCORR 1\n"


def test_a_doubled_line_is_collapsed():
    """Old tempo2 builds emit ``NE_SW`` twice from ``-gr transform``."""
    text = "F0 1.0\nNE_SW 4.0\nNE_SW 4.0\n"
    assert dedupe_nonrepeatable(text) == "F0 1.0\nNE_SW 4.0\n"


def test_a_doubled_line_with_different_values_is_refused():
    """Keeping one of them silently would change the model."""
    with pytest.raises(ParTextError, match="twice with different values"):
        dedupe_nonrepeatable("NE_SW 4.0\nNE_SW 8.0\n")


@pytest.mark.parametrize(
    "text",
    [
        "JUMP -fe A 1.0\nJUMP -fe B 2.0\n",
        "EFAC -f A 1.0\nEFAC -f B 2.0\n",
        "FD1JUMP -sys A 1.0\nFD2JUMP -sys B 2.0\n",
    ],
)
def test_repeatable_keys_are_left_alone(text):
    assert dedupe_nonrepeatable(text) == text


def test_comments_survive_the_dedupe():
    text = "# header\nF0 1.0\nC note\nF0 1.0\n"
    assert dedupe_nonrepeatable(text) == "# header\nF0 1.0\nC note\n"
