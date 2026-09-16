"""The par-text rules: text in, verdict or text out. No PINT anywhere."""

from __future__ import annotations

import pytest

from psrdata import ParTextError
from psrdata.partext import (
    NOISE_NAMES,
    REPEATABLE_KEYS,
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
        "CHI2 1234.5",
        # tempo2 EFAC family (TNEFAC/TNEQUAD)
        "TNEFAC -f L 1.0",
        "TNEQUAD -f L 0.1",
        # TNEF/TNEQ/TRES/DMRES family
        "TRES 1.04",
        "DMRES 0.5",
        "TNEF -f L 1.0",
        "TNEQ -f L -6.0",
        "TNSWAMP -12.0",
        "PLREDAMP -13.0",
        "CHI2R 1.2",
        "TNREDFLOG 2",
        "TNDMC 30",
    ],
)
def test_the_explicit_list_catches_both_vocabularies(line):
    """Exact first-token match against :data:`NOISE_NAMES`, no prefix net."""
    assert is_noise_line(line)


@pytest.mark.parametrize(
    "line",
    [
        "F0 61.485476554371304592 1 1.7e-11",
        "JUMP -fe Rcvr_800 0.1 1",
        "DMJUMP -fe Rcvr 1e-3",
        "DM 160.0 1",
        "ELAT 5.0 1",
        "FD1 0.001 1",
        "TZRMJD 55000.0",
        "TRACK -2",
        "TIMEEPH FB90",
        "T2CMETHOD IAU2000B",
        "RAJ 18:53:57.3",
        "RA 18:53:57.3",
        # not on the list: a prefix catch-all would have taken these
        "PLREDSIN_0001 1e-7",
        "TNUNKNOWN 1.0",
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
        "CLK  TT(BIPM2023)\nF0 1.0\n"
    )
    assert respell_clock_for_tempo2("CLKCORR 1\n") == "CLKCORR 1\n"


def test_clock_respelling_changes_only_the_first_token():
    text = "  CLOCK\tTT(BIPM2023)   # keep spacing\n"
    assert respell_clock_for_tempo2(text) == ("  CLK\tTT(BIPM2023)   # keep spacing\n")


def test_a_doubled_line_is_collapsed():
    """Old tempo2 builds emit ``NE_SW`` twice from ``-gr transform``."""
    text = "F0 1.0\nNE_SW 4.0\nNE_SW 4.0\n"
    assert dedupe_nonrepeatable(text) == "F0 1.0\nNE_SW 4.0\n"


def test_a_doubled_line_with_different_values_is_refused():
    """Keeping one of them silently would change the model."""
    with pytest.raises(ParTextError, match="twice with different values"):
        dedupe_nonrepeatable("NE_SW 4.0\nNE_SW 8.0\n")


@pytest.mark.parametrize(
    "first,second",
    [
        ("339.315687288152030000", "339.315687288152030001"),
        ("1e-400", "2e-400"),
    ],
)
def test_distinct_high_precision_values_are_not_deduplicated(first, second):
    with pytest.raises(ParTextError, match="twice with different values"):
        dedupe_nonrepeatable(f"F0 {first}\nF0 {second}\n")


def test_equivalent_decimal_spellings_are_deduplicated():
    assert dedupe_nonrepeatable("NE_SW 4.0\nNE_SW 4.00\n") == "NE_SW 4.0\n"


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


def test_the_noise_name_membership_is_the_contract():
    """R-7.2.1: the exact set, enumerated. Adding a name is a contract change."""
    assert NOISE_NAMES == frozenset(
        {
            "EFAC", "T2EFAC", "TNEF", "TNEFAC",
            "EQUAD", "T2EQUAD", "TNEQ", "TNEQUAD",
            "TNGLOBALEF", "TNGLOBALEQ",
            "ECORR", "TNECORR",
            "DMEFAC", "DMEQUAD",
            "RNAMP", "RNIDX",
            "TNREDAMP", "TNREDGAM", "TNREDC", "TNREDF", "TNREDFC",
            "TNREDFLOG", "TNREDFLOG_FACTOR", "TNREDTSPAN",
            "TNDMAMP", "TNDMGAM", "TNDMC", "TNDMFLOG", "TNDMFLOG_FACTOR",
            "TNDMTSPAN",
            "TNCHROMAMP", "TNCHROMGAM", "TNCHROMC", "TNCHROMIDX",
            "TNCHROMFLOG", "TNCHROMFLOG_FACTOR", "TNCHROMTSPAN",
            "TNSWAMP", "TNSWGAM", "TNSWC", "TNSWFLOG", "TNSWFLOG_FACTOR",
            "TNGAMMA", "TNAMP",
            "PLREDFREQ", "PLREDAMP",
            "TRES", "DMRES", "CHI2", "CHI2R",
        }
    )  # fmt: skip
    assert "DMJUMP" not in NOISE_NAMES  # R-7.2.2


def test_the_repeatable_keys_are_enumerated():
    assert REPEATABLE_KEYS == frozenset(
        {
            "JUMP", "DMJUMP", "EFAC", "EQUAD", "ECORR", "T2EFAC", "T2EQUAD",
            "TNEF", "TNEQ", "TNECORR", "DMEFAC", "DMEQUAD",
        }
    )  # fmt: skip


def test_dmjump_is_a_timing_line_not_noise():
    """R-7.2.2."""
    assert not is_noise_line("DMJUMP -fe Rcvr 1e-3")
    assert strip_noise_lines("DMJUMP -fe Rcvr 1e-3\n") == "DMJUMP -fe Rcvr 1e-3\n"


def test_a_tab_after_the_tempo2_comment_marker_is_still_a_comment():
    """R-7.1.1 speaks of the first *token*; whitespace kind does not matter."""
    assert not is_active_line("C\tF0 1.0")
    assert not is_noise_line("C\tEFAC -f x 1.0")


def test_line_key_is_the_upper_cased_first_token():
    from psrdata.partext import line_key

    assert line_key("f0 1.0") == "F0"
    assert line_key("  RAJ\t18:53") == "RAJ"


def test_respelling_changes_the_first_token_only():
    """R-7.4.1: a tab-separated tempo2 line keeps its remainder verbatim."""
    assert respell_fdjump_for_pint("FDJUMP2\t-sys A\t1.0\n") == "FD2JUMP\t-sys A\t1.0\n"
    assert respell_fdjump_for_pint("FDJUMPX 1.0\n") == "FDJUMPX 1.0\n"
