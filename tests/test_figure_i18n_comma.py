#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The save-time decimal-comma pass of the Spanish figure edition.

``scripts/figures/i18n.py`` rewrites ``digit.digit`` into ``digit,digit`` on
every non-mathtext string of a Spanish figure. A clause, equation, table or
annex NUMBER is not a decimal: the Spanish pages write "apartado 7.4" with the
dot, so the figures must too. These tests pin the reference guard (which
strings keep their dots) and the decimal behaviour it must NOT disturb.
"""

import pathlib
import sys

import pytest

_SCRIPTS = str(pathlib.Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from figures.i18n import _ES_PATTERNS, _decimal_comma, lookup


@pytest.mark.parametrize(
    "text",
    [
        # Two-part reference numbers after a singular token keep the dot
        # (the committed defect: these shipped as "7,4", "8,2", ...).
        "apartado 7.4",
        "(véase el apartado 5.4)",
        "apartado 8.2:  r >= 100 kPa.s",
        "capítulo 4.1, f ≥ 3 kHz",
        "cláusula 4.2",
        "Anexo 5.1",
        "Tabla 13.8",
        "Tabla 8.14",
        "tabla 5.1",  # lowercase, as shipped in infinite_mobilities
        "tabla 3.1",
        "Fórmula 3.1",
        "Formula 3.1",
        "Fórm. 3.1",
        "Form. 3.1",
        "Ec. 8.252",
        "Ec. 14.31",
        "Ec. 13.1  (peor Δ 13 dB)",
        "§ 7.4",
        "§7.4",
        # Compound references: every number of the list is a reference.
        "Bies Ecs. 8.44, 8.46",
        "Ecs. 13.27-13.33",
        "Ecs. 8.44, 8.46 y 8.50",
        "apartados 6.2 y 6.3",
        "capítulos 5.1.2 y 5.1.4",
        "cláusulas 4.2 y 4.3",
        # Dash-joined ranges read as reference-to-reference even after a
        # singular token.
        "apartado 4.1-4.3",
        # Pre-existing exemptions, locked in: three-part numbers and
        # letter-adjacent standard designations never took the comma.
        "apartado 7.4.3",
        "S3.5",
        "ANSI S1.11",
    ],
)
def test_reference_numbers_keep_their_dots(text: str) -> None:
    assert _decimal_comma(text) == text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Plain decimals still take the comma.
        ("3.5 dB", "3,5 dB"),
        ("de 0.5 a 1.2 kHz", "de 0,5 a 1,2 kHz"),
        ("0.08 c0", "0,08 c0"),
        ("8.0 sone_HMS", "8,0 sone_HMS"),
        # A decimal after the reference, in another syntactic position,
        # is still a decimal.
        ("Ec. 10.2: máximo 3.7 vacil", "Ec. 10.2: máximo 3,7 vacil"),
        ("apartado 8.2, 0.5 dB de margen", "apartado 8.2, 0,5 dB de margen"),
        ("Tabla 2 (±0.25 dB)", "Tabla 2 (±0,25 dB)"),
        ("Tabla 2 -0.5 dB", "Tabla 2 -0,5 dB"),
        # A letter-numbered table or annex does not shield what follows.
        ("Tabla A.6, α = 0.05", "Tabla A.6, α = 0,05"),
        ("Anexo E: 118.4 y 137.3 Hz", "Anexo E: 118,4 y 137,3 Hz"),
        # An integer reference is untouched either way; the decimal beside
        # it converts ("y"-lists only bind after a PLURAL token).
        ("Fórmula 7, eps = 0.9", "Fórmula 7, eps = 0,9"),
        ("Tabla 5 y 0.5 dB de margen", "Tabla 5 y 0,5 dB de margen"),
        # Token lookalikes are not tokens.
        ("Forma cerrada (FOM = 82.7 dB)", "Forma cerrada (FOM = 82,7 dB)"),
        ("la fórmula da 0.5", "la fórmula da 0,5"),
        ("la espec. 3.5 pide margen", "la espec. 3,5 pide margen"),
        # Already-Spanish strings pass through unchanged.
        ("ya son 3,5 dB", "ya son 3,5 dB"),
        # Bare numbers (the tick-label path).
        ("12.5", "12,5"),
        ("1000", "1000"),
    ],
)
def test_legitimate_decimals_still_take_the_comma(text: str, expected: str) -> None:
    assert _decimal_comma(text) == expected


def test_every_replacement_template_of_the_pattern_table_compiles() -> None:
    r"""One bad escape in one template silences the whole Spanish edition.

    ``lookup`` walks the pattern list until something matches, and
    ``re.sub`` compiles the *replacement* as it goes, so a template with a
    stray ``\l`` (a mathtext ``\lambda`` written with one backslash instead
    of two) raises for every string that reaches that row -- which is every
    string the exact table and the earlier patterns do not answer. The
    Spanish pass of a whole render run then dies on a label that has nothing
    to do with the offending row, which is exactly how it was found.
    """
    import re

    broken = []
    for pattern, replacement in _ES_PATTERNS:
        try:
            re.sub(pattern, replacement, "")
        except re.PatternError as exc:  # pragma: no cover - the failure text
            broken.append(f"{pattern!r}: {exc}")
    assert not broken, "\n".join(broken)


def test_a_string_the_tables_do_not_know_comes_back_unchanged() -> None:
    """The same guard, from the caller's side: no row may raise."""
    assert lookup("zzz not a label anybody wrote 123") == (
        "zzz not a label anybody wrote 123"
    )
