#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The parity gate between the errata registry and its Spanish edition.

``scripts/generate_site_reports.py`` refuses to render while
``docs/ERRATA.es.md`` drifts from ``docs/ERRATA.md``: same number of ``##``
entries, same order, and each pair of headings naming the same source
document through its first designation token. These tests pin the three
failure modes, the pass, and the one blind spot the token deliberately
accepts.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

_SCRIPTS = str(pathlib.Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import generate_site_reports

_ENGLISH = """## ISO 717-2:2020, Annex C, example C.1 (bare floor)

body

## Commission Directive (EU) 2015/996, Annex II 2.2.1 (road source)

body

## IEC 60268-16:2011, Table M.1 (the beta row)

body

## IEC 60268-16:2011, Table M.1 (step 3 at 250 Hz)

body
"""

_SPANISH = """## ISO 717-2:2020, Anexo C, ejemplo C.1 (suelo desnudo)

cuerpo

## Directiva (UE) 2015/996 de la Comisión, Anexo II 2.2.1 (fuente de carretera)

cuerpo

## IEC 60268-16:2011, Table M.1 (la fila beta)

cuerpo

## IEC 60268-16:2011, Table M.1 (paso 3 a 250 Hz)

cuerpo
"""


def _editions(
    tmp_path: pathlib.Path, english: str, spanish: str
) -> tuple[pathlib.Path, pathlib.Path]:
    en = tmp_path / "ERRATA.md"
    es = tmp_path / "ERRATA.es.md"
    en.write_text(english, encoding="utf8")
    es.write_text(spanish, encoding="utf8")
    return en, es


def test_matching_editions_pass(tmp_path: pathlib.Path) -> None:
    """Same count, same order, same designations: no complaint."""
    en, es = _editions(tmp_path, _ENGLISH, _SPANISH)
    generate_site_reports.check_edition_parity(en, es)


def test_a_missing_entry_fails_with_both_counts(tmp_path: pathlib.Path) -> None:
    """Dropping one Spanish entry names both counts in the refusal."""
    short = _SPANISH.replace(
        "## Directiva (UE) 2015/996 de la Comisión, Anexo II 2.2.1 "
        "(fuente de carretera)\n\ncuerpo\n\n",
        "",
    )
    en, es = _editions(tmp_path, _ENGLISH, short)
    with pytest.raises(SystemExit, match="3 entries against 4"):
        generate_site_reports.check_edition_parity(en, es)


def test_a_reordered_entry_fails_at_its_index(tmp_path: pathlib.Path) -> None:
    """Swapping entries about different documents is caught positionally."""
    swapped = _SPANISH.replace(
        "## ISO 717-2:2020, Anexo C, ejemplo C.1 (suelo desnudo)",
        "## MARK",
    ).replace(
        "## Directiva (UE) 2015/996 de la Comisión, Anexo II 2.2.1 "
        "(fuente de carretera)",
        "## ISO 717-2:2020, Anexo C, ejemplo C.1 (suelo desnudo)",
    )
    swapped = swapped.replace(
        "## MARK",
        "## Directiva (UE) 2015/996 de la Comisión, Anexo II 2.2.1 "
        "(fuente de carretera)",
    )
    en, es = _editions(tmp_path, _ENGLISH, swapped)
    with pytest.raises(SystemExit, match="entry 1 names '2015/996'"):
        generate_site_reports.check_edition_parity(en, es)


def test_a_mistranslated_designation_fails(tmp_path: pathlib.Path) -> None:
    """A digit slipping in a designation is a mistranslation, not a variant."""
    wrong = _SPANISH.replace("ISO 717-2:2020, Anexo", "ISO 717-3:2020, Anexo")
    en, es = _editions(tmp_path, _ENGLISH, wrong)
    with pytest.raises(SystemExit, match="'717-3:2020' where"):
        generate_site_reports.check_edition_parity(en, es)


def test_translated_issuer_words_do_not_fail_the_pair(
    tmp_path: pathlib.Path,
) -> None:
    """The issuer word moves and translates; the token must not require it.

    "Commission Directive (EU) 2015/996" pairs with "Directiva (UE) 2015/996
    de la Comisión": nothing about the issuer survives translation verbatim,
    which is why the token starts at the first digit-bearing word.
    """
    en, es = _editions(tmp_path, _ENGLISH, _SPANISH)
    generate_site_reports.check_edition_parity(en, es)


def test_the_accepted_blind_spot_is_within_one_document(
    tmp_path: pathlib.Path,
) -> None:
    """Swapping two entries of the same document passes, and may.

    The two IEC 60268-16:2011 Table M.1 entries share their designation
    token, so exchanging them is invisible to the gate. That is the accepted
    residual: both headings still name the source they describe, so no
    heading is paired with the wrong document; only the entry order within
    one document's run would be off, which the translation review reads, not
    this gate.
    """
    swapped = _SPANISH.replace(
        "## IEC 60268-16:2011, Table M.1 (la fila beta)", "## MARK"
    ).replace(
        "## IEC 60268-16:2011, Table M.1 (paso 3 a 250 Hz)",
        "## IEC 60268-16:2011, Table M.1 (la fila beta)",
    )
    swapped = swapped.replace(
        "## MARK", "## IEC 60268-16:2011, Table M.1 (paso 3 a 250 Hz)"
    )
    en, es = _editions(tmp_path, _ENGLISH, swapped)
    generate_site_reports.check_edition_parity(en, es)


def test_the_shipped_editions_pass_the_gate() -> None:
    """The committed registry and its Spanish edition satisfy the invariants."""
    root = pathlib.Path(generate_site_reports.__file__).resolve().parent.parent
    generate_site_reports.check_edition_parity(
        root / "docs" / "ERRATA.md", root / "docs" / "ERRATA.es.md"
    )
