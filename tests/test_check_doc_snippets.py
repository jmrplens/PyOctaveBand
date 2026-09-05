#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The documentation-snippet gate must fail on the defects it exists for.

A gate that only ever passes proves nothing, so each check is fed the defect
it was written for and asserted to report it, and then fed the correct page
and asserted to stay quiet. The shadowing case is the one that motivated the
script: ``from scipy import signal`` next to ``from phonometry import
signals`` is fine, but rebinding an imported name is not, and Python says
nothing either way.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

_SCRIPT = (
    pathlib.Path(__file__).resolve().parent.parent / "scripts" / "check_doc_snippets.py"
)
_spec = importlib.util.spec_from_file_location("check_doc_snippets", _SCRIPT)
assert _spec is not None
assert _spec.loader is not None
check_doc_snippets = importlib.util.module_from_spec(_spec)
sys.modules["check_doc_snippets"] = check_doc_snippets
_spec.loader.exec_module(check_doc_snippets)


def _page(tmp_path: pathlib.Path, *blocks: str, name: str = "page.md") -> pathlib.Path:
    body = "\n".join(f"```python\n{b.strip()}\n```\n" for b in blocks)
    path = tmp_path / name
    path.write_text("Prose.\n\n" + body, encoding="utf-8")
    return path


def test_rebinding_an_imported_name_is_reported(tmp_path: pathlib.Path) -> None:
    page = _page(
        tmp_path,
        """
from phonometry import signals

signals = [1.0, 2.0]
""",
    )
    (failure,) = check_doc_snippets.check_shadowing([page])
    assert "'signals = ...' rebinds" in failure


def test_second_import_of_the_same_name_is_reported(tmp_path: pathlib.Path) -> None:
    page = _page(
        tmp_path,
        """
from phonometry import signals
from scipy import signals
""",
    )
    (failure,) = check_doc_snippets.check_shadowing([page])
    assert "rebinds the name imported from phonometry" in failure


def test_the_rebinding_is_caught_across_blocks(tmp_path: pathlib.Path) -> None:
    """A page is read top to bottom; the import of block 0 is still in scope."""
    page = _page(
        tmp_path,
        "from phonometry import signals\n\nx = signals.leq([1.0], 48000)",
        "signals = [1.0, 2.0]\nprint(signals)",
    )
    (failure,) = check_doc_snippets.check_shadowing([page])
    assert "block 1" in failure


def test_a_generated_dump_does_not_carry_imports_across_pages(
    tmp_path: pathlib.Path,
) -> None:
    """A dump glues unrelated pages: their blocks share no scope."""
    dump = _page(
        tmp_path,
        "from phonometry import envelope\n\nenvelope(x, fs)",
        "envelope = 2.0\nprint(envelope)",
        name="dump.txt",
    )
    assert check_doc_snippets.check_shadowing([dump]) != []
    assert check_doc_snippets.check_shadowing([dump], carry=False) == []


def test_a_different_name_next_to_scipy_is_fine(tmp_path: pathlib.Path) -> None:
    """The permitted arrangement: distinct names, so neither is shadowed."""
    page = _page(
        tmp_path,
        """
from scipy import signal
from phonometry import signals

b, a = signal.butter(2, 0.2)
level = signals.leq([1.0], 48000)
""",
    )
    assert check_doc_snippets.check_shadowing([page]) == []


def test_a_translation_that_drops_an_import_is_reported(tmp_path: pathlib.Path) -> None:
    en = _page(tmp_path, "from phonometry import leq, sel", name="en.md")
    es = _page(tmp_path, "from phonometry import leq", name="es.md")
    (failure,) = check_doc_snippets.check_translations([(en, es)])
    assert "missing phonometry.sel" in failure


def test_a_translation_that_only_changes_its_strings_is_fine(
    tmp_path: pathlib.Path,
) -> None:
    en = _page(tmp_path, 'from phonometry import leq\nprint("Level")', name="en.md")
    es = _page(
        tmp_path,
        'from phonometry import leq\nprint("Nivel")  # traducido',
        name="es.md",
    )
    assert check_doc_snippets.check_translations([(en, es)]) == []


def test_a_page_that_does_not_run_is_reported(tmp_path: pathlib.Path) -> None:
    page = _page(tmp_path, "raise SystemExit('boom')")
    (failure,) = check_doc_snippets.check_execution([page])
    assert "boom" in failure


def test_a_sketch_of_a_call_is_read_but_not_run(tmp_path: pathlib.Path) -> None:
    """``f(...)`` shows the shape of a call; running it proves nothing."""
    page = _page(tmp_path, "from phonometry import leq\n\nlevel = leq(...)")
    assert check_doc_snippets.check_execution([page]) == []
    # A keyword placeholder is the same sketch written differently.
    kw = _page(
        tmp_path,
        "from phonometry import leq\n\nlevel = leq(x=...)",
        name="kw.md",
    )
    assert check_doc_snippets.check_execution([kw]) == []
    # It is still read: a rebinding in the same block is still a failure.
    bad = _page(
        tmp_path,
        "from phonometry import leq\n\nlevel = leq(...)\nleq = 3.0",
        name="bad.md",
    )
    assert check_doc_snippets.check_shadowing([bad]) != []


def test_a_stale_skip_entry_is_reported(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A page that starts running must leave the skip list."""
    page = _page(tmp_path, "print('runs fine')", name="fine.md")
    monkeypatch.setitem(check_doc_snippets._SKIP, "fine", "reason")
    (failure,) = check_doc_snippets.check_execution([page])
    assert "runs now" in failure


def test_a_skip_entry_stays_while_one_edition_still_fails(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An entry names a guide, and a guide is written twice.

    The site page and the hand-written mirror under ``docs/`` share a stem, so
    one entry covers both. Reporting it stale the moment either one runs would
    turn the report into a failing page on the next run.
    """
    site = _page(tmp_path, "print('runs fine')", name="fine.md")
    mirror_dir = tmp_path / "mirror"
    mirror_dir.mkdir()
    mirror = _page(mirror_dir, "raise SystemExit('still broken')", name="fine.md")
    monkeypatch.setitem(check_doc_snippets._SKIP, "fine", "reason")
    assert check_doc_snippets.check_execution([site, mirror]) == []
    # Once both editions run, the entry has to go, and it is reported once.
    mirror.write_text("```python\nprint('now fine too')\n```\n", encoding="utf-8")
    failures = check_doc_snippets.check_execution([site, mirror])
    assert len(failures) == 1
    assert "runs now" in failures[0]
