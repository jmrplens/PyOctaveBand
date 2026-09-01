#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The fence reading-order gate.

``scripts/check_fence_names.py`` holds every documentation page to the
sequential-example convention: a fence may only use names an earlier fence of
the same page defined, with reader-owned placeholders registered explicitly.
These tests pin the failure modes, the placeholder path, the twin-sharing
route key, and that the shipped tree passes.
"""

from __future__ import annotations

import pathlib
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

_SCRIPTS = str(pathlib.Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import check_fence_names


def _page(tmp_path: pathlib.Path, body: str) -> pathlib.Path:
    page = tmp_path / "example.md"
    page.write_text(body, encoding="utf8")
    return page


def test_a_forward_reference_is_flagged(tmp_path: pathlib.Path) -> None:
    """A name defined only by a later fence breaks the reading order."""
    page = _page(
        tmp_path,
        "```python\nprint(levels.mean())\n```\n\n"
        "```python\nlevels = compute()\n```\n\n"
        "```python\ncompute = min\n```\n",
    )
    problems = check_fence_names.check_page(page)
    assert len(problems) == 2
    assert "fence 1 uses levels before the page defines" in problems[0]
    assert "fence 2 uses compute before the page defines" in problems[1]


def test_an_ordered_page_passes(tmp_path: pathlib.Path) -> None:
    """The convention itself: later fences may use earlier names."""
    page = _page(
        tmp_path,
        "```python\nimport numpy as np\nsignal = np.zeros(8)\n```\n\n"
        "```python\nprint(signal.sum(), np.pi)\n```\n",
    )
    assert check_fence_names.check_page(page) == []


def test_an_undefined_name_is_flagged_toward_the_registry(
    tmp_path: pathlib.Path,
) -> None:
    """A name no fence defines points the author at the placeholder table."""
    page = _page(tmp_path, "```python\nprint(spl.max())\n```\n")
    problems = check_fence_names.check_page(page)
    assert len(problems) == 1
    assert "fence 1 uses spl, which no fence of the page defines" in problems[0]
    assert "PLACEHOLDERS" in problems[0]


def test_a_registered_placeholder_is_accepted(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The registry clears exactly the declared names and nothing else."""
    page = _page(tmp_path, "```python\nprint(spl.max(), other)\n```\n")
    monkeypatch.setitem(check_fence_names.PLACEHOLDERS, "example", frozenset({"spl"}))
    problems = check_fence_names.check_page(page)
    assert len(problems) == 1
    assert "uses other," in problems[0]


def test_a_fence_that_does_not_parse_is_flagged(tmp_path: pathlib.Path) -> None:
    """An unparseable fence cannot be checked, and says so."""
    page = _page(tmp_path, "```python\ndef broken(:\n```\n")
    problems = check_fence_names.check_page(page)
    assert len(problems) == 1
    assert "does not parse as Python" in problems[0]


def test_the_route_key_is_shared_by_the_language_twins() -> None:
    """The Spanish twin and the docs mirror resolve to the English route."""
    english = check_fence_names.CONTENT / "signals" / "levels" / "page.mdx"
    spanish = check_fence_names.CONTENT / "es" / "signals" / "levels" / "page.mdx"
    mirror = check_fence_names.DOCS / "signals" / "levels" / "page.md"
    assert check_fence_names._route(english) == "signals/levels/page"
    assert check_fence_names._route(spanish) == "signals/levels/page"
    assert check_fence_names._route(mirror) == "docs/signals/levels/page"


def test_a_stale_registry_line_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A placeholder route with no page behind it must be deleted."""
    monkeypatch.setitem(
        check_fence_names.PLACEHOLDERS, "no/such/page", frozenset({"ghost"})
    )
    assert "no/such/page" in check_fence_names._stale_placeholder_routes()


def test_the_shipped_tree_reads_in_order() -> None:
    """Every committed page passes the gate CI runs."""
    assert check_fence_names.main() == 0
