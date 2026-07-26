#  Copyright (c) 2026. Jose M. Requena-Plens
"""Tests for the generated API reference (taxonomy + generator).

Covers the contract the site relies on: every public module is mapped to
exactly one section, every ``phonometry.__all__`` name lands on exactly one
generated page, docstrings parse, roles rewrite to intra-site links, pages
carry valid frontmatter and two runs are byte-identical (the CI drift gate
depends on determinism).
"""

from __future__ import annotations

import inspect
import pathlib
import sys

import pytest

import phonometry

_SCRIPTS = str(pathlib.Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import api_taxonomy
import check_api_reference as car
import generate_api_docs as gad

# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------


def test_every_public_name_maps_to_a_taxonomy_section() -> None:
    """Each __all__ name resolves to a module that has a section."""
    for name in phonometry.__all__:
        module = gad.attribute_module(name, getattr(phonometry, name))
        section = api_taxonomy.module_section(module)
        assert section.key in api_taxonomy.SECTIONS


def test_no_duplicate_module_assignment() -> None:
    seen: set[str] = set()
    for section in api_taxonomy.SECTIONS.values():
        for module in section.modules:
            assert module not in seen, module
            seen.add(module)


def test_unmapped_module_raises_helpful_keyerror() -> None:
    with pytest.raises(KeyError, match="not mapped"):
        api_taxonomy.module_section("phonometry.does_not_exist")


def test_section_labels_are_bilingual() -> None:
    for section in api_taxonomy.SECTIONS.values():
        assert section.label_en
        assert section.label_es


# ---------------------------------------------------------------------------
# Docstring parsing
# ---------------------------------------------------------------------------


def test_parse_docstring_on_real_docstring() -> None:
    doc = inspect.getdoc(phonometry.leq)
    assert doc is not None
    parsed = gad.parse_docstring(doc)
    assert not parsed.issues
    assert [name for name, _ in parsed.params] == [
        "x",
        "calibration_factor",
        "dbfs",
    ]
    assert "Pascals" in dict(parsed.params)["calibration_factor"]
    assert parsed.returns.startswith("Scalar for 1D input")
    assert "Equivalent continuous sound level" in parsed.description


def test_parse_docstring_ivar_and_raises() -> None:
    doc = inspect.getdoc(phonometry.UncertaintyResult)
    assert doc is not None
    parsed = gad.parse_docstring(doc)
    assert not parsed.issues
    ivar_names = [name for name, _ in parsed.ivars]
    assert "value" in ivar_names
    assert "combined_uncertainty" in ivar_names


def test_parse_docstring_flags_unsupported_field() -> None:
    parsed = gad.parse_docstring("Summary.\n\n:cvar x: not supported here\n")
    assert parsed.issues
    assert "cvar" in parsed.issues[0]


def test_parse_docstring_multiline_field_joins() -> None:
    parsed = gad.parse_docstring(
        "Summary.\n\n:param a: first line\n    second line\n:return: ok\n"
    )
    assert parsed.params == [("a", "first line second line")]
    assert parsed.returns == "ok"


# ---------------------------------------------------------------------------
# reST -> Markdown
# ---------------------------------------------------------------------------


def test_rest_roles_to_links_known_and_unknown() -> None:
    xref = {"leq": "/phonometry/reference/api/levels/levels/#leq"}
    text = "Use :func:`leq` but not :class:`numpy.ndarray`."
    out = gad.rest_roles_to_links(text, xref)
    assert "[`leq`](/phonometry/reference/api/levels/levels/#leq)" in out
    assert "`numpy.ndarray`" in out
    assert ":func:" not in out
    assert ":class:" not in out


def test_rest_roles_to_links_tilde_shortens_display() -> None:
    xref = {"leq": "/x/#leq"}
    out = gad.rest_roles_to_links(":func:`~phonometry.leq`", xref)
    assert out == "[`leq`](/x/#leq)"


def test_rest_blocks_note_and_literal_block() -> None:
    text = (
        "Intro paragraph.\n"
        "\n"
        ".. note:: Something important\n"
        "   continued here.\n"
        "\n"
        "A formula::\n"
        "\n"
        "    y = 2 * x\n"
    )
    code: list[str] = []
    out = gad.rest_blocks_to_markdown(text, code)
    out = gad._restore_code(out, code)
    assert ":::note" in out
    assert "Something important" in out
    assert "```text\ny = 2 * x\n```" in out
    assert "A formula:" in out


# ---------------------------------------------------------------------------
# Full generation
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    root = tmp_path_factory.mktemp("apidocs")
    gad.generate(root / "api", root / "api-sidebar.mjs")
    return root


def test_generation_is_deterministic(
    generated: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    gad.generate(tmp_path / "api", tmp_path / "api-sidebar.mjs")
    first = {
        p.relative_to(generated): p.read_text(encoding="utf-8")
        for p in sorted(generated.rglob("*"))
        if p.is_file()
    }
    second = {
        p.relative_to(tmp_path): p.read_text(encoding="utf-8")
        for p in sorted(tmp_path.rglob("*"))
        if p.is_file()
    }
    assert second.keys() == first.keys(), "generated file sets differ"
    for relpath, first_content in first.items():
        assert second[relpath] == first_content, f"{relpath} is not deterministic"


def test_every_public_name_on_exactly_one_page(
    generated: pathlib.Path,
) -> None:
    pages, _, _ = gad.build_model()
    counts: dict[str, int] = {}
    for page in pages:
        for member in page.members:
            counts[member.name] = counts.get(member.name, 0) + 1
    assert set(counts) == set(phonometry.__all__)
    duplicated = [name for name, count in counts.items() if count > 1]
    assert not duplicated


def test_no_docstring_parse_failures() -> None:
    """The whole corpus renders without degrading to verbatim blocks."""
    pages, xref, _ = gad.build_model()
    stats = gad.RoleStats()
    failures: list[str] = []
    for page in pages:
        gad.render_module_page(page, xref, stats, failures)
    assert failures == []


def test_generated_pages_have_valid_frontmatter(
    generated: pathlib.Path,
) -> None:
    md_files = sorted((generated / "api").rglob("*.md"))
    assert len(md_files) > 80
    for path in md_files:
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "---", path
        end = lines[1:].index("---") + 1
        block = lines[1:end]
        keys = {line.split(":", 1)[0] for line in block if not line.startswith(" ")}
        assert "title" in keys, path
        assert "description" in keys, path
        # Every page announces it is generated.
        banner = "\n".join(lines[end:])
        assert "Auto-generated" in banner, path


def test_xref_anchors_point_to_emitted_headings(
    generated: pathlib.Path,
) -> None:
    """Anchor slugs in the xref map match github-slugger applied to headings."""
    pages, xref, _ = gad.build_model()
    for page in pages:
        text = (generated / "api" / page.relpath).read_text(encoding="utf-8")
        for member in page.members:
            assert xref[member.name] == f"{page.url}#{member.anchor}"
            heading = f"## {member.name}"
            plain = f"## {member.name.replace('_', chr(92) + '_')}"
            assert heading in text or plain in text, (page.module, member.name)


def test_sidebar_fragment_lists_every_page(generated: pathlib.Path) -> None:
    sidebar = (generated / "api-sidebar.mjs").read_text(encoding="utf-8")
    pages, _, _ = gad.build_model()
    assert (
        "{ slug: 'reference/api', label: 'Overview', translations: { es: 'Resumen' } }"
        in sidebar
    )
    assert sidebar.count("collapsed: true") == len(gad.SECTIONS) + 1
    for page in pages:
        assert f"'reference/api/{page.section.key}/{page.slug}'" in sidebar


# ---------------------------------------------------------------------------
# Curated quick-table coverage gate (scripts/check_api_reference.py)
# ---------------------------------------------------------------------------

_SAMPLE_TABLE = """\
# API Reference

Some prose with `inline_code` that must not count.

| Name | Type | Description | Usage |
| :--- | :--- | :--- | :--- |
| `leq` | `function` | **Equivalent level.**<br>Uses `laeq` internally | `leq(x)` |
| `reverberation_index` / `estimate_reverberation_index` | `function` | ISO 10052 | `k = reverberation_index(t)` |
| `OctaveFilterBank.spectrogram` | `method` | Band levels over time | `bank.spectrogram(x)` |
| `.plot()` | `method` | Canonical figure | `res.plot()` |
  | `indented_name` | `function` | Indented rows still count | `indented_name()` |
| `piped_name` \\| alias | `function` | Escaped pipe stays in the cell | `piped_name()` |
"""


def test_table_names_extracts_first_column_only() -> None:
    """Every backticked name in a first cell counts; other cells do not."""
    names = car.table_names(_SAMPLE_TABLE)
    assert names == {
        "leq",
        "reverberation_index",
        "estimate_reverberation_index",
        "OctaveFilterBank.spectrogram",
        ".plot()",
        "indented_name",
        "piped_name",
    }
    # Backticks outside tables and in later columns are ignored.
    assert "inline_code" not in names
    assert "laeq" not in names


def test_missing_names_orders_by_all_and_allows_extras() -> None:
    missing = car.missing_names(_SAMPLE_TABLE, ["laeq", "leq", "sel"])
    assert missing == ["laeq", "sel"]


def test_quick_table_covers_public_api() -> None:
    """The committed docs/api-reference.md never misses an __all__ name."""
    path = pathlib.Path(__file__).resolve().parent.parent / "docs" / "api-reference.md"
    markdown = path.read_text(encoding="utf-8")
    assert car.missing_names(markdown, list(phonometry.__all__)) == []
