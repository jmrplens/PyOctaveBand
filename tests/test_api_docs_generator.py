#  Copyright (c) 2026. Jose Manuel Requena Plens
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


def test_math_directive_becomes_display_math() -> None:
    text = (
        "The level follows:\n"
        "\n"
        ".. math::\n"
        "\n"
        "   L_p = 10 \\lg\\!\\left( \\frac{S}{S_0} \\right)\n"
        "   \\tag{Eq. 12}\n"
        "\n"
        "   K_2 = 10 \\lg(1 + 4S/A) \\tag{Eq. A.2}\n"
        "\n"
        "Trailing prose.\n"
    )
    out = gad.render_prose(text, {}, gad.RoleStats())
    # Blank lines split the directive into one $$ block per equation, and
    # each equation's physical lines are joined into a single line so
    # remark-math keeps the whole block in one paragraph.
    assert (
        "$$\nL_p = 10 \\lg\\!\\left( \\frac{S}{S_0} \\right)"
        " \\tag{Eq. 12}\n$$" in out
    )
    assert "$$\nK_2 = 10 \\lg(1 + 4S/A) \\tag{Eq. A.2}\n$$" in out
    assert ".. math::" not in out


def test_math_role_becomes_inline_math_untouched_by_escaping() -> None:
    text = (
        "Area :math:`S = 2\\pi r^2` and :math:`a = 0.5\\,l_1+d` with "
        "``S`` in m^2."
    )
    out = gad.render_prose(text, {}, gad.RoleStats())
    # The TeX passes through verbatim: no intraword-asterisk or ``<``
    # escaping may reach inside the ``$`` span.
    assert "$S = 2\\pi r^2$" in out
    assert "$a = 0.5\\,l_1+d$" in out
    assert ":math:" not in out
    # Plain prose around it still gets the normal treatment.
    assert "`S` in m^2" in out


def test_math_role_wrapped_across_lines_joins() -> None:
    out = gad.render_inline(
        ":math:`L_W = L_p +\n    10 \\lg(S/S_0)`", {}, gad.RoleStats()
    )
    assert out == "$L_W = L_p + 10 \\lg(S/S_0)$"


def test_math_in_table_cell_keeps_its_bars_single() -> None:
    # GFM decodes the ``\|`` cell escape for prose but not inside math, so
    # escaping a formula's bar would silently typeset ``\|`` (a double bar)
    # where the standard means a modulus. The bars become TeX commands.
    out = gad.render_cell(
        "Magnitude :math:`20 \\lg |H|`, in dB.", {}, gad.RoleStats()
    )
    assert "$20 \\lg \\vert H\\vert $" in out
    assert "\\|" not in out
    # A norm that was already written ``\|`` keeps both bars.
    norm = gad.render_cell(":math:`\\|v\\|`", {}, gad.RoleStats())
    assert norm == "$\\Vert v\\Vert $"
    assert "\\|" not in norm


def test_render_prose_leaves_math_bars_alone() -> None:
    # Outside a table there is nothing to escape, so the formula is verbatim.
    out = gad.render_prose(
        "Magnitude :math:`20 \\lg |H|`.", {}, gad.RoleStats()
    )
    assert "$20 \\lg |H|$" in out


def test_math_directive_nested_in_a_list_keeps_the_item_indent() -> None:
    # A ``$$`` block flush left would close the list, ejecting both the
    # equation and the prose after it from the bullet.
    text = (
        "* **Sensitivity**. The level referred to 1 W is\n"
        "\n"
        "  .. math::\n"
        "\n"
        "     L_M = L + 20 \\lg(d / d_0)\n"
        "\n"
        "  where :math:`L` is the band mean.\n"
        "\n"
        "* **Second characteristic** follows.\n"
    )
    out = gad.render_prose(text, {}, gad.RoleStats())
    assert "  $$\n  L_M = L + 20 \\lg(d / d_0)\n  $$" in out
    # A directive that was not nested still starts at column zero.
    flat = gad.render_prose(".. math::\n\n   a = b\n", {}, gad.RoleStats())
    assert flat == "$$\na = b\n$$"


def test_math_directive_keeps_one_environment_in_one_block() -> None:
    # A multi-row environment must stay in a single ``$$`` block: splitting
    # it would leave an orphan ``\begin{aligned}`` that KaTeX cannot close.
    text = (
        ".. math::\n"
        "\n"
        "   \\begin{aligned}\n"
        "   a &= b \\\\\n"
        "   c &= d\n"
        "   \\end{aligned}\n"
    )
    out = gad.render_prose(text, {}, gad.RoleStats())
    assert out.count("$$") == 2
    assert out.count("\\begin{aligned}") == out.count("\\end{aligned}") == 1


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
    """One group per section, every page in it, and nothing else.

    The topics of the site mount these groups by section, so a page missing
    here is a page missing from the sidebar of its own domain.
    """
    sidebar = (generated / "api-sidebar.mjs").read_text(encoding="utf-8")
    pages, _, _ = gad.build_model()
    assert "export const apiSections = {" in sidebar
    assert sidebar.count("collapsed: true") == len(gad.SECTIONS)
    for key in gad.SECTIONS:
        assert f"  '{key}': {{" in sidebar, key
    for page in pages:
        assert f"'reference/api/{page.section.key}/{page.slug}'" in sidebar
    listed = sidebar.count("      'reference/api/")
    assert listed == len(pages), (listed, len(pages))


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
    """The committed docs/reference/api/index.md never misses an __all__ name."""
    path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "docs" / "reference" / "api" / "index.md"
    )
    markdown = path.read_text(encoding="utf-8")
    assert car.missing_names(markdown, list(phonometry.__all__)) == []


_VERSION_ROW = """\
| Name | Type | Description | Usage |
| :--- | :--- | :--- | :--- |
| `__version__` | `str` | **Package version string.** | `phonometry.__version__  # '3.3.0'` |
"""


def test_version_problems_flags_a_literal_that_drifted() -> None:
    """A release bump leaves the literal behind; the gate must say so."""
    assert car.version_problems(_VERSION_ROW, "3.3.0") == []
    assert car.version_problems(_VERSION_ROW, "3.4.0") == ["shows '3.3.0'"]
    assert car.version_problems(_VERSION_ROW, "4.0.0") == ["shows '3.3.0'"]


def test_version_problems_flags_a_row_that_shows_no_version_at_all() -> None:
    """Finding nothing must fail, not pass while claiming the version is fine.

    Deleting the row breaks the coverage check, but emptying its example
    cell leaves the row in place, and a check that only compares literals it
    finds would report the current version for a page that shows none.
    """
    emptied = _VERSION_ROW.replace("  # '3.3.0'", "")
    assert "__version__" in emptied
    assert car.version_problems(emptied, "3.3.0") == [
        "no `phonometry.__version__  # '...'` example to check"
    ]


def test_version_problems_ignores_a_lookalike_comment() -> None:
    """Only ``phonometry.__version__  # '...'`` counts, not any old string."""
    unrelated = "| `leq` | `function` | Level | `leq(x)  # '3.2.0' looking text` |"
    assert car.version_problems(_VERSION_ROW + unrelated, "3.3.0") == []


def test_quick_table_version_literal_is_current() -> None:
    """The committed table shows the version the package actually reports."""
    path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "docs" / "reference" / "api" / "index.md"
    )
    markdown = path.read_text(encoding="utf-8")
    assert car.version_problems(markdown, phonometry.__version__) == []
    # And it says it exactly once, so no second copy can drift unseen.
    assert car._VERSION_LITERAL.findall(markdown) == [phonometry.__version__]


# ---------------------------------------------------------------------------
# Overloaded call forms
# ---------------------------------------------------------------------------


def test_overloaded_function_publishes_every_call_form() -> None:
    """A function with overloads has more than one way to be called.

    The implementation signature is not one of them: it is the permissive
    union that accepts every head plus the combinations the heads refuse, so
    publishing only that would document a contract the library rejects.
    """
    from phonometry.building.measurement.insulation import airborne_insulation

    rendered = gad.format_signature("airborne_insulation", airborne_insulation)
    forms = rendered.split("\n\n")
    assert len(forms) == 2, rendered
    # One form takes the pair, the other takes neither; no form takes one.
    with_pair = [f for f in forms if "area" in f]
    without = [f for f in forms if "area" not in f]
    assert len(with_pair) == 1 and len(without) == 1
    assert "volume" in with_pair[0]
    assert "volume" not in without[0]
    # The permissive union is exactly what must NOT be published.
    assert "area: float | None = None" not in rendered


def test_overload_placeholder_default_reads_as_ellipsis() -> None:
    """``= ...`` is the head's placeholder; ``repr`` would print ``Ellipsis``."""
    from phonometry.emission.sound_power_anechoic import sound_power_anechoic

    rendered = gad.format_signature("sound_power_anechoic", sound_power_anechoic)
    assert "= ..." in rendered
    assert "Ellipsis" not in rendered


def test_plain_function_still_renders_one_form() -> None:
    """Nothing changes for the functions that declare no overloads."""
    from phonometry.signals.levels import leq

    rendered = gad.format_signature("leq", leq)
    assert "\n\n" not in rendered
