#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The subscript-slope gate, held against the shapes this corpus writes.

``scripts/check_subscript_slope.py`` reads mathematics as it is typed rather
than as it renders, and the two ways of writing an upright subscript are the
whole difficulty: the command may wrap the whole comma-separated run
(``L_\\mathrm{n,w,eq}``) or one component of it (``\\alpha_{\\mathrm{s},i}``).
A first version read the wrapped run component by component, decided that its
``w`` was italic, and reported four pages that were correctly set. These tests
fix both readings, the silence between files that the file scope depends on,
and the exclusions the gate claims.
"""

from __future__ import annotations

import pathlib
import sys

_SCRIPTS = str(pathlib.Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import check_subscript_slope as css


def _write(tmp_path: pathlib.Path, name: str, text: str) -> pathlib.Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_one_symbol_two_slopes_in_one_file_fails(tmp_path):
    """The defect the gate exists for, with both lines named."""
    path = _write(tmp_path, "guide.md",
                  "Positions average as $\\sum_i 10^{L_i/10}$.\n"
                  "\n"
                  "The impact level $L_\\mathrm{i}$ is normalised.\n")
    read, failures = css.check([path])
    assert read == 1
    assert len(failures) == 1
    assert "L_i" in failures[0]
    assert "italic on line 1" in failures[0]
    assert "upright on line 3" in failures[0]


def test_two_files_may_disagree(tmp_path):
    """The silence the file scope depends on: D_z is both, one per module."""
    outdoor = _write(tmp_path, "outdoor.md", "Screening $D_z$ from the path.\n")
    shock = _write(tmp_path, "shock.md", "Dose $D_\\mathrm{z}$ up the spine.\n")
    _, failures = css.check([outdoor, shock])
    assert failures == []


def test_component_of_a_compound_subscript_is_read(tmp_path):
    """Half of one formula upright and half italic is the same defect."""
    path = _write(tmp_path, "theory.md",
                  "$$\nA = \\sum_i \\alpha_{\\mathrm{s},i} S_i"
                  " + \\sum_k \\alpha_{s,k} S_k\n$$\n")
    _, failures = css.check([path])
    assert len(failures) == 1
    assert "\\alpha_s" in failures[0]


def test_a_run_inside_one_wrapper_is_upright_throughout(tmp_path):
    """``L_\\mathrm{n,w,eq}`` sets three upright components, not one."""
    path = _write(tmp_path, "impact.md",
                  "$L'_\\mathrm{n,w} = L_\\mathrm{n,w,eq}"
                  " - \\Delta L_\\mathrm{w} + K$\n")
    _, failures = css.check([path])
    assert failures == []


def test_an_index_beside_an_upright_run_is_not_a_collision(tmp_path):
    """The index keeps its own slope where the wrapper does not cover it."""
    path = _write(tmp_path, "flanking.md",
                  "$D_{\\mathrm{v},ij}$ and $K_{ij}$ over junction $ij$.\n")
    _, failures = css.check([path])
    assert failures == []


def test_a_translation_pattern_is_not_a_label(tmp_path):
    """A regex spells a backslash twice; nothing in it is mathematics."""
    path = _write(tmp_path, "i18n.py",
                  'PATTERNS = [\n'
                  '    (r"^\\$f_\\\\mathrm\\{c\\}\\$ = (\\d+) Hz$",'
                  ' r"$f_c$ = \\1 Hz"),\n'
                  ']\n')
    _, failures = css.check([path])
    assert failures == []


def test_docstring_mathematics_is_read(tmp_path):
    """A module states its formulae in ``:math:`` roles and ``.. math::``."""
    path = _write(tmp_path, "module.py",
                  'r"""Impact insulation.\n'
                  '\n'
                  'The average is :math:`\\sum_i 10^{L_i/10}`.\n'
                  '\n'
                  '.. math::\n'
                  '   L\'_\\mathrm{n} = L_\\mathrm{i} + 10 \\log_{10}(A/A_0)\n'
                  '\n'
                  '"""\n')
    _, failures = css.check([path])
    assert len(failures) == 1
    assert "L_i" in failures[0]


def test_a_declared_page_is_left_alone(tmp_path, monkeypatch):
    """The escape hatch, matched on the repository-relative tail of the path."""
    (tmp_path / "reference").mkdir()
    path = _write(tmp_path, "reference/glossary.md",
                  "$L_N$ is a percentile; $L_\\mathrm{N}$ is sonar noise.\n")
    assert css.check([path])[1], "undeclared, this page is a failure"
    monkeypatch.setitem(css.DECLARED, "reference/glossary.md",
                        {("L", "N"): "the ambiguity table's own subject"})
    _, failures = css.check([path])
    assert failures == []


def test_the_excluded_trees_are_not_collected(tmp_path):
    """Errata transcribe a source; drawing modules are filed by domain."""
    (tmp_path / "reference").mkdir()
    (tmp_path / "_plot").mkdir()
    for rel in ("reference/errata.md", "_plot/building.py", "guide.md"):
        _write(tmp_path, rel, "$L_i$ and $L_\\mathrm{i}$\n")
    collected = {p.name for p in css.collect([str(tmp_path)])}
    assert collected == {"guide.md"}


def test_the_tree_it_ships_with_passes():
    """The gate is green on this repository, which is what CI asserts."""
    root = pathlib.Path(__file__).resolve().parent.parent
    paths = css.collect([str(root / r) for r in css.DEFAULT_ROOTS])
    _, failures = css.check(paths)
    assert failures == []
