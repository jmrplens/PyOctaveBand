#  Copyright (c) 2026. Jose M. Requena-Plens
"""The shaded-region legibility gate reads real matplotlib SVG output.

``scripts/check_figure_contrast.py`` parses the committed figures rather than
the code that drew them, so its extraction has to survive everything
matplotlib emits: an omitted ``fill`` declaration where the colour is black
(the dark page), a background rectangle suppressed by ``set_axis_off``, area
regions expressed as ``<use>`` of a shared ``<defs>`` path, and glyphs and
legend swatches that are fills but not shaded regions. These tests render
figures with known-bad and known-good shading and check the verdict.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

pytest.importorskip("matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SCRIPTS = str(pathlib.Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import check_figure_contrast as cfc

from phonometry._plot.common import theme_fill


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _render(tmp_path: pathlib.Path, name: str) -> pathlib.Path:
    """Save the current figure as SVG and return its path."""
    path = tmp_path / f"{name}.svg"
    plt.savefig(path)
    return path


def _shaded_figure(dark: bool, *, faint: bool) -> None:
    """A band chart whose shaded corridor is either alpha-faint or derived."""
    plt.style.use("dark_background" if dark else "default")
    _fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [1.0, 1.4, 1.2])
    if faint:
        ax.fill_between([0, 2], 0.5, 1.5, color="#1f77b4", alpha=0.10)
    else:
        ax.fill_between([0, 2], 0.5, 1.5, color=theme_fill("#1f77b4", ax))


@pytest.mark.parametrize("dark", [False, True])
def test_a_faint_alpha_fill_is_reported(tmp_path: pathlib.Path, dark: bool) -> None:
    """A 10 % wash of a mid hue is below the threshold on either page."""
    _shaded_figure(dark, faint=True)
    regions = cfc.measure(_render(tmp_path, "faint"))
    assert len(regions) == 1
    assert not regions[0].ok
    assert regions[0].delta_e < cfc.DELTA_E_MIN


@pytest.mark.parametrize("dark", [False, True])
def test_a_derived_fill_passes(tmp_path: pathlib.Path, dark: bool) -> None:
    """The same corridor drawn with ``theme_fill`` clears the threshold."""
    _shaded_figure(dark, faint=False)
    regions = cfc.measure(_render(tmp_path, "derived"))
    assert len(regions) == 1
    assert regions[0].ok


def test_the_dark_page_is_read_from_the_suppressed_fill_declaration(
    tmp_path: pathlib.Path,
) -> None:
    """matplotlib omits ``fill`` for black, which must still read as black."""
    _shaded_figure(dark=True, faint=True)
    (region,) = cfc.measure(_render(tmp_path, "dark"))
    assert region.background == (0.0, 0.0, 0.0)


def test_axes_without_a_background_rectangle_do_not_confuse_the_page(
    tmp_path: pathlib.Path,
) -> None:
    """With ``set_axis_off`` the first patch is content, not the page.

    A schematic hides the axes frame, so matplotlib emits no background
    rectangle and the first ``patch_`` group is a drawn shape. Taking it for
    the page would compare it against itself and report a perfect zero.
    """
    from matplotlib.patches import Rectangle

    plt.style.use("default")
    _fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.add_patch(Rectangle((0.1, 0.1), 0.8, 0.8, facecolor="#9e9e9e", alpha=0.85))
    ax.autoscale_view()
    regions = cfc.measure(_render(tmp_path, "schematic"))
    assert [r.background for r in regions] == [(1.0, 1.0, 1.0)]
    assert all(r.ok for r in regions)


def test_text_and_legend_are_not_measured(tmp_path: pathlib.Path) -> None:
    """Glyphs and the legend frame are fills, but they are not shaded regions."""
    plt.style.use("default")
    _fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="series")
    ax.set_title("a title long enough to cover a good part of the axes")
    ax.legend(loc="center")
    assert cfc.measure(_render(tmp_path, "text-only")) == []


def test_wcag_contrast_ratio_spans_its_defined_range() -> None:
    """Black on white is the 21:1 maximum and a colour on itself is 1:1."""
    assert cfc.contrast_ratio((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)) == pytest.approx(21.0)
    assert cfc.contrast_ratio((0.3, 0.6, 0.9), (0.3, 0.6, 0.9)) == pytest.approx(1.0)


def test_composite_matches_source_over_alpha_blending() -> None:
    """The fill compositing is plain source-over, as a renderer does it."""
    assert cfc.composite((1.0, 0.0, 0.0), 0.25, (0.0, 0.0, 1.0)) == pytest.approx(
        (0.25, 0.0, 0.75)
    )


def test_main_reports_and_gates_a_faint_figure(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI exits non-zero on an offender and zero in report mode."""
    _shaded_figure(dark=True, faint=True)
    path = _render(tmp_path, "faint")
    assert cfc.main([str(path)]) == 1
    assert "faint.svg" in capsys.readouterr().out
    assert cfc.main(["--report", str(path)]) == 0
