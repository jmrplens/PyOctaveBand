#  Copyright (c) 2026. Jose Manuel Requena Plens

"""Tests for the ``.plot()`` methods of the heavy-impact, plenum and wall-tie results.

Content tests rather than smoke tests: the drawn artists have to echo the
fields of the result they came from, in both languages, and the branches that
only some inputs reach (a source outside its tolerance, a spectrum without band
centre frequencies, a rigid rather than resilient tie) have to be exercised.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import phonometry as ph

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D

# ISO 717-2:2020 Table D.4, the octave-band field measurement.
_TABLE_D4 = (65.3, 64.5, 58.0, 55.8)
# ALA 16-091-4 (2016), tested to ASTM E1414/E1414M-11a: CAC 34.
_ALA_DNC = (
    14.4,
    18.6,
    21.7,
    24.1,
    23.4,
    30.3,
    33.7,
    35.2,
    41.6,
    44.2,
    42.1,
    36.8,
    35.7,
    36.0,
    36.9,
    37.9,
)


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    yield
    plt.close("all")


def _line_by_label(ax: Axes, needle: str) -> Line2D:
    """The first line whose label contains *needle*."""
    for line in ax.lines:
        if needle in line.get_label():
            return line
    msg = f"no line labelled with {needle!r}; got {[ln.get_label() for ln in ax.lines]}"
    raise AssertionError(msg)


# ---------------------------------------------------------------------------
# Heavy impact source conformance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("language", ["en", "es"])
def test_source_check_plot_draws_the_measured_spectrum(language: str) -> None:
    measured = [39.0, 31.0, 23.0, 17.0, 12.5]
    res = ph.building.check_heavy_impact_source(measured)
    ax = res.plot(language=language)
    line = _line_by_label(ax, "L_{FE}")
    np.testing.assert_allclose(line.get_ydata(), measured)
    # Five octave bands on evenly spaced categorical positions.
    assert [t.get_text() for t in ax.get_xticklabels()][:2] != []
    assert ax.get_ylabel()


def test_source_check_plot_marks_the_failing_band() -> None:
    """A band outside its tolerance gets its own marker series."""
    measured = [39.0, 31.0, 23.0, 17.0, 20.0]
    res = ph.building.check_heavy_impact_source(measured)
    assert not res.passed
    ax = res.plot()
    marked = [ln for ln in ax.lines if ln.get_marker() == "X"]
    assert len(marked) == 1
    np.testing.assert_allclose(marked[0].get_ydata(), [20.0])


def test_source_check_plot_has_no_failure_marker_when_conforming() -> None:
    res = ph.building.check_heavy_impact_source([39.0, 31.0, 23.0, 17.0, 12.5])
    ax = res.plot()
    assert not [ln for ln in ax.lines if ln.get_marker() == "X"]


def test_source_check_plot_shades_the_printed_tolerance_band() -> None:
    res = ph.building.check_heavy_impact_source(
        [v for v, _ in ph.building.HEAVY_IMPACT_SOURCES["bang_machine"]],
        "bang_machine",
    )
    ax = res.plot()
    assert ax.collections, "the tolerance band was not drawn"


def test_source_check_plot_forwards_kwargs() -> None:
    res = ph.building.check_heavy_impact_source([39.0, 31.0, 23.0, 17.0, 12.5])
    ax = res.plot(linewidth=3.0)
    assert _line_by_label(ax, "measured").get_linewidth() == 3.0


# ---------------------------------------------------------------------------
# Standardized maximum impact level
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("language", ["en", "es"])
def test_standardization_plot_draws_both_spectra(language: str) -> None:
    res = ph.building.standardized_maximum_impact_level(
        _TABLE_D4,
        41.4,
        [1.43, 3.7, 3.1, 2.38],
        frequency=[63.0, 125.0, 250.0, 500.0],
    )
    ax = res.plot(language=language)
    assert len(ax.lines) == 2
    values = [ln.get_ydata() for ln in ax.lines]
    assert any(np.allclose(v, res.measured) for v in values)
    assert any(np.allclose(v, res.standardized) for v in values)
    assert ax.collections, "the standardization correction was not shaded"


def test_standardization_plot_without_frequencies_uses_band_indices() -> None:
    res = ph.building.standardized_maximum_impact_level(_TABLE_D4, 41.4, 2.0)
    ax = res.plot()
    assert [t.get_text() for t in ax.get_xticklabels()] == ["1", "2", "3", "4"]


def test_standardization_plot_forwards_kwargs() -> None:
    res = ph.building.standardized_maximum_impact_level(_TABLE_D4, 41.4, 2.0)
    ax = res.plot(linewidth=2.5)
    assert any(ln.get_linewidth() == 2.5 for ln in ax.lines)


# ---------------------------------------------------------------------------
# A-weighted rating
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("language", ["en", "es"])
def test_rating_plot_bars_carry_the_corrected_values(language: str) -> None:
    res = ph.building.a_weighted_maximum_impact_level(_TABLE_D4)
    ax = res.plot(language=language)
    heights = [p.get_height() for p in ax.patches]
    np.testing.assert_allclose(heights, res.corrected, atol=1e-9)
    # The rating is drawn as a horizontal line.
    assert any(np.allclose(ln.get_ydata(), float(res.rating)) for ln in ax.lines)


def test_rating_plot_third_octave_has_twelve_bars() -> None:
    res = ph.building.a_weighted_maximum_impact_level([60.0] * 12)
    ax = res.plot()
    assert len(ax.patches) == 12


def test_rating_plot_forwards_kwargs_to_the_bars() -> None:
    res = ph.building.a_weighted_maximum_impact_level(_TABLE_D4)
    ax = res.plot(color="red")
    red = plt.matplotlib.colors.to_rgba("red")
    assert any(tuple(p.get_facecolor()) == red for p in ax.patches)


# ---------------------------------------------------------------------------
# Ceiling attenuation class
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("language", ["en", "es"])
def test_ceiling_attenuation_plot_draws_data_and_contour(language: str) -> None:
    res = ph.building.ceiling_attenuation_class(_ALA_DNC)
    ax = res.plot(language=language)
    values = [ln.get_ydata() for ln in ax.lines]
    assert any(np.allclose(v, res.measured) for v in values)
    assert any(np.allclose(v, res.shifted_reference) for v in values)
    assert "34" in ax.get_title()


def test_ceiling_attenuation_plot_shades_the_deficiencies() -> None:
    res = ph.building.ceiling_attenuation_class(_ALA_DNC)
    ax = res.plot()
    assert ax.collections, "the deficiencies were not shaded"


def test_ceiling_attenuation_plot_forwards_kwargs() -> None:
    res = ph.building.ceiling_attenuation_class(_ALA_DNC)
    ax = res.plot(linewidth=2.0)
    assert any(ln.get_linewidth() == 2.0 for ln in ax.lines)


# ---------------------------------------------------------------------------
# Plenum flanking path
# ---------------------------------------------------------------------------


def _plenum(**kwargs: list[float]) -> ph.building.PlenumFlankingResult:
    ceiling = [17.0, 21.0, 25.0, 29.0, 32.0]
    return ph.building.plenum_flanking_reduction_index(
        ceiling, ceiling, ceiling_length=4.75, plenum_height=0.43, **kwargs
    )


@pytest.mark.parametrize("language", ["en", "es"])
def test_plenum_plot_draws_the_path_against_the_two_ceilings(language: str) -> None:
    res = _plenum(frequency=[63.0, 125.0, 250.0, 500.0, 1000.0])
    ax = res.plot(language=language)
    total = res.reduction_index_source + res.reduction_index_receiving
    values = [ln.get_ydata() for ln in ax.lines]
    assert any(np.allclose(v, res.reduction_index) for v in values)
    assert any(np.allclose(v, total) for v in values)
    assert ax.collections, "the plenum penalty was not shaded"


def test_plenum_plot_without_frequencies_uses_band_indices() -> None:
    ax = _plenum().plot()
    assert [t.get_text() for t in ax.get_xticklabels()] == ["1", "2", "3", "4", "5"]


def test_plenum_plot_of_the_attenuated_model() -> None:
    res = _plenum(attenuation_source=[0.3] * 5, attenuation_receiving=[0.3] * 5)
    assert res.model == "attenuated"
    ax = res.plot()
    values = [ln.get_ydata() for ln in ax.lines]
    assert any(np.allclose(v, res.reduction_index) for v in values)


def test_plenum_plot_title_carries_the_geometry() -> None:
    ax = _plenum().plot()
    title = ax.get_title()
    assert "0.43" in title
    assert "4.75" in title


def test_plenum_plot_forwards_kwargs() -> None:
    ax = _plenum().plot(linewidth=2.5)
    assert any(ln.get_linewidth() == 2.5 for ln in ax.lines)


# ---------------------------------------------------------------------------
# Wall-tie coupling
# ---------------------------------------------------------------------------


def _coupling(**kwargs: str) -> ph.building.WallTieCouplingResult:
    freq = np.logspace(np.log10(50.0), np.log10(4000.0), 24)
    return ph.building.wall_tie_coupling_loss_factor(
        freq, 150.0, 170.0, 1.0e5, 1.2e5, ties_per_area=2.5, **kwargs
    )


@pytest.mark.parametrize("language", ["en", "es"])
def test_wall_tie_plot_draws_both_curves(language: str) -> None:
    res = _coupling(tie="butterfly")
    ax = res.plot(language=language)
    values = [ln.get_ydata() for ln in ax.lines]
    assert any(np.allclose(v, res.coupling_loss_factor) for v in values)
    assert any(np.allclose(v, res.rigid_coupling_loss_factor) for v in values)
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"
    assert ax.collections, "the isolation the tie buys was not shaded"


def test_wall_tie_plot_title_names_the_stiffness() -> None:
    ax = _coupling(tie="vertical_twist").plot()
    assert "MN/m" in ax.get_title()


def test_wall_tie_plot_of_a_rigid_connection_omits_the_stiffness() -> None:
    res = _coupling()
    assert res.tie_stiffness is None
    ax = res.plot()
    assert "MN/m" not in ax.get_title()


def test_wall_tie_plot_forwards_kwargs() -> None:
    ax = _coupling(tie="butterfly").plot(linewidth=3.0)
    assert any(ln.get_linewidth() == 3.0 for ln in ax.lines)


class _Plottable(Protocol):
    """What the language-validation test actually uses: the shared plot seam.

    ``object`` would claim the test never looks inside the result, and calling
    ``plot`` is the whole test; a union of the six concrete types would repeat
    the parametrize list. The protocol says exactly as much as the test needs.
    """

    def plot(self, *, language: str = ...) -> Axes: ...


# ---------------------------------------------------------------------------
# Language validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ph.building.check_heavy_impact_source([39.0, 31.0, 23.0, 17.0, 12.5]),
        lambda: ph.building.standardized_maximum_impact_level(_TABLE_D4, 41.4, 2.0),
        lambda: ph.building.a_weighted_maximum_impact_level(_TABLE_D4),
        lambda: ph.building.ceiling_attenuation_class(_ALA_DNC),
        _plenum,
        lambda: _coupling(tie="butterfly"),
    ],
)
def test_plot_rejects_an_unknown_language(factory: Callable[[], _Plottable]) -> None:
    result = factory()
    with pytest.raises(ValueError, match="Unknown language"):
        result.plot(language="fr")
