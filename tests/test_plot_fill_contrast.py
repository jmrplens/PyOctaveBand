#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Background-aware shaded fills and the colour difference behind them.

``theme_fill`` derives an area wash from the page it will be drawn on, so a
shaded region reads on the white documentation page and on the dark one alike
instead of dissolving into whichever background the fixed alpha was tuned
against. The decision measure is CIEDE2000, checked here against the CIE's own
reference data, and the derived washes are checked to actually reach the target
distance on both pages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

pytest.importorskip("matplotlib")
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

from phonometry._plot.common import (
    _C_PRIMARY,
    _C_SECONDARY,
    _C_TERTIARY,
    _FILL_DELTA_E,
    _FILL_WEIGHT_STEPS,
    _srgb_to_lab,
    contrast_ratio,
    delta_e_2000,
    page_is_dark,
    theme_fill,
    theme_fill_alpha,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# CIE 142-2001 reference pairs for CIEDE2000, as published with Sharma, Wu and
# Dalal (2005), *The CIEDE2000 color-difference formula: implementation notes,
# supplementary test data, and mathematical observations*, Color Research &
# Application 30(1), 21-30, and distributed by the first author at
# https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/ as
# ``ciede2000testdata.txt``. Columns: L*a*b* of the first colour, L*a*b* of the
# second, and the expected dE00. The set is built to exercise every branch of
# the formula, including the hue-angle wrap and the blue-region rotation term.
SHARMA_2005_PAIRS: tuple[tuple[tuple[float, ...], tuple[float, ...], float], ...] = (
    ((50.0000, 2.6772, -79.7751), (50.0000, 0.0000, -82.7485), 2.0425),
    ((50.0000, 3.1571, -77.2803), (50.0000, 0.0000, -82.7485), 2.8615),
    ((50.0000, 2.8361, -74.0200), (50.0000, 0.0000, -82.7485), 3.4412),
    ((50.0000, -1.3802, -84.2814), (50.0000, 0.0000, -82.7485), 1.0000),
    ((50.0000, -1.1848, -84.8006), (50.0000, 0.0000, -82.7485), 1.0000),
    ((50.0000, -0.9009, -85.5211), (50.0000, 0.0000, -82.7485), 1.0000),
    ((50.0000, 0.0000, 0.0000), (50.0000, -1.0000, 2.0000), 2.3669),
    ((50.0000, -1.0000, 2.0000), (50.0000, 0.0000, 0.0000), 2.3669),
    ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0009), 7.1792),
    ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0010), 7.1792),
    ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0011), 7.2195),
    ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0012), 7.2195),
    ((50.0000, -0.0010, 2.4900), (50.0000, 0.0009, -2.4900), 4.8045),
    ((50.0000, -0.0010, 2.4900), (50.0000, 0.0010, -2.4900), 4.8045),
    ((50.0000, -0.0010, 2.4900), (50.0000, 0.0011, -2.4900), 4.7461),
    ((50.0000, 2.5000, 0.0000), (50.0000, 0.0000, -2.5000), 4.3065),
    ((50.0000, 2.5000, 0.0000), (73.0000, 25.0000, -18.0000), 27.1492),
    ((50.0000, 2.5000, 0.0000), (61.0000, -5.0000, 29.0000), 22.8977),
    ((50.0000, 2.5000, 0.0000), (56.0000, -27.0000, -3.0000), 31.9030),
    ((50.0000, 2.5000, 0.0000), (58.0000, 24.0000, 15.0000), 19.4535),
    ((50.0000, 2.5000, 0.0000), (50.0000, 3.1736, 0.5854), 1.0000),
    ((50.0000, 2.5000, 0.0000), (50.0000, 3.2972, 0.0000), 1.0000),
    ((50.0000, 2.5000, 0.0000), (50.0000, 1.8634, 0.5757), 1.0000),
    ((50.0000, 2.5000, 0.0000), (50.0000, 3.2592, 0.3350), 1.0000),
    ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
    ((63.0109, -31.0961, -5.8663), (62.8187, -29.7946, -4.0864), 1.2630),
    ((61.2901, 3.7196, -5.3901), (61.4292, 2.2480, -4.9620), 1.8731),
    ((35.0831, -44.1164, 3.7933), (35.0232, -40.0716, 1.5901), 1.8645),
    ((22.7233, 20.0904, -46.6940), (23.0331, 14.9730, -42.5619), 2.0373),
    ((36.4612, 47.8580, 18.3852), (36.2715, 50.5065, 21.2231), 1.4146),
    ((90.8027, -2.0831, 1.4410), (91.1528, -1.6435, 0.0447), 1.4441),
    ((90.9257, -0.5406, -0.9208), (88.6381, -0.8985, -0.7239), 1.5381),
    ((6.7747, -0.2908, -2.4247), (5.8714, -0.0985, -2.2286), 0.6377),
    ((2.0776, 0.0795, -1.1350), (0.9033, -0.0636, -0.5514), 0.9082),
)

#: The two documentation pages: the white one and the ``dark_background`` one.
PAGES = ("#ffffff", "#000000")


@pytest.mark.parametrize(("lab_1", "lab_2", "expected"), SHARMA_2005_PAIRS)
def test_delta_e_2000_matches_cie_reference_data(
    monkeypatch: pytest.MonkeyPatch,
    lab_1: tuple[float, ...],
    lab_2: tuple[float, ...],
    expected: float,
) -> None:
    """Every CIE 142-2001 reference pair reproduces to the published 4 decimals.

    The reference data is tabulated in L*a*b*, so the sRGB conversion is
    bypassed and the colour-difference core is exercised directly.
    """
    monkeypatch.setattr("phonometry._plot.common._srgb_to_lab", lambda colour: colour)
    assert delta_e_2000(lab_1, lab_2) == pytest.approx(expected, abs=5e-5)


def test_srgb_to_lab_anchors_on_the_white_point() -> None:
    """White is L* = 100 with no chroma, black is L* = 0 (D65, 2 degrees)."""
    assert _srgb_to_lab((1.0, 1.0, 1.0)) == pytest.approx((100.0, 0.0, 0.0), abs=1e-3)
    assert _srgb_to_lab((0.0, 0.0, 0.0)) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)


@pytest.mark.parametrize("page", PAGES)
@pytest.mark.parametrize("hue", [_C_PRIMARY, _C_SECONDARY, _C_TERTIARY, "#9467bd"])
def test_theme_fill_reaches_the_target_distance_on_both_pages(
    page: str, hue: str
) -> None:
    """The derived wash clears the target on the white and the dark page."""
    fill = theme_fill(hue, background=page)
    assert delta_e_2000(fill, to_rgb(page)) >= _FILL_DELTA_E


@pytest.mark.parametrize("page", PAGES)
@pytest.mark.parametrize("hue", [_C_PRIMARY, _C_SECONDARY, _C_TERTIARY, "#9467bd"])
def test_theme_fill_is_the_smallest_wash_that_clears_the_target(
    page: str, hue: str
) -> None:
    """One grid step less than the returned wash falls short of the target.

    The wash has to be the *weakest* one that works: a stronger fill than
    necessary competes with the data drawn over it.
    """
    page_rgb = np.asarray(to_rgb(page))
    hue_rgb = np.asarray(to_rgb(hue))
    weight = theme_fill_alpha(hue, background=page)
    weaker = page_rgb + (weight - 1.0 / _FILL_WEIGHT_STEPS) * (hue_rgb - page_rgb)
    assert delta_e_2000(tuple(weaker), to_rgb(page)) < _FILL_DELTA_E


@pytest.mark.parametrize("page", PAGES)
def test_theme_fill_and_theme_fill_alpha_describe_the_same_colour(page: str) -> None:
    """The opacity form composites to exactly the opaque form."""
    page_rgb = np.asarray(to_rgb(page))
    hue_rgb = np.asarray(to_rgb(_C_PRIMARY))
    alpha = theme_fill_alpha(_C_PRIMARY, background=page)
    composited = alpha * hue_rgb + (1.0 - alpha) * page_rgb
    assert theme_fill(_C_PRIMARY, background=page) == pytest.approx(composited)


def test_theme_fill_moves_away_from_the_page_in_both_directions() -> None:
    """The wash is a pale tint over a light page and a raised shade over a dark one."""
    light = theme_fill(_C_PRIMARY, background="#ffffff")
    dark = theme_fill(_C_PRIMARY, background="#000000")
    assert min(light) > 0.5, "a wash on the white page must stay light"
    assert max(dark) < 0.5, "a wash on the dark page must stay dark"
    assert sum(light) < 3.0, "a wash on the white page must be darker than it"
    assert sum(dark) > 0.0, "a wash on the dark page must be lighter than it"


def _axes_on(page: str) -> Axes:
    """A bare axes whose patch is *page*, which is where the page is read from."""
    _fig, ax = plt.subplots()
    ax.set_facecolor(page)
    return ax


def test_theme_fill_reads_the_page_from_the_axes() -> None:
    """A dark axes patch yields the dark wash without an explicit background."""
    _fig, ax = plt.subplots()
    ax.set_facecolor("#000000")
    assert theme_fill(_C_PRIMARY, ax) == theme_fill(_C_PRIMARY, background="#000000")
    ax.set_facecolor("#ffffff")
    assert theme_fill(_C_PRIMARY, ax) == theme_fill(_C_PRIMARY, background="#ffffff")
    plt.close("all")


@pytest.mark.parametrize(
    "page",
    [
        "#ffffff",
        "#f0f2f5",
        "#b0b0b0",
        "#808080",
        "#767676",
        "#4d4d4d",
        "#000000",
        "#0d1117",
        "#161b22",
        "#1c2128",
    ],
)
def test_a_page_is_dark_exactly_where_white_ink_beats_black_ink(page: str) -> None:
    """The rule is the crossover, not a threshold picked for the look of it.

    ``page_is_dark`` decides which of two fixed answers a drawing gets: the
    black-centred wave-field colormap or the white-centred one, the dark chip
    behind a label or the light one. So the honest boundary is the luminance
    at which white ink and black ink on that page have the same contrast
    ratio, and the classification has to agree with the ratios themselves at
    every page, including the mid greys where a weighted sum of the raw sRGB
    channels and the gamma-corrected luminance part company.
    """
    rgb = to_rgb(page)
    white = contrast_ratio(rgb, (1.0, 1.0, 1.0))
    black = contrast_ratio(rgb, (0.0, 0.0, 0.0))
    assert page_is_dark(_axes_on(page)) is (white > black)
    plt.close("all")


def test_a_mid_grey_page_takes_black_ink_and_is_not_a_dark_page() -> None:
    """The case that separates relative luminance from a raw channel average.

    ``#808080`` is halfway along the sRGB scale and nowhere near halfway up
    the luminance one: its relative luminance is 0.216, which leaves black on
    it well ahead of white. Reading it as a dark page would put the dark chip
    and the black-centred field map on a light grey drawing.
    """
    rgb = to_rgb("#808080")
    assert contrast_ratio(rgb, (0.0, 0.0, 0.0)) > contrast_ratio(rgb, (1.0, 1.0, 1.0))
    assert not page_is_dark(_axes_on("#808080"))
    plt.close("all")


def test_theme_fill_falls_back_to_the_hue_when_the_target_is_unreachable() -> None:
    """A colour that *is* the page has nowhere to move, so it comes back unchanged."""
    assert theme_fill("#ffffff", background="#ffffff") == pytest.approx((1.0, 1.0, 1.0))
    assert theme_fill_alpha("#ffffff", background="#ffffff") == 1.0


@pytest.mark.parametrize("page", PAGES)
@pytest.mark.parametrize("hue", [_C_PRIMARY, _C_SECONDARY, _C_TERTIARY, "#9467bd"])
def test_theme_fill_weight_is_not_on_a_knife_edge(page: str, hue: str) -> None:
    """The chosen grid step clears the target by far more than float noise.

    The committed figures are compared byte for byte across machines, so the
    weight search must not be able to land on a different step because a
    colour-difference evaluation differed in its last bit. The margin at the
    crossing is a whole grid step's worth of distance, ~15 orders of magnitude
    above that.
    """
    margin = delta_e_2000(theme_fill(hue, background=page), to_rgb(page))
    assert margin - _FILL_DELTA_E > 1e-6
