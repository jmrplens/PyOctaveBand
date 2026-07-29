#  Copyright (c) 2026. Jose M. Requena-Plens

"""One-line canonical figures for the library's result objects.

Every public result dataclass exposes a thin ``plot(ax=None, **kwargs)``
method that delegates to a rendering function in this module, reproducing
the essence of its documentation figure in a single call::

    res = room_parameters(ir, fs)
    res.plot()

matplotlib is a *soft* dependency: importing :mod:`phonometry` and running
any computation works without it, and only calling ``.plot()`` (or the
functions here) requires it.  The import is therefore performed lazily and
raises a clear :class:`ImportError` with installation guidance when the
package is missing, mirroring :func:`phonometry.filter_design._showfilter`.

The functions are pure renderers: they never call ``plt.show``; when ``ax``
is ``None`` they create a fresh figure and axes, and they always *return*
the :class:`~matplotlib.axes.Axes` (or an array of axes for multi-panel
figures) so the plot can be composed into a larger layout.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.container import BarContainer
    from matplotlib.typing import ColorType

    from ..building.insulation import (
        ImpactRatingResult,
        WeightedRatingResult,
    )
    from ..room.room_acoustics import RoomAcousticsResult

_INSTALL_HINT = (
    "Plotting requires matplotlib. Install it with: pip install phonometry[plot]"
)

#: Nominal octave-band centres used by the STI computation (125 Hz - 8 kHz).
_STI_BAND_CENTERS = (125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)

#: Default legend placement shared by the band-spectrum figures.
_LEGEND_UPPER_RIGHT: Final = "upper right"

# ---------------------------------------------------------------------------
# Shared artist colors (matplotlib "tab10" hues plus neutral greys).
#
# The measured/primary series is drawn in _C_PRIMARY, reference curves and
# limit lines in _C_REFERENCE, unfavourable/secondary annotations in
# _C_SECONDARY, and de-emphasised context (invalid bands, threshold guides,
# curve families, companion series) in the single neutral _C_MUTED.
#
# Two renderers keep a fixed per-metric identity color instead: ECMA-418-2
# tonality is red and roughness is brown across the documentation, so those
# literals live in their renderers with a comment, not here.
# ---------------------------------------------------------------------------
_C_PRIMARY: Final = "#1f77b4"
_C_PRIMARY_LIGHT: Final = "#aec7e8"
_C_REFERENCE: Final = "#d62728"
_C_SECONDARY: Final = "#ff7f0e"
_C_SECONDARY_LIGHT: Final = "#ffbb78"
_C_TERTIARY: Final = "#2ca02c"
_C_QUATERNARY: Final = "#9467bd"
_C_MUTED: Final = "#9e9e9e"
_C_EDGE: Final = "#555555"

#: Standard-normal quantile of 0.9 (the 10 % / 90 % fractile offset).
_Z90: Final = 1.2816

# ---------------------------------------------------------------------------
# Signed wave-field colormaps: the zero of a diverging field should blend
# into the page, so the default is picked from the axes background at draw
# time -- RdBu_r (white centre) on a light background and the registered
# phonometry_field_dark (black centre) on a dark one.
# ---------------------------------------------------------------------------

#: Diverging wave-field colormap of a light axes background (white centre).
_FIELD_CMAP_LIGHT: Final = "RdBu_r"
#: Diverging wave-field colormap of a dark axes background (black centre).
_FIELD_CMAP_DARK: Final = "phonometry_field_dark"


def _register_field_dark_cmap() -> None:
    """Register ``phonometry_field_dark``, the dark-background wave-field map.

    Matplotlib's diverging ``berlin`` map anchors on a dark centre, but its
    centre is a dark maroon that reads as a wash over a dark page.  The
    registered map subtracts that centre bias instead of scaling it away::

        out(x) = clip(berlin(x) - w(x) * berlin(0.5), 0, 1)

    with the weight ``w(x)`` falling linearly from 1 at the centre to 0 at
    60 % amplitude.  Zero maps to true black -- the field blends into a dark
    background exactly as the white centre of ``RdBu_r`` blends into a light
    one -- while low-amplitude *differences* keep their size (a
    multiplicative fade toward black would crush them into invisibility).
    """
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap

    if _FIELD_CMAP_DARK in mpl.colormaps:
        return
    base = mpl.colormaps["berlin"]
    xs = np.linspace(0.0, 1.0, 256)
    cols = base(xs)[:, :3]
    c0 = np.asarray(base(0.5)[:3])
    w = np.clip(1.0 - np.abs(2.0 * xs - 1.0) / 0.6, 0.0, 1.0)
    out = np.clip(cols - w[:, None] * c0, 0.0, 1.0)
    mpl.colormaps.register(
        LinearSegmentedColormap.from_list(_FIELD_CMAP_DARK, out))


def _field_cmap(ax: Axes) -> str:
    """Background-aware diverging colormap name for a signed wave field.

    Reads the axes patch color (so a ``dark_background`` style, or any
    explicitly dark facecolor, is honoured) and returns the map whose centre
    matches it.  Registers :data:`_FIELD_CMAP_DARK` on first use, so the name
    is also available for an explicit ``cmap`` override.
    """
    from matplotlib.colors import to_rgb

    _register_field_dark_cmap()
    r, g, b = to_rgb(ax.get_facecolor())
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return _FIELD_CMAP_DARK if luminance < 0.5 else _FIELD_CMAP_LIGHT


# ---------------------------------------------------------------------------
# Background-aware shaded fills: the same idea as _field_cmap applied to the
# opaque area regions (acceptance corridors, period zones, tolerance bands).
# An alpha-composited fill cannot be legible on both pages at once -- 10 % of a
# mid hue is a pale tint over white but lands within a couple of levels of the
# near-black dark page -- so the wash is *derived* from the page instead, and
# returned opaque (svglib drops alpha when a fiche figure is rendered to PDF).
# ---------------------------------------------------------------------------

#: sRGB (IEC 61966-2-1) to CIE XYZ matrix, D65 white, 2 degree observer.
_SRGB_TO_XYZ: Final = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)

#: CIE D65 reference white in XYZ, the sRGB adopted white point.
_D65_WHITE: Final = np.array([0.95047, 1.00000, 1.08883])

#: CIE standard actual response constants (CIE 15:2004), 216/24389 and 24389/27.
_LAB_EPSILON: Final = 216.0 / 24389.0
_LAB_KAPPA: Final = 24389.0 / 27.0

#: Perceptual distance a derived fill keeps from the page it is drawn on.
#: A CIEDE2000 difference of 1 is the just-noticeable difference, 2-3 is
#: "visible when looked for", and a difference read at a glance without
#: hunting starts around 10; a documentation figure is skimmed rather than
#: inspected, so the wash is placed just above that with a margin over the
#: threshold ``scripts/check_figure_contrast.py`` gates on.
_FILL_DELTA_E: Final = 12.0

#: Mixing weights are searched on this fixed grid rather than by bisection, so
#: the returned colour is a quantised function of its inputs and the figure
#: bytes stay reproducible (a bisection would let a last-bit difference in the
#: colour-difference evaluation walk into a different final weight).
_FILL_WEIGHT_STEPS: Final = 512


def _srgb_to_lab(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    """CIE L*a*b* (D65, 2 degree observer) of a 0-1 sRGB triple."""
    channels = np.asarray(rgb, dtype=np.float64)
    linear = np.where(
        channels <= 0.04045, channels / 12.92, ((channels + 0.055) / 1.055) ** 2.4
    )
    ratio = (_SRGB_TO_XYZ @ linear) / _D65_WHITE
    f = np.where(
        ratio > _LAB_EPSILON,
        np.cbrt(ratio),
        (_LAB_KAPPA * ratio + 16.0) / 116.0,
    )
    return (
        float(116.0 * f[1] - 16.0),
        float(500.0 * (f[0] - f[1])),
        float(200.0 * (f[1] - f[2])),
    )


def delta_e_2000(
    first: tuple[float, float, float], second: tuple[float, float, float]
) -> float:
    """CIEDE2000 colour difference between two 0-1 sRGB triples.

    The CIE's current colour-difference formula (CIE 142-2001), implemented
    from the notes and reference data of Sharma, Wu and Dalal (2005), *The
    CIEDE2000 color-difference formula: implementation notes, supplementary
    test data, and mathematical observations*, Color Research & Application
    30(1), 21-30, with unit parametric factors ``kL = kC = kH = 1``.  A value
    of about 1 is the just-noticeable difference.

    :param first: First colour as an ``(r, g, b)`` triple in 0-1 sRGB.
    :param second: Second colour, same convention.
    :return: The colour difference ``dE00``, zero for identical colours.
    """
    l_1, a_1, b_1 = _srgb_to_lab(first)
    l_2, a_2, b_2 = _srgb_to_lab(second)

    c_mean = 0.5 * (np.hypot(a_1, b_1) + np.hypot(a_2, b_2))
    g = 0.5 * (1.0 - np.sqrt(c_mean**7 / (c_mean**7 + 25.0**7)))
    ap_1, ap_2 = (1.0 + g) * a_1, (1.0 + g) * a_2
    cp_1, cp_2 = np.hypot(ap_1, b_1), np.hypot(ap_2, b_2)
    hp_1 = float(np.degrees(np.arctan2(b_1, ap_1)) % 360.0) if (ap_1 or b_1) else 0.0
    hp_2 = float(np.degrees(np.arctan2(b_2, ap_2)) % 360.0) if (ap_2 or b_2) else 0.0

    d_l = l_2 - l_1
    d_c = cp_2 - cp_1
    if cp_1 * cp_2 == 0.0:
        d_h = 0.0
    elif abs(hp_2 - hp_1) <= 180.0:
        d_h = hp_2 - hp_1
    else:
        d_h = hp_2 - hp_1 - 360.0 * float(np.sign(hp_2 - hp_1))
    d_hh = 2.0 * np.sqrt(cp_1 * cp_2) * np.sin(np.radians(d_h) / 2.0)

    l_bar = 0.5 * (l_1 + l_2)
    c_bar = 0.5 * (cp_1 + cp_2)
    if cp_1 * cp_2 == 0.0:
        h_bar = hp_1 + hp_2
    elif abs(hp_1 - hp_2) <= 180.0:
        h_bar = 0.5 * (hp_1 + hp_2)
    elif hp_1 + hp_2 < 360.0:
        h_bar = 0.5 * (hp_1 + hp_2 + 360.0)
    else:
        h_bar = 0.5 * (hp_1 + hp_2 - 360.0)

    t = (
        1.0
        - 0.17 * np.cos(np.radians(h_bar - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * h_bar))
        + 0.32 * np.cos(np.radians(3.0 * h_bar + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * h_bar - 63.0))
    )
    s_l = 1.0 + 0.015 * (l_bar - 50.0) ** 2 / np.sqrt(20.0 + (l_bar - 50.0) ** 2)
    s_c = 1.0 + 0.045 * c_bar
    s_h = 1.0 + 0.015 * c_bar * t
    d_theta = 30.0 * np.exp(-(((h_bar - 275.0) / 25.0) ** 2))
    r_t = -2.0 * np.sqrt(c_bar**7 / (c_bar**7 + 25.0**7)) * np.sin(
        np.radians(2.0 * d_theta)
    )

    return float(
        np.sqrt(
            (d_l / s_l) ** 2
            + (d_c / s_c) ** 2
            + (d_hh / s_h) ** 2
            + r_t * (d_c / s_c) * (d_hh / s_h)
        )
    )


def _page_color(ax: Axes | None, background: str | None) -> tuple[float, float, float]:
    """The page a fill will be drawn on, as a 0-1 sRGB triple.

    Explicit ``background`` wins; otherwise the axes patch colour is read (so
    a ``dark_background`` style, or any explicitly dark facecolor, is
    honoured), falling back to the ``axes.facecolor`` setting when there is no
    axes yet.
    """
    from matplotlib import rcParams
    from matplotlib.colors import to_rgb

    if background is not None:
        return to_rgb(background)
    if ax is not None:
        return to_rgb(ax.get_facecolor())
    return to_rgb(rcParams["axes.facecolor"])


def theme_fill(
    color: ColorType,
    ax: Axes | None = None,
    *,
    background: str | None = None,
    delta_e: float = _FILL_DELTA_E,
) -> tuple[float, float, float]:
    """Opaque wash of ``color`` that stays legible on the page it is drawn on.

    Shading an area with ``alpha`` composites the hue onto whatever is behind
    it, which is why the same call reads as a pale tint on the white page and
    disappears into the near-black one: 10 % of a mid hue over black is within
    a couple of levels of black.  This mixes the page colour *towards* the hue
    instead, by the smallest amount that puts the result :data:`_FILL_DELTA_E`
    away from the page in :func:`delta_e_2000`.  The direction takes care of
    itself: over a light page the wash is a pale tint of the hue, over a dark
    one it is a raised shade of it, exactly as :func:`_field_cmap` picks the
    diverging map whose centre matches the page.

    The result is opaque, so it survives the fiche PDF pipeline (svglib drops
    alpha) and does not darken the gridlines it covers; pass it as ``color``
    and leave ``alpha`` unset::

        ax.fill_between(f, lower, upper, color=theme_fill(_C_PRIMARY, ax))

    Pass a saturated hue: a colour already close to the page (a pale tint, or
    white over white) has nowhere to move, and the strongest available wash --
    the hue itself -- is returned instead.

    :param color: Base hue, any matplotlib colour specification.
    :param ax: Axes whose patch colour is the page; ``None`` reads the current
        ``axes.facecolor`` setting.
    :param background: Explicit page colour, overriding ``ax``.
    :param delta_e: Target CIEDE2000 distance from the page.
    :return: The wash as an ``(r, g, b)`` triple in 0-1 sRGB.
    """
    from matplotlib.colors import to_rgb

    hue = to_rgb(color)
    page = _page_color(ax, background)
    weight = _fill_weight(hue, page, delta_e)
    red, green, blue = (
        p + weight * (h - p) for p, h in zip(page, hue, strict=True)
    )
    return (red, green, blue)


def theme_fill_alpha(
    color: ColorType,
    ax: Axes | None = None,
    *,
    background: str | None = None,
    delta_e: float = _FILL_DELTA_E,
) -> float:
    """Opacity at which ``color`` reaches :func:`theme_fill`'s contrast.

    Compositing a hue at opacity ``a`` over the page and mixing the page
    towards that hue by weight ``a`` are the same colour, so this is the same
    computation as :func:`theme_fill` read as an opacity instead of a colour.
    Prefer :func:`theme_fill`: an opaque wash survives the fiche PDF pipeline
    and never dulls what it covers.  Use this only where fills genuinely have
    to blend -- overlapping directivity lobes, nested uncertainty bands, a
    wash over a hand-drawn polar grid -- and the alternative would be hiding
    one artist behind another.

    :param color: Base hue, any matplotlib colour specification.
    :param ax: Axes whose patch colour is the page.
    :param background: Explicit page colour, overriding ``ax``.
    :param delta_e: Target CIEDE2000 distance from the page.
    :return: The opacity to pass as ``alpha``, in 0-1.
    """
    from matplotlib.colors import to_rgb

    return _fill_weight(to_rgb(color), _page_color(ax, background), delta_e)


@lru_cache(maxsize=256)
def _fill_weight(
    hue: tuple[float, float, float],
    page: tuple[float, float, float],
    delta_e: float,
) -> float:
    """Smallest grid weight mixing ``page`` towards ``hue`` to reach ``delta_e``.

    Cached: a figure draws the same handful of hues on the same two pages over
    and over, and each miss walks the :data:`_FILL_WEIGHT_STEPS` grid.
    """
    for step in range(1, _FILL_WEIGHT_STEPS + 1):
        weight = step / _FILL_WEIGHT_STEPS
        red, green, blue = (
            p + weight * (h - p) for p, h in zip(page, hue, strict=True)
        )
        if delta_e_2000((red, green, blue), page) >= delta_e:
            return weight
    return 1.0


#: Common axis labels reused across the underwater-propagation plots.
_LABEL_DEPTH_M: Final = "Depth [m]"
_LABEL_TL_DB: Final = "Transmission loss [dB]"
_LABEL_RANGE_KM: Final = "Range [km]"

#: Spanish translations of the handful of fixed strings these *shared*
#: renderers set directly (axis labels, generic legend entries). Each
#: ``_plot/<domain>.py`` module carries its own richer ``_STRINGS``/``_t`` for
#: titles and quantity-specific text; ``_t`` here only covers what common.py
#: itself draws, so a domain module that passes an already-localised string
#: through (e.g. an ``xlabel`` built with its own ``_t``) gets it back
#: unchanged: the lookup misses and falls back to the given text. English is
#: always a no-op, so the byte-identical English guarantee holds.
_STRINGS: dict[str, str] = {
    "Frequency [Hz]": "Frecuencia [Hz]",
    "Band": "Banda",
    "Measured": "Medido",
    "Shifted reference": "Referencia desplazada",
    "Unfavourable deviations": "Desviaciones desfavorables",
    "Time [s]": "Tiempo [s]",
    "Sample": "Muestra",
    "10-90 % fractile band": "Banda de fractiles 10-90 %",
}


def _t(text: str, language: str = "en") -> str:
    """Localise a fixed string; English is returned verbatim (byte-identical)."""
    return _STRINGS.get(text, text) if language == "es" else text


def _import_pyplot() -> Any:
    """Import :mod:`matplotlib.pyplot` lazily with an actionable error."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(_INSTALL_HINT) from exc
    return plt


def _new_axes() -> Axes:
    """Create a single fresh figure + axes and return the axes."""
    plt = _import_pyplot()
    _fig, ax = plt.subplots()
    return cast("Axes", ax)


def _new_axes_column(n: int, **kwargs: Any) -> np.ndarray:
    """Create a stacked column of ``n`` axes and return them as an array."""
    plt = _import_pyplot()
    _fig, axes = plt.subplots(n, 1, **kwargs)
    result: np.ndarray = np.atleast_1d(axes)
    return result


def _freq_axis(ax: Axes, freqs: np.ndarray, *, language: str = "en") -> None:
    """Configure a logarithmic frequency x-axis labelled with band centres."""
    import matplotlib.ticker as mticker

    ax.set_xscale("log")
    ax.set_xticks(list(freqs))
    ax.set_xticklabels([_format_freq(f) for f in freqs], rotation=45, ha="right")
    # Suppress the log-scale minor-tick labels (2x10^2, 3x10^2, ...) that would
    # otherwise collide with off-decade band centres such as 750 or 1500 Hz.
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel(_t("Frequency [Hz]", language))


def _format_freq(f: float) -> str:
    """Short human label for a band centre frequency."""
    if f >= 1000.0:
        text = f"{f / 1000.0:g}k"
    else:
        text = f"{f:g}"
    return text


#: Nominal octave-band centres (IEC 61672 / ISO 266) spanning the acoustic
#: range, used to label *continuous* logarithmic frequency axes.
_OCTAVE_NOMINAL: Final = (
    1.0, 2.0, 4.0, 8.0, 16.0, 31.5, 63.0, 125.0, 250.0, 500.0,
    1000.0, 2000.0, 4000.0, 8000.0, 16000.0, 31500.0, 63000.0, 125000.0,
)

#: Nominal one-third-octave centres for optional unlabelled minor ticks.
_THIRD_NOMINAL: Final = (
    1.0, 1.25, 1.6, 2.0, 2.5, 3.15, 4.0, 5.0, 6.3, 8.0, 10.0, 12.5, 16.0,
    20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0,
    250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0,
    2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0, 10000.0,
    12500.0, 16000.0, 20000.0, 25000.0, 31500.0, 40000.0, 50000.0,
    63000.0, 80000.0, 100000.0, 125000.0,
)


def format_frequency_axis(
    ax: Axes,
    fmin: float | None = None,
    fmax: float | None = None,
    *,
    minor: str | None = "thirds",
) -> None:
    """Label a *continuous* logarithmic frequency x-axis with band centres.

    Places major ticks at the nominal octave centres (16 Hz, 31.5 Hz, ...,
    16 kHz) that fall inside the axis range, labels them with the short
    :func:`_format_freq` form (``1k``, ``2k``, ...) and replaces the default
    power-of-ten log formatter (``10^2``, ``10^3``) that reads poorly in
    acoustics.  ``minor="thirds"`` adds unlabelled one-third-octave minor
    ticks; ``minor=None`` clears the minor ticks.

    The x-axis is switched to a log scale when it is not already one.  The
    tick set is clipped to the data range: pass ``fmin`` / ``fmax`` to fix it
    explicitly, otherwise the current axis limits (read after the data has
    been drawn) are used, so a plot that starts at 100 Hz is never dragged
    back to 16 Hz.  The function only touches ticks and never changes data.
    """
    import matplotlib.ticker as mticker

    if ax.get_xscale() != "log":
        ax.set_xscale("log")
    lo, hi = ax.get_xlim()
    if fmin is not None:
        lo = fmin
    if fmax is not None:
        hi = fmax
    lo, hi = min(lo, hi), max(lo, hi)
    majors = [f for f in _OCTAVE_NOMINAL if lo <= f <= hi]
    ax.xaxis.set_major_locator(mticker.FixedLocator(majors))
    ax.xaxis.set_major_formatter(
        mticker.FixedFormatter([_format_freq(f) for f in majors])
    )
    if minor == "thirds":
        minors = [
            f for f in _THIRD_NOMINAL if lo <= f <= hi and f not in majors
        ]
        ax.xaxis.set_minor_locator(mticker.FixedLocator(minors))
    else:
        ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())


def _band_axis(
    ax: Axes,
    labels_or_freqs: np.ndarray | Sequence[str] | Sequence[float],
    *,
    xlabel: str | None = "Frequency [Hz]",
    language: str = "en",
) -> np.ndarray:
    """Categorical band x-axis: evenly spaced positions labelled with centres.

    Band bars/curves are drawn on evenly spaced positions (a *linear* axis)
    so they stay legible; the tick labels carry the band centres (numeric
    input is shortened via :func:`_format_freq`) or the given strings.
    Returns the positions.  ``xlabel=None`` leaves the axis label untouched
    (e.g. a shared-x upper panel).
    """
    labels = [
        item if isinstance(item, str) else _format_freq(float(item))
        for item in list(labels_or_freqs)
    ]
    positions = np.arange(len(labels), dtype=np.float64)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    if xlabel is not None:
        ax.set_xlabel(_t(xlabel, language))
    return positions


def _fractile_band(
    ax: Axes,
    freqs: np.ndarray,
    median: np.ndarray,
    spread_lower: np.ndarray,
    spread_upper: np.ndarray,
    *,
    color: str,
    floor: float | None = None,
    language: str = "en",
) -> None:
    """Shade the 10-90 % fractile band around a median spectrum.

    The band spans ``median - z90*spread_lower`` to ``median +
    z90*spread_upper`` with the standard-normal :data:`_Z90` quantile;
    ``floor`` clamps the lower edge (e.g. NIPTS cannot be negative). The legend
    entry is drawn here, so it goes through this module's own :func:`_t` for
    parity with the localised labels its callers set.
    """
    lower = median - _Z90 * spread_lower
    if floor is not None:
        lower = np.maximum(lower, floor)
    ax.fill_between(freqs, lower, median + _Z90 * spread_upper, zorder=0,
                    color=theme_fill(color, ax),
                    label=_t("10-90 % fractile band", language))


def _hatch_invalid(bars: BarContainer, mask: np.ndarray) -> None:
    """Hatch (and outline) the bars flagged invalid/unusable by ``mask``."""
    for bar, bad in zip(bars, np.asarray(mask, dtype=bool), strict=True):
        if bad:
            bar.set_hatch("//")
            bar.set_edgecolor(_C_EDGE)


# ---------------------------------------------------------------------------
# Zwicker loudness (ISO 532-1)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# ECMA-418-2 Sottek loudness
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# ISO 532-2 Moore-Glasberg loudness
# ---------------------------------------------------------------------------














# ---------------------------------------------------------------------------
# Electroacoustics: distortion & frequency response (IEC 60268-3 / Bendat)
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# Underwater acoustics (ISO 17208 ship radiated noise, ISO 18406 pile driving)
# ---------------------------------------------------------------------------






































# ---------------------------------------------------------------------------
# Speech transmission index (IEC 60268-16)
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# Weighted single-number ratings (ISO 717-1 / ISO 717-2)
# ---------------------------------------------------------------------------


def _plot_rating(
    band_centers: np.ndarray,
    measured: np.ndarray,
    reference: np.ndarray,
    *,
    impact: bool,
    title: str,
    ylabel: str,
    measured_label: str = "Measured",
    ylim: tuple[float, float] | None = None,
    ax: Axes | None,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Shared renderer for the shifted-reference rating figures.

    Draws the measured curve against the shifted reference and shades the
    unfavourable deviations: where the reference exceeds the measurement
    for airborne insulation and absorption (higher is better) and where the
    measurement exceeds the reference for impact sound (lower is better).
    Extra keyword arguments style the measured curve (its primary artist).
    """
    ax = ax if ax is not None else _new_axes()
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("label", _t(measured_label, language))
    ax.plot(band_centers, measured, "o-", **kwargs)
    ax.plot(band_centers, reference, "s--", color=_C_REFERENCE,
            label=_t("Shifted reference", language))
    unfavourable = _unfavourable_mask(measured, reference, impact)
    ax.fill_between(
        band_centers,
        measured,
        reference,
        where=unfavourable.tolist(),
        color=_C_SECONDARY,
        alpha=0.4,
        label=_t("Unfavourable deviations", language),
        interpolate=True,
    )
    _freq_axis(ax, band_centers, language=language)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    return ax


def _unfavourable_mask(
    measured: np.ndarray, reference: np.ndarray, impact: bool
) -> np.ndarray:
    """Bands whose deviation is unfavourable.

    Airborne (ISO 717-1, higher is better): the reference exceeds the
    measurement.  Impact (ISO 717-2, lower is better): the measurement
    exceeds the reference (the opposite sign).
    """
    if impact:
        return np.asarray(measured > reference)
    return np.asarray(measured < reference)








def _annotate_impact_500(
    ax: Axes,
    band_centers: np.ndarray,
    reference: np.ndarray,
    rating: int,
    *,
    language: str = "en",
) -> None:
    """Mark the 500 Hz read value on the reference curve and, when the
    octave-band -5 dB reduction applies (ISO 717-2 Clause 4.3.2), annotate
    the rating as ``read - 5 dB`` so the figure stays normatively truthful.
    """
    from .._i18n import decimal_comma

    if band_centers.size == 0:
        return
    idx = int(np.argmin(np.abs(band_centers - 500.0)))
    read_value = float(reference[idx])
    read_str = decimal_comma(f"{read_value:.0f}", language)
    read_label = (
        f"lectura 500 Hz = {read_str} dB" if language == "es"
        else f"500 Hz read = {read_str} dB"
    )
    ax.plot(
        [band_centers[idx]],
        [read_value],
        marker="D",
        ls="",
        color=_C_REFERENCE,
        ms=9,
        mfc="none",
        mew=1.6,
        zorder=5,
        label=read_label,
    )
    offset = rating - read_value
    if abs(offset) >= 0.5:  # octave-band -5 dB rule (Clause 4.3.2)
        rating_str = decimal_comma(str(rating), language)
        annotation = (
            f"índice = {rating_str} dB = {read_str} - 5 dB (regla de octava)"
            if language == "es"
            else f"rating = {rating_str} dB = {read_str} - 5 dB (octave rule)"
        )
        ax.annotate(
            annotation,
            xy=(band_centers[idx], read_value),
            xytext=(0.0, -32.0),
            textcoords="offset points",
            ha="center",
            fontsize="small",
            arrowprops={"arrowstyle": "->", "color": _C_EDGE},
        )
    ax.legend(loc="best", fontsize="small")


def _require_rating_curve(
    result: WeightedRatingResult | ImpactRatingResult,
) -> None:
    if (
        result.band_centers is None
        or result.measured is None
        or result.shifted_reference is None
    ):
        raise ValueError(
            "This rating result carries no band curve to plot (it was "
            "constructed without measured/reference data)."
        )




def _facade_x_axis(
    ax: Axes, freqs: np.ndarray | None, n: int, *, language: str = "en"
) -> np.ndarray:
    """Frequency x-axis when centres are known, else a labelled band index."""
    if freqs is None:
        x = np.arange(n, dtype=np.float64)
        ax.set_xticks(x)
        band_word = _t("Band", language)
        ax.set_xticklabels(
            [f"{band_word} {i + 1}" for i in range(n)], rotation=45, ha="right"
        )
        ax.set_xlabel(_t("Band", language))
        return x
    x = np.asarray(freqs, dtype=np.float64)
    _freq_axis(ax, x, language=language)
    return x






# ---------------------------------------------------------------------------
# Room acoustics (ISO 3382)
# ---------------------------------------------------------------------------




def _draw_decay_times(
    ax: Axes, positions: np.ndarray, result: RoomAcousticsResult, **kwargs: Any
) -> None:
    """Grouped EDT/T20/T30 bars, invalid bands hatched and greyed."""
    width = 0.27
    series = (
        ("EDT", result.edt, result.edt_valid, -width, _C_PRIMARY),
        ("T20", result.t20, result.t20_valid, 0.0, _C_SECONDARY),
        ("T30", result.t30, result.t30_valid, width, _C_TERTIARY),
    )
    for label, values, valid, offset, color in series:
        vals = np.asarray(values, dtype=np.float64)
        valid_arr = np.asarray(valid, dtype=bool)
        colors = [color if v else _C_MUTED for v in valid_arr]
        # Merge per-series defaults with the user kwargs (user wins) freshly
        # each iteration so an overriding label/color is not frozen by the
        # first band group.
        bar_kwargs = {"color": colors, "label": label, **kwargs}
        bars = ax.bar(
            positions + offset,
            np.nan_to_num(vals),
            width=width,
            **bar_kwargs,
        )
        _hatch_invalid(bars, ~valid_arr)


# ---------------------------------------------------------------------------
# Sound power (ISO 3744 / ISO 3741 / ISO 9614-2)
# ---------------------------------------------------------------------------




def _sound_power_designation(result: Any) -> str:
    """The standard designation matching a sound-power result's method.

    Distinguishes the reverberation-room (ISO 3741) and intensity (ISO 9614)
    determinations by their result types; the enveloping-surface pressure
    methods (:class:`~phonometry.sound_power.SoundPowerResult` and any other
    duck-typed result) fall back to ISO 3744/3746.
    """
    from ..emission.sound_power_intensity import SoundPowerIntensityResult
    from ..emission.sound_power_reverberation import ReverberationSoundPowerResult

    if isinstance(result, ReverberationSoundPowerResult):
        return "ISO 3741"
    if isinstance(result, SoundPowerIntensityResult):
        return "ISO 9614"
    return "ISO 3744/3746"


# ---------------------------------------------------------------------------
# Sound intensity (ISO 9614 / IEC 61043)
# ---------------------------------------------------------------------------


def _bar_width(positions: np.ndarray, k: float = 0.2) -> np.ndarray:
    """Per-bar widths for a *logarithmic* frequency axis.

    A constant linear width makes high-frequency bars almost invisible on
    a log axis (a 25 Hz-wide bar spans a third of an octave at 100 Hz but
    a thousandth of one at 10 kHz).  Scaling each width with its own centre
    frequency (``width_i = k * f_i``, the fractional-band style) keeps the
    *drawn* (log-space) width visually constant across the whole spectrum,
    so ``get_width() / f`` is the same constant ``k`` for every band.
    """
    return k * np.asarray(positions, dtype=np.float64)




# ---------------------------------------------------------------------------
# Schroeder decay curve (ISO 3382)
# ---------------------------------------------------------------------------




def _fit_segment(
    time: np.ndarray, level: np.ndarray, lo: float, hi: float
) -> np.ndarray | None:
    """Least-squares straight line over ``hi <= level <= lo``, over full time."""
    mask = (level <= lo) & (level >= hi) & np.isfinite(level)
    if int(np.count_nonzero(mask)) < 2:
        return None
    slope, intercept = np.polyfit(time[mask], level[mask], 1)
    return np.asarray(slope * time + intercept, dtype=np.float64)


# ---------------------------------------------------------------------------
# Impulse-response acquisition (ISO 18233)
# ---------------------------------------------------------------------------


def _time_axis(
    n: int, fs: int | None, *, language: str = "en"
) -> tuple[np.ndarray, str]:
    """Sample times in seconds when ``fs`` is known, else sample index."""
    if fs:
        return np.arange(n) / float(fs), _t("Time [s]", language)
    return np.arange(n, dtype=np.float64), _t("Sample", language)






# ---------------------------------------------------------------------------
# Surface scattering & diffusion (ISO 17497)
# ---------------------------------------------------------------------------










# ---------------------------------------------------------------------------
# Human vibration (ISO 8041-1 / ISO 2631 / ISO 5349 / Directive 2002/44/EC)
# ---------------------------------------------------------------------------








# ---------------------------------------------------------------------------
# Room-noise criteria (ANSI/ASA S12.2-2019)
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# Age-related hearing threshold (ISO 7029)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Noise-induced hearing loss (ISO 1999)
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# Impulsive-sound prominence (NT ACOU 112)
# ---------------------------------------------------------------------------


















def _plot_band_level_bars(
    ax: Axes | None,
    levels: np.ndarray,
    frequencies: np.ndarray | None,
    total_level: float,
    *,
    ylabel: str,
    title: str,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Per-band level bar chart with a band-summed total line (shared helper)."""
    from .._i18n import decimal_comma

    ax = ax if ax is not None else _new_axes()
    lw = np.asarray(levels, dtype=np.float64)
    n = lw.size
    if frequencies is not None:
        labels = [decimal_comma(f"{f:g}", language) for f in np.asarray(frequencies)]
        ax.set_xlabel(_t("Frequency [Hz]", language))
    else:
        labels = [str(i + 1) for i in range(n)]
        ax.set_xlabel(_t("Band", language))
    positions = np.arange(n)
    kwargs.setdefault("color", _C_PRIMARY)
    ax.bar(positions, lw, width=0.7, edgecolor=_C_EDGE, linewidth=0.6, **kwargs)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.axhline(
        total_level, color=_C_REFERENCE, ls="--", lw=1.2,
        label=f"total {decimal_comma(f'{total_level:.1f}', language)} dB",
    )
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, axis="y", alpha=0.3)
    return ax










# ---------------------------------------------------------------------------
# Measurement uncertainty budget (GUM)
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# Open-plan offices (ISO 3382-3)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Outdoor sound propagation (ISO 9613-2)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Impedance tube (ISO 10534-2)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Occupational noise exposure (ISO 9612)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Static airflow resistance (ISO 9053-1)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Building performance prediction (EN 12354-1 / EN 12354-2)
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# Field sound insulation spectra (ISO 16283-1 / ISO 16283-2)
# ---------------------------------------------------------------------------






def _plot_insulation_bands(
    curves: Sequence[tuple[str, np.ndarray]],
    *,
    ylabel: str,
    title: str,
    ax: Axes | None,
    **kwargs: Any,
) -> Axes:
    """Shared per-band insulation renderer (measurement bands are index-only).

    The ISO 16283 results do not carry their band centres, so the curves are
    drawn over band indices; user kwargs style the first (primary) curve
    only, mirroring :func:`plot_facade_insulation`.
    """
    ax = ax if ax is not None else _new_axes()
    n = curves[0][1].size
    x = np.arange(n, dtype=np.float64)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Band {i + 1}" for i in range(n)],
                       rotation=45, ha="right")
    ax.set_xlabel("Band")
    for index, (label, y) in enumerate(curves):
        opts: dict[str, Any] = {"label": label}
        if index == 0:
            opts.update(kwargs)
        ax.plot(x, y, "o-", **opts)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, alpha=0.3)
    return ax


# ---------------------------------------------------------------------------
# Building-acoustics measurement uncertainty (ISO 12999-1)
# ---------------------------------------------------------------------------




_ABSORPTION_QUANTITY_LABELS: Final = {
    "absorption_coefficient": "Sound absorption coefficient alpha_s",
    "equivalent_area": "Equivalent absorption area A_T [m2]",
    "practical_coefficient": "Practical absorption coefficient alpha_p",
}




