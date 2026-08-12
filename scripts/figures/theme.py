#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The shared look of every documentation figure: palette, theme and saving.

One subject, in three parts that cannot be separated. The palette and the
theme state are the same names read by every generator (``COLOR_FG``,
``COLOR_GRID``, ``CMAP_FIELD`` ...), rebound by :func:`set_theme` when the run
switches between the light and dark pages of the site. :func:`save_figure` is
the only writer: it translates the figure, adds the language/theme suffixes
and chooses deterministic SVG or lossless WebP, which is what makes ``make
graphs`` byte-stable and lets CI diff it. The rest is the styling vocabulary
the figures apply to their axes, plus :func:`measure_weighting_response`, the
measurement the weighting figures are drawn from and the tests guard.
"""

import os
import pathlib
from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as scipy_signal

from phonometry._plot.common import _register_field_dark_cmap

from . import _publish
from .i18n import _LANG_SUFFIX, _translate_figure, audit_figure

# Constants for professional styling
LABEL_FREQ_HZ = "Frequency [Hz]"
LABEL_LEVEL_DB = "Level [dB]"
COLOR_PRIMARY = "#1f77b4"
COLOR_SECONDARY = "#d62728"
COLOR_TERTIARY = "#2ca02c"
# The fourth series colour, the same purple the library palette carries as
# ``_C_QUATERNARY``. Named here because the CSS "purple" (#800080) that a
# figure reaches for instead is too dark to draw a line on the dark page.
COLOR_QUATERNARY = "#9467bd"
COLOR_GRID = "#e0e0e0"
# Neutral for de-emphasised *data* (a context bar, a reference series): mid
# grey reads on both pages, unlike COLOR_GRID, which is tuned to disappear
# into the one it is drawn on.
COLOR_MUTED = "#9e9e9e"

# Theme state: every figure is generated twice (light + "_dark" suffix).
# COLOR_FG replaces literal black so annotations stay visible on dark bg.
COLOR_FG = "black"
# Solid backing for a text box that has to stay readable over the plotted
# data: a shade off the page rather than the page itself, so the box reads as
# a panel and its edge is not the only thing separating it from the figure.
COLOR_PANEL = "#f0f2f5"
_FILENAME_SUFFIX = ""


# The dark-theme wave-field colormap (berlin with its centre bias subtracted
# so zero maps to true black) now lives in the library, where the result
# ``.plot()`` renderers pick it whenever the axes background is dark; the
# clips address it by name, so it is registered eagerly here.
_register_field_dark_cmap()

# Theme-dependent wave-field styling, switched by set_theme(): the diverging
# colormap of the instantaneous-pressure/velocity panels (zero maps to the
# page background on both themes) and the ink/halo pair of the annotations
# drawn *inside* those fields. Sequential magma panels (RMS maps, level maps)
# are theme-independent and keep their hardcoded white ink.
CMAP_FIELD = "RdBu_r"
FIELD_INK = "black"
FIELD_STROKE = "white"

# Where a sequential colormap is read for a *family of curves*, per theme. A
# ramp like viridis runs from a near-black purple to a near-white yellow, so
# neither end can be drawn on both pages: sampled from 0.05, the first two
# curves of a six-curve family were within a level and a half of the dark
# page. Each theme therefore reads the stretch of the ramp its own page leaves
# room for, and both stretches keep every entry clear of the page it is drawn
# on. This is not the wave-field colormap above: that one maps a *signed
# field* onto a page, this one picks apart curves that share an axis.
_SERIES_SPAN_LIGHT = (0.05, 0.85)
_SERIES_SPAN_DARK = (0.32, 1.0)
_SERIES_SPAN = _SERIES_SPAN_LIGHT

# Global matplotlib configuration
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.5,
        "grid.linestyle": "--",
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def series_colors(count: int, cmap: str = "viridis") -> np.ndarray:
    """``count`` colours off ``cmap``, over the span the current page allows.

    The one call every family of curves that shares an axis should use:
    ``ax.plot(..., color=colour)`` for each of ``series_colors(len(curves))``.
    Sampling the colormap directly picks up both of its unusable ends, and the
    unusable end is a different one on each theme.

    :param count: Number of curves in the family.
    :param cmap: Any sequential matplotlib colormap name.
    :return: An ``(count, 4)`` RGBA array, dark-to-light along the ramp.
    """
    lo, hi = _SERIES_SPAN
    return np.asarray(plt.get_cmap(cmap)(np.linspace(lo, hi, count)))


def set_theme(dark: bool) -> None:
    """Switch between the light (default) and dark documentation themes."""
    global COLOR_FG, COLOR_GRID, COLOR_PANEL, _FILENAME_SUFFIX
    global CMAP_FIELD, FIELD_INK, FIELD_STROKE, _SERIES_SPAN
    if dark:
        plt.style.use("dark_background")
        COLOR_FG = "white"
        COLOR_GRID = "#555555"
        COLOR_PANEL = "#1c2128"
        _FILENAME_SUFFIX = "_dark"
        CMAP_FIELD = "phonometry_field_dark"
        FIELD_INK = "white"
        FIELD_STROKE = "black"
        _SERIES_SPAN = _SERIES_SPAN_DARK
    else:
        plt.style.use("default")
        COLOR_FG = "black"
        COLOR_GRID = "#e0e0e0"
        COLOR_PANEL = "#f0f2f5"
        _FILENAME_SUFFIX = ""
        CMAP_FIELD = "RdBu_r"
        FIELD_INK = "black"
        FIELD_STROKE = "white"
        _SERIES_SPAN = _SERIES_SPAN_LIGHT
    _publish(COLOR_FG=COLOR_FG, COLOR_GRID=COLOR_GRID, COLOR_PANEL=COLOR_PANEL,
             _FILENAME_SUFFIX=_FILENAME_SUFFIX, CMAP_FIELD=CMAP_FIELD,
             FIELD_INK=FIELD_INK, FIELD_STROKE=FIELD_STROKE)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.5,
            "grid.linestyle": "--",
            "figure.figsize": (10, 6),
            "figure.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


# Figures kept as raster (lossless WebP) because SVG would be strictly heavier,
# for two reasons:
#   * raster-backed (pcolormesh / specgram): the SVG would only wrap a base64
#     bitmap several times larger than the raster;
#   * dense time series (thousands of vertices): every sample becomes a path
#     coordinate, so the SVG runs 5.5-7.75x the raster (schroeder_decay
#     105->820 KB, calibration_stability 99->759 KB, impulse_response
#     134->747 KB, measured against the earlier PNG encoding).
# Both clear the ~4.5x "SVG no longer wins" bar; everything else is a vector plot
# written as deterministic SVG (moderate 2.4x cases like sel_concept /
# ln_levels_example stay SVG -- below the bar, and vector crispness is worth it).
_RASTER_FIGURES = frozenset(
    {
        "spectrogram_example",
        # imshow raster: the STFT cells would bloat an SVG (and the repo's
        # figure policy keeps rasters out of SVG files).
        "calibrated_spectrogram",
        "excitation_signals",
        "schroeder_decay",
        "calibration_stability",
        "impulse_response",
        "numerical_propagation",
        "atmospheric_refraction",
        "airport_contour",
        "fdtd_simulation",
        # The launcher field and the rasterised metadiffuser mask are both
        # imshow of a cell grid: hundreds of thousands of cells, so the SVG
        # would be far heavier than the raster (as for fdtd_simulation).
        "fdtd_plane_wave_launch",
        "metadiffuser_meshed_panel",
        "elastic_halfspace_waves",
        "scholte_interface_wave",
        # Dense reflectogram: hundreds of image-source stems/markers make the
        # SVG far heavier than the raster (as for schroeder_decay above).
        "image_source_reflectogram",
        # Dense time series (tens of thousands of vertices), as for
        # calibration_stability above: the noise-floor traces of the
        # correlation normalizations make the SVG several times the raster.
        "correlation_normalizations",
    }
)


def save_figure(output_dir: str, filename: str, **kwargs: Any) -> None:
    """Translate, theme-suffix and save the current figure.

    Vector figures are written as SVG made byte-reproducible with a fixed
    ``svg.hashsalt`` (deterministic element ids), ``svg.fonttype = "none"``
    (text kept as ``<text>`` rather than freetype glyph outlines, so the
    output does not depend on the font build) and no date metadata -- so
    ``make graphs`` is stable and CI can diff it. The figures in
    :data:`_RASTER_FIGURES` stay raster (SVG would be heavier), written as
    lossless WebP: matplotlib renders the canvas to an in-memory PNG with the
    version-stamped ``Software`` chunk (and any date) stripped, and Pillow
    re-encodes those exact pixels with the exhaustive ``method=6`` search,
    whose output is fixed by the input, so the WebP bytes are reproducible
    across matplotlib builds and runs. In both cases ``filename`` may carry
    any extension; the real one is chosen here.
    """
    stem = os.path.splitext(filename)[0]
    audit_figure(stem)
    _translate_figure(plt.gcf())
    ext = "webp" if stem in _RASTER_FIGURES else "svg"
    path = os.path.join(output_dir, f"{stem}{_LANG_SUFFIX}{_FILENAME_SUFFIX}.{ext}")
    if ext == "svg":
        plt.rcParams["svg.hashsalt"] = "phonometry"
        plt.rcParams["svg.fonttype"] = "none"
        kwargs.setdefault("metadata", {"Date": None})
        plt.savefig(path, **kwargs)
        # svg.fonttype='none' keeps text selectable, but matplotlib writes
        # mathtext runs ('$...$') with a bare font-family ('DejaVu Sans', no
        # generic), so viewers lacking that font fall back to a serif while
        # plain <text> (full chain ending in sans-serif) stays sans. Normalise
        # every declaration to the generic 'sans-serif' so all text renders in
        # the viewer's native sans font, consistently and viewer-independently.
        # Exception: mathtext draws radicals and extensible delimiters with a
        # dedicated 'STIXSizeOneSym' run sized to match the glyph next to it
        # (e.g. the bar over \sqrt{...}); rewriting that run to sans-serif
        # swaps in the reader's sans glyph at the wrong size, so the radical
        # bar no longer lines up with the symbol it covers. Leave any
        # declaration whose family starts with 'STIX' untouched. The
        # lookahead sits before the optional whitespace (not inside
        # ``\s*``): matplotlib emits a space after the colon, and a
        # trailing ``\s*(?!'STIX)`` would backtrack to zero width and test
        # the lookahead against that leading space instead of the quote,
        # matching (and clobbering) the STIX declaration it was meant to
        # protect.
        import re as _re
        svg_text = pathlib.Path(path).read_text(encoding="utf-8")
        svg_text = _re.sub(
            r"font-family:(?!\s*'STIX)\s*[^;\"]+", "font-family: sans-serif", svg_text
        )
        pathlib.Path(path).write_text(svg_text, encoding="utf-8")
        return
    import io

    from PIL import Image

    # Render the raster figures at 300 dpi so they stay crisp on
    # high-density displays and when zoomed in the docs. Individual call
    # sites may still override this.
    kwargs.setdefault("dpi", 300)
    # Drop matplotlib's version-stamped Software chunk (and any date). Merge
    # instead of setdefault: a caller-supplied metadata dict must not silently
    # reintroduce the version-dependent chunks.
    kwargs["metadata"] = {"Software": None, "Date": None,
                          **kwargs.get("metadata", {})}
    with io.BytesIO() as buffer:
        plt.savefig(buffer, format="png", **kwargs)
        buffer.seek(0)
        with Image.open(buffer) as image:
            # Lossless WebP: identical pixels to the rendered canvas at roughly
            # half the optimized-PNG size (flat-color plot areas compress better
            # lossless than lossy). method=6 is the slowest, most exhaustive
            # encoder search; its output is a pure function of the input pixels,
            # so the committed bytes are byte-stable across runs.
            image.save(path, "WEBP", lossless=True, quality=100, method=6)



def apply_axis_styling(ax: Any, title: str, xlim: tuple[float, float] | None = None, ylim: tuple[float, float] | None = None) -> None:
    """Apply consistent styling to plots."""
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel(LABEL_LEVEL_DB)
    ax.grid(which="major", color=COLOR_GRID, linestyle="-")
    ax.grid(which="minor", color=COLOR_GRID, linestyle=":", alpha=0.4)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    # Standard Octave Ticks
    xticks = [16, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    xticklabels = ["16", "31.5", "63", "125", "250", "500", "1k", "2k", "4k", "8k", "16k"]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)


def plot_psd(ax: Any, x: np.ndarray, fs: int, label: str = "Raw Signal PSD", color: str = "gray", alpha: float = 0.3) -> None:
    """Calculate and plot the Power Spectral Density of the raw signal."""
    # Use Welch's method for a smooth PSD estimate
    f, Pxx = scipy_signal.welch(x, fs, nperseg=4096)
    
    # Convert to dB (relative to max to match SPL scale roughly or just show shape)
    # Since SPL is calibrated differently, we just want to show the 'shape' of the spectrum
    # in the background. We can normalize Pxx to match the peak of the octave bands roughly
    # or just plot it as is if we had calibrated units.
    # Here we'll just plot relative dB
    
    # Avoid log(0)
    Pxx_db = 10 * np.log10(Pxx + 1e-12)
    
    # Normalize PSD peak to 0 dB for visualization shape, then shift down? 
    # Or better: don't normalize, just plot. But PSD density vs Octave Band Power (integrated) 
    # are different units (dB/Hz vs dB).
    # So we will plot it on a secondary Y axis or just scaled to fit nicely in background.
    
    # Let's shift it so its mean roughly aligns with the mean of the SPL for visualization
    # This is purely for qualitative comparison of "where the energy is".
    
    ax.semilogx(f, Pxx_db, color=color, alpha=alpha, linewidth=1, label=label, zorder=0)



def measure_weighting_response(
    fs: int, curve: str, freqs: "np.ndarray | None" = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Measure a weighting curve the way the docs figure plots it: impulse
    response through the real filter path, evaluated with ``freqz`` (DTFT
    of the measured response).

    The impulse is placed at the CENTER of the buffer: the high-accuracy
    weighting path resamples with polyphase FIRs, and an impulse at sample 0
    loses the anti-causal half of the interpolation kernel (edge truncation),
    which once shipped a figure with curves ~2.4 dB low. Centering only adds
    linear phase, which does not affect the magnitude.

    :param fs: Sample rate in Hz.
    :param curve: 'A', 'B', 'C', 'D', 'G', 'AU' or 'Z'.
    :param freqs: Optional exact frequencies to evaluate; defaults to a
        dense 8192-point grid for plotting.
    :return: Tuple (frequencies, magnitude in dB).
    """
    from phonometry import weighting_filter

    impulse = np.zeros(fs)
    impulse[fs // 2] = 1.0
    weighted = weighting_filter(impulse, fs, curve=curve)

    worn = 8192 if freqs is None else np.asarray(freqs, dtype=float)
    w, h = scipy_signal.freqz(weighted, [1], worN=worn, fs=fs)
    mag_db = 20 * np.log10(np.abs(h) + 1e-9)
    if freqs is None:
        # Drop the 0 Hz point: it cannot be drawn on a log frequency axis.
        return w[1:], mag_db[1:]
    return w, mag_db
# ---------------------------------------------------------------------------
# Building & structure-borne result figures (WP: guide figure coverage).
# Each figure is a realistic user result: example data -> real function ->
# result object, drawn the way the result's own .plot() presents it.
# ---------------------------------------------------------------------------
_THIRD_OCTAVE_16 = [100, 125, 160, 200, 250, 315, 400, 500,
                    630, 800, 1000, 1250, 1600, 2000, 2500, 3150]


def _band_index_axis(ax: Any, freqs: Sequence[float] | np.ndarray,
                     fontsize: int = 8) -> np.ndarray:
    """Rotated nominal band labels on an index axis (band-data figures)."""
    x = np.arange(len(freqs))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b:g}" for b in np.asarray(freqs)],
                       rotation=45, fontsize=fontsize)
    ax.set_xlabel(LABEL_FREQ_HZ)
    return x
