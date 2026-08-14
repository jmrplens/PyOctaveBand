#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Plot renderers for the broadcast domain (lazy imports from result .plot())."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from .common import (
    _C_MUTED,
    _C_PRIMARY,
    _C_REFERENCE,
    _C_SECONDARY,
    _LEGEND_UPPER_RIGHT,
    _new_axes,
    format_frequency_axis,
    theme_fill,
    theme_line,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..broadcast.program_loudness import (
        KWeightingResponse,
        ProgramLoudnessResult,
    )

#: Spanish translations of the fixed strings rendered by the broadcast
#: ``.plot()`` renderer, keyed by their verbatim English text. ``_t``
#: returns the English key unchanged for any language other than ``"es"``,
#: so the English output is byte-for-byte identical to the pre-i18n
#: renderer.
_STRINGS: dict[str, str] = {
    "Momentary (400 ms)": "Momentánea (400 ms)",
    "Short-term (3 s)": "Corto plazo (3 s)",
    "Time [s]": "Tiempo [s]",
    "Loudness [LUFS]": "Sonoridad [LUFS]",
    "Programme loudness (EBU R 128)": "Sonoridad de programa (EBU R 128)",
    "Integrated": "Integrada",
    "Frequency [Hz]": "Frecuencia [Hz]",
    "Magnitude [dB]": "Magnitud [dB]",
    "K-weighting (combined)": "Ponderación K (combinada)",
    "Stage 1: spherical-head shelf": "Etapa 1: realce de cabeza esférica",
    "Stage 2: RLB high-pass": "Etapa 2: paso alto RLB",
    "K-weighting frequency response (ITU-R BS.1770)":
        "Respuesta en frecuencia de la ponderación K (UIT-R BS.1770)",
}


def _t(text: str, language: str = "en") -> str:
    """Localise a fixed string; English is returned verbatim (byte-identical)."""
    return _STRINGS.get(text, text) if language == "es" else text


def plot_program_loudness(
    result: ProgramLoudnessResult, ax: Axes | None = None, *,
    language: str = "en", **kwargs: Any
) -> Axes:
    """EBU Mode loudness over time (ITU-R BS.1770-5 / EBU R 128).

    Draws the momentary (400 ms) and short-term (3 s) loudness series, the
    integrated (programme) loudness as a horizontal line and the loudness
    range as a shaded band between its 10th and 95th percentile edges.

    :param result: A :class:`~phonometry.broadcast.ProgramLoudnessResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the short-term ``plot``.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    if math.isfinite(result.lra_low) and math.isfinite(result.lra_high):
        lra = format_number(result.loudness_range, language, decimals=1)
        # Opaque wash derived from the page: the report pipeline renders
        # through svglib, which drops alpha, so a translucent fill would come
        # out saturated and hide the curves.
        ax.axhspan(
            result.lra_low,
            result.lra_high,
            facecolor=theme_fill(_C_SECONDARY, ax),
            edgecolor="none",
            zorder=0,
            label=f"LRA {lra} LU",
        )
    if result.momentary.size:
        ax.plot(
            result.momentary_time,
            result.momentary,
            color=_C_MUTED,
            linewidth=1.0,
            label=_t("Momentary (400 ms)", language),
        )
    if result.short_term.size:
        kwargs.setdefault("color", _C_PRIMARY)
        kwargs.setdefault("linewidth", 2.0)
        kwargs.setdefault("label", _t("Short-term (3 s)", language))
        ax.plot(
            result.short_term_time,
            result.short_term,
            **kwargs,
        )
    if math.isfinite(result.integrated):
        integ = format_number(result.integrated, language, decimals=1)
        ax.axhline(
            result.integrated,
            color=_C_REFERENCE,
            linestyle="--",
            linewidth=1.6,
            label=f"{_t('Integrated', language)} {integ} LUFS",
        )
    finite = np.concatenate(
        [
            result.momentary[np.isfinite(result.momentary)],
            result.short_term[np.isfinite(result.short_term)],
        ]
    )
    if finite.size:
        low = float(np.min(finite))
        high = float(np.max(finite))
        ax.set_ylim(low - 5.0, high + 5.0)
    ax.set_xlabel(_t("Time [s]", language))
    ax.set_ylabel(_t("Loudness [LUFS]", language))
    ax.set_title(_t("Programme loudness (EBU R 128)", language))
    ax.grid(True, alpha=0.3)
    ax.legend(loc=_LEGEND_UPPER_RIGHT, fontsize=9)
    localize_axes(ax, language)
    return ax


def plot_k_weighting_response(
    result: KWeightingResponse, ax: Axes | None = None, *,
    language: str = "en", **kwargs: Any
) -> Axes:
    """K-weighting magnitude frequency response (ITU-R BS.1770-5 Annex 1).

    Draws the combined K-weighting magnitude (dB) on a logarithmic frequency
    axis, with the two stages (the +4 dB spherical-head shelf and the RLB
    high-pass) as light companion curves.

    :param result: A :class:`~phonometry.broadcast.KWeightingResponse`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the combined-curve ``plot`` call.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    ax.plot(
        freqs, np.asarray(result.shelf_db, dtype=np.float64),
        color=_C_MUTED, linewidth=1.0, linestyle="--",
        label=_t("Stage 1: spherical-head shelf", language),
    )
    ax.plot(
        freqs, np.asarray(result.highpass_db, dtype=np.float64),
        color=_C_SECONDARY, linewidth=1.0, linestyle=":",
        label=_t("Stage 2: RLB high-pass", language),
    )
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("linewidth", 2.0)
    kwargs.setdefault("label", _t("K-weighting (combined)", language))
    ax.plot(freqs, np.asarray(result.magnitude_db, dtype=np.float64), **kwargs)
    # The +4 dB shelf plateau is the reference the whole response is read
    # against, so it sits behind the curves -- held back by colour rather than
    # by opacity, which on a dark page holds a line back all the way to
    # invisible.
    ax.axhline(4.0, color=theme_line(_C_REFERENCE, ax, quiet=0.6),
               linestyle="-", linewidth=0.8)
    format_frequency_axis(ax, float(freqs.min()), float(freqs.max()))
    ax.set_xlabel(_t("Frequency [Hz]", language))
    ax.set_ylabel(_t("Magnitude [dB]", language))
    ax.set_title(_t("K-weighting frequency response (ITU-R BS.1770)", language))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    localize_axes(ax, language)
    return ax
