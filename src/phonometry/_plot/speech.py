#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Plot renderers for the speech domain (lazy imports from result .plot())."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .common import (
    _C_PRIMARY,
    _C_PRIMARY_LIGHT,
    _C_REFERENCE,
    _C_SECONDARY,
    _C_TERTIARY,
    _LEGEND_UPPER_RIGHT,
    _STI_BAND_CENTERS,
    _band_axis,
    _new_axes,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..speech.objective_intelligibility import STOIResult
    from ..speech.sii import SIIProcedure, SIIResult, StandardSpeechSpectrum
    from ..speech.sti import STIResult

#: Spanish translations of the fixed strings rendered by the speech
#: ``.plot()`` renderers, keyed by their verbatim English text. ``_t``
#: returns the English key unchanged for any language other than ``"es"``,
#: so the English output is byte-for-byte identical to the pre-i18n
#: renderers.
_STRINGS: dict[str, str] = {
    "Frequency [Hz]": "Frecuencia [Hz]",
    "Band": "Banda",
    "Modulation transfer index MTI":
        "Índice de transferencia de modulación MTI",
    "One-third-octave band [Hz]": "Banda de tercio de octava [Hz]",
    "Mean intermediate correlation": "Correlación intermedia media",
    "Spectral correlation $d_m$": "Correlación espectral $d_m$",
    "Analysis segment": "Segmento de análisis",
    "Band audibility": "Audibilidad de banda",
    "Band audibility $A_i$": "Audibilidad de banda $A_i$",
    "Importance-weighted $I_i A_i$ (scaled)":
        "$I_i A_i$ ponderada por importancia (escalada)",
    "IEC 60268-16 STI = {sti}  (rating {rating})":
        "IEC 60268-16 STI = {sti}  (calificación {rating})",
    "{name} = {v}": "{name} = {v}",
    "ANSI S3.5 SII = {sii}": "ANSI S3.5 SII = {sii}",
    "Speech spectrum level [dB SPL]": "Nivel del espectro de voz [dB SPL]",
    "ANSI S3.5-1997 standard speech spectrum":
        "ANSI S3.5-1997 espectro de voz estándar",
    "ANSI S3.5-1997 band-importance function":
        "ANSI S3.5-1997 función de importancia de banda",
    "Band importance $I_i$": "Importancia de banda $I_i$",
    "Critical band (21)": "Banda crítica (21)",
    "Equally contributing (17)": "Contribución equitativa (17)",
    "One-third octave (18)": "Tercio de octava (18)",
    "Octave (6)": "Octava (6)",
    "Normal": "Normal",
    "Raised": "Elevada",
    "Loud": "Fuerte",
    "Shout": "Grito",
}


def _t(text: str, language: str = "en") -> str:
    """Localise a fixed string; English is returned verbatim (byte-identical)."""
    return _STRINGS.get(text, text) if language == "es" else text


def plot_sti(
    result: STIResult, ax: Axes | None = None, *, language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Per-band modulation transfer index bars with the STI and rating.

    :param result: A :class:`~phonometry.sti.STIResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to :meth:`~matplotlib.axes.Axes.bar`.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    mti = np.asarray(result.mti, dtype=np.float64)
    positions = np.arange(mti.size)
    kwargs.setdefault("color", _C_PRIMARY)
    ax.bar(positions, mti, **kwargs)
    banded = mti.size == len(_STI_BAND_CENTERS)
    if banded:
        _band_axis(ax, np.asarray(_STI_BAND_CENTERS), language=language)
        ax.set_xlabel(_t("Frequency [Hz]", language))
    else:
        ax.set_xticks(positions)
        ax.set_xlabel(_t("Band", language))
    ax.set_ylabel(_t("Modulation transfer index MTI", language))
    ax.set_ylim(0.0, 1.0)
    ax.set_title(_t("IEC 60268-16 STI = {sti}  (rating {rating})", language).format(
        sti=format_number(result.sti, language, decimals=2), rating=result.rating,
    ))
    ax.grid(True, axis="y", alpha=0.3)
    # localize_axes leaves the categorical band axis (a FuncFormatter) alone.
    localize_axes(ax, language)
    return ax


def plot_stoi(
    result: STOIResult, ax: Axes | None = None, *, language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Intermediate intelligibility that averages to the STOI/ESTOI index.

    For STOI the intermediate measure decomposes per one-third-octave band, so
    the mean correlation per band is drawn as bars; for ESTOI, whose index
    mixes the bands, the spectral correlation per analysis segment is drawn as
    a line.

    :param result: A :class:`~phonometry.speech.objective_intelligibility.STOIResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the underlying :meth:`bar`/:meth:`plot`.
    :return: The axes.
    """
    from .._i18n import decimal_comma, format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    name = "ESTOI" if result.extended else "STOI"
    banded = result.band_scores is not None
    if banded:
        freqs = np.asarray(result.band_frequencies, dtype=np.float64)
        scores = np.asarray(result.band_scores, dtype=np.float64)
        positions = np.arange(scores.size)
        kwargs.setdefault("color", _C_PRIMARY)
        ax.bar(positions, scores, **kwargs)
        ax.set_xticks(positions)
        # The STOI band centres are exact one-third-octave ratios from 150 Hz
        # (188.988..., 377.976...), so they are labelled at whole-hertz
        # resolution: the nominal band, not the generating float.
        ax.set_xticklabels([decimal_comma(f"{f:.0f}", language) for f in freqs],
                           rotation=45, ha="right")
        ax.set_xlabel(_t("One-third-octave band [Hz]", language))
        ax.set_ylabel(_t("Mean intermediate correlation", language))
    else:
        scores = np.asarray(result.segment_scores, dtype=np.float64)
        kwargs.setdefault("color", _C_PRIMARY)
        ax.plot(np.arange(scores.size), scores, **kwargs)
        ax.set_xlabel(_t("Analysis segment", language))
        ax.set_ylabel(_t("Spectral correlation $d_m$", language))
    # The intermediate correlations are cosine-similarity quantities in
    # [-1, 1] (unlike the [0, 1] ratios of plot_sti/plot_sii), so keep room
    # for anti-correlated bands rather than clipping them at zero.
    ax.set_ylim(-1.0, 1.0)
    ax.set_title(_t("{name} = {v}", language).format(
        name=name, v=format_number(result.value, language, decimals=3),
    ))
    ax.grid(True, axis="y", alpha=0.3)
    # localize_axes leaves the categorical band labels (a FuncFormatter) alone,
    # so the comma-localized labels set above survive.
    localize_axes(ax, language)
    return ax


def plot_sii(
    result: SIIResult, ax: Axes | None = None, *, language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Per-band audibility and its importance-weighted contribution to the SII.

    :param result: A :class:`~phonometry.sii.SIIResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the weighted-contribution :meth:`bar`.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    audibility = np.asarray(result.band_audibility, dtype=np.float64)
    contribution = audibility * np.asarray(result.band_importance, dtype=np.float64)
    positions = _band_axis(ax, freqs, language=language)
    ax.set_xlabel(_t("Frequency [Hz]", language))
    ax.bar(positions, audibility, color=_C_PRIMARY_LIGHT,
           label=_t("Band audibility $A_i$", language))
    kwargs.setdefault("color", _C_PRIMARY)
    # A fully masked speech signal (SII = 0) has an all-zero contribution;
    # keep the zero bars rather than dividing 0/0 into NaN.
    peak = float(contribution.max()) if contribution.size else 0.0
    scaled = contribution / peak if peak > 0.0 else contribution
    ax.bar(positions, scaled, width=0.5,
           label=_t("Importance-weighted $I_i A_i$ (scaled)", language), **kwargs)
    ax.set_ylabel(_t("Band audibility", language))
    ax.set_ylim(0.0, 1.0)
    ax.set_title(_t("ANSI S3.5 SII = {sii}", language).format(
        sii=format_number(result.sii, language, decimals=3),
    ))
    ax.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")
    ax.grid(True, axis="y", alpha=0.3)
    # localize_axes leaves the categorical band axis (a FuncFormatter) alone.
    localize_axes(ax, language)
    return ax


#: English legend label of each of the four ANSI S3.5-1997 band procedures,
#: keyed by the ``method=`` name. The label doubles as the ``_STRINGS`` key, so
#: ``_t`` returns it unchanged in English and its Spanish entry otherwise.
_SII_METHOD_LABELS: dict[str, str] = {
    "critical-band": "Critical band (21)",
    "equally-contributing": "Equally contributing (17)",
    "one-third-octave": "One-third octave (18)",
    "octave": "Octave (6)",
}


def plot_sii_procedure(
    result: SIIProcedure, ax: Axes | None = None, *, language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Band-importance function of one ANSI S3.5-1997 band procedure.

    Draws ``Ii`` as a step over the tabulated band limits on a continuous
    logarithmic frequency axis, so procedures with different band counts and
    widths overlay directly on the same axes.

    :param result: A :class:`~phonometry.speech.sii.SIIProcedure`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the step ``plot`` call.
    :return: The axes.
    """
    from .._i18n import localize_axes
    from .common import format_frequency_axis

    ax = ax if ax is not None else _new_axes()
    edges = np.asarray(result.band_edges, dtype=np.float64)
    importance = np.asarray(result.band_importance, dtype=np.float64)
    kwargs.setdefault("label", _t(_SII_METHOD_LABELS[result.method], language))
    # A post-step draws each band's Ii flat across its own width, which is what
    # makes a 6-band and a 21-band function comparable at a glance.
    ax.plot(edges, np.append(importance, importance[-1]), drawstyle="steps-post",
            **kwargs)
    ax.set_ylabel(_t("Band importance $I_i$", language))
    # Start at zero, and never let a fixed limit from an earlier procedure clip
    # a wider-band one: an octave band carries three times the importance of a
    # one-third-octave band at the same centre frequency.
    ax.set_ylim(
        0.0, max(float(ax.get_ylim()[1]), float(importance.max()) * 1.15)
    )
    ax.set_title(_t("ANSI S3.5-1997 band-importance function", language))
    # No explicit range: the tick set follows the axis limits, so overlaying a
    # second procedure that reaches lower or higher relabels the whole axis
    # instead of clipping the ticks to the first one drawn.
    format_frequency_axis(ax)
    ax.set_xlabel(_t("Frequency [Hz]", language))
    ax.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")
    ax.grid(True, alpha=0.3)
    localize_axes(ax, language)
    return ax


#: Distinct hues for the ordered vocal efforts (normal to shout); cycled when
#: fewer or more efforts are drawn.
_EFFORT_COLORS = (_C_PRIMARY, _C_TERTIARY, _C_SECONDARY, _C_REFERENCE)


def plot_standard_speech_spectrum(
    result: StandardSpeechSpectrum, ax: Axes | None = None, *,
    language: str = "en", **kwargs: Any,
) -> Axes:
    """Standard speech spectrum level by vocal effort (ANSI S3.5-1997 Table 3).

    Draws the standard speech spectrum level (dB SPL) over the 18
    one-third-octave bands (160 Hz to 8000 Hz) on a categorical band axis, one
    labelled line per vocal effort, so the whole spectrum lifting with vocal
    effort reads at a glance.

    :param result: A :class:`~phonometry.speech.sii.StandardSpeechSpectrum`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the per-effort ``plot`` calls.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    levels = np.asarray(result.levels, dtype=np.float64)
    positions = _band_axis(ax, freqs, xlabel=None, language=language)
    ax.set_xlabel(_t("One-third-octave band [Hz]", language))
    for i, effort in enumerate(result.vocal_efforts):
        # Explicit per-effort colour and label, but let any user kwargs win so
        # a supplied color=/label= overrides rather than collides.
        line_kwargs: dict[str, Any] = {
            "color": _EFFORT_COLORS[i % len(_EFFORT_COLORS)],
            "label": _t(effort.capitalize(), language),
        }
        line_kwargs.update(kwargs)
        ax.plot(positions, levels[i], "o-", **line_kwargs)
    ax.set_ylabel(_t("Speech spectrum level [dB SPL]", language))
    ax.set_title(_t("ANSI S3.5-1997 standard speech spectrum", language))
    ax.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")
    ax.grid(True, alpha=0.3)
    # localize_axes leaves the categorical band axis (a FuncFormatter) alone.
    localize_axes(ax, language)
    return ax
