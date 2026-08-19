#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Plot renderers for the psychoacoustics domain (lazy imports from result .plot())."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .common import (
    _C_EDGE,
    _C_MUTED,
    _C_PRIMARY,
    _C_PRIMARY_LIGHT,
    _C_QUATERNARY,
    _C_REFERENCE,
    _C_SECONDARY,
    _C_TERTIARY,
    _new_axes,
    _new_axes_column,
    format_frequency_axis,
    theme_fill,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..psychoacoustics.loudness.contours import EqualLoudnessContours
    from ..psychoacoustics.loudness.ecma import EcmaLoudness
    from ..psychoacoustics.loudness.moore_glasberg import MooreGlasbergLoudness
    from ..psychoacoustics.loudness.moore_glasberg_time import (
        MooreGlasbergTimeVaryingLoudness,
    )
    from ..psychoacoustics.loudness.zwicker import ZwickerLoudness
    from ..psychoacoustics.quality.annoyance import PsychoacousticAnnoyanceResult
    from ..psychoacoustics.quality.fluctuation_strength import FluctuationStrengthResult
    from ..psychoacoustics.quality.fluctuation_strength_ecma import (
        EcmaFluctuationStrength,
    )
    from ..psychoacoustics.quality.roughness_ecma import EcmaRoughness
    from ..psychoacoustics.quality.tonality import ToneAssessment
    from ..psychoacoustics.quality.tonality_ecma import EcmaTonality
    from ..psychoacoustics.quality.tone_audibility import ToneAudibilityResult

#: Critical-band rate axis label of the Zwicker/DIN 45631 family (Bark scale).
_AXIS_BARK = "Critical-band rate $z$ [Bark]"
#: Critical-band rate axis label of the ECMA-418-2 (Sottek Hearing Model) family.
_AXIS_BARK_HMS = r"Critical-band rate $z$ [$\mathrm{Bark}_{\mathrm{HMS}}$]"
#: Time axis label shared by every time-varying psychoacoustic renderer.
_AXIS_TIME = "Time [s]"
#: Frequency axis label shared by the spectral psychoacoustic renderers.
_AXIS_FREQUENCY = "Frequency [Hz]"

#: Spanish translations of the fixed strings rendered by the
#: psychoacoustics ``.plot()`` renderers, keyed by their verbatim English
#: text. ``_t`` returns the English key unchanged for any language other
#: than ``"es"``, so the English output is byte-for-byte identical to the
#: pre-i18n renderers.
_STRINGS: dict[str, str] = {
    # Every prime below is composed as ``^{\prime}``, never typed as U+2032:
    # the character is an ordinary glyph of the maths run, so its ink stops at
    # cap height and the overhang of an italic "f" runs into it; ``^{\prime}``
    # is raised as a superscript, above the ascender, and clears it.
    _AXIS_BARK: "Razón de banda crítica $z$ [Bark]",
    _AXIS_BARK_HMS: r"Razón de banda crítica $z$ [$\mathrm{Bark}_{\mathrm{HMS}}$]",
    r"Specific loudness $N^{\prime}$ [sone/Bark]":
        r"Sonoridad específica $N^{\prime}$ [sonios/Bark]",
    r"Specific loudness $N^{\prime}$ [$\mathrm{sone}_{\mathrm{HMS}}$/$\mathrm{Bark}_{\mathrm{HMS}}$]":
        r"Sonoridad específica $N^{\prime}$ [$\mathrm{sonios}_{\mathrm{HMS}}$/$\mathrm{Bark}_{\mathrm{HMS}}$]",
    r"Specific loudness $N^{\prime}$ [sone/Cam]":
        r"Sonoridad específica $N^{\prime}$ [sonios/Cam]",
    "ERB number [Cam]": "Número ERB [Cam]",
    _AXIS_TIME: "Tiempo [s]",
    "Loudness $N$ [sone]": "Sonoridad $N$ [sonios]",
    r"Loudness $N$ [$\mathrm{sone}_{\mathrm{HMS}}$]":
        r"Sonoridad $N$ [$\mathrm{sonios}_{\mathrm{HMS}}$]",
    "Loudness [sone]": "Sonoridad [sonios]",
    "Short-term loudness": "Sonoridad a corto plazo",
    "Long-term loudness": "Sonoridad a largo plazo",
    r"Specific tonality $T^{\prime}$ [$\mathrm{tu}_{\mathrm{HMS}}$]":
        r"Tonalidad específica $T^{\prime}$ [$\mathrm{tu}_{\mathrm{HMS}}$]",
    r"Tonality $T$ [$\mathrm{tu}_{\mathrm{HMS}}$]":
        r"Tonalidad $T$ [$\mathrm{tu}_{\mathrm{HMS}}$]",
    "Roughness $R$ [asper]": "Aspereza $R$ [asper]",
    r"Fluctuation strength $F$ [$\mathrm{vacil}_{\mathrm{HMS}}$]":
        r"Intensidad de fluctuación $F$ [$\mathrm{vacil}_{\mathrm{HMS}}$]",
    r"Specific fluctuation strength $f^{\prime}(z)$ [vacil/Bark]":
        r"Intensidad de fluctuación específica $f^{\prime}(z)$ [vacil/Bark]",
    "Value": "Valor",
    "Tone frequency [Hz]": "Frecuencia del tono [Hz]",
    r"Audibility $\Delta L$ [dB]": r"Audibilidad $\Delta L$ [dB]",
    r"threshold $\Delta L=0$ dB": r"umbral $\Delta L=0$ dB",
    "ISO/PAS 20065 tonal audibility": "Audibilidad tonal ISO/PAS 20065",
    "ISO 532-1 loudness $N$ = {n} sone ({ln} phon)": "ISO 532-1 sonoridad $N$ = {n} sonios ({ln} fonios)",
    r"ECMA-418-2 loudness $N$ = {n} $\mathrm{{sone}}_{{\mathrm{{HMS}}}}$":
        r"ECMA-418-2 sonoridad $N$ = {n} $\mathrm{{sonios}}_{{\mathrm{{HMS}}}}$",
    "ISO 532-2 loudness $N$ = {n} sone ({ln} phon)": "ISO 532-2 sonoridad $N$ = {n} sonios ({ln} fonios)",
    "ISO 532-3 peak long-term loudness $N$ = {n} sone ({ln} phon)": "ISO 532-3 sonoridad a largo plazo máxima $N$ = {n} sonios ({ln} fonios)",
    r"ECMA-418-2 tonality $T$ = {t} $\mathrm{{tu}}_{{\mathrm{{HMS}}}}$":
        r"ECMA-418-2 tonalidad $T$ = {t} $\mathrm{{tu}}_{{\mathrm{{HMS}}}}$",
    "ECMA-418-2 roughness $R$ = {r} asper": "ECMA-418-2 aspereza $R$ = {r} asper",
    r"ECMA-418-2 fluctuation strength $F$ = {f} $\mathrm{{vacil}}_{{\mathrm{{HMS}}}}$":
        r"ECMA-418-2 intensidad de fluctuación $F$ = {f} $\mathrm{{vacil}}_{{\mathrm{{HMS}}}}$",
    "Fluctuation strength $F$ = {f} vacil": "Intensidad de fluctuación $F$ = {f} vacil",
    "Psychoacoustic annoyance PA = {pa} ($N_5$ = {n5} sone)": "Molestia psicoacústica PA = {pa} ($N_5$ = {n5} sonios)",
    r"decisive $\Delta L$ = {da} dB @ {df} Hz": r"decisiva $\Delta L$ = {da} dB @ {df} Hz",
    r"tone level $L_{p\mathrm{t}}$": r"nivel del tono $L_{p\mathrm{t}}$",
    r"masking noise $L_{p\mathrm{n}}$ (critical band)": r"ruido enmascarante $L_{p\mathrm{n}}$ (banda crítica)",
    r"decisive $\Delta L_{{\mathrm{{ta}}}}$ = {da} dB @ {df} Hz": r"decisiva $\Delta L_{{\mathrm{{ta}}}}$ = {da} dB @ {df} Hz",
    "Sound pressure level [dB]": "Nivel de presión acústica [dB]",
    "ISO 1996-2 tone-audibility analysis": "Análisis de audibilidad tonal ISO 1996-2",
    "Tone-to-noise ratio TNR [dB]": "Relación tono/ruido TNR [dB]",
    "Prominence ratio PR [dB]": "Relación de prominencia PR [dB]",
    "prominence criterion": "criterio de prominencia",
    "assessed tone": "tono evaluado",
    "prominent": "prominente",
    "not prominent": "no prominente",
    "margin {m} dB": "margen {m} dB",
    "ECMA-418-1 {name} = {v} dB at {f} Hz: {verdict}":
        "ECMA-418-1 {name} = {v} dB a {f} Hz: {verdict}",
    _AXIS_FREQUENCY: "Frecuencia [Hz]",
    "Sound pressure level [dB re 20 µPa]": "Nivel de presión acústica [dB re 20 µPa]",
    r"Hearing threshold $T_f$": r"Umbral de audición $T_f$",
    "ISO 226:2023 equal-loudness contours": "Curvas isofónicas ISO 226:2023",
    "{p} phon": "{p} fonios",
    "ISO 226:2023 defines 20 to 90 phon; above 80 phon the contour is defined only up to 4 kHz.":
        "ISO 226:2023 define de 20 a 90 fonios; por encima de 80 fonios la curva solo se define hasta 4 kHz.",
}


def _t(text: str, language: str = "en") -> str:
    """Localise a fixed string; English is returned verbatim (byte-identical)."""
    return _STRINGS.get(text, text) if language == "es" else text


def plot_zwicker_loudness(
    result: ZwickerLoudness, ax: Axes | None = None, *, language: str = "en",
    **kwargs: Any,
) -> Axes | np.ndarray:
    """Specific loudness N'(z) over the Bark scale (ISO 532-1).

    When the result carries the time-varying loudness trace
    (``time`` / ``loudness_vs_time``) *and* ``ax`` is ``None``, a second
    panel with loudness vs time is added and an array of two axes is
    returned; otherwise (a stationary result, or an ``ax`` was supplied) a
    single axes is returned.

    :param result: A :class:`~phonometry.psychoacoustics.loudness.zwicker.ZwickerLoudness`.
    :param ax: Existing axes to draw on, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the specific-loudness line ``plot`` call.
    :return: The axes, or an array of two axes for time-varying input.
    """
    from .._i18n import format_number, localize_axes

    specific = np.asarray(result.specific, dtype=np.float64)
    bark = np.arange(1, specific.size + 1) * 0.1
    time_varying = (
        result.time is not None and result.loudness_vs_time is not None and ax is None
    )

    if time_varying:
        axes = _new_axes_column(2, figsize=(7.0, 6.0))
        ax_specific = cast("Axes", axes[0])
    else:
        ax_specific = ax if ax is not None else _new_axes()

    kwargs.setdefault("color", _C_PRIMARY)
    ax_specific.plot(bark, specific, **kwargs)
    ax_specific.fill_between(bark, specific, color=kwargs["color"], alpha=0.25)
    ax_specific.set_xlabel(_t(_AXIS_BARK, language))
    ax_specific.set_ylabel(_t(r"Specific loudness $N^{\prime}$ [sone/Bark]", language))
    ax_specific.set_xlim(0.0, bark[-1])
    ax_specific.set_ylim(bottom=0.0)
    ax_specific.set_title(_t("ISO 532-1 loudness $N$ = {n} sone ({ln} phon)", language).format(
        n=format_number(result.loudness, language, decimals=2),
        ln=format_number(result.loudness_level, language, decimals=1),
    ))
    ax_specific.grid(True, alpha=0.3)

    if not time_varying:
        localize_axes(ax_specific, language)
        return ax_specific

    ax_time = cast("Axes", axes[1])
    plot_zwicker_loudness_time(result, ax=ax_time, language=language)
    localize_axes(ax_specific, language)
    return axes


def plot_zwicker_loudness_time(
    result: ZwickerLoudness, ax: Axes | None = None, *, language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Loudness-versus-time function N(t) of a time-varying result (ISO 532-1).

    Draws the 500 Hz total-loudness trace (clause 6.5) with the N5/N10
    percentile levels marked; this is the "loudness time function in sones"
    clause 7 requires a time-varying loudness report to state. Also used as
    the second panel of :func:`plot_zwicker_loudness`.

    :param result: A time-varying
        :class:`~phonometry.psychoacoustics.loudness.zwicker.ZwickerLoudness` (with
        ``time`` / ``loudness_vs_time``).
    :param ax: Existing axes to draw on, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the loudness-trace ``plot`` call.
    :return: The axes.
    :raises ValueError: If the result carries no loudness-vs-time trace.
    """
    from .._i18n import format_number, localize_axes

    if result.time is None or result.loudness_vs_time is None:
        raise ValueError(
            "plot_zwicker_loudness_time() needs a time-varying result with "
            "'time' and 'loudness_vs_time'."
        )
    ax_time = ax if ax is not None else _new_axes()
    time = np.asarray(result.time, dtype=np.float64)
    lvt = np.asarray(result.loudness_vs_time, dtype=np.float64)
    kwargs.setdefault("color", _C_TERTIARY)
    kwargs.setdefault("label", "$N(t)$")
    ax_time.plot(time, lvt, **kwargs)
    if result.n5 is not None:
        ax_time.axhline(
            result.n5, color=_C_REFERENCE, ls="--", lw=1,
            label="$N_5$=" + format_number(result.n5, language, decimals=2),
        )
    if result.n10 is not None:
        ax_time.axhline(
            result.n10, color=_C_SECONDARY, ls=":", lw=1,
            label="$N_{10}$=" + format_number(result.n10, language, decimals=2),
        )
    ax_time.set_xlabel(_t(_AXIS_TIME, language))
    ax_time.set_ylabel(_t("Loudness $N$ [sone]", language))
    ax_time.set_ylim(bottom=0.0)
    ax_time.grid(True, alpha=0.3)
    ax_time.legend(loc="best", fontsize="small")
    localize_axes(ax_time, language)
    return ax_time


def plot_ecma_loudness(
    result: EcmaLoudness, ax: Axes | None = None, *, language: str = "en",
    **kwargs: Any,
) -> Axes | np.ndarray:
    """Average specific loudness N'(z) and time-dependent loudness N(l).

    When ``ax`` is ``None`` a two-panel figure is drawn (specific loudness
    over the critical-band-rate scale and loudness vs time) and an array of
    two axes is returned; when ``ax`` is supplied only the specific-loudness
    panel is drawn on it and that single axes is returned.

    :param result: An :class:`~phonometry.psychoacoustics.loudness.ecma.EcmaLoudness`.
    :param ax: Existing axes to draw on, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the specific-loudness line ``plot`` call.
    :return: The axes, or an array of two axes.
    """
    from .._i18n import format_number, localize_axes

    specific = np.asarray(result.specific_loudness, dtype=np.float64)
    bark = np.asarray(result.bark, dtype=np.float64)
    two_panel = ax is None

    if two_panel:
        axes = _new_axes_column(2, figsize=(7.0, 6.0))
        ax_specific = cast("Axes", axes[0])
    else:
        ax_specific = cast("Axes", ax)

    kwargs.setdefault("color", _C_PRIMARY)
    ax_specific.plot(bark, specific, **kwargs)
    ax_specific.fill_between(bark, specific, color=kwargs["color"], alpha=0.25)
    ax_specific.set_xlabel(_t(_AXIS_BARK_HMS, language))
    ax_specific.set_ylabel(_t(
        r"Specific loudness $N^{\prime}$ [$\mathrm{sone}_{\mathrm{HMS}}$/$\mathrm{Bark}_{\mathrm{HMS}}$]",
        language,
    ))
    ax_specific.set_xlim(0.0, bark[-1])
    ax_specific.set_ylim(bottom=0.0)
    ax_specific.set_title(_t(
        r"ECMA-418-2 loudness $N$ = {n} $\mathrm{{sone}}_{{\mathrm{{HMS}}}}$", language,
    ).format(
        n=format_number(result.loudness, language, decimals=2),
    ))
    ax_specific.grid(True, alpha=0.3)

    if not two_panel:
        localize_axes(ax_specific, language)
        return ax_specific

    ax_time = cast("Axes", axes[1])
    time = np.asarray(result.time, dtype=np.float64)
    lvt = np.asarray(result.loudness_vs_time, dtype=np.float64)
    ax_time.plot(time, lvt, color=_C_TERTIARY, label="$N(l)$")
    ax_time.set_xlabel(_t(_AXIS_TIME, language))
    ax_time.set_ylabel(_t(r"Loudness $N$ [$\mathrm{sone}_{\mathrm{HMS}}$]", language))
    ax_time.set_ylim(bottom=0.0)
    ax_time.grid(True, alpha=0.3)
    ax_time.legend(loc="best", fontsize="small")
    localize_axes(ax_specific, language)
    localize_axes(ax_time, language)
    return axes


def plot_moore_glasberg_loudness(
    result: MooreGlasbergLoudness, ax: Axes | None = None, *, language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Specific loudness N'(i) over the ERB-number (Cam) scale (ISO 532-2).

    :param result: A
        :class:`~phonometry.psychoacoustics.loudness.moore_glasberg.MooreGlasbergLoudness`.
    :param ax: Existing axes to draw on, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the specific-loudness line ``plot`` call.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    specific = np.asarray(result.specific, dtype=np.float64)
    erb_number = np.asarray(result.erb_number, dtype=np.float64)
    ax = ax if ax is not None else _new_axes()

    kwargs.setdefault("color", _C_PRIMARY)
    ax.plot(erb_number, specific, **kwargs)
    ax.fill_between(erb_number, specific, color=kwargs["color"], alpha=0.25)
    ax.set_xlabel(_t("ERB number [Cam]", language))
    ax.set_ylabel(_t(r"Specific loudness $N^{\prime}$ [sone/Cam]", language))
    ax.set_xlim(erb_number[0], erb_number[-1])
    ax.set_ylim(bottom=0.0)
    ax.set_title(_t("ISO 532-2 loudness $N$ = {n} sone ({ln} phon)", language).format(
        n=format_number(result.loudness, language, decimals=2),
        ln=format_number(result.loudness_level, language, decimals=1),
    ))
    ax.grid(True, alpha=0.3)
    localize_axes(ax, language)
    return ax


def plot_moore_glasberg_time_loudness(
    result: MooreGlasbergTimeVaryingLoudness, ax: Axes | None = None, *,
    language: str = "en", **kwargs: Any,
) -> Axes:
    """Short-term and long-term loudness against time (ISO 532-3).

    :param result: A
        :class:`~phonometry.psychoacoustics.loudness.moore_glasberg_time.MooreGlasbergTimeVaryingLoudness`.
    :param ax: Existing axes to draw on, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the long-term-loudness line ``plot`` call.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    time = np.asarray(result.time, dtype=np.float64)
    stl = np.asarray(result.short_term_loudness, dtype=np.float64)
    ltl = np.asarray(result.long_term_loudness, dtype=np.float64)
    ax = ax if ax is not None else _new_axes()

    ax.plot(time, stl, color=_C_PRIMARY_LIGHT, lw=1.0,
            label=_t("Short-term loudness", language))
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("lw", 1.8)
    kwargs.setdefault("label", _t("Long-term loudness", language))
    ax.plot(time, ltl, **kwargs)
    ax.axhline(result.n_max, color=_C_REFERENCE, ls="--", lw=1.0, alpha=0.7)
    ax.set_xlabel(_t(_AXIS_TIME, language))
    ax.set_ylabel(_t("Loudness [sone]", language))
    if time.size:
        ax.set_xlim(time[0], time[-1])
    ax.set_ylim(bottom=0.0)
    ax.set_title(_t("ISO 532-3 peak long-term loudness $N$ = {n} sone ({ln} phon)", language).format(
        n=format_number(result.n_max, language, decimals=2),
        ln=format_number(result.loudness_level_max, language, decimals=1),
    ))
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, alpha=0.3)
    localize_axes(ax, language)
    return ax


def plot_ecma_tonality(
    result: EcmaTonality, ax: Axes | None = None, *, language: str = "en",
    **kwargs: Any,
) -> Axes | np.ndarray:
    """Average specific tonality T'(z) and time-dependent tonality T(l).

    When ``ax`` is ``None`` a two-panel figure is drawn (specific tonality over
    the critical-band-rate scale and tonality vs time) and an array of two axes
    is returned; when ``ax`` is supplied only the specific-tonality panel is
    drawn on it and that single axes is returned.

    :param result: An :class:`~phonometry.psychoacoustics.quality.tonality_ecma.EcmaTonality`.
    :param ax: Existing axes to draw on, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the specific-tonality line ``plot`` call.
    :return: The axes, or an array of two axes.
    """
    from .._i18n import format_number, localize_axes

    specific = np.asarray(result.specific_tonality, dtype=np.float64)
    bark = np.asarray(result.bark, dtype=np.float64)
    two_panel = ax is None

    if two_panel:
        axes = _new_axes_column(2, figsize=(7.0, 6.0))
        ax_specific = cast("Axes", axes[0])
    else:
        ax_specific = cast("Axes", ax)

    # Tonality's per-metric identity color is red across the documentation
    # figures (roughness is brown); kept literal on purpose, see the module
    # color-constant note.
    kwargs.setdefault("color", "#d62728")
    ax_specific.plot(bark, specific, **kwargs)
    ax_specific.fill_between(bark, specific, color=kwargs["color"], alpha=0.25)
    ax_specific.set_xlabel(_t(_AXIS_BARK_HMS, language))
    ax_specific.set_ylabel(
        _t(r"Specific tonality $T^{\prime}$ [$\mathrm{tu}_{\mathrm{HMS}}$]", language)
    )
    ax_specific.set_xlim(0.0, bark[-1])
    ax_specific.set_ylim(bottom=0.0)
    ax_specific.set_title(_t(
        r"ECMA-418-2 tonality $T$ = {t} $\mathrm{{tu}}_{{\mathrm{{HMS}}}}$", language,
    ).format(
        t=format_number(result.tonality, language, decimals=2),
    ))
    ax_specific.grid(True, alpha=0.3)

    if not two_panel:
        localize_axes(ax_specific, language)
        return ax_specific

    ax_time = cast("Axes", axes[1])
    time = np.asarray(result.time, dtype=np.float64)
    tvt = np.asarray(result.tonality_vs_time, dtype=np.float64)
    ax_time.plot(time, tvt, color=_C_QUATERNARY, label="$T(l)$")
    ax_time.set_xlabel(_t(_AXIS_TIME, language))
    ax_time.set_ylabel(_t(r"Tonality $T$ [$\mathrm{tu}_{\mathrm{HMS}}$]", language))
    ax_time.set_ylim(bottom=0.0)
    ax_time.grid(True, alpha=0.3)
    ax_time.legend(loc="best", fontsize="small")
    localize_axes(ax_specific, language)
    localize_axes(ax_time, language)
    return axes


def _plot_hms_time_and_heatmap(
    time: np.ndarray,
    vs_time: np.ndarray,
    spec_vs_time: np.ndarray,
    bark: np.ndarray,
    ax: Axes | None,
    color: str,
    ylabel: str,
    title: str,
    heat_label: str,
    kwargs: dict[str, Any],
    language: str = "en",
) -> Axes | np.ndarray:
    """Shared renderer for the HMS time-trace + specific-value heatmaps.

    Used by the ECMA-418-2 roughness and fluctuation-strength results: when
    ``ax`` is ``None`` a two-panel figure (time trace + heatmap over the
    critical-band-rate scale) is drawn and an array of two axes is returned;
    otherwise only the time trace is drawn on ``ax`` and it is returned.
    """
    from .._i18n import localize_axes

    two_panel = ax is None
    if two_panel:
        axes = _new_axes_column(2, figsize=(7.0, 6.0))
        ax_time = cast("Axes", axes[0])
    else:
        ax_time = cast("Axes", ax)

    if "c" not in kwargs:  # matplotlib alias; injecting "color" too would raise
        kwargs.setdefault("color", color)
    (line,) = ax_time.plot(time, vs_time, **kwargs)
    ax_time.fill_between(time, vs_time, color=line.get_color(), alpha=0.25)
    ax_time.set_xlabel(_t(_AXIS_TIME, language))
    ax_time.set_ylabel(ylabel)
    ax_time.set_ylim(bottom=0.0)
    ax_time.set_title(title)
    ax_time.grid(True, alpha=0.3)

    if not two_panel:
        localize_axes(ax_time, language)
        return ax_time

    ax_heat = cast("Axes", axes[1])
    if time.size >= 2 and spec_vs_time.size:
        mesh = ax_heat.pcolormesh(
            time, bark, spec_vs_time.T, cmap="magma", shading="auto"
        )
        ax_heat.figure.colorbar(mesh, ax=ax_heat, label=heat_label)
    ax_heat.set_xlabel(_t(_AXIS_TIME, language))
    ax_heat.set_ylabel(_t(_AXIS_BARK_HMS, language))
    localize_axes(ax_time, language)
    localize_axes(ax_heat, language)
    return axes


def plot_ecma_roughness(
    result: EcmaRoughness, ax: Axes | None = None, *, language: str = "en",
    **kwargs: Any,
) -> Axes | np.ndarray:
    """Time-dependent roughness R(l50) and a specific-roughness heatmap.

    When ``ax`` is ``None`` a two-panel figure is drawn (roughness vs time and
    a specific-roughness R'(l50, z) heatmap over the critical-band-rate scale)
    and an array of two axes is returned; when ``ax`` is supplied only the
    time-dependent roughness is drawn on it and that single axes is returned.

    :param result: An :class:`~phonometry.psychoacoustics.quality.roughness_ecma.EcmaRoughness`.
    :param ax: Existing axes to draw on, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the roughness-vs-time line ``plot`` call.
    :return: The axes, or an array of two axes.
    """
    from .._i18n import format_number

    # Roughness's per-metric identity color is brown across the documentation
    # figures (tonality is red); kept literal on purpose, see the module
    # color-constant note.
    return _plot_hms_time_and_heatmap(
        np.asarray(result.time, dtype=np.float64),
        np.asarray(result.roughness_vs_time, dtype=np.float64),
        np.asarray(result.specific_roughness_vs_time, dtype=np.float64),
        np.asarray(result.bark, dtype=np.float64),
        ax,
        "#8c564b",
        _t("Roughness $R$ [asper]", language),
        _t("ECMA-418-2 roughness $R$ = {r} asper", language).format(
            r=format_number(result.roughness, language, decimals=2),
        ),
        r"$R^{\prime}$ [asper/$\mathrm{Bark}_{\mathrm{HMS}}$]",
        kwargs,
        language,
    )


def plot_ecma_fluctuation_strength(
    result: EcmaFluctuationStrength, ax: Axes | None = None, *, language: str = "en",
    **kwargs: Any,
) -> Axes | np.ndarray:
    """Time-dependent fluctuation strength F(l50) and a specific heatmap.

    When ``ax`` is ``None`` a two-panel figure is drawn (fluctuation strength
    vs time and a specific-fluctuation-strength F'(l50, z) heatmap over the
    critical-band-rate scale) and an array of two axes is returned; when
    ``ax`` is supplied only the time-dependent fluctuation strength is drawn
    on it and that single axes is returned.

    :param result: An :class:`~phonometry.psychoacoustics.quality.
        fluctuation_strength_ecma.EcmaFluctuationStrength`.
    :param ax: Existing axes to draw on, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the time-trace line ``plot`` call.
    :return: The axes, or an array of two axes.
    """
    from .._i18n import format_number

    # Fluctuation strength's per-metric identity color is teal across the
    # documentation figures (roughness is brown, tonality red); kept literal
    # on purpose, see the module color-constant note.
    return _plot_hms_time_and_heatmap(
        np.asarray(result.time, dtype=np.float64),
        np.asarray(result.fluctuation_strength_vs_time, dtype=np.float64),
        np.asarray(
            result.specific_fluctuation_strength_vs_time, dtype=np.float64
        ),
        np.asarray(result.bark, dtype=np.float64),
        ax,
        "#17becf",
        _t(r"Fluctuation strength $F$ [$\mathrm{vacil}_{\mathrm{HMS}}$]", language),
        _t(
            r"ECMA-418-2 fluctuation strength $F$ = {f} "
            r"$\mathrm{{vacil}}_{{\mathrm{{HMS}}}}$",
            language,
        ).format(
            f=format_number(result.fluctuation_strength, language, decimals=2),
        ),
        r"$F^{\prime}$ [$\mathrm{vacil}_{\mathrm{HMS}}$/$\mathrm{Bark}_{\mathrm{HMS}}$]",
        kwargs,
        language,
    )


def plot_fluctuation_strength(
    result: FluctuationStrengthResult, ax: Axes | None = None, *,
    language: str = "en", **kwargs: Any,
) -> Axes:
    """Specific fluctuation strength ``f(z)`` against critical-band rate.

    Draws the per-filter specific fluctuation strength over the Bark scale,
    annotated with the overall ``F`` in vacil.

    :param result: A
        :class:`~phonometry.psychoacoustics.quality.fluctuation_strength.FluctuationStrengthResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the ``plot`` call.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    z = np.asarray(result.bark_axis, dtype=np.float64)
    spec = np.asarray(result.specific, dtype=np.float64)
    kwargs.setdefault("color", _C_PRIMARY)
    ax.plot(z, spec, **kwargs)
    ax.fill_between(z, spec, color=kwargs["color"], alpha=0.25)
    ax.set_xlabel(_t(_AXIS_BARK, language))
    ax.set_ylabel(
        _t(r"Specific fluctuation strength $f^{\prime}(z)$ [vacil/Bark]", language)
    )
    ax.set_title(_t("Fluctuation strength $F$ = {f} vacil", language).format(
        f=format_number(result.fluctuation_strength, language, decimals=2),
    ))
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    localize_axes(ax, language)
    return ax


def plot_psychoacoustic_annoyance(
    result: PsychoacousticAnnoyanceResult, ax: Axes | None = None, *,
    language: str = "en", **kwargs: Any,
) -> Axes:
    """Psychoacoustic annoyance with its ``wS`` and ``wFR`` term contributions.

    Draws the PA value alongside the two loudness-weighted terms so the
    sharpness and fluctuation/roughness contributions are visible.

    :param result: A :class:`~phonometry.psychoacoustics.quality.annoyance.
        PsychoacousticAnnoyanceResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the bar ``bar`` call.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    labels = ["PA", r"$w_S$", r"$w_{FR}$"]
    values = [result.annoyance, result.w_s, result.w_fr]
    colors = [_C_REFERENCE, _C_PRIMARY, _C_PRIMARY]
    positions = np.arange(len(labels))
    kwargs.setdefault("width", 0.6)
    kwargs.setdefault("color", colors)
    kwargs.setdefault("edgecolor", _C_EDGE)
    ax.bar(positions, values, **kwargs)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(_t("Value", language))
    ax.set_title(_t("Psychoacoustic annoyance PA = {pa} ($N_5$ = {n5} sone)", language).format(
        pa=format_number(result.annoyance, language, decimals=1),
        n5=format_number(result.n5, language, decimals=1),
    ))
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    # localize_axes leaves the categorical x-axis (a FuncFormatter) alone.
    localize_axes(ax, language)
    return ax


#: Range of interest for discrete tones (ECMA-418-1 clauses 11.5 / 12.6).
_TONE_RANGE_HZ: tuple[float, float] = (89.1, 11200.0)


def plot_tone_assessment(
    result: ToneAssessment, ax: Axes | None = None, *, language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Assessed tone against the ECMA-418-1 prominence criterion.

    Draws the frequency-dependent prominence criterion over the 89.1 Hz -
    11.2 kHz range of interest with the assessed tone plotted at its own
    frequency, and the margin between the measured ratio and the criterion
    marked; a tone on or above the curve is prominent.

    The producing method (tone-to-noise ratio, clause 11, or prominence
    ratio, clause 12) is recovered from ``criterion_db``: the two criteria
    differ everywhere in the range of interest (8 dB against 9 dB from 1 kHz
    up, and different low-frequency slopes below), so the closer of the two
    identifies the curve to draw and the quantity to label.

    :param result: A :class:`~phonometry.psychoacoustics.quality.tonality.ToneAssessment`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the assessed-tone marker ``plot`` call.
    :return: The axes.
    """
    from .._i18n import decimal_comma, format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    ft = float(result.frequency)
    ratio = float(result.ratio_db)
    criterion = float(result.criterion_db)

    def _tnr(f: np.ndarray) -> np.ndarray:
        return np.where(f < 1000.0, 8.0 + 8.33 * np.log10(1000.0 / f), 8.0)

    def _pr(f: np.ndarray) -> np.ndarray:
        return np.where(f < 1000.0, 9.0 + 10.0 * np.log10(1000.0 / f), 9.0)

    ft_arr = np.array([max(ft, _TONE_RANGE_HZ[0])])
    is_tnr = abs(criterion - float(_tnr(ft_arr)[0])) <= abs(
        criterion - float(_pr(ft_arr)[0])
    )
    name = "TNR" if is_tnr else "PR"
    ylabel = ("Tone-to-noise ratio TNR [dB]" if is_tnr
              else "Prominence ratio PR [dB]")

    grid = np.logspace(np.log10(_TONE_RANGE_HZ[0]), np.log10(_TONE_RANGE_HZ[1]), 400)
    curve = _tnr(grid) if is_tnr else _pr(grid)
    ax.plot(grid, curve, color=_C_REFERENCE, lw=1.8,
            label=_t("prominence criterion", language))
    ax.plot([ft, ft], [criterion, ratio], color=_C_MUTED, ls=":", lw=1.2)
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("linestyle", "none")
    kwargs.setdefault("markersize", 9)
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("label", _t("assessed tone", language))
    ax.plot([ft], [ratio], markeredgecolor=_C_EDGE, zorder=4,
            **kwargs)
    ax.annotate(
        _t("margin {m} dB", language).format(
            m=format_number(ratio - criterion, language, decimals=1),
        ),
        xy=(ft, 0.5 * (ratio + criterion)), xytext=(6, 0),
        textcoords="offset points", va="center", fontsize="small",
    )

    verdict = _t("prominent" if result.prominent else "not prominent", language)
    ax.set_title(_t("ECMA-418-1 {name} = {v} dB at {f} Hz: {verdict}", language).format(
        name=name, v=format_number(ratio, language, decimals=1),
        f=decimal_comma(f"{ft:g}", language), verdict=verdict,
    ))
    ax.set_xlabel(_t(_AXIS_FREQUENCY, language))
    ax.set_ylabel(_t(ylabel, language))
    format_frequency_axis(ax, *_TONE_RANGE_HZ)
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, which="major", alpha=0.3)
    ax.set_axisbelow(True)
    localize_axes(ax, language)
    return ax


def plot_tone_audibility(
    result: ToneAudibilityResult, ax: Axes | None = None, *, language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Per-tone audibility ``ΔL`` against tone frequency (ISO/PAS 20065).

    Draws one bar per tone with the decisive (most audible) tone emphasised and
    the ``ΔL = 0`` audibility threshold marked; tones above it are present.

    :param result: A :class:`~phonometry.psychoacoustics.quality.tone_audibility.ToneAudibilityResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the bar ``bar`` call.
    :return: The axes.
    """
    from .._i18n import decimal_comma, format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.tone_frequencies, dtype=np.float64)
    delta = np.asarray(result.audibilities, dtype=np.float64)
    positions = np.arange(freqs.size)
    decisive = int(np.argmax(delta))
    colors = [_C_PRIMARY] * freqs.size
    colors[decisive] = _C_REFERENCE

    kwargs.setdefault("width", 0.7)
    bars = ax.bar(positions, delta, color=colors, edgecolor=_C_EDGE, **kwargs)
    bars[decisive].set_label(_t(r"decisive $\Delta L$ = {da} dB @ {df} Hz", language).format(
        da=format_number(result.decisive_audibility, language, decimals=1),
        df=decimal_comma(f"{result.decisive_frequency:g}", language),
    ))
    ax.axhline(0.0, color=_C_MUTED, ls="--", lw=1.0,
               label=_t(r"threshold $\Delta L=0$ dB", language))
    ax.set_xticks(positions)
    ax.set_xticklabels([decimal_comma(f"{f:g}", language) for f in freqs],
                       rotation=45, ha="right")
    ax.set_xlabel(_t("Tone frequency [Hz]", language))
    ax.set_ylabel(_t(r"Audibility $\Delta L$ [dB]", language))
    ax.set_title(_t("ISO/PAS 20065 tonal audibility", language))
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    # localize_axes leaves the categorical x-axis (a FuncFormatter) alone, so the
    # comma-localized tick labels set above survive.
    localize_axes(ax, language)
    return ax


def plot_tone_audibility_levels(
    result: ToneAudibilityResult, ax: Axes | None = None, *, language: str = "en",
    **kwargs: Any,
) -> Axes:
    r"""Tone level and critical-band masking noise against frequency (ISO 1996-2).

    Draws, for every detected tone, its tone level ``Lpt`` as a stem above the
    critical-band masking-noise level ``Lpn`` (drawn as a horizontal segment
    spanning the tone's critical band ``[f1, f2]``); the vertical gap above the
    masking threshold is the audibility ``ΔL_ta``. The decisive (most audible)
    tone is emphasised and the frequency axis is continuous and logarithmic.

    :param result: A :class:`~phonometry.psychoacoustics.quality.tone_audibility.ToneAudibilityResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the tone-level marker ``plot`` call.
    :return: The axes.
    """
    from .._i18n import decimal_comma, format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.tone_frequencies, dtype=np.float64)
    lt = np.asarray(result.tone_levels, dtype=np.float64)
    lg = np.asarray(result.critical_band_levels, dtype=np.float64)
    f1 = np.asarray(result.lower_corners, dtype=np.float64)
    f2 = np.asarray(result.upper_corners, dtype=np.float64)
    delta = np.asarray(result.audibilities, dtype=np.float64)
    decisive = int(np.argmax(delta))

    ymin = float(min(lg.min(), lt.min())) - 4.0
    # Shade the decisive tone's critical band below its masking-noise level so
    # the audible gap reads at a glance. svglib drops alpha, so the fill is an
    # *opaque* wash with no edge and a low zorder (below the curves); it is
    # derived from the page, so the same call reads on the white report page
    # and on a dark-theme documentation figure.
    ax.fill_between(
        [f1[decisive], f2[decisive]], ymin, lg[decisive],
        color=theme_fill(_C_PRIMARY, ax), edgecolor="none", zorder=0,
    )

    # Critical-band masking-noise level Lpn: one horizontal segment per tone
    # spanning its critical band; the decisive band is emphasised.
    for i in range(freqs.size):
        is_dec = i == decisive
        ax.plot(
            [f1[i], f2[i]], [lg[i], lg[i]],
            color=_C_MUTED if not is_dec else _C_SECONDARY,
            lw=2.0 if is_dec else 1.2, solid_capstyle="round", zorder=2,
            label=_t(r"masking noise $L_{p\mathrm{n}}$ (critical band)", language)
            if i == 0 else None,
        )

    # Tone level Lpt: a stem from the masking-noise level up to the tone level,
    # with a marker at the top; the decisive tone is drawn in the reference hue.
    for i in range(freqs.size):
        is_dec = i == decisive
        ax.plot(
            [freqs[i], freqs[i]], [lg[i], lt[i]],
            color=_C_REFERENCE if is_dec else _C_PRIMARY,
            lw=1.6 if is_dec else 1.2, zorder=3,
        )
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("linestyle", "none")
    kwargs.setdefault("label", _t(r"tone level $L_{p\mathrm{t}}$", language))
    ax.plot(
        freqs, lt, color=_C_PRIMARY, markeredgecolor=_C_EDGE, zorder=4,
        **kwargs,
    )
    ax.plot(
        [freqs[decisive]], [lt[decisive]], marker="o", linestyle="none",
        color=_C_REFERENCE, markeredgecolor=_C_EDGE, zorder=5,
        label=_t(r"decisive $\Delta L_{{\mathrm{{ta}}}}$ = {da} dB @ {df} Hz", language).format(
            da=format_number(result.decisive_audibility, language, decimals=1),
            df=decimal_comma(f"{result.decisive_frequency:g}", language),
        ),
    )

    ax.set_xlabel(_t(_AXIS_FREQUENCY, language))
    ax.set_ylabel(_t("Sound pressure level [dB]", language))
    ax.set_title(_t("ISO 1996-2 tone-audibility analysis", language))
    format_frequency_axis(ax, float(f1.min()), float(f2.max()))
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, which="major", axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    localize_axes(ax, language)
    return ax


def plot_equal_loudness_contours(
    result: EqualLoudnessContours, ax: Axes | None = None, *, language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Normal equal-loudness-level contour family (ISO 226:2023).

    Draws sound pressure level in dB against a logarithmic frequency axis: one
    line per loudness level in ``result.phons`` (each annotated near 1 kHz when
    the frequency range includes 1 kHz) and the threshold of hearing T_f. This
    is the iconic ISO 226 chart.

    :param result: An
        :class:`~phonometry.psychoacoustics.loudness.contours.EqualLoudnessContours`.
    :param ax: Existing axes to draw on, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the contour line ``plot`` calls.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    contours = np.asarray(result.contours, dtype=np.float64)
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("linewidth", 1.5)

    fmin, fmax = float(freqs.min()), float(freqs.max())
    # The per-contour labels sit at the 1 kHz crossing (SPL = phon by
    # definition); when a frequency subset excludes 1 kHz they would land
    # outside the axes, so they are skipped.
    annotate_phons = fmin <= 1000.0 <= fmax
    for phon, spl in zip(result.phons, contours, strict=True):
        ax.plot(freqs, spl, **kwargs)
        if annotate_phons:
            ax.annotate(
                _t("{p} phon", language).format(
                    p=format_number(phon, language, decimals=0, trim=True)
                ),
                xy=(1000.0, phon), xytext=(1180.0, phon + 1.0),
                fontsize=9, color=kwargs["color"],
            )
    ax.plot(
        freqs, np.asarray(result.threshold, dtype=np.float64),
        color=_C_SECONDARY, linestyle="--",
        label=_t(r"Hearing threshold $T_f$", language),
    )

    ax.set_xlim(fmin, fmax)
    format_frequency_axis(ax, fmin, fmax)
    ax.set_xlabel(_t(_AXIS_FREQUENCY, language))
    ax.set_ylabel(_t("Sound pressure level [dB re 20 µPa]", language))
    ax.set_title(_t("ISO 226:2023 equal-loudness contours", language))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize="small")
    # Subtle footnote so the standalone figure is self-explanatory: the 90 phon
    # contour stops at 4 kHz and no higher contours exist by the standard.
    ax.text(
        0.015, 0.02,
        _t(
            "ISO 226:2023 defines 20 to 90 phon; above 80 phon the contour is "
            "defined only up to 4 kHz.",
            language,
        ),
        transform=ax.transAxes, fontsize=7, color=_C_MUTED,
        ha="left", va="bottom",
    )
    localize_axes(ax, language)
    return ax
