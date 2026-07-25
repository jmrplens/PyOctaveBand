#  Copyright (c) 2026. Jose M. Requena-Plens
"""Plot renderers for the emission domain (lazy imports from result .plot())."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .common import (
    _C_MUTED,
    _C_PRIMARY,
    _C_REFERENCE,
    _C_TERTIARY,
    _band_axis,
    _bar_width,
    _freq_axis,
    _hatch_invalid,
    _new_axes,
    _plot_band_level_bars,
    _sound_power_designation,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..emission.intensity import FieldIndicators, IntensityResult
    from ..emission.sound_power import SoundPowerResult
    from ..emission.sound_power_intensity import SoundPowerIntensityResult
    from ..emission.sound_power_reverberation import ReverberationSoundPowerResult
    from ..emission.vibration_sound_power import VibrationSoundPowerResult

#: Spanish translations of the fixed labels/titles/legends rendered by the
#: emission-domain ``.plot()`` renderers, keyed by their verbatim English
#: text. ``_t`` returns the English key unchanged for any language other
#: than ``"es"``, so the English output is byte-for-byte identical to the
#: pre-i18n renderers.
_STRINGS: dict[str, str] = {
    "Band": "Banda",
    "Sound power level LW [dB]": "Nivel de potencia acústica LW [dB]",
    "sound power spectrum": "espectro de potencia acústica",
    "Non-positive band": "Banda no positiva",
    "Pressure level Lp": "Nivel de presión Lp",
    "Intensity level LI": "Nivel de intensidad LI",
    "Level [dB]": "Nivel [dB]",
    "Pressure-intensity index δpI [dB]": "Índice presión-intensidad δpI [dB]",
    "Sound power level $L_W$ [dB re 1 pW]": "Nivel de potencia acústica $L_W$ [dB re 1 pW]",
    "ISO/TS 7849 sound power from surface vibration": "Potencia acústica por vibración superficial ISO/TS 7849",
    "F2 (surface pressure-intensity)": "F2 (presión-intensidad superficial)",
    "F3 (negative partial power)": "F3 (potencia parcial negativa)",
    "Dynamic capability Ld": "Capacidad dinámica Ld",
    "F4 (non-uniformity)": "F4 (no uniformidad)",
    "Indicator [dB]": "Indicador [dB]",
    "Field non-uniformity F4": "No uniformidad del campo F4",
    "ISO 9614-1 field indicators": "Indicadores de campo ISO 9614-1",
}


def _t(text: str, language: str = "en") -> str:
    """Localise a fixed string; English is returned verbatim (byte-identical)."""
    return _STRINGS.get(text, text) if language == "es" else text


def plot_sound_power(
    result: (
        SoundPowerResult | ReverberationSoundPowerResult | SoundPowerIntensityResult | Any
    ),
    ax: Axes | None = None,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Sound power level spectrum with the A-weighted total annotated.

    Works for :class:`~phonometry.sound_power.SoundPowerResult`,
    :class:`~phonometry.sound_power_reverberation.ReverberationSoundPowerResult`
    and :class:`~phonometry.sound_power_intensity.SoundPowerIntensityResult`;
    for the intensity (scanning) variant the bands where the net power is
    non-positive (``negative_band``) are hatched and greyed as unusable.

    :param result: A sound-power result object exposing
        ``sound_power_level``, ``sound_power_level_a`` and (optionally)
        ``frequencies`` and ``negative_band``.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the band :meth:`~matplotlib.axes.Axes.bar`.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    lw = np.asarray(result.sound_power_level, dtype=np.float64)
    n = lw.size
    freqs = getattr(result, "frequencies", None)
    if freqs is None:
        positions = _band_axis(
            ax, [f"{_t('Band', language)} {i + 1}" for i in range(n)],
            xlabel=_t("Band", language), language=language,
        )
    else:
        positions = _band_axis(
            ax, np.asarray(freqs, dtype=np.float64), language=language
        )

    # ``negative_band`` (ISO 9614-2) and ``not_applicable_band`` (ISO 9614-3)
    # both flag bands whose net power is non-positive and therefore unusable.
    negative = getattr(result, "negative_band", None)
    if negative is None:
        negative = getattr(result, "not_applicable_band", None)
    neg = (
        np.asarray(negative, dtype=bool)
        if negative is not None
        else np.zeros(n, dtype=bool)
    )
    colors = [_C_MUTED if b else _C_PRIMARY for b in neg]
    kwargs.setdefault("color", colors)
    bars = ax.bar(positions, np.nan_to_num(lw), **kwargs)
    _hatch_invalid(bars, neg)

    ax.set_ylabel(_t("Sound power level LW [dB]", language))
    designation = _sound_power_designation(result)
    lwa = float(result.sound_power_level_a)
    if np.isfinite(lwa):
        ax.set_title(
            f"{designation} {_t('sound power spectrum', language)}  "
            f"(LWA = {format_number(lwa, language, decimals=1)} dB(A))"
        )
    else:
        ax.set_title(f"{designation} {_t('sound power spectrum', language)}")
    if np.any(neg):
        ax.plot([], [], color=_C_MUTED, marker="s", ls="",
                label=_t("Non-positive band", language))
    if np.any(neg) or "label" in kwargs:
        ax.legend(loc="best", fontsize="small")
    ax.grid(True, axis="y", alpha=0.3)
    localize_axes(ax, language)
    return ax


def plot_intensity(
    result: IntensityResult, ax: Axes | None = None, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Pressure vs intensity level per band with the pressure-intensity index.

    Draws Lp and LI per band and, on a twin axis, the per-band
    pressure-intensity index ``Lp - LI`` (the reactivity indicator); the
    total index is annotated in the title.

    :param result: An :class:`~phonometry.intensity.IntensityResult` with
        per-band data (obtained by requesting a band ``fraction``).
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the pressure-level curve ``plot`` call.
    :return: The axes.
    :raises ValueError: If the result carries no per-band data.
    """
    from .._i18n import format_number, localize_axes

    if result.frequency is None:
        raise ValueError(
            "plot() needs per-band intensity data; call sound_intensity(...) "
            "with a 'fraction' to obtain it."
        )
    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.frequency, dtype=np.float64)
    lp = np.asarray(result.pressure_level, dtype=np.float64)
    li = np.asarray(result.intensity_level, dtype=np.float64)
    index = np.asarray(result.pressure_intensity_index, dtype=np.float64)

    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("label", _t("Pressure level Lp", language))
    ax.plot(freqs, lp, "o-", **kwargs)
    ax.plot(freqs, li, "s--", color=_C_REFERENCE, label=_t("Intensity level LI", language))
    _freq_axis(ax, freqs, language=language)
    ax.set_ylabel(_t("Level [dB]", language))
    ax.grid(True, which="both", alpha=0.3)

    twin = ax.twinx()
    twin.bar(
        freqs,
        index,
        width=_bar_width(freqs),
        color=_C_TERTIARY,
        alpha=0.25,
        label="δpI = Lp - LI",
    )
    twin.set_ylabel(_t("Pressure-intensity index δpI [dB]", language))

    lines, labels = ax.get_legend_handles_labels()
    tlines, tlabels = twin.get_legend_handles_labels()
    ax.legend(lines + tlines, labels + tlabels, loc="best", fontsize="small")
    ax.set_title(
        "ISO 9614 Lp vs LI  "
        f"(total δpI = {format_number(result.total_pressure_intensity_index, language, decimals=1)} dB)"
    )
    localize_axes(ax, language)
    return ax


def plot_field_indicators(
    result: FieldIndicators,
    ax: Axes | None = None,
    *,
    dynamic_capability: float | np.ndarray | None = None,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Per-band ISO 9614-1 field indicators against the dynamic capability.

    Draws F2 (surface pressure-intensity) and F3 (negative partial power)
    per band, the optional dynamic capability index ``Ld`` as the
    criterion-1 reference line (the measurement arrangement is adequate
    where ``Ld > F2``) and, on a twin axis, the dimensionless field
    non-uniformity F4.

    :param result: A :class:`~phonometry.emission.intensity.FieldIndicators`
        with per-band data (2D input to
        :func:`~phonometry.emission.intensity.field_indicators`).
    :param ax: Existing axes, or ``None`` to create a figure.
    :param dynamic_capability: Optional ``Ld`` in dB (scalar or per band).
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the F2 curve ``plot`` call.
    :return: The axes.
    :raises ValueError: If the result carries no per-band data.
    """
    from .._i18n import localize_axes

    f2 = np.atleast_1d(np.asarray(result.f2, dtype=np.float64))
    if result.frequency is None or f2.size < 2:
        raise ValueError(
            "plot() needs per-band indicators; call field_indicators(...) with "
            "2D (positions, bands) arrays and 'frequencies'."
        )
    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.frequency, dtype=np.float64)
    f3 = np.atleast_1d(np.asarray(result.f3, dtype=np.float64))
    f4 = np.atleast_1d(np.asarray(result.f4, dtype=np.float64))

    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("label", _t("F2 (surface pressure-intensity)", language))
    ax.plot(freqs, f2, "o-", **kwargs)
    ax.plot(freqs, f3, "s--", color=_C_REFERENCE,
            label=_t("F3 (negative partial power)", language))
    if dynamic_capability is not None:
        ld = np.broadcast_to(
            np.asarray(dynamic_capability, dtype=np.float64), freqs.shape
        )
        ax.plot(freqs, ld, ls=":", lw=1.8, color=_C_MUTED,
                drawstyle="steps-mid", label=_t("Dynamic capability Ld", language))
    _freq_axis(ax, freqs, language=language)
    ax.set_ylabel(_t("Indicator [dB]", language))
    ax.grid(True, which="both", alpha=0.3)

    twin = ax.twinx()
    twin.bar(
        freqs,
        f4,
        width=_bar_width(freqs),
        color=_C_TERTIARY,
        alpha=0.25,
        label=_t("F4 (non-uniformity)", language),
    )
    twin.set_ylabel(_t("Field non-uniformity F4", language))

    lines, labels = ax.get_legend_handles_labels()
    tlines, tlabels = twin.get_legend_handles_labels()
    ax.legend(lines + tlines, labels + tlabels, loc="best", fontsize="small")
    ax.set_title(_t("ISO 9614-1 field indicators", language))
    localize_axes(ax, language)
    return ax


def plot_vibration_sound_power(
    result: VibrationSoundPowerResult, ax: Axes | None = None,
    language: str = "en", **kwargs: Any
) -> Axes:
    """Radiated sound power level per band (ISO/TS 7849).

    :param result: A :class:`~phonometry.vibration_sound_power.VibrationSoundPowerResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the bar ``plot``.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = _plot_band_level_bars(
        ax, result.sound_power_level, result.frequencies, result.total_level,
        ylabel=_t(r"Sound power level $L_W$ [dB re 1 pW]", language),
        title=_t("ISO/TS 7849 sound power from surface vibration", language),
        language=language,
        **kwargs,
    )
    localize_axes(ax, language)
    return ax
