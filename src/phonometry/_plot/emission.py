#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Plot renderers for the emission domain (lazy imports from result .plot())."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .common import (
    _C_EDGE,
    _C_MUTED,
    _C_PRIMARY,
    _C_REFERENCE,
    _C_SECONDARY,
    _C_TERTIARY,
    _band_axis,
    _bar_width,
    _freq_axis,
    _hatch_invalid,
    _new_axes,
    _plot_band_level_bars,
    _sound_power_designation,
    format_frequency_axis,
    theme_fill,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..emission.intensity import FieldIndicators, IntensityResult
    from ..emission.intensity_compliance import (
        IntensityInstrumentComplianceResult,
    )
    from ..emission.sound_power import SoundEnergyResult, SoundPowerResult
    from ..emission.sound_power_anechoic import PrecisionSoundPowerResult
    from ..emission.sound_power_in_duct import InDuctSoundPowerResult
    from ..emission.sound_power_in_situ import InSituSoundPowerResult
    from ..emission.sound_power_intensity import (
        PrecisionIntensityResult,
        SoundPowerIntensityResult,
    )
    from ..emission.sound_power_intensity_points import (
        DiscretePointIntensityResult,
    )
    from ..emission.sound_power_reverberation import (
        ReverberationSoundEnergyResult,
        ReverberationSoundPowerResult,
    )
    from ..emission.vibration_sound_power import VibrationSoundPowerResult
    from ..emission.workstation import EmissionPressureResult

#: Shared frequency-axis label of the spectral renderers.
_FREQ_LABEL = "Frequency [Hz]"
#: Y-axis label of the residual-index plots (identical in both languages,
#: the symbol carries the meaning).
_LABEL_RESIDUAL_INDEX = r"$\delta_{pI0}$ [dB]"
#: Fewest frequency bands that make a per-band ISO 9614-1 indicators result
#: drawable as a curve: a size-1 array after ``atleast_1d`` is the
#: scalar/overall form of ``field_indicators`` (1D per-position input), which
#: ``plot()`` rejects as carrying no per-band data.
_MIN_BANDS = 2

#: Axis labels drawn by more than one renderer here. They are named rather
#: than repeated because the English text is also the key into ``_STRINGS``,
#: so a typo in one copy would silently fall back to English for that plot
#: alone, which is the kind of defect a reader sees and a test does not.
_YLABEL_LW = "Sound power level $L_W$ [dB]"
_YLABEL_LW_ABSOLUTE = r"Sound power level $L_W$ [dB re 1 pW]"
_YLABEL_LJ = "Sound energy level $L_J$ [dB]"

#: Spanish translations of the fixed labels/titles/legends rendered by the
#: emission-domain ``.plot()`` renderers, keyed by their verbatim English
#: text. ``_t`` returns the English key unchanged for any language other
#: than ``"es"``, so the English output is byte-for-byte identical to the
#: pre-i18n renderers.
_STRINGS: dict[str, str] = {
    "Band": "Banda",
    _YLABEL_LW: "Nivel de potencia acústica $L_W$ [dB]",
    _YLABEL_LJ: "Nivel de energía acústica $L_J$ [dB]",
    "sound power spectrum": "espectro de potencia acústica",
    "sound energy spectrum": "espectro de energía acústica",
    "In situ sound power spectrum (ISO 3747)": "Espectro de potencia acústica in situ (ISO 3747)",
    "In situ sound energy spectrum (ISO 3747)": "Espectro de energía acústica in situ (ISO 3747)",
    "Upper bound: background margin below 6 dB": "Cota superior: margen de fondo inferior a 6 dB",
    "Non-positive band": "Banda no positiva",
    "Pressure level $L_p$": "Nivel de presión $L_p$",
    "Intensity level $L_I$": "Nivel de intensidad $L_I$",
    "Level [dB]": "Nivel [dB]",
    r"Pressure-intensity index $\delta_{pI}$ [dB]": r"Índice presión-intensidad $\delta_{pI}$ [dB]",
    _YLABEL_LW_ABSOLUTE: "Nivel de potencia acústica $L_W$ [dB re 1 pW]",
    "ISO/TS 7849 sound power from surface vibration": "Potencia acústica por vibración superficial ISO/TS 7849",
    "$F_2$ (surface pressure-intensity)": "$F_2$ (presión-intensidad superficial)",
    "$F_3$ (negative partial power)": "$F_3$ (potencia parcial negativa)",
    r"Dynamic capability $L_\mathrm{d}$": r"Capacidad dinámica $L_\mathrm{d}$",
    "$F_4$ (non-uniformity)": "$F_4$ (no uniformidad)",
    "$F_1$ (temporal variability)": "$F_1$ (variabilidad temporal)",
    "$F_1$ limit (Table B.3)": "Límite de $F_1$ (tabla B.3)",
    "Indicator [dB]": "Indicador [dB]",
    "Field non-uniformity $F_4$": "No uniformidad del campo $F_4$",
    "Dimensionless indicators $F_1$, $F_4$": "Indicadores adimensionales $F_1$, $F_4$",
    "ISO 9614-1 field indicators": "Indicadores de campo ISO 9614-1",
    "Class {cls} pass region": "Región de aceptación clase {cls}",
    "Class 1 minimum": "Mínimo clase 1",
    "Class 2 minimum": "Mínimo clase 2",
    r"Measured $\delta_{pI0}$": r"$\delta_{pI0}$ medido",
    "Below the class {cls} minimum": "Bajo el mínimo de clase {cls}",
    _LABEL_RESIDUAL_INDEX: _LABEL_RESIDUAL_INDEX,
    "IEC 61043 Table 2 — {device}, {spacing} mm separation": "Tabla 2 de IEC 61043 — {device}, separación de {spacing} mm",
    "probe": "sonda",
    "processor": "procesador",
    "complete instrument": "instrumento completo",
    "Frequency [Hz]": "Frecuencia [Hz]",
    "Measured $L'_p$": "$L'_p$ medido",
    "Background $K_1$": "Fondo $K_1$",
    "Room $K_3$": "Sala $K_3$",
    "Emission $L_p$": "$L_p$ de emisión",
    "Sound pressure level [dB]": "Nivel de presión sonora [dB]",
    "Emission sound pressure level at the work station ({std})": "Nivel de presión sonora de emisión en el puesto de trabajo ({std})",
    "upper bound: the background is too close": "cota superior: el fondo está demasiado cerca",
    "grade 2 (engineering)": "grado 2 (ingeniería)",
    "grade 3 (survey)": "grado 3 (control)",
}


def _t(text: str, language: str = "en", **fmt: Any) -> str:
    """Localise a fixed string; English is returned verbatim (byte-identical)."""
    s = _STRINGS.get(text, text) if language == "es" else text
    return s.format(**fmt) if fmt else s


def plot_emission_pressure(
    result: EmissionPressureResult,
    ax: Axes | None = None,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """The reading, the two corrections taken off it, and what is left.

    A waterfall, because that is the shape of the arithmetic: ISO 11201/11202/
    11204 all print :math:`L_p = L'_p - K_1 - K_3`, and a reader wants to see
    which of the two corrections did the work. The measured bar and the
    emission bar stand on the axis; the two correction bars float between them,
    each starting where the previous one ended.

    A determination whose background margin fell below the grade's minimum is
    hatched, since the level drawn is an upper bound and the figure has to say
    so as plainly as the report does.

    :param result: An
        :class:`~phonometry.emission.workstation.EmissionPressureResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to :meth:`~matplotlib.axes.Axes.bar`.
    :return: The axes.
    :raises ValueError: If the result carries per-band arrays rather than the
        single overall level this figure draws.
    """
    from matplotlib.patches import Patch

    from .._i18n import format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    values = [
        float(np.asarray(result.measured_level_db).reshape(-1)[0]),
        float(np.asarray(result.background_correction_db).reshape(-1)[0]),
        float(np.asarray(result.local_correction_db).reshape(-1)[0]),
        float(np.asarray(result.level_db).reshape(-1)[0]),
    ]
    if np.asarray(result.level_db).size != 1:
        msg = (
            "plot_emission_pressure draws one determination; this result holds "
            f"{np.asarray(result.level_db).size} bands. Pass a single level."
        )
        raise ValueError(msg)

    measured, k1, k3, level = values
    # The two corrections hang from where the previous step left off, so the
    # bars read as one subtraction rather than as four unrelated numbers.
    after_k1 = measured - k1
    bottoms = [0.0, after_k1, level, 0.0]
    heights = [measured, k1, k3, level]
    colours = [_C_PRIMARY, _C_SECONDARY, _C_TERTIARY, _C_PRIMARY]
    labels = [
        _t("Measured $L'_p$", language),
        _t("Background $K_1$", language),
        _t("Room $K_3$", language),
        _t("Emission $L_p$", language),
    ]
    # The three names this figure positions itself are refused rather than
    # silently overridden, because a bar chart whose bottoms come from the
    # caller is not this figure any more. Everything the caller may reasonably
    # want, colour included, goes through setdefault so a kwarg wins.
    fixed = {"x", "height", "bottom"} & set(kwargs)
    if fixed:
        msg = (
            f"plot_emission_pressure positions its own bars; "
            f"{', '.join(sorted(fixed))} cannot be overridden."
        )
        raise TypeError(msg)
    kwargs.setdefault("width", 0.62)
    kwargs.setdefault("color", colours)
    kwargs.setdefault("edgecolor", _C_EDGE)
    bars = ax.bar(range(len(heights)), heights, bottom=bottoms, **kwargs)
    if result.upper_bound:
        _hatch_invalid(bars, np.array([True, False, False, True]))

    for index, (bar, value) in enumerate(zip(bars, heights, strict=True)):
        # A correction is written with its sign, because the figure is about
        # what came off; a level is written as the level it is.
        shown = (
            f"-{format_number(value, language, decimals=1)}"
            if index in (1, 2)
            else format_number(value, language, decimals=1)
        )
        ax.annotate(
            shown,
            xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_y() + bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(_t("Sound pressure level [dB]", language))
    ax.set_title(
        _t(
            "Emission sound pressure level at the work station ({std})",
            language,
            std=result.standard,
        )
    )
    grade = _t(
        "grade 2 (engineering)"
        if result.grade == "engineering"
        else "grade 3 (survey)",
        language,
    )
    handles: list[Any] = [Patch(facecolor=_C_MUTED, edgecolor=_C_EDGE, label=grade)]
    if result.upper_bound:
        handles.append(
            Patch(
                facecolor=_C_PRIMARY,
                edgecolor=_C_EDGE,
                hatch="///",
                label=_t("upper bound: the background is too close", language),
            )
        )
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    localize_axes(ax, language)
    return ax


def plot_sound_power(
    result: (
        SoundPowerResult
        | PrecisionSoundPowerResult
        | ReverberationSoundPowerResult
        | SoundPowerIntensityResult
        | PrecisionIntensityResult
        | DiscretePointIntensityResult
        | InDuctSoundPowerResult
    ),
    ax: Axes | None = None,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Sound power level spectrum with the A-weighted total annotated.

    Works for :class:`~phonometry.emission.sound_power.SoundPowerResult`,
    :class:`~phonometry.emission.sound_power_anechoic.PrecisionSoundPowerResult`,
    :class:`~phonometry.emission.sound_power_reverberation.ReverberationSoundPowerResult`,
    the two intensity-scanning results,
    :class:`~phonometry.emission.sound_power_intensity.SoundPowerIntensityResult`
    and
    :class:`~phonometry.emission.sound_power_intensity.PrecisionIntensityResult`,
    the discrete-point one,
    :class:`~phonometry.emission.sound_power_intensity_points.DiscretePointIntensityResult`,
    and the in-duct one,
    :class:`~phonometry.emission.sound_power_in_duct.InDuctSoundPowerResult`;
    for the three intensity variants the bands where the net power is
    non-positive (``negative_band`` / ``not_applicable_band``) are hatched and
    greyed as unusable.

    :param result: One of the seven sound-power results named above.
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
            ax,
            [f"{_t('Band', language)} {i + 1}" for i in range(n)],
            xlabel=_t("Band", language),
            language=language,
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

    ax.set_ylabel(_t(_YLABEL_LW, language))
    designation = _sound_power_designation(result)
    lwa = float(result.sound_power_level_a)
    if np.isfinite(lwa):
        ax.set_title(
            f"{designation} {_t('sound power spectrum', language)}  "
            "($L_{{W\\mathrm{{A}}}}$ = "
            f"{format_number(lwa, language, decimals=1)} dB(A))"
        )
    else:
        ax.set_title(f"{designation} {_t('sound power spectrum', language)}")
    if np.any(neg):
        ax.plot(
            [],
            [],
            color=_C_MUTED,
            marker="s",
            ls="",
            label=_t("Non-positive band", language),
        )
    if np.any(neg) or "label" in kwargs:
        ax.legend(loc="best", fontsize="small")
    ax.grid(True, axis="y", alpha=0.3)
    localize_axes(ax, language)
    return ax


def plot_sound_energy(
    result: SoundEnergyResult | ReverberationSoundEnergyResult,
    ax: Axes | None = None,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    r"""Sound energy level spectrum with the A-weighted total annotated.

    The sound energy counterpart of :func:`plot_sound_power`, for the two
    single-event determinations,
    :class:`~phonometry.emission.sound_power.SoundEnergyResult` (ISO 3744
    clause 8.3 / ISO 3746 clause 8.4) and
    :class:`~phonometry.emission.sound_power_reverberation.ReverberationSoundEnergyResult`
    (ISO 3741 clause 9.2): one bar per band of :math:`L_J`, the standard the
    result came from in the title and :math:`L_{J\mathrm{A}}` beside it when
    the band centres were supplied. Neither determination flags an
    undeterminable band, so nothing is hatched.

    :param result: One of the two sound-energy results named above.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the band :meth:`~matplotlib.axes.Axes.bar`.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    lj = np.asarray(result.sound_energy_level, dtype=np.float64)
    freqs = result.frequencies
    if freqs is None:
        positions = _band_axis(
            ax,
            [f"{_t('Band', language)} {i + 1}" for i in range(lj.size)],
            xlabel=_t("Band", language),
            language=language,
        )
    else:
        positions = _band_axis(
            ax, np.asarray(freqs, dtype=np.float64), language=language
        )
    kwargs.setdefault("color", _C_PRIMARY)
    ax.bar(positions, lj, **kwargs)

    ax.set_ylabel(_t("Sound energy level $L_J$ [dB]", language))
    designation = _sound_power_designation(result)
    lja = float(result.sound_energy_level_a)
    title = f"{designation} {_t('sound energy spectrum', language)}"
    if np.isfinite(lja):
        title += (
            "  ($L_{J\\mathrm{A}}$ = "
            f"{format_number(lja, language, decimals=1)} dB(A))"
        )
    ax.set_title(title)
    if "label" in kwargs:
        ax.legend(loc="best", fontsize="small")
    ax.grid(True, axis="y", alpha=0.3)
    localize_axes(ax, language)
    return ax


def plot_in_situ_sound_power(
    result: InSituSoundPowerResult,
    ax: Axes | None = None,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """In situ sound power (or energy) spectrum with the A-weighted total.

    One bar per octave band of ``LW``, or of ``LJ`` when the result is an
    energy determination; a band whose background margin fell below the
    6 dB of ISO 3747:2010 clause 8.1 is hatched, because the level drawn
    there is an upper bound and the report has to say so.

    :param result: An
        :class:`~phonometry.emission.sound_power_in_situ.InSituSoundPowerResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the band :meth:`~matplotlib.axes.Axes.bar`.
    :return: The axes.
    """
    from matplotlib.patches import Patch

    from .._i18n import format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    energy = result.quantity == "energy"
    levels = np.asarray(
        result.sound_energy_level if energy else result.sound_power_level,
        dtype=np.float64,
    )
    total = float(result.sound_energy_level_a if energy else result.sound_power_level_a)
    positions = _band_axis(
        ax, np.asarray(result.frequencies, dtype=np.float64), language=language
    )
    upper = ~np.asarray(result.background_requirement_met, dtype=bool)
    kwargs.setdefault("color", _C_PRIMARY)
    bars = ax.bar(positions, np.nan_to_num(levels), **kwargs)
    _hatch_invalid(bars, upper)

    if energy:
        ax.set_ylabel(_t(_YLABEL_LJ, language))
        title = _t("In situ sound energy spectrum (ISO 3747)", language)
        symbol = "$L_{J\\mathrm{A}}$"
    else:
        ax.set_ylabel(_t(_YLABEL_LW, language))
        title = _t("In situ sound power spectrum (ISO 3747)", language)
        symbol = "$L_{W\\mathrm{A}}$"
    if np.isfinite(total):
        title += f"  ({symbol} = {format_number(total, language, decimals=1)} dB(A))"
    ax.set_title(title)
    handles: list[Any] = []
    if np.any(upper):
        handles.append(
            Patch(
                facecolor=_C_PRIMARY,
                edgecolor=_C_EDGE,
                hatch="//",
                label=_t("Upper bound: background margin below 6 dB", language),
            )
        )
    if handles or "label" in kwargs:
        ax.legend(handles=handles or None, loc="best", fontsize="small")
    ax.grid(True, axis="y", alpha=0.3)
    localize_axes(ax, language)
    return ax


def plot_intensity(
    result: IntensityResult, ax: Axes | None = None, language: str = "en", **kwargs: Any
) -> Axes:
    """Pressure vs intensity level per band with the pressure-intensity index.

    Draws Lp and LI per band and, on a twin axis, the per-band
    pressure-intensity index ``Lp - LI`` (the reactivity indicator); the
    total index is annotated in the title.

    :param result: An :class:`~phonometry.emission.intensity.IntensityResult` with
        per-band data (obtained by requesting a band ``fraction``).
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the pressure-level curve ``plot`` call.
    :return: The axes.
    :raises ValueError: If the result carries no per-band data.
    """
    from .._i18n import format_number, localize_axes

    if result.frequency is None:
        msg = (
            "plot() needs per-band intensity data; call sound_intensity(...) "
            "with a 'fraction' to obtain it."
        )
        raise ValueError(msg)
    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.frequency, dtype=np.float64)
    lp = np.asarray(result.pressure_level, dtype=np.float64)
    li = np.asarray(result.intensity_level, dtype=np.float64)
    index = np.asarray(result.pressure_intensity_index, dtype=np.float64)

    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("label", _t("Pressure level $L_p$", language))
    ax.plot(freqs, lp, "o-", **kwargs)
    ax.plot(
        freqs,
        li,
        "s--",
        color=_C_REFERENCE,
        label=_t("Intensity level $L_I$", language),
    )
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
        label=r"$\delta_{pI} = L_p - L_I$",
    )
    twin.set_ylabel(_t(r"Pressure-intensity index $\delta_{pI}$ [dB]", language))

    lines, labels = ax.get_legend_handles_labels()
    tlines, tlabels = twin.get_legend_handles_labels()
    ax.legend(lines + tlines, labels + tlabels, loc="best", fontsize="small")
    ax.set_title(
        "ISO 9614 $L_p$ vs $L_I$  "
        r"(total $\delta_{pI}$ = "
        f"{format_number(result.total_pressure_intensity_index, language, decimals=1)}"
        " dB)"
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
    non-uniformity F4. When the result carries the temporal variability
    indicator F1 (that is, ``temporal_intensity`` was supplied), it is
    drawn on the same twin axis beside F4, together with the Table B.3
    limit of 0,6 that F1 must stay under.

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
    if result.frequency is None or f2.size < _MIN_BANDS:
        msg = (
            "plot() needs per-band indicators; call field_indicators(...) with "
            "2D (positions, bands) arrays and 'frequencies'."
        )
        raise ValueError(msg)
    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.frequency, dtype=np.float64)
    f3 = np.atleast_1d(np.asarray(result.f3, dtype=np.float64))
    f4 = np.atleast_1d(np.asarray(result.f4, dtype=np.float64))

    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("label", _t("$F_2$ (surface pressure-intensity)", language))
    ax.plot(freqs, f2, "o-", **kwargs)
    ax.plot(
        freqs,
        f3,
        "s--",
        color=_C_REFERENCE,
        label=_t("$F_3$ (negative partial power)", language),
    )
    if dynamic_capability is not None:
        ld = np.broadcast_to(
            np.asarray(dynamic_capability, dtype=np.float64), freqs.shape
        )
        ax.plot(
            freqs,
            ld,
            ls=":",
            lw=1.8,
            color=_C_MUTED,
            drawstyle="steps-mid",
            label=_t(r"Dynamic capability $L_\mathrm{d}$", language),
        )
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
        label=_t("$F_4$ (non-uniformity)", language),
    )
    twin.set_ylabel(_t("Field non-uniformity $F_4$", language))
    if result.f1 is not None:
        # F1 is dimensionless like F4, so it shares the twin axis; the
        # Table B.3 threshold above which the field is not stationary enough
        # is drawn alongside it.
        from ..emission.intensity import TEMPORAL_VARIABILITY_LIMIT

        f1 = np.broadcast_to(np.asarray(result.f1, dtype=np.float64), freqs.shape)
        twin.plot(
            freqs,
            f1,
            "^-",
            color=_C_SECONDARY,
            lw=1.4,
            label=_t("$F_1$ (temporal variability)", language),
        )
        twin.axhline(
            TEMPORAL_VARIABILITY_LIMIT,
            ls="-.",
            lw=1.0,
            color=_C_SECONDARY,
            label=_t("$F_1$ limit (Table B.3)", language),
        )
        twin.set_ylabel(_t("Dimensionless indicators $F_1$, $F_4$", language))

    lines, labels = ax.get_legend_handles_labels()
    tlines, tlabels = twin.get_legend_handles_labels()
    ax.legend(lines + tlines, labels + tlabels, loc="best", fontsize="small")
    ax.set_title(_t("ISO 9614-1 field indicators", language))
    localize_axes(ax, language)
    return ax


def plot_vibration_sound_power(
    result: VibrationSoundPowerResult,
    ax: Axes | None = None,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Radiated sound power level per band (ISO/TS 7849).

    :param result: A :class:`~phonometry.emission.vibration_sound_power.VibrationSoundPowerResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the bar ``plot``.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = _plot_band_level_bars(
        ax,
        result.sound_power_level,
        result.frequencies,
        result.total_level,
        ylabel=_t(_YLABEL_LW_ABSOLUTE, language),
        title=_t("ISO/TS 7849 sound power from surface vibration", language),
        language=language,
        **kwargs,
    )
    localize_axes(ax, language)
    return ax


_DEVICE_LABELS = {
    "probe": "probe",
    "processor": "processor",
    "instrument": "complete instrument",
}


def plot_intensity_class(
    result: IntensityInstrumentComplianceResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Measured pressure-residual intensity index over the IEC 61043 masks.

    Draws the measured ``delta_pI0`` per one-third-octave band against the
    Table 2 class 1 and class 2 *minima* for the device kind, already rescaled
    to the microphone separation in use. Because the requirement is a floor,
    the pass region of the reference class (the achieved class, or class 2 when
    the chain complies with neither) lies *above* its mask and is shaded,
    following the same convention as :func:`plot_filter_class`.

    The bands that cost the chain the *next* class up are ringed: for a class 2
    chain those are the bands under the class 1 minimum, and for a chain that
    meets no class those are the bands under the class 2 minimum. A class 1
    chain clears everything, so nothing is ringed.

    :param result: An
        :class:`~phonometry.emission.intensity_compliance.IntensityInstrumentComplianceResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the measured-curve ``plot`` call.
    :return: The axes.
    """
    from .._i18n import decimal_comma, localize_axes

    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.frequency, dtype=np.float64)
    measured = np.asarray(result.residual_index, dtype=np.float64)
    class1 = np.asarray(result.limit_class1, dtype=np.float64)
    class2 = np.asarray(result.limit_class2, dtype=np.float64)
    cls = result.reference_class()
    mask = class1 if cls == 1 else class2

    y_bot = float(np.floor(min(measured.min(), class2.min()) - 2.0))
    y_top = float(np.ceil(max(measured.max(), class1.max()) + 3.0))

    # Opaque, because the fiche renders this plot through svglib, which drops
    # alpha: a translucent fill would come out as a solid block over the
    # measured curve. theme_fill mixes the page towards the hue instead, so the
    # region reads the same way on either background.
    ax.fill_between(
        freqs,
        mask,
        y_top,
        step="mid",
        facecolor=theme_fill(_C_TERTIARY, ax),
        edgecolor="none",
        zorder=0,
        label=_t("Class {cls} pass region", language, cls=cls),
    )
    # Both Table 2 masks in the same amber, class 1 solid and class 2 dashed,
    # as the published intensity-analyser displays draw them.
    ax.plot(
        freqs,
        class1,
        drawstyle="steps-mid",
        color=_C_SECONDARY,
        lw=1.3,
        label=_t("Class 1 minimum", language),
    )
    ax.plot(
        freqs,
        class2,
        drawstyle="steps-mid",
        color=_C_SECONDARY,
        lw=1.3,
        ls="--",
        label=_t("Class 2 minimum", language),
    )

    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("lw", 1.6)
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("ms", 3.0)
    kwargs.setdefault("drawstyle", "steps-mid")
    kwargs.setdefault("label", _t(r"Measured $\delta_{pI0}$", language))
    ax.plot(freqs, measured, **kwargs)

    # Ring the bands that block the next class up: class 1 for a class 2 chain,
    # class 2 for a chain that meets neither. A class 1 chain has none.
    marked_cls = 1 if result.overall_class == 2 else 2  # noqa: PLR2004
    marked_mask = class1 if marked_cls == 1 else class2
    failing = (
        np.zeros(freqs.shape, dtype=bool)
        if result.overall_class == 1
        else measured < marked_mask - 1e-9
    )
    if np.any(failing):
        ax.plot(
            freqs[failing],
            measured[failing],
            ls="",
            marker="o",
            ms=6.0,
            mfc="none",
            mew=1.6,
            color=_C_REFERENCE,
            label=_t("Below the class {cls} minimum", language, cls=marked_cls),
        )

    format_frequency_axis(ax, float(freqs.min()), float(freqs.max()))
    ax.set_xlim(float(freqs.min()) / 1.15, float(freqs.max()) * 1.15)
    ax.set_ylim(y_bot, y_top)
    ax.set_xlabel(_t(_FREQ_LABEL, language))
    ax.set_ylabel(_t(_LABEL_RESIDUAL_INDEX, language))
    ax.set_title(
        _t(
            "IEC 61043 Table 2 — {device}, {spacing} mm separation",
            language,
            device=_t(_DEVICE_LABELS[result.device], language),
            # ``:g`` prints the separation exactly as the chain was verified
            # with (a 6.35 mm quarter-inch spacer stays 6.35, which a fixed
            # one-decimal format would round away); only its decimal separator
            # needs localising, and ``spacing`` is validated positive, so the
            # sign never enters.
            spacing=decimal_comma(f"{result.spacing * 1000.0:g}", language),
        )
    )
    ax.legend(loc="lower right", fontsize="small")
    ax.grid(True, which="both", alpha=0.3)
    localize_axes(ax, language)
    return ax
