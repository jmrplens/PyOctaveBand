#  Copyright (c) 2026. Jose M. Requena-Plens
"""Plot renderers for the environmental domain (lazy imports from result .plot())."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .common import (
    _C_MUTED,
    _C_PRIMARY,
    _C_PRIMARY_LIGHT,
    _C_QUATERNARY,
    _C_REFERENCE,
    _C_SECONDARY,
    _C_TERTIARY,
    _LEGEND_UPPER_RIGHT,
    _band_axis,
    _field_cmap,
    _freq_axis,
    _new_axes,
    format_frequency_axis,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..environmental.air_absorption import AtmosphericAttenuation
    from ..environmental.atmospheric_refraction import (
        AtmosphericPEResult,
        AtmosphericRayResult,
        EffectiveSoundSpeedProfile,
    )
    from ..environmental.cnossos_rail import RailwayEmissionResult
    from ..environmental.cnossos_road import RoadEmissionResult
    from ..environmental.ground_barriers import (
        BarrierInsertionLoss,
        SphericalGroundResult,
    )
    from ..environmental.impulse_prominence import ImpulseProminenceResult
    from ..environmental.measurement import TonalAssessmentResult
    from ..environmental.outdoor_propagation import OutdoorAttenuation
    from ..environmental.spanish_regulation import (
        ActivityAssessment,
        TonalCorrectionResult,
    )
    from ..environmental.wind_turbine_noise import WindTurbineTonalityResult

#: Spanish translations of the fixed strings rendered by the environmental
#: ``.plot()`` renderers, keyed by their verbatim English text.  ``_t``
#: returns the English key unchanged for any language other than ``"es"``,
#: so the English output is byte-for-byte identical to the pre-i18n
#: renderers.
_STRINGS: dict[str, str] = {
    "Narrowband spectrum": "Espectro de banda estrecha",
    "Critical band": "Banda crítica",
    "Masking level": "Nivel de enmascaramiento",
    "Tone": "Tono",
    "Frequency [Hz]": "Frecuencia [Hz]",
    "Level [dB]": "Nivel [dB]",
    "IEC 61400-11 tonal audibility": "Audibilidad tonal IEC 61400-11",
    "threshold": "umbral",
    "Impulses": "Impulsos",
    "Governing": "Determinante",
    "Predicted prominence $P$": "Prominencia prevista $P$",
    "Adjustment $K_I$ [dB]": "Ajuste $K_I$ [dB]",
    "NT ACOU 112 — impulse adjustment to $L_{Aeq}$": "NT ACOU 112 — ajuste por impulsos a $L_{Aeq}$",
    "knees $\\Delta L_{ta}=4,\\,10$ dB": "codos $\\Delta L_{ta}=4,\\,10$ dB",
    "Tonal audibility $\\Delta L_{ta}$ [dB]": "Audibilidad tonal $\\Delta L_{ta}$ [dB]",
    "Tonal adjustment $K_t$ [dB]": "Ajuste tonal $K_t$ [dB]",
    "ISO 1996-2 tonal adjustment": "Ajuste tonal ISO 1996-2",
    "$A_{div}$ — divergence": "$A_{div}$ — divergencia",
    "$A_{atm}$ — atmospheric": "$A_{atm}$ — atmosférica",
    "$A_{gr}$ — ground": "$A_{gr}$ — suelo",
    "$A_{bar}$ — barrier": "$A_{bar}$ — barrera",
    "$A$ — total": "$A$ — total",
    "Attenuation A [dB]": "Atenuación A [dB]",
    "ISO 9613-2 attenuation breakdown": "Desglose de atenuación ISO 9613-2",
    "CNOSSOS-EU railway source line power": (
        "Potencia de la línea fuente ferroviaria CNOSSOS-EU"
    ),
    "Both heights": "Ambas alturas",
    "Source A (0,5 m)": "Fuente A (0,5 m)",
    "Source B (4,0 m)": "Fuente B (4,0 m)",
    "Excess attenuation $\\Delta L$": "Atenuación en exceso $\\Delta L$",
    "Free field (0 dB)": "Campo libre (0 dB)",
    "Hard-ground limit (+6 dB)": "Límite de suelo duro (+6 dB)",
    "Level re free field [dB]": "Nivel re campo libre [dB]",
    "Spherical-wave ground effect (Weyl-Van der Pol)": "Efecto de suelo de onda esférica (Weyl-Van der Pol)",
    "Insertion loss": "Pérdida por inserción",
    "ground": "suelo",
    "Grazing limit (5 dB)": "Límite rasante (5 dB)",
    "Insertion loss [dB]": "Pérdida por inserción [dB]",
    "Barrier insertion loss": "Pérdida por inserción de barrera",
    "Effective sound speed [m/s]": "Velocidad efectiva del sonido [m/s]",
    "Height [m]": "Altura [m]",
    "Range [m]": "Distancia [m]",
    "Effective sound-speed profile": "Perfil de velocidad efectiva del sonido",
    "Source": "Fuente",
    "Atmospheric ray paths": "Trayectorias de rayos atmosféricos",
    "GFPE relative sound level": "Nivel sonoro relativo GFPE",
    r"Attenuation coefficient $\alpha$ [dB/km]":
        r"Coeficiente de atenuación $\alpha$ [dB/km]",
    "ISO 9613-1 atmospheric attenuation": "Atenuación atmosférica ISO 9613-1",
    "Band level": "Nivel de banda",
    "$L_t$ vs neighbour mean": "$L_t$ frente a la media de contiguas",
    "Band level [dB]": "Nivel de banda [dB]",
    "$L_t$ [dB]": "$L_t$ [dB]",
    "RD 1367/2007 tonal correction $K_t$ = {kt} dB":
        "Corrección tonal $K_t$ = {kt} dB (RD 1367/2007)",
    "max $L_{Keq,Ti}$": "máx. $L_{Keq,Ti}$",
    "$L_{Keq,x}$ (daily)": "$L_{Keq,x}$ (diario)",
    "$L_{K,x}$ (annual)": "$L_{K,x}$ (anual)",
    "limit + 5 dB": "límite + 5 dB",
    "limit + 3 dB": "límite + 3 dB",
    "limit": "límite",
    "Day": "Día",
    "Evening": "Tarde",
    "Night": "Noche",
    "Corrected level [dB]": "Nivel corregido [dB]",
    "RD 1367/2007 assessment vs limit values":
        "Evaluación RD 1367/2007 frente a los valores límite",
    "Total line": "Línea total",
    "Light vehicles (1)": "Vehículos ligeros (1)",
    "Medium heavy vehicles (2)": "Vehículos pesados medios (2)",
    "Heavy vehicles (3)": "Vehículos pesados (3)",
    "Mopeds (4a)": "Ciclomotores (4a)",
    "Motorcycles (4b)": "Motocicletas (4b)",
    "$L'_{W,eq,line}$ [dB re 1 pW/m]": "$L'_{W,eq,line}$ [dB re 1 pW/m]",
    "CNOSSOS-EU road source line power":
        "Potencia de la línea fuente viaria CNOSSOS-EU",
}

#: Localised legend labels of the five CNOSSOS-EU road vehicle categories,
#: keyed by the Table [2.2.a] category code.
_ROAD_CATEGORY_LABELS: dict[str, str] = {
    "1": "Light vehicles (1)",
    "2": "Medium heavy vehicles (2)",
    "3": "Heavy vehicles (3)",
    "4a": "Mopeds (4a)",
    "4b": "Motorcycles (4b)",
}


def _t(text: str, language: str = "en") -> str:
    """Localise a fixed string; English is returned verbatim (byte-identical)."""
    return _STRINGS.get(text, text) if language == "es" else text


def plot_atmospheric_attenuation(
    result: AtmosphericAttenuation, ax: Axes | None = None, *, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Pure-tone atmospheric attenuation coefficient vs frequency (ISO 9613-1).

    Draws ``alpha`` in dB/km (the Table 1 unit, i.e. the stored dB/m ``x 1000``)
    on a logarithmic frequency axis for the result's atmospheric conditions, the
    classic ISO 9613-1:1993 curve: the ``f^2`` low-frequency rise and the
    humidity-dependent relaxation roll-off.

    :param result: An
        :class:`~phonometry.environmental.air_absorption.AtmosphericAttenuation`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the ``alpha`` curve ``plot`` call.
    :return: The axes.
    """
    from .._i18n import decimal_comma, localize_axes

    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    alpha_km = np.asarray(result.attenuation_coefficient, dtype=np.float64) * 1000.0
    t_str = decimal_comma(f"{result.temperature:g}", language)
    rh_str = decimal_comma(f"{result.relative_humidity:g}", language)
    rh_unit = "% HR" if language == "es" else "% RH"
    label = f"{t_str} °C, {rh_str} {rh_unit}"
    # dB/km is already a logarithmic quantity, so the ordinate stays linear;
    # only the frequency axis is logarithmic (semilogx + format_frequency_axis).
    ax.semilogx(freqs, alpha_km, **{"color": _C_PRIMARY, "lw": 1.8,
                                    "label": label, **kwargs})
    fmin, fmax = float(freqs.min()), float(freqs.max())
    ax.set_xlim(fmin, fmax)
    format_frequency_axis(ax, fmin, fmax)
    ax.set_xlabel(_t("Frequency [Hz]", language))
    ax.set_ylabel(_t(r"Attenuation coefficient $\alpha$ [dB/km]", language))
    ax.set_title(_t("ISO 9613-1 atmospheric attenuation", language))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize="small")
    localize_axes(ax, language)
    return ax


def plot_wind_turbine_tonality(
    result: WindTurbineTonalityResult, ax: Axes | None = None, *, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Narrowband spectrum with the critical band, masking level and the tone.

    :param result: A :class:`~phonometry.wind_turbine_noise.WindTurbineTonalityResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the spectrum ``plot`` call.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes
    from ..environmental.wind_turbine_noise import _critical_band_edges

    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    levels = np.asarray(result.levels, dtype=np.float64)
    fc = result.tone_frequency
    lo, hi = _critical_band_edges(fc)
    ax.plot(freqs, levels, **{"color": _C_PRIMARY, "lw": 1.0, "label": _t("Narrowband spectrum", language), **kwargs})
    ax.axvspan(lo, hi, color=_C_TERTIARY, alpha=0.12, label=_t("Critical band", language))
    ax.axhline(result.masking_level, color=_C_MUTED, ls="--", lw=1.0,
               label=f"{_t('Masking level', language)} ({format_number(result.masking_level, language)} dB)")
    ax.plot([fc], [result.tone_level], "o", color=_C_REFERENCE,
            label=f"{_t('Tone', language)} ({format_number(result.tone_level, language)} dB)")
    ax.set_xlabel(_t("Frequency [Hz]", language))
    ax.set_ylabel(_t("Level [dB]", language))
    ax.set_title(f"{_t('IEC 61400-11 tonal audibility', language)} ΔLₐ = {format_number(result.tonal_audibility, language)} dB")
    ax.grid(True, alpha=0.3)
    ax.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")
    localize_axes(ax, language)
    return ax

def plot_impulse_prominence(
    result: ImpulseProminenceResult, ax: Axes | None = None, *, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Adjustment curve ``KI(P)`` with the candidate impulses marked.

    :param result: An :class:`~phonometry.impulse_prominence.ImpulseProminenceResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the impulses ``scatter``.
    :return: The axes.
    """
    from .._i18n import decimal_comma, format_number, localize_axes
    from ..environmental.impulse_prominence import (
        ADJUSTMENT_THRESHOLD,
        impulse_adjustment,
    )

    ax = ax if ax is not None else _new_axes()
    per = np.asarray(result.per_impulse, dtype=np.float64)
    per_max = float(per.max()) if per.size else 0.0
    p_max = max(per_max, result.prominence, 15.0) + 1.0
    grid = np.linspace(0.0, p_max, 200)
    ax.plot(grid, impulse_adjustment(grid), color=_C_PRIMARY,
            label=r"$K_I = 1.8\,(P-5)$")
    ax.axvline(ADJUSTMENT_THRESHOLD, color=_C_MUTED, ls=":",
               label=f"{_t('threshold', language)} $P = {decimal_comma(f'{ADJUSTMENT_THRESHOLD:g}', language)}$")

    kwargs.setdefault("color", _C_PRIMARY_LIGHT)
    kwargs.setdefault("zorder", 3)
    ax.scatter(per, impulse_adjustment(per), label=_t("Impulses", language), **kwargs)
    ax.scatter([result.prominence], [result.adjustment], color=_C_REFERENCE,
               zorder=4, s=90, marker="*",
               label=f"{_t('Governing', language)}  P = {format_number(result.prominence, language, decimals=2)},  "
                     f"$K_I$ = {format_number(result.adjustment, language)} dB")
    ax.set_xlabel(_t("Predicted prominence $P$", language))
    ax.set_ylabel(_t("Adjustment $K_I$ [dB]", language))
    ax.set_title(_t("NT ACOU 112 — impulse adjustment to $L_{Aeq}$", language))
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper left", fontsize="small")
    ax.grid(True, alpha=0.3)
    localize_axes(ax, language)
    return ax

def plot_tonal_adjustment(
    result: TonalAssessmentResult, ax: Axes | None = None, *, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Tonal adjustment curve ``Kt(ΔLta)`` with the assessed tone marked.

    :param result: A
        :class:`~phonometry.environmental_measurement.TonalAssessmentResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the assessed-tone ``scatter``.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes
    from ..environmental.measurement import tonal_adjustment

    ax = ax if ax is not None else _new_axes()
    top = max(result.audibility, 12.0) + 1.0
    grid = np.linspace(0.0, top, 200)
    curve = np.array([tonal_adjustment(d) for d in grid], dtype=np.float64)
    ax.plot(grid, curve, color=_C_PRIMARY, label=r"$K_t(\Delta L_{ta})$")
    ax.axvline(4.0, color=_C_MUTED, ls=":", label=_t("knees $\\Delta L_{ta}=4,\\,10$ dB", language))
    ax.axvline(10.0, color=_C_MUTED, ls=":")

    kwargs.setdefault("color", _C_REFERENCE)
    kwargs.setdefault("zorder", 4)
    kwargs.setdefault("s", 90)
    kwargs.setdefault("marker", "*")
    ax.scatter([result.audibility], [result.adjustment],
               label=rf"$\Delta L_{{ta}}$ = {format_number(result.audibility, language)} dB,  "
                     rf"$K_t$ = {format_number(result.adjustment, language)} dB", **kwargs)
    ax.set_xlabel(_t("Tonal audibility $\\Delta L_{ta}$ [dB]", language))
    ax.set_ylabel(_t("Tonal adjustment $K_t$ [dB]", language))
    ax.set_title(_t("ISO 1996-2 tonal adjustment", language))
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper left", fontsize="small")
    ax.grid(True, alpha=0.3)
    localize_axes(ax, language)
    return ax

def plot_outdoor_attenuation(
    result: OutdoorAttenuation, ax: Axes | None = None, *, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Stacked per-band attenuation terms with the total overlaid (ISO 9613-2).

    The divergence, atmospheric, ground and barrier terms are stacked per
    octave band on separate positive and negative baselines (the ground
    effect can be a net *gain*), and the total attenuation ``A`` is drawn
    as the primary marker line on top.

    :param result: An
        :class:`~phonometry.outdoor_propagation.OutdoorAttenuation`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the total-attenuation ``plot`` call.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    positions = _band_axis(ax, freqs, language=language)
    n = freqs.size

    # Separate positive and negative cumulative baselines so a negative term
    # stacks below zero instead of being drawn on top of the previous bars;
    # the signed heights sum to a_total.
    pos_bottom = np.zeros(n)
    neg_bottom = np.zeros(n)
    terms = (
        (result.a_div, _C_PRIMARY, _t("$A_{div}$ — divergence", language)),
        (result.a_atm, _C_TERTIARY, _t("$A_{atm}$ — atmospheric", language)),
        (result.a_gr, _C_QUATERNARY, _t("$A_{gr}$ — ground", language)),
        (result.a_bar, _C_SECONDARY, _t("$A_{bar}$ — barrier", language)),
    )
    for values, color, label in terms:
        term = np.asarray(values, dtype=np.float64)
        bottom = np.where(term >= 0.0, pos_bottom, neg_bottom)
        ax.bar(positions, term, bottom=bottom, color=color, label=label)
        pos_bottom += np.maximum(term, 0.0)
        neg_bottom += np.minimum(term, 0.0)

    kwargs.setdefault("color", _C_REFERENCE)
    kwargs.setdefault("marker", "D")
    kwargs.setdefault("label", _t("$A$ — total", language))
    ax.plot(positions, np.asarray(result.a_total, dtype=np.float64),
            zorder=4, **kwargs)
    ax.axhline(0.0, color=_C_MUTED, lw=0.8)
    ax.set_ylabel(_t("Attenuation A [dB]", language))
    ax.set_title(_t("ISO 9613-2 attenuation breakdown", language))
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, axis="y", alpha=0.3)
    localize_axes(ax, language)
    return ax


def plot_cnossos_rail_emission(
    result: RailwayEmissionResult, ax: Axes | None = None, *, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Per-metre railway source-line power of the two equivalent source heights.

    Draws the energy-summed line power of the track as bars over the eight
    CNOSSOS-EU octave bands, with source A (0,5 m) and source B (4,0 m) overlaid
    as marker lines, so the bands where the roof and pantograph sources matter
    are read directly off the chart.

    :param result: A
        :class:`~phonometry.environmental.cnossos_rail.RailwayEmissionResult`.
def plot_cnossos_road_emission(
    result: RoadEmissionResult, ax: Axes | None = None, *, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Per-metre road source-line power with its vehicle-category breakdown.

    Draws the energy-summed line power ``L'_W,eq,line,i`` of the traffic mix as
    bars over the eight CNOSSOS-EU octave bands, with the contribution of each
    vehicle category overlaid as a marker line, so the band where a category
    governs the source is read directly off the chart.

    :param result: A
        :class:`~phonometry.environmental.cnossos_road.RoadEmissionResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the total-line-power ``bar`` call.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    positions = _band_axis(ax, freqs, language=language)

    kwargs.setdefault("color", _C_PRIMARY_LIGHT)
    kwargs.setdefault("label", _t("Both heights", language))
    # A silent source carries -inf, which no axis can hold; masking it leaves
    # the band out of the drawing instead of collapsing the whole scale.
    total = np.asarray(result.total_line_power, dtype=np.float64)
    ax.bar(positions, np.where(np.isfinite(total), total, np.nan), **kwargs)

    styles = (
        (_C_PRIMARY, "o", "Source A (0,5 m)"),
        (_C_SECONDARY, "s", "Source B (4,0 m)"),
    )
    for row, (color, marker, label) in zip(
        np.asarray(result.line_power, dtype=np.float64), styles, strict=True
    ):
        ax.plot(
            positions, np.where(np.isfinite(row), row, np.nan), color=color,
            marker=marker, lw=1.2, ms=4, zorder=4, label=_t(label, language),
        )
    # Pure symbol notation, identical in every language, so it is set directly
    # rather than routed through the translation table.
    ax.set_ylabel("$L'_{W,eq,line}$ [dB re 1 pW/m]")
    ax.set_title(_t("CNOSSOS-EU railway source line power", language))
    kwargs.setdefault("label", _t("Total line", language))
    ax.bar(positions, np.asarray(result.total_line_power, dtype=np.float64), **kwargs)

    colors = (_C_PRIMARY, _C_TERTIARY, _C_REFERENCE, _C_SECONDARY, _C_QUATERNARY)
    markers = ("o", "s", "D", "^", "v")
    for row, category, color, marker in zip(
        np.asarray(result.line_power, dtype=np.float64), result.categories,
        colors, markers, strict=False,
    ):
        ax.plot(
            positions, row, color=color, marker=marker, lw=1.2, ms=4, zorder=4,
            label=_t(_ROAD_CATEGORY_LABELS[category.value], language),
        )
    ax.set_ylabel(_t("$L'_{W,eq,line}$ [dB re 1 pW/m]", language))
    ax.set_title(_t("CNOSSOS-EU road source line power", language))
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, axis="y", alpha=0.3)
    localize_axes(ax, language)
    return ax


def plot_spherical_ground(
    result: SphericalGroundResult, ax: Axes | None = None, *, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Excess attenuation (level re free field) of the spherical-wave ground effect.

    Draws the relative sound level ``dL`` versus frequency with the ``+6 dB``
    hard-ground enhancement ceiling and the ``0 dB`` free-field reference, so the
    ground-effect dip and any surface-wave enhancement are both visible.

    :param result: A
        :class:`~phonometry.environmental.ground_barriers.SphericalGroundResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the ``dL`` ``plot`` call.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    d_l = np.asarray(result.excess_attenuation, dtype=np.float64)
    ax.plot(freqs, d_l, **{"color": _C_PRIMARY, "lw": 1.4, "marker": "o",
                           "ms": 3.0, "label": _t("Excess attenuation $\\Delta L$", language),
                           **kwargs})
    ax.axhline(0.0, color=_C_MUTED, lw=0.8, label=_t("Free field (0 dB)", language))
    ax.axhline(6.0, color=_C_REFERENCE, ls="--", lw=0.9,
               label=_t("Hard-ground limit (+6 dB)", language))
    _freq_axis(ax, freqs, language=language)
    ax.set_ylabel(_t("Level re free field [dB]", language))
    ax.set_title(_t("Spherical-wave ground effect (Weyl-Van der Pol)", language))
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, which="both", alpha=0.3)
    localize_axes(ax, language)
    return ax


def plot_barrier_insertion_loss(
    result: BarrierInsertionLoss, ax: Axes | None = None, *, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Barrier insertion loss versus frequency.

    :param result: A
        :class:`~phonometry.environmental.ground_barriers.BarrierInsertionLoss`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the insertion-loss ``plot`` call.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    il = np.asarray(result.insertion_loss, dtype=np.float64)
    ground_frag = f", {_t('ground', language)}" if result.ground else ""
    label = f"{_t('Insertion loss', language)} ({result.method}{ground_frag})"
    ax.plot(freqs, il, **{"color": _C_SECONDARY, "lw": 1.4, "marker": "s",
                          "ms": 3.0, "label": label, **kwargs})
    ax.axhline(0.0, color=_C_MUTED, lw=0.8)
    ax.axhline(5.0, color=_C_MUTED, ls=":", lw=0.9,
               label=_t("Grazing limit (5 dB)", language))
    _freq_axis(ax, freqs, language=language)
    ax.set_ylabel(_t("Insertion loss [dB]", language))
    ax.set_title(_t("Barrier insertion loss", language))
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, which="both", alpha=0.3)
    localize_axes(ax, language)
    return ax


def plot_sound_speed_profile(
    profile: EffectiveSoundSpeedProfile, ax: Axes | None = None, *, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Effective sound-speed profile ``c_eff(z)`` (height on the vertical axis).

    :param profile: An
        :class:`~phonometry.environmental.atmospheric_refraction.EffectiveSoundSpeedProfile`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the profile ``plot`` call.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    z = np.asarray(profile.heights, dtype=np.float64)
    c = np.asarray(profile.sound_speeds, dtype=np.float64)
    label = profile.description or "c_eff(z)"
    ax.plot(c, z, **{"color": _C_PRIMARY, "lw": 1.4, "label": label, **kwargs})
    ax.set_xlabel(_t("Effective sound speed [m/s]", language))
    ax.set_ylabel(_t("Height [m]", language))
    ax.set_title(_t("Effective sound-speed profile", language))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    localize_axes(ax, language)
    return ax


def plot_atmospheric_rays(
    result: AtmosphericRayResult, ax: Axes | None = None, *, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Curved sound-ray paths over the ground (height on the vertical axis).

    :param result: An
        :class:`~phonometry.environmental.atmospheric_refraction.AtmosphericRayResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to each ray ``plot`` call.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    r = np.asarray(result.ranges, dtype=np.float64)
    z = np.asarray(result.heights, dtype=np.float64)
    for i in range(r.shape[0]):
        ax.plot(r[i], z[i], **{"color": _C_PRIMARY, "lw": 0.7, "alpha": 0.7, **kwargs})
    ax.plot([0.0], [result.source_height], "o", color=_C_REFERENCE, label=_t("Source", language))
    ax.axhline(0.0, color=_C_MUTED, lw=1.0)
    ax.set_xlabel(_t("Range [m]", language))
    ax.set_ylabel(_t("Height [m]", language))
    ax.set_ylim(bottom=0.0)
    ax.set_title(_t("Atmospheric ray paths", language))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize="small")
    localize_axes(ax, language)
    return ax


def plot_atmospheric_pe(
    result: AtmosphericPEResult, ax: Axes | None = None, *, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Parabolic-equation relative-level field over the range-height plane.

    The field is drawn as a single raster image (``imshow``): per-cell vector
    quads are avoided so the figure stays light and free of moire (the repo's
    pcolormesh-in-SVG policy).  The default diverging colormap follows the
    axes background so the centre of the scale blends into the page --
    ``RdBu_r`` on a light background, the black-centred
    ``phonometry_field_dark`` on a dark one -- and ``cmap`` overrides it.

    :param result: An
        :class:`~phonometry.environmental.atmospheric_refraction.AtmosphericPEResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to ``imshow``.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    r = np.asarray(result.ranges, dtype=np.float64)
    z = np.asarray(result.heights, dtype=np.float64)
    dl = np.asarray(result.relative_level, dtype=np.float64)
    finite = dl[np.isfinite(dl)]
    vmax = float(np.percentile(finite, 99)) if finite.size else 6.0
    vmax = max(vmax, 6.0)
    dl = np.where(np.isfinite(dl), dl, np.nan)
    img = ax.imshow(
        dl,
        **{
            "cmap": _field_cmap(ax),
            "vmin": -30.0,
            "vmax": vmax,
            "aspect": "auto",
            "origin": "lower",
            "interpolation": "bilinear",
            "extent": (float(r[0]), float(r[-1]), float(z[0]), float(z[-1])),
            **kwargs,
        },
    )
    ax.figure.colorbar(img, ax=ax, label=_t("Level re free field [dB]", language))
    ax.plot([0.0], [result.source_height], "o", color="k", ms=4.0, label=_t("Source", language))
    ax.set_xlabel(_t("Range [m]", language))
    ax.set_ylabel(_t("Height [m]", language))
    ax.set_title(f"{_t('GFPE relative sound level', language)} ({format_number(result.frequency, language, decimals=0)} Hz)")
    ax.legend(loc="upper right", fontsize="small")
    localize_axes(ax, language)
    return ax


def plot_tonal_correction_rd1367(
    result: TonalCorrectionResult, ax: Axes | None = None, *, language: str = "en",
    **kwargs: Any
) -> Axes:
    """One-third-octave spectrum with the RD 1367/2007 emergence ``Lt`` per band.

    The band levels are drawn as bars on the left axis and the emergence
    ``Lt = Lf - Ls`` (the band level above the arithmetic mean of its two
    neighbours) on a twin right axis, with the band that governs ``Kt``
    highlighted.

    :param result: A
        :class:`~phonometry.environmental.spanish_regulation.TonalCorrectionResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the band-level ``bar`` call.
    :return: The axes (the left, band-level axes).
    """
    from .._i18n import format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    levels = np.asarray(result.levels, dtype=np.float64)
    lt = np.asarray(result.differences, dtype=np.float64)
    positions = _band_axis(ax, freqs, language=language)

    kwargs.setdefault("color", _C_PRIMARY_LIGHT)
    kwargs.setdefault("label", _t("Band level", language))
    ax.bar(positions, levels, width=0.72, **kwargs)
    ax.set_ylabel(_t("Band level [dB]", language))

    twin = ax.twinx()
    twin.plot(positions, lt, "o-", color=_C_SECONDARY, ms=4.0,
              label=_t("$L_t$ vs neighbour mean", language))
    if result.governing_frequency is not None:
        index = int(np.argmin(np.abs(freqs - result.governing_frequency)))
        twin.plot([positions[index]], [lt[index]], "*", color=_C_REFERENCE,
                  ms=14.0, zorder=5)
    twin.set_ylabel(_t("$L_t$ [dB]", language))
    twin.axhline(0.0, color=_C_MUTED, lw=0.8)

    ax.set_title(
        _t("RD 1367/2007 tonal correction $K_t$ = {kt} dB", language).format(
            kt=format_number(result.correction, language, decimals=0)
        )
    )
    handles, labels = ax.get_legend_handles_labels()
    extra_handles, extra_labels = twin.get_legend_handles_labels()
    ax.legend(handles + extra_handles, labels + extra_labels,
              loc="upper left", fontsize="small")
    ax.grid(True, axis="y", alpha=0.3)
    localize_axes(ax, language)
    localize_axes(twin, language)
    return ax


def plot_activity_assessment(
    result: ActivityAssessment, ax: Axes | None = None, *, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Per-period RD 1367/2007 indices against their limit values.

    For every assessed evaluation period the three indices of Article 25.1 b
    are drawn as grouped bars (the largest measured ``LKeq,Ti``, the daily
    ``LKeq,x`` and the annual ``LK,x``) with their own limits marked on each
    group: the table limit plus 5 dB, plus 3 dB and the limit itself. A bar
    that exceeds its own limit is outlined in the exceedance colour.

    :param result: An
        :class:`~phonometry.environmental.spanish_regulation.ActivityAssessment`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the ``LKeq,x`` ``bar`` call.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    periods = list(result.periods)
    labels = [_t(p.period.capitalize(), language) for p in periods]
    positions = _band_axis(ax, labels, xlabel=None, language=language)

    step = 0.28
    width = 0.24
    series = (
        (
            [p.max_phase_level for p in periods],
            [p.phase_limit for p in periods],
            [p.phase_pass for p in periods],
            _C_PRIMARY_LIGHT,
            _t("max $L_{Keq,Ti}$", language),
            _t("limit + 5 dB", language),
            -step,
            ":",
            False,
        ),
        (
            [float(p.reported_level) for p in periods],
            [p.daily_limit for p in periods],
            [p.daily_pass for p in periods],
            _C_PRIMARY,
            _t("$L_{Keq,x}$ (daily)", language),
            _t("limit + 3 dB", language),
            0.0,
            "--",
            True,
        ),
        (
            [
                float(p.reported_long_term)
                if p.reported_long_term is not None
                else np.nan
                for p in periods
            ],
            [p.limit for p in periods],
            [p.long_term_pass for p in periods],
            _C_TERTIARY,
            _t("$L_{K,x}$ (annual)", language),
            _t("limit", language),
            step,
            "-",
            False,
        ),
    )
    for (
        values, limits, verdicts, colour, label, limit_label, offset, dash, styled
    ) in series:
        # matplotlib types the dash style as a Literal; the series table above
        # carries it as a plain str, so narrow it back for the hlines call.
        style = cast("Any", dash)
        opts: dict[str, Any] = {"color": colour, "label": label}
        if styled:
            # The caller's bar kwargs apply to the daily LKeq,x series, the one
            # the fiche and the guides treat as the headline result.
            opts.update(kwargs)
        bars = ax.bar(positions + offset, values, width=width, **opts)
        for bar, ok in zip(bars, verdicts, strict=True):
            if ok is False:
                bar.set_edgecolor(_C_REFERENCE)
                bar.set_linewidth(1.6)
        for index, limit in enumerate(limits):
            ax.hlines(
                limit,
                positions[index] + offset - 0.62 * width,
                positions[index] + offset + 0.62 * width,
                colors=_C_REFERENCE, linestyles=style, linewidth=1.5,
                label=limit_label if index == 0 else "_nolegend_",
                zorder=4,
            )

    # The categorical axis here carries two or three period names, not a band
    # sequence, so the 45-degree band-axis rotation is not needed.
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_xlim(positions[0] - 0.5 - step, positions[-1] + 0.5 + step)

    ax.set_ylabel(_t("Corrected level [dB]", language))
    ax.set_title(_t("RD 1367/2007 assessment vs limit values", language))
    ax.legend(loc="upper right", fontsize="small", ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    localize_axes(ax, language)
    return ax
