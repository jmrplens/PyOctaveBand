#  Copyright (c) 2026. Jose M. Requena-Plens
"""Plot renderers for the environmental domain (lazy imports from result .plot())."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
    from ..environmental.ground_barriers import (
        BarrierInsertionLoss,
        SphericalGroundResult,
    )
    from ..environmental.impulse_prominence import ImpulseProminenceResult
    from ..environmental.measurement import TonalAssessmentResult
    from ..environmental.outdoor_propagation import OutdoorAttenuation
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
