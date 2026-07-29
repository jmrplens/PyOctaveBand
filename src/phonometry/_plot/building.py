#  Copyright (c) 2026. Jose M. Requena-Plens
"""Plot renderers for the building domain (lazy imports from result .plot())."""

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
    _C_SECONDARY_LIGHT,
    _C_TERTIARY,
    _annotate_impact_500,
    _band_axis,
    _facade_x_axis,
    _format_freq,
    _freq_axis,
    _new_axes,
    _plot_band_level_bars,
    _plot_insulation_bands,
    _plot_rating,
    _require_rating_curve,
    format_frequency_axis,
    theme_fill,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..building.aperture_transmission import ApertureTransmissionResult
    from ..building.building_prediction import (
        AirbornePredictionResult,
        ImpactPredictionResult,
    )
    from ..building.building_uncertainty import BandUncertainty
    from ..building.detailed_prediction import (
        DetailedAirborneResult,
        DetailedImpactResult,
        InSituElementResult,
    )
    from ..building.facade_prediction import FacadePredictionResult, RadiatedPowerResult
    from ..building.flanking_transmission import VibrationReductionResult
    from ..building.floor_covering_improvement import FloorCoveringImprovementResult
    from ..building.installed_structure_borne import InstalledSourceResult
    from ..building.insulation import (
        AirborneInsulationResult,
        ExtendedImpactRatingResult,
        ExtendedWeightedRatingResult,
        FacadeInsulationResult,
        ImpactInsulationResult,
        ImpactRatingResult,
        WeightedRatingResult,
    )
    from ..building.panel_transmission import SoundReductionResult
    from ..building.spanish_building_code import (
        DbHrAssessment,
        DbHrGlobalIndexResult,
    )
    from ..building.structure_borne_power import StructureBornePowerResult

#: Shared x-axis label for the frequency-domain building plots.
_FREQ_LABEL = "Frequency [Hz]"

#: Spanish translations of the fixed labels/titles/legends rendered by the
#: building-domain ``.plot()`` renderers, keyed by their verbatim English
#: text. ``_t`` returns the English key unchanged for any language other
#: than ``"es"``, so the English output is byte-for-byte identical to the
#: pre-i18n renderers.
_STRINGS: dict[str, str] = {
    "Frequency [Hz]": "Frecuencia [Hz]",
    "Band": "Banda",
    "Band index": "Índice de banda",
    "predicted $R$": "$R$ previsto",
    "Sound reduction index $R$ [dB]": "Índice de reducción acústica $R$ [dB]",
    "Predicted sound insulation": "Aislamiento acústico previsto",
    "coincidence plateau (A to B)": "meseta de coincidencia (A a B)",
    "aperture $R$": "$R$ de abertura",
    "Aperture sound transmission (Gomperts / Wilson-Soroka)": "Transmisión sonora por abertura (Gomperts / Wilson-Soroka)",
    "Sound reduction index [dB]": "Índice de reducción acústica [dB]",
    "Sigma unfav.": "Σ desfav.",
    "impact rating": "índice de impacto",
    "Impact sound pressure level [dB]": "Nivel de presión acústica de impactos [dB]",
    "Level difference / reduction index [dB]": "Diferencia de nivel / índice de reducción [dB]",
    "Façade sound insulation (ISO 16283-3)": "Aislamiento acústico de fachada (ISO 16283-3)",
    "Reduction index / level difference [dB]": "Índice de reducción / diferencia de nivel [dB]",
    "Façade insulation prediction (EN 12354-3)": "Predicción del aislamiento de fachada (EN 12354-3)",
    "Radiated sound power level [dB]": "Nivel de potencia acústica radiada [dB]",
    "Radiated sound power (EN 12354-4)": "Potencia acústica radiada (EN 12354-4)",
    "Vibration reduction index $K_{ij}$ [dB]": "Índice de reducción de vibraciones $K_{ij}$ [dB]",
    "Vibration reduction index (ISO 10848)": "Índice de reducción de vibraciones (ISO 10848)",
    "Structure-borne power level $L_{Ws}$ [dB re 1 pW]": "Nivel de potencia estructural $L_{Ws}$ [dB re 1 pW]",
    "EN 15657 characteristic structure-borne sound power": "Potencia acústica estructural característica EN 15657",
    "paths": "trayectos",
    "total $L_{n,s}$": "total $L_{n,s}$",
    "Normalised SPL $L_{n,s}$ [dB]": "NPS normalizado $L_{n,s}$ [dB]",
    "EN 12354-5 installed structure-borne sound": "Ruido estructural instalado EN 12354-5",
    "Transmission path": "Trayecto de transmisión",
    "Share of transmitted energy [%]": "Fracción de energía transmitida [%]",
    "flanking prediction": "predicción de transmisión por flancos",
    "Level / correction [dB]": "Nivel / corrección [dB]",
    "impact prediction": "predicción de impacto",
    "Airborne sound insulation (ISO 16283-1)": "Aislamiento a ruido aéreo (ISO 16283-1)",
    "Impact sound insulation (ISO 16283-2)": "Aislamiento a ruido de impacto (ISO 16283-2)",
    "Standard uncertainty u [dB]": "Incertidumbre típica u [dB]",
    "band uncertainty": "incertidumbre por banda",
    "situation": "situación",
    "sigma_R95 upper limit": "límite superior sigma_R95",
    "limit of measurement (> delta-L)": "límite de medición (> delta-L)",
    "enlarged range (Annex B)": "rango ampliado (Anexo B)",
    "enlarged range (A.2.1)": "rango ampliado (A.2.1)",
    "Measured": "Medido",
    "Shifted reference (core bands)": "Referencia desplazada (bandas 100-3150 Hz)",
    "Improvement of impact sound insulation delta-L [dB]": "Mejora del aislamiento a ruido de impacto delta-L [dB]",
    "ISO 16251-1 Floor-Covering Impact Sound Improvement": "Mejora del aislamiento a ruido de impacto de revestimiento de suelo ISO 16251-1",
    "band insulation": "aislamiento por banda",
    "transmitted level $L_{x,i} - X_i$": "nivel transmitido $L_{x,i} - X_i$",
    "Band insulation $X_i$ [dB]": "Aislamiento por banda $X_i$ [dB]",
    "Transmitted level [dBA]": "Nivel transmitido [dBA]",
    "pink noise": "ruido rosa",
    "road traffic": "ruido de automóviles",
    "railway": "ruido ferroviario",
    "aircraft": "ruido de aeronaves",
    "CTE DB-HR global index": "Índice global CTE DB-HR",
    "achieved": "obtenido",
    "required": "exigido",
    "Value": "Valor",
    "CTE DB-HR requirement check": "Comprobación de exigencias CTE DB-HR",
    "detailed prediction": "predicción detallada",
    "other paths": "otros trayectos",
    "In-situ element performance (ISO 12354)": "Comportamiento del elemento in situ (ISO 12354)",
    "Reduction index / impact level [dB]": "Índice de reducción / nivel de impactos [dB]",
}


def _t(text: str, language: str = "en") -> str:
    """Localise a fixed string; English is returned verbatim (byte-identical)."""
    return _STRINGS.get(text, text) if language == "es" else text


def plot_sound_reduction(
    result: SoundReductionResult, ax: Axes | None = None, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Predicted sound reduction index ``R(f)`` (Bies 7.2).

    :param result: A
        :class:`~phonometry.building.panel_transmission.SoundReductionResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the ``R(f)`` curve ``plot``.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    freq = np.asarray(result.frequencies, dtype=np.float64)
    r = np.asarray(result.transmission_loss, dtype=np.float64)
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("markersize", 3)
    ax.semilogx(freq, r, label=_t("predicted $R$", language), **kwargs)
    if result.critical_frequency is not None:
        ax.axvline(
            result.critical_frequency, color=_C_REFERENCE, ls="--", lw=1.0,
            label=f"$f_c$ = {format_number(result.critical_frequency, language, decimals=0)} Hz",
        )
    if result.resonance_frequency is not None:
        ax.axvline(
            result.resonance_frequency, color=_C_SECONDARY, ls="--", lw=1.0,
            label=f"$f_0$ = {format_number(result.resonance_frequency, language, decimals=0)} Hz",
        )
    if result.plateau_start is not None and result.plateau_end is not None:
        # Shade the coincidence plateau of the Watters construction, whose two
        # construction points A and B are what the whole estimate hangs on.
        ax.axvspan(
            result.plateau_start, result.plateau_end,
            color=theme_fill(_C_SECONDARY, ax), lw=0, zorder=0,
            label=_t("coincidence plateau (A to B)", language),
        )
    format_frequency_axis(ax, float(freq.min()), float(freq.max()))
    ax.set_xlabel(_t(_FREQ_LABEL, language))
    ax.set_ylabel(_t("Sound reduction index $R$ [dB]", language))
    ax.set_title(f"{_t('Predicted sound insulation', language)} ({result.model})")
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, which="both", alpha=0.3)
    localize_axes(ax, language)
    return ax


def plot_aperture_transmission(
    result: ApertureTransmissionResult, ax: Axes | None = None,
    language: str = "en", **kwargs: Any
) -> Axes:
    """Aperture sound reduction index ``R(f) = -10 lg(tau)`` (Hopkins 4.3.10).

    :param result: An
        :class:`~phonometry.building.aperture_transmission.ApertureTransmissionResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the ``R(f)`` curve ``plot``.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    freq = np.asarray(result.frequencies, dtype=np.float64)
    r = np.asarray(result.transmission_loss, dtype=np.float64)
    kwargs.setdefault("color", _C_PRIMARY)
    ax.semilogx(freq, r, label=f"{result.kind} {_t('aperture $R$', language)}", **kwargs)
    ax.axhline(0.0, color=_C_MUTED, ls=":", lw=0.9)
    format_frequency_axis(ax, float(freq.min()), float(freq.max()))
    ax.set_xlabel(_t(_FREQ_LABEL, language))
    ax.set_ylabel(_t("Sound reduction index $R$ [dB]", language))
    ax.set_title(_t("Aperture sound transmission (Gomperts / Wilson-Soroka)", language))
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, which="both", alpha=0.3)
    localize_axes(ax, language)
    return ax


def plot_weighted_rating(
    result: WeightedRatingResult, ax: Axes | None = None, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Airborne rating curve vs shifted reference (ISO 717-1).

    :param result: A :class:`~phonometry.insulation.WeightedRatingResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the measured-curve ``plot`` call.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    _require_rating_curve(result)
    ax = _plot_rating(
        np.asarray(result.band_centers, dtype=np.float64),
        np.asarray(result.measured, dtype=np.float64),
        np.asarray(result.shifted_reference, dtype=np.float64),
        impact=False,
        title=(
            # Sign only when negative, the style of ISO 717-1's own examples.
            f"ISO 717-1 Rw (C={result.c:d}; Ctr={result.ctr:d}) = "
            f"{result.rating} dB  ({_t('Sigma unfav.', language)} = "
            f"{format_number(result.unfavourable_sum, language, decimals=1)} dB)"
        ),
        ylabel=_t("Sound reduction index [dB]", language),
        ax=ax,
        language=language,
        **kwargs,
    )
    localize_axes(ax, language)
    return ax


def plot_impact_rating(
    result: ImpactRatingResult, ax: Axes | None = None, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Impact rating curve vs shifted reference (ISO 717-2).

    The drawn shifted-reference curve is the normatively honest ``ref -
    shift``; for octave-band data the rating is that curve read at 500 Hz
    *minus 5 dB* (Clause 4.3.2), so the plot marks the 500 Hz read value on
    the (undistorted) curve and annotates the -5 dB reduction rather than
    pulling the curve down to the rating.

    :param result: An :class:`~phonometry.insulation.ImpactRatingResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the measured-curve ``plot`` call.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    _require_rating_curve(result)
    band_centers = np.asarray(result.band_centers, dtype=np.float64)
    reference = np.asarray(result.shifted_reference, dtype=np.float64)
    ax = _plot_rating(
        band_centers,
        np.asarray(result.measured, dtype=np.float64),
        reference,
        impact=True,
        # The rated quantity depends on the input (Ln,w, L'n,w or L'nT,w);
        # the dataclass does not carry which, so the figure uses the neutral
        # "impact rating" label rather than hard-coding one specific symbol.
        title=(
            # Sign only when negative, the style of ISO 717-2's own examples.
            f"ISO 717-2 {_t('impact rating', language)} (CI={result.ci:d}) = "
            f"{result.rating} dB  ({_t('Sigma unfav.', language)} = "
            f"{format_number(result.unfavourable_sum, language, decimals=1)} dB)"
        ),
        ylabel=_t("Impact sound pressure level [dB]", language),
        ax=ax,
        language=language,
        **kwargs,
    )
    _annotate_impact_500(
        ax, band_centers, reference, int(result.rating), language=language
    )
    localize_axes(ax, language)
    return ax


def _require_extended_curve(
    result: ExtendedWeightedRatingResult | ExtendedImpactRatingResult,
) -> None:
    if result.band_centers is None or result.measured is None:
        raise ValueError(
            "This extended rating result carries no band curve to plot (it "
            "was constructed without band_centers/measured data)."
        )


def _plot_extended_rating(
    result: ExtendedWeightedRatingResult | ExtendedImpactRatingResult,
    *,
    impact: bool,
    title: str,
    ylabel: str,
    span_label: str,
    ax: Axes | None,
    language: str,
    **kwargs: Any,
) -> Axes:
    """Shared renderer for the two enlarged-range ISO 717 rating plots."""
    from .._i18n import localize_axes
    from .common import _t as _t_common

    _require_extended_curve(result)
    _require_rating_curve(result.core)
    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.band_centers, dtype=np.float64)
    measured = np.asarray(result.measured, dtype=np.float64)
    core_freqs = np.asarray(result.core.band_centers, dtype=np.float64)
    core_measured = np.asarray(result.core.measured, dtype=np.float64)
    core_ref = np.asarray(result.core.shifted_reference, dtype=np.float64)

    # Mark the bands outside the 100-3150 Hz core as the enlarged range.
    if float(freqs.min()) < float(core_freqs.min()):
        ax.axvspan(float(freqs.min()), float(core_freqs.min()),
                   color=_C_MUTED, alpha=0.12, label=_t(span_label, language))
    if float(freqs.max()) > float(core_freqs.max()):
        label = None if float(freqs.min()) < float(core_freqs.min()) else _t(
            span_label, language
        )
        ax.axvspan(float(core_freqs.max()), float(freqs.max()),
                   color=_C_MUTED, alpha=0.12, label=label)

    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("label", _t("Measured", language))
    ax.plot(freqs, measured, "o-", **kwargs)
    ax.plot(core_freqs, core_ref, "s--", color=_C_REFERENCE,
            label=_t("Shifted reference (core bands)", language))
    unfavourable = (
        core_measured > core_ref if impact else core_measured < core_ref
    )
    ax.fill_between(
        core_freqs, core_measured, core_ref, where=unfavourable.tolist(),
        color=_C_SECONDARY, alpha=0.4,
        label=_t_common("Unfavourable deviations", language), interpolate=True,
    )
    _freq_axis(ax, freqs, language=language)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    localize_axes(ax, language)
    return ax


def _extended_terms_line(
    terms: list[tuple[str, float | None]], language: str, decimals: int
) -> str:
    """Format the available Annex B adaptation terms as ``name = value dB``."""
    from .._i18n import format_number

    parts = [
        f"{name} = {format_number(value, language, decimals=decimals)}"
        for name, value in terms
        if value is not None
    ]
    return "; ".join(parts)


def plot_extended_weighted_rating(
    result: ExtendedWeightedRatingResult, ax: Axes | None = None,
    language: str = "en", **kwargs: Any
) -> Axes:
    """Enlarged-range airborne rating curve vs shifted reference (ISO 717-1 Annex B).

    :param result: An
        :class:`~phonometry.insulation.ExtendedWeightedRatingResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the measured-curve ``plot`` call.
    :return: The axes.
    """
    from .._i18n import format_number

    decimals = 0 if float(result.rating).is_integer() else 1
    title = (
        f"ISO 717-1 Rw (C={format_number(result.c, language, decimals=decimals)}; "
        f"Ctr={format_number(result.ctr, language, decimals=decimals)}) = "
        f"{format_number(result.rating, language, decimals=decimals)} dB"
    )
    extended = _extended_terms_line(
        [
            ("C50-3150", result.c_50_3150),
            ("C50-5000", result.c_50_5000),
            ("C100-5000", result.c_100_5000),
            ("Ctr,50-3150", result.ctr_50_3150),
            ("Ctr,50-5000", result.ctr_50_5000),
            ("Ctr,100-5000", result.ctr_100_5000),
        ],
        language, decimals,
    )
    if extended:
        title = f"{title}\n{extended}"
    return _plot_extended_rating(
        result, impact=False, title=title,
        ylabel=_t("Sound reduction index [dB]", language),
        span_label="enlarged range (Annex B)", ax=ax, language=language,
        **kwargs,
    )


def plot_extended_impact_rating(
    result: ExtendedImpactRatingResult, ax: Axes | None = None,
    language: str = "en", **kwargs: Any
) -> Axes:
    """Enlarged-range impact rating curve vs shifted reference (ISO 717-2 A.2.1).

    :param result: An
        :class:`~phonometry.insulation.ExtendedImpactRatingResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the measured-curve ``plot`` call.
    :return: The axes.
    """
    from .._i18n import format_number

    decimals = 0 if float(result.rating).is_integer() else 1
    title = (
        f"ISO 717-2 {_t('impact rating', language)} "
        f"(CI={format_number(result.ci, language, decimals=decimals)}) = "
        f"{format_number(result.rating, language, decimals=decimals)} dB"
    )
    if result.ci_50_2500 is not None:
        title = (
            f"{title}\nCI,50-2500 = "
            f"{format_number(result.ci_50_2500, language, decimals=decimals)}"
        )
    return _plot_extended_rating(
        result, impact=True, title=title,
        ylabel=_t("Impact sound pressure level [dB]", language),
        span_label="enlarged range (A.2.1)", ax=ax, language=language,
        **kwargs,
    )


def plot_facade_insulation(
    result: FacadeInsulationResult, ax: Axes | None = None, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Per-band façade sound-insulation profile (ISO 16283-3).

    Draws the standardized level difference ``D2m,nT`` first, then the
    other available quantities (``D2m``, ``D2m,n``, ``R'``) against
    frequency. Works for
    :class:`~phonometry.insulation.FacadeInsulationResult`.

    :param result: A façade result exposing ``d_2m``, ``d_2m_nt``,
        ``d_2m_n``, ``r_prime`` and (optionally) ``frequencies``.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the primary ``D2m,nT`` curve ``plot`` call.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    dnt = np.asarray(result.d_2m_nt, dtype=np.float64)
    n = dnt.size
    x = _facade_x_axis(
        ax, getattr(result, "frequencies", None), n, language=language
    )

    # D2m,nT first so it is lines[0]; other quantities follow when present.
    curves = [("$D_{2m,nT}$", dnt)]
    curves.append(("$D_{2m}$", np.asarray(result.d_2m, dtype=np.float64)))
    if result.d_2m_n is not None:
        curves.append(("$D_{2m,n}$", np.asarray(result.d_2m_n, dtype=np.float64)))
    if result.r_prime is not None:
        curves.append(("$R'$", np.asarray(result.r_prime, dtype=np.float64)))
    # Forward user kwargs to the primary D2m,nT curve only, so styling kwargs
    # (label=, color=) neither collide with the per-curve labels nor make the
    # companion curves indistinguishable.
    for index, (label, y) in enumerate(curves):
        opts: dict[str, Any] = {"label": label}
        if index == 0:
            opts.update(kwargs)
        ax.plot(x, y, "o-", **opts)

    ax.set_ylabel(_t("Level difference / reduction index [dB]", language))
    ax.set_title(_t("Façade sound insulation (ISO 16283-3)", language))
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, alpha=0.3)
    localize_axes(ax, language)
    return ax


def plot_facade_prediction(
    result: FacadePredictionResult, ax: Axes | None = None, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Predicted façade insulation profile (EN 12354-3:2000).

    Draws the per-element partial indices ``Rp = -10 lg τ`` as thin dashed
    lines, then the façade apparent reduction ``R'`` and the standardized
    level difference ``D2m,nT`` as bold curves, against frequency. Works for
    :class:`~phonometry.facade_prediction.FacadePredictionResult`.

    :param result: A façade prediction result exposing ``r_prime``,
        ``d_2m_nt``, ``element_r`` and (optionally) ``frequencies``.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the primary ``R'`` curve ``plot`` call.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    r_prime = np.asarray(result.r_prime, dtype=np.float64)
    n = r_prime.size
    x = _facade_x_axis(ax, result.frequencies, n, language=language)

    for name, rp in result.element_r.items():
        ax.plot(x, np.asarray(rp, dtype=np.float64), "--", lw=0.9, alpha=0.6, label=name)

    opts: dict[str, Any] = {"label": "$R'$", "color": "black", "lw": 2.0}
    opts.update(kwargs)
    ax.plot(x, r_prime, "o-", **opts)
    ax.plot(
        x,
        np.asarray(result.d_2m_nt, dtype=np.float64),
        "s-",
        color="tab:blue",
        lw=2.0,
        label="$D_{2m,nT}$",
    )

    ax.set_ylabel(_t("Reduction index / level difference [dB]", language))
    ax.set_title(_t("Façade insulation prediction (EN 12354-3)", language))
    ax.legend(loc="best", fontsize="small", ncol=2)
    ax.grid(True, alpha=0.3)
    localize_axes(ax, language)
    return ax


def plot_radiated_power(
    result: RadiatedPowerResult, ax: Axes | None = None, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Radiated sound power level ``LW`` per band (EN 12354-4:2000).

    Draws the segment radiated power level as bars, annotating the A-weighted
    single number when available. Works for
    :class:`~phonometry.facade_prediction.RadiatedPowerResult`.

    :param result: A :class:`~phonometry.facade_prediction.RadiatedPowerResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the ``bar`` call.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    l_w = np.asarray(result.l_w, dtype=np.float64)
    n = l_w.size
    positions = np.arange(n, dtype=np.float64)
    if result.frequencies is None:
        labels = [f"{_t('Band', language)} {i + 1}" for i in range(n)]
    else:
        labels = [_format_freq(f) for f in np.asarray(result.frequencies, dtype=np.float64)]

    opts: dict[str, Any] = {"color": "tab:red", "alpha": 0.8, "label": "$L_W$"}
    opts.update(kwargs)
    ax.bar(positions, l_w, **opts)
    _band_axis(ax, labels, xlabel=_t(_FREQ_LABEL, language), language=language)

    if result.l_w_dba is not None:
        ax.axhline(
            result.l_w_dba,
            color="black",
            ls="--",
            lw=1.2,
            label=f"$L_{{WA}}$ = {format_number(result.l_w_dba, language, decimals=1)} dB(A)",
        )
    ax.set_ylabel(_t("Radiated sound power level [dB]", language))
    ax.set_title(_t("Radiated sound power (EN 12354-4)", language))
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, axis="y", alpha=0.3)
    localize_axes(ax, language)
    return ax


def plot_vibration_reduction(
    result: VibrationReductionResult, ax: Axes | None = None, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Vibration reduction index ``Kij`` versus frequency (ISO 10848).

    Draws the per-band ``Kij`` and, when available, a dashed line at the
    single-number mean ``K̄ij`` (200-1250 Hz, Annex A). Falls back to a
    band-index axis when the result carries no frequencies.

    :param result: A
        :class:`~phonometry.flanking_transmission.VibrationReductionResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the ``Kij`` curve ``plot`` call.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    k_ij = np.asarray(result.k_ij, dtype=np.float64)
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("label", "$K_{ij}$")
    if result.frequencies is not None:
        freqs = np.asarray(result.frequencies, dtype=np.float64)
        ax.plot(freqs, k_ij, **kwargs)
        _freq_axis(ax, freqs, language=language)
    else:
        ax.plot(np.arange(k_ij.size), k_ij, **kwargs)
        ax.set_xlabel(_t("Band index", language))
    if result.single_number is not None:
        ax.axhline(
            result.single_number,
            color=_C_REFERENCE,
            ls="--",
            lw=1.0,
            label=rf"$\overline{{K}}_{{ij}}$ = {format_number(result.single_number, language, decimals=1)} dB",
        )
    ax.set_ylabel(_t("Vibration reduction index $K_{ij}$ [dB]", language))
    ax.set_title(_t("Vibration reduction index (ISO 10848)", language))
    ax.grid(True, alpha=0.3)
    ax.legend()
    localize_axes(ax, language)
    return ax


def plot_structure_borne_power(
    result: StructureBornePowerResult, ax: Axes | None = None, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Characteristic structure-borne sound power level per band (EN 15657).

    :param result: A :class:`~phonometry.structure_borne_power.StructureBornePowerResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the bar ``plot``.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = _plot_band_level_bars(
        ax, result.power_level, result.frequencies, result.total_level,
        ylabel=_t(r"Structure-borne power level $L_{Ws}$ [dB re 1 pW]", language),
        title=_t("EN 15657 characteristic structure-borne sound power", language),
        language=language,
        **kwargs,
    )
    localize_axes(ax, language)
    return ax


def plot_installed_structure_borne(
    result: InstalledSourceResult, ax: Axes | None = None, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Per-path and total normalised structure-borne SPL (EN 12354-5).

    :param result: An :class:`~phonometry.installed_structure_borne.InstalledSourceResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the total-level ``plot``.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    paths = np.atleast_2d(np.asarray(result.path_levels, dtype=np.float64))
    total = np.atleast_1d(np.asarray(result.total_level, dtype=np.float64))
    n_bands = total.size
    if result.frequencies is not None:
        x = np.asarray(result.frequencies, dtype=np.float64)
        ax.set_xscale("log")
        ax.set_xlabel(_t(_FREQ_LABEL, language))
    else:
        x = np.arange(1, n_bands + 1, dtype=np.float64)
        ax.set_xlabel(_t("Band", language))
    for k, path in enumerate(paths):
        ax.plot(x, path, color=_C_MUTED, lw=1.0, ls=":", marker=".",
                label=_t("paths", language) if k == 0 else None)
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("lw", 2.2)
    ax.plot(x, total, label=_t(r"total $L_{n,s}$", language), **kwargs)
    ax.set_ylabel(_t(r"Normalised SPL $L_{n,s}$ [dB]", language))
    ax.set_title(_t("EN 12354-5 installed structure-borne sound", language))
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, which="both", alpha=0.3)
    if result.frequencies is not None:
        format_frequency_axis(ax, float(x.min()), float(x.max()))
    localize_axes(ax, language)
    return ax


def plot_airborne_prediction(
    result: AirbornePredictionResult, ax: Axes | None = None, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Per-path shares of the transmitted energy (EN 12354-1).

    One bar per transmission path (direct plus flanking), sorted by its
    share of the total transmitted sound energy, largest first.

    :param result: An
        :class:`~phonometry.building_prediction.AirbornePredictionResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the path :meth:`~matplotlib.axes.Axes.bar`.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    contribs = sorted(result.paths, key=lambda c: c.fraction, reverse=True)
    shares = [100.0 * c.fraction for c in contribs]
    positions = np.arange(len(shares), dtype=np.float64)
    kwargs.setdefault(
        "color", [_C_PRIMARY if c.kind == "Dd" else _C_MUTED for c in contribs]
    )
    ax.bar(positions, shares, **kwargs)
    ax.set_xticks(positions)
    ax.set_xticklabels([c.label for c in contribs], rotation=45, ha="right")
    ax.set_xlabel(_t("Transmission path", language))
    ax.set_ylabel(_t("Share of transmitted energy [%]", language))
    ax.set_title(
        f"EN 12354-1 {_t('flanking prediction', language)} — R'w = "
        f"{format_number(result.r_prime_w, language, decimals=1)} dB "
        f"(RDd,w = {format_number(result.r_direct_w, language, decimals=1)} dB)"
    )
    ax.grid(True, axis="y", alpha=0.3)
    localize_axes(ax, language)
    return ax


def plot_impact_prediction(
    result: ImpactPredictionResult, ax: Axes | None = None, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Terms of the apparent impact-level prediction (EN 12354-2).

    Bars for the Formula 21 terms (the bare-floor equivalent level, the
    covering improvement, the flanking correction) and the resulting
    apparent level ``L'n,w = Ln,w,eq - DLw + K``.

    :param result: An
        :class:`~phonometry.building_prediction.ImpactPredictionResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the term :meth:`~matplotlib.axes.Axes.bar`.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    labels = ("$L_{n,w,eq}$", r"$-\Delta L_w$", "$+K$", "$L'_{n,w}$")
    values = (
        result.ln_w_eq,
        -result.delta_l_w,
        result.k_correction,
        result.l_prime_n_w,
    )
    positions = np.arange(len(values), dtype=np.float64)
    kwargs.setdefault("color", [_C_MUTED, _C_TERTIARY, _C_SECONDARY, _C_PRIMARY])
    ax.bar(positions, values, **kwargs)
    ax.axhline(0.0, color=_C_MUTED, lw=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(_t("Level / correction [dB]", language))
    ax.set_title(
        f"EN 12354-2 {_t('impact prediction', language)} — L'n,w = "
        f"{format_number(result.l_prime_n_w, language, decimals=1)} dB"
    )
    ax.grid(True, axis="y", alpha=0.3)
    localize_axes(ax, language)
    return ax


#: Maximum number of individually coloured paths in the detailed-prediction
#: stacked bars; the remaining ones are pooled into a single "other paths" bar.
_MAX_NAMED_PATHS = 6


def _plot_path_shares(
    result: Any,
    total: np.ndarray,
    *,
    total_label: str,
    ylabel: str,
    title: str,
    ax: Axes | None,
    language: str,
    **kwargs: Any,
) -> Axes:
    """Stacked per-band path shares with the resulting total on a twin axis.

    Shared body of the two detailed-model renderers (EN/ISO 12354-1 airborne
    and -2 impact): one stacked bar per band showing which transmission path
    carries the energy, plus the resulting apparent quantity on a right-hand
    axis. The paths are ordered by their *largest* per-band share, so every
    path that dominates a band is named, and any beyond
    :data:`_MAX_NAMED_PATHS` are pooled.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    fractions = np.atleast_2d(np.asarray(result.fractions, dtype=np.float64))
    labels = [p.label for p in result.paths]
    order = list(np.argsort(-fractions.max(axis=1)))
    named = order[:_MAX_NAMED_PATHS]
    pooled = order[_MAX_NAMED_PATHS:]

    positions = _band_axis(ax, np.asarray(result.frequencies), language=language)
    palette = (_C_PRIMARY, _C_SECONDARY, _C_TERTIARY, _C_QUATERNARY,
               _C_PRIMARY_LIGHT, _C_SECONDARY_LIGHT)
    bottom = np.zeros(positions.size, dtype=np.float64)
    for colour, k in zip(palette, named):
        share = 100.0 * fractions[k]
        ax.bar(positions, share, bottom=bottom, width=0.85, color=colour,
               edgecolor="none", zorder=0, label=labels[k], **kwargs)
        bottom = bottom + share
    if pooled:
        share = 100.0 * fractions[pooled].sum(axis=0)
        ax.bar(positions, share, bottom=bottom, width=0.85, color=_C_MUTED,
               edgecolor="none", zorder=0, label=_t("other paths", language))
    ax.set_ylabel(_t("Share of transmitted energy [%]", language))
    ax.set_ylim(0.0, 100.0)
    ax.set_title(title)

    twin = ax.twinx()
    twin.plot(positions, np.asarray(total, dtype=np.float64), color=_C_REFERENCE,
              lw=2.0, marker="o", ms=4, label=total_label, zorder=3)
    twin.set_ylabel(_t(ylabel, language))
    handles, texts = ax.get_legend_handles_labels()
    extra_handles, extra_texts = twin.get_legend_handles_labels()
    ax.legend(handles + extra_handles, texts + extra_texts, loc="best",
              fontsize="small", ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    localize_axes(ax, language)
    localize_axes(twin, language)
    return ax


def plot_detailed_airborne_prediction(
    result: DetailedAirborneResult, ax: Axes | None = None, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Per-band path contributions and ``R'`` (EN/ISO 12354-1 detailed model).

    :param result: A
        :class:`~phonometry.building.detailed_prediction.DetailedAirborneResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the stacked :meth:`~matplotlib.axes.Axes.bar`.
    :return: The axes.
    """
    from .._i18n import format_number

    title = f"EN 12354-1 {_t('detailed prediction', language)}"
    if result.rating is not None:
        title += (
            f" — R'w = {format_number(result.rating.rating, language, decimals=0)}"
            " dB"
        )
    return _plot_path_shares(
        result, result.r_prime, total_label="$R'$",
        ylabel="Sound reduction index [dB]", title=title, ax=ax,
        language=language, **kwargs,
    )


def plot_detailed_impact_prediction(
    result: DetailedImpactResult, ax: Axes | None = None, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Per-band path contributions and ``L'n`` (EN/ISO 12354-2 detailed model).

    :param result: A
        :class:`~phonometry.building.detailed_prediction.DetailedImpactResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the stacked :meth:`~matplotlib.axes.Axes.bar`.
    :return: The axes.
    """
    from .._i18n import format_number

    title = f"EN 12354-2 {_t('detailed prediction', language)}"
    if result.rating is not None:
        title += (
            f" — L'n,w = "
            f"{format_number(result.rating.rating, language, decimals=0)} dB"
        )
    return _plot_path_shares(
        result, result.l_prime_n, total_label="$L'_n$",
        ylabel="Impact sound pressure level [dB]", title=title, ax=ax,
        language=language, **kwargs,
    )


def plot_in_situ_element(
    result: InSituElementResult, ax: Axes | None = None, language: str = "en",
    **kwargs: Any
) -> Axes:
    """In-situ ``Rsitu`` and ``Ln,situ`` of one element (EN/ISO 12354).

    :param result: An
        :class:`~phonometry.building.detailed_prediction.InSituElementResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the ``Rsitu`` curve ``plot`` call.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("ms", 4)
    ax.plot(freqs, result.sound_reduction_index, label="$R_{situ}$", **kwargs)
    ax.plot(freqs, result.impact_level, color=_C_SECONDARY, marker="s", ms=4,
            label="$L_{n,situ}$")
    ax.set_xscale("log")
    ax.set_xlabel(_t(_FREQ_LABEL, language))
    ax.set_ylabel(_t("Reduction index / impact level [dB]", language))
    ax.set_title(
        f"{_t('In-situ element performance (ISO 12354)', language)} — "
        f"{result.label}"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    format_frequency_axis(ax, float(freqs.min()), float(freqs.max()))
    localize_axes(ax, language)
    return ax


def plot_airborne_insulation(
    result: AirborneInsulationResult, ax: Axes | None = None, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Per-band airborne insulation quantities (ISO 16283-1).

    Draws the standardized level difference ``DnT`` first (the primary
    curve), then the level difference ``D`` and, when available, the
    apparent sound reduction index ``R'``.

    :param result: An :class:`~phonometry.insulation.AirborneInsulationResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the primary ``DnT`` curve ``plot`` call.
    :return: The axes.
    """
    from .._i18n import localize_axes

    curves = [
        ("$D_{nT}$", np.asarray(result.dnt, dtype=np.float64)),
        ("$D$", np.asarray(result.d, dtype=np.float64)),
    ]
    if result.r_prime is not None:
        curves.append(("$R'$", np.asarray(result.r_prime, dtype=np.float64)))
    ax = _plot_insulation_bands(
        curves,
        ylabel=_t("Level difference / reduction index [dB]", language),
        title=_t("Airborne sound insulation (ISO 16283-1)", language),
        ax=ax,
        **kwargs,
    )
    localize_axes(ax, language)
    return ax


def plot_impact_insulation(
    result: ImpactInsulationResult, ax: Axes | None = None, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Per-band impact sound pressure levels (ISO 16283-2).

    Draws the standardized level ``L'nT`` first (the primary curve) and,
    when available, the normalized level ``L'n``.

    :param result: An :class:`~phonometry.insulation.ImpactInsulationResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the primary ``L'nT`` curve ``plot`` call.
    :return: The axes.
    """
    from .._i18n import localize_axes

    curves = [("$L'_{nT}$", np.asarray(result.l_n_t, dtype=np.float64))]
    if result.l_n is not None:
        curves.append(("$L'_n$", np.asarray(result.l_n, dtype=np.float64)))
    ax = _plot_insulation_bands(
        curves,
        ylabel=_t("Impact sound pressure level [dB]", language),
        title=_t("Impact sound insulation (ISO 16283-2)", language),
        ax=ax,
        **kwargs,
    )
    localize_axes(ax, language)
    return ax


def plot_band_uncertainty(
    result: BandUncertainty, ax: Axes | None = None, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Per-band standard uncertainty of an insulation quantity (ISO 12999-1).

    :param result: A
        :class:`~phonometry.building_uncertainty.BandUncertainty`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the uncertainty curve ``plot`` call.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    freqs, u = result.to_arrays()
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("marker", "o")
    ax.plot(freqs, u, **kwargs)
    _freq_axis(ax, freqs, language=language)
    ax.set_ylabel(_t("Standard uncertainty u [dB]", language))
    ax.set_ylim(bottom=0.0)
    quantity = _t("sigma_R95 upper limit", language) if result.upper_limit else "u"
    ax.set_title(
        f"ISO 12999-1 {_t('band uncertainty', language)} ({quantity}) — "
        f"{result.measurand}, {_t('situation', language)} {result.situation}"
    )
    ax.grid(True, which="both", alpha=0.3)
    localize_axes(ax, language)
    return ax


def plot_floor_covering_improvement(
    result: FloorCoveringImprovementResult, ax: Axes | None = None,
    language: str = "en", **kwargs: Any
) -> Axes:
    """Impact-sound improvement spectrum ΔL of a floor covering (ISO 16251-1).

    :param result: A
        :class:`~phonometry.floor_covering_improvement.FloorCoveringImprovementResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the improvement-curve ``plot`` call.
    :return: The axes.
    """
    from .._i18n import decimal_comma, localize_axes

    ax = ax if ax is not None else _new_axes()
    freqs, dl = result.frequencies, result.improvement
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("marker", "o")
    ax.plot(freqs, dl, **kwargs)
    # Mark bands at the limit of measurement (reported as > delta-L).
    if result.limited.size and bool(np.any(result.limited)):
        ax.plot(
            freqs[result.limited], dl[result.limited], ls="", marker="v",
            color=_C_SECONDARY, ms=9, mfc="none", mew=1.6, zorder=5,
            label=_t("limit of measurement (> delta-L)", language),
        )
    _freq_axis(ax, freqs, language=language)
    ax.set_ylabel(_t("Improvement of impact sound insulation delta-L [dB]", language))
    ax.set_ylim(bottom=0.0)
    title = _t("ISO 16251-1 Floor-Covering Impact Sound Improvement", language)
    if result.delta_lw is not None:
        title += f"  (delta-Lw = {decimal_comma(str(result.delta_lw), language)} dB)"
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    localize_axes(ax, language)
    return ax


#: Localised names of the DB-HR normalised source spectra.
_DB_HR_SPECTRUM_LABELS = {
    "pink": "pink noise",
    "traffic": "road traffic",
    "railway": "railway",
    "aircraft": "aircraft",
}


def plot_db_hr_global_index(
    result: DbHrGlobalIndexResult, ax: Axes | None = None, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Band insulation and per-band transmitted level of a DB-HR global index.

    The band insulation ``X_i`` is drawn as bars and the weighted per-band
    transmitted level ``L_x,i - X_i`` as a line: the global index is minus the
    energy sum of that line, so the bands where it peaks are the ones that set
    the index.

    :param result: A
        :class:`~phonometry.building.spanish_building_code.DbHrGlobalIndexResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the band-insulation ``bar`` call.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    values = np.asarray(result.band_values, dtype=np.float64)
    contributions = np.asarray(result.band_contributions, dtype=np.float64)
    positions = _band_axis(ax, freqs, language=language)

    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("alpha", 0.85)
    kwargs.setdefault("label", _t("band insulation", language))
    ax.bar(positions, values, width=0.72, **kwargs)
    ax.set_ylabel(_t("Band insulation $X_i$ [dB]", language))

    # The two quantities live on disjoint ranges (a positive insulation of
    # tens of dB against a negative weighted level), so the transmitted level
    # goes on its own axis; the bands where it peaks are the ones that set the
    # index, which is minus the energy sum of that curve.
    twin = ax.twinx()
    twin.plot(positions, contributions, "o-", color=_C_SECONDARY, ms=4.0,
              label=_t("transmitted level $L_{x,i} - X_i$", language))
    twin.set_ylabel(_t("Transmitted level [dBA]", language))

    spectrum = _t(_DB_HR_SPECTRUM_LABELS[result.spectrum], language)
    ax.set_title(
        f"{_t('CTE DB-HR global index', language)}: {result.name} = "
        f"{format_number(result.value, language, decimals=1)} dBA ({spectrum})"
    )
    handles, labels = ax.get_legend_handles_labels()
    extra_handles, extra_labels = twin.get_legend_handles_labels()
    ax.legend(handles + extra_handles, labels + extra_labels,
              loc="upper left", fontsize="small")
    ax.grid(True, axis="y", alpha=0.3)
    localize_axes(ax, language)
    localize_axes(twin, language)
    return ax


def plot_db_hr_assessment(
    result: DbHrAssessment, ax: Axes | None = None, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Achieved values against their CTE DB-HR requirements.

    Each check is a horizontal lollipop from its requirement to the achieved
    value: a stem reaching to the right of the requirement marker means a
    compliant "at least" requirement, and the exceedance colour marks a check
    that is not met.

    :param result: A
        :class:`~phonometry.building.spanish_building_code.DbHrAssessment`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the achieved-value ``scatter`` call.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    checks = list(result.checks)
    rows = np.arange(len(checks), dtype=np.float64)
    achieved = np.array([c.reported for c in checks], dtype=np.float64)
    limits = np.array([c.requirement.limit for c in checks], dtype=np.float64)
    colours = [_C_TERTIARY if c.complies else _C_REFERENCE for c in checks]

    ax.hlines(rows, limits, achieved, colors=colours, linewidth=2.0)
    ax.scatter(limits, rows, marker="|", s=220, color=_C_MUTED,
               label=_t("required", language), zorder=4)
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("s", 55)
    kwargs.setdefault("label", _t("achieved", language))
    ax.scatter(achieved, rows, color=colours, zorder=5, **kwargs)

    ax.set_yticks(rows)
    ax.set_yticklabels(
        [f"{c.requirement.quantity} ({c.requirement.unit})" for c in checks]
    )
    ax.invert_yaxis()
    ax.set_xlabel(_t("Value", language))
    ax.set_title(_t("CTE DB-HR requirement check", language))
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, axis="x", alpha=0.3)
    localize_axes(ax, language)
    return ax
