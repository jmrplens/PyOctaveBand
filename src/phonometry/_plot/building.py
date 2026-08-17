#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Plot renderers for the building domain (lazy imports from result .plot())."""

from __future__ import annotations

from collections.abc import Sequence
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

    from ..building.measurement.flanking_transmission import VibrationReductionResult
    from ..building.measurement.floor_covering_improvement import (
        FloorCoveringImprovementResult,
    )
    from ..building.measurement.heavy_impact import (
        AWeightedMaximumImpactResult,
        HeavyImpactSourceCheck,
        StandardizedMaximumImpactResult,
    )
    from ..building.measurement.insulation import (
        AirborneInsulationResult,
        ExtendedImpactRatingResult,
        ExtendedWeightedRatingResult,
        FacadeInsulationResult,
        ImpactInsulationResult,
        ImpactRatingResult,
        WeightedRatingResult,
    )
    from ..building.measurement.structure_borne_power import StructureBornePowerResult
    from ..building.measurement.uncertainty import BandUncertainty
    from ..building.prediction.aperture_transmission import ApertureTransmissionResult
    from ..building.prediction.ceiling_plenum import (
        CeilingAttenuationResult,
        PlenumFlankingResult,
    )
    from ..building.prediction.detailed_model import (
        DetailedAirborneResult,
        DetailedImpactResult,
        InSituElementResult,
    )
    from ..building.prediction.facade import FacadePredictionResult, RadiatedPowerResult
    from ..building.prediction.installed_structure_borne import InstalledSourceResult
    from ..building.prediction.masonry_cavity_wall import WallTieCouplingResult
    from ..building.prediction.panel_transmission import SoundReductionResult
    from ..building.prediction.resilient_layers import (
        CoveringImprovementResult,
        FloatingFloorImprovementResult,
        LiningImprovementResult,
        TappingForceResult,
    )
    from ..building.prediction.simplified_model import (
        AirbornePredictionResult,
        ImpactPredictionResult,
    )
    from ..building.regulation.spain import (
        DbHrAssessment,
        DbHrGlobalIndexResult,
    )

#: Shared x-axis label for the frequency-domain building plots.
_FREQ_LABEL = "Frequency [Hz]"

#: Shared x-axis label of the plots drawn against band ordinals rather than
#: centre frequencies (the ISO 717 reference-curve shifts).
_BAND_INDEX_LABEL = "Band index"

#: Shared y-axis label of the heavy-impact level figures.
_MAX_IMPACT_LABEL = "Maximum impact sound pressure level [dB]"

#: Shared y-axis label of the figures drawing the sound reduction index under
#: its symbol (the panel and aperture transmission predictions).
_R_INDEX_LABEL = "Sound reduction index $R$ [dB]"

#: The apparent sound reduction index's symbol, as every curve label writes it:
#: composed prime (never the U+2032 character, which sits at x-height and takes
#: a following subscript with it).
_R_PRIME = r"$R^{\prime}$"

#: Shared y-axis label of the sound-insulation figures that spell the quantity
#: out instead (the airborne measurement and prediction curves).
_REDUCTION_INDEX_LABEL = "Sound reduction index [dB]"

#: Shared y-axis label of the impact sound pressure level figures.
_IMPACT_LEVEL_LABEL = "Impact sound pressure level [dB]"

#: Shared y-axis label of the façade figures, which mix level differences and
#: reduction indices on the same axis.
_LEVEL_DIFFERENCE_LABEL = "Level difference / reduction index [dB]"

#: Shared y-axis label of the path-contribution figures (the simplified
#: single-number bars and the two detailed per-band ones).
_SHARE_LABEL = "Share of transmitted energy [%]"

#: Shared y-axis label of the impact-improvement figures (the ISO 16251-1
#: measurement and the two resilient-layer predictions).
_IMPROVEMENT_LABEL = r"Improvement of impact sound insulation $\Delta L$ [dB]"

#: Spanish translations of the fixed labels/titles/legends rendered by the
#: building-domain ``.plot()`` renderers, keyed by their verbatim English
#: text. ``_t`` returns the English key unchanged for any language other
#: than ``"es"``, so the English output is byte-for-byte identical to the
#: pre-i18n renderers.
_STRINGS: dict[str, str] = {
    _FREQ_LABEL: "Frecuencia [Hz]",
    "Band": "Banda",
    _BAND_INDEX_LABEL: "Índice de banda",
    "predicted $R$": "$R$ previsto",
    _R_INDEX_LABEL: "Índice de reducción acústica $R$ [dB]",
    "Predicted sound insulation": "Aislamiento acústico previsto",
    "coincidence plateau (A to B)": "meseta de coincidencia (A a B)",
    "coincidence range ($f_{c1}$ to $f_{c2}$)": "rango de coincidencia ($f_{c1}$ a $f_{c2}$)",
    "aperture $R$": "$R$ de abertura",
    "Aperture sound transmission (Gomperts / Wilson-Soroka)": "Transmisión sonora por abertura (Gomperts / Wilson-Soroka)",
    _REDUCTION_INDEX_LABEL: "Índice de reducción acústica [dB]",
    r"$\Sigma$ unfav.": r"$\Sigma$ desfav.",
    "impact rating": "índice de impacto",
    _IMPACT_LEVEL_LABEL: "Nivel de presión acústica de impactos [dB]",
    _LEVEL_DIFFERENCE_LABEL: "Diferencia de nivel / índice de reducción [dB]",
    "Façade sound insulation (ISO 16283-3)": "Aislamiento acústico de fachada (ISO 16283-3)",
    "Reduction index / level difference [dB]": "Índice de reducción / diferencia de nivel [dB]",
    "Façade insulation prediction (EN 12354-3)": "Predicción del aislamiento de fachada (EN 12354-3)",
    "Radiated sound power level [dB]": "Nivel de potencia acústica radiada [dB]",
    "Radiated sound power (EN 12354-4)": "Potencia acústica radiada (EN 12354-4)",
    "Vibration reduction index $K_{ij}$ [dB]": "Índice de reducción de vibraciones $K_{ij}$ [dB]",
    "Vibration reduction index (ISO 10848)": "Índice de reducción de vibraciones (ISO 10848)",
    "Structure-borne power level $L_{W\\mathrm{s}}$ [dB re 1 pW]": "Nivel de potencia estructural $L_{W\\mathrm{s}}$ [dB re 1 pW]",
    "EN 15657 characteristic structure-borne sound power": "Potencia acústica estructural característica EN 15657",
    "paths": "trayectos",
    "total $L_\\mathrm{n,s}$": "total $L_\\mathrm{n,s}$",
    "Normalised SPL $L_\\mathrm{n,s}$ [dB]": "NPS normalizado $L_\\mathrm{n,s}$ [dB]",
    "EN 12354-5 installed structure-borne sound": "Ruido estructural instalado EN 12354-5",
    "Transmission path": "Trayecto de transmisión",
    "Share of transmitted energy [%]": "Fracción de energía transmitida [%]",
    "flanking prediction": "predicción de transmisión por flancos",
    "Level / correction [dB]": "Nivel / corrección [dB]",
    "impact prediction": "predicción de impacto",
    "Airborne sound insulation (ISO 16283-1)": "Aislamiento a ruido aéreo (ISO 16283-1)",
    "Impact sound insulation (ISO 16283-2)": "Aislamiento a ruido de impacto (ISO 16283-2)",
    "Standard uncertainty $u$ [dB]": "Incertidumbre típica $u$ [dB]",
    "band uncertainty": "incertidumbre por banda",
    "situation": "situación",
    r"$\sigma_\mathrm{R95}$ upper limit": r"límite superior $\sigma_\mathrm{R95}$",
    r"limit of measurement (> $\Delta L$)": r"límite de medición (> $\Delta L$)",
    "enlarged range (Annex B)": "rango ampliado (Anexo B)",
    "enlarged range (A.2.1)": "rango ampliado (A.2.1)",
    "Measured": "Medido",
    "Shifted reference (core bands)": "Referencia desplazada (bandas 100-3150 Hz)",
    _IMPROVEMENT_LABEL: r"Mejora del aislamiento a ruido de impacto $\Delta L$ [dB]",
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
    "tolerance band": "banda de tolerancia",
    "nominal $L_{FE}$": "$L_{FE}$ nominal",
    "measured $L_{FE}$": "$L_{FE}$ medido",
    "outside tolerance": "fuera de tolerancia",
    "Impact force exposure level $L_{FE}$ [dB re 1 N]":
        "Nivel de exposición a la fuerza de impacto $L_{FE}$ [dB re 1 N]",
    "conforms": "cumple",
    "does not conform": "no cumple",
    "Heavy impact source conformance": "Conformidad de la fuente de impacto pesada",
    "rubber ball": "pelota de caucho",
    "bang machine": "máquina de neumático",
    r"$L_\mathrm{i,Fmax}$ (measured)": r"$L_\mathrm{i,Fmax}$ (medido)",
    r"$L^{\prime}_{\mathrm{i,Fmax},V,T}$ (standardized)":
        r"$L^{\prime}_{\mathrm{i,Fmax},V,T}$ (estandarizado)",
    "standardization correction": "corrección de estandarización",
    _MAX_IMPACT_LABEL: "Nivel máximo de presión acústica de impactos [dB]",
    "ISO 16283-2 rubber-ball standardization":
        "Estandarización de la pelota de caucho ISO 16283-2",
    "A-weighted contribution": "contribución ponderada A",
    r"$X_\mathrm{i,Fmax}$ (unweighted)": r"$X_\mathrm{i,Fmax}$ (sin ponderar)",
    "ISO 717-2 Annex D heavy-impact rating":
        "Índice de impacto pesado ISO 717-2 Anexo D",
    "one-third octave": "tercio de octava",
    "octave": "octava",
    "Normalized ceiling attenuation": "Diferencia de niveles normalizada del techo",
    "Normalized ceiling attenuation $D_\\mathrm{n,c}$ [dB]":
        "Diferencia de niveles normalizada del techo $D_\\mathrm{n,c}$ [dB]",
    "$R_\\mathrm{S} + R_\\mathrm{R}$ (two ceilings)": "$R_\\mathrm{S} + R_\\mathrm{R}$ (dos techos)",
    r"$R_\mathrm{cl}$ (ceiling/plenum path)": r"$R_\mathrm{cl}$ (trayecto techo/plenum)",
    "plenum penalty": "penalización del plenum",
    "Suspended-ceiling plenum path": "Trayecto por plenum de techo suspendido",
    "rigid connection ($Y_\\mathrm{c}$ = 0)": "unión rígida ($Y_\\mathrm{c}$ = 0)",
    "resilient tie array": "conjunto de llaves elásticas",
    "isolation gained by the tie": "aislamiento aportado por la llave",
    "Coupling loss factor $\\eta_{ij}$": "Factor de pérdidas por acoplamiento $\\eta_{ij}$",
    "Wall-tie structure-borne coupling": "Acoplamiento estructural por llaves de muro",
    "ties/m²": "llaves/m²",
    "Frequency [Hz]": "Frecuencia [Hz]",
    "Band index": "Índice de banda",
    "force spectrum $|F_n|$": "espectro de fuerza $|F_n|$",
    "Magnitude of the peak force $|F_n|$ [N]": "Magnitud de la fuerza de pico $|F_n|$ [N]",
    "ISO tapping machine force spectrum": "Espectro de fuerza de la máquina de impactos ISO",
    "over-critical": "sobrecrítico",
    "under-critical": "subcrítico",
    "band force ratio (Eq. 4.114)": "cociente de fuerzas por banda (Ec. 4.114)",
    "two-line estimate (0 dB, 12 dB/oct)": "estimación de dos rectas (0 dB, 12 dB/oct)",
    "Soft floor covering improvement (Hopkins 4.4.3.1)": "Reducción por revestimiento de suelo blando (Hopkins 4.4.3.1)",
    "Floating floor improvement (ISO 12354-2 Annex C)": "Reducción por suelo flotante (ISO 12354-2 Anexo C)",
    "Sound reduction index improvement [dB]": "Mejora del índice de reducción acústica [dB]",
    "Additional-layer rating (ISO 12354-1 Annex D)": "Magnitud global de capa adicional (ISO 12354-1 Anexo D)",
}

#: Localised names of the two standard heavy and soft impact sources.
_HEAVY_IMPACT_SOURCE_LABELS = {
    "rubber_ball": "rubber ball",
    "bang_machine": "bang machine",
}

#: Localised names of the two ISO 717-2 Annex D rating band widths.
_HEAVY_IMPACT_BAND_LABELS = {"third": "one-third octave", "octave": "octave"}


def _t(text: str, language: str = "en") -> str:
    """Localise a fixed string; English is returned verbatim (byte-identical)."""
    return _STRINGS.get(text, text) if language == "es" else text


def plot_sound_reduction(
    result: SoundReductionResult, ax: Axes | None = None, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Predicted sound reduction index ``R(f)`` (Bies 7.2).

    :param result: A
        :class:`~phonometry.building.prediction.panel_transmission.SoundReductionResult`.
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
        symbol = "$f_{c1}$" if result.critical_frequency_upper is not None else "$f_\\mathrm{c}$"
        ax.axvline(
            result.critical_frequency, color=_C_REFERENCE, ls="--", lw=1.0,
            label=f"{symbol} = "
                  f"{format_number(result.critical_frequency, language, decimals=0)} Hz",
        )
    if result.critical_frequency is not None and result.critical_frequency_upper is not None:
        # An orthotropic panel has a coincidence *range*, not a dip: shade it,
        # because that band is where R flattens below the mass law.
        ax.axvspan(
            result.critical_frequency, result.critical_frequency_upper,
            color=theme_fill(_C_TERTIARY, ax), lw=0, zorder=0,
            label=_t("coincidence range ($f_{c1}$ to $f_{c2}$)", language),
        )
        ax.axvline(
            result.critical_frequency_upper, color=_C_REFERENCE, ls="--", lw=1.0,
            label="$f_{c2}$ = "
                  f"{format_number(result.critical_frequency_upper, language, decimals=0)} Hz",
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
    ax.set_ylabel(_t(_R_INDEX_LABEL, language))
    ax.set_title(f"{_t('Predicted sound insulation', language)} ({result.model})")
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, which="both", alpha=0.3)
    localize_axes(ax, language)
    return ax


def plot_aperture_transmission(
    result: ApertureTransmissionResult, ax: Axes | None = None,
    language: str = "en", **kwargs: Any
) -> Axes:
    """Aperture sound reduction index ``R(f) = -10 log10(tau)`` (Hopkins 4.3.10).

    :param result: An
        :class:`~phonometry.building.prediction.aperture_transmission.ApertureTransmissionResult`.
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
    ax.set_ylabel(_t(_R_INDEX_LABEL, language))
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

    :param result: A :class:`~phonometry.building.measurement.insulation.WeightedRatingResult`.
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
            rf"ISO 717-1 $R_\mathrm{{w}}$ "
            rf"($C$={format_number(result.c, language, decimals=0)}; "
            rf"$C_\mathrm{{tr}}$={format_number(result.ctr, language, decimals=0)}) = "
            rf"{result.rating} dB  ({_t(r'$\Sigma$ unfav.', language)} = "
            rf"{format_number(result.unfavourable_sum, language, decimals=1)} dB)"
        ),
        ylabel=_t(_REDUCTION_INDEX_LABEL, language),
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

    :param result: An :class:`~phonometry.building.measurement.insulation.ImpactRatingResult`.
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
            rf"ISO 717-2 {_t('impact rating', language)} "
            rf"($C_\mathrm{{I}}$={format_number(result.ci, language, decimals=0)}) = "
            rf"{result.rating} dB  ({_t(r'$\Sigma$ unfav.', language)} = "
            rf"{format_number(result.unfavourable_sum, language, decimals=1)} dB)"
        ),
        ylabel=_t(_IMPACT_LEVEL_LABEL, language),
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
        :class:`~phonometry.building.measurement.insulation.ExtendedWeightedRatingResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the measured-curve ``plot`` call.
    :return: The axes.
    """
    from .._i18n import format_number

    decimals = 0 if float(result.rating).is_integer() else 1
    title = (
        rf"ISO 717-1 $R_\mathrm{{w}}$ "
        rf"($C$={format_number(result.c, language, decimals=decimals)}; "
        rf"$C_\mathrm{{tr}}$="
        rf"{format_number(result.ctr, language, decimals=decimals)}) = "
        rf"{format_number(result.rating, language, decimals=decimals)} dB"
    )
    extended = _extended_terms_line(
        [
            # U+2010 HYPHEN, not an ASCII one: inside mathematics a hyphen is
            # the binary minus, and mathtext would space the range like a
            # subtraction ("50 − 3150").
            ("$C_{50‐3150}$", result.c_50_3150),
            ("$C_{50‐5000}$", result.c_50_5000),
            ("$C_{100‐5000}$", result.c_100_5000),
            (r"$C_{\mathrm{tr},50‐3150}$", result.ctr_50_3150),
            (r"$C_{\mathrm{tr},50‐5000}$", result.ctr_50_5000),
            (r"$C_{\mathrm{tr},100‐5000}$", result.ctr_100_5000),
        ],
        language, decimals,
    )
    if extended:
        title = f"{title}\n{extended}"
    return _plot_extended_rating(
        result, impact=False, title=title,
        ylabel=_t(_REDUCTION_INDEX_LABEL, language),
        span_label="enlarged range (Annex B)", ax=ax, language=language,
        **kwargs,
    )


def plot_extended_impact_rating(
    result: ExtendedImpactRatingResult, ax: Axes | None = None,
    language: str = "en", **kwargs: Any
) -> Axes:
    """Enlarged-range impact rating curve vs shifted reference (ISO 717-2 A.2.1).

    :param result: An
        :class:`~phonometry.building.measurement.insulation.ExtendedImpactRatingResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the measured-curve ``plot`` call.
    :return: The axes.
    """
    from .._i18n import format_number

    decimals = 0 if float(result.rating).is_integer() else 1
    title = (
        rf"ISO 717-2 {_t('impact rating', language)} "
        rf"($C_\mathrm{{I}}$="
        rf"{format_number(result.ci, language, decimals=decimals)}) = "
        rf"{format_number(result.rating, language, decimals=decimals)} dB"
    )
    if result.ci_50_2500 is not None:
        title = (
            f"{title}\n"
            # U+2010 HYPHEN: a range, not the binary minus (see Annex B above).
            rf"$C_{{\mathrm{{I}},50‐2500}}$ = "
            rf"{format_number(result.ci_50_2500, language, decimals=decimals)}"
        )
    return _plot_extended_rating(
        result, impact=True, title=title,
        ylabel=_t(_IMPACT_LEVEL_LABEL, language),
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
    :class:`~phonometry.building.measurement.insulation.FacadeInsulationResult`.

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
        curves.append((_R_PRIME, np.asarray(result.r_prime, dtype=np.float64)))
    # Forward user kwargs to the primary D2m,nT curve only, so styling kwargs
    # (label=, color=) neither collide with the per-curve labels nor make the
    # companion curves indistinguishable.
    for index, (label, y) in enumerate(curves):
        opts: dict[str, Any] = {"label": label}
        if index == 0:
            opts.update(kwargs)
        ax.plot(x, y, "o-", **opts)

    ax.set_ylabel(_t(_LEVEL_DIFFERENCE_LABEL, language))
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

    Draws the per-element partial indices ``Rp = -10 log10 τ`` as thin dashed
    lines, then the façade apparent reduction ``R'`` and the standardized
    level difference ``D2m,nT`` as bold curves, against frequency. Works for
    :class:`~phonometry.building.prediction.facade.FacadePredictionResult`.

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

    opts: dict[str, Any] = {"label": _R_PRIME, "color": "black", "lw": 2.0}
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
    :class:`~phonometry.building.prediction.facade.RadiatedPowerResult`.

    :param result: A :class:`~phonometry.building.prediction.facade.RadiatedPowerResult`.
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
            label=f"$L_{{W\\mathrm{{A}}}}$ = {format_number(result.l_w_dba, language, decimals=1)} dB(A)",
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
        :class:`~phonometry.building.measurement.flanking_transmission.VibrationReductionResult`.
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
        ax.set_xlabel(_t(_BAND_INDEX_LABEL, language))
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

    :param result: A :class:`~phonometry.building.measurement.structure_borne_power.StructureBornePowerResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the bar ``plot``.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = _plot_band_level_bars(
        ax, result.power_level, result.frequencies, result.total_level,
        ylabel=_t(r"Structure-borne power level $L_{W\mathrm{s}}$ [dB re 1 pW]", language),
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

    :param result: An :class:`~phonometry.building.prediction.installed_structure_borne.InstalledSourceResult`.
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
    ax.plot(x, total, label=_t(r"total $L_\mathrm{n,s}$", language), **kwargs)
    ax.set_ylabel(_t(r"Normalised SPL $L_\mathrm{n,s}$ [dB]", language))
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
        :class:`~phonometry.building.prediction.simplified_model.AirbornePredictionResult`.
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
    ax.set_ylabel(_t(_SHARE_LABEL, language))
    ax.set_title(
        rf"EN 12354-1 {_t('flanking prediction', language)} — "
        rf"$R^{{\prime}}_\mathrm{{w}}$ = "
        rf"{format_number(result.r_prime_w, language, decimals=1)} dB "
        rf"($R_\mathrm{{Dd,w}}$ = "
        rf"{format_number(result.r_direct_w, language, decimals=1)} dB)"
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
        :class:`~phonometry.building.prediction.simplified_model.ImpactPredictionResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the term :meth:`~matplotlib.axes.Axes.bar`.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    labels = (
        r"$L_\mathrm{n,w,eq}$", r"$-\Delta L_\mathrm{w}$", "$+K$",
        r"$L^{\prime}_\mathrm{n,w}$",
    )
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
        rf"EN 12354-2 {_t('impact prediction', language)} — "
        rf"$L^{{\prime}}_\mathrm{{n,w}}$ = "
        rf"{format_number(result.l_prime_n_w, language, decimals=1)} dB"
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
               edgecolor="none", zorder=0, label=_t("other paths", language),
               **kwargs)
    ax.set_ylabel(_t(_SHARE_LABEL, language))
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
        :class:`~phonometry.building.prediction.detailed_model.DetailedAirborneResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the stacked :meth:`~matplotlib.axes.Axes.bar`.
    :return: The axes.
    """
    from .._i18n import format_number

    title = f"EN 12354-1 {_t('detailed prediction', language)}"
    if result.rating is not None:
        title += (
            rf" — $R^{{\prime}}_\mathrm{{w}}$ = "
            rf"{format_number(result.rating.rating, language, decimals=0)} dB"
        )
    return _plot_path_shares(
        result, result.r_prime, total_label=_R_PRIME,
        ylabel=_REDUCTION_INDEX_LABEL, title=title, ax=ax,
        language=language, **kwargs,
    )


def plot_detailed_impact_prediction(
    result: DetailedImpactResult, ax: Axes | None = None, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Per-band path contributions and ``L'n`` (EN/ISO 12354-2 detailed model).

    :param result: A
        :class:`~phonometry.building.prediction.detailed_model.DetailedImpactResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the stacked :meth:`~matplotlib.axes.Axes.bar`.
    :return: The axes.
    """
    from .._i18n import format_number

    title = f"EN 12354-2 {_t('detailed prediction', language)}"
    if result.rating is not None:
        title += (
            rf" — $L^{{\prime}}_\mathrm{{n,w}}$ = "
            rf"{format_number(result.rating.rating, language, decimals=0)} dB"
        )
    return _plot_path_shares(
        result, result.l_prime_n, total_label=r"$L^{\prime}_\mathrm{n}$",
        ylabel=_IMPACT_LEVEL_LABEL, title=title, ax=ax,
        language=language, **kwargs,
    )


def plot_in_situ_element(
    result: InSituElementResult, ax: Axes | None = None, language: str = "en",
    **kwargs: Any
) -> Axes:
    """In-situ ``Rsitu`` and ``Ln,situ`` of one element (EN/ISO 12354).

    :param result: An
        :class:`~phonometry.building.prediction.detailed_model.InSituElementResult`.
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
    ax.plot(freqs, result.sound_reduction_index, label=r"$R_\mathrm{situ}$", **kwargs)
    ax.plot(freqs, result.impact_level, color=_C_SECONDARY, marker="s", ms=4,
            label=r"$L_\mathrm{n,situ}$")
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

    :param result: An :class:`~phonometry.building.measurement.insulation.AirborneInsulationResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the primary ``DnT`` curve ``plot`` call.
    :return: The axes.
    """
    from .._i18n import localize_axes

    curves = [
        ("$D_\\mathrm{nT}$", np.asarray(result.dnt, dtype=np.float64)),
        ("$D$", np.asarray(result.d, dtype=np.float64)),
    ]
    if result.r_prime is not None:
        curves.append((_R_PRIME, np.asarray(result.r_prime, dtype=np.float64)))
    ax = _plot_insulation_bands(
        curves,
        ylabel=_t(_LEVEL_DIFFERENCE_LABEL, language),
        title=_t("Airborne sound insulation (ISO 16283-1)", language),
        ax=ax,
        language=language,
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

    :param result: An :class:`~phonometry.building.measurement.insulation.ImpactInsulationResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the primary ``L'nT`` curve ``plot`` call.
    :return: The axes.
    """
    from .._i18n import localize_axes

    curves = [
        (r"$L^{\prime}_\mathrm{nT}$", np.asarray(result.l_n_t, dtype=np.float64))
    ]
    if result.l_n is not None:
        curves.append(
            (r"$L^{\prime}_\mathrm{n}$", np.asarray(result.l_n, dtype=np.float64))
        )
    ax = _plot_insulation_bands(
        curves,
        ylabel=_t(_IMPACT_LEVEL_LABEL, language),
        title=_t("Impact sound insulation (ISO 16283-2)", language),
        ax=ax,
        language=language,
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
        :class:`~phonometry.building.measurement.uncertainty.BandUncertainty`.
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
    ax.set_ylabel(_t("Standard uncertainty $u$ [dB]", language))
    ax.set_ylim(bottom=0.0)
    quantity = (
        _t(r"$\sigma_\mathrm{R95}$ upper limit", language)
        if result.upper_limit else "$u$"
    )
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
        :class:`~phonometry.building.measurement.floor_covering_improvement.FloorCoveringImprovementResult`.
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
            label=_t(r"limit of measurement (> $\Delta L$)", language),
        )
    _freq_axis(ax, freqs, language=language)
    ax.set_ylabel(_t(_IMPROVEMENT_LABEL, language))
    ax.set_ylim(bottom=0.0)
    title = _t("ISO 16251-1 Floor-Covering Impact Sound Improvement", language)
    if result.delta_lw is not None:
        title += (
            r"  ($\Delta L_\mathrm{w}$ = "
            f"{decimal_comma(str(result.delta_lw), language)} dB)"
        )
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
        :class:`~phonometry.building.regulation.spain.DbHrGlobalIndexResult`.
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
        :class:`~phonometry.building.regulation.spain.DbHrAssessment`.
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


def _plot_shaded_band_pair(
    ax: Axes | None,
    frequencies: np.ndarray | None,
    reference: np.ndarray,
    curve: np.ndarray,
    *,
    reference_label: str,
    curve_label: str,
    fill_label: str,
    ylabel: str,
    title: str,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Two band curves on a categorical axis with the gap between them shaded.

    Shared by the heavy-impact standardization figure (measured against
    standardized level) and the ceiling/plenum figure (the two ceilings against
    the path they form): both draw a dashed reference, a solid main curve and
    the difference as a filled band, on band positions labelled either with the
    centre frequencies or with a plain band index when none were supplied.
    """
    ax = ax if ax is not None else _new_axes()
    labels = frequencies if frequencies is not None else np.arange(curve.size) + 1.0
    positions = _band_axis(
        ax, labels,
        xlabel=_FREQ_LABEL if frequencies is not None else _BAND_INDEX_LABEL,
        language=language,
    )
    ax.plot(positions, reference, "s--", color=_C_REFERENCE, lw=1.2,
            label=_t(reference_label, language))
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("marker", "o")
    ax.plot(positions, curve, "-", label=_t(curve_label, language), **kwargs)
    ax.fill_between(
        positions, reference, curve,
        color=theme_fill(_C_SECONDARY, ax), lw=0, zorder=0,
        label=_t(fill_label, language),
    )
    ax.set_ylabel(_t(ylabel, language))
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    return ax


def plot_heavy_impact_source(
    result: HeavyImpactSourceCheck, ax: Axes | None = None,
    language: str = "en", **kwargs: Any
) -> Axes:
    """Measured heavy-impact source ``LFE`` against its printed tolerance band.

    :param result: A
        :class:`~phonometry.building.measurement.heavy_impact.HeavyImpactSourceCheck`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the measured-curve ``plot`` call.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    positions = _band_axis(ax, result.frequencies, language=language)
    lower = result.nominal - result.tolerance
    upper = result.nominal + result.tolerance
    ax.fill_between(
        positions, lower, upper, color=theme_fill(_C_TERTIARY, ax), lw=0, zorder=0,
        label=_t("tolerance band", language),
    )
    ax.plot(positions, result.nominal, "s--", color=_C_REFERENCE, lw=1.2,
            label=_t("nominal $L_{FE}$", language))
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("marker", "o")
    ax.plot(positions, result.measured, "-", label=_t("measured $L_{FE}$", language),
            **kwargs)
    failing = ~result.within_tolerance
    if bool(np.any(failing)):
        ax.plot(positions[failing], result.measured[failing], ls="", marker="X",
                color=_C_SECONDARY, ms=11, zorder=6,
                label=_t("outside tolerance", language))
    ax.set_ylabel(_t("Impact force exposure level $L_{FE}$ [dB re 1 N]", language))
    verdict = _t("conforms", language) if result.passed else _t("does not conform", language)
    source = _t(_HEAVY_IMPACT_SOURCE_LABELS[result.source], language)
    ax.set_title(
        f"{_t('Heavy impact source conformance', language)}: {source} ({verdict})"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    localize_axes(ax, language)
    return ax


def plot_standardized_maximum_impact(
    result: StandardizedMaximumImpactResult, ax: Axes | None = None,
    language: str = "en", **kwargs: Any
) -> Axes:
    """Measured and standardized maximum impact levels (ISO 16283-2 3.16).

    :param result: A
        :class:`~phonometry.building.measurement.heavy_impact.StandardizedMaximumImpactResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the standardized-curve ``plot`` call.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    ax = _plot_shaded_band_pair(
        ax,
        result.frequencies,
        result.measured,
        result.standardized,
        reference_label=r"$L_\mathrm{i,Fmax}$ (measured)",
        curve_label=r"$L^{\prime}_{\mathrm{i,Fmax},V,T}$ (standardized)",
        fill_label="standardization correction",
        ylabel=_MAX_IMPACT_LABEL,
        title=(
            f"{_t('ISO 16283-2 rubber-ball standardization', language)} "
            f"($V$ = {format_number(result.volume, language, decimals=1)} m³)"
        ),
        language=language,
        **kwargs,
    )
    localize_axes(ax, language)
    return ax


def plot_a_weighted_maximum_impact(
    result: AWeightedMaximumImpactResult, ax: Axes | None = None,
    language: str = "en", **kwargs: Any
) -> Axes:
    """A-weighted maximum impact level and its band contributions (ISO 717-2 D).

    :param result: An
        :class:`~phonometry.building.measurement.heavy_impact.AWeightedMaximumImpactResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the ``bar`` call for the corrected values.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    positions = _band_axis(ax, result.frequencies, language=language)
    kwargs.setdefault("color", _C_PRIMARY)
    ax.bar(positions, result.corrected, width=0.7, zorder=2,
           label=_t("A-weighted contribution", language), **kwargs)
    ax.plot(positions, result.levels, "s--", color=_C_REFERENCE, lw=1.2, zorder=3,
            label=_t(r"$X_\mathrm{i,Fmax}$ (unweighted)", language))
    ax.axhline(result.rating, color=_C_SECONDARY, ls="-", lw=1.6, zorder=4,
               label=rf"$X_\mathrm{{iA,Fmax}}$ = {result.rating} dB")
    ax.set_ylabel(_t(_MAX_IMPACT_LABEL, language))
    ax.set_title(
        f"{_t('ISO 717-2 Annex D heavy-impact rating', language)} "
        f"({_t(_HEAVY_IMPACT_BAND_LABELS[result.band], language)}, "
        f"{format_number(result.unrounded, language, decimals=2)} dB)"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    localize_axes(ax, language)
    return ax


def plot_ceiling_attenuation(
    result: CeilingAttenuationResult, ax: Axes | None = None,
    language: str = "en", **kwargs: Any
) -> Axes:
    """Normalized ceiling attenuation against the fitted ASTM E413 contour.

    :param result: A
        :class:`~phonometry.building.prediction.ceiling_plenum.CeilingAttenuationResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the measured-curve ``plot`` call.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    ax = _plot_rating(
        result.frequencies,
        result.measured,
        result.shifted_reference,
        impact=False,
        title=(
            f"ASTM E413 CAC = {result.rating} dB  "
            rf"({_t(r'$\Sigma$ unfav.', language)} = "
            f"{format_number(result.deficiency_sum, language, decimals=1)} dB)"
        ),
        measured_label="Normalized ceiling attenuation",
        ylabel=_t("Normalized ceiling attenuation $D_\\mathrm{n,c}$ [dB]", language),
        ax=ax,
        language=language,
        **kwargs,
    )
    localize_axes(ax, language)
    return ax


def plot_plenum_flanking(
    result: PlenumFlankingResult, ax: Axes | None = None,
    language: str = "en", **kwargs: Any
) -> Axes:
    """Ceiling/plenum flanking path ``Rcl`` against the two ceilings.

    :param result: A
        :class:`~phonometry.building.prediction.ceiling_plenum.PlenumFlankingResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the ``Rcl`` curve ``plot`` call.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    ax = _plot_shaded_band_pair(
        ax,
        result.frequencies,
        result.reduction_index_source + result.reduction_index_receiving,
        result.reduction_index,
        reference_label="$R_\\mathrm{S} + R_\\mathrm{R}$ (two ceilings)",
        curve_label=r"$R_\mathrm{cl}$ (ceiling/plenum path)",
        fill_label="plenum penalty",
        ylabel=_R_INDEX_LABEL,
        title=(
            f"{_t('Suspended-ceiling plenum path', language)} "
            f"($h$ = {format_number(result.plenum_height, language, decimals=2)} m, "
            f"$L_\\mathrm{{R}}$ = {format_number(result.ceiling_length, language, decimals=2)} m, "
            f"$\\varepsilon$ = {result.epsilon:.0f})"
        ),
        language=language,
        **kwargs,
    )
    localize_axes(ax, language)
    return ax


def plot_wall_tie_coupling(
    result: WallTieCouplingResult, ax: Axes | None = None,
    language: str = "en", **kwargs: Any
) -> Axes:
    """Wall-tie coupling loss factor against the rigid-connection ceiling.

    :param result: A
        :class:`~phonometry.building.prediction.masonry_cavity_wall.WallTieCouplingResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the ``eta_ij`` curve ``plot`` call.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    freq = np.asarray(result.frequencies, dtype=np.float64)
    ax.loglog(freq, result.rigid_coupling_loss_factor, "--", color=_C_REFERENCE,
              lw=1.2, label=_t("rigid connection ($Y_\\mathrm{c}$ = 0)", language))
    kwargs.setdefault("color", _C_PRIMARY)
    ax.loglog(freq, result.coupling_loss_factor,
              label=_t("resilient tie array", language), **kwargs)
    ax.fill_between(
        freq, result.coupling_loss_factor, result.rigid_coupling_loss_factor,
        color=theme_fill(_C_TERTIARY, ax), lw=0, zorder=0,
        label=_t("isolation gained by the tie", language),
    )
    format_frequency_axis(ax, float(freq.min()), float(freq.max()))
    ax.set_xlabel(_t(_FREQ_LABEL, language))
    ax.set_ylabel(_t("Coupling loss factor $\\eta_{ij}$", language))
    stiffness = (
        "" if result.tie_stiffness is None
        else ", $k$ = "
             f"{format_number(result.tie_stiffness / 1e6, language, decimals=1)} MN/m"
    )
    ax.set_title(
        f"{_t('Wall-tie structure-borne coupling', language)} "
        f"({format_number(result.ties_per_area, language, decimals=1)} "
        f"{_t('ties/m²', language)}{stiffness})"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    localize_axes(ax, language)
    return ax


# --------------------------------------------------------------------------- #
# Resilient-layer prediction (tapping force, coverings, floating floors,
# linings). The two improvement spectra share one drawing helper so the
# reference-slope decoration is written once.
# --------------------------------------------------------------------------- #
def _plot_improvement_spectrum(
    ax: Axes,
    freqs: np.ndarray,
    curves: Sequence[tuple[np.ndarray, str, str, str]],
    marker_frequency: float | None,
    marker_label: str,
    language: str,
    kwargs: dict[str, Any],
) -> None:
    """Draw one or more ``ΔL(f)`` curves with an optional vertical marker."""
    first, *rest = curves
    values, label, colour, style = first
    kwargs.setdefault("color", colour)
    kwargs.setdefault("ls", style)
    # The caller may name the curve; do not hand matplotlib two labels.
    ax.plot(freqs, values, label=kwargs.pop("label", _t(label, language)), **kwargs)
    for values, label, colour, style in rest:
        ax.plot(freqs, values, color=colour, ls=style, label=_t(label, language))
    if marker_frequency is not None:
        ax.axvline(
            marker_frequency, color=_C_MUTED, ls=":", lw=1.2,
            label=f"{_t(marker_label, language)} = {_format_freq(marker_frequency)} Hz",
        )
    ax.set_xlabel(_t(_FREQ_LABEL, language))
    ax.set_ylabel(_t(_IMPROVEMENT_LABEL, language))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    format_frequency_axis(ax, float(freqs.min()), float(freqs.max()))


def plot_tapping_force(
    result: TappingForceResult, ax: Axes | None = None, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Force spectrum of the ISO tapping machine on one walking surface.

    Draws ``|Fn|(f)`` from the mass-spring-dashpot model with the two
    low-frequency asymptotes ``|Fn|lower`` / ``|Fn|upper`` and the cut-off
    frequency ``fco`` marked.

    :param result: A
        :class:`~phonometry.building.prediction.resilient_layers.TappingForceResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the force-spectrum ``plot`` call.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    kwargs.setdefault("color", _C_PRIMARY)
    ax.plot(freqs, result.peak_force,
            label=kwargs.pop("label", _t("force spectrum $|F_n|$", language)),
            **kwargs)
    ax.axhline(result.upper_limit, color=_C_REFERENCE, ls="--", lw=1.2,
               label=r"$|F_n|_\mathrm{upper}$")
    ax.axhline(result.lower_limit, color=_C_MUTED, ls="--", lw=1.2,
               label=r"$|F_n|_\mathrm{lower}$")
    ax.axvline(
        result.cut_off_frequency, color=_C_SECONDARY, ls=":", lw=1.2,
        label=rf"$f_\mathrm{{co}}$ = {_format_freq(result.cut_off_frequency)} Hz",
    )
    ax.set_xlabel(_t(_FREQ_LABEL, language))
    ax.set_ylabel(_t("Magnitude of the peak force $|F_n|$ [N]", language))
    ax.set_yscale("log")
    regime = "over-critical" if result.over_critical else "under-critical"
    ax.set_title(
        f"{_t('ISO tapping machine force spectrum', language)} "
        f"({_t(regime, language)})"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    format_frequency_axis(ax, float(freqs.min()), float(freqs.max()))
    localize_axes(ax, language)
    return ax


def plot_covering_improvement(
    result: CoveringImprovementResult, ax: Axes | None = None, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Predicted ``ΔL`` of a soft floor covering beside its two-line estimate.

    :param result: A
        :class:`~phonometry.building.prediction.resilient_layers.CoveringImprovementResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the force-ratio curve ``plot`` call.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    _plot_improvement_spectrum(
        ax, freqs,
        [
            (result.improvement, "band force ratio (Eq. 4.114)", _C_PRIMARY, "-"),
            (result.two_line, "two-line estimate (0 dB, 12 dB/oct)",
             _C_SECONDARY, "--"),
        ],
        result.cut_off_frequency, r"$f_\mathrm{co}$", language, kwargs,
    )
    ax.set_title(_t("Soft floor covering improvement (Hopkins 4.4.3.1)", language))
    localize_axes(ax, language)
    return ax


def plot_floating_floor_improvement(
    result: FloatingFloorImprovementResult, ax: Axes | None = None,
    language: str = "en", **kwargs: Any
) -> Axes:
    """Predicted ``ΔL`` of a floating floor above its mass-spring resonance.

    :param result: A
        :class:`~phonometry.building.prediction.resilient_layers.FloatingFloorImprovementResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the improvement curve ``plot`` call.
    :return: The axes.
    """
    from .._i18n import decimal_comma, localize_axes

    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    label = rf"{result.model} ({result.slope:.0f} log10(f/$f_\mathrm{{o}}$))"
    _plot_improvement_spectrum(
        ax, freqs, [(result.improvement, label, _C_PRIMARY, "-")],
        result.resonance_frequency, r"$f_\mathrm{o}$", language, kwargs,
    )
    title = _t("Floating floor improvement (ISO 12354-2 Annex C)", language)
    if result.delta_lw is not None:
        value = decimal_comma(f"{result.delta_lw:.1f}", language)
        # ``decimal_comma`` has already localised the value: the dollars below
        # would otherwise stop the save-time comma pass over the whole string.
        title += rf"  ($\Delta L_\mathrm{{w}}$ = {value} dB)"
    ax.set_title(title)
    localize_axes(ax, language)
    return ax


def plot_lining_improvement(
    result: LiningImprovementResult, ax: Axes | None = None, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Annex D single-number ratings of an additional layer against ``fo``.

    Sweeps the resonance frequency over the Annex D range, draws ``ΔRw``,
    ``ΔRA`` and ``ΔRA,tr`` for this system (the analogue of Figures D.2 and
    D.3) and marks the result's own resonance frequency.

    :param result: A
        :class:`~phonometry.building.prediction.resilient_layers.LiningImprovementResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the ``ΔRw`` curve ``plot`` call.
    :return: The axes.
    """
    from .._i18n import localize_axes
    from ..building.prediction.resilient_layers import lining_improvement

    ax = ax if ax is not None else _new_axes()
    # The endpoints are written back exactly: np.logspace lands a few ulps
    # below 30 Hz, which is outside the range Annex D is stated for and would
    # make drawing the curve emit an extrapolation warning.
    sweep = np.logspace(np.log10(30.0), np.log10(5000.0), 300)
    sweep[0], sweep[-1] = 30.0, 5000.0
    ratings: list[tuple[float, float, float]] = [
        lining_improvement(
            float(f), system=result.system, anchors=result.anchors,
            glued_area=result.glued_area,
        ).ratings
        for f in sweep
    ]
    curves = np.asarray(ratings, dtype=np.float64)
    kwargs.setdefault("color", _C_PRIMARY)
    ax.plot(sweep, curves[:, 0], label=kwargs.pop("label", r"$\Delta R_\mathrm{w}$"), **kwargs)
    ax.plot(sweep, curves[:, 1], color=_C_SECONDARY, ls="--", label=r"$\Delta R_\mathrm{A}$")
    ax.plot(sweep, curves[:, 2], color=_C_TERTIARY, ls="-.", label=r"$\Delta R_\mathrm{A,tr}$")
    ax.plot([result.resonance_frequency], [result.delta_rw], marker="o", ms=8,
            color=_C_REFERENCE, ls="",
            label=rf"$f_\mathrm{{o}}$ = "
                  rf"{_format_freq(result.resonance_frequency)} Hz")
    ax.set_xlabel(_t(_FREQ_LABEL, language))
    ax.set_ylabel(_t("Sound reduction index improvement [dB]", language))
    ax.set_title(
        f"{_t('Additional-layer rating (ISO 12354-1 Annex D)', language)} "
        f"— {result.system}"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    format_frequency_axis(ax, float(sweep.min()), float(sweep.max()))
    localize_axes(ax, language)
    return ax
