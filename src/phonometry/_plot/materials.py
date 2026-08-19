#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Plot renderers for the materials domain (lazy imports from result .plot())."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .common import (
    _ABSORPTION_QUANTITY_LABELS,
    _C_MUTED,
    _C_PRIMARY,
    _C_PRIMARY_LIGHT,
    _C_REFERENCE,
    _band_axis,
    _freq_axis,
    _import_pyplot,
    _new_axes,
    _plot_rating,
    format_frequency_axis,
    theme_fill_alpha,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..materials.absorbers.airflow_resistance import StaticAirflowResult
    from ..materials.absorbers.biot import BiotWavesResult
    from ..materials.absorbers.four_microphone import TransferMatrix
    from ..materials.absorbers.impedance_tube import ImpedanceTubeResult
    from ..materials.absorbers.layered import (
        DiffuseFieldAbsorptionResult,
        LayeredAbsorberResult,
    )
    from ..materials.absorbers.porous import PorousMediumResult
    from ..materials.absorbers.rating import AbsorptionRatingResult
    from ..materials.absorbers.slow_sound import SlitResonatorAbsorberResult
    from ..materials.absorbers.sound_absorption import SoundAbsorptionMeasurement
    from ..materials.absorbers.uncertainty import AbsorptionUncertaintyResult
    from ..materials.diffusers.design import DiffuserPolarResponse
    from ..materials.diffusers.metadiffuser import MetadiffuserResult
    from ..materials.diffusers.reverberation_room_scattering import ScatteringResult
    from ..materials.diffusers.scattering_diffusion import (
        DiffusionResult,
        DiffusionSpectrum,
    )
    from ..materials.resilient.dynamic_stiffness import DynamicStiffnessResult
    from ..materials.surfaces.road_absorption import InsituAbsorptionResult

_FREQ_LABEL = "Frequency [Hz]"
_SCATTERING_TITLE = "Random-incidence scattering coefficient (ISO 17497-1)"
_DIFFUSION_TITLE = "Directional diffusion coefficient (ISO 17497-2)"
_HARD_BACKED_ALPHA_LABEL = r"Hard-backed absorption $\alpha$"
_SIGMA_UNFAV_LABEL = r"$\Sigma$ unfav. = "

#: Spanish translations of the fixed strings rendered by the materials
#: ``.plot()`` renderers, keyed by their verbatim English text.  ``_t`` returns
#: the English key unchanged for any language other than ``"es"``, so the
#: English output is byte-for-byte identical to the pre-i18n renderers.
_STRINGS: dict[str, str] = {
    _FREQ_LABEL: "Frecuencia [Hz]",
    "Sound absorption coefficient": "Coeficiente de absorción acústica",
    r"Sound absorption coefficient $\alpha_\mathrm{s}$":
        r"Coeficiente de absorción acústica $\alpha_\mathrm{s}$",
    "ISO 354 reverberation-room sound absorption":
        "Absorción acústica en cámara reverberante ISO 354",
    r"Practical $\alpha_\mathrm{p}$": r"$\alpha_\mathrm{p}$ práctico",
    "class ": "clase ",
    _SIGMA_UNFAV_LABEL: r"$\Sigma$ desfav. = ",
    "Scattering coefficient $s$": "Coeficiente de dispersión $s$",
    _SCATTERING_TITLE:
        "Coeficiente de dispersión de incidencia aleatoria (ISO 17497-1)",
    "Diffusion coefficient $d$ = ": "Coeficiente de difusión $d$ = ",
    "Predicted diffuser polar response":
        "Respuesta polar predicha del difusor",
    _DIFFUSION_TITLE:
        "Coeficiente de difusión direccional (ISO 17497-2)",
    "Reflected sound-pressure level L [dB]":
        "Nivel de presión acústica reflejado L [dB]",
    "Absorption coefficient": "Coeficiente de absorción",
    "In-situ road-surface absorption (ISO 13472-1)":
        "Absorción in situ de pavimentos (ISO 13472-1)",
    r"Dynamic stiffness per unit area $s^{\prime}$ [MN/m³]":
        r"Rigidez dinámica por unidad de área $s^{\prime}$ [MN/m³]",
    "Natural frequency $f_0$ [Hz]": "Frecuencia natural $f_0$ [Hz]",
    "EN 29052-1 floating-floor resonance": "Resonancia de suelo flotante EN 29052-1",
    r"Absorption $\alpha$": r"Absorción $\alpha$",
    "Reflection factor $|r|$": "Factor de reflexión $|r|$",
    "Coefficient": "Coeficiente",
    "ISO 10534-2 normal-incidence absorption":
        "Absorción a incidencia normal ISO 10534-2",
    r"Fit $\Delta p = a\,u + b\,u^2$": r"Ajuste $\Delta p = a\,u + b\,u^2$",
    "Evaluation point ($u$ = ": "Punto de evaluación ($u$ = ",
    "Linear airflow velocity $u$ [mm/s]": "Velocidad lineal del aire $u$ [mm/s]",
    r"Pressure difference $\Delta p$ [Pa]": r"Diferencia de presión $\Delta p$ [Pa]",
    "ISO 9053-1 static airflow resistance — ":
        "Resistencia al flujo de aire ISO 9053-1 — ",
    "ISO 12999-2 absorption uncertainty": "Incertidumbre de absorción ISO 12999-2",
    "reproducibility": "reproducibilidad",
    "repeatability": "repetibilidad",
    r"Equivalent absorption area $A_\mathrm{T}$ [m²]":
        r"Área de absorción equivalente $A_\mathrm{T}$ [m²]",
    r"Practical absorption coefficient $\alpha_\mathrm{p}$":
        r"Coeficiente de absorción práctico $\alpha_\mathrm{p}$",
    "Value": "Valor",
    "Porous medium": "Medio poroso",
    "Normalised characteristic value": "Valor característico normalizado",
    "Biot waves in a poroelastic layer": "Ondas de Biot en una capa poroelástica",
    "Wavenumber [rad/m]": "Número de onda [rad/m]",
    "Airborne": "Onda del aire",
    "Frame-borne": "Onda del esqueleto",
    "Shear": "Onda de cizalla",
    r"Absorption coefficient $\alpha$": r"Coeficiente de absorción $\alpha$",
    "Reflection factor $|R|$": "Factor de reflexión $|R|$",
    "Panel average": "Media del panel",
    "Well {n}": "Pozo {n}",
    "Metadiffuser per-well absorption": "Absorción por pozo del metadifusor",
    r"Absorption $\alpha(\theta)$": r"Absorción $\alpha(\theta)$",
    r"Absorption $\alpha_{\mathrm{dif}}$": r"Absorción $\alpha_{\mathrm{dif}}$",
    r"Transmission loss $TL_\mathrm{n}$": r"Pérdida de transmisión $TL_\mathrm{n}$",
    r"Transmission loss $TL_\mathrm{n}$ [dB]":
        r"Pérdida de transmisión $TL_\mathrm{n}$ [dB]",
    _HARD_BACKED_ALPHA_LABEL:
        r"Absorción con respaldo rígido $\alpha$",
    "ASTM E2611 transfer-matrix quantities":
        "Magnitudes de la matriz de transferencia ASTM E2611",
}


def _t(text: str, language: str = "en") -> str:
    """Localise a fixed string; English is returned verbatim (byte-identical)."""
    return _STRINGS.get(text, text) if language == "es" else text


def _localize_band_axes(ax: Any, language: str) -> None:
    """Comma-localise the numeric y-axis of a categorical band plot.

    :func:`~phonometry._i18n.localize_axes` reformats only the automatic numeric
    axis and leaves the categorical band tick labels (a ``FuncFormatter`` on the
    linear position axis) untouched, so no label restore is needed. English is a
    no-op.
    """
    from .._i18n import localize_axes

    localize_axes(ax, language)


def plot_weighted_absorption(
    result: AbsorptionRatingResult, ax: Axes | None = None, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Practical absorption curve vs the shifted reference (ISO 11654:1997).

    Draws the practical coefficients ``alpha_p`` against the shifted reference
    curve and shades the unfavourable deviations (measured below the shifted
    reference, Clause 4.2) through the shared rating renderer.

    :param result: An
        :class:`~phonometry.materials.absorbers.rating.AbsorptionRatingResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param kwargs: Forwarded to the measured-curve ``plot`` call.
    :return: The axes.
    """
    from .._i18n import decimal_comma, format_number, localize_axes

    ax = _plot_rating(
        np.asarray(result.band_centers, dtype=np.float64),
        np.asarray(result.measured, dtype=np.float64),
        np.asarray(result.shifted_reference, dtype=np.float64),
        impact=False,
        title=(
            rf"ISO 11654 $\alpha_\mathrm{{w}}$ = "
            f"{decimal_comma(result.rating_label, language)}  "
            f"({_t('class ', language)}{result.absorption_class}, "
            f"{_t(_SIGMA_UNFAV_LABEL, language)}"
            f"{format_number(result.unfavourable_sum, language, decimals=2)})"
        ),
        ylabel=_t("Sound absorption coefficient", language),
        measured_label=_t(r"Practical $\alpha_\mathrm{p}$", language),
        ylim=(0.0, 1.05),
        ax=ax,
        language=language,
        **kwargs,
    )
    localize_axes(ax, language)
    return ax

def plot_sound_absorption(
    result: SoundAbsorptionMeasurement, ax: Axes | None = None,
    language: str = "en", **kwargs: Any
) -> Axes:
    """Sound absorption coefficient ``alpha_s`` versus frequency (ISO 354:2003).

    Draws the one-third-octave ``alpha_s`` on a categorical band axis. Values
    above 1,0 (edge/diffraction effects, Clause 3.7 NOTE 2) are kept, so the
    axis grows to show them.

    :param result: A
        :class:`~phonometry.materials.absorbers.sound_absorption.SoundAbsorptionMeasurement`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param kwargs: Forwarded to the ``alpha_s`` curve ``plot`` call.
    :return: The axes.
    """
    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    alpha = np.asarray(result.alpha_s, dtype=np.float64)
    positions = _band_axis(
        ax, freqs, xlabel=_t(_FREQ_LABEL, language), language=language
    )
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("color", _C_PRIMARY)
    ax.plot(positions, alpha, **kwargs)
    ax.set_ylabel(_t(r"Sound absorption coefficient $\alpha_\mathrm{s}$", language))
    # alpha_s can exceed 1,0 (Clause 3.7 NOTE 2); grow the top so it stays shown.
    top = max(1.05, float(np.nanmax(alpha)) * 1.05) if alpha.size else 1.05
    ax.set_ylim(0.0, top)
    ax.set_title(_t("ISO 354 reverberation-room sound absorption", language))
    ax.grid(True, axis="y", alpha=0.3)
    _localize_band_axes(ax, language)
    return ax

def plot_scattering_coefficient(
    result: ScatteringResult, ax: Axes | None = None, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Random-incidence scattering coefficient ``s`` versus frequency.

    :param result: A :class:`~phonometry.materials.diffusers.reverberation_room_scattering.ScatteringResult`
        exposing ``frequencies`` and ``scattering``.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param kwargs: Forwarded to the coefficient curve ``plot`` call.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    s = np.asarray(result.scattering, dtype=np.float64)
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("color", _C_PRIMARY)
    ax.plot(freqs, s, **kwargs)
    _freq_axis(ax, freqs, language=language)
    ax.set_ylabel(_t("Scattering coefficient $s$", language))
    # s is normally in [0, 1], but edge effects (Clause 6.3.2) can push it above
    # 1 and those values are kept, not clipped; grow the top so they stay visible.
    top = max(1.05, float(np.nanmax(s)) * 1.05) if s.size else 1.05
    ax.set_ylim(0.0, top)
    ax.set_title(_t(_SCATTERING_TITLE, language))
    ax.grid(True, alpha=0.3)
    localize_axes(ax, language)
    return ax

def plot_diffusion_polar(
    result: DiffusionResult, ax: Axes | None = None, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Polar reflected-level response with the diffusion coefficient annotated.

    :param result: A :class:`~phonometry.materials.diffusers.scattering_diffusion.DiffusionResult`
        exposing ``angles`` (degrees), ``levels`` (dB) and ``coefficient``.
    :param ax: Existing (ideally polar) axes, or ``None`` to create a polar one.
    :param kwargs: Forwarded to the reflected-level curve ``plot`` call.
    :return: The polar axes.
    """
    if ax is None:
        plt = _import_pyplot()
        _fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    angles = np.radians(np.asarray(result.angles, dtype=np.float64))
    levels = np.asarray(result.levels, dtype=np.float64)
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("color", _C_PRIMARY)
    from .._i18n import format_number

    ax.plot(angles, levels, **kwargs)
    # Translucent so the polar grid keeps reading through the lobe.
    ax.fill(angles, levels, color=kwargs["color"],
            alpha=theme_fill_alpha(kwargs["color"], ax))
    ax.set_title(
        f"{_t('Diffusion coefficient $d$ = ', language)}"
        f"{format_number(float(result.coefficient), language, decimals=2)} "
        "(ISO 17497-2)"
    )
    return cast("Axes", ax)

def plot_scattering_report(
    result: ScatteringResult, ax: Axes | None = None, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Scattering coefficient ``s`` and ``alpha_s`` on a categorical band axis.

    The report-fiche variant of :func:`plot_scattering_coefficient`: the bands
    sit on evenly spaced positions with nominal labels (``_band_axis``) instead
    of a base-10 log axis, so the embedded fiche figure lines up band-for-band
    with the value table beside it. The area under ``s`` is a pale, fully opaque
    fill drawn below the curves (svglib drops alpha when it vectorises the SVG,
    so a translucent fill would print as a flat block).

    :param result: A :class:`~phonometry.materials.diffusers.reverberation_room_scattering.ScatteringResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param kwargs: Forwarded to the scattering-curve ``plot`` call.
    :return: The axes.
    """
    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    s = np.asarray(result.scattering, dtype=np.float64)
    a_s = np.asarray(result.random_incidence, dtype=np.float64)
    positions = _band_axis(
        ax, freqs, xlabel=_t(_FREQ_LABEL, language), language=language
    )
    ax.fill_between(
        positions, 0.0, np.clip(s, 0.0, None),
        color=_C_PRIMARY_LIGHT, edgecolor="none", zorder=1,
    )
    ax.plot(
        positions, a_s, marker="s", ms=4, color=_C_MUTED, zorder=3,
        label=r"$\alpha_\mathrm{s}$",
    )
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("label", r"$s$")
    ax.plot(positions, s, ms=4, zorder=3, **kwargs)
    ax.set_ylabel(_t("Coefficient", language))
    top = max(1.05, float(np.nanmax(s)) * 1.05) if s.size else 1.05
    ax.set_ylim(0.0, top)
    ax.set_title(
        _t(_SCATTERING_TITLE, language)
    )
    ax.grid(True, axis="y", alpha=0.3)
    _localize_band_axes(ax, language)
    return ax

def plot_diffusion_report(
    result: DiffusionSpectrum, ax: Axes | None = None, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Directional diffusion coefficient ``d(f)`` on a categorical band axis.

    The report-fiche figure of a :class:`DiffusionSpectrum`: the per-band
    directional (and, when present, normalised) diffusion coefficient over the
    one-third-octave bands, drawn on evenly spaced band positions with nominal
    labels (``_band_axis``, not a base-10 log axis) so the curve lines up with
    the value table. The area under ``d`` is a pale, fully opaque fill below the
    curves (svglib drops alpha on vectorisation).

    :param result: A
        :class:`~phonometry.materials.diffusers.scattering_diffusion.DiffusionSpectrum`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param kwargs: Forwarded to the ``d(f)`` curve ``plot`` call.
    :return: The axes.
    """
    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    d = np.asarray(result.diffusion, dtype=np.float64)
    positions = _band_axis(
        ax, freqs, xlabel=_t(_FREQ_LABEL, language), language=language
    )
    ax.fill_between(
        positions, 0.0, np.clip(d, 0.0, None),
        color=_C_PRIMARY_LIGHT, edgecolor="none", zorder=1,
    )
    if result.normalized is not None:
        d_n = np.asarray(result.normalized, dtype=np.float64)
        ax.plot(
            positions, d_n, marker="s", ms=4, color=_C_MUTED, zorder=3,
            label=r"$d_\mathrm{n}$",
        )
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("label", r"$d$")
    ax.plot(positions, d, ms=4, zorder=3, **kwargs)
    ax.set_ylabel(_t("Coefficient", language))
    ax.set_ylim(0.0, 1.05)
    ax.set_title(
        _t(_DIFFUSION_TITLE, language)
    )
    ax.grid(True, axis="y", alpha=0.3)
    _localize_band_axes(ax, language)
    return ax

def plot_diffusion_polar_report(
    result: DiffusionResult, ax: Axes | None = None, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Polar reflected-level response for the diffusion fiche (opaque fill).

    The report-fiche variant of :func:`plot_diffusion_polar`: identical polar
    geometry, but the enclosed area is a pale, fully opaque fill below the curve
    (svglib drops alpha when it vectorises the SVG). The axes must be polar; the
    fiche renderer creates one for it.

    :param result: A :class:`~phonometry.materials.diffusers.scattering_diffusion.DiffusionResult`.
    :param ax: Existing polar axes, or ``None`` to create one.
    :param kwargs: Forwarded to the reflected-level curve ``plot`` call.
    :return: The polar axes.
    """
    if ax is None:
        plt = _import_pyplot()
        _fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    angles_deg = np.asarray(result.angles, dtype=np.float64)
    angles = np.radians(angles_deg)
    levels = np.asarray(result.levels, dtype=np.float64)
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("color", _C_PRIMARY)
    ax.fill(angles, levels, color=_C_PRIMARY_LIGHT, edgecolor="none", zorder=1)
    ax.plot(angles, levels, ms=4, zorder=3, **kwargs)
    # The theta-orientation setters live on the polar axes, not the base Axes.
    polar_ax: Any = ax
    polar_ax.set_theta_zero_location("N")
    polar_ax.set_theta_direction(-1)
    if angles_deg.size and float(np.nanmin(angles_deg)) >= -90.0 and \
            float(np.nanmax(angles_deg)) <= 90.0:
        polar_ax.set_thetamin(-90)
        polar_ax.set_thetamax(90)
    ax.set_title(_t(_DIFFUSION_TITLE, language))
    from .._i18n import localize_axes

    localize_axes(ax, language)
    return cast("Axes", ax)

def plot_insitu_absorption(
    result: InsituAbsorptionResult, ax: Axes | None = None, language: str = "en",
    **kwargs: Any
) -> Axes:
    """In-situ one-third-octave absorption spectrum ``alpha(f)``.

    :param result: An
        :class:`~phonometry.materials.surfaces.road_absorption.InsituAbsorptionResult` exposing
        ``frequencies`` and ``absorption``.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param kwargs: Forwarded to the absorption :meth:`~matplotlib.axes.Axes.bar`.
    :return: The axes.
    """
    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    alpha = np.asarray(result.absorption, dtype=np.float64)
    positions = _band_axis(
        ax, freqs, xlabel=_t(_FREQ_LABEL, language), language=language
    )
    kwargs.setdefault("color", _C_PRIMARY)
    ax.bar(positions, np.nan_to_num(alpha), **kwargs)
    ax.set_ylabel(_t("Absorption coefficient", language))
    ax.set_ylim(0.0, 1.0)
    ax.set_title(_t("In-situ road-surface absorption (ISO 13472-1)", language))
    ax.grid(True, axis="y", alpha=0.3)
    _localize_band_axes(ax, language)
    return ax

def plot_dynamic_stiffness(
    result: DynamicStiffnessResult, ax: Axes | None = None, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Floating-floor natural frequency ``f0(s')`` with the design point marked.

    :param result: A
        :class:`~phonometry.materials.resilient.dynamic_stiffness.DynamicStiffnessResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param kwargs: Forwarded to the design-point ``scatter``.
    :return: The axes.
    """
    from .._i18n import decimal_comma, format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    m = result.floor_mass_per_area
    s_mn = result.dynamic_stiffness / 1e6
    grid = np.logspace(np.log10(max(s_mn * 0.2, 1e-2)), np.log10(s_mn * 5.0), 240)
    f0 = np.sqrt(grid * 1e6 / m) / (2.0 * np.pi)
    ax.plot(grid, f0, color=_C_PRIMARY,
            label=rf"$f_0 = \frac{{1}}{{2\pi}}\sqrt{{s^{{\prime}}/m^{{\prime}}}}$,  "
                 rf"$m^{{\prime}}$ = {decimal_comma(f'{m:g}', language)} kg/m²")
    ax.axhline(result.natural_frequency, color=_C_MUTED, ls=":", lw=0.8)
    ax.plot([s_mn, s_mn], [0.0, result.natural_frequency], color=_C_MUTED, ls=":", lw=0.8)

    kwargs.setdefault("color", _C_REFERENCE)
    kwargs.setdefault("zorder", 5)
    kwargs.setdefault("s", 80)
    kwargs.setdefault("label", rf"$s^{{\prime}}$ = {format_number(s_mn, language, decimals=2)} MN/m³,  "
                      f"$f_0$ = "
                      f"{format_number(result.natural_frequency, language, decimals=1)} Hz")
    ax.scatter([s_mn], [result.natural_frequency],
               **kwargs)
    ax.set_xscale("log")
    ax.set_xlabel(_t(r"Dynamic stiffness per unit area $s^{\prime}$ [MN/m³]", language))
    ax.set_ylabel(_t("Natural frequency $f_0$ [Hz]", language))
    ax.set_title(_t("EN 29052-1 floating-floor resonance", language))
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper left", fontsize="small")
    ax.grid(True, which="both", alpha=0.3)
    localize_axes(ax, language)
    return ax

def plot_impedance_tube(
    result: ImpedanceTubeResult, ax: Axes | None = None, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Normal-incidence absorption spectrum with |r| overlaid (ISO 10534-2).

    Draws the absorption coefficient ``alpha(f)`` as the primary curve and
    the magnitude of the reflection factor ``|r|(f)`` as a muted companion
    (both are dimensionless and share the 0..1 axis).

    :param result: An :class:`~phonometry.materials.absorbers.impedance_tube.ImpedanceTubeResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param kwargs: Forwarded to the absorption-curve ``plot`` call.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.frequency, dtype=np.float64)
    alpha = np.asarray(result.absorption, dtype=np.float64)
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("label", _t(r"Absorption $\alpha$", language))
    ax.plot(freqs, alpha, **kwargs)
    ax.plot(freqs, np.abs(np.asarray(result.reflection, dtype=np.complex128)),
            ls="--", color=_C_MUTED, label=_t("Reflection factor $|r|$", language))
    # A continuous logarithmic frequency axis with band-centre labels (1k, 2k)
    # matches the working plane-wave range of the tube and the rest of the
    # library's spectrum plots (never a power-of-ten log formatter).
    format_frequency_axis(ax, float(freqs.min()), float(freqs.max()))
    ax.set_xlabel(_t(_FREQ_LABEL, language))
    ax.set_ylabel(_t("Coefficient", language))
    ax.set_ylim(0.0, 1.05)
    ax.set_title(_t("ISO 10534-2 normal-incidence absorption", language))
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, which="both", alpha=0.3)
    localize_axes(ax, language)
    return ax

def plot_static_airflow(
    result: StaticAirflowResult, ax: Axes | None = None, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Fitted pressure-drop curve with the evaluation point (ISO 9053-1).

    Draws the clause 7.5 through-origin fit ``dp = a u + b u**2`` over twice
    the evaluation range and marks the reference evaluation point.

    :param result: A
        :class:`~phonometry.materials.absorbers.airflow_resistance.StaticAirflowResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param kwargs: Forwarded to the fitted-curve ``plot`` call.
    :return: The axes.
    """
    from .._i18n import decimal_comma, localize_axes

    ax = ax if ax is not None else _new_axes()
    u_eval = float(result.evaluation_velocity)
    u = np.linspace(0.0, 2.0 * u_eval, 200)
    dp = result.linear_coefficient * u + result.quadratic_coefficient * u**2

    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("label", _t(r"Fit $\Delta p = a\,u + b\,u^2$", language))
    # Millimetres per second keep the clause 7.5 reference (0.5 mm/s) legible.
    ax.plot(u * 1e3, dp, **kwargs)
    ax.plot([u_eval * 1e3], [result.pressure_drop], "D", color=_C_REFERENCE,
            ms=7, label=(f"{_t('Evaluation point ($u$ = ', language)}"
                         f"{decimal_comma(f'{u_eval * 1e3:g}', language)} mm/s)"))
    ax.set_xlabel(_t("Linear airflow velocity $u$ [mm/s]", language))
    ax.set_ylabel(_t(r"Pressure difference $\Delta p$ [Pa]", language))
    ax.set_title(
        f"{_t('ISO 9053-1 static airflow resistance — ', language)}"
        rf"$R_\mathrm{{s}}$ = "
        f"{decimal_comma(f'{result.specific_resistance:.3g}', language)} Pa s/m"
    )
    ax.legend(loc="upper left", fontsize="small")
    ax.grid(True, alpha=0.3)
    localize_axes(ax, language)
    return ax

def plot_absorption_uncertainty(
    result: AbsorptionUncertaintyResult, ax: Axes | None = None,
    language: str = "en", **kwargs: Any
) -> Axes:
    """Absorption quantity with its expanded-uncertainty ribbon (ISO 12999-2).

    Draws the per-band quantity (``alpha_s``, ``A_T`` or ``alpha_p``) as a curve
    with a shaded ``±U`` band using the exact expanded uncertainty ``U = k·u``.

    :param result: An
        :class:`~phonometry.materials.absorbers.uncertainty.AbsorptionUncertaintyResult`
        for a band quantity (single-number results have no spectrum to plot).
    :param ax: Existing axes, or ``None`` to create a figure.
    :param kwargs: Forwarded to the value-curve ``plot`` call.
    :return: The axes.
    :raises ValueError: The result is a single-number quantity (no bands).
    """
    if result.frequencies.size == 0:
        raise ValueError(
            "plot() needs a per-band result; single-number quantities "
            "(alpha_w, DLalpha) have no spectrum to plot."
        )
    from .._i18n import decimal_comma, localize_axes

    ax = ax if ax is not None else _new_axes()
    freqs = result.frequencies
    value = result.values
    u_expanded = result.expanded_uncertainty
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("marker", "o")
    ax.fill_between(
        freqs,
        value - u_expanded,
        value + u_expanded,
        color=_C_PRIMARY_LIGHT,
        alpha=0.5,
        label=rf"$\pm U$ ($k$ = "
              f"{decimal_comma(f'{result.coverage_factor:g}', language)})",
    )
    ax.plot(freqs, value, **kwargs)
    _freq_axis(ax, freqs, language=language)
    ylabel = _ABSORPTION_QUANTITY_LABELS.get(result.quantity, "Value")
    ax.set_ylabel(_t(ylabel, language))
    if result.quantity != "equivalent_area":
        ax.set_ylim(0.0, 1.05)
    sigma = (
        r"$\sigma_\mathrm{R}$" if result.condition == "reproducibility"
        else r"$\sigma_\mathrm{r}$"
    )
    ax.set_title(
        f"{_t('ISO 12999-2 absorption uncertainty', language)} "
        f"({sigma}) — {_t(result.condition, language)}"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    localize_axes(ax, language)
    return ax


def _absorption_spectrum_axes(
    ax: Axes | None,
    freqs: np.ndarray,
    alpha: np.ndarray,
    *,
    title: str,
    label: str,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Shared alpha(f) spectrum renderer for the absorber predictions."""
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("label", label)
    ax.semilogx(freqs, alpha, **kwargs)
    ax.set_xlabel(_t(_FREQ_LABEL, language))
    ax.set_ylabel(_t(r"Absorption coefficient $\alpha$", language))
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    format_frequency_axis(ax, float(freqs.min()), float(freqs.max()))
    localize_axes(ax, language)
    return ax


def _absorption_reflection_axes(
    ax: Axes | None,
    freqs: np.ndarray,
    alpha: np.ndarray,
    reflection: np.ndarray,
    *,
    title: str,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Shared alpha(f) + |R| renderer for the oblique-incidence absorbers.

    Draws the absorption spectrum as the primary curve and overlays the
    reflection-factor magnitude as a muted dashed companion, then adds the
    legend. Used by the layered-absorber and slit-resonator predictions.
    """
    kwargs.setdefault("label", _t(r"Absorption $\alpha(\theta)$", language))
    ax = _absorption_spectrum_axes(
        ax,
        freqs,
        alpha,
        title=title,
        language=language,
        **kwargs,
    )
    ax.semilogx(
        freqs,
        np.abs(np.asarray(reflection, dtype=np.complex128)),
        ls="--", color=_C_MUTED, label=_t("Reflection factor $|R|$", language),
    )
    ax.legend(loc="best", fontsize="small")
    return ax


def plot_porous_medium(
    result: PorousMediumResult, ax: Axes | None = None, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Normalised characteristic values of a porous medium vs frequency.

    Draws the real part and negative imaginary part of the normalised
    characteristic impedance ``Zc / (rho c)`` and wavenumber ``k / k0`` on a
    log-log grid, the classical presentation of the empirical porous models
    (Mechel 2e Sect. G.11; Cox & D'Antonio 3e Figs. 6.19-6.20).

    :param result: A :class:`~phonometry.materials.absorbers.porous.PorousMediumResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param kwargs: Forwarded to the ``Re(Zc)`` ``plot`` call.
    :return: The axes.
    """
    from .._i18n import decimal_comma, localize_axes

    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.frequency, dtype=np.float64)
    zn = np.asarray(result.normalized_impedance, dtype=np.complex128)
    kn = np.asarray(result.normalized_wavenumber, dtype=np.complex128)
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("label", r"$\mathrm{Re}(Z_\mathrm{c})/\rho c$")
    ax.loglog(freqs, zn.real, **kwargs)
    ax.loglog(freqs, -zn.imag, ls="--", color=_C_PRIMARY_LIGHT,
              label=r"$-\mathrm{Im}(Z_\mathrm{c})/\rho c$")
    ax.loglog(freqs, kn.real, color=_C_REFERENCE, label=r"$\mathrm{Re}(k)/k_0$")
    ax.loglog(freqs, -kn.imag, ls="--", color=_C_MUTED,
              label=r"$-\mathrm{Im}(k)/k_0$")
    format_frequency_axis(ax, float(freqs.min()), float(freqs.max()))
    ax.set_xlabel(_t(_FREQ_LABEL, language))
    ax.set_ylabel(_t("Normalised characteristic value", language))
    ax.set_title(
        f"{_t('Porous medium', language)} ({result.model}), "
        f"$\\sigma$ = {decimal_comma(f'{result.flow_resistivity:g}', language)} Pa s/m²"
    )
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, which="both", alpha=0.3)
    localize_axes(ax, language)
    return ax


#: Legend suffixes of the Biot wavenumber curves: the mathtext is the same for
#: every wave, only the wave name in front of it changes.
_BIOT_REAL_PART = r", $\mathrm{Re}(\delta)$"
_BIOT_IMAG_PART = r", $-\mathrm{Im}(\delta)$"


def plot_biot_waves(
    result: BiotWavesResult, ax: Axes | None = None, language: str = "en",
    **kwargs: Any
) -> Axes:
    """The three Biot wavenumbers of a poroelastic layer vs frequency.

    Draws the real part (propagation) and the negative imaginary part
    (attenuation) of the two compressional waves, and the real part of the
    shear wave, the presentation of Allard & Atalla 2e Fig. 6.6. The shear
    attenuation is left out because it tracks its own real part closely enough
    over this range to sit on top of it. The airborne and frame-borne labels follow the
    ``|mu|`` sorting of
    :attr:`~phonometry.materials.absorbers.biot.BiotWavesResult.airborne_is_second`,
    which is the physical labelling of Sect. 6.5.4, not a smoothing: read that
    property before reading a step in these curves as physics, because neither
    labelling of the two compressional roots is continuous in general.

    :param result: A :class:`~phonometry.materials.absorbers.biot.BiotWavesResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param kwargs: Forwarded to the airborne ``Re`` ``plot`` call.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.frequency, dtype=np.float64)
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault(
        "label", _t("Airborne", language) + _BIOT_REAL_PART
    )
    ax.semilogx(freqs, result.airborne_wavenumber.real, **kwargs)
    ax.semilogx(
        freqs, -result.airborne_wavenumber.imag, ls="--", color=_C_PRIMARY_LIGHT,
        label=_t("Airborne", language) + _BIOT_IMAG_PART,
    )
    ax.semilogx(
        freqs, result.frame_borne_wavenumber.real, color=_C_REFERENCE,
        label=_t("Frame-borne", language) + _BIOT_REAL_PART,
    )
    ax.semilogx(
        freqs, -result.frame_borne_wavenumber.imag, ls="--", color=_C_MUTED,
        label=_t("Frame-borne", language) + _BIOT_IMAG_PART,
    )
    ax.semilogx(
        freqs, result.shear_wavenumber.real, ls=":", color=_C_REFERENCE,
        label=_t("Shear", language) + _BIOT_REAL_PART,
    )
    format_frequency_axis(ax, float(freqs.min()), float(freqs.max()))
    ax.set_xlabel(_t(_FREQ_LABEL, language))
    ax.set_ylabel(_t("Wavenumber [rad/m]", language))
    ax.set_title(_t("Biot waves in a poroelastic layer", language))
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, which="both", alpha=0.3)
    localize_axes(ax, language)
    return ax


def plot_layered_absorber(
    result: LayeredAbsorberResult, ax: Axes | None = None, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Oblique-incidence absorption spectrum with |R| overlaid.

    Draws the predicted ``alpha(f)`` of the layer stack as the primary curve
    and the reflection-factor magnitude ``|R|(f)`` as a muted companion.

    :param result: A :class:`~phonometry.materials.absorbers.layered.LayeredAbsorberResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param kwargs: Forwarded to the absorption-curve ``plot`` call.
    :return: The axes.
    """
    from .._i18n import format_number

    angle_deg = np.degrees(result.angle)
    if language == "es":
        title = (f"Predicción de absorbente multicapa "
                 f"($\\theta$ = {format_number(float(angle_deg), language, decimals=0)}°)")
    else:
        title = f"Layered absorber prediction ($\\theta$ = {angle_deg:.0f}°)"
    return _absorption_reflection_axes(
        ax,
        np.asarray(result.frequency, dtype=np.float64),
        np.asarray(result.absorption, dtype=np.float64),
        np.asarray(result.reflection, dtype=np.complex128),
        title=title,
        language=language,
        **kwargs,
    )


def plot_slit_resonator_absorber(
    result: SlitResonatorAbsorberResult, ax: Axes | None = None,
    language: str = "en", **kwargs: Any
) -> Axes:
    """Absorption spectrum of a slit panel loaded with Helmholtz resonators.

    Draws the predicted ``alpha(f)`` of the slow-sound panel as the primary
    curve and the reflection-factor magnitude ``|R|(f)`` as a muted companion.

    :param result: A
        :class:`~phonometry.materials.absorbers.slow_sound.SlitResonatorAbsorberResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param kwargs: Forwarded to the absorption-curve ``plot`` call.
    :return: The axes.
    """
    from .._i18n import format_number

    angle_deg = np.degrees(result.angle)
    if language == "es":
        title = (f"Panel ranurado con resonadores de Helmholtz "
                 f"($\\theta$ = {format_number(float(angle_deg), language, decimals=0)}°)")
    else:
        title = (f"Slit panel with Helmholtz resonators "
                 f"($\\theta$ = {angle_deg:.0f}°)")
    return _absorption_reflection_axes(
        ax,
        np.asarray(result.frequency, dtype=np.float64),
        np.asarray(result.absorption, dtype=np.float64),
        np.asarray(result.reflection, dtype=np.complex128),
        title=title,
        language=language,
        **kwargs,
    )


def plot_diffuse_field_absorption(
    result: DiffuseFieldAbsorptionResult, ax: Axes | None = None,
    language: str = "en", **kwargs: Any
) -> Axes:
    """Random-incidence (Paris-integral) absorption spectrum.

    :param result: A :class:`~phonometry.materials.absorbers.layered.DiffuseFieldAbsorptionResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param kwargs: Forwarded to the absorption-curve ``plot`` call.
    :return: The axes.
    """
    from .._i18n import format_number

    limit_deg = np.degrees(result.angle_limit)
    if language == "es":
        title = (f"Absorción a incidencia aleatoria "
                 f"(integral de Paris hasta "
                 f"{format_number(float(limit_deg), language, decimals=0)}°)")
    else:
        title = f"Random-incidence absorption (Paris integral to {limit_deg:.0f}°)"
    kwargs.setdefault("label", _t(r"Absorption $\alpha_{\mathrm{dif}}$", language))
    ax = _absorption_spectrum_axes(
        ax,
        np.asarray(result.frequency, dtype=np.float64),
        np.asarray(result.absorption, dtype=np.float64),
        title=title,
        language=language,
        **kwargs,
    )
    ax.legend(loc="best", fontsize="small")
    return ax


def plot_diffuser_polar_response(
    result: DiffuserPolarResponse, ax: Axes | None = None,
    language: str = "en", **kwargs: Any
) -> Axes:
    """Predicted far-field polar response of a diffuser design.

    Semicircular polar plot of the predicted reflected sound-pressure levels
    (peak referenced to 0 dB) with the predicted directional diffusion
    coefficient and prediction frequency in the title.

    :param result: A
        :class:`~phonometry.materials.diffusers.design.DiffuserPolarResponse`.
    :param ax: Existing polar axes, or ``None`` to create one.
    :param kwargs: Forwarded to the reflected-level curve ``plot`` call.
    :return: The polar axes.
    """
    from .._i18n import format_number, localize_axes

    if ax is None:
        plt = _import_pyplot()
        _fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    angles_deg = np.asarray(result.angles, dtype=np.float64)
    angles = np.radians(angles_deg)
    levels = np.asarray(result.levels, dtype=np.float64)
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("color", _C_PRIMARY)
    # Translucent so the polar grid keeps reading through the lobe.
    ax.fill(angles, levels, color=kwargs["color"],
            alpha=theme_fill_alpha(kwargs["color"], ax))
    ax.plot(angles, levels, ms=4, **kwargs)
    polar_ax: Any = ax
    polar_ax.set_theta_zero_location("N")
    polar_ax.set_theta_direction(-1)
    if angles_deg.size and float(np.nanmin(angles_deg)) >= -90.0 and \
            float(np.nanmax(angles_deg)) <= 90.0:
        polar_ax.set_thetamin(-90)
        polar_ax.set_thetamax(90)
    freq = format_number(float(result.frequency), language, decimals=0)
    coeff = format_number(float(result.coefficient), language, decimals=2)
    ax.set_title(
        f"{_t('Predicted diffuser polar response', language)} "
        f"({freq} Hz), "
        f"{_t('Diffusion coefficient $d$ = ', language)}{coeff}"
    )
    localize_axes(ax, language)
    return cast("Axes", ax)


def plot_transfer_matrix(
    matrix: TransferMatrix,
    frequency: Any,
    characteristic_impedance: float,
    ax: Axes | None = None,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Transmission loss and hard-backed absorption of a transfer matrix.

    Reads the ASTM E2611-19 four-pole entries out as the two spectra a
    transmission-tube laboratory quotes: the normal-incidence transmission
    loss ``TLn(f)`` (Eq. (26)) as the primary curve on the left axis, and the
    hard-backed absorption coefficient ``alpha(f)`` (Eq. (28)) as a muted
    companion on a 0..1 right axis.

    :param matrix: A :class:`~phonometry.materials.absorbers.four_microphone.TransferMatrix`.
    :param frequency: Frequency vector ``f``, in hertz, matching the shape of
        the matrix entries.
    :param characteristic_impedance: Characteristic impedance ``rho c`` of the
        air in the tube, in rayls.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param kwargs: Forwarded to the transmission-loss ``plot`` call.
    :return: The axes carrying the transmission-loss curve.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(frequency, dtype=np.float64)
    tl = np.asarray(matrix.transmission_loss(characteristic_impedance),
                    dtype=np.float64)
    alpha = np.asarray(matrix.absorption_hard_backed(characteristic_impedance),
                       dtype=np.float64)
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("label", _t(r"Transmission loss $TL_\mathrm{n}$", language))
    ax.plot(freqs, tl, **kwargs)
    twin = ax.twinx()
    twin.plot(freqs, alpha, ls="--", color=_C_MUTED,
              label=_t(_HARD_BACKED_ALPHA_LABEL, language))
    twin.set_ylim(0.0, 1.05)
    twin.set_ylabel(_t(_HARD_BACKED_ALPHA_LABEL, language))
    format_frequency_axis(ax, float(freqs.min()), float(freqs.max()))
    ax.set_xlabel(_t(_FREQ_LABEL, language))
    ax.set_ylabel(_t(r"Transmission loss $TL_\mathrm{n}$ [dB]", language))
    ax.set_ylim(bottom=0.0)
    ax.set_title(_t("ASTM E2611 transfer-matrix quantities", language))
    lines, labels = ax.get_legend_handles_labels()
    tlines, tlabels = twin.get_legend_handles_labels()
    ax.legend(lines + tlines, labels + tlabels, loc="best", fontsize="small")
    ax.grid(True, which="both", alpha=0.3)
    localize_axes(ax, language)
    localize_axes(twin, language)
    return ax


def plot_metadiffuser_absorption(
    result: MetadiffuserResult, ax: Axes | None = None,
    language: str = "en", **kwargs: Any
) -> Axes:
    """Per-well and face-averaged absorption spectra of a metadiffuser.

    Draws the face-averaged ``alpha(f)`` as the primary curve and each
    well's ``alpha_n(f)`` as a muted companion, labelling only the first
    well to keep the legend compact.

    :param result: A
        :class:`~phonometry.materials.diffusers.metadiffuser.MetadiffuserResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param kwargs: Forwarded to the face-average ``plot`` call.
    :return: The axes.
    """
    freqs = np.asarray(result.frequency, dtype=np.float64)
    kwargs.setdefault("label", _t("Panel average", language))
    ax = _absorption_spectrum_axes(
        ax,
        freqs,
        np.asarray(result.absorption, dtype=np.float64),
        title=_t("Metadiffuser per-well absorption", language),
        language=language,
        **kwargs,
    )
    wells = np.asarray(result.well_absorption, dtype=np.float64)
    for n, alpha_n in enumerate(wells, start=1):
        label = (
            _t("Well {n}", language).format(n=f"1-{wells.shape[0]}")
            if n == 1 else None
        )
        ax.semilogx(freqs, alpha_n, lw=0.9, alpha=0.6, color=_C_MUTED,
                    label=label)
    ax.legend(loc="best", fontsize="small")
    return ax
