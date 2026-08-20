#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Plot renderers for the underwater domain (lazy imports from result .plot())."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import numpy as np

from .common import (
    _C_MUTED,
    _C_PRIMARY,
    _C_QUATERNARY,
    _C_REFERENCE,
    _C_SECONDARY,
    _C_TERTIARY,
    _LEGEND_LOWER_LEFT,
    _LEGEND_UPPER_RIGHT,
    _new_axes,
    _new_axes_column,
    format_frequency_axis,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..underwater.bioacoustics.audiograms import AudiogramResult
    from ..underwater.bioacoustics.weighting import (
        AuditoryWeightingResult,
        WeightedExposureResult,
    )
    from ..underwater.propagation.closed_form import PropagationLossResult
    from ..underwater.propagation.numerical import (
        EigenrayResult,
        GaussianBeamResult,
        NormalModeResult,
        ParabolicEquationResult,
        RayTraceResult,
    )
    from ..underwater.propagation.seabed_reflection import (
        BottomLossResult,
        SeabedReflection,
    )
    from ..underwater.propagation.sound_speed import SoundSpeedProfile
    from ..underwater.propagation.weston_regimes import WestonPropagationResult
    from ..underwater.sonar_equation import DetectionRangeResult, SonarEquationResult
    from ..underwater.sources.ambient_noise import AmbientNoiseResult
    from ..underwater.sources.pile_driving_noise import (
        PileStrikeResult,
        StrikeSelSpectrum,
    )
    from ..underwater.sources.ship_radiated_noise import ShipSourceLevelResult
    from ..underwater.sources.ship_traffic_noise import ShipTrafficSpectrum

#: Axis label and legend placement reused by several renderers in this module
#: (kept as named constants so the literal appears once).
_RANGE_LABEL = "Range [m]"
_RANGE_KM_LABEL = "Range [km]"
_BAND_SEL_LABEL = "Band SEL [dB re 1 µPa²·s]"
_FREQUENCY_LABEL = "Frequency [Hz]"
_TIME_LABEL = "Time [s]"
_DEPTH_LABEL = "Depth [m]"
_GRAZING_ANGLE_LABEL = "Grazing angle [°]"
_PROPAGATION_LOSS_LABEL = "Propagation loss [dB]"
_LEGEND_LOWER_RIGHT: Final = "lower right"

#: Spanish translations of the fixed strings rendered by the underwater
#: ``.plot()`` renderers, keyed by their verbatim English text.  ``_t``
#: returns the English key unchanged for any language other than ``"es"``,
#: so the English output is byte-for-byte identical to the pre-i18n
#: renderers.
_STRINGS: dict[str, str] = {
    r"Source level $L_\mathrm{s}$": r"Nivel de fuente $L_\mathrm{s}$",
    "Radiated noise level": "Nivel de ruido radiado",
    "Frequency [Hz]": "Frecuencia [Hz]",
    "Level [dB re 1 µPa·m]": "Nivel [dB re 1 µPa·m]",
    r"Surface correction $\Delta L$": r"Corrección de superficie $\Delta L$",
    r"Surface correction $\Delta L$ [dB]": r"Corrección de superficie $\Delta L$ [dB]",
    "ISO 17208-2 equivalent monopole source level": "Nivel de fuente monopolar equivalente ISO 17208-2",
    "Peak": "Pico",
    "Pressure [Pa]": "Presión [Pa]",
    "Time [s]": "Tiempo [s]",
    "ISO 18406 pile strike": "Golpe de pilote ISO 18406",
    "Cumulative energy": "Energía acumulada",
    "Cumulative energy (norm.)": "Energía acumulada (norm.)",
    "90 % pulse duration": "Duración del pulso al 90 %",
    "Sound speed [m/s]": "Velocidad del sonido [m/s]",
    "Depth [m]": "Profundidad [m]",
    "Sea-water sound-speed profile": "Perfil de velocidad del sonido en agua de mar",
    "Total PL": "PL total",
    "Spreading": "Divergencia",
    "Absorption": "Absorción",
    "Range [m]": "Distancia [m]",
    "Propagation loss [dB]": "Pérdida de propagación [dB]",
    "Underwater propagation loss": "Pérdida de propagación submarina",
    "Signal excess": "Exceso de señal",
    r"Detection limit ($\mathrm{SE}$ = 0)": r"Límite de detección ($\mathrm{SE}$ = 0)",
    "Figure of merit": "Cifra de mérito",
    "Signal excess [dB]": "Exceso de señal [dB]",
    "Sonar equation": "Ecuación del sonar",
    "Bottom loss": "Pérdida en el fondo",
    "Critical angle": "Ángulo crítico",
    "Grazing angle [°]": "Ángulo rasante [°]",
    "Bottom loss [dB]": "Pérdida en el fondo [dB]",
    "Seabed reflection loss": "Pérdida por reflexión en el fondo marino",
    "Reflection coefficient magnitude": "Módulo del coeficiente de reflexión",
    "Seabed reflection coefficient": "Coeficiente de reflexión del fondo marino",
    "Total": "Total",
    "Wind": "Viento",
    "Thermal": "Térmico",
    "Shipping": "Tráfico marítimo",
    "Spectrum level [dB re 1 µPa²/Hz]": "Nivel espectral [dB re 1 µPa²/Hz]",
    "Ocean ambient noise": "Ruido ambiental oceánico",
    "Source spectral density [dB re 1 µPa²/Hz at 1 m]": "Densidad espectral de la fuente [dB re 1 µPa²/Hz a 1 m]",
    "Ship traffic source level": "Nivel de fuente del tráfico marítimo",
    "modes": "modos",
    "Range [km]": "Distancia [km]",
    "Normal-mode propagation loss": "Pérdida de propagación por modos normales",
    "Source": "Fuente",
    "Seabed": "Fondo marino",
    "Ray trace": "Trazado de rayos",
    "Eigenray arrivals": "Llegadas de eigenrayos",
    "Travel time [s]": "Tiempo de propagación [s]",
    "Boundary reflections": "Reflexiones en los contornos",
    "Reflected paths": "Trayectos reflejados",
    "Refracted or direct": "Refractados o directos",
    "Parabolic-equation propagation loss": "Pérdida de propagación por ecuación parabólica",
    "Gaussian beam propagation loss": "Pérdida de propagación por haces gaussianos",
    "Weston regimes": "Regímenes de Weston",
    "Composite": "Compuesto",
    r"Spherical ($20\,\log_{10} r$)": r"Esférica ($20\,\log_{10} r$)",
    r"Cylindrical ($10\,\log_{10} r$)": r"Cilíndrica ($10\,\log_{10} r$)",
    r"Mode stripping ($15\,\log_{10} r$)": r"Descamado de modos ($15\,\log_{10} r$)",
    "Single mode": "Modo único",
    "Propagation loss [dB re 1 m²]": "Pérdida de propagación [dB re 1 m²]",
    "Hearing threshold [dB]": "Umbral de audición [dB]",
    "Group audiogram": "Audiograma de grupo",
    "Orca audiogram": "Audiograma de orca",
    "Best sensitivity": "Mejor sensibilidad",
    "Weighting $W(f)$ [dB]": "Ponderación $W(f)$ [dB]",
    "Auditory weighting function": "Función de ponderación auditiva",
    "Unweighted": "Sin ponderar",
    "Weighted": "Ponderada",
    "Band SEL [dB re 1 µPa²·s]": "SEL por banda [dB re 1 µPa²·s]",
    "Weighted exposure vs criteria": "Exposición ponderada frente a criterios",
    "Single-strike SEL per band": "SEL por banda de un golpe",
    "Detection range": "Alcance de detección",
    "Propagation loss vs figure of merit": "Pérdida de propagación frente a cifra de mérito",
}


def _t(text: str, language: str = "en") -> str:
    """Localise a fixed string; English is returned verbatim (byte-identical)."""
    return _STRINGS.get(text, text) if language == "es" else text


def plot_ship_source_level(
    result: ShipSourceLevelResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Radiated noise level, source level and the ΔL surface correction.

    Draws the input RNL and the equivalent monopole source level ``Ls`` versus
    frequency, with the Lloyd's-mirror correction ``ΔL`` on a twin axis.

    :param result: A
        :class:`~phonometry.underwater.sources.ship_radiated_noise.ShipSourceLevelResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the source-level ``semilogx`` call.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    rnl = np.asarray(result.radiated_noise_level, dtype=np.float64)
    ls = np.asarray(result.source_level, dtype=np.float64)
    dl = np.asarray(result.surface_correction, dtype=np.float64)

    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("label", _t(r"Source level $L_\mathrm{s}$", language))
    ax.semilogx(freqs, ls, "o-", **kwargs)
    ax.semilogx(
        freqs,
        rnl,
        "s--",
        color=_C_REFERENCE,
        label=_t("Radiated noise level", language),
    )
    ax.set_xlabel(_t(_FREQUENCY_LABEL, language))
    ax.set_ylabel(_t("Level [dB re 1 µPa·m]", language))
    ax.grid(True, which="both", alpha=0.3)
    ax.set_axisbelow(True)

    twin = ax.twinx()
    twin.semilogx(
        freqs,
        dl,
        ":",
        color=_C_TERTIARY,
        label=_t(r"Surface correction $\Delta L$", language),
    )
    twin.set_ylabel(_t(r"Surface correction $\Delta L$ [dB]", language))
    # After twinx() (it re-initialises the shared x-axis with the default log
    # locator) so the octave-band labelling is not reset back to 10^n ticks.
    format_frequency_axis(ax, float(freqs.min()), float(freqs.max()))

    lines, labels = ax.get_legend_handles_labels()
    tlines, tlabels = twin.get_legend_handles_labels()
    ax.legend(lines + tlines, labels + tlabels, loc="best", fontsize="small")
    ax.set_title(
        f"{_t('ISO 17208-2 equivalent monopole source level', language)} "
        rf"($d_\mathrm{{s}}$ = {format_number(result.source_depth, language)} m, "
        rf"$c$ = {format_number(result.sound_speed, language, decimals=0)} m/s)"
    )
    localize_axes(ax, language)
    return ax


def plot_pile_strike(
    result: PileStrikeResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes | np.ndarray:
    """Pile-strike pressure waveform and its cumulative energy.

    Two stacked panels: the pressure waveform with the peak marked on top, and
    the normalised cumulative energy with the 5 %/95 % pulse-duration bounds
    below. With ``ax`` given, only the waveform panel is drawn on it.

    :param result: A :class:`~phonometry.underwater.sources.pile_driving_noise.PileStrikeResult`.
    :param ax: Existing axes for the waveform panel, or ``None`` for a fresh
        two-panel figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the waveform ``plot`` call.
    :return: The waveform axes (``ax`` given) or the array of two axes.
    """
    from .._i18n import format_number, localize_axes

    pressure = np.asarray(result.pressure, dtype=np.float64)
    fs = float(result.fs)
    t = np.arange(pressure.size) / fs
    energy = np.cumsum(pressure**2)
    total = float(energy[-1]) if energy.size else 0.0
    cum = energy / total if total > 0.0 else energy
    peak_idx = int(np.argmax(np.abs(pressure)))
    color = kwargs.pop("color", _C_PRIMARY)

    def _waveform(axw: Axes) -> None:
        axw.plot(t, pressure, color=color, lw=0.8, **kwargs)
        axw.plot(
            [t[peak_idx]],
            [pressure[peak_idx]],
            "o",
            color=_C_REFERENCE,
            label=f"{_t('Peak', language)} ({format_number(result.peak_spl, language, decimals=0)} dB re 1 µPa)",
        )
        axw.set_ylabel(_t("Pressure [Pa]", language))
        axw.grid(True, alpha=0.3)
        axw.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")

    if ax is not None:
        _waveform(ax)
        ax.set_xlabel(_t(_TIME_LABEL, language))
        ax.set_title(
            f"{_t('ISO 18406 pile strike', language)} "
            rf"($\mathrm{{SEL}}_{{\mathrm{{ss}}}}$ = "
            f"{format_number(result.single_strike_sel, language, decimals=0)} dB)"
        )
        localize_axes(ax, language)
        return ax

    axes = _new_axes_column(2, sharex=True, figsize=(8.0, 6.0))
    _waveform(axes[0])
    axes[0].set_title(
        f"{_t('ISO 18406 pile strike', language)} "
        rf"($\mathrm{{SEL}}_{{\mathrm{{ss}}}}$ = "
        f"{format_number(result.single_strike_sel, language, decimals=0)} dB re 1 µPa²·s)"
    )
    axes[1].plot(t, cum, color=_C_TERTIARY, label=_t("Cumulative energy", language))
    for frac in (0.05, 0.95):
        axes[1].axhline(frac, color=_C_MUTED, ls="--", lw=0.8)
    axes[1].set_ylabel(_t("Cumulative energy (norm.)", language))
    axes[1].set_xlabel(_t(_TIME_LABEL, language))
    axes[1].set_title(
        f"{_t('90 % pulse duration', language)} = {format_number(result.pulse_duration * 1e3, language, decimals=0)} ms"
    )
    axes[1].grid(True, alpha=0.3)
    localize_axes(axes[0], language)
    localize_axes(axes[1], language)
    return axes


def plot_sound_speed_profile(
    result: SoundSpeedProfile,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Sound-speed profile: speed vs depth, with depth increasing downward.

    :param result: A :class:`~phonometry.underwater.propagation.sound_speed.SoundSpeedProfile`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the profile ``plot`` call.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    depth = np.asarray(result.depth, dtype=np.float64)
    speed = np.asarray(result.sound_speed, dtype=np.float64)
    label = f"{result.model} $c(z)$"
    ax.plot(speed, depth, **{"color": _C_PRIMARY, "lw": 1.4, "label": label, **kwargs})
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_xlabel(_t("Sound speed [m/s]", language))
    ax.set_ylabel(_t(_DEPTH_LABEL, language))
    ax.set_title(_t("Sea-water sound-speed profile", language))
    ax.grid(True, alpha=0.3)
    ax.legend(loc=_LEGEND_LOWER_LEFT, fontsize="small")
    localize_axes(ax, language)
    return ax


def plot_propagation_loss(
    result: PropagationLossResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Propagation loss versus range, with spreading and absorption split out.

    Loss increases downward (the usual PL convention).

    :param result: A :class:`~phonometry.underwater.propagation.PropagationLossResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the total-PL ``plot`` call.
    :return: The axes.
    """
    from .._i18n import decimal_comma, localize_axes

    ax = ax if ax is not None else _new_axes()
    r = np.asarray(result.range_m, dtype=np.float64)
    label = f"{_t('Total PL', language)} ({decimal_comma(f'{result.frequency / 1000.0:.3g}', language)} kHz)"
    ax.plot(
        r,
        np.asarray(result.pl),
        **{"color": _C_PRIMARY, "lw": 1.6, "label": label, **kwargs},
    )
    ax.plot(
        r,
        np.asarray(result.spreading),
        color=_C_MUTED,
        lw=1.0,
        ls="--",
        label=f"{_t('Spreading', language)} ({result.law})",
    )
    ax.plot(
        r,
        np.asarray(result.absorption),
        color=_C_SECONDARY,
        lw=1.0,
        ls=":",
        label=f"{_t('Absorption', language)} ({decimal_comma(f'{result.absorption_coefficient:.3g}', language)} dB/km)",
    )
    ax.set_xlabel(_t(_RANGE_LABEL, language))
    ax.set_ylabel(_t(_PROPAGATION_LOSS_LABEL, language))
    ax.set_title(f"{_t('Underwater propagation loss', language)} ({result.model})")
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(loc=_LEGEND_LOWER_RIGHT, fontsize="small")
    localize_axes(ax, language)
    return ax


def plot_sonar_equation(
    result: SonarEquationResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Signal excess versus propagation loss, with the detection limit (SE = 0).

    :param result: A :class:`~phonometry.underwater.sonar_equation.SonarEquationResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the signal-excess ``plot`` call.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    pl = np.asarray(result.propagation_loss, dtype=np.float64)
    se = np.asarray(result.signal_excess, dtype=np.float64)
    order = np.argsort(pl)
    label = f"{_t('Signal excess', language)} ({result.mode})"
    ax.plot(
        pl[order],
        se[order],
        **{"color": _C_PRIMARY, "lw": 1.6, "label": label, **kwargs},
    )
    ax.axhline(
        0.0,
        color=_C_REFERENCE,
        ls="--",
        lw=1.0,
        label=_t(r"Detection limit ($\mathrm{SE}$ = 0)", language),
    )
    ax.axvline(
        result.figure_of_merit,
        color=_C_MUTED,
        ls=":",
        lw=1.0,
        label=f"{_t('Figure of merit', language)} = {format_number(result.figure_of_merit, language)} dB",
    )
    ax.set_xlabel(_t(_PROPAGATION_LOSS_LABEL, language))
    ax.set_ylabel(_t("Signal excess [dB]", language))
    ax.set_title(_t("Sonar equation", language))
    ax.grid(True, alpha=0.3)
    ax.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")
    localize_axes(ax, language)
    return ax


def plot_bottom_loss(
    result: BottomLossResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Bottom reflection loss versus grazing angle, marking the critical angle.

    :param result: A :class:`~phonometry.underwater.propagation.seabed_reflection.BottomLossResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the bottom-loss ``plot`` call.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    phi = np.asarray(result.grazing_angle, dtype=np.float64)
    loss = np.asarray(result.reflection_loss, dtype=np.float64)
    ax.plot(
        phi,
        loss,
        **{
            "color": _C_PRIMARY,
            "lw": 1.6,
            "label": _t("Bottom loss", language),
            **kwargs,
        },
    )
    if result.critical_angle is not None:
        ax.axvline(
            result.critical_angle,
            color=_C_REFERENCE,
            ls="--",
            lw=1.0,
            label=f"{_t('Critical angle', language)} = {format_number(result.critical_angle, language)}°",
        )
    ax.set_xlabel(_t(_GRAZING_ANGLE_LABEL, language))
    ax.set_ylabel(_t("Bottom loss [dB]", language))
    ax.set_title(_t("Seabed reflection loss", language))
    ax.grid(True, alpha=0.3)
    ax.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")
    localize_axes(ax, language)
    return ax


def plot_seabed_reflection(
    result: SeabedReflection,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Seabed reflection-coefficient magnitude versus grazing angle.

    Draws ``|R|`` on a linear grazing-angle axis, marking the critical angle
    when the sediment is faster than the water.

    :param result: A :class:`~phonometry.underwater.propagation.seabed_reflection.SeabedReflection`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the magnitude ``plot`` call.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    phi = np.asarray(result.grazing_angle, dtype=np.float64)
    magnitude = np.asarray(result.magnitude, dtype=np.float64)
    ax.plot(
        phi, magnitude, **{"color": _C_PRIMARY, "lw": 1.6, "label": "$|R|$", **kwargs}
    )
    if result.critical_angle is not None:
        ax.axvline(
            result.critical_angle,
            color=_C_REFERENCE,
            ls="--",
            lw=1.0,
            label=f"{_t('Critical angle', language)} = {format_number(result.critical_angle, language)}°",
        )
    ax.set_xlabel(_t(_GRAZING_ANGLE_LABEL, language))
    ax.set_xlim(0.0, 90.0)
    ax.set_ylabel(f"{_t('Reflection coefficient magnitude', language)} $|R|$")
    ax.set_title(_t("Seabed reflection coefficient", language))
    ax.grid(True, alpha=0.3)
    ax.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")
    localize_axes(ax, language)
    return ax


def plot_ambient_noise(
    result: AmbientNoiseResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Composite ambient-noise spectrum and its components versus frequency.

    :param result: An :class:`~phonometry.underwater.sources.ambient_noise.AmbientNoiseResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the composite-level ``plot`` call.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    f = np.asarray(result.frequency, dtype=np.float64)
    label = f"{_t('Total', language)} ({format_number(result.wind_speed_knots, language)} kn)"
    ax.plot(
        f,
        np.asarray(result.spectrum_level),
        **{"color": _C_PRIMARY, "lw": 1.8, "label": label, **kwargs},
    )
    ax.plot(
        f,
        np.asarray(result.wind),
        color=_C_SECONDARY,
        lw=1.0,
        ls="--",
        label=_t("Wind", language),
    )
    ax.plot(
        f,
        np.asarray(result.thermal),
        color=_C_TERTIARY,
        lw=1.0,
        ls=":",
        label=_t("Thermal", language),
    )
    if result.shipping is not None:
        ax.plot(
            f,
            np.asarray(result.shipping),
            color=_C_REFERENCE,
            lw=1.0,
            ls="-.",
            label=_t("Shipping", language),
        )
    ax.set_xscale("log")
    ax.set_xlabel(_t(_FREQUENCY_LABEL, language))
    ax.set_ylabel(_t("Spectrum level [dB re 1 µPa²/Hz]", language))
    ax.set_title(_t("Ocean ambient noise", language))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")
    format_frequency_axis(ax, float(f.min()), float(f.max()))
    localize_axes(ax, language)
    return ax


def plot_ship_traffic_spectrum(
    result: ShipTrafficSpectrum,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Predicted ship source spectral-density level versus frequency.

    :param result: A :class:`~phonometry.underwater.sources.ship_traffic_noise.ShipTrafficSpectrum`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the source-PSD ``plot`` call.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    f = np.asarray(result.frequency, dtype=np.float64)
    psd = np.asarray(result.source_psd, dtype=np.float64)
    if result.vessel_class is not None:
        label = f"{result.model} ({result.vessel_class})"
    else:
        label = result.model
    ax.plot(f, psd, **{"color": _C_PRIMARY, "lw": 1.6, "label": label, **kwargs})
    ax.set_xscale("log")
    ax.set_xlabel(_t(_FREQUENCY_LABEL, language))
    ax.set_ylabel(_t("Source spectral density [dB re 1 µPa²/Hz at 1 m]", language))
    ax.set_title(_t("Ship traffic source level", language))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")
    format_frequency_axis(ax, float(f.min()), float(f.max()))
    localize_axes(ax, language)
    return ax


def plot_normal_modes(
    result: NormalModeResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Normal-mode propagation loss versus range (loss increasing downward).

    :param result: A :class:`~phonometry.underwater.propagation.numerical.NormalModeResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the propagation-loss ``plot`` call.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    r = np.asarray(result.ranges, dtype=np.float64)
    pl = np.asarray(result.propagation_loss, dtype=np.float64)
    label = f"{result.wavenumbers.size} {_t('modes', language)} ({format_number(result.frequency, language, decimals=0)} Hz)"
    ax.plot(
        r / 1000.0, pl, **{"color": _C_PRIMARY, "lw": 1.2, "label": label, **kwargs}
    )
    ax.set_xlabel(_t(_RANGE_KM_LABEL, language))
    ax.set_ylabel(_t(_PROPAGATION_LOSS_LABEL, language))
    ax.set_title(_t("Normal-mode propagation loss", language))
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(loc=_LEGEND_LOWER_RIGHT, fontsize="small")
    localize_axes(ax, language)
    return ax


def plot_ray_trace(
    result: RayTraceResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Ray paths through the water column (depth increasing downward).

    :param result: A :class:`~phonometry.underwater.propagation.numerical.RayTraceResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to each ray ``plot`` call.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    r = np.asarray(result.ranges, dtype=np.float64)
    z = np.asarray(result.depths, dtype=np.float64)
    for i in range(r.shape[0]):
        ax.plot(
            r[i] / 1000.0,
            z[i],
            **{"color": _C_PRIMARY, "lw": 0.7, "alpha": 0.7, **kwargs},
        )
    _draw_bathymetry(ax, result, float(np.max(r)), language, labelled=True)
    ax.plot(
        [0.0],
        [result.source_depth],
        "o",
        color=_C_REFERENCE,
        label=_t("Source", language),
    )
    ax.set_xlabel(_t(_RANGE_KM_LABEL, language))
    ax.set_ylabel(_t(_DEPTH_LABEL, language))
    ax.set_title(_t("Ray trace", language))
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(loc=_LEGEND_LOWER_RIGHT, fontsize="small")
    localize_axes(ax, language)
    return ax


def _draw_bathymetry(
    ax: Axes,
    result: Any,
    r_max: float,
    language: str,
    *,
    labelled: bool,
) -> None:
    """Draw the bottom polyline of a sloping-bathymetry result, if it has one.

    The polyline continues level past its last node, exactly as the solvers
    clamp it, so the drawn line is the boundary the rays actually reflected
    off rather than a segment that stops mid-picture.
    """
    br = getattr(result, "bathymetry_ranges", None)
    bd = getattr(result, "bathymetry_depths", None)
    if br is None or bd is None:
        return
    br = np.asarray(br, dtype=np.float64)
    bd = np.asarray(bd, dtype=np.float64)
    if br[-1] < r_max:
        br = np.append(br, r_max)
        bd = np.append(bd, bd[-1])
    label = _t("Seabed", language) if labelled else None
    ax.plot(br / 1000.0, bd, color=_C_SECONDARY, lw=1.4, label=label)


def plot_eigenrays(
    result: EigenrayResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Arrival structure of the eigenrays: per-path loss stems against delay.

    Each eigenray is a stem at its travel time whose head sits at the
    propagation loss of that single path (``-20 lg|a|``, increasing downward
    like every loss axis of the domain), so the picture is the channel's
    impulse-response skeleton: the refracted or direct paths (no boundary
    touches) in the primary colour, the reflected multipath coloured by its
    total count of boundary touches, which is what separates the arrival
    families the way Jensen Fig. 3.7 colours its eigenrays.

    :param result: An
        :class:`~phonometry.underwater.propagation.numerical.EigenrayResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the direct/refracted marker ``plot`` call.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    t = np.asarray(result.travel_times, dtype=np.float64)
    amp = np.abs(np.asarray(result.amplitudes))
    tiny = np.finfo(np.float64).tiny
    loss = -20.0 * np.log10(np.maximum(amp, tiny))
    bounces = np.asarray(
        result.surface_reflections + result.bottom_reflections, dtype=np.int_
    )
    if t.size:
        # Stems hang from the quiet end of the window (the axis is inverted
        # below, so that end is the bottom of the picture).
        base = float(loss.max()) + 6.0
        direct = bounces == 0
        reflected = ~direct
        if np.any(reflected):
            ax.vlines(
                t[reflected],
                base,
                loss[reflected],
                color=_C_MUTED,
                lw=0.6,
                alpha=0.6,
                zorder=2,
            )
            sc = ax.scatter(
                t[reflected],
                loss[reflected],
                c=bounces[reflected],
                cmap="viridis",
                s=16,
                zorder=3,
                label=_t("Reflected paths", language),
            )
            cbar = ax.figure.colorbar(sc, ax=ax, pad=0.02)
            cbar.set_label(_t("Boundary reflections", language))
        if np.any(direct):
            ax.vlines(t[direct], base, loss[direct], color=_C_PRIMARY, lw=1.4, zorder=4)
            kwargs.setdefault("label", _t("Refracted or direct", language))
            ax.plot(
                t[direct], loss[direct], "o", color=_C_PRIMARY, ms=6, zorder=5, **kwargs
            )
        ax.set_ylim(base, float(loss.min()) - 3.0)
        ax.legend(loc=_LEGEND_LOWER_RIGHT, fontsize="small")
    elif not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_xlabel(_t("Travel time [s]", language))
    ax.set_ylabel(_t(_PROPAGATION_LOSS_LABEL, language))
    ax.set_title(_t("Eigenray arrivals", language))
    ax.grid(True, alpha=0.3)
    localize_axes(ax, language)
    return ax


def _plot_loss_field(
    ax: Axes | None,
    ranges: np.ndarray,
    depths: np.ndarray,
    pl: np.ndarray,
    *,
    title: str,
    language: str,
    kwargs: dict[str, Any],
) -> Axes:
    """A propagation-loss field on a range-depth grid, shared by two solvers.

    ``imshow`` renders it as a single raster image rather than one vector quad
    per cell, which avoids moiré and keeps the figure light; ``pcolormesh``
    would put tens of thousands of paths into an SVG for the same picture.

    THE WINDOW IS ANCHORED AT THE LOUD END. 50 dB of range has to be placed
    somewhere, and the only robust end to hang it on is the strong one: a loss
    field is bounded below by the strongest arrival and unbounded above, because
    a shadow zone runs off to whatever number the solver's own noise floor
    allows. Anchoring on a *high* percentile instead lets the empty part of the
    picture decide where the window sits, and the emptier the field the further
    it drags it: on the caustic case of the solvers guide, 18% of the Gaussian
    beam field is infinite and its 95th percentile of finite loss is 160 dB, so
    a window hung there covers 110 to 160 dB and 85% of the field clips to one
    flat colour. It is not only the beams. The parabolic equation on the same
    case puts 74% of its cells under such a window. Hung at the 5th percentile
    and run 50 dB up, both show their caustics and their interference lobes,
    with 5% saturating at the bright end by construction.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    r = np.asarray(ranges, dtype=np.float64) / 1000.0
    z = np.asarray(depths, dtype=np.float64)
    pl = np.asarray(pl, dtype=np.float64)
    finite = pl[np.isfinite(pl)]
    vmin = float(np.percentile(finite, 5)) if finite.size else 50.0
    vmax = vmin + 50.0
    # The un-illuminated wedge is infinite; clip it to the quiet end of the
    # window so imshow renders it as the deepest shadow rather than as a hole.
    pl = np.where(np.isfinite(pl), pl, vmax)
    img = ax.imshow(
        pl,
        **{
            "cmap": "viridis_r",
            "vmin": vmin,
            "vmax": vmax,
            "aspect": "auto",
            "origin": "upper",
            "interpolation": "bilinear",
            "extent": (float(r[0]), float(r[-1]), float(z[-1]), float(z[0])),
            **kwargs,
        },
    )
    ax.figure.colorbar(img, ax=ax, label=_t(_PROPAGATION_LOSS_LABEL, language))
    ax.set_xlabel(_t(_RANGE_KM_LABEL, language))
    ax.set_ylabel(_t(_DEPTH_LABEL, language))
    ax.set_title(_t(title, language))
    localize_axes(ax, language)
    return ax


def plot_parabolic_equation(
    result: ParabolicEquationResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Parabolic-equation propagation-loss field (range x depth).

    :param result: A
        :class:`~phonometry.underwater.propagation.numerical.ParabolicEquationResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to ``imshow``.
    :return: The axes.
    """
    return _plot_loss_field(
        ax,
        result.ranges,
        result.depths,
        result.propagation_loss,
        title="Parabolic-equation propagation loss",
        language=language,
        kwargs=kwargs,
    )


def plot_gaussian_beams(
    result: GaussianBeamResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Gaussian beam propagation-loss field (range x depth).

    Drawn on the same axes, the same colour map and a window of the same 50 dB
    width as :func:`plot_parabolic_equation`. The width is shared; the
    *placement* is not, because each anchors on its own field, and two solvers
    over the same ocean do not agree on the loudest twentieth of it. Measured on
    the caustic case of the solvers guide the beams come out on [50.2, 100.2] dB
    and the parabolic equation on [53.3, 103.3], so a difference read off the
    two colour bars carries 3.1 dB that is the framing rather than the physics.
    Pass an explicit ``vmin`` and ``vmax`` to both, which go straight to
    ``imshow``, whenever they are meant to be compared cell by cell; the depth
    grids want pinning together the same way, since ``n_depth_points`` defaults
    to 200 here and 1024 there.

    :param result: A
        :class:`~phonometry.underwater.propagation.numerical.GaussianBeamResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to ``imshow``.
    :return: The axes.
    """
    ax = _plot_loss_field(
        ax,
        result.ranges,
        result.depths,
        result.propagation_loss,
        title="Gaussian beam propagation loss",
        language=language,
        kwargs=kwargs,
    )
    _draw_bathymetry(ax, result, float(np.max(result.ranges)), language, labelled=False)
    return ax


def _plottable(levels: Any) -> np.ndarray:
    """Levels with non-finite entries as ``nan``, which matplotlib skips.

    Band levels carry ``-inf`` where a band holds no energy at all; drawing
    that literally would collapse the vertical scale.
    """
    values = np.asarray(levels, dtype=np.float64)
    return np.where(np.isfinite(values), values, np.nan)


def _spectrum_axes(
    ax: Axes | None,
    freqs: np.ndarray,
    *,
    ylabel: str,
    title: str,
    language: str,
) -> Axes:
    """Shared frame for the frequency-domain underwater renderers."""
    ax = ax if ax is not None else _new_axes()
    ax.set_xscale("log")
    ax.set_xlabel(_t(_FREQUENCY_LABEL, language))
    ax.set_ylabel(_t(ylabel, language))
    ax.set_title(_t(title, language))
    ax.grid(True, which="both", alpha=0.3)
    ax.set_axisbelow(True)
    format_frequency_axis(ax, float(np.min(freqs)), float(np.max(freqs)))
    return ax


def plot_weston_regimes(
    result: WestonPropagationResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Composite Weston propagation loss with each regime's law and boundaries.

    :param result: A
        :class:`~phonometry.underwater.propagation.weston_regimes.WestonPropagationResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the composite-loss ``plot`` call.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    r = np.asarray(result.range_m, dtype=np.float64)
    ax.set_xscale("log")
    for label, curve, color, style in (
        (r"Spherical ($20\,\log_{10} r$)", result.spherical, _C_MUTED, ":"),
        (r"Cylindrical ($10\,\log_{10} r$)", result.cylindrical, _C_SECONDARY, "--"),
        (
            r"Mode stripping ($15\,\log_{10} r$)",
            result.mode_stripping,
            _C_TERTIARY,
            "-.",
        ),
        ("Single mode", result.single_mode, _C_QUATERNARY, (0, (3, 1, 1, 1))),
    ):
        ax.plot(
            r,
            np.asarray(curve, dtype=np.float64),
            ls=style,
            lw=1.0,
            color=color,
            label=_t(label, language),
        )
    ax.plot(
        r,
        np.asarray(result.propagation_loss, dtype=np.float64),
        **{
            "color": _C_PRIMARY,
            "lw": 2.0,
            "label": _t("Composite", language),
            **kwargs,
        },
    )
    for boundary in (
        result.boundaries.spherical_to_cylindrical,
        result.boundaries.cylindrical_to_mode_stripping,
        result.boundaries.mode_stripping_to_single_mode,
    ):
        if np.isfinite(boundary) and r[0] <= boundary <= r[-1]:
            ax.axvline(boundary, color=_C_REFERENCE, ls="--", lw=0.8, alpha=0.7)
    finite = np.asarray(result.propagation_loss)[np.isfinite(result.propagation_loss)]
    if finite.size:
        ax.set_ylim(float(finite.max()) + 10.0, float(finite.min()) - 5.0)
    ax.set_xlabel(_t(_RANGE_LABEL, language))
    ax.set_ylabel(_t("Propagation loss [dB re 1 m²]", language))
    ax.set_title(
        f"{_t('Weston regimes', language)} "
        f"({format_number(result.frequency, language, decimals=0)} Hz, "
        f"$H$ = {format_number(result.water_depth, language, decimals=0)} m, {result.seabed})"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc=_LEGEND_LOWER_LEFT, fontsize="small")
    localize_axes(ax, language)
    return ax


def plot_marine_mammal_audiogram(
    result: AudiogramResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Hearing threshold versus frequency with the point of best sensitivity.

    :param result: An
        :class:`~phonometry.underwater.bioacoustics.audiograms.AudiogramResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the threshold ``plot`` call.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    freqs = np.asarray(result.frequencies, dtype=np.float64)
    title = "Orca audiogram" if result.group == "orca" else "Group audiogram"
    ax = _spectrum_axes(
        ax, freqs, ylabel="Hearing threshold [dB]", title=title, language=language
    )
    ax.plot(
        freqs,
        np.asarray(result.threshold, dtype=np.float64),
        **{"color": _C_PRIMARY, "lw": 1.4, "label": result.group, **kwargs},
    )
    ax.plot(
        [result.best_frequency],
        [result.best_threshold],
        "o",
        color=_C_REFERENCE,
        label=(
            f"{_t('Best sensitivity', language)}: "
            f"{format_number(result.best_threshold, language)} dB"
        ),
    )
    ax.legend(loc="best", fontsize="small")
    localize_axes(ax, language)
    return ax


def plot_auditory_weighting(
    result: AuditoryWeightingResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Auditory weighting function of a marine-mammal hearing group.

    :param result: An
        :class:`~phonometry.underwater.bioacoustics.weighting.AuditoryWeightingResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the weighting ``plot`` call.
    :return: The axes.
    """
    from .._i18n import localize_axes

    freqs = np.asarray(result.frequencies, dtype=np.float64)
    ax = _spectrum_axes(
        ax,
        freqs,
        ylabel="Weighting $W(f)$ [dB]",
        title="Auditory weighting function",
        language=language,
    )
    label = f"{result.group} ({result.guidance})"
    ax.plot(
        freqs,
        np.asarray(result.weighting, dtype=np.float64),
        **{"color": _C_PRIMARY, "lw": 1.4, "label": label, **kwargs},
    )
    ax.axhline(0.0, color=_C_MUTED, ls=":", lw=0.8)
    ax.legend(loc="lower center", fontsize="small")
    localize_axes(ax, language)
    return ax


def plot_weighted_exposure(
    result: WeightedExposureResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Unweighted and weighted band spectra against the exposure criteria.

    :param result: A
        :class:`~phonometry.underwater.bioacoustics.weighting.WeightedExposureResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the weighted-spectrum ``plot`` call.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    freqs = np.asarray(result.frequencies, dtype=np.float64)
    ax = _spectrum_axes(
        ax,
        freqs,
        ylabel=_BAND_SEL_LABEL,
        title="Weighted exposure vs criteria",
        language=language,
    )
    ax.plot(
        freqs,
        _plottable(result.band_sel),
        "o--",
        ms=3,
        color=_C_MUTED,
        lw=1.0,
        label=_t("Unweighted", language),
    )
    ax.plot(
        freqs,
        _plottable(result.weighted_band_sel),
        **{
            "color": _C_PRIMARY,
            "lw": 1.6,
            "marker": "o",
            "ms": 3,
            "label": f"{_t('Weighted', language)} ({result.group}, {result.guidance})",
            **kwargs,
        },
    )
    for level, color, name in (
        (result.criteria.tts_sel, _C_SECONDARY, "TTS"),
        (result.criteria.injury_sel, _C_REFERENCE, result.criteria.injury_label),
    ):
        if level is not None:
            ax.axhline(
                level,
                color=color,
                ls="--",
                lw=1.2,
                label=f"{name} {format_number(level, language, decimals=0)} dB",
            )
    ax.axhline(
        result.cumulative_sel,
        color=_C_TERTIARY,
        ls="-.",
        lw=1.2,
        label=(
            rf"$\mathrm{{SEL}}_{{\mathrm{{cum}}}}$ "
            rf"{format_number(result.cumulative_sel, language)} dB "
            rf"($N$ = {result.n_events})"
        ),
    )
    ax.legend(loc="best", fontsize="small")
    localize_axes(ax, language)
    return ax


def plot_strike_sel_spectrum(
    result: StrikeSelSpectrum,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Per-band single-strike sound exposure level of a pile strike.

    :param result: A
        :class:`~phonometry.underwater.sources.pile_driving_noise.StrikeSelSpectrum`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the band-level ``plot`` call.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    freqs = np.asarray(result.frequencies, dtype=np.float64)
    ax = _spectrum_axes(
        ax,
        freqs,
        ylabel=_BAND_SEL_LABEL,
        title="Single-strike SEL per band",
        language=language,
    )
    ax.plot(
        freqs,
        _plottable(result.band_sel),
        **{
            "color": _C_PRIMARY,
            "lw": 1.4,
            "marker": "o",
            "ms": 3,
            "label": f"1/{result.fraction}",
            **kwargs,
        },
    )
    ax.axhline(
        result.total_sel,
        color=_C_REFERENCE,
        ls="--",
        lw=1.2,
        label=(
            rf"$\mathrm{{SEL}}_{{\mathrm{{ss}}}}$ "
            rf"{format_number(result.total_sel, language)} dB"
        ),
    )
    ax.legend(loc="best", fontsize="small")
    localize_axes(ax, language)
    return ax


def plot_detection_range(
    result: DetectionRangeResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Propagation loss against the figure of merit, with the detection range.

    :param result: A
        :class:`~phonometry.underwater.sonar_equation.DetectionRangeResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the propagation-loss ``plot`` call.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    r = np.asarray(result.range_m, dtype=np.float64)
    ax.plot(
        r,
        np.asarray(result.propagation_loss, dtype=np.float64),
        **{"color": _C_PRIMARY, "lw": 1.4, "label": _t("Total PL", language), **kwargs},
    )
    ax.axhline(
        result.figure_of_merit,
        color=_C_SECONDARY,
        ls="--",
        lw=1.2,
        label=(
            f"{_t('Figure of merit', language)} "
            f"{format_number(result.figure_of_merit, language)} dB"
        ),
    )
    if np.isfinite(result.detection_range) and result.detection_range > 0.0:
        ax.axvline(
            result.detection_range,
            color=_C_REFERENCE,
            ls=":",
            lw=1.2,
            label=(
                f"{_t('Detection range', language)} "
                f"{format_number(result.detection_range, language, decimals=0)} m"
            ),
        )
    ax.set_xlabel(_t(_RANGE_LABEL, language))
    ax.set_ylabel(_t(_PROPAGATION_LOSS_LABEL, language))
    ax.set_title(_t("Propagation loss vs figure of merit", language))
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc=_LEGEND_LOWER_LEFT, fontsize="small")
    localize_axes(ax, language)
    return ax
