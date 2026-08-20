#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Plot renderers for the electroacoustics domain (lazy imports from result .plot())."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import numpy as np

from .common import (
    _C_EDGE,
    _C_PRIMARY,
    _C_PRIMARY_LIGHT,
    _C_REFERENCE,
    _C_SECONDARY,
    _C_TERTIARY,
    _LEGEND_UPPER_RIGHT,
    _format_freq,
    _import_pyplot,
    _new_axes,
    _new_axes_column,
    format_frequency_axis,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..electroacoustics.distortion import HarmonicDistortionResult
    from ..electroacoustics.frequency_response import FrequencyResponseResult
    from ..electroacoustics.intermodulation import ModulationDistortionResult
    from ..electroacoustics.loudspeaker import LoudspeakerCharacteristics
    from ..electroacoustics.microphone import MicrophoneCharacteristics
    from ..electroacoustics.piston import PistonDirectivity, RadiatingPistonResult
    from ..electroacoustics.sound_reinforcement import FeedbackStabilityResult
    from ..electroacoustics.swept_sine import SweptSineDistortionResult

#: Shared frequency-axis label of the electroacoustics renderers.
_FREQ_LABEL = "Frequency [Hz]"
#: Excitation-frequency axis label of the swept-sine panels.
_EXCITATION_FREQ_LABEL = "Excitation frequency [Hz]"
#: Magnitude ordinate label of the Bode-style magnitude panels.
_MAGNITUDE_LABEL = "Magnitude [dB]"
#: Total-harmonic-distortion ordinate label, in percent.
_THD_LABEL = "THD [%]"
#: Panel title of the total-harmonic-distortion characteristic.
_THD_TITLE = "Total harmonic distortion"
#: Sound-pressure-level axis label of the datasheet panels.
_SPL_LABEL = "Sound pressure level [dB]"
#: Legend entry and title of the loudspeaker on-axis response panel.
_ON_AXIS_RESPONSE = "On-axis response"
#: Legend entry and title of the microphone free-field response panel.
_FREE_FIELD_RESPONSE = "Free-field response"
#: Legend entry of the rated tolerance band (formatted with ``tol``).
_TOLERANCE_LABEL = "Tolerance ±{tol} dB"
#: Legend entry of the effective-frequency-range markers.
_EFFECTIVE_RANGE = "Effective range"
#: Matplotlib ``textcoords`` mode for annotations displaced in points.
_OFFSET_POINTS = "offset points"
#: Legend placement of the wide datasheet response panels.
_LEGEND_LOWER_CENTER: Final = "lower center"

#: Datasheet-panel palette shared by the IEC 60268-4/-5 ``.report()`` fiches and
#: the ``.plot()`` data sheets. The tolerance band is drawn as a *pale opaque*
#: fill, not a translucent one: the report's PDF vector backend (svglib) does
#: not preserve alpha, so a translucent fill would render as a solid block that
#: hides the response curve.
_C_TOL_BAND = _C_PRIMARY_LIGHT
#: Unicode ohm sign for the impedance-panel axis label.
_OHM_TEXT = "Ω"
#: IEC 60263 clause 3 polar reference-circle span, in dB (radius = 25 dB).
_POLAR_SPAN_DB = 25.0
#: Radial span of the baffled-piston beam pattern, in dB (40 dB keeps the first
#: side lobes of a narrow high-``ka`` main lobe on scale below the 0 dB axis).
_POLAR_SPAN_PISTON_DB = 40.0
#: Ordinate span of the loudspeaker on-axis response panel, in dB (a multiple of
#: 25 keeps the IEC 60263 25 dB-per-decade grid on whole gridlines).
_RESPONSE_SPAN_LSP = 50.0
#: Ordinate span of the microphone free-field response panel, in dB (one 25 dB
#: span suits the small deviations of a microphone response around 0 dB).
_RESPONSE_SPAN_MIC = 25.0

#: Spanish translations of the fixed strings rendered by the
#: electroacoustics ``.plot()`` renderers, keyed by their verbatim English
#: text. ``_t`` returns the English key unchanged for any language other
#: than ``"es"``, so the English output is byte-for-byte identical to the
#: pre-i18n renderers.
_STRINGS: dict[str, str] = {
    _FREQ_LABEL: "Frecuencia [Hz]",
    _MAGNITUDE_LABEL: "Magnitud [dB]",
    "Phase [deg]": "Fase [grados]",
    r"Coherence $\gamma^2$": r"Coherencia $\gamma^2$",
    r"Harmonic order $n$  ($f = n \cdot f_1$)": r"Orden del armónico $n$  ($f = n \cdot f_1$)",
    "Level re fundamental [dB]": "Nivel respecto al fundamental [dB]",
    "Harmonics": "Armónicos",
    "Frequency response": "Respuesta en frecuencia",
    " and coherence": " y coherencia",
    _THD_LABEL: "THD [%]",
    _EXCITATION_FREQ_LABEL: "Frecuencia de excitación [Hz]",
    "Swept-sine THD (Farina / Novak)": "THD de barrido sinusoidal (Farina / Novak)",
    "Harmonic frequency responses ({method} sweep)": "Respuestas en frecuencia de los armónicos (barrido {method})",
    "$R_1$ (resistance)": "$R_1$ (resistencia)",
    "$X_1$ (reactance)": "$X_1$ (reactancia)",
    r"Normalized radiation impedance $Z_\mathrm{r} / \rho c S$": r"Impedancia de radiación normalizada $Z_\mathrm{r} / \rho c S$",
    "Baffled circular piston radiation impedance": "Impedancia de radiación de un pistón circular con pantalla",
    "Baffled circular piston directivity": "Directividad de un pistón circular con pantalla",
    # --- IEC 60268-4/-5 datasheet panels (shared by .report() and .plot()) ---
    _SPL_LABEL: "Nivel de presión acústica [dB]",
    _ON_AXIS_RESPONSE: "Respuesta en el eje",
    _TOLERANCE_LABEL: "Tolerancia ±{tol} dB",
    "−10 dB reference": "Referencia −10 dB",
    _EFFECTIVE_RANGE: "Rango efectivo",
    "Impedance $|Z|$ [{ohm}]": "Impedancia $|Z|$ [{ohm}]",
    "Impedance": "Impedancia",
    "Rated impedance": "Impedancia nominal",
    "80 % of rated": "80 % de la nominal",
    _THD_TITLE: "Distorsión armónica total",
    "Directional response": "Respuesta direccional",
    "Directional response at {freq} Hz": "Respuesta direccional a {freq} Hz",
    "Reference frequency": "Frecuencia de referencia",
    _FREE_FIELD_RESPONSE: "Respuesta en campo libre",
    "Relative response [dB]": "Respuesta relativa [dB]",
    "Band level [dB]": "Nivel de banda [dB]",
    "Inherent noise spectrum": "Espectro de ruido inherente",
    "{thd} % limit": "Límite del {thd} %",
    "Max. SPL": "SPL máx.",
    "Carrier $f_2$": "Portadora $f_2$",
    r"Sidebands $f_2 \pm n \cdot f_1$": r"Bandas laterales $f_2 \pm n \cdot f_1$",
    "Level re carrier [dB]": "Nivel respecto a la portadora [dB]",
    # --- Sound-reinforcement gain before feedback (Long Ch. 18) ---
    "Open-loop gain $Z_\\mathrm{S}$": "Ganancia en lazo abierto $Z_\\mathrm{S}$",
    "Feedback-loop gain $G_\\mathrm{S}$": "Ganancia del lazo de realimentación $G_\\mathrm{S}$",
    r"Open microphones $\Delta L_{\mathrm{nom}}$": r"Micrófonos abiertos $\Delta L_{\mathrm{nom}}$",
    "Total loop gain": "Ganancia total del lazo",
    "Gain [dB]": "Ganancia [dB]",
    "Oscillation (0 dB)": "Oscilación (0 dB)",
    "Stability limit ({margin} dB margin)": "Límite de estabilidad (margen de {margin} dB)",
    "Gain before feedback — stable, {head} dB spare": "Ganancia antes de la realimentación — estable, {head} dB de reserva",
    "Gain before feedback — unstable by {head} dB": "Ganancia antes de la realimentación — inestable en {head} dB",
    "Talker (T)": "Hablante (T)",
    "Microphone (M)": "Micrófono (M)",
    "Loudspeaker (H)": "Altavoz (H)",
    "Listener (L)": "Oyente (L)",
    "Sound-reinforcement feedback geometry": "Geometría de realimentación del sistema de refuerzo sonoro",
}


def _t(text: str, language: str = "en", **fmt: Any) -> str:
    """Localise a fixed string; English is returned verbatim (byte-identical)."""
    s = _STRINGS.get(text, text) if language == "es" else text
    return s.format(**fmt) if fmt else s


def plot_harmonic_distortion(
    result: HarmonicDistortionResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Harmonic amplitude spectrum with the harmonics marked and THD annotated.

    Draws the fundamental and its harmonics as a stem-style amplitude spectrum
    in dB relative to the fundamental, annotated with the THD (both
    conventions) and SINAD.

    :param result: A :class:`~phonometry.electroacoustics.distortion.HarmonicDistortionResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the marker ``plot`` call.
    :return: The axes.
    """
    from .._i18n import decimal_comma, format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    amps = np.asarray(result.harmonic_amplitudes, dtype=np.float64)
    tiny = np.finfo(np.float64).tiny
    ref = amps[0] if amps.size and amps[0] > 0.0 else 1.0
    levels_db = 20.0 * np.log10(np.maximum(amps, tiny) / ref)
    orders = np.arange(1, amps.size + 1)

    ax.vlines(orders, -160.0, levels_db, color=_C_PRIMARY, lw=1.5)
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("label", _t("Harmonics", language))
    ax.plot(orders, levels_db, "o", **kwargs)
    for order, level in zip(orders, levels_db):
        if level > -160.0:
            ax.annotate(
                f"{order}",
                (order, level),
                textcoords=_OFFSET_POINTS,
                xytext=(0, 5),
                ha="center",
                fontsize="x-small",
            )
    ax.set_xlabel(_t(r"Harmonic order $n$  ($f = n \cdot f_1$)", language))
    ax.set_ylabel(_t("Level re fundamental [dB]", language))
    ax.set_xticks(orders)
    ax.set_ylim(bottom=-160.0, top=10.0)
    thd_f = decimal_comma(f"{result.thd_f * 100.0:.3g}", language)
    thd_r = decimal_comma(f"{result.thd_r * 100.0:.3g}", language)
    sinad = format_number(result.sinad_db, language, decimals=1)
    freq = decimal_comma(_format_freq(result.fundamental), language)
    ax.set_title(
        f"IEC 60268-3 THD = {thd_f}% (F), "
        f"{thd_r}% (R); SINAD = {sinad} dB "
        f"($f_1$ = {freq}Hz)"
    )
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    localize_axes(ax, language)
    return ax


def plot_modulation_distortion(
    result: ModulationDistortionResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Carrier and modulation sidebands with the per-order distortion annotated.

    Draws the output amplitude at the carrier ``f2`` (the 0 dB reference) and
    the four intermodulation sidebands at ``f2 ± f1`` and ``f2 ± 2f1`` as a
    stem-style spectrum in dB relative to the carrier, annotated with the
    IEC 60268-3 per-order values ``d_m,2``/``d_m,3`` and the SMPTE combined
    RMS — the modulation counterpart of :func:`plot_harmonic_distortion`.

    :param result: A
        :class:`~phonometry.electroacoustics.intermodulation.ModulationDistortionResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the sideband marker ``plot`` call.
    :return: The axes.
    :raises ValueError: If the result carries no sideband spectrum data.
    """
    from .._i18n import decimal_comma, localize_axes

    if (
        result.sideband_frequencies is None
        or result.sideband_amplitudes is None
        or result.carrier_amplitude is None
        or result.f_high is None
    ):
        raise ValueError(
            "this result carries no sideband spectrum data to plot; obtain it "
            "from modulation_distortion()."
        )
    ax = ax if ax is not None else _new_axes()
    tiny = np.finfo(np.float64).tiny
    carrier = float(result.carrier_amplitude)
    sb_freqs = np.asarray(result.sideband_frequencies, dtype=np.float64)
    sb_amps = np.asarray(result.sideband_amplitudes, dtype=np.float64)
    sb_db = 20.0 * np.log10(np.maximum(sb_amps, tiny) / carrier)
    floor = -120.0

    ax.vlines([result.f_high], floor, [0.0], color=_C_SECONDARY, lw=1.8)
    ax.plot(
        [result.f_high],
        [0.0],
        "s",
        color=_C_SECONDARY,
        label=_t("Carrier $f_2$", language),
    )
    ax.vlines(sb_freqs, floor, np.maximum(sb_db, floor), color=_C_PRIMARY, lw=1.5)
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("label", _t(r"Sidebands $f_2 \pm n \cdot f_1$", language))
    ax.plot(sb_freqs, np.maximum(sb_db, floor), "o", **kwargs)
    for order, idx in ((3, 0), (2, 1), (2, 2), (3, 3)):
        if sb_db[idx] > floor:
            ax.annotate(
                f"$n$ = {order}",
                (sb_freqs[idx], max(sb_db[idx], floor)),
                textcoords=_OFFSET_POINTS,
                xytext=(0, 5),
                ha="center",
                fontsize="x-small",
            )
    ax.set_xlabel(_t(_FREQ_LABEL, language))
    ax.set_ylabel(_t("Level re carrier [dB]", language))
    ax.set_ylim(bottom=floor, top=10.0)
    d2 = decimal_comma(f"{result.d2 * 100.0:.3g}", language)
    d3 = decimal_comma(f"{result.d3 * 100.0:.3g}", language)
    smpte = decimal_comma(f"{result.smpte * 100.0:.3g}", language)
    f_low = decimal_comma(_format_freq(float(result.f_low or 0.0)), language)
    f_high = decimal_comma(_format_freq(float(result.f_high)), language)
    ax.set_title(
        f"IEC 60268-3 $d_2$ = {d2}%, $d_3$ = {d3}%; SMPTE = {smpte}% "
        f"($f_1$ = {f_low}Hz, $f_2$ = {f_high}Hz)"
    )
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")
    localize_axes(ax, language)
    return ax


def plot_frequency_response(
    result: FrequencyResponseResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes | np.ndarray:
    """Bode magnitude / phase and coherence of an estimated frequency response.

    Three stacked panels: the magnitude in dB, the phase in degrees and the
    ordinary coherence ``γ²``. With ``ax`` given, only the magnitude panel is
    drawn on it.

    :param result: A
        :class:`~phonometry.electroacoustics.frequency_response.FrequencyResponseResult`.
    :param ax: Existing axes for the magnitude panel, or ``None`` for a fresh
        three-panel figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the magnitude ``plot`` call.
    :return: The magnitude-panel axes (``ax`` given) or the array of three axes.
    """
    from .._i18n import localize_axes

    freqs = np.asarray(result.frequencies, dtype=np.float64)
    mag = np.asarray(result.magnitude_db, dtype=np.float64)
    phase_deg = np.degrees(np.asarray(result.phase, dtype=np.float64))
    coh = np.asarray(result.coherence, dtype=np.float64)
    pos = freqs > 0.0
    color = kwargs.pop("color", _C_PRIMARY)

    def _magnitude(axm: Axes) -> None:
        kwargs.setdefault("label", f"$|H|$ ({result.estimator})")
        axm.semilogx(freqs[pos], mag[pos], color=color, **kwargs)
        axm.set_ylabel(_t(_MAGNITUDE_LABEL, language))
        axm.grid(True, which="both", alpha=0.3)
        axm.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")

    fmin, fmax = float(freqs[pos].min()), float(freqs[pos].max())
    if ax is not None:
        _magnitude(ax)
        ax.set_xlabel(_t(_FREQ_LABEL, language))
        ax.set_title(f"{_t('Frequency response', language)} ({result.estimator})")
        format_frequency_axis(ax, fmin, fmax)
        localize_axes(ax, language)
        return ax

    axes = _new_axes_column(3, sharex=True, figsize=(8.0, 8.0))
    _magnitude(axes[0])
    axes[0].set_title(
        f"{_t('Frequency response', language)} ({result.estimator})"
        f"{_t(' and coherence', language)}"
    )
    axes[1].semilogx(freqs[pos], phase_deg[pos], color=_C_SECONDARY)
    axes[1].set_ylabel(_t("Phase [deg]", language))
    axes[1].grid(True, which="both", alpha=0.3)
    axes[2].semilogx(freqs[pos], coh[pos], color=_C_TERTIARY)
    axes[2].set_ylabel(_t(r"Coherence $\gamma^2$", language))
    axes[2].set_xlabel(_t(_FREQ_LABEL, language))
    axes[2].set_ylim(0.0, 1.05)
    axes[2].grid(True, which="both", alpha=0.3)
    for axf in axes:
        format_frequency_axis(axf, fmin, fmax)
        localize_axes(axf, language)
    return axes


def plot_swept_sine_distortion(
    result: SweptSineDistortionResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes | np.ndarray:
    """Harmonic frequency responses and THD(f) of a swept-sine measurement.

    Two stacked panels: the magnitudes of ``H1..HN`` in dB against their own
    frequency axes (each order drawn over its valid deconvolved band), and
    the total harmonic distortion in percent against the excitation
    frequency. With ``ax`` given, only the THD panel is drawn on it.

    :param result: A
        :class:`~phonometry.electroacoustics.swept_sine.SweptSineDistortionResult`.
    :param ax: Existing axes for the THD panel, or ``None`` for a fresh
        two-panel figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the THD ``plot`` call.
    :return: The THD axes (``ax`` given) or the array of two axes.
    """
    from .._i18n import localize_axes

    freqs = np.asarray(result.frequencies, dtype=np.float64)
    tiny = np.finfo(np.float64).tiny
    nyquist = result.fs / 2.0

    def _thd_panel(axt: Axes) -> None:
        kwargs.setdefault("color", _C_PRIMARY)
        kwargs.setdefault("label", r"$\mathrm{THD}(f)$")
        axt.loglog(
            result.thd_frequencies,
            100.0 * np.maximum(result.thd, tiny),
            **kwargs,
        )
        axt.set_ylabel(_t(_THD_LABEL, language))
        axt.grid(True, which="both", alpha=0.3)
        axt.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")

    if ax is not None:
        _thd_panel(ax)
        ax.set_xlabel(_t(_EXCITATION_FREQ_LABEL, language))
        ax.set_title(_t("Swept-sine THD (Farina / Novak)", language))
        format_frequency_axis(ax)
        localize_axes(ax, language)
        return ax

    axes = _new_axes_column(2, sharex=False, figsize=(8.0, 6.4))
    colors = (_C_PRIMARY, _C_SECONDARY, _C_TERTIARY)
    for k in range(result.n_harmonics):
        order = k + 1
        top = result.f2 if result.method == "farina" else order * result.f2
        band = (freqs >= order * result.f1) & (freqs <= min(top, nyquist))
        if not np.any(band):
            continue
        level = 20.0 * np.log10(
            np.maximum(np.abs(result.harmonic_responses[k][band]), tiny)
        )
        axes[0].semilogx(
            freqs[band],
            level,
            color=colors[k % len(colors)],
            ls="-" if k == 0 else "--",
            lw=1.6 if k == 0 else 1.2,
            label=f"$|H_{{{order}}}(f)|$",
        )
    axes[0].set_ylabel(_t(_MAGNITUDE_LABEL, language))
    axes[0].set_xlabel(_t(_FREQ_LABEL, language))
    axes[0].set_title(
        _t(
            "Harmonic frequency responses ({method} sweep)",
            language,
            method=result.method,
        )
    )
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")
    format_frequency_axis(axes[0])
    localize_axes(axes[0], language)
    _thd_panel(axes[1])
    axes[1].set_xlabel(_t(_EXCITATION_FREQ_LABEL, language))
    format_frequency_axis(axes[1])
    localize_axes(axes[1], language)
    return axes


def plot_piston_impedance(
    result: RadiatingPistonResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Normalized radiation resistance and reactance of a baffled piston.

    Draws the piston resistance ``R1(2ka)`` and reactance ``X1(2ka)`` against
    ``ka`` on a logarithmic ``ka`` axis (the classic Beranek & Mellow figure):
    ``R1`` rising to 1, ``X1`` peaking near ``ka ~ 0.5`` then decaying.

    :param result: A :class:`~phonometry.electroacoustics.piston.RadiatingPistonResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the resistance ``Axes.plot`` (its primary curve).
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    ka = np.asarray(result.ka, dtype=np.float64)
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("label", _t("$R_1$ (resistance)", language))
    ax.semilogx(ka, np.asarray(result.resistance), lw=1.8, **kwargs)
    ax.semilogx(
        ka,
        np.asarray(result.reactance),
        color=_C_SECONDARY,
        lw=1.8,
        ls="--",
        label=_t("$X_1$ (reactance)", language),
    )
    ax.axhline(
        1.0, color=_C_TERTIARY, ls=":", lw=1.0, label=r"$R_1 \to 1$ ($ka \gg 1$)"
    )
    ax.set_xlabel(r"$ka$")
    ax.set_ylabel(
        _t(r"Normalized radiation impedance $Z_\mathrm{r} / \rho c S$", language)
    )
    ax.set_title(_t("Baffled circular piston radiation impedance", language))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    localize_axes(ax, language)
    return ax


def _sign_theta_labels(ax: Any) -> None:
    """Sign the polar angle ticks with the typographic minus U+2212.

    Matplotlib's polar ``ThetaFormatter`` builds its degree labels with a plain
    format spec, so it ignores the ``axes.unicode_minus`` setting and draws a
    negative angle with an ASCII hyphen: a shorter, lower glyph than the minus
    the radial labels beside it carry, inside one figure. Only the glyph
    changes -- the tick positions stay the ones matplotlib's locator chose.
    """
    from matplotlib.ticker import FuncFormatter

    from .._i18n import fmt_minus

    def _degrees(value: float, _pos: Any = None) -> str:
        return f"{fmt_minus(round(float(np.degrees(value))), 'd')}°"

    ax.xaxis.set_major_formatter(FuncFormatter(_degrees))


def plot_piston_directivity(
    result: PistonDirectivity,
    ax: Any | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Far-field directivity (beam) pattern of a baffled circular piston.

    Draws the directivity in dB against the polar angle on a polar axes: one
    curve per ``ka`` in the result, shown as a single family (the directivity
    pattern). The on-axis level is the 0 dB reference at the top; the main lobe
    narrows and the first side lobes appear as ``ka`` grows.

    :param result: A
        :class:`~phonometry.electroacoustics.piston.PistonDirectivity`.
    :param ax: Existing polar axes, or ``None`` to create a polar figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the ``Axes.plot`` call when the result holds a
        single ``ka``; ignored for a multi-``ka`` family so one user color or
        label cannot collapse the per-curve styling.
    :return: The (polar) axes.
    """
    from .._i18n import decimal_comma, localize_axes

    ax = ax if ax is not None else _new_polar_axes()
    theta = np.asarray(result.angles, dtype=np.float64)
    ka = np.asarray(result.ka, dtype=np.float64)
    levels = np.clip(
        np.asarray(result.directivity_db, dtype=np.float64),
        -_POLAR_SPAN_PISTON_DB,
        0.0,
    )
    colors = (_C_PRIMARY, _C_SECONDARY, _C_TERTIARY, _C_REFERENCE, _C_EDGE)
    # Axis (theta = 0) points up; positive angles swing clockwise so the front
    # hemisphere (the baffle blocks the rear) fills the upper half-disk.
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetamin(-90.0)
    ax.set_thetamax(90.0)
    _sign_theta_labels(ax)
    for i, ka_value in enumerate(ka):
        label = f"$ka$ = {decimal_comma(f'{ka_value:.3g}', language)}"
        line_kwargs: dict[str, Any] = {
            "color": colors[i % len(colors)],
            "lw": 1.6,
            "label": label,
        }
        if ka.size == 1:
            # User overrides (color, label, ...) apply only to a single-curve
            # plot; with several ka they would collapse the family styling.
            line_kwargs.update(kwargs)
        ax.plot(theta, levels[i], **line_kwargs)
    ax.set_ylim(-_POLAR_SPAN_PISTON_DB, 0.0)
    ax.set_yticks([-40.0, -30.0, -20.0, -10.0, 0.0])
    ax.set_yticklabels(["−40", "−30", "−20", "−10", "0 dB"], fontsize="x-small")
    ax.tick_params(axis="x", labelsize="x-small")
    ax.grid(True, ls=":", lw=0.4, alpha=0.7)
    ax.set_title(_t("Baffled circular piston directivity", language))
    if ka.size > 1:
        ax.legend(
            loc=_LEGEND_LOWER_CENTER,
            bbox_to_anchor=(0.5, -0.12),
            ncol=min(ka.size, 3),
            fontsize="small",
        )
    localize_axes(ax, language)
    axes: Axes = ax
    return axes


# ---------------------------------------------------------------------------
# IEC 60268-4/-5 rated-characteristics data sheets
#
# The individual panel drawers below are the single source of truth for the
# loudspeaker (IEC 60268-5) and microphone (IEC 60268-4) characteristic
# graphs: the accredited ``.report()`` fiches (``phonometry._report``) and the
# public ``.plot()`` data sheets both compose them, so the two never diverge.
# Each drawer paints one panel onto a supplied axes, sets its own labels, grid,
# legend and (for the continuous-frequency panels) the 1k/2k-style major ticks,
# and leaves figure-level composition to the caller.
# ---------------------------------------------------------------------------


def _fmt_num(value: float, language: str) -> str:
    """A number with up to one decimal (locale separator), trailing ``.0`` dropped."""
    from .._i18n import format_number

    return format_number(value, language, decimals=1, trim=True)


def _freq_label(value: float, language: str) -> str:
    """A frequency rounded to the nearest hertz, locale-formatted."""
    from .._i18n import format_number

    return format_number(round(float(value)), language, decimals=0)


def _grid(ax: Axes) -> None:
    """The dotted both-scale grid the datasheet panels share."""
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)


def _draw_loudspeaker_response(
    result: LoudspeakerCharacteristics, ax: Axes, *, language: str = "en"
) -> None:
    """On-axis SPL with the tolerance band, reference lines and range markers."""
    f = np.asarray(result.frequencies, dtype=np.float64)
    spl = np.asarray(result.spl_db, dtype=np.float64)
    ref = float(result.reference_level_db)
    tol = float(result.tolerance_db)
    top = float(np.ceil((max(float(np.max(spl)), ref + tol) + 2.0) / 5.0) * 5.0)
    ax.axhspan(
        ref - tol,
        ref + tol,
        facecolor=_C_TOL_BAND,
        edgecolor="none",
        zorder=0,
        label=_t(_TOLERANCE_LABEL, language, tol=_fmt_num(tol, language)),
    )
    ax.axhline(ref, color=_C_EDGE, lw=0.8, ls="--")
    ax.axhline(
        ref - 10.0,
        color=_C_REFERENCE,
        lw=0.8,
        ls=":",
        label=_t("−10 dB reference", language),
    )
    ax.semilogx(f, spl, color=_C_PRIMARY, lw=1.4, label=_t(_ON_AXIS_RESPONSE, language))
    lo, hi = result.effective_range
    for edge in (lo, hi):
        ax.axvline(edge, color=_C_TERTIARY, lw=0.9, ls="-.")
    ax.plot(
        [], [], color=_C_TERTIARY, lw=0.9, ls="-.", label=_t(_EFFECTIVE_RANGE, language)
    )
    ax.set_xlim(float(np.min(f)), float(np.max(f)))
    ax.set_ylim(top - _RESPONSE_SPAN_LSP, top)
    ax.set_xlabel(_t(_FREQ_LABEL, language))
    ax.set_ylabel(_t(_SPL_LABEL, language))
    ax.set_title(_t(_ON_AXIS_RESPONSE, language))
    _grid(ax)
    format_frequency_axis(ax, float(np.min(f)), float(np.max(f)))
    ax.legend(loc=_LEGEND_LOWER_CENTER, fontsize="small", ncol=2, framealpha=0.85)


def _draw_impedance(
    result: LoudspeakerCharacteristics, ax: Axes, *, language: str = "en"
) -> None:
    """Impedance modulus with the rated and 80 %-of-rated reference lines (16)."""
    ax.semilogx(
        result.impedance_frequencies, result.impedance_modulus, color=_C_PRIMARY, lw=1.3
    )
    ax.axhline(
        result.rated_impedance,
        color=_C_EDGE,
        lw=0.8,
        ls="--",
        label=_t("Rated impedance", language),
    )
    ax.axhline(
        0.8 * result.rated_impedance,
        color=_C_REFERENCE,
        lw=0.8,
        ls=":",
        label=_t("80 % of rated", language),
    )
    ax.set_xlabel(_t(_FREQ_LABEL, language))
    ax.set_ylabel(_t("Impedance $|Z|$ [{ohm}]", language, ohm=_OHM_TEXT))
    ax.set_ylim(bottom=0.0)
    _grid(ax)
    ax.set_title(_t("Impedance", language))
    ax.legend(loc="upper right", fontsize="small")
    format_frequency_axis(ax)


def _draw_loudspeaker_thd(
    result: LoudspeakerCharacteristics, ax: Axes, *, language: str = "en"
) -> None:
    """Total harmonic distortion against frequency, in percent (24.1)."""
    ax.semilogx(result.thd_frequencies, result.thd_percent, color=_C_PRIMARY, lw=1.3)
    ax.set_xlabel(_t(_FREQ_LABEL, language))
    ax.set_ylabel(_t(_THD_LABEL, language))
    ax.set_ylim(bottom=0.0)
    _grid(ax)
    ax.set_title(_t(_THD_TITLE, language))
    format_frequency_axis(ax)


def _draw_datasheet_polar(
    result: LoudspeakerCharacteristics | MicrophoneCharacteristics,
    ax: Any,
    *,
    language: str = "en",
) -> None:
    """Polar directivity to the IEC 60263 clause 3 25 dB reference-circle scale."""
    angles = np.asarray(result.polar_angles_deg, dtype=np.float64)
    levels = np.clip(
        np.asarray(result.polar_db, dtype=np.float64), -_POLAR_SPAN_DB, 0.0
    )
    # Mirror a one-sided pattern about the reference axis for a full rose.
    if float(np.min(angles)) >= 0.0:
        angles = np.concatenate([-angles[::-1], angles])
        levels = np.concatenate([levels[::-1], levels])
    theta = np.radians(angles)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.plot(theta, levels, color=_C_PRIMARY, lw=1.3)
    ax.set_ylim(-_POLAR_SPAN_DB, 0.0)
    # Reference circle at 0 dB (relative level, IEC 60263 3.3); radial ticks
    # every 5 dB (multiple of 5 for a 25 dB radius, IEC 60263 3.2).
    ax.set_yticks([-20.0, -15.0, -10.0, -5.0, 0.0])
    ax.set_yticklabels(["−20", "", "−10", "", "0 dB"], fontsize="x-small")
    ax.tick_params(axis="x", labelsize="x-small")
    ax.grid(True, ls=":", lw=0.4, alpha=0.7)
    title = _t("Directional response", language)
    if result.polar_frequency is not None:
        title = _t(
            "Directional response at {freq} Hz",
            language,
            freq=_freq_label(result.polar_frequency, language),
        )
    ax.set_title(title)


def _draw_microphone_response(
    result: MicrophoneCharacteristics, ax: Axes, *, language: str = "en"
) -> None:
    """Free-field response with the tolerance band and reference markers (12)."""
    f = np.asarray(result.frequencies, dtype=np.float64)
    rel = np.asarray(result.response_db, dtype=np.float64)
    tol = float(result.tolerance_db)
    top = float(np.ceil((max(float(np.max(rel)), tol) + 2.0) / 5.0) * 5.0)
    ax.axhspan(
        -tol,
        tol,
        facecolor=_C_TOL_BAND,
        edgecolor="none",
        zorder=0,
        label=_t(_TOLERANCE_LABEL, language, tol=_fmt_num(tol, language)),
    )
    ax.axhline(0.0, color=_C_EDGE, lw=0.8, ls="--")
    ax.axvline(
        result.reference_frequency,
        color=_C_SECONDARY,
        lw=0.8,
        ls=":",
        label=_t("Reference frequency", language),
    )
    ax.semilogx(
        f, rel, color=_C_PRIMARY, lw=1.4, label=_t(_FREE_FIELD_RESPONSE, language)
    )
    lo, hi = result.effective_range
    for edge in (lo, hi):
        ax.axvline(edge, color=_C_TERTIARY, lw=0.9, ls="-.")
    ax.plot(
        [], [], color=_C_TERTIARY, lw=0.9, ls="-.", label=_t(_EFFECTIVE_RANGE, language)
    )
    ax.set_xlim(float(np.min(f)), float(np.max(f)))
    ax.set_ylim(top - _RESPONSE_SPAN_MIC, top)
    ax.set_xlabel(_t(_FREQ_LABEL, language))
    ax.set_ylabel(_t("Relative response [dB]", language))
    ax.set_title(_t(_FREE_FIELD_RESPONSE, language))
    _grid(ax)
    format_frequency_axis(ax, float(np.min(f)), float(np.max(f)))
    ax.legend(loc=_LEGEND_LOWER_CENTER, fontsize="small", ncol=2, framealpha=0.85)


def _draw_noise_spectrum(
    result: MicrophoneCharacteristics, ax: Axes, *, language: str = "en"
) -> None:
    """Inherent-noise equivalent band-level spectrum (17.2 b)."""
    ax.semilogx(
        result.noise_frequencies, result.noise_band_levels_db, color=_C_PRIMARY, lw=1.3
    )
    ax.set_xlabel(_t(_FREQ_LABEL, language))
    ax.set_ylabel(_t("Band level [dB]", language))
    _grid(ax)
    ax.set_title(_t("Inherent noise spectrum", language))
    format_frequency_axis(ax)


def _draw_microphone_distortion(
    result: MicrophoneCharacteristics, ax: Axes, *, language: str = "en"
) -> None:
    """Total harmonic distortion against sound pressure level (14.2/15.2)."""
    ax.plot(
        np.asarray(result.distortion_spl_db, dtype=np.float64),
        np.asarray(result.distortion_thd_percent, dtype=np.float64),
        color=_C_PRIMARY,
        lw=1.3,
    )
    ax.axhline(
        result.max_spl_thd_percent,
        color=_C_REFERENCE,
        lw=0.8,
        ls=":",
        label=_t(
            "{thd} % limit",
            language,
            thd=_fmt_num(result.max_spl_thd_percent, language),
        ),
    )
    if result.max_spl_db is not None:
        ax.axvline(
            result.max_spl_db,
            color=_C_EDGE,
            lw=0.8,
            ls="--",
            label=_t("Max. SPL", language),
        )
    ax.set_xlabel(_t(_SPL_LABEL, language))
    ax.set_ylabel(_t(_THD_LABEL, language))
    ax.set_ylim(bottom=0.0)
    _grid(ax)
    ax.set_title(_t(_THD_TITLE, language))
    ax.legend(loc="upper left", fontsize="small")


def _new_polar_axes() -> Any:
    """Create a single fresh polar figure + axes and return the axes."""
    plt = _import_pyplot()
    _fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    return ax


#: The loudspeaker ``.plot()`` quantities: the panel drawer, whether the panel
#: is polar, and the optional-data attribute that must be present (``None`` for
#: the always-available on-axis response).
_LOUDSPEAKER_QUANTITIES: dict[str, tuple[Any, bool, str | None]] = {
    "response": (_draw_loudspeaker_response, False, None),
    "impedance": (_draw_impedance, False, "impedance_frequencies"),
    "thd": (_draw_loudspeaker_thd, False, "thd_frequencies"),
    "directivity": (_draw_datasheet_polar, True, "polar_angles_deg"),
}

#: The microphone ``.plot()`` quantities (see :data:`_LOUDSPEAKER_QUANTITIES`).
_MICROPHONE_QUANTITIES: dict[str, tuple[Any, bool, str | None]] = {
    "response": (_draw_microphone_response, False, None),
    "directivity": (_draw_datasheet_polar, True, "polar_angles_deg"),
    "noise": (_draw_noise_spectrum, False, "noise_frequencies"),
    "distortion": (_draw_microphone_distortion, False, "distortion_spl_db"),
}


def _plot_one_quantity(
    result: Any,
    quantity: str,
    table: dict[str, tuple[Any, bool, str | None]],
    ax: Axes | None,
    language: str,
    missing: dict[str, str],
) -> Axes:
    """Draw exactly one rated-characteristic panel on a single axes.

    Shared by the loudspeaker and microphone ``.plot()`` methods: it validates
    ``quantity`` against ``table``, checks the optional data behind it is
    present, creates a (polar or cartesian) axes when ``ax`` is ``None`` and
    delegates to the shared panel drawer, so the interactive plot and the
    ``.report()`` fiche paint each concept with the same code.
    """
    from .._i18n import localize_axes

    if quantity not in table:
        allowed = ", ".join(repr(q) for q in table)
        raise ValueError(f"unknown quantity {quantity!r}; choose one of {allowed}.")
    drawer, polar, attr = table[quantity]
    if attr is not None and getattr(result, attr) is None:
        raise ValueError(missing[quantity])
    if ax is None:
        ax = _new_polar_axes() if polar else _new_axes()
    drawer(result, ax, language=language)
    localize_axes(ax, language)
    return ax


def plot_loudspeaker_characteristics(
    result: LoudspeakerCharacteristics,
    quantity: str = "response",
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Plot one IEC 60268-5 loudspeaker rated characteristic on a single axes.

    Each ``quantity`` is a single concept drawn by the same shared renderer the
    ``.report()`` fiche composes: ``"response"`` (the on-axis SPL response with
    its tolerance band and effective-range markers, the default),
    ``"impedance"`` (the modulus ``|Z|`` with the rated and 80 %-of-rated
    lines), ``"thd"`` (total harmonic distortion against frequency) and
    ``"directivity"`` (the polar response on the 25 dB reference circle).

    :param result: A
        :class:`~phonometry.electroacoustics.loudspeaker.LoudspeakerCharacteristics`.
    :param quantity: Which characteristic to plot (see above).
    :param ax: Existing axes to draw on, or ``None`` for a fresh figure (a polar
        axes is created for ``"directivity"``).
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Reserved for a uniform ``.plot()`` signature; unused.
    :return: The axes the characteristic was drawn on.
    :raises ValueError: If ``quantity`` is unknown or the result carries no data
        for it.
    """
    del kwargs  # each panel is a fixed datasheet artist; no primary override
    missing = {
        "impedance": "this result carries no impedance curve to plot.",
        "thd": "this result carries no total-harmonic-distortion curve to plot.",
        "directivity": "this result carries no directional response to plot.",
    }
    return _plot_one_quantity(
        result, quantity, _LOUDSPEAKER_QUANTITIES, ax, language, missing
    )


def plot_microphone_characteristics(
    result: MicrophoneCharacteristics,
    quantity: str = "response",
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Plot one IEC 60268-4 microphone rated characteristic on a single axes.

    Each ``quantity`` is a single concept drawn by the same shared renderer the
    ``.report()`` fiche composes: ``"response"`` (the free-field response with
    its tolerance band, reference-frequency and effective-range markers, the
    default), ``"directivity"`` (the polar directional pattern on the 25 dB
    reference circle), ``"noise"`` (the inherent-noise band-level spectrum) and
    ``"distortion"`` (total harmonic distortion against sound pressure level).

    :param result: A
        :class:`~phonometry.electroacoustics.microphone.MicrophoneCharacteristics`.
    :param quantity: Which characteristic to plot (see above).
    :param ax: Existing axes to draw on, or ``None`` for a fresh figure (a polar
        axes is created for ``"directivity"``).
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Reserved for a uniform ``.plot()`` signature; unused.
    :return: The axes the characteristic was drawn on.
    :raises ValueError: If ``quantity`` is unknown or the result carries no data
        for it.
    """
    del kwargs
    missing = {
        "directivity": "this result carries no directional pattern to plot.",
        "noise": "this result carries no inherent-noise spectrum to plot.",
        "distortion": "this result carries no distortion-against-level curve to plot.",
    }
    return _plot_one_quantity(
        result, quantity, _MICROPHONE_QUANTITIES, ax, language, missing
    )


def plot_feedback_stability(
    result: FeedbackStabilityResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Gain structure of a reinforcement loop against its stability limits.

    Bars for the open-loop gain ``Z_S``, the feedback-loop gain ``G_S`` and the
    open-microphone correction accumulate into the total loop gain (Long
    Equation (18.24)); the 0 dB oscillation threshold of Equation (18.16) and
    the required stability margin are drawn as reference lines.

    :param result: A
        :class:`~phonometry.electroacoustics.sound_reinforcement.FeedbackStabilityResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the component ``Axes.bar`` call.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    labels = [
        _t("Open-loop gain $Z_\\mathrm{S}$", language),
        _t("Feedback-loop gain $G_\\mathrm{S}$", language),
        _t(r"Open microphones $\Delta L_{\mathrm{nom}}$", language),
        _t("Total loop gain", language),
    ]
    values = np.array(
        [
            result.open_loop_gain,
            result.feedback_loop_gain,
            result.nom_correction,
            result.loop_gain,
        ],
        dtype=np.float64,
    )
    colors = [
        _C_PRIMARY,
        _C_SECONDARY,
        _C_TERTIARY,
        _C_REFERENCE if not result.is_stable else _C_EDGE,
    ]
    kwargs.setdefault("width", 0.62)
    ax.bar(
        np.arange(values.size),
        values,
        color=colors,
        edgecolor=_C_EDGE,
        linewidth=0.6,
        zorder=3,
        **kwargs,
    )
    for i, value in enumerate(values):
        ax.annotate(
            format_number(value, language, decimals=1) + " dB",
            xy=(i, value),
            xytext=(0, 4 if value >= 0.0 else -12),
            textcoords=_OFFSET_POINTS,
            ha="center",
            fontsize=8,
        )
    ax.axhline(
        0.0,
        color=_C_REFERENCE,
        ls="-",
        lw=1.2,
        zorder=2,
        label=_t("Oscillation (0 dB)", language),
    )
    ax.axhline(
        -result.stability_margin,
        color=_C_TERTIARY,
        ls="--",
        lw=1.3,
        zorder=2,
        label=_t(
            "Stability limit ({margin} dB margin)",
            language,
            margin=format_number(result.stability_margin, language, decimals=0),
        ),
    )
    ax.set_xticks(np.arange(values.size))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(_t("Gain [dB]", language))
    head = format_number(abs(result.headroom), language, decimals=1)
    ax.set_title(
        _t("Gain before feedback — stable, {head} dB spare", language, head=head)
        if result.is_stable
        else _t("Gain before feedback — unstable by {head} dB", language, head=head)
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")
    localize_axes(ax, language)
    return ax
