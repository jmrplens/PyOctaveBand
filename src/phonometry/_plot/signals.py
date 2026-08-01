#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Plot renderers for the signal domain (lazy imports from result .plot())."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..signals.cepstrum import (
        CepstrumResult,
        EchoDetectionResult,
        LifterResult,
    )
    from ..signals.correlation import (
        AlignedImpulseResponseResult,
        CorrelationResult,
        TimeDelayResult,
    )
    from ..signals.envelope import EnvelopeResult, EnvelopeSpectrumResult
    from ..signals.miso import MISOCoherenceResult
    from ..signals.phase import PhaseDecompositionResult
    from ..signals.spectra import (
        CoherentOutputSpectrumResult,
        CrossSpectralDensityResult,
        MultitaperSpectralDensityResult,
        SpectralDensityResult,
        WindowMetricsResult,
    )
    from ..signals.synchronous_average import SynchronousAverageResult
    from ..signals.test_signals import ResampledSignalResult, ToneBurstResult
    from ..signals.time_frequency import SpectrogramResult, ZoomFFTResult

from .common import (
    _C_EDGE,
    _C_MUTED,
    _C_PRIMARY,
    _C_PRIMARY_LIGHT,
    _C_REFERENCE,
    _C_SECONDARY,
    _C_TERTIARY,
    _LEGEND_UPPER_RIGHT,
    _new_axes,
    _new_axes_column,
    format_frequency_axis,
)

#: Spanish translations of the fixed strings rendered by the signal
#: ``.plot()`` renderers, keyed by their verbatim English text. ``_t``
#: returns the English key unchanged for any language other than ``"es"``,
#: so the English output is byte-for-byte identical to the pre-i18n
#: renderers.
#: Axis labels and templates the renderers repeat; the Spanish table is keyed
#: by the same constants, so a label is written once.
_FREQ_LABEL = "Frequency [Hz]"
_TIME_LABEL = "Time [s]"
_LAG_LABEL = "Lag [s]"
_MAGNITUDE_LABEL = "Magnitude [dB]"
_QUEFRENCY_LABEL = "Quefrency [ms]"
_CORRELATION_COEFFICIENT_LABEL = "Correlation coefficient"
_ENBW_LABEL = "ENBW {enbw} bins"

_STRINGS: dict[str, str] = {
    _FREQ_LABEL: "Frecuencia [Hz]",
    _LAG_LABEL: "Retardo [s]",
    _TIME_LABEL: "Tiempo [s]",
    "Amplitude": "Amplitud",
    _MAGNITUDE_LABEL: "Magnitud [dB]",
    "Phase [deg]": "Fase [grados]",
    "Phase [rad]": "Fase [rad]",
    "Spectral density [dB re 1/Hz]": "Densidad espectral [dB re 1/Hz]",
    "Power spectrum [dB]": "Espectro de potencia [dB]",
    r"{pct} % confidence ($\chi^2$, $n_d$ = {nd})":
        r"{pct} % de confianza ($\chi^2$, $n_d$ = {nd})",
    r"Welch spectral density — $\varepsilon_r$ = {er} %":
        r"Densidad espectral de Welch — $\varepsilon_r$ = {er} %",
    "Cross-spectral density (Bendat & Piersol)":
        "Densidad espectral cruzada (Bendat y Piersol)",
    r"$\pm$ s.d.$[\hat{\theta}_{xy}]$ (Eq. 9.52)":
        r"$\pm$ d.e.$[\hat{\theta}_{xy}]$ (Ec. 9.52)",
    r"$\hat{G}_{yy}$ (output)": r"$\hat{G}_{yy}$ (salida)",
    r"$\hat{G}_{nn}$ (noise)": r"$\hat{G}_{nn}$ (ruido)",
    "Coherent output spectrum (Bendat & Piersol 9.2.2)":
        "Espectro de salida coherente (Bendat y Piersol 9.2.2)",
    "Spectral SNR [dB]": "SNR espectral [dB]",
    r"$\hat{G}_{yy}$ (measured output)": r"$\hat{G}_{yy}$ (salida medida)",
    r"$\hat{G}_{nn}$ (residual noise)": r"$\hat{G}_{nn}$ (ruido residual)",
    "Input {i}": "Entrada {i}",
    "Partial coherent output spectra (Bendat & Piersol 7.3)":
        "Espectros de salida coherente parciales (Bendat y Piersol 7.3)",
    "Coherence": "Coherencia",
    r"$\gamma^2_{y:x}$ (multiple)": r"$\gamma^2_{y:x}$ (múltiple)",
    "Input {i} (partial)": "Entrada {i} (parcial)",
    "Multiple and partial coherence": "Coherencia múltiple y parcial",
    "Calibrated spectrogram (Bendat & Piersol 12.6.4.2)":
        "Espectrograma calibrado (Bendat y Piersol 12.6.4.2)",
    "Zoom FFT (Bendat & Piersol 11.5.4)":
        "FFT con zoom (Bendat y Piersol 11.5.4)",
    _CORRELATION_COEFFICIENT_LABEL: "Coeficiente de correlación",
    "Correlation ({norm})": "Correlación ({norm})",
    "{kind} estimate (Bendat & Piersol)":
        "Estimación de {kind} (Bendat y Piersol)",
    "Autocorrelation": "autocorrelación",
    "Cross-correlation": "correlación cruzada",
    r"$\hat{R}_{xy}(\tau)$ (context)": r"$\hat{R}_{xy}(\tau)$ (contexto)",
    "95 % interval (Eq. 8.130)": "Intervalo 95 % (Ec. 8.130)",
    "Normalized correlation": "Correlación normalizada",
    "Time-delay estimate — {method}":
        "Estimación del retardo temporal — {method}",
    "Reference IR": "RI de referencia",
    "Aligned IR (delay {n} samples)": "RI alineada (retardo {n} muestras)",
    "Impulse-response alignment (sub-sample)":
        "Alineación de la respuesta al impulso (submuestra)",
    "Signal": "Señal",
    "Envelope $A(t)$ (Eq. 13.17)": "Envolvente $A(t)$ (Ec. 13.17)",
    "Hilbert envelope (Bendat & Piersol Ch. 13)":
        "Envolvente de Hilbert (Bendat y Piersol Cap. 13)",
    "Instantaneous frequency [Hz]": "Frecuencia instantánea [Hz]",
    _QUEFRENCY_LABEL: "Quefrencia [ms]",
    "Cepstrum": "Cepstro",
    "Power cepstrum": "Cepstro de potencia",
    "Real cepstrum": "Cepstro real",
    "Complex cepstrum": "Cepstro complejo",
    "Lifter cutoff ({q} ms)": "Corte del lifter ({q} ms)",
    "Log spectrum": "Espectro logarítmico",
    "Liftered ({mode})": "Lifterado ({mode})",
    "lowpass": "paso bajo",
    "highpass": "paso alto",
    "Liftering at {q} ms ({mode})": "Liftering a {q} ms ({mode})",
    "Searched band": "Banda de búsqueda",
    "Echo: {delay} ms, a = {a}": "Eco: {delay} ms, a = {a}",
    "Echo detection on the power cepstrum":
        "Detección de ecos en el cepstro de potencia",
    "Envelope ({kind})": "Envolvente ({kind})",
    "magnitude": "magnitud",
    "squared": "cuadrática",
    "Mean level": "Nivel medio",
    "Modulation amplitude": "Amplitud de modulación",
    "Envelope spectrum (Bendat & Piersol 13.3)":
        "Espectro de la envolvente (Bendat y Piersol 13.3)",
    "Measured phase": "Fase medida",
    "Minimum phase (from |H|)": "Fase mínima (de |H|)",
    "Excess phase (all-pass)": "Fase de exceso (pasa-todo)",
    "Phase decomposition": "Descomposición de fase",
    "Minimum-phase / all-pass decomposition":
        "Descomposición fase mínima / pasa-todo",
    "Group delay": "Retardo de grupo",
    "Excess group delay": "Retardo de grupo de exceso",
    "Group delay [ms]": "Retardo de grupo [ms]",
    "biased": "sesgada",
    "unbiased": "insesgada",
    "Gating envelope": "Envolvente de conmutación",
    "Tone burst (IEC 60268-1): {f} Hz, {cycles} cycles":
        "Salva de tono (IEC 60268-1): {f} Hz, {cycles} ciclos",
    "Tone burst (IEC 60268-1): {f} Hz, {cycles} cycles, {rate}/s":
        "Salva de tono (IEC 60268-1): {f} Hz, {cycles} ciclos, {rate}/s",
    "Window w[m]": "Ventana w[m]",
    "Sample": "Muestra",
    "Frequency offset [DFT bins]":
        "Desplazamiento en frecuencia [bins de la DFT]",
    "Level re main lobe [dB]": "Nivel re lóbulo principal [dB]",
    _ENBW_LABEL: _ENBW_LABEL,
    "Highest sidelobe {sll} dB": "Lóbulo lateral máximo {sll} dB",
    "Scalloping loss {sl} dB": "Pérdida de festoneado {sl} dB",
    "Window metrics (Harris 1978): {window}":
        "Métricas de la ventana (Harris 1978): {window}",
    r"{pct} % confidence ($\chi^2$, $\bar\nu$ = {nu})":
        r"{pct} % de confianza ($\chi^2$, $\bar\nu$ = {nu})",
    "Thomson multitaper density — $K$ = {k} tapers, $NW$ = {nw}":
        "Densidad multitaper de Thomson — $K$ = {k} tapers, $NW$ = {nw}",
    "Measured response $|H|$": "Respuesta medida $|H|$",
    r"Inverse filter $|H_{\mathrm{inv}}|$":
        r"Filtro inverso $|H_{\mathrm{inv}}|$",
    r"Equalized $|H \cdot H_{\mathrm{inv}}|$":
        r"Ecualizado $|H \cdot H_{\mathrm{inv}}|$",
    "Equalized band": "Banda ecualizada",
    "Regularized inversion (Kirkeby) — flatness {flat} dB":
        "Inversión regularizada (Kirkeby) — planitud {flat} dB",
    "Time synchronous average (McFadden 1987)":
        "Promediado síncrono en el tiempo (McFadden 1987)",
    "Averaged periodic waveform (N = {n})":
        "Forma de onda periódica promediada (N = {n})",
    "Time [ms]": "Tiempo [ms]",
    "Frequency [orders]": "Frecuencia [órdenes]",
    "Comb filter $|C(f)|$ (Eq. 8)": "Filtro peine $|C(f)|$ (Ec. 8)",
    "Harmonics of $1/T$": "Armónicos de $1/T$",
    "Anti-alias filter $|H(f)|$": "Filtro antisolapamiento $|H(f)|$",
    "Passband edge": "Borde de la banda de paso",
    "Stopband edge (alias fold)":
        "Borde de la banda atenuada (pliegue de alias)",
    "Design attenuation −{a} dB": "Atenuación de diseño −{a} dB",
    "Rejected band (would fold back as aliases)":
        "Banda rechazada (se plegaría como alias)",
    "Polyphase resampling {fs0} Hz → {fs1} Hz (L/M = {up}/{down}, {taps} taps)":
        "Remuestreo polifásico {fs0} Hz → {fs1} Hz (L/M = {up}/{down}, {taps} coeficientes)",
}


def _t(text: str, language: str = "en", **fmt: Any) -> str:
    """Localise a fixed string; English is returned verbatim (byte-identical)."""
    s = _STRINGS.get(text, text) if language == "es" else text
    return s.format(**fmt) if fmt else s


def _db10(values: np.ndarray) -> np.ndarray:
    """``10·log10`` with -inf (not a warning) at empty bins."""
    with np.errstate(divide="ignore"):
        out: np.ndarray = 10.0 * np.log10(values)
    return out


def _finite_db_floor(curves: list[np.ndarray], *, margin: float = 5.0) -> float:
    """A fixed lower dB bound from the finite values of several curves.

    Returns the smallest finite level across ``curves`` minus ``margin`` (or
    a fallback when every value is non-finite), giving fills and y-limits a
    baseline that does not depend on autoscale or per-iteration axis state.
    """
    stacked = np.concatenate([np.asarray(c, dtype=np.float64) for c in curves])
    finite = stacked[np.isfinite(stacked)]
    return float(finite.min()) - margin if finite.size else -100.0


def _psd_ylabel(scaling: str, language: str = "en") -> str:
    return (
        _t("Spectral density [dB re 1/Hz]", language)
        if scaling == "density"
        else _t("Power spectrum [dB]", language)
    )


def _plot_density_with_band(
    result: SpectralDensityResult | MultitaperSpectralDensityResult,
    ax: Axes | None,
    language: str,
    kwargs: dict[str, Any],
    *,
    band_color: Any,
    band_alpha: float | None,
    band_label: str,
    line_label: str,
    title: str,
) -> Axes:
    """Shared renderer: density line in dB over its confidence band."""
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    pos = freqs > 0.0
    color = kwargs.pop("color", _C_PRIMARY)
    ax.fill_between(
        freqs[pos],
        _db10(np.asarray(result.ci_lower, dtype=np.float64)[pos]),
        _db10(np.asarray(result.ci_upper, dtype=np.float64)[pos]),
        color=band_color if band_color is not None else color,
        alpha=band_alpha,
        lw=0.0,
        label=band_label,
    )
    kwargs.setdefault("label", line_label)
    ax.semilogx(freqs[pos], _db10(np.asarray(result.psd)[pos]), color=color, **kwargs)
    ax.set_xlabel(_t(_FREQ_LABEL, language))
    ax.set_ylabel(_psd_ylabel(result.scaling, language))
    ax.set_title(title)
    ax.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")
    ax.grid(True, which="both", alpha=0.3)
    format_frequency_axis(ax, float(freqs[pos].min()), float(freqs[pos].max()))
    localize_axes(ax, language)
    return ax
def plot_spectral_density(
    result: SpectralDensityResult, ax: Axes | None = None, *,
    language: str = "en", **kwargs: Any
) -> Axes:
    """Spectral density in dB with its chi-square confidence band.

    :param result: A :class:`~phonometry.signals.spectra.SpectralDensityResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the density ``plot`` call.
    :return: The axes.
    """
    from .._i18n import decimal_comma, format_number

    pct = decimal_comma(f"{100.0 * result.confidence:g}", language)
    nd = format_number(result.n_averages, language, decimals=1)
    er = format_number(100.0 * result.random_error, language, decimals=1)
    return _plot_density_with_band(
        result,
        ax,
        language,
        kwargs,
        band_color=None,  # the band shares the line color, translucent
        band_alpha=0.25,
        band_label=_t(r"{pct} % confidence ($\chi^2$, $n_d$ = {nd})", language, pct=pct, nd=nd),
        line_label="$\\hat{G}_{xx}(f)$",
        title=_t(r"Welch spectral density — $\varepsilon_r$ = {er} %", language, er=er),
    )


def plot_multitaper_spectral_density(
    result: MultitaperSpectralDensityResult, ax: Axes | None = None, *,
    language: str = "en", **kwargs: Any
) -> Axes:
    """Multitaper spectral density in dB with its chi-square band.

    The confidence band uses the per-frequency degrees of freedom of the
    (possibly adaptive) estimator and is drawn as a pale opaque fill.

    :param result: A
        :class:`~phonometry.signals.spectra.MultitaperSpectralDensityResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the density ``plot`` call.
    :return: The axes.
    """
    from .._i18n import decimal_comma, format_number

    pct = decimal_comma(f"{100.0 * result.confidence:g}", language)
    # Average the per-bin dof over the interior bins; for a very short signal
    # the array can have <=2 bins, where trimming the DC/Nyquist edges would
    # leave an empty slice, so fall back to the full array there.
    dof = result.degrees_of_freedom
    interior = dof[1:-1] if dof.size > 2 else dof
    nu_mean = float(np.mean(interior))
    nu = format_number(nu_mean, language, decimals=1)
    nw = decimal_comma(f"{result.time_half_bandwidth:g}", language)
    return _plot_density_with_band(
        result,
        ax,
        language,
        kwargs,
        band_color=_C_PRIMARY_LIGHT,
        band_alpha=None,  # pale opaque fill
        band_label=_t(r"{pct} % confidence ($\chi^2$, $\bar\nu$ = {nu})",
                      language, pct=pct, nu=nu),
        line_label="$\\hat{S}^{(mt)}(f)$",
        title=_t(
            r"Thomson multitaper density — $K$ = {k} tapers, $NW$ = {nw}",
            language, k=result.n_tapers, nw=nw,
        ),
    )


def plot_cross_spectral_density(
    result: CrossSpectralDensityResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes | np.ndarray:
    """Cross-spectrum magnitude, phase (with ±σ band) and coherence.

    With ``ax`` given, only the magnitude panel is drawn on it.

    :param result: A
        :class:`~phonometry.signals.spectra.CrossSpectralDensityResult`.
    :param ax: Existing axes for the magnitude panel, or ``None`` for a
        fresh three-panel figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the magnitude ``plot`` call.
    :return: The magnitude axes (``ax`` given) or the array of three axes.
    """
    from .._i18n import localize_axes

    freqs = np.asarray(result.frequencies, dtype=np.float64)
    pos = freqs > 0.0
    color = kwargs.pop("color", _C_PRIMARY)

    def _magnitude(axm: Axes) -> None:
        kwargs.setdefault("label", "$|\\hat{G}_{xy}(f)|$")
        axm.semilogx(
            freqs[pos], _db10(result.magnitude[pos]), color=color, **kwargs
        )
        axm.set_ylabel(_psd_ylabel(result.scaling, language))
        axm.grid(True, which="both", alpha=0.3)
        axm.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")

    fmin, fmax = float(freqs[pos].min()), float(freqs[pos].max())
    if ax is not None:
        _magnitude(ax)
        ax.set_xlabel(_t(_FREQ_LABEL, language))
        format_frequency_axis(ax, fmin, fmax)
        localize_axes(ax, language)
        return ax

    axes = _new_axes_column(3, sharex=True, figsize=(8.0, 7.0))
    _magnitude(axes[0])
    axes[0].set_title(_t("Cross-spectral density (Bendat & Piersol)", language))
    phase = np.degrees(result.phase[pos])
    # Cap the drawn band at +/-180 deg: at near-zero coherence the Eq. 9.52
    # s.d. diverges and a literal band would blow the panel's autoscale.
    sigma = np.minimum(np.degrees(result.phase_std[pos]), 180.0)
    finite = np.isfinite(sigma)
    axes[1].fill_between(
        freqs[pos][finite],
        (phase - sigma)[finite],
        (phase + sigma)[finite],
        color=_C_SECONDARY,
        alpha=0.3,
        lw=0.0,
        label=_t(r"$\pm$ s.d.$[\hat{\theta}_{xy}]$ (Eq. 9.52)", language),
    )
    axes[1].semilogx(freqs[pos], phase, color=color)
    axes[1].set_ylabel(_t("Phase [deg]", language))
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")
    axes[2].semilogx(freqs[pos], result.coherence[pos], color=_C_MUTED)
    axes[2].set_ylabel("$\\gamma^2_{xy}$")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_xlabel(_t(_FREQ_LABEL, language))
    axes[2].grid(True, which="both", alpha=0.3)
    for axf in axes:
        format_frequency_axis(axf, fmin, fmax)
        localize_axes(axf, language)
    return axes


def plot_coherent_output_spectrum(
    result: CoherentOutputSpectrumResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes | np.ndarray:
    """Output, coherent and noise spectra plus the spectral SNR.

    With ``ax`` given, only the spectra panel is drawn on it.

    :param result: A
        :class:`~phonometry.signals.spectra.CoherentOutputSpectrumResult`.
    :param ax: Existing axes for the spectra panel, or ``None`` for a fresh
        two-panel figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the coherent-spectrum ``plot`` call.
    :return: The spectra axes (``ax`` given) or the array of two axes.
    """
    from .._i18n import localize_axes

    freqs = np.asarray(result.frequencies, dtype=np.float64)
    pos = freqs > 0.0
    color = kwargs.pop("color", _C_PRIMARY)

    def _spectra_panel(axs: Axes) -> None:
        axs.semilogx(
            freqs[pos],
            _db10(result.output_psd[pos]),
            color=_C_MUTED,
            label=_t(r"$\hat{G}_{yy}$ (output)", language),
        )
        kwargs.setdefault(
            "label", "$\\hat{G}_{vv} = \\hat{\\gamma}^2_{xy}\\hat{G}_{yy}$"
        )
        axs.semilogx(freqs[pos], _db10(result.coherent_psd[pos]), color=color,
                     **kwargs)
        axs.semilogx(
            freqs[pos],
            _db10(result.noise_psd[pos]),
            color=_C_REFERENCE,
            ls="--",
            label=_t(r"$\hat{G}_{nn}$ (noise)", language),
        )
        axs.set_ylabel(_psd_ylabel(result.scaling, language))
        axs.grid(True, which="both", alpha=0.3)
        axs.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")

    fmin, fmax = float(freqs[pos].min()), float(freqs[pos].max())
    if ax is not None:
        _spectra_panel(ax)
        ax.set_xlabel(_t(_FREQ_LABEL, language))
        format_frequency_axis(ax, fmin, fmax)
        localize_axes(ax, language)
        return ax

    axes = _new_axes_column(2, sharex=True, figsize=(8.0, 5.6))
    _spectra_panel(axes[0])
    axes[0].set_title(_t("Coherent output spectrum (Bendat & Piersol 9.2.2)", language))
    axes[1].semilogx(freqs[pos], result.snr_db[pos], color=_C_SECONDARY)
    axes[1].axhline(0.0, color=_C_MUTED, ls=":", lw=1.0)
    axes[1].set_ylabel(_t("Spectral SNR [dB]", language))
    axes[1].set_xlabel(_t(_FREQ_LABEL, language))
    axes[1].grid(True, which="both", alpha=0.3)
    for axf in axes:
        format_frequency_axis(axf, fmin, fmax)
        localize_axes(axf, language)
    return axes


#: Per-input artist colors for the MISO coherence figure (up to three inputs).
_MISO_COLORS = (_C_PRIMARY, _C_SECONDARY, _C_TERTIARY)


def _miso_spectra_panel(
    axs: Axes, result: MISOCoherenceResult, freqs: np.ndarray,
    pos: np.ndarray, language: str,
) -> None:
    """Draw the per-input coherent output spectra with pale opaque fills."""
    output_db = _db10(result.output_psd[pos])
    noise_db = _db10(result.noise_psd[pos])
    coherent_db = [
        _db10(result.coherent_output_spectra[i][pos])
        for i in range(result.n_inputs)
    ]
    # A single fixed baseline for every fill, derived once from the finite
    # dynamic range of the panel. A coherent output that dips to zero gives
    # -inf in dB; clipping to this floor keeps the fills and the y-limits
    # deterministic instead of letting one empty bin drag the axis.
    floor = _finite_db_floor([output_db, noise_db, *coherent_db], margin=5.0)
    axs.semilogx(freqs[pos], output_db, color=_C_MUTED, lw=1.4,
                 label=_t(r"$\hat{G}_{yy}$ (measured output)", language))
    for i in range(result.n_inputs):
        color = _MISO_COLORS[i % len(_MISO_COLORS)]
        level = np.clip(coherent_db[i], floor, None)
        axs.fill_between(freqs[pos], floor, level, color=color, alpha=0.12,
                         lw=0.0)
        axs.semilogx(freqs[pos], level, color=color, lw=1.2,
                     label=_t("Input {i}", language, i=i + 1))
    axs.semilogx(freqs[pos], np.clip(noise_db, floor, None),
                 color=_C_REFERENCE, ls="--", lw=1.0,
                 label=_t(r"$\hat{G}_{nn}$ (residual noise)", language))
    finite_top = output_db[np.isfinite(output_db)]
    top = float(np.max(finite_top)) if finite_top.size else floor + 1.0
    axs.set_ylim(floor, top + 3.0)
    axs.set_ylabel(_psd_ylabel(result.scaling, language))
    axs.grid(True, which="both", alpha=0.3)
    axs.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small", ncol=2)


def _miso_coherence_panel(
    axc: Axes, result: MISOCoherenceResult, freqs: np.ndarray,
    pos: np.ndarray, language: str,
) -> None:
    """Draw the multiple coherence over the per-input partial coherences."""
    for i in range(result.n_inputs):
        color = _MISO_COLORS[i % len(_MISO_COLORS)]
        axc.semilogx(freqs[pos], result.partial_coherence[i][pos], color=color,
                     lw=1.0, alpha=0.85,
                     label=_t("Input {i} (partial)", language, i=i + 1))
    axc.semilogx(freqs[pos], result.multiple_coherence[pos], color=_C_EDGE,
                 lw=1.8, label=_t(r"$\gamma^2_{y:x}$ (multiple)", language))
    axc.set_ylabel(_t("Coherence", language))
    axc.set_ylim(0.0, 1.05)
    axc.grid(True, which="both", alpha=0.3)
    axc.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small", ncol=2)


def plot_miso_coherence(
    result: MISOCoherenceResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes | np.ndarray:
    """Per-input coherent output spectra and the multiple/partial coherences.

    The upper panel decomposes the measured output autospectrum into the
    part each input contributes (a pale opaque fill under each line), the
    lower panel shows the multiple coherence over the per-input partial
    coherences. With ``ax`` given, only the spectra panel is drawn on it.

    :param result: A :class:`~phonometry.signals.miso.MISOCoherenceResult`.
    :param ax: Existing axes for the spectra panel, or ``None`` for a fresh
        two-panel figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Ignored (kept for signature parity with the other
        renderers).
    :return: The spectra axes (``ax`` given) or the array of two axes.
    """
    from .._i18n import localize_axes

    freqs = np.asarray(result.frequencies, dtype=np.float64)
    pos = freqs > 0.0
    fmin, fmax = float(freqs[pos].min()), float(freqs[pos].max())

    if ax is not None:
        _miso_spectra_panel(ax, result, freqs, pos, language)
        ax.set_xlabel(_t(_FREQ_LABEL, language))
        format_frequency_axis(ax, fmin, fmax)
        localize_axes(ax, language)
        return ax

    axes = _new_axes_column(2, sharex=True, figsize=(8.0, 6.4))
    _miso_spectra_panel(axes[0], result, freqs, pos, language)
    axes[0].set_title(
        _t("Partial coherent output spectra (Bendat & Piersol 7.3)", language)
    )
    _miso_coherence_panel(axes[1], result, freqs, pos, language)
    axes[1].set_xlabel(_t(_FREQ_LABEL, language))
    for axf in axes:
        format_frequency_axis(axf, fmin, fmax)
        localize_axes(axf, language)
    return axes


def plot_spectrogram(
    result: SpectrogramResult, ax: Axes | None = None, *,
    language: str = "en", **kwargs: Any
) -> Axes:
    """Spectrogram in dB over the time-frequency plane.

    The display is drawn as a single raster image (``imshow``): per-cell
    vector quads are avoided so the figure stays light and free of moire
    (the repo's pcolormesh-in-SVG policy). The default color range spans
    the 80 dB below the strongest cell; pass ``vmin``/``vmax`` to change
    it.

    :param result: A
        :class:`~phonometry.signals.time_frequency.SpectrogramResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to ``imshow``.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    times = np.asarray(result.times, dtype=np.float64)
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    level = _db10(np.asarray(result.power, dtype=np.float64))
    vmax = float(np.max(level[np.isfinite(level)]))
    fs = result.nperseg / result.time_resolution
    half_hop = 0.5 * result.hop / fs
    df = float(freqs[1] - freqs[0])
    img = ax.imshow(
        level,
        **{
            "cmap": "magma",
            "vmin": vmax - 80.0,
            "vmax": vmax,
            "aspect": "auto",
            "origin": "lower",
            "interpolation": "nearest",
            "extent": (
                float(times[0]) - half_hop,
                float(times[-1]) + half_hop,
                max(float(freqs[0]) - 0.5 * df, 0.0),
                float(freqs[-1]) + 0.5 * df,
            ),
            **kwargs,
        },
    )
    ax.figure.colorbar(img, ax=ax, label=_psd_ylabel(result.scaling, language))
    ax.set_xlabel(_t(_TIME_LABEL, language))
    ax.set_ylabel(_t(_FREQ_LABEL, language))
    ax.set_title(_t("Calibrated spectrogram (Bendat & Piersol 12.6.4.2)", language))
    localize_axes(ax, language)
    return ax


def plot_zoom_fft(
    result: ZoomFFTResult, ax: Axes | None = None, *,
    language: str = "en", **kwargs: Any
) -> Axes:
    """Zoom power spectrum in dB over the zoom band (linear axis).

    :param result: A
        :class:`~phonometry.signals.time_frequency.ZoomFFTResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the ``plot`` call.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    color = kwargs.pop("color", _C_PRIMARY)
    be = format_number(result.resolution_bandwidth, language, decimals=2)
    df = format_number(result.bin_spacing, language, decimals=3)
    kwargs.setdefault(
        "label", f"$B_e$ = {be} Hz, $\\Delta f$ = {df} Hz"
    )
    ax.plot(freqs, _db10(np.asarray(result.power, dtype=np.float64)),
            color=color, **kwargs)
    ax.set_xlim(float(freqs[0]), float(freqs[-1]))
    ax.set_xlabel(_t(_FREQ_LABEL, language))
    ax.set_ylabel(_psd_ylabel("spectrum", language))
    ax.set_title(_t("Zoom FFT (Bendat & Piersol 11.5.4)", language))
    ax.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")
    ax.grid(True, alpha=0.3)
    localize_axes(ax, language)
    return ax
_LAG_LABEL = "Lag [s]"
_TIME_LABEL = "Time [s]"
def plot_correlation(
    result: CorrelationResult, ax: Axes | None = None, *,
    language: str = "en", **kwargs: Any
) -> Axes:
    """Correlation estimate against the lag in seconds.

    :param result: A :class:`~phonometry.signals.correlation.CorrelationResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the ``plot`` call.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    symbol = (
        "\\hat{\\rho}" if result.normalization == "coefficient" else "\\hat{R}"
    )
    sub = "xx" if result.kind == "autocorrelation" else "xy"
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("label", f"${symbol}_{{{sub}}}(\\tau)$")
    ax.plot(result.lags, result.values, **kwargs)
    ax.axvline(0.0, color=_C_MUTED, ls=":", lw=1.0)
    ax.set_xlabel(_t(_LAG_LABEL, language))
    if result.normalization == "coefficient":
        ax.set_ylabel(_t(_CORRELATION_COEFFICIENT_LABEL, language))
    else:
        norm = (_t(result.normalization, language)
                if result.normalization in _STRINGS
                else result.normalization)
        ax.set_ylabel(_t("Correlation ({norm})", language, norm=norm))
    kind_en = result.kind.capitalize()
    kind = _t(kind_en, language) if kind_en in _STRINGS else kind_en
    ax.set_title(_t("{kind} estimate (Bendat & Piersol)", language, kind=kind))
    ax.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")
    ax.grid(True, alpha=0.3)
    localize_axes(ax, language)
    return ax


def plot_time_delay(
    result: TimeDelayResult, ax: Axes | None = None, *,
    language: str = "en", **kwargs: Any
) -> Axes:
    """Correlation function with the estimated delay marked.

    :param result: A :class:`~phonometry.signals.correlation.TimeDelayResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the correlation ``plot`` call.
    :return: The axes.
    """
    from .._i18n import decimal_comma, localize_axes

    ax = ax if ax is not None else _new_axes()
    label = {
        "direct": "$\\hat{\\rho}_{xy}(\\tau)$",
        "gcc": f"GCC ({result.weighting})",
        "phase": _t(r"$\hat{R}_{xy}(\tau)$ (context)", language),
    }[result.method]
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("label", label)
    ax.plot(result.lags, result.correlation, **kwargs)
    if result.delay_interval is not None:
        ax.axvspan(
            result.delay_interval[0],
            result.delay_interval[1],
            color=_C_SECONDARY,
            alpha=0.25,
            lw=0.0,
            label=_t("95 % interval (Eq. 8.130)", language),
        )
    tau = decimal_comma(f"{1e3 * result.delay:.4g}", language)
    ax.axvline(
        result.delay,
        color=_C_REFERENCE,
        ls="--",
        label=f"$\\hat{{\\tau}}_0$ = {tau} ms",
    )
    ax.set_xlabel(_t(_LAG_LABEL, language))
    ax.set_ylabel(
        _t(_CORRELATION_COEFFICIENT_LABEL, language)
        if result.method == "direct"
        else _t("Normalized correlation", language)
    )
    ax.set_title(_t("Time-delay estimate — {method}", language, method=result.method))
    ax.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")
    ax.grid(True, alpha=0.3)
    localize_axes(ax, language)
    return ax


def plot_aligned_impulse_response(
    result: AlignedImpulseResponseResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Reference and aligned impulse responses overlaid.

    :param result: An
        :class:`~phonometry.signals.correlation.AlignedImpulseResponseResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the aligned-IR ``plot`` call.
    :return: The axes.
    """
    from .._i18n import decimal_comma, localize_axes

    ax = ax if ax is not None else _new_axes()
    t = np.arange(result.reference.size) / result.fs
    ax.plot(t, result.reference, color=_C_MUTED, lw=1.0,
            label=_t("Reference IR", language))
    kwargs.setdefault("color", _C_PRIMARY)
    n = decimal_comma(f"{result.delay_samples:+.3f}", language)
    kwargs.setdefault("label", _t("Aligned IR (delay {n} samples)", language, n=n))
    ax.plot(t, result.aligned, lw=1.2, **kwargs)
    ax.set_xlabel(_t(_TIME_LABEL, language))
    ax.set_ylabel(_t("Amplitude", language))
    ax.set_title(_t("Impulse-response alignment (sub-sample)", language))
    ax.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")
    ax.grid(True, alpha=0.3)
    localize_axes(ax, language)
    return ax


def plot_envelope(
    result: EnvelopeResult, ax: Axes | None = None, *,
    language: str = "en", **kwargs: Any
) -> Axes | np.ndarray:
    """Signal with its Hilbert envelope, plus the instantaneous frequency.

    With ``ax`` given, only the signal/envelope panel is drawn on it.

    :param result: An :class:`~phonometry.signals.envelope.EnvelopeResult`.
    :param ax: Existing axes for the envelope panel, or ``None`` for a
        fresh two-panel figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the envelope ``plot`` call.
    :return: The envelope axes (``ax`` given) or the array of two axes.
    """
    from .._i18n import localize_axes

    t_signal = np.arange(result.signal.size) / result.signal_fs

    def _envelope_panel(axe: Axes) -> None:
        axe.plot(
            t_signal, result.signal, color=_C_MUTED, lw=0.7,
            label=_t("Signal", language)
        )
        kwargs.setdefault("color", _C_PRIMARY)
        kwargs.setdefault("label", _t("Envelope $A(t)$ (Eq. 13.17)", language))
        axe.plot(result.times, result.envelope, lw=1.8, **kwargs)
        axe.set_ylabel(_t("Amplitude", language))
        axe.grid(True, alpha=0.3)
        axe.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")

    if ax is not None:
        _envelope_panel(ax)
        ax.set_xlabel(_t(_TIME_LABEL, language))
        localize_axes(ax, language)
        return ax

    axes = _new_axes_column(2, sharex=True, figsize=(8.0, 5.6))
    _envelope_panel(axes[0])
    axes[0].set_title(_t("Hilbert envelope (Bendat & Piersol Ch. 13)", language))
    axes[1].plot(
        result.times,
        result.instantaneous_frequency,
        color=_C_SECONDARY,
        lw=1.0,
    )
    axes[1].set_ylabel(_t("Instantaneous frequency [Hz]", language))
    axes[1].set_xlabel(_t(_TIME_LABEL, language))
    axes[1].grid(True, alpha=0.3)
    for axf in axes:
        localize_axes(axf, language)
    return axes


def plot_phase_decomposition(
    result: PhaseDecompositionResult, ax: Axes | None = None, *,
    language: str = "en", **kwargs: Any
) -> Axes | np.ndarray:
    """Magnitude, phase decomposition and group delay of a response.

    Three stacked panels: ``|H|`` in dB, the measured / minimum / excess
    phases in radians, and the total and excess group delays in
    milliseconds. With ``ax`` given, only the phase panel is drawn on it.

    :param result: A :class:`~phonometry.signals.phase.PhaseDecompositionResult`.
    :param ax: Existing axes for the phase panel, or ``None`` for a fresh
        three-panel figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the measured-phase ``plot`` call.
    :return: The phase-panel axes (``ax`` given) or the array of three axes.
    """
    from .._i18n import localize_axes

    freqs = np.asarray(result.frequencies, dtype=np.float64)
    pos = freqs > 0.0

    def _phase_panel(axp: Axes) -> None:
        kwargs.setdefault("color", _C_PRIMARY)
        kwargs.setdefault("label", _t("Measured phase", language))
        axp.semilogx(freqs[pos], result.phase[pos], **kwargs)
        axp.semilogx(
            freqs[pos], result.minimum_phase[pos], color=_C_SECONDARY,
            ls="--", label=_t("Minimum phase (from |H|)", language),
        )
        axp.semilogx(
            freqs[pos], result.excess_phase[pos], color=_C_MUTED,
            label=_t("Excess phase (all-pass)", language),
        )
        axp.set_ylabel(_t("Phase [rad]", language))
        axp.grid(True, which="both", alpha=0.3)
        axp.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")

    fmin, fmax = float(freqs[pos].min()), float(freqs[pos].max())
    if ax is not None:
        _phase_panel(ax)
        ax.set_xlabel(_t(_FREQ_LABEL, language))
        ax.set_title(_t("Phase decomposition", language))
        format_frequency_axis(ax, fmin, fmax)
        localize_axes(ax, language)
        return ax

    axes = _new_axes_column(3, sharex=True, figsize=(8.0, 8.0))
    tiny = np.finfo(np.float64).tiny
    axes[0].semilogx(
        freqs[pos],
        20.0 * np.log10(np.maximum(result.magnitude[pos], tiny)),
        color=_C_PRIMARY,
    )
    axes[0].set_ylabel(_t(_MAGNITUDE_LABEL, language))
    axes[0].set_title(_t("Minimum-phase / all-pass decomposition", language))
    axes[0].grid(True, which="both", alpha=0.3)
    _phase_panel(axes[1])
    axes[2].semilogx(
        freqs[pos], 1e3 * result.group_delay[pos], color=_C_PRIMARY,
        label=_t("Group delay", language),
    )
    axes[2].semilogx(
        freqs[pos], 1e3 * result.excess_group_delay[pos], color=_C_MUTED,
        label=_t("Excess group delay", language),
    )
    axes[2].set_ylabel(_t("Group delay [ms]", language))
    axes[2].set_xlabel(_t(_FREQ_LABEL, language))
    axes[2].grid(True, which="both", alpha=0.3)
    axes[2].legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")
    for axf in axes:
        format_frequency_axis(axf, fmin, fmax)
        localize_axes(axf, language)
    return axes


def plot_tone_burst(
    result: ToneBurstResult, ax: Axes | None = None, *,
    language: str = "en", **kwargs: Any
) -> Axes:
    """Burst waveform with its rectangular gating envelope.

    :param result: A :class:`~phonometry.signals.test_signals.ToneBurstResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the waveform ``plot`` call.
    :return: The axes.
    """
    from .._i18n import decimal_comma, localize_axes

    ax = ax if ax is not None else _new_axes()
    t = np.arange(result.signal.size) / result.fs
    kwargs.setdefault("color", _C_PRIMARY)
    ax.plot(t, result.signal, **kwargs)
    for sign in (1.0, -1.0):
        ax.plot(
            t, sign * result.envelope, color=_C_SECONDARY, lw=1.4,
            linestyle="--",
            label=_t("Gating envelope", language) if sign > 0 else None,
        )
    f = decimal_comma(f"{result.frequency:g}", language)
    if result.repetition_rate is None:
        title = _t("Tone burst (IEC 60268-1): {f} Hz, {cycles} cycles",
                   language, f=f, cycles=result.cycles)
    else:
        rate = decimal_comma(f"{result.repetition_rate:g}", language)
        title = _t(
            "Tone burst (IEC 60268-1): {f} Hz, {cycles} cycles, {rate}/s",
            language, f=f, cycles=result.cycles, rate=rate,
        )
    ax.set_title(title)
    ax.set_xlabel(_t(_TIME_LABEL, language))
    ax.set_ylabel(_t("Amplitude", language))
    ax.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")
    ax.grid(True, alpha=0.3)
    localize_axes(ax, language)
    return ax


def plot_resampled_signal(
    result: ResampledSignalResult, ax: Axes | None = None, *,
    language: str = "en", **kwargs: Any
) -> Axes:
    """Anti-alias filter magnitude with the design edges and attenuation.

    The magnitude response of the Kaiser lowpass the polyphase engine
    applied (evaluated from :attr:`filter_taps` at the intermediate rate
    ``original_fs·up``), with the passband edge, the stopband edge at the
    smaller Nyquist frequency (where aliases fold), the designed stopband
    attenuation line and the rejected band shaded — the delivered
    anti-alias spec, read off the delivered filter.

    :param result: A
        :class:`~phonometry.signals.test_signals.ResampledSignalResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the magnitude ``plot`` call.
    :return: The axes.
    """
    from scipy import signal as sp_signal

    from .._i18n import format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    fs_up = result.original_fs * result.up
    # The interesting region is around the band edges; for large ``up`` the
    # intermediate Nyquist frequency sits decades above them, so the view is
    # capped at four times the stopband edge (the response beyond is just
    # more stopband floor).
    f_hi = min(fs_up / 2.0, 4.0 * result.stopband_edge_hz)
    f_lo = result.stopband_edge_hz / 8.0
    freqs, h = sp_signal.freqz(
        result.filter_taps, worN=1 << 16, fs=fs_up
    )
    tiny = np.finfo(np.float64).tiny
    mag_db = 20.0 * np.log10(np.maximum(np.abs(h), tiny))
    view = (freqs > 0.0) & (freqs <= f_hi)
    kwargs.setdefault("color", _C_PRIMARY)
    if "lw" not in kwargs and "linewidth" not in kwargs:
        kwargs["lw"] = 1.2
    ax.semilogx(freqs[view], mag_db[view],
                label=_t("Anti-alias filter $|H(f)|$", language), **kwargs)
    ax.axvline(result.passband_edge_hz, color=_C_TERTIARY, linestyle="--",
               lw=1.2, label=_t("Passband edge", language))
    ax.axvline(result.stopband_edge_hz, color=_C_SECONDARY, linestyle="--",
               lw=1.2, label=_t("Stopband edge (alias fold)", language))
    atten = format_number(result.stopband_attenuation_db, language,
                          decimals=0)
    ax.axhline(-result.stopband_attenuation_db, color=_C_MUTED,
               linestyle=":", lw=1.2,
               label=_t("Design attenuation −{a} dB", language, a=atten))
    ax.axvspan(result.stopband_edge_hz, f_hi, color=_C_SECONDARY,
               alpha=0.08,
               label=_t("Rejected band (would fold back as aliases)",
                        language))
    ax.set_xlabel(_t(_FREQ_LABEL, language))
    ax.set_ylabel(_t(_MAGNITUDE_LABEL, language))
    ax.set_ylim(-result.stopband_attenuation_db - 40.0, 10.0)
    fs0 = format_number(result.original_fs, language, decimals=0)
    fs1 = format_number(result.fs, language, decimals=0)
    ax.set_title(_t(
        "Polyphase resampling {fs0} Hz → {fs1} Hz (L/M = {up}/{down}, "
        "{taps} taps)",
        language, fs0=fs0, fs1=fs1, up=result.up, down=result.down,
        taps=result.n_taps,
    ))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize="small")
    ax.set_xlim(f_lo, f_hi)
    format_frequency_axis(ax, f_lo, f_hi, minor=None)
    localize_axes(ax, language)
    return ax
# ---------------------------------------------------------------------------
# Cepstral analysis (Havelock Chs. 27/87) and envelope spectrum (B&P 13.3)
# ---------------------------------------------------------------------------
_CEPSTRUM_TITLES = {
    "power": "Power cepstrum",
    "real": "Real cepstrum",
    "complex": "Complex cepstrum",
}


def plot_cepstrum(
    result: CepstrumResult, ax: Axes | None = None, *,
    language: str = "en", **kwargs: Any
) -> Axes:
    """Cepstrum against quefrency, over the unambiguous first half-axis.

    :param result: A :class:`~phonometry.signals.cepstrum.CepstrumResult`.
    :param ax: Existing axes, or ``None`` for a fresh figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the cepstrum line.
    :return: The axes.
    """
    from .._i18n import localize_axes

    if ax is None:
        ax = _new_axes()
        ax.set_title(_t(_CEPSTRUM_TITLES[result.kind], language))
    half = result.nfft // 2 + 1
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("lw", 1.0)
    ax.plot(1e3 * result.quefrencies[:half], result.cepstrum[:half], **kwargs)
    ax.set_xlabel(_t(_QUEFRENCY_LABEL, language))
    ax.set_ylabel(_t("Cepstrum", language))
    ax.grid(True, alpha=0.3)
    localize_axes(ax, language)
    return ax


def plot_window_metrics(
    result: WindowMetricsResult, ax: Axes | None = None, *,
    language: str = "en", **kwargs: Any
) -> Axes | np.ndarray:
    """Window shape and spectrum with the Harris figures of merit marked.

    With ``ax`` given, only the spectrum panel is drawn on it.

    :param result: A
        :class:`~phonometry.signals.spectra.WindowMetricsResult`.
    :param ax: Existing axes for the spectrum panel, or ``None`` for a
        fresh two-panel figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the spectrum ``plot`` call.
    :return: The spectrum axes (``ax`` given) or the array of two axes.
    """
    from .._i18n import decimal_comma, localize_axes
    from ..signals.spectra import _WINDOW_OVERSAMPLE, _window_spectrum_db

    max_bins = 24.0

    def _spectrum_panel(axs: Axes) -> None:
        level = _window_spectrum_db(result.taps, _WINDOW_OVERSAMPLE)
        bins = np.arange(level.size) / _WINDOW_OVERSAMPLE
        shown = bins <= max_bins
        kwargs.setdefault("color", _C_PRIMARY)
        enbw = decimal_comma(f"{result.enbw_bins:.3f}", language)
        kwargs.setdefault("label", _t(_ENBW_LABEL, language, enbw=enbw))
        axs.plot(bins[shown], level[shown], **kwargs)
        sll = decimal_comma(f"{result.highest_sidelobe_db:.1f}", language)
        axs.axhline(
            result.highest_sidelobe_db, color=_C_REFERENCE, lw=1.0,
            linestyle="--",
            label=_t("Highest sidelobe {sll} dB", language, sll=sll),
        )
        sl = decimal_comma(f"{result.scalloping_loss_db:.2f}", language)
        axs.plot(
            [0.5], [-result.scalloping_loss_db], "o", color=_C_SECONDARY,
            ms=5.0, label=_t("Scalloping loss {sl} dB", language, sl=sl),
        )
        axs.set_xlim(0.0, max_bins)
        axs.set_ylim(bottom=max(-140.0, float(np.min(level[shown])) - 5.0))
        axs.set_xlabel(_t("Frequency offset [DFT bins]", language))
        axs.set_ylabel(_t("Level re main lobe [dB]", language))
        axs.grid(True, alpha=0.3)
        axs.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")

    title = _t("Window metrics (Harris 1978): {window}", language,
               window=result.window)
    if ax is not None:
        _spectrum_panel(ax)
        ax.set_title(title)
        localize_axes(ax, language)
        return ax

    axes = _new_axes_column(2, figsize=(8.0, 6.5))
    axes[0].plot(np.arange(result.n), result.taps, color=_C_PRIMARY, lw=1.2)
    axes[0].set_xlabel(_t("Sample", language))
    axes[0].set_ylabel(_t("Window w[m]", language))
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)
    _spectrum_panel(axes[1])
    for axf in axes:
        localize_axes(axf, language)
    axes[0].figure.tight_layout()
    return axes


def plot_lifter(
    result: LifterResult, ax: Axes | None = None, *,
    language: str = "en", **kwargs: Any
) -> Axes | np.ndarray:
    """Real cepstrum with the lifter cutoff, plus the split log spectrum.

    With ``ax`` given, only the spectrum panel is drawn on it.

    :param result: A :class:`~phonometry.signals.cepstrum.LifterResult`.
    :param ax: Existing axes for the spectrum panel, or ``None`` for a
        fresh two-panel figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the liftered-spectrum line.
    :return: The spectrum axes (``ax`` given) or the array of two axes.
    """
    from .._i18n import format_number, localize_axes

    cutoff_ms = format_number(1e3 * result.cutoff, language, decimals=2,
                              trim=True)

    def _spectrum_panel(axe: Axes) -> None:
        axe.plot(
            result.frequencies, result.spectrum_db, color=_C_MUTED, lw=0.8,
            label=_t("Log spectrum", language),
        )
        kwargs.setdefault("color", _C_PRIMARY)
        kwargs.setdefault(
            "label",
            _t("Liftered ({mode})", language,
               mode=_t(result.mode, language)),
        )
        axe.plot(result.frequencies, result.liftered_db, lw=1.6, **kwargs)
        axe.set_xlabel(_t(_FREQ_LABEL, language))
        axe.set_ylabel(_t(_MAGNITUDE_LABEL, language))
        axe.grid(True, alpha=0.3)
        axe.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")

    if ax is not None:
        _spectrum_panel(ax)
        localize_axes(ax, language)
        return ax

    axes = _new_axes_column(2, figsize=(8.0, 6.4))
    half = result.nfft // 2 + 1
    axes[0].plot(
        1e3 * result.quefrencies[:half], result.cepstrum[:half],
        color=_C_SECONDARY, lw=1.0,
    )
    axes[0].axvline(
        1e3 * result.cutoff, color=_C_REFERENCE, linestyle="--", lw=1.2,
        label=_t("Lifter cutoff ({q} ms)", language, q=cutoff_ms),
    )
    axes[0].set_xlabel(_t(_QUEFRENCY_LABEL, language))
    axes[0].set_ylabel(_t("Cepstrum", language))
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")
    axes[0].set_title(
        _t("Liftering at {q} ms ({mode})", language, q=cutoff_ms,
           mode=_t(result.mode, language))
    )
    _spectrum_panel(axes[1])
    for axf in axes:
        localize_axes(axf, language)
    return axes


def plot_echo_detection(
    result: EchoDetectionResult, ax: Axes | None = None, *,
    language: str = "en", **kwargs: Any
) -> Axes:
    """Power cepstrum with the searched band and the detected echo marked.

    :param result: An
        :class:`~phonometry.signals.cepstrum.EchoDetectionResult`.
    :param ax: Existing axes, or ``None`` for a fresh figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the cepstrum line.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    if ax is None:
        ax = _new_axes()
        ax.set_title(_t("Echo detection on the power cepstrum", language))
    half = result.nfft // 2 + 1
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("lw", 1.0)
    ax.plot(1e3 * result.quefrencies[:half], result.cepstrum[:half], **kwargs)
    ax.axvspan(
        1e3 * result.search_range[0], 1e3 * result.search_range[1],
        color=_C_PRIMARY_LIGHT, alpha=0.25,
        label=_t("Searched band", language),
    )
    ax.plot(
        [1e3 * result.delay], [result.reflection_coefficient], "v",
        color=_C_SECONDARY, markersize=9,
        label=_t(
            "Echo: {delay} ms, a = {a}", language,
            delay=format_number(1e3 * result.delay, language, decimals=2,
                                trim=True),
            a=format_number(result.reflection_coefficient, language,
                            decimals=3, trim=True),
        ),
    )
    ax.set_xlabel(_t(_QUEFRENCY_LABEL, language))
    ax.set_ylabel(_t("Cepstrum", language))
    ax.grid(True, alpha=0.3)
    ax.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")
    localize_axes(ax, language)
    return ax


def plot_envelope_spectrum(
    result: EnvelopeSpectrumResult, ax: Axes | None = None, *,
    language: str = "en", **kwargs: Any
) -> Axes | np.ndarray:
    """Detected envelope over time and its amplitude spectrum.

    With ``ax`` given, only the spectrum panel is drawn on it.

    :param result: An
        :class:`~phonometry.signals.envelope.EnvelopeSpectrumResult`.
    :param ax: Existing axes for the spectrum panel, or ``None`` for a
        fresh two-panel figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the spectrum line.
    :return: The spectrum axes (``ax`` given) or the array of two axes.
    """
    from .._i18n import localize_axes

    def _spectrum_panel(axe: Axes) -> None:
        kwargs.setdefault("color", _C_PRIMARY)
        kwargs.setdefault("lw", 1.2)
        axe.plot(result.frequencies, result.amplitude, **kwargs)
        axe.set_xlabel(_t(_FREQ_LABEL, language))
        axe.set_ylabel(_t("Modulation amplitude", language))
        axe.grid(True, alpha=0.3)

    if ax is not None:
        _spectrum_panel(ax)
        localize_axes(ax, language)
        return ax

    axes = _new_axes_column(2, figsize=(8.0, 6.4))
    axes[0].plot(
        result.times, result.envelope, color=_C_SECONDARY, lw=1.0,
        label=_t("Envelope ({kind})", language,
                 kind=_t(result.kind, language)),
    )
    axes[0].axhline(
        result.mean_level, color=_C_REFERENCE, linestyle="--", lw=1.2,
        label=_t("Mean level", language),
    )
    axes[0].set_xlabel(_t(_TIME_LABEL, language))
    axes[0].set_ylabel(_t("Amplitude", language))
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")
    axes[0].set_title(_t("Envelope spectrum (Bendat & Piersol 13.3)", language))
    _spectrum_panel(axes[1])
    for axf in axes:
        localize_axes(axf, language)
    return axes


#: Y-axis labels of the stationarity plot, keyed by the segment statistic.
def plot_inverse_filter(
    result: Any, ax: Axes | None = None, *, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Measured, inverse and equalized magnitudes of a regularized inversion.

    One panel over log-frequency: the measured response ``|H|``
    (normalised to its peak), the inverse-filter gain ``|H_inv|`` on the
    same reference, and the equalized product ``|H*H_inv|`` that reads
    0 dB across the shaded equalized band and rolls off outside it, where
    the frequency-dependent regularization caps the gain.

    :param result: An :class:`~phonometry.signals.inversion.InverseFilterResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the measured-response ``plot`` call.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    pos = freqs > 0.0
    tiny = np.finfo(np.float64).tiny
    h_mag = np.abs(np.asarray(result.response_spectrum))
    peak = float(np.max(h_mag))
    inv_mag = np.abs(np.asarray(result.spectrum))
    eq_mag = h_mag * inv_mag
    color = kwargs.pop("color", _C_PRIMARY)
    f1, f2 = result.f_range

    ax.semilogx(
        freqs[pos], 20.0 * np.log10(np.maximum(h_mag[pos], tiny) / peak),
        color=color, lw=1.2, label=_t("Measured response $|H|$", language),
        **kwargs,
    )
    ax.semilogx(
        freqs[pos], 20.0 * np.log10(np.maximum(inv_mag[pos] * peak, tiny)),
        color=_C_SECONDARY, lw=1.2,
        label=_t(r"Inverse filter $|H_{\mathrm{inv}}|$", language),
    )
    ax.semilogx(
        freqs[pos], 20.0 * np.log10(np.maximum(eq_mag[pos], tiny)),
        color=_C_TERTIARY, lw=1.6,
        label=_t(r"Equalized $|H \cdot H_{\mathrm{inv}}|$", language),
    )
    ax.axvspan(f1, f2, color=color, alpha=0.08,
               label=_t("Equalized band", language))
    ax.set_xlabel(_t(_FREQ_LABEL, language))
    ax.set_ylabel(_t(_MAGNITUDE_LABEL, language))
    ax.set_ylim(bottom=-60.0, top=20.0)
    flat = format_number(result.flatness_db, language, decimals=2)
    ax.set_title(_t("Regularized inversion (Kirkeby) — flatness {flat} dB",
                    language, flat=flat))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")
    format_frequency_axis(ax, float(freqs[pos].min()), float(freqs[pos].max()))
    localize_axes(ax, language)
    return ax
def plot_synchronous_average(
    result: SynchronousAverageResult, ax: Axes | None = None, *,
    language: str = "en", **kwargs: Any
) -> Axes | np.ndarray:
    """Averaged periodic waveform and the synchronous-averaging comb filter.

    With ``ax`` given, only the averaged-waveform panel is drawn on it.

    :param result: A
        :class:`~phonometry.signals.synchronous_average.SynchronousAverageResult`.
    :param ax: Existing axes for the waveform panel, or ``None`` for a fresh
        two-panel (waveform + comb filter) figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the averaged-waveform line.
    :return: The waveform axes (``ax`` given) or the array of two axes.
    """
    from .._i18n import localize_axes

    def _waveform(axw: Axes) -> None:
        kwargs.setdefault("color", _C_PRIMARY)
        kwargs.setdefault("lw", 1.6)
        axw.plot(
            1e3 * result.times, result.period_waveform,
            label=_t("Averaged periodic waveform (N = {n})", language,
                     n=result.n_averages),
            **kwargs,
        )
        axw.set_xlabel(_t("Time [ms]", language))
        axw.set_ylabel(_t("Amplitude", language))
        axw.grid(True, alpha=0.3)
        axw.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")

    if ax is not None:
        _waveform(ax)
        localize_axes(ax, language)
        return ax

    axes = _new_axes_column(2, figsize=(8.0, 6.4))
    _waveform(axes[0])
    axes[0].set_title(_t("Time synchronous average (McFadden 1987)", language))

    orders = result.comb_frequencies * result.period
    axes[1].plot(orders, result.comb_response, color=_C_PRIMARY, lw=1.2)
    top = int(np.floor(orders[-1] + 1e-9))
    for k in range(1, top + 1):
        axes[1].axvline(
            float(k), color=_C_REFERENCE, linestyle=":", lw=0.8, alpha=0.6,
            label=_t("Harmonics of $1/T$", language) if k == 1 else None,
        )
    axes[1].set_xlabel(_t("Frequency [orders]", language))
    axes[1].set_ylabel(_t(r"Comb filter $|C(f)|$ (Eq. 8)", language))
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")
    for axf in axes:
        localize_axes(axf, language)
    return axes
