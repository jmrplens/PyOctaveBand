#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Plot renderers for the metrology domain (lazy imports from result .plot())."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..metrology.data_qualification import (
        LevelCrossingResult,
        PeakStatisticsResult,
        StationarityTestResult,
        TrendTestResult,
    )
    from ..metrology.intensity_compliance import (
        IntensityInstrumentComplianceResult,
    )
    from ..metrology.uncertainty import MonteCarloResult, UncertaintyResult

from .common import (
    _C_MUTED,
    _C_PRIMARY,
    _C_PRIMARY_LIGHT,
    _C_REFERENCE,
    _C_SECONDARY,
    _C_TERTIARY,
    _LEGEND_UPPER_RIGHT,
    _new_axes,
    format_frequency_axis,
    theme_fill,
)

#: Spanish translations of the fixed strings rendered by the metrology
#: ``.plot()`` renderers, keyed by their verbatim English text. ``_t``
#: returns the English key unchanged for any language other than ``"es"``,
#: so the English output is byte-for-byte identical to the pre-i18n
#: renderers.
_STRINGS: dict[str, str] = {
    "Frequency [Hz]": "Frecuencia [Hz]",
    "Class {cls} pass region": "Región de aceptación clase {cls}",
    "Class 1 minimum": "Mínimo clase 1",
    "Class 2 minimum": "Mínimo clase 2",
    r"Measured $\delta_{pI0}$": r"$\delta_{pI0}$ medido",
    "Below the class {cls} minimum": "Bajo el mínimo de clase {cls}",
    r"$\delta_{pI0}$ [dB]": r"$\delta_{pI0}$ [dB]",
    "IEC 61043 Table 2 — {device}, {spacing} mm separation":
        "Tabla 2 de IEC 61043 — {device}, separación de {spacing} mm",
    "probe": "sonda",
    "processor": "procesador",
    "complete instrument": "instrumento completo",
    r"Contribution to combined uncertainty $|c_i|\,u(x_i)$":
        r"Contribución a la incertidumbre combinada $|c_i|\,u(x_i)$",
    "GUM uncertainty budget — y = {value}":
        "Presupuesto de incertidumbre (GUM) — y = {value}",
    "{pct} % coverage interval": "Intervalo de cobertura {pct} %",
    "Output quantity y": "Magnitud de salida y",
    "Probability density": "Densidad de probabilidad",
    "Monte Carlo distribution (GUM Supplement 1) — u(y) = {uy}":
        "Distribución de Monte Carlo (GUM Suplemento 1) — u(y) = {uy}",
    "Sample": "Muestra",
    "Segment mean square": "Media cuadrática por segmento",
    "Segment RMS": "RMS por segmento",
    "Segment mean": "Media por segmento",
    "Segment variance": "Varianza por segmento",
    "Sequence median": "Mediana de la secuencia",
    "Segment index": "Índice de segmento",
    "Sample index": "Índice de muestra",
    "Sequence value": "Valor de la secuencia",
    "Trend test (Bendat & Piersol 4.5.2)":
        "Test de tendencia (Bendat y Piersol 4.5.2)",
    "no trend": "sin tendencia",
    "trend": "tendencia",
    "Stationarity test (Bendat & Piersol 10.3.1.1)":
        "Test de estacionariedad (Bendat y Piersol 10.3.1.1)",
    "stationary": "estacionario",
    "nonstationary": "no estacionario",
    "Reverse arrangements A = {a}, accept ({lo}, {hi}]: {verdict}":
        "Inversiones de orden A = {a}, aceptación ({lo}, {hi}]: {verdict}",
    "Runs r = {r}, accept ({lo}, {hi}]: {verdict}":
        "Rachas r = {r}, aceptación ({lo}, {hi}]: {verdict}",
    "Measured rate": "Tasa medida",
    "Rice expectation (Eq. 5.196)": "Expectativa de Rice (Ec. 5.196)",
    "Level a [signal units]": "Nivel a [unidades de la señal]",
    "Crossings per second [1/s]": "Cruces por segundo [1/s]",
    "Level-crossing rate (Bendat & Piersol 5.5.1)":
        "Tasa de cruces por nivel (Bendat y Piersol 5.5.1)",
    "Empirical peak exceedance": "Excedencia empírica de picos",
    "Rice (r = {r})": "Rice (r = {r})",
    "Rayleigh limit (r = 1)": "Límite de Rayleigh (r = 1)",
    "Gaussian limit (r = 0)": "Límite gaussiano (r = 0)",
    r"Standardized peak height $z = a/\sigma_x$":
        r"Altura de pico estandarizada $z = a/\sigma_x$",
    "Prob[peak > z]": "Prob[pico > z]",
    "Peak-height distribution (Bendat & Piersol 5.5.4)":
        "Distribución de alturas de pico (Bendat y Piersol 5.5.4)",
}


def _t(text: str, language: str = "en", **fmt: Any) -> str:
    """Localise a fixed string; English is returned verbatim (byte-identical)."""
    s = _STRINGS.get(text, text) if language == "es" else text
    return s.format(**fmt) if fmt else s


#: Shared frequency-axis label of the spectral renderers.
_FREQ_LABEL = "Frequency [Hz]"
#: Y-axis label of the residual-index plots (identical in both languages,
#: the symbol carries the meaning).
_LABEL_RESIDUAL_INDEX = r"$\delta_{pI0}$ [dB]"
_DEVICE_LABELS = {
    "probe": "probe",
    "processor": "processor",
    "instrument": "complete instrument",
}


def plot_intensity_class(
    result: IntensityInstrumentComplianceResult, ax: Axes | None = None, *,
    language: str = "en", **kwargs: Any
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
        :class:`~phonometry.metrology.intensity_compliance.IntensityInstrumentComplianceResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the measured-curve ``plot`` call.
    :return: The axes.
    """
    from .._i18n import localize_axes

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
        freqs, mask, y_top, step="mid", facecolor=theme_fill(_C_TERTIARY, ax),
        edgecolor="none", zorder=0,
        label=_t("Class {cls} pass region", language, cls=cls),
    )
    # Both Table 2 masks in the same amber, class 1 solid and class 2 dashed,
    # as the published intensity-analyser displays draw them.
    ax.plot(freqs, class1, drawstyle="steps-mid", color=_C_SECONDARY, lw=1.3,
            label=_t("Class 1 minimum", language))
    ax.plot(freqs, class2, drawstyle="steps-mid", color=_C_SECONDARY, lw=1.3,
            ls="--", label=_t("Class 2 minimum", language))

    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("lw", 1.6)
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("ms", 3.0)
    kwargs.setdefault("drawstyle", "steps-mid")
    kwargs.setdefault("label", _t(r"Measured $\delta_{pI0}$", language))
    ax.plot(freqs, measured, **kwargs)

    # Ring the bands that block the next class up: class 1 for a class 2 chain,
    # class 2 for a chain that meets neither. A class 1 chain has none.
    marked_cls = 1 if result.overall_class == 2 else 2
    marked_mask = class1 if marked_cls == 1 else class2
    failing = (
        np.zeros(freqs.shape, dtype=bool)
        if result.overall_class == 1
        else measured < marked_mask - 1e-9
    )
    if np.any(failing):
        ax.plot(
            freqs[failing], measured[failing], ls="", marker="o", ms=6.0,
            mfc="none", mew=1.6, color=_C_REFERENCE,
            label=_t("Below the class {cls} minimum", language, cls=marked_cls),
        )

    format_frequency_axis(ax, float(freqs.min()), float(freqs.max()))
    ax.set_xlim(float(freqs.min()) / 1.15, float(freqs.max()) * 1.15)
    ax.set_ylim(y_bot, y_top)
    ax.set_xlabel(_t(_FREQ_LABEL, language))
    ax.set_ylabel(_t(_LABEL_RESIDUAL_INDEX, language))
    ax.set_title(
        _t("IEC 61043 Table 2 — {device}, {spacing} mm separation", language,
           device=_t(_DEVICE_LABELS[result.device], language),
           spacing=f"{result.spacing * 1000.0:g}")
    )
    ax.legend(loc="lower right", fontsize="small")
    ax.grid(True, which="both", alpha=0.3)
    localize_axes(ax, language)
    return ax
def plot_uncertainty_budget(
    result: UncertaintyResult, ax: Axes | None = None, *,
    language: str = "en", **kwargs: Any
) -> Axes:
    """Bar chart of each input's contribution to the combined uncertainty.

    :param result: An :class:`~phonometry.uncertainty.UncertaintyResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to :meth:`barh`.
    :return: The axes.
    """
    # The y-axis carries categorical input names (a FixedFormatter), so
    # localize_axes is intentionally not applied here: it would overwrite
    # those labels with comma-formatted tick numbers.
    from .._i18n import decimal_comma

    ax = ax if ax is not None else _new_axes()
    contributions = np.asarray(result.contributions, dtype=np.float64)
    names = list(result.names) or [f"x{i + 1}" for i in range(contributions.size)]
    positions = np.arange(contributions.size)
    kwargs.setdefault("color", _C_PRIMARY)
    ax.barh(positions, contributions, **kwargs)
    uc = decimal_comma(f"{result.combined_uncertainty:.3g}", language)
    ax.axvline(result.combined_uncertainty, color=_C_REFERENCE, ls="--",
               label=f"$u_c$ = {uc}")
    ax.set_yticks(positions)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel(_t(r"Contribution to combined uncertainty $|c_i|\,u(x_i)$", language))
    value = decimal_comma(f"{result.value:.4g}", language)
    ax.set_title(_t("GUM uncertainty budget — y = {value}", language, value=value))
    ax.legend(loc="lower right", fontsize="small")
    ax.grid(True, axis="x", alpha=0.3)
    return ax

def plot_monte_carlo(
    result: MonteCarloResult, ax: Axes | None = None, *,
    language: str = "en", **kwargs: Any
) -> Axes:
    """Histogram of the Monte Carlo output with the coverage interval marked.

    :param result: A :class:`~phonometry.uncertainty.MonteCarloResult`
        obtained with ``keep_samples=True`` (the histogram needs the raw
        output sample).
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to :meth:`~matplotlib.axes.Axes.hist`.
    :return: The axes.
    :raises ValueError: If the result carries no output samples.
    """
    from .._i18n import decimal_comma, localize_axes

    if result.samples is None:
        raise ValueError(
            "plot() needs the Monte Carlo output samples; call "
            "monte_carlo(..., keep_samples=True) to retain them."
        )
    ax = ax if ax is not None else _new_axes()
    samples = np.asarray(result.samples, dtype=np.float64)
    kwargs.setdefault("color", _C_PRIMARY_LIGHT)
    kwargs.setdefault("bins", 120)
    kwargs.setdefault("density", True)
    ax.hist(samples, **kwargs)
    low, high = result.interval
    pct = decimal_comma(f"{100.0 * result.coverage:g}", language)
    ax.axvspan(low, high, color=_C_PRIMARY, alpha=0.12,
               label=_t("{pct} % coverage interval", language, pct=pct))
    value = decimal_comma(f"{result.value:.4g}", language)
    ax.axvline(result.value, color=_C_REFERENCE, ls="--",
               label=f"y = {value}")
    ax.set_xlabel(_t("Output quantity y", language))
    ax.set_ylabel(_t("Probability density", language))
    uy = decimal_comma(f"{result.standard_uncertainty:.3g}", language)
    ax.set_title(_t("Monte Carlo distribution (GUM Supplement 1) — u(y) = {uy}", language, uy=uy))
    ax.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")
    ax.grid(True, axis="y", alpha=0.3)
    localize_axes(ax, language)
    return ax
_SEGMENT_LABELS = {
    "mean_square": "Segment mean square",
    "rms": "Segment RMS",
    "mean": "Segment mean",
    "variance": "Segment variance",
}


def _trend_verdict_label(
    method: str, count: int, bounds: tuple[int, int], verdict: str,
    language: str,
) -> str:
    """Legend label naming the count, acceptance region and verdict.

    Shared by :func:`plot_trend_test` and :func:`plot_stationarity_test`,
    which draw the same reverse-arrangement / runs statistic against their
    own acceptance region.
    """
    template = (
        "Reverse arrangements A = {a}, accept ({lo}, {hi}]: {verdict}"
        if method == "reverse_arrangements"
        else "Runs r = {r}, accept ({lo}, {hi}]: {verdict}"
    )
    return _t(
        template, language, a=count, r=count,
        lo=bounds[0], hi=bounds[1], verdict=verdict,
    )


def _draw_sequence_median(ax: Axes, median: float, language: str) -> None:
    """Draw the runs-classification median as a dashed reference line."""
    ax.axhline(
        median, color=_C_REFERENCE, linestyle="--", lw=1.2,
        label=_t("Sequence median", language),
    )


def plot_trend_test(
    result: TrendTestResult, ax: Axes | None = None, *,
    language: str = "en", **kwargs: Any
) -> Axes:
    """Tested sequence against its sample index with the trend-test verdict.

    Draws the sequence of observations ``result.values`` against a plain
    sample index (1 to ``n``) and states the test outcome in the legend:
    the reverse-arrangement count ``A`` (or the run count ``r``), the B&P
    Table A.6 acceptance region and whether the no-trend hypothesis is
    accepted. For the runs test the sequence median is drawn as the
    reference line that classifies each value.

    :param result: A
        :class:`~phonometry.metrology.data_qualification.TrendTestResult`.
    :param ax: Existing axes, or ``None`` for a fresh figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the sequence line.
    :return: The axes.
    """
    from .._i18n import localize_axes

    if ax is None:
        ax = _new_axes()
        ax.set_title(
            _t("Trend test (Bendat & Piersol 4.5.2)", language)
        )
    verdict = _t("no trend" if result.trend_free else "trend", language)
    label = _trend_verdict_label(
        result.method, result.statistic, result.bounds, verdict, language
    )
    index = np.arange(1, result.n + 1)
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("lw", 1.2)
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("ms", 4.5)
    ax.plot(index, result.values, label=label, **kwargs)
    if result.method == "runs" and result.median is not None:
        # The runs test classifies each value against the median of the
        # *original* sequence (before values equal to it were discarded),
        # so draw that persisted classification median, not a median
        # recomputed on the filtered result.values.
        _draw_sequence_median(ax, result.median, language)
    ax.set_xlabel(_t("Sample index", language))
    ax.set_ylabel(_t("Sequence value", language))
    ax.grid(True, alpha=0.3)
    ax.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")
    localize_axes(ax, language)
    return ax


def plot_stationarity_test(
    result: StationarityTestResult, ax: Axes | None = None, *,
    language: str = "en", **kwargs: Any
) -> Axes:
    """Segment-statistic sequence with the trend-test verdict.

    :param result: A
        :class:`~phonometry.metrology.data_qualification.StationarityTestResult`.
    :param ax: Existing axes, or ``None`` for a fresh figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the segment-value line.
    :return: The axes.
    """
    from .._i18n import localize_axes

    if ax is None:
        ax = _new_axes()
        ax.set_title(
            _t("Stationarity test (Bendat & Piersol 10.3.1.1)", language)
        )
    verdict = _t(
        "stationary" if result.stationary else "nonstationary", language
    )
    label = _trend_verdict_label(
        result.method, result.count, result.bounds, verdict, language
    )
    index = np.arange(1, result.n_segments + 1)
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("lw", 1.2)
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("ms", 4.5)
    ax.plot(index, result.segment_values, label=label, **kwargs)
    if result.method == "runs":
        _draw_sequence_median(
            ax, float(np.median(result.segment_values)), language
        )
    ax.set_xlabel(_t("Segment index", language))
    ax.set_ylabel(_t(_SEGMENT_LABELS[result.statistic], language))
    ax.grid(True, alpha=0.3)
    ax.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")
    localize_axes(ax, language)
    return ax


def plot_level_crossing_rate(
    result: LevelCrossingResult, ax: Axes | None = None, *,
    language: str = "en", **kwargs: Any
) -> Axes:
    """Measured level-crossing rates against the Rice curve.

    :param result: A
        :class:`~phonometry.metrology.data_qualification.LevelCrossingResult`.
    :param ax: Existing axes, or ``None`` for a fresh figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the measured-rate markers.
    :return: The axes.
    """
    from .._i18n import localize_axes

    if ax is None:
        ax = _new_axes()
        ax.set_title(
            _t("Level-crossing rate (Bendat & Piersol 5.5.1)", language)
        )
    order = np.argsort(result.levels)
    ax.plot(
        result.levels[order], result.rice_rates[order], color=_C_REFERENCE,
        lw=1.4, label=_t("Rice expectation (Eq. 5.196)", language),
    )
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("ms", 6.0)
    ax.plot(
        result.levels, result.rates, "o",
        label=_t("Measured rate", language), **kwargs,
    )
    ax.set_yscale("log")
    ax.set_xlabel(_t("Level a [signal units]", language))
    ax.set_ylabel(_t("Crossings per second [1/s]", language))
    ax.grid(True, alpha=0.3)
    ax.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")
    localize_axes(ax, language)
    return ax


def plot_peak_statistics(
    result: PeakStatisticsResult, ax: Axes | None = None, *,
    language: str = "en", **kwargs: Any
) -> Axes:
    """Empirical peak exceedance against the Rice closed forms.

    :param result: A
        :class:`~phonometry.metrology.data_qualification.PeakStatisticsResult`.
    :param ax: Existing axes, or ``None`` for a fresh figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the empirical exceedance line.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes
    from ..metrology.data_qualification import _rice_peak_exceedance

    if ax is None:
        ax = _new_axes()
        ax.set_title(
            _t("Peak-height distribution (Bendat & Piersol 5.5.4)", language)
        )
    peaks = result.peak_values
    if peaks.size == 0:
        raise ValueError("The record has no local maxima to plot.")
    exceedance = 1.0 - np.arange(1, peaks.size + 1) / peaks.size
    z = np.linspace(float(peaks[0]), float(peaks[-1]), 400)
    ax.plot(
        z, _rice_peak_exceedance(z, 1.0), color=_C_MUTED, lw=1.0,
        linestyle="--", label=_t("Rayleigh limit (r = 1)", language),
    )
    ax.plot(
        z, _rice_peak_exceedance(z, 0.0), color=_C_MUTED, lw=1.0,
        linestyle=":", label=_t("Gaussian limit (r = 0)", language),
    )
    ax.plot(
        z, result.peak_exceedance(z), color=_C_REFERENCE, lw=1.5,
        label=_t(
            "Rice (r = {r})", language,
            r=format_number(result.irregularity_factor, language, decimals=3,
                            trim=True),
        ),
    )
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("lw", 1.2)
    ax.plot(
        peaks, exceedance, drawstyle="steps-post",
        label=_t("Empirical peak exceedance", language), **kwargs,
    )
    floor = max(1.0 / peaks.size, 1e-6)
    ax.set_yscale("log")
    ax.set_ylim(bottom=floor)
    ax.set_xlabel(_t(r"Standardized peak height $z = a/\sigma_x$", language))
    ax.set_ylabel(_t("Prob[peak > z]", language))
    ax.grid(True, alpha=0.3)
    ax.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")
    localize_axes(ax, language)
    return ax
