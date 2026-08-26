#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Plot renderers for the room domain (lazy imports from result .plot())."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .._internal.validation import require_choice
from .common import (
    _C_EDGE,
    _C_MUTED,
    _C_PRIMARY,
    _C_PRIMARY_LIGHT,
    _C_QUATERNARY,
    _C_REFERENCE,
    _C_SECONDARY,
    _C_SECONDARY_LIGHT,
    _C_TERTIARY,
    _LEGEND_UPPER_RIGHT,
    _band_axis,
    _draw_decay_times,
    _fit_segment,
    _format_freq,
    _freq_axis,
    _new_axes,
    _new_axes_column,
    _time_axis,
    format_frequency_axis,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike

    from ..room.acoustics import DecayCurve, RoomAcousticsResult
    from ..room.crowd_noise import CrowdNoiseResult
    from ..room.enclosed_space_absorption import ReverberationResult
    from ..room.image_source import ImageSourceResult
    from ..room.impulse_response import ImpulseResponseResult, ShapedSweepResult
    from ..room.modes import RoomModesResult
    from ..room.noise_criteria import NCResult, RCResult
    from ..room.open_plan import OpenPlanResult
    from ..room.reverberation_prediction import ReverberationModelResult
    from ..room.steady_field import SteadyFieldResult

#: Shared x-axis label of the frequency-domain room plots.
_FREQUENCY_LABEL = "Frequency [Hz]"

#: Shared time-axis label of the time-domain room plots.
_TIME_LABEL = "Time [s]"

#: Shared legend entry of the backward-integrated energy-decay curve.
_SCHROEDER_DECAY_LABEL = "Schroeder decay"

#: Shared y-axis label of the octave-band spectrum plots (NC / RC).
_OCTAVE_BAND_SPL_LABEL = "Octave-band SPL [dB]"

#: Shared y-axis label of the predicted reverberation-time plots.
_REVERBERATION_TIME_LABEL = "Reverberation time $T$ [s]"

#: Shared legend entry naming the equivalent absorption area of a curve.
_ABSORPTION_AREA_LABEL = "$A$ = {value} m²"

#: Highest octave band shaded with the +5 dB rumble tolerance in the RC Mark II
#: plot (ANSI/ASA S12.2 Annex D, clause D.3: at and below 500 Hz).
_RC_RUMBLE_MAX_HZ = 500.0

#: Lowest octave band shaded with the +3 dB hiss tolerance in the RC Mark II
#: plot (ANSI/ASA S12.2 Annex D, clause D.3: at and above 1000 Hz).
_RC_HISS_MIN_HZ = 1000.0

#: The excitation signals ISO 18233 covers and :func:`plot_excitation` draws,
#: in the order the parameter documents them.
_EXCITATION_KINDS = ("sweep", "mls")

#: Spanish translations of the fixed strings rendered by the room ``.plot()``
#: renderers, keyed by their verbatim English text.  ``_t`` returns the English
#: key unchanged for any language other than ``"es"``, so the English output is
#: byte-for-byte identical to the pre-i18n renderers.
_STRINGS: dict[str, str] = {
    "Reverberation time [s]": "Tiempo de reverberación [s]",
    "ISO 3382 decay times and clarity": "Tiempos de caída y claridad ISO 3382",
    _FREQUENCY_LABEL: "Frecuencia [Hz]",
    "Band": "Banda",
    "Broadband": "Banda ancha",
    "Clarity [dB]": "Claridad [dB]",
    _TIME_LABEL: "Tiempo [s]",
    "Level re steady state [dB]": "Nivel re estado estacionario [dB]",
    "ISO 3382 Schroeder decay curve": "Curva de caída de Schroeder ISO 3382",
    _SCHROEDER_DECAY_LABEL: "Decaimiento de Schroeder",
    "Log-magnitude envelope": "Envolvente de magnitud logarítmica",
    "Level re peak [dB]": "Nivel re pico [dB]",
    "Amplitude (norm.)": "Amplitud (norm.)",
    "ISO 18233 impulse response": "Respuesta al impulso ISO 18233",
    "Measured": "Medido",
    "Governing band": "Banda dominante",
    _OCTAVE_BAND_SPL_LABEL: "NPS por bandas de octava [dB]",
    "Reference RC-": "Referencia RC-",
    "Rumble tolerance (+5 dB)": "Tolerancia de retumbe (+5 dB)",
    "Hiss tolerance (+3 dB)": "Tolerancia de siseo (+3 dB)",
    _REVERBERATION_TIME_LABEL: "Tiempo de reverberación $T$ [s]",
    "EN 12354-6 reverberation time": "Tiempo de reverberación EN 12354-6",
    "Reverberation-time models — ": "Modelos de tiempo de reverberación — ",
    " dB per doubling": " dB por duplicación",
    " m (STI 0.50)": " m (STI 0,50)",
    " m (STI 0.20)": " m (STI 0,20)",
    "Distance from the sound source [m]": "Distancia a la fuente sonora [m]",
    "A-weighted SPL of speech [dB]": "NPS del habla ponderado A [dB]",
    "ISO 3382-3 spatial decay of speech": "Decaimiento espacial del habla ISO 3382-3",
    "Sample": "Muestra",
    "Amplitude": "Amplitud",
    "Magnitude [dB]": "Magnitud [dB]",
    "Magnitude spectrum (flat)": "Espectro de magnitud (plano)",
    "ISO 18233 exponential sine sweep": "Barrido sinusoidal exponencial ISO 18233",
    "Spectrogram (exponential frequency rise)": "Espectrograma (subida exponencial de frecuencia)",
    r"$1/r$ spreading envelope": r"Envolvente de propagación $1/r$",
    "Reflections": "Reflexiones",
    "Reflection order": "Orden de reflexión",
    "Direct sound": "Sonido directo",
    "Arrival time [ms]": "Tiempo de llegada [ms]",
    "Reflection level re direct [dB]": "Nivel de reflexión re directo [dB]",
    "Direct field": "Campo directo",
    "Reverberant field": "Campo reverberante",
    "Total": "Total",
    "Distance from source [m]": "Distancia a la fuente [m]",
    "Sound pressure level [dB]": "Nivel de presión acústica [dB]",
    "Shaped sweep (group-delay synthesis)": "Barrido conformado (síntesis del retardo de grupo)",
    "Welch spectrum of the sweep": "Espectro de Welch del barrido",
    "Synthesis target": "Objetivo de síntesis",
    "Sweep band": "Banda del barrido",
    "Level re in-band max [dB]": "Nivel re máximo en banda [dB]",
    "Crest factor": "Factor de cresta",
    # --- Rectangular room modes (Long Ch. 8) ---
    "axial": "axial",
    "tangential": "tangencial",
    "oblique": "oblicuo",
    "Mode kind": "Tipo de modo",
    "Modal density [modes/Hz]": "Densidad modal [modos/Hz]",
    "Modal density d$N$/d$f$": "Densidad modal d$N$/d$f$",
    "Room modes — {lx} × {ly} × {lz} m": "Modos de la sala — {lx} × {ly} × {lz} m",
    "Schroeder $f_\\mathrm{{s}}$ = {value} Hz": "$f_\\mathrm{{s}}$ de Schroeder = {value} Hz",
    # --- Restaurant crowd self-noise (Long Ch. 17) ---
    "Simultaneous talkers $N$": "Hablantes simultáneos $N$",
    "Self-generated noise level [dB]": "Nivel de ruido autogenerado [dB]",
    _ABSORPTION_AREA_LABEL: _ABSORPTION_AREA_LABEL,
    "Speech at {value} m": "Habla a {value} m",
    r"Communication limit ($L_\mathrm{SN}$ = −6 dB)": r"Límite de comunicación ($L_\mathrm{SN}$ = −6 dB)",
    "Crowd self-noise — $L_W$ = {lw} dB per talker": "Ruido autogenerado del público — $L_W$ = {lw} dB por hablante",
}


def _t(text: str, language: str = "en") -> str:
    """Localise a fixed string; English is returned verbatim (byte-identical)."""
    return _STRINGS.get(text, text) if language == "es" else text


def _localize_band_axes(ax: Axes, language: str) -> None:
    """Comma-localise the numeric y-axis of a categorical band plot.

    :func:`~phonometry._i18n.localize_axes` reformats only the automatic numeric
    axis and leaves the categorical band tick labels (a ``FuncFormatter`` on the
    linear position axis) untouched, so no label restore is needed. English is a
    no-op.
    """
    from .._i18n import localize_axes

    localize_axes(ax, language)


def plot_room_acoustics(
    result: RoomAcousticsResult,
    ax: Axes | None = None,
    language: str = "en",
    **kwargs: Any,
) -> Axes | np.ndarray:
    """Per-band decay times (EDT/T20/T30) and clarity (C50/C80).

    The first panel shows the three decay times as grouped bars per band,
    invalid bands (failing the ISO 3382 evaluation-range criterion) hatched
    and greyed; the second panel shows C50 and C80.  When ``ax`` is given
    the two series are drawn on that single axes (times only) so the plot
    can be composed.

    :param result: A :class:`~phonometry.room.acoustics.RoomAcousticsResult`.
    :param ax: Existing axes for a single-panel (decay-times only) plot, or
        ``None`` to create the full two-panel figure.
    :return: The axes, or an array of two axes for the default figure.
    """
    freq = result.frequency
    n = np.asarray(result.t30, dtype=np.float64).size
    if freq is None:
        labels = [_t("Broadband", language)] * n
        use_freq_axis = False
    else:
        centers = np.asarray(freq, dtype=np.float64)
        # Label the categorical band axis with the standard nominal octave /
        # one-third-octave centres (IEC 61260: 125, 250, 500, 1k, 2k, 4k), not
        # the exact base-ten filter centres (125.89..., 1.99526k...), so the
        # chart matches the nominal frequency table an ISO 3382 report prints.
        from ..filters.frequencies import (
            _infer_band_fraction,
            _nominal_freq_for_band,
        )

        fraction = _infer_band_fraction(centers)
        labels = [
            _format_freq(_nominal_freq_for_band(float(f), float(fraction)))
            for f in centers
        ]
        use_freq_axis = True

    positions = np.arange(n, dtype=np.float64)
    single = ax is not None
    if ax is not None:
        ax_times = ax
    else:
        axes = _new_axes_column(2, figsize=(7.5, 6.0), sharex=True)
        ax_times = cast("Axes", axes[0])

    _draw_decay_times(ax_times, positions, result, **kwargs)
    ax_times.set_ylabel(_t("Reverberation time [s]", language))
    ax_times.set_title(_t("ISO 3382 decay times and clarity", language))
    _band_axis(ax_times, labels, xlabel=None, language=language)
    ax_times.grid(True, axis="y", alpha=0.3)
    ax_times.legend(loc="best", fontsize="small")

    if single:
        if use_freq_axis:
            ax_times.set_xlabel(_t(_FREQUENCY_LABEL, language))
        _localize_band_axes(ax_times, language)
        return ax_times

    ax_clarity = cast("Axes", axes[1])
    ax_clarity.plot(
        positions,
        np.asarray(result.c50, dtype=np.float64),
        "o-",
        color=_C_TERTIARY,
        label="C50",
    )
    ax_clarity.plot(
        positions,
        np.asarray(result.c80, dtype=np.float64),
        "s--",
        color=_C_QUATERNARY,
        label="C80",
    )
    ax_clarity.set_ylabel(_t("Clarity [dB]", language))
    _band_axis(
        ax_clarity,
        labels,
        xlabel=_t(_FREQUENCY_LABEL if use_freq_axis else "Band", language),
        language=language,
    )
    ax_clarity.grid(True, alpha=0.3)
    ax_clarity.legend(loc="best", fontsize="small")
    _localize_band_axes(ax_times, language)
    _localize_band_axes(ax_clarity, language)
    return axes


def plot_decay_curve(
    result: DecayCurve,
    ax: Axes | None = None,
    fits: bool = True,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Schroeder decay curve with optional straight T-fit overlays.

    :param result: A :class:`~phonometry.room.acoustics.DecayCurve`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param fits: Overlay the EDT (0..-10 dB), T20 (-5..-25 dB) and T30
        (-5..-35 dB) straight-line fits computed from the curve's own data.
    :param kwargs: Forwarded to the decay-curve ``plot`` call.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    time = np.asarray(result.time, dtype=np.float64)
    level = np.asarray(result.level, dtype=np.float64)
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("label", _t(_SCHROEDER_DECAY_LABEL, language))
    ax.plot(time, level, **kwargs)

    if fits:
        for label, lo, hi, style in (
            ("EDT", 0.0, -10.0, "-"),
            ("T20", -5.0, -25.0, "--"),
            ("T30", -5.0, -35.0, "-."),
        ):
            fit = _fit_segment(time, level, lo, hi)
            if fit is not None:
                fit_label = f"ajuste {label}" if language == "es" else f"{label} fit"
                ax.plot(time, fit, style, lw=1, alpha=0.8, label=fit_label)

    ax.set_xlabel(_t(_TIME_LABEL, language))
    ax.set_ylabel(_t("Level re steady state [dB]", language))
    ax.set_ylim(top=3.0)
    ax.set_xlim(left=0.0, right=float(time[-1]) if time.size else None)
    band = result.band
    title = _t("ISO 3382 Schroeder decay curve", language)
    if band is not None:
        fb = _format_freq(float(band))
        title += f"  (banda {fb} Hz)" if language == "es" else f"  ({fb} Hz band)"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    localize_axes(ax, language)
    return ax


def plot_impulse_response(
    result: ImpulseResponseResult,
    ax: Axes | None = None,
    language: str = "en",
    **kwargs: Any,
) -> Axes | np.ndarray:
    """Impulse-response waveform and its log-magnitude / Schroeder decay.

    Two stacked panels: the (peak-normalised) time-domain waveform on top and,
    below it, the log-magnitude envelope in dB with the Schroeder
    backward-integrated energy-decay curve overlaid. With ``ax`` given, only
    the decay panel is drawn on it.

    :param result: An :class:`~phonometry.room.impulse_response.ImpulseResponseResult`.
    :param ax: Existing axes for the decay panel, or ``None`` for a fresh
        two-panel figure.
    :param kwargs: Forwarded to the waveform / envelope ``plot`` calls.
    :return: The decay-panel axes (``ax`` given) or the array of two axes.
    """
    h = np.asarray(result.ir, dtype=np.float64)
    n = h.shape[-1]
    if n == 0:
        msg = "impulse response is empty; nothing to plot."
        raise ValueError(msg)
    time, xlabel = _time_axis(n, result.fs, language=language)
    peak = float(np.max(np.abs(h)))
    tiny = np.finfo(np.float64).tiny
    norm = peak if peak > 0.0 else 1.0
    env_db = 20.0 * np.log10(np.maximum(np.abs(h), tiny) / norm)
    # Schroeder backward integration of the squared IR (broadband).
    energy = np.cumsum(h[::-1] ** 2)[::-1]
    total = float(energy[0]) if energy.size else 0.0
    edc_db = 10.0 * np.log10(np.maximum(energy, tiny) / (total if total > 0.0 else 1.0))

    color = kwargs.pop("color", _C_PRIMARY)

    from .._i18n import localize_axes

    title = f"{_t('ISO 18233 impulse response', language)} ({result.method})"

    def _decay(axd: Axes) -> None:
        axd.plot(
            time,
            env_db,
            color=_C_PRIMARY_LIGHT,
            lw=0.8,
            label=_t("Log-magnitude envelope", language),
        )
        axd.plot(
            time,
            edc_db,
            color=_C_REFERENCE,
            lw=1.8,
            label=_t(_SCHROEDER_DECAY_LABEL, language),
        )
        axd.set_xlabel(_t(xlabel, language))
        axd.set_ylabel(_t("Level re peak [dB]", language))
        axd.set_ylim(bottom=-80.0, top=5.0)
        axd.set_xlim(
            left=float(time[0]) if n else 0.0, right=float(time[-1]) if n else None
        )
        axd.grid(True, alpha=0.3)
        axd.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")
        localize_axes(axd, language)

    if ax is not None:
        _decay(ax)
        ax.set_title(title)
        return ax

    axes = _new_axes_column(2, sharex=True, figsize=(8.0, 6.0))
    axes[0].plot(time, h / norm, color=color, lw=0.8, **kwargs)
    axes[0].set_ylabel(_t("Amplitude (norm.)", language))
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)
    localize_axes(axes[0], language)
    _decay(axes[1])
    return axes


def plot_noise_criterion(
    result: NCResult, ax: Axes | None = None, language: str = "en", **kwargs: Any
) -> Axes:
    """Measured spectrum against the NC curve family (ANSI/ASA S12.2-2019).

    :param result: A :class:`~phonometry.room.noise_criteria.NCResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param kwargs: Forwarded to the measured-spectrum :meth:`plot`.
    :return: The axes.
    """
    from .._i18n import localize_axes
    from ..room.noise_criteria import NC_CURVES, NC_INDICES, OCTAVE_BANDS

    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    levels = np.asarray(result.levels, dtype=np.float64)
    for row, idx in zip(NC_CURVES, NC_INDICES, strict=True):
        ax.plot(OCTAVE_BANDS, row, color=_C_MUTED, lw=0.8, zorder=1)
        ax.annotate(
            f"{idx:.0f}",
            (OCTAVE_BANDS[-1], row[-1]),
            fontsize="x-small",
            color=_C_MUTED,
            va="center",
        )
    valid = ~np.isnan(levels)
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("label", _t("Measured", language))
    ax.plot(freqs[valid], levels[valid], "o-", zorder=3, **kwargs)
    # Nearest *valid* band rather than float equality against the stored
    # value; the marker sits on that band so its x and y stay paired. A
    # SIL-designated or below-family spectrum has no governing band (NaN),
    # so the marker is omitted.
    candidates = np.flatnonzero(valid)
    if candidates.size and np.isfinite(result.governing_frequency):
        governing = int(
            candidates[
                np.argmin(np.abs(freqs[candidates] - result.governing_frequency))
            ]
        )
        ax.plot(
            [freqs[governing]],
            [levels[governing]],
            "D",
            color=_C_REFERENCE,
            zorder=4,
            label=(
                f"{_t('Governing band', language)} "
                f"({_format_freq(result.governing_frequency)})"
            ),
        )
    _freq_axis(ax, OCTAVE_BANDS, language=language)
    ax.set_ylabel(_t(_OCTAVE_BAND_SPL_LABEL, language))
    ax.set_title(f"ANSI/ASA S12.2 {result.label}")
    ax.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")
    ax.grid(True, which="both", alpha=0.3)
    localize_axes(ax, language)
    return ax


def plot_room_criterion(
    result: RCResult, ax: Axes | None = None, language: str = "en", **kwargs: Any
) -> Axes:
    """Measured spectrum against the reference RC Mark II curve (Annex D).

    Shades the rumble tolerance (reference + 5 dB below 500 Hz) and the hiss
    tolerance (reference + 3 dB at and above 1000 Hz).

    :param result: A :class:`~phonometry.room.noise_criteria.RCResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param kwargs: Forwarded to the measured-spectrum :meth:`plot`.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    levels = np.asarray(result.levels, dtype=np.float64)
    reference = np.asarray(result.reference_curve, dtype=np.float64)
    valid = ~np.isnan(levels)

    ax.plot(
        freqs,
        reference,
        "s--",
        color=_C_MUTED,
        label=f"{_t('Reference RC-', language)}{result.rating}",
    )
    low = freqs <= _RC_RUMBLE_MAX_HZ
    high = freqs >= _RC_HISS_MIN_HZ
    ax.fill_between(
        freqs[low],
        reference[low],
        reference[low] + 5.0,
        color=_C_SECONDARY_LIGHT,
        alpha=0.35,
        label=_t("Rumble tolerance (+5 dB)", language),
    )
    ax.fill_between(
        freqs[high],
        reference[high],
        reference[high] + 3.0,
        color=_C_PRIMARY_LIGHT,
        alpha=0.45,
        label=_t("Hiss tolerance (+3 dB)", language),
    )
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("label", _t("Measured", language))
    ax.plot(freqs[valid], levels[valid], "o-", zorder=3, **kwargs)
    _freq_axis(ax, freqs, language=language)
    ax.set_ylabel(_t(_OCTAVE_BAND_SPL_LABEL, language))
    ax.set_title(f"ANSI/ASA S12.2 {result.label}")
    ax.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")
    ax.grid(True, which="both", alpha=0.3)
    localize_axes(ax, language)
    return ax


def plot_enclosed_space_absorption(
    result: ReverberationResult,
    ax: Axes | None = None,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Reverberation time over the octave bands (EN 12354-6).

    :param result: A :class:`~phonometry.room.enclosed_space_absorption.ReverberationResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param kwargs: Forwarded to the reverberation-time ``plot``.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    freq = np.asarray(result.frequencies, dtype=np.float64)
    rt = np.asarray(result.reverberation_time, dtype=np.float64)
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("marker", "o")
    ax.plot(freq, rt, **kwargs)
    _freq_axis(ax, freq, language=language)
    ax.set_ylabel(_t(_REVERBERATION_TIME_LABEL, language))
    ax.set_title(_t("EN 12354-6 reverberation time", language))
    ax.set_ylim(bottom=0.0)
    ax.grid(True, which="both", alpha=0.3)
    localize_axes(ax, language)
    return ax


def plot_reverberation_models(
    result: ReverberationModelResult,
    ax: Axes | None = None,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Reverberation time by five statistical models over the bands.

    Draws the Sabine, Eyring, Millington-Sette, Fitzroy and Arau-Puchades
    curves, with Arau-Puchades emphasised as the recommended model for a
    non-uniform absorption distribution.

    :param result: A
        :class:`~phonometry.room.reverberation_prediction.ReverberationModelResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param kwargs: Forwarded to every curve ``plot``.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    freq = np.asarray(result.frequencies, dtype=np.float64)
    styles = (
        ("Sabine", result.sabine, _C_SECONDARY, "s", 1.4),
        ("Eyring", result.eyring, _C_TERTIARY, "^", 1.4),
        ("Millington-Sette", result.millington_sette, _C_QUATERNARY, "v", 1.4),
        ("Fitzroy", result.fitzroy, _C_MUTED, "D", 1.4),
        ("Arau-Puchades", result.arau_puchades, _C_PRIMARY, "o", 2.4),
    )
    for label, curve, color, marker, lw in styles:
        # A copy per curve: `setdefault` mutates, so one before the loop would
        # give every model the first one's name.
        model_kwargs = dict(kwargs)
        model_kwargs.setdefault("label", label)
        ax.plot(
            freq,
            np.asarray(curve, dtype=np.float64),
            color=color,
            marker=marker,
            lw=lw,
            **model_kwargs,
        )
    _freq_axis(ax, freq, language=language)
    ax.set_ylabel(_t(_REVERBERATION_TIME_LABEL, language))
    ax.set_title(
        f"{_t('Reverberation-time models — ', language)}"
        f"$V$ = {format_number(result.volume, language, decimals=0)} m³, "
        f"$S$ = {format_number(result.surface_area, language, decimals=0)} m²"
    )
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, which="both", alpha=0.3)
    localize_axes(ax, language)
    return ax


def plot_open_plan(
    result: OpenPlanResult, ax: Axes | None = None, language: str = "en", **kwargs: Any
) -> Axes:
    """Spatial decay of speech with the distraction/privacy distances marked.

    Redraws the Clause 6.2 regression line ``Lp,A,S(r) = Lp,A,S,4m -
    D2,S log10(r/4)/log10 2`` over the 2 m to 16 m fitting range (extended to
    reach ``rP`` when it lies further out) and marks the distraction
    distance ``rD`` and the privacy distance ``rP`` from the STI regression.

    :param result: An :class:`~phonometry.room.open_plan.OpenPlanResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param kwargs: Forwarded to the decay-line ``plot`` call.
    :return: The axes.
    :raises ValueError: If the spatial-decay regression is undefined
        (``d2s`` / ``lp_as_4m`` are NaN).
    """
    if not (np.isfinite(result.d2s) and np.isfinite(result.lp_as_4m)):
        msg = (
            "plot() needs the spatial-decay regression; this result's d2s / "
            "lp_as_4m are NaN (fewer than two positions in the 2 m to 16 m "
            "range)."
        )
        raise ValueError(msg)
    ax = ax if ax is not None else _new_axes()
    import matplotlib.ticker as mticker

    from .._i18n import format_number, localize_axes

    r_max = 16.0
    for marker in (result.rd, result.rp):
        if np.isfinite(marker):
            r_max = max(r_max, 1.15 * marker)
    r = np.geomspace(2.0, r_max, 200)
    level = result.lp_as_4m - result.d2s * np.log2(r / 4.0)

    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault(
        "label",
        rf"$D_{{2,\mathrm{{S}}}}$ = {format_number(result.d2s, language, decimals=1)}"
        f"{_t(' dB per doubling', language)}",
    )
    ax.plot(r, level, **kwargs)
    ax.plot(
        [4.0],
        [result.lp_as_4m],
        "o",
        color=_C_PRIMARY,
        ms=7,
        label=rf"$L_{{p,\mathrm{{A,S}},4\,\mathrm{{m}}}}$ = "
        f"{format_number(result.lp_as_4m, language, decimals=1)} dB",
    )
    if np.isfinite(result.rd):
        ax.axvline(
            result.rd,
            color=_C_SECONDARY,
            ls="--",
            label=rf"$r_\mathrm{{D}}$ = {format_number(result.rd, language, decimals=1)}"
            f"{_t(' m (STI 0.50)', language)}",
        )
    if np.isfinite(result.rp):
        ax.axvline(
            result.rp,
            color=_C_REFERENCE,
            ls=":",
            label=rf"$r_\mathrm{{P}}$ = {format_number(result.rp, language, decimals=1)}"
            f"{_t(' m (STI 0.20)', language)}",
        )

    ax.set_xscale("log", base=2)
    ticks = [float(2**k) for k in range(1, int(np.ceil(np.log2(r_max))) + 1)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel(_t("Distance from the sound source [m]", language))
    ax.set_ylabel(_t("A-weighted SPL of speech [dB]", language))
    ax.set_title(_t("ISO 3382-3 spatial decay of speech", language))
    ax.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")
    ax.grid(True, which="both", alpha=0.3)
    localize_axes(ax, language)
    return ax


def plot_excitation(
    signal: ArrayLike,
    fs: int,
    *,
    kind: str = "sweep",
    ax: Axes | None = None,
    language: str = "en",
    **kwargs: Any,
) -> Axes | np.ndarray:
    """Plot an ISO 18233 excitation signal (sweep or MLS).

    A documented helper for the raw arrays returned by
    :func:`~phonometry.room.sweep_signal` and
    :func:`~phonometry.room.mls_signal`, which stay plain
    :class:`numpy.ndarray` (they are meant for playback). For a swept sine the
    waveform and its spectrogram are drawn; for an MLS the first samples of the
    bipolar sequence and its (flat) magnitude spectrum.

    :param signal: The excitation samples (1D array-like).
    :param fs: Sample rate in Hz (for the time and frequency axes).
    :param kind: ``"sweep"`` (default) or ``"mls"``.
    :param ax: Existing axes for the top (time-domain) panel, or ``None`` for a
        fresh two-panel figure.
    :param kwargs: Forwarded to the time-domain ``plot`` call.
    :return: The time-domain axes (``ax`` given) or the array of two axes.
    :raises ValueError: for an unknown ``kind``, or an empty signal.
    """
    from .._i18n import check_language, localize_axes

    check_language(language)
    # Pinned like ``language`` beside it: the branch below tests ``== "mls"``
    # and sent every other tag to the sweep drawing, so the typo "msl" for an
    # actual MLS array was published under the title "ISO 18233 exponential
    # sine sweep" beside a spectrogram announcing an exponential frequency
    # rise, with no warning.
    require_choice(kind, "kind", _EXCITATION_KINDS)
    x = np.asarray(signal, dtype=np.float64)
    n = x.shape[-1]
    if n == 0:
        msg = "excitation signal is empty; nothing to plot."
        raise ValueError(msg)
    t = np.arange(n) / float(fs)
    color = kwargs.pop("color", _C_PRIMARY)

    two_panel = ax is None
    if two_panel:
        axes = _new_axes_column(2, figsize=(8.0, 6.0))
        ax_time = cast("Axes", axes[0])
    else:
        ax_time = cast("Axes", ax)

    if kind == "mls":
        show = min(n, 120)
        ax_time.step(np.arange(show), x[:show], where="mid", color=color, **kwargs)
        ax_time.set_xlabel(_t("Sample", language))
        ax_time.set_ylabel(_t("Amplitude", language))
        ax_time.set_ylim(-1.4, 1.4)
        if language == "es":
            ax_time.set_title(
                f"Excitación MLS ISO 18233 (primeras {show} de {n} muestras)"
            )
        else:
            ax_time.set_title(f"ISO 18233 MLS excitation (first {show} of {n} samples)")
        ax_time.grid(True, alpha=0.3)
        localize_axes(ax_time, language)
        if not two_panel:
            return ax_time
        spec = np.abs(np.fft.rfft(x))
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        ax_f = axes[1]
        ac = spec[1:]
        denom = float(np.median(ac)) if ac.size else 1.0
        ax_f.semilogx(
            freqs[1:],
            20.0 * np.log10(np.maximum(ac, 1e-10) / (denom if denom > 0.0 else 1.0)),
            color=_C_REFERENCE,
            lw=0.8,
        )
        ax_f.set_xlabel(_t(_FREQUENCY_LABEL, language))
        ax_f.set_ylabel(_t("Magnitude [dB]", language))
        ax_f.set_title(_t("Magnitude spectrum (flat)", language))
        ax_f.grid(True, which="both", alpha=0.3)
        format_frequency_axis(ax_f, float(freqs[1]), float(freqs[-1]))
        localize_axes(ax_f, language)
        return axes

    # Swept sine.
    ax_time.plot(t, x, color=color, lw=0.6, **kwargs)
    ax_time.set_xlabel(_t(_TIME_LABEL, language))
    ax_time.set_ylabel(_t("Amplitude", language))
    ax_time.set_title(_t("ISO 18233 exponential sine sweep", language))
    ax_time.grid(True, alpha=0.3)
    localize_axes(ax_time, language)
    if not two_panel:
        return ax_time
    ax_s = axes[1]
    nperseg = min(n, max(256, min(2048, n // 16)))
    ax_s.specgram(x, NFFT=nperseg, Fs=fs, noverlap=nperseg // 2, cmap="magma")
    ax_s.set_xlabel(_t(_TIME_LABEL, language))
    ax_s.set_ylabel(_t(_FREQUENCY_LABEL, language))
    ax_s.set_title(_t("Spectrogram (exponential frequency rise)", language))
    localize_axes(ax_s, language)
    return axes


def plot_image_source_reflectogram(
    result: ImageSourceResult,
    ax: Axes | None = None,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Reflectogram of a synthetic image-source room impulse response.

    Stems each image's amplitude in dB relative to the direct sound against its
    arrival time, coloured by reflection order (the direct sound at order 0 in
    the primary colour), with the ``1 / r`` free-field spreading envelope
    overlaid. For a per-band result the first band is drawn.

    :param result: An :class:`~phonometry.room.image_source.ImageSourceResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param kwargs: Forwarded to the stem ``markerline`` styling.
    :return: The axes.
    """
    from .._i18n import decimal_comma, localize_axes

    ax = ax if ax is not None else _new_axes()
    times = np.asarray(result.times, dtype=np.float64)
    amp = np.asarray(result.amplitudes, dtype=np.float64)
    if amp.ndim == 2:  # noqa: PLR2004
        amp = amp[0]
    orders = np.asarray(result.orders, dtype=np.int_)
    tiny = np.finfo(np.float64).tiny
    direct = float(np.max(np.abs(amp))) or 1.0
    level = 20.0 * np.log10(np.maximum(np.abs(amp), tiny) / direct)

    ms = 1e3 * times
    order0 = orders == 0
    # 1/r free-field envelope (direct sound at its own distance is 0 dB).
    dist = np.asarray(result.distances, dtype=np.float64)
    d0 = float(dist[int(np.argmin(dist))])
    envelope = 20.0 * np.log10(np.maximum(d0 / dist, tiny))
    order_sort = np.argsort(ms)
    ax.plot(
        ms[order_sort],
        envelope[order_sort],
        color=_C_MUTED,
        lw=1.0,
        ls="--",
        label=_t(r"$1/r$ spreading envelope", language),
        zorder=1,
    )

    # Reflections coloured by order (higher orders fade toward grey). A
    # max_order=0 result has no reflections, so guard the scatter/colorbar
    # (an empty scatter has undefined colour limits).
    ref_mask = ~order0
    if np.any(ref_mask):
        sc = ax.scatter(
            ms[ref_mask],
            level[ref_mask],
            c=orders[ref_mask],
            cmap="viridis",
            s=14,
            zorder=3,
            label=_t("Reflections", language),
        )
        ax.vlines(
            ms[ref_mask],
            -120.0,
            level[ref_mask],
            color=_C_EDGE,
            lw=0.4,
            alpha=0.4,
            zorder=2,
        )
        cbar = ax.figure.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(_t("Reflection order", language))
    ax.vlines(ms[order0], -120.0, level[order0], color=_C_PRIMARY, lw=1.6, zorder=4)
    kwargs.setdefault("label", _t("Direct sound", language))
    ax.plot(ms[order0], level[order0], "o", color=_C_PRIMARY, ms=7, zorder=5, **kwargs)

    lx, ly, lz = result.dimensions
    # A bare ``:g`` never reaches the locale pass (``localize_axes`` reformats
    # tick labels only), so a 6.2 × 4.1 × 2.8 m room would publish English
    # decimal points inside the Spanish title. English is unchanged.
    dims = "×".join(decimal_comma(f"{d:g}", language) for d in (lx, ly, lz))
    finite = level[np.isfinite(level)]
    ax.set_ylim(
        bottom=max(-80.0, float(finite.min()) - 3.0) if finite.size else -80.0, top=5.0
    )
    ax.set_xlim(left=0.0, right=float(ms.max()) if ms.size else None)
    ax.set_xlabel(_t("Arrival time [ms]", language))
    ax.set_ylabel(_t("Reflection level re direct [dB]", language))
    if language == "es":
        ax.set_title(
            f"Reflectograma de fuentes imagen — sala de "
            f"{dims} m, orden ≤ {result.max_order}"
        )
    else:
        ax.set_title(
            f"Image-source reflectogram — {dims} m room, order ≤ {result.max_order}"
        )
    ax.grid(True, alpha=0.3)
    ax.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")
    localize_axes(ax, language)
    return ax


def plot_steady_field(
    result: SteadyFieldResult,
    ax: Axes | None = None,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Steady-state SPL against distance: direct, reverberant and total fields.

    Draws the total level (Bies Equation (6.43)) with its direct ``1/r^2`` and
    constant reverberant components, and marks the critical distance ``rc``
    where the two cross.

    :param result: A :class:`~phonometry.room.steady_field.SteadyFieldResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param kwargs: Forwarded to the total-level ``plot`` call.
    :return: The axes.
    """
    import matplotlib.ticker as mticker

    from .._i18n import decimal_comma, format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    r = np.asarray(result.distances, dtype=np.float64)
    ax.plot(
        r,
        np.asarray(result.direct, dtype=np.float64),
        color=_C_SECONDARY,
        ls="--",
        lw=1.4,
        label=_t("Direct field", language),
    )
    ax.plot(
        r,
        np.asarray(result.reverberant, dtype=np.float64),
        color=_C_TERTIARY,
        ls=":",
        lw=1.4,
        label=_t("Reverberant field", language),
    )
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("lw", 2.4)
    kwargs.setdefault("label", _t("Total", language))
    ax.plot(r, np.asarray(result.total, dtype=np.float64), **kwargs)
    ax.axvline(
        result.critical_distance,
        color=_C_REFERENCE,
        ls="-.",
        lw=1.2,
        label=rf"$r_\mathrm{{c}}$ = "
        f"{format_number(result.critical_distance, language, decimals=2)} m",
    )

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel(_t("Distance from source [m]", language))
    ax.set_ylabel(_t("Sound pressure level [dB]", language))
    # ``:g`` bypasses the locale pass, so a 92.5 dB source power or a Q of 2.5
    # would keep an English decimal point in the Spanish title beside the room
    # constant, which is localised. English is unchanged.
    lw = decimal_comma(f"{result.sound_power_level:g}", language)
    q = decimal_comma(f"{result.directivity:g}", language)
    if language == "es":
        ax.set_title(
            f"Campo estacionario de la sala — $L_W$ = {lw} dB, "
            f"$R$ = {format_number(result.room_constant, language, decimals=0)} m², "
            f"$Q$ = {q}"
        )
    else:
        ax.set_title(
            f"Steady-state room field — $L_W$ = {lw} dB, "
            f"$R$ = {result.room_constant:.0f} m², $Q$ = {q}"
        )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")
    localize_axes(ax, language)
    return ax


def plot_shaped_sweep(
    result: ShapedSweepResult,
    ax: Axes | None = None,
    language: str = "en",
    **kwargs: Any,
) -> Axes | np.ndarray:
    """Shaped-sweep waveform and its Welch spectrum against the target.

    Two stacked panels: the time-domain waveform on top and, below it, the
    sweep's Welch magnitude spectrum overlaid on the synthesis target (both
    in dB re their in-band maximum, so the match is read directly). With
    ``ax`` given, only the spectrum panel is drawn on it.

    :param result: A :class:`~phonometry.room.impulse_response.ShapedSweepResult`.
    :param ax: Existing axes for the spectrum panel, or ``None`` for a
        fresh two-panel figure.
    :param kwargs: Forwarded to the waveform ``plot`` call.
    :return: The spectrum axes (``ax`` given) or the array of two axes.
    """
    from scipy import signal as sp_signal

    from .._i18n import format_number, localize_axes

    x = np.asarray(result.signal, dtype=np.float64)
    fs = float(result.fs)
    f1, f2 = result.f_range
    time, xlabel = _time_axis(x.size, int(fs), language=language)
    color = kwargs.pop("color", _C_PRIMARY)

    # Welch magnitude of the sweep and the synthesis target, both in dB
    # re their in-band maximum (power dB and magnitude dB share the shape).
    # 75 % overlap: at the default 50 % the squared Hann windows do not sum
    # to a constant, and a sweep maps that temporal power ripple onto a
    # ~2 dB frequency ripple.
    nperseg = min(4096, x.size)
    freqs_w, psd = sp_signal.welch(x, fs=fs, nperseg=nperseg, noverlap=3 * nperseg // 4)
    tiny = np.finfo(np.float64).tiny
    band_w = (freqs_w >= f1) & (freqs_w <= f2)
    welch_db = 10.0 * np.log10(np.maximum(psd, tiny))
    # The Welch grid is resolved independently of the synthesis grid, so a
    # narrow band on a short signal can leave no Welch bin inside [f1, f2]
    # (unlike band_g, which synthesis validates). Fall back to the overall
    # positive-frequency maximum so the reference stays finite and the plot
    # still renders.
    ref_w = welch_db[band_w] if np.any(band_w) else welch_db[freqs_w > 0.0]
    welch_db -= float(np.max(ref_w))
    mag = np.asarray(result.magnitude, dtype=np.float64)
    grid = np.asarray(result.frequencies, dtype=np.float64)
    band_g = (grid >= f1) & (grid <= f2)
    target_db = 20.0 * np.log10(np.maximum(mag, tiny))
    target_db -= float(np.max(target_db[band_g]))

    title = _t("Shaped sweep (group-delay synthesis)", language)
    crest = format_number(result.crest_factor_db, language, decimals=1)

    def _spectrum(axs: Axes) -> None:
        pos = freqs_w > 0.0
        axs.semilogx(
            freqs_w[pos],
            welch_db[pos],
            color=color,
            lw=1.2,
            label=_t("Welch spectrum of the sweep", language),
        )
        posg = grid > 0.0
        axs.semilogx(
            grid[posg],
            target_db[posg],
            color=_C_REFERENCE,
            lw=1.4,
            ls="--",
            label=_t("Synthesis target", language),
        )
        axs.axvspan(f1, f2, color=color, alpha=0.08, label=_t("Sweep band", language))
        axs.set_xlabel(_t(_FREQUENCY_LABEL, language))
        axs.set_ylabel(_t("Level re in-band max [dB]", language))
        axs.set_ylim(bottom=-60.0, top=8.0)
        axs.grid(True, which="both", alpha=0.3)
        axs.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")
        format_frequency_axis(
            axs, max(f1 / 4.0, float(freqs_w[1])), min(2.0 * f2, fs / 2.0)
        )
        axs.set_xlim(max(f1 / 4.0, float(freqs_w[1])), min(2.0 * f2, fs / 2.0))
        localize_axes(axs, language)

    if ax is not None:
        _spectrum(ax)
        ax.set_title(title)
        return ax

    axes = _new_axes_column(2, figsize=(8.0, 6.0))
    axes[0].plot(time, x, color=color, lw=0.6, **kwargs)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(_t("Amplitude", language))
    axes[0].set_title(f"{title} — {_t('Crest factor', language)} {crest} dB")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(float(time[0]), float(time[-1]))
    localize_axes(axes[0], language)
    _spectrum(axes[1])
    return axes


def plot_room_modes(
    result: RoomModesResult, ax: Axes | None = None, language: str = "en", **kwargs: Any
) -> Axes | np.ndarray:
    """Mode ladder by kind and modal density of a rectangular room.

    The upper panel stems every enumerated eigenfrequency (Long Equation
    (8.43)) at a height set by its kind, so the axial, tangential and oblique
    families read apart at a glance; the lower panel draws the smooth modal
    density of Equation (8.46). When the result carries a Schroeder frequency
    it is marked on both panels, separating the modal regime on its left from
    the statistical regime on its right.

    :param result: A :class:`~phonometry.room.modes.RoomModesResult`.
    :param ax: Existing axes for the mode ladder alone, or ``None`` to create
        the two-panel figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the ``Axes.vlines`` calls of the ladder.
    :return: The two axes as an array, or the single axes when ``ax`` is given.
    """
    from .._i18n import format_number, localize_axes

    single = ax is not None
    axes = (
        np.array([ax], dtype=object)
        if ax is not None
        else _new_axes_column(2, sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    )
    ladder = cast("Axes", axes[0])

    freqs = np.asarray(result.frequencies, dtype=np.float64)
    kinds = np.asarray(result.kinds)
    # One raster row per family: axial on top, then tangential, then oblique.
    rows = {"axial": 3.0, "tangential": 2.0, "oblique": 1.0}
    colors = {"axial": _C_PRIMARY, "tangential": _C_SECONDARY, "oblique": _C_TERTIARY}
    for kind in ("axial", "tangential", "oblique"):
        mask = kinds == kind
        if not np.any(mask):
            continue
        row = rows[kind]
        kind_kwargs = dict(kwargs)
        kind_kwargs.setdefault("label", _t(kind, language))
        ladder.vlines(
            freqs[mask],
            row - 0.35,
            row + 0.35,
            color=colors[kind],
            lw=1.2,
            **kind_kwargs,
        )
    ladder.set_ylim(0.4, 4.1)
    order = ("oblique", "tangential", "axial")
    ladder.set_yticks([rows[k] for k in order])
    ladder.set_yticklabels([_t(k, language) for k in order], fontsize=8)
    ladder.set_ylabel(_t("Mode kind", language))
    lx, ly, lz = result.dimensions
    ladder.set_title(
        _t("Room modes — {lx} × {ly} × {lz} m", language).format(
            lx=format_number(lx, language, decimals=1, trim=True),
            ly=format_number(ly, language, decimals=1, trim=True),
            lz=format_number(lz, language, decimals=1, trim=True),
        )
    )
    ladder.grid(True, axis="x", alpha=0.3)

    fs = result.schroeder_frequency
    if fs is not None and np.isfinite(fs):
        label = _t("Schroeder $f_\\mathrm{{s}}$ = {value} Hz", language).format(
            value=format_number(fs, language, decimals=0)
        )
        for panel in axes:
            panel.axvline(fs, color=_C_REFERENCE, ls="-.", lw=1.2, label=label)
    ladder.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small", ncol=2)

    if single:
        ladder.set_xlabel(_t(_FREQUENCY_LABEL, language))
    else:
        density_axes = axes[1]
        grid = np.linspace(0.0, result.max_frequency, 200)
        density = np.asarray(result.density(grid), dtype=np.float64)
        density_axes.plot(
            grid,
            density,
            color=_C_QUATERNARY,
            lw=1.8,
            label=_t("Modal density d$N$/d$f$", language),
        )
        density_axes.set_ylabel(_t("Modal density [modes/Hz]", language))
        density_axes.set_xlabel(_t(_FREQUENCY_LABEL, language))
        density_axes.grid(True, alpha=0.3)
        density_axes.legend(loc="upper left", fontsize="small")
        localize_axes(density_axes, language)

    localize_axes(ladder, language)
    return axes[0] if single else axes


def plot_crowd_noise(
    result: CrowdNoiseResult,
    ax: Axes | None = None,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Self-generated crowd noise against occupancy, one curve per absorption.

    Draws Long Equation (17.51) over the occupancy axis for every absorption
    area of the result, with the direct-field speech level of Equation (17.50)
    at the listener across the table and the noise level at which the
    speech-to-noise ratio falls to the -6 dB communication limit of
    Equation (17.53).

    :param result: A :class:`~phonometry.room.crowd_noise.CrowdNoiseResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to each ``Axes.plot`` call.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    n = np.asarray(result.talkers, dtype=np.float64)
    levels = np.asarray(result.levels, dtype=np.float64)
    areas = np.asarray(result.absorption_areas, dtype=np.float64)
    palette = (_C_PRIMARY, _C_SECONDARY, _C_TERTIARY, _C_QUATERNARY, _C_MUTED)
    for row, (area, curve) in enumerate(zip(areas, levels, strict=True)):
        area_kwargs = dict(kwargs)
        area_kwargs.setdefault(
            "label",
            _t(_ABSORPTION_AREA_LABEL, language).format(
                value=format_number(float(area), language, decimals=0)
            ),
        )
        ax.plot(
            n,
            curve,
            color=palette[row % len(palette)],
            lw=1.8,
            **area_kwargs,
        )
    ax.axhline(
        result.signal_level,
        color=_C_EDGE,
        ls="--",
        lw=1.3,
        label=_t("Speech at {value} m", language).format(
            value=format_number(result.distance, language, decimals=1, trim=True)
        ),
    )
    ax.axhline(
        result.communication_level,
        color=_C_REFERENCE,
        ls=":",
        lw=1.4,
        label=_t(r"Communication limit ($L_\mathrm{SN}$ = −6 dB)", language),
    )
    import matplotlib.ticker as mticker

    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_xlabel(_t("Simultaneous talkers $N$", language))
    ax.set_ylabel(_t("Self-generated noise level [dB]", language))
    ax.set_title(
        _t("Crowd self-noise — $L_W$ = {lw} dB per talker", language).format(
            lw=format_number(result.sound_power_level, language, decimals=0)
        )
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize="small")
    localize_axes(ax, language)
    return ax
