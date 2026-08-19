#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Plot renderers for the io domain (lazy imports from result .plot())."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..io._signal import Signal

from .common import _C_PRIMARY, _LEGEND_UPPER_RIGHT, _new_axes

#: Spanish translations of the fixed strings rendered here, keyed by their
#: verbatim English text; ``_t`` returns the English key unchanged for any
#: other language, so English output is byte-for-byte unaffected.
_TIME_LABEL = "Time [s]"
_STRINGS: dict[str, str] = {
    _TIME_LABEL: "Tiempo [s]",
    "Sound pressure [Pa]": "Presión sonora [Pa]",
    "Amplitude [FS]": "Amplitud [FS]",
    "Calibrated waveform": "Forma de onda calibrada",
    "Waveform": "Forma de onda",
    "Channel {n}": "Canal {n}",
}


def _t(text: str, language: str = "en", **fmt: Any) -> str:
    """Localise a fixed string; English is returned verbatim."""
    s = _STRINGS.get(text, text) if language == "es" else text
    return s.format(**fmt) if fmt else s


def plot_signal(
    result: Signal, ax: Axes | None = None, *,
    language: str = "en", scale: str = "linear", **kwargs: Any
) -> Axes:
    """Waveform of a :class:`~phonometry.io.Signal`, calibrated when it can be.

    With a ``calibration_factor`` on the signal the samples are converted to
    pascals before drawing, so the figure shows the physical waveform the
    microphone saw; without one the digital full-scale units are drawn as
    they are, and the axis label says so rather than implying pressure.
    Each channel is one line, labelled from ``channel_labels`` when the
    file provided them.

    :param result: A :class:`~phonometry.io.Signal`.
    :param ax: Existing axes to draw on, or ``None`` for a fresh figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param scale: ``"linear"`` (default) draws the waveform; ``"db"`` draws
        its level against 20 uPa, which is the trace a sound level meter
        shows and needs a calibrated Signal to mean anything.
    :param kwargs: Forwarded to every channel's ``plot`` call.
    :return: The axes drawn on.
    """
    from .._i18n import localize_axes

    factor = result.calibration_factor
    calibrated = factor is not None
    y = result.data if factor is None else result.data * factor
    t = np.arange(result.n_samples) / float(result.fs)

    if scale == "db":
        if not calibrated:
            raise ValueError(
                "scale='db' needs a calibrated Signal: without a "
                "calibration_factor the samples are digital full-scale units "
                "and there is no reference to count decibels from"
            )
        # Deferred, not tidiness: signals.levels imports io, so reaching for
        # the reference pressure at module level here is an import cycle that
        # takes down `import phonometry` outright.
        from ..signals.levels import _REF_PRESSURE

        with np.errstate(divide="ignore"):
            y = 20.0 * np.log10(np.abs(y) / _REF_PRESSURE)
    elif scale != "linear":
        raise ValueError(f"scale must be 'linear' or 'db'; got {scale!r}")

    new_figure = ax is None
    axw = _new_axes() if ax is None else ax
    kwargs.setdefault("lw", 0.9)
    if result.n_channels == 1:
        kwargs.setdefault("color", _C_PRIMARY)
        axw.plot(t, y[0], **kwargs)
    else:
        for index, channel in enumerate(y):
            label = (
                result.channel_labels[index]
                if result.channel_labels is not None
                else _t("Channel {n}", language, n=index + 1)
            )
            axw.plot(t, channel, label=label, **kwargs)
        axw.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")
    axw.set_xlabel(_t(_TIME_LABEL, language))
    if scale == "db":
        axw.set_ylabel(_t("Sound pressure level [dB re 20 uPa]", language))
    else:
        axw.set_ylabel(_t(
            "Sound pressure [Pa]" if calibrated else "Amplitude [FS]", language
        ))
    axw.grid(True, alpha=0.3)
    if new_figure:
        axw.set_title(_t(
            "Calibrated waveform" if calibrated else "Waveform", language
        ))
    localize_axes(axw, language)
    return axw
