#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Figures of merit of a spectral-analysis taper (Harris 1978).

The window is the one choice every Fourier estimator forces and the one
the estimators of :mod:`phonometry.signals.spectra` leave open in their
``window`` parameter. Harris (1978, *On the use of windows for harmonic
analysis with the discrete Fourier transform*) turned that choice into a
table of numbers: equivalent noise bandwidth, coherent gain, scalloping
loss, worst-case processing loss, highest sidelobe level and the -3 dB
main-lobe width - the trade-off between resolution, leakage and
amplitude accuracy, read rather than guessed.

:func:`window_metrics` computes those figures for any taper
:func:`scipy.signal.get_window` accepts, sampling it DFT-even (periodic)
exactly as the Welch estimators of
:mod:`phonometry.signals.spectra` and the multitaper estimator of
:mod:`phonometry.signals.multitaper` apply it, so the numbers describe
the window as this library actually uses it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .._internal.validation import (
    require_axis_count,
    require_equal_counts,
    require_ranks,
)
from .spectra import _positive

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

__all__ = [
    "WindowMetricsResult",
    "window_metrics",
]

#: Oversampling factor of the window spectrum (samples per DFT bin).
_WINDOW_OVERSAMPLE = 256

#: Half-power level in decibels, 10·log10(1/2) ≈ -3.0103: the threshold
#: the two-sided -3 dB main-lobe width is measured at.
_HALF_POWER_DB = -3.0103

#: Minimum window length accepted by :func:`window_metrics`, in samples.
_MIN_WINDOW_LENGTH = 16


@dataclass(frozen=True)
class WindowMetricsResult:
    r"""Figures of merit of a taper (Harris 1978), DFT-even sampling.

    Losses are positive dB (how much is lost), sidelobe levels negative dB
    (relative to the main lobe), bandwidths in DFT bins (multiply by
    ``fs/n`` for Hz), matching the conventions of Harris' Table 1. The
    window is sampled DFT-even (periodic), exactly as
    :func:`scipy.signal.welch` and the estimators of
    :mod:`phonometry.signals.spectra` use it.

    :ivar window: The window specification as given (any name or
        ``(name, param)`` tuple :func:`scipy.signal.get_window` accepts).
    :ivar n: Window length, in samples.
    :ivar taps: The window samples ``w[m]`` (DFT-even).
    :ivar coherent_gain: Normalized DC gain :math:`\sum w / n` (1 for
        rectangular);
        the amplitude a bin-centered tone is scaled by before correction.
    :ivar enbw_bins: Equivalent noise bandwidth
        :math:`n \sum w^2 / \left( \sum w \right)^2`, in bins:
        the width of the ideal rectangular filter that would pass the same
        white-noise power (1 rectangular, 1.5 Hann, 1987/1458 Hamming).
    :ivar scalloping_loss_db: Attenuation of a tone midway between two
        bins, :math:`-20 \log_{10} \lvert W(1/2)/W(0) \rvert`, in dB
        (positive).
    :ivar worst_case_processing_loss_db: Scalloping loss plus the ENBW
        processing loss :math:`10 \log_{10}(\mathrm{ENBW})`, in dB: the
        worst-case reduction
        in output signal-to-noise ratio for a tone in white noise.
    :ivar highest_sidelobe_db: Level of the highest sidelobe relative to
        the main lobe, in dB (negative; -13.3 rectangular, -31.5 Hann).
    :ivar mainlobe_width_3db_bins: Two-sided -3 dB width of the main
        lobe, in bins.
    """

    window: str | tuple[Any, ...]
    n: int
    taps: NDArray[np.float64]
    coherent_gain: float
    enbw_bins: float
    scalloping_loss_db: float
    worst_case_processing_loss_db: float
    highest_sidelobe_db: float
    mainlobe_width_3db_bins: float

    def __post_init__(self) -> None:
        """Reject taps that disagree with the declared window length.

        Every figure of merit here was computed from ``n`` samples, and the
        waveform panel of :meth:`plot` draws :attr:`taps` against a sample
        index counted out of the declared :attr:`n` rather than out of the
        array beside it, so taps of another length are a different window
        than the one the metrics rate. Matplotlib does refuse to draw the
        two against each other, but from inside the renderer and about
        first dimensions, naming neither the field nor the result; the
        declared count is the authority here, and checking it at
        construction says which one moved.

        :raises ValueError: if the taps span a different number of samples
            than the declared length.
        """
        require_ranks(self, taps=1)
        owner = type(self).__name__
        require_equal_counts(
            owner,
            {
                "n": self.n,
                "taps": require_axis_count(
                    self.taps, owner, "taps", "sample", rank=None
                ),
            },
            "sample",
        )

    def enbw_hz(self, fs: float) -> float:
        """Equivalent noise bandwidth in Hz for a sample rate ``fs``.

        :param fs: Sample rate, in Hz.
        :return: ``enbw_bins·fs/n``, in Hz.
        """
        return self.enbw_bins * _positive(fs, "fs") / self.n

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes | NDArray[Any]:
        """Plot the window shape and its spectrum with the metrics marked.

        :param language: Label language, ``"en"`` (default) or ``"es"``.
        """
        from .._i18n import check_language
        from .._plot.signals import plot_window_metrics

        check_language(language)
        return plot_window_metrics(self, ax=ax, language=language, **kwargs)


def _window_spectrum_db(w: NDArray[np.float64], oversample: int) -> NDArray[np.float64]:
    """Magnitude of the window spectrum re its peak, in dB, oversampled.

    Zero-padding the DFT by ``oversample`` samples the underlying
    Dirichlet-kernel spectrum every ``1/oversample`` bins, so bin
    ``k/oversample`` of the result is exactly ``W(k/oversample)``.
    """
    spectrum = np.abs(np.fft.rfft(w, n=w.size * oversample))
    with np.errstate(divide="ignore"):
        level = 20.0 * np.log10(spectrum / spectrum[0])
    return np.asarray(level, dtype=np.float64)


def _mainlobe_edge(level_db: NDArray[np.float64]) -> int:
    """Index of the first local minimum (the edge of the main lobe)."""
    falling = np.diff(level_db) < 0.0
    edges = np.flatnonzero(~falling[1:] & falling[:-1]) + 1
    if edges.size == 0:  # pragma: no cover - every proper taper has a null
        msg = "The window spectrum has no sidelobe structure."
        raise ValueError(msg)
    return int(edges[0])


def _width_3db_bins(level_db: NDArray[np.float64], oversample: int) -> float:
    """Two-sided -3 dB main-lobe width in bins (linear dB interpolation)."""
    below = np.flatnonzero(level_db <= _HALF_POWER_DB)
    i = int(below[0])
    d0, d1 = float(level_db[i - 1]), float(level_db[i])
    frac = (_HALF_POWER_DB - d0) / (d1 - d0)
    return 2.0 * (i - 1 + frac) / oversample


def window_metrics(
    window: str | tuple[Any, ...],
    n: int = 1024,
) -> WindowMetricsResult:
    """Figures of merit of a spectral-analysis taper (Harris 1978).

    Computes the numbers behind the window trade-off for any taper the
    ``window`` parameter of the :mod:`phonometry.signals.spectra`
    estimators accepts: the
    equivalent noise bandwidth and coherent gain (closed forms of the
    samples, machine-exact), and the scalloping loss, worst-case
    processing loss, highest sidelobe level and -3 dB main-lobe width
    (measured on the spectrum of the sampled window, oversampled by
    zero-padding). The window is sampled DFT-even (periodic), exactly as
    the Welch estimators apply it.

    :param window: Window name or ``(name, param)`` tuple, anything
        :func:`scipy.signal.get_window` accepts (e.g. ``'hann'``,
        ``('kaiser', 8.6)``, ``('tukey', 0.5)``).
    :param n: Window length, in samples (at least 16).
    :return: A :class:`WindowMetricsResult`.
    :raises ValueError: If the inputs or parameters are invalid.
    """
    from scipy import signal as sp_signal

    n_v = int(n)
    if n_v < _MIN_WINDOW_LENGTH:
        msg = f"'n' must be at least {_MIN_WINDOW_LENGTH} samples."
        raise ValueError(msg)
    try:
        w = np.asarray(
            sp_signal.get_window(window, n_v, fftbins=True), dtype=np.float64
        )
    except (ValueError, TypeError) as exc:
        msg = f"Unknown window specification: {window!r}"
        raise ValueError(msg) from exc
    if np.any(~np.isfinite(w)) or float(np.sum(w)) <= 0.0:
        msg = f"Degenerate window: {window!r}"
        raise ValueError(msg)

    wsum = float(np.sum(w))
    coherent_gain = wsum / n_v
    enbw = n_v * float(np.sum(w * w)) / (wsum * wsum)

    level_db = _window_spectrum_db(w, _WINDOW_OVERSAMPLE)
    # Bin k/oversample of the padded DFT is exactly W(k/oversample), so
    # index oversample/2 evaluates the spectrum at the half-bin offset.
    scalloping = -float(level_db[_WINDOW_OVERSAMPLE // 2])
    edge = _mainlobe_edge(level_db)
    highest_sidelobe = float(np.max(level_db[edge:]))
    return WindowMetricsResult(
        window=window,
        n=n_v,
        taps=w,
        coherent_gain=coherent_gain,
        enbw_bins=enbw,
        scalloping_loss_db=scalloping,
        worst_case_processing_loss_db=scalloping + 10.0 * np.log10(enbw),
        highest_sidelobe_db=highest_sidelobe,
        mainlobe_width_3db_bins=_width_3db_bins(level_db, _WINDOW_OVERSAMPLE),
    )
