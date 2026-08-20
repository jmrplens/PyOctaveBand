#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Calibrated time-frequency analysis: STFT spectrogram and zoom FFT.

Fine-band time-frequency views of a record, following Bendat & Piersol,
*Random Data: Analysis and Measurement Procedures* (4th ed., 2010):

* the **spectrogram** (Section 12.6.4.2): short-time Fourier transforms of
  contiguous, tapered, possibly overlapped segments, displayed over the
  time-frequency plane (Eq. 12.173 defines the unweighted magnitude
  version; this module computes the power version with the exact
  ``'density'``/``'spectrum'`` calibration of
  :func:`~phonometry.signals.spectra.power_spectral_density`, so a
  signal in pascals reads directly in Pa²/Hz or Pa² and averaging the
  columns reproduces the Welch estimate bin by bin). Each cell trades the
  time resolution :math:`T_B = \text{nperseg}/f_\mathrm{s}` against the frequency
  resolution
  :math:`1/T_B` at a resolution-bandwidth-time product of one, so a single
  cell of random data is an unaveraged estimate: the power carries a
  normalized random error of :math:`1/\sqrt{n_\mathrm{d}} = 1` with
  :math:`n_\mathrm{d} = 1` (Eq. 8.158),
  and the magnitude display an error of
  :math:`\sqrt{2}/1.25 \approx 1.13` - the
  Rayleigh-ratio result Bendat & Piersol quote in Section 12.6.4.2.
  Deterministic structure (tones, sweeps, transients) is unaffected by
  that caveat and is what the spectrogram is for.

* the **zoom FFT** (Section 11.5.4): the spectrum of a narrow band
  :math:`[f_{\min}, f_{\max}]` computed on an arbitrarily fine frequency
  grid without the giant FFT block a full-band analysis would need
  (Eq. 11.122). The book's procedure - bandpass, complex demodulation by
  :math:`\exp(-j 2 \pi f_1 t)`, decimation by
  :math:`d = k_2/(k_2 - k_1)` and an FFT of the
  decimated record (Eqs. 11.123-11.130) - is realized here in its exact
  single-pass digital equivalent, the chirp-Z evaluation of the DFT on
  the zoom grid (:func:`scipy.signal.zoom_fft`): both compute the same
  DFT samples of the record, which the test suite verifies to machine
  precision against the demodulate-decimate-DFT chain. The bin spacing
  can be made arbitrarily fine, but the true resolution stays set by the
  record length and taper (the reported effective noise bandwidth
  :math:`B_\mathrm{e} = f_\mathrm{s} \sum w^2 / \left( \sum w \right)^2`): zooming refines
  the grid, only a longer record
  refines the resolution (Eq. 11.127).

Amplitudes are calibrated so that a sine of peak amplitude ``A`` on an
analysis frequency reads :math:`\lvert \text{spectrum} \rvert = A`,
:math:`\text{power} = A^2/2` (its mean
square) - consistent with the ``'spectrum'`` scaling of the Welch module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ..io._resolve import resolve_fs
from .spectra import (
    _DEFAULT_OVERLAP,
    _noverlap_samples,
    _positive,
    _validate_scaling,
    _validate_signal,
    _validate_welch_params,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

    from ..io._signal import Signal

__all__ = [
    "SpectrogramResult",
    "ZoomFFTResult",
    "spectrogram",
    "zoom_fft",
]


def _taper(window: str, nperseg: int) -> NDArray[np.float64]:
    """The (periodic) analysis taper, as the Welch core uses it."""
    from scipy import signal as sp_signal

    return np.asarray(sp_signal.get_window(window, nperseg), dtype=np.float64)


# ---------------------------------------------------------------------------
# Calibrated STFT spectrogram (Bendat & Piersol Section 12.6.4.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpectrogramResult:
    r"""Calibrated STFT power spectrogram (B&P Section 12.6.4.2).

    :ivar times: Segment-centre times, in seconds (one per column).
    :ivar frequencies: One-sided frequency axis, in Hz (one per row).
    :ivar power: Power spectrogram, shape ``(frequencies, times)``
        (units²/Hz for ``'density'`` scaling, units² for ``'spectrum'``).
        Each column is the tapered periodogram of one segment, with the
        exact calibration of
        :func:`~phonometry.signals.spectra.power_spectral_density`:
        the column mean over time reproduces the Welch spectrum bin by
        bin. Integrating a ``'density'`` column over frequency gives that
        segment's taper-weighted mean square
        :math:`\sum (x w)^2 / \sum w^2`; summing
        those over time *and multiplying by the hop duration* ``hop/fs``
        recovers the record energy :math:`\sum x^2 / f_\mathrm{s}` when the
        squared taper
        overlap-adds to a constant (e.g. Hann at 75 % overlap), up to the
        taper roll-off at the record edges (the first and last segments
        are under-weighted: about 1-2 % low for typical records).
    :ivar time_resolution: Segment duration
        :math:`T_B = \text{nperseg}/f_\mathrm{s}`, in
        seconds - the time resolution of the display.
    :ivar resolution_bandwidth: Effective noise bandwidth :math:`B_\mathrm{e}` of
        the tapered segment, in Hz - the frequency resolution
        (:math:`\approx 1/T_B`
        for a light taper; the :math:`B_\mathrm{e} T_B` product per cell is close
        to 1).
    :ivar random_error: Normalized random error of each (unaveraged)
        power cell for random data, :math:`1/\sqrt{n_\mathrm{d}} = 1` with
        :math:`n_\mathrm{d} = 1`
        (Eq. 8.158); Bendat & Piersol quote
        :math:`\sqrt{2}/1.25 \approx 1.13` for the
        magnitude display (Section 12.6.4.2). Deterministic components
        are unaffected.
    :ivar n_segments: Number of segments (columns).
    :ivar hop: Hop between segment starts, in samples.
    :ivar window: Taper name.
    :ivar nperseg: Segment length, in samples.
    :ivar overlap: Segment overlap fraction.
    :ivar scaling: ``'density'`` or ``'spectrum'``.
    """

    times: NDArray[np.float64]
    frequencies: NDArray[np.float64]
    power: NDArray[np.float64]
    time_resolution: float
    resolution_bandwidth: float
    random_error: float
    n_segments: int
    hop: int
    window: str
    nperseg: int
    overlap: float
    scaling: str

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the spectrogram in dB over the time-frequency plane.

        :param language: Label language, ``"en"`` (default) or ``"es"``.
        """
        from .._i18n import check_language
        from .._plot.signals import plot_spectrogram

        check_language(language)
        return plot_spectrogram(self, ax=ax, language=language, **kwargs)


def spectrogram(
    x: Signal | NDArray[np.float64] | list[float],
    fs: float | None = None,
    *,
    window: str = "hann",
    nperseg: int | None = None,
    overlap: float = _DEFAULT_OVERLAP,
    scaling: Literal["density", "spectrum"] = "density",
) -> SpectrogramResult:
    r"""Calibrated STFT power spectrogram (Bendat & Piersol 12.6.4.2).

    The record is split into tapered (Hann by default), overlapped
    segments - exactly the segmentation of
    :func:`~phonometry.signals.spectra.power_spectral_density` - and
    each segment's one-sided periodogram becomes one column of the
    time-frequency display, without the averaging that the Welch
    estimate applies (averaging the columns reproduces it bin by bin).
    No detrending is applied, so absolute calibration is preserved: a
    signal in pascals yields Pa²/Hz (``'density'``) or Pa²
    (``'spectrum'``), and a sine of amplitude ``A`` on an analysis
    frequency reads :math:`A^2/2` - its mean square - in every
    ``'spectrum'`` column it spans.

    The display trades time against frequency resolution through the
    segment length: :math:`T_B = \text{nperseg}/f_\mathrm{s}` of time resolution
    against
    :math:`B_\mathrm{e} \approx 1/T_B` of frequency resolution
    (Section 12.6.4.2). Because
    each cell is a single unaveraged estimate
    (:math:`B_\mathrm{e} T_B \approx 1`), random
    data carries a per-cell normalized random error of 1 (Eq. 8.158 with
    :math:`n_\mathrm{d} = 1`): the spectrogram is a tool for deterministic
    structure -
    tones, sweeps, transients - not a low-variance spectral estimator.

    :param x: Signal, 1-D. Accepts a :class:`phonometry.io.Signal`, whose
        calibration is applied to the samples, so every frame's power come out in
        Pa²/Hz, or Pa² for ``scaling='spectrum'``.
    :param fs: Sample rate, in Hz. Required for a bare array; a
        :class:`~phonometry.io.Signal` brings its own, and an explicit value
        that disagrees with it raises instead of silently winning.
    :param window: Segment taper (any scipy window name; default Hann).
    :param nperseg: Segment length; ``None`` picks a length giving a bin
        spacing of at most 4 Hz (the Welch-module default).
    :param overlap: Segment overlap fraction in [0, 1) (default 0.5).
    :param scaling: ``'density'`` (units²/Hz) or ``'spectrum'`` (units²).
    :return: A :class:`SpectrogramResult`.
    :raises ValueError: If the inputs or parameters are invalid.
    """
    xa = _validate_signal(x, "x", context="a spectrogram")
    fs_v = _positive(resolve_fs(x, fs), "fs")
    scaling_v = _validate_scaling(scaling)
    seg, ovl = _validate_welch_params(xa.size, fs_v, nperseg, overlap)
    nov = _noverlap_samples(seg, ovl)
    step = seg - nov
    k = (xa.size - nov) // step

    w = _taper(window, seg)
    starts = np.arange(k, dtype=np.int64) * step
    frames = np.lib.stride_tricks.sliding_window_view(xa, seg)[starts]
    coeffs = np.fft.rfft(frames * w, axis=-1)
    power = coeffs.real**2 + coeffs.imag**2
    if scaling_v == "density":
        power *= 2.0 / (fs_v * float(np.sum(w * w)))
    else:
        power *= 2.0 / float(np.sum(w)) ** 2
    # DC (and Nyquist, present when the segment length is even) carries a
    # single real Fourier component: no one-sided doubling there.
    power[:, 0] *= 0.5
    if seg % 2 == 0:
        power[:, -1] *= 0.5

    benbw = fs_v * float(np.sum(w * w)) / float(np.sum(w)) ** 2
    return SpectrogramResult(
        times=np.asarray((starts + seg / 2.0) / fs_v, dtype=np.float64),
        frequencies=np.fft.rfftfreq(seg, 1.0 / fs_v),
        power=np.ascontiguousarray(power.T),
        time_resolution=seg / fs_v,
        resolution_bandwidth=benbw,
        random_error=1.0,
        n_segments=k,
        hop=step,
        window=window,
        nperseg=seg,
        overlap=ovl,
        scaling=scaling_v,
    )


# ---------------------------------------------------------------------------
# Zoom FFT (Bendat & Piersol Section 11.5.4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ZoomFFTResult:
    r"""Narrow-band zoom spectrum on a fine frequency grid (B&P 11.5.4).

    :ivar frequencies: Zoom frequency grid from ``f_min`` to ``f_max``
        inclusive, in Hz.
    :ivar spectrum: Complex amplitude-calibrated coefficients: a sine of
        peak amplitude ``A`` on an analysis frequency reads
        :math:`\lvert \text{spectrum} \rvert = A` (calibration
        :math:`2 X(f) / \sum w`; no one-sided
        doubling at exactly 0 Hz or the Nyquist frequency).
    :ivar amplitude: :math:`\lvert \text{spectrum} \rvert` -
        peak-amplitude spectrum.
    :ivar power: Mean-square spectrum :math:`\text{amplitude}^2/2`
        (:math:`\text{amplitude}^2`
        at DC/Nyquist), consistent with the ``'spectrum'`` scaling of
        the Welch module: a tone reads its mean square :math:`A^2/2`.
    :ivar bin_spacing: Grid spacing, in Hz - freely chosen, finer than
        :math:`f_\mathrm{s}/N` if requested (the zoom gain of Eq. 11.127).
    :ivar resolution_bandwidth: Effective noise bandwidth
        :math:`B_\mathrm{e} = f_\mathrm{s} \sum w^2 / \left( \sum w \right)^2` of the
        tapered record, in Hz - the true
        resolution, set by the record length and taper, that no grid
        refinement improves.
    :ivar window: Taper name.
    :ivar n_points: Number of grid points.
    """

    frequencies: NDArray[np.float64]
    spectrum: NDArray[np.complex128]
    amplitude: NDArray[np.float64]
    power: NDArray[np.float64]
    bin_spacing: float
    resolution_bandwidth: float
    window: str
    n_points: int

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the zoom power spectrum in dB over the zoom band.

        :param language: Label language, ``"en"`` (default) or ``"es"``.
        """
        from .._i18n import check_language
        from .._plot.signals import plot_zoom_fft

        check_language(language)
        return plot_zoom_fft(self, ax=ax, language=language, **kwargs)


def zoom_fft(
    x: Signal | NDArray[np.float64] | list[float],
    fs: float | None = None,
    *,
    f_min: float,
    f_max: float,
    n_points: int | None = None,
    window: str = "hann",
) -> ZoomFFTResult:
    r"""Zoom FFT: the spectrum of a narrow band on a fine grid (B&P 11.5.4).

    Resolves closely spaced tones - gear sidebands, twin machines, power
    hum - separated by less than a practical full-band FFT bin, without
    the giant block size Eq. 11.122 would demand. Bendat & Piersol's
    procedure (bandpass, complex demodulation to shift ``f_min`` to zero,
    decimation by the bandwidth ratio, FFT of the decimated record;
    Eqs. 11.123-11.130) is computed here in its exact single-pass digital
    equivalent: the chirp-Z evaluation of the tapered record's DFT on the
    zoom grid (:func:`scipy.signal.zoom_fft`), which yields the same DFT
    samples to machine precision.

    Amplitudes are calibrated per taper coherent gain
    (:math:`2 X / \sum w`), so a
    sine of peak amplitude ``A`` on an analysis frequency reads
    :math:`\text{amplitude} = A` and :math:`\text{power} = A^2/2`
    exactly. The grid can be made
    arbitrarily fine, but the true resolution remains the reported
    effective noise bandwidth of the tapered record (:math:`1/T` for no
    taper): the zoom refines the *grid*; only a longer record separates
    tones closer than :math:`B_\mathrm{e}` (Eq. 11.127).

    :param x: Signal, 1-D. Accepts a :class:`phonometry.io.Signal`, whose
        calibration is applied to the samples, so the spectrum and its
        amplitude come out in Pa and the power in Pa².
    :param fs: Sample rate, in Hz. Required for a bare array; a
        :class:`~phonometry.io.Signal` brings its own, and an explicit value
        that disagrees with it raises instead of silently winning.
    :param f_min: Lower edge of the zoom band, in Hz (:math:`\ge 0`).
    :param f_max: Upper edge of the zoom band, in Hz (:math:`\le f_\mathrm{s}/2`).
    :param n_points: Grid points across ``[f_min, f_max]`` (endpoints
        included); ``None`` places one point per record-length resolution
        :math:`f_\mathrm{s}/N`.
    :param window: Record taper (any scipy window name; default Hann;
        ``'boxcar'`` for none).
    :return: A :class:`ZoomFFTResult`.
    :raises ValueError: If the inputs or parameters are invalid.
    """
    from scipy import signal as sp_signal

    xa = _validate_signal(x, "x", context="a zoom FFT")
    fs_v = _positive(resolve_fs(x, fs), "fs")
    lo = float(f_min)
    hi = float(f_max)
    if not 0.0 <= lo < hi <= fs_v / 2.0:
        raise ValueError("The zoom band must satisfy 0 <= f_min < f_max <= fs/2.")
    if n_points is None:
        m = int(np.ceil((hi - lo) * xa.size / fs_v)) + 1
    else:
        m = int(n_points)
    if m < 2:
        raise ValueError("'n_points' must be at least 2.")

    w = _taper(window, xa.size)
    coeffs = np.asarray(
        sp_signal.zoom_fft(xa * w, [lo, hi], m=m, fs=fs_v, endpoint=True),
        dtype=np.complex128,
    )
    freqs = np.linspace(lo, hi, m)
    # One-sided doubling, except at DC and Nyquist where the positive- and
    # negative-frequency components coincide. The grid is an increasing
    # linspace over [f_min, f_max] within [0, fs/2], so only its first
    # point can sit at DC and only its last at Nyquist; both are matched
    # with a tolerance far below any usable bin spacing.
    eps = 1e-12 * fs_v
    dc_first = abs(lo) <= eps
    nyquist_last = abs(hi - fs_v / 2.0) <= eps
    gain = np.full(m, 2.0 / float(np.sum(w)))
    if dc_first:
        gain[0] *= 0.5
    if nyquist_last:
        gain[-1] *= 0.5
    spectrum = coeffs * gain
    amplitude = np.abs(spectrum)
    power = 0.5 * amplitude**2
    if dc_first:
        power[0] = amplitude[0] ** 2
    if nyquist_last:
        power[-1] = amplitude[-1] ** 2

    return ZoomFFTResult(
        frequencies=freqs,
        spectrum=spectrum,
        amplitude=amplitude,
        power=power,
        bin_spacing=(hi - lo) / (m - 1),
        resolution_bandwidth=fs_v * float(np.sum(w * w)) / float(np.sum(w)) ** 2,
        window=window,
        n_points=m,
    )
