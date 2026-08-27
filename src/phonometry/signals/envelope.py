#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Envelope and instantaneous phase via the Hilbert transform.

Signal-envelope analysis following Bendat & Piersol, *Random Data:
Analysis and Measurement Procedures* (4th ed., 2010), Chapter 13. The
analytic signal :math:`z(t) = x(t) + j \tilde{x}(t)` (Eq. 13.15, with
:math:`\tilde{x}` the Hilbert transform of ``x``) yields

* the **envelope**
  :math:`A(t) = [x^2(t) + \tilde{x}^2(t)]^{1/2}` (Eq. 13.17),
* the **instantaneous phase**
  :math:`\theta(t) = \arctan[\tilde{x}(t)/x(t)]`, unwrapped
  (Eq. 13.18), and
* the **instantaneous frequency**
  :math:`f(t) = (1/2\pi) \, d\theta/dt` (Eq. 13.19).

The analytic signal is computed the way the book recommends
(Section 13.1.1): the one-sided spectrum construction
:math:`Z(f) = 2 X(f)` for :math:`f > 0`, :math:`X(0)` at DC and ``0``
for :math:`f < 0`
(Eq. 13.25) - which is exactly what :func:`scipy.signal.hilbert`
implements, and the same construction the ECMA-418-2 psychoacoustic chain
of :mod:`phonometry.psychoacoustics` applies per auditory band (its
Formulae 65/119 take ``|hilbert|`` and subsample by 32; the standard can
subsample directly because each band is narrow). Closed-form pairs from
Table 13.1 (``cos → sin``, an AM envelope recovered exactly) anchor the
tests.

The envelope of a band-limited signal is itself low-frequency, so the
result offers optional **decimation**: an anti-aliased zero-phase FIR
decimator for general records, or plain subsampling (``antialias=False``)
matching the ECMA-internal convention when the input is already
narrowband.

The **envelope spectrum** (:func:`envelope_spectrum`) transforms the
detected envelope itself: Section 13.3 of the book runs a band-pass
filter and a square-law envelope detector into a DC remover before
correlating (Figure 13.11), because the spectral content of the envelope
- not of the signal - is where amplitude modulations show as discrete
lines. The optional ``band`` argument reproduces the figure's band-pass
front end (the classical bearing-envelope chain: isolate the resonance
band, then envelope it). An AM tone with modulation frequency :math:`f_\mathrm{m}`
(on an analysis bin) and depth ``m`` puts a line of closed-form
amplitude at exactly :math:`f_\mathrm{m}`, the anchor the tests pin; off-bin
modulation lines read low by the taper's scalloping loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .._internal.validation import require_ranks, require_same_length
from ..io._resolve import resolve_fs
from .spectra import _positive, _validate_signal

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

    from ..io._signal import Signal

__all__ = [
    "EnvelopeResult",
    "EnvelopeSpectrumResult",
    "envelope",
    "envelope_spectrum",
]


@dataclass(frozen=True)
class EnvelopeResult:
    r"""Envelope and instantaneous phase of a signal (B&P Chapter 13).

    All output arrays share the (possibly decimated) time axis
    :attr:`times`; the original record is kept at full rate for plotting.

    :ivar times: Time axis of the outputs, in seconds.
    :ivar envelope: Envelope :math:`A(t) = \lvert z(t) \rvert`
        (Eq. 13.17).
    :ivar phase: Unwrapped instantaneous phase :math:`\theta(t)`, in
        radians (Eq. 13.18).
    :ivar instantaneous_frequency:
        :math:`f(t) = (1/2\pi) \, d\theta/dt`, in Hz
        (Eq. 13.19), differentiated at full rate before any decimation.
    :ivar fs: Sample rate of the outputs, in Hz (``signal_fs`` divided by
        :attr:`decimation_factor`).
    :ivar signal: The analysed record, at full rate.
    :ivar signal_fs: Sample rate of :attr:`signal`, in Hz.
    :ivar decimation_factor: Integer decimation applied to the outputs
        (1: none).
    :ivar antialias: Whether the decimation was anti-alias filtered.
    """

    times: NDArray[np.float64]
    envelope: NDArray[np.float64]
    phase: NDArray[np.float64]
    instantaneous_frequency: NDArray[np.float64]
    fs: float
    signal: NDArray[np.float64]
    signal_fs: float
    decimation_factor: int
    antialias: bool

    def __post_init__(self) -> None:
        """Reject outputs that do not all run over the decimated time axis.

        The three analytic-signal quantities are one account of the record
        over :attr:`times`, and the reader takes them that way: :meth:`plot`
        stacks the envelope panel and the instantaneous-frequency panel on a
        shared time axis, so the moment a modulation peaks in the upper panel
        is read against the frequency directly below it. Decimation is what
        makes that fragile -- the envelope goes through a filter and the
        phase through plain subsampling, and only the output rate ties the
        two results together afterwards.

        :attr:`signal` is deliberately left out. It is the analysed record at
        full rate, kept for the plot's background trace and drawn on a time
        axis of its own built from :attr:`signal_fs`; it matches
        :attr:`times` only when no decimation was asked for, and pinning the
        two together would reject every decimated result the module produces.

        :raises ValueError: if an output disagrees with the time axis.
        """
        require_ranks(
            self,
            times=1,
            envelope=1,
            phase=1,
            instantaneous_frequency=1,
            signal=1,
        )
        require_same_length(
            self,
            "times",
            "envelope",
            "phase",
            "instantaneous_frequency",
            axis="output sample",
        )

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes | NDArray[Any]:
        """Plot the signal with its envelope and the instantaneous frequency.

        :param language: Label language, ``"en"`` (default) or ``"es"``.
        """
        from .._i18n import check_language
        from .._plot.signals import plot_envelope

        check_language(language)
        return plot_envelope(self, ax=ax, language=language, **kwargs)


def _decimate_envelope(
    env: NDArray[np.float64], factor: int, antialias: bool
) -> NDArray[np.float64]:
    """Decimate the envelope, anti-aliased (zero-phase FIR) or plain."""
    if not antialias:
        return env[::factor].copy()
    from scipy import signal as sp_signal

    return np.asarray(
        sp_signal.decimate(env, factor, ftype="fir", zero_phase=True),
        dtype=np.float64,
    )


def envelope(
    x: Signal | NDArray[np.float64] | list[float],
    fs: float | None = None,
    *,
    decimation_factor: int = 1,
    antialias: bool = True,
) -> EnvelopeResult:
    r"""Envelope, instantaneous phase and frequency via Hilbert transform.

    Builds the analytic signal by the one-sided spectrum construction of
    Bendat & Piersol Eq. 13.25 (``scipy.signal.hilbert``) and returns the
    envelope :math:`\lvert z(t) \rvert`, the unwrapped instantaneous phase
    and the instantaneous frequency (Eqs. 13.17-13.19). For an
    amplitude-modulated carrier :math:`u(t) \cos(2 \pi f_0 t)` with ``u``
    low-frequency and non-negative
    the envelope recovers :math:`u(t)` exactly in the ideal continuous case
    (Eq. 13.27); a discrete record shows small edge effects at the record
    boundaries.

    The optional decimation reduces the output rate by an integer factor:
    the envelope is anti-alias filtered with a zero-phase FIR decimator
    by default, or plainly subsampled with ``antialias=False`` - the
    convention the ECMA-418-2 loudness/roughness chain applies internally
    after its auditory bandpass, appropriate when the input is already
    narrowband. The phase and instantaneous frequency, smooth after
    unwrapping and differentiated at full rate, are subsampled onto the
    same time axis.

    :param x: Signal, 1-D. Accepts a :class:`phonometry.io.Signal`, whose
        calibration is applied to the samples, so the envelope and the
        carried waveform come out in Pa. The phase and the instantaneous
        frequency do not move.
    :param fs: Sample rate, in Hz. Required for a bare array; a
        :class:`~phonometry.io.Signal` brings its own, and an explicit value
        that disagrees with it raises instead of silently winning.
    :param decimation_factor: Integer output decimation (default 1: off).
    :param antialias: Anti-alias filter the decimated envelope (default
        ``True``).
    :return: An :class:`EnvelopeResult`.
    :raises ValueError: If the inputs or parameters are invalid.
    """
    from scipy import signal as sp_signal

    xa = _validate_signal(x, "x", context="envelope analysis")
    fs_v = _positive(resolve_fs(x, fs), "fs")
    factor = int(decimation_factor)
    if factor < 1:
        msg = "'decimation_factor' must be a positive integer."
        raise ValueError(msg)
    if factor >= xa.size:
        msg = "'decimation_factor' must be smaller than the record length."
        raise ValueError(msg)

    analytic = sp_signal.hilbert(xa)
    env = np.asarray(np.abs(analytic), dtype=np.float64)
    phase = np.asarray(np.unwrap(np.angle(analytic)), dtype=np.float64)
    inst_freq = np.asarray(np.gradient(phase) * fs_v / (2.0 * np.pi), dtype=np.float64)

    if factor > 1:
        env = _decimate_envelope(env, factor, antialias)
        phase = phase[::factor].copy()
        inst_freq = inst_freq[::factor].copy()

    out_fs = fs_v / factor
    times = np.arange(env.size, dtype=np.float64) / out_fs
    return EnvelopeResult(
        times=times,
        envelope=env,
        phase=phase,
        instantaneous_frequency=inst_freq,
        fs=out_fs,
        signal=xa.copy(),
        signal_fs=fs_v,
        decimation_factor=factor,
        antialias=bool(antialias),
    )


@dataclass(frozen=True)
class EnvelopeSpectrumResult:
    r"""Amplitude spectrum of a signal's envelope (B&P Section 13.3).

    :ivar frequencies: Frequency axis of the spectrum, in Hz.
    :ivar amplitude: One-sided amplitude spectrum of the (mean-removed)
        envelope: the height of a discrete modulation line in the units of
        the envelope itself, exact when the modulation frequency falls on
        an analysis bin (off-bin lines read low by the taper's scalloping
        loss; see :func:`envelope_spectrum`). The zero-frequency bin is
        not doubled.
    :ivar mean_level: Mean of the detected envelope (the DC the remover of
        Figure 13.11 takes out): the carrier amplitude for
        ``kind="magnitude"``, its mean square for ``kind="squared"``.
    :ivar kind: ``"magnitude"`` (Hilbert envelope :math:`A(t)`) or
        ``"squared"`` (the book's square-law detector, :math:`A^2(t)`).
    :ivar times: Time axis of :attr:`envelope`, in seconds.
    :ivar envelope: The detector output that was transformed, at full rate
        (before mean removal and tapering).
    :ivar window: Taper name applied before the transform.
    :ivar remove_dc: Whether the envelope mean was removed first.
    :ivar fs: Sample rate of the analysed record, in Hz.
    :ivar nfft: FFT length used.
    :ivar band: ``(low, high)`` edges of the zero-phase band-pass
        pre-filter applied before envelope detection, in Hz, or ``None``
        (no pre-filter).
    """

    frequencies: NDArray[np.float64]
    amplitude: NDArray[np.float64]
    mean_level: float
    kind: str
    times: NDArray[np.float64]
    envelope: NDArray[np.float64]
    window: str
    remove_dc: bool
    fs: float
    nfft: int
    band: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        """Reject a result whose two panels do not run over their own axes.

        The result carries the two halves of Bendat & Piersol's Figure 13.11
        chain, and they are measured on different axes that are not the same
        length: the detected envelope over the record's ``n`` samples, its
        amplitude spectrum over the ``nfft // 2 + 1`` bins of the transform
        that was taken of it. Each is checked against its own companion only.
        :meth:`plot` draws them as two panels, the envelope against
        :attr:`times` above and the modulation lines against
        :attr:`frequencies` below, and reads a line's height as the
        modulation depth carried by the envelope in the panel above -- which
        holds only while each panel's pair is one measurement.

        :attr:`band` states the two edges of the optional band-pass front
        end, so it is a pair rather than an axis and stays out of both
        groups.

        :attr:`mean_level` is the third statement about the upper panel, and
        it is closed over the panel's own column: it is the plain mean of
        :attr:`envelope`, the DC that Figure 13.11's remover subtracts before
        the transform, so ``envelope - mean_level`` is what the lines below
        were measured on. Nothing about it is a decibel -- the detector output
        is a linear amplitude (or its square), and the documented closed forms
        :math:`A_0` and :math:`A_0^2 (1 + m^2/2)` are exactly the arithmetic
        means of those two columns -- so neither a level of the column nor its
        energy mean restates it. :meth:`plot` draws it as the labelled rule
        across the envelope panel, where the reader takes the modulation as
        the trace's excursion about that rule: a mean level belonging to
        another detector, or to the same column expressed some other way,
        lands as a rule the trace never crosses, silently stretching the
        panel's limits to reach it.

        The slack is relative and only relative, unlike the peak-rate guard
        of :mod:`phonometry.metrology.data_qualification` and unlike the
        decibel margins elsewhere in the tree: an envelope mean carries the
        record's own units and no bounded scale, so a caller who recomputed
        the mean along another summation path must not be refused over the
        last bit, while an absolute floor would be a floor in nothing --
        a squared detector reading a tone at the threshold of hearing means
        some :math:`10^{-9}` Pa^2, and a floor that size would wave through
        every statement about it, zero and negatives included.

        An empty pair is left alone: it carries no mean to restate, and the
        transform never makes one. So is an undetermined one:
        :func:`envelope_spectrum` refuses a non-finite record, but a finite
        one can still overflow inside the Hilbert transform and detect an
        all-NaN envelope, and NaN is then the mean's honest restatement --
        which no comparison of NaN with NaN can confirm. Such a column is
        one the producer legitimately emits, so it is passed rather than
        refused; only a column with a mean is held to it.

        :raises ValueError: if the spectrum disagrees with its frequency
            axis, the envelope with its time axis, or ``mean_level`` with the
            envelope it is the mean of.
        """
        require_ranks(self, frequencies=1, amplitude=1, times=1, envelope=1)
        require_same_length(self, "frequencies", "amplitude", axis="frequency")
        require_same_length(self, "times", "envelope", axis="sample")
        detector = np.asarray(self.envelope, dtype=np.float64)
        if detector.size == 0:
            return
        expected = float(np.mean(detector))
        if math.isnan(expected):
            return
        if not math.isclose(self.mean_level, expected, rel_tol=1e-9, abs_tol=0.0):
            msg = (
                "EnvelopeSpectrumResult: 'mean_level' must be the plain mean "
                "of 'envelope', the DC removed before the transform (Bendat & "
                f"Piersol Figure 13.11); got {self.mean_level!r} where "
                f"'envelope' means {expected!r}."
            )
            raise ValueError(msg)

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes | NDArray[Any]:
        """Plot the detected envelope and its amplitude spectrum.

        With ``ax`` given, only the spectrum panel is drawn on it.

        :param language: Label language, ``"en"`` (default) or ``"es"``.
        """
        from .._i18n import check_language
        from .._plot.signals import plot_envelope_spectrum

        check_language(language)
        return plot_envelope_spectrum(self, ax=ax, language=language, **kwargs)


def _bandpass_pre_filter(
    xa: NDArray[np.float64], fs: float, band: tuple[float, float]
) -> tuple[NDArray[np.float64], tuple[float, float]]:
    """Zero-phase 4th-order Butterworth band-pass ahead of the detector.

    Returns the filtered record and the validated ``(low, high)`` edges.
    """
    from scipy import signal as sp_signal

    try:
        low, high = (float(edge) for edge in band)
    except (TypeError, ValueError) as exc:
        msg = f"'band' must be a pair of numeric (low, high) edges in Hz, got {band!r}."
        raise ValueError(msg) from exc
    if not 0.0 < low < high < fs / 2.0:
        msg = (
            "'band' must satisfy 0 < low < high < fs/2; got "
            f"({low:g}, {high:g}) at fs = {fs:g} Hz."
        )
        raise ValueError(msg)
    sos = sp_signal.butter(4, (low, high), btype="bandpass", fs=fs, output="sos")
    # sosfiltfilt's default edge padding needs 3*(2*n_sections + 1)
    # samples; fail with a clear message instead of scipy's padlen error.
    min_length = 3 * (2 * sos.shape[0] + 1) + 1
    if xa.size < min_length:
        msg = (
            f"'x' is too short ({xa.size} samples) for the zero-phase "
            f"band-pass pre-filter, which needs at least {min_length} "
            "samples of padding; lengthen the record or omit 'band'."
        )
        raise ValueError(msg)
    filtered = np.asarray(sp_signal.sosfiltfilt(sos, xa), dtype=np.float64)
    return filtered, (low, high)


def envelope_spectrum(
    x: Signal | NDArray[np.float64] | list[float],
    fs: float | None = None,
    *,
    kind: str = "magnitude",
    window: str = "hann",
    nfft: int | None = None,
    remove_dc: bool = True,
    band: tuple[float, float] | None = None,
) -> EnvelopeSpectrumResult:
    r"""Amplitude spectrum of the envelope: where modulations become lines.

    Follows the structure of Bendat & Piersol Section 13.3 (Figure 13.11):
    a band-pass filter (optional here), an envelope detector, a DC
    remover, and a spectral view of what is left. The detector is the
    Hilbert envelope :math:`A(t) = \lvert z(t) \rvert`
    (``kind="magnitude"``, the practical default) or the book's
    square-law detector
    :math:`A^2(t) = x^2 + \tilde{x}^2` (``kind="squared"``); its mean is
    removed
    (kept in :attr:`EnvelopeSpectrumResult.mean_level`) and the remainder
    is tapered and transformed once, scaled by the taper's coherent gain
    so a sinusoidal modulation whose frequency falls on an analysis bin
    reads out as a line at its exact amplitude. An off-bin modulation
    frequency reads low by the taper's scalloping loss -- up to about
    1.4 dB (~15 %) for the default Hann midway between bins -- like any
    single-record amplitude spectrum.

    Closed forms for an AM tone
    :math:`A_0 (1 + m \cos(2 \pi f_\mathrm{m} t)) \cos(2 \pi f_\mathrm{c} t)`
    with :math:`0 \le m < 1` and :math:`f_\mathrm{m}` on an analysis bin:

    * ``kind="magnitude"``: a line of amplitude :math:`A_0 m` at
      :math:`f_\mathrm{m}`; mean level :math:`A_0`.
    * ``kind="squared"``: lines :math:`2 A_0^2 m` at :math:`f_\mathrm{m}` and
      :math:`A_0^2 m^2 / 2` at :math:`2 f_\mathrm{m}`; mean level
      :math:`A_0^2 (1 + m^2/2)`.

    Amplitude modulation of rotating machinery (bearing and gear defect
    frequencies), mains hum and wind-turbine amplitude modulation appear
    the same way: lines at the modulation frequency and its harmonics,
    separated from the carrier's own spectrum. For the classical bearing
    chain -- isolate a structural-resonance band excited by the defect
    impacts, then envelope it -- pass the resonance band as ``band``: the
    record is band-pass filtered (zero-phase, so the modulation phase is
    untouched) before the detector, the Figure 13.11 front end.

    :param x: Signal, 1-D. Accepts a :class:`phonometry.io.Signal`, whose
        calibration is applied to the samples, so the envelope, its spectral
        amplitude and the mean level come out in Pa under the default
        ``kind='magnitude'``, and in Pa² under ``'squared'``, which squares
        the envelope before transforming it.
    :param fs: Sample rate, in Hz. Required for a bare array; a
        :class:`~phonometry.io.Signal` brings its own, and an explicit value
        that disagrees with it raises instead of silently winning.
    :param kind: ``"magnitude"`` (default) or ``"squared"``.
    :param window: Taper (any scipy window name; default Hann). The
        amplitude is corrected for the taper's coherent gain.
    :param nfft: FFT length, at least ``x.size`` (default: the record
        length).
    :param remove_dc: Remove the envelope mean before the transform
        (default ``True``, the Figure 13.11 DC remover); the mean is
        reported either way.
    :param band: Optional ``(low, high)`` band-pass edges, in Hz
        (:math:`0 < \text{low} < \text{high} < f_\mathrm{s}/2`), applied to the
        record before
        envelope detection as a zero-phase 4th-order Butterworth
        (:func:`scipy.signal.sosfiltfilt`, giving an 8th-order magnitude
        roll-off). Default ``None``: detect on the record as given.
    :return: An :class:`EnvelopeSpectrumResult`.
    :raises ValueError: If the inputs or parameters are invalid.
    """
    from scipy import signal as sp_signal

    xa = _validate_signal(x, "x", context="an envelope spectrum")
    fs_v = _positive(resolve_fs(x, fs), "fs")
    if kind not in ("magnitude", "squared"):
        msg = f"'kind' must be 'magnitude' or 'squared', got {kind!r}."
        raise ValueError(msg)
    n = xa.size
    nfft_v = n if nfft is None else int(nfft)
    if nfft_v < n:
        msg = f"'nfft' must be at least the record length ({n} samples)."
        raise ValueError(msg)

    band_v: tuple[float, float] | None = None
    if band is not None:
        xa, band_v = _bandpass_pre_filter(xa, fs_v, band)

    env = np.asarray(np.abs(sp_signal.hilbert(xa)), dtype=np.float64)
    detector = env**2 if kind == "squared" else env
    mean_level = float(np.mean(detector))
    y = detector - mean_level if remove_dc else detector

    taper = np.asarray(sp_signal.get_window(window, n), dtype=np.float64)
    spectrum = np.abs(np.fft.rfft(y * taper, nfft_v)) / float(np.sum(taper))
    amplitude = 2.0 * spectrum
    amplitude[0] = spectrum[0]
    if nfft_v % 2 == 0:
        amplitude[-1] = spectrum[-1]  # the Nyquist bin is not doubled either

    return EnvelopeSpectrumResult(
        frequencies=np.fft.rfftfreq(nfft_v, 1.0 / fs_v),
        amplitude=np.asarray(amplitude, dtype=np.float64),
        mean_level=mean_level,
        kind=kind,
        times=np.arange(n, dtype=np.float64) / fs_v,
        envelope=detector,
        window=window,
        remove_dc=bool(remove_dc),
        fs=fs_v,
        nfft=nfft_v,
        band=band_v,
    )
