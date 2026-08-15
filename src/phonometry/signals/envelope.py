#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Envelope and instantaneous phase via the Hilbert transform.

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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .spectra import _positive, _validate_signal

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

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
    x: NDArray[np.float64] | list[float],
    fs: float,
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

    :param x: Signal, 1-D.
    :param fs: Sample rate, in Hz.
    :param decimation_factor: Integer output decimation (default 1: off).
    :param antialias: Anti-alias filter the decimated envelope (default
        ``True``).
    :return: An :class:`EnvelopeResult`.
    :raises ValueError: If the inputs or parameters are invalid.
    """
    from scipy import signal as sp_signal

    xa = _validate_signal(x, "x", context="envelope analysis")
    fs_v = _positive(fs, "fs")
    factor = int(decimation_factor)
    if factor < 1:
        raise ValueError("'decimation_factor' must be a positive integer.")
    if factor >= xa.size:
        raise ValueError(
            "'decimation_factor' must be smaller than the record length."
        )

    analytic = sp_signal.hilbert(xa)
    env = np.asarray(np.abs(analytic), dtype=np.float64)
    phase = np.asarray(np.unwrap(np.angle(analytic)), dtype=np.float64)
    inst_freq = np.asarray(
        np.gradient(phase) * fs_v / (2.0 * np.pi), dtype=np.float64
    )

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
        raise ValueError(
            "'band' must be a pair of numeric (low, high) edges in Hz, "
            f"got {band!r}."
        ) from exc
    if not 0.0 < low < high < fs / 2.0:
        raise ValueError(
            "'band' must satisfy 0 < low < high < fs/2; got "
            f"({low:g}, {high:g}) at fs = {fs:g} Hz."
        )
    sos = sp_signal.butter(4, (low, high), btype="bandpass", fs=fs,
                           output="sos")
    # sosfiltfilt's default edge padding needs 3*(2*n_sections + 1)
    # samples; fail with a clear message instead of scipy's padlen error.
    min_length = 3 * (2 * sos.shape[0] + 1) + 1
    if xa.size < min_length:
        raise ValueError(
            f"'x' is too short ({xa.size} samples) for the zero-phase "
            f"band-pass pre-filter, which needs at least {min_length} "
            "samples of padding; lengthen the record or omit 'band'."
        )
    filtered = np.asarray(sp_signal.sosfiltfilt(sos, xa), dtype=np.float64)
    return filtered, (low, high)


def envelope_spectrum(
    x: NDArray[np.float64] | list[float],
    fs: float,
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

    :param x: Signal, 1-D.
    :param fs: Sample rate, in Hz.
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
    fs_v = _positive(fs, "fs")
    if kind not in ("magnitude", "squared"):
        raise ValueError(
            f"'kind' must be 'magnitude' or 'squared', got {kind!r}."
        )
    n = xa.size
    nfft_v = n if nfft is None else int(nfft)
    if nfft_v < n:
        raise ValueError(
            f"'nfft' must be at least the record length ({n} samples)."
        )

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
