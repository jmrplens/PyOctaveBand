#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Test signals and sample-rate utilities.

The signal toolbox of the metrology domain: deterministic test signals and
the two sample-rate operations every measurement chain eventually needs,
with their accuracy stated instead of implied.

* :func:`noise_signal` - Gaussian noise with an exact power-law spectral
  slope: white (0 dB/octave), pink (-3.01), red (-6.02, also called
  Brownian), blue (+3.01) and violet (+6.02). The autospectral density
  follows :math:`G_{xx}(f) \propto f^\alpha` with :math:`\alpha` = 0,
  -1, -2, +1 and +2 respectively,
  so the level changes by exactly :math:`3.01 \alpha` dB per octave
  (:math:`10 \log_{10} 2 = 3.0103` dB). The colors are synthesized by filtering
  seeded white Gaussian noise in the frequency domain: the DFT of the
  white record is multiplied by the exact magnitude response
  :math:`\lvert H(f) \rvert = (f/f_{\mathrm{ref}})^{\alpha/2}`
  bin by bin (a zero-phase FIR filter applied circularly), so the *expected*
  spectrum follows the power law exactly at every synthesis bin above DC and
  a measured slope deviates only by the random error of the spectral
  estimate - not the piecewise or few-pole approximations whose pink slope
  ripples by fractions of a dB. The DC bin is zeroed for the colored
  variants (a power law has no finite DC value) and the record is rescaled
  to the requested RMS exactly. With the same ``seed`` the generator is
  fully deterministic across runs.

* :func:`tone_burst` - the gated sine burst of IEC 60268-1:1985 (Annex A,
  Clause A2): the tone starts at a zero crossing and lasts an integral
  number of full periods, either as a single burst or as a repetitive train
  with a stated repetition rate. The result records the rectangular gating
  envelope and the exact on/off sample bookkeeping, so meter ballistics and
  dynamic-response tests can state their stimulus instead of hand-rolling
  it.

* :func:`resample_signal` - polyphase resampling behind an explicit
  anti-alias specification. The lowpass FIR is designed here (Kaiser
  window method) from two numbers the caller controls - the stopband
  attenuation in dB and the transition-band fraction of the target
  Nyquist - and the designed filter is returned with the result, so the
  alias rejection of a resampled record is a documented property, not a
  library default.

* :func:`fractional_delay` - band-limited delay by an arbitrary
  (sub-sample) number of samples via a frequency-domain phase ramp,
  ``linear`` (zero-padded, for transients and impulse responses; the same
  kernel :func:`~phonometry.signals.correlation.align_impulse_responses`
  uses) or ``circular`` (for periodic records, exact to machine precision
  on bin-centered tones).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np

from .._internal.validation import require_ranks, require_same_length
from .._internal.warnings import PhonometryWarning
from ..io._resolve import apply_calibration, like_input, resolve_fs
from .spectra import _positive

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

    from ..io._signal import Signal

__all__ = [
    "ResampledSignalResult",
    "ToneBurstResult",
    "fractional_delay",
    "noise_signal",
    "resample_signal",
    "tone_burst",
]

#: Power-law exponent α of ``Gxx(f) ∝ f^α`` per color.
_COLOR_EXPONENTS: dict[str, float] = {
    "white": 0.0,
    "pink": -1.0,
    "red": -2.0,
    "blue": 1.0,
    "violet": 2.0,
}

#: Fewest samples ``round(fs * seconds)`` may give: the frequency-domain
#: color shaping needs bins to work with.
_MIN_NOISE_SAMPLES = 16

#: Fewest samples of a processable one-dimensional record. Deliberately
#: weaker than ``spectra._MIN_SAMPLES`` (the spectral-estimate floor these
#: consumers do not need), so that constant is not reused here.
_MIN_RECORD_SAMPLES = 2

#: Gate-close offset (in samples) below which ``fs`` and ``frequency``
#: count as commensurate (the gate closes sample-exactly on the tone's
#: final zero crossing) and the incommensurate-frequency warning is
#: suppressed. Same value as ``synchronous_average._ALIGN_TOL``, which
#: cannot be imported here (that module imports this one).
_GATE_ALIGN_TOL = 1e-9

#: Smallest accepted anti-alias stopband attenuation, in dB.
_MIN_STOPBAND_ATTENUATION_DB = 30.0

#: Widest accepted transition band, as a fraction of the smaller Nyquist
#: frequency.
_MAX_TRANSITION_WIDTH = 0.5


def noise_signal(
    fs: float,
    seconds: float = 1.0,
    *,
    color: Literal["white", "pink", "red", "blue", "violet"] = "white",
    rms: float = 1.0,
    seed: int | None = None,
) -> NDArray[np.float64]:
    r"""Generate Gaussian noise with an exact power-law spectral slope.

    :math:`G_{xx}(f) \propto f^\alpha` with :math:`\alpha` = 0 (white),
    -1 (pink, -3.01 dB/octave),
    -2 (red/Brownian, -6.02), +1 (blue, +3.01) or +2 (violet, +6.02),
    shaped by an exact frequency-domain filter (see the module docstring),
    zero-mean and rescaled to the requested RMS exactly.

    :param fs: Rate to generate at, in Hz.
    :param seconds: Duration, in seconds (at least 16 samples).
    :param color: Noise color: ``'white'``, ``'pink'``, ``'red'``,
        ``'blue'`` or ``'violet'``.
    :param rms: Root-mean-square value of the returned record.
    :param seed: Seed for :func:`numpy.random.default_rng`; the same seed
        reproduces the same record. ``None`` draws fresh entropy.
    :return: The noise record, ``round(fs * seconds)`` samples.
    :raises ValueError: If the inputs or parameters are invalid.
    """
    fs_v = float(fs)
    if not np.isfinite(fs_v) or fs_v <= 0.0:
        msg = "'fs' must be a positive, finite number."
        raise ValueError(msg)
    seconds_v = float(seconds)
    if not np.isfinite(seconds_v) or seconds_v <= 0.0:
        msg = "'seconds' must be a positive, finite number."
        raise ValueError(msg)
    n = round(fs_v * seconds_v)
    if n < _MIN_NOISE_SAMPLES:
        msg = f"'fs'*'seconds' must give at least 16 samples, got {n}."
        raise ValueError(msg)
    if color not in _COLOR_EXPONENTS:
        msg = "'color' must be one of 'white', 'pink', 'red', 'blue', 'violet'."
        raise ValueError(msg)
    rms_v = float(rms)
    if not np.isfinite(rms_v) or rms_v <= 0.0:
        msg = "'rms' must be a positive, finite number."
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    if color != "white":
        alpha = _COLOR_EXPONENTS[color]
        spectrum = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs_v)
        gain = np.zeros_like(freqs)
        # |H(f)| = (f/f_ref)^(α/2) shapes the PSD by exactly f^α; the
        # reference frequency only sets the overall gain, which the RMS
        # rescaling below removes.
        gain[1:] = (freqs[1:] / freqs[1]) ** (alpha / 2.0)
        x = np.fft.irfft(spectrum * gain, n)
    x = x - float(np.mean(x))
    scale = float(np.sqrt(np.mean(x * x)))
    if scale <= 0.0:  # pragma: no cover - white Gaussian is never all-zero
        msg = "Degenerate all-zero record; use another seed."
        raise ValueError(msg)
    return np.asarray(x * (rms_v / scale), dtype=np.float64)


# ---------------------------------------------------------------------------
# Shared validation (same conventions as the spectral estimators)
# ---------------------------------------------------------------------------


def _validate_1d_finite(
    x: Signal | NDArray[np.float64] | list[float], name: str
) -> NDArray[np.float64]:
    """Coerce, check and calibrate a one-dimensional record.

    The sibling of ``spectra._validate_signal`` for the consumers that do
    not need its minimum length: same contract, so a
    :class:`phonometry.io.Signal` arrives in pascals here too and a
    multichannel one is refused by the same complaint a 2-D array gets.
    """
    xa = np.asarray(x, dtype=np.float64)
    if xa.ndim != 1:
        msg = f"'{name}' must be one-dimensional."
        raise ValueError(msg)
    if xa.size < _MIN_RECORD_SAMPLES:
        msg = f"'{name}' must have at least 2 samples."
        raise ValueError(msg)
    if not np.all(np.isfinite(xa)):
        msg = f"'{name}' must be finite."
        raise ValueError(msg)
    return apply_calibration(x, xa)


# ---------------------------------------------------------------------------
# IEC 60268-1 tone bursts (Annex A, Clause A2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToneBurstResult:
    """Gated sine burst per IEC 60268-1:1985 (Annex A, Clause A2).

    The tone starts at a zero crossing (positive-going) and the gate stays
    open for an integral number of full periods, as Clause A2.1 requires
    of the dynamic-response stimulus. With a repetition rate the record is
    a train of identical bursts, one per repetition period (Clause A2.2).

    :ivar signal: The burst record (silence, bursts and gaps included).
    :ivar envelope: Rectangular gating envelope of :attr:`signal`
        (``amplitude`` while the gate is open, ``0`` elsewhere).
    :ivar fs: Sample rate, in Hz.
    :ivar frequency: Tone frequency, in Hz.
    :ivar cycles: Full tone periods per burst.
    :ivar amplitude: Peak amplitude of the tone.
    :ivar burst_seconds: Burst duration ``cycles/frequency``, in seconds.
    :ivar burst_samples: Samples per burst,
        ``round(fs * cycles / frequency)``.
    :ivar onset_sample: Index of the first sample of the first burst.
    :ivar repetitions: Number of bursts in the record.
    :ivar repetition_rate: Bursts per second, or ``None`` (single burst).
    :ivar period_samples: Samples per repetition period
        (``round(fs/repetition_rate)``), or ``None`` (single burst).
    :ivar duty_cycle: On fraction ``burst_samples/period_samples``, or
        ``None`` (single burst).
    """

    signal: NDArray[np.float64]
    envelope: NDArray[np.float64]
    fs: float
    frequency: float
    cycles: int
    amplitude: float
    burst_seconds: float
    burst_samples: int
    onset_sample: int
    repetitions: int
    repetition_rate: float | None
    period_samples: int | None
    duty_cycle: float | None

    def __post_init__(self) -> None:
        """Reject a burst whose gate does not cover the record it gates.

        :attr:`envelope` is not an independent curve but a statement about
        :attr:`signal`: it is ``amplitude`` exactly where the gate of Clause
        A2.1 is open and zero everywhere else, which is what makes the record
        readable as an integral number of full periods. The figure draws both
        on one time axis built from the length of :attr:`signal`, mirroring
        the envelope above and below the waveform, so an envelope of another
        length stops describing this record: shorter, the gate appears to
        close before the last burst has sounded; longer, it stays open over
        silence that is not there.

        :raises ValueError: if the envelope disagrees with the signal.
        """
        require_ranks(self, signal=1, envelope=1)
        require_same_length(self, "signal", "envelope", axis="sample")

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the burst waveform with its gating envelope.

        :param language: Label language, ``"en"`` (default) or ``"es"``.
        """
        from .._i18n import check_language
        from .._plot.signals import plot_tone_burst

        check_language(language)
        return plot_tone_burst(self, ax=ax, language=language, **kwargs)


def _tone_burst_scalars(
    fs: float,
    frequency: float,
    cycles: int,
    amplitude: float,
    repetitions: int,
    pre_silence: float,
    post_silence: float,
) -> tuple[float, float, int, float, int]:
    """Validate the scalar arguments of :func:`tone_burst`."""
    fs_v = _positive(fs, "fs")
    f_v = _positive(frequency, "frequency")
    if f_v >= fs_v / 2.0:
        msg = "'frequency' must be below the Nyquist rate fs/2."
        raise ValueError(msg)
    cycles_v = int(cycles)
    if cycles_v != cycles or cycles_v < 1:
        msg = "'cycles' must be a positive integer."
        raise ValueError(msg)
    amplitude_v = _positive(amplitude, "amplitude")
    repetitions_v = int(repetitions)
    if repetitions_v != repetitions or repetitions_v < 1:
        msg = "'repetitions' must be a positive integer."
        raise ValueError(msg)
    for name, value in (("pre_silence", pre_silence), ("post_silence", post_silence)):
        if not np.isfinite(float(value)) or float(value) < 0.0:
            msg = f"'{name}' must be a non-negative, finite number."
            raise ValueError(msg)
    return fs_v, f_v, cycles_v, amplitude_v, repetitions_v


def _tone_burst_period(
    fs_v: float,
    n_on: int,
    repetitions_v: int,
    repetition_rate: float | None,
) -> tuple[float | None, int | None, float | None]:
    """Repetition ``(rate, period_samples, duty_cycle)`` of the burst train.

    ``(None, None, None)`` for a single burst without a repetition rate.
    """
    if repetition_rate is None:
        if repetitions_v > 1:
            msg = "'repetition_rate' is required when 'repetitions' > 1."
            raise ValueError(msg)
        return None, None, None
    rate_v = _positive(repetition_rate, "repetition_rate")
    period = round(fs_v / rate_v)
    if period < n_on:
        msg = (
            "The burst does not fit in one repetition period: "
            f"{n_on} samples per burst, {period} per period."
        )
        raise ValueError(msg)
    return rate_v, period, n_on / period


def tone_burst(
    fs: float,
    frequency: float,
    cycles: int,
    *,
    amplitude: float = 1.0,
    repetitions: int = 1,
    repetition_rate: float | None = None,
    pre_silence: float = 0.0,
    post_silence: float = 0.0,
) -> ToneBurstResult:
    r"""Generate an IEC 60268-1 tone burst (single or repetitive).

    IEC 60268-1:1985, Clause A2.1: "The burst should start at the
    zero-crossing of the [...] tone and should consist of an integral
    number of full periods." The burst is a sine of ``cycles`` full
    periods gated by a rectangular envelope; with ``repetition_rate`` a
    train of ``repetitions`` identical bursts is produced, one per
    repetition period, as in the repetitive-burst test of Clause A2.2
    (there: 5 ms bursts of 5 kHz tone at 2, 10 or 100 bursts per second).

    The gate closes after ``round(fs * cycles / frequency)`` samples, so
    the "integral number of full periods" is sample-exact only when
    ``fs`` and ``frequency`` are commensurate (``fs * cycles/frequency``
    an integer). Otherwise the gate closes up to half a sample away from
    the tone's final zero crossing and the gated waveform carries a
    residual step of up to
    :math:`\text{amplitude} \cdot \sin(\pi \cdot \text{frequency}/f_\mathrm{s})`
    there
    (e.g. 10 cycles of 997 Hz at 48 kHz span 481.44 samples, gated at
    481); a :class:`~phonometry.PhonometryWarning` quantifies the
    realized residual.

    :param fs: Rate to generate at, in Hz.
    :param frequency: Tone frequency, in Hz (below the Nyquist rate).
    :param cycles: Full tone periods per burst (positive integer).
    :param amplitude: Peak amplitude of the tone.
    :param repetitions: Number of bursts (requires ``repetition_rate``
        when greater than 1).
    :param repetition_rate: Bursts per second; each burst then occupies
        one full repetition period (burst plus silence). ``None`` (the
        default) produces a single burst with no trailing period.
    :param pre_silence: Silence before the first burst, in seconds.
    :param post_silence: Silence after the last burst (or after the last
        repetition period), in seconds.
    :return: A :class:`ToneBurstResult`.
    :raises ValueError: If the inputs or parameters are invalid.
    """
    fs_v, f_v, cycles_v, amplitude_v, repetitions_v = _tone_burst_scalars(
        fs,
        frequency,
        cycles,
        amplitude,
        repetitions,
        pre_silence,
        post_silence,
    )

    burst_seconds = cycles_v / f_v
    exact_samples = fs_v * burst_seconds
    n_on = round(exact_samples)
    if n_on < _MIN_RECORD_SAMPLES:
        msg = "The burst is shorter than 2 samples; increase 'cycles' or 'fs'."
        raise ValueError(msg)
    rate_v, period, duty = _tone_burst_period(
        fs_v, n_on, repetitions_v, repetition_rate
    )

    # Warn only once every configuration check has passed, so an invalid
    # repetition setup raises instead of warning first (which would mask
    # the error under a warnings-as-errors filter).
    delta = n_on - exact_samples  # gate-close offset from the zero crossing
    if abs(delta) > _GATE_ALIGN_TOL:
        residual = amplitude_v * math.sin(2.0 * math.pi * f_v * delta / fs_v)
        warnings.warn(
            f"'frequency' = {f_v:g} Hz is incommensurate with fs = "
            f"{fs_v:g} Hz: {cycles_v} full periods span {exact_samples:.6g} "
            f"samples, gated at {n_on}. The gate closes {abs(delta):.3g} "
            "samples away from the tone's final zero crossing, leaving a "
            f"residual step of {residual:.3g} (peak amplitude "
            f"{amplitude_v:g}).",
            PhonometryWarning,
            stacklevel=2,
        )

    n_pre = round(float(pre_silence) * fs_v)
    n_post = round(float(post_silence) * fs_v)
    block = period if period is not None else n_on
    n_total = n_pre + repetitions_v * block + n_post

    # Clause A2.1: start at the zero crossing, integral number of full
    # periods. sin(2πf·m/fs) starts at 0, positive-going, and the underlying
    # continuous tone returns to a zero crossing exactly at t = cycles/f.
    m = np.arange(n_on, dtype=np.float64)
    burst = amplitude_v * np.sin(2.0 * np.pi * f_v * m / fs_v)

    signal = np.zeros(n_total, dtype=np.float64)
    envelope = np.zeros(n_total, dtype=np.float64)
    for k in range(repetitions_v):
        start = n_pre + k * block
        signal[start : start + n_on] = burst
        envelope[start : start + n_on] = amplitude_v

    return ToneBurstResult(
        signal=signal,
        envelope=envelope,
        fs=fs_v,
        frequency=f_v,
        cycles=cycles_v,
        amplitude=amplitude_v,
        burst_seconds=burst_seconds,
        burst_samples=n_on,
        onset_sample=n_pre,
        repetitions=repetitions_v,
        repetition_rate=rate_v,
        period_samples=period,
        duty_cycle=duty,
    )


# ---------------------------------------------------------------------------
# Polyphase resampling with an explicit anti-alias specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResampledSignalResult:
    r"""Resampled record with the designed anti-alias filter and its spec.

    The polyphase resampler filters at the intermediate rate
    ``fs_original * up`` with a linear-phase Kaiser-window lowpass designed
    from the two numbers below; the filter taps are returned so the spec
    can be verified against the filter itself.

    :ivar signal: The resampled record.
    :ivar fs: Sample rate of :attr:`signal`, in Hz.
    :ivar original_fs: Sample rate of the input, in Hz.
    :ivar up: Interpolation factor of the rational ratio ``up/down``.
    :ivar down: Decimation factor of the rational ratio ``up/down``.
    :ivar filter_taps: Anti-alias FIR taps (unit passband gain; the
        polyphase engine applies the ``up`` interpolation gain), designed
        at the intermediate rate ``original_fs * up``. A single ``1.0`` tap
        when the ratio is 1 (no filtering).
    :ivar passband_edge_hz: Passband edge of the design, in Hz.
    :ivar stopband_edge_hz: Stopband edge of the design (the smaller of
        the two Nyquist frequencies), in Hz.
    :ivar stopband_attenuation_db: Designed stopband attenuation, in dB
        (also the passband ripple bound: the Kaiser window method holds
        the ripple of both bands within the same
        :math:`\delta = 10^{-A/20}`,
        though -- unlike a true equiripple design -- its ripple decays
        away from the band edges rather than staying at the bound).
    :ivar transition_width: Transition-band width as a fraction of the
        smaller Nyquist frequency.
    """

    signal: NDArray[np.float64]
    fs: float
    original_fs: float
    up: int
    down: int
    filter_taps: NDArray[np.float64]
    passband_edge_hz: float
    stopband_edge_hz: float
    stopband_attenuation_db: float
    transition_width: float

    def __post_init__(self) -> None:
        """Reject a record or a filter that carries an axis nobody reads.

        The two arrays here measure nothing in common and are deliberately
        left unpinned against each other: :attr:`signal` runs over the
        resampled record, whose length follows the rate ratio, while
        :attr:`filter_taps` runs over the Kaiser design, whose length follows
        the transition width and the attenuation. A 24576-sample record
        beside a 1891-tap filter is the ordinary case, and any equality
        between the two would reject it.

        What both must be is flat. :attr:`n_taps` reports ``filter_taps.size``
        and the figure evaluates the magnitude response from the taps, so a
        second axis on either one is counted into the total and filtered as
        though it were more of the same filter -- a channel of a
        multichannel record read as extra samples, a bank of designs read as
        one long impulse response -- and both the reported tap count and the
        plotted spec come out of a filter that was never designed.

        :raises ValueError: if the record or the taps carry more than one
            axis.
        """
        require_ranks(self, signal=1, filter_taps=1)

    @property
    def n_taps(self) -> int:
        """Length of the designed anti-alias FIR."""
        return int(self.filter_taps.size)

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the delivered anti-alias filter against its design spec.

        The magnitude response of :attr:`filter_taps` with the passband
        edge, the stopband edge at the alias fold, the designed stopband
        attenuation and the rejected band shaded.

        :param ax: Existing axes, or ``None`` to create a figure.
        :param language: Label language, ``"en"`` (default) or ``"es"``.
        :param kwargs: Forwarded to the magnitude ``plot`` call.
        :return: The axes.
        """
        from .._i18n import check_language
        from .._plot.signals import plot_resampled_signal

        check_language(language)
        return plot_resampled_signal(self, ax=ax, language=language, **kwargs)


def resample_signal(
    x: Signal | NDArray[np.float64] | list[float],
    fs: float | None = None,
    *,
    fs_new: float,
    stopband_attenuation_db: float = 120.0,
    transition_width: float = 0.05,
    max_denominator: int = 1000,
) -> ResampledSignalResult:
    r"""Resample a record with a stated anti-alias specification.

    Polyphase rational resampling (:func:`scipy.signal.resample_poly`)
    behind a lowpass FIR designed *here* by the Kaiser window method: the
    stopband starts at the smaller of the two Nyquist frequencies and
    provides ``stopband_attenuation_db`` of alias rejection, the passband
    ends ``transition_width`` below it and is flat within the same ripple
    bound :math:`\delta = 10^{-A/20}`. The designed taps travel with the
    result, so
    the spec is a property of the returned filter, not of a library
    default.

    :param x: Input record, 1-D. Accepts a :class:`phonometry.io.Signal`, whose
        calibration is applied to the samples, so the resampled record comes
        out in Pa.
    :param fs: Sample rate of ``x``, in Hz. Required for a bare array; a
        :class:`~phonometry.io.Signal` brings its own, and an explicit value
        that disagrees with it raises instead of silently winning.
    :param fs_new: Target sample rate, in Hz. The ratio ``fs_new/fs``
        must be a rational number with denominator at most
        ``max_denominator`` (e.g. 48000/44100 = 160/147).
    :param stopband_attenuation_db: Anti-alias stopband attenuation, in
        dB (at least 30).
    :param transition_width: Transition-band width as a fraction of the
        smaller Nyquist frequency, in (0, 0.5].
    :param max_denominator: Largest denominator accepted for the rational
        rate ratio.
    :return: A :class:`ResampledSignalResult`.
    :raises ValueError: If the inputs or parameters are invalid, or if
        the rate ratio is not rational within ``max_denominator``.
    """
    from fractions import Fraction

    xa = _validate_1d_finite(x, "x")
    fs_v = _positive(resolve_fs(x, fs), "fs")
    fs_new_v = _positive(fs_new, "fs_new")
    atten = float(stopband_attenuation_db)
    if not np.isfinite(atten) or atten < _MIN_STOPBAND_ATTENUATION_DB:
        msg = "'stopband_attenuation_db' must be at least 30 dB."
        raise ValueError(msg)
    tw = float(transition_width)
    if not np.isfinite(tw) or not 0.0 < tw <= _MAX_TRANSITION_WIDTH:
        msg = "'transition_width' must be in (0, 0.5]."
        raise ValueError(msg)
    max_den = int(max_denominator)
    if max_den < 1:
        msg = "'max_denominator' must be a positive integer."
        raise ValueError(msg)

    ratio = Fraction(fs_new_v / fs_v).limit_denominator(max_den)
    up, down = ratio.numerator, ratio.denominator
    if up == 0 or abs(up / down - fs_new_v / fs_v) > 1e-9 * (fs_new_v / fs_v):
        msg = (
            f"'fs_new'/'fs' = {fs_new_v / fs_v!r} is not a rational ratio "
            f"with denominator <= {max_den}."
        )
        raise ValueError(msg)

    if up == down:  # Same rate: nothing to do, and nothing to filter.
        return ResampledSignalResult(
            signal=xa.copy(),
            fs=fs_new_v,
            original_fs=fs_v,
            up=1,
            down=1,
            filter_taps=np.ones(1, dtype=np.float64),
            passband_edge_hz=(1.0 - tw) * fs_v / 2.0,
            stopband_edge_hz=fs_v / 2.0,
            stopband_attenuation_db=atten,
            transition_width=tw,
        )

    from scipy import signal as sp_signal

    # Kaiser design at the intermediate rate fs·up: stopband from the
    # smaller Nyquist frequency (where aliases fold), passband up to
    # (1 - transition_width) of it, ripple bound 10^(-A/20) in both bands.
    # Kaiser's attenuation estimate is accurate to a few tenths of a dB,
    # so the design targets 1 dB past the request: the delivered filter
    # meets the stated spec, not an approximation of it.
    fs_up = fs_v * up
    f_nyq = min(fs_v, fs_new_v) / 2.0
    f_stop = f_nyq
    f_pass = (1.0 - tw) * f_nyq
    n_taps, beta = sp_signal.kaiserord(atten + 1.0, (f_stop - f_pass) / (fs_up / 2.0))
    n_taps += (n_taps + 1) % 2  # Odd length: type I linear phase.
    taps = np.asarray(
        sp_signal.firwin(
            n_taps, (f_pass + f_stop) / 2.0, window=("kaiser", beta), fs=fs_up
        ),
        dtype=np.float64,
    )
    resampled = np.asarray(
        sp_signal.resample_poly(xa, up, down, window=taps), dtype=np.float64
    )
    return ResampledSignalResult(
        signal=resampled,
        fs=fs_new_v,
        original_fs=fs_v,
        up=up,
        down=down,
        filter_taps=taps,
        passband_edge_hz=f_pass,
        stopband_edge_hz=f_stop,
        stopband_attenuation_db=atten,
        transition_width=tw,
    )


# ---------------------------------------------------------------------------
# Band-limited fractional delay (frequency-domain phase ramp)
# ---------------------------------------------------------------------------


def _fractional_advance(x: NDArray[np.float64], shift: float) -> NDArray[np.float64]:
    r"""Advance ``x`` by ``shift`` samples (band-limited, non-circular).

    Frequency-domain phase ramp
    :math:`e^{+j 2 \pi k \cdot \text{shift} / \text{nfft}}` over a record
    zero-padded past the shift, so the advanced samples leaving one end
    land in the padding instead of wrapping around. This is the alignment
    kernel of :func:`~phonometry.signals.correlation.align_impulse_responses`.
    """
    from scipy import fft as sp_fft

    n = x.size
    pad = int(np.ceil(abs(shift))) + 1
    nfft = int(sp_fft.next_fast_len(n + pad))
    spectrum = np.fft.rfft(x, n=nfft)
    freqs = np.fft.rfftfreq(nfft)
    shifted = np.fft.irfft(spectrum * np.exp(2j * np.pi * freqs * shift), n=nfft)
    return np.asarray(shifted[:n], dtype=np.float64)


@overload
def fractional_delay(
    x: Signal, delay: float, *, mode: Literal["linear", "circular"] = ...
) -> Signal: ...


@overload
def fractional_delay(
    x: NDArray[np.float64] | list[float],
    delay: float,
    *,
    mode: Literal["linear", "circular"] = ...,
) -> NDArray[np.float64]: ...


def fractional_delay(
    x: Signal | NDArray[np.float64] | list[float],
    delay: float,
    *,
    mode: Literal["linear", "circular"] = "linear",
) -> Signal | NDArray[np.float64]:
    r"""Delay a record by an arbitrary (sub-sample) number of samples.

    Band-limited delay via a frequency-domain phase ramp
    :math:`e^{-j 2 \pi k \cdot \text{delay} / N}`: every spectral
    component is delayed by exactly
    ``delay`` samples, i.e. its phase changes by
    :math:`-2 \pi f \cdot \text{delay} / f_\mathrm{s}`
    radians. Two boundary conventions:

    * ``'linear'`` (default): the record is zero-padded past the shift
      before the ramp, so samples leaving one end land in the padding
      instead of wrapping around - use it for transients and impulse
      responses. Content shifted beyond the record length is discarded
      (the output keeps the input length).
    * ``'circular'``: the ramp is applied over the record itself and the
      shift wraps around - use it for periodic records. For a tone
      centered on a DFT bin the delayed record equals the analytically
      delayed tone to machine precision.

    An integer ``delay`` in ``'linear'`` mode reduces to an exact sample
    shift with zero fill. Negative delays advance the record.

    A real record of even length cannot carry a fractionally delayed
    Nyquist-bin component (the inverse real FFT keeps its real part), so
    keep the signal band-limited below Nyquist - as any properly sampled
    signal is - or use odd lengths, and the operation is exact.

    :param x: Input record, 1-D. Accepts a :class:`phonometry.io.Signal`, whose
        calibration is applied to the samples, so the delayed record comes
        out in Pa.
    :param delay: Delay in samples (fractional and negative allowed);
        magnitude less than the record length.
    :param mode: Boundary convention, ``'linear'`` or ``'circular'``.
    :return: The delayed record, same length as ``x``.
    :raises ValueError: If the inputs or parameters are invalid.
    """
    xa = _validate_1d_finite(x, "x")
    delay_v = float(delay)
    if not np.isfinite(delay_v):
        msg = "'delay' must be a finite number."
        raise ValueError(msg)
    if abs(delay_v) >= xa.size:
        msg = (
            "'delay' magnitude must be smaller than the record length "
            f"({xa.size} samples)."
        )
        raise ValueError(msg)
    if mode not in ("linear", "circular"):
        msg = "'mode' must be 'linear' or 'circular'."
        raise ValueError(msg)
    if mode == "linear":
        # A delay is a negative advance; float negation is exact, so the
        # phase ramp is bit-identical to the alignment kernel's.
        return like_input(x, _fractional_advance(xa, -delay_v))
    n = xa.size
    freqs = np.fft.rfftfreq(n)
    delayed = np.fft.irfft(np.fft.rfft(xa) * np.exp(-2j * np.pi * freqs * delay_v), n=n)
    return like_input(x, np.asarray(delayed, dtype=np.float64))
