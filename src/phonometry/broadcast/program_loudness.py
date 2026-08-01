#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Programme loudness and true-peak level (ITU-R BS.1770-5, EBU R 128).

Implements the objective multichannel loudness measurement algorithm of
ITU-R BS.1770-5 Annex 1 and its companions:

* **K-weighting** (Annex 1): the two-stage pre-filter (a shelving filter
  modelling the acoustic effect of a spherical head, then the RLB high-pass),
  specified as second-order sections with tabulated coefficients at 48 kHz
  (Tables 1 and 2) and re-derived here for other rates so the frequency
  response is preserved, as the Recommendation requires.
* **Programme (integrated) loudness** (Annex 1): mean-square power in gating
  blocks of 400 ms overlapping 75 %, channel-weighted summation (surround
  +1.5 dB, LFE excluded, Table 3), and the two-stage gate: an absolute
  threshold at -70 LKFS and a relative threshold 10 LU below the
  absolute-gated loudness (Formulae 3-7).
* **Channel weights for advanced sound systems** (Annex 3): the
  position-dependent weighting :math:`G_i` from the azimuth and elevation of
  each loudspeaker (Table 4), covering the BS.2051 configurations (Table 5).
* **True-peak level** (Annex 2): the inter-sample peak in dBTP estimated by
  oversampling to at least 192 kHz before taking the absolute maximum.
* **EBU Mode momentary and short-term loudness** (EBU Tech 3341): the
  ungated sliding-window loudness over 400 ms (M) and 3 s (S), with their
  maxima; EBU R 128 normalises Programme Loudness to -23.0 LUFS and limits
  the true peak to -1 dBTP.
* **Loudness range** (EBU Tech 3342): the 10th-to-95th percentile spread of
  the short-term loudness distribution after a cascaded gate (absolute
  -70 LUFS, then relative -20 LU, deliberately different from the -10 LU of
  the integrated measure).

The whole module is validated against the EBU Tech 3341 and Tech 3342
"minimum requirements" test signals with their official tolerances, and
against the Recommendation's own 997 Hz anchor (a 0 dB FS sine on one front
channel reads -3.01 LKFS).

Loudness values are returned in LUFS (the EBU name; identical to the LKFS of
ITU-R BS.1770). 1 LU is equivalent to 1 dB.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike
from scipy import signal

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .._report.metadata import ReportMetadata

from .._internal.peaks import inter_sample_peak
from .._internal.types import as_float_or_array
from .._internal.utils import _typesignal
from .._internal.validation import require_positive

_EMPTY_SIGNAL = "Input signal 'x' cannot be empty."

# ---------------------------------------------------------------------------
# Normative constants (ITU-R BS.1770-5 Annex 1, EBU Tech 3341/3342).
# ---------------------------------------------------------------------------

#: The constant of Formula 2 that cancels the K-weighting gain at 997 Hz.
_K_OFFSET = -0.691

#: Absolute gating threshold of Formula 6, in LUFS (LKFS).
ABSOLUTE_GATE = -70.0

#: Relative gating offset of Formula 7 for the integrated loudness, in LU.
RELATIVE_GATE = -10.0

#: Gating-block duration ``Tg`` (Formula 3), in seconds.
BLOCK_DURATION = 0.4

#: Gating-block overlap (Formula 3), as a fraction of the block duration.
BLOCK_OVERLAP = 0.75

#: Short-term window length (EBU Tech 3341 section 2.2), in seconds.
SHORT_TERM_DURATION = 3.0

#: Relative gating offset of the loudness range (EBU Tech 3342), in LU.
LRA_RELATIVE_GATE = -20.0

#: Loudness-range percentiles (EBU Tech 3342): (lower, upper), in percent.
LRA_PERCENTILES = (10.0, 95.0)

#: The oversampled rate that the true-peak estimate targets (Annex 2), Hz.
_TRUE_PEAK_RATE = 192_000.0

#: Sample rate the Annex 1 coefficient tables are given for, in Hz.
_TABLE_RATE = 48_000.0

#: Stage 1 of the K-weighting pre-filter (spherical-head shelf): the
#: (b, a) coefficients of Table 1, valid at 48 kHz.
_STAGE1_B48 = (1.53512485958697, -2.69169618940638, 1.19839281085285)
_STAGE1_A48 = (1.0, -1.69065929318241, 0.73248077421585)

#: Stage 2 of the K-weighting pre-filter (RLB high-pass): the (b, a)
#: coefficients of Table 2, valid at 48 kHz.
_STAGE2_B48 = (1.0, -2.0, 1.0)
_STAGE2_A48 = (1.0, -1.99004745483398, 0.99007225036621)

#: Channel weights of Table 3 by channel count: mono, stereo (L/R),
#: 3/2 multichannel (L/R/C/Ls/Rs) and 5.1 (L/R/C/LFE/Ls/Rs; the LFE channel
#: is excluded from the measurement, hence its zero weight).
DEFAULT_CHANNEL_WEIGHTS: dict[int, tuple[float, ...]] = {
    1: (1.0,),
    2: (1.0, 1.0),
    5: (1.0, 1.0, 1.0, 1.41, 1.41),
    6: (1.0, 1.0, 1.0, 0.0, 1.41, 1.41),
}


# ---------------------------------------------------------------------------
# K-weighting (Annex 1, Tables 1-2).
# ---------------------------------------------------------------------------


def _analog_from_biquad(coeffs: tuple[float, float, float], rate: float) -> np.ndarray:
    """Map a digital biquad polynomial to its bilinear analog prototype.

    Substituting :math:`z^{-1} = (K - s)/(K + s)` with :math:`K = 2 f_s`
    (the inverse of the bilinear transform) into
    :math:`c_0 + c_1 z^{-1} + c_2 z^{-2}` gives the analog quadratic
    :math:`(c_0 - c_1 + c_2)\\,s^2 + 2K(c_0 - c_2)\\,s + K^2(c_0 + c_1 + c_2)`.

    :param coeffs: The digital polynomial coefficients ``(c0, c1, c2)``.
    :param rate: The sample rate the digital coefficients are valid at, Hz.
    :return: The analog polynomial ``[s^2, s^1, s^0]`` coefficients.
    """
    c0, c1, c2 = coeffs
    k = 2.0 * rate
    return np.array([c0 - c1 + c2, k * (2.0 * (c0 - c2)), k * k * (c0 + c1 + c2)])


def _redesign_biquad(
    b48: tuple[float, float, float], a48: tuple[float, float, float], fs: float
) -> tuple[np.ndarray, np.ndarray]:
    """Re-derive a Table 1/2 biquad for a sample rate other than 48 kHz.

    The Recommendation tabulates the coefficients at 48 kHz only and requires
    other rates to "provide the same frequency response". The digital filter
    is mapped back to its analog prototype by inverting the bilinear
    transform at 48 kHz, then discretised again at the target rate; at
    48 kHz the round trip reproduces the tabulated coefficients exactly.

    :param b48: Numerator coefficients at 48 kHz.
    :param a48: Denominator coefficients at 48 kHz.
    :param fs: Target sample rate, Hz.
    :return: The ``(b, a)`` coefficients at ``fs``.
    """
    num = _analog_from_biquad(b48, _TABLE_RATE)
    den = _analog_from_biquad(a48, _TABLE_RATE)
    b, a = signal.bilinear(num, den, fs=fs)
    return np.asarray(b, dtype=np.float64), np.asarray(a, dtype=np.float64)


def k_weighting_coefficients(
    fs: float,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """K-weighting biquad coefficients (BS.1770-5 Annex 1, Tables 1-2).

    At 48 kHz the tabulated values are returned verbatim; at any other rate
    the biquads are re-derived through their analog prototypes so the
    frequency response matches the 48 kHz specification, as the
    Recommendation requires. At 32 kHz and above the redesigned response
    stays within 0.02 dB of the specification across the audio band; at
    16 kHz the bilinear warping near Nyquist grows to about 0.13 dB while
    the 997 Hz anchor still holds within 0.03 LU. Below 16 kHz the warping
    would break the +/-0.1 LU metering tolerance, so such rates are
    rejected.

    :param fs: Sample rate, Hz (16 kHz or higher).
    :return: ``(stage1, stage2)``, each a ``(b, a)`` coefficient pair: the
        spherical-head shelving filter and the RLB high-pass filter.
    """
    fs = require_positive(float(fs), "fs")
    if fs < 16000.0:
        raise ValueError(
            "the K-weighting redesign requires fs >= 16000 Hz; below that "
            "the bilinear warping no longer preserves the specified response "
            f"within the metering tolerance; got fs={fs:g}."
        )
    if fs == _TABLE_RATE:
        return (
            (np.array(_STAGE1_B48), np.array(_STAGE1_A48)),
            (np.array(_STAGE2_B48), np.array(_STAGE2_A48)),
        )
    return (
        _redesign_biquad(_STAGE1_B48, _STAGE1_A48, fs),
        _redesign_biquad(_STAGE2_B48, _STAGE2_A48, fs),
    )


def k_weighting(x: list[float] | np.ndarray, fs: float) -> np.ndarray:
    """Apply the two-stage K-weighting pre-filter (BS.1770-5 Annex 1).

    Stage 1 models the acoustic effect of the head as a rigid sphere (a
    high-frequency shelf of about +4 dB); stage 2 is the RLB revised
    low-frequency B-curve high-pass. Their concatenation is the K-weighting.

    :param x: Input signal (1D or 2D ``[channels, samples]``), linear units.
    :param fs: Sample rate, Hz.
    :return: The K-weighted signal, same shape as the input.
    """
    x_proc = _typesignal(x)
    if x_proc.shape[-1] == 0:
        raise ValueError(_EMPTY_SIGNAL)
    (b1, a1), (b2, a2) = k_weighting_coefficients(fs)
    return np.asarray(
        signal.lfilter(b2, a2, signal.lfilter(b1, a1, x_proc, axis=-1), axis=-1)
    )


@dataclass(frozen=True)
class KWeightingResponse:
    """Magnitude frequency response of the K-weighting pre-filter (BS.1770-5).

    The two-stage K-weighting of Annex 1 evaluated as a transfer function: the
    spherical-head high-frequency shelf (stage 1, about +4 dB above 2 kHz), the
    RLB high-pass (stage 2) and their combination, each as a magnitude in dB
    over a shared logarithmic frequency grid. It is the frequency-domain view of
    the same biquads that :func:`k_weighting` applies in the time domain, built
    from the tabulated (or redesigned) coefficients of :func:`k_weighting_coefficients`.

    Build it with :func:`k_weighting_response`; the frozen instance then exposes
    :meth:`plot` (the magnitude response versus frequency).

    :ivar frequencies: Frequency grid, in Hz (log-spaced up to the Nyquist rate
        by default).
    :ivar magnitude_db: Combined K-weighting magnitude response, in dB (the
        product of the two stages). It tends to the +4 dB shelf plateau at high
        frequency and rolls off below a few hundred hertz.
    :ivar shelf_db: Stage 1 (spherical-head shelf) magnitude response, in dB.
    :ivar highpass_db: Stage 2 (RLB high-pass) magnitude response, in dB.
    :ivar fs: Sample rate the response was evaluated at, in Hz.
    """

    frequencies: np.ndarray
    magnitude_db: np.ndarray
    shelf_db: np.ndarray
    highpass_db: np.ndarray
    fs: float

    def plot(self, ax: Axes | None = None, *, language: str = "en",
             **kwargs: Any) -> Axes:
        """Plot the K-weighting magnitude response versus frequency.

        Draws the combined K-weighting magnitude (dB) on a logarithmic
        frequency axis, with the two stages (the +4 dB shelf and the RLB
        high-pass) as light companion curves.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes` and never calls ``plt.show``.

        :param ax: Existing axes, or ``None`` to create a figure.
        :param language: Label language, ``"en"`` (default) or ``"es"``.
        :param kwargs: Forwarded to the combined-curve ``plot`` call.
        :return: The axes.
        """
        from .._i18n import check_language
        from .._plot.broadcast import plot_k_weighting_response

        check_language(language)
        return plot_k_weighting_response(self, ax=ax, language=language, **kwargs)


def k_weighting_response(
    fs: float = _TABLE_RATE,
    *,
    frequencies: ArrayLike | None = None,
    n: int = 512,
) -> KWeightingResponse:
    """K-weighting magnitude frequency response (BS.1770-5 Annex 1, Tables 1-2).

    Evaluates the two-stage K-weighting pre-filter as a transfer function and
    returns its magnitude in dB. The biquad coefficients come from
    :func:`k_weighting_coefficients` (the verbatim 48 kHz tables, or the
    analog-prototype redesign at other rates), so the response is the exact
    frequency-domain counterpart of the filter :func:`k_weighting` applies; no
    coefficients are re-derived here. Each stage is evaluated with
    :func:`scipy.signal.freqz` and their magnitudes summed in dB.

    The combined response tends to the +4 dB shelf plateau of the spherical-head
    stage at high frequency and rolls off below a few hundred hertz through the
    RLB high-pass; near 1 kHz it passes through the ~0.69 dB gain that the
    ``-0.691`` constant of Formula 2 cancels.

    :param fs: Sample rate the response is evaluated at, Hz (16 kHz or higher,
        as :func:`k_weighting_coefficients` requires; default 48 kHz).
    :param frequencies: Explicit frequency grid, in Hz (each in the half-open
        interval ``(0, fs/2]``); ``None`` (the default) uses ``n``
        log-spaced points from 10 Hz to the Nyquist rate.
    :param n: Number of log-spaced points when ``frequencies`` is ``None``
        (default 512); ignored otherwise.
    :return: A frozen :class:`KWeightingResponse`.
    :raises ValueError: If ``fs`` is below 16 kHz, ``n`` is not positive, or a
        supplied frequency is outside ``(0, fs/2]``.
    """
    fs = require_positive(float(fs), "fs")
    (b1, a1), (b2, a2) = k_weighting_coefficients(fs)
    nyquist = fs / 2.0
    if frequencies is None:
        if n < 1:
            raise ValueError(f"n must be a positive integer; got {n}.")
        freqs = np.logspace(np.log10(10.0), np.log10(nyquist), int(n))
    else:
        freqs = np.asarray(frequencies, dtype=np.float64).ravel()
        if freqs.size == 0:
            raise ValueError("'frequencies' cannot be empty.")
        if np.any(freqs <= 0.0) or np.any(freqs > nyquist):
            raise ValueError(
                f"every frequency must lie in (0, fs/2] = (0, {nyquist:g}] Hz."
            )
    shelf_db = 20.0 * np.log10(np.abs(signal.freqz(b1, a1, worN=freqs, fs=fs)[1]))
    highpass_db = 20.0 * np.log10(np.abs(signal.freqz(b2, a2, worN=freqs, fs=fs)[1]))
    return KWeightingResponse(
        frequencies=freqs,
        magnitude_db=shelf_db + highpass_db,
        shelf_db=shelf_db,
        highpass_db=highpass_db,
        fs=fs,
    )


# ---------------------------------------------------------------------------
# Channel weights (Annex 1 Table 3, Annex 3 Table 4).
# ---------------------------------------------------------------------------


def channel_weight(
    azimuth: ArrayLike, elevation: ArrayLike = 0.0
) -> float | np.ndarray:
    """Position-dependent channel weight ``Gi`` (BS.1770-5 Annex 3, Table 4).

    Extends the Table 3 weights to arbitrarily placed loudspeakers of the
    advanced sound systems (Recommendation ITU-R BS.2051): a mid-layer
    loudspeaker to the side of the listener weighs 1.41 (+1.5 dB), every
    other position 1.0. The LFE channels are excluded from the measurement
    altogether (use a weight of 0 for them).

    :param azimuth: Azimuth angle(s) of the loudspeaker position, in degrees
        (0 in front, positive to either side; only the magnitude matters).
    :param elevation: Elevation angle(s), in degrees.
    :return: The weight ``Gi``: 1.41 where ``|elevation| < 30`` and
        ``60 <= |azimuth| <= 120``, else 1.0. Float for scalar inputs.
    """
    theta = np.abs(np.asarray(azimuth, dtype=np.float64))
    phi = np.abs(np.asarray(elevation, dtype=np.float64))
    if not (np.all(np.isfinite(theta)) and np.all(np.isfinite(phi))):
        raise ValueError("azimuth and elevation must be finite.")
    if np.any(theta > 360.0) or np.any(phi > 90.0):
        raise ValueError(
            "azimuth must be within +/-360 deg and elevation within +/-90 deg."
        )
    theta = np.where(theta > 180.0, 360.0 - theta, theta)
    side = (theta >= 60.0) & (theta <= 120.0) & (phi < 30.0)
    return as_float_or_array(np.where(side, 1.41, 1.0))


def _resolve_weights(n_channels: int, weights: ArrayLike | None) -> np.ndarray:
    """Return the per-channel weights, defaulting by channel count."""
    if weights is None:
        try:
            return np.array(DEFAULT_CHANNEL_WEIGHTS[n_channels])
        except KeyError:
            raise ValueError(
                f"no default channel weighting for {n_channels} channels: "
                "Table 3 covers mono, stereo, 3/2 multichannel (L/R/C/Ls/Rs) "
                "and 5.1 (L/R/C/LFE/Ls/Rs). Pass explicit weights (see "
                "channel_weight for the Annex 3 position-dependent values; "
                "use 0.0 to exclude an LFE channel)."
            ) from None
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or w.size != n_channels:
        raise ValueError(
            f"weights must be a 1D sequence with one entry per channel "
            f"({n_channels}); got shape {w.shape}."
        )
    if not np.all(np.isfinite(w)):
        raise ValueError("channel weights must be finite.")
    if np.any(w < 0.0):
        raise ValueError("channel weights must be non-negative.")
    return w


# ---------------------------------------------------------------------------
# Sliding-window loudness machinery (Formulae 1-7).
# ---------------------------------------------------------------------------


def _windowed_loudness(
    csum: np.ndarray, weights: np.ndarray, n_window: int, step: int
) -> tuple[np.ndarray, np.ndarray]:
    r"""Channel-weighted loudness of every full sliding window.

    :param csum: Per-channel cumulative sum of the squared K-weighted signal,
        shape ``(channels, samples + 1)``.
    :param weights: Per-channel weights ``Gi``.
    :param n_window: Window length in samples.
    :param step: Hop between consecutive windows in samples.
    :return: ``(loudness, end_samples)``: the loudness
        :math:`-0.691 + 10 \log_{10} (\sum_i G_i z_i)` of each window in LUFS,
        and the sample index of each window's end. Empty arrays when no full
        window fits.
    """
    n_samples = csum.shape[-1] - 1
    if n_window > n_samples:
        return np.empty(0), np.empty(0, dtype=np.intp)
    starts = np.arange(0, n_samples - n_window + 1, step, dtype=np.intp)
    ends = starts + n_window
    mean_square = (csum[:, ends] - csum[:, starts]) / float(n_window)
    total = weights @ mean_square
    with np.errstate(divide="ignore"):
        loudness = _K_OFFSET + 10.0 * np.log10(total)
    return loudness, ends


def _gated_power(block_power: np.ndarray, gate: np.ndarray) -> float:
    """Mean of ``block_power[gate]``, or 0.0 when the gate empties it."""
    if not np.any(gate):
        return 0.0
    return float(np.mean(block_power[gate]))


def _power_to_loudness(power: float) -> float:
    r""":math:`-0.691 + 10 \log_{10} \text{power}` with a -inf guard for zero power."""
    if power <= 0.0:
        return float("-inf")
    return _K_OFFSET + 10.0 * float(np.log10(power))


def _integrated_from_blocks(block_loudness: np.ndarray) -> tuple[float, float]:
    """Two-stage gated loudness of the gating blocks (Formulae 5-7).

    :param block_loudness: Loudness of the 400 ms gating blocks, LUFS.
    :return: ``(integrated, relative_threshold)``, both in LUFS (-inf when
        every block falls below the absolute gate).
    """
    with np.errstate(divide="ignore"):
        power = 10.0 ** ((block_loudness - _K_OFFSET) / 10.0)
    absolute = block_loudness > ABSOLUTE_GATE
    relative_threshold = (
        _power_to_loudness(_gated_power(power, absolute)) + RELATIVE_GATE
    )
    gate = absolute & (block_loudness > relative_threshold)
    return _power_to_loudness(_gated_power(power, gate)), relative_threshold


def _matlab_percentile(sorted_values: np.ndarray, percentile: float) -> float:
    """Nearest-rank percentile with the Tech 3342 reference indexing.

    The reference implementation picks ``sorted[round((n-1)*p/100 + 1)]``
    (1-based, MATLAB ``round`` = half away from zero).
    """
    n = sorted_values.size
    index = int(np.floor((n - 1) * percentile / 100.0 + 0.5))
    return float(sorted_values[index])


def _lra_distribution(short_term_loudness: np.ndarray) -> np.ndarray:
    """The sorted short-term readings surviving the Tech 3342 cascaded gate."""
    stl = short_term_loudness[short_term_loudness >= ABSOLUTE_GATE]
    if stl.size == 0:
        return stl
    mean_power = float(np.mean(10.0 ** (stl / 10.0)))
    threshold = 10.0 * float(np.log10(mean_power)) + LRA_RELATIVE_GATE
    return np.sort(stl[stl >= threshold])


def loudness_range(short_term_loudness: ArrayLike) -> float:
    """Loudness range LRA of a short-term loudness series (EBU Tech 3342).

    The input is the ungated short-term loudness (3 s sliding window,
    computed at 10 Hz or faster, as specified in ITU-R BS.1770). A cascaded
    gate first drops readings below the absolute threshold (-70 LUFS), then
    readings more than 20 LU below the power mean of what survived; the LRA
    is the spread between the 10th and 95th percentiles of the remaining
    distribution, following the Tech 3342 reference implementation.

    :param short_term_loudness: Short-term loudness readings, LUFS.
    :return: The loudness range, in LU (0.0 when no reading survives the
        gate).
    """
    stl = _lra_distribution(np.asarray(short_term_loudness, dtype=np.float64).ravel())
    if stl.size == 0:
        return 0.0
    low, high = LRA_PERCENTILES
    return _matlab_percentile(stl, high) - _matlab_percentile(stl, low)


# ---------------------------------------------------------------------------
# True-peak level (Annex 2).
# ---------------------------------------------------------------------------


def true_peak_level(
    x: list[float] | np.ndarray, fs: float, oversample: int | None = None
) -> float | np.ndarray:
    """True-peak level in dBTP (BS.1770-5 Annex 2).

    Estimates the inter-sample peak by oversampling the signal to at least
    192 kHz with a polyphase FIR interpolator before taking the absolute
    maximum (the same machinery behind
    :func:`phonometry.signal.levels.lc_peak`). At 48 kHz this is the
    4-times oversampling of the Annex 2 block diagram; higher input rates
    need proportionately less. The initial 12.04 dB attenuation of the
    Annex 2 integer pipeline is unnecessary in floating point and omitted.

    dBTP is referred to 100 % full scale: a sample value of 1.0 is 0 dBTP,
    and inter-sample peaks above full scale give positive values.

    :param x: Input signal (1D or 2D ``[channels, samples]``), full-scale
        units (1.0 = 0 dBFS).
    :param fs: Sample rate, Hz.
    :param oversample: Integer oversampling factor >= 1, or ``None`` (the
        default) for the smallest factor whose oversampled rate reaches
        192 kHz (4 at 48 kHz, 2 at 96 kHz, 1 at 192 kHz and above, matching
        the Annex 2 guidance that higher input rates need proportionately
        less oversampling).
    :return: The true-peak level in dBTP: a float for 1D input, an array of
        shape ``(channels,)`` for 2D input.
    """
    fs = require_positive(float(fs), "fs")
    if oversample is None:
        oversample = max(1, ceil(_TRUE_PEAK_RATE / fs))
    if (
        isinstance(oversample, bool)
        or not isinstance(oversample, (int, np.integer))
        or oversample < 1
    ):
        raise ValueError("oversample must be an integer >= 1.")
    x_proc = _typesignal(x)
    if x_proc.shape[-1] == 0:
        raise ValueError(_EMPTY_SIGNAL)
    peak = inter_sample_peak(x_proc, int(oversample))
    with np.errstate(divide="ignore"):
        out = 20.0 * np.log10(peak)
    return as_float_or_array(out)


# ---------------------------------------------------------------------------
# Programme loudness (Annex 1 + EBU Mode).
# ---------------------------------------------------------------------------


def integrated_loudness(
    x: list[float] | np.ndarray,
    fs: float,
    weights: ArrayLike | None = None,
) -> float:
    """Programme (integrated) loudness in LUFS (BS.1770-5 Annex 1).

    K-weights each channel, averages the power over gating blocks of 400 ms
    overlapping 75 %, sums the channels with their weights ``Gi`` and
    applies the two-stage gate: blocks below -70 LKFS are dropped, a
    relative threshold 10 LU below the loudness of the survivors is
    computed, and the loudness of the blocks above both thresholds is the
    programme loudness (Formulae 3-7). This is the quantity EBU R 128
    normalises to -23.0 LUFS.

    :param x: Input signal (1D mono or 2D ``[channels, samples]``),
        full-scale units (1.0 = 0 dBFS).
    :param fs: Sample rate, Hz.
    :param weights: Per-channel weights ``Gi``, or ``None`` for the Table 3
        defaults by channel count (see
        :data:`DEFAULT_CHANNEL_WEIGHTS` and :func:`channel_weight`).
    :return: The programme loudness, LUFS (``-inf`` when the signal is
        shorter than one gating block or entirely below the absolute gate).
    """
    x_proc, w = _prepare_signal(x, weights)
    csum = _power_cumsum(k_weighting(x_proc, fs))
    n_block, step = _block_geometry(fs)
    block_loudness, _ = _windowed_loudness(csum, w, n_block, step)
    return _integrated_from_blocks(block_loudness)[0]


def _prepare_signal(
    x: list[float] | np.ndarray, weights: ArrayLike | None
) -> tuple[np.ndarray, np.ndarray]:
    """Coerce to 2D float64, reject empty/non-finite input, resolve weights.

    A NaN or infinity anywhere in the signal would poison every gating
    block, empty the gate and silently read as digital silence (-inf), so
    non-finite input is rejected up front.
    """
    x_proc = np.atleast_2d(_typesignal(x))
    if x_proc.shape[-1] == 0:
        raise ValueError(_EMPTY_SIGNAL)
    if not np.all(np.isfinite(x_proc)):
        raise ValueError("Input signal 'x' must be finite (no NaN/inf samples).")
    return x_proc, _resolve_weights(x_proc.shape[0], weights)


def _power_cumsum(y: np.ndarray) -> np.ndarray:
    """Zero-prefixed cumulative sum of the squared signal, per channel."""
    power = y * y
    csum = np.empty((power.shape[0], power.shape[1] + 1))
    csum[:, 0] = 0.0
    np.cumsum(power, axis=-1, out=csum[:, 1:])
    return csum


def _block_geometry(fs: float) -> tuple[int, int]:
    """Gating-block length and hop in samples (400 ms, 75 % overlap)."""
    n_block = round(BLOCK_DURATION * fs)
    step = max(1, round(n_block * (1.0 - BLOCK_OVERLAP)))
    return n_block, step


@dataclass(frozen=True)
class ProgramLoudnessResult:
    """EBU Mode loudness measurement of a programme (BS.1770-5 / EBU R 128).

    :ivar integrated: Programme loudness ``I`` (gated, Annex 1), LUFS.
    :ivar loudness_range: Loudness range ``LRA`` (EBU Tech 3342), LU.
    :ivar true_peak: Maximum true-peak level over the channels, dBTP.
    :ivar momentary: Momentary loudness series ``M`` (400 ms, ungated), LUFS.
    :ivar momentary_time: Time of each ``M`` reading (window end), s.
    :ivar short_term: Short-term loudness series ``S`` (3 s, ungated), LUFS.
    :ivar short_term_time: Time of each ``S`` reading (window end), s.
    :ivar max_momentary: Maximum momentary loudness, LUFS.
    :ivar max_short_term: Maximum short-term loudness, LUFS.
    :ivar relative_threshold: Relative gating threshold of the integrated
        measurement (10 LU below the absolute-gated loudness), LUFS.
    :ivar lra_low: Lower (10th percentile) edge of the loudness range, LUFS.
    :ivar lra_high: Upper (95th percentile) edge of the loudness range, LUFS.
    :ivar true_peak_per_channel: True-peak level of each channel, dBTP.
    :ivar channel_weights: The channel weights ``Gi`` used.
    :ivar fs: Sample rate of the analysed signal, Hz.
    """

    integrated: float
    loudness_range: float
    true_peak: float
    momentary: np.ndarray
    momentary_time: np.ndarray
    short_term: np.ndarray
    short_term_time: np.ndarray
    max_momentary: float
    max_short_term: float
    relative_threshold: float
    lra_low: float
    lra_high: float
    true_peak_per_channel: np.ndarray
    channel_weights: np.ndarray
    fs: float

    def plot(self, ax: Axes | None = None, *, language: str = "en",
             **kwargs: Any) -> Axes:
        """Plot momentary and short-term loudness over time, with the
        integrated loudness and the loudness range annotated.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.

        :param language: Label language, ``"en"`` (default) or ``"es"``.
        """
        from .._i18n import check_language
        from .._plot.broadcast import plot_program_loudness

        check_language(language)
        return plot_program_loudness(self, ax=ax, language=language, **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
        tolerance: str = "qc",
    ) -> str:
        """Render an EBU R 128 programme-loudness compliance fiche to a PDF.

        Writes a one-page broadcast loudness-compliance sheet: the
        standard-basis line, an optional metadata header block, a full-width
        compliance table (integrated loudness and maximum true peak carry the
        verdict; the loudness range and the momentary/short-term maxima are
        informational), the result's own loudness-vs-time :meth:`plot`, the
        boxed ``I = X LUFS (LRA = Y LU, max TP = Z dBTP)`` result, a combined
        PASS/FAIL verdict row and a footer with the fixed disclaimer.

        :param path: Destination path of the PDF file.
        :param metadata: Optional
            :class:`~phonometry.ReportMetadata`; ``None`` produces a
            measurement fiche (compliance table, plot and verdict only). A
            supplied ``requirement`` is read as the target programme loudness
            in LUFS (defaulting to the EBU R 128 -23.0 LUFS).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: Accepted for a uniform signature; it has no effect on
            the single-layout programme-loudness fiche.
        :param language: Fiche language: ``"en"`` (default, English) or
            ``"es"`` (Spanish, with a comma decimal separator).
        :param tolerance: Programme-loudness tolerance rule of EBU R 128:
            ``"qc"`` (default) applies the +-0.2 LU allowance of item i) for
            measurement errors in loudness workflows such as Quality Control;
            ``"live"`` applies the +-1.0 LU tolerance of item h), permitted
            only where attaining the Target Level is not achievable
            practically (for example, live programmes). The fiche prints the
            applied rule and its R 128 item.
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is not ``"reportlab"`` or
            ``tolerance`` is not ``"qc"``/``"live"``.
        :raises ImportError: If reportlab is not installed
            (``pip install phonometry[report]``), or matplotlib is missing for
            the embedded figure (``pip install phonometry[plot]``).
        """
        from .._i18n import check_language

        check_language(language)
        if engine != "reportlab":
            raise ValueError(
                f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            )
        from .._report.broadcast import render_program_loudness_report

        return render_program_loudness_report(
            self, path, metadata=metadata, verbose=verbose, language=language,
            tolerance=tolerance,
        )


def _lra_edges(short_term: np.ndarray) -> tuple[float, float]:
    """The gated 10th/95th percentile edges of the LRA distribution, LUFS."""
    stl = _lra_distribution(short_term)
    if stl.size == 0:
        return float("-inf"), float("-inf")
    low, high = LRA_PERCENTILES
    return _matlab_percentile(stl, low), _matlab_percentile(stl, high)


def program_loudness(
    x: list[float] | np.ndarray,
    fs: float,
    weights: ArrayLike | None = None,
    *,
    momentary_step: float = 0.01,
    short_term_step: float = 0.1,
    oversample: int | None = None,
) -> ProgramLoudnessResult:
    """Measure a programme per EBU R 128: I, M, S, LRA and true peak.

    One call computes the full EBU Mode measurement set (EBU Tech 3341) of a
    finished programme:

    * the **integrated loudness** ``I`` with the BS.1770-5 two-stage gate
      (the quantity normalised to -23.0 LUFS by EBU R 128);
    * the ungated **momentary** (400 ms) and **short-term** (3 s) loudness
      series with their maxima;
    * the **loudness range** ``LRA`` from the short-term series with the
      Tech 3342 cascaded gate and 10-95 percentile spread; and
    * the **true-peak level** per channel and overall (Annex 2; EBU R 128
      permits at most -1 dBTP during production).

    :param x: Input signal (1D mono or 2D ``[channels, samples]``),
        full-scale units (1.0 = 0 dBFS).
    :param fs: Sample rate, Hz.
    :param weights: Per-channel weights ``Gi``, or ``None`` for the Table 3
        defaults by channel count (see :data:`DEFAULT_CHANNEL_WEIGHTS`; for
        other layouts derive the weights with :func:`channel_weight`).
    :param momentary_step: Hop of the momentary series, s (default 10 ms,
        a 100 Hz update rate; EBU Tech 3341 requires at least 10 Hz).
    :param short_term_step: Hop of the short-term series, s (default 100 ms,
        the 10 Hz minimum rate that Tech 3342 requires for the LRA input).
    :param oversample: True-peak oversampling factor; ``None`` picks the
        smallest factor reaching 192 kHz (4 at 48 kHz).
    :return: The frozen :class:`ProgramLoudnessResult`.
    """
    x_proc, w = _prepare_signal(x, weights)
    fs = require_positive(float(fs), "fs")
    momentary_step = require_positive(float(momentary_step), "momentary_step")
    short_term_step = require_positive(float(short_term_step), "short_term_step")
    if momentary_step > 0.1 or short_term_step > 0.1:
        raise ValueError(
            "momentary_step and short_term_step must be <= 0.1 s: EBU Tech "
            "3341 requires a meter update rate of at least 10 Hz, and Tech "
            "3342 needs it for the LRA input."
        )

    csum = _power_cumsum(k_weighting(x_proc, fs))
    n_block, block_step = _block_geometry(fs)

    # Integrated loudness from the normative gating blocks (75 % overlap).
    block_loudness, _ = _windowed_loudness(csum, w, n_block, block_step)
    integrated, relative_threshold = _integrated_from_blocks(block_loudness)

    # Momentary series: same 400 ms window, finer hop for metering fidelity.
    m_step = max(1, round(momentary_step * fs))
    momentary, m_ends = _windowed_loudness(csum, w, n_block, m_step)

    # Short-term series: 3 s window at >= 10 Hz (the Tech 3342 LRA input).
    n_short = round(SHORT_TERM_DURATION * fs)
    s_step = max(1, round(short_term_step * fs))
    short_term, s_ends = _windowed_loudness(csum, w, n_short, s_step)

    lra = loudness_range(short_term)
    lra_low, lra_high = _lra_edges(short_term)

    tp_channels = np.atleast_1d(
        np.asarray(true_peak_level(x_proc, fs, oversample), dtype=np.float64)
    )

    return ProgramLoudnessResult(
        integrated=integrated,
        loudness_range=lra,
        true_peak=float(np.max(tp_channels)),
        momentary=momentary,
        momentary_time=m_ends / fs,
        short_term=short_term,
        short_term_time=s_ends / fs,
        max_momentary=float(np.max(momentary)) if momentary.size else float("-inf"),
        max_short_term=float(np.max(short_term)) if short_term.size else float("-inf"),
        relative_threshold=relative_threshold,
        lra_low=lra_low,
        lra_high=lra_high,
        true_peak_per_channel=tp_channels,
        channel_weights=w,
        fs=fs,
    )
