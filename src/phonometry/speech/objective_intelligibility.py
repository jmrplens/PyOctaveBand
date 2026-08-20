#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Short-Time Objective Intelligibility (STOI and ESTOI).

Implements the two correlation-based objective intelligibility measures that
predict the intelligibility of time-frequency weighted noisy speech, where the
speech-transmission index (:mod:`phonometry.speech.sti`) and the speech
intelligibility index (:mod:`phonometry.speech.sii`) are less appropriate:

* **STOI** (Taal, Hendriks, Heusdens and Jensen 2011, *An Algorithm for
  Intelligibility Prediction of Time-Frequency Weighted Noisy Speech*, IEEE
  TASLP 19(7); short version Taal et al. 2010, ICASSP): the average, over
  short-time segments and one-third-octave bands, of the correlation between
  the clean and degraded short-time temporal envelopes, after a per-segment
  normalisation and a signal-to-distortion clipping.
* **ESTOI** (Jensen and Taal 2016, *An Algorithm for Predicting the
  Intelligibility of Speech Masked by Modulated Noise Maskers*, IEEE/ACM TASLP
  24(11)): the extended measure that mean- and variance-normalises the
  short-time band-by-frame spectrogram over both rows (band envelopes) and
  columns (spectra), so the intermediate index is a spectral correlation. It
  tracks intelligibility better under modulated maskers and competing talkers,
  where STOI's band-independent correlation overestimates.

Both measures share the same front end: resampling to 10 kHz, a 256-sample
(25.6 ms) Hann-windowed, 50 %-overlapping short-time Fourier transform
zero-padded to 512 points, removal of the clean-speech frames more than 40 dB
below the loudest clean frame, a 15-band one-third-octave grouping of the DFT
magnitudes from a lowest centre of 150 Hz, and 30-frame (384 ms) analysis
segments. The output is a scalar in roughly ``[0, 1]`` with a monotonic
relation to the fraction of correctly understood words: 1 for a degraded
signal identical to the clean one, and near 0 for uncorrelated noise.

The band grouping snaps each one-third-octave edge to the nearest DFT bin and
the analysis window is MATLAB's ``hanning`` (:func:`numpy.hanning` of length
``N+2`` with its zero end-points dropped), matching the reference MATLAB
implementation of the authors; ``pystoi`` reproduces the same conventions and
is used as an external cross-check in the test suite (it is not a runtime
dependency).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..io._resolve import apply_calibration, resolve_pair_fs

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

    from ..io._signal import Signal

# ---------------------------------------------------------------------------
# Algorithm constants (Taal et al. 2011 Section II; Jensen & Taal 2016 II-A).
# ---------------------------------------------------------------------------

#: Internal sample rate, in hertz (Taal et al. 2011 Section II).
SAMPLE_RATE: int = 10_000

#: Short-time frame length, in samples (256 = 25.6 ms at 10 kHz).
_FRAME: int = 256

#: FFT length after zero-padding (512 points).
_NFFT: int = 512

#: Number of one-third-octave bands (Taal et al. 2011 Section II).
_N_BANDS: int = 15

#: Centre frequency of the lowest one-third-octave band, in hertz.
_MIN_CENTER: float = 150.0

#: Frames per analysis segment (30 frames = 384 ms).
_N_SEGMENT: int = 30

#: Lower signal-to-distortion ratio bound for the clipping, in dB
#: (``beta`` in Taal et al. 2011 Eq. 3).
_SDR_DB: float = -15.0

#: Clean-speech dynamic range for silent-frame removal, in dB.
_DYN_RANGE_DB: float = 40.0

#: Machine epsilon guard used in the normalisations (avoids 0/0 on silence).
_EPS: float = float(np.finfo(np.float64).eps)


@dataclass(frozen=True)
class STOIResult:
    """Result of a STOI or ESTOI intelligibility computation.

    :ivar value: The overall intelligibility index (a scalar with a monotonic
        relation to the fraction of correctly understood words; 1 when the
        degraded signal equals the clean one).
    :ivar extended: ``True`` for ESTOI (Jensen & Taal 2016), ``False`` for
        STOI (Taal et al. 2011).
    :ivar segment_scores: Per-segment intermediate intelligibility (averaged
        over bands for STOI, the spectral correlation ``d_m`` for ESTOI).
    :ivar band_scores: Per-band mean intermediate correlation over the
        segments (STOI only; ``None`` for ESTOI, whose index mixes the bands).
    :ivar band_frequencies: The 15 one-third-octave band centre frequencies,
        in hertz.
    :ivar sample_rate: The internal sample rate the measure runs at (10 kHz).
    """

    value: float
    extended: bool
    segment_scores: NDArray[np.float64]
    band_scores: NDArray[np.float64] | None
    band_frequencies: NDArray[np.float64]
    sample_rate: int

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the intermediate intelligibility that averages to the index.

        For STOI this is the mean correlation per one-third-octave band; for
        ESTOI, whose index mixes the bands, it is the spectral correlation per
        analysis segment. Requires matplotlib (``pip install phonometry[plot]``);
        returns the :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.speech import plot_stoi

        return plot_stoi(self, ax=ax, language=check_language(language), **kwargs)


def _third_octave_matrix() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """One-third-octave grouping matrix and centre frequencies (15 bands).

    Each band spans the DFT bins from its lower to its upper one-third-octave
    edge, both snapped to the nearest bin of the length-``_NFFT`` rfft grid
    (Taal et al. 2011 Eq. 1). The upper-edge bin is exclusive, so adjacent
    bands do not double-count a shared bin.
    """
    f = np.linspace(0.0, SAMPLE_RATE, _NFFT + 1)[: _NFFT // 2 + 1]
    k = np.arange(_N_BANDS, dtype=np.float64)
    centers = _MIN_CENTER * 2.0 ** (k / 3.0)
    edges_low = _MIN_CENTER * 2.0 ** ((2.0 * k - 1.0) / 6.0)
    edges_high = _MIN_CENTER * 2.0 ** ((2.0 * k + 1.0) / 6.0)
    matrix = np.zeros((_N_BANDS, f.size), dtype=np.float64)
    for i in range(_N_BANDS):
        lo = int(np.argmin((f - edges_low[i]) ** 2))
        hi = int(np.argmin((f - edges_high[i]) ** 2))
        matrix[i, lo:hi] = 1.0
    return matrix, centers


def _analysis_window() -> NDArray[np.float64]:
    """MATLAB ``hanning(_FRAME)``: numpy Hann of length ``_FRAME+2``, ends cut."""
    return np.asarray(np.hanning(_FRAME + 2)[1:-1], dtype=np.float64)


def _n_frames(size: int, hop: int) -> int:
    """Number of length-``_FRAME`` frames at spacing ``hop`` (last edge dropped).

    Matches the reference STOI framing ``range(0, size - _FRAME, hop)`` (the
    trailing partial frame is excluded, as in the authors' MATLAB / pystoi).
    """
    return len(range(0, size - _FRAME, hop))


def _frame_signal(
    sig: NDArray[np.float64], window: NDArray[np.float64], hop: int
) -> NDArray[np.float64]:
    """Stack of ``window``-tapered, ``hop``-spaced frames of length ``_FRAME``.

    Vectorised frame extraction (a single strided gather then one broadcast
    multiply), numerically identical to windowing each frame in turn.
    """
    starts = np.arange(0, sig.size - _FRAME, hop)
    if starts.size == 0:
        return np.empty((0, _FRAME), dtype=np.float64)
    frames = sig[starts[:, None] + np.arange(_FRAME)]
    return np.asarray(frames * window, dtype=np.float64)


def _remove_silent_frames(
    clean: NDArray[np.float64], degraded: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Drop frames whose clean energy is > ``_DYN_RANGE_DB`` below the loudest.

    The keep/drop decision is taken on the clean signal alone (Taal et al.
    2011 Section II) and applied to both. The kept, Hann-tapered frames are
    reconstructed by overlap-add at a 50 % hop, so the retained signal carries
    the taper into the subsequent short-time transform (as in the reference
    implementation).
    """
    hop = _FRAME // 2
    window = _analysis_window()
    clean_frames = _frame_signal(clean, window, hop)
    degraded_frames = _frame_signal(degraded, window, hop)
    if clean_frames.size == 0:
        return clean[:0].copy(), degraded[:0].copy()
    energies = 20.0 * np.log10(np.linalg.norm(clean_frames, axis=1) + _EPS)
    keep = energies > energies.max() - _DYN_RANGE_DB
    clean_kept = clean_frames[keep]
    degraded_kept = degraded_frames[keep]
    return _overlap_add(clean_kept, hop), _overlap_add(degraded_kept, hop)


def _overlap_add(frames: NDArray[np.float64], hop: int) -> NDArray[np.float64]:
    """Overlap-add of ``hop``-spaced frames into one signal."""
    n_frames = frames.shape[0]
    if n_frames == 0:
        return np.zeros(0, dtype=np.float64)
    out = np.zeros((n_frames - 1) * hop + _FRAME, dtype=np.float64)
    for i in range(n_frames):
        out[i * hop : i * hop + _FRAME] += frames[i]
    return out


def _band_spectrogram(
    sig: NDArray[np.float64], window: NDArray[np.float64], matrix: NDArray[np.float64]
) -> NDArray[np.float64]:
    """One-third-octave band energies per short-time frame (Eq. 1).

    Returns a ``(15, frames)`` matrix whose rows are the temporal envelopes of
    the one-third-octave bands: the square root of the DFT magnitude energy
    summed over each band's bins.
    """
    frames = _frame_signal(sig, window, _FRAME // 2)  # (frames, _FRAME)
    spectra = np.fft.rfft(frames, n=_NFFT, axis=1)  # one batched transform
    power = np.abs(spectra) ** 2  # (frames, bins)
    band_power = power @ matrix.T  # (frames, bands)
    return np.asarray(np.sqrt(band_power).T, dtype=np.float64)  # (bands, frames)


def _resample_to_internal(sig: NDArray[np.float64], fs: int) -> NDArray[np.float64]:
    """Resample ``sig`` from ``fs`` to :data:`SAMPLE_RATE` (no-op when equal)."""
    if fs == SAMPLE_RATE:
        return sig
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(int(fs), SAMPLE_RATE)
    up = SAMPLE_RATE // g
    down = int(fs) // g
    return np.asarray(resample_poly(sig, up, down), dtype=np.float64)


def _validate_pair(
    clean: Signal | list[float] | np.ndarray,
    degraded: Signal | list[float] | np.ndarray,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate and return the clean/degraded 1-D float arrays."""
    x = np.asarray(clean, dtype=np.float64)
    y = np.asarray(degraded, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("'clean' and 'degraded' must be 1-D signals.")
    if x.shape != y.shape:
        raise ValueError(
            f"'clean' and 'degraded' must have equal length; got {x.size} and {y.size}."
        )
    if x.size == 0:
        raise ValueError("'clean' and 'degraded' must be non-empty.")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("'clean' and 'degraded' must be finite.")
    return apply_calibration(clean, x), apply_calibration(degraded, y)


def _segments(bands: NDArray[np.float64]) -> NDArray[np.float64]:
    """Sliding ``_N_SEGMENT``-frame windows of a band matrix.

    Returns a ``(segments, bands, _N_SEGMENT)`` array; segment ``m`` holds
    frames ``m-_N_SEGMENT .. m``.
    """
    n_frames = bands.shape[1]
    return np.array(
        [bands[:, m - _N_SEGMENT : m] for m in range(_N_SEGMENT, n_frames + 1)],
        dtype=np.float64,
    )


def _row_col_normalize(seg: NDArray[np.float64]) -> NDArray[np.float64]:
    """Mean- and variance-normalise rows then columns of each segment (ESTOI).

    ``seg`` is ``(segments, bands, frames)``. Rows (band envelopes) are made
    zero-mean and unit-norm, then columns (per-frame spectra) of the result
    are made zero-mean and unit-norm (Jensen & Taal 2016 Eqs. 4-7).
    """
    x = seg - np.mean(seg, axis=2, keepdims=True)
    x = x / (np.linalg.norm(x, axis=2, keepdims=True) + _EPS)
    x = x - np.mean(x, axis=1, keepdims=True)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + _EPS)
    return np.asarray(x, dtype=np.float64)


def stoi(
    clean: Signal | list[float] | np.ndarray,
    degraded: Signal | list[float] | np.ndarray,
    fs: int | None = None,
    *,
    extended: bool = False,
) -> STOIResult:
    """Short-time objective intelligibility of degraded speech.

    Computes STOI (Taal et al. 2011) or, with ``extended=True``, ESTOI (Jensen
    & Taal 2016): a scalar with a monotonic relation to the intelligibility of
    ``degraded`` relative to the clean reference ``clean``. Both signals are
    resampled internally to 10 kHz, split into one-third-octave short-time
    envelopes, and compared over 384 ms segments.

    :param clean: The clean reference speech (1-D). Accepts a
        :class:`phonometry.io.Signal`; STOI normalises every band of every
        segment before correlating them, so a calibration factor on either
        record cancels and the score does not move. The factor is applied
        anyway, so that the samples entering the computation are the same
        ones every other surface on this contract would see.
    :param degraded: The degraded/processed speech (1-D, same length as
        ``clean``, same sample rate).
    :param fs: Sample rate of both signals, in hertz. Required when both
        records are bare arrays; either may be a
        :class:`~phonometry.io.Signal` and supply it, and two Signals
        recorded at different rates are refused rather than arbitrated.
    :param extended: Use ESTOI (spectral correlation, robust to modulated
        maskers) instead of STOI.
    :return: A :class:`STOIResult` with the index ``.value`` and its
        ``.plot()``.
    :raises ValueError: if the signals are not equal-length finite 1-D arrays,
        ``fs`` is not positive, or fewer than 30 short-time frames survive the
        silent-frame removal (too little speech to score).
    """
    fs = resolve_pair_fs(clean, degraded, fs, names=("clean", "degraded"))
    x, y = _validate_pair(clean, degraded)
    if int(fs) <= 0:
        raise ValueError("'fs' must be a positive sample rate.")

    x = _resample_to_internal(x, int(fs))
    y = _resample_to_internal(y, int(fs))
    x, y = _remove_silent_frames(x, y)

    # Guard before the transform: with fewer than 30 short-time frames left
    # there is nothing to segment. Checking the frame count here keeps the
    # friendly message (a bare matmul on an empty spectrogram would raise a
    # cryptic shape error instead).
    if _n_frames(x.size, _FRAME // 2) < _N_SEGMENT:
        raise ValueError(
            "Too few short-time frames to score (need at least 30 after "
            "silent-frame removal, i.e. about 0.4 s of active speech); check "
            "the inputs are speech and long enough."
        )

    matrix, centers = _third_octave_matrix()
    window = _analysis_window()
    x_bands = _band_spectrogram(x, window, matrix)
    y_bands = _band_spectrogram(y, window, matrix)

    x_seg = _segments(x_bands)
    y_seg = _segments(y_bands)
    n_seg = x_seg.shape[0]

    if extended:
        x_n = _row_col_normalize(x_seg)
        y_n = _row_col_normalize(y_seg)
        # Spectral correlation per frame column, averaged over the frames of a
        # segment (Jensen & Taal 2016 Eq. 8), then over the segments.
        per_frame = np.sum(x_n * y_n, axis=1)  # (segments, frames)
        segment_scores = np.mean(per_frame, axis=1)  # (segments,)
        value = float(np.mean(segment_scores))
        band_scores: NDArray[np.float64] | None = None
    else:
        # Per-segment, per-band normalise-and-clip then correlate (Taal et al.
        # 2011 Eqs. 3-6).
        scale = np.linalg.norm(x_seg, axis=2, keepdims=True) / (
            np.linalg.norm(y_seg, axis=2, keepdims=True) + _EPS
        )
        y_scaled = y_seg * scale
        clip = 10.0 ** (-_SDR_DB / 20.0)
        y_clipped = np.minimum(y_scaled, x_seg * (1.0 + clip))
        y_c = y_clipped - np.mean(y_clipped, axis=2, keepdims=True)
        x_c = x_seg - np.mean(x_seg, axis=2, keepdims=True)
        y_c = y_c / (np.linalg.norm(y_c, axis=2, keepdims=True) + _EPS)
        x_c = x_c / (np.linalg.norm(x_c, axis=2, keepdims=True) + _EPS)
        corr = np.sum(x_c * y_c, axis=2)  # (segments, bands) = d_{j,m}
        value = float(np.sum(corr) / (n_seg * _N_BANDS))
        segment_scores = np.mean(corr, axis=1)  # (segments,)
        band_scores = np.mean(corr, axis=0)  # (bands,)

    return STOIResult(
        value=value,
        extended=extended,
        segment_scores=np.asarray(segment_scores, dtype=np.float64),
        band_scores=band_scores,
        band_frequencies=centers,
        sample_rate=SAMPLE_RATE,
    )
