#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Psychoacoustic tonality per ECMA-418-2:2025 (4th ed., Sottek Hearing Model).

Clean-room implementation of the tonality signal chain of ECMA-418-2:2025
(Clause 6.2). The shared auditory front-end (Clause 5) and the ACF-based
tonal/noise decomposition with the full Clause 6.2.3 band averaging
(Clause 6.2.2-6.2.7,
:func:`phonometry.psychoacoustics.loudness.ecma._tonal_noise_split`) are
reused from :mod:`phonometry.psychoacoustics.loudness.ecma` -- loudness and
tonality therefore report the same underlying ``N'_tonal(l, z)`` for the same
signal; this module adds

* the tonality output stages (Clause 6.2.8-6.2.11): the overall-SNR gate
  ``q(l)`` (Formulae 49-50), the time-dependent specific tonality
  :math:`T'(l, z) = c_\mathrm{T} q(l) N'_\mathrm{tonal}(l, z)` (Formula 51), the average
  specific tonality ``T'(z)`` and its frequency ``f_ton,z(z)`` (Formulae
  53-55), the time-dependent tonality ``T(l)`` with its frequency
  ``f_ton(l)`` (Formulae 61-62) and the representative single value ``T``
  (Formulae 63-64).

The calibration factor ``c_T`` of Formula (51) is fixed by the standard so
that a 1 kHz sinusoid at 40 dB SPL yields 1 tu_HMS.

The API is monaural; analyse each channel separately. (Unlike its
roughness and loudness, ECMA-418-2 defines no binaural combination for
tonality.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy import signal

from ...io._resolve import apply_calibration, resolve_fs

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ...io._signal import Signal

from ..._internal.utils import _typesignal
from ..._internal.validation import (
    require_1d_signal,
    require_ranks,
    require_same_length,
)
from ..loudness.ecma import (
    _CBF,
    _EPS,
    _F_CENTRE,
    _FS,
    _R_SD,
    _Z,
    _tonal_noise_split,
)

# Overall-SNR scaling gate q(l) (Formula 50, Table 9).
_Q_A = 35.0
_Q_B = 0.003

# Tonality calibration factor c_T (Formula 51); 1 kHz/40 dB -> 1 tu_HMS.
_C_T = 2.8758615

# Preconditions on the user band range [f_L, f_H] (Clause 6.2.10).
_F_LOW_MIN_HZ = 16.0  # Formula 56 precondition: f_L > 16 Hz
_F_HIGH_MAX_HZ = 20000.0  # Formula 57 precondition: f_H < 20 kHz

#: Prominence criterion (Clause 6.3): a signal has a prominent tonality when
#: the single value T (Formula 63) exceeds this value, in tu_HMS.
PROMINENT_TONALITY_TU_HMS = 0.4

# Averaging gate (Clause 6.2.9/6.2.11) and transient discard (Clause 6.2.9).
_T_GATE = 0.02  # tu_HMS
_TRANSIENT_BLOCKS = 56  # discard l in [0, 56]


@dataclass(frozen=True)
class EcmaTonality:
    """Result of an ECMA-418-2:2025 (Sottek) tonality calculation.

    ``tonality`` is the single representative tonality T in tu_HMS
    (Formula 63). ``specific_tonality`` is the average specific tonality
    T'(z) in tu_HMS over the 53 auditory bands (Formula 53), with ``bark``
    the critical-band-rate scale z (0.5..26.5 Bark_HMS), ``centre_frequencies``
    the band centre frequencies F(z) and ``tonal_frequencies`` the per-band
    tonal frequency f_ton,z(z) (Formula 55). ``time`` and ``tonality_vs_time``
    hold the time-dependent tonality T(l) at 187.5 Hz (Formula 61) and
    ``tonal_frequency_vs_time`` its frequency f_ton(l) (Formula 62). ``field``
    records the assumed sound field.
    """

    tonality: float
    specific_tonality: np.ndarray
    bark: np.ndarray
    centre_frequencies: np.ndarray
    tonal_frequencies: np.ndarray
    time: np.ndarray
    tonality_vs_time: np.ndarray
    tonal_frequency_vs_time: np.ndarray
    field: str

    def __post_init__(self) -> None:
        """Reject a result whose curves disagree with the axes they are drawn on.

        The figure this result draws is two panels, and each is one call
        pairing an axis with a curve: the average specific tonality over the
        critical-band-rate scale, and the time-dependent tonality over time.
        A curve of the wrong length reaches matplotlib, which raises about x
        and y differing in their first dimension and names neither field, from
        a call several frames below the one the caller made. Worse, a caller
        who supplies an ``ax`` gets only the specific-tonality panel, so a time
        axis that disagrees is drawn nowhere and goes unnoticed until somebody
        asks for the whole figure.

        The two tonal-frequency fields are quieter still: no panel draws
        f_ton,z(z) or f_ton(l) and nothing else in the library reads them, so a
        wrong length there is announced by no one. The figure comes out looking
        perfectly ordinary and the array reaches the caller intact, one band or
        one block out of step with the axis it is indexed against.

        The two axes are independent of each other: the auditory bands are the
        53 of the standard's filter bank whatever the signal, while the blocks
        are however many the signal ran to, so they are pinned separately.

        :raises ValueError: if the per-band or per-block quantities disagree.
        """
        require_ranks(
            self,
            specific_tonality=1,
            bark=1,
            centre_frequencies=1,
            tonal_frequencies=1,
            time=1,
            tonality_vs_time=1,
            tonal_frequency_vs_time=1,
        )
        require_same_length(
            self,
            "bark",
            "centre_frequencies",
            "specific_tonality",
            "tonal_frequencies",
            axis="auditory band",
        )
        require_same_length(
            self,
            "time",
            "tonality_vs_time",
            "tonal_frequency_vs_time",
            axis="time block",
        )

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes | np.ndarray:
        """Plot the average specific tonality T'(z) (see :mod:`phonometry._plot.psychoacoustics`).

        Adds a tonality-vs-time panel. Requires matplotlib
        (``pip install phonometry[plot]``).
        """
        from ..._i18n import check_language
        from ..._plot.psychoacoustics import plot_ecma_tonality

        return plot_ecma_tonality(
            self, ax=ax, language=check_language(language), **kwargs
        )


# --------------------------------------------------------------------------
# Tonality assembly (Clause 6.2.8-6.2.11)
# --------------------------------------------------------------------------


def _band_range(f_low: float | None, f_high: float | None) -> tuple[int, int]:
    r"""Critical-band index range [z_L, z_H] for a user band
    (Formulae 56-60).

    ``None`` limits default to the full 0..52 band range.  A band z is included
    when its edge midpoints to the neighbouring bands straddle the user edge:
    :math:`f_{\mathrm{low}} < (F(z) + F(z+0.5))/2` selects z_L (Formula 56)
    and
    :math:`f_{\mathrm{high}} > (F(z) + F(z-0.5))/2` selects z_H (Formula 57).
    ``mid[i]`` is
    the boundary between bands ``i`` and ``i+1`` on the 0.5-Bark_HMS grid.

    Enforces the Formulae 56/57 preconditions: 16 Hz < f_L, f_H < 20 kHz and
    f_L < f_H.
    """
    if f_low is not None and (not math.isfinite(f_low) or f_low <= _F_LOW_MIN_HZ):
        msg = "'f_low' must exceed 16 Hz (Formula 56)."
        raise ValueError(msg)
    if f_high is not None and (not math.isfinite(f_high) or f_high >= _F_HIGH_MAX_HZ):
        msg = "'f_high' must be below 20 kHz (Formula 57)."
        raise ValueError(msg)
    if f_low is not None and f_high is not None and f_low >= f_high:
        msg = "'f_low' must be below 'f_high'."
        raise ValueError(msg)
    z_lo = 0
    z_hi = _CBF - 1
    mid = (_F_CENTRE[:-1] + _F_CENTRE[1:]) / 2.0  # inter-band boundaries
    if f_low is not None:
        # z_L: lowest band whose upper boundary exceeds f_low (Formula 56).
        candidates = np.nonzero(f_low < mid)[0]
        z_lo = int(candidates[0]) if candidates.size else _CBF - 1
    if f_high is not None:
        # z_H: highest band whose lower boundary is below f_high (Formula 57);
        # f_high > mid[j] admits band j+1, so shift the index up by one.
        candidates = np.nonzero(f_high > mid)[0]
        z_hi = int(candidates[-1]) + 1 if candidates.size else 0
    return z_lo, min(z_hi, _CBF - 1)


def _assemble_tonality(
    n_tonal: np.ndarray,
    n_noise: np.ndarray,
    f_ton: np.ndarray,
    n_samples: int,
    z_lo: int,
    z_hi: int,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble the tonality metric (Clause 6.2.8-6.2.11).

    Returns ``(T, T'(z), f_ton,z(z), T(l), f_ton(l), time)``.
    """
    n_blocks = n_tonal.shape[0]
    # Overall-SNR gate (Formulae 49-50).
    snr = np.max(n_tonal, axis=1) / (_EPS + np.sum(n_noise, axis=1))  # F.49
    q = np.where(snr > _Q_B, 1.0 - np.exp(-_Q_A * (snr - _Q_B)), 0.0)  # F.50
    t_spec_lz = _C_T * q[:, None] * n_tonal  # Formula 51: T'(l, z)

    # Averaging window: discard transient, bound by l_end (Formulae 40, 54).
    l_end = min(math.ceil(n_samples / _FS * _R_SD), n_blocks - 1)
    keep = slice(_TRANSIENT_BLOCKS + 1, l_end + 1)
    if t_spec_lz[keep].shape[0] == 0:  # very short signals: fall back to all
        keep = slice(0, n_blocks)  # (matches loudness_ecma's fallback)
    t_win = t_spec_lz[keep]  # (n_kept, z)
    f_win = f_ton[keep]

    # Average specific tonality T'(z) and frequency f_ton,z(z) (Formulae 53-55).
    gate = t_win > _T_GATE
    counts = gate.sum(axis=0)
    t_spec = np.where(counts > 0, (t_win * gate).sum(axis=0) / (counts + _EPS), 0.0)
    f_spec = np.where(counts > 0, (f_win * gate).sum(axis=0) / (counts + _EPS), 0.0)

    # Time-dependent tonality T(l) and its frequency (Formulae 61-62).
    band_slice = slice(z_lo, z_hi + 1)
    t_time = np.max(t_spec_lz[:, band_slice], axis=1)  # Formula 61
    z_max = z_lo + np.argmax(t_spec_lz[:, band_slice], axis=1)
    f_time = f_ton[np.arange(n_blocks), z_max]  # Formula 62

    # Representative single value T (Formulae 63-64).
    t_time_win = t_time[keep]
    tmask = t_time_win > _T_GATE
    n_sel = int(tmask.sum())
    t_single = float(t_time_win[tmask].sum() / (n_sel + _EPS)) if n_sel else 0.0

    time = np.arange(n_blocks) / _R_SD  # Formula 52
    return t_single, t_spec, f_spec, t_time, f_time, time


def tonality_ecma(
    signal_in: Signal | np.ndarray,
    fs: float | None = None,
    field: Literal["free", "diffuse"] = "free",
    f_low: float | None = None,
    f_high: float | None = None,
) -> EcmaTonality:
    """Psychoacoustic tonality per ECMA-418-2:2025 (Sottek Hearing Model).

    :param signal_in: Calibrated sound pressure signal in pascals. Accepts a
        :class:`phonometry.io.Signal`, which is where "calibrated" comes
        from without arithmetic: this model reads absolute levels, so an
        uncalibrated record is taken as if one digital unit were one
        pascal and the answer is wrong by however far that is from true.
    :param fs: Sampling rate in Hz. Signals not at 48 kHz are resampled
        (Clause 5.1.1). Required for a bare array; a
        :class:`~phonometry.io.Signal` brings its own, and an explicit value
        that disagrees with it raises instead of silently winning.
    :param field: ``"free"`` (default) or ``"diffuse"`` sound field, selecting
        the outer/middle-ear filter of Clause 5.1.3.
    :param f_low: Optional lower edge (Hz) of a user frequency band for the
        time-dependent tonality maximum search (Formulae 56-60). ``None`` uses
        the full range.
    :param f_high: Optional upper edge (Hz) of the user frequency band.
    :return: An :class:`EcmaTonality` with the single value T (Formula 63),
        the average specific tonality T'(z) (Formula 53), the tonal
        frequencies f_ton,z(z) (Formula 55) and the time-dependent tonality
        T(l) (Formula 61) with its frequency (Formula 62).

    The 1 kHz / 40 dB SPL sinusoid yields 1 tu_HMS by construction of the
    calibration factor of Formula (51).
    """
    if field not in ("free", "diffuse"):
        msg = "field must be 'free' or 'diffuse'"
        raise ValueError(msg)
    fs = resolve_fs(signal_in, fs, name="signal_in")
    x = apply_calibration(
        signal_in,
        require_1d_signal(_typesignal(signal_in, name="signal_in")),
    )
    if x.size == 0:
        msg = "signal must not be empty"
        raise ValueError(msg)
    fs = float(fs)
    if fs <= 0.0:
        msg = "fs must be positive"
        raise ValueError(msg)
    if fs != _FS:
        x = signal.resample(x, round(x.size * _FS / fs))

    z_lo, z_hi = _band_range(f_low, f_high)
    n_tonal, n_noise, f_ton, _, n_samples = _tonal_noise_split(x, field)
    t_single, t_spec, f_spec, t_time, f_time, time = _assemble_tonality(
        n_tonal, n_noise, f_ton, n_samples, z_lo, z_hi
    )
    return EcmaTonality(
        tonality=t_single,
        specific_tonality=t_spec,
        bark=_Z.copy(),
        centre_frequencies=_F_CENTRE.copy(),
        tonal_frequencies=f_spec,
        time=time,
        tonality_vs_time=t_time,
        tonal_frequency_vs_time=f_time,
        field=field,
    )
