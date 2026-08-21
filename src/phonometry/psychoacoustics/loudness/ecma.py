#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Psychoacoustic loudness per ECMA-418-2:2025 (4th ed., Sottek Hearing Model).

Clean-room implementation of the loudness signal chain of ECMA-418-2:2025:

* the shared auditory *front-end* of Clause 5 - trigonometric fade-in and
  zero-padding (5.1.2), the cascaded outer & middle/inner ear filter
  (5.1.3, Table 1), the 53-band gammatone-like auditory filter bank
  (5.1.4, Formulae 6-17), band-dependent segmentation (5.1.5, Table 4),
  half-wave rectification (5.1.6), block RMS (5.1.7), the compressive
  nonlinearity that maps sound pressure to specific loudness (5.1.8,
  Formula 23, Table 2) and the threshold in quiet (5.1.9, Table 3),
  yielding the specific basis loudness ``N'_basis(l, z)`` (Formula 25);
* the autocorrelation-based split into tonal and noise specific loudness
  with the full Clause 6.2.3 band averaging, including the
  cross-block-size-group ACF recomputation (Clause 6.2.2-6.2.7,
  Formulae 27-48; :func:`_tonal_noise_split`, shared with the tonality
  metric so both report the same underlying N'_tonal); and
* the loudness assembly of Clause 8 - the tonal/noise power average
  (8.1.1, Formulae 113-114), the average specific loudness (8.1.2),
  the time-dependent loudness (8.1.3, Formula 116) and the single
  representative value (8.1.4, Formula 117).

The front-end helpers (:func:`_ear_filter_sos`, :func:`_auditory_bandpass`
and the band-parameter tables) are written to be reused by the later
tonality and roughness metrics of the same standard without refactoring.

The calibration constant ``c_N`` of Formula (23) is fixed by the standard so
that a 1 kHz sinusoid at 40 dB SPL yields 1 sone_HMS. With the full
Clause 6.2.3 averaging this chain computes 0.9845 sone_HMS (-1.55 %,
outside the +/-0.25 % adjustment the standard allows for c_N; c_N is kept
at the verbatim tabulated value rather than retuned). The residual is
driven by the mandated band averaging around the block-size-boundary bands
excited by the tone's lower flank (~800-900 Hz): without any band
averaging the chain reads 0.955, with averaging restricted to
same-block-size neighbours 0.996, and with the full cross-group
recomputation 0.9845. The block-time smoothing stage and the fade-in/LP
transient contribute < 0.01 % (the value is signal-length invariant).

The API is monaural: the quadratic-mean binaural combination of
Formula (118) (Clause 8.1.5) is not implemented -- analyse each channel
separately.
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
from ..._internal.validation import require_1d_signal

# --------------------------------------------------------------------------
# Global constants (Clause 5)
# --------------------------------------------------------------------------

_FS = 48000  # normative sampling rate r_s (Clause 5.1.1)
_P0 = 2e-5  # reference pressure p~_0 = 20 uPa (Clause 5.1.8)
_EPS = 1e-12  # additive constant used throughout the standard

_N_FADE = 240  # 5 ms fade-in, n_fadein = 0.005 * 48000 (Formula 1)

#: Audibility criterion (Clause 5.1.9): a signal is considered audible when
#: its total basis loudness (sum of the specific basis loudness over the 53
#: bands with dz = 0.5) exceeds this value, in sone_HMS.
AUDIBILITY_THRESHOLD_SONE_HMS = 0.01

# Auditory filter bank (Clause 5.1.4)
_DF0 = 81.9289  # delta f at f = 0 [Hz] (below Formula 10)
_C = 0.1618  # scaling constant c (below Formula 10)
_K = 5  # auditory filter order (footnote 5)
_E_K5 = np.array([0.0, 1.0, 11.0, 11.0, 1.0])  # e_i for k = 5 (below Formula 15)
_Z = np.arange(0.5, 26.5 + 0.25, 0.5)  # critical-band-rate scale, 0.5..26.5
_CBF = _Z.size  # number of critical band filters = 53

# Band-dependent segmentation (Clause 5.1.5, Table 4): breakpoints on the
# auditory-filter bandwidth df(z) selecting the block size s_b -- below each
# limit the named block size applies; at or above the last one the smallest
# 1024-sample block does.
_DF_LIMIT_SB_8192 = 85.0
_DF_LIMIT_SB_4096 = 170.0
_DF_LIMIT_SB_2048 = 340.0

# Outer and middle/inner ear filter, free field (Table 1): [b0,b1,b2,a1,a2].
_EAR_FREE = np.array(
    [
        [1.015896, -1.925299, 0.922118, -1.925299, 0.938014],
        [0.958943, -1.806088, 0.876439, -1.806088, 0.835382],
        [0.961372, -1.763632, 0.821788, -1.763632, 0.783160],
        [2.225804, -1.434650, -0.498204, -1.434650, 0.727599],
        [0.471735, -0.366092, 0.244145, -0.366092, -0.284120],
        [0.115267, 0.000000, -0.115267, -1.796003, 0.805838],
        [0.988029, -1.912434, 0.926132, -1.912434, 0.914161],
        [1.952238, 0.162320, -0.667994, 0.162320, 0.284244],
    ]
)

# Compressive nonlinearity (Table 2): threshold levels [dB] and exponents.
_NL_LEVELS = np.array([15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0])
_NL_NU = np.array([0.6602, 0.0864, 0.6384, 0.0328, 0.4068, 0.2082, 0.3994, 0.6434])
_NL_ALPHA = 1.5  # alpha (Formula 23)
_C_N = 0.0211964  # calibration factor c_N (Clause 5.1.8)

# Specific loudness threshold in quiet LTQ(z) (Table 3), one per band.
_LTQ = np.array(
    [
        0.3310,
        0.1625,
        0.1051,
        0.0757,
        0.0576,
        0.0453,
        0.0365,
        0.0298,
        0.0247,
        0.0207,
        0.0176,
        0.0151,
        0.0131,
        0.0115,
        0.0103,
        0.0093,
        0.0086,
        0.0081,
        0.0077,
        0.0074,
        0.0073,
        0.0072,
        0.0071,
        0.0072,
        0.0073,
        0.0074,
        0.0076,
        0.0079,
        0.0082,
        0.0086,
        0.0092,
        0.0100,
        0.0109,
        0.0122,
        0.0138,
        0.0157,
        0.0172,
        0.0180,
        0.0180,
        0.0177,
        0.0176,
        0.0177,
        0.0182,
        0.0190,
        0.0202,
        0.0217,
        0.0237,
        0.0263,
        0.0296,
        0.0339,
        0.0398,
        0.0485,
        0.0622,
    ]
)

# Loudness power-average exponent (Clause 8.1.2/8.1.4): e = 1/log10(2).
_E_AVG = 1.0 / math.log10(2.0)
_W_N = 0.5331  # noise-loudness weight w_n (Formula 113)
_E_A = 0.2918  # exponent-function parameter a (Table 12)
_E_B = 0.5459  # exponent-function parameter b (Table 12)
_TRANSIENT_BLOCKS = 56  # discard l in [0, 56] (~300 ms) (Clause 8.1.2)
_R_SD = _FS / 256.0  # common-grid rate 187.5 Hz (Clause 6.2.6)

# Width in time blocks of the Clause 6.2.3 neighbouring-block ACF average
# (previous + current + next).
_TIME_AVG_BLOCKS = 3

# Noise-reduction sigmoid (Table 7) and g(z) parameters (Table 8).
_NR_ALPHA = 20.0
_NR_BETA = 0.07


def _band_frequencies() -> tuple[np.ndarray, np.ndarray]:
    """Centre frequencies F(z) and bandwidths df(z) (Formulae 9, 10)."""
    f_centre = (_DF0 / _C) * np.sinh(_C * _Z)
    df = np.sqrt(_DF0**2 + (_C * f_centre) ** 2)
    return f_centre, df


_F_CENTRE, _DF = _band_frequencies()


def _block_sizes() -> tuple[np.ndarray, np.ndarray]:
    """Band-dependent block size s_b(z) and hop size s_h(z) (Table 4)."""
    s_b = np.empty(_CBF, dtype=np.int64)
    for i, df in enumerate(_DF):
        if df < _DF_LIMIT_SB_8192:
            s_b[i] = 8192
        elif df < _DF_LIMIT_SB_4096:
            s_b[i] = 4096
        elif df < _DF_LIMIT_SB_2048:
            s_b[i] = 2048
        else:
            s_b[i] = 1024
    s_h = s_b // 4  # 75 % overlap
    return s_b, s_h


_S_B, _S_H = _block_sizes()
_S_B_MAX = int(_S_B.max())  # 8192
_S_B_MIN = int(_S_B.min())  # 1024
_S_H_MAX = int(_S_H.max())  # 2048
# Interpolation factor to the common time grid (Table 6): s_h / 256.
_INTERP = (_S_H // 256).astype(np.int64)
# Number of bands to average N_B by block size (Table 5).
_N_B_BY_SB = {8192: 2, 4096: 2, 2048: 1, 1024: 0}
# g(z) parameters c, d by block size (Table 8).
_G_CD = {
    8192: (18.21, 0.36),
    4096: (12.14, 0.36),
    2048: (417.54, 0.71),
    1024: (962.68, 0.69),
}


def _g_of_z() -> np.ndarray:
    """Frequency-dependent factor g(z) = c / F(z)^d (Formula 46, Table 8)."""
    out = np.empty(_CBF)
    for i in range(_CBF):
        c_p, d_p = _G_CD[int(_S_B[i])]
        out[i] = c_p / (_F_CENTRE[i] ** d_p)
    return out


_G_Z = _g_of_z()


@dataclass(frozen=True)
class EcmaLoudness:
    """Result of an ECMA-418-2:2025 (Sottek) loudness calculation.

    ``loudness`` is the single representative loudness N in sone_HMS
    (Formula 117).  ``specific_loudness`` is the average specific loudness
    N'(z) in sone_HMS/Bark_HMS over the 53 auditory bands (Formula 115),
    with ``bark`` the critical-band-rate scale z (0,5..26,5 Bark_HMS) and
    ``centre_frequencies`` the band centre frequencies F(z).  ``time`` and
    ``loudness_vs_time`` hold the time-dependent loudness N(l) at 187,5 Hz
    (Formula 116).  ``field`` records the assumed sound field.
    """

    loudness: float
    specific_loudness: np.ndarray
    bark: np.ndarray
    centre_frequencies: np.ndarray
    time: np.ndarray
    loudness_vs_time: np.ndarray
    field: str

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes | np.ndarray:
        """Plot the average specific loudness N'(z) (see :mod:`phonometry._plot.psychoacoustics`).

        Adds a loudness-vs-time panel.  Requires matplotlib
        (``pip install phonometry[plot]``).
        """
        from ..._i18n import check_language
        from ..._plot.psychoacoustics import plot_ecma_loudness

        return plot_ecma_loudness(
            self, ax=ax, language=check_language(language), **kwargs
        )


# --------------------------------------------------------------------------
# Front-end building blocks (Clause 5) - reusable by tonality / roughness
# --------------------------------------------------------------------------


def _ear_filter_sos(field: str) -> np.ndarray:
    """Outer & middle/inner ear filter as second-order sections (Clause 5.1.3).

    Returns the Table 1 biquads as ``scipy`` SOS rows ``[b0,b1,b2,1,a1,a2]``.
    For a free field all eight sections are used; for a diffuse field only
    sections 3-8 are used (sections 1-2 model the free-field-specific outer
    ear effect and are dropped).
    """
    rows = _EAR_FREE if field == "free" else _EAR_FREE[2:]
    sos = np.zeros((rows.shape[0], 6))
    sos[:, 0:3] = rows[:, 0:3]  # b0, b1, b2
    sos[:, 3] = 1.0  # a0
    sos[:, 4:6] = rows[:, 3:5]  # a1, a2
    return sos


def _auditory_coeffs(band: int) -> tuple[np.ndarray, np.ndarray]:
    """Complex band-pass IIR coefficients (b', a') for band ``band``.

    Implements the discrete gammatone-like auditory filter of Clause 5.1.4:
    the low-pass coefficients of Formulae (14)/(15) modulated to the band
    centre frequency by Formulae (16)/(17).  Order k = 5.
    """
    df = _DF[band]
    f_c = _F_CENTRE[band]
    tau = (1.0 / 2.0 ** (2 * _K - 1)) * math.comb(2 * _K - 2, _K - 1) * (1.0 / df)
    d = math.exp(-1.0 / (_FS * tau))
    m_a = np.arange(_K + 1)  # 0..k
    a = ((-d) ** m_a) * np.array([math.comb(_K, int(m)) for m in m_a])
    m_b = np.arange(_K)  # 0..k-1
    denom = np.sum(_E_K5[1:] * d ** np.arange(1, _K))
    b = ((1.0 - d) ** _K / denom) * (d**m_b) * _E_K5
    # Modulate to band centre frequency (Formulae 16, 17).
    rot_a = np.exp(1j * 2.0 * np.pi * f_c * m_a / _FS)
    rot_b = np.exp(1j * 2.0 * np.pi * f_c * m_b / _FS)
    return b * rot_b, a * rot_a


def _auditory_bandpass(p_om: np.ndarray, band: int) -> np.ndarray:
    """Real-valued band-pass signal p_om,z(n) for band ``band`` (Clause 5.1.4).

    The real ear-filtered signal is filtered with the complex coefficients;
    the real band-pass signal is twice the real part of the complex result.
    """
    b, a = _auditory_coeffs(band)
    complex_out = signal.lfilter(b, a, p_om.astype(np.complex128))
    return np.asarray(2.0 * np.real(complex_out), dtype=np.float64)


def _fade_in(x: np.ndarray) -> np.ndarray:
    """Trigonometric 5 ms fade-in of the input signal (Clause 5.1.2, Formula 1).

    Shared by every ECMA-418-2 metric: Formula (1) applies to the input
    signal *before* the metric-specific zero-padding (5.1.2.1 for
    tonality/loudness, 5.1.2.2 for roughness). Returns a faded copy.
    """
    x = np.asarray(x, dtype=np.float64).copy()
    n_fade = min(_N_FADE, x.size)
    n_idx = np.arange(n_fade)
    x[:n_fade] *= 0.5 - 0.5 * np.cos(np.pi * n_idx / _N_FADE)
    return x


def _preprocess(x: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Fade-in and zero-pad the signal (Clause 5.1.2, Formulae 1-3).

    Returns the padded signal p(n), n_samples (original length) and n_new.
    """
    n_samples = np.asarray(x).size
    x = _fade_in(x)
    n_new = _S_H_MAX * (math.ceil((n_samples + _S_H_MAX + _S_B_MAX) / _S_H_MAX) - 1)
    n_zeros_end = n_new - n_samples
    padded = np.concatenate([np.zeros(_S_B_MAX), x, np.zeros(max(n_zeros_end, 0))])
    return padded, n_samples, n_new


def _segment_bs(p_band: np.ndarray, block_size: int, n_new: int) -> np.ndarray:
    """Segment a band-pass signal into blocks (Clause 5.1.5, Formulae 18-20).

    Takes the block size directly so a band can also be re-segmented at a
    *foreign* block size for the cross-group averaging of Clause 6.2.3.
    Returns an array of shape ``(n_blocks, block_size)``.
    """
    s_h = block_size // 4  # 75 % overlap
    i_start = _S_B_MAX - block_size  # Formula 19
    l_last = math.ceil((n_new + s_h) / s_h) - 1  # Formula 20
    starts = i_start + s_h * np.arange(l_last + 1)
    starts = starts[starts + block_size <= p_band.size]
    idx = starts[:, None] + np.arange(block_size)[None, :]
    return np.asarray(p_band[idx], dtype=np.float64)


def _specific_basis_loudness(rms: np.ndarray, band: int) -> np.ndarray:
    """Specific basis loudness from block RMS values (Clause 5.1.8-5.1.9).

    ``rms`` is p~(l, z) for one band. Applies the compressive nonlinearity
    (Formula 23) and subtracts the threshold in quiet (Formula 25).
    """
    p_ti = _P0 * 10.0 ** (_NL_LEVELS / 20.0)  # Table 2 thresholds in Pa
    nu_prev = np.concatenate([[1.0], _NL_NU[:-1]])  # v_{i-1}, v_0 = 1
    exps = (_NL_NU - nu_prev) / _NL_ALPHA
    ratio = rms[:, None] / p_ti[None, :]
    product = np.prod((1.0 + ratio**_NL_ALPHA) ** exps[None, :], axis=1)
    n_tilde = _C_N * (rms / _P0) * product  # Formula 23/24
    return np.asarray(np.maximum(n_tilde - _LTQ[band], 0.0), dtype=np.float64)  # F.25


# --------------------------------------------------------------------------
# ACF-based tonal / noise loudness (Clause 6.2.2-6.2.7)
# --------------------------------------------------------------------------


def _band_acf(
    rect: np.ndarray, s_b: int, energy: np.ndarray | None = None
) -> np.ndarray:
    """Unbiased, normalized ACF of the rectified blocks (Formulae 27-29).

    ``rect`` is (n_blocks, s_b). Returns phi_z(m) of shape
    ``(n_blocks, 2*s_b)`` with valid values for 0 <= m < 3/4 s_b.
    ``energy`` optionally passes a precomputed ``rect**2``.
    """
    n_fft = 2 * s_b
    spec = np.fft.fft(rect, n=n_fft, axis=1)
    unscaled = np.real(np.fft.ifft(np.abs(spec) ** 2, axis=1))  # Formula 27/28
    # Cumulative block energy from both ends for the unbiased normalization.
    if energy is None:
        energy = rect**2
    csum = np.cumsum(energy[:, ::-1], axis=1)[:, ::-1]  # sum_{n'>=m} p^2(n')
    m_max = (3 * s_b) // 4  # 0 <= m < m_max < s_b
    out = np.zeros_like(unscaled)
    # sum_{n'=0..s_b-m-1} p^2(n') = total - sum_{n'>=s_b-m}; the lag loop is
    # written as elementwise column arithmetic (bit-identical to a per-lag
    # loop): m = 0 keeps the full energy, m >= 1 subtracts csum[:, s_b - m].
    if m_max == 0:
        return out
    total = csum[:, 0]
    left = np.empty((rect.shape[0], m_max), dtype=csum.dtype)
    left[:, 0] = total
    left[:, 1:] = total[:, None]
    left[:, 1:] -= csum[:, s_b - m_max + 1 : s_b][:, ::-1]
    right = csum[:, :m_max]
    denom = np.sqrt(left * right + _EPS)
    out[:, :m_max] = unscaled[:, :m_max] / denom
    return out


def _scaled_acf_at(
    p_bands: list[np.ndarray],
    band: int,
    block_size: int,
    n_new: int,
    cache: dict[tuple[int, int], np.ndarray],
) -> np.ndarray:
    """Scaled ACF phi'_z(m) of ``band`` computed at ``block_size`` (Formulae 22-30).

    Segments the band-pass signal at ``block_size``, half-wave rectifies
    (Formula 21), forms the block RMS (Formula 22) and specific basis loudness
    (Formula 25), the unbiased normalized ACF (Formulae 27-29) and scales it by
    the basis loudness (Formula 30). Cached by ``(band, block_size)``.
    """
    key = (band, block_size)
    cached = cache.get(key)
    if cached is not None:
        return cached
    seg = _segment_bs(p_bands[band], block_size, n_new)
    rect = np.where(seg > 0.0, seg, 0.0)  # Formula 21
    energy = rect**2  # shared by Formula 22 and the ACF normalization
    rms = np.sqrt((2.0 / block_size) * np.sum(energy, axis=1))  # Formula 22
    basis = _specific_basis_loudness(rms, band)  # Formulae 23-25
    acf = _band_acf(rect, block_size, energy)  # Formulae 27-29
    scaled = np.asarray(acf * basis[:, None], dtype=np.float64)  # Formula 30
    cache[key] = scaled
    return scaled


def _average_bands_full(p_bands: list[np.ndarray], n_new: int) -> list[np.ndarray]:
    """Full Clause 6.2.3 band averaging with cross-group recomputation.

    For each target band z (block size B = s_b(z), reach = min(N_B, z, 52-z)),
    the neighbouring bands are re-segmented and re-correlated **at block size
    B** so the average is over identical block sizes -- including neighbours
    that natively belong to a different block-size group. The result stays on
    band z's native time grid. Shared by the loudness and tonality metrics
    (Clause 8.1.1 builds loudness on the same Clause 6.2 outputs).
    """
    cache: dict[tuple[int, int], np.ndarray] = {}
    out: list[np.ndarray] = []
    for band in range(_CBF):
        block_size = int(_S_B[band])
        n_b = _N_B_BY_SB[block_size]
        native = _scaled_acf_at(p_bands, band, block_size, n_new, cache)
        if n_b == 0:
            out.append(native)
            continue
        if band == 0:  # lowest band: averaged only with the second-lowest
            other = _scaled_acf_at(p_bands, 1, block_size, n_new, cache)
            out.append(0.5 * (native + other))
            continue
        # The ``_CBF-1-band`` upper-edge term is a defensive guard that cannot
        # fire: bands with N_B > 0 all sit at low index (0..24, block size
        # >= 2048), so their distance to the top band (>= 28) never limits the
        # symmetric reach (n_b <= 2).  Only the ``band`` lower-edge term bites.
        # Never zero here: the two branches above have already taken every
        # case that could make it so, which is why there is no early-out.
        reach = min(n_b, band, _CBF - 1 - band)
        # Sequential accumulation; bit-identical to np.mean over a stacked
        # axis-0 (numpy reduces a non-contiguous axis sequentially, so
        # pairwise summation never applies) while avoiding the stacked copy.
        acc = _scaled_acf_at(p_bands, band - reach, block_size, n_new, cache).copy()
        for off in range(-reach + 1, reach + 1):
            acc += _scaled_acf_at(p_bands, band + off, block_size, n_new, cache)
        out.append(acc / float(2 * reach + 1))
    return out


def _average_blocks(averaged: list[np.ndarray]) -> list[np.ndarray]:
    """Average ACFs over neighbouring time blocks (Clause 6.2.3).

    Applied only to the 8192- and 4096-sample block sizes; the first and last
    blocks are left unchanged.
    """
    out: list[np.ndarray] = []
    for band in range(_CBF):
        acf = averaged[band]
        if int(_S_B[band]) in (8192, 4096) and acf.shape[0] >= _TIME_AVG_BLOCKS:
            smoothed = acf.copy()
            smoothed[1:-1] = (acf[:-2] + acf[1:-1] + acf[2:]) / _TIME_AVG_BLOCKS
            out.append(smoothed)
        else:
            out.append(acf)
    return out


def _tonal_estimate(
    acf: np.ndarray, band: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Windowed-ACF tonal loudness estimate (Clause 6.2.4-6.2.5).

    Returns ``(Nhat'_tonal(l, z), N'_signal(l, z), f_ton(l, z))`` at the band's
    own block rate. ``f_ton`` (Formulae 38-39) is the DFT-peak frequency of the
    windowed ACF; loudness ignores it, the tonality metric consumes it.
    """
    df = _DF[band]
    tau_start = max(0.5 / df, 0.002)  # Formula 31, tau_min = 2 ms
    tau_end = max(4.0 / df, tau_start + 0.001)  # Formula 32
    m_start = math.ceil(tau_start * _FS) - 1  # Formula 33
    m_end = math.floor(tau_end * _FS) - 1  # Formula 34
    m_end = min(m_end, acf.shape[1] - 1)
    m_count = m_end - m_start + 1
    n_signal = acf[:, 0].copy()  # Formula 41: phi_bar'(0)
    window = acf[:, m_start : m_end + 1].copy()  # Formula 35
    window -= window.mean(axis=1, keepdims=True)
    spec = np.abs(np.fft.fft(window, n=16384, axis=1))  # Formula 36
    # Formula 37: 2 * max|Phi'| / (M/2); the 2 compensates the half-wave
    # rectification and M/2 is the DFT energy normalization (footnote 19).
    peak = (4.0 / m_count) * np.max(spec, axis=1)
    n_tonal = np.minimum(peak, n_signal)
    # Formula 38/39: DFT peak bin -> physical frequency. The magnitude spectrum
    # of the real windowed ACF is conjugate-symmetric, so the argmax is
    # restricted to the lower half [0, r_s/2] to return the physical bin (the
    # mirror bin above Nyquist would otherwise alias to > 24 kHz).
    k_max = np.argmax(spec[:, : 16384 // 2 + 1], axis=1)
    f_ton = k_max.astype(np.float64) * _FS / 16384.0
    return n_tonal, n_signal, f_ton


def _resample_common(values: np.ndarray, band: int, n_common: int) -> np.ndarray:
    """Linearly resample a band time series onto the common grid (Formula 40)."""
    factor = int(_INTERP[band])
    positions = np.arange(values.size) * factor
    return np.asarray(
        np.interp(np.arange(n_common), positions, values), dtype=np.float64
    )


def _lowpass_time(x: np.ndarray) -> np.ndarray:
    """3.5 Hz time low-pass used in the noise reduction (Clause 6.2.7).

    Order-3 low-pass of Formula (11) with tau = (1/32)*6/(7 Hz) (footnote 20).
    """
    tau = (1.0 / 32.0) * 6.0 / 7.0
    d = math.exp(-1.0 / (_R_SD * tau))
    m_a = np.arange(4)  # 0..3
    a = ((-d) ** m_a) * np.array([math.comb(3, int(m)) for m in m_a])
    e = np.array([0.0, 1.0, 1.0])
    m_b = np.arange(3)
    denom = e[1] * d + e[2] * d**2
    b = ((1.0 - d) ** 3 / denom) * (d**m_b) * e
    return np.asarray(signal.lfilter(b, a, x, axis=0), dtype=np.float64)


def _tonal_noise_split(
    x: np.ndarray, field: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Clause 5 front-end + Clause 6.2 tonal/noise split on the common grid.

    Runs the shared auditory front-end (Clause 5), the full Clause 6.2.3
    band/block averaging and the noise reduction (Formulae 41-48), tracking
    the tonal frequency f_ton(l, z) of Formulae 38-39 along the way. Returns
    ``(N'_tonal(l, z), N'_noise(l, z), f_ton(l, z), n_common, n_samples)``.
    This single path feeds both the loudness metric (Clause 8.1.1 builds on
    the Clause 6.2 outputs, Formulae 47/48) and the tonality metric, so the
    two report the same underlying N'_tonal for the same signal.
    """
    padded, n_samples, n_new = _preprocess(x)
    p_om = signal.sosfilt(_ear_filter_sos(field), padded)
    p_bands = [_auditory_bandpass(p_om, band) for band in range(_CBF)]

    averaged = _average_blocks(_average_bands_full(p_bands, n_new))

    # Common time grid (Clause 6.2.6): sized from the block count of the
    # smallest (1024-sample) stage. The larger stages segment up to 7 blocks
    # past it (their i_start reaches further back into the start padding);
    # sizing from their count would only append edge-held interpolations.
    n_common = int(averaged[int(np.argmax(_S_B == _S_B_MIN))].shape[0])
    n_tonal = np.zeros((n_common, _CBF))
    n_signal = np.zeros((n_common, _CBF))
    f_ton = np.zeros((n_common, _CBF))
    for band in range(_CBF):
        tonal_b, signal_b, fton_b = _tonal_estimate(averaged[band], band)
        n_tonal[:, band] = _resample_common(tonal_b, band, n_common)
        n_signal[:, band] = _resample_common(signal_b, band, n_common)
        f_ton[:, band] = _resample_common(fton_b, band, n_common)

    # Noise reduction (Formulae 42-48).
    snr_hat = n_tonal / (n_signal - n_tonal + _EPS)  # Formula 42
    n_tonal_lp = _lowpass_time(n_tonal)  # Formula 43
    snr_lp = _lowpass_time(snr_hat)  # Formula 44
    n_signal_lp = _lowpass_time(n_signal)
    arg = _NR_ALPHA * (snr_lp / _G_Z[None, :] - _NR_BETA)  # Formula 45
    nr = np.where(arg > 0.0, 1.0 - np.exp(-arg), 0.0)
    n_tonal_final = nr * n_tonal_lp  # Formula 47
    # Formula 48 prints no max(., 0) clamp; the low-passed N'_signal can dip
    # marginally below the gated N'_tonal, so the clamp is kept as a benign,
    # documented deviation that only removes sub-EPS negative noise loudness.
    n_noise_final = np.maximum(n_signal_lp - n_tonal_final, 0.0)  # Formula 48
    return n_tonal_final, n_noise_final, f_ton, n_common, n_samples


# --------------------------------------------------------------------------
# Loudness assembly (Clause 8)
# --------------------------------------------------------------------------


def _assemble_loudness(
    n_tonal: np.ndarray, n_noise: np.ndarray, n_samples: int
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Combine tonal/noise loudness into the loudness metric (Clause 8.1).

    Returns (N, N'(z), N(l), time).
    """
    # Per-block exponent e(z) (Formula 114): uses the max over bands of
    # tonal + noise specific loudness.
    max_band = np.max(n_tonal + n_noise, axis=1)  # per block l
    e_block = _E_A / (max_band + _EPS) + _E_B
    e_block = e_block[:, None]
    spec = (n_tonal**e_block + (_W_N * n_noise) ** e_block) ** (
        1.0 / e_block
    )  # Formula 113
    spec = np.nan_to_num(spec, nan=0.0)

    n_blocks = spec.shape[0]
    l_end = min(math.ceil(n_samples / _FS * _R_SD), n_blocks - 1)
    keep = slice(_TRANSIENT_BLOCKS + 1, l_end + 1)
    kept = spec[keep]
    if kept.shape[0] == 0:  # very short signals: fall back to all blocks
        kept = spec
    # Average specific loudness N'(z) (Formula 115).
    n_spec = (np.mean(kept**_E_AVG, axis=0)) ** (1.0 / _E_AVG)
    # Time-dependent loudness N(l) (Formula 116), dz = 0.5.
    n_time = np.sum(spec, axis=1) * 0.5
    kept_time = n_time[keep]
    if kept_time.size == 0:
        kept_time = n_time
    # Representative single value N (Formula 117).
    n_single = float((np.mean(kept_time**_E_AVG)) ** (1.0 / _E_AVG))
    time = np.arange(n_blocks) / _R_SD
    return n_single, n_spec, n_time, time


def loudness_ecma(
    signal_in: Signal | np.ndarray,
    fs: float | None = None,
    field: Literal["free", "diffuse"] = "free",
) -> EcmaLoudness:
    """Psychoacoustic loudness per ECMA-418-2:2025 (Sottek Hearing Model).

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
    :return: An :class:`EcmaLoudness` with the single loudness value N
        (Formula 117), the average specific loudness N'(z) (Formula 115) and
        the time-dependent loudness N(l) (Formula 116).

    The 1 kHz / 40 dB SPL sinusoid defines 1 sone_HMS via the calibration
    constant of Formula (23); with the full Clause 6.2.3 averaging this
    chain computes 0.9845 sone_HMS for it (see the module docstring for the
    residual's origin). Monaural only (no Clause 8.1.5 binaural combination).
    """
    if field not in ("free", "diffuse"):
        msg = "field must be 'free' or 'diffuse'"
        raise ValueError(msg)
    fs = resolve_fs(signal_in, fs, name="signal_in")
    x = apply_calibration(
        signal_in, require_1d_signal(_typesignal(np.asarray(signal_in)))
    )
    if x.size == 0:
        msg = "signal must not be empty"
        raise ValueError(msg)
    fs = float(fs)
    if fs <= 0.0:
        msg = "fs must be positive"
        raise ValueError(msg)
    if fs != _FS:
        n_target = round(x.size * _FS / fs)
        x = signal.resample(x, n_target)

    n_tonal, n_noise, _, _, n_samples = _tonal_noise_split(x, field)
    n_single, n_spec, n_time, time = _assemble_loudness(n_tonal, n_noise, n_samples)
    return EcmaLoudness(
        loudness=n_single,
        specific_loudness=n_spec,
        bark=_Z.copy(),
        centre_frequencies=_F_CENTRE.copy(),
        time=time,
        loudness_vs_time=n_time,
        field=field,
    )
