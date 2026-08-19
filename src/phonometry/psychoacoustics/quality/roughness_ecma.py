#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Psychoacoustic roughness per ECMA-418-2:2025 (4th ed., Sottek Hearing Model).

Clean-room implementation of the roughness signal chain of ECMA-418-2:2025
(Clause 7). The shared auditory front-end (Clause 5: outer/middle-ear filter,
53-band gammatone-like filter bank, half-wave rectification, compressive
nonlinearity to the specific basis loudness ``N'_basis(l, z)`` of Formula 25)
is reused from :mod:`phonometry.psychoacoustics.loudness.ecma`; this module
adds the roughness-specific chain:

* roughness-specific zero-padding (Clause 5.1.2.2) and segmentation
  (Clause 5.1.5.2) with the fixed block/hop :math:`s_\mathrm{b} = 16384` /
  :math:`s_\mathrm{h} = 4096`
  (Clause 7.1.1);
* the Hilbert envelope of each critical-band block and a factor-32 downsampling
  to 1500 Hz (Clause 7.1.2, Formula 65);
* the scaled envelope power spectrum ``Phi_E,l,z(k)`` (Clause 7.1.3,
  Formulae 66-67);
* the two-step noise reduction of the envelope spectra (Clause 7.1.4,
  Formulae 68-71);
* the four-stage spectral weighting (Clause 7.1.5, Formulae 72-96): peak
  picking with a quadratic-fit modulation-rate refinement and a bias
  correction, the high-modulation-rate weighting, the fundamental
  modulation-rate estimation, and the low-modulation-rate weighting;
* the interpolation to 50 Hz, the distribution-dependent nonlinear transform
  with the calibration constant ``c_R``, and the asymmetric time smoothing
  (Clause 7.1.7, Formulae 103-110); and
* the average specific roughness ``R'(z)`` (Clause 7.1.8), the time-dependent
  roughness ``R(l50)`` (Clause 7.1.9, Formula 111) and the representative
  90th-percentile single value ``R`` (Clause 7.1.10).

The optional entropy weighting of Clause 7.1.6 requires an external rotational
speed signal and is not implemented (see ``notes-ecma418-2-roughness.md``).
The API is monaural: the quadratic-mean binaural combination of
Formula (112) (Clause 7.1.11) is not implemented -- analyse each channel
separately.

The calibration constant ``c_R`` of Formula (104) is the standard's tabulated
value (not reverse-fit), and the chain reproduces the Clause 7 reference
point exactly: a 1 kHz carrier 100 %-amplitude-modulated at 70 Hz with an
overall sound pressure level of 60 dB SPL computes to 0.9999 asper against
the defined 1 asper. Note the level convention: Clause 7 states the *sound
pressure level of the signal* (its overall RMS level), not the level of the
unmodulated carrier -- a fully modulated signal whose carrier alone sits at
60 dB is +1.76 dB hot overall and reads ~4 % high.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy import signal
from scipy.interpolate import PchipInterpolator

from ...io._resolve import apply_calibration, resolve_fs
from ...io._signal import Signal

if TYPE_CHECKING:
    from matplotlib.axes import Axes

from ..._internal.utils import _typesignal
from ..._internal.validation import require_1d_signal
from ..loudness.ecma import (
    _CBF,
    _EPS,
    _F_CENTRE,
    _FS,
    _Z,
    _auditory_bandpass,
    _ear_filter_sos,
    _fade_in,
    _specific_basis_loudness,
)

# --------------------------------------------------------------------------
# Roughness-specific constants (Clause 7.1.1)
# --------------------------------------------------------------------------

_SB = 16384  # block size s_b (Clause 7.1.1)
_SH = 4096  # hop size s_h (75 % overlap)
_DOWN = 32  # envelope downsampling factor (Clause 7.1.2, footnote 27)
_SB_TILDE = _SB // _DOWN  # 512, downsampled block size
_R_TILDE = _FS / _DOWN  # 1500 Hz downsampled rate
_DELTA_F = _R_TILDE / _SB_TILDE  # 2.9297 Hz DFT resolution (Formula 76)
_K_ONE = _SB_TILDE // 2 + 1  # 257 one-sided modulation-rate bins
_R_S50 = 50.0  # interpolated rate r_s50 (Clause 7.1.7)
_DZ = 0.5  # critical-band overlap dz (Clause 7.1.9)
_TRANSIENT = 16  # discard l50 in [0, 15] (Clause 7.1.8)

# Calibration constant c_R (Formula 104, tabulated; adjustable +/- 0.25 %).
_C_R = 0.0180685

#: Prominence criterion (Clause 7.2): a signal has a prominent roughness when
#: the single value R (Clause 7.1.10) exceeds this value, in asper.
PROMINENT_ROUGHNESS_ASPER = 0.2
_A_THRESHOLD = 0.074376  # low-rate amplitude gate (Clause 7.1.5.4)

# Error-correction table E(theta), theta = 0..33 (Table 10; theta 33 is a
# guard row equal to 0 that simplifies the branch of Formula 81).
_E_THETA = np.array(
    [
        0.0000, 0.0457, 0.0907, 0.1346, 0.1765, 0.2157, 0.2515, 0.2828,
        0.3084, 0.3269, 0.3364, 0.3348, 0.3188, 0.2844, 0.2259, 0.1351,
        0.0000, -0.1351, -0.2259, -0.2844, -0.3188, -0.3348, -0.3364,
        -0.3269, -0.3084, -0.2828, -0.2515, -0.2157, -0.1765, -0.1346,
        -0.0907, -0.0457, 0.0000, 0.0000,
    ]
)

# Modulation-rate weighting parameters (Clause 7.1.5.2/7.1.5.4).
_Q1_HI = 1.2822  # q1 for the high-rate weighting (Formula 87)
_Q1_LO = 0.7066  # q1 for the low-rate weighting (below Formula 95)


def _f_max() -> np.ndarray:
    """Modulation rate of the weighting maximum f_max(z) (Formula 86)."""
    return 72.6937 * (1.0 - 1.1739 * np.exp(-5.4583 * _F_CENTRE / 1000.0))


def _r_max() -> np.ndarray:
    """Scaling factor r_max(z) (Formula 84, Table 11)."""
    lg = np.abs(np.log2(_F_CENTRE / 1000.0))
    r1 = np.where(_F_CENTRE < 1000.0, 0.3560, 0.8024)
    r2 = np.where(_F_CENTRE < 1000.0, 0.8049, 0.9333)
    return np.asarray(1.0 / (1.0 + r1 * lg**r2), dtype=np.float64)


def _q2_hi() -> np.ndarray:
    """q2(z) for the high-rate weighting (Formula 87)."""
    lg = np.log2(_F_CENTRE / 1000.0)
    out = 0.2471 * np.ones(_CBF)
    mask = (_F_CENTRE / 1000.0) >= 2.0**-3.4253
    out[mask] = 0.2471 + 0.0129 * (lg[mask] + 3.4253) ** 2
    return out


def _q2_lo() -> np.ndarray:
    """q2(z) for the low-rate weighting (Formula 96)."""
    return np.asarray(1.0967 - 0.0640 * np.log2(_F_CENTRE / 1000.0), dtype=np.float64)


_F_MAX = _f_max()
_R_MAX = _r_max()
_Q2_HI = _q2_hi()
_Q2_LO = _q2_lo()

# Scaled von-Hann window w_Hann(n~) (Clause 7.1.3, footnote 29).
_HANN = (
    0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(_SB_TILDE) / _SB_TILDE)
) / np.sqrt(0.375)


@dataclass(frozen=True)
class EcmaRoughness:
    """Result of an ECMA-418-2:2025 (Sottek) roughness calculation.

    ``roughness`` is the single representative roughness R in asper (the
    90th percentile of R(l50), Clause 7.1.10). ``specific_roughness`` is the
    average specific roughness R'(z) in asper/Bark_HMS over the 53 auditory
    bands (Clause 7.1.8), with ``bark`` the critical-band-rate scale z
    (0.5..26.5 Bark_HMS) and ``centre_frequencies`` the band centre
    frequencies F(z). ``time`` and ``roughness_vs_time`` hold the
    time-dependent roughness R(l50) at 50 Hz (Formula 111);
    ``specific_roughness_vs_time`` is the time-dependent specific roughness
    R'(l50, z) (Formula 109) of shape ``(n_times, 53)``. ``field`` records the
    assumed sound field.
    """

    roughness: float
    specific_roughness: np.ndarray
    bark: np.ndarray
    centre_frequencies: np.ndarray
    time: np.ndarray
    roughness_vs_time: np.ndarray
    specific_roughness_vs_time: np.ndarray
    field: str

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes | np.ndarray:
        """Plot the roughness result (see :mod:`phonometry._plot.psychoacoustics`).

        Draws the time-dependent roughness R(l50) and a specific-roughness
        heatmap. Requires matplotlib (``pip install phonometry[plot]``).
        """
        from ..._i18n import check_language
        from ..._plot.psychoacoustics import plot_ecma_roughness

        return plot_ecma_roughness(self, ax=ax, language=check_language(language), **kwargs)


# --------------------------------------------------------------------------
# Front-end: envelope + basis loudness per roughness block (Clause 5, 7.1.2)
# --------------------------------------------------------------------------


def _front_end(
    x: np.ndarray, field: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Envelopes, basis loudness and block times for the roughness chain.

    Applies the roughness zero-padding (Clause 5.1.2.2), ear filter and
    auditory band-pass (Clause 5.1.3-5.1.4), then per band segments at the
    fixed block/hop (Clause 5.1.5.2), forms the downsampled Hilbert envelope
    (Clause 7.1.2, Formula 65) and the specific basis loudness (Formula 25).

    Returns ``(envelopes[L, 512, Z], basis[L, Z], block_times[L], n_samples)``.
    """
    n_samples = x.size
    # Fade-in (Formula 1) then the roughness start pad only (5.1.2.2).
    padded = np.concatenate([np.zeros(_SB), _fade_in(x)])
    n_pad = padded.size
    p_om = signal.sosfilt(_ear_filter_sos(field), padded)

    starts = list(range(0, max(n_pad - _SB, 0) + 1, _SH))
    if not starts:
        starts = [0]
    if starts[-1] != n_pad - _SB and n_pad - _SB > starts[-1]:
        # Flush-to-end last block. NOTE (Clause 5.1.5.2): the printed
        # l_last = ceil((n + s_b)/s_h) would let blocks overrun the padded
        # signal (and makes the Formula-103 time grid non-monotonic); the
        # only self-consistent reading is to stop at the last block that
        # fits and align it flush with the padded end, as done here.
        starts.append(n_pad - _SB)
    start_idx = np.array(starts, dtype=np.int64)
    n_blocks = start_idx.size
    idx = start_idx[:, None] + np.arange(_SB)[None, :]

    envelopes = np.zeros((n_blocks, _SB_TILDE, _CBF))
    basis = np.zeros((n_blocks, _CBF))
    for band in range(_CBF):
        p_band = _auditory_bandpass(p_om, band)
        # clip block window to the available signal for the flush last block
        seg = p_band[np.minimum(idx, p_band.size - 1)]
        analytic = signal.hilbert(seg, axis=1)  # Formula 65
        envelopes[:, :, band] = np.abs(analytic)[:, :: _DOWN]  # downsample x32
        rect = np.where(seg > 0.0, seg, 0.0)  # Formula 21
        rms = np.sqrt((2.0 / _SB) * np.sum(rect**2, axis=1))  # Formula 22
        basis[:, band] = _specific_basis_loudness(rms, band)  # Formulae 23-25

    block_times = start_idx.astype(np.float64) / _FS
    return envelopes, basis, block_times, n_samples


# --------------------------------------------------------------------------
# Scaled power spectrum + noise reduction (Clause 7.1.3-7.1.4)
# --------------------------------------------------------------------------


def _noise_reduced_spectra(
    envelopes: np.ndarray, basis: np.ndarray
) -> np.ndarray:
    """Noise-reduced scaled envelope power spectra Phi_hat (Formulae 66-71).

    Returns the one-sided spectra of shape ``(L, 257, Z)``.
    """
    ew = envelopes * _HANN[None, :, None]  # windowed envelope
    energy = np.sum(ew**2, axis=1)  # phi_E,l,z(0) (Formula 67), (L, Z)
    n_max = np.max(basis, axis=1)  # N'_max(l) (below Formula 66)
    spec = np.abs(np.fft.fft(ew, axis=1)) ** 2  # |DFT|^2, (L, 512, Z)
    spec = spec[:, :_K_ONE, :]  # one-sided k = 0..256

    denom = n_max[:, None] * energy  # (L, Z)
    scale = np.zeros_like(basis)
    ok = denom > 0.0
    scale[ok] = basis[ok] ** 2 / denom[ok]  # Formula 66 scaling
    phi = spec * scale[:, None, :]

    # Band averaging over 3 bands; first/last retained (Formula 68).
    avg = phi.copy()
    avg[:, :, 1:-1] = (phi[:, :, :-2] + phi[:, :, 1:-1] + phi[:, :, 2:]) / 3.0

    s = np.sum(avg, axis=2)  # s(l, k) (Formula 68), (L, 257)
    s_med = np.median(s[:, 2:256], axis=1)  # median over k = 2..255
    k = np.arange(_K_ONE)
    expo = np.clip(0.1891 * np.exp(0.0120 * k), 0.0, 1.0)  # Formula 71 clip
    w_tilde = 0.0856 * s / (s_med[:, None] + 1e-10) * expo[None, :]
    thr = 0.05 * np.max(w_tilde[:, 2:256], axis=1)  # Formula 70 threshold
    w = np.where(
        w_tilde >= thr[:, None], np.clip(w_tilde - 0.1407, 0.0, 1.0), 0.0
    )  # Formula 70
    return np.asarray(avg * w[:, :, None], dtype=np.float64)  # Formula 69


# --------------------------------------------------------------------------
# Spectral weighting (Clause 7.1.5)
# --------------------------------------------------------------------------


def _g_weight(f: float, f_max: float, q1: float, q2: float) -> float:
    """Modulation-rate weighting G(f) (Formula 85); equals 1 at f = f_max."""
    inner = (f / f_max - f_max / f) * q1
    return float(1.0 / (1.0 + inner * inner) ** q2)


def _bias_correction(f_tilde: float) -> float:
    """Quadratic-fit bias correction rho (Formulae 78-81, Table 10)."""
    theta = np.arange(34)
    beta = (np.floor(f_tilde / _DELTA_F) + theta / 32.0) * _DELTA_F - (
        f_tilde + _E_THETA
    )  # Formula 79
    theta_min = int(np.argmin(np.abs(beta[:33])))  # Formula 80 (0 <= theta <= 32)
    if theta_min > 0 and beta[theta_min] * beta[theta_min - 1] < 0.0:
        theta_corr = theta_min  # Formula 81
    else:
        theta_corr = theta_min + 1
    theta_corr = min(theta_corr, 33)
    d_beta = beta[theta_corr] - beta[theta_corr - 1]
    return float(
        _E_THETA[theta_corr - 1]
        - (_E_THETA[theta_corr] - _E_THETA[theta_corr - 1])
        * beta[theta_corr - 1]
        / (d_beta if abs(d_beta) > _EPS else _EPS)
    )  # Formula 78


def _pick_peaks(spec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Peak modulation rates f_p,i and amplitudes A_i (Clause 7.1.5.1).

    ``spec`` is the noise-reduced spectrum Phi_hat for one (l, z). Returns
    ``(f_p, A)`` sorted by ascending modulation rate.
    """
    window = spec[2:256]  # k = 2..255
    peaks, props = signal.find_peaks(window, prominence=0.0)
    if peaks.size == 0:
        return np.empty(0), np.empty(0)
    kloc = peaks + 2  # absolute modulation-rate index
    proms = np.asarray(props["prominences"], dtype=np.float64)
    if kloc.size > 10:  # keep the 10 highest-prominence peaks
        top = np.argsort(proms)[::-1][:10]
        kloc = np.sort(kloc[top])
    amps = spec[kloc]
    keep = amps > 0.05 * np.max(amps)  # Formula 72
    kloc = kloc[keep]
    if kloc.size == 0:
        return np.empty(0), np.empty(0)

    f_p = np.empty(kloc.size)
    a_i = np.empty(kloc.size)
    for j, k in enumerate(kloc):
        y = spec[k - 1 : k + 2]
        mat = np.array(
            [
                [(k - 1) ** 2, k - 1, 1.0],
                [k**2, k, 1.0],
                [(k + 1) ** 2, k + 1, 1.0],
            ]
        )  # Formula 75
        try:
            c = np.linalg.solve(mat, y)  # Formula 73
        except np.linalg.LinAlgError:  # pragma: no cover - defensive
            c = np.array([0.0, 0.0, 0.0])
        f_tilde = (
            -(c[1] / (2.0 * c[0])) * _DELTA_F if abs(c[0]) > _EPS else k * _DELTA_F
        )
        f_p[j] = f_tilde + _bias_correction(f_tilde)  # Formulae 76-77
        a_i[j] = float(np.sum(y))  # Formula 82
    return f_p, a_i


def _harmonic_members(f_p: np.ndarray, i0: int) -> list[int]:
    """Peaks that are near-integer multiples of ``f_p[i0]`` (Formulae 88-90).

    Each integer ratio keeps the single peak whose relative deviation from the
    exact multiple is smallest (Formula 89), and a member is retained only when
    that deviation stays under 4 % (Formula 90).
    """
    ratios = np.round(f_p / f_p[i0]).astype(int)  # Formula 88
    cand: dict[int, tuple[int, float]] = {}
    for i in range(f_p.size):
        r = int(ratios[i])
        if r <= 0:
            continue
        err = abs(f_p[i] / (r * f_p[i0]) - 1.0)  # Formula 89
        if r not in cand or err < cand[r][1]:
            cand[r] = (i, err)
    return [i for i, err in cand.values() if err < 0.04]  # Formula 90


def _fundamental_set(
    f_p: np.ndarray, a_tilde: np.ndarray
) -> tuple[list[int], int]:
    """Harmonic-complex index set I_max and its argmax i_max (Formulae 88-91)."""
    n = f_p.size
    best_energy = -1.0
    i_max = 0
    i_set: list[int] = []
    for i0 in range(n):
        if f_p[i0] <= 0.0:
            continue
        members = _harmonic_members(f_p, i0)
        energy = float(np.sum(a_tilde[members])) if members else 0.0  # Formula 91
        if energy > best_energy:
            best_energy = energy
            i_set = members
            i_max = i0
    return i_set, i_max


def _amplitude(spec: np.ndarray, band: int) -> float:
    """Weighted, summed amplitude A(l, z) for one block/band (Clause 7.1.5)."""
    f_p, a_i = _pick_peaks(spec)
    if f_p.size == 0:
        return 0.0
    f_max = _F_MAX[band]
    r_max = _R_MAX[band]

    # High-modulation-rate weighting (Formula 83).
    a_tilde = np.zeros_like(a_i)
    for j in range(f_p.size):
        f = f_p[j]
        if f <= _DELTA_F:
            a_tilde[j] = 0.0
        elif f <= f_max:
            a_tilde[j] = a_i[j] * r_max
        else:
            a_tilde[j] = _g_weight(f, f_max, _Q1_HI, _Q2_HI[band]) * a_i[j] * r_max

    i_set, i_max = _fundamental_set(f_p, a_tilde)
    if not i_set:
        return 0.0
    f_fund = f_p[i_max]

    # Centre-of-gravity peak weighting (Formulae 92-94).
    a_set = a_tilde[i_set]
    f_set = f_p[i_set]
    f_peak = f_set[int(np.argmax(a_set))]
    cg = float(np.sum(f_set * a_set) / (np.sum(a_set) + _EPS))
    w_peak = 1.0 + 0.1 * abs(cg - f_peak) ** 0.749
    a_hat_sum = float(w_peak * np.sum(a_set))  # sum over I_max of A_hat_i

    # Low-modulation-rate weighting (Formula 95).
    if f_fund <= _DELTA_F:
        amp = 0.0
    elif f_fund <= f_max:
        amp = _g_weight(f_fund, f_max, _Q1_LO, _Q2_LO[band]) * a_hat_sum
    else:
        amp = a_hat_sum
    return amp if amp >= _A_THRESHOLD else 0.0


def _amplitudes(spectra: np.ndarray) -> np.ndarray:
    """Uncalibrated amplitude A(l, z) for every block/band (Clause 7.1.5)."""
    n_blocks = spectra.shape[0]
    out = np.zeros((n_blocks, _CBF))
    for lb in range(n_blocks):
        for band in range(_CBF):
            out[lb, band] = _amplitude(spectra[lb, :, band], band)
    return out


# --------------------------------------------------------------------------
# Time-dependent specific roughness + aggregation (Clause 7.1.7-7.1.10)
# --------------------------------------------------------------------------


def _time_dependent(
    a_lz: np.ndarray, block_times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Time-dependent specific roughness R'(l50, z) (Formulae 103-110)."""
    t_end = float(block_times[-1])
    n50 = int(np.floor(t_end * _R_S50)) + 1
    grid = np.arange(n50) / _R_S50
    if block_times.size >= 2:
        # Piecewise cubic Hermite (Clause 7.1.7), all 53 bands in one
        # interpolator; pchip treats every column independently, so the
        # result is bit-identical to a per-band loop.
        r_est = PchipInterpolator(block_times, a_lz, axis=0)(grid)
    else:
        r_est = np.zeros((n50, _CBF))
    r_est = np.maximum(r_est, 0.0)  # negatives -> 0 (Clause 7.1.7)

    # Distribution-dependent nonlinear transform + calibration (Formulae 104-108).
    r_rms = np.sqrt(np.sum(r_est**2, axis=1) / _CBF)  # Formula 107
    r_bar = np.sum(r_est, axis=1) / _CBF  # Formula 108
    b_l = np.zeros(n50)
    nz = np.abs(r_bar) > _EPS
    b_l[nz] = r_rms[nz] / r_bar[nz]  # Formula 106
    e_l = 0.37106 * (np.tanh(1.6407 * (b_l - 2.5804)) + 1.0) * 0.5 + 0.58449  # F.105
    r_hat = _C_R * (r_est ** e_l[:, None])  # Formula 104

    # Asymmetric first-order smoothing (Formulae 109-110).
    r_time = np.zeros_like(r_hat)
    r_time[0] = r_hat[0]
    for lb in range(1, n50):
        tau = np.where(r_hat[lb] >= r_time[lb - 1], 0.0625, 0.5)  # Formula 110
        a_coef = np.exp(-1.0 / (_R_S50 * tau))
        r_time[lb] = r_hat[lb] * (1.0 - a_coef) + r_time[lb - 1] * a_coef  # F.109
    return r_time, grid


def _aggregate(
    r_time: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Average specific roughness, R(l50) and single value (Clause 7.1.8-7.1.10)."""
    kept = r_time[_TRANSIENT:] if r_time.shape[0] > _TRANSIENT else r_time
    r_spec = np.mean(kept, axis=0) if kept.shape[0] else np.zeros(_CBF)  # 7.1.8
    r_vs_time = _DZ * np.sum(r_time, axis=1)  # Formula 111
    kept_time = (
        r_vs_time[_TRANSIENT:] if r_vs_time.size > _TRANSIENT else r_vs_time
    )
    r_single = float(np.percentile(kept_time, 90)) if kept_time.size else 0.0  # 7.1.10
    return r_single, np.asarray(r_spec, dtype=np.float64), r_vs_time


def roughness_ecma(
    signal_in: Signal | np.ndarray,
    fs: float | None = None,
    field: Literal["free", "diffuse"] = "free",
) -> EcmaRoughness:
    """Psychoacoustic roughness per ECMA-418-2:2025 (Sottek Hearing Model).

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
    :return: An :class:`EcmaRoughness` with the single value R (Clause 7.1.10),
        the average specific roughness R'(z) (Clause 7.1.8) and the
        time-dependent roughness R(l50) (Formula 111).

    A 1 kHz carrier 100 %-amplitude-modulated at 70 Hz with an overall level
    of 60 dB SPL yields 1 asper (Clause 7 calibration; reproduced to 0.9999
    asper with the tabulated c_R of Formula (104)).
    """
    if field not in ("free", "diffuse"):
        raise ValueError("field must be 'free' or 'diffuse'")
    fs = resolve_fs(signal_in, fs, name="signal_in")
    x = apply_calibration(signal_in, require_1d_signal(_typesignal(np.asarray(signal_in))))
    if x.size == 0:
        raise ValueError("signal must not be empty")
    fs = float(fs)
    if fs <= 0.0:
        raise ValueError("fs must be positive")
    if fs != _FS:
        x = signal.resample(x, round(x.size * _FS / fs))

    envelopes, basis, block_times, _ = _front_end(x, field)
    spectra = _noise_reduced_spectra(envelopes, basis)
    a_lz = _amplitudes(spectra)
    r_time, grid = _time_dependent(a_lz, block_times)
    r_single, r_spec, r_vs_time = _aggregate(r_time)
    return EcmaRoughness(
        roughness=r_single,
        specific_roughness=r_spec,
        bark=_Z.copy(),
        centre_frequencies=_F_CENTRE.copy(),
        time=grid,
        roughness_vs_time=r_vs_time,
        specific_roughness_vs_time=r_time,
        field=field,
    )
