#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Psychoacoustic fluctuation strength per ECMA-418-2:2025 (4th ed., Clause 9).

Clean-room implementation of the fluctuation-strength signal chain of
ECMA-418-2:2025 (Clause 9, Sottek Hearing Model). The shared auditory
front-end (Clause 5: outer/middle-ear filter, 53-band auditory filter bank,
compressive nonlinearity of Formulae 23-25) is reused from
:mod:`phonometry.psychoacoustics.loudness.ecma`; this module adds the
fluctuation-strength chain, which mirrors the roughness chain of Clause 7 but
replaces the DFT-based envelope analysis with High-resolution Spectral
Analysis (HSA):

* fluctuation-strength zero-padding (Clause 5.1.2.2) and segmentation
  (Clause 5.1.5.2) with the fixed block/hop :math:`s_\mathrm{b} = 65536` /
  :math:`s_\mathrm{h} = 16384`
  (Clause 9.1.1);
* the Hilbert envelope of each critical-band block and a factor-32
  downsampling to 1500 Hz (Clause 9.1.2, Formula 119);
* the envelope-dependent analysis windows with quieter-period detection
  (Clause 9.1.3, Formula 120);
* the HSA of the windowed envelopes -- a least-squares fit of window-kernel
  spectral line pairs to the k = 0..48 DFT bins (Clause 9.1.4,
  Formulae 121-142);
* the identification of prominent spectral line pairs from the local maxima
  of the power spectrum and the local minima of the HSA error over a
  logarithmic modulation-rate grid (Clause 9.1.5, Formulae 143-146);
* the band-pass modulation-rate weighting (Clause 9.1.6, Formulae 147-148),
  the damped-Newton fine tuning of the dominant modulation rate
  (Clause 9.1.7, Formulae 149-152), the harmonic analysis (Clause 9.1.8,
  Formulae 153-156) and the centre-of-gravity weighting (Clause 9.1.9,
  Formulae 157-158);
* the scaling with the HSA-based specific loudness (Clause 9.1.10,
  Formulae 159-161); and
* the interpolation to 50 Hz, the distribution-dependent nonlinear transform
  with the calibration constant ``c_F``, the first-order smoothing
  (Clause 9.1.11, Formulae 162-168) and the aggregation into the average
  specific fluctuation strength F'(z) (Clause 9.1.12), the time-dependent
  fluctuation strength F(l50) (Clause 9.1.13, Formula 169) and the
  representative 90th-percentile single value F (Clause 9.1.14).

The API is monaural: the quadratic-mean binaural combination of
Formula (170) (Clause 9.1.15) is not implemented -- analyse each channel
separately.

The calibration constant ``c_F`` of Formula (163) is the standard's tabulated
value (not reverse-fit), and the chain reproduces the Clause 9 reference
point: a 1 kHz carrier 100 %-amplitude-modulated at 4 Hz with an overall
sound pressure level of 60 dB SPL converges to 0.9958 vacil_HMS against the
defined 1 vacil_HMS as the 90th percentile settles (0.9931 for a 5 s signal,
0.9957 at 8 s, 0.9958 by 12 s). Footnote 47 allows a +/-0.25 % adjustment
of c_F, which is not used. The level convention follows Clause 7/9 as
established for the roughness metric: the stated 60 dB is the overall RMS
level of the modulated signal, not the carrier level.

Clause 9 interpretation notes (all resolved by internal consistency and
pinned by the calibration signal; the confirmed defects are recorded in
``docs/ERRATA.md``):

* Formula (127) prints the phase factor
  :math:`\exp(-j 2\pi f_n (\tilde{s}_\mathrm{b} - n_\mathrm{ze} + n_\mathrm{zb} - 1))`; the DFT of
  the rectangular analysis window requires :math:`\pi`
  in place of :math:`2\pi` (with :math:`2\pi` the HSA cannot reproduce the
  very
  spectra it fits, breaking the exact-recovery property claimed for it).
* Formula (144) subtracts 1 from the three-bin centroid before scaling by
  ``delta_f``; with the 0-based bin convention stated below Formula (122)
  (bin k maps to :math:`k \tilde{r}_\mathrm{s} / \tilde{s}_\mathrm{b}`) that offset shifts
  every modulation
  rate one bin low, so the centroid is used without the offset.
* Clause 9.1.7 states the Newton constants (differential step 1e-5, damped
  step cap 2e-4, stop tolerance 1e-7) without units; read in Hz they cap the
  total displacement at 2e-3 Hz, which makes the fine tuning inert and the
  failure check against 1.25 * delta_f (~0.92 Hz) unreachable. They are
  self-consistent as normalized frequencies (f / r~_s), which is how this
  module applies them.
* The amplitude of a spectral line pair (Formulae 146-147, 155, 157-160) is
  taken as the squared magnitude of the half-line solution components of
  Formula (123), :math:`x_{2m}^2 + x_{2m+1}^2 = |\hat{p}_m|^2 / 4`: with
  that reading
  Formula (160) is exactly the RMS of the modelled band signal (the loudness
  chain of Formulae 22-23 applied to the harmonic complex) and the tabulated
  c_F reproduces the calibration signal.
* The < 0.125 Hz discard of Clause 9.1.7 names ``f_c,1,opt``; it is applied
  here to the rate that survives the failure check (the original
  ``f~_c,imax`` when the fine tuning was cancelled), since discarding a
  block on a diverged optimizer output would drop healthy modulation.
* For signals shorter than the ~0.74 s transient (fewer than 37 frames at
  50 Hz), the Clause 9.1.12/9.1.14 discard of l50 = 0..35 would leave no
  frames; the aggregation then falls back to all frames instead of
  reporting an empty (zero) result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal
from scipy.interpolate import PchipInterpolator

from ...io._resolve import apply_calibration, resolve_fs
from ...io._signal import Signal

if TYPE_CHECKING:
    from matplotlib.axes import Axes

from ..._internal.utils import _typesignal
from ..._internal.validation import require_1d_signal, require_positive
from ..loudness.ecma import (
    _CBF,
    _F_CENTRE,
    _FS,
    _Z,
    _auditory_bandpass,
    _ear_filter_sos,
    _fade_in,
    _specific_basis_loudness,
)

# --------------------------------------------------------------------------
# Fluctuation-strength constants (Clause 9.1.1-9.1.2)
# --------------------------------------------------------------------------

_SB = 65536  # block size s_b (Clause 9.1.1)
_SH = 16384  # hop size s_h (75 % overlap)
_DOWN = 32  # envelope downsampling factor (Clause 9.1.2, footnote 35)
_SB_TILDE = _SB // _DOWN  # 2048, downsampled block size s~_b
_R_TILDE = _FS / _DOWN  # 1500 Hz downsampled rate r~_s
_DELTA_F = _R_TILDE / _SB_TILDE  # 0.7324 Hz DFT resolution
_N_K = 49  # DFT bins k = 0..48 used for the HSA (Clause 9.1.4)
_R_S50 = 50.0  # interpolated rate r_s50 (Clause 9.1.11)
_DZ = 0.5  # critical-band overlap dz (Clause 9.1.13)
_TRANSIENT = 36  # discard l50 in [0, 35] (Clause 9.1.12/9.1.14)
_TAU = 0.75  # smoothing time constant tau (Formula 168) [s]

# Calibration constant c_F (Formula 163, tabulated; adjustable +/- 0.25 %).
_C_F = 0.003840572

#: Prominence criterion (Clause 9.2): a signal has a prominent fluctuation
#: strength when the single value F (Clause 9.1.14) exceeds this value, in
#: vacil_HMS.
PROMINENT_FLUCTUATION_STRENGTH_VACIL = 0.2
_A_THRESHOLD = 5.2519  # amplitude gate below Formula (161)

# The standard's epsilon_0: the smallest positive number with 1 + eps > 1
# (Clause 9.1.4, below Formula 127).
_EPS0 = float(np.finfo(np.float64).eps)

# Analysis-window parameters (Clause 9.1.3).
_GUARD = _SB_TILDE // 32  # 64 guard zeros against Hilbert edge distortion
_MEDIAN_LEN = _SB_TILDE // 64 + 1  # 33, envelope moving-median length
_ENV_QUIET_PA = 5e-6  # quiet-block envelope threshold [Pa] (9.1.3.2)
_MIN_RUN = _SB_TILDE * 5 // 32  # 320: minimum quieter-period length and
# minimum count of window ones (9.1.3.4/9.1.3.6)
_MIN_N2 = _SB_TILDE // 4 - 1  # 511: minimum window end index (9.1.3.6)
_REG_MARGIN = _SB_TILDE * 5 // 1024  # 10, regression margin (9.1.3.6)
_REG_MIN_RSD = 1e-3  # minimum relative standard deviation (9.1.3.6)

# Prominent-line identification (Clause 9.1.5).
_PHI_MIN = 0.15  # Phi_E,min of Formula (143)
# Logarithmic modulation-rate grid f_i = 0.25 * 2^((i-2)/3), i = 1..16.
_F_GRID = 0.25 * 2.0 ** ((np.arange(1, 17) - 2.0) / 3.0)

# Modulation-rate weighting (Formula 148).
_Q1_LO, _Q2_LO = 0.33048, 0.85902
_Q1_HI, _Q2_HI = 0.21792, 4.6728
_F_MAX_W = 4.8659  # modulation rate of the weighting maximum [Hz]
# Carrier-frequency factor of the high-rate branch of Formula (148).
_W_CARRIER = 1.0 / (1.0 + 0.092623 * np.abs(np.log2(_F_CENTRE / 1000.0)) ** 1.24)

# Newton fine tuning (Clause 9.1.7); the constants are normalized
# frequencies f / r~_s (see the module docstring).
_NEWTON_DX = 1e-5
_NEWTON_CAP = 2e-4
_NEWTON_TOL = 1e-7
_NEWTON_KMAX = 40
_F_OPT_MIN = 0.125  # minimum accepted modulation rate [Hz] (9.1.7)

# Harmonic analysis (Clause 9.1.8).
_MAX_ORDER_SEED = 3  # tested orders of the tuned component
_MAX_ORDER = 5  # highest harmonic order considered
_HARMONIC_TOL = 0.04  # Formula (154)

# Centre-of-gravity weighting (Formula 158).
_W_BW_GAIN = 0.79577
_W_BW_EXP = 0.43461

_B_MEDIAN_LEN = 71  # moving-median length for B(l50) (below Formula 167)


@dataclass(frozen=True)
class EcmaFluctuationStrength:
    """Result of an ECMA-418-2:2025 (Sottek) fluctuation-strength calculation.

    ``fluctuation_strength`` is the single representative fluctuation
    strength F in vacil_HMS (the 90th percentile of F(l50), Clause 9.1.14).
    ``specific_fluctuation_strength`` is the average specific fluctuation
    strength F'(z) in vacil_HMS/Bark_HMS over the 53 auditory bands
    (Clause 9.1.12), with ``bark`` the critical-band-rate scale z
    (0.5..26.5 Bark_HMS) and ``centre_frequencies`` the band centre
    frequencies F(z). ``time`` and ``fluctuation_strength_vs_time`` hold the
    time-dependent fluctuation strength F(l50) at 50 Hz (Formula 169);
    ``specific_fluctuation_strength_vs_time`` is the time-dependent specific
    fluctuation strength F'(l50, z) (Formula 168) of shape
    ``(n_times, 53)``. ``field`` records the assumed sound field.
    """

    fluctuation_strength: float
    specific_fluctuation_strength: np.ndarray
    bark: np.ndarray
    centre_frequencies: np.ndarray
    time: np.ndarray
    fluctuation_strength_vs_time: np.ndarray
    specific_fluctuation_strength_vs_time: np.ndarray
    field: str

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes | np.ndarray:
        """Plot the fluctuation-strength result (see :mod:`phonometry._plot.psychoacoustics`).

        Draws the time-dependent fluctuation strength F(l50) and a
        specific-fluctuation-strength heatmap. Requires matplotlib
        (``pip install phonometry[plot]``).
        """
        from ..._i18n import check_language
        from ..._plot.psychoacoustics import plot_ecma_fluctuation_strength

        return plot_ecma_fluctuation_strength(self, ax=ax, language=check_language(language), **kwargs)


# --------------------------------------------------------------------------
# Front-end: downsampled Hilbert envelopes per block (Clause 5, 9.1.2)
# --------------------------------------------------------------------------


def _front_end(x: np.ndarray, field: str) -> tuple[np.ndarray, np.ndarray]:
    """Downsampled envelopes and block times for the fluctuation chain.

    Applies the fluctuation-strength zero-padding (Clause 5.1.2.2), ear
    filter and auditory band-pass (Clause 5.1.3-5.1.4), then per band
    segments at the fixed block/hop (Clause 5.1.5.2) and forms the
    downsampled Hilbert envelope (Clause 9.1.2, Formula 119).

    Returns ``(envelopes[L, 2048, Z], block_times[L])``.
    """
    # Fade-in (Formula 1) then the start pad only (Clause 5.1.2.2).
    padded = np.concatenate([np.zeros(_SB), _fade_in(x)])
    n_pad = padded.size
    p_om = signal.sosfilt(_ear_filter_sos(field), padded)

    starts = list(range(0, max(n_pad - _SB, 0) + 1, _SH))
    if not starts:  # pragma: no cover - n_pad >= _SB by construction
        starts = [0]
    if n_pad - _SB > starts[-1]:
        # Flush-to-end last block (Clause 5.1.5.2; see docs/ERRATA.md on the
        # printed l_last formula). Its start is n_samples, so the Formula
        # (162) time grid ends exactly at n_samples / r_s.
        starts.append(n_pad - _SB)
    start_idx = np.array(starts, dtype=np.int64)
    n_blocks = start_idx.size
    idx = start_idx[:, None] + np.arange(_SB)[None, :]

    envelopes = np.zeros((n_blocks, _SB_TILDE, _CBF))
    for band in range(_CBF):
        p_band = _auditory_bandpass(p_om, band)
        seg = p_band[idx]
        analytic = signal.hilbert(seg, axis=1)  # Formula 119
        envelopes[:, :, band] = np.abs(analytic)[:, ::_DOWN]  # downsample x32
    block_times = start_idx.astype(np.float64) / _FS
    return envelopes, block_times


# --------------------------------------------------------------------------
# Envelope-dependent analysis window (Clause 9.1.3)
# --------------------------------------------------------------------------


def _moving_median(x: np.ndarray, length: int) -> np.ndarray:
    """Centred moving median with truncated end windows (footnote 37)."""
    half = length // 2
    n = x.size
    out = np.empty(n, dtype=np.float64)
    if n >= length:
        out[half : n - half] = np.median(sliding_window_view(x, length), axis=1)
        for i in range(half):
            out[i] = np.median(x[: i + half + 1])
            out[n - 1 - i] = np.median(x[n - 1 - i - half :])
    else:
        # Reached by the length-71 B(l50) filter on very short signals
        # (fewer than 71 interpolated frames, i.e. under ~1.4 s).
        for i in range(n):
            out[i] = np.median(x[max(0, i - half) : min(n, i + half + 1)])
    return out


def _longest_quiet_run(quiet: np.ndarray) -> tuple[int, int] | None:
    """Longest run of ``True`` longer than the 9.1.3.4 minimum, or ``None``.

    ``quiet`` covers the updated interval only; runs cannot touch its edges
    (both edge samples are at or above the threshold by construction). Ties
    keep the first (earliest) run.
    """
    edges = np.flatnonzero(np.diff(np.concatenate([[0], quiet, [0]])))
    if edges.size == 0:
        return None
    begins, ends = edges[::2], edges[1::2] - 1
    lengths = ends - begins + 1
    best = int(np.argmax(lengths))
    if lengths[best] <= _MIN_RUN:
        return None
    return int(begins[best]), int(ends[best])


def _analysis_window(env: np.ndarray) -> tuple[int, int] | None:
    """Analysis-window zeros ``(n_zb, n_ze)`` for one block/band (9.1.3).

    Returns ``None`` when the entire block is a quieter period (the window
    is all zeros and the block carries no modulation information).
    """
    if float(env.max()) <= _ENV_QUIET_PA:  # smoothing cannot raise the max
        return None
    smoothed = np.round(_moving_median(env, _MEDIAN_LEN), 8)  # 9.1.3.2
    pw = smoothed.copy()
    pw[:_GUARD] = 0.0  # initial window, n_zb = n_ze = 64
    pw[_SB_TILDE - _GUARD :] = 0.0
    p_max = float(pw.max())
    if p_max <= _ENV_QUIET_PA:
        return None

    thr = round(0.01 * p_max, 8)  # 9.1.3.3
    above = np.flatnonzero(pw >= thr)
    if above.size == 0:  # pragma: no cover - thr << p_max
        return None
    n_zb = int(above[0])
    n_ze = _SB_TILDE - 1 - int(above[-1])
    n2 = _SB_TILDE - 1 - n_ze

    run = _longest_quiet_run(pw[n_zb : n2 + 1] < thr)  # 9.1.3.4
    if run is not None:
        qpb, qpe = run[0] + n_zb, run[1] + n_zb
        # Keep the longer of the parts left/right of the quieter period,
        # with 64 guard zeros against the quiet edge (9.1.3.5).
        if (qpb - (n_zb + _GUARD)) > ((n2 - _GUARD) - qpe):
            n_ze = _SB_TILDE - 1 - qpb + _GUARD
        else:
            n_zb = qpe + _GUARD
        n2 = _SB_TILDE - 1 - n_ze

    # Validity checks (9.1.3.6).
    if n2 - n_zb + 1 < _MIN_RUN or n2 < _MIN_N2:
        return None
    seg = smoothed[n_zb + _REG_MARGIN : n2 - _REG_MARGIN + 1]
    if seg.size < 2:  # pragma: no cover - implied by the length checks
        return None
    # Closed-form least-squares line fit over the uniform grid 0..n-1
    # (equivalent to a first-order polyfit, without the general-purpose
    # Vandermonde/lstsq overhead in this per-block, per-band hot path).
    n = seg.size
    grid = np.arange(n, dtype=np.float64)
    x_mean = 0.5 * (n - 1)
    mean = float(np.mean(seg))
    dx = grid - x_mean
    slope = float(np.dot(dx, seg)) / (n * (n * n - 1.0) / 12.0)
    intercept = mean - slope * x_mean
    resid = seg - (slope * grid + intercept)
    if mean <= 0.0 or float(np.std(resid)) / mean < _REG_MIN_RSD:
        return None
    return n_zb, n_ze


# --------------------------------------------------------------------------
# High-resolution Spectral Analysis (Clause 9.1.4)
# --------------------------------------------------------------------------


def _window_kernel(fc_hz: float, n_zb: int, n_ze: int, k: np.ndarray) -> np.ndarray:
    """Spectral kernel W_E,l,z,fc(k) of the analysis window (Formula 127).

    The DFT of the rectangular analysis window modulated to ``+fc``. The
    printed phase factor uses 2*pi; the DFT of the window requires pi (see
    the module docstring and docs/ERRATA.md).
    """
    fn = k / _SB_TILDE - fc_hz / _R_TILDE + _EPS0
    ones = _SB_TILDE - n_ze - n_zb
    return np.asarray(
        np.exp(-1j * np.pi * fn * (_SB_TILDE - n_ze + n_zb - 1))
        * np.sin(np.pi * fn * ones)
        / np.sin(np.pi * fn),
        dtype=np.complex128,
    )


def _round_half_away(v: float) -> int:
    """Round to nearest, ties away from zero (footnote 42)."""
    return math.floor(v + 0.5) if v >= 0.0 else math.ceil(v - 0.5)


def _k_limit(f_max_hz: float) -> int:
    """Number of DFT bins K_L used by the HSA fit (Formula 125)."""
    return min(max(17, _round_half_away(f_max_hz / _DELTA_F) + 8), _N_K)


def _hsa_solve(
    p_spec: np.ndarray,
    phi: np.ndarray,
    rates: np.ndarray,
    n_zb: int,
    n_ze: int,
) -> tuple[np.ndarray, float]:
    """HSA least-squares fit for the line pairs at ``rates`` (9.1.4).

    ``p_spec``/``phi`` are the k = 0..48 spectrum and power spectrum of the
    windowed envelope (Formulae 121-122); ``rates`` are the positive
    modulation rates f_c,1..f_c,Mc in Hz. Returns the solution vector ``x``
    of Formula (123) (constant part, then Re/Im half-line components per
    pair) and the error E_l,z(f_c) of Formula (135).
    """
    kl = _k_limit(float(np.max(rates)))
    k = np.arange(kl, dtype=np.float64)
    n_pairs = rates.size
    cols = np.empty((2 * n_pairs + 1, kl), dtype=np.complex128)
    cols[0] = _window_kernel(0.0, n_zb, n_ze, k)
    for m, f in enumerate(rates):
        w_pos = _window_kernel(float(f), n_zb, n_ze, k)
        w_neg = _window_kernel(-float(f), n_zb, n_ze, k)
        cols[2 * m + 1] = w_pos + w_neg  # W+ (Formula 128)
        cols[2 * m + 2] = np.conj(w_pos - w_neg)  # W-* (Formulae 126/129)

    # Formula (131)/(134): the "real type" indices I_R are i = 1 and even i
    # (1-based), i.e. index 0 and odd indices here.
    n_x = 2 * n_pairs + 1
    real_type = np.array([i == 0 or i % 2 == 1 for i in range(n_x)])
    re, im = np.real(cols), np.imag(cols)
    a_same = re @ re.T + im @ im.T
    a_cross = im @ re.T + re @ im.T
    same = real_type[:, None] == real_type[None, :]
    a_mat = np.where(same, a_same, a_cross)
    b_same = re @ np.real(p_spec[:kl]) + im @ np.imag(p_spec[:kl])
    b_cross = re @ np.imag(p_spec[:kl]) + im @ np.real(p_spec[:kl])
    b_vec = np.where(real_type, b_same, b_cross)

    try:
        x = np.linalg.solve(a_mat, b_vec)
    except np.linalg.LinAlgError:  # near-coincident rates
        x = np.linalg.lstsq(a_mat, b_vec, rcond=None)[0]
    err = float(np.sum(phi[:kl]) + x @ a_mat @ x - 2.0 * (b_vec @ x))
    return x, err


_K_ARR = np.arange(_N_K, dtype=np.float64)


def _hsa_single(
    p_spec: np.ndarray,
    phi: np.ndarray,
    f_hz: float,
    n_zb: int,
    n_ze: int,
    w0: np.ndarray,
) -> tuple[np.ndarray, float]:
    """One-line-pair HSA in closed form (Clause 9.1.4.1, Formulae 136-142).

    The analytic Cramer solution of the 3x3 system; equivalent to
    :func:`_hsa_solve` with one rate but without the matrix assembly (this is
    the hot path of the grid search, the Newton tuning and the harmonic
    refinement). ``w0`` is the precomputed k = 0..48 kernel at rate 0.
    """
    kl = _k_limit(f_hz)
    k = _K_ARR[:kl]
    w_0 = w0[:kl]
    w_pos = _window_kernel(f_hz, n_zb, n_ze, k)
    w_neg = _window_kernel(-f_hz, n_zb, n_ze, k)
    w_plus = w_pos + w_neg  # Formula 128
    w_minus_c = np.conj(w_pos - w_neg)  # Formulae 126/129
    p = p_spec[:kl]

    a11 = float(np.sum(w_0.real**2 + w_0.imag**2))  # Formula 137
    a22 = float(np.sum(w_plus.real**2 + w_plus.imag**2))
    a33 = float(np.sum(w_minus_c.real**2 + w_minus_c.imag**2))
    a12 = float(np.vdot(w_plus, w_0).real)  # Formula 138
    a13 = float(np.sum(w_0 * w_minus_c).imag)
    a23 = float(np.sum(w_plus * w_minus_c).imag)
    b1 = float(np.vdot(w_0, p).real)  # Formula 139
    b2 = float(np.vdot(w_plus, p).real)
    b3 = float(np.sum(w_minus_c * p).imag)

    det = (
        a11 * (a22 * a33 - a23**2)
        - a12 * (a12 * a33 - a23 * a13)
        + a13 * (a12 * a23 - a22 * a13)
    )  # Formula 141
    if abs(det) < _EPS0:  # pragma: no cover - degenerate window
        return _hsa_solve(p_spec, phi, np.array([f_hz]), n_zb, n_ze)
    x1 = (
        b1 * (a22 * a33 - a23**2)
        - a12 * (b2 * a33 - a23 * b3)
        + a13 * (b2 * a23 - a22 * b3)
    ) / det  # Formula 140
    x2 = (
        a11 * (b2 * a33 - a23 * b3)
        - b1 * (a12 * a33 - a23 * a13)
        + a13 * (a12 * b3 - b2 * a13)
    ) / det
    x3 = (
        a11 * (a22 * b3 - b2 * a23)
        - a12 * (a12 * b3 - b2 * a13)
        + b1 * (a12 * a23 - a22 * a13)
    ) / det
    err = float(
        np.sum(phi[:kl])
        + a11 * x1**2
        + a22 * x2**2
        + a33 * x3**2
        + 2.0
        * (a12 * x1 * x2 + a13 * x1 * x3 + a23 * x2 * x3 - b1 * x1 - b2 * x2 - b3 * x3)
    )  # Formula 142
    return np.array([x1, x2, x3]), err


def _hsa_error(
    p_spec: np.ndarray,
    phi: np.ndarray,
    f_hz: float,
    n_zb: int,
    n_ze: int,
    w0: np.ndarray,
) -> float:
    """Single-line-pair HSA error E_l,z((0, f)) (Clause 9.1.4.1)."""
    return _hsa_single(p_spec, phi, f_hz, n_zb, n_ze, w0)[1]


def _pair_amplitudes(x: np.ndarray) -> np.ndarray:
    r"""Squared line-pair magnitudes from a solution vector (Formula 123).

    :math:`x_{2m}^2 + x_{2m+1}^2 = \lvert \hat{p}_m \rvert^2 / 4` per pair;
    see the module docstring
    for why the half-line magnitude is the consistent amplitude convention.
    """
    return np.asarray(x[1::2] ** 2 + x[2::2] ** 2, dtype=np.float64)


# --------------------------------------------------------------------------
# Prominent line pairs, weighting, fine tuning, harmonics (9.1.5-9.1.9)
# --------------------------------------------------------------------------


def _w_lh(f_hz: float, band: int) -> float:
    """Band-pass modulation-rate weighting w_lh (Formula 148)."""
    if f_hz <= 0.0:  # first case of Formula (148); defensive, rates are > 0
        return 0.0
    inner = (f_hz / _F_MAX_W - _F_MAX_W / f_hz) ** 2
    if f_hz <= _F_MAX_W:
        return float(1.0 / (1.0 + inner * _Q1_LO**2) ** _Q2_LO)
    return float(_W_CARRIER[band] / (1.0 + inner * _Q1_HI**2) ** _Q2_HI)


def _peak_candidates(phi: np.ndarray) -> np.ndarray:
    """Modulation rates of the local power-spectrum maxima (Formulae 143-144)."""
    peaks, _ = signal.find_peaks(phi)
    if peaks.size == 0:
        return np.empty(0)
    gate = max(0.001 * phi[0], _PHI_MIN)  # Formula 143
    peaks = peaks[phi[peaks] >= gate]
    if peaks.size == 0:
        return np.empty(0)
    f_p = np.empty(peaks.size)
    for j, kp in enumerate(peaks):
        w = phi[kp - 1 : kp + 2]
        centroid = float(np.sum(np.arange(kp - 1, kp + 2) * w) / np.sum(w))
        # Formula (144) without the printed -1 bin offset (see ERRATA):
        # with the 0-based bin convention of Formula (122) the centroid maps
        # directly to centroid * delta_f.
        f_p[j] = centroid * _DELTA_F
    return f_p


def _grid_minimum(
    p_spec: np.ndarray, phi: np.ndarray, n_zb: int, n_ze: int, w0: np.ndarray
) -> float | None:
    """Modulation rate of the deepest local HSA-error minimum (9.1.5)."""
    errors = np.array(
        [_hsa_error(p_spec, phi, float(f), n_zb, n_ze, w0) for f in _F_GRID]
    )
    interior = np.arange(1, _F_GRID.size - 1)
    is_min = (errors[interior] < errors[interior - 1]) & (
        errors[interior] < errors[interior + 1]
    )
    candidates = interior[is_min]
    if candidates.size == 0:
        return None
    best = candidates[int(np.argmin(errors[candidates]))]
    return float(_F_GRID[best])


def _select_line_pairs(
    p_spec: np.ndarray, phi: np.ndarray, n_zb: int, n_ze: int, w0: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Preselected modulation rates and amplitudes A_i (Clause 9.1.5).

    Returns ``(rates, amplitudes)`` after the Formula (146) gate, or ``None``
    when the block carries no modulation.
    """
    f_p = _peak_candidates(phi)
    f_min = _grid_minimum(p_spec, phi, n_zb, n_ze, w0)

    if f_min is None:
        if f_p.size == 0:
            return None
        rates = np.sort(f_p)
        x, _ = _hsa_solve(p_spec, phi, rates, n_zb, n_ze)
    else:
        dup = np.abs(f_min - f_p) < 1.25 * _DELTA_F  # Formula 145
        if dup.any():
            rates_a = np.sort(np.append(f_p[~dup], f_min))
            x_a, e_a = _hsa_solve(p_spec, phi, rates_a, n_zb, n_ze)
            rates_b = np.sort(f_p)
            x_b, e_b = _hsa_solve(p_spec, phi, rates_b, n_zb, n_ze)
            rates, x = (rates_a, x_a) if e_a <= e_b else (rates_b, x_b)
        else:
            rates = np.sort(np.append(f_p, f_min))
            x, _ = _hsa_solve(p_spec, phi, rates, n_zb, n_ze)

    amps = _pair_amplitudes(x)
    peak = float(np.max(amps))
    if peak <= 0.0:
        return None
    keep = amps > 0.05 * peak  # Formula 146
    return rates[keep], amps[keep]


def _newton_tune(
    p_spec: np.ndarray,
    phi: np.ndarray,
    f0_hz: float,
    n_zb: int,
    n_ze: int,
    w0: np.ndarray,
) -> float:
    """Damped-Newton fine tuning of the dominant rate (Formulae 149-152).

    Operates on the normalized modulation rate x = f / r~_s (see the module
    docstring on the units of the printed constants).
    """
    x = f0_hz / _R_TILDE
    for _ in range(1, _NEWTON_KMAX):
        e_mid = _hsa_error(p_spec, phi, x * _R_TILDE, n_zb, n_ze, w0)
        e_hi = _hsa_error(p_spec, phi, (x + _NEWTON_DX) * _R_TILDE, n_zb, n_ze, w0)
        e_lo = _hsa_error(p_spec, phi, (x - _NEWTON_DX) * _R_TILDE, n_zb, n_ze, w0)
        d1 = (e_hi - e_lo) / (2.0 * _NEWTON_DX)  # Formula 149
        d2 = (e_hi - 2.0 * e_mid + e_lo) / _NEWTON_DX**2  # Formula 150
        step = (
            0.25 * float(np.sign(d1)) * min(abs(d1) / (abs(d2) + _EPS0), _NEWTON_CAP)
        )  # Formula 152
        x -= step  # Formula 151
        if abs(step) <= _NEWTON_TOL:
            break
    return x * _R_TILDE


def _harmonic_complex(
    rates: np.ndarray,
    a_tilde: np.ndarray,
    f_opt: float,
    p_spec: np.ndarray,
    phi: np.ndarray,
    n_zb: int,
    n_ze: int,
    band: int,
    w0: np.ndarray,
) -> tuple[float, float, float]:
    r"""Harmonic analysis and weighting (Clauses 9.1.8-9.1.9).

    Returns ``(a_hat, harmonic_power, p0)`` where ``a_hat`` is the weighted
    sum of the harmonic complex (Formula 157) and ``harmonic_power`` is
    :math:`\hat{p}_0^2 + 2 \sum A_i` over the refined complex
    (Formulae 159-160).
    """
    best_energy = -1.0
    best: tuple[np.ndarray, np.ndarray] = (np.empty(0, dtype=np.intp), np.empty(0))
    best_order = 1
    for order in range(1, _MAX_ORDER_SEED + 1):
        f_base = f_opt / order
        ratios = np.array([_round_half_away(float(f) / f_base) for f in rates])
        ratios[ratios > _MAX_ORDER] = 0  # Formula 153
        valid = ratios > 0
        rel_err = np.ones(rates.size)
        rel_err[valid] = np.abs(rates[valid] / (ratios[valid] * f_base) - 1.0)
        members = valid & (rel_err < _HARMONIC_TOL)  # Formula 154
        energy = float(np.sum(a_tilde[members]))  # Formula 155
        if energy > best_energy:
            best_energy = energy
            best = (np.flatnonzero(members), ratios[members])
            best_order = order

    idx, ratios = best
    if idx.size == 0:  # pragma: no cover - order 1 always keeps the seed
        return 0.0, 0.0, 0.0
    f_fund = f_opt / best_order  # Formula 156

    # Refine the complex at the corrected rates with single-pair HSA runs,
    # "one spectral line pair for each order of interest" (9.1.8): members
    # sharing the same integer ratio collapse onto one corrected rate, so
    # the orders are deduplicated to avoid double-counting a line pair. The
    # constant part is the mean of the per-order predictions.
    orders = np.unique(ratios)
    p0_parts = np.empty(orders.size)
    amps = np.empty(orders.size)
    rates_new = np.empty(orders.size)
    for j, ratio in enumerate(orders):
        f_corr = float(ratio) * f_fund
        x, _ = _hsa_single(p_spec, phi, f_corr, n_zb, n_ze, w0)
        p0_parts[j] = x[0]
        amps[j] = float(_pair_amplitudes(x)[0])
        rates_new[j] = f_corr
    p0 = float(np.mean(p0_parts))
    a_tilde_new = amps * np.array([_w_lh(float(f), band) for f in rates_new])

    a_sum = float(np.sum(a_tilde_new))
    cog = float(np.sum(rates_new * a_tilde_new)) / (a_sum + _EPS0)
    w_bw = 1.0 + _W_BW_GAIN * abs(cog - f_opt) ** _W_BW_EXP  # Formula 158
    a_hat = w_bw * a_sum  # Formula 157
    harmonic_power = p0**2 + 2.0 * float(np.sum(amps))
    return a_hat, harmonic_power, p0


def _band_block(env: np.ndarray, band: int) -> tuple[float, float]:
    """Weighted amplitude and harmonic power for one block/band (9.1.3-9.1.9).

    Returns ``(a_hat, harmonic_power)``; both are 0 for blocks without
    modulation.
    """
    window = _analysis_window(env)
    if window is None:
        return 0.0, 0.0
    n_zb, n_ze = window
    windowed = env.copy()
    windowed[:n_zb] = 0.0
    if n_ze:
        windowed[-n_ze:] = 0.0
    p_spec = np.fft.rfft(windowed)[:_N_K]  # Formula 121
    phi = np.abs(p_spec) ** 2  # Formula 122
    w0 = _window_kernel(0.0, n_zb, n_ze, _K_ARR)

    selected = _select_line_pairs(p_spec, phi, n_zb, n_ze, w0)
    if selected is None:
        return 0.0, 0.0
    rates, amps = selected
    a_tilde = amps * np.array([_w_lh(float(f), band) for f in rates])
    i_max = int(np.argmax(a_tilde))  # Formula 147

    f_opt = _newton_tune(p_spec, phi, float(rates[i_max]), n_zb, n_ze, w0)
    if abs(f_opt - rates[i_max]) > 1.25 * _DELTA_F:
        f_opt = float(rates[i_max])  # optimization failed, keep the start
    else:
        x, _ = _hsa_single(p_spec, phi, f_opt, n_zb, n_ze, w0)
        rates = rates.copy()
        a_tilde = a_tilde.copy()
        rates[i_max] = f_opt
        a_tilde[i_max] = float(_pair_amplitudes(x)[0]) * _w_lh(f_opt, band)
    if f_opt < _F_OPT_MIN:
        return 0.0, 0.0

    a_hat, harmonic_power, _ = _harmonic_complex(
        rates, a_tilde, f_opt, p_spec, phi, n_zb, n_ze, band, w0
    )
    return a_hat, harmonic_power


def _amplitudes(envelopes: np.ndarray) -> np.ndarray:
    """Loudness-scaled amplitude A(l, z) for every block/band (9.1.10).

    Combines the per-band weighted amplitudes with the HSA-based specific
    loudness scaling of Formulae (159)-(161) and the 5.2519 amplitude gate.
    """
    n_blocks = envelopes.shape[0]
    a_hat = np.zeros((n_blocks, _CBF))
    power = np.zeros((n_blocks, _CBF))
    for lb in range(n_blocks):
        for band in range(_CBF):
            a_hat[lb, band], power[lb, band] = _band_block(envelopes[lb, :, band], band)

    # N'_HSA (Formulae 160-161): the Clause 5 nonlinearity + LTQ threshold
    # applied to the RMS of the harmonic complex.
    rms = np.sqrt(0.5 * power)
    n_hsa = np.column_stack(
        [_specific_basis_loudness(rms[:, band], band) for band in range(_CBF)]
    )
    n_max = np.max(n_hsa, axis=1)  # per block
    a_lz = (
        (1.0 / (power + _EPS0))
        * (n_hsa**2 / (n_max[:, None] + _EPS0))
        * _SB_TILDE
        * a_hat
    )  # Formula 159
    a_lz[a_lz < _A_THRESHOLD] = 0.0  # gate below Formula (161)
    return np.asarray(a_lz, dtype=np.float64)


# --------------------------------------------------------------------------
# Time-dependent specific fluctuation strength + aggregation (9.1.11-9.1.14)
# --------------------------------------------------------------------------


def _time_dependent(
    a_lz: np.ndarray, block_times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Time-dependent specific fluctuation strength F'(l50, z) (9.1.11)."""
    t_end = float(block_times[-1])
    n50 = int(np.floor(t_end * _R_S50)) + 1
    grid = np.arange(n50) / _R_S50
    if block_times.size >= 2:
        f_est = PchipInterpolator(block_times, a_lz, axis=0)(grid)
    else:  # pragma: no cover - two blocks minimum for nonzero signals
        f_est = np.zeros((n50, _CBF))
    f_est = np.maximum(f_est, 0.0)  # negatives -> 0 (Clause 9.1.11)

    # Distribution-dependent nonlinear transform + calibration (163-167).
    f_rms = np.sqrt(np.sum(f_est**2, axis=1) / _CBF)  # Formula 166
    f_bar = np.sum(f_est, axis=1) / _CBF  # Formula 167
    b_hat = f_rms / (f_bar + _EPS0)  # Formula 165
    b_smooth = _moving_median(b_hat, _B_MEDIAN_LEN)  # below Formula 167
    e_l = (
        0.37106 * (np.tanh(1.6407 * (b_smooth - 2.5804)) + 1.0) * 0.5 + 0.58449
    )  # Formula 164
    f_hat = _C_F * (f_est ** e_l[:, None])  # Formula 163

    # First-order low-pass smoothing, tau = 0.75 s (Formula 168).
    a_coef = math.exp(-1.0 / (_R_S50 * _TAU))
    f_time = np.empty_like(f_hat)
    f_time[0] = f_hat[0]
    for lb in range(1, n50):
        f_time[lb] = f_hat[lb] * (1.0 - a_coef) + f_time[lb - 1] * a_coef
    return f_time, grid


def _aggregate(
    f_time: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """F'(z), F(l50) and the single value F (Clauses 9.1.12-9.1.14)."""
    kept = f_time[_TRANSIENT:] if f_time.shape[0] > _TRANSIENT else f_time
    f_spec = np.mean(kept, axis=0) if kept.shape[0] else np.zeros(_CBF)
    f_vs_time = _DZ * np.sum(f_time, axis=1)  # Formula 169
    kept_time = f_vs_time[_TRANSIENT:] if f_vs_time.size > _TRANSIENT else f_vs_time
    f_single = float(np.percentile(kept_time, 90)) if kept_time.size else 0.0
    return f_single, np.asarray(f_spec, dtype=np.float64), f_vs_time


def fluctuation_strength_ecma(
    signal_in: Signal | np.ndarray,
    fs: float | None = None,
    field: Literal["free", "diffuse"] = "free",
) -> EcmaFluctuationStrength:
    """Psychoacoustic fluctuation strength per ECMA-418-2:2025 (Clause 9).

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
    :return: An :class:`EcmaFluctuationStrength` with the single value F
        (Clause 9.1.14), the average specific fluctuation strength F'(z)
        (Clause 9.1.12) and the time-dependent fluctuation strength F(l50)
        (Formula 169).
    :raises ValueError: for an empty, multichannel or non-finite signal, a
        non-finite or non-positive ``fs``, or an unknown ``field``.

    A 1 kHz carrier 100 %-amplitude-modulated at 4 Hz with an overall level
    of 60 dB SPL yields 1 vacil_HMS (Clause 9 calibration; reproduced to
    0.9958 vacil_HMS with the tabulated c_F of Formula (163) once the 90th
    percentile settles, by a 12 s signal).
    """
    if field not in ("free", "diffuse"):
        raise ValueError("field must be 'free' or 'diffuse'")
    fs = resolve_fs(signal_in, fs, name="signal_in")
    x = apply_calibration(signal_in, require_1d_signal(_typesignal(np.asarray(signal_in))))
    if x.size == 0:
        raise ValueError("signal must not be empty")
    if not np.all(np.isfinite(x)):
        raise ValueError("'signal' must be finite.")
    fs = require_positive(float(fs), "fs")
    if fs != _FS:
        x = signal.resample(x, max(1, round(x.size * _FS / fs)))

    envelopes, block_times = _front_end(x, field)
    a_lz = _amplitudes(envelopes)
    f_time, grid = _time_dependent(a_lz, block_times)
    f_single, f_spec, f_vs_time = _aggregate(f_time)
    return EcmaFluctuationStrength(
        fluctuation_strength=f_single,
        specific_fluctuation_strength=f_spec,
        bark=_Z.copy(),
        centre_frequencies=_F_CENTRE.copy(),
        time=grid,
        fluctuation_strength_vs_time=f_vs_time,
        specific_fluctuation_strength_vs_time=f_time,
        field=field,
    )
