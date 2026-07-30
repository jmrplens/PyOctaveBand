#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Weighting filters (A, B, C, D, G, AU, Z) and time weighting utilities.

A/C/Z per IEC 61672-1:2013; G (infrasound) per ISO 7196:1995.

B is the historical weighting of ANSI S1.4-1983 (Appendix C): the C curve
with one extra zero at the origin and one extra real pole at
``f5 = 158.48932 Hz``. It was dropped from the sound-level-meter standards
when IEC 61672-1 replaced IEC 60651 (first edition 2002) and is provided for
historical data and older national codes only.

AU per IEC 61012:1990: the A weighting cascaded with the U low-pass
(six poles, Table 2: a double real pole at -12 200 Hz and complex pairs at
-7 850 +/- j8 800 Hz and -2 900 +/- j12 150 Hz) for measuring audible sound
in the presence of ultrasound. It is flat relative to A up to 10 kHz and
cuts steeply above (U alone, Table 1: -2.8 dB at 12.5 kHz; -61.8 dB at
40 kHz). The Table 2 poles reproduce every Table 1 nominal value within
0.05 dB.

D per the withdrawn IEC 537:1976 (aircraft-noise weighting): implemented
from the widely published rational transfer function
``k s (s^2 + 6532 s + 4.0975e7) / ((s + 1776.3)(s + 7288.5)
(s^2 + 21514 s + 3.8836e8))``, with ``k`` renormalized to exactly 0 dB at
1 kHz. The standard itself is withdrawn and unavailable, so the constants
are corroborated against two independent implementations: SQAT
(``sound_level_meter/Gen_weighting_filters.m``: identical zeros and poles;
note its display-only ``freqResp`` line prints 1773.6 where its pole list,
and every other source, has 1776.3) and librosa (``librosa.D_weighting``,
an independent frequency-domain closed form; agreement within 0.002 dB
from 10 Hz to 20 kHz). The response also reproduces the tabulated IEC 537
curve republished in the NASA Handbook of Aircraft Noise Metrics
(NASA CR-3406, 1981, Table SLD-I) within 0.1 dB at every one-third-octave
frequency from 50 Hz to 10 kHz except 1600 Hz (0.15 dB) and 2500 Hz
(0.28 dB), where that table appears to round a different source curve.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import cast

import numpy as np
from scipy import signal

from .._internal.utils import _sos_initial_state, _sos_state_mismatch, _typesignal

try:
    from numba import jit as _numba_jit
except ImportError:  # pragma: no cover - depends on install extras
    # unused-ignore: with numba absent its import is Any and the ignore is
    # unnecessary; with numba installed the assignment needs it.
    _numba_jit = None  # type: ignore[assignment, unused-ignore]


class WeightingFilter:
    """
    Class-based frequency weighting filter (A, B, C, D, G, AU, Z).
    Allows pre-calculating and reusing filter coefficients.
    """

    def __init__(self, fs: int, curve: str = "A",
                 stateful: bool = False, steady_ic: bool = False,
                 high_accuracy: bool | None = None) -> None:
        """
        Initialize the weighting filter.

        :param fs: Sample rate in Hz.
        :param curve: 'A', 'C' (IEC 61672-1), 'B' (ANSI S1.4-1983,
            historical: removed from the IEC sound-level-meter standards),
            'D' (withdrawn IEC 537 aircraft-noise weighting), 'G' (ISO 7196
            infrasound), 'AU' (IEC 61012, audible sound in the presence of
            ultrasound) or 'Z'.
        :param stateful: If True, the weighting filter is stateful. Useful for block processing.
        :param steady_ic: If True, calculate steady state initial conditions for filter.
        :param high_accuracy: If True, design and run the filter at an internal
            oversampled rate (target >= 144 kHz) so the response stays within
            IEC 61672-1 class 1 tolerances up to 16 kHz. At 48 kHz this
            oversamples x3, keeping the deviation from the analytic curve to
            about -0.44 dB @16k / -0.85 dB @20k. The plain bilinear design
            still holds class 1 at fs = 44.1/48 kHz (about -2.7 dB at
            12.5 kHz, inside the +2.0/-5.0 class 1 limits) but degrades to
            class 2 for fs <= 32 kHz. Defaults to True except in stateful
            mode (the internal FIR resampling is incompatible with block
            processing).
        """
        if fs <= 0:
            raise ValueError("Sample rate 'fs' must be positive.")
        if high_accuracy is None:
            high_accuracy = not stateful
        if high_accuracy and stateful:
            raise ValueError("high_accuracy is not compatible with stateful processing.")

        self.fs = fs
        self.curve = curve.upper()
        self.stateful = stateful
        self.high_accuracy = high_accuracy
        # Oversample target 144 kHz: fs=48k -> x3, fs=44.1k -> x4, fs=96k -> x2,
        # fs=128k -> x2, fs>=144k -> x1.
        # A 96 kHz target left the common 48 kHz rate at only x2 (-1.1 dB @16k /
        # -2.1 dB @20k vs analytic); 144 kHz halves that residual (audit N1 A6).
        self._oversample = min(8, max(1, math.ceil(144000 / fs))) if high_accuracy else 1

        if self.curve == "Z":
            self.sos = np.array([])
            if self.stateful:
                self.zi = np.array([])
            return

        if self.curve not in ["A", "B", "C", "D", "G", "AU"]:
            raise ValueError(
                "Weighting curve must be 'A', 'B', 'C', 'D', 'G', 'AU' or 'Z'"
            )

        z, p, k = self._analog_design()

        design_fs = self.fs * self._oversample
        zd, pd, kd = signal.bilinear_zpk(z, p, k, design_fs)
        self.sos = signal.zpk2sos(zd, pd, kd)

        # Initialize filter state for stateful block-wise processing.
        # Uses lazy allocation: zi is sized on first filter() call so that
        # the channel dimension matches the actual input shape.
        if self.stateful:
            self.zi = np.array([])
            self._steady_ic = steady_ic

    def _analog_design(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Analog ZPK of the selected curve, normalised at its reference.

        Also adjusts ``self._oversample`` for the curves whose action extends
        beyond the default design target (G toward low sample rates, AU's U
        roll-off toward 40 kHz).
        """
        # Analog ZPK for the A and C weightings.
        # f1, f2, f3, f4 constants as per IEC 61672-1. ANSI S1.4-1983
        # Appendix C prints the same poles to fewer digits (f1 = 20.598997,
        # f4 = 12194.22), so the B weighting below shares them.
        f1 = 20.598997
        f4 = 12194.217

        if self.curve == "G":
            # ISO 7196:1995 Table 1 (p. 2): nominal pole/zero coordinates in
            # the complex frequency plane, in Hz. Four zeros at the origin
            # and four complex-conjugate pole pairs. The curve is defined
            # with 0 dB gain at 10 Hz (clause 4).
            z = np.zeros(4)
            pole_coords_hz = np.array(
                [
                    -0.707 + 0.707j,
                    -0.707 - 0.707j,
                    -19.27 + 5.16j,
                    -19.27 - 5.16j,
                    -14.11 + 14.11j,
                    -14.11 - 14.11j,
                    -5.16 + 19.27j,
                    -5.16 - 19.27j,
                ]
            )
            p = 2 * np.pi * pole_coords_hz
            # Normalize to 0 dB at 10 Hz.
            w = 2 * np.pi * 10.0
            k = 1.0 / np.abs(np.prod(1j * w - z) / np.prod(1j * w - p))
            # G acts on 0.25 Hz - 315 Hz, far below Nyquist at audio rates:
            # the bilinear warping (no prewarping) is negligible there
            # (~0.014% at 315 Hz for fs = 48 kHz, under 0.01 dB), so the
            # high-accuracy oversampling used for A/C (whose action extends
            # to 16 kHz) is unnecessary. At the low sample rates common for
            # infrasound recordings, however, 315 Hz approaches Nyquist and
            # the warping grows quadratically; oversample the design toward
            # 48 kHz so the response stays within ~0.05 dB regardless of fs.
            self._oversample = min(8, max(1, math.ceil(48000 / self.fs)))
        elif self.curve in ("A", "AU"):
            f2 = 107.65265
            f3 = 737.86223
            # Zeros at 0 Hz
            z = np.array([0, 0, 0, 0])
            # Poles
            p = np.array(
                [
                    -2 * np.pi * f1,
                    -2 * np.pi * f1,
                    -2 * np.pi * f4,
                    -2 * np.pi * f4,
                    -2 * np.pi * f2,
                    -2 * np.pi * f3,
                ]
            )
            # k chosen to give 0 dB at 1000 Hz
            k = 3.5174303309e13
            if self.curve == "AU":
                # IEC 61012:1990 subclause 2.2: the AU weighting is the A
                # weighting cascaded with the U low-pass filter, whose six
                # poles are prescribed in Table 2 (in Hz, no zeros):
                # a double real pole at -12 200 and complex-conjugate pairs
                # at -7 850 +/- j8 800 and -2 900 +/- j12 150. The gain is
                # renormalized to 0 dB at the 1 kHz reference frequency
                # below (Table 1 note: zero tolerance at the reference
                # frequency of IEC 651 subclause 3.7).
                p_u = 2 * np.pi * np.array(
                    [
                        -12200.0,
                        -12200.0,
                        -7850.0 + 8800.0j,
                        -7850.0 - 8800.0j,
                        -2900.0 + 12150.0j,
                        -2900.0 - 12150.0j,
                    ]
                )
                p = np.concatenate([p.astype(complex), p_u])
                if self.high_accuracy:
                    # The U poles act up to 40 kHz, twice as high as the A/C
                    # action (16-20 kHz) the 144 kHz design target was sized
                    # for, so double the target to keep the same relative
                    # bilinear-warping accuracy over the U roll-off. At
                    # fs = 48 kHz this designs at 288 kHz and keeps the
                    # 16 kHz deviation within about -0.7 dB of the IEC 61012
                    # Table 1 nominal (+/-3 dB tolerance there).
                    self._oversample = min(8, max(1, math.ceil(288000 / self.fs)))

        elif self.curve == "B":
            # ANSI S1.4-1983 Appendix C (C2): the B weighting is the C
            # weighting times f^2 / (f^2 + f5^2) in power, i.e. one more
            # zero at the origin and one extra real pole at f5. Historical:
            # B was dropped when IEC 61672-1 replaced the older meter
            # standards; keep it for legacy data only.
            f5 = 158.48932
            z = np.array([0, 0, 0])
            p = np.array(
                [
                    -2 * np.pi * f1,
                    -2 * np.pi * f1,
                    -2 * np.pi * f4,
                    -2 * np.pi * f4,
                    -2 * np.pi * f5,
                ]
            )
            k = 1.0

        elif self.curve == "D":
            # Withdrawn IEC 537:1976 aircraft-noise weighting, from the
            # published rational transfer function (module docstring):
            # zeros at the origin and at the roots of s^2 + 6532 s
            # + 4.0975e7; poles at -1776.3, -7288.5 and the roots of
            # s^2 + 21514 s + 3.8836e8 (all in rad/s). The complex pairs
            # are expanded with the quadratic formula so the design is
            # exact and deterministic. Corroborated against SQAT and
            # librosa (independent implementations).
            zr = -6532.0 / 2.0
            zi = math.sqrt(4.0975e7 - zr**2)
            pr = -21514.0 / 2.0
            pi = math.sqrt(3.8836e8 - pr**2)
            z = np.array([0.0, complex(zr, zi), complex(zr, -zi)])
            p = np.array(
                [-1776.3, -7288.5, complex(pr, pi), complex(pr, -pi)]
            )
            k = 1.0

        else:  # C weighting
            z = np.array([0, 0])
            p = np.array([-2 * np.pi * f1, -2 * np.pi * f1, -2 * np.pi * f4, -2 * np.pi * f4])
            k = 5.91797e8

        if self.curve != "G":
            # Recalculate k to ensure 0 dB at 1 kHz, the reference frequency
            # shared by every audio-band weighting (IEC 61672-1 for A/C,
            # ANSI S1.4-1983 for B, IEC 537 for D, IEC 61012 for AU).
            w = 2 * np.pi * 1000
            h = k * np.prod(1j * w - z) / np.prod(1j * w - p)
            k = k / np.abs(h)

        return z, p, k

    def _init_filter_state(self, x_proc: np.ndarray) -> None:
        """Allocate or reallocate ``zi`` to match the input shape."""
        self.zi = _sos_initial_state(self.sos, x_proc, self._steady_ic)

    def _needs_zi_reinit(self, x_proc: np.ndarray) -> bool:
        """Check whether ``zi`` must be (re)allocated for *x_proc*."""
        return _sos_state_mismatch(self.zi, x_proc)

    def filter(self, x: list[float] | np.ndarray) -> np.ndarray:
        """
        Apply the weighting filter to a signal.

        :param x: Input signal (1D or 2D [channels, samples]).
        :return: Weighted signal.
        """
        x_proc = _typesignal(x)
        if self.curve == "Z":
            return x_proc

        if self.stateful:
            if self._needs_zi_reinit(x_proc):
                self._init_filter_state(x_proc)
            y, self.zi = signal.sosfilt(self.sos, x_proc, axis=-1, zi=self.zi)
        elif self._oversample > 1:
            if x_proc.shape[-1] == 0:
                return x_proc  # resample_poly rejects empty input
            up = signal.resample_poly(x_proc, self._oversample, 1, axis=-1)
            y_up = signal.sosfilt(self.sos, up, axis=-1)
            y = signal.resample_poly(y_up, 1, self._oversample, axis=-1)
        else:
            y = signal.sosfilt(self.sos, x_proc, axis=-1)

        return cast(np.ndarray, y)


@lru_cache(maxsize=32)
def _cached_weighting_filter(
    fs: int, curve: str, high_accuracy: bool
) -> WeightingFilter:
    """Reuse the (immutable, non-stateful) weighting-filter design.

    A non-stateful ``WeightingFilter`` never mutates its SOS in ``filter()``,
    so the design (bilinear + zpk2sos, ~0.9 ms) can be cached and shared across
    repeated ``weighting_filter()`` calls at the same rate/curve. The
    high-accuracy filtering cost itself (oversample -> sosfilt -> decimate) is
    inherent to IEC 61672-1 class 1 accuracy and is not cached.
    """
    return WeightingFilter(fs, curve, high_accuracy=high_accuracy)


def weighting_filter(
    x: list[float] | np.ndarray, fs: int, curve: str = "A", high_accuracy: bool = True
) -> np.ndarray:
    """
    Apply a frequency weighting to a signal.

    :param x: Input signal.
    :param fs: Sample rate.
    :param curve: 'A', 'C' (IEC 61672-1), 'B' (ANSI S1.4-1983, historical),
        'D' (withdrawn IEC 537 aircraft-noise weighting), 'G' (ISO 7196
        infrasound), 'AU' (IEC 61012) or 'Z' (bypass).
    :param high_accuracy: Use internal oversampling for IEC 61672-1 class 1
        accuracy at high frequencies (default True).
    :return: Weighted signal.
    """
    wf = _cached_weighting_filter(fs, curve, high_accuracy)
    return wf.filter(x)


def _prepare_time_weighting_initial_state(
    x_sq: np.ndarray,
    initial_state: str | float | np.ndarray | None,
) -> np.ndarray:
    """Return the previous output state ``y[-1]`` for time weighting."""
    invalid_initial_state_message = "initial_state must be None, 'zero', 'first', a scalar, or an array"
    state_shape = x_sq.shape[:-1]

    if initial_state is None:
        return np.zeros(state_shape, dtype=x_sq.dtype)

    if isinstance(initial_state, str):
        state_name = initial_state.lower()
        if state_name == "zero":
            return np.zeros(state_shape, dtype=x_sq.dtype)
        if state_name == "first":
            if x_sq.shape[-1] == 0:
                raise ValueError(invalid_initial_state_message)
            return np.asarray(np.take(x_sq, 0, axis=-1), dtype=x_sq.dtype).copy()
        raise ValueError(invalid_initial_state_message)

    state = np.asarray(initial_state, dtype=x_sq.dtype)
    if state.shape == ():
        return np.full(state_shape, state.item(), dtype=x_sq.dtype)

    try:
        return np.broadcast_to(state, state_shape).astype(x_sq.dtype, copy=True)
    except ValueError as exc:
        raise ValueError(
            "initial_state must be scalar or broadcastable to the input shape without the time axis"
        ) from exc


def _impulse_kernel_py(
    x_t: np.ndarray,
    alpha_rise: float,
    alpha_fall: float,
    initial_state: np.ndarray,
) -> np.ndarray:
    """Asymmetric time-weighting kernel (pure Python; jitted when numba is present)."""
    y_t = np.zeros_like(x_t)
    curr_y = initial_state.copy()

    for i in range(x_t.shape[0]):
        val = x_t[i]
        rising = val > curr_y

        diff = val - curr_y
        factor = np.where(rising, alpha_rise, alpha_fall)
        curr_y += factor * diff
        y_t[i] = curr_y

    return y_t


if _numba_jit is not None:
    _apply_impulse_kernel = _numba_jit(nopython=True, cache=True)(_impulse_kernel_py)
else:  # pragma: no cover - exercised only without numba installed
    _apply_impulse_kernel = _impulse_kernel_py

def time_weighting(
    x: list[float] | np.ndarray,
    fs: int,
    mode: str = "fast",
    initial_state: str | float | np.ndarray | None = None,
) -> np.ndarray:
    """
    Apply time weighting to a signal (Exponential averaging).
    
    :param x: Input signal (raw pressure/voltage). The function squares it internally.
    :param fs: Sample rate.
    :param mode: 'fast' (125ms), 'slow' (1000ms), 'impulse' (35ms rise, 1500ms fall).
    :param initial_state: Previous mean-square output state ``y[-1]``. Use None/'zero' for
        zero initialization (default), 'first' to initialize from the first input energy,
        or a scalar/array broadcastable to the input shape without the time axis.
    :return: Time-weighted squared signal (sound pressure level envelope).
    """
    x_proc = _typesignal(x)
    if fs <= 0:
        raise ValueError("Sample rate 'fs' must be positive.")
    x_sq = x_proc**2
    initial = _prepare_time_weighting_initial_state(x_sq, initial_state)
    
    mode_lower = mode.lower()
    
    if mode_lower in ["fast", "slow"]:
        tau = 0.125 if mode_lower == "fast" else 1.0
        alpha = 1 - np.exp(-1 / (fs * tau))
        b = [alpha]
        a = [1, -(1 - alpha)]
        # We apply the weighting to the squared signal to get the Mean Square value
        zi = np.expand_dims((1 - alpha) * initial, axis=-1)
        y, _ = signal.lfilter(b, a, x_sq, axis=-1, zi=zi)
        return cast(np.ndarray, y)
        
    elif mode_lower == "impulse":
        # IEC 61672-1: 35ms for rising, 1500ms for falling
        tau_rise = 0.035
        tau_fall = 1.5
        
        alpha_rise = 1 - np.exp(-1 / (fs * tau_rise))
        alpha_fall = 1 - np.exp(-1 / (fs * tau_fall))
        
        # Move time axis to front for iteration
        x_t = np.moveaxis(x_sq, -1, 0)
        
        # Ensure contiguous array for Numba
        x_t = np.ascontiguousarray(x_t)
        initial_kernel = initial if initial.ndim == 0 else np.ascontiguousarray(initial)
        y_t = _apply_impulse_kernel(x_t, alpha_rise, alpha_fall, initial_kernel)
            
        # Move time axis back
        return np.moveaxis(y_t, 0, -1)

    else:
        raise ValueError("Invalid time weighting mode. Use ['fast', 'slow', 'impulse']")


class TimeWeighting:
    """
    Stateful time weighting for block processing.

    Wraps :func:`time_weighting` carrying the exponential integrator state
    across blocks, so concatenated block outputs equal a single continuous call.
    """

    def __init__(self, fs: int, mode: str = "fast") -> None:
        """
        :param fs: Sample rate in Hz.
        :param mode: 'fast' (125 ms), 'slow' (1000 ms) or 'impulse' (35 ms / 1.5 s).
        """
        if fs <= 0:
            raise ValueError("Sample rate 'fs' must be positive.")
        if mode.lower() not in ("fast", "slow", "impulse"):
            raise ValueError("Invalid time weighting mode. Use ['fast', 'slow', 'impulse']")
        self.fs = fs
        self.mode = mode.lower()
        self._state: np.ndarray | None = None

    def process(self, x: list[float] | np.ndarray) -> np.ndarray:
        """Apply time weighting to a block, continuing from the previous block."""
        x_proc = _typesignal(x)
        if x_proc.shape[-1] == 0:
            return x_proc  # nothing to process; keep the carried state
        env = time_weighting(x_proc, self.fs, mode=self.mode, initial_state=self._state)
        self._state = np.asarray(env[..., -1]).copy()
        return env

    def reset(self) -> None:
        """Forget the carried state (the next block starts from rest)."""
        self._state = None


def linkwitz_riley(
    x: list[float] | np.ndarray, 
    fs: int, 
    freq: float, 
    order: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """
    Linkwitz-Riley crossover filter (Butterworth squared).
    Splits signal into low and high bands with flat sum response.
    
    :param x: Input signal.
    :param fs: Sample rate.
    :param freq: Crossover frequency.
    :param order: Total order (must be even, typically 2 or 4).
    :return: (low_pass_signal, high_pass_signal)
    """
    x_proc = _typesignal(x)
    if order % 2 != 0:
        raise ValueError("Linkwitz-Riley order must be even (typically 2 or 4).")
    
    # A Linkwitz-Riley filter of order N is two Butterworth filters of order N/2 in series
    half_order = order // 2
    wn = freq / (fs / 2)
    
    sos_lp = signal.butter(half_order, wn, btype='low', output='sos')
    sos_hp = signal.butter(half_order, wn, btype='high', output='sos')
    
    # Pass twice
    lp = signal.sosfilt(sos_lp, x_proc)
    lp = signal.sosfilt(sos_lp, lp)
    
    hp = signal.sosfilt(sos_hp, x_proc)
    hp = signal.sosfilt(sos_hp, hp)
    
    return lp, hp
