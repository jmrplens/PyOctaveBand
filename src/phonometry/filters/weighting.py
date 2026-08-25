#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Weighting filters (A, B, C, D, G, AU, Z), time weighting utilities and
the Linkwitz-Riley crossover.

A/C/Z per IEC 61672-1:2013; G (infrasound) per ISO 7196:1995.

B is the historical weighting of ANSI S1.4-1983 (Appendix C): the C curve
with one extra zero at the origin and one extra real pole at
:math:`f_5 = 158.48932` Hz. It was dropped from the sound-level-meter
standards
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

.. math::

   \frac{k s \left( s^2 + 6532 s + 4.0975 \times 10^7 \right)}
        {(s + 1776.3)(s + 7288.5)
         \left( s^2 + 21514 s + 3.8836 \times 10^8 \right)}

with ``k`` renormalized to exactly 0 dB at
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
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast, overload

import numpy as np
from scipy import signal

from .._internal.utils import _sos_initial_state, _sos_state_mismatch
from ..io._resolve import (
    like_input,
    refuse_foreign_rate,
    resolve_fs,
    resolve_samples,
)
from ..io._signal import Signal

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import DTypeLike

#: Rejection message shared by the three entry points that take ``fs``.
_FS_POSITIVE = "Sample rate 'fs' must be positive."

try:
    from numba import jit as _numba_jit
except ImportError:  # pragma: no cover - depends on install extras
    # unused-ignore: with numba absent its import is Any and the ignore is
    # unnecessary; with numba installed the assignment needs it.
    _numba_jit = None  # type: ignore[assignment, unused-ignore]


class WeightingFilter:
    """Class-based frequency weighting filter (A, B, C, D, G, AU, Z).
    Allows pre-calculating and reusing filter coefficients.
    """

    def __init__(
        self,
        fs: int,
        curve: str = "A",
        stateful: bool = False,
        steady_ic: bool = False,
        high_accuracy: bool | None = None,
    ) -> None:
        """Initialize the weighting filter.

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
            IEC 61672-1 class 1 tolerances up to 16 kHz, provided 16 kHz is
            well clear of the input Nyquist frequency (fs >= 40 kHz). At
            48 kHz this oversamples x3, keeping the deviation from the design
            goal to -0.44 dB at the 16 kHz nominal frequency and -0.86 dB at
            the 20 kHz one. Oversampling cannot rescue the top of the band at
            low sample rates, because the resampling stages it adds around
            the sections carry an anti-alias transition band centred on the
            input Nyquist frequency: above roughly 0.9 x fs/2 the response
            rolls off steeply whatever the design rate. What the roll-off
            costs is per curve, since each is graded against its own design
            goal: at fs = 32 kHz the 15 848.9 Hz nominal point falls 16.2 dB
            below the A goal but 15.3 dB below the C one (class 1 allows
            -16.0 dB there, class 2 has no lower limit), so the verified
            class at 32 kHz is 2 for A and still 1 for C. At fs = 16 kHz the
            7 943.3 Hz point falls 12.0 dB below the A goal and 13.7 dB below
            the C one (class 1 allows -2.5 dB, class 2 -5.0 dB), so neither
            curve verifies to any class there. The plain bilinear
            design holds class 1 for fs >= 40 kHz (-2.8 dB at the 12.5 kHz
            nominal frequency at 48 kHz, -3.5 dB at 44.1 kHz, inside the
            +2.0/-5.0 class 1 limits), degrades to class 2 between 22.05 and
            32 kHz and meets no class at fs <= 20 kHz. Defaults to True
            except in stateful mode (the internal FIR resampling is
            incompatible with block processing).
        """
        if fs <= 0:
            raise ValueError(_FS_POSITIVE)
        if high_accuracy is None:
            high_accuracy = not stateful
        if high_accuracy and stateful:
            msg = "high_accuracy is not compatible with stateful processing."
            raise ValueError(msg)

        self.fs = fs
        self.curve = curve.upper()
        self.stateful = stateful
        self.high_accuracy = high_accuracy
        # Oversample target 144 kHz: fs=48k -> x3, fs=44.1k -> x4, fs=96k -> x2,
        # fs=128k -> x2, fs>=144k -> x1.
        # A 96 kHz target left the common 48 kHz rate at only x2 (-1.1 dB @16k /
        # -2.1 dB @20k vs analytic); 144 kHz halves that residual (audit N1 A6).
        self._oversample = (
            min(8, max(1, math.ceil(144000 / fs))) if high_accuracy else 1
        )

        if self.curve == "Z":
            self.sos = np.array([])
            if self.stateful:
                self.zi = np.array([])
            return

        if self.curve not in ["A", "B", "C", "D", "G", "AU"]:
            msg = "Weighting curve must be 'A', 'B', 'C', 'D', 'G', 'AU' or 'Z'"
            raise ValueError(msg)

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
            # Guarded like AU's target doubling below: only the high-accuracy
            # path resamples around the filter, so only it may design above
            # the input rate. Stateful mode filters block by block at self.fs,
            # and an SOS designed for a higher rate applied at the input rate
            # read -36 dB at G's own 10 Hz reference at fs = 2000. Stateful
            # therefore runs the plain design at the input rate, which is
            # what its documentation always said it did.
            if self.high_accuracy:
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
                p_u = (
                    2
                    * np.pi
                    * np.array(
                        [
                            -12200.0,
                            -12200.0,
                            -7850.0 + 8800.0j,
                            -7850.0 - 8800.0j,
                            -2900.0 + 12150.0j,
                            -2900.0 - 12150.0j,
                        ]
                    )
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
            p = np.array([-1776.3, -7288.5, complex(pr, pi), complex(pr, -pi)])
            k = 1.0

        else:  # C weighting
            z = np.array([0, 0])
            p = np.array(
                [-2 * np.pi * f1, -2 * np.pi * f1, -2 * np.pi * f4, -2 * np.pi * f4]
            )
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

    def filter(self, x: Signal | list[float] | np.ndarray) -> Signal | np.ndarray:
        """Apply the weighting filter to a signal.

        :param x: Input signal (1D or 2D [channels, samples]), or a
            :class:`phonometry.io.Signal`. A Signal recorded at another rate
            than this filter was designed for is refused rather than
            weighted by the wrong response; a calibrated one is weighted in
            pascals, exactly as :func:`weighting_filter` does, so the two
            entry points cannot disagree about the same recording.
        :return: The weighted record. A bare array in gives a bare array
            back; a :class:`~phonometry.io.Signal` gives a Signal, on the
            same terms as :func:`weighting_filter`, so the object and the
            function cannot disagree about the same recording.
        :raises ValueError: If a Signal's rate is not this filter's.
        """
        refuse_foreign_rate(x, self.fs, "weighting filter")
        x_proc = resolve_samples(x)
        if self.curve == "Z":
            return like_input(x, x_proc)

        if self.stateful:
            if self._needs_zi_reinit(x_proc):
                self._init_filter_state(x_proc)
            y, self.zi = signal.sosfilt(self.sos, x_proc, axis=-1, zi=self.zi)
        elif self._oversample > 1:
            if x_proc.shape[-1] == 0:
                return like_input(x, x_proc)  # resample_poly rejects empty input
            up = signal.resample_poly(x_proc, self._oversample, 1, axis=-1)
            y_up = signal.sosfilt(self.sos, up, axis=-1)
            y = signal.resample_poly(y_up, 1, self._oversample, axis=-1)
        else:
            y = signal.sosfilt(self.sos, x_proc, axis=-1)

        return like_input(x, cast(np.ndarray, y))


def _resample_poly_fir(rate: int) -> np.ndarray:
    """Anti-alias FIR that ``resample_poly`` designs for a 1:*rate* ratio.

    ``scipy.signal.resample_poly`` documents (and implements) its default
    filter as ``firwin(2 * half_len + 1, f_c, window=('kaiser', 5.0))`` with
    ``half_len = 10 * max(up, down)`` and ``f_c = 1 / max(up, down)``
    relative to Nyquist, then scales it by ``up``. Both stages of the
    high-accuracy path use ``max(up, down) = rate``, so they share these taps;
    the ``up`` scaling of the interpolation stage exactly offsets the energy
    lost to zero-stuffing and is left out here, giving unit passband gain.
    The cutoff lands on the *input* Nyquist frequency, so the transition band
    straddles ``fs / 2``.
    """
    return np.asarray(
        signal.firwin(2 * (10 * rate) + 1, 1.0 / rate, window=("kaiser", 5.0)),
        dtype=np.float64,
    )


def _runtime_frequency_response(
    wf: WeightingFilter, frequencies: np.ndarray
) -> np.ndarray:
    r"""Complex steady-state response of the *whole* filter path.

    A non-stateful ``WeightingFilter`` with ``high_accuracy`` does not apply
    its second-order sections to the input directly: it interpolates by
    ``L = _oversample``, filters at ``L * fs`` and decimates back, so the
    signal passes the ``resample_poly`` anti-alias FIR twice as well. That
    cascade is linear and time-invariant at the input rate (interpolating by
    L, filtering, then decimating by L convolves the input with
    ``h[nL]``), and its response is the sum of the L spectral images that
    the decimation folds onto each output frequency,

    .. math::

       G(f) = \sum_{k=0}^{L-1} H_\mathrm{FIR}^2(f - k f_s)\,
              H_\mathrm{SOS}(f - k f_s),

    evaluated at the ``L * fs`` design rate (the transfer functions are
    periodic there, so images beyond the design Nyquist frequency need no
    wrapping, and Hermitian symmetry gives the negative ones). Only the
    ``k = 0`` term matters until the image at ``fs - f`` enters the
    anti-alias transition band, which is exactly what happens as *f*
    approaches ``fs / 2``.

    :param wf: The weighting filter whose path is measured.
    :param frequencies: Frequencies in Hz, below the input Nyquist frequency.
    :return: Complex voltage response at each frequency (not normalized).
    """
    f = np.asarray(frequencies, dtype=np.float64)
    if wf.curve == "Z" or wf.sos.size == 0:
        return np.ones_like(f, dtype=np.complex128)

    fs_proc = wf.fs * wf._oversample
    if wf._oversample == 1:
        # Nothing is resampled, so the sections are the whole path. This also
        # covers stateful mode, which is rejected at construction time
        # together with ``high_accuracy`` and therefore always has an
        # oversample factor of 1.
        _, h = signal.sosfreqz(wf.sos, worN=f, fs=fs_proc)
        return np.asarray(h, dtype=np.complex128)

    images = f[None, :] - np.arange(wf._oversample)[:, None] * float(wf.fs)
    flat = np.abs(images).ravel()
    _, h_sos = signal.sosfreqz(wf.sos, worN=flat, fs=fs_proc)
    _, h_fir = signal.freqz(_resample_poly_fir(wf._oversample), worN=flat, fs=fs_proc)
    terms = (h_sos * h_fir**2).reshape(images.shape)
    terms = np.where(images < 0.0, np.conj(terms), terms)
    return np.asarray(terms.sum(axis=0), dtype=np.complex128)


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


def _as_envelope(
    x: Signal | list[float] | np.ndarray,
    mean_square: np.ndarray,
    fs: int,
    mode: str,
) -> TimeWeightedEnvelope | np.ndarray:
    """Wrap the envelope for a Signal input, leave a bare array alone.

    Same conditional shape as the rest of the contract: a caller that passed
    an array gets the array back, so nothing that works today changes, and
    the six places that divide the envelope in place keep working.
    """
    if not isinstance(x, Signal):
        return mean_square
    return TimeWeightedEnvelope(
        mean_square=mean_square,
        fs=fs,
        mode=mode,
        calibrated=x.calibration_factor is not None,
    )


@dataclass(frozen=True)
class TimeWeightedEnvelope:
    """The exponentially averaged mean square of a record, and its rate.

    What :func:`time_weighting` computes is not a waveform: it is the
    running mean SQUARE, in pascals squared when the record was calibrated,
    which is why it cannot come back as a
    :class:`~phonometry.io.Signal`. That class means a record of pressure,
    and labelling a squared quantity as one would be the kind of quiet lie
    the calibration contract exists to prevent.

    What it needs instead is the rate, so the envelope can be read against a
    time axis, and a plot that knows the trace is a level. That is this
    object. It stands in for the bare array it replaced everywhere the array
    was used: :func:`numpy.asarray`, ``len()``, indexing and the
    ``shape``/``ndim``/``size``/``dtype`` attributes all forward to the
    envelope, so a caller that only wanted the numbers never notices.

    :ivar mean_square: The weighted mean square, ``(channels, samples)`` or
        1-D for one channel, in Pa2 when the record was calibrated.
    :ivar fs: Sample rate, in Hz.
    :ivar mode: The weighting used: ``"fast"``, ``"slow"`` or ``"impulse"``.
    :ivar calibrated: Whether the samples that produced it were in pascals,
        which is what decides whether a level read off it means dB SPL.
    """

    mean_square: np.ndarray
    fs: int
    mode: str
    calibrated: bool

    def __array__(self, dtype: DTypeLike | None = None) -> np.ndarray:
        """Return the envelope as an array (optionally recast)."""
        return np.asarray(self.mean_square, dtype=dtype)

    def __len__(self) -> int:
        """Length of the leading axis: channels when 2-D, samples for one channel."""
        return int(self.mean_square.shape[0])

    def __getitem__(self, key: Any) -> Any:  # noqa: ANN401  # mirrors ndarray indexing, whose key union numpy does not export
        """Index the mean square; the result is bare values, not another envelope."""
        return self.mean_square[key]

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the envelope."""
        return tuple(self.mean_square.shape)

    @property
    def ndim(self) -> int:
        """Number of dimensions of the envelope."""
        return int(self.mean_square.ndim)

    @property
    def size(self) -> int:
        """Number of values in the envelope."""
        return int(self.mean_square.size)

    @property
    def dtype(self) -> np.dtype[Any]:
        """Data type of the envelope."""
        return self.mean_square.dtype

    @property
    def times(self) -> np.ndarray:
        """Sample times, in seconds from the start of the record."""
        return np.arange(self.mean_square.shape[-1]) / float(self.fs)

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the level trace this envelope stands for.

        Draws ``10 lg(mean square / p0^2)`` against time. That is a
        time-weighted sound pressure level, and it is ``L_pAF`` only when
        the record was A-weighted before it got here: this function applies
        the time weighting and nothing else. Needs a calibrated record to
        mean dB SPL, and says so rather than drawing a number counted from
        nothing.
        """
        from .._i18n import check_language
        from .._plot.filters import plot_time_weighted_envelope

        check_language(language)
        return plot_time_weighted_envelope(self, ax=ax, language=language, **kwargs)


@overload
def weighting_filter(
    x: Signal,
    fs: int | None = ...,
    curve: str = ...,
    high_accuracy: bool = ...,
) -> Signal: ...


@overload
def weighting_filter(
    x: list[float] | np.ndarray,
    fs: int,
    curve: str = ...,
    high_accuracy: bool = ...,
) -> np.ndarray: ...


def weighting_filter(
    x: Signal | list[float] | np.ndarray,
    fs: int | None = None,
    curve: str = "A",
    high_accuracy: bool = True,
) -> Signal | np.ndarray:
    """Apply a frequency weighting to a signal.

    :param x: Input signal, or a :class:`phonometry.io.Signal` read from a
        measurement file. A calibrated Signal is weighted in pascals, so the
        weighted samples come back in pascals too; a bare array keeps
        whatever unit it arrived in.
    :param fs: Sample rate. Required for a bare array; a
        :class:`~phonometry.io.Signal` brings its own, and an explicit value
        that disagrees with it raises instead of silently winning.
    :param curve: 'A', 'C' (IEC 61672-1), 'B' (ANSI S1.4-1983, historical),
        'D' (withdrawn IEC 537 aircraft-noise weighting), 'G' (ISO 7196
        infrasound), 'AU' (IEC 61012) or 'Z' (bypass).
    :param high_accuracy: Use internal oversampling for IEC 61672-1 class 1
        accuracy at high frequencies (default True).
    :return: The weighted record. A bare array in gives a bare array back; a
        :class:`~phonometry.io.Signal` gives a Signal, whose samples are
        already in pascals and whose factor therefore reads 1.0.
    """
    fs = resolve_fs(x, fs)
    wf = _cached_weighting_filter(fs, curve, high_accuracy)
    # The object form wraps for itself now, so hand it the caller's input
    # rather than the resolved samples: doing both would wrap twice.
    return wf.filter(x)


def _prepare_time_weighting_initial_state(
    x_sq: np.ndarray,
    initial_state: str | float | np.ndarray | None,
) -> np.ndarray:
    """Return the previous output state ``y[-1]`` for time weighting."""
    invalid_initial_state_message = (
        "initial_state must be None, 'zero', 'first', a scalar, or an array"
    )
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
        msg = "initial_state must be scalar or broadcastable to the input shape without the time axis"
        raise ValueError(msg) from exc


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


@overload
def time_weighting(
    x: Signal,
    fs: int | None = ...,
    mode: str = ...,
    initial_state: str | float | np.ndarray | None = ...,
) -> TimeWeightedEnvelope: ...


@overload
def time_weighting(
    x: list[float] | np.ndarray,
    fs: int,
    mode: str = ...,
    initial_state: str | float | np.ndarray | None = ...,
) -> np.ndarray: ...


def time_weighting(
    x: Signal | list[float] | np.ndarray,
    fs: int | None = None,
    mode: str = "fast",
    initial_state: str | float | np.ndarray | None = None,
) -> TimeWeightedEnvelope | np.ndarray:
    """Apply time weighting to a signal (Exponential averaging).

    :param x: Input signal (raw pressure/voltage), or a
        :class:`phonometry.io.Signal` read from a measurement file. The
        function squares it internally, so a calibrated Signal yields a
        mean-square envelope in Pa2 rather than in digital units squared.
    :param fs: Sample rate. Required for a bare array; a
        :class:`~phonometry.io.Signal` brings its own, and an explicit value
        that disagrees with it raises.
    :param mode: 'fast' (125ms), 'slow' (1000ms), 'impulse' (35ms rise, 1500ms fall).
    :param initial_state: Previous mean-square output state ``y[-1]``. Use None/'zero' for
        zero initialization (default), 'first' to initialize from the first input energy,
        or a scalar/array broadcastable to the input shape without the time axis.
    :return: The time-weighted mean square. A bare array in gives a bare
        array back; a :class:`~phonometry.io.Signal` gives a
        :class:`TimeWeightedEnvelope`, which stands in for that array
        everywhere it was used and adds the rate and a level plot. It is not
        a Signal, because a mean square is not a pressure record.
    """
    fs = resolve_fs(x, fs)
    x_proc = resolve_samples(x)
    if fs <= 0:
        raise ValueError(_FS_POSITIVE)
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
        return _as_envelope(x, cast(np.ndarray, y), fs, mode_lower)

    if mode_lower == "impulse":
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
        return _as_envelope(x, np.moveaxis(y_t, 0, -1), fs, mode_lower)

    msg = "Invalid time weighting mode. Use ['fast', 'slow', 'impulse']"
    raise ValueError(msg)


class TimeWeighting:
    """Stateful time weighting for block processing.

    Wraps :func:`time_weighting` carrying the exponential integrator state
    across blocks, so concatenated block outputs equal a single continuous call.
    """

    def __init__(self, fs: int, mode: str = "fast") -> None:
        """:param fs: Sample rate in Hz.
        :param mode: 'fast' (125 ms), 'slow' (1000 ms) or 'impulse' (35 ms / 1.5 s).
        """
        if fs <= 0:
            raise ValueError(_FS_POSITIVE)
        if mode.lower() not in ("fast", "slow", "impulse"):
            msg = "Invalid time weighting mode. Use ['fast', 'slow', 'impulse']"
            raise ValueError(msg)
        self.fs = fs
        self.mode = mode.lower()
        self._state: np.ndarray | None = None

    def process(
        self, x: Signal | list[float] | np.ndarray
    ) -> TimeWeightedEnvelope | np.ndarray:
        """Apply time weighting to a block, continuing from the previous block.

        The block form of :func:`time_weighting`, and it returns the same
        thing on the same terms: a mean square, in pascals squared when the
        record was calibrated, wrapped in a
        :class:`TimeWeightedEnvelope` when the block arrived as a
        :class:`~phonometry.io.Signal` and left as a bare array otherwise.
        The envelope stands in for that array, so a loop that concatenates
        the blocks keeps working either way.

        :param x: The block, or a :class:`phonometry.io.Signal`. A Signal at
            another rate than this integrator was built for is refused; a
            calibrated one is squared in pascals, exactly as
            :func:`time_weighting` does.
        :return: Time-weighted mean-square envelope of the block.
        :raises ValueError: If a Signal's rate is not this integrator's.
        """
        refuse_foreign_rate(x, self.fs, "time weighting")
        x_proc = resolve_samples(x)
        if x_proc.shape[-1] == 0:
            # Nothing to process; keep the carried state. The empty block is
            # handed back in the shape it arrived in, like any other.
            return _as_envelope(x, x_proc, self.fs, self.mode)
        env = time_weighting(x_proc, self.fs, mode=self.mode, initial_state=self._state)
        self._state = np.asarray(env[..., -1]).copy()
        return _as_envelope(x, np.asarray(env), self.fs, self.mode)

    def reset(self) -> None:
        """Forget the carried state (the next block starts from rest)."""
        self._state = None


@overload
def linkwitz_riley(
    x: Signal, fs: int | None = ..., *, freq: float, order: int = ...
) -> tuple[Signal, Signal]: ...


@overload
def linkwitz_riley(
    x: list[float] | np.ndarray, fs: int, *, freq: float, order: int = ...
) -> tuple[np.ndarray, np.ndarray]: ...


def linkwitz_riley(
    x: Signal | list[float] | np.ndarray,
    fs: int | None = None,
    *,
    freq: float,
    order: int = 4,
) -> tuple[Signal, Signal] | tuple[np.ndarray, np.ndarray]:
    """Linkwitz-Riley crossover filter (Butterworth squared).
    Splits signal into low and high bands with flat sum response.

    :param x: Input signal, or a :class:`phonometry.io.Signal` read from a
        measurement file. A calibrated Signal is split in pascals, so both
        bands come back in pascals.
    :param fs: Sample rate. Required for a bare array; a
        :class:`~phonometry.io.Signal` brings its own, and an explicit value
        that disagrees with it raises.
    :param freq: Crossover frequency, in Hz. Keyword-only and required: it
        sits behind an optional ``fs``, and a default here would be a
        signature that lies about what the call needs.
    :param order: Total order (must be even, typically 2 or 4).
    :return: (low_pass_signal, high_pass_signal)
    """
    fs = resolve_fs(x, fs)
    x_proc = resolve_samples(x)
    if order % 2 != 0:
        msg = "Linkwitz-Riley order must be even (typically 2 or 4)."
        raise ValueError(msg)

    # A Linkwitz-Riley filter of order N is two Butterworth filters of order N/2 in series
    half_order = order // 2
    wn = freq / (fs / 2)

    sos_lp = signal.butter(half_order, wn, btype="low", output="sos")
    sos_hp = signal.butter(half_order, wn, btype="high", output="sos")

    # Pass twice
    lp = signal.sosfilt(sos_lp, x_proc)
    lp = signal.sosfilt(sos_lp, lp)

    hp = signal.sosfilt(sos_hp, x_proc)
    hp = signal.sosfilt(sos_hp, hp)

    # The two branches are one measurement split in two, so they are wrapped
    # together: a caller that got a Signal for the low band and an array for
    # the high one would have to test which it holds before using either.
    if isinstance(x, Signal):
        return like_input(x, lp), like_input(x, hp)
    return lp, hp
