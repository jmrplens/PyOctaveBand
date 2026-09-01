#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Weighting filters (A, B, C, D, G, AU, 468, Z), time weighting utilities
and the Linkwitz-Riley crossover.

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

468 per ITU-R BS.468-4, the psophometric weighting for audio-frequency
noise in sound broadcasting: a bandpass peaking at +12.22 dB near 6.3 kHz
and falling at about -30 dB/octave above 12.5 kHz, which is what makes
broadband noise audible in a programme chain rather than what makes it
loud. Clause 1 defines it as the response of the passive network of Fig. 1a,
so it is built from that network's seven printed component values
(:func:`_itu_r_468_prototype`) and reproduces all 21 rows of the Table 1
sampling to 0.0503 dB. Its skirt is steep enough that the plain bilinear
design at the input rate reads 23 dB out at 16 kHz, so ``high_accuracy=False``
is refused for this curve rather than shipped: the Recommendation prints one
mask and no lower grade to fall back to.

**How the prototypes become filters.** Every curve above is a set of poles and
zeros in the s plane, and the bilinear transform that turns those into a
digital filter is exact in magnitude but wrong in frequency: it puts the
prototype's response at ``2 f_s tan(pi f / f_s)`` instead of at ``2 pi f``,
which costs 0.86 dB at 20 kHz for A at 48 kHz and 61 dB at 15 848.9 Hz when
fs = 32 kHz. The library used to hide that by interpolating, filtering at
three to eight times the rate and decimating back. It no longer does: the
prototype is fitted at the sample rate instead
(:mod:`phonometry.filters._weighting_design`), and what runs is one cascade of
second-order sections. Three consequences worth stating in one place, because
each of them used to be a documented limitation of this module:

* the anti-alias filter of those resampling stages had its transition band on
  the input Nyquist frequency and the signal crossed it twice, which put a
  floor of about 1.7 dB on everything above 0.9 of Nyquist whatever the design
  rate. That floor is gone, so the rows near Nyquist -- the 15 848.9 Hz row at
  32 kHz, the 20 kHz row of BS.468-4 Table 1 at 44.1 kHz -- are inside their
  masks instead of over them, and A and C verify to class 1 at every sample
  rate from 8 kHz up;
* block processing was incompatible with the accurate design, because a
  resampler cannot be driven block by block. It no longer is: ``stateful`` and
  ``high_accuracy`` are independent, and stitched blocks reproduce a single
  call exactly; and
* one minute of 44.1 kHz audio costs about 18 ms instead of 377 ms for A and
  775 ms for 468, and holds 21 MB of intermediates instead of 169 MB
  (measured back to back in one process, mean of five runs). The design
  itself costs about 260 ms and is cached, so even a first call is quicker
  than the path it replaces: about 280 ms against 377 for A, 285 against
  775 for 468.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast, overload

import numpy as np
from scipy import signal

from .._internal.utils import _sos_initial_state, _sos_state_mismatch
from .._internal.validation import require_positive
from ..io._resolve import (
    like_input,
    refuse_foreign_rate,
    resolve_fs,
    resolve_samples,
)
from ..io._signal import Signal
from ._weighting_design import design_sos

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import DTypeLike

#: Rejection message shared by the four entry points that take ``fs``.
_FS_POSITIVE = "Sample rate 'fs' must be positive."

try:
    from numba import jit as _numba_jit
except ImportError:  # pragma: no cover - depends on install extras
    # unused-ignore: with numba absent its import is Any and the ignore is
    # unnecessary; with numba installed the assignment needs it.
    _numba_jit = None  # type: ignore[assignment, unused-ignore]


def _require_str(value: object, name: str) -> str:
    """Require *value* to be a string before any case folding touches it.

    The choice checks in this module fold case first (``curve.upper()``,
    ``mode.lower()``), so a non-string used to die in the stdlib's attribute
    lookup before the module's own message could name the parameter.

    :param value: The value to validate.
    :param name: Parameter name used in the error message.
    :return: The validated string.
    :raises TypeError: for a non-string value.
    """
    if not isinstance(value, str):
        msg = f"'{name}' must be a string; got {type(value).__name__}."
        raise TypeError(msg)
    return value


#: Source resistance of ITU-R BS.468-4 Fig. 1a, in series with the generator.
_ITU_R_468_SOURCE_OHM = 600.0
#: Load the Fig. 1a response is taken across: the amplifier input, drawn in
#: the figure as ``Z = 600 ohm``.
_ITU_R_468_LOAD_OHM = 600.0
#: The seven reactive elements of ITU-R BS.468-4 Fig. 1a, "Weighting network,
#: simple form" (printed p. 1), in ladder order left to right. Kinds are
#: ``"shunt_c"`` (capacitor to ground, admittance ``s C``), ``"series_l"``
#: (inductor in the series arm, impedance ``s L``) and ``"series_c"`` (the one
#: capacitor in the series arm, impedance ``1 / (s C)``). Values in SI units;
#: the figure prints nF and mH.
_ITU_R_468_LADDER: tuple[tuple[str, float], ...] = (
    ("shunt_c", 13.85e-9),
    ("series_l", 12.88e-3),
    ("shunt_c", 26.82e-9),
    ("series_c", 33.06e-9),
    ("shunt_c", 9.21e-9),
    ("series_l", 26.49e-3),
    ("shunt_c", 31.47e-9),
)


def _abcd_product(
    left: list[list[np.ndarray]], right: list[list[np.ndarray]]
) -> list[list[np.ndarray]]:
    """Product of two 2x2 ABCD matrices whose entries are polynomials in ``s``.

    Each entry is a coefficient array in ``numpy.polynomial`` convention
    (highest power first), so the chain rule of two-port theory becomes an
    ordinary matrix product over the polynomial ring and the ladder's
    transfer function comes out exactly, with no per-frequency evaluation.

    :param left: The upstream two-port, as ``[[A, B], [C, D]]``.
    :param right: The downstream two-port, in the same form.
    :return: The cascade ``left @ right``.
    """
    return [
        [
            np.polyadd(
                np.polymul(left[row][0], right[0][col]),
                np.polymul(left[row][1], right[1][col]),
            )
            for col in (0, 1)
        ]
        for row in (0, 1)
    ]


@lru_cache(maxsize=1)
def _itu_r_468_prototype() -> tuple[tuple[complex, ...], float]:
    r"""Poles and gain of the ITU-R BS.468-4 weighting network.

    Clause 1 of the Recommendation makes the passive network of Fig. 1a the
    primitive: "The nominal response curve of the weighting network is given
    in Fig. 1b which is the theoretical response of the passive network shown
    in Fig. 1a. Table 1 gives the values of this response at various
    frequencies." So the nominal curve is defined at every frequency, and the
    21 rows of Table 1 are that curve sampled and rounded to 0.1 dB, not the
    definition. The network is rebuilt here from the seven printed component
    values by a polynomial ABCD chain, rather than stored as precomputed
    roots, so a reader can check the constants against the figure.

    Chaining the ladder scaled by ``s`` clears the series capacitor's
    ``1 / (s C)`` entry, which is the whole of the algebra: the scaling comes
    back as the numerator of

    .. math::

       H(s) = \frac{K\,s}{s^6 + a_5 s^5 + \dots + a_0},

    one zero at the origin because that series capacitor blocks dc, and six
    poles because ``C2``, ``C3`` and ``C4`` form a capacitive loop whose
    third state is dependent (seven reactive elements, order six). Evaluated
    in double precision the six coefficients agree with exact rational
    arithmetic over the same component values to 3.4e-16 relative, and the
    resulting curve reproduces all 21 rows of Table 1 to 0.0503 dB maximum
    and 0.0264 dB rms, which is the 0.05 dB rounding quantum of the printed
    table (the 100 Hz row overshoots it by 0.0002 dB, a rounding tie the
    four-significant-figure component values cannot resolve).

    :return: The six poles in rad/s, sorted for reproducibility, and the
        gain ``K`` that puts the response at 0 dB at the 1 kHz reference
        frequency Table 1 prints as its zero row.
    """
    one = np.array([1.0])
    zero = np.array([0.0])
    s = np.array([1.0, 0.0])

    abcd = [[one, np.array([_ITU_R_468_SOURCE_OHM])], [zero, one]]
    for kind, value in _ITU_R_468_LADDER:
        if kind == "shunt_c":
            element = [[one, zero], [np.array([value, 0.0]), one]]
        elif kind == "series_l":
            element = [[one, np.array([value, 0.0])], [zero, one]]
        else:  # "series_c": s * [[1, 1 / (s C)], [0, 1]], the scaling above
            element = [[s, np.array([1.0 / value])], [zero, s]]
        abcd = _abcd_product(abcd, element)

    # Loading the two-port: V_out / V_in = 1 / (A + B / R_load), times the s
    # the series capacitor's matrix was scaled by.
    denominator = np.polyadd(abcd[0][0], abcd[0][1] / _ITU_R_468_LOAD_OHM)
    poles = np.roots(denominator / denominator[0])
    w_ref = 2.0 * np.pi * 1000.0
    gain = float(np.abs(np.prod(1j * w_ref - poles)) / w_ref)
    return tuple(sorted((complex(v) for v in poles), key=lambda v: (v.real, v.imag))), (
        gain
    )


#: The curves that have an analog prototype, i.e. everything but the ``Z``
#: bypass. Used to reject an unknown name before anything is designed.
_PROTOTYPE_CURVES = ("A", "B", "C", "D", "G", "AU", "468")

#: Frequency at which each curve's own standard fixes the response to 0 dB:
#: 1 kHz for every audio-band weighting (IEC 61672-1 for A/C, ANSI S1.4-1983
#: for B, IEC 537 for D, IEC 61012 for AU, ITU-R BS.468-4 Table 1's zero row),
#: 10 Hz for G (ISO 7196 clause 4).
_REFERENCE_HZ: dict[str, float] = {"G": 10.0}

#: The frequency interval each standard states its curve over, which is the
#: interval the design is fitted on and the only one over which the realised
#: response is claimed. A/B/C/Z: IEC 61672-1:2013 Table 3 and ANSI S1.4-1983
#: Table IV, 10 Hz to 20 kHz. D: the same span, the withdrawn IEC 537 having
#: left no range of its own (the tabulated curve republished as NASA CR-3406
#: Table SLD-I covers 50 Hz to 10 kHz, well inside). G: ISO 7196:1995 Table 2,
#: 0.25 Hz to 315 Hz. AU: IEC 61012:1990 Table 1, 10 Hz to 40 kHz. 468:
#: ITU-R BS.468-4 Table 1, 31.5 Hz to 31.5 kHz.
_STANDARD_BAND_HZ: dict[str, tuple[float, float]] = {
    "A": (10.0, 20000.0),
    "B": (10.0, 20000.0),
    "C": (10.0, 20000.0),
    "D": (10.0, 20000.0),
    "G": (0.25, 315.0),
    "AU": (10.0, 40000.0),
    "468": (31.5, 31500.0),
}

#: Fraction of the Nyquist frequency the fit band is allowed to reach.
#: Chosen as the smallest single value that keeps every frequency any standard
#: in this corpus states a requirement at inside the fit band at every sample
#: rate the library serves: the binding case is IEC 61672-1 Table 3's 16 kHz
#: row, whose exact frequency 15 848.9 Hz sits at 0.9906 of the Nyquist
#: frequency when fs = 32 kHz. One fraction for eight curves rather than eight
#: tables, and it leaves no undeclared band except the top half percent. It is
#: not free: the magnitude of a real-coefficient filter has zero slope at the
#: Nyquist frequency, so a curve still falling steeply there has to flatten
#: inside the last half percent of the band, which costs about 0.1 dB on the
#: steepest curve in the corpus (BS.468-4, -30 dB/octave, at 44.1 and 48 kHz,
#: against a +/-2 dB mask) and about 0.01 dB on A and C.
_FIT_NYQUIST_FRACTION = 0.995

#: Biquad sections per curve. The floor is the prototype's own degree; the
#: question each row answers is whether one more section buys accuracy the
#: prototype's own order cannot reach. Measured against the analog prototype
#: over the standard's band at 48 kHz, it does for the four curves whose poles
#: are all real and split between the ends of a three-decade band -- A
#: 0.037 -> 0.00027 dB, B 0.015 -> 0.00024, C 0.035 -> 0.00024, D 0.112 ->
#: 0.00023 -- and it does not for the other three, whose residual is set by the
#: flattening at the Nyquist frequency rather than by their order: 468 with a
#: fourth section moves 0.102 to 0.083 dB and AU with a seventh moves 0.108 to
#: 0.089, so both keep the order the standard prints. G needs nothing: its
#: eight poles act below 315 Hz, where the warp is negligible at any audio
#: rate, and four sections land within 1e-7 dB.
_FIT_SECTIONS: dict[str, int] = {
    "A": 4,
    "B": 4,
    "C": 3,
    "D": 3,
    "G": 4,
    "AU": 6,
    "468": 3,
}


def _fit_band(curve: str, fs: int) -> tuple[float, float]:
    """The interval the design is fitted over, in Hz, and claimed over.

    The curve's own standardised range, with the top clipped to
    :data:`_FIT_NYQUIST_FRACTION` of the Nyquist frequency because no digital
    filter can track an analog curve past it.

    :param curve: An upper-case curve name with a prototype.
    :param fs: Sample rate in Hz.
    :return: ``(low, high)`` in Hz.
    """
    low, high = _STANDARD_BAND_HZ[curve]
    return low, min(high, _FIT_NYQUIST_FRACTION * fs / 2.0)


def _a_weighting_zpk() -> tuple[np.ndarray, np.ndarray, float]:
    """Analog A weighting, IEC 61672-1:2013 Annex E (E.4.2).

    ``f1`` and ``f4`` are shared with C (and, through C, with B); ANSI
    S1.4-1983 Appendix C prints the same poles to fewer digits
    (f1 = 20.598997, f4 = 12194.22).
    """
    f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
    poles = -2 * np.pi * np.array([f1, f1, f4, f4, f2, f3])
    return np.zeros(4), poles, 3.5174303309e13


def _u_poles() -> np.ndarray:
    """IEC 61012:1990 Table 2: the six poles of the U low-pass, in rad/s.

    A double real pole at -12 200 Hz and complex-conjugate pairs at
    -7 850 +/- j8 800 Hz and -2 900 +/- j12 150 Hz. Cascaded with A they make
    the AU weighting of subclause 2.2, for measuring audible sound in the
    presence of ultrasound.
    """
    return (
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


def _c_weighting_zpk() -> tuple[np.ndarray, np.ndarray, float]:
    """Analog C weighting, IEC 61672-1:2013 Annex E (E.4.1)."""
    f1, f4 = 20.598997, 12194.217
    poles = -2 * np.pi * np.array([f1, f1, f4, f4])
    return np.zeros(2), poles, 5.91797e8


def _b_weighting_zpk() -> tuple[np.ndarray, np.ndarray, float]:
    """Analog B weighting, ANSI S1.4-1983 Appendix C, Formula (C2).

    The C weighting times ``f^2 / (f^2 + f5^2)`` in power, i.e. one more zero
    at the origin and one extra real pole at ``f5 = 158.48932`` Hz. Historical:
    B was dropped when IEC 61672-1 replaced the older meter standards, and it
    is kept for legacy data only.
    """
    f1, f4, f5 = 20.598997, 12194.217, 158.48932
    poles = -2 * np.pi * np.array([f1, f1, f4, f4, f5])
    return np.zeros(3), poles, 1.0


def _d_weighting_zpk() -> tuple[np.ndarray, np.ndarray, float]:
    """Analog D weighting, withdrawn IEC 537:1976, from the published form.

    Zeros at the origin and at the roots of ``s^2 + 6532 s + 4.0975e7``; poles
    at -1776.3, -7288.5 and the roots of ``s^2 + 21514 s + 3.8836e8``, all in
    rad/s (module docstring). The complex pairs are expanded with the quadratic
    formula so the design is exact and deterministic. Corroborated against SQAT
    and librosa, two independent implementations.
    """
    zero_real = -6532.0 / 2.0
    zero_imag = math.sqrt(4.0975e7 - zero_real**2)
    pole_real = -21514.0 / 2.0
    pole_imag = math.sqrt(3.8836e8 - pole_real**2)
    zeros = np.array(
        [0.0, complex(zero_real, zero_imag), complex(zero_real, -zero_imag)]
    )
    poles = np.array(
        [
            -1776.3,
            -7288.5,
            complex(pole_real, pole_imag),
            complex(pole_real, -pole_imag),
        ]
    )
    return zeros, poles, 1.0


def _g_weighting_zpk() -> tuple[np.ndarray, np.ndarray, float]:
    """Analog G weighting, ISO 7196:1995 Table 1 (p. 2).

    Nominal pole/zero coordinates in the complex frequency plane, in Hz: four
    zeros at the origin and four complex-conjugate pole pairs. The curve is
    defined with 0 dB gain at 10 Hz (clause 4), which is why G is the one curve
    whose reference frequency is not 1 kHz.
    """
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
    return np.zeros(4), 2 * np.pi * pole_coords_hz, 1.0


def _itu_r_468_zpk() -> tuple[np.ndarray, np.ndarray, float]:
    """Analog 468 weighting, ITU-R BS.468-4 clause 1 via Fig. 1a.

    One zero at the origin, because the series capacitor blocks dc, and the six
    poles :func:`_itu_r_468_prototype` reads off the network's seven printed
    component values. Its gain already puts 0 dB at 1 kHz, so the shared
    renormalisation below is an identity here; it is left in place rather than
    special-cased, because it costs one evaluation and keeps the reference in
    one place for every curve.
    """
    poles, gain = _itu_r_468_prototype()
    return np.zeros(1), np.array(poles), gain


def _analog_weighting_zpk(curve: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Analog prototype of *curve*, normalised to 0 dB at its reference.

    :param curve: An upper-case name in :data:`_PROTOTYPE_CURVES`.
    :return: ``(zeros, poles, gain)`` in rad/s.
    """
    if curve == "AU":
        zeros, poles, gain = _a_weighting_zpk()
        poles = np.concatenate([poles.astype(complex), _u_poles()])
    else:
        builders = {
            "A": _a_weighting_zpk,
            "B": _b_weighting_zpk,
            "C": _c_weighting_zpk,
            "D": _d_weighting_zpk,
            "G": _g_weighting_zpk,
            "468": _itu_r_468_zpk,
        }
        zeros, poles, gain = builders[curve]()
    w_ref = 2 * np.pi * _REFERENCE_HZ.get(curve, 1000.0)
    response = gain * np.prod(1j * w_ref - zeros) / np.prod(1j * w_ref - poles)
    return zeros, poles, gain / float(np.abs(response))


@lru_cache(maxsize=64)
def _cached_weighting_sos(curve: str, fs: int, high_accuracy: bool) -> np.ndarray:
    """The shared, read-only design behind :func:`_weighting_sos`."""
    zeros, poles, gain = _analog_weighting_zpk(curve)
    if high_accuracy:
        sos = design_sos(
            zeros,
            poles,
            float(fs),
            _fit_band(curve, fs),
            _FIT_SECTIONS[curve],
            _REFERENCE_HZ.get(curve, 1000.0),
        )
    else:
        sos = signal.zpk2sos(*signal.bilinear_zpk(zeros, poles, gain, fs))
    shared = np.array(sos, dtype=np.float64)
    shared.flags.writeable = False
    return shared


def _weighting_sos(curve: str, fs: int, high_accuracy: bool) -> np.ndarray:
    """Second-order sections of *curve* at *fs*, cached on the three inputs.

    Two designs live here, and ``high_accuracy`` picks between them:

    * ``True`` (the default) fits the prototype at the sample rate with
      :func:`~phonometry.filters._weighting_design.design_sos`, which is the
      accurate realisation and the one the library grades itself on.
    * ``False`` is the plain bilinear transform of the printed prototype: the
      closed-form design a reader can check line by line against the standard,
      at the cost of the frequency warping the fit exists to remove.

    One comparison is worth stating because it looks like a regression and is
    not. Against the *sections* the oversampled design used to produce, the fit
    is worse at 494 of the 8047 (curve, rate) pairs swept from 2 Hz to 200 kHz,
    all but two of them below 2 kHz, worst 6.2 dB for AU at 34 Hz. Against what
    that design *delivered*, it is better at every one of the 8047, because the
    sections were never the whole path: the signal crossed the resampler's
    anti-alias FIR twice, and measured end to end the old median error is
    7.68 dB against this one's 0.0012 dB. So the old sections are a filter this
    library never actually applied, and the rates where they win are ones no
    standard in the corpus places a requirement at.

    The fit costs about 260 ms, which is why it is cached: it is paid once per
    (curve, rate, mode), and it is still a third of the 775 ms the resampled
    path it replaces spent *filtering* one minute of 468-weighted audio, and
    about 70 % of the A curve's 377. Where the first start
    does not land close enough, the routine works through the other starts of
    :func:`~phonometry.filters._weighting_design._spare_placements` and the
    design takes about 3 s instead. That is common below about 2 kHz, and it
    also happens at scattered rates inside the audio band: measured over every
    rate from 2 Hz to 200 kHz, 52 of them at or above 2 kHz take the retry, of
    which five are at or above 8 kHz -- 468 at nine rates between 45 and
    64 kHz, AU at two near 34 kHz. None of the thirteen standard rates does,
    and where it fires it always improves the fit. It is the same cache and the
    same once.

    :return: A fresh copy each call. ``WeightingFilter.sos`` is a public
        attribute and a caller may edit it in place (a test in this tree
        appends a notch section to one), so what the cache holds must not be
        the same array.
    """
    return _cached_weighting_sos(curve, fs, high_accuracy).copy()


class WeightingFilter:
    """Class-based frequency weighting filter (A, B, C, D, G, AU, 468, Z).
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
            ultrasound), '468' (ITU-R BS.468-4 psophometric noise weighting,
            see :func:`_itu_r_468_prototype`) or 'Z'.
        :param stateful: If True, carry the section state between calls, so
            concatenated blocks equal one continuous call. Available for every
            curve and independent of ``high_accuracy``: both designs are a
            plain cascade of second-order sections at the input rate, and
            ``sosfilt`` with a carried ``zi`` reproduces a single call bit for
            bit.
        :param steady_ic: If True, calculate steady state initial conditions for filter.
        :param high_accuracy: Which of the two designs to build, both of them
            second-order sections at the input rate.

            ``True`` (the default) fits the analog prototype at *fs*, undoing
            the bilinear frequency warping instead of tolerating it
            (:mod:`phonometry.filters._weighting_design`). Measured against
            the prototype over the standard's own band, the worst deviation is
            0.008 dB for A at 32 kHz and 0.0003 dB at 48 kHz, and at most
            0.06 dB for any curve at any rate in this corpus -- the 0.06
            belonging to the 468 curve at 44.1 and 48 kHz, where a
            -30 dB/octave skirt has to flatten inside the last half percent
            below the Nyquist frequency. A and C verify to class 1 at every
            sample rate from 8 kHz up, and at every Table 3 row their
            deviation stays inside the 0.05 dB the table itself is rounded to.
            The design is fitted over the interval :func:`_fit_band` returns
            and is not claimed outside it.

            ``False`` is the plain bilinear transform of the printed
            prototype: the closed-form design a reader can check against the
            standard term by term, at the cost of the warping. That cost grows
            quadratically toward the Nyquist frequency -- for A at 48 kHz it
            reads 15.7 dB below the design goal at the 19 952.6 Hz row, which
            the class 1 mask does not see because its lower limit there is
            -inf, and 61.4 dB below it at 15 848.9 Hz when fs = 32 kHz, which
            it does. So it verifies to class 1 for fs >= 44 100 Hz, degrades to
            class 2 at 32 000 and 22 050 Hz, and meets no class at 16 000 Hz.
            It is refused for the '468' curve, whose skirt puts it 23 dB out at
            16 kHz with no lower grade in the Recommendation to fall back to.

            Defaults to True (``None`` selects the default too, which is
            what it used to mean in every mode but the stateful one).
        """
        if fs <= 0:
            raise ValueError(_FS_POSITIVE)
        curve = _require_str(curve, "curve")
        if high_accuracy is None:
            # ``None`` used to mean "True unless stateful", because the two
            # were mutually exclusive. They are not any more, so it resolves
            # to the accurate design in every mode. The sentinel is kept
            # rather than replaced by a plain ``True`` default so that a
            # caller still passing ``None`` gets the default rather than a
            # value that reads as false.
            high_accuracy = True

        self.fs = fs
        self.curve = curve.upper()
        self.stateful = stateful
        self.high_accuracy = high_accuracy

        if self.curve == "Z":
            self.sos = np.array([])
            if self.stateful:
                self.zi = np.array([])
            return

        if self.curve not in _PROTOTYPE_CURVES:
            msg = "Weighting curve must be 'A', 'B', 'C', 'D', 'G', 'AU', '468' or 'Z'"
            raise ValueError(msg)

        if self.curve == "468" and not self.high_accuracy:
            # The plain bilinear design compresses frequency quadratically in
            # f / fs over a skirt falling at about -30 dB/octave: at 48 kHz it
            # reads -23.2 dB at the 16 kHz row of ITU-R BS.468-4 Table 1, whose
            # tolerance there is +/-1.6 dB. Unlike A and C, the Recommendation
            # prints one mask and no class-2 grade to degrade to, so this is
            # refused rather than documented. Stateful processing no longer
            # implies it, so stateful '468' is now available.
            msg = (
                "Weighting curve '468' needs the fitted design: "
                "high_accuracy=False puts the response 23 dB below the "
                "ITU-R BS.468-4 Table 1 nominal at 16 kHz for fs = 48000 Hz, "
                "against a +/-1.6 dB tolerance."
            )
            raise ValueError(msg)

        self.sos = _weighting_sos(self.curve, fs, self.high_accuracy)

        # Initialize filter state for stateful block-wise processing.
        # Uses lazy allocation: zi is sized on first filter() call so that
        # the channel dimension matches the actual input shape.
        if self.stateful:
            self.zi = np.array([])
            self._steady_ic = steady_ic

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
        if self.curve == "Z" or x_proc.shape[-1] == 0:
            # ``sosfilt`` refuses a record with no samples, and there is
            # nothing to weight in one: hand it back in the shape it arrived
            # in, leaving any carried state where it was.
            return like_input(x, x_proc)

        if self.stateful:
            if self._needs_zi_reinit(x_proc):
                self._init_filter_state(x_proc)
            y, self.zi = signal.sosfilt(self.sos, x_proc, axis=-1, zi=self.zi)
        else:
            y = signal.sosfilt(self.sos, x_proc, axis=-1)

        return like_input(x, cast(np.ndarray, y))


def _runtime_frequency_response(
    wf: WeightingFilter, frequencies: np.ndarray
) -> np.ndarray:
    """Complex steady-state response of the *whole* filter path.

    The path is one cascade of second-order sections at the input rate, for
    every curve and in both stateful and single-shot use, so this is a
    ``sosfreqz`` and nothing else. It stays a function of its own because it is
    what :mod:`phonometry.filters.weighting_compliance` and the conformance
    report measure, and a verdict must describe the filter the caller runs. The
    library used to reach its sections through an interpolation and a
    decimation stage, and this function had to fold the images those stages
    aliased back; with the design fitted at the input rate
    (:mod:`phonometry.filters._weighting_design`) there is nothing left between
    the input and the sections to model.

    :param wf: The weighting filter whose path is measured.
    :param frequencies: Frequencies in Hz, below the input Nyquist frequency.
    :return: Complex voltage response at each frequency (not normalized).
    """
    f = np.asarray(frequencies, dtype=np.float64)
    if wf.curve == "Z" or wf.sos.size == 0:
        return np.ones_like(f, dtype=np.complex128)
    _, h = signal.sosfreqz(wf.sos, worN=f, fs=wf.fs)
    return np.asarray(h, dtype=np.complex128)


@lru_cache(maxsize=32)
def _cached_weighting_filter(
    fs: int, curve: str, high_accuracy: bool
) -> WeightingFilter:
    """Reuse the (immutable, non-stateful) weighting-filter object.

    A non-stateful ``WeightingFilter`` never mutates its SOS in ``filter()``,
    so the object can be shared across repeated ``weighting_filter()`` calls at
    the same rate and curve. The coefficients behind it are cached separately
    by :func:`_weighting_sos`, which is where the design cost actually sits, so
    a caller that builds the class directly does not pay the fit twice either.
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

    def __array__(
        self, dtype: DTypeLike | None = None, copy: bool | None = None
    ) -> np.ndarray:
        """Return the envelope as an array (optionally recast)."""
        return np.asarray(self.mean_square, dtype=dtype, copy=copy)

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
        infrasound), 'AU' (IEC 61012), '468' (ITU-R BS.468-4 psophometric
        noise weighting) or 'Z' (bypass).
    :param high_accuracy: Design the filter by fitting the analog prototype at
        *fs* (default True), rather than by the plain bilinear transform, which
        warps frequency and loses class 1 below 44.1 kHz. See
        :meth:`WeightingFilter.__init__` for what each design costs. The '468'
        curve requires the fitted design and refuses False.
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

    try:
        state = np.asarray(initial_state, dtype=x_sq.dtype)
    except (TypeError, ValueError):
        # numpy refuses an object with TypeError and an unconvertible string
        # array with ValueError; neither message names the parameter or the
        # accepted forms, so both become the module's own TypeError.
        raise TypeError(invalid_initial_state_message) from None
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
    mode = _require_str(mode, "mode")
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
        mode = _require_str(mode, "mode")
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
    if fs <= 0:
        raise ValueError(_FS_POSITIVE)
    if order % 2 != 0:
        msg = "Linkwitz-Riley order must be even (typically 2 or 4)."
        raise ValueError(msg)
    if order <= 0:
        # An even order of zero slipped through the parity check and returned
        # both bands as the untouched input (their sum is twice the signal).
        msg = (
            f"'order' must be a positive even integer (typically 2 or 4); got {order}."
        )
        raise ValueError(msg)
    freq = require_positive(freq, "freq")
    nyquist = fs / 2
    if freq >= nyquist:
        msg = f"'freq' must be below the Nyquist frequency ({nyquist:g} Hz); got {freq:g}."
        raise ValueError(msg)

    # A Linkwitz-Riley filter of order N is two Butterworth filters of order N/2 in series
    half_order = order // 2
    wn = freq / nyquist

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
