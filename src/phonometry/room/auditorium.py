#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Auditorium measures derived from impulse responses (ISO 3382-1:2009, Annex A).

Annex A is **informative**. Nothing about the sound strength is normative in
ISO 3382-1:2009: not Equations (A.1) to (A.9), not the 1 dB just-noticeable
difference of Table A.1, not the typical range. What the annex fixes is the
definition everyone quotes, and that is what this module implements.

Reverberation time says what the room does to energy over time and nothing
about how loud the room is. Sound strength, G, is the quantity that does:
the energy of the measured impulse response against the energy the same
source puts out at 10 m in a free field (A.2.1, Equation (A.1)). It is one
of the two measures in Table A.1 that need a calibrated source, the other
being the late lateral sound level of A.2.5, which is referred to the same
place; and it is the reason the annex spends four equations on how to obtain
that free-field reference when there is no anechoic room 10 m across to
measure it in.

The three printed routes to the reference are all here, and they are
routes to the same number:

- measure at a distance ``d >= 3 m`` and correct by the inverse square law,
  Equations (A.4) and (A.8);
- measure the source in a reverberation room of known absorption area,
  Equation (A.5);
- take the source's sound power level and subtract the free-field spread,
  Equation (A.9).

The last two carry printed integers, 37 dB and 31 dB, and both are the
correctly rounded value of a closed form: :math:`10\lg(1600\pi)` is
37,0127 dB and :math:`10\lg(400\pi)` is 30,9921 dB. Rounded to whole
decibels they land 6 dB apart, where the exact offsets are
:math:`10\lg 4 = 6{,}0206` dB apart, so the reverberation-room route and
the sound-power route cannot agree to better than 0,0206 dB. That is
2 % of the 1 dB just-noticeable difference Table A.1 prints for G, and
this module reproduces the standard's integers rather than the closed
forms: a library that quietly used 30,9921 dB would disagree with every
hand calculation done from the printed page.

Both closed forms hold at a characteristic impedance of exactly
:math:`400~\text{N s/m}^3`, which is the value that makes the three reference
quantities consistent: :math:`p_0^2 S_0 / \rho c = (20\ \mu\mathrm{Pa})^2
/ 400 = 1` pW, the reference sound power. Neither equation prints that
caveat. Air at 20 degrees and 101,325 kPa is nearer :math:`413~\text{N s/m}^3`,
worth 0,14 dB, an order of magnitude more than either rounding: the offsets
are a convention of the decibel scales, not a property of the air in the
hall, and this module does not make them follow the weather.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .._internal.validation import require_choice, require_finite_array
from .._internal.warnings import PhonometryWarning
from ..io._resolve import resolve_fs
from ._shared import (
    noise_power,
    onset_index,
    split_bands,
    truncation,
    validate_ir,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike, NDArray

    from ..io._signal import Signal

#: Reference sound pressure, 20 uPa (ISO 3382-1:2009, A.2.1).
_P0 = 2.0e-5

#: Reference time, 1 s, of the sound pressure exposure level
#: (ISO 3382-1:2009, Equations (A.2) and (A.3)).
_T0 = 1.0

#: Default octave-band analysis range (ISO 3382-1:2009, 5.1).
_DEFAULT_BANDS = (125.0, 4000.0)

#: Free-field distance the sound strength is referenced to, in metres
#: (ISO 3382-1:2009, A.2.1).
REFERENCE_DISTANCE_M = 10.0

#: Shortest source-to-microphone distance the inverse-square correction of
#: Equations (A.4) and (A.8) is printed for, in metres: both are given for
#: "a point which is d (>= 3 m) from the source".
MINIMUM_REFERENCE_DISTANCE_M = 3.0

#: Coarsest angular step the note under Equation (A.4) allows when averaging
#: the source's directivity out of a free-field measurement, in degrees.
#: The note prints "every 12,5 degrees", which does not divide a full turn:
#: see :func:`directivity_energy_average`.
MAXIMUM_DIRECTIVITY_STEP_DEG = 12.5

#: The integral of Equations (A.2) and (A.3) runs to a time "greater than or
#: equal to the point at which the decay curve has decreased by 30 dB"
#: (ISO 3382-1:2009, A.2.1). A room response whose noise floor sits less
#: than this far below its peak was cut short of that point.
#:
#: The rule is checked on the room response only. The free-field response is
#: a direct arrival with no reverberant decay behind it, and what falls away
#: after its peak is the band filter's own ring-down: a legitimate 50 ms
#: anechoic window gives the 125 Hz octave band about 28 dB of it, which
#: says nothing about the measurement and everything about the filter.
_MINIMUM_DECAY_RANGE_DB = 30.0

#: Filter ring-downs a response must hold after its direct sound before its
#: lowest band can be said to have finished.
#:
#: A fractional-octave filter is a resonator, and its own energy decays over
#: roughly 2,2 Q / f_c seconds to 60 dB, which at 125 Hz in an octave band is
#: about 25 ms. A record that stops before that has not collected the band's
#: energy, and its exposure level is short. Measured against a long
#: reference, an anechoic window of 40 ms after the direct sound is 0,04 dB
#: light at 125 Hz, one of 20 ms is 0,9 dB light, one of 10 ms is 13 dB light
#: and one of 2 ms is 40 dB light. One ring-down separates the harmless from
#: the rest, and it is what this module asks for.
_MINIMUM_RING_DOWNS = 1.0

#: Offset of Equation (A.9), in decibels: the free-field spread from a sound
#: power level to the sound pressure level at 10 m. The closed form is
#: 10 lg(4 pi * 100) = 30,9921 dB; the standard prints the rounded integer
#: and so does this module.
SOUND_STRENGTH_POWER_OFFSET_DB = 31.0

#: Offset of Equation (A.5), in decibels, for the reverberation-room route to
#: the free-field reference level. The closed form is 10 lg(1600 pi) =
#: 37,0127 dB; the standard prints the rounded integer and so does this
#: module. See the module docstring for why the two integers cannot close
#: to better than 0,0206 dB.
DIFFUSE_FIELD_REFERENCE_OFFSET_DB = 37.0


class AuditoriumWarning(PhonometryWarning):
    """A measurement outside the conditions ISO 3382-1:2009 prints for it."""


def _filter_quality_factor(fraction: int) -> float:
    r"""Quality factor of an IEC 61260 filter of the given bandwidth fraction.

    The band edges sit at :math:`f_c 2^{\pm 1/2n}`, so the relative
    bandwidth is :math:`2^{1/2n} - 2^{-1/2n}` and Q is its reciprocal: 1,41
    for an octave and 4,32 for a one-third octave, whatever the centre.
    """
    edge = float(2.0 ** (1.0 / (2.0 * fraction)))
    return 1.0 / (edge - 1.0 / edge)


def _exposure_level(p2: NDArray[np.float64], fs: int) -> float:
    """Sound pressure exposure level of one squared band response.

    The integral is truncated where the fitted decay meets the noise floor
    and the missing tail is compensated with the fitted rate, which is the
    treatment ISO 3382-1:2009, 5.3.3, Equation (3) prints for exactly this
    integral and which :func:`phonometry.room.decay_curve` already gives the
    same response. Without it the answer would depend on how long the tape
    ran: A.2.1 puts a lower bound on the upper limit and no upper one, and
    every second of noise past the decay adds energy that is not the
    source's. A response with no measurable noise floor is integrated whole,
    so a synthetic one is unaffected.
    """
    index, tail, _ = truncation(p2, fs, noise_power(p2))
    energy = float(np.sum(p2[:index])) / fs + tail
    return float(10.0 * np.log10(energy / (_T0 * _P0**2)))


def _band_energy(
    band: NDArray[np.float64], index: int, name: str
) -> NDArray[np.float64]:
    """One band squared, from its own direct sound onwards (A.2.1)."""
    p2 = band.astype(np.float64) ** 2
    if not np.any(p2 > 0.0):
        msg = f"Band {index} of '{name}' has no energy."
        raise ValueError(msg)
    return p2[onset_index(p2) :]


def _decay_range_is_thin(p2: NDArray[np.float64]) -> bool:
    """Whether a band never falls the 30 dB A.2.1 asks the integral to reach.

    A response with no measurable noise floor decays as far as it is asked
    to, so it is not thin.
    """
    floor = noise_power(p2)
    if floor <= 0.0:
        return False
    return bool(float(np.max(p2)) / floor < 10.0 ** (_MINIMUM_DECAY_RANGE_DB / 10.0))


def _ring_down_seconds(centre_hz: float, fraction: int) -> float:
    """How long a band filter of that centre needs to ring down.

    The 2,2 is the number of cycles a one-pole envelope of quality factor
    :math:`Q` takes to fall 60 dB divided by :math:`Q`, so the product is
    the settling time of the filter and not of the room.
    """
    return _MINIMUM_RING_DOWNS * 2.2 * _filter_quality_factor(fraction) / centre_hz


def _warn_short_ring_down(name: str, short: list[tuple[float, float]]) -> None:
    """Warn about the band that is furthest from having rung down."""
    if not short:
        return
    needed, held = max(short)
    warnings.warn(
        f"'{name}' holds {held * 1e3:.0f} ms after its direct sound and "
        f"a band of it needs about {needed * 1e3:.0f} ms for its filter "
        "to ring down, so that band's exposure level is short of the "
        "energy the response actually carries.",
        AuditoriumWarning,
        stacklevel=4,
    )


def _warn_cut_short(name: str) -> None:
    """Warn about a room response that never reaches 30 dB of decay."""
    warnings.warn(
        f"'{name}' has a band whose noise floor is less than "
        f"{_MINIMUM_DECAY_RANGE_DB:g} dB below its peak, so the room "
        "response was cut short of the point ISO 3382-1:2009, A.2.1 asks "
        "the integral to reach.",
        AuditoriumWarning,
        stacklevel=4,
    )


def _band_exposure_levels(
    ir: Signal | list[float] | NDArray[np.float64],
    fs: int,
    limits: tuple[float, float] | None,
    fraction: int,
    name: str,
    *,
    check_decay: bool = False,
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64]]:
    """Per-band exposure levels of one response, from its own direct sound."""
    x = validate_ir(ir, fs)
    frequency, bands = split_bands(x, fs, limits, fraction, name="limits")
    levels = np.empty(len(bands), dtype=np.float64)
    thin = False
    short: list[tuple[float, float]] = []
    for index, band in enumerate(bands):
        p2 = _band_energy(band, index, name)
        thin = thin or _decay_range_is_thin(p2)
        if frequency is not None:
            needed = _ring_down_seconds(float(frequency[index]), fraction)
            held = p2.size / fs
            if held < needed:
                short.append((needed, held))
        levels[index] = _exposure_level(p2, fs)
    _warn_short_ring_down(name, short)
    if thin and check_decay:
        _warn_cut_short(name)
    return frequency, levels


def sound_pressure_exposure_level(
    ir: Signal | list[float] | NDArray[np.float64],
    fs: int | None = None,
    *,
    limits: tuple[float, float] | None = _DEFAULT_BANDS,
    fraction: int = 1,
) -> NDArray[np.float64] | float:
    r"""Sound pressure exposure level of an impulse response, per band.

    ISO 3382-1:2009, Equation (A.2):

    .. math::

       L_{pE} = 10 \lg \left[ \frac{1}{T_0}
                \int_0^{\infty} \frac{p^2(t)}{p_0^2} \mathrm{d}t \right]
                \ \mathrm{dB}

    with :math:`T_0 = 1` s and :math:`p_0 = 20` uPa. Time zero is the start
    of the direct sound (A.2.1), found per band with the A.3.4 trigger, and
    the integral runs to the end of the response supplied.

    A.2.1 asks that end to lie at or beyond the point where the decay curve
    has fallen 30 dB. Whether a room response reaches it is visible in the
    ``dynamic_range`` of
    :func:`phonometry.room.room_parameters`, and
    :func:`sound_strength` raises :class:`AuditoriumWarning` when it does
    not; this function evaluates whatever it is given, because the same
    equation is (A.3) applied to a free-field response, which has no
    reverberant decay to reach 30 dB of.

    This is the quantity Equations (A.3) to (A.5) also operate on: the same
    function applied to the free-field response at 10 m gives
    :math:`L_{pE,10}`.

    :param ir: Measured impulse response (1D). A
        :class:`phonometry.io.Signal` brings its calibration, which is
        applied; a bare array is read as pascals, so its exposure level is
        only referenced to 20 uPa if the samples already are.
    :param fs: Sample rate in Hz. Required for a bare array; a
        :class:`~phonometry.io.Signal` brings its own, and an explicit value
        that disagrees with it raises instead of silently winning.
    :param limits: ``(f_min, f_max)`` band-centre limits in Hz; default the
        octave bands 125 Hz to 4 kHz (ISO 3382-1:2009, 5.1). ``None``
        integrates the broadband response as a single band.
    :param fraction: Bandwidth fraction (1 = octave, 3 = one-third octave).
    :return: The exposure level in dB, one entry per band, or a
        :class:`float` for a broadband response.
    :raises ValueError: If the response is not one-dimensional or is silent,
        if ``limits`` is malformed, or if a band has no energy.
    """
    fs = resolve_fs(ir, fs, name="ir")
    _, levels = _band_exposure_levels(ir, fs, limits, fraction, "ir")
    return float(levels[0]) if limits is None else levels


def free_field_reference_level(
    level: ArrayLike,
    distance: float,
) -> NDArray[np.float64] | float:
    r"""Free-field level at 10 m from one measured at another distance.

    ISO 3382-1:2009, Equations (A.4) and (A.8), which are the same
    inverse-square correction written once for the exposure level and once
    for the stationary-source pressure level:

    .. math::

       L_{pE,10} = L_{pE,d} + 20 \lg (d/10)\ \mathrm{dB}

    Both are printed for a point "d (>= 3 m) from the source", far enough
    for the free field to have taken over; a shorter distance raises
    :class:`AuditoriumWarning`.

    The note under (A.4) adds that the measurement is to be repeated around
    the source and energy-averaged, "at every 12,5 degrees", to average out
    the source's own directivity. Feed this function the averaged level,
    not one bearing; :func:`directivity_energy_average` is that mean, and
    its docstring says what the printed step does and does not determine.

    :param level: The level measured at ``distance``, in dB. Any shape.
    :param distance: Source-to-microphone distance of that measurement, in
        metres.
    :return: The level referred to 10 m, in dB, in the shape of ``level``.
    :raises ValueError: If ``distance`` is not a positive, finite length.
    """
    values = np.asarray(level, dtype=np.float64)
    if not np.isfinite(distance) or distance <= 0.0:
        msg = f"'distance' must be a positive, finite length in m, got {distance!r}."
        raise ValueError(msg)
    if distance < MINIMUM_REFERENCE_DISTANCE_M:
        warnings.warn(
            f"ISO 3382-1:2009, Equations (A.4) and (A.8) are printed for a "
            f"distance of at least {MINIMUM_REFERENCE_DISTANCE_M:g} m; "
            f"{distance:g} m is inside that, where the inverse-square law "
            "need not hold.",
            AuditoriumWarning,
            stacklevel=2,
        )
    corrected = values + 20.0 * np.log10(distance / REFERENCE_DISTANCE_M)
    return float(corrected) if corrected.ndim == 0 else corrected


def directivity_energy_average(
    levels: ArrayLike, axis: int = -1
) -> NDArray[np.float64] | float:
    r"""Average a free-field measurement over bearings around the source.

    The note under ISO 3382-1:2009, Equation (A.4) asks for the free-field
    reference to be measured all the way round the source and combined as
    an energy mean, so that the source's own directivity does not decide
    the reference level:

    .. math::

       \overline{L} = 10 \lg \left( \frac{1}{N}
                       \sum_{i=1}^{N} 10^{L_i/10} \right)\ \mathrm{dB}

    An arithmetic mean of the decibel values is a different number, and a
    lower one, for any source that is not omnidirectional.

    The note asks for the measurement "at every 12,5 degrees", and 12,5
    degrees does not divide 360: 28 steps stop at 350 degrees and 29
    overshoot to 362,5. There is no set of bearings that follows the
    printed instruction literally, so what this function checks is what
    the instruction can mean, a uniform sampling of the full turn no
    coarser than the printed step: ``N >= 29`` bearings, 360/N degrees
    apart. The 5 degree survey of 4.2.1, which the same standard uses for
    the source directivity of Table 1, does divide 360 exactly, into 72.

    The mean itself is the one
    :func:`phonometry.building.energy_average_level` computes; this
    function adds the bearing count the note constrains, and averages one
    band per row when it is handed a two-dimensional survey.

    :param levels: The levels measured at the bearings, in dB, one per
        bearing, evenly spaced around the source. A two-dimensional array
        averages one band per row (or per column, see ``axis``), which is
        the shape the rest of this module works in.
    :param axis: Axis the bearings run along; the last by default.
    :return: Their energy mean, in dB: a :class:`float` for a single turn,
        an array with ``axis`` removed for several.
    :raises ValueError: If fewer than ``ceil(360 / 12,5) = 29`` bearings lie
        along ``axis``, or the levels are empty or non-finite.
    """
    values = np.asarray(levels, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        msg = "'levels' must be a non-empty array of finite decibel values."
        raise ValueError(msg)
    bearings = values.shape[axis] if values.ndim else 0
    minimum = math.ceil(360.0 / MAXIMUM_DIRECTIVITY_STEP_DEG)
    if bearings < minimum:
        msg = (
            f"'levels' holds {bearings} bearing(s) along axis {axis}; "
            "averaging the source directivity out of a free-field reference "
            f"needs a full turn sampled no coarser than "
            f"{MAXIMUM_DIRECTIVITY_STEP_DEG:g} degrees, so at least "
            f"{minimum}."
        )
        raise ValueError(msg)
    mean = np.asarray(
        10.0 * np.log10(np.mean(10.0 ** (values / 10.0), axis=axis)),
        dtype=np.float64,
    )
    return float(mean) if mean.ndim == 0 else mean


def reverberation_room_reference_level(
    reverberation_room_level: ArrayLike,
    absorption_area: ArrayLike,
) -> NDArray[np.float64] | float:
    r"""Free-field reference level at 10 m from a reverberation-room measurement.

    ISO 3382-1:2009, Equation (A.5):

    .. math::

       L_{pE,10} = L_{pE} + 10 \lg (A/S_0) - 37\ \mathrm{dB}

    with :math:`S_0 = 1` m². The route exists because a room 10 m across
    and anechoic is rarer than a reverberation room: the source is measured
    in the diffuse field instead, and the absorption area converts that
    reading into the free-field one. ``A`` follows from the room's
    reverberation time through Sabine's formula, Equation (A.6), which the
    library publishes as
    :func:`phonometry.room.sabine_absorption_area`. Watch the constant when
    reproducing a hand calculation: (A.6) prints :math:`A = 0{,}16\,V/T`,
    which is :math:`24 \ln 10 / c_0` at :math:`c_0 = 345{,}4` m/s, while
    that function defaults to 343 m/s and so to 0,1611. The difference moves
    :math:`10\lg(A/S_0)` by 0,030 dB; pass ``speed_of_sound=345.39`` to get
    the printed constant back.

    The printed 37 dB is :math:`10\lg(1600\pi) = 37{,}0127` dB rounded;
    see the module docstring for what that rounding costs.

    (A.5) also carries no Waterhouse correction, unlike the reverberation-room
    sound power method of ISO 3741 that it otherwise mirrors. The omitted
    :math:`10\lg(1 + S\lambda/8V)` is worth over a decibel in the 125 Hz
    band of a small room, above the 1 dB just-noticeable difference Table A.1
    gives G. That is a property of the printed method, and this function
    reproduces the method rather than quietly improving it.

    :param reverberation_room_level: Spatial-average sound pressure exposure
        level measured in the reverberation room, in dB. The standard calls
        this :math:`L_{pE}` too, which is the same symbol it gave the level
        measured in the hall under test a page earlier; substituting
        (A.5) into (A.1) as printed would cancel the hall out of G
        altogether. The two roles get different names here, and
        ``docs/ERRATA.md`` carries the rest.
    :param absorption_area: Equivalent sound absorption area of that room,
        in m², broadcast against ``reverberation_room_level``.
    :return: The reference exposure level at 10 m, in dB.
    :raises ValueError: If the absorption area is not a positive, finite area.
    """
    levels = np.asarray(reverberation_room_level, dtype=np.float64)
    area = np.asarray(absorption_area, dtype=np.float64)
    if not np.all(np.isfinite(area)) or np.any(area <= 0.0):
        msg = "'absorption_area' must be a positive, finite area in m^2."
        raise ValueError(msg)
    reference = np.asarray(
        levels + 10.0 * np.log10(area) - DIFFUSE_FIELD_REFERENCE_OFFSET_DB,
        dtype=np.float64,
    )
    return float(reference) if reference.ndim == 0 else reference


def sound_strength_from_power(
    pressure_level: ArrayLike,
    power_level: ArrayLike,
) -> NDArray[np.float64] | float:
    r"""Sound strength from the source's sound power level.

    ISO 3382-1:2009, Equation (A.9):

    .. math::

       G = L_p - L_W + 31\ \mathrm{dB}

    The third route to G, and the only one that needs no free-field
    measurement at all: the 31 dB is the spread of a point source over the
    sphere of radius 10 m, :math:`10\lg(4\pi \cdot 100) = 30{,}9921` dB,
    rounded. A.2.1 asks for the source's power level to be measured to
    ISO 3741, which the library implements in
    :mod:`phonometry.emission.sound_power`.

    :param pressure_level: Sound pressure level at the measurement point in
        the room under test, in dB.
    :param power_level: Sound power level of the source, in dB, broadcast
        against ``pressure_level``.
    :return: The sound strength G, in dB.
    """
    pressure = np.asarray(pressure_level, dtype=np.float64)
    power = np.asarray(power_level, dtype=np.float64)
    strength = np.asarray(
        pressure - power + SOUND_STRENGTH_POWER_OFFSET_DB, dtype=np.float64
    )
    return float(strength) if strength.ndim == 0 else strength


@dataclass(frozen=True)
class SoundStrengthResult:
    """Per-band sound strength G and the two levels it is the difference of.

    ``frequency`` holds the exact band centre frequencies in Hz, or is
    ``None`` for a broadband measurement, in which case every array has
    length 1. ``strength`` is G in dB (ISO 3382-1:2009, Equation (A.1)),
    ``exposure_level`` the sound pressure exposure level of the response
    measured in the room (Equation (A.2)) and ``reference_level`` that of
    the free-field response at 10 m (Equation (A.3)), however it was
    obtained. All three are in decibels and ``strength`` is exactly
    ``exposure_level - reference_level``.
    """

    frequency: NDArray[np.float64] | None
    strength: NDArray[np.float64]
    exposure_level: NDArray[np.float64]
    reference_level: NDArray[np.float64]

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes | np.ndarray:
        """Plot the per-band sound strength against the Table A.1 range.

        With ``ax`` given, only the strength panel is drawn on it; otherwise a
        second panel shows the two levels G is the difference of. Requires
        matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes` (or an array of two).
        """
        from .._i18n import check_language
        from .._plot.room import plot_sound_strength

        check_language(language)
        return plot_sound_strength(self, ax=ax, language=language, **kwargs)


def sound_strength(
    ir: Signal | list[float] | NDArray[np.float64],
    reference_ir: Signal | list[float] | NDArray[np.float64] | None = None,
    fs: int | None = None,
    *,
    reference_level: ArrayLike | None = None,
    limits: tuple[float, float] | None = _DEFAULT_BANDS,
    fraction: int = 1,
) -> SoundStrengthResult:
    r"""Sound strength G of a measured impulse response, per band.

    ISO 3382-1:2009, Equation (A.1):

    .. math::

       G = 10 \lg \frac{\int_0^{\infty} p^2(t) \mathrm{d}t}
                         {\int_0^{\infty} p_{10}^2(t) \mathrm{d}t}
         = L_{pE} - L_{pE,10}\ \mathrm{dB}

    Equation (A.7) is the same difference between stationary-source levels
    rather than exposure levels, which is why this function accepts the
    reference either as a second impulse response or as a level already in
    hand: exactly one of ``reference_ir`` and ``reference_level`` is
    required. A level obtained from
    :func:`free_field_reference_level`,
    :func:`reverberation_room_reference_level` or, through
    :func:`sound_strength_from_power`, from the source's power level, goes
    in the second slot.

    Both responses are split into the same bands and each is integrated
    from its own direct sound, so a common gain on the pair cancels
    exactly. A gain applied to only one of them does not: the two
    recordings must share a calibration, which is the whole reason G needs
    a calibrated source where every other measure in Table A.1 does not.

    :param ir: Impulse response measured in the room under test (1D).
    :param reference_ir: Impulse response of the same source at 10 m in a
        free field (1D). Mutually exclusive with ``reference_level``.
    :param fs: Sample rate in Hz. Required for bare arrays; a
        :class:`~phonometry.io.Signal` brings its own.
    :param reference_level: The free-field reference exposure level
        :math:`L_{pE,10}` in dB, as a scalar or one value per band.
        Mutually exclusive with ``reference_ir``.
    :param limits: ``(f_min, f_max)`` band-centre limits in Hz; default the
        octave bands 125 Hz to 4 kHz (ISO 3382-1:2009, 5.1). ``None``
        measures the broadband response as a single band.
    :param fraction: Bandwidth fraction (1 = octave, 3 = one-third octave).
    :return: A :class:`SoundStrengthResult` with one entry per band.
    :raises ValueError: If neither or both reference forms are given, if a
        response is not one-dimensional or is silent, or if
        ``reference_level`` does not broadcast onto the band axis.
    """
    if (reference_ir is None) == (reference_level is None):
        msg = (
            "Give the free-field reference exactly once: either "
            "'reference_ir', the response measured at 10 m, or "
            "'reference_level', the level already obtained from it."
        )
        raise ValueError(msg)

    fs = resolve_fs(ir, fs, name="ir")
    frequency, levels = _band_exposure_levels(
        ir, fs, limits, fraction, "ir", check_decay=True
    )

    if reference_ir is not None:
        reference_fs = resolve_fs(reference_ir, fs, name="reference_ir")
        _, reference = _band_exposure_levels(
            reference_ir, reference_fs, limits, fraction, "reference_ir"
        )
    else:
        given = np.asarray(reference_level, dtype=np.float64)
        try:
            reference = np.broadcast_to(given, levels.shape).astype(np.float64)
        except ValueError as exc:
            msg = (
                f"'reference_level' has shape {given.shape}, which does not "
                f"broadcast onto the {levels.size} analysis band(s)."
            )
            raise ValueError(msg) from exc

    return SoundStrengthResult(
        frequency=frequency,
        strength=levels - reference,
        exposure_level=levels,
        reference_level=reference,
    )


#: Integration window of the numerator of Equations (A.14) and (A.15), in
#: seconds from the direct sound. The 5 ms lower limit exists to keep the
#: direct sound out of the lateral fraction: it is not the 0 the denominator
#: starts at, and both bounds are printed.
EARLY_LATERAL_WINDOW_S = (0.005, 0.080)

#: Upper limit of the denominator of Equations (A.14) and (A.15), in seconds
#: from the direct sound: the total early energy the lateral share is taken of.
EARLY_ENERGY_LIMIT_S = 0.080

#: Lower limit of the numerator of Equation (A.16), in seconds from the direct
#: sound: the late lateral level counts what arrives after the early window.
LATE_LATERAL_START_S = 0.080

#: The four octave bands Equation (A.17) energy-averages, in Hz.
LATE_LATERAL_AVERAGE_BANDS_HZ = (125.0, 250.0, 500.0, 1000.0)

#: Half-width of the search window of Equation (B.2), in seconds: IACC is the
#: largest magnitude the correlation function reaches within one millisecond
#: of coincidence, which is about the interaural delay of a head.
IACC_SEARCH_S = 0.001

#: Time window of the early interaural coefficient, IACC_E (B.4), in seconds.
IACC_EARLY_WINDOW_S = (0.0, 0.080)

#: Start of the late interaural coefficient, IACC_L (B.4), in seconds. B.4
#: leaves its end open, "a time greater than the reverberation time".
IACC_LATE_START_S = 0.080

#: Just-noticeable difference of IACC, assumed by B.4.
IACC_JND = 0.075

#: The two angular weightings Equations (A.14) and (A.15) print for the early
#: lateral energy fraction.
LATERAL_WEIGHTINGS = ("squared", "cosine")


def _window_energy(
    p2: NDArray[np.float64], fs: int, start: float, stop: float | None
) -> float:
    """Energy of a squared response between two times from its own sample 0.

    A closed window that runs past the end of the response raises rather
    than shortening: an 80 ms early energy taken over the 48 ms that were
    recorded is not the printed quantity, and it is larger than it, because
    the missing tail only ever appears in the denominator.
    """
    first = round(start * fs)
    wanted = None if stop is None else round(stop * fs)
    if wanted is not None and wanted > p2.size:
        msg = (
            f"The response holds {p2.size / fs:.4g} s after its direct "
            f"sound, and ISO 3382-1:2009 integrates to {stop:g} s; "
            "shortening the window would not give the printed quantity."
        )
        raise ValueError(msg)
    last = p2.size if wanted is None else wanted
    if first >= last:
        msg = (
            f"The response is too short for the {start:g} s to "
            f"{stop if stop is not None else float('inf'):g} s window "
            f"ISO 3382-1:2009 asks for at fs={fs} Hz."
        )
        raise ValueError(msg)
    return float(np.sum(p2[first:last])) / fs


@dataclass(frozen=True)
class LateralEnergyResult:
    r"""Per-band early lateral energy fraction (ISO 3382-1:2009, A.2.4).

    ``frequency`` holds the exact band centre frequencies in Hz, or is
    ``None`` for a broadband measurement. ``energy_fraction`` is
    :math:`J_\mathrm{LF}` or :math:`J_\mathrm{LFC}` depending on
    ``weighting``, which is ``"squared"`` for Equation (A.14) and
    ``"cosine"`` for Equation (A.15). Both are dimensionless and Table A.1
    gives them a just-noticeable difference of 0,05 and a typical range of
    0,05 to 0,35 over the 125 Hz to 1 kHz octave bands.
    """

    frequency: NDArray[np.float64] | None
    energy_fraction: NDArray[np.float64]
    weighting: str

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the per-band lateral fraction against the Table A.1 range.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.room import plot_lateral_energy

        check_language(language)
        return plot_lateral_energy(self, ax=ax, language=language, **kwargs)


def early_lateral_energy_fraction(
    ir: Signal | list[float] | NDArray[np.float64],
    lateral_ir: Signal | list[float] | NDArray[np.float64],
    fs: int | None = None,
    *,
    weighting: str = "squared",
    limits: tuple[float, float] | None = _DEFAULT_BANDS,
    fraction: int = 1,
) -> LateralEnergyResult:
    r"""Early lateral energy fraction of a pair of responses, per band.

    ISO 3382-1:2009, Equation (A.14), the share of the first 80 ms that
    arrives from the side:

    .. math::

       J_\mathrm{LF} = \frac{\int_{0,005}^{0,080} p_L^2(t)\ \mathrm{d}t}
                            {\int_{0}^{0,080} p^2(t)\ \mathrm{d}t}

    and Equation (A.15), the cosine-weighted variant that A.2.4 calls
    subjectively more accurate:

    .. math::

       J_\mathrm{LFC} = \frac{\int_{0,005}^{0,080}
                              \left| p_L(t) \cdot p(t) \right|\ \mathrm{d}t}
                             {\int_{0}^{0,080} p^2(t)\ \mathrm{d}t}

    The modulus in (A.15) is printed and it matters: the figure-of-eight
    response changes sign with the side a reflection comes from, so without
    it two mirror-image reflections cancel to zero instead of adding.

    The two lower limits differ. The numerator starts at 5 ms because the
    null of the figure-of-eight microphone is pointed at the source and the
    5 ms keeps whatever leaks through it out of the lateral share; the
    denominator starts at 0 because the direct sound belongs to the total.

    **Time zero comes from the omnidirectional response.** A figure-of-eight
    microphone aimed as A.2.4 asks has no direct sound to trigger on, so its
    own onset would land on the first strong reflection and shift both
    limits by that much. Both integrals here are timed from the direct sound
    of ``ir``, band by band.

    :param ir: Impulse response measured with the omnidirectional microphone
        at the measurement point (1D).
    :param lateral_ir: Impulse response measured at the same point with a
        figure-of-eight microphone whose null points at the source (1D).
    :param fs: Sample rate in Hz. Required for bare arrays; a
        :class:`~phonometry.io.Signal` brings its own.
    :param weighting: ``"squared"`` for :math:`J_\mathrm{LF}`,
        Equation (A.14), whose contributions vary as the square of the
        cosine of the angle of incidence; ``"cosine"`` for
        :math:`J_\mathrm{LFC}`, Equation (A.15), whose contributions vary as
        the cosine itself.
    :param limits: ``(f_min, f_max)`` band-centre limits in Hz; default the
        octave bands 125 Hz to 4 kHz. Table A.1 averages this quantity over
        125 Hz to 1 kHz. ``None`` measures the broadband responses.
    :param fraction: Bandwidth fraction (1 = octave, 3 = one-third octave).
    :return: A :class:`LateralEnergyResult` with one entry per band.
    :raises ValueError: If ``weighting`` is not one of
        :data:`LATERAL_WEIGHTINGS`, if a response is not one-dimensional or
        is silent, if the two responses do not split into the same number of
        bands or are of different lengths, or if they are too short for the
        80 ms window.
    """
    require_choice(weighting, "weighting", LATERAL_WEIGHTINGS)
    fs = resolve_fs(ir, fs, name="ir")
    omni = validate_ir(ir, fs)
    lateral = validate_ir(lateral_ir, resolve_fs(lateral_ir, fs, name="lateral_ir"))
    if omni.shape != lateral.shape:
        msg = (
            f"'ir' has {omni.size} samples and 'lateral_ir' has "
            f"{lateral.size}; the two microphones record one event at one "
            "point, so their responses share a time axis."
        )
        raise ValueError(msg)

    frequency, omni_bands = split_bands(omni, fs, limits, fraction)
    _, lateral_bands = split_bands(lateral, fs, limits, fraction)
    start, stop = EARLY_LATERAL_WINDOW_S

    fractions = np.empty(len(omni_bands), dtype=np.float64)
    for index, (band, lateral_band) in enumerate(
        zip(omni_bands, lateral_bands, strict=True)
    ):
        p2 = band.astype(np.float64) ** 2
        if not np.any(p2 > 0.0):
            msg = f"Band {index} of 'ir' has no energy."
            raise ValueError(msg)
        onset = onset_index(p2)
        p2 = p2[onset:]
        lateral_p = lateral_band.astype(np.float64)[onset:]
        numerator = (
            lateral_p**2
            if weighting == "squared"
            else np.abs(lateral_p * band.astype(np.float64)[onset:])
        )
        fractions[index] = _window_energy(numerator, fs, start, stop) / _window_energy(
            p2, fs, 0.0, EARLY_ENERGY_LIMIT_S
        )

    return LateralEnergyResult(
        frequency=frequency, energy_fraction=fractions, weighting=weighting
    )


@dataclass(frozen=True)
class LateLateralResult:
    """Per-band late lateral sound level (ISO 3382-1:2009, A.2.5).

    ``frequency`` holds the exact band centre frequencies in Hz, or is
    ``None`` for a broadband measurement. ``level`` is
    :math:`L_J` in dB (Equation (A.16)) and ``reference_level`` the sound
    pressure exposure level of the free-field response at 10 m it is
    referred to, however that was obtained. Table A.1 gives
    :math:`L_J` a typical range of -14 dB to +1 dB over the 125 Hz to 1 kHz
    octave bands and no just-noticeable difference at all: "Not known".
    """

    frequency: NDArray[np.float64] | None
    level: NDArray[np.float64]
    reference_level: NDArray[np.float64]

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the per-band late lateral level against the Table A.1 range.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.room import plot_late_lateral

        check_language(language)
        return plot_late_lateral(self, ax=ax, language=language, **kwargs)


def late_lateral_sound_level(
    ir: Signal | list[float] | NDArray[np.float64],
    lateral_ir: Signal | list[float] | NDArray[np.float64],
    reference_ir: Signal | list[float] | NDArray[np.float64] | None = None,
    fs: int | None = None,
    *,
    reference_level: ArrayLike | None = None,
    limits: tuple[float, float] | None = _DEFAULT_BANDS,
    fraction: int = 1,
) -> LateLateralResult:
    r"""Late lateral sound level of a measured response, per band.

    ISO 3382-1:2009, Equation (A.16), the level of the lateral energy that
    arrives after the early window, against the same free-field reference
    the sound strength uses:

    .. math::

       L_J = 10 \lg \frac{\int_{0,080}^{\infty} p_L^2(t)\ \mathrm{d}t}
                         {\int_{0}^{\infty} p_{10}^2(t)\ \mathrm{d}t}
             \ \mathrm{dB}

    Where :math:`J_\mathrm{LF}` is a fraction and cancels its own
    calibration, :math:`L_J` is a level and does not: it needs the
    calibrated omnidirectional source of A.2.1 and a free-field reference,
    which reaches this function the same two ways it reaches
    :func:`sound_strength`. It also needs the relative sensitivity of the
    two microphones to have been calibrated in a free field (A.3.2).

    **Time zero comes from the omnidirectional response**, for the reason
    given under :func:`early_lateral_energy_fraction`: a figure-of-eight
    microphone aimed as A.2.5 asks has no direct sound to trigger on, and an
    80 ms boundary placed on its first reflection is not the printed one.
    ``ir`` supplies that time zero and nothing else.

    :param ir: Impulse response measured with the omnidirectional microphone
        at the measurement point (1D), for its direct sound.
    :param lateral_ir: Impulse response measured at the same point with a
        figure-of-eight microphone whose null points at the source (1D).
    :param reference_ir: Impulse response of the same source at 10 m in a
        free field (1D). Mutually exclusive with ``reference_level``.
    :param fs: Sample rate in Hz. Required for bare arrays; a
        :class:`~phonometry.io.Signal` brings its own.
    :param reference_level: The free-field reference exposure level
        :math:`L_{pE,10}` in dB, as a scalar or one value per band.
        Mutually exclusive with ``reference_ir``.
    :param limits: ``(f_min, f_max)`` band-centre limits in Hz; default the
        octave bands 125 Hz to 4 kHz. Equation (A.17) averages this quantity
        over 125 Hz to 1 kHz. ``None`` measures the broadband responses.
    :param fraction: Bandwidth fraction (1 = octave, 3 = one-third octave).
    :return: A :class:`LateLateralResult` with one entry per band.
    :raises ValueError: If neither or both reference forms are given, if a
        response is not one-dimensional or is silent, if the two hall
        responses are of different lengths, or if they are too short to
        reach the 80 ms boundary.
    """
    if (reference_ir is None) == (reference_level is None):
        msg = (
            "Give the free-field reference exactly once: either "
            "'reference_ir', the response measured at 10 m, or "
            "'reference_level', the level already obtained from it."
        )
        raise ValueError(msg)

    fs = resolve_fs(ir, fs, name="ir")
    omni = validate_ir(ir, fs)
    lateral = validate_ir(lateral_ir, resolve_fs(lateral_ir, fs, name="lateral_ir"))
    if omni.shape != lateral.shape:
        msg = (
            f"'ir' has {omni.size} samples and 'lateral_ir' has "
            f"{lateral.size}; the two microphones record one event at one "
            "point, so their responses share a time axis."
        )
        raise ValueError(msg)

    frequency, omni_bands = split_bands(omni, fs, limits, fraction)
    _, lateral_bands = split_bands(lateral, fs, limits, fraction)

    levels = np.empty(len(omni_bands), dtype=np.float64)
    for index, (band, lateral_band) in enumerate(
        zip(omni_bands, lateral_bands, strict=True)
    ):
        p2 = band.astype(np.float64) ** 2
        if not np.any(p2 > 0.0):
            msg = f"Band {index} of 'ir' has no energy."
            raise ValueError(msg)
        late = lateral_band.astype(np.float64)[onset_index(p2) :] ** 2
        energy = _window_energy(late, fs, LATE_LATERAL_START_S, None)
        levels[index] = 10.0 * np.log10(energy / (_T0 * _P0**2))

    if reference_ir is not None:
        _, reference = _band_exposure_levels(
            reference_ir,
            resolve_fs(reference_ir, fs, name="reference_ir"),
            limits,
            fraction,
            "reference_ir",
        )
        if reference.shape != levels.shape:
            msg = (
                f"'lateral_ir' splits into {levels.size} band(s) and "
                f"'reference_ir' into {reference.size}; the two responses "
                "must be long enough for the same bands."
            )
            raise ValueError(msg)
    else:
        given = np.asarray(reference_level, dtype=np.float64)
        try:
            reference = np.broadcast_to(given, levels.shape).astype(np.float64)
        except ValueError as exc:
            msg = (
                f"'reference_level' has shape {given.shape}, which does not "
                f"broadcast onto the {levels.size} analysis band(s)."
            )
            raise ValueError(msg) from exc

    return LateLateralResult(
        frequency=frequency,
        level=levels - reference,
        reference_level=reference,
    )


def late_lateral_average(levels: ArrayLike) -> float:
    r"""Frequency-average the late lateral level over its four octave bands.

    ISO 3382-1:2009, Equation (A.17):

    .. math::

       L_{J,\mathrm{avg}} = 10 \lg \left[ 0{,}25
                            \sum_{i=1}^{4} 10^{L_{J_i}/10} \right]
                            \ \mathrm{dB}

    The 0,25 is one quarter, so this is an energy **mean** and four equal
    band values return that value unchanged. It is the one exception
    footnote a of Table A.1 makes: every other quantity in that table is
    averaged arithmetically over its bands, and only :math:`L_J` is averaged
    over energy. The four bands are the 125 Hz, 250 Hz, 500 Hz and 1 kHz
    octaves, in that order.

    :param levels: The four octave-band values of :math:`L_J`, in dB, in the
        order of :data:`LATE_LATERAL_AVERAGE_BANDS_HZ`.
    :return: :math:`L_{J,\mathrm{avg}}` in dB.
    :raises ValueError: If four values are not given, or they are not finite.
    """
    values = require_finite_array(levels, "levels")
    if values.size != len(LATE_LATERAL_AVERAGE_BANDS_HZ):
        msg = (
            f"'levels' holds {values.size} value(s); Equation (A.17) is "
            f"printed for the {len(LATE_LATERAL_AVERAGE_BANDS_HZ)} octave "
            f"bands {LATE_LATERAL_AVERAGE_BANDS_HZ} Hz, and its printed "
            "weight is one quarter of exactly that many."
        )
        raise ValueError(msg)
    weight = 1.0 / len(LATE_LATERAL_AVERAGE_BANDS_HZ)
    return float(10.0 * np.log10(weight * np.sum(10.0 ** (values / 10.0))))


@dataclass(frozen=True)
class InterauralCorrelationResult:
    r"""Per-band interaural cross correlation (ISO 3382-1:2009, Annex B).

    ``frequency`` holds the exact band centre frequencies in Hz, or is
    ``None`` for a broadband measurement. ``coefficient`` is the IACC of
    Equation (B.2), the largest magnitude the normalised correlation
    function reaches inside the +/-1 ms search window, and ``delay`` the lag
    in seconds at which it reaches it: positive when the right ear leads,
    because (B.1) evaluates the right channel at :math:`t + \tau`.

    ``lag`` and ``correlation`` carry the function itself over the search
    window, one row per band, so it can be drawn rather than summarised.
    B.4 assumes a just-noticeable difference of 0,075.
    """

    frequency: NDArray[np.float64] | None
    coefficient: NDArray[np.float64]
    delay: NDArray[np.float64]
    lag: NDArray[np.float64]
    correlation: NDArray[np.float64]

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the correlation function over the search window, per band.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.room import plot_interaural_correlation

        check_language(language)
        return plot_interaural_correlation(self, ax=ax, language=language, **kwargs)


def interaural_cross_correlation(
    left: Signal | list[float] | NDArray[np.float64],
    right: Signal | list[float] | NDArray[np.float64],
    fs: int | None = None,
    *,
    window: tuple[float, float | None] = (0.0, None),
    limits: tuple[float, float] | None = _DEFAULT_BANDS,
    fraction: int = 1,
) -> InterauralCorrelationResult:
    r"""Interaural cross correlation of a binaural response, per band.

    ISO 3382-1:2009, Equation (B.1) defines the normalised function

    .. math::

       \mathrm{IACF}_{t_1,t_2}(\tau) =
           \frac{\int_{t_1}^{t_2} p_l(t)\, p_r(t + \tau)\ \mathrm{d}t}
                {\sqrt{\int_{t_1}^{t_2} p_l^2(t)\ \mathrm{d}t
                       \int_{t_1}^{t_2} p_r^2(t)\ \mathrm{d}t}}

    and Equation (B.2) takes the coefficient from it:

    .. math::

       \mathrm{IACC}_{t_1,t_2} = \max \left|
           \mathrm{IACF}_{t_1,t_2} \right|
           \quad \text{for } -1\ \mathrm{ms} < \tau < +1\ \mathrm{ms}

    Both the square root and the modulus are printed, and both matter. The
    root is what bounds the function by one; a text layer that drops it
    leaves a quantity that is not a correlation at all. The modulus is what
    makes two anti-phase ears give 1 rather than 0, which is the answer that
    matches what they describe, two signals as dissimilar as they can be
    only in sign.

    B.4 puts three windows to it. The general form runs from the direct
    sound to a time of the order of the reverberation time, which is the
    default here; :data:`IACC_EARLY_WINDOW_S` is the early coefficient and
    :data:`IACC_LATE_START_S` starts the reverberant one. Time zero is the
    direct sound, located on the two channels together so that a listener
    turned away from the source does not move it.

    :param left: Impulse response at the entrance to the left ear canal (1D).
    :param right: Impulse response at the entrance to the right ear canal
        (1D), of the same length.
    :param fs: Sample rate in Hz. Required for bare arrays; a
        :class:`~phonometry.io.Signal` brings its own.
    :param window: ``(t_1, t_2)`` in seconds from the direct sound, with
        ``t_2`` ``None`` for the end of the response.
    :param limits: ``(f_min, f_max)`` band-centre limits in Hz; default the
        octave bands 125 Hz to 4 kHz, which is the range B.4 names. ``None``
        correlates the broadband responses.
    :param fraction: Bandwidth fraction (1 = octave, 3 = one-third octave).
    :return: An :class:`InterauralCorrelationResult` with one entry per band.
    :raises ValueError: If a response is not one-dimensional or is silent, if
        the two channels are of different lengths, if the window is empty or
        reversed, or if a band has no energy inside it.
    """
    fs = resolve_fs(left, fs, name="left")
    left_x = validate_ir(left, fs)
    right_x = validate_ir(right, resolve_fs(right, fs, name="right"))
    if left_x.shape != right_x.shape:
        msg = (
            f"'left' has {left_x.size} samples and 'right' has "
            f"{right_x.size}; the two ears hear one event, so their "
            "responses share a time axis."
        )
        raise ValueError(msg)
    start, stop = window
    if stop is not None and stop <= start:
        msg = f"'window' must run forwards in time, got {window!r}."
        raise ValueError(msg)

    frequency, left_bands = split_bands(left_x, fs, limits, fraction, name="limits")
    _, right_bands = split_bands(right_x, fs, limits, fraction, name="limits")

    # Time zero from the two channels together: a head turned away from the
    # source hears the direct sound in one ear before the other, and the
    # window must not follow the ear that happens to be nearer.
    onset = onset_index(
        left_x.astype(np.float64) ** 2 + right_x.astype(np.float64) ** 2
    )
    reach = round(IACC_SEARCH_S * fs)
    lags = np.arange(-reach, reach + 1, dtype=np.int64)

    curves = np.empty((len(left_bands), lags.size), dtype=np.float64)
    for index, (left_band, right_band) in enumerate(
        zip(left_bands, right_bands, strict=True)
    ):
        curves[index] = _correlation_curve(
            left_band.astype(np.float64)[onset:],
            right_band.astype(np.float64)[onset:],
            fs,
            start,
            stop,
            lags,
            index,
        )

    peak = np.argmax(np.abs(curves), axis=1)
    return InterauralCorrelationResult(
        frequency=frequency,
        coefficient=np.abs(curves[np.arange(curves.shape[0]), peak]),
        delay=lags[peak] / fs,
        lag=lags / fs,
        correlation=curves,
    )


def _correlation_curve(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    fs: int,
    start: float,
    stop: float | None,
    lags: NDArray[np.int64],
    band: int,
) -> NDArray[np.float64]:
    """Equation (B.1) evaluated at every lag of the (B.2) search window.

    The denominator is the one printed: the energies of both channels over
    the same fixed window, not over the window the lag shifts the right
    channel into. Over a millisecond the two readings differ by little, but
    the printed one is what a hand calculation from the page gives.
    """
    first = round(start * fs)
    last = left.size if stop is None else min(left.size, round(stop * fs))
    if first >= last:
        msg = (
            f"The window {start:g} s to "
            f"{stop if stop is not None else float('inf'):g} s holds no "
            f"samples of band {band} at fs={fs} Hz."
        )
        raise ValueError(msg)
    window_left = left[first:last]
    energy_left = float(np.sum(window_left**2))
    energy_right = float(np.sum(right[first:last] ** 2))
    if energy_left <= 0.0 or energy_right <= 0.0:
        msg = (
            f"Band {band} has no energy inside the window, so its "
            "correlation is not defined."
        )
        raise ValueError(msg)

    # p_r(t + tau) with tau running over the search window: the samples that
    # fall outside the record contribute nothing, which is what the upper
    # limit of an integral over a finite recording means.
    products = np.empty(lags.size, dtype=np.float64)
    for position, lag in enumerate(lags):
        low, high = first + int(lag), last + int(lag)
        shifted = np.zeros(window_left.size, dtype=np.float64)
        taken = right[max(0, low) : max(0, min(right.size, high))]
        shifted[max(0, -low) : max(0, -low) + taken.size] = taken
        products[position] = float(np.sum(window_left * shifted))
    return np.asarray(products / np.sqrt(energy_left * energy_right))
