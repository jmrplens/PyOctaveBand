#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Evaluation of machine vibration by measurement (ISO 20816-1:2016).

Condition monitoring answers two questions, and they are not the same one.
:mod:`~phonometry.vibration.machinery.diagnostics` answers *where a fault
would show*, by turning the geometry of a bearing or a gear pair into the
frequencies it excites. This module answers the prior question a plant
actually asks: **is this machine acceptable at all**, from one broad-band
magnitude measured at a bearing.

ISO 20816-1 is the basis document of the series, merging what used to be
ISO 10816-1 and ISO 7919-1. It fixes the shape of the answer and leaves the
numbers to the machine-specific parts.

**Four evaluation zones** (6.3.2.3) grade a machine rather than pass or fail
it. Zone **A** is where newly commissioned machines normally fall; zone **B**
is acceptable for unrestricted long-term operation; zone **C** is
unsatisfactory for long-term continuous running, though the machine may run a
limited period until remedial action can be arranged; zone **D** is severe
enough to cause damage. Three boundaries separate them, and
:func:`evaluation_zone` is the comparison itself, blind to the quantity: the
specific parts set boundaries on shaft displacement, housing velocity or
housing acceleration, and the grading is the same in all three.

**Criterion I** (6.3.2) is that comparison applied to the vibration severity,
the largest broad-band magnitude measured at any bearing at rated speed.
Velocity carries it over a wide speed range, but a single velocity limit
regardless of frequency allows unacceptable displacement at low frequency and
unacceptable acceleration at high frequency. So the criterion is a curve, flat
between two corner frequencies and sloped outside them
(Figure 9, Formula (C.1)):

.. math::

   v_\mathrm{rms} = v_A \, Z_\mathrm{bound}
   \left(\frac{f_z}{f_x}\right)^{k}
   \left(\frac{f_y}{f_w}\right)^{m} \tag{C.1}

with :math:`f_z = f` below the lower corner :math:`f_x` and :math:`f_x` above
it, and :math:`f_w = f` above the upper corner :math:`f_y` and :math:`f_y`
below it, so both bracketed factors are unity between the corners.
:math:`Z_\mathrm{bound}` moves the one curve onto the three boundaries, and
Annex C.2 prints the factors it takes: 1 for the limit of zone A, **2,56** for
zone B and **6,4** for zone C.

**Criterion II** (6.3.3) judges a *change* from an established baseline, and a
change is a vector. Annex D makes the point with a machine whose magnitude
fell from 3 mm/s to 2,5 mm/s while its phase swung from 40° to 180°: the
magnitude moved by half a millimetre per second, and the vibration itself
moved by **5,2 mm/s**, ten times as much. :func:`vibration_vector_change` is
that subtraction.

Where no part of the series covers a machine and no experience is available,
Annex C.1 offers Table C.1: a ladder of preferred magnitudes and the range
each boundary is typically drawn from, with small machines at the low end and
large flexibly supported ones at the high end. They are a starting point for
agreement between supplier and customer, not an acceptance specification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from ..._internal.types import as_float_or_array
from ..._internal.validation import (
    require_choice,
    require_finite_fields,
    require_non_negative,
    require_positive,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike, NDArray

#: The four grades of ISO 20816-1, 6.3.2.3, worst last.
EvaluationZone = Literal["A", "B", "C", "D"]

#: The ladder of preferred magnitudes Table C.1 is drawn on, in millimetres
#: per second r.m.s. It is an R10 series over most of its length, which is why
#: consecutive boundaries sit about 1,6 apart.
TYPICAL_BOUNDARY_LADDER_MM_S: tuple[float, ...] = (
    0.28,
    0.45,
    0.71,
    1.12,
    1.8,
    2.8,
    4.5,
    7.1,
    9.3,
    11.2,
    14.7,
    18.0,
    28.0,
    45.0,
)

#: Table C.1: the range each zone boundary is typically drawn from, for
#: vibration measured on non-rotating parts, in millimetres per second r.m.s.
#: Small machines (electric motors up to 15 kW) sit at the low end and large
#: ones on flexible supports at the high end.
TYPICAL_ZONE_BOUNDARY_RANGES_MM_S: dict[str, tuple[float, float]] = {
    "A/B": (0.71, 4.5),
    "B/C": (1.8, 9.3),
    "C/D": (4.5, 14.7),
}

#: Annex C.2: the factor ``Zbound`` of Formula (C.1) that moves the zone A
#: curve onto the limit of each zone.
ZONE_LIMIT_FACTORS: dict[str, float] = {"A": 1.0, "B": 2.56, "C": 6.4}


@dataclass(frozen=True)
class ZoneBoundaries:
    """The three magnitudes that separate the four evaluation zones.

    The unit is whichever the machine-specific part states: micrometres of
    shaft displacement, millimetres per second of housing velocity or metres
    per second squared of housing acceleration. Nothing here converts between
    them, and the boundaries and the magnitude judged against them have to be
    the same quantity.

    :ivar a_b: The zone A/B boundary, below which a newly commissioned machine
        normally sits.
    :ivar b_c: The zone B/C boundary, the limit of unrestricted long-term
        operation.
    :ivar c_d: The zone C/D boundary, above which the vibration is severe
        enough to damage the machine.
    """

    a_b: float
    b_c: float
    c_d: float

    def __post_init__(self) -> None:
        """Reject boundaries that do not rise through the four zones.

        A grading is only a grading if its cuts are ordered: boundaries out of
        order would put zone C below zone B and grade a good machine as a bad
        one without raising anything.

        :raises ValueError: if a boundary is not positive and finite, or the
            three do not increase.
        """
        require_finite_fields(self, "a_b", "b_c", "c_d")
        for name in ("a_b", "b_c", "c_d"):
            require_positive(float(getattr(self, name)), name)
        if not self.a_b < self.b_c < self.c_d:
            msg = (
                "ZoneBoundaries: the three boundaries must increase through "
                f"the zones; got a_b={self.a_b!r}, b_c={self.b_c!r}, "
                f"c_d={self.c_d!r}."
            )
            raise ValueError(msg)

    @property
    def as_tuple(self) -> tuple[float, float, float]:
        """The three boundaries in order, for a plot or a table."""
        return (float(self.a_b), float(self.b_c), float(self.c_d))


def evaluation_zone(
    magnitude: ArrayLike, boundaries: ZoneBoundaries
) -> EvaluationZone | NDArray[np.str_]:
    """Grade a vibration magnitude into zone A, B, C or D (6.3.2.3).

    The boundaries belong to a zone each: a magnitude exactly on the A/B
    boundary is the limit of zone A and is graded ``"A"``, which is how a
    limit reads in the tables of the machine-specific parts.

    :param magnitude: The vibration severity, in the same quantity and unit as
        ``boundaries`` (scalar or array).
    :param boundaries: The three zone boundaries of the applicable part of
        ISO 20816.
    :return: ``"A"``, ``"B"``, ``"C"`` or ``"D"``; a string for a scalar
        input, otherwise an array of them.
    :raises ValueError: If a magnitude is negative or not finite.
    """
    values = np.asarray(magnitude, dtype=np.float64)
    if np.any(values < 0.0) or not np.all(np.isfinite(values)):
        msg = "'magnitude' must be non-negative and finite."
        raise ValueError(msg)
    cuts = np.asarray(boundaries.as_tuple, dtype=np.float64)
    # side="left" puts a magnitude exactly on a boundary in the lower zone.
    index = np.searchsorted(cuts, values, side="left")
    zones = np.array(["A", "B", "C", "D"])[index]
    # Indexing with a 0-d index yields a numpy string scalar, not a 0-d array,
    # so it is converted rather than indexed.
    return str(zones) if zones.ndim == 0 else zones  # type: ignore[return-value]


def allowable_velocity(
    frequency: ArrayLike,
    *,
    constant_velocity_mm_s: float,
    zone_factor: float = 1.0,
    corner_low_hz: float,
    corner_high_hz: float,
    exponent_low: float = 1.0,
    exponent_high: float = 1.0,
) -> np.ndarray | float:
    r"""Frequency-shaped velocity criterion of Figure 9 (Formula (C.1)).

    Flat between the two corner frequencies and sloped outside them. The
    default exponents of 1 are the physical reading of that shape: below the
    lower corner the criterion holds displacement constant, so the allowable
    velocity *rises* with frequency at 6 dB per octave, and above the upper
    corner it holds acceleration constant, so it falls at the same rate. The
    curve is therefore lowest where the machine is slowest and where it is
    fastest, and flat in between. A machine-specific part that states its own
    ``k`` and ``m`` overrides them.

    :param frequency: Frequency ``f``, in hertz (scalar or array).
    :param constant_velocity_mm_s: The constant r.m.s. velocity ``vA`` that
        applies between the corners for zone A, in millimetres per second.
    :param zone_factor: ``Zbound``, the factor that moves the curve onto a
        zone limit; see :data:`ZONE_LIMIT_FACTORS` for the 1 / 2,56 / 6,4 of
        Annex C.2.
    :param corner_low_hz: The lower corner ``fx``, in hertz.
    :param corner_high_hz: The upper corner ``fy``, in hertz; must exceed
        ``corner_low_hz``.
    :param exponent_low: ``k``, the slope below the lower corner.
    :param exponent_high: ``m``, the slope above the upper corner.
    :return: The allowable r.m.s. velocity, in millimetres per second; a float
        for a scalar frequency, otherwise an array.
    :raises ValueError: If a frequency is not positive, a velocity or corner
        is not positive and finite, or the corners are not in order.
    """
    v_a = require_positive(constant_velocity_mm_s, "constant_velocity_mm_s")
    z = require_positive(zone_factor, "zone_factor")
    f_x = require_positive(corner_low_hz, "corner_low_hz")
    f_y = require_positive(corner_high_hz, "corner_high_hz")
    if f_y <= f_x:
        msg = "'corner_high_hz' must exceed 'corner_low_hz'."
        raise ValueError(msg)
    for name, value in (
        ("exponent_low", exponent_low),
        ("exponent_high", exponent_high),
    ):
        if not math.isfinite(value):
            msg = f"'{name}' must be finite."
            raise ValueError(msg)
    f = np.asarray(frequency, dtype=np.float64)
    if np.any(f <= 0.0) or not np.all(np.isfinite(f)):
        msg = "'frequency' must be positive and finite."
        raise ValueError(msg)
    f_z = np.minimum(f, f_x)
    f_w = np.maximum(f, f_y)
    values = v_a * z * (f_z / f_x) ** exponent_low * (f_y / f_w) ** exponent_high
    return as_float_or_array(values)


@dataclass(frozen=True)
class VectorChangeResult:
    """A change in vibration between two states, as a vector (Annex D).

    :ivar magnitude: The magnitude of the change, in the unit of the two
        states it was built from.
    :ivar phase_deg: The direction of the change, in degrees within [0, 360).
    :ivar initial: The magnitude and phase of the initial state.
    :ivar final: The magnitude and phase of the final state.
    """

    magnitude: float
    phase_deg: float
    initial: tuple[float, float]
    final: tuple[float, float]

    @property
    def magnitude_change(self) -> float:
        """What a magnitude-only comparison would have reported.

        The difference of the two magnitudes, signed. Annex D exists because
        this number and :attr:`magnitude` can disagree by an order of
        magnitude, and only the second is the change in the vibration.
        """
        return float(self.final[0] - self.initial[0])

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Draw the polar diagram of Figure D.1.

        The two states as vectors from the origin and the change as the vector
        joining their tips, which is the picture that makes the point.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.

        :param ax: Existing polar axes, or ``None`` to create a figure.
        :param language: Label language, ``"en"`` (default) or ``"es"``.
        :param kwargs: ``unit``, the name of the unit the two states were
            given in, and anything forwarded to the change chord; see
            :func:`phonometry._plot.vibration.plot_vector_change`.
        """
        from ..._i18n import check_language
        from ..._plot.vibration import plot_vector_change

        check_language(language)
        return plot_vector_change(self, ax=ax, language=language, **kwargs)


def vibration_vector_change(
    initial_magnitude: float,
    initial_phase_deg: float,
    final_magnitude: float,
    final_phase_deg: float,
) -> VectorChangeResult:
    """The vector change in vibration between two steady states (Annex D).

    Criterion II is written on a change from an established baseline, and a
    broad-band magnitude cannot express one: a component that swings in phase
    changes the vibration even as the magnitude it contributes falls. Annex D
    prints the case, 3 mm/s at 40 degrees becoming 2,5 mm/s at 180 degrees,
    where the magnitude drops by half a millimetre per second and the
    vibration moves by 5,2.

    :param initial_magnitude: Magnitude of the reference state, in any unit;
        the result carries the same one.
    :param initial_phase_deg: Phase of the reference state, in degrees.
    :param final_magnitude: Magnitude of the later state, in the same unit.
    :param final_phase_deg: Phase of the later state, in degrees.
    :return: The :class:`VectorChangeResult`.
    :raises ValueError: If a magnitude is negative or a value is not finite.
    """
    for name, value in (
        ("initial_magnitude", initial_magnitude),
        ("final_magnitude", final_magnitude),
    ):
        if not math.isfinite(value) or value < 0.0:
            msg = f"'{name}' must be non-negative and finite."
            raise ValueError(msg)
    for name, value in (
        ("initial_phase_deg", initial_phase_deg),
        ("final_phase_deg", final_phase_deg),
    ):
        if not math.isfinite(value):
            msg = f"'{name}' must be finite."
            raise ValueError(msg)
    first = initial_magnitude * np.exp(1j * math.radians(initial_phase_deg))
    second = final_magnitude * np.exp(1j * math.radians(final_phase_deg))
    change = second - first
    return VectorChangeResult(
        magnitude=float(abs(change)),
        phase_deg=float(math.degrees(math.atan2(change.imag, change.real)) % 360.0),
        initial=(float(initial_magnitude), float(initial_phase_deg)),
        final=(float(final_magnitude), float(final_phase_deg)),
    )


# --------------------------------------------------------------------------- #
# Industrial machines: the boundaries the series prints for them
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MachineZoneLimits:
    """The two boundary sets one machine class is judged on at once.

    ISO 10816-3:2009 states each class twice, once in displacement and once
    in velocity, and 5.2.3 says which wins when they disagree: the more
    restrictive zone applies. Both are read on the same non-rotating part,
    and A.1 says velocity alone is enough in most cases and that a spectrum
    expected to carry low-frequency components should be judged on both.

    :ivar displacement_um: The three boundaries in r.m.s. displacement,
        micrometres.
    :ivar velocity_mm_s: The three boundaries in r.m.s. velocity, millimetres
        per second.
    """

    displacement_um: ZoneBoundaries
    velocity_mm_s: ZoneBoundaries


#: Tables A.1 and A.2 of ISO 10816-3:2009, keyed by machine group and support
#: class. Group 1 is large machines above 300 kW and up to 50 MW, and
#: electrical machines of shaft height 315 mm or more; group 2 is medium-sized
#: machines above 15 kW up to 300 kW, and electrical machines of shaft height
#: from 160 mm up to but not including 315 mm. A support system whose lowest
#: natural frequency in the measuring direction is at least 25 % above the
#: main excitation frequency counts as rigid, and every other one as flexible
#: (4.3). The boundaries hold for the broad-band r.m.s. value between 10 Hz
#: and 1 kHz, or from 2 Hz for a machine below 600 r/min (A.1).
#:
#: The edition is superseded, by ISO 20816-3, which is not held. It is the
#: predecessor of that part rather than a competing document, and the ISO
#: 20816-1 framework these boundaries are used with cites the series as the
#: place its numbers live.
INDUSTRIAL_MACHINE_ZONES: dict[tuple[str, str], MachineZoneLimits] = {
    ("group_1", "rigid"): MachineZoneLimits(
        displacement_um=ZoneBoundaries(29.0, 57.0, 90.0),
        velocity_mm_s=ZoneBoundaries(2.3, 4.5, 7.1),
    ),
    ("group_1", "flexible"): MachineZoneLimits(
        displacement_um=ZoneBoundaries(45.0, 90.0, 140.0),
        velocity_mm_s=ZoneBoundaries(3.5, 7.1, 11.0),
    ),
    ("group_2", "rigid"): MachineZoneLimits(
        displacement_um=ZoneBoundaries(22.0, 45.0, 71.0),
        velocity_mm_s=ZoneBoundaries(1.4, 2.8, 4.5),
    ),
    ("group_2", "flexible"): MachineZoneLimits(
        displacement_um=ZoneBoundaries(37.0, 71.0, 113.0),
        velocity_mm_s=ZoneBoundaries(2.3, 4.5, 7.1),
    ),
}

#: The fraction of the upper limit of zone B that 5.3 calls a significant
#: change, and that 5.4.1 adds to the baseline to set an ALARM.
SIGNIFICANT_CHANGE_FRACTION = 0.25
#: The multiple of a zone's upper limit that 5.4.1 and 5.4.2 recommend an
#: ALARM and a TRIP should not exceed.
OPERATIONAL_LIMIT_HEADROOM = 1.25


def industrial_machine_zone(
    group: str,
    support: str,
    *,
    displacement_um: float | None = None,
    velocity_mm_s: float | None = None,
) -> EvaluationZone:
    """Grade an industrial machine against Table A.1 or A.2.

    Give whichever quantities were measured. With both, 5.2.3 applies and the
    result is the more restrictive of the two gradings, which is the whole
    reason the tables state each class twice.

    :param group: ``"group_1"`` (above 300 kW and not more than 50 MW, or an
        electrical machine of shaft height 315 mm or more) or ``"group_2"``
        (above 15 kW up to and including 300 kW, or shaft height from 160 mm
        up to but not including 315 mm), from 4.2.
    :param support: ``"rigid"`` or ``"flexible"``; rigid means the lowest
        natural frequency of the support in the measuring direction is at
        least 25 % above the main excitation frequency (4.3).
    :param displacement_um: Measured broad-band r.m.s. displacement of the
        bearing, pedestal or housing, in micrometres.
    :param velocity_mm_s: Measured broad-band r.m.s. velocity, in millimetres
        per second.
    :return: ``"A"``, ``"B"``, ``"C"`` or ``"D"``.
    :raises ValueError: If the group or support class is unknown, or neither
        measured quantity is given.
    """
    key = (
        require_choice(group, "group", ("group_1", "group_2")),
        require_choice(support, "support", ("rigid", "flexible")),
    )
    limits = INDUSTRIAL_MACHINE_ZONES[key]
    graded = [
        str(evaluation_zone(value, boundaries))
        for value, boundaries in (
            (displacement_um, limits.displacement_um),
            (velocity_mm_s, limits.velocity_mm_s),
        )
        if value is not None
    ]
    if not graded:
        msg = (
            "Give at least one of 'displacement_um' and 'velocity_mm_s'; the "
            "tables grade a machine on whichever was measured."
        )
        raise ValueError(msg)
    # "The severity zone which is most restrictive shall apply" (5.2.3), and
    # the zone letters sort in the order of severity.
    return cast("EvaluationZone", max(graded))


def is_significant_change(change: float, zone_b_upper: float) -> bool:
    """Whether a change from the baseline is significant (5.3).

    A change exceeding a quarter of the upper limit of zone B, in either
    direction, is one that 5.3 asks to be investigated even when zone C of
    Criterion I has not been reached: the machine can go wrong well inside
    zone B, and the criterion that catches it is the change rather than the
    magnitude.

    :param change: The change from the established baseline, in the unit the
        boundaries are in; the sign is ignored.
    :param zone_b_upper: The upper limit of zone B, which is the B/C boundary.
    :return: Whether the change is significant.
    :raises ValueError: If the change is not finite or the limit is not
        positive.
    """
    if not math.isfinite(change):
        msg = "'change' must be finite."
        raise ValueError(msg)
    upper = require_positive(zone_b_upper, "zone_b_upper")
    return abs(change) > SIGNIFICANT_CHANGE_FRACTION * upper


def alarm_limit(baseline: float, zone_b_upper: float) -> float:
    """The ALARM setting a baseline and a zone B/C boundary imply (5.4.1).

    The recommendation is two rules at once: set the ALARM a quarter of the
    upper limit of zone B above the established baseline, and do not normally
    let it exceed 1,25 times that limit. A machine with a low baseline
    therefore alarms below zone C, which the clause says in as many words, and
    one with a high baseline is capped rather than allowed to drift up with
    it.

    :param baseline: The steady-state baseline for that measurement position
        and direction, in the unit the boundaries are in.
    :param zone_b_upper: The upper limit of zone B, which is the B/C boundary.
    :return: The ALARM setting.
    :raises ValueError: If the baseline is negative or the limit is not
        positive.
    """
    base = require_non_negative(baseline, "baseline")
    upper = require_positive(zone_b_upper, "zone_b_upper")
    return min(
        base + SIGNIFICANT_CHANGE_FRACTION * upper,
        OPERATIONAL_LIMIT_HEADROOM * upper,
    )


def trip_limit(zone_c_upper: float) -> float:
    """The largest TRIP setting 5.4.2 recommends.

    Unlike an ALARM, a TRIP is not set from a baseline: it relates to the
    mechanical integrity of the machine and is the same for every machine of
    a design. The clause declines to give absolute values and gives a ceiling
    instead, 1,25 times the upper limit of zone C.

    :param zone_c_upper: The upper limit of zone C, which is the C/D boundary.
    :return: The recommended maximum TRIP setting.
    :raises ValueError: If the limit is not positive.
    """
    return OPERATIONAL_LIMIT_HEADROOM * require_positive(zone_c_upper, "zone_c_upper")
