#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Evaluation of machine vibration by measurement (ISO 20816-1).

The basis document of the ISO 20816 series states the shape of the judgement
and leaves the numbers to the machine-specific parts: four evaluation zones,
a velocity criterion that is flat between two corner frequencies and sloped
outside them, and a change read as a vector rather than as a difference of
magnitudes.

Three things in it are numeric all the same, and they are what these rows
pin. Table C.1 prints the ladder of preferred magnitudes and the range each
zone boundary is typically drawn from, for machines no other part covers.
Annex C.2 prints the factors that move the one criterion curve onto the three
boundaries. And Annex D.2 works one change through, with its answer printed.

Oracle: ISO 20816-1:2016, Table C.1 on printed folio 29, Annex C.2 on printed
folio 30 and Annex D.2 on printed folio 31.
"""

from __future__ import annotations

import functools

import numpy as np

import phonometry as ph

from ..registry import Outcome, numeric, register

_MACHINE_VIB = "Machine vibration evaluation (ISO 20816)"

#: Annex D.2: the two steady states the annex compares.
_INITIAL = (3.0, 40.0)
_FINAL = (2.5, 180.0)
#: Annex D.2: "an r.m.s. magnitude of 5,2 mm/s".
_PRINTED_VECTOR_CHANGE = 5.2
#: Annex D.2: "the vibration magnitude has decreased by 0,5 mm/s".
_PRINTED_MAGNITUDE_CHANGE = -0.5
#: The annex prints one decimal.
_TOLERANCE = 0.05

#: Table C.1: the range each boundary is typically drawn from, mm/s r.m.s.
_PRINTED_RANGES = {
    "A/B": (0.71, 4.5),
    "B/C": (1.8, 9.3),
    "C/D": (4.5, 14.7),
}
#: Annex C.2: the factors that move the zone A curve onto the other limits.
_PRINTED_FACTORS = {"A": 1.0, "B": 2.56, "C": 6.4}

#: A criterion shaped like Figure 9, for the rows that exercise Formula (C.1).
_V_A = 1.12
_CORNER_LOW = 10.0
_CORNER_HIGH = 1000.0


@register(
    _MACHINE_VIB,
    "ISO 20816-1:2016 Annex D.2",
    "Vector change between two steady states, mm/s",
)
def _chk_vector_change() -> Outcome:
    """The whole point of Criterion II, in one printed number.

    A magnitude that fell by half a millimetre per second while the phase swung
    140 degrees is a change of 5,2 mm/s, ten times what the magnitudes say.
    """
    result = ph.vibration.vibration_vector_change(*_INITIAL, *_FINAL)
    return numeric(
        _PRINTED_VECTOR_CHANGE, result.magnitude, _TOLERANCE, unit="mm/s", places=3
    )


@register(
    _MACHINE_VIB,
    "ISO 20816-1:2016 Annex D.2",
    "Change a magnitude comparison would report, mm/s",
)
def _chk_magnitude_change() -> Outcome:
    """The number the annex prints to argue against: 2,5 minus 3,0."""
    result = ph.vibration.vibration_vector_change(*_INITIAL, *_FINAL)
    return numeric(
        _PRINTED_MAGNITUDE_CHANGE,
        result.magnitude_change,
        _TOLERANCE,
        unit="mm/s",
        places=3,
    )


def _chk_zone_factor(zone: str) -> Outcome:
    """One of the three factors of Annex C.2, read back off the criterion.

    Recovered as the ratio of the zone limit to the zone A limit rather than
    from the constant, so the row exercises Formula (C.1) rather than
    restating a number.
    """
    common = {
        "constant_velocity_mm_s": _V_A,
        "corner_low_hz": _CORNER_LOW,
        "corner_high_hz": _CORNER_HIGH,
    }
    reference = float(ph.vibration.allowable_velocity(100.0, **common))
    limit = float(
        ph.vibration.allowable_velocity(
            100.0,
            zone_factor=ph.vibration.ZONE_LIMIT_FACTORS[zone],
            **common,
        )
    )
    return numeric(_PRINTED_FACTORS[zone], limit / reference, 0.005, places=4)


def _register_zone_factors() -> None:
    """Register the three zone-limit factors of Annex C.2."""
    for zone in ("A", "B", "C"):
        register(
            _MACHINE_VIB,
            "ISO 20816-1:2016 Annex C.2",
            f"Zone {zone} limit factor Zbound of Formula (C.1)",
        )(functools.partial(_chk_zone_factor, zone))


_register_zone_factors()


def _chk_range(boundary: str, index: int) -> Outcome:
    """One end of one typical boundary range of Table C.1, in mm/s."""
    printed = _PRINTED_RANGES[boundary][index]
    published = ph.vibration.TYPICAL_ZONE_BOUNDARY_RANGES_MM_S[boundary][index]
    return numeric(printed, published, 0.0005, unit="mm/s", places=4)


def _register_ranges() -> None:
    """Register the six ends of the three printed ranges."""
    for boundary in _PRINTED_RANGES:
        for index, end in enumerate(("low", "high")):
            register(
                _MACHINE_VIB,
                "ISO 20816-1:2016 Table C.1",
                f"Typical zone {boundary} boundary, {end} end of the range, mm/s",
            )(functools.partial(_chk_range, boundary, index))


_register_ranges()


@register(
    _MACHINE_VIB,
    "ISO 20816-1:2016 Table C.1",
    "Every range end is a rung of the printed ladder",
)
def _chk_ranges_sit_on_the_ladder() -> Outcome:
    """The six range ends are drawn from the fourteen printed magnitudes.

    Reported as the count of ends that miss the ladder, which is nought.
    """
    ladder = set(ph.vibration.TYPICAL_BOUNDARY_LADDER_MM_S)
    ends = [
        end
        for span in ph.vibration.TYPICAL_ZONE_BOUNDARY_RANGES_MM_S.values()
        for end in span
    ]
    missing = sum(1 for end in ends if end not in ladder)
    return numeric(0.0, float(missing), 0.5, places=1)


@register(
    _MACHINE_VIB,
    "ISO 20816-1:2016 Figure 9",
    "Criterion is flat between the corner frequencies, worst deviation, mm/s",
)
def _chk_criterion_plateau() -> Outcome:
    """Both bracketed factors of Formula (C.1) are unity between the corners."""
    values = np.asarray(
        ph.vibration.allowable_velocity(
            np.geomspace(_CORNER_LOW, _CORNER_HIGH, 40),
            constant_velocity_mm_s=_V_A,
            corner_low_hz=_CORNER_LOW,
            corner_high_hz=_CORNER_HIGH,
        )
    )
    deviation = float(np.max(np.abs(values - _V_A)))
    return numeric(0.0, deviation, 1e-9, unit="mm/s", places=12)


@register(
    _MACHINE_VIB,
    "ISO 20816-1:2016 Figure 9",
    "Constant-displacement slope below the lower corner, dB per octave",
)
def _chk_low_slope() -> Outcome:
    """With k = 1 the allowable velocity halves per octave down, which is 6 dB."""
    low, high = _CORNER_LOW / 4.0, _CORNER_LOW / 2.0
    values = np.asarray(
        ph.vibration.allowable_velocity(
            [low, high],
            constant_velocity_mm_s=_V_A,
            corner_low_hz=_CORNER_LOW,
            corner_high_hz=_CORNER_HIGH,
        )
    )
    slope = 20.0 * np.log10(values[1] / values[0])
    return numeric(6.0206, float(slope), 0.001, unit="dB", places=4)


@register(
    _MACHINE_VIB,
    "ISO 20816-1:2016 Figure 9",
    "Constant-acceleration slope above the upper corner, dB per octave",
)
def _chk_high_slope() -> Outcome:
    """With m = 1 it halves per octave up, the same 6 dB the other way."""
    low, high = _CORNER_HIGH * 2.0, _CORNER_HIGH * 4.0
    values = np.asarray(
        ph.vibration.allowable_velocity(
            [low, high],
            constant_velocity_mm_s=_V_A,
            corner_low_hz=_CORNER_LOW,
            corner_high_hz=_CORNER_HIGH,
        )
    )
    slope = 20.0 * np.log10(values[0] / values[1])
    return numeric(6.0206, float(slope), 0.001, unit="dB", places=4)
