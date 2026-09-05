#  Copyright (c) 2026. Jose Manuel Requena Plens
"""HVAC noise by the German method (VDI 2081), against its own worked example.

VDI 2081 Part 2 exists to anchor Part 1: it carries one supply air duct network
worked element by element, in tabular form, with every intermediate quantity
printed. That makes it an oracle of a kind the ASHRAE side of ``noise_control``
does not have, and for a genuinely different model rather than a restatement of
the same one.

Oracle: VDI 2081 Part 2:2005-05, Table 1 on printed folio 12, element 1.
Method: VDI 2081 Part 1:2001-07, Section 4.3 on printed folios 18 to 25.

Both prints are superseded, by Part 1:2022-04 and Part 2:2022-10, and neither
successor is held. The pair is self-consistent: Part 2:2005 was written against
Part 1:2001 and every cross-reference in its tables resolves there.
"""

from __future__ import annotations

import functools
import math

import numpy as np

import phonometry as ph

from ..registry import Outcome, numeric, register

_VDI2081 = "HVAC noise (VDI 2081)"

#: Table 1, element 1: the supply air fan, 16 000 m3/h against 600 Pa, a radial
#: fan with rearwards curved blades (assembly RR) turning at 1250 min^-1.
_VOLUME_FLOW = 16000.0 / 3600.0
_TOTAL_PRESSURE = 600.0
_SPEED_RPM = 1250.0

#: Table 1, row "Ventilatorspektrum", dB re 1e-12 W.
_PRINTED_SPECTRUM = (90.4, 88.8, 86.3, 82.9, 78.6, 73.4, 67.2, 60.2)
#: Table 1, header row "Frequenzen", Hz. Printed here rather than taken from
#: the library, so the oracle does not lean on the thing under test.
_PRINTED_BANDS = (63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)
#: Table 1, row "A-Korrektur", dB.
_PRINTED_A_WEIGHTING = (-26.2, -16.1, -8.6, -3.2, 0.0, 1.2, 1.0, -1.1)
#: The table prints one decimal.
_PRINTED_TOLERANCE = 0.05


def _worst_band(
    printed: tuple[float, ...] | np.ndarray,
    computed: np.ndarray,
    tolerance: float,
    *,
    unit: str = "dB",
) -> Outcome:
    """Compare two spectra by their worst band rather than by their sum.

    A sum is blind to the shape: any pair of compensating errors passes it, and
    for a correction curve running from +21 to -6 dB a swapped pair or a
    flipped sign leaves the total where it was. The largest absolute deviation
    over the bands is one number that no such error survives.
    """
    deviation = float(np.max(np.abs(np.asarray(printed, dtype=float) - computed)))
    return numeric(0.0, deviation, tolerance, unit=unit, places=4)


def _fan() -> np.ndarray:
    """Element 1 of Table 1, from the printed service data alone."""
    return ph.noise_control.fan_sound_power(
        _VOLUME_FLOW,
        model="vdi2081",
        fan_total_pressure_pa=_TOTAL_PRESSURE,
        assembly="rr",
        fan_speed_rpm=_SPEED_RPM,
    ).values


def _band_outcome(index: int) -> Outcome:
    """One octave of the printed fan spectrum."""
    return numeric(
        _PRINTED_SPECTRUM[index],
        float(_fan()[index]),
        _PRINTED_TOLERANCE,
        unit="dB",
        places=2,
    )


def _register_fan_spectrum() -> None:
    """Register the eight Table 1 rows, one per octave."""
    for index, band in enumerate(_PRINTED_BANDS):
        register(
            _VDI2081,
            "VDI 2081 Blatt 2:2005 Table 1, element 1",
            f"Supply fan sound power at {band:g} Hz, dB",
        )(functools.partial(_band_outcome, index))


_register_fan_spectrum()


@register(
    _VDI2081,
    "VDI 2081 Blatt 1:2001 Eq. (13)",
    "Fan sound power level L_W4 from the duty, dB",
)
def _chk_overall_level() -> Outcome:
    """``L_W4 = L_WSM + 10 lg V + 20 lg dp_t`` with the 34 dB of assembly RR.

    Recovered from the module's own spectrum by taking off the shape of
    Equation (15) and the tenth of a decibel the best duty point is worth, so
    the row exercises the implementation rather than restating the arithmetic.
    """
    strouhal = np.array(_PRINTED_BANDS) * 60.0 / (math.pi * _SPEED_RPM)
    shape = -5.0 - 5.0 * (np.log10(strouhal) + 0.4) ** 2
    recovered = float(np.mean(_fan() - shape - 0.1))
    return numeric(96.0, recovered, _PRINTED_TOLERANCE, unit="dB", places=3)


@register(
    _VDI2081,
    "VDI 2081 Blatt 2:2005 Table 1, element 1",
    "Supply fan total sound power level, dB",
)
def _chk_total() -> Outcome:
    """The unweighted sum of the eight octaves, printed as 94,1 dB."""
    total = 10.0 * math.log10(float(np.sum(10.0 ** (_fan() / 10.0))))
    return numeric(94.1, total, _PRINTED_TOLERANCE, unit="dB", places=3)


@register(
    _VDI2081,
    "VDI 2081 Blatt 2:2005 Table 1, element 1",
    "Supply fan A-weighted sound power level, dB",
)
def _chk_total_a_weighted() -> Outcome:
    """The A-weighted sum, printed as 84,5 dB.

    The weighting is the one Table 1 prints in its own header row, not the one
    the library computes, so the row checks the spectrum and not the weighting.
    """
    weighted = _fan() + np.array(_PRINTED_A_WEIGHTING)
    total = 10.0 * math.log10(float(np.sum(10.0 ** (weighted / 10.0))))
    return numeric(84.5, total, _PRINTED_TOLERANCE, unit="dB", places=3)


# ---------------------------------------------------------------------------
# Table 1 again: the duct elements the same chain runs through
# ---------------------------------------------------------------------------
#: Table 1, element 5: a 500 x 400 mm rectangular duct 4 m long, and the level
#: reduction the table prints for it in every octave.
_RECT_RUN = (0.500, 0.400, 4.000)
_PRINTED_RECT = (2.4, 2.4, 1.2, 0.6, 0.6, 0.6, 0.6, 0.6)
#: Table 1, element 13: a 160 mm round duct, 1 m long.
_PRINTED_ROUND = (0.1, 0.1, 0.15, 0.15, 0.3, 0.3, 0.3, 0.3)
#: Table 1, element 14: a 160 mm round bend in air at 340 m/s, whose limit
#: frequency the table prints as 1245 Hz.
_BEND_DIAMETER = 0.160
_EXAMPLE_SPEED = 340.0
_PRINTED_BEND = (0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0)


@register(
    _VDI2081,
    "VDI 2081 Blatt 2:2005 Table 1, element 5",
    "Rectangular duct 500 x 400 mm over 4 m, worst octave deviation, dB",
)
def _chk_rectangular_run() -> Outcome:
    """Table 5 read by the largest side, which is what puts 250 Hz at 0,3 dB/m."""
    width, height, length = _RECT_RUN
    values = ph.noise_control.unlined_rectangular_duct_attenuation(
        np.array(_PRINTED_BANDS), width, height, length, model="vdi2081"
    ).values
    return _worst_band(_PRINTED_RECT, values, 1e-6)


@register(
    _VDI2081,
    "VDI 2081 Blatt 2:2005 Table 1, element 13",
    "Round duct 160 mm over 1 m, worst octave deviation, dB",
)
def _chk_round_run() -> Outcome:
    """The round rows of Table 5, which do depend on the bore."""
    values = ph.noise_control.unlined_circular_duct_attenuation(
        np.array(_PRINTED_BANDS), 1.000, diameter=0.160, model="vdi2081"
    ).values
    return _worst_band(_PRINTED_ROUND, values, 1e-6)


@register(
    _VDI2081,
    "VDI 2081 Blatt 1:2001 Eq. (34)",
    "Limit frequency of a 160 mm round duct, Hz",
)
def _chk_limit_frequency() -> Outcome:
    """``f_G = 0,586 c / d``, printed beside element 14 as 1245 Hz.

    Checked against the library's own first cut-on frequency rather than
    against the printed constant recomputed: 0,586 is the guideline's rounding
    of the first circular mode's 1,8412 over pi, so the two agree to the
    resolution the value is printed at and the row exercises the code.
    """
    return numeric(
        1245.0,
        ph.noise_control.plane_wave_limit(
            diameter=_BEND_DIAMETER, speed_of_sound=_EXAMPLE_SPEED
        ),
        0.5,
        unit="Hz",
        places=1,
    )


@register(
    _VDI2081,
    "VDI 2081 Blatt 2:2005 Table 1, element 14",
    "Round bend 160 mm, Table 7 shifted onto its limit frequency, worst octave deviation, dB",
)
def _chk_bend() -> Outcome:
    """Table 7 carried three octaves up, since f_G lands in the 1 kHz octave."""
    values = ph.noise_control.elbow_insertion_loss(
        np.array(_PRINTED_BANDS),
        _BEND_DIAMETER,
        bend_type="round",
        speed_of_sound=_EXAMPLE_SPEED,
        model="vdi2081",
    ).values
    return _worst_band(_PRINTED_BEND, values, 1e-6)


def _junction_outcome(
    fed: float, branches: tuple[float, ...], printed: float
) -> Outcome:
    """One junction of Table 1, by Equation (35)."""
    areas = list(branches)
    computed = ph.noise_control.split_loss(
        sum(areas), areas, branch=areas.index(fed), model="vdi2081"
    )
    return numeric(printed, computed, 0.05, unit="dB", places=3)


def _register_junctions() -> None:
    """Register the three Table 1 junctions, one row each."""
    junctions = (
        (3, 0.30, (0.30, 0.36, 0.42), 5.6),
        (7, 0.049, (0.049, 0.049, 0.049), 4.8),
        (16, 0.020, (0.020, 0.020), 3.0),
    )
    for element, fed, branches, printed in junctions:
        register(
            _VDI2081,
            f"VDI 2081 Blatt 2:2005 Table 1, element {element}",
            f"Junction into {fed:g} m2 of {sum(branches):g} m2 total, dB",
        )(functools.partial(_junction_outcome, fed, branches, printed))


_register_junctions()


# ---------------------------------------------------------------------------
# Table 1 once more: the flow noise each element regenerates
# ---------------------------------------------------------------------------
#: Table 1, element 5: the straight run, whose flow noise the table carries
#: only as the two overall levels of Equations (16) and (17).
_STRAIGHT_AREA = 0.5 * 0.4
_STRAIGHT_VELOCITY = (4200.0 / 3600.0) / _STRAIGHT_AREA
#: Table 1, element 3: the junction. Its approach velocity comes from the whole
#: system's 16 000 m3/h over the 0,90 m2 feeder, not from the branch's own duty.
_JUNCTION_APPROACH = (16000.0 / 3600.0) / 0.90
_JUNCTION_BRANCH = (4200.0 / 3600.0) / 0.30
_PRINTED_JUNCTION_NOISE = (39.1, 33.5, 27.4, 20.7, 13.7, 6.2, -1.5, -9.6)
#: Table 1, element 14: the bend, the same law with one velocity and no
#: rounding correction.
_BEND_AREA = math.pi * 0.08**2
_BEND_VELOCITY = (280.0 / 3600.0) / _BEND_AREA
_PRINTED_BEND_NOISE = (26.9, 23.0, 18.1, 12.5, 6.5, -0.1, -7.0, -14.4)


@register(
    _VDI2081,
    "VDI 2081 Blatt 1:2001 Eq. (16)",
    "Flow noise of a straight duct, overall sound power level, dB",
)
def _chk_straight_flow_noise() -> Outcome:
    """Element 5 of Table 1, printed as 38 dB."""
    overall = ph.noise_control.flow_noise_straight_duct_overall(
        _STRAIGHT_VELOCITY, _STRAIGHT_AREA
    )
    return numeric(38.0, overall, 0.5, unit="dB", places=2)


@register(
    _VDI2081,
    "VDI 2081 Blatt 1:2001 Eq. (17)",
    "Flow noise of a straight duct, A-weighted sound power level, dB",
)
def _chk_straight_flow_noise_a() -> Outcome:
    """The same element, printed as 22 dB."""
    weighted = ph.noise_control.flow_noise_straight_duct_overall(
        _STRAIGHT_VELOCITY, _STRAIGHT_AREA, weighting="A"
    )
    return numeric(22.0, weighted, 0.5, unit="dB", places=2)


def _flow_noise_outcome(index: int) -> Outcome:
    """One octave of element 3's junction flow noise."""
    values = ph.noise_control.flow_noise_bend(
        np.array(_PRINTED_BANDS),
        _JUNCTION_BRANCH,
        0.30,
        0.6,
        model="vdi2081",
        branch_diameter=0.62,
        approach_velocity=_JUNCTION_APPROACH,
        rounding_ratio=0.025,
    ).values
    return numeric(
        _PRINTED_JUNCTION_NOISE[index],
        float(values[index]),
        _PRINTED_TOLERANCE,
        unit="dB",
        places=2,
    )


def _register_junction_flow_noise() -> None:
    """Register the eight octaves of the element 3 flow-noise row."""
    for index, band in enumerate(_PRINTED_BANDS):
        register(
            _VDI2081,
            "VDI 2081 Blatt 2:2005 Table 1, element 3",
            f"Junction flow noise at {band:g} Hz, dB",
        )(functools.partial(_flow_noise_outcome, index))


_register_junction_flow_noise()


@register(
    _VDI2081,
    "VDI 2081 Blatt 2:2005 Table 1, element 14",
    "Bend flow noise, worst octave deviation, dB",
)
def _chk_bend_flow_noise() -> Outcome:
    """A bend is Equation (18) with the two velocities equal and no ``K``."""
    values = ph.noise_control.flow_noise_bend(
        np.array(_PRINTED_BANDS),
        _BEND_VELOCITY,
        _BEND_AREA,
        0.16,
        model="vdi2081",
        branch_diameter=0.160,
    ).values
    return _worst_band(_PRINTED_BEND_NOISE, values, _PRINTED_TOLERANCE)


# ---------------------------------------------------------------------------
# The silencer and the nozzle
# ---------------------------------------------------------------------------
#: Table 1, element 2: a splitter silencer, five 200 mm splitters with 100 mm
#: gaps in a 1500 x 600 mm duct, 2 m long.
_SILENCER_GAP_VELOCITY = 14.81
_SILENCER_PRESSURE_DROP = 145.0
_SILENCER_APPROACH_AREA = 1.5 * 0.6
_SILENCER_GAP = 0.100
_PRINTED_SILENCER_NOISE = (62.7, 58.3, 53.7, 49.4, 45.4, 41.9, 38.6, 35.6)
#: Table 2, element 18: a 200 mm nozzle in a ceiling, the reduction printed
#: both as computed and capped at the 15 dB of Section 6.6.
_NOZZLE_DIAMETER = 0.200
_PRINTED_NOZZLE = (15.8, 10.2, 5.3, 2.1, 0.7, 0.2, 0.1, 0.1)


@register(
    _VDI2081,
    "VDI 2081 Blatt 1:2001 Eq. (49)",
    "Splitter silencer self-noise, A-weighted sound power level, dB",
)
def _chk_silencer_a_weighted() -> Outcome:
    """Element 2 of Table 1, printed as 52 dB.

    Taken from the spectrum the implementation returns rather than by
    re-evaluating Equation (49) here: recomputing it from the same scalars
    would pass whatever the spectrum did, which is the one thing this row is
    for. The A-weighting is the one Table 1 prints in its own header.
    """
    spectrum = ph.noise_control.silencer_self_noise(
        np.array(_PRINTED_BANDS),
        _SILENCER_GAP_VELOCITY,
        5,
        0.6,
        model="vdi2081",
        pressure_drop_pa=_SILENCER_PRESSURE_DROP,
        approach_area=_SILENCER_APPROACH_AREA,
        airway_width=_SILENCER_GAP,
    ).values
    weighted = spectrum + np.array(_PRINTED_A_WEIGHTING)
    total = 10.0 * math.log10(float(np.sum(10.0 ** (weighted / 10.0))))
    return numeric(52.0, total, 0.5, unit="dB", places=3)


def _silencer_outcome(index: int) -> Outcome:
    """One octave of element 2's regenerated noise."""
    values = ph.noise_control.silencer_self_noise(
        np.array(_PRINTED_BANDS),
        _SILENCER_GAP_VELOCITY,
        5,
        0.6,
        model="vdi2081",
        pressure_drop_pa=_SILENCER_PRESSURE_DROP,
        approach_area=_SILENCER_APPROACH_AREA,
        airway_width=_SILENCER_GAP,
    ).values
    return numeric(
        _PRINTED_SILENCER_NOISE[index],
        float(values[index]),
        _PRINTED_TOLERANCE,
        unit="dB",
        places=2,
    )


def _register_silencer_noise() -> None:
    """Register the eight octaves of element 2's flow-noise row."""
    for index, band in enumerate(_PRINTED_BANDS):
        register(
            _VDI2081,
            "VDI 2081 Blatt 2:2005 Table 1, element 2",
            f"Splitter silencer self-noise at {band:g} Hz, dB",
        )(functools.partial(_silencer_outcome, index))


_register_silencer_noise()


@register(
    _VDI2081,
    "VDI 2081 Blatt 2:2005 Table 2, element 18",
    "End reflection of a 200 mm nozzle in a ceiling, worst octave deviation, dB",
)
def _chk_end_reflection() -> Outcome:
    """Figure 28 in closed form, before the 15 dB ceiling of Section 6.6."""
    values = ph.noise_control.end_reflection_loss(
        np.array(_PRINTED_BANDS),
        _NOZZLE_DIAMETER,
        termination="wall",
        method="vdi2081",
        speed_of_sound=_EXAMPLE_SPEED,
        maximum_reduction_db=None,
    ).values
    return _worst_band(_PRINTED_NOZZLE, values, 0.05)


# ---------------------------------------------------------------------------
# The chain, and the curve it is measured against
# ---------------------------------------------------------------------------
#: Section 1.1 of Blatt 2: the spectral assessment correction, dB.
_PRINTED_KA = (21.0, 11.0, 4.0, -2.0, -5.0, -6.0, -6.0, -4.0)
#: Table 1, element 3: the running total after the fan, the splitter silencer
#: and the first junction, which is where three separate models compose.
_PRINTED_AFTER_JUNCTION = (80.8, 68.3, 48.9, 44.9, 40.2, 39.5, 44.0, 39.1)
_SILENCER_ATTENUATION = (6.0, 17.0, 42.0, 41.0, 47.0, 33.0, 20.0, 18.0)


@register(
    _VDI2081,
    "VDI 2081 Blatt 2:2005 Section 1.1",
    "Spectral assessment correction K_A, worst octave deviation, dB",
)
def _chk_assessment_curve() -> Outcome:
    """``K_A = -A - 5``, the inverse A-weighting less the eight-band allowance."""
    return _worst_band(_PRINTED_KA, ph.noise_control.VDI2081_SPECTRAL_CORRECTION, 1e-9)


@register(
    _VDI2081,
    "VDI 2081 Blatt 2:2005 Table 1, elements 1 to 3",
    "Chained level after fan, silencer and junction, worst octave deviation, dB",
)
def _chk_chain() -> Outcome:
    """Three models composed, which is what the worked example exists to check.

    Each piece is pinned against its own printed row elsewhere in this domain;
    this row is the one that fails if they do not compose. The example rounds
    to one decimal at every step and carries the rounded value forward, so the
    running total is held to a tenth.
    """
    bands = np.array(_PRINTED_BANDS)
    fan = _fan()
    fan = fan + (96.0 - 10.0 * math.log10(float(np.sum(10.0 ** (fan / 10.0)))))

    silencer = ph.noise_control.silencer_self_noise(
        bands,
        _SILENCER_GAP_VELOCITY,
        5,
        0.6,
        model="vdi2081",
        pressure_drop_pa=_SILENCER_PRESSURE_DROP,
        approach_area=_SILENCER_APPROACH_AREA,
        airway_width=_SILENCER_GAP,
    ).values
    running = 10.0 * np.log10(
        10.0 ** ((fan - np.array(_SILENCER_ATTENUATION)) / 10.0)
        + 10.0 ** (silencer / 10.0)
    )

    split = ph.noise_control.split_loss(
        0.30 + 0.36 + 0.42, [0.30, 0.36, 0.42], branch=0, model="vdi2081"
    )
    junction = ph.noise_control.flow_noise_bend(
        bands,
        _JUNCTION_BRANCH,
        0.30,
        0.6,
        model="vdi2081",
        branch_diameter=0.62,
        approach_velocity=_JUNCTION_APPROACH,
        rounding_ratio=0.025,
    ).values
    running = 10.0 * np.log10(
        10.0 ** ((running - split) / 10.0) + 10.0 ** (junction / 10.0)
    )

    return _worst_band(_PRINTED_AFTER_JUNCTION, running, 0.1)


# ---------------------------------------------------------------------------
# Table 1 again: the room the supply air finally arrives in
# ---------------------------------------------------------------------------
#: Table 1, element 19: the sound power of the two swirl diffusers together,
#: which is what enters the room, dB re 1e-12 W.
_ENTERING_SOUND_POWER = (53.4, 51.4, 50.1, 41.2, 29.9, 27.8, 33.0, 32.6)
#: Table 1, element 20, row "Richtwirkungsmass (Abstrahlwinkel 0 deg)": the
#: eight values Figure 30 gives for this outlet. The figure itself is a chart
#: the held copy does not resolve to a tenth, so the guideline's own reading of
#: it is the input here, which is what a caller does with a manufacturer's.
_OUTLET_DIRECTIVITY = (2.1, 2.4, 3.0, 4.0, 5.5, 6.7, 7.0, 7.2)
#: Table 1, element 20: the room, and where the listener stands in it.
_ROOM_ABSORPTION_AREA = 20.0
_ROOM_DISTANCE = 1.5
#: Table 1, element 20, row "Raumdaempfung": L_W - L_p, dB, and the single
#: value printed beside it.
_PRINTED_ROOM_ATTENUATION = (5.6, 5.5, 5.1, 4.7, 4.0, 3.6, 3.5, 3.4)
_PRINTED_ROOM_ATTENUATION_SINGLE = 5.7
#: Table 1, element 20, row "Schalldruckpegel": the band levels, printed whole.
_PRINTED_ROOM_LEVELS = (48.0, 46.0, 45.0, 37.0, 26.0, 24.0, 30.0, 29.0)
#: The two summed columns of that row: L_p and L_pA, dB.
_PRINTED_ROOM_TOTAL = 51.4
_PRINTED_ROOM_TOTAL_A = 40.0


def _room_attenuation() -> np.ndarray:
    """Element 20 by Equation (36), from the printed room data alone."""
    return np.asarray(
        ph.noise_control.room_effect(
            _ROOM_DISTANCE,
            absorption_area=_ROOM_ABSORPTION_AREA,
            directivity=np.array(_OUTLET_DIRECTIVITY),
        )
    )


def _room_levels() -> np.ndarray:
    """The band levels at the listener, before the table rounds them whole."""
    return np.asarray(
        np.array(_ENTERING_SOUND_POWER) - _room_attenuation(), dtype=np.float64
    )


@register(
    _VDI2081,
    "VDI 2081 Blatt 2:2005 Table 1, element 20",
    "Room attenuation of a ceiling diffuser, worst octave deviation, dB",
)
def _chk_room_attenuation() -> Outcome:
    """``L_W - L_p = -10 lg[Q/(4 pi r^2) + 4/A]`` over the eight octaves."""
    return _worst_band(
        _PRINTED_ROOM_ATTENUATION, _room_attenuation(), _PRINTED_TOLERANCE
    )


@register(
    _VDI2081,
    "VDI 2081 Blatt 1:2001 Eq. (36)",
    "Room attenuation of a hemispherical outlet, dB",
)
def _chk_room_attenuation_hemispherical() -> Outcome:
    """The single 5,7 dB printed beside the row, which no octave of it equals.

    It is the same room and distance with the directivity a half space gives,
    Q = 2, before Figure 30 makes the factor a function of frequency.
    """
    return numeric(
        _PRINTED_ROOM_ATTENUATION_SINGLE,
        float(
            ph.noise_control.room_effect(
                _ROOM_DISTANCE, absorption_area=_ROOM_ABSORPTION_AREA
            )
        ),
        _PRINTED_TOLERANCE,
        unit="dB",
        places=3,
    )


@register(
    _VDI2081,
    "VDI 2081 Blatt 2:2005 Table 1, element 20",
    "Sound pressure level in room 102, worst octave deviation, dB",
)
def _chk_room_levels() -> Outcome:
    """The row is printed whole, so a band is met when it rounds onto it."""
    return _worst_band(_PRINTED_ROOM_LEVELS, _room_levels(), 0.5)


@register(
    _VDI2081,
    "VDI 2081 Blatt 2:2005 Table 1, element 20",
    "Total sound pressure level in room 102, dB",
)
def _chk_room_total() -> Outcome:
    """The unweighted sum of the eight octaves, printed as 51,4 dB."""
    levels = _room_levels()
    total = 10.0 * math.log10(float(np.sum(10.0 ** (levels / 10.0))))
    return numeric(_PRINTED_ROOM_TOTAL, total, _PRINTED_TOLERANCE, unit="dB", places=3)


@register(
    _VDI2081,
    "VDI 2081 Blatt 2:2005 Table 1, element 20",
    "A-weighted sound pressure level in room 102, dB",
)
def _chk_room_total_a_weighted() -> Outcome:
    """The A-weighted sum, printed as 40,0 dB, against the table's own weighting."""
    levels = _room_levels() + np.array(_PRINTED_A_WEIGHTING)
    total = 10.0 * math.log10(float(np.sum(10.0 ** (levels / 10.0))))
    return numeric(
        _PRINTED_ROOM_TOTAL_A, total, _PRINTED_TOLERANCE, unit="dB", places=3
    )


# ---------------------------------------------------------------------------
# Section 6.3: the reflection at a sudden change of duct section
# ---------------------------------------------------------------------------
#: Figure 26 on printed folio 39 prints the reduction as a closed form rather
#: than as a curve, so the oracle is the expression itself, evaluated at the
#: ratios its own chart is drawn over.
_PRINTED_SECTION_RATIOS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 7.0, 10.0)
#: VDI 3733's recommendation, quoted by 6.3: take no more than this.
_PRINTED_SECTION_CAP = 5.0


def _chk_section_change_ratio(ratio: float) -> Outcome:
    """One abscissa of Figure 26, uncapped, against the printed expression."""
    printed = 10.0 * math.log10((ratio + 1.0) ** 2 / (4.0 * ratio))
    # A reduction reflects in every band, so the first one carries the value
    # whichever way the section goes; the increase branch is checked below.
    # The duct is round and its diameter follows from the area, which is what
    # keeps the 63 Hz octave below the limit frequency at every ratio here.
    computed = float(
        ph.noise_control.section_change_loss(
            np.array(_PRINTED_BANDS),
            ratio,
            1.0,
            shape="round",
            cap=100.0,
        ).values[0]
    )
    return numeric(printed, computed, _PRINTED_TOLERANCE, unit="dB", places=4)


def _register_section_change() -> None:
    """Register the nine abscissae the figure's chart is drawn over."""
    for ratio in _PRINTED_SECTION_RATIOS:
        register(
            _VDI2081,
            "VDI 2081 Blatt 1:2001 Section 6.3, Figure 26",
            f"Reflection at a section change of ratio {ratio:g}, dB",
        )(functools.partial(_chk_section_change_ratio, ratio))


_register_section_change()


@register(
    _VDI2081,
    "VDI 2081 Blatt 1:2001 Section 6.3, Figure 26",
    "Bands a sudden increase still reflects in, of eight",
)
def _chk_section_change_increase_band_count() -> Outcome:
    """Above the limit frequency of the upstream duct the figure prints nought.

    A 0,8 m side puts Equation (33) at 214,4 Hz, which leaves the 63 and
    125 Hz octaves below it and the other six above.
    """
    values = ph.noise_control.section_change_loss(
        np.array(_PRINTED_BANDS),
        0.2,
        0.5,
        shape="rectangular",
        upstream_size=0.8,
    ).values
    return numeric(2.0, float(np.count_nonzero(values)), 0.5, places=1)


@register(
    _VDI2081,
    "VDI 2081 Blatt 1:2001 Section 6.3",
    "Ceiling VDI 3733 recommends for a section change, dB",
)
def _chk_section_change_cap() -> Outcome:
    """However far the sections differ, 6.3 says to take no more than 5 dB."""
    values = ph.noise_control.section_change_loss(
        np.array(_PRINTED_BANDS), 1000.0, 1.0, shape="round"
    ).values
    return numeric(
        _PRINTED_SECTION_CAP,
        float(np.max(values)),
        _PRINTED_TOLERANCE,
        unit="dB",
        places=3,
    )
