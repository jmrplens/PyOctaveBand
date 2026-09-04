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
            "VDI 2081 Blatt 2 Table 1, element 1",
            f"Supply fan sound power at {band:g} Hz, dB",
        )(functools.partial(_band_outcome, index))


_register_fan_spectrum()


@register(
    _VDI2081,
    "VDI 2081 Blatt 1 Eq. (13)",
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
    "VDI 2081 Blatt 2 Table 1, element 1",
    "Supply fan total sound power level, dB",
)
def _chk_total() -> Outcome:
    """The unweighted sum of the eight octaves, printed as 94,1 dB."""
    total = 10.0 * math.log10(float(np.sum(10.0 ** (_fan() / 10.0))))
    return numeric(94.1, total, _PRINTED_TOLERANCE, unit="dB", places=3)


@register(
    _VDI2081,
    "VDI 2081 Blatt 2 Table 1, element 1",
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
    "VDI 2081 Blatt 2 Table 1, element 5",
    "Rectangular duct 500 x 400 mm over 4 m, level reduction per octave, dB",
)
def _chk_rectangular_run() -> Outcome:
    """Table 5 read by the largest side, which is what puts 250 Hz at 0,3 dB/m."""
    width, height, length = _RECT_RUN
    values = ph.noise_control.unlined_rectangular_duct_attenuation(
        np.array(_PRINTED_BANDS), width, height, length, model="vdi2081"
    ).values
    return numeric(
        float(np.sum(_PRINTED_RECT)),
        float(np.sum(values)),
        1e-6,
        unit="dB",
        places=3,
    )


@register(
    _VDI2081,
    "VDI 2081 Blatt 2 Table 1, element 13",
    "Round duct 160 mm over 1 m, level reduction per octave, dB",
)
def _chk_round_run() -> Outcome:
    """The round rows of Table 5, which do depend on the bore."""
    values = ph.noise_control.unlined_circular_duct_attenuation(
        np.array(_PRINTED_BANDS), 1.000, diameter=0.160, model="vdi2081"
    ).values
    return numeric(
        float(np.sum(_PRINTED_ROUND)),
        float(np.sum(values)),
        1e-6,
        unit="dB",
        places=3,
    )


@register(
    _VDI2081,
    "VDI 2081 Blatt 1 Eq. (34)",
    "Limit frequency of a 160 mm round duct, Hz",
)
def _chk_limit_frequency() -> Outcome:
    """``f_G = 0,586 c / d``, printed beside element 14 as 1245 Hz."""
    return numeric(
        1245.0, 0.586 * _EXAMPLE_SPEED / _BEND_DIAMETER, 0.5, unit="Hz", places=1
    )


@register(
    _VDI2081,
    "VDI 2081 Blatt 2 Table 1, element 14",
    "Round bend 160 mm, Table 7 shifted onto its limit frequency, dB",
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
    return numeric(
        float(np.sum(_PRINTED_BEND)),
        float(np.sum(values)),
        1e-6,
        unit="dB",
        places=3,
    )


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
            f"VDI 2081 Blatt 2 Table 1, element {element}",
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
    "VDI 2081 Blatt 1 Eq. (16)",
    "Flow noise of a straight duct, overall sound power level, dB",
)
def _chk_straight_flow_noise() -> Outcome:
    """Element 5 of Table 1, printed as 38 dB."""
    overall = (
        7.0 + 50.0 * math.log10(_STRAIGHT_VELOCITY) + 10.0 * math.log10(_STRAIGHT_AREA)
    )
    return numeric(38.0, overall, 0.5, unit="dB", places=2)


@register(
    _VDI2081,
    "VDI 2081 Blatt 1 Eq. (17)",
    "Flow noise of a straight duct, A-weighted sound power level, dB",
)
def _chk_straight_flow_noise_a() -> Outcome:
    """The same element, printed as 22 dB."""
    weighted = (
        -25.0
        + 70.0 * math.log10(_STRAIGHT_VELOCITY)
        + 10.0 * math.log10(_STRAIGHT_AREA)
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
            "VDI 2081 Blatt 2 Table 1, element 3",
            f"Junction flow noise at {band:g} Hz, dB",
        )(functools.partial(_flow_noise_outcome, index))


_register_junction_flow_noise()


@register(
    _VDI2081,
    "VDI 2081 Blatt 2 Table 1, element 14",
    "Bend flow noise, sum over the eight octaves, dB",
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
    printed = 10.0 * math.log10(
        float(np.sum(10.0 ** (np.array(_PRINTED_BEND_NOISE) / 10.0)))
    )
    computed = 10.0 * math.log10(float(np.sum(10.0 ** (values / 10.0))))
    return numeric(printed, computed, _PRINTED_TOLERANCE, unit="dB", places=3)


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
    "VDI 2081 Blatt 1 Eq. (49)",
    "Splitter silencer self-noise, A-weighted sound power level, dB",
)
def _chk_silencer_a_weighted() -> Outcome:
    """Element 2 of Table 1, printed as 52 dB."""
    weighted = (
        56.6 * math.log10(_SILENCER_GAP_VELOCITY)
        - 0.5 * math.log10(_SILENCER_PRESSURE_DROP)
        + 10.0 * math.log10(_SILENCER_APPROACH_AREA)
        - 12.7
    )
    return numeric(52.0, weighted, 0.5, unit="dB", places=3)


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
            "VDI 2081 Blatt 2 Table 1, element 2",
            f"Splitter silencer self-noise at {band:g} Hz, dB",
        )(functools.partial(_silencer_outcome, index))


_register_silencer_noise()


@register(
    _VDI2081,
    "VDI 2081 Blatt 2 Table 2, element 18",
    "End reflection of a 200 mm nozzle in a ceiling, sum over the octaves, dB",
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
    return numeric(
        float(np.sum(_PRINTED_NOZZLE)),
        float(np.sum(values)),
        0.05,
        unit="dB",
        places=3,
    )


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
    "VDI 2081 Blatt 2 Section 1.1",
    "Spectral assessment correction K_A, dB per octave",
)
def _chk_assessment_curve() -> Outcome:
    """``K_A = -A - 5``, the inverse A-weighting less the eight-band allowance."""
    computed = ph.noise_control.VDI2081_SPECTRAL_CORRECTION
    return numeric(
        float(np.sum(_PRINTED_KA)),
        float(np.sum(computed)),
        1e-9,
        unit="dB",
        places=3,
    )


@register(
    _VDI2081,
    "VDI 2081 Blatt 2 Table 1, elements 1 to 3",
    "Chained level after fan, silencer and junction, sum over the octaves, dB",
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

    printed = 10.0 * math.log10(
        float(np.sum(10.0 ** (np.array(_PRINTED_AFTER_JUNCTION) / 10.0)))
    )
    computed = 10.0 * math.log10(float(np.sum(10.0 ** (running / 10.0))))
    return numeric(printed, computed, 0.1, unit="dB", places=3)
