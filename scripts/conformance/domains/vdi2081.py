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
