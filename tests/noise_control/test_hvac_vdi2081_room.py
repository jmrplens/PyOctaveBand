#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the last step of the VDI 2081 chain: the level in the room.

Oracle: VDI 2081 Part 2:2005-05, Table 1 on printed folio 16, element 20
(``Raum 102``). Every expected value below is a cell of that printed row.

The method is VDI 2081 Part 1:2001-07, Section 6.7.3 on printed folios 43 to
45: Equation (36) for the level, Equation (37) for the equivalent absorption
area, and Figure 30 for the directivity factor of the outlet.

Figure 30 itself is not implemented. It is a chart of the directivity factor
against the product of frequency and the square root of the outlet area, for
two radiation angles and four positions in the room, and the copy in hand is a
scan whose gridlines do not resolve better than about a tenth of the quantity
it carries. The eight values the example reads off it are supplied here as
input, which is what a caller does with a manufacturer's own figure.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phonometry import room
from phonometry.noise_control import hvac

#: Table 1, element 19: the sound power of the two swirl diffusers together,
#: which is what enters the room, dB re 1e-12 W.
ENTERING_SOUND_POWER_DB = np.array([53.4, 51.4, 50.1, 41.2, 29.9, 27.8, 33.0, 32.6])
#: Table 1, element 20, row "Richtwirkungsmaß (Abstrahlwinkel 0 deg)": the
#: eight values Figure 30 gives for this outlet.
DIRECTIVITY = np.array([2.1, 2.4, 3.0, 4.0, 5.5, 6.7, 7.0, 7.2])
#: Table 1, element 20: the room and where the listener stands in it.
ROOM_WIDTH_M = 3.60
ROOM_LENGTH_M = 5.40
ROOM_HEIGHT_M = 2.80
ABSORPTION_AREA_M2 = 20.0
DISTANCE_M = 1.5
#: Table 1, element 20, row "Raumdämpfung": ``L_W - L_p``, dB.
PRINTED_ROOM_ATTENUATION_DB = (5.6, 5.5, 5.1, 4.7, 4.0, 3.6, 3.5, 3.4)
#: The same row's single printed value, dB.
PRINTED_ROOM_ATTENUATION_TOTAL_DB = 5.7
#: Table 1, element 20, row "Schalldruckpegel": the band levels, dB.
PRINTED_LEVELS_DB = (48, 46, 45, 37, 26, 24, 30, 29)
#: Row "A-Korrektur" of the same table, dB.
A_WEIGHTING_DB = np.array([-26.2, -16.1, -8.6, -3.2, 0.0, 1.2, 1.0, -1.1])
#: The two summed columns of the level row: L_p and L_pA, dB.
PRINTED_TOTAL_DB = 51.4
PRINTED_TOTAL_A_DB = 40.0

#: The table prints the room attenuation to one decimal.
TOLERANCE_DB = 0.05


def _room_attenuation() -> np.ndarray:
    """Element 20 by Equation (36), from the printed room data alone."""
    return np.asarray(
        hvac.room_effect(
            DISTANCE_M,
            absorption_area=ABSORPTION_AREA_M2,
            directivity=DIRECTIVITY,
        )
    )


def test_the_room_attenuation_reproduces_every_printed_octave() -> None:
    """``L_W - L_p = -10 lg[Q/(4 pi r^2) + 4/A]``, band by band."""
    assert _room_attenuation() == pytest.approx(
        PRINTED_ROOM_ATTENUATION_DB, abs=TOLERANCE_DB
    )


def test_the_band_levels_round_onto_the_printed_integers() -> None:
    """Element 20 prints its levels whole, so each is met within half a decibel."""
    levels = ENTERING_SOUND_POWER_DB - _room_attenuation()
    assert levels == pytest.approx(PRINTED_LEVELS_DB, abs=0.5)


def test_both_printed_sums_come_out_of_the_band_levels() -> None:
    """51,4 dB and 40,0 dB, from the unrounded bands rather than the printed ones."""
    levels = ENTERING_SOUND_POWER_DB - _room_attenuation()
    total = 10.0 * math.log10(float(np.sum(10.0 ** (levels / 10.0))))
    weighted = 10.0 * math.log10(
        float(np.sum(10.0 ** ((levels + A_WEIGHTING_DB) / 10.0)))
    )
    assert total == pytest.approx(PRINTED_TOTAL_DB, abs=0.05)
    assert weighted == pytest.approx(PRINTED_TOTAL_A_DB, abs=0.05)


def test_the_single_printed_room_attenuation_is_the_hemispherical_one() -> None:
    """Element 20 prints 5,7 dB beside the row, which no octave of it equals.

    The eight-band row runs from 5,6 down to 3,4 dB, so the single value is not
    a band of it, and it is not their energy sum either. It is the attenuation
    the same room and distance give an outlet radiating into a half space,
    ``Q = 2``: the value before Figure 30 makes the directivity a function of
    frequency, and the default this entry point carries.
    """
    hemispherical = float(
        hvac.room_effect(DISTANCE_M, absorption_area=ABSORPTION_AREA_M2)
    )
    assert hemispherical == pytest.approx(
        PRINTED_ROOM_ATTENUATION_TOTAL_DB, abs=TOLERANCE_DB
    )


def test_a_more_directional_outlet_is_louder_at_the_listener() -> None:
    """Q multiplies the direct field, so the attenuation falls as Q rises."""
    values = [
        float(
            hvac.room_effect(
                DISTANCE_M, absorption_area=ABSORPTION_AREA_M2, directivity=q
            )
        )
        for q in (1.0, 2.0, 4.0, 8.0)
    ]
    assert values == sorted(values, reverse=True)


def test_far_from_the_outlet_only_the_absorption_area_is_left() -> None:
    """Past the reverberation radius the direct term dies and Equation (36a) holds."""
    far = float(
        hvac.room_effect(
            50.0, absorption_area=ABSORPTION_AREA_M2, directivity=DIRECTIVITY[-1]
        )
    )
    reverberant = -10.0 * math.log10(4.0 / ABSORPTION_AREA_M2)
    assert far == pytest.approx(reverberant, abs=0.05)


def test_the_two_absorption_measures_are_not_interchangeable() -> None:
    """R and A part company as the room gets deader, so neither stands in.

    Passing the equivalent absorption area where the room constant belongs is
    the one mistake this pair of arguments exists to prevent, and it is silent:
    both are areas in square metres and both are positive.
    """
    alpha = 0.3
    surface = 2.0 * (
        ROOM_WIDTH_M * ROOM_LENGTH_M
        + ROOM_WIDTH_M * ROOM_HEIGHT_M
        + ROOM_LENGTH_M * ROOM_HEIGHT_M
    )
    area = surface * alpha
    constant = float(room.room_constant(surface, alpha))
    assert constant == pytest.approx(area / (1.0 - alpha))
    by_area = float(hvac.room_effect(DISTANCE_M, absorption_area=area))
    by_constant = float(hvac.room_effect(DISTANCE_M, constant))
    assert by_area != pytest.approx(by_constant, abs=0.1)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({}, "got neither"),
        ({"room_constant": 20.0, "absorption_area": 20.0}, "got both"),
        ({"absorption_area": -1.0}, "must be positive"),
        ({"absorption_area": 20.0, "directivity": 0.0}, "must be positive"),
        ({"absorption_area": [[1.0, 2.0]]}, "scalar or a non-empty 1-D array"),
    ],
)
def test_room_effect_refuses_an_ill_posed_room(
    kwargs: dict[str, object], match: str
) -> None:
    """Each guard names the argument it is about."""
    with pytest.raises(ValueError, match=match):
        hvac.room_effect(DISTANCE_M, **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Section 6.7.3 as room acoustics: Equations (37) and the reverberation radius
# ---------------------------------------------------------------------------
def test_equation_37_is_sabine_read_the_other_way_about() -> None:
    """``A = 0,163 V / T``, which is the 0,161 constant at 339 m/s."""
    volume, reverberation = 54.432, 0.5
    area = float(
        room.sabine_absorption_area(volume, reverberation, speed_of_sound=339.0)
    )
    assert area == pytest.approx(0.163 * volume / reverberation, rel=2e-3)
    # And it inverts the forward relation the library already carries, to the
    # width of the two constants: this one is 24 ln 10 = 55,26 and ISO 11690-3
    # Formula 5 rounds the same numerator to 55,3.
    assert float(
        room.reverberation_time(area, volume, speed_of_sound=339.0)
    ) == pytest.approx(reverberation, rel=1e-3)


def test_the_room_of_element_20_wants_about_half_a_second() -> None:
    """The printed A = 20 m2 in a 54,4 m3 room is a reverberation time near 0,44 s.

    Element 20 prints the absorption area and not the time it came from, so
    this is a consistency check on the two rather than an oracle: an office of
    that size damped to 20 m2 is a plausible room, and one damped to 2 m2 would
    not be.
    """
    volume = ROOM_WIDTH_M * ROOM_LENGTH_M * ROOM_HEIGHT_M
    assert volume == pytest.approx(54.432, abs=0.001)
    implied = 0.163 * volume / ABSORPTION_AREA_M2
    assert implied == pytest.approx(0.44, abs=0.01)


def test_the_reverberation_radius_is_the_hemispherical_critical_distance() -> None:
    """``rH = 0,2 sqrt(A)`` is ``sqrt(Q A / 16 pi)`` at Q = 2, not at Q = 1.

    Section 6.7.3 prints the constant as 0,2 and the German column says the
    propagation is *halbkugelförmig*, hemispherical. The English column of the
    same sentence says spherical, which would put the constant at 0,141.
    """
    hemispherical = float(
        room.critical_distance(absorption_area=ABSORPTION_AREA_M2, directivity=2.0)
    )
    assert hemispherical == pytest.approx(0.2 * math.sqrt(ABSORPTION_AREA_M2), rel=3e-3)
    spherical = float(
        room.critical_distance(absorption_area=ABSORPTION_AREA_M2, directivity=1.0)
    )
    assert spherical == pytest.approx(0.141 * math.sqrt(ABSORPTION_AREA_M2), rel=3e-3)


def test_the_listener_of_element_20_stands_beyond_the_reverberation_radius() -> None:
    """1,5 m against an rH of 0,9 m, so the printed levels are reverberant-leaning."""
    radius = float(
        room.critical_distance(absorption_area=ABSORPTION_AREA_M2, directivity=2.0)
    )
    assert radius < DISTANCE_M


def test_steady_state_spl_takes_either_measure_and_a_per_band_directivity() -> None:
    """The room package's own entry point reaches the same printed levels."""
    levels = room.steady_state_spl(
        ENTERING_SOUND_POWER_DB,
        DISTANCE_M,
        absorption_area=ABSORPTION_AREA_M2,
        directivity=DIRECTIVITY,
    )
    assert levels == pytest.approx(
        ENTERING_SOUND_POWER_DB - np.array(PRINTED_ROOM_ATTENUATION_DB),
        abs=TOLERANCE_DB,
    )


def test_steady_state_spl_refuses_both_measures_at_once() -> None:
    """Exactly one, and the message says which two it is choosing between."""
    with pytest.raises(ValueError, match="got both"):
        room.steady_state_spl(
            50.0,
            1.0,
            20.0,
            absorption_area=20.0,  # type: ignore[call-overload]
        )


def test_an_empty_spectrum_is_refused_rather_than_answered() -> None:
    """Every range guard is vacuously true over nothing, so the size is checked.

    ``np.any(arr <= 0)`` is false over an empty array and ``np.all(isfinite)``
    is true, so an empty absorption measure passed every check and handed back
    an empty level spectrum: a room with no bands rather than the mistake it
    is.
    """
    for call in (
        lambda: room.steady_state_spl(50.0, 1.0, absorption_area=[]),
        lambda: room.critical_distance(absorption_area=[]),
        lambda: room.sabine_absorption_area(50.0, []),
        lambda: hvac.room_effect(1.5, absorption_area=[]),
    ):
        with pytest.raises(ValueError, match="non-empty 1-D array"):
            call()
