#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the restaurant crowd self-noise model (Long Chapter 17).

Oracle: **Long, Architectural Acoustics 2nd ed., printed pp. 665-666**, the
"Restaurant Design" worked case, Equations (17.50) to (17.54). Long states, for
a hard room with a total absorption of about 20 metric sabins, a talker with
Lw = 70 dB and a listener at 1.2 m:

* a direct field of 60 dB at 1.2 m;
* a self-generated reverberant noise of 63 dB with one talker, hence a
  signal-to-noise ratio of -3 dB and 75 % intelligibility;
* 76 dB with 20 tables each with one talker;
* an absorptive ceiling of alpha 0.9 over a 13.7 x 13.7 m room adding
  170 metric sabins, which drops the 20-table level to 66 dB;
* a talker-to-listener distance of 1 m requiring "at least 6.3 or more square
  metres (68 sq ft) of absorption per table" (Equation (17.53));
* at a 2.5 m table spacing, "no more than 20 m2 (215 ft2) per table"
  (Equation (17.54)).

Two printed statements are *not* reproduced by the equations and are recorded
here rather than asserted:

* the direct field at an adjacent table 3 m away, printed as "about 54 dB";
  Equation (17.50) with the same Q = 2 that yields the 60 dB at 1.2 m gives
  52.5 dB (the pure 20 lg(3/1.2) = 8.0 dB distance drop from 60.4 dB);
* the constant of Equation (17.53), printed as 6.33; the closed form that
  yields Long's own 3.16 of Equation (17.54) gives 6.31, which is also what
  his conversion "6.3 or more square metres (68 sq ft)" implies
  (68 sq ft = 6.317 m2).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phonometry import room

#: Long, printed p. 665: the hard restaurant's total absorption, metric sabins.
LONG_HARD_ABSORPTION = 20.0

#: Long, printed p. 665: the treated room, 13.7 x 13.7 m with an alpha 0.9
#: ceiling, "adds 170 metric sabins".
LONG_TREATED_ROOM_SIDE = 13.7
LONG_CEILING_ALPHA = 0.9


def test_long_direct_field_60_db_at_1_2_m() -> None:
    """Long p. 665: Lw = 70 dB gives "a direct field of 60 dB at 1.2 m"."""
    lp = float(np.asarray(room.speech_direct_level(1.2))[()])
    assert round(lp) == 60


def test_long_self_noise_63_db_for_one_talker() -> None:
    """Long p. 665: "With 20 metric sabins, our self-generated reverberant-field
    noise is 63 dB"."""
    lp = float(np.asarray(room.crowd_noise_level(1, LONG_HARD_ABSORPTION))[()])
    assert round(lp) == 63


def test_long_signal_to_noise_is_minus_3_db() -> None:
    """Long p. 665: "our signal-to-noise ratio is -3 dB"."""
    signal = float(np.asarray(room.speech_direct_level(1.2))[()])
    noise = float(
        np.asarray(room.crowd_noise_level(1, LONG_HARD_ABSORPTION))[()]
    )
    assert round(signal - noise) == -3


def test_long_twenty_talkers_reach_76_db() -> None:
    """Long p. 665: "If there are 20 tables in the room ... the reverberant
    noise level rises to 76 dB"."""
    lp = float(
        np.asarray(room.crowd_noise_level(20, LONG_HARD_ABSORPTION))[()]
    )
    assert round(lp) == 76


def test_long_absorptive_ceiling_adds_170_sabins() -> None:
    """Long p. 665: an alpha 0.9 ceiling over 13.7 x 13.7 m "adds 170 metric
    sabins"."""
    added = LONG_CEILING_ALPHA * LONG_TREATED_ROOM_SIDE**2
    assert round(added, -1) == 170.0


def test_long_treated_room_drops_the_20_table_level_to_66_db() -> None:
    """Long p. 665: "The 20 table reverberant noise level drops to 66 dB"."""
    total = LONG_HARD_ABSORPTION + 170.0
    lp = float(np.asarray(room.crowd_noise_level(20, total))[()])
    assert round(lp) == 66


def test_long_communication_bound_is_6_3_square_metres_at_1_m() -> None:
    """Long p. 666: at 1 m, "at least 6.3 or more square metres (68 sq ft)"."""
    a_tab = float(
        np.asarray(room.absorption_per_table(1.0, room.COMMUNICATION_SNR))[()]
    )
    assert round(a_tab, 1) == 6.3
    # 68 sq ft, Long's own conversion of the same number.
    assert a_tab == pytest.approx(68.0 * 0.09290304, abs=0.01)


def test_long_privacy_bound_is_20_square_metres_at_2_5_m() -> None:
    """Long p. 666: at 2.5 m spacing, "no more than 20 m2 (215 ft2) per table"."""
    a_tab = float(
        np.asarray(room.absorption_per_table(2.5, room.PRIVACY_SNR))[()]
    )
    assert round(a_tab, -1) == 20.0
    # 215 sq ft is Long's conversion of the rounded 20 m2, so it lands 0.2 m2
    # above the closed form.
    assert a_tab == pytest.approx(215.0 * 0.09290304, abs=0.25)


def test_long_privacy_constant_is_3_16() -> None:
    """Long Equation (17.54): A_tab < 3.16 rt^2 (printed constant, Q = 2)."""
    for rt in (1.0, 2.0, 3.5):
        a_tab = float(
            np.asarray(room.absorption_per_table(rt, room.PRIVACY_SNR))[()]
        )
        assert a_tab / rt**2 == pytest.approx(3.16, abs=0.005)


def test_communication_constant_is_6_31_not_the_printed_6_33() -> None:
    """The Equation (17.53) constant follows from Equation (17.52) exactly.

    ``16 pi 10^(-0.6) / Q`` with ``Q = 2`` is 6.3131, whereas Long prints
    6.33; the same closed form reproduces his 3.16 of Equation (17.54)
    exactly, and his own "68 sq ft" is 6.317 m2, so 6.33 is a slip in the
    printed equation.
    """
    expected = 16.0 * math.pi * 10.0 ** (-0.6) / room.TALKER_DIRECTIVITY
    a_tab = float(
        np.asarray(room.absorption_per_table(1.0, room.COMMUNICATION_SNR))[()]
    )
    assert a_tab == pytest.approx(expected)
    assert a_tab == pytest.approx(6.3131, abs=1e-4)


def test_speech_to_noise_is_independent_of_power_and_occupancy() -> None:
    """Equation (17.52) drops both Lw and N: a room fills with its absorption.

    The difference of Equations (17.50) and (17.51) with ``A = N A_tab``.
    """
    a_tab, r = 8.0, 1.2
    direct = float(np.asarray(room.speech_direct_level(r))[()])
    for n in (1.0, 5.0, 20.0):
        noise = float(np.asarray(room.crowd_noise_level(n, n * a_tab))[()])
        assert direct - noise == pytest.approx(
            float(np.asarray(room.speech_to_noise_ratio(r, a_tab))[()])
        )


def test_absorption_per_table_inverts_the_ratio() -> None:
    for snr in (-3.0, -6.0, -9.0, -12.0):
        a_tab = float(np.asarray(room.absorption_per_table(1.7, snr))[()])
        assert float(
            np.asarray(room.speech_to_noise_ratio(1.7, a_tab))[()]
        ) == (pytest.approx(snr))


def test_limit_of_a_perfectly_absorptive_room() -> None:
    """As the absorption grows without bound the self-noise vanishes.

    The reverberant term is ``10 lg(4/A)``, so doubling A costs 3 dB and the
    level falls monotonically toward minus infinity.
    """
    levels = np.asarray(
        room.crowd_noise_level(20, np.array([100.0, 200.0, 400.0, 1e12]))
    )
    assert np.all(np.diff(levels) < 0.0)
    assert levels[1] == pytest.approx(levels[0] - 10.0 * math.log10(2.0))
    assert levels[-1] < 0.0


def test_empty_room_is_the_single_talker_case() -> None:
    """N = 1 is the lowest occupancy the model admits, and N talkers add 10 lg N."""
    one = float(np.asarray(room.crowd_noise_level(1, 50.0))[()])
    ten = float(np.asarray(room.crowd_noise_level(10, 50.0))[()])
    assert ten - one == pytest.approx(10.0)


def test_crowd_noise_result_shape_and_fields() -> None:
    result = room.crowd_noise([20.0, 190.0], talkers=[1.0, 20.0])
    assert isinstance(result, room.CrowdNoiseResult)
    assert result.levels.shape == (2, 2)
    assert round(float(result.levels[0, 0])) == 63
    assert round(float(result.levels[0, 1])) == 76
    assert round(float(result.levels[1, 1])) == 66
    assert round(result.signal_level) == 60
    assert result.sound_power_level == room.NORMAL_VOICE_POWER_LEVEL
    assert result.directivity == room.TALKER_DIRECTIVITY
    # Above this noise level the -6 dB communication limit is breached.
    assert result.communication_level == pytest.approx(
        result.signal_level - room.COMMUNICATION_SNR
    )
    assert np.allclose(result.speech_to_noise(), result.signal_level - result.levels)


def test_crowd_noise_default_occupancy_axis() -> None:
    result = room.crowd_noise(20.0)
    assert np.array_equal(result.talkers, np.arange(1.0, 21.0))
    assert result.levels.shape == (1, 20)


def test_plot_draws_one_curve_per_absorption_plus_two_references() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = room.crowd_noise([20.0, 90.0, 190.0])
    ax = result.plot()
    # Three absorption curves plus the speech level and the -6 dB limit.
    assert len(ax.lines) == 5
    curve = ax.lines[0]
    assert np.allclose(curve.get_ydata(), result.levels[0])
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("Communication limit" in label for label in labels)
    assert "Simultaneous talkers" in ax.get_xlabel()
    plt.close("all")


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: room.crowd_noise_level(0, 20.0), "at least 1"),
        (lambda: room.crowd_noise_level(1, 0.0), "positive and finite"),
        (lambda: room.speech_direct_level(0.0), "positive and finite"),
        (
            lambda: room.speech_direct_level(1.0, directivity=0.0),
            "directivity",
        ),
        (lambda: room.speech_to_noise_ratio(1.0, -1.0), "positive and finite"),
        (lambda: room.absorption_per_table(1.0, math.nan), "finite"),
        (lambda: room.crowd_noise([]), "non-empty"),
        (lambda: room.crowd_noise(20.0, talkers=[0.5]), "at least 1"),
        (lambda: room.crowd_noise(-1.0), "positive and finite"),
    ],
)
def test_validation_errors(call: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()  # type: ignore[operator]
