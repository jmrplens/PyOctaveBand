#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the gain before feedback of a reinforcement system.

Oracle: **Long, Architectural Acoustics 2nd ed., printed pp. 695-699**,
Equations (18.13) to (18.24). Long prints no fully numeric worked case for this
chain, so this module is *closed-form anchored*: the tests reproduce the
equations themselves and every special case Long states in words.

* Equation (18.16): sustained oscillation at ``Z_S + G_S = 0``.
* Equation (18.20): stable when ``Z_S + L_H-M <= L_H-L - D_M(theta) - 10``,
  with the 10 dB feedback stability margin Long adopts for an equalised system
  ("6 dB in-phase reinforcement plus a 4 dB safety factor").
* Equation (18.21): with an open-loop gain of -6 dB the criterion collapses to
  ``L_H-M <= L_H-L - D_M(theta) - 4``, so an omnidirectional microphone must
  see the loudspeaker 4 dB below the average level in the audience.
* Equation (18.22): with a cardioid microphone (-2 dB of relative directivity)
  the limit becomes ``L_H-M <= L_H-L - 2``.
* Equation (18.23): the number-of-open-microphones correction
  ``dL_nom = 10 log10 N_m``.

Long prints Equation (18.24) with ``+ D_M(theta)`` on the right, which
contradicts Equations (18.20) to (18.22); the implementation follows
Equation (18.20), and the test below pins that choice by reproducing the two
special cases with ``N_m = 1``.
"""

from __future__ import annotations

import math

import pytest

from phonometry import electroacoustics

#: Long, printed p. 697: "the desired open-loop system gain might be of the
#: order of -6 dB", typical of an auditorium or church.
LONG_OPEN_LOOP_GAIN = -6.0

#: An arbitrary but fixed audience level for the special-case checks, dB.
LEVEL_AT_LISTENER = 80.0


def test_stability_margin_default_is_10_db() -> None:
    """Long p. 698: "This text uses a middle value of 10 dB for an equalised
    system" (12 dB unequalised, 6 dB carefully equalised).
    """
    assert electroacoustics.DEFAULT_STABILITY_MARGIN == 10.0


def test_feedback_loop_gain_is_equation_18_18() -> None:
    """G_S = L_H-M - L_H-L + D_M(theta)."""
    assert electroacoustics.feedback_loop_gain(74.0, 80.0) == pytest.approx(-6.0)
    assert electroacoustics.feedback_loop_gain(
        74.0, 80.0, microphone_directivity=-2.0
    ) == (pytest.approx(-8.0))


def test_oscillation_criterion_equation_18_16() -> None:
    """Equation (18.16): the loop oscillates when Z_S + G_S = 0.

    Choosing L_H-M so that G_S = -Z_S puts the loop gain exactly at 0 dB: no
    margin at all, and the system is unstable for any positive margin.
    """
    l_hm = LEVEL_AT_LISTENER + LONG_OPEN_LOOP_GAIN
    result = electroacoustics.feedback_stability(
        -electroacoustics.feedback_loop_gain(l_hm, LEVEL_AT_LISTENER),
        l_hm,
        LEVEL_AT_LISTENER,
    )
    assert result.loop_gain == pytest.approx(0.0)
    assert result.margin == pytest.approx(0.0)
    assert not result.is_stable
    # With a zero margin required, sitting exactly at oscillation is the edge.
    edge = electroacoustics.feedback_stability(
        -electroacoustics.feedback_loop_gain(l_hm, LEVEL_AT_LISTENER),
        l_hm,
        LEVEL_AT_LISTENER,
        stability_margin=0.0,
    )
    assert edge.is_stable


def test_equation_18_21_omnidirectional_microphone_4_db_below() -> None:
    """Equation (18.21): with Z_S = -6 dB, L_H-M <= L_H-L - D_M - 4.

    For an omnidirectional microphone (D_M = 0) the loudspeaker level at the
    microphone must be 4 dB below the average level in the audience.
    """
    result = electroacoustics.feedback_stability(
        LONG_OPEN_LOOP_GAIN, LEVEL_AT_LISTENER - 4.0, LEVEL_AT_LISTENER
    )
    assert result.maximum_level_at_microphone == pytest.approx(LEVEL_AT_LISTENER - 4.0)
    assert result.is_stable
    assert result.headroom == pytest.approx(0.0)
    # One decibel more at the microphone and the criterion fails.
    louder = electroacoustics.feedback_stability(
        LONG_OPEN_LOOP_GAIN, LEVEL_AT_LISTENER - 3.0, LEVEL_AT_LISTENER
    )
    assert not louder.is_stable
    assert louder.headroom == pytest.approx(-1.0)


def test_equation_18_22_cardioid_microphone_2_db_below() -> None:
    """Equation (18.22): a cardioid (-2 dB relative) allows L_H-M <= L_H-L - 2."""
    result = electroacoustics.feedback_stability(
        LONG_OPEN_LOOP_GAIN,
        LEVEL_AT_LISTENER - 2.0,
        LEVEL_AT_LISTENER,
        microphone_directivity=electroacoustics.CARDIOID_RELATIVE_DIRECTIVITY,
    )
    assert result.maximum_level_at_microphone == pytest.approx(LEVEL_AT_LISTENER - 2.0)
    assert result.is_stable
    assert result.headroom == pytest.approx(0.0)


def test_cardioid_buys_exactly_its_directivity_index() -> None:
    """Long p. 698: a directional microphone buys gain equal to -D_M(theta)."""
    omni = electroacoustics.feedback_stability(
        LONG_OPEN_LOOP_GAIN, 74.0, LEVEL_AT_LISTENER
    )
    cardioid = electroacoustics.feedback_stability(
        LONG_OPEN_LOOP_GAIN,
        74.0,
        LEVEL_AT_LISTENER,
        microphone_directivity=-3.0,
    )
    assert cardioid.headroom - omni.headroom == pytest.approx(3.0)


def test_equation_18_23_number_of_open_microphones() -> None:
    """dL_nom = 10 log10 N_m: doubling the open microphones costs 3 dB."""
    assert electroacoustics.open_microphone_correction(1) == 0.0
    assert electroacoustics.open_microphone_correction(2) == pytest.approx(
        3.0103, abs=1e-4
    )
    assert electroacoustics.open_microphone_correction(4) == pytest.approx(
        6.0206, abs=1e-4
    )
    assert electroacoustics.open_microphone_correction(10) == pytest.approx(10.0)


def test_open_microphones_eat_the_headroom() -> None:
    """Equation (18.24): dL_nom enters the criterion on the loop-gain side."""
    one = electroacoustics.feedback_stability(
        LONG_OPEN_LOOP_GAIN, 74.0, LEVEL_AT_LISTENER
    )
    four = electroacoustics.feedback_stability(
        LONG_OPEN_LOOP_GAIN, 74.0, LEVEL_AT_LISTENER, open_microphones=4
    )
    assert four.nom_correction == pytest.approx(10.0 * math.log10(4.0))
    assert one.headroom - four.headroom == pytest.approx(four.nom_correction)
    assert four.maximum_open_loop_gain == pytest.approx(
        one.maximum_open_loop_gain - four.nom_correction
    )


def test_maximum_open_loop_gain_is_the_stability_edge() -> None:
    """Setting Z_S to its maximum lands the loop exactly on the margin line."""
    probe = electroacoustics.feedback_stability(
        0.0, 74.0, LEVEL_AT_LISTENER, open_microphones=3
    )
    at_limit = electroacoustics.feedback_stability(
        probe.maximum_open_loop_gain,
        74.0,
        LEVEL_AT_LISTENER,
        open_microphones=3,
    )
    assert at_limit.loop_gain == pytest.approx(-at_limit.stability_margin)
    assert at_limit.is_stable
    assert at_limit.headroom == pytest.approx(0.0)


def test_loop_gain_is_the_sum_of_its_parts() -> None:
    result = electroacoustics.feedback_stability(
        -4.0, 70.0, 78.0, microphone_directivity=-2.5, open_microphones=6
    )
    assert result.loop_gain == pytest.approx(
        result.open_loop_gain + result.feedback_loop_gain + result.nom_correction
    )
    assert result.margin == pytest.approx(-result.loop_gain)
    assert result.headroom == pytest.approx(-result.stability_margin - result.loop_gain)
    assert isinstance(result, electroacoustics.FeedbackStabilityResult)


def test_result_is_frozen() -> None:
    result = electroacoustics.feedback_stability(-6.0, 74.0, 80.0)
    with pytest.raises(AttributeError):
        result.open_loop_gain = 0.0  # type: ignore[misc]


def test_plot_bars_echo_the_gain_structure() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = electroacoustics.feedback_stability(-6.0, 74.0, 80.0, open_microphones=2)
    ax = result.plot()
    heights = [patch.get_height() for patch in ax.patches]
    assert heights == pytest.approx(
        [
            result.open_loop_gain,
            result.feedback_loop_gain,
            result.nom_correction,
            result.loop_gain,
        ]
    )
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("Oscillation" in label for label in labels)
    assert any("Stability limit" in label for label in labels)
    assert "stable" in ax.get_title()
    plt.close("all")


def test_geometry_drawing_annotates_every_path() -> None:
    """The schematic is not to scale, so each path carries its own length.

    A talker 0.3 m from the microphone and a listener 12 m from the
    loudspeaker cannot share one usable drawing scale, so the renderer places
    the four points schematically and writes the real distance beside each
    path. The test checks that contract: three paths, three lengths.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ax = electroacoustics.plot_sound_reinforcement_geometry(0.3, 4.0, 12.0)
    texts = [t.get_text() for t in ax.texts]
    assert "0.3 m" in texts
    assert "4 m" in texts
    assert "12 m" in texts
    # The talker, microphone, loudspeaker and listener are all named.
    for name in ("Talker (T)", "Microphone (M)", "Loudspeaker (H)", "Listener (L)"):
        assert name in texts
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert {"Signal path", "Feedback path"} <= set(labels)
    assert "feedback loop" in ax.get_title()
    plt.close("all")

    with pytest.raises(ValueError, match="must be positive"):
        electroacoustics.plot_sound_reinforcement_geometry(0.0, 4.0, 12.0)
    with pytest.raises(ValueError, match="must be positive"):
        electroacoustics.plot_sound_reinforcement_geometry(0.3, math.inf, 12.0)


@pytest.mark.parametrize("bad", [0, -1, 2.5])
def test_open_microphone_validation(bad: float) -> None:
    with pytest.raises(ValueError, match="integer of at least 1"):
        electroacoustics.open_microphone_correction(bad)  # type: ignore[arg-type]


def test_validation_errors() -> None:
    with pytest.raises(ValueError, match="finite"):
        electroacoustics.feedback_stability(math.nan, 74.0, 80.0)
    with pytest.raises(ValueError, match="finite"):
        electroacoustics.feedback_stability(-6.0, math.inf, 80.0)
    with pytest.raises(ValueError, match="non-negative"):
        electroacoustics.feedback_stability(-6.0, 74.0, 80.0, stability_margin=-1.0)
    with pytest.raises(ValueError, match="integer of at least 1"):
        electroacoustics.feedback_stability(-6.0, 74.0, 80.0, open_microphones=0)
