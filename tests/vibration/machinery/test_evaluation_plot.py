#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The polar reading of a vector change (ISO 20816-1, Annex D, Figure D.1).

Looking at the figure is not covering it: these read the artists back, so a
chord drawn as an arc, a legend that lost a vector or a title that dropped its
unit fails here rather than in a visual pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from phonometry import vibration


@pytest.fixture(autouse=True)
def _agg() -> None:
    """Every case draws, and none of them wants a window."""
    pytest.importorskip("matplotlib")
    import matplotlib as mpl

    mpl.use("Agg")


def _annex_d() -> vibration.VectorChangeResult:
    """The worked case of D.2: 3 mm/s at 40 degrees to 2,5 at 180."""
    return vibration.vibration_vector_change(3.0, 40.0, 2.5, 180.0)


def test_three_artists_and_their_labels() -> None:
    ax = _annex_d().plot(unit="mm/s")
    labels = [line.get_label() for line in ax.get_lines()]
    assert labels == ["initial $A_1$", "final $A_2$", "change $A_2 - A_1$"]
    assert ax.name == "polar"


def test_the_two_phasors_run_from_the_origin_to_the_states() -> None:
    ax = _annex_d().plot()
    initial, final = ax.get_lines()[0], ax.get_lines()[1]
    for line, magnitude, phase in ((initial, 3.0, 40.0), (final, 2.5, 180.0)):
        theta, radius = line.get_xdata(), line.get_ydata()
        assert radius[0] == pytest.approx(0.0)
        assert radius[1] == pytest.approx(magnitude)
        assert np.degrees(theta[1]) % 360.0 == pytest.approx(phase)


def test_the_change_is_a_straight_chord_not_an_arc() -> None:
    """A polar axes bows a two-point line, which would read as a path."""
    ax = _annex_d().plot()
    theta, radius = ax.get_lines()[2].get_xdata(), ax.get_lines()[2].get_ydata()
    assert len(theta) > 2
    x, y = radius * np.cos(theta), radius * np.sin(theta)
    # Every sample sits on the segment joining the two tips, so the length of
    # the sampled path is the length of the chord itself.
    walked = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
    assert walked == pytest.approx(_annex_d().magnitude, rel=1e-9)


def test_the_radial_labels_are_parked_off_every_drawn_line() -> None:
    ax = _annex_d().plot()
    parked = float(ax.get_rlabel_position()) % 360.0
    drawn = [
        40.0,
        180.0,
        np.degrees(np.angle(-2.5 - 3.0 * np.exp(1j * np.radians(40.0)))) % 360.0,
    ]
    assert min(abs((parked - angle + 180.0) % 360.0 - 180.0) for angle in drawn) > 20.0


def test_the_title_carries_both_readings_and_the_unit() -> None:
    result = _annex_d()
    assert result.plot(unit="mm/s").get_title() == (
        "Change in vibration: magnitude \u22120.5 mm/s, vector 5.17 mm/s"
    )
    # Without a unit the numbers stay bare: the result cannot know it.
    assert result.plot().get_title() == (
        "Change in vibration: magnitude \u22120.5, vector 5.17"
    )


def test_spanish_labels_and_title() -> None:
    ax = _annex_d().plot(language="es", unit="mm/s")
    labels = [line.get_label() for line in ax.get_lines()]
    assert labels == ["inicial $A_1$", "final $A_2$", "cambio $A_2 - A_1$"]
    assert ax.get_title() == (
        "Cambio de vibración: magnitud \u22120,5 mm/s, vector 5,17 mm/s"
    )


def test_the_chord_takes_the_style_the_caller_passes() -> None:
    ax = _annex_d().plot(color="#123456", linestyle=":")
    chord = ax.get_lines()[2]
    assert chord.get_color() == "#123456"
    assert chord.get_linestyle() == ":"


def test_it_draws_on_the_polar_axes_it_is_given() -> None:
    import matplotlib.pyplot as plt

    _fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    assert _annex_d().plot(ax=ax) is ax
    assert len(ax.get_lines()) == 3


def test_a_cartesian_axes_is_refused() -> None:
    import matplotlib.pyplot as plt

    _fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="must be a polar axes"):
        _annex_d().plot(ax=ax)
