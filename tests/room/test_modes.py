#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the normal modes of a rectangular room (Long Chapter 8).

Oracles, all independent of this implementation:

* **Long, Architectural Acoustics 2nd ed., Table 8.1** (printed p. 325, room
  7 x 5 x 3 m): the first six modes, their orders and their frequencies as
  printed, 24.6 / 34.5 / 42.4 / 49.2 / 57.4 / 60.1 Hz.
* **Long, printed p. 326**: "For the room dimensions given in Fig. 8.8, at
  1000 Hz, the modal density is 34 modes per Hertz" (Equation (8.46)).
* the closed forms of Equations (8.43), (8.45) and (8.46) themselves, which
  are exact, plus the degeneracy of a cubic room and the consistency of the
  smooth count with the exact enumeration at high frequency.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from phonometry import room

#: Long's example room, printed p. 324 (7 x 5 x 3 m high).
LONG_ROOM = (7.0, 5.0, 3.0)

#: Speed of sound of Long's Table 2.1 for air at 20 degC, printed p. 59.
LONG_C0 = 344.0

#: Long, Table 8.1 (printed p. 325): (nx, ny, nz) -> f_n [Hz], in the printed
#: order (the table is sorted by frequency).
LONG_TABLE_8_1 = (
    ((1, 0, 0), 24.6),
    ((0, 1, 0), 34.5),
    ((1, 1, 0), 42.4),
    ((2, 0, 0), 49.2),
    ((0, 0, 1), 57.4),
    ((2, 1, 0), 60.1),
)


def test_long_table_8_1_orders_and_frequencies() -> None:
    """Long Table 8.1: the first six modes of the 7 x 5 x 3 m room.

    The orders and their ordering must match digit for digit. The printed
    frequencies are reproduced to better than 0.13 Hz: Long's Chapter 2 quotes
    c0 = 344 m/s for air at 20 degC (Table 2.1, printed p. 59), while the table
    itself is consistent with c0 = 344.7 m/s, so each printed value sits
    slightly above the c0 = 344 m/s closed form (see the next test).
    """
    result = room.room_modes(LONG_ROOM, max_frequency=61.0, speed_of_sound=LONG_C0)
    assert len(result.frequencies) == len(LONG_TABLE_8_1)
    for row, (orders, _) in zip(result.orders, LONG_TABLE_8_1, strict=True):
        assert tuple(int(v) for v in row) == orders
    for computed, (_, printed) in zip(result.frequencies, LONG_TABLE_8_1, strict=True):
        assert float(computed) == pytest.approx(printed, abs=0.13)


def test_long_table_8_1_is_reproduced_exactly_at_344_7() -> None:
    """The printed table rounds exactly with c0 = 344.7 m/s.

    A one-parameter check that the discrepancy of the previous test is Long's
    speed of sound and nothing else: at 344.7 m/s every printed value of
    Table 8.1 is recovered to its printed precision.
    """
    for orders, printed in LONG_TABLE_8_1:
        f = room.room_mode_frequency(orders, LONG_ROOM, speed_of_sound=344.7)
        assert float(f) == pytest.approx(printed, abs=0.05)


def test_mode_frequency_closed_form() -> None:
    """Equation (8.43) for the axial modes is c0 n / (2 l), exactly."""
    for axis, length in enumerate(LONG_ROOM):
        orders = [0, 0, 0]
        orders[axis] = 3
        f = room.room_mode_frequency(orders, LONG_ROOM, speed_of_sound=LONG_C0)
        assert float(f) == pytest.approx(3.0 * LONG_C0 / (2.0 * length))


def test_mode_frequency_accepts_an_array_of_triples() -> None:
    orders = np.array([[1, 0, 0], [0, 1, 0]])
    f = np.asarray(room.room_mode_frequency(orders, LONG_ROOM, speed_of_sound=LONG_C0))
    assert f.shape == (2,)
    assert f[0] == pytest.approx(LONG_C0 / 14.0)
    assert f[1] == pytest.approx(LONG_C0 / 10.0)


def test_classification_counts_non_zero_orders() -> None:
    result = room.room_modes(LONG_ROOM, max_frequency=200.0, speed_of_sound=LONG_C0)
    for orders, kind in zip(result.orders, result.kinds, strict=True):
        assert kind == room.MODE_KINDS[int(np.count_nonzero(orders)) - 1]
    counts = result.count_by_kind()
    assert sum(counts.values()) == len(result.frequencies)
    # A room this size has all three families below 200 Hz.
    assert all(counts[kind] > 0 for kind in room.MODE_KINDS)


def test_cubic_room_modes_are_degenerate() -> None:
    """In a cube the three axial fundamentals coincide, and so do permutations.

    A 4 m cube has f(1,0,0) = f(0,1,0) = f(0,0,1) = c0/8, the three
    permutations of (1,1,0) coincide at c0 sqrt(2)/8, and (1,1,1) sits at
    c0 sqrt(3)/8: the coalescing that Long warns about for integer-ratio rooms.
    """
    c0 = 344.0
    result = room.room_modes((4.0, 4.0, 4.0), max_frequency=80.0, speed_of_sound=c0)
    freqs = np.asarray(result.frequencies)
    axial = freqs[np.asarray(result.kinds) == "axial"]
    assert np.count_nonzero(np.isclose(axial, c0 / 8.0)) == 3
    tangential = freqs[np.asarray(result.kinds) == "tangential"]
    assert np.count_nonzero(np.isclose(tangential, c0 * math.sqrt(2.0) / 8.0)) == 3
    oblique = freqs[np.asarray(result.kinds) == "oblique"]
    assert oblique[0] == pytest.approx(c0 * math.sqrt(3.0) / 8.0)
    # The listing stays sorted despite the exact ties.
    assert np.all(np.diff(freqs) >= 0.0)


def test_long_modal_density_34_modes_per_hertz_at_1_khz() -> None:
    """Long, printed p. 326: 34 modes per hertz at 1 kHz for the 7 x 5 x 3 m room."""
    density = room.room_modal_density(1000.0, LONG_ROOM, speed_of_sound=LONG_C0)
    assert round(float(density)) == 34


def test_mode_count_matches_its_closed_form() -> None:
    """Equation (8.45) written out term by term for the 7 x 5 x 3 m room."""
    lx, ly, lz = LONG_ROOM
    volume = lx * ly * lz
    surface = 2.0 * (lx * ly + lx * lz + ly * lz)
    edges = 4.0 * (lx + ly + lz)
    f = 300.0
    x = f / LONG_C0
    expected = (
        4.0 * math.pi / 3.0 * volume * x**3
        + math.pi / 4.0 * surface * x**2
        + edges / 8.0 * x
    )
    assert float(room.room_mode_count(f, LONG_ROOM, speed_of_sound=LONG_C0)) == (
        pytest.approx(expected)
    )


def test_modal_density_is_the_derivative_of_the_count() -> None:
    """Equation (8.46) is d/df of Equation (8.45) (central difference)."""
    f, h = 500.0, 1e-3
    up = float(room.room_mode_count(f + h, LONG_ROOM, speed_of_sound=LONG_C0))
    down = float(room.room_mode_count(f - h, LONG_ROOM, speed_of_sound=LONG_C0))
    numeric = (up - down) / (2.0 * h)
    analytic = float(room.room_modal_density(f, LONG_ROOM, speed_of_sound=LONG_C0))
    assert numeric == pytest.approx(analytic, rel=1e-7)


def test_smooth_count_tracks_the_exact_enumeration_at_high_frequency() -> None:
    """The asymptotic count of Equation (8.45) meets the exact mode list.

    At 400 Hz the 7 x 5 x 3 m room already holds hundreds of modes, where the
    lattice-counting approximation is expected to be tight.
    """
    f_max = 400.0
    exact = len(
        room.room_modes(
            LONG_ROOM, max_frequency=f_max, speed_of_sound=LONG_C0
        ).frequencies
    )
    smooth = float(room.room_mode_count(f_max, LONG_ROOM, speed_of_sound=LONG_C0))
    assert exact > 800
    assert smooth == pytest.approx(exact, rel=0.01)


def test_result_geometry_properties() -> None:
    result = room.room_modes(LONG_ROOM, max_frequency=80.0)
    assert result.volume == pytest.approx(105.0)
    assert result.surface_area == pytest.approx(142.0)
    assert result.edge_length == pytest.approx(60.0)
    assert result.density(1000.0) == pytest.approx(
        room.room_modal_density(1000.0, LONG_ROOM, speed_of_sound=result.speed_of_sound)
    )


def test_schroeder_frequency_is_the_shared_implementation() -> None:
    """The result reuses phonometry.room.schroeder_frequency, not a copy."""
    result = room.room_modes(LONG_ROOM, max_frequency=80.0, reverberation_time=0.8)
    assert result.schroeder_frequency == pytest.approx(
        float(np.asarray(room.schroeder_frequency(0.8, 105.0))[()])
    )
    assert room.room_modes(LONG_ROOM, max_frequency=80.0).schroeder_frequency is None


def test_zero_order_mode_is_excluded() -> None:
    result = room.room_modes(LONG_ROOM, max_frequency=30.0)
    assert not np.any(np.all(result.orders == 0, axis=1))
    assert np.all(result.frequencies > 0.0)


def test_enumeration_refuses_an_unreasonable_lattice() -> None:
    """A whole-band request is refused with a pointer to the smooth count.

    The lattice grows as f_max^3, so asking for every mode of an ordinary room
    up to 20 kHz is hundreds of millions of triples. That regime is exactly
    where Equations (8.45) and (8.46) are accurate, so the error says so.
    """
    with pytest.raises(ValueError, match="room_mode_count"):
        room.room_modes(LONG_ROOM, max_frequency=20000.0)
    # The smooth count answers the same question there without enumerating.
    assert float(room.room_mode_count(20000.0, LONG_ROOM)) > 8e7


def test_result_is_a_frozen_dataclass() -> None:
    result = room.room_modes(LONG_ROOM, max_frequency=30.0)
    assert isinstance(result, room.RoomModesResult)
    with pytest.raises(AttributeError):
        result.speed_of_sound = 340.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"dimensions": (7.0, 5.0)}, "three room dimensions"),
        ({"dimensions": (7.0, 5.0, 0.0)}, "positive and finite"),
        ({"dimensions": (7.0, 5.0, math.nan)}, "positive and finite"),
    ],
)
def test_dimension_validation(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        room.room_modes(**kwargs)  # type: ignore[arg-type]


def test_plot_returns_two_panels_and_marks_the_schroeder_frequency() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = room.room_modes(LONG_ROOM, max_frequency=150.0, reverberation_time=0.8)
    axes = result.plot()
    assert isinstance(axes, np.ndarray)
    assert axes.size == 2
    ladder_labels = [t.get_text() for t in axes[0].get_legend().get_texts()]
    assert {"axial", "tangential", "oblique"} <= set(ladder_labels)
    assert any("Schroeder" in label for label in ladder_labels)
    assert "Modal density" in axes[1].get_ylabel()
    plt.close("all")


def test_plot_on_supplied_axes_draws_the_ladder_only() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _fig, ax = plt.subplots()
    result = room.room_modes(LONG_ROOM, max_frequency=80.0)
    returned = result.plot(ax=ax)
    assert returned is ax
    assert "Frequency" in ax.get_xlabel()
    plt.close("all")


def test_other_validation_errors() -> None:
    with pytest.raises(ValueError, match="'max_frequency' must be positive"):
        room.room_modes(LONG_ROOM, max_frequency=0.0)
    with pytest.raises(ValueError, match="'speed_of_sound' must be positive"):
        room.room_modes(LONG_ROOM, speed_of_sound=-1.0)
    with pytest.raises(ValueError, match="'reverberation_time' must be positive"):
        room.room_modes(LONG_ROOM, reverberation_time=0.0)
    with pytest.raises(ValueError, match="non-negative integers"):
        room.room_mode_frequency((1, -1, 0), LONG_ROOM)
    with pytest.raises(ValueError, match="non-negative integers"):
        room.room_mode_frequency((1.5, 0, 0), LONG_ROOM)
    two_dimensional = np.zeros((2, 4))
    with pytest.raises(ValueError, match="triple"):
        room.room_mode_frequency(two_dimensional, LONG_ROOM)
    with pytest.raises(ValueError, match="non-negative and finite"):
        room.room_mode_count(-1.0, LONG_ROOM)
    with pytest.raises(ValueError, match="non-negative and finite"):
        room.room_modal_density(math.inf, LONG_ROOM)


def _long_table_result() -> room.RoomModesResult:
    """Long Table 8.1 as a result: six modes, four axial and two tangential."""
    return room.room_modes(LONG_ROOM, max_frequency=61.0, speed_of_sound=LONG_C0)


@pytest.mark.parametrize(
    ("field", "slice_"),
    [("orders", slice(None, -1)), ("frequencies", slice(None, -1))],
)
def test_a_column_short_of_a_mode_is_refused(field: str, slice_: slice) -> None:
    """The three columns are one enumeration, and only one reader complains.

    The ladder masks the frequencies with ``kinds == kind`` and raises numpy's
    "boolean index did not match indexed array" from inside the drawing call,
    naming no column; ``count_by_kind`` reads the kinds alone and tallies five
    modes for a six-rung ladder without a word; and ``orders`` is read by
    neither, so a triple table one row short leaves ``orders[-1]`` holding the
    ``(0, 0, 1)`` triple of the 57.3 Hz mode against the 60.0 Hz frequency
    beside it.
    """
    result = _long_table_result()
    short = getattr(result, field)[slice_]
    with pytest.raises(ValueError, match=f"'{field}'.*per mode"):
        dataclasses.replace(result, **{field: short})


def test_a_kind_column_of_another_length_is_refused() -> None:
    """A kind column one long tallies seven modes on a six-mode ladder."""
    result = _long_table_result()
    longer = np.concatenate([result.kinds, result.kinds[:1]])
    with pytest.raises(ValueError, match="'kinds' \\(7\\).*per mode"):
        dataclasses.replace(result, kinds=longer)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("orders", "stacked", "'orders' must have 2 axes"),
        ("frequencies", "column", "'frequencies' must have one axis"),
        ("kinds", "column", "'kinds' must have one axis"),
    ],
)
def test_an_extra_axis_on_a_column_is_refused(
    field: str, value: str, match: str
) -> None:
    """Counting the modes leaves an extra axis free to reach the reader.

    A triple table stacked into ``(modes, 3, 2)`` keeps its six rows, so it
    draws and tallies exactly as a sound one does, and hands ``orders[0]``
    back as a 3 x 2 block; a frequency column of shape ``(modes, 1)`` draws
    the ladder at the right frequencies and then fails ``float`` on a single
    entry; a kind column of that shape tallies correctly and raises "too many
    indices for array" from the ladder.
    """
    result = _long_table_result()
    original = getattr(result, field)
    reshaped = (
        np.stack([original, original], axis=-1)
        if value == "stacked"
        else original.reshape(-1, 1)
    )
    with pytest.raises(ValueError, match=match):
        dataclasses.replace(result, **{field: reshaped})


def test_a_triple_missing_a_coordinate_is_refused() -> None:
    """The room lengths and the width of the triple table are one count.

    The rank check does not pin it: a two-column triple table has one row per
    mode and draws and tallies in silence, and is caught only later, by
    :func:`room_mode_frequency`, complaining about its own argument.
    """
    result = _long_table_result()
    with pytest.raises(ValueError, match="'orders \\(axis 1\\)'.*per coordinate"):
        dataclasses.replace(result, orders=result.orders[:, :2])
    with pytest.raises(ValueError, match="'dimensions' \\(2\\).*per coordinate"):
        dataclasses.replace(result, dimensions=(7.0, 5.0))


def test_an_empty_enumeration_is_accepted() -> None:
    """A limit below the fundamental leaves no modes, and that is not a slip."""
    result = room.room_modes(LONG_ROOM, max_frequency=1.0)
    assert result.orders.shape == (0, 3)
    assert result.frequencies.shape == (0,)
    assert result.kinds.shape == (0,)
    assert result.count_by_kind() == dict.fromkeys(room.modes.MODE_KINDS, 0)
