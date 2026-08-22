#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the steady-state room field (Bies 6.4 / Kuttruff 5.6).

The oracles are the closed forms themselves (room constant, the direct- and
reverberant-field limits of the steady-state level, the critical distance as
the exact crossover of the two fields, and the Schroeder frequency) plus the
one numeric anchor the source texts print: Kuttruff's classroom example
(V = 200 m3, T = 1 s gives f_s about 140 Hz, Room Acoustics 6th ed. p. 68).
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from phonometry import room


def test_room_constant_closed_form() -> None:
    # R = S alpha / (1 - alpha); S = 100, alpha = 0.2 -> 25.
    assert room.room_constant(100.0, 0.2) == pytest.approx(25.0)
    # Live room (small alpha) -> small R; dead room (large alpha) -> large R.
    assert room.room_constant(100.0, 0.05) < room.room_constant(100.0, 0.5)


def test_room_constant_per_band() -> None:
    r = room.room_constant(150.0, np.array([0.1, 0.2, 0.4]))
    assert np.allclose(
        r, 150.0 * np.array([0.1, 0.2, 0.4]) / (1.0 - np.array([0.1, 0.2, 0.4]))
    )


def test_room_constant_domain() -> None:
    with pytest.raises(ValueError, match="strictly in"):
        room.room_constant(100.0, 1.0)
    with pytest.raises(ValueError, match="strictly in"):
        room.room_constant(100.0, 0.0)
    with pytest.raises(ValueError, match="'surface_area' must be positive"):
        room.room_constant(-1.0, 0.2)


def test_critical_distance_is_field_crossover() -> None:
    # rc is where the direct term Q/(4 pi r^2) equals the reverberant term 4/R.
    R, Q = 25.0, 1.0
    rc = float(room.critical_distance(R, directivity=Q))
    assert rc == pytest.approx(math.sqrt(Q * R / (16.0 * math.pi)))
    direct = Q / (4.0 * math.pi * rc**2)
    reverberant = 4.0 / R
    assert direct == pytest.approx(reverberant)


def test_critical_distance_directivity_scaling() -> None:
    # rc scales as sqrt(Q): a corner source (Q = 8) reaches sqrt(8) further.
    assert float(room.critical_distance(30.0, directivity=8.0)) == pytest.approx(
        math.sqrt(8.0) * float(room.critical_distance(30.0, directivity=1.0))
    )


def test_schroeder_frequency_kuttruff_classroom() -> None:
    # Kuttruff Room Acoustics 6e p. 68: V = 200 m3, T = 1 s -> f_s ~ 140 Hz.
    f_s = float(room.schroeder_frequency(1.0, 200.0))
    assert f_s == pytest.approx(2000.0 * math.sqrt(1.0 / 200.0))
    assert 139.0 < f_s < 143.0


def test_steady_state_spl_direct_and_reverberant_limits() -> None:
    Lw, R = 90.0, 25.0
    # Far field -> reverberant only: Lp -> Lw + 10 log10(4 / R).
    far = float(room.steady_state_spl(Lw, 1000.0, R))
    assert far == pytest.approx(Lw + 10.0 * math.log10(4.0 / R), abs=1e-3)
    # Very near -> direct dominates: Lp -> Lw + 10 log10(1 / (4 pi r^2)).
    r = 0.02
    near = float(room.steady_state_spl(Lw, r, R))
    assert near == pytest.approx(
        Lw + 10.0 * math.log10(1.0 / (4.0 * math.pi * r**2)), abs=0.2
    )


def test_steady_state_spl_reverberant_only() -> None:
    """``distance=None`` is the reverberant field alone, the r -> inf limit."""
    Lw, R = 90.0, 25.0
    reverberant = float(room.steady_state_spl(Lw, None, R))
    assert reverberant == pytest.approx(Lw + 10.0 * math.log10(4.0 / R))
    # Approached from below by a far receiver, and never exceeded.
    assert float(room.steady_state_spl(Lw, 1e4, R)) == pytest.approx(
        reverberant, abs=1e-6
    )
    assert float(room.steady_state_spl(Lw, 1.0, R)) > reverberant
    # The directivity factor alone does not move it: the reverberant field is
    # position-independent for a constant-power source.
    assert float(room.steady_state_spl(Lw, None, R, directivity=8.0)) == pytest.approx(
        reverberant
    )


@pytest.mark.parametrize(
    ("model", "exponent"),
    [("constant_power", 0.0), ("constant_volume", 1.0), ("constant_pressure", -1.0)],
)
def test_steady_state_spl_source_power_models(model: str, exponent: float) -> None:
    """Norton & Karczub 2e Table 4.5: ``Pi = Pi_0 Q^n`` with ``n`` = 0, 1, -1.

    A source in the intersection of three flat surfaces has ``Q = 8``, so the
    three models sit at ``+0``, ``+9.03`` and ``-9.03 dB`` of one another. The
    printed table rounds those to 0, +9 and -9 dB.
    """
    Lw, R, q = 100.0, 40.0, 8.0
    base = Lw + 10.0 * math.log10(4.0 / R)
    level = float(room.steady_state_spl(Lw, None, R, directivity=q, source_model=model))
    assert level - base == pytest.approx(exponent * 10.0 * math.log10(q))
    assert round(level - base) == pytest.approx(round(exponent * 9.03))


def test_steady_state_spl_characteristic_impedance_term() -> None:
    # The optional 10 log10(rho c / 400) term is about +0.14 dB at 20 degC.
    base = float(room.steady_state_spl(90.0, 2.0, 25.0))
    corrected = float(
        room.steady_state_spl(90.0, 2.0, 25.0, characteristic_impedance=413.0)
    )
    assert corrected - base == pytest.approx(10.0 * math.log10(413.0 / 400.0))
    assert 0.13 < corrected - base < 0.15


def test_steady_state_spl_at_critical_distance() -> None:
    # At rc the total is the incoherent sum of two equal fields: +3 dB over each.
    Lw, R = 95.0, 40.0
    rc = float(room.critical_distance(R))
    total = float(room.steady_state_spl(Lw, rc, R))
    one_field = Lw + 10.0 * math.log10(4.0 / R)
    assert total == pytest.approx(one_field + 10.0 * math.log10(2.0), abs=1e-6)


def test_steady_state_field_bundle() -> None:
    res = room.steady_state_field(90.0, 100.0, 0.2)
    assert isinstance(res, room.SteadyFieldResult)
    assert res.room_constant == pytest.approx(25.0)
    assert res.critical_distance == pytest.approx(math.sqrt(25.0 / (16.0 * math.pi)))
    # Total is the incoherent sum of the two component fields per distance.
    d = 10.0 ** (res.direct / 10.0)
    rv = 10.0 ** (res.reverberant / 10.0)
    assert np.allclose(res.total, 10.0 * np.log10(d + rv))
    # The direct field crosses the reverberant one at rc.
    i = int(np.argmin(np.abs(res.distances - res.critical_distance)))
    assert abs(res.direct[i] - res.reverberant[i]) < 1.5


def test_steady_state_field_custom_distances() -> None:
    r = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    res = room.steady_state_field(85.0, 200.0, 0.15, distances=r)
    assert np.array_equal(res.distances, r)
    # Reverberant field is position-independent.
    assert np.allclose(res.reverberant, res.reverberant[0])


def test_steady_field_validation() -> None:
    with pytest.raises(ValueError, match="'distance' must be positive"):
        room.steady_state_spl(90.0, -1.0, 25.0)
    with pytest.raises(ValueError, match="'room_constant' must be positive"):
        room.steady_state_spl(90.0, 1.0, -25.0)
    with pytest.raises(ValueError, match="source_model"):
        room.steady_state_spl(90.0, 1.0, 25.0, source_model="constant_intensity")
    with pytest.raises(ValueError, match="'reverberation_time' must be positive"):
        room.schroeder_frequency(-1.0, 200.0)
    empty = np.array([])
    with pytest.raises(ValueError, match="'distances' must be a non-empty"):
        room.steady_state_field(90.0, 100.0, 0.2, distances=empty)


def test_field_curves_must_run_over_one_distance_grid() -> None:
    """Curves off the distance grid are refused when built, not when drawn.

    The plot draws each level against ``distances`` point for point, so a
    length disagreement surfaces only as matplotlib's "x and y must have same
    first dimension" and two bare shapes, naming neither field nor result. An
    extra axis is worse: it counts one row per distance, passes every length
    check, and reaches the plot as a second curve in the same colour and dash,
    under a repeated legend entry.
    """
    good = room.steady_state_field(90.0, 100.0, 0.2)
    per_distance = "one value per distance"
    cases = (
        ("distances", good.distances[:-1], per_distance),
        ("direct", good.direct[:-1], per_distance),
        ("direct", np.append(good.direct, good.direct[-1]), per_distance),
        ("reverberant", good.reverberant[:-1], per_distance),
        ("reverberant", np.append(good.reverberant, 0.0), per_distance),
        ("total", good.total[:-1], per_distance),
        ("total", np.append(good.total, good.total[-1]), per_distance),
        ("distances", np.column_stack([good.distances] * 2), "must have one axis"),
        ("direct", np.column_stack([good.direct] * 2), "must have one axis"),
        ("reverberant", np.column_stack([good.reverberant] * 2), "must have one axis"),
        ("total", np.column_stack([good.total] * 2), "must have one axis"),
    )
    for field, value, fragment in cases:
        with pytest.raises(ValueError, match=rf"'{field}'.*{fragment}"):
            dataclasses.replace(good, **{field: value})


def test_steady_field_plot_smoke() -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    res = room.steady_state_field(90.0, 100.0, 0.2)
    ax = res.plot()
    assert ax.get_xlabel() == "Distance from source [m]"
