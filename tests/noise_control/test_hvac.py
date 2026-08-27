#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the HVAC duct methods (Bies Chapter 8).

Oracles: exact points of the ASHRAE end-reflection table (Bies Table 8.14) and
elbow table (Bies Table 8.11), Wells' plenum closed form (Eq. (8.275)) with its
room-constant reverberant term and limits, and the VDI 2081 flow-noise formulas
(Eqs. (8.251), (8.254)) evaluated directly.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from phonometry.noise_control import hvac


def test_end_reflection_table_nodes_flush() -> None:
    # Bies Table 8.14, flush: D = 200 mm -> [15, 10, 5, 2, 1, 0] dB.
    bands = [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0]
    res = hvac.end_reflection_loss(bands, 0.200, termination="flush")
    assert np.allclose(res.values, [15, 10, 5, 2, 1, 0], atol=1e-6)


def test_end_reflection_table_nodes_free() -> None:
    # Bies Table 8.14, free space: D = 150 mm -> [20, 14, 9, 5, 2, 1] dB.
    bands = [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0]
    res = hvac.end_reflection_loss(bands, 0.150, termination="free")
    assert np.allclose(res.values, [20, 14, 9, 5, 2, 1], atol=1e-6)


def test_end_reflection_free_exceeds_flush() -> None:
    res_flush = hvac.end_reflection_loss([125.0], 0.3, termination="flush")
    res_free = hvac.end_reflection_loss([125.0], 0.3, termination="free")
    assert res_free.values[0] > res_flush.values[0]


def test_elbow_table_bands() -> None:
    # Square, no vanes, unlined W = 0.3 m; W/lambda selects the table step.
    c = 343.0
    # f such that W/lambda = 0.4 (in 0.28-0.55) -> 5 dB
    f = 0.4 * c / 0.3
    res = hvac.elbow_insertion_loss([f], 0.3, bend_type="square")
    assert res.values[0] == pytest.approx(5.0)
    # W/lambda < 0.14 -> 0 dB
    res0 = hvac.elbow_insertion_loss([0.1 * c / 0.3], 0.3)
    assert res0.values[0] == 0.0


def test_elbow_bin_edges_belong_to_the_row_they_open() -> None:
    """Table 8.11 rows read "a <= W/lambda < b", so an edge opens its row.

    Square, no vanes, unlined (0, 1, 5, 8, 4, 3 dB): a ratio exactly on 0.14,
    0.28, 0.55, 1.11 or 2.22 takes the value of the row starting there. Unit
    width and speed of sound make ``W f / c`` reproduce each ratio bit for bit,
    so the test exercises the binning rather than a floating-point round trip.
    """
    c = 1.0
    w = 1.0
    edges = [0.14, 0.28, 0.55, 1.11, 2.22]
    expected = [1.0, 5.0, 8.0, 4.0, 3.0]
    for ratio, value in zip(edges, expected, strict=True):
        res = hvac.elbow_insertion_loss(
            [ratio], w, bend_type="square", speed_of_sound=c
        )
        assert res.values[0] == pytest.approx(value), ratio
    # Just below an edge still belongs to the row below it.
    below = hvac.elbow_insertion_loss(
        [0.13999], w, bend_type="square", speed_of_sound=c
    )
    assert below.values[0] == 0.0


def test_elbow_lined_beats_unlined_high_freq() -> None:
    c = 343.0
    f = 1.5 * c / 0.3  # W/lambda = 1.5 (1.11-2.22 band)
    unlined = hvac.elbow_insertion_loss([f], 0.3, bend_type="square").values[0]
    lined = hvac.elbow_insertion_loss([f], 0.3, bend_type="square", lined=True).values[
        0
    ]
    assert lined > unlined  # 10 vs 4 dB


def test_elbow_round_rejects_options() -> None:
    with pytest.raises(ValueError, match="neither vanes nor lining"):
        hvac.elbow_insertion_loss([500.0], 0.3, bend_type="round", lined=True)


def test_plenum_wells_closed_form() -> None:
    # TL = -10 log10[S_out(cos theta/(pi r^2) + (1-alpha)/(S_w alpha))].
    s_out, r, s_w, alpha = 0.1, 1.0, 20.0, 0.2
    tl = hvac.plenum_attenuation(s_out, r, s_w, alpha)
    direct = 1.0 / (math.pi * r**2)
    reverb = (1.0 - alpha) / (s_w * alpha)
    expected = -10.0 * math.log10(s_out * (direct + reverb))
    assert tl == pytest.approx(expected)


def test_plenum_more_absorption_more_loss() -> None:
    a = hvac.plenum_attenuation(0.1, 1.0, 20.0, 0.1)
    b = hvac.plenum_attenuation(0.1, 1.0, 20.0, 0.5)
    assert b > a


def test_plenum_per_band() -> None:
    tl = hvac.plenum_attenuation(0.1, 1.0, 20.0, np.array([0.1, 0.3, 0.5]))
    assert isinstance(tl, np.ndarray)
    assert tl.shape == (3,)


def test_plenum_angle_endpoints_accepted() -> None:
    # cos(pi/2) = 0: at the endpoint only the reverberant term is left.
    grazing = hvac.plenum_attenuation(0.1, 1.0, 20.0, 0.2, angle=math.pi / 2.0)
    head_on = hvac.plenum_attenuation(0.1, 1.0, 20.0, 0.2, angle=0.0)
    assert grazing > head_on


@pytest.mark.parametrize("angle", [math.nan, math.pi, -0.3], ids=["nan", "pi", "neg"])
def test_plenum_angle_outside_range_raises(angle: float) -> None:
    # A NaN used to come back as a NaN result and an obtuse angle drove
    # log10 negative with only a RuntimeWarning; both refuse by name now.
    with pytest.raises(ValueError, match=r"'angle' must lie in \[0, pi/2\]"):
        hvac.plenum_attenuation(0.1, 1.0, 20.0, 0.2, angle=angle)


def test_plenum_empty_absorption_raises() -> None:
    # np.any/np.all pass vacuously on an empty array; refuse it by name.
    with pytest.raises(ValueError, match="'mean_absorption' must be a scalar"):
        hvac.plenum_attenuation(0.1, 1.0, 20.0, [])


def test_plenum_two_dimensional_absorption_raises() -> None:
    with pytest.raises(ValueError, match="'mean_absorption' must be a scalar"):
        hvac.plenum_attenuation(0.1, 1.0, 20.0, np.full((2, 3), 0.2))


def test_flow_noise_straight_duct_formula() -> None:
    # Bies Eq. (8.251), VDI 2081-1: note the -2 constant term.
    f = np.array([250.0])
    u, s = 10.0, 0.04
    res = hvac.flow_noise_straight_duct(f, u, s)
    expected = (
        7.0
        + 50.0 * math.log10(u)
        + 10.0 * math.log10(s)
        - 2.0
        - 26.0 * math.log10(1.14 + 0.02 * 250.0 / u)
    )
    assert res.values[0] == pytest.approx(expected)
    # Pinned numeric anchor so the constant cannot silently drift.
    assert res.values[0] == pytest.approx(35.4347, abs=1e-3)


def test_flow_noise_scales_with_velocity() -> None:
    # Regenerated noise rises steeply with flow speed (dominant 50 log10 U term,
    # tempered by the frequency term which also depends on U).
    f = np.array([250.0])
    levels = [
        hvac.flow_noise_straight_duct(f, u, 0.04).values[0] for u in (5.0, 10.0, 15.0)
    ]
    assert levels[0] < levels[1] < levels[2]
    assert levels[2] - levels[0] > 20.0


def test_flow_noise_bend_formula() -> None:
    f = np.array([500.0])
    u, s, h, rho = 12.0, 0.04, 0.2, 1.206
    res = hvac.flow_noise_bend(f, u, s, h, density=rho)
    lws = 30.0 * math.log10(u) + 10.0 * math.log10(s) + 10.0 * math.log10(rho) + 117.0
    ns = 500.0 * h / u
    expected = (
        lws - 10.0 * math.log10(1.0 + 0.165 * ns**2) + 30.0 * math.log10(u) - 103.0
    )
    assert res.values[0] == pytest.approx(expected)


def test_plot_and_validation() -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    hvac.end_reflection_loss([125.0, 250.0], 0.3).plot()
    hvac.flow_noise_straight_duct([250.0, 500.0], 10.0, 0.04).plot()
    with pytest.raises(ValueError, match="'termination' must be"):
        hvac.end_reflection_loss([125.0], 0.3, termination="bad")
    with pytest.raises(ValueError, match="'mean_absorption' must lie strictly"):
        hvac.plenum_attenuation(0.1, 1.0, 20.0, 1.0)


def test_spectrum_must_run_over_one_band_axis() -> None:
    """A spectrum off its own band axis is refused when built, not when read.

    Both readers pair the two arrays, so a length that disagrees is loud but
    far from the mistake: the fiche's row loop raises a bare ``IndexError:
    list index out of range`` for more values than labels, and the figure it
    embeds, the same one ``plot()`` draws, reports matplotlib's ``x and y must
    have same first dimension`` and two bare shapes naming neither field. The
    extra axis is the silent one: a ``(bands, 2)`` array counts one entry per
    band and ``plot()`` draws each column as an ordinary curve, handing back a
    figure of two spectra that looks like a spectrum.
    """
    good = hvac.elbow_insertion_loss(
        [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0], 0.4, lined=True
    )
    per_band = "one value per band"
    cases = (
        ("values", good.values[:-1], per_band),
        ("values", np.append(good.values, good.values[-1]), per_band),
        ("frequencies", good.frequencies[:-1], per_band),
        ("frequencies", np.append(good.frequencies, 4000.0), per_band),
        ("values", np.column_stack([good.values] * 2), "must have one axis"),
        ("frequencies", np.column_stack([good.frequencies] * 2), "must have one axis"),
    )
    for field, value, fragment in cases:
        with pytest.raises(ValueError, match=rf"'{field}'.*{fragment}"):
            dataclasses.replace(good, **{field: value})


def test_a_spectrum_refuses_a_quantity_tag_it_cannot_report() -> None:
    """The tag decides what the sheet says and which way the verdict runs.

    ``quantity`` is a ``Literal`` for the type checker only; at run time an
    unexpected string used to fall through every ``== "sound_power_level"``
    test, so a regenerated-noise spectrum with a typo'd tag rendered a
    complete attenuation sheet on which 85 dB of duct noise PASSED a 40 dB
    maximum requirement.
    """
    good = hvac.elbow_insertion_loss(
        [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0], 0.4, lined=True
    )
    with pytest.raises(ValueError, match="'quantity' must be one of"):
        dataclasses.replace(good, quantity="power")  # type: ignore[arg-type]
