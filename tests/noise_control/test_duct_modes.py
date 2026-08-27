#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the higher-order duct modes with mean flow.

Oracle: Norton & Karczub, *Fundamentals of Noise and Vibration Analysis for
Engineers* 2nd ed. (CUP, 2003), section 7.3 (Eqs. 7.6, 7.8, 7.9, 7.10 and the
Table 7.1 eigenvalues) with the *printed answers* to problems 7.1 and 7.2
(Answers to problems, chapter 7):

* Problem 7.1, a 254 mm circular duct carrying steam, no flow and 200 m/s::

      mode    f_co no flow   f_co with flow   k_x with flow
      (1, 0)      934 Hz          812 Hz        -8.23 1/m
      (2, 0)     1550 Hz         1348 Hz       -13.66 1/m
      (0, 1)     1945 Hz         1691 Hz       -17.13 1/m
      (3, 0)     2132 Hz         1854 Hz       -18.79 1/m
      (4, 0)     2699 Hz         2347 Hz       -23.78 1/m
      (1, 1)     2706 Hz         2354 Hz       -23.84 1/m

  The book does not print the speed of sound it used; 405 m/s reproduces the
  whole no-flow column to the decibel-equivalent of its own rounding (every
  value to within 0.5 Hz), and is the value used here.

* Problem 7.2, a 0.65 m x 0.4 m rectangular air-conditioning duct with a mean
  air flow of 15 m/s: the first three cut-on frequencies are 264 Hz, 428 Hz and
  503 Hz. Note the third is the (1, 1) mode at 503 Hz and not the (2, 0) mode
  at 527 Hz, so the ordering of the returned modes is itself under test.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from phonometry.noise_control import duct_modes

#: Speed of sound in the steam of Norton problem 7.1, m/s (see module docstring).
_C_STEAM = 405.0

#: Norton, answer to problem 7.1: modes, no-flow cut-on (Hz), cut-on with a
#: 200 m/s mean flow (Hz) and the axial wavenumber at cut-on (1/m).
_PROBLEM_7_1 = (
    ((1, 0), 934.0, 812.0, -8.23),
    ((2, 0), 1550.0, 1348.0, -13.66),
    ((0, 1), 1945.0, 1691.0, -17.13),
    ((3, 0), 2132.0, 1854.0, -18.79),
    ((4, 0), 2699.0, 2347.0, -23.78),
    ((1, 1), 2706.0, 2354.0, -23.84),
)


def test_norton_problem_7_1_no_flow() -> None:
    res = duct_modes.circular_duct_cut_on(0.254, speed_of_sound=_C_STEAM, count=6)
    assert res.mach == 0.0
    assert res.modes == tuple(mode for mode, _, _, _ in _PROBLEM_7_1)
    expected = [f for _, f, _, _ in _PROBLEM_7_1]
    assert np.allclose(res.cut_on_no_flow, expected, atol=0.6)
    assert np.allclose(res.cut_on, expected, atol=0.6)
    assert np.allclose(res.axial_wavenumber, 0.0)


def test_norton_problem_7_1_with_flow() -> None:
    res = duct_modes.circular_duct_cut_on(
        0.254, flow_velocity=200.0, speed_of_sound=_C_STEAM, count=6
    )
    assert res.modes == tuple(mode for mode, _, _, _ in _PROBLEM_7_1)
    # The printed answers are rounded to the hertz; the (1, 1) entry (2354 Hz)
    # is one hertz above the exact 2353.0 Hz, the book's own rounding of an
    # intermediate.
    assert np.allclose(res.cut_on, [f for _, _, f, _ in _PROBLEM_7_1], atol=1.1)
    assert np.allclose(
        res.axial_wavenumber, [k for _, _, _, k in _PROBLEM_7_1], atol=0.006
    )
    # Eq. 7.8: the flow lowers every cut-on by the same sqrt(1 - M^2).
    beta = np.sqrt(1.0 - (200.0 / _C_STEAM) ** 2)
    assert np.allclose(res.cut_on, res.cut_on_no_flow * beta)


def test_norton_problem_7_2() -> None:
    res = duct_modes.rectangular_duct_cut_on(0.65, 0.4, flow_velocity=15.0, count=3)
    # The (1, 1) mode at 503 Hz cuts on before the (2, 0) mode at 527 Hz.
    assert res.modes == ((1, 0), (0, 1), (1, 1))
    assert np.allclose(res.cut_on, [264.0, 428.0, 503.0], atol=0.6)
    assert res.plane_wave_limit == pytest.approx(263.6, abs=0.1)


def test_plane_wave_eigenvalue_is_the_first_cut_on() -> None:
    # Norton section 7.3: plane waves only while k a_i < 1.8412.
    diameter, c = 0.254, 343.0
    limit = duct_modes.plane_wave_limit(diameter=diameter, speed_of_sound=c)
    k = 2.0 * np.pi * limit / c
    assert k * (diameter / 2.0) == pytest.approx(
        duct_modes.PLANE_WAVE_EIGENVALUE, rel=1e-12
    )


def test_plane_wave_limit_accepts_area_and_rectangle() -> None:
    from_diameter = duct_modes.plane_wave_limit(diameter=0.4)
    from_area = duct_modes.plane_wave_limit(area=np.pi * 0.4**2 / 4.0)
    assert from_area == pytest.approx(from_diameter)
    # Eq. 7.10 with p = 1, q = 0: c / (2 a) for the wider side.
    assert duct_modes.plane_wave_limit(width=0.65, height=0.4) == pytest.approx(
        343.0 / (2.0 * 0.65)
    )


def test_plane_wave_limit_requires_a_cross_section() -> None:
    with pytest.raises(
        ValueError, match=r"give 'diameter', or both 'width' and 'height', or 'area'"
    ):
        duct_modes.plane_wave_limit()


def test_warn_above_plane_wave_limit() -> None:
    with pytest.warns(
        duct_modes.PlaneWaveWarning,
        match=r"test duct: .* above the first duct cut-on frequency",
    ):
        mask = duct_modes.warn_above_plane_wave_limit(
            [125.0, 250.0, 500.0, 1000.0], 263.6, "test duct"
        )
    assert mask.tolist() == [False, False, True, True]


def test_no_warning_below_the_limit() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mask = duct_modes.warn_above_plane_wave_limit([63.0, 125.0], 263.6, "test duct")
    assert not mask.any()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"diameter": 0.0}, r"'diameter' must be positive"),
        (
            {"diameter": 0.2, "flow_velocity": -1.0},
            r"'flow_velocity' must be non-negative",
        ),
        (
            {"diameter": 0.2, "flow_velocity": 400.0},
            r"'flow_velocity' must be subsonic",
        ),
        ({"diameter": 0.2, "count": 0}, r"'count' must be between"),
        ({"diameter": 0.2, "count": 13}, r"'count' must be between"),
    ],
)
def test_circular_validation(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        duct_modes.circular_duct_cut_on(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"width": 0.0, "height": 0.4}, r"'width' must be positive"),
        ({"width": 0.65, "height": -0.4}, r"'height' must be positive"),
        ({"width": 0.65, "height": 0.4, "count": 0}, r"'count' must be at least"),
        (
            {"width": 0.65, "height": 0.4, "flow_velocity": 400.0},
            r"'flow_velocity' must be subsonic",
        ),
    ],
)
def test_rectangular_validation(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        duct_modes.rectangular_duct_cut_on(**kwargs)  # type: ignore[arg-type]


def test_plot_draws_both_ladders_and_shades_the_plane_wave_band() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib as mpl

    mpl.use("Agg")
    from matplotlib.axes import Axes

    res = duct_modes.circular_duct_cut_on(
        0.254, flow_velocity=200.0, speed_of_sound=_C_STEAM, count=6
    )
    ax = res.plot()
    assert isinstance(ax, Axes)
    lines = {str(line.get_label()): line for line in ax.get_lines()}
    assert np.allclose(lines["No flow"].get_ydata(), res.cut_on_no_flow)
    assert np.allclose(lines[f"$M$ = {res.mach:.3f}"].get_ydata(), res.cut_on)
    assert [t.get_text() for t in ax.get_xticklabels()] == [
        f"({p}, {q})" for p, q in res.modes
    ]
    mpl.pyplot.close(ax.figure)


def test_plot_language_is_validated() -> None:
    pytest.importorskip("matplotlib")
    res = duct_modes.rectangular_duct_cut_on(0.65, 0.4)
    with pytest.raises(ValueError, match=r"Unknown language 'xx'"):
        res.plot(language="xx")


def test_silencer_result_reports_its_plane_wave_limit() -> None:
    from phonometry.noise_control import silencers

    # A 0.1 m2 chamber is 357 mm across: plane waves only below about 1.26 kHz.
    with pytest.warns(duct_modes.PlaneWaveWarning):
        res = silencers.expansion_chamber([50.0, 100.0, 2000.0], 0.3, 0.02, 0.002)
    assert res.plane_wave_limit is not None
    assert res.plane_wave_limit == pytest.approx(
        duct_modes.plane_wave_limit(area=0.02), rel=1e-12
    )


def test_silencer_below_cut_on_does_not_warn() -> None:
    from phonometry.noise_control import silencers

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = silencers.expansion_chamber([50.0, 100.0], 0.3, 0.02, 0.002)
    assert res.plane_wave_limit is not None
