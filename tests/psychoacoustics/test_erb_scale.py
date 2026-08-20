#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the public ERB_N / Cam utilities (Glasberg and Moore 1990).

Oracle: **Moore, An Introduction to the Psychology of Hearing 6th ed., printed
pp. 76-77**:

* p. 76, the published closed form ``ERB_N = 24.7 (4.37 F + 1)`` Hz with
  ``F`` in kHz, and the ERB_N-number (Cam) scale
  ``ERB_N number = 21.4 log10(4.37 F + 1)``;
* p. 76, the stated check value: "the value of ERB_N for a centre frequency of
  1 kHz is about 130 Hz, and so an increase in frequency from 935 to 1065 Hz
  corresponds to one ERB_N";
* p. 77, the caption of Figure 3.6, "a frequency of 1000 Hz corresponds to
  15.59 Cams on the ERB_N-number scale".

The **oracle is the printed formula**, evaluated here independently in
``_published_erb`` and ``_published_cam``. The module states the same fit with
one more significant digit (24.673, 0.004368, 21.366), the precision the
ISO 532-2 loudness model uses, so the two agree to better than 0.3 %.

Moore's printed 15.59 Cam is *not* used as the oracle: his own two-digit
formula gives 15.62, and only the extended constants land on 15.59. It is
asserted below alongside that 15.62 purely to record which of the two rounding
levels the caption was computed with. The tests never call the implementation
to obtain an expected value.
"""

from __future__ import annotations

import importlib
import math

import numpy as np
import pytest

from phonometry import psychoacoustics

#: The ISO 532-2 loudness *module* (the package attribute of that name is the
#: re-exported function, so it has to be imported explicitly).
mg = importlib.import_module("phonometry.psychoacoustics.loudness.moore_glasberg")


def _published_erb(frequency: float) -> float:
    """Moore 6th ed. p. 76: ERB_N = 24.7 (4.37 F + 1), F in kHz."""
    return 24.7 * (4.37 * frequency / 1000.0 + 1.0)


def _published_cam(frequency: float) -> float:
    """Moore 6th ed. p. 76: ERB_N number = 21.4 log10(4.37 F + 1), F in kHz."""
    return 21.4 * math.log10(4.37 * frequency / 1000.0 + 1.0)


def test_erb_bandwidth_at_1_khz_is_about_130_hz() -> None:
    """Moore p. 76: "the value of ERB_N ... for 1 kHz is about 130 Hz"."""
    erb = float(np.asarray(psychoacoustics.erb_bandwidth(1000.0))[()])
    assert round(erb, -1) == 130.0


def test_935_to_1065_hz_spans_one_erb() -> None:
    """Moore p. 76: "an increase in frequency from 935 to 1065 Hz corresponds
    to one ERB_N"."""
    span = 1065.0 - 935.0
    centre = 0.5 * (935.0 + 1065.0)
    erb = float(np.asarray(psychoacoustics.erb_bandwidth(centre))[()])
    assert erb == pytest.approx(span, abs=3.0)
    lo, hi = (
        float(v)
        for v in np.asarray(psychoacoustics.cam_from_frequency([935.0, 1065.0]))
    )
    assert hi - lo == pytest.approx(1.0, abs=0.03)


def test_cam_of_1000_hz_is_15_59() -> None:
    """Moore p. 77: "a frequency of 1000 Hz corresponds to 15.59 Cams".

    The extended constants (21.366, 0.004368) recover the printed value to its
    two decimals; the rounded 21.4 / 4.37 of p. 76 give 15.62.
    """
    cam = float(np.asarray(psychoacoustics.cam_from_frequency(1000.0))[()])
    assert round(cam, 2) == 15.59
    assert round(_published_cam(1000.0), 2) == 15.62


@pytest.mark.parametrize(
    "frequency", [50.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 15000.0]
)
def test_matches_the_published_closed_form(frequency: float) -> None:
    """The extended constants agree with Moore's printed formulas to 0.3 %."""
    erb = float(np.asarray(psychoacoustics.erb_bandwidth(frequency))[()])
    assert erb == pytest.approx(_published_erb(frequency), rel=3e-3)
    cam = float(np.asarray(psychoacoustics.cam_from_frequency(frequency))[()])
    assert cam == pytest.approx(_published_cam(frequency), rel=3e-3)


def test_erb_bandwidth_is_the_stated_straight_line() -> None:
    """ERB_N = C1 (C2 f + 1): the intercept at 0 Hz is C1, the slope C1 C2."""
    assert float(np.asarray(psychoacoustics.erb_bandwidth(0.0))[()]) == pytest.approx(
        psychoacoustics.ERB_C1
    )
    lo = float(np.asarray(psychoacoustics.erb_bandwidth(1000.0))[()])
    hi = float(np.asarray(psychoacoustics.erb_bandwidth(2000.0))[()])
    assert hi - lo == pytest.approx(
        psychoacoustics.ERB_C1 * psychoacoustics.ERB_C2 * 1000.0
    )


def test_cam_is_the_integral_of_one_over_the_erb() -> None:
    """The Cam scale counts auditory-filter widths: di/df = 1 / ERB_N(f).

    Numerically integrating ``1 / ERB_N`` from 0 Hz must reproduce
    :func:`cam_from_frequency`; that identity fixes the relation
    ``C_CAM = ln(10) / (C1 C2)`` between the three constants.
    """
    assert (
        pytest.approx(
            math.log(10.0) / (psychoacoustics.ERB_C1 * psychoacoustics.ERB_C2),
            rel=5e-5,
        )
        == psychoacoustics.CAM_C
    )
    f = np.linspace(0.0, 4000.0, 400001)
    integral = float(
        np.trapezoid(1.0 / np.asarray(psychoacoustics.erb_bandwidth(f)), f)
    )
    # The published 21.366 is itself a rounding of ln(10)/(C1 C2) = 21.36534,
    # so the identity holds to the 3e-5 that rounding costs.
    assert integral == pytest.approx(
        float(np.asarray(psychoacoustics.cam_from_frequency(4000.0))[()]),
        rel=1e-4,
    )


@pytest.mark.parametrize("frequency", [0.0, 20.0, 440.0, 1000.0, 12500.0])
def test_cam_frequency_round_trip(frequency: float) -> None:
    cam = psychoacoustics.cam_from_frequency(frequency)
    assert float(np.asarray(psychoacoustics.frequency_from_cam(cam))[()]) == (
        pytest.approx(frequency, abs=1e-9, rel=1e-12)
    )


def test_round_trip_from_the_cam_side() -> None:
    cams = np.linspace(1.8, 38.9, 373)
    freqs = psychoacoustics.frequency_from_cam(cams)
    assert np.allclose(np.asarray(psychoacoustics.cam_from_frequency(freqs)), cams)


def test_scalar_in_scalar_out_array_in_array_out() -> None:
    assert isinstance(psychoacoustics.erb_bandwidth(1000.0), float)
    assert isinstance(psychoacoustics.cam_from_frequency(1000.0), float)
    assert isinstance(psychoacoustics.frequency_from_cam(15.0), float)
    assert np.asarray(psychoacoustics.erb_bandwidth([100.0, 200.0])).shape == (2,)


def test_loudness_model_shares_these_constants() -> None:
    """ISO 532-2 Formulae (1) and (6) are the same functions, not a copy."""
    assert mg._ERB_C1 is psychoacoustics.ERB_C1
    assert mg._ERB_C2 is psychoacoustics.ERB_C2
    assert mg._CAM_C is psychoacoustics.CAM_C
    grid = np.array([100.0, 1000.0, 8000.0])
    assert np.allclose(
        mg._erb_bandwidth(grid),
        np.asarray(psychoacoustics.erb_bandwidth(grid)),
    )
    cams = np.array([5.0, 20.0, 35.0])
    assert np.allclose(
        mg._fc_from_cam(cams), np.asarray(psychoacoustics.frequency_from_cam(cams))
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda: psychoacoustics.erb_bandwidth(-1.0),
        lambda: psychoacoustics.erb_bandwidth(math.nan),
        lambda: psychoacoustics.cam_from_frequency(-1.0),
        lambda: psychoacoustics.cam_from_frequency(math.inf),
        lambda: psychoacoustics.frequency_from_cam(-0.5),
        lambda: psychoacoustics.frequency_from_cam(math.nan),
    ],
)
def test_validation_errors(call: object) -> None:
    with pytest.raises(ValueError, match="non-negative and finite"):
        call()  # type: ignore[operator]
