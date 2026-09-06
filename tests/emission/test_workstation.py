#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""The shared core of the ISO 11200 group, against the worked examples.

The oracle is ISO 11200:2014 Annex B, which prints four case studies with every
intermediate value: the equivalent absorption area, the measurement surface,
:math:`K_3`, the energy mean, the final emission level, the total standard
deviation and the expanded uncertainty. Those numbers were obtained
independently of this code, by reading the printed pages.

The one thing Annex B is not trusted on is :math:`\sigma_\mathrm{omc}`: its two
tables use different estimators for the same quantity, and ``docs/ERRATA.md``
records it. The library follows Equation (C.1), which prints :math:`1/(N-1)`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phonometry.emission import workstation as ws
from phonometry.emission._shared import SoundPowerWarning


def _energy_mean(levels: list[float]) -> float:
    """Energy average, the way Annex B takes its repeated readings."""
    return float(10.0 * np.log10(np.mean(np.power(10.0, np.asarray(levels) / 10.0))))


# --------------------------------------------------------------------------
# ISO 11200:2014 Annex B, Example 2 (Table B.2): ISO 11202 method A.1
# --------------------------------------------------------------------------


def test_annex_b_example_2_reproduces_end_to_end() -> None:
    """A machine in an assembly workshop, 73,2 dB at the work station.

    Every printed intermediate of Table B.2, in the order the table prints
    them. The room is 11 m x 8 m x 4 m with a 1,2 s reverberation time, and the
    work station is 1,6 m from the dominating source.
    """
    distance, volume, reverberation = 1.6, 11.0 * 8.0 * 4.0, 1.2
    surface = 2.0 * math.pi * distance**2
    absorption = 0.16 * volume / reverberation

    assert volume == pytest.approx(352.0)
    assert surface == pytest.approx(16.0, abs=0.1)
    assert absorption == pytest.approx(47.0, abs=0.1)

    ratio = ws.environmental_ratio_from_absorption(absorption, surface)
    k3 = ws.local_environmental_correction(ratio)
    assert k3 == pytest.approx(3.7, abs=0.05)

    mean = _energy_mean([77.5, 76.0, 77.2, 77.7, 75.9])
    assert mean == pytest.approx(76.9, abs=0.05)

    level = ws.emission_sound_pressure_level(mean, local_correction_db=k3)
    assert level == pytest.approx(73.2, abs=0.05)

    sigma = ws.total_standard_deviation(1.5, 1.0)
    assert sigma == pytest.approx(1.8, abs=0.05)
    assert ws.emission_expanded_uncertainty(sigma) == pytest.approx(2.9, abs=0.05)


def test_annex_b_example_2_earns_grade_2() -> None:
    """K3 of 3,7 dB is under the 4 dB boundary, which is what the table says."""
    k3 = ws.local_environmental_correction(
        ws.environmental_ratio_from_absorption(46.933, 16.085)
    )
    assert ws.grade_from_local_correction(k3) == "engineering"


# --------------------------------------------------------------------------
# ISO 11200:2014 Annex B, Example 3 (Table B.3): ISO 11202 method A.2
# --------------------------------------------------------------------------


def test_annex_b_example_3_background_correction() -> None:
    """79,0 dB against a 70,0 dB background is a 0,6 dB correction."""
    k1, held = ws.background_noise_correction_at_workstation(79.0, 70.0)
    assert k1 == pytest.approx(0.6, abs=0.05)
    assert not held
    assert ws.emission_sound_pressure_level(
        79.0, background_correction_db=k1
    ) == pytest.approx(78.4, abs=0.05)


def test_annex_b_example_3_directivity_mean() -> None:
    """The six directivity points energy-average to the printed 77,7 dB."""
    assert _energy_mean([80.5, 81.0, 76.0, 75.0, 74.0, 72.0]) == pytest.approx(
        77.7, abs=0.05
    )


# --------------------------------------------------------------------------
# ISO 11200:2014 Annex B, Example 1 (Table B.1): ISO 11201
# --------------------------------------------------------------------------


def test_annex_b_example_1_uncertainty() -> None:
    """1,5 dB of reproducibility and k = 1,6 give the printed 2,4 dB."""
    sigma = ws.total_standard_deviation(1.5, 0.3)
    assert sigma == pytest.approx(1.5, abs=0.05)
    assert ws.emission_expanded_uncertainty(sigma) == pytest.approx(2.4, abs=0.05)


def test_the_two_annex_b_tables_disagree_on_the_estimator() -> None:
    r"""The errata, pinned so the library's choice cannot drift.

    Equation (C.1) prints :math:`1/(N-1)`, and Table B.3 agrees with it while
    Table B.1 divides by :math:`N`. The library follows the equation, which
    means it does not reproduce Table B.1's 0,3 dB and says so.
    """
    table_b1 = [94.5, 94.3, 93.8]
    table_b3 = [79.0, 80.2, 82.9]

    # Table B.3 agrees with Equation (C.1).
    assert ws.operating_standard_deviation(table_b3) == pytest.approx(2.0, abs=0.05)

    # Table B.1 does not: the equation gives 0,4 dB where the table prints 0,3.
    by_equation = ws.operating_standard_deviation(table_b1)
    assert by_equation == pytest.approx(0.36, abs=0.005)
    assert round(by_equation, 1) == 0.4
    # 0,3 is what dividing by N gives, which is not what (C.1) prints.
    assert float(np.std(table_b1, ddof=0)) == pytest.approx(0.29, abs=0.005)


# --------------------------------------------------------------------------
# The piecewise K3
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [
        (0.05, 7.0),  # deep in the capped branch
        (0.2, 7.0),  # the cap's own boundary
        (0.5, 3.0103),  # -10 lg 0,5
        (1.0, 0.0),  # the upper boundary
        (2.0, 0.0),  # above it, nothing to correct
    ],
)
def test_k3_follows_the_printed_branches(ratio: float, expected: float) -> None:
    assert ws.local_environmental_correction(ratio) == pytest.approx(expected, abs=1e-4)


def test_k3_is_continuous_where_the_cap_takes_over() -> None:
    """The 7 dB cap is the curve's own value rounded, not a separate rule.

    Just above the boundary the middle branch gives -10 lg 0,2 = 6,9897 dB, so
    the step onto the cap is 0,0103 dB: two orders below the 0,1 dB the
    standard prints its corrections to, and a tenth of the 0,1 dB a sound level
    meter resolves. Pinned exactly, because a real discontinuity here would be
    a defect and a rounded assertion would not see one appear.
    """
    just_above = ws.local_environmental_correction(0.2 + 1e-9)
    assert just_above == pytest.approx(6.9897, abs=1e-4)
    assert ws.MAX_K3_DB - just_above == pytest.approx(0.0103, abs=1e-4)


def test_k3_never_comes_back_as_minus_zero() -> None:
    """-10 lg 1 is -0.0, and a correction of minus nothing reads as a defect."""
    assert math.copysign(1.0, ws.local_environmental_correction(1.0)) > 0


def test_k3_refuses_a_ratio_that_is_not_positive() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        ws.local_environmental_correction(0.0)


# --------------------------------------------------------------------------
# The two routes to z
# --------------------------------------------------------------------------


def test_the_two_routes_to_z_are_the_same_quantity() -> None:
    r"""ISO 11204 A.1.2 says so, and the algebra is why.

    Under :math:`K_2 = 10 \lg (1 + 4 S_M / A)` the two expressions are
    identically equal, so a room described either way corrects the same.
    """
    absorption, surface = 47.0, 16.0
    k2 = 10.0 * math.log10(1.0 + 4.0 * surface / absorption)
    assert ws.environmental_ratio_from_k2(k2) == pytest.approx(
        ws.environmental_ratio_from_absorption(absorption, surface), abs=1e-12
    )


def test_without_directivity_k3_is_exactly_k2() -> None:
    """A work station that sees no more than the surface needs the same
    correction the surface needed, which is the expression collapsing.
    """
    for k2 in (1.0, 2.5, 4.0, 6.0, 7.0):
        k3 = ws.local_environmental_correction(ws.environmental_ratio_from_k2(k2))
        assert k3 == pytest.approx(k2, abs=1e-9)


def test_it_stops_following_k2_at_the_cap() -> None:
    """The claim above holds only as far as 7 dB, which is where the cap is.

    Every value the first test tries sits below the cap, so on its own it
    would let "K3 equals K2" be written without the qualifier.
    """
    for k2 in (8.0, 10.0, 20.0):
        k3 = ws.local_environmental_correction(ws.environmental_ratio_from_k2(k2))
        assert k3 == pytest.approx(ws.MAX_K3_DB)
        assert k3 < k2


def test_directivity_reduces_the_correction() -> None:
    """A work station that hears the machine more strongly than the surface
    does hears proportionally less room, so K3 falls.
    """
    plain = ws.local_environmental_correction(ws.environmental_ratio_from_k2(4.0))
    directional = ws.local_environmental_correction(
        ws.environmental_ratio_from_k2(4.0, 6.0)
    )
    assert directional < plain


def test_the_ratio_refuses_a_negative_environmental_correction() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        ws.environmental_ratio_from_k2(-1.0)


# --------------------------------------------------------------------------
# The background correction and its thresholds
# --------------------------------------------------------------------------


def test_a_background_far_below_needs_no_correction() -> None:
    """Past 15 dB of margin the standard says to take K1 as zero."""
    k1, held = ws.background_noise_correction_at_workstation(90.0, 70.0)
    assert k1 == 0.0
    assert not held


def test_a_background_too_close_clamps_and_warns() -> None:
    """The reading is still worth reporting; it stops being a determination."""
    with pytest.warns(SoundPowerWarning, match="upper bound"):
        k1, held = ws.background_noise_correction_at_workstation(80.0, 77.0)
    assert held
    # Held at the grade-2 floor of 6 dB, which is a 1,3 dB correction.
    assert k1 == pytest.approx(1.256, abs=0.005)


def test_the_survey_grade_reaches_three_decibels_lower() -> None:
    """Grade 3 may be determined from a 3 dB margin, where grade 2 may not."""
    with pytest.warns(SoundPowerWarning):
        _, held_engineering = ws.background_noise_correction_at_workstation(
            80.0, 76.0, grade="engineering"
        )
    _, held_survey = ws.background_noise_correction_at_workstation(
        80.0, 76.0, grade="survey"
    )
    assert held_engineering
    assert not held_survey


def test_the_two_readings_must_line_up() -> None:
    with pytest.raises(ValueError, match="same shape"):
        ws.background_noise_correction_at_workstation([80.0, 79.0], [70.0])


# --------------------------------------------------------------------------
# Uncertainty and the sub-interval summation
# --------------------------------------------------------------------------


def test_the_components_add_in_quadrature() -> None:
    assert ws.total_standard_deviation(1.5, 2.0) == pytest.approx(2.5)


def test_a_stable_machine_contributes_nothing() -> None:
    assert ws.total_standard_deviation(1.5) == pytest.approx(1.5)


def test_the_standard_deviations_cannot_be_negative() -> None:
    with pytest.raises(ValueError, match="reproducibility_db"):
        ws.total_standard_deviation(-1.0)


def test_one_reading_is_not_a_standard_deviation() -> None:
    with pytest.raises(ValueError, match="at least two"):
        ws.operating_standard_deviation([80.0])


def test_subintervals_are_weighted_by_how_long_they_last() -> None:
    """Two states, one ten times longer than the other: the long one wins.

    Equal durations would give the plain energy mean, and the point of the
    equation is that they usually are not equal.
    """
    equal = ws.subinterval_level([80.0, 90.0], [1.0, 1.0])
    assert equal == pytest.approx(_energy_mean([80.0, 90.0]))

    mostly_quiet = ws.subinterval_level([80.0, 90.0], [10.0, 1.0])
    assert mostly_quiet < equal
    expected = 10.0 * math.log10((10.0 * 10.0**8.0 + 1.0 * 10.0**9.0) / 11.0)
    assert mostly_quiet == pytest.approx(expected)


def test_a_duration_is_needed_for_every_subinterval() -> None:
    with pytest.raises(ValueError, match="one duration per sub-interval"):
        ws.subinterval_level([80.0, 90.0], [1.0])


# --------------------------------------------------------------------------
# Shape
# --------------------------------------------------------------------------


def test_a_scalar_in_gives_a_scalar_out() -> None:
    """Every quantity here is as much an overall value as a spectrum."""
    assert isinstance(ws.local_environmental_correction(0.5), float)
    assert isinstance(ws.environmental_ratio_from_k2(3.0), float)
    assert isinstance(ws.environmental_ratio_from_absorption(47.0, 16.0), float)
    assert isinstance(ws.emission_sound_pressure_level(80.0), float)


def test_a_per_band_directivity_keeps_every_band() -> None:
    """The shape follows the broadcast, not the first argument.

    One environmental correction against a directivity index that varies by
    band is one ratio per band. Reading the rank off the correction alone
    returned the first band and dropped the rest without a word.
    """
    directivity = np.array([0.0, 3.0, 6.0])
    ratio = ws.environmental_ratio_from_k2(4.0, directivity)
    assert isinstance(ratio, np.ndarray)
    assert ratio.shape == directivity.shape
    # Rising directivity means the work station hears proportionally less room.
    assert np.all(np.diff(ratio) > 0)

    absorbed = ws.environmental_ratio_from_absorption(47.0, 16.0, directivity)
    assert isinstance(absorbed, np.ndarray)
    assert absorbed.shape == directivity.shape


def test_a_spectrum_in_gives_a_spectrum_out() -> None:
    bands = np.array([1.0, 2.0, 3.0, 4.5])
    k3 = ws.local_environmental_correction(ws.environmental_ratio_from_k2(bands))
    assert isinstance(k3, np.ndarray)
    assert k3.shape == bands.shape
    # With no directivity the correction is the environmental one, band by band.
    assert k3 == pytest.approx(bands, abs=1e-9)


# --------------------------------------------------------------------------
# The result and its figure
# --------------------------------------------------------------------------


def _result(**overrides: object) -> ws.EmissionPressureResult:
    """Annex B Example 2 as a result, which is what the figure draws."""
    fields: dict[str, object] = {
        "level_db": 73.2,
        "measured_level_db": 76.9,
        "background_correction_db": 0.0,
        "local_correction_db": 3.7,
        "grade": "engineering",
        "upper_bound": False,
        "standard": "ISO 11202",
    }
    fields.update(overrides)
    return ws.EmissionPressureResult(**fields)  # type: ignore[arg-type]


def test_the_figure_draws_the_subtraction_it_describes() -> None:
    """Four bars, and the two corrections hang between the two levels.

    Looking at the figure is not covering it: the arithmetic that positions
    the floating bars is the part a defect would hide in, so the bar geometry
    is asserted rather than the fact that a figure appeared.
    """
    import matplotlib.pyplot as plt

    ax = _result().plot()
    bars = ax.patches[:4]
    heights = [round(bar.get_height(), 3) for bar in bars]
    bottoms = [round(bar.get_y(), 3) for bar in bars]

    assert heights == [76.9, 0.0, 3.7, 73.2]
    # The measured and emission bars stand on the axis; K1 starts where the
    # reading left off and K3 ends on the emission level.
    assert bottoms[0] == 0.0
    assert bottoms[3] == 0.0
    assert bottoms[1] == pytest.approx(76.9)
    assert bottoms[2] == pytest.approx(73.2)
    assert bottoms[2] + heights[2] == pytest.approx(76.9)
    plt.close("all")


def test_the_figure_names_the_part_it_followed() -> None:
    import matplotlib.pyplot as plt

    assert "ISO 11202" in _result().plot().get_title()
    plt.close("all")


def test_the_figure_says_when_the_level_is_an_upper_bound() -> None:
    """A held background correction is not a footnote; it is hatching."""
    import matplotlib.pyplot as plt

    plain = _result().plot()
    assert all(bar.get_hatch() is None for bar in plain.patches[:4])
    plt.close("all")

    bounded = _result(upper_bound=True).plot()
    hatches = [bar.get_hatch() for bar in bounded.patches[:4]]
    assert hatches[0] == "//"
    assert hatches[3] == "//"
    assert hatches[1] is None
    plt.close("all")


def test_the_figure_localises_its_labels() -> None:
    import matplotlib.pyplot as plt

    spanish = _result().plot(language="es")
    assert "Nivel de presión sonora" in spanish.get_ylabel()
    assert "puesto de trabajo" in spanish.get_title()
    labels = [t.get_text() for t in spanish.get_xticklabels()]
    assert any("medido" in label for label in labels)
    plt.close("all")


def test_the_figure_reports_the_grade_in_its_legend() -> None:
    import matplotlib.pyplot as plt

    engineering = _result().plot()
    assert "grade 2" in engineering.get_legend().get_texts()[0].get_text()
    plt.close("all")

    survey = _result(grade="survey").plot()
    assert "grade 3" in survey.get_legend().get_texts()[0].get_text()
    plt.close("all")


def test_the_figure_refuses_a_spectrum() -> None:
    """It draws one determination; a per-band result is a different figure."""
    import matplotlib.pyplot as plt

    bands = _result(
        level_db=np.array([70.0, 71.0]),
        measured_level_db=np.array([74.0, 75.0]),
        local_correction_db=np.array([4.0, 4.0]),
    )
    with pytest.raises(ValueError, match="one determination"):
        bands.plot()
    plt.close("all")


def test_the_figure_rejects_an_unknown_language() -> None:
    """Built first, so the refusal can only be the one under test."""
    result = _result()
    with pytest.raises(ValueError, match="Unknown language"):
        result.plot(language="xx")
