#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for ISO 9612:2009 occupational noise exposure (occupational_exposure).

The three worked examples of ISO 9612:2009 Annexes D (task-based), E (job-based)
and F (full-day) are the primary oracles: their LEX,8h and expanded uncertainty U
are reproduced to the standard's one-decimal precision. Closed-form identities
(single task at 85 dB -> 85; equal-level tasks; -3.01 dB per halved duration;
k = 1.65) and the Annex C intermediate quantities (Eq C.4-C.7, C.12, Tables C.4/C.5,
Table 1) anchor the remaining behaviour.
"""

from __future__ import annotations

import math

import pytest

from phonometry.hearing.occupational_exposure import (
    COVERAGE_FACTOR,
    INSTRUMENT_U2,
    ExposureResult,
    OccupationalExposureWarning,
    Task,
    TaskContribution,
    full_day_exposure,
    job_based_exposure,
    minimum_cumulative_duration_hours,
    table_c4_contribution,
    task_based_exposure,
)


# --------------------------------------------------------------------------- #
# Closed-form oracles.
# --------------------------------------------------------------------------- #
def test_single_task_85db_8h_is_85() -> None:
    """A single task at 85 dB sustained for 8 h gives LEX,8h = 85 exactly."""
    result = task_based_exposure([Task(samples=(85.0, 85.0, 85.0), duration_hours=8.0)])
    assert result.lex_8h == pytest.approx(85.0, abs=1e-9)
    assert result.strategy == "task"


def test_two_equal_half_day_tasks_equal_level() -> None:
    """Two 4-h tasks at level L give LEX,8h = L (energy sum, no time normalization loss)."""
    tasks = [
        Task(samples=(88.0,), duration_hours=4.0),
        Task(samples=(88.0,), duration_hours=4.0),
    ]
    result = task_based_exposure(tasks)
    assert result.lex_8h == pytest.approx(88.0, abs=1e-9)


def test_halving_duration_drops_contribution_by_3db() -> None:
    """Halving a task duration lowers its LEX,8h contribution by 10*log10(2) = 3.01 dB."""
    full = task_based_exposure([Task(samples=(90.0,), duration_hours=8.0)])
    half = task_based_exposure([Task(samples=(90.0,), duration_hours=4.0)])
    assert full.lex_8h - half.lex_8h == pytest.approx(10.0 * math.log10(2.0), abs=1e-9)


def test_coverage_factor_is_1_65() -> None:
    """The one-sided 95 % coverage factor is exactly 1.65 and U = 1.65*u."""
    assert COVERAGE_FACTOR == 1.65
    result = job_based_exposure([80.0, 82.0, 84.0, 81.0, 83.0], 8.0)
    assert result.expanded_uncertainty == pytest.approx(
        1.65 * result.combined_standard_uncertainty, rel=1e-12
    )


def test_energy_average_single_task_matches_lex_8h_primitive() -> None:
    """Task energy average agrees with 10*log10(mean(10^(0.1 L)))."""
    samples = (80.1, 82.2, 79.6)
    result = task_based_exposure([Task(samples=samples, duration_hours=8.0)])
    expected = 10.0 * math.log10(sum(10 ** (0.1 * s) for s in samples) / 3)
    assert result.tasks[0].lp_aeqt == pytest.approx(expected, abs=1e-9)
    # 8 h task: contribution equals the level itself.
    assert result.lex_8h == pytest.approx(expected, abs=1e-9)


# --------------------------------------------------------------------------- #
# Table lookups.
# --------------------------------------------------------------------------- #
def test_table_c4_exact_cells() -> None:
    """Exact Table C.4 cells are returned at the tabulated (N, u1) grid points."""
    assert table_c4_contribution(6, 2.0) == pytest.approx(1.4, abs=1e-9)
    assert table_c4_contribution(3, 0.5) == pytest.approx(0.6, abs=1e-9)
    assert table_c4_contribution(30, 6.0) == pytest.approx(2.0, abs=1e-9)
    assert table_c4_contribution(6, 3.5) == pytest.approx(3.3, abs=1e-9)
    assert table_c4_contribution(20, 2.0) == pytest.approx(0.5, abs=1e-9)


def test_table_c4_interpolates_annex_f_cell() -> None:
    """Annex F reads N=6, u1=1.65 dB as c1*u1 = 1.0 dB (interpolated)."""
    assert table_c4_contribution(6, 1.65) == pytest.approx(1.0, abs=0.05)


def test_table_c4_origin_and_clamp() -> None:
    """u1 = 0 gives 0 dB; below/above the grid clamps to the edge behaviour."""
    assert table_c4_contribution(6, 0.0) == pytest.approx(0.0, abs=1e-9)
    # Clamp N above 30 to the N=30 row.
    assert table_c4_contribution(50, 6.0) == pytest.approx(2.0, abs=1e-9)


def test_table_1_minimum_durations() -> None:
    """Table 1 breakpoints for the cumulative measurement duration."""
    assert minimum_cumulative_duration_hours(5) == 5.0
    assert minimum_cumulative_duration_hours(6) == 5.5
    assert minimum_cumulative_duration_hours(15) == 10.0
    assert minimum_cumulative_duration_hours(18) == 10.75
    assert minimum_cumulative_duration_hours(40) == 16.25
    assert minimum_cumulative_duration_hours(41) == 17.0


def test_instrument_u2_values() -> None:
    """Table C.5 instrument standard uncertainties."""
    assert INSTRUMENT_U2["class1"] == 0.7
    assert INSTRUMENT_U2["class2"] == 1.5
    assert INSTRUMENT_U2["personal_exposimeter"] == 1.5


# --------------------------------------------------------------------------- #
# Annex D — task-based worked example (welders).
# --------------------------------------------------------------------------- #
def _annex_d_tasks() -> list[Task]:
    return [
        Task(samples=(70.0,), duration_hours=1.5, label="planning/breaks"),
        Task(
            samples=(80.1, 82.2, 79.6),
            duration_hours=5.0,
            duration_range=(4.0, 6.0),
            label="welding",
        ),
        Task(
            samples=(86.5, 92.4, 89.3, 93.2, 87.8, 86.2),
            duration_hours=1.5,
            duration_range=(1.0, 2.0),
            label="cutting/grinding",
        ),
    ]


def test_annex_d_task_levels_and_contributions() -> None:
    """Annex D: task levels 80.8/90.1 dB and contributions 62.7/78.8/82.8 dB."""
    with pytest.warns(OccupationalExposureWarning):  # cutting/grinding spans > 3 dB
        result = task_based_exposure(_annex_d_tasks())
    by_label = {t.label: t for t in result.tasks}
    assert by_label["welding"].lp_aeqt == pytest.approx(80.8, abs=0.05)
    assert by_label["cutting/grinding"].lp_aeqt == pytest.approx(90.1, abs=0.05)
    # Contributions match the standard to rounding (it rounds Lp to 80.8/90.1
    # before Eq 8; unrounded welding gives 78.74 -> reported 78.8).
    assert by_label["planning/breaks"].lex_8h_contribution == pytest.approx(
        62.7, abs=0.1
    )
    assert by_label["welding"].lex_8h_contribution == pytest.approx(78.8, abs=0.1)
    assert by_label["cutting/grinding"].lex_8h_contribution == pytest.approx(
        82.8, abs=0.1
    )


def test_annex_d_daily_level() -> None:
    """Annex D: LEX,8h = 84.3 dB."""
    result = task_based_exposure(_annex_d_tasks(), warn=False)
    assert result.lex_8h == pytest.approx(84.3, abs=0.05)


def test_annex_d_sampling_uncertainties_and_sensitivities() -> None:
    """Annex D: u1a = 0.8/1.2 dB, c1a = 0.28/0.71 (Eq C.6, C.4)."""
    result = task_based_exposure(_annex_d_tasks(), warn=False)
    by_label = {t.label: t for t in result.tasks}
    assert by_label["welding"].u1a == pytest.approx(0.8, abs=0.05)
    assert by_label["cutting/grinding"].u1a == pytest.approx(1.2, abs=0.05)
    assert by_label["welding"].c1a == pytest.approx(0.28, abs=0.01)
    assert by_label["cutting/grinding"].c1a == pytest.approx(0.71, abs=0.01)
    assert by_label["planning/breaks"].c1a == pytest.approx(0.0, abs=0.01)


def test_annex_d_duration_sensitivities() -> None:
    """Annex D: c1b = 0.24 (welding) and 2.1 (cutting) [dB/h] (Eq C.5)."""
    result = task_based_exposure(_annex_d_tasks(), warn=False)
    by_label = {t.label: t for t in result.tasks}
    assert by_label["welding"].u1b == pytest.approx(1.0, abs=1e-9)
    assert by_label["cutting/grinding"].u1b == pytest.approx(0.5, abs=1e-9)
    assert by_label["welding"].c1b == pytest.approx(0.24, abs=0.02)
    assert by_label["cutting/grinding"].c1b == pytest.approx(2.1, abs=0.05)


def test_duration_samples_u1b_is_standard_error_of_mean() -> None:
    """Eq C.7 divides by J(J-1): samples (4, 6) h give u1b = 1.0 h, not sqrt(2)."""
    result = task_based_exposure(
        [Task(samples=(85.0, 85.0), duration_hours=5.0, duration_samples=(4.0, 6.0))],
        warn=False,
    )
    assert result.tasks[0].u1b == pytest.approx(1.0, abs=1e-9)


def test_annex_d_expanded_uncertainty_without_duration() -> None:
    """Annex D case a: U = 2.7 dB when the duration uncertainty is omitted."""
    result = task_based_exposure(
        _annex_d_tasks(), include_duration_uncertainty=False, warn=False
    )
    # The standard rounds c1a/u1a to two significant figures before squaring
    # (u^2 = 2.67); the unrounded budget is ~2.72. U rounds to 2.7 dB either way.
    assert result.combined_standard_uncertainty**2 == pytest.approx(2.67, abs=0.07)
    assert result.expanded_uncertainty == pytest.approx(2.7, abs=0.05)


def test_annex_d_expanded_uncertainty_with_duration() -> None:
    """Annex D case b: U = 3.2 dB when the duration uncertainty is included."""
    result = task_based_exposure(_annex_d_tasks(), warn=False)
    assert result.combined_standard_uncertainty**2 == pytest.approx(3.83, abs=0.07)
    assert result.expanded_uncertainty == pytest.approx(3.2, abs=0.05)
    assert result.upper_limit == pytest.approx(84.3 + 3.2, abs=0.1)


def test_annex_d_spread_rule_advisory() -> None:
    """The cutting/grinding task (range > 3 dB) trips the 3 dB spread advisory."""
    result = task_based_exposure(_annex_d_tasks(), warn=False)
    by_label = {t.label: t for t in result.tasks}
    assert by_label["cutting/grinding"].spread_advisory is True
    assert by_label["welding"].spread_advisory is False
    assert result.sampling_advisory is True


# --------------------------------------------------------------------------- #
# Annex E — job-based worked example (production line, 18 workers).
# --------------------------------------------------------------------------- #
def test_annex_e_job_based() -> None:
    """Annex E: Lp,A,eqTe = 88.4, u1 = 2.0, c1u1 = 1.4, LEX,8h = 88.1, U = 3.8 dB."""
    samples = [88.1, 86.1, 89.7, 86.5, 91.1, 86.7]
    result = job_based_exposure(
        samples,
        effective_duration_hours=7.5,
        instrument="personal_exposimeter",
        n_workers=18,
        sample_duration_hours=2.0,
    )
    assert result.lp_aeqte == pytest.approx(88.4, abs=0.05)
    assert result.u1 == pytest.approx(2.0, abs=0.05)
    assert result.c1u1 == pytest.approx(1.4, abs=0.05)
    # The standard rounds Lp,A,eqTe to 88.4 before Eq 12; the unrounded value
    # gives 88.16, which the standard reports as 88.1.
    assert result.lex_8h == pytest.approx(88.1, abs=0.1)
    assert result.combined_standard_uncertainty == pytest.approx(2.3, abs=0.05)
    assert result.expanded_uncertainty == pytest.approx(3.8, abs=0.05)
    assert result.strategy == "job"


def test_annex_e_table1_satisfied_no_advisory() -> None:
    """Annex E: 12 h cumulative >= 10.75 h Table 1 minimum -> no advisory."""
    samples = [88.1, 86.1, 89.7, 86.5, 91.1, 86.7]
    result = job_based_exposure(samples, 7.5, n_workers=18, sample_duration_hours=2.0)
    assert result.sampling_advisory is False


def test_job_based_below_table1_minimum_warns() -> None:
    """Too-short cumulative duration raises the Table 1 advisory."""
    samples = [88.0, 86.0, 89.0, 86.0, 91.0]
    with pytest.warns(OccupationalExposureWarning):
        result = job_based_exposure(
            samples, 8.0, n_workers=18, sample_duration_hours=0.5
        )
    assert result.sampling_advisory is True


# --------------------------------------------------------------------------- #
# Annex F — full-day worked example (forklift drivers).
# --------------------------------------------------------------------------- #
def test_annex_f_full_day() -> None:
    """Annex F: Lp,A,eqTe = 89.5, u1 = 1.65, c1u1 = 1.0, LEX,8h = 90.1, U = 3.4 dB."""
    samples = [88.0, 91.9, 87.6, 90.4, 89.0, 88.4]
    result = full_day_exposure(
        samples, effective_duration_hours=9.25, instrument="personal_exposimeter"
    )
    assert result.lp_aeqte == pytest.approx(89.5, abs=0.05)
    assert result.u1 == pytest.approx(1.65, abs=0.05)
    assert result.c1u1 == pytest.approx(1.0, abs=0.05)
    assert result.lex_8h == pytest.approx(90.1, abs=0.05)
    assert result.combined_standard_uncertainty == pytest.approx(2.06, abs=0.05)
    assert result.expanded_uncertainty == pytest.approx(3.4, abs=0.05)
    assert result.strategy == "full_day"


def test_full_day_three_measurements_spread_advisory() -> None:
    """Three full-day measurements spanning >= 3 dB trip the Clause 11.3 advisory."""
    with pytest.warns(OccupationalExposureWarning):
        result = full_day_exposure([85.0, 89.0, 86.0], 8.0)
    assert result.sampling_advisory is True


def test_full_day_three_close_measurements_no_advisory() -> None:
    """Three full-day measurements within 3 dB need no additional measurements."""
    result = full_day_exposure([85.0, 86.0, 87.0], 8.0)
    assert result.sampling_advisory is False


# --------------------------------------------------------------------------- #
# Validation and edge cases.
# --------------------------------------------------------------------------- #
def test_job_based_high_spread_triggers_c4_advisory() -> None:
    """A large sampling spread pushes c1*u1 above 3.5 dB and warns (Clause 10.4)."""
    samples = [70.0, 80.0, 90.0, 75.0, 85.0]  # wide spread, small N
    with pytest.warns(OccupationalExposureWarning):
        result = job_based_exposure(samples, 8.0)
    assert result.c1u1 is not None
    assert result.c1u1 > 3.5
    assert result.sampling_advisory is True


def test_task_based_requires_tasks() -> None:
    with pytest.raises(ValueError, match="At least one task"):
        task_based_exposure([])


def test_task_rejects_nonpositive_duration() -> None:
    # Task.__post_init__ is what rejects it: task_based_exposure never gets to
    # see an object with a non-positive duration.
    with pytest.raises(ValueError, match="'duration_hours' must be positive"):
        Task(samples=(85.0,), duration_hours=0.0)


def test_task_rejects_empty_samples() -> None:
    with pytest.raises(ValueError, match="sample"):
        Task(samples=(), duration_hours=8.0)


def test_task_rejects_nan_sample() -> None:
    # A NaN sample would pass the emptiness check, make lex_8h NaN and stop
    # the exposure plot inside matplotlib ("Axis limits cannot be NaN or
    # Inf"), naming neither the task nor the sample.
    with pytest.raises(ValueError, match="'samples' must contain only finite"):
        Task(samples=(float("nan"), 84.0, 85.2), duration_hours=4.0)


def test_task_rejects_nan_duration() -> None:
    # NaN passes a bare `<= 0.0`; require_positive refuses it by name.
    with pytest.raises(ValueError, match="'duration_hours' must be positive"):
        Task(samples=(85.0,), duration_hours=float("nan"))


def test_job_based_rejects_nonpositive_sample_duration() -> None:
    with pytest.raises(ValueError, match="sample_duration_hours"):
        job_based_exposure(
            [80.0, 82.0, 81.0], 8.0, n_workers=6, sample_duration_hours=0.0
        )


def test_job_based_requires_two_samples() -> None:
    with pytest.raises(ValueError, match="At least two samples"):
        job_based_exposure([85.0], 8.0)


def test_duration_range_order_validated() -> None:
    reversed_range = Task(
        samples=(85.0, 85.0), duration_hours=4.0, duration_range=(6.0, 4.0)
    )
    with pytest.raises(ValueError, match=r"duration_range must be \(T_min, T_max\)"):
        task_based_exposure([reversed_range])


def test_result_rejects_an_unknown_strategy_tag() -> None:
    """The fiche resolves the strategy through a dict to name its clause.

    Unpinned, ``"daily"`` passed construction and died inside the renderer as
    a bare ``KeyError: 'daily'``, naming neither the field, the class nor the
    three strategies ISO 9612 defines.
    """
    import dataclasses

    result = job_based_exposure([80.0, 82.0, 81.0, 83.0, 80.0], 8.0)
    with pytest.raises(ValueError, match="'strategy' must be one of"):
        dataclasses.replace(result, strategy="daily")


def test_result_rejects_an_unknown_instrument_tag() -> None:
    """``instrument`` is a ``Literal`` with no runtime force behind it.

    The fiche's instrumentation line meets the same dict lookup, so
    ``"Class 1"`` -- the human spelling of the tag ``"class1"`` -- reached it
    as a bare ``KeyError: 'Class 1'`` out of the middle of the render.
    """
    import dataclasses

    result = job_based_exposure([80.0, 82.0, 81.0, 83.0, 80.0], 8.0)
    with pytest.raises(ValueError, match="'instrument' must be one of"):
        dataclasses.replace(result, instrument="Class 1")


@pytest.mark.parametrize(
    "field_name",
    ["lex_8h", "combined_standard_uncertainty", "expanded_uncertainty", "u1", "u2"],
)
def test_result_rejects_a_non_finite_level(field_name: str) -> None:
    """Every level on the fiche is printed through ``display_round``.

    Its ``math.floor`` dies on a NaN with "cannot convert float NaN to
    integer", a crash that names no field, no class and no cause. Each
    strategy computes these from samples already pinned finite, so a NaN is
    never the library's own output.
    """
    import dataclasses

    result = job_based_exposure([80.0, 82.0, 81.0, 83.0, 80.0], 8.0)
    with pytest.raises(ValueError, match=f"ExposureResult: '{field_name}' must be"):
        dataclasses.replace(result, **{field_name: float("nan")})


def test_a_job_result_without_its_sampling_summary_is_refused() -> None:
    """The sampling summary is Clause 15 d), and the fiche prints it always.

    Built on the field defaults, the sheet rendered the literal string
    ``None`` for the number of samples and a fabricated 0,0 for every
    Formula (C.9) budget term, then closed on a normal PASS verdict: an
    accredited page stating a zero uncertainty budget nobody measured. The
    refusal names every term that is missing.
    """
    with pytest.raises(ValueError, match="'n_samples'.*must be supplied for a 'job'"):
        ExposureResult(
            lex_8h=82.0,
            combined_standard_uncertainty=1.2,
            expanded_uncertainty=2.0,
            strategy="job",
        )


def test_a_task_result_without_its_task_contributions_is_refused() -> None:
    """A task-based fiche renders one row per task (Clause 15 b)."""
    with pytest.raises(ValueError, match="'tasks' is empty"):
        ExposureResult(
            lex_8h=85.0,
            combined_standard_uncertainty=1.0,
            expanded_uncertainty=1.65,
            strategy="task",
        )


def test_result_dataclasses_are_frozen() -> None:
    result = job_based_exposure([80.0, 82.0, 81.0, 83.0, 80.0], 8.0)
    with pytest.raises(AttributeError):
        result.lex_8h = 0.0  # type: ignore[misc]
    tc = TaskContribution(
        label="x",
        lp_aeqt=85.0,
        duration_hours=8.0,
        lex_8h_contribution=85.0,
        n_samples=3,
        sample_range_db=0.0,
        spread_advisory=False,
        u1a=0.0,
        c1a=1.0,
        u1b=0.0,
        c1b=0.0,
        u2=0.7,
        u3=1.0,
    )
    with pytest.raises(AttributeError):
        tc.u1a = 1.0  # type: ignore[misc]
    assert isinstance(result, ExposureResult)


def test_per_task_instrument_override() -> None:
    """A per-task instrument class overrides the call-level default u2."""
    result = task_based_exposure(
        [Task(samples=(85.0, 85.0, 85.0), duration_hours=8.0, instrument="class1")],
        instrument="personal_exposimeter",
        warn=False,
    )
    assert result.tasks[0].u2 == 0.7
