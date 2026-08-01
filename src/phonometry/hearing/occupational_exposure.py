#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Occupational noise exposure: measurement strategies and uncertainty (ISO 9612:2009).

ISO 9612:2009 is the engineering method (accuracy grade 2) for determining a
worker's daily noise exposure level ``LEX,8h`` from measurements of the
A-weighted equivalent continuous sound pressure level ``Lp,A,eqT``. The raw
levels themselves come from the dosimetry primitives in
:mod:`phonometry.signals.levels` (:func:`leq`/:func:`lex_8h`); this module adds the three
**measurement strategies**, the energy combination of their contributions, and
the normative **Annex C** uncertainty budget.

**Three strategies (Clause 8):**

- *Task-based* (Clause 9): the nominal day is split into tasks; each task level is
  the energy average of ``I >= 3`` samples (Eq 7), its contribution is
  :math:`L_{EX,8h,m} = L_{p,A,eqT,m} + 10 \log_{10}(T_m/T_0)` (Eq 8), and the
  daily level is the energy sum over tasks (Eq 9/10).
- *Job-based* (Clause 10): ``N >= 5`` random samples over a homogeneous exposure
  group; the effective-day level is their energy average (Eq 11) and
  :math:`L_{EX,8h} = L_{p,A,eqTe} + 10 \log_{10}(T_e/T_0)` (Eq 12). The minimum
  cumulative measurement duration follows Table 1.
- *Full-day* (Clause 11): three (or more) whole-day measurements averaged (Eq 11),
  then Eq 13, the same arithmetic as the job method.

**Uncertainty (Annex C, normative).** Combined
:math:`u^2 = \sum c_i^2 u_i^2` (C.1),
expanded :math:`U = k u` with :math:`k = 1.65` for a one-sided 95 %
confidence interval
(Clause 14). Task-based uses Eq C.3 with sampling ``u1a`` (C.6), duration ``u1b``
(C.7) and sensitivity coefficients ``c1a`` (C.4)/``c1b`` (C.5). Job-based and
full-day use Eq C.9 with :math:`c_1 u_1` read from Table C.4 and
:math:`c_2 = c_3 = 1`. The
instrument standard uncertainty ``u2`` is from Table C.5 and the microphone
position :math:`u_3 = 1.0` dB (C.6). Peak levels ``Lp,Cpeak`` are reported
without an
uncertainty: Annex C gives no method for them (Table C.5, NOTE 1).

The three worked examples of Annexes D (task), E (job) and F (full-day) are
reproduced to the standard's printed precision in the test suite (Annex E rounds
its effective-day level to 88.4 dB before Eq 12 and prints 88.1; the library
keeps intermediates unrounded and yields 88.2). Clause/equation/table numbers
refer to ISO 9612:2009(E).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from math import log10, sqrt
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .._internal.levels_math import energy_mean
from .._internal.warnings import PhonometryWarning, _warn_renamed

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .._report.metadata import ReportMetadata

#: Reference duration T0 = 8 h (Clause 4).
_T0: float = 8.0

#: Coverage factor for a one-sided 95 % confidence interval (Annex C, Clause 14).
COVERAGE_FACTOR: float = 1.65

#: Microphone-position standard uncertainty u3 (Clause C.6), dB.
_U3_DEFAULT: float = 1.0

#: 10/ln(10), which converts a natural-log derivative to dB (the 4.34 of Eq C.5).
_C5_FACTOR: float = 10.0 / np.log(10.0)  # = 4.3429...

InstrumentClass = Literal["class1", "class2", "personal_exposimeter"]

#: Instrument standard uncertainty u2 by class (Table C.5), dB.
INSTRUMENT_U2: dict[str, float] = {
    "class1": 0.7,  # sound level meter IEC 61672-1:2002 class 1
    "class2": 1.5,  # sound level meter IEC 61672-1:2002 class 2
    "personal_exposimeter": 1.5,  # personal sound exposure meter IEC 61252
}


class OccupationalExposureWarning(PhonometryWarning):
    """Advisory raised when an ISO 9612 sampling rule recommends more measurements."""


# --------------------------------------------------------------------------- #
# Table C.4: uncertainty contribution c1*u1 (dB) for job/full-day sampling,
# as a function of the number of samples N and the standard uncertainty u1.
# --------------------------------------------------------------------------- #
_C4_U1_AXIS: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0)
_C4_N_AXIS: tuple[int, ...] = (3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30)
_C4_TABLE: tuple[tuple[float, ...], ...] = (
    (0.6, 1.6, 3.1, 5.2, 8.0, 11.5, 15.7, 20.6, 26.1, 32.2, 39.0, 46.5),  # N=3
    (0.4, 0.9, 1.6, 2.5, 3.6, 5.0, 6.7, 8.6, 10.9, 13.4, 16.1, 19.2),  # N=4
    (0.3, 0.7, 1.2, 1.7, 2.4, 3.3, 4.4, 5.6, 6.9, 8.5, 10.2, 12.1),  # N=5
    (0.3, 0.6, 0.9, 1.4, 1.9, 2.6, 3.3, 4.2, 5.2, 6.3, 7.6, 8.9),  # N=6
    (0.2, 0.5, 0.8, 1.2, 1.6, 2.2, 2.8, 3.5, 4.3, 5.1, 6.1, 7.2),  # N=7
    (0.2, 0.5, 0.7, 1.1, 1.4, 1.9, 2.4, 3.0, 3.6, 4.4, 5.2, 6.1),  # N=8
    (0.2, 0.4, 0.7, 1.0, 1.3, 1.7, 2.1, 2.6, 3.2, 3.9, 4.6, 5.4),  # N=9
    (0.2, 0.4, 0.6, 0.9, 1.2, 1.5, 1.9, 2.4, 2.9, 3.5, 4.1, 4.8),  # N=10
    (0.2, 0.3, 0.5, 0.8, 1.0, 1.3, 1.7, 2.0, 2.5, 2.9, 3.5, 4.0),  # N=12
    (0.1, 0.3, 0.5, 0.7, 0.9, 1.2, 1.5, 1.8, 2.2, 2.6, 3.0, 3.5),  # N=14
    (0.1, 0.3, 0.5, 0.6, 0.8, 1.1, 1.3, 1.6, 2.0, 2.3, 2.7, 3.2),  # N=16
    (0.1, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.1, 2.5, 2.9),  # N=18
    (0.1, 0.3, 0.4, 0.5, 0.7, 0.9, 1.1, 1.4, 1.7, 2.0, 2.3, 2.6),  # N=20
    (0.1, 0.2, 0.3, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.7, 2.0, 2.3),  # N=25
    (0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 2.0),  # N=30
)

#: c1*u1 above this value (bold cells of Table C.4) → revise the measurement plan (10.4).
_C4_ADVISORY_THRESHOLD: float = 3.5


def table_c4_contribution(n_samples: int, u1: float) -> float:
    r"""
    Uncertainty contribution :math:`c_1 u_1` (dB) from Table C.4
    (job/full-day sampling).

    Bilinear interpolation on the sample count ``N`` and the sampling standard
    uncertainty ``u1``. The ``u1`` axis is anchored at the origin (``u1 = 0`` gives
    ``0`` dB) and both axes are clamped to the tabulated range ``[3, 30]`` x
    ``[0, 6]``. Rows ``N = 3`` and ``N = 4`` apply to full-day measurements only
    (ISO 9612:2009 Table C.4, NOTE 1).

    :param n_samples: Number of job/full-day samples ``N`` (>= 2).
    :param u1: Standard uncertainty of the samples, dB (>= 0).
    :return: The contribution :math:`c_1 u_1` in dB.
    """
    if n_samples < 2:
        raise ValueError("Table C.4 needs at least 2 samples.")
    if u1 < 0:
        raise ValueError("'u1' must be non-negative.")

    n_axis = np.asarray(_C4_N_AXIS, dtype=float)
    u_axis = np.asarray((0.0,) + _C4_U1_AXIS, dtype=float)
    # Prepend the origin column (u1 = 0 -> 0 dB) to every row.
    table = np.column_stack([np.zeros(len(_C4_N_AXIS)), np.asarray(_C4_TABLE, dtype=float)])

    n_clamped = float(np.clip(n_samples, n_axis[0], n_axis[-1]))
    u_clamped = float(np.clip(u1, u_axis[0], u_axis[-1]))

    # Interpolate along u1 for every tabulated N, then along N.
    per_row = np.array([np.interp(u_clamped, u_axis, table[r]) for r in range(len(n_axis))])
    return float(np.interp(n_clamped, n_axis, per_row))


def minimum_cumulative_duration_hours(n_workers: int) -> float:
    """
    Minimum cumulative measurement duration (h) for a homogeneous exposure group (Table 1).

    :param n_workers: Number of workers ``n_G`` in the homogeneous exposure group.
    :return: The minimum cumulative measurement duration in hours. For
        ``n_G > 40`` the standard advises splitting the group; the returned 17 h
        is the tabulated fallback.
    """
    if n_workers < 1:
        raise ValueError("'n_workers' must be a positive integer.")
    if n_workers <= 5:
        return 5.0
    if n_workers <= 15:
        return 5.0 + (n_workers - 5) * 0.5
    if n_workers <= 40:
        return 10.0 + (n_workers - 15) * 0.25
    return 17.0


# --------------------------------------------------------------------------- #
# Energy helpers.
# --------------------------------------------------------------------------- #
def _sampling_std(levels: Sequence[float]) -> float:
    """Sample standard deviation about the arithmetic mean (Eq C.12), dB.

    Used for the job/full-day sampling uncertainty ``u1`` (denominator ``N-1``).
    """
    arr = np.asarray(levels, dtype=float)
    if arr.size < 2:
        return 0.0
    return float(np.std(arr, ddof=1))


def _task_sampling_uncertainty(levels: Sequence[float]) -> float:
    """Task noise-sampling standard uncertainty ``u1a`` (Eq C.6), dB.

    Eq C.6 divides the sum of squared deviations by ``I(I-1)``, i.e. it is the
    standard error of the mean (sample standard deviation / sqrt(I)), which is
    smaller than the Eq C.12 sample standard deviation used for the job method.
    """
    arr = np.asarray(levels, dtype=float)
    if arr.size < 2:
        return 0.0
    return float(np.std(arr, ddof=1) / sqrt(arr.size))


# --------------------------------------------------------------------------- #
# Result dataclasses.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Task:
    """
    One task of a task-based measurement (ISO 9612:2009 Clause 9).

    :param samples: Measured ``Lp,A,eqT,mi`` levels for the task, dB. At least
        three are recommended (Clause 9.3); a single conservative value is
        allowed for negligible tasks (its ``u1a`` is then zero).
    :param duration_hours: Arithmetic mean task duration ``T_m`` (Eq 5), hours.
    :param duration_samples: Optional independent duration observations ``T_m,j``
        (hours) used for ``u1b`` via Eq C.7.
    :param duration_range: Optional ``(T_min, T_max)`` range (hours); ``u1b`` is
        then ``0.5*(T_max - T_min)`` (Eq C.7 NOTE). Ignored if
        ``duration_samples`` is given.
    :param label: Optional human-readable task name.
    :param instrument: Optional per-task instrument class overriding the call
        default (selects ``u2`` from Table C.5).
    """

    samples: tuple[float, ...]
    duration_hours: float
    duration_samples: tuple[float, ...] | None = None
    duration_range: tuple[float, float] | None = None
    label: str | None = None
    instrument: InstrumentClass | None = None

    def __post_init__(self) -> None:
        if len(self.samples) == 0:
            raise ValueError("A Task needs at least one Lp,A,eqT sample.")
        if self.duration_hours <= 0.0:
            raise ValueError("'duration_hours' must be positive.")


@dataclass(frozen=True)
class TaskContribution:
    """Per-task results and uncertainty terms of a task-based determination."""

    label: str
    lp_aeqt: float  # Eq 7, energy average of the task samples, dB
    duration_hours: float  # T_m, hours
    lex_8h_contribution: float  # Eq 8, dB
    n_samples: int
    sample_range_db: float  # max - min of the samples, dB
    spread_advisory: bool  # True when the 3 dB spread rule (9.3) is triggered
    u1a: float  # sampling standard uncertainty (Eq C.6), dB
    c1a: float  # noise sensitivity coefficient (Eq C.4)
    u1b: float  # duration standard uncertainty (Eq C.7), hours
    c1b: float  # duration sensitivity coefficient (Eq C.5), dB/h
    u2: float  # instrument standard uncertainty (Table C.5), dB
    u3: float  # microphone-position standard uncertainty (Clause C.6), dB

    @property
    def variance_contribution(self) -> float:
        r"""This task's contribution to :math:`u^2(L_{EX,8h})` (a term of
        Eq C.3), dB²."""
        return self.c1a**2 * (self.u1a**2 + self.u2**2 + self.u3**2) + (self.c1b * self.u1b) ** 2


@dataclass(frozen=True)
class ExposureResult:
    r"""
    Daily noise exposure level and its expanded uncertainty (ISO 9612:2009).

    :ivar lex_8h: A-weighted daily noise exposure level ``LEX,8h``, dB.
    :ivar combined_standard_uncertainty: Combined standard uncertainty ``u``
        (Eq C.1), dB.
    :ivar expanded_uncertainty: Expanded uncertainty :math:`U = 1.65 u` for a
        one-sided 95 % confidence interval, dB.
    :ivar strategy: ``"task"``, ``"job"`` or ``"full_day"``.
    :ivar instrument: Instrument class the measurement was made with (the call
        default; individual tasks may override it): ``"class1"``, ``"class2"``
        or ``"personal_exposimeter"``. Printed on the ``.report()`` fiche
        (ISO 9612:2009 Clause 15 c).
    :ivar upper_limit: :math:`L_{EX,8h} + U`, the value 95 % of readings fall
        below.
    """

    lex_8h: float
    combined_standard_uncertainty: float
    expanded_uncertainty: float
    strategy: str
    coverage_factor: float = COVERAGE_FACTOR
    lp_aeqte: float | None = None  # energy-averaged level (job/full-day), dB
    effective_duration_hours: float | None = None  # T_e, hours
    u1: float | None = None  # sampling std (job/full-day), dB
    c1u1: float | None = None  # Table C.4 contribution (job/full-day), dB
    u2: float | None = None
    u3: float | None = None
    n_samples: int | None = None
    sampling_advisory: bool = False  # c1*u1 > 3.5 dB, or 3 dB spread on 3 samples
    instrument: InstrumentClass | None = None
    tasks: tuple[TaskContribution, ...] = field(default_factory=tuple)

    @property
    def upper_limit(self) -> float:
        """Upper limit ``LEX,8h + U`` of the one-sided 95 % interval, dB."""
        return self.lex_8h + self.expanded_uncertainty

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the per-task contributions with the ``LEX,8h`` line.

        Only task-based results carry per-task contributions (the job and
        full-day strategies raise :class:`ValueError`). Requires matplotlib
        (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.hearing import plot_occupational_exposure

        return plot_occupational_exposure(self, ax=ax, language=check_language(language), **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render an ISO 9612 occupational noise-exposure fiche to a PDF.

        Writes a one-page measurement report with the information ISO 9612:2009
        Clause 15 asks for: the standard-basis line naming the applied strategy,
        a metadata header (company, worker/job, workplace, instrumentation,
        calibration), the work analysis (the per-task table of durations,
        ``Lp,A,eqT,m`` levels and ``LEX,8h,m`` contributions for the task-based
        strategy, or the sampling summary for the job-based/full-day
        strategies), the per-task contribution chart for a task-based result,
        the boxed ``LEX,8h`` with its expanded uncertainty ``U`` stated
        separately (Clause 15 e), an assessment table against the exposure
        action values (80/85 dB(A)) and the exposure limit value (87 dB(A)) of
        Directive 2003/10/EC, a PASS/FAIL verdict against the limit value, and
        a footer identity/disclaimer block.

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata` supplying
            the header identity (``client`` is the company, ``specimen`` the
            worker(s)/job, ``test_room`` the workplace) plus the
            ``instrumentation`` and ``calibration`` free-text fields and the
            footer identity.
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: When True, the task-based work-analysis table adds the
            Annex C per-task uncertainty columns (``u1a``, ``u1b``, ``u2``).
            The job-based/full-day sampling summary always shows its budget.
        :param language: Fiche language: ``"en"`` (default, English) or
            ``"es"`` (Spanish, with a comma decimal separator).
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is not ``"reportlab"`` or ``language``
            is unknown.
        :raises ImportError: If reportlab (or, for a task-based result's chart,
            matplotlib) is not installed (``pip install phonometry[report]``).
        """
        from .._i18n import check_language

        check_language(language)
        if engine != "reportlab":
            raise ValueError(
                f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            )
        from .._report.iso9612 import render_iso9612_report

        return render_iso9612_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


# --------------------------------------------------------------------------- #
# Strategy 1: task-based (Clauses 9 and C.2).
# --------------------------------------------------------------------------- #
def task_based_exposure(
    tasks: Sequence[Task],
    instrument: InstrumentClass = "personal_exposimeter",
    u3: float = _U3_DEFAULT,
    include_duration_uncertainty: bool = True,
    warn: bool = True,
) -> ExposureResult:
    """
    Daily noise exposure level from task-based measurements (ISO 9612:2009 Clause 9).

    Each task level is the energy average of its samples (Eq 7); the daily level
    is the energy sum of the task contributions (Eq 9/10). The uncertainty budget
    follows Eq C.3 with sampling ``u1a`` (Eq C.6), optional duration ``u1b``
    (Eq C.7), and sensitivity coefficients ``c1a`` (Eq C.4) and ``c1b`` (Eq C.5).

    :param tasks: The tasks making up the nominal day.
    :param instrument: Default instrument class selecting ``u2`` (Table C.5);
        may be overridden per :class:`Task`.
    :param u3: Microphone-position standard uncertainty, dB (Clause C.6 gives 1.0).
    :param include_duration_uncertainty: Include the :math:`(c_{1b} u_{1b})^2`
        duration term (Eq C.3). When False the budget omits it (ISO 9612
        Annex D case a).
    :param warn: Emit :class:`OccupationalExposureWarning` when a task triggers the 3 dB
        spread rule (Clause 9.3).
    :return: An :class:`ExposureResult` with per-task contributions.
    """
    if not tasks:
        raise ValueError("At least one task is required.")
    if u3 < 0:
        raise ValueError("'u3' must be non-negative.")

    # First pass: task levels, durations and contributions (Eq 7, 8).
    levels: list[float] = []
    durations: list[float] = []
    for task in tasks:
        if len(task.samples) == 0:
            raise ValueError("Each task needs at least one sample.")
        if task.duration_hours <= 0:
            raise ValueError("Task 'duration_hours' must be positive.")
        levels.append(energy_mean(task.samples))  # Eq 7
        durations.append(task.duration_hours)

    # Daily level: energy sum of contributions (Eq 9 == Eq 10).
    energy = sum((t_m / _T0) * 10.0 ** (0.1 * lp) for lp, t_m in zip(levels, durations))
    lex_8h = float(10.0 * log10(energy))

    # Second pass: sensitivity coefficients and uncertainty terms.
    contributions: list[TaskContribution] = []
    for idx, (task, lp, t_m) in enumerate(zip(tasks, levels, durations)):
        samples = np.asarray(task.samples, dtype=float)
        n = int(samples.size)
        sample_range = float(samples.max() - samples.min()) if n >= 2 else 0.0
        spread = n >= 2 and sample_range >= 3.0

        u1a = _task_sampling_uncertainty(task.samples)
        # c1a (Eq C.4): (T_m/T0) * 10^(0.1 (Lp - LEX,8h)); L* ~ Lp since Q2,Q3 ~ 0.
        c1a = (t_m / _T0) * 10.0 ** (0.1 * (lp - lex_8h))
        # c1b (Eq C.5): 4.34 * c1a / T_m.
        c1b = _C5_FACTOR * c1a / t_m if include_duration_uncertainty else 0.0

        u1b = 0.0
        if include_duration_uncertainty:
            if task.duration_samples is not None:
                j = np.asarray(task.duration_samples, dtype=float)
                if j.size >= 2:
                    # Eq C.7 divides by J(J-1), like Eq C.6: standard error of the mean.
                    u1b = float(np.std(j, ddof=1) / sqrt(j.size))
            elif task.duration_range is not None:
                t_min, t_max = task.duration_range
                if t_max < t_min:
                    raise ValueError("duration_range must be (T_min, T_max) with T_max >= T_min.")
                u1b = 0.5 * (t_max - t_min)

        u2 = INSTRUMENT_U2[task.instrument] if task.instrument is not None else INSTRUMENT_U2[instrument]

        label = task.label if task.label is not None else f"task {idx + 1}"
        if spread and warn:
            warnings.warn(
                f"Task '{label}': the {n} measurements span {sample_range:.1f} dB "
                "(>= 3 dB); ISO 9612 Clause 9.3 recommends additional measurements, "
                "subdivision, or longer measurement durations.",
                OccupationalExposureWarning,
                stacklevel=2,
            )
        contributions.append(
            TaskContribution(
                label=label,
                lp_aeqt=lp,
                duration_hours=t_m,
                lex_8h_contribution=lp + 10.0 * log10(t_m / _T0),
                n_samples=n,
                sample_range_db=sample_range,
                spread_advisory=spread,
                u1a=u1a,
                c1a=c1a,
                u1b=u1b,
                c1b=c1b,
                u2=u2,
                u3=u3,
            )
        )

    variance = sum(c.variance_contribution for c in contributions)
    u = sqrt(variance)
    return ExposureResult(
        lex_8h=lex_8h,
        combined_standard_uncertainty=u,
        expanded_uncertainty=COVERAGE_FACTOR * u,
        strategy="task",
        u3=u3,
        sampling_advisory=any(c.spread_advisory for c in contributions),
        instrument=instrument,
        tasks=tuple(contributions),
    )


# --------------------------------------------------------------------------- #
# Strategies 2 and 3: job-based (Clauses 10, C.3) and full-day (Clauses 11, C.4).
# --------------------------------------------------------------------------- #
def _sampled_exposure(
    samples: Sequence[float],
    effective_duration_hours: float,
    instrument: InstrumentClass,
    u3: float,
    strategy: str,
    warn: bool,
    spread_advisory: bool,
) -> ExposureResult:
    """Shared engine for job-based and full-day strategies (Eq 11-13, Eq C.9)."""
    arr = np.asarray(samples, dtype=float)
    n = int(arr.size)
    if n < 2:
        raise ValueError("At least two samples are required.")
    if effective_duration_hours <= 0:
        raise ValueError("'effective_duration_hours' must be positive.")
    if u3 < 0:
        raise ValueError("'u3' must be non-negative.")

    lp_aeqte = energy_mean(samples)  # Eq 11
    lex_8h = lp_aeqte + 10.0 * log10(effective_duration_hours / _T0)  # Eq 12/13

    u1 = _sampling_std(samples)  # Eq C.12
    c1u1 = table_c4_contribution(n, u1)  # Table C.4
    u2 = INSTRUMENT_U2[instrument]
    # Eq C.9 with c2 = c3 = 1.
    u = sqrt(c1u1**2 + u2**2 + u3**2)

    advisory = c1u1 > _C4_ADVISORY_THRESHOLD or spread_advisory
    if warn and c1u1 > _C4_ADVISORY_THRESHOLD:
        warnings.warn(
            f"Job/full-day sampling contribution c1*u1 = {c1u1:.1f} dB exceeds "
            "3.5 dB (ISO 9612 Table C.4 / Clause 10.4); revise the exposure group "
            "or increase the number of measurements.",
            OccupationalExposureWarning,
            stacklevel=3,
        )
    return ExposureResult(
        lex_8h=lex_8h,
        combined_standard_uncertainty=u,
        expanded_uncertainty=COVERAGE_FACTOR * u,
        strategy=strategy,
        lp_aeqte=lp_aeqte,
        effective_duration_hours=effective_duration_hours,
        u1=u1,
        c1u1=c1u1,
        u2=u2,
        u3=u3,
        n_samples=n,
        sampling_advisory=advisory,
        instrument=instrument,
    )


def job_based_exposure(
    samples: Sequence[float],
    effective_duration_hours: float,
    instrument: InstrumentClass = "personal_exposimeter",
    u3: float = _U3_DEFAULT,
    n_workers: int | None = None,
    sample_duration_hours: float | None = None,
    warn: bool = True,
) -> ExposureResult:
    r"""
    Daily noise exposure level from job-based measurements (ISO 9612:2009 Clause 10).

    The effective-day level is the energy average of ``N >= 5`` random job samples
    (Eq 11); the daily level follows Eq 12. The sampling uncertainty ``u1`` is the
    sample standard deviation (Eq C.12) and its contribution :math:`c_1 u_1`
    is read from Table C.4; the combined uncertainty is Eq C.9 with
    :math:`c_2 = c_3 = 1`.

    :param samples: Measured ``Lp,A,eqT,n`` job samples, dB (at least five per
        Clause 10.2; a minimum of two is enforced numerically).
    :param effective_duration_hours: Effective working-day duration ``T_e``, hours.
    :param instrument: Instrument class selecting ``u2`` (Table C.5).
    :param u3: Microphone-position standard uncertainty, dB (Clause C.6 gives 1.0).
    :param n_workers: Optional homogeneous-group size ``n_G``; when given with
        ``sample_duration_hours``, the cumulative duration is checked against
        Table 1 and an advisory is raised if it falls short.
    :param sample_duration_hours: Optional per-sample duration (hours) for the
        Table 1 cumulative-duration check.
    :param warn: Emit :class:`OccupationalExposureWarning` for the Table C.4 / Table 1 advisories.
    :return: An :class:`ExposureResult`.
    """
    if sample_duration_hours is not None and sample_duration_hours <= 0.0:
        raise ValueError("'sample_duration_hours' must be positive.")
    result = _sampled_exposure(
        samples, effective_duration_hours, instrument, u3, "job", warn, spread_advisory=False
    )
    if n_workers is not None and sample_duration_hours is not None:
        required = minimum_cumulative_duration_hours(n_workers)
        cumulative = len(samples) * sample_duration_hours
        if cumulative < required - 1e-9:
            if warn:
                warnings.warn(
                    f"Cumulative measurement duration {cumulative:.2f} h is below the "
                    f"Table 1 minimum of {required:.2f} h for {n_workers} workers.",
                    OccupationalExposureWarning,
                    stacklevel=2,
                )
            result = _with_advisory(result)
    return result


def full_day_exposure(
    samples: Sequence[float],
    effective_duration_hours: float,
    instrument: InstrumentClass = "personal_exposimeter",
    u3: float = _U3_DEFAULT,
    warn: bool = True,
) -> ExposureResult:
    """
    Daily noise exposure level from full-day measurements (ISO 9612:2009 Clause 11).

    Three (or more) whole-day ``Lp,A,eqT`` measurements are energy-averaged (Eq 11)
    and the daily level follows Eq 13. If only three measurements are supplied and
    they span 3 dB or more, Clause 11.3 requires at least two more; this raises the
    sampling advisory. The uncertainty is the job-based budget (Eq C.9, Table C.4).

    :param samples: Whole-day ``Lp,A,eqT,n`` measurements, dB (three minimum per
        Clause 11.3; two is enforced numerically).
    :param effective_duration_hours: Effective working-day duration ``T_e``, hours.
    :param instrument: Instrument class selecting ``u2`` (Table C.5).
    :param u3: Microphone-position standard uncertainty, dB (Clause C.6 gives 1.0).
    :param warn: Emit :class:`OccupationalExposureWarning` for the 3 dB spread rule and the
        Table C.4 advisory.
    :return: An :class:`ExposureResult`.
    """
    arr = np.asarray(samples, dtype=float)
    spread = arr.size == 3 and float(arr.max() - arr.min()) >= 3.0
    if spread and warn:
        warnings.warn(
            f"Only three full-day measurements spanning {float(arr.max() - arr.min()):.1f} dB "
            "(>= 3 dB); ISO 9612 Clause 11.3 requires at least two additional measurements.",
            OccupationalExposureWarning,
            stacklevel=2,
        )
    return _sampled_exposure(
        samples, effective_duration_hours, instrument, u3, "full_day", warn, spread_advisory=spread
    )


def _with_advisory(result: ExposureResult) -> ExposureResult:
    """Return a copy of ``result`` with the sampling advisory flag set."""
    return replace(result, sampling_advisory=True)


# --- Deprecated alias (phonometry 3.1 rename; remove in 4.0) -------------

def __getattr__(name: str) -> Any:
    """PEP 562 shim warning for the renamed warning class.

    Returns the class object itself, so ``isinstance``/``except`` checks and
    warning filters against the old name keep matching the new one.
    """
    if name == "ExposureWarning":
        _warn_renamed("ExposureWarning", "OccupationalExposureWarning")
        return OccupationalExposureWarning
    raise AttributeError(
        f"module 'phonometry.occupational_exposure' has no attribute {name!r}"
    )
