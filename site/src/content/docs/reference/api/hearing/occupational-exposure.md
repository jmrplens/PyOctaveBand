---
title: "hearing.occupational_exposure"
description: "Occupational noise exposure: measurement strategies and uncertainty (ISO 9612:2009)."
sidebar:
  label: "occupational_exposure"
---

Occupational noise exposure: measurement strategies and uncertainty (ISO 9612:2009).

ISO 9612:2009 is the engineering method (accuracy grade 2) for determining a
worker's daily noise exposure level `LEX,8h` from measurements of the
A-weighted equivalent continuous sound pressure level `Lp,A,eqT`. The raw
levels themselves come from the dosimetry primitives in
[`phonometry.levels`](/phonometry/reference/api/levels/levels/) ([`leq`](/phonometry/reference/api/levels/levels/#leq)/[`lex_8h`](/phonometry/reference/api/levels/levels/#lex_8h)); this module adds the three
**measurement strategies**, the energy combination of their contributions, and
the normative **Annex C** uncertainty budget.

**Three strategies (Clause 8):**

- *Task-based* (Clause 9): the nominal day is split into tasks; each task level is
  the energy average of `I >= 3` samples (Eq 7), its contribution is
  $L_{EX,8h,m} = L_{p,A,eqT,m} + 10 \lg(T_m/T_0)$ (Eq 8), and the
  daily level is the energy sum over tasks (Eq 9/10).
- *Job-based* (Clause 10): `N >= 5` random samples over a homogeneous exposure
  group; the effective-day level is their energy average (Eq 11) and
  $L_{EX,8h} = L_{p,A,eqTe} + 10 \lg(T_e/T_0)$ (Eq 12). The minimum
  cumulative measurement duration follows Table 1.
- *Full-day* (Clause 11): three (or more) whole-day measurements averaged (Eq 11),
  then Eq 13, the same arithmetic as the job method.

**Uncertainty (Annex C, normative).** Combined
$u^2 = \sum c_i^2 u_i^2$ (C.1),
expanded $U = k u$ with $k = 1.65$ for a one-sided 95 %
confidence interval
(Clause 14). Task-based uses Eq C.3 with sampling `u1a` (C.6), duration `u1b`
(C.7) and sensitivity coefficients `c1a` (C.4)/`c1b` (C.5). Job-based and
full-day use Eq C.9 with $c_1 u_1$ read from Table C.4 and
$c_2 = c_3 = 1$. The
instrument standard uncertainty `u2` is from Table C.5 and the microphone
position $u_3 = 1.0$ dB (C.6). Peak levels `Lp,Cpeak` are reported
without an
uncertainty: Annex C gives no method for them (Table C.5, NOTE 1).

The three worked examples of Annexes D (task), E (job) and F (full-day) are
reproduced to the standard's printed precision in the test suite (Annex E rounds
its effective-day level to 88.4 dB before Eq 12 and prints 88.1; the library
keeps intermediates unrounded and yields 88.2). Clause/equation/table numbers
refer to ISO 9612:2009(E).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## COVERAGE_FACTOR

*Constant* (`float`).

```python
COVERAGE_FACTOR = 1.65
```

## ExposureResult

```python
ExposureResult(
    lex_8h: float,
    combined_standard_uncertainty: float,
    expanded_uncertainty: float,
    strategy: str,
    coverage_factor: float = 1.65,
    lp_aeqte: float | None = None,
    effective_duration_hours: float | None = None,
    u1: float | None = None,
    c1u1: float | None = None,
    u2: float | None = None,
    u3: float | None = None,
    n_samples: int | None = None,
    sampling_advisory: bool = False,
    instrument: InstrumentClass | None = None,
    tasks: tuple[TaskContribution, ...] = ...,
)
```

Daily noise exposure level and its expanded uncertainty (ISO 9612:2009).

**Attributes**

| Name | Description |
| :--- | :--- |
| `lex_8h` | A-weighted daily noise exposure level `LEX,8h`, dB. |
| `combined_standard_uncertainty` | Combined standard uncertainty `u` (Eq C.1), dB. |
| `expanded_uncertainty` | Expanded uncertainty $U = 1.65 u$ for a one-sided 95 % confidence interval, dB. |
| `strategy` | `"task"`, `"job"` or `"full_day"`. |
| `instrument` | Instrument class the measurement was made with (the call default; individual tasks may override it): `"class1"`, `"class2"` or `"personal_exposimeter"`. Printed on the `.report()` fiche (ISO 9612:2009 Clause 15 c). |
| `upper_limit` | $L_{EX,8h} + U$, the value 95 % of readings fall below. |

### ExposureResult.plot()

```python
ExposureResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the per-task contributions with the `LEX,8h` line.

Only task-based results carry per-task contributions (the job and
full-day strategies raise `ValueError`). Requires matplotlib
(`pip install phonometry[plot]`); returns the
`Axes`.

### ExposureResult.report()

```python
ExposureResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render an ISO 9612 occupational noise-exposure fiche to a PDF.

Writes a one-page measurement report with the information ISO 9612:2009
Clause 15 asks for: the standard-basis line naming the applied strategy,
a metadata header (company, worker/job, workplace, instrumentation,
calibration), the work analysis (the per-task table of durations,
`Lp,A,eqT,m` levels and `LEX,8h,m` contributions for the task-based
strategy, or the sampling summary for the job-based/full-day
strategies), the per-task contribution chart for a task-based result,
the boxed `LEX,8h` with its expanded uncertainty `U` stated
separately (Clause 15 e), an assessment table against the exposure
action values (80/85 dB(A)) and the exposure limit value (87 dB(A)) of
Directive 2003/10/EC, a PASS/FAIL verdict against the limit value, and
a footer identity/disclaimer block.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata) supplying the header identity (`client` is the company, `specimen` the worker(s)/job, `test_room` the workplace) plus the `instrumentation` and `calibration` free-text fields and the footer identity. |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When True, the task-based work-analysis table adds the Annex C per-task uncertainty columns (`u1a`, `u1b`, `u2`). The job-based/full-day sampling summary always shows its budget. |
| `language` | Fiche language: `"en"` (default, English) or `"es"` (Spanish, with a comma decimal separator). |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"` or `language` is unknown. |
| ImportError | If reportlab (or, for a task-based result's chart, matplotlib) is not installed (`pip install phonometry[report]`). |

### ExposureResult.upper_limit

*property*

Upper limit `LEX,8h + U` of the one-sided 95 % interval, dB.

## full_day_exposure

```python
full_day_exposure(
    samples: Sequence[float],
    effective_duration_hours: float,
    instrument: InstrumentClass = 'personal_exposimeter',
    u3: float = 1.0,
    warn: bool = True,
) -> ExposureResult
```

Daily noise exposure level from full-day measurements (ISO 9612:2009 Clause 11).

Three (or more) whole-day `Lp,A,eqT` measurements are energy-averaged (Eq 11)
and the daily level follows Eq 13. If only three measurements are supplied and
they span 3 dB or more, Clause 11.3 requires at least two more; this raises the
sampling advisory. The uncertainty is the job-based budget (Eq C.9, Table C.4).

**Parameters**

| Name | Description |
| :--- | :--- |
| `samples` | Whole-day `Lp,A,eqT,n` measurements, dB (three minimum per Clause 11.3; two is enforced numerically). |
| `effective_duration_hours` | Effective working-day duration `T_e`, hours. |
| `instrument` | Instrument class selecting `u2` (Table C.5). |
| `u3` | Microphone-position standard uncertainty, dB (Clause C.6 gives 1.0). |
| `warn` | Emit [`OccupationalExposureWarning`](/phonometry/reference/api/hearing/occupational-exposure/#occupationalexposurewarning) for the 3 dB spread rule and the Table C.4 advisory. |

**Returns:** An [`ExposureResult`](/phonometry/reference/api/hearing/occupational-exposure/#exposureresult).

## INSTRUMENT_U2

*Constant* (`dict`).

```python
INSTRUMENT_U2 = {'class1': 0.7, 'class2': 1.5, 'personal_exposimeter': 1.5}
```

## job_based_exposure

```python
job_based_exposure(
    samples: Sequence[float],
    effective_duration_hours: float,
    instrument: InstrumentClass = 'personal_exposimeter',
    u3: float = 1.0,
    n_workers: int | None = None,
    sample_duration_hours: float | None = None,
    warn: bool = True,
) -> ExposureResult
```

Daily noise exposure level from job-based measurements (ISO 9612:2009 Clause 10).

The effective-day level is the energy average of `N >= 5` random job samples
(Eq 11); the daily level follows Eq 12. The sampling uncertainty `u1` is the
sample standard deviation (Eq C.12) and its contribution $c_1 u_1$
is read from Table C.4; the combined uncertainty is Eq C.9 with
$c_2 = c_3 = 1$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `samples` | Measured `Lp,A,eqT,n` job samples, dB (at least five per Clause 10.2; a minimum of two is enforced numerically). |
| `effective_duration_hours` | Effective working-day duration `T_e`, hours. |
| `instrument` | Instrument class selecting `u2` (Table C.5). |
| `u3` | Microphone-position standard uncertainty, dB (Clause C.6 gives 1.0). |
| `n_workers` | Optional homogeneous-group size `n_G`; when given with `sample_duration_hours`, the cumulative duration is checked against Table 1 and an advisory is raised if it falls short. |
| `sample_duration_hours` | Optional per-sample duration (hours) for the Table 1 cumulative-duration check. |
| `warn` | Emit [`OccupationalExposureWarning`](/phonometry/reference/api/hearing/occupational-exposure/#occupationalexposurewarning) for the Table C.4 / Table 1 advisories. |

**Returns:** An [`ExposureResult`](/phonometry/reference/api/hearing/occupational-exposure/#exposureresult).

## minimum_cumulative_duration_hours

```python
minimum_cumulative_duration_hours(n_workers: int) -> float
```

Minimum cumulative measurement duration (h) for a homogeneous exposure group (Table 1).

**Parameters**

| Name | Description |
| :--- | :--- |
| `n_workers` | Number of workers `n_G` in the homogeneous exposure group. |

**Returns:** The minimum cumulative measurement duration in hours. For `n_G > 40` the standard advises splitting the group; the returned 17 h is the tabulated fallback.

## OccupationalExposureWarning

Advisory raised when an ISO 9612 sampling rule recommends more measurements.

## table_c4_contribution

```python
table_c4_contribution(n_samples: int, u1: float) -> float
```

Uncertainty contribution $c_1 u_1$ (dB) from Table C.4
(job/full-day sampling).

Bilinear interpolation on the sample count `N` and the sampling standard
uncertainty `u1`. The `u1` axis is anchored at the origin (`u1 = 0` gives
`0` dB) and both axes are clamped to the tabulated range `[3, 30]` x
`[0, 6]`. Rows `N = 3` and `N = 4` apply to full-day measurements only
(ISO 9612:2009 Table C.4, NOTE 1).

**Parameters**

| Name | Description |
| :--- | :--- |
| `n_samples` | Number of job/full-day samples `N` (>= 2). |
| `u1` | Standard uncertainty of the samples, dB (>= 0). |

**Returns:** The contribution $c_1 u_1$ in dB.

## Task

```python
Task(
    samples: tuple[float, ...],
    duration_hours: float,
    duration_samples: tuple[float, ...] | None = None,
    duration_range: tuple[float, float] | None = None,
    label: str | None = None,
    instrument: InstrumentClass | None = None,
)
```

One task of a task-based measurement (ISO 9612:2009 Clause 9).

**Parameters**

| Name | Description |
| :--- | :--- |
| `samples` | Measured `Lp,A,eqT,mi` levels for the task, dB. At least three are recommended (Clause 9.3); a single conservative value is allowed for negligible tasks (its `u1a` is then zero). |
| `duration_hours` | Arithmetic mean task duration `T_m` (Eq 5), hours. |
| `duration_samples` | Optional independent duration observations `T_m,j` (hours) used for `u1b` via Eq C.7. |
| `duration_range` | Optional `(T_min, T_max)` range (hours); `u1b` is then `0.5*(T_max - T_min)` (Eq C.7 NOTE). Ignored if `duration_samples` is given. |
| `label` | Optional human-readable task name. |
| `instrument` | Optional per-task instrument class overriding the call default (selects `u2` from Table C.5). |

## task_based_exposure

```python
task_based_exposure(
    tasks: Sequence[Task],
    instrument: InstrumentClass = 'personal_exposimeter',
    u3: float = 1.0,
    include_duration_uncertainty: bool = True,
    warn: bool = True,
) -> ExposureResult
```

Daily noise exposure level from task-based measurements (ISO 9612:2009 Clause 9).

Each task level is the energy average of its samples (Eq 7); the daily level
is the energy sum of the task contributions (Eq 9/10). The uncertainty budget
follows Eq C.3 with sampling `u1a` (Eq C.6), optional duration `u1b`
(Eq C.7), and sensitivity coefficients `c1a` (Eq C.4) and `c1b` (Eq C.5).

**Parameters**

| Name | Description |
| :--- | :--- |
| `tasks` | The tasks making up the nominal day. |
| `instrument` | Default instrument class selecting `u2` (Table C.5); may be overridden per [`Task`](/phonometry/reference/api/hearing/occupational-exposure/#task). |
| `u3` | Microphone-position standard uncertainty, dB (Clause C.6 gives 1.0). |
| `include_duration_uncertainty` | Include the `(c1b*u1b)^2` duration term (Eq C.3). When False the budget omits it (ISO 9612 Annex D case a). |
| `warn` | Emit [`OccupationalExposureWarning`](/phonometry/reference/api/hearing/occupational-exposure/#occupationalexposurewarning) when a task triggers the 3 dB spread rule (Clause 9.3). |

**Returns:** An [`ExposureResult`](/phonometry/reference/api/hearing/occupational-exposure/#exposureresult) with per-task contributions.

## TaskContribution

```python
TaskContribution(
    label: str,
    lp_aeqt: float,
    duration_hours: float,
    lex_8h_contribution: float,
    n_samples: int,
    sample_range_db: float,
    spread_advisory: bool,
    u1a: float,
    c1a: float,
    u1b: float,
    c1b: float,
    u2: float,
    u3: float,
)
```

Per-task results and uncertainty terms of a task-based determination.

### TaskContribution.variance_contribution

*property*

This task's contribution to $u^2(L_{EX,8h})$ (a term of
Eq C.3), dB².
