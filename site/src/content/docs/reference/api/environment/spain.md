---
title: "environment.assessment.spain"
description: "Spanish noise regulation: the corrected level LKeq (Real Decreto 1367/2007)."
sidebar:
  label: "spain"
---

Spanish noise regulation: the corrected level LKeq (Real Decreto 1367/2007).

Real Decreto 1367/2007 develops Ley 37/2003 del Ruido on acoustic zoning,
quality objectives and emitter limit values. Its assessment chain is built on
one index the ISO 1996 family does not define: the **corrected equivalent
continuous level** $L_{\mathrm{Keq},T} = L_{\mathrm{Aeq},T} + K_\mathrm{t} + K_\mathrm{f} + K_\mathrm{i}$
(Annex I A.2 c), where the
three corrections penalise emergent tonal components, low-frequency components
and impulsive character. Each is 0, 3 or 6 dB and their sum is capped at 9 dB
(Annex IV A.3.3).

**Corrections (Annex IV A.3.3).** The reference procedures are:

* `Kt`: unweighted one-third-octave analysis; for the band `f` holding the
  tone, $L_\mathrm{t} = L_f - L_s$ with `Ls` the *arithmetic* mean of the two
  adjacent
  band levels. `Kt` is 0/3/6 dB by the thresholds 8/12 dB (20 to 125 Hz),
  5/8 dB (160 to 400 Hz) and 3/5 dB (500 Hz to 10 kHz); with several emergent
  tones the largest applies.
* `Kf`: $L_f = L_{\mathrm{Ceq},Ti} - L_{\mathrm{Aeq},Ti}$ (background-corrected),
  giving 0 dB for
  $L_f \le 10$, 3 dB for $10 < L_f \le 15$ and 6 dB above.
* `Ki`: $L_\mathrm{i} = L_{\mathrm{AIeq},Ti} - L_{\mathrm{Aeq},Ti}$ (background-corrected),
  with the same
  0/3/6 dB thresholds as `Kf`.

**Relationship with the ISO 1996 procedures already in the library.** They are
*relatives, not the same procedure*, so this module implements the RD's own
variants rather than delegating:

* [`tonal_audibility`](/phonometry/reference/api/environment/measurement/#tonal_audibility) /
  [`tonal_adjustment`](/phonometry/reference/api/environment/measurement/#tonal_adjustment) are the
  ISO 1996-2 Annex C engineering method: a critical-band audibility
  `ΔLta` mapped to a *continuous* `Kt` in 0 to 6 dB. The RD works on
  one-third-octave band differences and yields 0/3/6 dB only.
  [`tonal_seeking_survey`](/phonometry/reference/api/environment/measurement/#tonal_seeking_survey) is the
  closest relative (ISO 1996-2:2017 Annex K also splits the spectrum at
  125/400 Hz), but it requires the band to exceed *both* neighbours by
  15/8/5 dB and returns a boolean flag, whereas the RD compares against the
  *mean* of the neighbours with 8/5/3 dB thresholds and grades the result.
* [`impulsive_sound_adjustment`](/phonometry/reference/api/environment/impulsive-sound/#impulsive_sound_adjustment)
  is the ISO/PAS 1996-3 onset-rate method on a calibrated time signal. The
  RD's `Ki` is the classic $L_\mathrm{AIeq} - L_\mathrm{Aeq}$ impulse-vs-fast
  difference read
  off a sound level meter.
* No ISO 1996 counterpart exists for `Kf`: the $L_\mathrm{Ceq} - L_\mathrm{Aeq}$
  difference is
  specific to the RD.

**Evaluation periods and integration.** Day 07:00-19:00 (12 h), evening
19:00-23:00 (4 h) and night 23:00-07:00 (8 h) (Annex I A.1). A period whose
emission varies is split into *noise phases* `Ti` of steady level, and the
period level is the energy mean weighted by phase duration (Annex IV
A.3.4.2 b):

$$
L_{\mathrm{Keq},T} = 10 \log_{10}\left[ (1/T) \sum_i T_i \cdot 10^{L_{\mathrm{Keq},Ti}/10} \right]
$$

The result is
rounded by adding 0.5 dB and taking the integer part. The long-term index
`LK,x` is the energy mean of the daily `LKeq,x` over a year (Annex I
A.2 d).

**Limit tables.** Annex II holds the acoustic quality objectives (Table A
outdoor by area type, as amended by RD 1038/2012; Table B indoor by room; Table
C vibration), Annex III the emitter immission limits (Table A1/A2 new road,
rail and airport infrastructure; Table B1 port infrastructure and activities;
Table B2 noise transmitted to acoustically adjacent premises).

**Compliance.** For activities and port infrastructure (Article 25.1 b) the
limits are respected when no annual average `LK,x` exceeds the table value,
no daily `LKeq,x` exceeds it by more than 3 dB and no measured `LKeq,Ti`
exceeds it by more than 5 dB. For the inspection of an activity already in
operation (Article 25.2) only the last two apply.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## ACOUSTIC_AREA_TYPES

*Constant* (`mappingproxy`).

```python
ACOUSTIC_AREA_TYPES = {'e': 'sanitary, educational and cultural land use requiring special protection', 'a': 'residential land use', 'd': 'tertiary land use other than type c', 'c': 'recreational and public-entertainment land use', 'b': 'industrial land use', 'f': 'general transport-infrastructure systems and public facilities'}
```

## activity_limits

```python
activity_limits(area_type: str) -> RegulationLimits
```

Outdoor immission limits for activities and ports (Annex III Table B1).

The `LK,d`/`LK,e`/`LK,n` an installation, establishment or activity
(industrial, commercial, storage, sports, recreational or leisure) and port
infrastructure must not exceed in the outdoor environment of the acoustic
area they sit in (Article 24.1).

**Parameters**

| Name | Description |
| :--- | :--- |
| `area_type` | Acoustic area type letter, or an alias. |

**Returns:** The applicable [`RegulationLimits`](/phonometry/reference/api/environment/spain/#regulationlimits) for the index `LK,x`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | For an unknown area type. |

## ActivityAssessment

```python
ActivityAssessment(
    periods: tuple[PeriodAssessment, ...],
    limits: RegulationLimits,
    new_activity: bool,
)
```

Compliance of an activity or port infrastructure (Article 25).

**Attributes**

| Name | Description |
| :--- | :--- |
| `periods` | The per-period assessments, in day/evening/night order. |
| `limits` | The limit table row the assessment was made against. |
| `new_activity` | `True` when the activity is *new* in the sense of the regulation, so the annual criterion of Article 25.1 b i applies; for the inspection of an activity already in operation only the daily and phase criteria apply (Article 25.2). |

### ActivityAssessment.complies

*property*

Whether every period meets every evaluated criterion.

### ActivityAssessment.plot()

```python
ActivityAssessment.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the per-period indices against their RD 1367/2007 limits.

### ActivityAssessment.report()

```python
ActivityAssessment.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = 'es',
) -> str
```

Render a one-page noise inspection fiche (`acta`) to `path`.

The layout follows the Spanish municipal / accredited-laboratory noise
inspection report: an identification header, the per-phase measurement
table with the `Kt`/`Kf`/`Ki` corrections, the per-period
assessment against the applicable limits and the boxed verdict.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata) supplying the header and footer identity fields. |
| `verbose` | Add the per-phase duration and correction breakdown columns. |
| `language` | Fiche language: `"es"` (default, the language of the regulation) or `"en"`. |

**Returns:** The written `path`.

**Raises**

| Exception | When |
| :--- | :--- |
| ImportError | If reportlab (or, for the figure, matplotlib) is not installed. |

## adjacent_premises_limits

```python
adjacent_premises_limits(
    building_use: str,
    room_type: str,
) -> RegulationLimits
```

Limits on noise transmitted to adjacent premises (Annex III Table B2).

Two premises are acoustically adjacent when noise never travels between
emitter and receiver through the outdoor environment (Article 24.3): the
workshop on the ground floor and the flat above it are, the building across
the street is not.

**Parameters**

| Name | Description |
| :--- | :--- |
| `building_use` | `"residential"`, `"office"` (alias `"administrative"`), `"sanitary"` or `"educational"`. |
| `room_type` | `"living"`, `"bedrooms"`, `"professional_offices"`, `"offices"`, `"classrooms"` or `"reading_rooms"`. |

**Returns:** The applicable [`RegulationLimits`](/phonometry/reference/api/environment/spain/#regulationlimits) for the index `LK,x`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | For an unknown combination. |

## assess_activity

```python
assess_activity(
    measurements: Mapping[str, Sequence[NoisePhase]],
    limits: RegulationLimits,
    *,
    long_term_levels: Mapping[str, float] | None = None,
    operating_days: int | None = None,
    year_days: int = 365,
    closed_level: float = 0.0,
    new_activity: bool = True,
    period_hours: Mapping[str, float] | None = None,
) -> ActivityAssessment
```

Assess an activity against the RD 1367/2007 limit values (Article 25).

For every evaluation period supplied the function integrates the noise
phases into `LKeq,x` (Annex IV A.3.4.2 b), derives the annual `LK,x`
when annual information is given, and checks the three criteria of Article
25.1 b: no annual `LK,x` above the table limit, no daily `LKeq,x` more
than 3 dB above it, and no measured `LKeq,Ti` more than 5 dB above it.
With `new_activity=False` only the last two apply (Article 25.2, the
inspection of an activity in operation).

The annual index can be supplied directly through `long_term_levels` or
derived from `operating_days`: the reported daily level then represents
`operating_days` days of the year and `closed_level` the remaining
`year_days - operating_days`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `measurements` | Noise phases keyed by evaluation period (`"day"`, `"evening"`, `"night"`). Periods that are absent are not assessed. |
| `limits` | The applicable limit row, from [`activity_limits`](/phonometry/reference/api/environment/spain/#activity_limits) or [`adjacent_premises_limits`](/phonometry/reference/api/environment/spain/#adjacent_premises_limits). |
| `long_term_levels` | Annual `LK,x` per period, in dB, when known. |
| `operating_days` | Number of days a year the activity operates; used to derive `LK,x` from the daily `LKeq,x` when `long_term_levels` is not given. |
| `year_days` | Days in the year considered (default 365). |
| `closed_level` | Level, in dB, representing a day on which the activity does not operate (default 0 dB, as in the Spanish worked literature). |
| `new_activity` | Whether the annual criterion applies. |
| `period_hours` | Period durations `T`, in hours, overriding [`RD1367_PERIOD_HOURS`](/phonometry/reference/api/environment/spain/#rd1367_period_hours). |

**Returns:** An [`ActivityAssessment`](/phonometry/reference/api/environment/spain/#activityassessment).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | For an unknown period key, empty phase list, inconsistent durations, or invalid annual parameters. |

## corrected_level

```python
corrected_level(
    laeq: float,
    *,
    kt: float = 0.0,
    kf: float = 0.0,
    ki: float = 0.0,
) -> float
```

Corrected equivalent continuous level `LKeq,T` (Annex I A.2 c).

$L_{\mathrm{Keq},T} = L_{\mathrm{Aeq},T} + K_\mathrm{t} + K_\mathrm{f} + K_\mathrm{i}$ with the sum of the
corrections capped
at 9 dB. Although it is derived from an A-weighted level, the index is
expressed in dB by definition.

**Parameters**

| Name | Description |
| :--- | :--- |
| `laeq` | A-weighted equivalent continuous level `LAeq,T`, in dB, already corrected for background noise. |
| `kt` | Tonal correction, in dB. |
| `kf` | Low-frequency correction, in dB. |
| `ki` | Impulsive correction, in dB. |

**Returns:** `LKeq,T` in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `laeq` is not finite or a correction is negative. |

## evaluation_period_level

```python
evaluation_period_level(
    phases: Sequence[NoisePhase],
    *,
    hours: float | None = None,
) -> float
```

Evaluation-period level `LKeq,T` from its noise phases.

$L_{\mathrm{Keq},T} = 10 \log_{10}\left[ (1/T) \sum_i T_i \cdot 10^{L_{\mathrm{Keq},Ti}/10} \right]$ (Annex IV
A.3.4.2 b): the duration-weighted energy mean of the phase levels. The
returned value is **not** rounded; apply [`round_reported_level`](/phonometry/reference/api/environment/spain/#round_reported_level) for the value
the regulation asks to report.

**Parameters**

| Name | Description |
| :--- | :--- |
| `phases` | The noise phases of the period. |
| `hours` | Total period duration `T`, in hours. `None` (default) uses the sum of the phase durations, as the regulation requires $\sum T_i = T$. |

**Returns:** `LKeq,T` in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `phases` is empty, `hours` is not positive, or the phase durations do not sum to `hours`. |

## impulsive_correction

```python
impulsive_correction(laieq: float, laeq: float) -> float
```

Impulsive correction `Ki` (Annex IV A.3.3).

From the impulse- and fast-time-weighted equivalent levels of the same
noise phase, both already corrected for background noise,
$L_\mathrm{i} = L_{\mathrm{AIeq},Ti} - L_{\mathrm{Aeq},Ti}$ gives $K_\mathrm{i} = 0$ for
$L_\mathrm{i} \le 10$ dB,
$K_\mathrm{i} = 3$ for $10 < L_\mathrm{i} \le 15$ dB and $K_\mathrm{i} = 6$ above.

:::note
This is the classic sound-level-meter route. The onset-rate method of
[`impulsive_sound_adjustment`](/phonometry/reference/api/environment/impulsive-sound/#impulsive_sound_adjustment)
(ISO/PAS 1996-3) is a different, signal-based procedure and its `KI`
is not interchangeable with this `Ki`. The same misprint as in
[`low_frequency_correction`](/phonometry/reference/api/environment/spain/#low_frequency_correction) affects the 3 dB row of the printed
table.
:::

**Parameters**

| Name | Description |
| :--- | :--- |
| `laieq` | Impulse-weighted equivalent level `LAIeq,Ti`, in dB. |
| `laeq` | A-weighted equivalent continuous level `LAeq,Ti`, in dB. |

**Returns:** `Ki` in dB (0, 3 or 6).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If either level is not finite. |

## indoor_quality_objectives

```python
indoor_quality_objectives(
    building_use: str,
    room_type: str,
) -> RegulationLimits
```

Indoor acoustic quality objectives `Ld`/`Le`/`Ln` (Annex II Table B).

The objectives that the *whole set* of emitters reaching a habitable room
must respect: dwellings and residential uses, hospitals, and educational or
cultural buildings.

**Parameters**

| Name | Description |
| :--- | :--- |
| `building_use` | `"residential"`, `"sanitary"` (alias `"hospital"`) or `"educational"`. |
| `room_type` | `"living"` (living areas), `"bedrooms"`, `"classrooms"` or `"reading_rooms"`. |

**Returns:** The applicable [`RegulationLimits`](/phonometry/reference/api/environment/spain/#regulationlimits) for the index `Lx`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | For an unknown combination. |

## infrastructure_limits

```python
infrastructure_limits(area_type: str) -> RegulationLimits
```

Immission limits for new transport infrastructure (Annex III Table A1).

New road, rail and airport infrastructure must not transmit to the outdoor
environment of an acoustic area levels above these `Ld`/`Le`/`Ln`
values (Article 23.1).

**Parameters**

| Name | Description |
| :--- | :--- |
| `area_type` | Acoustic area type letter, or an alias. |

**Returns:** The applicable [`RegulationLimits`](/phonometry/reference/api/environment/spain/#regulationlimits) for the index `Lx`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | For an unknown area type. |

## long_term_corrected_level

```python
long_term_corrected_level(
    daily_levels: Sequence[float] | np.ndarray,
    *,
    weights: Sequence[float] | np.ndarray | None = None,
) -> float
```

Long-term index `LK,x` from the daily period levels (Annex I A.2 d).

$L_{\mathrm{K},x} = 10 \log_{10}\left[ (1/n) \sum_i 10^{L_{\mathrm{Keq},x,i}/10} \right]$:
the energy mean of the
daily corrected levels of the same evaluation period over a year. With
`weights` the mean is weighted, which lets a whole block of identical
days be entered once (e.g. 303 operating days at one level and 62 closed
days at another).

**Parameters**

| Name | Description |
| :--- | :--- |
| `daily_levels` | Daily `LKeq,x` values, in dB. |
| `weights` | Optional number of days each level represents (positive). |

**Returns:** `LK,x` in dB, unrounded.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the input is empty, non-finite, or the weights do not match the levels or are not positive. |

## low_frequency_correction

```python
low_frequency_correction(lceq: float, laeq: float) -> float
```

Low-frequency correction `Kf` (Annex IV A.3.3).

From the C- and A-weighted equivalent levels of the same noise phase, both
already corrected for background noise,
$L_f = L_{\mathrm{Ceq},Ti} - L_{\mathrm{Aeq},Ti}$ gives
$K_\mathrm{f} = 0$ for $L_f \le 10$ dB, $K_\mathrm{f} = 3$ for
$10 < L_f \le 15$ dB and
$K_\mathrm{f} = 6$ above.

:::note
The printed table reads "Si 10 >Lf \<=15" for the 3 dB row, a misprint
for $10 < L_f \le 15$; the bracketing rows ($L_f \le 10$
and
$L_f > 15$) leave no other consistent reading. See
`docs/ERRATA.md`.
:::

**Parameters**

| Name | Description |
| :--- | :--- |
| `lceq` | C-weighted equivalent continuous level `LCeq,Ti`, in dB. |
| `laeq` | A-weighted equivalent continuous level `LAeq,Ti`, in dB. |

**Returns:** `Kf` in dB (0, 3 or 6).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If either level is not finite. |

## max_infrastructure_limit

```python
max_infrastructure_limit(area_type: str) -> float
```

Maximum `LAmax` for rail and airport infrastructure (Annex III Table A2).

The limit is respected when, over a year, 97 % of all daily values stay at
or below it (Article 25.1 a iii).

**Parameters**

| Name | Description |
| :--- | :--- |
| `area_type` | Acoustic area type letter, or an alias. |

**Returns:** The `LAmax` limit, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | For an unknown area type. |

## NoisePhase

```python
NoisePhase(
    hours: float,
    laeq: float,
    kt: float = 0.0,
    kf: float = 0.0,
    ki: float = 0.0,
    label: str | None = None,
)
```

A noise phase `Ti` of steady emission within an evaluation period.

RD 1367/2007 (Annex IV A.3.4.2 b) splits an evaluation period whose
emission varies into phases in which the sound pressure level at the
assessment point is perceived uniformly, measures `LAeq,Ti` over at
least 5 s in each, and corrects it for tonal, low-frequency and impulsive
character.

**Attributes**

| Name | Description |
| :--- | :--- |
| `hours` | Duration `Ti` of the phase, in hours (positive). |
| `laeq` | Background-corrected `LAeq,Ti`, in dB. A phase in which the activity is shut down carries the level actually measured; the worked examples of the Spanish literature enter it as 0 dB. |
| `kt` | Tonal correction of the phase, in dB. |
| `kf` | Low-frequency correction of the phase, in dB. |
| `ki` | Impulsive correction of the phase, in dB. |
| `label` | Optional free-text description of the phase. |

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `hours` is not positive and finite, `laeq` is not finite, or a correction is negative. |

### NoisePhase.correction

*property*

Summed correction $K = K_\mathrm{t} + K_\mathrm{f} + K_\mathrm{i}$, capped at 9 dB.

### NoisePhase.lkeq

*property*

The corrected level `LKeq,Ti` of the phase, in dB.

## outdoor_quality_objectives

```python
outdoor_quality_objectives(
    area_type: str,
    *,
    urbanisation: str = 'existing',
) -> RegulationLimits
```

Outdoor acoustic quality objectives `Ld`/`Le`/`Ln` (Annex II Table A).

The values are those of Table A as amended by RD 1038/2012. For an area
urbanised **after** the regulation entered into force (24 October 2007) the
objective is the same table reduced by 5 dB (Article 14.2); the same 5 dB
reduction defines the objective of quiet areas (Article 14.4).

Area type `"f"` (general transport-infrastructure systems) carries no
numeric objective: footnote (2) refers it to the adjoining areas.

**Parameters**

| Name | Description |
| :--- | :--- |
| `area_type` | Acoustic area type: the letter `"e"`, `"a"`, `"d"`, `"c"` or `"b"`, or an alias such as `"residential"`. |
| `urbanisation` | `"existing"` (default) for areas already urbanised on 24 October 2007, or `"new"` for the rest (5 dB stricter). |

**Returns:** The applicable [`RegulationLimits`](/phonometry/reference/api/environment/spain/#regulationlimits) for the index `Lx`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | For an unknown area type or urbanisation state. |

## PeriodAssessment

```python
PeriodAssessment(
    period: str,
    phases: tuple[NoisePhase, ...],
    duration_hours: float,
    evaluation_period_level: float,
    reported_level: int,
    long_term_corrected_level: float | None,
    reported_long_term: int | None,
    limit: float,
    max_phase_level: float,
    phase_pass: bool,
    daily_pass: bool,
    long_term_pass: bool | None,
)
```

The assessment of one evaluation period against its limit.

**Attributes**

| Name | Description |
| :--- | :--- |
| `period` | `"day"`, `"evening"` or `"night"`. |
| `phases` | The noise phases the period was split into. |
| `duration_hours` | The period duration `T`, in hours. |
| `evaluation_period_level` | `LKeq,x` of the period, in dB, unrounded. |
| `reported_level` | `LKeq,x` rounded per Annex IV A.3.4.2. |
| `long_term_corrected_level` | Annual `LK,x`, in dB, unrounded, or `None` when no annual information was supplied. |
| `reported_long_term` | `LK,x` rounded, or `None`. |
| `limit` | The table limit of the period, in dB. |
| `max_phase_level` | The largest `LKeq,Ti` of the period, in dB. |
| `phase_pass` | Whether every `LKeq,Ti` stays within `limit + 5` dB. |
| `daily_pass` | Whether `LKeq,x` stays within `limit + 3` dB. |
| `long_term_pass` | Whether `LK,x` stays at or below `limit`, or `None` when the criterion was not evaluated. |

### PeriodAssessment.complies

*property*

Whether every evaluated criterion of this period is met.

### PeriodAssessment.daily_limit

*property*

The daily limit `limit + 3` dB (Article 25.1 b ii).

### PeriodAssessment.phase_limit

*property*

The phase limit `limit + 5` dB (Article 25.1 b iii).

## RD1367_CORRECTION_VALUES

*Constant* (`tuple`).

```python
RD1367_CORRECTION_VALUES = (0.0, 3.0, 6.0)
```

## RD1367_EVALUATION_PERIODS

*Constant* (`tuple`).

```python
RD1367_EVALUATION_PERIODS = ('day', 'evening', 'night')
```

## RD1367_MAX_CORRECTION

*Constant* (`float`).

```python
RD1367_MAX_CORRECTION = 9.0
```

## RD1367_PERIOD_CLOCK_LIMITS

*Constant* (`mappingproxy`).

```python
RD1367_PERIOD_CLOCK_LIMITS = {'day': (7, 19), 'evening': (19, 23), 'night': (23, 7)}
```

## RD1367_PERIOD_HOURS

*Constant* (`mappingproxy`).

```python
RD1367_PERIOD_HOURS = {'day': 12.0, 'evening': 4.0, 'night': 8.0}
```

## RegulationLimits

```python
RegulationLimits(
    day: float,
    evening: float,
    night: float,
    index: str,
    reference: str,
    description: str,
)
```

A day/evening/night limit triple read from RD 1367/2007.

**Attributes**

| Name | Description |
| :--- | :--- |
| `day` | Limit of the day period, in dB. |
| `evening` | Limit of the evening period, in dB. |
| `night` | Limit of the night period, in dB. |
| `index` | Noise index the limits apply to (e.g. `"LK,x"`, `"Lx"`). |
| `reference` | The table the values were read from. |
| `description` | The row of that table, in plain words. |

### RegulationLimits.as_dict()

```python
RegulationLimits.as_dict() -> dict[str, float]
```

The three limits keyed by evaluation period.

## round_reported_level

```python
round_reported_level(value: float) -> int
```

Round a level the way RD 1367/2007 prescribes (Annex IV A.3.4.2).

"El valor del nivel sonoro resultante se redondeará incrementándolo en
0,5 dB(A), tomando la parte entera como valor resultante": add 0,5 dB and
take the integer part. This is a half-up rounding towards `+inf`, which
for a negative level differs from Python's banker's `round`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `value` | Sound level, in dB. |

**Returns:** The rounded level, as an `int`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `value` is not finite. |

## tonal_correction

```python
tonal_correction(
    levels: Sequence[float] | np.ndarray,
    frequencies: Sequence[float] | np.ndarray,
) -> TonalCorrectionResult
```

Tonal correction `Kt` from a one-third-octave spectrum (Annex IV A.3.3).

The spectrum must be **unweighted** (no frequency weighting applied, as
required by step a). For every interior band `f` the procedure forms
$L_\mathrm{t} = L_f - L_s$ with `Ls` the arithmetic mean of the levels of
the bands
immediately above and below (step b), and reads `Kt` off the table of
step c: with 20 Hz to 125 Hz bands $L_\mathrm{t} < 8$ gives 0 dB,
$8 \le L_\mathrm{t} \le 12$
gives 3 dB and $L_\mathrm{t} > 12$ gives 6 dB; the thresholds are 5/8 dB over
160 Hz to 400 Hz and 3/5 dB over 500 Hz to 10 kHz. With more than one
emergent tone the largest `Kt` governs (step d).

:::note
This is *not* the ISO 1996-2 tonal adjustment. The closest ISO relative
is the Annex K survey method
([`tonal_seeking_survey`](/phonometry/reference/api/environment/measurement/#tonal_seeking_survey)),
which compares the band against *both* neighbours with 15/8/5 dB
thresholds and only flags prominence; the RD compares against their
arithmetic mean with 8/5/3 dB thresholds and grades the result 0/3/6 dB.
:::

**Parameters**

| Name | Description |
| :--- | :--- |
| `levels` | Unweighted one-third-octave band levels, in dB. |
| `frequencies` | Band centre frequencies, in Hz, in ascending order (one per level). |

**Returns:** A [`TonalCorrectionResult`](/phonometry/reference/api/environment/spain/#tonalcorrectionresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If fewer than three bands are given, the shapes differ, the frequencies are not positive and strictly ascending, or any value is not finite. |

## TonalCorrectionResult

```python
TonalCorrectionResult(
    frequencies: np.ndarray,
    levels: np.ndarray,
    differences: np.ndarray,
    band_corrections: np.ndarray,
    correction: float,
    governing_frequency: float | None,
)
```

Tonal correction `Kt` of RD 1367/2007 (Annex IV A.3.3).

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | One-third-octave band centre frequencies, in Hz. |
| `levels` | Unweighted band sound pressure levels, in dB. |
| `differences` | $L_\mathrm{t} = L_f - L_s$ per band, in dB, where `Ls` is the arithmetic mean of the two adjacent band levels. `NaN` for bands that cannot be evaluated (the two end bands, and bands outside the 20 Hz to 10 kHz range of the table). |
| `band_corrections` | The `Kt` each band would contribute, in dB (0, 3 or 6); `NaN` where `differences` is `NaN`. |
| `correction` | The governing `Kt`, in dB: the largest band contribution (Annex IV A.3.3 d), or 0 dB if no band qualifies. |
| `governing_frequency` | Centre frequency of the governing band, in Hz, or `None` when `correction` is 0 dB. |

### TonalCorrectionResult.plot()

```python
TonalCorrectionResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the band spectrum with the emergent-tone differences `Lt`.

## total_correction

```python
total_correction(kt: float = 0.0, kf: float = 0.0, ki: float = 0.0) -> float
```

Summed correction $K = K_\mathrm{t} + K_\mathrm{f} + K_\mathrm{i}$, capped at 9 dB
(Annex IV A.3.3).

Each of the three tables of Annex IV A.3.3 grades its parameter 0, 3 or
6 dB, so any other value is rejected rather than silently accepted: a
correction of, say, 4.5 dB is not a reading the regulation can produce.

**Parameters**

| Name | Description |
| :--- | :--- |
| `kt` | Tonal correction, in dB (0, 3 or 6). |
| `kf` | Low-frequency correction, in dB (0, 3 or 6). |
| `ki` | Impulsive correction, in dB (0, 3 or 6). |

**Returns:** The summed correction, in dB, never above [`RD1367_MAX_CORRECTION`](/phonometry/reference/api/environment/spain/#rd1367_max_correction).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a correction is not one of 0, 3 or 6 dB. |

## vibration_quality_objective

```python
vibration_quality_objective(building_use: str) -> float
```

Indoor vibration quality objective `Law`, in dB (Annex II Table C).

**Parameters**

| Name | Description |
| :--- | :--- |
| `building_use` | `"residential"`, `"sanitary"` (alias `"hospital"`) or `"educational"`. |

**Returns:** The `Law` objective, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | For an unknown building use. |
