---
title: "emission.workstation"
description: "What the ISO 11200 group shares: the emission sound pressure level at a work station, its two corrections and its uncertainty."
sidebar:
  label: "workstation"
---

What the ISO 11200 group shares: the emission sound pressure level at a work
station, its two corrections and its uncertainty.

The sound power level says how much noise a machine makes. The **emission sound
pressure level** says how much of it reaches the person working at it, and it is
the number a machine is declared and bought by. ISO 4871 already has this
library declaring $L_{p\mathrm{A}}$; nothing computed it until now.

Five standards determine it, and they differ only in how they get rid of the
room:

======================  =====================================================
ISO 11201:2010          Free field over a reflecting plane, so there is no
                        room to get rid of and $K_3 = 0$.
ISO 11202:2010          Two approximate routes to $K_3$, one for a
                        machine with a dominating source and one from the
                        directivity the work station sees.
ISO 11203:1995          No measurement at all: the level is derived from the
                        sound power level.
ISO 11204:2010          The same piecewise $K_3$ as ISO 11202 method
                        A.2, reached accurately rather than approximately.
ISO 11205:2003          By sound intensity, and not implemented here.
======================  =====================================================

Everything they share is in this module, transcribed once, and each part's own
method sits beside it in its own module.

**The quantity.** ISO 11201:2010 Equation (7), ISO 11202:2010 Equation (10) and
ISO 11204:2010 Equation (9) print one law three times,

$$
L_p = L'_p - K_1 - K_3
$$

where $L'_p$ is what the meter read, $K_1$ removes the background
noise and $K_3$ removes the reflections the room sent back. ISO 11201
prints it without the $K_3$ term because its environment is qualified so
that the term is negligible, which is the same equation with a zero in it.

**Peak levels take no correction at all.** ISO 11204:2010 clause 7 and
ISO 11202:2010 clause 8 both say so: $L_{p\mathrm{C,peak}}$ is reported as
measured. A correction derived from mean-square pressures has no meaning for a
single largest excursion, so [`emission_sound_pressure_level`](/phonometry/reference/api/power/workstation/#emission_sound_pressure_level) refuses a
`peak` result that carries either correction rather than quietly applying it.

**The background correction** is the same expression the sound-power side
already uses, ISO 3744:2010 Equation (16), but this group sets its own
thresholds: 15 dB of margin makes it negligible, and 6 dB (grade 2) or 3 dB
(grade 3) is as far down as a result may be claimed. Below that the correction
is clamped and the level becomes an upper bound, which is why
[`background_noise_correction_at_workstation`](/phonometry/reference/api/power/workstation/#background_noise_correction_at_workstation) returns the clamp rather than
raising: the reading is still worth reporting, it just stops being a
determination.

**The local environmental correction** is where the group divides. Both
ISO 11202 Equation (A.5) and ISO 11204 Equations (A.2)/(A.5) print the same
piecewise function of one dimensionless ratio $z$,

$$
K_3 = \begin{cases} 7\ \mathrm{dB}, & z \le 0{,}2 \\ -10 \lg z\ \mathrm{dB}, & 0{,}2 < z \le 1 \\ 0\ \mathrm{dB}, & z > 1 \end{cases}
$$

and differ only in how $z$ is reached. The two branches meet: at
$z = 0{,}2$ the middle branch gives $-10 \lg 0{,}2 = 6{,}99$ dB, so
the 7 dB cap is the curve's own value rounded, not a discontinuity.

**The uncertainty** is one pair of equations in all three measuring parts
(ISO 11201 Equations (10) and (11), ISO 11202 (13) and (14), ISO 11204 (12) and
(13)):

$$
\sigma_\mathrm{tot} = \sqrt{\sigma_{R0}^2 + \sigma_\mathrm{omc}^2}, \qquad U = k\,\sigma_\mathrm{tot}
$$

with $\sigma_{R0}$ the reproducibility of the method and
$\sigma_\mathrm{omc}$ the instability of the machine itself, estimated
from repeated measurements by Equation (C.1).

:::note
Equation (C.1) prints the **sample** standard deviation, with
$1/(N-1)$, and that is what [`operating_standard_deviation`](/phonometry/reference/api/power/workstation/#operating_standard_deviation)
computes. The two worked examples of ISO 11200:2014 Annex B do not agree
with each other on this: Table B.1 divides by $N$ and Table B.3 by
$N-1$. See `docs/ERRATA.md`.
:::

Sources (clean-room, implemented from the standard texts): ISO 11201:2010,
ISO 11202:2010 and its Amendment 1:2020, ISO 11203:1995 and its Amendment
1:2020, ISO 11204:2010, with the worked examples of ISO 11200:2014 Annex B.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## background_noise_correction_at_workstation

```python
background_noise_correction_at_workstation(
    measured_level_db: ArrayLike,
    background_level_db: ArrayLike,
    *,
    grade: Grade = 'engineering',
) -> tuple[float | NDArray[np.float64], bool]
```

Background-noise correction $K_1$ for the ISO 11200 group.

$$
K_1 = -10 \lg \left( 1 - 10^{-0,1 \Delta L} \right)\ \mathrm{dB}, \qquad \Delta L = L'_p - L_p(B)
$$

ISO 11201:2010 Equation (5), ISO 11202:2010 Equation (8) and
ISO 11204:2010 Equation (7), one expression printed three times. It is the
same closed form as ISO 3744:2010 Equation (16) on the sound-power side, and
this group puts its own thresholds around it: above
[`NEGLIGIBLE_BACKGROUND_MARGIN_DB`](/phonometry/reference/api/power/workstation/#negligible_background_margin_db) the correction is taken as zero, and
below the grade's entry in [`MINIMUM_BACKGROUND_MARGIN_DB`](/phonometry/reference/api/power/workstation/#minimum_background_margin_db) it is held
at the value it has there. A held correction does not fail the measurement;
it makes the level an upper bound, which the second return value reports and
the caller must carry into what it publishes.

**Parameters**

| Name | Description |
| :--- | :--- |
| `measured_level_db` | The reading with the machine running, $L'_p$, in decibels; a scalar or one value per band. |
| `background_level_db` | The reading with it stopped, $L_p(B)$, in decibels, of the same shape. |
| `grade` | `'engineering'` (grade 2, 6 dB) or `'survey'` (grade 3, 3 dB), which sets how far down a determination may be claimed. |

**Returns:** The correction in decibels, and whether any value was clamped.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the two arguments do not have the same shape, or the grade is neither of the two. |

## DEFAULT_COVERAGE_FACTOR

*Constant* (`float`).

```python
DEFAULT_COVERAGE_FACTOR = 1.6
```

## emission_expanded_uncertainty

```python
emission_expanded_uncertainty(
    total_standard_deviation_db: float,
    coverage_factor: float = 1.6,
) -> float
```

Expanded uncertainty $U = k\,\sigma_\mathrm{tot}$.

ISO 11201:2010 Equation (11), ISO 11202:2010 Equation (14) and
ISO 11204:2010 Equation (13). The coverage factor is the caller's to choose:
$k = 2$ gives the two-sided 95 % interval of a normal distribution,
while the worked examples of ISO 11200:2014 Annex B all print
$k = 1{,}6$, which is the one-sided factor used when the result is
compared with a limit value, and is this function's default.

**Parameters**

| Name | Description |
| :--- | :--- |
| `total_standard_deviation_db` | $\sigma_\mathrm{tot}$ in decibels. |
| `coverage_factor` | $k$ (default [`DEFAULT_COVERAGE_FACTOR`](/phonometry/reference/api/power/workstation/#default_coverage_factor)). |

**Returns:** $U$ in decibels.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If either argument is negative. |

## emission_sound_pressure_level

```python
emission_sound_pressure_level(
    measured_level_db: ArrayLike,
    *,
    background_correction_db: ArrayLike = 0.0,
    local_correction_db: ArrayLike = 0.0,
) -> float | NDArray[np.float64]
```

The emission sound pressure level, reading less both corrections.

$$
L_p = L'_p - K_1 - K_3
$$

ISO 11201:2010 Equation (7) (which prints no $K_3$ because its
environment makes the term negligible), ISO 11202:2010 Equation (10) and
ISO 11204:2010 Equation (9).

Never call this for a peak level. ISO 11202:2010 clause 8 and ISO 11204:2010
clause 7 both forbid correcting $L_{p\mathrm{C,peak}}$, which is
reported exactly as measured: neither correction has a meaning for a single
largest excursion, both being derived from mean-square pressures.

**Parameters**

| Name | Description |
| :--- | :--- |
| `measured_level_db` | The uncorrected reading $L'_p$, in decibels. |
| `background_correction_db` | $K_1$ in decibels (default 0). |
| `local_correction_db` | $K_3$ in decibels (default 0). |

**Returns:** $L_p$ in decibels, of the broadcast shape.

## EmissionPressureResult

```python
EmissionPressureResult(
    level_db: float | NDArray[np.float64],
    measured_level_db: float | NDArray[np.float64],
    background_correction_db: float | NDArray[np.float64],
    local_correction_db: float | NDArray[np.float64],
    grade: Grade,
    upper_bound: bool,
    standard: str,
)
```

An emission sound pressure level and the two corrections behind it.

**Attributes**

| Name | Description |
| :--- | :--- |
| `level_db` | Emission sound pressure level $L_p$, in decibels re 20 uPa: what the meter read, less both corrections. |
| `measured_level_db` | The uncorrected reading $L'_p$, in decibels. |
| `background_correction_db` | $K_1$, in decibels. |
| `local_correction_db` | $K_3$, in decibels. |
| `grade` | Accuracy grade the determination earns. |
| `upper_bound` | `True` when the background margin fell below the grade's minimum, so the level is an upper bound rather than a determination. |
| `standard` | The part of the group the determination followed. |

### EmissionPressureResult.plot()

```python
EmissionPressureResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the reading, the two corrections and what is left of them.

## environmental_ratio_from_absorption

```python
environmental_ratio_from_absorption(
    absorption_area_m2: ArrayLike,
    measurement_surface_m2: float,
    directivity_index_db: ArrayLike = 0.0,
) -> float | NDArray[np.float64]
```

The ratio $z$ from the equivalent sound absorption area.

$$
z = 1 - \frac{1}{1 + A / (4 S_M)}\, 10^{-0,1 D^*_{I,\mathrm{op}}}
$$

ISO 11204:2010 Equation (A.6). It is the same quantity
[`environmental_ratio_from_k2`](/phonometry/reference/api/power/workstation/#environmental_ratio_from_k2) returns, reached without going through
$K_2$: under the ISO 3744 definition
$K_2 = 10 \lg (1 + 4 S_M / A)$ the two are identically equal, which is
why ISO 11204 A.1.2 says the two routes rest on the same assumptions.

**Parameters**

| Name | Description |
| :--- | :--- |
| `absorption_area_m2` | Equivalent sound absorption area $A$ of the test room, in square metres, strictly positive. |
| `measurement_surface_m2` | Area $S_M$ of the reference measurement surface, in square metres, strictly positive. |
| `directivity_index_db` | $D^*_{I,\mathrm{op}}$ in decibels (default 0). |

**Returns:** The ratio $z$.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If either area is not strictly positive. |

## environmental_ratio_from_k2

```python
environmental_ratio_from_k2(
    environmental_correction_db: ArrayLike,
    directivity_index_db: ArrayLike = 0.0,
) -> float | NDArray[np.float64]
```

The ratio $z$ from the environmental correction of the test room.

$$
z = 1 - \left( 1 - 10^{-0,1 K_2} \right) 10^{-0,1 D^*_{I,\mathrm{op}}}
$$

ISO 11202:2010 Equation (A.4) and ISO 11204:2010 Equation (A.3). $K_2$
is the average environmental correction of the reference measurement
surface, the quantity [`environmental_correction`](/phonometry/reference/api/power/sound-power/#environmental_correction)
computes for the sound-power methods, and $D^*_{I,\mathrm{op}}$ is the
apparent directivity index the work station sees.

With no directivity to speak of the expression collapses to
$z = 10^{-0,1 K_2}$, so $K_3 = K_2$ exactly: a work station that
sees the machine no more strongly than the measurement surface does needs
the same correction the surface needed.

**Parameters**

| Name | Description |
| :--- | :--- |
| `environmental_correction_db` | $K_2$ in decibels, non-negative. |
| `directivity_index_db` | $D^*_{I,\mathrm{op}}$ in decibels (default 0, no directivity). |

**Returns:** The ratio $z$, of the broadcast shape.

## GRADE_2_MAX_K3_DB

*Constant* (`float`).

```python
GRADE_2_MAX_K3_DB = 4.0
```

## grade_from_local_correction

```python
grade_from_local_correction(local_correction_db: ArrayLike) -> Grade
```

The accuracy grade a local environmental correction earns.

ISO 11202:2010 A.1.3 puts the boundary at
[`GRADE_2_MAX_K3_DB`](/phonometry/reference/api/power/workstation/#grade_2_max_k3_db): a greatest possible $K_3$ of 4 dB or less
is grade 2 (engineering), and more than that is grade 3 (survey). Method A.2
reaches the same boundary by a different road, Condition (A.6), which is
algebraically the same 4 dB once (A.4) and (A.5) are substituted into it.

The worst band decides, since a determination is only as good as its
weakest part.

**Parameters**

| Name | Description |
| :--- | :--- |
| `local_correction_db` | $K_3$ in decibels; a scalar or one value per band. |

**Returns:** `'engineering'` or `'survey'`.

## local_environmental_correction

```python
local_environmental_correction(
    ratio: ArrayLike,
) -> float | NDArray[np.float64]
```

Local environmental correction $K_3$ from the ratio $z$.

$$
K_3 = \begin{cases} 7\ \mathrm{dB}, & z \le 0{,}2 \\ -10 \lg z\ \mathrm{dB}, & 0{,}2 < z \le 1 \\ 0\ \mathrm{dB}, & z > 1 \end{cases}
$$

ISO 11202:2010 Equation (A.5) and ISO 11204:2010 Equations (A.2) and (A.5),
the same three lines printed three times; only the route to $z$
differs between them, and that is [`environmental_ratio_from_k2`](/phonometry/reference/api/power/workstation/#environmental_ratio_from_k2) and
[`environmental_ratio_from_absorption`](/phonometry/reference/api/power/workstation/#environmental_ratio_from_absorption).

The cap is the curve's own value, not a separate rule:
$-10 \lg 0{,}2 = 6{,}99$ dB, so the function is continuous where the
7 dB takes over. The upper branch is a floor for the same reason a
correction cannot be negative: the room can only add to the reading.

**Parameters**

| Name | Description |
| :--- | :--- |
| `ratio` | The dimensionless $z$, strictly positive; a scalar or one value per band. |

**Returns:** $K_3$ in decibels, of the same shape.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If any value is not strictly positive. |

## MAX_K3_DB

*Constant* (`float`).

```python
MAX_K3_DB = 7.0
```

## MINIMUM_BACKGROUND_MARGIN_DB

*Constant* (`dict`).

```python
MINIMUM_BACKGROUND_MARGIN_DB = {'engineering': 6.0, 'survey': 3.0}
```

## NEGLIGIBLE_BACKGROUND_MARGIN_DB

*Constant* (`float`).

```python
NEGLIGIBLE_BACKGROUND_MARGIN_DB = 15.0
```

## operating_standard_deviation

```python
operating_standard_deviation(levels_db: ArrayLike) -> float
```

Standard deviation of the operating and mounting conditions.

$$
\sigma_\mathrm{omc} = \sqrt{\frac{1}{N-1} \sum_{j=1}^{N} \left( L'_{p,j} - \overline{L'_p} \right)^2}
$$

Equation (C.1), identical in ISO 11201:2010, ISO 11202:2010 and
ISO 11204:2010. It is the sample standard deviation of levels measured under
the same nominal conditions, and it answers how repeatable the machine is
rather than how good the method is: the measurements are made in situ, so
the readings need no correction before going in.

The divisor is $N-1$, as printed. The two worked examples of
ISO 11200:2014 Annex B disagree with each other about that, and
`docs/ERRATA.md` records it; this library follows the equation.

**Parameters**

| Name | Description |
| :--- | :--- |
| `levels_db` | Repeated readings under the same conditions, in decibels; at least two. |

**Returns:** $\sigma_\mathrm{omc}$ in decibels.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If fewer than two readings are given. |

## subinterval_level

```python
subinterval_level(levels_db: ArrayLike, durations_s: ArrayLike) -> float
```

One level for a cycle made of operating periods of different lengths.

$$
L_p = 10 \lg \left[ \frac{1}{T} \sum_{i=1}^{N} T_i\, 10^{0,1 L_{p,T_i}} \right]\ \mathrm{dB}, \qquad T = \sum_i T_i
$$

ISO 11201:2010 Equation (8), ISO 11202:2010 Equation (11) and
ISO 11204:2010 Equation (10). A machine that idles, cuts and returns spends
a different length of time in each state, so the states are energy-averaged
weighted by how long each lasts, not by how many there are.

**Parameters**

| Name | Description |
| :--- | :--- |
| `levels_db` | The level of each sub-interval, in decibels. |
| `durations_s` | How long each lasted, in seconds, strictly positive and of the same length. |

**Returns:** The level of the whole interval, in decibels.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the two are of different lengths, or a duration is not strictly positive. |

## total_standard_deviation

```python
total_standard_deviation(
    reproducibility_db: float,
    operating_db: float = 0.0,
) -> float
```

Total standard deviation of the determination.

$$
\sigma_\mathrm{tot} = \sqrt{\sigma_{R0}^2 + \sigma_\mathrm{omc}^2}
$$

ISO 11201:2010 Equation (10), ISO 11202:2010 Equation (13) and
ISO 11204:2010 Equation (12). The two components are taken as statistically
independent, which is what lets them add in quadrature: one is a property of
the method and the other of the machine.

**Parameters**

| Name | Description |
| :--- | :--- |
| `reproducibility_db` | $\sigma_{R0}$ of the method, in decibels. |
| `operating_db` | $\sigma_\mathrm{omc}$ of the machine, in decibels (default 0, a source whose emission does not wander). |

**Returns:** $\sigma_\mathrm{tot}$ in decibels.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If either component is negative. |
