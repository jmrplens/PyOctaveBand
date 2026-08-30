---
title: "emission.sound_power_intensity_points"
description: "Sound power level of a noise source from sound intensity measured at discrete points: ISO 9614-1:1993."
sidebar:
  label: "sound_power_intensity_points"
---

Sound power level of a noise source from sound intensity measured at
**discrete points**: ISO 9614-1:1993.

A p-p probe is held still at each of `N` points, one per segment of a
hypothetical surface enclosing the source, and reports the signed normal
intensity $I_{\mathrm{n}i}$ there. Each point stands for its segment, so
the partial power of the segment and the sound power of the source are
(clause 9.1 equation (11), clause 9.2 equation (12)):

$$
P_i = I_{\mathrm{n}i} \, S_i \tag{Eq. 11}
$$

$$
L_W = 10 \log_{10} \frac{\sum_{i=1}^{N} P_i}{P_0}, \qquad P_0 = 10^{-12}~\text{W} \tag{Eq. 12}
$$

Equation (12) prints without the absolute-value bars of the general definition
(equation (8)), because clause 9.2 disposes of the negative case instead: **the
method is not applicable to a band in which** $\sum_i P_i$ **is
negative**. A single segment may still carry negative power, and normally
does; that is energy flowing inward through part of the surface, which is what
$F_3$ exists to quantify, not an error to reject.

A.2.3 makes a second refusal, on a different quantity: where
$\sum_i I_{\mathrm{n}i}$ is not positive, "the test conditions do not
satisfy the requirements of this part of ISO 9614 in that frequency band". That sum is unweighted over the `N` positions, so equal
segments make the two refusals agree and unequal ones let them part company: a
band clause 9.2 keeps, with a positive total power and a finite level, can
still be one A.2.3 refuses. Both are reported, each in its own terms.

The sign lives in the print, not in the number. ISO 9614-1 writes a normal
intensity level as `XX dB` when the flow is outward and as `(-) XX dB` when
it is inward, `XX` being a positive number in both cases (clause 3.5, and the
two unnumbered equations of clauses 9.1 and A.2.3):

$$
I_{\mathrm{n}i} = I_0 \times 10^{XX/10}, \qquad I_{\mathrm{n}i} = -I_0 \times 10^{XX/10}
$$

with $I_0 = 10^{-12}$ W/m^2. [`normal_intensity_from_levels`](/phonometry/reference/api/power/sound-power-intensity-points/#normal_intensity_from_levels) takes
both forms; its `negative` argument is the `(-)` of the print.

The four Annex A field indicators come from
[`phonometry.emission.field_indicators`](/phonometry/reference/api/power/intensity/#field_indicators), which is written for this part of
ISO 9614 and averages over positions without area weighting, exactly as
equations (A.4), (A.5), (A.7) and (A.9) do. What this module adds is Annex B:
the qualification of the surface and of the position set, and what to change
when they do not qualify.

Annex B numbers **two** criteria, not three (B.1.1 and B.1.2). Figure B.1
gates the determination on four questions, in this order, and Table B.3 gives
the action to take when each fails:

$$
F_1 \le 0.6 \quad\text{(Table B.3, action e)}
$$

$$
\text{criterion 1:} \quad L_\mathrm{d} > F_2, \qquad L_\mathrm{d} = \delta_{pI0} - K \quad\text{(Eq. (B.1), actions a or b)}
$$

$$
F_3 - F_2 \le 3~\text{dB} \quad\text{(Figure B.1, actions a or b)}
$$

$$
\text{criterion 2:} \quad N > C F_4^2 \quad\text{(Eq. (B.2), actions c or d)}
$$

The third gate is unnumbered in the print: it appears in Figure B.1 and shares
the second row of Table B.3 with criterion 1, and it is *not* a "criterion 3".
Only Figure B.1's first failing gate is acted on, because every action box in
the figure returns to the next measurement rather than to the gate below it, so
[`DiscretePointIntensityResult.required_actions`](/phonometry/reference/api/power/sound-power-intensity-points/#discretepointintensityresultrequired_actions) reports one action set per
band and stops there.

**Grade 3 is an A-weighted determination only.** Table B.2 tabulates the
criterion-2 factor `C` for grades 1 and 2 band by band and gives grade 3 a
single A-weighted value (8) and no per-band column at all; Table B.1 does the
same with the error factor $\Delta$ (0,20 and 0,29 for all bands at
grades 1 and 2, 0,60 A-weighted at grade 3); and Table 2 does the same with the
standard deviation `s` of the determination. Three tables agree, so the
asymmetry is the standard's design and not a gap in it: a per-band
determination can reach grade 1 or grade 2, and grade 3 is reached, if at all,
by the A-weighted sum. Asking any of the three lookups here for a per-band
grade-3 figure raises rather than returning a plausible number.

The uncertainty of the determination is Table 2's `s`, with footnote 1
placing the true value within $\pm 2s$ of the measured one at 95 %
confidence. Clause 10.6 says which row of the table to read it from: "the
grade of accuracy attained in the final test, according to table 2, shall
be stated", the grade **attained** and not the grade set out for, so a band
that only reached grade 2 carries the grade-2 figure even where grade 1 was asked
for. A band that fails criterion 2 may still be recorded, provided the 95 %
confidence interval of equation (B.3) accompanies it (B.1.2, clause 10.5 c)
and clause 10.6):

$$
10 \log_{10}\!\left( 1 \pm \frac{2 F_4}{\sqrt{N}} \right)~\text{dB} \tag{Eq. B.3}
$$

[`partial_power_concentration`](/phonometry/reference/api/power/sound-power-intensity-points/#partial_power_concentration) is the optional procedure of clause 8.3.2
and B.1.3: where criterion 1 holds, criterion 2 does not and
$F_3 - F_2 \le 1$ dB, most of the power may pass through a minority of
the segments, and adding positions only there is cheaper than densifying the
whole surface. It is implemented because it is the only consumer of Table B.1
and because equation (B.4) is the standard's own answer to the commonest way a
discrete-point survey fails.

Two things this module does not do. It does not scan: the continuous sweep of
ISO 9614-2:1996 and ISO 9614-3:2002, whose indicators are area weighted and
whose criteria are numbered differently, is
[`phonometry.emission.sound_power_intensity`](/phonometry/reference/api/power/sound-power-intensity/). And it does not grade the
instrument: $\delta_{pI0}$ is a property of the probe-spacer-analyser
chain, classified against IEC 61043:1993 Table 2 by
[`phonometry.emission.intensity_class_compliance`](/phonometry/reference/api/power/intensity-compliance/#intensity_class_compliance).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## ActionCode

```python
ActionCode(*values)
```

An action of ISO 9614-1:1993 Table B.3, by its printed code letter.

Table B.3 answers the question the criteria leave open: the surface or the
position set does not qualify, so what is to be *changed*. Its five actions
are lettered a to e, and Figure B.1 routes each failing gate to one or two
of them. The letter is the member value, because that is what a test report
cites; `criterion` and `action` carry the row it was read from
and what it asks for.

### ActionCode.action

*property*

What Table B.3 asks the operator to change, in one sentence.

### ActionCode.criterion

*property*

The Table B.3 criterion row that calls for this action.

## determination_standard_deviation

```python
determination_standard_deviation(
    grade: DeterminationGrade,
    frequency: float | None = None,
    *,
    band_type: BandType = 'third',
) -> float
```

Standard deviation `s` of the determination, ISO 9614-1:1993 Table 2.

Footnote 1 of the table states what `s` is for: the true sound power
level is expected to lie within $\pm 2s$ of the measured one with
95 % confidence, so twice this figure is the expanded uncertainty of a
qualified determination.

The table has the same shape as Table B.2 and the same asymmetry, and this
is where the asymmetry is load bearing rather than merely odd: the standard
defines no per-band uncertainty for grade 3 at all, only an A-weighted one,
which is why grade 3 under this part of ISO 9614 is an A-weighted
determination. Footnote 3 calls the grade-3 figure tentative.

**Parameters**

| Name | Description |
| :--- | :--- |
| `grade` | `'precision'` (grade 1), `'engineering'` (grade 2) or `'survey'` (grade 3). |
| `frequency` | Nominal mid-band centre in Hz, or `None` for the A-weighted row. |
| `band_type` | `'octave'` or `'third'`; ignored when `frequency` is `None`. |

**Returns:** `s` in decibels.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | On the same mismatches [`position_count_factor`](/phonometry/reference/api/power/sound-power-intensity-points/#position_count_factor) refuses. |

## DiscretePointIntensityResult

```python
DiscretePointIntensityResult(
    frequencies: np.ndarray | None,
    partial_power: np.ndarray,
    sound_power: np.ndarray,
    sound_power_level: np.ndarray,
    not_applicable_band: np.ndarray,
    f1: np.ndarray | None,
    f2: np.ndarray | None,
    f3: np.ndarray | None,
    f4: np.ndarray | None,
    dynamic_capability_index: np.ndarray | None,
    criterion_1: np.ndarray | None,
    negative_power_within_limit: np.ndarray | None,
    criterion_2: np.ndarray | None,
    minimum_positions: np.ndarray | None,
    achieved_grade: np.ndarray | None,
    confidence_interval: np.ndarray | None,
    expanded_uncertainty: np.ndarray | None,
    surface_area: float,
    positions: int,
    sound_power_level_a: float,
    a_weighting_omitted_bands: np.ndarray | None,
    field_nonuniformity_a: float,
    achieved_grade_a: str | None,
    grade: str,
)
```

Result of an ISO 9614-1:1993 discrete-point sound-power determination.

`partial_power` is the signed $P_i = I_{\mathrm{n}i} S_i$ per
position and band (equation (11)), `sound_power` its signed band total
and `sound_power_level` the level of that total (equation (12)), `NaN`
in a band whose total is not positive: `not_applicable_band` is `True`
there and clause 9.2 puts the band outside the method.

`f1` to `f4` are the Annex A field indicators per band, `None` when
the inputs they need were not supplied. `f2`, `f3` and `f4` are
`NaN` in a band whose algebraic mean normal intensity is not positive,
which A.2.3 makes a failure of the test conditions in that band. `f1`
is not: equation (A.1) is the spread of the M short-time samples at one
position over time, A.2.1 puts no positivity condition on it, and a band
A.2.3 refuses still has a perfectly good temporal variability. Measured on
such a band, `f1` comes back finite while the other three are `NaN`. A.2.3's refusal is not
`not_applicable_band`, whose quantity is clause 9.2's area-weighted sum,
and the determination warns about it separately, since a band can fail
A.2.3 and still carry a finite level here. `criterion_1`
($L_\mathrm{d} > F_2$, equation (B.1)), `negative_power_within_limit`
(Figure B.1's unnumbered $F_3 - F_2 \le 3$ dB gate) and
`criterion_2` ($N > C F_4^2$, equation (B.2)) are the per-band
verdicts, `None` where they could not be evaluated;
`minimum_positions` is the $C F_4^2$ that criterion 2 compares
`positions` against.

`achieved_grade` is the per-band grade, one of `'precision'`,
`'engineering'` and `'none'`. Grade 3 is never among them: Table B.2
gives it no per-band `C`, so criterion 2 has no per-band form there and
grade 3 is reached, if at all, by `achieved_grade_a` on the A-weighted
sum, whose own field non-uniformity is `field_nonuniformity_a`
(B.1.2, computed from the A-weighted band intensities of each position).

`confidence_interval` is the pair
$10 \lg (1 \pm 2 F_4 / \sqrt{N})$ of equation (B.3) per band, which
clause 10.5 c) requires beside the level of any band that failed criterion
2, and `expanded_uncertainty` the $2s$ of Table 2 footnote 1 read
at the grade the band *achieved*, that being the grade clause 10.6 has a
report state: `NaN` in a band that reached no grade, for which Table 2
prints no `s`, and `None` for the whole determination where
`achieved_grade` could not be established either.
`sound_power_level_a` omits the bands outside the method and, per clause
10.5 b), those failing criteria 1 and/or 2, which
`a_weighting_omitted_bands` flags.

### DiscretePointIntensityResult.plot()

```python
DiscretePointIntensityResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the LW spectrum; bands outside the method are hatched.

Draws the same figure as the scanning determinations of ISO 9614-2 and
ISO 9614-3, because the quantity is the same one: the band sound power
level, with the bands of non-positive net power (clause 9.2) hatched as
unusable and the A-weighted total in the title. The field indicators
behind the qualification have their own figure, on the
[`FieldIndicators`](/phonometry/reference/api/power/intensity/#fieldindicators) that
[`field_indicators`](/phonometry/reference/api/power/intensity/#field_indicators) returns.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `ax` | Existing axes, or `None` to create a figure. |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the band `bar`. |

### DiscretePointIntensityResult.required_actions()

```python
DiscretePointIntensityResult.required_actions() -> tuple[tuple[ActionCode, ...], ...]
```

The Table B.3 actions each band calls for, in Figure B.1's order.

Figure B.1 gates the determination on four questions and sends the
first failing one to an action box, from which the flow returns to the
next measurement rather than to the gate below. So a band gets one
action set, not a list of everything that went wrong: the tuple is
empty for a band that passed every gate it could be judged on, and
holds one or two codes otherwise. Two codes mean the standard offers a
choice, which is how Table B.3's second row prints them ("a **o** b").

A gate whose inputs are absent is skipped rather than failed. `F1`
comes from the initial test and is legitimately missing; criterion 2
needs `frequencies` for the Table B.2 lookup, and has no per-band
form at all at the survey grade. Where criterion 2 was not evaluated,
the fourth gate and its actions (c) and (d) cannot be reached, and a
band that clears the first three is reported as clear.

Action (d) is the one Table B.3 conditions on the operator: it applies
when criterion 2 fails with $F_3 - F_2 \le 1$ dB and the optional
procedure of clause 8.3.2 "either fails or is not selected". Not
selecting it is the default, so it is reported here; see
[`partial_power_concentration`](/phonometry/reference/api/power/sound-power-intensity-points/#partial_power_concentration) for the alternative.

**Returns:** One tuple of [`ActionCode`](/phonometry/reference/api/power/sound-power-intensity-points/#actioncode) per band.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the determination was never qualified, i.e. it carries no `criterion_1` because `pressure_levels` and `pressure_residual_index` were not supplied. |

## error_factor

```python
error_factor(
    grade: DeterminationGrade,
    *,
    a_weighted: bool = False,
) -> float
```

Error factor $\Delta$ of ISO 9614-1:1993 Table B.1.

$\Delta$ is the sampling error the optional procedure of B.1.3 is
allowed to leave on the determination, and equation (B.4) is its only
consumer. Table B.1 prints one row for all bands, holding 0,20 at grade 1
and 0,29 at grade 2, and one A-weighted row holding 0,60 at grade 3; the
remaining four cells are blank in the print.

**Parameters**

| Name | Description |
| :--- | :--- |
| `grade` | `'precision'` (grade 1), `'engineering'` (grade 2) or `'survey'` (grade 3). |
| `a_weighted` | `True` reads the A-weighted row (grade 3 only), `False` (default) the all-bands row (grades 1 and 2 only). |

**Returns:** $\Delta$, dimensionless.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `grade` is unknown, or the row asked for is blank at that grade. |

## normal_intensity_from_levels

```python
normal_intensity_from_levels(
    levels: ArrayLike,
    *,
    negative: ArrayLike = False,
) -> np.ndarray
```

Signed normal intensity from printed intensity levels (clause 9.1).

ISO 9614-1 does not print a signed level. A normal intensity level is
written `XX dB` when the flow through the segment is outward and
`(-) XX dB` when it is inward, with `XX` a positive number in both
cases (clause 3.5, and the two unnumbered equations of clauses 9.1 and
A.2.3):

$$
I_{\mathrm{n}i} = I_0 \times 10^{XX/10}, \qquad I_{\mathrm{n}i} = -I_0 \times 10^{XX/10}, \qquad I_0 = 10^{-12}~\text{W/m}^2
$$

So the sign is not in the number, and a caller reading a printed table has
to carry it separately: `negative` is the `(-)` of the print, and it
broadcasts against `levels`, which is what lets one position of a
measurement surface flow inward while the rest flow outward. Negative
partial power is normal and is what $F_3$ measures; it is not an
error, and only the *sum* going negative puts a band outside the method
(clause 9.2).

**Parameters**

| Name | Description |
| :--- | :--- |
| `levels` | Normal intensity levels `XX` in decibels, of any shape. Each is the level of the magnitude, so it is positive for a level above the reference intensity and negative for one below it; the direction of flow is `negative`, never the sign of this number. |
| `negative` | `True` where the printed level carried the `(-)` prefix, i.e. where the flow is inward. A single bool applies to every level; an array broadcasts against `levels`. |

**Returns:** The signed normal intensity in W/m^2, of the shape of `levels`. `negative` is broadcast onto that shape and never widens it: one flag per level, or one flag for all of them, and a mask of any other shape is refused rather than returning more intensities than levels went in.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a level is not finite, or `negative` cannot be broadcast against `levels`. |

## partial_power_concentration

```python
partial_power_concentration(
    normal_intensity: ArrayLike,
    areas: ArrayLike,
    *,
    grade: DeterminationGrade = 'engineering',
) -> PartialPowerConcentration
```

Positive partial power concentration and the new positions it needs.

The optional procedure of clause 8.3.2, computed as B.1.3 specifies. It
applies to a band in which criterion 1 holds, criterion 2 does not and
$F_3 - F_2 \le 1$ dB: little power flows inward, so most of it may be
leaving through a minority of the segments, and densifying only those
segments qualifies the surface for far less work than densifying all of it.

The positive partial powers are ranked in decreasing order and the top
segments are taken until more than half the total sound power has been
accounted for. That subset is $N_\alpha$ segments of total area
$S_\alpha$ carrying the fraction $\alpha > 0,5$, and B.1.3
requires $N_\alpha$ to be fewer than half of `N`. The field
non-uniformity is then evaluated separately over the subset and over the
remainder (equations (A.8) and (A.9)), and:

$$
N^* \ge 4 \left[ \frac{F_4(\alpha)}{\Delta_\alpha} \right]^2 \tag{Eq. B.4}
$$

$$
\Delta_\alpha = \frac{1}{\alpha} \left[ \Delta - (1 - \alpha) \frac{2}{\sqrt{N_{1-\alpha}}} F_4(1 - \alpha) \right], \qquad N_{1-\alpha} = N - N_\alpha
$$

$\Delta_\alpha$ is the share of the Table B.1 error budget left for
the subset once the remainder, measured at its existing density, has taken
its own; a remainder too non-uniform to leave anything over exhausts the
budget, and the procedure cannot help.

**Parameters**

| Name | Description |
| :--- | :--- |
| `normal_intensity` | Signed normal intensity $I_{\mathrm{n}i}$ at each position of one frequency band, in W/m^2 (1D). |
| `areas` | Segment areas $S_i$ in m^2, one per position (1D). |
| `grade` | The grade whose Table B.1 $\Delta$ is spent: `'precision'` (0,20), `'engineering'` (0,29) or `'survey'` (0,60, A-weighted). |

**Returns:** A [`PartialPowerConcentration`](/phonometry/reference/api/power/sound-power-intensity-points/#partialpowerconcentration).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the positions and areas disagree in length, an area is not positive and finite, an intensity is not finite, the total sound power is not positive (clause 9.2 puts the band outside the method), no subset satisfies the two conditions of B.1.3, the subset is a single segment (equation (A.8) has no spread over one position, so equation (B.4) is undefined), the algebraic mean normal intensity over the remainder is not positive (A.2.3, which happens when the subset takes all the outward flow), or the remainder leaves no error budget for the subset. In the last four cases the selective modification cannot be carried out, and clause 8.3.2 then asks for the appropriate alternative actions in accordance with clause B.2 and Table B.3. Which row of that table applies is not settled here: its two lower rows are conditioned on criterion 2 and on $F_3 - F_2$, neither of which this function is given. |

## PartialPowerConcentration

```python
PartialPowerConcentration(
    positions: int,
    subset_positions: int,
    subset_area: float,
    power_fraction: float,
    subset_nonuniformity: float,
    remainder_nonuniformity: float,
    error_factor: float,
    subset_error_factor: float,
    additional_positions: int,
)
```

Outcome of the optional procedure of ISO 9614-1:1993 clause 8.3.2/B.1.3.

The segments carrying most of the sound power, and how many new positions
equation (B.4) asks to be spread over them.

**Attributes**

| Name | Description |
| :--- | :--- |
| `positions` | `N`, the number of positions on the whole surface. |
| `subset_positions` | $N_\alpha$, the segments in the selected subset. B.1.3 requires this to be fewer than half of `N`. |
| `subset_area` | $S_\alpha$, the total area of the subset, m^2; the new positions are distributed over it in proportion to segment area. |
| `power_fraction` | $\alpha$, the fraction of the total sound power passing through the subset, always above 0,5. |
| `subset_nonuniformity` | $F_4(\alpha)$, the field non-uniformity of the subset alone (equations (A.8), (A.9)). |
| `remainder_nonuniformity` | $F_4(1-\alpha)$, that of the remaining segments. |
| `error_factor` | $\Delta$ for the requested grade (Table B.1). |
| `subset_error_factor` | $\Delta_\alpha$, the share of that error budget the subset may spend, after the remainder has taken its own. |
| `additional_positions` | $N^*$, the smallest whole number of new positions satisfying equation (B.4). |

## position_count_factor

```python
position_count_factor(
    grade: DeterminationGrade,
    frequency: float | None = None,
    *,
    band_type: BandType = 'third',
) -> float
```

Criterion-2 factor `C` of ISO 9614-1:1993 Table B.2.

Criterion 2 (equation (B.2)) asks for $N > C F_4^2$ measurement
positions, so `C` converts the field non-uniformity into a position
count. It depends on the band and on the grade claimed: a surface needs
roughly twice as many positions for precision as for engineering grade.

The table's two halves do not overlap. Grades 1 and 2 have a value in every
band and none for the A-weighted sum; grade 3 has the single A-weighted 8
and no band column at all. So `frequency` selects which half is being
asked for, and asking the wrong half raises.

**Parameters**

| Name | Description |
| :--- | :--- |
| `grade` | `'precision'` (grade 1), `'engineering'` (grade 2) or `'survey'` (grade 3). |
| `frequency` | Nominal mid-band centre in Hz, as a Table B.2 label or the exact base-ten centre behind it. `None` asks for the A-weighted row instead, whose footnote fixes the summed range at 63 Hz to 4 kHz (octave) or 50 Hz to 6,3 kHz (one-third octave). |
| `band_type` | `'octave'` or `'third'`, selecting the frequency column; ignored when `frequency` is `None`. |

**Returns:** `C`, dimensionless.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `grade` or `band_type` is unknown; if a per-band value is asked for at the survey grade, or an A-weighted one at the precision or engineering grade; or if `frequency` is not a tabulated band. |

## sound_power_intensity_points

```python
sound_power_intensity_points(
    normal_intensity: ArrayLike,
    areas: ArrayLike,
    *,
    pressure_levels: ArrayLike | None = None,
    pressure_residual_index: float | ArrayLike | None = None,
    temporal_intensity: ArrayLike | None = None,
    frequencies: ArrayLike | None = None,
    band_type: BandType = 'third',
    grade: DeterminationGrade = 'engineering',
) -> DiscretePointIntensityResult
```

Sound power by sound intensity at discrete points (ISO 9614-1:1993).

`normal_intensity` is an `(N, bands)` array (or `(N,)` for a single
band) of the signed normal intensity $I_{\mathrm{n}i}$ measured with
the probe held still at each of the `N` points, and `areas` the
`(N,)` areas $S_i$ of the segments those points stand for. The
partial powers $P_i = I_{\mathrm{n}i} S_i$ (equation (11)) sum to the
band sound power and its level $L_W = 10 \lg(\sum_i P_i / P_0)$
(equation (12)); a band whose sum is not positive is flagged
`not_applicable_band` and reported as `NaN`, the method not applying to
it (clause 9.2).

A single position may carry inward flow and usually does. Levels printed as
`(-) XX dB` are converted by [`normal_intensity_from_levels`](/phonometry/reference/api/power/sound-power-intensity-points/#normal_intensity_from_levels), whose
`negative` argument is that `(-)`.

A.2.3 conditions the Annex A indicators on a different quantity, the
unweighted mean of the `N` normal intensities, and makes a band whose
mean is not positive a band in which the test conditions do not satisfy
this part of ISO 9614. Its indicators come back `NaN` and the
determination warns, because that band need not be flagged anywhere else:
where the segments differ in area it can be a band clause 9.2 keeps, with a
finite level and `not_applicable_band` false.

Supplying `pressure_levels` evaluates the spatial Annex A indicators
`F2` and `F3`; `F4` is evaluated from the intensities alone. All
three are spatial, so all three need at least two positions, which is the
fewest equation (A.8) has a spread over, and all three are absent below
that however much else was supplied. `F1` is the temporal one and does
not take part in that: see `temporal_intensity` below. Supplying `pressure_residual_index` gives the dynamic
capability $L_\mathrm{d} = \delta_{pI0} - K$ and criterion 1; supplying
`frequencies` gives criterion 2 through the Table B.2 factor `C` and
the A-weighted total. `temporal_intensity` carries the `M` short-time
samples of the initial test into `F1`.

The requested `grade` selects `K`, the omission rule of clause 10.5 b)
and the tabulated factors; the grade each band actually reaches is reported
per band in `achieved_grade`, and for the A-weighted sum in
`achieved_grade_a`. The survey grade appears only in the latter, Table
B.2 giving it no per-band `C`. `expanded_uncertainty` follows the
achieved grade rather than the requested one, which is the grade clause
10.6 has a report state, so it needs everything `achieved_grade` needs.

**Parameters**

| Name | Description |
| :--- | :--- |
| `normal_intensity` | `(N, bands)` or `(N,)` signed normal intensity, W/m^2. |
| `areas` | `(N,)` segment areas $S_i$, m^2. |
| `pressure_levels` | Optional `(N, bands)` or `(N,)` sound pressure levels $L_{pi}$ at the same positions, dB. |
| `pressure_residual_index` | Optional $\delta_{pI0}$ of the instrument, dB, as a scalar or one value per band. |
| `temporal_intensity` | Optional `(M, bands)` or `(M,)` short-time samples of the normal intensity at one typical position (clause 8.2), W/m^2, for `F1`. |
| `frequencies` | Optional `(bands,)` nominal mid-band centres, Hz. |
| `band_type` | `'octave'` or `'third'`, the column of Tables B.2 and 2 the frequencies are read in. |
| `grade` | `'precision'` (grade 1), `'engineering'` (grade 2, default) or `'survey'` (grade 3). |

**Returns:** A [`DiscretePointIntensityResult`](/phonometry/reference/api/power/sound-power-intensity-points/#discretepointintensityresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the positions and areas disagree in length, the position set is empty, an area is not positive and finite, an input is not finite, the optional arrays do not span the same positions or bands, or a frequency is not a band of Tables B.2 and 2. |
