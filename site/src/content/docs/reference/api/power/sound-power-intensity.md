---
title: "emission.sound_power_intensity"
description: "Sound power level of a noise source by sound-intensity scanning: ISO 9614-2:1996 (engineering, grade 2; survey/control, grade 3) and ISO 9614-3:2002 (precision, grade 1)."
sidebar:
  label: "sound_power_intensity"
---

Sound power level of a noise source by sound-intensity **scanning**:
ISO 9614-2:1996 (engineering, grade 2; survey/control, grade 3) and
ISO 9614-3:2002 (precision, grade 1).

A probe is swept continuously over each segment of a hypothetical surface that
encloses the source, reporting the time-averaged signed normal intensity
$\langle I_{n,i} \rangle$ and mean-square pressure per segment. The
sound power follows from the partial powers
$P_i = \langle I_{n,i} \rangle S_i$ summed over the `N` segments
(clause 9, equations (5), (6), (12), (13)):

$$
P_i = \langle I_{n,i} \rangle \, S_i \tag{Eq. 12}
$$

$$
P = \sum_i P_i \tag{Eq. 6}
$$

$$
L_W = 10 \log_{10}\frac{P}{P_0}, \qquad P_0 = 10^{-12}~\text{W} \tag{Eq. 13}
$$

The method is **not applicable to any band in which** $P < 0$
(clause 9.2): a strong parasitic source outside the surface makes the net
energy flow inward and the determination invalid for that band.

Two scanning-method field indicators qualify the determination, with
$[L_p]$ the area-weighted surface sound pressure level (Annex A,
normative):

$$
F_{pI} = [L_p] - L_W + 10 \log_{10}\frac{S}{S_0} \tag{Eq. A.1}
$$

$$
[L_p] = 10 \log_{10}\!\left[ \frac{1}{S} \sum_i S_i \, 10^{0.1 L_{pi}} \right]
$$

$$
F_{+/-} = 10 \log_{10}\frac{\sum_i |P_i|}{\left| \sum_i P_i \right|} \tag{Eq. A.2}
$$

`FpI` is the surface pressure-intensity indicator (equivalent to ISO 9614-1
`F3` for uniform-area segments, Note 14); `F+/-` the negative-partial-power
indicator (equivalent to ISO 9614-1 `F3-F2`, Note 15). Because Part 2 weights
by segment area `Si` while [`phonometry.field_indicators`](/phonometry/reference/api/power/intensity/#field_indicators) (ISO 9614-1)
assumes equal-area positions, the indicators are computed directly here; only
the dynamic-capability index $L_d = \delta_{pI0} - K$ is shared with
[`phonometry.dynamic_capability_index`](/phonometry/reference/api/power/intensity/#dynamic_capability_index).

Qualification criteria per band (Annex B), where `K` is 10 (engineering) or
7 (survey) per Table 1, criterion 2 is mandatory for grade 2 and optional for
grade 3, and the per-segment repeatability limit `s` comes from Table 2:

$$
\text{criterion 1:} \quad L_d > F_{pI}, \qquad L_d = \delta_{pI0} - K
$$

$$
\text{criterion 2:} \quad F_{+/-} \le 3~\text{dB}
$$

$$
\text{criterion 3:} \quad |L_{Wi}(1) - L_{Wi}(2)| \le s \quad \text{per segment}
$$

A band achieves the **engineering** grade when criteria 1, 2 and 3 hold, the
**survey** grade when criteria 1 and 3 hold (clause 8.4), otherwise none.
An A-weighted sound power level omits, besides the non-determinable
$P \le 0$ bands, the bands in which criteria 1 and/or 2 are not
satisfied (clause 10.6 b).

ISO 9614-3:2002 is the same method at precision grade, and that is why it is
filed here and not in a module of its own: the same probe swept over the same
enclosing surface, the same partial powers summed the same way (equations (5),
(8), (9)), only a stricter procedure around them. Part 3 recognises a single
grade, fixes the bias-error factor at $K = 10$ dB, takes as its input the
result of the two scans that its repeatability criterion compares, and refers
the level to the reference atmosphere:

$$
L_{W0} = L_W - 15 \log_{10}\!\left( \frac{B}{101325} \cdot \frac{296.15}{273.15 + \theta} \right) \tag{Eq. 10}
$$

Qualification is stricter in the same proportion: four field indicators
(Annex B) feed five acceptance criteria evaluated per band (Annex C), and a
band that satisfies the scan-density criterion 5 is qualified as a final
result even where the field non-uniformity of criterion 4 is not met
(C.1.6.2). The exclusion of the net-negative bands is the one rule both parts
state alike (clause 9.2).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## precision_field_indicators

```python
precision_field_indicators(
    segment_intensity: np.ndarray,
    segment_pressure_levels: np.ndarray,
    *,
    time_window_intensity: np.ndarray | None = None,
) -> PrecisionFieldIndicators
```

ISO 9614-3:2002 Annex B field indicators from segment data.

Over the `N` segments of the whole measurement surface (per band):

$$
\overline{L_p} = 10 \log_{10}\!\left[ \frac{1}{N} \sum_j 10^{0.1 L_{pj}} \right] \tag{Eq. B.4}
$$

$$
L_{|I_n|} = 10 \log_{10}\!\left[ \frac{1}{N} \sum_j \frac{|I_{nj}|}{I_0} \right] \tag{Eq. B.5}
$$

$$
L_{I_n} = 10 \log_{10}\!\left[ \frac{1}{I_0} \left| \frac{1}{N} \sum_j I_{nj} \right| \right] \tag{Eq. B.7}
$$

$$
F_{pI_n}^{\mathrm{unsigned}} = \overline{L_p} - L_{|I_n|} \tag{Eq. B.3}
$$

$$
F_{pI_n}^{\mathrm{signed}} = \overline{L_p} - L_{I_n} \tag{Eq. B.6}
$$

$$
F_S = \frac{1}{\overline{I_n}} \sqrt{ \frac{1}{N-1} \sum_j \left( I_{nj} - \overline{I_n} \right)^2 } \tag{Eq. B.8}
$$

With `time_window_intensity` (an `(M, NB)` array of window-averaged
intensities) the temporal-variability indicator `FT` (Eq. B.1) is also
returned.

**Parameters**

| Name | Description |
| :--- | :--- |
| `segment_intensity` | `(N, NB)` signed segment normal intensity, W/m^2. |
| `segment_pressure_levels` | `(N, NB)` segment pressure levels, dB. |
| `time_window_intensity` | Optional `(M, NB)` window intensities for FT. |

**Returns:** [`PrecisionFieldIndicators`](/phonometry/reference/api/power/sound-power-intensity/#precisionfieldindicators).

## precision_qualification

```python
precision_qualification(
    indicators: PrecisionFieldIndicators,
    *,
    scan_intensity_level_1: np.ndarray | None = None,
    scan_intensity_level_2: np.ndarray | None = None,
    pressure_residual_index: float | np.ndarray | None = None,
    field_nonuniformity_1: np.ndarray | None = None,
    field_nonuniformity_2: np.ndarray | None = None,
    frequencies: np.ndarray | None = None,
    repeatability_limit: float | np.ndarray | None = None,
) -> PrecisionCriteria
```

Evaluate the five ISO 9614-3:2002 Annex C acceptance criteria per band.

**Parameters**

| Name | Description |
| :--- | :--- |
| `indicators` | The [`PrecisionFieldIndicators`](/phonometry/reference/api/power/sound-power-intensity/#precisionfieldindicators) (gives criteria 3 and 4 directly). |
| `scan_intensity_level_1` | `LIn(1)` per band (dB), first scan. |
| `scan_intensity_level_2` | `LIn(2)` per band (dB), second scan; with the first scan and `s` this gives criterion 1 ($\lvert \Delta L \rvert \le s/2$). |
| `pressure_residual_index` | `delta_pI0` (dB), scalar or per band; with $K = 10$ gives `Ld` for criterion 2 ($L_d \ge F_{pI_n}^{\mathrm{signed}}$). |
| `field_nonuniformity_1` | `FS(1)` per band (initial scan density). |
| `field_nonuniformity_2` | `FS(2)` per band (doubled density); with `FS(1)` gives criterion 5. |
| `frequencies` | `(NB,)` nominal mid-band frequencies (Hz), selecting the criterion-1 limit `s` from Table 1. |
| `repeatability_limit` | Override for `s` (dB), scalar or per band. |

**Returns:** [`PrecisionCriteria`](/phonometry/reference/api/power/sound-power-intensity/#precisioncriteria).

## PrecisionCriteria

```python
PrecisionCriteria(
    criterion_1: np.ndarray | None,
    criterion_2: np.ndarray | None,
    criterion_3: np.ndarray,
    criterion_4: np.ndarray,
    criterion_5: np.ndarray | None,
    qualified: np.ndarray | None,
)
```

ISO 9614-3:2002 Annex C acceptance criteria (per band, pass/fail).

Each attribute is a boolean array (True = satisfied) or `None` when its
inputs are absent. `criterion_1` scan repeatability
$\lvert L_{I_n}(1) - L_{I_n}(2) \rvert \le s/2$ (Eq. C.1);
`criterion_2` dynamic-capability
adequacy $L_d \ge F_{pI_n}^{\mathrm{signed}}$ (Eq. C.2);
`criterion_3`
$F_{pI_n}^{\mathrm{signed}} - F_{pI_n}^{\mathrm{unsigned}} \le 3$ dB
(Eq. C.3); `criterion_4`
$F_S \le 2$ (Eq. C.4); `criterion_5` scan-density convergence
$0.83 \le F_S(1)/F_S(2) \le 1.2$ (Eq. C.5). `qualified` is the
conjunction of criteria 1-3 with the field non-uniformity accepted
through criterion 4
or, where evaluated, criterion 5 (C.1.6.2: a band satisfying criterion 5
is qualified as a final result even if $F_S(2) \ge 2$); `None`
unless both criterion 1 and criterion 2 are evaluable.

## PrecisionFieldIndicators

```python
PrecisionFieldIndicators(
    ft: np.ndarray | None,
    f_pi_unsigned: np.ndarray,
    f_pi_signed: np.ndarray,
    fs: np.ndarray,
)
```

ISO 9614-3:2002 Annex B field indicators (per band).

`ft` is the temporal-variability indicator (= F1 of ISO 9614-1, Eq. B.1),
`None` unless time-window intensities are supplied. `f_pi_unsigned` is
the unsigned pressure-intensity indicator (= F2, Eq. B.3, using the mean
magnitude of the segment intensities) and `f_pi_signed` the signed one
(= F3, Eq. B.6, using the algebraic mean); by construction
$F_{pI_n}^{\mathrm{signed}} \ge F_{pI_n}^{\mathrm{unsigned}}$.
`fs` is the field-non-uniformity indicator (= F4, Eq. B.8).

## PrecisionIntensityResult

```python
PrecisionIntensityResult(
    frequencies: np.ndarray | None,
    partial_power: np.ndarray,
    sound_power: np.ndarray,
    sound_power_level: np.ndarray,
    sound_power_level_normalized: np.ndarray,
    not_applicable_band: np.ndarray,
    surface_area: float,
    sound_power_level_a: float,
)
```

Result of an ISO 9614-3:2002 sound-power-by-scanning determination.

`partial_power` is the signed $P_i = I_{n,i} S_i$ per partial
surface and band (Eq. 5); `sound_power` the signed band total
$P = \sum P_i$ (Eq. 8) and `sound_power_level` its level
$L_W = 10 \log_{10}(P/P_0)$ (Eq. 9), `NaN`
where $P \le 0$ (`not_applicable_band` True, clause 9.2).
`sound_power_level_normalized` is `LW0` normalized to 23 deg C /
101 325 Pa (Eq. 10). `sound_power_level_a` is the A-weighted total over
applicable bands (`NaN` without `frequencies` and more than one band).

### PrecisionIntensityResult.plot()

```python
PrecisionIntensityResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the `LW` spectrum; non-applicable bands are hatched/greyed.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### PrecisionIntensityResult.report()

```python
PrecisionIntensityResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
    indicators: PrecisionFieldIndicators | None = None,
    criteria: PrecisionCriteria | None = None,
    residual_index: float | Sequence[float] | np.ndarray | None = None,
) -> str
```

Render an ISO 9614-3 precision sound-power determination fiche.

Writes the one-page sound-power test sheet with what ISO 9614-3:2002
clause 10 asks a report of this method to state: the standard-basis
line naming the precision scanning method and its single accuracy
grade, an optional metadata header (client, noise source, test
environment, instrumentation, air temperature, relative humidity,
barometric pressure and date, clause 10 a) to d)), a per-band table of
the band sound-power level `LW`, the normalized level `LW0` the
standard reports (Eq. 10, clause 10 f) 2)) and the expanded uncertainty
`U` of clause 4.3 (clause 10 f) 4)), the sound-power spectrum
`LW(f)` with the non-applicable bands hatched, the boxed A-weighted
sound power level `LWA` (dB re 1 pW) with the totals, the measurement
surface area and the grade, an optional verdict row against a declared
limit, and a measurement-basis strip carrying the partial-power model,
the meteorological normalization, the Annex B field indicators and the
Annex C criteria.

Supplying `criteria` makes the fiche state what clause 10 f) 2)
requires it to state: the bands whose criteria are not satisfied are
dropped from the A-weighted determination and named on the sheet
alongside the bands the method is not applicable to (clause 9.2). The
boxed `LWA` is then the level of the qualified bands, which differs
from the result's own `sound_power_level_a` whenever a band is
rejected; without `criteria` the fiche boxes the result's value and
says that no qualification was supplied.

The items of clause 10 that are free description rather than computed
quantities (the scan geometry and speed, the drawing of the scanning
paths, the scanning time per partial surface, the calibration and
field-check history, the windscreen, and the probe-reversal checks of
clause 6.2.3) belong in the metadata `notes` and `calibration`
fields; the fiche prints them verbatim in its footer.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata) supplying the header (`client`, `specimen` the noise source, `test_room` the test environment, `instrumentation`, `temperature`, `relative_humidity`, `pressure`, `test_date`), the footer identity (`laboratory`, `operator`, `report_id`, `notes`) and, via `requirement`, a declared A-weighted sound-power limit the fiche checks the result against (lower is better). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When `True` the per-band table adds the four Annex B field indicators and the per-band grade cell. |
| `language` | Fiche language: `"en"` (default) or `"es"`. |
| `indicators` | Optional [`PrecisionFieldIndicators`](/phonometry/reference/api/power/sound-power-intensity/#precisionfieldindicators) from [`precision_field_indicators`](/phonometry/reference/api/power/sound-power-intensity/#precision_field_indicators), tabulated per band with `verbose` and summarised in the basis strip (clause 10 f) 1)). |
| `criteria` | Optional [`PrecisionCriteria`](/phonometry/reference/api/power/sound-power-intensity/#precisioncriteria) from [`precision_qualification`](/phonometry/reference/api/power/sound-power-intensity/#precision_qualification), which decides the per-band grade cell and the clause 10 f) 2) omission described above. |
| `residual_index` | Optional pressure-residual intensity index `delta_pI0` of the probe and analyser (clause 10 d) 5)), a scalar or a per-band array; the strip states it and the dynamic capability `Ld = delta_pI0 - K` that criterion 2 tests. |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"`, `language` is unknown, or a supplied `indicators`, `criteria` or `residual_index` does not span the result's bands. |
| ImportError | If reportlab (or, for the figure, matplotlib) is not installed (`pip install phonometry[report]`). |

## sound_power_intensity

```python
sound_power_intensity(
    normal_intensity: np.ndarray,
    areas: np.ndarray,
    *,
    normal_intensity_2: np.ndarray | None = None,
    pressure_levels: np.ndarray | None = None,
    pressure_residual_index: float | np.ndarray | None = None,
    frequencies: np.ndarray | None = None,
    band_type: BandType = 'third',
    grade: Grade = 'engineering',
    repeatability_limit: float | np.ndarray | None = None,
) -> SoundPowerIntensityResult
```

Sound power level by sound-intensity scanning (ISO 9614-2:1996).

`normal_intensity` is an `(N_seg, N_bands)` array (or `(N_seg,)` for a
single band) of the signed, segment-averaged normal sound intensity
$\langle I_{n,i} \rangle$ (W/m^2), and `areas` the `(N_seg,)`
segment areas `Si` (m^2). The partial powers
$P_i = \langle I_{n,i} \rangle S_i$ are summed to the band sound
power `P` and level $L_W = 10 \log_{10}(P/P_0)$ (equations (12), (6),
(13)). Bands with $P < 0$ are flagged (`negative_band`) and
reported as `NaN` (clause 9.2).

Supplying `normal_intensity_2` (the second grade-2 sweep) makes
`normal_intensity` the first sweep, uses their mean for the partial powers
(Eq. 12), and evaluates the repeatability criterion 3. Supplying
`pressure_levels` (`Lpi`) evaluates `FpI` (Eq. A.1) and, with
`pressure_residual_index` (`delta_pI0`), criterion 1. The per-band
achieved grade (clause 8.4) is returned when both a second sweep and
`delta_pI0` are available. When criteria 1 and 2 are evaluable
(`pressure_levels` and `pressure_residual_index` supplied), the bands
failing them are omitted from the A-weighted total and flagged in
`a_weighting_omitted_bands` (clause 10.6 b); otherwise every determinable
band is summed and a [`SoundPowerWarning`](/phonometry/reference/api/power/sound-power/#soundpowerwarning) notes the missing
screening.

**Parameters**

| Name | Description |
| :--- | :--- |
| `normal_intensity` | `(N_seg, N_bands)` signed normal intensity, W/m^2. |
| `areas` | `(N_seg,)` segment areas `Si`, m^2. |
| `normal_intensity_2` | Optional second sweep, same shape (criterion 3). |
| `pressure_levels` | Optional `(N_seg, N_bands)` `Lpi` (dB) for FpI. |
| `pressure_residual_index` | `delta_pI0` (dB), scalar or per band, for the dynamic-capability index / criterion 1. |
| `frequencies` | `(N_bands,)` nominal band centres (Hz), for the A-weighted total and the Table 2 repeatability limits. |
| `band_type` | `'octave'` or `'third'` (Table 2 lookup). |
| `grade` | `'engineering'` (grade 2) or `'survey'` (grade 3); selects `K` for the reported `Ld` and the criterion-2 warning. |
| `repeatability_limit` | Override for the criterion-3 limit `s` (dB), scalar or per band; defaults to ISO 9614-2 Table 2 by `frequencies` for `'engineering'`. For `'survey'` the default is the A-weighted 4 dB reused per band (extrapolated -- non-normative). |

**Returns:** [`SoundPowerIntensityResult`](/phonometry/reference/api/power/sound-power-intensity/#soundpowerintensityresult).

## sound_power_intensity_precision

```python
sound_power_intensity_precision(
    partial_intensity: np.ndarray,
    areas: np.ndarray,
    *,
    frequencies: np.ndarray | None = None,
    temperature: float = 23.0,
    barometric_pressure: float = 101325.0,
) -> PrecisionIntensityResult
```

Sound power by intensity scanning, precision (ISO 9614-3:2002).

`partial_intensity` is an `(N, NB)` array (or `(N,)` for a single
band) of the signed normal intensity $I_{ni}$ on each of the `N`
partial surfaces (already the two-scan result), and `areas` the `(N,)`
partial surface areas $S_i$. The partial powers
$P_i = I_{ni} S_i$ (Eq. 5) are summed to $P$ (Eq. 8) and
$L_W = 10 \log_{10}(P/P_0)$ (Eq. 9); a band with net $P \le 0$ is
flagged (`not_applicable_band`, clause 9.2) and reported as `NaN`.
$L_{W0}$ normalizes to reference meteorology:

$$
L_{W0} = L_W - 15 \log_{10}\!\left( \frac{B}{101325} \cdot \frac{296.15}{273.15 + \theta} \right) \tag{Eq. 10}
$$

**Parameters**

| Name | Description |
| :--- | :--- |
| `partial_intensity` | `(N, NB)` signed normal intensity, W/m^2. |
| `areas` | `(N,)` partial surface areas `Si`, m^2. |
| `frequencies` | `(NB,)` nominal mid-band frequencies (Hz), for LWA. |
| `temperature` | Air temperature `theta` (deg C), for LW0 (Eq. 10). |
| `barometric_pressure` | Barometric pressure `B` (Pa), for LW0. |

**Returns:** [`PrecisionIntensityResult`](/phonometry/reference/api/power/sound-power-intensity/#precisionintensityresult).

## SoundPowerIntensityResult

```python
SoundPowerIntensityResult(
    frequencies: np.ndarray | None,
    partial_power: np.ndarray,
    partial_power_level: np.ndarray,
    sound_power: np.ndarray,
    sound_power_level: np.ndarray,
    negative_band: np.ndarray,
    surface_pressure_intensity_index: np.ndarray | None,
    negative_partial_power_index: np.ndarray | None,
    repeatability: np.ndarray | None,
    dynamic_capability_index: np.ndarray | None,
    achieved_grade: np.ndarray | None,
    surface_area: float,
    sound_power_level_a: float,
    a_weighting_omitted_bands: np.ndarray | None,
    grade: str,
)
```

Result of an ISO 9614-2:1996 sound-power-by-scanning determination.

`partial_power` is the signed $P_i = \langle I_{n,i} \rangle S_i$
per segment and band (Eq. 12); `partial_power_level` the magnitude level
$10 \log_{10}(|P_i|/P_0)$ (Eq. 8), with the sign carried by
`partial_power`. `sound_power` is the signed band total
$P = \sum P_i$ (Eq. 6) and `sound_power_level` its level
$10 \log_{10}(P/P_0)$ (Eq. 13), `NaN` where $P \le 0$
(`negative_band` True, method not applicable, clause 9.2).
`surface_pressure_intensity_index`
(FpI, Eq. A.1) and `negative_partial_power_index` (F+/-, Eq. A.2) are
per band, `None` when the inputs they need are absent. `repeatability`
is $|L_{Wi}(1) - L_{Wi}(2)|$ per segment and band (criterion 3),
`None` without
a second scan; it is $+\infty$ where the two sweeps reverse the flow
direction on a segment (opposite-sign partial powers), a gross
non-repeatability that criterion 3 must reject even when the magnitudes
happen to match. `dynamic_capability_index` is `Ld` for the requested
grade. `achieved_grade` is the per-band class `'engineering'`/
`'survey'`/`'none'` (clause 8.4), `None` when the qualifying inputs
(`delta_pI0` and a second scan) are absent. `sound_power_level_a` is the
A-weighted total over determinable bands (`NaN` without `frequencies`
and more than one band), which omits the bands failing criteria 1 and/or 2
(clause 10.6 b) whenever those criteria are evaluable.
`a_weighting_omitted_bands` flags the bands so omitted (per band,
`True` = omitted); it is `None` when the criteria inputs
(`pressure_levels` and `pressure_residual_index`) are absent, in which
case every determinable band is summed and a warning is emitted.

### SoundPowerIntensityResult.plot()

```python
SoundPowerIntensityResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the LW spectrum; non-positive bands are hatched as unusable.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### SoundPowerIntensityResult.report()

```python
SoundPowerIntensityResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render an ISO 9614-2 sound-power-by-intensity determination fiche.

Writes a one-page sound-power test sheet: the standard-basis line naming
the intensity-scanning method and its measurement grade (ISO 9614-2:1996
engineering grade 2 or survey grade 3), an optional metadata header
(client, noise source, test environment, instrumentation, climate,
date), a per-band table (nominal octave/one-third-octave frequency and
the intensity-derived band sound-power level `LW`), the sound-power
spectrum `LW(f)` with net-negative bands hatched as unusable, the
boxed A-weighted sound power level `LWA` (dB re 1 pW) with the total
`LW`, the measurement surface area `S` and the determination grade,
an optional verdict row against a declared limit, and a
measurement-basis strip stating the partial-power model, the field
indicators (`FpI`, `F+/-`) and the Annex B qualification criteria.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata) supplying the header (`client`, `specimen` the noise source, `test_room` the test environment, `instrumentation`, `temperature`, `relative_humidity`, `pressure`, `test_date`), the footer identity (`laboratory`, `operator`, `report_id`, `notes`) and, via `requirement`, a declared A-weighted sound-power limit the fiche checks the result against (lower is better). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When `True` the per-band table adds the field indicators `FpI` and `F+/-` and the per-band achieved grade. |
| `language` | Fiche language: `"en"` (default) or `"es"`. |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"` or `language` is unknown. |
| ImportError | If reportlab (or, for the figure, matplotlib) is not installed (`pip install phonometry[report]`). |
