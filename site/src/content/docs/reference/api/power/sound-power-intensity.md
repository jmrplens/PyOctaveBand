---
title: "emission.sound_power_intensity"
description: "Sound power level of a noise source by sound-intensity scanning: ISO 9614-2:1996 (engineering, grade 2; survey/control, grade 3)."
sidebar:
  label: "sound_power_intensity"
---

Sound power level of a noise source by sound-intensity **scanning**:
ISO 9614-2:1996 (engineering, grade 2; survey/control, grade 3).

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

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

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
