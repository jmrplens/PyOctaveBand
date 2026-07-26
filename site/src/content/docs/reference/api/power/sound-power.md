---
title: "emission.sound_power"
description: "Sound power level of a noise source from sound pressure measurements over an enveloping measurement surface: ISO 3744:2010 (engineering, accuracy grade 2) and ISO 3746:2010 (survey, accuracy grade 3)."
sidebar:
  label: "sound_power"
---

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

Sound power level of a noise source from sound pressure measurements over an
enveloping measurement surface: ISO 3744:2010 (engineering, accuracy grade 2)
and ISO 3746:2010 (survey, accuracy grade 3).

The source stands on one (or more) reflecting plane(s). Sound pressure levels
are measured at an array of microphone positions on a hypothetical surface of
area `S` enveloping the source (a hemisphere or a right parallelepiped). The
sound power level follows from the surface-averaged pressure level and the
surface area (ISO 3744:2010 clause 8.2, equations (12), (16)-(18)):

```text
Lp_mean = 10*lg( (1/NM) * sum_i 10^(0,1*Lpi) )      (energy average, Eq. 12)
K1      = -10*lg( 1 - 10^(-0,1*dLp) )               (background, Eq. 16)
K2      = 10*lg( 1 + 4*S/A )                        (environment, Eq. A.2)
Lp      = Lp_mean - K1 - K2                         (surface SPL, Eq. 17)
LW      = Lp + 10*lg(S/S0)     S0 = 1 m^2           (Eq. 18)
```

The measurement surface area is a closed form of the source geometry: a full
hemisphere `S = 2*pi*r^2` (half `pi*r^2`, quarter `pi*r^2/2`) for one,
two or three reflecting planes (ISO 3744 clause 7.2.3); a parallelepiped
`S = 4(ab+bc+ca)` with `a = 0,5*l1+d`, `b = 0,5*l2+d`, `c = l3+d` for
one plane (clause 7.2.4, equations (9)-(11)).

The A-weighted sound power level is combined from band levels with the
A-weighting band corrections `Ck` of ISO 3744 Annex E (Tables E.1/E.2):

```text
LWA = 10*lg( sum_k 10^(0,1*(LWk + Ck)) )            (Eq. E.1)
```

ISO 3746:2010 shares the surfaces, the energy average and the LW/K1/K2 forms
but is coarser: fewer microphone positions (clause 8.2.1), a background
criterion of 3 dB instead of 6 dB (clause 8.4.1) and validity up to
`K2A <= 7 dB` instead of `4 dB` (clause 4.3).

## background_noise_correction

```python
background_noise_correction(
    source_levels: np.ndarray,
    background_levels: np.ndarray,
    grade: Grade = 'engineering',
) -> np.ndarray
```

Background-noise correction `K1` per band (ISO 3744:2010 Eq. 16).

`K1 = -10*lg(1 - 10^(-0,1*dLp))` with `dLp = source - background`. For
`dLp` strictly above the upper criterion (15 dB engineering, 10 dB
survey) the background is negligible and `K1 = 0`; at the criterion
itself Eq. (16) still applies (ISO 3744:2010, 8.2.3: 6 dB \<= dLp \<= 15 dB;
ISO 3746:2010, 8.3.3: 3 dB \<= dLp \<= 10 dB). For `dLp` below the lower
criterion (6 dB engineering, 3 dB survey) the accuracy is reduced: `K1`
is clamped to its value at that criterion and a [`SoundPowerWarning`](/phonometry/reference/api/power/sound-power/#soundpowerwarning)
is emitted, the result then being an upper bound (clause 8.2.3).

**Parameters**

| Name | Description |
| :--- | :--- |
| `source_levels` | Levels with the source operating, in decibels. |
| `background_levels` | Background-noise levels, in decibels. |
| `grade` | `'engineering'` (ISO 3744) or `'survey'` (ISO 3746). |

**Returns:** `K1` per band, in decibels.

## environmental_correction

```python
environmental_correction(
    surface_area: float,
    *,
    absorption_area: float | np.ndarray | None = None,
    reverberation_time: float | np.ndarray | None = None,
    volume: float | None = None,
    mean_absorption_coefficient: float | np.ndarray | None = None,
    room_surface: float | None = None,
    room_volume: float | str | None = 'deprecated',
) -> float | np.ndarray
```

Environmental correction `K2` (ISO 3744:2010 Eq. A.2).

`K2 = 10*lg(1 + 4*S/A)` where `A` is the equivalent sound absorption
area of the room. `A` is taken directly from `absorption_area`, or from
the Sabine reverberation time `A = 0,16*V/T` (Eq. A.3, `reverberation_
time` + `volume`), or from the mean absorption coefficient
`A = alpha*Sv` (Eq. A.7, `mean_absorption_coefficient` + `room_
surface`). With no room data the field is treated as free and `K2 = 0`;
supplying only one member of a pair raises `ValueError` rather than
silently falling back to the free-field result.

The room absorption is frequency dependent (`T`, `alpha` and hence `A`
vary with the band). Passing `absorption_area`, `reverberation_time` or
`mean_absorption_coefficient` as a per-band array returns `K2` per band
with that shape; scalar inputs return a scalar, unchanged.

**Parameters**

| Name | Description |
| :--- | :--- |
| `surface_area` | Measurement surface area `S`, in square metres. |
| `absorption_area` | Equivalent absorption area `A` (m^2), scalar or per band. |
| `reverberation_time` | Sabine `T` (s), scalar or per band, with `volume` (Eq. A.3). |
| `volume` | Room volume `V` (m^3), with `reverberation_time`. |
| `mean_absorption_coefficient` | `alpha` in (0, 1], scalar or per band, with `room_surface` (Eq. A.7). |
| `room_surface` | Room boundary area `Sv` (m^2), with `alpha`. |
| `room_volume` | Deprecated alias of `volume` (remove in 4.0). |

**Returns:** `K2` in decibels; a scalar for scalar inputs, otherwise an array per band.

## measurement_positions

```python
measurement_positions(
    surface: Surface,
    *,
    radius: float | None = None,
    reflecting_planes: int = 1,
    tones: bool = True,
    grade: Grade = 'engineering',
) -> np.ndarray
```

Normative microphone coordinates on the measurement surface.

For a `'hemisphere'` the coordinates come from ISO 3744:2010 Annex B:
Table B.1 for sources that may emit discrete tones (`tones=True`) and
Table B.2 for broadband sources. The engineering grade uses the 10 key
positions for one reflecting plane (5 for two, 3 for three); the survey
grade uses the reduced arrays of ISO 3746:2010 clause 8.2.1 (positions
4, 5, 6, 10 for one plane). Coordinates are scaled by `radius` and
returned as an `(N, 3)` array of Cartesian `(x, y, z)` in metres.

**Parameters**

| Name | Description |
| :--- | :--- |
| `surface` | `'hemisphere'` (only shape with a coordinate table). |
| `radius` | Hemisphere radius `r`, in metres. |
| `reflecting_planes` | Number of reflecting planes (1, 2 or 3). |
| `tones` | If True use Table B.1, else Table B.2. |
| `grade` | `'engineering'` or `'survey'`. |

**Returns:** `(N, 3)` microphone coordinates, in metres.

## meteorological_corrections

```python
meteorological_corrections(
    temperature: float = 23.0,
    static_pressure: float = 101.325,
    *,
    air_absorption_coefficient: float | np.ndarray | None = None,
    radius: float = 1.0,
) -> MeteorologicalCorrection
```

Meteorological corrections C1, C2, C3 (ISO 3745:2012 Eq. 14 block).

Using the measured static pressure `ps` (kPa) and air temperature
`theta` (deg C) form:

```text
C1 = -10*lg(ps/ps0) + 5*lg((273+theta)/theta0)     theta0 = 314 K
C2 = -10*lg(ps/ps0) + 15*lg((273+theta)/theta1)    theta1 = 296 K
C3 = A0*(1,005 3 - 0,001 2*A0)^1,6                  A0 = a(f)*r
```

`ps0 = 101,325 kPa`. This is the `ps`/`theta` form of C1 (not the
characteristic-impedance form), chosen because it needs only the measured
`ps` and `theta` and is consistent with C2. At the reference conditions
(23 deg C, 101,325 kPa) `C2 = 0` exactly while `C1 = 5*lg(296/314) =
-0,128 dB`. C3 requires the atmospheric attenuation coefficient `a(f)`
from ISO 9613-1 (not computed here); without it `C3 = 0`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `temperature` | Air temperature `theta` at the test, in degrees C. |
| `static_pressure` | Static pressure `ps` at the test, in kilopascals. |
| `air_absorption_coefficient` | `a(f)` (dB/m), scalar or per band, for C3; `None` leaves `C3 = 0`. |
| `radius` | Measurement radius `r` (m), used only in `A0 = a(f)*r`. |

**Returns:** [`MeteorologicalCorrection`](/phonometry/reference/api/power/sound-power/#meteorologicalcorrection).

## MeteorologicalCorrection

```python
MeteorologicalCorrection(c1: float, c2: float, c3: float | np.ndarray)
```

Meteorological corrections C1, C2, C3 (ISO 3745:2012 Eq. 14 block).

`c1` is the reference-quantity (impedance) correction and `c2` the
radiation-impedance correction, both scalars in decibels; `c3` is the
air-absorption correction (scalar, or per band when the attenuation
coefficient `a(f)` is supplied per band). All three are added to
`Lp_bar + 10*lg(S/S0)` to obtain `LW`.

## precision_background_correction

```python
precision_background_correction(
    source_levels: np.ndarray,
    background_levels: np.ndarray,
    frequencies: np.ndarray,
) -> np.ndarray
```

Per-position background correction `K1i` (ISO 3745:2012 Eq. 11).

`K1i = -10*lg(1 - 10^(-0,1*dLpi))` with `dLpi = L'pi(ST) - Lpi(B)`
evaluated at each microphone position `i` and band. Above the upper
criterion (`dLpi >= 15 dB`) the background is negligible and
`K1i = 0`. The lower criterion is frequency dependent: `10 dB` for
one-third-octave mid-bands 250 Hz to 5000 Hz and `6 dB` for bands
`<= 200 Hz` and `>= 6300 Hz`. Below it, `K1i` is clamped to its value
at the criterion (`0,46 dB` and `1,26 dB` respectively), a
[`SoundPowerWarning`](/phonometry/reference/api/power/sound-power/#soundpowerwarning) is emitted and those band results are upper
bounds (clause 9.4.2).

**Parameters**

| Name | Description |
| :--- | :--- |
| `source_levels` | `L'pi(ST)` per position and band, in decibels; shape `(NM, NB)` (or `(NB,)` for one position). |
| `background_levels` | `Lpi(B)` in the same shape (or a single spectrum broadcast to every position). |
| `frequencies` | `(NB,)` nominal mid-band frequencies (Hz), selecting the per-band lower criterion. |

**Returns:** `K1i` per position and band, in decibels, matching the broadcast shape of the inputs.

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

```text
Lp_bar       = 10*lg( (1/N) sum 10^(0,1*Lpj) )          (Eq. B.4)
LIn_unsigned = 10*lg( (1/N) sum |In_j| / I0 )           (Eq. B.5)
LIn_signed   = 10*lg( |(1/N) sum In_j| / I0 )           (Eq. B.7)
F_pIn(unsigned) = Lp_bar - LIn_unsigned                 (Eq. B.3)
F_pIn(signed)   = Lp_bar - LIn_signed                   (Eq. B.6)
FS = (1/In_bar) sqrt( (1/(N-1)) sum (In_j - In_bar)^2 ) (Eq. B.8)
```

With `time_window_intensity` (an `(M, NB)` array of window-averaged
intensities) the temporal-variability indicator `FT` (Eq. B.1) is also
returned.

**Parameters**

| Name | Description |
| :--- | :--- |
| `segment_intensity` | `(N, NB)` signed segment normal intensity, W/m^2. |
| `segment_pressure_levels` | `(N, NB)` segment pressure levels, dB. |
| `time_window_intensity` | Optional `(M, NB)` window intensities for FT. |

**Returns:** [`PrecisionFieldIndicators`](/phonometry/reference/api/power/sound-power/#precisionfieldindicators).

## precision_positions

```python
precision_positions(
    surface: PrecisionSurface,
    *,
    radius: float | None = None,
    array: PrecisionArray = 'general',
    count: int = 40,
) -> np.ndarray
```

Normative ISO 3745:2012 microphone coordinates, scaled by `radius`.

For a `'sphere'` (anechoic room) the coordinates come from Annex D
Table D.1; for a `'hemisphere'` (hemi-anechoic room) from Annex E
Table E.1 (`array='general'`) or Table E.2 (`array='broadband'`, an
omnidirectional broadband source). Positions 1-20 are the primary array;
the full 40 add the mirror set (positions 21-40), used when the band-SPL
spread exceeds NM/2 (clause 9.3). Each row is a unit vector (self-checked)
scaled to metres by `radius`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `surface` | `'sphere'` or `'hemisphere'`. |
| `radius` | Measurement radius `r`, in metres. |
| `array` | `'general'` (Table E.1) or `'broadband'` (Table E.2); ignored for a sphere (only Table D.1 exists). |
| `count` | `20` (primary array) or `40` (full array). |

**Returns:** `(count, 3)` microphone coordinates, in metres.

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
| `indicators` | The [`PrecisionFieldIndicators`](/phonometry/reference/api/power/sound-power/#precisionfieldindicators) (gives criteria 3 and 4 directly). |
| `scan_intensity_level_1` | `LIn(1)` per band (dB), first scan. |
| `scan_intensity_level_2` | `LIn(2)` per band (dB), second scan; with the first scan and `s` this gives criterion 1 (`\|dL\| <= s/2`). |
| `pressure_residual_index` | `delta_pI0` (dB), scalar or per band; with `K = 10` gives `Ld` for criterion 2 (`Ld >= F_pIn(signed)`). |
| `field_nonuniformity_1` | `FS(1)` per band (initial scan density). |
| `field_nonuniformity_2` | `FS(2)` per band (doubled density); with `FS(1)` gives criterion 5. |
| `frequencies` | `(NB,)` nominal mid-band frequencies (Hz), selecting the criterion-1 limit `s` from Table 1. |
| `repeatability_limit` | Override for `s` (dB), scalar or per band. |

**Returns:** [`PrecisionCriteria`](/phonometry/reference/api/power/sound-power/#precisioncriteria).

## precision_uncertainty

```python
precision_uncertainty(
    sigma_r0: float | np.ndarray,
    sigma_omc: float = 0.0,
    coverage_factor: float = 2.0,
) -> float | np.ndarray
```

Expanded uncertainty `U = k*sqrt(sigma_R0^2 + sigma_omc^2)`.

ISO 3745:2012 Eq. 24/25: `sigma_tot = sqrt(sigma_R0^2 + sigma_omc^2)` and
`U = k*sigma_tot`, with `k = 2` (95 %, two-sided) or `k = 1,6` (95 %,
one-sided, when comparing to a limit).

**Parameters**

| Name | Description |
| :--- | :--- |
| `sigma_r0` | Reproducibility standard deviation (Tables 2/3), dB. |
| `sigma_omc` | Operating/mounting standard deviation `sigma_omc`, dB. |
| `coverage_factor` | `k` (typically 2 or 1,6). |

**Returns:** `U` in decibels, scalar or per band matching `sigma_r0`.

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
`|LIn(1)-LIn(2)| <= s/2` (Eq. C.1); `criterion_2` dynamic-capability
adequacy `Ld >= F_pIn(signed)` (Eq. C.2); `criterion_3`
`F_pIn(signed) - F_pIn(unsigned) <= 3 dB` (Eq. C.3); `criterion_4`
`FS <= 2` (Eq. C.4); `criterion_5` scan-density convergence
`0,83 <= FS(1)/FS(2) <= 1,2` (Eq. C.5). `qualified` is the conjunction
of criteria 1-3 with the field non-uniformity accepted through criterion 4
or, where evaluated, criterion 5 (C.1.6.2: a band satisfying criterion 5
is qualified as a final result even if `FS(2) >= 2`); `None` unless
both criterion 1 and criterion 2 are evaluable.

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
`f_pi_signed >= f_pi_unsigned`. `fs` is the field-non-uniformity
indicator (= F4, Eq. B.8).

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

`partial_power` is the signed `Pi = In_i*Si` per partial surface and
band (Eq. 5); `sound_power` the signed band total `P = sum Pi` (Eq. 8)
and `sound_power_level` its level `LW = 10*lg(P/P0)` (Eq. 9), `NaN`
where `P <= 0` (`not_applicable_band` True, clause 9.2).
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

## PrecisionSoundPowerResult

```python
PrecisionSoundPowerResult(
    frequencies: np.ndarray | None,
    sound_power_level: np.ndarray,
    surface_pressure_level: np.ndarray,
    mean_pressure_level: np.ndarray,
    background_correction: np.ndarray,
    c1: float,
    c2: float,
    c3: np.ndarray,
    directivity_index: np.ndarray,
    non_uniformity_index: np.ndarray,
    surface_area: float,
    surface: str,
    sound_power_level_a: float,
    uncertainty: float,
    uncertainty_bands: np.ndarray,
    coverage_factor: float,
)
```

Result of an ISO 3745:2012 (precision) sound power determination.

`sound_power_level` is the per-band `LW = Lp_bar + 10*lg(S/S0) + C1 +
C2 + C3` (Eq. 14/15). `surface_pressure_level` is the surface time-
averaged level `Lp_bar` after the per-position background correction
(Eq. 12/13); `mean_pressure_level` the same energy average of the raw
(uncorrected) position levels. `background_correction` is the per-position
per-band `K1i` (Eq. 11), shape `(NM, NB)`. `c1`/`c2`/`c3` are the
meteorological corrections (Eq. 14). `directivity_index` is `DIi = Lpi -
Lp_bar` per position and band (Eq. 21); `non_uniformity_index` the
per-band `VIr` sample standard deviation about the arithmetic mean
(Eq. 22). `uncertainty` is the A-weighted expanded uncertainty `U =
k*sqrt(sigma_R0^2 + sigma_omc^2)` (Eq. 24/25) and `uncertainty_bands` the
per-band value (`NaN` without `frequencies`). `sound_power_level_a` is
the A-weighted total `LWA` (Eq. C.1).

### PrecisionSoundPowerResult.plot()

```python
PrecisionSoundPowerResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the precision `LW` spectrum with the A-weighted total.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### PrecisionSoundPowerResult.report()

```python
PrecisionSoundPowerResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render an ISO 3745:2012 precision sound-power fiche to a PDF.

Writes the same one-page sound-power test sheet as
[`SoundPowerResult.report`](/phonometry/reference/api/power/sound-power/#soundpowerresultreport), with the standard-basis line naming the
precision method in an anechoic or hemi-anechoic room (ISO 3745:2012,
accuracy grade 1) and the measurement-basis strip stating the applied
meteorological corrections `C1`/`C2`/`C3` instead of the ISO 3744
`K1`/`K2`. The per-band table shows the surface time-averaged level
`Lp` and the band sound-power level `LW`; `verbose` adds the
energy-averaged level `Lp'`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata) supplying the header and footer identity and, via `requirement`, a declared A-weighted sound-power limit (lower is better). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When `True` the per-band table adds the energy-averaged level `Lp'`. |
| `language` | Fiche language: `"en"` (default) or `"es"`. |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"` or `language` is unknown. |
| ImportError | If reportlab (or, for the figure, matplotlib) is not installed (`pip install phonometry[report]`). |

## sound_power_anechoic

```python
sound_power_anechoic(
    levels_positions: np.ndarray,
    surface: PrecisionSurface,
    *,
    radius: float | None = None,
    background_levels: np.ndarray | None = None,
    frequencies: np.ndarray | None = None,
    areas: np.ndarray | None = None,
    temperature: float = 23.0,
    static_pressure: float = 101.325,
    air_absorption_coefficient: float | np.ndarray | None = None,
    sigma_omc: float = 0.0,
    coverage_factor: float = 2.0,
) -> PrecisionSoundPowerResult
```

Sound power level in an (hemi-)anechoic room (ISO 3745:2012, precision).

`levels_positions` is an `(NM, NB)` array of time-averaged position
levels `L'pi(ST)` (one row per microphone, one column per band). Each
position is background-corrected by `K1i` (Eq. 11, from
`background_levels` and `frequencies`), the corrected levels are
surface-averaged (equal-area Eq. 12, or area-weighted Eq. 13 when `areas`
are given) and combined with the surface area and the meteorological
corrections:

```text
LW = 10*lg((1/NM) sum 10^(0,1*(L'pi - K1i))) + 10*lg(S/S0) + C1+C2+C3
```

`S = 4*pi*r^2` for a `'sphere'` (anechoic, Eq. 14) or `2*pi*r^2` for a
`'hemisphere'` (hemi-anechoic, Eq. 15). There is no ISO 3744 `K2`
environmental term. The reproducibility `sigma_R0` is taken from Table 3
(sphere/anechoic) or Table 2 (hemisphere/hemi-anechoic).

**Parameters**

| Name | Description |
| :--- | :--- |
| `levels_positions` | `(NM, NB)` position levels, in decibels. |
| `surface` | `'sphere'` or `'hemisphere'`. |
| `radius` | Measurement radius `r`, in metres. |
| `background_levels` | `(NM, NB)` (or single-spectrum) background levels for `K1i`; requires `frequencies`. |
| `frequencies` | `(NB,)` nominal mid-band frequencies (Hz), for the K1 criterion, the A-weighted total and the per-band uncertainty. |
| `areas` | `(NM,)` partial areas `Si` for the area-weighted average (Eq. 13); omit for the equal-area average (Eq. 12). |
| `temperature` | Air temperature `theta` (deg C), for C1/C2. |
| `static_pressure` | Static pressure `ps` (kPa), for C1/C2. |
| `air_absorption_coefficient` | `a(f)` (dB/m) for C3, scalar or per band; `None` leaves `C3 = 0`. |
| `sigma_omc` | Operating/mounting standard deviation, dB. |
| `coverage_factor` | `k` (2 two-sided, 1,6 one-sided). |

**Returns:** [`PrecisionSoundPowerResult`](/phonometry/reference/api/power/sound-power/#precisionsoundpowerresult).

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
band) of the signed normal intensity `In_i` on each of the `N` partial
surfaces (already the two-scan result), and `areas` the `(N,)` partial
surface areas `Si`. The partial powers `Pi = In_i*Si` (Eq. 5) are summed
to `P` (Eq. 8) and `LW = 10*lg(P/P0)` (Eq. 9); a band with net `P <= 0`
is flagged (`not_applicable_band`, clause 9.2) and reported as `NaN`.
`LW0` normalizes to reference meteorology (Eq. 10):

```text
LW0 = LW - 15*lg( (B/101325) * (296,15/(273,15+theta)) )
```

**Parameters**

| Name | Description |
| :--- | :--- |
| `partial_intensity` | `(N, NB)` signed normal intensity, W/m^2. |
| `areas` | `(N,)` partial surface areas `Si`, m^2. |
| `frequencies` | `(NB,)` nominal mid-band frequencies (Hz), for LWA. |
| `temperature` | Air temperature `theta` (deg C), for LW0 (Eq. 10). |
| `barometric_pressure` | Barometric pressure `B` (Pa), for LW0. |

**Returns:** [`PrecisionIntensityResult`](/phonometry/reference/api/power/sound-power/#precisionintensityresult).

## sound_power_pressure

```python
sound_power_pressure(
    levels_positions: np.ndarray,
    surface: Surface,
    *,
    radius: float | None = None,
    dimensions: tuple[float, float, float] | None = None,
    distance: float | None = None,
    reflecting_planes: int = 1,
    background_levels: np.ndarray | None = None,
    frequencies: np.ndarray | None = None,
    absorption_area: float | None = None,
    reverberation_time: float | None = None,
    volume: float | None = None,
    mean_absorption_coefficient: float | None = None,
    room_surface: float | None = None,
    grade: Grade = 'engineering',
    omc_uncertainty: float = 0.0,
    room_volume: float | str | None = 'deprecated',
) -> SoundPowerResult
```

Sound power level from surface pressure levels (ISO 3744/3746:2010).

`levels_positions` is an `(NM, NB)` array of time-averaged sound
pressure levels: one row per microphone position, one column per frequency
band (or a single column for a directly measured A-weighted level). The
surface-averaged level is corrected for background noise (`K1`, from
`background_levels`) and for the test environment (`K2`, from the room
absorption data) and combined with the measurement surface area:

```text
LW = 10*lg((1/NM) sum 10^(0,1*Lpi)) - K1 - K2 + 10*lg(S/S0)
```

The surface area `S` is computed from the geometry: `radius` for a
`'hemisphere'` (clause 7.2.3) or `dimensions` + `distance` for a
`'box'` (clause 7.2.4). When `frequencies` are given the A-weighted
sound power level is combined via ISO 3744 Annex E.

**Parameters**

| Name | Description |
| :--- | :--- |
| `levels_positions` | `(NM, NB)` sound pressure levels, in decibels. |
| `surface` | `'hemisphere'` or `'box'`. |
| `radius` | Hemisphere radius `r` (metres), for `surface='hemisphere'`. |
| `dimensions` | Reference box `(l1, l2, l3)` (metres), for `'box'`. |
| `distance` | Measurement distance `d` (metres), for `'box'`. |
| `reflecting_planes` | Number of reflecting planes (1, 2 or 3). |
| `background_levels` | `(NM, NB)` background levels for `K1`, or a single spectrum `(NB,)` / `(1, NB)` broadcast to every position. |
| `frequencies` | Band mid-band frequencies (Hz) for the A-weighted total. |
| `absorption_area` | Equivalent absorption area `A` (m^2) for `K2`. |
| `reverberation_time` | Sabine `T` (s), with `volume`, for `K2`. |
| `volume` | Room volume `V` (m^3), with `reverberation_time`. |
| `mean_absorption_coefficient` | `alpha`, with `room_surface`, for `K2`. |
| `room_surface` | Room boundary area `Sv` (m^2), with `alpha`. |
| `grade` | `'engineering'` (ISO 3744) or `'survey'` (ISO 3746). |
| `omc_uncertainty` | `sigma_omc` (dB), operating/mounting instability. |
| `room_volume` | Deprecated alias of `volume` (remove in 4.0). |

**Returns:** [`SoundPowerResult`](/phonometry/reference/api/power/sound-power/#soundpowerresult).

## SoundPowerResult

```python
SoundPowerResult(
    frequencies: np.ndarray | None,
    sound_power_level: np.ndarray,
    surface_pressure_level: np.ndarray,
    mean_pressure_level: np.ndarray,
    background_correction: np.ndarray,
    environmental_correction: np.ndarray,
    directivity_index: np.ndarray,
    surface_area: float,
    sound_power_level_a: float,
    uncertainty: float,
    grade: str,
)
```

Result of a sound power determination from surface pressure levels.

`sound_power_level` is the per-band `LW` (ISO 3744 Eq. 18);
`surface_pressure_level` the surface SPL `Lp` after the K1/K2
corrections (Eq. 17); `mean_pressure_level` the raw energy-averaged
level `Lp'(ST)` (Eq. 12). `background_correction` (K1) and
`environmental_correction` (K2) are per band. `sound_power_level_a` is
the A-weighted total `LWA` (Eq. E.1), computed only when `frequencies`
are supplied; for a single band it equals `LW`, and for several bands
without `frequencies` it is `NaN` (A-weighting needs the band centres).
`directivity_index` is the apparent directivity index `DIi*` per
microphone position and frequency band, shape `(NM, NB)` (Eq. 7,
evaluated per band per clause 8.4). `uncertainty` is the expanded
uncertainty
`U = 2*sqrt(sigma_R0^2 + sigma_omc^2)` (95 %, ISO 3744 clause 9.5).

### SoundPowerResult.declare()

```python
SoundPowerResult.declare(
    *,
    uncertainty: float | None = None,
    mode: str = 'Operating mode 1',
    emission_pressure_level: float | None = None,
    emission_pressure_uncertainty: float | None = None,
    verification_level: float | None = None,
    machine: str | None = None,
    operating_conditions: str | None = None,
    noise_test_code: str | None = None,
    basic_standards: str | Sequence[str] = (),
    form: DeclarationForm = 'dual-number',
) -> NoiseEmissionDeclaration
```

Build an ISO 4871:1996 noise-emission declaration from this result.

Wraps the A-weighted sound power level `LWA` of this measurement as the
declared measured value `L_WA` of a single operating mode, with the
uncertainty `K_WA` defaulting to the result's own expanded uncertainty
`U` (ISO 3744/3746 clause 9.5). The declared single-number value is
`L_WAd = L_WA + K_WA` (ISO 4871 clause 3.15).

**Parameters**

| Name | Description |
| :--- | :--- |
| `uncertainty` | `K_WA` in decibels; defaults to this result's expanded uncertainty `uncertainty`. |
| `mode` | Operating-mode label for the declaration column. |
| `emission_pressure_level` | Optional A-weighted emission sound pressure level `L_pA` at a work station, in decibels re 20 uPa. |
| `emission_pressure_uncertainty` | `K_pA` in decibels; required with `emission_pressure_level`. |
| `verification_level` | Optional verification measurement `L_1` of the A-weighted sound power level (ISO 4871 clause 6). |
| `machine` | Machine identification (clause 5 a). |
| `operating_conditions` | Operating/mounting conditions (clause 5 c). |
| `noise_test_code` | Noise test code the values were determined to (clause 5 b). |
| `basic_standards` | Basic emission standard(s) used (clause 5 b). |
| `form` | `"dual-number"` (default) or `"single-number"`. |

**Returns:** A single-mode [`NoiseEmissionDeclaration`](/phonometry/reference/api/power/declaration/#noiseemissiondeclaration).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the A-weighted sound power level is not finite (several bands were combined without `frequencies`). |

### SoundPowerResult.plot()

```python
SoundPowerResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the LW spectrum with the A-weighted total annotated.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### SoundPowerResult.report()

```python
SoundPowerResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render an ISO 3744/3746 sound-power determination fiche to a PDF.

Writes a one-page sound-power test sheet: the standard-basis line naming
the applied method and accuracy grade (ISO 3744:2010 engineering grade 2
or ISO 3746:2010 survey grade 3), an optional metadata header (client,
noise source, test environment, instrumentation, climate, date), a
per-band table (nominal octave/one-third-octave frequency, the surface
sound-pressure level `Lp` and the band sound-power level `LW`), the
sound-power spectrum `LW(f)`, the boxed A-weighted sound power level
`LWA` (dB re 1 pW) with the total `LW`, the expanded uncertainty
`U` and the measurement surface area `S`, an optional verdict row
against a declared limit, and a measurement-basis strip stating the
applied background (`K1`) and environmental (`K2`) corrections.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata) supplying the header (`client`, `specimen` the noise source, `test_room` the test environment, `instrumentation`, `temperature`, `relative_humidity`, `pressure`, `test_date`), the footer identity (`laboratory`, `operator`, `report_id`, `notes`) and, via `requirement`, a declared A-weighted sound-power limit the fiche checks the result against (lower is better). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When `True` the per-band table adds the energy-averaged level `Lp'` and the background (`K1`) and environmental (`K2`) corrections. |
| `language` | Fiche language: `"en"` (default) or `"es"`. |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"` or `language` is unknown. |
| ImportError | If reportlab (or, for the figure, matplotlib) is not installed (`pip install phonometry[report]`). |

## SoundPowerWarning

Non-fatal qualification issue in any of the sound-power methods.

Emitted for ISO 3744/3746 background margin below the criterion and for
`K2` beyond the method's validity limit (8.2.3, 4.3.2); for ISO 3741
reverberation-room qualification (room volume vs Table 1, mean absorption)
and microphone/source-position sampling; and for ISO 9614-2 negative total
partial power and unmet field-indicator criteria. Where a lower criterion
is only just met the returned levels represent upper bounds and must be
reported as such.
