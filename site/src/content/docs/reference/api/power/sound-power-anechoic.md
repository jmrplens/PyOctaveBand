---
title: "emission.sound_power_anechoic"
description: "Sound power level of a noise source in an anechoic or hemi-anechoic room, from sound pressure measurements over an enveloping surface: ISO 3745:2012 (precision, accuracy grade 1)."
sidebar:
  label: "sound_power_anechoic"
---

Sound power level of a noise source in an anechoic or hemi-anechoic room, from
sound pressure measurements over an enveloping surface: ISO 3745:2012
(precision, accuracy grade 1).

This is the precision sibling of ISO 3744:2010 (engineering) and ISO 3746:2010
(survey), and the path from the surface-averaged pressure to the sound power
level is the same one. What sets it apart is the room. In a qualified
(hemi-)free field there is no reflected energy to correct for, so the
environmental correction $K_2$ disappears altogether; in its place the
standard asks for more of everything else. The background correction is
evaluated at every microphone position and every band against a
frequency-dependent criterion (clause 9.4.2), the microphone array is a fixed
40-position equal-area set tabulated for the sphere (Annex D) and the
hemisphere (Annex E), and three meteorological corrections refer the
determination to the reference atmosphere (clause 9.5):

$$
K_{1i} = -10 \log_{10}\!\left( 1 - 10^{-0.1 \Delta L_{pi}} \right) \tag{Eq. 11}
$$

$$
\overline{L_p} = 10 \log_{10}\!\left[ \frac{1}{N_\mathrm{M}} \sum_i 10^{0.1 (L'_{pi} - K_{1i})} \right] \tag{Eq. 12}
$$

$$
L_W = \overline{L_p} + 10 \log_{10}\frac{S}{S_0} + C_1 + C_2 + C_3 \tag{Eq. 14/15}
$$

The measurement surface is a full sphere $S = 4\pi r^2$ in an anechoic
room (Eq. 14) or a hemisphere $S = 2\pi r^2$ over the reflecting plane
of a hemi-anechoic room (Eq. 15), and each tabulated position carries an equal
share of it; unequal partial areas $S_i$ are averaged by Eq. 13 instead.
The A-weighted total is combined with the ISO 3744 Annex E band corrections
(Annex C, Eq. C.1) and the expanded uncertainty
$U = k\sqrt{\sigma_{\mathrm{R}0}^2 + \sigma_\mathrm{omc}^2}$ (Eq. 24/25) takes its
reproducibility standard deviation from Table 3 (anechoic) or Table 2
(hemi-anechoic).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

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

Using the measured static pressure $p_\mathrm{s}$ (kPa) and air temperature
$\theta$ (deg C) form:

$$
C_1 = -10 \log_{10}\frac{p_\mathrm{s}}{p_{\mathrm{s},0}} + 5 \log_{10}\frac{273 + \theta}{\theta_0}, \qquad \theta_0 = 314~\text{K}
$$

$$
C_2 = -10 \log_{10}\frac{p_\mathrm{s}}{p_{\mathrm{s},0}} + 15 \log_{10}\frac{273 + \theta}{\theta_1}, \qquad \theta_1 = 296~\text{K}
$$

$$
C_3 = A_0 \left( 1.0053 - 0.0012\,A_0 \right)^{1.6}, \qquad A_0 = \alpha(f)\,r
$$

$p_{\mathrm{s},0} = 101.325$ kPa. This is the $p_\mathrm{s}$/$\theta$
form of C1 (not the characteristic-impedance form), chosen because it
needs only the measured $p_\mathrm{s}$ and $\theta$ and is consistent
with C2. At the reference conditions (23 deg C, 101.325 kPa)
$C_2 = 0$ exactly while $C_1 = 5 \log_{10}(296/314) = -0.128$ dB.
C3 requires the atmospheric attenuation coefficient $\alpha(f)$
from ISO 9613-1 (not computed here); without it $C_3 = 0$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `temperature` | Air temperature `theta` at the test, in degrees C. |
| `static_pressure` | Static pressure `ps` at the test, in kilopascals. |
| `air_absorption_coefficient` | `a(f)` (dB/m), scalar or per band, for C3; `None` leaves $C_3 = 0$. |
| `radius` | Measurement radius `r` (m), used only in $A_0 = \alpha(f)\,r$. |

**Returns:** [`MeteorologicalCorrection`](/phonometry/reference/api/power/sound-power-anechoic/#meteorologicalcorrection).

## MeteorologicalCorrection

```python
MeteorologicalCorrection(c1: float, c2: float, c3: float | np.ndarray)
```

Meteorological corrections C1, C2, C3 (ISO 3745:2012 Eq. 14 block).

`c1` is the reference-quantity (impedance) correction and `c2` the
radiation-impedance correction, both scalars in decibels; `c3` is the
air-absorption correction (scalar, or per band when the attenuation
coefficient `a(f)` is supplied per band). All three are added to
$\overline{L_p} + 10 \log_{10}(S/S_0)$ to obtain `LW`.

## precision_background_correction

```python
precision_background_correction(
    source_levels: np.ndarray,
    background_levels: np.ndarray,
    frequencies: np.ndarray,
) -> np.ndarray
```

Per-position background correction `K1i` (ISO 3745:2012 Eq. 11).

$K_{1i} = -10 \log_{10}\left( 1 - 10^{-0.1 \Delta L_{pi}} \right)$ with
$\Delta L_{pi} = L'_{pi(\mathrm{ST})} - L_{pi(\mathrm{B})}$
evaluated at each microphone position `i` and band. Above the upper
criterion ($\Delta L_{pi} \ge 15$ dB) the background is negligible
and $K_{1i} = 0$. The lower criterion is frequency dependent:
`10 dB` for one-third-octave mid-bands 250 Hz to 5000 Hz and `6 dB`
for bands $\le 200$ Hz and $\ge 6300$ Hz. Below it,
$K_{1i}$ is clamped to its value
at the criterion (`0.46 dB` and `1.26 dB` respectively), a
[`SoundPowerWarning`](/phonometry/reference/api/power/sound-power/#soundpowerwarning) is emitted and those band results are upper
bounds (clause 9.4.2).

**Parameters**

| Name | Description |
| :--- | :--- |
| `source_levels` | `L'pi(ST)` per position and band, in decibels; shape `(NM, NB)` (or `(NB,)` for one position). |
| `background_levels` | `Lpi(B)` in the same shape (or a single spectrum broadcast to every position). |
| `frequencies` | `(NB,)` nominal mid-band frequencies (Hz), selecting the per-band lower criterion. |

**Returns:** `K1i` per position and band, in decibels, matching the broadcast shape of the inputs.

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

## precision_uncertainty

```python
precision_uncertainty(
    sigma_r0: float | np.ndarray,
    sigma_omc: float = 0.0,
    coverage_factor: float = 2.0,
) -> float | np.ndarray
```

Expanded uncertainty $U = k \sigma_\mathrm{tot}$ (ISO 3745:2012).

ISO 3745:2012 Eq. 24/25:
$\sigma_\mathrm{tot} = \sqrt{\sigma_{\mathrm{R}0}^2 + \sigma_\mathrm{omc}^2}$ and
$U = k \sigma_\mathrm{tot}$, with $k = 2$ (95 %, two-sided) or
$k = 1.6$ (95 %, one-sided, when comparing to a limit).

**Parameters**

| Name | Description |
| :--- | :--- |
| `sigma_r0` | Reproducibility standard deviation (Tables 2/3), dB. |
| `sigma_omc` | Operating/mounting standard deviation `sigma_omc`, dB. |
| `coverage_factor` | `k` (typically 2 or 1.6). |

**Returns:** `U` in decibels, scalar or per band matching `sigma_r0`.

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

`sound_power_level` is the per-band
$L_W = \overline{L_p} + 10\log_{10}(S/S_0) + C_1 + C_2 + C_3$
(Eq. 14/15). `surface_pressure_level` is the surface time-
averaged level `Lp_bar` after the per-position background correction
(Eq. 12/13); `mean_pressure_level` the same energy average of the raw
(uncorrected) position levels. `background_correction` is the
per-position per-band `K1i` (Eq. 11), shape `(NM, NB)`.
`c1`/`c2`/`c3` are the
meteorological corrections (Eq. 14). `directivity_index` is
$DI_i = L_{pi} - \overline{L_p}$
per position and band (Eq. 21); `non_uniformity_index` the
per-band `VIr` sample standard deviation about the arithmetic mean
(Eq. 22). `uncertainty` is the A-weighted expanded uncertainty
$U = k\sqrt{\sigma_{\mathrm{R}0}^2 + \sigma_\mathrm{omc}^2}$
(Eq. 24/25) and `uncertainty_bands` the
per-band value (`NaN` without `frequencies`).
`sound_power_level_a` is the A-weighted total `LWA` (Eq. C.1).

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
levels $L'_{pi(\mathrm{ST})}$ (one row per microphone, one column
per band). Each position is background-corrected by $K_{1i}$
(Eq. 11, from `background_levels` and `frequencies`), the corrected
levels are surface-averaged (equal-area Eq. 12, or area-weighted Eq. 13
when `areas` are given) and combined with the surface area and the
meteorological corrections:

$$
L_W = 10 \log_{10}\!\left[ \frac{1}{N_\mathrm{M}} \sum_i 10^{0.1 (L'_{pi} - K_{1i})} \right] + 10 \log_{10}\frac{S}{S_0} + C_1 + C_2 + C_3
$$

$S = 4\pi r^2$ for a `'sphere'` (anechoic, Eq. 14) or
$2\pi r^2$ for a `'hemisphere'` (hemi-anechoic, Eq. 15). There is
no ISO 3744 `K2`
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
| `air_absorption_coefficient` | `a(f)` (dB/m) for C3, scalar or per band; `None` leaves $C_3 = 0$. |
| `sigma_omc` | Operating/mounting standard deviation, dB. |
| `coverage_factor` | `k` (2 two-sided, 1.6 one-sided). |

**Returns:** [`PrecisionSoundPowerResult`](/phonometry/reference/api/power/sound-power-anechoic/#precisionsoundpowerresult).
