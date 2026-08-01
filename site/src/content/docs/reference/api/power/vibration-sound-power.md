---
title: "emission.vibration_sound_power"
description: "Airborne sound power from surface vibration (ISO/TS 7849-1/-2:2009)."
sidebar:
  label: "vibration_sound_power"
---

Airborne sound power from surface vibration (ISO/TS 7849-1/-2:2009).

The airborne sound power a machine radiates through the structure-borne
vibration of its outer surface is estimated from the surface vibratory velocity
and a **radiation factor** `epsilon` (the radiation efficiency). The radiated
power, in watts, is (ISO/TS 7849-1, Equation 6):

$$
P = Z_c \, \langle v^2 \rangle \, S \, \epsilon
$$

with $Z_c$ the characteristic impedance of air and
$\langle v^2 \rangle$ the mean-square vibratory velocity averaged over
the radiating area $S$. The vibratory velocity is reported as a
**level**, in decibels, re $v_0 = 5 \times 10^{-8}$ m/s (Equation 3):

$$
L_v = 10 \log_{10}\frac{\langle v^2 \rangle}{v_0^2} = 20 \log_{10}\frac{v}{v_0}
$$

so the A-weighted sound power level follows in logarithmic form, in decibels
(ISO/TS 7849-1, Equation 12; ISO/TS 7849-2, Equation 15):

$$
L_W = L_v + 10 \log_{10}\frac{S}{S_0} + 10 \log_{10} \epsilon + 10 \log_{10}\frac{Z_{c,n}}{Z_{c,0}}
$$

where $S_0 = 1~\text{m}^2$, the normalized characteristic impedance
$Z_{c,n} = 411~\text{N s/m}^3$ (at 23 degC, 101.3 kPa) and the
reference acoustic impedance $Z_{c,0} = 400~\text{N s/m}^3$ give the
fixed $10 \log_{10}(411/400) = 0.118$ dB term.

The two parts differ only in `epsilon`:

* **Part 1 (survey)** assumes $\epsilon = 1$ and yields the *upper
  limit* `L_W,max` of the radiated power, needing only
  $\langle v^2 \rangle$ and `S`.
* **Part 2 (engineering)** applies a frequency-band radiation factor
  `epsilon_j` determined (per ISO 9614) as
  $\epsilon_j = P_j / (Z_{c,n} \langle v_j^2 \rangle S)$ (Equation 8).

This module feeds the structure-borne source characterisation standards
(ISO 9611, EN 15657, EN 12354-5).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## extraneous_velocity_correction

```python
extraneous_velocity_correction(level_difference: float) -> float
```

Correction K1A for extraneous vibration (ISO/TS 7849-1, Table 2).

$\Delta L_v$ is the difference between the operating and the
extraneous vibratory velocity levels. The correction is subtracted from
the measured level; per the standard $\Delta L_v \ge 10$ dB gives
0 dB, and $\Delta L_v < 3$ dB uses the 3 dB value (the result is
then an upper boundary). The level difference is rounded to the nearest
integer decibel to index the standard's table.

**Parameters**

| Name | Description |
| :--- | :--- |
| `level_difference` | Level difference $\Delta L_v$, in dB. |

**Returns:** The correction `K1A` to subtract, in dB.

## mean_velocity_level

```python
mean_velocity_level(
    levels: ArrayLike,
    areas: ArrayLike | None = None,
) -> float
```

Mean vibratory velocity level over the surface (ISO/TS 7849-1, Eq. 10/11).

With uniformly distributed positions (`areas` is `None`) this is the
energetic average (Equation 10); with per-position partial areas it is the
area-weighted average (Equation 11).

**Parameters**

| Name | Description |
| :--- | :--- |
| `levels` | Velocity levels `L_v,i` at the positions, in dB. |
| `areas` | Partial areas `S_i` (Equation 11), or `None` for the uniform energetic average (Equation 10). |

**Returns:** The mean velocity level, in dB.

## NORMALIZED_IMPEDANCE

*Constant* (`float`).

```python
NORMALIZED_IMPEDANCE = 411.0
```

## radiated_sound_power_level

```python
radiated_sound_power_level(
    velocity_level: ArrayLike,
    area: float,
    *,
    radiation_factor: ArrayLike = 1.0,
    reference_area: float = 1.0,
    normalized_impedance: float = 411.0,
    reference_impedance: float = 400.0,
) -> np.ndarray
```

Radiated sound power level (ISO/TS 7849-1 Eq. 12, -2 Eq. 15).

$$
L_W = L_v + 10 \log_{10}\frac{S}{S_0} + 10 \log_{10} \epsilon + 10 \log_{10}\frac{Z_{c,n}}{Z_{c,0}}
$$

With the default `radiation_factor = 1` this is the Part 1 *upper limit*
`L_W,max`; pass a measured `epsilon` for the Part 2 engineering value.

**Parameters**

| Name | Description |
| :--- | :--- |
| `velocity_level` | Mean vibratory velocity level `L_v` (scalar or array, e.g. per band), in dB re 5e-8 m/s. |
| `area` | Radiating surface area `S`, in m^2 (> 0). |
| `radiation_factor` | Radiation factor `epsilon` (Default: 1.0). |
| `reference_area` | Reference area `S0` (Default: 1 m^2). |
| `normalized_impedance` | `Z_c,n` (Default: 411 N.s/m^3). |
| `reference_impedance` | `Z_c,0` (Default: 400 N.s/m^3). |

**Returns:** The sound power level `L_W`, in dB re 1 pW.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive area, reference area or impedance. |

## radiation_factor

```python
radiation_factor(
    sound_power: ArrayLike,
    area: float,
    mean_square_velocity: ArrayLike,
    *,
    impedance: float = 411.0,
) -> np.ndarray
```

A-weighted radiation factor `epsilon` (ISO/TS 7849-1 Eq. 4, -2 Eq. 8).

$\epsilon = P / (Z_c \langle v^2 \rangle S)$, the sound-radiation
efficiency, from an independently measured radiated power (ISO 9614), the
surface area and the mean-square vibratory velocity.

**Parameters**

| Name | Description |
| :--- | :--- |
| `sound_power` | Radiated airborne sound power `P` (scalar or array), in W. |
| `area` | Radiating surface area `S`, in m^2 (> 0). |
| `mean_square_velocity` | Mean-square vibratory velocity $\langle v^2 \rangle$, in (m/s)^2. |
| `impedance` | Characteristic impedance `Z_c` (Default: 411 N.s/m^3). |

**Returns:** The radiation factor `epsilon` (dimensionless).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive area or impedance. |

## REFERENCE_VELOCITY

*Constant* (`float`).

```python
REFERENCE_VELOCITY = 5e-08
```

## sound_power_from_vibration

```python
sound_power_from_vibration(
    velocity_level: ArrayLike,
    area: float,
    *,
    radiation_factor: ArrayLike = 1.0,
    frequencies: ArrayLike | None = None,
) -> VibrationSoundPowerResult
```

Bundle a sound-power-from-vibration determination (ISO/TS 7849).

**Parameters**

| Name | Description |
| :--- | :--- |
| `velocity_level` | Mean vibratory velocity level `L_v` (per band), dB. |
| `area` | Radiating surface area `S`, in m^2 (> 0). |
| `radiation_factor` | Radiation factor `epsilon` (Default: 1.0 -> the Part 1 upper limit); scalar or per band. |
| `frequencies` | Band centre frequencies, in hertz, or `None`. |

**Returns:** The [`VibrationSoundPowerResult`](/phonometry/reference/api/power/vibration-sound-power/#vibrationsoundpowerresult).

## velocity_level

```python
velocity_level(
    velocity: ArrayLike,
    *,
    reference: float = 5e-08,
) -> np.ndarray
```

Vibratory velocity level (ISO/TS 7849-1, Eq. 3).

$L_v = 20 \log_{10}(v/v_0)$ with `v0` the reference velocity.

**Parameters**

| Name | Description |
| :--- | :--- |
| `velocity` | R.m.s. vibratory velocity `v` (scalar or array), in m/s. |
| `reference` | Reference velocity `v0` (Default: 5e-8 m/s). |

**Returns:** The velocity level `L_v`, in dB re `v0`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive reference. |

## velocity_level_from_acceleration

```python
velocity_level_from_acceleration(
    peak_acceleration: ArrayLike,
    frequency: ArrayLike,
    *,
    reference: float = 5e-08,
) -> np.ndarray
```

Velocity level from a sinusoidal acceleration (ISO/TS 7849-1, Eq. 8).

$L_v = 20 \log_{10}\!\left( \frac{a_{\text{peak}}}{2\pi f v_0 \sqrt{2}} \right)$, used to convert a calibration acceleration to the equivalent
r.m.s. velocity level.

**Parameters**

| Name | Description |
| :--- | :--- |
| `peak_acceleration` | Peak acceleration `a_peak` (scalar or array), in m/s^2. |
| `frequency` | Frequency `f`, in hertz. |
| `reference` | Reference velocity `v0` (Default: 5e-8 m/s). |

**Returns:** The velocity level `L_v`, in dB re `v0`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive frequency or reference. |

## VibrationSoundPowerResult

```python
VibrationSoundPowerResult(
    velocity_level: np.ndarray,
    sound_power_level: np.ndarray,
    radiation_factor: np.ndarray,
    area: float,
    frequencies: np.ndarray | None = None,
)
```

Sound power radiated by surface vibration (ISO/TS 7849).

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Band centre frequencies, in hertz, or `None` for a single broadband value. |
| `velocity_level` | Mean vibratory velocity level `L_v` per band, in dB. |
| `sound_power_level` | Radiated sound power level `L_W` per band, in dB. |
| `radiation_factor` | Radiation factor `epsilon` per band. |
| `area` | Radiating surface area `S`, in m^2. |

### VibrationSoundPowerResult.plot()

```python
VibrationSoundPowerResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the radiated sound power level per band.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### VibrationSoundPowerResult.report()

```python
VibrationSoundPowerResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render an ISO/TS 7849 sound-power-from-vibration determination fiche.

Writes a one-page sound-power test sheet: the standard-basis line naming
the vibration method (the ISO/TS 7849-1 survey method with a fixed
radiation factor when every band uses `epsilon = 1`, otherwise the
ISO/TS 7849-2 engineering method with a determined radiation factor), an
optional metadata header (client, machine/source, test environment,
instrumentation, climate, date), a per-band table (nominal
octave/one-third-octave frequency, the surface vibratory velocity level
`Lv` and the radiated band sound-power level `LW`), the sound-power
spectrum `LW(f)` with a nominal band axis, the boxed A-weighted sound
power level `LWA` (dB re 1 pW) with the total `LW`, the radiating
area `S` and the applied method, an optional verdict row against a
declared limit, and a measurement-basis strip stating the sound-power
relation $L_W = L_v + 10 \log_{10}(S/S_0) + 10 \log_{10} \epsilon + 10 \log_{10}(Z_{c,n}/Z_{c,0})$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata) supplying the header (`client`, `specimen` the machine/source, `test_room` the test environment, `instrumentation`, `temperature`, `relative_humidity`, `pressure`, `test_date`), the footer identity (`laboratory`, `operator`, `report_id`, `notes`) and, via `requirement`, a declared A-weighted sound-power limit the fiche checks the result against (lower is better). The radiating area `S` comes from the result itself. |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When `True` the per-band table adds the radiation factor `epsilon` column. |
| `language` | Fiche language: `"en"` (default) or `"es"`. |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"` or `language` is unknown. |
| ImportError | If reportlab (or, for the figure, matplotlib) is not installed (`pip install phonometry[report]`). |

### VibrationSoundPowerResult.sound_power_level_a

*property*

A-weighted sound power level `L_WA`, in dB re 1 pW.

Combines the band levels with the A-weighting band corrections of
ISO 3744:2010 Annex E (the standard tabulation reused by the vibration
method) when band centre frequencies are known. Without band
frequencies the result is an unweighted broadband level that cannot be
A-weighted, so `L_WA` is undefined and `nan` is returned (the
report then boxes the unweighted total `L_W` instead of an `L_WA`
claim, and no A-weighted verdict is drawn).

### VibrationSoundPowerResult.total_level

*property*

Band-summed sound power level, in dB:
$10 \log_{10} \sum_j 10^{0.1 L_{Wj}}$.
