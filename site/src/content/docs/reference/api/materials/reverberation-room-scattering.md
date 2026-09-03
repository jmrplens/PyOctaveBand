---
title: "materials.diffusers.reverberation_room_scattering"
description: "Random-incidence scattering coefficient in a reverberation room."
sidebar:
  label: "reverberation_room_scattering"
---

Random-incidence scattering coefficient in a reverberation room.

**ISO 17497-1:2004+A1:2014.** Four reverberation times (Table 2) taken with and
without the test sample, with a static and a rotating turntable, give two
Sabine-form absorption coefficients: the random-incidence absorption
coefficient `alpha_s` (Clause 8.1.1, Eq. (1)) and the specular absorption
coefficient `alpha_spec` (Clause 8.1.2, Eq. (4)). Their ratio yields the
scattering coefficient
$s = (\alpha_{\mathrm{spec}} - \alpha_\mathrm{s}) / (1 - \alpha_\mathrm{s})$
(Clause 8.1.3, Eq. (5)). The turntable base plate is qualified through its
own scattering coefficient (Clause 8.1.4, Eq. (6)) against the Table 1
limits (Clause 6.2). Air properties come from the speed-of-sound and
energy-attenuation relations of Clause 8 (Eqs. (2)/(3), after ISO 9613-1),
and measurement accuracy from Annex A (Eqs. (A.1)-(A.5)).

One subject: everything the reverberation-room method needs, from the four
measured decay times to the scattering coefficient and its uncertainty. Part 2
of ISO 17497 is a different measurement in a free field and lives in
[`phonometry.materials.diffusers.scattering_diffusion`](/phonometry/reference/api/materials/scattering-diffusion/); the two parts share
no formula, and the helpers are named per part so they are never mixed. Neither
part contains a numeric worked example.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## absorption_coefficient_uncertainty

```python
absorption_coefficient_uncertainty(
    volume: float,
    area: float,
    *,
    c: ArrayLike,
    t_a: ArrayLike,
    u_a: ArrayLike,
    t_b: ArrayLike,
    u_b: ArrayLike,
) -> Real
```

Uncertainty of a Sabine absorption coefficient (ISO 17497-1, A.3/A.4).

$$
u_\alpha = \frac{55.3 V}{c S} \sqrt{(u_b / t_b^2)^2 + (u_a / t_a^2)^2}
$$

With situations `(t1, t2)` this is `u(alpha_s)` (Eq. (A.3)); with
`(t3, t4)` it is `u(alpha_spec)` (Eq. (A.4)). The unsubscripted `c` of
the standard is taken as a single (mean) speed of sound.

**Parameters**

| Name | Description |
| :--- | :--- |
| `volume` | Reverberation-room volume `V`, in cubic metres. |
| `area` | Test-sample area `S`, in square metres. |
| `c` | Speed of sound `c`, in m/s. |
| `t_a` | Reverberation time of the first situation, in seconds. |
| `u_a` | Standard uncertainty of `t_a` (Eq. (A.1)), in seconds. |
| `t_b` | Reverberation time of the second situation, in seconds. |
| `u_b` | Standard uncertainty of `t_b` (Eq. (A.1)), in seconds. |

**Returns:** Combined standard uncertainty of the absorption coefficient (per band).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for non-positive `V`, `S`, `c` or `T`. |

## air_attenuation_coefficient

```python
air_attenuation_coefficient(
    pressure_attenuation_db_per_m: ArrayLike,
) -> Real
```

Energy attenuation coefficient `m` (ISO 17497-1, Clause 8, Eq. (3)).

$m = \alpha / (10 \log_{10}(e)) \approx \alpha / 4.343$ (1/m), where
`alpha` is
the sound-*pressure* attenuation coefficient in dB/m obtained from
ISO 9613-1 using the measured temperature and relative humidity.

**Parameters**

| Name | Description |
| :--- | :--- |
| `pressure_attenuation_db_per_m` | Pressure attenuation coefficient `alpha` from ISO 9613-1, in decibels per metre (scalar or per band). |

**Returns:** Energy (power) attenuation coefficient `m`, in reciprocal metres.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if any value is negative or non-finite. |

## BASE_PLATE_BANDS

*Constant* (`tuple`).

```python
BASE_PLATE_BANDS = (100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000)
```

## BASE_PLATE_MAX_SCATTERING

*Constant* (`dict`).

```python
BASE_PLATE_MAX_SCATTERING = {100: 0.05, 125: 0.05, 160: 0.05, 200: 0.05, 250: 0.05, 315: 0.05, 400: 0.05, 500: 0.05, 630: 0.1, 800: 0.1, 1000: 0.1, 1250: 0.15, 1600: 0.15, 2000: 0.15, 2500: 0.2, 3150: 0.2, 4000: 0.2, 5000: 0.25}
```

## base_plate_scattering

```python
base_plate_scattering(
    volume: float,
    area: float,
    *,
    c1: ArrayLike,
    t1: ArrayLike,
    c3: ArrayLike,
    t3: ArrayLike,
    m1: ArrayLike = 0.0,
    m3: ArrayLike = 0.0,
) -> Real
```

Scattering coefficient of the base plate alone (ISO 17497-1, Eq. (6)).

$$
s_{\mathrm{base}} = 55.3 \frac{V}{S} \left( \frac{1}{c_3 T_3} - \frac{1}{c_1 T_1} \right) - \frac{4 V}{S} (m_3 - m_1)
$$

Ideally $T_1 = T_3$; a slightly non-symmetrical base plate shortens
`t3`
and this quality metric captures the resulting spurious scattering, which
must not exceed the Table 1 limits (Clause 6.2). See
[`check_base_plate_scattering`](/phonometry/reference/api/materials/reverberation-room-scattering/#check_base_plate_scattering).

**Parameters**

| Name | Description |
| :--- | :--- |
| `volume` | Reverberation-room volume `V`, in cubic metres. |
| `area` | Test-sample area `S`, in square metres. |
| `c1` | Speed of sound during `t1`, in m/s. |
| `t1` | Reverberation time with the static base plate, in seconds. |
| `c3` | Speed of sound during `t3`, in m/s. |
| `t3` | Reverberation time with the rotating base plate, in seconds. |
| `m1` | Energy attenuation coefficient during `t1`, in 1/m; defaults to 0. |
| `m3` | Energy attenuation coefficient during `t3`, in 1/m; defaults to 0. |

**Returns:** Base-plate scattering coefficient `s_base` (per band).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for non-positive `V`, `S`, `c` or `T`. |

## check_base_plate_scattering

```python
check_base_plate_scattering(
    scattering: Mapping[Any, float] | Sequence[float] | ArrayLike,
) -> tuple[int, ...]
```

Verify base-plate scattering against Table 1 (ISO 17497-1, Clause 6.2).

Every band whose measured base-plate scattering coefficient exceeds the
[`BASE_PLATE_MAX_SCATTERING`](/phonometry/reference/api/materials/reverberation-room-scattering/#base_plate_max_scattering) limit is collected and a single
[`ScatteringDiffusionWarning`](/phonometry/reference/api/materials/reverberation-room-scattering/#scatteringdiffusionwarning) is issued when any band is over the
limit.

**Parameters**

| Name | Description |
| :--- | :--- |
| `scattering` | Measured base-plate scattering coefficients, either a mapping keyed by one-third-octave centre frequency (Hz) or a sequence of 18 values ordered as [`BASE_PLATE_BANDS`](/phonometry/reference/api/materials/reverberation-room-scattering/#base_plate_bands). |

**Returns:** Tuple of the centre frequencies (Hz) that exceed the limit, in ascending order (empty if the base plate is compliant).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a mapping missing a band or a sequence of the wrong length. |

## random_incidence_absorption

```python
random_incidence_absorption(
    volume: float,
    area: float,
    *,
    c1: ArrayLike,
    t1: ArrayLike,
    c2: ArrayLike,
    t2: ArrayLike,
    m1: ArrayLike = 0.0,
    m2: ArrayLike = 0.0,
) -> Real
```

Random-incidence absorption coefficient `alpha_s` (ISO 17497-1, Eq. (1)).

$$
\alpha_\mathrm{s} = 55.3 \, \frac{V}{S} \left( \frac{1}{c_2 T_2} - \frac{1}{c_1 T_1} \right) - \frac{4 V}{S} (m_2 - m_1)
$$

Situation 1 is the empty room with the (static) base plate present;
situation 2 adds the test sample, still without turntable rotation
(Table 2, rows t1 and t2).

**Parameters**

| Name | Description |
| :--- | :--- |
| `volume` | Reverberation-room volume `V`, in cubic metres. |
| `area` | Test-sample area `S`, in square metres. |
| `c1` | Speed of sound during `t1`, in m/s (see [`speed_of_sound_iso17497`](/phonometry/reference/api/materials/reverberation-room-scattering/#speed_of_sound_iso17497)). |
| `t1` | Reverberation time without sample (base plate only), in seconds. |
| `c2` | Speed of sound during `t2`, in m/s. |
| `t2` | Reverberation time with the test sample, in seconds. |
| `m1` | Energy attenuation coefficient during `t1`, in 1/m (see [`air_attenuation_coefficient`](/phonometry/reference/api/materials/reverberation-room-scattering/#air_attenuation_coefficient)); defaults to 0. |
| `m2` | Energy attenuation coefficient during `t2`, in 1/m; defaults to 0. |

**Returns:** Random-incidence absorption coefficient `alpha_s` (per band).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for non-positive `V`, `S`, `c` or `T`. |

## reverberation_time_uncertainty

```python
reverberation_time_uncertainty(times: ArrayLike) -> Real
```

Standard uncertainty of a reverberation time (ISO 17497-1, Eq. (A.1)).

$$
u = \sqrt{ \sum_i (T_i - \overline{T})^2 / (N (N - 1)) }
$$

with $\overline{T}$ the mean of
the `N` spatially-averaged measurements (Eq. (A.2)); this is the standard
error of the mean.

**Parameters**

| Name | Description |
| :--- | :--- |
| `times` | The $N \ge 2$ reverberation-time measurements, in seconds. |

**Returns:** Standard uncertainty `u` of the mean reverberation time (0-d).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if fewer than two measurements are supplied. |

## scattering_coefficient

```python
scattering_coefficient(
    alpha_spec: ArrayLike,
    alpha_s: ArrayLike,
    *,
    truncate_negative: bool = True,
) -> Real
```

Random-incidence scattering coefficient `s` (ISO 17497-1, Eq. (5)).

$$
s = 1 - \frac{1 - \alpha_{\text{spec}}}{1 - \alpha_\mathrm{s}} = \frac{\alpha_{\text{spec}} - \alpha_\mathrm{s}}{1 - \alpha_\mathrm{s}}
$$

Following the presentation rule of Clause 8.3, negative results are
truncated to 0 while values greater than 1 (which can occur through edge
effects, Clause 6.3.2) are **kept** and reported. Rounding to 0,01 for a
results table is left to the caller.

**Parameters**

| Name | Description |
| :--- | :--- |
| `alpha_spec` | Specular absorption coefficient `alpha_spec` (Eq. (4)). |
| `alpha_s` | Random-incidence absorption coefficient `alpha_s` (Eq. (1)). |
| `truncate_negative` | If `True` (default), clip negative `s` to 0 per Clause 8.3; values above 1 are never clipped. |

**Returns:** Scattering coefficient `s` (per band).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if any `alpha_s` equals 1 (undefined ratio). |

## scattering_coefficient_spectrum

```python
scattering_coefficient_spectrum(
    frequencies: ArrayLike,
    specular_absorption: ArrayLike,
    random_absorption: ArrayLike,
    *,
    truncate_negative: bool = True,
) -> ScatteringResult
```

Scattering-coefficient spectrum `s(f)` (ISO 17497-1, Eq. (5)).

Convenience wrapper over [`scattering_coefficient`](/phonometry/reference/api/materials/reverberation-room-scattering/#scattering_coefficient) that pairs the
per-band specular `alpha_spec` (Eq. (4)) and random-incidence `alpha_s`
(Eq. (1)) absorptions with their band centres and returns a plottable
[`ScatteringResult`](/phonometry/reference/api/materials/reverberation-room-scattering/#scatteringresult).

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | One-third-octave band centres, in hertz (1-D). |
| `specular_absorption` | Specular absorption `alpha_spec` per band. |
| `random_absorption` | Random-incidence absorption `alpha_s` per band. |
| `truncate_negative` | Clip negative `s` to 0 (Clause 8.3 default). |

**Returns:** A [`ScatteringResult`](/phonometry/reference/api/materials/reverberation-room-scattering/#scatteringresult) with `.plot()`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if the three inputs differ in shape, the band centres are empty or not 1-D, or any `alpha_s` equals 1. |

## scattering_coefficient_uncertainty

```python
scattering_coefficient_uncertainty(
    alpha_spec: ArrayLike,
    alpha_s: ArrayLike,
    u_alpha_spec: ArrayLike,
    u_alpha_s: ArrayLike,
) -> ScatteringUncertainty
```

Uncertainty of the scattering coefficient (ISO 17497-1, Eq. (A.5)).

$$
u_s = \left\lvert \frac{\alpha_{\mathrm{spec}} - 1}{1 - \alpha_\mathrm{s}} \right\rvert \sqrt{\left( \frac{u_{\alpha_{\mathrm{spec}}}} {\alpha_{\mathrm{spec}} - 1} \right)^{2} + \left( \frac{u_{\alpha_\mathrm{s}}}{1 - \alpha_\mathrm{s}} \right)^{2}}
$$

with the expanded uncertainty $U = 2 u_s$ (95 % confidence).

**Parameters**

| Name | Description |
| :--- | :--- |
| `alpha_spec` | Specular absorption coefficient `alpha_spec` (Eq. (4)). |
| `alpha_s` | Random-incidence absorption coefficient `alpha_s` (Eq. (1)). |
| `u_alpha_spec` | Standard uncertainty of `alpha_spec` (Eq. (A.4)). |
| `u_alpha_s` | Standard uncertainty of `alpha_s` (Eq. (A.3)). |

**Returns:** A [`ScatteringUncertainty`](/phonometry/reference/api/materials/reverberation-room-scattering/#scatteringuncertainty) with `u_s` and $U = 2 u_s$.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if any `alpha_s` equals 1 or any `alpha_spec` equals 1. |

## ScatteringDiffusionWarning

Advisory for out-of-range scattering/diffusion measurement conditions.

## ScatteringResult

```python
ScatteringResult(
    frequencies: Real,
    scattering: Real,
    random_incidence: Real,
    specular: Real,
)
```

A random-incidence scattering-coefficient spectrum (ISO 17497-1).

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | One-third-octave band centre frequencies, in hertz. |
| `scattering` | Scattering coefficient `s` per band (Eq. (5)). |
| `random_incidence` | Random-incidence absorption `alpha_s` (Eq. (1)). |
| `specular` | Specular absorption `alpha_spec` (Eq. (4)). |

### ScatteringResult.plot()

```python
ScatteringResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the scattering coefficient `s` versus frequency.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes` and never calls `plt.show`.

### ScatteringResult.report()

```python
ScatteringResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render an ISO 17497-1 scattering-coefficient test-report fiche to a PDF.

Writes a one-page accredited random-incidence scattering report
(ISO 17497-1:2004+A1:2014): the standard-basis line, an optional
metadata header block (client, specimen, test room, sample area `S`,
temperature, humidity ...), a two-panel body with the per-band table
(frequency, the random-incidence absorption `alpha_s` and the
scattering coefficient `s`) beside the `s(f)` curve on a categorical
band axis, and a footer with the fixed disclaimer. ISO 17497-1 is a
characterisation, so there is no pass/fail verdict and no single-number
rating.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata); `None` produces a body-and-disclaimer fiche whose header shows only the measured frequency range. The applicable descriptive fields are `client`, `manufacturer`, `specimen`, `area`, `room_volume`, `mounting`, `test_room`, `test_date`, `temperature`, `relative_humidity`, `pressure`, `measurement_standard`, `laboratory`, `operator`, `report_id` and `notes`. The `requirement` field is ignored (ISO 17497-1 has no verdict). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When `True`, the value table inserts the specular absorption `alpha_spec` column beside `alpha_s` and `s`. |
| `language` | Fiche language: `"en"` (default, English, decimal point) or `"es"` (Spanish, decimal comma). |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"`. |
| ImportError | If reportlab is not installed (`pip install phonometry[report]`), or matplotlib is missing for the embedded figure (`pip install phonometry[plot]`). |

## ScatteringUncertainty

```python
ScatteringUncertainty(u_scattering: Real, expanded: Real)
```

Uncertainty of the scattering coefficient (ISO 17497-1, Annex A).

**Attributes**

| Name | Description |
| :--- | :--- |
| `u_scattering` | Combined standard uncertainty `u_s` of the scattering coefficient (Eq. (A.5)). |
| `expanded` | Expanded uncertainty $U = 2 u_s$ at 95 % confidence (Annex A). |

## specular_absorption_coefficient

```python
specular_absorption_coefficient(
    volume: float,
    area: float,
    *,
    c3: ArrayLike,
    t3: ArrayLike,
    c4: ArrayLike,
    t4: ArrayLike,
    m3: ArrayLike = 0.0,
    m4: ArrayLike = 0.0,
) -> Real
```

Specular absorption coefficient `alpha_spec` (ISO 17497-1, Eq. (4)).

$$
\alpha_{\text{spec}} = 55.3 \, \frac{V}{S} \left( \frac{1}{c_4 T_4} - \frac{1}{c_3 T_3} \right) - \frac{4 V}{S} (m_4 - m_3)
$$

Situation 3 is the rotating base plate without the sample; situation 4 is
the sample on the rotating turntable (Table 2, rows t3 and t4). The
apparent (specular) absorption includes the energy lost to scattering.

**Parameters**

| Name | Description |
| :--- | :--- |
| `volume` | Reverberation-room volume `V`, in cubic metres. |
| `area` | Test-sample area `S`, in square metres. |
| `c3` | Speed of sound during `t3`, in m/s. |
| `t3` | Reverberation time, rotating base plate without sample, in seconds. |
| `c4` | Speed of sound during `t4`, in m/s. |
| `t4` | Reverberation time, sample on the rotating turntable, in seconds. |
| `m3` | Energy attenuation coefficient during `t3`, in 1/m; defaults to 0. |
| `m4` | Energy attenuation coefficient during `t4`, in 1/m; defaults to 0. |

**Returns:** Specular absorption coefficient `alpha_spec` (per band).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for non-positive `V`, `S`, `c` or `T`. |

## speed_of_sound_iso17497

```python
speed_of_sound_iso17497(*, temperature_c: ArrayLike) -> Real
```

Speed of sound in air (ISO 17497-1:2004, Clause 8, Eq. (2)).

$c = 343.2 \sqrt{(273.15 + t) / 293.15}$ (m/s).

**Parameters**

| Name | Description |
| :--- | :--- |
| `temperature_c` | Air temperature, in **degrees Celsius** (scalar or per band). |

**Returns:** Speed of sound `c`, in metres per second.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if any temperature is at or below -273.15 degC. |
