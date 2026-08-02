---
title: "materials.surfaces.road_absorption"
description: "In-situ sound absorption of road surfaces (ISO 13472-1 / ISO 13472-2)."
sidebar:
  label: "road_absorption"
---

In-situ sound absorption of road surfaces (ISO 13472-1 / ISO 13472-2).

Two complementary standardised in-situ methods are supported here. They target
opposite ends of the absorption scale and are **not** interchangeable:

* **BS ISO 13472-1:2002** - *extended surface method*. A free-field impulse
  response is measured over the road, the direct (incident) and surface
  (reflected) components are separated in the time domain by the subtraction
  technique and an **Adrienne-type temporal window** (Clause 6.4), transformed
  to frequency, and ratioed. The normal-incidence sound absorption coefficient
  is (Clause 4.1):

  $$
  \alpha(f) = 1 - Q_W(f) = 1 - \frac{1}{K_r^{2}} \left\lvert \frac{H_r(f)}{H_i(f)} \right\rvert^{2}
  $$

  with `Hi`/`Hr` the incident/reflected transfer functions and `Kr` the
  geometrical-spreading factor $K_r = (d_s - d_m) / (d_s + d_m)$ for the
  mandatory geometry $d_s = 1.25$ m (source-to-plane) and
  $d_m = 0.25$ m (mic-to-plane), giving $K_r = 2/3$ (Clause 4.2 /
  Annex C). The complex pressure reflection factor
  $Q_p = (1/K_r)(H_r/H_i)\, e^{+j 2 \pi f \Delta\tau}$ with
  $\Delta\tau = 2 d_m / c$ (Annex C) is available for theory comparison.
  A highly reflective reference surface removes the electro-acoustic chain
  error and the geometry factor by a ratio (Annex B), and non-normal incidence
  uses $K_{r,\theta}$ (Annex F).

* **BS ISO 13472-2:2010** - *spot method*. This is an **in-situ application of
  the ISO 10534-2 two-microphone impedance tube** for reflective surfaces
  (measured `alpha` below ~0.15). Part 2 does not restate the
  transfer-function / reflection-factor / absorption mathematics; Clauses 4,
  5.7 and 6.6 defer it to ISO 10534-2. The core computation therefore lives in
  [`phonometry.impedance_tube.two_microphone_impedance`](/phonometry/reference/api/materials/impedance-tube/#two_microphone_impedance) and is **not**
  reimplemented here. This module contributes only the Part-2 tube
  geometry/validity helpers (upper usable frequency, microphone-spacing bounds,
  the 250-1600 Hz one-third-octave working range) and the Annex A internal-loss
  system correction
  $\alpha \approx \alpha_{\text{measured}} - \alpha_{\text{system}}$.

Sign / normalisation convention (ISO 13472-1): the forward transform is
NumPy's `rfft` with kernel $e^{-j 2 \pi f t}$ (unnormalised); every
quantity here is a ratio of two transforms from the same processing chain, so
the transform normalisation cancels (Clause 6.1). A pure time delay `tau` of
the reflected path scales its spectrum by $e^{-j 2 \pi f \tau}$; the
optional phase restoration multiplies by $e^{+j 2 \pi f \tau}$ to
recover `Qp` (Annex C /
Annex G, resolving the Clause 4.1 NOTE shorthand to the frequency-dependent
form).

The window durations of the Adrienne window are **not** fixed by the standard:
Clause 6.4 mandates only a sharp leading edge, a 5 ms flat portion and a
cosine-squared or Blackman-Harris trailing edge, with the shape and lengths
**reported per measurement**. They are therefore configurable here and default
to a short leading edge, a 5 ms flat top and a Blackman-Harris trailing edge
(the Annex E example report); the historical fixed edge timings are **not**
hard-coded as if normative.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## absorption_reference_corrected

```python
absorption_reference_corrected(
    road_reflection: ArrayLike,
    reference_reflection: ArrayLike,
) -> Real
```

Reference-corrected road absorption (ISO 13472-1:2002, Annex B).

Dividing the road and reference measured pressure reflection factors removes
both the electro-acoustic chain error `e(f)` and, because the geometry is
identical, the `Kr` factor:

$$
Q_{p,\text{road}}(f) = \frac{Q_{p,\text{road,meas}}(f)}{Q_{p,\text{ref,meas}}(f)}
$$

$$
\alpha_{\text{road}}(f) = 1 - \left\lvert \frac{Q_{p,\text{road,meas}}(f)}{Q_{p,\text{ref,meas}}(f)} \right\rvert^{2}
$$

The reference surface is assumed totally reflecting
($\lvert Q_{p,\text{ref}} \rvert = 1$, checked in an impedance tube
to have absorption \< 0.05, Annex B).

**Parameters**

| Name | Description |
| :--- | :--- |
| `road_reflection` | Measured road pressure reflection factor `Qp,road,meas` (complex; the `1 / Kr` scaling need not be removed as it cancels). |
| `reference_reflection` | Measured reference pressure reflection factor `Qp,ref,meas` (complex, same geometry and chain). |

**Returns:** Reference-corrected road absorption coefficient `alpha_road(f)`.

## adrienne_window

```python
adrienne_window(
    fs: float | None = None,
    *,
    flat_duration: float = 0.005,
    leading_duration: float = 0.0005,
    trailing_duration: float = 0.005,
    leading_edge: str = 'blackman-harris',
    trailing_edge: str = 'blackman-harris',
    sample_rate: float | str = 'deprecated',
) -> Real
```

Adrienne-type temporal window (ISO 13472-1:2002, Clause 6.4).

Clause 6.4 mandates a **sharp leading edge**, a **5 ms flat portion** and a
**cosine-squared or Blackman-Harris trailing edge** so as to suppress
frequency-domain oscillations (Figure 4); the shape and the durations are
*reported per measurement* rather than fixed by the standard. All three
durations are therefore configurable. The window is the concatenation of a
rising leading half-taper, a unit-valued flat portion and a falling trailing
half-taper:

```text
w = [ rising_edge (0 -> 1) | flat (== 1) | trailing_edge (1 -> 0) ]
```

The defaults (short 0.5 ms leading edge, 5 ms flat top, 5 ms Blackman-Harris
trailing edge) follow the Annex E example report; they are **not** a
normative fixed set of timings. The lower usable frequency scales as
`~ 1 / T_window` (Clause 6.4), so report the durations used.

**Parameters**

| Name | Description |
| :--- | :--- |
| `fs` | Sampling frequency, in hertz (ISO 13472-1 Clause 6.2 requires `fs > 40 kHz` in practice). |
| `flat_duration` | Flat-portion duration, in seconds (default 5 ms). |
| `leading_duration` | Leading-edge (rise) duration, in seconds; 0 gives a sharp step onset. |
| `trailing_duration` | Trailing-edge (fall) duration, in seconds. |
| `leading_edge` | Leading-edge shape, `"blackman-harris"` or `"cosine-squared"`. |
| `trailing_edge` | Trailing-edge shape, `"blackman-harris"` or `"cosine-squared"`. |
| `sample_rate` | Deprecated alias of `fs` (remove in 4.0). |

**Returns:** The time-domain window, one sample per `1 / fs` (length `round((leading + flat + trailing) * fs)` samples).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `fs` is missing or not positive, a duration is negative, the flat duration is not positive, or an edge shape is unknown. |

## check_spot_frequency_range

```python
check_spot_frequency_range(frequency: ArrayLike) -> None
```

Advise when a spot-method frequency is out of range (ISO 13472-2, Scope).

The spot method is valid over the one-third-octave bands 250-1600 Hz
(narrow-band 220-1800 Hz). Frequencies outside [`SPOT_FREQUENCY_RANGE`](/phonometry/reference/api/materials/road-absorption/#spot_frequency_range)
raise a [`RoadAbsorptionWarning`](/phonometry/reference/api/materials/road-absorption/#roadabsorptionwarning); results there are advisory.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequencies to check, in hertz. |

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If any frequency is negative. |

## DEFAULT_MIC_HEIGHT

*Constant* (`float`).

```python
DEFAULT_MIC_HEIGHT = 0.25
```

## DEFAULT_SOURCE_HEIGHT

*Constant* (`float`).

```python
DEFAULT_SOURCE_HEIGHT = 1.25
```

## DEFAULT_SPEED_OF_SOUND

*Constant* (`float`).

```python
DEFAULT_SPEED_OF_SOUND = 340.0
```

## geometric_spreading_factor

```python
geometric_spreading_factor(
    source_height: float = 1.25,
    mic_height: float = 0.25,
) -> float
```

Geometrical-spreading factor `Kr` (ISO 13472-1:2002, Clause 4.1).

$K_r = (d_s - d_m) / (d_s + d_m)$. It corrects the reflected path
for the extra spherical spreading over the image-source distance
$d_s + d_m$ relative to the direct distance $d_s - d_m$
(Annex C). The mandatory geometry $d_s = 1.25$ m,
$d_m = 0.25$ m gives $K_r = 2/3$ (Clause 4.2).

**Parameters**

| Name | Description |
| :--- | :--- |
| `source_height` | Source-to-reference-plane distance `ds`, in metres. |
| `mic_height` | Microphone-to-reference-plane distance `dm`, in metres. |

**Returns:** Geometrical-spreading factor `Kr` (dimensionless, $0 < K_r < 1$).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `ds` or `dm` is not positive or $d_s \le d_m$. |

## geometric_spreading_factor_angle

```python
geometric_spreading_factor_angle(
    incidence_angle: float,
    source_height: float = 1.25,
    mic_height: float = 0.25,
) -> float
```

Oblique geometrical-spreading factor (ISO 13472-1, Annex F).

$K_{r,\theta}^2 = 1 - \cos^2(\theta) (1 - K_r^2)$ with `Kr` the
normal-incidence factor of [`geometric_spreading_factor`](/phonometry/reference/api/materials/road-absorption/#geometric_spreading_factor). At
$\theta = 0$ the
cosine is unity and `Kr,theta` collapses to `Kr` (Clause 4.1).

**Parameters**

| Name | Description |
| :--- | :--- |
| `incidence_angle` | Incidence angle `theta`, in **radians**. |
| `source_height` | Source-to-reference-plane distance `ds`, in metres. |
| `mic_height` | Microphone-to-reference-plane distance `dm`, in metres. |

**Returns:** Oblique factor `Kr,theta` (positive root, dimensionless).

## insitu_absorption_coefficient

```python
insitu_absorption_coefficient(
    incident_ir: ArrayLike,
    reflected_ir: ArrayLike,
    *,
    source_height: float = 1.25,
    mic_height: float = 0.25,
    incidence_angle: float = 0.0,
    n: int | None = None,
) -> Real
```

Normal-incidence absorption coefficient `alpha(f)` (ISO 13472-1, 4.1).

$\alpha(f) = 1 - Q_W(f) = 1 - (1 / K_r^2) \lvert H_r(f) / H_i(f) \rvert^2$ (the direct
energy route via [`power_reflection_coefficient`](/phonometry/reference/api/materials/road-absorption/#power_reflection_coefficient); for oblique incidence
`Kr` is replaced by `Kr,theta`, Annex F). Non-specularly reflected energy
is treated as absorbed, so `alpha` may be slightly overestimated
(Clause 4.1).

**Parameters**

| Name | Description |
| :--- | :--- |
| `incident_ir` | Windowed incident impulse response `hi(t)`, real. |
| `reflected_ir` | Windowed reflected impulse response `hr(t)`, real. |
| `source_height` | Source-to-plane distance `ds`, in metres. |
| `mic_height` | Microphone-to-plane distance `dm`, in metres. |
| `incidence_angle` | Incidence angle `theta`, in radians (0 = normal). |
| `n` | FFT length; defaults to the longer input. |

**Returns:** Absorption coefficient `alpha(f)` at the `rfft` frequency bins.

## insitu_absorption_from_reflection

```python
insitu_absorption_from_reflection(reflection: ArrayLike) -> Real
```

Absorption coefficient from the reflection factor (ISO 13472-1, 4.1).

$\alpha = 1 - \lvert r \rvert^2$. With `r` already carrying the
$1 / K_r$ factor
(see [`insitu_reflection_factor`](/phonometry/reference/api/materials/road-absorption/#insitu_reflection_factor)) this is the **reflection-factor route** to
`alpha` and equals the direct energy route of
[`insitu_absorption_coefficient`](/phonometry/reference/api/materials/road-absorption/#insitu_absorption_coefficient).

**Parameters**

| Name | Description |
| :--- | :--- |
| `reflection` | Complex reflection factor `r`. |

**Returns:** Absorption coefficient `alpha` (real).

## insitu_absorption_spectrum

```python
insitu_absorption_spectrum(
    incident_ir: ArrayLike,
    reflected_ir: ArrayLike,
    fs: float | None = None,
    *,
    source_height: float = 1.25,
    mic_height: float = 0.25,
    incidence_angle: float = 0.0,
    n: int | None = None,
    f_min: float = 250.0,
    f_max: float = 4000.0,
    clip_negative: bool = True,
    sample_rate: float | str = 'deprecated',
) -> InsituAbsorptionResult
```

In-situ one-third-octave absorption spectrum (ISO 13472-1, Clause 4.1).

End-to-end convenience: the windowed incident and reflected impulse
responses give the narrow-band absorption via
[`insitu_absorption_coefficient`](/phonometry/reference/api/materials/road-absorption/#insitu_absorption_coefficient), which is then reduced to
one-third-octave bands with [`one_third_octave_absorption`](/phonometry/reference/api/materials/road-absorption/#one_third_octave_absorption) and wrapped
in a plottable [`InsituAbsorptionResult`](/phonometry/reference/api/materials/road-absorption/#insituabsorptionresult).

**Parameters**

| Name | Description |
| :--- | :--- |
| `incident_ir` | Windowed incident (direct-path) impulse response `hi`. |
| `reflected_ir` | Windowed reflected-path impulse response `hr`. |
| `fs` | Sampling frequency, in hertz. |
| `source_height` | Source-to-plane distance `ds`, in metres. |
| `mic_height` | Microphone-to-plane distance `dm`, in metres. |
| `incidence_angle` | Incidence angle `theta`, in radians (0 = normal). |
| `n` | FFT length; defaults to the longer of the two impulse responses. |
| `f_min` | Lowest band centre to report, in hertz (default 250 Hz). |
| `f_max` | Highest band centre to report, in hertz (default 4000 Hz). |
| `clip_negative` | Clip negative band results to zero (default `True`). |
| `sample_rate` | Deprecated alias of `fs` (remove in 4.0). |

**Returns:** An [`InsituAbsorptionResult`](/phonometry/reference/api/materials/road-absorption/#insituabsorptionresult) with `.plot()`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | On empty inputs, invalid geometry, or a missing or non-positive `fs`. |

## insitu_reflection_factor

```python
insitu_reflection_factor(
    incident_ir: ArrayLike,
    reflected_ir: ArrayLike,
    *,
    source_height: float = 1.25,
    mic_height: float = 0.25,
    incidence_angle: float = 0.0,
    fs: float | None = None,
    delay: float | None = None,
    n: int | None = None,
    sample_rate: float | str = 'deprecated',
) -> Complex
```

Complex pressure reflection factor `r(f)` (ISO 13472-1, Clause 4.1).

$r(f) = (1 / K_r) H_r(f) / H_i(f)$ from the windowed reflected and
incident
impulse responses, with `Hr`/`Hi` their real FFTs and `Kr` the
geometrical-spreading factor (or `Kr,theta` when `incidence_angle` is
given, Annex F). When both `fs` and `delay` are supplied the
reflected-path time offset is undone by
$\exp(+j 2 \pi f \, \text{delay})$, yielding
the complex `Qp` of the Clause 4.1 NOTE (with `delay` set to
$\Delta\tau = 2 d_m / c$,
Annex C; the frequency-dependent form of Annex G).

**Parameters**

| Name | Description |
| :--- | :--- |
| `incident_ir` | Windowed incident (direct-path) impulse response `hi(t)`, real, one sample per `1 / fs`. |
| `reflected_ir` | Windowed reflected-path impulse response `hr(t)`, real, same sampling as `incident_ir`. |
| `source_height` | Source-to-plane distance `ds`, in metres. |
| `mic_height` | Microphone-to-plane distance `dm`, in metres. |
| `incidence_angle` | Incidence angle `theta`, in radians (0 = normal). |
| `fs` | Sampling frequency, in hertz; required with `delay` for phase restoration. |
| `delay` | Reflected-path delay `dtau` to undo, in seconds; `None` returns the raw spectral ratio. |
| `n` | FFT length; defaults to the longer of the two impulse responses. |
| `sample_rate` | Deprecated alias of `fs` (remove in 4.0). |

**Returns:** Complex reflection factor `r(f)` at the `rfft` frequency bins.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | On empty inputs, invalid geometry, or `delay` given without `fs`. |

## InsituAbsorptionResult

```python
InsituAbsorptionResult(
    frequencies: Real,
    absorption: Real,
    source_height: float | None = None,
    mic_height: float | None = None,
)
```

An in-situ one-third-octave absorption spectrum (ISO 13472-1).

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | One-third-octave band centre frequencies, in hertz. |
| `absorption` | Sound-absorption coefficient `alpha` per band (a band with no contributing narrow-band samples is `nan`). |
| `source_height` | Source height the spectrum was reduced with, in metres, retained (with `mic_height`) so `plot_geometry` can draw the set-up; `None` for hand-built results. |
| `mic_height` | Microphone height, in metres, or `None`. |

### InsituAbsorptionResult.plot()

```python
InsituAbsorptionResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the in-situ absorption spectrum `alpha(f)`.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes` and never calls `plt.show`.

### InsituAbsorptionResult.plot_geometry()

```python
InsituAbsorptionResult.plot_geometry(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Draw the measurement set-up to scale.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the result does not retain its geometry. |

## max_sampled_area_radius

```python
max_sampled_area_radius(
    window_width: float,
    *,
    source_height: float = 1.25,
    mic_height: float = 0.25,
    speed_of_sound: float = 340.0,
) -> float
```

Radius of the maximum sampled area (ISO 13472-1:2002, Annex A).

For normal incidence the maximum sampled area is a circle of radius (m):

$$
r = \frac{1}{d_s + d_m + c T_w} \sqrt{(d_s + d_m + c T_w/2)(d_s + c T_w/2) (2 d_m + c T_w)(c T_w)}
$$

with `Tw` the width of the temporal window isolating the reflected wave.
The Annex A worked example ($d_s = 1.25$, $d_m = 0.25$,
$c = 340$ m/s, and the 5 ms flat window) gives
$r \approx 1.34$ m.

**Parameters**

| Name | Description |
| :--- | :--- |
| `window_width` | Temporal-window width `Tw` (reflected wave), seconds. |
| `source_height` | Source-to-plane distance `ds`, in metres. |
| `mic_height` | Microphone-to-plane distance `dm`, in metres. |
| `speed_of_sound` | Speed of sound `c`, in metres per second. |

**Returns:** Maximum-sampled-area radius `r`, in metres.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | On non-positive geometry or window width. |

## msa_major_axis

```python
msa_major_axis(
    window_width: float,
    projected_distance: float,
    *,
    source_height: float = 1.25,
    mic_height: float = 0.25,
    speed_of_sound: float = 340.0,
) -> float
```

Major axis of the oblique sampled-area ellipsoid (ISO 13472-1,
Annex F).

$a = c\,T_w + \sqrt{(d_s + d_m)^2 + d_p^2}$ for the ellipsoid of
revolution with the source and microphone at its foci; `dp` is the
source-to-microphone distance projected on the reference plane.

**Parameters**

| Name | Description |
| :--- | :--- |
| `window_width` | Temporal-window width `Tw`, in seconds. |
| `projected_distance` | Projected source-to-mic distance `dp`, in metres. |
| `source_height` | Source-to-plane distance `ds`, in metres. |
| `mic_height` | Microphone-to-plane distance `dm`, in metres. |
| `speed_of_sound` | Speed of sound `c`, in metres per second. |

**Returns:** Major axis `a`, in metres.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | On non-positive window width, speed, or negative `dp`. |

## one_third_octave_absorption

```python
one_third_octave_absorption(
    frequency: ArrayLike,
    absorption: ArrayLike,
    *,
    f_min: float = 250.0,
    f_max: float = 4000.0,
    clip_negative: bool = True,
) -> tuple[Real, Real]
```

Aggregate narrow-band absorption into one-third-octave bands.

Both parts require a **linear average** of the narrow-band absorption over
each one-third-octave band (ISO 13472-1 Clause 4.1; ISO 13472-2 Clause 6.6),
keeping negative narrow-band values during the averaging. The band edges are
the IEC base-two limits $f_c \cdot 2^{\pm 1/6}$. Negative one-third-octave
results are set to zero when `clip_negative` is true (ISO 13472-2
Clause 6.6 step 5); Part 1 does not mandate clipping, so pass
`clip_negative=False` to reproduce its raw output.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Narrow-band frequencies, in hertz. |
| `absorption` | Narrow-band absorption values aligned with `frequency`. |
| `f_min` | Lowest band centre to report, in hertz (default 250 Hz). |
| `f_max` | Highest band centre to report, in hertz (4000 Hz for Part 1; pass 1600 Hz for the Part-2 range). |
| `clip_negative` | Set negative band results to zero (default `True`). |

**Returns:** Tuple `(band_centres, band_absorption)`; a band with no narrow-band samples yields `nan`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `frequency` and `absorption` differ in length or are empty. |

## PART1_FREQUENCY_RANGE

*Constant* (`tuple`).

```python
PART1_FREQUENCY_RANGE = (250.0, 4000.0)
```

## plot_insitu_geometry

```python
plot_insitu_geometry(
    ax: Axes | None = None,
    *,
    source_height: float = 1.25,
    mic_height: float = 0.25,
    sampled_radius: float | None = 1.34,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Draw the in-situ surface-absorption set-up to scale.

Loudspeaker on its mast above the surface, microphone below it, direct
and surface-reflected paths, and the sampled-area radius on the ground.
Defaults are the standard heights (1,25 m source, 0,25 m microphone)
with the sampled radius of the standard 5 ms window.

**Parameters**

| Name | Description |
| :--- | :--- |
| `ax` | Existing axes, or `None` to create a figure. |
| `source_height` | Source height above the surface, in metres. |
| `mic_height` | Microphone height, in metres (\< `source_height`). |
| `sampled_radius` | Sampled-area radius drawn on the ground, in metres; `None` omits it. |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the surface rectangle. |

**Returns:** The axes.

## power_reflection_coefficient

```python
power_reflection_coefficient(
    incident_ir: ArrayLike,
    reflected_ir: ArrayLike,
    *,
    source_height: float = 1.25,
    mic_height: float = 0.25,
    incidence_angle: float = 0.0,
    n: int | None = None,
) -> Real
```

Sound-power reflection factor `QW(f)` (ISO 13472-1, 4.1 / Annex C).

The **direct energy route**
$Q_W(f) = (1 / K_r^2) \lvert H_r(f) / H_i(f) \rvert^2$
(`Kr,theta` for oblique incidence). It equals
$\lvert r \rvert^2$ from
[`insitu_reflection_factor`](/phonometry/reference/api/materials/road-absorption/#insitu_reflection_factor) but is formed from magnitudes only, so it is
independent of any reflected-path time offset.

**Parameters**

| Name | Description |
| :--- | :--- |
| `incident_ir` | Windowed incident impulse response `hi(t)`, real. |
| `reflected_ir` | Windowed reflected impulse response `hr(t)`, real. |
| `source_height` | Source-to-plane distance `ds`, in metres. |
| `mic_height` | Microphone-to-plane distance `dm`, in metres. |
| `incidence_angle` | Incidence angle `theta`, in radians. |
| `n` | FFT length; defaults to the longer input. |

**Returns:** Sound-power reflection factor `QW(f)` (real).

## reflected_path_delay

```python
reflected_path_delay(
    mic_height: float = 0.25,
    speed_of_sound: float = 340.0,
) -> float
```

Reflected-path arrival delay `dtau` (ISO 13472-1:2002, Annex C).

$\Delta\tau = 2 d_m / c$ is the time difference between the direct
and the surface-reflected impulses for the normal-incidence geometry; it
is the delay undone by the phase-restoration term of
[`insitu_reflection_factor`](/phonometry/reference/api/materials/road-absorption/#insitu_reflection_factor).

**Parameters**

| Name | Description |
| :--- | :--- |
| `mic_height` | Microphone-to-reference-plane distance `dm`, in metres. |
| `speed_of_sound` | Speed of sound `c`, in metres per second. |

**Returns:** Delay `dtau`, in seconds.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `dm` or `c` is not positive. |

## RoadAbsorptionWarning

Advisory for out-of-range in-situ road-absorption frequencies.

## SPOT_FREQUENCY_RANGE

*Constant* (`tuple`).

```python
SPOT_FREQUENCY_RANGE = (250.0, 1600.0)
```

## spot_internal_loss_correction

```python
spot_internal_loss_correction(
    measured_absorption: ArrayLike,
    system_absorption: ArrayLike,
    *,
    clip_negative: bool = True,
) -> Real
```

Internal-loss (system) correction (ISO 13472-2:2010, Annex A).

The tube reads all thermal/viscous losses between the microphones and the
surface as absorption. For reflective surfaces this is removed by
subtracting the reading on a totally reflecting reference plate:

$$
\alpha(f) \approx \alpha_{\text{measured}}(f) - \alpha_{\text{system}}(f)
$$

Valid because both terms are small (measured \< 0.15, internal \< 0.03). This
is the **subtractive** Part-2 correction and must not be confused with the
Part-1 ratio correction ([`absorption_reference_corrected`](/phonometry/reference/api/materials/road-absorption/#absorption_reference_corrected)). Negative
one-third-octave results are set to zero when `clip_negative` is true
(Clause 6.6 step 5).

**Parameters**

| Name | Description |
| :--- | :--- |
| `measured_absorption` | Measured road absorption `alpha_m(f)`. |
| `system_absorption` | Reference-plate (system) absorption `alpha_system(f)`, same bands. |
| `clip_negative` | Set negative corrected values to zero (default `True`). |

**Returns:** Corrected absorption coefficient `alpha(f)`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the two inputs differ in shape. |

## spot_microphone_spacing_bounds

```python
spot_microphone_spacing_bounds(
    speed_of_sound: float = 340.0,
    *,
    f_min: float = 220.0,
    f_max: float = 1800.0,
) -> tuple[float, float]
```

Microphone-spacing bounds `(s_min, s_max)` (ISO 13472-2:2010, 5.4.2).

$s_{\mathrm{max}} < 0.45 c_0 / f_{\mathrm{max}}$ avoids spacing
approaching half a wavelength at
the top frequency, and $s_{\mathrm{min}} > 0.05 c_0 / f_{\mathrm{min}}$
keeps the spacing above
5 % of a wavelength at the bottom frequency. For the narrow band
220-1800 Hz ($c_0 \approx 340$ m/s) these give
$s_{\mathrm{max}} \approx 85$ mm and
$s_{\mathrm{min}} \approx 77$ mm, bracketing the nominal
$s = (81 \pm 4)$ mm.

**Parameters**

| Name | Description |
| :--- | :--- |
| `speed_of_sound` | Speed of sound `c0`, in metres per second. |
| `f_min` | Lowest frequency of interest `f_min`, in hertz. |
| `f_max` | Highest frequency of interest `f_max`, in hertz. |

**Returns:** Tuple `(s_min, s_max)` of the lower and upper spacing limits, m.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | On non-positive speed or frequency, or `f_min >= f_max`. |

## SPOT_NARROW_BAND_RANGE

*Constant* (`tuple`).

```python
SPOT_NARROW_BAND_RANGE = (220.0, 1800.0)
```

## spot_tube_upper_frequency

```python
spot_tube_upper_frequency(
    diameter: float,
    speed_of_sound: float = 340.0,
) -> float
```

Upper usable frequency of the spot tube (ISO 13472-2:2010, 5.4.1).

$f_u = 0.58 c_0 / d$ (circular tube), the highest frequency at which
only plane waves propagate. A 100 mm tube at $c_0 = 340$ m/s gives
$f_u \approx 1972$ Hz, comfortably above the 1800 Hz narrow-band
top.

**Parameters**

| Name | Description |
| :--- | :--- |
| `diameter` | Tube diameter `d`, in metres. |
| `speed_of_sound` | Speed of sound `c0`, in metres per second. |

**Returns:** Upper usable frequency `f_u`, in hertz.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `d` or `c0` is not positive. |
