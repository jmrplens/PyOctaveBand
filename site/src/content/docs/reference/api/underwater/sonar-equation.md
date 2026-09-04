---
title: "underwater.sonar_equation"
description: "The sonar equation (passive and active), in decibels."
sidebar:
  label: "sonar_equation"
---

The sonar equation (passive and active), in decibels.

Combines the sonar performance terms -- source level `SL`, propagation loss
`PL`, noise level `NL`, directivity index `DI`, detection threshold `DT`,
target strength `TS` and reverberation level `RL` -- into the signal excess
`SE`, the signal-to-noise ratio and the figure of merit (the maximum allowable
propagation loss at the detection limit $\mathrm{SE} = 0$):

* [`passive_sonar_equation`](/phonometry/reference/api/underwater/sonar-equation/#passive_sonar_equation) --
  $\mathrm{SE} = \mathrm{SL} - \mathrm{PL} - (\mathrm{NL} - \mathrm{DI}) - \mathrm{DT}$.
* [`active_sonar_equation`](/phonometry/reference/api/underwater/sonar-equation/#active_sonar_equation) -- monostatic, noise-limited
  $\mathrm{SE} = \mathrm{SL} - 2\,\mathrm{PL} + \mathrm{TS} - (\mathrm{NL} - \mathrm{DI}) - \mathrm{DT}$ or, when a reverberation level
  is given, reverberation-limited
  $\mathrm{SE} = \mathrm{SL} - 2\,\mathrm{PL} + \mathrm{TS} - \mathrm{RL} - \mathrm{DT}$.

Two of those terms have their own model here rather than having to be supplied
from outside: [`array_directivity_index`](/phonometry/reference/api/underwater/sonar-equation/#array_directivity_index) gives `DI` from the length of a
line array and the wavelength, which is also its array gain when the noise is
isotropic, and [`detection_threshold`](/phonometry/reference/api/underwater/sonar-equation/#detection_threshold) gives `DT` from the false-alarm
probability alone. Both are Ainslie (2010).

All quantities are in dB (levels re a plane wave of 1 µPa rms; the terms are
spectrum levels, i.e. referred to a 1 Hz band). Source: Urick, *Principles of
Underwater Sound*, via Etter (2003), Table 10.2. The loss term is the
propagation loss $N_\mathrm{PL} = L_\mathrm{S} - L_p(x)$ of ISO 18405:2017,
3.4.1.4, which is also the term its own passive and active sonar equations
(3.6.2.7 and 3.6.2.11) are written with.

The figure of merit is the *maximum allowable propagation loss*, so inverting
a propagation-loss law at $\mathrm{PL} = \mathrm{FOM}$ gives the
**detection range**, the
range at which the detection probability is 50 %:

* [`detection_range`](/phonometry/reference/api/underwater/sonar-equation/#detection_range) inverts the closed-form loss of
  [`phonometry.underwater.propagation.closed_form`](/phonometry/reference/api/underwater/closed-form/) (spreading plus volume absorption),
  which is strictly increasing with range and therefore has a single crossing;
* [`detection_range_from_curve`](/phonometry/reference/api/underwater/sonar-equation/#detection_range_from_curve) reads the crossing off any computed loss
  curve -- a normal-mode, parabolic-equation or Weston-regime prediction --
  where the oscillatory loss of a real waveguide can cross the figure of merit
  more than once (Ainslie, *Principles of Sonar Performance Modelling*, §11.2.8
  makes exactly that point about convergence zones).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## active_sonar_equation

```python
active_sonar_equation(
    source_level: float,
    propagation_loss: NDArray[np.float64] | list[float] | float,
    target_strength: float,
    noise_level: float,
    *,
    directivity_index: float = 0.0,
    detection_threshold: float = 0.0,
    reverberation_level: float | None = None,
) -> SonarEquationResult
```

Monostatic active sonar equation with a two-way propagation loss.

Noise-limited: $\mathrm{SE} = \mathrm{SL} - 2\,\mathrm{PL} + \mathrm{TS} - (\mathrm{NL} - \mathrm{DI}) - \mathrm{DT}$. When
`reverberation_level` is given, reverberation-limited:
$\mathrm{SE} = \mathrm{SL} - 2\,\mathrm{PL} + \mathrm{TS} - \mathrm{RL} - \mathrm{DT}$ (`DI` does not apply to reverberation).

**Parameters**

| Name | Description |
| :--- | :--- |
| `source_level` | Source level `SL`, in dB. |
| `propagation_loss` | One-way propagation loss `PL`, in dB (scalar or array); the equation applies $2\,\mathrm{PL}$. |
| `target_strength` | Target strength `TS`, in dB. |
| `noise_level` | Background noise level `NL`, in dB. |
| `directivity_index` | Receiver directivity index `DI`, in dB. |
| `detection_threshold` | Detection threshold `DT`, in dB. |
| `reverberation_level` | Reverberation level `RL` in dB; when given, the case is reverberation-limited. |

**Returns:** A [`SonarEquationResult`](/phonometry/reference/api/underwater/sonar-equation/#sonarequationresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If an input is not finite. |

## array_directivity_index

```python
array_directivity_index(
    array_length_m: float,
    wavelength_m: float,
    *,
    steer_angle_rad: float = 0.0,
) -> float
```

Directivity index of an unshaded line array (Ainslie 2010, Eq. (6.49)).

$\mathrm{DI} = 10 \log_{10} G_\mathrm{D}$ with the directivity factor
$G_\mathrm{D} = 4\pi / \delta\Omega$ the reciprocal of the solid-angle
footprint of the beam. For a steered unshaded line array, Equation (6.56)
on printed folio 267 gives that footprint in closed form:

$$
\delta\Omega = \frac{4}{G_0} \left\{ \sigma\!\left[\frac{\pi G_0}{2}(1 - \sin\psi)\right] + \sigma\!\left[\frac{\pi G_0}{2}(1 + \sin\psi)\right] \right\}
$$

with $\sigma(x) = \int_0^x \mathrm{d}u \sin^2 u / u^2 = \mathrm{Si}(2x) - \sin^2 x / x$ (Eq. (6.54)) and
$G_0 = 2L/\lambda$ (Eq. (6.57)), the high-frequency limit of the
broadside directivity factor.

This is the **array gain** whenever the noise is isotropic and the signal a
plane wave, which is the case the sonar equation is written for
(Section 6.1.3.1): the two coincide in that limit, so this is what
[`passive_sonar_equation`](/phonometry/reference/api/underwater/sonar-equation/#passive_sonar_equation) wants for its `directivity_index`.

The book states three limits, and they are what the implementation is
checked against: $10 \log_{10}(2L/\lambda)$ at high frequency for
every steer direction but endfire, $10 \log_{10}(4L/\lambda)$ near
endfire, where the footprint halves, and 0 dB as $L/\lambda \to 0$,
where the array stops resolving anything. That last one is a limit and not
a cutoff: a finite array a wavelength long still returns 3.45 dB, and half
a wavelength 1.11 dB. It is reached exactly only where the ratio itself
underflows, and the value there is 0 dB rather than an error.

**Parameters**

| Name | Description |
| :--- | :--- |
| `array_length_m` | Array length `L`, in metres (> 0). |
| `wavelength_m` | Acoustic wavelength `lambda`, in metres (> 0). |
| `steer_angle_rad` | Steer angle `psi` from broadside, in radians (Default: 0, broadside). Only its sine enters, so the two sides of broadside give the same index. |

**Returns:** The directivity index `DI`, in dB (>= 0).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive or non-finite length or wavelength, or a non-finite steer angle. |

## detection_range

```python
detection_range(
    figure_of_merit: float,
    frequency_hz: float,
    *,
    law: str = 'spherical',
    transition_range: float | None = None,
    temperature: float = 10.0,
    salinity: float = 35.0,
    depth: float = 0.0,
    ph: float = 8.0,
    model: str = 'francois-garrison',
    max_range: float = 500000.0,
    n_points: int = 400,
) -> DetectionRangeResult
```

Range at which the closed-form propagation loss equals the figure of
merit.

Solves $\mathrm{PL}(r) = \mathrm{FOM}$ for the loss of
[`propagation_loss`](/phonometry/reference/api/underwater/closed-form/#propagation_loss), which is
strictly increasing in range, so the root is unique. A **one-way** figure of
merit works for both sonar modes: the active figure of merit returned by
[`active_sonar_equation`](/phonometry/reference/api/underwater/sonar-equation/#active_sonar_equation) is already the maximum allowable one-way loss.

**Parameters**

| Name | Description |
| :--- | :--- |
| `figure_of_merit` | Maximum allowable one-way propagation loss, in dB. |
| `frequency_hz` | Acoustic frequency, in Hz. |
| `law` | Spreading law (see [`spreading_loss`](/phonometry/reference/api/underwater/closed-form/#spreading_loss)). |
| `transition_range` | Transition range for the `"practical"` law, in m. |
| `temperature` | Temperature `T`, in degrees Celsius. |
| `salinity` | Salinity `S`, in parts per thousand. |
| `depth` | Depth, in metres. |
| `ph` | Acidity (default 8). |
| `model` | Absorption model (see [`seawater_absorption`](/phonometry/reference/api/underwater/closed-form/#seawater_absorption)). |
| `max_range` | Upper bound of the search, in metres. |
| `n_points` | Number of ranges kept on the returned loss curve. |

**Returns:** A [`DetectionRangeResult`](/phonometry/reference/api/underwater/sonar-equation/#detectionrangeresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If an input is invalid. |

## detection_range_from_curve

```python
detection_range_from_curve(
    figure_of_merit: float,
    range_m: NDArray[np.float64] | list[float],
    propagation_loss: NDArray[np.float64] | list[float],
    *,
    crossing: str = 'first',
) -> float
```

Detection range read off a computed propagation-loss curve.

Finds where `PL(r)` crosses the figure of merit from below, interpolating
linearly between the two bracketing samples. Real waveguides oscillate, so
`crossing` selects which crossing to report.

**Parameters**

| Name | Description |
| :--- | :--- |
| `figure_of_merit` | Maximum allowable propagation loss, in dB. |
| `range_m` | Ranges, in metres (1-D, strictly increasing). |
| `propagation_loss` | Loss at each range, in dB (same length). |
| `crossing` | `"first"` (default) or `"last"` upward crossing. |

**Returns:** The detection range, in metres. Two limiting cases carry no crossing and are distinguished by the loss at the **last** sample: `inf` when the loss is still below the figure of merit there (the target stays detectable past the end of the grid) and `0.0` when the loss exceeds it there, which without an upward crossing means it exceeded it at every sample and the target is detectable nowhere. [`detection_range`](/phonometry/reference/api/underwater/sonar-equation/#detection_range) returns the same two values for the same two situations.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## detection_threshold

```python
detection_threshold(false_alarm_probability: float) -> float
```

Detection threshold at 50 % detection probability (Ainslie Eq. (11.22)).

$$
\mathrm{DT}_{50}(p_\mathrm{fa}) \approx 10 \log_{10}\left(\log_2 \frac{1}{2 p_\mathrm{fa}}\right) - 0.8 \ \mathrm{dB}
$$

printed on folio 581. `DT` is $10 \log_{10} R_{50}$, the
signal-to-noise ratio after all processing that a 50 % detection
probability needs (Eq. (3.31)); this closed form estimates it from the
false-alarm probability alone.

The logarithm inside is **base two**, not a square. The book states the
approximation is accurate to +/- 0.1 dB for $p_\mathrm{fa} < 10^{-2}$
with one-dominant-plus-Rayleigh signal statistics, which is the
intermediate choice to make when the target statistics are unknown, and
that assuming those statistics anyway costs no more than 0.8 dB even for a
stable signal or a fully Rayleigh one.

**Parameters**

| Name | Description |
| :--- | :--- |
| `false_alarm_probability` | `p_fa`, the probability of declaring a detection with no target present, in (0, 1/2). |

**Returns:** The detection threshold `DT`, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-finite `p_fa`, or one outside (0, 1/2). At $p_\mathrm{fa} = 1/2$ the inner logarithm is zero and the threshold diverges: half the empty beams are already called detections. |

## DetectionRangeResult

```python
DetectionRangeResult(
    detection_range: float,
    figure_of_merit: float,
    frequency: float,
    range_m: NDArray[np.float64],
    propagation_loss: NDArray[np.float64],
    absorption_coefficient: float,
    law: str,
    model: str,
)
```

Detection range obtained by inverting a propagation-loss law.

**Attributes**

| Name | Description |
| :--- | :--- |
| `detection_range` | Range at which `PL` equals the figure of merit, in metres. `inf` when the loss never reaches it inside `max_range` (detectable throughout) and `0.0` when it already exceeds it at the search floor (detectable nowhere). |
| `figure_of_merit` | The figure of merit inverted, in dB. |
| `frequency` | Acoustic frequency, in Hz. |
| `range_m` | Range grid over which the loss was evaluated, in metres. |
| `propagation_loss` | Propagation loss at each range, in dB. |
| `absorption_coefficient` | Absorption coefficient $\alpha$, in dB/km. |
| `law` | The spreading law used. |
| `model` | The absorption model used. |

### DetectionRangeResult.plot()

```python
DetectionRangeResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the propagation loss against the figure of merit.

## passive_sonar_equation

```python
passive_sonar_equation(
    source_level: float,
    propagation_loss: NDArray[np.float64] | list[float] | float,
    noise_level: float,
    *,
    directivity_index: float = 0.0,
    detection_threshold: float = 0.0,
) -> SonarEquationResult
```

Passive sonar equation $\mathrm{SE} = \mathrm{SL} - \mathrm{PL} - (\mathrm{NL} - \mathrm{DI}) - \mathrm{DT}$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `source_level` | Source level `SL` (of the target), in dB. |
| `propagation_loss` | One-way propagation loss `PL`, in dB (scalar or array). |
| `noise_level` | Background noise level `NL`, in dB. |
| `directivity_index` | Receiver directivity index `DI`, in dB. |
| `detection_threshold` | Detection threshold `DT`, in dB. |

**Returns:** A [`SonarEquationResult`](/phonometry/reference/api/underwater/sonar-equation/#sonarequationresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If an input is not finite. |

## SonarEquationResult

```python
SonarEquationResult(
    mode: str,
    signal_excess: NDArray[np.float64],
    snr: NDArray[np.float64],
    figure_of_merit: float,
    propagation_loss: NDArray[np.float64],
    source_level: float,
    noise_level: float,
    directivity_index: float,
    detection_threshold: float,
    target_strength: float | None,
    reverberation_limited: bool,
)
```

Sonar-equation solution.

**Attributes**

| Name | Description |
| :--- | :--- |
| `mode` | `"passive"` or `"active"`. |
| `signal_excess` | Signal excess `SE` per propagation loss, in dB (detection when `SE >= 0`). |
| `snr` | Signal-to-noise (or signal-to-reverberation) ratio, in dB ($\mathrm{SE} + \mathrm{DT}$). |
| `figure_of_merit` | Maximum allowable (one-way) propagation loss at the detection limit $\mathrm{SE} = 0$, in dB. |
| `propagation_loss` | The propagation-loss values, in dB. |
| `source_level` | Source level `SL`, in dB. |
| `noise_level` | Background noise level `NL` input, in dB. The masking term is $\mathrm{NL} - \mathrm{DI}$, except when `reverberation_limited` is true, where the reverberation level `RL` masks instead. |
| `directivity_index` | Receiver directivity index `DI`, in dB. |
| `detection_threshold` | Detection threshold `DT`, in dB. |
| `target_strength` | Target strength `TS`, in dB (`None` for passive). |
| `reverberation_limited` | Whether the active case is reverberation-limited. |

### SonarEquationResult.plot()

```python
SonarEquationResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot signal excess versus propagation loss with the detection limit.
