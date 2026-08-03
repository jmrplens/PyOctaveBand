---
title: "underwater.sonar_equation"
description: "The sonar equation (passive and active), in decibels."
sidebar:
  label: "sonar_equation"
---

The sonar equation (passive and active), in decibels.

Combines the sonar performance terms -- source level `SL`, transmission loss
`TL`, noise level `NL`, directivity index `DI`, detection threshold `DT`,
target strength `TS` and reverberation level `RL` -- into the signal excess
`SE`, the signal-to-noise ratio and the figure of merit (the maximum allowable
transmission loss at the detection limit $\mathrm{SE} = 0$):

* [`passive_sonar_equation`](/phonometry/reference/api/underwater/sonar-equation/#passive_sonar_equation) --
  $\mathrm{SE} = \mathrm{SL} - \mathrm{TL} - (\mathrm{NL} - \mathrm{DI}) - \mathrm{DT}$.
* [`active_sonar_equation`](/phonometry/reference/api/underwater/sonar-equation/#active_sonar_equation) -- monostatic, noise-limited
  $\mathrm{SE} = \mathrm{SL} - 2\,\mathrm{TL} + \mathrm{TS} - (\mathrm{NL} - \mathrm{DI}) - \mathrm{DT}$ or, when a reverberation level
  is given, reverberation-limited
  $\mathrm{SE} = \mathrm{SL} - 2\,\mathrm{TL} + \mathrm{TS} - \mathrm{RL} - \mathrm{DT}$.

All quantities are in dB (levels re a plane wave of 1 µPa rms; the terms are
spectrum levels, i.e. referred to a 1 Hz band). Source: Urick, *Principles of
Underwater Sound*, via Etter (2003), Table 10.2.

The figure of merit is the *maximum allowable transmission loss*, so inverting
a transmission-loss law at $\mathrm{TL} = \mathrm{FOM}$ gives the
**detection range**, the
range at which the detection probability is 50 %:

* [`detection_range`](/phonometry/reference/api/underwater/sonar-equation/#detection_range) inverts the closed-form loss of
  `phonometry.underwater.propagation` (spreading plus volume absorption),
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
    transmission_loss: NDArray[np.float64] | list[float] | float,
    target_strength: float,
    noise_level: float,
    *,
    directivity_index: float = 0.0,
    detection_threshold: float = 0.0,
    reverberation_level: float | None = None,
) -> SonarEquationResult
```

Monostatic active sonar equation with a two-way transmission loss.

Noise-limited: $\mathrm{SE} = \mathrm{SL} - 2\,\mathrm{TL} + \mathrm{TS} - (\mathrm{NL} - \mathrm{DI}) - \mathrm{DT}$. When
`reverberation_level` is given, reverberation-limited:
$\mathrm{SE} = \mathrm{SL} - 2\,\mathrm{TL} + \mathrm{TS} - \mathrm{RL} - \mathrm{DT}$ (`DI` does not apply to reverberation).

**Parameters**

| Name | Description |
| :--- | :--- |
| `source_level` | Source level `SL`, in dB. |
| `transmission_loss` | One-way transmission loss `TL`, in dB (scalar or array); the equation applies $2\,\mathrm{TL}$. |
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

Range at which the closed-form transmission loss equals the figure of
merit.

Solves $\mathrm{TL}(r) = \mathrm{FOM}$ for the loss of
[`transmission_loss`](/phonometry/reference/api/underwater/closed-form/#transmission_loss), which is
strictly increasing in range, so the root is unique. A **one-way** figure of
merit works for both sonar modes: the active figure of merit returned by
[`active_sonar_equation`](/phonometry/reference/api/underwater/sonar-equation/#active_sonar_equation) is already the maximum allowable one-way loss.

**Parameters**

| Name | Description |
| :--- | :--- |
| `figure_of_merit` | Maximum allowable one-way transmission loss, in dB. |
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
    transmission_loss: NDArray[np.float64] | list[float],
    *,
    crossing: str = 'first',
) -> float
```

Detection range read off a computed transmission-loss curve.

Finds where `TL(r)` crosses the figure of merit from below, interpolating
linearly between the two bracketing samples. Real waveguides oscillate, so
`crossing` selects which crossing to report.

**Parameters**

| Name | Description |
| :--- | :--- |
| `figure_of_merit` | Maximum allowable transmission loss, in dB. |
| `range_m` | Ranges, in metres (1-D, strictly increasing). |
| `transmission_loss` | Loss at each range, in dB (same length). |
| `crossing` | `"first"` (default) or `"last"` upward crossing. |

**Returns:** The detection range, in metres. Two limiting cases carry no crossing and are distinguished by the loss at the **last** sample: `inf` when the loss is still below the figure of merit there (the target stays detectable past the end of the grid) and `0.0` when the loss exceeds it there, which without an upward crossing means it exceeded it at every sample and the target is detectable nowhere. [`detection_range`](/phonometry/reference/api/underwater/sonar-equation/#detection_range) returns the same two values for the same two situations.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## DetectionRangeResult

```python
DetectionRangeResult(
    detection_range: float,
    figure_of_merit: float,
    frequency: float,
    range_m: NDArray[np.float64],
    transmission_loss: NDArray[np.float64],
    absorption_coefficient: float,
    law: str,
    model: str,
)
```

Detection range obtained by inverting a transmission-loss law.

**Attributes**

| Name | Description |
| :--- | :--- |
| `detection_range` | Range at which `TL` equals the figure of merit, in metres. `inf` when the loss never reaches it inside `max_range` (detectable throughout) and `0.0` when it already exceeds it at the search floor (detectable nowhere). |
| `figure_of_merit` | The figure of merit inverted, in dB. |
| `frequency` | Acoustic frequency, in Hz. |
| `range_m` | Range grid over which the loss was evaluated, in metres. |
| `transmission_loss` | Transmission loss at each range, in dB. |
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

Plot the transmission loss against the figure of merit.

## passive_sonar_equation

```python
passive_sonar_equation(
    source_level: float,
    transmission_loss: NDArray[np.float64] | list[float] | float,
    noise_level: float,
    *,
    directivity_index: float = 0.0,
    detection_threshold: float = 0.0,
) -> SonarEquationResult
```

Passive sonar equation $\mathrm{SE} = \mathrm{SL} - \mathrm{TL} - (\mathrm{NL} - \mathrm{DI}) - \mathrm{DT}$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `source_level` | Source level `SL` (of the target), in dB. |
| `transmission_loss` | One-way transmission loss `TL`, in dB (scalar or array). |
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
    transmission_loss: NDArray[np.float64],
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
| `signal_excess` | Signal excess `SE` per transmission loss, in dB (detection when `SE >= 0`). |
| `snr` | Signal-to-noise (or signal-to-reverberation) ratio, in dB ($\mathrm{SE} + \mathrm{DT}$). |
| `figure_of_merit` | Maximum allowable (one-way) transmission loss at the detection limit $\mathrm{SE} = 0$, in dB. |
| `transmission_loss` | The transmission-loss values, in dB. |
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

Plot signal excess versus transmission loss with the detection limit.
