---
title: "underwater.sonar_equation"
description: "The sonar equation (passive and active), in decibels."
sidebar:
  label: "sonar_equation"
---

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

The sonar equation (passive and active), in decibels.

Combines the sonar performance terms -- source level `SL`, transmission loss
`TL`, noise level `NL`, directivity index `DI`, detection threshold `DT`,
target strength `TS` and reverberation level `RL` -- into the signal excess
`SE`, the signal-to-noise ratio and the figure of merit (the maximum allowable
transmission loss at the detection limit `SE = 0`):

* [`passive_sonar_equation`](/phonometry/reference/api/underwater/sonar-equation/#passive_sonar_equation) -- `SE = SL − TL − (NL − DI) − DT`.
* [`active_sonar_equation`](/phonometry/reference/api/underwater/sonar-equation/#active_sonar_equation) -- monostatic, noise-limited
  `SE = SL − 2·TL + TS − (NL − DI) − DT` or, when a reverberation level is
  given, reverberation-limited `SE = SL − 2·TL + TS − RL − DT`.

All quantities are in dB (levels re a plane wave of 1 µPa rms; the terms are
spectrum levels, i.e. referred to a 1 Hz band). Source: Urick, *Principles of
Underwater Sound*, via Etter (2003), Table 10.2.

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

Noise-limited: `SE = SL − 2·TL + TS − (NL − DI) − DT`. When
`reverberation_level` is given, reverberation-limited:
`SE = SL − 2·TL + TS − RL − DT` (`DI` does not apply to reverberation).

**Parameters**

| Name | Description |
| :--- | :--- |
| `source_level` | Source level `SL`, in dB. |
| `transmission_loss` | One-way transmission loss `TL`, in dB (scalar or array); the equation applies `2·TL`. |
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

Passive sonar equation `SE = SL − TL − (NL − DI) − DT`.

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
| `snr` | Signal-to-noise (or signal-to-reverberation) ratio, in dB (`SE + DT`). |
| `figure_of_merit` | Maximum allowable (one-way) transmission loss at the detection limit `SE = 0`, in dB. |
| `transmission_loss` | The transmission-loss values, in dB. |
| `source_level` | Source level `SL`, in dB. |
| `noise_level` | Background noise level `NL` input, in dB. The masking term is `NL − DI`, except when `reverberation_limited` is true, where the reverberation level `RL` masks instead. |
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
