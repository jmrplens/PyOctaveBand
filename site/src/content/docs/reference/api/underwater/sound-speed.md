---
title: "underwater.propagation.sound_speed"
description: "Sound-speed profiles over a column of water."
sidebar:
  label: "sound_speed"
---

Sound-speed profiles over a column of water.

A profile is a description of a place rather than of a substance, so it stays
with the marchers that consume it while the point state of sea water lives in
[`phonometry.fluids.water`](/phonometry/reference/api/fluids/water/). The four sound-speed equations moved there with
it; this module builds a column out of them.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## sound_speed_profile

```python
sound_speed_profile(
    depths: NDArray[np.float64] | list[float],
    temperatures: NDArray[np.float64] | list[float] | float,
    salinities: NDArray[np.float64] | list[float] | float,
    *,
    model: str = 'unesco',
    latitude: float = 45.0,
) -> SoundSpeedProfile
```

Evaluate a sound-speed profile over a depth column.

**Parameters**

| Name | Description |
| :--- | :--- |
| `depths` | Depths, in metres (1-D, non-negative, increasing). |
| `temperatures` | Temperature per depth, in °C (array or a scalar broadcast to every depth). |
| `salinities` | Salinity per depth, in PSU (array or scalar). |
| `model` | Sound-speed equation (see [`sea_water_sound_speed`](/phonometry/reference/api/fluids/water/#sea_water_sound_speed)). |
| `latitude` | Latitude for the depth→pressure conversion, in degrees. |

**Returns:** A [`SoundSpeedProfile`](/phonometry/reference/api/underwater/sound-speed/#soundspeedprofile).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## SoundSpeedProfile

```python
SoundSpeedProfile(
    depth: NDArray[np.float64],
    sound_speed: NDArray[np.float64],
    gradient: NDArray[np.float64],
    model: str,
)
```

Sound-speed profile `c(z)` over a column of water.

**Attributes**

| Name | Description |
| :--- | :--- |
| `depth` | Depths, in metres (increasing downward). |
| `sound_speed` | Sound speed at each depth, in m/s. |
| `gradient` | Vertical sound-speed gradient `dc/dz`, in (m/s)/m. |
| `model` | The equation used. |

### SoundSpeedProfile.plot()

```python
SoundSpeedProfile.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the sound-speed profile (speed vs depth, depth increasing down).
