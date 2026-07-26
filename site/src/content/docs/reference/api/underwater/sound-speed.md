---
title: "underwater.sound_speed"
description: "Speed of sound in sea water (empirical equations)."
sidebar:
  label: "sound_speed"
---

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

Speed of sound in sea water (empirical equations).

Three coexisting equations for the sound speed `c` as a function of
temperature, salinity and depth/pressure, selectable through `model`:

* `"unesco"` -- the UNESCO / Chen & Millero (1977) algorithm, the
  international standard, in the Wong & Zhu (1995) ITS-90 recalculation. Default.
* `"del_grosso"` -- the Del Grosso (1974) equation (Wong & Zhu 1995 form),
  a high-accuracy alternative over a narrower domain.
* `"mackenzie"` -- the Mackenzie (1981) nine-term depth-based equation.

The UNESCO and Del Grosso equations use pressure, not depth, so a depth is first
converted with the Leroy & Parthiot (1998) standard-ocean formula
([`depth_to_pressure`](/phonometry/reference/api/underwater/sound-speed/#depth_to_pressure)). [`SoundSpeedProfile`](/phonometry/reference/api/underwater/sound-speed/#soundspeedprofile) evaluates `c` over a
depth profile and exposes the sound-speed gradient.

Sources (clean-room, implemented from the equations, validated by cross-model
agreement and the canonical Mackenzie check value 1550.744 m/s at 25 °C, 35 ppt,
1000 m): NPL Technical Guide "Speed of Sound in Sea-Water" (Wong & Zhu 1995
coefficient tables), Mackenzie (1981) JASA 70, Del Grosso (1974) JASA 56,
Leroy & Parthiot (1998) JASA 103.

## depth_to_pressure

```python
depth_to_pressure(depth: float, latitude: float = 45.0) -> float
```

Gauge pressure at a given ocean depth (Leroy & Parthiot 1998), in MPa.

Standard-ocean formula (an ideal medium of 0 °C and 35 ppt); no local
corrections are applied.

**Parameters**

| Name | Description |
| :--- | :--- |
| `depth` | Depth below the surface `Z`, in metres (`>= 0`). |
| `latitude` | Latitude `φ`, in degrees (default 45°). |

**Returns:** Gauge pressure, in megapascals.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the depth is negative or non-finite. |

## sea_water_sound_speed

```python
sea_water_sound_speed(
    temperature: float,
    salinity: float,
    depth: float,
    *,
    model: str = 'unesco',
    latitude: float = 45.0,
) -> float
```

Speed of sound in sea water, in metres per second.

**Parameters**

| Name | Description |
| :--- | :--- |
| `temperature` | Temperature `T`, in degrees Celsius. |
| `salinity` | Salinity `S`, in parts per thousand (PSU). |
| `depth` | Depth below the surface, in metres (`>= 0`). |
| `model` | `"unesco"` (default), `"del_grosso"` or `"mackenzie"`. |
| `latitude` | Latitude for the depth→pressure conversion, in degrees (used by `"unesco"` and `"del_grosso"`; default 45°). |

**Returns:** The sound speed `c`, in m/s.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `model` is unknown or an input is non-finite. |

:::note
Each equation is a fit over a bounded oceanographic domain and
**extrapolates silently outside it** (e.g. Del Grosso abused at
T = 40 °C, S = 0, z = 11 km returns an unphysical ~1995 m/s).
Published validity domains: UNESCO/Chen-Millero T 0-40 °C, S 0-40,
P 0-1000 bar; Del Grosso T 0-30 °C, S 30-40, P 0-1000 kg/cm²;
Mackenzie T 2-30 °C, S 25-40, depth 0-8000 m.
:::

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
| `model` | Sound-speed equation (see [`sea_water_sound_speed`](/phonometry/reference/api/underwater/sound-speed/#sea_water_sound_speed)). |
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
