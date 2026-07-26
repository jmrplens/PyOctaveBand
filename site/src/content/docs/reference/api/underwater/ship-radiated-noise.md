---
title: "underwater.ship_radiated_noise"
description: "Ship radiated noise and equivalent monopole source level (ISO 17208-1/-2)."
sidebar:
  label: "ship_radiated_noise"
---

Ship radiated noise and equivalent monopole source level (ISO 17208-1/-2).

A surface ship measured in deep water is characterised by its **radiated noise
level** and then by an **equivalent monopole source level** referred to a point
source below the sea surface:

* [`radiated_noise_level`](/phonometry/reference/api/underwater/ship-radiated-noise/#radiated_noise_level) -- `LRN = 20·lg(p_rms/p₀) + 20·lg(r/r₀)`
  dB re 1 µPa·m (ISO 17208-1), the level of the product of the far-field RMS
  pressure and the source distance.
* [`monopole_source_level`](/phonometry/reference/api/underwater/ship-radiated-noise/#monopole_source_level) -- converts `LRN` to the source level
  `Ls = LRN + ΔL` with the Lloyd's-mirror surface correction `ΔL` of
  ISO 17208-2 Formula 3, for a nominal source depth `d_s = 0.7·D` (Formula 1).

Supporting helpers give the ISO 17208-1 three-hydrophone measurement depths
([`hydrophone_depths`](/phonometry/reference/api/underwater/ship-radiated-noise/#hydrophone_depths)) and the ISO 17208-2 tabulated source-level
uncertainty ([`source_level_uncertainty`](/phonometry/reference/api/underwater/ship-radiated-noise/#source_level_uncertainty)). The conversion assumes an ideal
pressure-release sea surface and ignores wind; the reported source level is an
*equivalent monopole broadside* value and must be quoted with its source depth.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## hydrophone_depths

```python
hydrophone_depths(
    cpa_distance: float,
    angles: tuple[float, ...] = (15.0, 30.0, 45.0),
) -> NDArray[np.float64]
```

Hydrophone depths for the ISO 17208-1 deep-water geometry.

At the closest point of approach the three hydrophones sit at depression
angles from the sea surface seen from the ship reference point; at a
horizontal range equal to `cpa_distance` the depth of each is
`d = cpa·tan(angle)`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `cpa_distance` | Horizontal distance at the closest point of approach, in m (`dCPA = max(100 m, ship length)`). |
| `angles` | Depression angles, in degrees (default 15°, 30°, 45°). |

**Returns:** The hydrophone depths, in m.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the distance or any angle is out of range. |

## monopole_source_level

```python
monopole_source_level(
    rnl: float | NDArray[np.float64] | list[float],
    frequency: float | NDArray[np.float64] | list[float],
    draught: float,
    *,
    c: float = 1500.0,
) -> ShipSourceLevelResult
```

Equivalent monopole source level from radiated noise level (ISO 17208-2).

`Ls = LRN + ΔL` with the surface correction (Formula 3)
`ΔL = −10·lg[(2u⁴ + 14u²) / (14 + 2u² + u⁴)]`, `u = k·d_s`,
`k = 2πf/c` and the nominal source depth `d_s = 0.7·D` (Formula 1).

**Parameters**

| Name | Description |
| :--- | :--- |
| `rnl` | Radiated noise level per frequency, in dB re 1 µPa·m (scalar or array; array length must match `frequency`). |
| `frequency` | Frequency or frequencies, in Hz. |
| `draught` | Ship draught `D` (mean of bow and stern), in m. |
| `c` | Speed of sound in sea water, in m/s (default 1500). |

**Returns:** A [`ShipSourceLevelResult`](/phonometry/reference/api/underwater/ship-radiated-noise/#shipsourcelevelresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid or the shapes mismatch. |

## radiated_noise_level

```python
radiated_noise_level(rms_pressure: float, distance: float) -> float
```

Radiated noise level `LRN` (ISO 17208-1), dB re 1 µPa·m.

`LRN = 20·lg(p_rms/p₀) + 20·lg(r/r₀)` -- the level of the product of the
far-field RMS sound pressure and the source distance, referred to
1 µPa·m.

**Parameters**

| Name | Description |
| :--- | :--- |
| `rms_pressure` | Far-field RMS sound pressure `p_rms`, in Pa. |
| `distance` | Distance `r` from the ship reference point, in m. |

**Returns:** Radiated noise level, in dB re 1 µPa·m.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the pressure or distance is not positive. |

## ShipSourceLevelResult

```python
ShipSourceLevelResult(
    frequencies: NDArray[np.float64],
    radiated_noise_level: NDArray[np.float64],
    surface_correction: NDArray[np.float64],
    source_level: NDArray[np.float64],
    source_depth: float,
    sound_speed: float,
)
```

Equivalent monopole source level of a ship (ISO 17208-2).

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Frequencies, in Hz. |
| `radiated_noise_level` | Input RNL per frequency, in dB re 1 µPa·m. |
| `surface_correction` | Lloyd's-mirror correction `ΔL` per frequency, dB. |
| `source_level` | Equivalent monopole source level `Ls = LRN + ΔL`, in dB re 1 µPa·m. |
| `source_depth` | Nominal source depth `d_s = 0.7·D`, in m. |
| `sound_speed` | Speed of sound used, in m/s. |

### ShipSourceLevelResult.plot()

```python
ShipSourceLevelResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot RNL, source level and the ΔL surface correction vs frequency.

## source_level_uncertainty

```python
source_level_uncertainty(frequency: float) -> float
```

Tabulated expanded source-level uncertainty (ISO 17208-2 §5), in dB.

5 dB for the low-frequency bands (the standard lists 10 Hz-100 Hz; values
below 10 Hz reuse it), 3 dB for the mid-frequency bands (125 Hz-16 kHz)
and 4 dB for the high-frequency bands. The standard's own wording leaves
the 20 kHz band unassigned (its high band starts "above 20 000 Hz");
this implementation takes the conservative reading and applies the 4 dB
high-band value from just above 16 kHz. These are representative values,
not exact.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | One-third-octave band centre frequency, in Hz. |

**Returns:** The representative expanded uncertainty, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the frequency is not positive. |
