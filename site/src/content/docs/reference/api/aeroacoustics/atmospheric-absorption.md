---
title: "aircraft.atmospheric_absorption"
description: "One-third-octave-band atmospheric absorption for aircraft noise (SAE ARP 5534)."
sidebar:
  label: "atmospheric_absorption"
---

One-third-octave-band atmospheric absorption for aircraft noise (SAE ARP 5534).

Aircraft noise certification (14 CFR Part 36, ICAO Annex 16 Vol. I) works with
one-third-octave-band spectra, and correcting a measured flyover to reference
atmospheric conditions requires the band attenuation over the propagation path.
The pure-tone attenuation coefficient is the ISO 9613-1 one (identical, per ARP
5534 §3.1) already provided by [`phonometry.air_absorption.air_attenuation`](/phonometry/reference/api/environment/air-absorption/#air_attenuation);
this module adds the **SAE Method** (ARP 5534 §3.2.2), a regression that turns
the pure-tone mid-band path-length attenuation into the one-third-octave-band
attenuation and stays consistent with the ISO/ANSI Exact Method well beyond the
50 dB limit of the older Approximate Method.

* [`sae_band_attenuation`](/phonometry/reference/api/aeroacoustics/atmospheric-absorption/#sae_band_attenuation) -- the one-third-octave-band attenuation over a
  path, returned as a [`AircraftBandAttenuation`](/phonometry/reference/api/aeroacoustics/atmospheric-absorption/#aircraftbandattenuation) with a `.plot()`.

Source (clean-room, implemented from the standard text): SAE ARP 5534 (2021),
*Application of Pure-Tone Atmospheric Absorption Losses to One-Third-Octave-Band
Data*, Eqs. 7-10; the pure-tone coefficient is ISO 9613-1:1993.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## AircraftBandAttenuation

```python
AircraftBandAttenuation(
    frequency: NDArray[np.float64],
    band_attenuation: NDArray[np.float64],
    midband_attenuation: NDArray[np.float64],
    coefficient: NDArray[np.float64],
    path_length: float,
    temperature: float,
    relative_humidity: float,
    pressure: float,
)
```

One-third-octave-band atmospheric attenuation over a path (SAE ARP 5534).

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequency` | Nominal one-third-octave-band centre frequencies, in Hz. |
| `band_attenuation` | SAE-Method band attenuation `δ_B` per band, in dB. |
| `midband_attenuation` | Pure-tone mid-band path-length attenuation $\delta_t = \alpha \cdot s$ per band, in dB (ISO 9613-1 coefficient). |
| `coefficient` | Pure-tone mid-band attenuation coefficient `α` per band, in dB/m. |
| `path_length` | Propagation path length `s`, in metres. |
| `temperature` | Air temperature, in degrees Celsius. |
| `relative_humidity` | Relative humidity, in percent. |
| `pressure` | Ambient atmospheric pressure, in kPa. |

### AircraftBandAttenuation.plot()

```python
AircraftBandAttenuation.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the band and pure-tone mid-band attenuation versus frequency.

## sae_band_attenuation

```python
sae_band_attenuation(
    frequencies: NDArray[np.float64] | list[float],
    path_length: float,
    *,
    temperature: float = 25.0,
    relative_humidity: float = 70.0,
    pressure: float = 101.325,
) -> AircraftBandAttenuation
```

One-third-octave-band atmospheric attenuation (SAE ARP 5534, SAE Method).

Computes the pure-tone attenuation coefficient at each band's exact mid-band
frequency (ISO 9613-1, $f_{m,i} = 10^{i/10}$), forms the mid-band
path-length attenuation $\delta_t = \alpha \cdot s$ and maps it to
the band attenuation
`δ_B` with the SAE-Method regression (Eqs. 7-8).

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Nominal one-third-octave-band centre frequencies, in Hz (standard range 50 Hz-10 kHz; the method extends to 25 Hz-20 kHz). |
| `path_length` | Propagation path length `s`, in metres (`>= 0`). |
| `temperature` | Air temperature, in degrees Celsius (SAE window ~6-32 °C; default 25 °C, the ARP 5534 reference point). |
| `relative_humidity` | Relative humidity, in percent (SAE window ~20-95 %; default 70 %). |
| `pressure` | Ambient atmospheric pressure, in kPa (default 101.325). |

**Returns:** An [`AircraftBandAttenuation`](/phonometry/reference/api/aeroacoustics/atmospheric-absorption/#aircraftbandattenuation).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |
