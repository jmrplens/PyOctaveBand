---
title: "underwater.ship_traffic_noise"
description: "Predicted source-level spectrum of shipping traffic (semi-empirical models)."
sidebar:
  label: "ship_traffic_noise"
---

Predicted source-level spectrum of shipping traffic (semi-empirical models).

When no measured spectrum is available, the underwater radiated-noise source
level of a ship can be *estimated* from readily available traffic parameters
(vessel class, speed, length) with published semi-empirical models. This module
implements three, selectable through `model`:

* `"jomopans-echo"` -- the JOMOPANS-ECHO reference spectrum (MacGillivray &
  de Jong 2021), per **vessel class** with class reference speeds; validated
  against 1862 source-level measurements (ECHO programme), σ ≈ 6 dB. Default.
* `"randi"` -- the RANDI 3.1 semi-empirical model: an "average ship" baseline
  spectrum scaled by speed and length.
* `"wales-heitmeyer"` -- the Wales & Heitmeyer (2002) ensemble merchant-ship
  spectrum (no speed/length dependence), valid 30 Hz-1200 Hz.

All three return an equivalent-monopole source spectral-density level (dB re
1 µPa²/Hz at 1 m, source depth 6 m) and the decidecade-band source level
(dB re 1 µPa m). The predicted spectrum can be used as the `shipping` input of
[`phonometry.underwater.ocean_ambient_noise.ocean_ambient_noise`](/phonometry/reference/api/underwater/ocean-ambient-noise/#ocean_ambient_noise) or placed at range
with [`phonometry.underwater.propagation.transmission_loss`](/phonometry/reference/api/underwater/propagation/#transmission_loss).

Source (clean-room, implemented from the equations, validated against the
authors' own Excel reference implementation, File S1): MacGillivray, A.;
de Jong, C. (2021), "A Reference Spectrum Model for Estimating Source Levels of
Marine Shipping Based on Automated Identification System Data", J. Mar. Sci.
Eng. 9(4), 369, https://doi.org/10.3390/jmse9040369 (CC-BY) -- which also
reproduces RANDI 3.1 [Breeding et al.] and Wales & Heitmeyer (2002).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## ship_source_spectrum

```python
ship_source_spectrum(
    speed_knots: float = 12.0,
    length_m: float = 100.0,
    *,
    vessel_class: str = 'containership',
    model: str = 'jomopans-echo',
    frequency_hz: NDArray[np.float64] | list[float] | None = None,
) -> ShipTrafficSpectrum
```

Predicted underwater source-level spectrum of a ship.

**Parameters**

| Name | Description |
| :--- | :--- |
| `speed_knots` | Vessel speed, in knots (used by `"jomopans-echo"` and `"randi"`; ignored by `"wales-heitmeyer"`). |
| `length_m` | Vessel length, in metres (used by `"jomopans-echo"` and `"randi"`; ignored by `"wales-heitmeyer"`). |
| `vessel_class` | JOMOPANS-ECHO vessel class (see [`VESSEL_CLASSES`](/phonometry/reference/api/underwater/ship-traffic-noise/#vessel_classes)); used only by `"jomopans-echo"`. |
| `model` | `"jomopans-echo"` (default), `"randi"` or `"wales-heitmeyer"`. |
| `frequency_hz` | Frequencies, in Hz; defaults to the decidecade bands 10 Hz-31.5 kHz of the JOMOPANS-ECHO validation range. |

**Returns:** A [`ShipTrafficSpectrum`](/phonometry/reference/api/underwater/ship-traffic-noise/#shiptrafficspectrum).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `model`/`vessel_class` is unknown or an input is invalid. |

## ShipTrafficSpectrum

```python
ShipTrafficSpectrum(
    frequency: NDArray[np.float64],
    source_psd: NDArray[np.float64],
    band_level: NDArray[np.float64],
    model: str,
    vessel_class: str | None,
    speed_knots: float | None,
    length_m: float | None,
)
```

Predicted ship source-level spectrum.

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequencies, in Hz. |
| `source_psd` | Source pressure spectral-density level, in dB re 1 µPa²/Hz at 1 m (equivalent monopole). |
| `band_level` | Decidecade-band source level, in dB re 1 µPa m (`source_psd + 10·log10(0.231·f)`). |
| `model` | The model used. |
| `vessel_class` | The vessel class (JOMOPANS-ECHO only; else `None`). |
| `speed_knots` | Speed used, in knots (`None` if the model ignores it). |
| `length_m` | Length used, in metres (`None` if the model ignores it). |

### ShipTrafficSpectrum.plot()

```python
ShipTrafficSpectrum.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the predicted source spectral-density level versus frequency.

## VESSEL_CLASSES

*Constant* (`tuple`).

```python
VESSEL_CLASSES = ('bulker', 'containership', 'cruise', 'dredger', 'fishing', 'government/research', 'naval', 'other', 'passenger', 'recreational', 'tanker', 'tug', 'vehicle carrier')
```
