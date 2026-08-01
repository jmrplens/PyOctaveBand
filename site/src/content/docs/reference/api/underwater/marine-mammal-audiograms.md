---
title: "underwater.marine_mammal_audiograms"
description: "Marine-mammal hearing thresholds (group audiograms and the orca audiogram)."
sidebar:
  label: "marine_mammal_audiograms"
---

Marine-mammal hearing thresholds (group audiograms and the orca audiogram).

Two independent published descriptions of how well a marine mammal hears:

* [`group_audiogram`](/phonometry/reference/api/underwater/marine-mammal-audiograms/#group_audiogram) -- the **group audiogram** of Southall et al. (2019),
  a four-parameter band-pass fit (their Equation 1, after Finneran 2016)

$$
T(f) = T_0 + A \lg\!\left(1 + \frac{F_1}{f}\right) + (f/F_2)^{B}
$$

  with the group parameters of their Table 2 (absolute thresholds) and Table 3
  (normalised to 0 dB at best sensitivity).
* [`orca_audiogram`](/phonometry/reference/api/underwater/marine-mammal-audiograms/#orca_audiogram) -- the killer-whale (*Orcinus orca*) audiogram of
  Wensveen & Van Roij (2007) as printed in Ainslie, *Principles of Sonar
  Performance Modelling* (Springer 2010), Equation (11.159), a **three-branch**
  power law over 0.5 to 80 kHz fitted to the measurements of Hall & Johnson
  (1972) and Szymanski et al. (1999).

Thresholds are sound pressure levels in dB re 1 µPa under water and dB re
20 µPa in air (the two in-air carnivore groups `"PCA"` and `"OCA"`).

:::note
Southall et al. publish **no fitted audiogram for low-frequency (LF)
cetaceans**: no audiometric data exist for them. The article gives
$A = 20$ dB/decade, $B = 3.2$, $F_2 = 9.4$ kHz and
$T_0 = 53.2$ dB (0.8 dB normalised) in prose but never prints
$F_1$, only the criterion used to choose it. The group is
therefore absent from
[`AUDIOGRAM_GROUPS`](/phonometry/reference/api/underwater/marine-mammal-audiograms/#audiogram_groups) rather than reconstructed by guesswork.
:::

Group codes follow Southall et al.: `HF` and `VHF` cetaceans, `SI`
sirenians, `PCW`/`OCW` phocid and otariid carnivores in water and
`PCA`/`OCA` the same in air. Beware that NMFS (2018) calls the Southall
`HF` group `MF` and the Southall `VHF` group `HF`; see
[`phonometry.underwater.marine_mammal_weighting`](/phonometry/reference/api/underwater/marine-mammal-weighting/).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## AUDIOGRAM_GROUPS

*Constant* (`tuple`).

```python
AUDIOGRAM_GROUPS = ('HF', 'VHF', 'SI', 'PCW', 'OCW', 'PCA', 'OCA')
```

## audiogram_parameters

```python
audiogram_parameters(
    group: str,
    *,
    normalized: bool = False,
) -> AudiogramParameters
```

Fit parameters of a published group audiogram.

**Parameters**

| Name | Description |
| :--- | :--- |
| `group` | Hearing-group code, one of [`AUDIOGRAM_GROUPS`](/phonometry/reference/api/underwater/marine-mammal-audiograms/#audiogram_groups) (case-insensitive). |
| `normalized` | Return the Table 3 normalised fit instead of the Table 2 absolute one. |

**Returns:** The [`AudiogramParameters`](/phonometry/reference/api/underwater/marine-mammal-audiograms/#audiogramparameters) for that group.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the group has no published audiogram. |

## AudiogramParameters

```python
AudiogramParameters(
    group: str,
    t0: float,
    f1_khz: float,
    f2_khz: float,
    a: float,
    b: float,
    r_squared: float,
    in_air: bool,
)
```

Group-audiogram fit parameters (Southall et al. 2019, Tables 2 and 3).

**Attributes**

| Name | Description |
| :--- | :--- |
| `group` | Hearing-group code. |
| `t0` | Vertical position `T0`, in dB. |
| `f1_khz` | Low-frequency inflection `F1`, in kHz. |
| `f2_khz` | High-frequency inflection `F2`, in kHz. |
| `a` | Low-frequency slope parameter `A`, in dB/decade. |
| `b` | High-frequency exponent `B`. |
| `r_squared` | Goodness of fit $R^2$ reported with the row. |
| `in_air` | Whether the group's reference pressure is 20 µPa (in air). |

## AudiogramResult

```python
AudiogramResult(
    frequencies: NDArray[np.float64],
    threshold: NDArray[np.float64],
    group: str,
    source: str,
    in_air: bool,
    best_frequency: float,
    best_threshold: float,
)
```

Hearing threshold versus frequency.

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Frequencies, in Hz. |
| `threshold` | Hearing threshold at each frequency, in dB re 1 µPa (under water) or dB re 20 µPa (in air). |
| `group` | Hearing-group code, or `"orca"` for the species audiogram. |
| `source` | Short citation of the fit used. |
| `in_air` | Whether the reference pressure is 20 µPa. |
| `best_frequency` | Frequency of the minimum threshold on the evaluated grid, in Hz. |
| `best_threshold` | The minimum threshold on the evaluated grid, in dB. |

### AudiogramResult.plot()

```python
AudiogramResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the hearing threshold versus frequency.

## BEST_HEARING_FREQUENCY_KHZ

*Constant* (`dict`).

```python
BEST_HEARING_FREQUENCY_KHZ = {'HF': (55.0, 58.0), 'VHF': (105.0, 105.0), 'SI': (16.0, 12.0), 'PCW': (8.6, 13.0), 'OCW': (12.0, 10.0), 'PCA': (2.3, 2.3), 'OCA': (10.0, 10.0)}
```

## group_audiogram

```python
group_audiogram(
    frequency_hz: NDArray[np.float64] | list[float] | float,
    group: str,
    *,
    normalized: bool = False,
) -> AudiogramResult
```

Marine-mammal group audiogram (Southall et al. 2019, Equation 1).

$T(f) = T_0 + A \lg(1 + F_1/f) + (f/F_2)^B$, with `f` in
kilohertz and the group parameters of Table 2 (`normalized=False`) or
Table 3 (`normalized=True`).

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency_hz` | Frequency or frequencies, in Hz (strictly positive). |
| `group` | Hearing-group code, one of [`AUDIOGRAM_GROUPS`](/phonometry/reference/api/underwater/marine-mammal-audiograms/#audiogram_groups). |
| `normalized` | Use the normalised fit (0 dB at best sensitivity). |

**Returns:** An [`AudiogramResult`](/phonometry/reference/api/underwater/marine-mammal-audiograms/#audiogramresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the group is unknown or a frequency is invalid. |

## orca_audiogram

```python
orca_audiogram(
    frequency_hz: NDArray[np.float64] | list[float] | float,
) -> AudiogramResult
```

Killer-whale hearing threshold (Ainslie 2010, Equation 11.159).

A three-branch fit in $F = f/(1~\text{kHz})$, valid over 0.5 to
80 kHz:

* $445.2 F^{-0.05401} - 344.3$ for $0.5 \le F < 11.3$,
* $242.9 F^{-0.7578} + 0.5643 F^{1.076}$ for
  $11.3 \le F < 46.2$,
* $2.792 F^{0.7537} - 2.064$ for $46.2 \le F \le 80$.

The published check points are the minimum, 39.0 dB re 1 µPa at 22.6 kHz
(second branch), and 51.2 dB re 1 µPa at 50 kHz -- the latter **needs the
third branch**; evaluating the second one there returns 50.5 dB instead.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency_hz` | Frequency or frequencies, in Hz, within [`ORCA_AUDIOGRAM_RANGE_KHZ`](/phonometry/reference/api/underwater/marine-mammal-audiograms/#orca_audiogram_range_khz) scaled to hertz. |

**Returns:** An [`AudiogramResult`](/phonometry/reference/api/underwater/marine-mammal-audiograms/#audiogramresult) in dB re 1 µPa.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a frequency falls outside the fitted range. |

## ORCA_AUDIOGRAM_RANGE_KHZ

*Constant* (`tuple`).

```python
ORCA_AUDIOGRAM_RANGE_KHZ = (0.5, 80.0)
```
