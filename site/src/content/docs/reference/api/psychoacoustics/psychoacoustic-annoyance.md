---
title: "psychoacoustics.psychoacoustic_annoyance"
description: "Psychoacoustic annoyance (PA) after Fastl & Zwicker."
sidebar:
  label: "psychoacoustic_annoyance"
---

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

Psychoacoustic annoyance (PA) after Fastl & Zwicker.

Psychoacoustic annoyance combines four hearing sensations -- loudness,
sharpness, fluctuation strength and roughness -- into a single figure that
tracks annoyance ratings from listening experiments. The model is due to
Widmann (1992) and is given in Fastl & Zwicker, *Psychoacoustics: Facts and
Models* (Equations 16.2-16.4):

```text
PA = N5 * (1 + sqrt(wS**2 + wFR**2))
```

with the percentile loudness `N5` in sone and the two loudness-weighted
terms:

```text
wS  = (S - 1.75) * 0.25 * lg(N5 + 10)          for S > 1.75 acum, else 0
wFR = (2.18 / N5**0.4) * (0.4 * F + 0.6 * R)
```

describing sharpness `S` (acum) and the joint influence of fluctuation
strength `F` (vacil) and roughness `R` (asper). Note the "1 +" sits
*outside* the radical (Fastl & Zwicker 2006, Eq. (16.2), p. 328). There is
no ISO standard for PA; the formula is exact, verified against a
hand-computed worked tuple, and matches the combination implemented by the
open SQAT reference implementation.

[`psychoacoustic_annoyance`](/phonometry/reference/api/psychoacoustics/psychoacoustic-annoyance/#psychoacoustic_annoyance) evaluates the model from the four quantities
directly. [`psychoacoustic_annoyance_from_signal`](/phonometry/reference/api/psychoacoustics/psychoacoustic-annoyance/#psychoacoustic_annoyance_from_signal) is a convenience that
derives them from a calibrated pressure signal using the library's existing
models -- N5/S from ISO 532-1 Zwicker loudness and DIN 45692 sharpness, R from
ECMA-418-2 roughness and F from [`fluctuation_strength`](/phonometry/reference/api/psychoacoustics/fluctuation-strength/#fluctuation_strength). That composite mixes model families (Zwicker N5/S,
Sottek R, Osses F); the original PA model was calibrated with Zwicker-family
sensations, so the signal convenience is an engineering estimate, while
[`psychoacoustic_annoyance`](/phonometry/reference/api/psychoacoustics/psychoacoustic-annoyance/#psychoacoustic_annoyance) is the exact model.

## psychoacoustic_annoyance

```python
psychoacoustic_annoyance(
    n5: float,
    sharpness: float,
    fluctuation_strength: float,
    roughness: float,
) -> PsychoacousticAnnoyanceResult
```

Psychoacoustic annoyance from the four hearing sensations (16.2-16.4).

`PA = N5 * (1 + sqrt(wS**2 + wFR**2))` with the loudness-weighted sharpness
term `wS` (Equation 16.3) and the fluctuation/roughness term `wFR`
(Equation 16.4). The sharpness term is zero for `S <= 1.75 acum`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `n5` | Percentile loudness `N5`, in sone (the loudness exceeded 5 % of the time; [`n5`](/phonometry/reference/api/psychoacoustics/loudness-zwicker/#zwickerloudness)). |
| `sharpness` | Sharpness `S`, in acum (DIN 45692). |
| `fluctuation_strength` | Fluctuation strength `F`, in vacil. |
| `roughness` | Roughness `R`, in asper (ECMA-418-2). |

**Returns:** A [`PsychoacousticAnnoyanceResult`](/phonometry/reference/api/psychoacoustics/psychoacoustic-annoyance/#psychoacousticannoyanceresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If any quantity is negative or non-finite. |

## psychoacoustic_annoyance_from_signal

```python
psychoacoustic_annoyance_from_signal(
    x: list[float] | np.ndarray,
    fs: int,
    *,
    field: Literal['free', 'diffuse'] = 'free',
    calibration_factor: float = 1.0,
) -> PsychoacousticAnnoyanceResult
```

Psychoacoustic annoyance from a calibrated pressure signal (convenience).

Derives the four sensations from the library's models and combines them
with [`psychoacoustic_annoyance`](/phonometry/reference/api/psychoacoustics/psychoacoustic-annoyance/#psychoacoustic_annoyance): `N5` from the ISO 532-1 Zwicker
time-varying loudness, `S` from DIN 45692 sharpness, `R` from
ECMA-418-2 roughness and `F` from
[`fluctuation_strength`](/phonometry/reference/api/psychoacoustics/fluctuation-strength/#fluctuation_strength).

:::note
This composite mixes model families (Zwicker `N5`/`S`, Sottek
`R`, Osses `F`); the PA model was calibrated with Zwicker-family
sensations, so treat the signal convenience as an engineering estimate.
For exact, reproducible results pass the four quantities to
[`psychoacoustic_annoyance`](/phonometry/reference/api/psychoacoustics/psychoacoustic-annoyance/#psychoacoustic_annoyance) directly.

`field` selects the sound field for the loudness, sharpness and
roughness front-ends; the fluctuation strength `F` is always computed
in the free field, because the Osses 2016 model has no diffuse-field
variant (see [`fluctuation_strength`](/phonometry/reference/api/psychoacoustics/fluctuation-strength/#fluctuation_strength)).
:::

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Calibrated sound-pressure signal (1-D), in Pa after `calibration_factor`. |
| `fs` | Sample rate, in Hz. |
| `field` | `'free'` (default) or `'diffuse'` sound field for the loudness/sharpness/roughness front-ends (`F` is always free-field). |
| `calibration_factor` | Digital-units-to-Pa factor applied to `x`. |

**Returns:** A [`PsychoacousticAnnoyanceResult`](/phonometry/reference/api/psychoacoustics/psychoacoustic-annoyance/#psychoacousticannoyanceresult).

## PsychoacousticAnnoyanceResult

```python
PsychoacousticAnnoyanceResult(
    annoyance: float,
    n5: float,
    sharpness: float,
    fluctuation_strength: float,
    roughness: float,
    w_s: float,
    w_fr: float,
)
```

Psychoacoustic annoyance and its contributing terms (Fastl & Zwicker).

**Attributes**

| Name | Description |
| :--- | :--- |
| `annoyance` | Psychoacoustic annoyance `PA` (Equation 16.2), dimensionless. |
| `n5` | Percentile loudness `N5` used, in sone. |
| `sharpness` | Sharpness `S` used, in acum. |
| `fluctuation_strength` | Fluctuation strength `F` used, in vacil. |
| `roughness` | Roughness `R` used, in asper. |
| `w_s` | Sharpness term `wS` (Equation 16.3). |
| `w_fr` | Fluctuation/roughness term `wFR` (Equation 16.4). |

### PsychoacousticAnnoyanceResult.plot()

```python
PsychoacousticAnnoyanceResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the PA value and the `wS` / `wFR` term contributions.
