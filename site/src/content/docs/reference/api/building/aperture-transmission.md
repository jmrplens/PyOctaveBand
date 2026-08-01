---
title: "building.aperture_transmission"
description: "Sound transmission through slits, holes and apertures (Hopkins 2007, Sound Insulation, Section 4.3.10; Gomperts 1964; Wilson & Soroka 1965)."
sidebar:
  label: "aperture_transmission"
---

Sound transmission through slits, holes and apertures (Hopkins 2007, Sound
Insulation, Section 4.3.10; Gomperts 1964; Wilson & Soroka 1965).

Air paths are the real limit on the sound insulation of an otherwise heavy
construction: a small slit or hole caps the achievable sound reduction index no
matter how massive the wall. This module predicts the transmission coefficient
`tau` of the two canonical apertures and combines them with the surrounding
wall into a composite sound reduction index, the practical answer to "why do I
never reach the catalogue `Rw`".

**Straight-edged slit (Hopkins Eq. 4.99, Gomperts).** With the acoustic
wavenumber $k = 2\pi f / c_0$, $K = k w$ (`w` slit width),
$X = d / w$ (`d` slit depth) and the end correction
$e = (1/\pi)(\ln(8/K) - 0.57722)$:

$$
\tau = \frac{m K \cos^2(Ke)} {2 n^2 \left[ \frac{\sin^2(KX + 2Ke)}{\cos^2(Ke)} + \frac{K^2}{2 n^2} \left( 1 + \cos(KX) \cos(KX + 2Ke) \right) \right]}
$$

where $m = 8$ (diffuse field) or `4` (normal incidence), and
$n = 1$ (slit in the middle of a plate) or `0.5` (slit along an
edge). The model assumes an inviscid air path; maxima in `tau` (dips in
`R`) occur at the resonances $d + 2e = z\lambda/2$ (Eq. 4.101,
$z = 1, 2, 3, \ldots$).

**Circular aperture (Hopkins Eq. 4.102, Wilson & Soroka).** With the piston
radiation resistance $R_0(2ka) = 1 - 2 J_1(2ka)/(2ka)$ and reactance
$X_0(2ka) = 2 H_1(2ka)/(2ka)$ ($J_1$ Bessel, $H_1$ Struve;
radius `a`, depth `d`):

$$
\tau = \frac{4 R_0} {4 R_0^2 \left[ \cos(kd) - X_0 \sin(kd) \right]^2 + \left[ \left( R_0^2 - X_0^2 + 1 \right) \sin(kd) + 2 X_0 \cos(kd) \right]^2}
$$

**Composite (Hopkins Eq. 4.92).** For elements of area $S_n$ and sound
reduction index $R_n$ the resultant is the area-weighted energy sum

$$
R = -10 \log_{10}\!\left[ \frac{1}{\sum S_n} \sum S_n \, 10^{-R_n/10} \right]
$$

so a bare opening ($R = 0$, $\tau = 1$) of relative area
$S_a/S$ caps the composite at $10 \log_{10}(S / S_a)$. This is the same
energetic combination used by the EN 12354-3/-4 facade model of
[`phonometry.building.facade_prediction`](/phonometry/reference/api/building/facade-prediction/).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## ApertureTransmissionResult

```python
ApertureTransmissionResult(
    frequencies: np.ndarray,
    transmission_coefficient: np.ndarray,
    kind: str,
    width: float | None = None,
    radius: float | None = None,
    depth: float | None = None,
)
```

Transmission through a slit or circular aperture (Hopkins 4.3.10).

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Band centre frequencies, in hertz. |
| `transmission_coefficient` | Transmission coefficient `tau` per band. |
| `kind` | `"slit"` or `"circular"`. |
| `width` | Slit width, in metres, retained (with `depth`) so `plot_geometry` can draw the section; appended after the original fields and `None` for circular apertures or hand-built results. |
| `radius` | Circular-aperture radius, in metres, or `None`. |
| `depth` | Wall thickness, in metres, or `None`. |

### ApertureTransmissionResult.plot()

```python
ApertureTransmissionResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the aperture sound reduction index `R(f)`.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### ApertureTransmissionResult.plot_geometry()

```python
ApertureTransmissionResult.plot_geometry(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Draw the wall-aperture cross-section to scale.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the result does not retain its geometry. |

### ApertureTransmissionResult.transmission_loss

*property*

Aperture sound reduction index $R = -10 \log_{10}(\tau)$ per
band, dB.

## circular_aperture_transmission_coefficient

```python
circular_aperture_transmission_coefficient(
    frequency: ArrayLike,
    radius: float,
    depth: float,
    *,
    speed_of_sound: float = 343.0,
) -> ApertureTransmissionResult
```

Transmission coefficient of a circular aperture (Hopkins Eq. 4.102).

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Band centre frequencies `f`, in hertz (array, > 0). |
| `radius` | Aperture radius `a`, in m (> 0). |
| `depth` | Aperture depth `d` (wall thickness), in m (> 0). |
| `speed_of_sound` | Speed of sound in air `c0` (Default: 343 m/s). |

**Returns:** An [`ApertureTransmissionResult`](/phonometry/reference/api/building/aperture-transmission/#aperturetransmissionresult) (kind `"circular"`).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive input. |

## composite_transmission_loss

```python
composite_transmission_loss(
    areas: ArrayLike,
    reduction_indices: ArrayLike,
) -> np.ndarray
```

Composite sound reduction index of parallel elements (Hopkins
Eq. 4.92).

The area-weighted energy combination of `N` elements (wall, window,
slit, open aperture ...) sharing a partition:

$$
R = -10 \log_{10}\!\left[ \frac{1}{\sum S_n} \sum S_n \, 10^{-R_n/10} \right]
$$

A bare opening enters with $R = 0$ ($\tau = 1$).

**Parameters**

| Name | Description |
| :--- | :--- |
| `areas` | Element areas `S_n`, in m^2 (1-D, length `N`, all > 0). |
| `reduction_indices` | Element sound reduction indices `R_n`, in dB. Either a 1-D array of length `N` (one value per element) or a 2-D array of shape `(N, M)` (`N` elements over `M` bands). |

**Returns:** The composite `R`: a scalar array for 1-D input, or one value per band (length `M`) for 2-D input.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive area, mismatched shapes, or an empty element set. |

## plot_aperture_geometry

```python
plot_aperture_geometry(
    depth: float,
    ax: Axes | None = None,
    *,
    width: float | None = None,
    radius: float | None = None,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Draw the section through a wall aperture to scale.

Wall of thickness `depth` with a slit of the given `width` (or the
diametral section of a circular hole of the given `radius`), incident
sound on the left and the transmitted wavefronts sketched on the right.
Give exactly one of `width`/`radius`, matching
[`slit_transmission_coefficient`](/phonometry/reference/api/building/aperture-transmission/#slit_transmission_coefficient) /
[`circular_aperture_transmission_coefficient`](/phonometry/reference/api/building/aperture-transmission/#circular_aperture_transmission_coefficient).

**Parameters**

| Name | Description |
| :--- | :--- |
| `depth` | Wall thickness `d`, in metres. |
| `ax` | Existing axes, or `None` to create a figure. |
| `width` | Slit width `w`, in metres. |
| `radius` | Circular-hole radius, in metres. |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the wall rectangles. |

**Returns:** The axes.

## slit_resonance_frequencies

```python
slit_resonance_frequencies(
    depth: float,
    width: float,
    *,
    orders: int = 3,
    speed_of_sound: float = 343.0,
) -> np.ndarray
```

Slit resonance frequencies $d + 2e = z\lambda/2$ (Hopkins
Eq. 4.101).

Maxima in the transmission coefficient (dips in `R`) occur where the
effective slit depth is a half-wavelength multiple. Solved iteratively
because the end correction `e` depends weakly on frequency.

**Parameters**

| Name | Description |
| :--- | :--- |
| `depth` | Slit depth `d`, in m (> 0). |
| `width` | Slit width `w`, in m (> 0). |
| `orders` | Number of resonance orders `z = 1..orders` (>= 1). |
| `speed_of_sound` | Speed of sound in air `c0` (Default: 343 m/s). |

**Returns:** The resonance frequencies (Hz), one per order.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive input, `orders < 1`, or a slit so wide relative to its depth that the effective depth $d + 2e$ is non-positive (no resonance exists; `width` must be much less than the wavelength). |

## slit_transmission_coefficient

```python
slit_transmission_coefficient(
    frequency: ArrayLike,
    width: float,
    depth: float,
    *,
    field: str = 'diffuse',
    position: str = 'mid',
    speed_of_sound: float = 343.0,
) -> ApertureTransmissionResult
```

Transmission coefficient of a straight-edged slit (Hopkins Eq. 4.99).

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Band centre frequencies `f`, in hertz (array, > 0). |
| `width` | Slit width `w`, in m (> 0). |
| `depth` | Slit depth `d` (wall thickness across the slit), in m (> 0). |
| `field` | `"diffuse"` ($m = 8$) or `"normal"` ($m = 4$). |
| `position` | `"mid"` ($n = 1$) or `"edge"` ($n = 0.5$). |
| `speed_of_sound` | Speed of sound in air `c0` (Default: 343 m/s). |

**Returns:** An [`ApertureTransmissionResult`](/phonometry/reference/api/building/aperture-transmission/#aperturetransmissionresult) (kind `"slit"`).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive input or unknown field/position. |

## transmission_loss_from_coefficient

```python
transmission_loss_from_coefficient(tau: ArrayLike) -> np.ndarray
```

Sound reduction index $R = -10 \log_{10}(\tau)$ from a transmission
coefficient.

**Parameters**

| Name | Description |
| :--- | :--- |
| `tau` | Transmission coefficient(s) `tau` (> 0). Values above 1 (a resonating aperture that transmits more than the incident intensity) give a negative `R`. |

**Returns:** The sound reduction index `R`, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive coefficient. |
