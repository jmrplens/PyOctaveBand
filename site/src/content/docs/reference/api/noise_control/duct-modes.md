---
title: "noise_control.duct_modes"
description: "Higher-order acoustic modes in ducts, with the mean-flow cut-on shift."
sidebar:
  label: "duct_modes"
---

Higher-order acoustic modes in ducts, with the mean-flow cut-on shift.

Every plane-wave duct method has the same expiry date: the frequency at which
the first higher-order mode cuts on. Below it a duct carries only plane waves,
the four-pole (transfer-matrix) silencer algebra of
[`phonometry.noise_control.silencers`](/phonometry/reference/api/noise_control/silencers/) is exact and a single sound pressure
describes the whole cross section. Above it several modes propagate at once,
each with its own axial wavenumber, and a plane-wave prediction quietly stops
being right.

This module implements the cut-on analysis of Norton & Karczub, *Fundamentals
of Noise and Vibration Analysis for Engineers* 2nd ed., section 7.3:

* circular ducts (Eq. 7.6),
  $(f_{co})_{pq} = \pi \alpha_{pq} c / (2 \pi a_i)$, with the
  $\pi \alpha_{pq}$ eigenvalues of Table 7.1 that solve
  $J'_p(\kappa_{pq} a_i) = 0$ (the first is 1.8412, which sets the
  plane-wave limit $k a_i < 1.8412$);
* rectangular ducts (Eq. 7.10),
  $(f_{co})_{pq} = (c / 2) \sqrt{(p / a)^2 + (q / b)^2}$;
* the **mean-flow correction** (Eq. 7.8): a uniform axial flow of Mach number
  `M` lowers every cut-on frequency by $\sqrt{1 - M^2}$ and moves the
  cut-on from $k_x = 0$ to
  $k_x = -M \kappa_{pq} / \sqrt{1 - M^2}$ (Eq. 7.9), so the
  dispersion curve, symmetric about the frequency axis in still air, becomes
  asymmetric. For a turbulent rather than uniform profile Norton notes that
  replacing `M` by the centre-line Mach number `M_0` represents the
  convective effect adequately.

The practical consequence in a ventilation duct is blunt: a 0.65 m x 0.4 m
air-conditioning duct cuts on at 264 Hz, so plane-wave silencer algebra is
valid only in the 63 Hz to 250 Hz octaves. [`plane_wave_limit`](/phonometry/reference/api/noise_control/duct-modes/#plane_wave_limit) returns
that frequency and `warn_above_plane_wave_limit` issues the
[`PlaneWaveWarning`](/phonometry/reference/api/noise_control/duct-modes/#planewavewarning) the silencer and duct-path results raise on the
caller's behalf.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## circular_duct_cut_on

```python
circular_duct_cut_on(
    diameter: float,
    *,
    flow_velocity: float = 0.0,
    speed_of_sound: float = 343.0,
    count: int = 6,
) -> DuctModeResult
```

Cut-on frequencies of the higher-order modes of a circular duct.

Norton & Karczub Eq. 7.6 with the Table 7.1 eigenvalues, corrected for a
uniform axial mean flow by Eqs. 7.8 and 7.9:

$$
(f_{co})_{pq} = \frac{\pi \alpha_{pq} c \sqrt{1 - M^2}}{2 \pi a_i}
$$

$$
k_x = \frac{-M \kappa_{pq}}{\sqrt{1 - M^2}}, \qquad \kappa_{pq} = \frac{\pi \alpha_{pq}}{a_i}
$$

The flow lowers every cut-on frequency and shifts the cut-on away from
$k_x = 0$: the mode is already travelling upstream, against the
flow, at
the frequency at which it appears. Only the first twelve modes are
tabulated by Norton, so `count` cannot exceed twelve.

**Parameters**

| Name | Description |
| :--- | :--- |
| `diameter` | Internal duct diameter, m (`a_i` is half of it). |
| `flow_velocity` | Mean axial flow speed `U`, m/s (0 for still air; use the centre-line speed for a turbulent profile). |
| `speed_of_sound` | Speed of sound `c` in the duct fluid, m/s. |
| `count` | How many higher-order modes to return, 1 to 12. |

**Returns:** A [`DuctModeResult`](/phonometry/reference/api/noise_control/duct-modes/#ductmoderesult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | For a non-positive diameter, a sonic or supersonic flow, or a `count` outside 1 to 12. |

## DuctModeResult

```python
DuctModeResult(
    modes: tuple[tuple[int, int], ...],
    cut_on: np.ndarray,
    cut_on_no_flow: np.ndarray,
    axial_wavenumber: np.ndarray,
    mach: float,
    section: str,
    label: str,
)
```

Cut-on frequencies of the higher-order acoustic modes of a duct.

**Attributes**

| Name | Description |
| :--- | :--- |
| `modes` | Mode orders `(p, q)`, ordered by ascending no-flow cut-on frequency. |
| `cut_on` | Cut-on frequency of each mode with the mean flow applied, Hz. |
| `cut_on_no_flow` | Cut-on frequency of each mode in still air, Hz. |
| `axial_wavenumber` | Axial wavenumber `k_x` at cut-on, 1/m (zero without flow, negative with it). |
| `mach` | Mean-flow Mach number $M = U / c$. |
| `section` | `"circular"` or `"rectangular"`. |
| `label` | A short human label of the duct. |

### DuctModeResult.plane_wave_limit

*property*

The lowest cut-on frequency, Hz: plane waves only below it.

### DuctModeResult.plot()

```python
DuctModeResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the cut-on ladder, with and without flow, over the mode order.

Requires matplotlib (`pip install phonometry[plot]`).

**Parameters**

| Name | Description |
| :--- | :--- |
| `ax` | Existing axes, or `None` to create a figure. |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the with-flow `Axes.plot`. |

**Returns:** The axes.

## plane_wave_limit

```python
plane_wave_limit(
    *,
    diameter: float | None = None,
    width: float | None = None,
    height: float | None = None,
    area: float | None = None,
    flow_velocity: float = 0.0,
    speed_of_sound: float = 343.0,
) -> float
```

The first cut-on frequency of a duct: plane waves only below it.

A convenience over [`circular_duct_cut_on`](/phonometry/reference/api/noise_control/duct-modes/#circular_duct_cut_on) and
[`rectangular_duct_cut_on`](/phonometry/reference/api/noise_control/duct-modes/#rectangular_duct_cut_on) that takes whichever description of the
cross section is at hand. Give either `diameter`, or both `width` and
`height`, or `area` (treated as a circular duct of the equivalent
diameter $\sqrt{4S/\pi}$).

**Parameters**

| Name | Description |
| :--- | :--- |
| `diameter` | Internal diameter of a circular duct, m. |
| `width` | Cross-sectional dimension `a` of a rectangular duct, m. |
| `height` | Cross-sectional dimension `b` of a rectangular duct, m. |
| `area` | Cross-sectional area, m2, for a duct described by its area. |
| `flow_velocity` | Mean axial flow speed `U`, m/s. |
| `speed_of_sound` | Speed of sound `c`, m/s. |

**Returns:** The lowest cut-on frequency, Hz.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the cross section is not described exactly once. |

## PlaneWaveWarning

A plane-wave duct method is evaluated above the first cut-on frequency.

Raised by the results that assume one-dimensional propagation (the reactive
silencers, the duct-path cascade) when the analysis reaches frequencies at
which higher-order modes propagate in the duct cross section. The numbers
are still returned: above cut-on they describe the plane-wave mode only,
and a measurement will show the extra modes.

## rectangular_duct_cut_on

```python
rectangular_duct_cut_on(
    width: float,
    height: float,
    *,
    flow_velocity: float = 0.0,
    speed_of_sound: float = 343.0,
    count: int = 3,
) -> DuctModeResult
```

Cut-on frequencies of the higher-order modes of a rectangular duct.

Norton & Karczub Eq. 7.10, with the same $\sqrt{1 - M^2}$
convective factor as the circular case:

$$
(f_{co})_{pq} = \frac{c}{2} \sqrt{\left(\frac{p}{a}\right)^2 + \left(\frac{q}{b}\right)^2} \sqrt{1 - M^2}
$$

where `a` and `b` are the cross-sectional dimensions and `(p, q)`
the mode order; `(0, 0)` is the plane wave and is excluded. The modes are
returned in ascending cut-on order, so the first is always the half
wavelength across the wider side.

**Parameters**

| Name | Description |
| :--- | :--- |
| `width` | Cross-sectional dimension `a`, m. |
| `height` | Cross-sectional dimension `b`, m. |
| `flow_velocity` | Mean axial flow speed `U`, m/s. |
| `speed_of_sound` | Speed of sound `c` in the duct fluid, m/s. |
| `count` | How many higher-order modes to return (at least 1). |

**Returns:** A [`DuctModeResult`](/phonometry/reference/api/noise_control/duct-modes/#ductmoderesult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | For non-positive dimensions, a sonic or supersonic flow, or a non-positive `count`. |
