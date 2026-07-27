---
title: "vibration.junction_transmission"
description: "Bending-wave transmission coefficients for rigid plate junctions (Hopkins 2007, Sound Insulation, Section 5.2.1.3; Cremer et al. 1973; Craik 1981, 1996)."
sidebar:
  label: "junction_transmission"
---

Bending-wave transmission coefficients for rigid plate junctions
(Hopkins 2007, *Sound Insulation*, Section 5.2.1.3; Cremer et al. 1973;
Craik 1981, 1996).

The **wave approach** models a plane bending wave that is incident on a rigid
junction of thin plates at an angle `theta` and, assuming the junction beam is
simply supported (pinned so it can rotate but not translate), produces only
reflected and transmitted **bending** waves (no in-plane conversion). The
resulting angle-resolved transmission coefficients are *frequency independent*,
which is what makes them convenient closed-form building blocks for
statistical-energy-analysis (SEA) and the EN 12354 flanking model. This module
implements the rigid X, T, L and in-line junctions of two thin, homogeneous,
isotropic plates.

**Wave parameters (Hopkins Eqs 5.10 and 5.11, after Cremer et al. 1973).** With
plate `i` of thickness `h_i`, quasi-longitudinal wave speed `cL_i`,
surface density `rho_s,i` (kg/m^2), bending stiffness per unit width `B_i`
and critical frequency `fc_i`:

```text
chi = kB2 / kB1 = (rho_s2 B1 / (rho_s1 B2))**0.25
    = sqrt(h1 cL1 / (h2 cL2)) = sqrt(fc2 / fc1)                    (5.10)

psi = B2 kB2**2 / (B1 kB1**2)
    = (h2 cL2 rho_s2) / (h1 cL1 rho_s1) = (rho_s2 fc1) / (rho_s1 fc2)  (5.11)
```

`chi` is the ratio of bending wavenumbers (it fixes the total-internal-
reflection cut-off `theta_co = arcsin(chi)`) and `psi` is the ratio of the
plates' bending-moment mobilities.

**Transmission around a corner (Hopkins Eq. 5.12, Craik 1981/1996).** For an
incident wave on plate 1, if `chi >= sin(theta)`:

```text
                 0.5 J1 J2 psi cos(theta) sqrt(chi**2 - sin**2(theta))
tau12(theta) = --------------------------------------------------------
                (J2 psi)**2 + chi**2 + J2 psi ( sqrt((1 + sin**2 theta)
                (chi**2 + sin**2 theta)) + sqrt((1 - sin**2 theta)
                (chi**2 - sin**2 theta)) )
```

and `tau12(theta) = 0` for `chi < sin(theta)` (no propagating transmitted
wave beyond the cut-off angle).

**Transmission across a straight section (Hopkins Eq. 5.13, Craik 1981/1996).**
Only the X-junction and T-junction (1) have an in-line (straight-through)
section. If `chi >= sin(theta)`:

```text
                      0.5 chi**2 cos**2(theta)
tau13(theta) = -----------------------------------------  (same denominator
                (J3 psi)**2 + chi**2 + J3 psi ( ... )       shape as 5.12)
```

and for `chi < sin(theta)`:

```text
                                cos**2(theta)
tau13(theta) = ------------------------------------------------------
                2 + (J3 psi)**2 C**2 / chi**4
                    + (2 J3 psi C / chi**2) sqrt(1 + sin**2 theta)
```

with `C = sqrt(chi**2 + sin**2 theta) + sqrt(sin**2 theta - chi**2)`.

**Junction constants.** `J1`, `J2` set the corner coefficient and `J3` the
straight one:

===============  ====  =====  =====
Junction         J1    J2     J3
===============  ====  =====  =====
X                1     1      1
T-junction (1)   2     0.5    0.5
T-junction (2)   2     2      --
L                4     1      --
===============  ====  =====  =====

For T-junction (1) plates 1 and 3 are identical; for T-junction (2) plates 2
and 4 are identical. The straight section is undefined for T-junction (2) and
for the L-junction.

**In-line junction (Hopkins Eq. 5.14, Cremer et al. 1973).** Two collinear
plates (a change of section). Only normal incidence is used; it is within 1 dB
of the angular average when `chi >= 1`:

```text
                         2 (1 + chi)(1 + psi) sqrt(chi psi)     2
tau12 ~= tau12(0 deg) = [ ------------------------------------ ]   (5.14)
                          chi (1 + psi)**2 + 2 psi (1 + chi**2)
```

**Angular average (Hopkins Eq. 5.6).** In a diffuse vibration field every angle
of incidence is equally probable and the incident intensity carries a
`cos(theta)` obliquity factor, so the average transmission coefficient is:

```text
tau_bar_ij = integral_0^(pi/2) tau_ij(theta) cos(theta) d(theta)   (5.6)
```

(the `cos(theta)` weight already normalises the average, since
`integral_0^(pi/2) cos(theta) d(theta) = 1`).

**Coupling loss factor (Hopkins Eq. 2.154).** For a source plate `i` of area
`S_i`, bending-wave group velocity `cg_i` and junction length `L_ij`:

```text
eta_ij = cg_i L_ij tau_ij / (2 pi**2 f S_i)                        (2.154)
```

**Vibration reduction index (Hopkins Eq. 5.116).** The wave-approach value of
the EN 12354 junction descriptor, with `fc_j` the critical frequency of the
**receiving** plate and the reference frequency `f_ref = 1000 Hz`:

```text
K_ij = 10 lg(1 / tau_ij) + 5 lg(fc_j / f_ref)                      (5.116)
```

Combined with the reciprocity relationship below (`tau_bar_12 = chi
tau_bar_21` with `chi = sqrt(fc_2 / fc_1)`) this form is symmetric,
`K_ij = K_ji`, as EN 12354 and ISO 10848 require of the junction descriptor.

**Reciprocity (Hopkins Eq. 5.7, the SEA consistency relationship).** The angular
averages of the two directions are linked by
`tau_bar_ij = tau_bar_ji sqrt(h_i cL_i / (h_j cL_j)) = tau_bar_ji sqrt(fc_j /
fc_i)`, i.e. `tau_bar_12 = chi tau_bar_21`.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## angular_average_transmission_coefficient

```python
angular_average_transmission_coefficient(
    chi: float,
    psi: float,
    junction: str = 'X',
    *,
    section: str = 'corner',
) -> float
```

Diffuse-field angular average of a transmission coefficient (Hopkins 5.6).

`tau_bar = integral_0^(pi/2) tau(theta) cos(theta) d(theta)`, evaluated by
adaptive quadrature.

**Parameters**

| Name | Description |
| :--- | :--- |
| `chi` | Wave parameter `chi` (Eq. 5.10, > 0). |
| `psi` | Wave parameter `psi` (Eq. 5.11, > 0). |
| `junction` | `"X"`, `"T1"`, `"T2"` or `"L"`. |
| `section` | `"corner"` (`tau12`, default) or `"straight"` (`tau13`; only for `"X"`/`"T1"`). |

**Returns:** The angular-average transmission coefficient `tau_bar`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive `chi`/`psi`, an unknown junction or section, or a straight section that does not exist. |

## corner_transmission_coefficient

```python
corner_transmission_coefficient(
    angle: ArrayLike,
    chi: float,
    psi: float,
    junction: str = 'X',
) -> NDArray[np.float64]
```

Transmission around a corner `tau12(theta)` (Hopkins Eq. 5.12).

Returns `0` for angles beyond the cut-off `arcsin(chi)` (only reached
when `chi < 1`).

**Parameters**

| Name | Description |
| :--- | :--- |
| `angle` | Incidence angle `theta`, in **radians** (scalar or array, `0 <= theta <= pi/2`). |
| `chi` | Wave parameter `chi` (Eq. 5.10, > 0). |
| `psi` | Wave parameter `psi` (Eq. 5.11, > 0). |
| `junction` | `"X"`, `"T1"`, `"T2"` or `"L"`. |

**Returns:** `tau12(theta)` (same shape as *angle*).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive `chi`/`psi`, an out-of-range angle or an unknown junction. |

## coupling_loss_factor

```python
coupling_loss_factor(
    transmission_coefficient: ArrayLike,
    group_velocity: float,
    junction_length: float,
    frequency: ArrayLike,
    plate_area: float,
) -> NDArray[np.float64]
```

Coupling loss factor from a transmission coefficient (Hopkins Eq. 2.154).

`eta_ij = cg_i L_ij tau_ij / (2 pi**2 f S_i)` with the source-plate
bending-wave group velocity `cg_i`, the junction length `L_ij`, the
frequency `f` and the source-plate area `S_i`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `transmission_coefficient` | Angular-average `tau_ij` (scalar/array). |
| `group_velocity` | Source-plate bending-wave group velocity `cg_i`, in m/s (> 0). For a thin plate `cg = 2 cB` with the bending phase speed `cB` (see [`phonometry.vibration.point_mobility.plate_bending_wave_speed`](/phonometry/reference/api/vibration/point-mobility/)). |
| `junction_length` | Junction length `L_ij`, in m (> 0). |
| `frequency` | Frequency `f`, in hertz (scalar or array, > 0). |
| `plate_area` | Source-plate area `S_i`, in m^2 (> 0). |

**Returns:** The coupling loss factor `eta_ij` (broadcast of the inputs).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive input. |

## inline_transmission_coefficient

```python
inline_transmission_coefficient(chi: float, psi: float) -> float
```

Normal-incidence transmission across an in-line junction (Hopkins 5.14).

`tau12 = [2 (1 + chi)(1 + psi) sqrt(chi psi) / (chi (1 + psi)**2 +
2 psi (1 + chi**2))]**2` (Cremer et al. 1973). For identical plates
(`chi = psi = 1`) this is 1 (a continuous plate transmits fully).

**Parameters**

| Name | Description |
| :--- | :--- |
| `chi` | Wave parameter `chi` (Eq. 5.10, > 0). |
| `psi` | Wave parameter `psi` (Eq. 5.11, > 0). |

**Returns:** `tau12(0 deg)`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive `chi`/`psi`. |

## junction_transmission

```python
junction_transmission(
    junction: str,
    thickness1: float,
    wave_speed1: float,
    surface_density1: float,
    thickness2: float,
    wave_speed2: float,
    surface_density2: float,
    *,
    angles_deg: ArrayLike | None = None,
) -> JunctionTransmissionResult
```

Bending-wave transmission of a rigid perpendicular plate junction.

Builds the angle-resolved corner (and, for X / T-junction (1), straight)
transmission coefficients of Hopkins Eqs 5.12/5.13 and their diffuse-field
angular averages (Eq. 5.6) from the two plates' properties, together with
the thin-plate critical frequencies `fc = sqrt(12) c0**2 / (2 pi h cL)`
(`c0 = 343 m/s`) used by the Eq. 5.116 vibration reduction index. For the
in-line junction (normal incidence only) use
[`inline_transmission_coefficient`](/phonometry/reference/api/vibration/junction-transmission/#inline_transmission_coefficient).

**Parameters**

| Name | Description |
| :--- | :--- |
| `junction` | `"X"`, `"T1"`, `"T2"` or `"L"`. |
| `thickness1` | Thickness `h1` of the source plate, in m (> 0). |
| `wave_speed1` | Quasi-longitudinal wave speed `cL1` of the source plate, in m/s (> 0). |
| `surface_density1` | Surface density `rho_s1` of the source plate, in kg/m^2 (> 0). |
| `thickness2` | Thickness `h2` of the receiving plate, in m (> 0). |
| `wave_speed2` | Quasi-longitudinal wave speed `cL2` of the receiving plate, in m/s (> 0). |
| `surface_density2` | Surface density `rho_s2` of the receiving plate, in kg/m^2 (> 0). |
| `angles_deg` | Incidence-angle grid in degrees (Default: 0 to 90 in 91 one-degree steps). |

**Returns:** A [`JunctionTransmissionResult`](/phonometry/reference/api/vibration/junction-transmission/#junctiontransmissionresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive input or an unknown junction. |

## junction_wave_parameters

```python
junction_wave_parameters(
    thickness1: float,
    wave_speed1: float,
    surface_density1: float,
    thickness2: float,
    wave_speed2: float,
    surface_density2: float,
) -> tuple[float, float]
```

Wave parameters `chi` and `psi` of a plate pair (Hopkins 5.10/5.11).

`chi = sqrt(h1 cL1 / (h2 cL2))` (Eq. 5.10) and
`psi = (h2 cL2 rho_s2) / (h1 cL1 rho_s1)` (Eq. 5.11), with plate 1 the
plate carrying the incident wave.

**Parameters**

| Name | Description |
| :--- | :--- |
| `thickness1` | Thickness `h1` of plate 1, in m (> 0). |
| `wave_speed1` | Quasi-longitudinal wave speed `cL1` of plate 1, in m/s (> 0). |
| `surface_density1` | Surface density `rho_s1` of plate 1, in kg/m^2 (> 0). |
| `thickness2` | Thickness `h2` of plate 2, in m (> 0). |
| `wave_speed2` | Quasi-longitudinal wave speed `cL2` of plate 2, in m/s (> 0). |
| `surface_density2` | Surface density `rho_s2` of plate 2, in kg/m^2 (> 0). |

**Returns:** The pair `(chi, psi)`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive input. |

## JunctionTransmissionResult

```python
JunctionTransmissionResult(
    junction: str,
    chi: float,
    psi: float,
    critical_frequency1: float,
    critical_frequency2: float,
    angles_deg: np.ndarray,
    corner: np.ndarray,
    straight: np.ndarray | None,
    corner_average: float,
    straight_average: float | None,
    thickness1: float | None = None,
    thickness2: float | None = None,
)
```

Bending-wave transmission across a rigid plate junction (Hopkins 5.2.1.3).

**Attributes**

| Name | Description |
| :--- | :--- |
| `junction` | Junction type (`"X"`, `"T1"`, `"T2"` or `"L"`). |
| `chi` | Wave parameter `chi` (Eq. 5.10). |
| `psi` | Wave parameter `psi` (Eq. 5.11). |
| `critical_frequency1` | Critical frequency `fc_1` of the source plate, in hertz (thin plate, `c0 = 343 m/s`). |
| `critical_frequency2` | Critical frequency `fc_2` of the receiving plate, in hertz (thin plate, `c0 = 343 m/s`). |
| `angles_deg` | Incidence-angle grid, in degrees. |
| `corner` | Corner transmission coefficient `tau12(theta)` on the grid. |
| `straight` | Straight-section coefficient `tau13(theta)` on the grid, or `None` when the junction has no straight section. |
| `corner_average` | Diffuse-field angular average `tau_bar_12` (Eq. 5.6). |
| `straight_average` | Angular average `tau_bar_13`, or `None`. |
| `thickness1` | Plate 1 thickness, in metres, retained (with `thickness2`) so `plot_geometry` can draw the junction; appended after the original fields and `None` for hand-built results. |
| `thickness2` | Plate 2 thickness, in metres, or `None`. |

### JunctionTransmissionResult.corner_reduction_index

*property*

Wave-approach `K_12` of the corner path, in dB (Hopkins Eq. 5.116).

`K_12 = 10 lg(1 / tau_bar_12) + 5 lg(fc_2 / 1000)` with the receiving
plate's critical frequency `fc_2`. The value is symmetric: building
the reverse junction (plates swapped, and for a T-junction the matching
constants `T1` \<-> `T2`) gives the same `K_21 = K_12`.

### JunctionTransmissionResult.plot()

```python
JunctionTransmissionResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot `tau(theta)` versus incidence angle for this junction.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### JunctionTransmissionResult.plot_geometry()

```python
JunctionTransmissionResult.plot_geometry(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Draw the plate-junction cross-section to scale.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the result does not retain its geometry. |

## plot_junction_geometry

```python
plot_junction_geometry(
    junction: str,
    thickness1: float,
    thickness2: float,
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Draw a plate junction cross-section to scale.

Plate 1 runs horizontally; the perpendicular plate(s) of thickness 2
form the L, T or X. The incident bending wave arrives on plate 1 and
the junction type follows
[`junction_transmission`](/phonometry/reference/api/vibration/junction-transmission/).

**Parameters**

| Name | Description |
| :--- | :--- |
| `junction` | `"L"`, `"T1"`, `"T2"` or `"X"`. |
| `thickness1` | Plate 1 thickness, in metres. |
| `thickness2` | Plate 2 thickness, in metres. |
| `ax` | Existing axes, or `None` to create a figure. |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the plate-1 rectangle. |

**Returns:** The axes.

## straight_transmission_coefficient

```python
straight_transmission_coefficient(
    angle: ArrayLike,
    chi: float,
    psi: float,
    junction: str = 'X',
) -> NDArray[np.float64]
```

Transmission across a straight section `tau13(theta)` (Hopkins 5.13).

Defined only for the X-junction and T-junction (1); both incidence regimes
`chi >= sin(theta)` and `chi < sin(theta)` are covered.

**Parameters**

| Name | Description |
| :--- | :--- |
| `angle` | Incidence angle `theta`, in **radians** (scalar or array, `0 <= theta <= pi/2`). |
| `chi` | Wave parameter `chi` (Eq. 5.10, > 0). |
| `psi` | Wave parameter `psi` (Eq. 5.11, > 0). |
| `junction` | `"X"` or `"T1"` (the only junctions with a straight section). |

**Returns:** `tau13(theta)` (same shape as *angle*).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive `chi`/`psi`, an out-of-range angle, or a junction without a straight section. |

## wave_vibration_reduction_index

```python
wave_vibration_reduction_index(
    transmission_coefficient: ArrayLike,
    critical_frequency_receiver: float,
) -> NDArray[np.float64]
```

Vibration reduction index from a transmission coefficient (Hopkins 5.116).

`K_ij = 10 lg(1 / tau_ij) + 5 lg(fc_j / f_ref)` with `fc_j` the
critical frequency of the **receiving** plate and the reference frequency
`f_ref = 1000 Hz`. Because the angular-average transmission coefficients
satisfy the reciprocity relationship `tau_bar_ij = tau_bar_ji
sqrt(fc_j / fc_i)` (Eq. 5.7), this form is symmetric: `K_ij = K_ji`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `transmission_coefficient` | `tau_ij` (scalar or array, > 0). |
| `critical_frequency_receiver` | Critical frequency `fc_j` of the receiving plate, in hertz (> 0). |

**Returns:** The vibration reduction index `K_ij`, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive `tau` or `fc_j`. |
