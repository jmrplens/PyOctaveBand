---
title: "vibration.radiation_efficiency"
description: "Radiation efficiency of a plate in bending (Hopkins 2007, Sound Insulation, Section 2.9; Leppington et al. 1982; Maidanik 1962)."
sidebar:
  label: "radiation_efficiency"
---

Radiation efficiency of a plate in bending (Hopkins 2007, Sound Insulation,
Section 2.9; Leppington et al. 1982; Maidanik 1962).

The **radiation efficiency** `sigma` of a vibrating plate relates the airborne
sound power it radiates to its mean-square surface velocity:

```text
P = rho0 c0 S <v**2> sigma                                  [W]
```

so `sigma` is exactly the radiation factor `epsilon` that
[`phonometry.sound_power_from_vibration`](/phonometry/reference/api/power/vibration-sound-power/#sound_power_from_vibration) (ISO/TS 7849) otherwise takes as a
measured input: this module predicts it from the plate geometry and its
coincidence (critical) frequency, closing the ISO 7849 chain without a power
measurement, and it supplies the *resonant* transmission path of the single- and
double-leaf sound-reduction-index predictions in
[`phonometry.building.panel_transmission`](/phonometry/reference/api/building/panel-transmission/).

**Coincidence (critical) frequency (Hopkins Eq. 2.201).** Below it the free
bending wavelength is shorter than the acoustic wavelength, so the plate
radiates weakly; above it the bending wave is supersonic and radiates
efficiently:

```text
fc = (c0**2 / (2 pi)) sqrt(m'' / B')
```

with `m''` the mass per unit area (kg/m^2) and `B'` the bending stiffness
per unit width (N.m). This is the closed form `fc = 0.55 c0**2 / (cL h)` in
terms of the plate longitudinal wave speed `cL` and thickness `h`.

**Frequency-averaged efficiency (Hopkins 2.9.4, "method no. 1").** With
`mu = sqrt(fc / f)` (Eq. 2.228), the perimeter `U`, area `S`, the boundary
constant `C_BC` (1 simply supported, 2 clamped) and the baffle-orientation
constant `C_OB` (1 plate flush in an infinite baffle, 2 baffles perpendicular
to the edges):

* below `fc` (Eq. 2.227):

```text
    sigma = U / (2 pi mu k S sqrt(mu**2 - 1))
            * [ ln((mu + 1)/(mu - 1)) + 2 mu/(mu**2 - 1) ]
            * [ C_BC C_OB - mu**-8 (C_BC C_OB - 1) ]

with ``k = 2 pi f / c0`` the acoustic wavenumber;
```

* above `fc` (Eq. 2.229): `sigma = 1 / sqrt(1 - mu**2) = (1 - fc/f)**-0.5`,
  so `sigma -> 1` well above coincidence;
* in the band that contains `fc` (Eq. 2.230):
  `sigma ~= (0.5 - 0.15 L1/L2) sqrt(k fc L1)` with `k fc = 2 pi fc / c0`,
  `L1` the smaller and `L2` the larger plate dimension.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## coincidence_frequency

```python
coincidence_frequency(
    mass_per_area: float,
    bending_stiffness: float,
    *,
    speed_of_sound: float = 343.0,
) -> float
```

Coincidence (critical) frequency `fc` of a thin plate (Hopkins 2.201).

`fc = (c0**2 / 2 pi) sqrt(m'' / B')` (identical to Bies Eq. 7.3).

**Parameters**

| Name | Description |
| :--- | :--- |
| `mass_per_area` | Mass per unit area `m''`, in kg/m^2. |
| `bending_stiffness` | Bending stiffness per unit width `B'`, in N.m (see [`phonometry.vibration.point_mobility.plate_bending_stiffness`](/phonometry/reference/api/vibration/point-mobility/#plate_bending_stiffness)). |
| `speed_of_sound` | Speed of sound in air `c0` (Default: 343 m/s). |

**Returns:** The coincidence frequency `fc`, in hertz.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive input. |

## plot_plate_geometry

```python
plot_plate_geometry(
    length_x: float,
    length_y: float,
    ax: Axes | None = None,
    *,
    boundary: str = 'simply_supported',
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Draw the baffled rectangular plate of the radiation model, to scale.

Plate a x b inside its baffle frame, boundary condition in the title.

**Parameters**

| Name | Description |
| :--- | :--- |
| `length_x` | Plate length `a`, in metres. |
| `length_y` | Plate width `b`, in metres. |
| `ax` | Existing axes, or `None` to create a figure. |
| `boundary` | Boundary-condition label of the model. |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the plate rectangle. |

**Returns:** The axes.

## radiation_efficiency

```python
radiation_efficiency(
    frequency: ArrayLike,
    length_x: float,
    length_y: float,
    critical_frequency: float,
    *,
    boundary: str = 'simply_supported',
    baffle: str = 'infinite',
    speed_of_sound: float = 343.0,
) -> RadiationEfficiencyResult
```

Frequency-averaged radiation efficiency of a plate (Hopkins 2.9.4).

Implements Hopkins "method no. 1" (Eqs 2.227, 2.229, 2.230): the below-,
above- and at-coincidence expressions of Leppington/Maidanik. The band whose
centre lies closest (on a log scale) to *critical_frequency* uses the
at-coincidence expression (Eq. 2.230); all others use the below/above
expressions.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Band centre frequencies `f`, in hertz (array, > 0). |
| `length_x` | Plate dimension `Lx`, in m (> 0). |
| `length_y` | Plate dimension `Ly`, in m (> 0). |
| `critical_frequency` | Coincidence frequency `fc`, in hertz (> 0); see [`coincidence_frequency`](/phonometry/reference/api/vibration/radiation-efficiency/#coincidence_frequency). |
| `boundary` | `"simply_supported"` (`C_BC = 1`) or `"clamped"` (`C_BC = 2`). |
| `baffle` | `"infinite"` (`C_OB = 1`, plate flush in a rigid baffle) or `"perpendicular"` (`C_OB = 2`, baffles perpendicular to the edges). |
| `speed_of_sound` | Speed of sound in air `c0` (Default: 343 m/s). |

**Returns:** A [`RadiationEfficiencyResult`](/phonometry/reference/api/vibration/radiation-efficiency/#radiationefficiencyresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive input or unknown boundary/baffle. |

## RadiationEfficiencyResult

```python
RadiationEfficiencyResult(
    frequencies: np.ndarray,
    radiation_efficiency: np.ndarray,
    critical_frequency: float,
    length_x: float,
    length_y: float,
    boundary: str,
    baffle: str,
)
```

Frequency-averaged plate radiation efficiency (Hopkins 2.9.4).

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Band centre frequencies, in hertz. |
| `radiation_efficiency` | Radiation efficiency `sigma` per band. |
| `critical_frequency` | Coincidence frequency `fc`, in hertz. |
| `length_x` | Plate dimension `Lx`, in m. |
| `length_y` | Plate dimension `Ly`, in m. |
| `boundary` | Boundary condition (`"simply_supported"` / `"clamped"`). |
| `baffle` | Baffle orientation (`"infinite"` / `"perpendicular"`). |

### RadiationEfficiencyResult.plot()

```python
RadiationEfficiencyResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the radiation efficiency `sigma(f)` on log-log axes.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### RadiationEfficiencyResult.plot_geometry()

```python
RadiationEfficiencyResult.plot_geometry(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Draw the baffled plate to scale.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### RadiationEfficiencyResult.radiation_index

*property*

Radiation index `10 lg(sigma)` per band, in dB.
