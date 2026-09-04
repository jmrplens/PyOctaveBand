---
title: "noise_control.hvac"
description: "HVAC duct acoustics: fan power, duct losses, plenums and flow-generated noise."
sidebar:
  label: "hvac"
---

HVAC duct acoustics: fan power, duct losses, plenums and flow-generated noise.

A ventilation duct network attenuates fan noise through several mechanisms
that add up along the path, and it *regenerates* noise wherever the airflow is
disturbed. This module gathers the element models that a duct-borne noise
calculation needs, from two engineering references that are kept side by side
rather than merged:

* **Bies, Hansen & Howard**, *Engineering Noise Control* 5th ed., Chapter 8,
  for the **duct end reflection** (§8.13, Table 8.14), the **bends/elbows**
  (§8.11, Table 8.11), the **plenum chambers** (§8.17, Wells' method) and the
  **flow-generated (self) noise** of straight ducts and bends (§8.15).
* **Long**, *Architectural Acoustics* 2nd ed., Chapters 13 and 14, for the
  **fan sound power** from the operating point (Eq. 13.1 with the ASHRAE
  Tables 13.5-13.7), the **straight-duct attenuation** of unlined and lined
  rectangular and circular ducts (Eqs. 14.9-14.13 with Tables 14.1-14.3, the
  Reynolds regressions), the **lined flexible duct** insertion loss
  (Table 14.4), the **branch split loss** (Eq. 14.17), the closed-form **end
  reflection** (Eqs. 14.14-14.16), the **silencer self-noise** (Eq. 14.31 with
  Table 14.8) and the **room effect** that turns the sound power arriving at
  the terminal device into a sound pressure level in the room.

Both references trace back to the same ASHRAE data for the elbows: Bies
Table 8.11 is indexed by $W / \lambda$ and Long Tables 14.5-14.7 by the
frequency-width product `f w` (kHz times inches), and the two indexings agree
band by band ($W / \lambda = 0.074 f w$), so [`elbow_insertion_loss`](/phonometry/reference/api/noise_control/hvac/#elbow_insertion_loss)
serves both. Where they genuinely differ -- the end reflection, tabulated by
Bies and given in closed form by Long -- both are selectable
(`method="bies"` or `method="long"`) and neither replaces the other.

[`phonometry.noise_control.duct_path`](/phonometry/reference/api/noise_control/duct-path/) chains these elements into the
end-to-end fan-to-room calculation.

:::note
Bies 5th ed. gives the duct end reflection only as the ASHRAE Table 8.14
look-up (there is no closed form in this edition); this module reproduces
that table and interpolates it. Rectangular ducts use the equivalent
diameter $D = \sqrt{4S/\pi}$.
:::

.. warning::
   Long's worked duct-borne sheet (Table 14.9) was produced by a commercial
   computer program, not by hand from the tables printed alongside it, and
   several of its element rows do **not** follow from the book's own data.
   The functions here implement the *printed* equations and tables, so they
   reproduce some rows of that sheet and not others. Verified band by band:

   * `split_loss` reproduces the 25 per cent split row (-6 dB) exactly, and
     [`elbow_insertion_loss`](/phonometry/reference/api/noise_control/hvac/#elbow_insertion_loss) reproduces the unlined-elbow row exactly
     when the elbow is read as round (Table 14.7) at $w = 24$ in;
   * [`lined_rectangular_duct_attenuation`](/phonometry/reference/api/noise_control/hvac/#lined_rectangular_duct_attenuation) with `include_unlined=True`
     reproduces the 18 x 12 in run from 500 Hz up (11/25/22/16/13 dB) and the
     36 x 24 in run at 500 Hz and 8 kHz, but is 1-2 dB low at 63-250 Hz on
     one run and 2 dB high on the other;
   * the fan row (90/86/82/79/77/75/71/61 dB) is **not** reproducible from
     Eq. 13.1 with the Table 13.5 forward-curved constants, which give
     99/99/89/84/82/77/72/67 dB at the same duty; the printed spectrum is not
     a level shift of the tabulated one, so it comes from other data;
   * the flexible-duct row (14/14/16/15/17/22/16/13 dB) is **not** the
     Table 14.4 entry for 12 in by 6 ft (3/5/10/15/17/16/9 dB);
   * [`diffuser_sound_power`](/phonometry/reference/api/noise_control/hvac/#diffuser_sound_power) reproduces the supply diffuser row
     (33/32/29/23/15/4/0/0 dB) to better than 1 dB in five of the six bands
     that carry it (+0.4/+0.4/+0.2/+0.7/+0.9 dB from 63 Hz to 1 kHz) and to
     1.9 dB in the sixth (2 kHz), reading the device as a 24 x 24 in
     rectangular diffuser;
   * the silencer and grille rows are manufacturer data, which is what a
     real sheet uses and what [`DuctElement`](/phonometry/reference/api/noise_control/duct-path/#ductelement) accepts.

   The cascade arithmetic of that sheet is reproduced exactly (see the
   duct-path tests, which feed it its own printed element rows), and the
   sheet's own internal rounding is 1 dB.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## air_terminal_damper_correction

```python
air_terminal_damper_correction(
    pressure_ratio: float,
    *,
    location: str = 'diffuser_neck',
) -> float
```

Level to add to a diffuser sound rating for a throttled volume damper.

ASHRAE (2019) *HVAC Applications Handbook* Chapter 49, Table 10. A balancing
damper throttled in the neck of a diffuser turns the pressure it drops into
noise right at the outlet, where the room hears it: at a damper pressure
ratio of 3 the published penalty is 15 dB in the neck, 5 dB in the inlet
plenum and 2 dB when the damper sits at least 1.5 m back in the supply duct.
That ordering is the whole design rule: throttle far from the outlet, or
balance the system with duct sizing instead.

The table is interpolated linearly between its tabulated pressure ratios
(1.5 to 6) and held flat outside them.

**Parameters**

| Name | Description |
| :--- | :--- |
| `pressure_ratio` | Damper pressure ratio, the total pressure drop across the damper divided by the pressure drop of the outlet itself. |
| `location` | Where the damper sits: `"diffuser_neck"` (in the neck of a linear diffuser), `"plenum_inlet"` (in the inlet of the plenum of a linear diffuser) or `"supply_duct"` (in the supply duct at least 1.5 m from the inlet plenum). |

**Returns:** The level to add to the diffuser's rated sound power, dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `pressure_ratio` is not positive or `location` is unknown. |

## air_terminal_velocity_limit

```python
air_terminal_velocity_limit(
    design_criterion: float,
    *,
    opening: str = 'supply',
) -> float
```

Maximum recommended neck velocity of a diffuser or register.

ASHRAE (2019) *HVAC Applications Handbook* Chapter 49, Table 9: the "free"
opening airflow velocity not to be exceeded if the room is to reach a given
design `RC(N)`, for use when no sound data is available for the selected
device. It is a screening check, not a spectrum: the sound power of a real
grille, register or diffuser comes from manufacturer data measured to
ASHRAE Standard 70, and [`diffuser_sound_power`](/phonometry/reference/api/noise_control/hvac/#diffuser_sound_power) estimates it when that
data is not to hand. Several devices in the same room, or a damper
throttled in the neck, raise the level further and the allowable velocity
has to be reduced accordingly.

**Parameters**

| Name | Description |
| :--- | :--- |
| `design_criterion` | Design `RC(N)` of the room; one of 25, 30, 35, 40 or 45. |
| `opening` | `"supply"` (supply air outlet) or `"return"` (return air opening). |

**Returns:** The maximum recommended neck velocity, m/s.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the design criterion is not tabulated. |

## blade_passing_frequency

```python
blade_passing_frequency(rotational_speed: float, blades: int) -> float
```

Blade passing frequency (Long Eq. 13.4).

$f_{bp} = \mathrm{rpm} \times \mathrm{blades} / 60$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `rotational_speed` | Fan speed, revolutions per minute. |
| `blades` | Number of impeller blades. |

**Returns:** The blade passing frequency, Hz.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `blades` is not a positive integer. |

## diffuser_sound_power

```python
diffuser_sound_power(
    frequencies: ArrayLike | None,
    face_area: float,
    volume_flow: float,
    pressure_drop: float,
    *,
    shape: str = 'rectangular',
    count: int = 1,
) -> HvacSpectrumResult
```

Regenerated (self) noise of a grille, register or diffuser.

Reynolds's estimate as Long Eqs. 13.27 to 13.33, for when the
manufacturer's ASHRAE Standard 70 data is not to hand. The overall sound
power level is Eq. 13.27:

$$
L_W = 10 \log_{10} S_\mathrm{G} + 30 \log_{10} \xi + 60 \log_{10} U_\mathrm{G} - 31.3
$$

with `S_\mathrm{G}` the face area of the device (ft2),
$U_\mathrm{G} = Q / (60 S_\mathrm{G})$ the approach velocity (ft/s) and
$\xi = 334.9\, dP / (\rho_0 U_\mathrm{G}^2)$ the normalised pressure-drop
coefficient of Eq. 13.28 (`dP` in inches of water gauge,
$\rho_0 = 0.075$ lb/ft3); this function takes and returns SI and
converts internally.

The octave-band spectrum follows from Eq. 13.29,
$L_{W,\mathrm{oct}} = L_W + C_\mathrm{D}$, with the shape functions of Eqs. 13.30
and 13.31:

$$
C_\mathrm{D} = -5.82 - 0.15 A - 1.13 A^2 \qquad \text{(round)}
$$

$$
C_\mathrm{D} = -11.82 - 0.15 A - 1.13 A^2 \qquad \text{(rectangular, including slot)}
$$

normalised to the peak frequency $f_P = 48.8 U_\mathrm{G}$ of Eq. 13.32,
where $A = N_B(f_P) - N_B(f)$ is the distance in octaves from the
peak band (Eq. 13.33) counted on Long's band numbering, 0 at 32 Hz.

The sixth power of velocity in Eq. 13.27 is the design message: the level
rises about 18 dB for every doubling of the approach velocity, and for a
given air volume doubling the face area buys about 15 dB. Nothing
downstream can take that noise back out, because there is no ductwork
left, which is why the terminal device usually sets the room criterion in
the mid and high bands.

Several identical devices serving the same room add $10 \log_{10} n$,
which is what `count` applies.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Octave-band centres, Hz; `None` uses [`OCTAVE_BANDS`](/phonometry/reference/api/materials/rating/#octave_bands). |
| `face_area` | Cross-sectional face area `S_\mathrm{G}` of one device, m2. |
| `volume_flow` | Volume flow `Q` through one device, m3/s. |
| `pressure_drop` | Static pressure drop `dP` across the device, Pa. |
| `shape` | `"rectangular"` (Eq. 13.31, includes slot diffusers) or `"round"` (Eq. 13.30). |
| `count` | Number of identical devices `n` in the room. |

**Returns:** An [`HvacSpectrumResult`](/phonometry/reference/api/noise_control/hvac/#hvacspectrumresult) of the band sound power level, dB re 1e-12 W.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a dimension is not positive, `count` is not a positive integer or `shape` is unknown. |

## elbow_insertion_loss

```python
elbow_insertion_loss(
    frequencies: ArrayLike,
    width: float,
    *,
    bend_type: str = 'square',
    vanes: bool = False,
    lined: bool = False,
    speed_of_sound: float = 343.0,
) -> HvacSpectrumResult
```

Duct bend/elbow insertion loss per bend (Bies Table 8.11, ASHRAE).

Indexed by the frequency-to-width ratio $W / \lambda$
($\lambda = c / f$).
Lined bends assume the lining extends at least three duct diameters up- and
downstream. Round bends are treated as unlined with no vanes.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Frequencies `f`, Hz (1-D array). |
| `width` | Duct width `W` in the plane of the bend, m. |
| `bend_type` | `"square"` or `"round"`. |
| `vanes` | Turning vanes fitted (square bends only). |
| `lined` | Acoustically lined bend (square bends only). |
| `speed_of_sound` | Speed of sound `c`, m/s. |

**Returns:** A [`HvacSpectrumResult`](/phonometry/reference/api/noise_control/hvac/#hvacspectrumresult) of the insertion loss, dB per bend.

## end_reflection_loss

```python
end_reflection_loss(
    frequencies: ArrayLike,
    diameter: float,
    *,
    termination: str = 'flush',
    method: str = 'bies',
    speed_of_sound: float = 343.0,
) -> HvacSpectrumResult
```

Duct end reflection loss (Bies Table 8.14, ASHRAE; or Long's closed form).

The low-frequency reflection of sound back up a duct at its open
termination into a room. Two published methods are offered and neither
replaces the other:

* `method="bies"` (default) interpolates the ASHRAE look-up of Bies
  Table 8.14 over `log` diameter and `log` frequency, passing exactly
  through the tabulated `(diameter, octave band)` nodes. The table covers
  63 Hz to 2 kHz and 150 mm to 1830 mm.
* `method="long"` evaluates Reynolds' closed form as given by Long
  (Eqs. 14.14-14.15), [`end_reflection_loss_closed_form`](/phonometry/reference/api/noise_control/hvac/#end_reflection_loss_closed_form), which has no
  frequency or diameter range limit.

The two agree within a couple of decibels over the bands both cover.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Frequencies `f`, Hz (1-D array). |
| `diameter` | Duct internal diameter `D`, m (use [`equivalent_diameter`](/phonometry/reference/api/noise_control/hvac/#equivalent_diameter) for a rectangular duct of area `S`). |
| `termination` | `"flush"` (duct flush with a wall/ceiling) or `"free"` (free space / suspended in the room). |
| `method` | `"bies"` (Table 8.14 look-up) or `"long"` (closed form). |
| `speed_of_sound` | Speed of sound `c`, m/s (used by the closed form; the table is indexed by frequency directly). |

**Returns:** A [`HvacSpectrumResult`](/phonometry/reference/api/noise_control/hvac/#hvacspectrumresult) of the reflection loss, dB.

## end_reflection_loss_closed_form

```python
end_reflection_loss_closed_form(
    frequencies: ArrayLike,
    diameter: float,
    *,
    termination: str = 'flush',
    speed_of_sound: float = 343.0,
) -> HvacSpectrumResult
```

Duct end reflection loss in closed form (Long Eqs. 14.14-14.15, Reynolds).

$R = 10 \log_{10}[1 + (c / (\pi f d))^{1.88}]$ for a duct terminated in
free space and $R = 10 \log_{10}[1 + (0.8 c / (\pi f d))^{1.88}]$ for one
terminated flush with
a wall, `d` being the duct diameter (use the equivalent diameter
[`equivalent_diameter`](/phonometry/reference/api/noise_control/hvac/#equivalent_diameter) for a rectangular duct, Eq. 14.16). The
exponent 1.88 is Reynolds' empirical fit: the plane-wave area-change result
over-predicts at high frequency, where the sound leaves the duct as a beam
and never sees the expansion. This is the closed-form alternative to the
Bies/ASHRAE table look-up of [`end_reflection_loss`](/phonometry/reference/api/noise_control/hvac/#end_reflection_loss); the two agree
within a couple of decibels over the bands where both are defined.

End-reflection loss does not occur when the duct terminates in a diffuser,
whose flare smooths the impedance transition into the room.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Frequencies `f`, Hz (1-D array). |
| `diameter` | Duct internal diameter `d`, m. |
| `termination` | `"flush"` (flush with a wall or ceiling) or `"free"` (free space). |
| `speed_of_sound` | Speed of sound `c`, m/s. |

**Returns:** An [`HvacSpectrumResult`](/phonometry/reference/api/noise_control/hvac/#hvacspectrumresult) of the reflection loss, dB.

## equivalent_diameter

```python
equivalent_diameter(area: float) -> float
```

Equivalent duct diameter $d = \sqrt{4S/\pi}$ (Long Eq. 14.16).

**Parameters**

| Name | Description |
| :--- | :--- |
| `area` | Duct cross-sectional area `S`, m2. |

**Returns:** The equivalent diameter, m.

## fan_casing_attenuation

```python
fan_casing_attenuation(
    frequencies: ArrayLike | None = None,
) -> HvacSpectrumResult
```

Fan-housing (casing) attenuation of the radiated power (Long Table 13.8).

Subtracted from the sound power level of [`fan_sound_power`](/phonometry/reference/api/noise_control/hvac/#fan_sound_power) to
estimate what the fan radiates *through its housing* into the plant room
rather than into the duct. The values assume no separate enclosure and no
absorption inside the housing, but a silencer or a lining in the ductwork
close to the fan; at low frequency the vibrating casing radiates as much as
the unhoused fan would, hence the zeroes. Miller (1980) states them as
approximate: real values depend strongly on the gauge and construction of
the housing.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Octave-band centres, Hz; `None` (default) uses [`OCTAVE_BANDS`](/phonometry/reference/api/materials/rating/#octave_bands). |

**Returns:** An [`HvacSpectrumResult`](/phonometry/reference/api/noise_control/hvac/#hvacspectrumresult) of the attenuation, dB.

## fan_efficiency_correction

```python
fan_efficiency_correction(*, relative_efficiency_percent: float) -> float
```

Off-peak efficiency correction `C_EFF` (Long Table 13.6).

A fan running away from its peak static efficiency is noisier at the same
duty. The correction is a step function of the static efficiency expressed
as a percentage of the peak (Long Eq. 13.3): 90 per cent of peak and above
adds nothing, and anything below 50 per cent adds 16 dB. When the peak
efficiency is unknown Long recommends assuming 80 per cent, which lands in
the 6 dB step.

**Parameters**

| Name | Description |
| :--- | :--- |
| `relative_efficiency_percent` | **ASHRAE only.** Static efficiency as a **percentage** of the peak, in `(0, 100]`. A fraction is not accepted in disguise: the table is tabulated from 50 % up, so 0,8 would fall to its bottom row and return 16 dB where 80 % returns 6, ten decibels with nothing to say it happened. Below 50 % the caller is warned that the value is outside the span Table 13.6 tabulates. |

**Returns:** The correction `C_EFF`, dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the efficiency is not in `(0, 100]`. |

## fan_sound_power

```python
fan_sound_power(
    volume_flow: float,
    *,
    fan_static_pressure_pa: float,
    model: Literal['ashrae'] = ...,
    fan_type: str = ...,
    relative_efficiency_percent: float = ...,
    blade_frequency: float | None = ...,
    frequencies: ArrayLike | None = ...,
) -> HvacSpectrumResult

fan_sound_power(
    volume_flow: float,
    *,
    model: Literal['vdi2081'],
    fan_total_pressure_pa: float,
    assembly: str,
    fan_speed_rpm: float,
    specific_sound_power_level: float | None = ...,
    blade_count: int | None = ...,
    relative_flow: float = ...,
    frequencies: ArrayLike | None = ...,
) -> HvacSpectrumResult
```

Octave-band fan sound power from the operating point, by either method.

Two schools of calculation answer the same question and do not agree on
how. `model="ashrae"` (the default) is the scaling law below;
`model="vdi2081"` is the German method, described after it. Each takes
the arguments its own standard is written on, so neither can be handed the
other's: the ASHRAE law scales the **static** pressure, VDI 2081 the
**total** pressure rise, and confusing them is worth 20 log of the ratio.

The ASHRAE (1987) scaling law, originally due to Beranek and published by
Graham (1975):

$$
L_W = K_\mathrm{F} + 10 \log_{10}(Q_\mathrm{F} / Q_\mathrm{REF}) + 10 \log_{10}(P_\mathrm{F} / P_\mathrm{REF}) + C_{EFF} + C_{BFI}
$$

with the spectral constant `K_\mathrm{F}` of Long Table 13.5 (one row per fan
type), the off-peak efficiency correction `C_EFF` of Table 13.6
([`fan_efficiency_correction`](/phonometry/reference/api/noise_control/hvac/#fan_efficiency_correction)) and the blade frequency increment
`C_BFI` of Table 13.7, added to the single octave band that contains the
blade passing frequency. In SI the reference volume flow is
$Q_\mathrm{REF} = 0.472$ L/s and the reference pressure
$P_\mathrm{REF} = 249$ Pa, so the two logarithmic terms take the same
values as the foot-pound form in cfm and inches of water gauge.

The law assumes ideal inlet and outlet flow conditions and gives the power
radiated into the duct; the fan radiates the same power from its intake and
from its discharge. Manufacturer data measured to AMCA 300 should be
preferred wherever it exists: this model is the early-design fallback, and
ASHRAE's own current guidance (2019 *HVAC Applications Handbook*, Ch. 49)
is that a fan's sound power "is best obtained from manufacturers' test data"
to AMCA Standard 300 or ASHRAE Standard 68. Long's worked sheet (Table 14.9)
prints a forward-curved row that this equation does not reproduce; see the
module warning.

**VDI 2081 Part 1:2001-07, Section 4.3.** The German method describes a fan
by its assembly type rather than by a per-type band table:

$$
L_{W4} = L_\mathrm{WSM} + 10 \log_{10} \dot{V} + 20 \log_{10} \Delta p_\mathrm{t}
$$

$$
\Delta L_{W,\mathrm{oct}} = -5 - 5 \left( \log_{10} St + c_3 \right)^{2}, \qquad St = f \, 60 / (\pi n)
$$

Equation (13) and Equation (15). The factor 20 on the pressure is the
general $5(\gamma - 1)$ of Equation (11) with the Mach number
exponent taken as 5, which Section 4.3.2 does for every ventilation fan.
The Strouhal number carries no diameter: it cancels between the tip speed
and the impeller circumference, so the impeller size a nomogram gives is
not an input here. Section 4.3.3 sets the specific level and the spectral
parameter for each of the three assemblies of VDI 3731 Part 2, and
Figures 13 and 14 add a cubic allowance for running away from the best
duty point, worth 0,1 dB at the optimum itself.

**Parameters**

| Name | Description |
| :--- | :--- |
| `volume_flow` | Volume flow through the fan `Q_\mathrm{F}`, m3/s. |
| `fan_static_pressure_pa` | **ASHRAE only.** Fan static pressure `P_\mathrm{F}`, in **pascals gauge**. This is the pressure rise the fan produces across itself, not an ambient pressure, and it shares neither the unit nor the datum of the `static_pressure` the ISO 3740 family takes in kilopascals absolute. No plausibility guard can separate the two: 101,325 Pa is a legitimate duty for a panel or propeller fan, so the name is what keeps them apart. |
| `fan_type` | **ASHRAE only.** One of `"airfoil_large"` / `"airfoil_small"` (backward-curved or backward-inclined centrifugal wheels above and below 36 in diameter), `"forward_curved"`, `"radial_low"` / `"radial_medium"` / `"radial_high"` (radial blades by total pressure), `"vaneaxial_hub_low"` / `"vaneaxial_hub_medium"` / `"vaneaxial_hub_high"` (hub ratios 0.3-0.4, 0.4-0.6 and 0.6-0.8), `"tubeaxial_large"` / `"tubeaxial_small"` (above and below 40 in wheel diameter) or `"propeller"`. |
| `relative_efficiency_percent` | Static efficiency as a **percentage** of the peak (default 80, Long's recommendation when the peak is unknown). Table 13.6 is tabulated from 50 % up, so a fraction such as 0,8 falls through to the table's bottom row and returns its worst-case 16 dB correction instead of the 6 dB that 80 % earns. That is what [`HvacWarning`](/phonometry/reference/api/noise_control/hvac/#hvacwarning) says when it fires below the floor. |
| `blade_frequency` | **ASHRAE only.** Blade passing frequency `f_bp`, Hz (from [`blade_passing_frequency`](/phonometry/reference/api/noise_control/hvac/#blade_passing_frequency)). `None` (default) places the increment in the octave band Table 13.7 tabulates for the fan type. |
| `model` | `"ashrae"` (default) or `"vdi2081"`. |
| `fan_total_pressure_pa` | **VDI 2081 only.** Total pressure rise `\Delta p_\mathrm{t}` across the fan, Pa. Not the static pressure of the ASHRAE law: the total pressure carries the dynamic head as well, and Equation (13) scales it by 20 rather than by 10. |
| `assembly` | **VDI 2081 only.** `"rr"` (radial, rearwards curved blades), `"t"` (cylindrical rotor, forwards curved blades) or `"am"` (axial with a downstream diffuser). |
| `fan_speed_rpm` | **VDI 2081 only.** Impeller speed `n`, min^-1. |
| `specific_sound_power_level` | **VDI 2081 only.** The specific sound power level `L_\mathrm{WSM}`, dB. `None` (default) takes the representative value of the assembly, 34, 36 or 42 dB. Section 4.3.3 says a fan can sit up to 7 dB above its assembly average at the optimum duty point, so a manufacturer's own value belongs here. |
| `blade_count` | **VDI 2081 only.** Number of impeller blades `z`, which places the blade-frequency allowance of Section 4.3.4 in the octave holding `n z / 60`. `None` (default) omits it. The allowance is nought for assemblies RR and T built to the state of the art and 4 dB for AM. |
| `relative_flow` | **VDI 2081 only.** Duty as a fraction of the best efficiency point, `\dot{V} / \dot{V}_\mathrm{opt}` (default 1). |
| `frequencies` | Octave-band centres, Hz; `None` (default) uses the 63 Hz to 8 kHz bands of [`OCTAVE_BANDS`](/phonometry/reference/api/materials/rating/#octave_bands). |

**Returns:** An [`HvacSpectrumResult`](/phonometry/reference/api/noise_control/hvac/#hvacspectrumresult) of the band sound power level, dB re 1e-12 W.

## flexible_duct_insertion_loss

```python
flexible_duct_insertion_loss(
    frequencies: ArrayLike | None,
    diameter: float,
    length: float,
) -> HvacSpectrumResult
```

Insertion loss of a lined round flexible duct (Long Table 14.4, ASHRAE 1995).

The last run of a supply branch is usually flexible duct: a fabric liner
inside a lightweight fibreglass fill inside a plastic membrane. Its
published insertion loss is remarkably high, 2 to 3 dB per foot in the mid
bands, partly because the test replaces a length of sheet-metal duct and so
credits the flexible duct's breakout as well as its dissipation. That same
property makes a serpentine run of flexible duct in an attic or a joist
space work as an improvised breakout silencer. The table is interpolated
linearly over length and over log diameter; it stops at 4 kHz, so no 8 kHz
value is returned.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Octave-band centres, Hz, within 63 Hz to 4 kHz; `None` uses all seven tabulated bands. |
| `diameter` | Internal diameter, m (100 mm to 406 mm tabulated). |
| `length` | Duct run length, m (0.9 m to 3.7 m tabulated). |

**Returns:** An [`HvacSpectrumResult`](/phonometry/reference/api/noise_control/hvac/#hvacspectrumresult) of the insertion loss, dB.

## flow_noise_bend

```python
flow_noise_bend(
    frequencies: ArrayLike,
    flow_velocity: float,
    area: float,
    height: float,
    *,
    density: float = 1.206,
) -> HvacSpectrumResult
```

Flow-generated octave-band sound power of a mitred bend (Bies Eqs. (8.252), (8.254)).

$$
L_{W\mathrm{B}} = L_{W\mathrm{s}} - 10 \log_{10}(1 + 0.165 N_\mathrm{s}^2) + 30 \log_{10}(U) - 103
$$

$$
L_{W\mathrm{s}} = 30 \log_{10}(U) + 10 \log_{10}(S) + 10 \log_{10}(\rho) + 117
$$

with the stream power level $L_{W\mathrm{s}}$ (Bies Eq. (8.252)) and the
Strouhal number $N_\mathrm{s} = f H / U$ (`H` the duct
height in the plane of the bend). The radiated sound power grows as the
sixth power of the stream speed at low `N_\mathrm{s}` (the inner-corner drag
dipole) and the eighth power at high `N_\mathrm{s}` (the outer-corner shear
quadrupole); equivalently, the *efficiency* referenced to the stream power
grows as $U^3$ and $U^5$ respectively.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Octave-band centre frequencies `f`, Hz (1-D array). |
| `flow_velocity` | Mean flow speed `U`, m/s. |
| `area` | Duct cross-sectional area `S`, m2. |
| `height` | Duct height `H` in the plane of the bend, m. |
| `density` | Air density `rho`, kg/m3. |

**Returns:** A [`HvacSpectrumResult`](/phonometry/reference/api/noise_control/hvac/#hvacspectrumresult) of the band sound power level, dB re 1e-12 W.

## flow_noise_straight_duct

```python
flow_noise_straight_duct(
    frequencies: ArrayLike,
    flow_velocity: float,
    area: float,
) -> HvacSpectrumResult
```

Flow-generated octave-band sound power of a straight duct (Bies Eq. (8.251)).

$$
L_{W\mathrm{B}} = 7 + 50 \log_{10}(U) + 10 \log_{10}(S) - 2 - 26 \log_{10}(1.14 + 0.02 f / U)
$$

in dB re 1e-12 W (VDI 2081-1), for airflow speed `U` in a duct of area
`S`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Octave-band centre frequencies `f`, Hz (1-D array). |
| `flow_velocity` | Mean flow speed `U`, m/s. |
| `area` | Duct cross-sectional area `S`, m2. |

**Returns:** A [`HvacSpectrumResult`](/phonometry/reference/api/noise_control/hvac/#hvacspectrumresult) of the band sound power level, dB re 1e-12 W.

## HvacSpectrumResult

```python
HvacSpectrumResult(
    frequencies: np.ndarray,
    values: np.ndarray,
    quantity: HvacQuantity,
    label: str,
)
```

A per-frequency HVAC quantity (attenuation or regenerated power level).

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Frequencies `f`, Hz. |
| `values` | The quantity per frequency (dB, or dB re 1e-12 W for a sound power level). |
| `quantity` | What `values` holds (`"attenuation"` or `"sound_power_level"`). |
| `label` | A short human label of the element. |

### HvacSpectrumResult.plot()

```python
HvacSpectrumResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the quantity against a continuous log-frequency axis.

Requires matplotlib (`pip install phonometry[plot]`).

### HvacSpectrumResult.report()

```python
HvacSpectrumResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render an HVAC duct-noise-spectrum fiche to `path`.

Writes a one-page HVAC-noise sheet: the method-basis line naming the
reported quantity and the Bies, Hansen & Howard chapter (Engineering
Noise Control 5th ed., Chapter 8), an optional metadata header (client,
duct element, test environment, instrumentation, climate, date), a
per-band table (nominal frequency and the reported quantity) beside the
spectrum, the boxed single-number result (for a regenerated-noise
spectrum the A-weighted sound power level `L_WA` re 1 pW with the
overall unweighted total; for an attenuation spectrum the mean
attenuation with its band range), an optional verdict row against a
declared limit, and a method-basis strip stating the reported quantity's
relation.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata) supplying the header (`client`, `specimen` the duct element, `test_room` the test environment, `instrumentation`, `temperature`, `relative_humidity`, `pressure`, `test_date`), the footer identity (`laboratory`, `operator`, `report_id`, `notes`) and, via `requirement`, a declared maximum A-weighted sound power level for a regenerated-noise spectrum (lower is better) or a declared minimum mean attenuation for an attenuation spectrum (more is better). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When `True` a regenerated-noise table adds the A-weighting correction and the A-weighted band level columns. |
| `language` | Fiche language: `"en"` (default) or `"es"`. |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"` or `language` is unknown. |
| ImportError | If reportlab (or, for the figure, matplotlib) is not installed (`pip install phonometry[report]`). |

## HvacWarning

An HVAC input outside the span the table it feeds was tabulated from.

## lined_circular_duct_attenuation

```python
lined_circular_duct_attenuation(
    frequencies: ArrayLike | None,
    diameter: float,
    length: float,
    lining_thickness: float,
) -> HvacSpectrumResult
```

Insertion loss of a lined circular duct (Long Eq. 14.13, Table 14.3).

The Reynolds (1990) third-order regression
$R = (A + B t + C t^2 + D d + E d^2 + F d^3) l$, with the lining
thickness `t` and the internal diameter `d` in inches and the length
`l` in feet. It was developed for spiral ducts with a 12 kg/m3 fibreglass
lining 25 mm to 76 mm thick behind a 25 per cent open perforated facing,
over internal diameters from 150 mm to 1.5 m. Negative regression values
are clipped to zero and, as for rectangular ducts, flanking limits the run
to 40 dB. The unlined contribution is so small for circular ducts that Long
ignores it.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Octave-band centres, Hz; `None` uses [`OCTAVE_BANDS`](/phonometry/reference/api/materials/rating/#octave_bands). |
| `diameter` | Internal diameter `d`, m. |
| `length` | Duct run length `l`, m. |
| `lining_thickness` | Lining thickness `t`, m. |

**Returns:** An [`HvacSpectrumResult`](/phonometry/reference/api/noise_control/hvac/#hvacspectrumresult) of the attenuation, dB.

## lined_rectangular_duct_attenuation

```python
lined_rectangular_duct_attenuation(
    frequencies: ArrayLike | None,
    width: float,
    height: float,
    length: float,
    lining_thickness: float,
    *,
    include_unlined: bool = False,
) -> HvacSpectrumResult
```

Insertion loss of a lined rectangular duct (Long Eq. 14.12, Table 14.2).

The Reynolds (1990) regression $R = B (P/S)^C t^D l$, with the duct
perimeter `P` in feet, its area `S` in square feet, the lining
thickness `t` in inches and the run length `l` in feet. It was fitted
to 25 mm to 52 mm linings of 24 to 48 kg/m3 density over `P / S` from
1.1667 to 6 ft^-1; linings thinner than 25 mm are generally ineffective.
The insertion loss is measured by substituting the lined section for an
unlined one of the same face size, so the unlined attenuation may be added
on top (`include_unlined=True`, which Long recommends for rectangular
ducts). Flanking limits the total to 40 dB.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Octave-band centres, Hz; `None` uses [`OCTAVE_BANDS`](/phonometry/reference/api/materials/rating/#octave_bands). |
| `width` | Duct width, m. |
| `height` | Duct height, m. |
| `length` | Duct run length `l`, m. |
| `lining_thickness` | Lining thickness `t`, m. |
| `include_unlined` | Add the unlined-duct attenuation of [`unlined_rectangular_duct_attenuation`](/phonometry/reference/api/noise_control/hvac/#unlined_rectangular_duct_attenuation), the side-wall contribution the insertion-loss measurement subtracts out. |

**Returns:** An [`HvacSpectrumResult`](/phonometry/reference/api/noise_control/hvac/#hvacspectrumresult) of the attenuation, dB.

## plenum_attenuation

```python
plenum_attenuation(
    exit_area: float,
    line_of_sight: float,
    wall_area: float,
    mean_absorption: ArrayLike,
    *,
    angle: float = 0.0,
) -> np.ndarray | float
```

Plenum-chamber transmission loss by Wells' method (Bies Eq. (8.275)).

$$
\mathrm{TL} = -10 \log_{10}\!\left[S_{\mathrm{out}} \left(\frac{\cos(\theta)}{\pi r^2} + \frac{1 - \alpha}{S_\mathrm{w} \alpha}\right)\right],
$$

where the reverberant term uses the plenum room constant
$R = S_\mathrm{w} \alpha / (1 - \alpha)$
([`phonometry.room.room_constant`](/phonometry/reference/api/rooms/steady-field/#room_constant)). The
method holds above the inlet cut-on and when the plenum is large compared
with the wavelength; it underpredicts the low-frequency loss by 5-10 dB.

**Parameters**

| Name | Description |
| :--- | :--- |
| `exit_area` | Outlet-opening area `S_out`, m2. |
| `line_of_sight` | Straight-line inlet-to-outlet distance `r`, m. |
| `wall_area` | Total internal wall area `S_\mathrm{w}`, m2. |
| `mean_absorption` | Mean Sabine wall absorption `alpha` in `(0, 1)` (scalar or per-band). |
| `angle` | Angle `theta` between the inlet axis and the line to the outlet, in `[0, pi/2]` rad (default 0). |

**Returns:** The transmission loss, dB (float for scalar absorption, else a per-band array).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a dimension is not positive, `mean_absorption` leaves `(0, 1)` or `angle` leaves `[0, pi/2]`. |

## plot_plenum_geometry

```python
plot_plenum_geometry(
    exit_area: float,
    line_of_sight: float,
    wall_area: float,
    ax: Axes | None = None,
    *,
    angle: float = 0.0,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Draw the plenum-chamber section honouring the acoustic geometry.

The two truly geometric parameters of
[`plenum_attenuation`](/phonometry/reference/api/noise_control/hvac/#plenum_attenuation) are drawn exactly:
the inlet-to-outlet line of sight `r` and its `angle` off the inlet
axis fix the box; the exit area sets the drawn outlet mouth (square-duct
side `sqrt(S_out)`) and the wall area is annotated.

**Parameters**

| Name | Description |
| :--- | :--- |
| `exit_area` | Outlet area `S_out`, in m2. |
| `line_of_sight` | Inlet-to-outlet distance `r`, in metres. |
| `wall_area` | Total internal wall area `S_w`, in m2 (annotation). |
| `ax` | Existing axes, or `None` to create a figure. |
| `angle` | Angle between the inlet axis and the line of sight, in radians (0 \<= angle \< pi/2). |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the wall-segment `plot` calls (line properties such as `linewidth` or `color`). |

**Returns:** The axes.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | naming the first of the three that is not finite and positive, or for an angle outside `[0, pi/2)`. |

## room_effect

```python
room_effect(
    distance: float,
    room_constant: ArrayLike,
    *,
    directivity: float = 2.0,
) -> np.ndarray | float
```

Room effect: the drop from the terminal sound power to the room level.

The last step of a duct-path calculation turns the sound power arriving at
the terminal device into a sound pressure level at the listener, through
the steady-state room relation
$L_p = L_W + 10 \log_{10}[Q / (4 \pi r^2) + 4 / R]$ (Long Eq. 14.40; Bies
Eq. (6.43), [`phonometry.room.steady_state_spl`](/phonometry/reference/api/rooms/steady-field/#steady_state_spl)). This function
returns the *attenuation*, the positive number
$-10 \log_{10}[Q / (4 \pi r^2) + 4 / R]$, so it drops into a duct-path
cascade beside every other loss; Long's worked sheets print it as the
negative level change. A ceiling diffuser radiates into a half space,
hence the default $Q = 2$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `distance` | Terminal-to-listener distance `r`, m. |
| `room_constant` | Room constant $R = S \alpha / (1 - \alpha)$, m2 (scalar or per-band; from [`phonometry.room.room_constant`](/phonometry/reference/api/rooms/steady-field/#room_constant)). |
| `directivity` | Directivity factor `Q` of the terminal device (`2` flush in a ceiling or wall, `4` at an edge, `8` in a corner). |

**Returns:** The room effect as a positive attenuation, dB (a float for a scalar room constant, otherwise a per-band array).

## silencer_self_noise

```python
silencer_self_noise(
    frequencies: ArrayLike | None,
    airway_velocity: float,
    passages: int,
    height: float,
) -> HvacSpectrumResult
```

Regenerated (self) noise of a splitter silencer (Long Eq. 14.31).

Fry's (1988) estimate, for when manufacturer self-noise data is not
available:

$$
L_W = 55 \log_{10}(V / V_0) + 10 \log_{10} N + 10 \log_{10}(H / H_0) - 45
$$

with `V` the velocity in the splitter airway ($V_0 = 1$ m/s),
`N` the number of air passages and `H` the silencer height or, for a
round unit, its circumference ($H_0 = 1$ mm). The octave-band
spectrum follows by subtracting the corrections of Table 14.8, which
fall steeply above 500 Hz.

The fifth-and-a-half power of the airway velocity is the practical message:
doubling the face velocity of a silencer adds about 17 dB, which is how a
silencer ends up *making* the noise it was bought to remove.

Manufacturer self-noise data is measured on a 600 x 600 mm face, so a
published spectrum has to be corrected by $10 \log_{10}(S / S_0)$ for
the actual face area before it is used; this estimate needs no such
correction because the face size enters through `N` and `H`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Octave-band centres, Hz; `None` uses [`OCTAVE_BANDS`](/phonometry/reference/api/materials/rating/#octave_bands). |
| `airway_velocity` | Velocity `V` in the splitter airway, m/s. |
| `passages` | Number of air passages `N` between the splitters. |
| `height` | Silencer height `H` (or circumference, if round), m. |

**Returns:** An [`HvacSpectrumResult`](/phonometry/reference/api/noise_control/hvac/#hvacspectrumresult) of the band sound power level, dB re 1e-12 W.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `passages` is not a positive integer. |

## split_loss

```python
split_loss(
    main_area: float,
    branch_areas: ArrayLike,
    *,
    branch: int = 0,
) -> float
```

Power split loss into one branch of a duct division (Long Eq. 14.17).

Where a duct divides, the sound power is shared between the branches in
proportion to their areas, and a further reflection occurs when the total
branch area does not match the feeder area:

$$
R = -10 \log_{10}\!\left[ 1 - \left( \frac{\sum S_i - S_m}{\sum S_i + S_m} \right)^2 \right] - 10 \log_{10}\!\left( \frac{S_i}{\sum S_i} \right)
$$

Long prints this as a negative level change (a 25 per cent area split shows
as -6 dB in his worked sheet); this function returns it as a positive
attenuation, like every other loss in the module.

**Parameters**

| Name | Description |
| :--- | :--- |
| `main_area` | Cross-sectional area of the main feeder duct `S_m`, m2. |
| `branch_areas` | Areas `S_i` of the branches continuing on from the main duct, m2 (1-D array-like). |
| `branch` | Index into `branch_areas` of the branch being followed. |

**Returns:** The split loss, dB (positive).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the areas are not positive or `branch` is out of range. |

## splitter_silencer_insertion_loss

```python
splitter_silencer_insertion_loss(
    frequencies: ArrayLike | None,
    height: float,
    length: float,
    airway_widths: ArrayLike,
    splitter_thickness: float,
) -> HvacSpectrumResult
```

Insertion loss of a parallel-splitter (dissipative) silencer.

A splitter silencer divides the duct into parallel airways separated by
absorbent baffles. Bies, Hansen & Howard (§8.10.5) reduce it to a lined
duct: each airway is calculated *as a lined duct whose liner thickness is
half the splitter thickness*, because each face of a splitter lines the
airway beside it, and the insertion losses of the airways combine as

$$
\mathrm{IL}_{tot} = -10 \log_{10}\!\left[ \frac{1}{N} \sum_i 10^{-\mathrm{IL}_i / 10} \right] \tag{8.241}
$$

which is the energy average over the airways: when they are identical the
total equals the loss of a single passage, and when they differ the leakiest
airway dominates, exactly as a real unit does. The airway loss itself comes
from the Reynolds (1990) lined-rectangular-duct regression of
[`lined_rectangular_duct_attenuation`](/phonometry/reference/api/noise_control/hvac/#lined_rectangular_duct_attenuation) (Long Eq. 14.12), so the same
validity envelope applies: linings of 25 mm to 52 mm at 24 to 48 kg/m3 and
a perimeter-to-area ratio of the airway between 1.1667 and 6 ft^-1.

Published dynamic insertion loss (DIL) from the silencer manufacturer,
measured with the design airflow and in the design direction, should be
preferred wherever it exists; this estimate is the early-design fallback and
ignores the entrance and exit losses of the unit. The unit's regenerated
noise is a separate quantity, [`silencer_self_noise`](/phonometry/reference/api/noise_control/hvac/#silencer_self_noise).

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Octave-band centres, Hz; `None` (default) uses [`OCTAVE_BANDS`](/phonometry/reference/api/materials/rating/#octave_bands). |
| `height` | Height of the silencer face, m (the airway dimension the splitters do not divide). |
| `length` | Length of the silencer in the flow direction, m. |
| `airway_widths` | Free width of each airway between splitters, m (a scalar is taken as a single airway; give one value per airway when they differ). |
| `splitter_thickness` | Full thickness of a splitter baffle, m; the equivalent liner thickness of an airway is half of it. |

**Returns:** An [`HvacSpectrumResult`](/phonometry/reference/api/noise_control/hvac/#hvacspectrumresult) of the insertion loss, dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If any dimension is not positive, or if `airway_widths` is not a non-empty 1-D array. |

## unlined_circular_duct_attenuation

```python
unlined_circular_duct_attenuation(
    frequencies: ArrayLike | None,
    length: float,
) -> HvacSpectrumResult
```

Attenuation of an unlined circular sheet-metal duct (Long Table 14.1).

A circular duct is far stiffer than a rectangular one in its breathing
mode, so the sound field can hardly excite it: the loss is about a tenth of
the rectangular value and is tabulated as a length rate alone, 0.03 dB/ft
up to 250 Hz and 0.05 to 0.07 dB/ft above. The published table stops at
4 kHz; the 4 kHz rate is held for the 8 kHz band.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Octave-band centres, Hz; `None` uses [`OCTAVE_BANDS`](/phonometry/reference/api/materials/rating/#octave_bands). |
| `length` | Duct run length, m. |

**Returns:** An [`HvacSpectrumResult`](/phonometry/reference/api/noise_control/hvac/#hvacspectrumresult) of the attenuation, dB.

## unlined_rectangular_duct_attenuation

```python
unlined_rectangular_duct_attenuation(
    frequencies: ArrayLike,
    width: float,
    height: float,
    length: float,
    *,
    wrapped: bool = False,
) -> HvacSpectrumResult
```

Attenuation of an unlined rectangular sheet-metal duct (Long Eqs. 14.9-14.11).

Sound running down an unlined duct loses energy into the induced motion of
the duct walls, so the loss grows with the perimeter-to-area ratio `P / S`
(a wide, shallow duct has floppier side walls). Reynolds (1990) fits the
63 Hz to 250 Hz bands with $R = 17.0 (P/S)^{0.25} f^{-0.85} l$ for
$P/S \ge 3$ ft^-1 and $R = 1.64 (P/S)^{0.73} f^{-0.58} l$
below it, and
everything above 250 Hz with $R = 0.02 (P/S)^{0.8} l$. An external
fibreglass blanket adds surface mass and doubles the low-frequency loss
(`wrapped=True`).

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Octave-band centre frequencies `f`, Hz (1-D array). |
| `width` | Duct width, m. |
| `height` | Duct height, m. |
| `length` | Duct run length `l`, m. |
| `wrapped` | The duct is externally wrapped with a fibreglass blanket, which doubles the 63 Hz to 250 Hz attenuation. |

**Returns:** An [`HvacSpectrumResult`](/phonometry/reference/api/noise_control/hvac/#hvacspectrumresult) of the attenuation, dB.
