---
title: "building.panel_transmission"
description: "Predicted airborne sound reduction index of panels (Bies, Hansen & Howard 2017, Engineering Noise Control 5e, Section 7.2; Sharp 1973)."
sidebar:
  label: "panel_transmission"
---

Predicted airborne sound reduction index of panels (Bies, Hansen & Howard
2017, Engineering Noise Control 5e, Section 7.2; Sharp 1973).

Where EN 12354-1 ([`phonometry.building.building_prediction`](/phonometry/reference/api/building/building-prediction/)) takes the
element sound reduction index `R` as a *measured* input, this module
**predicts** `R(f)` from the physical properties of the construction: the mass
per unit area, bending stiffness (through the coincidence frequency) and loss
factor. The prediction feeds the same ISO 717-1 weighting
([`phonometry.weighted_rating`](/phonometry/reference/api/building/insulation/#weighted_rating)) as the measured quantities, closing the
chain from panel physics to the single-number `Rw`.

**Mass law (Bies Eq. 7.40/7.42).** A non-stiff panel transmits by forced motion;
the transmission coefficient of an infinite limp panel gives the normal- and
field-incidence transmission loss:

$$
\mathrm{TL}_{\mathrm{normal}} = 10 \lg\!\left[ 1 + \left( \frac{\pi f m''}{\rho_0 c_0} \right)^{2} \right]
$$

$$
\mathrm{TL}_{\mathrm{field}} = \mathrm{TL}_{\mathrm{normal}} - \mathrm{dB}(\text{band})
$$

with `m''` the mass per unit area, $\rho_0 c_0$ the characteristic
impedance of air and the field-incidence correction
$\mathrm{dB} = 5.5$ dB for one-third-octave or `4.0` dB for octave
bands (Eq. 7.42). The mass law rises 6 dB per octave and 6 dB per doubling of
mass.

**Single panel, Sharp's method (Bies 7.2.4.1).** Below the coincidence region
the field-incidence mass law holds; from the coincidence frequency `fc`
upwards the loss factor `eta` controls the transmission (Eq. 7.44):

$$
\mathrm{TL} = 10 \lg\!\left[ 1 + \left( \frac{\pi f m''}{\rho_0 c_0} \right)^{2} \right] + 10 \lg\frac{2 \eta f}{\pi f_c}
$$

and between $f_c/2$ and $f_c$ the curve is a straight line on
$\mathrm{TL}$ versus $\log_{10} f$. The coincidence dip at
$f_c$ sits $10 \lg(2\eta/\pi)$ below the extrapolated mass law
(Bies design-chart point B,
$\mathrm{TL} = 20 \lg(f_c m'') + 10 \lg\eta - 44$).

**Double wall (Bies 7.2.6, Eq. 7.62-7.64).** Two leaves `m1`, `m2` separated
by a gap `d` behave as a mass-spring-mass system. Below the resonance
$f_0 = \frac{1}{2\pi} \sqrt{s'' (m_1 + m_2)/(m_1 m_2)}$ the pair
follows the mass law of the combined mass $m_1 + m_2$; above it the two
mass laws add, boosted by the cavity (Eq. 7.64):

$$
\mathrm{TL} = \mathrm{TL}_M, \qquad f \le f_0
$$

$$
\mathrm{TL} = \mathrm{TL}_1 + \mathrm{TL}_2 + 20 \lg(2 k d), \qquad f_0 < f < f_l, \quad k = 2 \pi f / c_0
$$

$$
\mathrm{TL} = \mathrm{TL}_1 + \mathrm{TL}_2 + 6, \qquad f \ge f_l = \frac{c_0}{2 \pi d}
$$

The cavity stiffness `s''` is $\rho_0 c_0^{2} / d$ for an empty
(adiabatic) air gap; a porous fill (a
[`PorousMediumResult`](/phonometry/reference/api/materials/porous-absorber/#porousmediumresult) from
[`phonometry.materials.porous_absorber`](/phonometry/reference/api/materials/porous-absorber/)) lowers the resonance through its
softer, near-isothermal effective bulk modulus and damps the cavity so the
mid-band slope is realised without standing-wave dips.

**Orthotropic panels (Bies 7.2.4.5; Vigran, Building Acoustics, 3.7.3 and
6.5.3).** Ribbed and corrugated cladding is stiff along the corrugations and
limp across them, so a single coincidence frequency no longer exists: the panel
has a *range* $f_{c1} \le f \le f_{c2}$ bounded by the stiffest and the
least stiff direction (Vigran Eq. 6.107). The bending-wave impedance then
depends on the azimuth `theta` as well as the incidence angle `phi`
(Heckl 1960; Hansen
1993; Vigran Eq. 6.108 = Bies Eq. 7.30), and the diffuse-field average is a
double integral (Vigran Eq. 6.111 = Bies Eq. 7.38). The consequence is the
whole point of the model: over one to two decades the resonant transmission
dominates and `R` flattens far below the mass law of a flat plate of the same
mass. See [`orthotropic_transmission_loss`](/phonometry/reference/api/building/panel-transmission/#orthotropic_transmission_loss),
[`orthotropic_critical_frequencies`](/phonometry/reference/api/building/panel-transmission/#orthotropic_critical_frequencies), [`corrugated_plate_stiffness`](/phonometry/reference/api/building/panel-transmission/#corrugated_plate_stiffness)
and [`orthotropic_plate_resonance`](/phonometry/reference/api/building/panel-transmission/#orthotropic_plate_resonance).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## corrugated_plate_mass_factor

```python
corrugated_plate_mass_factor(
    corrugation_amplitude: float,
    corrugation_wavelength: float,
) -> float
```

Surface-density increase of a sine-corrugated plate.

Corrugating a sheet does not change its thickness, so its mass per unit
area grows in proportion to the **developed** length of the profile.
For a sinusoid of amplitude `H` and wavelength `L` the developed length
per period, divided by the period, is the closed form

$$
\frac{m''}{m''_{\mathrm{flat}}} = \frac{2}{\pi} \sqrt{1 + q^{2}}\, E\!\left( \frac{q^{2}}{1 + q^{2}} \right), \qquad q = \frac{2 \pi H}{L}
$$

with `E` the complete elliptic integral of the second kind. Vigran, in
the worked example following his Eq. (3.115) (printed p. 96), warns that
"we have to take into account the fact that the mass per unit area will
increase when making the corrugations", and it is exactly this factor that
reproduces his published eigenfrequencies.

**Parameters**

| Name | Description |
| :--- | :--- |
| `corrugation_amplitude` | Corrugation amplitude `H`, in m (> 0); the total peak-to-trough depth of the profile is $2H$. |
| `corrugation_wavelength` | Corrugation wavelength `L`, in m (> 0). |

**Returns:** The factor (>= 1) multiplying the flat-sheet surface density.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive input. |

## corrugated_plate_stiffness

```python
corrugated_plate_stiffness(
    thickness: float,
    corrugation_amplitude: float,
    corrugation_wavelength: float,
    *,
    youngs_modulus: float,
    poisson_ratio: float = 0.3,
) -> tuple[float, float, float]
```

Equivalent orthotropic stiffnesses of a "wavy" corrugated plate.

Timoshenko & Woinowsky-Krieger's (1959) equivalent bending stiffnesses of a
plate of thickness `h` whose profile is a sinusoid of amplitude `H` and
wavelength `L`, as transcribed by Vigran Eq. (3.115) (printed p. 96):

$$
B_x = \frac{E h^{3}}{12 (1 - \nu^{2}) \left[ 1 + (\pi H / L)^{2} \right]}
$$

$$
B_z = \frac{E H^{2} h}{2} \left[ 1 - \frac{0.81}{1 + 2.5\,(H/L)^{2}} \right]
$$

$$
B_{xz} = \frac{E h^{3}}{12 (1 + \nu)} \left[ 1 + (\pi H / L)^{2} \right]
$$

`Bx` is the stiffness **across** the corrugations (slightly *below* the
flat-plate value), `Bz` the stiffness **along** them (larger by orders of
magnitude: that is what corrugating buys) and `Bxz` the twisting term
Eq. (3.113) needs. Vigran's footnote records that the same equations appear
in Blevins (1979) "unfortunately, with a misprint in the expression for
`Bz`".

Feed `(Bx, Bz)` to [`orthotropic_critical_frequencies`](/phonometry/reference/api/building/panel-transmission/#orthotropic_critical_frequencies) for the
coincidence range and all three to [`orthotropic_plate_resonance`](/phonometry/reference/api/building/panel-transmission/#orthotropic_plate_resonance) for
the eigenfrequencies. Remember to scale the surface density by
[`corrugated_plate_mass_factor`](/phonometry/reference/api/building/panel-transmission/#corrugated_plate_mass_factor).

**Parameters**

| Name | Description |
| :--- | :--- |
| `thickness` | Sheet thickness `h`, in m (> 0). |
| `corrugation_amplitude` | Corrugation amplitude `H`, in m (> 0). |
| `corrugation_wavelength` | Corrugation wavelength `L`, in m (> 0). |
| `youngs_modulus` | Young's modulus `E`, in Pa (> 0). |
| `poisson_ratio` | Poisson's ratio `nu` (Default: 0.3). |

**Returns:** The triple `(Bx, Bz, Bxz)` in N.m.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive input or $\lvert\nu\rvert \ge 1$. |

## double_wall_transmission_loss

```python
double_wall_transmission_loss(
    frequency: ArrayLike,
    mass1: float,
    mass2: float,
    gap: float,
    *,
    loss_factor: float = 0.1,
    cavity_medium: PorousMediumResult | None = None,
    tie_stiffness_per_area: float = 0.0,
    band: str = 'third',
    speed_of_sound: float = 343.0,
    air_density: float = 1.205,
) -> SoundReductionResult
```

Sound reduction index of a double wall (Bies 7.2.6, Eq. 7.64).

Piecewise Sharp model: below the mass-spring-mass resonance `f0` the pair
behaves as the mass law of the combined mass; between `f0` and the
limiting frequency $f_l = c_0/(2 \pi d)$ the two mass laws add plus
$20 \lg(2 k d)$; above `f_l` they add plus 6 dB. The curve is
continuous at `f_l` ($20 \lg(2 k d) = 6$ there).

Ties or mounts bridging the cavity stiffen it (Hopkins Eq. 4.89), pushing
`f0` up and extending the combined-mass branch; pass their stiffness per
unit area as *tie_stiffness_per_area* (see
[`phonometry.wall_tie_stiffness_per_area`](/phonometry/reference/api/building/masonry-cavity-wall/#wall_tie_stiffness_per_area)).

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Band centre frequencies `f`, in hertz (array, > 0). |
| `mass1` | Surface density of leaf 1 `m1`, in kg/m^2 (> 0). |
| `mass2` | Surface density of leaf 2 `m2`, in kg/m^2 (> 0). |
| `gap` | Cavity depth `d`, in m (> 0). |
| `loss_factor` | Leaf loss factor `eta` (> 0, Default: 0.1); reserved for the coincidence extension and reported for reference. |
| `cavity_medium` | Optional porous fill; see [`mass_spring_mass_resonance`](/phonometry/reference/api/building/panel-transmission/#mass_spring_mass_resonance). |
| `tie_stiffness_per_area` | Stiffness per unit area $N k / S$ of a connection array bridging the cavity, in N/m^3 (>= 0, Default: 0). |
| `band` | Band width for the field correction (`"third"`/`"octave"`). |
| `speed_of_sound` | Speed of sound in air `c0` (Default: 343 m/s). |
| `air_density` | Air density `rho0` (Default: 1.205 kg/m^3). |

**Returns:** A [`SoundReductionResult`](/phonometry/reference/api/building/panel-transmission/#soundreductionresult) (model `"double-wall"`).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive input. |

## field_incidence_correction

```python
field_incidence_correction(band: str = 'third') -> float
```

Field-incidence mass-law correction `dB` (Bies Eq. 7.42).

**Parameters**

| Name | Description |
| :--- | :--- |
| `band` | `"third"` (5.5 dB) or `"octave"` (4.0 dB). |

**Returns:** The correction subtracted from the normal-incidence mass law, dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for an unknown band width. |

## mass_law_transmission_loss

```python
mass_law_transmission_loss(
    frequency: ArrayLike,
    mass_per_area: float,
    *,
    incidence: str = 'field',
    band: str = 'third',
    field_correction: float | None = None,
    speed_of_sound: float = 343.0,
    air_density: float = 1.205,
) -> np.ndarray
```

Mass-law transmission loss of a limp panel (Bies Eq. 7.40/7.42).

$\mathrm{TL}_{\mathrm{normal}} = 10 \lg[1 + (\pi f m'' / \rho_0 c_0)^{2}]$; the field-incidence value subtracts the
band correction of [`field_incidence_correction`](/phonometry/reference/api/building/panel-transmission/#field_incidence_correction),
or the explicit *field_correction* when one is given (Norton & Karczub
Eq. 3.106 uses a flat 5 dB, the line [`plateau_transmission_loss`](/phonometry/reference/api/building/panel-transmission/#plateau_transmission_loss)
builds its estimate on).

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency `f`, in hertz (scalar or array, > 0). |
| `mass_per_area` | Mass per unit area `m''`, in kg/m^2 (> 0). |
| `incidence` | `"normal"` or `"field"` (Default: `"field"`). |
| `band` | Band width for the field correction (`"third"`/`"octave"`). |
| `field_correction` | Explicit field-incidence correction, in dB (>= 0), overriding the band table (Default: `None`). |
| `speed_of_sound` | Speed of sound in air `c0` (Default: 343 m/s). |
| `air_density` | Air density `rho0` (Default: 1.205 kg/m^3). |

**Returns:** The transmission loss `TL`, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive input or unknown incidence/band. |

## mass_spring_mass_resonance

```python
mass_spring_mass_resonance(
    mass1: float,
    mass2: float,
    gap: float,
    *,
    cavity_medium: PorousMediumResult | None = None,
    tie_stiffness_per_area: float = 0.0,
    speed_of_sound: float = 343.0,
    air_density: float = 1.205,
) -> float
```

Mass-spring-mass resonance `f0` of a double wall (Bies Eq. 7.62).

$f_0 = \frac{1}{2 \pi} \sqrt{s'' (m_1 + m_2)/(m_1 m_2)}$ with the
cavity stiffness per unit area `s''`. For an empty air gap
$s'' = \rho_0 c_0^{2} / d$ (adiabatic,
Hopkins Eq. 4.72); with a porous *cavity_medium* the fill's effective
(near-isothermal) bulk modulus at the lowest supplied frequency sets a
softer $s'' = \operatorname{Re}(K_e) / d$, lowering `f0`.

An array of mechanical connections across the cavity (wall ties in a
masonry cavity wall, resilient mounts under a floating floor) acts as a
spring **in parallel** with the cavity, adding $N k / S$ to `s''`
(Hopkins Eq. 4.89). Pass that term as *tie_stiffness_per_area*; the helper
[`phonometry.wall_tie_stiffness_per_area`](/phonometry/reference/api/building/masonry-cavity-wall/#wall_tie_stiffness_per_area) builds it from a tie density
and Hopkins' Table A4.

**Parameters**

| Name | Description |
| :--- | :--- |
| `mass1` | Surface density of leaf 1 `m1`, in kg/m^2 (> 0). |
| `mass2` | Surface density of leaf 2 `m2`, in kg/m^2 (> 0). |
| `gap` | Cavity depth `d`, in m (> 0). |
| `cavity_medium` | Optional porous fill (a [`PorousMediumResult`](/phonometry/reference/api/materials/porous-absorber/#porousmediumresult)) whose effective bulk modulus sets the cavity stiffness. |
| `tie_stiffness_per_area` | Stiffness per unit area $N k / S$ of a connection array bridging the cavity, in N/m^3 (>= 0, Default: 0). |
| `speed_of_sound` | Speed of sound in air `c0` (Default: 343 m/s). |
| `air_density` | Air density `rho0` (Default: 1.205 kg/m^3). |

**Returns:** The mass-spring-mass resonance `f0`, in hertz.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive input. |

## orthotropic_critical_frequencies

```python
orthotropic_critical_frequencies(
    mass_per_area: float,
    bending_stiffness_1: float,
    bending_stiffness_2: float,
    *,
    speed_of_sound: float = 343.0,
) -> tuple[float, float]
```

Coincidence range `(fc1, fc2)` of orthotropic panels (Vigran 6.107).

$f_c = \frac{c_0^{2}}{2 \pi} \sqrt{m'' / B}$ evaluated for both
principal bending stiffnesses (Vigran Eq. (6.107), printed p. 252; the
same closed form as the isotropic
[`coincidence_frequency`](/phonometry/reference/api/vibration/radiation-efficiency/#coincidence_frequency)).
The stiffest direction gives the **lowest** coincidence frequency, so the
returned pair is sorted: `fc1` from the larger stiffness, `fc2` from the
smaller. For a corrugated sheet `fc1` can sit at a few hundred hertz while
`fc2` reaches 15 kHz to 30 kHz, and the resonant transmission then
dominates over most of the useful frequency range.

**Parameters**

| Name | Description |
| :--- | :--- |
| `mass_per_area` | Mass per unit area `m''`, in kg/m^2 (> 0), including the developed-length increase of a corrugated sheet (see [`corrugated_plate_mass_factor`](/phonometry/reference/api/building/panel-transmission/#corrugated_plate_mass_factor)). |
| `bending_stiffness_1` | One principal bending stiffness, in N.m (> 0). |
| `bending_stiffness_2` | The other principal bending stiffness, in N.m (> 0). The argument order does not matter. |
| `speed_of_sound` | Speed of sound in air `c0` (Default: 343 m/s). |

**Returns:** The pair `(fc1, fc2)` in hertz, with $f_{c1} \le f_{c2}$.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive input. |

## orthotropic_plate_resonance

```python
orthotropic_plate_resonance(
    mode_x: int,
    mode_z: int,
    *,
    length_x: float,
    length_z: float,
    mass_per_area: float,
    bending_stiffness_x: float,
    bending_stiffness_z: float,
    bending_stiffness_xz: float,
) -> float
```

Eigenfrequency of a simply supported orthotropic plate (Vigran 3.113).

$$
f_{i,n} = \frac{\pi}{2 \sqrt{m''}} \sqrt{ \frac{i^{4} B_x}{a^{4}} + \frac{n^{4} B_z}{b^{4}} + \frac{2 i^{2} n^{2} B_{xz}}{a^{2} b^{2}} }
$$

(Vigran Eq. (3.113), printed p. 95; identical to Bies Eq. (7.27) after
Hearmon 1959). It collapses to the isotropic Eq. (3.109) when
$B_x = B_z = B$ and $B_{xz} = B\,(\nu + 2 (1 - \nu)/2) = B$.

The lowest eigenfrequency $f_{1,1}$ matters to the transmission-loss
prediction because the infinite-panel models of
[`orthotropic_transmission_loss`](/phonometry/reference/api/building/panel-transmission/#orthotropic_transmission_loss) and
[`single_panel_transmission_loss`](/phonometry/reference/api/building/panel-transmission/#single_panel_transmission_loss) are only valid above about
$1.5 f_{1,1}$ (Bies, Sect. 7.2.4).

**Parameters**

| Name | Description |
| :--- | :--- |
| `mode_x` | Mode order `i` along `a` (integer >= 1). |
| `mode_z` | Mode order `n` along `b` (integer >= 1). |
| `length_x` | Plate dimension `a`, in m (> 0), along the axis whose bending stiffness is *bending_stiffness_x*. |
| `length_z` | Plate dimension `b`, in m (> 0). |
| `mass_per_area` | Mass per unit area `m''`, in kg/m^2 (> 0). |
| `bending_stiffness_x` | `Bx`, in N.m (> 0). |
| `bending_stiffness_z` | `Bz`, in N.m (> 0). |
| `bending_stiffness_xz` | `Bxz`, in N.m (> 0). |

**Returns:** The eigenfrequency, in hertz.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive input or a mode order below 1. |

## orthotropic_transmission_loss

```python
orthotropic_transmission_loss(
    frequency: ArrayLike,
    mass_per_area: float,
    *,
    critical_frequency_lower: float,
    critical_frequency_upper: float,
    loss_factor: float = 0.01,
    method: str = 'integral',
    area: float | None = None,
    limiting_angle: float = 78.0,
    band: str = 'third',
    speed_of_sound: float = 343.0,
    air_density: float = 1.205,
) -> SoundReductionResult
```

Orthotropic-panel sound reduction index (Vigran 6.5.3, Bies 7.2.4.5).

A ribbed or corrugated sheet is stiff along the corrugations and limp across
them, so instead of one coincidence dip it has a whole **coincidence range**
`fc1` to `fc2` (see [`orthotropic_critical_frequencies`](/phonometry/reference/api/building/panel-transmission/#orthotropic_critical_frequencies)). Over that
range the resonant transmission dominates and `R` flattens well below the
mass law of a flat plate of the same surface density, which is the price
paid for the strength-to-weight ratio.

Two prediction routes, both from the same wall impedance (Heckl 1960;
Hansen 1993; Vigran Eq. (6.108) = Bies Eq. (7.30))

$$
Z_w = j \omega m'' \left[ 1 - \left( (f/f_{c1}) \cos^{2}\theta + (f/f_{c2}) \sin^{2}\theta \right)^{2} (1 + j \eta) \sin^{4}\phi \right]
$$

* `method="integral"` (Default) averages the angular transmission
  coefficient
  $\tau = \lvert 1 + Z_w \cos\phi / (2 \rho_0 c_0) \rvert^{-2}$
  (Vigran Eq. (6.109) = Bies Eq. (7.31)) over azimuth and incidence
  angle, $\tau_F = \frac{2}{\pi} \int_0^{\pi/2} \int_0^{\sin^{2}\theta_L} \tau \,d(\sin^{2}\phi)\, d\theta$
  (Vigran Eq. (6.111) = Bies Eq. (7.38)), numerically. The
  near-grazing angles are excluded by the limiting angle: pass *area* for
  the size-dependent limit of Bies Eq. (7.36) (the correction Vigran writes
  as Eq. (6.113)) or leave it out for the fixed *limiting_angle*. This is
  the only route that responds to the loss factor.
* `method="heckl"` is Heckl's closed-form approximation for
  $\eta = 0$, the design chart of Bies Figure 7.9(b):
  field-incidence mass law below $f_{c1}/2$, Eq. (7.59) (the first
  of Vigran Eq. (6.112)) from `fc1` to $f_{c2}/2$, Eq. (7.60)
  (the second) above $2 f_{c2}$, and straight lines in
  $\log_{10} f$ across the two gaps. It is cheap and it needs no
  loss factor, but it cannot show the depth of the coincidence region and
  it requires $f_{c2} > 4 f_{c1}$ for its four construction points
  to stay ordered.

The two routes are not interchangeable. Above $2 f_{c2}$ they
converge as the loss factor falls: with $\eta \to 0$ the integral
lands within about 0.3 dB of Eq. (7.60), which is a useful independent
check on both
transcriptions. Across the coincidence range Eq. (7.59) is a much rougher
approximation and stays a few decibels above the integral even at
$\eta \to 0$, as Vigran's Figure 6.27 shows for its own worked case.

Both models are infinite-panel models, valid above roughly
$1.5 f_{1,1}$ ([`orthotropic_plate_resonance`](/phonometry/reference/api/building/panel-transmission/#orthotropic_plate_resonance)). Bies also
notes two systematic departures of the Heckl branch from measurement:
below about $0.7 f_{c1}$ it underestimates `R` on small panels,
and real corrugated panels show a dip of up to 5 dB between 2 kHz and
4 kHz caused by resonances of the panel sections between the ribs, which
no smooth model predicts.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Band centre frequencies `f`, in hertz (array, > 0). |
| `mass_per_area` | Mass per unit area `m''`, in kg/m^2 (> 0). |
| `critical_frequency_lower` | Lower coincidence frequency `fc1`, in hertz (> 0), from the stiffest direction. |
| `critical_frequency_upper` | Upper coincidence frequency `fc2`, in hertz (> `fc1`). |
| `loss_factor` | Total loss factor `eta` (> 0, Default: 0.01); used only by `method="integral"`, but validated on both routes. |
| `method` | `"integral"` (Default) or `"heckl"`. |
| `area` | Panel area `S`, in m^2 (> 0), selecting the size-dependent limiting angle of Bies Eq. (7.36) (Default: `None`); used only by `method="integral"`, but validated on both routes. |
| `limiting_angle` | Fixed limiting angle `theta_L`, in degrees ($0 < \theta_L < 90$, Default: 78.0), used when *area* is `None` and only by `method="integral"`, but validated on both routes. |
| `band` | Band width for the field correction of the Heckl mass-law branch (`"third"`/`"octave"`). |
| `speed_of_sound` | Speed of sound in air `c0` (Default: 343 m/s). |
| `air_density` | Air density `rho0` (Default: 1.205 kg/m^3). |

**Returns:** A [`SoundReductionResult`](/phonometry/reference/api/building/panel-transmission/#soundreductionresult) (model `"orthotropic-integral"` or `"orthotropic-heckl"`) carrying `fc1` in [`critical_frequency`](/phonometry/reference/api/building/flanking-transmission/#critical_frequency) and `fc2` in [`critical_frequency_upper`](/phonometry/reference/api/building/panel-transmission/#soundreductionresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive input, an unknown method, a coincidence range that is not increasing, or a Heckl construction whose points would be out of order. |

## PLATEAU_MATERIALS

*Constant* (`dict`).

```python
PLATEAU_MATERIALS = {'aluminium': (2.66, 29.0, 11.0), 'brick': (2.1, 37.0, 4.5), 'concrete': (2.28, 38.0, 4.5), 'glass': (2.47, 27.0, 10.0), 'lead': (11.2, 56.0, 4.0), 'plaster': (1.71, 30.0, 8.0), 'plywood': (0.57, 19.0, 6.5), 'steel': (7.6, 40.0, 11.0)}
```

## plateau_transmission_loss

```python
plateau_transmission_loss(
    frequency: ArrayLike,
    *,
    material: str | None = None,
    thickness_mm: float | None = None,
    mass_per_area: float | None = None,
    plateau_height: float | None = None,
    frequency_ratio: float | None = None,
    field_correction: float = 5.0,
    speed_of_sound: float = 343.0,
    air_density: float = 1.205,
) -> SoundReductionResult
```

Plateau-method estimate of a single panel's TL (Norton 3.9.1).

The plateau (Watters) construction is the empirical shortcut practitioners
draw by hand, and it approximates the whole curve from three numbers per
material (Norton & Karczub Table 3.1, tabulated in
[`PLATEAU_MATERIALS`](/phonometry/reference/api/building/panel-transmission/#plateau_materials)):

1. the **field-incidence mass law**
   $\mathrm{TL} = 10 \lg[1 + (\pi f m''/\rho_0 c_0)^{2}] - 5$
   (Eqs. 3.104/3.106), rising 6 dB per octave;
2. a horizontal **coincidence plateau** at the material's plateau height;
   point **A** is where the mass-law line reaches it;
3. point **B** at `frequency_ratio x fA`, above which the estimate
   recovers at **10 dB per octave**.

Unlike the physical model of [`single_panel_transmission_loss`](/phonometry/reference/api/building/panel-transmission/#single_panel_transmission_loss) it
needs neither the bending stiffness nor the loss factor: the material's
tabulated plateau absorbs both. The price is that it is only an estimate,
and it assumes a diffuse field on both sides of a panel whose length and
width are at least twenty times its thickness.

Give a tabulated *material* with its *thickness_mm* (the surface density
then follows from the table), or give *mass_per_area* together with
*plateau_height* and *frequency_ratio*. An explicit *mass_per_area*,
*plateau_height* or *frequency_ratio* always overrides the table.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Band centre frequencies `f`, in hertz (array, > 0). |
| `material` | Key into [`PLATEAU_MATERIALS`](/phonometry/reference/api/building/panel-transmission/#plateau_materials) (Default: `None`). |
| `thickness_mm` | Panel thickness, in **millimetres** (> 0), used with *material* to get the surface density. |
| `mass_per_area` | Mass per unit area `m''`, in kg/m^2 (> 0). |
| `plateau_height` | Coincidence plateau height, in dB (> 0). |
| `frequency_ratio` | Ratio $B/A$ locating the 10 dB/octave recovery (> 1). |
| `field_correction` | Field-incidence correction of the mass-law line, in dB (Default: 5.0, Norton Eq. 3.106). |
| `speed_of_sound` | Speed of sound in air `c0` (Default: 343 m/s). |
| `air_density` | Air density `rho0` (Default: 1.205 kg/m^3). |

**Returns:** A [`SoundReductionResult`](/phonometry/reference/api/building/panel-transmission/#soundreductionresult) (model `"plateau"`) carrying [`plateau_height`](/phonometry/reference/api/building/panel-transmission/#soundreductionresult), [`plateau_start`](/phonometry/reference/api/building/panel-transmission/#soundreductionresult) (point A) and [`plateau_end`](/phonometry/reference/api/building/panel-transmission/#soundreductionresult) (point B).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive input, an unknown material, or an under-specified panel. |

## plot_double_wall_geometry

```python
plot_double_wall_geometry(
    mass1: float,
    mass2: float,
    gap: float,
    ax: Axes | None = None,
    *,
    resonance_frequency: float | None = None,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Draw the mass-spring-mass double wall to scale.

Two leaves separated by the `gap`; leaf thicknesses are drawn from the
surface densities at a nominal board density, and the mass-spring-mass
resonance is annotated when given.

**Parameters**

| Name | Description |
| :--- | :--- |
| `mass1` | Surface density of the first leaf, in kg/m2. |
| `mass2` | Surface density of the second leaf, in kg/m2. |
| `gap` | Cavity depth, in metres. |
| `ax` | Existing axes, or `None` to create a figure. |
| `resonance_frequency` | Optional `f0` to annotate, in Hz. |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the leaf rectangles. |

**Returns:** The axes.

## single_panel_transmission_loss

```python
single_panel_transmission_loss(
    frequency: ArrayLike,
    mass_per_area: float,
    *,
    critical_frequency: float | None = None,
    bending_stiffness: float | None = None,
    loss_factor: float = 0.01,
    band: str = 'third',
    coincidence_model: str = 'sharp',
    field_correction: float | None = None,
    speed_of_sound: float = 343.0,
    air_density: float = 1.205,
) -> SoundReductionResult
```

Sound reduction index of a single panel, Sharp's method (Bies 7.2.4.1).

Field-incidence mass law up to $f_c/2$, Eq. 7.44 from `fc`
upwards, and a straight line in $\log_{10} f$ across the
coincidence region between them.

With `coincidence_model="cremer"` the region above `fc` follows Cremer's
empirical relationship instead (Norton & Karczub Eq. 3.110),

$$
\mathrm{TL} = \mathrm{TL}_0 + 10 \lg(f/f_c - 1) + 10 \lg\eta - 2~\text{dB}
$$

which also rises at 10 dB per octave far above coincidence but starts from
the singularity at `fc` itself rather than from a finite value. Norton
pairs it with the field-incidence mass law below `fc` and treats the two
as the whole model, so there is no interpolated bridge: the mass law runs
all the way to `fc`.

The empirical line is floored at $\mathrm{TL} = 0$ dB, which is
where it lands at $f = f_c$: Norton's Eq. (3.109) has
$\theta_{\mathrm{CO}} = 90$ degrees there and the
panel "offers no resistance to incident sound waves", $\tau = 1$.
It is also the hard bound of a passive panel, so without the floor a band
centre landing on `fc` would report an arbitrarily large negative TL
and a transmission coefficient above one.

Provide the coincidence frequency directly through *critical_frequency*, or
let it be computed from *bending_stiffness* and *mass_per_area* through
[`coincidence_frequency`](/phonometry/reference/api/vibration/radiation-efficiency/#coincidence_frequency).

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Band centre frequencies `f`, in hertz (array, > 0). |
| `mass_per_area` | Mass per unit area `m''`, in kg/m^2 (> 0). |
| `critical_frequency` | Coincidence frequency `fc`, in hertz (> 0). |
| `bending_stiffness` | Bending stiffness per unit width `B'`, in N.m, used to compute `fc` when *critical_frequency* is not given. |
| `loss_factor` | Total loss factor `eta` (> 0, Default: 0.01). |
| `band` | Band width for the field correction (`"third"`/`"octave"`). |
| `coincidence_model` | `"sharp"` (Default, Bies Eq. 7.44 above `fc` with the interpolated bridge from $f_c/2$) or `"cremer"` (Norton Eq. 3.110, mass law right up to `fc`). |
| `field_correction` | Explicit field-incidence correction of the mass-law region, in dB (>= 0), overriding the band table (Default: `None`; Norton's Eq. 3.106 uses a flat 5 dB). |
| `speed_of_sound` | Speed of sound in air `c0` (Default: 343 m/s). |
| `air_density` | Air density `rho0` (Default: 1.205 kg/m^3). |

**Returns:** A [`SoundReductionResult`](/phonometry/reference/api/building/panel-transmission/#soundreductionresult) (model `"sharp-single"` or `"cremer-single"`).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive input, an unknown coincidence model, or if neither *critical_frequency* nor *bending_stiffness* is given. |

## SoundReductionResult

```python
SoundReductionResult(
    frequencies: np.ndarray,
    transmission_loss: np.ndarray,
    model: str,
    critical_frequency: float | None = None,
    resonance_frequency: float | None = None,
    mass1: float | None = None,
    mass2: float | None = None,
    gap: float | None = None,
    plateau_height: float | None = None,
    plateau_start: float | None = None,
    plateau_end: float | None = None,
    critical_frequency_upper: float | None = None,
)
```

Predicted airborne sound reduction index `R(f)` of a construction.

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Band centre frequencies, in hertz. |
| `transmission_loss` | Sound reduction index `R` per band, in dB. |
| `model` | Prediction model (e.g. `"sharp-single"`, `"double-wall"`). |
| `critical_frequency` | Coincidence frequency `fc`, in hertz, or `None` (double wall reports the mass-spring-mass resonance instead). |
| `resonance_frequency` | Mass-spring-mass resonance `f0`, in hertz, or `None` (single panel). |
| `mass1` | First-leaf surface density, in kg/m2, retained (with `mass2` and `gap`) by the double-wall constructor so `plot_geometry` can draw the section; `None` otherwise. |
| `mass2` | Second-leaf surface density, in kg/m2, or `None`. |
| `gap` | Cavity depth, in metres, or `None`. |
| `plateau_height` | Height of the coincidence plateau, in dB, or `None` (only the plateau model sets these three). |
| `plateau_start` | Frequency of point A, where the mass-law line meets the plateau, in hertz, or `None`. |
| `plateau_end` | Frequency of point B, where the 10 dB/octave recovery starts, in hertz, or `None`. |
| `critical_frequency_upper` | Upper coincidence frequency `fc2` of an orthotropic panel, in hertz, or `None`; `critical_frequency` then carries the lower bound `fc1` and the pair spans the flattened coincidence range. |

### SoundReductionResult.plot()

```python
SoundReductionResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the predicted sound reduction index `R(f)`.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### SoundReductionResult.plot_geometry()

```python
SoundReductionResult.plot_geometry(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Draw the mass-spring-mass cross-section to scale.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the result does not retain its geometry. |

### SoundReductionResult.rating()

```python
SoundReductionResult.rating(
    bands: str | None = None,
) -> WeightedRatingResult
```

Single-number weighted rating `Rw` of the predicted `R(f)`.

Delegates to [`phonometry.weighted_rating`](/phonometry/reference/api/building/insulation/#weighted_rating) (ISO 717-1); requires
the spectrum to be on the 16 one-third-octave bands (100 Hz to
3150 Hz) or the 5 octave bands (125 Hz to 2000 Hz).

**Parameters**

| Name | Description |
| :--- | :--- |
| `bands` | Band set forwarded to [`phonometry.weighted_rating`](/phonometry/reference/api/building/insulation/#weighted_rating). |

**Returns:** The [`WeightedRatingResult`](/phonometry/reference/api/building/insulation/#weightedratingresult).

### SoundReductionResult.report()

```python
SoundReductionResult.report(path: str, **kwargs: Any) -> str
```

Render the ISO 717-1 Annex C rating fiche of `R(f)` to a PDF.

Convenience wrapper delegating to
[`report`](/phonometry/reference/api/building/insulation/)
on `rating`; requires the predicted spectrum to be on the 16
one-third-octave bands (100 Hz to 3150 Hz) or the 5 octave bands
(125 Hz to 2000 Hz).

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `kwargs` | Forwarded to [`report`](/phonometry/reference/api/building/insulation/) (e.g. `engine`). |

**Returns:** The written `path` as a `str`.

### SoundReductionResult.transmission_coefficient

*property*

Transmission coefficient $\tau = 10^{-R/10}$ per band.
