---
title: "noise_control.valves"
description: "Control valve aerodynamic noise (IEC 60534-8-3:2010)."
sidebar:
  label: "valves"
---

Control valve aerodynamic noise (IEC 60534-8-3:2010).

A control valve throttles a compressible fluid by turning pressure into
velocity and then throwing that velocity away in a free jet inside the pipe.
A small, well-characterised fraction of the jet's stream power comes back as
sound, most of it radiated not by the valve but by the **pipe wall
downstream**, which is why the method ends in a transmission loss and not in a
sound power level.

The standard is a chain with a branch in the middle. The branch is the
**regime**: how far the throttling has gone, from subsonic flow in the vena
contracta (regime I) through the onset of choking to the fully developed
shock cells of regime V. Five printed pressure ratios,
Equations (3) to (7), cut the differential pressure ratio $x$ into
those five intervals, and Table 3 gives each one its own Mach number, its own
acoustical efficiency and its own peak frequency. Everything before the
branch (the pressure ratios, the jet diameter) and everything after it (the
internal level at the pipe wall, the pipe transmission loss, the level
outside) is common to all five.

**What is new in the 2010 edition, and what this module therefore does.** The
1997 method produced one number. This one produces a **third-octave
spectrum**: Equation (19) spreads the internal level around the peak
frequency, Equation (20a) gives the pipe a transmission loss that changes with
frequency through the ring and coincidence frequencies of Equations (21) to
(23), and only Equation (25) collapses the result back to a single A-weighted
level at 1 m. The band set is the 33 one-third-octave bands from 12,5 Hz to
20 kHz, printed as Table 5.

**Three things in Annex A do not reproduce themselves**, and all three are
recorded in `docs/ERRATA.md`:

* The piping geometry factor is printed as $F_p = 0{,}98$, but every one
  of the six printed vena contracta pressures needs $0{,}984$ to come
  out. The five examples that print a value of $p_{vc}$ all give
  $(F_{LP}/F_P)^2 = 0{,}647\,83$, which is $F_p = 0{,}984$ to six
  digits and not $0{,}98$.
* The equivalent orifice diameter is printed as $d_o = 0{,}010$ m in all
  six columns, where Equation (8c) with the annex's own $N_O = 6$ and
  $A = 0{,}00137$ m² gives $0{,}102$ m. The valve style modifier
  printed on the next row, $F_d = 0{,}30$, is the ratio of the printed
  $d_H = 0{,}030$ m to $0{,}102$ m, so the annex computed with the
  larger value and printed the smaller one.
* Two frequency factors of Table A.2 are printed one power of ten low,
  $G_{x,5}$ and $G_{x,10}$, in a column Table 6 makes
  proportional to $f_i^4$ and which therefore has to rise. The
  transmission losses printed two rows below them are what the corrected
  factors give.

This module implements Clause 5, the standard trim case. The noise-reducing
trims of Clause 6, the expander of Clause 7 and the hydrodynamic case of
IEC 60534-8-4 are separate.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## AERODYNAMIC_A_WEIGHTING_DB

*Constant* (`tuple`).

```python
AERODYNAMIC_A_WEIGHTING_DB = (-63.4, -56.7, -50.5, -44.7, -39.4, -34.6, -30.2, -26.2, -22.5, -19.1, -16.1, -13.4, -10.9, -8.6, -6.6, -4.8, -3.2, -1.9, -0.8, 0.0, 0.6, 1.0, 1.2, 1.3, 1.2, 1.0, 0.5, -0.1, -1.1, -2.5, -4.3, -6.6, -9.3)
```

## AerodynamicValveNoise

```python
AerodynamicValveNoise(
    regime: int,
    boundaries: RegimeBoundaries,
    pressure_ratio: float,
    vena_contracta_pressure: float,
    jet_diameter: float,
    mach: float,
    acoustical_efficiency: float,
    stream_power: float,
    sound_power: float,
    sound_power_level: float,
    peak_frequency: float,
    outlet_mach: float,
    pipe_mach: float,
    velocity_correction: float,
    internal_level: float,
    frequency: NDArray[np.float64],
    band_internal_level: NDArray[np.float64],
    band_transmission_loss: NDArray[np.float64],
    band_external_level: NDArray[np.float64],
    external_level: float,
    pipe_frequencies: PipeFrequencies,
    expander: ExpanderNoise | None,
)
```

What IEC 60534-8-3 Clause 5 says about one operating point.

**Attributes**

| Name | Description |
| :--- | :--- |
| `regime` | Which of the five regimes of Clause 5.2 the valve is in. |
| `boundaries` | The four pressure ratios that placed it there. |
| `pressure_ratio` | $x$ of Equation (1). |
| `vena_contracta_pressure` | $p_{vc}$ of Equation (2), in Pa. It goes negative past the choking point, where the equation is being read outside the range it means anything in. |
| `jet_diameter` | $D_j$ of Equation (9), in m. |
| `mach` | The Mach number Table 3 uses in this regime. |
| `acoustical_efficiency` | $\eta$, the fraction of the stream power that leaves as sound. |
| `stream_power` | $W_m$, in W. |
| `sound_power` | $W_a$ of Equation (11), in W. |
| `sound_power_level` | $L_{wi}$ of Equation (12), in dB. |
| `peak_frequency` | $f_p$ from Table 3, in Hz. |
| `outlet_mach` | $M_o$ of Equation (15), which Clause 5 is only valid below 0,3. |
| `pipe_mach` | $M_2$ of Equation (17), before the 0,3 limit. |
| `velocity_correction` | $L_g$ of Equation (16), in dB. |
| `internal_level` | $L_{pi}$ of Equation (18), in dB. |
| `frequency` | The 33 one-third-octave band centres of Table 5, in Hz. |
| `band_internal_level` | $L_{pi}(f_i)$ of Equation (19), in dB. |
| `band_transmission_loss` | $TL(f_i)$ of Equation (20a), in dB. |
| `band_external_level` | $L_{pe,1m}(f_i)$ of Equation (24), in dB. |
| `external_level` | $L_{pAe,1m}$ of Equation (25), in dB. |
| `pipe_frequencies` | The ring and coincidence frequencies the transmission loss is shaped by. |
| `expander` | What Clause 7 says the flow leaving the valve outlet makes, or `None` when no expander was given. When it is present its spectrum is already in `band_external_level` and in `external_level`, combined with the trim by Equation (43). |

## AIR_SOUND_SPEED_M_S

*Constant* (`float`).

```python
AIR_SOUND_SPEED_M_S = 343.0
```

## coincidence_frequencies

```python
coincidence_frequencies(
    internal_diameter: float,
    wall_thickness: float,
    downstream_sound_speed: float,
    *,
    pipe_sound_speed: float = 5000.0,
    air_sound_speed: float = 343.0,
) -> PipeFrequencies
```

Equations (21), (22) and (23).

$$
f_r = \frac{c_s}{\pi D_i}, \qquad f_o = \frac{f_r}{4}\left(\frac{c_2}{c_a}\right), \qquad f_g = \frac{\sqrt{3}}{\pi t_S}\frac{c_a^2}{c_s}
$$

**Parameters**

| Name | Description |
| :--- | :--- |
| `internal_diameter` | $D_i$ of the downstream pipe, in m. |
| `wall_thickness` | $t_S$ of the pipe wall, in m. |
| `downstream_sound_speed` | $c_2$ in the fluid downstream of the valve, in m/s. |
| `pipe_sound_speed` | $c_s$, 5 000 m/s for steel by NOTE 4. |
| `air_sound_speed` | $c_a$, 343 m/s by NOTE 3. |

**Returns:** The three frequencies, in Hz.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If any argument is not positive and finite. |

## combine_internal_levels

```python
combine_internal_levels(*levels: NDArray[np.float64]) -> NDArray[np.float64]
```

Equation (43): two internal spectra at the same pipe wall, added.

$$
L_{piS}(f_i) = 10 \lg\left( 10^{L_{pi}(f_i)/10} + 10^{L_{piR}(f_i)/10}\right)
$$

The valve trim and the expander are two sources inside one pipe, so they
add in energy and not in level, and the sum is what Equation (24) then
takes through the wall.

**Parameters**

| Name | Description |
| :--- | :--- |
| `levels` | Two or more band level arrays of the same shape, in dB. |

**Returns:** Their energy sum, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If fewer than two are given, or they disagree in shape. |

## DEFAULT_EXPANDER

*Constant* (`phonometry.noise_control.valves.Expander`).

## Expander

```python
Expander(
    contraction: float = 0.93,
    efficiency_correction: float = -3.0,
    strouhal_number: float = 0.2,
)
```

The transition piece downstream of the valve (Clause 7).

A valve whose outlet is narrower than the pipe it discharges into makes a
second jet, at the step. Clause 7 is the method for it, and 7.1 limits
the method to a transition of 30 degrees total included angle: a steeper
cone makes the flow unstable in ways the standard does not model.

**Attributes**

| Name | Description |
| :--- | :--- |
| `contraction` | $\beta$ of Equation (35). NOTE 1 puts it at 0,93 for straight pattern globe valves and as low as 0,7 for some rotary ones, and says there are no data for the rest. |
| `efficiency_correction` | $A_\eta$ for the expander, which is its own row of Table 4 and not the valve's: the table prints -3,0. |
| `strouhal_number` | $St_p$ for the expander, 0,2 in Table 4. |

## expander_noise

```python
expander_noise(
    frequency: NDArray[np.float64],
    *,
    mass_flow: float,
    downstream_density: float,
    downstream_sound_speed: float,
    internal_diameter: float,
    throat_diameter: float,
    velocity_correction: float,
    expander: Expander = ...,
) -> ExpanderNoise
```

Clause 7: the noise the flow makes leaving the valve outlet.

$$
U_p = \frac{4 \dot m}{\pi \rho_2 D_i^2}, \qquad U_R = \frac{U_p D_i^2}{\beta d_i^2}, \qquad M_R = \frac{U_R}{c_2}
$$

$$
W_{mR} = \frac{\dot m U_R^2}{2} \left[\left(1 - \frac{d_i^2}{D_i^2}\right)^2 + 0{,}2\right], \qquad \eta_R = 10^{A_\eta} M_R^3, \qquad f_{pR} = \frac{St_p U_R}{d_i}
$$

The two caps are the clause's: $U_p$ is limited to Mach 0,8 and
$U_R$ to the sonic velocity, so a step that would otherwise be
computed as supersonic is computed at Mach one instead.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | The band centre frequencies, in Hz. |
| `mass_flow` | $\dot m$, in kg/s. |
| `downstream_density` | $\rho_2$, in kg/m³. |
| `downstream_sound_speed` | $c_2$, in m/s. |
| `internal_diameter` | $D_i$ of the downstream pipe, in m. |
| `throat_diameter` | $d_i$, the smaller of the valve outlet and the expander inlet, in m. |
| `velocity_correction` | $L_g$ of Equation (16), in dB, which Equation (41) adds exactly as Equation (18) does. |
| `expander` | The transition piece. |

**Returns:** An [`ExpanderNoise`](/phonometry/reference/api/noise_control/valves/#expandernoise).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a value is not positive and finite, or the throat is wider than the pipe. |

## EXPANDER_PIPE_MACH_LIMIT

*Constant* (`float`).

```python
EXPANDER_PIPE_MACH_LIMIT = 0.8
```

## ExpanderNoise

```python
ExpanderNoise(
    pipe_velocity: float,
    inlet_velocity: float,
    mach: float,
    stream_power: float,
    acoustical_efficiency: float,
    sound_power: float,
    peak_frequency: float,
    internal_level: float,
    band_internal_level: NDArray[np.float64],
)
```

What Clause 7 says the flow leaving the valve outlet makes.

**Attributes**

| Name | Description |
| :--- | :--- |
| `pipe_velocity` | $U_p$ of Equation (34), in m/s, after the Mach 0,8 cap. |
| `inlet_velocity` | $U_R$ of Equation (35), in m/s, after the sonic cap. |
| `mach` | $M_R$ of Equation (39). |
| `stream_power` | $W_{mR}$ of Equation (36), in W. |
| `acoustical_efficiency` | $\eta_R$ of Equation (38). |
| `sound_power` | $W_{aR}$ of Equation (40), in W. |
| `peak_frequency` | $f_{pR}$ of Equation (37), in Hz. |
| `internal_level` | $L_{piR}$ of Equation (41), in dB. |
| `band_internal_level` | $L_{piR}(f_i)$ of Equation (42), in dB. |

## FLOW_COEFFICIENT_CONSTANTS

*Constant* (`dict`).

```python
FLOW_COEFFICIENT_CONSTANTS = {'Cv': 0.0046, 'Kv': 0.0049}
```

## flow_regime

```python
flow_regime(pressure_ratio: float, boundaries: RegimeBoundaries) -> int
```

Which of the five regimes of Clause 5.2 a pressure ratio falls in.

The clause prints the five intervals half open, each one closed at the
top: $x \le x_C$, then $x_C < x \le x_{vcc}$, then
$x_{vcc} < x \le x_B$, then $x_B < x \le x_{CE}$, and finally
$x_{CE} < x$.

Table 3 prints the last one as $x_{CE} \le x$, which would put the
single point $x = x_{CE}$ in two regimes at once. Clause 5.2 is the
normative text and its list is consistent, so this follows the clause;
`docs/ERRATA.md` records the disagreement.

**Parameters**

| Name | Description |
| :--- | :--- |
| `pressure_ratio` | $x$ of Equation (1). |
| `boundaries` | The output of [`pressure_ratio_boundaries`](/phonometry/reference/api/noise_control/valves/#pressure_ratio_boundaries). |

**Returns:** The regime number, 1 to 5.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the pressure ratio is not a finite number in (0, 1). |

## GLOBE_CONTRACTION_COEFFICIENT

*Constant* (`float`).

```python
GLOBE_CONTRACTION_COEFFICIENT = 0.93
```

## internal_spectrum

```python
internal_spectrum(
    internal_level: float,
    peak_frequency: float,
    frequency: NDArray[np.float64],
) -> NDArray[np.float64]
```

Equation (19): the internal level spread over the third-octave bands.

$$
L_{pi}(f_i) = L_{pi} - 8 - 10 \lg\left\{ \left[1 + \left(\frac{f_i}{2 f_p}\right)^{2,5}\right] \left[1 + \left(\frac{f_p}{2 f_i}\right)^{1,7}\right]\right\}
$$

The two brackets are not symmetric: the spectrum falls as
$f^{-2,5}$ above the peak and as $f^{1,7}$ below it, so a
valve is heard further above its peak than below it. The 8 dB is what
turns an overall level into a one-third-octave one; the NOTE to Table 7
puts 3 dB there for octave bands instead.

**Parameters**

| Name | Description |
| :--- | :--- |
| `internal_level` | $L_{pi}$ of Equation (18), in dB. |
| `peak_frequency` | $f_p$ from Table 3, in Hz. |
| `frequency` | The band centre frequencies, in Hz. |

**Returns:** The internal level in each band, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the peak frequency is not positive and finite, or a band centre is not. |

## jet_diameter

```python
jet_diameter(
    flow_coefficient: float,
    style_modifier: float,
    pressure_recovery: float,
    *,
    coefficient: str = 'Cv',
) -> float
```

The jet diameter of Equation (9).

$$
D_j = N_{14}\, F_d \sqrt{C\, F_{LP}/F_P}
$$

**Parameters**

| Name | Description |
| :--- | :--- |
| `flow_coefficient` | $C$, the required flow coefficient of the valve at the travel being examined. |
| `style_modifier` | $F_d$, from [`valve_style_modifier`](/phonometry/reference/api/noise_control/valves/#valve_style_modifier). |
| `pressure_recovery` | $F_{LP}/F_p$, or $F_L$ for a valve with no attached fittings. |
| `coefficient` | Which flow coefficient `flow_coefficient` is, `"Cv"` or `"Kv"`, which selects $N_{14}$ from Table 1. |

**Returns:** $D_j$, in m.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a value is not positive and finite, or the coefficient is not one Table 1 prints a constant for. |

## MACH_LIMIT_STANDARD_TRIM

*Constant* (`float`).

```python
MACH_LIMIT_STANDARD_TRIM = 0.3
```

## PIPE_SOUND_SPEED_M_S

*Constant* (`float`).

```python
PIPE_SOUND_SPEED_M_S = 5000.0
```

## pipe_transmission_loss

```python
pipe_transmission_loss(
    frequency: NDArray[np.float64],
    *,
    internal_diameter: float,
    wall_thickness: float,
    valve_outlet_diameter: float,
    downstream_density: float,
    downstream_sound_speed: float,
    pipe_density: float,
    pipe_sound_speed: float = 5000.0,
    air_sound_speed: float = 343.0,
    atmospheric_pressure: float = 101325.0,
    standard_pressure: float = 101325.0,
) -> NDArray[np.float64]
```

Equation (20a): what the pipe wall keeps in, band by band.

$$
TL(f_i) = 10 \lg\left[ 8{,}25\times10^{-7} \left(\frac{c_2}{t_S f_i}\right)^2 \frac{G_x(f_i)} {\dfrac{\rho_2 c_2 + 2\pi t_S f_i \rho_s \eta_s(f_i)} {415\, G_y(f_i)} + 1} \frac{p_a}{p_s}\right] - \Delta TL
$$

The result is a large negative number, and Equation (24) *adds* it to the
internal level, so the sign is not a convention this module chose.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | The band centre frequencies, in Hz. |
| `internal_diameter` | $D_i$, in m. |
| `wall_thickness` | $t_S$, in m. |
| `valve_outlet_diameter` | $D$, in m, which selects the damping factor of Equation (20b) and is the valve outlet and not the pipe. |
| `downstream_density` | $\rho_2$, in kg/m³. |
| `downstream_sound_speed` | $c_2$, in m/s. |
| `pipe_density` | $\rho_s$ of the pipe material, in kg/m³. |
| `pipe_sound_speed` | $c_s$, in m/s. |
| `air_sound_speed` | $c_a$, in m/s. |
| `atmospheric_pressure` | $p_a$, in Pa. |
| `standard_pressure` | $p_s$, in Pa. |

**Returns:** The transmission loss in each band, in dB, negative.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If an argument is not positive and finite. |

## PIPE_WALL_MACH_LIMIT

*Constant* (`float`).

```python
PIPE_WALL_MACH_LIMIT = 0.3
```

## PipeFrequencies

```python
PipeFrequencies(
    ring: float,
    internal_coincidence: float,
    external_coincidence: float,
)
```

The three frequencies Clause 5.5 shapes the transmission loss with.

**Attributes**

| Name | Description |
| :--- | :--- |
| `ring` | $f_r$ of Equation (21), where the pipe rings as a circumference of one wavelength. |
| `internal_coincidence` | $f_o$ of Equation (22). |
| `external_coincidence` | $f_g$ of Equation (23). |

## pressure_ratio_boundaries

```python
pressure_ratio_boundaries(
    specific_heat_ratio: float,
    pressure_recovery: float,
) -> RegimeBoundaries
```

The regime boundaries of Equations (3) to (7).

$$
x_{vcc} = 1 - \left(\frac{2}{\gamma + 1}\right)^{\gamma/(\gamma-1)}, \qquad x_C = F_L^2\, x_{vcc}, \qquad \alpha = \frac{1 - x_{vcc}}{1 - x_C}
$$

$$
x_B = 1 - \frac{1}{\alpha} \left(\frac{1}{\gamma}\right)^{\gamma/(\gamma-1)}, \qquad x_{CE} = 1 - \frac{1}{22\,\alpha}
$$

**Parameters**

| Name | Description |
| :--- | :--- |
| `specific_heat_ratio` | $\gamma$ of the flowing fluid. |
| `pressure_recovery` | $F_L$, or $F_{LP}/F_p$ when the valve has attached fittings, which is what the NOTE to Table 3 asks for and what every example in Annex A uses. |

**Returns:** The four boundaries and the recovery factor behind two of them.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If either argument is not positive and finite, or if the specific heat ratio is not above one. |

## REGIME_CHOKED

*Constant* (`int`).

```python
REGIME_CHOKED = 2
```

## REGIME_CONSTANT_EFFICIENCY

*Constant* (`int`).

```python
REGIME_CONSTANT_EFFICIENCY = 5
```

## REGIME_COUNT

*Constant* (`int`).

```python
REGIME_COUNT = 5
```

## REGIME_SHOCK

*Constant* (`int`).

```python
REGIME_SHOCK = 4
```

## REGIME_SUBSONIC

*Constant* (`int`).

```python
REGIME_SUBSONIC = 1
```

## REGIME_SUPERSONIC

*Constant* (`int`).

```python
REGIME_SUPERSONIC = 3
```

## RegimeBoundaries

```python
RegimeBoundaries(
    vena_contracta: float,
    critical: float,
    break_point: float,
    constant_efficiency: float,
    recovery: float,
)
```

The four pressure ratios that cut Clause 5.2 into five regimes.

**Attributes**

| Name | Description |
| :--- | :--- |
| `vena_contracta` | $x_{vcc}$, where the flow in the vena contracta first reaches the speed of sound, Equation (3). |
| `critical` | $x_C$, the same point seen from the valve inlet, Equation (4). |
| `break_point` | $x_B$, where the jet stops growing and shock cells take over, Equation (6). |
| `constant_efficiency` | $x_{CE}$, where the acoustical efficiency stops rising with pressure ratio, Equation (7). |
| `recovery` | $\alpha$, the recovery correction factor of Equation (5), which the other two are written in terms of. |

## STRUCTURAL_LOSS_REFERENCE_HZ

*Constant* (`float`).

```python
STRUCTURAL_LOSS_REFERENCE_HZ = 1.0
```

## UNIVERSAL_GAS_CONSTANT

*Constant* (`float`).

```python
UNIVERSAL_GAS_CONSTANT = 8314.0
```

## VALVE_ACOUSTIC_STYLES

*Constant* (`dict`).

```python
VALVE_ACOUSTIC_STYLES = {'globe parabolic plug': (-4.2, 0.19), 'globe V-port plug': (-4.2, 0.19), 'globe ported cage': (-3.8, 0.2), 'globe multihole to open': (-4.8, 0.2), 'globe multihole to close': (-4.4, 0.2), 'butterfly eccentric': (-4.2, 0.3), 'butterfly swing-through': (-4.2, 0.3), 'butterfly fluted vane': (-4.2, 0.3), 'butterfly 60 deg flat disk': (-4.2, 0.3), 'eccentric rotary plug': (-3.6, 0.3), 'segmented ball 90 deg': (-3.6, 0.3), 'drilled hole plate': (-4.8, 0.2), 'expander': (-3.0, 0.2)}
```

## valve_aerodynamic_noise

```python
valve_aerodynamic_noise(
    *,
    mass_flow: float,
    inlet_pressure: float,
    outlet_pressure: float,
    inlet_density: float,
    inlet_temperature: float,
    specific_heat_ratio: float,
    molecular_mass: float,
    flow_coefficient: float,
    style_modifier: float,
    pressure_recovery: float,
    valve_outlet_diameter: float,
    internal_diameter: float,
    wall_thickness: float,
    pipe_density: float,
    efficiency_correction: float,
    strouhal_number: float,
    coefficient: str = 'Cv',
    expander: Expander | None = None,
    pipe_sound_speed: float = 5000.0,
    air_sound_speed: float = 343.0,
    atmospheric_pressure: float = 101325.0,
    standard_pressure: float = 101325.0,
) -> AerodynamicValveNoise
```

The whole of Clause 5, from the operating point to the level at 1 m.

The chain is Clause 5.7's own flow chart: the pressure ratios of 5.1 and
5.2, the geometry of 5.3, the regime-dependent stream power and
acoustical efficiency of 5.4, then the pipe transmission loss of 5.5 and
the external level of 5.6, which are common to every regime.

**Parameters**

| Name | Description |
| :--- | :--- |
| `mass_flow` | $\dot m$, in kg/s. |
| `inlet_pressure` | $p_1$, absolute, in Pa. |
| `outlet_pressure` | $p_2$, absolute, in Pa. |
| `inlet_density` | $\rho_1$, in kg/m³. |
| `inlet_temperature` | $T_1$, absolute, in K. |
| `specific_heat_ratio` | $\gamma$. |
| `molecular_mass` | $M$, in kg/kmol. |
| `flow_coefficient` | $C$ at the travel being examined. |
| `style_modifier` | $F_d$, from [`valve_style_modifier`](/phonometry/reference/api/noise_control/valves/#valve_style_modifier). |
| `pressure_recovery` | $F_L$, or $F_{LP}/F_p$ with attached fittings. |
| `valve_outlet_diameter` | $D$, in m. |
| `internal_diameter` | $D_i$ of the downstream pipe, in m. |
| `wall_thickness` | $t_S$, in m. |
| `pipe_density` | $\rho_s$, in kg/m³. |
| `efficiency_correction` | $A_\eta$ from Table 4. |
| `strouhal_number` | $St_p$ from Table 4. |
| `coefficient` | `"Cv"` or `"Kv"`, selecting $N_{14}$. |
| `expander` | The transition piece downstream of the valve. Give one when the valve outlet is narrower than the pipe and the outlet Mach number has passed 0,3, which is when NOTE 1 to Equation (15) sends the calculation to Clause 7; the flow leaving the outlet is then a second source and Equation (43) adds it to the trim. |
| `pipe_sound_speed` | $c_s$, in m/s. |
| `air_sound_speed` | $c_a$, in m/s. |
| `atmospheric_pressure` | $p_a$, in Pa. |
| `standard_pressure` | $p_s$, in Pa. |

**Returns:** An [`AerodynamicValveNoise`](/phonometry/reference/api/noise_control/valves/#aerodynamicvalvenoise) carrying every printed intermediate as well as the level at 1 m.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a value is outside the range its equation is written for. |

## valve_style_modifier

```python
valve_style_modifier(
    passage_area: float,
    wetted_perimeter: float,
    passages: int,
) -> float
```

The valve style modifier of Equations (8a) to (8c).

$$
d_H = \frac{4A}{l_w}, \qquad d_o = \sqrt{\frac{4 N_o A}{\pi}}, \qquad F_d = \frac{d_H}{d_o}
$$

$F_d$ compares the hydraulic diameter of one flow passage with the
diameter of the single circular orifice that would pass the same total
area. A cage full of small holes has a small $F_d$ and a small jet;
a single large port has $F_d$ near one.

**Parameters**

| Name | Description |
| :--- | :--- |
| `passage_area` | $A$, the area of a single flow passage, in m². |
| `wetted_perimeter` | $l_w$ of that passage, in m. |
| `passages` | $N_o$, the number of independent flow passages. |

**Returns:** $F_d$, dimensionless.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If an argument is not positive and finite, or if the passage count is not a whole number. |

## ValveNoiseWarning

A valve read outside the conditions IEC 60534-8-3 prints for it.
