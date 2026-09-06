---
title: "noise_control.valves_hydrodynamic"
description: "Control valve hydrodynamic noise (IEC 60534-8-4:2005)."
sidebar:
  label: "valves_hydrodynamic"
---

Control valve hydrodynamic noise (IEC 60534-8-4:2005).

A liquid does not compress, so a control valve on a water line cannot make
the shock cells that IEC 60534-8-3 spends five regimes on. It makes two
things instead, and the whole of this part is the sum of them:

* **turbulence** in the jet leaving the vena contracta, whose acoustical
  efficiency is a straight line in the jet velocity, Equation (8);
* **cavitation**, once the differential pressure passes the point where the
  liquid flashes to vapour in the vena contracta and the bubbles collapse
  again downstream. Equation (9) gives that its own efficiency, and it is a
  steep function of how far past the threshold the valve is: a fifth power of
  $x_F/x_{Fzp1}$ multiplied by an exponential.

The threshold is the **characteristic pressure ratio** $x_{Fz}$, a
measured property of the valve (IEC 60534-8-2) that Equations (3a) and (3b)
estimate when no measurement exists, corrected to the working inlet pressure
by Equation (3c). Everything in the method turns on where the operating ratio
$x_F$ of Equation (1) sits with respect to it, which is why Annex A's
third example perturbs $x_{Fz}$ by 0,1 and watches the answer move
14 dB.

After the source, the chain is the same shape as the aerodynamic one: an
internal level at the pipe wall, Equation (10); a transmission loss through
the wall, negative by construction and anchored at the ring frequency,
Equations (14) and (15); and a level 1 m outside, Equations (18a) and (18b).
The band-by-band route of 5.4 spreads the internal level around the peak
frequency with Equations (20a) and (20b) and gives the wall a
frequency-dependent loss with Equations (22a) and (22b).

**Six defects of the printed document**, all confirmed on the page and all
recorded in `docs/ERRATA.md`:

* Equation (12) is printed **twice, differently**: Clause 5.1 gives the
  Strouhal number a leading 0,02 and no valve style modifier, Annex A's
  Table A.1 gives it 0,036 and a factor $F_d^{0,75}$. Only the annex
  form reproduces the annex's own printed $N_{Str}$, which is why
  [`STROUHAL_CONSTANTS`](/phonometry/reference/api/noise_control/valves-hydrodynamic/#strouhal_constants) carries both and the default is `"annex"`.
* Table A.1 prints the band transmission loss as `TL(8 000 Hz) = 51,76 dB`,
  positive, where its own two inputs sum to $-51{,}763$ dB.
* The seat diameter formula of 6.3.2 b), $d_o = 5{,}2\sqrt{N_{34}C_n}$,
  returns millimetres for a symbol Clause 3 declares in metres, which is why
  [`last_stage_seat_diameter_mm`](/phonometry/reference/api/noise_control/valves-hydrodynamic/#last_stage_seat_diameter_mm) carries the unit in its name.
* Equation (23b) computes each stage's inlet pressure from the **next**
  stage's, which contradicts Equation (23a) and runs the pressure the wrong
  way along the trim.
* Equations (18a) and (18b) are printed with conditions on two different
  thresholds, $x_F \le x_{Fz}$ and $x_{Fzp1} < x_F \le 1$, which
  divide the domain between them only at the one inlet pressure where those
  two are equal. Every other regime statement in the document tests the
  corrected ratio, and so does this module.
* Three intermediates of Table A.1 do not follow from the intermediates
  printed beside them, by up to 0,08 dB.

Clause 6, the multistage trim, is the same method with per-stage inputs:
[`stage_conditions`](/phonometry/reference/api/noise_control/valves-hydrodynamic/#stage_conditions) splits the differential, and either the stages are
summed in energy by Equation (27) or, for a fixed device with increasing flow
areas, only the last stage is calculated at all.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## ACOUSTIC_POWER_RATIOS

*Constant* (`dict`).

```python
ACOUSTIC_POWER_RATIOS = {'globe parabolic plug': 0.25, 'globe 3 V-port plug': 0.25, 'globe 4 V-port plug': 0.25, 'globe 6 V-port plug': 0.25, 'globe 60 hole drilled cage': 0.25, 'globe 120 hole drilled cage': 0.25, 'butterfly swing-through': 0.5, 'butterfly fluted vane': 0.5, 'butterfly 60 deg flat disk': 0.5, 'eccentric rotary plug': 0.25, 'segmented ball 90 deg': 0.25, 'expander': 1.0}
```

## AIR_DENSITY_KG_M3

*Constant* (`float`).

```python
AIR_DENSITY_KG_M3 = 1.293
```

## band_internal_levels

```python
band_internal_levels(
    frequency: ArrayLike,
    internal_level: float,
    *,
    turbulent_peak: float,
    cavitation_peak: float | None = None,
    cavitation_fraction: float = 0.0,
) -> NDArray[np.float64]
```

Equations (19a) and (19b): the internal level, band by band.

$$
L_{pi}(f_i) = L_{pi} + F_{turb}(f_i)
$$

$$
L_{pi}(f_i) = L_{pi} + 10 \lg\left( \frac{\eta_{turb}}{\eta_{turb}+\eta_{cav}} 10^{0,1 F_{turb}(f_i)} + \frac{\eta_{cav}}{\eta_{turb}+\eta_{cav}} 10^{0,1 F_{cav}(f_i)} \right)
$$

The cavitating form is the turbulent and the cavitating spectra added in
energy, each weighted by the share of the sound power its own efficiency
accounts for. Since the two peak frequencies differ by a factor of a few,
the sum is a two-humped spectrum, and which hump is taller is decided by
$\eta_{cav}/(\eta_{turb}+\eta_{cav})$ alone.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | $f_i$, the band centres, in Hz. |
| `internal_level` | $L_{pi}$ of Equation (10), in dB. |
| `turbulent_peak` | $f_{p,turb}$, in Hz. |
| `cavitation_peak` | $f_{p,cav}$, in Hz, or `None` for the turbulent branch. |
| `cavitation_fraction` | $\eta_{cav}/(\eta_{turb}+\eta_{cav})$, between 0 and 1. Zero gives Equation (19a) whatever else is passed. |

**Returns:** $L_{pi}(f_i)$, in dB, one value per band.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a value is out of range, or the cavitating branch was asked for without its peak frequency. |

## CAPACITY_SCALE_CONSTANTS

*Constant* (`dict`).

```python
CAPACITY_SCALE_CONSTANTS = {'Cv': 1.17, 'Kv': 1.0}
```

## cavitation_differential

```python
cavitation_differential(
    *,
    inlet_pressure: float,
    outlet_pressure: float,
    vapour_pressure: float,
    pressure_recovery: float,
) -> float
```

Equation (2): the differential the jet velocity is computed from.

$$
\Delta p_c = \min\left[(p_1 - p_2),\; F_L^2 (p_1 - p_v)\right]
$$

The second candidate is where the flow chokes. Past it the valve cannot
turn any more differential into velocity, so $\Delta p_c$ stops
following $p_1 - p_2$ and Equation (5) stops accelerating the jet,
even though the noise keeps rising because cavitation takes over.

The printed equation says "lower than … or …", with no `min` operator
and no inequality; the minimum is what it means.

**Parameters**

| Name | Description |
| :--- | :--- |
| `inlet_pressure` | $p_1$, absolute, in Pa. |
| `outlet_pressure` | $p_2$, absolute, in Pa. |
| `vapour_pressure` | $p_v$, absolute, in Pa. |
| `pressure_recovery` | $F_L$ of the valve, dimensionless. |

**Returns:** $\Delta p_c$, in Pa.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a pressure is not positive and finite, if the valve does not drop pressure, or if the recovery factor is outside the range a recovery factor lives in. |

## cavitation_distribution

```python
cavitation_distribution(
    frequency: ArrayLike,
    peak: float,
) -> NDArray[np.float64]
```

Equation (20b): how cavitation noise spreads over the bands.

$$
F_{cav}(f_i) = -10 \lg\left[ \frac{1}{4}\left(\frac{f_i}{f_{p,cav}}\right)^{1,5} + \left(\frac{f_i}{f_{p,cav}}\right)^{-1,5}\right] - 3{,}5
$$

The same shape as Equation (20a) with every numeral changed. Both
exponents are $\pm 1,5$ instead of 3 and −1, so both flanks fall at
the same 4,5 dB per octave and the hump is symmetric, but about
$\sqrt[3]{4}\, f_{p,cav}$, two thirds of an octave above the
frequency Equation (13) names, because the quarter in front of the
rising branch shifts the maximum up. Against Equation (20a)'s 3 dB up and 9 dB down
that is a far broader spectrum: cavitation is heard as a wide band of
gravel where turbulence is heard as a hiss around one frequency.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | $f_i$, the band centres, in Hz. |
| `peak` | $f_{p,cav}$ from [`cavitation_peak_frequency`](/phonometry/reference/api/noise_control/valves-hydrodynamic/#cavitation_peak_frequency), in Hz. |

**Returns:** $F_{cav}(f_i)$, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a value is not positive and finite. |

## cavitation_efficiency

```python
cavitation_efficiency(
    *,
    turbulent: float,
    differential: float,
    choked_differential: float,
    pressure_ratio: float,
    corrected_ratio: float,
) -> float
```

Equation (9): what the collapsing bubbles add.

$$
\eta_{cav} = 0{,}32\, \eta_{turb} \sqrt{\frac{p_1 - p_2}{\Delta p_c}\cdot\frac{1}{x_{Fzp1}}}\; e^{5 x_{Fzp1}} \left(\frac{1 - x_{Fzp1}}{1 - x_F}\right)^{0,5} \left(\frac{x_F}{x_{Fzp1}}\right)^{5} \left(x_F - x_{Fzp1}\right)^{1,5}
$$

Three of those factors are what makes cavitation noise behave the way it
does. $(x_F - x_{Fzp1})^{1,5}$ starts the term at exactly zero on
the threshold, so the two regimes meet without a step;
$(x_F/x_{Fzp1})^5$ then makes it climb almost vertically once the
threshold is passed; and $(1-x_F)^{-0,5}$ sends it towards
infinity as the valve approaches flashing, which is where the method
stops.

**Parameters**

| Name | Description |
| :--- | :--- |
| `turbulent` | $\eta_{turb}$ from [`turbulent_efficiency`](/phonometry/reference/api/noise_control/valves-hydrodynamic/#turbulent_efficiency). |
| `differential` | $p_1 - p_2$, in Pa. |
| `choked_differential` | $\Delta p_c$ from [`cavitation_differential`](/phonometry/reference/api/noise_control/valves-hydrodynamic/#cavitation_differential), in Pa. |
| `pressure_ratio` | $x_F$ of Equation (1). |
| `corrected_ratio` | $x_{Fzp1}$ of Equation (3c). |

**Returns:** $\eta_{cav}$, dimensionless, and exactly zero on the threshold.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a value is not positive and finite, if the operating point is below the threshold, or if it is at or past flashing, where the equation has no value. |

## CAVITATION_FLOOR_WIDTH

*Constant* (`float`).

```python
CAVITATION_FLOOR_WIDTH = 0.1
```

## cavitation_peak_frequency

```python
cavitation_peak_frequency(
    turbulent_peak: float,
    pressure_ratio: float,
    corrected_ratio: float,
) -> float
```

Equation (13): the peak frequency of the cavitation noise.

$$
f_{p,cav} = 6 f_{p,turb} \left(\frac{1 - x_F}{1 - x_{Fzp1}}\right)^{2} \left(\frac{x_{Fzp1}}{x_F}\right)^{2,5}
$$

Both brackets are the reciprocals of the ones in Equation (9), and that
is deliberate rather than a misprint: the same factors that make the
cavitation *level* rise as the valve is opened further into cavitation
make its peak frequency fall, because the bubbles grow larger and take
longer to collapse. Just past the threshold the collapse is fast and the
noise is hissy, six times the turbulent peak; deep into cavitation it
drops back down into a rumble.

**Parameters**

| Name | Description |
| :--- | :--- |
| `turbulent_peak` | $f_{p,turb}$ from [`turbulent_peak_frequency`](/phonometry/reference/api/noise_control/valves-hydrodynamic/#turbulent_peak_frequency), in Hz. |
| `pressure_ratio` | $x_F$ of Equation (1). |
| `corrected_ratio` | $x_{Fzp1}$ of Equation (3c). |

**Returns:** $f_{p,cav}$, in Hz.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a value is not positive and finite, or the operating point is at or past flashing. |

## cavitation_transmission_loss

```python
cavitation_transmission_loss(
    turbulent_loss: float,
    *,
    turbulent_peak: float,
    cavitation_peak: float,
    efficiency_ratio: float,
    pressure_ratio: float | None = None,
    corrected_ratio: float | None = None,
) -> float
```

Equation (17): the transmission loss once the valve cavitates.

$$
TL_{cav} = TL_{turb} + 10 \lg\left( 250\, \frac{f_{p,cav}^{1,5}}{f_{p,turb}^{2}}\, \frac{\eta_{cav}}{\eta_{turb} + \eta_{cav}}\right)
$$

Cavitation noise peaks higher in frequency than turbulent noise, and the
pipe wall passes high frequencies better, so the correction is normally
positive: the wall becomes *less* effective when the valve cavitates,
which is one reason cavitating valves are heard from far away.

The NOTE to the equation floors the efficiency ratio at
$f_{p,turb}^2/(250 f_{p,cav}^{1,5})$ while $x_F$ is within
0,1 of the threshold, which is exactly the value that makes the bracket
equal 1. Just above incipient cavitation, where the cavitating efficiency
is still a small fraction of the total, the floor therefore keeps the
cavitating transmission loss from falling below the turbulent one. Pass
both ratios to apply it; leave them out to evaluate the equation as
printed.

**Parameters**

| Name | Description |
| :--- | :--- |
| `turbulent_loss` | $TL_{turb}$ of Equation (16a), in dB. |
| `turbulent_peak` | $f_{p,turb}$, in Hz. |
| `cavitation_peak` | $f_{p,cav}$, in Hz. |
| `efficiency_ratio` | $\eta_{cav}/(\eta_{turb}+\eta_{cav})$. |
| `pressure_ratio` | $x_F$, for the NOTE's floor. |
| `corrected_ratio` | $x_{Fzp1}$, for the NOTE's floor. |

**Returns:** $TL_{cav}$, in dB, negative.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a value is not positive and finite, or only one of the two ratios the floor needs was given. |

## combine_stage_levels

```python
combine_stage_levels(*levels: float) -> float
```

Equation (27): the stages of a multistage trim, added.

$$
L_{pAe,1m} = 10 \lg \sum_{i=1}^{n} 10^{0,1 L_{pAe,1m,i}}
$$

6.3.1 calculates each stage as if it were a valve of its own and sums
them in energy here. That is the branch for a trim whose stages all
radiate into the pipe, Figures 1 and 3; the fixed device of 6.3.2 with
increasing flow areas does not use it, because everything but the last
stage is absorbed inside the trim.

**Parameters**

| Name | Description |
| :--- | :--- |
| `levels` | $L_{pAe,1m,i}$, one per stage, in dB. |

**Returns:** Their energy sum, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If fewer than two levels are given, or one is not finite. |

## corrected_incipient_ratio

```python
corrected_incipient_ratio(
    incipient_ratio: float,
    inlet_pressure: float,
) -> float
```

Equation (3c): the threshold moved to the working inlet pressure.

$$
x_{Fzp1} = x_{Fz} \left(\frac{6 \times 10^5}{p_1}\right)^{0,125}
$$

Equation (3a) and the charts of Figures 4 to 9 are drawn at 6 × 10⁵ Pa.
Raising the inlet pressure lowers the threshold, because the same
differential ratio now means a larger absolute pressure drop and a
livelier vena contracta, but the eighth-power root makes it a slow
correction: ten times the inlet pressure moves the threshold by a
quarter.

**Parameters**

| Name | Description |
| :--- | :--- |
| `incipient_ratio` | $x_{Fz}$ at 6 × 10⁵ Pa, measured or from [`incipient_cavitation_ratio`](/phonometry/reference/api/noise_control/valves-hydrodynamic/#incipient_cavitation_ratio). |
| `inlet_pressure` | $p_1$, absolute, in Pa. |

**Returns:** $x_{Fzp1}$, dimensionless.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a value is not positive and finite, or the threshold is at or above 1, where the method has already stopped. |

## differential_pressure_ratio

```python
differential_pressure_ratio(
    *,
    inlet_pressure: float,
    outlet_pressure: float,
    vapour_pressure: float,
) -> float
```

Equation (1): the differential pressure ratio.

$$
x_F = \frac{p_1 - p_2}{p_1 - p_v}
$$

The denominator is the differential the valve would need to take the
liquid all the way down to its vapour pressure, so $x_F$ says how
far towards flashing this operating point is, and 1 is the whole way.

**Parameters**

| Name | Description |
| :--- | :--- |
| `inlet_pressure` | $p_1$, absolute, in Pa. |
| `outlet_pressure` | $p_2$, absolute, in Pa. |
| `vapour_pressure` | $p_v$ of the liquid at the inlet temperature, absolute, in Pa. |

**Returns:** $x_F$, dimensionless.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a pressure is not positive and finite, if the valve does not drop pressure, or if the inlet is already at the vapour pressure. |

## HydrodynamicValveNoise

```python
HydrodynamicValveNoise(
    regime: str,
    pressure_ratio: float,
    differential: float,
    cavitation_differential: float,
    incipient_ratio: float,
    corrected_ratio: float,
    jet_diameter: float,
    velocity: float,
    stream_power: float,
    turbulent_efficiency: float,
    cavitation_efficiency: float | None,
    sound_power: float,
    internal_level: float,
    strouhal_number: float,
    turbulent_peak: float,
    cavitation_peak: float | None,
    pipe_ring_frequency: float,
    reference_transmission_loss: float,
    turbulent_transmission_loss: float,
    cavitation_transmission_loss: float | None,
    transmission_loss: float,
    external_level: float,
    frequency: NDArray[np.float64],
    band_internal_level: NDArray[np.float64],
    band_transmission_loss: NDArray[np.float64],
    band_external_level: NDArray[np.float64],
)
```

What IEC 60534-8-4 says about one operating point on a liquid line.

**Attributes**

| Name | Description |
| :--- | :--- |
| `regime` | `"turbulent"` or `"cavitating"`, from the test of 5.1: the valve cavitates when $p_1 - p_2$ exceeds $x_{Fzp1}(p_1 - p_v)$. |
| `pressure_ratio` | $x_F$ of Equation (1). |
| `differential` | $p_1 - p_2$, in Pa. |
| `cavitation_differential` | $\Delta p_c$ of Equation (2), in Pa. It stops following the differential once the flow chokes. |
| `incipient_ratio` | $x_{Fz}$, the threshold as given, at 6 × 10⁵ Pa. |
| `corrected_ratio` | $x_{Fzp1}$ of Equation (3c), the threshold at the working inlet pressure. This is the number the regime test is made against. |
| `jet_diameter` | $D_j$ of Equation (4), in m. |
| `velocity` | $U_{vc}$ of Equation (5), in m/s. |
| `stream_power` | $W_m$ of Equation (6), in W. |
| `turbulent_efficiency` | $\eta_{turb}$ of Equation (8). |
| `cavitation_efficiency` | $\eta_{cav}$ of Equation (9), or `None` in the turbulent regime. |
| `sound_power` | $W_a$ of Equation (7a) or (7b), in W. |
| `internal_level` | $L_{pi}$ of Equation (10), in dB. |
| `strouhal_number` | $N_{STR}$ of Equation (12). |
| `turbulent_peak` | $f_{p,turb}$ of Equation (11), in Hz. |
| `cavitation_peak` | $f_{p,cav}$ of Equation (13), in Hz, or `None` in the turbulent regime. |
| `pipe_ring_frequency` | $f_r$ of Equation (14), in Hz. |
| `reference_transmission_loss` | $TL_{fr}$ of Equation (15), in dB, negative. |
| `turbulent_transmission_loss` | $TL_{turb}$ of Equation (16a), in dB. |
| `cavitation_transmission_loss` | $TL_{cav}$ of Equation (17), in dB, or `None` in the turbulent regime. |
| `transmission_loss` | whichever of the two the regime calls for, which is what Equation (18a) or (18b) uses. |
| `external_level` | $L_{pAe,1m}$ of Equation (18a) or (18b), in dB at 1 m from the pipe wall. The standard calls it A-weighted, but neither equation applies a weighting: the label describes what the fit was made against, not an operation on this number. |
| `frequency` | The band centres of 5.4.1, in Hz. |
| `band_internal_level` | $L_{pi}(f_i)$ of Equation (19a) or (19b), in dB. |
| `band_transmission_loss` | $TL(f_i)$ of Equation (22a), in dB. |
| `band_external_level` | $L_{pe,1m}(f_i)$ of Equation (21), in dB, unweighted. |

## incipient_cavitation_ratio

```python
incipient_cavitation_ratio(
    flow_coefficient: float,
    style_modifier: float,
    pressure_recovery: float,
    *,
    coefficient: str = 'Cv',
) -> float
```

Equation (3a): where cavitation becomes audible, estimated.

$$
x_{Fz} = \frac{0{,}90} {\sqrt{1 + 3 F_d \sqrt{\dfrac{C}{N_{34} F_L}}}}
$$

4.2 asks for a measured $x_{Fz}$ (IEC 60534-8-2) and offers this
only as an estimate, warning that a prediction built on it "can create
uncertainties as illustrated in Annex A". Annex A's third example is
exactly that illustration: 0,1 on this number moves the answer 14 dB.

The nesting is worth reading twice. The outer radical covers the whole of
$1 + 3F_d\sqrt{\cdots}$; the inner one covers only the capacity
group. A valve with a small style modifier, a cage full of small holes,
keeps $x_{Fz}$ high and stays quiet longer.

**Parameters**

| Name | Description |
| :--- | :--- |
| `flow_coefficient` | $C$ at the travel being examined. |
| `style_modifier` | $F_d$, the valve style modifier, taken from IEC 60534-8-3 (4.3 prints no table of its own). |
| `pressure_recovery` | $F_L$, dimensionless. |
| `coefficient` | Which flow coefficient `flow_coefficient` is, `"Cv"` or `"Kv"`, which selects $N_{34}$ from Table 1. |

**Returns:** $x_{Fz}$ at an inlet pressure of 6 × 10⁵ Pa, dimensionless.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a value is not positive and finite, or the coefficient is not one Table 1 prints a constant for. |

## internal_sound_pressure_level

```python
internal_sound_pressure_level(
    *,
    sound_power: float,
    density: float,
    sound_speed: float,
    internal_diameter: float,
) -> float
```

Equation (10): the level inside, at the pipe wall.

$$
L_{pi} = 10 \lg\left( \frac{3{,}2 \times 10^9\, W_a\, \rho_L\, c_L}{D_i^2}\right)
$$

The sound power is spread over the pipe cross-section and turned into a
pressure through the impedance of the liquid, which is why the density
and the speed of sound multiply rather than divide: water's impedance is
3 400 times air's, so the same acoustic power makes a level some 35 dB
higher inside a water line than inside an air line. Levels of 150 dB in
the pipe are ordinary here, and it is the transmission loss, not the
source, that makes the outside habitable.

In the printed equation the density has lost its Greek base glyph and
reads as a bare subscript; Table A.1 prints the same equation with
$\rho_L$ intact, which settles it.

**Parameters**

| Name | Description |
| :--- | :--- |
| `sound_power` | $W_a$ of Equation (7a) or (7b), in W. |
| `density` | $\rho_L$ of the liquid, in kg/m³. |
| `sound_speed` | $c_L$ in the liquid, in m/s. |
| `internal_diameter` | $D_i$ of the downstream pipe, in m. |

**Returns:** $L_{pi}$, in dB re 2 × 10⁻⁵ Pa.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a value is not positive and finite. |

## jet_strouhal_number

```python
jet_strouhal_number(
    *,
    flow_coefficient: float,
    style_modifier: float,
    pressure_recovery: float,
    corrected_ratio: float,
    valve_diameter: float,
    seat_diameter: float,
    inlet_pressure: float,
    vapour_pressure: float,
    coefficient: str = 'Cv',
    form: str = 'annex',
) -> float
```

Equation (12): the Strouhal number of the jet.

$$
N_{STR} = \frac{0{,}036\, F_L^2\, C\, F_d^{0,75}} {N_{34}\, x_{Fzp1}^{1,5}\, d\, d_o} \left(\frac{1}{p_1 - p_v}\right)^{0,57}
$$

This is the one place where the two printings of the standard disagree
with each other. The form above is Table A.1's; Clause 5.1 prints the
same equation with a leading 0,02 and **no** $F_d^{0,75}$. Only the
annex form reproduces the annex's own $N_{Str} = 0{,}399$, so it is
the default here; pass `form="clause"` for the normative text's version
and see `docs/ERRATA.md`.

Unlike the Strouhal number of a free jet, which is a constant near 0,2,
this one is a fitted group that carries the whole geometry of the valve
and comes out anywhere between about 0,2 and 0,5.

**Parameters**

| Name | Description |
| :--- | :--- |
| `flow_coefficient` | $C$ at the travel being examined. |
| `style_modifier` | $F_d$, used only by the `"annex"` form. |
| `pressure_recovery` | $F_L$, dimensionless. |
| `corrected_ratio` | $x_{Fzp1}$ of Equation (3c). |
| `valve_diameter` | $d$, the valve inlet internal diameter, in m. |
| `seat_diameter` | $d_o$, the seat or orifice diameter, in m. |
| `inlet_pressure` | $p_1$, absolute, in Pa. |
| `vapour_pressure` | $p_v$, absolute, in Pa. |
| `coefficient` | `"Cv"` or `"Kv"`, selecting $N_{34}$. |
| `form` | Which printing of Equation (12) to use, `"annex"` or `"clause"`. |

**Returns:** $N_{STR}$, dimensionless.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a value is not positive and finite, if the inlet is at or below the vapour pressure, or if a choice is not one the standard prints. |

## last_stage_differential

```python
last_stage_differential(
    *,
    inlet_pressure: float,
    outlet_pressure: float,
    vapour_pressure: float,
    corrected_ratio: float,
) -> float
```

Equation (28): the differential of the last stage of a fixed device.

$$
\Delta p_c = \min\left[ (p_{1,n} - p_2),\; x_{Fzp1,n}(p_{1,n} - p_v)\right]
$$

This is **not** Equation (2) with different symbols. Equation (2) caps
the differential at the choking point, $F_L^2(p_1 - p_v)$; this one
caps it at the *cavitation threshold* of the last stage,
$x_{Fzp1,n}(p_{1,n} - p_v)$, which is a smaller number. A fixed
multistage device is designed so that the last stage never cavitates, and
the cap says so.

**Parameters**

| Name | Description |
| :--- | :--- |
| `inlet_pressure` | $p_{1,n}$ of the last stage, in Pa. |
| `outlet_pressure` | $p_2$ at the valve outlet, in Pa. |
| `vapour_pressure` | $p_v$, in Pa. |
| `corrected_ratio` | $x_{Fzp1,n}$ of the last stage. |

**Returns:** $\Delta p_c$, in Pa.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a value is not positive and finite, or the last stage does not drop pressure. |

## last_stage_seat_diameter_mm

```python
last_stage_seat_diameter_mm(
    flow_coefficient: float,
    *,
    coefficient: str = 'Cv',
) -> float
```

6.3.2 b): the seat diameter of the last stage, estimated.

$$
d_o = 5{,}2 \sqrt{N_{34}\, C_n}
$$

The one display formula in the standard that carries no equation number,
and the one whose unit does not survive its own arithmetic: Clause 3
declares $d_o$ in metres, and for any real last stage this returns
tens. It is millimetres, which is why the unit is in the name of this
function; see `docs/ERRATA.md`. Equation (12) then wants the result in
metres, so divide by 1 000 before passing it on.

**Parameters**

| Name | Description |
| :--- | :--- |
| `flow_coefficient` | $C_n$ of the exit stage. |
| `coefficient` | `"Cv"` or `"Kv"`, selecting $N_{34}$. |

**Returns:** $d_o$, in **millimetres**.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the coefficient is not positive and finite, or is not one Table 1 prints a constant for. |

## mechanical_stream_power

```python
mechanical_stream_power(
    mass_flow: float,
    velocity: float,
    pressure_recovery: float,
) -> float
```

Equation (6): the stream power the valve dissipates.

$$
W_m = \frac{\dot m\, U_{vc}^2\, F_L^2}{2}
$$

The kinetic power of the jet, $\dot m U_{vc}^2/2$, scaled back by
$F_L^2$ to the part of it that is actually thrown away rather than
recovered as pressure downstream. Equation (7a) then takes a part in
$10^{-6}$ of this and calls it sound.

**Parameters**

| Name | Description |
| :--- | :--- |
| `mass_flow` | $\dot m$, in kg/s. |
| `velocity` | $U_{vc}$ from [`vena_contracta_velocity`](/phonometry/reference/api/noise_control/valves-hydrodynamic/#vena_contracta_velocity), in m/s. |
| `pressure_recovery` | $F_L$, dimensionless. |

**Returns:** $W_m$, in W.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a value is not positive and finite, or the recovery factor is outside its range. |

## multihole_incipient_cavitation_ratio

```python
multihole_incipient_cavitation_ratio(
    passages: int,
    hole_diameter: float,
    pressure_recovery: float,
) -> float
```

Equation (3b): the same threshold for a multihole trim.

$$
x_{Fz} = \frac{1} {\sqrt{4{,}5 + 1\,650\,\dfrac{N_o d_H^2}{F_L}}}
$$

A multihole trim is not described by its capacity and style modifier but
by how many holes it has and how big they are, which is what this form
takes. The group $N_o d_H^2$ is the total hole area to within
$\pi/4$, so two trims with the same open area and different hole
counts get the same threshold here.

**Parameters**

| Name | Description |
| :--- | :--- |
| `passages` | $N_o$, the number of independent, identical flow passages. |
| `hole_diameter` | $d_H$, the hole diameter, in m. |
| `pressure_recovery` | $F_L$, dimensionless. |

**Returns:** $x_{Fz}$ at an inlet pressure of 6 × 10⁵ Pa, dimensionless.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the passage count is not a whole number of one or more, or another value is not positive and finite. |

## pipe_ring_frequency

```python
pipe_ring_frequency(
    internal_diameter: float,
    *,
    pipe_sound_speed: float = 5000.0,
) -> float
```

Equation (14): the ring frequency of the pipe.

$$
f_r = \frac{c_p}{\pi D_i}
$$

The frequency at which one wavelength of a compressional wave in the wall
material wraps exactly once around the circumference. The wall is at its
most transparent there, so the transmission loss of Equation (15) is
anchored at this frequency and Equations (16b) and (22b) only ever make
it worse.

**Parameters**

| Name | Description |
| :--- | :--- |
| `internal_diameter` | $D_i$, in m. |
| `pipe_sound_speed` | $c_p$, 5 000 m/s for steel, in m/s. |

**Returns:** $f_r$, in Hz.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a value is not positive and finite. |

## REFERENCE_INLET_PRESSURE_PA

*Constant* (`float`).

```python
REFERENCE_INLET_PRESSURE_PA = 600000.0
```

## reference_transmission_loss

```python
reference_transmission_loss(
    internal_diameter: float,
    wall_thickness: float,
    *,
    pipe_density: float,
    pipe_sound_speed: float = 5000.0,
    air_density: float = 1.293,
    air_sound_speed: float = 343.0,
) -> float
```

Equation (15): the transmission loss at the ring frequency.

$$
TL_{fr} = -10 - 10 \lg\left( \frac{c_p \rho_p t_p}{c_o \rho_o D_i}\right)
$$

A mass law written as a ratio of two impedances: the wall's, per unit
area, against the air's, scaled by how much wall there is per unit bore.
Both terms are negative, and the standard keeps them that way, so this
quantity is a **negative number that is added** to the internal level all
the way to Equation (18). A DN 100 steel pipe with a 3,6 mm wall comes
out at −44,7 dB.

**Parameters**

| Name | Description |
| :--- | :--- |
| `internal_diameter` | $D_i$, in m. |
| `wall_thickness` | $t_p$, in m. |
| `pipe_density` | $\rho_p$, 7 800 kg/m³ for steel. |
| `pipe_sound_speed` | $c_p$, 5 000 m/s for steel, in m/s. |
| `air_density` | $\rho_o$ outside the pipe, in kg/m³. |
| `air_sound_speed` | $c_o$ outside the pipe, in m/s. |

**Returns:** $TL_{fr}$, in dB, negative.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a value is not positive and finite. |

## stage_conditions

```python
stage_conditions(
    *,
    inlet_pressure: float,
    outlet_pressure: float,
    vapour_pressure: float,
    stage_coefficients: Sequence[float],
    flow_coefficient: float,
) -> tuple[StageConditions, ...]
```

Equations (23a) to (24b) and (26): the differential, stage by stage.

$$
p_{1,1} = p_1, \qquad p_{1,i} = p_{1,i-1} - \frac{p_1 - p_2}{\left(C_{i-1}/C\right)^2}, \qquad p_{2,i} = p_{1,i+1}, \qquad p_{2,n} = p_2
$$

Each stage takes a share of the total differential in inverse proportion
to the square of its own capacity, which is the series law for flow
resistances: $1/C^2 = \sum_i 1/C_i^2$. A trim whose stages all have
the same $C_i$ splits the drop evenly; one with an increasing flow
area, the device of 6.3.2 and Figure 2, puts most of the drop in the
first stages and leaves the last one working at a differential small
enough not to cavitate.

Equation (23b) is printed with $p_{1,i+1}$ on the right, which
would compute each stage's inlet from the **next** stage's and run the
pressure backwards along the trim, contradicting (23a). The recursion
implemented here is the forward one the index $C_{i-1}$ calls for;
see `docs/ERRATA.md`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `inlet_pressure` | $p_1$ at the valve, absolute, in Pa. |
| `outlet_pressure` | $p_2$ at the valve, absolute, in Pa. |
| `vapour_pressure` | $p_v$, absolute, in Pa. |
| `stage_coefficients` | $C_i$, the rated flow coefficient of each stage in flow order, two or more of them. |
| `flow_coefficient` | $C$ of the whole valve, in the same units. |

**Returns:** One [`StageConditions`](/phonometry/reference/api/noise_control/valves-hydrodynamic/#stageconditions) per stage, in flow order.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a value is not positive and finite, if fewer than two stages were given, or if the stages between them would take more than the differential the valve has. |
| ValveNoiseWarning | If the stage coefficients miss the series law by more than 5 %, which leaves the last stage carrying a differential nobody chose. |

## StageConditions

```python
StageConditions(
    inlet_pressure: float,
    outlet_pressure: float,
    pressure_ratio: float,
)
```

What one throttling stage of a multistage trim sees.

**Attributes**

| Name | Description |
| :--- | :--- |
| `inlet_pressure` | $p_{1,i}$ of Equations (23a) and (23b), in Pa. |
| `outlet_pressure` | $p_{2,i}$ of Equations (24a) and (24b), in Pa. |
| `pressure_ratio` | $x_{F,i}$ of Equation (26), the stage's own differential pressure ratio, which 6.3 tests against that stage's $x_{Fzp1,i}$. |

## STROUHAL_CONSTANTS

*Constant* (`dict`).

```python
STROUHAL_CONSTANTS = {'annex': 0.036, 'clause': 0.02}
```

## transmission_loss_correction

```python
transmission_loss_correction(
    frequency: ArrayLike,
    ring: float,
) -> NDArray[np.float64]
```

Equations (16b) and (22b): how far the wall is from its ring.

$$
\Delta TL(f) = -20 \log\left[ \left(\frac{f_r}{f}\right) + \left(\frac{f}{f_r}\right)^{1,5}\right]
$$

One expression covers both printed equations: (16b) evaluates it at the
turbulent peak frequency and (22b) at each band. The bracket is a sum of
two branches, one falling as $1/f$ and one rising as
$f^{1,5}$, so the correction is worst far from the ring frequency
on either side. It is never zero: where the two branches together are
smallest, at $(2/3)^{0,4} f_r = 0{,}85 f_r$, the bracket is still
1,96 and the correction still costs 5,85 dB, and at $f_r$ itself it
costs 6,02.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | $f$, in Hz. A scalar or a 1-D array. |
| `ring` | $f_r$ from [`pipe_ring_frequency`](/phonometry/reference/api/noise_control/valves-hydrodynamic/#pipe_ring_frequency), in Hz. |

**Returns:** $\Delta TL$, in dB, one value per frequency, and always negative: the correction is worth at least 5,85 dB even where it is smallest.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a value is not positive and finite. |

## turbulent_distribution

```python
turbulent_distribution(
    frequency: ArrayLike,
    peak: float,
) -> NDArray[np.float64]
```

Equation (20a): how turbulent noise spreads over the bands.

$$
F_{turb}(f_i) = -10 \lg\left[ \frac{1}{4}\left(\frac{f_i}{f_{p,turb}}\right)^{3} + \left(\frac{f_i}{f_{p,turb}}\right)^{-1}\right] - 3{,}1
$$

A band correction, in dB, that adds to the overall internal level. The
two terms in the bracket are the two sides of the peak: below it the
$f^{-1}$ term dominates and the level rises at 3 dB per octave;
above it the $f^{3}$ term takes over and the level falls at 9 dB
per octave. The trailing 3,1 dB is a printed offset and not a
normalisation: over the band set of 5.4.1 these corrections do not sum
back to $L_{pi}$, they sum about 5 dB above it, so the band route
and the overall route of Equation (18a) are two answers and not one
answer twice. The maximum is not exactly at $f_{p,turb}$ either:
the quarter in front of the rising branch puts it at
$(4/3)^{1/4} f_{p,turb}$, a few per cent above.

The negative exponent is easy to lose. Text extracted from the printed
page renders it as a bare 1, which flattens the low-frequency side.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | $f_i$, the band centres, in Hz. |
| `peak` | $f_{p,turb}$ from [`turbulent_peak_frequency`](/phonometry/reference/api/noise_control/valves-hydrodynamic/#turbulent_peak_frequency), in Hz. |

**Returns:** $F_{turb}(f_i)$, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a value is not positive and finite. |

## turbulent_efficiency

```python
turbulent_efficiency(velocity: float, sound_speed: float) -> float
```

Equation (8): the acoustical efficiency of the turbulent jet.

$$
\eta_{turb} = 10^{-4}\left(\frac{U_{vc}}{c_L}\right)
$$

5.1 argues the case: at these velocities the jet is slow enough to be a
monopole, and a monopole's efficiency rises with the first power of the
Mach number, reaching $10^{-4}$ when the jet reaches the speed of
sound in the liquid. Water carries sound at about 1 400 m/s and a control
valve jet runs at tens of metres per second, so the efficiency comes out
in the $10^{-6}$ range: one part in a million of the stream power.

**Parameters**

| Name | Description |
| :--- | :--- |
| `velocity` | $U_{vc}$, in m/s. |
| `sound_speed` | $c_L$ in the liquid, in m/s. |

**Returns:** $\eta_{turb}$, dimensionless.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a value is not positive and finite. |

## turbulent_peak_frequency

```python
turbulent_peak_frequency(
    strouhal_number: float,
    velocity: float,
    jet: float,
) -> float
```

Equation (11): the peak frequency of the turbulent noise.

$$
f_{p,turb} = N_{STR}\, \frac{U_{vc}}{D_j}
$$

A jet radiates around the frequency at which its own eddies pass a fixed
point, which is the velocity divided by the size of the eddies. The jet
diameter of Equation (4) stands for that size.

**Parameters**

| Name | Description |
| :--- | :--- |
| `strouhal_number` | $N_{STR}$ from [`jet_strouhal_number`](/phonometry/reference/api/noise_control/valves-hydrodynamic/#jet_strouhal_number). |
| `velocity` | $U_{vc}$, in m/s. |
| `jet` | $D_j$ of Equation (4), in m. |

**Returns:** $f_{p,turb}$, in Hz.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a value is not positive and finite. |

## uniform_passage_style_modifier

```python
uniform_passage_style_modifier(passages: int) -> float
```

Equation (29): the style modifier of a last stage full of openings.

$$
F_d = \sqrt{\frac{1}{N_o}}
$$

IEC 60534-8-3 defines $F_d$ as the hydraulic diameter of one
passage over the diameter of the single orifice of the same total area.
For $N_o$ identical round openings that ratio collapses to
$1/\sqrt{N_o}$, which is what this equation prints. Sixteen
openings therefore give a quarter of the jet diameter, a sixteenth of the
jet area, and a peak frequency four times higher.

**Parameters**

| Name | Description |
| :--- | :--- |
| `passages` | $N_o$, the number of uniform openings within the last stage. |

**Returns:** $F_d$, dimensionless.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the count is not a whole number of one or more. |

## valve_hydrodynamic_noise

```python
valve_hydrodynamic_noise(
    *,
    mass_flow: float,
    inlet_pressure: float,
    outlet_pressure: float,
    vapour_pressure: float,
    liquid_density: float,
    liquid_sound_speed: float,
    flow_coefficient: float,
    style_modifier: float,
    pressure_recovery: float,
    incipient_ratio: float,
    power_ratio: float,
    valve_diameter: float,
    seat_diameter: float,
    internal_diameter: float,
    wall_thickness: float,
    pipe_density: float,
    coefficient: str = 'Cv',
    strouhal_form: str = 'annex',
    frequency: ArrayLike | None = None,
    pipe_sound_speed: float = 5000.0,
    air_density: float = 1.293,
    air_sound_speed: float = 343.0,
) -> HydrodynamicValveNoise
```

The whole of Clauses 4 and 5, from the operating point to 1 m.

The chain is the standard's own: the pressure ratios of 4.1 and 4.2, the
geometry and the stream power of 4.4 to 4.6, the regime test and the two
efficiencies of 5.1, the pipe transmission loss of 5.2, the external
level of 5.3, and the band route of 5.4 alongside it.

Which regime the valve is in is decided once, on
$p_1 - p_2$ against $x_{Fzp1}(p_1 - p_v)$, and it selects the
sound power of Equation (7a) or (7b), the transmission loss of (16a) or
(17), the external level of (18a) or (18b), and the band spectrum of
(19a) or (19b) together. On the threshold itself Equation (9) returns
exactly zero, so the two branches meet without a step.

**Parameters**

| Name | Description |
| :--- | :--- |
| `mass_flow` | $\dot m$, in kg/s. |
| `inlet_pressure` | $p_1$, absolute, in Pa. |
| `outlet_pressure` | $p_2$, absolute, in Pa. |
| `vapour_pressure` | $p_v$ of the liquid, absolute, in Pa. |
| `liquid_density` | $\rho_L$, in kg/m³. |
| `liquid_sound_speed` | $c_L$, in m/s. |
| `flow_coefficient` | $C$ at the travel being examined. |
| `style_modifier` | $F_d$, from IEC 60534-8-3. |
| `pressure_recovery` | $F_L$, dimensionless. |
| `incipient_ratio` | $x_{Fz}$ at 6 × 10⁵ Pa, measured to IEC 60534-8-2 or estimated with [`incipient_cavitation_ratio`](/phonometry/reference/api/noise_control/valves-hydrodynamic/#incipient_cavitation_ratio). Equation (3c) corrects it here. |
| `power_ratio` | $r_W$ from Table 2, the share of the sound power radiated into the pipe. See [`ACOUSTIC_POWER_RATIOS`](/phonometry/reference/api/noise_control/valves-hydrodynamic/#acoustic_power_ratios). |
| `valve_diameter` | $d$, the valve inlet internal diameter, in m. |
| `seat_diameter` | $d_o$, in m. |
| `internal_diameter` | $D_i$ of the downstream pipe, in m. |
| `wall_thickness` | $t_p$, in m. |
| `pipe_density` | $\rho_p$, in kg/m³. |
| `coefficient` | `"Cv"` or `"Kv"`. |
| `strouhal_form` | Which printing of Equation (12) to follow, `"annex"` or `"clause"`; see [`STROUHAL_CONSTANTS`](/phonometry/reference/api/noise_control/valves-hydrodynamic/#strouhal_constants). |
| `frequency` | The band centres to report, in Hz. The default is the one-third-octave set 5.4.1 prints, 50 Hz to 20 kHz. |
| `pipe_sound_speed` | $c_p$, in m/s. |
| `air_density` | $\rho_o$, in kg/m³. |
| `air_sound_speed` | $c_o$, in m/s. |

**Returns:** A [`HydrodynamicValveNoise`](/phonometry/reference/api/noise_control/valves-hydrodynamic/#hydrodynamicvalvenoise) carrying every printed intermediate as well as the level at 1 m.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a value is outside the range its equation is written for, or if the operating point is at or past flashing, where Equations (9) and (13) divide by zero. |

## vena_contracta_velocity

```python
vena_contracta_velocity(
    differential: float,
    density: float,
    pressure_recovery: float,
) -> float
```

Equation (5): the jet velocity.

$$
U_{vc} = \frac{1}{F_L}\sqrt{\frac{2 \Delta p_c}{\rho_L}}
$$

Bernoulli's velocity for the differential of Equation (2), divided by the
recovery factor because $F_L$ is defined as the fraction of the
ideal velocity head the valve actually reaches at the vena contracta.

**Parameters**

| Name | Description |
| :--- | :--- |
| `differential` | $\Delta p_c$ from [`cavitation_differential`](/phonometry/reference/api/noise_control/valves-hydrodynamic/#cavitation_differential), in Pa. |
| `density` | $\rho_L$ of the liquid, in kg/m³. |
| `pressure_recovery` | $F_L$, dimensionless. |

**Returns:** $U_{vc}$, in m/s.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a value is not positive and finite, or the recovery factor is outside its range. |
