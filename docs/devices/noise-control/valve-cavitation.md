← [Documentation index](../../README.md)

# Hydrodynamic valve noise (IEC 60534-8-4)

A liquid does not compress, so a control valve on a water line cannot build
the shock cells that [the aerodynamic part](control-valve-noise.md) spends
five regimes on. It makes noise two ways instead, and the whole of
IEC 60534-8-4 is the sum of them: **turbulence** in the jet leaving the vena contracta, and
**cavitation**, once the pressure there falls low enough for the liquid to
flash to vapour and the bubbles to collapse again downstream.

The difference between the two is not a matter of degree. Turbulent noise
follows the jet velocity gently; cavitation arrives at a threshold and then
climbs as a fifth power. Everything on this page turns on where the operating
point sits with respect to that threshold.

## 1. The threshold, and why the standard asks you to measure it

Two ratios decide the regime. The first is the **differential pressure
ratio**, Equation (1), which says how far towards flashing this operating
point is:

$$
x_F = \frac{p_1 - p_2}{p_1 - p_v}
$$

The second is the **characteristic pressure ratio** $x_{Fz}$: the value of
$x_F$ at which cavitation first becomes audible on this valve. It is a
property of the valve, measured to IEC 60534-8-2, and 4.2 offers Equation (3a)
only as an estimate for when no measurement exists. Equation (3c) then moves
it from the 6 × 10⁵ Pa the estimate is drawn at to the working inlet pressure.

```python
from phonometry import noise_control

x_f = noise_control.differential_pressure_ratio(
    inlet_pressure=1.0e6, outlet_pressure=6.5e5, vapour_pressure=2.32e3
)
print(round(x_f, 4))                                    # 0.3508

x_fz = noise_control.incipient_cavitation_ratio(90.0, 0.42, 0.92)
print(round(x_fz, 4))                                   # 0.2543  at 6e5 Pa
print(round(noise_control.corrected_incipient_ratio(x_fz, 1.0e6), 4))
#                                                       # 0.2386  at 10 bar
```

This valve is past its threshold, so it cavitates. A multihole trim of the
same capacity would not be: Equation (3b) answers on the hole geometry rather
than on the capacity, and a hundred and twenty holes of three millimetres push
the threshold up by half. The comparison has to be made against the **corrected**
value, which is the half of it a reader is most likely to skip.

```python
holes = noise_control.multihole_incipient_cavitation_ratio(120, 0.003, 0.92)
print(round(holes, 4))                                  # 0.3941  at 6e5 Pa
print(round(noise_control.corrected_incipient_ratio(holes, 1.0e6), 4))
#                                                       # 0.3698  still above x_F
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/valve_cavitation_noise_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/valve_cavitation_noise.svg" alt="Three panels. Left: the level one metre from the pipe against the differential pressure ratio, drawn twice for the same valve with characteristic pressure ratios 0,1 apart, each curve rising smoothly, stepping up sharply at its own threshold and flattening near 90 decibels, with the three worked examples marked. Middle: the sound pressure level in each one-third-octave band from 50 Hz to 20 kHz, showing the turbulent and cavitating shares of the internal spectrum, their sum, and the level one metre outside sixty to a hundred decibels below, with both peak frequencies marked. Right: the two acoustical efficiencies against the differential pressure ratio on a logarithmic axis, the turbulent one almost flat near one part in a million and the cavitating one rising from zero at the threshold to cross it and keep climbing" width="100%"></picture>

*The left panel is Annex A's own Figure A.1: the same valve at the same flow,
with a characteristic pressure ratio 0,1 apart, and 14 dB between the two
markers at $x_F = 0{,}35$.*

That 14 dB is the reason 4.2 warns that a prediction built on the estimate
"can create uncertainties". The estimate is not a small correction to the
answer; on a cavitating valve it **is** the answer.

## 2. Two efficiencies, and one stream power to share

The source half of the method is short. The differential of Equation (2) is
capped at the choking point, Equation (5) turns it into a jet velocity and
Equation (6) into the stream power the valve dissipates:

```python
choked = noise_control.cavitation_differential(
    inlet_pressure=1.0e6,
    outlet_pressure=6.5e5,
    vapour_pressure=2.32e3,
    pressure_recovery=0.92,
)
speed = noise_control.vena_contracta_velocity(choked, 997.0, 0.92)
power = noise_control.mechanical_stream_power(40.0, speed, 0.92)
print(round(speed, 3), round(power, 1))                 # 28.801 14042.1
```

Then two efficiencies say what fraction of those fourteen kilowatts leaves as
sound. Equation (8) makes the turbulent one linear in the jet Mach number,
reaching $10^{-4}$ only when the jet reaches the speed of sound in the liquid,
which for water is 1 400 m/s and never happens in a control valve:

$$
\eta_{turb} = 10^{-4}\left(\frac{U_{vc}}{c_L}\right)
$$

Equation (9) is the other one, and it is worth reading factor by factor:

$$
\eta_{cav} = 0{,}32\, \eta_{turb}
  \sqrt{\frac{p_1 - p_2}{\Delta p_c}\cdot\frac{1}{x_{Fzp1}}}\;
  e^{5 x_{Fzp1}}
  \left(\frac{1 - x_{Fzp1}}{1 - x_F}\right)^{0,5}
  \left(\frac{x_F}{x_{Fzp1}}\right)^{5}
  \left(x_F - x_{Fzp1}\right)^{1,5}
$$

Three of those factors do all the work. $(x_F - x_{Fzp1})^{1,5}$ starts the
term at **exactly zero** on the threshold, so the two regimes meet without a
step. $(x_F/x_{Fzp1})^5$ then makes it climb almost vertically. And
$(1 - x_F)^{-0,5}$ sends it towards infinity as the valve approaches flashing,
which is where the method stops.

```python
turbulent = noise_control.turbulent_efficiency(speed, 1400.0)
cavitating = noise_control.cavitation_efficiency(
    turbulent=turbulent,
    differential=3.5e5,
    choked_differential=choked,
    pressure_ratio=0.3508,
    corrected_ratio=0.2386,
)
print(f"{turbulent:.3e} {cavitating:.3e}")              # 2.057e-06 1.242e-06
```

Just past the threshold the cavitation term is already comparable with the
turbulent one, and Equation (7b) adds them before Table 2's acoustic power
ratio $r_W$ takes the share that is radiated into the pipe rather than lost in
the body: a quarter for every globe and rotary valve, a half for the
butterflies, one for an expander.

## 3. The whole chain in one call

```python
valve = dict(
    inlet_pressure=1.0e6,
    vapour_pressure=2.32e3,
    liquid_density=997.0,
    liquid_sound_speed=1400.0,
    flow_coefficient=90.0,
    style_modifier=0.42,
    pressure_recovery=0.92,
    power_ratio=0.25,
    valve_diameter=0.1,
    seat_diameter=0.1,
    internal_diameter=0.1071,
    wall_thickness=0.0036,
    pipe_density=7800.0,
)
res = noise_control.valve_hydrodynamic_noise(
    **valve, mass_flow=40.0, outlet_pressure=6.5e5, incipient_ratio=x_fz
)
print(res.regime)                                       # cavitating
print(round(res.sound_power, 5), "W")                   # 0.01158 W
print(round(res.internal_level, 3), "dB")               # 156.543 dB
print(round(res.turbulent_peak, 2), round(res.cavitation_peak, 2))
#                                                       # 654.35 1088.94
print(round(res.external_level, 1), "dB at 1 m")        # 81.0 dB at 1 m
```

Every one of those is a printed cell of Annex A's Table A.1. The same call on
the first column, 30 kg/s into 8 bar, stays turbulent and answers 62,7 dB; the
third column is the second one again with $x_{Fz}$ shifted by 0,1, and answers
66,9 dB.

The two peak frequencies are Equations (11) and (13), and (13) is the
surprising one: on the threshold the cavitating peak is **six times** the
turbulent peak, and it falls from there as the valve is opened further into
cavitation, because the bubbles grow larger and take longer to collapse. Just
past onset, cavitation is a hiss; deep into it, a rumble.

## 4. The pipe, and a printed sign that cannot be right

Inside the pipe this valve runs at 156 dB. What makes the outside habitable is
the wall, and the standard anchors its transmission loss at the **ring
frequency**, Equation (14), where one wavelength in the wall material wraps
exactly once around the circumference:

```python
print(round(res.pipe_ring_frequency, 1), "Hz")          # 14860.4 Hz
print(round(res.reference_transmission_loss, 2), "dB")  # -44.71 dB
print(round(res.transmission_loss, 2), "dB")            # -62.86 dB
```

Both terms of Equation (15) are printed with a minus sign, so the
transmission loss of this method is a **negative number that is added** all
the way to the end. Away from the ring frequency Equations (16b) and (22b)
only make it worse, and the peak frequencies of a control valve are more than
a decade below the ring frequency of its pipe, which is where the other 18 dB
of this valve's loss come from: Equation (16b) costs 27 dB at the turbulent
peak, and Equation (17) hands 9 of them back because the valve cavitates.

Band by band, 5.4 spreads the internal level with Equations (20a) and (20b)
and gives the wall a frequency-dependent loss with Equation (22a):

```python
import numpy as np

band = int(np.argmin(np.abs(res.frequency - 8000.0)))
print(round(float(res.band_internal_level[band]), 1))     # 141.9
print(round(float(res.band_transmission_loss[band]), 2))  # -51.76
print(round(float(res.band_external_level[band]), 1))     # 77.4
```

Those three are printed in Table A.1 as 141,9 dB, **51,76 dB** and 77,4 dB.
The middle one is printed without its minus sign, and it cannot be: its own
two inputs, $-44{,}71$ and $-7{,}053$ dB, sum to $-51{,}763$, and the row
below only reproduces its printed 77,4 dB with the negative value.
[The errata register](../../ERRATA.md) records it, along with three
intermediates of the same table that its own equations do not reproduce.

## 5. One equation, printed two ways

Equation (12), the Strouhal number that places the turbulent peak, appears
twice in the document and not identically. Clause 5.1 prints a leading 0,02
and no valve style modifier; Table A.1 prints 0,036 and a factor
$F_d^{0,75}$. Both are on the page, and they are not the same function of the
valve.

```python
form = dict(
    flow_coefficient=90.0,
    style_modifier=0.42,
    pressure_recovery=0.92,
    corrected_ratio=0.2386,
    valve_diameter=0.1,
    seat_diameter=0.1,
    inlet_pressure=1.0e6,
    vapour_pressure=2.32e3,
)
print(round(noise_control.jet_strouhal_number(**form, form="annex"), 3))   # 0.399
print(round(noise_control.jet_strouhal_number(**form, form="clause"), 3))  # 0.425
```

Only the annex form reproduces the annex's own printed $N_{Str} = 0{,}399$,
so it is the default here and the clause form is one keyword away. For this
valve the two differ by 6 %; for a single-port valve with $F_d = 1$ the annex
form is 80 % above the clause one, which is five sixths of an octave in the
peak frequency and a few decibels through the transmission loss.

## 6. Stage by stage

Clause 6 is the same method with per-stage inputs. Each stage takes a share of
the differential in inverse proportion to the square of its own flow
coefficient, which is the series law $1/C^2 = \sum_i 1/C_i^2$:

```python
stages = noise_control.stage_conditions(
    inlet_pressure=1.0e6,
    outlet_pressure=4.0e5,
    vapour_pressure=2.32e3,
    stage_coefficients=[130.0, 160.0, 199.1],
    flow_coefficient=90.0,
)
for stage in stages:
    print(round(stage.inlet_pressure), round(stage.pressure_ratio, 3))
# 1000000 0.288
# 712426 0.267
# 522582 0.236

print(round(noise_control.combine_stage_levels(78.0, 74.0, 71.0), 1))   # 80.0
```

Those three coefficients increase along the flow, which is the device of
Figure 2: most of the pressure is taken in the first stages and the last one
is left working at a differential small enough not to cavitate. 6.3.2 then
calculates **only** that last stage, because the sound the earlier ones make
is absorbed inside the trim before it reaches the pipe, and it caps the
differential at the last stage's own cavitation threshold rather than at the
choking point:

```python
print(noise_control.last_stage_differential(
    inlet_pressure=6.0e5,
    outlet_pressure=4.0e5,
    vapour_pressure=2.32e3,
    corrected_ratio=0.30,
))                                                      # 179304.0

print(noise_control.uniform_passage_style_modifier(16))            # 0.25
print(round(noise_control.last_stage_seat_diameter_mm(45.0), 1))   # 37.7
```

The last of those carries its unit in its name for a reason. The formula it
implements is the one display formula in the standard with no equation number,
$d_o = 5{,}2\sqrt{N_{34} C_n}$, and Clause 3 declares $d_o$ in metres, which
for any real last stage it cannot be: 37,7 of anything is millimetres. Divide
by a thousand before handing it to Equation (12).

## See also

- [Control Valve Noise (IEC 60534-8-3)](control-valve-noise.md): the same
  chain for a compressible fluid, where five flow regimes take the place of
  one threshold.
- [Reactive Silencers](silencers.md): the four-pole chain that would be put
  downstream of a line this loud.
- [Duct-Borne Noise](duct-path.md): the same question for a ventilation
  system, where the source is a fan rather than a jet.
- [Errata in published sources](../../ERRATA.md): the six entries this page
  rests on, from a sign to an equation printed two ways.
- API reference: [`noise_control.valves_hydrodynamic`](https://jmrplens.github.io/phonometry/reference/api/noise_control/valves-hydrodynamic/).

## Standards

IEC 60534-8-4:2005, Clauses 4, 5 and 6. The preliminary calculations of
Clause 4: the pressure ratio of Equation (1), the capped differential of (2),
the characteristic
pressure ratio of (3a) to (3c), the jet diameter of (4), the vena contracta
velocity of (5) and the stream power of (6). The prediction of Clause 5: the
two acoustical efficiencies of (8) and (9), the sound power of (7a) and (7b),
the internal level of (10), the Strouhal number and both peak frequencies of
(11) to (13), the transmission loss of (14) to (17) with the floor of the
NOTE, the external level of (18a) and (18b), and the band route of 5.4. And
Clause 6 for a multistage trim: the stage pressures and ratios of (23a) to
(26), the energy sum of (27), and the last stage of 6.3.2. Validated against
all three worked examples of Annex A in the
[conformance report](../../CONFORMANCE.md); the six defects that annex and
its clauses carry are in the [errata register](../../ERRATA.md). The
laboratory measurement of x_Fz (IEC 60534-8-2) and the flashing regime past
x_F = 1 are not implemented.
