← [Documentation index](README.md)

# Underwater sound propagation: transmission loss, sound speed, sonar, seabed and ambient noise

Closed-form underwater propagation: the **transmission loss** (geometrical
spreading plus volume absorption), the **speed of sound** in sea water, the
**sonar equation**, **seabed reflection loss** and the **ocean ambient-noise**
spectrum (wind, thermal and shipping-traffic contributions). These complement
the underwater reference levels (ISO 18405/17208/18406) in
[Underwater Acoustics](underwater-acoustics.md).

## 1. Transmission loss

The transmission loss is

$$
TL = \text{spreading} + \alpha R .
$$

Geometrical spreading is $20 \lg R$ (spherical), $10 \lg R$ (cylindrical) or
spherical up to a transition range $R_0$ and cylindrical beyond it
(`"practical"`). The volume absorption coefficient $\alpha$ (dB/km) comes from one of three models: **Francois–Garrison**
(1982, the default and reference), **Ainslie–McColm** (1998, a legible
simplification) or **Thorp** (1967, frequency-only).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/underwater_transmission_loss_dark.svg">
  <img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/underwater_transmission_loss.svg" alt="Underwater transmission loss versus range at 10 kHz: the total loss with the geometrical-spreading and volume-absorption contributions drawn separately, loss increasing downward" width="82%">
</picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import underwater

# 10 kHz at 10 °C, 35 ppt, 100 m depth; practical spreading with R0 = 1000 m.
ranges = np.linspace(10.0, 20_000.0, 400)
tl = underwater.transmission_loss(ranges, 10e3, law="practical",
                                  transition_range=1000.0, temperature=10.0,
                                  salinity=35.0, depth=100.0)
print(f"alpha = {tl.absorption_coefficient:.2f} dB/km")   # alpha = 0.95 dB/km
tl.plot()   # total TL with the spreading and absorption contributions
plt.show()
```

</details>

```python
import numpy as np
from phonometry import underwater

# Absorption coefficient at 10 kHz, 10 °C, 35 ppt, 100 m, pH 8 (dB/km).
# seawater_absorption accepts scalar or array frequencies and returns an array.
alpha = underwater.seawater_absorption(10e3, temperature=10.0, salinity=35.0,
                               depth=100.0, model="francois-garrison")[0]
print(f"α = {alpha:.3f} dB/km")

ranges = np.linspace(10.0, 20_000.0, 400)
tl = underwater.transmission_loss(ranges, 10e3, law="practical", transition_range=1000.0,
                          temperature=10.0, salinity=35.0, depth=100.0)
print(tl.absorption_coefficient, tl.tl[-1])
tl.plot()   # TL vs range with the spreading/absorption split (needs matplotlib)
```

`transmission_loss` returns a `TransmissionLossResult` with `tl`, `spreading`,
`absorption` (arrays), the `frequency` and the `absorption_coefficient`
(dB/km). Francois–Garrison and Ainslie–McColm agree to within ~10 % across
100 Hz–1 MHz, so either is a good cross-check of the other.

## 2. Speed of sound in sea water

`sea_water_sound_speed(T, S, depth, model=…)` evaluates the sound speed with the
**UNESCO / Chen–Millero** equation (default, the international standard, in the
Wong & Zhu 1995 ITS-90 form), **Del Grosso** (1974) or **Mackenzie** (1981). The
UNESCO and Del Grosso equations use pressure, so depth is first converted with
the Leroy & Parthiot (1998) formula (`depth_to_pressure`). The three agree to
within ~1 m/s in their common domain; Mackenzie's canonical check value is
`1550.744 m/s` at 25 °C, 35 ppt, 1000 m.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/underwater_sound_speed_dark.svg">
  <img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/underwater_sound_speed.svg" alt="A sea-water sound-speed profile computed with the UNESCO equation: a warm mixed layer near the surface, a thermocline where the speed drops, a sound-channel axis at the minimum, and the speed rising again with pressure at depth" width="62%">
</picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import underwater

# A warm mixed layer, a thermocline down to 4 °C and an isothermal deep layer.
depths = np.linspace(0.0, 3000.0, 121)
temps = 4.0 + 14.0 / (1.0 + (np.maximum(depths - 80.0, 0.0) / 250.0) ** 2)
profile = underwater.sound_speed_profile(depths, temps, 35.0, model="unesco")
profile.plot()   # sound speed vs depth, minimum at the sound-channel axis
plt.show()
```

</details>

```python
import numpy as np
from phonometry import underwater

c = underwater.sea_water_sound_speed(25.0, 35.0, 1000.0, model="mackenzie")  # 1550.744

# A profile: warm mixed layer, thermocline, isothermal deep layer.
depths = np.linspace(0.0, 3000.0, 121)
temps = 4.0 + 14.0 / (1.0 + (np.maximum(depths - 80.0, 0.0) / 250.0) ** 2)
profile = underwater.sound_speed_profile(depths, temps, 35.0, model="unesco")
profile.plot()   # sound speed vs depth (needs matplotlib)
```

The $c(z)$ minimum acts as a waveguide (the SOFAR channel): wavefronts that
stray from the axis are refracted back toward it, while sound generated
outside the channel leaks away to depth, as the simulation below shows with an
intentionally exaggerated gradient. This trapping is why low-frequency sound
can cross entire oceans.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_fdtd_ducting_dark.gif"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_fdtd_ducting.gif" alt="Animation: a 2D FDTD simulation of a low-frequency pulse in a SOFAR-like underwater sound channel with the sound-speed profile drawn beside the field; launched on the channel axis the wavefronts refract back toward the sound-speed minimum and stay trapped, launched near the surface the energy crosses the channel and leaks away to depth" width="640" height="360" loading="lazy"></picture>

[Watch the high-resolution video (WebM)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_fdtd_ducting.webm)

## 3. Sonar equation

The sonar equation combines the performance terms into the **signal excess**
$SE$ (detection when $SE \ge 0$) and the **figure of merit** (the maximum
allowable transmission loss at $SE = 0$):

$$
SE = SL - TL - (NL - DI) - DT \ \ \text{(passive)},
$$

$$
SE = SL - 2\,TL + TS - (NL - DI) - DT \ \ \text{(active, monostatic)},
$$

or reverberation-limited with $RL$ in place of $NL - DI$.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/sonar_equation_dark.svg">
  <img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/sonar_equation.svg" alt="The passive sonar equation: signal excess falling with transmission loss, crossing zero (the detection limit) at the figure of merit" width="82%">
</picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import underwater

tl = np.linspace(40.0, 120.0, 400)
se = underwater.passive_sonar_equation(source_level=140.0, transmission_loss=tl,
                                       noise_level=60.0, directivity_index=15.0,
                                       detection_threshold=8.0)
print(f"figure of merit = {se.figure_of_merit:.1f} dB")  # figure of merit = 87.0 dB
se.plot()   # signal excess vs transmission loss, zero crossing at the FOM
plt.show()
```

</details>

```python
import numpy as np
from phonometry import underwater

tl = np.linspace(40.0, 120.0, 400)
se = underwater.passive_sonar_equation(source_level=140.0, transmission_loss=tl,
                               noise_level=60.0, directivity_index=15.0,
                               detection_threshold=8.0)
print(se.figure_of_merit)   # max allowable one-way TL at SE = 0
se.plot()   # signal excess vs transmission loss (needs matplotlib)

# Active, monostatic (two-way loss, target strength):
underwater.active_sonar_equation(220.0, 70.0, target_strength=15.0, noise_level=60.0,
                         directivity_index=20.0, detection_threshold=10.0)
```

Sonar, propagation and ambient levels are in dB re a plane wave of 1 µPa rms
(spectrum levels are referred to a 1 Hz band). Source levels (§6) instead use
the source convention, dB re 1 µPa²/Hz **at 1 m**.

## 4. Seabed reflection loss

A plane wave striking the seabed reflects with the fluid–fluid **Rayleigh
reflection coefficient** (Medwin & Clay). For a faster bottom ($c_2 > c_1$) there
is a **critical grazing angle** $\varphi_c = \arccos(c_1/c_2)$, below which
the wave is totally reflected ($|R| = 1$, zero loss). The bottom loss is
$BL = -20 \lg |R|$.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/seabed_reflection_dark.svg">
  <img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/seabed_reflection.svg" alt="Bottom reflection loss versus grazing angle for a fast sandy seabed: zero loss below the critical grazing angle (total reflection) rising sharply above it" width="82%">
</picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import underwater

# Rayleigh fluid-fluid reflection: water over a fast sandy bottom.
phi = np.linspace(0.0, 90.0, 361)
bl = underwater.bottom_reflection_loss(phi, rho1=1000.0, c1=1500.0,
                                       rho2=1900.0, c2=1650.0)
print(f"critical angle = {bl.critical_angle:.1f} deg")   # critical angle = 24.6 deg
bl.plot()   # bottom loss vs grazing angle
plt.show()
```

</details>

```python
import numpy as np
from phonometry import underwater

phi = np.linspace(0.0, 90.0, 361)   # grazing angle from the interface, degrees
bl = underwater.bottom_reflection_loss(phi, rho1=1000.0, c1=1500.0,  # water
                               rho2=1900.0, c2=1650.0)         # sand
print(bl.critical_angle)            # 24.6° for this sand/water pair
bl.plot()                           # bottom loss vs grazing angle (needs matplotlib)
```

`bottom_reflection_loss` returns a `BottomLossResult` with `reflection_loss`,
the complex `reflection_coefficient` and the `critical_angle` (`None` for a
slower bottom). `reflection_coefficient(…)` and `critical_angle(c1, c2)` are
also exposed directly. The model is lossless (real densities and sound speeds);
sediment attenuation is out of scope.

The companion `seabed_reflection` bundles the complex `reflection_coefficient`,
its `magnitude` $|R|$, the `bottom_loss` (dB) and the interface parameters into
a `SeabedReflection` whose `.plot()` draws the reflection-coefficient magnitude
$|R|$ directly (unity below the critical angle, dropping to the normal-incidence
value at $90°$).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/seabed_reflection_coefficient_dark.svg">
  <img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/seabed_reflection_coefficient.svg" alt="Seabed reflection-coefficient magnitude versus grazing angle for a fast sandy seabed: total reflection (|R| = 1) below the critical grazing angle, falling to the normal-incidence value above it" width="82%">
</picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import underwater

# Rayleigh reflection-coefficient magnitude: water over a fast sandy bottom.
phi = np.linspace(0.0, 90.0, 361)
sr = underwater.seabed_reflection(phi, rho1=1000.0, c1=1500.0,
                                  rho2=1900.0, c2=1650.0)
print(f"|R| at normal incidence = {sr.magnitude[-1]:.3f}")  # |R| at normal incidence = 0.353
sr.plot()   # reflection-coefficient magnitude vs grazing angle
plt.show()
```

</details>

```python
import numpy as np
from phonometry import underwater

phi = np.linspace(0.0, 90.0, 361)   # grazing angle from the interface, degrees
sr = underwater.seabed_reflection(phi, rho1=1000.0, c1=1500.0,  # water
                          rho2=1900.0, c2=1650.0)         # sand
print(sr.magnitude[-1])             # 0.353 = |R| at normal incidence
sr.plot()                           # |R| vs grazing angle (needs matplotlib)
```

## 5. Ocean ambient noise

The ambient-noise spectrum level (dB re 1 µPa²/Hz) is the energy sum of the two
physically grounded Wenz components: **wind / sea-surface** noise via the "rule
of fives" ($51.02 - (5/3)\,10\,(\lg f_{\text{kHz}} - \lg(U/5))$: the historical "25 dB (5 × 5)" anchor at 1 kHz for 5 knots is re 20 µPa, i.e. $25 + 20 \lg 20 \approx 51.02$ dB re 1 µPa) and
**Mellen thermal** noise ($4 \pi k T \rho f^2 / c$, dominant above ~50 kHz). A
**shipping** spectrum may be supplied by the caller, for example one predicted
by the traffic model in §6.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/ocean_ambient_noise_dark.svg">
  <img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/ocean_ambient_noise.svg" alt="Wenz ambient-noise spectrum levels for two wind speeds: wind-dominated at mid frequencies falling at 5 dB per octave and thermal noise rising above about 50 kHz" width="82%">
</picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import underwater

# Wenz ambient noise (wind rule of fives + Mellen thermal) at two wind speeds.
freqs = np.logspace(2, 5.5, 300)
fig, ax = plt.subplots()
for u in (5.0, 20.0):
    noise = underwater.ocean_ambient_noise(freqs, wind_speed_knots=u)
    ax.semilogx(noise.frequency, noise.spectrum_level, label=f"Total ({u:.0f} kn)")
ax.semilogx(freqs, underwater.thermal_noise_spectrum(freqs), ":", label="Thermal")
ax.set(xlabel="Frequency [Hz]", ylabel="Spectrum level [dB re 1 µPa²/Hz]")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
plt.show()
```

</details>

```python
import numpy as np
from phonometry import underwater

freqs = np.logspace(2, 5.5, 300)
noise = underwater.ocean_ambient_noise(freqs, wind_speed_knots=15.0)
noise.plot()   # composite spectrum with wind/thermal components (needs matplotlib)

# Individual components are available directly:
underwater.wind_noise_spectrum(1000.0, 5.0)      # 51.02 dB (rule-of-fives anchor re 1 µPa)
underwater.thermal_noise_spectrum(5e4)           # molecular thermal-noise limit
```

The wind component is strictly valid over roughly 500 Hz–5 kHz; the wide example
range keeps it plotted mainly to show where the thermal component takes over
above ~50 kHz (the wind curve beyond ~5 kHz is an extrapolation). The
low-frequency turbulence band and a built-in distant-shipping model are out of
scope (Wenz notes these bands are strongly variable); a shipping spectrum is
supplied through the `shipping` argument.

## 6. Ship-traffic source level

When no measured spectrum is available, a ship's underwater source level can be
**estimated** from its class, speed and length. Three semi-empirical models are
available: **JOMOPANS-ECHO** (MacGillivray & de Jong 2021, per vessel class,
validated against 1862 measurements, the default), **RANDI 3.1** and
**Wales & Heitmeyer** (2002). All return the source spectral-density level and
the decidecade-band source level.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/ship_traffic_noise_dark.svg">
  <img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/ship_traffic_noise.svg" alt="JOMOPANS-ECHO predicted source-level spectra for a container ship, a cruise ship and a tug: cargo vessels show a low-frequency hump below 100 Hz" width="82%">
</picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
from phonometry import underwater

# JOMOPANS-ECHO source spectra for three vessel classes (speed, length).
fig, ax = plt.subplots()
for vessel_class, speed, length in (("containership", 18.0, 300.0),
                                    ("cruise", 17.1, 250.0),
                                    ("tug", 3.7, 30.0)):
    s = underwater.ship_source_spectrum(speed, length, vessel_class=vessel_class)
    ax.semilogx(s.frequency, s.source_psd,
                label=f"{vessel_class} ({speed:.0f} kn, {length:.0f} m)")
ax.set(xlabel="Frequency [Hz]",
       ylabel="Source spectral density [dB re 1 µPa²/Hz at 1 m]")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
plt.show()
```

</details>

```python
from phonometry import underwater

# A container ship at 18 knots, 300 m long (JOMOPANS-ECHO default):
ship = underwater.ship_source_spectrum(18.0, 300.0, vessel_class="containership")
ship.plot()                     # source spectral density vs frequency

print(underwater.VESSEL_CLASSES)        # the 13 JOMOPANS-ECHO vessel classes

# Feed the predicted spectrum into the ambient noise as the shipping term:
noise = underwater.ocean_ambient_noise(ship.frequency, wind_speed_knots=10.0,
                               shipping=ship.source_psd)
```

`ship_source_spectrum` returns a `ShipTrafficSpectrum` with `source_psd`
(dB re 1 µPa²/Hz m) and `band_level` (dB re 1 µPa m). Cargo vessels (container
ships, bulkers, vehicle carriers, tankers) carry an extra low-frequency hump
below 100 Hz. The implementation is validated to the authors' own reference
calculator (File S1) to better than 0.01 dB.

## 7. Numerical solvers (normal modes, rays, parabolic equation)

For range-independent (horizontally stratified) environments the field can be
computed numerically. Three solvers are provided (Jensen et al.,
*Computational Ocean Acoustics*):

- **`normal_modes`** solves the depth-separated Sturm-Liouville eigenvalue
  problem by finite differences and sums the propagating modes into the
  transmission loss. Validated against the ideal (pressure-release) waveguide's
  exact modes.
- **`ray_trace`** integrates the ray-trajectory equations (Runge-Kutta,
  vectorised over all rays at once) through a sound-speed profile, reflecting at
  the surface and bottom. Validated against the circular-arc paths of a linear
  gradient.
- **`parabolic_equation`** marches the standard (Tappert) PE with the split-step
  Fourier algorithm. Validated against free-field spherical spreading; it agrees
  with the normal-mode transmission loss in trend.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/numerical_propagation_dark.webp">
  <img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/numerical_propagation.webp" alt="Three numerical solvers: a Munk sound-speed profile, ray paths forming convergence zones, and transmission loss versus range from the normal-mode and parabolic-equation solvers agreeing in trend" width="100%">
</picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import underwater

# A Munk deep-water sound-speed profile.
z = np.linspace(0.0, 5000.0, 60)
eta = 2.0 * (z - 1300.0) / 1300.0
c = 1500.0 * (1.0 + 0.00737 * (eta - 1.0 + np.exp(-eta)))

# Split-step Fourier PE at 50 Hz; a coarse grid keeps the run fast.
field = underwater.parabolic_equation(50.0, z, c, source_depth=1000.0,
                                      max_range=50_000.0, range_step=50.0,
                                      n_depth_points=512)
field.plot()   # TL(z, r) field showing the convergence zones
plt.show()
```

</details>

```python
import numpy as np
from phonometry import underwater

# A Munk deep-water profile.
z = np.linspace(0.0, 5000.0, 60)
eta = 2.0 * (z - 1300.0) / 1300.0
c = 1500.0 * (1.0 + 0.00737 * (eta - 1.0 + np.exp(-eta)))

rays = underwater.ray_trace(z, c, source_depth=1000.0,
                    launch_angles_deg=np.linspace(-12.0, 12.0, 21), max_range=100e3)
rays.plot()   # ray paths / convergence zones (needs matplotlib)

# Shallow isovelocity waveguide: modes and PE.
modes = underwater.normal_modes(50.0, [0.0, 200.0], [1500.0, 1500.0],
                        source_depth=50.0, receiver_depth=100.0)
print(modes.wavenumbers.size, "propagating modes")
field = underwater.parabolic_equation(50.0, [0.0, 200.0], [1500.0, 1500.0],
                              source_depth=50.0, max_range=20e3)
field.plot()  # TL field over range x depth (needs matplotlib)
```

`normal_modes` returns a `NormalModeResult` (`wavenumbers`, `mode_functions`,
`transmission_loss`); `ray_trace` a `RayTraceResult` (`ranges`, `depths` per
ray); `parabolic_equation` a `ParabolicEquationResult` (the `transmission_loss`
field). All assume a range-independent water column with a pressure-release
surface.

## 8. Choosing a model

Every function above answers the same question, "how much level survives the
path", at a different price in physics. Terminology throughout follows
ISO 18405:2017 (propagation loss, source level, levels re 1 µPa).

**Sound speed.** The three equations agree to within about 1 m/s inside their
common domain, so the choice is about *validity range*, not accuracy. The
default **UNESCO / Chen-Millero** form (as recast by Wong & Zhu 1995) covers
0–40 °C, 0–40 ppt and 0–1000 bar, the widest envelope, and is the
international standard. **Del Grosso** (1974) is restricted to 0–30 °C and
30–40 ppt but is preferred by some authors for deep-ocean work inside that
domain (much of the SOFAR-channel literature uses it). **Mackenzie** (1981)
trades pressure for depth directly (2–30 °C, 25–40 ppt, 0–8000 m), which makes
it the convenient choice when you have an echo-sounder depth rather than a CTD
pressure; the other two convert depth to pressure through Leroy & Parthiot
(1998) internally.

**Absorption.** **Francois–Garrison** (1982) is the reference and the default:
it carries the boric-acid, magnesium-sulfate and pure-water relaxations with
their full temperature, salinity, depth and pH-implicit dependences, and is
trusted from about 100 Hz to 1 MHz. **Ainslie–McColm** (1998) is a deliberate
simplification of the same physics that stays within about 10 % of it across
that range; use it when a legible formula matters more than the last percent.
**Thorp** (1967) depends on frequency only (it bakes in 4 °C water near
1000 m) and predates both; keep it for quick low-frequency estimates below a
few tens of kHz and for comparison with older literature that used it.

**Spreading law.** Spherical spreading (`20 lg R`) describes a wavefront that
expands freely in three dimensions, before any boundary confines it;
cylindrical spreading (`10 lg R`) describes energy trapped between the surface
and the bottom (or in the SOFAR channel) that can only expand in range. The
`"practical"` law splices the two at a transition range `R0`, which is
physically of the order of the water (or channel) depth: spherical while the
wavefront has not yet filled the duct, cylindrical once it has. In the 10 kHz
example of section 1 the choice is not cosmetic: against the same figure of
merit of 87 dB, spherical-only spreading predicts detection out to about
8.7 km while the practical law with `R0 = 1000 m` stretches it to about
15.8 km. When the spreading law is the biggest uncertainty in the budget, that
is the cue to stop using a closed form and compute the field.

**Closed form or solver.** The closed-form transmission loss knows nothing of
the sound-speed profile, the seabed or the surface; it is honest for short,
direct, boundary-free paths and for first-cut sonar budgets. When refraction
and boundaries decide the answer, pick the solver by frequency and geometry
(Jensen et al. 2011, Ch. 1):

| Solver | Natural regime | What it buys you |
|---|---|---|
| `ray_trace` | High frequency (water depth ≫ λ), deep water | Eigenray geometry, travel times, convergence zones; cost independent of frequency |
| `normal_modes` | Low frequency, shallow water, range-independent | Exact modal sum with few propagating modes (`m < kD/π`); the reference solution for its regime |
| `parabolic_equation` | Low frequency, long one-way paths | Full-field TL(z, r) with refraction; marches arbitrary profiles in range |

The boundaries blur in practice: rays remain usable at surprisingly low
frequencies for travel-time work, and the PE remains the workhorse well above
its formal small-angle regime. When two of the three agree on a case, as the
modes and the PE do in the section 7 figure, that agreement is the practical
convergence test.

**A worked sonar budget.** Chain the pieces end to end: a 140 dB re
1 µPa²/Hz source at 10 kHz, a 60 dB ambient spectrum level, a 15 dB array
gain and an 8 dB detection threshold give the figure of merit
`FOM = 140 - (60 - 15) - 8 = 87 dB` computed by `passive_sonar_equation` in
section 3. The transmission-loss curve of section 1 (10 °C, 35 ppt, 100 m,
`alpha = 0.95 dB/km`) crosses 87 dB at about 15.8 km with the practical law:
that crossing *is* the predicted detection range, and every term of the budget
moves it. Trim the directivity index to 7.5 dB and the figure of merit falls
to 79.5 dB, so the range drops to wherever the TL curve crosses that value;
double the frequency to 20 kHz and `alpha` more than triples to 3.3 dB/km,
pulling the crossing sharply inward. This coupling between the absorption
model, the spreading law and the sonar equation is why the three live in one
module.

## References

- Francois, R. E., & Garrison, G. R. (1982). Sound absorption based on ocean
  measurements: Part I: Pure water and magnesium sulfate contributions.
  *The Journal of the Acoustical Society of America*, 72(3), 896-907.
  [doi:10.1121/1.388170](https://doi.org/10.1121/1.388170).
  The pure-water and magnesium-sulfate halves of the default absorption
  model of section 1.
- Francois, R. E., & Garrison, G. R. (1982). Sound absorption based on ocean
  measurements. Part II: Boric acid contribution and equation for total
  absorption. *The Journal of the Acoustical Society of America*, 72(6),
  1879-1890.
  [doi:10.1121/1.388673](https://doi.org/10.1121/1.388673).
  The boric-acid term and the complete Francois-Garrison total-absorption
  equation, the implemented default.
- Ainslie, M. A., & McColm, J. G. (1998). A simplified formula for viscous and
  chemical absorption in sea water. *The Journal of the Acoustical Society of
  America*, 103(3), 1671-1672.
  [doi:10.1121/1.421258](https://doi.org/10.1121/1.421258).
  The legible simplified absorption model (`"ainslie-mccolm"`).
- Thorp, W. H. (1967). Analytic description of the low-frequency attenuation
  coefficient. *The Journal of the Acoustical Society of America*, 42(1), 270.
  [doi:10.1121/1.1910566](https://doi.org/10.1121/1.1910566).
  The frequency-only low-frequency absorption formula (`"thorp"`).
- Chen, C.-T., & Millero, F. J. (1977). Speed of sound in seawater at high
  pressures. *The Journal of the Acoustical Society of America*, 62(5),
  1129-1135.
  [doi:10.1121/1.381646](https://doi.org/10.1121/1.381646).
  The UNESCO international-standard sound-speed equation of section 2.
- Wong, G. S. K., & Zhu, S. (1995). Speed of sound in seawater as a function
  of salinity, temperature, and pressure. *The Journal of the Acoustical
  Society of America*, 97(3), 1732-1736.
  [doi:10.1121/1.413048](https://doi.org/10.1121/1.413048).
  The ITS-90 recast of the UNESCO coefficients, the implemented form.
- Del Grosso, V. A. (1974). New equation for the speed of sound in natural
  waters (with comparisons to other equations). *The Journal of the
  Acoustical Society of America*, 56(4), 1084-1091.
  [doi:10.1121/1.1903388](https://doi.org/10.1121/1.1903388).
  The alternative pressure-based sound-speed equation (`"del-grosso"`).
- Mackenzie, K. V. (1981). Nine-term equation for sound speed in the oceans.
  *The Journal of the Acoustical Society of America*, 70(3), 807-812.
  [doi:10.1121/1.386920](https://doi.org/10.1121/1.386920).
  The depth-based nine-term equation and its 1550.744 m/s check value.
- Leroy, C. C., & Parthiot, F. (1998). Depth-pressure relationships in the
  oceans and seas. *The Journal of the Acoustical Society of America*, 103(3),
  1346-1352.
  [doi:10.1121/1.421275](https://doi.org/10.1121/1.421275).
  The depth-to-pressure conversion feeding the UNESCO and Del Grosso
  equations.
- Urick, R. J. (1983). *Principles of underwater sound* (3rd ed.).
  McGraw-Hill; reprinted 1996 by Peninsula Publishing.
  ISBN 978-0-932146-62-5.
  [Open Library record](https://openlibrary.org/books/OL9317725M).
  The sonar-equation framework (signal excess, figure of merit) of section 3.
- Medwin, H., & Clay, C. S. (1998). *Fundamentals of acoustical oceanography*.
  Academic Press. ISBN 978-0-12-487570-8.
  [Publisher page](https://shop.elsevier.com/books/fundamentals-of-acoustical-oceanography/medwin/978-0-12-487570-8).
  The fluid-fluid Rayleigh reflection coefficient and critical grazing angle
  of section 4.
- Wenz, G. M. (1962). Acoustic ambient noise in the ocean: Spectra and
  sources. *The Journal of the Acoustical Society of America*, 34(12),
  1936-1956.
  [doi:10.1121/1.1909155](https://doi.org/10.1121/1.1909155).
  The ambient-noise survey behind the wind and thermal components of
  section 5.
- Carey, W. M., & Evans, R. B. (2011). *Ocean ambient noise: Measurement and
  theory*. Springer.
  [doi:10.1007/978-1-4419-7832-5](https://doi.org/10.1007/978-1-4419-7832-5).
  The wind "rule of fives" anchor and the Mellen thermal-noise derivation
  used by section 5.
- MacGillivray, A., & de Jong, C. (2021). A reference spectrum model for
  estimating source levels of marine shipping based on automated
  identification system data. *Journal of Marine Science and Engineering*,
  9(4), 369.
  [doi:10.3390/jmse9040369](https://doi.org/10.3390/jmse9040369).
  The JOMOPANS-ECHO ship source-level model of section 6 (open access); its
  File S1 calculator is the validation oracle.
- Wales, S. C., & Heitmeyer, R. M. (2002). An ensemble source spectra model
  for merchant ship-radiated noise. *The Journal of the Acoustical Society of
  America*, 111(3), 1211-1231.
  [doi:10.1121/1.1427355](https://doi.org/10.1121/1.1427355).
  The ensemble merchant-ship spectrum model of section 6.
- Jensen, F. B., Kuperman, W. A., Porter, M. B., & Schmidt, H. (2011).
  *Computational ocean acoustics* (2nd ed.). Springer.
  [doi:10.1007/978-1-4419-8678-8](https://doi.org/10.1007/978-1-4419-8678-8).
  The normal-mode (Ch. 5), ray-tracing (Ch. 3) and split-step Fourier
  parabolic-equation (Ch. 6) solvers of section 7, and the model-selection
  guidance of Ch. 1 used by section 8.
- ISO 18405:2017. *Underwater acoustics — Terminology*.
  [ISO page](https://www.iso.org/standard/62406.html).
  The standardized definitions (propagation loss, source level, sound
  pressure level re 1 µPa) behind the quantities of this page.

## Standards & sources

- Speed of sound: UNESCO / Chen & Millero (1977, Wong & Zhu 1995 ITS-90),
  Del Grosso (1974), Mackenzie (1981), Leroy & Parthiot (1998).
- Absorption: Francois & Garrison (1982), Ainslie & McColm (1998), Thorp (1967).
- Sonar equation: Urick, *Principles of Underwater Sound*.
- Seabed reflection: Medwin & Clay, *Fundamentals of Acoustical Oceanography*
  (Rayleigh fluid–fluid reflection coefficient).
- Ambient noise: Wenz (1962); Carey & Evans, *Ocean Ambient Noise* (2011), for the
  wind "rule of fives" and the Mellen thermal-noise derivation.
- Ship-traffic source level: MacGillivray & de Jong (2021),
  *J. Mar. Sci. Eng.* 9(4) 369 (CC-BY, JOMOPANS-ECHO), which also reproduces
  RANDI 3.1 and Wales & Heitmeyer (2002).
- Numerical solvers: Jensen, Kuperman, Porter & Schmidt,
  *Computational Ocean Acoustics* (2nd ed., Springer 2011), for normal modes
  (Ch. 5), ray tracing (Ch. 3) and the split-step Fourier parabolic equation
  (Ch. 6).
