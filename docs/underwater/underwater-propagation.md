← [Documentation index](../README.md)

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

Geometrical spreading is $20 \log_{10} R$ (spherical), $10 \log_{10} R$ (cylindrical) or
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

### 1.1 Weston's shallow-water regimes

Fixing the spreading law by hand works offshore, but in shallow water the law
*changes with range* as the sea floor takes over. Weston's energy-flux theory,
as set out in Ainslie §9.1.1.2, derives the four successive regimes and the
ranges at which they hand over, from the seabed reflectivity alone:

| Regime | Propagation factor $F$ | Loss law | Ends at |
|---|---|---|---|
| Spherical | $1/r^2$ | $20 \log_{10} r$ | $r = H/(2\psi_c)$ |
| Cylindrical | $2\psi_c/(rH)$ | $10 \log_{10} r$ | $r_{CS} = \pi H/(4\eta\psi_c^2)$ |
| Mode stripping | $\sqrt{\pi/(\eta H)}\, r^{-3/2}$ | $15 \log_{10} r$ | $r_{MS} = k^2 H_e^3/(9\pi\eta)$ |
| Single mode | Eq. (9.54), exponential | steeper than $15 \log_{10} r$ | — |

$\psi_c = \arccos(c_w/c_{sed})$ is the critical grazing angle, $\eta$ the
**reflection loss gradient** in nepers per radian ($|R(\theta)| \approx
e^{-\eta\theta}$) and $H_e$ the Weston effective depth, the level a short
distance below the true seabed at which a pressure-release boundary appears to
lie. `weston_propagation_loss` assembles the composite loss
$PL = -10 \log_{10} F$ and returns each regime's own law alongside it; the
`boundaries` field carries the three transition ranges plus the waveguide
cut-off frequency and the number of cut-on modes.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/weston_regimes_dark.svg">
  <img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/weston_regimes.svg" alt="Weston's four shallow-water propagation regimes at 250 Hz over medium sand in 50 m of water: the composite propagation loss follows spherical spreading to about 43 m, cylindrical spreading to 412 m, the 15 lg r mode-stripping law to about 20 km and then the exponential single-mode decay, with each individual law drawn for comparison" width="82%">
</picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import underwater

# 50 m of water over medium sand (Ainslie Table 9.1) at 250 Hz.
ranges = np.logspace(1.0, 5.3, 500)
res = underwater.weston_propagation_loss(ranges, 250.0, 50.0, seabed="sand",
                                         source_depth=10.0, receiver_depth=25.0)
b = res.boundaries
print(f"psi_c = {np.degrees(b.critical_angle):.1f} deg, eta = {b.reflection_loss_gradient:.2f} Np/rad")
print(f"boundaries: {b.spherical_to_cylindrical:.0f} / "
      f"{b.cylindrical_to_mode_stripping:.0f} / {b.mode_stripping_to_single_mode:.0f} m")
res.plot()   # composite loss with each regime law and the boundaries
plt.show()
```

</details>

```python
import numpy as np
from phonometry import underwater

b = underwater.weston_regime_boundaries(250.0, 50.0, seabed="sand")
print(b.cylindrical_to_mode_stripping)   # r_CS, in metres
print(b.cutoff_frequency, b.mode_count)  # ducted propagation needs f > f_c

# The seabed properties themselves (Ainslie Table 9.1) are addressable:
print(underwater.WESTON_SEABEDS["sand"].density_ratio)      # 2.1
print(underwater.reflection_loss_gradient("sand"))          # 0.278 Np/rad
print(underwater.reflection_loss_gradient("mud", frequency_hz=250.0))
```

Because the flux result is *incoherent* it describes the range-averaged field,
which is exactly what makes it a reference for the numerical solvers of
[Underwater propagation solvers](underwater-solvers.md). Set the critical angle
to 90° and the loss gradient to zero and the cylindrical branch reduces to
$F = \pi/(rH)$, the exact many-mode limit of an ideal waveguide; the
depth- and range-averaged normal-mode loss lands on it to within a decibel.

```python
import numpy as np
from phonometry import underwater

ranges = np.linspace(20_000.0, 30_000.0, 2001)
flux = underwater.weston_propagation_loss(ranges, 100.0, 100.0, critical_angle=90.0,
                                          reflection_loss_gradient_value=0.0)
modes = underwater.normal_modes(100.0, [0.0, 100.0], [1500.0, 1500.0],
                                source_depth=41.0, receiver_depth=57.0, ranges_m=ranges)
mean = lambda tl: -10.0 * np.log10(np.mean(10.0 ** (-tl / 10.0)))
print(mean(flux.propagation_loss), mean(modes.transmission_loss))
```

> Ainslie's printed Equation (9.57) for $r_{MS}$ reads $k^2H_e^3/(9\eta)$. Its
> own derivation rule, equating Equations (9.47) and (9.56) exactly as printed,
> gives $k^2H_e^2H/(9\pi\eta)$, smaller by $\pi H_e/H$. The library implements
> the derivation-consistent form; see
> [Errata found in published sources](../ERRATA.md).

## 2. Speed of sound in sea water

`sea_water_sound_speed(T, S, depth, model=…)` evaluates the sound speed with the
**UNESCO / Chen–Millero** equation (default, the international standard, in the
Wong & Zhu 1995 ITS-90 form), **Del Grosso** (1974), **Mackenzie** (1981) or
**Medwin** (1975). The UNESCO and Del Grosso equations use pressure, so depth is
first converted with the Leroy & Parthiot (1998) formula (`depth_to_pressure`).
The four agree to within ~2.5 m/s in their common domain; Mackenzie's canonical
check value is 1550.744 m/s at 25 °C, 35 ppt, 1000 m.

Medwin's six-term form is the simplest of the family and the one worth
remembering, because its partial derivatives are the classic rules of thumb:
$\partial c/\partial T \approx 4.6 - 0.110\,T$ m/s per °C (3.5 m/s per °C at
10 °C) and $\partial c/\partial z \approx 0.016$ m/s per metre. It is a
deliberately coarse fit and should not be used where the UNESCO standard
applies.

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

At real ocean scale the channel axis sits near 1200 m, and the trapped
arrivals are rays cycling about the sound-speed minimum over tens of
kilometres.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_sofar_channel_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_sofar_channel.svg" alt="The SOFAR channel: a North Atlantic sound-speed profile with 1524 m/s at the surface, a minimum near 1492 m/s at the 1200 m channel axis and 1527 m/s at the 4800 m bottom, beside ray paths from a source on the axis that oscillate about the sound-speed minimum and stay trapped without touching the surface or the bottom" width="92%"></picture>

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

### 3.1 Detection range

Since the figure of merit *is* the maximum allowable transmission loss,
inverting a loss law at $TL = FOM$ gives the **detection range**, the range at
which the detection probability is 50 %. `detection_range` inverts the
closed-form loss of §1, which grows monotonically with range and therefore has
a single crossing; `detection_range_from_curve` reads the crossing off any
computed curve, including the oscillating loss of a real waveguide where there
may be several.

```python
from phonometry import underwater

# Ainslie's active CW example: FOM = 82.7 dB re m2 at 50 kHz -> r50 ~ 1.3 km.
res = underwater.detection_range(82.7, 50e3)
print(res.detection_range)          # metres
res.plot()                          # TL vs FOM with the crossing marked

# Off a numerical prediction instead:
import numpy as np
modes = underwater.normal_modes(200.0, [0.0, 100.0], [1500.0, 1500.0],
                                source_depth=40.0, receiver_depth=60.0,
                                ranges_m=np.linspace(100.0, 20_000.0, 800))
print(underwater.detection_range_from_curve(60.0, modes.ranges, modes.transmission_loss))
print(underwater.detection_range_from_curve(60.0, modes.ranges, modes.transmission_loss,
                                            crossing="last"))
```

Sonar, propagation and ambient levels are in dB re a plane wave of 1 µPa rms
(spectrum levels are referred to a 1 Hz band). Source levels (§6) instead use
the source convention, dB re 1 µPa²/Hz **at 1 m**.

## 4. Seabed reflection loss

A plane wave striking the seabed reflects with the fluid–fluid **Rayleigh
reflection coefficient** (Medwin & Clay). For a faster bottom ($c_2 > c_1$) there
is a **critical grazing angle** $\varphi_c = \arccos(c_1/c_2)$, below which
the wave is totally reflected ($|R| = 1$, zero loss). The bottom loss is
$BL = -20 \log_{10} |R|$.

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
of fives" ($51.02 - (5/3)\,10\,(\log_{10} f_{\text{kHz}} - \log_{10}(U/5))$: the historical "25 dB (5 × 5)" anchor at 1 kHz for 5 knots is re 20 µPa, i.e. $25 + 20 \log_{10} 20 \approx 51.02$ dB re 1 µPa) and
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

Every closed form above stops being enough when refraction and boundaries
decide the answer: a sound-speed minimum that traps energy, surface and
bottom reflections in shallow water, or a detection range that swings with
the choice of spreading law. At that point the field has to be computed,
and the normal-mode, ray-tracing and parabolic-equation solvers of the
module, together with the guidance for choosing between them and these
closed forms, have their own guide:
[Underwater propagation solvers](underwater-solvers.md).

## See also

- [Underwater propagation solvers](underwater-solvers.md):
  the normal-mode, ray-tracing and parabolic-equation solvers for the cases
  where these closed forms are not enough, and the model-selection guidance.
- API reference: [`underwater.propagation.closed_form`](https://jmrplens.github.io/phonometry/reference/api/underwater/closed-form/) and [`underwater.propagation.sound_speed`](https://jmrplens.github.io/phonometry/reference/api/underwater/sound-speed/).

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
- Ainslie, M. A. (2010). *Principles of Sonar Performance Modelling*.
  Springer/Praxis.
  [doi:10.1007/978-3-540-87662-5](https://doi.org/10.1007/978-3-540-87662-5).
  The Weston shallow-water regimes of section 1.1 (§9.1.1.2, Equations 9.42 to
  9.61 and Table 9.1), the Medwin sound-speed formula of section 2
  (Equation 1.2) and the seven numeric sonar worked examples of chapters 3 and
  11 that pin section 3.
- ISO 18405:2017. *Underwater acoustics — Terminology*.
  [ISO page](https://www.iso.org/standard/62406.html).
  The standardized definitions (propagation loss, source level, sound
  pressure level re 1 µPa) behind the quantities of this page.

## Standards & sources

- Speed of sound: UNESCO / Chen & Millero (1977, Wong & Zhu 1995 ITS-90),
  Del Grosso (1974), Mackenzie (1981), Medwin (1975), Leroy & Parthiot (1998).
- Absorption: Francois & Garrison (1982), Ainslie & McColm (1998), Thorp (1967).
- Shallow-water propagation regimes and the detection-range inversion:
  Ainslie, *Principles of Sonar Performance Modelling* (Springer 2010),
  §9.1.1.2 and the worked examples of chapters 3 and 11.
- Sonar equation: Urick, *Principles of Underwater Sound*.
- Seabed reflection: Medwin & Clay, *Fundamentals of Acoustical Oceanography*
  (Rayleigh fluid–fluid reflection coefficient).
- Ambient noise: Wenz (1962); Carey & Evans, *Ocean Ambient Noise* (2011), for the
  wind "rule of fives" and the Mellen thermal-noise derivation.
- Ship-traffic source level: MacGillivray & de Jong (2021),
  *J. Mar. Sci. Eng.* 9(4) 369 (CC-BY, JOMOPANS-ECHO), which also reproduces
  RANDI 3.1 and Wales & Heitmeyer (2002).
- Numerical solvers (normal modes, ray tracing, parabolic equation):
  covered in [Underwater propagation solvers](underwater-solvers.md).
