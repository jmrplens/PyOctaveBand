← [Documentation index](README.md)

# Impedance Tube

The impedance tube is the materials laboratory shrunk to a bench: a rigid duct
a few centimetres across in which a loudspeaker drives a single plane wave
against a small sample, and everything the sample does to that wave is read
back from microphones in the tube wall. Because there is one wave and one
angle, the tube recovers what a reverberation room cannot: the **complex**
reflection factor and surface impedance at normal incidence, magnitude and
phase, from a specimen the size of a coffee-cup lid. Three standards share
the hardware and differ in method, and the library keeps their helpers
separate and never mixes them: the **standing-wave-ratio** method of
ISO 10534-1, the two-microphone **transfer-function** method of ISO 10534-2,
and the four-microphone **transfer-matrix** method of ASTM E2611 that adds
the transmission loss. The closing section runs the whole instrument, sample
included, inside the [FDTD solver](fdtd-simulation.md) and recovers the
analytic answer through the same reduction chains.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_impedance_tube_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_impedance_tube.svg" alt="ISO 10534-2 two-microphone impedance tube: a loudspeaker radiating a plane wave down the tube, two microphones flush in the wall at spacing s and distance x1 from the specimen face, the test specimen against a rigid backing, and the incident and reflected waves" width="92%"></picture>

## 1. The working frequency range

Everything the tube reports (ISO 10534-2, Clause 4) assumes the field inside
is a single plane wave, and the geometry sets both ends of the usable band.
Above the cut-on of the first cross-sectional mode the field stops being
planar: a circular tube of diameter $d$ needs
$f\,d < 0.58\,c_0$ (Eq. (2); $f\,d < 0.50\,c_0$ for a rectangular tube,
Eq. (3)). The microphone spacing $s$ must also stay clear of the
half-wavelength singularity of the transfer-function method,
$f\,s < 0.45\,c_0$ (Eq. (4)). At the low end the opposite problem appears: a
spacing much shorter than the wavelength leaves almost no phase difference
between the microphones to measure, so the Clause 4.2 guideline keeps the
spacing above 5 % of the wavelength:

$$
\frac{c_0}{20\,s} \;<\; f \;<\;
\min\!\left(0.58\,\frac{c_0}{d},\ 0.45\,\frac{c_0}{s}\right).
$$

No single tube covers the building-acoustics range. A 100 mm tube with a
100 mm spacing works from roughly 170 Hz to 1.5 kHz; reaching the 5 kHz bands
takes a small tube (29 mm) with a close spacing, which in turn cannot see the
low bands. Laboratories therefore pair a large and a small tube (or one tube
with two spacings) and splice the spectra; the two must agree in the overlap
bands, and a mismatch there points at the sample cut or mounting, not at the
physics.

```python
from phonometry import materials

# A 100 mm tube with 100 mm spacing, and a 29 mm tube with 20 mm spacing.
f_l, f_u = materials.plane_wave_frequency_range(0.100, 343.2, diameter=0.100)
print(round(f_l, 1), round(f_u, 1))     # 171.6 1544.4
f_l, f_u = materials.plane_wave_frequency_range(0.020, 343.2, diameter=0.029)
print(round(f_l, 1), round(f_u, 1))     # 858.0 6864.0
```

Those limits are easier to see on the hardware itself.
`plot_impedance_tube_geometry` draws a tube to scale with its plane-wave
band worked out, and a measured `two_microphone_impedance` result that was
given its geometry (`spacing`, `x1`, `diameter`) redraws its own setup with
`result.plot_geometry()`.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/impedance_tube_geometry_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/impedance_tube_geometry.svg" alt="To-scale side view of a 100 mm ISO 10534-2 impedance tube: the loudspeaker at the left end, microphones 1 and 2 flush in the wall at s = 50 mm spacing with microphone 1 at x1 = 150 mm from the specimen face, the specimen against the rigid backing at the right, the circular cross-section beside the tube and the plane-wave working range of 343 to 1991 Hz that this geometry sets" width="92%"></picture>

*Everything the working-range inequalities talk about, in one to-scale side
view: the 100 mm bore sets the 1991 Hz top end, the 50 mm spacing sets the
343 Hz bottom end, and the microphones sit at $s$ = 50 mm with the farther
one $x_1$ = 150 mm from the specimen face.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
from phonometry import materials

materials.plot_impedance_tube_geometry(spacing=0.05, x1=0.15, diameter=0.10)
plt.show()

# A measured result that retains its geometry draws its own tube:
#   res = materials.two_microphone_impedance(
#       h12, frequency=f, spacing=0.05, x1=0.15, diameter=0.10, ...)
#   res.plot_geometry()
```

</details>

The same helper covers square and rectangular tubes: pass `shape="square"` or
`shape="rectangular"` and the Eq. (3) factor replaces the circular one, with
$d$ the maximum side length. The four-microphone branch keeps its own limits
in `plane_wave_frequency_range_astm`: ASTM E2611 retains the unrounded
circular constant ($f\,d < 0.586\,c$, 6.2.4.1), the same rectangular
$f\,d < 0.500\,c$ (6.2.5), a slightly stricter spacing bound
$f\,s < 0.40\,c$ (6.5.4) and a laxer low end at 1 % of the wavelength
(6.2.3). Passing `diameter=` (and `shape=`) to `two_microphone_impedance`,
`wave_decomposition` or the `transfer_matrix_*` solvers turns the matching
check into an advisory warning, and for the Annex A attenuation estimate of a
rectangular tube `hydraulic_diameter(width, height)` supplies the $4A/P$
diameter that `tube_attenuation_constant` expects.

The whole tube, sample included, can also be simulated: section 5 runs this
exact measurement virtually inside the FDTD solver and recovers the analytic
absorption of the modelled sample through the same reduction chain.

## 2. Standing-wave-ratio method (ISO 10534-1)

A probe traverses the standing wave and reads the level difference
$\Delta L = L_\text{max} - L_\text{min}$ between a pressure maximum and the
adjacent minimum. The standing-wave ratio, reflection
magnitude and absorption follow in closed form:

$$
s = 10^{\Delta L/20}, \qquad |r| = \frac{s-1}{s+1}, \qquad
\alpha = 1 - |r|^2.
$$

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/impedance_tube_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/impedance_tube.svg" alt="ISO 10534-1 standing-wave-ratio method: the absorption coefficient and the reflection factor magnitude as functions of the standing-wave level difference, showing that a 9.54 dB difference corresponds to a standing-wave ratio of 3, a reflection magnitude of 0.5 and an absorption of 0.75" width="80%"></picture>

*A small level difference means a near-perfect absorber; a level difference of
9.54 dB gives $s = 3$, $|r| = 0.5$ and $\alpha = 0.75$.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import materials

level_diff = np.linspace(0.5, 40.0, 300)    # L_max - L_min [dB]
swr = materials.standing_wave_ratio_from_level(level_diff)
fig, ax = plt.subplots()
ax.plot(level_diff, materials.standing_wave_absorption(swr),
        label="Absorption coefficient alpha")
ax.plot(level_diff, materials.standing_wave_reflection_magnitude(swr), "--",
        label="Reflection factor magnitude |r|")
ax.set_xlabel("Standing-wave level difference L_max - L_min [dB]")
ax.set_ylabel("alpha, |r|")
ax.legend()
plt.show()
```

</details>

```python
from phonometry import materials

s = float(materials.standing_wave_ratio_from_level(9.542))   # level difference [dB] -> SWR
print(round(s, 2))                                  # 3.0
print(round(float(materials.standing_wave_absorption(s)), 2)) # 0.75
```

The ratio method predates the FFT analyser and survives because it is
self-contained. The apparatus is a **probe microphone** on a graduated
carriage: a thin tube sliding along the axis, sampling the interior field
point by point while the loudspeaker holds a single pure tone. At each
frequency the operator (or the stepper motor) locates one pressure maximum
and the adjacent minimum, reads the level difference, and moves on to the
next tone. Slow, but there is nothing to calibrate against anything else: one
microphone measures both levels, so its sensitivity cancels exactly, and
there is no inter-channel phase mismatch because there is only one channel.
That is why the standing-wave method remains the arbitration method when two
transfer-function tubes disagree.

The closed forms above use only the level *difference*; the *positions* of
the minima carry the rest of the information. With $x$ measured from the
sample face towards the source, the interior field of a wave with reflection
factor $r = |r|\,e^{j\Phi}$ is

$$
p(x) = A\left(e^{jkx} + |r|\,e^{j\Phi}e^{-jkx}\right), \qquad
|p|^2 = A^2\left(1 + |r|^2 + 2|r|\cos(2kx - \Phi)\right),
$$

so pressure minima sit where the cosine is $-1$, at

$$
2k\,x_{\min,n} - \Phi = (2n - 1)\,\pi, \qquad
\Phi = \frac{4\pi\,x_{\min,1}}{\lambda} - \pi ,
$$

with $x_{\min,1}$ the minimum nearest the sample. A minimum a quarter
wavelength from the face means $\Phi = 0$ (a hard-backed sample at
resonance); a minimum right at the face means $\Phi = -\pi$ (a rigid wall).
Magnitude from the ratio, phase from the position: together they give the
same complex $r$, and hence the same normalised impedance
$Z/\rho c_0 = (1+r)/(1-r)$, that the transfer-function method of section 3
computes from $H_{12}$ in one shot.

```python
import numpy as np
from phonometry import materials

# One 500 Hz reading: 9.54 dB between max and min, first minimum 12 cm out.
f, c0 = 500.0, 343.2
swr = float(materials.standing_wave_ratio_from_level(9.542))
r_mag = float(materials.standing_wave_reflection_magnitude(swr))
phase = 4 * np.pi * 0.12 * f / c0 - np.pi         # Phi from x_min,1
refl = r_mag * np.exp(1j * phase)
print(round(r_mag, 2), round(np.degrees(phase), 1))   # 0.5 -54.1
print(np.round((1 + refl) / (1 - refl), 2))           # Z / rho c0: (1.13-1.22j)
```

Two practical cautions from the standard's own text. The minima far from the
sample are shallower than the theory above says, because the travelling waves
decay along the tube (viscous and thermal losses at the wall); ISO 10534-1
has the operator read the minimum **nearest** the sample and, for precision
work, extrapolate the minimum levels to the sample face. And the method
leans on the purity of the tone: any harmonic distortion from a hard-driven
loudspeaker puts energy at frequencies whose minima sit elsewhere, partially
filling the notch being measured, so the analyser must be narrowband around
the drive frequency rather than a broadband level meter.

## 3. Transfer-function method (ISO 10534-2)

Two fixed microphones measure the
complex transfer function $H_{12}$; from it the reflection factor at the sample
face, the absorption and the normalised surface impedance follow (Eqs. (17)–(19)):

$$
r = \frac{H_{12} - H_I}{H_R - H_{12}}\,e^{\,2 j k_0 x_1}, \qquad
\alpha = 1 - |r|^2, \qquad \frac{Z}{\rho c_0} = \frac{1+r}{1-r},
$$

with $H_I = e^{-j k_0 s}$, $H_R = e^{+j k_0 s}$, microphone spacing $s$ and $x_1$
the distance from the sample to the farther microphone.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_standing_wave_tube_dark.gif"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_standing_wave_tube.gif" alt="Animation: incident and reflected waves sum into a standing wave inside the impedance tube; a rigid termination gives deep envelope nodes, a porous sample gives shallow ones, sampled by the two wall microphones" width="640" height="360" loading="lazy"></picture>

[Watch the high-resolution video (WebM)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_standing_wave_tube.webm)

```python
import numpy as np
from phonometry import materials

f = np.array([500.0, 1000.0, 1800.0])
x1, spacing, c0 = 0.12, 0.03, 343.2
k0 = materials.tube_wavenumber(f, c0)

# A measured transfer function H12 (here synthesised from r = 0.3 - 0.4j)
target = 0.3 - 0.4j
x2 = x1 - spacing
h12 = (np.exp(1j*k0*x2) + target*np.exp(-1j*k0*x2)) / \
      (np.exp(1j*k0*x1) + target*np.exp(-1j*k0*x1))

r = materials.reflection_factor(h12, spacing=spacing, x1=x1, wavenumber=k0)
print(np.round(materials.absorption_from_reflection(r), 3))     # [0.75 0.75 0.75]
print(np.round(materials.normalized_surface_impedance(r), 2))   # Z / rho c0
# [1.15-1.23j 1.15-1.23j 1.15-1.23j]
```

The high-level `two_microphone_impedance` wraps this chain and returns an
`ImpedanceTubeResult` with absorption, reflection factor, surface impedance and
normalised impedance, applying the plane-wave frequency-range check and optional
tube attenuation; correct any microphone mismatch beforehand with
`apply_mic_calibration`. Its `.plot()` draws the absorption spectrum
$\alpha(f)$ with the reflection-factor magnitude $|r|$ overlaid.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/impedance_tube_result_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/impedance_tube_result.svg" alt="ISO 10534-2 two-microphone tube result for a 50 mm porous absorber: the normal-incidence absorption coefficient rising from about 0.2 at 200 Hz towards 0.97 above 1 kHz, with the reflection-factor magnitude falling as its mirror image" width="80%"></picture>

*A 50 mm porous absorber measured over the working band of a 100 mm tube:
the absorption climbs as the layer thickness grows against the wavelength,
and $|r|$ falls as its mirror image ($\alpha = 1 - |r|^2$).*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import materials

# A 50 mm porous absorber (Miki, sigma = 20 kPa s/m^2) in a 100 mm tube with
# 100 mm spacing (working band ~170 Hz to 1.5 kHz): the layer model supplies
# the true reflection factor, from which the transfer function H12 follows.
f = np.linspace(200.0, 1500.0, 260)
med = materials.miki(f, 20000.0)
layer = materials.layered_absorber(f, [materials.PorousLayer(0.05, med)])
spacing, x1, c0 = 0.10, 0.20, 343.2
k0 = materials.tube_wavenumber(f, c0)
x2 = x1 - spacing
r_true = layer.reflection
h12 = (np.exp(1j*k0*x2) + r_true*np.exp(-1j*k0*x2)) / \
      (np.exp(1j*k0*x1) + r_true*np.exp(-1j*k0*x1))
result = materials.two_microphone_impedance(
    h12, frequency=f, spacing=spacing, x1=x1, speed_of_sound=c0,
    characteristic_impedance=407.0, diameter=0.10,
)

# One line: alpha(f) with |r| overlaid.
result.plot()
plt.show()

# By hand, from the result's fields:
fig, ax = plt.subplots()
ax.plot(f, result.absorption, label="Absorption alpha")
ax.plot(f, np.abs(result.reflection), "--", label="Reflection factor |r|")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Coefficient")
ax.legend()
plt.show()
```

</details>

### ISO 10534-2 report (`.report()`)

`ImpedanceTubeResult.report(path)` renders a one-page PDF fiche laid out like an
accredited normal-incidence impedance-tube test report (ISO 10534-2:2001): a
standard-basis line, a metadata header block, the per-frequency table (the
absorption coefficient $\alpha$ and the real and imaginary parts of the
normalised surface impedance $z = Z/\rho c_0$) beside the $\alpha(f)$ curve (the
result's own `.plot()`, on a continuous logarithmic frequency axis), a boxed
characterisation headline and a footer with the fixed disclaimer. ISO 10534-2 is
a characterisation, so the fiche carries no pass/fail verdict and no
single-number rating; the normal-incidence coefficient is not comparable to the
random-incidence $\alpha_s$/$\alpha_w$ of ISO 354 / ISO 11654. Setting
`verbose=True` inserts the reflection-factor magnitude $|r|$ column.

It uses the same `ReportMetadata` container and rendering engine as the other
fiches. The measured frequency range is taken from the result; the applicable
descriptive and geometric `ReportMetadata` fields are `client`, `manufacturer`,
`specimen`, `tube_diameter`, `tube_shape`, `mic_spacing`, `mounting`, `test_room`,
`test_date`, `temperature`, `pressure`, `measurement_standard`, `laboratory`,
`operator`, `report_id` and `notes` (`tube_diameter` and `mic_spacing` are given
in metres and printed in millimetres). The `requirement` field is ignored
(ISO 10534-2 has no verdict). Rendering needs reportlab
(`pip install phonometry[report]`); only `engine="reportlab"` is supported. Pass
`language="es"` for a Spanish fiche (translated fixed strings and a comma
decimal separator).

```python
from phonometry import materials, ReportMetadata

result = materials.two_microphone_impedance(
    h12, frequency=freqs, spacing=0.05, x1=0.10,
    speed_of_sound=c0, characteristic_impedance=rho_c, diameter=0.10,
)
result.report(
    "alpha_fiche.pdf",
    metadata=ReportMetadata(
        specimen="Resistive facing over an 86 mm rigidly-backed air cavity",
        tube_diameter=0.10,            # m (printed as 100 mm)
        mic_spacing=0.05,              # m (printed as 50 mm)
        measurement_standard="ISO 10534-2",
        laboratory="Phonometry Reference Laboratory",
    ),
)                       # normal-incidence alpha and impedance over the tube band
```

The example fiche, regenerated with `make reports`, is kept rendered in the
repository. Click the preview to open the PDF:

[![ISO 10534-2 impedance-tube example report: a metadata header with the tube diameter, microphone spacing and measured frequency range, the per-frequency table of the absorption coefficient alpha and the real and imaginary parts of the normalised surface impedance z beside the alpha curve on a logarithmic frequency axis, and the boxed characterisation headline](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso10534_impedance_tube_example.webp)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso10534_impedance_tube_example.pdf)

*Normal-incidence impedance-tube fiche (`ImpedanceTubeResult.report`), the $\alpha$ spectrum with the surface impedance.*

## 4. Transmission loss (ASTM E2611)

With four microphones (two upstream, two
downstream of the sample) a two-load (or one-load) measurement recovers the
sample's transfer matrix, whose entries give the normal-incidence transmission
loss, reflection and wavenumber:

$$
\mathrm{TL} = 20\log_{10}\left|\frac{T_{11} + T_{12}/\rho c
             + \rho c\,T_{21} + T_{22}}{2}\right|.
$$

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_astm_tube_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_astm_tube.svg" alt="ASTM E2611 four-microphone transmission-loss tube: a sound source, two microphones upstream and two downstream of the test specimen at spacings s1 and s2 and offsets l1 and l2, an adjustable termination for the two-load method, the upstream A and B and downstream C and D travelling waves, and the transfer matrix and transmission-loss relations" width="92%"></picture>

To scale it looks like this. A `TransferMatrix` recovered by the two-load or
one-load solvers retains `l1`, `s1`, `l2`, `s2` and the specimen thickness,
so `tm.plot_geometry()` redraws the tube a measurement was taken in.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/transmission_tube_geometry_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/transmission_tube_geometry.svg" alt="To-scale side view of a 100 mm ASTM E2611 transmission tube: the loudspeaker at the left, microphones 1 and 2 upstream at s1 = 50 mm spacing and l1 = 100 mm from the specimen, the 50 mm specimen in the middle of the tube, microphones 3 and 4 downstream at s2 = 50 mm spacing and l2 = 200 mm, the changeable termination of the two-load method at the right, the circular cross-section beside the tube and the plane-wave working range of 69 to 2011 Hz" width="92%"></picture>

*Where the four microphones of the transfer-matrix method actually sit
around a 50 mm specimen in a 100 mm tube: $s_1 = s_2$ = 50 mm,
$l_1$ = 100 mm, $l_2$ = 200 mm, and the changeable termination that
provides the second load. The ASTM working range for this geometry is
69 to 2011 Hz.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
from phonometry import materials

materials.plot_transmission_tube_geometry(
    l1=0.10, s1=0.05, l2=0.20, s2=0.05, thickness=0.05, diameter=0.10,
)
plt.show()

# A TransferMatrix from transfer_matrix_two_load / _one_load retains its
# geometry, so tm.plot_geometry() redraws the tube it was measured in.
```

</details>

```python
import numpy as np
from phonometry import materials

# An air layer is a known transfer matrix; TL = 0 dB (nothing is lost)
f = np.array([500.0, 1000.0, 2000.0])
k0 = 2*np.pi*f / 343.2
rho_c = 1.186 * 343.2
tm = materials.air_layer_transfer_matrix(thickness=0.05, wavenumber=k0,
                               characteristic_impedance=rho_c)
print(np.round(tm.transmission_loss(rho_c), 6))   # [0. 0. 0.]
```

`transfer_matrix_two_load` / `transfer_matrix_one_load` build the
`TransferMatrix` from the four measured microphone transfer functions
`(H1, H2, H3, H4)` of each load; its methods
(`transmission_loss`, `reflection_hard_backed`, `absorption_hard_backed`,
`characteristic_impedance_material`, `material_wavenumber`) then read off the
ASTM E2611 quantities, and its `.plot()` draws the transmission loss with
the hard-backed absorption overlaid (a matrix built by the solvers retains
$\rho c$ and, when supplied, the frequency vector, so only a hand-built
matrix needs them as arguments). The matrix does
not have to come from a measurement: the
[multilayer solver](porous-absorbers.md) exposes the chain matrix of any
modelled stack in the same convention, so a predicted specimen can be read
out exactly like a measured one.

```python
import numpy as np
from phonometry import materials

# The chain matrix of a modelled 50 mm porous layer, read back through the
# ASTM E2611 machinery: TL and hard-backed absorption of the same specimen.
f = np.linspace(200.0, 1600.0, 300)
med = materials.miki(f, 20000.0)
layer = materials.layered_absorber(f, [materials.PorousLayer(0.05, med)])
chain = layer.transfer_matrix                       # shape (2, 2, len(f))
tm = materials.TransferMatrix(t11=chain[0, 0], t12=chain[0, 1],
                              t21=chain[1, 0], t22=chain[1, 1])
print(np.round(float(tm.transmission_loss(407.0)[-1]), 1))   # 9.7 dB at 1.6 kHz
tm.plot(f, 407.0)   # TL(f) with the hard-backed absorption overlaid
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/transfer_matrix_tl_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/transfer_matrix_tl.svg" alt="ASTM E2611 transfer-matrix quantities of a 50 mm porous layer: the normal-incidence transmission loss rising from about 6.6 dB at 200 Hz to over 9 dB at 1.6 kHz on the left axis, and the hard-backed absorption coefficient rising from 0.19 to about 0.97 on the right axis" width="80%"></picture>

*The same four-pole entries answer two different questions: how much sound
the free-standing layer lets through (the transmission loss, Eq. (26)) and
how much the rigidly-backed layer absorbs (Eq. (28)).*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import materials

f = np.linspace(200.0, 1600.0, 300)
med = materials.miki(f, 20000.0)
layer = materials.layered_absorber(f, [materials.PorousLayer(0.05, med)])
chain = layer.transfer_matrix                       # shape (2, 2, len(f))
tm = materials.TransferMatrix(t11=chain[0, 0], t12=chain[0, 1],
                              t21=chain[1, 0], t22=chain[1, 1])

# One line: TL(f) on the left axis, hard-backed absorption on the right.
tm.plot(f, 407.0)
plt.show()

# By hand, from the matrix methods:
fig, ax = plt.subplots()
ax.plot(f, tm.transmission_loss(407.0), label="Transmission loss TL_n")
twin = ax.twinx()
twin.plot(f, tm.absorption_hard_backed(407.0), "--", color="gray",
          label="Hard-backed absorption alpha")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Transmission loss TL_n [dB]")
twin.set_ylabel("Hard-backed absorption alpha")
plt.show()
```

</details>

## Mounting pitfalls

Most bad tube data are made at the sample holder, long
before the signal processing. The recurring failures:

- **Perimeter gaps.** A specimen cut slightly undersize leaves an air sliver
  along the tube wall. Sound short-circuits around and behind the sample, and
  the gap itself resonates, so the measured absorption grows a spurious
  low-to-mid-frequency hump. Cut for a snug slide fit and seal the rim (a thin
  film of petroleum jelly is the classic remedy) without loading the front
  face.
- **Compression.** A specimen cut oversize and forced in is denser than the
  product it is supposed to represent: the flow resistivity rises, a limp
  frame is stiffened, and the absorption curve shifts and flattens. The result
  is repeatable and wrong.
- **Hidden back cavity.** If the specimen does not sit flush on the rigid
  backing, the unintended air layer acts as a quarter-wavelength cavity and
  moves the absorption peak down in frequency, flattering the material.
  Backing air gaps are perfectly legitimate mountings, but only when they are
  deliberate, dimensioned and reported with the result.
- **Face not plane.** A bulging, tilted or torn front face scatters into
  cross-sectional modes below the nominal cut-on and breaks the
  normal-incidence assumption the equations rest on. Cut with a sharp tool;
  do not tear fibrous materials to size.
- **One specimen is not the material.** Porous products are inhomogeneous at
  the scale of a tube sample. Measure several cuts and average; a spread
  between specimens is product variance worth reporting, not measurement noise
  to hide.


## 5. The virtual tube: the solver measured by the standards

A three-row rigid-walled domain of the [2D FDTD solver](fdtd-simulation.md)
is a plane-wave tube, and with the per-cell `damping` map a porous sample
becomes an equivalent fluid (density, speed and loss maps). That closes a
remarkable loop: the FDTD probe histories can be reduced through the library's own ISO 10534-2 and ASTM E2611 chains as
if they were measurements, and the recovered spectra agree with the exact
analytic answer for the same lossy layer to within 0.035 in absorption and
0.1 dB in transmission loss (the `tests/simulation` cross-checks run this
on every commit).

```python
import numpy as np
from phonometry.materials import two_microphone_impedance
from phonometry.simulation import FDTD2D

c0, rho0, dx = 343.0, 1.2, 0.005
nx, d_cells = 280, 20                     # 1.4 m tube, 10 cm sample
c = np.full((3, nx), c0); rho = np.full((3, nx), rho0)
sigma = np.zeros((3, nx))
c[:, -d_cells:] = 0.6 * c0                # the sample: slower,
rho[:, -d_cells:] = 3.0 * rho0            # denser and lossy
sigma[:, -d_cells:] = 600.0
sim = FDTD2D(c, dx, rho=rho, damping=sigma,
             edge_impedance={"left": rho0 * c0})   # anechoic source end
sim.add_plane_wave("right", center=0.35, width=0.05)
mics = (219, 229)                         # the ISO microphone pair
records = np.zeros((2, 9000))
for n in range(records.shape[1]):
    sim.step()
    records[:, n] = sim.p[1, mics]
spec = np.fft.rfft(records, axis=1)
freqs = np.fft.rfftfreq(records.shape[1], sim.dt)
band = (freqs > 300.0) & (freqs < 1200.0)
result = two_microphone_impedance(
    spec[1, band] / spec[0, band], frequency=freqs[band],
    spacing=0.05, x1=(nx - d_cells) * dx - (mics[0] + 0.5) * dx,
    speed_of_sound=c0, characteristic_impedance=rho0 * c0)
result.plot()   # the absorption the virtual tube "measured"
```

The two clips below run exactly this experiment. In the impedance tube a
sustained plane tone builds the standing wave the two microphones read:
against the rigid end the minima are deep, in front of the sample they stay
shallow. In the transmission tube a carrier packet crosses an anechoic
duct: the empty tube passes it unchanged, the lossy layer splits it into a
reflection and an attenuated transmission that the four ASTM microphones
resolve.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_fdtd_impedance_tube_dark.gif"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_fdtd_impedance_tube.gif" alt="Animation: a loudspeaker drives a sustained 850 Hz plane tone into a rigid-walled virtual impedance tube drawn as the real instrument; deep envelope minima against the rigid plug, shallow minima in front of the 10 cm lossy sample, the ISO 10534-2 microphone pair and the recovered absorption of 0.54 annotated" width="640" height="360" loading="lazy"></picture>

[Watch the high-resolution video (WebM)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_fdtd_impedance_tube.webm)

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_fdtd_transmission_tube_dark.gif"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_fdtd_transmission_tube.gif" alt="Animation: the loudspeaker end fires a carrier packet down a rigid-walled virtual transmission tube drawn as the real instrument with its anechoic termination; the empty tube passes it unchanged while a 10 cm lossy layer splits it into a reflection and an attenuated transmission, the four ASTM E2611 microphones and the 3.1 dB transmission loss annotated" width="640" height="360" loading="lazy"></picture>

[Watch the high-resolution video (WebM)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_fdtd_transmission_tube.webm)


## See also

- [Sound Absorption Measurement and Rating](absorption-measurement.md): the
  random-incidence reverberation-room coefficient, its ISO 11654 rating, and
  when the tube or the room is the right instrument.
- [Airflow Resistance](airflow-resistance.md): the ISO 9053 flow resistivity
  that anchors the models a tube measurement is fitted against.
- [Porous and Multilayer Absorbers](porous-absorbers.md): the equivalent-fluid
  models and the multilayer solver whose chain matrix reads out through the
  same ASTM E2611 machinery as a measured specimen.
- [FDTD Simulation](fdtd-simulation.md): the 2D solver behind the virtual
  tubes of section 5.
- [Calibration and dBFS](calibration.md): microphone calibration ahead of
  the two-microphone transfer function.
- API reference: [`materials.impedance_tube`](https://jmrplens.github.io/phonometry/reference/api/materials/impedance-tube/).

## References

- Allard, J. F., & Atalla, N. (2009). *Propagation of sound in porous media:
  Modelling sound absorbing materials* (2nd ed.). Wiley.
  ISBN 978-0-470-74661-5.
  [doi:10.1002/9780470747339](https://doi.org/10.1002/9780470747339).
  The porous-material theory behind the measured quantities: the surface
  impedance and absorption the tube recovers, and the models fitted to them.
- International Organization for Standardization. (1996). *Acoustics —
  Determination of sound absorption coefficient and impedance in impedance
  tubes — Part 1: Method using standing wave ratio* (ISO 10534-1:1996; the
  edition implemented here is its European adoption BS EN ISO 10534-1:2001).
  [iso.org catalogue](https://www.iso.org/standard/18603.html).
  The standing-wave-ratio method of section 2.
- International Organization for Standardization. (1998). *Acoustics —
  Determination of sound absorption coefficient and impedance in impedance
  tubes — Part 2: Transfer-function method* (ISO 10534-2:1998; adopted in
  Europe as EN ISO 10534-2:2001, the edition implemented here; since
  revised as [ISO 10534-2:2023](https://www.iso.org/standard/81294.html)).
  [iso.org catalogue](https://www.iso.org/standard/22851.html).
  The two-microphone transfer-function method and its plane-wave
  frequency-range limits.
- ASTM International. (2019). *Standard test method for normal incidence
  determination of porous material acoustical properties based on the
  transfer matrix method* (ASTM E2611-19, the edition implemented here;
  since revised as [ASTM E2611-24](https://store.astm.org/e2611-24.html)).
  [ASTM store](https://store.astm.org/e2611-19.html).
  The four-microphone transfer-matrix method behind the transmission-loss
  helpers.

## Standards

BS EN ISO 10534-1:2001 (standing-wave-ratio method); BS EN ISO 10534-2:2001
(transfer-function method); ASTM E2611-19 (four-microphone transmission
loss). Every equation is derived from the standard text; the
[conformance report](CONFORMANCE.md) validates the library against
closed-form identities and synthetic round-trips, and the virtual tubes of
section 5 cross-check the FDTD solver against the same reduction chains on
every commit.
