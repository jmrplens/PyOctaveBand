← [Documentation index](../README.md)

# Elastic waves and fluid-solid coupling

A fluid carries one wave; a solid carries a family of them. This guide covers
the elastic companion of the [2D FDTD wave simulation](fdtd-simulation.md):
`elastic_fdtd_simulation` integrates the plane-strain velocity-stress system
of Virieux (1986) on the same staggered grid as the acoustic solver, and with
it the compressional **P wave**, the shear **S wave**, the **Rayleigh wave**
of a free surface and the **Scholte wave** of a fluid-solid contact all
emerge from the same update. Because a fluid is just the shear-free limit of
the elastic equations, one pair of material maps couples water, air,
sediment and steel in a single run: oblique reflection with **mode
conversion**, seabed interface waves and the transmission of immersed plates
come out of the maps alone, each pinned to its exact closed form by the
validation suite.

The guide assumes the acoustic solver's vocabulary (sources, probes,
obstacle masks, sponges and the Courant bound); read
[2D FDTD wave simulation](fdtd-simulation.md) first if those are new. The
closed forms this solver cross-checks live in the building, vibration and
underwater guides: the mass law of
[panel sound insulation](../buildings/design/panel-sound-insulation.md), the flexural waves of
[bending-wave transmission at plate junctions](../vibration/structural/junction-transmission.md),
and the shear-supporting seabed that the fluid-bottom reflection model of
[underwater sound propagation](../underwater/underwater-propagation.md) cannot represent.

## 1. The P-SV velocity-stress scheme

A solid carries more than one wave. Alongside the compressional **P wave**
at $c_\mathrm{P} = \sqrt{(\lambda + 2\mu)/\rho}$ a shear **S wave** propagates at
$c_\mathrm{S} = \sqrt{\mu/\rho}$, and every traction-free surface guides a
**Rayleigh wave** just below $c_\mathrm{S}$. `elastic_fdtd_simulation` integrates
the 2D plane-strain velocity-stress system of Virieux (1986, Eq. 2),

$$
\rho\,\frac{\partial \mathbf{v}}{\partial t}
  = \nabla\!\cdot\!\boldsymbol{\tau},
\qquad
\frac{\partial \boldsymbol{\tau}}{\partial t}
  = \lambda\,(\nabla\!\cdot\!\mathbf{v})\,\mathbf{I}
  + \mu\left(\nabla\mathbf{v} + \nabla\mathbf{v}^{\mathsf T}\right),
$$

on the same staggered layout as the acoustic solver: the normal stresses
``txx``/``tyy`` share the cell centres where the acoustic pressure lives,
the velocities sit on the faces and the shear stress ``txy`` on the
corners (Virieux's fully staggered cell, shifted half a cell). The API
mirrors `fdtd_simulation` piece by piece: wave-speed maps `c_p` and `c_s`
(plus `rho`) instead of `c`, `ExplosionSource` (an isotropic stress
injection) and `ForceSource` (a directional body force) as sources, probes
that record ``p``, ``vx`` or ``vy``, the same sponge and obstacle
machinery, and a frozen `ElasticFDTDResult` with the same `.plot()`. What
the run writes down travels as one `ElasticRecording` (the probe cells and
their fields, the snapshot cadence and its field) and how the domain ends
as one `ElasticBoundaries` (the sides and the sponge thickness). The
Courant bound depends only on the fastest $c_\mathrm{P}$ in the map (Virieux
Eqs. 6-7), while the resolution rule uses the slowest wave speed anywhere
in the domain -- the slowest non-zero $c_\mathrm{S}$ of the solid cells or, if a
fluid region carries a still slower $c_\mathrm{P}$, that fluid speed:
$\Delta x \le c_\text{min} / (10 f)$ (in a wholly solid map the S wavelength is always
the shortest; in mixed domains the fluid P wavelength can be shorter, as
in water at 1480 m/s over a 2000 m/s-shear sediment).

```python
import numpy as np
from phonometry import simulation

# An aluminium block hit by a tiny explosion: the P front reaches two
# probes 100 and 220 cells away; c_P falls out of the differential delay.
w = 8e-6
res = simulation.elastic_fdtd_simulation(
    6320.0, 3130.0, 0.002, 1.1e-4, rho=2700.0, shape=(501, 501),
    sources=[simulation.ExplosionSource(
        ix=250, iy=250,
        waveform=simulation.GaussianPulse(0, 0, width=w).value)],
    recording=simulation.ElasticRecording(
        probes=[(350, 250), (470, 250)], probe_fields=("p",)),
)
t = res.times[1:]
t1, t2 = (t[np.abs(res.signals[k, 0, 1:]).argmax()] for k in range(2))
print(round(0.24 / (t2 - t1)))    # 6316  c_P = sqrt((lambda+2mu)/rho) = 6320
```

## 2. Fluids inside the elastic solver

Setting ``c_s = 0`` marks a **fluid** cell: the shear modulus vanishes,
the system degenerates to the acoustic equations, and with a uniform
``c_s = 0`` map the elastic solver reproduces the acoustic one **bit for
bit** (the regression suite asserts exact equality). A water column over a
steel half-space is therefore just two bands of the material maps: with
the density averaged arithmetically onto the faces and the shear modulus
harmonically onto the corners (Moczo et al. 2007, Eqs. 7.37-7.39), the
traction continuity of every internal interface, fluid-solid contacts
included, emerges from the maps alone. The validation suite measures the
normal-incidence reflection of a water-steel interface within 2 % of
$(Z_2 - Z_1)/(Z_2 + Z_1)$ (typically a fraction of a percent) and recovers
the normal-incidence mass law of a 3 mm immersed steel plate within
0.3 dB of `mass_law_transmission_loss`.

## 3. Free surfaces and Rayleigh waves

A side declared ``"free"`` becomes a traction-free surface through
**stress imaging** (Moczo et al. Eq. 9.9): the normal stress is pinned to
zero on the surface plane and the shear stress above it is its
antisymmetric image. That single boundary condition is what makes surfaces
wave-bearing: strike the free surface of an aluminium block vertically
(Lamb's problem) and, besides the P and S body fronts, a **Rayleigh wave**
rolls along the surface at $c_\mathrm{R} \approx 0.93\,c_\mathrm{S}$, the root of the exact
Rayleigh characteristic equation (Cremer, Heckl & Petersson Eq. 3.149).
The tests pin the measured Rayleigh speed to that root within 2 % and the
flexural wave of a thin free-free strip to the Kirchhoff plate dispersion
$c_\mathrm{B} \propto \sqrt{\omega}$ in its thin-plate domain. Free surfaces ride
the Rayleigh sampling rule: allow 15-20 cells per wavelength there, the
second-order imaging surface being the most dispersive part of the scheme.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/elastic_halfspace_waves_dark.webp"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/elastic_halfspace_waves.webp" alt="Snapshot of the vertical particle velocity in a 0.6 by 0.3 metre aluminium block a few hundredths of a millisecond after a vertical hit on its free upper surface: the compressional P front is the outer arc, the shear S front the inner arc at about half its radius, and the strongest lobes hug the surface just behind the S front, labelled as the Rayleigh wave; dotted arcs mark the exact P and S radii" width="96%"></picture>

*One snapshot, three speeds: the P front has covered twice the distance of
the S front (dotted arcs at the exact $c_\mathrm{P} t$ and $c_\mathrm{S} t$ radii), and the
strongest motion travels along the free surface as the Rayleigh wave, a
whisker slower than S.*

<details>
<summary>Show the code for this figure</summary>

```python
import numpy as np
import matplotlib.pyplot as plt
from phonometry import simulation

# A 0.6 x 0.3 m aluminium block, struck vertically at its free surface.
dx, w = 0.001, 8e-6
dt = 0.6 * dx / (6320.0 * np.sqrt(2.0))
steps = round(7.3e-5 / dt)
res = simulation.elastic_fdtd_simulation(
    6320.0, 3130.0, dx, 7.3e-5, rho=2700.0, shape=(300, 600),
    sources=[simulation.ForceSource(
        ix=300, iy=0, direction="y", amplitude=1e6,
        waveform=simulation.GaussianPulse(0, 0, width=w).value)],
    boundaries=simulation.ElasticBoundaries({"top": "free"}),
    recording=simulation.ElasticRecording(snapshot_every=steps,
                                          snapshot_field="vy"),
)
res.plot(kind="snapshot")
plt.show()
```

</details>

One frame is enough for the two body speeds — the dotted arcs are drawn at
the exact $c_\mathrm{P} t$ and $c_\mathrm{S} t$ radii, so they measure rather than label. It is
not enough for the third wave, whose speed differs from the shear front's by
7 % and whose defining property is that it stays on the surface while the
body fronts leave. Run the same hit for 163 µs, with sponges on the other
three sides and the corner between a sponge and the free surface pushed
outside the frame, and run it twice: top declared `"free"`, then top left at
the solver default, the clamped rigid wall. Same aluminium, same hit, same
two body fronts, no surface train at all in the second.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_elastic_halfspace_waves_dark.gif"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_elastic_halfspace_waves.gif" alt="Animation: a vertical hit on an aluminium half-space, with dotted analytic arcs tracking the compressional and shear fronts as they expand and leave the frame, a train of surface lobes staying behind on the free-surface panel, and no surface train on the rigid-surface panel" width="640" height="360" loading="lazy"></picture>

[Watch the high-resolution video (WebM)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_elastic_halfspace_waves.webm)

Both body arcs run out of the frame and keep going. The surface lobes do not
go anywhere but along the surface, and they are still there when the body
waves have left — which is the whole of "a Rayleigh wave". The inset traces
two surface probes 0.15 m apart, and the delay between their peaks is the
measurement: the simulated $c_\mathrm{R}$ against 2921 m/s, the root of the exact
characteristic equation, which the tests pin to within 2 %.

The two field panels carry their own colour scales, and say so: a vertical
force applied *on* a clamped surface nearly cancels against its own image, so
that run radiates far less than the free one. The probe traces share a single
scale, which is where the comparison stays quantitative — the missing surface
arrival is a missing peak, not a rescaled one.

The other wave a free surface makes possible — the **flexural** wave of a
thin free-free strip, the one pinned to the Kirchhoff dispersion above — is
worth watching in motion too. The clip below is this same
solver launching a 4 kHz bending packet along a 10 mm steel plate: on the
control panel the plate runs straight and the packet simply leaves, and on
the junction panel a perpendicular plate of the same thickness turns the
corner into a scatterer. The packet splits there into the reflected and
transmitted bending waves the closed form prices at $\tau_{12}(0°) = 0.5$,
plus the fast in-plane precursor that races ahead down the receiving plate —
the mode conversion the pinned-junction model deliberately leaves out, and
the reason this page needs an elastic solver rather than a flexural one. The
[bending-wave transmission guide](../vibration/structural/junction-transmission.md) takes
the same run apart against the EN 12354 vibration reduction index.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_elastic_plate_junction_dark.gif"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_elastic_plate_junction.gif" alt="Animation: a 4 kHz bending-wave packet running along a 10 mm steel plate, passing straight through on the control panel and splitting at an L-junction into a reflected wave, a transmitted wave descending the perpendicular plate and a faster in-plane precursor" width="640" height="360" loading="lazy"></picture>

[Watch the high-resolution video (WebM)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_elastic_plate_junction.webm)

## 4. Fluid-solid coupling at normal incidence

Because the fluid is just the ``c_s = 0`` limit of the elastic scheme, a
water-steel contact is **one solver, two bands of the material maps**, and
the whole physics of fluid-solid coupling comes out of the same update. The
`Material` constants and `ElasticFDTD2D.from_regions` build those layered
maps without manual array surgery: a background material and a list of
painted regions.

```python
import numpy as np
from phonometry.simulation import ElasticFDTD2D, STEEL, WATER

# A 12 m water column over a steel half-space, as a 1D-like strip.
dx = 0.005
sim = ElasticFDTD2D.from_regions(
    (2400, 3), dx, background=WATER, regions=[(np.s_[1200:, :], STEEL)])
y = (np.arange(2400) + 0.5) * dx
p0 = np.exp(-(((y - 3.0) / 0.15) ** 2))[:, None]     # plane pulse
sim.txx[:] = -p0
sim.tyy[:] = -p0
y_face = np.arange(1, 2400) * dx + 0.5 * WATER.c_p * sim.dt
sim.vy += (np.exp(-(((y_face - 3.0) / 0.15) ** 2))
           / (WATER.rho * WATER.c_p))[:, None]       # one-way, leapfrog-consistent
trace = []
for _ in range(round(3.4e-3 / sim.dt)):
    sim.step()
    trace.append(sim.p[900, 1])
trace = np.asarray(trace)
t = (np.arange(trace.size) + 1) * sim.dt
incident = trace[t < 1.8e-3].max()
echo = trace[t > 2.6e-3]
print(round(float(echo[np.abs(echo).argmax()] / incident), 3))   # 0.938
# (Z2 - Z1)/(Z2 + Z1) = 0.938: at normal incidence no shear is excited
# and the steel behaves as a liquid of its rho and c_P (B&G Eq. 4.2.27)
```

## 5. Oblique incidence: mode conversion

Away from normal incidence the solid stops being a liquid. The incident
sound refracts into **two** transmitted waves, P and SV (**mode
conversion**), each with its own critical angle: for water over steel,
14.5° (P) and 27.5° (S). Between them the transmitted P is evanescent and
the shear wave carries the transmitted power, so the reflection dips to
$|V| = 0.918$ at 20° instead of reflecting totally as a fluid bottom of the
same $c_\mathrm{P}$ would; beyond the S critical angle the reflection is total,
$|V| = 1$ with a phase. The validation suite launches oblique carrier beams
at 10° and 20° and matches the exact reflection coefficient of
Brekhovskikh & Godin (Eqs. 4.2.22-4.2.26) within a fraction of a percent,
mode conversion included.

The regimes are worth watching rather than tabulating, because "the
transmitted P is evanescent" is not a shallower beam but the absence of one.
The clip runs the scene at 10°, 20° and 35°, one incidence from each of the
first three regimes, driving a sustained phase-graded beam onto the contact.
Read it as a picture of the regimes, not as a second measurement: each panel
carries the closed-form $|V|$, because the beam a 90 mm panel can hold is six
wavelengths wide and at that width a probe sits in the source's near field
rather than in a formed beam. The measured column above comes from the
validation suite, which has the room to do it properly.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_elastic_mode_conversion_dark.gif"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_elastic_mode_conversion.gif" alt="Animation: a sustained oblique beam in water hitting a steel half-space at three incidences, with two transmitted beams at ten degrees, only a shear beam at twenty degrees, and total reflection at thirty-five degrees" width="640" height="360" loading="lazy"></picture>

[Watch the high-resolution video (WebM)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_elastic_mode_conversion.webm)

At 10° two beams leave along the dashed Snell directions, at different angles
because they travel at different speeds: the slower shear wave at 22.1° from
the normal, the faster compressional wave bent much further out, to 43.8°. At
20° nothing departs at the compressional angle; a skin clings to the contact
instead, and the shear beam alone carries what crosses. At 35° both are
evanescent and nothing propagates *away* into the steel — what fills the solid
there runs along the contact and dies with depth — while the water above
settles into the interference of the incident
beam with a reflection that is total in amplitude and shifted in phase. Each
panel draws its steel half with its own annotated display gain: at 10° the
transmitted energy leaves in two beams and the amplitude anywhere is small,
while at 35° nothing leaves and the evanescent field piles up against the
contact, bright and going nowhere. Brightness in the steel is not power — read
the printed $|V|$ and the gain factor for that.

This is exactly the shear physics that the
fluid-bottom `reflection_coefficient` of the
[underwater module](../underwater/underwater-propagation.md) cannot represent, and the
reason a shear-supporting seabed loses more energy than its
equivalent-fluid model.

## 6. The Scholte interface wave

Total reflection beyond the S critical angle leaves both media carrying
only evanescent fields, and those fields can lock together into a true
interface wave: the **Scholte wave**, the fluid-solid analogue of the
Rayleigh wave. It travels slower than both the fluid sound speed and the
solid shear speed, has no low-frequency cut-off and does not disperse over
homogeneous half-spaces; `scholte_speed` solves its exact characteristic
equation (B&G Eq. 4.4.20).

```python
from phonometry.simulation import Material, STEEL, WATER, scholte_speed

print(round(scholte_speed(WATER, STEEL), 1))       # 1479.6, 0.03 % below water
seabed = Material(c_p=3500.0, c_s=2000.0, rho=2500.0)
print(round(scholte_speed(Material(1500.0, 0.0, 1000.0), seabed), 1))  # 1436.0
```

Those two numbers tell the whole story. Over a **stiff** bed (steel) the
Scholte wave hugs the water speed (1480 m/s) to within 0.03 % and its
evanescent tail reaches ~7 wavelengths up into the water: it is essentially
a grazing water wave and cannot be separated by time of flight in any
reasonable domain
(over air-solid contacts the deficit collapses to ~$10^{-12}$ of $c$ and the
wave is unobservable outright). Over a **soft** sediment the speed drops
well below the water speed and the wave squeezes to within half a
wavelength of the contact, which is why measured seabed interface waves are
a standard probe of sediment shear speed. The snapshot below runs that soft
case: an explosive shot just above the bottom, and the strongest late
feature is the interface-hugging Scholte train, timed by the test suite at
1436 m/s within 2 % between two contact probes.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/scholte_interface_wave_dark.webp"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/scholte_interface_wave.webp" alt="Snapshot of the vertical particle velocity in a 500 by 200 metre domain of water over a soft sediment half-space, 0.2 seconds after an explosive shot 10 metres above the contact: the direct water wavefront arcs up and to the right, oblique wavefronts radiate into the sediment, and the strongest lobes form a compact train hugging the dotted interface line, labelled as the Scholte wave, evanescent on both sides" width="96%"></picture>

*The van Vossen benchmark media (water over a 3500/2000/2500 sediment):
the Scholte train crawls along the contact at 1436 m/s, evanescent into
both media, while the direct water wave runs ahead at 1500 m/s.*

<details>
<summary>Show the code for this figure</summary>

```python
import numpy as np
import matplotlib.pyplot as plt
from phonometry import simulation

# Water over a soft seabed; explosive 50 Hz Ricker 10 m above the contact.
ny, nx, dx = 200, 500, 1.0
c_p = np.full((ny, nx), 1500.0)
c_s = np.zeros((ny, nx))
rho = np.full((ny, nx), 1000.0)
c_p[100:], c_s[100:], rho[100:] = 3500.0, 2000.0, 2500.0
f0, t0 = 50.0, 0.030
dt = 0.6 * dx / (3500.0 * np.sqrt(2.0))
steps = round(0.232 / dt)

def ricker(t):
    a = (np.pi * f0 * (t - t0)) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)

res = simulation.elastic_fdtd_simulation(
    c_p, c_s, dx, 0.232, rho=rho,
    sources=[simulation.ExplosionSource(ix=60, iy=89, waveform=ricker,
                                        amplitude=1e3)],
    boundaries=simulation.ElasticBoundaries("absorbing",
                                            absorbing_layer_cells=20),
    recording=simulation.ElasticRecording(snapshot_every=steps,
                                          snapshot_field="vy"),
)
res.plot(kind="snapshot")
plt.show()
```

</details>

## 7. Immersed plates and the mass law

An **immersed plate** closes the loop with the building-acoustics module.
At normal incidence no shear is excited, so an elastic plate in water is
exactly the three-media fluid layer of B&G §2.4: its transmission follows
the closed form of Eq. 2.4.14, which reduces to the familiar mass law for
thin plates and low frequencies and predicts **total transmission** at the
half-wave thickness resonances $f_n = n c_\mathrm{P} / (2 h)$ (Eq. 2.4.19). For a
10 mm steel plate that first resonance sits at 295 kHz, and one broadband
FDTD run reproduces the whole curve: 5.8 dB at 10 kHz (where
`mass_law_transmission_loss` with water as the ambient fluid agrees with
the exact form to 0.02 dB), 18.1 dB at 50 kHz, and a transmission-loss
dip within 0.1 % of the 295 kHz resonance. The same suite stress-tests the
extreme contrast of an air-steel contact (impedance ratio ~$10^5$:1): stable
over 10 000 steps with the reflected amplitude conserved to 0.5 %.

At oblique incidence the plate physics gets richer, and the clip below is
this solver driving the same 10 mm steel plate, now lying in air, with a
sustained 45° plane wave arriving on it. The two panels differ in **one
number only** — the drive frequency, $f_\mathrm{c}/2 = 603$ Hz on the left and
$2 f_\mathrm{c} = 2413$ Hz on the right, either side of the 1206 Hz coincidence
frequency the library computes from the same $m''$ and $B'$ used above.
Everything else, the plate, the angle, the mesh and the colour scale, is
held fixed. Below $f_\mathrm{c}$ the plate reflects almost everything and the
transmitted level lands on the oblique mass law; above it the acoustic trace
wavelength matches the free bending wavelength, the plate re-radiates a 45°
beam that grows along the lit span, and the transmitted level holds at the
low-frequency figure where the mass law demanded 12 dB more blocking. The
air below the plate is drawn on both panels with the display gain measured
off the settled field of the two runs together (×150, that is +44 dB) and
printed on the canvas: read the *annotations* for levels, not the
brightness. The
[panel sound insulation guide](../buildings/design/panel-sound-insulation.md) takes the
same run apart against the plateau method and the mass law.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_elastic_coincidence_dark.gif"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_elastic_coincidence.gif" alt="Animation: two elastic FDTD panels of the same 10 mm steel plate in air under a 45-degree plane wave, at 603 Hz where the plate blocks almost everything and at 2413 Hz where a transmitted beam grows below it" width="640" height="360" loading="lazy"></picture>

[Watch the high-resolution video (WebM)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_elastic_coincidence.webm)

## What this solver does not do

Everything the 2D acoustic solver cannot do, this one cannot either: the
domain is a cross-section, so a source is a line source with cylindrical
$1/\sqrt{r}$ spreading rather than the $1/r$ of a 3D point source, and
absolute levels and decay rates are not those of a 3D problem. The solid is
isotropic and purely elastic: no anisotropy, and no viscoelastic damping
beyond the bulk `damping` decay rate, so material loss factors are not
modelled. There is no elastic perfectly matched layer — the sponge absorbs
grazing Rayleigh waves less effectively than body waves. And guided-wave
dispersion beyond the thin-plate Kirchhoff regime (full Lamb modes) is
observable in the fields but has no dedicated closed-form oracle here.

## Quick answers

### When do I need the elastic solver instead of the acoustic one?

Whenever any part of the domain supports shear: mode conversion at oblique
incidence, Rayleigh waves on free surfaces, Scholte waves at fluid-solid
contacts and plate transmission all need the elastic equations. If every
cell is fluid, use the acoustic solver: with a uniform ``c_s = 0`` map the
elastic solver reproduces it bit for bit, at roughly twice the memory and
stepping cost.

### What grid resolution does an elastic simulation need?

Resolve at least 10 cells per shortest wavelength using the slowest wave
speed in the domain -- the slowest non-zero shear speed or a slower fluid
sound speed if one is present -- $\Delta x \le c_\text{min} / (10 f)$.
Allow 15-20 cells per wavelength along free
surfaces (the stress-imaging boundary is the most dispersive part of the
scheme), and at least 15 points per wavelength when an interface wave has
to be timed (van Vossen et al. 2002).

### What is a Scholte wave and how fast does it travel?

The Scholte wave is the true interface wave of a fluid-solid contact:
evanescent into both media, elliptical particle motion, no low-frequency
cut-off and non-dispersive over homogeneous half-spaces. Its speed solves
the exact characteristic equation (Brekhovskikh & Godin Eq. 4.4.20,
`scholte_speed`) and always lies below both the fluid sound speed and the
solid shear speed: 0.03 % below the water speed (1480 m/s) over steel
(1479.6 m/s),
but 4 % below the 1500 m/s ocean water of the sediment example (1436 m/s
for that water over a 3500/2000/2500 bed), which is why seabed interface
waves probe the sediment shear stiffness.

## See also

- [2D FDTD wave simulation](fdtd-simulation.md): the acoustic solver this
  guide extends, with the staggered grid, the Courant bound, the sources,
  probes and boundary machinery shared by both.
- [Predicting Panel Sound Insulation](../buildings/design/panel-sound-insulation.md): the mass
  law and coincidence closed forms; the immersed-plate run of section 7 is
  their full-wave cross-check.
- [Bending-wave transmission at plate junctions](../vibration/structural/junction-transmission.md):
  the flexural-wave closed forms of the building model; the Kirchhoff
  dispersion validated in section 3 is the wave they transport.
- [Underwater sound propagation](../underwater/underwater-propagation.md): the
  fluid-bottom Rayleigh reflection model whose missing shear physics
  section 5 quantifies.
- API reference: [`simulation.elastic-fdtd`](https://jmrplens.github.io/phonometry/reference/api/simulation/elastic-fdtd/).

## References

- Virieux, J. (1986). P-SV wave propagation in heterogeneous media:
  velocity-stress finite-difference method. *Geophysics*, 51(4), 889-901.
  [doi:10.1190/1.1442147](https://doi.org/10.1190/1.1442147). The elastic
  solver of section 1: the velocity-stress system (Eq. 2), the fully
  staggered cell and update (Fig. 1, Eq. 5), the P-only Courant bound
  (Eqs. 6-7), the dispersion relations (Eqs. 13-14) and the liquid as the
  shear-free limit.
- Moczo, P., Kristek, J., Galis, M., Pazak, P., & Balazovjech, M. (2007).
  The finite-difference and finite-element modeling of seismic wave
  propagation and earthquake motion. *Acta Physica Slovaca*, 57(2),
  177-406. The heterogeneous effective parameters of section 2 (harmonic
  shear modulus, arithmetic density; Eqs. 7.37-7.39) and the stress-imaging
  free surface of section 3 (Eq. 9.9).
- Cremer, L., Heckl, M., & Petersson, B. A. T. (2005). *Structure-borne
  sound* (3rd ed.). Springer.
  [doi:10.1007/b137728](https://doi.org/10.1007/b137728). The analytic
  oracles of the elastic solver: the exact Rayleigh characteristic
  equation (Eq. 3.149) and the Kirchhoff flexural dispersion with its
  thickness correction (Eqs. 3.83-3.89, 3.196b).
- Brekhovskikh, L. M., & Godin, O. A. (1990). *Acoustics of layered media
  I: Plane and quasi-plane waves*. Springer.
  [doi:10.1007/978-3-642-52369-4](https://doi.org/10.1007/978-3-642-52369-4).
  The fluid-solid oracles of sections 4-7: the oblique reflection
  coefficient with mode conversion (Eqs. 4.2.22-4.2.26) and its
  critical-angle limits (Eqs. 4.2.27-4.2.31), the exact Scholte
  characteristic equation (Eqs. 4.4.18-4.4.20) with the light-fluid
  asymptotics (Eqs. 4.4.21-4.4.24), and the three-media layer transmission
  (Eqs. 2.4.10-2.4.19).
- van Vossen, R., Robertsson, J. O. A., & Chapman, C. H. (2002).
  Finite-difference modeling of wave propagation in a fluid-solid
  configuration. *Geophysics*, 67(2), 618-624.
  [doi:10.1190/1.1468623](https://doi.org/10.1190/1.1468623). The
  fluid-solid benchmark of the same staggered scheme: the harmonic shear
  modulus and arithmetic density averages (Eqs. 9-10) that satisfy the
  interface conditions implicitly, the soft-bed configuration of the
  Scholte test of section 6 and the >= 15 points-per-wavelength rule for
  the O(2,2) scheme with an interface wave.

## Standards and sources

No measurement standard governs the elastic solver: it implements the
textbook velocity-stress method of Virieux (1986) with the
heterogeneous-medium and free-surface treatments of Moczo et al. (2007).
Validation is anchored to closed forms: the P and S body-wave speeds, the
exact Rayleigh root and the Kirchhoff flexural dispersion (Cremer, Heckl &
Petersson), the bit-exact acoustic limit, the fluid-solid reflection at
normal and oblique incidence with mode conversion, the exact Scholte-wave
speed and the immersed-plate transmission with its thickness resonance
(Brekhovskikh & Godin), the immersed-panel mass law, and the van Vossen
(2002) soft-bed interface-wave benchmark.
