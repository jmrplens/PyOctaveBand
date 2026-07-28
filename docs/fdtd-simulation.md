← [Documentation index](README.md)

# 2D FDTD wave simulation

Most of phonometry predicts a **number**: a level, a reverberation time, a
transmission loss. The `simulation` domain computes the **wave field
itself**: `fdtd_simulation` integrates the linear acoustic equations on a 2D
grid with the **finite-difference time-domain (FDTD)** method, so reflection,
diffraction, interference, refraction through inhomogeneous media and modal
behaviour all emerge from first principles instead of being modelled term by
term. The implementation follows the reference formulation for outdoor sound
of Attenborough & Van Renterghem, *Predicting Outdoor Sound* (2nd ed., CRC
Press 2021), chapter 4: the staggered-in-place, staggered-in-time
pressure-velocity scheme (Eqs. 4.11-4.12), the Courant stability condition
(Eqs. 4.13-4.14), rigid boundaries as zero normal face velocity (Eq. 4.32)
and the frequency-independent real-impedance boundary (Eqs. 4.33-4.35).

The solver is **deterministic by design**: float64 arithmetic, no random
numbers and single-threaded numpy stepping, so the same inputs produce
bit-identical outputs on the same platform. It is the engine behind the FDTD
animations of this documentation, promoted to a public API with sources,
pressure probes, rasterised obstacles, per-side boundary conditions and a
frozen result object.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_fdtd_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_fdtd.svg" alt="Pipeline from the domain definition (sound-speed and density maps with the grid spacing dx) and the geometry (obstacle mask and per-side boundary conditions), through the sources injected at grid cells, the staggered-grid leapfrog update of velocity and pressure, and the Courant stability condition, to the frozen FDTDResult with probe histories, field snapshots and a plot method" width="86%"></picture>

## 1. The scheme: a wave equation on a grid

In a non-moving medium the linearised equations of fluid dynamics reduce to a
first-order system in the acoustic pressure ``p`` and particle velocity ``v``
(Attenborough & Van Renterghem Eqs. 4.3-4.4):

$$
\frac{\partial p}{\partial t} + \rho c^2\,\nabla\!\cdot\!\mathbf{v} = 0,
\qquad
\frac{\partial \mathbf{v}}{\partial t} + \frac{1}{\rho}\,\nabla p = 0 .
$$

FDTD discretises both on a **staggered grid** (the acoustic analogue of the
Yee cell): pressure lives at cell centres and each velocity component on the
cell faces, half a cell away, and the two fields **leapfrog** in time, half a
time step apart (Eqs. 4.11-4.12). Evaluating each spatial gradient exactly
where the other field needs it gives a fourfold accuracy gain over a
collocated grid (Eq. 4.9 vs 4.10) and allows in-place updates. Because only
interior faces are stored, the domain edge is a perfectly **rigid wall**
(zero normal velocity, Eq. 4.32) unless another boundary is requested.

The explicit scheme is stable only while a wavefront crosses at most one cell
per time step. With square cells the **Courant number** (Eq. 4.13) is

$$
\mathrm{CN} = c\,\Delta t\sqrt{\frac{1}{\Delta x^2} + \frac{1}{\Delta y^2}}
            = \frac{c\,\Delta t\,\sqrt{2}}{\Delta x} \le 1,
$$

and `fdtd_simulation` derives the time step from the `cfl` parameter (the
Courant number, default 0.6) and the largest sound speed in the map; values
outside ``(0, 1)`` are rejected because the scheme is unconditionally
meaningless beyond the bound (Eq. 4.14).

```python
from phonometry import simulation

# A 3.0 x 2.0 m air domain: 300 x 200 cells of 1 cm.
res = simulation.fdtd_simulation(
    343.0, 0.01, 2.0e-3, shape=(200, 300),
    sources=[simulation.GaussianPulse(ix=60, iy=100, width=3.0e-4)],
    probes=[(200, 100)],
)
print(res.size)                  # (3.0, 2.0)  metres
print(round(res.dt * 1e6, 2))    # 12.37  microseconds (CN = 0.6)
res.plot()                       # probe pressure histories (figure in §3)
```

The grid is index-based: cell ``(ix, iy)`` has its centre at
``((ix + 0.5) * dx, (iy + 0.5) * dx)`` metres, with rows plotted downward
(the ``imshow`` convention), so a position in metres maps to
``ix = round(x / dx - 0.5)``.

## 2. Sources, probes, obstacles and boundaries

Three source types inject a **soft source** (an additive pressure
contribution that does not scatter passing waves) at a grid cell:
`GaussianPulse` (a broadband pulse of temporal half-width `width`),
`CWSource` (a sine tone faded in with a raised-cosine ramp so its onset does
not splash a broadband transient) and `SignalSource` (an arbitrary sampled
waveform, linearly interpolated onto the simulation time steps). Probes
record the pressure at their cell every time step into the result.

**Plane waves "from infinity".** Point sources in 2D are really line
sources, so a diffuser or a barrier is often better interrogated with a
plane wavefront. Two tools cover it, each one-way through its own mechanism:

- `sim.add_plane_wave(direction, center=..., width=..., wavelength=...)`
  superimposes a Gaussian packet (optionally carrying a sine) as an
  initial condition travelling toward `"down"`, `"up"`, `"left"` or
  `"right"`; the leapfrog-consistent velocity written half a step back
  is what makes it one-way, and behind the front the residual energy is
  at numerical noise level. This is what the QRD diffusion animation
  uses.
- `PlaneWaveSource(direction, waveform, offset=...)` registered through
  `add_source()` injects a sustained plane wave on a line of cells: the
  incident pressure and the adjacent face velocity are driven together,
  so the launched field is transversely plane to machine precision and
  anything scattered back crosses the line untouched; with a sponge
  configured behind the line it is then absorbed (about -38 dB residual
  for a CW in the set-up above; the figure depends on the sponge).

```python
from phonometry.simulation import FDTD2D, CWSource, PlaneWaveSource

sim = FDTD2D(343.0, 0.01, shape=(160, 80), sponge_width=20,
             sponge_sides=("top", "bottom"))
tone = CWSource(0, 0, frequency=1000.0)          # reused as a waveform
sim.add_source(PlaneWaveSource("down", tone.value, offset=22))
# or, for a single packet: sim.add_plane_wave("down", center=0.4,
#                                             width=0.08, wavelength=0.34)
```

Geometry is **rasterised**: `obstacle_mask` marks rigid cells, and every
face touching a masked cell is closed (Eq. 4.32 again), so walls, barriers
and scatterers of any shape are just boolean arrays. Each domain side can
carry its own boundary condition:

- ``"rigid"`` (default): a perfect reflector, ``R = +1``.
- ``"absorbing"``: a sponge layer of `absorbing_layer_cells` cells whose
  absorption rate ramps quadratically, emulating an open boundary (the
  simple precursor of the perfectly matched layers of section 4.2.3).
- a **real specific impedance** ``Z`` in Pa·s/m (a scalar or one value per
  edge cell): the locally reacting boundary of Eqs. 4.33-4.35, updated
  implicitly, with the normal-incidence reflection coefficient
  ``R = (Z - ρc)/(Z + ρc)``; ``Z = ρc`` is anechoic.

The stepping engine `FDTD2D` is public too: it exposes `step()`, `run()`,
the field arrays and the energy, for callers that need frame-by-frame access
(the documentation animations use it directly). A plane pulse launched down
a duct against an impedance edge reproduces the textbook reflection
coefficient:

```python
import numpy as np
from phonometry import simulation

rho, c, dx = 1.2, 343.0, 0.01
sim = simulation.FDTD2D(c, dx, rho=rho, shape=(3, 1200),
                        edge_impedance={"right": 3.0 * rho * c})
x = (np.arange(1200) + 0.5) * dx
sim.p[:] = np.exp(-(((x - 6.0) / 0.15) ** 2))[None, :]   # plane pulse

trace = []
for _ in range(int(round(0.032 / sim.dt))):
    sim.step()
    trace.append(sim.p[1, 900])
trace = np.asarray(trace)
t = (np.arange(trace.size) + 1) * sim.dt
t_return = 6.0 / c + 3.0 / c                  # via the wall, back to x = 9 m
incident = trace[t < t_return - 0.001].max()
echo = trace[t > t_return]
print(round(float(echo[np.abs(echo).argmax()] / incident), 2))  # 0.5
# (Z - rho c)/(Z + rho c) = (3 - 1)/(3 + 1) = +0.5
```

Before spending a single time step, `sim.plot_geometry()` draws the
configured domain: edges, sponges, obstacles, sources and the probes you
intend to record. Catching a sponge on the wrong side or a probe in the
wrong place costs seconds here and a full re-run later.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/fdtd_domain_geometry_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/fdtd_domain_geometry.svg" alt="Setup drawing of a 4.5 by 3 metre FDTD domain before any time stepping: pale blue sponge layers along the left and right edges, an orange impedance edge along the top, a grey rectangular obstacle just left of centre, the source star at (0.5, 1.5) and two probe circles at (3, 1.5) and (4, 2), with a legend naming the sponge layer, impedance edge, rigid edge, source and probe" width="88%"></picture>

*Everything the run will see, before it runs: the sponge layers eat the
left and right boundaries, the top edge carries the anechoic `ρc` impedance,
the untreated bottom stays rigid, and both probes sit clear of the obstacle
and the sponges.*

<details>
<summary>Show the code for this figure</summary>

```python
import numpy as np
import matplotlib.pyplot as plt
from phonometry import simulation

mask = np.zeros((60, 90), dtype=bool)
mask[25:35, 40:44] = True
sim = simulation.FDTD2D(343.0, 0.05, shape=(60, 90), sponge_width=8,
                        sponge_sides=("left", "right"),
                        edge_impedance={"top": 413.0}, obstacle_mask=mask)
sim.add_source(simulation.GaussianPulse(10, 30, width=1e-3))

# Check the domain before running it: nothing has been stepped yet.
sim.plot_geometry(probes=[(3.0, 1.5), (4.0, 2.0)])
plt.show()
```

</details>

A three-row rigid-walled domain is a plane-wave tube, and with the per-cell
`damping` map a porous sample becomes an equivalent fluid, which turns the
solver itself into a measurable specimen: the
[impedance-tube guide](impedance-tube.md) runs the ISO 10534-2 and
ASTM E2611 measurements virtually on exactly that domain, animations
included, and recovers the analytic absorption and transmission loss of the
modelled sample through the library's own reduction chains (the
`tests/simulation` cross-checks run this on every commit).

## 2b. From the near field to the far field

A polar response or a diffusion coefficient is a far-field quantity, but an
FDTD box ends a couple of wavelengths from the scatterer. The
`add_contour_probe` / `far_field_from_contour` pair bridges that gap with
the 2D Kirchhoff-Helmholtz integral: the probe folds the steady-state
pressure and outward normal velocity on a closed rectangle of cell faces
into complex accumulators (an on-the-fly DFT per point and frequency, so a
continuous-wave run stores no time histories), and the integral propagates
those phasors to infinity with the free-space Green function
`-(j/4) H0⁽²⁾(kR)` — the same construction full-wave FEM solvers use to
report scattering patterns from near-field data. Two properties carry the
scheme: the staggered grid already holds `v_n` exactly on the contour
faces, and any field whose sources lie *outside* the contour integrates to
nothing (extinction), so the total-field phasors of a plane-wave scattering
run transform directly into the *scattered* far field. In the discrete
solver that cancellation leaves a grid-dispersion residual (below 0.01 dB
on the scenes validated here); `ContourPhasors.subtract()` removes it with
a no-scatterer reference run when that last fraction matters.

```python
import numpy as np
from phonometry import simulation

c0, dx, f0 = 343.0, 0.005, 2000.0
sim = simulation.FDTD2D(c0, dx, shape=(300, 300), sponge_width=40)
sim.add_source(simulation.CWSource(ix=150, iy=150, frequency=f0))
probe = sim.add_contour_probe(90, 210, 90, 210, frequencies=[f0])
sim.run(round(4.5e-3 / sim.dt))            # run the transient out
probe.reset()                              # then integrate the DFT window
sim.run(round(10.0 / f0 / sim.dt))
pattern = simulation.far_field_from_contour(
    probe.phasors(f0), np.arange(0.0, 360.0, 5.0),
    origin=(150.5 * dx, 150.5 * dx))
levels = 20 * np.log10(np.abs(pattern))
print(round(float(levels.max() - levels.min()), 2))   # 0.04 dB of ripple:
# a line source is omnidirectional, and its level matches the 2D
# free-space Green function to 0.11 dB
```

The analytic oracles behind those numbers run on every commit
(`tests/simulation`): the reconstructed monopole pattern is flat within
0.05 dB and sits on the 2D Green-function level within 0.11 dB, an
antiphase pair reproduces the two-source array factor within 0.4 % of the
peak, and a contour that does not enclose the source transforms to less
than 1 % of one that does. On top of them sit two meshed-panel
cross-checks at 2 kHz: a quadratic-residue diffuser with wells to 27.4 cm,
whose NTFF polar response tracks the Fraunhofer prediction of
`predict_diffuser_polar_response` (pattern correlation 0.94, ISO
17497-2 directional diffusion coefficient within 0.08), and the deep
subwavelength metadiffuser below.

The far-field chain is what finally closes the metadiffuser loop end to
end: the Table-1 panel of the paper — every slit, neck and cavity meshed
at 0.5 mm — driven to steady state by a plane wave, captured on a contour
and transformed, against the `metadiffuser_polar_response` model (TMM +
Fraunhofer) of the materials module. That is the library-side counterpart
of the TMM-vs-FEM comparison the metadiffuser paper itself reports, small
discrepancies included: the transfer-matrix model homogenises each 7 cm
cell into one locally reacting reflection coefficient and ignores the
evanescent coupling between neighbouring mouths, so the lobe structure
agrees while individual nulls shift by a few degrees.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/metadiffuser_ntff_polar_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/metadiffuser_ntff_polar.svg" alt="Semicircular polar plot at 2 kHz overlaying the far-field response of the meshed Table-1 metadiffuser computed by FDTD plus the Kirchhoff-Helmholtz near-to-far-field integral, solid line, on the TMM plus Fraunhofer prediction of the library, dashed line: the specular and grating lobes coincide over the arc from minus 90 to plus 90 degrees with small shifts of the deep nulls" width="92%"></picture>

*Two independent routes to the same far field: the meshed panel in a
full-wave time-domain solver (solid) against the homogenised
transfer-matrix chain (dashed). The lobes agree; the nulls, sensitive to
the millimetre end corrections the TMM models analytically, shift by a
few degrees.*

<details>
<summary>Show the code for this figure</summary>

```python
import numpy as np
import matplotlib.pyplot as plt
from phonometry import (HelmholtzResonator, MetadiffuserWell,
                        metadiffuser_polar_response, simulation)

c0, dx, f0, pitch = 343.0, 0.0005, 2000.0, 0.07
rows = [(14.7, 13.0, 16.4, 6.2, 9.0), (30.9, 9.1, 4.3, 3.5, 9.0),
        (30.9, 9.1, 4.3, 3.5, 9.0), (15.7, 13.3, 17.0, 6.3, 9.0),
        (20.3, 18.0, 20.7, 3.2, 9.0)]     # Table 1: h, l_n, l_c, w_n, w_c

# Mesh the real panel: a rigid slab with the slits, necks and cavities
# carved out at 0.5 mm (the 3.2 mm narrowest neck spans six cells).
sponge, gap, marg, front = 60, 60, 20, 40
face = round(5 * pitch / dx)
lat = marg + gap + sponge
r_face = sponge + gap + front
slab = round(0.023 / dx)                  # 2 cm panel + 3 mm back wall
mask = np.zeros((r_face + slab + marg + gap + sponge,
                 face + 2 * lat), dtype=bool)
mask[r_face:r_face + slab, lat:lat + face] = True
for n, (h, ln, lc, wn, wc) in enumerate(rows):
    xs = (n + 0.12) * pitch
    c0s, c1s = lat + round(xs / dx), lat + round((xs + h * 1e-3) / dx)
    mask[r_face:r_face + round(0.02 / dx), c0s:c1s] = False
    for m in range(2):                    # two resonators per slit
        ym, xn = (m + 0.5) * 0.01, xs + h * 1e-3
        r0 = r_face + round((ym - 0.5e-3 * wn) / dx)
        r1 = r_face + round((ym + 0.5e-3 * wn) / dx)
        mask[r0:r1, c1s:lat + round((xn + ln * 1e-3) / dx)] = False
        r0 = r_face + round((ym - 0.5e-3 * wc) / dx)
        r1 = r_face + round((ym + 0.5e-3 * wc) / dx)
        mask[r0:r1, lat + round((xn + ln * 1e-3) / dx):
             lat + round((xn + (ln + lc) * 1e-3) / dx)] = False

sim = simulation.FDTD2D(c0, dx, shape=mask.shape, sponge_width=sponge,
                        cfl=0.9, obstacle_mask=mask)   # 340 cells/lambda:
sim.add_source(simulation.PlaneWaveSource(           # dispersion negligible
    "down", simulation.CWSource(0, 0, f0).value, offset=sponge))
probe = sim.add_contour_probe(lat - marg, lat + face + marg - 1,
                              r_face - front, r_face + slab + marg - 1,
                              frequencies=[f0])
sim.run(round(8e-3 / sim.dt))             # transient (ramp + ring-up) out
probe.reset()
sim.run(round(10.0 / f0 / sim.dt))        # a 10-period DFT window (in steps)
angles = np.arange(-90.0, 90.1, 5.0)      # from the panel normal
pattern = simulation.far_field_from_contour(
    probe.phasors(f0), angles - 90.0,     # the normal points along -y
    origin=((lat + face / 2.0) * dx, r_face * dx))
levels = 20 * np.log10(np.abs(pattern) / np.abs(pattern).max())

wells = [MetadiffuserWell(h * 1e-3,
                          (HelmholtzResonator(ln * 1e-3, wn * 1e-3,
                                              lc * 1e-3, wc * 1e-3),) * 2)
         for h, ln, lc, wn, wc in rows]
model = metadiffuser_polar_response(f0, wells, depth=0.02, period=pitch,
                                    angles=angles, periods=1)
ax = model.plot(color="#1f77b4", marker="", linestyle="--",
                label="TMM + Fraunhofer model")
ax.plot(np.radians(angles), levels, color="#d62728", lw=2.2,
        label="FDTD + NTFF, panel meshed at 0.5 mm")
ax.set_ylim(-40.0, 2.0)
ax.legend(loc="lower center")
plt.show()
```

</details>

## 3. When to use it, and the 2D limits

FDTD earns its cost when the **geometry drives the physics**: diffraction
around a barrier or through an opening, interference of direct and reflected
paths, scattering from obstacles, modal behaviour of odd-shaped enclosures,
refraction through a sound-speed gradient. One run captures **all
frequencies at once** (a pulse excites the whole band; one FFT of a probe
yields the spectrum), where a frequency-domain method needs one solve per
frequency.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/fdtd_simulation_dark.webp"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/fdtd_simulation.webp" alt="Two panels. Left: a snapshot of the pressure field in a 3 by 2 metre domain with a thin vertical rigid barrier, showing the direct wavefront, the reflection travelling back towards the source and the wave diffracted around the barrier edge, with the source marked by a star and two probes by dots. Right: the pressure history at both probes; the line-of-sight probe shows the direct pulse and the barrier reflection, the shadowed probe a weaker, delayed diffracted arrival" width="96%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import numpy as np
import matplotlib.pyplot as plt
from phonometry import simulation

# A 3.0 x 2.0 m free field (absorbing edges) with a thin rigid barrier:
# probe A sees the direct pulse plus the barrier reflection, probe B sits
# in the shadow and only receives the wave diffracted around the edge.
mask = np.zeros((200, 300), dtype=bool)
mask[60:, 150:154] = True
res = simulation.fdtd_simulation(
    343.0, 0.01, 9.0e-3, shape=(200, 300),
    sources=[simulation.GaussianPulse(ix=60, iy=100, width=3.0e-4)],
    probes=[(100, 100), (240, 100)],
    obstacle_mask=mask,
    boundaries="absorbing", absorbing_layer_cells=30,
    snapshot_every=75,
)

fig, (ax_f, ax_p) = plt.subplots(
    1, 2, figsize=(12.5, 5.0), gridspec_kw={"width_ratios": [1.25, 1.0]})
res.plot(kind="snapshot", frame=7, ax=ax_f)
res.plot(ax=ax_p)
plt.tight_layout()
plt.show()
```

</details>

Conversely, when a validated closed form exists (statistical
reverberation, ISO 9613-2 outdoor attenuation, image sources in a rectangular
room) the closed form is thousands of times cheaper: this solver is the
cross-check and the demonstrator, not the replacement. The oracle works both
ways; a rigid-box run reproduces the analytic room modes:

```python
import numpy as np
from phonometry import simulation

lx, ly, dx = 1.0, 0.7, 0.02
nx, ny = round(lx / dx), round(ly / dx)
res = simulation.fdtd_simulation(
    343.0, dx, 0.35, shape=(ny, nx),
    sources=[simulation.GaussianPulse(ix=7, iy=5, width=2.0e-4)],
    probes=[(nx - 4, ny - 3)],
)
p = res.pressures[0]
spec = np.abs(np.fft.rfft(p * np.hanning(p.size), n=8 * p.size))
freqs = np.fft.rfftfreq(8 * p.size, res.dt)
sel = (freqs > 250) & (freqs < 350)
print(round(0.5 * 343.0 * float(np.hypot(1 / lx, 1 / ly)), 1))  # 299.1  exact (1,1) mode
print(round(float(freqs[sel][np.argmax(spec[sel])]), 1))        # 298.9  measured
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/fdtd_room_modes_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/fdtd_room_modes.svg" alt="Spectrum of the probe pressure of a rigid 1.0 by 0.7 metre FDTD box between 100 and 450 Hz: five sharp peaks that land on the dotted analytic mode frequencies of the (1,0), (0,1), (1,1), (2,0) and (2,1) modes" width="88%"></picture>

*The probe spectrum of the rigid-box run peaks exactly on the analytic mode
frequencies $f = (c/2)\sqrt{(n_x/L_x)^2 + (n_y/L_y)^2}$ (Kuttruff 6e,
Ch. 3). The barely visible leftward offset of the highest peaks is the
numerical dispersion of the accuracy section below: short wavelengths
propagate slightly slow on the grid, so the modelled resonances under-read
by a fraction of a percent.*

<details>
<summary>Show the code for this figure</summary>

```python
import numpy as np
import matplotlib.pyplot as plt
from phonometry import simulation

lx, ly, dx, c = 1.0, 0.7, 0.02, 343.0
nx, ny = round(lx / dx), round(ly / dx)
res = simulation.fdtd_simulation(
    c, dx, 0.35, shape=(ny, nx),
    sources=[simulation.GaussianPulse(ix=7, iy=5, width=2.0e-4)],
    probes=[(nx - 4, ny - 3)],
)

# One line: the raw probe pressure history the spectrum is computed from.
res.plot()
plt.show()

# The mode check: probe spectrum against the analytic rigid-room modes.
p = res.pressures[0]
spec = np.abs(np.fft.rfft(p * np.hanning(p.size), n=8 * p.size))
freqs = np.fft.rfftfreq(8 * p.size, res.dt)
sel = (freqs >= 100) & (freqs <= 450)
fig, ax = plt.subplots()
ax.plot(freqs[sel], 20 * np.log10(spec[sel] / spec[sel].max()))
for mx, my in [(1, 0), (0, 1), (1, 1), (2, 0), (2, 1)]:
    ax.axvline(0.5 * c * np.hypot(mx / lx, my / ly), ls=":", color="tab:red")
ax.set(xlabel="Frequency [Hz]", ylabel="Probe spectrum [dB re max]",
       ylim=(-60, 6))
plt.show()
```

</details>

The domain is **two-dimensional**, and that changes the physics, not just
the cost. A 2D point source is physically an infinite **line source**: its
amplitude spreads cylindrically as ``1/sqrt(r)`` (3.0 dB per doubling of
distance) instead of the spherical ``1/r`` (6.0 dB) of a 3D point source,
and the 2D impulse response trails a wake behind the wavefront instead of
passing cleanly. Interference and diffraction *patterns* are faithful;
absolute levels and decay rates are not those of a 3D room. Treat 2D runs as
cross-sections and demonstrations, and validate any 3D-quantitative claim
against a closed form or a 3D solver.

## 4. Numerical dispersion and accuracy

The discrete grid propagates each frequency at a slightly wrong speed: short
wavelengths lag, so a sharp pulse develops a ripple tail and resonances shift
slightly. This **numerical dispersion** is the discrete counterpart of
Eq. 4.15; on the axes of a square grid the scheme's dispersion relation is

$$
\sin\!\left(\frac{\omega\,\Delta t}{2}\right)
  = \frac{c\,\Delta t}{\Delta x}\,
    \sin\!\left(\frac{k\,\Delta x}{2}\right),
$$

with a leading-order relative frequency error of magnitude
``(1 - S^2) (k dx)^2 / 24`` along the grid axes (the modelled frequency
under-reads, so the signed error is negative), where ``S = c dt / dx``;
the error is largest exactly on-axis and vanishes along the cell diagonal
at the Courant limit ``CN = 1``. The practical rule is to resolve **at
least 10 cells per shortest wavelength**, ``dx <= c_min / (10 f_max)``
with the smallest sound speed of the domain: at exactly 10 cells the
small-Courant bound ``(k dx)^2 / 24`` evaluates to about 1.6 %, reduced
by the ``1 - S^2`` factor to about 1.4 % at the default ``cfl = 0.6``
(in a heterogeneous domain the time step follows the fastest cells, so
slower regions run at a lower local Courant number and sit nearer the
1.6 % bound), and every finer-resolved or off-axis component is more
accurate. With ``dx = 1`` cm the 10-cell point in air sits at roughly
3.4 kHz, and halving ``dx`` quarters the error (the scheme is second
order, and the validation suite measures that observed order under grid
refinement). The tests pin the
solver to analytic oracles: box and duct eigenfrequencies, free-field
arrival times and cylindrical decay, the rigid-wall image echo, the
impedance reflection coefficient above and the dispersion relation itself.

The frozen `FDTDResult` carries the time axis, the per-probe pressure
histories, the probe positions in metres, the grid metadata, the sources,
the optional field snapshots with their times and the obstacle mask; its
`.plot()` draws the probe histories, and `.plot(kind="snapshot")` renders
one recorded field with the geometry overlaid.

The same staggered lattice extends beyond fluids: the companion
`elastic_fdtd_simulation` integrates the P-SV velocity-stress system of
Virieux (1986) on the same grid, adding shear waves, stress-imaging free
surfaces with Rayleigh waves, and fluid-solid coupling with mode
conversion, Scholte interface waves and immersed-plate transmission. That
solver has its own guide,
[Elastic waves and fluid-solid coupling](elastic-waves.md).

## Quick answers

### How do I choose the FDTD grid spacing?

Resolve at least 10 cells per shortest wavelength,
``dx <= c_min / (10 f_max)``, using the smallest sound speed in the domain.
At exactly 10 cells the on-axis dispersion error bound is about 1.6 %,
reduced to about 1.4 % at the default ``cfl = 0.6``, and halving ``dx``
quarters the error (the scheme is second order). With ``dx = 1`` cm the
10-cell point in air sits at roughly 3.4 kHz.

### What Courant number keeps an FDTD simulation stable?

The explicit scheme is stable only while a wavefront crosses at most one
cell per time step. With square cells the Courant number is
$\mathrm{CN} = c\,\Delta t\,\sqrt{2}/\Delta x \le 1$ (Attenborough & Van
Renterghem Eq. 4.13). `fdtd_simulation` derives the time step from the
`cfl` parameter (the Courant number, default 0.6) and the largest sound
speed in the map, rejecting values outside $(0, 1)$.

### Can I trust the absolute levels from a 2D FDTD run?

No. A 2D point source is physically an infinite line source: its amplitude
spreads cylindrically as ``1/sqrt(r)``, 3.0 dB per doubling of distance,
instead of the spherical ``1/r`` (6.0 dB per doubling) of a 3D point source.
Interference and diffraction patterns are faithful, but absolute levels and
decay rates are not those of a 3D room; validate any 3D-quantitative claim
against a closed form or a 3D solver.

## References

- Attenborough, K., & Van Renterghem, T. (2021). *Predicting outdoor sound*
  (2nd ed.). CRC Press.
  [doi:10.1201/9780429470806](https://doi.org/10.1201/9780429470806).
  Chapter 4: the pressure-velocity FDTD reference model implemented here,
  from the governing equations (4.3-4.4) and the staggered leapfrog update
  (4.11-4.12) through the Courant condition (4.13-4.14), the phase-error
  analysis (4.15) and the rigid and finite-impedance boundary conditions
  (4.32-4.35).
- Kuttruff, H. (2016). *Room acoustics* (6th ed.). CRC Press.
  [doi:10.1201/9781315372150](https://doi.org/10.1201/9781315372150).
  Section 3.5 places time-domain wave-based methods among the numerical
  approaches to the wave equation in enclosures, and chapter 3 gives the
  rigid-room normal modes used as the analytic oracle.
- Williams, E. G. (1999). *Fourier acoustics: Sound radiation and
  nearfield acoustical holography*. Academic Press.
  [doi:10.1016/B978-0-12-753960-7.X5000-1](https://doi.org/10.1016/B978-0-12-753960-7.X5000-1).
  Chapter 8: the Helmholtz integral equation behind
  `far_field_from_contour`, with the outgoing free-space Green function
  and the far-field limit of §2b.
- Jiménez, N., Cox, T. J., Romero-García, V., & Groby, J.-P. (2017).
  Metadiffusers: Deep-subwavelength sound diffusers. *Scientific Reports*,
  7, 5389. [doi:10.1038/s41598-017-05710-5](https://doi.org/10.1038/s41598-017-05710-5).
  The meshed Table-1 panel and the near-to-far-field polar comparison of
  §2b reproduce, with this solver, the TMM-vs-full-wave cross-check the
  paper reports.

## Standards and sources

The module implements a textbook numerical method rather than a measurement
standard: the discretisation, stability bound and boundary conditions follow
Attenborough & Van Renterghem (2021) chapter 4 as the citable reference
formulation. Validation is anchored to closed forms (rigid-box
eigenfrequencies, cylindrical spreading, image sources, the
normal-incidence reflection coefficient and the scheme's dispersion
relation); two of those anchors run in the
[conformance report](CONFORMANCE.md). The elastic P-SV companion solver
and its own oracles are covered in
[Elastic waves and fluid-solid coupling](elastic-waves.md).
