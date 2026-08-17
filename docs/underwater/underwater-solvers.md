← [Documentation index](../README.md)

# Underwater propagation solvers: normal modes, rays and the parabolic equation

The closed-form propagation loss of
[Underwater sound propagation](underwater-propagation.md) knows nothing of
the sound-speed profile, the seabed or the surface. When refraction and
boundaries decide the answer, the field has to be **computed**: this guide
covers the four numerical solvers of the `underwater` module, the physics
each one discretises, and how to choose between them and the closed forms.
All four assume a range-independent (horizontally stratified) ocean with a
pressure-release surface, take the same $c(z)$ profile as input, and follow
Jensen, Kuperman, Porter & Schmidt, *Computational Ocean Acoustics*.

## 1. The four solvers at a glance

For range-independent (horizontally stratified) environments the field can be
computed numerically. Four solvers are provided (Jensen et al.,
*Computational Ocean Acoustics*):

- **`normal_modes`** solves the depth-separated Sturm-Liouville eigenvalue
  problem by finite differences and sums the propagating modes into the
  propagation loss. Validated against the ideal (pressure-release) waveguide's
  exact modes.
- **`ray_trace`** integrates the ray-trajectory equations (Runge-Kutta,
  vectorised over all rays at once) through a sound-speed profile, reflecting at
  the surface and bottom, and carries the travel time along each ray as a state
  of the same integration. Validated against the circular-arc paths of a linear
  gradient and the closed-form travel time along them.
- **`parabolic_equation`** marches the standard (Tappert) PE with the split-step
  Fourier algorithm. Validated against free-field spherical spreading; it agrees
  with the normal-mode propagation loss in trend.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/numerical_propagation_dark.webp">
  <img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/numerical_propagation.webp" alt="Three numerical solvers: a Munk sound-speed profile, ray paths forming convergence zones, and propagation loss versus range from the normal-mode and parabolic-equation solvers agreeing in trend" width="100%">
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
field.plot()   # PL(z, r) field showing the convergence zones
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
field.plot()  # PL field over range x depth (needs matplotlib)
```

`normal_modes` returns a `NormalModeResult` (`wavenumbers`, `mode_functions`,
`propagation_loss`); `ray_trace` a `RayTraceResult` (`ranges`, `depths` and
`travel_times` per ray); `parabolic_equation` a `ParabolicEquationResult` (the
`propagation_loss` field). All assume a range-independent water column with a pressure-release
surface, and the bottom is pressure-release too (or, for the modes, optionally
rigid): there is no absorbing or elastic bottom, no sediment attenuation and
no real bathymetry, so range-dependent problems are out of scope. For the
elastic seabed physics these fluid solvers leave out, see
[Elastic waves and fluid-solid coupling](../simulation/elastic-waves.md).

## 2. Normal modes: the waveguide as a sum of standing waves

In a horizontally stratified ocean the Helmholtz equation separates in
cylindrical coordinates, $p(r, z) = \Phi(r)\,\Psi(z)$, and the depth factor
obeys a Sturm-Liouville eigenvalue problem (Jensen Eq. 5.3):

$$
\frac{d^2 \Psi_m}{dz^2}
  + \left[\frac{\omega^2}{c^2(z)} - k_{rm}^2\right] \Psi_m = 0,
\qquad \Psi_m(0) = 0,
$$

with a pressure-release surface at $z = 0$ and, at the bottom $z = D$,
$\Psi(D) = 0$ for a pressure-release bed or $d\Psi/dz|_{D} = 0$ for a
rigid one. Each eigenfunction $\Psi_m(z)$ is a standing wave in depth that
travels in range as $e^{\,i k_{rm} r}$ with its own horizontal wavenumber
$k_{rm}$; only the modes with real $k_{rm}$ **propagate**, the rest are
evanescent and die within a few water depths. The field is the modal sum
(Eq. 5.14),

$$
p(r, z) \simeq \frac{i\,e^{-i\pi/4}}{\rho(z_\mathrm{s})\sqrt{8\pi r}}
  \sum_m \Psi_m(z_\mathrm{s})\,\Psi_m(z)\,
  \frac{e^{\,i k_{rm} r}}{\sqrt{k_{rm}}},
$$

each mode weighted by its excitation at the source depth $\Psi_m(z_\mathrm{s})$ and
its amplitude at the receiver depth $\Psi_m(z)$, and the coherent
propagation loss follows as $PL = -20 \log_{10}\,\lvert p(r,z)/p_0(1\,\mathrm{m})
\rvert$ (Eq. 5.15). `normal_modes` discretises the depth equation by finite
differences (a symmetric tridiagonal eigenproblem) on a grid refined enough
to keep the near-cutoff eigenvalues honest, and warns when a retained mode
sits too close to its discretisation error band.

The mode count is the physics: an isovelocity channel of depth $D$ carries
$M \approx kD/\pi$ propagating modes, so low frequency and shallow water
mean few modes and a compact, essentially exact description. The ideal
pressure-release waveguide is the validation oracle, with
$k_{rm} = \sqrt{k^2 - (m\pi/D)^2}$ in closed form:

```python
import numpy as np
from phonometry import underwater

# A 200 m isovelocity channel at 50 Hz: kD/pi = 2 f D / c = 13.3.
modes = underwater.normal_modes(50.0, [0.0, 200.0], [1500.0, 1500.0],
                                source_depth=50.0, receiver_depth=100.0)
print(modes.wavenumbers.size)                   # 13  propagating modes
k = 2 * np.pi * 50.0 / 1500.0
print(round(float(modes.wavenumbers[0]), 5))    # 0.20885  computed kr1
print(round(np.sqrt(k**2 - (np.pi / 200.0) ** 2), 5))   # 0.20885  exact
```

## 3. Ray tracing: turning points and travel times

In the high-frequency limit the Helmholtz equation collapses to the eikonal
equation, and its characteristics are **rays**: trajectories integrated from
the first-order system (Jensen Eqs. 3.23-3.24)

$$
\frac{dr}{ds} = c\,\xi, \quad \frac{dz}{ds} = c\,\zeta,
\qquad
\frac{d\zeta}{ds} = -\frac{1}{c^2}\,\frac{\partial c}{\partial z},
$$

with $s$ the arc length and $(\xi, \zeta)$ the ray slowness. In a
range-independent profile the horizontal slowness is conserved along each
ray, which is **Snell's law** in continuous form,
$\cos\theta(z)/c(z) = \cos\theta_0/c(z_0)$: a ray bends toward lower sound
speed, flattens as $c$ grows, and turns where $c(z_\mathrm{t}) =
c(z_\mathrm{s})/\cos\theta_0$. In a linear gradient $c(z) = c_0 + gz$ the arcs are
exactly circular with radius $R = c_0/(g \cos\theta_0)$, the closed form
the solver is validated against; in a deep-water profile the family of rays
refocuses periodically into the **convergence zones** of the section 1
figure. `ray_trace` integrates all launch angles at once with a fixed-step
fourth-order Runge-Kutta scheme, reflecting at the surface and the bottom, and
carries the travel time along with them as a third state of the same step
($dt/dr = 1/(\xi c^2)$, with $\xi = \cos\theta_0/c(z_\mathrm{s})$ the Snell invariant):

```python
import numpy as np
from phonometry import underwater

# An isothermal deep layer: c rises 0.017 (m/s)/m with pressure, so a ray
# launched 6 degrees downward from 100 m turns back up where Snell gives
# c(z_t) = c(z_s)/cos(6 deg).
z = [0.0, 1000.0]
c = [1490.0, 1507.0]                        # linear gradient, g = 0.017 1/s
rays = underwater.ray_trace(z, c, source_depth=100.0,
                            launch_angles_deg=[6.0], max_range=40e3,
                            n_steps=20000)
z_turn = (1491.7 / np.cos(np.radians(6.0)) - 1490.0) / 0.017
print(round(z_turn, 1))                     # 583.3  analytic turning depth
print(round(float(rays.depths.max()), 1))   # 583.3  traced

# The arc also fixes the time to that turn: the sound speed cancels and only
# the launch angle and the gradient are left,
# t = (1/g) ln[(1 + sin theta_0) / cos theta_0].
th = np.radians(6.0)
r_turn = np.sin(th) / (np.cos(th) / 1491.7 * 0.017)   # 9222.6 m
t_turn = np.log((1.0 + np.sin(th)) / np.cos(th)) / 0.017
print(round(t_turn, 3))                                        # 6.171  analytic
traced = np.interp(r_turn, rays.ranges[0], rays.travel_times[0])
print(round(float(traced), 3))                                 # 6.171  traced
```

Rays buy geometry and timing: paths, turning depths, convergence-zone ranges
and the travel time along every one of them, at a cost independent of
frequency. Because the time rides the same four Runge-Kutta stages as the
trajectory, and takes its sound speed from the interpolation those stages
already do, it describes the path actually returned rather than a second
reading of it, and it matches the closed form for a constant gradient
(Medwin & Clay 1998, Eq. (3.3.20)) to about $10^{-14}$ s. What rays do not
carry here is a full amplitude: the geometric ray-tube intensity diverges at
caustics, so `RayTraceResult` reports no level at all. Section 4 is what fixes
that, on the same rays and through the same marcher.

## 4. Gaussian beams: a field where rays give up

Section 3 stops at geometry on purpose. The classical ray amplitude
(Jensen Eq. 3.65) divides by the ray-tube spreading $q$, and $q$ vanishes
wherever the family of rays folds over on itself. That fold is a **caustic**:
ray theory answers infinity there while the true field is merely loud, and a
ray that crosses one picks up a $-\pi/2$ that the whole interference pattern
beyond it depends on (Jensen §3.4.1, Figs. 3.13-3.14). Past the last ray of a
family lies a **shadow zone**, where ray theory returns not a small number but
no number at all.

**Gaussian beam tracing** removes both at once, by widening every ray into a
beam. The spreading obeys the *dynamic* ray equations (Jensen Eq. 3.58),

$$
\frac{dq}{ds} = c\,p, \qquad \frac{dp}{ds} = -\frac{c_{nn}}{c^2}\,q ,
$$

which the ray marcher integrates alongside the trajectory. Started from
**complex** initial conditions, $p(0) = 1$ and $q(0) = i\omega W_0^2/2$
(Eq. 3.91), each ray becomes the axis of a beam of initial half-width $W_0$
and flat wavefront, and the field of that beam is (Eq. 3.88)

$$
p^{\text{beam}}(s, n) = A\,\sqrt{\frac{c(s)}{r\,q(s)}}\;
  \exp\left\{-i\omega\left[\tau(s) + \frac{p(s)}{2\,q(s)}\,n^2\right]\right\},
$$

with $n$ the distance from the central ray and $\tau$ the travel time along
it. The total field is that expression summed over the launch fan with the
weights of Eq. (3.92).

**Why it stays finite.** Eq. (3.58) is linear with real coefficients, so the
real and imaginary parts of $(q, p)$ are two real solutions of it, and their
Wronskian $q_R p_I - q_I p_R$ is conserved. The impulses the spreading takes
at a profile kink and at a reflection are shears of unit determinant, so they
cannot change it either. It starts at $-\omega W_0^2/2$ and stays there, and
that single constant carries the whole method: $q$ can never reach zero, so
**there is no caustic singularity left to patch**, no KMAH index to count and
no minimum-width floor to impose; $\mathrm{Im}[p/q] < 0$ always, so the beam
always decays away from its axis; and the beam half-width of Eq. (3.89)
collapses to $W(s) = 2|q(s)|/(\omega W_0)$, a hyperbola in free space with its
waist at the source and the Rayleigh range $kW_0^2/2$ for its scale.

```python
import numpy as np
from phonometry import underwater

# Jensen's n^2-linear profile, c(z) = c0/sqrt(1 + 2.4 z/c0) (Eq. 3.77). With
# the source near the bottom the up-going rays turn, their envelope is a
# caustic, and above it the fan runs out into a shadow zone.
c0 = 1550.0
z_beam = np.linspace(0.0, 1000.0, 201)
c_beam = c0 / np.sqrt(1.0 + 2.4 * z_beam / c0)

beams = underwater.gaussian_beams(600.0, z_beam, c_beam, source_depth=992.5,
                                  max_range=2500.0, range_step=25.0,
                                  max_angle_deg=45.0, n_depth_points=80)
print(beams.propagation_loss.shape)            # (80, 101)  depth x range
print(beams.launch_angles.size)                # 439 beams over +-45 degrees
print(round(beams.initial_beam_width, 1))      # 35.9 m: W0 from Eq. (3.86)
beams.plot()   # the PL field, on the same frame as parabolic_equation
```

**The one free parameter.** $W_0$ is it, and the book is candid that "the
optimal choice of these initial conditions is a matter of current research",
recommending 10 to 50 wavelengths. The default here is sharper than a rule of
thumb because §3.5.1 does the optimisation explicitly: differentiating the
free-space width of Eq. (3.86) with respect to the complex offset gives
$W_0 = \sqrt{\lambda\,r_{\max}/\pi}$, the width that resolves the field best
at the far end of the run, where it is resolved worst. That is also where the
launch-angle integral behind Eq. (3.92) is a genuine Gaussian rather than a
Fresnel integral, which is why it is not merely a tidy choice: against the free
field at 100 Hz the error in $|p|$ is $7.5\times10^{-5}$ at that width and the
same at every range tried, growing to $5\times10^{-4}$ at a fifth of it and to
$0.36$ at fifteen times it. The book's 10-50 wavelength band and a quarter of
the water depth clamp it, and the channel clamp has the last word: a beam
comparable to the water depth breaks the bookkeeping that folds a reflected
ray back into the column, rather than merely costing accuracy.

**What it is validated against.** Free-field spherical spreading, to
$10^{-3}$ dB, which is the one comparison that pins the amplitude
normalisation and the phase convention together; the two-ray Lloyd-mirror field
with one surface reflection, to 0.01 dB; and the image-source sum of the ideal
pressure-release waveguide, to 0.0004 dB with the fan opened to 88 degrees.
Both boundary conditions come out of the beam sum rather than being imposed:
the field at a pressure-release surface or bottom is 3 parts in $10^5$ of its
mid-column value, and a rigid bottom doubles it.

**Where it stops.** Three limits, in the order they bite.

- **Ray theory's own regime** (Jensen §3.4.2): "the wavelength should be
  substantially smaller than any physical scale in the problem". This is the
  limit that bites hardest and that a plausible-looking answer hides best. At
  20 Hz in 100 m of water the depth is 1.3 wavelengths, two modes propagate,
  and the quarter-depth cap leaves a beam a third of a wavelength across:
  against the image-source sum from 200 m to 5 km the loss comes out 2 to 8 dB
  high, and it moves by decibels when the fan is opened or the beam count
  multiplied by 150, so there is nothing it is converging to. Read the beams
  above a few hundred hertz in shallow water, and use `normal_modes` below
  that, where two modes are the exact answer.
- **The fan is truncated** at `max_angle_deg`, and a waveguide with two
  perfectly reflecting boundaries is the worst case for that, because nothing
  but $1/R$ attenuates the steep multiple bounces. On the ideal 1000 m guide at
  300 Hz, against the image-source sum at 2, 5 and 10 km: 0.27, 4.06 and
  2.52 dB with the default 80 degrees, falling to 0.0002, 0.0003 and 0.0004 dB
  when the fan is opened to 88 degrees. Cutting the *oracle* to the same
  half-angle moves it by 0.25, 3.95 and 2.31 dB, so this is the fan and not the
  method. A real, lossy seabed
  absorbs those bounces and the default is then ample. Opening the fan means
  cutting `range_step` with it, since one step has to resolve
  $\tan\theta_{\max}$ depth units of climb per unit range; the solver warns
  when that pairing is wrong.
- **The far shadow is floored.** Each beam is summed out to four half-widths,
  140 dB below its own axis, so a receiver that no beam of the fan comes that
  close to gets exactly zero and an infinite loss. That is the unilluminated
  wedge outside the traced aperture, not the graded penumbra just past the
  limiting ray, which is where the interesting part of a shadow zone is and
  which the beams do resolve.

The other flavour, **geometric beams** (Jensen §3.3.5.5, Eqs. 3.72-3.76), is
not implemented. It keeps $q$ real and takes the width from the ray tube
itself, $W(s) = |q(s)\,\delta\theta_0|$, so the width vanishes at a caustic and
has to be propped up with the Weinberg-Keenan $\pi\lambda$ floor and the KMAH
index of Eq. (3.79). That is the patched approach this deliberately does not
take, and the book's own verdict is worth quoting anyway: geometric beams
"have generally proven to be more satisfactory" at low frequency, where the
physics makes the beam large compared to the channel.

## 5. The parabolic equation: a one-way field, marched in range

The parabolic equation trades the boundary-value Helmholtz problem for an
initial-value problem in range. Factor out the fast outgoing oscillation,
$p(r, z) \simeq \psi(r, z)\,e^{\,i k_0 r}/\sqrt{k_0 r}$ with a reference
wavenumber $k_0 = \omega/c_0$, and for energy travelling within a small
angle of the horizontal the envelope obeys the **standard (Tappert) PE**
(Jensen §6.2):

$$
2 i k_0 \frac{\partial \psi}{\partial r}
  + \frac{\partial^2 \psi}{\partial z^2}
  + k_0^2\left[n^2(z) - 1\right]\psi = 0,
\qquad n(z) = \frac{c_0}{c(z)} .
$$

The **split-step Fourier** algorithm marches it by operator splitting,
alternating two individually exact
half-physics steps: diffraction is a multiplication by
$e^{-i k_z^2 \Delta r / 2 k_0}$ in the vertical-wavenumber domain, and
refraction a phase screen $e^{\,i k_0 (n^2 - 1) \Delta r / 2}$ back in
depth, with one transform pair per range step. `parabolic_equation` starts
from a Gaussian field matched to the point source and uses a discrete sine
transform in depth, which enforces the pressure-release surface and bottom
by construction. The price is the **paraxial** approximation: the standard
PE is accurate within roughly ±15-20° of the horizontal, and steeper energy
carries a phase error that shows at short range in shallow waveguides
(Jensen §6.2). The free-field calibration is the oracle: with no gradient
at all, the marched field must reproduce spherical spreading,

```python
import numpy as np
from phonometry import underwater

# Free field: in a 5000 m isovelocity column, before any boundary is felt,
# the PE must reproduce spherical spreading, PL = 20 lg R.
field = underwater.parabolic_equation(50.0, [0.0, 5000.0], [1500.0, 1500.0],
                                      source_depth=2500.0, max_range=2000.0,
                                      range_step=10.0)
iz = np.argmin(np.abs(field.depths - 2500.0))
ir = np.argmin(np.abs(field.ranges - 1000.0))
print(round(float(field.propagation_loss[iz, ir]), 2))   # 60.0 = 20 lg 1000
```

and it does so to about $10^{-4}$ dB at the default range step.

## 6. Choosing a model

Every propagation function of the `underwater` module answers the same
question, "how much level survives the path", at a different price in
physics. Terminology throughout follows ISO 18405:2017 (propagation loss,
source level, levels re 1 µPa). The sound-speed and absorption models named
below are implemented and referenced in
[Underwater sound propagation](underwater-propagation.md).

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

**Spreading law.** Spherical spreading ($20\log_{10} R$) describes a wavefront that
expands freely in three dimensions, before any boundary confines it;
cylindrical spreading ($10\log_{10} R$) describes energy trapped between the surface
and the bottom (or in the SOFAR channel) that can only expand in range. The
`"practical"` law splices the two at a transition range $R_0$, which is
physically of the order of the water (or channel) depth: spherical while the
wavefront has not yet filled the duct, cylindrical once it has. In the 10 kHz
example of the
[propagation-loss section](underwater-propagation.md) the choice is not
cosmetic: against the same figure of merit of 87 dB, spherical-only spreading
predicts detection out to about 8.7 km while the practical law with
$R_0 = 1000$ m stretches it to about 15.8 km. When the spreading law is the
biggest uncertainty in the budget, that is the cue to stop using a closed
form and compute the field.

**Closed form or solver.** The closed-form propagation loss knows nothing of
the sound-speed profile, the seabed or the surface; it is honest for short,
direct, boundary-free paths and for first-cut sonar budgets. When refraction
and boundaries decide the answer, pick the solver by frequency and geometry
(Jensen et al. 2011, Ch. 1):

| Solver | Natural regime | What it buys you |
|---|---|---|
| `ray_trace` | High frequency (water depth ≫ λ), deep water | Ray-path geometry, turning depths, travel times, convergence zones; cost independent of frequency, and no amplitude |
| `gaussian_beams` | High frequency, wherever a *level* is wanted from rays | The same geometry turned into PL($z$,$r$): finite at caustics, graded into shadow zones, and like the rays it is built on, its cost does not grow with frequency |
| `normal_modes` | Low frequency, shallow water, range-independent | Finite-difference modal sum with few propagating modes ($m < kD/\pi$); the reference solution for its regime, validated against the ideal waveguide's exact modes |
| `parabolic_equation` | Low frequency, long one-way paths | Full-field PL($z$,$r$) with refraction, marched in range over the range-independent $c(z)$ all four solvers assume |

The boundaries blur in practice: rays remain usable at surprisingly low
frequencies for travel-time work, and the PE remains the workhorse well above
its formal small-angle regime. When two of the four agree on a case, as the
modes and the PE do in the section 1 figure, that agreement is the practical
convergence test. The two extremes divide the labour cleanly: `normal_modes`
and `parabolic_equation` both get more expensive as the frequency rises, since
both have to resolve the wavelength on a grid, while the ray core does not,
so `gaussian_beams` is the one that stays affordable exactly where the others
stop being so, and it is also the one whose approximation is best justified
there.

**A worked sonar budget.** Chain the pieces end to end: a 140 dB re
1 µPa²/Hz source at 10 kHz, a 60 dB ambient spectrum level, a 15 dB array
gain and an 8 dB detection threshold give the figure of merit
$FOM = 140 - (60 - 15) - 8 = 87$ dB computed by `passive_sonar_equation` in
the [sonar-equation section](underwater-propagation.md) of the propagation
guide. The propagation-loss curve of that guide's first section (10 °C,
35 ppt, 100 m, $\alpha = 0.95$ dB/km) crosses 87 dB at about 15.8 km with the
practical law: that crossing *is* the predicted detection range, and every
term of the budget moves it. Trim the directivity index to 7.5 dB and the
figure of merit falls to 79.5 dB, so the range drops to wherever the PL
curve crosses that value; double the frequency to 20 kHz and $\alpha$ more
than triples to 3.3 dB/km, pulling the crossing sharply inward. This
coupling between the absorption model, the spreading law and the sonar
equation is why they all live in one module.

## Quick answers

### Which underwater propagation solver should I use?

Pick by frequency and geometry (Jensen et al. 2011, Ch. 1): rays for high
frequency and deep water (ray-path geometry, travel times and convergence
zones at a cost independent of frequency, and no level at all), Gaussian beams
when that same high-frequency geometry has to yield a level (the PL($z$, $r$)
field, finite at caustics and graded into shadow zones), normal modes for low
frequency in shallow water (few propagating modes, $m < kD/\pi$, the reference
solution for its regime) and the parabolic equation for low-frequency, long
one-way paths (the full PL($z$, $r$) field with refraction). When two solvers
agree on a case, that agreement is the practical convergence test.

### When is a closed-form propagation loss no longer enough?

When refraction or boundaries decide the answer: a sound-speed minimum that
traps energy (the SOFAR channel), surface and bottom reflections in shallow
water, or a detection range that swings with the choice of spreading law.
The closed form $PL = \text{spreading} + \alpha R$ is honest for short,
direct, boundary-free paths and first-cut sonar budgets; beyond that,
compute the field.

## See also

- [Underwater sound propagation](underwater-propagation.md): the closed
  forms these solvers replace when refraction and boundaries matter, and
  the sound-speed profiles they consume.
- [Underwater acoustics: radiated noise and pile driving](underwater-acoustics.md):
  the ISO 18405 reference levels in which every propagation loss here is
  expressed.
- [Atmospheric refraction: rays and the GFPE](../environment/propagation/atmospheric-refraction.md):
  the airborne siblings of these solvers, with the same ray bending and a
  Green's-function PE marched over ground impedance instead of a seabed.
- [2D FDTD wave simulation](../simulation/fdtd-simulation.md): the time-domain
  alternative behind the SOFAR ducting animation of the propagation guide.
- API reference: [`underwater.propagation.numerical`](https://jmrplens.github.io/phonometry/reference/api/underwater/numerical/).

## References

- Jensen, F. B., Kuperman, W. A., Porter, M. B., & Schmidt, H. (2011).
  *Computational ocean acoustics* (2nd ed.). Springer.
  [doi:10.1007/978-1-4419-8678-8](https://doi.org/10.1007/978-1-4419-8678-8).
  The reference monograph implemented here: the modal derivation of
  section 2 (Ch. 5, Eqs. 5.3-5.17), the ray equations of section 3
  (Ch. 3, Eqs. 3.23-3.24), the Gaussian beams of section 4
  (Ch. 3, §3.5, Eqs. 3.88-3.92), the split-step Fourier parabolic equation of
  section 5 (Ch. 6) and the model-selection guidance of section 6 (Ch. 1).
- Munk, W. H. (1974). Sound channel in an exponentially stratified ocean,
  with application to SOFAR. *The Journal of the Acoustical Society of
  America*, 55(2), 220-226.
  [doi:10.1121/1.1914492](https://doi.org/10.1121/1.1914492).
  The canonical deep-water sound-speed profile used by the section 1
  figure and snippets.
- ISO 18405:2017. *Underwater acoustics — Terminology*.
  [ISO page](https://www.iso.org/standard/62406.html).
  The standardized definitions (propagation loss, source level, sound
  pressure level re 1 µPa) behind the quantities of this page.

## Standards & sources

No measurement standard governs the solvers themselves: they are
implemented clean-room from Jensen, Kuperman, Porter & Schmidt,
*Computational Ocean Acoustics* (2nd ed., Springer 2011), for normal modes
(Ch. 5), ray tracing and Gaussian beams (Ch. 3, the latter §3.5) and the
split-step Fourier parabolic equation
(Ch. 6), with terminology per ISO 18405:2017. Validation is anchored to
closed forms: the ideal pressure-release waveguide's exact modes and its
image-source sum, the
circular-arc ray paths of a linear sound-speed gradient, free-field
spherical spreading for the PE and for the beams, and the mutual agreement of the modal and
PE propagation loss.
