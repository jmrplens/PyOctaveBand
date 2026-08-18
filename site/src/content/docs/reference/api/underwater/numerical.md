---
title: "underwater.propagation.numerical"
description: "Numerical models of underwater sound propagation (range-independent water, with an optionally sloping bottom for the ray-based solvers)."
sidebar:
  label: "numerical"
---

Numerical models of underwater sound propagation (range-independent water,
with an optionally sloping bottom for the ray-based solvers).

Four complementary numerical solvers for the acoustic field in a
horizontally-stratified ocean waveguide, complementing the closed-form
propagation loss of [`phonometry.underwater.propagation.closed_form`](/phonometry/reference/api/underwater/closed-form/):

* [`normal_modes`](/phonometry/reference/api/underwater/numerical/#normal_modes) -- the normal-mode expansion. Solves the depth-separated
  Sturm-Liouville eigenvalue problem by finite differences and assembles the
  propagation loss from the propagating modes.
* [`ray_trace`](/phonometry/reference/api/underwater/numerical/#ray_trace) -- ray tracing. Integrates the ray-trajectory equations
  through a sound-speed profile (Runge-Kutta), returning the ray paths and the
  travel time accumulated along each of them, and no amplitude.
* [`gaussian_beams`](/phonometry/reference/api/underwater/numerical/#gaussian_beams) -- Gaussian beam tracing. Hangs a beam on each of those
  rays and sums them into a propagation-loss field, which is finite at a caustic
  and decays smoothly into a shadow zone where ray theory has nothing to say.
* [`eigenrays`](/phonometry/reference/api/underwater/numerical/#eigenrays) -- the arrival structure. Takes a traced fan and a receiver
  and refines, by bisection on fresh traces, the rays that actually connect the
  source to that receiver, each with its travel time, launch and arrival
  angles, boundary-touch counts and classical complex amplitude: the list the
  sonar equation, a channel impulse response and communications work consume.
* [`parabolic_equation`](/phonometry/reference/api/underwater/numerical/#parabolic_equation) -- the standard (Tappert) parabolic equation, solved
  with the split-step Fourier algorithm, returning the propagation-loss field.

All four are implemented clean-room from Jensen, Kuperman, Porter & Schmidt,
*Computational Ocean Acoustics* (2nd ed., Springer 2011): the modal derivation
(Ch. 5, Eqs. 5.3-5.17), the ray equations (Ch. 3, Eqs. 3.23-3.24), the Gaussian
beams of Sect. 3.5 (Eqs. 3.88-3.92) and the split-step Fourier PE (Ch. 6). They
are validated against analytic oracles: the ideal (pressure-release) waveguide's
exact modes and its image-source sum, that same image sum over a lossy fluid
seabed with the Rayleigh coefficient of each image's own grazing angle raised
to its count of bottom touches (Jensen Eq. 2.138 with Eq. 3.126 at every
touch), the circular-arc ray paths of a linear
sound-speed gradient together with the closed-form travel time along them
(Medwin & Clay, *Fundamentals of Acoustical Oceanography*, Academic Press 1998,
Eq. (3.3.20)), free-field spherical spreading, mutual agreement of the PE
and normal-mode propagation loss for a range-independent waveguide, and, for
the sloping bottom the two ray-based solvers accept, the ideal wedge's exact
image fan (the folded geometry to eleven digits, the beam field in dB).

The three field solvers report the same quantity on the same terms, so their
propagation losses can be laid side by side: `normal_modes` on a range slice
at one receiver depth, `gaussian_beams` and `parabolic_equation` on a
(depth, range) grid. Which of them to reach for is a question of frequency and
of what is being asked; the guide's solver table sets it out.

Densities are in kg/m3, sound speeds in m/s, depths and ranges in metres,
frequencies in Hz. The water column has a pressure-release surface at z = 0.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## BeamFan

```python
BeamFan(
    max_angle_deg: float = 80.0,
    n_beams: int | None = None,
    beam_width: float | None = None,
)
```

The launch fan [`gaussian_beams`](/phonometry/reference/api/underwater/numerical/#gaussian_beams) hangs its beams on.

Aperture, density and initial width travel together because they are one
bargain: the overlap condition that sizes the default fan reads the
width, and the width's own default is per launch angle. Every field has
a default, so `BeamFan()` is the stock fan and one field is named to
move one lever.

**Attributes**

| Name | Description |
| :--- | :--- |
| `max_angle_deg` | Half-angle of the fan, in degrees from the horizontal. Beams are spread symmetrically over `[-max_angle_deg, +max_angle_deg]`. |
| `n_beams` | Number of beams in the fan. Default (`None`): from the overlap condition. Adjacent beams are $s\,\delta\theta_0$ apart at arc length $s$ while each has spread to $W \to s\lambda/(\pi W_0)$, so the condition that they still overlap, $\delta\theta_0 \lesssim \lambda/(\pi W_0)$, is range-independent; the default takes four times that margin. Too coarse a fan shows as a periodic ripple in range at the beam spacing, which is easy to mistake for physical interference. |
| `beam_width` | The $W_0$ of Eq. (3.91), in metres: the beam's initial half-width, at the $e^{-2}$ folding distance in intensity, applied to every beam of the fan when passed. Default (`None`): one width per launch angle, the free-space optimum of each beam's own flight; see `_default_beam_widths`. |

## EigenrayResult

```python
EigenrayResult(
    launch_angles: NDArray[np.float64],
    arrival_angles: NDArray[np.float64],
    travel_times: NDArray[np.float64],
    amplitudes: NDArray[np.complex128],
    surface_reflections: NDArray[np.int_],
    bottom_reflections: NDArray[np.int_],
    caustic_crossings: NDArray[np.int_],
    receiver_range: float,
    receiver_depth: float,
    source_depth: float,
    water_depth: float,
)
```

The eigenrays connecting one source to one receiver, earliest first.

Every per-arrival array has one entry per eigenray, sorted by travel time.
The frequency-independent pieces of each arrival are recorded separately
(delay, complex amplitude, angles, boundary counts) so that one search
serves every frequency: in the module's $e^{-i\omega t}$ convention
the pressure a tone of angular frequency $\omega$ produces at the
receiver is

$$
p(\omega) = \sum_j a_j\, e^{i \omega \tau_j},
$$

with $a_j$ the `amplitudes` and $\tau_j$ the
`travel_times`; the band-limited channel impulse response is the inverse
transform of that sum, a spike of complex weight $a_j$ at each
$\tau_j$.

**Attributes**

| Name | Description |
| :--- | :--- |
| `launch_angles` | Launch angle of each eigenray at the source, from the horizontal, in degrees, positive downward: the same convention the fan was launched with. |
| `arrival_angles` | The angle each eigenray crosses the receiver with, same convention. In a range-independent medium its magnitude is fixed by Snell's invariant at the receiver depth; its sign says whether the arrival is descending or climbing, which is what a vertical array steers on. |
| `travel_times` | Travel time of each eigenray, in seconds: the marcher's third Runge-Kutta state read at the receiver, not a quadrature over the finished path. |
| `amplitudes` | Complex amplitude of each arrival, dimensionless, normalised to unit pressure at 1 m from the source (the reference of Jensen Eqs. (3.67)-(3.68), the same one every field solver of this module reports its loss against). The magnitude is the classical ray amplitude of Eq. (3.65), $\vert c(z_\mathrm{R})\cos\theta_0 / (c(z_\mathrm{S})\, r\, q(r))\vert ^{1/2}$ with $q$ integrated from the point-source initial conditions of Eq. (3.63); the phase is the caustic factor $(-i)^m$ of Eq. (3.79) times the boundary factors $(-1)^{n_\mathrm{s}}$ and $\mathcal{R}^{n_b}$ (Eqs. 3.125-3.126). See [`eigenrays`](/phonometry/reference/api/underwater/numerical/#eigenrays) for why that convention and not another. |
| `surface_reflections` | Sea-surface touches of each eigenray. |
| `bottom_reflections` | Seabed touches of each eigenray. The pair classifies the arrivals the way Fig. 3.7 colours them: refracted paths carry zeros, and every multipath family is named by its counts. |
| `caustic_crossings` | The KMAH index $m$ of Eq. (3.79): how many times each eigenray's ray-tube spreading vanished on the way, each crossing turning the amplitude by $-\pi/2$. Zero for every path in an isovelocity channel, where straight rays cannot form caustics. |
| `receiver_range` | Range of the receiver the list connects to, in m. |
| `receiver_depth` | Its depth, in metres. |
| `source_depth` | Source depth, in metres. |
| `water_depth` | Water-column depth, in metres. |

### EigenrayResult.plot()

```python
EigenrayResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the arrival structure (per-path loss stems against delay).

## eigenrays

```python
eigenrays(
    trace: RayTraceResult,
    *,
    receiver_range: float,
    receiver_depth: float,
    bottom: str | FluidSeabed = 'pressure-release',
    max_arrivals: int = 64,
    n_steps: int | None = None,
) -> EigenrayResult
```

The eigenrays a traced fan brackets between a source and a receiver.

Finding an eigenray is finding a launch angle whose ray's depth at the
receiver range equals the receiver depth: a root of
$f(\theta_0) = z(r_\mathrm{R}; \theta_0) - z_\mathrm{R}$, which is
continuous in $\theta_0$ because a specular reflection folds the
trajectory continuously. The fan of `trace` supplies the brackets, one
per adjacent pair of rays whose $f$ changes sign, and each bracket
is then closed by bisection on *fresh marches through the same profile*,
never by interpolating between the traced rays: interpolation across a
fan is exactly the hazard Jensen Sect. 3.7.5.1 illustrates (two rays of
one bracket taking different bounce histories have no path between them),
and a root polished on real traces is a real ray, whose travel time,
bounce counts and amplitude are its own rather than a blend's. The
marcher that traces the fan is the marcher that closes the brackets, with
the dynamic pair of Eq. (3.58) riding along under the real point-source
initial conditions of Eq. (3.63), so every arrival's amplitude is the
Jacobian of the very trajectory that hit the receiver.

**The amplitude convention, stated once.** Each arrival's complex
amplitude is

$$
a_j = \left| \frac{c(z_\mathrm{R})\,\cos\theta_0} {c(z_\mathrm{S})\, r_\mathrm{R}\, q_j} \right|^{1/2} (-i)^{m_j}\, (-1)^{n_{\mathrm{s},j}}\, \mathcal{R}^{n_{b,j}},
$$

in the module's $e^{-i\omega t}$ convention, normalised to unit
pressure at 1 m. Why each factor:

* The magnitude is Eq. (3.65) with the $1/(4\pi)$ cancelled against
  the free-field reference of Eqs. (3.67)-(3.68), which is how the book
  itself defines transmission loss from these amplitudes and how every
  solver of this module already normalises: the coherent sum
  $\sum_j a_j e^{i\omega\tau_j}$ over a complete arrival set is
  directly comparable to [`GaussianBeamResult.pressure`](/phonometry/reference/api/underwater/numerical/#gaussianbeamresult), and
  $-20\lg|{\sum}|$ to every propagation loss here.
* $(-i)^m$ is Eq. (3.79) exactly as printed. Sect. 3.3 writes the
  ray field as $A\,e^{i\omega\tau}$ (Eq. 3.57), which *is* the
  $e^{-i\omega t}$ convention, so the printed factor transfers
  unchanged; it is the same $-\pi/2$ per caustic the beam solver's
  tracked square-root branch spends continuously, taken here in the
  discrete form the classical amplitude needs, since with real initial
  conditions $q$ passes through zero instead of around it.
* $(-1)^{n_\mathrm{s}}$ and $\mathcal{R}^{n_b}$ are
  Eqs. (3.125)-(3.126) applied at every boundary touch, collapsed to
  powers because Snell's invariant fixes one crossing angle per ray at a
  flat boundary. With a [`FluidSeabed`](/phonometry/reference/api/underwater/numerical/#fluidseabed) as the `bottom`,
  $\mathcal{R}$ is the Rayleigh coefficient of
  [`reflection_coefficient`](/phonometry/reference/api/underwater/seabed-reflection/#reflection_coefficient)
  at that angle, **not conjugated**: that function returns the
  coefficient in the $e^{-i\omega t}$ convention these amplitudes
  are declared in, so it enters as printed. ([`gaussian_beams`](/phonometry/reference/api/underwater/numerical/#gaussian_beams)
  conjugates the very same coefficient because its internal sum is
  assembled in the conjugate convention and conjugated once at the end;
  neither solver's choice transfers to the other, which is why both
  spell it out.)

A receiver standing exactly on a caustic is the one place the list is
honest rather than useful: $q_j \to 0$ there and the classical
amplitude of Eq. (3.65) diverges, as Sect. 3.4.1 says it must. The
infinity is ray theory's own, not an artefact to clamp;
[`gaussian_beams`](/phonometry/reference/api/underwater/numerical/#gaussian_beams) is the solver whose field stays finite there.

**What the fan resolves is what the search can find.** A bracket exists
where $f$ changes sign between adjacent fan rays, so a pair of
eigenrays standing between the same two rays (the two sides of a fold
near a caustic do this first) merges into no bracket at all, and an
eigenray steeper than the fan's aperture does not exist to it. The fan's
density and half-angle are the completeness levers, and they belong to
the caller's [`ray_trace`](/phonometry/reference/api/underwater/numerical/#ray_trace), where they are visible, rather than to a
hidden retrace here. Tangential contact (an $f$ that touches zero
without crossing) is likewise invisible, which is also why a receiver on
a boundary is rejected: a folded trajectory only ever grazes the
boundaries.

**The cap.** The multipath count grows without bound as paths steepen: in
an ideal waveguide nothing but spreading attenuates the high-order
bounces, and every extra $2D$ of unfolded depth is two more
arrivals. `max_arrivals` bounds what is returned: the earliest
`max_arrivals` arrivals are kept, because the early paths are the flat,
least-bounced, least-attenuated ones that carry the energy, and the tail
being discarded is the part every physical seabed drains fastest. A
[`PhonometryWarning`](/phonometry/reference/api/filters/phonometry/#phonometrywarning) says when the cap truncated; raise
it to keep everything the fan bracketed.

**Parameters**

| Name | Description |
| :--- | :--- |
| `trace` | A [`ray_trace`](/phonometry/reference/api/underwater/numerical/#ray_trace) result: the fan to bracket on, the profile to retrace through, and the source the rays leave from. |
| `receiver_range` | Receiver range, in metres, within the traced range. |
| `receiver_depth` | Receiver depth, in metres, strictly inside the water column. |
| `bottom` | `"pressure-release"` (default) or `"rigid"`, the perfect reflector whose coefficient ($-1$ or $+1$) each bottom touch multiplies into the amplitude; or a [`FluidSeabed`](/phonometry/reference/api/underwater/numerical/#fluidseabed), whose complex Rayleigh coefficient is charged instead. The sea surface is always pressure-release. The choice touches amplitudes only; the geometry, times and angles of the eigenrays are specular either way. |
| `max_arrivals` | Most arrivals to return (earliest kept, see above). |
| `n_steps` | Range samples per refinement march, receiver column included. Default (`None`): the step of `trace` itself carried over, so the search resolves what the fan resolved. |

**Returns:** An [`EigenrayResult`](/phonometry/reference/api/underwater/numerical/#eigenrayresult), possibly with zero arrivals: a receiver the traced fan never crosses (a shadow zone, or simply outside the aperture) has no eigenrays to list, which is an answer and not an error.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## FluidSeabed

```python
FluidSeabed(
    density: float,
    sound_speed: float,
    water_density: float = 1000.0,
)
```

A lossy fluid seabed, passed as the `bottom` of a solver.

The fluid half-space of
[`reflection_coefficient`](/phonometry/reference/api/underwater/seabed-reflection/#reflection_coefficient)
under the water column: handing one to [`eigenrays`](/phonometry/reference/api/underwater/numerical/#eigenrays) or
[`gaussian_beams`](/phonometry/reference/api/underwater/numerical/#gaussian_beams) as `bottom=` replaces the perfect reflector with
the Rayleigh interface, and every bottom touch of a path is charged the
complex coefficient at the one grazing angle Snell's invariant fixes for
it. The water's own sound speed at the interface comes from the
sound-speed profile, so it is not repeated here.

**Attributes**

| Name | Description |
| :--- | :--- |
| `density` | Sediment density $\rho_2$, in the same unit as `water_density` (kg/m³ by convention; only the ratio enters). |
| `sound_speed` | Sediment sound speed $c_2$, in m/s. A sediment faster than the water at the bottom has a critical grazing angle, below which the reflection is total in magnitude and lossy in phase alone. |
| `water_density` | Water density $\rho_1$ above the seabed (default 1000 kg/m³). It enters only through the impedance ratio, the fields themselves being density-normalised already. |

## gaussian_beams

```python
gaussian_beams(
    frequency_hz: float,
    depths: NDArray[np.float64] | list[float],
    sound_speeds: NDArray[np.float64] | list[float],
    *,
    source_depth: float,
    max_range: float = 10000.0,
    ranges_m: NDArray[np.float64] | list[float] | None = None,
    receiver_depths_m: NDArray[np.float64] | list[float] | None = None,
    n_depth_points: int = 200,
    fan: BeamFan | None = None,
    range_step: float = 25.0,
    bottom: str | FluidSeabed = 'pressure-release',
    absorption: str | VolumeAbsorption | None = None,
    bathymetry: tuple[NDArray[np.float64] | list[float], NDArray[np.float64] | list[float]] | None = None,
) -> GaussianBeamResult
```

Propagation-loss field from Gaussian beam tracing.

Hangs a Gaussian beam on each ray of a launch fan (Jensen Eq. 3.88) and sums
them over the fan with the weight of Eq. (3.92). The rays are the ones
[`ray_trace`](/phonometry/reference/api/underwater/numerical/#ray_trace) draws, integrated by the same marcher through the same
profile; what is added is the dynamic pair $(q, p)$ of Eq. (3.58),
started from the complex conditions of Eq. (3.91) that make each ray the
axis of a beam of initial half-width `fan.beam_width` and flat
wavefront.

The point of the beams is that the answer stays finite. Ray theory's
amplitude, Eq. (3.65), divides by the ray-tube spreading, which vanishes on
a caustic and gives an infinity there (Sect. 3.4.1) and nothing at all in a
shadow zone. Complex $q$ cannot vanish, so this field needs no KMAH
index and no minimum-width floor, is finite wherever a beam reaches, and
falls into a shadow zone gradually rather than off a cliff, which is what
the exact solution does (Figs. 3.11, 3.17). See
[`GaussianBeamResult`](/phonometry/reference/api/underwater/numerical/#gaussianbeamresult) for the one place it still reports an infinity,
which is the wedge no beam of the fan illuminates at all.

The limits are worth knowing before the numbers are believed.

* **Ray theory's own regime** (Sect. 3.4.2): "the wavelength should be
  substantially smaller than any physical scale in the problem". This is
  the limit that bites hardest and the one a plausible-looking answer
  hides best -- and one earlier version of this paragraph blamed it for an
  error that was really the beam width's. At 20 Hz in 100 m of water the
  depth is 1.3 wavelengths and two modes propagate; the quarter-depth cap
  this module used to put on $W_0$ left a beam a third of a
  wavelength across, and the loss came out decibels high against the
  image-source sum. With the cap retired the same guide (source 36 m,
  receiver 64 m, energy-averaged 0.2 to 5 km) measures -0.001 dB in the
  mean and 0.03 dB at worst, at the ten-wavelength floor's 750 m width.
  That clean bill is narrower than it looks: an isovelocity column over
  perfect reflectors is pure geometry, which the folded receiver images
  reproduce exactly at any frequency, so it says nothing about a channel
  the low-frequency field actually refracts through, and there
  [`normal_modes`](/phonometry/reference/api/underwater/numerical/#normal_modes) remains the solver to trust, exact in that regime
  for the cost of two modes.
* **There is no near field**, and this is the largest error the function
  makes. Eq. (3.92) weights the fan by matching it to a point source in the
  far field, and Eq. (3.88) divides by a cylindrical range that goes to zero
  on the axis every ray leaves from, so close in the sum has nothing to
  converge to. The cylindrical range is floored at one wavelength,
  which is what keeps the answer bounded rather than what makes it right.
  The scale it recovers on is the initial beam width, not a fixed distance.
  Worst error against $20\lg R$ over a +-500 m depth cut in an
  unbounded medium at 100 Hz, at three settings whose $W_0$ spans
  150 to 437 m: 17, 13 and 4.1 dB at a quarter of $W_0$, 1.2, 0.64 and
  0.36 dB at $W_0$, 0.012, 0.005 and 0.002 dB at 2.5 $W_0$, and
  a thousandth of a decibel or better from 3 $W_0$ out. Read nothing
  inside about three beam widths of the source; since the default's
  free-space term grows as $\sqrt{r_\mathrm{max}}$, a longer run
  pushes that boundary out rather than in. [`parabolic_equation`](/phonometry/reference/api/underwater/numerical/#parabolic_equation) is
  the solver to reach for close to the source.
* **The fan is truncated** at `fan.max_angle_deg`, and a waveguide with
  two
  perfectly reflecting boundaries is the worst case for that, because
  nothing but $1/R$ attenuates the steep multiple bounces. Measured on
  the ideal 1000 m guide at 300 Hz, source at 300 m and receiver at 600 m,
  against the image-source sum at 2, 5 and 10 km: a fan to 80 degrees is
  0.27, 4.06 and 2.52 dB out, a fan to 85 degrees 0.21, 1.32 and 1.91 dB,
  and a fan to 88 degrees 0.0002, 0.0003 and 0.0004 dB. Cutting the *oracle*
  to the same half-angle moves it by 0.25, 3.95 and 2.31 dB, so what is left
  at 80 degrees is the fan and not the method. A real seabed (the
  [`FluidSeabed`](/phonometry/reference/api/underwater/numerical/#fluidseabed) below) absorbs those
  bounces and the default is then ample; a perfect reflector needs the
  fan opened and `range_step` cut with it, since a step has to resolve
  $\tan\theta_\mathrm{max}$ depth units of climb per unit range. The
  warning below says when that pairing is wrong.
* **A shallow channel sets its own width**, and the default now pays it
  per launch angle rather than clamping against it. An earlier version of
  this module capped $W_0$ at a quarter of the water depth, reading
  Sect. 3.5's caution that a beam "large compared to the channel ...
  causes a variety of problems" as a ceiling; measured against the
  closed-form Airy modes of an $n^2$-linear 200 m guide at 200 Hz
  (source 30.5 m, receiver 120.5 m, energy-averaged over 0.5 to 4 km),
  that cap's 50 m width came out +3.08 dB in the mean and +5.86 dB at
  worst, systematically too quiet, while the per-angle default measures
  +0.19 dB on the same cut. What a shallow guide actually demands is the
  opposite bound, a beam wide enough to resolve the channel's modes in
  launch angle, and `_default_beam_widths` says why that is
  $W_0 \ge 4D\cos\theta_0/\pi$ and what the folded receiver images
  do to make the width affordable. The same profile in 1000 m of water,
  where the modal criterion is out of the band's reach and the free-space
  optimum stands, comes out at +0.72 dB with a 1.37 dB worst bin, closer
  to the exact field than [`normal_modes`](/phonometry/reference/api/underwater/numerical/#normal_modes) on the same cut. An
  explicit `fan.beam_width` is taken as given, whatever its size: the
  old
  quarter-depth warning went with the cap, since the measurements put the
  fault on the cap's side.

**Seawater absorption is off by default** and the field is then optimistic
beyond a few kilometres at sonar frequencies, exactly as ray theory without
a volume loss must be. Passing `absorption` multiplies each beam by
$e^{-\alpha s}$ with $s$ the **arc length along its central
ray**, which is Sect. 3.6.2 done as printed: perturbing the eikonal with
the complex sound speed a volume loss implies leaves the real rays standing
and attaches $e^{-\int_0^s \alpha(s')\,ds'}$ to each (Eq. 3.116),
an integral along the path flown, not along the range axis. The distinction
is not pedantry. The same section notes that adding $\alpha r$ to the
loss "is used in many ray models", and that shortcut under-charges every
steep or multiply-reflected path by the obliquity of its climb: a path at
60 degrees is twice as long as the range it covers, and it is precisely the
steep multiples of a waveguide that absorption is supposed to be killing.
The marcher integrates $s$ with the very Runge-Kutta stages that
place the ray ($ds/dr = 1/(\xi c)$), so the length the loss is
charged over is the length of the geometry actually summed. The
coefficient itself comes from
[`seawater_absorption`](/phonometry/reference/api/underwater/closed-form/#seawater_absorption),
one $\alpha$ per run, evaluated at the source frequency and at the
source depth (the same point the reference sound speed $c_0$ is read
at); over a water column the coefficient's own depth terms move it by
around a percent per hundred metres, which is far inside the method's
error budget. The default stays off so the validation figures quoted
throughout, all measured without absorption, remain reproducible as
printed.

**The seabed is a perfect reflector by default**, for the same reason, and
real shallow-water propagation loss is dominated by what that default
leaves out: the seabed absorbs part of every bottom bounce. Passing
a [`FluidSeabed`](/phonometry/reference/api/underwater/numerical/#fluidseabed) as the `bottom` replaces the perfect
reflector with the lossy fluid half-space of
[`reflection_coefficient`](/phonometry/reference/api/underwater/seabed-reflection/#reflection_coefficient)
(the Rayleigh interface, `density` standing for the water above it and
the profile's own bottom sound speed for its `c1`). This is Sect. 3.6.3
done as printed: "most ray codes treat the bottom simply as a reflector",
and each boundary touch multiplies the ray amplitude by
$|\mathcal{R}(\theta)|$ and adds $\arg \mathcal{R}(\theta)$
to its phase (Eqs. 3.125-3.126), while the dynamic pair $(q, p)$
crosses the reflection exactly as before, the curvature term of
Eq. (3.122) vanishing at a flat bottom, so the beam keeps its width and
only its complex amplitude is docked. The phase is not a refinement to
skip: below the critical angle $|\mathcal{R}| = 1$, *only* the
phase distinguishes the lossy seabed from a perfect one, and it moves the
interference fringes of every bottom-interacting path. The grazing angle
each beam is charged at is the one Snell's invariant fixes,
$\cos\theta = \xi\,c(z)$ along the whole ray, so at a flat seabed a
given beam arrives at one and the same angle at every touch whatever the
profile above did in between: its coefficient is evaluated once, exactly,
and raised to the marcher's count of bottom touches, in the running
product and in the receiver-image ladder alike. The book is candid that a
plane-wave coefficient applied to a field that is not a plane wave is an
approximation (p. 189), and it is the approximation the whole method
already breathes; sediment attenuation and elasticity are outside the
fluid-fluid model here as they are outside `seabed_reflection` itself.

**The bottom may slope.** The `bathymetry` pair replaces the level
bottom with the same
piecewise-linear `depth(r)` polyline [`ray_trace`](/phonometry/reference/api/underwater/numerical/#ray_trace) takes, and it is
the first range dependence in this module; the sound-speed profile stays
range independent, deliberately -- see [`ray_trace`](/phonometry/reference/api/underwater/numerical/#ray_trace)'s scope note for
why full $c(r, z)$ is excluded (no exact oracle exists to hold it
to). Four things follow from the slope, each stated with its cost:

* Every beam's central ray reflects specularly off the local facet
  (Eq. 3.121), so an upslope bounce steepens it by twice the slope, and
  the dynamic pair takes the reflection impulse of Eqs. (3.122)-(3.123)
  evaluated on that facet (curvature zero, since the facets are straight;
  `phonometry._internal.rays` records the closed form and its
  flat-bottom limit).
* A beam steepened past the vertical would run backward in range, which a
  range-marching solver cannot carry: it is terminated at that bounce and
  its weight is zero from there on, so the field keeps only what still
  travels forward, exactly as the one-way parabolic equation keeps no
  backscatter. Upslope propagation toward an apex therefore *loses* the
  energy the wedge sends back down the slope; the ideal-wedge oracle in
  the tests prices that truncation next to everything else.
* The receiver-image ladder folds each receiver column about its own
  local facet: the vertical stack of mirrors becomes the dihedral fan
  about the local apex, exact for a single facet and local to each
  column's facet on a general polyline; at slope zero it is the level
  ladder bit for bit, which a test pins. The fan is what lets the
  *tails* keep representing paths the marched axes cannot: an arrival
  that went up the slope, turned past the vertical and came back is a
  wrapped rung of the fan, analytic rather than marched, and the solver
  recovers most of it.
* The lossy fluid seabed cannot be combined with a slope, and the
  rejection is of the wiring, not the physics: the one-coefficient-per-
  beam collapse rests on Snell's invariant fixing a single grazing angle
  per ray at the bottom, and a slope rotates that invariant at every
  touch. A sloping run takes the perfect reflectors of `bottom`.

Validated against the ideal wedge, which has an exact solution by images:
an isovelocity wedge under a pressure-release surface with a rigid sloping
bottom of angle $\beta = \pi/n$ ($n$ even) unfolds into a
closed fan of $2\pi/\beta$ image sources on a circle about the apex.
The tests build that fan from pure geometry and quantify the agreement in
dB, cross-sections in range and depth, upslope: a tenth of a decibel
where a single facet bounce carries the field, and within about two
decibels of the *complete* wedge field (mean two thirds of one) across a
thin 2.8-degree wedge whose every cell is dense multipath, most of it
arrivals near their own turning point, which is where a one-way marcher
pays its way; the test module's docstring records the measurements and
their stability under step, width and aperture.

What it costs is `fan.n_beams` times the size of the receiver grid,
and none
of the three factors depends on the frequency: the ray core does not have to
resolve a wavelength on a grid, and the fan only widens as
$\lambda/W_0$, which the default width holds nearly fixed. On a
5000 m Munk column at 100 Hz over 10 km, everything left at its default
(512 beams, a 200 by 401 field), this takes 14 s against 0.1 s for
[`parabolic_equation`](/phonometry/reference/api/underwater/numerical/#parabolic_equation) and 177 s for [`normal_modes`](/phonometry/reference/api/underwater/numerical/#normal_modes); raise the
frequency and the first number stays where it is while the other two climb.
Shrinking `n_depth_points` or handing in a coarser `ranges_m` is the
direct way to trade resolution for time.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency_hz` | Source frequency, in Hz. |
| `depths` | Depth samples of the profile, in metres, from `z = 0`. |
| `sound_speeds` | Sound speed at each depth, in m/s. |
| `source_depth` | Source depth, in metres, inside the water column. |
| `max_range` | Maximum range to march to, in metres. |
| `ranges_m` | Ranges at which to evaluate the field, in metres. Default (`None`): the marching grid itself, which puts every receiver on a column the rays were actually sampled at. |
| `receiver_depths_m` | Depths at which to evaluate the field, in metres. Default (`None`): `n_depth_points` points spread over the water column, on the interior grid [`parabolic_equation`](/phonometry/reference/api/underwater/numerical/#parabolic_equation) uses, so the two fields land on the same depths. |
| `n_depth_points` | Size of that default depth grid. |
| `fan` | The launch fan, as a [`BeamFan`](/phonometry/reference/api/underwater/numerical/#beamfan): its half-angle `max_angle_deg`, its density `n_beams` and its initial width `beam_width`, each with the default that class documents. Default (`None`): the stock `BeamFan()`, the fan every published validation number of this module was measured with. |
| `range_step` | Marching step in range, in metres, and the spacing of the default `ranges_m`. |
| `bottom` | `"pressure-release"` (default) or `"rigid"`, perfect reflectors; or a [`FluidSeabed`](/phonometry/reference/api/underwater/numerical/#fluidseabed), the lossy fluid half-space whose complex Rayleigh coefficient every bottom touch is charged. The sea surface is always pressure-release, and the perfect default is what every published validation number of this module returns. |
| `absorption` | Seawater volume absorption applied along each beam's central ray: the bare model name (`"francois-garrison"`, `"ainslie-mccolm"` or `"thorp"`) for the standard water of [`VolumeAbsorption`](/phonometry/reference/api/underwater/numerical/#volumeabsorption)'s defaults, or a [`VolumeAbsorption`](/phonometry/reference/api/underwater/numerical/#volumeabsorption) naming its own temperature, salinity and pH. Default (`None`): no volume absorption, so the published validation numbers of this module are what the solver returns. |
| `bathymetry` | A `(ranges_m, depths_m)` pair of arrays describing a piecewise-linear bottom profile, the same pair [`ray_trace`](/phonometry/reference/api/underwater/numerical/#ray_trace) takes: node ranges in metres, strictly increasing from `r = 0` and level past the last node, with the bottom depth at each node in metres, strictly positive and never below the sound-speed profile's last depth. Not alongside a [`FluidSeabed`](/phonometry/reference/api/underwater/numerical/#fluidseabed). Default (`None`): the level bottom at the profile's last depth, which is every validation number of this module. |

**Returns:** A [`GaussianBeamResult`](/phonometry/reference/api/underwater/numerical/#gaussianbeamresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

**Warns**

| Warning | When |
| :--- | :--- |
| PhonometryWarning | when the source sits on a kink of the profile (Sect. 3.7.4's spurious horizontal jet), and when one marching step carries the steepest beam of the fan across more than a quarter of the water column, which is the pairing between `fan.max_angle_deg` and `range_step` that is easiest to get wrong. |

## GaussianBeamResult

```python
GaussianBeamResult(
    frequency: float,
    ranges: NDArray[np.float64],
    depths: NDArray[np.float64],
    propagation_loss: NDArray[np.float64],
    pressure: NDArray[np.complex128],
    launch_angles: NDArray[np.float64],
    ray_ranges: NDArray[np.float64],
    ray_depths: NDArray[np.float64],
    beam_widths: NDArray[np.float64],
    wavefront_curvatures: NDArray[np.float64],
    initial_beam_widths: NDArray[np.float64],
    absorption_model: str | None,
    absorption_coefficient: float,
    seabed_density: float | None,
    seabed_sound_speed: float | None,
    source_depth: float,
    water_depth: float,
    bathymetry_ranges: NDArray[np.float64] | None = None,
    bathymetry_depths: NDArray[np.float64] | None = None,
)
```

Gaussian beam solution of a range-independent waveguide.

The propagation-loss field is on the same footing as
[`ParabolicEquationResult`](/phonometry/reference/api/underwater/numerical/#parabolicequationresult)'s: same shape, same reference, so the two
can be subtracted.

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequency` | Source frequency, in Hz. |
| `ranges` | Range grid of the field, in metres. |
| `depths` | Depth grid of the field, in metres. |
| `propagation_loss` | Propagation-loss field `PL(z, r)`, in dB, shape `(n_depths, n_ranges)`. Infinite where the field is exactly zero, which happens in the wedge no beam of the fan reaches: each beam is summed out to four half-widths, 140 dB below its own axis, so a point that far from every one of them is outside the traced aperture rather than merely in shadow. The graded penumbra just past a limiting ray, which is the part of a shadow zone worth having, is finite and carries the beams' tails. Many ordinary cases have no infinity at all: an isovelocity 1000 m guide at 300 Hz over 10 km, everything default, has none in 80200 cells. The source column is **not** one of the infinities, and is not to be read. [`parabolic_equation`](/phonometry/reference/api/underwater/numerical/#parabolic_equation) divides by $\sqrt{r}$ and so genuinely diverges at `r = 0`; the beam sum does not, and hands back a finite number there instead, 13.6 dB in the case above. It means nothing, and neither does anything else within about three initial beam widths of the source: see [`gaussian_beams`](/phonometry/reference/api/underwater/numerical/#gaussian_beams) on why this method has no near field. The plausible size of these numbers is the point worth knowing about them. |
| `pressure` | The complex field the loss was taken from, same shape, in the module's own $e^{-i\omega t}$ convention (the conjugate of the one Jensen Eq. (3.88) is printed in) and normalised to unit pressure at 1 m, so `propagation_loss = -20 lg\|pressure\|`. |
| `launch_angles` | Launch angle of each beam's central ray, from the horizontal, in degrees. |
| `ray_ranges` | Range of each central ray at each marching step, in metres, shape `(n_beams, n_steps)`. This is the marching grid, which is finer than (and independent of) `ranges`. |
| `ray_depths` | Depth of each central ray on that grid, in metres. |
| `beam_widths` | Beam half-width $W(s)$ on that grid, in metres: Jensen Eq. (3.89), the distance at which the beam's own pressure has fallen by $e^{-1}$ and its intensity by $e^{-2}$. |
| `wavefront_curvatures` | Beam wavefront curvature $K(s)$ on that grid, in 1/m: Jensen Eq. (3.90) with the sign that belongs to the conjugated field this result exposes, so that a beam spreading in free space reproduces Eq. (3.85), $K = x/(x^2 + a^2)$, as a positive number. |
| `initial_beam_widths` | The $W_0$ of Eq. (3.91) actually used by each beam of the fan, in metres, shape `(n_beams,)`. An explicit `fan.beam_width` fills it with one value; the default is per launch angle (see `_default_beam_widths`), widest on the axis of the fan whenever a shallow channel's modal-resolution term is in play and flat across it otherwise. |
| `absorption_model` | The seawater absorption model applied along the beams, or `None` when the run propagated without volume absorption (the default). |
| `absorption_coefficient` | The absorption coefficient $\alpha$ actually applied, in dB/km (0.0 when `absorption_model` is `None`), as [`seawater_absorption`](/phonometry/reference/api/underwater/closed-form/#seawater_absorption) evaluated it at the source frequency and depth. Recorded so a run's loss can be decomposed without re-deriving what was subtracted. |
| `seabed_density` | Sediment density of the fluid seabed the bottom bounces were charged with, or `None` when the bottom was one of the perfect reflectors (the default). |
| `seabed_sound_speed` | Sediment sound speed of that seabed, in m/s, or `None` likewise. Together the pair names the Rayleigh interface of [`reflection_coefficient`](/phonometry/reference/api/underwater/seabed-reflection/#reflection_coefficient) each beam's bottom reflections multiplied it by. |
| `source_depth` | Source depth, in metres. |
| `water_depth` | Water-column depth, in metres: the sound-speed profile's last depth, which over a sloping bottom is the deepest water the medium description reaches while the column itself is the bathymetry below. |
| `bathymetry_ranges` | Node ranges of the bottom profile the run was marched over, in metres, or `None` for the level bottom (the default). When present, the per-beam histories (`ray_depths`, `beam_widths`, `wavefront_curvatures`) are `NaN` from the column at which a beam was terminated by a reflection past the vertical, the same convention [`RayTraceResult`](/phonometry/reference/api/underwater/numerical/#raytraceresult) uses and for the same reason: from there on the beam no longer exists in the forward field, and its weight in the sum is zero. |
| `bathymetry_depths` | Bottom depth at each of those nodes, in metres (`None` likewise). |

### GaussianBeamResult.plot()

```python
GaussianBeamResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the propagation-loss field (depth increasing downward).

## normal_modes

```python
normal_modes(
    frequency_hz: float,
    depths: NDArray[np.float64] | list[float],
    sound_speeds: NDArray[np.float64] | list[float],
    *,
    source_depth: float,
    receiver_depth: float,
    ranges_m: NDArray[np.float64] | list[float] | None = None,
    density: float = 1000.0,
    bottom: str = 'pressure-release',
    n_depth_points: int | None = None,
) -> NormalModeResult
```

Normal-mode propagation loss for a range-independent waveguide.

Solves the depth-separated Sturm-Liouville problem (Jensen Eq. 5.3) on a
uniform finite-difference grid, then assembles the coherent propagation
loss from the propagating modes (Eq. 5.17).

The finite-difference eigenvalues carry an $O(dz^2)$ error that
grows with the mode's vertical wavenumber, so near-cutoff modes need a
fine grid. Two guards apply: eigenvalues inside the scheme's error band
($k_r^2 \le \max(k^2)^2 \, dz^2 / 12$) are discarded as numerically
indistinguishable from cutoff, and a
[`PhonometryWarning`](/phonometry/reference/api/filters/phonometry/#phonometrywarning) is emitted when a
retained mode sits within ten times that band (increase `n_depth_points`
to resolve it).

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency_hz` | Source frequency, in Hz. |
| `depths` | Depth samples of the sound-speed profile, in metres, starting at the surface `z = 0` and strictly increasing to the bottom. |
| `sound_speeds` | Sound speed at each depth, in m/s. |
| `source_depth` | Source depth `zs`, in metres. |
| `receiver_depth` | Receiver depth for the propagation-loss slice, in m. |
| `ranges_m` | Ranges at which to evaluate the loss, in metres; defaults to 100 m to 10 km. |
| `density` | Water density (constant), in kg/m3. |
| `bottom` | `"pressure-release"` (default) or `"rigid"`. |
| `n_depth_points` | Number of finite-difference depth points. Default (`None`): derived from the physics as $\max(400, \operatorname{ceil}(60 D f / c_{\mathrm{min}}))$, which keeps the near-cutoff eigenvalue error small at any frequency/depth combination, capped at 20 000 points (very high $f D$ products exceed the cap; the near-cutoff warning then indicates whether the capped grid suffices, and an explicit `n_depth_points` overrides the cap). |

**Returns:** A [`NormalModeResult`](/phonometry/reference/api/underwater/numerical/#normalmoderesult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## NormalModeResult

```python
NormalModeResult(
    frequency: float,
    wavenumbers: NDArray[np.float64],
    mode_depths: NDArray[np.float64],
    mode_functions: NDArray[np.float64],
    ranges: NDArray[np.float64],
    propagation_loss: NDArray[np.float64],
    receiver_depth: float,
    source_depth: float,
)
```

Normal-mode solution of a range-independent waveguide.

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequency` | Source frequency, in Hz. |
| `wavenumbers` | Horizontal wavenumbers `krm` of the propagating modes, in rad/m (descending order). |
| `mode_depths` | Depth grid of the mode functions, in metres. |
| `mode_functions` | Orthonormalised mode shapes `Ψm(z)`, shape `(n_modes, n_depths)`. |
| `ranges` | Ranges at which the propagation loss is evaluated, in metres. |
| `propagation_loss` | Coherent propagation loss at `receiver_depth` per range, in dB. |
| `receiver_depth` | Receiver depth of the propagation-loss slice, in m. |
| `source_depth` | Source depth, in metres. |

### NormalModeResult.plot()

```python
NormalModeResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the propagation loss versus range (loss increasing downward).

## parabolic_equation

```python
parabolic_equation(
    frequency_hz: float,
    depths: NDArray[np.float64] | list[float],
    sound_speeds: NDArray[np.float64] | list[float],
    *,
    source_depth: float,
    max_range: float = 10000.0,
    range_step: float = 10.0,
    n_depth_points: int = 1024,
) -> ParabolicEquationResult
```

Propagation-loss field from the standard (Tappert) parabolic
equation.

Marches the split-step Fourier solution (Jensen Ch. 6) in range with a
discrete sine transform in depth, enforcing a pressure-release surface at
`z = 0` and bottom at `z = water_depth`. The envelope is related to
pressure by $p = \psi \, e^{i(k_0 r - \pi/4)} / \sqrt{r}$ and
$\mathrm{PL} = -20 \log_{10}(\lvert \psi \rvert / \sqrt{r})$
(Eqs. 6.70-6.71), using a Gaussian starter.

The standard PE is **paraxial**: it is accurate for propagation within
roughly ±15-20° of the horizontal (Jensen §6.2). Steep modes therefore
carry a phase error that shows at short and intermediate range in
shallow-waveguide problems (a few dB against the exact field below a few
water depths of range), converging at long range; the free-field
calibration itself is exact to ~1e-4 dB at the default `range_step`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency_hz` | Source frequency, in Hz. |
| `depths` | Depth samples of the profile, in metres, from `z = 0`. |
| `sound_speeds` | Sound speed at each depth, in m/s. |
| `source_depth` | Source depth, in metres. |
| `max_range` | Maximum range, in metres. |
| `range_step` | Range marching step $\Delta r$, in metres. |
| `n_depth_points` | Number of depth points (interior sine-transform grid). |

**Returns:** A [`ParabolicEquationResult`](/phonometry/reference/api/underwater/numerical/#parabolicequationresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## ParabolicEquationResult

```python
ParabolicEquationResult(
    frequency: float,
    ranges: NDArray[np.float64],
    depths: NDArray[np.float64],
    propagation_loss: NDArray[np.float64],
    source_depth: float,
)
```

Parabolic-equation propagation-loss field.

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequency` | Source frequency, in Hz. |
| `ranges` | Range grid, in metres. |
| `depths` | Depth grid, in metres. |
| `propagation_loss` | Propagation-loss field `PL(z, r)`, in dB, shape `(n_depths, n_ranges)`. |
| `source_depth` | Source depth, in metres. |

### ParabolicEquationResult.plot()

```python
ParabolicEquationResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the propagation-loss field (depth increasing downward).

## ray_trace

```python
ray_trace(
    depths: NDArray[np.float64] | list[float],
    sound_speeds: NDArray[np.float64] | list[float],
    *,
    source_depth: float,
    launch_angles_deg: NDArray[np.float64] | list[float],
    max_range: float = 10000.0,
    n_steps: int = 2000,
    bathymetry: tuple[NDArray[np.float64] | list[float], NDArray[np.float64] | list[float]] | None = None,
) -> RayTraceResult
```

Trace acoustic rays through a range-independent sound-speed profile.

Integrates the ray-trajectory equations (Jensen Eqs. 3.23-3.24) with a
fixed-step fourth-order Runge-Kutta scheme, reflecting at the pressure-release
surface (`z = 0`) and the bottom (`z = water_depth`).

**The bottom may slope.** Passing the `bathymetry` pair replaces the
level bottom with a
piecewise-linear depth profile `depth(r)`, the faceted boundary model of
Jensen Fig. 3.20, and the first range dependence in this module; the
sound-speed profile stays range independent (see the scope note below).
The marcher then finds each boundary crossing against the interpolated
polyline and reflects the ray specularly about the local facet
(Eq. 3.121), so a bounce off a slope of angle $\beta$ changes the
ray's inclination by $2\beta$: upslope bounces steepen a ray, which
is the whole one-line physics of wedge propagation, and downslope bounces
flatten it. Snell's invariant $\xi$ is therefore no longer a
constant of each ray but a constant *between* its bottom bounces, and one
consequence is drawn honestly rather than papered over: a ray steepened
past the vertical runs backward in range, which a marcher whose
independent variable is range cannot carry (the same one-way surgery the
parabolic equation performs), so such a ray is terminated at that bounce
and its samples are `NaN` from there on -- see
[`RayTraceResult.bathymetry_ranges`](/phonometry/reference/api/underwater/numerical/#raytraceresult). The polyline continues level
past its last node exactly as `numpy.interp` clamps the sound-speed
profile; a bathymetric feature narrower than one range step can hide
between two samples of the crossing search, so `n_steps` must resolve
the bathymetry as well as the rays.

**Scope, stated plainly.** Range dependence enters here through the
boundary alone: full $c(r, z)$ is *not* implemented, deliberately.
Every solver in this module ships with an exact published oracle, and the
sloping-bottom geometry has one (the ideal wedge unfolds into a closed
fan of images, which is what the tests hold it to), while a
range-dependent water column has none: there is no closed form to hold a
$c(r, z)$ marcher to, so it would ship on trust, and this module
does not ship on trust.

The travel time is a third state of that same Runge-Kutta step rather than a
quadrature run over the finished path: with the range-invariant Snell
parameter $\xi = \cos\theta_0 / c(z_\mathrm{s})$ it obeys
$dt/dr = 1/(\xi c^2)$, so it is integrated with the very stages that
place the ray and cannot drift from the geometry actually returned. The arc
length is a fourth state on the same footing, $ds/dr = 1/(\xi c)$,
because it is the measure volume absorption needs (see
[`RayTraceResult`](/phonometry/reference/api/underwater/numerical/#raytraceresult)) and reading it off the finished path would demote
it to first order. This is
the same ray core, and the same travel-time equation, as the atmospheric
[`atmospheric_ray_paths`](/phonometry/reference/api/environment/refraction/#atmospheric_ray_paths)
(which reflects at the ground instead of at the sea surface). Reflections
cost no time and no path, so both odometers stay continuous across them.
They are counted, though, per boundary: see [`RayTraceResult`](/phonometry/reference/api/underwater/numerical/#raytraceresult) on why
the two cumulative counts, with the crossing angle Snell's invariant fixes
per ray, are the entire per-bounce record a downstream amplitude needs.

**Parameters**

| Name | Description |
| :--- | :--- |
| `depths` | Depth samples of the profile, in metres, from `z = 0`. |
| `sound_speeds` | Sound speed at each depth, in m/s. |
| `source_depth` | Source depth, in metres. |
| `launch_angles_deg` | Launch angles from the horizontal, in degrees (positive downward). |
| `max_range` | Maximum horizontal range to trace, in metres. |
| `n_steps` | Number of integration steps per ray. |
| `bathymetry` | A `(ranges_m, depths_m)` pair of arrays describing a piecewise-linear bottom profile: node ranges in metres, strictly increasing from `r = 0` and level past the last node, with the bottom depth at each node in metres, strictly positive and never below the sound-speed profile's last depth. Default (`None`): the level bottom at the profile's last depth. |

**Returns:** A [`RayTraceResult`](/phonometry/reference/api/underwater/numerical/#raytraceresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## RayTraceResult

```python
RayTraceResult(
    launch_angles: NDArray[np.float64],
    ranges: NDArray[np.float64],
    depths: NDArray[np.float64],
    travel_times: NDArray[np.float64],
    arc_lengths: NDArray[np.float64],
    surface_reflections: NDArray[np.int_],
    bottom_reflections: NDArray[np.int_],
    source_depth: float,
    water_depth: float,
    profile_depths: NDArray[np.float64],
    profile_speeds: NDArray[np.float64],
    bathymetry_ranges: NDArray[np.float64] | None = None,
    bathymetry_depths: NDArray[np.float64] | None = None,
)
```

Ray-tracing solution through a sound-speed profile.

**Attributes**

| Name | Description |
| :--- | :--- |
| `launch_angles` | Launch angles from the horizontal, in degrees. |
| `ranges` | Per-ray horizontal ranges, in metres, shape `(n_rays, n_steps)`. |
| `depths` | Per-ray depths, in metres, shape `(n_rays, n_steps)`. |
| `travel_times` | Per-ray cumulative travel times, in seconds, shape `(n_rays, n_steps)` (zero at the source, increasing along the ray). |
| `arc_lengths` | Per-ray cumulative arc length along the ray, in metres, same shape (zero at the source). It is never less than the range column it stands in, exceeds it by the obliquity of the path, and a reflection leaves it continuous. This, and not the range, is the measure seawater absorption acts along: Jensen Sect. 3.6.2 carries a volume loss $\alpha$ into the ray solution by perturbing the eikonal and lands on $e^{-\int_0^s \alpha(s')\,ds'}$ (Eq. 3.116), an integral over the path actually flown, so a caller hanging amplitudes on these rays multiplies by $e^{-\alpha s}$ with the $s$ read off here. |
| `surface_reflections` | Per-ray cumulative count of sea-surface reflections by each range sample, same shape (zero at the source). |
| `bottom_reflections` | The same count for the seabed. The two counts, and not the reflection coefficients themselves, are the whole of the per-bounce record an amplitude carrier needs from the geometry. Jensen Sect. 3.6.3 treats a boundary interaction as multiplying the ray amplitude by $\vert \mathcal{R}(\theta)\vert $ and adding $\arg \mathcal{R}(\theta)$ to its phase (Eqs. 3.125-3.126), with $\theta$ the local angle of incidence; and in a range-independent medium that angle is the *same* at every touch of the same flat boundary, because the direction a ray crosses a depth with is fixed by Snell's invariant, $\cos\theta = \xi\,c$, not by how many times it has bounced. Any boundary coefficient therefore enters a path's amplitude only as $\mathcal{R}^n$ with the $n$ read off here, which is how [`gaussian_beams`](/phonometry/reference/api/underwater/numerical/#gaussian_beams) charges its lossy seabed and how [`eigenrays`](/phonometry/reference/api/underwater/numerical/#eigenrays) charges each arrival; `ray_trace` itself carries no amplitude, so the counts are what it can meaningfully expose. |
| `source_depth` | Source depth, in metres. |
| `water_depth` | Water-column depth, in metres. |
| `profile_depths` | Depth samples of the sound-speed profile the rays were traced through, in metres, exactly as cleaned on the way in. The pair below *is* the medium (both solvers interpolate it piecewise linearly and nothing else about the water enters the geometry), so recording it makes the result self-contained: [`eigenrays`](/phonometry/reference/api/underwater/numerical/#eigenrays) needs it to put fresh rays through the same water the fan flew. |
| `profile_speeds` | Sound speed at each of those depths, in m/s. |
| `bathymetry_ranges` | Node ranges of the bottom profile the rays were traced over, in metres, or `None` for the level bottom at `water_depth` (the default). With a sloping bottom two of the flat record's invariants fall, and the arrays here say so honestly rather than quietly keep their old meaning. A ray reflected past the vertical by the accumulating slope cannot be carried by a range march (see [`ray_trace`](/phonometry/reference/api/underwater/numerical/#ray_trace)); its `depths`, `travel_times` and `arc_lengths` are `NaN` from the sample of the terminating bounce on, so a plot simply ends where the ray turned and nothing downstream can mistake a frozen sample for a traced one (the reflection counts, being integers, instead hold their last value). And the crossing angle at the bottom is no longer one per ray: each slope bounce rotates Snell's invariant, so the per-bounce record that sufficed for a flat guide (counts alone) does not price a sloping one, which is why [`eigenrays`](/phonometry/reference/api/underwater/numerical/#eigenrays) declines such a trace. |
| `bathymetry_depths` | Bottom depth at each of those nodes, in metres (`None` likewise). Between nodes the bottom is the straight facet, beyond the last node it continues level: exactly the boundary the marcher reflected off. |

### RayTraceResult.plot()

```python
RayTraceResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the ray paths (depth increasing downward).

## VolumeAbsorption

```python
VolumeAbsorption(
    model: str,
    temperature: float = 10.0,
    salinity: float = 35.0,
    ph: float = 8.0,
)
```

Seawater volume absorption: a model and the water it is evaluated in.

The same models, spelled the same way, as
[`seawater_absorption`](/phonometry/reference/api/underwater/closed-form/#seawater_absorption);
[`gaussian_beams`](/phonometry/reference/api/underwater/numerical/#gaussian_beams) evaluates the coefficient once, at the source
frequency and the source depth, and charges it along each beam's own arc
length (Jensen Eq. 3.116). Passing the bare model name to the solver is
the same as passing this class with its defaults.

**Attributes**

| Name | Description |
| :--- | :--- |
| `model` | `"francois-garrison"`, `"ainslie-mccolm"` or `"thorp"`. |
| `temperature` | Temperature `T`, in degrees Celsius. |
| `salinity` | Salinity `S`, in parts per thousand. |
| `ph` | Acidity (Thorp ignores it always). |
