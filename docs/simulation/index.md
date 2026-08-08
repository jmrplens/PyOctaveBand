← [Documentation index](../README.md)

# Wave simulation

Most of this library predicts a number; this section computes the **wave
field itself**. A finite-difference time-domain (FDTD) solver integrates the
linear acoustic equations on a 2D grid, so reflection, diffraction,
interference, modal behaviour and refraction through inhomogeneous media all
emerge from first principles, and its elastic companion carries the same
scheme into solids. Both solvers are deterministic (identical inputs give
bit-identical outputs on the same platform), validated against analytic
oracles, and double as a cross-check engine for the closed-form models of
the other sections.

The section splits along the media it simulates. The acoustic page explains
the numerical method (the staggered leapfrog scheme and its Courant
stability bound), the building blocks (sources, probes, obstacles and
boundary conditions, including the locally reacting real-impedance edge),
the near-to-far-field transformation, when a wave-based simulation is worth
its cost, what a 2D domain can and cannot say about a 3D problem, and how
numerical dispersion sets the cells-per-wavelength resolution rule. The
elastic page extends the same staggered grid to solids: shear waves, free
surfaces with Rayleigh waves, and fluid-solid coupling with mode
conversion, Scholte interface waves and immersed-plate transmission.

A good way to read it is alongside the closed-form pages it cross-checks:
the modal frequencies of [room acoustics](../buildings/rooms/room-acoustics.md)
reappear as peaks in a simulated room spectrum, the barrier insertion loss of
[ground effect and barriers](../environment/propagation/ground-barriers.md) can be
re-derived by placing an obstacle in the domain, and the ray bending of
[atmospheric refraction](../environment/propagation/atmospheric-refraction.md) emerges
from a height-dependent sound-speed profile. When a geometry is too irregular
for those models (odd-shaped rooms, multiple barriers, mixed impedance
ground), the simulation is the fallback that still gives a quantitative
answer; when a closed form exists, prefer it, and use the solver to verify
the assumptions it rests on.

Those cross-checks are not only arguments: fifteen of the animations in this
documentation are output from these two solvers, and they are filed on the
guides whose physics they settle rather than here. Room modes growing on and
off resonance appear in [room acoustics](../buildings/rooms/room-acoustics.md)
and [reverberation prediction](../buildings/rooms/reverberation-prediction.md),
which also carries the hall of columns that turns one wavefront into a mixed
field; barrier diffraction at two wavelengths in
[outdoor propagation](../environment/propagation/outdoor-propagation.md)
and [ground effect and barriers](../environment/propagation/ground-barriers.md);
downwind and upwind refraction in
[atmospheric refraction](../environment/propagation/atmospheric-refraction.md);
the ground-effect lobe pattern in outdoor propagation and in
[airport noise](../aircraft/airport-noise.md); the standing-wave and
transmission tubes in [the impedance tube](../materials/absorbers/impedance-tube.md);
the QRD and metadiffuser panels in
[diffusers](../materials/diffusers/diffusers.md) and
[metadiffusers](../materials/diffusers/metadiffusers.md); the slit
absorber in [metamaterial absorbers](../materials/absorbers/metamaterial-absorbers.md);
the expansion chamber in [silencers](../devices/noise-control/silencers.md);
the wall aperture in [panel sound insulation](../buildings/design/panel-sound-insulation.md);
and the SOFAR duct in [underwater propagation](../underwater/underwater-propagation.md).
The elastic solver adds two: the bending packet entering an L-junction, on
[junction transmission](../vibration/structural/junction-transmission.md),
and the coincidence plate, on panel sound insulation. Both also appear on the
elastic page below, where the solver that produced them is explained.

## Pages in this section

- [2D FDTD wave simulation](fdtd-simulation.md): the
  staggered-grid pressure-velocity FDTD method following Attenborough & Van
  Renterghem (2021) chapter 4, its sources, probes, obstacles and boundary
  conditions, the near-to-far-field chain, the 2D limits and the numerical
  dispersion rule.
- [Elastic waves and fluid-solid coupling](elastic-waves.md):
  the P-SV velocity-stress companion solver (Virieux 1986) on the same
  grid, with stress-imaging free surfaces, Rayleigh waves, mode conversion,
  Scholte interface waves and immersed-plate transmission, each validated
  against its exact closed form.
