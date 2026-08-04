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
