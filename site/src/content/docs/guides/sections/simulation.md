---
title: "Wave simulation"
description: "Computing the sound field itself: a deterministic 2D acoustic FDTD solver on a staggered pressure-velocity grid, with sources, pressure probes, rasterised obstacles and per-side rigid, impedance or absorbing boundaries."
---

Most of this library predicts a number; this section computes the **wave
field itself**. A finite-difference time-domain (FDTD) solver integrates the
linear acoustic equations on a 2D grid, so reflection, diffraction,
interference, modal behaviour and refraction through inhomogeneous media all
emerge from first principles. The solver is deterministic (identical inputs
give bit-identical outputs on the same platform), validated against analytic
oracles, and doubles as a cross-check engine for the closed-form models of
the other sections.

The single page of this section explains the numerical method (the staggered
leapfrog scheme and its Courant stability bound), the building blocks
(sources, probes, obstacles and boundary conditions, including the
locally reacting real-impedance edge), when a wave-based simulation is worth
its cost, what a 2D domain can and cannot say about a 3D problem, and how
numerical dispersion sets the cells-per-wavelength resolution rule.

A good way to read it is alongside the closed-form pages it cross-checks:
the modal frequencies of [room acoustics](/phonometry/guides/room-acoustics/)
reappear as peaks in a simulated room spectrum, the barrier insertion loss of
[ground effect and barriers](/phonometry/guides/ground-barriers/) can be
re-derived by placing an obstacle in the domain, and the ray bending of
[atmospheric refraction](/phonometry/guides/atmospheric-refraction/) emerges
from a height-dependent sound-speed profile. When a geometry is too irregular
for those models (odd-shaped rooms, multiple barriers, mixed impedance
ground), the simulation is the fallback that still gives a quantitative
answer; when a closed form exists, prefer it, and use the solver to verify
the assumptions it rests on.

## Pages in this section

- [2D FDTD wave simulation](/phonometry/guides/fdtd-simulation/): the
  staggered-grid pressure-velocity FDTD method following Attenborough & Van
  Renterghem (2021) chapter 4, its sources, probes, obstacles and boundary
  conditions, the 2D limits and the numerical dispersion rule.
