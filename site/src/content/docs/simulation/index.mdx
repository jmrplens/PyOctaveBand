---
title: "Wave simulation"
description: "Computing the sound field itself: a deterministic 2D FDTD solver on a staggered pressure-velocity grid, with sources, probes, rasterised obstacles and per-side boundaries, plus its elastic P-SV companion with Rayleigh waves, fluid-solid coupling and Scholte interface waves."
---

Most of this library predicts a number; this section computes the **wave
field itself**. A finite-difference time-domain (FDTD) solver integrates the
linear acoustic equations on a 2D grid, so reflection, diffraction,
interference, modal behaviour and refraction through inhomogeneous media all
emerge from first principles, and its elastic companion carries the same
scheme into solids. Both solvers are deterministic (identical inputs give
bit-identical outputs on the same platform), validated against analytic
oracles, and double as a cross-check engine for the closed-form models of
the other sections.

The 2D domain is a **cross-section**, and that has one consequence worth
settling before any number leaves the solver: a point in the plane is
physically an infinite line source, so amplitudes fall as the inverse square
root of distance, about 3 dB per doubling instead of 6. Interference and
diffraction patterns, arrival times and modal frequencies are faithful;
absolute levels and decay rates belong to that geometry and not to a 3D room.
A quantitative 3D claim needs a closed form or a 3D solver behind it.

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
the modal frequencies of [room acoustics](/phonometry/buildings/rooms/room-acoustics/)
reappear as peaks in a simulated room spectrum, the barrier insertion loss of
[ground effect and barriers](/phonometry/environment/propagation/ground-barriers/) can be
re-derived by placing an obstacle in the domain, and the ray bending of
[atmospheric refraction](/phonometry/environment/propagation/atmospheric-refraction/) emerges
from a height-dependent sound-speed profile. When a geometry is too irregular
for those models (odd-shaped rooms, multiple barriers, mixed impedance
ground), the simulation is the fallback that still gives a quantitative
answer; when a closed form exists, prefer it, and use the solver to verify
the assumptions it rests on.

Those cross-checks are not only arguments: fifteen of the animations in this
documentation are output from these two solvers, and they are filed on the
guides whose physics they settle rather than here. Room modes growing on and
off resonance appear in [room acoustics](/phonometry/buildings/rooms/room-acoustics/)
and [reverberation prediction](/phonometry/buildings/rooms/reverberation-prediction/),
which also carries the hall of columns that turns one wavefront into a mixed
field; barrier diffraction at two wavelengths in
[outdoor propagation](/phonometry/environment/propagation/outdoor-propagation/)
and [ground effect and barriers](/phonometry/environment/propagation/ground-barriers/);
downwind and upwind refraction in
[atmospheric refraction](/phonometry/environment/propagation/atmospheric-refraction/);
the ground-effect lobe pattern in outdoor propagation and in
[airport noise](/phonometry/aircraft/airport-noise/); the standing-wave and
transmission tubes in [the impedance tube](/phonometry/materials/absorbers/impedance-tube/);
the QRD and metadiffuser panels in
[diffusers](/phonometry/materials/diffusers/diffusers/) and
[metadiffusers](/phonometry/materials/diffusers/metadiffusers/); the slit
absorber in [metamaterial absorbers](/phonometry/materials/absorbers/metamaterial-absorbers/);
the expansion chamber in [silencers](/phonometry/devices/noise-control/silencers/);
the wall aperture in [panel sound insulation](/phonometry/buildings/design/panel-sound-insulation/);
and the SOFAR duct in [underwater propagation](/phonometry/underwater/underwater-propagation/).
The elastic solver adds two: the bending packet entering an L-junction, on
[junction transmission](/phonometry/vibration/structural/junction-transmission/),
and the coincidence plate, on panel sound insulation. Both also appear on the
elastic page below, where the solver that produced them is explained.

Several of those guides do more than illustrate: they run a **whole
standardised measurement inside the domain**. The impedance-tube guide performs
the ISO 10534-2 and ASTM E2611 reductions on a simulated tube and recovers the
sample's analytic absorption and transmission loss; the diffuser and
metadiffuser guides drive meshed panels with a plane wave and transform the near
field into a polar response; the panel-insulation and plate-junction guides
launch bending waves into a plate and watch coincidence and junction splitting
happen. In each case the solver is standing in for the laboratory, which is
what makes the closed-form comparison a real test rather than a demonstration.

Setting up a run is a chain, and each link fixes the next. The highest frequency
you need and the **slowest** sound speed anywhere in the domain fix the cell
size, through the cells-per-wavelength rule that numerical dispersion sets. The
cell size and the **fastest** speed then fix the time step, through the Courant
stability bound. The domain has to hold the geometry plus clearance for the
absorbing layers, which are themselves sized by the **lowest** frequency. The
run must last long enough for the field to cross the domain, and for a
steady-state answer long enough for the transient to leave before the analysis
window opens. The cost is cells times steps, so halving the cell size costs
eight times more in 2D — a factor of four in cells and a factor of two in steps.
The acoustic page gives the numbers for each link, and the elastic page adds the
extra sampling that free surfaces and interface waves demand.

## Pages in this section

Read the acoustic page first: the elastic page assumes its vocabulary and says
so in its own opening.

- [2D FDTD wave simulation](/phonometry/simulation/fdtd-simulation/): the
  staggered-grid pressure-velocity FDTD method following Attenborough & Van
  Renterghem (2021) chapter 4, its sources, probes, obstacles and boundary
  conditions, the near-to-far-field chain, the 2D limits and the numerical
  dispersion rule.
- [Elastic waves and fluid-solid coupling](/phonometry/simulation/elastic-waves/):
  the P-SV velocity-stress companion solver (Virieux 1986) on the same
  grid, with stress-imaging free surfaces, Rayleigh waves, mode conversion,
  Scholte interface waves and immersed-plate transmission, each validated
  against its exact closed form.

## What this section does not cover

**Two dimensions, and no way around it.** Everything a 2D cross-section cannot
say about a 3D room, neither solver says, and the cylindrical spreading above is
only the most visible consequence. The open boundary is a quadratic-ramp
absorbing layer — the simple precursor of a perfectly matched layer, not a PML —
so grazing incidence is absorbed less cleanly than normal incidence, and the
elastic solver has no elastic PML at all, which shows most on grazing Rayleigh
waves. The medium is non-moving: **wind and flow advection are not modelled**,
so a refraction study here comes from a height-dependent sound-speed profile and
not from a flow field. The only impedance boundary is a frequency-independent
real one, so a porous absorber has to be meshed rather than declared. On the
elastic side the solid is isotropic and purely elastic: no anisotropy, and no
viscoelastic damping beyond the bulk decay rate, so a material loss factor
cannot be entered.

And nothing here is a room-acoustics package. There is no geometry importer, no
material library, no ray tracer, no auralisation and no 3D solver: obstacles are
rasterised onto the grid from shapes you define, and the output is a field you
analyse yourself.

## Before and after these pages

The fields these solvers produce are read back with the same tools as a
measurement: the filtering, weighting and level functions of [Signal
analysis](/phonometry/signals/), with [Build a sound level
meter](/phonometry/signals/sound-level-meter/) running that chain end to end on
one runnable page. There is no theory-reference page for the solvers; the
derivations, the stability conditions and the analytic oracles stay inside the
two guides above.

If you arrived here from a search and want the shape of the whole library,
[What do you need to measure?](/phonometry/start/tasks/) indexes it by the job
and [All guides](/phonometry/start/guides/) lists every page with a line on
each.
