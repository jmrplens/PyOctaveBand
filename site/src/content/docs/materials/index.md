---
title: "Materials and surfaces"
description: "Characterising acoustic materials and surfaces: reverberation-room absorption and its rating (ISO 354, ISO 11654), airflow resistance (ISO 9053), the impedance tube (ISO 10534, ASTM E2611), porous, multilayer and metamaterial absorber prediction, diffusers with their scattering and diffusion coefficients (ISO 17497), metadiffusers, and in-situ road-surface absorption (ISO 13472)."
---

Every room prediction and many an insulation model end up consuming a
coefficient that describes what a material or a surface does to sound. This
section covers where those coefficients come from: the laboratory instruments
that measure them, the single-number ratings that summarise them, the
prediction models that anticipate them, and the in-situ methods that recover
them outside the laboratory.

The **Absorbers** subsection covers how much energy a material takes out of
the field, one instrument or model family per guide.
[Sound Absorption Measurement and Rating](/phonometry/materials/absorbers/absorption-measurement/)
is the reverberation room: the random-incidence coefficients of ISO 354 and
the ISO 11654 weighted rating α_w with its letter class, the figure absorber
datasheets quote, with the ISO 12999-2 measurement uncertainty.
[Airflow Resistance](/phonometry/materials/absorbers/airflow-resistance/) is the flow rig:
resistance and resistivity per ISO 9053-1/-2, the parameter that governs a
porous absorber's low-frequency behaviour and anchors most material models.
[Impedance Tube](/phonometry/materials/absorbers/impedance-tube/) is the bench instrument:
the complex surface impedance, reflection factor and absorption of a small
sample at normal incidence (ISO 10534-1/-2) and, with four microphones, its
transmission loss (ASTM E2611), plus the virtual FDTD tube that cross-checks
the wave solver against the same standards.
[Porous and Multilayer Absorbers](/phonometry/materials/absorbers/porous-absorbers/) turns
the measured flow resistivity into *predictions*: the Delany-Bazley, Miki and
Johnson-Champoux-Allard equivalent-fluid models give a porous material's
characteristic impedance and wavenumber, and a transfer-matrix stack of
porous, air, perforated, microperforated (Maa) and membrane layers predicts
the absorption of a whole construction before anything is built.
[Metamaterial Absorbers](/phonometry/materials/absorbers/metamaterial-absorbers/) pushes
the same transfer matrices past the classical thickness rules: slow-sound
slit panels loaded by Helmholtz resonators reach perfect absorption at
critical coupling from deep-subwavelength panels.

The **Diffusers and surfaces** subsection moves from samples to *surfaces*,
asking not how much energy a surface absorbs but where it sends what it
reflects. [Diffusers and Their Coefficients](/phonometry/materials/diffusers/diffusers/)
covers the two standardised gradings, the random-incidence **scattering
coefficient** (ISO 17497-1) and the **diffusion coefficient** (ISO 17497-2),
together with Schroeder diffuser design and its far-field prediction.
[Metadiffusers](/phonometry/materials/diffusers/metadiffusers/) rebuilds the Schroeder
diffuser from resonator-loaded slits, one to two orders of magnitude thinner.
And [In-situ Road-Surface Absorption](/phonometry/materials/surfaces/road-absorption/)
measures the absorption of a pavement where it lies, by the ISO 13472-1
subtraction technique over an extended surface or the ISO 13472-2 spot tube.

The consumers of these numbers are spread across the site: absorption
coefficients feed the reverberation predictions in
[Room acoustics](/phonometry/buildings/rooms/), dynamic
stiffness (measured by a related load-plate method) feeds the floating-floor
model in [Sound insulation](/phonometry/buildings/insulation/),
and the road-surface methods connect to the outdoor-noise interest of the
[Environment and transport](/phonometry/environment/)
section.

## Pages in this section

- [Absorbers overview](/phonometry/materials/absorbers/): the
  measurement chain from reverberation room to flow rig to impedance tube,
  and the prediction models that tie them together.
- [Sound Absorption Measurement and Rating](/phonometry/materials/absorbers/absorption-measurement/):
  the ISO 354 measurement, the ISO 11654 weighted rating and class, and the
  ISO 12999-2 uncertainty.
- [Airflow Resistance](/phonometry/materials/absorbers/airflow-resistance/): the ISO 9053
  static and alternating methods.
- [Impedance Tube](/phonometry/materials/absorbers/impedance-tube/): normal-incidence
  absorption, impedance and ASTM E2611 transmission loss, plus the virtual
  FDTD tube.
- [Porous and Multilayer Absorbers](/phonometry/materials/absorbers/porous-absorbers/):
  the Delany-Bazley, Miki and JCA porous models, the transfer-matrix
  multilayer solver with perforated, microperforated and membrane layers,
  and the random-incidence Paris integral.
- [Metamaterial Absorbers](/phonometry/materials/absorbers/metamaterial-absorbers/):
  critical coupling and the slow-sound slit panel with its design solver.
- [Diffusers and surfaces overview](/phonometry/materials/diffusers/):
  what a surface does with the sound it returns, from coefficients to
  metamaterial panels to pavements.
- [Diffusers and Their Coefficients](/phonometry/materials/diffusers/diffusers/):
  ISO 17497-1/2 scattering and diffusion coefficients, Schroeder design and
  the far-field prediction.
- [Metadiffusers](/phonometry/materials/diffusers/metadiffusers/): deep-subwavelength
  Schroeder diffusers from resonator-loaded slits.
- [Surfaces measured in place overview](/phonometry/materials/surfaces/):
  surfaces that cannot be taken to a laboratory, characterised where they lie.
- [In-situ Road-Surface Absorption](/phonometry/materials/surfaces/road-absorption/):
  ISO 13472-1/-2 in-situ road-surface absorption.
- [Resilient layers overview](/phonometry/materials/resilient/):
  what a resilient layer does under a floating floor, and the dynamic
  stiffness that sets it.
- [Dynamic stiffness of resilient materials (EN 29052-1)](/phonometry/materials/resilient/dynamic-stiffness/):
  the resonance method that measures what a resilient layer does under a
  floating floor, and the apparent stiffness the insulation design chapter
  asks it for.
