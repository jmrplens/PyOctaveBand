---
title: "Materials and surfaces"
description: "Characterising acoustic materials and surfaces: reverberation-room absorption and its rating (ISO 354, ISO 11654), airflow resistance (ISO 9053), the impedance tube (ISO 10534, ASTM E2611), porous and multilayer absorber prediction (Delany-Bazley, Miki, JCA, Maa), scattering and diffusion coefficients (ISO 17497) and in-situ road-surface absorption (ISO 13472)."
---

Every room prediction and many an insulation model end up consuming a
coefficient that describes what a material or a surface does to sound. This
section covers where those coefficients come from: the laboratory instruments
that measure them, the single-number ratings that summarise them, and the
in-situ methods that recover them outside the laboratory.

The **Absorbers** subsection covers how much energy a material takes out of
the field, one instrument per guide.
[Sound Absorption Measurement and Rating](/phonometry/guides/absorption-measurement/)
is the reverberation room: the random-incidence coefficients of ISO 354 and
the ISO 11654 weighted rating α_w with its letter class, the figure absorber
datasheets quote, with the ISO 12999-2 measurement uncertainty.
[Airflow Resistance](/phonometry/guides/airflow-resistance/) is the flow rig:
resistance and resistivity per ISO 9053-1/-2, the parameter that governs a
porous absorber's low-frequency behaviour and anchors most material models.
[Impedance Tube](/phonometry/guides/impedance-tube/) is the bench instrument:
the complex surface impedance, reflection factor and absorption of a small
sample at normal incidence (ISO 10534-1/-2) and, with four microphones, its
transmission loss (ASTM E2611), plus the virtual FDTD tube that cross-checks
the wave solver against the same standards.
[Porous and Multilayer Absorbers](/phonometry/guides/porous-absorbers/) turns
the measured flow resistivity into *predictions*: the Delany-Bazley, Miki and
Johnson-Champoux-Allard equivalent-fluid models give a porous material's
characteristic impedance and wavenumber, and a transfer-matrix stack of
porous, air, perforated, microperforated (Maa) and membrane layers predicts
the absorption of a whole construction before anything is built.

[Surface Scattering, Diffusion and In-situ Absorption](/phonometry/guides/surface-scattering/)
moves from samples to *surfaces*, asking not how much energy a surface
absorbs but where it sends what it reflects. The random-incidence
**scattering coefficient** (ISO 17497-1) quantifies how much energy leaves the
specular direction; the **diffusion coefficient** (ISO 17497-2) quantifies how
uniformly the reflected energy spreads; and the ISO 13472 methods measure the
absorption of a road surface in situ, by the extended-surface subtraction
technique (Part 1) or the spot tube (Part 2).

The consumers of these numbers are spread across the site: absorption
coefficients feed the reverberation predictions in
[Room acoustics](/phonometry/guides/sections/room-acoustics/), dynamic
stiffness (measured by a related load-plate method) feeds the floating-floor
model in [Sound insulation](/phonometry/guides/sections/sound-insulation/),
and the road-surface methods connect to the outdoor-noise interest of the
[Environment and transport](/phonometry/guides/sections/environment-transport/)
section.

## Pages in this section

- [Absorbers overview](/phonometry/guides/sections/absorbers/): the
  measurement chain from reverberation room to flow rig to impedance tube,
  and the prediction models that tie them together.
- [Sound Absorption Measurement and Rating](/phonometry/guides/absorption-measurement/):
  the ISO 354 measurement, the ISO 11654 weighted rating and class, and the
  ISO 12999-2 uncertainty.
- [Airflow Resistance](/phonometry/guides/airflow-resistance/): the ISO 9053
  static and alternating methods.
- [Impedance Tube](/phonometry/guides/impedance-tube/): normal-incidence
  absorption, impedance and ASTM E2611 transmission loss, plus the virtual
  FDTD tube.
- [Porous and Multilayer Absorbers](/phonometry/guides/porous-absorbers/):
  the Delany-Bazley, Miki and JCA porous models, the transfer-matrix
  multilayer solver with perforated, microperforated and membrane layers,
  and the random-incidence Paris integral.
- [Surface Scattering, Diffusion and In-situ Absorption](/phonometry/guides/surface-scattering/):
  ISO 17497-1/2 scattering and diffusion coefficients, and ISO 13472-1/-2
  in-situ road-surface absorption.
