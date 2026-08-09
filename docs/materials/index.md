← [Documentation index](../README.md)

# Materials and surfaces

Every room prediction and many an insulation model end up consuming a
coefficient that describes what a material or a surface does to sound. This
section covers where those coefficients come from: the laboratory instruments
that measure them, the single-number ratings that summarise them, the
prediction models that anticipate them, and the in-situ methods that recover
them outside the laboratory.

The **Absorbers** subsection covers how much energy a material takes out of
the field, one instrument or model family per guide.
[Sound Absorption Measurement and Rating](absorbers/absorption-measurement.md)
is the reverberation room: the random-incidence coefficients of ISO 354 and
the ISO 11654 weighted rating α_w with its letter class, the figure absorber
datasheets quote, with the ISO 12999-2 measurement uncertainty.
[Airflow Resistance](absorbers/airflow-resistance.md) is the flow rig:
resistance and resistivity per ISO 9053-1/-2, the parameter that governs a
porous absorber's low-frequency behaviour and anchors most material models.
[Impedance Tube](absorbers/impedance-tube.md) is the bench instrument:
the complex surface impedance, reflection factor and absorption of a small
sample at normal incidence (ISO 10534-1/-2) and, with four microphones, its
transmission loss (ASTM E2611), plus the virtual FDTD tube that cross-checks
the wave solver against the same standards.
[Porous and Multilayer Absorbers](absorbers/porous-absorbers.md) turns
the measured flow resistivity into *predictions*: the Delany-Bazley, Miki and
Johnson-Champoux-Allard equivalent-fluid models give a porous material's
characteristic impedance and wavenumber, and a transfer-matrix stack of
porous, air, perforated, microperforated (Maa) and membrane layers predicts
the absorption of a whole construction before anything is built.
[Metamaterial Absorbers](absorbers/metamaterial-absorbers.md) pushes
the same transfer matrices past the classical thickness rules: slow-sound
slit panels loaded by Helmholtz resonators reach perfect absorption at
critical coupling from deep-subwavelength panels.

The **Diffusers and surfaces** subsection moves from samples to *surfaces*,
asking not how much energy a surface absorbs but where it sends what it
reflects. [Diffusers and Their Coefficients](diffusers/diffusers.md)
covers the two standardised gradings, the random-incidence **scattering
coefficient** (ISO 17497-1) and the **diffusion coefficient** (ISO 17497-2),
together with Schroeder diffuser design and its far-field prediction.
[Metadiffusers](diffusers/metadiffusers.md) rebuilds the Schroeder
diffuser from resonator-loaded slits, one to two orders of magnitude thinner.

Nested inside that group is **Surfaces measured in place**, for the surfaces
that have no sample. A pavement cannot be cut out and carried indoors without
destroying the connected pore structure that governs its absorption, so the
laboratory geometry is replaced by a time window over an extended surface, or
by a tube pressed onto the road.
[In-situ Road-Surface Absorption](surfaces/road-absorption.md)
measures it where it lies, by the ISO 13472-1 subtraction technique or the
ISO 13472-2 spot tube, and says which of the two a given pavement allows.

The **Resilient layers** subsection covers the one material property here that
is mechanical rather than acoustic: a resilient layer is characterised not by
what it does to airborne sound but by how softly it supports a mass, so its
measurement is a resonance and not an absorption. A floating floor is a
mass-spring system, the screed is the mass and the layer is the spring, and the
dynamic stiffness per unit area s' of the layer sets the resonance above which
the floor starts working.
[Dynamic stiffness of resilient materials (EN 29052-1)](resilient/dynamic-stiffness.md)
is the load-plate resonance measurement that produces s', with the enclosed-gas
term that makes an air-permeable layer stiffer than its frame alone.

The consumers of these numbers are spread across the site: absorption
coefficients feed the reverberation predictions in
[Room acoustics](../buildings/rooms/index.md); the dynamic stiffness measured
here feeds the floating-floor model of
[Sound insulation](../buildings/insulation/index.md) through
[Predicting resilient-layer performance](../buildings/design/resilient-layers.md);
and the road-surface methods connect to the outdoor-noise interest of the
[Environment and transport](../environment/index.md)
section.

## [Absorbers](absorbers/index.md)

How much energy a material takes out of the field, one instrument or model
family per guide.

- [Absorbers overview](absorbers/index.md): the
  measurement chain from reverberation room to flow rig to impedance tube,
  and the prediction models that tie them together.
- [Sound Absorption Measurement and Rating](absorbers/absorption-measurement.md):
  the ISO 354 measurement, the ISO 11654 weighted rating and class, and the
  ISO 12999-2 uncertainty.
- [Airflow Resistance](absorbers/airflow-resistance.md): the ISO 9053
  static and alternating methods.
- [Impedance Tube](absorbers/impedance-tube.md): normal-incidence
  absorption, impedance and ASTM E2611 transmission loss, plus the virtual
  FDTD tube.
- [Porous and Multilayer Absorbers](absorbers/porous-absorbers.md):
  the Delany-Bazley, Miki and JCA porous models, the transfer-matrix
  multilayer solver with perforated, microperforated and membrane layers,
  and the random-incidence Paris integral.
- [Metamaterial Absorbers](absorbers/metamaterial-absorbers.md):
  critical coupling and the slow-sound slit panel with its design solver.

## [Diffusers and surfaces](diffusers/index.md)

Where a surface sends what it reflects, and the surfaces that can only be
measured in place.

- [Diffusers and surfaces overview](diffusers/index.md):
  what a surface does with the sound it returns, from coefficients to
  metamaterial panels.
- [Diffusers and Their Coefficients](diffusers/diffusers.md):
  ISO 17497-1/2 scattering and diffusion coefficients, Schroeder design and
  the far-field prediction.
- [Metadiffusers](diffusers/metadiffusers.md): deep-subwavelength
  Schroeder diffusers from resonator-loaded slits.
- [Surfaces measured in place overview](surfaces/index.md):
  surfaces that cannot be taken to a laboratory, characterised where they lie.
- [In-situ Road-Surface Absorption](surfaces/road-absorption.md):
  ISO 13472-1/-2 in-situ road-surface absorption.

## [Resilient layers](resilient/index.md)

The mechanical property a floating floor is designed around.

- [Resilient layers overview](resilient/index.md):
  what a resilient layer does under a floating floor, and the dynamic
  stiffness that sets it.
- [Dynamic stiffness of resilient materials (EN 29052-1)](resilient/dynamic-stiffness.md):
  the resonance method that measures what a resilient layer does under a
  floating floor, and the apparent stiffness the insulation design chapter
  asks it for.

## What this section does not cover

Everything here characterises a **material or a surface**, never a
construction. The transmission loss of a wall, the impact improvement of a
floor and the flanking paths of a junction are
[Sound insulation](../buildings/insulation/index.md) and
[Insulation design](../buildings/design/index.md); this section supplies the
coefficients they consume. Two boundaries inside the measurements themselves
are worth knowing before you start. The in-situ road methods implement
ISO 13472-1:2002 and ISO 13472-2:2010; **their 2022 and 2025 revisions are not
implemented**. And the resilient-layer measurement expects a resonant frequency
that has already been extrapolated to zero force amplitude by clause 7 of
EN 29052-1, a procedure that is not implemented, and an airflow resistivity
supplied as an input rather than measured in place. Nothing in this section
predicts a material from its chemistry or its manufacture: the models run
forwards from measured macroscopic parameters — flow resistivity, porosity,
tortuosity — to an impedance, and there is no inverse solver that recovers
those parameters from a measured impedance curve.

## Before and after these pages

Every coefficient on these pages is derived from band levels or from a
transfer function between microphones, so the filtering, weighting and
calibration that produce them are in [Signal analysis](../signals/index.md),
and [Build a sound level meter](../signals/sound-level-meter.md) runs
that chain end to end on one runnable page. The derivations are in [Materials
and surfaces theory](../reference/theory/materials-surfaces.md): the characterisation
quantities, the in-situ subtraction and the scattering and diffusion
coefficients.

If you arrived here from a search and want the shape of the whole library,
[What do you need to measure?](https://jmrplens.github.io/phonometry/start/tasks/) indexes it by the job
and [All guides](../README.md) lists every page with a line on
each.
