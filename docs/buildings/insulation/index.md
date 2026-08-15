← [Documentation index](../../README.md)

# Sound insulation

Sound insulation has a life cycle, and the pages of this section follow it.
An element (a wall build-up, a floating floor, a window) is first
characterised **in the laboratory**, where suppressed flanking isolates its
direct transmission. That laboratory data feeds a **prediction** of how a
whole building will perform, flanking paths included. The finished building is
then **verified in the field**. At every stage the band spectrum is collapsed
to the **single number** regulations quote, and for almost everything that
collapse is one shared reference-curve engine rather than a step of any single
method. The exception is the heavy-impact rating of ISO 717-2 Annex D, which
shifts no curve at all: it sums A-weighted band levels in energy.

**Laboratory.**
[Laboratory Insulation Measurement](insulation-lab.md) covers
the ISO 10140 sound reduction index and normalized impact level with their
background-noise correction. Two laboratory methods sit beside it:
[Sound Insulation by Intensity (ISO 15186)](insulation-intensity.md)
reads the transmitted power off the radiating face when flanking is too high
for the pressure method, and
[Laboratory Flanking Transmission (ISO 10848)](flanking-lab.md)
measures the junction data the prediction consumes. A third, the
floor-covering improvement of ISO 16251-1, is filed with the design pages it
feeds.

**Prediction.** The design-stage half has a section of its own,
[Insulation design](../design/index.md): the
EN 12354 flanking model between rooms, the theoretical insulation of a panel,
and the two material measurements a floor design consumes.

**Field.**
[Field Insulation Measurement (ISO 16283)](insulation-field.md)
covers the engineering-grade airborne and impact measurement in the building,
its Clause 14 test report and the ISO 12999-1 uncertainty that qualifies it.
The same standard specifies two more impact sources, a rubber ball and a bang
machine, for the slow low-frequency thumps a tapping machine says nothing
about;
[Heavy and Soft Impact Sources (ISO 16283-2)](heavy-impact-sources.md)
covers their specification, the Fast-weighted standardization of the maximum
level and the Annex D rating.
When the question does not deserve that effort,
[Sound Insulation Survey Method (ISO 10052)](insulation-survey.md)
trades accuracy for speed with octave bands and a reverberation index.

**Ratings and the envelope.**
[Insulation Ratings (ISO 717)](insulation-ratings.md) is the
reference-curve engine every one of those methods ends on, with its spectrum
adaptation terms C, Ctr and CI. And
[Façade Sound Insulation](facade-insulation.md) keeps the
building envelope in one place: measured per ISO 16283-3, predicted per
EN 12354-3, and radiating outwards per EN 12354-4. National building codes
restate those ratings in their own global quantities, and
[Spanish Building Code (CTE DB-HR)](spanish-building-code.md)
implements the Spanish one: the direct Annex A index over eighteen bands, the
clause 2 requirement tables and the window-size correction.

Two neighbouring sections complete the picture: the room-side quantities
(reverberation time, absorption) live in
[Room acoustics](../rooms/index.md), and the
structure-borne noise of building *equipment*, predicted by the closely
related EN 12354-5, lives in
[Structure-borne sources](../../vibration/structural/index.md).

## Pages in this section

- [Field Insulation Measurement (ISO 16283)](insulation-field.md):
  ISO 16283-1/2 field measurement, its test report and the ISO 12999-1
  uncertainty.
- [Laboratory Insulation Measurement](insulation-lab.md):
  the ISO 10140 characterisation of an element with flanking suppressed.
- [Sound Insulation by Intensity (ISO 15186)](insulation-intensity.md):
  the ISO 15186-1/-2 direct-power route to the same indices.
- [Sound Insulation Survey Method (ISO 10052)](insulation-survey.md):
  the octave-band control method, its reverberation index and its survey
  quantities.
- [Laboratory Flanking Transmission (ISO 10848)](flanking-lab.md):
  the measured vibration reduction index Kij, the flanking descriptors Dn,f
  and Ln,f, and the suspended-ceiling plenum path with its normalized ceiling
  attenuation Dn,c and ceiling attenuation class.
- [Heavy and Soft Impact Sources (ISO 16283-2)](heavy-impact-sources.md):
  the rubber ball and the bang machine, the impact force exposure levels that
  specify them, the Fast-weighted maximum level and the ISO 717-2 Annex D
  single number.
- [Insulation Ratings (ISO 717)](insulation-ratings.md):
  the ISO 717-1/-2 reference-curve engines, C, Ctr and CI, the enlarged-range
  terms and the ISO 717 fiche.
- [Façade Sound Insulation](facade-insulation.md):
  the envelope measured (ISO 16283-3), predicted (EN 12354-3) and radiating
  outwards (EN 12354-4).
- [Spanish Building Code (CTE DB-HR)](spanish-building-code.md):
  the DB-HR global indices RA, RA,tr, DnT,A and D2m,nT,Atr, the clause 2
  requirements and the window-size correction.

## What this section does not cover

**The library starts after the microphone.** Every function here accepts
per-position spectra and energy-averages them for you, but nothing verifies
how the measurement was made: not the number and placement of the source and
microphone positions behind those spectra, not the low-frequency procedures
of ISO 16283-1/-2, and not the test-facility and mounting requirements of
ISO 10140-1. Those are the operator's responsibility and the report's, and
they are what makes the numbers here mean something. Background noise is the
one correction genuinely left to the caller: field levels must arrive already
corrected, their 6 dB signal-to-background floor checked by the operator,
while the ISO 10140-4 laboratory helper applies its rule itself, warning when
its own floor is broken and capping the correction. Two consequences worth
naming: the field and laboratory background corrections are *different*
rules, so that laboratory helper must not be applied to field data; and the
intensity route takes both the pressure and the intensity level as inputs,
with the scanning probe and its phase-mismatch calibration outside the
library.

**Coverage inside the standards is partial in two places.** Of ISO 10848 only
the Part 1 formulae are implemented generically, plus the Part 4 modal-overlap
validity check, not the facility-specific setups of Parts 2, 3 and 4. Of the
Spanish code, only the verification indices are implemented: the simplified
option's solution tables of clause 3, the execution conditions of clause 5 and
the maintenance conditions of clause 6 are out of scope, and the general
option's calculation route is [Predicting Sound
Insulation](../design/insulation-prediction.md).

**And there is no heavy-impact prediction at all.** A floor construction can
be carried to a tapping-machine level by the models in [Insulation
design](../design/index.md); nothing does the same for the rubber
ball, because the complexity of the input force and the use of a time-weighted
maximum leave no simple counterpart. The heavy-impact page rates a
measurement, and only a measurement.
