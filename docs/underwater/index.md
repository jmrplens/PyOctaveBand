← [Documentation index](../README.md)

# Underwater acoustics

Underwater acoustics runs on the same physics as airborne acoustics but on a
different scale and a different reference: levels are expressed re **1 µPa**
(not 20 µPa), exposure re 1 µPa²·s, and the medium itself, with its
depth-dependent sound speed, refracts sound into channels that carry it for
kilometres. This section covers the discipline along the source-path-receiver
chain of the rest of the library.

That reference difference is the commonest trap for a reader arriving from
airborne acoustics, and it is worth settling before anything else. The same
pressure expressed re 1 µPa is **26 dB larger** than expressed re 20 µPa, which
is arithmetic. On top of that, the same pressure in water carries far less
intensity than in air, because sea water's characteristic impedance is some
3 700 times that of air. An underwater 120 dB and an airborne 120 dB therefore
describe entirely different physical situations, and the two must never be
compared. The rule this section follows is simple: every level carries its
reference explicitly, a conversion between the two conventions is pure
re-referencing and never an energy equivalence, and the only place the airborne
reference appears here at all is for the two in-air carnivore hearing groups on
the exposure page.

The **source** stage, in
[Underwater acoustics: radiated noise and pile driving](underwater-acoustics.md),
sets up the ISO 18405 terminology (SPL, SEL and peak levels and their
references) and applies it to two regulated measurement cases: ships, with
the radiated noise level of ISO 17208-1 and the equivalent monopole source
level of ISO 17208-2 via the Lloyd's-mirror surface correction, and
percussive pile driving, with the single-strike, peak and cumulative sound
exposure of ISO 18406.

The **path** stage spans two pages.
[Underwater sound propagation](underwater-propagation.md)
predicts what the sea does to that sound in closed form: geometrical
spreading plus volume absorption (Francois-Garrison, Ainslie-McColm or
Thorp), Weston's four shallow-water regimes with their transition ranges, the
speed of sound in sea water by four formulations, the passive and active sonar
equation with its detection-range inversion, Rayleigh seabed reflection loss,
and the Wenz ambient-noise spectrum with JOMOPANS-ECHO ship traffic. When refraction and
boundaries decide the answer,
[Underwater propagation solvers](underwater-solvers.md)
computes the field instead: the normal-mode expansion, ray tracing, the
Gaussian beam sum built on those same rays and the split-step Fourier
parabolic equation, with the guidance for choosing between them and the
closed forms.

A **receiver** stage closes the loop.
[Marine-mammal noise exposure](marine-mammal-exposure.md)
takes the level a source and a path produce and asks what it does to the
animals that hear it: the group audiograms of Southall et al., the regulatory
auditory weighting functions of the NMFS guidance, the TTS and injury onset
criteria, and the weighted cumulative exposure of a piling campaign measured
against them.

Read the pages in that order: the reference levels come first because every
propagation result is expressed in them, and the exposure criteria come last
because they consume both. Unusually for this site, the theory for these pages
lives inline with the guides rather than in the theory reference.

## Pages in this section

- [Underwater acoustics: radiated noise and pile driving](underwater-acoustics.md):
  ISO 18405 reference levels, ISO 17208 ship radiated noise and monopole
  source level, and ISO 18406 pile-driving sound exposure.
- [Underwater sound propagation](underwater-propagation.md):
  propagation loss, sound speed, the sonar equation, seabed reflection and
  ocean ambient noise, in closed form.
- [Underwater propagation solvers](underwater-solvers.md):
  the normal-mode, ray-tracing, Gaussian-beam and parabolic-equation solvers
  of the stratified waveguide, each validated against an exact closed form,
  and how to choose a propagation model.
- [Marine-mammal noise exposure](marine-mammal-exposure.md):
  group audiograms, the regulatory auditory weighting functions with the
  guidance version selectable, the TTS and injury onset criteria, and a worked
  pile-driving assessment.

## What this section does not cover

**The measurement discipline is not implemented, only its arithmetic.** ISO
17208-1's four-run, three-hydrophone averaging, its closest-point-of-approach
and water-depth geometry checks, its ±30° data-window scoring and its
background-noise correction are the operator's; the library supplies the
closed-form radiated-noise and monopole source levels that follow. ISO 18406
itself excludes vibro- and sheet-piling from its scope, so continuous
pile-driving noise has no closed form here or anywhere in the library.

**The seabed is thin.** The closed-form page models it as a lossless
fluid-fluid Rayleigh reflection, so sediment attenuation is out of scope, and
all four solvers assume a **range-independent** water column with no
elastic bottom and no real bathymetry — which rules out
range-dependent problems entirely; the beam solver can bounce off that same
Rayleigh seabed instead of a perfect reflector, and that is as much bottom as
any of them has. The ray solver still returns paths and
travel times and no level of its own; `eigenrays` searches its fan for the
paths that join a source and a receiver and lists their arrivals with
classical per-path amplitudes, and the field comes from `gaussian_beams`,
which widens each of those rays into a beam and so answers a finite number at
a caustic and a graded one in a shadow zone, at two prices: it has no near
field within about three beam widths of the source, and the geometric-beam
variant with its KMAH index and its
minimum-width floor is not implemented. The parabolic equation is the standard
small-angle Tappert form rather than a wide-angle Padé variant. For the
elastic seabed physics these fluid solvers leave out, the [elastic wave
solver](../simulation/elastic-waves.md) is the nearest thing the
library has.

**The exposure page rates hearing, not behaviour.** Only the auditory-effect
criteria are implemented; behavioural-disturbance thresholds, the ones a
harassment take estimate turns on, are out of scope. Nothing chooses a hearing
group or an accumulation period for you, and nothing models the animal moving
relative to the source, so the cumulative exposure reported is the
stationary-receiver worst case. There is no audiogram for low-frequency
cetaceans, because the source publication does not print one of its
parameters.

Two smaller boundaries: the ambient-noise spectrum leaves out the
low-frequency turbulence band and has no built-in distant-shipping model —
supply a shipping spectrum yourself — and the active sonar equation is
monostatic only.

## Before and after these pages

Every level here is a level re 1 µPa computed from a hydrophone record, so the
calibration, weighting and spectral estimation behind it are in [Signal
analysis](../signals/index.md), and [Build a sound level
meter](../signals/sound-level-meter.md) runs that chain end to end on
one runnable page, in air but with the same functions. The underwater theory is
deliberately not in the theory reference: it lives inline with the four guides
above, where the quantity system of ISO 18405 is introduced with them.

If you arrived here from a search and want the shape of the whole library,
[What do you need to measure?](https://jmrplens.github.io/phonometry/start/tasks/) indexes it by the job
and [All guides](../README.md) lists every page with a line on
each.
