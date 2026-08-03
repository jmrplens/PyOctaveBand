---
title: "Underwater acoustics"
description: "Sound in the sea: the ISO 18405 reference levels, ship radiated noise (ISO 17208) and pile-driving exposure (ISO 18406), propagation from closed-form transmission loss and the sonar equation to ambient noise and full numerical solvers, and the marine-mammal hearing criteria that turn a level into an assessment."
---

Underwater acoustics runs on the same physics as airborne acoustics but on a
different scale and a different reference: levels are expressed re **1 µPa**
(not 20 µPa), exposure re 1 µPa²·s, and the medium itself, with its
depth-dependent sound speed, refracts sound into channels that carry it for
kilometres. This section covers the discipline along the source-path-receiver
chain of the rest of the library.

The **source** half, in
[Underwater acoustics: radiated noise and pile driving](/phonometry/underwater/underwater-acoustics/),
sets up the ISO 18405 terminology (SPL, SEL and peak levels and their
references) and applies it to two regulated measurement cases: ships, with
the radiated noise level of ISO 17208-1 and the equivalent monopole source
level of ISO 17208-2 via the Lloyd's-mirror surface correction, and
percussive pile driving, with the single-strike, peak and cumulative sound
exposure of ISO 18406.

The **path** half now spans two pages.
[Underwater sound propagation](/phonometry/underwater/underwater-propagation/)
predicts what the sea does to that sound in closed form: geometrical
spreading plus volume absorption (Francois-Garrison, Ainslie-McColm or
Thorp), Weston's four shallow-water regimes with their transition ranges, the
speed of sound in sea water by four formulations, the passive and active sonar
equation with its detection-range inversion, Rayleigh seabed reflection loss,
and the Wenz ambient-noise spectrum with JOMOPANS-ECHO ship traffic. When refraction and
boundaries decide the answer,
[Underwater propagation solvers](/phonometry/underwater/underwater-solvers/)
computes the field instead: the normal-mode expansion, ray tracing and the
split-step Fourier parabolic equation, with the guidance for choosing
between them and the closed forms.

A **receiver** half closes the loop.
[Marine-mammal noise exposure](/phonometry/underwater/marine-mammal-exposure/)
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

- [Underwater acoustics: radiated noise and pile driving](/phonometry/underwater/underwater-acoustics/):
  ISO 18405 reference levels, ISO 17208 ship radiated noise and monopole
  source level, and ISO 18406 pile-driving sound exposure.
- [Underwater sound propagation](/phonometry/underwater/underwater-propagation/):
  transmission loss, sound speed, the sonar equation, seabed reflection and
  ocean ambient noise, in closed form.
- [Underwater propagation solvers](/phonometry/underwater/underwater-solvers/):
  the normal-mode, ray-tracing and parabolic-equation solvers of the
  stratified waveguide, each validated against an exact closed form, and
  how to choose a propagation model.
- [Marine-mammal noise exposure](/phonometry/underwater/marine-mammal-exposure/):
  group audiograms, the regulatory auditory weighting functions with the
  guidance version selectable, the TTS and injury onset criteria, and a worked
  pile-driving assessment.
