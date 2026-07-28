---
title: "Underwater acoustics"
description: "Sound in the sea: the ISO 18405 reference levels, ship radiated noise (ISO 17208) and pile-driving exposure (ISO 18406), and propagation from closed-form transmission loss and the sonar equation to ambient noise and full numerical solvers."
---

Underwater acoustics runs on the same physics as airborne acoustics but on a
different scale and a different reference: levels are expressed re **1 µPa**
(not 20 µPa), exposure re 1 µPa²·s, and the medium itself, with its
depth-dependent sound speed, refracts sound into channels that carry it for
kilometres. This section covers the discipline in two halves that mirror the
source-path split of the rest of the library.

The **source** half, in
[Underwater acoustics: radiated noise and pile driving](/phonometry/guides/underwater-acoustics/),
sets up the ISO 18405 terminology (SPL, SEL and peak levels and their
references) and applies it to two regulated measurement cases: ships, with
the radiated noise level of ISO 17208-1 and the equivalent monopole source
level of ISO 17208-2 via the Lloyd's-mirror surface correction, and
percussive pile driving, with the single-strike, peak and cumulative sound
exposure of ISO 18406.

The **path** half now spans two pages.
[Underwater sound propagation](/phonometry/guides/underwater-propagation/)
predicts what the sea does to that sound in closed form: geometrical
spreading plus volume absorption (Francois-Garrison, Ainslie-McColm or
Thorp), the speed of sound in sea water by three formulations, the passive
and active sonar equation, Rayleigh seabed reflection loss, and the Wenz
ambient-noise spectrum with JOMOPANS-ECHO ship traffic. When refraction and
boundaries decide the answer,
[Underwater propagation solvers](/phonometry/guides/underwater-solvers/)
computes the field instead: the normal-mode expansion, ray tracing and the
split-step Fourier parabolic equation, with the guidance for choosing
between them and the closed forms.

Read the pages in that order: the reference levels come first because every
propagation result is expressed in them. Unusually for this site, the theory
for these pages lives inline with the guides rather than in the theory
reference.

## Pages in this section

- [Underwater acoustics: radiated noise and pile driving](/phonometry/guides/underwater-acoustics/):
  ISO 18405 reference levels, ISO 17208 ship radiated noise and monopole
  source level, and ISO 18406 pile-driving sound exposure.
- [Underwater sound propagation](/phonometry/guides/underwater-propagation/):
  transmission loss, sound speed, the sonar equation, seabed reflection and
  ocean ambient noise, in closed form.
- [Underwater propagation solvers](/phonometry/guides/underwater-solvers/):
  the normal-mode, ray-tracing and parabolic-equation solvers of the
  stratified waveguide, each validated against an exact closed form, and
  how to choose a propagation model.
