---
title: "Hearing and exposure"
description: "The statistics of hearing and the noise that damages it: the ISO 7029 age-related threshold distribution and ISO 389-7 reference zero, the ISO 1999 noise-induced permanent threshold shift, and the ISO 9612 occupational-exposure survey."
---

This section connects three quantities that regulation treats as one story:
where a population's **hearing threshold** sits, how occupational **noise
exposure** is measured, and how that exposure **shifts the threshold**
permanently over a working life.

[Hearing threshold (age and reference zero)](/phonometry/perception/hearing/hearing-threshold/)
establishes the baseline. **ISO 7029:2017** gives the statistical distribution
of hearing threshold with age for an otologically normal population: the slow,
high-frequency-first loss of presbycusis, resolved by age, sex and population
fractile. **ISO 389-7:2005** fixes the other end of the scale: the reference
threshold of hearing, the physical sound pressure level that audiometric
0 dB HL corresponds to under free-field and diffuse-field listening.

[Occupational Noise Exposure (ISO 9612)](/phonometry/perception/hearing/occupational-exposure/)
measures the cause. A regulated daily exposure level LEX,8h is assembled from
samples of a real working day by one of three strategies (task-based,
job-based or full-day), and reported with the normative Annex C uncertainty
budget and its one-sided 95 % upper limit, which is what an occupational
hygienist actually compares against action values.

[Noise-induced hearing loss (ISO 1999)](/phonometry/perception/hearing/noise-induced-hearing-loss/)
predicts the effect. From the exposure level and duration it gives the
noise-induced permanent threshold shift (NIPTS) with its population spread,
concentrated at the characteristic 4 kHz notch, and combines it with the
ISO 7029 age component into the hearing threshold level associated with age
and noise (HTLAN): the quantity used to estimate hearing handicap and
compensation risk in an exposed population.

Read the threshold page first, then ISO 9612, then ISO 1999: baseline,
exposure, damage. That way the pipeline is explicit: ISO 9612 produces the
LEX,8h that ISO 1999 consumes, and ISO 1999 builds on the ISO 7029
statistics. The perceptual consequences of a shifted
threshold, such as reduced speech intelligibility, are picked up by the SII in
the [Speech section](/phonometry/perception/speech/).

**Three pages, three different decibels**, and keeping them apart is most of the
work. A **hearing threshold level** is in dB HL, measured relative to the
audiometric zero, so 0 dB HL is a *different* sound pressure at every frequency
— exactly what ISO 389-7 tabulates. A **daily exposure level** is in
A-weighted decibels normalised to eight hours: an energy dose of the sound
outside the ear, with no listener in it. A **threshold shift** is a difference
of two dB HL values, so it may be added to a hearing level and never to a sound
pressure level. The chain between them runs one way only: ISO 9612 delivers a
single A-weighted LEX,8h into the ISO 1999 formulae, which return dB HL. The
only bridge back from hearing level to physical sound pressure is the ISO 389-7
reference threshold on the threshold page — which is also what the SII needs
when a raised threshold is used as an input.

## Pages in this section

In the order the chain runs.

- **Baseline** — [Hearing threshold (age and reference zero)](/phonometry/perception/hearing/hearing-threshold/):
  the ISO 7029:2017 age-related threshold distribution and the ISO 389-7:2005
  reference threshold of hearing.
- **Exposure** — [Occupational Noise Exposure (ISO 9612)](/phonometry/perception/hearing/occupational-exposure/):
  the three measurement strategies for LEX,8h with the Annex C uncertainty
  budget.
- **Damage** — [Noise-induced hearing loss (ISO 1999)](/phonometry/perception/hearing/noise-induced-hearing-loss/):
  NIPTS and its population distribution, and the combination with age into
  HTLAN.

## What this section does not cover

**Nothing here is a verdict about a person.** ISO 1999 does not define a hearing
handicap or a compensable fence — that line is set by national regulation, and
the library applies none of it, so you supply and check the criterion yourself.
The same is true of the exposure action values: the LEX,8h and its one-sided
95 % upper limit come out of ISO 9612, and the numbers they are compared against
are in your jurisdiction's directive, not here.

Two implementation boundaries follow the standards. Only **database A** is
implemented for ISO 1999: `htlan` always draws its age component from
ISO 7029:2017, and substituting a nationally measured control population
(clauses 6.2.3 and 6.2.4) means computing that database elsewhere and passing it
in. Of ISO 389-7, only the Table 1 reference values are implemented, not the
procedures by which they were established.

And no audiometry happens here. Nothing generates a test tone, drives an
audiometer or corrects for an earphone coupler: the pages consume and produce
threshold levels as data.
