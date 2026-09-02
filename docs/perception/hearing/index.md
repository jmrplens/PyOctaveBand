← [Documentation index](../../README.md)

# Hearing and exposure

This section connects three quantities that regulation treats as one story:
where a population's **hearing threshold** sits, how occupational **noise
exposure** is measured, and how that exposure **shifts the threshold**
permanently over a working life.

[Hearing threshold (age and reference zero)](hearing-threshold.md)
establishes the baseline. **ISO 7029:2017** gives the statistical distribution
of hearing threshold with age for an otologically normal population: the slow,
high-frequency-first loss of presbycusis, resolved by age, sex and population
fractile. **ISO 389-7:2005** fixes the other end of the scale: the reference
threshold of hearing, the physical sound pressure level that audiometric
0 dB HL corresponds to under free-field and diffuse-field listening.

[Occupational Noise Exposure (ISO 9612)](occupational-exposure.md)
measures the cause. A regulated daily exposure level LEX,8h is assembled from
samples of a real working day by one of three strategies (task-based,
job-based or full-day), and reported with the normative Annex C uncertainty
budget and its one-sided 95 % upper limit, which is what an occupational
hygienist actually compares against action values.

[Noise-induced hearing loss (ISO 1999)](noise-induced-hearing-loss.md)
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
the [Speech section](../speech/index.md).

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

- **Baseline** — [Hearing threshold (age and reference zero)](hearing-threshold.md):
  the ISO 7029:2017 age-related threshold distribution and the ISO 389-7:2005
  reference threshold of hearing.
- **Exposure** — [Occupational Noise Exposure (ISO 9612)](occupational-exposure.md):
  the three measurement strategies for LEX,8h with the Annex C uncertainty
  budget.
- **Damage** — [Noise-induced hearing loss (ISO 1999)](noise-induced-hearing-loss.md):
  NIPTS and its population distribution, and the combination with age into
  HTLAN.
- **Protection** — [Hearing Protectors (ISO 4869-2)](hearing-protectors.md):
  the octave-band, HML and SNR methods that say what a protector actually
  leaves at the ear, and the assumed protection value all three start from.

## What this section does not cover

**Nothing here is a verdict about a person.** ISO 1999 does not define a
hearing handicap or a compensable fence — that line is set by national
regulation, and the library applies none of it, so you supply and check the
criterion yourself. The exposure side stops one step later: the LEX,8h and its
one-sided 95 % upper limit come out of ISO 9612, and the fiche written by
`ExposureResult.report()` does compare the LEX,8h against the Directive
2003/10/EC action and limit values (80, 85 and 87 dB(A)) and issues the
PASS/FAIL verdict against the limit value — but that is a verdict on an
unprotected exposure, not on a person: the effective exposure behind hearing
protectors, and any stricter national transposition, are still yours to check.

Two implementation boundaries follow the standards. Only **database A** is
implemented for ISO 1999: `htlan` always draws its age component from ISO
7029:2017, and substituting a nationally measured control population (clauses
6.2.3 and 6.2.4) means computing that database elsewhere and passing it in. Of
ISO 389-7, only the Table 1 reference values are implemented, not the
procedures by which they were established.

And no audiometry happens here. Nothing generates a test tone, drives an
audiometer or corrects for an earphone coupler: the pages consume and produce
threshold levels as data.
