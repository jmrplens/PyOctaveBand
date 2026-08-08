← [Documentation index](../../README.md)

# Surfaces measured in place

A reverberation-room or impedance-tube coefficient describes a *sample*. Some
surfaces have no sample. A pavement cannot be cut out and carried to a
laboratory without destroying the very thing that governs its absorption — the
connected pore structure of the laid and compacted layer — and a core taken from
it is no longer the surface a tyre rolls on. In-situ methods answer the question
where the surface is, and they pay for it with a geometry problem: the
microphone hears the direct sound and the surface reflection together, so the
method is built around separating them in **time** rather than in space.

[In-situ road-surface absorption](road-absorption.md)
implements both parts of ISO 13472 and, more usefully, states which one a given
pavement allows. The **subtraction technique** of Part 1 puts a source and a
microphone above the surface, subtracts a free-field reference measurement and
applies the Adrienne window to keep the reflection and discard everything after
it. It handles the full range from reflective to highly absorbing pavements,
covers 250 Hz to 4 kHz, and averages over a patch metres across — a 5 ms window
gives a maximum sampled-area radius of about 1.34 m, roughly 5.6 m² of road, so
it sees texture and joints rather than one spot. The **spot method** of Part 2
seals a short portable tube onto the pavement and reads it with the two-microphone
transfer-function routine. It needs only a flat, sealable patch and minutes per
point, so it can sit in a wheel track or on a narrow strip, but it is scoped to
reflective surfaces, is declared unreliable once the measured absorption exceeds
0.15, and stops at 1600 Hz — which matters, because the tyre-road noise the
measurement usually serves peaks around 1 kHz and has content beyond that
ceiling.

They are complements, not competitors: Part 2's own introduction expects the two
to agree between 315 Hz and 1600 Hz, and both report the same quantity, the
normal-incidence absorption coefficient in one-third-octave bands. A
low-absorption lane can therefore be surveyed with the tube and anchored with a
subtraction measurement at a few positions. That number is what a low-noise
pavement specification is written against, and what the ground term of an
outdoor propagation model consumes.

## Pages in this section

- [In-situ road-surface absorption](road-absorption.md):
  the ISO 13472-1 subtraction technique with the Adrienne window and its
  geometry and validity helpers, the ISO 13472-2 spot tube with its
  applicability limits, and the comparison that decides between them.

## See also

Pages elsewhere on the site that this section leans on:

- [Impedance Tube](../absorbers/impedance-tube.md): the
  ISO 10534-2 two-microphone reduction the spot method reuses unchanged.
- [Sound Absorption Measurement and Rating](../absorbers/absorption-measurement.md):
  the laboratory route, for materials that can be brought indoors.
- [Environment and transport](../../environment/index.md): where a road's
  absorption is consumed, as the ground term of an outdoor prediction.

## What this section does not cover

**Edition status matters here more than anywhere else in this area.** The
implementation follows ISO 13472-1:2002 and ISO 13472-2:2010; both have since
been revised — 2022 and 2025 respectively — and those revisions are **not**
implemented, so a report that cites the current edition cannot cite these
functions without qualification. The spot method's own signal processing is not
duplicated either: only its geometry, validity and correction helpers live here,
and the two-microphone transfer-function reduction is the ISO 10534-2 routine of
[Impedance Tube](../absorbers/impedance-tube.md). Nothing in
this subsection measures the noise a surface *generates* — the tyre-road source
term is CNOSSOS territory, in [Environmental
sources](../../environment/sources/index.md) — and no in-situ method is provided
for any surface other than a road: a wall or a ceiling measured in place is
outside both parts of ISO 13472.
