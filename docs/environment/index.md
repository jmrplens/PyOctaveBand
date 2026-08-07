← [Documentation index](../README.md)

# Environment and transport

Environmental noise is a source-path-receiver problem stretched over hundreds
of metres of open air. This section covers all three of them. The **propagation**
pages handle the path: ISO 9613 predicts, band by band, how much level survives
divergence, air absorption, the ground and any barrier on the way to a receiver,
and the wave-acoustic ground and refraction models say when that engineering
method stops being enough.

The **assessment** pages handle what happens once the sound has arrived: the
ISO 1996 rating level and the day-evening-night indicators, their Spanish
application in RD 1367/2007, and the NT ACOU 112 adjustment that quantifies when
impulsive character makes a received sound more annoying than its LAeq suggests.

The **source** pages handle the other end: what emits, described the way an
environmental model wants it. CNOSSOS-EU gives road traffic and railways a
source power per band and per category, and IEC 61400-11 rates a wind turbine
by its apparent sound power and its tonal audibility. What unites them is the
pattern: a carefully standardised source descriptor that the path model above
then attenuates.

This section leans on the core toolkit, but only up to the period level.
[Integrated and Statistical Levels](../signals/levels/levels.md) supplies
the LAeq, percentile and event levels of each reference period; what turns those
period levels into Lden, Ldn and the rating level, with the tonal adjustment,
the residual-noise correction and the uncertainty budget on top, is
[Environmental Levels (ISO 1996-1/-2)](assessment/environmental-levels.md),
in this section. The atmospheric absorption that every propagation model
consumes is shared with the room and materials pages. Start with
[Outdoor Sound Propagation](propagation/outdoor-propagation.md); it
introduces the source-path-receiver bookkeeping the transport pages reuse.

## [Assessment and regulation](assessment/index.md)

What the received sound is rated against, once it has arrived.

- [Environmental Levels (ISO 1996-1/-2)](assessment/environmental-levels.md):
  the day-evening-night indicators, the rating level and the adjustments that
  turn a measured LAeq into an assessed one.
- [Spanish Noise Regulation (RD 1367/2007)](assessment/spanish-noise-regulation.md):
  the national application of that chain, with its own limits and its own
  tonal and impulsive corrections.
- [Impulsive-sound prominence (NT ACOU 112)](assessment/impulsive-sound.md):
  the predicted prominence of impulsive sounds and the graduated adjustment
  added to LAeq.

## [Outdoor sound](propagation/index.md)

The path from an outdoor source to a receiver, and the character of what
arrives.

- [Outdoor Sound Propagation](propagation/outdoor-propagation.md):
  atmospheric absorption (ISO 9613-1) and the ISO 9613-2 general method with
  its per-term attenuation breakdown.
- [Ground effect and barriers](propagation/ground-barriers.md):
  the ground attenuation of ISO 9613-2 and the insertion loss a barrier adds
  to the path.
- [Atmospheric refraction](propagation/atmospheric-refraction.md):
  how wind and temperature gradients bend a ray into or out of a shadow zone.

## [Environmental sources](sources/index.md)

What emits, described the way an environmental model wants it: a source
strength per band, ready for the path above to attenuate.

- [CNOSSOS-EU road traffic source emission](sources/cnossos-road-emission.md):
  the rolling and propulsion power of a traffic stream, per vehicle category
  and per band.
- [CNOSSOS-EU railway source emission](sources/cnossos-rail-emission.md):
  the equivalent for rail, with its source heights and its rolling, traction
  and aerodynamic contributions.
- [Wind-turbine noise: sound power and tonal audibility](sources/wind-turbine-noise.md):
  the IEC 61400-11 apparent sound power level and tonal-audibility chain.

Aircraft are the other transport source with internationally fixed metrics,
and they have a topic of their own:
[Aircraft noise](../aircraft/index.md).
