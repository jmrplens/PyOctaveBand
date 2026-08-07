---
title: "Aircraft noise"
description: "Aircraft noise with internationally fixed metrics: the ICAO Annex 16 EPNL certification chain, the ECAC Doc 29 airport contour machinery, the ECAC Doc 32 rotorcraft hemisphere method and the EASA ANP fleet database that feeds them."
---

Aircraft noise is computed under internationally negotiated methods of two
kinds. **Certification** fixes a single number per aircraft type to the last
decimal, at reference points a standard places around the runway. **Contour
methods** take that certified fleet and predict what an airport does to the
ground around it. The four pages of this section cover both, and they share a
common anatomy: a rigorously standardised **source descriptor** — a spectral
time history, a noise-power-distance table or a noise hemisphere — plus
standardised **propagation adjustments** that place the source at a receiver.

[Aircraft noise: Effective Perceived Noise Level](/phonometry/aircraft/aircraft-noise/)
covers fixed-wing certification. The **EPNL** of ICAO Annex 16 condenses a
one-third-octave time history of a flyover into a single EPNdB value through
perceived noisiness, a tone correction and a duration correction; the page
adds the IEC 61265 measurement-system verifier and the SAE ARP 5534
atmospheric absorption used in the certification chain.
[Airport Noise (ECAC Doc 29)](/phonometry/aircraft/airport-noise/) picks the
aeroplane up from there: the noise-power-distance tables, the per-segment
corrections of a flight path (impedance, lateral attenuation, engine
installation, duration, noise fraction and start-of-roll directivity) and the
single-event contour over a ground grid.

[Rotorcraft noise: the hemisphere method](/phonometry/aircraft/rotorcraft-noise/)
covers helicopters, whose strong directivity defeats a single-number source
level. ECAC Doc 32 instead describes the source as a **noise hemisphere**
(band levels on a grid of emission angles at a 60 m reference distance),
propagates each ray with spherical spreading, atmospheric absorption and the
Chien-Soroka ground effect, interpolates between the measured flight
conditions along the track, and integrates the received history into the
single-event SEL, LASmax and EPNL and their ground-grid contours.

[The ANP fleet database](/phonometry/aircraft/anp-fleet/) closes the loop on
the two above: the noise-power-distance tables and default trajectories EASA and
EUROCONTROL publish for real aircraft types, ready to feed the Doc 29 chain
without writing a table by hand.

The shared physics connects outward: atmospheric absorption comes from the
same ISO 9613-1 model as
[Outdoor Sound Propagation](/phonometry/environment/propagation/outdoor-propagation/), and the
same type-testing logic governs
[Wind-turbine noise](/phonometry/environment/sources/wind-turbine-noise/),
which is filed with the other environmental sources: its IEC 61400-11 apparent
sound power level and tonal-audibility chain answer the same question for a
source that is not an aircraft. That tonality test is in turn a cousin of the
methods in [Psychoacoustics](/phonometry/perception/psychoacoustics/).

Start from the question. To check an aeroplane against a certification limit,
or to understand where the published numbers for a type come from, start with
the EPNL page. To predict what a movement does at a street address, use the
Doc 29 page, with the ANP page supplying the aircraft data. For helicopters the
hemisphere page replaces both. Read the fixed-wing pages in that order: the EPNL
page defines the certified metric, the Doc 29 page turns certified aeroplanes
into ground contours from tables written by hand, and the ANP page replaces
those hand-written tables with the published fleet data. The rotorcraft page
stands on its own — a different standard and a different source model — and can
be read first if helicopters are what you came for.

The three metrics are not interchangeable. EPNL is a *certification* metric of
one aeroplane at one prescribed point; SEL and LASmax are *single-event*
assessment metrics at an arbitrary receiver; neither is the long-term index a
land-use study is finally judged on. And the boundary: this section does not
compute cumulative multi-event indices, does not synthesise NPD tables from
engine data, does not model hover, idle or taxi rotorcraft operations, and does
not touch sonic boom.

## Pages in this section

- [Aircraft noise: Effective Perceived Noise Level](/phonometry/aircraft/aircraft-noise/):
  the ICAO Annex 16 EPNL chain, the IEC 61265 verifier and the SAE ARP 5534
  absorption.
- [Airport Noise (ECAC Doc 29)](/phonometry/aircraft/airport-noise/): the NPD
  engine, the single-event segment chain and the ground-grid SEL contour.
- [Rotorcraft noise: the hemisphere method](/phonometry/aircraft/rotorcraft-noise/):
  the ECAC Doc 32 noise-hemisphere source model, its propagation adjustments
  and the single-event metrics and contours.
- [The ANP fleet database](/phonometry/aircraft/anp-fleet/): the EASA tables of
  noise-power-distance curves and default trajectories that run the Doc 29
  chain for a real aircraft type.
