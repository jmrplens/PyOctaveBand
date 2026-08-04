← [Documentation index](../../README.md)

# Outdoor sound

Outdoor sound assessment has two halves: predicting the level a source
delivers to a distant receiver, and judging the character of the sound that
actually arrives. The pages of this section split along that line, with the
prediction half starting one step earlier, at the source itself.

[Outdoor Sound Propagation](outdoor-propagation.md) is the
prediction half. Starting from a source's **sound power**, the ISO 9613-2
general method subtracts, octave band by octave band, every mechanism that
attenuates sound on its way: geometrical divergence, atmospheric absorption
(supplied by the pure-tone coefficient of **ISO 9613-1**), the ground effect
and barrier screening, with a meteorological correction for long-term
averages. The page keeps the per-term breakdown visible, so a prediction is
never a black box: you can see exactly which mechanism buys how many decibels
at which frequency.

[CNOSSOS-EU railway source emission](../sources/cnossos-rail-emission.md)
is the source half for the railway. Section 2.3 of Annex II to Directive
2002/49/EC turns a roughness spectrum, a set of transfer functions and a train
flow into the two equivalent source lines every strategic noise map in the
European Union starts from, at 0,5 m and at 4,0 m above the rail head, with
impact noise, curve squeal, traction, aerodynamic noise above 200 km/h and the
bridge term each allocated to the height it radiates from.

[Spherical ground effect and advanced barriers](ground-barriers.md)
goes underneath the ISO 9613-2 fits to the wave acoustics they approximate: the
Weyl-Van der Pol spherical-wave reflection coefficient of a finite-impedance
ground, and barrier diffraction by the Kurze-Anderson Fresnel number, the exact
rigid half-plane, thick barriers and the coherent four-path barrier on the
ground, all resolving the frequency-dependent interference the octave-band terms
smooth away.

[CNOSSOS-EU road traffic source emission](../sources/cnossos-road-emission.md)
supplies the source power that a prediction starts from, for the one source
that dominates almost every noise map: road traffic. The common EU method of
Annex II to Directive 2002/49/EC builds a rolling and a propulsion sound power
for each vehicle category, corrects the rolling term for pavement, air
temperature and studded tyres and the propulsion term for pavement and road
gradient, applies the junction correction to both, and delivers a directional
sound power per metre of source line.

[Impulsive-sound prominence (NT ACOU 112)](../assessment/impulsive-sound.md)
is the assessment half. Noise containing distinct impulses (hammering,
riveting, pile driving) annoys more than a steady sound of the same LAeq, and
the Nordtest method quantifies that: from the onset rate and level difference
of each impulse it computes a predicted **prominence**, and converts it into
the graduated adjustment KI that is added to the measured LAeq in a rating
level.

The surrounding machinery lives nearby: the rating levels and Lden that
assessments end in are covered in
[Integrated and Statistical Levels](../../signal/levels/levels.md), the tonal
counterpart of the impulsive adjustment in
[Objective audibility of tones in noise](../../perception/psychoacoustics/tone-audibility.md),
and the sources that feed a propagation calculation in the
[Sound power and intensity](../../devices/emission/index.md) and
[Aircraft and wind energy](../../aircraft/index.md)
sections.

## Pages in this section

- [Outdoor Sound Propagation](outdoor-propagation.md):
  ISO 9613-1 atmospheric absorption and the ISO 9613-2 general method with a
  per-term octave-band attenuation breakdown.
- [CNOSSOS-EU railway source emission](../sources/cnossos-rail-emission.md):
  the common EU railway emission method, from rail and wheel roughness to the
  directional sound power per metre of the two equivalent source lines.
- [CNOSSOS-EU road traffic source emission](../sources/cnossos-road-emission.md):
  the road source of Annex II to Directive 2002/49/EC: rolling and propulsion
  sound power per vehicle category with the Appendix F database, and the
  directional sound power per metre of source line.
- [Spherical ground effect and advanced barriers](ground-barriers.md):
  the Weyl-Van der Pol spherical-wave ground reflection and wave-theoretic
  barrier diffraction (Kurze-Anderson, exact rigid half-plane, thick barriers
  and the coherent four-path barrier on the ground).
- [Impulsive-sound prominence (NT ACOU 112)](../assessment/impulsive-sound.md):
  the predicted prominence of impulsive sounds and the graduated LAeq
  adjustment KI.
