← [Documentation index](../../README.md)

# Environmental sources

A propagation model does not accept a machine; it accepts a **source
descriptor** with a fixed geometry. For traffic that means an incoherent source
line carrying a sound power per metre at a standardised height; for a turbine it
means an apparent sound power referred to an equivalent point source at the
rotor centre. The height, the band range and the directivity are part of the
definition, not details of the measurement, which is why an emission method is
a standard in its own right and not a preliminary step. What every page here
produces is that descriptor, in the form
[Outdoor Sound Propagation](../propagation/outdoor-propagation.md)
consumes.

[CNOSSOS-EU road traffic source emission](cnossos-road-emission.md)
implements section 2.2 of Annex II to Directive 2002/49/EC in its consolidated
text: Directive (EU) 2015/996 as corrected by the OJ L 5 corrigendum of 2018,
which restores the 63 Hz to 8 kHz octave range the original clause contradicted,
and amended by Delegated Directive (EU) 2021/1226, which replaces Tables F-1 and
F-4 outright and makes the current source some 2,5 to 3,5 dB(A) louder than the
2015 one — so any comparison with pre-2021 literature carries that offset. Each
vehicle is a point source 0,05 m above the pavement, with the first pavement
reflection already inside its power. Per category (light, medium heavy, heavy,
mopeds, motorcycles) a rolling and a propulsion term are energy-summed,
corrected for pavement, air temperature, studded tyres and gradient, adjusted
near junctions, and turned into a directional power per metre of source line.

[CNOSSOS-EU railway source emission](cnossos-rail-emission.md)
implements section 2.3 on the same pattern, but with **two** equivalent source
lines, at 0,5 m and at 4,0 m above the rail head, because the physical sources
radiate from different heights. It starts one step further back than the road
method: from wheel and rail roughness spectra, passed through the contact filter
and the vehicle and track transfer functions, with the wavelength-to-frequency
conversion at the train speed that makes rail arithmetic different from road
arithmetic. Impact noise at joints and switches, curve squeal, traction,
aerodynamic noise above 200 km/h and a bridge term are each allocated to the
height they radiate from.

[Wind-turbine noise: sound power and tonal audibility](wind-turbine-noise.md)
is IEC 61400-11, where the descriptor is **measured** rather than tabulated.
With the microphone on a ground board at the horizontal distance R0 = H + D/2,
the apparent sound power per band follows from the measured pressure level and
the slant distance to the rotor centre, the −6 dB in the formula accounting for
the pressure doubling on the board; results are binned by standardised wind
speed. The same page carries the tonal audibility that decides whether a
blade-passing, gearbox or generator tone stands above its masking noise, and
ends in a `.report()` assessment fiche.

Read the road page first even for a railway job: it introduces the source-line
bookkeeping and the Annex II layering that the rail page reuses. The turbine
page is independent of both.

## Pages in this section

- [CNOSSOS-EU road traffic source emission](cnossos-road-emission.md):
  the rolling and propulsion sound power per vehicle category, its pavement,
  temperature, studded-tyre, gradient and junction corrections, and the
  directional power per metre of source line.
- [CNOSSOS-EU railway source emission](cnossos-rail-emission.md):
  roughness and transfer functions to the two equivalent source lines at 0,5 m
  and 4,0 m, with the impact, squeal, traction, aerodynamic and bridge terms.
- [Wind-turbine noise: sound power and tonal audibility](wind-turbine-noise.md):
  the IEC 61400-11 apparent sound power referred to the rotor centre, its
  wind-speed binning and the tonal-audibility chain, with the assessment fiche.

## See also

Pages elsewhere on the site that this section leans on:

- [Outdoor Sound Propagation](../propagation/outdoor-propagation.md):
  the path model every descriptor here is built to feed.
- [Sound power and intensity](../../devices/emission/index.md): how a machine
  that is not a vehicle is characterised.

## What this section does not cover

Two of the four CNOSSOS sources are missing, by omission rather than
oversight: the **industrial source** of section 2.4 and Appendix H is not
implemented, and neither is the **aircraft source** of sections 2.6 and 2.7 —
aircraft noise is covered by the ICAO and ECAC methods in [Aircraft
noise](../../aircraft/index.md), which is a different family of models entirely.
Neither is the **CNOSSOS propagation method** of section 2.5: it differs from
the ISO 9613-2 model this library implements, so pairing these source powers
with [Outdoor Sound
Propagation](../propagation/outdoor-propagation.md) does
not give a CNOSSOS result. Inside the two methods that are here, three gaps
come from the source documents themselves: the open vehicle category 5 has no
coefficients in Appendix F and is not modelled, rail roughness classes N and B
carry no spectrum in Appendix G and must be supplied by the Member State, and
how a source line is split into point sources is declared out of scope by the
method. Depots, stations and loudspeakers are railway sources under 2.3.3 but
are treated by the industrial method, so they are not here either.
