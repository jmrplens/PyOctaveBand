← [Documentation index](../../README.md)

# Human vibration

Vibration transmitted to a person is evaluated with a measurement chain
deliberately parallel to a sound level meter's: the acceleration is
**frequency-weighted** to reflect how the body responds at each frequency,
reduced to a **weighted r.m.s.** value or a dose, combined across axes, and
normalised to a **daily exposure** that regulation can act on. The two pages
of this section cover the general chain and the special case that breaks it.

[Human Vibration](human-vibration.md) is the general chain.
It covers the whole-body and hand-arm frequency weightings of **ISO 8041-1**,
the weighted r.m.s. acceleration and the running and dose measures of
**ISO 2631-1** (MTVV, VDV, MSDV, crest factor) that flag shocks a plain
r.m.s. would hide, the direction-independent `Wm` weighting that **ISO 2631-2**
prescribes for building occupants on every axis, the hand-arm
vibration total value and daily exposure A(8) of **ISO 5349-1/-2**, and the
exposure action and limit values of **Directive 2002/44/EC** that make A(8)
legally meaningful.

[Multiple-shock whole-body vibration (ISO 2631-5)](multiple-shock-vibration.md)
handles exposures the r.m.s. philosophy underestimates: repeated mechanical
shocks from off-road vehicles, high-speed craft or earth-moving machinery,
which load the lumbar spine far beyond what their energy average suggests.
**ISO 2631-5:2018** models the seat-to-spine transfer explicitly, accumulates
a dose from the response peaks, and converts it into compressive stress on the
vertebral endplates and a probability of lumbar injury over a working life.

Use the ISO 2631-1 metrics first, and let two numbers decide when to move on. A
**crest factor above 9** says the basic weighted r.m.s. method is no longer
adequate for that record, which is the ISO 2631-1 trigger for reaching for the
running and dose measures. A **band-limited vertical peak acceleration above
9.81 m/s²** — 1 g, the free-fall threshold — puts the exposure in ISO 2631-5's
clause 5 regime, the severe shocks with possible loss of contact with the seat
that this library implements, rather than in its Annex A finite-element model
for exposures in which the occupant stays seated. The measurement front-end
(weighting filters, band analysis) is shared with the [core signal
analysis](../../signals/index.md) section.

## Pages in this section

- [Human Vibration](human-vibration.md): ISO 8041-1
  weightings, ISO 2631-1 r.m.s. and dose measures, ISO 5349 daily exposure
  A(8) and the Directive 2002/44/EC values.
- [Multiple-shock whole-body vibration (ISO 2631-5)](multiple-shock-vibration.md):
  the spinal-response model, acceleration dose and probability of lumbar
  injury.

## What this section does not cover

**No meter is type-tested.** ISO 8041-1's own subject — the design and
type-testing of general-purpose human-vibration meters — is not implemented;
only its frequency-weighting definitions are taken from it, so nothing here
assigns a class to an instrument.

**Building vibration stops at the weighting.** Of ISO 2631-2 the library
implements the direction-independent `Wm` curve and nothing else, and that is
closer to the standard than it looks: the 2003 edition deliberately deleted its
predecessor's guidance values, so there are no acceptable magnitudes for
building vibration to compare against. A reader looking for a limit will not
find one here, and will not find one in the standard either.

**Of ISO 2631-5, the clause 5 model only.** The Annex A finite-element model
for less severe seated exposures is distributed separately by ISO and is not
implemented, which is what makes the 1 g delineation above a routing decision
rather than a preference.

And no exposure verdict is issued. The action and limit values of Directive
2002/44/EC are stated so that an A(8) can be read against them, but the
library applies no national implementation of the directive, and a
risk-assessment conclusion is not a number this section produces.
