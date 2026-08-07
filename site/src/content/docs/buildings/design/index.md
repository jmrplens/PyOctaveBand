---
title: "Insulation design"
description: "Sound insulation before it is built: the EN/ISO 12354 flanking prediction between rooms in its simplified and per-band forms, the theoretical insulation of a panel from its physical properties, the resilient layers a floor or lining design turns on, and the structure-borne sound of building service equipment from the EN 15657 reception plate to the EN 12354-5 receiving-room level."
---

The pages of the [Sound insulation](/phonometry/buildings/insulation/)
section answer the question *what does this building achieve?* The pages here
answer the one that comes first: *what will it achieve, and what should I
build?* Both halves speak the same language of $R$, $L_n$ and their weighted
single numbers, but a prediction is assembled from element data instead of
measured in a finished room, so its inputs, its assumptions and its error bars
are its own subject.

[Predicting Sound Insulation (EN 12354)](/phonometry/buildings/design/insulation-prediction/)
is the normative model: the airborne and impact flanking transmission between
two rooms, path by path, from the direct element and the junction vibration
reduction indices $K_{ij}$. It consumes laboratory element data measured per
ISO 10140 and junction data measured per ISO 10848, both of which live in
[Sound insulation](/phonometry/buildings/insulation/) beside the
field measurement the prediction is checked against, and beside the façade
guide that carries the same family across the building envelope.

[Detailed Per-Band Prediction (ISO 12354)](/phonometry/buildings/design/detailed-prediction/)
runs the same standard band by band instead of on single numbers: the
laboratory element and junction data are converted to their in-situ values,
every path is formed per band, and the result shows which path dominates each
band rather than only whether the room passes.

[Predicting Panel Sound Insulation](/phonometry/buildings/design/panel-sound-insulation/)
goes one level deeper, to where the element $R$ itself comes from: the mass law
and the coincidence dip of a single panel, the mass-spring-mass behaviour of a
double wall, transmission through slits and apertures, plate radiation
efficiency and point mobilities. It is the physics a catalogue value expresses
in one number.

Two pages here carry the floor half of any design, one measuring and one
predicting.
[Floor-Covering Impact Improvement (ISO 16251-1)](/phonometry/buildings/design/impact-improvement/)
gives the weighted improvement $\Delta L_w$ of a covering that exists, on a
small heavyweight mock-up, and that is the term EN 12354-2 subtracts from the
bare-floor level.
[Predicting Resilient-Layer Performance](/phonometry/buildings/design/resilient-layers/)
predicts it for a covering that does not yet exist, from the tapping machine's
own force spectrum, the cut-off frequency of a soft covering, the 30 lg and
40 lg floating-floor laws and the ISO 12354-1 Annex D rating of a wall lining.
Both start from the stiffness per unit area $s'$ of the resilient layer,
measured per EN 29052-1 in
[Dynamic stiffness of resilient materials](/phonometry/materials/resilient/dynamic-stiffness/)
over in the materials section, which sets the resonance the whole improvement
hangs on.

Building service equipment is a chain of its own, and the two pages only read
correctly in order.
[Structure-borne sound power of equipment (EN 15657)](/phonometry/buildings/design/structure-borne-power/)
characterises a pump, fan or cistern by the power it injects into the
structure, measured on a reception plate of known dissipation and then made
plate-independent.
[Installed structure-borne sound (EN 12354-5)](/phonometry/buildings/design/installed-structure-borne/)
takes that source description, loses part of it to the coupling term the source
and receiver mobilities set, and carries the rest to a room that may be several
junctions away.

One bookkeeping note runs through the whole section: the family exists as
EN 12354:2000 and as ISO 12354:2017, and the two are not interchangeable in
every clause. The simplified models on
[Predicting Sound Insulation](/phonometry/buildings/design/insulation-prediction/)
follow the 2000 text — including the tabulated flanking correction $K$ that the
2017 impact part replaced with explicit per-path formulae — while
[Detailed Per-Band Prediction](/phonometry/buildings/design/detailed-prediction/)
follows the 2017 text. Check which edition your regulation calls up before
quoting a correction from either.

## Pages in this section

- [Predicting Sound Insulation (EN 12354)](/phonometry/buildings/design/insulation-prediction/):
  the airborne and impact flanking models between rooms (EN 12354-1/2) with
  their junction vibration reduction indices and prediction fiches.
- [Detailed Per-Band Prediction (ISO 12354)](/phonometry/buildings/design/detailed-prediction/):
  the per-band detailed model of ISO 12354-1/-2 with in-situ element and
  junction conversion, the flanking indices per band and the per-path
  contributions behind the rating.
- [Predicting Panel Sound Insulation](/phonometry/buildings/design/panel-sound-insulation/):
  the mass law and coincidence dip (Sharp), double walls (Bies), slits and
  apertures (Gomperts, Wilson-Soroka), radiation efficiency
  (Leppington/Maidanik) and point mobilities (Cremer).
- [Floor-Covering Impact Improvement (ISO 16251-1)](/phonometry/buildings/design/impact-improvement/):
  the weighted improvement of a soft floor covering measured on a small
  heavyweight mock-up.
- [Predicting Resilient-Layer Performance](/phonometry/buildings/design/resilient-layers/):
  the tapping-machine force model, the cut-off frequency of a soft covering,
  the floating-floor improvement laws and the ISO 12354-1 Annex D rating of a
  wall lining.
- [Structure-borne sound power of equipment (EN 15657)](/phonometry/buildings/design/structure-borne-power/):
  the characteristic power a machine injects into a building element, measured
  on a reception plate.
- [Installed structure-borne sound (EN 12354-5)](/phonometry/buildings/design/installed-structure-borne/):
  what that power becomes once the machine is mounted on a real element, and
  the level it produces in the receiving room.

## See also

Pages elsewhere on the site that this section leans on:

- [Dynamic stiffness of resilient materials (EN 29052-1)](/phonometry/materials/resilient/dynamic-stiffness/):
  the load-plate resonance measurement, the enclosed-gas term and the
  floating-floor natural frequency.
