← [Documentation index](../../README.md)

# Insulation design

The pages of the [Sound insulation](../insulation/index.md)
section answer the question *what does this building achieve?* The pages here
answer the one that comes first: *what will it achieve, and what should I
build?* Both halves speak the same language of $R$, $L_n$ and their weighted
single numbers, but a prediction is assembled from element data instead of
measured in a finished room, so its inputs, its assumptions and its error bars
are its own subject.

[Predicting Sound Insulation (EN 12354)](insulation-prediction.md)
is the normative model: the airborne and impact flanking transmission between
two rooms, path by path, from the direct element and the junction vibration
reduction indices $K_{ij}$. It consumes laboratory element data measured per
ISO 10140 and junction data measured per ISO 10848, both of which live in
[Sound insulation](../insulation/index.md) beside the
field measurement the prediction is checked against, and beside the façade
guide that carries the same family across the building envelope.

[Detailed Per-Band Prediction (ISO 12354)](detailed-prediction.md)
runs the same standard band by band instead of on single numbers: the
laboratory element and junction data are converted to their in-situ values,
every path is formed per band, and the result shows which path dominates each
band rather than only whether the room passes.

[Predicting Panel Sound Insulation](panel-sound-insulation.md)
goes one level deeper, to where the element $R$ itself comes from: the mass law
and the coincidence dip of a single panel, the mass-spring-mass behaviour of a
double wall, transmission through slits and apertures, plate radiation
efficiency and point mobilities. It is the physics a catalogue value expresses
in one number.

Two measurements feed the floor half of any design.
[Floor-Covering Impact Improvement (ISO 16251-1)](impact-improvement.md)
gives the weighted improvement $\Delta L_w$ of a soft covering on a small
heavyweight mock-up, the term EN 12354-2 subtracts from the bare-floor level,
and
[Dynamic stiffness of resilient materials (EN 29052-1)](../../materials/resilient/dynamic-stiffness.md)
gives the stiffness per unit area $s'$ of the resilient layer under a floating
floor, and with it the resonance frequency the whole improvement hangs on.

## Pages in this section

- [Predicting Sound Insulation (EN 12354)](insulation-prediction.md):
  the airborne and impact flanking models between rooms (EN 12354-1/2) with
  their junction vibration reduction indices and prediction fiches.
- [Detailed Per-Band Prediction (ISO 12354)](detailed-prediction.md):
  the per-band detailed model of ISO 12354-1/-2 with in-situ element and
  junction conversion, the flanking indices per band and the per-path
  contributions behind the rating.
- [Predicting Panel Sound Insulation](panel-sound-insulation.md):
  the mass law and coincidence dip (Sharp), double walls (Bies), slits and
  apertures (Gomperts, Wilson-Soroka), radiation efficiency
  (Leppington/Maidanik) and point mobilities (Cremer).
- [Floor-Covering Impact Improvement (ISO 16251-1)](impact-improvement.md):
  the weighted improvement of a soft floor covering measured on a small
  heavyweight mock-up.
- [Predicting Resilient-Layer Performance](resilient-layers.md):
  the tapping-machine force model, the cut-off frequency of a soft covering,
  the floating-floor improvement laws and the ISO 12354-1 Annex D rating of a
  wall lining.
- [Dynamic stiffness of resilient materials (EN 29052-1)](../../materials/resilient/dynamic-stiffness.md):
  the load-plate resonance measurement, the enclosed-gas term and the
  floating-floor natural frequency.
- [Structure-borne sound power of equipment (EN 15657)](structure-borne-power.md):
  the characteristic power a machine injects into a building element, measured
  on a reception plate.
- [Installed structure-borne sound (EN 12354-5)](installed-structure-borne.md):
  what that power becomes once the machine is mounted on a real element, and
  the level it produces in the receiving room.
