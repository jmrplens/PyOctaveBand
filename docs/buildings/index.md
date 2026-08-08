← [Documentation index](../README.md)

# Rooms and buildings

This section covers sound in the built environment, and it splits along a
natural line: what happens **inside** one room, and what passes **between**
rooms. Inside a room, the governing quantity is absorption: it sets the
reverberation time and the clarity that room acoustics measures (ISO 3382)
and predicts (Sabine and its refinements, EN 12354-6), while the background
noise the room settles into is rated against criterion curves
(ANSI/ASA S12.2). Between rooms, the governing quantity is insulation: how much
airborne and impact sound a partition and its flanking paths transmit, measured
in the field (ISO 16283) or the laboratory (ISO 10140) and predicted from
element data (EN 12354).

Both halves consume coefficients measured elsewhere: the
[Materials and surfaces](../materials/index.md)
section characterises the absorption, impedance and scattering data that the
room and insulation predictions here rely on.

Start with
[Measuring the Room Impulse Response](rooms/room-impulse-response.md)
and [Room Acoustics](rooms/room-acoustics.md): the impulse response
the first acquires and the parameters the second derives are the vocabulary
the rest of the section speaks. If your interest is insulation, read
[Field Insulation Measurement (ISO 16283)](insulation/insulation-field.md)
next, and note that impact sources other than the tapping machine have their own
page, since ISO 16283-2 is the clause a field engineer usually arrives with; if
it is design-stage prediction, go to
[Reverberation-time prediction (Sabine, Eyring, Arau)](rooms/reverberation-prediction.md)
and [Predicting Sound Insulation (EN 12354)](design/insulation-prediction.md).

## [Room acoustics](rooms/index.md)

The sound field inside a single room: measured from an impulse response, rated
against criterion curves, and predicted from volume and absorption.

- [Measuring the Room Impulse Response](rooms/room-impulse-response.md):
  the ISO 18233 deterministic acquisition, sweep deconvolution and MLS.
- [Room Acoustics](rooms/room-acoustics.md): the ISO 3382-1/2 room
  parameters derived from that impulse response.
- [Open-Plan Office Acoustics (ISO 3382-3)](rooms/open-plan-acoustics.md):
  the speech-privacy quantities of an open-plan floor.
- [Image sources and the steady-state room field](rooms/room-image-sources.md):
  the deterministic image-source impulse response and the statistical
  steady-state level.
- [Room-noise criteria (NC / RC Mark II)](rooms/room-noise.md): the
  ANSI/ASA S12.2-2019 room-noise ratings with their spectral tags.
- [Reverberation-time prediction (Sabine, Eyring, Arau)](rooms/reverberation-prediction.md):
  Sabine, Eyring, Millington-Sette, Fitzroy and Arau-Puchades models with the
  air-absorption term.
- [Sound absorption in enclosed spaces (EN 12354-6)](rooms/enclosed-space-absorption.md):
  the normative prediction of a room's total equivalent absorption area and
  reverberation time.

## [Sound insulation](insulation/index.md)

Airborne and impact insulation: measured in the building, characterised in the
laboratory, and predicted from element data.

- [Field Insulation Measurement (ISO 16283)](insulation/insulation-field.md):
  ISO 16283-1/2 field measurement, the Clause 14 report and the ISO 12999-1
  uncertainty that qualifies it.
- [Laboratory Insulation Measurement](insulation/insulation-lab.md):
  the ISO 10140 laboratory characterisation of an element.
- [Sound Insulation by Intensity (ISO 15186)](insulation/insulation-intensity.md):
  the direct-power route to the same indices when flanking is high.
- [Sound Insulation Survey Method (ISO 10052)](insulation/insulation-survey.md):
  the octave-band control method and its reverberation index.
- [Heavy and Soft Impact Sources (ISO 16283-2)](insulation/heavy-impact-sources.md):
  the rubber ball and the bang machine, the impact force exposure level that
  specifies them and the ISO 717-2 Annex D single number.
- [Laboratory Flanking Transmission (ISO 10848)](insulation/flanking-lab.md):
  the measured junction vibration reduction index and the flanking descriptors.
- [Insulation Ratings (ISO 717)](insulation/insulation-ratings.md): the
  reference-curve engines behind Rw, DnT,w, Ln,w and their adaptation terms.
- [Façade Sound Insulation](insulation/facade-insulation.md): the
  building envelope measured (ISO 16283-3) and predicted (EN 12354-3/4).
- [Spanish Building Code (CTE DB-HR)](insulation/spanish-building-code.md):
  the DB-HR global indices, the clause 2 requirements and the window-size
  correction.

## [Insulation design](design/index.md)

The same quantities before the building exists: predicted from element data,
and from the physics of the element itself.

- [Predicting Sound Insulation (EN 12354)](design/insulation-prediction.md):
  airborne and impact flanking transmission between rooms (EN 12354-1/2).
- [Detailed Per-Band Prediction (ISO 12354)](design/detailed-prediction.md):
  the same prediction band by band, with the per-path contributions behind
  R'w and L'n,w.
- [Predicting Panel Sound Insulation](design/panel-sound-insulation.md):
  the mass law, coincidence dip, double walls and apertures.
- [Floor-Covering Impact Improvement (ISO 16251-1)](design/impact-improvement.md):
  the weighted improvement of a soft covering on a small heavyweight mock-up.
- [Predicting Resilient-Layer Performance](design/resilient-layers.md):
  the prediction side of coverings, floating floors and wall linings.
- [Structure-borne sound power of equipment (EN 15657)](design/structure-borne-power.md):
  the reception-plate method and the plate-independent source quantities.
- [Installed structure-borne sound (EN 12354-5)](design/installed-structure-borne.md):
  the receiving-room sound pressure level predicted from source and receiver
  mobilities.
- [Dynamic stiffness of resilient materials (EN 29052-1)](../materials/resilient/dynamic-stiffness.md):
  the load-plate resonance measurement behind every floating-floor prediction.

## What this section does not cover

**The library starts after the microphone and stops before the geometry.** On
the measurement side, every function takes band levels already averaged over
positions and already corrected for background noise: the position counts, the
low-frequency procedures, the signal-to-background floors and the test-facility
qualifications of ISO 16283, ISO 10140 and ISO 3382 are the operator's job, and
nothing here checks that they were done. On the prediction side, the element
ratings, the junction indices and the covering improvements are inputs you
supply from measurement or from a standard's own annex; none is derived from a
drawing.

**Nothing here is a wave solver or a room model.** There is no geometry
importer, no material database, no ray tracer and no auralisation: the room
pages take dimensions, absorption coefficients and impulse responses and give
back parameters, and the image-source model is specular only. An actual
low-frequency field in a real shape is [wave
simulation](../simulation/index.md).

**And a prediction is not a verdict.** The single-number ratings and the
national indices are computed here, but the limit values they are judged
against are national — the Spanish code is implemented as a worked example of
one such framework, not as the rule everywhere — and the requirement always
comes from your regulation.

## Before and after these pages

Every quantity on these pages starts from band levels or from a filtered
impulse response, so the calibration, weighting and fractional-octave
filtering behind them are in [Signal analysis](../signals/index.md), and
[Build a sound level meter](../signals/sound-level-meter.md) runs that
chain end to end on one runnable page. The derivations sit in [Rooms and
buildings theory](../reference/theory/rooms-buildings.md), from the Schroeder integration
to the EN 12354 path sums.

If you arrived here from a search and want the shape of the whole library,
[What do you need to measure?](https://jmrplens.github.io/phonometry/start/tasks/) indexes it by the job
and [All guides](https://jmrplens.github.io/phonometry/start/guides/) lists every page with a line on
each.
