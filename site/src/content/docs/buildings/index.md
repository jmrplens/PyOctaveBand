---
title: "Rooms and buildings"
description: "Sound inside rooms and through buildings: room-acoustic measurement and prediction (ISO 3382, EN 12354-6), and sound insulation measured in the field and the laboratory and predicted with EN 12354."
---

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
[Materials and surfaces](/phonometry/materials/)
section characterises the absorption, impedance and scattering data that the
room and insulation predictions here rely on.

Start with
[Measuring the Room Impulse Response](/phonometry/buildings/rooms/room-impulse-response/)
and [Room Acoustics](/phonometry/buildings/rooms/room-acoustics/): the impulse response
the first acquires and the parameters the second derives are the vocabulary
the rest of the section speaks. If your interest is insulation, read
[Field Insulation Measurement (ISO 16283)](/phonometry/buildings/insulation/insulation-field/)
next, and note that impact sources other than the tapping machine have their own
page, since ISO 16283-2 is the clause a field engineer usually arrives with; if
it is design-stage prediction, go to
[Reverberation-time prediction (Sabine, Eyring, Arau)](/phonometry/buildings/rooms/reverberation-prediction/)
and [Predicting Sound Insulation (EN 12354)](/phonometry/buildings/design/insulation-prediction/).

## [Room acoustics](/phonometry/buildings/rooms/)

The sound field inside a single room: measured from an impulse response, rated
against criterion curves, and predicted from volume and absorption.

- [Measuring the Room Impulse Response](/phonometry/buildings/rooms/room-impulse-response/):
  the ISO 18233 deterministic acquisition, sweep deconvolution and MLS.
- [Room Acoustics](/phonometry/buildings/rooms/room-acoustics/): the ISO 3382-1/2 room
  parameters derived from that impulse response.
- [Open-Plan Office Acoustics (ISO 3382-3)](/phonometry/buildings/rooms/open-plan-acoustics/):
  the speech-privacy quantities of an open-plan floor.
- [Image sources and the steady-state room field](/phonometry/buildings/rooms/room-image-sources/):
  the deterministic image-source impulse response and the statistical
  steady-state level.
- [Room-noise criteria (NC / RC Mark II)](/phonometry/buildings/rooms/room-noise/): the
  ANSI/ASA S12.2-2019 room-noise ratings with their spectral tags.
- [Reverberation-time prediction (Sabine, Eyring, Arau)](/phonometry/buildings/rooms/reverberation-prediction/):
  Sabine, Eyring, Millington-Sette, Fitzroy and Arau-Puchades models with the
  air-absorption term.
- [Sound absorption in enclosed spaces (EN 12354-6)](/phonometry/buildings/rooms/enclosed-space-absorption/):
  the normative prediction of a room's total equivalent absorption area and
  reverberation time.

## [Sound insulation](/phonometry/buildings/insulation/)

Airborne and impact insulation: measured in the building, characterised in the
laboratory, and predicted from element data.

- [Field Insulation Measurement (ISO 16283)](/phonometry/buildings/insulation/insulation-field/):
  ISO 16283-1/2 field measurement, the Clause 14 report and the ISO 12999-1
  uncertainty that qualifies it.
- [Laboratory Insulation Measurement](/phonometry/buildings/insulation/insulation-lab/):
  the ISO 10140 laboratory characterisation of an element.
- [Sound Insulation by Intensity (ISO 15186)](/phonometry/buildings/insulation/insulation-intensity/):
  the direct-power route to the same indices when flanking is high.
- [Sound Insulation Survey Method (ISO 10052)](/phonometry/buildings/insulation/insulation-survey/):
  the octave-band control method and its reverberation index.
- [Heavy and Soft Impact Sources (ISO 16283-2)](/phonometry/buildings/insulation/heavy-impact-sources/):
  the rubber ball and the bang machine, the impact force exposure level that
  specifies them and the ISO 717-2 Annex D single number.
- [Laboratory Flanking Transmission (ISO 10848)](/phonometry/buildings/insulation/flanking-lab/):
  the measured junction vibration reduction index, the flanking descriptors,
  and the suspended-ceiling plenum path with its ceiling attenuation class.
- [Insulation Ratings (ISO 717)](/phonometry/buildings/insulation/insulation-ratings/): the
  reference-curve engines behind Rw, DnT,w, Ln,w and their adaptation terms.
- [Façade Sound Insulation](/phonometry/buildings/insulation/facade-insulation/): the
  building envelope measured (ISO 16283-3) and predicted (EN 12354-3/4).
- [Spanish Building Code (CTE DB-HR)](/phonometry/buildings/insulation/spanish-building-code/):
  the DB-HR global indices, the clause 2 requirements and the window-size
  correction.

## [Insulation design](/phonometry/buildings/design/)

The same quantities before the building exists: predicted from element data,
and from the physics of the element itself.

- [Predicting Sound Insulation (EN 12354)](/phonometry/buildings/design/insulation-prediction/):
  airborne and impact flanking transmission between rooms (EN 12354-1/2).
- [Detailed Per-Band Prediction (ISO 12354)](/phonometry/buildings/design/detailed-prediction/):
  the same prediction band by band, with the per-path contributions behind
  R'w and L'n,w.
- [Predicting Panel Sound Insulation](/phonometry/buildings/design/panel-sound-insulation/):
  the mass law, coincidence dip, double walls and apertures.
- [Floor-Covering Impact Improvement (ISO 16251-1)](/phonometry/buildings/design/impact-improvement/):
  the weighted improvement of a soft covering on a small heavyweight mock-up.
- [Predicting Resilient-Layer Performance](/phonometry/buildings/design/resilient-layers/):
  the prediction side of coverings, floating floors and wall linings.
- [Structure-borne sound power of equipment (EN 15657)](/phonometry/buildings/design/structure-borne-power/):
  the reception-plate method and the plate-independent source quantities.
- [Installed structure-borne sound (EN 12354-5)](/phonometry/buildings/design/installed-structure-borne/):
  the receiving-room sound pressure level predicted from source and receiver
  mobilities.
- [Dynamic stiffness of resilient materials (EN 29052-1)](/phonometry/materials/resilient/dynamic-stiffness/):
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
simulation](/phonometry/simulation/).

**And a prediction is not a verdict.** The single-number ratings and the
national indices are computed here, but the limit values they are judged
against are national — the Spanish code is implemented as a worked example of
one such framework, not as the rule everywhere — and the requirement always
comes from your regulation.

## Before and after these pages

Every quantity on these pages starts from band levels or from a filtered
impulse response, so the calibration, weighting and fractional-octave
filtering behind them are in [Signal analysis](/phonometry/signals/), and
[Build a sound level meter](/phonometry/signals/sound-level-meter/) runs that
chain end to end on one runnable page. The derivations sit in [Rooms and
buildings theory](/phonometry/reference/theory/rooms-buildings/), from the Schroeder integration
to the EN 12354 path sums.

If you arrived here from a search and want the shape of the whole library,
[What do you need to measure?](/phonometry/start/tasks/) indexes it by the job
and [All guides](/phonometry/start/guides/) lists every page with a line on
each.
