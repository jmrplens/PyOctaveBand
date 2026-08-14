← [Documentation index](../../README.md)

# Room acoustics

Almost everything about the sound field inside a room follows from two
quantities: its **impulse response**, which can be measured, and its **sound
absorption**, which can be designed. The pages of this section cover the
measurement chain built on the first, the prediction chain built on the
second, and the rating of the background noise that occupies the room in
between.

One boundary runs through all of it: the **Schroeder frequency**. Above it a
room has so many overlapping modes that a statistical description is the honest
one, and every reverberation formula and decay parameter on these pages lives
there. Below it the modes are discrete and separable, and no statistical model
applies — which is why the measurement and prediction pages alike carry validity
caveats at their lowest bands. A reader chasing a low-frequency problem should
start from the modal treatment rather than from the decay parameters.

The measurement chain starts in
[Measuring the Room Impulse Response](room-impulse-response.md):
the deterministic excitation signals of ISO 18233, the sweep deconvolution
that turns a recording into an impulse response, and the MLS alternative.
[Room Acoustics](room-acoustics.md) then derives the ISO 3382
parameters from it (reverberation time, EDT, clarity, definition and centre
time), and
[Open-Plan Office Acoustics (ISO 3382-3)](open-plan-acoustics.md)
answers the question a single closed room does not raise: how far speech stays
intelligible across an open floor, through the spatial decay rate and the
distraction and privacy distances.
[Image sources and the steady-state room field](room-image-sources.md)
approaches the same room deterministically, building its impulse response from
mirrored sources, its steady-state level from the room constant, and, below the
Schroeder frequency where both of those give out, the discrete normal modes of
the shoebox itself.

Before the two prediction pages, one page answers a different question about the
same room.
[Room-noise criteria (NC / RC Mark II)](room-noise.md)
asks whether its steady background noise (ventilation, distant traffic) is
acceptable for its use, rated against the ANSI/ASA S12.2 criterion curves, with
the RC Mark II rumble/hiss tag diagnosing *why* a spectrum fails.

Prediction gets two pages because two traditions coexist. Both are diffuse-field
statistical models fed by the same laboratory absorption coefficients, so they
are not rival physics; they differ in what they are admissible for.
[Reverberation-time prediction (Sabine, Eyring, Arau)](reverberation-prediction.md)
covers the classical statistical formulae (Sabine, Eyring, Millington-Sette,
Fitzroy and Arau-Puchades), including the models that handle a non-uniform
absorption distribution.
[Sound absorption in enclosed spaces (EN 12354-6)](enclosed-space-absorption.md)
covers the normative European version of the same physics: the total
equivalent absorption area assembled from surfaces, objects and air, and the
reverberation time that follows from it, as a standard a design report can
cite.

**Which one?** Cite EN 12354-6 when the deliverable is a design report under a
European building-acoustics framework, when the room is an ordinary building
space inside the clause 4.6 validity limits, and when the receiving-room
absorption has to feed an EN 12354 insulation prediction. Use the classical
family when the room falls outside that scope — a hall, a theatre, an
industrial space, or a room whose absorption is concentrated on one axis so that
an axial model is needed — or when a *band* of predictions rather than a single
normative value is what the situation deserves. Both share one failure mode, the
loss of diffusivity, and they fail in the same direction: the measured
reverberation time comes out longer than predicted, by up to a factor of two in
the low-diffusivity rooms the standard's own accuracy clause records. And
neither replaces a measurement — the measured counterpart is
[Room Acoustics](room-acoustics.md).

Related pages elsewhere: the absorption coefficient the prediction chain
consumes is measured in
[Sound Absorption Measurement and Rating](../../materials/absorbers/absorption-measurement.md),
insulation *between* rooms continues in
[Sound insulation](../insulation/index.md), and the
speech intelligibility a room affords is quantified by the
[Speech Transmission Index](../../perception/speech/speech-transmission.md).

## Pages in this section

- [Measuring the Room Impulse Response](room-impulse-response.md):
  the ISO 18233 deterministic methods, exponential sweeps and their
  deconvolution, and MLS.
- [Room Acoustics](room-acoustics.md): the ISO 3382-1/2
  parameters (T20, T30, EDT, C50, C80, D50, Ts) with the Schroeder integration
  and the accredited fiche.
- [Open-Plan Office Acoustics (ISO 3382-3)](open-plan-acoustics.md):
  the spatial decay rate of speech and the distraction and privacy distances.
- [Image sources and the steady-state room field](room-image-sources.md):
  the deterministic image-source room impulse response (Kuttruff/Vorländer),
  the statistical steady-state level with the room constant, critical distance
  and Schroeder frequency (Bies), and the rectangular-room normal modes with
  their axial, tangential and oblique families, mode count and modal density
  (Long).
- [Room-noise criteria (NC / RC Mark II)](room-noise.md):
  the ANSI/ASA S12.2-2019 NC tangency and RC Mark II ratings.
- [Reverberation-time prediction (Sabine, Eyring, Arau)](reverberation-prediction.md):
  the five statistical models with the air-absorption term.
- [Sound absorption in enclosed spaces (EN 12354-6)](enclosed-space-absorption.md):
  the normative equivalent-absorption-area and reverberation-time prediction.

## What this section does not cover

**Nothing here is a wave solver.** The image-source model is specular only: it
carries no diffraction, no scattering off a diffuser and no finite-impedance
boundary, and it stops when the reflection order runs out rather than when the
sound does. Below the Schroeder frequency, where the statistical models give
out, what this section offers is the mode *positions* of a rigid rectangular
box — not the field of a real room at low frequency. For that, the [wave
simulation](../../simulation/index.md) section runs an FDTD solver on the actual
geometry.

**No auralisation, no ray tracer, no room model.** There is no geometry
importer, no material database and no renderer: the pages take dimensions,
coefficients and impulse responses as inputs, and give back parameters. The
absorption coefficients themselves come from [Materials and
surfaces](../../materials/absorbers/index.md), and the model errs optimistically
when the room is not diffuse — outside the EN 12354-6 clause 4.6 limits (no
dimension more than five times another, opposite surfaces within a factor of
three in absorption, object fraction below 0.2) the measured reverberation
time can reach twice the predicted one.

Two coverage boundaries follow the standards. Only the normative clause 4
model of EN 12354-6 is implemented, not its informative Annex D method for
irregular spaces. And nothing in this section measures insulation *between*
rooms: that is [Sound insulation](../insulation/index.md).
