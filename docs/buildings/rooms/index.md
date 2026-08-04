← [Documentation index](../../README.md)

# Room acoustics

Almost everything about the sound field inside a room follows from two
quantities: its **impulse response**, which can be measured, and its **sound
absorption**, which can be designed. The pages of this section cover the
measurement chain built on the first, the prediction chain built on the
second, and the rating of the background noise that occupies the room in
between.

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
mirrored sources and its steady-state level from the room constant.

Prediction gets two pages because two traditions coexist.
[Reverberation-time prediction (Sabine, Eyring, Arau)](reverberation-prediction.md)
covers the classical statistical formulae (Sabine, Eyring, Millington-Sette,
Fitzroy and Arau-Puchades), including the models that handle a non-uniform
absorption distribution.
[Sound absorption in enclosed spaces (EN 12354-6)](enclosed-space-absorption.md)
covers the normative European version of the same physics: the total
equivalent absorption area assembled from surfaces, objects and air, and the
reverberation time that follows from it, as a standard a design report can
cite.

[Room-noise criteria (NC / RC Mark II)](room-noise.md)
answers a different question about the same room: whether its steady
background noise (ventilation, distant traffic) is acceptable for its use,
rated against the ANSI/ASA S12.2 criterion curves, with the RC Mark II
rumble/hiss tag diagnosing *why* a spectrum fails.

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
  the deterministic image-source room impulse response (Kuttruff/Vorländer) and
  the statistical steady-state level with the room constant, critical distance
  and Schroeder frequency (Bies).
- [Room-noise criteria (NC / RC Mark II)](room-noise.md):
  the ANSI/ASA S12.2-2019 NC tangency and RC Mark II ratings.
- [Reverberation-time prediction (Sabine, Eyring, Arau)](reverberation-prediction.md):
  the five statistical models with the air-absorption term.
- [Sound absorption in enclosed spaces (EN 12354-6)](enclosed-space-absorption.md):
  the normative equivalent-absorption-area and reverberation-time prediction.
