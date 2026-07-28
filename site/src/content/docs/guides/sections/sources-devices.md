---
title: "Sources and devices"
description: "Characterising what emits the sound: sound power determination by pressure, reverberation-room and intensity methods (ISO 3740 and ISO 9614 series), two-microphone sound intensity (IEC 61043), the IEC 60268 distortion, loudspeaker and microphone metrics of audio equipment, swept-sine harmonic separation with THD(f) (Farina / Novak), reactive silencers and industrial noise control, and the ITU-R BS.1770-5 / EBU R 128 programme loudness and true peak."
---

Every prediction elsewhere in this documentation starts from a source
descriptor, and this section is where those descriptors are measured. Its
common thread is **emission**: numbers that belong to the device rather than
to the room or the distance it is heard at.

The central quantity is the **sound power**: the total acoustic energy per
second a source radiates. Expressed in decibels as the sound power level, it
is the figure that goes on a datasheet, feeds a room or outdoor prediction
and is checked against noise-emission limits.
[Sound Power](/phonometry/guides/sound-power/) chooses between the
standardised routes and closes the job with the ISO 4871 emission
declaration, and each route has its own page:
[Sound Power by Pressure Methods](/phonometry/guides/sound-power-pressure/)
for the enveloping surface of ISO 3744/3746 and the precision anechoic grade
of ISO 3745,
[Sound Power in the Reverberation Room](/phonometry/guides/sound-power-reverberation/)
for the direct and comparison methods of ISO 3741, and
[Sound Power by Intensity Scanning](/phonometry/guides/sound-power-intensity/)
for the on-site scanning of ISO 9614-2 and its ISO 9614-3 precision grade.
Behind the intensity-based routes sits **sound intensity** itself:
the signed power flux that can localise sources and separate them from
background noise, measured with a two-microphone probe per IEC 61043 and
qualified by the ISO 9614-1 field indicators, covered in
[Sound Intensity (p-p)](/phonometry/guides/intensity/).

[Electroacoustics](/phonometry/guides/electroacoustics/) turns to devices that
are *supposed* to make sound: amplifiers, loudspeakers and microphones. It
covers the IEC 60268-3 distortion set (THD, THD+N and SINAD, intermodulation
and DIM) and the H1/H2 frequency-response estimators with coherence. The two
device families have their own type-test pages:
[Loudspeaker Characterisation (IEC 60268-5)](/phonometry/guides/loudspeakers/)
with the radiating-piston model behind a loudspeaker's directivity, and
[Microphone Characterisation (IEC 60268-4)](/phonometry/guides/microphones/)
with the polar patterns and the inherent-noise conventions, both ending in a
one-page accredited fiche.
[Swept-sine distortion and phase utilities](/phonometry/guides/swept-sine-distortion/)
extends the bench with the one-sweep alternative: the Farina / Novak
harmonic separation that turns a single exponential sweep into the full set
of harmonic frequency responses and a THD measured as a function of the
excitation frequency, plus the minimum-phase, group-delay and excess-phase
utilities that dissect what the measured response's phase is made of.

Machinery is the other half of the section. **Industrial noise control**
attacks the source, the path and the receiver in turn:
[Silencers](/phonometry/guides/silencers/) is the path measure, the reactive
four-pole elements (expansion chambers, Helmholtz, quarter-wave and
extended-tube resonators) and the choice between reflection and dissipation,
while [Industrial Noise Control](/phonometry/guides/noise-control/) keeps
the HVAC duct attenuation and flow noise of an installation and the
insertion loss of a machine enclosure.

[Programme loudness](/phonometry/guides/program-loudness/) covers the signal
the devices carry: the ITU-R BS.1770-5 loudness of a broadcast or streaming
programme in LUFS, the EBU R 128 normalisation to -23 LUFS, the EBU Mode
momentary/short-term/integrated meters, the loudness range and the
oversampled true-peak level in dBTP.

If you are here to measure a machine, start with
[Sound Power](/phonometry/guides/sound-power/) and let its decision guidance
pick the route; read [Sound Intensity (p-p)](/phonometry/guides/intensity/)
when that route involves an intensity probe. If you are here to bench-test
audio gear, go straight to
[Electroacoustics](/phonometry/guides/electroacoustics/); if you are here to
level a programme, go to
[Programme loudness](/phonometry/guides/program-loudness/).

## Pages in this section

- [Sound Intensity (p-p)](/phonometry/guides/intensity/): two-microphone
  sound intensity per IEC 61043 with the ISO 9614-1 field indicators.
- [Sound Power](/phonometry/guides/sound-power/): choosing the determination
  method and declaring the noise emission per ISO 4871.
- [Sound Power by Pressure Methods](/phonometry/guides/sound-power-pressure/):
  the enveloping surface of ISO 3744/3746 and the precision anechoic grade of
  ISO 3745.
- [Sound Power in the Reverberation Room](/phonometry/guides/sound-power-reverberation/):
  the direct and comparison methods of ISO 3741.
- [Sound Power by Intensity Scanning](/phonometry/guides/sound-power-intensity/):
  the on-site scanning of ISO 9614-2 and the ISO 9614-3 precision grade.
- [Electroacoustics: distortion and frequency response](/phonometry/guides/electroacoustics/):
  the IEC 60268-3 distortion metrics and frequency-response estimation with
  coherence.
- [Loudspeaker Characterisation (IEC 60268-5)](/phonometry/guides/loudspeakers/):
  the sensitivity conventions, the radiating piston and the IEC 60268-5
  characteristics fiche.
- [Microphone Characterisation (IEC 60268-4)](/phonometry/guides/microphones/):
  the sensitivity references, directional patterns and inherent noise of the
  IEC 60268-4 fiche.
- [Silencers](/phonometry/guides/silencers/): reactive silencers by the
  four-pole method and the reactive-versus-dissipative choice.
- [Industrial Noise Control: HVAC and Enclosures](/phonometry/guides/noise-control/):
  duct attenuation, flow noise and machine-enclosure insertion loss.
- [Swept-sine distortion and phase utilities](/phonometry/guides/swept-sine-distortion/):
  harmonic separation and THD(f) from one exponential sweep (Farina /
  Novak synchronized swept-sine), and minimum phase, group delay and
  excess phase from a measured response.
- [Programme loudness and true peak](/phonometry/guides/program-loudness/):
  the ITU-R BS.1770-5 programme loudness and true-peak level with the
  EBU R 128 normalisation practice, EBU Mode metering and loudness range.
