---
title: "Sources and devices"
description: "Characterising what emits the sound: sound power determination by pressure, reverberation-room and intensity methods (ISO 3740 and ISO 9614 series), two-microphone sound intensity (IEC 61043), the IEC 60268 distortion, loudspeaker and microphone metrics of audio equipment, swept-sine harmonic separation with THD(f) (Farina / Novak), reactive silencers and industrial noise control, and the ITU-R BS.1770-5 / EBU R 128 programme loudness and true peak."
---

Every prediction elsewhere in this documentation starts from a source
descriptor, and this section is where those descriptors are measured. Its
common thread is **emission**: numbers that belong to the device rather than
to the room or the distance it is heard at.

The **sound power and intensity** pages determine the central emission
quantity, the sound power level: the figure that goes on a datasheet, feeds a
room or outdoor prediction and is checked against noise-emission limits. The
**electroacoustics** pages turn to devices that are *supposed* to make sound
(amplifiers, loudspeakers and microphones) and to the broadcast programme
they carry, and the **noise control** pages hold the path measures that quiet
a machine once its emission is known.

If you are here to measure a machine, start with
[Sound Power](/phonometry/devices/emission/sound-power/) and let its decision guidance
pick the route; read [Sound Intensity (p-p)](/phonometry/devices/emission/intensity/)
when that route involves an intensity probe. If you are here to bench-test
audio gear, go straight to
[Electroacoustics](/phonometry/devices/electroacoustics/electroacoustics/); if you are here to
level a programme, go to
[Programme loudness](/phonometry/devices/broadcast/program-loudness/).

## [Sound power and intensity](/phonometry/devices/emission/)

The total acoustic emission of a source, and the power flux it is built on.

- [Sound Intensity (p-p)](/phonometry/devices/emission/intensity/): two-microphone
  sound intensity per IEC 61043 with the ISO 9614-1 field indicators.
- [Sound Power](/phonometry/devices/emission/sound-power/): choosing the determination
  method and declaring the noise emission per ISO 4871.
- [Sound Power by Pressure Methods](/phonometry/devices/emission/sound-power-pressure/):
  the enveloping surface of ISO 3744/3746 and the precision anechoic grade of
  ISO 3745.
- [Sound Power in the Reverberation Room](/phonometry/devices/emission/sound-power-reverberation/):
  the direct and comparison methods of ISO 3741.
- [Sound Power by Intensity Scanning](/phonometry/devices/emission/sound-power-intensity/):
  the on-site scanning of ISO 9614-2 and the ISO 9614-3 precision grade.

## [Electroacoustics](/phonometry/devices/electroacoustics/)

Amplifiers, loudspeakers and microphones on the bench, and the programme
signal they carry.

- [Electroacoustics: distortion and frequency response](/phonometry/devices/electroacoustics/electroacoustics/):
  the IEC 60268-3 distortion metrics and frequency-response estimation with
  coherence.
- [Loudspeaker Characterisation (IEC 60268-5)](/phonometry/devices/electroacoustics/loudspeakers/):
  the sensitivity conventions, the radiating piston and the IEC 60268-5
  characteristics fiche.
- [Microphone Characterisation (IEC 60268-4)](/phonometry/devices/electroacoustics/microphones/):
  the sensitivity references, directional patterns and inherent noise of the
  IEC 60268-4 fiche.
- [Swept-sine distortion and phase utilities](/phonometry/devices/electroacoustics/swept-sine-distortion/):
  harmonic separation and THD(f) from one exponential sweep (Farina /
  Novak synchronized swept-sine), and minimum phase, group delay and
  excess phase from a measured response.
- [Programme loudness and true peak](/phonometry/devices/broadcast/program-loudness/):
  the ITU-R BS.1770-5 programme loudness and true-peak level with the
  EBU R 128 normalisation practice, EBU Mode metering and loudness range.

## [Noise control](/phonometry/devices/noise-control/)

Industrial noise control on the path, between the machine and whoever hears
it.

- [Silencers](/phonometry/devices/noise-control/silencers/): reactive silencers by the
  four-pole method and the reactive-versus-dissipative choice.
- [Industrial Noise Control: HVAC and Enclosures](/phonometry/devices/noise-control/noise-control/):
  duct attenuation, flow noise and machine-enclosure insertion loss.
