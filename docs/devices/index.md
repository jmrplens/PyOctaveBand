← [Documentation index](../README.md)

# Sources and devices

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
[Sound Power](emission/sound-power.md) and let its decision guidance
pick the route; read [Sound Intensity (p-p)](emission/intensity.md)
when that route involves an intensity probe. If you are here to bench-test
audio gear, go straight to
[Electroacoustics](electroacoustics/electroacoustics.md); if you are here to
level a programme, go to
[Programme loudness](broadcast/program-loudness.md).

## [Sound power and intensity](emission/index.md)

The total acoustic emission of a source, and the power flux it is built on.

- [Sound Intensity (p-p)](emission/intensity.md): two-microphone
  sound intensity per IEC 61043 with the ISO 9614-1 field indicators.
- [Sound Power](emission/sound-power.md): choosing the determination
  method and declaring the noise emission per ISO 4871.
- [Sound Power by Pressure Methods](emission/sound-power-pressure.md):
  the enveloping surface of ISO 3744/3746 and the precision anechoic grade of
  ISO 3745.
- [Sound Power in the Reverberation Room](emission/sound-power-reverberation.md):
  the direct and comparison methods of ISO 3741.
- [Sound Power by Intensity Scanning](emission/sound-power-intensity.md):
  the on-site scanning of ISO 9614-2 and the ISO 9614-3 precision grade.

## [Electroacoustics](electroacoustics/index.md)

Amplifiers, loudspeakers and microphones on the bench, and the programme
signal they carry.

- [Electroacoustics: distortion and frequency response](electroacoustics/electroacoustics.md):
  the IEC 60268-3 distortion metrics and frequency-response estimation with
  coherence.
- [Loudspeaker Characterisation (IEC 60268-5)](electroacoustics/loudspeakers.md):
  the sensitivity conventions, the radiating piston and the IEC 60268-5
  characteristics fiche.
- [Microphone Characterisation (IEC 60268-4)](electroacoustics/microphones.md):
  the sensitivity references, directional patterns and inherent noise of the
  IEC 60268-4 fiche.
- [Swept-sine distortion and phase utilities](electroacoustics/swept-sine-distortion.md):
  harmonic separation and THD(f) from one exponential sweep (Farina /
  Novak synchronized swept-sine), and minimum phase, group delay and
  excess phase from a measured response.
- [Broadcast](broadcast/index.md): the loudness problem solved
  with a measurement rather than a compressor, one gated number per programme
  and the range that says how much it moves.
- [Programme loudness and true peak](broadcast/program-loudness.md):
  the ITU-R BS.1770-5 programme loudness and true-peak level with the
  EBU R 128 normalisation practice, EBU Mode metering and loudness range.

## [Noise control](noise-control/index.md)

Industrial noise control on the path, between the machine and whoever hears
it.

- [Silencers](noise-control/silencers.md): reactive silencers by the
  four-pole method and the reactive-versus-dissipative choice.
- [Duct-Borne Noise: Fan to Room](noise-control/duct-path.md): the
  end-to-end fan-to-room calculation against a room criterion, and the
  higher-order-mode cut-on that limits every plane-wave method.
- [Room to Room: Partition, Receiving Room, Criterion](noise-control/room-to-room.md):
  the composed source-room to receiving-room chain and the transmission loss a
  partition or an enclosure needs to meet a noise criterion.
- [Industrial Noise Control: HVAC and Enclosures](noise-control/noise-control.md):
  duct attenuation, flow noise and machine-enclosure insertion loss.
