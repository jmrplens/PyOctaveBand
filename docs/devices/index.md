← [Documentation index](../README.md)

# Sources and devices

Every prediction elsewhere in this documentation starts from a source
descriptor, and this section is where those descriptors are measured. Its
common thread is **emission**: numbers that belong to the device rather than
to the room or the distance it is heard at.

The **sound power and intensity** pages determine the central emission
quantity, the sound power level: the figure that goes on a datasheet, feeds a
room or outdoor prediction and is checked against noise-emission limits —
including the route that reads it off the casing's own vibration when no
microphone can be placed. The
**electroacoustics** pages turn to devices that are *supposed* to make sound
(amplifiers, loudspeakers and microphones) and to the broadcast programme
they carry, and the **noise control** pages hold the path measures that quiet
a machine once its emission is known.

If you are here to measure a machine, start with
[Sound Power](emission/sound-power.md) and let its decision guidance
pick the route, which may end on an intensity probe or, when only vibration can
be measured, on the radiating surface itself; read
[Sound Intensity (p-p)](emission/intensity.md)
when that route involves an intensity probe. If you are here to bench-test
audio gear, go straight to
[Electroacoustics](electroacoustics/electroacoustics.md); if you are here to
level a programme, go to
[Programme loudness](broadcast/program-loudness.md).

## [Sound power and intensity](emission/index.md)

The total acoustic emission of a source, and the power flux it is built on.

- [Sound Intensity (p-p)](emission/intensity.md): two-microphone
  sound intensity per IEC 61043, with the ISO 9614-1 field indicators and its
  sound power determination at discrete points.
- [Sound Power](emission/sound-power.md): choosing the determination
  method and declaring the noise emission per ISO 4871.
- [Sound Power by Pressure Methods](emission/sound-power-pressure.md):
  the enveloping surface of ISO 3744/3746 and the precision anechoic grade of
  ISO 3745.
- [Sound Power in the Reverberation Room](emission/sound-power-reverberation.md):
  the direct and comparison methods of ISO 3741.
- [Sound Power in Situ by Comparison](emission/sound-power-in-situ.md):
  the ISO 3747 comparison against a reference sound source where the machine
  works.
- [Sound Power in a Duct](emission/sound-power-in-duct.md):
  the ISO 5136 in-duct method for fans.
- [Sound Power by Intensity Scanning](emission/sound-power-intensity.md):
  the on-site scanning of ISO 9614-2 and the ISO 9614-3 precision grade.
- [Sound power from surface vibration (ISO/TS 7849)](emission/vibration-sound-power.md):
  the radiated power from the surface-averaged velocity level and the radiation
  factor, for the case where the machine cannot be moved, the room is not
  qualified and only an accelerometer is available: the Part 1 upper limit and
  the Part 2 engineering value.

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

## What this section does not cover

**No facility is qualified here.** ISO 3745's free-field qualification of an
anechoic room, ISO 3741's reverberation-room qualification and IEC 61043's
residual-intensity test of a probe are all assumed to have been done: the
library warns on the coarse advisory criteria a standard states explicitly and
grades a residual index you supply, but it does not certify a room or an
instrument. The same boundary runs through the electroacoustics pages, which
**reduce and report curves the laboratory supplies** rather than telling you
how to acquire them, and through ISO/TS 7849, whose clauses 5 to 7 on
instrumentation, installation and measurement positions are laboratory
practice this library assumes.

Two specific absences are worth knowing before you plan a job. Dissipative
duct-lining silencers are **not modelled from liner properties** anywhere:
the reactive
elements are computed exactly within the no-flow plane-wave model, and the
lined-elbow figure is a table lookup (Bies Table 8.11) and the plenum
attenuation Wells' closed form driven by a declared mean absorption — neither is a liner model.
And no page here predicts a panel's transmission loss:
`enclosure_insertion_loss` combines a value you supply with the interior
correction, and the prediction itself is [Insulation
design](../buildings/design/index.md).

Editions are pinned rather than current in two places: the distortion metrics
follow AES17-2015 and not the 2020 revision, and the microphone
rated-characteristics report follows IEC 60268-4:2014 and not the 2018 one.
Object-based audio (BS.1770-5 Annex 4) is out of scope, and the library
implements no spatial renderer, so an object-based programme has to be
rendered to a loudspeaker layout before it can be measured.

## Before and after these pages

Every emission quantity here is computed from band levels or from an intensity
pair, so the calibration, weighting and filtering behind them are in [Signal
analysis](../signals/index.md), and [Build a sound level
meter](../signals/sound-level-meter.md) runs that chain end to end on
one runnable page. The derivations are split by physics rather than by topic:
[sound power determination](../reference/theory/environment-transport.md#sound-power-determination-iso-374437453746-iso-3741-iso-9614-123)
is under Environment and transport, and [sound intensity](../reference/theory/signal-analysis.md#sound-intensity-iec-61043)
under Signal analysis. The electroacoustics and noise-control pages carry their
derivations inline.

If you arrived here from a search and want the shape of the whole library,
[What do you need to measure?](https://jmrplens.github.io/phonometry/start/tasks/) indexes it by the job
and [All guides](../README.md) lists every page with a line on
each.
