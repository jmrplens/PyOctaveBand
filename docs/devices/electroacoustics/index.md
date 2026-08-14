← [Documentation index](../../README.md)

# Electroacoustics

This section turns to devices that are *supposed* to make sound: amplifiers,
loudspeakers and microphones, and the broadcast programme they end up
carrying.

[Electroacoustics](electroacoustics.md) covers the
IEC 60268-3 distortion set (THD, THD+N and SINAD, intermodulation and DIM)
and the H1/H2 frequency-response estimators with coherence. The two device
families have their own type-test pages:
[Loudspeaker Characterisation (IEC 60268-5)](loudspeakers.md)
with the radiating-piston model behind a loudspeaker's directivity, and
[Microphone Characterisation (IEC 60268-4)](microphones.md)
with the polar patterns and the inherent-noise conventions, both ending in a
one-page report laid out the way an accredited laboratory lays one out, with
the rated-characteristics table beside the response, polar and noise panels
drawn to the IEC 60263 scale conventions. What makes such a sheet a test report
rather than a design study is the measured input data and the declared
standard, and both of those you supply. The loudspeaker page then puts the
characterised device back in a room with an open microphone, where the question
stops being a datasheet number and becomes whether the loop is stable: Long's
gain-before-feedback criterion, the correction for the number of open
microphones and the 10 dB margin an equalised system is designed to are
computed there. If you are designing or diagnosing a sound-reinforcement system
rather than testing a loudspeaker, that is the section to go to.

The three device pages divide by what is under test, and therefore by what the
bench looks like. An **amplifier** works into its rated load impedance from its
rated supply and source impedance, and IEC 60268-3's standard measuring
conditions (clause 3.1.3) then drop the source e.m.f. 10 dB below the rated
value, so a distortion figure is meaningless without saying which of the two it
was taken at. A **loudspeaker** is measured on its reference axis at a stated
distance in a free or half-space free field, with its mounting declared. A
**microphone** is measured against a calibrated reference. The two type-test
pages start from measured curves rather than telling you how to acquire them,
so the conditions under which those curves were taken travel with every number
in the resulting fiche and belong in it.
[Swept-sine distortion and phase utilities](swept-sine-distortion.md)
extends the bench with the one-sweep alternative: the Farina / Novak
harmonic separation that turns a single exponential sweep into the full set
of harmonic frequency responses and a THD measured as a function of the
excitation frequency, plus the minimum-phase, group-delay and excess-phase
utilities that dissect what the measured response's phase is made of.

[Programme loudness](../broadcast/program-loudness.md) covers the signal
the devices carry: the ITU-R BS.1770-5 loudness of a broadcast or streaming
programme in LUFS, the EBU R 128 normalisation to -23 LUFS, the EBU Mode
momentary/short-term/integrated meters, the loudness range and the
oversampled true-peak level in dBTP.

If you are here to bench-test audio gear, go straight to
[Electroacoustics](electroacoustics.md); if you are here to
level a programme, go to
[Programme loudness](../broadcast/program-loudness.md). To measure what a
machine emits rather than what a transducer reproduces, the
[Sound power and intensity](../emission/index.md) pages
are the place to start.

## Pages in this section

- [Electroacoustics: distortion and frequency response](electroacoustics.md):
  the IEC 60268-3 distortion metrics and frequency-response estimation with
  coherence.
- [Loudspeaker Characterisation (IEC 60268-5)](loudspeakers.md):
  the sensitivity conventions, the radiating piston and the IEC 60268-5
  characteristics fiche.
- [Microphone Characterisation (IEC 60268-4)](microphones.md):
  the sensitivity references, directional patterns and inherent noise of the
  IEC 60268-4 fiche.
- [Swept-sine distortion and phase utilities](swept-sine-distortion.md):
  harmonic separation and THD(f) from one exponential sweep (Farina /
  Novak synchronized swept-sine), and minimum phase, group delay and
  excess phase from a measured response.
- [Broadcast](../broadcast/index.md): the loudness problem solved
  with a measurement rather than a compressor, one gated number per programme
  and the range that says how much it moves.
- [Programme loudness and true peak](../broadcast/program-loudness.md):
  the ITU-R BS.1770-5 programme loudness and true-peak level with the
  EBU R 128 normalisation practice, EBU Mode metering and loudness range.

## What this section does not cover

The type-test pages **reduce and report** what a laboratory measured; they do
not acquire it. The free-field response, the polar cuts, the noise spectrum
and the distortion-against-level sweep come in as data, and no procedure here
tells you how to run the anechoic room or the substitution measurement that
produced them. Two implemented editions are pinned rather than current: the
distortion metrics follow AES17-2015 and not the 2020 revision, and the
microphone report follows IEC 60268-4:2014 and not the 2018 one. Thiele-Small
parameter extraction from an impedance curve is not implemented, and the
electrical and mechanical power-handling ratings of IEC 60268-5 clause 17 are
stated by the manufacturer rather than computed. The feedback criterion is
level bookkeeping, not an acoustic model: it consumes two direct-field levels
you supply, does not compute them from a coverage pattern, and predicts
neither the ring frequency nor the effect of an equaliser or frequency
shifter. And IEC 60268-16, the speech transmission index, is not part of this
section at all — it is in [Speech](../../perception/speech/index.md).
