← [Documentation index](../../README.md)

# Speech

Both pages in this section reduce speech intelligibility to a number in
[0, 1], and the art is knowing which number answers your question. The
**Speech Transmission Index** (STI) rates a *transmission channel*: a room, a
public-address system, an intercom. The **Speech Intelligibility Index** (SII)
rates a *listening condition*: this speech spectrum, in this noise, heard by
this listener. A reverberant lecture hall is an STI problem; a hearing-aid
fitting or a cockpit warning heard over engine noise is an SII problem.

The physical difference sits in what each index models. STI
(**IEC 60268-16**) works on the speech *envelope*: intelligibility degrades
when reverberation and noise flatten the slow intensity modulations of speech,
and the index measures how much of that modulation survives the channel, via
the modulation transfer function. It can be computed indirectly from a
measured impulse response or measured directly with the STIPA test signal.
[Speech Transmission Index (STI)](speech-transmission.md)
covers the modulation physics, both methods and the Annex F rating bands.

SII (**ANSI S3.5-1997**) works on *audibility*: intelligibility is predicted
from how much of the speech-bearing spectrum rises above the listener's
effective threshold, band by band, weighted by each band's importance to
speech. Noise, self-masking, upward spread of masking and the listener's own
hearing threshold all enter explicitly, which is why SII extends naturally to
hearing loss.
[Speech Intelligibility Index](speech-intelligibility.md)
covers all four of the standard's band procedures (critical band,
equally-contributing critical band, one-third octave and octave), including the
standard speech spectra for normal to shouted vocal effort.

A third pair of measures, **STOI** and **ESTOI**, answers yet another question:
given a clean reference *and* a degraded or processed version of the same
speech, how intelligible is the result? They rate the processing itself,
which is why they are the standard yardstick for noise reduction and source
separation.

The two connect back to the rest of the library naturally: the STI consumes
the impulse responses of
[Room Acoustics](../../buildings/rooms/room-acoustics.md), and the SII consumes the
hearing thresholds quantified in
[Hearing threshold](../hearing/hearing-threshold.md).

## Pages in this section

- [Speech Transmission Index (STI)](speech-transmission.md):
  the IEC 60268-16 modulation transfer function, the indirect method from an
  impulse response, and direct STIPA measurement.
- [Speech Intelligibility Index](speech-intelligibility.md):
  the ANSI S3.5-1997 band-importance and band-audibility method, in noise and
  in hearing loss.
- [Objective Intelligibility (STOI & ESTOI)](objective-intelligibility.md):
  the correlation-based measures for time-frequency weighted noisy speech, from
  a clean/degraded pair.
