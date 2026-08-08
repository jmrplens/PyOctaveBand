← [Documentation index](../../README.md)

# Speech

All three pages in this section reduce speech intelligibility to a number in
[0, 1], and the art is knowing which number answers your question. The
**Speech Transmission Index** (STI) rates a *transmission channel*: a room, a
public-address system, an intercom. The **Speech Intelligibility Index** (SII)
rates a *listening condition*: this speech spectrum, in this noise, heard by
this listener. A reverberant lecture hall is an STI problem; a hearing-aid
fitting or a cockpit warning heard over engine noise is an SII problem.

The shared [0, 1] range is a coincidence of normalisation, not a common scale,
and 0.6 means three different things on the three. An **STI** of 0.6 falls in
band D of the IEC 60268-16 Annex F qualification ladder, whose eleven letters
run from U below 0.36 to A+ at 0.76 and above; that is a good lecture room, and
a voice-alarm specification typically sets its minimum a couple of bands lower.
An **SII** of 0.6 means roughly 60 % of the importance-weighted speech spectrum
is audible to that listener in that noise; the index is a fraction by
construction and carries no standardised qualification ladder at all. A
**STOI** of 0.6 has no absolute meaning: the mapping from the index to a
percentage of words understood is fitted per listening-test corpus and is
deliberately not implemented, so STOI is only ever read as a difference between
two processors on the same material. Never substitute one index for another in
a specification, and when a requirement quotes a number, check which standard
it belongs to before computing anything.

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

The two standardised indices connect back to the rest of the library naturally:
the STI consumes the impulse responses of
[Room Acoustics](../../buildings/rooms/room-acoustics.md), and the SII consumes the
hearing thresholds quantified in
[Hearing threshold](../hearing/hearing-threshold.md). STOI and
ESTOI have an upstream too, but a different kind of one: they take waveforms,
so what feeds them is whatever produced the clean and the degraded recording —
which is why they sit beside the signal-processing tools of [Signals and
spectra](../../signals/spectra/index.md) rather than beside a measurement
standard.

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

## What this section does not cover

**No listener is tested, and no score is predicted.** STOI returns the
correlation index and not the percentage of words understood, because the
logistic mapping is fitted per listening-test corpus; the SII returns an
audibility fraction rather than a score; and no page here reproduces a
subjective intelligibility test. **No signal is acquired either**: the STI page
implements the STIPA direct signal and the indirect computation from an impulse
response, but the full 14-modulation-frequency direct measurement of clause 6.3
is not implemented, so a chain with severe distortion needs measuring equipment
rather than this library.

Two coverage limits inside the SII are worth checking before use: the raised,
loud and shouted speech spectra are carried for the one-third-octave procedure
only, and the tabulated band-importance functions are each table's
average-speech compromise, with Annex B's material-specific alternatives left to
you through the `band_importance=` argument. There is no resampling between the
four band procedures — each is fed spectra on its own bands. And the female
speech option is not missing from the STI: Edition 5 of IEC 60268-16 removed it,
so there is nothing left to implement.
