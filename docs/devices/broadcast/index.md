← [Documentation index](../../README.md)

# Broadcast

Broadcasting solved the loudness problem with a measurement rather than a
compressor: one number per programme, gated so that silence does not dilute it,
and a range that says how much the programme moves.

The word *loudness* carries two meanings on this site, and they are not versions
of each other. Here it is an **energy measure**: a K-weighted mean square over
the whole programme, gated, reported in LUFS, designed so that two programmes
normalised to the same number feel equally loud on the same playback chain. In
[Psychoacoustics](../../perception/psychoacoustics/loudness.md) it is a
**perceptual magnitude in sones**, computed by an auditory model with masking
and compression. A broadcast deliverable is specified in LUFS; a product-noise
sensation is specified in sones. Reaching for the wrong one is the commonest
mistake in this area.

The quantities are few. **Loudness** is reported in LUFS by the EBU and in LKFS
by the ITU — identical units — and **1 LU is 1 dB**, so a loudness difference
and a level difference are the same size. **EBU R 128** sets the delivery
target at **−23.0 LUFS** with a true-peak ceiling of **−1 dBTP**. The **loudness
range**, in LU, says how far the programme moves between its quiet and loud
passages, which is what decides whether it needs dynamic treatment before
normalisation.

Four documents own four different things, and the section is easier to read once
that is clear. **ITU-R BS.1770** defines the algorithm: the K-weighting
pre-filter — a roughly +4 dB spherical-head shelf followed by the RLB high-pass
— the mean square in 400 ms blocks at 75 % overlap, the channel-weighted sum,
and the **two-stage gate** that makes the number usable on real programme (an
absolute gate at −70 LKFS drops digital silence, then a relative gate 10 LU
below the mean of the survivors drops the quiet passages that would otherwise
dilute a dialogue level). **EBU R 128** sets the target and the ceiling.
**EBU Tech 3341** defines the EBU Mode meter — the momentary, short-term and
integrated time scales, and the compliance test set. **EBU Tech 3342** defines
the loudness range. True peak is measured on an oversampled signal because an
inter-sample peak can exceed every sample value, so a file that reads −0.2 dBFS
can still clip a converter.

## Pages in this section

- [Programme loudness (EBU R 128)](program-loudness.md):
  the ITU-R BS.1770 K-weighting, gated 400 ms blocks and channel-weighted sum,
  the EBU R 128 target and ceiling, the Tech 3341 momentary, short-term and
  integrated meters, the Tech 3342 loudness range, the Annex 2 oversampled true
  peak and the Annex 3 channel weights for advanced sound systems — validated
  against the EBU test signals and ending in an EBU R 128 report fiche.

## See also

Pages elsewhere on the site that this section leans on:

- [Loudness](../../perception/psychoacoustics/loudness.md) (ISO 532-1): the
  other loudness, the perceptual magnitude in sones, for when the question is
  how loud something *sounds* rather than how a programme should be delivered.
- [Frequency Weighting (A, C, Z)](../../signals/levels/weighting.md): the
  weighting family K-weighting sits beside, and does not belong to.

## What this section does not cover

BS.1770-5 **Annex 4, object-based audio, is out of scope**, and the library
implements no spatial renderer, so an object-based programme has to be rendered
to a loudspeaker layout before any of this applies. **EBU Tech 3343** is cited
as production practice around these numbers, not as an algorithm: nothing here
runs it. And loudness normalisation itself — the gain change, and any limiting
that follows it — is a production step this library does not perform: it
measures the programme and tells you the offset, and applying it is your
encoder's job.
