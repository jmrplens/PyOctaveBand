---
title: "Psychoacoustics"
description: "The perceptual metrics of sound: loudness models (ISO 532, ECMA-418-2), sharpness, tonality, roughness and fluctuation strength, the two tonal-assessment methods (ECMA-418-1 prominence and ISO/PAS 20065 audibility), and the Fastl & Zwicker psychoacoustic annoyance."
---

Psychoacoustics replaces the question "how many decibels?" with "what does the
listener perceive?". Its base quantity is **loudness**: a perceptual magnitude
in sones, computed by auditory models that account for the ear's filtering,
masking and compression. On top of loudness sit the **sound quality**
sensations that distinguish two equally loud sounds: sharpness (high-frequency
emphasis), tonality (audible discrete tones), roughness (fast modulation)
and fluctuation strength (slow modulation). And on top sits a combined **annoyance** metric that weighs loudness,
sharpness, roughness and fluctuation strength into a single scalar.

Every metric here is a magnitude fixed by a **reference sound** rather than by a
physical unit, and knowing the anchor is what makes a number readable: 1 sone is
a 1 kHz tone at 40 dB SPL, 1 acum a critical-band-wide noise at 1 kHz and 60 dB,
1 asper a 1 kHz carrier fully modulated at 70 Hz at 60 dB, 1 vacil the same
carrier modulated at 4 Hz, and 1 tu_HMS a 1 kHz tone at 40 dB. They are
tabulated together, beside the speech and hearing scales, under "Reading the
numbers" on the [section overview](/phonometry/perception/).

The two families of pages differ in purpose, and that difference decides what
you can conclude. Loudness, sharpness, roughness and fluctuation strength are
**open-ended magnitudes** for comparing designs: there is no pass mark, and the
useful statement is always a comparison. The two tonal pages end in a **verdict
against a criterion**, because they exist to justify a declaration or a penalty.
ECMA-418-2 sits between the two: it attaches informative prominence criteria to
its tonality (0.4 tu_HMS on a band), roughness (0.2 asper) and fluctuation
strength (0.2 vacil_HMS), which is the closest thing to a pass mark in the
magnitude family. All of them share one prerequisite: an absolutely calibrated
signal in pascals, because every metric here is level-dependent.

[Loudness](/phonometry/perception/psychoacoustics/loudness/) is the foundation page: the Zwicker
reference method of ISO 532-1 with its one-page fiche, together with the
ISO 226:2023 equal-loudness contours that anchor the perceptual scale for pure
tones. The newer model families,
Moore-Glasberg per ISO 532-2/-3 and the Sottek Hearing Model of ECMA-418-2,
continue in
[Advanced Loudness](/phonometry/perception/psychoacoustics/advanced-loudness/), which also
carries the model-choice table.
[Sound Quality Metrics](/phonometry/perception/psychoacoustics/sound-quality/) adds
sharpness per DIN 45692 and the ECMA-418-2 tonality, roughness and
fluctuation strength that share the Sottek front-end.

Tones in noise get two dedicated pages because two different questions are
asked of them.
[Prominent Discrete Tones (ECMA-418-1)](/phonometry/perception/psychoacoustics/tone-prominence/)
answers a product-noise question: is this tone *prominent* by the
tone-to-noise and prominence-ratio criteria used in IT-equipment declarations?
[Objective audibility of tones in noise (ISO/PAS 20065)](/phonometry/perception/psychoacoustics/tone-audibility/)
answers an environmental one: by how many decibels does the tone exceed its
masking threshold, the audibility that feeds the tonal penalty of
ISO 1996-2.

[Psychoacoustic annoyance and fluctuation strength](/phonometry/perception/psychoacoustics/psychoacoustic-annoyance/)
closes the chain with the Fastl & Zwicker model, which combines loudness,
sharpness, roughness and the slow-modulation sensation of fluctuation strength
into a single annoyance value. Read it last: three of its four inputs come from
the earlier pages, and it supplies the fourth, fluctuation strength, itself, in
both the Fastl & Zwicker closed form and the Osses 2016 signal model. The
ECMA-418-2 fluctuation strength on the Sound Quality page is a further,
normative model of the same sensation, under a different unit name.

## Pages in this section

- [Loudness](/phonometry/perception/psychoacoustics/loudness/): the ISO 532-1 Zwicker loudness in
  sones, plus the ISO 226:2023 equal-loudness contours.
- [Advanced Loudness (ISO 532-2/-3, ECMA-418-2)](/phonometry/perception/psychoacoustics/advanced-loudness/):
  the Moore-Glasberg and Sottek loudness models and the model-choice table.
- [Sound Quality Metrics](/phonometry/perception/psychoacoustics/sound-quality/): sharpness
  (DIN 45692) and ECMA-418-2 tonality, roughness and fluctuation strength.
- [Prominent Discrete Tones (ECMA-418-1)](/phonometry/perception/psychoacoustics/tone-prominence/):
  tone-to-noise and prominence ratios with prominence verdicts.
- [Objective audibility of tones in noise (ISO/PAS 20065)](/phonometry/perception/psychoacoustics/tone-audibility/):
  the audibility of a tone above the masking threshold, feeding the
  ISO 1996-2 tonal adjustment.
- [Psychoacoustic annoyance and fluctuation strength](/phonometry/perception/psychoacoustics/psychoacoustic-annoyance/):
  the Fastl & Zwicker annoyance model and the fluctuation-strength models it
  consumes.

## What this section does not cover

**Everything here is monaural.** The binaural combinations ECMA-418-2 defines
for loudness, roughness and fluctuation strength are not implemented, so a
stereo or binaural recording is analysed one channel at a time, and no model
here accounts for localisation or for spatial release from masking. Two
optional refinements are also left out: the entropy weighting of clause 7.1.6,
which needs an external rotational-speed signal, and the small adjustment
footnote 47 permits.

**A verdict is never complete.** The `prominent` flag the tone-prominence
functions return is the numeric criterion alone; ECMA-418-1 also requires aural
confirmation and a lower-threshold-of-hearing screen, both of which stay with
the caller. The tone-audibility module is weighting-agnostic and does **not**
apply the A-weighting clause 5.3.2 requires, so A-weight the spectrum before
passing it, and it takes an already-computed narrow-band spectrum rather than
building one from a recording.

Two documented deviations are worth knowing. ISO 532-3 prescribes resampling to
32 kHz before the running FFT; this implementation works at the native rate, a
deviation that stays inside the standard's expanded uncertainty but that you
should undo by resampling first if strict clause-by-clause conformance matters.
And the Osses 2016 fluctuation-strength signal model is validated for
amplitude-modulated stimuli only, with a documented floor — a steady 1 kHz tone
reads about 0.09 vacil rather than 0.

Finally, none of these metrics is a community response: annoyance here is a
laboratory sensation computed from a signal, while the annoyance a
neighbourhood reports is a social-survey quantity handled through the
indicators of [Environment and transport](/phonometry/environment/).
