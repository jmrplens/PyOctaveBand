← [Documentation index](../README.md)

# Hearing and perception

A sound pressure level says how much sound there is; this section is about
what a **listener** makes of it. Its three subsections answer three different
questions. **Psychoacoustics** quantifies sensations: how loud a sound is
perceived to be, how sharp, rough or tonal it is, and how those sensations
combine into annoyance. **Speech** asks how well spoken words survive the trip
from a talker to a listener, through a room, a sound system or background
noise. And **hearing and exposure** covers the ear itself: where the hearing
threshold sits across age and population, how noise permanently shifts it, and
how a working day's exposure is measured and reported.

The three build on each other in one direction: the psychoacoustic models
consume calibrated signals or band spectra from the
[core analysis](../signal/index.md); the speech
indices consume band levels and, in the SII's case, the hearing thresholds
that the hearing pages quantify; and the exposure metrics feed the
hearing-damage model of ISO 1999.

A good entry point is [Loudness](psychoacoustics/loudness.md): it introduces
the perceptual scale (the sone) and the auditory models that most other
metrics in this section reuse or extend.

## [Psychoacoustics](psychoacoustics/index.md)

The perceptual sensations of sound: loudness and the metrics layered on it.

- [Loudness](psychoacoustics/loudness.md): the ISO 532-1 Zwicker loudness in
  sones, plus the ISO 226:2023 equal-loudness contours.
- [Advanced Loudness (ISO 532-2/-3, ECMA-418-2)](psychoacoustics/advanced-loudness.md):
  the Moore-Glasberg stationary and time-varying methods and the Sottek
  Hearing Model loudness, with the model-choice table.
- [Sound Quality Metrics](psychoacoustics/sound-quality.md): sharpness
  (DIN 45692) and the ECMA-418-2 Sottek Hearing Model tonality, roughness
  and fluctuation strength.
- [Prominent Discrete Tones (ECMA-418-1)](psychoacoustics/tone-prominence.md):
  the tone-to-noise and prominence ratios that decide whether a discrete tone
  is prominent.
- [Objective audibility of tones in noise (ISO/PAS 20065)](psychoacoustics/tone-audibility.md):
  the engineering method for the audibility of a tone above the masking
  threshold, feeding the ISO 1996-2 tonal adjustment.
- [Psychoacoustic annoyance and fluctuation strength](psychoacoustics/psychoacoustic-annoyance.md):
  the Fastl & Zwicker annoyance model combining loudness, sharpness, roughness
  and fluctuation strength.

## [Speech](speech/index.md)

Two complementary indices of speech intelligibility, STI for the
transmission channel and SII for the listening condition, plus the
signal-based STOI and ESTOI measures.

- [Speech Transmission Index (STI)](speech/speech-transmission.md):
  the IEC 60268-16 modulation transfer function, the indirect method from an
  impulse response, and direct STIPA measurement.
- [Speech Intelligibility Index](speech/speech-intelligibility.md):
  the ANSI S3.5-1997 one-third-octave-band SII from speech, noise and hearing
  threshold spectra.
- [Objective Intelligibility (STOI & ESTOI)](speech/objective-intelligibility.md):
  the correlation-based measures for time-frequency weighted noisy speech, from
  a clean/degraded pair.

## [Hearing and exposure](hearing/index.md)

The hearing threshold, what noise does to it, and how exposure is measured.

- [Hearing threshold (age and reference zero)](hearing/hearing-threshold.md):
  the ISO 7029:2017 age-related threshold distribution and the ISO 389-7:2005
  reference threshold of hearing.
- [Noise-induced hearing loss (ISO 1999)](hearing/noise-induced-hearing-loss.md):
  the noise-induced permanent threshold shift and its combination with age
  into HTLAN.
- [Occupational Noise Exposure (ISO 9612)](hearing/occupational-exposure.md):
  the task-based, job-based and full-day strategies for LEX,8h with the
  Annex C uncertainty budget.
