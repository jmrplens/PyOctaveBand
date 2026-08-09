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
[core analysis](../signals/index.md); the speech
indices consume band levels and, in the SII's case, the hearing thresholds
that the hearing pages quantify; and the exposure metrics feed the
hearing-damage model of ISO 1999.

A good entry point is [Loudness](psychoacoustics/loudness.md): it introduces
the perceptual scale (the sone) and the auditory models that most other
metrics in this section reuse or extend. The derivations behind these methods —
the critical-band and excitation-pattern models, the masking formulations and
the modulation-transfer chain — are gathered on the [perception theory
page](../reference/theory/perception.md), which the individual guides
cite clause by clause.

### Reading the numbers

Almost every perceptual scale in this section is defined by a **reference
sound** rather than by a physical unit, so the first thing to learn about each
is its anchor: the sound that reads exactly 1.

| Quantity | Unit | The sound that reads 1 | Criterion? | Page |
|---|---|---|---|---|
| Loudness | sone | 1 kHz tone at 40 dB SPL (also 40 phon) | none | [Loudness](psychoacoustics/loudness.md) |
| Sharpness | acum | critical-band-wide noise at 1 kHz, 60 dB SPL | none | [Sound Quality Metrics](psychoacoustics/sound-quality.md) |
| Roughness | asper | 1 kHz tone at 60 dB, fully modulated at 70 Hz | 0.2 asper (informative) | [Sound Quality Metrics](psychoacoustics/sound-quality.md) |
| Fluctuation strength | vacil | the same carrier modulated at 4 Hz | 0.2 vacil (informative) | [Sound Quality Metrics](psychoacoustics/sound-quality.md) |
| Tonality | tu_HMS | 1 kHz tone at 40 dB SPL | 0.4 tu_HMS (informative) | [Sound Quality Metrics](psychoacoustics/sound-quality.md) |
| Tone audibility | dB | — (a level difference above masking) | ISO 1996-2 adjustment | [Tone audibility](psychoacoustics/tone-audibility.md) |
| STI | 0 to 1 | — | Annex F letters, U to A+ | [Speech Transmission Index](speech/speech-transmission.md) |
| SII | 0 to 1 | — | none standardised | [Speech Intelligibility Index](speech/speech-intelligibility.md) |
| Threshold shift | dB HL | — (a difference of two hearing levels) | ISO 1999 statistics | [Noise-induced hearing loss](hearing/noise-induced-hearing-loss.md) |

Loudness, sharpness, roughness and fluctuation strength are **ratio scales
with no pass/fail line**: twice the number means twice the sensation, so a
20-sone appliance is heard as about twice as loud as a 10-sone one, which is
why appliance declarations set limits in sones rather than in decibels. The
tonal metrics and the speech indices do carry criteria, which is why the tone
pages end in a verdict and the loudness pages do not.

The three speech numbers all live in [0, 1] and are **not the same number**. An
STI of 0.6 falls in Annex F band D, typical of a good lecture room; an SII of
0.6 means roughly 60 % of the importance-weighted speech spectrum is audible;
and a STOI of 0.6 has no absolute meaning at all, because the mapping from
index to words understood is fitted per listening-test corpus, so STOI is only
ever read as a difference between two processors on the same material. Never
substitute one for another in a specification, and when a requirement quotes a
number, check which standard it belongs to before computing anything.

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

## What this section does not cover

**No listener is tested here, and no verdict about a person is issued.** Every
model on these pages predicts what a population, or a standard listener, would
perceive from a calibrated signal: none of them runs an audiometric session,
none diagnoses a hearing loss, and ISO 1999 explicitly declines to define a
hearing handicap or a compensable fence — that line is national regulation, and
nothing here applies one. Every prominence and audibility verdict is likewise
the numeric criterion only: ECMA-418-1 also requires aural confirmation of a
prominent tone, and that stays with you.

**Everything is monaural.** The binaural combinations of ECMA-418-2 are not
implemented, so a two-channel recording is analysed one ear at a time, and
nothing here models localisation, spatial release from masking or binaural
loudness summation.

**No listening test is replaced.** STOI returns the correlation-based index and
not a percentage of words understood, because that mapping is fitted per
listening-test corpus; the SII returns an audibility fraction and not a score;
and no page predicts annoyance in a community, which is a social-survey
quantity rather than a psychoacoustic one — the community indicators are
[Environment and transport](../environment/index.md).

Finally, these models start from a **calibrated** signal or spectrum in
pascals, because every one of them is level-dependent. Feeding them raw
soundcard samples produces a number with an arbitrary reference, which is a
different failure from a wrong answer: it looks plausible.

## Before and after these pages

Every model here consumes a calibrated signal or a calibrated spectrum, so the
calibration and weighting that produce one are in [Signal
analysis](../signals/index.md), and [Build a sound level
meter](../signals/sound-level-meter.md) runs that chain end to end on
one runnable page. The derivations are in [Perception and hearing
theory](../reference/theory/perception.md), from the equal-loudness contours to the
modulation transfer function.

If you arrived here from a search and want the shape of the whole library,
[What do you need to measure?](https://jmrplens.github.io/phonometry/start/tasks/) indexes it by the job
and [All guides](../README.md) lists every page with a line on
each.
