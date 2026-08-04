← [Documentation index](../../README.md)

# Psychoacoustics

Psychoacoustics replaces the question "how many decibels?" with "what does the
listener perceive?". Its base quantity is **loudness**: a perceptual magnitude
in sones, computed by auditory models that account for the ear's filtering,
masking and compression. On top of loudness sit the **sound quality**
sensations that distinguish two equally loud sounds: sharpness (high-frequency
emphasis), tonality (audible discrete tones), roughness (fast modulation)
and fluctuation strength (slow modulation). And on top sits a combined **annoyance** metric that weighs loudness,
sharpness, roughness and fluctuation strength into a single scalar.

[Loudness](loudness.md) is the foundation page: the Zwicker
reference method of ISO 532-1 with its one-page fiche, together with the
ISO 226:2023 equal-loudness contours that anchor the perceptual scale for pure
tones. The newer model families,
Moore-Glasberg per ISO 532-2/-3 and the Sottek Hearing Model of ECMA-418-2,
continue in
[Advanced Loudness](advanced-loudness.md), which also
carries the model-choice table.
[Sound Quality Metrics](sound-quality.md) adds
sharpness per DIN 45692 and the ECMA-418-2 tonality, roughness and
fluctuation strength that share the Sottek front-end.

Tones in noise get two dedicated pages because two different questions are
asked of them.
[Prominent Discrete Tones (ECMA-418-1)](tone-prominence.md)
answers a product-noise question: is this tone *prominent* by the
tone-to-noise and prominence-ratio criteria used in IT-equipment declarations?
[Objective audibility of tones in noise (ISO/PAS 20065)](tone-audibility.md)
answers an environmental one: by how many decibels does the tone exceed its
masking threshold, the audibility that feeds the tonal penalty of
ISO 1996-2.

[Psychoacoustic annoyance and fluctuation strength](psychoacoustic-annoyance.md)
closes the chain with the Fastl & Zwicker model, which combines loudness,
sharpness, roughness and the slow-modulation sensation of fluctuation strength
into a single annoyance value. Read it last: its four inputs all come from
the earlier pages.

## Pages in this section

- [Loudness](loudness.md): the ISO 532-1 Zwicker loudness in
  sones, plus the ISO 226:2023 equal-loudness contours.
- [Advanced Loudness (ISO 532-2/-3, ECMA-418-2)](advanced-loudness.md):
  the Moore-Glasberg and Sottek loudness models and the model-choice table.
- [Sound Quality Metrics](sound-quality.md): sharpness
  (DIN 45692) and ECMA-418-2 tonality, roughness and fluctuation strength.
- [Prominent Discrete Tones (ECMA-418-1)](tone-prominence.md):
  tone-to-noise and prominence ratios with prominence verdicts.
- [Objective audibility of tones in noise (ISO/PAS 20065)](tone-audibility.md):
  the audibility of a tone above the masking threshold, feeding the
  ISO 1996-2 tonal adjustment.
- [Psychoacoustic annoyance and fluctuation strength](psychoacoustic-annoyance.md):
  the Fastl & Zwicker annoyance model and the fluctuation-strength models it
  consumes.
