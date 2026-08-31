---
title: "Theory"
description: "Where the derivations live: six domain pages that map each shared method back to the clause, equation and table it comes from, and the areas whose theory stays inline with the guide that uses it."
---

The theory reference collects the derivations, clause references and design
decisions for the areas whose mathematics is **shared across many guides**. A
theory page maps an implemented method back to the clause, equation and table of
the standard or textbook it comes from, states the physics behind each
correction term and the assumptions that bound it, and gives the reference
values the validation suite checks against. It does not show workflows: that is
what the guides are for. The natural pattern is to arrive here *from* a guide
when a term needs justifying, or to read a domain page first when deciding
whether a method applies at all.

The six domains are not independent. **Signal analysis** underpins everything,
because every other page consumes band levels, weighting curves and time
integration from it; **perception and hearing** explains the curves several
other pages then use as weightings; **rooms and buildings**, **materials and
surfaces** and **environment and transport** are the three application domains,
and they share the sound-power and absorption machinery between them;
**vibration** supplies the structural quantities that panel and flanking
prediction rest on.

Several areas keep their theory **inside their guides** rather than here,
because the derivation and the single method it serves would otherwise be
separated for nothing. That is the case for the underwater modules
([Underwater Acoustics](/phonometry/underwater/underwater-acoustics/),
[Underwater Propagation](/phonometry/underwater/underwater-propagation/),
[Underwater Propagation Solvers](/phonometry/underwater/underwater-solvers/) and
[Marine-Mammal Noise Exposure](/phonometry/underwater/marine-mammal-exposure/));
for the aircraft certification and contour methods in
[Aircraft noise](/phonometry/aircraft/); for the CNOSSOS-EU road and railway
emission models in
[Environmental sources](/phonometry/environment/sources/); for the IEC 60268
electroacoustic measurements and the BS.1770 broadcast chain in
[Electroacoustics](/phonometry/devices/electroacoustics/) and
[Broadcast](/phonometry/devices/broadcast/); for the silencer, duct-path and
room-to-room models of [Noise control](/phonometry/devices/noise-control/); and
for the FDTD and elastic solvers, whose numerical method, stability bound and
dispersion rule are developed on the [wave simulation](/phonometry/simulation/)
pages themselves. Everything below is listed with the sections each domain page
hosts.

## [Signal Analysis](/phonometry/reference/theory/signal-analysis/)

Where the band edges, the filter magnitude responses and the weighting curves
come from, and the numerical reasons the bank is built as a decimated cascade of
second-order sections; also the time-integration, intensity and GUM uncertainty
derivations.

- [Octave Band Frequencies (ANSI S1.11 / IEC 61260)](/phonometry/reference/theory/signal-analysis/#octave-band-frequencies-ansi-s111--iec-61260)
- [Frequency Resolution vs FFT Bin Spacing](/phonometry/reference/theory/signal-analysis/#frequency-resolution-vs-fft-bin-spacing)
- [Magnitude Responses](/phonometry/reference/theory/signal-analysis/#magnitude-responses)
- [Filter Bank Design & Numerical Stability](/phonometry/reference/theory/signal-analysis/#filter-bank-design--numerical-stability)
- [Weighting Curves (IEC 61672-1)](/phonometry/reference/theory/signal-analysis/#weighting-curves-iec-61672-1)
- [Time Integration](/phonometry/reference/theory/signal-analysis/#time-integration)
- [G-weighting (ISO 7196)](/phonometry/reference/theory/signal-analysis/#g-weighting-iso-7196)
- [Event and dose metrics](/phonometry/reference/theory/signal-analysis/#event-and-dose-metrics)
- [Sound intensity (IEC 61043)](/phonometry/reference/theory/signal-analysis/#sound-intensity-iec-61043)
- [Measurement uncertainty (ISO/IEC Guide 98-3: GUM and Supplement 1)](/phonometry/reference/theory/signal-analysis/#measurement-uncertainty-isoiec-guide-98-3-gum-and-supplement-1)

## [Perception and Hearing](/phonometry/reference/theory/perception/)

The longest of the six: the equal-loudness contours, the excitation-pattern and
masking models behind loudness and sound quality, the modulation-transfer chain
of the STI and the band-importance and audibility construction of the SII, then
the hearing-threshold statistics and the damage model built on them.

- [Equal-loudness contours (ISO 226:2023)](/phonometry/reference/theory/perception/#equal-loudness-contours-iso-2262023)
- [Zwicker loudness (ISO 532-1)](/phonometry/reference/theory/perception/#zwicker-loudness-iso-532-1)
- [Advanced loudness models & sound quality](/phonometry/reference/theory/perception/#advanced-loudness-models--sound-quality)
- [Tone prominence: TNR and PR (ECMA-418-1)](/phonometry/reference/theory/perception/#tone-prominence-tnr-and-pr-ecma-418-1)
- [Modulation transfer and STI (IEC 60268-16)](/phonometry/reference/theory/perception/#modulation-transfer-and-sti-iec-60268-16)
- [Speech Intelligibility Index (ANSI S3.5)](/phonometry/reference/theory/perception/#speech-intelligibility-index-ansi-s35)
- [Hearing thresholds and presbycusis (ISO 389-7, ISO 7029)](/phonometry/reference/theory/perception/#hearing-thresholds-and-presbycusis-iso-389-7-iso-7029)
- [Noise-induced hearing loss (ISO 1999)](/phonometry/reference/theory/perception/#noise-induced-hearing-loss-iso-1999)

## [Rooms and Buildings](/phonometry/reference/theory/rooms-buildings/)

Five groups: the ANSI S12.2 criterion curves, the impulse response and the
parameters read off it, sound insulation and absorption as measured, the same
quantities as predicted, and the uncertainty that qualifies either.

- [Room noise criteria (ANSI S12.2)](/phonometry/reference/theory/rooms-buildings/#room-noise-criteria-ansi-s122)
- [Impulse response and room-acoustic parameters (ISO 18233, ISO 3382-1/-2/-3)](/phonometry/reference/theory/rooms-buildings/#impulse-response-and-room-acoustic-parameters-iso-18233-iso-3382-1-2-3)
  - [Deterministic-excitation impulse response (ISO 18233)](/phonometry/reference/theory/rooms-buildings/#deterministic-excitation-impulse-response-iso-18233)
  - [Schroeder backward integration (ISO 3382-1, 5.3.3)](/phonometry/reference/theory/rooms-buildings/#schroeder-backward-integration-iso-3382-1-533)
  - [Regression windows and validity (ISO 3382-2, Clause 6, Annex B/C)](/phonometry/reference/theory/rooms-buildings/#regression-windows-and-validity-iso-3382-2-clause-6-annex-bc)
  - [Clarity, definition and centre time (ISO 3382-1, Annex A)](/phonometry/reference/theory/rooms-buildings/#clarity-definition-and-centre-time-iso-3382-1-annex-a)
  - [Open-plan spatial decay (ISO 3382-3, Clause 6)](/phonometry/reference/theory/rooms-buildings/#open-plan-spatial-decay-iso-3382-3-clause-6)
- [Sound insulation and absorption, measured (ISO 16283, ISO 10140, ISO 717, ISO 354)](/phonometry/reference/theory/rooms-buildings/#sound-insulation-and-absorption-measured-iso-16283-iso-10140-iso-717-iso-354)
  - [Field insulation and weighted rating (ISO 16283-1, ISO 717-1)](/phonometry/reference/theory/rooms-buildings/#field-insulation-and-weighted-rating-iso-16283-1-iso-717-1)
  - [Impact insulation (ISO 16283-2, ISO 717-2)](/phonometry/reference/theory/rooms-buildings/#impact-insulation-iso-16283-2-iso-717-2)
  - [Absorption in a reverberation room (ISO 354)](/phonometry/reference/theory/rooms-buildings/#absorption-in-a-reverberation-room-iso-354)
  - [Laboratory vs field normalization (ISO 10140, ISO 16283)](/phonometry/reference/theory/rooms-buildings/#laboratory-vs-field-normalization-iso-10140-iso-16283)
- [Sound insulation and absorption, predicted (EN 12354-1/-2/-6, Bies, Cremer, Hopkins)](/phonometry/reference/theory/rooms-buildings/#sound-insulation-and-absorption-predicted-en-12354-1-2-6-bies-cremer-hopkins)
  - [Flanking transmission prediction (EN 12354-1/2)](/phonometry/reference/theory/rooms-buildings/#flanking-transmission-prediction-en-12354-12)
  - [Absorption in enclosed spaces (EN 12354-6)](/phonometry/reference/theory/rooms-buildings/#absorption-in-enclosed-spaces-en-12354-6)
  - [Predicted panel sound insulation (Bies 7.2, Hopkins 2.9/4.3.10, Cremer 5)](/phonometry/reference/theory/rooms-buildings/#predicted-panel-sound-insulation-bies-72-hopkins-294310-cremer-5)
- [Measurement uncertainty (ISO 12999-1)](/phonometry/reference/theory/rooms-buildings/#measurement-uncertainty-iso-12999-1)

## [Materials and Surfaces](/phonometry/reference/theory/materials-surfaces/)

The characterisation standards rather than the prediction models: what a
scattering coefficient and a diffusion coefficient each measure and why they
must not be swapped, how an in-situ road measurement separates the reflection in
time, and the definitions behind the laboratory absorption and impedance
quantities.

- [Surface scattering and diffusion (ISO 17497-1, ISO 17497-2)](/phonometry/reference/theory/materials-surfaces/#surface-scattering-and-diffusion-iso-17497-1-iso-17497-2)
- [In-situ road surface absorption (ISO 13472-1, ISO 13472-2)](/phonometry/reference/theory/materials-surfaces/#in-situ-road-surface-absorption-iso-13472-1-iso-13472-2)
  - [Extended surface method (ISO 13472-1)](/phonometry/reference/theory/materials-surfaces/#extended-surface-method-iso-13472-1)
  - [Spot method (ISO 13472-2)](/phonometry/reference/theory/materials-surfaces/#spot-method-iso-13472-2)
- [Acoustic material characterisation (ISO 11654, ISO 9053-1/2, ISO 10534-1/2, ASTM E2611)](/phonometry/reference/theory/materials-surfaces/#acoustic-material-characterisation-iso-11654-iso-9053-12-iso-10534-12-astm-e2611)

## [Environment and Transport](/phonometry/reference/theory/environment-transport/)

The descriptors and the attenuation terms: how the ISO 1996-1 indicators are
built, what the NT ACOU 112 prominence criterion measures, where each ISO 9613
term comes from, and — filed here rather than under devices, because the
mathematics is the same — the sound-power determination derivations and the
ISO 9612 occupational-exposure uncertainty.

- [Environmental descriptors (ISO 1996-1)](/phonometry/reference/theory/environment-transport/#environmental-descriptors-iso-1996-1)
- [Impulsive-sound prominence (NT ACOU 112)](/phonometry/reference/theory/environment-transport/#impulsive-sound-prominence-nt-acou-112)
- [Outdoor propagation (ISO 9613-1/2)](/phonometry/reference/theory/environment-transport/#outdoor-propagation-iso-9613-12)
- [Occupational noise exposure (ISO 9612)](/phonometry/reference/theory/environment-transport/#occupational-noise-exposure-iso-9612)
- [Sound power determination (ISO 3744/3745/3746, ISO 3741, ISO 9614-1/2/3)](/phonometry/reference/theory/environment-transport/#sound-power-determination-iso-374437453746-iso-3741-iso-9614-123)

## [Vibration](/phonometry/reference/theory/vibration/)

The shortest: the human-vibration weightings and dose measures with the
ISO 2631-5 spinal model, plus the point-mobility and radiation-efficiency
results the structure-borne pages use.

- [Human vibration (ISO 8041-1, ISO 2631-1/2, ISO 5349-1/2, Directive 2002/44/EC)](/phonometry/reference/theory/vibration/#human-vibration-iso-8041-1-iso-2631-12-iso-5349-12-directive-200244ec)
  - [Getting the acceleration right](/phonometry/reference/theory/vibration/#getting-the-acceleration-right)
  - [Weighted metrics and daily exposure](/phonometry/reference/theory/vibration/#weighted-metrics-and-daily-exposure)
- [Multiple shocks (ISO 2631-5)](/phonometry/reference/theory/vibration/#multiple-shocks-iso-2631-5)
- [Point mobilities and radiation efficiency (Cremer 5, Hopkins 2.9)](/phonometry/reference/theory/vibration/#point-mobilities-and-radiation-efficiency-cremer-5-hopkins-29)
