← [Documentation index](../../README.md)

# Theoretical Background

The theory reference collects the derivations, clause references and design decisions for the areas whose mathematics is **shared across many guides**. A theory page maps an implemented method back to the clause, equation and table of the standard or textbook it comes from, states the physics behind each correction term and the assumptions that bound it, and gives the reference values the validation suite checks against. It does not show workflows: that is what the guides are for.

Several areas keep their theory **inside their guides** instead, because the derivation and the single method it serves would otherwise be separated for nothing: the underwater modules ([Underwater Acoustics](../../underwater/underwater-acoustics.md), [Underwater Propagation](../../underwater/underwater-propagation.md), [Underwater Propagation Solvers](../../underwater/underwater-solvers.md) and [Marine-Mammal Noise Exposure](../../underwater/marine-mammal-exposure.md)), the aircraft certification and contour methods ([Aircraft noise](../../aircraft/index.md)), the CNOSSOS-EU road and railway emission models ([Environment and transport](../../environment/index.md)), the IEC 60268 electroacoustic measurements, the BS.1770 broadcast chain and the noise-control models ([Sources and devices](../../devices/index.md)), and the FDTD and elastic solvers ([Wave simulation](../../simulation/index.md)). Everything below is listed with the sections each domain page hosts.

## [Signal Analysis](signal-analysis.md)

- [Octave Band Frequencies (ANSI S1.11 / IEC 61260)](signal-analysis.md#octave-band-frequencies-ansi-s111--iec-61260)
- [Frequency Resolution vs FFT Bin Spacing](signal-analysis.md#frequency-resolution-vs-fft-bin-spacing)
- [Magnitude Responses |H(jw)|](signal-analysis.md#magnitude-responses-hjw)
- [Filter Bank Design & Numerical Stability](signal-analysis.md#filter-bank-design--numerical-stability)
- [Weighting Curves (IEC 61672-1)](signal-analysis.md#weighting-curves-iec-61672-1)
- [Time Integration](signal-analysis.md#time-integration)
- [G-weighting (ISO 7196)](signal-analysis.md#g-weighting-iso-7196)
- [Event and dose metrics](signal-analysis.md#event-and-dose-metrics)
- [Sound intensity (IEC 61043)](signal-analysis.md#sound-intensity-iec-61043)
- [Measurement uncertainty (ISO/IEC Guide 98-3: GUM and Supplement 1)](signal-analysis.md#measurement-uncertainty-isoiec-guide-98-3-gum-and-supplement-1)

## [Perception and Hearing](perception.md)

- [Equal-loudness contours (ISO 226:2023)](perception.md#equal-loudness-contours-iso-2262023)
- [Tone prominence: TNR and PR (ECMA-418-1)](perception.md#tone-prominence-tnr-and-pr-ecma-418-1)
- [Zwicker loudness (ISO 532-1)](perception.md#zwicker-loudness-iso-532-1)
- [Advanced loudness models & sound quality](perception.md#advanced-loudness-models--sound-quality)
- [Modulation transfer and STI (IEC 60268-16)](perception.md#modulation-transfer-and-sti-iec-60268-16)
- [Speech Intelligibility Index (ANSI S3.5)](perception.md#speech-intelligibility-index-ansi-s35)
- [Hearing thresholds and presbycusis (ISO 389-7, ISO 7029)](perception.md#hearing-thresholds-and-presbycusis-iso-389-7-iso-7029)
- [Noise-induced hearing loss (ISO 1999)](perception.md#noise-induced-hearing-loss-iso-1999)

## [Rooms and Buildings](rooms-buildings.md)

- [Room noise criteria (ANSI S12.2)](rooms-buildings.md#room-noise-criteria-ansi-s122)
- [Room and building acoustics (ISO 18233, ISO 3382, ISO 16283, ISO 10140, EN 12354, ISO 12999, ISO 717, ISO 354)](rooms-buildings.md#room-and-building-acoustics-iso-18233-iso-3382-iso-16283-iso-10140-en-12354-iso-12999-iso-717-iso-354)

## [Materials and Surfaces](materials-surfaces.md)

- [Surface scattering and diffusion (ISO 17497-1, ISO 17497-2)](materials-surfaces.md#surface-scattering-and-diffusion-iso-17497-1-iso-17497-2)
- [In-situ road surface absorption (ISO 13472-1, ISO 13472-2)](materials-surfaces.md#in-situ-road-surface-absorption-iso-13472-1-iso-13472-2)
- [Acoustic material characterisation (ISO 11654, ISO 9053-1/2, ISO 10534-1/2, ASTM E2611)](materials-surfaces.md#acoustic-material-characterisation-iso-11654-iso-9053-12-iso-10534-12-astm-e2611)

## [Environment and Transport](environment-transport.md)

- [Environmental descriptors (ISO 1996-1)](environment-transport.md#environmental-descriptors-iso-1996-1)
- [Impulsive-sound prominence (NT ACOU 112)](environment-transport.md#impulsive-sound-prominence-nt-acou-112)
- [Outdoor propagation and occupational exposure (ISO 9613-1/2, ISO 9612)](environment-transport.md#outdoor-propagation-and-occupational-exposure-iso-9613-12-iso-9612)
- [Sound power determination (ISO 3744/3745/3746, ISO 3741, ISO 9614-2/3)](environment-transport.md#sound-power-determination-iso-374437453746-iso-3741-iso-9614-23)

## [Vibration](vibration.md)

- [Human vibration (ISO 8041-1, ISO 2631-1/2, ISO 5349-1/2, Directive 2002/44/EC)](vibration.md#human-vibration-iso-8041-1-iso-2631-12-iso-5349-12-directive-200244ec)
- [Multiple shocks (ISO 2631-5)](vibration.md#multiple-shocks-iso-2631-5)
- [Point mobilities and radiation efficiency (Cremer 5, Hopkins 2.9)](vibration.md#point-mobilities-and-radiation-efficiency-cremer-5-hopkins-29)
