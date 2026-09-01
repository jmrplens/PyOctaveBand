<a href="https://jmrplens.github.io/phonometry/"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/brand/banner.webp" alt="phonometry, acoustic measurement toolkit for Python" width="100%"></a>

<!-- Package -->
[![PyPI version](https://img.shields.io/pypi/v/phonometry?logo=pypi&logoColor=white)](https://pypi.org/project/phonometry/)
[![Python versions](https://img.shields.io/pypi/pyversions/phonometry?logo=python&logoColor=white)](https://pypi.org/project/phonometry/)
[![PyPI downloads](https://img.shields.io/pypi/dm/phonometry?logo=pypi&logoColor=white)](https://pypistats.org/packages/phonometry)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/jmrplens/phonometry/blob/main/LICENSE)

<!-- Quality -->
[![CI](https://github.com/jmrplens/phonometry/actions/workflows/python-app.yml/badge.svg)](https://github.com/jmrplens/phonometry/actions/workflows/python-app.yml)
[![Quality Gate](https://sonarcloud.io/api/project_badges/measure?project=jmrplens_phonometry&metric=alert_status)](https://sonarcloud.io/summary/overall?id=jmrplens_phonometry)
[![codecov](https://codecov.io/gh/jmrplens/phonometry/branch/main/graph/badge.svg)](https://codecov.io/gh/jmrplens/phonometry)

<!-- Citation & support -->
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21215280-blue?logo=doi&logoColor=white)](https://doi.org/10.5281/zenodo.21215280)

# phonometry

> *phonometry*, the measurement of sound. Formerly published as **PyOctaveBand**.

Acoustic measurement toolkit for Python, from fractional octave-band filters,
weighting and sound level metrology to psychoacoustics, rooms and buildings,
materials, vibration, environmental, aircraft and underwater acoustics,
electroacoustics and wave simulation. Every metric is implemented from its
governing standard and numerically checked against it: the auto-generated
[conformance report](https://github.com/jmrplens/phonometry/blob/main/docs/CONFORMANCE.md)
runs 581 conformance checks across 59 domains and 374 standards, each pinning
an expected normative value to the value the library computes, and CI
regenerates it on every pull request. Filters are class 1 per
**IEC 61260-1:2014 / ANSI S1.11-2004** and weightings and levels class 1 per
**IEC 61672-1:2013**.

<a href="https://github.com/jmrplens/phonometry/blob/main/docs/CONFORMANCE.md"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/badges/conformance-summary_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/badges/conformance-summary.svg" alt="All 581 conformance checks pass, across 59 domains and 374 standards"></picture></a>

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_fdtd_pillar_hall_dark.gif"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_fdtd_pillar_hall.gif" alt="Animation: an 800 Hz plane wavefront sweeps through a hall of rigid columns in a 2D FDTD simulation; every column diffracts the front and the scattered wavelets interfere until the whole hall is filled" width="100%"></picture>

*An 800 Hz wavefront threading a hall of columns, computed with the library's
own [2D FDTD engine](https://github.com/jmrplens/phonometry/blob/main/docs/simulation/fdtd-simulation.md)
(`phonometry.simulation`) and rendered by the same script that generates every
figure in the documentation.*

## 🚀 Installation

```bash
pip install phonometry
```

Optional extras: `phonometry[plot]` (matplotlib for response plots and result
`.plot()` methods), `phonometry[perf]` (numba for faster impulse ballistics),
`phonometry[report]` (reportlab and svglib, so result `.report()` methods can
render normative PDF fiches, whose figure panel also needs matplotlib),
`phonometry[audio]` (python-soundfile, so `phonometry.io` also reads FLAC,
AIFF, Ogg/Opus, MP3 and compressed WAV), `phonometry[full]` (all of the
above).

I recommend `pip install phonometry[full]`: it brings matplotlib, numba,
reportlab, svglib and soundfile in one go, so every feature of the library is
enabled. The base install computes every metric on NumPy and SciPy alone — and
that includes reading every linear measurement WAV (24-bit, multichannel
EXTENSIBLE, RF64) through `phonometry.io`; the only things it leaves
unavailable are the figures (`.plot()` and the filter response plots), the
normative PDF fiches (`.report()`), the compiled kernel that speeds up the
`impulse` time weighting, and the compressed audio formats. One licensing note
on `[audio]`: it installs python-soundfile, whose wheel bundles libsndfile
under the LGPL-2.1 (dynamically linked); the base install deliberately stays
free of it.

One caveat about `[full]`: numba is the only extra that caps NumPy, and it
raises that cap only once it supports a new NumPy minor. So in the weeks after
a NumPy minor release, `phonometry[full]` (like `phonometry[perf]`) can resolve
one minor behind what a plain install gets. numba only makes the `impulse` time
weighting faster, so if you need the newest NumPy the day it ships, install
`phonometry[plot,report]` and leave `[perf]` out.

## ⚡ Quick start

```python
import numpy as np
from phonometry import filters

fs = 48000
t = np.linspace(0, 1, fs, endpoint=False)
# Composite signal: 100Hz + 1000Hz
signal = np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 1000 * t)

# Apply 1/3 octave filter bank
spl, freq = filters.octave_filter(signal, fs=fs, fraction=3)

print(f"Bands: {freq}")
print(f"SPL [dB]: {spl}")
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/signal_response_fraction_3_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/signal_response_fraction_3.svg" alt="One-third-octave spectrum analysis of a multi-tone signal with the raw PSD in the background" width="80%"></picture>

*1/3 octave band spectrum analysis of a complex signal. More examples in the
[documentation](https://jmrplens.github.io/phonometry/).*

## ✨ What's inside

The library is organized into domain namespaces, and a function is reached
through the one that owns it. Reading the domain at the call site is the point:
two packages can hold a `transmission_loss` without either being renamed, and
the line says which one it means.

```python
from phonometry import building, underwater

r = building.airborne_insulation(...)
pl = underwater.propagation_loss(...)
```

| Namespace | Coverage |
| :--- | :--- |
| `filters` | 1/1, 1/3 and arbitrary fractional octave filter banks (stable SOS + multirate decimation) in five architectures with per-band class verdicts (IEC 61260-1 / ANSI S1.11); A/C/Z weighting within IEC 61672-1 class 1 tolerances plus G weighting (ISO 7196); Fast/Slow/Impulse ballistics; octave spectrogram and zero-phase filtering; RBJ parametric equalizer sections |
| `signals` | Leq, SEL, L10/L50/L90 and noise dose (IEC 61252); calibrated Welch PSD/CSD with chi-square confidence intervals, coherent output spectrum, 1/n-octave smoothing and colored-noise generators (Bendat & Piersol); MISO multiple/partial coherence; correlation and GCC time-delay estimation (Knapp & Carter); Hilbert envelope, cepstrum and echoes, time synchronous averaging, calibrated STFT and zoom FFT; regularized inverse filtering for system measurement; IEC 60268-1 tone bursts and resampling |
| `metrology` | Physical SPL calibration with IEC 60942 stability validation and dBFS modes; GUM uncertainty with Monte Carlo (ISO/IEC Guide 98-3 and Supplement 1); Bendat & Piersol data qualification (stationarity, trends, level crossings, peak statistics); IEC 61043 intensity-instrument class verification |
| `io` | Measurement audio files: every linear WAV a meter writes (PCM at any depth, EXTENSIBLE, RF64/BW64) read into a calibrated `Signal` with its `bext` provenance (EBU Tech 3285); headers-only `info()`; block streaming into the stateful filters; BWF writing with exact codes, loud clipping, optional TPDF dither and measured R 128 loudness; the calibration sidecar; lossless conversion to and from FLAC with provenance intact |
| `psychoacoustics` | Loudness in sones three ways: Zwicker (ISO 532-1 Annex B validated), Moore-Glasberg stationary and time-varying (ISO 532-2/3) and Sottek Hearing Model (ECMA-418-2); DIN 45692 sharpness; ECMA-418-2 tonality, roughness (asper) and fluctuation strength (vacil_HMS); tone prominence TNR/PR (ECMA-418-1); tonal audibility (ISO/PAS 20065); Fastl & Zwicker psychoacoustic annoyance; ISO 226:2023 contours |
| `speech` | Speech Transmission Index STI/STIPA with signal generator (IEC 60268-16 Ed. 5); Speech Intelligibility Index (ANSI S3.5-1997) with the four band-importance procedures and the standard speech spectra; STOI and ESTOI |
| `hearing` | Age-related thresholds (ISO 7029) and reference thresholds (ISO 389-7); noise-induced hearing loss with HTLAN (ISO 1999); daily noise exposure LEX,8h with Annex C uncertainty (ISO 9612) |
| `room` | Swept-sine/MLS/Golay impulse responses (ISO 18233); EDT/T20/T30/C50/C80/Ts (ISO 3382-1/2); open-plan speech metrics (ISO 3382-3); reverberation-room absorption (ISO 354); reverberation-time prediction (Sabine to Arau-Puchades); total absorption of furnished rooms (EN 12354-6); image-source impulse responses and the steady-state field; room-noise criteria NC and RC Mark II (ANSI/ASA S12.2) |
| `building` | Measurement: field airborne, impact and façade insulation with R′w/DnT,w/L′nT,w/D2m,nT,w and C/Ctr/CI (ISO 16283-1/2/3, ISO 717-1/2), laboratory R/Ln (ISO 10140), survey method (ISO 10052), intensity method (ISO 15186), laboratory flanking (ISO 10848), heavy impact sources, floor-covering improvement (ISO 16251-1), reception-plate power (EN 15657) and measurement uncertainty (ISO 12999-1). Prediction: EN 12354-1/2 global and detailed models, façade and outdoor radiation (EN 12354-3/4), installed structure-borne sources (EN 12354-5), panel transmission theory (mass law, coincidence, double walls, slits and apertures), ceiling plenums, masonry cavity walls and resilient layers. Regulation: the Spanish CTE DB-HR |
| `materials` | Absorbers: ratings αw with classes (ISO 11654) and uncertainty (ISO 12999-2), impedance-tube absorption, impedance and transmission loss (ISO 10534-1/2, ASTM E2611) plus a virtual FDTD tube, porous and multilayer models (Delany-Bazley, Miki, JCA, TMM with MPP and membranes), Biot poroelasticity and slow-sound metamaterial absorbers at critical coupling. Diffusers: scattering and diffusion coefficients (ISO 17497-1/2), Schroeder diffuser design and far-field prediction, deep-subwavelength metadiffusers. Surfaces: in-situ road-surface absorption (ISO 13472-1/2). Resilient: dynamic stiffness of layers under floating floors (EN 29052-1) |
| `emission` | Sound power by enveloping surface (ISO 3744/3746), reverberation room (ISO 3741), precision anechoic rooms (ISO 3745) and intensity scanning with field indicators and grades (ISO 9614-2/3) and the discrete-point power summation of ISO 9614-1; two-microphone p-p intensity (IEC 61043); sound power from surface vibration (ISO/TS 7849); noise-emission declarations (ISO 4871) |
| `environment` | Sources: CNOSSOS-EU road and rail emission, wind-turbine apparent sound power and tonal audibility (IEC 61400-11). Propagation: atmospheric absorption (ISO 9613-1) and the ISO 9613-2 general method with per-term octave breakdown, spherical ground effect and wave-theoretic barriers, refraction ray tracing and the GFPE. Assessment: rating levels, Lden/Ldn and adjustments (ISO 1996-1/2), impulsive-sound prominence (NT ACOU 112) and the Spanish RD 1367/2007 |
| `aircraft` | EPNL certification chain (ICAO Annex 16) with IEC 61265 verification and SAE ARP 5534 absorption; airport noise contours (ECAC Doc 29) with the EASA ANP fleet database; rotorcraft hemisphere method (ECAC Doc 32) |
| `underwater` | Levels re 1 µPa (ISO 18405); ship radiated noise (ISO 17208-1/2); pile driving (ISO 18406); ship-traffic source levels (JOMOPANS-ECHO) and Wenz ambient noise; sonar equation and detection range; sound speed and seabed reflection; propagation loss from spreading laws and Weston's shallow-water regimes to normal-mode, ray, Gaussian-beam and parabolic-equation solvers; marine-mammal audiograms and regulatory auditory weighting (NMFS 2024/2018, Southall et al. 2019) |
| `vibration` | Human vibration: weightings (ISO 8041-1), whole-body metrics and buildings (ISO 2631-1/2), multiple shocks (ISO 2631-5), hand-arm and A(8) (ISO 5349); mobility and the FRF family (ISO 7626); isolator transfer stiffness (ISO 10846); plate-junction transmission and Kij; radiation efficiency and point mobilities |
| `electroacoustics` | Distortion per IEC 60268-3: THD, THD+N and SINAD (AES17), SMPTE/CCIF intermodulation and DIM; swept-sine harmonic separation and THD(f) (Farina, Novak synchronized sweep); frequency response and coherence; rigid-piston radiation; loudspeaker and microphone rated characteristics (IEC 60268-5/-4) |
| `noise_control` | Reactive silencers by the four-pole transmission-matrix method; HVAC duct elements and flow noise; machine enclosures |
| `broadcast` | ITU-R BS.1770-5 programme loudness and true peak in dBTP; EBU R 128 with the Tech 3341 EBU Mode meters and Tech 3342 loudness range, validated against the official EBU signals |
| `simulation` | Deterministic 2D acoustic FDTD with sources, probes, rasterised obstacles and rigid/impedance/absorbing boundaries; elastic P-SV solver with free surfaces and fluid-solid interfaces; near-to-far-field transform |

Cross-cutting, everywhere in the library:

- 📄 Typed, frozen result dataclasses with `.plot(language="en"|"es")`
  figures, and normative `.report()` PDF fiches for the metrics with a
  standardized reporting format (ISO 717, ISO 11654, ISO 532-1, EBU R 128,
  ICAO EPNL, IEC 61260-1, ISO 4871, IEC 60268-5/-4)
- ⚡ Vectorized multichannel processing and stateful block (real-time)
  workflows
- 🌐 Documentation fully in English and Spanish

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/filter_type_comparison_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/filter_type_comparison.svg" alt="Magnitude response comparison of the five filter architectures for the 1 kHz octave band, with a zoom at the -3 dB crossover" width="80%"></picture>

*The five filter architectures on the 1 kHz octave band, with the −3 dB
points on the ANSI band edges.*

## 📚 Documentation

**Full documentation website: https://jmrplens.github.io/phonometry/**
(English / Español)

The same content is browsable as Markdown in
[docs/](https://github.com/jmrplens/phonometry/blob/main/docs/README.md);
a map of where to start:

| Area | Guides |
| :--- | :--- |
| Getting started | [Getting Started](https://github.com/jmrplens/phonometry/blob/main/docs/start/getting-started.md) · [Build a sound level meter](https://github.com/jmrplens/phonometry/blob/main/docs/signals/sound-level-meter.md) · [Calibration and dBFS](https://github.com/jmrplens/phonometry/blob/main/docs/signals/metrology/calibration.md) |
| Filters, levels & weighting | [Filter Banks](https://github.com/jmrplens/phonometry/blob/main/docs/signals/filters/filter-banks.md) · [Filter Gallery](https://github.com/jmrplens/phonometry/blob/main/docs/signals/filters/filter-gallery.md) · [Levels](https://github.com/jmrplens/phonometry/blob/main/docs/signals/levels/levels.md) · [Environmental Levels](https://github.com/jmrplens/phonometry/blob/main/docs/environment/assessment/environmental-levels.md) · [Spanish Noise Regulation](https://github.com/jmrplens/phonometry/blob/main/docs/environment/assessment/spanish-noise-regulation.md) · [Frequency Weighting](https://github.com/jmrplens/phonometry/blob/main/docs/signals/levels/weighting.md) · [Special Weightings](https://github.com/jmrplens/phonometry/blob/main/docs/signals/levels/special-weightings.md) · [Time Weighting](https://github.com/jmrplens/phonometry/blob/main/docs/signals/levels/time-weighting.md) · [Block Processing](https://github.com/jmrplens/phonometry/blob/main/docs/signals/filters/block-processing.md) · [Multichannel](https://github.com/jmrplens/phonometry/blob/main/docs/signals/filters/multichannel.md) |
| Signal analysis | [Calibrated spectral analysis](https://github.com/jmrplens/phonometry/blob/main/docs/signals/spectra/spectral-analysis.md) · [Correlation, time delay & envelope](https://github.com/jmrplens/phonometry/blob/main/docs/signals/spectra/correlation-delay.md) · [Time-frequency analysis](https://github.com/jmrplens/phonometry/blob/main/docs/signals/spectra/time-frequency.md) · [System measurement](https://github.com/jmrplens/phonometry/blob/main/docs/signals/spectra/system-measurement.md) · [Measurement uncertainty](https://github.com/jmrplens/phonometry/blob/main/docs/signals/metrology/gum-uncertainty.md) |
| Psychoacoustics & sound quality | [Loudness](https://github.com/jmrplens/phonometry/blob/main/docs/perception/psychoacoustics/loudness.md) · [Advanced Loudness](https://github.com/jmrplens/phonometry/blob/main/docs/perception/psychoacoustics/advanced-loudness.md) · [Sound Quality Metrics](https://github.com/jmrplens/phonometry/blob/main/docs/perception/psychoacoustics/sound-quality.md) · [Tone Prominence](https://github.com/jmrplens/phonometry/blob/main/docs/perception/psychoacoustics/tone-prominence.md) · [Psychoacoustic annoyance](https://github.com/jmrplens/phonometry/blob/main/docs/perception/psychoacoustics/psychoacoustic-annoyance.md) |
| Speech & hearing | [Speech Transmission Index](https://github.com/jmrplens/phonometry/blob/main/docs/perception/speech/speech-transmission.md) · [Speech Intelligibility Index](https://github.com/jmrplens/phonometry/blob/main/docs/perception/speech/speech-intelligibility.md) · [Objective intelligibility](https://github.com/jmrplens/phonometry/blob/main/docs/perception/speech/objective-intelligibility.md) · [Noise-induced hearing loss](https://github.com/jmrplens/phonometry/blob/main/docs/perception/hearing/noise-induced-hearing-loss.md) · [Occupational exposure](https://github.com/jmrplens/phonometry/blob/main/docs/perception/hearing/occupational-exposure.md) |
| Rooms & buildings | [Room Acoustics](https://github.com/jmrplens/phonometry/blob/main/docs/buildings/rooms/room-acoustics.md) · [Field Insulation](https://github.com/jmrplens/phonometry/blob/main/docs/buildings/insulation/insulation-field.md) · [Low-Frequency Procedure (ISO 16283)](https://github.com/jmrplens/phonometry/blob/main/docs/buildings/insulation/low-frequency-procedure.md) · [Laboratory Insulation](https://github.com/jmrplens/phonometry/blob/main/docs/buildings/insulation/insulation-lab.md) · [Predicting Insulation (EN 12354)](https://github.com/jmrplens/phonometry/blob/main/docs/buildings/design/insulation-prediction.md) · [Image Sources](https://github.com/jmrplens/phonometry/blob/main/docs/buildings/rooms/room-image-sources.md) · [Impulse Response (ISO 18233)](https://github.com/jmrplens/phonometry/blob/main/docs/buildings/rooms/room-impulse-response.md) · [Open-Plan Offices](https://github.com/jmrplens/phonometry/blob/main/docs/buildings/rooms/open-plan-acoustics.md) · [Insulation Ratings (ISO 717)](https://github.com/jmrplens/phonometry/blob/main/docs/buildings/insulation/insulation-ratings.md) · [Façade Insulation](https://github.com/jmrplens/phonometry/blob/main/docs/buildings/insulation/facade-insulation.md) · [Spanish Building Code (CTE DB-HR)](https://github.com/jmrplens/phonometry/blob/main/docs/buildings/insulation/spanish-building-code.md) · [Survey Method (ISO 10052)](https://github.com/jmrplens/phonometry/blob/main/docs/buildings/insulation/insulation-survey.md) · [Insulation by Intensity](https://github.com/jmrplens/phonometry/blob/main/docs/buildings/insulation/insulation-intensity.md) · [Impact Improvement](https://github.com/jmrplens/phonometry/blob/main/docs/buildings/design/impact-improvement.md) · [Flanking (ISO 10848)](https://github.com/jmrplens/phonometry/blob/main/docs/buildings/insulation/flanking-lab.md) |
| Materials & surfaces | [Porous Absorbers](https://github.com/jmrplens/phonometry/blob/main/docs/materials/absorbers/porous-absorbers.md) · [Metamaterial Absorbers](https://github.com/jmrplens/phonometry/blob/main/docs/materials/absorbers/metamaterial-absorbers.md) · [Diffusers](https://github.com/jmrplens/phonometry/blob/main/docs/materials/diffusers/diffusers.md) · [Metadiffusers](https://github.com/jmrplens/phonometry/blob/main/docs/materials/diffusers/metadiffusers.md) · [Impedance Tube](https://github.com/jmrplens/phonometry/blob/main/docs/materials/absorbers/impedance-tube.md) · [Absorption Measurement](https://github.com/jmrplens/phonometry/blob/main/docs/materials/absorbers/absorption-measurement.md) · [Airflow Resistance](https://github.com/jmrplens/phonometry/blob/main/docs/materials/absorbers/airflow-resistance.md) · [Road Absorption](https://github.com/jmrplens/phonometry/blob/main/docs/materials/surfaces/road-absorption.md) |
| Environment & outdoors | [Outdoor Propagation](https://github.com/jmrplens/phonometry/blob/main/docs/environment/propagation/outdoor-propagation.md) · [CNOSSOS-EU Road Emission](https://github.com/jmrplens/phonometry/blob/main/docs/environment/sources/cnossos-road-emission.md) · [Ground & Barriers](https://github.com/jmrplens/phonometry/blob/main/docs/environment/propagation/ground-barriers.md) · [Atmospheric Refraction](https://github.com/jmrplens/phonometry/blob/main/docs/environment/propagation/atmospheric-refraction.md) |
| Transport | [Aircraft Noise](https://github.com/jmrplens/phonometry/blob/main/docs/aircraft/aircraft-noise.md) · [Rotorcraft Noise](https://github.com/jmrplens/phonometry/blob/main/docs/aircraft/rotorcraft-noise.md) · [Wind-Turbine Noise](https://github.com/jmrplens/phonometry/blob/main/docs/environment/sources/wind-turbine-noise.md) · [Airport Noise](https://github.com/jmrplens/phonometry/blob/main/docs/aircraft/airport-noise.md) |
| Underwater | [Underwater Acoustics](https://github.com/jmrplens/phonometry/blob/main/docs/underwater/underwater-acoustics.md) · [Underwater Propagation](https://github.com/jmrplens/phonometry/blob/main/docs/underwater/underwater-propagation.md) · [Underwater Solvers](https://github.com/jmrplens/phonometry/blob/main/docs/underwater/underwater-solvers.md) · [Marine-Mammal Exposure](https://github.com/jmrplens/phonometry/blob/main/docs/underwater/marine-mammal-exposure.md) |
| Vibration | [Human Vibration](https://github.com/jmrplens/phonometry/blob/main/docs/vibration/human/human-vibration.md) · [Mechanical Mobility](https://github.com/jmrplens/phonometry/blob/main/docs/vibration/structural/mechanical-mobility.md) · [Transfer Stiffness](https://github.com/jmrplens/phonometry/blob/main/docs/vibration/structural/transfer-stiffness.md) |
| Electroacoustics & broadcast | [Electroacoustics](https://github.com/jmrplens/phonometry/blob/main/docs/devices/electroacoustics/electroacoustics.md) · [Swept-Sine Distortion](https://github.com/jmrplens/phonometry/blob/main/docs/devices/electroacoustics/swept-sine-distortion.md) · [Programme Loudness](https://github.com/jmrplens/phonometry/blob/main/docs/devices/broadcast/program-loudness.md) · [Loudspeakers](https://github.com/jmrplens/phonometry/blob/main/docs/devices/electroacoustics/loudspeakers.md) · [Microphones](https://github.com/jmrplens/phonometry/blob/main/docs/devices/electroacoustics/microphones.md) |
| Emission | [Sound Power](https://github.com/jmrplens/phonometry/blob/main/docs/devices/emission/sound-power.md) · [Sound Intensity](https://github.com/jmrplens/phonometry/blob/main/docs/devices/emission/intensity.md) · [Sound Power by Pressure](https://github.com/jmrplens/phonometry/blob/main/docs/devices/emission/sound-power-pressure.md) · [Reverberation Room](https://github.com/jmrplens/phonometry/blob/main/docs/devices/emission/sound-power-reverberation.md) · [By Intensity](https://github.com/jmrplens/phonometry/blob/main/docs/devices/emission/sound-power-intensity.md) · [Silencers](https://github.com/jmrplens/phonometry/blob/main/docs/devices/noise-control/silencers.md) · [Filter Compliance](https://github.com/jmrplens/phonometry/blob/main/docs/signals/filters/filter-compliance.md) |
| Simulation | [2D FDTD Wave Simulation](https://github.com/jmrplens/phonometry/blob/main/docs/simulation/fdtd-simulation.md) · [Elastic Waves](https://github.com/jmrplens/phonometry/blob/main/docs/simulation/elastic-waves.md) |
| Reference | [API Reference](https://github.com/jmrplens/phonometry/blob/main/docs/reference/api/index.md) · [Theory](https://github.com/jmrplens/phonometry/blob/main/docs/reference/theory/index.md) · [Bibliography](https://github.com/jmrplens/phonometry/blob/main/docs/reference/bibliography.md) · [Why phonometry](https://github.com/jmrplens/phonometry/blob/main/docs/start/why-phonometry.md) · [Conformance report](https://github.com/jmrplens/phonometry/blob/main/docs/CONFORMANCE.md) · [Standards errata](https://github.com/jmrplens/phonometry/blob/main/docs/ERRATA.md) |

## 🧾 Citing

If phonometry is useful in your research, cite the archived release:

> Requena-Plens, J. M. *phonometry: acoustic measurement, analysis and
> prediction for Python.*
> Zenodo. https://doi.org/10.5281/zenodo.21215280

The repository ships a
[`CITATION.cff`](https://github.com/jmrplens/phonometry/blob/main/CITATION.cff),
so GitHub's *Cite this repository* button produces BibTeX and APA entries.

## 🧪 Development

```bash
make install   # dependencies + editable install
make check     # ruff + mypy + bandit + tests
make graphs    # regenerate documentation images
```

See the
[contributing guide](https://github.com/jmrplens/phonometry/blob/main/CONTRIBUTING.md)
and the
[changelog](https://github.com/jmrplens/phonometry/blob/main/CHANGELOG.md).
Suspected vulnerabilities go through
[GitHub's private reporting](https://github.com/jmrplens/phonometry/security/advisories/new),
as set out in the
[security policy](https://github.com/jmrplens/phonometry/blob/main/SECURITY.md).

## 📄 License

[MIT](https://github.com/jmrplens/phonometry/blob/main/LICENSE)
