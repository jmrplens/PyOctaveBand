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
pins hundreds of expected normative values, spanning more than 250 standards,
to the values the library computes, and CI regenerates it on every pull
request. Filters are class 1 per **IEC 61260-1:2014 / ANSI S1.11-2004** and
weightings and levels class 1 per **IEC 61672-1:2013**.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_fdtd_pillar_hall_dark.gif"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_fdtd_pillar_hall.gif" alt="Animation: an 800 Hz plane wavefront sweeps through a hall of rigid columns in a 2D FDTD simulation; every column diffracts the front and the scattered wavelets interfere until the whole hall is filled" width="100%"></picture>

*An 800 Hz wavefront threading a hall of columns, computed with the library's
own [2D FDTD engine](https://github.com/jmrplens/phonometry/blob/main/docs/fdtd-simulation.md)
(`phonometry.simulation`) and rendered by the same script that generates every
figure in the documentation.*

## 🚀 Installation

```bash
pip install phonometry
```

Optional extras: `phonometry[plot]` (matplotlib for response plots and result
`.plot()` methods), `phonometry[perf]` (numba for faster impulse ballistics),
`phonometry[report]` (reportlab and svglib, so result `.report()` methods can
render normative PDF fiches), `phonometry[full]` (all of the above).

## ⚡ Quick start

```python
import numpy as np
from phonometry import metrology

fs = 48000
t = np.linspace(0, 1, fs, endpoint=False)
# Composite signal: 100Hz + 1000Hz
signal = np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 1000 * t)

# Apply 1/3 octave filter bank
spl, freq = metrology.octave_filter(signal, fs=fs, fraction=3)

print(f"Bands: {freq}")
print(f"SPL [dB]: {spl}")
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/signal_response_fraction_3_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/signal_response_fraction_3.svg" alt="One-third-octave spectrum analysis of a multi-tone signal with the raw PSD in the background" width="80%"></picture>

*1/3 octave band spectrum analysis of a complex signal. More examples in the
[documentation](https://jmrplens.github.io/phonometry/).*

## ✨ What's inside

The library is organized into domain namespaces; nearly every public name is
also re-exported at the top level, so `from phonometry import octave_filter`
keeps working:

```python
from phonometry import building, underwater

r = building.airborne_insulation(...)
tl = underwater.transmission_loss(...)
```

| Namespace | Coverage |
| :--- | :--- |
| `metrology` | 1/1, 1/3 and arbitrary fractional octave filter banks (stable SOS + multirate decimation) in five architectures with per-band class verdicts (IEC 61260-1 / ANSI S1.11); A/C/Z weighting within IEC 61672-1 class 1 tolerances plus G weighting (ISO 7196); Fast/Slow/Impulse ballistics, Leq, SEL, L10/L50/L90, noise dose (IEC 61252); octave spectrogram and zero-phase filtering; physical SPL calibration with IEC 60942 stability validation and dBFS modes; calibrated Welch PSD/CSD with chi-square confidence intervals, coherent output spectrum, 1/n-octave smoothing and colored-noise generators (Bendat & Piersol); MISO multiple/partial coherence; correlation and GCC time-delay estimation (Knapp & Carter); Hilbert envelope, cepstrum and echoes, time synchronous averaging, calibrated STFT and zoom FFT; Golay/shaped-sweep system measurement with regularized inversion; IEC 60268-1 tone bursts and resampling; GUM uncertainty (ISO/IEC Guide 98-3) and Bendat & Piersol data qualification |
| `psychoacoustics` | Loudness in sones three ways: Zwicker (ISO 532-1 Annex B validated), Moore-Glasberg stationary and time-varying (ISO 532-2/3) and Sottek Hearing Model (ECMA-418-2); DIN 45692 sharpness; ECMA-418-2 tonality, roughness (asper) and fluctuation strength (vacil_HMS); tone prominence TNR/PR (ECMA-418-1); tonal audibility (ISO/PAS 20065); Fastl & Zwicker psychoacoustic annoyance; ISO 226:2023 contours |
| `hearing` | Speech Transmission Index STI/STIPA with signal generator (IEC 60268-16 Ed. 5); Speech Intelligibility Index (ANSI S3.5-1997); STOI and ESTOI; age-related thresholds (ISO 7029) and reference thresholds (ISO 389-7); noise-induced hearing loss with HTLAN (ISO 1999); daily noise exposure LEX,8h with Annex C uncertainty (ISO 9612) |
| `room` | Swept-sine/MLS impulse responses (ISO 18233); EDT/T20/T30/C50/C80/Ts (ISO 3382-1/2); open-plan speech metrics (ISO 3382-3); reverberation-room absorption (ISO 354); reverberation-time prediction (Sabine to Arau-Puchades); total absorption of furnished rooms (EN 12354-6); image-source impulse responses and the steady-state field; room-noise criteria NC and RC Mark II (ANSI/ASA S12.2) |
| `building` | Field airborne, impact and façade insulation with R′w/DnT,w/L′nT,w/D2m,nT,w and C/Ctr/CI (ISO 16283-1/2/3, ISO 717-1/2); laboratory R/Ln (ISO 10140) and survey method (ISO 10052); insulation by intensity (ISO 15186); flanking transmission measurement and prediction (ISO 10848, EN 12354-1/2) and façade/outdoor radiation (EN 12354-3/4); measurement uncertainty (ISO 12999-1); panel transmission theory (mass law, coincidence, double walls, slits and apertures); floor-covering improvement (ISO 16251-1); reception-plate power (EN 15657) and installed structure-borne prediction (EN 12354-5); dynamic stiffness (EN 29052-1) |
| `materials` | Absorption ratings αw with classes (ISO 11654) and uncertainty (ISO 12999-2); impedance-tube absorption, impedance and transmission loss (ISO 10534-1/2, ASTM E2611) plus a virtual FDTD tube; porous and multilayer absorber models (Delany-Bazley, Miki, JCA, TMM with MPP and membranes); slow-sound metamaterial absorbers at critical coupling; scattering and diffusion coefficients (ISO 17497-1/2); Schroeder diffuser design and far-field prediction; deep-subwavelength metadiffusers; in-situ road-surface absorption (ISO 13472-1/2); airflow resistance (ISO 9053-1/2) |
| `emission` | Sound power by enveloping surface (ISO 3744/3746), reverberation room (ISO 3741), precision anechoic rooms (ISO 3745) and intensity scanning with field indicators and grades (ISO 9614-2/3); two-microphone p-p intensity (IEC 61043, ISO 9614-1); sound power from surface vibration (ISO/TS 7849); noise-emission declarations (ISO 4871) |
| `environmental` | Rating levels, Lden/Ldn and adjustments (ISO 1996-1/2); impulsive-sound prominence (NT ACOU 112); atmospheric absorption (ISO 9613-1) and the ISO 9613-2 general method with per-term octave breakdown; spherical ground effect and wave-theoretic barriers; refraction ray tracing and the GFPE; wind-turbine apparent sound power and tonal audibility (IEC 61400-11) |
| `aircraft` | EPNL certification chain (ICAO Annex 16) with IEC 61265 verification and SAE ARP 5534 absorption; airport noise contours (ECAC Doc 29) with the EASA ANP fleet database; rotorcraft hemisphere method (ECAC Doc 32) |
| `underwater` | Levels re 1 µPa (ISO 18405); ship radiated noise (ISO 17208-1/2); pile driving (ISO 18406); ship-traffic source levels (JOMOPANS-ECHO) and Wenz ambient noise; sonar equation; sound speed and seabed reflection; transmission loss from spreading laws to normal-mode, ray and parabolic-equation solvers |
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
| Getting started | [Getting Started](https://github.com/jmrplens/phonometry/blob/main/docs/getting-started.md) · [Build a sound level meter](https://github.com/jmrplens/phonometry/blob/main/docs/sound-level-meter.md) · [Calibration and dBFS](https://github.com/jmrplens/phonometry/blob/main/docs/calibration.md) |
| Filters, levels & weighting | [Filter Banks](https://github.com/jmrplens/phonometry/blob/main/docs/filter-banks.md) · [Filter Gallery](https://github.com/jmrplens/phonometry/blob/main/docs/filter-gallery.md) · [Levels](https://github.com/jmrplens/phonometry/blob/main/docs/levels.md) · [Environmental Levels](https://github.com/jmrplens/phonometry/blob/main/docs/environmental-levels.md) · [Spanish Noise Regulation](https://github.com/jmrplens/phonometry/blob/main/docs/spanish-noise-regulation.md) · [Frequency Weighting](https://github.com/jmrplens/phonometry/blob/main/docs/weighting.md) · [Special Weightings](https://github.com/jmrplens/phonometry/blob/main/docs/special-weightings.md) · [Time Weighting](https://github.com/jmrplens/phonometry/blob/main/docs/time-weighting.md) · [Block Processing](https://github.com/jmrplens/phonometry/blob/main/docs/block-processing.md) · [Multichannel](https://github.com/jmrplens/phonometry/blob/main/docs/multichannel.md) |
| Signal analysis | [Calibrated spectral analysis](https://github.com/jmrplens/phonometry/blob/main/docs/spectral-analysis.md) · [Correlation, time delay & envelope](https://github.com/jmrplens/phonometry/blob/main/docs/correlation-delay.md) · [Time-frequency analysis](https://github.com/jmrplens/phonometry/blob/main/docs/time-frequency.md) · [System measurement](https://github.com/jmrplens/phonometry/blob/main/docs/system-measurement.md) · [Measurement uncertainty](https://github.com/jmrplens/phonometry/blob/main/docs/gum-uncertainty.md) |
| Psychoacoustics & sound quality | [Loudness](https://github.com/jmrplens/phonometry/blob/main/docs/loudness.md) · [Advanced Loudness](https://github.com/jmrplens/phonometry/blob/main/docs/advanced-loudness.md) · [Sound Quality Metrics](https://github.com/jmrplens/phonometry/blob/main/docs/sound-quality.md) · [Tone Prominence](https://github.com/jmrplens/phonometry/blob/main/docs/tone-prominence.md) · [Psychoacoustic annoyance](https://github.com/jmrplens/phonometry/blob/main/docs/psychoacoustic-annoyance.md) |
| Speech & hearing | [Speech Transmission Index](https://github.com/jmrplens/phonometry/blob/main/docs/speech-transmission.md) · [Speech Intelligibility Index](https://github.com/jmrplens/phonometry/blob/main/docs/speech-intelligibility.md) · [Objective intelligibility](https://github.com/jmrplens/phonometry/blob/main/docs/objective-intelligibility.md) · [Noise-induced hearing loss](https://github.com/jmrplens/phonometry/blob/main/docs/noise-induced-hearing-loss.md) · [Occupational exposure](https://github.com/jmrplens/phonometry/blob/main/docs/occupational-exposure.md) |
| Rooms & buildings | [Room Acoustics](https://github.com/jmrplens/phonometry/blob/main/docs/room-acoustics.md) · [Field Insulation](https://github.com/jmrplens/phonometry/blob/main/docs/insulation-field.md) · [Laboratory Insulation](https://github.com/jmrplens/phonometry/blob/main/docs/insulation-lab.md) · [Predicting Insulation (EN 12354)](https://github.com/jmrplens/phonometry/blob/main/docs/insulation-prediction.md) · [Image Sources](https://github.com/jmrplens/phonometry/blob/main/docs/room-image-sources.md) · [Impulse Response (ISO 18233)](https://github.com/jmrplens/phonometry/blob/main/docs/room-impulse-response.md) · [Open-Plan Offices](https://github.com/jmrplens/phonometry/blob/main/docs/open-plan-acoustics.md) · [Insulation Ratings (ISO 717)](https://github.com/jmrplens/phonometry/blob/main/docs/insulation-ratings.md) · [Façade Insulation](https://github.com/jmrplens/phonometry/blob/main/docs/facade-insulation.md) · [Spanish Building Code (CTE DB-HR)](https://github.com/jmrplens/phonometry/blob/main/docs/spanish-building-code.md) · [Survey Method (ISO 10052)](https://github.com/jmrplens/phonometry/blob/main/docs/insulation-survey.md) · [Insulation by Intensity](https://github.com/jmrplens/phonometry/blob/main/docs/insulation-intensity.md) · [Impact Improvement](https://github.com/jmrplens/phonometry/blob/main/docs/impact-improvement.md) · [Flanking (ISO 10848)](https://github.com/jmrplens/phonometry/blob/main/docs/flanking-lab.md) |
| Materials & surfaces | [Porous Absorbers](https://github.com/jmrplens/phonometry/blob/main/docs/porous-absorbers.md) · [Metamaterial Absorbers](https://github.com/jmrplens/phonometry/blob/main/docs/metamaterial-absorbers.md) · [Diffusers](https://github.com/jmrplens/phonometry/blob/main/docs/diffusers.md) · [Metadiffusers](https://github.com/jmrplens/phonometry/blob/main/docs/metadiffusers.md) · [Impedance Tube](https://github.com/jmrplens/phonometry/blob/main/docs/impedance-tube.md) · [Absorption Measurement](https://github.com/jmrplens/phonometry/blob/main/docs/absorption-measurement.md) · [Airflow Resistance](https://github.com/jmrplens/phonometry/blob/main/docs/airflow-resistance.md) · [Road Absorption](https://github.com/jmrplens/phonometry/blob/main/docs/road-absorption.md) |
| Environment & outdoors | [Outdoor Propagation](https://github.com/jmrplens/phonometry/blob/main/docs/outdoor-propagation.md) · [Ground & Barriers](https://github.com/jmrplens/phonometry/blob/main/docs/ground-barriers.md) · [Atmospheric Refraction](https://github.com/jmrplens/phonometry/blob/main/docs/atmospheric-refraction.md) |
| Transport | [Aircraft Noise](https://github.com/jmrplens/phonometry/blob/main/docs/aircraft-noise.md) · [Rotorcraft Noise](https://github.com/jmrplens/phonometry/blob/main/docs/rotorcraft-noise.md) · [Wind-Turbine Noise](https://github.com/jmrplens/phonometry/blob/main/docs/wind-turbine-noise.md) · [Airport Noise](https://github.com/jmrplens/phonometry/blob/main/docs/airport-noise.md) |
| Underwater | [Underwater Acoustics](https://github.com/jmrplens/phonometry/blob/main/docs/underwater-acoustics.md) · [Underwater Propagation](https://github.com/jmrplens/phonometry/blob/main/docs/underwater-propagation.md) · [Underwater Solvers](https://github.com/jmrplens/phonometry/blob/main/docs/underwater-solvers.md) |
| Vibration | [Human Vibration](https://github.com/jmrplens/phonometry/blob/main/docs/human-vibration.md) · [Mechanical Mobility](https://github.com/jmrplens/phonometry/blob/main/docs/mechanical-mobility.md) · [Transfer Stiffness](https://github.com/jmrplens/phonometry/blob/main/docs/transfer-stiffness.md) |
| Electroacoustics & broadcast | [Electroacoustics](https://github.com/jmrplens/phonometry/blob/main/docs/electroacoustics.md) · [Swept-Sine Distortion](https://github.com/jmrplens/phonometry/blob/main/docs/swept-sine-distortion.md) · [Programme Loudness](https://github.com/jmrplens/phonometry/blob/main/docs/program-loudness.md) · [Loudspeakers](https://github.com/jmrplens/phonometry/blob/main/docs/loudspeakers.md) · [Microphones](https://github.com/jmrplens/phonometry/blob/main/docs/microphones.md) |
| Emission | [Sound Power](https://github.com/jmrplens/phonometry/blob/main/docs/sound-power.md) · [Sound Intensity](https://github.com/jmrplens/phonometry/blob/main/docs/intensity.md) · [Sound Power by Pressure](https://github.com/jmrplens/phonometry/blob/main/docs/sound-power-pressure.md) · [Reverberation Room](https://github.com/jmrplens/phonometry/blob/main/docs/sound-power-reverberation.md) · [By Intensity](https://github.com/jmrplens/phonometry/blob/main/docs/sound-power-intensity.md) · [Silencers](https://github.com/jmrplens/phonometry/blob/main/docs/silencers.md) · [Filter Compliance](https://github.com/jmrplens/phonometry/blob/main/docs/filter-compliance.md) |
| Simulation | [2D FDTD Wave Simulation](https://github.com/jmrplens/phonometry/blob/main/docs/fdtd-simulation.md) · [Elastic Waves](https://github.com/jmrplens/phonometry/blob/main/docs/elastic-waves.md) |
| Reference | [API Reference](https://github.com/jmrplens/phonometry/blob/main/docs/api-reference.md) · [Theory](https://github.com/jmrplens/phonometry/blob/main/docs/theory.md) · [Bibliography](https://github.com/jmrplens/phonometry/blob/main/docs/references.md) · [Why phonometry](https://github.com/jmrplens/phonometry/blob/main/docs/why-phonometry.md) · [Conformance report](https://github.com/jmrplens/phonometry/blob/main/docs/CONFORMANCE.md) · [Standards errata](https://github.com/jmrplens/phonometry/blob/main/docs/ERRATA.md) |

## 🧾 Citing

If phonometry is useful in your research, cite the archived release:

> Requena-Plens, J. M. *phonometry: acoustic measurement toolkit for Python.*
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
