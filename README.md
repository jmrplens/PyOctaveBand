<a href="https://jmrplens.github.io/phonometry/"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/brand/banner.webp" alt="phonometry, acoustic measurement toolkit for Python" width="100%"></a>

<!-- Package -->
[![PyPI version](https://img.shields.io/pypi/v/phonometry?logo=pypi&logoColor=white)](https://pypi.org/project/phonometry/)
[![Python versions](https://img.shields.io/pypi/pyversions/phonometry?logo=python&logoColor=white)](https://pypi.org/project/phonometry/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/jmrplens/phonometry/blob/main/LICENSE)

<!-- Quality -->
[![CI](https://github.com/jmrplens/phonometry/actions/workflows/python-app.yml/badge.svg)](https://github.com/jmrplens/phonometry/actions/workflows/python-app.yml)
[![Quality Gate](https://sonarcloud.io/api/project_badges/measure?project=jmrplens_phonometry&metric=alert_status)](https://sonarcloud.io/summary/overall?id=jmrplens_phonometry)
[![codecov](https://codecov.io/gh/jmrplens/phonometry/branch/main/graph/badge.svg)](https://codecov.io/gh/jmrplens/phonometry)

<!-- Citation & support -->
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21215280-blue?logo=doi&logoColor=white)](https://doi.org/10.5281/zenodo.21215280)

# phonometry

> *phonometry* — the measurement of sound. Formerly published as **PyOctaveBand**.

Acoustic measurement toolkit for Python, from fractional octave-band filters, weighting and sound level metrology to psychoacoustics, rooms and buildings, vibration, environmental and aircraft noise, underwater acoustics, electroacoustics and wave simulation. Every metric is conformance-tested against its governing standard (371 checks across 46 domains and 235 standards), with class 1 filters per **IEC 61260-1:2014 / ANSI S1.11-2004** and class 1 weightings and levels per **IEC 61672-1:2013**.

<img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/filter_type_comparison.svg" alt="Magnitude response comparison of the five filter architectures for the 1 kHz octave band, with a zoom at the -3 dB crossover" width="80%">

## ✨ Highlights

- 🎛️ 1/1, 1/3 and arbitrary fractional octave filter banks (stable SOS + multirate decimation)
- 🏗️ Five architectures: Butterworth, Chebyshev I/II, Elliptic, Bessel — all with −3 dB points on the ANSI band edges
- 🔊 A/C/Z frequency weighting within IEC 61672-1 class 1 tolerances, plus G weighting for infrasound (ISO 7196)
- ⏱️ Fast/Slow/Impulse time ballistics, `Leq`, `LAeq` and `L10/L50/L90` statistical levels
- 🗺️ Octave spectrogram (band levels over time) and zero-phase offline filtering
- 🧠 Loudness in sones three ways: Zwicker (ISO 532-1 Annex B validated), Moore-Glasberg stationary & time-varying (ISO 532-2/3) and Sottek Hearing Model (ECMA-418-2); DIN 45692 sharpness, ISO 226:2023 contours
- 🎻 Sound-quality metrics: ECMA-418-2 tonality (tu_HMS), roughness (asper) and fluctuation strength (vacil_HMS)
- 🗣️ Speech Transmission Index: STI and STIPA per IEC 60268-16 Ed. 5, with signal generator
- 🎯 Tone prominence (TNR/PR, ECMA-418-1), environmental Lden/Ldn (ISO 1996-1), IEC 61252 noise dose
- ↗️ Two-microphone sound intensity (IEC 61043) with ISO 9614-1 field indicators
- 🏛️ Room & building acoustics: swept-sine/MLS impulse responses (ISO 18233), EDT/T20/T30/C50/C80/Ts (ISO 3382-1/2), open-plan speech metrics (ISO 3382-3), field airborne + impact + façade insulation with R′w/DnT,w/L′nT,w/D2m,nT,w and C/Ctr/CI (ISO 16283-1/2/3, ISO 717-1/2), laboratory R/Ln (ISO 10140), flanking-transmission prediction of R′w/L′n,w (EN 12354-1/2), measurement uncertainty (ISO 12999-1), sound absorption (ISO 354)
- 🌬️ Outdoor propagation & occupational exposure: atmospheric absorption α(f) (ISO 9613-1), the ISO 9613-2 general method (divergence + atmospheric + ground + barrier terms) with a per-term octave-band breakdown, and daily noise exposure LEX,8h with task/job/full-day strategies and Annex C uncertainty (ISO 9612)
- 🔊 Sound power LW five ways: enveloping-surface pressure (ISO 3744/3746), reverberation-room precision with Waterhouse/C1/C2 (ISO 3741), intensity scanning with field indicators and grade (ISO 9614-2), precision anechoic rooms (ISO 3745) and precision intensity scanning (ISO 9614-3), plus ISO 4871 noise-emission declarations
- 🦻 Hearing: age-related threshold distributions (ISO 7029) and noise-induced permanent threshold shift with HTLAN (ISO 1999)
- 🧱 Materials: absorption ratings αw (ISO 11654), impedance-tube absorption (ISO 10534-2), porous-absorber models, airflow resistance, scattering and diffusion (ISO 17497), dynamic stiffness
- 🔩 Vibration & structure-borne sound: mobility and FRFs (ISO 7626), isolator transfer stiffness (ISO 10846), sound power from surface vibration (ISO/TS 7849), reception-plate power (EN 15657), installed-source prediction (EN 12354-5) and human vibration (ISO 2631-1/-5, ISO 5349)
- ✈️ Aircraft noise: EPNL certification chain (ICAO Annex 16) with IEC 61265 verification and SAE ARP 5534 absorption, ECAC Doc 29 airport contours with the EASA ANP fleet database, and the ECAC Doc 32 rotorcraft hemisphere method
- 🌊 Underwater acoustics: ISO 18405 levels re 1 µPa, ship radiated noise (ISO 17208-1/2), pile driving (ISO 18406), sonar equation, Wenz/JOMOPANS-ECHO ambient noise, and transmission loss from spreading laws to normal-mode, ray and parabolic-equation solvers
- 🔈 Electroacoustics: distortion per IEC 60268-3 (THD, THD+N, SMPTE/CCIF intermodulation, DIM), frequency response and coherence, rigid-piston radiation, and loudspeaker/microphone rated characteristics (IEC 60268-5/-4)
- 🔇 Noise control: silencer insertion loss, enclosures and HVAC spectra
- 🌐 Deterministic 2D FDTD wave simulation with sources, probes, rasterised obstacles and rigid/impedance/absorbing boundaries
- 📄 Typed, frozen result dataclasses with `.plot(language="en"|"es")` figures and normative `.report()` PDF fiches (ISO 717, ISO 11654, ISO 532-1, EBU R 128, ICAO EPNL, IEC 61260-1, ISO 4871, IEC 60268-5/-4); documentation fully in English and Spanish
- 📏 Physical SPL calibration with IEC 60942:2017 stability validation, and dBFS modes
- 📉 Calibrated spectral analysis (Bendat & Piersol): PSD/CSD with chi-square confidence intervals and random errors, coherent output spectrum & spectral SNR, 1/n-octave smoothing, exact-slope colored-noise generators
- ⏱️ Correlation & time-delay estimation (Bendat & Piersol, Knapp & Carter): biased/unbiased/coefficient correlation with random errors, GCC with Roth/SCOT/PHAT/ML weightings, sub-sample IR delay & alignment, Hilbert envelope with instantaneous frequency
- 🌀 Swept-sine distortion & phase utilities: harmonic separation and THD(f) from one exponential sweep (Farina 2000, Novak et al. 2015 synchronized sweep), minimum phase from |H|, group delay & excess phase
- ⚡ Vectorized multichannel processing and stateful block (real-time) workflows

## 🚀 Installation

```bash
pip install phonometry
```

Optional extras: `phonometry[plot]` (matplotlib for response plots and result `.plot()` methods), `phonometry[perf]` (numba for faster impulse ballistics), `phonometry[report]` (reportlab and svglib, so result `.report()` methods can render normative PDF fiches), `phonometry[full]` (all of the above).

## 📚 Documentation

**Full documentation website: https://jmrplens.github.io/phonometry/** (English / Español)

Or browse the Markdown docs on GitHub:

| Page | Contents |
| :--- | :--- |
| [Getting Started](https://github.com/jmrplens/phonometry/blob/main/docs/getting-started.md) | Installation, first analysis, WAV files |
| [Filter Banks](https://github.com/jmrplens/phonometry/blob/main/docs/filter-banks.md) | Architectures, response gallery, band decomposition, zero-phase |
| [Frequency Weighting](https://github.com/jmrplens/phonometry/blob/main/docs/weighting.md) | A/C/Z curves, class 1 high-accuracy mode |
| [Time Weighting](https://github.com/jmrplens/phonometry/blob/main/docs/time-weighting.md) | Fast/Slow/Impulse ballistics, initial state |
| [Levels](https://github.com/jmrplens/phonometry/blob/main/docs/levels.md) | Leq, LAeq, percentiles, LCpeak, SEL, noise dose (IEC 61252), Lden and rating levels (ISO 1996-1), octave spectrogram |
| [Occupational Exposure](https://github.com/jmrplens/phonometry/blob/main/docs/occupational-exposure.md) | ISO 9612 task-based, job-based and full-day strategies with the Annex C uncertainty budget (LEX,8h + U) |
| [Tone Prominence](https://github.com/jmrplens/phonometry/blob/main/docs/tone-prominence.md) | ECMA-418-1 tone-to-noise ratio and prominence ratio with frequency-dependent prominence criteria |
| [Loudness](https://github.com/jmrplens/phonometry/blob/main/docs/loudness.md) | Zwicker (ISO 532-1), Moore-Glasberg (ISO 532-2/3) and Sottek (ECMA-418-2) loudness in sones, plus the equal-loudness contours (ISO 226) |
| [Sound Quality Metrics](https://github.com/jmrplens/phonometry/blob/main/docs/sound-quality.md) | Sharpness (DIN 45692) and the ECMA-418-2 Sottek Hearing Model tonality, roughness & fluctuation strength |
| [Speech Transmission](https://github.com/jmrplens/phonometry/blob/main/docs/speech-transmission.md) | STI/STIPA (IEC 60268-16): modulation transfer function, indirect method from impulse responses and direct STIPA measurement |
| [Speech Intelligibility Index](https://github.com/jmrplens/phonometry/blob/main/docs/speech-intelligibility.md) | SII (ANSI S3.5-1997): band importance, masking and audibility, the index in noise and hearing loss, standard vocal-effort spectra |
| [Electroacoustics](https://github.com/jmrplens/phonometry/blob/main/docs/electroacoustics.md) | Distortion (IEC 60268-3): THD, nth-order harmonic, THD+N & SINAD (AES17), SMPTE & CCIF intermodulation, DIM and weighted THD; frequency response & coherence (Bendat & Piersol H1/H2) |
| [Swept-Sine Distortion & Phase](https://github.com/jmrplens/phonometry/blob/main/docs/swept-sine-distortion.md) | Harmonic separation & THD(f) from one exponential sweep (Farina 2000; Novak et al. 2015 synchronized sweep with coherent harmonic phases); minimum phase from \|H\| (real cepstrum), group delay & excess phase |
| [Calibrated Spectral Analysis](https://github.com/jmrplens/phonometry/blob/main/docs/spectral-analysis.md) | Welch PSD/CSD with effective averages, random errors & chi-square confidence intervals; coherent output spectrum & spectral SNR; 1/n-octave smoothing; colored-noise generators (Bendat & Piersol) |
| [Programme Loudness](https://github.com/jmrplens/phonometry/blob/main/docs/program-loudness.md) | ITU-R BS.1770-5 programme loudness (K-weighting, gating, multichannel weights incl. Annex 3) and true peak in dBTP; EBU R 128 with the Tech 3341 EBU Mode M/S/I meters and the Tech 3342 loudness range |
| [Correlation, Time Delay & Envelope](https://github.com/jmrplens/phonometry/blob/main/docs/correlation-delay.md) | Auto/cross-correlation with B&P normalizations & random errors; TDE by direct correlation, cross-spectrum phase slope & Knapp-Carter GCC (Roth/SCOT/PHAT/ML); sub-sample IR delay & alignment; Hilbert envelope & instantaneous frequency |
| [Underwater Acoustics](https://github.com/jmrplens/phonometry/blob/main/docs/underwater-acoustics.md) | Reference levels re 1 µPa (SPL, SEL, peak; ISO 18405); ship radiated noise & equivalent monopole source level (ISO 17208); pile-driving single-strike, peak & cumulative SEL (ISO 18406) |
| [Underwater Sound Propagation](https://github.com/jmrplens/phonometry/blob/main/docs/underwater-propagation.md) | Transmission loss (geometrical spreading + volume absorption: Francois-Garrison, Ainslie-McColm, Thorp); speed of sound in sea water (UNESCO/Chen-Millero, Del Grosso, Mackenzie); passive & active sonar equation; seabed reflection loss (Rayleigh); ocean ambient noise (Wenz wind/thermal + JOMOPANS-ECHO ship traffic) |
| [Underwater Propagation Solvers](https://github.com/jmrplens/phonometry/blob/main/docs/underwater-solvers.md) | Numerical solvers of the stratified ocean (Jensen et al.): normal modes, ray tracing and the split-step Fourier parabolic equation, each validated against an exact closed form, with model-selection guidance |
| [Aircraft Noise](https://github.com/jmrplens/phonometry/blob/main/docs/aircraft-noise.md) | Effective Perceived Noise Level (ICAO Annex 16): perceived noisiness & PNL, tone correction, 10 dB-down duration correction (EPNL); IEC 61265 measurement-system verification; SAE ARP 5534 one-third-octave-band atmospheric absorption; ECAC Doc 29 noise-power-distance (NPD) event-level interpolation |
| [Wind-Turbine Noise](https://github.com/jmrplens/phonometry/blob/main/docs/wind-turbine-noise.md) | Apparent sound power level referred to the rotor centre and tonal audibility (Zwicker critical band, masking-noise level, audibility criterion) — IEC 61400-11 |
| [Sound Intensity](https://github.com/jmrplens/phonometry/blob/main/docs/intensity.md) | Two-microphone p-p intensity (IEC 61043), ISO 9614-1 field indicators |
| [Measuring the Room Impulse Response](https://github.com/jmrplens/phonometry/blob/main/docs/room-impulse-response.md) | Deterministic impulse-response acquisition (ISO 18233): exponential sweeps and their deconvolution, and MLS |
| [Room Acoustics](https://github.com/jmrplens/phonometry/blob/main/docs/room-acoustics.md) | Room parameters from the impulse response (ISO 3382-1/2): T20, T30, EDT, C50, C80, D50 and Ts |
| [Open-Plan Office Acoustics](https://github.com/jmrplens/phonometry/blob/main/docs/open-plan-acoustics.md) | Speech privacy on an open-plan floor (ISO 3382-3): spatial decay rate of speech, distraction, privacy and comfort distances |
| [Image Sources & Steady-State Field](https://github.com/jmrplens/phonometry/blob/main/docs/room-image-sources.md) | Image-source room impulse response of a shoebox (Kuttruff/Vorländer), and the steady-state level with the room constant, critical distance and Schroeder frequency (Bies) |
| [Field Insulation Measurement (ISO 16283)](https://github.com/jmrplens/phonometry/blob/main/docs/insulation-field.md) | Field airborne + impact insulation (ISO 16283-1/2), the field test report and the measurement uncertainty that qualifies it (ISO 12999-1) |
| [Laboratory Insulation Measurement](https://github.com/jmrplens/phonometry/blob/main/docs/insulation-lab.md) | Laboratory sound reduction index and normalized impact level with flanking suppressed (ISO 10140), with the background-noise correction and the accredited fiches |
| [Sound Insulation by Intensity (ISO 15186)](https://github.com/jmrplens/phonometry/blob/main/docs/insulation-intensity.md) | Sound reduction index from the intensity scanned over the radiating face, whole element or element by element (ISO 15186-1/-2) |
| [Sound Insulation Survey Method (ISO 10052)](https://github.com/jmrplens/phonometry/blob/main/docs/insulation-survey.md) | Octave-band control method (ISO 10052): reverberation index, airborne, impact, façade and service-equipment quantities with their survey reports |
| [Laboratory Flanking Transmission (ISO 10848)](https://github.com/jmrplens/phonometry/blob/main/docs/flanking-lab.md) | Junction vibration reduction index Kij and the flanking descriptors Dn,f and Ln,f measured on a test facility (ISO 10848) |
| [Insulation Ratings (ISO 717)](https://github.com/jmrplens/phonometry/blob/main/docs/insulation-ratings.md) | The ISO 717-1 airborne and ISO 717-2 impact reference-curve ratings with C, Ctr and CI, the enlarged-range and one-decimal variants, and the ISO 717 fiche |
| [Predicting Sound Insulation (EN 12354)](https://github.com/jmrplens/phonometry/blob/main/docs/insulation-prediction.md) | Airborne and impact flanking-transmission prediction between rooms (EN 12354-1/2) |
| [Façade Sound Insulation](https://github.com/jmrplens/phonometry/blob/main/docs/facade-insulation.md) | The building envelope measured (ISO 16283-3), predicted from element indices (EN 12354-3) and radiating an indoor source outwards (EN 12354-4) |
| [Floor-Covering Impact Improvement (ISO 16251-1)](https://github.com/jmrplens/phonometry/blob/main/docs/impact-improvement.md) | Weighted improvement of impact sound insulation given by a soft floor covering, measured on a small heavyweight mock-up (ISO 16251-1) |
| [Outdoor Sound Propagation](https://github.com/jmrplens/phonometry/blob/main/docs/outdoor-propagation.md) | Atmospheric absorption α(f) (ISO 9613-1) and the ISO 9613-2 general method: geometrical divergence, atmospheric absorption, ground effect, barrier screening and meteorological correction |
| [Sound Power](https://github.com/jmrplens/phonometry/blob/main/docs/sound-power.md) | Choosing the determination method and declaring the noise emission (ISO 4871), with a guide per route: enveloping surface (ISO 3744/3746), anechoic room (ISO 3745), reverberation room (ISO 3741) and intensity scanning (ISO 9614-2/-3) |
| [Calibration and dBFS](https://github.com/jmrplens/phonometry/blob/main/docs/calibration.md) | Physical SPL, digital full-scale, RMS vs peak |
| [Block Processing](https://github.com/jmrplens/phonometry/blob/main/docs/block-processing.md) | Stateful streaming workflows |
| [Multichannel](https://github.com/jmrplens/phonometry/blob/main/docs/multichannel.md) | Vectorized multichannel analysis, performance |
| [API Reference](https://github.com/jmrplens/phonometry/blob/main/docs/api-reference.md) | Every public function and class |
| [Theory](https://github.com/jmrplens/phonometry/blob/main/docs/theory.md) | Standards, math, design decisions |
| [Why phonometry](https://github.com/jmrplens/phonometry/blob/main/docs/why-phonometry.md) | IEC compliance verification vs other libraries |
| [Conformance report](https://github.com/jmrplens/phonometry/blob/main/docs/CONFORMANCE.md) | Live per-standard numerical validation (expected vs computed) regenerated by `make conformance` |

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

The library is organized into fifteen domain namespaces; every public name is
also re-exported at the top level, so `from phonometry import octave_filter`
keeps working:

```python
from phonometry import building, underwater

r = building.airborne_insulation(...)
tl = underwater.transmission_loss(...)
```

<img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/signal_response_fraction_3.svg" alt="One-third-octave spectrum analysis of a multi-tone signal with the raw PSD in the background" width="80%">

*1/3 Octave Band spectrum analysis of a complex signal. More examples in the
[documentation](https://jmrplens.github.io/phonometry/).*

## 🧪 Development

```bash
make install   # dependencies + editable install
make check     # ruff + mypy + bandit + tests
make graphs    # regenerate documentation images
```

See https://github.com/jmrplens/phonometry/blob/main/CONTRIBUTING.md and the
https://github.com/jmrplens/phonometry/blob/main/CHANGELOG.md

## 📄 License

[MIT](https://github.com/jmrplens/phonometry/blob/main/LICENSE)
