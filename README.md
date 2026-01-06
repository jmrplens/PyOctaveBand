[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.com/donate?hosted_button_id=BLP3R6VGYJB4Q)
[![Donate](https://img.shields.io/badge/Donate-Ko--fi-brightgreen?color=ff5f5f)](https://ko-fi.com/jmrplens) 
[![Python application](https://github.com/jmrplens/PyOctaveBand/actions/workflows/python-app.yml/badge.svg)](https://github.com/jmrplens/PyOctaveBand/actions/workflows/python-app.yml)

# PyOctaveBand
Advanced Octave-Band and Fractional Octave-Band filter bank for signals in the time domain. Fully compliant with **ANSI s1.11-2004** and **IEC 61260-1-2014**.

This library provides professional-grade tools for acoustic analysis, including frequency weighting (A, C, Z), time ballistics (Fast, Slow, Impulse), and multiple filter architectures (Butterworth, Chebyshev, Elliptic, Bessel).

---

## 🚀 Getting Started

### Installation

```bash
pip install .
```

### Basic Usage: 1/3 Octave Analysis
Analyze a signal and get the Sound Pressure Level (SPL) per frequency band.

```python
import numpy as np
from pyoctaveband import octavefilter

fs = 48000
t = np.linspace(0, 1, fs)
# Composite signal: 100Hz + 1000Hz
signal = np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 1000 * t)

# Apply 1/3 octave filter bank
spl, freq = octavefilter(signal, fs=fs, fraction=3)

print(f"Bands: {freq}")
print(f"SPL [dB]: {spl}")
```

---

## 🛠️ Advanced Filter Architecture

PyOctaveBand allows choosing between different filter types to balance between roll-off steepness and transient response (ringing).

### Filter Type Comparison
Different architectures offer different trade-offs in the stopband attenuation.

<img src=".github/images/filter_type_comparison.png" width="80%"></img>

| Type | Name | Description |
| :--- | :--- | :--- |
| `butter` | **Butterworth** | Maximally flat passband. Standard for acoustic measurements. |
| `cheby1` | **Chebyshev I** | Steeper roll-off than Butterworth, with ripple in the passband. |
| `ellip` | **Elliptic** | Steepest transition, with ripples in both passband and stopband. |
| `bessel` | **Bessel** | Best phase response and minimal group delay (no overshoot). |

#### Usage:
```python
# Use a high-selectivity Elliptic filter for better isolation
spl, freq = octavefilter(signal, fs, filter_type='ellip', attenuation=80.0)
```

---

## 🔊 Acoustic Weighting (A, C, Z)

Frequency weighting curves are used to simulate the human ear's sensitivity to different frequencies at different levels.

<img src=".github/images/weighting_responses.png" width="80%"></img>

*   **A-Weighting (`A`):** Standard for environmental noise and hearing protection (IEC 61672-1).
*   **C-Weighting (`C`):** Used for peak sound pressure and high-level noise analysis.
*   **Z-Weighting (`Z`):** Zero weighting, completely flat response.

```python
from pyoctaveband import weighting_filter, octavefilter

# 1. Apply A-weighting to the raw signal
weighted_signal = weighting_filter(signal, fs, curve='A')

# 2. Perform octave analysis on the weighted signal
spl, freq = octavefilter(weighted_signal, fs, fraction=1)
```

---

## ⏱️ Time Weighting and Integration

In acoustics, sound pressure level is often measured with specific time ballistics to capture the "perceived" energy over time.

<img src=".github/images/time_weighting_analysis.png" width="80%"></img>

*   **Fast (`fast`):** 125ms time constant. Used for most noise measurements.
*   **Slow (`slow`):** 1000ms time constant. Used for steady-state noise.
*   **Impulse (`impulse`):** 35ms rise time. Used for impulsive sounds like gunshots or impacts.

```python
from pyoctaveband import time_weighting

# Calculate the time-varying Mean Square value (energy envelope)
energy_envelope = time_weighting(signal, fs, mode='fast')

# Convert to instantaneous SPL (dB)
spl_t = 10 * np.log10(energy_envelope / (2e-5)**2)
```

---

## ⚡ High Performance: OctaveFilterBank Class

For applications that process many signals with the same configuration (e.g., real-time monitoring), use the `OctaveFilterBank` class. It pre-calculates the filter coefficients only once.

```python
from pyoctaveband import OctaveFilterBank

# Initialize the bank (expensive operation)
bank = OctaveFilterBank(fs=48000, fraction=3, filter_type='butter')

# Process signals efficiently (reusing SOS coefficients)
for frame in stream:
    spl, freq = bank.filter(frame)
```

---

## 🔀 Linkwitz-Riley Crossover

Used in professional audio to split signals into low and high frequency bands while maintaining a perfectly flat sum and aligned phase.

<img src=".github/images/crossover_lr4.png" width="80%"></img>

```python
from pyoctaveband import linkwitz_riley

# Split at 800 Hz using a 4th order Linkwitz-Riley crossover
low_band, high_band = linkwitz_riley(signal, fs, freq=800, order=4)

# Summing low + high results in the original signal (flat response)
original_reconstructed = low_band + high_band
```

---

## 📊 Signal Decomposition and Stability

By setting `sigbands=True`, you can retrieve the time-domain components of each band. PyOctaveBand ensures stability even in the lowest frequency bands (down to 16Hz) using high-precision poliphase resampling.

<img src=".github/images/signal_decomposition.png" width="80%"></img>

*The bottom plot shows the **Impulse Response** of a band, demonstrating the stability and decay characteristics of the filter.*

---

## 🧪 Development and Verification

PyOctaveBand includes a rigorous test suite that verifies:
- Spectral isolation (>20dB at 2 octaves).
- Standard A/C weighting gains at 100Hz, 1kHz, and 8kHz.
- Filter stability via IR tail energy analysis.
- Multichannel processing integrity.

### Run Tests
```bash
pytest tests/
```

### Generate Benchmark Report
```bash
python scripts/benchmark_filters.py
```

# Author
Jose M. Requena Plens, 2020 - 2026.