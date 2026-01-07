[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.com/donate?hosted_button_id=BLP3R6VGYJB4Q)
[![Donate](https://img.shields.io/badge/Donate-Ko--fi-brightgreen?color=ff5f5f)](https://ko-fi.com/jmrplens) 
[![PyPI version](https://img.shields.io/pypi/v/PyOctaveBand)](https://pypi.org/project/PyOctaveBand/)
[![Python application](https://github.com/jmrplens/PyOctaveBand/actions/workflows/python-app.yml/badge.svg)](https://github.com/jmrplens/PyOctaveBand/actions/workflows/python-app.yml)

# PyOctaveBand
Advanced Octave-Band and Fractional Octave-Band filter bank for signals in the time domain. Fully compliant with **ANSI S1.11-2004** (Filters) and **IEC 61672-1:2013** (Time Weighting).

This library provides professional-grade tools for acoustic analysis, including frequency weighting (A, C, Z), time ballistics (Fast, Slow, Impulse), and multiple filter architectures.

Now available on [PyPI](https://pypi.org/project/PyOctaveBand/).

---

## 📑 Table of Contents
1. [🚀 Getting Started](#-getting-started)
    - [Installation](#installation)
    - [Basic Usage](#basic-usage-13-octave-analysis)
2. [🛠️ Filter Architectures](#️-filter-architectures)
    - [Filter Comparison and Zoom](#filter-comparison-and-zoom)
    - [Gallery of Responses](#gallery-of-filter-bank-responses)
3. [🔊 Acoustic Weighting (A, C, Z)](#-acoustic-weighting-a-c-z)
4. [⏱️ Time Weighting and Integration](#️-time-weighting-and-integration)
5. [⚡ Performance: Multichannel & Vectorization](#-performance-multichannel--vectorization)
6. [🔍 Filter Usage and Examples](#-filter-usage-and-examples)
    - [1. Butterworth](#1-butterworth-butter)
    - [2. Chebyshev I](#2-chebyshev-i-cheby1)
    - [3. Chebyshev II](#3-chebyshev-ii-cheby2)
    - [4. Elliptic](#4-elliptic-ellip)
    - [5. Bessel](#5-bessel-bessel)
    - [6. Linkwitz-Riley](#6-linkwitz-riley-lr)
7. [📏 Calibration and dBFS](#-calibration-and-dbfs)
    - [Physical Calibration](#physical-calibration-sound-level-meter)
    - [Digital Analysis (dBFS)](#digital-analysis-dbfs)
8. [📊 Signal Decomposition](#-signal-decomposition-and-stability)
9. [📖 Theory and Equations](#-theoretical-background)
    - [Octave Band Frequencies](#octave-band-frequencies-ansi-s111--iec-61260)
    - [Magnitude Responses](#magnitude-responses-hjw)
    - [Weighting Curves](#weighting-curves-iec-61672-1)
    - [Time Integration](#time-integration)
10. [🧪 Testing and Quality](#-development-and-verification)
    - [Test Categories](#test-categories)
    - [Commands](#commands)

---

## 🚀 Getting Started

### Installation

**Option 1: From PyPI (Recommended)**
Install `PyOctaveBand` directly using pip:
```bash
pip install PyOctaveBand
```

**Option 2: Cloning and Installing**
Clone the repository and install it manually:
```bash
git clone https://github.com/jmrplens/PyOctaveBand.git
cd PyOctaveBand
pip install .
```

**Option 3: Git Submodule**
Add `PyOctaveBand` as a dependency within your own git repository:
```bash
git submodule add https://github.com/jmrplens/PyOctaveBand.git
# Then install in editable mode to use it from your project
pip install -e ./PyOctaveBand
```

---

## 📖 Quick API Reference

All core functionality can be imported directly from the `pyoctaveband` package.

| Name | Type | Description (Inputs) | Usage Snippet (Outputs) |
| :--- | :--- | :--- | :--- |
| `octavefilter` | `function` | **High-level analysis.**<br>• `x`: Signal array<br>• `fs`: Sample rate [Hz]<br>• `fraction`: 1, 3, etc. (Default: 1)<br>• `order`: Filter order (Default: 6)<br>• `limits`: [f_min, f_max] (Default: [12, 20000])<br>• `filter_type`: 'butter', 'cheby1', 'cheby2', 'ellip', 'bessel' (Default: 'butter')<br>• `sigbands`: Return time signals (Default: False)<br>• `detrend`: Remove DC offset (Default: True)<br>• `calibration_factor`: Sensitivity multiplier (Default: 1.0)<br>• `dbfs`: Output in dBFS instead of dB SPL (Default: False)<br>• `mode`: 'rms' or 'peak' (Default: 'rms')<br>• `show`: Plot response (Default: False)<br>• `plot_file`: Path to save plot (Default: None)<br>• `ripple`: Passband ripple [dB] (for cheby1/ellip)<br>• `attenuation`: Stopband atten. [dB] (for cheby2/ellip) | `spl, freq = octavefilter(x, fs, ...)`<br>• `spl`: levels [dB]<br>• `freq`: frequencies [Hz]<br><br>**With `sigbands=True`:**<br>`spl, freq, xb = octavefilter(x, fs, sigbands=True)`<br>• `xb`: List of filtered signals (one per band)<br><br>**Calibrated usage:**<br>`spl, f = octavefilter(x, fs, calibration_factor=0.05)` |
| `OctaveFilterBank` | `class` | **Efficient bank implementation.**<br>• `fs`: Sample rate [Hz]<br>• `fraction`: 1, 3, etc.<br>• `order`: Filter order<br>• `limits`: [f_min, f_max] (Default: [12, 20000])<br>• `filter_type`: Architecture name<br>• `show`: Plot response (Default: False)<br>• `plot_file`: Path to save plot (Default: None)<br>• `calibration_factor`: Sensitivity multiplier<br>• `dbfs`: Use dBFS (Default: False)<br>• `ripple`: Passband ripple [dB]<br>• `attenuation`: Stopband attenuation [dB] | `bank = OctaveFilterBank(fs=48000, fraction=3, order=6, filter_type='butter', show=True)`<br>`spl, f = bank.filter(x, sigbands=False, mode='rms', detrend=True)`<br><br>• `bank`: Instance of the filter bank |
| `weighting_filter` | `function` | **Acoustic weighting.**<br>• `x`: Signal array<br>• `fs`: Sample rate [Hz]<br>• `curve`: 'A', 'C', or 'Z' (Default: 'A') | `y = weighting_filter(x, fs, curve='A')`<br><br>• `y`: 1D array of weighted signal |
| `time_weighting` | `function` | **Energy capture.**<br>• `x`: Raw signal array (squared internally)<br>• `fs`: Sample rate [Hz]<br>• `mode`: 'fast', 'slow', or 'impulse' | `env = time_weighting(x, fs, mode='fast')`<br><br>• `env`: 1D array of energy envelope (Mean Square) |
| `linkwitz_riley` | `function` | **Audio crossover.**<br>• `x`: Signal array<br>• `fs`: Sample rate [Hz]<br>• `freq`: Crossover frequency [Hz]<br>• `order`: Any even number (Default: 4) | `lo, hi = linkwitz_riley(x, fs, freq=1000, order=4)`<br><br>• `lo`: Low-pass filtered signal<br>• `hi`: High-pass filtered signal |
| `calculate_sensitivity` | `function`| **SPL Calibration.**<br>• `ref_signal`: Calibration signal<br>• `target_spl`: Level of calibrator (Default: 94.0)<br>• `ref_pressure`: Reference pressure (Default: 20e-6) | `s = calculate_sensitivity(ref_signal, target_spl=94.0)`<br><br>• `s`: Float (multiplier for pressure) |
| `getansifrequencies` | `function` | **ANSI Frequency generator.**<br>• `fraction`: 1, 3, etc. (Required)<br>• `limits`: [f_min, f_max] (Default: [12, 20000]) | `f_cen, f_low, f_high = getansifrequencies(fraction=3)`<br><br>• `f_cen`: List of center frequencies [Hz]<br>• `f_low`: List of lower edges [Hz]<br>• `f_high`: List of upper edges [Hz] |
| `normalizedfreq` | `function` | **Standard IEC Frequencies.**<br>• `fraction`: 1 or 3 | `freqs = normalizedfreq(fraction=3)`<br><br>• `freqs`: List of standard center frequencies [Hz] |

---

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

# OR: Import an audio file
from scipy.io import wavfile

# Load standard WAV file
fs, signal = wavfile.read("measurement.wav")

# Analyze
# Note: To obtain real-world SPL values, you must calibrate the input.
# See the [Physical Calibration](#physical-calibration-sonómetro) section.
spl, freq = octavefilter(signal, fs=fs, fraction=3)
```

<img src="https://raw.githubusercontent.com/jmrplens/PyOctaveBand/main/.github/images/signal_response_fraction_3.png" width="80%"></img>

*Example of a 1/3 Octave Band spectrum analysis of a complex signal.*

### Multichannel Support
PyOctaveBand natively supports multichannel signals (e.g., Stereo, 5.1, Microphone Arrays) using **fully vectorized operations**. Input arrays of shape `(N_channels, N_samples)` are processed in parallel, offering significant performance gains over iterative loops.

<img src="https://raw.githubusercontent.com/jmrplens/PyOctaveBand/main/.github/images/signal_response_multichannel.png" width="80%"></img>

*Simultaneous analysis of a Stereo signal: Left Channel (Pink Noise) vs Right Channel (Log Sine Sweep).*

---

## 🛠️ Filter Architectures

PyOctaveBand supports several filter types, each with its own transfer function characteristic.

### Filter Comparison and Zoom
We use Second-Order Sections (SOS) for all filters to ensure numerical stability. The following plot compares the architectures focusing on the -3 dB crossover point.

<img src="https://raw.githubusercontent.com/jmrplens/PyOctaveBand/main/.github/images/filter_type_comparison.png" width="80%"></img>

| Type | Name | Usage Example | Best For | 
| :--- | :--- | :--- | :--- |
| `butter` | **Butterworth** | `octavefilter(x, fs, filter_type='butter')` | General acoustic measurement. |
| `cheby1` | **Chebyshev I** | `octavefilter(x, fs, filter_type='cheby1', ripple=0.1)` | Sharper roll-off at the cost of ripple. |
| `cheby2` | **Chebyshev II** | `octavefilter(x, fs, filter_type='cheby2', attenuation=60)` | Flat passband with stopband zeros. |
| `ellip` | **Elliptic** | `octavefilter(x, fs, filter_type='ellip', ripple=0.1, attenuation=60)` | Maximum selectivity. |
| `bessel` | **Bessel** | `octavefilter(x, fs, filter_type='bessel')` | Preserving transient waveform shapes. |

### Gallery of Filter Bank Responses
Full spectral view of the filter banks for Octave (1/1) and 1/3-Octave fractions.

| Architecture | 1/1 Octave (Fraction=1) | 1/3 Octave (Fraction=3) |
| :--- | :--- | :--- |
| **Butterworth** | <img src="https://raw.githubusercontent.com/jmrplens/PyOctaveBand/main/.github/images/filter_butter_fraction_1_order_6.png" width="100%"> | <img src="https://raw.githubusercontent.com/jmrplens/PyOctaveBand/main/.github/images/filter_butter_fraction_3_order_6.png" width="100%"> |
| **Chebyshev I** | <img src="https://raw.githubusercontent.com/jmrplens/PyOctaveBand/main/.github/images/filter_cheby1_fraction_1_order_6.png" width="100%"> | <img src="https://raw.githubusercontent.com/jmrplens/PyOctaveBand/main/.github/images/filter_cheby1_fraction_3_order_6.png" width="100%"> |
| **Chebyshev II** | <img src="https://raw.githubusercontent.com/jmrplens/PyOctaveBand/main/.github/images/filter_cheby2_fraction_1_order_6.png" width="100%"> | <img src="https://raw.githubusercontent.com/jmrplens/PyOctaveBand/main/.github/images/filter_cheby2_fraction_3_order_6.png" width="100%"> |
| **Elliptic** | <img src="https://raw.githubusercontent.com/jmrplens/PyOctaveBand/main/.github/images/filter_ellip_fraction_1_order_6.png" width="100%"> | <img src="https://raw.githubusercontent.com/jmrplens/PyOctaveBand/main/.github/images/filter_ellip_fraction_3_order_6.png" width="100%"> |
| **Bessel** | <img src="https://raw.githubusercontent.com/jmrplens/PyOctaveBand/main/.github/images/filter_bessel_fraction_1_order_6.png" width="100%"> | <img src="https://raw.githubusercontent.com/jmrplens/PyOctaveBand/main/.github/images/filter_bessel_fraction_3_order_6.png" width="100%"> |

---

## 🔊 Acoustic Weighting (A, C, Z)

Frequency weighting curves simulate the human ear\'s sensitivity.

<img src="https://raw.githubusercontent.com/jmrplens/PyOctaveBand/main/.github/images/weighting_responses.png" width="80%"></img>

*   **A-Weighting (`A`):** Standard for environmental noise (IEC 61672-1).
*   **C-Weighting (`C`):** Used for peak sound pressure and high-level noise.
*   **Z-Weighting (`Z`):** Zero weighting, completely flat response.

```python
from pyoctaveband import weighting_filter

# Apply A-weighting to the raw signal
weighted_signal = weighting_filter(signal, fs, curve='A')

# Apply C-weighting for peak analysis
c_weighted_signal = weighting_filter(signal, fs, curve='C')
```

---

## ⏱️ Time Weighting and Integration

Accurate SPL measurement requires capturing energy over specific time windows. PyOctaveBand implements exact time constants per **IEC 61672-1:2013**.

<img src="https://raw.githubusercontent.com/jmrplens/PyOctaveBand/main/.github/images/time_weighting_analysis.png" width="80%"></img>

*   **Fast (`fast`):** $\tau = 125$ ms. Standard for noise fluctuations.
*   **Slow (`slow`):** $\tau = 1000$ ms. Standard for steady noise.
*   **Impulse (`impulse`):** **Asymmetric** ballistics. 35 ms rise time for rapid onset capture, 1500 ms decay for readability.

```python
from pyoctaveband import time_weighting

# Calculate energy envelope (Mean Square)
energy_envelope = time_weighting(signal, fs, mode='fast')
# dB SPL relative to 20μPa
spl_t = 10 * np.log10(energy_envelope / (2e-5)**2)
```

---

## ⚡ Performance: Multichannel & Vectorization

The `OctaveFilterBank` class is highly optimized for real-time and batch processing. It uses NumPy vectorization to handle multichannel audio arrays (e.g., 64-channel microphone arrays) without explicit Python loops, ensuring maximum throughput.

```python
from pyoctaveband import OctaveFilterBank

bank = OctaveFilterBank(fs=48000, fraction=3, filter_type='butter')

# Access computed properties
# bank.freq (center), bank.freq_d (lower), bank.freq_u (upper), bank.sos (coefficients)

# Process multiple signals efficiently
for frame in stream:
    # detrend=True (default) removes DC offset to improve low-freq accuracy
    spl, freq = bank.filter(frame, detrend=True)
```

---

## 🔍 Filter Usage and Examples

This section provides detailed examples and characteristics for each supported filter architecture.

### 1. Butterworth (`butter`)
The Butterworth filter is known for its **maximally flat passband**. It is the standard choice for acoustic measurements where no ripple is allowed within the frequency bands.

```python
from pyoctaveband import octavefilter
# Default standard measurement
spl, freq = octavefilter(x, fs, filter_type='butter')
```
<img src="https://raw.githubusercontent.com/jmrplens/PyOctaveBand/main/.github/images/filter_butter_fraction_3_order_6.png" width="60%"></img>

### 2. Chebyshev I (`cheby1`)
Chebyshev Type I filters provide a **steeper roll-off** than Butterworth at the expense of ripples in the passband. Useful when high selectivity is needed near the cut-off frequencies.

```python
# Selectivity with 0.1 dB passband ripple
spl, freq = octavefilter(x, fs, filter_type='cheby1', ripple=0.1)
```
<img src="https://raw.githubusercontent.com/jmrplens/PyOctaveBand/main/.github/images/filter_cheby1_fraction_3_order_6.png" width="60%"></img>

### 3. Chebyshev II (`cheby2`)
Also known as Inverse Chebyshev, it has a **flat passband** and ripples in the stopband. It provides faster roll-off than Butterworth without affecting the signal in the passband.

```python
# Flat passband with 60 dB stopband attenuation
spl, freq = octavefilter(x, fs, filter_type='cheby2', attenuation=60)
```
<img src="https://raw.githubusercontent.com/jmrplens/PyOctaveBand/main/.github/images/filter_cheby2_fraction_3_order_6.png" width="60%"></img>

### 4. Elliptic (`ellip`)
Elliptic (Cauer) filters have the **shortest transition width** (steepest roll-off) for a given order. They feature ripples in both the passband and stopband.

```python
# Maximum selectivity for extreme band isolation
spl, freq = octavefilter(x, fs, filter_type='ellip', ripple=0.1, attenuation=60)
```
<img src="https://raw.githubusercontent.com/jmrplens/PyOctaveBand/main/.github/images/filter_ellip_fraction_3_order_6.png" width="60%"></img>

### 5. Bessel (`bessel`)
Bessel filters are optimized for **linear phase response** and minimal group delay. They preserve the shape of filtered waveforms (transients) better than any other type, but have the slowest roll-off.

```python
# Best for pulse analysis and transient preservation
spl, freq = octavefilter(x, fs, filter_type='bessel')
```
<img src="https://raw.githubusercontent.com/jmrplens/PyOctaveBand/main/.github/images/filter_bessel_fraction_3_order_6.png" width="60%"></img>

### 6. Linkwitz-Riley (`lr`)
Specifically designed for **audio crossovers**. Linkwitz-Riley filters (typically 4th order, but any even order is supported) allow splitting a signal into bands that, when summed, result in a perfectly flat magnitude response and zero phase difference between bands at the crossover.

```python
from pyoctaveband import linkwitz_riley
# Split signal into Low and High bands at 1000 Hz
low, high = linkwitz_riley(signal, fs, freq=1000, order=4)
# Reconstruction: low + high == signal (flat response)
```
<img src="https://raw.githubusercontent.com/jmrplens/PyOctaveBand/main/.github/images/crossover_lr4.png" width="60%"></img>

---

## 📏 Calibration and dBFS

PyOctaveBand can return results in physical **Sound Pressure Level (dB SPL)** or digital **decibels relative to Full Scale (dBFS)**.

### Physical Calibration (Sound Level Meter)
To get accurate SPL measurements from a digital recording, you must first calculate the sensitivity of your measurement chain using a reference tone (e.g., 94 dB @ 1kHz).

```python
from pyoctaveband import octavefilter, calculate_sensitivity

# 1. Record your 94dB calibrator signal
# ref_signal = ... (your recording)

# 2. Calculate sensitivity factor
sensitivity = calculate_sensitivity(ref_signal, target_spl=94.0)

# 3. Apply calibration to your measurements
spl, freq = octavefilter(signal, fs, calibration_factor=sensitivity)
# Now 'spl' values are in real-world dB SPL!
```

### Digital Analysis (dBFS)
If you are working with digital audio files (e.g., WAV, FLAC) and want to analyze levels relative to Full Scale rather than physical pressure, you can use the `dbfs=True` parameter.

In this mode:
*   **0 dBFS** corresponds to a numeric signal level of 1.0 (RMS or Peak).
*   Useful for analyzing headroom, digital mastering, or normalized signals.

```python
# Assume 'signal' is normalized between -1.0 and 1.0
spl_dbfs, freq = octavefilter(signal, fs, dbfs=True)
# Results will be negative (e.g., -20 dBFS)
```

### RMS vs Peak Levels
PyOctaveBand supports two measurement modes to align with professional software like BK:
- **RMS (`mode='rms'`)**: Energy-based level (standard).
- **Peak (`mode='peak'`)**: Absolute maximum value reached in the frame (Peak-holding).

```python
# Measure peak-holding levels for impact analysis
spl_peak, freq = octavefilter(signal, fs, mode='peak')
```

---

## 📊 Signal Decomposition and Stability

By setting `sigbands=True`, you can retrieve the time-domain components of each band. This allows for advanced analysis or comparing how different architectures (e.g., Butterworth vs Chebyshev) affect the signal phase and transient response.

```python
import numpy as np
from pyoctaveband import octavefilter

# 1. Generate a signal (Sum of 250Hz and 1000Hz)
fs = 48000
t = np.linspace(0, 0.5, int(fs * 0.5), endpoint=False)
y = np.sin(2 * np.pi * 250 * t) + np.sin(2 * np.pi * 1000 * t)

# 2. Compare architectures (Butterworth vs Chebyshev II)
# Filter with Butterworth (default)
spl_b, freq, xb_butter = octavefilter(y, fs=fs, fraction=1, sigbands=True, filter_type='butter')

# Filter with Chebyshev II (flat passband, ripples in stopband)
spl_c2, _, xb_cheby2 = octavefilter(y, fs=fs, fraction=1, sigbands=True, filter_type='cheby2')

# 'xb_butter' and 'xb_cheby2' contain the time-domain signals per band
```

<img src="https://raw.githubusercontent.com/jmrplens/PyOctaveBand/main/.github/images/signal_decomposition.png" width="80%"></img>

*The plot compares the **Butterworth** (solid blue) and **Chebyshev II** (dashed red) responses. The bottom plot shows the **Impulse Response**, highlighting the differences in stability and transient decay.*

> [!NOTE]
> **Why do the signals look shifted in time?**
> Digital IIR filters (like Butterworth or Chebyshev) have **non-linear phase responses**, which results in frequency-dependent **Group Delay**. In the 250 Hz band, you can see that the Chebyshev II filter has a different propagation delay compared to the Butterworth filter. This is a normal physical property of these architectures: more aggressive frequency roll-offs usually come at the cost of higher group delay and phase distortion.

---

## 📖 Theoretical Background

### Octave Band Frequencies (ANSI S1.11 / IEC 61260)
The mid-band frequencies (fm) and edges (f1, f2) use a base-10 ratio:

$$
G = 10^{0.3}
$$

**Mid-band:**

$$
f_m = 1000 \cdot G^{x/b}
$$

(for odd b)

**Band edges:**

$$
f_1 = f_m \cdot G^{-1/2b}, \quad f_2 = f_m \cdot G^{1/2b}
$$

### Magnitude Responses |H(jw)|
The library implements standard classical filter prototypes:

**1. Butterworth:** Maximally flat passband.

$$
|H(j\omega)| = \frac{1}{\sqrt{1 + (\omega/\omega_c)^{2n}}}
$$

**2. Chebyshev I:** Equiripple in passband, steeper roll-off.

$$
|H(j\omega)| = \frac{1}{\sqrt{1 + \epsilon^2 T_n^2(\omega/\omega_c)}}
$$

**3. Chebyshev II:** Inverse Chebyshev, equiripple in stopband, flat passband.

$$
|H(j\omega)| = \frac{1}{\sqrt{1 + \frac{1}{\epsilon^2 T_n^2(\omega_{stop}/\omega)}}}
$$

**4. Elliptic:** Equiripple in both, maximum selectivity.

$$
|H(j\omega)| = \frac{1}{\sqrt{1 + \epsilon^2 R_n^2(\omega/\omega_c, L)}}
$$

**5. Bessel:** Maximally flat group delay (linear phase).

$$
H(s) = \frac{\theta_n(0)}{\theta_n(s/\omega_0)}
$$

(Where $\theta_n$ is the reverse Bessel polynomial)

### Filter Bank Design & Numerical Stability
To ensure **100% stability** across the entire audible spectrum (even at low frequencies like 16Hz with high sample rates), PyOctaveBand employs two critical strategies:

1.  **Second-Order Sections (SOS):** All filters are implemented as a series of cascaded biquads. This avoids the catastrophic numerical precision loss associated with high-order transfer functions (coefficients a, b).
2.  **Multi-rate Decimation:** For low-frequency bands, the signal is automatically downsampled (decimated) before filtering and upsampled afterwards. This keeps the digital pole locations far from the unit circle boundary, preventing oscillation and noise.

### Weighting Curves (IEC 61672-1)
The A-weighting transfer function:

$$
R_A(f) = \frac{12194^2 \cdot f^4}{(f^2 + 20.6^2)\sqrt{(f^2 + 107.7^2)(f^2 + 737.9^2)}(f^2 + 12194^2)}
$$

$$
A(f) = 20 \log_{10}(R_A(f)) + 2.00
$$

### Time Integration
Implemented as a first-order IIR exponential integrator:

$$
y[n] = \alpha \cdot x^2[n] + (1 - \alpha) \cdot y[n-1]
$$

$$
\alpha = 1 - e^{-1 / (f_s \cdot \tau)}
$$

Where `tau` is the time constant (e.g., 125ms for Fast).

---

## 🧪 Development and Verification

We maintain 100% stability and compliance through a rigorous test suite.

### Test Categories
1.  **Isolation Tests:** Verifies that a pure 1kHz tone is attenuated by >20dB in the 250Hz and 4kHz bands.
2.  **Weighting Response:** Checks gains at 100Hz (-19.1dB for A) and 1kHz (0dB).
3.  **Stability (IR Tail):** Analyzes the Impulse Response of every filter. Energy in the last 100ms must be $< 10^{-6}$ to pass.
4.  **Crossover Flatness:** Verifies that the sum of Linkwitz-Riley bands has $< 0.1$ dB deviation.

### Commands
```bash
# Run full suite
pytest tests/

# Generate technical report
python scripts/benchmark_filters.py
```

# Author
Jose M. Requena Plens, 2020 - 2026.
