[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.com/donate?hosted_button_id=BLP3R6VGYJB4Q)
[![Donate](https://img.shields.io/badge/Donate-Ko--fi-brightgreen?color=ff5f5f)](https://ko-fi.com/jmrplens) 
[![Python application](https://github.com/jmrplens/PyOctaveBand/actions/workflows/python-app.yml/badge.svg)](https://github.com/jmrplens/PyOctaveBand/actions/workflows/python-app.yml)

# PyOctaveBand
Octave-Band and Fractional Octave-Band filter for signals in the time domain.

### Getting Started

#### Installation

To use `PyOctaveBand` in your project, you can either clone the repository or add it as a git submodule:

**Cloning the repository:**
```bash
git clone https://github.com/jmrplens/PyOctaveBand.git
cd PyOctaveBand
pip install .
```

**As a Git Submodule:**
```bash
git submodule add https://github.com/jmrplens/PyOctaveBand.git
pip install -e ./PyOctaveBand
```

#### Integration / Usage

Here is a simple example of how to use `PyOctaveBand` in your own Python project:

```python
import numpy as np
from pyoctaveband import octavefilter

# 1. Prepare your signal (e.g., a 1 second sine wave at 1000 Hz)
fs = 48000
t = np.linspace(0, 1, fs)
signal = np.sin(2 * np.pi * 1000 * t)

# 2. Apply the 1/3 octave band filter
spl, freq = octavefilter(signal, fs=fs, fraction=3)

# 3. Print results
print(f"Center Frequencies: {freq}")
print(f"SPL per band: {spl}")
```

#### Multichannel Support
PyOctaveBand supports multichannel signals. Input `x` can be a 1D array (single channel) or a 2D array with shape `(channels, samples)`.

### Public Methods

##### octavefilter
The function that filters the input signal according to the selected parameters.
```python
# Returns Sound Pressure Level and Frequency array
spl, freq = octavefilter(x, fs, fraction=1, order=6, limits=None, show=False, sigbands=False)

# Returns SPL, frequencies, and the signals filtered into bands
spl, freq, xb = octavefilter(x, fs, fraction=1, order=6, limits=None, show=False, sigbands=True)
```

##### getansifrequencies
Returns the frequency vector according to ANSI s1.11-2004 and IEC 61260-1-2014 standards.
```python
freq, freq_d, freq_u = getansifrequencies(fraction, limits=None)
```

##### normalizedfreq
Returns the normalized frequency vector according to ANSI s1.11-2004 and IEC 61260-1-2014.
```python
freq = normalizedfreq(fraction)
```

### The filter
The filter bank is designed using Butterworth filters with Second-Order Sections (SOS) coefficients. Automatic downsampling is applied to ensure filter stability and accuracy, especially for low-frequency bands.

### Examples of filter responses
| Fraction | Butterworth order: 6       | Butterworth order: 16      | 
|:-------------:|:-------------:|:-------------:|
| 1-octave | <img src=".github/images/filter_fraction_1_order_6.png" width="100%"></img>      | <img src=".github/images/filter_fraction_1_order_16.png" width="100%"></img>  |
| 1/3-octave | <img src=".github/images/filter_fraction_3_order_6.png" width="100%"></img>      | <img src=".github/images/filter_fraction_3_order_16.png" width="100%"></img>  |
| 2/3-octave | <img src=".github/images/filter_fraction_1.5_order_6.png" width="100%"></img>      | <img src=".github/images/filter_fraction_1.5_order_16.png" width="100%"></img>  |

### Signal Analysis Examples

| One Octave Analysis       | One-Third Octave Analysis      | 
|:-------------:|:-------------:|
| <img src=".github/images/signal_response_fraction_1.png" width="100%"></img>      | <img src=".github/images/signal_response_fraction_3.png" width="100%"></img>  |

#### Multichannel Processing
This plot shows the simultaneous analysis of a stereo signal (Left: Pink Noise, Right: Logarithmic Sweep).

<img src=".github/images/signal_response_multichannel.png" width="100%"></img>

#### Signal Decomposition (Time Domain)
By setting `sigbands=True`, you can retrieve the signal components for each individual frequency band.

<img src=".github/images/signal_decomposition.png" width="100%"></img>

# Development

### Running Tests
```bash
python tests/test_basic.py
python tests/test_multichannel.py
python tests/test_audio_processing.py
```

### Generating Graphs
To regenerate the images used in this README:
```bash
python generate_graphs.py
```

### Code Quality & Security
Run local checks using the `Makefile`:
```bash
make check
```

# Roadmap
- Performance optimizations for very long signals
- Support for more filter types (Chebyshev, etc.)

## Contributing
If you have any suggestions or found an error, please check [CONTRIBUTING.md](CONTRIBUTING.md) and open an [Issue](https://github.com/jmrplens/PyOctaveBand/issues) or a [Pull Request](https://github.com/jmrplens/PyOctaveBand/pulls).

# Author
Jose M. Requena Plens, 2020.
