# PyOctave
Octave-Band and Fractional Octave-Band filter. For signal in time domain.

### Automatic downsampling
To obtain the best filter coefficients, especially at low frequency, it is necessary to downsampling, this is done automatically by calculating the necessary downsampling factor for each frequency band.

```python
fs # sample rate
freq # frequency
factor = ((fs / 2) / freq)
```
The resampling is done with the decimate function of the [SciPy library](https://www.scipy.org/scipylib/index.html):

```python
x # signal
xdown = scipy.signal.decimate(x, factor)
```

### Examples of filter responses
| Fraction | Butterworth order: 6       | Butterworth order: 16      | 
|:-------------:|:-------------:|:-------------:|
| 1-octave | <img src="http://jmrplens.com/GitHub_PyOctave/one.png" width="100%"></img>      | <img src="http://jmrplens.com/GitHub_PyOctave/one16.png" width="100%"></img>  |
| 1/3-octave | <img src="http://jmrplens.com/GitHub_PyOctave/third.png" width="100%"></img>      | <img src="http://jmrplens.com/GitHub_PyOctave/third16.png" width="100%"></img>  |
| 2/3-octave | <img src="http://jmrplens.com/GitHub_PyOctave/twothird.png" width="100%"></img>      | <img src="http://jmrplens.com/GitHub_PyOctave/twothird16.png" width="100%"></img>  |

### Usage example

This example is included in the file test.py.

```python
import PyOctave
import numpy as np

# Sample rate and duration
fs = 48000
duration = 5  # In seconds

# Time array
x = np.arange(np.round(fs * duration)) / fs

# Signal with 6 frequencies
f1, f2, f3, f4, f5, f6 = 20, 100, 500, 2000, 4000, 15000
# Multi Sine wave signal
y = 100 \
    * (np.sin(2 * np.pi * f1 * x)
       + np.sin(2 * np.pi * f2 * x)
       + np.sin(2 * np.pi * f3 * x)
       + np.sin(2 * np.pi * f4 * x)
       + np.sin(2 * np.pi * f5 * x)
       + np.sin(2 * np.pi * f6 * x))

# Filter
spl, freq = PyOctave.octaveFilter(y, fs=fs, fraction=3, order=6, limits=[12, 20000], show=1)
```

The result is as follows:

| One Octave filter       | One-Third Octave filter      | 
|:-------------:|:-------------:|
| <img src="http://jmrplens.com/GitHub_PyOctave/response1.png" width="100%"></img>      | <img src="http://jmrplens.com/GitHub_PyOctave/response.png" width="100%"></img>  |
