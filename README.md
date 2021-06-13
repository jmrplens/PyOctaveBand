[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.com/donate?hosted_button_id=BLP3R6VGYJB4Q)
[![Donate](https://img.shields.io/badge/Donate-Ko--fi-brightgreen?color=ff5f5f)](https://ko-fi.com/jmrplens) 

# PyOctaveBand
Octave-Band and Fractional Octave-Band filter. For signal in time domain.

### Public Methods

##### octavefilter
The function that filters the input signal according to the selected parameters.
```python
x # signal
fs # sample rate
fraction # Bandwidth 'b'. Examples: 1/3-octave b=3, 1-octave b=1, 2/3-octave b = 3/2. [Optional] Default: 1
order # Order of Butterworth filter. [Optional] Default: 6.
limits # Minimum and maximum limit frequencies. [Optional] Default [12,20000]
show # Boolean for plot o not the filter response.
sigbands # Boolean to also return the signal in the time domain divided into bands. A list with as many arrays as there are frequency bands

# Only octave spectra
spl, freq = octavefilter(x, fs, fraction=1, order=6, limits=None, show=0, sigbands=0)

# Octave spectra and bands in time domain
spl, freq, xb = octavefilter(x, fs, fraction=1, order=6, limits=None, show=0, sigbands=1)
```

##### getansifrequencies
Returns the frequency vector according to ANSI s1.11-2004 and IEC 61260-1-2014 standards.

```python
fraction # Bandwidth 'b'. Examples: 1/3-octave b=3, 1-octave b=1, 2/3-octave b = 3/2.
limits # Minimum and maximum limit frequencies. [Optional] Default [12,20000]
freq = getansifrequencies(fraction, limits=None)
```

##### normalizedfreq
Returns the normalized frequency vector according to ANSI s1.11-2004 and IEC 61260-1-2014. Only for octave and third octave bands.
```python
fraction # Bandwidth 'b'. For 1/3-octave b=3 and b=1 for one-octave.
freq = normalizedfreq(fraction)
```

### The filter
The filter used to design the octave filter bank is a Butterworth with SOS coefficients. You can find more information about the filter here: [scipy.signal.butter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html).

### Frequency values
The values of the center frequencies and the upper and lower edges are obtained with the calculation defined in the ANSI s1.11-2004 and IEC 61260-1-2014 standards.

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

### Anti-aliasing
The frequency bands of the filters that are above the Nyquist's frequency (`sample_rate/2`) are automatically removed because the values will not be correct.


### Examples of filter responses
| Fraction | Butterworth order: 6       | Butterworth order: 16      | 
|:-------------:|:-------------:|:-------------:|
| 1-octave | <img src="http://jmrplens.com/GitHub_PyOctave/one.png" width="100%"></img>      | <img src="http://jmrplens.com/GitHub_PyOctave/one16.png" width="100%"></img>  |
| 1/3-octave | <img src="http://jmrplens.com/GitHub_PyOctave/third.png" width="100%"></img>      | <img src="http://jmrplens.com/GitHub_PyOctave/third16.png" width="100%"></img>  |
| 2/3-octave | <img src="http://jmrplens.com/GitHub_PyOctave/twothird.png" width="100%"></img>      | <img src="http://jmrplens.com/GitHub_PyOctave/twothird16.png" width="100%"></img>  |

### Usage example

This example is included in the file test.py.

```python
import PyOctaveBand
import numpy as np
import scipy.io.wavfile

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

# Filter (only octave spectra)
spl, freq = PyOctaveBand.octavefilter(y, fs=fs, fraction=3, order=6, limits=[12, 20000], show=1)

# Filter (get spectra and signal in bands)
splb, freqb, xb = PyOctaveBand.octavefilter(y, fs=fs, fraction=3, order=6, limits=[12, 20000], show=0, sigbands=1)

# Store signal in bands in separated wav files
for idx in range(len(freq)):
    scipy.io.wavfile.write(
            "test_"+str(round(freq[idx]))+"_Hz.wav",
            fs,
            xb[idx]/np.max(xb[idx]))
```

The result is as follows:

| One Octave filter       | One-Third Octave filter      | 
|:-------------:|:-------------:|
| <img src="http://jmrplens.com/GitHub_PyOctave/response1.png" width="100%"></img>      | <img src="http://jmrplens.com/GitHub_PyOctave/response.png" width="100%"></img>  |

| 1/12 Octave filter       | 1/24 Octave filter      | 
|:-------------:|:-------------:|
| <img src="http://jmrplens.com/GitHub_PyOctave/12.png" width="100%"></img>      | <img src="http://jmrplens.com/GitHub_PyOctave/24.png" width="100%"></img>  |

# Roadmap

- Add multichannel support

If you have any suggestions or you found an error please, make a [Pull Request](https://github.com/jmrplens/PyOctave/pulls) or [contact me](mailto:info@jmrplens.com).

# Author
Jose M. Requena Plens, 2020. joreple@upv.es

[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.com/donate?hosted_button_id=BLP3R6VGYJB4Q)
[![Donate](https://img.shields.io/badge/Donate-Ko--fi-brightgreen?color=ff5f5f)](https://ko-fi.com/jmrplens) 
