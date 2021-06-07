#  Copyright (c) 2020. Jose M. Requena-Plens

"""
Demo test for PyOctaveBand.py
"""

import PyOctaveBand
import numpy as np
import matplotlib.pyplot as plt
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


# Show octave spectrum
fig, ax = plt.subplots()
ax.semilogx(freq, spl, 'b')
ax.grid(which='major')
ax.grid(which='minor', linestyle=':')
ax.set_xlabel(r'Frequency [Hz]')
ax.set_ylabel('Level [dB]')
plt.xlim(11, 25000)
ax.set_xticks([16, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
ax.set_xticklabels(['16', '31.5', '63', '125', '250', '500', '1k', '2k', '4k', '8k', '16k'])
plt.show()

