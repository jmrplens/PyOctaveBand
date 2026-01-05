#  Copyright (c) 2020. Jose M. Requena-Plens

"""
Basic test and usage example for pyoctaveband.
"""

import os
import sys

# Ensure the local package is used
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

import pyoctaveband as PyOctaveBand

# Configuration
fs = 48000
duration = 5.0  # seconds

# Generate multi-tone signal
t = np.linspace(0, duration, int(fs * duration), endpoint=False)
freqs = [20, 100, 500, 2000, 4000, 15000]
y = 100 * np.sum([np.sin(2 * np.pi * f * t) for f in freqs], axis=0)

print("Processing signal...")

# 1. Filter and get only SPL spectrum
# show=True will display the filter response plot
spl, freq = PyOctaveBand.octavefilter(y, fs=fs, fraction=3, order=6, limits=[12, 20000], show=False)

# 2. Filter and get signals in time-domain bands
spl_b, freq_b, xb = PyOctaveBand.octavefilter(
    y, fs=fs, fraction=3, order=6, limits=[12, 20000], show=False, sigbands=True
)

# Save the filtered bands to WAV files
print(f"Saving {len(freq_b)} band files...")
for idx, f_center in enumerate(freq_b):
    filename = f"test_band_{round(f_center)}_Hz.wav"
    # Normalize for WAV export
    band_audio = xb[idx] / np.max(np.abs(xb[idx]))
    wavfile.write(filename, fs, band_audio.astype(np.float32))

# Visualize the resulting spectrum
print("Showing results plot...")
plt.figure(figsize=(10, 6))
plt.semilogx(freq, spl, "b-o", linewidth=1.5, markerfacecolor="white")
plt.grid(which="major", color="#e0e0e0", linestyle="-")
plt.grid(which="minor", color="#e0e0e0", linestyle=":", alpha=0.4)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Level [dB]")
plt.title("Analyzed Signal Spectrum (1/3 Octave)")
plt.xlim(11, 25000)

xticks = [16, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
xticklabels = ["16", "31.5", "63", "125", "250", "500", "1k", "2k", "4k", "8k", "16k"]
plt.xticks(xticks, xticklabels)

plt.tight_layout()
plt.show()