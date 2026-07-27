← [Documentation index](README.md)

# Objective intelligibility (STOI & ESTOI)

**STOI** and **ESTOI** are correlation-based objective intelligibility measures.
Each compares a clean reference to a degraded or processed version of the same
speech and returns a scalar with a monotonic relation to the fraction of words
a listener would understand: `1` when the degraded signal equals the clean one,
and near `0` for uncorrelated noise. Unlike the
[Speech Transmission Index](speech-transmission.md), which characterises a
transmission channel, and the [Speech Intelligibility Index](speech-intelligibility.md),
which predicts audibility from spectra, STOI and ESTOI work directly on the two
waveforms, which makes them the standard yardstick for **time-frequency
weighted** speech: noise reduction, source separation and binary-mask
processing, where separating the clean speech from its distortion is not
straightforward.

- **STOI** (Taal, Hendriks, Heusdens and Jensen, 2011) averages the correlation
  between the clean and degraded short-time temporal envelopes over
  one-third-octave bands and 384 ms segments, after a per-segment normalisation
  and a signal-to-distortion clipping.
- **ESTOI** (Jensen and Taal, 2016) mean- and variance-normalises the
  short-time band-by-frame spectrogram over both its rows (band envelopes) and
  its columns (spectra), so the intermediate measure is a spectral correlation.
  It tracks intelligibility better under **modulated maskers** and competing
  talkers, where STOI's band-independent correlation misses the benefit of the
  quiet gaps.

> [!NOTE]
> `pystoi` (a public reimplementation of the authors' MATLAB) is used only as
> an external cross-check in the test suite. phonometry reimplements both
> measures from the two papers and never imports `pystoi` at runtime; the two
> agree to well under `1e-3` on shared inputs.

## 1. The shared front end

Both measures run the same processing before they diverge (Taal et al. 2011,
Section II):

1. **Resample to 10 kHz.** A rate chosen to cover the speech-bearing range; the
   library resamples internally, so the sample rate of the inputs is free.
2. **Short-time transform.** 256-sample (25.6 ms) frames, 50 % overlap, a Hann
   window, zero-padded to a 512-point DFT.
3. **Remove silent frames.** Frames whose *clean* energy is more than 40 dB
   below the loudest clean frame carry no intelligibility and are dropped from
   both signals.
4. **One-third-octave grouping.** The DFT magnitudes are grouped into 15 bands
   from a lowest centre of 150 Hz, giving a band-by-frame envelope
   spectrogram.
5. **384 ms segments.** 30-frame sliding segments are the unit of comparison,
   long enough to carry the slow modulations that matter for intelligibility.

```python
import numpy as np
from phonometry import stoi

fs = 16000
rng = np.random.default_rng(0)
# A clean reference and a noisy version at ~5 dB SNR.
clean = rng.standard_normal(3 * fs)          # stand-in for a speech waveform
noise = rng.standard_normal(3 * fs)
degraded = clean + 0.56 * noise

d = stoi(clean, degraded, fs)
print(round(d.value, 3))              # a scalar in roughly [0, 1]
print(stoi(clean, clean, fs).value)   # 1.0  (a signal against itself)
```

Everything up to the 384 ms segments is shared; the two measures only part
ways at the correlation. The diagram lays the chain out with the numbers of
the flat-masker example.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_objective_intelligibility_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_objective_intelligibility.svg" alt="Block diagram of the STOI and ESTOI processing chain: a clean reference and a degraded version, in the guide's example speech-like material in a flat masker at 0 dB signal-to-noise ratio, are resampled to 10 kilohertz with silent frames dropped, transformed by a short-time DFT of 256-sample Hann frames at 50 percent overlap grouped into 15 one-third-octave bands from 150 hertz, and compared in 384 millisecond segments; the chain then splits into the STOI clipped envelope correlation and the ESTOI row- and column-normalised spectral correlation, and the example reads STOI equal to 0.727, with the lowest band keeping 0.27 of the correlation and the bands above 1.9 kilohertz reaching 0.90" width="92%"></picture>

## 2. STOI: envelope correlation with clipping

For every band and segment STOI normalises the degraded envelope to the clean
one, clips it at a lower signal-to-distortion bound (`beta = -15` dB) so that a
fully degraded unit cannot drag the score below its floor, and takes the
sample correlation of the two envelopes (Taal et al. 2011, Eqs. 3-6). The index
is the average of those intermediate correlations over all bands and segments
(Eq. 6). Because the normalisation divides out a per-segment gain, STOI is
**invariant to the overall playback level** of the degraded signal.

```python
import numpy as np
from phonometry import stoi

fs = 10000
rng = np.random.default_rng(1)
clean = rng.standard_normal(3 * fs)
# Higher SNR gives a higher STOI: the relation is monotonic.
for snr_db in (-10, 0, 10, 20):
    g = 10.0 ** (-snr_db / 20.0)
    print(snr_db, round(stoi(clean, clean + g * rng.standard_normal(clean.size), fs).value, 3))
```

The `STOIResult` carries the per-band mean correlation (`band_scores`) and the
per-segment scores (`segment_scores`) that average to `value`, and its
`.plot()` draws the per-band intermediate correlation. That per-band view is
worth looking at before quoting the index: it shows *where* the degradation
bites, which the single number cannot.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/stoi_band_scores_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/stoi_band_scores.svg" alt="Mean intermediate correlation per one-third-octave band from 150 Hz to 3810 Hz for speech-like material in a flat masker at 0 dB signal-to-noise ratio: the lowest band keeps only about 0.27 of the envelope correlation and the value climbs steadily to about 0.9 in the consonant bands above 1.9 kHz, averaging to a STOI of 0.727" width="88%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from phonometry import stoi

# Speech-like material: band-limited noise with a 3.5 Hz syllabic envelope,
# in a flat masker at 0 dB SNR.
fs = 10000
rng = np.random.default_rng(11)
t = np.arange(4 * fs) / fs
b, a = butter(2, [200 / (fs / 2), 4000 / (fs / 2)], btype="band")
carrier = lfilter(b, a, rng.standard_normal(t.size))
clean = carrier * (0.15 + 0.85 * np.abs(np.sin(2 * np.pi * 3.5 * t)) ** 2)
masker = rng.standard_normal(clean.size)
gain = np.sqrt(np.mean(clean ** 2)) / np.sqrt(np.mean(masker ** 2))
res = stoi(clean, clean + gain * masker, fs)
print(round(res.value, 3))    # 0.727

# One line: the per-band intermediate correlation behind the index.
res.plot()
plt.show()

# By hand, mirroring what STOIResult.plot() draws:
pos = np.arange(res.band_scores.size)
fig, ax = plt.subplots()
ax.bar(pos, res.band_scores)
ax.set_xticks(pos)
ax.set_xticklabels([f"{f:.0f}" for f in res.band_frequencies], rotation=45, ha="right")
ax.set_xlabel("One-third-octave band [Hz]")
ax.set_ylabel("Mean intermediate correlation")
ax.set_title(f"STOI = {res.value:.3f}")
plt.show()
```

</details>

The flat masker costs the low bands most, because the speech spectrum falls
there while the masker does not, and the correlation recovers in the
consonant-bearing bands above 1 kHz. A band profile that is flat, or that
collapses only in one band, points at something other than broadband noise:
a notch filter, a resonance, or a processing artefact of the enhancement
under test.

## 3. ESTOI: spectral correlation for modulated maskers

ESTOI (`extended=True`) replaces the band-independent correlation with a
joint spectro-temporal one. Within each 384 ms segment it normalises the
spectrogram rows (the band envelopes) and then the columns (the per-frame
spectra) to zero mean and unit norm, and averages the correlation of the
normalised columns (Jensen and Taal 2016, Eqs. 4-8). Making the columns compete
means a masker that leaves quiet gaps, where the clean speech is briefly
audible, is credited for the speech glimpsed there, which STOI's per-band
average largely misses.

```python
from phonometry import stoi

estoi = stoi(clean, degraded, fs, extended=True)
print(round(estoi.value, 3))
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/stoi_intelligibility_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/stoi_intelligibility.svg" alt="Two panels of intelligibility index versus SNR from -15 to 20 dB. Left (STOI): the stationary-masker and modulated-masker curves nearly overlap, so STOI barely separates the two maskers. Right (ESTOI): the modulated-masker curve sits clearly above the stationary one across the whole SNR range, so ESTOI credits the speech glimpsed in the masker's quiet gaps" width="90%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import numpy as np
import matplotlib.pyplot as plt
from phonometry import stoi

fs = 10000
rng = np.random.default_rng(20)
t = np.arange(3 * fs) / fs
# A speech-like clean signal: amplitude-modulated formant-ish tones.
clean = np.zeros_like(t)
for f0 in (200.0, 400.0, 700.0, 1100.0, 1800.0, 2600.0):
    depth = 0.5 * (1.0 + np.sin(2 * np.pi * rng.uniform(2.0, 6.0) * t + rng.uniform(0.0, 2 * np.pi)))
    clean += depth * np.sin(2 * np.pi * f0 * t + rng.uniform(0.0, 2 * np.pi))
p_clean = np.sqrt(np.mean(clean ** 2))

base = rng.standard_normal(clean.size)
gate = 0.5 * (1.0 + np.sign(np.sin(2 * np.pi * 5.0 * t)))   # 5 Hz on/off gate
modulated = base * (0.05 + 0.95 * gate)
snrs = np.arange(-15.0, 20.1, 5.0)

def curve(masker, extended):
    p_m = np.sqrt(np.mean(masker ** 2))
    out = []
    for snr in snrs:
        g = p_clean / (p_m * 10.0 ** (snr / 20.0))
        out.append(stoi(clean, clean + g * masker, fs, extended=extended).value)
    return out

fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, extended, title in ((a, False, "STOI"), (b, True, "ESTOI")):
    ax.plot(snrs, curve(base, extended), "o-", label="Stationary masker")
    ax.plot(snrs, curve(modulated, extended), "s--", label="Modulated masker")
    ax.set_title(title); ax.set_xlabel("SNR [dB]"); ax.set_ylim(0, 1); ax.legend()
a.set_ylabel("Intelligibility index")
plt.show()
```

</details>

## 4. Which measure, and when

| | STOI | ESTOI |
| :--- | :--- | :--- |
| Intermediate quantity | Per-band envelope correlation, clipped | Row- and column-normalised spectral correlation |
| Level invariance | Yes (per-segment normalisation) | Yes (per-row and per-column normalisation) |
| Stationary maskers | Well validated | Well validated |
| Modulated maskers, competing talkers | Underrates the glimpsing benefit | Tracks it |
| Cost | Lower | A little higher |

For additive stationary noise the two are interchangeable and STOI is the
lighter default; when the interference fluctuates in time, or when comparing
processors that reshape the speech in time and frequency, prefer ESTOI.

Both answer a narrower question than the two standardised speech metrics of
the library, and the choice between the three families is really a choice of
what you can measure. STOI and ESTOI need a *clean reference* and the
degraded signal, so they belong to processing work: noise reduction, source
separation, codecs, hearing-aid algorithms. The
[STI](speech-transmission.md) (IEC 60268-16) needs no clean reference, only
the channel, so it is what rates a room or a public-address system. The
[SII](speech-intelligibility.md) (ANSI S3.5-1997) needs neither signal, only
spectra, so it is what predicts audibility for a listener with a given
hearing threshold. When more than one is available, they answer different
questions rather than confirming each other: a processor can raise STOI while
the room keeps the STI poor.

## See also

- [Speech Transmission Index](speech-transmission.md): rates a transmission
  channel from its impulse response or a STIPA recording.
- [Speech Intelligibility Index](speech-intelligibility.md): predicts
  intelligibility from speech, noise and hearing-threshold spectra.
- [Filter banks](filter-banks.md): the one-third-octave bands the front end
  groups the DFT into.
