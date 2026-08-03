← [Documentation index](../../README.md)

# Advanced Loudness (ISO 532-2/-3, ECMA-418-2)

The Zwicker method of ISO 532-1 is the reference route to loudness in sones
and lives in [Loudness](loudness.md), together with the ISO 226
equal-loudness contours. This page covers the newer model families
phonometry ships beside it: the **Moore-Glasberg** loudness of
ISO 532-2/532-3 and the **Sottek Hearing Model** loudness of
ECMA-418-2:2025, whose shared auditory front-end also powers the tonality
and roughness metrics of [Sound Quality Metrics](sound-quality.md).

The page opens with the choice: which model fits which measurement, and why
the sone values of the four methods agree at the 1 kHz / 40 dB anchor yet
are not interchangeable digit for digit. Each model then gets its own
section with a worked example, its figure and its parameter table.

## Choosing a loudness model

| Model | Standard | Stationary / time-varying | Output | When to use |
| :--- | :--- | :--- | :--- | :--- |
| [Zwicker](loudness.md) | ISO 532-1:2017 | both | sone | Reference method; one-third-octave input; fast and widely cited |
| Moore-Glasberg | ISO 532-2:2017 | stationary | sone | roex excitation pattern; better for tones and explicit binaural summation |
| Moore-Glasberg-Schlittenlacher | ISO 532-3:2023 | time-varying | sone (STL/LTL) | Time-varying loudness with short-/long-term traces and the peak $N_\text{max}$ |
| Sottek (Hearing Model) | ECMA-418-2:2025 | time-varying | sone_HMS | Shares one auditory front-end with the ECMA tonality and roughness metrics |

All four methods are anchored so a **1 kHz tone at 40 dB SPL is ≈ 1 sone**; the values
are not interchangeable digit-for-digit because the models differ in their
auditory filters and their loudness summation.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/loudness_models_comparison_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/loudness_models_comparison.svg" alt="Loudness of a 1 kHz tone as a function of level for the Zwicker, Moore-Glasberg and Sottek models, all passing close to 1 sone at 40 dB SPL" width="80%"></picture>

*The three models pass through approximately 1 sone at the 40 dB anchor
(the Sottek front-end documents 0.984 sone_HMS there) and diverge with level:
Zwicker doubles the sone value every +10 phon, while the Sottek model grows
more slowly (about 1.65× per 10 dB), an intrinsic difference between the
auditory summations, not a calibration error.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import psychoacoustics

# 1 kHz tone, 20..80 dB SPL: all three models pass close to 1 sone at 40 dB
fs = 48000
t = np.arange(fs) / fs
levels = np.arange(20.0, 81.0, 10.0)
zw, mg, ec = [], [], []
for spl in levels:
    x = np.sqrt(2) * 2e-5 * 10 ** (spl / 20) * np.sin(2 * np.pi * 1000 * t)
    zw.append(psychoacoustics.loudness_zwicker(x, fs, stationary=True).loudness)
    mg.append(
        psychoacoustics.loudness_moore_glasberg_from_spectrum([(1000.0, float(spl))]).loudness
    )
    ec.append(psychoacoustics.loudness_ecma(x, fs).loudness)

fig, ax = plt.subplots()
ax.plot(levels, zw, "o-", label="Zwicker (ISO 532-1)")
ax.plot(levels, mg, "s--", label="Moore-Glasberg (ISO 532-2)")
ax.plot(levels, ec, "^-.", label="Sottek (ECMA-418-2)")
ax.plot(40.0, 1.0, "o", color="k", markerfacecolor="none", markersize=10)   # the shared anchor
ax.set(xlabel="Sound pressure level [dB SPL]", ylabel="Total loudness N [sone]")
ax.legend()
plt.show()
```

</details>

## The ERB_N scale and the Cam axis

Every model on this page except Zwicker's is written on the
**ERB<sub>N</sub> number** axis, so it is worth having the scale itself in
hand. The cochlea behaves as a
bank of overlapping band-pass **auditory filters**; the width of the one centred
on a given frequency is summarised by its *equivalent rectangular bandwidth*,
the bandwidth of the rectangular filter that would pass the same power with the
same peak response. Fitting notched-noise data for young listeners at moderate
levels, Glasberg and Moore (1990) make it a straight line in frequency (Moore,
*An Introduction to the Psychology of Hearing* 6e, p. 76):

$$
\begin{aligned}
\mathrm{ERB}_N &= 24.7\,(4.37\,F + 1)\ \text{Hz},\\
\mathrm{ERB}_N\ \text{number} &= 21.4 \log_{10}(4.37\,F + 1)\ \text{Cam},
\end{aligned}
\qquad F \text{ in kHz}.
$$

The second line integrates $df/\mathrm{ERB}_N(f)$, so its unit is one
auditory-filter width: the **ERB<sub>N</sub> number**, whose unit is called
the **Cam** (after Cambridge).

```python
from phonometry import psychoacoustics

cam = psychoacoustics.cam_from_frequency(1000.0)
print(round(psychoacoustics.erb_bandwidth(1000.0), 1))       # 132.4 Hz
print(round(cam, 2))                                         # 15.59 Cam
print(round(psychoacoustics.frequency_from_cam(cam), 1))     # 1000.0 Hz
```

The library states the same fit with one more significant digit,
$\mathrm{ERB}_N = 24.673\,(0.004368 f + 1)$ and
$21.366 \log_{10}(0.004368 f + 1)$, the precision the ISO 532-2
implementation uses; the two forms agree to better than 0.2 % over the
audible range, and the extra digits are what make `frequency_from_cam` an
exact inverse. The functions are not a private helper of
the loudness model: `loudness_moore_glasberg` imports these very functions, so
the Cam grid of a `MooreGlasbergLoudness.erb_number` and a hand-built Cam axis
cannot drift apart.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/erb_bandwidth_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/erb_bandwidth.svg" alt="Auditory-filter bandwidth against centre frequency on log-log axes: the ERB_N line of Glasberg and Moore rising from about 30 Hz at 50 Hz to above 1 kHz at 16 kHz, the constant-percentage one-third-octave bandwidth crossing it near 500 Hz, the 1 kHz point annotated at 15.59 Cam, and a second top axis giving the Cam scale" width="85%"></picture>

*The ear's filter is not a constant fraction of frequency. Below about 500 Hz
it is markedly narrower than a one-third octave, which is why the "old"
critical-band function, flat at low frequency, fits the direct measurements so
poorly there; above that it is wider. The top axis counts the same curve in
Cams, so equal steps along it are equal numbers of auditory filters.*

### `erb_bandwidth()`, `cam_from_frequency()`, `frequency_from_cam()` parameters

| Parameter | Type | Units | Range / default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `frequency` | float or array | Hz | ≥ 0 | `erb_bandwidth`, `cam_from_frequency` |
| `cam` | float or array | Cam | ≥ 0 | `frequency_from_cam` |

Each returns a float for a scalar input and an array otherwise. The three
constants are exported as `ERB_C1`, `ERB_C2` and `CAM_C`.

## Moore-Glasberg loudness (ISO 532-2)

Where Zwicker uses fixed critical bands on the Bark scale, Moore-Glasberg builds
an **excitation pattern** with level-dependent rounded-exponential (roex)
auditory filters on the ERB-number ("Cam") scale, then applies a compressive
excitation → specific-loudness transform with $C = 0.0617\ \text{sone/Cam}$
(ISO 532-2:2017, Formula 7) and a binaural-inhibition stage. It reproduces the
tone and broadband cases of Annex B to a percent or two and, unlike ISO 532-1,
models binaural summation explicitly.

```python
import numpy as np
from phonometry import psychoacoustics

# The definitional anchor: one 1 kHz sinusoidal component at 40 dB SPL,
# free field, binaural -> 1 sone / 40 phon by construction of the sone.
res = psychoacoustics.loudness_moore_glasberg_from_spectrum([(1000.0, 40.0)], field="free")
print(f"N = {res.loudness:.3f} sone  ({res.loudness_level:.1f} phon)")   # 1.000 sone (40.0 phon)

# From a calibrated recording: the narrowband (FFT) line spectrum is formed
# (power-preserving normalization) and fed to the exact sinusoidal-component
# method (ISO 532-2 clauses 5.2/5.4).
fs = 48000
x = np.sqrt(2) * 2e-5 * 10 ** (40 / 20) * np.sin(2 * np.pi * 1000 * np.arange(fs) / fs)
res = psychoacoustics.loudness_moore_glasberg(x, fs, field="free", presentation="binaural")

res.plot()   # specific loudness N'(i) over the ERB-number (Cam) scale
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/moore_glasberg_specific_loudness_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/moore_glasberg_specific_loudness.svg" alt="ISO 532-2 specific loudness over the ERB-number scale for a 1 kHz tone at 40 dB SPL: a single rounded peak near 15 Cam whose area is the total loudness of 1 sone" width="80%"></picture>

*The ISO 532-2 pattern of the definitional 1 sone anchor. The peak is not a
spectral line but the excitation the roex filters produce around the tone, so
its width is the auditory filter, not the analysis resolution.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import psychoacoustics

# From a calibrated recording: the narrowband (FFT) line spectrum is formed
# (power-preserving normalization) and fed to the exact sinusoidal-component
# method (ISO 532-2 clauses 5.2/5.4).
fs = 48000
x = np.sqrt(2) * 2e-5 * 10 ** (40 / 20) * np.sin(2 * np.pi * 1000 * np.arange(fs) / fs)
res = psychoacoustics.loudness_moore_glasberg(x, fs, field="free", presentation="binaural")

# One line — the specific-loudness pattern N'(i) straight from the result:
res.plot()
plt.show()

# Or draw it by hand from the ERB-number grid the result already carries:
fig, ax = plt.subplots()
ax.fill_between(res.erb_number, res.specific, alpha=0.3)
ax.plot(res.erb_number, res.specific)
ax.set_xlabel("ERB number [Cam]")
ax.set_ylabel("Specific loudness N' [sone/Cam]")
plt.show()
```

</details>

### `loudness_moore_glasberg()` parameters

| Parameter | Type | Units | Range / default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `x` | 1D array | Pa | non-empty | Calibrated pressure signal (signal wrapper) |
| `components` | list of `(f, L)` | Hz, dB SPL | — | `_from_spectrum`: discrete sinusoidal components |
| `band_levels` | 29-vector | dB SPL | 25 Hz .. 16 kHz | `_from_third_octave` input (IEC 61260-1 bands) |
| `fs` | int | Hz | > 0 | Signal wrapper only |
| `field` | str | — | `'free'` (default) / `'diffuse'` / `'eardrum'` | Outer-ear transfer |
| `presentation` | str | — | `'binaural'` (default) / `'diotic'` / `'monaural'` | Binaural summation |

Returns a `MooreGlasbergLoudness`: `loudness` ($N$, sone), `loudness_level`
(phon), `specific` ($N'(i)$, 372 bins of 0.1 Cam), `erb_number`,
`centre_frequencies`, `field`, `presentation`.

## Time-varying loudness (ISO 532-3)

ISO 532-3 wraps the same excitation / specific-loudness model in a running
multi-resolution spectral analysis (six parallel FFTs, updated every 1 ms) and
two cascaded temporal integrators: the fast **short-term loudness** $S'(t)$
and the slower **long-term loudness** $S''(t)$. The peak long-term loudness
$N_\text{max}$ predicts the loudness of sounds up to about 5 s.

```python
import numpy as np
from phonometry import psychoacoustics

fs = 32000
t = np.arange(int(1.3 * fs)) / fs
x = np.sqrt(2) * 2e-5 * 10 ** (40 / 20) * np.sin(2 * np.pi * 1000 * t)

res = psychoacoustics.loudness_moore_glasberg_time(x, fs, field="free")
print(f"N_max = {res.n_max:.3f} sone  ({res.loudness_level_max:.0f} phon)")   # 1.000 sone (40 phon)
print(f"long-term loudness exceeded 5% of the time: {res.percentiles[5.0]:.3f} sone")   # 0.999 sone

res.plot()   # short-term S'(t) and long-term S''(t) loudness vs time
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/moore_glasberg_time_loudness_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/moore_glasberg_time_loudness.svg" alt="Short-term and long-term Moore-Glasberg loudness traces for a tone burst, showing the fast attack of the short-term loudness and the slower release of the long-term loudness" width="80%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import psychoacoustics

fs = 32000
t = np.arange(int(1.3 * fs)) / fs
x = np.sqrt(2) * 2e-5 * 10 ** (40 / 20) * np.sin(2 * np.pi * 1000 * t)
res = psychoacoustics.loudness_moore_glasberg_time(x, fs, field="free")

# The result carries both traces on a 1 ms time axis:
res.plot()
plt.show()

# Or plot them directly to see the fast STL vs the slow LTL:
fig, ax = plt.subplots()
ax.plot(res.time, res.short_term_loudness, label="Short-term S'(t)")
ax.plot(res.time, res.long_term_loudness, label="Long-term S''(t)")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Loudness [sone]")
ax.legend()
plt.show()
```

</details>

### `loudness_moore_glasberg_time()` parameters

| Parameter | Type | Units | Range / default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `signal` | 1D or `(n, 2)` array | Pa | non-empty | Mono = diotic; two columns = left/right ears |
| `fs` | int | Hz | > 0 | |
| `field` | str | — | `'free'` (default) / `'diffuse'` / `'eardrum'` | Outer-ear transfer |
| `presentation` | str | — | `'binaural'` (default) / `'diotic'` / `'monaural'` | Binaural summation |
| `percentiles` | sequence | percent | default `(1, 5, 10, 50, 90, 95)` | Exceeded long-term loudness levels |

Returns a `MooreGlasbergTimeVaryingLoudness`: `time` (1 ms grid),
`short_term_loudness` / `long_term_loudness` (sone), their `_level` in phon,
`n_max`, `loudness_level_max`, a `percentiles` dict, `field`, `presentation`.

## Sottek Hearing Model loudness (ECMA-418-2)

ECMA-418-2:2025 specifies a single auditory front-end (outer/middle-ear
filtering, a 53-band gammatone-like filter bank on the Bark_HMS scale
with $z = 0.5\ ..\ 26.5$, half-wave rectification, block RMS and a compressive
nonlinearity, Formula 23) that is **shared** by its loudness, tonality and
roughness metrics. The loudness $N$ is reported in **sone_HMS**, and the same
1 kHz/40 dB anchor calibrates the front-end (our clean-room value 0.984,
with the full Clause 6.2.3 band averaging; the residual's origin is
documented in the module docstring).

```python
import numpy as np
from phonometry import psychoacoustics

fs = 48000
t = np.arange(int(1.2 * fs)) / fs
x = np.sqrt(2) * 2e-5 * 10 ** (40 / 20) * np.sin(2 * np.pi * 1000 * t)

res = psychoacoustics.loudness_ecma(x, fs, field="free")
print(f"N = {res.loudness:.3f} sone_HMS")   # 0.984 sone_HMS
print(res.specific_loudness.shape)          # (53,) average specific loudness N'(z)

res.plot()   # average specific loudness N'(z) + time-dependent N(l) at 187.5 Hz
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/sottek_specific_loudness_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/sottek_specific_loudness.svg" alt="Sottek Hearing Model average specific loudness N'(z) over the 53 Bark_HMS bands for a 1 kHz tone, peaking at the tone's critical band" width="80%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import psychoacoustics

fs = 48000
t = np.arange(int(1.2 * fs)) / fs
x = np.sqrt(2) * 2e-5 * 10 ** (40 / 20) * np.sin(2 * np.pi * 1000 * t)
res = psychoacoustics.loudness_ecma(x, fs, field="free")

# The result carries the average specific loudness over the 53 Bark_HMS bands:
res.plot()
plt.show()

# Or draw N'(z) by hand against the critical-band-rate scale:
fig, ax = plt.subplots()
ax.fill_between(res.bark, res.specific_loudness, alpha=0.3)
ax.plot(res.bark, res.specific_loudness)
ax.set_xlabel("Critical-band rate z [Bark_HMS]")
ax.set_ylabel("Specific loudness N' [sone_HMS/Bark_HMS]")
plt.show()
```

</details>

### `loudness_ecma()` parameters

| Parameter | Type | Units | Range / default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `signal_in` | 1D array | Pa | non-empty | Calibrated pressure signal |
| `fs` | float | Hz | > 0 | Resampled to 48 kHz internally if needed (Clause 5.1.1) |
| `field` | str | — | `'free'` (default) / `'diffuse'` | Outer/middle-ear filter (Clause 5.1.3) |

Returns an `EcmaLoudness`: `loudness` ($N$, sone_HMS), `specific_loudness`
($N'(z)$, 53 bands), `bark`, `centre_frequencies`, `time`, `loudness_vs_time`
($N(l)$ at 187.5 Hz), `field`.

## Quick answers

### Which loudness model should I choose: Zwicker, Moore-Glasberg or Sottek?

The Zwicker method (ISO 532-1:2017) is the reference: stationary and time-varying, one-third-octave input, fast and widely cited. Moore-Glasberg (ISO 532-2:2017) is stationary, builds roex excitation patterns and models binaural summation explicitly; ISO 532-3:2023 adds time-varying short- and long-term loudness with the peak $N_\text{max}$. The Sottek model (ECMA-418-2:2025) reports sone_HMS and shares its auditory front-end with the ECMA tonality and roughness metrics.

## See also

- [Loudness](loudness.md): the Zwicker reference method (ISO 532-1), its
  accredited fiche and the ISO 226 equal-loudness contours.
- [Sound Quality Metrics](sound-quality.md): the tonality, roughness and
  fluctuation strength built on the same ECMA-418-2 front-end.
- [Theory](../../reference/theory/perception.md): the equations behind the loudness models.
- API reference: [`psychoacoustics.loudness.moore_glasberg`](https://jmrplens.github.io/phonometry/reference/api/psychoacoustics/moore-glasberg/), [`psychoacoustics.loudness.moore_glasberg_time`](https://jmrplens.github.io/phonometry/reference/api/psychoacoustics/moore-glasberg-time/) and [`psychoacoustics.loudness.ecma`](https://jmrplens.github.io/phonometry/reference/api/psychoacoustics/ecma/).

## References

- Moore, B. C. J. (2013). *An introduction to the psychology of hearing*
  (6th ed.). Brill.
  [doi:10.1163/9789004252424](https://doi.org/10.1163/9789004252424).
  Pages 76–77: the ERB_N auditory-filter bandwidth and the Cam (ERB_N number)
  scale of Glasberg and Moore (1990).
- International Organization for Standardization. (2017). *Acoustics —
  Methods for calculating loudness — Part 2: Moore-Glasberg method*
  (ISO 532-2:2017).
  [iso.org catalogue](https://www.iso.org/standard/63078.html).
  Stationary loudness from roex excitation patterns on the ERB-number scale,
  with explicit binaural summation.
- International Organization for Standardization. (2023). *Acoustics —
  Methods for calculating loudness — Part 3: Moore-Glasberg-Schlittenlacher
  method* (ISO 532-3:2023).
  [iso.org catalogue](https://www.iso.org/standard/69856.html).
  Time-varying short-term and long-term loudness and the peak N_max.
- Ecma International. (2025). *Psychoacoustic metrics for ITT equipment —
  Part 2 (methods for describing human perception based on the Sottek
  Hearing Model)* (ECMA-418-2, 4th ed.).
  [ecma-international.org](https://ecma-international.org/publications-and-standards/standards/ecma-418/).
  The Sottek Hearing Model loudness (sone_HMS).

## Standards

ISO 532-2:2017, *Acoustics — Methods for calculating loudness — Part 2:
Moore-Glasberg method*: stationary loudness from roex excitation patterns on
the ERB-number scale, with explicit binaural summation. ISO 532-3:2023,
*... Part 3: Moore-Glasberg-Schlittenlacher method*: time-varying short-term
and long-term loudness and the peak N_max. ECMA-418-2:2025, *Psychoacoustic
metrics for ITT equipment — Part 2 (methods for describing human perception
based on the Sottek Hearing Model)*: the Sottek Hearing Model loudness
(sone_HMS).
