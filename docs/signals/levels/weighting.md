← [Documentation index](../../README.md)

# Frequency Weighting (A, C, Z)

Frequency weighting curves simulate the human ear's sensitivity. This guide
covers **A**, **C** and **Z**, the curves specified by **IEC 61672-1:2013**:
where they come from, how to apply them, the `high_accuracy` design and the
Table 3 class verification. The rest of the family, the infrasound G curve,
the historical B and D and the AU curve, is
[Special Weightings](special-weightings.md).

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/weighting_responses_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/weighting_responses.svg" alt="A, C and Z frequency weighting curves of IEC 61672-1 with a zoom showing the positive region of the A curve (+1.27 dB at 2.5 kHz)" width="80%"></picture>

*The three curves of IEC 61672-1, measured through the library's own filters
at 48 kHz: A, which discards the bass; C, which keeps it; and Z, which
weights nothing at all. The inset magnifies the small positive region of A
around 2.5 kHz. The special B, D and AU curves have their own chart in
[Special Weightings](special-weightings.md), together with the infrasound
G curve.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import filters

# Measure each curve's response: weight a centered unit impulse and take
# its spectrum (1 s buffer -> 1 Hz frequency resolution).
fs = 48000
impulse = np.zeros(fs)
impulse[fs // 2] = 1.0
freqs = np.fft.rfftfreq(fs, 1 / fs)

fig, ax = plt.subplots(figsize=(9, 5))
for curve in ("A", "C", "Z"):
    spectrum = np.fft.rfft(filters.weighting_filter(impulse, fs, curve=curve))
    ax.semilogx(freqs[1:], 20 * np.log10(np.abs(spectrum[1:]) + np.finfo(float).eps),
                label=curve)
ax.set(xlim=(10, 22000), ylim=(-72, 15),
       xlabel="Frequency [Hz]", ylabel="Response [dB]")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
plt.show()
```

</details>

* **A-Weighting (`A`):** Standard for environmental noise (IEC 61672-1).
* **C-Weighting (`C`):** Used for peak sound pressure and high-level noise.
* **Z-Weighting (`Z`):** flat by specification, not by omission.
  IEC 61672-1 defines Z as a nominally flat response from 10 Hz to
  20 kHz with the same Table 3 tolerances as A and C, which is why
  `verify_weighting_class` can grade it at all. The library implements it
  as a bypass, so the *effective* bandwidth of a Z-weighted level is
  whatever your capture chain delivered: remove DC (`detrend`) and
  high-pass the wind noise yourself if the recording extends below 10 Hz.

The `curve` argument also accepts the four special weightings, charted and
documented in [Special Weightings](special-weightings.md): `'G'` for
infrasound (ISO 7196), the historical `'B'` (ANSI S1.4-1983) and `'D'`
(IEC 537), and `'AU'` for audible sound in the presence of ultrasound
(IEC 61012).

## 1. Where the curves come from

The A and C curves are inverted equal-loudness contours, frozen into filters:
**A** approximates the inverse of the historic 40-phon contour (quiet levels,
where the ear discards bass most aggressively) and **C** the flatter ~100-phon
one (loud levels). IEC 61672-1:2013 (Annex E) defines both analytically from
four corner frequencies:

$$
f_1 = 20.599\ \text{Hz}, \quad f_2 = 107.653\ \text{Hz}, \quad
f_3 = 737.862\ \text{Hz}, \quad f_4 = 12194.217\ \text{Hz}
$$

C is a band-pass with double poles at $f_1$ and $f_4$ (2 zeros at the origin);
A adds the $f_2$ and $f_3$ poles (4 zeros), which is why it keeps falling
through the low-mids. Both are normalized to exactly 0 dB at 1 kHz. Z applies
no shaping inside the specified band; its design goal is 0 dB everywhere from
10 Hz to 20 kHz. The full pole/zero derivation is in the
[Theory](../../reference/theory/signal-analysis.md) page.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_equal_loudness_weighting_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_equal_loudness_weighting.svg" alt="Equal-loudness contours per ISO 226 on the left, with the 40-phon contour highlighted; on the right the A-weighting curve overlaid on the inverted 40-phon contour, showing that A is the flipped contour frozen into a realizable filter" width="92%"></picture>

### A short history: A, B, C and Z

The chain runs from Fletcher and Munson's 1933 equal-loudness measurements
to the first American sound level meter standard (1936), which gave meters
switchable responses so the reading could approximate loudness at different
levels: **A** from the 40-phon contour for quiet sounds, **B** from the
~70-phon contour for moderate ones, and a flat response for loud ones (the
**C** curve proper, mirroring the flatter ~100-phon contour, arrived with
the 1944 revision). Switching curves by level died in practice (readings jumped at the
switch points, and field measurements became incomparable), but A survived
alone: decades of hearing-damage and community-annoyance data had been
collected with it, and it correlates with both about as well as far more
elaborate metrics. IEC 61672-1 (first edition 2002) finished the cleanup:
B was dropped, A and C were kept with tightened tolerances, and **Z** was
introduced to replace the vaguely specified "linear" of older meters, which
varied by manufacturer. The B curve (and the aircraft-noise D curve that met
the same fate) remains available for historical data; see
[Special Weightings](special-weightings.md).

### When C − A matters

Because A discards bass and C keeps it, the difference
$L_\mathrm{Ceq} - L_\mathrm{Aeq}$ is a one-number indicator of low-frequency content:

- **Below about 10 dB**: an ordinary broadband spectrum; the A-weighted
  level rates it fairly.
- **Around 15 to 20 dB or more**: the energy is concentrated at low
  frequencies (HVAC rumble, compressors, music bass through a wall). The
  A-weighted level then understates the problem; look at the octave
  spectrum, and below 20 Hz switch to the G curve.
- **Hearing-protector selection**: the HML method of ISO 4869-2 keys on
  exactly this C-minus-A difference to decide how much low-frequency
  attenuation a protector must provide (the simpler SNR method sidesteps it
  by working from the C-weighted level directly).

```python
import numpy as np
from phonometry import filters, signals

# A 50 Hz rumble under a light broadband hiss: quiet in A, loud in C.
fs = 48000
t = np.arange(10 * fs) / fs
rng = np.random.default_rng(1)
x = 0.2 * np.sin(2 * np.pi * 50 * t) + 0.01 * rng.standard_normal(t.size)

la = signals.leq(filters.weighting_filter(x, fs, curve="A"))
lc = signals.leq(filters.weighting_filter(x, fs, curve="C"))
print(f"LAeq = {la:.1f} dB   LCeq = {lc:.1f} dB   C - A = {lc - la:.1f} dB")
# LAeq = 52.4 dB   LCeq = 75.7 dB   C - A = 23.2 dB
# C - A above 20 dB: the A-weighted number alone would hide the rumble.
```

## 2. Basic usage

```python
import numpy as np
from phonometry import filters

# recording: a calibrated microphone capture (Pa) — recorded through your measurement chain. Synthesized here so the guide runs standalone.
fs = 48000
recording = 0.2 * np.sin(2 * np.pi * 1000 * np.arange(fs) / fs)

# Apply A-weighting to the raw recording
weighted_signal = filters.weighting_filter(recording, fs, curve='A')

# Apply C-weighting for peak analysis
c_weighted_signal = filters.weighting_filter(recording, fs, curve='C')
```

The special weightings take the same `curve` argument; each is documented,
with its own response chart, in [Special Weightings](special-weightings.md).

## 3. `weighting_filter()` / `WeightingFilter` parameters

| Parameter | Type | Units | Range / default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `x` | 1D or 2D array | any | non-empty | 2D is `[channels, samples]` |
| `fs` | int | Hz | > 0 | |
| `curve` | str | — | `'A'` (default), `'B'`, `'C'`, `'D'`, `'G'`, `'AU'`, `'Z'` | `'G'` per ISO 7196 (infrasound), `'B'`/`'D'` historical and `'AU'` per IEC 61012 are covered in [Special Weightings](special-weightings.md); `'Z'` is a bypass |
| `high_accuracy` | bool | — | default `True` (function); class default `None` resolves to `not stateful` | Internal oversampling keeps A/C in class 1 up to 16 kHz when $f_\mathrm{s} \ge 40$ kHz; details in §5 |
| `stateful` | bool (class only) | — | default `False` | Carries filter state across blocks (streaming) |
| `steady_ic` | bool (class only) | — | default `False` | Steady-state initial conditions (no onset transient) |

## 4. Reusable filter object

If you weight many signals with the same parameters, design the filter once:

```python
import numpy as np
from phonometry import filters

# recording: a calibrated microphone capture (Pa) — recorded through your measurement chain. Synthesized here so the guide runs standalone.
fs = 48000
recording = 0.2 * np.sin(2 * np.pi * 1000 * np.arange(fs) / fs)

wf = filters.WeightingFilter(fs, "A")
batch = [recording]                  # your batch of recordings
for recording in batch:
    weighted = wf.filter(recording)
```

## 5. High-frequency accuracy (`high_accuracy`)

A plain bilinear-transform design compresses the response near Nyquist: at
$f_\mathrm{s} = 48$ kHz the A-curve error at 12.5 kHz reaches −2.7 dB, outside the IEC
61672-1 **class 1** tolerance (+2.0/−2.5 dB).

By default (`high_accuracy=True`), phonometry designs and runs the weighting
filter at an internally oversampled rate (up to 8×, reaching ≥ 144 kHz at
common audio rates; a 96 kHz input runs ×2) and decimates back, keeping
the response within class 1 tolerances up to 16 kHz (error ≈ −0.5 dB at
12.5 kHz for $f_\mathrm{s} = 48$ kHz).

Oversampling repairs the bilinear warping, not the sample rate. The
interpolation and decimation stages it adds around the filter carry an
anti-alias filter whose transition band sits on the *input* Nyquist
frequency, so above roughly 90 % of that frequency the response rolls off
however high the internal design rate is. That only bites when a Table 3 row
lands there, which is what happens below 40 kHz: at $f_\mathrm{s} = 32$ kHz
the 15 848.9 Hz row falls 16.2 dB below the A design goal (class 1 allows
−16.0 dB there, class 2 sets no lower limit) and at $f_\mathrm{s} = 16$ kHz
the 7 943.3 Hz row falls 12.0 dB below it (class 1 allows −2.5 dB, class 2
−5.0 dB). `verify_weighting_class` measures that whole path, so for A it
reports class 2 at 32 kHz and no class at all at 16 kHz rather than a grade
the filter cannot deliver. The verdict is per curve, each graded against its
own design goal: the C response lands 15.3 dB low at 32 kHz, inside that
same −16.0 dB limit, so C keeps class 1 there, and 13.7 dB low at 16 kHz,
where it meets no class either. Sample at 44.1 kHz or above for a class 1
A-weighting across the full table.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/weighting_accuracy_hf_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/weighting_accuracy_hf.svg" alt="A-weighting high-frequency accuracy at 48 kHz: analytic curve versus plain bilinear versus oversampled design, with error subplot" width="80%"></picture>

*The plain bilinear design (red) crosses the class 1 tolerance near 12.5 kHz;
the oversampled design (blue) stays close to the analytic curve.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import filters

# Measured response of both designs at fs = 48 kHz: weight a centered
# unit impulse and take its spectrum...
fs = 48000
impulse = np.zeros(fs)
impulse[fs // 2] = 1.0
freqs = np.fft.rfftfreq(fs, 1 / fs)[1:]

# ...versus the analytic IEC 61672-1 A-curve built from the four corner
# frequencies of section 1, normalized to 0 dB at 1 kHz.
f1, f2, f3, f4 = 20.599, 107.653, 737.862, 12194.217
gain = (f4**2 * freqs**4) / ((freqs**2 + f1**2)
        * np.sqrt((freqs**2 + f2**2) * (freqs**2 + f3**2))
        * (freqs**2 + f4**2))
analytic = 20 * np.log10(gain / gain[np.argmin(np.abs(freqs - 1000))])

fig, ax = plt.subplots(figsize=(9, 5))
ax.semilogx(freqs, analytic, "k--", label="Analytic (IEC 61672-1)")
for high_accuracy, label in ((False, "Plain bilinear"),
                             (True, "Oversampled (default)")):
    weighted = filters.weighting_filter(impulse, fs, curve="A",
                                high_accuracy=high_accuracy)
    response = 20 * np.log10(np.abs(np.fft.rfft(weighted))
                             + np.finfo(float).eps)[1:]
    ax.semilogx(freqs, response, label=label)
ax.set(xlim=(1000, 20000), ylim=(-12, 3),
       xlabel="Frequency [Hz]", ylabel="A-weighting response [dB]")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
plt.show()
```

</details>

- `high_accuracy=False` restores the legacy plain-bilinear behavior.
- For `'G'` the flag works like the others': the default design is
  oversampled toward 48 kHz, which is what keeps infrasound rates accurate;
  `high_accuracy=False` runs the plain design at the input rate, costing about
  a decibel at 315 Hz at fs = 2000 and nothing at the 10 Hz reference (see
  [Special Weightings](special-weightings.md)).
- **Stateful (block) processing** always uses the legacy design: the internal
  FIR resampling is incompatible with block continuity. Passing
  `high_accuracy=True` together with `stateful=True` raises a `ValueError`.

```python
import numpy as np
from phonometry import filters

# recording: a calibrated microphone capture (Pa) — recorded through your measurement chain. Synthesized here so the guide runs standalone.
fs = 48000
recording = 0.2 * np.sin(2 * np.pi * 1000 * np.arange(fs) / fs)

# Explicit legacy behavior
y = filters.weighting_filter(recording, fs, curve="A", high_accuracy=False)

# Stateful block processing (legacy design, state carried between blocks)
wf = filters.WeightingFilter(fs, "A", stateful=True)
blocks = [recording]                 # your sequence of recording blocks
for block in blocks:
    weighted = wf.filter(block)
```

See [Block Processing](../filters/block-processing.md) for the streaming workflow and
[Theory](../../reference/theory/signal-analysis.md) for the analytic curve definitions.

## 6. Verifying against the tolerance tables (IEC 61672-1)

`verify_weighting_class` checks a weighting filter against the acceptance
limits of **IEC 61672-1:2013** (Table 3). It evaluates the filter's relative
response at the *exact* base-10 frequency behind each nominal label below
Nyquist (Table 3's design goals are computed at $f = 1000 \cdot 10^{n/10}$,
e.g. 15 848.9 Hz for "16 kHz"; IEC 61672-3 tests at the same frequencies),
subtracts the design-goal weighting, and reports the performance class per
frequency with its margin in dB. A dense logarithmic sweep additionally
enforces subclause 5.5.7 *between* the nominal frequencies (the deviation
from the analytic Annex E goal must stay within the larger of the two
adjacent limits, so a resonance or notch between nominals cannot pass), and
when Table 3 rows with finite lower limits fall beyond Nyquist the verdict is
flagged `range_limited` (it then attests the checked frequencies only, not
full 10 Hz-20 kHz conformance):

```python
from phonometry import filters

result = filters.verify_weighting_class(filters.WeightingFilter(48000, "A"))
print(result["overall_class"])          # 1
print(result["range_limited"])          # False
print(result["between_nominals"])       # {'worst_freq': ..., 'margin_class1_db': ...}
print(result["bands"][20])
# {'freq': 1000.0, 'class': 1, 'deviation_db': 0.0, 'margin_class1_db': 0.7, 'margin_class2_db': 1.0}
```

The Table 3 acceptance mask itself is public too: `weighting_class_limits(1)`
returns the 34 nominal frequencies with the lower/upper deviation limits (a
lower limit of `-inf` means only the upper limit applies). The limits qualify
the *deviation* from the design goal, so they are the same for A, C and Z.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/weighting_class_mask_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/weighting_class_mask.svg" alt="A and C weighting deviations at 48 kHz threading within the IEC 61672-1 Table 3 class 1 acceptance corridor, with the wider class 2 limits dotted" width="80%"></picture>

*The oversampled A and C designs (blue, purple) stay near zero deviation,
well inside the class 1 corridor (shaded); the wider class 2 limits are
dotted. The corridor widens at the band extremes where only a one-sided
limit applies.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import filters

freqs, lower1, upper1 = filters.weighting_class_limits(1)
_, lower2, upper2 = filters.weighting_class_limits(2)
lo1, lo2 = np.clip(lower1, -7, 7), np.clip(lower2, -7, 7)

fig, ax = plt.subplots(figsize=(10, 6.5))
ax.fill_between(freqs, lo1, upper1, step="mid", alpha=0.10,
                label="Class 1 acceptance region")
ax.plot(freqs, upper1, drawstyle="steps-mid", label="Class 1 upper/lower limit")
ax.plot(freqs, lo1, drawstyle="steps-mid", color="C1")
ax.plot(freqs, upper2, ":", drawstyle="steps-mid", label="Class 2 upper/lower limit")
ax.plot(freqs, lo2, ":", drawstyle="steps-mid", color="C2")

for curve, marker in (("A", "o"), ("C", "s")):
    bands = filters.verify_weighting_class(filters.WeightingFilter(48000, curve))["bands"]
    f = [b["freq"] for b in bands]
    dev = [b["deviation_db"] for b in bands]
    ax.plot(f, dev, marker=marker, label=f"{curve} weighting deviation (48 kHz)")

ax.set(xscale="log", xlim=(10, 20000), ylim=(-7, 7),
       xlabel="Frequency [Hz]", ylabel="Deviation from design goal [dB]")
ax.legend(fontsize=8, ncol=2)
plt.show()
```

</details>

## Quick answers

### When should I use C-weighting instead of A-weighting?

Use C-weighting for peak sound pressure and high-level noise, and use the difference $L_\mathrm{Ceq} - L_\mathrm{Aeq}$ as a low-frequency indicator: below about 10 dB the A-weighted level rates the spectrum fairly, while around 15 to 20 dB or more the energy is concentrated at low frequencies and the A-weighted level understates the problem. The HML method of ISO 4869-2 keys on exactly this C minus A difference for hearing-protector selection.

### Is A-weighting accurate near 16 kHz at a 48 kHz sample rate?

Not with a plain bilinear design: at $f_\mathrm{s} = 48$ kHz the A-curve error reaches −2.7 dB at 12.5 kHz, outside the IEC 61672-1 class 1 tolerance (+2.0/−2.5 dB). The default `high_accuracy=True` oversamples internally (up to 8×, reaching 144 kHz or more at common audio rates) and keeps the response within class 1 tolerances up to 16 kHz, with an error of about −0.5 dB at 12.5 kHz. That holds while 16 kHz stays clear of Nyquist, so from about $f_\mathrm{s} = 40$ kHz upwards; §5 covers what the resampler's anti-alias transition band costs below that.

## See also

- [Special Weightings (G, B, D, AU)](special-weightings.md):
  the infrasound G curve, the historical B and D, and AU for audible sound
  in the presence of ultrasound.
- API reference: [`filters.weighting`](https://jmrplens.github.io/phonometry/reference/api/filters/weighting/) and [`filters.compliance`](https://jmrplens.github.io/phonometry/reference/api/filters/compliance/).
- Theory: [Weighting Curves (IEC 61672-1)](../../reference/theory/signal-analysis.md#weighting-curves-iec-61672-1): the pole-zero definition of the A, C and Z curves and the bilinear design the oversampling repairs.

## References

- Fletcher, H., & Munson, W. A. (1933). Loudness, its definition, measurement
  and calculation. *The Journal of the Acoustical Society of America*, 5(2),
  82-108. [doi:10.1121/1.1915637](https://doi.org/10.1121/1.1915637).
  The original equal-loudness measurements whose 40-phon contour the A-curve
  inverts (section 1).
- International Organization for Standardization. (2023). *Acoustics —
  Normal equal-loudness-level contours* (ISO 226:2023).
  [iso.org catalogue](https://www.iso.org/standard/83117.html).
  The modern successors of the Fletcher-Munson curves, drawn in the diagram
  of section 1.
- International Electrotechnical Commission. (2013). *Electroacoustics —
  Sound level meters — Part 1: Specifications* (IEC 61672-1:2013).
  [IEC webstore](https://webstore.iec.ch/en/publication/5708).
  The normative A/C/Z definitions, the analytic Annex E curves and the
  Table 3 acceptance limits verified in section 6.

## Standards

IEC 61672-1:2013, *Electroacoustics — Sound level meters —
Part 1: Specifications*: the A, C and Z frequency-weighting curves (the
Annex E analytic definition from four corner frequencies, normalized to 0 dB
at 1 kHz), the class 1 tolerances the `high_accuracy` design keeps up to
16 kHz at 44.1 kHz and above, and the Table 3 class 1/class 2 acceptance
limits checked by
`verify_weighting_class`; the special G, B, D and AU curves are covered in
[Special Weightings](special-weightings.md).
