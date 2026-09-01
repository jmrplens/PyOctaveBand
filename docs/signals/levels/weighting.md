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
| `x` | 1D or 2D array, or `Signal` | any | non-empty | 2D is `[channels, samples]` |
| `fs` | int | Hz | > 0; taken from `x` when `x` is a `Signal` | |
| `curve` | str | — | `'A'` (default), `'B'`, `'C'`, `'D'`, `'G'`, `'AU'`, `'Z'`, `'468'` | `'G'` per ISO 7196 (infrasound), `'B'`/`'D'` historical and `'AU'` per IEC 61012 are covered in [Special Weightings](special-weightings.md); `'Z'` is a bypass; `'468'` is the ITU-R BS.468-4 programme-level curve, which needs the fitted design and refuses `high_accuracy=False` |
| `high_accuracy` | bool | — | default `True` | Fit the analog prototype at $f_\mathrm{s}$ instead of transforming it blind; keeps A/C in class 1 at every sample rate. Details in §5 |
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

The curves of §1 are analog: poles and zeros in the $s$ plane. Turning them
into a digital filter with the bilinear transform is exact in magnitude and
wrong in frequency — it puts the prototype's response at
$2 f_\mathrm{s} \tan(\pi f / f_\mathrm{s})$ instead of at $2 \pi f$ — and the
error grows quadratically toward Nyquist. At $f_\mathrm{s} = 48$ kHz that plain
design reads 15.7 dB below the A design goal at the 19 952.6 Hz row, and at
$f_\mathrm{s} = 32$ kHz it reads 61.4 dB below it at 15 848.9 Hz.

By default (`high_accuracy=True`) phonometry does not transform the prototype
blind: it **fits** an analog prototype of the same structure whose response
*at the warped frequencies* is the printed prototype's response at the true
ones, and transforms that. What runs is one cascade of second-order sections
at the input rate, with nothing around it. Measured against the printed
prototype over each standard's own band:

| Curve | 32 kHz | 44.1 kHz | 48 kHz | 96 kHz |
| :--- | ---: | ---: | ---: | ---: |
| A | 0.008 dB | 0.003 dB | 0.0003 dB | 0.00001 dB |
| C | 0.002 dB | 0.002 dB | 0.0004 dB | 0.00001 dB |
| AU | 0.003 dB | 0.005 dB | 0.004 dB | 0.001 dB |
| 468 | 0.041 dB | 0.052 dB | 0.060 dB | 0.00002 dB |

A and C therefore verify to **class 1 at every sample rate from 8 kHz up**, and
at every Table 3 row their deviation stays inside the 0.05 dB the table itself
is rounded to.

The band the fit controls is the curve's own standardised range, clipped at
the top to 99.5 % of the Nyquist frequency — enough to contain every frequency
these standards state a requirement at, the closest approach being Table 3's
15 848.9 Hz row at 0.9906 of Nyquist when $f_\mathrm{s} = 32$ kHz. Above that
last half percent the response is not claimed: no digital filter can track an
analog curve past Nyquist, and the magnitude of a real-coefficient filter has
zero slope there. That is where the 468 row of the table above comes from —
its skirt is still falling at about $-30$ dB/octave where the band ends — and
it is 3 % of that curve's $\pm 2$ dB tolerance, not an accuracy shortfall.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/weighting_accuracy_hf_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/weighting_accuracy_hf.svg" alt="A-weighting high-frequency accuracy at 48 kHz: analytic curve versus plain bilinear versus the design fitted at the sample rate, with error subplot" width="80%"></picture>

*The plain bilinear design (red) crosses the class 1 tolerance near 12.5 kHz;
the fitted design (blue) sits on the analytic curve.*

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
                             (True, "Fitted at fs (default)")):
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

- `high_accuracy=False` gives the plain bilinear design: the closed form a
  reader can check against the standard term by term, at the cost above. It
  verifies to class 1 for $f_\mathrm{s} \ge 44\,100$ Hz, degrades to class 2 at
  32 000 and 22 050 Hz, and meets no class at 16 000 Hz.
- The `'468'` curve refuses `high_accuracy=False`: its skirt puts the plain
  design 23 dB out at 16 kHz, and ITU-R BS.468-4 prints one tolerance mask and
  no lower grade to fall back to.
- **Stateful (block) processing** carries no penalty. Both designs are plain
  second-order sections at the input rate, so `stateful` and `high_accuracy`
  are independent, stateful defaults to the fitted design like everything
  else, and stitched blocks reproduce a single call exactly.
- The fit costs about 260 ms and is cached per curve and sample rate, so it
  is paid once. Even including it, weighting one minute of 44.1 kHz audio is
  faster than it used to be: about 280 ms against 377 ms for A, and 285 ms
  against 775 ms for the 468 curve. Once the design is cached the filtering
  alone is about 18 ms, and it holds 21 MB of intermediates instead of 169 MB.

```python
import numpy as np
from phonometry import filters

# recording: a calibrated microphone capture (Pa) — recorded through your measurement chain. Synthesized here so the guide runs standalone.
fs = 48000
recording = 0.2 * np.sin(2 * np.pi * 1000 * np.arange(fs) / fs)

# The closed-form bilinear design, explicitly
y = filters.weighting_filter(recording, fs, curve="A", high_accuracy=False)

# Stateful block processing (same fitted design, state carried between blocks)
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

**The laboratory grade lives in the superseded edition.** IEC 61672-1:2013
publishes classes 1 and 2 only; **Type 0**, the tightest of the four
instrument types of **IEC 651:1979**, survives in that standard's Table V.
Pass `edition="1979"` to grade against it, exactly as `verify_filter_class`
reaches the IEC 61260:1995 class 0. Class *N* is then the standard's
instrument Type *N*, and every band and the sweep carry one margin per type,
`margin_class0_db` through `margin_class3_db`:

```python
from phonometry import filters

result = filters.verify_weighting_class(
    filters.WeightingFilter(48000, "A"), edition="1979"
)
print(result["overall_class"])                             # 0
print(min(b["margin_class0_db"] for b in result["bands"])) # 0.650...
```

It is a different mask, not a rename, and it sees errors class 1 cannot:
Type 0 holds +2/-3 dB at both 16 kHz and 20 kHz, where class 1 opens to
+2.5/-16 dB and +3/-inf. The undersampled `high_accuracy=False` design at
48 kHz droops 15.7 dB at the 20 kHz row and still earns class 1 for it; under
Table V that row is refused and the filter is graded Type 1. The 1979 edition
covers A, B and C, the weightings IEC 651 defines: its Table V footnote makes
one mask govern every weighting characteristic, so B is held to the same
limits there rather than borrowing the ANSI ones.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/weighting_class_mask_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/weighting_class_mask.svg" alt="A and C weighting deviations at 48 kHz threading within the IEC 61672-1 Table 3 class 1 acceptance corridor, with the wider class 2 limits dotted" width="80%"></picture>

*The fitted A and C designs (blue, purple) stay near zero deviation,
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

Not with a plain bilinear design: at $f_\mathrm{s} = 48$ kHz the A-curve error reaches −2.7 dB at 12.5 kHz, outside the IEC 61672-1 class 1 tolerance (+2.0/−2.5 dB). The default `high_accuracy=True` fits the prototype at the sample rate instead, which leaves 0.0003 dB anywhere in the table at 48 kHz and 0.008 dB at 32 kHz, so class 1 holds at every sample rate from 8 kHz up. §5 has the numbers and the band the fit controls.

## See also

- [Special Weightings (G, B, D, AU)](special-weightings.md):
  the infrasound G curve, the historical B and D, and AU for audible sound
  in the presence of ultrasound.
- API reference: [`filters.weighting`](https://jmrplens.github.io/phonometry/reference/api/filters/weighting/) and [`filters.compliance`](https://jmrplens.github.io/phonometry/reference/api/filters/compliance/).
- Theory: [Weighting Curves (IEC 61672-1)](../../reference/theory/signal-analysis.md#weighting-curves-iec-61672-1): the pole-zero definition of the A, C and Z curves and the bilinear warping the fitted design cancels.

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
at 1 kHz), the class 1 tolerances the `high_accuracy` design keeps at every
sample rate from 8 kHz up, and the Table 3 class 1/class 2 acceptance
limits checked by
`verify_weighting_class`; the special G, B, D and AU curves are covered in
[Special Weightings](special-weightings.md).
