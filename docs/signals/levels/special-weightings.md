← [Documentation index](../../README.md)

# Special Weightings (G, B, D, AU)

Beyond A, C and Z, the weighting family keeps four special-purpose curves,
each the right tool for a narrower job: the **G** curve of **ISO 7196:1995**
rates infrasound below 20 Hz the way A-weighting rates audible noise, the
historical **B** (ANSI S1.4-1983) reproduces measurements taken under older
national codes, the withdrawn aircraft-noise **D** (IEC 537) serves
comparisons with legacy data, and **AU** (IEC 61012), still in force like the
G curve, keeps ultrasonic components out of an audible-exposure reading.

All four share the machinery of the IEC 61672-1 curves (0 dB at 1 kHz where
applicable, multichannel and stateful block processing), and B, D and AU also
take the `high_accuracy` oversampling, and G takes it too: **the default G
design oversamples toward 48 kHz**, so its 0.25 Hz to 315 Hz range stays
within about 0.05 dB whatever the input rate, while stateful block processing
runs the plain design at the input rate, about a decibel low at 315 Hz at
fs = 2000 and exactly on the 0 dB reference at 10 Hz. That matters exactly where G is used — infrasound is
often recorded at 1 kHz or 2 kHz, where 315 Hz sits close to Nyquist and the
bilinear warping grows quadratically; at audio rates the correction is
negligible. As with A/C/Z, `high_accuracy` cannot be combined with stateful
processing. The A, C and Z curves themselves, where they come from, the
`high_accuracy` design and the class verification against IEC 61672-1
Table 3 are the subject of [Frequency Weighting](weighting.md).

## 1. Infrasound: G-weighting (ISO 7196)

The **G frequency weighting** (ISO 7196:1995) rates infrasound the way A-weighting
rates audible noise. It is defined by a pole-zero configuration with 0 dB gain at
10 Hz, rises at 12 dB/octave from 1 Hz to 20 Hz (matching the steep growth of
perception in that band) and falls off at 24 dB/octave outside it. Use it for
sources with significant energy below 20 Hz (wind turbines, HVAC, blasting):

```python
import numpy as np
from phonometry import filters

# recording: a calibrated microphone capture (Pa) — recorded through your measurement chain. Synthesized here so the guide runs standalone.
fs = 48000
recording = 0.2 * np.sin(2 * np.pi * 1000 * np.arange(fs) / fs)

g_weighted = filters.weighting_filter(recording, fs, curve='G')
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/g_weighting_response_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/g_weighting_response.svg" alt="G-weighting frequency response from 0.1 Hz to 1 kHz with the ISO 7196 Table 2 nominal values overlaid" width="80%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import filters

# Measure the G response: weight a centered unit impulse and take its
# spectrum. A long buffer gives the resolution the infrasound range
# needs (20 s -> 0.05 Hz).
fs = 4000
impulse = np.zeros(20 * fs)
impulse[impulse.size // 2] = 1.0
freqs = np.fft.rfftfreq(impulse.size, 1 / fs)
spectrum = np.fft.rfft(filters.weighting_filter(impulse, fs, curve="G"))

fig, ax = plt.subplots(figsize=(9, 5))
ax.semilogx(freqs[1:],
            20 * np.log10(np.abs(spectrum[1:]) + np.finfo(float).eps))
ax.plot(10, 0, "o", color="tab:red", label="0 dB at 10 Hz")
ax.set(xlim=(0.1, 1000), ylim=(-90, 15),
       xlabel="Frequency [Hz]", ylabel="G-weighting response [dB]")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
plt.show()
```

</details>

The implementation follows the ISO 7196 Table 1 pole/zero values exactly and is
verified in CI against every Table 2 nominal response value (0.25 Hz to 315 Hz).
`WeightingFilter(fs, "G")` supports the same multichannel and stateful block
processing as A/C — but stateful mode cannot carry the internal oversampling
across blocks, so if you stream G-weighting at a sample rate below 48 kHz,
verify the response before trusting the level. Levels measured with the G
curve are reported as $L_{p\mathrm{G}}$ (or $L_\mathrm{Geq}$ for the equivalent level over
time).

## 2. Historical and special-purpose curves: B, D and AU

Three more curves complete the family. All three work in the audible range,
so they share one chart, drawn against the A curve because that is what each
of them is defined or described against: B as the curve between A and C, D as
the one with a hump A does not have, AU as A itself with a low-pass added.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/special_weighting_responses_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/special_weighting_responses.svg" alt="B, D and AU weighting curves measured at 96 kHz against a wide grey A-weighting reference, with the +11.5 dB D hump at 3.15 kHz and the AU cutoff 13 dB below A at 16 kHz annotated" width="80%"></picture>

*The three curves against the A reference (wide grey), measured at 96 kHz so
the axis reaches the 40 kHz where IEC 61012 still specifies the U low-pass.
B (green, dashed) discards less bass than A; D (purple) carries the +11.5 dB
hump at 3.15 kHz where jet turbomachinery whine annoys most; AU (orange) runs
inside the A reference up to 10 kHz and then falls away with U, reaching 13 dB
below A at 16 kHz. The infrasound G curve keeps its own chart in section 1,
and A, C and Z are in [Frequency Weighting](weighting.md).*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import filters

# Measure each curve's response: weight a centered unit impulse and take its
# spectrum. 96 kHz, not 48 kHz: it reaches the 40 kHz top row of the
# IEC 61012 U-weighting table, which is the whole point of AU.
fs = 96000
impulse = np.zeros(fs)
impulse[fs // 2] = 1.0
freqs = np.fft.rfftfreq(fs, 1 / fs)

fig, ax = plt.subplots(figsize=(9, 5))
# A goes first and wide, as the reference the other three are read against.
for curve, width in (("A", 4.0), ("B", 1.8), ("D", 1.8), ("AU", 1.8)):
    spectrum = np.fft.rfft(filters.weighting_filter(impulse, fs, curve=curve))
    ax.semilogx(freqs[1:], 20 * np.log10(np.abs(spectrum[1:]) + np.finfo(float).eps),
                label=curve, linewidth=width)
ax.set(xlim=(10, 40000), ylim=(-90, 18),
       xlabel="Frequency [Hz]", ylabel="Response [dB]")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
plt.show()
```

</details>

### B (ANSI S1.4-1983, historical)

The middle curve of the original A/B/C level-switching scheme, drawn from
the ~70-phon equal-loudness contour. Analytically it is the C weighting with
one more zero at the origin and one extra real pole at
$f_5 = 158.49\ \text{Hz}$ (Appendix C of ANSI S1.4-1983), so it discards
less bass than A and more than C. It was dropped when IEC 61672-1 replaced
the older sound-level-meter standards; use it only to reproduce historical
data and measurements taken under older national codes (some legacy
automotive test procedures reported dB(B)). The implementation follows the
ANSI S1.4-1983 Appendix C constants and is pinned in CI against the Table IV
response values, within the strictest Table V mask (Type 0).

### D (IEC 537, withdrawn: aircraft noise)

The D weighting approximated the *perceived noisiness* contours used by the
perceived-noise-level (PNL) rating, so a plain sound level meter could
estimate aircraft noise: the +11.5 dB hump around 3.15 kHz is where jet
turbomachinery whine annoys most (it is deliberately *not* an equal-loudness
feature). NASA's aircraft-noise handbook gives the classic rule of thumb
$L_\mathrm{PN} \approx L_\mathrm{D} + 7\ \text{dB}$. IEC 537 was withdrawn and current
certification practice reports EPNL from one-third-octave analysis or plain
A-weighted levels, so `D` is provided for historical data and comparisons.
With the standard unavailable, the implementation uses the widely published
IEC 537 rational transfer function and is cross-checked against two
independent implementations (SQAT's zeros/poles and librosa's closed form,
which agree within 0.002 dB) and pinned in CI against the IEC 537 table
republished in NASA CR-3406.

```python
import numpy as np
from phonometry import filters, signals

# A 3.15 kHz whine sits right on the D-weighting hump: D rates it
# 10 dB *louder* than A does.
fs = 96000
t = np.arange(fs) / fs
whine = 0.1 * np.sin(2 * np.pi * 3150 * t)

ld = signals.leq(filters.weighting_filter(whine, fs, curve="D"))
la = signals.leq(filters.weighting_filter(whine, fs, curve="A"))
print(f"LD = {ld:.1f} dB   LA = {la:.1f} dB")
# LD = 82.5 dB   LA = 72.2 dB
```

### AU (IEC 61012, current: audible sound in the presence of ultrasound)

The only one of the three still in force. `AU` is the A weighting cascaded
with the **U** low-pass filter of IEC 61012:1990 (six poles, Table 2): flat
relative to A up to 10 kHz, then a steep cutoff (-13 dB at 16 kHz, -61.8 dB
at 40 kHz for U alone). Use it when strong ultrasonic components (ultrasonic
cleaners and welders, rodent repellers, some public-space deterrents) would
otherwise leak into an A-weighted reading through the meter's imperfect
high-frequency roll-off and overstate the *audible* exposure:

```python
import numpy as np
from phonometry import filters, signals

# 1 kHz tone (audible) buried under a strong 25 kHz ultrasonic component.
fs = 96000
t = np.arange(fs) / fs
audible = 0.1 * np.sin(2 * np.pi * 1000 * t)
x = audible + 1.0 * np.sin(2 * np.pi * 25000 * t)

la = signals.leq(filters.weighting_filter(x, fs, curve="A"))
lau = signals.leq(filters.weighting_filter(x, fs, curve="AU"))
la_ref = signals.leq(filters.weighting_filter(audible, fs, curve="A"))
print(f"LA = {la:.1f} dB   LAU = {lau:.1f} dB   audible alone = {la_ref:.1f} dB")
# LA = 78.6 dB   LAU = 71.0 dB   audible alone = 71.0 dB
# The ultrasound inflates LA by 7.6 dB; AU recovers the audible level.
```

Ultrasound only reaches a digital filter when the sample rate captures it,
so measure at 96 kHz or more (at 48 kHz there is nothing above 24 kHz to
reject); the AU design internally oversamples toward 288 kHz to keep the
steep U roll-off accurate. Levels are reported as $L_\mathrm{AU}$. The
implementation follows the Table 2 pole locations exactly (they reproduce
every Table 1 nominal value within 0.05 dB) and is verified in CI against
the Table 1 tolerances up to 40 kHz.

## 3. Verifying B and AU against their tolerance tables

The `verify_weighting_class` verifier, described in section 6 of
[Frequency Weighting](weighting.md), also covers the curves
of this guide that have published tolerance tables. For `B` it uses
ANSI S1.4-1983 (Table IV design goals, Table V
limits) and the "class" verdicts read as the standard's instrument **Types**
1 and 2. For `AU` it uses IEC 61012:1990 Table 1 (nominal A + nominal U with
the separate-unit tolerances, zero at the 1 kHz reference); IEC 61012
publishes a single tolerance set, so both margin slots agree and the verdict
is simply complies (1) or not (`None`) — note that checking the rows above
20 kHz needs `fs` ≥ 96 kHz (below that they are dropped and the verdict is
`range_limited`). `G` and `D` are rejected: ISO 7196 defines one ±1 dB
tolerance with no class structure, and the withdrawn IEC 537 left no
tolerance table behind (both curves are pinned numerically in the CI
conformance report instead).

## Quick answers

### Which weighting should I use for infrasound below 20 Hz?

Use the G frequency weighting of ISO 7196:1995, which rates infrasound the way A-weighting rates audible noise. It has 0 dB gain at 10 Hz, rises at 12 dB/octave from 1 Hz to 20 Hz and falls off at 24 dB/octave outside that band. Apply it to sources such as wind turbines, HVAC and blasting, and report levels as $L_{p\mathrm{G}}$ (or $L_\mathrm{Geq}$ for the equivalent level over time).

## See also

- [Frequency Weighting](weighting.md): the A, C and Z curves, the
  `high_accuracy` design and the IEC 61672-1 Table 3 class verification
  these curves build on.
- API reference: [`filters.weighting`](https://jmrplens.github.io/phonometry/reference/api/filters/weighting/) and [`filters.compliance`](https://jmrplens.github.io/phonometry/reference/api/filters/compliance/).

## References

- International Organization for Standardization. (1995). *Acoustics —
  Frequency-weighting characteristic for infrasound measurements*
  (ISO 7196:1995). [iso.org catalogue](https://www.iso.org/standard/13813.html).
  The G-weighting pole/zero definition (Table 1), verified against every
  Table 2 nominal response value (0.25 Hz to 315 Hz).
- American National Standards Institute. (1983). *Specification for Sound
  Level Meters* (ANSI S1.4-1983). The historical B weighting: Appendix C
  analytic definition (Formula C2), Table IV design goals and Table V
  tolerance limits checked by `verify_weighting_class` in section 3.
- International Electrotechnical Commission. (1990). *Filters for the
  measurement of audible sound in the presence of ultrasound*
  (IEC 61012:1990). [IEC webstore](https://webstore.iec.ch/en/publication/4296).
  The AU weighting: U-weighting pole locations (Table 2), nominal responses
  and tolerances (Table 1) and the combined AU definition of subclause 2.2.
- International Electrotechnical Commission. (1976). *Frequency weighting
  for the measurement of aircraft noise (D-weighting)* (IEC 537:1976,
  withdrawn). Implemented from its published rational transfer function and
  cross-checked against independent implementations (section 2).
- Bennett, R. L., & Pearsons, K. S. (1981). *Handbook of Aircraft Noise
  Metrics* (NASA CR-3406). NASA.
  [ntrs.nasa.gov](https://ntrs.nasa.gov/citations/19810013341).
  Republishes the IEC 537 D-weighting table (Table SLD-I) used to pin the
  D response in CI.

## Standards

ISO 7196:1995, *Acoustics — Frequency-weighting characteristic for
infrasound measurements*: the G-weighting pole/zero definition (Table 1),
verified against every Table 2 nominal response value (0.25 Hz to 315 Hz).
ANSI S1.4-1983, *Specification for Sound Level Meters*: the historical B
weighting (Appendix C, Tables IV and V). IEC 61012:1990, *Filters for the
measurement of audible sound in the presence of ultrasound*: the AU
weighting (Tables 1 and 2, subclause 2.2). IEC 537:1976 (withdrawn),
*Frequency weighting for the measurement of aircraft noise*: the D
weighting, from its published rational transfer function.
