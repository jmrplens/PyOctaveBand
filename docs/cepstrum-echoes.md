← [Documentation index](README.md)

# Cepstrum, echoes and the envelope spectrum (Havelock / Bendat & Piersol)

The [spectral estimators](spectral-analysis.md) describe *what
frequencies* a signal contains; this page covers what hides in the *shape* of
that spectrum. The **cepstrum** - the inverse Fourier transform of the log
spectrum - lives in `phonometry.signal` and turns two hard spectral
problems into easy peak-picking: periodic spectral ripple (an echo, a harmonic
family) collapses onto a single spike at the **quefrency** of its period, and
the smooth spectral envelope separates from the fine structure by plain
windowing - **liftering** - in the quefrency domain. The same machinery
extends the [Hilbert envelope](correlation-delay.md) with an
**envelope spectrum** in which amplitude modulations become discrete lines.

The whole trick is a chain of four steps: the logarithm turns a
multiplicative echo into additive spectral ripple, and the inverse FFT
collapses that ripple onto a single quefrency spike.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_cepstrum_echoes_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_cepstrum_echoes.svg" alt="Block diagram of the cepstrum chain: a signal made of a source wavelet plus an echo with reflection coefficient 0.5 at 8 milliseconds produces a spectrum rippled with a 125 hertz period, taking the log of the squared magnitude turns the multiplicative echo into an additive ripple, and the inverse FFT lands on the quefrency axis, drawn below with the source wavelet concentrated under 2 milliseconds, a spike of height 0.5 at exactly 8 milliseconds, a negative second rahmonic of minus 0.125 at twice the delay, and a dashed lifter cutoff at 4 milliseconds separating the lowpass envelope side from the highpass ripple side; captions give the rahmonic height series and the plus 3.5 and minus 6.0 decibel ripple bounds" width="92%"></picture>

## 1. The cepstrum and its three variants

Because the log turns the convolution $x = h \ast u$ into the sum
$\ln X = \ln H + \ln U$, components that overlap in the spectrum add - and
separate - in the cepstral domain (Havelock Ch. 27, Eqs. (22)-(23)). `cepstrum`
computes the three standard variants over the quefrency axis:

- `'power'` (default): the inverse DFT of $\ln|X|^2$ (Milner's Fig. 21). Real,
  even and phase-blind - the workhorse for echo and harmonic-family
  detection. This is Milner's *signed* cepstrum of the log-power spectrum;
  Bogert's original 1963 "power cepstrum" squares once more and is
  non-negative - the library follows Milner throughout, so negative
  rahmonics keep their sign;
- `'real'`: the inverse DFT of $\ln|X|$ - exactly half the power cepstrum,
  and the quantity whose causal folding is the minimum-phase reconstruction
  (below);
- `'complex'`: the inverse DFT of $\ln|X| + j\arg X$ with the phase unwrapped
  and its linear component removed (Havelock Ch. 87, Eq. (14)). It keeps the
  phase, so it is **invertible**: the entry point to homomorphic
  deconvolution.

```python
import numpy as np
from phonometry import cepstrum

fs = 48000.0
rng = np.random.default_rng(1)
x = rng.standard_normal(4096)

res = cepstrum(x, fs, kind="power")
print(res.quefrencies[:3], res.cepstrum.shape)   # quefrency axis, in s
res.plot()
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/cepstrum_variants_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/cepstrum_variants.svg" alt="The power, real and complex cepstra of a wavelet with one 8 millisecond echo overlaid against quefrency up to 20 milliseconds: all three carry a sharp positive spike at 8 milliseconds and a small negative one at 16 milliseconds, the source wavelet fills the region below about 2 milliseconds, and an inset zoom on the first rahmonic shows the power and complex spikes reaching 0.5 while the real cepstrum reaches exactly half of that" width="86%"></picture>

*The three variants of one echo-carrying record (a band-limited wavelet
plus a reflection at 8 ms, $a = 0.5$). All three carry the rahmonics at 8
and 16 ms with heights $a$ and $-a^2/2$ (Milner's signed convention); the inset
shows the real cepstrum at exactly half the power cepstrum, and the source
wavelet concentrates below 2 ms.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sp_signal
from phonometry import cepstrum

fs = 48000.0
# A band-limited source wavelet plus one reflection at 8 ms (a = 0.5)
b, a = sp_signal.butter(2, 0.3)
s = np.zeros(4096)
s[37:37 + 256] = sp_signal.lfilter(b, a, np.r_[1.0, np.zeros(255)])
x = s + 0.5 * np.roll(s, 384)

# One line per variant — each CepstrumResult draws itself:
cepstrum(x, fs, kind="power").plot()
plt.show()

# The three variants overlaid by hand on one quefrency axis:
fig, ax = plt.subplots()
for kind, style in (("power", "-"), ("real", "--"), ("complex", ":")):
    res = cepstrum(x, fs, kind=kind)
    q_ms = 1e3 * res.quefrencies
    mask = (q_ms > 0.5) & (q_ms <= 20.0)
    ax.plot(q_ms[mask], res.cepstrum[mask], style, label=f"{kind} cepstrum")
ax.set(xlabel="Quefrency [ms]", ylabel="Cepstrum")
ax.legend()
plt.show()
```

</details>

The result carries the full periodic quefrency axis (`0 .. (nfft-1)/fs`);
quefrencies above `nfft/(2·fs)` are the mirrored negative quefrencies, where
the even power and real cepstra repeat and the complex cepstrum keeps its
anticausal (non-minimum-phase) content. Zero-padding via `nfft` reduces
cepstral time-aliasing when the log spectrum has sharp features, exactly like
the `oversample` padding of
[`minimum_phase`](swept-sine-distortion.md).

## 2. Echo detection: the rahmonic spike train

A single reflection $x(t) = s(t) + a\,s(t-t_0)$ multiplies the spectrum by
$1 + a e^{-j 2\pi f t_0}$ - a ripple of period $1/t_0$ across the whole band.
Its logarithm expands, for $|a| < 1$, into the exactly summable series

$$
\ln\!\left(1 + a e^{-j\theta}\right)
= \sum_{n \ge 1} (-1)^{n+1} \frac{a^n}{n}\, e^{-jn\theta},
$$

so the cepstrum carries a spike train at the **rahmonics** $n t_0$ with
amplitudes $a, -a^2/2, a^3/3, \dots$ (their sum is $\ln(1+a)$), regardless of
the spectrum of $s$ itself, which concentrates at low quefrencies. On the signed
cepstrum of the log-power spectrum (`kind='power'`, Milner's convention) the
first spike's height is the reflection coefficient $a$ - of either
sign - plus whatever the source cepstrum contributes at that quefrency
(negligible for broadband sources, whose cepstrum concentrates at low
quefrencies); on an ideal impulse-plus-echo the identity is a closed form
the tests and the conformance suite pin to 1e-10. `echo_detection` automates the reading: it picks the largest
`|cepstrum|` peak in the searched band (so an inverting reflection, $a < 0$,
is found at its true delay rather than missed), refines the delay by
quadratic interpolation through the peak and its neighbours, and reports the
signed peak value as the reflection coefficient. When the true delay falls
between samples the rahmonic splits across quefrency bins: the interpolated
delay still lands on it, but the reported coefficient underestimates $|a|$
(down to about 65 % of it midway between samples).

```python
import numpy as np
from phonometry import echo_detection

fs = 48000.0
rng = np.random.default_rng(2)
s = rng.standard_normal(12000)               # broadband source
x = s + 0.5 * np.roll(s, 384)                # echo: 8 ms, a = 0.5

res = echo_detection(x, fs, min_quefrency=0.002)
print(res.delay, res.reflection_coefficient)  # 0.008 s, ~0.5
res.plot()
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/cepstrum_echo_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/cepstrum_echo.svg" alt="Power cepstrum of an impulse response with one reflection against quefrency in milliseconds: a sharp positive spike at exactly 8 milliseconds marked as the detected echo, a smaller negative second rahmonic at 16 milliseconds, the low-quefrency source envelope outside the shaded searched band, and a dashed line at the true delay" width="82%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sp_signal
from phonometry import echo_detection, noise_signal

fs = 48000.0
n = 12000
impulse = np.zeros(n)
impulse[0] = 1.0
b, a = sp_signal.butter(2, [0.004, 0.9], btype="bandpass")
direct = sp_signal.lfilter(b, a, impulse)              # broadband click
ir = direct + 0.5 * np.roll(direct, int(0.008 * fs))   # echo at 8 ms
ir += noise_signal(fs, n / fs, color="white", rms=1e-4, seed=13)

res = echo_detection(ir, fs, min_quefrency=0.002)

fig, ax = plt.subplots(figsize=(10, 6))
half = res.nfft // 2 + 1
ax.plot(1e3 * res.quefrencies[:half], res.cepstrum[:half], lw=1.1)
ax.axvline(8.0, ls="--", color="k", label="True echo delay")
ax.plot([1e3 * res.delay], [res.reflection_coefficient], "v", ms=10,
        label="Detected peak (height = reflection a)")
ax.set_xlim(0.0, 30.0)
ax.set_xlabel("Quefrency [ms]")
ax.set_ylabel("Cepstrum")
ax.legend()
plt.show()
```

</details>

The searched band starts above the low-quefrency region occupied by the
source's own spectral envelope (`min_quefrency`, default 16 samples) and ends
at the unambiguous half of the axis (`max_quefrency`). The seismic
reverberation spike trains of Havelock Ch. 87 are the same signature at
geophysical scale. Note the negative second rahmonic at $2 t_0$ in the figure:
the $-a^2/2$ term of the series, a useful confirmation that a peak really is
an echo and not an unrelated spectral periodicity.

## 3. Liftering: envelope versus fine structure

Filtering in the quefrency domain is called **liftering** (Havelock Ch. 27,
Sec. 4.3). A *lowpass* lifter keeps the quefrencies below the cutoff and
returns the smooth log-spectral envelope with the ripple removed; a
*highpass* lifter keeps the complement - the ripple alone. The two modes are
exactly complementary in dB, because the split is linear in the log domain:

```python
import numpy as np
from phonometry import lifter

fs = 48000.0
rng = np.random.default_rng(3)
s = rng.standard_normal(12000)
x = s + 0.5 * np.roll(s, 384)                # the same 8 ms echo

low = lifter(x, fs, cutoff=0.004, mode="lowpass")    # envelope of ln|X|
high = lifter(x, fs, cutoff=0.004, mode="highpass")  # the echo ripple
print(np.allclose(low.liftered_db + high.liftered_db, low.spectrum_db))
low.plot()
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/lifter_split_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/lifter_split.svg" alt="Two panels over 500 to 2000 hertz for a wavelet carrying a pure 8 millisecond echo: the log spectrum oscillates with a 125 hertz ripple around the flat lowpass-liftered envelope, and below it the highpass-liftered ripple alone swings exactly between the dashed closed-form bounds at plus 3.5 and minus 6 decibels" width="86%"></picture>

*The 4 ms lifter split on a pure-echo record: the lowpass side returns the
smooth spectral envelope, the highpass side isolates the echo's 125 Hz
ripple, swinging exactly between the closed forms
$20\log_{10}(1 \pm a) = +3.5$ and $-6.0$ dB.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sp_signal
from phonometry import lifter

fs = 48000.0
# A band-limited wavelet carrying a pure 8 ms echo (a = 0.5), so the
# highpass ripple has the exact closed-form bounds.
b, a = sp_signal.butter(2, 0.3)
s = np.zeros(4096)
s[37:37 + 256] = sp_signal.lfilter(b, a, np.r_[1.0, np.zeros(255)])
x = s + 0.5 * np.roll(s, 384)

low = lifter(x, fs, cutoff=0.004, mode="lowpass")
high = lifter(x, fs, cutoff=0.004, mode="highpass")

# One line — the cepstrum with the cutoff and the two log spectra:
low.plot()
plt.show()

# The envelope/ripple split by hand, zoomed on 500-2000 Hz:
band = (low.frequencies >= 500) & (low.frequencies <= 2000)
fig, axes = plt.subplots(2, 1, sharex=True)
axes[0].semilogx(low.frequencies[band], low.spectrum_db[band], "0.6",
                 lw=0.7, label="Log spectrum")
axes[0].semilogx(low.frequencies[band], low.liftered_db[band], lw=2,
                 label="Lowpass lifter: envelope")
axes[1].semilogx(high.frequencies[band], high.liftered_db[band], "r",
                 label="Highpass lifter: ripple")
for bound in (20 * np.log10(1.5), 20 * np.log10(0.5)):
    axes[1].axhline(bound, color="g", linestyle="--")
axes[1].set_xlabel("Frequency [Hz]")
for ax in axes:
    ax.set_ylabel("Magnitude [dB]")
    ax.legend()
plt.show()
```

</details>

For the pure-echo signal the highpass ripple swings between the closed forms
$20\log_{10}(1+a)$ and $20\log_{10}(1-a)$ dB, another oracle the tests pin. In
speech analysis the identical operation separates the vocal-tract envelope
(formants) from the excitation harmonics; here it is the general tool for
"smooth versus periodic" splits of any measured magnitude response.

## 4. The complex cepstrum and the minimum-phase connection

The complex cepstrum keeps the unwrapped phase, so the transform is a round
trip: `CepstrumResult.invert()` restores the record to machine precision,
including the linear-phase (pure delay) component that the forward transform
removes and stores in `linear_phase_samples`:

```python
import numpy as np
from scipy import signal as sp_signal
from phonometry import cepstrum

fs = 48000.0
x = np.zeros(2048)
b, a = sp_signal.butter(2, 0.3)
x[37:293] = sp_signal.lfilter(b, a, np.r_[1.0, np.zeros(255)])

res = cepstrum(x, fs, kind="complex")
print(res.linear_phase_samples)              # negative: a bulk delay removed
print(np.max(np.abs(res.invert() - x)))      # ~1e-14
res.plot()   # the complex cepstrum against quefrency (as in the figure above)
```

Between the log and the inverse transform anything can be edited - that is
homomorphic deconvolution (Havelock Ch. 87, Sec. 3.3): zero the rahmonics to
remove an echo, keep only the low quefrencies to extract a source wavelet.
A minimum-phase signal has a causal complex cepstrum, which is why folding
the real cepstrum onto positive quefrencies reconstructs the minimum phase
from $|H|$ alone: [`minimum_phase`](swept-sine-distortion.md)
and `phase_decomposition` run on that same folding core (Bendat & Piersol
Sec. 13.1.4; Tohyama in Havelock Ch. 75 edits reverberation by manipulating
exactly these causal/anticausal parts).

## 5. The envelope spectrum: modulations as lines

Where the cepstrum finds periodicities *of the spectrum*, the **envelope
spectrum** finds periodicities *of the amplitude*. Bendat & Piersol
Section 13.3 (Fig. 13.11) formalizes the structure: an envelope detector, a
DC remover, and a spectral view of what remains. `envelope_spectrum` runs the
[Hilbert envelope](correlation-delay.md) (`kind="magnitude"`,
the practical default) or the book's square-law detector (`kind="squared"`)
through exactly that chain, scaled by the taper's coherent gain so a
sinusoidal modulation whose frequency falls on an analysis bin reads out as
a line at its exact amplitude (off-bin lines read low by the taper's
scalloping loss, up to about 1.4 dB for the default Hann). The optional
`band=(low, high)` argument reproduces the figure's band-pass front end -
the classical bearing-envelope chain: isolate the structural-resonance band
excited by the defect impacts with a zero-phase band-pass, then envelope
it - so an out-of-band interferer is strongly attenuated before the
detector (the roll-off of a fourth-order Butterworth applied forward and
backward; the rejection is finite, and the zero-phase pass leaves small
transients at the record edges).

For an AM tone $A_0 (1 + m \cos(2\pi f_m t)) \cos(2\pi f_c t)$ with $f_m$
on an analysis bin the closed forms are:

| `kind` | mean level | line at $f_m$ | line at $2f_m$ |
|---|---|---|---|
| `'magnitude'` | $A_0$ | $A_0 m$ | - |
| `'squared'` | $A_0^2 (1 + m^2/2)$ | $2 A_0^2 m$ | $A_0^2 m^2/2$ |

```python
import numpy as np
from phonometry import envelope_spectrum

fs = 8192.0
t = np.arange(int(4 * fs)) / fs
x = (1.0 + 0.4 * np.cos(2 * np.pi * 25.0 * t)) * np.cos(2 * np.pi * 1000.0 * t)

res = envelope_spectrum(x, fs)
k = int(round(25.0 * res.nfft / fs))
print(res.mean_level, res.amplitude[k])      # ~1.0 and ~0.4
res.plot()
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/envelope_spectrum_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/envelope_spectrum.svg" alt="Envelope spectrum of an amplitude-modulated one kilohertz tone in noise against frequency up to one hundred hertz: a single sharp line at the twenty-five hertz modulation frequency reaching exactly the dotted reference at amplitude zero point four, with a flat noise floor elsewhere" width="82%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import envelope_spectrum, noise_signal

fs = 8192.0
seconds = 4.0
t = np.arange(int(seconds * fs)) / fs
x = (1.0 + 0.4 * np.cos(2 * np.pi * 25.0 * t)) * np.cos(2 * np.pi * 1000.0 * t)
x += noise_signal(fs, seconds, color="white", rms=0.03, seed=8)

res = envelope_spectrum(x, fs)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(res.frequencies, res.amplitude, lw=1.4, label="Envelope spectrum")
ax.axvline(25.0, ls="--", color="k", label="Modulation frequency")
ax.axhline(0.4, ls=":", color="r", label=r"Exact line amplitude $A_0 m$")
ax.set_xlim(0.0, 100.0)
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Modulation amplitude")
ax.legend()
plt.show()
```

</details>

Bearing and gear defects, mains hum and wind-turbine amplitude modulation all
appear this way: lines at the modulation frequency and its harmonics, cleanly
separated from the carrier's own spectrum. The envelope mean removed by the
DC step is kept in `mean_level` (the carrier amplitude for the magnitude
detector), and `remove_dc=False` skips the remover when the absolute DC line
matters.

## Relation to the other estimators

The cepstrum starts from the same one-record FFT conventions as the
[calibrated spectral estimators](spectral-analysis.md), and its
folding core is literally the one inside
[`minimum_phase`](swept-sine-distortion.md) - the refactor is
pinned bit-exact in the tests. The envelope spectrum is the frequency-domain
view of the same analytic signal the
[Hilbert envelope](correlation-delay.md) returns in time, and a
natural pre-analysis before the dedicated
[wind-turbine amplitude-modulation](wind-turbine-noise.md)
metrics: the envelope spectrum tells you *whether and at what rate* a signal
is modulated, the domain metrics quantify it normatively.

## References

- Havelock, D., Kuwano, S., & Vorländer, M. (Eds.) (2008). *Handbook of
  Signal Processing in Acoustics*. Springer. ISBN 978-0-387-77698-9.
  [doi:10.1007/978-0-387-30441-0](https://doi.org/10.1007/978-0-387-30441-0).
  Chapter 27 (Milner: the cepstral transform as the inverse DFT of the log
  power spectrum, quefrency, lowpass/highpass liftering), Chapter 87
  (Neelamani: the complex cepstrum, homomorphic deconvolution, periodic
  spike trains from reverberation) and Chapter 75 (Tohyama: minimum-phase /
  all-pass manipulation in the cepstral domain).
- Bendat, J. S., & Piersol, A. G. (2010). *Random Data: Analysis and
  Measurement Procedures* (4th ed.). Wiley. ISBN 978-0-470-24877-5.
  [doi:10.1002/9781118032428](https://doi.org/10.1002/9781118032428).
  Section 13.1.4 (the Hilbert relation between log magnitude and phase
  behind the minimum-phase folding) and Section 13.3 with Figure 13.11
  (envelope detection followed by DC removal, the structure of the
  envelope spectrum).
