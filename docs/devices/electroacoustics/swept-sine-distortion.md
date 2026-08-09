← [Documentation index](../../README.md)

# Swept-sine distortion and phase utilities (Farina / Novak)

A single exponential sine sweep characterises the linear response and
every harmonic distortion order of a weakly nonlinear system at once.
After deconvolution, the distortion products of order $n$ pack
into separate impulse responses that *precede* the linear response by the
fixed advance (Farina 2000)

$$
\Delta t_n = T\,\frac{\ln n}{\ln (f_2/f_1)} = L \ln n ,
\qquad L = \frac{T}{\ln(f_2/f_1)} ,
$$

so windowing each arrival yields the **higher harmonic frequency responses**
$H_1(f), H_2(f), \dots, H_N(f)$ and, from them, the total harmonic distortion as
a function of the excitation frequency with one sweep instead of a
tone-by-tone stepping. This page covers that separation in
`phonometry.electroacoustics`, with the phase-coherent **synchronized
sweep** of Novak, Lotton & Simon (2015) as the default, and the companion
**phase utilities** in `phonometry.signals`: minimum phase from $|H|$,
group delay and excess phase.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/swept_sine_thd_dark.svg">
  <img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/swept_sine_thd.svg" alt="Total harmonic distortion and the second and third harmonic ratios of a cubic polynomial followed by a 3 kHz low-pass, against excitation frequency: each order sits exactly on its Chebyshev level at low frequency and rolls off where its own product crosses the filter corner" width="82%">
</picture>

## 1. One sweep, every harmonic

For an exponential sweep the instantaneous frequency rises as
$f(t) = f_1 e^{t/L}$, so the moment the excitation passes $f$, the n-th
harmonic distortion product appears at $n f$: exactly where the sweep
itself will be $L\ln(n)$ seconds later. Deconvolving the recording against
the sweep therefore time-compresses each order into its own impulse
response, $L\ln(n)$ *before* the linear one. `swept_sine_distortion`
windows each arrival (with the exact fractional-sample alignment), Fourier
transforms it into $H_n(f)$, and reads the distortion of order $n$ at
excitation frequency $f$ from $|H_n(n f)|$:

$$
\mathrm{THD}(f) =
\frac{\sqrt{\sum_{n\ge 2} |H_n(nf)|^2}}{|H_1(f)|} .
$$

The chain is exactly the sketch below: play the sweep, record, convolve with
the inverse filter, and the harmonics land ahead of the linear response by
their fixed advances.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_swept_sine_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_swept_sine.svg" alt="Swept-sine distortion measurement chain: an exponential sweep from 20 Hz to 6 kHz in T equal to 4 s drives the weakly nonlinear device under test, the recording is deconvolved with the time-reversed inverse filter with its plus 6 dB per octave tilt, and on the resulting time axis the linear impulse response h1 sits at t equal to zero while the second, third and fourth harmonic responses appear as smaller pre-arrivals in their own dashed windows, advanced by L times ln n, 0.49 s for h2 and 0.77 s for h3 with L equal to 0.70 s" width="92%"></picture>

```python
import numpy as np
from phonometry import swept_sine_distortion, synchronized_sweep_signal

fs, f1, f2, seconds = 48000, 20.0, 6000.0, 4.0
x = synchronized_sweep_signal(fs, f1, f2, seconds)   # play this...
# ... record the device response into `y` (include the decay tail) ...
res = swept_sine_distortion(y, fs, f1, f2, seconds, n_harmonics=3)

res.harmonic_responses    # complex H1..H3 on res.frequencies
res.thd, res.thd_frequencies
res.distortion_ratios     # |Hn(n f)| / |H1(f)| per order
res.plot()                # |Hn| magnitudes + THD(f)
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/swept_sine_thd_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/swept_sine_thd.svg" alt="Total harmonic distortion and the second and third harmonic ratios of a cubic polynomial followed by a 3 kHz low-pass, against excitation frequency: each order sits exactly on its Chebyshev level at low frequency and rolls off where its own product crosses the filter corner" width="82%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sp_signal
from phonometry import swept_sine_distortion, synchronized_sweep_signal

fs, f1, f2, seconds = 48000, 20.0, 6000.0, 4.0
a2, a3 = 0.12, 0.08
x = synchronized_sweep_signal(fs, f1, f2, seconds)
b, a = sp_signal.butter(2, 3000.0, fs=fs)             # 3 kHz post-filter
y = sp_signal.lfilter(b, a, x + a2 * x**2 + a3 * x**3)
res = swept_sine_distortion(y, fs, f1, f2, seconds, n_harmonics=3)

h1 = 1.0 + 3.0 * a3 / 4.0                              # Chebyshev gain
fig, ax = plt.subplots(figsize=(10, 6))
ax.loglog(res.thd_frequencies, 100.0 * res.thd, label="Total THD(f)")
ax.loglog(res.thd_frequencies, 100.0 * res.distortion_ratios[0],
          ls="--", label="2nd harmonic d2(f)")
ax.loglog(res.thd_frequencies, 100.0 * res.distortion_ratios[1],
          ls="--", label="3rd harmonic d3(f)")
ax.axhline(100.0 * (a2 / 2.0) / h1, ls=":",
           label="Chebyshev asymptote (a2/2)/H1")
ax.axhline(100.0 * (a3 / 4.0) / h1, ls=":",
           label="Chebyshev asymptote (a3/4)/H1")
ax.set_xlabel("Excitation frequency [Hz]")
ax.set_ylabel("Distortion re fundamental [%]")
ax.legend()
plt.show()
```

</details>

The oracle behind the implementation is the memoryless polynomial: driving
$y = x + a_2 x^2 + a_3 x^3$ with a unit sweep must return, by the Chebyshev
identities, $|H_1| = 1 + 3a_3/4$, $|H_2| = a_2/2$ (phase $-\pi/2$), $|H_3| = a_3/4$
(phase $\pi$) and $\mathrm{THD} = \sqrt{(a_2/2)^2 + (a_3/4)^2}/(1 + 3a_3/4)$. The test and
conformance suites pin all four, and the same THD measured tone by tone
with `phonometry.thd` agrees to 0.1 %.

## 2. The synchronized sweep (Novak et al. 2015)

Windowing separates the harmonic *magnitudes* with any exponential sweep,
but the *phases* of $H_2 \dots H_N$ are only meaningful if delaying the sweep by
$L\ln(n)$ is exactly equivalent to generating its n-th harmonic. That holds
only for

$$
x(t) = \sin\!\big[2\pi f_1 L\, e^{t/L}\big],
\qquad
L = \frac{1}{f_1}\,\mathrm{round}\!\Big(\frac{f_1\,\tilde T}{\ln(f_2/f_1)}\Big),
$$

the **synchronized swept-sine**: the rounding makes $f_1 L$ an integer, so
the sweep starts at zero phase and every harmonic copy lines up.
`synchronized_sweep_signal` generates it (the duration is quantized
slightly; when $f_2/f_1$ is an integer the sweep also ends at zero phase),
and `swept_sine_distortion(..., method="synchronized")`, the default,
deconvolves with the closed-form spectrum of the inverse filter,

$$
\tilde X(f) = 2\sqrt{f/L}\;
e^{-j 2\pi f L\,(1 - \ln(f/f_1)) + j\pi/4},
$$

rather than an FFT of the signal. Besides being exact, the analytic
deconvolution extends the usable band of each $H_n$ to $[n f_1, n f_2]$
(Novak et al., Fig. 6): the second harmonic of a 6 kHz sweep is measured
up to 12 kHz.

Two practical notes from the paper are built in: the recording mean is
subtracted by default (`remove_dc=True`; a DC offset otherwise leaks a
scaled inverse filter into the impulse response), and the non-integer part
of each arrival $L\ln(n) f_s$ is removed in the frequency domain, so the
harmonic phases carry no residual sub-sample skew.

## 3. Analysing classical ESS recordings (`method="farina"`)

Recordings made with the plain exponential sweep of
`phonometry.sweep_signal` (the ISO 18233 excitation used by
[`impulse_response`](../../buildings/rooms/room-acoustics.md)) are analysed with
`method="farina"`: the same windowing over the time-reversed,
amplitude-compensated inverse filter of Farina (2000). The harmonic
**magnitudes** and the THD are correct (the Chebyshev oracle passes
identically), but the sweep's `-1` phase term breaks the time-shift
equivalence, so the phases of $H_2 \dots H_N$ depend on the excitation and should
be ignored; the band of every $H_n$ is also capped at $f_2$ by the inverse
filter.

```python
from phonometry import sweep_signal, swept_sine_distortion

x = sweep_signal(fs, f1, f2, seconds)          # the ISO 18233 ESS
res = swept_sine_distortion(y, fs, f1, f2, seconds, method="farina")
res.plot()   # same |Hn| + THD(f) panels as the synchronized method (needs matplotlib)
```

The result is the same plottable `SweptSineDistortionResult` as the
synchronized method, so the `|Hn|` and `THD(f)` panels of the section-1
figure read identically; only the harmonic phases (and the top of each
order's band) differ between the two methods.

Sizing rules for both methods: the closest pair of arrivals is spaced
$L\ln(N/(N-1))$ seconds, so the per-order window (`ir_length`, default the
largest power of two that fits, capped at 8192 samples) must not exceed
it: lengthen the sweep or lower `n_harmonics` for reverberant systems whose
tails need longer windows. Keep `n_harmonics` $\cdot f_2$ below Nyquist: distortion
products above it fold back in any real recording. The analysis is
referenced to the excitation `amplitude`, so $H_1$ is the linear gain and
the THD is level-referenced exactly as driven.

## 4. Phase utilities: minimum phase, group delay, excess phase

For a causal, stable, minimum-phase system the log-magnitude and phase of
the frequency response are a Hilbert-transform pair (Bendat & Piersol,
Sec. 13.1.4): the phase is fully determined by `|H(f)|`. The
`phonometry.signals` utilities compute that reconstruction with the real
cepstrum and decompose any measured response into its invertible and
all-pass parts:

```python
import numpy as np
from phonometry import (
    excess_phase, group_delay, minimum_phase, phase_decomposition,
)

H = np.fft.rfft(ir)                    # one-sided response, DC..Nyquist
h_min = minimum_phase(np.abs(H))       # phase from the magnitude alone
tau_g = group_delay(H, fs)             # -(1/2pi) dphi/df, seconds
phi_x = excess_phase(H)                # unwrap(arg H) - phi_min

res = phase_decomposition(H, fs)       # everything on one axis
res.excess_group_delay                 # the all-pass part, in seconds
res.plot()                             # magnitude, phases, group delays
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/phase_decomposition_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/phase_decomposition.svg" alt="Three stacked panels of the minimum-phase / all-pass decomposition of a +6 dB peaking equalizer measured through a 2.5 ms latency: the magnitude bump at 1 kHz, the measured phase diving with frequency while the minimum phase stays small and the excess phase carries the linear delay ramp, and the group delays where the excess group delay reads a flat 2.5 ms" width="82%"></picture>

*A +6 dB peaking equalizer measured through a 2.5 ms processing latency: the
minimum-phase part carries only the small phase wiggle an equalizer could
invert, the excess phase is the pure $-2\pi f t_0$ ramp of the delay, and the
excess group delay reads the latency directly as a flat 2.5 ms line.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sp_signal
from phonometry import phase_decomposition

fs = 48000.0
delay = int(0.0025 * fs)                      # a 2.5 ms processing latency
gain_a = 10.0 ** (6.0 / 40.0)                 # +6 dB peaking EQ at 1 kHz, Q = 1
w0 = 2.0 * np.pi * 1000.0 / fs
alpha = np.sin(w0) / 2.0
b = np.array([1 + alpha * gain_a, -2 * np.cos(w0), 1 - alpha * gain_a])
a = np.array([1 + alpha / gain_a, -2 * np.cos(w0), 1 - alpha / gain_a])
imp = np.zeros(16384)
imp[delay] = 1.0
ir = sp_signal.lfilter(b / a[0], a / a[0], imp)

res = phase_decomposition(np.fft.rfft(ir), fs)
res.plot()   # |H|, the three phases and the group delays
plt.show()
```

</details>

The decomposition $H = H_\text{min} H_\text{ap}$ splits what an equalizer can invert
($H_\text{min}$, minimum phase, causal and causally invertible) from what it never
can ($H_\text{ap}$, the all-pass excess: latency plus non-minimum-phase zeros such
as reflections). The excess phase is `0` for a minimum-phase response and
exactly $-2\pi f t_0$ for a pure latency $t_0$; its group delay reads the
latency in seconds.

Numerical contract, pinned by the tests: on a strictly minimum-phase biquad
sampled on a dense grid the reconstructed phase matches the true phase to
better than `1e-12` rad; the group delay of a first-order allpass matches
the closed form $(1-a^2)/(1+2a\cos\omega+a^2)$ to `1e-5` samples; the excess group
delay of a delayed biquad returns the delay to `1e-6` samples. The
precautions are documented with the API: the response must be sampled
uniformly from DC to Nyquist inclusive (the `rfft` layout) and densely
enough that the underlying impulse response fits the implied record;
magnitude zeros (band-pass edges, notch bottoms) are floored and are not
representable by a minimum-phase system; the `oversample` factor
(trigonometric interpolation of the magnitude before the cepstrum)
mitigates the cepstral aliasing that near-circle zeros cause on coarse
grids.

## Relation to other tools

- [`impulse_response`](../../buildings/rooms/room-acoustics.md) recovers the *linear* IR from the
  same sweep recording and simply discards the negative-time distortion
  products; `swept_sine_distortion` is the tool that reads them.
- [`thd` / `harmonic_analysis`](electroacoustics.md) measure distortion
  from a steady tone at one frequency; the sweep separator returns the same
  ratios as a continuous function of frequency, from one measurement.
- The phase utilities operate on any one-sided *complex* response: an
  `rfft` of a measured IR, or the `response` of a
  [`transfer_function`](electroacoustics.md) estimate on a uniform grid.
  `minimum_phase` alone also accepts a plain magnitude array, e.g. a
  design target for equalization.

## What this guide covers

**Covered.** Farina's exponential-sweep deconvolution (AES preprint 5093, 2000)
and the phase-coherent synchronized swept sine of Novak, Lotton & Simon (2015):
the harmonic separation, the closed-form inverse-filter spectrum and the
fractional-sample de-skewing, through `swept_sine_distortion` and
`synchronized_sweep_signal`. With them, the acquisition conditions the answer
depends on — the drive level and its `amplitude` reference, the `fade` that has
to reach both calls, the clipping and tail requirements, the time-variance and
impulsive-noise failure modes, and the sizing rules relating $T$, $N$ and
$f_2$. The phase utilities `minimum_phase`, `group_delay`, `excess_phase` and
`phase_decomposition` implement the Hilbert relation of Bendat & Piersol
Section 13.1.4 through the real cepstrum.

**Not covered.** Intermodulation and dynamic intermodulation
(IEC 60268-3 clauses 14.12.7-10) come from steady tones on the
[electroacoustics](electroacoustics.md) page, not from a sweep; this page
separates harmonic orders and nothing else. Müller & Massarani is cited for
practice — fades, inverse-filter technique — not for an implemented formula.
Under `method="farina"` the harmonic phases of $H_2 \dots H_N$ are returned but
should be ignored: the plain exponential sweep breaks the time-shift/harmonic
equivalence the synchronized method depends on. And nothing here polices the
acquisition conditions: the functions take the arrays they are handed, so the
drive level, the absence of clipping and the length of the tail are the
operator's to keep and to report.

## See also

- [Electroacoustics](electroacoustics.md): the
  steady-tone $\mathrm{THD}$, THD+N and intermodulation set of IEC 60268-3
  that this sweep complements, and the operating point they are all defined at.
- [Loudspeaker Characterisation (IEC 60268-5)](loudspeakers.md):
  the THD(f) curve produced here is the distortion panel of that
  rated-characteristics report.
- [Room acoustics](../../buildings/rooms/room-acoustics.md): `impulse_response`,
  which is the linear half of the same recording.
- API reference: [`electroacoustics.swept_sine`](https://jmrplens.github.io/phonometry/reference/api/electroacoustics/swept-sine/)
  and [`signals.phase`](https://jmrplens.github.io/phonometry/reference/api/signals/phase/).

## References

- Farina, A. (2000). Simultaneous measurement of impulse response and
  distortion with a swept-sine technique. *108th AES Convention*, Paris,
  preprint 5093. The exponential-sweep deconvolution and the $L\ln(n)$
  harmonic separation.
- Novak, A., Lotton, P., & Simon, L. (2015). Synchronized swept-sine:
  Theory, application and implementation. *Journal of the Audio Engineering
  Society*, 63(10), 786-798.
  [doi:10.17743/jaes.2015.0071](https://doi.org/10.17743/jaes.2015.0071).
  The synchronization condition, the analytic inverse-filter spectrum and
  the fractional-delay separation.
- Müller, S., & Massarani, P. (2001). Transfer-function measurement with
  sweeps. *Journal of the Audio Engineering Society*, 49(6), 443-471.
  The sweep-measurement monograph behind the practice notes (inverse
  filters, fades, distortion rejection).
- Bendat, J. S., & Piersol, A. G. (2010). *Random Data: Analysis and
  Measurement Procedures* (4th ed.). Wiley. ISBN 978-0-470-24877-5.
  [doi:10.1002/9781118032428](https://doi.org/10.1002/9781118032428).
  Section 13.1.4: the Hilbert relation between log-magnitude and phase
  behind the minimum-phase reconstruction.
