← [Documentation index](../../README.md)

# Calibrated spectral analysis (Bendat & Piersol)

A spectrum without its uncertainty is half a measurement. This page covers the
Welch spectral estimators of `phonometry.signals` that report, next to the
spectrum itself, the statistical quality of the estimate following Bendat &
Piersol, *Random Data: Analysis and Measurement Procedures* (4th ed., 2010):
the **power spectral density** and **cross-spectral density** with the
effective number of averages, the normalized random error and chi-square
confidence intervals; the **coherent output spectrum** that splits a measured
output into the part linearly explained by the input and the noise remainder,
with the spectral signal-to-noise ratio; a **fractional-octave smoother** with
a constant-power kernel; **colored-noise generators** with an exact
power-law slope for exercising all of the above; the **window figures of
merit** of Harris (1978) for choosing the taper; and a **Thomson multitaper
estimator** (Percival & Walden, 1993) for records too short to segment.
Every error formula is a closed form from the sources, verified by seeded
Monte Carlo in the test suite.

## 1. Power spectral density with its statistical error

`power_spectral_density` estimates the one-sided autospectral density
$G_{xx}(f)$ by Welch's method: the record is split into tapered (Hann by
default), 50 %-overlapped segments whose periodograms are averaged. No
detrending is applied, so absolute calibration is preserved: a signal in
pascals yields Pa²/Hz. Two scalings are available: `'density'` (units²/Hz,
integrates to the signal power) and `'spectrum'` (units², reads the power of
discrete tones directly).

The estimator is a fixed pipeline: the window sets the resolution bandwidth,
and the effective averages follow from the record length (the raw segment
count) together with the window and overlap. Once those are fixed every
quality figure follows from the main design choice, the segment length. The
diagram traces it with the numbers of this page's example.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_spectral_analysis_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_spectral_analysis.svg" alt="Block diagram of the Welch PSD pipeline: a 20 second pink noise record at 48 kilohertz is split into 50 percent overlapped segments of 4096 samples giving 467 segments and an 11.7 hertz bin spacing, each segment is Hann tapered for a resolution bandwidth of 17.6 hertz, the one-sided squared FFT periodograms are averaged into 442 effective averages, and the result is Gxx of f with a chi-square confidence interval, a random error of 1 over the square root of n d equal to 4.8 percent and about 885 degrees of freedom; a final note states the trade-off that longer segments buy resolution but spend averages" width="92%"></picture>

Averaging $n_d$ independent segments gives the estimate $2 n_d$ chi-square
degrees of freedom (Eq. 8.162), from which everything else follows:

$$
\varepsilon_r[\hat{G}_{xx}] = \frac{1}{\sqrt{n_d}}, \qquad
\frac{n\,\hat{G}_{xx}}{\chi^2_{n;\,\alpha/2}} \le G_{xx} \le
\frac{n\,\hat{G}_{xx}}{\chi^2_{n;\,1-\alpha/2}}, \quad n = 2 n_d .
$$

With overlapped, tapered segments the averages are correlated, so the result
reports both the raw segment count (`n_segments`) and the **effective**
number of independent averages (`n_averages`), computed with the
window-correlation formula of Welch (1967) that Bendat & Piersol reference in
Section 11.5.2.2; for a Hann taper at 50 % overlap it is roughly 0.95 of the raw
count. The random error and the confidence interval use the effective value.
At DC, and at Nyquist for an even segment length, the one-sided spectrum
has a single real Fourier component, so those bins carry half the degrees of
freedom and a correspondingly wider interval.

```python
from phonometry import power_spectral_density

res = power_spectral_density(signal, fs)          # Hann, 50 % overlap, 95 % CI
print(res.n_averages, res.random_error)           # nd and 1/sqrt(nd)
print(res.ci_lower[10], res.psd[10], res.ci_upper[10])
res.plot()                                        # PSD in dB with the CI band
```

The **resolution bias** is the other half of the error budget: a finite
analysis bandwidth $B_e$ (reported as `resolution_bandwidth`, the effective
noise bandwidth of the taper) smooths sharp spectral features, always in the
direction of reduced dynamic range (Eq. 8.139). For a resonance peak of
half-power bandwidth $B_r$, the first-order normalized bias is the closed form
of Eq. 8.141, exposed as `resolution_bias_error`:

$$
\varepsilon_b[\hat{G}_{xx}(f_r)] \approx -\frac{1}{3}\left(\frac{B_e}{B_r}\right)^2 .
$$

```python
from phonometry import resolution_bias_error

eps_b = resolution_bias_error(res.resolution_bandwidth, 25.0)  # Br = 25 Hz peak
```

Narrow $B_e$ (long segments) suppresses the bias but leaves fewer averages and
a larger random error; the two requirements on segment length pull in
opposite directions, which is exactly the trade-off the reported numbers make
visible.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/psd_confidence_smoothing_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/psd_confidence_smoothing.svg" alt="Welch power spectral density of pink noise in dB per Hz over 20 Hz to 20 kHz, with the 95 percent chi-square confidence band shaded around the estimate, the 1/3-octave smoothed curve on top and the exact -3.01 dB per octave power law as a dashed reference line" width="82%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import (
    fractional_octave_smoothing,
    noise_signal,
    power_spectral_density,
)

fs = 48000.0
x = noise_signal(fs, 20.0, color="pink", seed=11)
res = power_spectral_density(x, fs, nperseg=4096)
band = (res.frequencies >= 20.0) & (res.frequencies <= 20000.0)
freqs = res.frequencies[band]
smooth = fractional_octave_smoothing(res.frequencies, res.psd, 3.0)[band]

fig, ax = plt.subplots(figsize=(10, 6))
ax.fill_between(freqs, 10 * np.log10(res.ci_lower[band]),
                10 * np.log10(res.ci_upper[band]), alpha=0.3,
                label="95 % chi-square confidence interval")
ax.semilogx(freqs, 10 * np.log10(res.psd[band]), lw=1.0,
            label="Welch PSD estimate")
ax.semilogx(freqs, 10 * np.log10(smooth), lw=2.2,
            label="1/3-octave smoothed")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("PSD [dB re 1/Hz]")
ax.legend()
plt.show()
```

</details>

## 2. Cross-spectral density

`cross_spectral_density` estimates the complex $G_{xy}(f)$ between two channels
with the same Welch core, and reports the ordinary coherence
$\gamma^2_{xy} = |G_{xy}|^2/(G_{xx}\,G_{yy})$ together with the Bendat &
Piersol random errors of
the magnitude and phase (Eqs. 9.33 and 9.52, with the measured coherence in
place of the unknown true value, as the book recommends for measured data):

$$
\varepsilon_r[|\hat{G}_{xy}|] = \frac{1}{|\gamma_{xy}|\sqrt{n_d}}, \qquad
\mathrm{s.d.}[\hat{\theta}_{xy}] =
\frac{\left(1-\gamma^2_{xy}\right)^{1/2}}{|\gamma_{xy}|\sqrt{2 n_d}} .
$$

Both shrink as the coherence approaches one: a strongly coherent pair needs
far fewer averages for the same confidence. The phase is unwrapped, so its
slope against frequency is the group delay
$\tau_g = -\mathrm{d}\varphi/(2\pi\,\mathrm{d}f)$; for a pure
delay path the phase is linear and that slope reads the propagation delay
directly.

```python
from phonometry import cross_spectral_density

res = cross_spectral_density(x, y, fs)
print(res.magnitude_random_error[100], res.phase_std[100])  # bin 100 errors
res.plot()   # magnitude, phase with ±sigma band, coherence
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/cross_spectral_density_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/cross_spectral_density.svg" alt="Two panels for the cross-spectral density of a two-sensor path with a 2 millisecond delay: the Welch magnitude estimate fluctuating around a flat level, and below it the unwrapped cross-spectrum phase falling as a straight line on a logarithmic frequency axis, lying exactly on the dashed minus two pi f tau reference with a narrow one-sigma band around it" width="86%"></picture>

*The cross-spectral density of a 2 ms delay path: the unwrapped phase is
exactly the line $-2\pi f\tau$, so its slope reads the propagation delay
directly,
and the ±1 s.d. band of Eq. 9.52 quantifies how far to trust it per
frequency.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import cross_spectral_density, noise_signal

fs = 8000.0
tau = 0.002                                   # 2 ms = 16 samples
delay = int(tau * fs)
x = noise_signal(fs, 8.0, seed=8)
noise = noise_signal(fs, 8.0, rms=0.3, seed=9)
y = 0.9 * np.concatenate([np.zeros(delay), x[:-delay]]) + noise

res = cross_spectral_density(x, y, fs)

# One line — magnitude, phase with its ±sigma band and coherence:
res.plot()
plt.show()

# By hand, from the fields the result carries:
band = (res.frequencies >= 20) & (res.frequencies <= 3500)
freqs = res.frequencies[band]
fig, (ax_m, ax_p) = plt.subplots(2, 1, sharex=True)
ax_m.semilogx(freqs, 10 * np.log10(res.magnitude[band]),
              label="|Gxy| (Welch estimate)")
ax_m.set_ylabel("Magnitude [dB]")
ax_p.semilogx(freqs, res.phase[band], label="Unwrapped phase")
ax_p.fill_between(freqs, res.phase[band] - res.phase_std[band],
                  res.phase[band] + res.phase_std[band], alpha=0.25,
                  label="±1 s.d. (Eq. 9.52)")
ax_p.semilogx(freqs, -2 * np.pi * freqs * tau, "r--",
              label="slope -2·pi·f·tau")
ax_p.set(xlabel="Frequency [Hz]", ylabel="Phase [rad]")
for ax in (ax_m, ax_p):
    ax.legend()
plt.show()
```

</details>

## 3. Coherent output spectrum and spectral SNR

In the single-input/single-output model the measured output autospectrum
splits exactly into the part linearly explained by the input and the
uncorrelated remainder (Eqs. 9.55–9.57):

$$
G_{vv} = \gamma^2_{xy}\,G_{yy}, \qquad
G_{nn} = \left(1-\gamma^2_{xy}\right) G_{yy}, \qquad
\mathrm{SNR}(f) = \frac{\gamma^2_{xy}}{1-\gamma^2_{xy}} .
$$

`coherent_output_spectrum` returns all three spectra, the spectral
signal-to-noise ratio (linear and in dB) and the random error of the coherent
output estimate (Eq. 9.73), plus the first-order propagation of the coherence
error through the SNR:

$$
\varepsilon_r[\hat{G}_{vv}] =
\frac{\left(2-\gamma^2_{xy}\right)^{1/2}}{|\gamma_{xy}|\sqrt{n_d}}, \qquad
\varepsilon_r[\widehat{\mathrm{SNR}}] = \frac{\sqrt{2}}{|\gamma_{xy}|\sqrt{n_d}} .
$$

For additive uncorrelated output noise of known level the coherence has the
closed form $\gamma^2 = \mathrm{SNR}/(1+\mathrm{SNR})$, which makes the whole
chain verifiable with a
synthetic signal:

```python
import numpy as np
from phonometry import coherent_output_spectrum, noise_signal

fs = 48000.0
x = noise_signal(fs, 8.0, color="white", seed=1)
noise = noise_signal(fs, 8.0, color="white", rms=0.5, seed=2)
y = 0.8 * x + noise                      # SNR = 0.64/0.25 at every frequency

res = coherent_output_spectrum(x, y, fs)
print(np.median(res.coherence))          # -> SNR/(1+SNR) = 0.719
print(np.median(res.snr_db))             # -> 10·lg(2.56) = 4.1 dB
res.plot()                               # Gyy, Gvv, Gnn and the SNR panel
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/coherent_output_snr_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/coherent_output_snr.svg" alt="Two panels for the coherent output spectrum of a white noise system with additive noise: the measured output spectrum with the coherent part about 1.5 decibels below it and the uncorrelated noise floor about 6 decibels lower still, and below them the spectral signal-to-noise ratio fluctuating around the dashed closed-form line at 4.1 decibels" width="86%"></picture>

*The exact split of the snippet's model: $G_{vv} = \gamma^2 G_{yy}$ explains
the output except for the flat noise remainder $G_{nn}$, and the spectral SNR
scatters around its closed form $10\log_{10}(0.64/0.25) = 4.1$ dB at every
frequency.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import coherent_output_spectrum, noise_signal

fs = 48000.0
x = noise_signal(fs, 8.0, color="white", seed=1)
noise = noise_signal(fs, 8.0, color="white", rms=0.5, seed=2)
y = 0.8 * x + noise                      # SNR = 0.64/0.25 per band

res = coherent_output_spectrum(x, y, fs, nperseg=2048)

# One line — the three spectra and the SNR panel:
res.plot()
plt.show()

# By hand, from the fields the result carries:
band = (res.frequencies >= 20) & (res.frequencies <= 20000)
freqs = res.frequencies[band]
fig, (ax_g, ax_s) = plt.subplots(2, 1, sharex=True)
for values, style, label in ((res.output_psd, "-", "Gyy (measured)"),
                             (res.coherent_psd, "--", "Gvv (coherent)"),
                             (res.noise_psd, ":", "Gnn (noise)")):
    ax_g.semilogx(freqs, 10 * np.log10(values[band]), style, label=label)
ax_g.set_ylabel("Spectral density [dB re 1/Hz]")
ax_s.semilogx(freqs, res.snr_db[band], label="Spectral SNR [dB]")
ax_s.axhline(10 * np.log10(0.64 / 0.25), color="r", linestyle="--",
             label="closed form 4.1 dB")
ax_s.set(xlabel="Frequency [Hz]", ylabel="SNR [dB]")
for ax in (ax_g, ax_s):
    ax.legend()
plt.show()
```

</details>

The `coherence_bias` field reports the small positive bias of the coherence
estimate, $b[\hat{\gamma}^2] \approx (1-\gamma^2)^2/n_d$ (Eq. 9.75),
negligible once $n_d$ reaches a
few hundred, and another reason to average generously before trusting a low
coherence.

## 4. Fractional-octave smoothing

`fractional_octave_smoothing` averages a spectrum over a rectangular window
of constant relative width: 1/n octave, $[f\cdot 2^{-1/2n},\ f\cdot 2^{+1/2n}]$
around each frequency. This is the constant-percentage resolution bandwidth that
Bendat & Piersol recommend for the spectra of resonant systems
(Section 8.5.3), and the de facto standard for presenting loudspeaker and
room responses. The average is always computed on **power** (amplitudes are
squared first, dB levels converted and back), so band power is conserved
rather than amplitude, and a flat spectrum passes through exactly unchanged.

```python
from phonometry import fractional_octave_smoothing

smooth_psd = fractional_octave_smoothing(res.frequencies, res.psd, 3.0)
smooth_mag = fractional_octave_smoothing(freqs, np.abs(response), 6.0,
                                         domain="amplitude")  # an FRF |H|
smooth_db = fractional_octave_smoothing(freqs, levels, 3.0, domain="db")  # dB curve
```

A single spectral line with PSD ordinate $P$ (units²/Hz) in a bin of width
$\Delta f$ smooths to the closed-form level
$P\,\Delta f / (f_0(2^{1/2n} - 2^{-1/2n}))$
over one kernel width, the oracle pinned in the tests.

## 5. Colored-noise generators

`noise_signal` produces Gaussian noise whose PSD follows
$G_{xx}(f) \propto f^\alpha$ exactly in expectation: seeded white noise is
shaped in the frequency domain by the exact magnitude response
$(f/f_\text{ref})^{\alpha/2}$ bin by bin (a zero-phase filter applied
circularly), so a measured slope deviates from the power law
only by the random error of the spectral estimate, not by the piecewise or
few-pole pink approximations whose slope ripples by fractions of a dB. The
record is zero-mean and rescaled to the requested RMS exactly, and the same
seed reproduces the same record bit for bit.

| color | $\alpha$ | PSD slope |
|---|---|---|
| `white` | 0 | 0 dB/octave |
| `pink` | -1 | -3.01 dB/octave |
| `red` (Brownian) | -2 | -6.02 dB/octave |
| `blue` | +1 | +3.01 dB/octave |
| `violet` | +2 | +6.02 dB/octave |

```python
from phonometry import noise_signal

pink = noise_signal(48000, 10.0, color="pink", seed=7)     # deterministic
white = noise_signal(48000, 10.0, color="white", rms=0.5, seed=7)
```

Measured over three decades (20 Hz – 20 kHz) with the estimator of section 1,
the regression slope of each color lands within a few thousandths of a
dB/octave of the exact value; the conformance suite pins the pink slope at
-3.0116 against the exact -3.0103.

## 6. Choosing the window

Every estimator on this page accepts any window `scipy.signal.get_window`
knows, but the choice is a quantified trade-off, not a preference.
`window_metrics` computes the figures of merit Harris (1978) tabulated, for
any taper and length, sampled DFT-even exactly as the Welch estimators apply
it:

- **ENBW** (equivalent noise bandwidth, in bins): how much wider than one
  bin the effective analysis bandwidth is. This is the same number the PSD
  result reports as `resolution_bandwidth` (`ENBW·fs/nperseg` in Hz), and
  it enters directly in the tone/noise trade: a broadband noise floor read
  from a windowed spectrum sits $10\log_{10}(\mathrm{ENBW})$ dB above the true
  density.
- **Coherent gain**: the DC gain $\sum w/N$ a bin-centered tone is scaled by.
- **Scalloping loss**: the worst-case attenuation of a tone that falls
  midway between two bins (3.92 dB for rectangular, 1.42 dB for Hann).
- **Worst-case processing loss**: scalloping plus $10\log_{10}(\mathrm{ENBW})$, the
  worst-case reduction in output SNR for tone detection in white noise.
- **Highest sidelobe** and **main-lobe -3 dB width**: leakage floor versus
  resolution.

```python
from phonometry import window_metrics

m = window_metrics("hann", 2048)
print(m.enbw_bins)              # 1.5, exactly
print(m.scalloping_loss_db)     # 1.42 dB
print(m.highest_sidelobe_db)    # -31.5 dB
m.plot()                        # window + spectrum with metrics marked
```

The closed forms anchor the tests: ENBW is exactly 1 for rectangular, 3/2
for Hann, 1987/1458 for Hamming and 1523/882 for Blackman (DFT-even
sampling), and the rectangular scalloping loss is $20\log_{10}(N\sin(\pi/2N))$,
the Dirichlet kernel evaluated half a bin off center.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/window_functions_tradeoff_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/window_functions_tradeoff.svg" alt="Spectra of the rectangular, Hann, Hamming and Blackman windows over 16 DFT bins, showing the trade-off between main-lobe width and sidelobe level, with each window's equivalent noise bandwidth and highest sidelobe level in the legend" width="82%"></picture>

The Hann default of this module is the balanced choice: fast sidelobe
falloff (-18 dB/octave) protects noise spectra from leakage, ENBW 1.5 costs
only 1.76 dB against the rectangular window, and its overlap correlation at
50 % keeps nearly all segment information in the effective average count.
Reach for a lower-sidelobe taper (Blackman, Kaiser with a high beta) when a
weak tone must be found next to a strong one, and accept the wider main
lobe; reach for the rectangular window only for self-windowing records
(transients that decay inside the segment) or bin-centered synthesis.

## 7. Multitaper estimation for short records

Welch's method buys stability with record length: each independent segment
adds two degrees of freedom, so a record that only fits a couple of segments
leaves an estimate barely better than a periodogram. `multitaper_psd`
implements the alternative of Thomson (1982), as developed by Percival &
Walden (1993, Chapter 7): the *whole* record is multiplied by $K$ orthogonal
discrete prolate spheroidal (Slepian) tapers - the sequences that concentrate
the most spectral-window energy inside a chosen design band $[-W, W]$ - and
the $K$ resulting *eigenspectra* are averaged:

$$
\hat{S}^{(mt)}(f) = \frac{1}{K}\sum_{k=0}^{K-1} \hat{S}_k(f), \qquad
\hat{S}_k(f) = \Delta t\,\Bigl|\sum_{t=1}^{N} h_{t,k}\,x_t\,
e^{-i 2\pi f t \Delta t}\Bigr|^2 .
$$

Because the tapers are orthogonal the eigenspectra are nearly uncorrelated,
so the average carries about $2K$ chi-square degrees of freedom from a
single record - the same statistical machinery as the Welch result (random
error, chi-square confidence interval), without segmenting. The
half-bandwidth $W = NW\,f_s/N$ is set through the duration x half-bandwidth
product $NW$ (default 4). $2W$ is the resolution of the estimate (reported
as `resolution_bandwidth`), and only the tapers below the Shannon number
$2NW$ keep their spectral-window energy inside the design band - their
concentrations $\lambda_k$ are reported as `eigenvalues`, and the default
taper count is $K = 2NW - 1$, all tapers with near-unity concentration.
Larger $NW$ admits more tapers (lower variance) at the cost of resolution.

```python
from phonometry import multitaper_psd

res = multitaper_psd(signal, fs)                 # NW = 4, K = 7, adaptive
print(res.degrees_of_freedom.mean())             # ~2K from one record
print(res.eigenvalues)                           # taper concentrations
res.plot()                                       # density with the CI band
```

By default the eigenspectra are combined with Thomson's **adaptive weights**
(P&W Eqs. 368a/370a, iterated to convergence). Each taper's weight at each
frequency balances the local spectrum against the broad-band leakage the
taper could carry:

$$
b_k(f) = \frac{S(f)}{\lambda_k S(f) + (1-\lambda_k)\,\sigma^2 \Delta t},
\qquad
\hat{S}^{(amt)}(f) =
\frac{\sum_k b_k^2(f)\,\lambda_k\,\hat{S}_k(f)}
     {\sum_k b_k^2(f)\,\lambda_k},
$$

so the leakier high-order tapers are downweighted exactly where the spectrum
is locally weak, and nothing is lost where it is locally white (for white
noise the weights are uniform). The price is bookkept honestly: the
equivalent degrees of freedom become frequency dependent,
$\nu(f) = 2\left(\sum_k d_k\right)^2/\sum_k d_k^2$ with
$d_k = b_k^2\lambda_k$ (P&W Eq. 370b), and the
confidence interval widens wherever leakage protection spent them.
`adaptive=False` selects the plain eigenvalue-weighted average instead.

Calibration matches the Welch estimators exactly: no detrending,
`'density'` integrates to the signal power, and `'spectrum'` reads $A^2/2$ at
the peak of a sinusoid of amplitude $A$ (a tone's power in `'density'`
scaling spreads over the $2W$ band). The Slepian tapers themselves come from
`scipy.signal.windows.dpss`; their concentrations reproduce the
quadruple-precision table of Percival & Walden (Table 382) to machine
precision, which is the anchor oracle of the test suite.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/multitaper_psd_confidence_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/multitaper_psd_confidence.svg" alt="Thomson multitaper spectral density of a 171 millisecond pink noise record in dB per Hz over 20 Hz to 20 kHz, with the 95 percent chi-square confidence band around the seven-taper adaptive estimate, the single-taper estimate as a jagged gray context line and the exact -3.01 dB per octave power law as a dashed reference" width="82%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import multitaper_psd, noise_signal

fs = 48000.0
x = noise_signal(fs, 8192 / fs, color="pink", seed=11)   # 171 ms record
single = multitaper_psd(x, fs, n_tapers=1, adaptive=False)
res = multitaper_psd(x, fs)                              # NW = 4, K = 7
band = (res.frequencies >= 20.0) & (res.frequencies <= 20000.0)
freqs = res.frequencies[band]

fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogx(freqs, 10 * np.log10(single.psd[band]), color="gray",
            alpha=0.45, lw=0.7, label="Single Slepian taper (K = 1)")
ax.fill_between(freqs, 10 * np.log10(res.ci_lower[band]),
                10 * np.log10(res.ci_upper[band]), alpha=0.3,
                label="95 % chi-square confidence interval")
ax.semilogx(freqs, 10 * np.log10(res.psd[band]), lw=1.2,
            label="Multitaper estimate (K = 7, adaptive)")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("PSD [dB re 1/Hz]")
ax.legend()
plt.show()
```

</details>

Reach for `multitaper_psd` when the record is too short to segment (room
impulse response tails, transient captures, single machine cycles) or when
a high-dynamic-range spectrum needs leakage protection that a Hann-windowed
Welch average cannot give; stay with `power_spectral_density` for long
records, where segment averaging is cheaper than $K$ full-length FFTs and
the two estimators agree.

## Relation to the H1/H2 estimators

The [frequency-response estimators](../../devices/electroacoustics/electroacoustics.md) `transfer_function`
and `coherence`, the two-microphone [sound intensity](../../devices/emission/intensity.md) probe and
these estimators all share one Welch core (same taper, overlap policy and
detrend-off calibration), so a PSD, a coherence and an $H_1$ computed with the
same segment length are mutually consistent bin by bin. The same cross-spectral
matrix underlies [multiple and partial coherence](miso-coherence.md), which
extends the ordinary coherence to several correlated inputs and one
output.

## What this guide covers

**Covered.** Bendat & Piersol's Welch estimators (*Random Data*, 4th ed., 2010,
Sections 5.2, 8.5, 9.1-9.2 and 11.5): `power_spectral_density` and
`cross_spectral_density` with the effective number of averages, the chi-square
confidence intervals and the resolution-bias error; the coherent output
spectrum and the spectral SNR; the Harris (1978) window figures of merit in
`window_metrics`; and the Thomson (1982) multitaper estimator of
`multitaper_psd`, following Percival & Walden Chapters 7-8.
`fractional_octave_smoothing` and `noise_signal` complete the toolbox.

**Not covered.** The frequency-response estimators `transfer_function` and
`coherence`, and the sound-intensity probe, share this page's Welch core but
are documented on
[Electroacoustics](../../devices/electroacoustics/electroacoustics.md) and
[Sound intensity](../../devices/emission/intensity.md). So is multiple and
partial coherence, on [MISO coherence](miso-coherence.md). And these are a
textbook's estimators, not a certification standard, so there is no clause
number and no compliance limit to check the page against.

## See also

- [Data qualification](../metrology/data-qualification.md): the stationarity every chi-square interval on this page assumes.
- [Calibration and dBFS](../metrology/calibration.md): where the factor that turns this page's densities into Pa²/Hz comes from.
- [MISO coherence](miso-coherence.md): the same cross-spectral matrix with several correlated inputs.
- [Time-frequency analysis](time-frequency.md): the same Welch segmentation displayed instead of averaged.
- [Levels](../levels/levels.md): the band-level side of the density/band conversion of section 4.
- API reference: [`signals.spectra`](https://jmrplens.github.io/phonometry/reference/api/signals/spectra/), [`signals.windows`](https://jmrplens.github.io/phonometry/reference/api/signals/windows/) and [`signals.multitaper`](https://jmrplens.github.io/phonometry/reference/api/signals/multitaper/).
- Theory: [Frequency Resolution vs FFT Bin Spacing](../../reference/theory/signal-analysis.md#frequency-resolution-vs-fft-bin-spacing): why the bin spacing of an FFT is not the resolution of the estimate, and what the window does to both.

## References

- Bendat, J. S., & Piersol, A. G. (2010). *Random Data: Analysis and
  Measurement Procedures* (4th ed.). Wiley. ISBN 978-0-470-24877-5.
  [doi:10.1002/9781118032428](https://doi.org/10.1002/9781118032428).
  Sections 5.2 and 8.5 (autospectra and their random/bias errors,
  chi-square intervals), 9.1–9.2 (cross-spectra, coherent output spectrum
  and their errors) and 11.5 (Welch processing, tapering and overlap).
- Welch, P. D. (1967). The use of fast Fourier transform for the estimation
  of power spectra: A method based on time averaging over short, modified
  periodograms. *IEEE Transactions on Audio and Electroacoustics*, 15(2),
  70–73. [doi:10.1109/TAU.1967.1161901](https://doi.org/10.1109/TAU.1967.1161901).
  The overlapped-segment variance formula behind the effective number of
  averages (Bendat & Piersol Section 11.5.2.2, Ref. 11).
- Harris, F. J. (1978). On the use of windows for harmonic analysis with the
  discrete Fourier transform. *Proceedings of the IEEE*, 66(1), 51–83.
  [doi:10.1109/PROC.1978.10837](https://doi.org/10.1109/PROC.1978.10837).
  The window figures of merit (Table 1) computed by `window_metrics`.
- Thomson, D. J. (1982). Spectrum estimation and harmonic analysis.
  *Proceedings of the IEEE*, 70(9), 1055–1096.
  [doi:10.1109/PROC.1982.12433](https://doi.org/10.1109/PROC.1982.12433).
  The multitaper method: Slepian tapers, eigenspectra and the adaptive
  weights implemented by `multitaper_psd`.
- Percival, D. B., & Walden, A. T. (1993). *Spectral Analysis for Physical
  Applications: Multitaper and Conventional Univariate Techniques*.
  Cambridge University Press. ISBN 978-0-521-43541-3.
  [doi:10.1017/CBO9780511622762](https://doi.org/10.1017/CBO9780511622762).
  Chapter 7 (multitaper estimation: eigenspectra, adaptive weighting,
  equivalent degrees of freedom) and Chapter 8 (the Slepian sequences);
  the Table 382 eigenvalues anchor the taper oracle in the test suite.
