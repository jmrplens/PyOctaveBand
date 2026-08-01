← [Documentation index](README.md)

# Multiple and partial coherence (Bendat & Piersol)

When several partially correlated sources drive one response, the ordinary
coherence of each source with the output is misleading: a source that only
*correlates* with the true cause inherits a spurious coherence through it, so
reading the ordinary coherences alone can credit the wrong source. Bendat &
Piersol, *Random Data* (4th ed., 2010, Chapter 7), resolve this for a
multiple-input/single-output (MISO) system with the **multiple** and
**partial** coherence functions. `miso_coherence` computes them from the same
Welch cross-spectral core as the rest of `phonometry.signals`, for several
correlated inputs and one output.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/miso_coherence_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/miso_coherence.svg" alt="Two-panel figure. Top: the measured output autospectrum in dB with the coherent output contribution of two inputs shaded underneath; input 1 fills the low band and input 2 the high band, with the residual noise far below. Bottom: for the correlated second input, its ordinary coherence sits around 0.3 across the low band even though it drives no low-frequency path, while its partial coherence collapses to zero there once the first input is conditioned out; the multiple coherence stays near one except at the crossover null" width="82%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from phonometry import miso_coherence, noise_signal

fs = 8192.0
# Input 1 drives a low-frequency path; input 2 = 0.7*x1 + independent noise
# drives a high-frequency path, so input 2 is correlated with input 1.
x1 = noise_signal(fs, 32.0, color="white", seed=1)
x2 = 0.7 * x1 + noise_signal(fs, 32.0, color="white", seed=2)
low = signal.butter(4, 400.0, fs=fs, output="sos")
high = signal.butter(4, 1500.0, btype="high", fs=fs, output="sos")
noise = noise_signal(fs, 32.0, color="white", rms=0.05, seed=3)
y = signal.sosfilt(low, x1) + signal.sosfilt(high, x2) + noise

res = miso_coherence([x1, x2], y, fs, nperseg=2048)
f = res.frequencies
band = (f >= 20.0) & (f <= 4000.0)

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 7.4), sharex=True)
db = lambda v: 10 * np.log10(v)
ax_top.semilogx(f[band], db(res.output_psd[band]), color="gray",
                label="Measured output")
for i, color in ((0, "#1f77b4"), (1, "#2ca02c")):
    ax_top.semilogx(f[band], db(res.coherent_output_spectra[i][band]),
                    color=color, label=f"Input {i + 1} contribution")
ax_top.semilogx(f[band], db(res.noise_psd[band]), "--", color="#d62728",
                label="Residual noise")
ax_top.set_ylabel("Coherent output [dB re 1/Hz]")
ax_top.legend()

ax_bot.semilogx(f[band], res.ordinary_coherence[1][band], ":", color="#2ca02c",
                label="Input 2 ordinary (inflated by x1)")
ax_bot.semilogx(f[band], res.partial_coherence[1][band], color="#2ca02c",
                label="Input 2 partial (x1 removed)")
ax_bot.semilogx(f[band], res.multiple_coherence[band], color="black",
                label="Multiple")
ax_bot.set_xlabel("Frequency [Hz]")
ax_bot.set_ylabel("Coherence")
ax_bot.set_ylim(0, 1.05)
ax_bot.legend()
plt.show()
```

</details>

The `.plot()` method draws the same two panels in one call (`res.plot()`, or
`res.plot(language="es")` for Spanish labels).

## 1. Ordinary, multiple and partial coherence

For a system with $q$ inputs $x_1..x_q$ and output $y$, `miso_coherence`
estimates every auto- and cross-spectrum by Welch's method and reports three
coherence functions.

The conditioning story fits in one block diagram: two correlated inputs
drive their own paths into one measured output, and the Welch cross-spectral
matrix is what turns the tangle into per-source numbers.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_miso_coherence_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_miso_coherence.svg" alt="Block diagram of a two-input MISO system: input x1 is white noise and input x2 equals 0.7 times x1 plus independent noise, x1 drives a 400 hertz low-pass path and x2 a 1.5 kilohertz high-pass path, and both sum with additive noise into the output y; the output feeds the Welch cross-spectral matrix Gxx and Gxy at nperseg 2048, conditioned by Schur steps in descending ordinary coherence order, which yields on one side the multiple and partial coherences, where input 2 in the 100 to 300 hertz band drops from an ordinary coherence of 0.32 to a partial coherence of 0.00, and on the other the per-source contributions Gvi that with the residual noise reconstruct Gyy exactly; a footnote records that each conditioning step spends one of the 242 effective averages" width="92%"></picture>

The **ordinary coherence** of input $i$ with the output, on its own
(Eq. 7.109), is the familiar single-input quantity
$\gamma^2_{iy} = |G_{iy}|^2/(G_{ii}\,G_{yy})$.

The **multiple coherence** (Eq. 7.35) is the fraction of the output
autospectrum linearly explained by *all* inputs jointly,
$\gamma^2_{y:x} = G_{xy}^H\,G_{xx}^{-1}\,G_{xy}/G_{yy} = 1 - G_{nn}/G_{yy}$,
where $G_{nn}$ is the residual output
spectrum uncorrelated with every input. If the output carries additive
uncorrelated noise of per-band signal-to-noise ratio $\mathrm{SNR}$, this is
exactly $\mathrm{SNR}/(1+\mathrm{SNR})$, the closed-form oracle used to verify
the implementation.

The **partial coherence** of input $i$ (Eq. 7.87),
$\gamma^2_{iy\cdot(i-1)!} = |G_{iy\cdot(i-1)!}|^2/(G_{ii\cdot(i-1)!}\,G_{yy})$,
is its coherence with the
output once the linear effect of the inputs *before it in the conditioning
order* has been removed. The 4th-edition definition keeps the *total* output
$G_{yy}$ in the denominator, so the partial coherences of the ordered inputs sum
to the multiple coherence (Eq. 7.116) and reduce exactly to the ordinary
coherences when the inputs are mutually uncorrelated (Eq. 7.117).

```python
res = miso_coherence([x1, x2], y, fs)
res.ordinary_coherence   # shape (q, F): each input on its own
res.multiple_coherence   # shape (F,):  all inputs jointly
res.partial_coherence    # shape (q, F): conditioned on the preceding inputs
```

## 2. Conditioning: separating cause from correlation

The partial coherences come from the **conditioned spectra** $G_{ij\cdot r!}$,
computed by the Gaussian-elimination recursion of Section 7.3. Removing the
linear effect of a pivot input $r$ from every remaining record is one Schur
complement step (Eq. 7.94),
$G_{ij\cdot r!} = G_{ij\cdot(r-1)!} - G_{rj\cdot(r-1)!}\,G_{ir\cdot(r-1)!}/G_{rr\cdot(r-1)!}$,
applied for each
input in the conditioning order until only the residual output spectrum
$G_{yy\cdot q!}$ remains. In the figure above, input 2 is
$0.7\,x_1 + \text{independent noise}$, so it correlates with input 1 but
drives no low-frequency path. Its
ordinary coherence with the output therefore reads about 0.3 across the low
band, borrowed entirely through input 1, while its partial coherence collapses
to zero there once input 1 is conditioned out.

```python
low = (res.frequencies > 100.0) & (res.frequencies < 300.0)
res.ordinary_coherence[1][low].mean()   # ~0.32  (inflated through x1)
res.partial_coherence[1][low].mean()    # ~0.00  (x1 removed)
```

## 3. Which source dominates each band

The conditioning also splits the output power source by source. The **partial
coherent output spectrum** of input $i$ (Eq. 7.86),
$G_{v_i} = \gamma^2_{iy\cdot(i-1)!}\,G_{yy}$,
is the share of the output autospectrum it contributes, and the shares plus
the residual noise reconstruct the output exactly (Eqs. 7.88/7.121):
$\sum_i G_{v_i} + G_{nn} = G_{yy}$. Comparing the shares band by band answers "which
source dominates here?". `dominant_input()` returns, for every frequency, the
index of the input with the largest share.

```python
res.coherent_output_spectra        # shape (q, F): Gvi per input
dominant = res.dominant_input()    # index of the strongest source per bin
f = res.frequencies
dominant[np.argmin(abs(f - 200.0))]    # 0 (input 1 drives the low band)
dominant[np.argmin(abs(f - 2500.0))]   # 1 (input 2 drives the high band)
```

## 4. Statistical quality and the conditioning order

The random errors follow Section 9.3. Conditioning on the $i-1$ preceding
inputs costs $i-1$ degrees of freedom, so the $i$-th ordered input carries
$n_d-(i-1)$ effective averages (Eqs. 9.100/9.101) and the $q$-input multiple
coherence carries $n_d-(q-1)$ (Eq. 9.98). The result exposes
`multiple_coherence_random_error` and `coherent_output_random_error` alongside
the effective average count `n_averages`.

The ordinary and multiple coherences do not depend on the conditioning order,
but the partial coherences and the coherent-output decomposition do: each
input is conditioned on whatever precedes it. Absent a physical ordering,
Bendat & Piersol (Section 7.2.4) recommend ordering the inputs by descending
ordinary coherence with the output. Pass `order` to choose:

```python
res = miso_coherence([x1, x2, x3], y, fs, order=(2, 0, 1))
res.order            # (2, 0, 1): the order actually applied
res.plot()           # the two panels, recomputed in the applied order
```

Two practical pitfalls are worth naming. *Averaging*: every conditioning
step spends a degree of freedom, so with few segments the partial
coherences of the last-ordered inputs are the least trustworthy numbers on
the page; average generously before reading a small partial coherence as
zero. *Delay bias*: like the ordinary coherence, the Welch estimate is
biased low when a bulk delay between an input and the output becomes a
noticeable fraction of the segment length; remove known propagation delays
first (see [Correlation and delay](correlation-delay.md)) and keep
`nperseg` well above the longest remaining delay.

The estimators share the Welch core of the
[calibrated spectral analysis](spectral-analysis.md) page, so a MISO
coherence and a `power_spectral_density` computed with the same segment length
are consistent bin by bin.

## See also

- [Spectral analysis](spectral-analysis.md): the single-input coherent
  output spectrum this page generalizes, and the shared Welch core.
- [Correlation and delay](correlation-delay.md): estimating and removing
  the bulk delays that bias coherence low.
- [Multichannel and Performance](multichannel.md): the per-channel path
  this cross-channel analysis complements.

## References

- Bendat, J. S., & Piersol, A. G. (2010). *Random Data: Analysis and
  Measurement Procedures* (4th ed.). Wiley. ISBN 978-0-470-24877-5.
  [doi:10.1002/9781118032428](https://doi.org/10.1002/9781118032428).
  Chapter 7 (multiple-input/output relationships: ordinary coherence
  Eq. 7.109, multiple coherence Eq. 7.35, partial coherence Eq. 7.87, the
  conditioned-spectrum recursion Eq. 7.94, the coherent output decomposition
  Eqs. 7.86/7.116/7.121) and Section 9.3 (statistical errors of the multiple
  and conditioned estimates, Eqs. 9.98-9.101).
