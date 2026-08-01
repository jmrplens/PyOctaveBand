← [Documentation index](README.md)

# Data qualification: stationarity, level crossings and peaks (Bendat & Piersol)

Every average in this documentation - a [PSD](spectral-analysis.md), a
[Leq](levels.md), a
[GUM budget](gum-uncertainty.md) - assumes the record is
**stationary**: that the process generating it did not drift while it was
being measured. Bendat & Piersol devote Section 10.3 of *Random Data* to
qualifying records before analysis, and `phonometry.metrology` implements its
quantitative core: distribution-free **trend and stationarity tests** with
the book's own acceptance regions, and the **Rice statistics** - level
crossings, apparent frequency, peak rates and heights - that summarize what
a qualified Gaussian record looks like and flag one that is not.

Qualification is a decision, not a statistic, so it reads best as a flow:
segment the record, count reverse arrangements, and let the Table A.6 region
say whether averaging is allowed.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_data_qualification_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_data_qualification.svg" alt="Decision flow diagram of data qualification: a time record, before trusting any PSD, Leq or GUM average, is split into 20 equal segments whose mean square values form a sequence, the reverse arrangement count A with trend-free mean 95 is compared against the Table A.6 acceptance region, above 64 and at most 125, at the 5 percent level, and the diamond branches to a green stationary box, steady noise with A equal to 91 accepted so the chi-square intervals and error formulas hold, or to a red nonstationary box, a 20 percent gain ramp with A equal to 7 rejected, advising to split at the change or move to short-time analysis; footnotes name the runs test companion and the mean-square blind spot for frequency glides" width="92%"></picture>

## 1. The reverse arrangement test

Given a sequence of $N$ observations $x_1, \dots, x_N$ - parameter estimates,
segment levels, anything - count the pairs $i < j$ with $x_i > x_j$. Each
such pair is a **reverse arrangement**, and for independent observations of
one random variable their total $A$ has (B&P Eqs. (4.54)-(4.55))

$$
\mu_A = \frac{N(N-1)}{4}, \qquad
\sigma_A^2 = \frac{N(2N+5)(N-1)}{72},
$$

with no assumption about the distribution of the $x_i$. A monotonic trend
pushes $A$ to an extreme (0 for a rising sequence, $N(N-1)/2$ for a falling
one), so the hypothesis of *no trend* is accepted at significance $\alpha$
when $A$ falls inside a two-sided region - B&P Table A.6, whose
$\alpha = 0.05$ rows `trend_test` reproduces exactly and the conformance
suite pins. The book's Example 4.4 (twenty observations, $A = 86$, accepted
between 64 and 125) runs verbatim:

```python
from phonometry import trend_test

values = [5.2, 6.2, 3.7, 6.4, 3.9, 4.0, 3.9, 5.3, 4.0, 4.6,
          5.9, 6.5, 4.3, 5.7, 3.1, 5.6, 5.2, 3.9, 6.2, 5.0]

res = trend_test(values)                  # B&P Example 4.4
print(res.statistic, res.bounds)          # 86, (64, 125)
print(res.trend_free, round(res.p_value, 3))   # True, 0.586
res.plot()
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/trend_test_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/trend_test.svg" alt="Two sequences of twenty observations plotted against the sample index: Bendat and Piersol Example 4.4, which fluctuates around five with eighty-six reverse arrangements and is accepted as trend free, and the same fluctuations with an added rising drift climbing from about five to nine, whose thirty-eight reverse arrangements fall below the lower acceptance bound of sixty-four and are rejected as trended at the five percent level" width="82%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import trend_test

example = np.array([5.2, 6.2, 3.7, 6.4, 3.9, 4.0, 3.9, 5.3, 4.0, 4.6,
                    5.9, 6.5, 4.3, 5.7, 3.1, 5.6, 5.2, 3.9, 6.2, 5.0])
drifting = example + np.linspace(0.0, 4.0, example.size)   # rising drift

res_flat = trend_test(example)
res_drift = trend_test(drifting)

fig, ax = plt.subplots(figsize=(10, 6))
index = np.arange(1, example.size + 1)
ax.plot(index, res_flat.values, "o-",
        label=f"Example 4.4: A = {res_flat.statistic}, accepted")
ax.plot(index, res_drift.values, "s-",
        label=f"Rising drift: A = {res_drift.statistic}, rejected")
ax.set_xlabel("Sample index")
ax.set_ylabel("Sequence value")
ax.legend()
plt.show()
```

</details>

The p-value comes from the exact null distribution of $A$ (the inversion
counts of a random permutation), computed up to $N = 100$ - the range of
Table A.6 - and from the book's normal approximation beyond; the verdict
follows the book's tabulated region. `.plot()` draws the tested sequence
against its sample index with the count, the acceptance region and the
verdict in the legend (for `method="runs"` it also marks the sequence
median that classifies each value).

## 2. Stationarity of a record

The B&P Sec. 10.3.1.1 procedure turns the trend test into a stationarity
test for a single time history: divide the record into $N$ equal intervals
long enough to be independent, compute a **mean square value per interval**,
and test that sequence. Nothing needs to be known about the record's
bandwidth, averaging distribution or units, and the test works on mean
values, rms values or variances just as well (`statistic=`). A running
20 % gain drift - the book's Example 10.3 scenario - is caught immediately,
while the same noise without the drift passes:

```python
import numpy as np
from phonometry import stationarity_test

fs = 8192.0
n = 1 << 16
noise = np.random.default_rng(42).standard_normal(n)

res = stationarity_test(noise, fs)        # 20 segments, mean squares
print(res.stationary, res.count, res.bounds)   # True, 91, (64, 125)

drifting = noise * np.linspace(1.0, 1.2, n)    # +20 % gain ramp
res = stationarity_test(drifting, fs)
print(res.stationary, res.count)          # False, 7  (upward trend -> low A)
res.plot()
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/stationarity_test_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/stationarity_test.svg" alt="Twenty segment mean square values for two noise records: a steady record whose values fluctuate around one with a reverse arrangement count of ninety-one, accepted as stationary, and the same noise with a twenty percent gain ramp whose segment mean squares climb steadily to one point five, giving only seven reverse arrangements, rejected as nonstationary at the five percent level" width="82%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import stationarity_test

fs = 8192.0
n = 1 << 16
steady = np.random.default_rng(42).standard_normal(n)
ramp = np.random.default_rng(42).standard_normal(n) * np.linspace(1.0, 1.2, n)

res_steady = stationarity_test(steady, fs)
res_ramp = stationarity_test(ramp, fs)

fig, ax = plt.subplots(figsize=(10, 6))
index = np.arange(1, res_steady.n_segments + 1)
ax.plot(index, res_steady.segment_values, "o-",
        label=f"Steady noise: A = {res_steady.count}, accepted")
ax.plot(index, res_ramp.segment_values, "s-",
        label=f"+20 % gain ramp: A = {res_ramp.count}, rejected")
ax.set_xlabel("Segment index")
ax.set_ylabel("Segment mean square")
ax.legend()
plt.show()
```

</details>

The result records the per-segment sequence (`segment_values`,
`segment_times`), the count, the Table A.6 `bounds`, the exact `p_value`
and the `stationary` verdict; `.plot()` draws the sequence with the verdict
in the legend. The default of 20 segments matches the book's worked
examples - more segments resolve faster drifts but each interval must stay
long against the record's lowest frequencies for the values to be
independent.

Two caveats from the book are worth repeating. A record can be
nonstationary with a stationary mean square (a frequency glide, for
instance), so pass `statistic="mean"` or test band-filtered versions when
that matters; and the test needs the *trend* to be slow against the segment
length, or it dissolves into the random fluctuation of the segment values.

## 3. The runs test

The classical companion (tabulated in the third edition of *Random Data*;
the exact distribution is Wald & Wolfowitz 1940) classifies each value as
above or below the sequence median and counts **runs** of like
classification. A trend or slow drift produces few long runs; rapid
alternation produces too many. `method="runs"` applies it with the exact
conditional distribution at any $N$ - for twenty values split 10/10 the
acceptance region at $\alpha = 0.05$ is the classical $(6, 15]$:

```python
import numpy as np
from phonometry import trend_test

rng = np.random.default_rng(3)
res = trend_test(rng.standard_normal(40), method="runs")
print(res.statistic, res.bounds, res.trend_free)
res.plot()   # the sequence about its median with the runs verdict

alternating = np.tile([1.0, -1.0], 10)   # 20 runs: rejected the other way
print(trend_test(alternating, method="runs").trend_free)   # False
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/runs_test_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/runs_test.svg" alt="Two panels of sequences classified about their median, points above in blue and below in red. Left: forty Gaussian observations with twenty runs inside the acceptance region from fourteen to twenty-seven, verdict trend-free. Right: a strictly alternating twenty-value sequence also giving twenty runs, but against the acceptance region from six to fifteen for twenty values, verdict rejected for too-rapid alternation" width="92%"></picture>

*The two-sided verdict of the runs test: 20 runs is unremarkable for 40
Gaussian observations (left, accepted), but the same count from a strictly
alternating 20-value sequence exceeds its upper bound (right, rejected):
too *many* runs is as non-random as too few.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import trend_test

rng = np.random.default_rng(3)
sequences = [rng.standard_normal(40), np.tile([1.0, -1.0], 10)]

# One line per sequence — the tested values with the verdict:
trend_test(sequences[0], method="runs").plot()
plt.show()

# Both verdicts side by side, classified about the median:
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, seq in zip(axes, sequences):
    res = trend_test(seq, method="runs")
    idx = np.arange(1, seq.size + 1)
    median = np.median(seq)
    above = seq > median
    ax.plot(idx, seq, "0.7", lw=0.7)
    ax.scatter(idx[above], seq[above], label="above the median")
    ax.scatter(idx[~above], seq[~above], color="r",
               label="below the median")
    ax.axhline(median, color="k", linestyle="--")
    lo, hi = res.bounds
    verdict = "trend-free" if res.trend_free else "rejected"
    ax.set_title(f"r = {res.statistic} runs, accept ({lo}, {hi}]: {verdict}")
    ax.set(xlabel="Sample index", ylabel="Sequence value")
    ax.legend()
plt.show()
```

</details>

The reverse arrangement test is the more powerful of the two against the
monotonic trends that dominate practice (B&P Sec. 4.5.2), which is why it
is the default everywhere; the runs test adds sensitivity to non-monotonic
clustering.

## 4. Level crossings and the apparent frequency

For a zero-mean Gaussian record with one-sided autospectrum $G(f)$, Rice's
classical results give the expected rate of zero crossings (both slopes,
B&P Eq. (5.195)) from the plain frequency moments
$m_k = \int f^k\,G(f)\,\mathrm{d}f$:

$$
N_0 = \frac{1}{\pi}\frac{\sigma_v}{\sigma_x} = 2\sqrt{\frac{m_2}{m_0}},
\qquad
N_a = N_0\, e^{-a^2/2\sigma_x^2},
$$

where $N_a$ is the crossing rate of level $a$ (Eq. (5.196)). $N_0/2$ is the
record's **apparent frequency**: a 60 Hz sine crosses zero 120 times per
second, low-pass noise of bandwidth $B$ gives $N_0 = 2B/\sqrt{3}$ (an
apparent frequency of $0.58B$, Example 5.12) and a band centred on $f_c$
gives $N_0 = 2\sqrt{f_c^2 + B^2/12}$ (Example 5.13).
`level_crossing_rate` counts the actual crossings of each level and puts
the Rice curve next to them, taking the moments from the record's own
[Welch autospectrum](spectral-analysis.md):

```python
import numpy as np
from phonometry import level_crossing_rate

fs = 8192.0
t = np.arange(1 << 16) / fs
x = np.sin(2 * np.pi * 60.0 * t)          # a 60 Hz sine ...

res = level_crossing_rate(x, fs)
print(round(res.zero_crossing_rate, 1))   # ... has 120 zeros per second
print(round(res.apparent_frequency, 1))   # 60.0
res.plot()
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/rice_level_crossings_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/rice_level_crossings.svg" alt="Measured level-crossing rates of a bandlimited Gaussian noise record between eight hundred and twelve hundred hertz, plotted as dots against the crossing level from minus three point five to plus three point five signal units on a logarithmic rate axis, falling from about two thousand crossings per second at level zero to a few per second at three sigma, with the Rice exponential curve passing through every measured point" width="82%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import level_crossing_rate

fs = 20480.0
n = 1 << 19
rng = np.random.default_rng(0)
freqs = np.fft.rfftfreq(n, 1 / fs)        # bandlimited Gaussian noise:
spec = rng.standard_normal(freqs.size) + 1j * rng.standard_normal(freqs.size)
spec[(freqs < 800.0) | (freqs > 1200.0)] = 0.0
x = np.fft.irfft(spec, n)

res = level_crossing_rate(x, fs, levels=np.linspace(-3.5, 3.5, 29) * np.std(x))

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(res.levels, res.rice_rates, label="Rice (Eq. 5.196)")
ax.plot(res.levels, res.rates, "o", label="Measured")
ax.set_yscale("log")
ax.set_xlabel("Level a [signal units]")
ax.set_ylabel("Crossings per second [1/s]")
ax.legend()
plt.show()
```

</details>

The Rice curve holds for Gaussian records, and that is precisely its second
use: measured rates that fall systematically off the curve are a quick
non-Gaussianity screen (B&P Sec. 5.5.1.1) - clipping shows up as missing
high-level crossings, impulsive contamination as an excess. Count-based
rates need the record comfortably oversampled: a crossing between two
samples of the same sign goes uncounted.

## 5. Peaks: rates, the irregularity factor and heights

The same moments fix the expected rate of local maxima
$M = (1/2\pi)(\sigma_a/\sigma_v) = \sqrt{m_4/m_2}$ (Eq. (5.211)) and with
it the dimensionless **irregularity factor**

$$
r = \frac{N_0}{2M} = \frac{m_2}{\sqrt{m_0\, m_4}} \in (0, 1],
$$

the single number that fixes the distribution of peak heights
(B&P Sec. 5.5.4). At $r = 1$ - narrow bandwidth data, one maximum per
zero-crossing cycle - peaks are **Rayleigh** distributed:
$\mathrm{Prob}[\text{peak} > a] = e^{-a^2/2\sigma_x^2}$ (Eq. (5.206)), which
is the one-in-3000 chance of a peak beyond $4\sigma$ of B&P Example 5.14.
As $r \to 0$ ever more ripples ride on each cycle, negative maxima appear,
and the peak heights approach the plain **Gaussian** amplitude distribution.
In between, Rice's mixture (Eqs. (5.217)/(5.223)) interpolates the two,
available as `peak_exceedance()` / `peak_density()` on the result:

```python
import numpy as np
from phonometry import peak_statistics

fs = 8192.0
t = np.arange(1 << 16) / fs
x = np.sin(2 * np.pi * 60.0 * t)          # narrowband: r is essentially 1

res = peak_statistics(x, fs)
print(round(res.irregularity_factor, 3))       # 0.997
print(res.peak_exceedance(4.0))                # exp(-8): about 1 in 3000
res.plot()   # empirical exceedance against the Rice mixture (figure below)
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/rice_peak_distribution_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/rice_peak_distribution.svg" alt="Peak-height exceedance probability of low-pass Gaussian noise on a logarithmic axis against the standardized peak height from minus two point five to four point five: the empirical staircase from half a million samples follows the Rice mixture curve for irregularity factor zero point seven four six, clearly below the dashed Rayleigh limit and above the dotted Gaussian limit, with a visible fraction of negative maxima" width="82%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import peak_statistics
from phonometry.metrology.data_qualification import _rice_peak_exceedance

fs = 20480.0
n = 1 << 19
rng = np.random.default_rng(3)
freqs = np.fft.rfftfreq(n, 1 / fs)        # low-pass noise: r = sqrt(5)/3
spec = rng.standard_normal(freqs.size) + 1j * rng.standard_normal(freqs.size)
spec[freqs > 2000.0] = 0.0
x = np.fft.irfft(spec, n)

res = peak_statistics(x, fs)
peaks = res.peak_values
empirical = 1.0 - np.arange(1, peaks.size + 1) / peaks.size
z = np.linspace(-2.5, 4.5, 400)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(z, _rice_peak_exceedance(z, 1.0), "--", label="Rayleigh (r = 1)")
ax.plot(z, _rice_peak_exceedance(z, 0.0), ":", label="Gaussian (r = 0)")
ax.plot(z, res.peak_exceedance(z),
        label=f"Rice (r = {res.irregularity_factor:.3f})")
ax.plot(peaks, empirical, drawstyle="steps-post", label="Empirical")
ax.set_yscale("log")
ax.set_ylim(1e-5, 1.5)
ax.set_xlabel(r"Standardized peak height $z = a/\sigma_x$")
ax.set_ylabel("Prob[peak > z]")
ax.legend()
plt.show()
```

</details>

For an ideal low-pass band the closed forms give $r = \sqrt{5}/3 = 0.745$,
and the measured value lands on it. The irregularity factor is the standard
bridge to fatigue and vibro-acoustic damage estimation, where it selects
the cycle-counting correction between the narrow-band Rayleigh assumption
and broad-band rainflow corrections. One practical warning: $m_4$ weights
the spectrum by $f^4$, so wideband instrumentation noise far above the
physical band silently inflates $M$ and deflates $r$ - band-limit the
record to the physically meaningful range first.

## Where this fits

Qualification comes *before* the statistics this section's other pages
compute: the chi-square confidence interval of a
[Welch PSD](spectral-analysis.md) and the random-error
formulas of the [correlation estimators](correlation-delay.md)
all assume the record is stationary, as does the very idea of *the* $L_{eq}$ of
a measurement in [Levels](levels.md). When a record fails
the test, split it at the change (the `segment_values` sequence shows
where), analyse the pieces, or move to the short-time views - the
[calibrated spectrogram](https://jmrplens.github.io/phonometry/guides/time-frequency/) - that do not
assume stationarity. And when it passes, the
[GUM machinery](gum-uncertainty.md) can propagate the
*remaining* random error of every averaged estimate with a clean
conscience.

## References

- Bendat, J. S., & Piersol, A. G. (2010). *Random Data: Analysis and
  Measurement Procedures* (4th ed.). Wiley. ISBN 978-0-470-24877-5.
  [doi:10.1002/9781118032428](https://doi.org/10.1002/9781118032428).
  Section 4.5.2 with Table A.6 (the nonparametric reverse arrangement trend
  test, its mean and variance, the tabulated percentage points and
  Example 4.4), Section 10.3 (data qualification; the segment mean-square
  stationarity procedure of 10.3.1.1 and Example 10.3) and Section 5.5
  (level crossings and peak values, Examples 5.12-5.15). The runs companion
  appeared in the third edition's percentage points of the run distribution.
- Wald, A., & Wolfowitz, J. (1940). On a test whether two samples are from
  the same population. *The Annals of Mathematical Statistics*, 11(2),
  147-162. [doi:10.1214/aoms/1177731909](https://doi.org/10.1214/aoms/1177731909).
  The exact conditional distribution of the number of runs.
- Rice, S. O. (1945). Mathematical analysis of random noise. *The Bell
  System Technical Journal*, 24(1), 46-156.
  [doi:10.1002/j.1538-7305.1945.tb00453.x](https://doi.org/10.1002/j.1538-7305.1945.tb00453.x).
  The original level-crossing and peak derivations (Parts I-II are in
  volume 23, 1944).
