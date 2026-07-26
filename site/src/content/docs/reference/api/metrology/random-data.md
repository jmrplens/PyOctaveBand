---
title: "metrology.random_data"
description: "Random-data qualification: stationarity tests and Rice crossing statistics."
sidebar:
  label: "random_data"
---

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

Random-data qualification: stationarity tests and Rice crossing statistics.

Before a record is averaged into a PSD, condensed into a Leq or fed to a GUM
budget, Bendat & Piersol (*Random Data*, 4th ed., Sec. 10.3) require it to be
**qualified**: classified as stationary or not, checked for periodicities and
validated against acquisition anomalies. This module implements the two
quantitative pillars of that chapter.

**Stationarity** (B&P Sec. 10.3.1.1). The record is divided into `N` equal
intervals, a mean square value (or another moment) is computed for each, and
the sequence is tested for underlying trends with a *nonparametric* test --
no assumption about the sampling distribution of the values is needed. The
book's test is the **reverse arrangement test** (Sec. 4.5.2): count the pairs
`i < j` with `x_i > x_j`. For an ordered sequence of independent
observations that count `A` has mean `N(N-1)/4` (Eq. (4.54)) and variance
`N(2N+5)(N-1)/72` (Eq. (4.55)), and the hypothesis of no trend is accepted
at significance `alpha` when `A` falls inside the two-sided acceptance
region of Table A.6 (for `N = 20` and `alpha = 0.05`: more than 64 and at
most 125 reverse arrangements, the book's Examples 4.4 and 10.3). The
classical companion **runs test** (the run distribution tabulated in the
third edition; Wald & Wolfowitz 1940) classifies each value as above or below
the sequence median and counts runs of like classification: a trend produces
too few runs, rapid alternation too many. Both tests are distribution-free
and, per Sec. 10.3.1.1, work equally well on mean values, rms values,
standard deviations or any other parameter sequence.

**Level crossings and peaks** (B&P Sec. 5.5, originally Rice). For a zero
mean value stationary Gaussian record with one-sided autospectrum `G(f)`,
the geometric moments

`sigma_x^2 = int G df`, `sigma_v^2 = int (2 pi f)^2 G df` and
`sigma_a^2 = int (2 pi f)^4 G df`

(Eqs. (5.214)-(5.216)) determine every crossing and peak statistic:

* the expected number of zero crossings per unit time (both slopes),
  `N0 = (1/pi) (sigma_v / sigma_x)` (Eq. (5.195)) -- equivalently
  `2 sqrt(m2 / m0)` with the plain frequency moments
  `mk = int f^k G df`; twice the record's *apparent frequency*;
* the expected number of crossings of level `a`,
  `Na = N0 exp(-a^2 / (2 sigma_x^2))` (Eq. (5.196));
* the expected number of maxima per unit time,
  `M = (1/2 pi) (sigma_a / sigma_v) = sqrt(m4 / m2)` (Eq. (5.211));
* the **irregularity factor** `r = N0 / (2 M)`, between 0 and 1
  (Eq. (5.220)): 1 for narrow bandwidth data (one maximum per zero-crossing
  cycle), toward 0 as ever more local maxima ride on each cycle;
* the peak probability functions: Rayleigh for narrow bandwidth data
  (Eqs. (5.206)-(5.207)), the standardized Gaussian for `r -> 0`
  (Eq. (5.221)) and, in between, Rice's mixture for the standardized peak
  height `z` (Eqs. (5.217) and (5.223) with `epsilon = sqrt(1 - r^2)`):

  `P[peak > z] = Q(z / epsilon)
  + r exp(-z^2 / 2) [1 - Q(r z / epsilon)]`

  where `Q` is the standardized normal exceedance (Eq. (5.250)).

[`stationarity_test`](/phonometry/reference/api/metrology/random-data/#stationarity_test) and [`trend_test`](/phonometry/reference/api/metrology/random-data/#trend_test) implement the first block,
[`level_crossing_rate`](/phonometry/reference/api/metrology/random-data/#level_crossing_rate) and [`peak_statistics`](/phonometry/reference/api/metrology/random-data/#peak_statistics) the second, each
comparing the counts measured on the record with the closed-form
expectations.

## level_crossing_rate

```python
level_crossing_rate(
    x: NDArray[np.float64] | list[float],
    fs: float,
    *,
    levels: NDArray[np.float64] | list[float] | None = None,
    nperseg: int | None = None,
) -> LevelCrossingResult
```

Level-crossing rates of a record and their Rice expectations.

Counts the crossings of each level `a` (both slopes, B&P Sec. 5.5.1)
and compares them with the closed forms for Gaussian data: the
zero-crossing rate `N0 = (1/pi)(sigma_v/sigma_x) = 2 sqrt(m2/m0)`
(Eq. (5.195)) and the level dependence
`Na = N0 exp(-a^2/(2 sigma_x^2))` (Eq. (5.196)), with the spectral
moments taken from the record's own Welch autospectrum. For low-pass
white noise of bandwidth `B` the expectation is
`N0 = 2 B / sqrt(3)` -- an apparent frequency of `0.58 B` (B&P
Example 5.12) -- and for bandwidth-limited white noise centred on
`fc`, `N0 = 2 sqrt(fc^2 + B^2/12)` (Example 5.13).

The record mean is removed first (the formulas hold for zero mean
value records); `levels` are then relative to that mean. Crossings
are counted as sign changes of the sampled record, so the sample rate
must comfortably oversample the signal bandwidth for the count not to
miss crossings between samples.

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Signal, 1-D. |
| `fs` | Sample rate, in Hz. |
| `levels` | Crossing levels in signal units about the mean (default: 13 levels evenly spaced over +-3 RMS). |
| `nperseg` | Welch segment length for the spectral moments (default: the [`power_spectral_density`](/phonometry/reference/api/spectra/spectra/#power_spectral_density) default). |

**Returns:** A [`LevelCrossingResult`](/phonometry/reference/api/metrology/random-data/#levelcrossingresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs or parameters are invalid. |

## LevelCrossingResult

```python
LevelCrossingResult(
    levels: NDArray[np.float64],
    rates: NDArray[np.float64],
    rice_rates: NDArray[np.float64],
    zero_crossing_rate: float,
    zero_crossing_rate_rice: float,
    apparent_frequency: float,
    sigma: float,
    duration: float,
    fs: float,
)
```

Measured level-crossing rates against the Rice expectation.

All rates count crossings with *both* slopes per unit time, following
B&P Sec. 5.5.1; the rate of zero crossings is twice the record's
apparent frequency. The Rice curve
`Na = N0 exp(-a^2 / (2 sigma^2))` (Eq. (5.196)) holds for Gaussian
records; systematic departures of the measured rates from it are
themselves a useful non-Gaussianity screen (B&P Sec. 5.5.1.1).

**Attributes**

| Name | Description |
| :--- | :--- |
| `levels` | Crossing levels `a`, in signal units (about the removed record mean). |
| `rates` | Measured crossing rates per level, in 1/s. |
| `rice_rates` | Rice expectation `N0 exp(-a^2/(2 sigma^2))`, 1/s. |
| `zero_crossing_rate` | Measured zero-crossing rate `N0`, in 1/s. |
| `zero_crossing_rate_rice` | Expected `N0 = 2 sqrt(m2/m0)` from the record's Welch autospectrum moments (Eq. (5.195)), in 1/s. |
| `apparent_frequency` | `N0 / 2` from the spectral moments, in Hz (a 60 Hz sine crosses zero 120 times per second). |
| `sigma` | RMS value of the demeaned record, in signal units. |
| `duration` | Record duration used for the rates, in seconds. |
| `fs` | Sample rate, in Hz. |

### LevelCrossingResult.plot()

```python
LevelCrossingResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot measured crossing rates against the Rice curve.

**Parameters**

| Name | Description |
| :--- | :--- |
| `language` | Label language, `"en"` (default) or `"es"`. |

## peak_statistics

```python
peak_statistics(
    x: NDArray[np.float64] | list[float],
    fs: float,
    *,
    nperseg: int | None = None,
) -> PeakStatisticsResult
```

Peak rate, irregularity factor and peak-height distribution of a record.

Detects the record's local maxima and compares their rate and height
distribution with the Rice closed forms for Gaussian data (B&P
Secs. 5.5.2 to 5.5.4): expected maxima rate `M = sqrt(m4/m2)`
(Eq. (5.211)), irregularity factor `r = N0/(2M)` (Eq. (5.220)) and
the standardized peak-height distribution that interpolates between
Rayleigh (narrow bandwidth, `r = 1`) and Gaussian (wide bandwidth,
`r -> 0`) via [`PeakStatisticsResult.peak_exceedance`](/phonometry/reference/api/metrology/random-data/#peakstatisticsresultpeak_exceedance). The
irregularity factor is the bridge to fatigue and vibro-acoustic
damage models, where it selects the cycle-counting correction.

The record mean is removed first. The `m4` moment weights the
autospectrum by `f^4`, so broadband instrumentation noise far above
the physical band inflates `M` and deflates `r`: band-limit the
record to the physically meaningful band first.

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Signal, 1-D. |
| `fs` | Sample rate, in Hz. |
| `nperseg` | Welch segment length for the spectral moments (default: the [`power_spectral_density`](/phonometry/reference/api/spectra/spectra/#power_spectral_density) default). |

**Returns:** A [`PeakStatisticsResult`](/phonometry/reference/api/metrology/random-data/#peakstatisticsresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs or parameters are invalid. |

## PeakStatisticsResult

```python
PeakStatisticsResult(
    peak_rate: float,
    peak_rate_rice: float,
    zero_crossing_rate_rice: float,
    irregularity_factor: float,
    peak_values: NDArray[np.float64],
    sigma: float,
    duration: float,
    fs: float,
)
```

Peak (maxima) statistics of a record against the Rice expectations.

"Positive peaks" are the record's local maxima, positive or negative
in value (B&P Sec. 5.5.3). For Gaussian records the expected rate of
maxima is `M = (1/2 pi)(sigma_a/sigma_v) = sqrt(m4/m2)`
(Eq. (5.211)), and the ratio `r = N0/(2M)` -- the **irregularity
factor**, between 0 and 1 by the Schwartz inequality (Eq. (5.220)) --
fixes the standardized peak-height distribution: Rayleigh at
`r = 1` (every cycle carries one maximum), Gaussian as `r -> 0`
(Sec. 5.5.4).

**Attributes**

| Name | Description |
| :--- | :--- |
| `peak_rate` | Measured rate of local maxima, in 1/s. |
| `peak_rate_rice` | Expected rate `M = sqrt(m4/m2)` from the Welch autospectrum moments, in 1/s. |
| `zero_crossing_rate_rice` | Expected `N0 = 2 sqrt(m2/m0)`, 1/s. |
| `irregularity_factor` | `N0 / (2 M)` from the spectral moments. |
| `peak_values` | Standardized heights `z = peak / sigma` of the detected maxima. |
| `sigma` | RMS value of the demeaned record, in signal units. |
| `duration` | Record duration used for the rates, in seconds. |
| `fs` | Sample rate, in Hz. |

### PeakStatisticsResult.peak_density()

```python
PeakStatisticsResult.peak_density(
    z: NDArray[np.float64] | list[float] | float,
) -> NDArray[np.float64]
```

Peak probability density `w(z)` (B&P Eq. (5.217)).

**Parameters**

| Name | Description |
| :--- | :--- |
| `z` | Standardized peak heights (units of the record RMS). |

**Returns:** Probability density values, same shape.

### PeakStatisticsResult.peak_exceedance()

```python
PeakStatisticsResult.peak_exceedance(
    z: NDArray[np.float64] | list[float] | float,
) -> NDArray[np.float64]
```

`P[peak > z]` at the record's irregularity factor.

B&P Eq. (5.223): the probability that a maximum chosen at random
exceeds `z` record RMS units. `exp(-z^2/2)` for narrow
bandwidth data -- `exp(-8) = 0.00033` at `z = 4`, B&P
Example 5.14 -- and the Gaussian exceedance in the wide bandwidth
limit.

**Parameters**

| Name | Description |
| :--- | :--- |
| `z` | Standardized peak heights (units of the record RMS). |

**Returns:** Exceedance probabilities, same shape.

### PeakStatisticsResult.plot()

```python
PeakStatisticsResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the empirical peak exceedance against the Rice curves.

**Parameters**

| Name | Description |
| :--- | :--- |
| `language` | Label language, `"en"` (default) or `"es"`. |

## stationarity_test

```python
stationarity_test(
    x: NDArray[np.float64] | list[float],
    fs: float,
    *,
    n_segments: int = 20,
    statistic: str = 'mean_square',
    method: str = 'reverse_arrangements',
    alpha: float = 0.05,
) -> StationarityTestResult
```

Test a record for stationarity via trends in segment statistics.

The B&P Sec. 10.3.1.1 procedure: (1) divide the record into
`n_segments` equal intervals, assumed long enough for the values to
be independent; (2) compute a mean square value (or another moment)
for each interval; (3) test the sequence for underlying trends with a
nonparametric test -- no knowledge of the record's bandwidth or
sampling distribution required. Nonstationarities that show in the
mean square value (the usual case: gain drifts, level ramps, machinery
running up) are detected regardless of the record's spectrum; the
default 20 segments matches the book's worked Example 10.3, which
rejects stationarity for a noise record with a 20 % gain ramp
(`A = 52`, below the Table A.6 lower bound of 64).

Any trailing samples that do not fill the last segment are discarded.
The segment count trades resolution against independence: each
segment must remain long against the record's lowest frequencies.

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Signal, 1-D. |
| `fs` | Sample rate, in Hz. |
| `n_segments` | Number of equal intervals (default 20, as in B&P Example 10.3); at least 10, at most the record length in samples. |
| `statistic` | Per-segment statistic: `"mean_square"` (default, the book's choice), `"rms"`, `"mean"` or `"variance"`. |
| `method` | Trend test: `"reverse_arrangements"` (default) or `"runs"`. |
| `alpha` | Two-sided significance level (default 0.05). |

**Returns:** A [`StationarityTestResult`](/phonometry/reference/api/metrology/random-data/#stationaritytestresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs or parameters are invalid. |

## StationarityTestResult

```python
StationarityTestResult(
    segment_values: NDArray[np.float64],
    segment_times: NDArray[np.float64],
    statistic: str,
    method: str,
    count: int,
    mean: float,
    std: float,
    bounds: tuple[int, int],
    p_value: float,
    stationary: bool,
    alpha: float,
    n_segments: int,
    segment_duration: float,
    fs: float,
)
```

A stationarity test on segment statistics of a single record.

The record was divided into `n_segments` equal intervals, the
per-segment `statistic` sequence was computed, and the sequence
was tested for trends with [`trend_test`](/phonometry/reference/api/metrology/random-data/#trend_test) (the B&P Sec. 10.3.1.1
procedure). The hypothesis of stationarity is accepted when
`bounds[0] < count <= bounds[1]`.

**Attributes**

| Name | Description |
| :--- | :--- |
| `segment_values` | Per-segment values of the tested statistic. |
| `segment_times` | Segment centre times, in seconds. |
| `statistic` | The per-segment statistic: `"mean_square"`, `"rms"`, `"mean"` or `"variance"`. |
| `method` | Trend test used: `"reverse_arrangements"` or `"runs"`. |
| `count` | Observed test statistic (reverse arrangements or runs). |
| `mean` | Null mean of the count. |
| `std` | Null standard deviation of the count. |
| `bounds` | Acceptance region `(lower, upper)` at `alpha`. |
| `p_value` | Two-sided p-value of the observed count. |
| `stationary` | `True` when the count falls inside the region. |
| `alpha` | Significance level (default 0.05). |
| `n_segments` | Number of segments the record was divided into. |
| `segment_duration` | Duration of each segment, in seconds. |
| `fs` | Sample rate of the record, in Hz. |

### StationarityTestResult.plot()

```python
StationarityTestResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the segment-statistic sequence with the test verdict.

**Parameters**

| Name | Description |
| :--- | :--- |
| `language` | Label language, `"en"` (default) or `"es"`. |

## trend_test

```python
trend_test(
    values: NDArray[np.float64] | list[float],
    *,
    method: str = 'reverse_arrangements',
    alpha: float = 0.05,
) -> TrendTestResult
```

Nonparametric trend test on a sequence of observations or estimates.

Tests the hypothesis that the sequence holds independent observations
of one random variable -- no underlying trend -- without assuming any
sampling distribution (B&P Sec. 4.5.2). Two classical statistics:

* `"reverse_arrangements"` (default): the count `A` of pairs
  `i < j` with `x_i > x_j`. A downward trend inflates `A`, an
  upward trend depresses it. The acceptance region reproduces B&P
  Table A.6 (`alpha = 0.05`, `N` = 10 to 100); the p-value uses
  the exact Mahonian null distribution up to `N = 100`.
* `"runs"`: values are classified against the sequence median and
  runs of like classification are counted (the run test of B&P's
  earlier editions; Wald & Wolfowitz 1940). Trends and slow drifts
  produce too few runs, alternation too many. Exact conditional
  distribution at any `N`; values equal to the median are discarded.

The reverse arrangement test is the more powerful of the two against
monotonic trends (B&P Sec. 4.5.2); the runs test also reacts to
non-monotonic clustering.

**Parameters**

| Name | Description |
| :--- | :--- |
| `values` | Sequence of observations or parameter estimates (e.g. segment mean square values), at least 10. |
| `method` | `"reverse_arrangements"` (default) or `"runs"`. |
| `alpha` | Two-sided significance level (default 0.05, the level tabulated by B&P). |

**Returns:** A [`TrendTestResult`](/phonometry/reference/api/metrology/random-data/#trendtestresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs or parameters are invalid. |

## TrendTestResult

```python
TrendTestResult(
    values: NDArray[np.float64],
    method: str,
    statistic: int,
    n: int,
    mean: float,
    std: float,
    bounds: tuple[int, int],
    p_value: float,
    trend_free: bool,
    alpha: float,
    median: float | None = None,
)
```

A nonparametric trend test on a sequence of parameter estimates.

The hypothesis of *no underlying trend* is accepted at significance
`alpha` when `statistic` lies inside the two-sided
acceptance region `lower < statistic <= upper` (B&P Sec. 4.5.2 and
Table A.6 for reverse arrangements); `p_value` is the exact
two-sided tail probability of the observed count.

**Attributes**

| Name | Description |
| :--- | :--- |
| `values` | The tested sequence (for `"runs"`, after discarding values exactly equal to the median). |
| `method` | `"reverse_arrangements"` or `"runs"`. |
| `statistic` | Observed count: reverse arrangements `A` or runs `r`. |
| `n` | Number of observations used. |
| `mean` | Null mean of the statistic (B&P Eq. (4.54) for `A`). |
| `std` | Null standard deviation (B&P Eq. (4.55) for `A`). |
| `bounds` | Acceptance region `(lower, upper)`: percentage points such that the no-trend hypothesis is accepted when `lower < statistic <= upper`. For reverse arrangements these follow B&P Table A.6's own convention (normal approximation with continuity correction, which reproduces the book's tabulated `alpha = 0.05` entries exactly; the table is not derivable from the exact Mahonian distribution), so at an acceptance boundary the verdict and the exact `p_value` can disagree by one count. |
| `p_value` | Two-sided p-value from the exact null distribution (normal approximation above `n = 100` for reverse arrangements). |
| `trend_free` | `True` when the statistic falls inside the acceptance region. |
| `alpha` | Significance level of the region (default 0.05). |
| `median` | For `"runs"`, the median of the *original* sequence against which each value was classified (before values equal to it were discarded); `None` for `"reverse_arrangements"`. |

### TrendTestResult.plot()

```python
TrendTestResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the tested sequence against its sample index with the verdict.

Draws the sequence of observations against a plain sample index and
states the outcome in the legend: the reverse-arrangement count
`A` (or the run count `r`), the acceptance region and whether
the no-trend hypothesis is accepted. For the runs test the
classification median is drawn as a reference line.

**Parameters**

| Name | Description |
| :--- | :--- |
| `ax` | Existing `Axes` to draw on, or `None` (default) to create a fresh figure and axes. |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | Extra keyword arguments forwarded to the sequence `plot` call (e.g. `color`, `lw`, `marker`). |

**Returns:** The `Axes` the sequence was drawn on, so the figure can be composed further.
