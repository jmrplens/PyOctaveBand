---
title: "filters.weighting"
description: "Weighting filters (A, B, C, D, G, AU, 468, Z), time weighting utilities and the Linkwitz-Riley crossover."
sidebar:
  label: "weighting"
---

Weighting filters (A, B, C, D, G, AU, 468, Z), time weighting utilities
and the Linkwitz-Riley crossover.

A/C/Z per IEC 61672-1:2013; G (infrasound) per ISO 7196:1995.

B is the historical weighting of ANSI S1.4-1983 (Appendix C): the C curve
with one extra zero at the origin and one extra real pole at
$f_5 = 158.48932$ Hz. It was dropped from the sound-level-meter
standards
when IEC 61672-1 replaced IEC 60651 (first edition 2002) and is provided for
historical data and older national codes only.

AU per IEC 61012:1990: the A weighting cascaded with the U low-pass
(six poles, Table 2: a double real pole at -12 200 Hz and complex pairs at
-7 850 +/- j8 800 Hz and -2 900 +/- j12 150 Hz) for measuring audible sound
in the presence of ultrasound. It is flat relative to A up to 10 kHz and
cuts steeply above (U alone, Table 1: -2.8 dB at 12.5 kHz; -61.8 dB at
40 kHz). The Table 2 poles reproduce every Table 1 nominal value within
0.05 dB.

D per the withdrawn IEC 537:1976 (aircraft-noise weighting): implemented
from the widely published rational transfer function

$$
\frac{k s \left( s^2 + 6532 s + 4.0975 \times 10^7 \right)} {(s + 1776.3)(s + 7288.5) \left( s^2 + 21514 s + 3.8836 \times 10^8 \right)}
$$

with `k` renormalized to exactly 0 dB at
1 kHz. The standard itself is withdrawn and unavailable, so the constants
are corroborated against two independent implementations: SQAT
(`sound_level_meter/Gen_weighting_filters.m`: identical zeros and poles;
note its display-only `freqResp` line prints 1773.6 where its pole list,
and every other source, has 1776.3) and librosa (`librosa.D_weighting`,
an independent frequency-domain closed form; agreement within 0.002 dB
from 10 Hz to 20 kHz). The response also reproduces the tabulated IEC 537
curve republished in the NASA Handbook of Aircraft Noise Metrics
(NASA CR-3406, 1981, Table SLD-I) within 0.1 dB at every one-third-octave
frequency from 50 Hz to 10 kHz except 1600 Hz (0.15 dB) and 2500 Hz
(0.28 dB), where that table appears to round a different source curve.

468 per ITU-R BS.468-4, the psophometric weighting for audio-frequency
noise in sound broadcasting: a bandpass peaking at +12.22 dB near 6.3 kHz
and falling at about -30 dB/octave above 12.5 kHz, which is what makes
broadband noise audible in a programme chain rather than what makes it
loud. Clause 1 defines it as the response of the passive network of Fig. 1a,
so it is built from that network's seven printed component values
(`_itu_r_468_prototype`) and reproduces all 21 rows of the Table 1
sampling to 0.0503 dB. Its skirt is steep enough that the plain bilinear
design at the input rate reads 23 dB out at 16 kHz, so `high_accuracy=False`
is refused for this curve rather than shipped: the Recommendation prints one
mask and no lower grade to fall back to.

**How the prototypes become filters.** Every curve above is a set of poles and
zeros in the s plane, and the bilinear transform that turns those into a
digital filter is exact in magnitude but wrong in frequency: it puts the
prototype's response at `2 f_s tan(pi f / f_s)` instead of at `2 pi f`,
which costs 0.86 dB at 20 kHz for A at 48 kHz and 61 dB at 15 848.9 Hz when
fs = 32 kHz. The library used to hide that by interpolating, filtering at
three to eight times the rate and decimating back. It no longer does: the
prototype is fitted at the sample rate instead
(`phonometry.filters._weighting_design`), and what runs is one cascade of
second-order sections. Three consequences worth stating in one place, because
each of them used to be a documented limitation of this module:

* the anti-alias filter of those resampling stages had its transition band on
  the input Nyquist frequency and the signal crossed it twice, which put a
  floor of about 1.7 dB on everything above 0.9 of Nyquist whatever the design
  rate. That floor is gone, so the rows near Nyquist -- the 15 848.9 Hz row at
  32 kHz, the 20 kHz row of BS.468-4 Table 1 at 44.1 kHz -- are inside their
  masks instead of over them, and A and C verify to class 1 at every sample
  rate from 8 kHz up;
* block processing was incompatible with the accurate design, because a
  resampler cannot be driven block by block. It no longer is: `stateful` and
  `high_accuracy` are independent, and stitched blocks reproduce a single
  call exactly; and
* one minute of 44.1 kHz audio costs about 18 ms instead of 377 ms for A and
  775 ms for 468, and holds 21 MB of intermediates instead of 169 MB
  (measured back to back in one process, mean of five runs). The design
  itself costs about 200 ms and is cached, so even a first call is quicker
  than the path it replaces: about 215 ms against 377 for A, 220 against
  775 for 468.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## linkwitz_riley

```python
linkwitz_riley(
    x: Signal,
    fs: int | None = ...,
    *,
    freq: float,
    order: int = ...,
) -> tuple[Signal, Signal]

linkwitz_riley(
    x: list[float] | np.ndarray,
    fs: int,
    *,
    freq: float,
    order: int = ...,
) -> tuple[np.ndarray, np.ndarray]
```

Linkwitz-Riley crossover filter (Butterworth squared).
Splits signal into low and high bands with flat sum response.

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Input signal, or a [`phonometry.io.Signal`](/phonometry/reference/api/io/io/#signal) read from a measurement file. A calibrated Signal is split in pascals, so both bands come back in pascals. |
| `fs` | Sample rate. Required for a bare array; a [`Signal`](/phonometry/reference/api/io/io/#signal) brings its own, and an explicit value that disagrees with it raises. |
| `freq` | Crossover frequency, in Hz. Keyword-only and required: it sits behind an optional `fs`, and a default here would be a signature that lies about what the call needs. |
| `order` | Total order (must be even, typically 2 or 4). |

**Returns:** (low_pass_signal, high_pass_signal)

## time_weighting

```python
time_weighting(
    x: Signal,
    fs: int | None = ...,
    mode: str = ...,
    initial_state: str | float | np.ndarray | None = ...,
) -> TimeWeightedEnvelope

time_weighting(
    x: list[float] | np.ndarray,
    fs: int,
    mode: str = ...,
    initial_state: str | float | np.ndarray | None = ...,
) -> np.ndarray
```

Apply time weighting to a signal (Exponential averaging).

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Input signal (raw pressure/voltage), or a [`phonometry.io.Signal`](/phonometry/reference/api/io/io/#signal) read from a measurement file. The function squares it internally, so a calibrated Signal yields a mean-square envelope in Pa2 rather than in digital units squared. |
| `fs` | Sample rate. Required for a bare array; a [`Signal`](/phonometry/reference/api/io/io/#signal) brings its own, and an explicit value that disagrees with it raises. |
| `mode` | 'fast' (125ms), 'slow' (1000ms), 'impulse' (35ms rise, 1500ms fall). |
| `initial_state` | Previous mean-square output state `y[-1]`. Use None/'zero' for zero initialization (default), 'first' to initialize from the first input energy, or a scalar/array broadcastable to the input shape without the time axis. |

**Returns:** The time-weighted mean square. A bare array in gives a bare array back; a [`Signal`](/phonometry/reference/api/io/io/#signal) gives a [`TimeWeightedEnvelope`](/phonometry/reference/api/filters/weighting/#timeweightedenvelope), which stands in for that array everywhere it was used and adds the rate and a level plot. It is not a Signal, because a mean square is not a pressure record.

## TimeWeightedEnvelope

```python
TimeWeightedEnvelope(
    mean_square: np.ndarray,
    fs: int,
    mode: str,
    calibrated: bool,
)
```

The exponentially averaged mean square of a record, and its rate.

What [`time_weighting`](/phonometry/reference/api/filters/weighting/#time_weighting) computes is not a waveform: it is the
running mean SQUARE, in pascals squared when the record was calibrated,
which is why it cannot come back as a
[`Signal`](/phonometry/reference/api/io/io/#signal). That class means a record of pressure,
and labelling a squared quantity as one would be the kind of quiet lie
the calibration contract exists to prevent.

What it needs instead is the rate, so the envelope can be read against a
time axis, and a plot that knows the trace is a level. That is this
object. It stands in for the bare array it replaced everywhere the array
was used: `numpy.asarray`, `len()`, indexing and the
`shape`/`ndim`/`size`/`dtype` attributes all forward to the
envelope, so a caller that only wanted the numbers never notices.

**Attributes**

| Name | Description |
| :--- | :--- |
| `mean_square` | The weighted mean square, `(channels, samples)` or 1-D for one channel, in Pa2 when the record was calibrated. |
| `fs` | Sample rate, in Hz. |
| `mode` | The weighting used: `"fast"`, `"slow"` or `"impulse"`. |
| `calibrated` | Whether the samples that produced it were in pascals, which is what decides whether a level read off it means dB SPL. |

### TimeWeightedEnvelope.dtype

*property*

Data type of the envelope.

### TimeWeightedEnvelope.ndim

*property*

Number of dimensions of the envelope.

### TimeWeightedEnvelope.plot()

```python
TimeWeightedEnvelope.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the level trace this envelope stands for.

Draws `10 lg(mean square / p0^2)` against time. That is a
time-weighted sound pressure level, and it is `L_pAF` only when
the record was A-weighted before it got here: this function applies
the time weighting and nothing else. Needs a calibrated record to
mean dB SPL, and says so rather than drawing a number counted from
nothing.

### TimeWeightedEnvelope.shape

*property*

Shape of the envelope.

### TimeWeightedEnvelope.size

*property*

Number of values in the envelope.

### TimeWeightedEnvelope.times

*property*

Sample times, in seconds from the start of the record.

## TimeWeighting

```python
TimeWeighting(fs: int, mode: str = 'fast')
```

Stateful time weighting for block processing.

Wraps [`time_weighting`](/phonometry/reference/api/filters/weighting/#time_weighting) carrying the exponential integrator state
across blocks, so concatenated block outputs equal a single continuous call.

**Parameters**

| Name | Description |
| :--- | :--- |
| `fs` | Sample rate in Hz. |
| `mode` | 'fast' (125 ms), 'slow' (1000 ms) or 'impulse' (35 ms / 1.5 s). |

### TimeWeighting.process()

```python
TimeWeighting.process(
    x: Signal | list[float] | np.ndarray,
) -> TimeWeightedEnvelope | np.ndarray
```

Apply time weighting to a block, continuing from the previous block.

The block form of [`time_weighting`](/phonometry/reference/api/filters/weighting/#time_weighting), and it returns the same
thing on the same terms: a mean square, in pascals squared when the
record was calibrated, wrapped in a
[`TimeWeightedEnvelope`](/phonometry/reference/api/filters/weighting/#timeweightedenvelope) when the block arrived as a
[`Signal`](/phonometry/reference/api/io/io/#signal) and left as a bare array otherwise.
The envelope stands in for that array, so a loop that concatenates
the blocks keeps working either way.

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | The block, or a [`phonometry.io.Signal`](/phonometry/reference/api/io/io/#signal). A Signal at another rate than this integrator was built for is refused; a calibrated one is squared in pascals, exactly as [`time_weighting`](/phonometry/reference/api/filters/weighting/#time_weighting) does. |

**Returns:** Time-weighted mean-square envelope of the block.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a Signal's rate is not this integrator's. |

### TimeWeighting.reset()

```python
TimeWeighting.reset() -> None
```

Forget the carried state (the next block starts from rest).

## weighting_filter

```python
weighting_filter(
    x: Signal,
    fs: int | None = ...,
    curve: str = ...,
    high_accuracy: bool = ...,
) -> Signal

weighting_filter(
    x: list[float] | np.ndarray,
    fs: int,
    curve: str = ...,
    high_accuracy: bool = ...,
) -> np.ndarray
```

Apply a frequency weighting to a signal.

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Input signal, or a [`phonometry.io.Signal`](/phonometry/reference/api/io/io/#signal) read from a measurement file. A calibrated Signal is weighted in pascals, so the weighted samples come back in pascals too; a bare array keeps whatever unit it arrived in. |
| `fs` | Sample rate. Required for a bare array; a [`Signal`](/phonometry/reference/api/io/io/#signal) brings its own, and an explicit value that disagrees with it raises instead of silently winning. |
| `curve` | 'A', 'C' (IEC 61672-1), 'B' (ANSI S1.4-1983, historical), 'D' (withdrawn IEC 537 aircraft-noise weighting), 'G' (ISO 7196 infrasound), 'AU' (IEC 61012), '468' (ITU-R BS.468-4 psophometric noise weighting) or 'Z' (bypass). |
| `high_accuracy` | Design the filter by fitting the analog prototype at *fs* (default True), rather than by the plain bilinear transform, which warps frequency and loses class 1 below 44.1 kHz. See [`WeightingFilter.__init__`](/phonometry/reference/api/filters/weighting/#weightingfilter) for what each design costs. The '468' curve requires the fitted design and refuses False. |

**Returns:** The weighted record. A bare array in gives a bare array back; a [`Signal`](/phonometry/reference/api/io/io/#signal) gives a Signal, whose samples are already in pascals and whose factor therefore reads 1.0.

## WeightingFilter

```python
WeightingFilter(
    fs: int,
    curve: str = 'A',
    stateful: bool = False,
    steady_ic: bool = False,
    high_accuracy: bool | None = None,
)
```

Class-based frequency weighting filter (A, B, C, D, G, AU, 468, Z).
Allows pre-calculating and reusing filter coefficients.

Initialize the weighting filter.

**Parameters**

| Name | Description |
| :--- | :--- |
| `fs` | Sample rate in Hz. |
| `curve` | 'A', 'C' (IEC 61672-1), 'B' (ANSI S1.4-1983, historical: removed from the IEC sound-level-meter standards), 'D' (withdrawn IEC 537 aircraft-noise weighting), 'G' (ISO 7196 infrasound), 'AU' (IEC 61012, audible sound in the presence of ultrasound), '468' (ITU-R BS.468-4 psophometric noise weighting, see `_itu_r_468_prototype`) or 'Z'. |
| `stateful` | If True, carry the section state between calls, so concatenated blocks equal one continuous call. Available for every curve and independent of `high_accuracy`: both designs are a plain cascade of second-order sections at the input rate, and `sosfilt` with a carried `zi` reproduces a single call bit for bit. |
| `steady_ic` | If True, calculate steady state initial conditions for filter. |
| `high_accuracy` | Which of the two designs to build, both of them second-order sections at the input rate. `True` (the default) fits the analog prototype at *fs*, undoing the bilinear frequency warping instead of tolerating it (`phonometry.filters._weighting_design`). Measured against the prototype over the standard's own band, the worst deviation is 0.008 dB for A at 32 kHz and 0.0003 dB at 48 kHz, and at most 0.06 dB for any curve at any rate in this corpus -- the 0.06 belonging to the 468 curve at 44.1 and 48 kHz, where a -30 dB/octave skirt has to flatten inside the last half percent below the Nyquist frequency. A and C verify to class 1 at every sample rate from 8 kHz up, and at every Table 3 row their deviation stays inside the 0.05 dB the table itself is rounded to. The design is fitted over the interval `_fit_band` returns and is not claimed outside it. `False` is the plain bilinear transform of the printed prototype: the closed-form design a reader can check against the standard term by term, at the cost of the warping. That cost grows quadratically toward the Nyquist frequency -- for A at 48 kHz it reads 15.7 dB below the design goal at the 19 952.6 Hz row, which the class 1 mask does not see because its lower limit there is -inf, and 61.4 dB below it at 15 848.9 Hz when fs = 32 kHz, which it does. So it verifies to class 1 for fs >= 44 100 Hz, degrades to class 2 at 32 000 and 22 050 Hz, and meets no class at 16 000 Hz. It is refused for the '468' curve, whose skirt puts it 23 dB out at 16 kHz with no lower grade in the Recommendation to fall back to. Defaults to True (`None` selects the default too, which is what it used to mean in every mode but the stateful one). |

### WeightingFilter.filter()

```python
WeightingFilter.filter(
    x: Signal | list[float] | np.ndarray,
) -> Signal | np.ndarray
```

Apply the weighting filter to a signal.

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Input signal (1D or 2D [channels, samples]), or a [`phonometry.io.Signal`](/phonometry/reference/api/io/io/#signal). A Signal recorded at another rate than this filter was designed for is refused rather than weighted by the wrong response; a calibrated one is weighted in pascals, exactly as [`weighting_filter`](/phonometry/reference/api/filters/weighting/#weighting_filter) does, so the two entry points cannot disagree about the same recording. |

**Returns:** The weighted record. A bare array in gives a bare array back; a [`Signal`](/phonometry/reference/api/io/io/#signal) gives a Signal, on the same terms as [`weighting_filter`](/phonometry/reference/api/filters/weighting/#weighting_filter), so the object and the function cannot disagree about the same recording.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a Signal's rate is not this filter's. |
