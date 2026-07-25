---
title: "metrology.signals"
description: "Public API of phonometry.metrology.signals (auto-generated)."
sidebar:
  label: "signals"
---

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

Test signals and sample-rate utilities.

The signal toolbox of the metrology domain: deterministic test signals and
the two sample-rate operations every measurement chain eventually needs,
with their accuracy stated instead of implied.

* [`noise_signal`](/phonometry/reference/api/spectra/signals/#noise_signal) - Gaussian noise with an exact power-law spectral
  slope: white (0 dB/octave), pink (-3.01), red (-6.02, also called
  Brownian), blue (+3.01) and violet (+6.02). The autospectral density
  follows `Gxx(f) ∝ f^α` with `α` = 0, -1, -2, +1 and +2 respectively,
  so the level changes by exactly `3.01·α` dB per octave
  (`10·lg 2 = 3.0103` dB). The colors are synthesized by filtering seeded
  white Gaussian noise in the frequency domain: the DFT of the white record
  is multiplied by the exact magnitude response `|H(f)| = (f/f_ref)^(α/2)`
  bin by bin (a zero-phase FIR filter applied circularly), so the *expected*
  spectrum follows the power law exactly at every synthesis bin above DC and
  a measured slope deviates only by the random error of the spectral
  estimate - not the piecewise or few-pole approximations whose pink slope
  ripples by fractions of a dB. The DC bin is zeroed for the colored
  variants (a power law has no finite DC value) and the record is rescaled
  to the requested RMS exactly. With the same `seed` the generator is
  fully deterministic across runs.

* [`tone_burst`](/phonometry/reference/api/spectra/signals/#tone_burst) - the gated sine burst of IEC 60268-1:1985 (Annex A,
  Clause A2): the tone starts at a zero crossing and lasts an integral
  number of full periods, either as a single burst or as a repetitive train
  with a stated repetition rate. The result records the rectangular gating
  envelope and the exact on/off sample bookkeeping, so meter ballistics and
  dynamic-response tests can state their stimulus instead of hand-rolling
  it.

* [`resample_signal`](/phonometry/reference/api/spectra/signals/#resample_signal) - polyphase resampling behind an explicit
  anti-alias specification. The lowpass FIR is designed here (Kaiser
  window method) from two numbers the caller controls - the stopband
  attenuation in dB and the transition-band fraction of the target
  Nyquist - and the designed filter is returned with the result, so the
  alias rejection of a resampled record is a documented property, not a
  library default.

* [`fractional_delay`](/phonometry/reference/api/spectra/signals/#fractional_delay) - band-limited delay by an arbitrary
  (sub-sample) number of samples via a frequency-domain phase ramp,
  `linear` (zero-padded, for transients and impulse responses; the same
  kernel [`align_impulse_responses`](/phonometry/reference/api/correlation/correlation/#align_impulse_responses)
  uses) or `circular` (for periodic records, exact to machine precision
  on bin-centered tones).

## fractional_delay

```python
fractional_delay(
    x: NDArray[np.float64] | list[float],
    delay: float,
    *,
    mode: Literal['linear', 'circular'] = 'linear',
) -> NDArray[np.float64]
```

Delay a record by an arbitrary (sub-sample) number of samples.

Band-limited delay via a frequency-domain phase ramp
`e^{-j2πk·delay/N}`: every spectral component is delayed by exactly
`delay` samples, i.e. its phase changes by `-2π·f·delay/fs`
radians. Two boundary conventions:

* `'linear'` (default): the record is zero-padded past the shift
  before the ramp, so samples leaving one end land in the padding
  instead of wrapping around - use it for transients and impulse
  responses. Content shifted beyond the record length is discarded
  (the output keeps the input length).
* `'circular'`: the ramp is applied over the record itself and the
  shift wraps around - use it for periodic records. For a tone
  centered on a DFT bin the delayed record equals the analytically
  delayed tone to machine precision.

An integer `delay` in `'linear'` mode reduces to an exact sample
shift with zero fill. Negative delays advance the record.

A real record of even length cannot carry a fractionally delayed
Nyquist-bin component (the inverse real FFT keeps its real part), so
keep the signal band-limited below Nyquist - as any properly sampled
signal is - or use odd lengths, and the operation is exact.

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Input record, 1-D. |
| `delay` | Delay in samples (fractional and negative allowed); magnitude less than the record length. |
| `mode` | Boundary convention, `'linear'` or `'circular'`. |

**Returns:** The delayed record, same length as `x`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs or parameters are invalid. |

## noise_signal

```python
noise_signal(
    fs: float,
    seconds: float = 1.0,
    *,
    color: Literal['white', 'pink', 'red', 'blue', 'violet'] = 'white',
    rms: float = 1.0,
    seed: int | None = None,
) -> NDArray[np.float64]
```

Generate Gaussian noise with an exact power-law spectral slope.

`Gxx(f) ∝ f^α` with α = 0 (white), -1 (pink, -3.01 dB/octave),
-2 (red/Brownian, -6.02), +1 (blue, +3.01) or +2 (violet, +6.02),
shaped by an exact frequency-domain filter (see the module docstring),
zero-mean and rescaled to the requested RMS exactly.

**Parameters**

| Name | Description |
| :--- | :--- |
| `fs` | Sample rate, in Hz. |
| `seconds` | Duration, in seconds (at least 16 samples). |
| `color` | Noise color: `'white'`, `'pink'`, `'red'`, `'blue'` or `'violet'`. |
| `rms` | Root-mean-square value of the returned record. |
| `seed` | Seed for `numpy.random.default_rng`; the same seed reproduces the same record. `None` draws fresh entropy. |

**Returns:** The noise record, `round(fs·seconds)` samples.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs or parameters are invalid. |

## resample_signal

```python
resample_signal(
    x: NDArray[np.float64] | list[float],
    fs: float,
    fs_new: float,
    *,
    stopband_attenuation_db: float = 120.0,
    transition_width: float = 0.05,
    max_denominator: int = 1000,
) -> ResampledSignalResult
```

Resample a record with a stated anti-alias specification.

Polyphase rational resampling (`scipy.signal.resample_poly`)
behind a lowpass FIR designed *here* by the Kaiser window method: the
stopband starts at the smaller of the two Nyquist frequencies and
provides `stopband_attenuation_db` of alias rejection, the passband
ends `transition_width` below it and is flat within the same ripple
bound `δ = 10^(-A/20)`. The designed taps travel with the result, so
the spec is a property of the returned filter, not of a library
default.

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Input record, 1-D. |
| `fs` | Sample rate of `x`, in Hz. |
| `fs_new` | Target sample rate, in Hz. The ratio `fs_new/fs` must be a rational number with denominator at most `max_denominator` (e.g. 48000/44100 = 160/147). |
| `stopband_attenuation_db` | Anti-alias stopband attenuation, in dB (at least 30). |
| `transition_width` | Transition-band width as a fraction of the smaller Nyquist frequency, in (0, 0.5]. |
| `max_denominator` | Largest denominator accepted for the rational rate ratio. |

**Returns:** A [`ResampledSignalResult`](/phonometry/reference/api/spectra/signals/#resampledsignalresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs or parameters are invalid, or if the rate ratio is not rational within `max_denominator`. |

## ResampledSignalResult

```python
ResampledSignalResult(
    signal: NDArray[np.float64],
    fs: float,
    original_fs: float,
    up: int,
    down: int,
    filter_taps: NDArray[np.float64],
    passband_edge_hz: float,
    stopband_edge_hz: float,
    stopband_attenuation_db: float,
    transition_width: float,
)
```

Resampled record with the designed anti-alias filter and its spec.

The polyphase resampler filters at the intermediate rate
`fs_original·up` with a linear-phase Kaiser-window lowpass designed
from the two numbers below; the filter taps are returned so the spec
can be verified against the filter itself.

**Attributes**

| Name | Description |
| :--- | :--- |
| `signal` | The resampled record. |
| `fs` | Sample rate of `signal`, in Hz. |
| `original_fs` | Sample rate of the input, in Hz. |
| `up` | Interpolation factor of the rational ratio `up/down`. |
| `down` | Decimation factor of the rational ratio `up/down`. |
| `filter_taps` | Anti-alias FIR taps (unit passband gain; the polyphase engine applies the `up` interpolation gain), designed at the intermediate rate `original_fs·up`. A single `1.0` tap when the ratio is 1 (no filtering). |
| `passband_edge_hz` | Passband edge of the design, in Hz. |
| `stopband_edge_hz` | Stopband edge of the design (the smaller of the two Nyquist frequencies), in Hz. |
| `stopband_attenuation_db` | Designed stopband attenuation, in dB (also the passband ripple bound: the Kaiser window method holds the ripple of both bands within the same `δ = 10^(-A/20)`, though -- unlike a true equiripple design -- its ripple decays away from the band edges rather than staying at the bound). |
| `transition_width` | Transition-band width as a fraction of the smaller Nyquist frequency. |

### ResampledSignalResult.n_taps

*property*

Length of the designed anti-alias FIR.

## tone_burst

```python
tone_burst(
    fs: float,
    frequency: float,
    cycles: int,
    *,
    amplitude: float = 1.0,
    repetitions: int = 1,
    repetition_rate: float | None = None,
    pre_silence: float = 0.0,
    post_silence: float = 0.0,
) -> ToneBurstResult
```

Generate an IEC 60268-1 tone burst (single or repetitive).

IEC 60268-1:1985, Clause A2.1: "The burst should start at the
zero-crossing of the [...] tone and should consist of an integral
number of full periods." The burst is a sine of `cycles` full
periods gated by a rectangular envelope; with `repetition_rate` a
train of `repetitions` identical bursts is produced, one per
repetition period, as in the repetitive-burst test of Clause A2.2
(there: 5 ms bursts of 5 kHz tone at 2, 10 or 100 bursts per second).

The gate closes after `round(fs·cycles/frequency)` samples, so the
"integral number of full periods" is sample-exact only when `fs`
and `frequency` are commensurate (`fs·cycles/frequency` an
integer). Otherwise the gate closes up to half a sample away from
the tone's final zero crossing and the gated waveform carries a
residual step of up to `amplitude·sin(π·frequency/fs)` there
(e.g. 10 cycles of 997 Hz at 48 kHz span 481.44 samples, gated at
481); a [`PhonometryWarning`](/phonometry/reference/api/filters/phonometry/#phonometrywarning) quantifies the
realized residual.

**Parameters**

| Name | Description |
| :--- | :--- |
| `fs` | Sample rate, in Hz. |
| `frequency` | Tone frequency, in Hz (below the Nyquist rate). |
| `cycles` | Full tone periods per burst (positive integer). |
| `amplitude` | Peak amplitude of the tone. |
| `repetitions` | Number of bursts (requires `repetition_rate` when greater than 1). |
| `repetition_rate` | Bursts per second; each burst then occupies one full repetition period (burst plus silence). `None` (the default) produces a single burst with no trailing period. |
| `pre_silence` | Silence before the first burst, in seconds. |
| `post_silence` | Silence after the last burst (or after the last repetition period), in seconds. |

**Returns:** A [`ToneBurstResult`](/phonometry/reference/api/spectra/signals/#toneburstresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs or parameters are invalid. |

## ToneBurstResult

```python
ToneBurstResult(
    signal: NDArray[np.float64],
    envelope: NDArray[np.float64],
    fs: float,
    frequency: float,
    cycles: int,
    amplitude: float,
    burst_seconds: float,
    burst_samples: int,
    onset_sample: int,
    repetitions: int,
    repetition_rate: float | None,
    period_samples: int | None,
    duty_cycle: float | None,
)
```

Gated sine burst per IEC 60268-1:1985 (Annex A, Clause A2).

The tone starts at a zero crossing (positive-going) and the gate stays
open for an integral number of full periods, as Clause A2.1 requires
of the dynamic-response stimulus. With a repetition rate the record is
a train of identical bursts, one per repetition period (Clause A2.2).

**Attributes**

| Name | Description |
| :--- | :--- |
| `signal` | The burst record (silence, bursts and gaps included). |
| `envelope` | Rectangular gating envelope of `signal` (`amplitude` while the gate is open, `0` elsewhere). |
| `fs` | Sample rate, in Hz. |
| `frequency` | Tone frequency, in Hz. |
| `cycles` | Full tone periods per burst. |
| `amplitude` | Peak amplitude of the tone. |
| `burst_seconds` | Burst duration `cycles/frequency`, in seconds. |
| `burst_samples` | Samples per burst, `round(fs·cycles/frequency)`. |
| `onset_sample` | Index of the first sample of the first burst. |
| `repetitions` | Number of bursts in the record. |
| `repetition_rate` | Bursts per second, or `None` (single burst). |
| `period_samples` | Samples per repetition period (`round(fs/repetition_rate)`), or `None` (single burst). |
| `duty_cycle` | On fraction `burst_samples/period_samples`, or `None` (single burst). |

### ToneBurstResult.plot()

```python
ToneBurstResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the burst waveform with its gating envelope.

**Parameters**

| Name | Description |
| :--- | :--- |
| `language` | Label language, `"en"` (default) or `"es"`. |
