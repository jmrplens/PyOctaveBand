← [Documentation index](README.md)

# Test signals and sample-rate tools (IEC 60268-1)

A measurement is only as trustworthy as its stimulus and its sample-rate
bookkeeping. This page covers the signal toolbox of `phonometry.metrology`:
**tone bursts** with the exact gating IEC 60268-1 prescribes, the
**colored-noise generators** (detailed in the
[spectral analysis guide](spectral-analysis.md#5-colored-noise-generators)),
**resampling** whose anti-alias rejection is a stated, verifiable
specification rather than a library default, and **fractional delay** that
shifts a record by any sub-sample amount with band-limited exactness.

## 1. Tone bursts (IEC 60268-1)

The gated sine burst is the standard stimulus for dynamic behaviour:
sound-level-meter ballistics, quasi-peak meters, loudspeaker power handling.
IEC 60268-1:1985 (Clause A2.1) pins down what a well-formed burst is: it
"should start at the zero-crossing of the tone and should consist of an
integral number of full periods". `tone_burst` generates exactly that, as a
single burst or as the repetitive train of Clause A2.2 in which each burst
occupies one full repetition period:

```python
from phonometry import tone_burst

# One 5 ms burst of 5 kHz tone (25 full periods), as in Table AII.
single = tone_burst(48000, 5000, 25)
print(single.burst_samples)         # 240 samples = 5 ms at 48 kHz

# Clause A2.2: 5 ms bursts at 10 bursts per second.
train = tone_burst(48000, 5000, 25, repetitions=4, repetition_rate=10)
print(train.period_samples, train.duty_cycle)   # 4800, 0.05
train.plot()                        # waveform with the gating envelope
```

The result carries the record, the rectangular gating **envelope** and the
exact sample bookkeeping (`burst_samples`, `onset_sample`, `period_samples`,
`duty_cycle`), so a test report can state its stimulus numerically. Because
the gate spans an integral number of full periods starting at a zero
crossing, the burst energy has the closed form `A²N/2` exactly, which is how
the generator is verified.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/tone_burst_train_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/tone_burst_train.svg" alt="IEC 60268-1 tone bursts: a single 5 ms burst of 5 kHz tone starting at a zero crossing with its rectangular gating envelope, and a repetitive train of four bursts at 10 bursts per second with a 5 percent duty cycle" width="82%"></picture>

These are the bursts behind the
[Fast/Slow/Impulse ballistics](time-weighting.md) reference responses
(IEC 61672-1 Table 4 uses 4 kHz tonebursts of 200, 50 and 10 ms) and the
quasi-peak dynamic tests of IEC 60268-1 itself (5 kHz bursts of 1 to 200 ms,
Table AII).

## 2. Resampling with a stated anti-alias specification

Sample-rate conversion hides a filter, and that filter decides how much
aliased energy contaminates the result. `resample_signal` performs rational
polyphase resampling (44.1 to 48 kHz is the ratio 160/147) with a lowpass
FIR designed *inside the function* by the Kaiser window method, from two
numbers the caller controls:

- `stopband_attenuation_db` (default 120): the alias rejection, with the
  stopband starting exactly at the smaller of the two Nyquist frequencies,
  where folding happens;
- `transition_width` (default 0.05): the fraction of that Nyquist frequency
  given up to the filter's transition band, so the passband ends at
  `(1 - transition_width)·f_Nyq` and is flat within the same Kaiser ripple
  bound `δ = 10^(-A/20)`.

```python
from phonometry import noise_signal, resample_signal

x = noise_signal(44100, 5.0, color="pink", seed=1)
res = resample_signal(x, 44100, 48000)   # 120 dB alias rejection
print(res.up, res.down)                  # 160, 147
print(res.n_taps, res.passband_edge_hz)  # designed FIR, 20947.5 Hz
```

The designed taps travel with the result (`filter_taps`), so the
specification is *checkable*: the test suite measures the frequency response
of the returned filter and asserts the passband deviation and stopband
leakage against the design's own ripple bound (the design internally targets
1 dB past the request, so the delivered filter meets the stated numbers
rather than a Kaiser-formula approximation of them). A passband tone
resampled through the default specification matches the analytic tone at the
new rate within `10^(-120/20) = 10^-6`.

Several estimators resample internally at fixed rates (the ECMA-418-2
psychoacoustics at 48 kHz, STOI at 10 kHz); this function is the public,
documented counterpart for preparing records outside those chains.

## 3. Fractional delay

`fractional_delay` shifts a record by any number of samples, including
sub-sample amounts, by multiplying the spectrum with the phase ramp
`e^(-j2πf·D/fs)`: every component is delayed by exactly `D` samples. Two
boundary conventions cover the two use cases:

- `mode="linear"` (default) zero-pads the record past the shift, so samples
  leaving one end land in padding instead of wrapping around. Use it for
  transients and impulse responses; it is bit-identical to the alignment
  kernel inside [`align_impulse_responses`](correlation-delay.md). An
  integer delay reduces to an exact sample shift.
- `mode="circular"` applies the ramp over the record itself and wraps. For
  periodic records it is exact: a tone centered on a DFT bin delayed by
  `D` samples equals the analytically delayed tone to machine precision,
  and its phase changes by exactly `-2πf·D/fs` radians.

```python
import numpy as np
from phonometry import fractional_delay

y = fractional_delay(x, 0.37)                    # 0.37 samples later
z = fractional_delay(x, -2.5, mode="circular")   # advance, wrapped
```

One subtlety worth knowing: a real record of even length cannot carry a
fractionally delayed Nyquist-bin component (the inverse real FFT keeps only
its real part). Any properly sampled signal is band-limited below Nyquist,
so in practice the operation is exact; for synthetic corner cases, odd
lengths avoid the bin entirely.

## 4. Colored noise

The deterministic colored-noise generators (`noise_signal`: white, pink,
red, blue, violet with an exact power-law slope and bit-reproducible seeds)
complete the toolbox; they are documented with their spectral verification
in the [spectral analysis guide](spectral-analysis.md#5-colored-noise-generators).

## Where these tools are used

The [window figures of merit](spectral-analysis.md#6-choosing-the-window)
quantify the taper every spectral estimate in the library rests on; the
burst generator feeds ballistics and dynamic-response testing; and the
resampler and fractional delay are the sample-rate half of
[correlation and delay work](correlation-delay.md), where sub-sample
alignment is the difference between averaging impulse responses and
smearing them.

## References

- International Electrotechnical Commission. (1985). *Sound system
  equipment — Part 1: General* (IEC 60268-1:1985).
  [IEC webstore](https://webstore.iec.ch/en/publication/1204).
  Annex A, Clause A2: tone bursts starting at the zero crossing of the tone
  with an integral number of full periods (A2.1), repetitive burst trains
  at a stated repetition rate (A2.2), and the Table AII burst durations the
  sample counts are hand-checked against.
- Bendat, J. S., & Piersol, A. G. (2010). *Random Data: Analysis and
  Measurement Procedures* (4th ed.). Wiley. ISBN 978-0-470-24877-5.
  [doi:10.1002/9781118032428](https://doi.org/10.1002/9781118032428).
  Section 10.2 (data preparation: sampling, aliasing and the anti-alias
  filtering requirement the resampler states explicitly).
