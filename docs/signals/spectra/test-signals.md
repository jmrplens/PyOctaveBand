← [Documentation index](../../README.md)

# Test signals and sample-rate tools (IEC 60268-1)

A measurement is only as trustworthy as its stimulus and its sample-rate
bookkeeping. This page covers the signal toolbox of `phonometry.signals`:
**tone bursts** with the exact gating IEC 60268-1 prescribes, the
**colored-noise generators** (detailed in the
[spectral analysis guide](spectral-analysis.md#5-colored-noise-generators)),
**resampling** whose anti-alias rejection is a stated, verifiable
specification rather than a library default, and **fractional delay** that
shifts a record by any sub-sample amount with band-limited exactness.

Before the details, the family portrait: what each stimulus looks like in
time and where its energy sits over frequency.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_test_signals_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_test_signals.svg" alt="Grid of five labelled test-signal miniatures: white noise with a jagged waveform and a flat power spectral density of 0 dB per octave, equal power per hertz; pink noise with a smoother waveform and a minus 3 dB per octave density, equal power per octave; an MLS binary chip stream of period 2 to the m minus 1 samples with a flat line spectrum; a time-frequency sketch where a linear sweep is a straight line while the exponential sweep hugs low frequencies and spends equal time per octave; and an IEC 60268-1 tone burst of whole periods starting at a zero crossing, 25 periods of 5 kHz lasting 5 ms" width="92%"></picture>

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
crossing, the burst energy has the closed form $A^2 N/2$ exactly, which is how
the generator is verified.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/tone_burst_train_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/tone_burst_train.svg" alt="IEC 60268-1 tone bursts: a single 5 ms burst of 5 kHz tone starting at a zero crossing with its rectangular gating envelope, and a repetitive train of four bursts at 10 bursts per second with a 5 percent duty cycle" width="82%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import tone_burst

fs = 48000.0
single = tone_burst(fs, 5000, 25, pre_silence=0.001, post_silence=0.001)
train = tone_burst(fs, 5000, 25, repetitions=4, repetition_rate=10)

fig, axes = plt.subplots(2, 1, figsize=(10, 6.4))
t_ms = 1e3 * np.arange(single.signal.size) / single.fs
axes[0].plot(t_ms, single.signal, lw=0.9)
axes[0].plot(t_ms, single.envelope, "r--", label="Gating envelope")
axes[0].plot(t_ms, -single.envelope, "r--")
axes[0].set_xlabel("Time [ms]")

t_s = np.arange(train.signal.size) / train.fs
axes[1].plot(t_s, train.signal, lw=0.5)
axes[1].plot(t_s, train.envelope, "r--", label="Gating envelope")
axes[1].plot(t_s, -train.envelope, "r--")
axes[1].set_xlabel("Time [s]")

for ax in axes:
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right")
plt.tight_layout()
plt.show()
```

</details>

These are the bursts behind the
[Fast/Slow/Impulse ballistics](../levels/time-weighting.md) reference responses
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
  bound $\delta = 10^{-A/20}$.

```python
from phonometry import noise_signal, resample_signal

x = noise_signal(44100, 5.0, color="pink", seed=1)
res = resample_signal(x, 44100, 48000)   # 120 dB alias rejection
print(res.up, res.down)                  # 160, 147
print(res.n_taps, res.passband_edge_hz)  # designed FIR, 20947.5 Hz
res.plot()   # the delivered anti-alias filter against its design spec
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/resampling_antialias_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/resampling_antialias.svg" alt="Magnitude response of the anti-alias filter designed for a 44.1 to 48 kilohertz polyphase resampling on a logarithmic frequency axis: a flat passband up to the dashed passband edge just below 21 kilohertz, a steep transition to the dashed stopband edge at 22.05 kilohertz, and a stopband floor staying below the dotted minus 120 decibel design-attenuation line across the shaded rejected band" width="82%"></picture>

*The delivered anti-alias filter of the default 44.1 → 48 kHz conversion:
the stopband starts exactly at the smaller Nyquist frequency (where aliases
fold) and stays below the −120 dB design line; the passband ends 5 % below
it, flat within the same ripple bound.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from phonometry import noise_signal, resample_signal

x = noise_signal(44100, 5.0, color="pink", seed=1)
res = resample_signal(x, 44100, 48000)   # 120 dB alias rejection

# res is the ResampledSignalResult computed in the example above.
# One line — the delivered anti-alias filter against its design spec:
res.plot()
plt.show()

# By hand, from the taps the result carries — mirroring what
# ResampledSignalResult.plot() draws:
fs_up = res.original_fs * res.up
freqs, h = signal.freqz(res.filter_taps, worN=1 << 18, fs=fs_up)
mag_db = 20 * np.log10(np.maximum(np.abs(h), 1e-300))
view = (freqs > 0) & (freqs <= 4 * res.stopband_edge_hz)

fig, ax = plt.subplots()
ax.semilogx(freqs[view], mag_db[view], label="Anti-alias filter |H(f)|")
ax.axvline(res.passband_edge_hz, color="g", linestyle="--",
           label="Passband edge")
ax.axvline(res.stopband_edge_hz, color="r", linestyle="--",
           label="Stopband edge (alias fold)")
ax.axhline(-res.stopband_attenuation_db, color="k", linestyle=":",
           label="Design attenuation -120 dB")
ax.axvspan(res.stopband_edge_hz, 4 * res.stopband_edge_hz,
           color="r", alpha=0.08)
ax.set(xlabel="Frequency [Hz]", ylabel="Magnitude [dB]")
ax.legend()
plt.show()
```

</details>

The designed taps travel with the result (`filter_taps`), so the
specification is *checkable*: the test suite measures the frequency response
of the returned filter and asserts the passband deviation and stopband
leakage against the design's own ripple bound (the design internally targets
1 dB past the request, so the delivered filter meets the stated numbers
rather than a Kaiser-formula approximation of them). A passband tone
resampled through the default specification matches the analytic tone at the
new rate within $10^{-120/20} = 10^{-6}$.

Several estimators resample internally at fixed rates (the ECMA-418-2
psychoacoustics at 48 kHz, STOI at 10 kHz); this function is the public,
documented counterpart for preparing records outside those chains.

## 3. Fractional delay

`fractional_delay` shifts a record by any number of samples, including
sub-sample amounts, by multiplying the spectrum with the phase ramp
$e^{-j 2\pi f D/f_s}$: every component is delayed by exactly `D` samples. Two
boundary conventions cover the two use cases:

- `mode="linear"` (default) zero-pads the record past the shift, so samples
  leaving one end land in padding instead of wrapping around. Use it for
  transients and impulse responses; it is bit-identical to the alignment
  kernel inside [`align_impulse_responses`](correlation-delay.md). An
  integer delay reduces to an exact sample shift.
- `mode="circular"` applies the ramp over the record itself and wraps. For
  periodic records it is exact: a tone centered on a DFT bin delayed by
  `D` samples equals the analytically delayed tone to machine precision,
  and its phase changes by exactly $-2\pi f D/f_s$ radians.

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

## What this guide covers

**Covered.** IEC 60268-1:1985 Annex A, Clause A2: the tone burst that starts at
a zero crossing of the tone and consists of an integral number of full periods
(A2.1) and the repetitive burst train of A2.2, implemented by `tone_burst` and
hand-checked against the Table AII durations. Bendat & Piersol Section 10.2's
anti-alias requirement, turned into an explicit, checkable specification by
`resample_signal`'s stopband attenuation and transition width.

**Not covered.** IEC 60268-1:1985 is a general standard for sound system
equipment, and only its Annex A tone-burst clauses are implemented; amplifiers,
loudspeakers, connectors and the rest are untouched. `fractional_delay` and the
coloured-noise generators of `noise_signal` are general-purpose DSP tools with
no governing standard of their own, and their accuracy claims are closed-form,
not normative.

## See also

- [Time Weighting](../levels/time-weighting.md): the detector ballistics the tone
  bursts exercise (IEC 61672-1 Table 4).
- [Correlation and delay](correlation-delay.md): the alignment work built
  on the fractional-delay kernel.
- [Synchronous averaging](synchronous-averaging.md): period alignment with
  the same band-limited shift when $f_s T$ is not an integer.
- [Spectral analysis](spectral-analysis.md): the colored-noise verification
  and the window metrics.

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
